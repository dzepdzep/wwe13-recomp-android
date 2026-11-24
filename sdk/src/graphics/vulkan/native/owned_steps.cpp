// wwe2k14 patch (Session 80, OWN_RENDERER_PLAN.md "The shadows" step 2; gpu_native_own_steps). Owned steps: a pass of
// the 360's frame whose output is only ever read through its copy, drawn into an exact-size image of ours instead of
// the truth images. On the console such a draw goes into an EDRAM surface laid out in 80-pixel tiles (a 256x256
// blur into a 320-wide surface whose tiles another surface's pixels alias), then a resolve copies the rectangle into a
// texture in guest memory; NATIVE's truth frame emulates all of that exactly (the surface's tiles alias-loaded, its
// pass stored, the copy run as a compute batch into the texture). For a listed shader here, the draws - the game's own
// pipelines and bindings, its interpolation and our pixel shaders as before - are recorded into our image in a render
// pass of its own, the copy that follows is left out, and every draw that samples the copy's texture page is given our
// image's view (the texture's memory and its own image are never written while the page is ours).
//
// Owned passes (Session 80, the shadow casters): a pass of several listed draws - the clears of its truth pass first,
// of one colour over a box from the surface's origin, with depth and stencil when they wrote them whole - goes into
// our image (cleared as they cleared it, depth and stencil kept), resumed when something else ends its render pass.
// At its copy, the truth images' tiles of the box lack its draws (and, when the clears' pass is turned off, its
// clears): they are held as dirty, and our image's pixels are copied into them before anything reads them - a draw
// over them that is not a whole clear, an alias load or store of their image, a copy of them, a write-out - or dropped
// from the set when a clear writes them whole or another image takes them over. So the truth images are what the 360
// would have wherever they are looked at. A pass that started from a covering draw (no clears) leaves its tiles as
// they were, as gpu_shadow_half_rate leaves a skipped map's. Map of the hooks: docs\native\OWNED_STEPS_MAP.md (game
// repo).

#include <rex/graphics/vulkan/native/native_renderer.h>
#include "diag_layer.h"  // Session 95 (the diagnostic run)
#include "shader_swap.h"  // Session 94 (gpu_native_instance_world)
#include "owned_dead_clear.h"  // (Session 92, M2d)
#include "owned_shadow_skip.h"  // Session 95 (r7 contract 3: half-phase)
#include <rex/graphics/free_counters.h>

#include <algorithm>
#include <atomic>
#include <string_view>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <rex/cvar.h>
#include <rex/graphics/shared_memory.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>
#include <rex/ui/vulkan/gpu_fatal.h>  // Session 95 (PLAIN)

REXCVAR_DECLARE(bool, gpu_cp_trim);  // wwe2k14 patch (Session 87, step 4): defined in command_processor.cpp
REXCVAR_DECLARE(int32_t, gpu_tex_bind_memo);  // (Session 95, r6tex: pipeline\texture\cache.cpp)
REXCVAR_DECLARE(bool, gpu_tex_bind_memo_verify);
REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)
REXCVAR_DECLARE(int32_t, gpu_native);  // (Session 92, M2d: native_renderer.cpp)
REXCVAR_DECLARE(int32_t, gpu_native_truth_draw_height);  // (Session 95: native_truth.cpp - the launchers' 360P / 480P / 540P / 1080P rows)
REXCVAR_DECLARE(bool, gpu_edram_rename);
REXCVAR_DECLARE(bool, gpu_dlss);
REXCVAR_DECLARE(bool, gpu_dlaa);
REXCVAR_DECLARE(bool, gpu_native_skin_bind_lean);  // (Session 95, r3gpu: defined in skin_once.cpp)

REXCVAR_DEFINE_DOUBLE(gpu_native_owned_draw_ratio, 1.0, "GPU/Native",
                      "Session 94 (dynamic resolution under TRUE NATIVE; written per second by the game's dynamic_res.h, or set for a test): the owned "
                      "frame's scene pass (colour + depth, the chosen resolution's size) is drawn into images of this fraction (0.5..1) of its size - "
                      "viewport, scissor and depth-clear rectangles scaled - and its colour and depth copies are brought up to the full size by a "
                      "blit (linear for colour, nearest for depth), so no later reader changes. Read once per frame, at its first owned pass: never "
                      "two ratios in one frame. 1.0 = off (the passes as before). Session 95: under gpu_native_owned_only the chosen resolution row's "
                      "gpu_native_truth_draw_height (360P / 480P / 540P / 1080P) is the base and this cvar a fraction of it: the pass draws at "
                      "this x height / (720 x resolution scale), never below 0.5");

REXCVAR_DEFINE_INT32(gpu_native_owned_dres_depth_pass, 1, "GPU/Native",
                     "Session 95: how a scaled scene pass's depth goes up to the full size (gpu_native_owned_draw_ratio, the resolution rows, dynamic resolution). 1 = by a blit, "
                     "and by a fragment pass of ours where the device cannot blit the depth format (Adreno 830, AMD on Windows); 2 = always the fragment pass (to test it on a "
                     "device that can blit); 0 = never the pass: such a device draws the whole size")
    .range(0, 2);

namespace {
// Session 94 (gpu_native_owned_draw_ratio): file-level state (no new header members). The ratio the frame in progress uses, what was scaled / blitted
// since the last change, a histogram of the ratio per frame (10..100 %, steps of 5) for the ~600-frame line, and the scaled image sizes made (at most
// two stay resident: the older are retired when no pass holds them).
struct DresState {
  uint64_t frame = ~0ull;
  double ratio = 1.0;
  bool announced = false;
  uint32_t scaled = 0, blits = 0, refused = 0;
  uint64_t hist[21] = {};
  uint64_t frames_hist = 0, last_summary = 0;
  std::vector<std::pair<uint32_t, uint32_t>> sizes;  // newest last
  int depth_blit_ok = -1;  // BLIT_SRC | BLIT_DST on the depth format (queried once)
};
DresState g_dres;
}  // namespace
uint32_t g_dres_depth_blits = 0;  // (read by owned_scene.cpp's depth copy)

REXCVAR_DEFINE_STRING(gpu_native_own_steps, "", "GPU/Native",
                      "Session 80 (OWN_RENDERER_PLAN.md \"The shadows\"): guest pixel shaders (hex hashes) or vertex "
                      "shaders (v<hash>), comma-separated, whose truth draws are owned: a step (one draw into an RGBA8 "
                      "surface covering its rectangle from the surface's origin) or a pass of them after the clears of "
                      "its truth pass (with depth and stencil when the clears wrote them whole) is recorded into an "
                      "exact-size image of ours, the copy of that box into a texture that follows is left out, and the "
                      "draws sampling that texture are given our image. Not in readback windows. Empty = off");
REXCVAR_DEFINE_BOOL(gpu_native_own_steps_drop_clears, true, "GPU/Native",
                    "Session 80 (with gpu_native_own_steps): an owned pass that starts from the clears of its truth pass "
                    "turns their draws off at its copy (their whole pass when it loaded nothing) - our image holds what "
                    "they wrote, copied into the truth images before anything reads those tiles");
REXCVAR_DEFINE_BOOL(gpu_native_own_steps_debug_replay, false, "GPU/Native",
                    "Session 80 (diagnostic): every owned output's copy taken as not matching - the draws replayed into "
                    "the truth images and the copy run from them, the shaders still owned (the replay path's check: "
                    "same-frame shots equal to the owned steps off)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame, false, "GPU/Native",
                    "Session 82 (the owned match frame, docs\\native\\OWNED_FRAME_DESIGN.md): every truth draw of the "
                    "frame is a candidate for an owned pass (gpu_native_own_steps is ignored); a pass that is not owned is "
                    "a counted fallback with one named reason ([native][owned-frame] census). Read once a frame, at its "
                    "end: a change applies from the next frame (off again: dirty tiles copied in, owned pages kept until "
                    "the game copies into them)");
REXCVAR_DEFINE_INT32(gpu_native_owned_frame_log, 0, "GPU/Native",
                     "Session 82 (with gpu_native_owned_frame): the census logs one [native][owned-frame] line per pass "
                     "for the frames the segment log covers (gpu_native_segment_log N, set live as the routes do), or - "
                     "set live to N > 0 after the run started - for the next N frames. The "
                     "600-frame summary is always logged while the switch is on")
    .range(0, 1000);
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_readback_copy, true, "GPU/Native",
                    "Session 82 (P2, with gpu_native_owned_frame): in a readback window (the app's: a character file just "
                    "loaded, a match loading) passes are owned as outside it, and only a COPY the CPU gets - one the "
                    "recurring-resolve skip does not turn off, the attire builds - is not served from our image: its pass "
                    "is replayed on the truth path and the game's copy reads it back. Default on since the committed defined-start replay (docs/native/OWNED_ENTRANCES_SCOPE.md section 7). Off = the old rule, every pass in "
                    "the window falls back (reason readback)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_cube_derive, true, "GPU/Native",
                    "Session 82 (P2, with gpu_native_owned_frame): the face copies of a 6-face strip pass into a cube the texture cache has "
                    "not seen yet (nothing sampled it) make our own cube from the first build - its base page and layer derived from "
                    "the copy; the first load checks it (dropped if it is another cube). Off = such a copy is the game's, replayed on "
                    "the truth path once a frame until the cube is sampled");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_cube_derive_scaled, true, "GPU/Native",
                    "Session 82 (P2): gpu_native_owned_frame_cube_derive also at a resolution scale above 1 (kill switch: off = replayed there)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_restore_skip, true, "GPU/Native",
                    "Session 82 (P2, with gpu_native_owned_frame): a restore draw - the 360 writing a scene's colour and depth back "
                    "into the EDRAM from the two copies it took out - is left out when an owned pass holds that surface "
                    "and the draw covers its whole box: our image never lost the pixels (a 360 quirk, hard rule 6). Off = "
                    "restore draws are the truth path's (reason other:restore)");
REXCVAR_DEFINE_INT32(gpu_native_owned_frame_readback_min_w, 0, "GPU/Native",
                     "Session 82 (P2, diagnostic): a read-back copy narrower than this many texels is served like any other "
                     "(bisecting which copies matter)")
    .range(0, 4096);
REXCVAR_DEFINE_BOOL(gpu_native_owned_wb_commit_uncovered, false, "GPU/Native", "Session 90 (Q, diagnosis): the uncovered write-back branch commits the pass (the behaviour before review #4)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_wb_drop_pages, false, "GPU/Native",
                    "Session 90 (Q): the owned page(s) of a copy written back for the CPU are dropped after the write (the texture loads from the exact memory); feedback pages stay");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_readback_materialise, false, "GPU/Native",
                    "Session 90 (Q): a read-back copy no writer exists for (the old replay path) first tries the Session 87 chains materialise (the pass's "
                    "pixels as dirty tiles of the truth image, the game's own copy from there). Off (default): the pass is replayed. For one A/B only: "
                    "every copy a writer exists for is written from our image whatever this says");
REXCVAR_DEFINE_INT32(gpu_native_owned_frame_budget_mb, 512, "GPU/Native",
                     "Session 82 (with gpu_native_owned_frame): the memory our images may take (colour, depth, page "
                     "images; an estimate) - the pools grow on demand up to it, then a pass that needs one more falls "
                     "back (reason pool)")
    .range(16, 8192);
REXCVAR_DEFINE_STRING(gpu_native_owned_frame_exclude, "", "GPU/Native",
                      "Session 82 (diagnostic): pixel shaders (hex hashes, comma-separated) whose draws the owned frame refuses - "
                      "they stay on the truth path (bisecting a picture difference)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_rect_pages, true, "GPU/Native",
                    "Session 82: the rectangles a 16-bit pass is copied in, at the same place, into a texture no page of ours holds "
                    "(the blur inputs of the wrestler maps) make that texture a 16-bit page image of ours, filled from the pass image: "
                    "no truth copy-in and no game copy for them");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_hold_msaa_clears, false, "GPU/Native",
                    "Session 82 (default off: Bryan's entrance differs with it on, cause unconfirmed): the fast clears the game issues in a multisampled surface mode (the clear quad of the 560-wide "
                    "passes: pitch field 280, sx 2) are held like the others - not opened as truth passes (a begin, two clear "
                    "draws and an end recorded and then turned into no-ops by the owned pass that takes them as its start)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_hold_flush_at_textures, true, "GPU/Native",
                    "s82 hold investigation: held clears go into the truth path before the next draw's texture request "
                    "whenever a deferred (carried) copy is pending - the request's serve / rename / sampled-flush decisions then "
                    "see the clear where the game issued it (else the clear is recorded inside the next draw, after them)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_alpha_learn, false, "GPU/Native",
                    "Session 82: the scene's alpha-blended (0x07060706) draws of swapped shaders whose alpha is 1 in 8-bit terms "
                    "everywhere are learned with an occlusion probe (two clean probes) and drawn with the blend off; probed again "
                    "every gpu_native_owned_alpha_learn_interval frames, a translucent fragment sends the draw back to blended");
REXCVAR_DEFINE_BOOL(gpu_native_owned_alpha_learn_constants, false, "GPU/Native",
                    "Session 82: the pixel shader's float constants are part of an alpha-learn key, so a fade (a constant) starts a key "
                    "unproven at once. Off by default: the match's pixel constants change from frame to frame and the keys then never "
                    "settle (8k-30k keys, ~40% of the candidate draws unblended instead of ~89%)");
REXCVAR_DEFINE_INT32(gpu_native_owned_alpha_learn_interval, 60, "GPU/Native",
                     "Session 82: frames between the probes of a proven draw (1 = every frame)")
    .range(1, 120);
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_chains, false, "GPU/Native",
                    "Session 87 (entrance passes): the shadow / glare chains as surfaces of our own. A clear that no draw takes starts its surface's pass in our images (colour and depth together) instead of running in the truth path; a pass another surface's draw or copy meets is parked, never replayed, and no truth step pauses an owned surface; draws and copies are matched to a pass by its surface (base, pitch, class), not by shared EDRAM tiles; a depth-only clear is recorded into the pass that holds the depth, or pended (run only if something reads that depth, else dropped); the 16-bit class-2 defined start is on. Default off: the owned frame is exactly as before.");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_dead_relearn, false, "GPU/Native",
                    "Session 87 (step 3): a pass key is dead when its pages were read by no sampler for the last 240 copies "
                    "(each revival doubles that, up to 16x) - not 'never read, ever outside a readback window'. One read in "
                    "the entrances (the crowd's 20 impostor poses are sampled every frame there) kept the match's 64x64 renders "
                    "alive for good (dead passes not drawn 0.0). A sampler that asks for a dropped page still brings the pass "
                    "back, as before");
REXCVAR_DEFINE_BOOL(gpu_native_owned_dead_sticky, false, "GPU/Native",
                    "Session 95 (Lane E): a pass key that a sampler has read once (in a readback window or not, or a dropped page that "
                    "was asked for) is never learned dead again. For class phone (the quirk sets it): a pass dropped while its page is "
                    "read later shows a stale texture for a frame (Purple Savage, the crowd signs). Off: the dead-pass learning is "
                    "exactly as before, which the Surface keeps for the dead-pass saving");
REXCVAR_DEFINE_BOOL(gpu_native_owned_dead_held, true, "GPU/Native",
                    "Session 95 (phone black, r083): frames held while pipelines are made have no swap, so nothing reads the front page: they do not "
                    "count as unread copies, and a swap of a dropped pass's page revives the pass even with no page of ours left (owned_post.cpp). "
                    "Off: as before, a hold of 240 frames kills the frame's last pass and the screen stays black");
REXCVAR_DEFINE_INT32(gpu_native_owned_dont_care_clear, 0, "GPU/Native",
                     "Session 95 (Lane E): the owned passes' DONT_CARE loads (the colour of the cover pass, the depth of every variant "
                     "that loads none) become CLEAR. 0 = DONT_CARE as before; 1 = cleared to 0 (colour) and 1.0 (depth) - for class "
                     "phone, where a tile GPU leaves undefined memory and a read-before-write shows as garbage; 2 = cleared to magenta "
                     "and depth 0.0, a PC switch: any magenta on screen is a pass that read what it never wrote. Takes effect at start")
    .range(0, 2);
// (Session 95, Lane E, gpu_native_owned_dont_care_clear: the clear values for a begin whose pass has CLEAR where it had DONT_CARE.
// Slots the variant clears for itself (keep0 / keep1) keep their values.)
// Review fix: the value is latched when the passes are made (a live change would give CLEAR attachments no values), 0 until then.
static std::atomic<int> g_owned_dc_mode{-1};
static int OwnedDcMode() {
  int m = g_owned_dc_mode.load(std::memory_order_relaxed);
  return m < 0 ? 0 : m;
}
static void OwnedDontCareValues(VkClearValue out[2], const VkClearValue* keep0, const VkClearValue* keep1) {
  const bool magenta = OwnedDcMode() == 2;
  out[0] = VkClearValue();
  out[1] = VkClearValue();
  if (magenta) {
    out[0].color.float32[0] = 1.0f;
    out[0].color.float32[2] = 1.0f;
    out[0].color.float32[3] = 1.0f;
    out[1].depthStencil = {0.0f, 0u};
  } else {
    out[1].depthStencil = {1.0f, 0u};
  }
  if (keep0) out[0] = *keep0;
  if (keep1) out[1] = *keep1;
}

REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_strip_batch, false, "GPU/Native",
                    "Session 87 (step 3): the six face copies of a cube strip pass (Bryan's five light-cube strips: 30 copies "
                    "with 60 barriers a frame) are one copy of six faces at the first face; the other five are answered from "
                    "it. Resolution scale 1 only, and only while the pass has not drawn again");
REXCVAR_DEFINE_BOOL(gpu_tiler_poison, false, "GPU/Native",
                    "Session 95 (TILER, debug; owned frame): what a tile-based GPU would not keep is filled with poison instead of left alone, so a pass that "
                    "relied on kept contents shows on the PC. Every owned pass attachment whose load op is DONT_CARE (the cover pass's colour with "
                    "gpu_owned_cover_dont_care; the depth / stencil of the cover, clear-colour and resume-without-depth passes) is CLEARED to magenta "
                    "(1,0,1,1) colour / depth 0.0 stencil 0xA5 instead. Latched at the first read (when the owned render passes are made): set it at launch. Does not cover depth stored "
                    "DONT_CARE (dead-depth: the existing LEARNED WRONG line) or the post / truth passes. Logs '[native][owned-tiler] poison'");
REXCVAR_DEFINE_BOOL(gpu_tiler_sync_control, false, "GPU/Native",
                    "Session 95 (TILER, debug; owned frame): the POSITIVE CONTROL of owned_check -Sync. Every owned pass start recorded by OwnedRecord (not OwnedBeginClearedPass / OwnedResumeNow, which keep their full barriers) waits for nothing (srcStage TOP_OF_PIPE, "
                    "srcAccess 0) - a deliberate write-after-write hazard against the earlier pass, so the synchronization validation must report it. Never for play");
REXCVAR_DEFINE_BOOL(gpu_owned_tiler_passes, false, "GPU/Native",
                    "Session 93 (tile-based GPUs: Mali, Adreno, PowerVR; owned frame only): (1) a resumed pass without depth of its own does not "
                    "load and store the shared scratch depth; (2) a pass whose depth nothing has read after it for 8 passes in a row stores no "
                    "depth / stencil (a later read clears that for good and is counted as a learned wrong); (3) the scratch depth images are "
                    "transient attachments in lazily allocated memory where the device has such a memory type; (4) owned colour and 16-bit page "
                    "images declare their view formats (VkImageFormatListCreateInfo) so drivers can keep framebuffer compression. Read when an "
                    "image or a pass is made; off = the older behaviour exactly (the would-be counts are still logged)");
REXCVAR_DEFINE_BOOL(gpu_owned_image_format_list, false, "GPU/Native",
                    "Session 94 (tile-based GPUs, ARM Mali framebuffer compression): owned colour and 16-bit page images chain a VkImageFormatListCreateInfo naming "
                    "exactly the formats they are viewed as (RGBA8: UNORM + SNORM; page images: their own + the UINT storage view + the signed twin), and the "
                    "16-bit class colour image, only ever viewed as itself, is made without MUTABLE_FORMAT. The page images keep STORAGE (compute fills write them "
                    "through the UINT view); the owned colour images never had it. Images are made once, at first use. Off = the older behaviour (gpu_owned_tiler_passes "
                    "still lists formats, as in Session 93)");
REXCVAR_DEFINE_BOOL(gpu_owned_cover_dont_care, true, "GPU/Native",
                    "Session 95 (PLAIN): the owned 'cover' pass (every pixel is drawn) does not load the colour (LOAD_OP_DONT_CARE) - what the "
                    "tile-based paths want. false = it loads the colour (LOAD_OP_LOAD; legal Vulkan, a little slower on tilers; the first barrier still comes from UNDEFINED, so the loaded pixels are undefined too: this drops the DONT_CARE hint only) "
                    "The plain mode of an unknown GPU (gpu_plain_mode) sets false. Read when the render passes are made; "
                    "load / store ops do not change render pass compatibility");
REXCVAR_DEFINE_BOOL(gpu_owned_transient_attachments, false, "GPU/Native",
                    "Session 94 (tile-based GPUs): the shared scratch depth (the depth attachment of every owned pass without depth of its own - proven never "
                    "loaded, sampled, copied or resolved) is a TRANSIENT_ATTACHMENT image in lazily allocated memory where the device has such a type, and its passes "
                    "load and store nothing (the ResumeNoDepth variant). Image depths that are sampled / copied / written back stay ordinary. Needs a fresh start "
                    "(images are made once); gpu_owned_tiler_passes makes the same scratch depths transient too");
REXCVAR_DEFINE_BOOL(gpu_owned_tiler_scene_reset, false, "GPU/Native",
                    "Session 94 (with gpu_owned_tiler_passes): what the dead-depth learning knows is reset at a scene change - a burst of new pass keys (16 within "
                    "120 frames, a loading screen / a new match), a key silent for 600 frames, and after any learned wrong - and a key is dead only when its "
                    "depth stayed unread for 8 passes AND 120 frames of the current scene, with no dead depth applied for 240 frames after a reset. Off = Session 93 exactly");
REXCVAR_DEFINE_BOOL(gpu_native_owned_pass_barrier_lean, false, "GPU/Native",
                    "Session 82: a pass that keeps its own depth and starts in an image whose last recorded use the GPU has "
                    "finished begins with a layout transition that waits for nothing, not the all-stage drain every pass "
                    "start had; every other pass keeps the full barrier");
REXCVAR_DEFINE_BOOL(gpu_native_crowd_pose_half, false, "GPU/Native",
                    "Session 95 (resolves, off by default; LOW sets it): the crowd pose renders (the 64x64 'G2' passes: ps ED94EC2F in matches / 803205E3 in entrances, 20 a "
                    "frame, 1 draw and 1 copy each) are drawn every other frame - on the even frames each pass is dropped whole (its draw, held clears and copy; its page stays mapped) "
                    "and the cards sample last frame's pose pages. Needs gpu_native_owned_frame_drop_dead");

namespace {
// Session 95 (resolves): a pose render key (ps of the two crowd pose shaders, a surface one tile wide: the 64x64 renders).
bool OwnedPoseKey(uint64_t ps, uint32_t pitch) {
  return pitch == 1 && (ps == 0xED94EC2F6D52E14Full || ps == 0x803205E3BEA932C8ull);
}
uint64_t g_pose_dropped_frame = UINT64_MAX, g_pose_frames = 0, g_pose_logged_to = 0;
// (the frame's one decision for every pose pass, and the last frame they were drawn)
uint64_t g_pose_dec_frame = UINT64_MAX, g_pose_drawn_frame = UINT64_MAX - 8;
bool g_pose_dec_drop = false;
}  // namespace

REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_drop_dead, true, "GPU/Native",
                    "Session 82: a pass whose pages no sampler has read in its first 240 copies is not drawn any more (its draw, the "
                    "clear held for it and its copy are dropped); the first sampler of one of its pages brings it back");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_park_keep, true, "GPU/Native",
                    "Session 82 (experiment): a truth draw of ANOTHER surface over a parked pass's tiles does not drop it (the pass is "
                    "taken up again by a later draw of its surface, as if the other surfaces' tiles were not shared)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_pagelog, false, "GPU/Native",
                    "Session 82 (diagnostic): a [native][pagelog] line when an owned page is made and when a sampler reads it (the "
                    "pages nothing ever reads are the passes to drop).");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_dedupe, true, "GPU/Native",
                    "Session 82 (CPU): an owned draw records only the pipeline, sets and dynamic state that differ from what its "
                    "render pass has bound (the truth path's dedupe). Off: every command per draw, for an A/B.");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_hold_clears, true, "GPU/Native",
                    "Session 82 (with gpu_native_owned_frame): colour clears are held until an owned pass takes them (never recorded in the "
                    "truth path); false = recorded at once as before (diagnostic)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_detach_pages, false, "GPU/Native",
                    "Session 87 (default off): a committed owned pass whose pages are its own image (a whole colour copy served "
                    "directly, a direct depth page) and that goes on drawing is not ended: the pages are given a snapshot of the "
                    "image / a converted depth page first (what they hold is exactly what the game's copies took), and the pass "
                    "continues. Off: the pass ends and its next draws are the truth path's ('continued page' in the census)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_continue_copy, false, "GPU/Native",
                    "Session 87 (default off; replaces gpu_native_owned_frame_detach_pages when on): a committed owned pass that goes on "
                    "drawing while pages of ours ARE its image goes on in a fresh image of ours holding a copy of its colour and depth / "
                    "stencil; the pages keep the old image untouched, so every view handed out for it (this draw's, kept draws', earlier "
                    "readers') stays valid and a draw that samples its own page reads the old image and writes the new one");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_restore_draw, false, "GPU/Native",
                    "Session 87 (default off): a restore draw (the 360's 'put the saved scene back') that cannot be left out as "
                    "already holding the pass's pixels is recorded as an ordinary draw of the owned pass (it samples the pages of the "
                    "two copies the game took, ours) - the pass it continues was kept by gpu_native_owned_frame_restore_skip's "
                    "restorable list; clears held for the restore's surface are dropped (the restore overwrites colour and depth)");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_carry_passes, false, "GPU/Native",
                    "Session 87 (default off): a committed owned pass (and the committed passes parked) is not ended by a submission's "
                    "end - its render pass ends, its image stays, a draw of the next submission resumes it (it ends at the frame's end). "
                    "Off: the pass ends and the rest of its draws are the truth path's");
REXCVAR_DEFINE_BOOL(gpu_native_owned_frame_tiles_memo, false, "GPU/Native",
                    "Session 87 (default off, CPU): EdramTiles keeps its last results (it is a pure function of its arguments), so "
                    "the per-draw tile sets of the truth path's checks and the owned pass's dirty tracking are not rebuilt tile by tile");
REXCVAR_DEFINE_BOOL(gpu_native_own_steps_debug_materialize, false, "GPU/Native",
                    "Session 80 (diagnostic): an owned pass's dirty tiles copied into the truth images right at its copy "
                    "(the copy-back path's check: same-frame shots equal to the owned steps off)");

REXCVAR_DEFINE_BOOL(gpu_native_owned_only, false, "GPU/Native",
                    "Session 92 (M2a, docs\\native\\d3d\\S92_M2_DESIGN.md section 4): the owned-only latch. With the owned frame and its chains on (TRUE NATIVE), "
                    "nothing is handed back to the 360 truth path: a draw the owned frame does not take is dropped (counted, named), a refused claim is "
                    "claimed and dropped, no replay, no pause, no dirty tile, no truth image at a commit; a copy is served from the pass, from the "
                    "surface's newest record (a pass that ended), written from our image, or dropped (counted by source line). Set by src\\d3d_owned.h "
                    "with gpu_native_truth_prewarm=false. A hard owned failure releases it for the session. Default off: nothing changes");

namespace rex::graphics::vulkan {
extern uint64_t g_draw_rec_generation;  // (Session 95, r3cmd: draw_record.h)
}
namespace rex::graphics::vulkan::native {
// Session 95 (r3gpu): the pass log's shader census: per owned render pass, draws and index count per vertex / pixel shader pair ([native][passdraws]; only while gpu_native_truth_pass_log runs).
namespace {
struct PassCensusRow { uint64_t vs, ps; uint32_t draws, indices, attr; uint64_t area; };
std::vector<PassCensusRow> g_pass_census;
void OwnedPassCensusNote(uint64_t vs, uint64_t ps, uint32_t count, bool attr, const int32_t* b) {
  for (PassCensusRow& r : g_pass_census) {
    if (r.vs == vs && r.ps == ps) {
      ++r.draws; r.indices += count; r.attr += attr ? 1 : 0;
      r.area += uint64_t(std::max(0, b[2] - b[0])) * uint64_t(std::max(0, b[3] - b[1]));
      return;
    }
  }
  g_pass_census.push_back({vs, ps, 1, count, attr ? 1u : 0u, uint64_t(std::max(0, b[2] - b[0])) * uint64_t(std::max(0, b[3] - b[1]))});
}
void OwnedPassCensusFlush(uint64_t frame, uint32_t w, uint32_t h) {
  for (const PassCensusRow& r : g_pass_census) {
    REXGPU_INFO("[native][passdraws] frame {} {}x{} vs {:016X} ps {:016X}: draws {} indices {} attr {} box_px {}", frame, w, h, r.vs, r.ps, r.draws, r.indices, r.attr, r.area);
  }
  g_pass_census.clear();
}
}  // namespace
double OwnedRowRatio(uint32_t scale_x, uint32_t scale_y);  // (Session 95: below; native_truth.cpp logs it)

namespace {
uint64_t g_dead_held_copies = 0;  // (Session 95, phone black: copies of held frames, left out of the dead-pass learning; [owned-dead] summary)
// Session 92 (M2a): the latch's counters, file level (no new header members for them); printed by OwnedFrameSummary as
// "[native][owned-only]" and reset with the window. Strings group the drops by signature.
struct M2aStats {
  uint32_t draws_dropped = 0, draws_skipped_pipeline = 0, draws_skipped_empty = 0, claims_dropped = 0;
  uint32_t draw_reasons[16] = {};
  uint32_t copies_written_after_loss = 0, drops_kept_page = 0, readback_dropped = 0, front_dropped = 0, front_written = 0;
  uint32_t copies_dropped = 0, from_record = 0, replays_avoided = 0, fallbacks_ignored = 0, writeouts_ignored = 0, pauses_avoided = 0;
  uint32_t clearing_resolves = 0, held_dropped = 0, dclear_dropped = 0, records_noted = 0, records_evicted = 0, records_superseded = 0;
  uint32_t loaded_from_record = 0, loaded_no_record = 0, depth_from_record = 0, lost_clears = 0, latch_released = 0;
  uint32_t copies_filled = 0, readback_filled = 0, dead_owned = 0;  // (Session 92, M2c: the defined fill)
  uint32_t composites = 0, composite_readback = 0, composite_lower_filled = 0;  // (Session 92, M2d, Part 4: copies larger than their pass's box, written from a composite image; the lower part filled)
  uint32_t depth_from_dead_clear = 0;  // (Session 92, M2d: colour-only passes given depth from a dead depth clear's value)
  uint32_t latch_kept_no_edram = 0;  // (Session 92, M2d: a hard failure that would have released the latch, kept owned-only - no EDRAM buffer for the truth path)
  std::vector<std::pair<std::string, uint32_t>> draw_groups, copy_groups, avoided_why, fill_groups;
};
std::string g_drop_note;  // (M2c: why the copy being dropped was lost - set where the shape check fails, read by from_truth)
M2aStats g_m2a;
bool g_m2d_edram_less = false;  // (Session 92, M2d: TRUE NATIVE + the latch + no EDRAM buffer (the render target cache made none) - decided at the frame boundary, OwnedRefreshFrame)
void M2aGroup(std::vector<std::pair<std::string, uint32_t>>& groups, const std::string& key, size_t cap = 96) {
  for (auto& g : groups) {
    if (g.first == key) {
      ++g.second;
      return;
    }
  }
  if (groups.size() < cap) groups.emplace_back(key, 1u);
}
}  // namespace

// Session 92: the lighting-cube derivation counters (owned_feedback.cpp)
void OwnedCubeCounters(uint32_t& stale, uint32_t& older, uint32_t& contradicted, uint32_t& reused, bool reset);
void OwnedEntryCounters(uint64_t& claims, uint64_t& lean, uint64_t& repeat, uint32_t& surfaces_new, uint32_t& would_alias, uint32_t& table_full, bool reset);  // (Session 92, M2b: owned_entry.cpp)

// Session 82 (P2, diagnostic): every state change of the 1280x720 scene pass (base 0 pitch 16), in the census window.
#define OWNED_SCENE_LOG(pass_, ...)                                                              do {                                                                                             if (owned_frame_logging_ && (pass_).base == 0 && (pass_).pitch_tiles == 16) REXGPU_INFO(__VA_ARGS__);   } while (0)

namespace {
// The truth images' formats by class (kTruthColorFormats: 1 RGBA8 for k_8_8_8_8, 2 RGBA16_SNORM - the packed 16-bit
// values - for k_16_16 / k_16_16_16_16).
VkFormat OwnedFormat(uint32_t kind) {
  return kind == 2 ? VK_FORMAT_R16G16B16A16_SNORM : VK_FORMAT_R8G8B8A8_UNORM;
}
constexpr uint32_t kOwnedImagesPerSize = 4;
bool Listed(const std::vector<uint64_t>& list, uint64_t hash) {
  return std::find(list.begin(), list.end(), hash) != list.end();
}
// A face's stencil ops (12 bits) writing every bit it covers: REPLACE where it passes, ALWAYS passing.
bool StencilWhole(uint32_t ops) {
  return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS);
}
constexpr VkPipelineStageFlags kOwnedShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
constexpr VkPipelineStageFlags kOwnedAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
// (Session 82, sync validation: a transition from UNDEFINED is itself a write; the image's earlier writes - the previous pass's
// attachment store, a snapshot or page fill - must be in its first access scope, or it is a write-after-write hazard.)
constexpr VkAccessFlags kOwnedAnyWrite = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;

VkImageMemoryBarrier OwnedBarrier(VkImage image, VkImageAspectFlags aspects, VkAccessFlags src, VkAccessFlags dst,
                                  VkImageLayout from, VkImageLayout to) {
  VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.srcAccessMask = src;
  b.dstAccessMask = dst;
  b.oldLayout = from;
  b.newLayout = to;
  b.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects);
  return b;
}
}  // namespace

bool NativeRenderer::OwnedDisabled(const std::vector<std::pair<uint64_t, uint64_t>>& disabled, uint64_t hash) const {
  return std::any_of(disabled.begin(), disabled.end(),
                     [&](const std::pair<uint64_t, uint64_t>& d) { return d.first == hash && frame_ < d.second; });
}

bool NativeRenderer::OwnedWants(uint64_t pixel_shader_hash) const {
  // (The lists as copied at the last frame's end - OwnedRefreshList; a shader that lost an output is out for 600
  // frames.) Session 82: with the owned frame on every draw is a candidate (pauses are per pass, OwnedPaused).
  if (owned_frame_on_) return true;
  return !owned_list_.empty() && Listed(owned_list_, pixel_shader_hash) &&
         !OwnedDisabled(owned_disabled_, pixel_shader_hash);
}

bool NativeRenderer::OwnedWants(uint64_t pixel_shader_hash, uint64_t vertex_shader_hash) const {
  if (owned_frame_on_) return true;
  return OwnedWants(pixel_shader_hash) || (!owned_vs_list_.empty() && Listed(owned_vs_list_, vertex_shader_hash) &&
                                           !OwnedDisabled(owned_disabled_vs_, vertex_shader_hash));
}

bool NativeRenderer::OwnedImageHeld(int16_t image) const {
  // (The pass waiting for its copy; a page a sampler may still read later in the frame; dirty tiles still to come
  // from it.)
  // (Session 94: the full-size image a scaled scene pass blits its copies into is held as long as a pass or record has it.)
  if (owned_pending_.dres_full == image || std::any_of(owned_parked_.begin(), owned_parked_.end(), [&](const OwnedPass& q) { return q.dres_full == image; }) ||
      std::any_of(owned_restorable_.begin(), owned_restorable_.end(), [&](const OwnedPass& q) { return q.dres_full == image; }) ||
      std::any_of(owned_newest_.begin(), owned_newest_.end(), [&](const OwnedPass& q) { return q.dres_full == image; })) {
    return true;
  }
  return (owned_pending_.valid && owned_pending_.image == image) ||
         std::any_of(owned_parked_.begin(), owned_parked_.end(), [&](const OwnedPass& q) { return q.image == image; }) ||
         // (Session 82, P2, review: a pass the front buffer's composite dropped, kept for its restore - its image is not free.)
         std::any_of(owned_restorable_.begin(), owned_restorable_.end(), [&](const OwnedPass& q) { return q.image == image; }) ||
         std::any_of(owned_pages_.begin(), owned_pages_.end(), [&](const OwnedPage& p) { return p.image == image; }) ||
         std::any_of(owned_dirty_.begin(), owned_dirty_.end(), [&](const OwnedDirty& d) { return d.image == image; }) ||
         // (Session 92, M2a: a surface's newest record - the image a later copy or start is served from - is held until a later pass replaces it
         // or the budget trims it (OwnedEvictFor).)
         std::any_of(owned_newest_.begin(), owned_newest_.end(), [&](const OwnedPass& q) { return q.image == image; });
}

void NativeRenderer::OwnedRetireImage(size_t index) {
  OwnedDepthAccount(int16_t(index));  // (Session 93: the image's last pass with depth is over)
  OwnedImage& o = owned_images_[index];
  const uint64_t submission = command_processor_.GetCurrentSubmission();
  Retired colour;
  colour.submission = submission;
  colour.images[0] = o.image;
  colour.memory[0] = o.memory;
  colour.views[0] = o.view;
  colour.views[1] = o.array_view;  // (Session 94: gpu_d3d_decouple)
  colour.framebuffer = o.framebuffer;
  retired_.push_back(colour);
  if (o.framebuffer_depth_up != VK_NULL_HANDLE) {  // (Session 95)
    Retired up;
    up.submission = submission;
    up.framebuffer = o.framebuffer_depth_up;
    retired_.push_back(up);
  }
  if (o.depth != VK_NULL_HANDLE || o.framebuffer_depth != VK_NULL_HANDLE) {
    Retired depth;
    depth.submission = submission;
    depth.images[0] = o.depth;
    depth.memory[0] = o.depth_memory;
    depth.views[0] = o.depth_view;
    depth.framebuffer = o.framebuffer_depth;
    retired_.push_back(depth);
  }
  if (o.depth_sample_view != VK_NULL_HANDLE || o.depth_stencil_view != VK_NULL_HANDLE) {  // (review: the depth copy's views were leaked)
    Retired dv;
    dv.submission = submission;
    dv.views[0] = o.depth_sample_view;
    dv.views[1] = o.depth_stencil_view;
    retired_.push_back(dv);
  }
  for (auto it = owned_views_.begin(); it != owned_views_.end();) {  // (the views samplers were given of it)
    if (it->image == o.image || (o.depth != VK_NULL_HANDLE && it->image == o.depth)) {
      Retired view;
      view.submission = submission;
      view.views[0] = it->view;
      retired_.push_back(view);
      it = owned_views_.erase(it);
      ++owned_view_generation_;  // (Session 95, r6tex)
    } else {
      ++it;
    }
  }
  owned_bytes_ -= std::min(owned_bytes_, o.bytes);
  OwnedSpareFillSet(o.fill_set);  // (its fill source sets are kept for the next image that needs one, not leaked with the slot)
  OwnedSpareFillSet(o.depth_fill_set);
  if (o.wb_set != VK_NULL_HANDLE) owned_wb_spare_sets_.push_back({o.wb_set, command_processor_.GetCurrentSubmission()});  // (Session 90, Q)
  if (owned_strip_batch_.valid && owned_strip_batch_.image == int16_t(index)) owned_strip_batch_.valid = false;  // (review: an evicted slot may be reused)
  ++owned_draw_serial_;
  o = OwnedImage();  // (the slot stays, empty)
}

void NativeRenderer::OwnedEvictFor(uint64_t more, int16_t keep) {
  // The least recently used image that nothing holds (OwnedImageHeld: the pending pass, every parked and restorable pass, any image a page of ours stands
  // on, any image with dirty tiles to copy in) and that is not `keep`; retired the way every retirement is - its objects are freed only after the
  // submission that may still use them has completed.
  while (OwnedOverBudget(more)) {
    int best = -1;
    for (size_t i = 0; i < owned_images_.size(); ++i) {
      const OwnedImage& o = owned_images_[i];
      if (!o.width || int16_t(i) == keep || OwnedImageHeld(int16_t(i))) continue;
      if (best < 0 || o.used < owned_images_[best].used) best = int(i);
    }
    if (best < 0) {
      // (Session 92, M2a: a surface's newest record is the last to go - the oldest one's hold is let go, then the image is a free one. A record
      // the keep image belongs to stays; an image a page or a pending pass still uses is held by them (OwnedImageHeld) and is not retired.)
      if (OwnedOnly() && !owned_newest_.empty()) {
        size_t oldest = SIZE_MAX;
        for (size_t i = 0; i < owned_newest_.size(); ++i) {
          const int16_t img = owned_newest_[i].image;
          if (img == keep) continue;
          // (Review C4: only a record whose image nothing else holds frees anything.)
          if ((owned_pending_.valid && owned_pending_.image == img) ||
              std::any_of(owned_parked_.begin(), owned_parked_.end(), [&](const OwnedPass& q) { return q.image == img; }) ||
              std::any_of(owned_restorable_.begin(), owned_restorable_.end(), [&](const OwnedPass& q) { return q.image == img; }) ||
              std::any_of(owned_pages_.begin(), owned_pages_.end(), [&](const OwnedPage& pg) { return pg.image == img; })) {
            continue;
          }
          if (oldest == SIZE_MAX || owned_newest_[i].last_tick < owned_newest_[oldest].last_tick) oldest = i;
        }
        if (oldest != SIZE_MAX) {
          owned_newest_.erase(owned_newest_.begin() + oldest);
          ++g_m2a.records_evicted;
          continue;
        }
      }
      return;
    }
    OwnedRetireImage(size_t(best));
  }
}

bool NativeRenderer::OwnedImageAvailable(uint32_t width, uint32_t height, uint32_t kind, bool depth) {
  if (!owned_frame_on_) return true;  // (the list mode's checks are OwnedImageFor's, as they were)
  const uint64_t colour = uint64_t(width) * height * (kind == 2 ? 8u : 4u) * OwnedLayers(), depth_bytes = uint64_t(width) * height * 8u * OwnedLayers();  // (Session 94: x the views' layers)
  for (size_t i = 0; i < owned_images_.size(); ++i) {
    const OwnedImage& o = owned_images_[i];
    if (o.width != width || o.height != height || o.kind != kind || OwnedImageHeld(int16_t(i))) continue;
    const uint64_t more = depth && o.depth == VK_NULL_HANDLE ? depth_bytes : 0;
    if (!more) return true;
    OwnedEvictFor(more, int16_t(i));  // (not the image this very request is about)
    return !OwnedOverBudget(more);
  }
  bool has_scratch = false;
  for (const OwnedScratchDepth& d : owned_scratch_depths_) has_scratch = has_scratch || (d.width == width && d.height == height);
  const uint64_t needed = colour + (has_scratch ? 0u : depth_bytes) + (depth ? depth_bytes : 0u);
  OwnedEvictFor(needed);
  return !OwnedOverBudget(needed);
}

bool NativeRenderer::OwnedOverBudget(uint64_t more) const {
  const uint64_t budget = uint64_t(std::max(16, REXCVAR_GET(gpu_native_owned_frame_budget_mb))) << 20;
  return owned_bytes_ + more > budget;
}

namespace {
// Session 93 (gpu_owned_tiler_passes): what was made, for the window line (cumulative: images are made once, at first use).
uint32_t g_tiler_transient_made = 0, g_tiler_format_lists = 0, g_tiler_transient_fallback = 0;
// Session 94 (gpu_owned_image_format_list): colour images made without MUTABLE_FORMAT; usage STORAGE dropped (none: no owned image has storage it does not use).
uint32_t g_afbc_mutable_dropped = 0;
// An image of ours with its view (Session 82: one place, as colour images, scratch depth images and depth images all
// need it). Session 93: view_formats (the formats of every view made of the image, its own first) chains a format list.
bool OwnedMakeImage(const ui::vulkan::VulkanDevice* vulkan_device, uint32_t width, uint32_t height, VkFormat format,
                    VkImageUsageFlags usage, VkImageAspectFlags aspects, VkImageCreateFlags flags, VkImage& image,
                    VkDeviceMemory& memory, VkImageView& view, const VkFormat* view_formats = nullptr, uint32_t view_format_count = 0,
                    uint32_t layers = 1, VkImageView* array_view = nullptr) {
  // (Session 94, gpu_d3d_decouple: `layers` > 1 makes a layered image for the multiview passes. With `array_view` given, `view` is the 2D view of LAYER 0 -
  // the view whose picture is today's, which every consumer of `view` keeps seeing - and *array_view the whole-image 2D_ARRAY view the framebuffer takes; without it
  // `view` is the whole 2D_ARRAY view (depth images: only ever framebuffer attachments).)
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  VkImageFormatListCreateInfo format_list = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
  VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.flags = flags;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format;
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = layers;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (view_format_count && vulkan_device->extensions().ext_1_2_KHR_image_format_list) {
    format_list.viewFormatCount = view_format_count;
    format_list.pViewFormats = view_formats;
    image_info.pNext = &format_list;
  }
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, image_info,
                                                        ui::vulkan::util::MemoryPurpose::kDeviceLocal, image, memory)) {
    return false;
  }
  if (image_info.pNext) ++g_tiler_format_lists;
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  if (layers > 1) {
    if (array_view) {
      view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects, 0, 1, 0, 1);  // (review flip: layer 0 is the picture)
      if (dfn.vkCreateImageView(vulkan_device->device(), &view_info, nullptr, &view) != VK_SUCCESS) return false;
    }
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects, 0, 1, 0, layers);
    return dfn.vkCreateImageView(vulkan_device->device(), &view_info, nullptr, array_view ? array_view : &view) == VK_SUCCESS;
  }
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(aspects);
  return dfn.vkCreateImageView(vulkan_device->device(), &view_info, nullptr, &view) == VK_SUCCESS;
}

// Session 93 (gpu_owned_tiler_passes, piece 3): a depth + stencil image that only ever lives inside a render pass - TRANSIENT_ATTACHMENT usage, bound to
// lazily allocated memory (a tile-based GPU never gives it backing store). False when the device has no lazily allocated memory type the image can use
// (nothing is left made; the caller makes an ordinary one). The image is never loaded, stored, sampled or copied.
bool OwnedMakeTransientDepth(const ui::vulkan::VulkanDevice* vulkan_device, uint32_t width, uint32_t height, VkFormat format, VkImage& image,
                             VkDeviceMemory& memory, VkImageView& view) {
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkPhysicalDeviceMemoryProperties properties = {};
  vulkan_device->vulkan_instance()->functions().vkGetPhysicalDeviceMemoryProperties(vulkan_device->physical_device(), &properties);
  bool any_lazy = false;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    any_lazy = any_lazy || (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
  }
  if (!any_lazy) return false;
  VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format;
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (dfn.vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
    image = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryRequirements requirements = {};
  dfn.vkGetImageMemoryRequirements(device, image, &requirements);
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if (!(requirements.memoryTypeBits & (1u << i))) continue;
    const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
    if (!(flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)) continue;
    // (never a host-visible one: an image of optimal tiling needs none, and a lazily allocated type that is also host visible is not a tile memory)
    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) continue;
    type = i;
    break;
  }
  if (type == UINT32_MAX) {
    dfn.vkDestroyImage(device, image, nullptr);
    image = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryAllocateInfo allocate_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = type;
  VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
    dedicated.image = image;
    allocate_info.pNext = &dedicated;
  }
  if (dfn.vkAllocateMemory(device, &allocate_info, nullptr, &memory) != VK_SUCCESS) {
    memory = VK_NULL_HANDLE;
    dfn.vkDestroyImage(device, image, nullptr);
    image = VK_NULL_HANDLE;
    return false;
  }
  if (dfn.vkBindImageMemory(device, image, memory, 0) != VK_SUCCESS) {
    dfn.vkFreeMemory(device, memory, nullptr);
    memory = VK_NULL_HANDLE;
    dfn.vkDestroyImage(device, image, nullptr);
    image = VK_NULL_HANDLE;
    return false;
  }
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  if (dfn.vkCreateImageView(device, &view_info, nullptr, &view) != VK_SUCCESS) {
    view = VK_NULL_HANDLE;
    dfn.vkDestroyImage(device, image, nullptr);
    dfn.vkFreeMemory(device, memory, nullptr);
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    return false;
  }
  return true;
}
}  // namespace

// ---- Session 93: the tile-based render pass path (gpu_owned_tiler_passes) --------------------------------------------------------------------------
// The depth of a pass is "dead" when nothing reads it after the pass. It is not proven from the code (reads come from the guest's copies and samplers, which
// vary with the scene); it is LEARNED per pass key (base, pitch, format, first pixel shader): after kTilerDeadAfter passes of a key in a row whose depth
// nothing read before the image's next pass began, the key's next passes that start with ClearBoth are begun with the depth / stencil store DONT_CARE.
// A read of such a depth (every site that reads an image's depth calls OwnedNoteDepthRead; a resume with depth LOADs it, which counts) clears the key for
// good (banned) and is counted as a learned wrong - that one frame's read sees an undefined depth. The scene pass's depth is read by the post combine
// every frame, so its key never gets there.
namespace {
constexpr uint32_t kTilerDeadAfter = 8;
// Session 94 (gpu_owned_tiler_scene_reset): frames of unread streak a key needs; new keys in a window that mean a scene change; the window; a key silent
// this long is forgotten; frames after a reset with no dead depth applied.
constexpr uint32_t kTilerProofFrames = 120, kTilerBurstKeys = 16, kTilerBurstFrames = 120, kTilerStaleFrames = 600, kTilerQuietFrames = 240;
}  // namespace

void NativeRenderer::OwnedTilerSceneReset(const char* why) {
  // What the dead-depth learning knew is dropped (bans stay: a proven wrong is a fact about the game, not the scene); no dead depth is applied for a while.
  uint32_t cleared = 0;
  for (OwnedDepthLearn& l : owned_depth_learn_) {
    cleared += l.dead ? 1u : 0u;
    l.dead = false;
    l.unread = 0;
    l.streak_frame = 0;
  }
  tl_quiet_until_ = frame_ + kTilerQuietFrames;
  tl_ring_count_ = 0;
  ++tl_resets_;
  static uint32_t logged = 0;
  if (logged < 24) {
    ++logged;
    REXGPU_INFO("[native][owned-tiler] frame {}: scene reset #{} ({}): {} dead keys cleared of {}, no dead depth for {} frames", frame_, tl_resets_, why, cleared,
                owned_depth_learn_.size(), kTilerQuietFrames);
  }
}

bool NativeRenderer::OwnedTilerOn() const { return REXCVAR_GET(gpu_owned_tiler_passes); }
// (Session 94: the format-list / transient helpers - the older switch implies both, as in Session 93.)
static bool OwnedFormatListOn() { return REXCVAR_GET(gpu_owned_tiler_passes) || REXCVAR_GET(gpu_owned_image_format_list); }
static bool OwnedTransientOn() { return REXCVAR_GET(gpu_owned_tiler_passes) || REXCVAR_GET(gpu_owned_transient_attachments); }

// ---- Session 95 (TILER): gpu_tiler_poison ------------------------------------------------------------------------------------------------------------
// The poisoned load ops of the owned render passes (made in OwnedMakeImage below) and the clear values their begins give. Both read the same table so the render
// pass and the begin cannot disagree: colour poisoned where the cover pass drops the load (gpu_owned_cover_dont_care), depth + stencil poisoned where the
// variant does not load it (cover, clear colour, resume without depth). The variant numbers are NativeRenderer's private enum (native_renderer.h, kOwnedPassCover ...).
namespace {
constexpr uint32_t kPzCover = 0, kPzClearColor = 1, kPzClearBoth = 2, kPzResume = 3, kPzResumeNoDepth = 4, kPzClearBothDead = 5;
}  // namespace
// (Latched at the first read: the render passes are made once, so a live flip would give a CLEAR-load pass a begin with no clear values.)
static bool OwnedPoisonOn() {
  static const bool on = REXCVAR_GET(gpu_tiler_poison);
  return on;
}
static bool OwnedPoisonColour(uint32_t v) { return v == kPzCover && REXCVAR_GET(gpu_owned_cover_dont_care); }
static bool OwnedPoisonDepth(uint32_t v) { return v == kPzCover || v == kPzClearColor || v == kPzResumeNoDepth; }
// Fills `out` (two values) for a begin of render pass variant `v` whose own clear values are `real` (null: none); returns the clear value count to give.
// `site`: 0 = OwnedRecord begin (first start), 1 = OwnedRecord begin (resumed), 2 = OwnedBeginClearedPass, 3 = OwnedResumeNow. `first_depth`: a first begin of the
// Resume variant that has depth of its own (loaded start: depth goes UNDEFINED -> LOAD, not poisoned - counted so the gap is visible).
static uint32_t OwnedPoisonClears(uint32_t v, const VkClearValue* real, VkClearValue* out, uint32_t width, uint32_t height, uint32_t site = 0,
                                  bool first_depth = false) {
  static std::atomic<uint32_t> colour{0}, depth{0}, begins{0}, nodepth_site[4], resume_first_depth{0};
  if (v == kPzResumeNoDepth) ++nodepth_site[site & 3];
  if (first_depth && v == kPzResume) ++resume_first_depth;
  const bool pc = OwnedPoisonColour(v), pd = OwnedPoisonDepth(v);
  const bool colour_cleared = v == kPzClearColor || v == kPzClearBoth || v == kPzClearBothDead;
  const bool depth_cleared = v == kPzClearBoth || v == kPzClearBothDead;
  out[0] = out[1] = VkClearValue();
  if (real) {
    if (colour_cleared) out[0] = real[0];
    if (depth_cleared) out[1] = real[1];
  }
  if (pc) out[0].color = {{1.0f, 0.0f, 1.0f, 1.0f}};
  if (pd) out[1].depthStencil = {0.0f, 0xA5u};
  if (pc) ++colour;
  if (pd) ++depth;
  if (pc || pd) {
    const uint32_t n = ++begins;
    if (n <= 12 || n % 4000 == 0) {
      REXGPU_INFO("[native][owned-tiler] poison: pass #{} variant {} {}x{}: colour {} depth {} (totals: colour {} depth {}; variant 4 by site first {} resumed {} cleared {} resume-now {}; "
                  "first Resume begins with depth, UNDEFINED -> LOAD, not poisoned: {})",
                  n, v, width, height, pc ? "magenta" : "kept", pd ? "0.0/0xA5" : "kept", colour.load(), depth.load(), nodepth_site[0].load(),
                  nodepth_site[1].load(), nodepth_site[2].load(), nodepth_site[3].load(), resume_first_depth.load());
    }
  }
  return v == kPzResume ? 0u : 2u;
}

void NativeRenderer::OwnedDepthAccount(int16_t image) {
  if (image < 0 || size_t(image) >= owned_images_.size()) return;
  OwnedImage& o = owned_images_[image];
  if (o.depth_learn < 0 || size_t(o.depth_learn) >= owned_depth_learn_.size()) {
    o.depth_learn = -1;
    return;
  }
  OwnedDepthLearn& learn = owned_depth_learn_[o.depth_learn];
  if (!o.depth_read_since) {
    ++owned_frame_stats_.tl_unread_passes;
    if (REXCVAR_GET(gpu_owned_tiler_scene_reset)) {
      // (Session 94: dead only on proof from THIS scene - K passes AND kTilerProofFrames frames of an unread streak.)
      if (learn.unread == 0) learn.streak_frame = frame_;
      ++learn.unread;
      if (!learn.banned && !learn.read_ever && learn.unread >= kTilerDeadAfter && frame_ - learn.streak_frame >= kTilerProofFrames) learn.dead = true;
    } else if (!learn.banned && ++learn.unread >= kTilerDeadAfter) {
      learn.dead = true;
    }
  } else {
    learn.unread = 0;
  }
  o.depth_learn = -1;
  o.depth_read_since = false;
  o.depth_dead_applied = false;
}

void NativeRenderer::OwnedNoteDepthRead(int16_t image, const char* site) {
  if (image < 0 || size_t(image) >= owned_images_.size()) return;
  OwnedImage& o = owned_images_[image];
  o.depth_read_since = true;
  if (o.depth_learn < 0 || size_t(o.depth_learn) >= owned_depth_learn_.size()) return;
  OwnedDepthLearn& learn = owned_depth_learn_[o.depth_learn];
  learn.unread = 0;
  learn.read_ever = true;
  if (o.depth_dead_applied) {
    o.depth_dead_applied = false;
    learn.dead = false;
    learn.banned = true;
    ++owned_frame_stats_.tl_learned_wrong;
    if (REXCVAR_GET(gpu_owned_tiler_scene_reset)) OwnedTilerSceneReset("a learned wrong");
    static uint32_t logged = 0;
    if (logged < 16) {
      ++logged;
      REXGPU_INFO("[native][owned] frame {}: LEARNED WRONG: the depth of the pass base {} pitch {} format {} ps {:016X} (image {}) was read after it was stored dead by {}; the key is banned",
                  frame_, learn.key.base, learn.key.pitch, learn.key.format, learn.key.ps, image, site);
    }
  } else if (learn.dead) {
    learn.dead = false;  // (the switch is off or the pass did not start with ClearBoth: it would have been wrong)
    ++owned_frame_stats_.tl_would_wrong;
  }
}

uint32_t NativeRenderer::OwnedTilerVariant(int16_t image, uint32_t variant, bool has_depth, bool first, const OwnedPassKey* key) {
  if (image < 0 || size_t(image) >= owned_images_.size()) return variant;
  OwnedFrameStats& st = owned_frame_stats_;
  const bool on = OwnedTilerOn();
  if (has_depth) {
    if (!on) return variant;  // (switch off: the learned table is not touched)
    if (!first) {
      // (A resume with depth LOADs it: the depth is read - by the pass itself.)
      ++st.tl_resume_depth;
      OwnedNoteDepthRead(image, "the pass's own resume (LOAD)");
      return variant;
    }
    OwnedDepthAccount(image);  // (the pass whose depth the image held is over)
    ++st.tl_depth_passes;
    OwnedImage& o = owned_images_[image];
    o.depth_read_since = false;
    o.depth_dead_applied = false;
    o.depth_learn = -1;
    if (key && (key->base || key->pitch || key->format || key->ps)) {
      // (Session 94: a sliding window - the last kTilerBurstKeys creations within kTilerBurstFrames frames is a scene change.)
      const auto note_new_key = [&]() {
        if (!REXCVAR_GET(gpu_owned_tiler_scene_reset)) return;
        tl_ring_[tl_ring_pos_] = frame_;
        tl_ring_pos_ = (tl_ring_pos_ + 1) % kTilerBurstKeys;
        if (tl_ring_count_ < kTilerBurstKeys) ++tl_ring_count_;
        if (tl_ring_count_ >= kTilerBurstKeys && frame_ - tl_ring_[tl_ring_pos_] <= kTilerBurstFrames) OwnedTilerSceneReset("a burst of new pass keys");
      };
      const auto it = owned_depth_learn_index_.find(*key);
      if (it != owned_depth_learn_index_.end()) {
        o.depth_learn = it->second;
        if (REXCVAR_GET(gpu_owned_tiler_scene_reset)) {
          OwnedDepthLearn& known = owned_depth_learn_[o.depth_learn];
          if (known.last_frame && frame_ - known.last_frame > kTilerStaleFrames) {  // (silent for 10 s: another scene's knowledge)
            known.dead = false;
            known.unread = 0;
            known.streak_frame = 0;
          }
          known.last_frame = frame_;
        }
      } else if (owned_depth_learn_.size() < 4096) {
        OwnedDepthLearn learn;
        learn.key = *key;
        learn.last_frame = frame_;
        owned_depth_learn_.push_back(learn);
        o.depth_learn = int32_t(owned_depth_learn_.size() - 1);
        owned_depth_learn_index_.emplace(*key, o.depth_learn);
        note_new_key();
      } else {
        note_new_key();  // (refused by the 4096 cap: still a new key for the burst count)
      }
    }
    if (variant == kOwnedPassClearBoth && o.depth_learn >= 0 && owned_depth_learn_[o.depth_learn].dead && !owned_depth_learn_[o.depth_learn].read_ever &&
        !(REXCVAR_GET(gpu_owned_tiler_scene_reset) && frame_ < tl_quiet_until_) &&
        owned_render_passes_[kOwnedPassClearBothDead][o.kind] != VK_NULL_HANDLE) {
      ++st.tl_dead_would;
      if (on) {
        ++st.tl_dead_stored;
        o.depth_dead_applied = true;
        return kOwnedPassClearBothDead;
      }
    }
    return variant;
  }
  if (variant != kOwnedPassResume) return variant;
  // A resume of a pass without depth of its own: its framebuffer holds the shared scratch depth, which nothing reads.
  if (!first) ++st.tl_resume_nodepth;
  const bool scratch_transient = owned_images_[image].scratch >= 0 && size_t(owned_images_[image].scratch) < owned_scratch_depths_.size() &&
                                 owned_scratch_depths_[owned_images_[image].scratch].transient;
  if ((on || scratch_transient) && owned_render_passes_[kOwnedPassResumeNoDepth][owned_images_[image].kind] != VK_NULL_HANDLE) {  // (a transient image is never loaded or stored, whatever the switch says now)
    if (!first) ++st.tl_resume_nodepth_used;
    return kOwnedPassResumeNoDepth;
  }
  return variant;
}

int32_t NativeRenderer::OwnedScratchDepthFor(uint32_t width, uint32_t height) {
  // (Session 82: the depth attachment every owned pass without kept depth is given - the truth pipelines' render pass
  // has one - shared by the images of a size: never read, its contents are not kept.)
  for (size_t i = 0; i < owned_scratch_depths_.size(); ++i) {
    if (owned_scratch_depths_[i].width == width && owned_scratch_depths_[i].height == height) return int32_t(i);
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  OwnedScratchDepth d;
  d.width = width;
  d.height = height;
  // (Session 93, gpu_owned_tiler_passes: a transient image where the device has lazily allocated memory - then no pass ever loads or stores it, the
  // resume-without-depth variant takes the resumes; the TRANSFER_SRC usage nothing used - the scratch images appear only in pass-start barriers and
  // framebuffers - is dropped with the switch on.)
  const bool tiler = OwnedTransientOn();
  d.transient = tiler && OwnedLayers() == 1 && OwnedMakeTransientDepth(vulkan_device, width, height, post_ds_format_, d.image, d.memory, d.view);  // (Session 94: no transient image for the layered ones)
  if (d.transient) {
    ++g_tiler_transient_made;
    REXGPU_INFO("[native][owned] scratch depth {}x{} made transient (lazily allocated memory)", width, height);
  } else if (tiler) {
    ++g_tiler_transient_fallback;
  }
  if (!d.transient && !OwnedMakeImage(vulkan_device, width, height, post_ds_format_,
                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | (tiler ? 0u : VkImageUsageFlags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)),
                      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, d.image, d.memory, d.view, nullptr, 0, OwnedLayers())) {
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, d.view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, d.image);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, d.memory);
    return -1;
  }
  owned_bytes_ += uint64_t(width) * height * 8 * OwnedLayers();
  owned_scratch_depths_.push_back(d);
  return int32_t(owned_scratch_depths_.size() - 1);
}

bool NativeRenderer::OwnedEnsureDepth(int16_t index) {
  // (Session 82: the depth image of an owned image, made the first time a pass keeps depth: with its framebuffer.)
  if (index < 0 || size_t(index) >= owned_images_.size()) return false;
  if (owned_images_[index].depth != VK_NULL_HANDLE) return true;
  const uint64_t more = uint64_t(owned_images_[index].width) * owned_images_[index].height * 8 * OwnedLayers();
  if (owned_frame_on_ && OwnedOverBudget(more)) {
    // (Session 87, chains: as the feedback texture - unused images are evicted first, never `index` itself, which the caller is about to use.)
    if (OwnedChains()) OwnedEvictFor(more, index);
    if (OwnedOverBudget(more)) {
      static uint32_t budget_logged = 0;
      if (OwnedChains() && budget_logged < 8) {
        ++budget_logged;
        REXGPU_INFO("[native][owned] frame {}: no depth image for our image {} ({}x{}): {} MB in use + {} MB over the owned-frame budget", frame_, index,
                    owned_images_[index].width, owned_images_[index].height, owned_bytes_ >> 20, more >> 20);
      }
      return false;
    }
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  OwnedImage& o = owned_images_[index];
  bool ok = OwnedMakeImage(vulkan_device, o.width, o.height, post_ds_format_,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,  // (Session 82, P1: sampled by the depth copy's fill)
                           VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, o.depth, o.depth_memory,
                           o.depth_view, nullptr, 0, OwnedLayers());
  if (ok) {
    const VkImageView views[2] = {o.array_view != VK_NULL_HANDLE ? o.array_view : o.view, o.depth_view};
    VkFramebufferCreateInfo fb_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb_info.renderPass = owned_render_passes_[kOwnedPassCover][o.kind];
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = views;
    fb_info.width = o.width;
    fb_info.height = o.height;
    fb_info.layers = 1;
    ok = dfn.vkCreateFramebuffer(device, &fb_info, nullptr, &o.framebuffer_depth) == VK_SUCCESS;
  }
  if (!ok) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, o.framebuffer_depth);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_sample_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_stencil_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, o.depth);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, o.depth_memory);
    REXGPU_ERROR("[native][owned] could not make a {}x{} depth image", o.width, o.height);
    return false;
  }
  o.bytes += more;
  owned_bytes_ += more;
  return true;
}

int16_t NativeRenderer::OwnedImageFor(uint32_t width, uint32_t height, uint32_t kind) {
  // One not held now: a step reads the page an earlier one left in another image.
  const auto find = [&](uint32_t& of_size) -> int16_t {
    of_size = 0;
    for (size_t i = 0; i < owned_images_.size(); ++i) {
      const OwnedImage& o = owned_images_[i];
      if (o.width != width || o.height != height || o.kind != kind) continue;
      // (Session 82: with the owned frame off the list mode's cap of four counts its own images, not the frame's.)
      if (owned_frame_on_ || !o.frame_made) ++of_size;
      if (!OwnedImageHeld(int16_t(i))) {
        owned_images_[i].graph_dropped = false;  // (Session 95: a new use of the image starts its contents over)
        return int16_t(i);
      }
    }
    return -1;
  };
  // (Session 82: with the owned frame on, the pool grows on demand up to the memory budget - no count caps; with it off
  // the caps are as they were: 4 of a size, 32 in all.)
  const bool frame = owned_frame_on_;
  uint32_t of_size = 0;
  int16_t found = find(of_size);
  if (found >= 0) {
    owned_images_[found].used = frame_;
    return found;
  }
  const bool has_scratch = [&]() {
    for (const OwnedScratchDepth& d : owned_scratch_depths_) {
      if (d.width == width && d.height == height) return true;
    }
    return false;
  }();
  const uint64_t needed = (uint64_t(width) * height * (kind == 2 ? 8u : 4u) + (has_scratch ? 0u : uint64_t(width) * height * 8u)) * OwnedLayers();
  if (frame ? OwnedOverBudget(needed) : of_size >= kOwnedImagesPerSize) {
    // All of this size held (or the budget is full): those held only by dirty tiles give them to the truth images now
    // and are free again.
    for (OwnedDirty& d : owned_dirty_) {
      if (d.image < 0 || size_t(d.image) >= owned_images_.size()) continue;
      const OwnedImage& o = owned_images_[d.image];
      if (o.width == width && o.height == height && o.kind == kind) {
        owned_mat_why_ = "image pool pressure";
        OwnedMaterialize(d, d.tiles);
      }
    }
    OwnedPruneDirty();
    found = find(of_size);
    if (found >= 0) {
      owned_images_[found].used = frame_;
      return found;
    }
    if (frame) OwnedEvictFor(needed);
    if (frame ? OwnedOverBudget(needed) : of_size >= kOwnedImagesPerSize) return -1;
  }
  if (!frame) {
    uint32_t live = 0;
    for (const OwnedImage& o : owned_images_) live += (o.width && !o.frame_made) ? 1u : 0u;
    if (live >= 32) return -1;
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  // The render passes: the truth passes' attachments (colour of the class, the post depth format) and, as the truth
  // pipelines' passes, no subpass dependencies (their count is part of render pass compatibility: validation
  // VUID-vkCmdDraw-renderPass-02684) - the barriers around them are OwnedRecord's and the commit's. Both attachments
  // stay in their attachment layouts. Cover: the colour not loaded (every pixel is drawn), depth unused; clear colour:
  // the colour cleared, depth unused; clear both: colour, depth and stencil cleared and kept; resume: all loaded and
  // kept.
  if (owned_render_passes_[kOwnedPassCover][kind] == VK_NULL_HANDLE) {
    VkAttachmentDescription attachments[2] = {};
    attachments[0].format = OwnedFormat(kind);
    attachments[1].format = post_ds_format_;
    for (VkAttachmentDescription& a : attachments) a.samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].initialLayout = attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1].initialLayout = attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference ds_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &ds_ref;
    VkRenderPassCreateInfo info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 2;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    bool ok = true;
    bool any_transient = false;
    VkRenderPassMultiviewCreateInfo decouple_mv = {};
    DecoupleRenderPass(info, decouple_mv);  // (Session 94: the view mask, when gpu_d3d_decouple is on)
    for (const OwnedScratchDepth& sd : owned_scratch_depths_) any_transient = any_transient || sd.transient;
    const bool tiler_variants = OwnedTilerOn() || OwnedTransientOn() || any_transient;  // (Session 93: the two tiler variants only when the switch is on or a transient scratch needs one)
    for (uint32_t v = 0; v < kOwnedPasses && ok; ++v) {
      if ((v == kOwnedPassResumeNoDepth || v == kOwnedPassClearBothDead) && !tiler_variants) continue;
      // (Session 93: ResumeNoDepth loads the colour and neither loads nor stores depth / stencil; ClearBothDead clears depth / stencil and does not store them.)
      // (Session 95, Lane E: gpu_native_owned_dont_care_clear turns the DONT_CARE loads into CLEAR; the begins pass the values.)
      // (Review s95-all: latched once, by the first kind made: a later kind must not change what the begins of the passes already made give.)
      if (v == 0 && g_owned_dc_mode.load(std::memory_order_relaxed) < 0) g_owned_dc_mode.store(REXCVAR_GET(gpu_native_owned_dont_care_clear), std::memory_order_relaxed);
      const VkAttachmentLoadOp dc_load = OwnedDcMode() ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                                                      : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      const VkAttachmentLoadOp color_load = v == kOwnedPassCover    ? (REXCVAR_GET(gpu_owned_cover_dont_care) ? dc_load : VK_ATTACHMENT_LOAD_OP_LOAD)
                                            : (v == kOwnedPassResume || v == kOwnedPassResumeNoDepth) ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                                    : VK_ATTACHMENT_LOAD_OP_CLEAR;
      const bool keep_depth = v == kOwnedPassClearBoth || v == kOwnedPassResume;
      const VkAttachmentLoadOp depth_load = (v == kOwnedPassClearBoth || v == kOwnedPassClearBothDead) ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                            : v == kOwnedPassResume  ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                                     : dc_load;
      attachments[0].loadOp = OwnedPoisonOn() && OwnedPoisonColour(v) ? VK_ATTACHMENT_LOAD_OP_CLEAR : color_load;  // (Session 95: gpu_tiler_poison)
      attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      attachments[1].loadOp = attachments[1].stencilLoadOp = OwnedPoisonOn() && OwnedPoisonDepth(v) ? VK_ATTACHMENT_LOAD_OP_CLEAR : depth_load;
      attachments[1].storeOp = attachments[1].stencilStoreOp =
          keep_depth ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
      ok = dfn.vkCreateRenderPass(device, &info, nullptr, &owned_render_passes_[v][kind]) == VK_SUCCESS;
    }
    if (!ok) {
      for (uint32_t v = 0; v < kOwnedPasses; ++v) {
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, owned_render_passes_[v][kind]);
      }
      owned_failed_ = true;
      REXGPU_ERROR("[native][owned] could not make the render passes: owned steps off");
      return -1;
    }
  }
  OwnedImage o;
  o.width = width;
  o.height = height;
  o.kind = kind;
  // (Colour: copied into the truth images by OwnedMaterialize. Sampled through UNORM and SNORM views over the same
  // bits - the textures' unsigned and signed host formats: MUTABLE_FORMAT.)
  // (Session 93, gpu_owned_tiler_passes: the formats of every view made of the image - its own (the fill / write-back sources use `view`), and the
  // sampler views OwnedServeView makes of an RGBA8 image, UNORM and SNORM; a 16-bit class image is only ever viewed as itself - declared so a
  // tile-based driver can keep the image compressed. No view of an owned colour image is made anywhere else.)
  const VkFormat colour_view_formats[2] = {OwnedFormat(kind), VK_FORMAT_R8G8B8A8_SNORM};
  // (Session 94, gpu_owned_image_format_list: a kind 2 (16-bit class) image is viewed only as itself - OwnedServeView refuses every sampler of it
  // (`o.kind != 1`), the fill / write-back sources use `view` - so it is made without MUTABLE_FORMAT, and then without a format list: nothing to list.)
  const bool drop_mutable = kind == 2 && REXCVAR_GET(gpu_owned_image_format_list);
  const bool format_list = OwnedFormatListOn() && !drop_mutable;
  if (drop_mutable) ++g_afbc_mutable_dropped;
  {
    static bool logged = false;
    if (!logged && (drop_mutable || format_list)) {
      logged = true;
      REXGPU_INFO("[native] format lists on: first owned colour image {}x{} kind {}: {} (extension image_format_list {})", width, height, kind,
                  drop_mutable ? "no MUTABLE_FORMAT, no list (viewed only as itself)" : "format list chained",
                  vulkan_device->extensions().ext_1_2_KHR_image_format_list ? "present" : "ABSENT: no list can be chained");
    }
  }
  bool ok = OwnedMakeImage(vulkan_device, width, height, OwnedFormat(kind),
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT, drop_mutable ? 0u : VkImageCreateFlags(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT), o.image, o.memory, o.view,
                           format_list ? colour_view_formats : nullptr, format_list ? (kind == 2 ? 1u : 2u) : 0u, OwnedLayers(),
                           OwnedLayers() > 1 ? &o.array_view : nullptr);
  // (Session 82: the depth attachment of the passes that keep none is the scratch image of this size; its own depth is
  // made when a pass first keeps depth, OwnedEnsureDepth.)
  const int32_t scratch = ok ? OwnedScratchDepthFor(width, height) : -1;
  ok = ok && scratch >= 0;
  if (ok) {
    const VkImageView views[2] = {o.array_view != VK_NULL_HANDLE ? o.array_view : o.view, owned_scratch_depths_[scratch].view};
    VkFramebufferCreateInfo fb_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb_info.renderPass = owned_render_passes_[kOwnedPassCover][kind];
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = views;
    fb_info.width = width;
    fb_info.height = height;
    fb_info.layers = 1;
    ok = dfn.vkCreateFramebuffer(device, &fb_info, nullptr, &o.framebuffer) == VK_SUCCESS;
  }
  if (!ok) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, o.framebuffer);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.array_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, o.image);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, o.memory);
    // (Session 82: with the owned frame on a failure is a counted `pool` fallback every time, not the end of owned
    // passes for the run.)
    static bool failure_logged = false;
    if (!failure_logged || !frame) REXGPU_ERROR("[native][owned] could not make a {}x{} image{}", width, height,
                                                 frame ? "" : ": owned steps off");
    failure_logged = true;
    if (!frame) owned_failed_ = true;
    return -1;
  }
  o.scratch = scratch;
  o.bytes = uint64_t(width) * height * (kind == 2 ? 8u : 4u) * OwnedLayers();
  o.used = frame_;
  o.frame_made = frame;
  owned_bytes_ += o.bytes;
  ++owned_frame_stats_.images_made;
  REXGPU_INFO("[native][owned] image {}x{} (class {}) made", width, height, kind);
  for (size_t i = 0; i < owned_images_.size(); ++i) {  // (an evicted slot first)
    if (!owned_images_[i].width) {
      owned_images_[i] = o;
      return int16_t(i);
    }
  }
  owned_images_.push_back(o);
  return int16_t(owned_images_.size() - 1);
}

VkResult NativeRenderer::OwnedAcquireFillSet(VkDescriptorSet& set) {
  // A fill source set (layout truth_store_set_layout_): a spare one whose last use has completed, its descriptors rewritten by the caller as for a new one
  // (rewriting a set a command of the running submission still uses is not allowed); else a new one from the pools.
  if (OwnedChains()) {
    const uint64_t completed = command_processor_.GetCompletedSubmission();
    for (size_t i = 0; i < owned_spare_sets_.size(); ++i) {
      if (owned_spare_sets_[i].submission > completed) continue;
      set = owned_spare_sets_[i].set;
      owned_spare_sets_.erase(owned_spare_sets_.begin() + i);
      ++owned_frame_stats_.fill_sets_reused;
      return VK_SUCCESS;
    }
  }
  if (!EnsureWbSetLayout()) return VK_ERROR_INITIALIZATION_FAILED;  // (Session 92, M2d: no truth objects needed for the layout)
  return OwnedAllocateSet(truth_store_set_layout_, set);
}

void NativeRenderer::OwnedSpareFillSet(VkDescriptorSet set) {
  if (set == VK_NULL_HANDLE || !OwnedChains()) return;
  owned_spare_sets_.push_back({set, command_processor_.GetCurrentSubmission()});
}

VkResult NativeRenderer::OwnedAllocateSet(VkDescriptorSetLayout layout, VkDescriptorSet& set) {
  // (Session 82: pools of a fixed size, a new one when the current is full - the page images and fill sets grow with
  // the budget, not with a count.)
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkResult result = VK_ERROR_OUT_OF_POOL_MEMORY;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (owned_pool_ == VK_NULL_HANDLE) {
      // (A fill source set: one buffer and three samplers - its unwritten stencil binding counts; review.)
      const VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
                                             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 192},
                                             {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32}};
      VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool_info.maxSets = 96;
      pool_info.poolSizeCount = 3;
      pool_info.pPoolSizes = sizes;
      if (dfn.vkCreateDescriptorPool(device, &pool_info, nullptr, &owned_pool_) != VK_SUCCESS) {
        owned_pool_ = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_POOL_MEMORY;
      }
      owned_pools_.push_back(owned_pool_);
    }
    VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = owned_pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &layout;
    result = dfn.vkAllocateDescriptorSets(device, &alloc, &set);
    if (result == VK_SUCCESS) return result;
    if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) return result;
    owned_pool_ = VK_NULL_HANDLE;  // (full: the next attempt makes another)
  }
  return result;
}

void NativeRenderer::OwnedRefreshList() {
  // (Copied under the cvar registry lock - a live set assigns the string from another thread; review.)
  const std::string list = rex::cvar::Query<std::string>("gpu_native_own_steps");
  if (list == owned_list_text_) return;
  owned_list_text_ = list;
  owned_list_.clear();
  owned_vs_list_.clear();
  size_t i = 0;
  while (i < list.size()) {
    const size_t end = std::min(list.find(',', i), list.size());
    std::string item = list.substr(i, end - i);
    const bool vertex = !item.empty() && (item[0] == 'v' || item[0] == 'V');
    if (vertex) item.erase(0, 1);
    char* stop = nullptr;
    const unsigned long long hash = std::strtoull(item.c_str(), &stop, 16);
    if (hash && stop && stop != item.c_str()) (vertex ? owned_vs_list_ : owned_list_).push_back(uint64_t(hash));
    i = end + 1;
  }
}

void NativeRenderer::OwnedDisable(uint64_t hash, bool vertex, const char* why) {
  std::vector<std::pair<uint64_t, uint64_t>>& disabled = vertex ? owned_disabled_vs_ : owned_disabled_;
  if (OwnedDisabled(disabled, hash)) return;
  disabled.erase(std::remove_if(disabled.begin(), disabled.end(),
                                [&](const std::pair<uint64_t, uint64_t>& d) { return d.first == hash; }),
                 disabled.end());
  disabled.emplace_back(hash, frame_ + 600);
  REXGPU_WARN("[native][owned] frame {}: {} {:016X} not owned for 600 frames ({})", frame_, vertex ? "vs" : "ps", hash,
              why);
}

void NativeRenderer::OwnedDisableAll(const std::vector<std::pair<uint64_t, bool>>& producers, const char* why) {
  for (const auto& [hash, vertex] : producers) OwnedDisable(hash, vertex, why);
}

std::bitset<2048> NativeRenderer::OwnedDrawTiles(const Draw& draw) const {
  // The EDRAM tiles a draw writes or tests (its colour; its depth / stencil), in guest pixels from its rectangle.
  std::bitset<2048> tiles;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  const uint32_t gx0 = uint32_t(std::max(rect[0], 0)) / scale_x, gy0 = uint32_t(std::max(rect[1], 0)) / scale_y;
  const uint32_t gx1 = (uint32_t(std::max(rect[2], 0)) + scale_x - 1) / scale_x;
  const uint32_t gy1 = (uint32_t(std::max(rect[3], 0)) + scale_y - 1) / scale_y;
  if (gx1 <= gx0 || gy1 <= gy0) return tiles;
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  if (draw.truth_color >= 0 && (draw.color_write_mask & 0xF)) {
    const auto format = xenos::ColorRenderTargetFormat((draw.color_info >> 16) & 0xF);
    tiles |= EdramTiles(draw.color_info & 0xFFF, pitch, xenos::IsColorRenderTargetFormat64bpp(format), gx0, gy0, gx1,
                        gy1);
  }
  // (Owned passes: a depth test reads the tiles too - an owned pass's depth is ours until its copy.)
  if (draw.truth_depth >= 0 &&
      (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil || draw.fast_clear)) {
    tiles |= EdramTiles(draw.depth_info & 0xFFF, pitch, false, gx0, gy0, gx1, gy1);
  }
  return tiles;
}

void NativeRenderer::OwnedEndPass() {
  if (!owned_pending_.open) return;
  owned_pending_.open = false;
  if (segment_log_frames_ > 0) {
    // (Session 87, step 3 diagnosis: who ends an owned pass's render pass - the splits of the entrance scene pass. Stack frames
    // symbolized with tools\perf\pass_ends.py like the truth passes' ends.)
    uintptr_t stack[6] = {};
    const uintptr_t base = VulkanCommandProcessor::DebugCaptureStack(stack, 6, 0);
    REXGPU_INFO("[native][seg] frame {} owned render pass end ({}x{}, {} draws recorded): base {:X} stack {:X} {:X} {:X} {:X} {:X} {:X}",
                frame_, owned_pending_.width, owned_pending_.height, owned_pending_.timer_draws, base, stack[0], stack[1], stack[2],
                stack[3], stack[4], stack[5]);
  }
  // (Session 95, r3gpu: which shaders a pass drew, while the pass log is on.)
  OwnedPassCensusFlush(frame_, owned_pending_.width, owned_pending_.height);
  command_processor_.deferred_command_buffer().CmdVkEndRenderPass();
  OwnedTimerEnd(owned_pending_.timer_segment, owned_pending_.timer_draws);  // (Session 82, P1)
  owned_pending_.timer_segment = UINT32_MAX;
}

void NativeRenderer::OwnedReplayPending(const char* why, bool pause, bool replay_committed) {
  // An owned pass whose copy did not follow as expected: its draws recorded into the truth images now, in order, as
  // they would have been (the same submission - their bindings and constants still hold - and nothing recorded since
  // reads or writes its tiles: this runs before what found it is recorded; its clears' pass is still on), so the
  // EDRAM state and the copy are exact; its shaders are not owned any more (review: a loss must not leave the tiles
  // without its draws). (A committed pass - copied already, its tiles dirty - just ends.)
  OwnedPass& p = owned_pending_;
  if (!p.valid) return;
  if (OwnedOnly()) return OwnedEndAsIs(why);  // (Session 92, M2a: nothing to replay into - the pass ends as it is, its image the surface's newest record)
  // (chains: a LOADED start holds what the truth path held, so its replay - the draws over that content - is exact: no reason to pause its key.)
  if (pause && OwnedChains() && p.start_kind == kOwnedStartResumed) pause = false;
  if (p.committed && !(replay_committed && p.edram_free && !p.draws.empty())) return OwnedEndCommitted(why);
  // (Session 82, P2: a committed defined-start pass - what an earlier copy served from our image does not change: the truth
  // images hold nothing of it - is replayed like an uncommitted one: the game's copy that follows reads its pixels, not
  // tiles it never drew into. The entrance replay's light cubes: the strip pass served the first cube's six faces, the
  // seventh copy - a cube the texture cache had not seen yet - ran the game's from empty tiles: blank cube in the match.)
  p.committed = false;
  OwnedEndPass();
  p.valid = false;
  ++truth_counts_.owned_lost_draw;
  if (pause && !REXCVAR_GET(gpu_native_own_steps_debug_replay)) OwnedPause(p.key, p.producers, why);
  OwnedFrameLost(p.census, owned_replay_reason_ ? owned_replay_reason_ : (pause ? kOwnedReasonLost : kOwnedReasonPresent), why);  // (Session 90, Q: a read-back copy no writer exists for)
  owned_replay_reason_ = 0;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned pass replayed ({} draws): {}", frame_, p.draws.size(), why);
  }
  OwnedReplayDraws();
}

void NativeRenderer::OwnedReplayDraws() {
  if (OwnedOnly()) return;  // (Session 92, M2a: no replay exists under the latch; every caller is gated - this is the net)
  OwnedPass& p = owned_pending_;
  // (What the draw being recorded was given for its served bindings is its own, kept across the replays - each
  // TruthRecordDraw drops serve_pending_ at its end; owned draws never had served bindings.)
  std::vector<VkImage> serve_pending;
  serve_pending.swap(serve_pending_);
  const ServeRename serve_rename = serve_rename_;
  serve_rename_.valid = false;
  // (Session 87, review A1: the replayed draws go through OwnedTakeDraw, which clears the sampled-images list - that list is the draw
  // being recorded now, kept across as serve_pending_ is.)
  std::vector<int16_t> sampled_images;
  sampled_images.swap(owned_sampled_images_);
  std::vector<Draw> draws;
  draws.swap(p.draws);
  p.kept_sampled.clear();  // (Session 87: the kept draws go to the truth path now)
  std::vector<OwnedCrowdCall> crowd;  // (Session 82: our crowd draws, each after the draw it follows)
  crowd.swap(p.crowd_calls);
  std::vector<std::pair<size_t, std::vector<VkImage>>> serves;  // (Session 82, P1 item 2: a draw's served images)
  serves.swap(p.serves);
  std::vector<Draw> held_clears;  // (Session 82: the clears this pass started from, never recorded: first)
  held_clears.swap(p.held_clears);
  // (s82 hold investigation: clears held for a LATER pass stay held across the replay - each replayed draw goes through
  // OwnedTakeDraw, whose "not taken" path would otherwise record them now, ahead of the earlier pass's draws.)
  OwnedHeldClears later_held;
  std::swap(later_held, owned_held_);
  owned_replaying_ = true;
  for (const Draw& d : held_clears) {
    truth_foreign_bind_ = true;
    truth_bound_valid_ = false;
    serve_pending_.clear();
    TruthRecordDraw(d);
  }
  for (size_t i = 0; i < draws.size(); ++i) {
    truth_foreign_bind_ = true;  // (its sets are not the command processor's current ones: bound again)
    truth_bound_valid_ = false;
    serve_pending_.clear();
    for (const auto& s : serves) {
      if (s.first == i) serve_pending_ = s.second;  // (the truth draw makes them readable and drops them at its end)
    }
    TruthRecordDraw(draws[i]);
    for (const OwnedCrowdCall& call : crowd) {
      // (Right after its first member's state, in the truth pass that draw is in; a draw the truth path left out - no
      // pipeline yet - has no pass to draw into: as RecordOwnCrowd would have refused.)
      if (call.after_draws == i + 1 && truth_pass_open_) OwnCrowdRecordCall(call);
    }
  }
  truth_foreign_bind_ = true;
  truth_bound_valid_ = false;
  owned_replaying_ = false;
  std::swap(later_held, owned_held_);
  serve_pending_.swap(serve_pending);
  serve_rename_ = serve_rename;
  owned_sampled_images_.swap(sampled_images);
  command_processor_.NativeForgetState();
}

// ---------------------------------------------------------------------------------------------------------------
// Session 82 (gpu_native_owned_alpha_learn): the learned opaque draws - see native_renderer.h.
// ---------------------------------------------------------------------------------------------------------------

bool NativeRenderer::EnsureProbePool() {
  if (probe_pool_ != VK_NULL_HANDLE) return true;
  static bool failed = false;
  if (failed) return false;
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkQueryPoolCreateInfo info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_OCCLUSION;
  info.queryCount = kFrameSlots * kProbesPerFrame * OwnedLayers();  // (Session 94: a query inside a multiview pass takes one slot per view)
  if (vulkan_device->functions().vkCreateQueryPool(vulkan_device->device(), &info, nullptr, &probe_pool_) != VK_SUCCESS) {
    probe_pool_ = VK_NULL_HANDLE;
    failed = true;
    REXGPU_WARN("[native][owned] no occlusion query pool for the alpha probes: gpu_native_owned_alpha_learn stays off");
    return false;
  }
  return true;
}

bool NativeRenderer::OwnedAlphaCandidate(const Draw& draw) const {
  return owned_frame_on_ && probe_slot_ok_ && draw.indexed && !draw.own_skip_call && draw.blend_control == 0x07060706u &&
         (draw.color_write_mask & 0xF) == 0xF && !draw.raster.stencil && draw.pixel_shader != VK_NULL_HANDLE &&
         (draw.pixel_ext & SpirvShaderTranslator::kNativeExtSwap) != 0 && draw.instance_count == 1 &&
         draw.xfb_buffer == VK_NULL_HANDLE && !draw.fast_clear && !draw.restore && !draw.memexport &&
         !draw.occlusion_query && draw.pipeline != VK_NULL_HANDLE;
}

uint64_t NativeRenderer::OwnedAlphaKeyOf(const Draw& draw) const {
  // The geometry's identity: the shaders and the index range in the shared memory (a static mesh keeps its address).
  uint64_t h = draw.vertex_shader_hash * 0x9E3779B97F4A7C15ull;
  const auto mix = [&](uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h *= 0xBF58476D1CE4E5B9ull;
  };
  mix(draw.pixel_shader_hash);
  // (Session 95: an attribute draw's index_offset / index_buffer are the pool's and move on every remake: its guest index address is the identity.)
  mix(draw.attribute ? uint64_t(draw.guest_index_base) : uint64_t(draw.index_offset));
  mix(uint64_t(draw.vertex_count));
  mix(draw.attribute ? uint64_t(1) : uint64_t(reinterpret_cast<uintptr_t>(draw.index_buffer)));
  // The textures bound (the fetch constants' base addresses; not the constants: the world matrix changes every frame): a
  // texture change behind a proven key starts it unproven.
  mix(draw.texture_key);
  return h ? h : 1;
}

VkPipeline NativeRenderer::OwnedAlphaPipeline(const Draw& draw) {
  if (!REXCVAR_GET(gpu_native_owned_alpha_learn) || !OwnedAlphaCandidate(draw)) return draw.pipeline;
  ++owned_frame_stats_.alpha_candidates;
  const auto it = owned_alpha_keys_.find(OwnedAlphaKeyOf(draw));
  if (it == owned_alpha_keys_.end() || !it->second.proven) return draw.pipeline;
  // (A proven key whose re-probe is overdue - the per-frame probe budget ran out before it - is drawn blended again.)
  if (frame_ >= it->second.last_probe + 2 * uint64_t(REXCVAR_GET(gpu_native_owned_alpha_learn_interval))) return draw.pipeline;
  Draw unblended = draw;
  unblended.blend_control = 0x00010001u;  // (one, zero, add: no blending)
  unblended.no_store = true;
  bool pending = false;
  const VkPipeline pipeline = GetPipeline(unblended, &pending);  // (compiled on the workers: blended until it is made)
  if (pending || pipeline == VK_NULL_HANDLE) return draw.pipeline;
  ++owned_frame_stats_.alpha_proven;
  return pipeline;
}

void NativeRenderer::OwnedAlphaProbe(const Draw& draw) {
  if (!REXCVAR_GET(gpu_native_owned_alpha_learn) || !OwnedAlphaCandidate(draw)) return;
  const uint64_t key = OwnedAlphaKeyOf(draw);
  OwnedAlphaKey& k = owned_alpha_keys_[key];
  k.last_seen = frame_;
  if (probe_issued_ >= kProbesPerFrame) return;
  const uint64_t interval = uint64_t(REXCVAR_GET(gpu_native_owned_alpha_learn_interval));
  // (A proven key is probed again every interval; a key that failed a probe backs off to the same pace.)
  if ((k.proven || k.fails) && frame_ < k.last_probe + interval) return;
  // The probe module: the swapped shader's translation with the probe patch (made once per shader and modification).
  uint64_t mkey = draw.pixel_shader_hash * 0x9E3779B97F4A7C15ull;
  mkey ^= draw.pixel_modification + 0x9E3779B97F4A7C15ull + (mkey << 6) + (mkey >> 2);
  mkey ^= uint64_t(draw.pixel_ext) << 40 ^ uint64_t(draw.ps_quantize) << 32 ^ uint64_t(draw.ps_twin) << 33 ^
          uint64_t(draw.ps_float24) << 34;
  auto found = owned_probe_modules_.find(mkey);
  if (found == owned_probe_modules_.end()) {
    VkShaderModule module = VK_NULL_HANDLE;
    VulkanPipelineCache& cache = command_processor_.NativePipelineCache();
    if (VulkanShader* shader = cache.FindShader(draw.pixel_shader_hash)) {
      if (auto* translation = cache.GetNativePixelShaderTranslation(
              *shader, SpirvShaderTranslator::Modification(draw.pixel_modification), draw.ps_float24, draw.ps_quantize,
              draw.ps_twin, draw.pixel_ext | SpirvShaderTranslator::kNativeExtAlphaProbe)) {
        module = translation->shader_module();
      }
    }
    found = owned_probe_modules_.emplace(mkey, module).first;
  }
  if (found->second == VK_NULL_HANDLE) return;
  Draw probe = draw;
  probe.pixel_shader = found->second;
  probe.color_write_mask = 0;
  probe.blend_control = 0x00010001u;
  probe.raster.depth_test = false;
  probe.raster.depth_write = false;
  probe.raster.depth_compare = VK_COMPARE_OP_ALWAYS;
  probe.raster.stencil = false;
  probe.no_store = true;
  bool pending = false;
  const VkPipeline pipeline = GetPipeline(probe, &pending);  // (compiled on the workers: probed once it is made)
  if (pending || pipeline == VK_NULL_HANDLE) return;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t query = (SlotOf(frame_) * kProbesPerFrame + probe_issued_) * OwnedLayers();  // (Session 94: its first view's slot; the picture is the first of the K)
  cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  owned_bound_.pipeline = pipeline;
  cb.CmdVkBeginQuery(probe_pool_, query, 0);
  cb.CmdVkDrawIndexed(draw.vertex_count, draw.instance_count, 0, draw.vertex_offset, 0);  // (Session 95: an attribute draw's buffers are still bound from the draw this probes; OwnedAlphaCandidate takes indexed draws only)
  cb.CmdVkEndQuery(probe_pool_, query);
  slots_[SlotOf(frame_)].alpha_frame = frame_;
  slots_[SlotOf(frame_)].alpha_probes.emplace_back(key, query);
  ++probe_issued_;
  k.last_probe = frame_;
  ++owned_frame_stats_.alpha_probes;
}

void NativeRenderer::OwnedAlphaMaintain() {
  // (A key whose draws are gone - a texture set or constants that changed for good - is dropped after 600 frames unseen, so the
  // table cannot grow; checked every 300 frames, in frames with no probes too.)
  if (frame_ < alpha_evict_frame_ + 300) return;
  alpha_evict_frame_ = frame_;
  for (auto it = owned_alpha_keys_.begin(); it != owned_alpha_keys_.end();) {
    it = (it->second.last_seen + 600 < frame_) ? owned_alpha_keys_.erase(it) : std::next(it);
  }
}

void NativeRenderer::OwnedAlphaRead(FrameSlot& slot) {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const size_t n = slot.alpha_probes.size();
  if (!n || probe_pool_ == VK_NULL_HANDLE) return;
  // (Session 94, gpu_d3d_decouple: K slots per probe, in order; the probe's answer is the last view's - the picture's.)
  const uint32_t views = OwnedLayers();
  std::vector<uint64_t> samples(n * views, 0);
  const uint32_t first = slot.alpha_probes.front().second;
  if (vulkan_device->functions().vkGetQueryPoolResults(vulkan_device->device(), probe_pool_, first, uint32_t(n * views),
                                                       n * views * sizeof(uint64_t), samples.data(), sizeof(uint64_t),
                                                       VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
    slot.alpha_probes.clear();
    return;
  }
  if (views > 1) {
    for (size_t i = 0; i < n; ++i) { uint64_t sum = 0; for (uint32_t k = 0; k < views; ++k) sum += samples[i * views + k]; samples[i] = sum; }  // (review: the K slots are summed - the split between views is implementation dependent)
  }
  for (size_t i = 0; i < n; ++i) {
    const auto it = owned_alpha_keys_.find(slot.alpha_probes[i].first);
    if (it == owned_alpha_keys_.end()) continue;
    OwnedAlphaKey& k = it->second;
    if (samples[i] == 0) {
      // (Two clean probes of different frames prove a key; the same key drawn twice in one frame is one probe.)
      if (k.last_clean != slot.alpha_frame) {
        k.last_clean = slot.alpha_frame;
        ++k.clean;
      }
      if (k.clean >= 2) k.proven = true;
    } else {
      if (k.proven) ++owned_frame_stats_.alpha_reverts;
      k.clean = 0;
      k.proven = false;
      ++k.fails;
    }
  }
  slot.alpha_probes.clear();
}

bool NativeRenderer::OwnedRecord(const Draw& draw, const int32_t rect[4]) {
  // A draw of the owned pass, as TruthRecordDraw records it: its pipeline, every set it binds, its viewport, its
  // rectangle as the scissor (our image's pixels are the surface's), its depth bias, blend constants and stencil
  // state - in our pass, begun (or resumed) first.
  // (A committed pass going on after its copy: not while a page is our image itself - this draw may sample it (map
  // B's blurs read map B): the pass ends, the surface remembered - its next copies are served from a snapshot.)
  OwnedPass& p = owned_pending_;
  if (p.committed && std::any_of(owned_pages_.begin(), owned_pages_.end(),
                                 [&](const OwnedPage& q) { return q.image == p.image; }) &&
      !(REXCVAR_GET(gpu_native_owned_frame_continue_copy)
            ? OwnedLeanGoOn(p, draw, rect)  // (Session 95: OwnedContinueCopy unless gpu_owned_copy_lean is on)
            : (REXCVAR_GET(gpu_native_owned_frame_detach_pages) && OwnedDetachPages(p)))) {
    OwnedNoteContinued(p);
    OwnedEndCommitted("a committed pass's draw samples its own page");
    if (owned_frame_on_) OwnedNoteRefusal(kOwnedReasonOther, "continued page");
    return false;
  }
  const OwnedImage& o = owned_images_[p.image];
  const uint64_t image_last_use = o.last_use;  // (before this draw's touch)
  OwnedTouch(p.image);
  // (Session 82: a pass that keeps depth has the image's own depth, made now; every other pass the scratch image of
  // the size, which nothing reads.)
  if (!p.started && p.depth && !OwnedEnsureDepth(p.image)) {
    // (Nothing of the pass was recorded: it never began - the truth path takes this draw, its clears' pass still on.)
    p.valid = false;
    if (owned_frame_on_) OwnedNoteRefusal(kOwnedReasonPool, "depth image");
    return false;
  }
  const VkImage depth_image = p.depth ? o.depth : owned_scratch_depths_[o.scratch].image;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  // Session 82 (P1 item 2, served bindings): the truth images this draw samples in place of a copy (TruthServeView,
  // from the texture request) are readable for as long as our pass is - made so before it begins, as the truth pass
  // does (truth_alias_read_owners_: restored to their attachment layouts when the pass ends). A draw that samples one
  // that is not readable yet ends our render pass (no barrier inside it) and resumes it below.
  const auto serve_unreadable = [&]() {
    return std::any_of(serve_pending_.begin(), serve_pending_.end(), [&](VkImage image) {
      return std::none_of(truth_alias_read_owners_.begin(), truth_alias_read_owners_.end(),
                          [&](const std::pair<VkImage, bool>& o2) { return o2.first == image; });
    });
  };
  if (p.open && serve_unreadable()) OwnedEndPass();
  if (!p.open) {
    const bool first = !p.started;
    if (!first) {
      // Resumed: out of the open truth pass and the command processor's (ours is a pass of its own).
      EndTruthPass();
      command_processor_.EndRenderPass();
      command_processor_.SubmitBarriers(true);
      ++truth_counts_.owned_resumes;
    }
    if (serve_unreadable()) {
      std::vector<VkImageMemoryBarrier> serve_barriers;
      for (const VkImage image : serve_pending_) {
        if (std::any_of(truth_alias_read_owners_.begin(), truth_alias_read_owners_.end(),
                        [&](const std::pair<VkImage, bool>& o2) { return o2.first == image; })) {
          continue;
        }
        serve_barriers.push_back(OwnedBarrier(image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                              VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        truth_alias_read_owners_.emplace_back(image, false);
      }
      cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                              nullptr, 0, nullptr, uint32_t(serve_barriers.size()), serve_barriers.data());
    }
    // At the start: the images' earlier reads (a sampler, a copy into the truth images) and writes done, their old
    // pixels not kept (from UNDEFINED). Resumed: our pass's own writes before the next draws' (same layouts).
    VkImageMemoryBarrier barriers[2];
    if (first) {
      // (Session 82: a loaded start - OwnedStartExtra - keeps the pixels it was given: from the attachment layout.)
      barriers[0] = OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT,
                                 p.loaded_start ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : kOwnedAnyWrite,
                                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 p.loaded_start ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      barriers[1] = OwnedBarrier(depth_image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,  // (its last pass's, review)
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
      owned_images_[p.image].graph_dropped = false;  // (Session 95: a new pass over the image)
      owned_images_[p.image].readable = false;
      if (p.depth) owned_images_[p.image].depth_read = false;  // (a new pass: its depth begins from UNDEFINED)
    } else {
      // (After a copy that left it readable: back from SHADER_READ_ONLY.)
      barriers[0] = OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 owned_images_[p.image].readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                  : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      owned_images_[p.image].readable = false;
      // (gpu_native_owned_depth_direct: a depth that pages sample is in SHADER_READ_ONLY; the resumed pass takes it back.)
      const bool depth_ro = p.depth && owned_images_[p.image].depth_read;
      if (depth_ro) owned_images_[p.image].depth_read = false;
      barriers[1] = OwnedBarrier(depth_image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                                 depth_ro ? 0 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                 depth_ro ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                          : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }
    owned_bound_ = OwnedBound();  // (a render pass of ours begins: nothing is known to be bound)
    // (gpu_native_owned_pass_barrier_lean: a new pass that keeps its own depth, in an image whose last recorded use the GPU has
    // finished - the transition from UNDEFINED has nothing in flight to wait for. The scratch depth image is shared by the passes
    // that keep none, so those keep the full barrier.)
    // (Session 95, gpu_tiler_sync_control: the lean start for EVERY pass start, with nothing waited for - a deliberate hazard, the positive control of the sync arm.)
    const bool lean_start = (first && p.depth && !p.loaded_start && REXCVAR_GET(gpu_native_owned_pass_barrier_lean) &&
                             image_last_use <= command_processor_.GetCompletedSubmission()) ||
                            (first && REXCVAR_GET(gpu_tiler_sync_control));
    if (lean_start) {
      barriers[0].srcAccessMask = 0;
      barriers[1].srcAccessMask = 0;
      ++owned_frame_stats_.lean_starts;
    }
    cb.CmdVkPipelineBarrier(lean_start ? VkPipelineStageFlags(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
                                       : VkPipelineStageFlags(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT),
                            kOwnedAttachmentStages, 0, 0, nullptr, 0, nullptr, 2, barriers);
    VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    const uint32_t variant = first ? p.start : uint32_t(kOwnedPassResume);
    // (Session 93, gpu_owned_tiler_passes: the load / store variant - the clear values below are those of `variant`, which the dead-depth twin shares.)
    const uint32_t used_variant = OwnedTilerVariant(p.image, variant, p.depth, first, &p.key);
    begin_info.renderPass = owned_render_passes_[used_variant][o.kind];
    begin_info.framebuffer = p.depth ? o.framebuffer_depth : o.framebuffer;
    begin_info.renderArea = {{0, 0}, {o.width, o.height}};
    begin_info.clearValueCount = variant == kOwnedPassClearBoth ? 2 : variant == kOwnedPassClearColor ? 1 : 0;
    begin_info.pClearValues = begin_info.clearValueCount ? p.clear_values : nullptr;
    VkClearValue poison_clears[2];  // (Session 95, gpu_tiler_poison)
    VkClearValue dc_values[2];  // (Session 95, Lane E: every variant has a CLEAR slot now, so both values are given; it wins over the poison, which then has no DONT_CARE left to fill)
    if (OwnedDcMode()) {
      OwnedDontCareValues(dc_values, variant == kOwnedPassClearBoth || variant == kOwnedPassClearColor ? &p.clear_values[0] : nullptr,
                          variant == kOwnedPassClearBoth ? &p.clear_values[1] : nullptr);
      begin_info.clearValueCount = 2;
      begin_info.pClearValues = dc_values;
    } else if (OwnedPoisonOn()) {
      begin_info.clearValueCount = OwnedPoisonClears(used_variant, p.clear_values, poison_clears, o.width, o.height, first ? 0 : 1, first && p.depth);
      begin_info.pClearValues = begin_info.clearValueCount ? poison_clears : nullptr;
    }
    p.timer_segment = OwnedTimerBegin("pass", o.width, o.height);  // (Session 82, P1: the pass log)
    p.timer_draws = 0;
    cb.CmdVkBeginRenderPass(&begin_info, VK_SUBPASS_CONTENTS_INLINE);
    p.open = true;
    p.started = true;
  }
  // (Session 82, CPU: only the state that differs from what this render pass of ours has bound - the truth path's
  // dedupe, for the owned pass. gpu_native_owned_frame_dedupe off: every command, as before.)
  OwnedRsNopBegin();  // (Session 95)
  OwnedBound& ob = owned_bound_;
  own_crowd_sig_valid_ = false;  // Session 93 (the crowd merge: set below for a crowd run's first member)
  const bool dedupe = REXCVAR_GET(gpu_native_owned_frame_dedupe) && ob.valid;
  uint32_t cmds = 0, skipped = 0;
  const auto emit = [&](bool needed) {
    ++cmds;
    if (!needed) ++skipped;
    return needed;
  };
  const VkPipeline use_pipeline = OwnedAlphaPipeline(draw);  // (Session 82: the unblended one for a proven draw)
  if (emit(!dedupe || ob.pipeline != use_pipeline)) cb.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, use_pipeline);
  ob.pipeline = use_pipeline;
  const bool same_layout = dedupe && ob.layout == draw.layout;
  const uint32_t dyn_count = draw.dynamic_offset_count;
  // (Session 95, r3gpu, gpu_native_skin_bind_lean: a skin-once side draw reads a few constants, and the last one's slot holds the same: the slot stays bound, the draw needs no descriptor bind.)
  const bool skin_same = draw.skin_side_key && same_layout && ob.skin_side_key == draw.skin_side_key && ob.dynamic_count == dyn_count;
  if (REXCVAR_GET(gpu_native_skin_bind_lean)) {  // ([skin1] proof line, Session 95 review: keyed vs seen vs kept, so "never keyed" reads apart from "keyed, never kept"; the first draw at once, then every 60000)
    static uint64_t seen = 0, keyed = 0, kept = 0;
    ++seen;
    keyed += draw.skin_side_key ? 1 : 0;
    kept += skin_same ? 1 : 0;
    if (seen == 1 || seen % 60000 == 0) {
      REXGPU_INFO("[skin1] bind lean: {} of {} draws reaching the pass were keyed side draws, {} kept the last draw's constant slots (waited for a pipeline: {})", keyed, seen, kept, truth_counts_.waited);
    }
  }
  for (uint32_t s = 0; s < SpirvShaderTranslator::kDescriptorSetCount; ++s) {
    if (!(draw.sets_bound & (1u << s)) || draw.sets[s] == VK_NULL_HANDLE) continue;
    const bool constants = s == SpirvShaderTranslator::kDescriptorSetConstants && dyn_count;
    if (skin_same && ob.sets[s] == draw.sets[s]) {
      ++skipped;
      ++cmds;
      continue;
    }
    const bool same = same_layout && ob.sets[s] == draw.sets[s] &&
                      (!constants || (ob.dynamic_count == dyn_count &&
                                      !std::memcmp(ob.dynamic, draw.dynamic_offsets, dyn_count * sizeof(uint32_t))));
    if (emit(!same)) {
      cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, draw.layout, s, 1, &draw.sets[s],
                                 constants ? dyn_count : 0, constants ? draw.dynamic_offsets : nullptr);
    }
    ob.sets[s] = draw.sets[s];
    if (constants) {
      ob.dynamic_count = dyn_count;
      std::memcpy(ob.dynamic, draw.dynamic_offsets, dyn_count * sizeof(uint32_t));
    }
  }
  ob.layout = draw.layout;
  ob.skin_side_key = draw.skin_side_key;
  VkViewport viewport = draw.viewport;
  if (p.half) {  // (look lever: the pass's image is half the box; Session 94: or the ratio's fraction of it)
    viewport.x *= p.ratio_x, viewport.y *= p.ratio_y, viewport.width *= p.ratio_x, viewport.height *= p.ratio_y;
  }
  VkRect2D scissor;
  if (p.dres) {
    // (Session 94: floor for the near edge, ceil for the far one - the pixel-centre rule of native_truth.cpp's pass_rect - clamped to the image.)
    const int32_t iw = int32_t(o.width), ih = int32_t(o.height);
    const int32_t x0 = std::clamp(int32_t(std::floor(double(rect[0]) * p.ratio_x)), 0, iw);
    const int32_t y0 = std::clamp(int32_t(std::floor(double(rect[1]) * p.ratio_y)), 0, ih);
    const int32_t x1 = std::clamp(int32_t(std::ceil(double(rect[2]) * p.ratio_x)), x0, iw);
    const int32_t y1 = std::clamp(int32_t(std::ceil(double(rect[3]) * p.ratio_y)), y0, ih);
    scissor = VkRect2D{{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
  } else if (p.half) {
    scissor = VkRect2D{{rect[0] / 2, rect[1] / 2}, {uint32_t(rect[2] - rect[0] + 1) / 2, uint32_t(rect[3] - rect[1] + 1) / 2}};
  } else {
    scissor = VkRect2D{{rect[0], rect[1]}, {uint32_t(rect[2] - rect[0]), uint32_t(rect[3] - rect[1])}};
  }
  // (Session 84: the viewport and scissor through the command processor's tracking, as TruthRecordDraw - it records the
  // guest draw's own before the claim, into this command buffer. On the host render target path that viewport is the
  // halved 20e4 range and ours the guest's, so a private "already bound" skipped ours: every owned scene draw but the
  // crowd's (set directly) drew in 0..0.5 and the crowd covered the wrestlers (RenderDoc, run 5780).)
  if (dedupe) {
    command_processor_.SetViewport(viewport);
    command_processor_.SetScissor(scissor);
    cmds += 2;
  } else {
    emit(true);
    cb.CmdVkSetViewport(0, 1, &viewport);
    emit(true);
    cb.CmdVkSetScissor(0, 1, &scissor);
  }
  // (Session 95: a scaled pass's pixels are 1 / ratio bigger, so a slope per pixel is 1 / ratio steeper: the factor scaled, as the draw twins did. Here and not at the typed draw:
  // only the pass knows it is scaled, the ratio changes per frame, and the draw's memo stays valid.)
  const float bias_slope = p.dres ? draw.raster.depth_bias_slope * std::max(p.ratio_x, p.ratio_y) : draw.raster.depth_bias_slope;
  if (emit(!dedupe || ob.bias[0] != draw.raster.depth_bias_constant || ob.bias[1] != bias_slope)) {
    cb.CmdVkSetDepthBias(draw.raster.depth_bias_constant, 0.0f, bias_slope);
  }
  ob.bias[0] = draw.raster.depth_bias_constant;
  ob.bias[1] = bias_slope;
  if (emit(!dedupe || std::memcmp(ob.blend, draw.blend_constants, sizeof(ob.blend)))) {
    cb.CmdVkSetBlendConstants(draw.blend_constants);
  }
  std::memcpy(ob.blend, draw.blend_constants, sizeof(ob.blend));
  if (draw.raster.stencil) {  // (the guest's reference and masks, per face)
    for (uint32_t f = 0; f < 2; ++f) {
      const VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
      const bool known = dedupe && ob.stencil_valid;
      if (emit(!known || ob.stencil[f][0] != draw.stencil_refs[f])) cb.CmdVkSetStencilReference(face, draw.stencil_refs[f]);
      if (emit(!known || ob.stencil[f][1] != draw.stencil_read_masks[f])) {
        cb.CmdVkSetStencilCompareMask(face, draw.stencil_read_masks[f]);
      }
      if (emit(!known || ob.stencil[f][2] != draw.stencil_write_masks[f])) {
        cb.CmdVkSetStencilWriteMask(face, draw.stencil_write_masks[f]);
      }
      ob.stencil[f][0] = draw.stencil_refs[f];
      ob.stencil[f][1] = draw.stencil_read_masks[f];
      ob.stencil[f][2] = draw.stencil_write_masks[f];
    }
    ob.stencil_valid = true;
  }
  OwnedGraphReadsLand(draw);  // (Session 95, round 7: what this draw asked for is read here, in this pass; a repeat reads what its base asked for)
  // (A crowd run's first member: the run that follows records its own pipeline, sets and state into this pass.)
  ob.valid = !draw.own_skip_call;
  if (!ob.valid) ob.stencil_valid = false;
  owned_frame_stats_.rec_state_cmds += cmds;
  owned_frame_stats_.rec_state_skipped += skipped;
  if (draw.own_skip_call) {
    // Session 82 (P1 item 1): a crowd run's first member - its state only, as the truth path records it; the run is
    // drawn by ours right after (RecordOwnCrowd, into this pass), or by the game's own draw if ours is not taken.
    owned_crowd_first_ = draw.indexed && draw.instance_count == 1 &&
                         !(draw.screen_space && draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1]) &&
                         !(draw.bounds[2] > draw.bounds[0] && draw.bounds[3] > draw.bounds[1]);
    truth_last_vs_hash_ = draw.vertex_shader_hash;
    truth_last_index_.indexed = draw.indexed;
    truth_last_index_.buffer = draw.index_buffer;
    truth_last_index_.offset = draw.index_offset;
    truth_last_index_.type = draw.index_type;
    truth_last_index_.count = draw.vertex_count;
    truth_last_index_.attribute = draw.attribute;  // (Session 95: never, by the plan's crowd gate; own crowd declines it all the same)
    OwnCrowdNoteSig(viewport, scissor, draw.raster.depth_bias_constant, bias_slope, draw.blend_constants,
                    draw.raster.stencil, draw.stencil_refs, draw.stencil_read_masks, draw.stencil_write_masks);
  } else if (draw.indexed) {
    if (draw.diag_id) [[unlikely]] cb.CmdDiagMark(diag::kMarkNextDraw, draw.diag_id, draw.diag_aux);  // (Session 95, the diagnostic run)
    BindDrawInputs(cb, draw);  // (Session 95)
    cb.CmdVkBindIndexBuffer(draw.index_buffer, draw.index_offset, draw.index_type);
    cb.CmdVkDrawIndexed(draw.vertex_count, draw.instance_count, 0, draw.vertex_offset, 0);
  } else {
    if (draw.diag_id) [[unlikely]] cb.CmdDiagMark(diag::kMarkNextDraw, draw.diag_id, draw.diag_aux);
    BindDrawInputs(cb, draw);  // (Session 95)
    cb.CmdVkDraw(draw.vertex_count, draw.instance_count, uint32_t(draw.vertex_offset), 0);
  }
  OwnedAlphaProbe(draw);  // (Session 82: a candidate draw's probe, right after the draw)
  // (Session 82, P2: a defined-start pass keeps its draws past the commit - it owes the EDRAM nothing, so a later copy of it
  // the owned frame cannot serve is exact by replaying them, OwnedReplayPending; the others are never replayed once copied.)
  // (Session 92, M2a: under the latch nothing is ever replayed - no draw is kept, no served image, no sampled-image list.)
  const bool keeps_draws = (!p.committed || p.edram_free) && !p.carried_defined && !OwnedOnly();  // (Session 90: a carried defined start has no draws to replay - a partial list would be wrong)
  p.last_tick = ++owned_tick_;
  if ((draw.color_write_mask & 0xF) != 0) p.rec_colour_gone = false;  // (a pass taken up from a record that draws on: what it holds is current again)
  if (draw.truth_depth >= 0 && draw.raster.depth_write) p.rec_depth_gone = false;
  if (draw.truth_depth >= 0 && (draw.raster.depth_write || draw.raster.stencil)) p.depth_tick = p.last_tick;
  if (keeps_draws && !serve_pending_.empty()) p.serves.emplace_back(p.draws.size(), serve_pending_);
  // (What this draw was given for its served bindings is spent - TruthRecordDraw's scope drops it at its end.)
  serve_pending_.clear();
  serve_rename_.valid = false;
  if (keeps_draws) {
    p.draws.push_back(draw);  // (replayed on a loss)
    for (const int16_t image : owned_sampled_images_) {
      if (std::find(p.kept_sampled.begin(), p.kept_sampled.end(), image) == p.kept_sampled.end()) p.kept_sampled.push_back(image);
    }
  }
  ++p.draws_since_color;
  ++p.draws_since_depth;
  OwnedRsPassNote(p, draw, rect);  // (Session 95, watch only)
  if (p.edram_free) {  // (Session 82, P2, G4: the drawn bounding box, kept past the commit)
    uint32_t sc_x = 1, sc_y = 1;
    command_processor_.NativeResolutionScale(sc_x, sc_y);
    const int32_t gx0 = rect[0] / int32_t(sc_x), gy0 = rect[1] / int32_t(sc_y);
    const int32_t gx1 = (rect[2] + int32_t(sc_x) - 1) / int32_t(sc_x), gy1 = (rect[3] + int32_t(sc_y) - 1) / int32_t(sc_y);
    if (p.drawn[2] <= p.drawn[0]) {
      p.drawn[0] = gx0, p.drawn[1] = gy0, p.drawn[2] = gx1, p.drawn[3] = gy1;
    } else {
      p.drawn[0] = std::min(p.drawn[0], gx0), p.drawn[1] = std::min(p.drawn[1], gy0);
      p.drawn[2] = std::max(p.drawn[2], gx1), p.drawn[3] = std::max(p.drawn[3], gy1);
    }
  }
  p.draws_depth = p.draws_depth || (draw.truth_depth >= 0 && draw.raster.depth_write);
  if (pass_log_frames_ > 0) OwnedPassCensusNote(draw.vertex_shader_hash, draw.pixel_shader_hash, draw.vertex_count, draw.attribute, draw.bounds);  // (Session 95, r3gpu)
  if (pass_log_frames_ > 0) {  // (Session 95, r3gpu: the wrestler materials' float constants, one line per material a window: which terms are 0)
    const uint32_t top = uint32_t(draw.pixel_shader_hash >> 32);
    static uint64_t last[8] = {};
    static const uint32_t kList[5] = {0xB1F58C0Au, 0x1D20CB5Du, 0x91BF021Cu, 0xF75AE782u, 0x925085B6u};
    for (uint32_t i = 0; i < 5; ++i) {
      if (top != kList[i] || (last[i] && frame_ - last[i] < 200)) continue;
      last[i] = frame_;
      size_t size = 0;
      const uint8_t* bytes = command_processor_.OwnedConstantsBytes(SpirvShaderTranslator::kConstantBufferFloatPixel, size);
      if (!bytes) continue;
      std::string line;
      const size_t count = std::min<size_t>(size / 4, 48 * 4);
      for (size_t k = 0; k < count; ++k) {
        float f;
        std::memcpy(&f, bytes + 4 * k, 4);
        line += fmt::format("{}{:.5g}", (k & 3) ? "," : " | ", f);
      }
      REXGPU_INFO("[native][passconst] frame {} ps {:016X}:{}", frame_, draw.pixel_shader_hash, line);
    }
  }
  ++p.timer_draws;
  p.draws_stencil = p.draws_stencil || (draw.truth_depth >= 0 && draw.raster.stencil);
  // (Going on after its copy: its box is dirty again - our image is the surface's there - and the surface noted.)
  if (p.committed) {
    OwnedMarkDirty(false);
    OwnedNoteContinued(p);
    OwnedLeanNoteDraw(p, draw, rect);  // (Session 95)
    OwnedRsNote(p, draw, rect);  // (Session 95)
    if (p.graph_snap) {  // (Session 95: the predicted snapshot was right)
      p.graph_snap = false;
      OwnedGraphCount(kGraphCountSnapshotContinued);
    }
    ++truth_counts_.owned_continued;
  }
  const bool by_ps = OwnedWants(draw.pixel_shader_hash);
  const std::pair<uint64_t, bool> key = {by_ps ? draw.pixel_shader_hash : draw.vertex_shader_hash, !by_ps};
  if (std::find(p.producers.begin(), p.producers.end(), key) == p.producers.end()) p.producers.push_back(key);
  // (gpu_native_truth_skip_restores: the images it would have gone into changed.)
  for (const int16_t slot : {p.color_slot, draw.truth_depth >= 0 ? p.depth_slot : int16_t(-1)}) {
    if (slot < 0) continue;
    if (size_t(slot) >= truth_image_writes_.size()) truth_image_writes_.resize(size_t(slot) + 1, 0);
    ++truth_image_writes_[slot];
  }
  // What the next truth draw may take as bound is not (as after own crowd's draw); nothing merges into this one; the
  // command processor's dynamic state is not what it recorded.
  truth_bound_valid_ = false;
  truth_bound_pipeline_ = VK_NULL_HANDLE;
  truth_bound_index_buffer_ = VK_NULL_HANDLE;
  truth_bound_stencil_valid_ = false;
  truth_foreign_bind_ = true;
  // Session 94 (gpu_native_instance_world): the owned draw of one of OUR instanceable arena vertex shaders stays the last recorded draw (nothing was recorded after
  // it: the command stream ends with its draw command) - the repeat merge (RepeatDrawFast) may add an instance to it. Not a crowd run's first member, not a draw
  // cut to its proven coverage / vertex box (an instance may lie elsewhere), not under replay (a kept draw is replayed without its instances). Off: false, as ever.
  truth_last_recorded_ = OwnedOnly() && !draw.own_skip_call && draw.merge_stride != 0 && draw.instance_count == 1 && !draw.fast_clear &&
                         !(draw.screen_space && draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1]) &&
                         !(draw.bounds[2] > draw.bounds[0] && draw.bounds[3] > draw.bounds[1]) && shader_swap::VertexInstanceable(draw.vertex_shader_hash);
  drew_this_frame_ = true;
  command_processor_.NativeForgetState();
  ++truth_counts_.owned_draws;
  ++owned_draw_serial_;
  if (p.census >= 0 && size_t(p.census) < owned_frame_passes_.size()) ++owned_frame_passes_[p.census].draws;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned draw: vs {:016X} ps {:016X} into our image {} ({}x{}) rect {},{}-{},{}, "
                "surface base {} pitch {}, draw {} of the pass",
                frame_, draw.vertex_shader_hash, draw.pixel_shader_hash, p.image, o.width, o.height, rect[0], rect[1],
                rect[2], rect[3], p.base, p.pitch_tiles, p.draws.size());
  }
  return true;
}

bool NativeRenderer::OwnedTakeDraw(const Draw& draw) {
  if (owned_flushing_held_) return false;
  if (OwnedOnly() && OwnedPendBitOn()) {  // (Session 92, M4f-B: a newer clear of a slot replaces its pending clear; a draw of a slot with a pended colour clear comes after it: the clear is held again first)
    if (draw.fast_clear) OwnedPendClearEraseFor(draw);
    else OwnedPendClearMaterialize(draw);
  }
  if (!owned_depth_clears_.empty() || !owned_depth_dropped_.empty()) OwnedDepthClearsConsult(draw);  // (Session 87)
  const uint64_t pend_tick0 = owned_tick_;  // (review fix 2)
  const bool taken = OwnedTakeDrawImpl(draw);
  if (taken && !draw.fast_clear && draw.truth_color >= 0 && OwnedOnly() && OwnedPendBitOn()) OwnedPendClearDropNewerThan(draw.truth_color, pend_tick0);  // (a held clear of this draw's own colour slot dropped inside it: the draw is already recorded)
  OwnedRsAfterDraw(draw);  // (Session 95)
  owned_sampled_images_.clear();
  OwnedGraphReadsClose();  // (Session 95, round 7)
  // (A draw the truth path records now: the clears held before it go in first.)
  if (!taken && !owned_held_.draws.empty()) {
    // (chains: a draw of another surface does not need the clears run in the truth path - they become a pass of ours.)
    const bool apart = OwnedChains() && draw.truth_color != owned_held_.color &&
                       (draw.truth_depth < 0 || draw.truth_depth != owned_held_.depth);
    OwnedFlushHeld(apart);
  }
  return taken;
}

bool NativeRenderer::OwnedDepthClearInPass(const Draw& draw) {
  // A depth-only clear (colour mask 0) of a depth surface an owned pass has open: the depth - and the stencil when the clear writes it whole
  // - cleared inside that pass by vkCmdClearAttachments, in order with its draws (the restore draw's stencil clear does the same). The clear is
  // never held as a pipeline-less draw and never flushed through the truth path. Values from the draw's own constants (NativeClearDepth: the
  // vertex shader's constant depth; the stencil reference). Not whole / not constant / another surface -> false, the old path.
  OwnedFrameStats& st = owned_frame_stats_;
  if (!owned_frame_on_ || owned_replaying_ || !draw.fast_clear || draw.truth_depth < 0 || (draw.color_write_mask & 0xF) != 0) return false;
  OwnedPass& p = owned_pending_;
  if (!p.valid || p.committed) {
    ++st.depth_clear_no_pass;
    return false;
  }
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  const bool rec_depth_only = OwnedOnly() && OwnedPendBitOn() && draw.d3d_record && draw.truth_color < 0;  // (review fix 4a: a record depth-only clear has no colour base - matched by depth slot + pitch)
  if (p.depth_slot < 0 || p.depth_slot != draw.truth_depth || (!rec_depth_only && p.base != (draw.color_info & 0xFFF)) || p.pitch_tiles != pitch ||
      p.kind != draw.truth_class || draw.truth_sx != 1 || draw.truth_sy != 1) {
    ++st.depth_clear_other_depth;
    return false;
  }
  const auto stencil_whole = [](uint32_t ops) {
    return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS);
  };
  const bool depth_whole = draw.clear_depth >= 0.0f && draw.raster.depth_test && draw.raster.depth_write &&
                           draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS && draw.raster.depth_bias_constant == 0.0f &&
                           draw.raster.depth_bias_slope == 0.0f;
  const bool stencil_used = draw.raster.stencil;
  const bool stencil_ok = !stencil_used ||
                          (stencil_whole(draw.raster.stencil_ops & 0xFFF) && stencil_whole((draw.raster.stencil_ops >> 12) & 0xFFF) &&
                           draw.stencil_write_masks[0] == 0xFF && draw.stencil_write_masks[1] == 0xFF &&
                           draw.stencil_refs[0] == draw.stencil_refs[1]);
  if (!depth_whole || !stencil_ok) {
    ++st.depth_clear_not_whole;
    return false;
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  rect[0] = std::clamp(rect[0], 0, int32_t(p.host_width));
  rect[1] = std::clamp(rect[1], 0, int32_t(p.host_height));
  rect[2] = std::clamp(rect[2], rect[0], int32_t(p.host_width));
  rect[3] = std::clamp(rect[3], rect[1], int32_t(p.host_height));
  if (rect[2] <= rect[0] || rect[3] <= rect[1]) {
    ++st.depth_clear_not_whole;
    return false;
  }
  if (!p.open && !OwnedResumeNow()) {
    // (The pass's render pass is not open - ended for a sampled binding, a copy ... - and begun again here, so the clear lands in order
    // with its draws and is never lost to a commit or a copy that comes before the next draw; review F4b.)
    ++st.depth_clear_not_open;
    return false;
  }
  VkClearAttachment clear = {};
  clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | (stencil_used ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
  clear.clearValue.depthStencil = {draw.clear_depth, stencil_used ? uint32_t(draw.stencil_refs[0]) : 0u};
  VkClearRect clear_rect = {{{rect[0], rect[1]}, {uint32_t(rect[2] - rect[0]), uint32_t(rect[3] - rect[1])}}, 0, 1};
  if (p.dres) {  // (Session 94: the pass image is a fraction of the box - floor / ceil as the draws' scissors)
    const int32_t iw = int32_t(owned_images_[p.image].width), ih = int32_t(owned_images_[p.image].height);
    const int32_t x0 = std::clamp(int32_t(std::floor(double(rect[0]) * p.ratio_x)), 0, iw), y0 = std::clamp(int32_t(std::floor(double(rect[1]) * p.ratio_y)), 0, ih);
    const int32_t x1 = std::clamp(int32_t(std::ceil(double(rect[2]) * p.ratio_x)), x0, iw), y1 = std::clamp(int32_t(std::ceil(double(rect[3]) * p.ratio_y)), y0, ih);
    clear_rect.rect = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
  }
  command_processor_.deferred_command_buffer().CmdVkClearAttachments(1, &clear, 1, &clear_rect);
  p.draws_depth = true;
  p.draws_stencil = p.draws_stencil || stencil_used;
  if (!OwnedOnly()) p.draws.push_back(draw);  // (review F4a: a replay of the pass needs the clear too; Session 92, M2a: none under the latch)
  p.last_tick = p.depth_tick = ++owned_tick_;
  ++st.depth_clear_taken;
  ++truth_counts_.owned_clears_dropped;  // (a clear our pass absorbed: counted with the dropped ones)
  return true;
}

bool NativeRenderer::OwnedDepthClearPend(const Draw& d) {
  // A depth(-stencil)-only clear (colour mask 0) that no owned pass takes (OwnedDepthClearInPass declined: no pass / another depth surface). Whole and
  // exact only (the depth through ALWAYS, written whole to the draw's constant), with a pipeline to replay it by. A colour clear held for the same depth
  // surface makes the order matter: not pended then.
  if (!owned_frame_on_ || owned_replaying_ || !OwnedChains() || !d.fast_clear ||
      (d.pipeline == VK_NULL_HANDLE && !(OwnedOnly() && OwnedPendBitOn())) || d.truth_depth < 0 || (d.color_write_mask & 0xF) != 0 || d.clear_depth < 0.0f ||  // (Session 92, M4f-B: the pipeline is never used under the latch - the flush below pends the value)
      !(d.raster.depth_test && d.raster.depth_write && d.raster.depth_compare == VK_COMPARE_OP_ALWAYS) ||
      d.raster.depth_bias_constant != 0.0f || d.raster.depth_bias_slope != 0.0f) {
    return false;
  }
  if (!owned_held_.draws.empty() && owned_held_.depth == d.truth_depth) return false;
  // (chains, review F4c: a pass that holds this depth surface - pending, or with dirty depth tiles to copy in - keeps it; the clear is not pended past it.)
  if (owned_pending_.valid && owned_pending_.depth && owned_pending_.depth_slot == d.truth_depth) return false;
  for (const OwnedDirty& e : owned_dirty_) {
    if (e.slot == d.truth_depth && (e.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) && e.tiles.any()) return false;
  }
  if (OwnedOnly()) {  // (Session 92, M2a: the surface's newest record plays the dirty entry's part - an image of ours holds this depth)
    for (const OwnedPass& q : owned_newest_) {
      if (q.depth && !q.rec_depth_gone && q.depth_slot == d.truth_depth) return false;
    }
    for (const OwnedPass& q : owned_parked_) {  // (review D8)
      if (q.depth && q.depth_slot == d.truth_depth) return false;
    }
  }
  if (owned_depth_clears_.size() >= 8) return false;
  OwnedPendingDepthClear e;
  e.draw = d;
  e.slot = d.truth_depth;
  e.depth_base = d.depth_info & 0xFFF;
  TruthDrawRect(d, e.rect);
  // (An earlier pending clear of this surface that this one covers is superseded.)
  for (size_t i = 0; i < owned_depth_clears_.size();) {
    const OwnedPendingDepthClear& o = owned_depth_clears_[i];
    if (o.slot == e.slot && e.rect[0] <= o.rect[0] && e.rect[1] <= o.rect[1] && e.rect[2] >= o.rect[2] && e.rect[3] >= o.rect[3]) {
      owned_depth_clears_.erase(owned_depth_clears_.begin() + i);
      ++owned_frame_stats_.dclear_superseded;
    } else {
      ++i;
    }
  }
  owned_depth_clears_.push_back(std::move(e));
  ++owned_frame_stats_.dclear_pending;
  serve_pending_.clear();
  serve_rename_.valid = false;
  return true;
}

void NativeRenderer::OwnedDepthClearsFlush(const char* why, int32_t only_slot, int64_t only_base) {
  if (owned_depth_clears_.empty()) return;
  std::vector<OwnedPendingDepthClear> take;
  for (size_t i = 0; i < owned_depth_clears_.size();) {
    const OwnedPendingDepthClear& e = owned_depth_clears_[i];
    if ((only_slot < 0 || e.slot == only_slot) && (only_base < 0 || int64_t(e.depth_base) == only_base)) {
      take.push_back(std::move(owned_depth_clears_[i]));
      owned_depth_clears_.erase(owned_depth_clears_.begin() + i);
    } else {
      ++i;
    }
  }
  if (take.empty()) return;
  if (OwnedOnly()) {  // (Session 92, M2a: no truth path to run them in - dropped, counted; "dclear_dropped" of the frame's end does the same)
    g_m2a.dclear_dropped += uint32_t(take.size());
    for (const OwnedPendingDepthClear& e : take) {
      OwnedMarkDepthStale(e.slot);  // (review D8: a dropped depth clear leaves what holds that depth stale)
      if (OwnedPendBitOn()) OwnedPendClearMake(e.draw, kPendDepthFlush, false);  // (Session 92, M4f-B: the clear is the slot's pending clear, not lost)
    }
    static uint32_t logged = 0;
    if (logged < 12) {
      ++logged;
      REXGPU_INFO("[native][owned-only] frame {}: {} pending depth-only clear(s) dropped ({}): depth slot {} vs {:016X} ps {:016X}", frame_, take.size(), why,
                  take[0].slot, take[0].draw.vertex_shader_hash, take[0].draw.pixel_shader_hash);
    }
    return;
  }
  // (As OwnedFlushHeld: what the draw being recorded was given for its served bindings is its own, kept across.)
  std::vector<VkImage> serve_pending;
  serve_pending.swap(serve_pending_);
  const ServeRename serve_rename = serve_rename_;
  serve_rename_.valid = false;
  const bool was_flushing = owned_flushing_held_;
  owned_flushing_held_ = true;
  for (const OwnedPendingDepthClear& e : take) {
    truth_foreign_bind_ = true;
    truth_bound_valid_ = false;
    if (owned_frame_on_ && owned_frame_pending_reason_ == kOwnedReasonNone)
      OwnedNoteRefusal(kOwnedReasonStart, fmt::format("depth-only clear run in the truth path: {}", why));
    TruthRecordDraw(e.draw);
    ++owned_frame_stats_.dclear_flushed;
  }
  truth_foreign_bind_ = true;
  truth_bound_valid_ = false;
  owned_flushing_held_ = was_flushing;
  serve_pending_.swap(serve_pending);
  serve_rename_ = serve_rename;
  command_processor_.NativeForgetState();
}

void NativeRenderer::OwnedDepthClearsConsult(const Draw& draw) {
  // (A draw about to be taken or recorded.) One that uses the depth surface of a pending clear runs it first; one that uses a depth surface whose
  // clear was dropped, with no clear since, is logged - the assumption "nothing reads it" was wrong there.
  const bool uses_depth = draw.truth_depth >= 0 && (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil) && !draw.fast_clear;
  if (!uses_depth) return;
  OwnedDepthClearsFlush("a draw uses that depth surface", draw.truth_depth);
  for (auto it = owned_depth_dropped_.begin(); it != owned_depth_dropped_.end(); ++it) {
    if (it->first != draw.truth_depth) continue;
    ++owned_frame_stats_.dclear_read_after_drop;
    static uint32_t logged = 0;
    if (logged < 12) {
      ++logged;
      REXGPU_WARN("[native][owned] frame {}: a draw (vs {:016X} ps {:016X}) uses depth slot {} whose depth-only clear was dropped at frame {} and not "
                  "cleared since", frame_, draw.vertex_shader_hash, draw.pixel_shader_hash, draw.truth_depth, it->second);
    }
    owned_depth_dropped_.erase(it);
    break;
  }
}

void NativeRenderer::OwnedDepthClearsDrop() {
  for (const OwnedPendingDepthClear& e : owned_depth_clears_) {
    ++owned_frame_stats_.dclear_dropped;
    if (OwnedOnly()) {  // (Session 92, M2a, review D8)
      ++g_m2a.dclear_dropped;
      OwnedMarkDepthStale(e.slot);
      if (OwnedPendBitOn()) OwnedPendClearMake(e.draw, kPendDepthDrop, false);  // (Session 92, M4f-B: the clear is the slot's pending clear, not lost)
    }
    bool found = false;
    for (auto& d : owned_depth_dropped_) {
      if (d.first == e.slot) {
        d.second = frame_;
        found = true;
      }
    }
    if (!found && owned_depth_dropped_.size() < 16) owned_depth_dropped_.emplace_back(e.slot, frame_);
  }
  owned_depth_clears_.clear();
}

bool NativeRenderer::OwnedHoldClear(const Draw& draw) {
  // (Segment log only: which test turned a clear down - the line of the return.)
  const auto hold_no = [&](int line) {
    // (Session 87: the census names why a clear was not held - the pass it then opens in the truth path is a "start" fallback.)
    if (owned_frame_on_) {
      if (owned_frame_on_ && owned_frame_pending_reason_ == kOwnedReasonNone)
        OwnedNoteRefusal(kOwnedReasonStart, fmt::format("clear not held, owned_steps.cpp:{} (class {} colour known {} mask {:X} blend {:08X} depth {} stencil {})",
                                                      line, draw.truth_class, draw.clear_color_known, draw.color_write_mask & 0xF,
                                                      draw.blend_control, draw.raster.depth_test || draw.raster.depth_write, draw.raster.stencil));
    }
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} clear not held (owned_steps.cpp:{}): colour {} base {} pitch {}", frame_, line,
                  draw.truth_color, draw.color_info & 0xFFF, draw.surface_info & 0x3FFF);
    }
    return false;
  };
  // (Session 87, D3D layer F2: a record clear (draw.d3d_record) has no pipeline by design.)
  if (!REXCVAR_GET(gpu_native_owned_frame_hold_clears) || !owned_frame_on_ || !draw.fast_clear ||
      (draw.pipeline == VK_NULL_HANDLE && !draw.d3d_record) || draw.truth_color < 0 ||
      size_t(draw.truth_color) >= OwnedSlotCount() || draw.truth_class < 1 || draw.truth_class > 2) {
    return hold_no(__LINE__);
  }
  const int16_t color = draw.truth_color, depth = draw.truth_depth;
  if (depth >= int16_t(OwnedSlotCount())) return hold_no(__LINE__);
  const OwnedSurface c = OwnedSlotKey(color);
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  const bool msaa_ok = REXCVAR_GET(gpu_native_owned_frame_hold_msaa_clears);
  if (((draw.truth_sx != 1 || draw.truth_sy != 1) && !msaa_ok) || TruthTwinPitch(pitch) || c.base == kTruthScratchBase) {
    return hold_no(__LINE__);
  }
  // (A pass waiting for its copy: an uncommitted one that owes the EDRAM something, over these tiles, is replayed first as for
  // any truth draw - not held; an uncommitted one of this very surface cannot be overwritten before its copy. A committed pass
  // of this surface or over these tiles is ended by this clear (it would otherwise take the next draw as one of its own); a
  // parked one is dropped. A defined start owes the tiles nothing.)
  const bool same_surface = owned_pending_.valid && owned_pending_.base == (draw.color_info & 0xFFF) &&
                            owned_pending_.pitch_tiles == pitch && owned_pending_.kind == draw.truth_class;
  if (owned_pending_.valid) {
    // (chains: surfaces, not shared EDRAM tiles - another surface's pass is not concerned; the same depth surface is.)
    const bool over = OwnedChains() ? (depth >= 0 && owned_pending_.depth && owned_pending_.depth_slot == depth)
                                    : (OwnedDrawTiles(draw) & owned_pending_.tiles).any();
    if (!owned_pending_.committed && ((over && !owned_pending_.edram_free) || same_surface)) {
      // (chains, run 13/14: a pass nobody copied - the crowd atlas copy, 0/16/0 ps 7ABD, a covered 1-draw pass - was pending when the scene's clear
      // arrived and refused it: the whole scene fell to the truth path, once a frame. A pass the commit can keep is committed and parked, as any
      // pass another event meets; its pixels reach the truth images dirty, before this clear.)
      if (OwnedChains()) {
        if (!same_surface) {
          // (follow-up (a) 2: only the DEPTH surface is shared - a pass of another colour surface must not be parked with a depth this clear
          // replaces: it is committed (its colour, and the depth it wrote, reach the truth images dirty) and ended.)
          if (!owned_pending_.edram_free && (owned_pending_.start_kind == kOwnedStartCovered || OwnedTruthHolds())) {
            OwnedCommit();
            OwnedEndCommitted("a held clear of its depth surface");
          } else {
            OwnedReplayPending("a held clear of its depth surface", false);
          }
          if (OwnedOnly() && depth >= 0 && (draw.raster.depth_write || draw.raster.stencil)) {  // (review D2; failure read 1b: only a clear that writes depth / stencil: the clear replaces that depth - the record just noted holds the old one)
            OwnedMarkDepthStale(depth);
          }
        } else if (OwnedPassDeadUnder(draw)) {
          // (follow-up item 5: a pass nobody copied, whose box and depth this clear writes whole, is dead: it ends, uncommitted - no dirty tiles.)
          OwnedPass& dp = owned_pending_;
          OwnedEndPass();
          dp.valid = false;
          dp.committed = false;
          if (dp.census >= 0 && size_t(dp.census) < owned_frame_passes_.size()) {
            OwnedFramePass& rr = owned_frame_passes_[dp.census];
            rr.owned = true;
            rr.lost = false;
            rr.reason = kOwnedReasonNone;
          }
          dp.draws.clear();
          dp.crowd_calls.clear();
          dp.serves.clear();
          dp.held_clears.clear();
          ++owned_frame_stats_.ch_dead_ended;
        } else if (!owned_pending_.edram_free && (owned_pending_.start_kind == kOwnedStartCovered || OwnedTruthHolds())) {
          OwnedParkOrReplay("a held clear of the pending pass's surface");
        } else {
          // (a defined start meeting a partial clear - its pixels outside the clear are still needed - and a pass that owes the EDRAM and that the commit cannot keep: replayed, not paused - the clear is held, it does not go ahead of it.)
          OwnedReplayPending("a held clear of the pending pass's surface", false);
        }
      } else {
        return hold_no(__LINE__);
      }
    }
    if (owned_pending_.committed && ((over && !owned_pending_.edram_free) || same_surface)) OwnedEndCommitted("a held clear over its tiles or surface");
  }
  OwnedDropParkedIf("a held clear over its tiles or surface", [&](const OwnedPass& q) {
    return q.base == (draw.color_info & 0xFFF) && q.pitch_tiles == pitch && (!OwnedChains() || q.kind == uint32_t(draw.truth_class));
  });
  if (!owned_held_.draws.empty() &&
      (owned_held_.color != color || owned_held_.depth != depth || owned_held_.kind != draw.truth_class)) {
    OwnedFlushHeld(true);
    // (review F10: the pass just begun is of this very colour surface - this clear must not go ahead of it.)
    if (OwnedChains() && owned_pending_.valid && owned_pending_.chain_pass && owned_pending_.base == (draw.color_info & 0xFFF) &&
        owned_pending_.pitch_tiles == pitch && owned_pending_.kind == draw.truth_class) {
      // (it is committed and parked, as any pass another event meets: its pixels reach the truth images dirty, before anything of this clear.)
      OwnedParkOrReplay("a clear of the same colour surface follows the pass just begun");
      // (and the parked pass is dropped like any same-surface pass: left parked it would be taken up ahead of this newer clear.)
      OwnedDropParkedIf("a held clear over its tiles or surface", [&](const OwnedPass& q) {
        return q.base == (draw.color_info & 0xFFF) && q.pitch_tiles == pitch && q.kind == uint32_t(draw.truth_class);
      });
    }
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  // (As TruthRecordDraw clamps it: the rows the image can have, the image's width.)
  const uint32_t max_rows = (2048u + c.row_tiles() - 1) / c.row_tiles() * 16 * scale_y;
  rect[3] = std::min(rect[3], int32_t(max_rows));
  rect[0] = std::clamp(rect[0], 0, int32_t(c.pitch_tiles * 80 * scale_x));
  rect[2] = std::clamp(rect[2], rect[0], int32_t(c.pitch_tiles * 80 * scale_x));
  rect[1] = std::clamp(rect[1], 0, rect[3]);
  if (rect[2] <= rect[0] || rect[3] <= rect[1]) return hold_no(__LINE__);
  const bool writes_color = (draw.color_write_mask & 0xF) != 0;
  const bool d_real = depth >= 0 && OwnedSlotKey(depth).base != kTruthScratchBase;
  ClearsInfo trial = owned_held_.info;
  ClearsAccumulate(trial, draw, rect, owned_held_.draws.empty(), writes_color, d_real, c.kind);
  // (chains, the other order: a depth-only clear of this very box was pended first - it joins these colour clears; its draw goes first.)
  std::vector<Draw> merged_depth;
  if (OwnedChains() && trial.valid && writes_color && owned_held_.draws.empty() && depth >= 0 && !trial.uses_depth) {
    for (size_t i = 0; i < owned_depth_clears_.size(); ++i) {
      const OwnedPendingDepthClear& e = owned_depth_clears_[i];
      if (e.slot != depth || e.rect[0] != rect[0] || e.rect[1] != rect[1] || e.rect[2] != rect[2] || e.rect[3] != rect[3] || e.draw.raster.stencil) continue;
      trial.uses_depth = true;
      trial.depth = true;
      trial.depth_value = e.draw.clear_depth;
      merged_depth.push_back(e.draw);
      owned_depth_clears_.erase(owned_depth_clears_.begin() + i);
      ++owned_frame_stats_.ch_dmerge;
      break;
    }
  }
  if (!trial.valid) {
    // (not a clear the owned frame can hold: the truth path takes it, after those before it - chains: those of another surface become a pass)
    if (OwnedOnly()) {
      // Failure read fix 6: a second clear of the surface in another colour. The held ones become a pass of ours; this clear starts a new held set (it is
      // not dropped) - the pass just begun is parked first, as for any clear of the same colour surface (review F10).
      OwnedFlushHeld(true);
      if (OwnedChains() && owned_pending_.valid && owned_pending_.chain_pass && owned_pending_.base == (draw.color_info & 0xFFF) &&
          owned_pending_.pitch_tiles == pitch && owned_pending_.kind == uint32_t(draw.truth_class)) {
        OwnedParkOrReplay("a clear of the same colour surface follows the pass just begun");
        OwnedDropParkedIf("a held clear over its tiles or surface", [&](const OwnedPass& q) {
          return q.base == (draw.color_info & 0xFFF) && q.pitch_tiles == pitch && q.kind == uint32_t(draw.truth_class);
        });
      }
      if (owned_held_.draws.empty()) {
        ClearsInfo fresh;
        ClearsAccumulate(fresh, draw, rect, true, writes_color, d_real, c.kind);
        if (fresh.valid) {
          owned_held_.info = fresh;
          owned_held_.color = color;
          owned_held_.depth = depth;
          owned_held_.kind = draw.truth_class;
          owned_held_.draws.push_back(draw);
          serve_pending_.clear();
          serve_rename_.valid = false;
          return true;
        }
      }
      return hold_no(__LINE__);
    }
    OwnedFlushHeld(OwnedChains() && owned_held_.color != color && (depth < 0 || owned_held_.depth != depth));
    return hold_no(__LINE__);
  }
  owned_held_.info = trial;
  owned_held_.color = color;
  owned_held_.depth = depth;
  owned_held_.kind = draw.truth_class;
  for (Draw& md : merged_depth) owned_held_.draws.push_back(std::move(md));
  owned_held_.draws.push_back(draw);
  serve_pending_.clear();
  serve_rename_.valid = false;
  return true;
}

void NativeRenderer::OwnedFlushHeld(bool as_pass, int site) {
  if (owned_held_.draws.empty()) return;
  const bool only = OwnedOnly();
  if ((as_pass || only) && OwnedChains() && OwnedBeginClearedPass()) return;  // (Session 87: the clears are a pass of ours, not truth work; Session 92, M2a: always, under the latch)
  if (only) {
    // (Session 92, M2a: the held clears could not become a pass of ours (budget, a shape, a paused key ...) and there is no truth path to run them in:
    // dropped, counted by what they were - M2c turns these into the surface's pending clear.)
    const uint32_t n = uint32_t(owned_held_.draws.size());
    g_m2a.held_dropped += n;
    const bool pend = OwnedPendBitOn();  // (Session 92, M4f-B: the dropped clears are the surfaces' pending clears - consumed by the next pass start / copy, not lost)
    if (pend) {
      // (review fix 6: one record per slot - a held set is pended only when its merged clears are ONE whole value (the last colour clear, whose own rectangle is the set's box); anything else is counted in the pend line.)
      const ClearsInfo& hi = owned_held_.info;
      const bool whole = hi.valid && hi.rect[0] == 0 && hi.rect[1] == 0 && hi.rect[2] > 0 && hi.rect[3] > 0 && hi.area == uint64_t(hi.rect[2]) * uint64_t(hi.rect[3]);
      const int32_t none[4] = {-1, -1, -1, -1};
      size_t last_colour = SIZE_MAX;
      for (size_t i = 0; i < owned_held_.draws.size(); ++i) {
        if ((owned_held_.draws[i].color_write_mask & 0xF) != 0) last_colour = i;
      }
      for (size_t i = 0; i < owned_held_.draws.size(); ++i) {
        const Draw& d = owned_held_.draws[i];
        if ((d.color_write_mask & 0xF) != 0 && i != last_colour) continue;  // (superseded inside the set)
        OwnedPendClearMake(d, kPendHeldDrop, false, 0, 0, (d.color_write_mask & 0xF) != 0 ? (whole ? hi.rect : none) : nullptr);
      }
    } else {
      for (const Draw& d : owned_held_.draws) {
        if (d.d3d_record) ++g_m2a.lost_clears;
      }
    }
    static uint32_t logged = 0;
    if (logged < 16) {
      ++logged;
      const Draw& d = owned_held_.draws.front();
      REXGPU_INFO("[native][owned-only] frame {}: {} held clear(s) dropped (owned_steps.cpp:{}): colour slot {} depth slot {} class {} record {} {}", frame_, n, site,
                  owned_held_.color, owned_held_.depth, d.truth_class, d.d3d_record, owned_flush_note_);
    }
    owned_held_.draws.clear();
    owned_held_.info = ClearsInfo();
    owned_flush_note_.clear();
    return;
  }
  std::vector<Draw> draws;
  draws.swap(owned_held_.draws);
  const std::string flush_note = std::move(owned_flush_note_);
  owned_flush_note_.clear();
  owned_held_.info = ClearsInfo();
  // (As OwnedReplayDraws: what the draw being recorded was given for its served bindings is its own, kept across.)
  std::vector<VkImage> serve_pending;
  serve_pending.swap(serve_pending_);
  const ServeRename serve_rename = serve_rename_;
  serve_rename_.valid = false;
  owned_flushing_held_ = true;
  for (const Draw& d : draws) {
    truth_foreign_bind_ = true;
    truth_bound_valid_ = false;
    // (Session 87: the truth pass this clear opens says why it was flushed.)
    if (owned_frame_on_) {
      if (owned_frame_on_ && owned_frame_pending_reason_ == kOwnedReasonNone)
        OwnedNoteRefusal(kOwnedReasonStart, fmt::format("held clear flushed (owned_steps.cpp:{}): class {} colour known {} mask {:X} depth {}{}{}", site,
                                                      d.truth_class, d.clear_color_known, d.color_write_mask & 0xF, d.clear_depth >= 0.0f,
                                                      flush_note.empty() ? "" : "; ", flush_note));
    }
    TruthRecordDraw(d);
  }
  truth_foreign_bind_ = true;
  truth_bound_valid_ = false;
  owned_flushing_held_ = false;
  serve_pending_.swap(serve_pending);
  serve_rename_ = serve_rename;
  command_processor_.NativeForgetState();
}

bool NativeRenderer::OwnedCopyOfPass(const draw_util::ResolveInfo& info, const OwnedPass& q) const {
  // (Session 87, gpu_native_owned_frame_chains, design C.) A copy belongs to a pass by the surface it reads - colour: base, pitch and class;
  // depth: the pass's depth surface - never by EDRAM tiles it happens to share with other surfaces (they are all at base 0 here).
  if (!q.valid) return false;
  if (info.IsCopyingDepth()) {
    return q.depth && !q.rec_depth_gone && q.depth_slot >= 0 && size_t(q.depth_slot) < OwnedSlotCount() &&
           uint32_t(info.depth_original_base) == OwnedSlotKey(q.depth_slot).base;
  }
  const auto fmt = xenos::ColorRenderTargetFormat(uint32_t(info.color_edram_info.format));
  const bool is_rgba8 = fmt == xenos::ColorRenderTargetFormat::k_8_8_8_8;
  return !q.rec_colour_gone && uint32_t(info.color_original_base) == q.base && info.color_edram_info.pitch_tiles == (q.pitch_tiles << (q.is_64bpp ? 1 : 0)) &&
         bool(info.color_edram_info.format_is_64bpp) == q.is_64bpp && is_rgba8 == (q.kind == 1);
}

bool NativeRenderer::OwnedDrawOfPass(const Draw& draw, uint32_t base, uint32_t pitch, const OwnedPass& q) const {
  // (As above for a truth draw: the surface it draws, or the depth surface it uses.)
  if (!q.valid) return false;
  if (draw.truth_color >= 0 && base == q.base && pitch == q.pitch_tiles && uint32_t(draw.truth_class) == q.kind) return true;
  const bool depth_used = draw.truth_depth >= 0 && (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil);
  return depth_used && q.depth && draw.truth_depth == q.depth_slot;
}

bool NativeRenderer::OwnedBeginClearedPass() {
  // Session 87, gpu_native_owned_frame_chains, design A. The held clears of a surface that no draw took as its start (a copy reads the surface,
  // the next draw is another surface's, the next draw's texture request ...) used to run in the truth path as a pass of their own - whose truth
  // copy then replayed the pending owned pass and paused its key. Here they START the surface's pass in our images instead: colour (and depth /
  // stencil where the clears wrote them whole) cleared as a `cleared` start clears them, the render pass begun now, the clears kept with the pass
  // (replayed first if it is lost). Draws of the surface that follow are recorded into it (fits_of), its copy is served from it. False = not
  // begun (the caller runs the clears in the truth path as before): nothing is changed then but the pending pass, parked.
  OwnedFrameStats& st = owned_frame_stats_;
  if (!owned_frame_on_ || owned_replaying_ || owned_held_.draws.empty() || !OwnedChains() || !OwnedActive() || owned_failed_ || !truth_ ||
      !replacing_ || truth_failed_) {
    return false;
  }
  const auto decline = [&]() {
    ++st.ch_declined;
    return false;
  };
  const Draw first = owned_held_.draws.front();
  const ClearsInfo pc = owned_held_.info;
  const int16_t color = owned_held_.color, depth = owned_held_.depth;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  if (color < 0 || size_t(color) >= OwnedSlotCount() || first.truth_sx != 1 || first.truth_sy != 1) return decline();
  const auto rt_format = xenos::ColorRenderTargetFormat((first.color_info >> 16) & 0xF);
  const bool rgba16 = first.truth_class == 2 && (rt_format == xenos::ColorRenderTargetFormat::k_16_16 ||
                                                 rt_format == xenos::ColorRenderTargetFormat::k_16_16_16_16);
  if (!(first.truth_class == 1 && rt_format == xenos::ColorRenderTargetFormat::k_8_8_8_8) && !rgba16) return decline();
  const bool is_64bpp = xenos::IsColorRenderTargetFormat64bpp(rt_format);
  const uint32_t base = first.color_info & 0xFFF;
  const uint32_t pitch = std::max(1u, ((first.surface_info & 0x3FFF) * first.truth_sx + 79) / 80);
  if (!pc.valid || pc.rect[0] != 0 || pc.rect[1] != 0 || pc.rect[2] <= 0 || pc.rect[3] <= 0 || pc.rect[2] > 4096 || pc.rect[3] > 4096 ||
      pc.rect[2] % int32_t(scale_x) != 0 || pc.rect[3] % int32_t(scale_y) != 0 ||
      pc.area != uint64_t(pc.rect[2]) * uint64_t(pc.rect[3]) || (rgba16 && !pc.quantized)) {
    return decline();
  }
  const bool depth_real = depth >= 0 && size_t(depth) < OwnedSlotCount() && OwnedSlotKey(depth).base != kTruthScratchBase;
  const bool keep_depth = pc.uses_depth && pc.depth && depth_real;
  const bool keep_stencil = keep_depth && pc.stencil;
  if ((pc.uses_depth && !keep_depth) || (pc.stencil_touched && !keep_stencil)) return decline();
  if (command_processor_.NativeReadbackActive() && !REXCVAR_GET(gpu_native_owned_frame_readback_copy)) return decline();
  const OwnedPassKey key = OwnedKeyOf(first);
  if (OwnedPaused(key) || OwnedPresentKeyed(key)) return decline();
  const uint32_t width = uint32_t(pc.rect[2]), height = uint32_t(pc.rect[3]);
  if (!OwnedImageAvailable(width, height, first.truth_class, keep_depth)) return decline();
  // (The pass pending for another surface is parked - never replayed for this.)
  OwnedPass& p = owned_pending_;
  if (p.valid) {
    OwnedParkOrReplay("a clear starts its surface's pass");
    if (p.valid) return decline();
  }
  if (owned_held_.draws.empty()) return false;
  EndTruthPass();  // (our pass is a pass of its own)
  owned_clears_ended_ = false;
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  const int16_t index = OwnedImageFor(width, height, first.truth_class);
  if (index < 0) return decline();
  if (keep_depth && !OwnedEnsureDepth(index)) return decline();
  p = OwnedPass();
  p.valid = true;
  p.half = false;
  p.dres = false;
  p.dres_full = -1;
  p.dres_color_used = p.dres_depth_used = false;
  p.kind = first.truth_class;
  p.is_64bpp = is_64bpp;
  p.committed = p.started = p.open = false;
  p.carried_defined = false;  // (Session 90)
  p.cleared = true;
  p.depth = keep_depth;
  p.stencil = keep_stencil;
  p.start = keep_depth ? uint32_t(kOwnedPassClearBoth) : uint32_t(kOwnedPassClearColor);
  for (uint32_t i = 0; i < 4; ++i) {
    const float c = pc.color[i];
    if (rgba16) {
      const float s = std::min(std::max(c, -32.0f), 32.0f) * (32767.0f / 32.0f);
      p.clear_values[0].color.float32[i] = std::trunc(s + (s < 0.0f ? -0.5f : 0.5f)) / 32767.0f;
    } else {
      const float s = std::min(std::max(c, 0.0f), 1.0f) * 255.0f;
      p.clear_values[0].color.float32[i] = std::trunc(s + 0.5f) / 255.0f;
    }
  }
  p.clear_values[1].depthStencil = {pc.depth_value, keep_stencil ? uint32_t(pc.stencil_value) : 0u};
  p.base = base;
  p.pitch_tiles = pitch;
  p.color_slot = color;
  p.depth_slot = keep_depth ? depth : int16_t(-1);
  p.width = width / scale_x;
  p.height = height / scale_y;
  p.host_width = width;
  p.host_height = height;
  p.image = index;
  p.frame = frame_;
  p.last_tick = ++owned_tick_;  // (Session 92, M2a)
  p.submission = command_processor_.GetCurrentSubmission();
  p.tiles = OwnedOnly() ? std::bitset<2048>() : EdramTiles(base, pitch, is_64bpp, 0, 0, p.width, p.height);  // (Session 92, M2b: no tile bookkeeping under the latch)
  if (keep_depth && !OwnedOnly()) p.tiles |= EdramTiles(OwnedSlotKey(depth).base, pitch, false, 0, 0, p.width, p.height);
  p.clears_ours = true;
  p.clears_depth = pc.uses_depth;
  p.clears_stencil = pc.stencil_touched;
  p.clears_submission = p.submission;
  p.clears_held = true;
  const size_t clear_draws = owned_held_.draws.size();
  OwnedPendRetryDone(color);  // (review fix 5: a re-held pended clear that became a pass is off the retry list)
  p.held_clears = std::move(owned_held_.draws);
  owned_held_.draws.clear();
  owned_held_.info = ClearsInfo();
  p.key = key;
  p.chain_pass = p.chain_unkeyed = true;
  p.start_kind = uint8_t(kOwnedStartCleared);
  ++owned_draw_serial_;  // (a new pass image state: the strip batch / 16-bit page fill caches must not answer for the old one; review)
  // (Older restorable passes of the surface go first, then the drop: under chains the drop keeps the pass the restore draw that follows needs - the
  // crowd atlas pass draws over the scene's surface between the scene's copies and the restore from them, run 21.)
  owned_restorable_.erase(std::remove_if(owned_restorable_.begin(), owned_restorable_.end(), [&](const OwnedPass& q) {
    return q.base == p.base && q.pitch_tiles == p.pitch_tiles && q.kind == p.kind;
  }), owned_restorable_.end());
  OwnedDropParkedIf("the same surface starts again", [&](const OwnedPass& q) {
    return q.base == p.base && q.pitch_tiles == p.pitch_tiles && q.kind == p.kind;  // (review F7's drop by depth slot is NOT done: it dropped the scene pass the restore draw needs, run 12)
  });
  if (keep_depth) p.depth_tick = p.last_tick;  // (Session 92, M2a)
  OwnedSupersedeRecords(p.base, p.pitch_tiles, p.kind, p.depth ? p.depth_slot : int16_t(-1));  // (review D4)
  {
    const int32_t rec = OwnedFrameNewPass(first);
    OwnedFramePass& r = owned_frame_passes_[rec];
    r.start = p.start_kind;
    r.reason = kOwnedReasonNone;
    r.detail.clear();
    r.box_w = p.width;
    r.box_h = p.height;
    p.census = rec;
  }
  ++truth_counts_.owned_passes;
  // The render pass begun now (nothing is drawn yet: a copy that follows would read an image never cleared): the clears are its load ops.
  const OwnedImage& o = owned_images_[index];
  OwnedTouch(index);
  const VkImage depth_image = keep_depth ? o.depth : owned_scratch_depths_[o.scratch].image;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkImageMemoryBarrier barriers[2];
  barriers[0] = OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, kOwnedAnyWrite,
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  barriers[1] = OwnedBarrier(depth_image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  owned_images_[index].readable = false;
  if (keep_depth) owned_images_[index].depth_read = false;
  owned_bound_ = OwnedBound();
  cb.CmdVkPipelineBarrier(VkPipelineStageFlags(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT),
                          kOwnedAttachmentStages, 0, 0, nullptr, 0, nullptr, 2, barriers);
  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  const uint32_t used_variant = OwnedTilerVariant(index, p.start, keep_depth, true, &p.key);
  begin_info.renderPass = owned_render_passes_[used_variant][o.kind];
  begin_info.framebuffer = keep_depth ? o.framebuffer_depth : o.framebuffer;
  begin_info.renderArea = {{0, 0}, {o.width, o.height}};
  begin_info.clearValueCount = p.start == kOwnedPassClearBoth ? 2 : 1;
  begin_info.pClearValues = p.clear_values;
  VkClearValue poison_clears[2];  // (Session 95, gpu_tiler_poison)
  VkClearValue dc_values[2];  // (Session 95, Lane E, gpu_native_owned_dont_care_clear: wins over the poison)
  if (OwnedDcMode()) {
    OwnedDontCareValues(dc_values, &p.clear_values[0], p.start == kOwnedPassClearBoth ? &p.clear_values[1] : nullptr);
    begin_info.clearValueCount = 2;
    begin_info.pClearValues = dc_values;
  } else if (OwnedPoisonOn()) {
    begin_info.clearValueCount = OwnedPoisonClears(used_variant, p.clear_values, poison_clears, o.width, o.height, 2);
    begin_info.pClearValues = begin_info.clearValueCount ? poison_clears : nullptr;
  }
  p.timer_segment = OwnedTimerBegin("pass", o.width, o.height);
  p.timer_draws = 0;
  cb.CmdVkBeginRenderPass(&begin_info, VK_SUBPASS_CONTENTS_INLINE);
  p.open = true;
  p.started = true;
  ++st.ch_started;
  if (owned_chain_site_ == 1) ++st.ch_site_sub;
  if (owned_chain_site_ == 2) ++st.ch_site_frame;
  if (owned_chain_site_ == 3) ++st.ch_site_write;
  st.ch_clears += uint32_t(clear_draws);
  (keep_depth ? st.ch_clears_cd : st.ch_clears_c) += uint32_t(clear_draws);
  (first.truth_class == 1 ? st.ch_class1 : st.ch_class2) += uint32_t(clear_draws);
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned pass from flushed clears: our image {} ({}x{}), colour {} depth {} ({}), {} clear draws", frame_, index, width,
                height, color, p.depth_slot, keep_depth ? "kept" : "none", clear_draws);
  }
  return true;
}

bool NativeRenderer::OwnedPresentKeyed(const OwnedPassKey& key) const {
  if (OwnedChains()) return false;  // (chains: nothing is ever banned - the composite is a covered start and its front copy is served)
  return std::find(owned_present_keys_.begin(), owned_present_keys_.end(), key) != owned_present_keys_.end();
}

void NativeRenderer::OwnedPresentAdd(const OwnedPassKey& key, const char* why) {
  if (OwnedChains()) {
    // (A front copy that cannot be served replays its pass that once; nothing is remembered.)
    static uint32_t logged = 0;
    if (logged < 12) {
      ++logged;
      REXGPU_INFO("[native][owned] frame {}: a front copy of the pass (base {} pitch {} format {} ps {:016X}) was not served: {} (replayed once, nothing remembered)",
                  frame_, key.base, key.pitch, key.format, key.ps, why);
    }
    return;
  }
  if (std::find(owned_present_keys_.begin(), owned_present_keys_.end(), key) == owned_present_keys_.end()) owned_present_keys_.push_back(key);
}

bool NativeRenderer::OwnedResumeNow() {
  // The pending pass's render pass begun again (as OwnedRecord resumes it), with nothing drawn yet.
  OwnedPass& p = owned_pending_;
  if (!p.valid || !p.started || p.open || p.committed || p.image < 0 || size_t(p.image) >= owned_images_.size()) return false;
  const OwnedImage& o = owned_images_[p.image];
  OwnedTouch(p.image);
  const VkImage depth_image = p.depth ? o.depth : owned_scratch_depths_[o.scratch].image;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  EndTruthPass();
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  ++truth_counts_.owned_resumes;
  VkImageMemoryBarrier barriers[2];
  barriers[0] = OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             owned_images_[p.image].readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                             : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  owned_images_[p.image].readable = false;
  const bool depth_ro = p.depth && owned_images_[p.image].depth_read;
  if (depth_ro) owned_images_[p.image].depth_read = false;
  barriers[1] = OwnedBarrier(depth_image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                             depth_ro ? 0 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                             depth_ro ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  owned_bound_ = OwnedBound();
  cb.CmdVkPipelineBarrier(VkPipelineStageFlags(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT),
                          kOwnedAttachmentStages, 0, 0, nullptr, 0, nullptr, 2, barriers);
  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  const uint32_t used_variant = OwnedTilerVariant(p.image, kOwnedPassResume, p.depth, false, &p.key);
  begin_info.renderPass = owned_render_passes_[used_variant][o.kind];
  begin_info.framebuffer = p.depth ? o.framebuffer_depth : o.framebuffer;
  begin_info.renderArea = {{0, 0}, {o.width, o.height}};
  begin_info.clearValueCount = 0;
  begin_info.pClearValues = nullptr;
  VkClearValue poison_clears[2];  // (Session 95, gpu_tiler_poison)
  VkClearValue dc_values[2];  // (Session 95, Lane E, gpu_native_owned_dont_care_clear: the resume variants' tiler twin clears depth; wins over the poison)
  if (OwnedDcMode()) {
    OwnedDontCareValues(dc_values, nullptr, nullptr);
    begin_info.clearValueCount = 2;
    begin_info.pClearValues = dc_values;
  } else if (OwnedPoisonOn()) {
    begin_info.clearValueCount = OwnedPoisonClears(used_variant, nullptr, poison_clears, o.width, o.height, 3);
    begin_info.pClearValues = begin_info.clearValueCount ? poison_clears : nullptr;
  }
  p.timer_segment = OwnedTimerBegin("pass", o.width, o.height);
  p.timer_draws = 0;
  cb.CmdVkBeginRenderPass(&begin_info, VK_SUBPASS_CONTENTS_INLINE);
  p.open = true;
  return true;
}

void NativeRenderer::OwnedClaimClearTiles(const OwnedPass& p) {
  // Design review F3: clears held for a pass never ran in the truth path, so no truth image owns the box - the dirty tiles a commit marks
  // would then be skipped by OwnedMaterialize (it copies only into tiles the image owns) and a truth reader would see the old EDRAM.
  // The box (colour, and the depth the clears wrote) is claimed for the truth images here, as the truth pass of the clears would have.
  if (OwnedOnly()) return;  // (Session 92, M2a: no truth image is made or claimed at a commit)
  if (p.half || p.color_slot < 0 || size_t(p.color_slot) >= truth_images_.size()) return;
  EndTruthPass();
  command_processor_.EndRenderPass();  // (review: EnsureTruthImage may record a flush / a grow copy - outside any render pass)
  const bool depth = p.depth && p.depth_slot >= 0 && size_t(p.depth_slot) < truth_images_.size();
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const auto fits = [&](int16_t slot) {  // (an image that is there and tall enough needs no EnsureTruthImage - its deferred stores flush)
    const TruthImage& ti = truth_images_[slot];
    return ti.initialized && ti.image != VK_NULL_HANDLE && ti.width == ti.pitch_tiles * 80 * scale_x && ti.height >= p.host_height;
  };
  if ((!fits(p.color_slot) && !EnsureTruthImage(p.color_slot, p.host_height)) ||
      (depth && !fits(p.depth_slot) && !EnsureTruthImage(p.depth_slot, p.host_height))) {
    ++owned_frame_stats_.ch_claim_failed;
    return;
  }
  const TruthImage& t = truth_images_[p.color_slot];
  // (Only tiles no truth image owns: a claim of tiles another surface's image owns would strip that image - the covered passes whose copies
  // are served rely on owning their box - for a place the surfaces merely share (run 12: 9C85DA5F, 'truth holds' false from frame 4415).)
  const std::bitset<2048> ct = EdramTiles(t.base, t.pitch_tiles, t.is_64bpp(), 0, 0, p.width, p.height) & ~truth_owned_any_;
  std::bitset<2048> dt;
  if (depth) {
    const TruthImage& d = truth_images_[p.depth_slot];
    dt = EdramTiles(d.base, d.pitch_tiles, false, 0, 0, p.width, p.height) & ~truth_owned_any_;
  }
  if (ct.none() && dt.none()) return;
  TruthSetOwners(ct, p.color_slot, dt, depth ? p.depth_slot : int16_t(-1));
  if (depth) truth_images_[p.depth_slot].stencil_valid &= ~dt;
  ++owned_frame_stats_.ch_claims;
}

void NativeRenderer::OwnedReplayParkedPass(OwnedPass q, const char* why) {
  // The draws of a pass parked without its copy, recorded into the truth path now (the same submission: their sets and constants hold), as a
  // pending pass lost before its copy is replayed; the pending pass is set aside meanwhile (its render pass ends, a later draw resumes it).
  if (OwnedOnly()) return;  // (Session 92, M2a: no replay under the latch - every caller is gated; the pass q is simply not replayed)
  OwnedPass& p = owned_pending_;
  if (p.valid && p.open) OwnedEndPass();
  OwnedPass saved = std::move(p);
  p = std::move(q);
  p.valid = true;
  p.committed = false;
  p.open = false;
  ++truth_counts_.owned_lost_draw;
  ++owned_frame_stats_.ch_parked_replays;
  OwnedFrameLost(p.census, kOwnedReasonPresent, why);
  OwnedReplayDraws();
  p = std::move(saved);
}

bool NativeRenderer::OwnedPassDeadUnder(const Draw& clear) const {
  // The pending pass nobody has copied, and this clear writes everything it holds: its whole box in all four channels, and the depth (and stencil)
  // it keeps, whole to the clear's constant. Nothing of it can be read afterwards.
  const OwnedPass& pp = owned_pending_;
  if (!pp.valid || pp.committed || pp.copy_count != 0 || pp.feedback_pass || pp.half || !clear.fast_clear) return false;  // (a defined start too: a clear over its whole box leaves nothing of it)
  int32_t r[4];
  TruthDrawRect(clear, r);
  if (r[0] > 0 || r[1] > 0 || r[2] < int32_t(pp.host_width) || r[3] < int32_t(pp.host_height)) return false;
  if ((clear.color_write_mask & 0xF) != 0xF || clear.blend_control != 0x00010001u || (clear.color_mask_all & ~0xFu)) return false;
  if (pp.depth) {
    if (clear.truth_depth != pp.depth_slot || clear.clear_depth < 0.0f || !(clear.raster.depth_test && clear.raster.depth_write &&
        clear.raster.depth_compare == VK_COMPARE_OP_ALWAYS) || clear.raster.depth_bias_constant != 0.0f || clear.raster.depth_bias_slope != 0.0f) {
      return false;
    }
    if (pp.stencil) {
      const auto whole = [](uint32_t ops) {
        return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS);
      };
      if (!clear.raster.stencil || !whole(clear.raster.stencil_ops & 0xFFF) || !whole((clear.raster.stencil_ops >> 12) & 0xFFF) ||
          clear.stencil_write_masks[0] != 0xFF || clear.stencil_write_masks[1] != 0xFF || clear.stencil_refs[0] != clear.stencil_refs[1]) {
        return false;
      }
    }
  }
  return true;
}

bool NativeRenderer::OwnedLoadFromTruth(int16_t slot, int16_t image, uint32_t width, uint32_t height) {
  // Session 87, chains (design "Scene-change composite"): the box of a surface the truth path holds content of, copied once into our image. The dirty tiles of
  // our own earlier passes go into the truth image first; only the tiles that image owns are copied (the rest of the box is transparent black, as the
  // converted start had it); the image is left in its attachment layout for a render pass that loads.
  if (OwnedOnly()) return false;  // (Session 92, M2a: the truth path holds nothing of any surface; a resumed start comes from the surface's newest record - OwnedStartExtra)
  if (!OwnedChains() || !owned_frame_on_ || owned_materializing_ || slot < 0 || size_t(slot) >= truth_images_.size() || image < 0 ||
      size_t(image) >= owned_images_.size()) {
    return false;
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const std::bitset<2048> box = [&]() {
    const TruthImage& t0 = truth_images_[slot];
    return EdramTiles(t0.base, t0.pitch_tiles, t0.is_64bpp(), 0, 0, width / scale_x, height / scale_y);
  }();
  if (!owned_dirty_.empty()) OwnedDirtyBeforeRead(box, slot, "a loaded start");
  {
    const TruthImage& t = truth_images_[slot];
    // (review A3: only a box the truth image owns WHOLE is loaded - a partial one would leave transparent-black holes over content the pass then 'holds'.)
    if (!t.initialized || t.image == VK_NULL_HANDLE || (box & ~t.owned).any()) return false;
  }
  owned_materializing_ = true;
  EndTruthPass();
  {
    DeferConcern c;
    c.all = false;
    c.images[0] = truth_images_[slot].image;
    c.images[1] = truth_images_[slot].rename_source;
    TruthFlushDeferredFor(kDeferFlushPassIntoSource, c);
  }
  owned_materializing_ = false;
  command_processor_.EndRenderPass();  // (outside any render pass: the barriers and the copy below)
  command_processor_.SubmitBarriers(true);
  const TruthImage& t = truth_images_[slot];
  if (!t.initialized || t.image == VK_NULL_HANDLE) return false;
  if ((box & ~t.owned).any()) return false;
  const std::bitset<2048> tiles = box & t.owned;
  if (tiles.none()) return false;
  std::vector<std::array<int32_t, 4>> rects;
  TruthTileRects(t, tiles, rects);
  std::vector<VkImageCopy> regions;
  for (const auto& r : rects) {
    const int32_t x0 = std::max(r[0], 0), y0 = std::max(r[1], 0);
    const int32_t x1 = std::min({r[2], int32_t(width), int32_t(t.width)}), y1 = std::min({r[3], int32_t(height), int32_t(t.height)});
    if (x1 <= x0 || y1 <= y0) continue;
    VkImageCopy region = {};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffset = region.dstOffset = {x0, y0, 0};
    region.extent = {uint32_t(x1 - x0), uint32_t(y1 - y0), 1};
    regions.push_back(region);
  }
  if (regions.empty()) return false;
  const OwnedImage& o = owned_images_[image];
  OwnedTouch(image);
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const VkAccessFlags attachment_access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  VkImageMemoryBarrier barriers[2] = {
      OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, kOwnedAnyWrite | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
      OwnedBarrier(t.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)};
  const uint32_t timer = OwnedTimerBegin("load_from_truth", width, height);
  cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, barriers);
  const VkClearColorValue zero = {};
  const VkImageSubresourceRange range = ui::vulkan::util::InitializeSubresourceRange();
  cb.CmdVkClearColorImage(o.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
  const VkImageMemoryBarrier clear_done =
      OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &clear_done);
  OwnedCopyImage(cb, t.image, false, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, o.image, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), regions.data());  // (Session 94: into every layer)
  barriers[0] = OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, attachment_access, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  barriers[1] = OwnedBarrier(t.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                             attachment_access | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, barriers);
  OwnedTimerEnd(timer, uint32_t(tiles.count()));
  owned_images_[image].readable = false;
  return true;
}

void NativeRenderer::OwnedHeldBeforeTextures() {
  // (s82 hold investigation: the next draw's texture request decides serves, renames and sampled flushes from the truth
  // images' state - TruthServeView, TruthFlushDeferredTexture. A clear the game issued before that draw must already be
  // in that state, as it is when it is recorded at once. Only while a carried / deferred copy is pending: with none, the
  // request decides nothing from the truth images and the held clears may still become the next owned pass's start.)
  if (owned_held_.draws.empty() || owned_flushing_held_ || owned_replaying_ || !owned_frame_on_) return;
  if (!REXCVAR_GET(gpu_native_owned_frame_hold_flush_at_textures) || truth_deferred_.empty()) return;
  OwnedFlushHeld(true);
}

extern uint32_t g_s92_held_not_aging;  // (Session 92, E2: owned_post.cpp)
namespace {
// Session 92 (the "start / format" family): file-level counters (no new members in the header the exe includes), read and
// reset by OwnedFrameSummary - colour-less draws joined to the owned pass holding their depth, and restores begun from held clears.
struct S92Stats {
  uint32_t joined = 0, no_depth_pass = 0, outside_box = 0, no_stencil = 0, held_clears = 0;
  uint32_t restores_taken = 0, restores_no_held = 0, restores_not_cleared = 0;
  uint32_t fallen_front_drops = 0, stencil_claims = 0, unproven_stencil = 0;  // (C, D1)
  // (E1: a depth-using draw's colour-only pass given depth.)
  uint32_t depth_given = 0, depth_from_dirty = 0, depth_from_pass = 0, depth_from_clear = 0;  // (depth_from_clear: Session 92, M2d, a dead depth clear's value)
  uint32_t depth_held = 0, depth_edram_free = 0, depth_no_colour = 0, depth_no_source = 0, depth_size = 0, depth_half = 0, depth_stencil = 0, depth_budget = 0;
};
S92Stats g_s92;
}  // namespace

bool NativeRenderer::OwnedTakeDrawImpl(const Draw& draw) {
  ++g_m2d_draw_index;  // (Session 92, M2d: the order log's draw index of the frame)
  M2dWatchDraw(frame_, draw.vertex_shader_hash, draw.pixel_shader_hash);
  if (owned_replaying_) return false;
  OwnedPass& p = owned_pending_;
  const bool frame = owned_frame_on_;
  owned_frame_pending_reason_ = kOwnedReasonNone;  // (Session 82: this draw's refusal, for the pass it lands in)
  owned_crowd_first_ = false;                      // (set again by OwnedRecord of a crowd run's first member)
  if (owned_watch_pending_.load(std::memory_order_acquire)) OwnedDrainCpuWrites();
  if (!OwnedActive() || owned_failed_ || !truth_ || !replacing_ || truth_failed_) {
    if (p.valid) OwnedReplayPending("owned steps off before its copy");
    if (frame && owned_failed_ && truth_ && replacing_) OwnedNoteRefusal(kOwnedReasonFailed, "owned images failed");
    return false;
  }
  const uint32_t base = draw.color_info & 0xFFF;
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  // Not taken: a truth draw that writes or tests tiles of an owned pass waiting for its copy (colour or depth, any
  // surface over them) - the pass replayed into the truth images first, then this draw over it as on the console.
  const auto not_taken = [&](bool refused, int site = 0) {
    // (Session 87: a return with no refuse() note names its source line as the pass's reason detail, so the census says which
    // path let a draw fall to the truth path without a reason.)
    if (site && frame && (owned_frame_on_ && owned_frame_pending_reason_ == kOwnedReasonNone)) OwnedNoteRefusal(kOwnedReasonStart, fmt::format("no reason, owned_steps.cpp:{}", site));
    if (refused) ++truth_counts_.owned_refused;
    // Session 92, M2a: under the latch the draw is dropped by TruthRecordDraw (OwnedDropDraw) - it touches no pending or parked pass, pauses nothing,
    // replays nothing: the passes of other draws and their copies go on as they are.
    if (OwnedOnly()) return false;
    if (!owned_parked_.empty()) {
      const auto dt = OwnedDrawTiles(draw);
      const bool chains = OwnedChains();  // (Session 87: by surface, not by shared tiles)
      const auto overlaps = [&](const OwnedPass& q) { return chains ? OwnedDrawOfPass(draw, base, pitch, q) : (dt & q.tiles).any(); };
      // (A truth draw over a parked pass's tiles: as for the pending one - a pass only our image holds is paused.)
      const bool keep = REXCVAR_GET(gpu_native_owned_frame_park_keep);
      const auto own_surface = [&](const OwnedPass& q) { return q.base == base && q.pitch_tiles == pitch; };
      for (const OwnedPass& q : owned_parked_) {
        if (!chains && keep && !own_surface(q)) continue;
        if (overlaps(q) && !draw.fast_clear && !q.edram_free && !OwnedTruthHolds(q)) {
          ++truth_counts_.owned_refused;
          OwnedPause(q.key, q.producers, "a truth draw over a pass only our image holds");
        }
      }
      OwnedDropParkedIf("a truth draw over its tiles",
                        [&](const OwnedPass& q) { return overlaps(q) && (chains || !keep || own_surface(q)); });
    }
    if (frame && p.valid && !p.committed) {
      // Session 82 (review): a pass waiting for its copy is replayed before any truth draw that is not its own - it
      // would otherwise be replayed later, after uploads, texture reloads and copies this draw's inputs went through
      // (its replay could read newer inputs than the truth path did).
      // (A clear that is not over its tiles - G10's colour-less depth clear before the front buffer's copy - has no
      // inputs from memory and changes nothing the pass's replay reads: the pass waits for its copy.)
      const bool concerns = OwnedChains() ? OwnedDrawOfPass(draw, base, pitch, p)
                                          : !(draw.fast_clear && !(OwnedDrawTiles(draw) & p.tiles).any());
      if (concerns && OwnedChains()) {
        // (Session 87: which draw replays a pending pass, and why it was refused - the first ones named in the log.)
        ++owned_frame_stats_.ch_draw_replays;
        static uint32_t replay_logged = 0;
        if (replay_logged < 24) {
          ++replay_logged;
          int32_t rr[4];
          TruthDrawRect(draw, rr);
          REXGPU_INFO("[native][owned] frame {}: a truth draw replays the pending pass (base {} pitch {} {}x{}, {} draws, depth {}, start {}): "
                      "vs {:016X} ps {:016X} class {} blend {:08X} depth {}{} stencil {} mask {:X} rect {},{}-{},{} fast_clear {}; refused as: reason {} {}",
                      frame_, p.base, p.pitch_tiles, p.host_width, p.host_height, p.draws.size(), p.depth, int(p.start_kind),
                      draw.vertex_shader_hash, draw.pixel_shader_hash, draw.truth_class, draw.blend_control, draw.raster.depth_test ? "t" : "-",
                      draw.raster.depth_write ? "w" : "-", draw.raster.stencil, draw.color_write_mask & 0xF, rr[0], rr[1], rr[2], rr[3],
                      draw.fast_clear, int(owned_frame_pending_reason_), owned_frame_pending_detail_);
        }
      }
      if (concerns && OwnedChains() && !p.edram_free && (p.start_kind == kOwnedStartCovered || OwnedTruthHolds())) {
        // (review F8: a late replay reads inputs newer than the pass's draws did; the pass is committed - its tiles dirty, copied into the truth
        // images before this draw - and ends. No pause.)
        OwnedCommit();
        ++owned_frame_stats_.ch_commit_instead;
        OwnedEndCommitted("a truth draw of its surface before its copy");
      } else if (concerns) {
        OwnedReplayPending("a truth draw before its pass's copy", !p.edram_free && !(OwnedChains() && draw.fast_clear));  // (P2: a defined start has no state to protect: not paused)
      }
    } else if (p.valid && (OwnedChains() ? OwnedDrawOfPass(draw, base, pitch, p) : (OwnedDrawTiles(draw) & p.tiles).any())) {
      // (Session 82, P2: a committed pass whose every tile this draw writes whole - a proven cover, unblended, all
      // channels, nothing read - is dead: it just ends, no pause (the post blur's tiles under the combine).)
      if (frame && p.committed && OwnedDrawOverwrites(draw, p)) {
        OwnedEndCommitted("a truth draw overwrites its tiles");
        return false;
      }
      // (Session 82: which draw ends the pass - the first few logged, all while the census log runs.)
      static uint32_t over_logged = 0;
      if (frame && (owned_frame_logging_ || over_logged < 24)) {
        ++over_logged;
        int32_t r[4];
        TruthDrawRect(draw, r);
        REXGPU_INFO("[native][owned] frame {}: a draw over the tiles of the owned pass (base {} pitch {} ps {:016X}, {} "
                    "draws) before its copy: vs {:016X} ps {:016X} colour 0x{:08X} c{} d{} mask 0x{:X} blend 0x{:08X} "
                    "clear {} depth {}{} stencil {} rect {},{}-{},{} {} wrote it",
                    frame_, p.base, p.pitch_tiles, p.key.ps, p.draws.size(), draw.vertex_shader_hash,
                    draw.pixel_shader_hash, draw.color_info, draw.truth_color, draw.truth_depth, draw.color_write_mask,
                    draw.blend_control, draw.fast_clear, draw.raster.depth_test ? "t" : "-",
                    draw.raster.depth_write ? "w" : "-", draw.raster.stencil, r[0], r[1], r[2], r[3],
                    refused ? "refused" : "left");
      }
      // (Review, Session 80: a committed pass only our image holds - the truth images never got its draws: a clear
      // overwrites its tiles as on the console; any other draw over them would read or keep stale pixels - its
      // producers not owned for a while.)
      if (p.edram_free) {
        // (A defined start owes the truth tiles nothing: a truth draw over them does not concern it.)
      } else {
        if (p.committed && !draw.fast_clear && !OwnedTruthHolds()) {
          ++truth_counts_.owned_refused;
          OwnedPause(p.key, p.producers, "a truth draw over a pass only our image holds");
        }
        OwnedReplayPending("a draw over its tiles before its copy");
      }
    }
    return false;
  };
  if (!OwnedWants(draw.pixel_shader_hash, draw.vertex_shader_hash)) return not_taken(false);
  // Session 82 (the owned frame): a clear is part of a pass's start (its clears' pass), not a pass of its own - no
  // reason; a draw with no pipeline yet is skipped as the truth path skips it (counted, no loss, no pause: on a cold
  // pipeline cache every new pipeline would turn its pass off for 10 s).
  // (Session 82, P2: the frame's last depth-only clear of the scene's depth surface - dead in our world, no pass at all.)
  if (frame && draw.fast_clear && OwnedDeadDepthClear(draw)) {
    serve_pending_.clear();
    serve_rename_.valid = false;
    return true;
  }
  if (frame && draw.fast_clear) {
    if (OwnedChains() && OwnedDepthClearInPass(draw)) return true;  // (Session 87)
    if (draw.truth_depth >= 0 && draw.raster.depth_write && draw.clear_depth >= 0.0f) {  // (a clear that writes this depth: the old record is moot)
      owned_depth_dropped_.erase(std::remove_if(owned_depth_dropped_.begin(), owned_depth_dropped_.end(),
                                                [&](const std::pair<int16_t, uint64_t>& r) { return r.first == draw.truth_depth; }),
                                 owned_depth_dropped_.end());
    }
    if (OwnedDepthClearPend(draw)) return true;  // (Session 87)
    if (OwnedHoldClear(draw)) return true;  // (held: nothing recorded - an owned pass may take it as its start)
    return not_taken(false, __LINE__);
  }
  if (frame && !draw.fast_clear) {
    // (Session 87, step 4: a reference - a copy of the list per draw before; the list is read only here, the loop ends first.)
    const std::string& excl = REXCVAR_GET(gpu_native_owned_frame_exclude);
    for (size_t k = 0; !excl.empty() && k < excl.size();) {
      const size_t end = std::min(excl.find(',', k), excl.size());
      const unsigned long long hash = std::strtoull(excl.substr(k, end - k).c_str(), nullptr, 16);
      if (hash && hash == draw.pixel_shader_hash) {
        OwnedNoteRefusal(kOwnedReasonOther, "excluded");
        return not_taken(true);
      }
      k = end + 1;
    }
  }
  if (frame && draw.pipeline == VK_NULL_HANDLE) {
    // (Counted; TruthRecordDraw skips the draw itself, as it always has - its own bookkeeping, no replay, no pause.)
    ++owned_frame_stats_.no_pipeline;
    return false;
  }
  // (The first refusals are logged with the draw's state - with the owned frame on, every one while the census log
  // runs, else 16 per reason.)
  static uint32_t refusals_logged = 0;
  static uint32_t refusals_by_reason[kOwnedReasonCount] = {};
  const auto refuse = [&](const char* why, uint8_t reason, const char* detail = nullptr) {
    reason = std::min<uint8_t>(reason, uint8_t(kOwnedReasonOther));  // (an extension's reason is indexed below)
    const bool log = frame ? (owned_frame_logging_ || refusals_by_reason[reason] < 16) : refusals_logged < 16;
    if (log) {
      ++refusals_logged;
      ++refusals_by_reason[reason];
      REXGPU_INFO("[native][owned] frame {}: vs {:016X} ps {:016X} refused ({}): class {} colour 0x{:08X} c{} d{} mask "
                  "0x{:X} all 0x{:X} blend 0x{:08X} clear {} depth {}{} stencil {} sx {} cover {},{}-{},{} restore {} "
                  "served {} pass c{} d{} clears {} {},{}-{},{} depth {}/{} stencil {}/{}",
                  frame_, draw.vertex_shader_hash, draw.pixel_shader_hash, detail ? detail : why, draw.truth_class, draw.color_info,
                  draw.truth_color, draw.truth_depth, draw.color_write_mask, draw.color_mask_all, draw.blend_control,
                  draw.fast_clear, draw.raster.depth_test ? "t" : "-", draw.raster.depth_write ? "w" : "-",
                  draw.raster.stencil, draw.truth_sx, draw.cover[0], draw.cover[1], draw.cover[2], draw.cover[3],
                  draw.restore, serve_pending_.size(), truth_pass_open_ ? truth_pass_color_ : -1,
                  truth_pass_open_ ? truth_pass_depth_ : -1, truth_pass_clears_.valid, truth_pass_clears_.rect[0],
                  truth_pass_clears_.rect[1], truth_pass_clears_.rect[2], truth_pass_clears_.rect[3],
                  truth_pass_clears_.uses_depth, truth_pass_clears_.depth, truth_pass_clears_.stencil_touched,
                  truth_pass_clears_.stencil);
    }
    if (frame) {
      OwnedNoteRefusal(reason, detail ? detail : why);
      // (Session 87: the signature of what is refused, for the window's summary.)
      int32_t gr[4];
      TruthDrawRect(draw, gr);
      const std::string sig = fmt::format(
          "{}/{} ps {:016X} vs {:016X} cls {} sx {} blend {:08X} z {}{} st {} mask {:X} rect {},{}-{},{} pass_open {} parked {}", OwnedReasonName(reason),
          detail ? detail : why, draw.pixel_shader_hash, draw.vertex_shader_hash, draw.truth_class, draw.truth_sx, draw.blend_control,
          draw.raster.depth_test ? "t" : "-", draw.raster.depth_write ? "w" : "-", draw.raster.stencil, draw.color_write_mask, gr[0],
          gr[1], gr[2], gr[3], truth_pass_open_, owned_pending_.valid);
      auto& groups = owned_frame_stats_.refusal_groups;
      auto it = std::find_if(groups.begin(), groups.end(), [&](const std::pair<std::string, uint32_t>& g) { return g.first == sig; });
      if (it != groups.end()) {
        ++it->second;
      } else if (groups.size() < 256) {
        groups.emplace_back(sig, 1u);
      }
    }
    return not_taken(true);
  };
  // (Not where the CPU reads copies back - the readback mode, readback_resolve_override or the setting; review.)
  // (Session 82, P2: not any more for every pass - the copy decides, in OwnedTakeResolve: only a copy whose bytes the CPU
  // gets is the truth path's, the frame's own recurring copies are served as outside a window.)
  if (command_processor_.NativeReadbackActive() && !REXCVAR_GET(gpu_native_owned_frame_readback_copy)) {
    return refuse("readback window", kOwnedReasonReadback);
  }
  // (RGBA8 k_8_8_8_8 surfaces, and - class 2 - the 16-bit k_16_16 / k_16_16_16_16 ones: the shadow maps.)
  const auto rt_format = xenos::ColorRenderTargetFormat((draw.color_info >> 16) & 0xF);
  const bool rgba16 = draw.truth_class == 2 && (rt_format == xenos::ColorRenderTargetFormat::k_16_16 ||
                                                rt_format == xenos::ColorRenderTargetFormat::k_16_16_16_16);
  // (Session 82: a multisampled surface first - its target is no owned class either.)
  if (draw.truth_sx != 1 || draw.truth_sy != 1) return refuse("kind", kOwnedReasonMsaa);
  // (Session 92, mechanism A: a colour-less draw - no colour target, colour mask 0, depth and / or stencil - is not refused as
  // "format": it joins the owned pass that holds its depth slot, below. Its colour_info is meaningless, so the format and class
  // tests do not apply to it.)
  const bool colourless = frame && OwnedChains() && draw.truth_color < 0 && (draw.color_write_mask & 0xF) == 0 && draw.truth_depth >= 0 &&
                          (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil) && !draw.fast_clear &&
                          !draw.memexport && !draw.restore;
  if (!colourless && ((!(draw.truth_class == 1 && rt_format == xenos::ColorRenderTargetFormat::k_8_8_8_8) && !rgba16) ||
                      draw.truth_color < 0)) {
    return refuse("target", kOwnedReasonFormat);
  }
  const bool is_64bpp = xenos::IsColorRenderTargetFormat64bpp(rt_format);
  if (draw.fast_clear) return refuse("kind", kOwnedReasonOther, "clear");
  if (draw.memexport) return refuse("kind", kOwnedReasonOther, "memexport");
  if (draw.occlusion_query) return refuse("kind", kOwnedReasonOther, "occlusion query");
  if (draw.xfb_buffer != VK_NULL_HANDLE) return refuse("kind", kOwnedReasonOther, "stream out");
  if (!draw.instance_count) return refuse("kind", kOwnedReasonOther, "no instances");
  if (draw.restore && !(frame && REXCVAR_GET(gpu_native_owned_frame_restore_skip))) {
    return refuse("kind", kOwnedReasonOther, "restore");  // (with the owned frame: below, once the passes are known)
  }
  if (draw.own_skip_call && !frame) return refuse("kind", kOwnedReasonOther, "own crowd");  // (Session 82: with the owned frame on, recorded - the run drawn by ours into the pass)
  if (TruthTwinPitch(pitch)) return refuse("kind", kOwnedReasonOther, "draw twin pitch");
  if (draw.pipeline == VK_NULL_HANDLE) return refuse("no pipeline", kOwnedReasonNoPipeline);
  if (!serve_pending_.empty() && !frame) return refuse("served binding", kOwnedReasonInputServed);  // (Session 82, P1 item 2: with the owned frame on, made readable around our pass - OwnedRecord)
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  if (rect[2] <= rect[0] || rect[3] <= rect[1]) {
    // (follow-up (b) 2: nothing is drawn, so nothing reads or writes the pending pass's surface: no commit, no replay.)
    if (OwnedChains()) return false;
    return not_taken(false, __LINE__);  // (nothing drawn: as the truth path)
  }
  const bool depth_used =
      draw.truth_depth >= 0 && (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil);
  const bool unblended = (draw.color_write_mask & 0xF) == 0xF && !(draw.color_mask_all & ~0xFu) &&
                         draw.blend_control == 0x00010001u;
  const bool covered = !depth_used && unblended && rect[0] == 0 && rect[1] == 0 && rect[2] % int32_t(scale_x) == 0 &&
                       rect[3] % int32_t(scale_y) == 0 && rect[2] <= 4096 && rect[3] <= 4096 &&
                       draw.cover[2] > draw.cover[0] && draw.cover[3] > draw.cover[1] && draw.cover[0] <= 0 &&
                       draw.cover[1] <= 0 && draw.cover[2] * int32_t(scale_x) >= rect[2] &&
                       draw.cover[3] * int32_t(scale_y) >= rect[3];
  // (Review: a committed pass's box covered whole by this draw - nothing of it is kept: a pass of its own, as before
  // passes went on after their copy - no snapshot, no join.)
  if (p.valid && p.committed && covered && rect[2] >= int32_t(p.host_width) && rect[3] >= int32_t(p.host_height) &&
      draw.truth_color == p.color_slot) {
    OwnedEndCommitted("a draw covers its whole box");
  }
  // A draw of the pass waiting for its copy: the same surface and images, inside its box, depth and stencil only where
  // our image holds them.
  const auto fits_of = [&](const OwnedPass& q) {
    if (colourless) {  // (Session 92: matched on the depth slot, not the colour base / slot / class; same pitch, inside the box, stencil kept when used)
      return q.kind == 1 && q.pitch_tiles == pitch && q.depth && !q.rec_depth_gone && draw.truth_depth == q.depth_slot && (!draw.raster.stencil || q.stencil) &&
             rect[0] >= 0 && rect[1] >= 0 && rect[2] <= int32_t(q.host_width) && rect[3] <= int32_t(q.host_height);
    }
    return !q.rec_colour_gone && q.base == base && q.pitch_tiles == pitch && draw.truth_color == q.color_slot &&
           draw.truth_class == q.kind && (!depth_used || (q.depth && !q.rec_depth_gone && draw.truth_depth == q.depth_slot)) &&
           (!draw.raster.stencil || q.stencil) && rect[0] >= 0 && rect[1] >= 0 &&
           rect[2] <= int32_t(q.host_width) && rect[3] <= int32_t(q.host_height);
  };
  // (Session 82: a parked committed pass this draw fits is taken up again - the pending one, if it is a committed pass,
  // ends.)
  if (frame && !owned_parked_.empty() && !(p.valid && fits_of(p)) && (!p.valid || p.committed || OwnedChains())) {
    for (size_t pj = 0; pj < owned_parked_.size(); ++pj) {
      const size_t pi = OwnedChains() ? owned_parked_.size() - 1 - pj : pj;  // (review F7: the newest parked pass first)
      if (!fits_of(owned_parked_[pi])) continue;
      // (Session 82, P2: a draw that covers the parked pass's whole box - the combine over the scene - starts a pass of its own
      // into a fresh image, as for a pending committed pass above: it samples the scene's page, and joining would draw into
      // the image it reads; the front buffer's copy is then that pass's, served.)
      if (covered && rect[2] >= int32_t(owned_parked_[pi].host_width) && rect[3] >= int32_t(owned_parked_[pi].host_height)) {
        continue;
      }
      OwnedPass taken_up = std::move(owned_parked_[pi]);
      owned_parked_.erase(owned_parked_.begin() + pi);
      if (p.valid) {
        if (p.committed) OwnedEndCommitted("another parked pass is taken up");
        else OwnedParkOrReplay("another parked pass is taken up");  // (chains: an uncommitted pass is parked, not replayed)
      }
      OWNED_SCENE_LOG(taken_up, "[native][owned][scene] frame {}: parked pass base {} pitch {} taken up again", frame_, taken_up.base, taken_up.pitch_tiles);
      owned_pending_ = std::move(taken_up);
      if (segment_log_frames_ > 0) {
        REXGPU_INFO("[native][seg] frame {} the parked pass (base {} pitch {}) is taken up again", frame_,
                    owned_pending_.base, owned_pending_.pitch_tiles);
      }
      if (OwnedChains()) ++owned_frame_stats_.ch_takenup;
      if (owned_pending_.carried_defined) ++owned_frame_stats_.ch_carried_taken_draw;
      break;
    }
  }
  // Session 82 (P2): a restore draw (the 360 puts a scene's colour and depth back from the two copies it took out) over a
  // pass of ours that holds the surface, covering its whole box: nothing to restore - our image never lost them. Left out
  // like a dropped dead draw. Else the truth path's, as before.
  if (colourless) {
    // Session 92, mechanism A: recorded into the pass that holds this draw's depth slot, with the pass's colour attachment bound
    // and the draw's own colour write mask 0 (its pipeline's; as the truth path records such a draw into an open pass).
    // Clears held for that depth would run after the draw in the pass: those go the old way.
    if (p.valid && fits_of(p)) {
      if (!owned_held_.draws.empty() && owned_held_.depth == draw.truth_depth) {
        ++g_s92.held_clears;
        return refuse("target", kOwnedReasonFormat, "no colour target: clears held for its depth");
      }
      if (OwnedRecord(draw, rect)) {
        ++g_s92.joined;
        return true;
      }
      return not_taken(true, __LINE__);
    }
    // Refused: why. (Nothing holds the depth / the pass has no stencil / the draw is outside the pass's box or pitch.)
    bool depth_pass = false, stencil_missing = false;
    const auto look = [&](const OwnedPass& q) {
      if (!q.valid || !q.depth || q.depth_slot != draw.truth_depth) return;
      depth_pass = true;
      if (draw.raster.stencil && !q.stencil) stencil_missing = true;
    };
    look(p);
    for (const OwnedPass& q : owned_parked_) look(q);
    const char* detail;
    if (!depth_pass) {
      ++g_s92.no_depth_pass;
      detail = "no colour target, no owned pass holds its depth";
    } else if (stencil_missing) {
      ++g_s92.no_stencil;
      detail = "no colour target, the owned pass keeps no stencil";
    } else {
      ++g_s92.outside_box;
      detail = "no colour target, outside the owned pass's box";
    }
    return refuse("target", kOwnedReasonFormat, detail);
  }
  bool restore_recorded = false;  // (Session 87)
  bool restore_from_held = false;  // (Session 92, mechanism B: this restore begins a pass from the clears held for its surface)
  if (draw.restore && owned_frame_logging_) {
    std::string parked;
    for (const OwnedPass& q : owned_parked_) parked += fmt::format(" [{} {}]", q.base, q.pitch_tiles);
    REXGPU_INFO("[native][owned][scene] frame {}: RESTORE draw: pending valid {} committed {} (base {} pitch {}) since copy {} fits {}; parked:{}",
                frame_, p.valid, p.committed, p.base, p.pitch_tiles, p.draws_since_color + p.draws_since_depth, p.valid && fits_of(p), parked);
  }
  // (Session 82, P2: the pass the front buffer's composite dropped - the restore that follows undoes that composite: the pass is
  // taken up again, our image is what the restore brings back.)
  if (draw.restore && !(p.valid && fits_of(p)) && !owned_restorable_.empty()) {
    for (size_t ri = 0; ri < owned_restorable_.size(); ++ri) {
      if (!fits_of(owned_restorable_[ri]) || rect[0] != 0 || rect[1] != 0 ||
          rect[2] < int32_t(owned_restorable_[ri].host_width) || rect[3] < int32_t(owned_restorable_[ri].host_height)) {
        continue;
      }
      if (p.valid) OwnedParkOrReplay("a restore of a dropped pass");
      OWNED_SCENE_LOG(owned_restorable_[ri], "[native][owned][scene] frame {}: pass base {} pitch {} taken back for the restore", frame_, owned_restorable_[ri].base, owned_restorable_[ri].pitch_tiles);
      owned_pending_ = owned_restorable_[ri];
      if (owned_pending_.carried_defined) ++owned_frame_stats_.ch_carried_taken_draw;
      owned_restorable_.erase(owned_restorable_.begin() + ri);
      OwnedMarkDirty(false);  // (review: the composite's draw took the tiles' dirty state - what the restore would have put back is ours again)
      break;
    }
  }
  if (draw.restore) {
    // (The proof the surface is as the copies left it: the pass was copied (committed) and nothing was drawn into it since its
    // last copy - an overlay drawn in between would be reverted by the 360's restore and kept by the skip.)
    // (Review: the restore samples the very pages of the pass's last whole copies - a sub-rectangle copy of another texture is not it.)
    if (p.valid && p.committed && p.draws_since_color == 0 && p.draws_since_depth == 0 && p.copy_color_page &&
        p.copy_color_page == draw.restore_color_page && p.copy_depth_page == draw.restore_depth_page && fits_of(p) && rect[0] == 0 &&
        rect[1] == 0 && rect[2] >= int32_t(p.host_width) && rect[3] >= int32_t(p.host_height)) {
      static uint32_t restore_logged = 0;
      if (segment_log_frames_ > 0 || owned_frame_logging_ || restore_logged < 6) {
        ++restore_logged;
        REXGPU_INFO("[native][owned] frame {}: a restore draw (vs {:016X} ps {:016X}) over the owned pass of base {} pitch {} "
                    "left out ({}x{}, {} draws, committed {})",
                    frame_, draw.vertex_shader_hash, draw.pixel_shader_hash, p.base, p.pitch_tiles, p.host_width, p.host_height,
                    p.draws.size(), p.committed);
      }
      serve_pending_.clear();
      serve_rename_.valid = false;
      ++owned_frame_stats_.restores_left_out;
      OwnedMarkDirty(false);  // (review: as the restore would have left the truth tiles - ours to copy in before anything reads them)
      return true;
    }
    // (Session 87, gpu_native_owned_frame_restore_draw: a restore the proof above cannot leave out - the pass was drawn into
    // after the copies the restore samples, or its last copy is another one - is the game's own draw: it reads the pages of the
    // two copies the game took (ours: a snapshot and a converted depth page) and writes colour and depth over the whole box,
    // so it is recorded into the pass like any draw. The clears held for the surface before it are dead under it.)
    if (!(frame && REXCVAR_GET(gpu_native_owned_frame_restore_draw) && p.valid && fits_of(p) && rect[0] == 0 && rect[1] == 0 &&
          rect[2] >= int32_t(p.host_width) && rect[3] >= int32_t(p.host_height))) {
      // Session 92, mechanism B (S90_BLACK_ARENA 8.5 item 2): no pass of ours holds the surface, but the clears held for it are whole
      // and this restore draws the whole of their box - they are its cleared start (colour + depth + stencil as the clears left them),
      // and the restore is recorded as the pass's first draw. The same conditions as the cleared start below (which decides again;
      // a restore it does not clear is refused there as before). Nothing goes through the truth path: no per-call flag to nest.
      bool take = false;
      if (frame && OwnedChains() && REXCVAR_GET(gpu_native_owned_frame_restore_draw) && !(p.valid && fits_of(p))) {
        if (owned_held_.draws.empty()) {
          ++g_s92.restores_no_held;
        } else {
          const ClearsInfo& hc = owned_held_.info;
          const bool keep_d = hc.uses_depth && hc.depth && owned_held_.depth >= 0;
          const bool keep_s = keep_d && hc.stencil;
          const bool depth_used_r = draw.truth_depth >= 0 && (draw.raster.depth_test || draw.raster.depth_write || draw.raster.stencil);
          take = owned_held_.color == draw.truth_color && hc.valid && hc.rect[0] == 0 && hc.rect[1] == 0 && hc.rect[2] > 0 &&
                 hc.rect[3] > 0 && hc.rect[2] <= 4096 && hc.rect[3] <= 4096 && hc.rect[2] % int32_t(scale_x) == 0 &&
                 hc.rect[3] % int32_t(scale_y) == 0 && hc.area == uint64_t(hc.rect[2]) * uint64_t(hc.rect[3]) && !rgba16 &&
                 rect[0] == 0 && rect[1] == 0 && rect[2] == hc.rect[2] && rect[3] == hc.rect[3] &&
                 (!depth_used_r || (keep_d && draw.truth_depth == owned_held_.depth)) && (!draw.raster.stencil || keep_s) &&
                 (!hc.uses_depth || keep_d) && (!hc.stencil_touched || keep_s);
          if (!take) ++g_s92.restores_not_cleared;
        }
      }
      if (!take) return refuse("kind", kOwnedReasonOther, "restore");
      restore_from_held = true;
    } else {
      restore_recorded = true;
    }
  }
  // Session 92 (E1, TRUE NATIVE): a depth-using draw whose colour fits a colour-only pass of ours (scratch depth) - Jericho (Retro)'s entrance
  // effect: ~881 blended depth-tested quads a frame onto the combine's image. If one of our images holds the newest pixels of its depth slot
  // (a dirty entry, or a pass with that depth slot), the colour pass gets a depth image of its own, filled by a copy of that one (depth +
  // stencil); the draw and the frame's other quads then fit and record normally. No image of ours holds the depth: refused as before, counted.
  if (frame && OwnedChains() && depth_used && !draw.fast_clear && !draw.restore && !(p.valid && fits_of(p)) &&
      size_t(draw.truth_depth) < OwnedSlotCount()) {
    const auto colour_match = [&](const OwnedPass& q) {
      return q.valid && q.base == base && q.pitch_tiles == pitch && draw.truth_color == q.color_slot && draw.truth_class == q.kind &&
             rect[0] >= 0 && rect[1] >= 0 && rect[2] <= int32_t(q.host_width) && rect[3] <= int32_t(q.host_height);
    };
    const OwnedSurface dt = OwnedSlotKey(draw.truth_depth);
    bool saw_size = false, saw_stencil = false;
    struct Source {
      bool ok = false, stencil = false, pass = false, clear = false;  // (clear: Session 92, M2d - a dead depth clear's value, no image)
      float clear_depth = 0.0f;
      uint32_t clear_stencil = 0;
      int16_t image = -1;
    };
    // The image of ours that holds slot D's newest pixels, for the colour pass c (its size, a different image).
    const auto find_source = [&](const OwnedPass& c) {
      Source s;
      const auto size_ok = [&](int16_t index) {
        if (index < 0 || size_t(index) >= owned_images_.size() || index == c.image || c.image < 0 || size_t(c.image) >= owned_images_.size()) return false;
        const OwnedImage& so = owned_images_[index];
        if (so.depth == VK_NULL_HANDLE) return false;
        if (so.width != owned_images_[c.image].width || so.height != owned_images_[c.image].height) {
          saw_size = true;
          return false;
        }
        return true;
      };
      if (OwnedOnly()) {
        // Session 92, M2a: the truth images own no tile and there are no dirty entries - the newest holder of slot D's depth is found among our
        // passes (pending, parked) and the surfaces' newest records by the tick of their last draw; the colour pass gets a copy of that image's depth.
        const OwnedPass* best = nullptr;
        bool best_record = false;
        const auto consider = [&](const OwnedPass& q, bool record) {
          if (!q.valid || &q == &c || q.image == c.image || !q.depth || q.depth_slot != draw.truth_depth || (q.half && !(q.dres && q.dres_depth_used && q.dres_full >= 0)) || q.rec_depth_gone) return;  // (Session 94: a scaled scene pass counts once its depth copy made the full-size depth)
          if (!best || q.depth_tick > best->depth_tick) {  // (review D3: the last write of the DEPTH, not of the pass)
            best = &q;
            best_record = record;
          }
        };
        consider(p, false);
        for (const OwnedPass& q : owned_parked_) consider(q, false);
        for (const OwnedPass& q : owned_newest_) consider(q, true);
        // Session 92, M2d (Part 3): the newest thing to happen to slot D's depth is a dead depth clear of this frame (no live holder newer than it): the colour-only pass
        // gets a depth image CLEARED to that value, as the 360's quads test against the cleared depth, not the scene's (S92_M1A_BOX_READ.md A).
        if (const OwnedDeadDClearRec* dc = DeadDClearFor(draw.truth_depth, truth_claim_epoch_); dc && (!best || best->depth_tick < dc->tick)) {
          if (c.image < 0 || size_t(c.image) >= owned_images_.size() || c.width > dc->box_w || c.height > dc->box_h) {
            saw_size = true;
            return s;
          }
          if (draw.raster.stencil && !dc->stencil_written) {
            saw_stencil = true;
            return s;
          }
          s.ok = s.clear = true;
          s.clear_depth = dc->depth;
          s.clear_stencil = dc->stencil;
          s.stencil = dc->stencil_written;
          return s;
        }
        const int16_t best_image = best && best->dres ? best->dres_full : (best ? best->image : int16_t(-1));  // (Session 94)
        if (!best || !size_ok(best_image)) return s;
        if (draw.raster.stencil && !best->stencil) {
          saw_stencil = true;
          return s;
        }
        s.ok = true;
        s.pass = !best_record;
        s.image = best_image;
        s.stencil = best->stencil;
        return s;
      }
      const std::bitset<2048> box = EdramTiles(dt.base, dt.pitch_tiles, dt.is_64bpp(), 0, 0, c.width, c.height) & (size_t(draw.truth_depth) < truth_images_.size() ? truth_images_[draw.truth_depth].owned : std::bitset<2048>());
      const auto from_pass = [&](const OwnedPass& q) {
        if (s.ok || !q.valid || &q == &c || !q.depth || q.depth_slot != draw.truth_depth || q.half || !size_ok(q.image)) return;
        if (q.committed) {  // (review: a committed pass's depth is the newest only while a dirty depth entry of its image says so)
          bool newest = false;
          for (const OwnedDirty& e : owned_dirty_) {
            if (e.slot == draw.truth_depth && e.image == q.image && (e.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) && box.any() && (box & ~e.tiles).none()) newest = true;
          }
          if (!newest) return;
        }
        if (draw.raster.stencil && !q.stencil) {
          saw_stencil = true;
          return;
        }
        s.ok = s.pass = true;
        s.image = q.image;
        s.stencil = q.stencil;
      };
      from_pass(p);
      for (size_t i = owned_parked_.size(); i-- > 0;) from_pass(owned_parked_[i]);
      if (s.ok) return s;
      if (box.none()) return s;
      for (size_t i = owned_dirty_.size(); i-- > 0;) {
        const OwnedDirty& e = owned_dirty_[i];
        if (e.slot != draw.truth_depth || !(e.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) || (box & ~e.tiles).any() || !size_ok(e.image)) continue;
        bool stencil = false;
        for (const OwnedDirty& f : owned_dirty_) {
          if (f.slot == e.slot && f.image == e.image && (f.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) && (box & ~f.tiles).none()) stencil = true;
        }
        if (draw.raster.stencil && !stencil) {
          saw_stencil = true;
          continue;
        }
        s.ok = true;
        s.stencil = stencil;
        s.image = e.image;
        return s;
      }
      return s;
    };
    // The colour pass: the pending one, else the newest parked one.
    const OwnedPass* cand = nullptr;
    bool cand_parked = false;
    size_t cand_index = 0;
    if (colour_match(p)) {
      cand = &p;
    } else {
      for (size_t i = owned_parked_.size(); i-- > 0;) {
        if (colour_match(owned_parked_[i])) {
          cand = &owned_parked_[i];
          cand_parked = true;
          cand_index = i;
          break;
        }
      }
    }
    // (Session 92, M4f-B, review fix 4b: a candidate whose depth slot is D but whose depth was made stale by a clear the pass could not take, with that clear pended and newer than the pass's depth: its depth image is cleared to the pended value - the existing clear branch below.)
    const OwnedDeadDClearRec* regive = (OwnedOnly() && OwnedPendBitOn() && cand && cand->depth && cand->rec_depth_gone && cand->depth_slot == draw.truth_depth) ? DeadDClearFor(draw.truth_depth, truth_claim_epoch_) : nullptr;
    if (regive && regive->tick <= cand->depth_tick) regive = nullptr;
    if (!cand || (cand->depth && !regive) || !cand->started) {
      ++g_s92.depth_no_colour;
    } else if (cand->edram_free && !OwnedOnly()) {  // (Session 92, M2a: the truth holds nothing of any pass - an edram-free colour pass is as good as any)
      ++g_s92.depth_edram_free;  // (review: the truth holds nothing of it - the quads' depth would live only in this image)
    } else if (cand->half) {
      ++g_s92.depth_half;
    } else if (!owned_held_.draws.empty() && owned_held_.depth == draw.truth_depth) {
      ++g_s92.depth_held;  // (review: clears held for this depth must run first - the old path flushes them)
    } else {
      Source src = find_source(*cand);
      if (!src.ok) {
        if (saw_size) ++g_s92.depth_size;
        else if (saw_stencil) ++g_s92.depth_stencil;
        else ++g_s92.depth_no_source;
      } else {
        if (cand_parked) {
          OwnedPass taken_up = std::move(owned_parked_[cand_index]);
          owned_parked_.erase(owned_parked_.begin() + cand_index);
          if (p.valid) {
            if (p.committed) OwnedEndCommitted("another parked pass is taken up (depth)");
            else OwnedParkOrReplay("another parked pass is taken up (depth)");
          }
          owned_pending_ = std::move(taken_up);
          if (OwnedChains()) ++owned_frame_stats_.ch_takenup;
          if (owned_pending_.carried_defined) ++owned_frame_stats_.ch_carried_taken_draw;
          src = find_source(owned_pending_);  // (the pass that was pending may have been replayed)
        }
        if (!src.ok) {
          if (saw_size) ++g_s92.depth_size;
          else if (saw_stencil) ++g_s92.depth_stencil;
          else ++g_s92.depth_no_source;
        } else {
          const int16_t ci = p.image, si = src.image;
          if (!src.clear) OwnedTouch(si);
          OwnedTouch(ci);
          // (Copy commands: no render pass of the truth path or the command processor may be open - as at a resume.)
          OwnedEndPass();
          EndTruthPass();
          command_processor_.EndRenderPass();
          command_processor_.SubmitBarriers(true);
          if (!OwnedEnsureDepth(ci) || (!src.clear && owned_images_[si].depth == VK_NULL_HANDLE)) {
            ++g_s92.depth_budget;
          } else if (src.clear) {
            // Session 92, M2d (Part 3): a depth image cleared to the dead depth clear's value (depth + stencil), instead of a copy of an older record's.
            OwnedDepthAccount(ci);  // (Session 93: the destination's old pass is over: its depth is overwritten)
            OwnedImage& d = owned_images_[ci];
            constexpr VkImageAspectFlags kDS = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
            const uint32_t timer = OwnedTimerBegin("depth_into_colour_pass_clear", d.width, d.height);
            VkImageMemoryBarrier barrier = OwnedBarrier(d.depth, kDS, kOwnedAnyWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                    nullptr, 0, nullptr, 1, &barrier);
            const VkClearDepthStencilValue value = {src.clear_depth, src.clear_stencil};
            cb.CmdVkClearDepthStencilImage(d.depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, ui::vulkan::util::InitializeSubresourceRange(kDS));
            barrier = OwnedBarrier(d.depth, kDS, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                    nullptr, 1, &barrier);
            OwnedTimerEnd(timer);
            d.depth_read = false;
            p.depth = true;
            p.depth_slot = draw.truth_depth;
            p.stencil = src.stencil;
            p.depth_tick = ++owned_tick_;
            p.rec_depth_gone = false;
            ++g_s92.depth_given;
            ++g_s92.depth_from_clear;
            ++g_m2a.depth_from_dead_clear;
            PendNoteGivenDepth(DeadDClearFor(draw.truth_depth, truth_claim_epoch_));  // (Session 92, M4f-B: the pending clear the pass was given the value of)
            if (OwnedOnly() && owned_frame_stats_.frames < 4) M2dOrderLog(true, frame_, "colour-only pass given depth", fmt::format("slot {} from a dead clear's value {} (stencil {}) for vs {:016X} ps {:016X}",
                        draw.truth_depth, src.clear_depth, src.clear_stencil, draw.vertex_shader_hash, draw.pixel_shader_hash));
            static uint32_t logged = 0;
            if (logged < 8 || owned_frame_logging_) {
              ++logged;
              REXGPU_INFO("[native][owned] frame {}: the colour-only pass base {} pitch {} ({}x{}) is given depth slot {} CLEARED to the dead depth clear's value {} (stencil {}) for vs {:016X} ps {:016X}",
                          frame_, p.base, p.pitch_tiles, p.host_width, p.host_height, p.depth_slot, src.clear_depth, src.clear_stencil, draw.vertex_shader_hash, draw.pixel_shader_hash);
            }
          } else {
            OwnedNoteDepthRead(si, "depth copy into a colour-only pass");   // (Session 93: the source's depth is copied)
            OwnedDepthAccount(ci);    // (and the destination's old pass is over: its depth is overwritten)
            const OwnedImage& s = owned_images_[si];
            OwnedImage& d = owned_images_[ci];
            const bool src_depth_read = s.depth_read;
            constexpr VkImageAspectFlags kDS = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
            const uint32_t timer = OwnedTimerBegin("depth_into_colour_pass", d.width, d.height);
            VkImageMemoryBarrier barriers[2];
            barriers[0] = OwnedBarrier(s.depth, kDS, src_depth_read ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                       VK_ACCESS_TRANSFER_READ_BIT,
                                       src_depth_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            barriers[1] = OwnedBarrier(d.depth, kDS, kOwnedAnyWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    0, 0, nullptr, 0, nullptr, 2, barriers);
            VkImageCopy region = {};
            region.srcSubresource = region.dstSubresource = {kDS, 0, 0, 1};
            region.extent = {d.width, d.height, 1};
            OwnedCopyImage(cb, s.depth, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.depth, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);  // (Session 94: per layer)
            barriers[0] = OwnedBarrier(s.depth, kDS, 0,
                                       src_depth_read ? VK_ACCESS_SHADER_READ_BIT
                                                      : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       src_depth_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            barriers[1] = OwnedBarrier(d.depth, kDS, VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    0, 0, nullptr, 0, nullptr, 2, barriers);
            OwnedTimerEnd(timer);
            d.depth_read = false;
            p.depth = true;
            p.depth_slot = draw.truth_depth;
            p.stencil = src.stencil;
            p.depth_tick = ++owned_tick_;
            p.rec_depth_gone = false;
            if (!OwnedOnly()) p.tiles |= EdramTiles(dt.base, pitch, false, 0, 0, p.width, p.height);
            ++g_s92.depth_given;
            if (src.pass) ++g_s92.depth_from_pass;
            else ++g_s92.depth_from_dirty;
            if (!src.pass && OwnedOnly()) ++g_m2a.depth_from_record;
            if (OwnedOnly() && owned_frame_stats_.frames < 4) M2dOrderLog(true, frame_, "colour-only pass given depth", fmt::format("slot {} copied from our image {} ({}) for vs {:016X} ps {:016X}", draw.truth_depth, si,
                        src.pass ? "a pass" : (OwnedOnly() ? "a surface record" : "a dirty entry"), draw.vertex_shader_hash, draw.pixel_shader_hash));
            static uint32_t logged = 0;
            if (logged < 8 || owned_frame_logging_) {
              ++logged;
              REXGPU_INFO("[native][owned] frame {}: the colour-only pass base {} pitch {} ({}x{}) is given depth slot {} copied from our image {} ({}, stencil {}) for vs {:016X} ps {:016X}",
                          frame_, p.base, p.pitch_tiles, p.host_width, p.host_height, p.depth_slot, si, src.pass ? "a pass" : (OwnedOnly() ? "a surface record" : "a dirty entry"),
                          src.stencil, draw.vertex_shader_hash, draw.pixel_shader_hash);
            }
          }
        }
      }
    }
  }
  if (p.valid) {
    const bool fits = fits_of(p) && !(OwnedChains() && p.chain_unkeyed && (OwnedPaused(OwnedKeyOf(draw)) || OwnedPresentKeyed(OwnedKeyOf(draw))));  // (review F5)
    if (fits) {
      // (Session 87, review C1: the clears held for the restore's surface are dead under it only when the restore overwrites everything
      // they wrote - they lie inside the box, their depth is the pass's own, no stencil clear, and the restore writes all colour channels
      // and depth. Else they go into the truth path first, as for any draw; and they are dropped only once the draw is recorded.)
      bool drop_held = false;
      bool forced_drop = false, forced_stencil_ok = false;  // (Session 92, M2a, review D6)
      if (restore_recorded && !owned_held_.draws.empty() && owned_held_.color == draw.truth_color) {
        const ClearsInfo& hc = owned_held_.info;
        // (Diagnosis run 024: Bryan's restore clears fail on the stencil alone - the whole box, the pass's depth, mask 0xF, depth
        // write. A stencil the clears wrote whole to one value, under a restore that does not use stencil, is cleared in our pass
        // after the restore draw: ours, not a truth pass.)
        drop_held = hc.valid && hc.rect[0] >= 0 && hc.rect[1] >= 0 && hc.rect[2] <= int32_t(p.host_width) &&
                    hc.rect[3] <= int32_t(p.host_height) && (owned_held_.depth < 0 || owned_held_.depth == p.depth_slot) &&
                    (!hc.stencil_touched || (hc.stencil && !draw.raster.stencil && owned_held_.depth == p.depth_slot)) &&
                    (draw.color_write_mask & 0xF) == 0xF && draw.raster.depth_write;
        if (!drop_held) {
          static uint32_t logged_keep = 0;  // (Session 87 diagnosis: which condition keeps the clears)
          if (logged_keep < 12) {
            ++logged_keep;
            REXGPU_INFO("[native][owned] frame {}: restore over base {} pitch {} keeps its held clears: valid {} rect {},{}-{},{} "
                        "box {}x{} held depth {} pass depth {} stencil {} mask {:X} depth write {}",
                        frame_, p.base, p.pitch_tiles, hc.valid, hc.rect[0], hc.rect[1], hc.rect[2], hc.rect[3], p.host_width,
                        p.host_height, owned_held_.depth, p.depth_slot, hc.stencil_touched, draw.color_write_mask & 0xF,
                        draw.raster.depth_write);
          }
          if (OwnedOnly()) {
            // Session 92, M2a (review D6): a flush here would switch the pass under the draw (a new pass of the held clears' surface). The restore rewrites
            // colour and depth over the box: the held clears are dropped, counted, their stencil clear applied in the pass after the draw (the drop_held branch).
            g_m2a.held_dropped += uint32_t(owned_held_.draws.size());
            drop_held = true;
            forced_drop = true;
            // (The in-pass stencil clear only where the original condition's box and stencil parts hold - else the clears are just dropped: restore_stencil_lost.)
            forced_stencil_ok = hc.valid && !p.half && hc.rect[0] >= 0 && hc.rect[1] >= 0 && hc.rect[2] <= int32_t(p.host_width) &&
                                hc.rect[3] <= int32_t(p.host_height) && owned_held_.depth == p.depth_slot && hc.stencil && !draw.raster.stencil;
          } else {
            OwnedFlushHeld();
          }
        }
      }
      if (p.chain_unkeyed) {  // (Session 87: a pass begun by a clear is keyed by its first draw's pixel shader, as a pass started by a draw is)
        p.key = OwnedKeyOf(draw);
        p.chain_unkeyed = false;
      }
      if (OwnedRecord(draw, rect)) {
        if (restore_recorded) {
          ++owned_frame_stats_.restore_drawn;
          if (drop_held) {
            const ClearsInfo& hc = owned_held_.info;
            if (hc.stencil_touched && p.open && (!forced_drop || forced_stencil_ok)) {
              VkClearAttachment clear = {};
              clear.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
              clear.clearValue.depthStencil = {0.0f, uint32_t(hc.stencil_value)};
              VkClearRect clear_rect = {{{hc.rect[0], hc.rect[1]},
                                         {uint32_t(hc.rect[2] - hc.rect[0]), uint32_t(hc.rect[3] - hc.rect[1])}},
                                        0, 1};
              if (p.dres) {  // (Session 94: the pass image is a fraction of the box)
                const int32_t iw = int32_t(owned_images_[p.image].width), ih = int32_t(owned_images_[p.image].height);
                const int32_t x0 = std::clamp(int32_t(std::floor(double(hc.rect[0]) * p.ratio_x)), 0, iw), y0 = std::clamp(int32_t(std::floor(double(hc.rect[1]) * p.ratio_y)), 0, ih);
                const int32_t x1 = std::clamp(int32_t(std::ceil(double(hc.rect[2]) * p.ratio_x)), x0, iw), y1 = std::clamp(int32_t(std::ceil(double(hc.rect[3]) * p.ratio_y)), y0, ih);
                clear_rect.rect = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
              }
              command_processor_.deferred_command_buffer().CmdVkClearAttachments(1, &clear, 1, &clear_rect);
              ++owned_frame_stats_.restore_stencil_cleared;
            } else if (hc.stencil_touched) {
              ++owned_frame_stats_.restore_stencil_lost;  // (not expected: a recorded draw leaves our pass open; counted, visible)
            }
            owned_held_.draws.clear();
            owned_held_.info = ClearsInfo();
            ++owned_frame_stats_.restore_clears_dropped;
          }
        }
        return true;
      }
      return not_taken(true, __LINE__);  // (a committed pass that cannot go on ended: this draw is the truth path's)
    }
    // (Session 82, the owned frame: every draw is a candidate, so a draw elsewhere is not "a listed draw outside its pass"
    // - the pending pass ends (is replayed) only when this draw would start a pass of its own; one that could not is
    // refused below, and the refusal replays the pass only if its tiles are under it.)
    bool starts = true;
    if (frame) {
      const bool held0 = !owned_held_.draws.empty();
      const auto& pc0 = held0 ? owned_held_.info : truth_pass_clears_;
      const int16_t color0 = held0 ? owned_held_.color : truth_pass_color_;
      const int16_t depth_slot0 = held0 ? owned_held_.depth : truth_pass_depth_;
      const bool box0 = (held0 || truth_pass_open_ ||
                         (owned_clears_ended_ && owned_clears_submission_ == command_processor_.GetCurrentSubmission())) &&
                        color0 == draw.truth_color && pc0.valid && pc0.rect[0] == 0 &&
                        pc0.rect[1] == 0 && pc0.rect[2] > 0 && pc0.rect[3] > 0 && pc0.rect[2] <= 4096 &&
                        pc0.rect[3] <= 4096 && pc0.rect[2] % int32_t(scale_x) == 0 &&
                        pc0.rect[3] % int32_t(scale_y) == 0 &&
                        pc0.area == uint64_t(pc0.rect[2]) * uint64_t(pc0.rect[3]) && rect[0] >= 0 && rect[1] >= 0 &&
                        rect[2] <= pc0.rect[2] && rect[3] <= pc0.rect[3] && (!rgba16 || pc0.quantized);
      const bool depth0 = box0 && pc0.uses_depth && pc0.depth && depth_slot0 >= 0;
      const bool cleared0 = box0 && (!depth_used || (depth0 && draw.truth_depth == depth_slot0)) &&
                            (!draw.raster.stencil || (depth0 && pc0.stencil)) &&
                            (!covered || (pc0.rect[2] == rect[2] && pc0.rect[3] == rect[3]));
      const OwnedPassKey key0 = OwnedKeyOf(draw);
      starts = (covered || cleared0 || OwnedConvertible(draw, rect, depth_used, scale_x, scale_y)) &&
               !OwnedPaused(key0) &&
               !OwnedPresentKeyed(key0);
    }
    if (starts) OwnedParkOrReplay("a listed draw outside its pass before its copy");
  }
  const OwnedPassKey key = OwnedKeyOf(draw);
  // A new pass. From a covering draw (proven coverage: every pixel of its rectangle from the surface's origin drawn,
  // unblended, all channels, no depth): our image is exactly that rectangle, not loaded. Or from the clears of the
  // open truth pass into the same images (only clears so far, of one known colour, their union a box from the
  // surface's origin cleared whole - and depth / stencil whole to one value where this draw uses them): our image is
  // that box, cleared as they cleared it (the colour rounded as the target stores it - as gpu_native_truth_clear_load:
  // 1/255), so it holds what the surface holds under the pass.
  const bool held = frame && !owned_held_.draws.empty();
  const auto& pc = held ? owned_held_.info : truth_pass_clears_;
  const int16_t clears_color_slot = held ? owned_held_.color : truth_pass_color_;
  const int16_t clears_depth_slot = held ? owned_held_.depth : truth_pass_depth_;
  const bool clears_live = held || truth_pass_open_ || (frame && owned_clears_ended_ &&
                                                        owned_clears_submission_ == command_processor_.GetCurrentSubmission());
  const bool clears_box = clears_live && clears_color_slot == draw.truth_color && pc.valid && pc.rect[0] == 0 &&
                          pc.rect[1] == 0 && pc.rect[2] > 0 && pc.rect[3] > 0 && pc.rect[2] <= 4096 &&
                          pc.rect[3] <= 4096 && pc.rect[2] % int32_t(scale_x) == 0 &&
                          pc.rect[3] % int32_t(scale_y) == 0 &&
                          pc.area == uint64_t(pc.rect[2]) * uint64_t(pc.rect[3]) && rect[0] >= 0 && rect[1] >= 0 &&
                          rect[2] <= pc.rect[2] && rect[3] <= pc.rect[3] && (!rgba16 || pc.quantized);
  const bool keep_depth = clears_box && pc.uses_depth && pc.depth && clears_depth_slot >= 0;
  const bool keep_stencil = keep_depth && pc.stencil;
  // (Held clears are taken only whole: nothing of them is left in the truth path to run when our pass is not theirs.)
  const bool held_whole = !held || ((!pc.uses_depth || keep_depth) && (!pc.stencil_touched || keep_stencil));
  const bool cleared = clears_box && (!depth_used || (keep_depth && draw.truth_depth == clears_depth_slot)) &&
                       (!draw.raster.stencil || keep_stencil) &&
                       (!covered || (pc.rect[2] == rect[2] && pc.rect[3] == rect[3])) && held_whole;
  if (restore_from_held && !cleared) return refuse("kind", kOwnedReasonOther, "restore");  // (Session 92: the start below is the held clears' or nothing)
  // (Session 95, review r5, 1: the whole window, not the quiet gate: a drop is decided at the pass's first draw and its copy's read-back only at the copy. A read-back on a dropped pass has nothing to
  // give the CPU: the Heenan noise. Frames with no read-back still lose the dead segments, at the frame's end, when the frame's read-backs are known: owned_graph.cpp.)
  if (OwnedChains() && frame && !draw.fast_clear && !command_processor_.NativeReadbackActive() && OwnedKeyDead(key)) {
    // (Session 87, chains: Bryan's shadow-map passes (0/7/0 ps DA494C93) are dead - no sampler reads their copies - and their depth-tested draws cannot start a pass
    // from colour-only clears. A dead pass is dropped whole, as the cleared case above drops it: the first draw with its clears, then the pass's other draws
    // until its copy.)
    const bool depthless_first = held && !cleared && clears_box && depth_used && !pc.uses_depth && !pc.stencil_touched && held_whole;
    const bool same_pass = owned_drop_.valid && owned_drop_.base == base && owned_drop_.pitch == pitch && owned_drop_.key == key;
    // (review A5: a dead pass's depth writes are dropped only when nothing else holds that depth surface - no pass of ours, dirty depth tiles or parked pass.)
    const auto depth_slot_free = [&](int16_t slot) {
      if (slot < 0) return true;
      if (owned_pending_.valid && owned_pending_.depth && owned_pending_.depth_slot == slot) return false;
      for (const OwnedPass& q : owned_parked_) {
        if (q.depth && q.depth_slot == slot) return false;
      }
      for (const OwnedPass& q : owned_restorable_) {
        if (q.depth && q.depth_slot == slot) return false;
      }
      for (const OwnedDirty& d : owned_dirty_) {
        if (d.slot == slot && (d.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) return false;
      }
      // (Failure read fix 2: under the latch a long-lived record does not block the dead drop - the tile overlap that released the dirty entries is gone; the
      // dead drop marks the slot stale instead, below.)
      if (!OwnedOnly()) {
        for (const OwnedPass& q : owned_newest_) {
          if (q.depth && !q.rec_depth_gone && q.depth_slot == slot) return false;
        }
      }
      return true;
    };
    const bool depth_ok = !(draw.raster.depth_write || draw.raster.stencil) || depth_slot_free(draw.truth_depth);
    if (depth_ok && (depthless_first || same_pass)) {
      if (depthless_first) {
        owned_held_.draws.clear();
        owned_held_.info = ClearsInfo();
        owned_drop_.valid = true;
        owned_drop_.base = base;
        owned_drop_.pitch = pitch;
        owned_drop_.key = key;
      }
      owned_drop_.frame = frame_;
      serve_pending_.clear();
      serve_rename_.valid = false;
      if (OwnedOnly() && (draw.raster.depth_write || draw.raster.stencil)) OwnedMarkDepthStale(draw.truth_depth);
      ++owned_frame_stats_.dropped_draws;
      ++owned_frame_stats_.ch_dead_depthless;
      return true;
    }
  }
  // (Held clears this draw does not take go into the truth path now, before the pass it starts - as they always were.)
  if (held && !cleared) {
    owned_flush_note_ = fmt::format("not cleared: clears_box {} (valid {} origin {} whole {} fits {}) depth_used {} keep_depth {} stencil {} covered {} held_whole {} rgba16 {} quantized {}",
                                    clears_box, pc.valid, pc.rect[0] == 0 && pc.rect[1] == 0, pc.area == uint64_t(pc.rect[2]) * uint64_t(pc.rect[3]),
                                    rect[2] <= pc.rect[2] && rect[3] <= pc.rect[3], depth_used, keep_depth, draw.raster.stencil, covered, held_whole,
                                    rgba16, pc.quantized);
    OwnedFlushHeld(true);
    if (OwnedChains() && p.valid && p.chain_pass && fits_of(p) && !(p.chain_unkeyed && (OwnedPaused(key) || OwnedPresentKeyed(key)))) {  // (the clears' pass took them and holds this draw too; review F5: not a draw whose own key is paused / front-buffer's)
      if (p.chain_unkeyed) {
        p.key = key;
        p.chain_unkeyed = false;
      }
      if (OwnedRecord(draw, rect)) return true;
      return not_taken(true, __LINE__);
    }
    // (review F1: a draw that does not join the pass just begun must not start its own over it: the pass is parked first.)
    if (OwnedChains() && p.valid && p.chain_pass && p.chain_unkeyed) OwnedParkOrReplay("a clear's pass began; this draw is of another surface");
  }
  if (!covered && !cleared) {
    // (Session 82: the extension point for the starts the base cases do not give - resumed, converted.)
    if (frame) {
      if (OwnedPaused(key)) return refuse("paused", kOwnedReasonPaused);
      if (OwnedPresentKeyed(key)) {
        return refuse("front buffer", kOwnedReasonPresent);
      }
      if (p.valid) OwnedParkOrReplay("a listed draw outside its pass before its copy");  // (before an extension starts)
      OwnedStartContext ctx;
      ctx.draw = &draw;
      ctx.rect = rect;
      ctx.scale_x = scale_x;
      ctx.scale_y = scale_y;
      if (OwnedStartExtra(ctx)) {
        if (owned_pending_.census < 0) owned_pending_.census = OwnedFrameNewPass(draw);
        if (OwnedRecord(draw, rect)) return true;
        return not_taken(true, __LINE__);
      }
      return refuse("coverage", ctx.reason, ctx.detail.empty() ? nullptr : ctx.detail.c_str());
    }
    return refuse("coverage", kOwnedReasonStart);
  }
  if (frame) {
    if (OwnedPaused(key)) return refuse("paused", kOwnedReasonPaused);
    if (OwnedPresentKeyed(key)) {
      return refuse("front buffer", kOwnedReasonPresent);
    }
  }
  const uint32_t width = uint32_t(cleared ? pc.rect[2] : rect[2]), height = uint32_t(cleared ? pc.rect[3] : rect[3]);
  // The clears' truth pass: its range, kept until the commit (a replay lands on its cleared tiles), turned off there
  // when it wrote nothing our image does not hold (depth / stencil only whole - our pass clears them as it did).
  size_t clears_begin = SIZE_MAX, clears_draws = SIZE_MAX, clears_end_cmd = SIZE_MAX, clears_end = SIZE_MAX;
  uint64_t clears_serial = 0;
  const bool clears_loaded = cleared && !held && truth_pass_loaded_;
  if (cleared && !held) {
    clears_begin = truth_pass_begin_offset_;
    clears_serial = truth_pass_begin_serial_;
    clears_draws = truth_pass_draws_offset_;
  }
  // (Session 82: a pass no sampler ever read - its clears are the ones held for it: the draw, the clears and the copy that
  // follows are dropped. Never one that would not start an owned pass: it gets here only after every refusal above.)
  if (frame && cleared && held && held_whole && OwnedKeyDead(key) && !command_processor_.NativeReadbackActive()) {  // (review r5, 1: as above)
    owned_held_.draws.clear();
    owned_held_.info = ClearsInfo();
    owned_drop_.valid = true;
    owned_drop_.base = base;
    owned_drop_.pitch = pitch;
    owned_drop_.key = key;
    owned_drop_.frame = frame_;
    serve_pending_.clear();
    serve_rename_.valid = false;
    ++owned_frame_stats_.dropped_draws;
    return true;
  }
  if (frame && cleared && OwnedKeyDead(key)) {
    // (Session 87, step 3 diagnosis: a dead key whose pass was drawn anyway - why.)
    static uint32_t dead_not_dropped_logged = 0;
    if (dead_not_dropped_logged < 12) {
      ++dead_not_dropped_logged;
      REXGPU_INFO("[native][owned-dead] frame {}: dead key ps {:016X} surface {}/{} drawn anyway: held {} whole {} readback window {}",
                  frame_, key.ps, key.base, key.pitch, held, held_whole, command_processor_.NativeReadbackActive());
    }
  }
  // Session 94 (gpu_native_owned_draw_ratio): the ratio is fixed at the first owned pass of a frame (never two ratios in one frame). The scene
  // pass (cleared with depth, RGBA8, the chosen resolution's size: 1280 x 720 guest pixels) is drawn into images of that fraction of its size.
  uint32_t dres_w = width, dres_h = height;
  bool dres = false;
  if (frame) {
    if (g_dres.frame != frame_) {
      g_dres.frame = frame_;
      // (Session 95, run 6561: the resolution rows under TRUE NATIVE - the row's height is the base, dynamic resolution's steps a fraction of it.)
      double r = std::clamp(REXCVAR_GET(gpu_native_owned_draw_ratio) * OwnedRowRatio(scale_x, scale_y), 0.5, 1.0);
      if (r > 0.999) r = 1.0;
      if (r != g_dres.ratio) {
        REXGPU_INFO("[native] owned draw ratio {:.3f} (frame {}; scaled surfaces {}, blits {} + {} depth, refused {} at the ratio before)", r, frame_, g_dres.scaled, g_dres.blits,
                    g_dres_depth_blits, g_dres.refused);
        g_dres.ratio = r;
        g_dres.scaled = g_dres.blits = g_dres.refused = 0;
        g_dres_depth_blits = 0;
      }
      if (r < 1.0 && !g_dres.announced) {
        g_dres.announced = true;
        REXGPU_INFO("[native] owned draw ratio on (frame {}, ratio {:.3f}): the scene pass is drawn smaller and brought up by a blit at its copies", frame_, r);
      }
      ++g_dres.hist[std::min<size_t>(20, size_t(std::lround(r * 20.0)))];
      ++g_dres.frames_hist;
      // (At most two ratios resident: the sizes beyond the newest two are retired as soon as nothing holds their images.)
      while (g_dres.sizes.size() > 2) {
        const auto old = g_dres.sizes.front();
        bool left = false;
        for (size_t i = 0; i < owned_images_.size(); ++i) {
          const OwnedImage& oi = owned_images_[i];
          if (!oi.width || oi.width != old.first || oi.height != old.second) continue;
          if (OwnedImageHeld(int16_t(i))) left = true; else OwnedRetireImage(i);
        }
        if (left) break;  // (tried again at the next frame)
        // (Session 94: the size's scratch depth goes with its images - the slot is nulled, the indices of the others stay; freed through retired_.)
        for (OwnedScratchDepth& sd : owned_scratch_depths_) {
          if (!sd.width || sd.width != old.first || sd.height != old.second) continue;
          Retired dead;
          dead.submission = command_processor_.GetCurrentSubmission();
          dead.images[0] = sd.image;
          dead.memory[0] = sd.memory;
          dead.views[0] = sd.view;
          retired_.push_back(dead);
          owned_bytes_ -= std::min<uint64_t>(owned_bytes_, uint64_t(sd.width) * sd.height * 8);
          sd = OwnedScratchDepth();
        }
        g_dres.sizes.erase(g_dres.sizes.begin());
      }
      if (g_dres.announced && g_dres.frames_hist - g_dres.last_summary >= 600) {
        g_dres.last_summary = g_dres.frames_hist;
        std::string h;
        for (size_t i = 0; i <= 20; ++i) {
          if (g_dres.hist[i]) h += fmt::format(" {}%:{}", i * 5, g_dres.hist[i]);
        }
        // (Session 95: the counts since the ratio last changed - a row or a step that reads "refused" here is not saving anything.)
        REXGPU_INFO("[native] owned draw ratio histogram (frames at each ratio, 600 frames):{}; resident scaled sizes {}; scaled passes {}, colour blits {}, depth copies up {}, refused {}", h,
                    g_dres.sizes.size(), g_dres.scaled, g_dres.blits, g_dres_depth_blits, g_dres.refused);
        std::memset(g_dres.hist, 0, sizeof(g_dres.hist));
      }
    }
    if (g_dres.ratio < 0.999 && cleared && keep_depth && draw.truth_class == 1 && width == 1280 * scale_x && height == 720 * scale_y &&
        OwnedOnly()) {
      dres_w = std::max(16u, uint32_t(std::lround(double(width) * g_dres.ratio)));
      dres_h = std::max(16u, uint32_t(std::lround(double(height) * g_dres.ratio)));
      dres = dres_w < width || dres_h < height;
      if (dres && g_dres.depth_blit_ok < 0) {
        // (Session 94: the depth blit-up needs BLIT_SRC | BLIT_DST on the depth format - AMD's Windows driver has none for D32S8, native_truth.cpp.)
        const ui::vulkan::VulkanDevice* vd = command_processor_.GetVulkanDevice();
        VkFormatProperties fp = {};
        vd->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(vd->physical_device(), post_ds_format_, &fp);
        constexpr VkFormatFeatureFlags kBlit = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
        const bool can_blit = (fp.optimalTilingFeatures & kBlit) == kBlit;
        // (Session 95: Adreno 830 and AMD on Windows have no depth blit - there, our fragment pass, one layer only (gpu_d3d_decouple's layers are not its job).)
        const int32_t pass_mode = REXCVAR_GET(gpu_native_owned_dres_depth_pass);
        owned_depth_up_on_ = (pass_mode == 2 || (pass_mode == 1 && !can_blit)) && OwnedLayers() == 1 && EnsureOwnedDepthUpObjects();
        g_dres.depth_blit_ok = can_blit || owned_depth_up_on_ ? 1 : 0;
        if (owned_depth_up_on_) {
          REXGPU_INFO("[native][owned-dres] the scene pass's depth goes up by the fragment pass ({})", can_blit ? "gpu_native_owned_dres_depth_pass 2" : "the depth format cannot be blitted here");
        } else if (!can_blit) {
          REXGPU_WARN("[native][owned-dres] the depth format cannot be blitted and the fragment pass is not available: no scaling - resolution rows and dynamic resolution draw the whole size");
        }
      }
      // (A read-back window needs the pass's image written to guest memory: refused for a scaled pass - no scaling then. Item: the scaled images AND the
      // full-size one (colour + depth each) must fit the budget together.)
      if (dres && (!g_dres.depth_blit_ok || command_processor_.NativeReadbackActive() ||
                   !OwnedImageAvailable(width, height, draw.truth_class, true) ||
                   !OwnedImageAvailable(dres_w, dres_h, draw.truth_class, keep_depth))) {
        dres = false;  // (this pass at the full size, counted)
        ++g_dres.refused;
        dres_w = width, dres_h = height;
      }
    }
  }
  // (Review fix 7: the pool / budget checked before anything of the truth pass ends - a failure after it would count
  // the pass twice.)
  if (frame && !dres && !OwnedImageAvailable(width, height, draw.truth_class, keep_depth)) {
    OwnedNoteRefusal(kOwnedReasonPool, "no image within the budget");
    return not_taken(true, __LINE__);
  }
  EndTruthPass();  // (the clears' pass ends here: our pass is a pass of its own)
  owned_clears_ended_ = false;  // (consumed by this pass, or touched by it)
  if (cleared && !held) {
    clears_end_cmd = truth_pass_end_cmd_offset_;
    clears_end = truth_pass_end_offset_;
  }
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  const bool post_half = frame && OwnedPostHalfWanted(draw, covered, cleared, scale_x, scale_y, width, height, keep_depth, pitch);
  if (dres && post_half) dres = false;  // (not expected: the post chain's gate needs a covered start)
  int16_t dres_full = -1;
  if (dres) {
    // (Session 94: the pass's one full-size image, made now - held through the pass (OwnedImageHeld) so no budget surprise comes at its copies.)
    dres_full = OwnedImageFor(width, height, draw.truth_class);
    if (dres_full < 0 || !OwnedEnsureDepth(dres_full)) {
      dres = false;
      dres_full = -1;
      ++g_dres.refused;
    } else {
      owned_images_[dres_full].depth_read = false;
      owned_pending_.dres_full = dres_full;
    }
  }
  const int16_t index = dres ? OwnedImageFor(dres_w, dres_h, draw.truth_class)
                             : OwnedImageFor(post_half ? width / 2 : width, post_half ? height / 2 : height, draw.truth_class);
  if (index < 0) {  // (all of this size held, or one could not be made: owned_failed_ / the budget)
    if (frame) OwnedNoteRefusal(owned_failed_ ? kOwnedReasonFailed : kOwnedReasonPool);
    owned_pending_.dres_full = -1;
    return not_taken(true, __LINE__);
  }
  if (dres) {
    const std::pair<uint32_t, uint32_t> sz{dres_w, dres_h};
    if (std::find(g_dres.sizes.begin(), g_dres.sizes.end(), sz) == g_dres.sizes.end()) g_dres.sizes.push_back(sz);
    ++g_dres.scaled;
  }
  p.valid = true;
  p.half = post_half || dres;
  p.dres = dres;
  p.dres_full = dres ? dres_full : int16_t(-1);
  p.dres_color_used = p.dres_depth_used = false;
  p.ratio_x = dres ? float(dres_w) / float(width) : 0.5f;
  p.ratio_y = dres ? float(dres_h) / float(height) : 0.5f;
  p.kind = draw.truth_class;
  p.is_64bpp = is_64bpp;
  p.committed = p.started = false;
  p.carried_defined = false;  // (Session 90)
  p.open = false;
  p.cleared = cleared;
  p.depth = keep_depth;
  p.stencil = keep_stencil;
  p.draws_depth = p.draws_stencil = false;
  p.start = !cleared ? uint32_t(kOwnedPassCover) : keep_depth ? uint32_t(kOwnedPassClearBoth)
                                                               : uint32_t(kOwnedPassClearColor);
  p.clear_values[0] = p.clear_values[1] = VkClearValue();
  if (cleared) {
    for (uint32_t i = 0; i < 4; ++i) {
      // (Review: the product its own statement, so no compiler fuses it with the add - the GPU's is unfused.)
      // (16-bit: the 360's -32..32 in the SNORM image, as gpu_native_truth_clear_load rounds it.)
      const float c = pc.color[i];
      if (rgba16) {
        const float s = std::min(std::max(c, -32.0f), 32.0f) * (32767.0f / 32.0f);
        p.clear_values[0].color.float32[i] = std::trunc(s + (s < 0.0f ? -0.5f : 0.5f)) / 32767.0f;
      } else {
        const float s = std::min(std::max(c, 0.0f), 1.0f) * 255.0f;
        p.clear_values[0].color.float32[i] = std::trunc(s + 0.5f) / 255.0f;
      }
    }
    p.clear_values[1].depthStencil = {pc.depth_value, keep_stencil ? uint32_t(pc.stencil_value) : 0u};
  }
  p.base = base;
  p.pitch_tiles = pitch;
  p.color_slot = draw.truth_color;
  p.depth_slot = keep_depth ? clears_depth_slot : int16_t(-1);
  p.width = width / scale_x;
  p.height = height / scale_y;
  p.host_width = width;
  p.host_height = height;
  p.image = index;
  p.frame = frame_;
  p.last_tick = ++owned_tick_;  // (Session 92, M2a)
  p.rec_colour_gone = p.rec_depth_gone = false;
  p.draws_since_color = p.draws_since_depth = 0;
  p.copy_color_page = p.copy_depth_page = 0;
  p.submission = command_processor_.GetCurrentSubmission();
  p.tiles = OwnedOnly() ? std::bitset<2048>() : EdramTiles(base, pitch, is_64bpp, 0, 0, p.width, p.height);
  if (keep_depth && !OwnedOnly() && size_t(p.depth_slot) < OwnedSlotCount()) {  // (Session 92, M2b: no tile bookkeeping under the latch)
    p.tiles |= EdramTiles(OwnedSlotKey(p.depth_slot).base, pitch, false, 0, 0, p.width, p.height);
  }
  p.draws.clear();
  p.crowd_calls.clear();
  p.serves.clear();
  p.producers.clear();
  p.clears_begin = clears_begin;
  p.clears_serial = clears_serial;
  p.clears_draws = clears_draws;
  p.clears_end_cmd = clears_end_cmd;
  p.clears_end = clears_end;
  p.clears_loaded = clears_loaded;
  p.clears_ours = cleared && (!pc.uses_depth || keep_depth) && (!pc.stencil_touched || keep_stencil);
  p.clears_depth = cleared && pc.uses_depth;
  p.clears_stencil = cleared && pc.stencil_touched;
  p.clears_submission = command_processor_.GetCurrentSubmission();
  // (Held clears this pass takes: never recorded in the truth path - kept with the pass, replayed first if it is lost.)
  p.clears_held = false;
  p.held_clears.clear();
  if (cleared && held) {
    OwnedPendRetryDone(owned_held_.color);  // (review fix 5)
    p.held_clears = std::move(owned_held_.draws);
    owned_held_.draws.clear();
    owned_held_.info = ClearsInfo();
    p.clears_held = true;
  }
  p.key = key;
  // (Hard rule 6: another surface's tiles in the same EDRAM place do not end a parked pass - only the same surface starting again.)
  // (Older restorable passes of the surface go first, then the drop: under chains the drop keeps the pass the restore draw that follows needs - the
  // crowd atlas pass draws over the scene's surface between the scene's copies and the restore from them, run 21.)
  owned_restorable_.erase(std::remove_if(owned_restorable_.begin(), owned_restorable_.end(), [&](const OwnedPass& q) {
    return q.base == p.base && q.pitch_tiles == p.pitch_tiles && q.kind == p.kind;
  }), owned_restorable_.end());
  OwnedDropParkedIf("the same surface starts again", [&](const OwnedPass& q) {
    return q.base == p.base && q.pitch_tiles == p.pitch_tiles && q.kind == p.kind;  // (review F7's drop by depth slot is NOT done: it dropped the scene pass the restore draw needs, run 12)
  });
  if (keep_depth) p.depth_tick = p.last_tick;  // (Session 92, M2a)
  OwnedSupersedeRecords(p.base, p.pitch_tiles, p.kind, p.depth ? p.depth_slot : int16_t(-1));  // (review D4)
  p.loaded_start = false;
  p.edram_free = false;
  p.start_kind = cleared ? uint8_t(kOwnedStartCleared) : uint8_t(kOwnedStartCovered);
  p.chain_pass = p.chain_unkeyed = p.chain_parked = false;  // (review F6; follow-up: chain_parked too)
  p.carried_defined = false;  // (Session 90)
  p.census = -1;
  if (frame) {
    // (Session 82: the census record - the clears' truth pass it started from, or a new one.)
    int32_t rec = -1;
    if (cleared && !held && owned_frame_last_ >= 0 && size_t(owned_frame_last_) < owned_frame_passes_.size()) {
      const OwnedFramePass& last = owned_frame_passes_[owned_frame_last_];
      if (!last.owned && !last.lost && last.start == kOwnedStartNone) rec = owned_frame_last_;
    }
    if (rec < 0) rec = OwnedFrameNewPass(draw);
    OwnedFramePass& r = owned_frame_passes_[rec];
    r.start = p.start_kind;
    r.reason = kOwnedReasonNone;
    r.detail.clear();
    r.box_w = p.width;
    r.box_h = p.height;
    p.census = rec;
  }
  ++truth_counts_.owned_passes;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned pass: {} start, our image {} ({}x{}), colour {} depth {} ({}{}), clears' "
                "pass {} (loaded {}), clear 0x{:08X}/{:.6f}/{}",
                frame_, cleared ? "cleared" : "covered", index, width, height, p.color_slot, p.depth_slot,
                keep_depth ? "depth kept" : "no depth", keep_stencil ? " + stencil" : "",
                p.clears_ours ? "ours" : "kept", clears_loaded, draw.color_info, pc.depth_value, pc.stencil_value);
  }
  if (OwnedRecord(draw, rect)) {
    if (restore_from_held) {
      ++g_s92.restores_taken;  // (Session 92)
      ++owned_frame_stats_.restore_drawn;
    }
    return true;
  }
  return not_taken(true, __LINE__);  // (Session 82: no depth image within the budget - the pass was dropped before it began)
}

bool NativeRenderer::OwnedContinues(const OwnedPass& p, uint32_t copy_index) const {
  // (Session 82: per copy - the last copy of a pass that goes on after its first one is the image itself, no snapshot.)
  return std::any_of(owned_continued_.begin(), owned_continued_.end(), [&](const OwnedContinued& c) {
    return c.base == p.base && c.pitch_tiles == p.pitch_tiles && c.kind == p.kind && c.width == p.width &&
           c.height == p.height && ((c.after >> std::min(copy_index, 31u)) & 1u);
  });
}

void NativeRenderer::OwnedNoteContinued(const OwnedPass& p) {
  for (OwnedContinued& c : owned_continued_) {
    if (c.base == p.base && c.pitch_tiles == p.pitch_tiles && c.kind == p.kind && c.width == p.width &&
        c.height == p.height) {
      c.frame = frame_;
      c.after |= 1u << std::min(p.copy_count, 31u);
      return;
    }
  }
  owned_continued_.push_back({p.base, p.pitch_tiles, p.kind, p.width, p.height, frame_, 1u << std::min(p.copy_count, 31u)});
}

void NativeRenderer::OwnedEndCommitted(const char* why) {
  OWNED_SCENE_LOG(owned_pending_, "[native][owned][scene] frame {}: pass base {} pitch {} {}x{} ENDS (committed {}, copies {}, drawn since copy {}): {}",
                  frame_, owned_pending_.base, owned_pending_.pitch_tiles, owned_pending_.host_width, owned_pending_.host_height,
                  owned_pending_.committed, owned_pending_.copy_count, owned_pending_.draws_since_color + owned_pending_.draws_since_depth, why);
  // A committed pass something else touches (or the frame ends): it simply ends - its tiles are dirty, our image
  // holds them; nothing to replay.
  OwnedPass& p = owned_pending_;
  OwnedEndPass();
  if (OwnedOnly() && p.valid) {
    // Session 92, M2a (review D2): every pass that ends is noted; a newer noted pass of its surface (or its depth slot) supersedes it by tick.
    OwnedNoteNewest(p);
  }
  p.valid = false;
  p.committed = false;
  p.draws.clear();
  p.crowd_calls.clear();
  p.serves.clear();
}

void NativeRenderer::OwnedEndAsIs(const char* why) {
  // Session 92, M2a: a pending pass whose copy did not follow (a loss before: replayed into the truth images). Nothing to replay into - its pixels are
  // in our image: committed (no tiles marked) and ended, its image the surface's newest record. Counted by why.
  OwnedPass& p = owned_pending_;
  if (!p.valid) return;
  if (!p.committed) {
    ++g_m2a.replays_avoided;
    M2aGroup(g_m2a.avoided_why, why);
    OwnedCommit();
  }
  OwnedEndCommitted(why);
}

void NativeRenderer::OwnedMarkDirty(bool dropped) {
  if (OwnedOnly()) return;  // (Session 92, M2a: no truth image gets dirty tiles - the surface's newest record is the pass's image)
  // The truth images' tiles of the box: without its draws (and, dropped, its clears) - dirty, ours to copy in before
  // anything reads them. A pass from a covering draw too when its colour image owns every tile of the box (its draws
  // write every pixel of the box: our image is what the surface holds there); else its tiles are left as they were -
  // other images' (taking them would need their pixels outside the box: an alias load). Merged into an entry of the
  // same image and aspect (a pass that resumed after a copy marks its box again).
  OwnedPass& p = owned_pending_;
  if (p.half) return;  // (look lever: our image is half size - never copied into the truth images)
  const auto box = [&](const TruthImage& t) {
    return EdramTiles(t.base, t.pitch_tiles, t.is_64bpp(), 0, 0, p.width, p.height);
  };
  if (p.edram_free) return;  // (no truth image holds anything of it: nothing dirty)
  if (!OwnedTruthHolds()) return;
  const auto add = [&](int16_t slot, VkImageAspectFlags aspects) {
    if (slot < 0 || size_t(slot) >= truth_images_.size() || !aspects) return;
    const TruthImage& t = truth_images_[slot];
    const std::bitset<2048> tiles = box(t) & t.owned;
    if (tiles.none()) return;
    // (Review: another image's dirty tiles here - a pass from a covering draw cleans nothing before it, its draws are
    // not the truth path's: those go in first, so ours, copied in later, land over them in the right order.)
    for (OwnedDirty& e : owned_dirty_) {
      if (e.slot == slot && e.aspects == aspects && e.image != p.image && (e.tiles & tiles).any()) {
        // (Session 82, speed: the same surface, and this pass wrote its whole box - cleared, or drawn all over - which
        // holds the older pass's box: its pixels in these tiles are overwritten, not copied in first. Colour; depth /
        // stencil when this pass cleared them whole.)
        // (A pass of whole tiles - P2's case, the tiles of an aligned box - overwrites whatever an older colour image left in
        // them, whatever that image's box was.)
        const bool aligned = p.width % (t.is_64bpp() ? 40u : 80u) == 0 && p.height % 16u == 0;
        // (Session 82: the same surface - base and pitch - and per axis: the new box is as large as the old one, or its edge is a
        // tile edge: every tile it overlaps has the old pass's pixels inside the new box only. A 512x256 pass over a 512x512 one of
        // the same surface overwrites the old pixels in its 112 tiles: nothing to copy in.)
        const bool same_surface = e.base == t.base && e.pitch_tiles == t.pitch_tiles && e.kind == t.kind;
        const bool axes_ok = same_surface &&
                             (p.host_width >= e.host_width || p.width % (t.is_64bpp() ? 40u : 80u) == 0) &&
                             (p.host_height >= e.host_height || p.height % 16u == 0);
        const bool overwritten =
            (p.cleared || p.start_kind == kOwnedStartCovered) &&
            ((aspects & VK_IMAGE_ASPECT_COLOR_BIT) ? ((p.host_width >= e.host_width && p.host_height >= e.host_height) || aligned || axes_ok)
             : (p.host_width >= e.host_width && p.host_height >= e.host_height) &&
                   ((aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? (p.cleared && p.depth) : (p.cleared && p.stencil)));
        if (overwritten) {
          owned_frame_stats_.dead_dropped += uint32_t((e.tiles & tiles).count());
          e.tiles &= ~tiles;
        } else {
          owned_mat_why_ = "a later pass over the same tiles";
          OwnedMaterialize(e, e.tiles & tiles);
        }
      }
    }
    for (OwnedDirty& e : owned_dirty_) {
      if (e.slot == slot && e.aspects == aspects && e.image == p.image) {
        e.tiles |= tiles;
        e.frame = frame_;
        return;
      }
    }
    OwnedDirty e;
    e.slot = slot;
    e.base = t.base;
    e.pitch_tiles = t.pitch_tiles;
    e.kind = t.kind;
    e.format = t.format;
    e.image = p.image;
    e.aspects = aspects;
    e.host_width = p.host_width;
    e.host_height = p.host_height;
    e.tiles = tiles;
    e.frame = frame_;
    owned_dirty_.push_back(e);
  };
  // (Depth and stencil apart: a later clear may write one of them whole and not the other.)
  add(p.color_slot, VK_IMAGE_ASPECT_COLOR_BIT);
  if (p.depth) {
    if ((dropped && p.clears_depth) || p.draws_depth) add(p.depth_slot, VK_IMAGE_ASPECT_DEPTH_BIT);
    // (Session 92, part D2, TRUE NATIVE: a pass that keeps stencil always leaves it dirty - the truth path may use it (ClaimTruth lets a stencil draw through over it).)
    if (p.stencil && (OwnedChains() || (dropped && p.clears_stencil) || p.draws_stencil)) add(p.depth_slot, VK_IMAGE_ASPECT_STENCIL_BIT);
  }
  if (REXCVAR_GET(gpu_native_own_steps_debug_materialize)) {
    for (OwnedDirty& e : owned_dirty_) {
      if (e.image == p.image) OwnedMaterialize(e, e.tiles);
    }
    OwnedPruneDirty();
  }
}

bool NativeRenderer::OwnedTruthHolds() const { return OwnedTruthHolds(owned_pending_); }

void NativeRenderer::OwnedDropParked(const char* why) {
  OwnedDropParkedIf(why, [](const OwnedPass&) { return true; });
}

void NativeRenderer::OwnedDropParkedIf(const char* why, const std::function<bool(const OwnedPass&)>& pred) {
  std::vector<OwnedPass> to_replay;  // (chains, review F2: defined starts parked without their copy that are dropped for a reason that reads them)
  const std::string_view reason(why);
  // (follow-up (b) 1: only the reasons that READ the pass; a submission or frame end reads nothing - the pass was drawn, its copy served or dead.)
  // (A parked defined start that is still without its copy is replayed when it is dropped for a reason that reads it - and at a submission's / a
  // frame's end, where its draws would otherwise be lost. Run 023's loss of the damage / sweat maps was NOT this replay: it was the owned-image budget
  // latch (598c566, 4e36709); the replay only changed the order the images were made in.)
  // (Session 90, P: not at a submission's or a frame's end any more - those readers do not exist: a defined start parked at a submission's end is
  // carried without its draws (OwnedOnSubmissionEnd), at a frame's end it ends as a committed pass does; its pixels stay in our image.)
  const bool needs = OwnedChains() && (reason == "a truth draw over its tiles" || reason == "a write-out" || reason == "a copy it cannot serve" ||
                                       reason == "a loaded start of the same surface");
  const size_t why_idx = reason == "the submission ends" ? 0 : reason == "the frame ends" ? 1 : reason == "a truth draw over its tiles" ? 2 :
                         reason == "a write-out" ? 3 : reason == "a copy it cannot serve" ? 4 : reason == "a loaded start of the same surface" ? 5 : 6;
  for (size_t i = owned_parked_.size(); i-- > 0;) {
    if (!pred(owned_parked_[i])) continue;
    if (needs && owned_parked_[i].chain_parked && owned_parked_[i].edram_free && !owned_parked_[i].draws.empty()) {
      to_replay.push_back(owned_parked_[i]);
      ++owned_frame_stats_.ch_replay_by[why_idx];
    } else if (OwnedChains() && owned_parked_[i].chain_parked && owned_parked_[i].edram_free && !owned_parked_[i].draws.empty() && why_idx <= 1) {
      ++owned_frame_stats_.ch_frame_end_unreplayed;  // (a defined start that would have been replayed here before Session 90)
    }
    if (owned_parked_[i].carried_defined) {
      ++owned_frame_stats_.ch_carried_dropped[why_idx];
      if (needs) {  // (Session 90: a reader of tiles that never got its pixels - no draws to replay: the key is paused)
        ++owned_frame_stats_.ch_carried_paused_read[why_idx];
        OwnedPause(owned_parked_[i].key, owned_parked_[i].producers, "a carried defined start read by the truth path");
      }
      if (why_idx == 1 && owned_frame_end_carried_.size() < 32) {  // (item 5: remembered to detect a copy of it next frame)
        owned_frame_end_carried_.push_back({owned_parked_[i].base, uint32_t(owned_parked_[i].pitch_tiles) << (owned_parked_[i].is_64bpp ? 1 : 0), owned_parked_[i].kind});
      }
    }
    OWNED_SCENE_LOG(owned_parked_[i], "[native][owned][scene] frame {}: parked pass base {} pitch {} DROPPED: {}", frame_, owned_parked_[i].base, owned_parked_[i].pitch_tiles, why);
    // (Session 87, gpu_native_owned_frame_restore_draw: a pass drawn into since its copies is kept too - the restore that follows is recorded into it, not left out.)
    if (REXCVAR_GET(gpu_native_owned_frame_restore_skip) && owned_parked_[i].committed &&
        ((owned_parked_[i].draws_since_color == 0 && owned_parked_[i].draws_since_depth == 0) || REXCVAR_GET(gpu_native_owned_frame_restore_draw)) &&
        (std::string_view(why) == "a truth draw over its tiles" || (OwnedChains() && std::string_view(why) == "the same surface starts again"))) {
      owned_restorable_.push_back(owned_parked_[i]);
      if (!owned_parked_[i].edram_free) owned_restorable_.back().draws.clear();  // (review: a defined start keeps its draws - a later replay needs them all)
      if (owned_restorable_.size() > 2) owned_restorable_.erase(owned_restorable_.begin());
    }
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} the parked pass (base {} pitch {}) is dropped: {}", frame_,
                  owned_parked_[i].base, owned_parked_[i].pitch_tiles, why);
    }
    // (Session 92, M2a: a parked pass that ends with its submission or its frame is its surface's newest record; the other reasons - a clear of the
    // surface, a new pass of it, its pixels taken into a loaded start - supersede it.)
    if (OwnedOnly()) {
      // (Failure read fix 1: EVERY dropped parked pass is noted - a pass holds two surfaces. When the reason replaces its colour (a clear or a new pass of the
      // surface, its pixels taken into a loaded start) it stays only as its depth slot's record, until a pass holding that slot supersedes it - the scene pass
      // parked after its resolves, dropped by the combine's covered start, was the E1 depth source.)
      const bool ends = reason == "the submission ends" || reason == "the frame ends" || reason == "a write-out";
      if (reason == "a held clear over its tiles or surface") {  // (review D2: the clear replaces that surface's content - older records lose it too)
        OwnedSupersedeRecords(owned_parked_[i].base, owned_parked_[i].pitch_tiles, owned_parked_[i].kind, -1);
      }
      OwnedPass dropped = owned_parked_[i];
      if (!ends) dropped.rec_colour_gone = true;
      OwnedNoteNewest(dropped);
    }
    owned_parked_.erase(owned_parked_.begin() + i);
  }
  // (Oldest first: they were found newest first.)
  for (size_t k = to_replay.size(); k-- > 0;) OwnedReplayParkedPass(std::move(to_replay[k]), why);
}

void NativeRenderer::OwnedParkOrReplay(const char* why) {
  OwnedPass& p = owned_pending_;
  if (!p.valid) return;
  OWNED_SCENE_LOG(p, "[native][owned][scene] frame {}: park or replay base {} pitch {}: committed {} same submission {} ({})", frame_, p.base, p.pitch_tiles, p.committed, p.submission == command_processor_.GetCurrentSubmission(), why);
  const bool chains = OwnedChains() && owned_frame_on_;
  if (chains && !p.committed && (p.edram_free || p.start_kind == kOwnedStartCovered || OwnedTruthHolds() || OwnedOnly())) {  // (review F2: a defined start keeps its draws, and is replayed if it is dropped before its copy; Session 92, M2a: every pass is parked, none replayed)
    if (p.edram_free) p.chain_parked = true;
    // (Session 87, design B: an uncommitted pass is parked too. Its pixels are in our image and its tiles become dirty - as at a submission's end,
    // where a pass the truth images can be given later is committed without its copy - so nothing needs replaying; its copy comes later.)
    OwnedCommit();
    ++owned_frame_stats_.ch_parked;
  }
  if (owned_frame_on_ && p.committed &&
      (p.submission == command_processor_.GetCurrentSubmission() || (REXCVAR_GET(gpu_native_owned_frame_carry_passes) && !p.edram_free))) {
    OwnedEndPass();  // (its render pass ends; a later draw resumes it)
    OWNED_SCENE_LOG(p, "[native][owned][scene] frame {}: pass base {} pitch {} PARKED ({}), {} parked", frame_, p.base, p.pitch_tiles, why, owned_parked_.size() + 1);
    owned_parked_.push_back(p);
    if (!p.edram_free) owned_parked_.back().draws.clear();  // (review: a defined start keeps its draws: a replay after the pass is taken up again needs all of them)
    owned_parked_.back().open = false;
    if (owned_parked_.size() > (chains ? 16u : 4u)) {
      OwnedPass evicted = std::move(owned_parked_.front());
      if (evicted.carried_defined) ++owned_frame_stats_.ch_carried_dropped[6];
      owned_parked_.erase(owned_parked_.begin());
      if (OwnedOnly()) OwnedNoteNewest(evicted);  // (Session 92, M2a: the oldest parked pass becomes its surface's newest record, not a loss)
      if (chains && evicted.chain_parked && evicted.edram_free && !evicted.draws.empty()) {
        ++owned_frame_stats_.ch_replay_by[6];
        OwnedReplayParkedPass(std::move(evicted), "evicted from the parked passes");
      }
    }
    p.valid = false;
    p.committed = false;
    p.draws.clear();
    return;
  }
  // (Session 82, P2, G4: a defined start owes nothing to the EDRAM - its box guess being too small is no reason to pause the
  // key for 600 frames.)
  OwnedReplayPending(why, !p.edram_free);
}

bool NativeRenderer::OwnedTruthHolds(const OwnedPass& p) const {
  // Whether the truth images get the pass's pixels as dirty tiles (OwnedMarkDirty): a pass from its clears, or from a
  // covering draw when its colour image owns every tile of the box. Else only our image holds them (the ring
  // downscale: its tiles are the shadow map's) - a copy of it is exact only served from our image.
  if (OwnedOnly()) return false;  // (Session 92, M2a: the truth images hold nothing of any pass - only our image does; every "only our image holds it" path is the right one)
  if (p.cleared) return true;
  if (p.color_slot < 0 || size_t(p.color_slot) >= truth_images_.size()) return false;
  const TruthImage& t = truth_images_[p.color_slot];
  return (EdramTiles(t.base, t.pitch_tiles, t.is_64bpp(), 0, 0, p.width, p.height) & ~t.owned).none();
}

void NativeRenderer::OwnedTouch(int16_t image) {
  if (image >= 0 && size_t(image) < owned_images_.size()) owned_images_[image].last_use = command_processor_.GetCurrentSubmission();
}

bool NativeRenderer::OwnedCommit() {
  // A copy of the pass is taken: its render pass ends; the first time, the clears it started from are turned off
  // (gpu_native_own_steps_drop_clears: our image holds what they wrote - their draws only when their pass loaded tiles
  // at its start: those loads wrote pixels outside the box; else the whole pass; only in the submission it was
  // recorded in: the offsets are that stream's) and its tiles become dirty. Returns whether the clears were dropped.
  OwnedPass& p = owned_pending_;
  OwnedEndPass();
  OwnedTouch(p.image);
  if (p.committed) return false;
  p.committed = true;
  p.copy_count = 0;
  ++truth_counts_.owned_copies;
  bool dropped = false;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  if (p.cleared && REXCVAR_GET(gpu_native_own_steps_drop_clears)) {
    if (p.clears_held) {
      dropped = p.clears_ours;  // (nothing of them was recorded: nothing to turn off)
    } else if (p.clears_ours && p.clears_begin != SIZE_MAX && p.clears_submission == command_processor_.GetCurrentSubmission() &&
               truth_pass_map_submission_ == p.clears_submission && truth_pass_by_offset_.count(p.clears_begin) &&
               truth_pass_by_offset_[p.clears_begin] == p.clears_serial) {  // (review F9: nothing else began a pass at these offsets)
      dropped = p.clears_loaded ? cb.NopDraws(p.clears_draws, p.clears_end_cmd)
                                : cb.NopRenderPass(p.clears_begin, p.clears_end, true);
    }
    ++(dropped ? truth_counts_.owned_clears_dropped : truth_counts_.owned_clears_kept);
    // (gpu_native_truth_drop_dead_passes: that pass is off already, if it is the candidate.)
    if (dropped && dead_pass_.valid && dead_pass_.begin == p.clears_begin) dead_pass_.valid = false;
  }
  p.clears_begin = SIZE_MAX;
  if (!p.edram_free) p.held_clears.clear();  // (a committed pass is never replayed - a defined start is: see keeps_draws)
  if (p.census >= 0 && size_t(p.census) < owned_frame_passes_.size()) {
    OwnedFramePass& r = owned_frame_passes_[p.census];
    r.owned = true;
    r.lost = false;
    r.reason = kOwnedReasonNone;
    r.clears_kept = p.cleared && !dropped;
  }
  if (OwnedChains() && owned_frame_on_ && p.clears_held && !p.edram_free) OwnedClaimClearTiles(p);  // (review F3)
  OwnedMarkDirty(dropped);
  if (!p.edram_free) {
    p.draws.clear();  // (never replayed from here on)
    p.crowd_calls.clear();
    p.serves.clear();
  }
  return dropped;
}

int16_t NativeRenderer::OwnedSnapshot(uint32_t gap) {
  // Our image (an attachment, its pass ended) copied whole into a free image of its size: the page a later draw of
  // the same pass may sample. With `gap` (Session 80: at a resolution scale with the half-pixel offset the game's copy
  // fills the first host columns / rows - never drawn - from the first covered one, the truth fill's max(p, gap)):
  // texel (x, y) from (max(x, gx), max(y, gy)) - our image keeps its own pixels (the source of its dirty tiles).
  OwnedGraphTripwire(owned_pending_.image);  // (Session 95)
  OwnedGraphNoteRead(owned_pending_.image);  // (Session 95, round 7)
  OwnedPass& p = owned_pending_;
  OwnedTouch(p.image);
  const OwnedImage& o = owned_images_[p.image];
  const uint32_t gx = std::min(gap & 0xFFu, o.width - 1), gy = std::min(gap >> 8, o.height - 1);
  if (gx > 8 || gy > 8) return -1;  // (scales above 16: not ours)
  const int16_t index = OwnedImageFor(o.width, o.height, o.kind);
  if (index < 0) return -1;
  OwnedTouch(index);  // (the snapshot is written in this submission)
  const OwnedImage& s = owned_images_[index];
  const OwnedImage& m = owned_images_[p.image];  // (again: OwnedImageFor may have grown the vector)
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t timer = OwnedTimerBegin("snapshot", m.width, m.height);  // (Session 82, P1)
  VkImageMemoryBarrier barriers[2] = {
      OwnedBarrier(m.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   owned_images_[p.image].readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                    : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
      OwnedBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, kOwnedAnyWrite, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)};
  cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
  // (The covered part as it is; each gap column / row from the first covered one, the corner from its first covered
  // texel - disjoint destinations, one copy.)
  VkImageCopy regions[1 + 8 + 8 + 64] = {};
  uint32_t count = 0;
  const auto add = [&](uint32_t sx, uint32_t sy, uint32_t dx, uint32_t dy, uint32_t w, uint32_t h) {
    VkImageCopy& r = regions[count++];
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.srcOffset = {int32_t(sx), int32_t(sy), 0};
    r.dstOffset = {int32_t(dx), int32_t(dy), 0};
    r.extent = {w, h, 1};
  };
  add(gx, gy, gx, gy, m.width - gx, m.height - gy);
  for (uint32_t i = 0; i < gx; ++i) add(gx, gy, i, gy, 1, m.height - gy);
  for (uint32_t j = 0; j < gy; ++j) add(gx, gy, gx, j, m.width - gx, 1);
  for (uint32_t i = 0; i < gx; ++i) {
    for (uint32_t j = 0; j < gy; ++j) add(gx, gy, i, j, 1, 1);
  }
  OwnedCopyImage(cb, m.image, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.image, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, count,
                 regions);  // (Session 94: per layer)
  // (Review: our image back in the layout it had - readable when a page of ours still samples it, else an attachment.)
  const bool was_readable = owned_images_[p.image].readable;
  barriers[0] = OwnedBarrier(m.image, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                             was_readable ? VK_ACCESS_SHADER_READ_BIT
                                          : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             was_readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                          : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  barriers[1] = OwnedBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, barriers);
  OwnedTimerEnd(timer);
  OwnedGraphCount(kGraphCountSnapshot);  // (Session 95)
  owned_images_[index].readable = true;
  ++truth_counts_.owned_snapshots;
  ++owned_frame_stats_.snapshots;
  return index;
}

bool NativeRenderer::OwnedContinueCopy(OwnedPass& p) {
  // Session 87 (gpu_native_owned_frame_continue_copy). The pass is committed and its next draw would change an image that pages of ours
  // are: the pass goes on in a fresh image of ours that starts as a copy of this one (colour, and depth + stencil where the pass keeps
  // them). The old image is left exactly as it is for its pages - in the layouts it has - so no view handed out for it changes meaning.
  // Hard rule 6: our own images only; no EDRAM, no truth pass.
  const auto no = [&](const char* why) {
    static uint32_t logged = 0;
    if (owned_frame_logging_ || logged < 8) {
      ++logged;
      REXGPU_INFO("[native][owned] frame {}: pass base {} pitch {} cannot go on in a copy of its image: {}", frame_, p.base, p.pitch_tiles, why);
    }
    return false;
  };
  if (p.half) return no("a half-size pass");
  if (p.image < 0 || size_t(p.image) >= owned_images_.size()) return no("no image");
  const int16_t old_index = p.image;
  const uint32_t w = owned_images_[old_index].width, h = owned_images_[old_index].height, kind = owned_images_[old_index].kind;
  const bool with_depth = p.depth;
  if (with_depth && owned_images_[old_index].depth == VK_NULL_HANDLE) return no("the pass keeps depth but its image has none");
  if (!OwnedImageAvailable(w, h, kind, with_depth)) return no("no image within the budget");
  // (Copy commands: no render pass of the truth path or the command processor may be open - as at a resume.)
  OwnedEndPass();
  EndTruthPass();
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  const int16_t new_index = OwnedImageFor(w, h, kind);
  if (new_index < 0 || new_index == old_index) return no("no free image");
  if (with_depth && !OwnedEnsureDepth(new_index)) return no("no depth image");
  if (with_depth) {  // (Session 93: the old image's depth is copied into the new one; the new image's old pass is over)
    OwnedNoteDepthRead(old_index, "continue_copy");
    OwnedDepthAccount(new_index);
  }
  const OwnedImage& s = owned_images_[old_index];  // (after every call that may grow the vector)
  OwnedImage& d = owned_images_[new_index];
  const bool src_readable = s.readable, src_depth_read = s.depth_read;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  const uint32_t timer = OwnedTimerBegin("continue_copy", w, h);
  VkImageMemoryBarrier barriers[4];
  uint32_t count = 0;
  barriers[count++] = OwnedBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                   src_readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  barriers[count++] = OwnedBarrier(d.image, VK_IMAGE_ASPECT_COLOR_BIT, kOwnedAnyWrite, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  if (with_depth) {
    barriers[count++] = OwnedBarrier(s.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                                     src_depth_read ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     VK_ACCESS_TRANSFER_READ_BIT,
                                     src_depth_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    barriers[count++] = OwnedBarrier(d.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, kOwnedAnyWrite,
                                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }
  cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                          nullptr, 0, nullptr, count, barriers);
  VkImageCopy region = {};
  region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.extent = {w, h, 1};
  OwnedCopyImage(cb, s.image, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.image, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);  // (Session 94: per layer)
  if (with_depth) {
    region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1};
    OwnedCopyImage(cb, s.depth, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.depth, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);  // (Session 94: per layer)
  }
  // (The old image back in the layouts it had; the new one ready to be resumed: attachment layouts.)
  count = 0;
  barriers[count++] = OwnedBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                   src_readable ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   src_readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  barriers[count++] = OwnedBarrier(d.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  if (with_depth) {
    barriers[count++] = OwnedBarrier(s.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0,
                                     src_depth_read ? VK_ACCESS_SHADER_READ_BIT
                                                    : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                     src_depth_read ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    barriers[count++] = OwnedBarrier(d.depth, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                          nullptr, 0, nullptr, count, barriers);
  OwnedTimerEnd(timer);
  d.readable = false;
  OwnedGraphCount(kGraphCountContinueCopy);  // (Session 95)
  d.depth_read = false;
  p.image = new_index;
  OwnedTouch(old_index);
  OwnedTouch(new_index);
  ++owned_frame_stats_.detached;
  command_processor_.NativeOwnedPagesChanged();
  static uint32_t done_logged = 0;
  if (owned_frame_logging_ || done_logged < 8) {
    ++done_logged;
    REXGPU_INFO("[native][owned] frame {}: pass base {} pitch {} goes on after its copies in a copy of its image ({} -> {}, {}x{}{})", frame_,
                p.base, p.pitch_tiles, old_index, new_index, w, h, with_depth ? ", with depth" : "");
  }
  return true;
}

bool NativeRenderer::OwnedDetachPages(OwnedPass& p) {
  // Session 87 (gpu_native_owned_frame_detach_pages). The pass is committed and goes on drawing; a page that IS its image would
  // change under its readers. What a page holds is what the image held at the game's copy, and nothing was drawn into the pass
  // since (the guard runs at the first draw after a copy; the counters prove it), so a copy of the image now is that content.
  // Hard rule 6: this is our own pass keeping the pixels its readers were promised - not an EDRAM reload and no truth pass.
  std::vector<size_t> colour, depth;
  for (size_t i = 0; i < owned_pages_.size(); ++i) {
    if (owned_pages_[i].image != p.image) continue;
    (owned_pages_[i].depth_direct ? depth : colour).push_back(i);
  }
  const auto log_no = [&](const char* why) {
    static uint32_t logged = 0;
    if (owned_frame_logging_ || logged < 8) {
      ++logged;
      REXGPU_INFO("[native][owned] frame {}: pass base {} pitch {} cannot detach its pages ({} colour, {} depth): {}", frame_, p.base,
                  p.pitch_tiles, colour.size(), depth.size(), why);
    }
    return false;
  };
  if (colour.empty() && depth.empty()) return true;
  if (p.half) return log_no("a half-size pass");
  // (The draw being recorded was given these pages' views by its texture request, before this call: a draw that samples a page of its
  // own image - map B's blurs read map B - cannot go on in that image; the pass ends as it always did.)
  if (std::find(owned_sampled_images_.begin(), owned_sampled_images_.end(), p.image) != owned_sampled_images_.end()) {
    return log_no("this draw samples a page of its own image");
  }
  if (!colour.empty() && p.draws_since_color != 0) return log_no("drawn since its whole colour copy");
  if (!depth.empty() && p.draws_since_depth != 0) return log_no("drawn since its depth copy");
  const OwnedImage& image = owned_images_[p.image];
  if (!depth.empty() && (image.depth == VK_NULL_HANDLE || !image.depth_read)) return log_no("the depth is not in the read layout");
  // (Session 87, review A2: a defined-start pass keeps its draws, with their descriptor sets, past its commit - parked, restorable or this
  // pass itself - for a replay on a later loss. A kept draw may have sampled a direct page of this image; detached, the replay would sample
  // the image in its attachment layout. Run 013: refusing whenever any pass kept draws refused the entrance scene pass's detach
  // every time (its leftovers went back to the truth path): refused only when a kept draw sampled this image.)
  const auto keeps = [&p](const OwnedPass& q) {
    return !q.draws.empty() && std::find(q.kept_sampled.begin(), q.kept_sampled.end(), p.image) != q.kept_sampled.end();
  };
  if (keeps(p) || std::any_of(owned_parked_.begin(), owned_parked_.end(), keeps) ||
      std::any_of(owned_restorable_.begin(), owned_restorable_.end(), keeps)) {
    return log_no("a pass holds kept draws a replay could sample its pages with");
  }
  // (Copy and compute commands: no render pass of the truth path or the command processor may be open - as at a resume.)
  OwnedEndPass();  // (p.open is false by invariant; the review asked for it)
  EndTruthPass();
  command_processor_.EndRenderPass();
  command_processor_.SubmitBarriers(true);
  if (!colour.empty()) {
    const int16_t snapshot = OwnedSnapshot(0);
    if (snapshot < 0) return log_no("no image for the colour snapshot");
    for (const size_t i : colour) owned_pages_[i].image = snapshot;
    ++owned_frame_stats_.detached;
  }
  if (!depth.empty()) {
    // (One converted page image for every depth page of the image: they hold the same depth - no draw between the copies.)
    uint32_t scale_x = 1, scale_y = 1;
    command_processor_.NativeResolutionScale(scale_x, scale_y);
    const OwnedPage& first = owned_pages_[depth.front()];
    const uint32_t w = first.width * scale_x, h = first.height * scale_y;
    const int16_t page_image = OwnedPageImageFor(w, h, uint32_t(xenos::TextureFormat::k_24_8_FLOAT));
    if (page_image < 0 || !OwnedFillDepth(page_image, p.image, w, h, 0, true)) {
      if (!colour.empty()) command_processor_.NativeOwnedPagesChanged();
      return log_no("the depth conversion could not be recorded");
    }
    owned_images_[p.image].depth_read = false;  // (back in its attachment layout: the resume does not expect the read layout)
    for (const size_t i : depth) {
      owned_pages_[i].depth_direct = false;
      owned_pages_[i].image = -1;
      owned_pages_[i].page_image = page_image;
    }
    ++owned_frame_stats_.detached;
  }
  command_processor_.NativeOwnedPagesChanged();
  OwnedWatchSync();
  static uint32_t done_logged = 0;
  if (owned_frame_logging_ || done_logged < 8) {
    ++done_logged;
    REXGPU_INFO("[native][owned] frame {}: pass base {} pitch {} goes on after its copies: {} colour page(s) given a snapshot, {} depth "
                "page(s) a converted page image", frame_, p.base, p.pitch_tiles, colour.size(), depth.size());
  }
  return true;
}

int16_t NativeRenderer::OwnedPageImageFor(uint32_t width, uint32_t height, uint32_t format) {
  // A free one of this size and format (no page holds it), or a new one.
  for (size_t i = 0; i < owned_page_images_.size(); ++i) {
    const OwnedPageImage& q = owned_page_images_[i];
    if (q.width != width || q.height != height || q.format != format) continue;
    if (std::none_of(owned_pages_.begin(), owned_pages_.end(),
                     [&](const OwnedPage& p) { return p.page_image == int16_t(i); })) {
      return int16_t(i);
    }
  }
  const bool depth32 = format == uint32_t(xenos::TextureFormat::k_24_8_FLOAT);  // (Session 82, P1: a depth copy's page: R32_SFLOAT)
  const uint64_t page_bytes = uint64_t(width) * height * (depth32 || format == uint32_t(xenos::ColorFormat::k_16_16) ? 4u : 8u);
  if (owned_frame_on_ ? OwnedOverBudget(page_bytes) : owned_page_images_.size() >= 16) {
    // (Session 87, chains: unused owned images are evicted first, as for an image.)
    if (owned_frame_on_ && OwnedChains()) OwnedEvictFor(page_bytes);
    if (owned_frame_on_ ? OwnedOverBudget(page_bytes) : owned_page_images_.size() >= 16) {
      static uint32_t budget_logged = 0;
      if (OwnedChains() && owned_frame_on_ && budget_logged < 8) {
        ++budget_logged;
        REXGPU_INFO("[native][owned] frame {}: no page image {}x{} (format {}): {} MB in use + {} MB over the owned-frame budget", frame_, width, height, format,
                    owned_bytes_ >> 20, page_bytes >> 20);
      }
      return -1;
    }
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const VkDescriptorSetLayout storage_layout = command_processor_.NativeTextureSetLayout();
  if (storage_layout == VK_NULL_HANDLE) return -1;
  const bool pair = format == uint32_t(xenos::ColorFormat::k_16_16) || depth32;  // (a 32-bit texel: the R32_UINT storage view)
  const VkFormat image_format = depth32 ? VK_FORMAT_R32_SFLOAT : pair ? VK_FORMAT_R16G16_UNORM : VK_FORMAT_R16G16B16A16_UNORM;
  // (Review: the UNORM format sampled - as the texture cache's own; stored through its UINT view - R32_UINT /
  // R16G16B16A16_UINT have storage everywhere; where the UNORM format has not, the image takes EXTENDED_USAGE.)
  VkImageCreateFlags flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;  // (UNORM / SNORM sampled, UINT storage views)
  {
    const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
    VkFormatProperties properties = {}, storage = {};
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(), image_format, &properties);
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(),
                                            pair ? VK_FORMAT_R32_UINT : VK_FORMAT_R16G16B16A16_UINT, &storage);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ||
        !(storage.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
      return -1;
    }
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
      flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    }
  }
  OwnedPageImage q;
  q.width = width;
  q.height = height;
  q.format = format;
  q.bytes = page_bytes;
  q.kind = depth32 ? 1u : 0u;
  VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.flags = flags;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = image_format;
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  // (Session 93, gpu_owned_tiler_passes: every format a view of a page image takes - the image's own, the storage view's UINT (fills write through it:
  // the page images keep STORAGE usage, they are written by compute), and the signed twin OwnedServeView makes for a 16-bit sampler.)
  VkFormat page_view_formats[3];
  uint32_t page_view_format_count = 0;
  VkImageFormatListCreateInfo page_format_list = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
  if (OwnedFormatListOn() && vulkan_device->extensions().ext_1_2_KHR_image_format_list) {
    page_view_formats[page_view_format_count++] = image_format;
    page_view_formats[page_view_format_count++] = pair ? VK_FORMAT_R32_UINT : VK_FORMAT_R16G16B16A16_UINT;
    if (!depth32) page_view_formats[page_view_format_count++] = pair ? VK_FORMAT_R16G16_SNORM : VK_FORMAT_R16G16B16A16_SNORM;
    page_format_list.viewFormatCount = page_view_format_count;
    page_format_list.pViewFormats = page_view_formats;
    image_info.pNext = &page_format_list;
  }
  bool ok = ui::vulkan::util::CreateDedicatedAllocationImage(vulkan_device, image_info,
                                                             ui::vulkan::util::MemoryPurpose::kDeviceLocal, q.image,
                                                             q.memory);
  if (ok && image_info.pNext) ++g_tiler_format_lists;
  if (ok) {
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = q.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = pair ? VK_FORMAT_R32_UINT : VK_FORMAT_R16G16B16A16_UINT;  // (the fill's r32ui / rgba16ui)
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
    ok = dfn.vkCreateImageView(device, &view_info, nullptr, &q.storage_view) == VK_SUCCESS;
  }
  if (ok) ok = OwnedAllocateSet(storage_layout, q.storage_set) == VK_SUCCESS;
  if (!ok) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, q.storage_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, q.image);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, q.memory);
    REXGPU_ERROR("[native][owned] could not make a {}x{} 16-bit page image", width, height);
    return -1;
  }
  VkDescriptorImageInfo storage_info = {VK_NULL_HANDLE, q.storage_view, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = q.storage_set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  write.pImageInfo = &storage_info;
  dfn.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  REXGPU_INFO("[native][owned] 16-bit page image {}x{} (format {}) made", width, height, format);
  owned_bytes_ += q.bytes;
  owned_page_images_.push_back(q);
  return int16_t(owned_page_images_.size() - 1);
}

bool NativeRenderer::OwnedFill(int16_t page_image, bool fresh, uint32_t src_x, uint32_t src_y, uint32_t dst_x,
                               uint32_t dst_y, uint32_t width, uint32_t height, uint32_t gap) {
  // A rectangle of the pass's 16-bit image into a page image as the game's copy would write it: the truth path's
  // texture fill (its pipelines, kTruthFillComputeShader mode 1 / 3) with our image as the source. Host texels; `gap`
  // the copy's half-pixel offset gap as TruthFillTexture passes it (review: at a resolution scale the first host
  // columns / rows come from the first covered one).
  OwnedPass& p = owned_pending_;
  if (page_image < 0 || size_t(page_image) >= owned_page_images_.size() || !EnsureTruthStaging() ||
      !EnsureTruthFillTexPipelines()) {
    return false;
  }
  OwnedGraphTripwire(p.image);  // (Session 95)
  OwnedGraphNoteRead(p.image);  // (Session 95, round 7: the fill's dispatch)
  OwnedTouch(p.image);
  OwnedImage& o = owned_images_[p.image];
  const OwnedPageImage& q = owned_page_images_[page_image];
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  // (Review: its binding 0 names the staging buffer - statically used by the shader; a new staging buffer, a new set.
  // The old one stays allocated in our pool: the buffer changes only with the EDRAM's size.)
  if (o.fill_set != VK_NULL_HANDLE && o.fill_set_buffer != truth_staging_) {
    OwnedSpareFillSet(o.fill_set);
    o.fill_set = VK_NULL_HANDLE;
  }
  if (o.fill_set == VK_NULL_HANDLE) {
    o.fill_set_buffer = truth_staging_;
    if (OwnedAcquireFillSet(o.fill_set) != VK_SUCCESS) {
      o.fill_set = VK_NULL_HANDLE;
      return false;
    }
    // (Binding 0, the store's buffer, is not written by the fill's texture output: the staging buffer stands in.)
    VkDescriptorBufferInfo buffer_info = {truth_staging_, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo image_info = {sampler_nearest(), o.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = o.fill_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &buffer_info;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &image_info;
    dfn.vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  }
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  command_processor_.SubmitBarriers(true);  // (as TruthFillTexture: the command processor's own first)
  const uint32_t timer = OwnedTimerBegin("page_fill", width, height);  // (Session 82, P1)
  // Our image readable by the compute shader; the page image writable (whole: its old texels not kept).
  VkImageMemoryBarrier barriers[2];
  uint32_t count = 0;
  if (!owned_images_[p.image].readable) {
    barriers[count++] =
        OwnedBarrier(o.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    owned_images_[p.image].readable = true;
  }
  barriers[count++] = OwnedBarrier(q.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                   VK_ACCESS_SHADER_WRITE_BIT,
                                   fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_LAYOUT_GENERAL);
  cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                          nullptr, 0, nullptr, count, barriers);
  const bool pair = q.format == uint32_t(xenos::ColorFormat::k_16_16);
  command_processor_.BindExternalComputePipeline(truth_fill_tex_pipelines_[pair ? 0 : 1]);
  const VkDescriptorSet sets[2] = {o.fill_set, q.storage_set};
  cb.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, truth_fill_tex_layout_, 0, 2, sets, 0, nullptr);
  // (origin, size, mode - 1 k_16_16 / 3 k_16_16_16_16 -, no gap, the destination's first texel.)
  const uint32_t constants[10] = {src_x, src_y, width, height, pair ? 1u : 3u, gap, dst_x, dst_y, 0u, 0u};
  cb.CmdVkPushConstants(truth_fill_tex_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);
  cb.CmdVkDispatch((width + 7) / 8, (height + 7) / 8, 1);
  barriers[0] = OwnedBarrier(q.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                             VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, kOwnedShaderStages, 0, 0, nullptr, 0, nullptr, 1,
                          barriers);
  OwnedTimerEnd(timer);
  truth_foreign_bind_ = true;  // (a compute pipeline and sets of ours bound)
  return true;
}

// Session 82 (P2, diagnostic): which return of OwnedTakeResolve lost the pass (the line), for the 9C85DA5F log.
static int g_owned_lose_line = 0;
#define OWNED_LOSE(...) (g_owned_lose_line = __LINE__, lose(__VA_ARGS__))

bool NativeRenderer::OwnedTakeResolve(const draw_util::ResolveInfo& info, bool copy_known) {
  // Session 90 (Q): a copy served while the CPU is to read its bytes (a read-back window) is written into guest memory from our image - the
  // inner function plans it (owned_wb_planned_) once the copy's shape is known, and this arms it when the copy was in fact served: the
  // render target cache then runs OwnedWriteBackRun in its destination block.
  if (owned_wb_.armed) OwnedWbRefused("armed, but the cache never ran it (no destination block)");  // (loud, never silent)
  owned_wb_.armed = false;
  g_drop_note.clear();  // (Session 92, M2c)
  owned_wb_planned_ = false;
  owned_replay_reason_ = 0;
  const bool taken = OwnedTakeResolveInner(info, copy_known);
  owned_replay_reason_ = 0;
  if (taken && owned_frame_on_) {
    if (owned_wb_planned_) {
      owned_wb_.armed = true;
      // Session 90 (Q fix): a page the CPU reads back is not left ours: its memory is exact (written from our image), so the game's own samplers of it
      // load the texture from there. Diagnosis (Heenan's attire, dumps of the old replay vs the write-back): the first two layer copies were byte-identical,
      // the later ones - whose draws sample the previous layer's page - came out with red only: sampling our page (a snapshot of the pass) differs from the
      // texture the game's copy makes. Feedback pages (damage maps, cubes) stay ours.
      if (REXCVAR_GET(gpu_native_owned_wb_drop_pages) && owned_wb_.path != 4 && !owned_wb_.front) {
        const uint64_t start = info.copy_dest_extent_start, end = start + info.copy_dest_extent_length;
        const size_t before = owned_pages_.size();
        owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                          [&](const OwnedPage& q) {
                                            return q.feedback < 0 && start < uint64_t(q.extent_start) + q.extent_length && uint64_t(q.extent_start) < end;
                                          }),
                           owned_pages_.end());
        if (owned_pages_.size() != before) {
          command_processor_.NativeOwnedPagesChanged();
          OwnedWatchSync();
        }
      }
    } else if (command_processor_.NativeCopyReadsBack() &&
               int32_t(info.coordinate_info.width_div_8) * 8 >= REXCVAR_GET(gpu_native_owned_frame_readback_min_w)) {
      ++owned_frame_stats_.wb_unplanned;  // (served in a window, no write-back planned: a dead copy, or one that never reached the plan)
    }
  }
  owned_wb_planned_ = false;
  return taken;
}

bool NativeRenderer::OwnedTakeResolveInner(const draw_util::ResolveInfo& info, bool copy_known) {
  // (A copy may read what the clears held would have cleared: the truth path runs them first - those of the surface it reads;
  // the clears of another surface stay held.)
  if (!owned_held_.draws.empty()) {
    const OwnedSurface hc = OwnedSlotKey(owned_held_.color);
    const bool reads_color = !info.IsCopyingDepth() && uint32_t(info.color_original_base) == hc.base &&
                             (!OwnedChains() ||
                              (info.color_edram_info.pitch_tiles == (hc.pitch_tiles << (hc.is_64bpp() ? 1 : 0)) &&
                               bool(info.color_edram_info.format_is_64bpp) == hc.is_64bpp() &&
                               (xenos::ColorRenderTargetFormat(uint32_t(info.color_edram_info.format)) == xenos::ColorRenderTargetFormat::k_8_8_8_8) == (owned_held_.kind == 1)));
    const bool reads_depth = info.IsCopyingDepth() && owned_held_.depth >= 0 &&
                             uint32_t(info.depth_original_base) == OwnedSlotKey(owned_held_.depth).base;
    if (reads_color || reads_depth) OwnedFlushHeld(true);
  }
  if (!owned_depth_clears_.empty() && info.IsCopyingDepth()) OwnedDepthClearsFlush("a copy reads that depth", -1, int64_t(info.depth_original_base));  // (Session 87)
  if (OwnedChains() && owned_frame_on_ && (!truth_ || !replacing_) && info.copy_dest_extent_length && !owned_pages_.empty()) {
    // Session 92, part C (TRUE NATIVE only): a frame that fell back to the truth path still runs the game's own copy, which writes this memory -
    // every owned page it overlaps (feedback pages too) is the game's again, as unmap() below does for a served frame. Else the presenter keeps
    // showing the stale page of ours (the black ladder match).
    const uint64_t start = info.copy_dest_extent_start, end = start + info.copy_dest_extent_length;
    const size_t before = owned_pages_.size();
    owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                      [&](const OwnedPage& q) {
                                        return start < uint64_t(q.extent_start) + q.extent_length && uint64_t(q.extent_start) < end;
                                      }),
                       owned_pages_.end());
    if (owned_pages_.size() != before) {
      g_s92.fallen_front_drops += uint32_t(before - owned_pages_.size());
      command_processor_.NativeOwnedPagesChanged();
      OwnedWatchSync();
    }
  }
  if (!truth_ || !replacing_ || !info.copy_dest_extent_length) return false;
  if (owned_watch_pending_.load(std::memory_order_acquire)) OwnedDrainCpuWrites();
  const uint32_t page = (info.copy_dest_base >> 12) & 0x1FFFF;
  const uint32_t texture_page = (info.copy_dest_texture_base >> 12) & 0x1FFFF;
  // Pages whose memory a copy writes (colour or depth, any target, any copy shader) are the game's again (review) -
  // unless the copy goes into the page's image of ours (below).
  const auto unmap = [&]() {
    if (owned_pages_.empty()) return;
    const uint64_t start = info.copy_dest_extent_start, end = start + info.copy_dest_extent_length;
    const size_t before = owned_pages_.size();
    owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                      [&](const OwnedPage& p) {
                                        return start < uint64_t(p.extent_start) + p.extent_length &&
                                               uint64_t(p.extent_start) < end;
                                      }),
                       owned_pages_.end());
    if (owned_pages_.size() != before) {
      command_processor_.NativeOwnedPagesChanged();
      OwnedWatchSync();
      OwnedLeanSettle(false);  // (Session 95: a copy of ours may have lost its last page unread)
      OwnedRsPagesChanged();  // (Session 95)
    }
  };
  // The copy's source tiles: dirty ones get our pixels first when the copy runs from the truth images (it reads them);
  // none of the owned pass's - not its copy; some - its copy, if a plain one of a rectangle of its box.
  uint32_t span_base = 0, row_length = 0, rows = 0, span_pitch = 0;
  info.GetCopyEdramTileSpan(span_base, row_length, rows, span_pitch);
  std::bitset<2048> source;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < row_length; ++c) source.set((span_base + r * span_pitch + c) & 2047u);
  }
  OwnedPass& p = owned_pending_;
  // Session 82 (the census): the copy counted on the pass it belongs to - the pending one when it reads its tiles, else
  // the last pass recorded: 0 served view, 1 page image, 2 dead, 3 run by the game from our pixels, -1 the truth
  // path's.
  if (owned_clears_ended_ && (source & owned_clears_tiles_).any()) OwnedClearsTouched();
  if (OwnedChains() && owned_frame_on_ && !owned_parked_.empty() && !(p.valid && OwnedCopyOfPass(info, p))) {
    // (Session 87, design C: a copy of the surface of a parked pass takes that pass up - served from it - whatever pass is pending.)
    for (size_t pj = 0; pj < owned_parked_.size(); ++pj) {
      const size_t pi = owned_parked_.size() - 1 - pj;  // (review F7: the newest parked pass first)
      if (!OwnedCopyOfPass(info, owned_parked_[pi])) continue;
      OwnedPass taken_up = std::move(owned_parked_[pi]);
      owned_parked_.erase(owned_parked_.begin() + pi);
      if (p.valid) OwnedParkOrReplay("a copy of a parked surface");
      owned_pending_ = std::move(taken_up);
      ++owned_frame_stats_.ch_takenup;
      if (owned_pending_.carried_defined) ++owned_frame_stats_.ch_carried_taken_copy;
      break;
    }
  }
  if (OwnedOnly() && !owned_newest_.empty() && !(p.valid && OwnedCopyOfPass(info, p))) {
    // Session 92, M2a (design 4 item 6): a copy of a surface no pass of ours holds now is served from the surface's newest record - a pass that ended
    // (a submission's or a frame's end, a loss that used to be replayed) - taken up as a committed pass and served by the pass-copy code below.
    for (size_t ri = owned_newest_.size(); ri-- > 0;) {
      if (!OwnedCopyOfPass(info, owned_newest_[ri])) continue;
      OwnedPass taken_up = std::move(owned_newest_[ri]);
      owned_newest_.erase(owned_newest_.begin() + ri);
      if (p.valid) OwnedParkOrReplay("a copy of a surface's newest record");
      owned_pending_ = std::move(taken_up);
      owned_pending_.valid = true;
      owned_pending_.committed = true;
      owned_pending_.open = false;
      ++g_m2a.from_record;
      break;
    }
  }
  if (!OwnedChains()) {
    // (A copy that runs from the truth images over a parked pass only our image holds would read stale tiles.)
    const auto cannot = [&](const OwnedPass& q) {
      return (source & q.tiles).any() && !q.edram_free && !OwnedTruthHolds(q);
    };
    // (Session 82, P2: a copy of the pending pass's own tiles - the combine's, over a box that holds the parked blur's tiles
    // - is that pass's copy, not the game's over the parked one: the parked pass is dead under it, dropped, not paused.)
    const bool pending_copy = p.valid && (source & p.tiles).any() && !p.edram_free;
    for (const OwnedPass& q : owned_parked_) {
      if (!cannot(q) || pending_copy) continue;
      ++truth_counts_.owned_refused;
      OwnedPause(q.key, q.producers, "a copy of a pass only our image holds run as the game's");
    }
    OwnedDropParkedIf("a copy it cannot serve", cannot);
  }
  // (Session 82: a copy belongs to a defined-start pass by its surface, not by the tiles it shares with other surfaces - the
  // tile numbers of different surfaces mean nothing for a pass that owes the EDRAM nothing.)
  const bool same_surface = !info.IsCopyingDepth() && uint32_t(info.color_original_base) == p.base &&
                            info.color_edram_info.pitch_tiles == (p.pitch_tiles << (p.is_64bpp ? 1 : 0));
  const bool of_pass = p.valid && (OwnedChains() ? OwnedCopyOfPass(info, p) : (p.edram_free ? same_surface : (source & p.tiles).any()));
  const int32_t census_rec = of_pass ? p.census : owned_frame_last_;
  bool noted = false;
  const auto note = [&](int kind) {
    noted = true;
    if (!owned_frame_on_) return;
    ++owned_frame_stats_.copies;
    if (of_pass && p.chain_pass && kind >= 0 && kind <= 2) ++owned_frame_stats_.ch_served;
    if (of_pass && kind >= 0 && kind <= 2) p.chain_parked = false;  // (a served copy: its pixels are delivered; nothing is left to replay)
    if (census_rec < 0 || size_t(census_rec) >= owned_frame_passes_.size()) return;
    OwnedFramePass& r = owned_frame_passes_[census_rec];
    ++r.copies;
    if (kind == 0) ++r.served, ++owned_frame_stats_.served;
    if (kind == 1) ++r.page, ++owned_frame_stats_.page;
    if (kind == 2) ++r.dead, ++owned_frame_stats_.dead;
    if (kind == 3) ++r.written_back, ++owned_frame_stats_.written_back;
  };
  const auto from_truth = [&](int line = __builtin_LINE(), bool no_source = false) {
    if (OwnedOnly()) {
      // Session 92, M2a (design 4 item 6): there is no truth copy to run. A copy of a surface of a pass or a record was taken up above; this one has
      // no image of ours (or a shape/page our copy code cannot serve): DROPPED - the destination pages are the game's memory as they are (unmap), the
      // copy counted by the source line. M2c fills these with a defined value.
      if (!noted) note(-1);
      if (owned_wb_planned_) {
        // (Review C6: a write-back was planned - the copy IS written from our image by the cache's destination block; the destination is the game's
        // memory again (unmap), counted as written, not dropped.)
        unmap();
        ++g_m2a.copies_written_after_loss;
        return true;
      }
      // (Review C1: a dropped copy keeps the pages of ours that cover its destination - the cache marks a scaled destination resolved but nothing writes it, so an
      // unmap would leave garbage: cube feedback pages, damage maps and a front page stay as they were. Counted: "kept a page".)
      bool dest_is_ours = false;
      {
        const uint64_t dstart = info.copy_dest_extent_start, dend = dstart + info.copy_dest_extent_length;
        for (const OwnedPage& q : owned_pages_) {
          if (dstart < uint64_t(q.extent_start) + q.extent_length && uint64_t(q.extent_start) < dend) {
            ++g_m2a.drops_kept_page;
            dest_is_ours = true;
            break;
          }
        }
      }
      // Session 92, M2c: who the source is. Whether a pass of ours ever drew the surface (the registry), named in the group beside the line.
      std::string src_note;
      if (no_source) {
        const uint32_t src_base = uint32_t(info.IsCopyingDepth() ? info.depth_original_base : info.color_original_base);
        const int drawn = OwnedSourceDrawn(src_base, uint32_t(info.IsCopyingDepth() ? info.depth_edram_info.pitch_tiles : info.color_edram_info.pitch_tiles), info.IsCopyingDepth());
        static const char* const kDrawn[4] = {"surface not in the registry", "surface never drawn", "surface drawn in an earlier frame", "surface drawn this frame"};
        src_note = fmt::format(" [no pass or record holds it; {}]", kDrawn[drawn]);
        // The defined fill (design 2.5): not into a page of ours (it keeps its image) and only a shape the 8888 writer serves; else named and dropped.
        // (review F6: only a surface the registry never saw drawn - an evicted / superseded record or a reinterpretation drawn lately keeps the old drop, whose bytes are better than black; review F7: not a front page - a repeated frame beats black)
        bool dest_front = false;
        for (const uint32_t f : owned_front_pages_) dest_front = dest_front || (f && (f == page || f == texture_page));
        // Session 92, M4f-B: a colour clear pended for this very surface is its content - the fill takes its value (rounded as the target stores it) instead of transparent black,
        // and it is the newest thing that happened to the surface, so "drawn in an earlier frame" does not stop it (a pend is made after, and supersedes, every pass and record of the surface).
        float pend_value[4] = {};
        bool pend_fill = false;
        if (!info.IsCopyingDepth() && OwnedOnly() && OwnedPendBitOn()) {
          uint32_t fsx = 1, fsy = 1;
          command_processor_.NativeResolutionScale(fsx, fsy);
          const int32_t cx0 = int32_t(info.coordinate_info.edram_offset_x_div_8) * 8 * int32_t(fsx), cy0 = int32_t(info.coordinate_info.edram_offset_y_div_8) * 8 * int32_t(fsy);  // (review fix 7: the copy's real source rectangle)
          pend_fill = OwnedPendColourFor(src_base, uint32_t(info.color_edram_info.pitch_tiles), cx0, cy0, cx0 + int32_t(info.coordinate_info.width_div_8) * 8 * int32_t(fsx),
                                         cy0 + int32_t(info.height_div_8) * 8 * int32_t(fsy), pend_value);
        }
        if (drawn > 1 && !pend_fill) {
          src_note += " [not filled: the surface was drawn - a lost record, the old bytes stay]";
        } else if (dest_front) {
          src_note += " [not filled: a front page]";
        } else if (!dest_is_ours) {
          std::string why;
          if (OwnedWbPlanFill(info, why, pend_fill ? pend_value : nullptr)) {
            unmap();
            owned_wb_planned_ = true;  // (OwnedTakeResolve arms it: the cache's destination block writes the zero image)
            ++g_m2a.copies_filled;
            if (command_processor_.NativeCopyReadsBack()) ++g_m2a.readback_filled;
            M2aGroup(g_m2a.fill_groups, fmt::format("{} {}x{} base {} pitch {} -> page {:05X}{}", info.IsCopyingDepth() ? "depth" : "colour",
                                                    uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8, src_base,
                                                    uint32_t(info.color_edram_info.pitch_tiles), page, src_note));
            static uint32_t filled_logged = 0;
            if (filled_logged < 16) {
              ++filled_logged;
              REXGPU_INFO("[native][owned-only] frame {}: a copy of a surface no pass of ours holds is FILLED (defined transparent black): {} {}x{} base {} pitch {} into page 0x{:05X} dest format {}{}",
                          frame_, info.IsCopyingDepth() ? "depth" : "colour", uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8, src_base,
                          uint32_t(info.color_edram_info.pitch_tiles), page, uint32_t(info.copy_dest_info.copy_dest_format), src_note);
            }
            return true;
          }
          src_note += fmt::format(" [not filled: {}]", why);
        } else {
          src_note += " [not filled: the destination is a page of ours]";
        }
      } else if (!g_drop_note.empty()) {
        src_note = " [" + g_drop_note + "]";
      }
      g_drop_note.clear();
      ++g_m2a.copies_dropped;
      if (command_processor_.NativeCopyReadsBack()) ++g_m2a.readback_dropped;  // (review C3: the bar says 0)
      for (const uint32_t f : owned_front_pages_) {
        if (f && (f == page || f == texture_page)) {
          ++g_m2a.front_dropped;
          break;
        }
      }
      M2aGroup(g_m2a.copy_groups, fmt::format("{} {}x{} base {} pitch {} -> page {:05X} fmt {}", info.IsCopyingDepth() ? "depth" : "colour",
                                                uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8,
                                                uint32_t(info.IsCopyingDepth() ? info.depth_original_base : info.color_original_base),
                                                uint32_t(info.color_edram_info.pitch_tiles), page, uint32_t(info.copy_dest_info.copy_dest_format)) + fmt::format(" (line {})", line) + src_note);
      static uint32_t logged = 0;
      if (logged < 24) {
        ++logged;
        REXGPU_INFO("[native][owned-only] frame {}: a copy is dropped (owned_steps.cpp:{}): {} {}x{} base {} pitch {} into page 0x{:05X} (texture 0x{:05X}) dest format {}; EDRAM offset {},{}; pending pass {} (base {} pitch {}), {} records{}",
                    frame_, line, info.IsCopyingDepth() ? "depth" : "colour", uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8,
                    uint32_t(info.IsCopyingDepth() ? info.depth_original_base : info.color_original_base), uint32_t(info.color_edram_info.pitch_tiles), page,
                    texture_page, uint32_t(info.copy_dest_info.copy_dest_format), uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8,
                    uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8, p.valid, p.base, p.pitch_tiles, owned_newest_.size(), src_note);
      }
      return true;
    }
    if (!noted) note(-1);
    if (OwnedChains() && owned_frame_on_ && !owned_frame_end_carried_.empty() && !info.IsCopyingDepth()) {
      // (Session 90, detection only: a copy that runs from the truth tiles of a surface whose carried defined start was dropped at the last frame's end.)
      for (const OwnedCarriedGone& g : owned_frame_end_carried_) {
        if (g.base == uint32_t(info.color_original_base) && g.pitch == uint32_t(info.color_edram_info.pitch_tiles)) {
          ++owned_frame_stats_.ch_copy_of_dropped_carried;
          if (owned_carried_gone_logged_ < 12) {
            ++owned_carried_gone_logged_;
            REXGPU_INFO("[native][owned] frame {}: a copy of a carried pass dropped at the last frame end runs from the truth tiles: base {} pitch {} page 0x{:05X}",
                        frame_, g.base, g.pitch, page);
          }
          break;
        }
      }
    }
    unmap();
    if (!owned_dirty_.empty()) {
      OwnedDirtyBeforeRead(source, -1, "a truth copy");
      if (OwnedChains()) ++owned_frame_stats_.ch_materialised;
    }
    if (owned_frame_on_) ++owned_frame_stats_.truth_copies;
    return false;
  };
  // Session 82 (P2): this copy's bytes go to the CPU (readback window, not a recurring destination): not served from our
  // image - the pass is replayed on the truth path, the game's copy reads it back. No pause: a copy that is read back
  // today is served once its destination has recurred.
  // (Not into a texture one of our pages already holds - the damage / sweat maps' feedback chain, the wrestler maps: a texture
  // the GPU reads through our page, not a picture the CPU keeps. Served as outside a window: a truth copy there would fork
  // the chain - our image and the game's memory - and drop the page. Entrance replay, run 8: the wrestlers' skin differed
  // from the owned-off arm in the match after the window; 0x1EDCA was the copy read back at the window's frames.)
  const bool dest_ours = std::any_of(owned_pages_.begin(), owned_pages_.end(),
                                     [&](const OwnedPage& q) { return q.page == texture_page && q.feedback >= 0; });  // (review: feedback pages only)
  // (A committed pass whose pixels only our image holds (a covered start the truth images cannot hold) cannot be read back: a replay is
  // impossible and the game's copy would read tiles that never had its draws - an inexact pause. Served, as before the window rule.)
  const bool only_ours = p.valid && p.committed && !p.edram_free && !OwnedTruthHolds(p) && !OwnedOnly();  // (Session 92, M2a: under the latch every pass is "only ours" - a read-back copy is written from our image or dropped, never run by the game)
  if (only_ours && owned_frame_on_ && !dest_ours && command_processor_.NativeCopyReadsBack()) {
    ++owned_frame_stats_.ch_only_ours_readback;  // (review A9: the removed readback guard reopens this case - counted)
    // (Session 82, P2: the case with no exact answer yet - a copy the CPU reads, of a committed pass only our image holds: served, the CPU
    // keeps the game's old bytes. Counted in the log to see whether it ever happens: the scope doc, section 20.)
    static uint32_t unserved_logged = 0;
    if (unserved_logged < 12) {
      ++unserved_logged;
      REXGPU_INFO("[native][owned] frame {}: a copy the CPU reads back (page 0x{:05X}, {}x{}) of a committed pass only our image holds (base {} "
                  "pitch {} {}x{}): served, not read back", frame_, page, uint32_t(info.coordinate_info.width_div_8) * 8,
                  uint32_t(info.height_div_8) * 8, p.base, p.pitch_tiles, p.width, p.height);
    }
  }
  // (Session 90, Q: wb_wanted - the CPU is to get this copy's bytes (the readback mode is on for it, the copy wide enough): the copy is served as
  // outside a window AND written from our image. reads_back - the same for a copy none of our pages / committed passes holds: if no writer exists
  // for it, the old replay stays (counted by reason).)
  const bool wb_wanted = owned_frame_on_ && command_processor_.NativeCopyReadsBack() &&
                         int32_t(info.coordinate_info.width_div_8) * 8 >= REXCVAR_GET(gpu_native_owned_frame_readback_min_w);
  const bool reads_back = wb_wanted && !dest_ours && !only_ours;
  if (OwnedChains() && owned_drop_.valid && !OwnedKeyDead(owned_drop_.key)) owned_drop_.valid = false;  // (review A6: a revived key's copy is the game's)
  // (Session 95, review r5, 1: a drop no copy took. A read-back of the dropped surface - the attire latch's first copy can still get here: it arms in the copy itself - takes it, and the frame after,
  // a copy of that surface is another pass's: left valid, it would be called "not drawn" and its page unmapped. The read-back goes on and is counted with the dropped read-backs below.)
  if (owned_drop_.valid && (reads_back || owned_drop_.frame != frame_) && !info.IsCopyingDepth() && uint32_t(info.color_original_base) == owned_drop_.base &&
      info.color_edram_info.pitch_tiles == owned_drop_.pitch) {
    owned_drop_.valid = false;
    static uint32_t unconsumed_logged = 0;
    if (unconsumed_logged < 12) {
      ++unconsumed_logged;
      REXGPU_INFO("[native][owned-dead] frame {}: a copy of the dropped surface {}/{} (ps {:016X}, dropped frame {}) is not the drop's: {}", frame_, owned_drop_.base, owned_drop_.pitch,
                  owned_drop_.key.ps, owned_drop_.frame, reads_back ? "the CPU reads it back, its bytes are lost" : "a later frame's");
    }
  }
  if (owned_drop_.valid && !reads_back && !info.IsCopyingDepth() && uint32_t(info.color_original_base) == owned_drop_.base &&
      info.color_edram_info.pitch_tiles == owned_drop_.pitch) {
    owned_drop_.valid = false;  // (the copy of a pass that was not drawn: nothing to copy)
    if (REXCVAR_GET(gpu_native_crowd_pose_half) && OwnedPoseKey(owned_drop_.key.ps, owned_drop_.key.pitch)) {
      bool learned = false;
      for (const OwnedKeyStat& k : owned_key_stats_) {
        if (k.key == owned_drop_.key) learned = k.dead;
      }
      if (!learned) {
        // (Session 95: a half-rate drop of a live pose key: the page is NOT unmapped and not listed dead - the cards keep sampling our image, last frame's pose.)
        if (g_pose_dropped_frame != frame_) {
          g_pose_dropped_frame = frame_;
          ++g_pose_frames;
          if (g_pose_frames == 1) {
            REXGPU_INFO("[native][pose-half] frame {}: first pose pass dropped (ps {:016X} surface {}/{}); the cards sample last frame's pages", frame_, owned_drop_.key.ps, owned_drop_.key.base, owned_drop_.key.pitch);
          }
          if (frame_ / 600 != g_pose_logged_to) {
            g_pose_logged_to = frame_ / 600;
            REXGPU_INFO("[native][pose-half] frame {}: {} frames with the pose passes dropped so far", frame_, g_pose_frames);
          }
        }
        note(2);
        return true;
      }
    }
    unmap();
    if (owned_dead_pages_.size() < 256) {
      owned_dead_pages_.push_back({page, texture_page, owned_drop_.key});
      // (Session 87, step 4, gpu_cp_trim: the pages of the list by their low 17 bits; a clear bit means no entry has the page.)
      owned_dead_mask_[(page & 0x1FFFF) >> 6] |= uint64_t(1) << (page & 63);
      owned_dead_mask_[(texture_page & 0x1FFFF) >> 6] |= uint64_t(1) << (texture_page & 63);
    }
    note(2);
    return true;
  }
  if (!of_pass && owned_frame_on_ && p.valid && !p.committed && !OwnedChains()) {  // (chains: another surface's copy does not concern the pass)
    OwnedReplayPending("a truth copy before its pass's copy");  // (review: as a truth draw that is not its own)
  }
  if (!of_pass) return from_truth(__LINE__, true);  // (Session 92, M2c: no pass or record of ours holds the source surface)
  // Session 82 (the owned frame): a copy into a page the swap presented lately is the front buffer's - the presentation
  // reads the texture, not our image (until the swap hook, P2).
  const auto is_front = [&](uint32_t pg) {
    for (const uint32_t f : owned_front_pages_) {
      if (f && f == pg) return true;
    }
    return false;
  };
  const auto lose = [&](bool present = false, bool remember = true, bool pause = true) {
    // (Session 82, P2, G4: a further copy of a pass that fed a feedback texture, one the feedback path cannot serve - the
    // packed mip tail's strips - is dropped: the texture of ours has no memory to write, and the game's copy from the truth
    // images would read tiles this pass never wrote.)
    // (Session 82, P0, from the RenderDoc captures of the entrance replay's match: only the damage maps' k_4_4_4_4 tail strips - a cube's
    // face copy (k_8_8_8_8) that cannot be served is NOT dropped: the light cubes of 0x1AB06, 0x1AE86 ... were never written, black in
    // the scene's lighting. It takes the ordinary loss below: its key paused, the game's own passes and copies write the memory.)
    if (p.valid && p.committed && p.edram_free && p.feedback_pass &&
        info.copy_dest_info.copy_dest_format == xenos::ColorFormat::k_4_4_4_4) {
      static uint32_t dropped_logged = 0;
      if (segment_log_frames_ > 0 || dropped_logged < 12) {
        ++dropped_logged;
        REXGPU_INFO("[native][owned] frame {}: a copy of a feedback pass dropped: page 0x{:05X} (texture 0x{:05X}) format {} "
                    "{}x{} from {},{}", frame_, page, texture_page, uint32_t(info.copy_dest_info.copy_dest_format),
                    uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8,
                    uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8,
                    uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8);
      }
      note(2);
      return true;
    }
    // Session 92, M2a (design 4 item 6): the pass cannot serve this copy (a shape, a page, a front copy the swap would not read ...) and nothing replays it:
    // the copy is DROPPED, counted by the line that lost it; the pass is untouched (it may serve a later copy, or end as its surface's newest record).
    if (OwnedOnly()) {
      if (OwnedChains()) {
        auto& ll = owned_frame_stats_.lose_lines;
        auto it = std::find_if(ll.begin(), ll.end(), [&](const std::pair<int, uint32_t>& e) { return e.first == g_owned_lose_line; });
        if (it != ll.end()) ++it->second; else if (ll.size() < 64) ll.emplace_back(g_owned_lose_line, 1u);
      }
      return from_truth(g_owned_lose_line);
    }
    // (The first ones logged with the copy against the box.)
    static uint32_t losses_logged = 0;
    static uint32_t ps9c_logged = 0;
    if (OwnedChains() && owned_frame_on_) {
      auto& ll = owned_frame_stats_.lose_lines;
      auto it = std::find_if(ll.begin(), ll.end(), [&](const std::pair<int, uint32_t>& e) { return e.first == g_owned_lose_line; });
      if (it != ll.end()) ++it->second; else if (ll.size() < 64) ll.emplace_back(g_owned_lose_line, 1u);
    }
    const bool ps9c = p.key.ps == 0x9C85DA5F0879DA76ull && ps9c_logged < 12;
    if (ps9c) ++ps9c_logged;
    if (ps9c) REXGPU_INFO("[native][owned] frame {}: 9C85DA5F loss at owned_steps.cpp line {}: copy {}x{} pass box {}x{} start {} cleared {} edram_free {} committed {} truth_holds {} copies {}", frame_, g_owned_lose_line, uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8, p.width, p.height, int(p.start_kind), p.cleared, p.edram_free, p.committed, OwnedTruthHolds(), p.copy_count);
    if (segment_log_frames_ > 0 || losses_logged < (OwnedChains() ? 400u : 40u) || ps9c) {
      ++losses_logged;
      REXGPU_INFO("[native][owned] frame {}: owned pass {} - a copy of its tiles into page 0x{:05X} not of its box / "
                  "plain: known {} depth {} source {} format {} pitch {} msaa {} offset {},{} size {}x{} span base {} "
                  "rows {} row length {}; dest format {} number {} bias {} swap {} array {} at {},{} / {},{}; box base "
                  "{} pitch {} {}x{} ({} draws)",
                  frame_, p.committed ? "ended" : "replayed", page, copy_known, info.IsCopyingDepth(),
                  uint32_t(info.rb_copy_control.copy_src_select), uint32_t(info.color_edram_info.format),
                  uint32_t(info.color_edram_info.pitch_tiles), uint32_t(info.color_edram_info.msaa_samples),
                  uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8,
                  uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8,
                  uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8, span_base, rows,
                  row_length, uint32_t(info.copy_dest_info.copy_dest_format),
                  uint32_t(info.copy_dest_info.copy_dest_number), int32_t(info.copy_dest_info.copy_dest_exp_bias),
                  uint32_t(info.copy_dest_info.copy_dest_swap), uint32_t(info.copy_dest_info.copy_dest_array),
                  uint32_t(info.copy_dest_base_x), uint32_t(info.copy_dest_base_y),
                  uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8,
                  uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8, p.base, p.pitch_tiles, p.width,
                  p.height, p.draws.size());
      if (OwnedChains()) REXGPU_INFO("[native][owned]   (that loss came from owned_steps.cpp line {}, present {} remember {} pause {})", g_owned_lose_line, present, remember, pause);
    }
    // (Review: a committed pass only our image holds cannot be replayed - this copy reads truth tiles without its
    // draws: its producers not owned for a while.)
    // (Session 90: a carried defined start has no draws to replay, and the truth tiles never got its pixels: a copy of it that is not served
    // reads stale tiles every frame - its key is paused, whatever OwnedTruthHolds says for an edram_free pass.)
    if (p.committed && (p.carried_defined || (!OwnedTruthHolds() && !(p.edram_free && !p.draws.empty())))) {
      if (p.carried_defined) ++owned_frame_stats_.ch_carried_paused_copy;
      ++truth_counts_.owned_refused;
      OwnedPause(p.key, p.producers, "a copy of a pass only our image holds run as the game's");
    }
    if (present) {
      // (Not a loss to pause: this pass's copy is the front buffer's for as long as the swap reads the texture - its
      // key is refused up front from now on.)
      if (OwnedChains()) ++owned_frame_stats_.ch_front_lost;
      if (remember) OwnedPresentAdd(p.key, "a copy into the front buffer lost");
      OwnedReplayPending("a copy into the front buffer", false);
    } else {
      OwnedReplayPending("a copy of its tiles not of its box / not plain", pause, true);  // (a committed defined-start pass is replayed: its copy follows)
    }
    return from_truth();
  };
  // (Session 82, P2: a copy into the front buffer is left out when it is a whole plain copy the swap reads exactly as it
  // is written - checked below, with the box known - else it is the truth path's, as in P0a.)
  bool front = owned_frame_on_ && (is_front(page) || is_front(texture_page));  // (Session 90, Q: not const - a front copy the swap would not read, written from our image, goes on as a page)
  // Session 82 (P1 item 3, owned_scene.cpp): the scene's depth copy into a depth page of ours (never a front buffer).
  bool wb_depth = false;
  if (wb_wanted && owned_frame_on_ && !front && copy_known && info.IsCopyingDepth() && !REXCVAR_GET(gpu_native_own_steps_debug_replay)) {
    std::string wb_why;
    wb_depth = OwnedWbPlanDepth(info, wb_why);  // (Session 90, Q: the depth writer exists - the copy is served and written)
    if (!wb_depth) OwnedWbRefused("depth: " + wb_why);
  }
  if (owned_frame_on_ && !front && (!reads_back || wb_depth) && copy_known && info.IsCopyingDepth() &&
      !REXCVAR_GET(gpu_native_own_steps_debug_replay)) {
    std::string depth_why;
    if (OwnedDepthCopy(info, page, texture_page, depth_why)) {
      p.draws_since_depth = 0;  // (Session 82, P2, review: the restore skip's proof)
      p.copy_depth_page = texture_page;
      owned_wb_planned_ = wb_depth;
      note(1);
      return true;
    }
    static uint32_t depth_refusals_logged = 0;
    if (depth_refusals_logged < 12 || owned_frame_logging_) {
      ++depth_refusals_logged;
      REXGPU_INFO("[native][owned] frame {}: the depth copy into page 0x{:05X} is not taken: {}", frame_, page, depth_why);
    }
    wb_depth = false;
  }
  if (!copy_known || info.IsCopyingDepth() || info.rb_copy_control.copy_src_select != 0 ||
      REXCVAR_GET(gpu_native_own_steps_debug_replay)) {
    if (reads_back && info.IsCopyingDepth()) owned_replay_reason_ = kOwnedReasonReadback;
    return OWNED_LOSE(front, true, !reads_back);
  }
  // (Session 90, Q: the read-back replay / materialise that stood here moved below the shape check: it runs only for a copy no writer exists for.)
  // A plain copy of a rectangle of the box: from an RGBA8 surface an 8-bit-per-channel unsigned destination (k_8: the
  // red channel; k_8_8_8_8: all four, red and blue swapped or not - the view undoes it); from a 16-bit one the same
  // format (the packed values as they are, as TruthResolveScaledFull copies them); no exponent bias. Its rectangle in
  // the surface: its first tile's place from the surface's base, plus its offset (a 64bpp surface's tiles are 40
  // pixels wide, its EDRAM rows twice its pitch).
  const reg::RB_COPY_DEST_INFO d = info.copy_dest_info;
  const auto src_format = xenos::ColorRenderTargetFormat(uint32_t(info.color_edram_info.format));
  const bool format_ok =
      p.kind == 2
          ? ((src_format == xenos::ColorRenderTargetFormat::k_16_16 &&
              d.copy_dest_format == xenos::ColorFormat::k_16_16) ||
             (src_format == xenos::ColorRenderTargetFormat::k_16_16_16_16 &&
              d.copy_dest_format == xenos::ColorFormat::k_16_16_16_16)) &&
                !d.copy_dest_swap
          : src_format == xenos::ColorRenderTargetFormat::k_8_8_8_8 &&
                (d.copy_dest_format == xenos::ColorFormat::k_8 ||
                 d.copy_dest_format == xenos::ColorFormat::k_8_8_8_8 ||
                 d.copy_dest_format == xenos::ColorFormat::k_4_4_4_4) &&
                d.copy_dest_number == xenos::SurfaceNumberFormat::kUnsignedRepeatingFraction &&
                (!d.copy_dest_swap || d.copy_dest_format == xenos::ColorFormat::k_8_8_8_8 ||
                 d.copy_dest_format == xenos::ColorFormat::k_4_4_4_4);
  const uint32_t row_tiles = p.pitch_tiles << (p.is_64bpp ? 1 : 0);
  const uint32_t tile_w = p.is_64bpp ? 40u : 80u;
  const uint32_t delta = (uint32_t(info.color_edram_info.base_tiles) - p.base) & 2047u;
  const uint32_t sx = (delta % std::max(1u, row_tiles)) * tile_w + uint32_t(info.coordinate_info.edram_offset_x_div_8) * 8;
  const uint32_t sy = (delta / std::max(1u, row_tiles)) * 16 + uint32_t(info.coordinate_info.edram_offset_y_div_8) * 8;
  uint32_t w = uint32_t(info.coordinate_info.width_div_8) * 8, h = uint32_t(info.height_div_8) * 8;
  // (Session 82, P2: the copy's 8-pixel rounding past the box - the 320x184 copies of 320x180 - clamped to the box.)
  const uint32_t w_copied = w, h_copied = h;
  const bool box_ok = OwnedCopyBox(p, sx, sy, w, h);
  if (box_ok && (w != w_copied || h != h_copied)) ++owned_frame_stats_.post_boxes;
  const uint32_t dx = uint32_t(info.copy_dest_base_x) + uint32_t(info.copy_dest_coordinate_info.offset_x_div_8) * 8;
  const uint32_t dy = uint32_t(info.copy_dest_base_y) + uint32_t(info.copy_dest_coordinate_info.offset_y_div_8) * 8;
  if (!format_ok || bool(info.color_edram_info.format_is_64bpp) != p.is_64bpp ||
      info.color_edram_info.pitch_tiles != row_tiles || uint32_t(info.color_original_base) != p.base ||
      info.color_edram_info.msaa_samples != xenos::MsaaSamples::k1X || !w || !h || !box_ok ||
      d.copy_dest_exp_bias || d.copy_dest_array) {
    // (Session 82, P2, diagnostic: every condition of the shape check for the 9C85DA5F pass - the 1440p pause.)
    static uint32_t shape_logged = 0;
    if (p.key.ps == 0x9C85DA5F0879DA76ull && shape_logged < 12) {
      ++shape_logged;
      REXGPU_INFO("[native][owned] frame {}: 9C85DA5F shape check failed: format_ok {} (src fmt {} dest fmt {} number {} swap {}) is64 {} vs {} "
                  "pitch_tiles {} vs row_tiles {} color_original_base {} vs p.base {} msaa {} w {} h {} (copy {}x{}) box_ok {} sx {} sy {} "
                  "delta {} row_tiles {} p.width {} p.height {} exp_bias {} array {}",
                  frame_, format_ok, uint32_t(src_format), uint32_t(d.copy_dest_format), uint32_t(d.copy_dest_number),
                  uint32_t(d.copy_dest_swap), bool(info.color_edram_info.format_is_64bpp), p.is_64bpp,
                  uint32_t(info.color_edram_info.pitch_tiles), row_tiles, uint32_t(info.color_original_base), p.base,
                  uint32_t(info.color_edram_info.msaa_samples), w, h, w_copied, h_copied, box_ok, sx, sy, delta, row_tiles,
                  p.width, p.height, int32_t(d.copy_dest_exp_bias), uint32_t(d.copy_dest_array));
    }
    if (OwnedOnly()) {
      // Session 92, M2c ("identify first"): which condition of the shape check refused the copy, for the drop's group (the census names it) and the first ones in full.
      g_drop_note = fmt::format("shape: {}{}{}{}{}{}{}{} pass ps {:016X} box {}x{} start {} edram_free {} copy {}x{} at {},{} -> box_ok {} w {} h {} fmt {}->{} swap {} number {}",
                                format_ok ? "" : "format ", bool(info.color_edram_info.format_is_64bpp) != p.is_64bpp ? "is64 " : "",
                                info.color_edram_info.pitch_tiles != row_tiles ? "pitch " : "", uint32_t(info.color_original_base) != p.base ? "base " : "",
                                info.color_edram_info.msaa_samples != xenos::MsaaSamples::k1X ? "msaa " : "", (!w || !h) ? "empty " : "", !box_ok ? "box " : "",
                                (d.copy_dest_exp_bias || d.copy_dest_array) ? "bias/array " : "", p.key.ps, p.width, p.height, int(p.start_kind), p.edram_free, w_copied,
                                h_copied, sx, sy, box_ok, w, h, uint32_t(src_format), uint32_t(d.copy_dest_format), uint32_t(d.copy_dest_swap), uint32_t(d.copy_dest_number));
      static uint32_t shape_only_logged = 0;
      if (shape_only_logged < 16) {
        ++shape_only_logged;
        REXGPU_INFO("[native][owned-only] frame {}: a copy fails the shape check: {} (page 0x{:05X}, reads back {})", frame_, g_drop_note, page, reads_back);
      }
    }
    if (OwnedOnly() && !box_ok && format_ok && bool(info.color_edram_info.format_is_64bpp) == p.is_64bpp && info.color_edram_info.pitch_tiles == row_tiles &&
        uint32_t(info.color_original_base) == p.base && info.color_edram_info.msaa_samples == xenos::MsaaSamples::k1X && w_copied && h_copied &&
        !d.copy_dest_exp_bias && !d.copy_dest_array && sx == 0 && sy == 0 && !front && p.kind == 1 && p.start_kind == kOwnedStartCleared) {
      // Session 92, M2d (Part 4, S92_M1A_BOX_READ.md B): the ONLY failing term is the box - a copy larger than the box of the held-clear pass it reads (512x512 pitch 7
      // from a 512x256 pass). Written from a composite image: the pass's image in the top, a defined transparent-black fill below (hard rule 6: the 360's lower half is other
      // surfaces' EDRAM read as colour). Covers the copies inside read-back windows and outside; a shape the writer cannot take stays the drop below, named.
      std::string composite_why;
      if (OwnedWbPlanComposite(info, w_copied, h_copied, composite_why)) {
        g_drop_note.clear();
        unmap();
        owned_wb_planned_ = true;  // (OwnedTakeResolve arms it: the cache's destination block writes the composite)
        note(3);
        ++g_m2a.composites;
        ++g_m2a.composite_lower_filled;
        if (reads_back) ++g_m2a.composite_readback;
        M2dWatchAdd(page);
        if (texture_page != page) M2dWatchAdd(texture_page);
        M2aGroup(g_m2a.copy_groups, fmt::format("SERVED AS COMPOSITE (not dropped) colour {}x{} base {} pitch {} -> page {:05X} fmt {}: pass box {}x{} in the top, the lower part filled", w_copied, h_copied,
                                                 p.base, uint32_t(info.color_edram_info.pitch_tiles), page, uint32_t(d.copy_dest_format), p.width, p.height));
        static uint32_t composite_logged = 0;
        if (composite_logged < 8) {
          ++composite_logged;
          REXGPU_INFO("[native][owned-only] frame {}: a copy larger than its pass's box is SERVED AS COMPOSITE: copy {}x{} pass box {}x{} (start {}) base {} pitch {} into page 0x{:05X} (texture 0x{:05X}) dest format {} "
                      "reads back {}; the pass's image in the top, the lower {} rows a defined transparent black",
                      frame_, w_copied, h_copied, p.width, p.height, int(p.start_kind), p.base, p.pitch_tiles, page, texture_page, uint32_t(d.copy_dest_format), reads_back, h_copied - p.height);
        }
        return true;
      }
      static uint32_t composite_refused_logged = 0;
      if (composite_refused_logged < 8) {
        ++composite_refused_logged;
        REXGPU_INFO("[native][owned-only] frame {}: a box-only shape failure could not be written as a composite: {}", frame_, composite_why);
      }
    }
    if (reads_back) {  // (Session 90, Q: as before - a read-back copy of another shape is replayed, no pause; no writer exists for it)
      OwnedWbRefused("shape: not a plain copy of the pass's box");
      owned_replay_reason_ = kOwnedReasonReadback;
      return OWNED_LOSE(front, true, false);
    }
    return OWNED_LOSE(front);
  }
  // (Session 82, P2, G4: a defined start away from the origin is trusted only where a draw covered; not paused - the
  // chain's first build has copies of other shapes.)
  if (p.edram_free && p.nonorigin && !OwnedRectDrawn(p, sx, sy, w, h)) {
    if (reads_back) owned_replay_reason_ = kOwnedReasonReadback;
    return OWNED_LOSE(front, true, false);
  }
  // Session 90 (Q): the copy's bytes go to the CPU - plan the write from our image. The copy is then served exactly as outside a window (below)
  // and the render target cache writes the bytes. A copy no writer exists for keeps the old replay (counted, with the reason).
  if (wb_wanted) {
    uint32_t plan_scale_x = 1, plan_scale_y = 1;
    command_processor_.NativeResolutionScale(plan_scale_x, plan_scale_y);
    const uint32_t plan_gap = info.color_edram_info.fill_half_pixel_offset ? (plan_scale_x / 2) | ((plan_scale_y / 2) << 8) : 0u;
    std::string plan_why;
    if (OwnedWbPlan(info, sx, sy, w, h, plan_gap, front, plan_why)) {
      owned_wb_planned_ = true;
      ++owned_frame_stats_.wb_planned;
      {
        // (Session 90, Q diagnosis: the state of the pass a write-back is planned from - the garbled Heenan attire.)
        static uint32_t plan_logged = 0;
        if (plan_logged < 60) {
          ++plan_logged;
          REXGPU_INFO("[native][owned][wbdiag] frame {}: write-back planned page 0x{:05X} {}x{} at {},{}: pass base {} pitch {} kind {} box {}x{} (host {}x{}) start_kind {} edram_free {} nonorigin {} cleared {} committed {} loaded_start {} "
                      "draws {} timer_draws {} drawn [{},{},{},{}] copies so far {} image {} dest fmt {} swap {} endian {} number {} census {} first draw: mask {:X} blend {:08X} ps {:016X} all-mask {:X}",
                      frame_, (info.copy_dest_base >> 12) & 0x1FFFF, w, h, sx, sy, p.base, p.pitch_tiles, p.kind, p.width, p.height, p.host_width, p.host_height,
                      int(p.start_kind), p.edram_free, p.nonorigin, p.cleared, p.committed, p.loaded_start, p.draws.size(), p.timer_draws, p.drawn[0], p.drawn[1],
                      p.drawn[2], p.drawn[3], p.copy_count, p.image, uint32_t(info.copy_dest_info.copy_dest_format), uint32_t(info.copy_dest_info.copy_dest_swap),
                      uint32_t(info.copy_dest_info.copy_dest_endian), uint32_t(info.copy_dest_info.copy_dest_number), p.census,
                      p.draws.empty() ? 0u : p.draws[0].color_write_mask, p.draws.empty() ? 0u : p.draws[0].blend_control, p.draws.empty() ? 0ull : p.draws[0].pixel_shader_hash,
                      p.draws.empty() ? 0u : p.draws[0].color_mask_all);
        }
      }
      if (dest_ours) ++owned_frame_stats_.wb_dest_ours;
      if (only_ours) ++owned_frame_stats_.wb_only_ours;
    } else {
      OwnedWbRefused(plan_why);
    }
  }
  if (reads_back && !owned_wb_planned_) {
    // (Session 82, P2 / Session 87: no writer for this copy - the pass is replayed on the truth path and the game's copy reads it back.)
    ++truth_counts_.owned_readback_copies;
    static uint32_t readback_logged = 0;
    if (segment_log_frames_ > 0 || owned_frame_logging_ || readback_logged < 40) {
      ++readback_logged;
      REXGPU_INFO("[native][owned] frame {}: a copy the CPU reads back (page 0x{:05X}, {}x{}, dest format {}) has no writer from our image: owned pass {} "
                  "(kind {}, edram_free {}, feedback {}, copies so far {}, {} draws), the game's copy runs", frame_, page,
                  uint32_t(info.coordinate_info.width_div_8) * 8, uint32_t(info.height_div_8) * 8,
                  uint32_t(info.copy_dest_info.copy_dest_format), p.committed ? "ended" : "replayed", p.kind, p.edram_free,
                  p.feedback_pass, p.copy_count, p.draws.size());
    }
    if (OwnedChains() && !OwnedOnly() && REXCVAR_GET(gpu_native_owned_frame_readback_materialise) && p.valid && !p.edram_free && !p.half && !front &&
        p.color_slot >= 0 && size_t(p.color_slot) < truth_images_.size()) {
      // (Session 87, chains, kept behind gpu_native_owned_frame_readback_materialise for one A/B: the bytes the CPU reads are the pass's pixels - they go to the truth image as its dirty tiles (OwnedCommit) and the game's own copy runs from
      // there. Only when the truth image owns every tile of the box - else replayed.)
      const TruthImage& tt = truth_images_[p.color_slot];
      const std::bitset<2048> box = EdramTiles(tt.base, tt.pitch_tiles, tt.is_64bpp(), 0, 0, p.width, p.height);
      if ((box & ~tt.owned).none() && tt.initialized && tt.image != VK_NULL_HANDLE && (p.committed || p.cleared || p.start_kind == kOwnedStartCovered || OwnedTruthHolds())) {
        OwnedCommit();
        ++owned_frame_stats_.ch_readback_mat;
        return from_truth();
      }
    }
    owned_replay_reason_ = kOwnedReasonReadback;
    return OWNED_LOSE(front, true, false);
  }
  const bool whole = sx == 0 && sy == 0 && w == p.width && h == p.height && dx == 0 && dy == 0 && page == texture_page;
  if (whole) {  // (Session 82, P2, review: the restore skip's proof counts from the last WHOLE copy, and names its texture)
    p.draws_since_color = 0;
    p.copy_color_page = texture_page;
  }
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  // (Review: at a resolution scale with the half-pixel offset the copy fills its first host columns / rows from the
  // first covered one - the texture fill takes that gap; a whole 8-bit page is a snapshot of our image with it filled.)
  const uint32_t gap = info.color_edram_info.fill_half_pixel_offset ? (scale_x / 2) | ((scale_y / 2) << 8) : 0u;
  if (front) {
    // (Session 82, P2: the swap reads this page - our image goes to the presenter instead (owned_post.cpp). Only a whole
    // copy of an RGBA8 pass to a k_8_8_8_8 page the swap has read lately with the same size, format, byte order and
    // scaled flag; else the truth path's copy - not remembered while the swap has not read the page yet.)
    bool seen = false;
    for (const OwnedSwapSeen& s : owned_swap_seen_) {
      // (chains: a page the swap read long ago is not 'seen' - staleness alone, after a load, must not ban the key.)
      if (s.page == page && s.page && (!OwnedChains() || s.frame + 16 >= frame_)) seen = true;
    }
    const bool scaled_now = scale_x > 1 || scale_y > 1;
    // (A defined start - a pass blended onto a cleared box - is not left out here: the surface may hold what the truth
    // path drew there, which the box lacks (the entrance's overlay pass: black frame). Such a pass stays the truth path's.)
    // (chains: only a converted start away from the origin cannot be served - every other start holds the whole box.)
    // (review A1: a converted start can have transparent-black holes over content, at the origin too - every converted start is refused under chains.)
    bool page_unserved = false;
    if (OwnedChains()) {
      for (const auto& u : owned_unserved_pages_) {
        if ((u.first == page || u.first == texture_page) && frame_ < u.second) page_unserved = true;  // (review A2)
      }
    }
    // Session 92, M2a (review C2): a front copy is never unmap-dropped under the latch. The converted-start term goes (a converted start is a defined box, the
    // best image there is), and a front copy the swap cannot read from our image is WRITTEN into the page's memory from it (the front write-back, as the
    // wb_rect shape below); only when no writer exists is it dropped, keeping the page (from_truth).
    const bool front_unserved = OwnedChains() ? ((p.start_kind == kOwnedStartConverted && !OwnedOnly()) || page_unserved) : p.edram_free;
    const auto front_write = [&]() -> bool {
      if (!OwnedOnly() || p.half) return false;
      std::string why;
      if (!OwnedWbPlan(info, sx, sy, w, h, gap, /*front=*/true, why)) {
        OwnedWbRefused("front: " + why);
        return false;
      }
      owned_wb_planned_ = true;
      OwnedCommit();
      ++p.copy_count;
      unmap();
      ++owned_frame_stats_.wb_rect;
      ++g_m2a.front_written;
      p.chain_parked = false;
      note(3);
      return true;
    };
    if (!whole || p.kind != 1 || d.copy_dest_format != xenos::ColorFormat::k_8_8_8_8 || front_unserved) {
      if (OwnedChains()) {
        static uint32_t front_logged = 0;
        if (front_logged < 40) {
          ++front_logged;
          REXGPU_INFO("[native][owned] frame {}: a copy into the front buffer is not taken (page 0x{:05X}): whole {} (sx {} sy {} {}x{} vs box {}x{}, dx {} dy {}, page {:X} texture {:X}) kind {} dest format {} edram_free {} start {} draws {}",
                      frame_, page, whole, sx, sy, w, h, p.width, p.height, dx, dy, page, texture_page, p.kind, uint32_t(d.copy_dest_format), p.edram_free,
                      int(p.start_kind), p.draws.size());
        }
      }
      if (front_write()) return true;
      return OWNED_LOSE(true);
    }
    if (!OwnedPresentOk(page, w, h, uint32_t(d.copy_dest_format), uint32_t(d.copy_dest_endian), scaled_now)) {
      if (OwnedChains()) {
        static uint32_t ok_logged = 0;
        if (ok_logged < 40) {
          ++ok_logged;
          std::string sw;
          for (const OwnedSwapSeen& sn : owned_swap_seen_) {
            if (sn.page) sw += fmt::format(" [page {:X} {}x{} fmt {} endian {} scaled {} frame {}]", sn.page, sn.width, sn.height, sn.format, sn.endian, sn.scaled, sn.frame);
          }
          REXGPU_INFO("[native][owned] frame {}: the swap would not read our image for the front copy to page 0x{:05X}: wants {}x{} fmt {} endian {} scaled {}; seen {}:{}",
                      frame_, page, w, h, uint32_t(d.copy_dest_format), uint32_t(d.copy_dest_endian), scaled_now, seen, sw);
        }
      }
      if (owned_wb_planned_ && !p.half) {
        // Session 90 (Q): the swap would not read our image for this page (it last sighted it too long ago), but the copy's bytes are written into
        // the page's memory from our image: whatever the swap reads from memory is right. Served and written like any copy; no loss, no pause.
        owned_wb_.front = true;
        front = false;
      } else {
        if (front_write()) return true;
        return OWNED_LOSE(true, seen);
      }
    }
  }
  // (Session 82, P2, G7: a rectangle of a defined-start pass into a face of a cube the texture cache knows - the reflection
  // cube's six faces from one strip: a cube of ours; not a cube face: the base cases below.)
  if (d.copy_dest_format == xenos::ColorFormat::k_8_8_8_8 && p.kind == 1 && p.edram_free && !front) {
    const int r = OwnedFeedbackCopy(info, sx, sy, w, h, dx, dy, d.copy_dest_swap != 0, true);
    {
      // (Session 82, P2, diagnostic: how each face copy of a strip went - served 1, not ours 0, replayed -1.)
      static uint32_t cube_logged = 0;
      if (cube_logged < 24 && w == 64 && h == 64) {
        ++cube_logged;
        REXGPU_INFO("[native][owned] frame {}: strip face copy page 0x{:05X} {}x{} from {},{}: {} (pass copies so far {}, committed {})",
                    frame_, page, w, h, sx, sy, r > 0 ? "served" : (r == 0 ? "not a cube face we know" : "refused (replayed / lost)"),
                    p.copy_count, p.committed);
      }
    }
    if (r) {
      if (r > 0) {
        note(1);
        return true;
      }
      return OWNED_LOSE(false, true, false);
    }
  }
  // (Session 82, P2, G4: a rectangle into a k_4_4_4_4 texture - the damage maps' levels: a texture of ours with mips.)
  if (d.copy_dest_format == xenos::ColorFormat::k_4_4_4_4 && p.kind == 1) {
    if (OwnedFeedbackCopy(info, sx, sy, w, h, dx, dy, d.copy_dest_swap != 0)) {
      note(1);
      return true;
    }
    // Session 92 (damage-map resets): a texture the cache has not seen loaded yet has no key, so no feedback texture can take the copy - it is
    // written from our image instead (the pass's cleared box; the cache's writer puts it where the stock copy would, and learns the key at the
    // first load). Not for a packed-tail strip of a known feedback texture (feedback_pass): that stays dropped below.
    if (OwnedChains() && owned_frame_on_ && !p.feedback_pass && !dest_ours && !front && !p.half) {
      bool planned = owned_wb_planned_;
      if (!planned) {
        std::string k_why;
        if (OwnedWbPlan(info, sx, sy, w, h, gap, false, k_why)) {
          owned_wb_planned_ = true;
          ++owned_frame_stats_.wb_planned;
          planned = true;
        } else {
          OwnedWbRefused("k_4_4_4_4 texture not loaded yet: " + k_why);
        }
      }
      if (planned) {
        OwnedCommit();
        ++p.copy_count;
        unmap();
        ++owned_frame_stats_.k4444_unloaded_wb;
        p.chain_parked = false;
        note(3);
        return true;
      }
    }
    return OWNED_LOSE(false, true, false);  // (no pause: the packed tail's strips, a texture not seen loaded yet ...)
  }
  // The page of ours a rectangle may go into: a 16-bit page image the rectangle is filled into, or (8-bit) a page's
  // image of ours of the same class and byte order.
  OwnedPage* held = nullptr;
  for (OwnedPage& q : owned_pages_) {
    if (q.page == texture_page) held = &q;
  }
  const auto fills_held = [&]() {
    return p.kind == 2 && held && held->page_image >= 0 && size_t(held->page_image) < owned_page_images_.size() &&
           held->format == uint32_t(d.copy_dest_format) && held->endian == uint32_t(d.copy_dest_endian) &&
           dx + w <= held->width && dy + h <= held->height;
  };
  const auto takes = [&]() {
    return held && held->image >= 0 && size_t(held->image) < owned_images_.size() && !gap &&
           owned_images_[held->image].kind == p.kind && held->format == uint32_t(d.copy_dest_format) &&
           held->endian == uint32_t(d.copy_dest_endian) && held->swap == (d.copy_dest_swap != 0) &&
           dx + w <= held->width && dy + h <= held->height;
  };
  // (Session 80, 1440p: a pass only our image holds - from a covering draw over tiles its colour image does not own,
  // the ring downscale - has no dirty tiles: a copy of it that runs as the game's would read the truth images without
  // its draws. Before its commit (its draws still replayable) such a copy goes ahead only where it is served from our
  // image - a rectangle into a page of ours, a whole copy with the image its page needs at hand (an older page here
  // let go first: review) - else it is replayed into the truth images; after it, a copy run as the game's pauses its
  // producers.)
  const bool truth_holds = OwnedTruthHolds();
  // (Session 82, P2, review: in a readback window a big covered pass the truth images cannot hold (an attire build, 2048 wide) is not
  // served: its first copy served and a later one read back would run the game's copy from tiles that never had its draws, and the
  // CPU would keep them. Replayed while it is uncommitted - the game's copies then read real pixels.)
  // (chains: not at all - it tests whether the readback MODE is configured, not whether THIS copy is read back; a copy the CPU really reads is the reads_back
  // branch above, and the front copy of the scene-change composite was lost here, which banned its key: run 168, frames 4285-4290.)
  if (!OwnedChains() && !truth_holds && !p.committed && !p.edram_free && owned_frame_on_ && REXCVAR_GET(gpu_native_owned_frame_readback_copy) &&
      command_processor_.NativeReadbackActive() && p.width >= 512) {
    return OWNED_LOSE(front, true, false);
  }
  if (!truth_holds && !p.committed && !OwnedOnly()) {  // (failure read fix 3: the gate only guards a replay - none exists under the latch; the 16-bit rectangle page path below is reached)
    bool served = whole || fills_held() || takes();
    if (whole && (p.kind == 2 || gap || OwnedContinues(p, 1))) {
      unmap();
      served = p.kind == 2 ? OwnedPageImageFor(w * scale_x, h * scale_y, uint32_t(d.copy_dest_format)) >= 0
                           : OwnedImageFor(owned_images_[p.image].width, owned_images_[p.image].height, p.kind) >= 0;
    }
    if (!served && owned_wb_planned_) {
      // Session 90 (Q): a read-back copy no page of ours can take is still written from our image (never replayed / paused).
      // (Review: not committed - the pass keeps its draws, so a later non-read-back copy of it can still be served or replayed; the render pass ends in the writer.)
      unmap();
      if (REXCVAR_GET(gpu_native_owned_wb_commit_uncovered)) { OwnedCommit(); ++p.copy_count; }  // (Session 90 diagnosis: the Q behaviour before review #4)
      ++owned_frame_stats_.wb_rect;
      p.chain_parked = false;  // (its pixels are delivered: nothing is left to replay)
      note(3);
      return true;
    }
    if (!served) return OWNED_LOSE(false, true, !p.edram_free);  // (Session 82, P2: a defined start owes the EDRAM nothing - the cube strips' first copy into a cube the texture cache has not seen is replayed this once, not paused for 600 frames)
  }
  const auto stale = [&]() {
    if (!truth_holds) {
      ++truth_counts_.owned_refused;
      OwnedPause(p.key, p.producers, "a copy of a pass only our image holds run as the game's");
    }
    OwnedEndCommitted("a copy run as the game's from our pixels");
    if (!OwnedOnly() || owned_wb_planned_) note(3);  // (review C6: under the latch a lost copy is not "written back" unless it is)
    return from_truth();
  };
  // A plain copy of the pass: committed (its clears' pass off, its tiles dirty - the first time).
  OwnedCommit();
  ++p.copy_count;
  if (front && OwnedChains()) ++owned_frame_stats_.ch_front_served;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  if (!whole && p.kind == 2 && owned_frame_on_ && REXCVAR_GET(gpu_native_owned_frame_rect_pages) &&
      sx == dx && sy == dy && !d.copy_dest_swap &&
      (d.copy_dest_format == xenos::ColorFormat::k_16_16 || d.copy_dest_format == xenos::ColorFormat::k_16_16_16_16)) {
    // (Session 82: a rectangle of a 16-bit pass, at its own place, into a texture none of our pages holds - the wrestler maps'
    // blur inputs, copied in four rectangles: the texture becomes a 16-bit page image of ours, filled whole from the pass
    // now (what the game's texture holds outside its rectangles is last frame's - ours too), the rectangles after it fill.)
    bool have = false;
    for (const OwnedPage& q : owned_pages_) have = have || q.page == texture_page;
    if (!have) {
      uint32_t tp = 0, w0 = 0, h0 = 0, levels = 0, level = 0, extent = 0;
      bool scaled = false;
      const xenos::TextureFormat tex_format = d.copy_dest_format == xenos::ColorFormat::k_16_16
                                                  ? xenos::TextureFormat::k_16_16
                                                  : xenos::TextureFormat::k_16_16_16_16;
      if (command_processor_.NativeResolveTarget(texture_page, uint32_t(info.copy_dest_coordinate_info.pitch_aligned_div_32),
                                                 uint32_t(d.copy_dest_endian), tex_format, tp, w0, h0, levels, level, scaled,
                                                 extent) &&
          tp == texture_page && level == 0 && levels == 1 && scaled == (scale_x > 1 || scale_y > 1) && w0 <= p.width &&
          h0 <= p.height && dx + w <= w0 && dy + h <= h0) {
        const int16_t page_image = OwnedPageImageFor(w0 * scale_x, h0 * scale_y, uint32_t(d.copy_dest_format));
        if (page_image >= 0 && OwnedFill(page_image, true, 0, 0, 0, 0, w0 * scale_x, h0 * scale_y, 0)) {
          OwnedPage pg;
          pg.page = texture_page;
          pg.page_image = page_image;
          pg.width = w0;
          pg.height = h0;
          pg.format = uint32_t(d.copy_dest_format);
          pg.endian = uint32_t(d.copy_dest_endian);
          pg.extent_start = texture_page << 12;
          pg.extent_length = extent;
          pg.frame = frame_;
          pg.producers = p.producers;
          pg.key = p.key;
          pg.frame_owned = owned_frame_on_;
          pg.census = p.census;
          pg.fill_stamp = (uint64_t(p.image + 1) << 40) | owned_draw_serial_;
          OwnedPageLog("made", pg, 0);
          OwnedKeyMade(pg.key);
          owned_pages_.push_back(pg);
          command_processor_.NativeOwnedPagesChanged();
          OwnedWatchRegister();
          OwnedWatchSync();
          OwnedMarkLeftOut(pg);
          note(1);
          ++truth_counts_.owned_fills;
          if (segment_log_frames_ > 0) {
            REXGPU_INFO("[native][seg] frame {} owned rectangle {},{} {}x{}: texture 0x{:05X} ({}x{}, format {}) is now our "
                        "16-bit page image {}", frame_, sx, sy, w, h, texture_page, w0, h0, pg.format, page_image);
          }
          return true;
        }
      }
    }
  }
  if (whole && p.kind == 2) {
    // A 16-bit page: an image of the texture's format, filled from ours as the copy writes it (its own image: later
    // draws of the pass never sample what they draw into). (Review: an older page here let go first - its page image
    // free for this one.)
    unmap();
    const int16_t page_image = OwnedPageImageFor(w * scale_x, h * scale_y, uint32_t(d.copy_dest_format));
    if (page_image < 0 || !OwnedFill(page_image, true, 0, 0, 0, 0, w * scale_x, h * scale_y, gap)) return stale();
    OwnedPage pg;
    pg.page = page;
    pg.page_image = page_image;
    pg.width = w;
    pg.height = h;
    pg.format = uint32_t(d.copy_dest_format);
    pg.endian = uint32_t(d.copy_dest_endian);
    pg.extent_start = info.copy_dest_extent_start;
    pg.extent_length = info.copy_dest_extent_length;
    pg.frame = frame_;
    pg.producers = p.producers;
    pg.key = p.key;
    pg.frame_owned = owned_frame_on_;
    pg.census = p.census;
    OwnedPageLog("made", pg, 0);
    OwnedKeyMade(pg.key);
    owned_pages_.push_back(pg);
    command_processor_.NativeOwnedPagesChanged();
    OwnedWatchRegister();
    OwnedWatchSync();
    OwnedMarkLeftOut(pg);
    note(1);
    ++truth_counts_.owned_fills;
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} owned copy left out: page 0x{:05X} ({}x{}, format {}) is our 16-bit page "
                  "image {}; dirty entries {}",
                  frame_, page, w, h, pg.format, page_image, owned_dirty_.size());
    }
    return true;
  }
  if (p.dres && !whole) {  // (Session 94: a rectangle of a scaled scene pass - not served from a smaller image)
    ++g_dres.refused;
    static uint32_t dres_rect_logged = 0;
    if (dres_rect_logged < 8) {
      ++dres_rect_logged;
      REXGPU_INFO("[native][owned-dres] frame {}: a rectangle copy ({}x{} at {},{}) of the scaled scene pass is not served", frame_, w, h, sx, sy);
    }
    return OWNED_LOSE(false, true, false);
  }
  if (p.half && !p.dres && (!whole || gap)) return OWNED_LOSE(false, true, false);  // (a half pass that goes on after its copy: a snapshot, as any)
  if (whole) {
    // The copy of the whole box into a texture's origin: left out - the page is our image (its samplers given our
    // view), or a snapshot of it when this surface's pass went on drawing after its copy before (so those draws do not
    // sample the image they draw into), or with the half-pixel offset gap to fill (Session 80, 1440p).
    // (Review: an older page here let go first - its snapshot free for this one.)
    unmap();
    int16_t image = p.image;
    if (p.half && (front || p.dres)) {
      // (look lever: the presenter reads real texels - the half-size image brought up to the full size, linear. Session 94: every whole copy of a scaled scene pass; the page is that
      // full-size image, a snapshot already, so a pass that goes on drawing needs no other, and the 1440p half-pixel gap is not filled.)
      image = OwnedBlitUp(p);
      if (p.dres && image >= 0) ++g_dres.blits;
      if (image < 0) return stale();
    } else if (gap || (OwnedContinues(p, p.copy_count) && !OwnedLeanAlias(p))) {
      if (!gap) {  // (Session 95: a predicted snapshot - counted, and whether the pass goes on after it)
        p.graph_snap = true;
        OwnedGraphCount(kGraphCountPredictedSnapshot);
      }
      image = OwnedSnapshot(gap);
      if (image < 0) return stale();
    } else if (!owned_images_[p.image].readable) {
      const VkImageMemoryBarrier readable = OwnedBarrier(
          owned_images_[p.image].image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, kOwnedShaderStages, 0, 0, nullptr, 0,
                              nullptr, 1, &readable);
      owned_images_[p.image].readable = true;
    }
    OwnedPage pg;
    pg.page = page;
    pg.image = image;
    pg.width = w;
    pg.height = h;
    pg.format = uint32_t(d.copy_dest_format);
    pg.endian = uint32_t(d.copy_dest_endian);
    pg.swap = d.copy_dest_swap != 0;
    pg.extent_start = info.copy_dest_extent_start;
    pg.extent_length = info.copy_dest_extent_length;
    pg.frame = frame_;
    pg.producers = p.producers;
    pg.key = p.key;
    pg.frame_owned = owned_frame_on_;
    pg.census = p.census;
    OwnedPageLog("made", pg, 0);
    OwnedKeyMade(pg.key);
    owned_pages_.push_back(pg);
    command_processor_.NativeOwnedPagesChanged();
    OwnedWatchRegister();
    OwnedWatchSync();
    OwnedMarkLeftOut(pg);
    note(0);
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} owned copy left out: page 0x{:05X} ({}x{}, format {}) is our image {}{}; "
                  "dirty entries {}",
                  frame_, page, w, h, pg.format, image, image != p.image ? " (a snapshot)" : "", owned_dirty_.size());
    }
    return true;
  }
  // A rectangle: into the page's image of ours when the texture is ours (a snapshot, not the pass's own image; the
  // same class and byte order) - left out, the page stays ours; else the copy runs from the truth images (its tiles'
  // pixels ours first) into the game's texture. (The page looked up again: the commit may have changed the list.)
  held = nullptr;
  for (OwnedPage& q : owned_pages_) {
    if (q.page == texture_page) held = &q;
  }
  // (A 16-bit page image of ours: the rectangle filled into it as the copy writes it.)
  if (fills_held()) {
    // (Session 82: rectangles at their own place, one after the other from one pass image - the wrestler maps' four: the page
    // is filled whole from the image once, the others are in it already. What the page holds outside its rectangles is the
    // game's stale memory: ours the image's.)
    const uint64_t stamp = (uint64_t(p.image + 1) << 40) | owned_draw_serial_;
    const bool identity = sx == dx && sy == dy && REXCVAR_GET(gpu_native_owned_frame_rect_pages) &&
                          held->fill_stamp != 0 && held->width <= p.width && held->height <= p.height;
    if (identity && held->fill_stamp == stamp) {
      held->frame = frame_;
      note(1);
      ++truth_counts_.owned_region_copies;
      return true;
    }
    const bool whole_fill = identity;
    if (!OwnedFill(held->page_image, false, whole_fill ? 0 : sx * scale_x, whole_fill ? 0 : sy * scale_y,
                   whole_fill ? 0 : dx * scale_x, whole_fill ? 0 : dy * scale_y, (whole_fill ? held->width : w) * scale_x,
                   (whole_fill ? held->height : h) * scale_y, whole_fill ? 0u : gap)) {
      return stale();
    }
    if (whole_fill) held->fill_stamp = stamp;
    held->frame = frame_;
    note(1);
    ++truth_counts_.owned_region_copies;
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} owned rectangle {},{} {}x{} into our 16-bit page 0x{:05X} (page image {}) "
                  "at {},{}",
                  frame_, sx, sy, w, h, texture_page, held->page_image, dx, dy);
    }
    return true;
  }
  // (Review: a page that is this pass's own image takes the rectangle in a snapshot of it - the pass may still draw.)
  if (takes() && held->image == p.image) {
    const int16_t snapshot = OwnedSnapshot();
    if (snapshot >= 0) {
      held->image = snapshot;
      command_processor_.NativeOwnedPagesChanged();
    OwnedWatchSync();
    }
  }
  const bool into_ours = takes() && held->image != p.image;
  if (into_ours && owned_rs_arm_.armed && held->image == owned_rs_arm_.image) owned_rs_arm_ = OwnedRsArm();  // (Session 95: a rectangle into the restore's pages: nothing is proved)
  // (Session 82, P2, G7: a defined-start pass's rectangle into a page nothing of ours holds - the reflection cube's faces
  // before the cube is known - is the truth path's this once: replayed, not paused for 600 frames.)
  if (!into_ours && p.edram_free && !held) return OWNED_LOSE(false, true, false);
  if (!into_ours) {
    ++truth_counts_.owned_region_copies_run;
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} owned rectangle {},{} {}x{} copied by the game into page 0x{:05X} (texture "
                  "0x{:05X}) at {},{}",
                  frame_, sx, sy, w, h, page, texture_page, dx, dy);
    }
    if (owned_wb_planned_ && !truth_holds) {
      // Session 90 (Q): a rectangle of a committed pass only our image holds, read back by the CPU: written from our image - not the game's copy from truth
      // tiles that never had the pass's draws (wrong bytes), not a pause. Its destination is unmapped (the texture loads from the written memory).
      unmap();
      ++owned_frame_stats_.wb_rect;
      p.chain_parked = false;  // (delivered: note(3) does not clear it)
      note(3);
      return true;
    }
    // (Review: a page of ours this rectangle cannot go into is the game's again with its texture stale outside the
    // rectangle - its whole copy was left out: its producers not owned for a while, so the next frame's copy runs.)
    if (held) {
      ++truth_counts_.owned_refused;
      OwnedPause(held->key, held->producers, "a rectangle copy into its page it cannot take");
    }
    if (!OwnedOnly() || owned_wb_planned_) note(3);
    if (!truth_holds) return stale();  // (a committed pass only our image holds: its earlier draws not in the truth)
    return from_truth();
  }
  // (Review: the page's image may still be the source of dirty tiles - a pass's own image served whole: those go into
  // the truth images first, before this rectangle changes it.)
  for (OwnedDirty& e : owned_dirty_) {
    if (e.image == held->image) {
      owned_mat_why_ = "a rectangle into the page of the pass's own image";
      OwnedMaterialize(e, e.tiles);
    }
  }
  OwnedPruneDirty();
  OwnedTouch(p.image);
  OwnedTouch(held->image);  // (the rectangle's destination is written in this submission)
  const OwnedImage& m = owned_images_[p.image];
  const OwnedImage& s = owned_images_[held->image];
  VkImageMemoryBarrier barriers[2] = {
      OwnedBarrier(m.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   owned_images_[p.image].readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                    : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
      OwnedBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)};
  const uint32_t timer = OwnedTimerBegin("rect_copy", w * scale_x, h * scale_y);  // (Session 82, P1)
  cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
  VkImageCopy region = {};
  region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.srcOffset = {int32_t(sx * scale_x), int32_t(sy * scale_y), 0};
  region.dstOffset = {int32_t(dx * scale_x), int32_t(dy * scale_y), 0};
  region.extent = {w * scale_x, h * scale_y, 1};
  OwnedCopyImage(cb, m.image, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.image, true, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &region);  // (Session 94: per layer)
  barriers[0] = OwnedBarrier(m.image, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                             owned_images_[p.image].readable ? VK_ACCESS_SHADER_READ_BIT
                                              : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             owned_images_[p.image].readable ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                              : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  barriers[1] = OwnedBarrier(s.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, barriers);
  OwnedTimerEnd(timer);
  held->frame = frame_;
  note(1);
  ++truth_counts_.owned_region_copies;
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned rectangle {},{} {}x{} into our page 0x{:05X} (image {}) at {},{}",
                frame_, sx, sy, w, h, texture_page, held->image, dx, dy);
  }
  return true;
}

void NativeRenderer::OwnedPruneDirty() {
  // Tiles another image took over (or a write-out gave the EDRAM) hold nothing of this image's any more; a slot that is
  // no longer that surface lost its pixels.
  for (OwnedDirty& e : owned_dirty_) {
    if (e.slot < 0 || size_t(e.slot) >= truth_images_.size()) {
      e.tiles.reset();
      continue;
    }
    const TruthImage& t = truth_images_[e.slot];
    if (t.base != e.base || t.pitch_tiles != e.pitch_tiles || t.kind != e.kind || t.format != e.format ||
        !t.initialized || t.image == VK_NULL_HANDLE) {
      e.tiles.reset();
    } else {
      e.tiles &= t.owned;
    }
  }
  owned_dirty_.erase(
      std::remove_if(owned_dirty_.begin(), owned_dirty_.end(), [](const OwnedDirty& e) { return e.tiles.none(); }),
      owned_dirty_.end());
}

void NativeRenderer::OwnedMaterialize(OwnedDirty& e, const std::bitset<2048>& which) {
  // Our pixels of these dirty tiles (inside the box) copied into the truth image: it then holds what the 360's EDRAM
  // would. Outside any render pass (the truth pass and ours end); the copies reading this image flushed first, as for
  // any write into it (a renamed one gets its other pixels then - before ours, not over them).
  // (The flush below may record stores: their own checks do not run meanwhile - owned_materializing_ - and the entry
  // is copied, not used after it.)
  if (owned_materializing_) return;
  std::bitset<2048> tiles = e.tiles & which;
  e.tiles &= ~which;
  const OwnedDirty d = e;
  OwnedTouch(d.image);
  if (tiles.none()) return;
  if (d.slot < 0 || size_t(d.slot) >= truth_images_.size() || d.image < 0 ||
      size_t(d.image) >= owned_images_.size()) {
    return;
  }
  {
    const TruthImage& t = truth_images_[d.slot];
    if (t.base != d.base || t.pitch_tiles != d.pitch_tiles || t.kind != d.kind || t.format != d.format ||
        !t.initialized || t.image == VK_NULL_HANDLE) {
      return;
    }
    tiles &= t.owned;
    if (tiles.none()) return;
    if ((tiles & owned_clears_tiles_).any()) OwnedClearsTouched();
  }
  owned_materializing_ = true;
  EndTruthPass();
  {
    DeferConcern c;
    c.all = false;
    c.images[0] = truth_images_[d.slot].image;
    c.images[1] = truth_images_[d.slot].rename_source;
    TruthFlushDeferredFor(kDeferFlushPassIntoSource, c);
  }
  owned_materializing_ = false;
  const TruthImage& t = truth_images_[d.slot];
  if (!t.initialized || t.image == VK_NULL_HANDLE) return;
  std::vector<std::array<int32_t, 4>> rects;
  TruthTileRects(t, tiles, rects);
  std::vector<VkImageCopy> regions;
  for (const auto& r : rects) {
    const int32_t x0 = std::max(r[0], 0), y0 = std::max(r[1], 0);
    const int32_t x1 = std::min(r[2], int32_t(d.host_width)), y1 = std::min(r[3], int32_t(d.host_height));
    if (x1 <= x0 || y1 <= y0) continue;
    VkImageCopy region = {};
    region.srcSubresource = {d.aspects, 0, 0, 1};
    region.dstSubresource = {d.aspects, 0, 0, 1};
    region.srcOffset = region.dstOffset = {x0, y0, 0};
    region.extent = {uint32_t(x1 - x0), uint32_t(y1 - y0), 1};
    regions.push_back(region);
  }
  if (regions.empty()) return;
  const OwnedImage& o = owned_images_[d.image];
  const bool color = (d.aspects & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
  const VkImageAspectFlags all_aspects =
      color ? VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT) : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  if (!color) OwnedNoteDepthRead(d.image, "materialize into the truth images");  // (Session 93: the depth is copied into the truth images)
  const VkImage ours = color ? o.image : o.depth;
  // (Ours: colour as it is now - readable, or an attachment while its pass goes on or was copied only by rectangles;
  // depth an attachment. The truth image an attachment - its pass ended.)
  const bool ours_readable = color && o.readable;
  // (Session 82: a depth that pages sample directly is in SHADER_READ_ONLY, and read-only it has no writes to wait for.)
  const bool ours_readable_any = ours_readable || (!color && o.depth_read);
  const VkImageLayout ours_layout = ours_readable_any ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                    : color       ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                                  : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  const VkImageLayout truth_layout =
      color ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  const VkAccessFlags attachment_access =
      color ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  const VkAccessFlags attachment_write =
      color ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  VkImageMemoryBarrier barriers[2] = {
      OwnedBarrier(ours, all_aspects, ours_readable_any ? 0 : attachment_write, VK_ACCESS_TRANSFER_READ_BIT, ours_layout,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
      OwnedBarrier(t.image, all_aspects, attachment_write | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, truth_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)};
  const uint32_t timer = OwnedTimerBegin(color ? "copy_in_col" : "copy_in_ds", d.host_width, d.host_height);  // (Session 82, P1)
  cb.CmdVkPipelineBarrier(kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
  OwnedCopyImage(cb, ours, true, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.image, false, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 uint32_t(regions.size()), regions.data());  // (Session 94: from the picture's layer)
  barriers[0] = OwnedBarrier(ours, all_aspects, 0, ours_readable_any ? VK_ACCESS_SHADER_READ_BIT : attachment_access,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ours_layout);
  barriers[1] = OwnedBarrier(t.image, all_aspects, VK_ACCESS_TRANSFER_WRITE_BIT,
                             attachment_access | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, truth_layout);
  cb.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                          kOwnedShaderStages | kOwnedAttachmentStages | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                          nullptr, 2, barriers);
  OwnedTimerEnd(timer, uint32_t(tiles.count()));  // (draws = the tiles copied)
  // Session 92, part D3 (TRUE NATIVE): the truth image now holds our stencil in these tiles.
  if (OwnedChains() && (d.aspects & VK_IMAGE_ASPECT_STENCIL_BIT)) truth_images_[d.slot].stencil_valid |= tiles;
  truth_counts_.owned_materialized += uint32_t(tiles.count());
  ++owned_frame_stats_.mat_calls;
  owned_frame_stats_.mat_tiles += uint32_t(tiles.count());
  {
    auto it = std::find_if(owned_frame_stats_.mat_why.begin(), owned_frame_stats_.mat_why.end(),
                           [&](const std::pair<std::string, uint32_t>& w) { return w.first == owned_mat_why_; });
    if (it == owned_frame_stats_.mat_why.end()) {
      owned_frame_stats_.mat_why.emplace_back(owned_mat_why_, 1u);
    } else {
      ++it->second;
    }
    owned_mat_why_ = "other";
  }
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned dirty tiles copied in: truth image {} ({}), {} tiles in {} rectangles "
                "from our image {}",
                frame_, d.slot,
                color                                     ? "colour"
                : (d.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? "depth"
                                                          : "stencil",
                tiles.count(), regions.size(), d.image);
  }
}

void NativeRenderer::OwnedDirtyBeforeDraw(const Draw& draw) {
  // A truth draw into a dirty tile's image (colour it writes, or depth / stencil it uses): the tiles whose part inside
  // the box it clears whole (every channel, unblended; depth and the dirty stencil written whole) are clean - it
  // overwrites them, as on the console, reading nothing; any other gets our pixels first.
  if (owned_materializing_) return;
  OwnedPruneDirty();
  if (owned_dirty_.empty()) return;
  // (Session 82, CPU: only the entries of the images this draw writes or tests matter - no tile sets built for the others.)
  {
    bool any = false;
    for (const OwnedDirty& e : owned_dirty_) {
      if (e.slot == draw.truth_color || e.slot == draw.truth_depth) {
        any = true;
        break;
      }
    }
    if (!any) return;
  }
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  if (rect[2] <= rect[0] || rect[3] <= rect[1]) return;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const uint32_t gx0 = uint32_t(std::max(rect[0], 0)) / scale_x, gy0 = uint32_t(std::max(rect[1], 0)) / scale_y;
  const uint32_t gx1 = (uint32_t(rect[2]) + scale_x - 1) / scale_x, gy1 = (uint32_t(rect[3]) + scale_y - 1) / scale_y;
  const uint32_t pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  // (Per aspect: colour it writes; depth it tests or writes; stencil it uses.)
  const bool color_touch = draw.truth_color >= 0 && (draw.color_write_mask & 0xF);
  const bool depth_touch = draw.truth_depth >= 0 && (draw.raster.depth_test || draw.raster.depth_write);
  const bool stencil_touch = draw.truth_depth >= 0 && draw.raster.stencil;
  const bool depth_whole = draw.fast_clear && draw.raster.depth_test && draw.raster.depth_write &&
                           draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS &&
                           draw.raster.depth_bias_constant == 0.0f && draw.raster.depth_bias_slope == 0.0f;
  const bool stencil_whole = draw.fast_clear && draw.raster.stencil && StencilWhole(draw.raster.stencil_ops & 0xFFF) &&
                             StencilWhole((draw.raster.stencil_ops >> 12) & 0xFFF) &&
                             draw.stencil_write_masks[0] == 0xFF && draw.stencil_write_masks[1] == 0xFF;
  for (OwnedDirty& e : owned_dirty_) {
    const bool color = (e.aspects & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
    const bool depth = (e.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    if (color ? (!color_touch || e.slot != draw.truth_color)
              : (!(depth ? depth_touch : stencil_touch) || e.slot != draw.truth_depth)) {
      continue;
    }
    const TruthImage& t = truth_images_[e.slot];
    const std::bitset<2048> touched =
        e.tiles & EdramTiles(t.base, pitch, t.is_64bpp(), gx0, gy0, gx1, gy1);
    if (touched.none()) continue;
    const bool whole = color ? draw.fast_clear && (draw.color_write_mask & 0xF) == 0xF &&
                                   !(draw.color_mask_all & ~0xFu) && draw.blend_control == 0x00010001u &&
                                   !PostNeedsDst(draw)
                             : depth ? depth_whole : stencil_whole;
    std::bitset<2048> clean;
    if (whole) {
      const uint32_t tw = (t.is_64bpp() ? 40u : 80u) * scale_x, th = 16u * scale_y;
      for (uint32_t tile = 0; tile < 2048; ++tile) {
        if (!touched.test(tile)) continue;
        // (By the image's rows of tiles: twice the pitch for 64bpp.)
        const uint32_t row_tiles = std::max(1u, t.row_tiles());
        const uint32_t rel = (tile - t.base) & 2047u, row = rel / row_tiles, col = rel % row_tiles;
        const int32_t x0 = int32_t(col * tw), y0 = int32_t(row * th);
        const int32_t x1 = std::min(int32_t((col + 1) * tw), int32_t(e.host_width));
        const int32_t y1 = std::min(int32_t((row + 1) * th), int32_t(e.host_height));
        if (rect[0] <= x0 && rect[1] <= y0 && rect[2] >= x1 && rect[3] >= y1) clean.set(tile);
      }
    }
    // (Review: dropped from the set only once the clear is recorded - TruthRecordDraw may still leave it out: a pipeline
    // not compiled yet, a failure - OwnedApplyClean, right after its draw command.)
    if (clean.any()) owned_clean_pending_.push_back({e.slot, e.aspects, clean});
    const std::bitset<2048> rest = touched & ~clean;
    if (rest.any()) {
      owned_mat_why_ = "a truth draw over them";
      OwnedMaterialize(e, rest);
    }
  }
}

void NativeRenderer::OwnedApplyClean() {
  // (By the image and aspect the clear wrote: any entry of them - entries may have been pruned since.)
  for (const OwnedClean& c : owned_clean_pending_) {
    for (OwnedDirty& e : owned_dirty_) {
      if (e.slot != c.slot || e.aspects != c.aspects) continue;
      truth_counts_.owned_cleaned += uint32_t((e.tiles & c.tiles).count());
      e.tiles &= ~c.tiles;
    }
  }
  owned_clean_pending_.clear();
  OwnedPruneDirty();
}

void NativeRenderer::OwnedDirtyBeforeRead(const std::bitset<2048>& tiles, int16_t slot, const char* why) {
  // Something reads these tiles' pixels from their images (an alias load, a store, a copy): our pixels first.
  if (owned_materializing_) return;
  OwnedPruneDirty();
  for (OwnedDirty& e : owned_dirty_) {
    if (slot >= 0 && e.slot != slot) continue;
    if ((e.tiles & tiles).any()) {
      owned_mat_why_ = why;
      OwnedMaterialize(e, tiles);
    }
  }
  OwnedPruneDirty();
}

void NativeRenderer::OwnedBeforeWriteOut() {
  if (OwnedOnly()) {  // (Session 92, M2a: no write-out exists - TruthFallback is ignored; nothing is materialised or replayed. Counted if it is ever reached)
    ++g_m2a.writeouts_ignored;
    return;
  }
  // Every image goes into the EDRAM: an owned pass still waiting for its copy is replayed into them, every dirty tile
  // gets our pixels. (Not from inside a copy-in's flush - a failure's write-out there leaves them.)
  // (chains: the held clears are a pass of ours - a load-op clear, no sets needed - then written out with the rest, never a truth pass.)
  owned_chain_site_ = 3;
  OwnedFlushHeld(OwnedChains() && !owned_materializing_);
  owned_chain_site_ = 0;
  OwnedDepthClearsFlush("the EDRAM is written out");  // (Session 87)
  OwnedClearsTouched();
  if (owned_materializing_) return;
  OwnedDropParked("a write-out");
  if (owned_pending_.valid) {
    // (chains: a pass whose copy never came is committed - its tiles dirty, written out just below - not replayed, not paused.)
    OwnedPass& wp = owned_pending_;
    if (OwnedChains() && owned_frame_on_ && !wp.committed && !wp.edram_free && (wp.start_kind == kOwnedStartCovered || OwnedTruthHolds())) {
      OwnedCommit();
      ++owned_frame_stats_.ch_commit_write;
    }
    OwnedReplayPending("a write-out before its copy");
  }
  OwnedPruneDirty();
  for (OwnedDirty& e : owned_dirty_) {
    owned_mat_why_ = "a write-out";
    OwnedMaterialize(e, e.tiles);
  }
  OwnedPruneDirty();
}

void NativeRenderer::OwnedOnSubmissionEnd() {
  OwnedRsSubmissionEnd();  // (Session 95: before the stream is handed on)
  OwnedLeanSettle(true);  // (Session 95: before the stream is handed on)
  // (gpu_native_owned_pass_barrier_lean: a page's samplers may be reused by later draws of any later submission without asking
  // OwnedView again - so every image a live page stands on counts as used up to the end of this submission.)
  for (const OwnedPage& pg : owned_pages_) OwnedTouch(pg.image);
  // (Its draws' descriptor sets and constants are this submission's: replayed now, not lost. A committed pass ends.)
  // (chains: their sets and constants are this submission's, so they become a pass of ours now - a load-op clear needs none; the pass is
  // committed just below like any pass at a submission's end.)
  owned_chain_site_ = 1;
  OwnedFlushHeld(OwnedChains());  // (held clears: their descriptor sets and constants are this submission's)
  owned_chain_site_ = 0;
  OwnedDepthClearsDrop();  // (Session 87: their sets and constants are this submission's too - nothing consulted them: dropped, counted)
  OwnedClearsTouched();  // (Session 82: the clears' pass offsets are this submission's)
  // (Session 87, gpu_native_owned_frame_carry_passes: a committed pass owes the EDRAM nothing and keeps no draws - its image is all
  // it is; it goes on in the next submission: its render pass ends here, a later draw resumes it. It still ends at the frame's end.)
  const bool carry = REXCVAR_GET(gpu_native_owned_frame_carry_passes) && owned_frame_on_;
  if (carry && OwnedChains()) {
    // (Session 90, P: a committed defined start parked without its copy is carried like any committed pass: its render pass ended, its draws -
    // sets and constants of this submission - are gone and never replayed; its image is the pass. A later copy of its surface is served from the
    // image, a later draw of it resumes it.)
    for (OwnedPass& q : owned_parked_) {
      if (!q.committed || !q.edram_free || q.carried_defined) continue;
      q.draws.clear();
      q.serves.clear();
      q.crowd_calls.clear();
      q.held_clears.clear();
      q.kept_sampled.clear();
      q.carried_defined = true;
      ++owned_frame_stats_.ch_carried_defined;
    }
  }
  OwnedDropParkedIf("the submission ends", [&](const OwnedPass& q) { return !(carry && q.committed && (OwnedChains() || !q.edram_free)); });
  owned_restorable_.clear();
  if (!owned_pending_.valid) return;
  if (owned_pending_.committed) {
    if (carry && !owned_pending_.edram_free) {
      OwnedEndPass();
      return;
    }
    return OwnedEndCommitted("the submission ends");
  }
  // (Session 82, P2: a pass whose pixels the truth images can be given later - the dirty tiles, as after any copy - need not
  // be replayed because the command processor ended its submission before the copy (it submits when it waits: the
  // combine and the front buffer's copy are always in different submissions). It is committed here, without a copy: our
  // image holds its pixels, the truth tiles are dirty (copied in before anything reads them), and the pass stays pending
  // for its copy in the next submission - or ends at the frame's end. A pass the truth images cannot hold is replayed as
  // before.)
  // (A covered start - every pixel of its box drawn - is held by our image whole whatever the truth images' tiles say: the
  // combine's pass, since the scene is owned and no truth pass writes those tiles any more.)
  if (owned_frame_on_ && (OwnedTruthHolds() || owned_pending_.start_kind == kOwnedStartCovered || OwnedOnly())) {  // (Session 92, M2a: every pass - nothing replays)
    OwnedCommit();
    ++owned_frame_stats_.sub_commits;
    if (segment_log_frames_ > 0) {
      REXGPU_INFO("[native][seg] frame {} owned pass committed at its submission's end, its copy to come (tiles dirty)",
                  frame_);
    }
    return;
  }
  ++truth_counts_.owned_lost_submission;
  OwnedPass& p = owned_pending_;
  OwnedEndPass();
  p.valid = false;
  OwnedFrameLost(p.census, kOwnedReasonLost, "the submission ends before its copy");
  // (Session 82: with the owned frame on a pass that ends its submission without a copy - a full-screen UI draw - is
  // paused as any loss, not started again every frame.)
  if (owned_frame_on_) {
    // (Review of the front buffer's pass, P2: a submission boundary that falls between a pass and its copy is not always
    // structural - paused only when it repeats: the third loss in a row within 120 frames.)
    auto it = std::find_if(owned_sub_losses_.begin(), owned_sub_losses_.end(),
                           [&](const OwnedSubLoss& l) { return l.key == p.key; });
    if (it == owned_sub_losses_.end()) {
      owned_sub_losses_.push_back({p.key, frame_, 1u});
    } else if (frame_ - it->frame > 120) {
      it->frame = frame_;
      it->count = 1;
    } else {
      it->frame = frame_;
      if (++it->count >= 3) {
        OwnedPause(p.key, p.producers, "the submission ends before its copy (three times in a row)");
        owned_sub_losses_.erase(it);
      }
    }
  }
  if (segment_log_frames_ > 0) {
    REXGPU_INFO("[native][seg] frame {} owned pass replayed ({} draws): the submission ends before its copy", frame_,
                p.draws.size());
  }
  OwnedReplayDraws();  // (review: with the served bindings kept, as any replay)
}

void NativeRenderer::OwnedPageLog(const char* what, const OwnedPage& pg, uint32_t, uint32_t rw, uint32_t rh,
                                  uint32_t rformat) {
  if (!REXCVAR_GET(gpu_native_owned_frame_pagelog)) return;
  const uint64_t ps = pg.producers.empty() ? 0 : pg.producers.front().first;
  REXGPU_INFO("[native][pagelog] frame {} {} page 0x{:05X} {}x{} fmt {} by {:016X} (surface {}/{} made frame {}) as {}x{} fmt {}", frame_,
              what, pg.page, pg.width, pg.height, pg.format, ps, pg.key.base, pg.key.pitch, pg.frame, rw ? rw : pg.width,
              rh ? rh : pg.height, rw ? rformat : pg.format);
}

NativeRenderer::OwnedKeyStat& NativeRenderer::OwnedKeyStatFor(const OwnedPassKey& key) {
  for (OwnedKeyStat& k : owned_key_stats_) {
    if (k.key == key) return k;
  }
  owned_key_stats_.push_back({});
  owned_key_stats_.back().key = key;
  return owned_key_stats_.back();
}

bool NativeRenderer::OwnedKeyDead(const OwnedPassKey& key) {
  if (!REXCVAR_GET(gpu_native_owned_frame_drop_dead)) return false;
  if (REXCVAR_GET(gpu_native_crowd_pose_half) && OwnedPoseKey(key.ps, key.pitch)) {
    // (Session 95: a key learned dead stays dropped as before (its pages listed, a sampler revives it). A live key is dropped on EVEN frames (under gpu_native_half_phase the strip / ring skip on even frames too) and only when the passes were drawn the frame before. One decision a frame for all the passes.)
    for (const OwnedKeyStat& k : owned_key_stats_) {
      if (k.key == key && k.dead) return true;
    }
    if (g_pose_dec_frame != frame_) {
      g_pose_dec_frame = frame_;
      // (Session 95, r7 contract 3, gpu_native_half_phase: the parity and the "drawn the frame before" test use shadow_frame_, the counter the shadow maps, strip and ring use.)
      const bool phase = shadow_skip::HalfPhase();
      const uint64_t pf = phase ? command_processor_.shadow_frame_counter() : frame_;
      g_pose_dec_drop = !(pf & 1) && g_pose_drawn_frame + 1 == pf;
      if (!g_pose_dec_drop) g_pose_drawn_frame = pf;
      shadow_skip::NotePhase(3, pf, !g_pose_dec_drop);  // (census of decisions: the caller may still refuse the drop, so pose 'skipped' is an upper bound; real drops are in [native][pose-half])
    }
    return g_pose_dec_drop;
  }
  for (const OwnedKeyStat& k : owned_key_stats_) {
    if (k.key == key) return k.dead;
  }
  return false;
}

void NativeRenderer::OwnedKeyMade(const OwnedPassKey& key) {
  OwnedKeyStat& k = OwnedKeyStatFor(key);
  // (Session 95, phone black r083: a frame held while pipelines are made returns before the swap (no OwnedSwapView), so frame_ runs on while no page is read: a front page's pass looked unread. 3302 held frames on the
  // phone's first run of new shaders, the menu's only pass dead at frame 657, black from the hold's end. The learning waits for the swaps to come back; measured as OwnedPresentOk measures it, against the newest swap seen.
  // The swap stamps frame_ after the frame's end, so a frame with a swap before it has newest == frame_; the 2 is slack, a real hold runs for many.)
  bool learning = true;
  if (REXCVAR_GET(gpu_native_owned_dead_held)) {
    uint64_t newest = 0;
    for (const OwnedSwapSeen& s : owned_swap_seen_) newest = std::max<uint64_t>(newest, s.frame);
    learning = newest + 2 >= frame_;
    if (!learning) ++g_dead_held_copies;
  }
  if (learning) ++k.made;
  ++k.made_total;
  if (learning && REXCVAR_GET(gpu_native_owned_frame_dead_relearn)) {
    // (Session 87: dead = no read in the last N copies, N = 240 doubled per revival; any read, in a window or not, restarts the count.)
    const uint32_t need = 240u << std::min(k.revivals, 4u);
    if (!k.dead && k.made >= need && !(k.read_ever && REXCVAR_GET(gpu_native_owned_dead_sticky))) {
      k.dead = true;
      REXGPU_INFO("[native][owned] frame {}: the pass of ps {:016X} on surface {}/{} is read by no sampler in its last {} copies "
                  "(revivals {}): dropped from now on (relearn)", frame_, key.ps, key.base, key.pitch, k.made, k.revivals);
    }
  }
  if (k.first_frame == 0) {
    k.first_frame = frame_;
    // (Session 87, step 3 diagnosis: every key's first copy, once.)
    REXGPU_INFO("[native][owned-dead] frame {}: key ps {:016X} surface {}/{} format {} made for the first time", frame_, key.ps,
                key.base, key.pitch, key.format);
  }
  if (k.made_total >= 50 && frame_ / 300 != k.summary_slot) {
    k.summary_slot = frame_ / 300;
    REXGPU_INFO("[native][owned-dead] frame {}: key ps {:016X} surface {}/{}: made {} (since its last read {}), reads outside windows {}, "
                "inside {}, last read frame {}, dead {}", frame_, key.ps, key.base, key.pitch, k.made_total, k.made, k.reads,
                k.reads_window, k.last_read_frame, k.dead);
  }
  if (learning && !REXCVAR_GET(gpu_native_owned_frame_dead_relearn) && !k.dead && k.reads == 0 && k.made >= 240 &&
      !(k.read_ever && REXCVAR_GET(gpu_native_owned_dead_sticky))) {
    k.dead = true;
    REXGPU_INFO("[native][owned] frame {}: the pass of ps {:016X} on surface {}/{} is read by no sampler in its first {} copies: "
                "dropped from now on", frame_, key.ps, key.base, key.pitch, k.made);
  } else if (learning && !k.dead && k.reads != 0 && k.made == 240) {
    // (Session 87, step 3 diagnosis: a key that could be dead but for an earlier read outside a readback window.)
    REXGPU_INFO("[native][owned-dead] frame {}: key ps {:016X} surface {}/{} has 240 unread copies since its last window read but "
                "{} read(s) outside a window (last read frame {}, window reads {}, made in all {}): never dead", frame_, key.ps,
                key.base, key.pitch, k.reads, k.last_read_frame, k.reads_window, k.made_total);
  }
}

void NativeRenderer::OwnedKeyRead(const OwnedPassKey& key) { OwnedKeyReadStat(OwnedKeyStatFor(key)); }

void NativeRenderer::OwnedKeyReadStat(OwnedKeyStat& k) {
  const OwnedPassKey& key = k.key;
  k.dead = false;
  const bool window = command_processor_.NativeReadbackActive();
  // (Session 95, Lane E, gpu_native_owned_dead_sticky: review - only a read outside a readback window pins the key; a window read
  // is the entrance / load read that Sessions 82 and 87 removed from the learning.)
  if (!window) k.read_ever = true;
  // (Session 82, P2: passes are owned inside a readback window now - the load and the match's opening - and their pages are
  // sampled there as the arena's materials come up. A read in a window does not count for good: the G2 small renders were never
  // learned dead in the match (32 passes a frame instead of 17). The unread copies are counted again from it.)
  k.last_read_frame = frame_;
  // (Session 87, step 3 diagnosis: the first reads of each key, with where they come from.)
  if (k.made_total != 0 && k.read_logs < 4) {
    ++k.read_logs;
    REXGPU_INFO("[native][owned-dead] frame {}: key ps {:016X} surface {}/{} read ({}), made so far {} ({} since the last window read)",
                frame_, key.ps, key.base, key.pitch, window ? "inside a readback window" : "OUTSIDE a window", k.made_total, k.made);
  }
  if (REXCVAR_GET(gpu_native_owned_frame_dead_relearn)) {
    // (Session 87: a read restarts the unread count, wherever it is; reads are not remembered for good.)
    if (window) ++k.reads_window; else ++k.reads;
    k.made = 0;
    return;
  }
  if (window) {
    ++k.reads_window;
    k.made = 0;
    return;
  }
  ++k.reads;
}

// A sampler asks for a page of a dropped pass: the pass is drawn again from its next frame (this binding reads the game's
// texture as it is - stale for this one frame).
void NativeRenderer::OwnedDeadPageRead(uint32_t base_page) {
  for (size_t i = owned_dead_pages_.size(); i-- > 0;) {
    const OwnedDeadPage d = owned_dead_pages_[i];
    if (d.page != base_page && d.texture_page != base_page) continue;
    OwnedKeyStat& k = OwnedKeyStatFor(d.key);
    ++k.reads;
    k.dead = false;
    if (!command_processor_.NativeReadbackActive()) k.read_ever = true;  // (Session 95, Lane E; review: outside a window only)
    // (Session 95, Lane E: the counter. A page of a dropped pass is read: this binding gets the game's stale texture for this
    // frame, so each one is a frame of possible wrong picture. The first are logged below, the total in the [owned-dead] summary.)
    ++owned_dead_page_reads_;
    if (REXCVAR_GET(gpu_native_owned_frame_dead_relearn)) {
      ++k.revivals;  // (the next time it takes longer to call it dead)
      k.made = 0;
    }
    REXGPU_INFO("[native][owned] frame {}: page 0x{:05X} of a dropped pass (ps {:016X} on surface {}/{}) is sampled: the pass "
                "is drawn again", frame_, base_page, d.key.ps, d.key.base, d.key.pitch);
    owned_dead_pages_.erase(std::remove_if(owned_dead_pages_.begin(), owned_dead_pages_.end(),
                                           [&](const OwnedDeadPage& e) { return e.key == d.key; }),
                            owned_dead_pages_.end());
    if (owned_dead_pages_.empty()) owned_dead_mask_.fill(0);  // (Session 87, gpu_cp_trim: stale bits only cost a scan)
    return;
  }
}

// What OwnedView does for every page asked for, in its order. Running it twice for one request changes nothing the second time (a stale restore page and a dead page are gone from their lists, the watch queue is empty).
void NativeRenderer::OwnedViewPre(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format) {
  M2dWatchRead(base_page, frame_, width, height, format);  // (Session 92, M2d, Part 4: who samples the composite copies' pages)
  if (!owned_rs_stale_.empty()) OwnedRsPageRead(base_page);  // (Session 95)
  if (owned_watch_pending_.load(std::memory_order_acquire)) OwnedDrainCpuWrites();
  // (Session 87, step 4, gpu_cp_trim: the list is scanned per sampled page; the mask says at once when no entry has it.)
  if (!owned_dead_pages_.empty() &&
      (!(REXCVAR_GET(gpu_cp_trim) || (REXCVAR_GET(gpu_d3d_wall) & 1u)) || ((owned_dead_mask_[(base_page & 0x1FFFF) >> 6] >> (base_page & 63)) & 1))) {
    OwnedDeadPageRead(base_page);
  }
}

bool NativeRenderer::OwnedPageMayHold(uint32_t page) {
  if (owned_page_filter_version_ != owned_pages_.version) {
    owned_page_filter_version_ = owned_pages_.version;
    owned_page_filter_.fill(0);
    for (const OwnedPage& q : std::as_const(owned_pages_)) owned_page_filter_[(q.page >> 6) & 63] |= uint64_t(1) << (q.page & 63);
  }
  return (owned_page_filter_[(page >> 6) & 63] >> (page & 63)) & 1;
}

bool NativeRenderer::OwnedPageFree(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format) {
  OwnedViewPre(base_page, width, height, format);
  return owned_pages_.empty() || !truth_ || !replacing_ || !OwnedPageMayHold(base_page);
}

// wwe2k14 patch (Session 95, r6tex, gpu_tex_bind_memo bits 2 and 4): the same answer as OwnedViewFull, without its scans of the page, key and view lists: a page no list member can hold has none; a read of a
// page that was served before under the same list, with the same views alive and the page still as it was then, is served that view again with all a read does. Everything else goes the full way, which
// is also the only way a refusal happens.
VkImageView NativeRenderer::OwnedView(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format,
                                      uint32_t endian, bool scaled_ok, const VkComponentMapping& mapping,
                                      bool is_signed, bool servable, bool has_mips, bool is_cube) {
  const int32_t memo_bits = REXCVAR_GET(gpu_tex_bind_memo);
  if (!(memo_bits & 6)) return OwnedViewFull(base_page, width, height, format, endian, scaled_ok, mapping, is_signed, servable, has_mips, is_cube);
  OwnedViewPre(base_page, width, height, format);
  if (owned_pages_.empty() || !truth_ || !replacing_ || !OwnedPageMayHold(base_page)) return VK_NULL_HANDLE;
  if (!(memo_bits & 4)) return OwnedViewFull(base_page, width, height, format, endian, scaled_ok, mapping, is_signed, servable, has_mips, is_cube);
  // (Review: the swap asks with owned_view_type_ 2D, the texture cache with 2D_ARRAY: a view of one is not an answer for the other. OwnedSwapView, owned_post.cpp.)
  const uint32_t flags = (scaled_ok ? 1u : 0u) | (is_signed ? 2u : 0u) | (servable ? 4u : 0u) | (has_mips ? 8u : 0u) | (is_cube ? 16u : 0u) | (owned_view_type_ == VK_IMAGE_VIEW_TYPE_2D ? 32u : 0u);
  const uint32_t memo_index = (base_page * 31u + uint32_t(mapping.r) * 7u + uint32_t(mapping.g) * 5u + uint32_t(mapping.b) * 3u + uint32_t(mapping.a) + flags * 11u) & 255u;
  OwnedViewMemo& m = owned_view_memo_[memo_index];
  owned_view_memo_last_ = memo_index;
  if (m.page == base_page && m.width == width && m.height == height && m.format == format && m.endian == endian && m.flags == flags && !std::memcmp(&m.mapping, &mapping, sizeof(mapping)) &&
      OwnedViewMemoHolds(m)) {
    if (g_tex_stat_on) ++g_tex_stat[kTsViewMemoHits];
    if (REXCVAR_GET(gpu_tex_bind_memo_verify)) {
      const VkImageView fresh = OwnedViewFull(base_page, width, height, format, endian, scaled_ok, mapping, is_signed, servable, has_mips, is_cube);
      if (fresh != m.view && ++g_tex_bind_mismatches <= 16) REXGPU_WARN("[tex] gpu_tex_bind_memo verify: page 0x{:05X}: the memo's view is not the full way's", base_page);
      return fresh;
    }
    OwnedViewMemoRead(m);
    return m.view;
  }
  const VkImageView view = OwnedViewFull(base_page, width, height, format, endian, scaled_ok, mapping, is_signed, servable, has_mips, is_cube);
  if (view == VK_NULL_HANDLE) return view;
  // A view from the plain way only (not the reader extension's, which serves what the plain checks refuse), and not a depth page (its read notes more).
  for (size_t i = owned_pages_.size(); i-- > 0;) {
    const OwnedPage& p = std::as_const(owned_pages_)[i];
    if (p.page != base_page) continue;
    const bool feedback = p.feedback >= 0 && size_t(p.feedback) < owned_feedback_.size();
    const bool plain_servable = servable && (!has_mips || feedback) && !(feedback && is_signed) && is_cube == (feedback && owned_feedback_[p.feedback].cube);
    if (!p.depth_direct && p.width == width && p.height == height && p.format == format && p.endian == endian && scaled_ok && plain_servable) {
      m.list_version = owned_pages_.version;
      m.view_generation = owned_view_generation_;
      m.page = base_page;
      m.width = width;
      m.height = height;
      m.format = format;
      m.endian = endian;
      m.flags = flags;
      m.mapping = mapping;
      m.index = uint32_t(i);
      m.key_index = uint32_t(&OwnedKeyStatFor(p.key) - owned_key_stats_.data());
      m.image = p.image;
      m.page_image = p.page_image;
      m.feedback = p.feedback;
      m.swap = p.swap;
      m.view = view;
    }
    break;
  }
  return view;
}

// The entry still answers: the list and the views are as they were, and the page as it was served.
bool NativeRenderer::OwnedViewMemoHolds(const OwnedViewMemo& m) const {
  if (m.list_version != owned_pages_.version || m.view_generation != owned_view_generation_ || m.index >= owned_pages_.size()) return false;
  const OwnedPage& p = owned_pages_[m.index];
  return p.page == m.page && p.image == m.image && p.page_image == m.page_image && p.feedback == m.feedback && p.swap == m.swap && p.width == m.width && p.height == m.height && p.format == m.format &&
         p.endian == m.endian && !p.depth_direct && m.key_index < owned_key_stats_.size() && owned_key_stats_[m.key_index].key == p.key;
}

// All a read of a served page does. A key read before in this frame with nothing made since gains nothing from a second read but its counters of the [owned-dead] line. (Review: a key not pinned yet still
// reads the full way: its first read of the frame may have been inside a readback window, and this one outside it pins it.)
void NativeRenderer::OwnedViewMemoRead(const OwnedViewMemo& m) {
  const OwnedPage& p = std::as_const(owned_pages_)[m.index];
  OwnedKeyStat& k = owned_key_stats_[m.key_index];
  if (k.last_read_frame != frame_ || k.made != 0 || k.dead || !k.read_ever) OwnedKeyReadStat(k);
  if (p.image >= 0 && p.page_image < 0 && p.feedback < 0) {
    OwnedGraphTripwire(p.image);
    OwnedGraphNoteRead(p.image, true);  // (Session 95, round 7: gpu_tex_bind_memo bit 4's kept binding reads here)
  }
  OwnedTouch(p.image);
  OwnedLeanNoteRead(p.image);
  if (p.image >= 0 && std::find(owned_sampled_images_.begin(), owned_sampled_images_.end(), p.image) == owned_sampled_images_.end()) {
    owned_sampled_images_.push_back(p.image);
  }
  OwnedPageLog("read", p, 0, m.width, m.height, m.format);
  ++truth_counts_.owned_served;
}

bool NativeRenderer::OwnedReadAgain(uint32_t memo_index, uint32_t base_page, VkImageView view) {
  if (memo_index >= owned_view_memo_.size()) return false;
  const OwnedViewMemo& m = owned_view_memo_[memo_index];
  if (m.page != base_page || m.view != view) return false;
  OwnedViewPre(base_page, m.width, m.height, m.format);
  if (owned_pages_.empty() || !truth_ || !replacing_ || !OwnedViewMemoHolds(m)) return false;
  OwnedViewMemoRead(m);
  return true;
}

VkImageView NativeRenderer::OwnedViewFull(uint32_t base_page, uint32_t width, uint32_t height, uint32_t format,
                                          uint32_t endian, bool scaled_ok, const VkComponentMapping& mapping,
                                          bool is_signed, bool servable, bool has_mips, bool is_cube) {
  OwnedViewPre(base_page, width, height, format);
  if (owned_pages_.empty() || !truth_ || !replacing_) return VK_NULL_HANDLE;
  if (g_tex_stat_on) g_tex_stat[kTsPagesScanned] += owned_pages_.size();  // (Session 95, r6tex)
  const OwnedPage* p = nullptr;
  if (REXCVAR_GET(gpu_d3d_wall) & 32u) {  // (Session 95: the last match, found from the end)
    for (size_t i = owned_pages_.size(); i-- > 0;) {
      if (owned_pages_[i].page == base_page) {
        p = &owned_pages_[i];
        break;
      }
    }
  } else {
    for (const OwnedPage& q : owned_pages_) {
      if (q.page == base_page) p = &q;
    }
  }
  // (Any page still held - its copies were left out: the texture is stale; one the game copies into again was
  // dropped by that copy.)
  if (p) {
    OwnedKeyRead(p->key);
    if (p->image >= 0 && p->page_image < 0 && p->feedback < 0) {
      OwnedGraphTripwire(p->image);  // (Session 95)
      OwnedGraphNoteRead(p->image, true);  // (Session 95, round 7)
    }
    OwnedTouch(p->image);
    OwnedLeanNoteRead(p->image);  // (Session 95)
    // (Session 87, review A3: see OwnedDetachPages. Distinct image indices only, so the list stays small without ever dropping the current
    // draw's entries; stale ones from a view request that never reached OwnedTakeDraw only make a detach refuse, never wrongly allow it.
    // Cleared after every draw the owned frame sees and at the frame's end.)
    if (p->image >= 0 && std::find(owned_sampled_images_.begin(), owned_sampled_images_.end(), p->image) == owned_sampled_images_.end()) {
      owned_sampled_images_.push_back(p->image);
    }
    OwnedPageLog("read", *p, 0, width, height, format);
  }
  const bool page_image = p && p->page_image >= 0 && size_t(p->page_image) < owned_page_images_.size();
  const bool feedback = p && p->feedback >= 0 && size_t(p->feedback) < owned_feedback_.size();
  if (!p || (!feedback && !page_image && (p->image < 0 || size_t(p->image) >= owned_images_.size()))) {
    return VK_NULL_HANDLE;
  }
  // (Session 82, P2, G4: a binding with mip levels is the feedback texture's to serve; any other page refuses it.)
  if (has_mips && !feedback) servable = false;
  if (feedback && is_signed) servable = false;
  // (G7: a cube binding is served by a cube of ours only, and a cube of ours serves a cube binding only.)
  if (is_cube != (feedback && owned_feedback_[p->feedback].cube)) servable = false;
  // A sampler that reads the page otherwise than the copy wrote it (size, format, byte order), with a resolution scale
  // not ours (review: the texture's scaled-resolve flag decides the fetch's texel maths), or as a binding the texture
  // cache cannot give our image (3D, an array, mips): not given our image - it reads the texture, which the left-out
  // copy did not write - and the shaders that made it are not owned for a while (from their next draws the game's copy
  // runs again; logged).
  if (p->width != width || p->height != height || p->format != format || p->endian != endian || !scaled_ok ||
      !servable) {
    // (The first few logged with what differs.)
    static uint32_t view_refusals_logged = 0;
    if (view_refusals_logged < 12) {
      ++view_refusals_logged;
      REXGPU_INFO("[native][owned] frame {}: page 0x{:05X} ({}x{} format {} endian {}) sampled as {}x{} format {} endian "
                  "{} scaled ok {} servable {}",
                  frame_, base_page, p->width, p->height, p->format, p->endian, width, height, format, endian,
                  scaled_ok, servable);
    }
    // (Session 82: the extension point - a reader the base cases refuse that a later piece serves.)
    {
      OwnedReaderContext ctx;
      ctx.base_page = base_page;
      ctx.width = width;
      ctx.height = height;
      ctx.format = format;
      ctx.endian = endian;
      ctx.scaled_ok = scaled_ok;
      ctx.is_signed = is_signed;
      ctx.servable = servable;
      ctx.mapping = mapping;
      if (const VkImageView extra = OwnedReaderExtra(ctx); extra != VK_NULL_HANDLE) return extra;
    }
    ++truth_counts_.owned_refused;
    OwnedFrameReaderRefused(*p, servable ? "a sampler of its page reads it otherwise"
                                         : "a binding of its page the substitution cannot take");
    OwnedPause(p->key, p->producers, servable ? "a sampler of its page reads it otherwise"
                                              : "a binding of its page the substitution cannot take");
    return VK_NULL_HANDLE;
  }
  // (Session 82, P2, G4: a texture of ours with mip levels - a view over all of them.)
  if (feedback) {
    if (is_cube && frame_ > 0) {
      static uint32_t cube_view_logged = 0;
      uint32_t sx = 1, sy = 1;
      command_processor_.NativeResolutionScale(sx, sy);
      if (sx > 1 && cube_view_logged < 16) {
        ++cube_view_logged;
        REXGPU_INFO("[native][owned] frame {}: cube page 0x{:05X} sampled {}x{} format {} endian {} scaled_ok {} servable {}: given our cube ({}x{}, {} layers)",
                    frame_, base_page, width, height, format, endian, scaled_ok, servable, owned_feedback_[p->feedback].width,
                    owned_feedback_[p->feedback].height, owned_feedback_[p->feedback].layers);
      }
    }
    ++truth_counts_.owned_served;
    return OwnedFeedbackView(*p, mapping);
  }
  // (Session 82, gpu_native_owned_depth_direct: a depth page that is the pass image's depth - a depth-only view of it with the
  // swizzle as given, which holds the host format's RRRR as the converted page's did.)
  if (p->depth_direct) {
    OwnedNoteDepthRead(p->image, "direct depth page sampled");  // (Session 93: a sampler is given the depth)
    const OwnedImage& od = owned_images_[p->image];
    if (od.depth == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    for (const OwnedViewEntry& v : owned_views_) {
      if (v.image == od.depth && v.format == post_ds_format_ && v.type == owned_view_type_ &&
          !std::memcmp(&v.components, &mapping, sizeof(mapping))) {
        ++truth_counts_.owned_served;
        return v.view;
      }
    }
    const ui::vulkan::VulkanDevice* vd = command_processor_.GetVulkanDevice();
    VkImageViewCreateInfo depth_view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depth_view_info.image = od.depth;
    depth_view_info.viewType = owned_view_type_;
    depth_view_info.format = post_ds_format_;
    depth_view_info.components = mapping;
    // (Session 94: an array view covers every layer - the translated shaders read the layer of the view they draw; a 2D view (the swap's) the picture's)
    depth_view_info.subresourceRange = owned_view_type_ == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_DEPTH_BIT)
                                                                                       : OwnedLastLayer(VK_IMAGE_ASPECT_DEPTH_BIT);
    VkImageView depth_view = VK_NULL_HANDLE;
    if (vd->functions().vkCreateImageView(vd->device(), &depth_view_info, nullptr, &depth_view) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    owned_views_.push_back({od.depth, mapping, depth_view, post_ds_format_, owned_view_type_});
    ++truth_counts_.owned_served;
    return depth_view;
  }
  // The texture's swizzle over our image: a k_8 texture's single channel is our red, its missing ones read as an R8
  // view's do (green, blue 0; alpha 1); a k_8_8_8_8 one's are ours as they are - with the copy's red / blue swap undone
  // (the texture's texel r is our b; as TruthServeView). A 16-bit one (the unsigned binding: R16G16 / R16G16B16A16
  // UNORM over the copied bits) reads our RGBA16 image's bits through an UNORM view - k_16_16: red and green, blue 0,
  // alpha 1.
  const bool single = format == uint32_t(xenos::ColorFormat::k_8);
  const bool pair = format == uint32_t(xenos::ColorFormat::k_16_16);
  const bool rgba16 = pair || format == uint32_t(xenos::ColorFormat::k_16_16_16_16);
  const ui::vulkan::VulkanDevice* const device_info = command_processor_.GetVulkanDevice();
  // (A 16-bit page image: the texture's own format and channels - its views as the texture cache's, over the same
  // bits; the swizzle as given.)
  if (page_image) {
    const OwnedPageImage& q = owned_page_images_[p->page_image];
    // (Session 82, P1: a depth page - the texture cache's R32_SFLOAT for both signs, the swizzle as given (it holds the
    // host format's RRRR).)
    const VkFormat view_format = q.kind == 1 ? VK_FORMAT_R32_SFLOAT
                                 : pair      ? (is_signed ? VK_FORMAT_R16G16_SNORM : VK_FORMAT_R16G16_UNORM)
                                             : (is_signed ? VK_FORMAT_R16G16B16A16_SNORM : VK_FORMAT_R16G16B16A16_UNORM);
    for (const OwnedViewEntry& v : owned_views_) {
      if (v.image == q.image && v.format == view_format && v.type == owned_view_type_ &&
          !std::memcmp(&v.components, &mapping, sizeof(mapping))) {
        ++truth_counts_.owned_served;
        return v.view;
      }
    }
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = q.image;
    view_info.viewType = owned_view_type_;
    view_info.format = view_format;
    view_info.components = mapping;
    view_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageView view = VK_NULL_HANDLE;
    if (device_info->functions().vkCreateImageView(device_info->device(), &view_info, nullptr, &view) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    owned_views_.push_back({q.image, mapping, view, view_format, owned_view_type_});
    ++truth_counts_.owned_served;
    return view;
  }
  VkComponentMapping components = mapping;
  const VkComponentSwizzle own[4] = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                     VK_COMPONENT_SWIZZLE_A};
  VkComponentSwizzle* const parts[4] = {&components.r, &components.g, &components.b, &components.a};
  for (uint32_t i = 0; i < 4; ++i) {
    VkComponentSwizzle& s = *parts[i];
    if (s == VK_COMPONENT_SWIZZLE_IDENTITY) s = own[i];
    if (single) {
      if (s == VK_COMPONENT_SWIZZLE_G || s == VK_COMPONENT_SWIZZLE_B) s = VK_COMPONENT_SWIZZLE_ZERO;
      if (s == VK_COMPONENT_SWIZZLE_A) s = VK_COMPONENT_SWIZZLE_ONE;
    } else if (pair) {
      if (s == VK_COMPONENT_SWIZZLE_B) s = VK_COMPONENT_SWIZZLE_ZERO;
      if (s == VK_COMPONENT_SWIZZLE_A) s = VK_COMPONENT_SWIZZLE_ONE;
    } else if (p->swap) {
      if (s == VK_COMPONENT_SWIZZLE_R) {
        s = VK_COMPONENT_SWIZZLE_B;
      } else if (s == VK_COMPONENT_SWIZZLE_B) {
        s = VK_COMPONENT_SWIZZLE_R;
      }
    }
  }
  const OwnedImage& o = owned_images_[p->image];
  // (An owned image serves the 8-bit textures only - an RGBA8 one; a 16-bit page is a page image, above: the copy
  // converts its values, a view of the pass's image would not.)
  if (o.kind != 1 || rgba16) {
    ++truth_counts_.owned_refused;
    OwnedFrameReaderRefused(*p, "a sampler of its page reads it as another class");
    OwnedPause(p->key, p->producers, "a sampler of its page reads it as another class");
    return VK_NULL_HANDLE;
  }
  // (The view's format: the fetch's interpretation of the copied bits - UNORM for the unsigned one, SNORM for the
  // signed one, over the same image: MUTABLE_FORMAT.)
  const VkFormat view_format = rgba16 ? (is_signed ? VK_FORMAT_R16G16B16A16_SNORM : VK_FORMAT_R16G16B16A16_UNORM)
                                      : (is_signed ? VK_FORMAT_R8G8B8A8_SNORM : VK_FORMAT_R8G8B8A8_UNORM);
  for (const OwnedViewEntry& v : owned_views_) {
    if (v.image == o.image && v.format == view_format && v.type == owned_view_type_ &&
        !std::memcmp(&v.components, &components, sizeof(components))) {
      ++truth_counts_.owned_served;
      return v.view;
    }
  }
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = o.image;
  view_info.viewType = owned_view_type_;  // (the texture cache's views of 2D textures are arrays; the swap's is 2D)
  view_info.format = view_format;
  view_info.components = components;
  view_info.subresourceRange = owned_view_type_ == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? ui::vulkan::util::InitializeSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT)
                                                                               : OwnedLastLayer(VK_IMAGE_ASPECT_COLOR_BIT);  // (Session 94: see above)
  VkImageView view = VK_NULL_HANDLE;
  if (vulkan_device->functions().vkCreateImageView(vulkan_device->device(), &view_info, nullptr, &view) !=
      VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  owned_views_.push_back({o.image, components, view, view_format, owned_view_type_});
  ++truth_counts_.owned_served;
  return view;
}

void NativeRenderer::OwnedOnFrameEnd() {
  // A pass whose copy never came (the frame ended first): replayed into the truth images while its submission is
  // still open, else lost (its tiles are left as gpu_shadow_half_rate leaves a skipped map's); counted, its shaders not
  // owned any more. (A committed pass ends; surfaces not seen going on after a copy for 120 frames are forgotten.)
  owned_chain_site_ = 2;
  OwnedPendSetFrameEnd(true);  // (review fix 9: the held / depth clears this flushes were used by nothing this frame: dropped, not pended)
  OwnedFlushHeld(OwnedChains());
  owned_chain_site_ = 0;
  OwnedDepthClearsDrop();  // (Session 87)
  OwnedPendSetFrameEnd(false);
  if (OwnedOnly() && OwnedPendBitOn()) OwnedPendClearFrameEnd();  // (Session 92, M4f-B: pending clears nothing consumed this frame are dropped, counted)
  OwnedClearsTouched();
  owned_frame_end_carried_.clear();  // (Session 90: the list is the last frame end's - refilled by the drop just below)
  OwnedDropParked("the frame ends");
  owned_sampled_images_.clear();
  owned_restorable_.clear();
  if (OwnedChains() && owned_frame_on_ && owned_pending_.valid && !owned_pending_.committed &&
      !owned_pending_.edram_free && (owned_pending_.start_kind == kOwnedStartCovered || OwnedTruthHolds())) {
    // (chains: a pass whose copy never came by the frame's end is committed - its pixels stay in our image, its tiles dirty - not replayed, not paused.)
    OwnedCommit();
    ++owned_frame_stats_.ch_commit_frame;
  }
  if (owned_pending_.valid && OwnedOnly()) OwnedEndAsIs("the frame ends");  // (Session 92, M2a: committed if it was not, ended - its image the surface's newest record)
  if (owned_pending_.valid && owned_pending_.committed) OwnedEndCommitted("the frame ends");
  owned_continued_.erase(std::remove_if(owned_continued_.begin(), owned_continued_.end(),
                                        [&](const OwnedContinued& c) { return c.frame + 120 < frame_; }),
                         owned_continued_.end());
  OwnedLeanFrame();  // (Session 95)
  OwnedRsFrame();  // (Session 95)
  if (owned_pending_.valid) {
    ++truth_counts_.owned_lost_copy;
    if (owned_pending_.submission == command_processor_.GetCurrentSubmission()) {
      OwnedReplayPending("no copy of it before the frame's end");
    } else {
      OwnedEndPass();
      owned_pending_.valid = false;
      owned_pending_.draws.clear();
      owned_pending_.crowd_calls.clear();
      owned_pending_.serves.clear();
      OwnedFrameLost(owned_pending_.census, kOwnedReasonLost, "no copy of it before the frame's end");
      OwnedPause(owned_pending_.key, owned_pending_.producers, "no copy of it before the frame's end");
    }
  }
  // Pages not copied into for 30 frames are their textures' again (their images free; a texture sampled after that
  // is stale - its copies were left out: kept this long for half-rate maps and skipped frames). Session 82: not the
  // pages of the owned frame - they stay until the game's next copy into them or a CPU write drops them (a texture the
  // frame reads from them may be many frames old: the arena screen's, the ring shadow's).
  const size_t before = owned_pages_.size();
  owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                    [&](const OwnedPage& p) { return !p.frame_owned && p.frame + 30 <= frame_; }),
                     owned_pages_.end());
  if (owned_pages_.size() != before) {
    command_processor_.NativeOwnedPagesChanged();
    OwnedWatchSync();
  }
  OwnedPruneDirty();
  DecoupleFrameEnd();  // Session 94 (gpu_d3d_decouple): the A/A compare of the presented image's layers, the window line
  OwnedFrameFinish();  // Session 82 (the census of this frame)
  OwnedRefreshList();  // (the lists for the next frame)
  OwnedRefreshFrame();  // Session 82 (the switch and the log window for the next frame)
}


// ---- Session 82: the owned match frame - reasons, pauses, the census, the switch, CPU writes ------------------------

// Session 82 (the owned frame): the clears of a pass so far - one known colour, all four channels, no blending (pieces
// of one clear: the console splits a clear at the tile boundary); anything else ends the run. Shared by the truth pass's
// own record and by the clears the owned frame holds until a consumer takes them (OwnedHoldClear).
void NativeRenderer::ClearsAccumulate(ClearsInfo& pc, const Draw& draw, const int32_t rect[4], bool first,
                                      bool writes_color, bool uses_depth, uint32_t kind) {
  // (chains) A depth(-stencil)-only clear of exactly the box the held colour clears cleared whole - and whose depth they left alone - is part of
  // them: the box's depth is cleared too, as one clear of colour and depth would. (Bryan's shadow-map passes: colour clear, depth clear, draws.)
  if (OwnedChains() && !first && pc.valid && draw.fast_clear && !writes_color && !pc.uses_depth && uses_depth && draw.clear_depth >= 0.0f &&
      draw.raster.depth_test && draw.raster.depth_write && draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS &&
      draw.raster.depth_bias_constant == 0.0f && draw.raster.depth_bias_slope == 0.0f && rect[0] == pc.rect[0] && rect[1] == pc.rect[1] &&
      rect[2] == pc.rect[2] && rect[3] == pc.rect[3] && pc.area == uint64_t(pc.rect[2] - pc.rect[0]) * uint64_t(pc.rect[3] - pc.rect[1])) {
    const auto stencil_whole_m = [](uint32_t ops) {
      return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS);
    };
    const bool st_used = draw.raster.stencil;
    const bool st_whole = st_used && stencil_whole_m(draw.raster.stencil_ops & 0xFFF) && stencil_whole_m((draw.raster.stencil_ops >> 12) & 0xFFF) &&
                          draw.stencil_write_masks[0] == 0xFF && draw.stencil_write_masks[1] == 0xFF && draw.stencil_refs[0] == draw.stencil_refs[1];
    pc.uses_depth = true;
    pc.depth = true;
    pc.depth_value = draw.clear_depth;
    pc.stencil_touched = st_used;
    pc.stencil = st_whole;
    pc.stencil_value = draw.stencil_refs[0];
    ++owned_frame_stats_.ch_dmerge;
    return;
  }
  const bool clear = draw.fast_clear && draw.clear_color_known && writes_color &&
                     (draw.color_write_mask & 0xF) == 0xF && draw.blend_control == 0x00010001u;
  // (Owned passes: what the clear does to depth and stencil - written whole to a known value, as
  // gpu_native_truth_clear_attachments takes them: depth through ALWAYS without bias; stencil by REPLACE / ALWAYS on
  // both faces, every bit, one reference.)
  const auto stencil_whole = [](uint32_t ops) {
    return ((ops >> 3) & 7) == uint32_t(VK_STENCIL_OP_REPLACE) && ((ops >> 9) & 7) == uint32_t(VK_COMPARE_OP_ALWAYS);
  };
  const bool clear_depth = uses_depth && (draw.raster.depth_test || draw.raster.depth_write);
  const bool clear_depth_whole = clear_depth && draw.clear_depth >= 0.0f && draw.raster.depth_test &&
                                 draw.raster.depth_write && draw.raster.depth_compare == VK_COMPARE_OP_ALWAYS &&
                                 draw.raster.depth_bias_constant == 0.0f && draw.raster.depth_bias_slope == 0.0f;
  const bool clear_quantized = draw.color_quantize == kind &&
                               (draw.pixel_ext & SpirvShaderTranslator::kNativeExtEpilogueIdle) &&
                               (draw.color_mask_all & ~0xFu) == 0;
  const bool clear_stencil = uses_depth && draw.raster.stencil;
  const bool clear_stencil_whole = clear_stencil && stencil_whole(draw.raster.stencil_ops & 0xFFF) &&
                                   stencil_whole((draw.raster.stencil_ops >> 12) & 0xFFF) &&
                                   draw.stencil_write_masks[0] == 0xFF && draw.stencil_write_masks[1] == 0xFF &&
                                   draw.stencil_refs[0] == draw.stencil_refs[1];
  if (!clear || (!first && !pc.valid) ||
      (!first && std::memcmp(pc.color, draw.clear_color, sizeof(pc.color)))) {
    pc.valid = false;
  } else {
    bool repeat = false;
    if (first) {
      std::copy(rect, rect + 4, pc.rect);
      std::memcpy(pc.color, draw.clear_color, sizeof(pc.color));
      pc.area = 0;
      pc.uses_depth = clear_depth;
      pc.depth = clear_depth_whole;
      pc.depth_value = draw.clear_depth;
      pc.stencil_touched = clear_stencil;
      pc.stencil = clear_stencil_whole;
      pc.stencil_value = draw.stencil_refs[0];
      pc.quantized = clear_quantized;
    } else {
      pc.quantized = pc.quantized && clear_quantized;
      // (Every piece the same: depth whole to one value, stencil whole to one value - else not whole.)
      pc.depth = pc.depth && clear_depth_whole && pc.depth_value == draw.clear_depth;
      pc.uses_depth = pc.uses_depth || clear_depth;
      pc.stencil = pc.stencil && clear_stencil_whole && pc.stencil_value == draw.stencil_refs[0];
      pc.stencil_touched = pc.stencil_touched || clear_stencil;
      // (Review: a piece overlapping the earlier ones ends the run - the summed area then proves a whole box only for
      // pieces that do not overlap, as a clear split at a tile boundary.)
      const bool overlaps = rect[0] < pc.rect[2] && pc.rect[0] < rect[2] && rect[1] < pc.rect[3] && pc.rect[1] < rect[3];
      // (Session 82, P2: the same clear again over a box already cleared whole - the entrances' scene passes clear their
      // 1280x720 colour / depth twice, identical in every value: the union stays the box, not an overlap that ends the run.)
      const bool whole_now = pc.area == uint64_t(pc.rect[2] - pc.rect[0]) * uint64_t(pc.rect[3] - pc.rect[1]);
      repeat = overlaps && whole_now && rect[0] >= pc.rect[0] && rect[1] >= pc.rect[1] && rect[2] <= pc.rect[2] &&
               rect[3] <= pc.rect[3];
      pc.rect[0] = std::min(pc.rect[0], rect[0]), pc.rect[1] = std::min(pc.rect[1], rect[1]);
      pc.rect[2] = std::max(pc.rect[2], rect[2]), pc.rect[3] = std::max(pc.rect[3], rect[3]);
      if (overlaps && !repeat) pc.area = UINT64_MAX / 2;
    }
    if (!repeat) pc.area += uint64_t(rect[2] - rect[0]) * uint64_t(rect[3] - rect[1]);
    pc.valid = pc.area < UINT64_MAX / 2;
  }
  if (segment_log_frames_ > 0 && draw.fast_clear) {
    REXGPU_INFO("[native][seg] frame {} clears: colour {} depth {} sx {} first {} clear {} known {} draw clear_depth {:.3f} uses_depth {} "
                "writes_color {} rect {},{}-{},{} -> valid {} depth_whole {} depth_value {:.3f} area {}",
                frame_, draw.truth_color, draw.truth_depth, draw.truth_sx, first, clear, draw.clear_color_known, draw.clear_depth,
                uses_depth, writes_color, rect[0], rect[1], rect[2], rect[3], pc.valid, pc.depth, pc.depth_value, pc.area);
  }
}

const char* NativeRenderer::OwnedReasonName(uint32_t reason) {
  static const char* const kNames[kOwnedReasonCount] = {"-",       "start",   "msaa",       "format",        "readback",
                                                        "lost",    "reader",  "vertex_fetch", "pool",        "paused",
                                                        "input_served", "no_pipeline", "clears_kept", "failed", "present",
                                                        "other", "dead"};
  return reason < kOwnedReasonCount ? kNames[reason] : "other";
}

NativeRenderer::OwnedPassKey NativeRenderer::OwnedKeyOf(const Draw& draw) const {
  OwnedPassKey key;
  key.base = draw.color_info & 0xFFF;
  key.pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  key.format = (draw.color_info >> 16) & 0xF;
  key.ps = draw.pixel_shader_hash;
  return key;
}

bool NativeRenderer::OwnedPaused(const OwnedPassKey& key) const {
  return std::any_of(owned_paused_.begin(), owned_paused_.end(),
                     [&](const std::pair<OwnedPassKey, uint64_t>& d) { return d.first == key && frame_ < d.second; });
}

void NativeRenderer::OwnedPause(const OwnedPassKey& key, const std::vector<std::pair<uint64_t, bool>>& producers,
                                const char* why) {
  // (Session 82: with the owned frame on, a loss pauses the pass - its surface and first pixel shader - for 600 frames,
  // not every shader that drew it, frame-wide; else the listed shaders, as before.)
  if (OwnedOnly()) {  // (Session 92, M2a: nothing to protect without a truth path - no pause; counted by why)
    ++g_m2a.pauses_avoided;
    M2aGroup(g_m2a.avoided_why, std::string("pause: ") + why);
    return;
  }
  if (!owned_frame_on_) return OwnedDisableAll(producers, why);
  if (OwnedPaused(key)) return;
  owned_paused_.erase(std::remove_if(owned_paused_.begin(), owned_paused_.end(),
                                     [&](const std::pair<OwnedPassKey, uint64_t>& d) { return d.first == key; }),
                      owned_paused_.end());
  owned_paused_.emplace_back(key, frame_ + 600);
  ++owned_frame_stats_.pauses;
  if (command_processor_.NativeReadbackActive()) {  // (Session 90, Q: a pause inside a read-back window - there should be none)
    ++owned_frame_stats_.wb_window_pauses;
    REXGPU_INFO("[native][owned] frame {}: PAUSE INSIDE A READ-BACK WINDOW: ps {:016X} ({})", frame_, key.ps, why);
  }
  REXGPU_WARN("[native][owned] frame {}: pass base {} pitch {} format {} ps {:016X} not owned for 600 frames ({})",
              frame_, key.base, key.pitch, key.format, key.ps, why);
}

void NativeRenderer::OwnedNoteRefusal(uint8_t reason, const std::string& detail) {
  // (The first refusal of a draw: the pass it lands in gets it - OwnedFrameTruthDrawn.)
  if (!owned_frame_on_ || owned_frame_pending_reason_ != kOwnedReasonNone) return;
  owned_frame_pending_reason_ = reason;
  owned_frame_pending_detail_ = detail;
}

int32_t NativeRenderer::OwnedFrameNewPass(const Draw& draw) {
  OwnedFramePass r;
  r.base = draw.color_info & 0xFFF;
  r.pitch = std::max(1u, ((draw.surface_info & 0x3FFF) * draw.truth_sx + 79) / 80);
  r.format = (draw.color_info >> 16) & 0xF;
  r.vs = draw.vertex_shader_hash;
  r.ps = draw.pixel_shader_hash;
  owned_frame_passes_.push_back(std::move(r));
  owned_frame_last_ = int32_t(owned_frame_passes_.size() - 1);
  return owned_frame_last_;
}

void NativeRenderer::OwnedFramePassOpened(const Draw& draw) {
  if (!owned_frame_on_ || owned_replaying_ || !truth_) return;
  OwnedFrameNewPass(draw);
}

void NativeRenderer::OwnedFrameTruthDrawn(const Draw& draw, const int32_t rect[4]) {
  (void)draw;
  if (!owned_frame_on_ || owned_replaying_ || owned_frame_last_ < 0 ||
      size_t(owned_frame_last_) >= owned_frame_passes_.size()) {
    return;
  }
  OwnedFramePass& r = owned_frame_passes_[owned_frame_last_];
  if (r.owned || r.lost) return;  // (a draw of the truth path after an owned pass: its own record comes with its pass)
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  ++r.draws;
  r.box_w = std::max(r.box_w, (uint32_t(std::max(rect[2], 0)) + scale_x - 1) / scale_x);
  r.box_h = std::max(r.box_h, (uint32_t(std::max(rect[3], 0)) + scale_y - 1) / scale_y);
  if (r.reason == kOwnedReasonNone && owned_frame_pending_reason_ != kOwnedReasonNone) {
    r.reason = owned_frame_pending_reason_;
    r.detail = owned_frame_pending_detail_;
  }
  owned_frame_pending_reason_ = kOwnedReasonNone;
}

void NativeRenderer::OwnedFrameLost(int32_t census, uint8_t reason, const char* why) {
  if (census < 0 || size_t(census) >= owned_frame_passes_.size()) return;
  OwnedFramePass& r = owned_frame_passes_[census];
  r.owned = false;
  r.lost = true;
  r.reason = reason;
  r.detail = why;
}

void NativeRenderer::OwnedFrameReaderRefused(const OwnedPage& page, const char* why) {
  if (!owned_frame_on_) return;
  ++owned_frame_stats_.reader_events;
  // (A page made this frame: its pass did not serve what read it - a fallback; a page of an earlier frame is counted as
  // an event only.)
  if (page.census >= 0 && size_t(page.census) < owned_frame_passes_.size()) {
    OwnedFramePass& r = owned_frame_passes_[page.census];
    r.owned = false;
    r.lost = true;
    r.reason = kOwnedReasonReader;
    r.detail = why;
  }
}

void NativeRenderer::OwnedFrameFinish() {
  // The frame's census: one line per pass while it is logged, the totals into the window's summary.
  const uint64_t frame = frame_ > 0 ? frame_ - 1 : 0;  // (frame_ is the next frame's number here)
  for (OwnedPage& pg : owned_pages_) pg.census = -1;
  if (!owned_frame_on_ || owned_frame_passes_.empty()) {
    owned_frame_passes_.clear();
    owned_frame_last_ = -1;
    return;
  }
  OwnedFrameStats& st = owned_frame_stats_;
  if (!st.frames) st.first_frame = frame;
  ++st.frames;
  uint32_t index = 0;
  for (const OwnedFramePass& r : owned_frame_passes_) {
    bool owned = r.owned && !r.lost;
    uint8_t reason = owned ? uint8_t(kOwnedReasonNone) : (r.reason ? r.reason : uint8_t(kOwnedReasonStart));
    if (!owned && !r.reason && !r.lost && OwnedOnly() && r.copies > 0 && r.dead == r.copies) {
      // Session 92, M2c (census bookkeeping): a pass never committed whose only copies were dropped as dead (the dead-pass drop - the circle-shadow pass
      // behind the dead caster pass, surface 0/7) handed nothing back and replayed nothing: counted under its own reason "dead", as owned - not as a "start" fallback.
      owned = true;
      ++g_m2a.dead_owned;  // (review F9: printed apart from the fallback reasons - these passes are owned)
    }
    ++st.passes;
    if (owned) {
      ++st.owned;
      if (r.clears_kept) ++st.clears_kept;
    } else {
      ++st.fallback;
      ++st.reasons[reason];
      {  // (Session 87: which passes these are)
        static const char* const kStartNames[] = {"-", "cleared", "covered", "resumed", "converted"};
        const std::string sig = fmt::format("{}{}{} start {} surf {}/{}/{} box {}x{} draws {} vs {:016X} ps {:016X}",
                                            OwnedReasonName(reason), r.detail.empty() ? "" : "/", r.detail,
                                            kStartNames[std::min<uint32_t>(r.start, 4)], r.base, r.pitch, r.format, r.box_w, r.box_h,
                                            r.draws, r.vs, r.ps);
        auto it = std::find_if(st.fallback_groups.begin(), st.fallback_groups.end(),
                               [&](const std::pair<std::string, uint32_t>& g) { return g.first == sig; });
        if (it != st.fallback_groups.end()) {
          ++it->second;
        } else if (st.fallback_groups.size() < 256) {
          st.fallback_groups.emplace_back(sig, 1u);
        }
      }
      if (reason == kOwnedReasonLost && !r.detail.empty()) {
        auto it = std::find_if(st.lost_why.begin(), st.lost_why.end(),
                               [&](const std::pair<std::string, uint32_t>& w) { return w.first == r.detail; });
        if (it == st.lost_why.end()) {
          st.lost_why.emplace_back(r.detail, 1u);
        } else {
          ++it->second;
        }
      }
    }
    if (owned_frame_logging_) {
      static const char* const kStarts[] = {"-", "cleared", "covered", "resumed", "converted"};
      std::string status = owned ? (r.clears_kept ? "owned clears_kept" : "owned")
                                 : std::string("fallback ") + (reason == kOwnedReasonOther
                                                                   ? "other:" + (r.detail.empty() ? "?" : r.detail)
                                                                   : OwnedReasonName(reason));
      REXGPU_INFO("[native][owned-frame] frame {} pass {}: {} start {} surface {}/{}/{} box {}x{} draws {} vs {:016X} ps "
                  "{:016X} copies {} (served {}, page {}, dead {}, written back {}){}{}",
                  frame, index, status, kStarts[std::min<uint32_t>(r.start, 4)], r.base, r.pitch, r.format, r.box_w,
                  r.box_h, r.draws, r.vs, r.ps, r.copies, r.served, r.page, r.dead, r.written_back,
                  !owned && !r.detail.empty() && reason != kOwnedReasonOther ? "; why: " : "",
                  !owned && reason != kOwnedReasonOther ? r.detail : std::string());
    }
    ++index;
  }
  owned_frame_passes_.clear();
  owned_frame_last_ = -1;
  if (st.frames >= 600) OwnedFrameSummary();
}

void NativeRenderer::OwnedFrameSummary() {
  OwnedFrameStats& st = owned_frame_stats_;
  if (!st.frames) return;
  // (Session 92 M0: the "free from the 360" counters, one line per window under TRUE NATIVE; reset either way)
  rex::graphics::free_counters::PrintAndReset(OwnedChains(), st.first_frame, st.first_frame + st.frames - 1);
  OwnedGraphSummary(st.first_frame, st.first_frame + st.frames - 1);  // (Session 95, gpu_owned_graph)
  std::string reasons;
  for (uint32_t i = 1; i < kOwnedReasonCount; ++i) {
    if (!st.reasons[i]) continue;
    reasons += fmt::format("{}{} {}", reasons.empty() ? "" : ", ", i == kOwnedReasonOther ? "other" : OwnedReasonName(i),
                           st.reasons[i]);
  }
  const double per_frame = st.frames ? 1.0 / double(st.frames) : 0.0;
  std::string mat_why;
  for (const auto& w : st.mat_why) {
    mat_why += fmt::format("{}{} {:.2f}", mat_why.empty() ? "" : "; ", w.first, w.second * per_frame);
  }
  const std::string costs = fmt::format(
      "; per frame: dirty copy-ins {:.2f} ({:.1f} tiles), dirty tiles dropped as overwritten {:.1f}, converted starts "
      "{:.2f}, snapshots {:.2f}; TRUTH WORK per frame: passes recorded {:.2f}, tiles loaded {:.1f}, staging stores "
      "{:.2f}, copies run {:.2f}, dirty copy-ins {:.2f} [{}]; owned draw state commands per frame {:.1f}, left out {:.1f}, dead passes not drawn {:.1f}, restore draws left out {:.2f}, restore draws recorded {:.2f} (held clears dropped {:.2f}, their stencil cleared in our pass {:.2f}, lost {:.2f}), pages detached {:.2f}; lean pass starts {:.2f}; cube strips copied as one {:.2f} (faces answered {:.2f}); alpha learn per frame: candidates {:.1f}, drawn unblended {:.1f}, probes {:.1f}, reverts {:.2f} (keys {}, proven {})",
      st.mat_calls * per_frame, st.mat_tiles * per_frame, st.dead_dropped * per_frame, st.converted * per_frame,
      st.snapshots * per_frame, st.truth_passes * per_frame, st.truth_tile_loads * per_frame, st.truth_stores * per_frame,
      st.truth_copies * per_frame, st.mat_calls * per_frame, mat_why, double(st.rec_state_cmds) * per_frame,
      double(st.rec_state_skipped) * per_frame, st.dropped_draws * per_frame, st.restores_left_out * per_frame,
      st.restore_drawn * per_frame, st.restore_clears_dropped * per_frame, st.restore_stencil_cleared * per_frame, st.restore_stencil_lost * per_frame, st.detached * per_frame,
      st.lean_starts * per_frame, st.strip_batches * per_frame, st.strip_faces_answered * per_frame,
      st.alpha_candidates * per_frame, st.alpha_proven * per_frame, st.alpha_probes * per_frame,
      st.alpha_reverts * per_frame, owned_alpha_keys_.size(),
      std::count_if(owned_alpha_keys_.begin(), owned_alpha_keys_.end(), [](const auto& kv) { return kv.second.proven; }));
  std::string lost;
  for (const auto& w : st.lost_why) lost += fmt::format("{}{}: {}", lost.empty() ? "" : "; ", w.first, w.second);
  if (st.dclear_pending || st.dclear_flushed || st.dclear_dropped || st.dclear_superseded || st.dclear_read_after_drop) {
    REXGPU_INFO("[native][owned-frame] frames {}-{}: depth-only clears pending: {}; run in the truth path (a draw / a copy / a write-out used that "
                "depth first): {}; dropped at a submission's end: {}; superseded: {}; draws that used a depth after its clear was dropped: {}",
                st.first_frame, st.first_frame + st.frames - 1, st.dclear_pending, st.dclear_flushed, st.dclear_dropped, st.dclear_superseded,
                st.dclear_read_after_drop);
  }
  if (st.depth_clear_taken || st.depth_clear_deferred || st.depth_clear_no_pass || st.depth_clear_not_open || st.depth_clear_other_depth || st.depth_clear_not_whole) {
    REXGPU_INFO("[native][owned-frame] frames {}-{}: depth-only clears into an owned pass: taken {} (waited for the pass to resume: {}); not taken: no pass {}, pass not open {}, "
                "other depth surface {}, not whole / constant {}",
                st.first_frame, st.first_frame + st.frames - 1, st.depth_clear_taken, st.depth_clear_deferred, st.depth_clear_no_pass, st.depth_clear_not_open,
                st.depth_clear_other_depth, st.depth_clear_not_whole);
  }
  if (!st.fallback_groups.empty()) {  // (Session 87: what the fallback passes are, most frequent first)
    auto groups = st.fallback_groups;
    std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string top;
    for (size_t i = 0; i < groups.size() && i < 12; ++i) top += fmt::format("\n    {} x {}", groups[i].second, groups[i].first);
    REXGPU_INFO("[native][owned-frame] frames {}-{}: fallback passes by signature (reason, start, surface base/pitch/format, box, draws, "
                "first draw's shaders), top {} of {}:{}",
                st.first_frame, st.first_frame + st.frames - 1, std::min<size_t>(groups.size(), 12), groups.size(), top);
  }
  if (!st.refusal_groups.empty()) {  // (Session 87: what the refused draws are, most frequent first)
    auto groups = st.refusal_groups;
    std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string top;
    for (size_t i = 0; i < groups.size() && i < 10; ++i) top += fmt::format("\n    {} x {}", groups[i].second, groups[i].first);
    REXGPU_INFO("[native][owned-frame] frames {}-{}: refused draws by signature (reason/detail, shaders, class, blend, depth, mask, "
                "rect, whether a truth pass was open, whether an owned pass was parked), top {} of {}:{}",
                st.first_frame, st.first_frame + st.frames - 1, std::min<size_t>(groups.size(), 10), groups.size(), top);
  }
  if (OwnedChains()) {  // (Session 92: the lighting-cube derivation, per window; the contradictions are cumulative)
    uint32_t cube_stale = 0, cube_older = 0, cube_contra = 0, cube_reused = 0;
    OwnedCubeCounters(cube_stale, cube_older, cube_contra, cube_reused, true);
    REXGPU_INFO("[native][owned-frame] frames {}-{}: cube derivation: stale cubes dropped {}, older derived dropped {}, same-frame contradictions {}, dead cubes reused {}, owned images {} MB, derivation off: {}",
                st.first_frame, st.first_frame + st.frames - 1, cube_stale, cube_older, cube_contra, cube_reused,
                owned_bytes_ >> 20, owned_cube_derive_off_ ? "yes" : "no");
  }
  if (OwnedChains()) {
    REXGPU_INFO("[native][owned-frame] frames {}-{}: owned surfaces: clears into our images {} (colour + depth {} / colour {}; class 1 {} / class 2 {}), "
                "passes started by a clear {} (declined, back to the truth path: {}), depth-only clears into a pass {} (waited {}) / pended {} (dropped {}, run in the truth path {}), "
                "uncommitted passes parked {}, parked passes taken up by a copy or a draw {}, copies of passes begun by a clear served {}, "
                "truth copies that followed a dirty copy-in (truth readers materialised) {}; at the ends: passes begun from held clears at a submission's end {} / frame's end {} / write-out {}, "
                "uncommitted passes committed instead of replayed at a frame's end {} / write-out {}; truth draws that replayed a pending pass {}",
                st.first_frame, st.first_frame + st.frames - 1, st.ch_clears, st.ch_clears_cd, st.ch_clears_c, st.ch_class1, st.ch_class2, st.ch_started,
                st.ch_declined, st.depth_clear_taken, st.depth_clear_deferred, st.dclear_pending, st.dclear_dropped, st.dclear_flushed, st.ch_parked,
                st.ch_takenup, st.ch_served, st.ch_materialised, st.ch_site_sub, st.ch_site_frame, st.ch_site_write, st.ch_commit_frame, st.ch_commit_write,
                st.ch_draw_replays);
    std::string ll;
    {
      auto lines = st.lose_lines;
      std::sort(lines.begin(), lines.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      for (size_t i = 0; i < lines.size() && i < 12; ++i) ll += fmt::format(" {}x line {}", lines[i].second, lines[i].first);
    }
    REXGPU_INFO("[native][owned-frame] frames {}-{}: owned surfaces (2): depth-only clears merged into held colour clears {}; front copies served {} / replayed {}, swap pages unserved {}; starts resumed {} / loaded from truth {} / converted on 0/16/0 {}; readback copies served by materialising {}; dead depth-tested draws dropped {}; only-ours read-back copies {}; copy losses by source line:{}; truth tiles claimed for held-clear passes {} (failed {}); truth draws that committed the pass instead of replaying it {}; parked defined starts replayed before a drop {} [by reason: submission end {}, frame end {}, truth draw {}, write-out {}, copy {}, loaded start {}, other (evicted) {}]; defined starts carried past a submission's end {} (taken up by a copy {} / a draw {}; dropped later by [sub {}, frame {}, truth draw {}, write-out {}, copy {}, loaded start {}, other {}]; parked ones dropped unreplayed at an end {}); carried passes paused: copy not served {}, read by the truth path [sub {}, frame {}, truth draw {}, write-out {}, copy {}, loaded start {}, other {}]; copies of a carried pass dropped at the last frame end {}; dead passes ended uncommitted by a clear {}; owned images made {}, fill sets reused {}",
                st.first_frame, st.first_frame + st.frames - 1, st.ch_dmerge, st.ch_front_served, st.ch_front_lost, st.ch_swap_unserved, st.ch_resumed, st.ch_loaded, st.ch_conv016, st.ch_readback_mat, st.ch_dead_depthless, st.ch_only_ours_readback, ll.empty() ? " none" : ll,
                st.ch_claims, st.ch_claim_failed, st.ch_commit_instead, st.ch_parked_replays,
                st.ch_replay_by[0], st.ch_replay_by[1], st.ch_replay_by[2], st.ch_replay_by[3], st.ch_replay_by[4], st.ch_replay_by[5], st.ch_replay_by[6],
                st.ch_carried_defined, st.ch_carried_taken_copy, st.ch_carried_taken_draw, st.ch_carried_dropped[0], st.ch_carried_dropped[1],
                st.ch_carried_dropped[2], st.ch_carried_dropped[3], st.ch_carried_dropped[4], st.ch_carried_dropped[5], st.ch_carried_dropped[6],
                st.ch_frame_end_unreplayed, st.ch_carried_paused_copy, st.ch_carried_paused_read[0], st.ch_carried_paused_read[1], st.ch_carried_paused_read[2],
                st.ch_carried_paused_read[3], st.ch_carried_paused_read[4], st.ch_carried_paused_read[5], st.ch_carried_paused_read[6], st.ch_copy_of_dropped_carried, st.ch_dead_ended, st.images_made, st.fill_sets_reused);
    {
      // (Session 90, Q: the read-back copies written from our image.)
      std::string refused;
      for (const auto& e : st.wb_refused) refused += fmt::format("{}{} x {}", refused.empty() ? "" : "; ", e.second, e.first);
      REXGPU_INFO("[native][owned-frame] frames {}-{}: owned surfaces (3): read-back copies written from our image {} [8888 1x {}, 8888 scaled {}, full 1x {}, full scaled {}, depth {}] "
                  "(rectangles / uncovered copies written {}, pauses inside windows {}, loaded starts inside windows {}; planned {}, of them into a feedback page of ours {} / of a committed pass only our image holds {}, front-buffer pages {}); write-backs refused or not possible by reason: {}; "
                  "served in a read-back window without a plan {}; parked passes loaded into a new image from their own images {}; read-back copies that kept the replay (reason readback in the census) see 'fallback' above; "
                  "readback copies served by materialising {}",
                  st.first_frame, st.first_frame + st.frames - 1, st.wb_written[0] + st.wb_written[1] + st.wb_written[2] + st.wb_written[3] + st.wb_written[4],
                  st.wb_written[0], st.wb_written[1], st.wb_written[2], st.wb_written[3], st.wb_written[4], st.wb_rect, st.wb_window_pauses, st.wb_window_loads, st.wb_planned, st.wb_dest_ours, st.wb_only_ours, st.wb_front,
                  refused.empty() ? "none" : refused, st.wb_unplanned, st.wb_loaded_parked, st.ch_readback_mat);
      REXGPU_INFO("[native][owned-frame] frames {}-{}: damage-map resets written back from our image {}", st.first_frame, st.first_frame + st.frames - 1,
                  st.k4444_unloaded_wb);
    }
  }
  REXGPU_INFO("[native][owned-frame] frames {}-{}: passes {} owned {} fallback {} (by reason: {}), copies served {} / "
              "page {} / dead {} / written back {} (of {}), clears kept {}, pauses {}, reader refusals {}, no pipeline "
              "{}, flip kept {}, images {} + {} page + {} scratch depth, {} MB, swaps given our image {}, copies clamped to "
              "their box {}, passes committed at a submission's end {}{}{}",
              st.first_frame, st.first_frame + st.frames - 1, st.passes, st.owned, st.fallback,
              reasons.empty() ? "none" : reasons, st.served, st.page, st.dead, st.written_back, st.copies,
              st.clears_kept, st.pauses, st.reader_events, st.no_pipeline, st.flip_kept, owned_images_.size(),
              owned_page_images_.size(), owned_scratch_depths_.size(), owned_bytes_ >> 20, st.presented, st.post_boxes,
              st.sub_commits, lost.empty() ? "" : "; lost by [", lost.empty() ? costs : lost + "]" + costs);
  // Session 95, Lane E: the stale-texture counters. Pages of dropped passes that a sampler asked for (each is a frame of stale texture), whole run.
  REXGPU_INFO("[native][owned-dead] frames {}-{}: dropped-pass pages sampled, whole run {}; copies of held frames left out of the learning, whole run {}; dead_sticky {}; dont_care_clear {}", st.first_frame,
              st.first_frame + st.frames - 1, owned_dead_page_reads_, g_dead_held_copies, REXCVAR_GET(gpu_native_owned_dead_sticky) ? "on" : "off",
              REXCVAR_GET(gpu_native_owned_dont_care_clear));
  // Session 93: the tile-based pass path (gpu_owned_tiler_passes). The would-be counts are kept with the switch off.
  {
    uint32_t transient_scratch = 0;
    for (const OwnedScratchDepth& d : owned_scratch_depths_) transient_scratch += d.transient ? 1u : 0u;
    uint32_t learned_dead = 0, banned = 0;
    for (const OwnedDepthLearn& l : owned_depth_learn_) {
      learned_dead += l.dead ? 1u : 0u;
      banned += l.banned ? 1u : 0u;
    }
    if (REXCVAR_GET(gpu_owned_image_format_list) || OwnedTilerOn()) {
      REXGPU_INFO("[native] format lists on: {} images listed, {} mutable dropped, 0 storage dropped (none to drop: owned colour images have no STORAGE, page images use it) (gpu_owned_image_format_list={})", g_tiler_format_lists,
                  g_afbc_mutable_dropped, REXCVAR_GET(gpu_owned_image_format_list));
    }
    if (REXCVAR_GET(gpu_owned_transient_attachments) || OwnedTilerOn()) {
      REXGPU_INFO("[native] transient attachments on: {} scratch depths transient of {} (ordinary-memory fallbacks {})", transient_scratch,
                  owned_scratch_depths_.size(), g_tiler_transient_fallback);
    }
    REXGPU_INFO("[native][owned-tiler] frames {}-{}: switch {}; resumes {} (with depth of their own {}, WITHOUT {} -> no-depth variant used {}); depth-keeping passes begun {} "
                "(begun with a key learned dead {} [would-be], with the depth store DONT_CARE {}); passes over with their depth never read {} (would-be dead-depth candidates); "
                "learned wrong (a read of a depth stored dead) {}, would-have-been wrong with the switch off {}; keys tracked {} (dead now {}, banned {}); "
                "images made so far: scratch depth transient {} of {} (fell back to ordinary memory {}), images with a format list {}",
                st.first_frame, st.first_frame + st.frames - 1, OwnedTilerOn() ? "ON" : "off", st.tl_resume_depth + st.tl_resume_nodepth, st.tl_resume_depth,
                st.tl_resume_nodepth, st.tl_resume_nodepth_used, st.tl_depth_passes, st.tl_dead_would, st.tl_dead_stored, st.tl_unread_passes, st.tl_learned_wrong,
                st.tl_would_wrong, owned_depth_learn_.size(), learned_dead, banned, transient_scratch, owned_scratch_depths_.size(), g_tiler_transient_fallback,
                g_tiler_format_lists);
  }
  // Session 92: the start / format family.
  if (OwnedChains()) {
    REXGPU_INFO("[native][owned-frame] frames {}-{}: colour-less draws joined {} / refused {} [no depth pass {}, outside box {}, stencil {}, held clears {}]; "
                "restores taken from held clears {} (refused: no held clears {}, not cleared {}); front copies in a fallen frame: pages dropped {}; "
                "stencil claims allowed by an owned pass's stencil {}; stencil draws over unproven stencil {}",
                st.first_frame, st.first_frame + st.frames - 1, g_s92.joined,
                g_s92.no_depth_pass + g_s92.outside_box + g_s92.no_stencil + g_s92.held_clears, g_s92.no_depth_pass, g_s92.outside_box,
                g_s92.no_stencil, g_s92.held_clears, g_s92.restores_taken, g_s92.restores_no_held, g_s92.restores_not_cleared,
                g_s92.fallen_front_drops, g_s92.stencil_claims, g_s92.unproven_stencil);
    REXGPU_INFO("[native][owned-frame] frames {}-{}: colour-only passes given depth {} (from a dirty entry {}, from a pass {}, from a dead depth clear's value {}) (refused: no depth image of ours {}, "
                "no colour pass or an ordinary start (not a refusal) {}, edram-free pass {}, held clears for its depth {}, size {}, half {}, stencil {}, budget {}); held frames not aging the front page {}",
                st.first_frame, st.first_frame + st.frames - 1, g_s92.depth_given, g_s92.depth_from_dirty, g_s92.depth_from_pass, g_s92.depth_from_clear,
                g_s92.depth_no_source, g_s92.depth_no_colour, g_s92.depth_edram_free, g_s92.depth_held, g_s92.depth_size, g_s92.depth_half, g_s92.depth_stencil, g_s92.depth_budget,
                g_s92_held_not_aging);
  }
  if (REXCVAR_GET(gpu_native_owned_only)) {
    // Session 92, M2a: the owned-only latch's window line (docs\native\d3d\S92_M2_DESIGN.md section 4 item 7).
    const auto top = [](std::vector<std::pair<std::string, uint32_t>> groups, size_t n) {
      std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      std::string out;
      for (size_t i = 0; i < groups.size() && i < n; ++i) out += fmt::format("{}{} x{}", out.empty() ? "" : "; ", groups[i].first, groups[i].second);
      return out.empty() ? std::string("none") : out;
    };
    std::string by_reason;
    for (uint32_t i = 0; i < 16; ++i) {
      if (!g_m2a.draw_reasons[i]) continue;
      by_reason += fmt::format("{}{} {}", by_reason.empty() ? "" : ", ", i == 15 ? "claim refused" : OwnedReasonName(i), g_m2a.draw_reasons[i]);
    }
    uint32_t truth_images_alive = 0;
    for (const TruthImage& t : truth_images_) truth_images_alive += t.image != VK_NULL_HANDLE ? 1u : 0u;
    REXGPU_INFO("[native][owned-only] frames {}-{}: latch {}; draws dropped {} (by reason: {}; claims refused and dropped {}), skipped for no pipeline {} / no pixels {}; "
                "copies dropped {} (that kept a page of ours {}; CPU read-backs dropped {} [bar 0]; front copies dropped {} [bar 0]), written after a loss {}, front copies written {}; copies served from a surface record {}; replays avoided {}; pauses avoided {}; fallbacks ignored {} (clearing resolves {}, write-outs {}); "
                "held clears dropped {} (record clears lost {}), pending depth-only clears dropped {}; surface records {} live ({} noted, {} superseded, {} evicted); "
                "starts resumed from a record {} (none {}); colour-only passes given depth from a record {} (from a dead depth clear's value {}); truth images alive {} of {} slots; latch released {} (kept owned-only for want of an EDRAM buffer {}); stock EDRAM paths asked for with no EDRAM buffer, whole run {} (boot copies dropped, no EDRAM buffer, before the first frame boundary {})",
                st.first_frame, st.first_frame + st.frames - 1, OwnedOnly() ? "on" : "RELEASED", g_m2a.draws_dropped, by_reason.empty() ? "none" : by_reason,
                g_m2a.claims_dropped, g_m2a.draws_skipped_pipeline, g_m2a.draws_skipped_empty, g_m2a.copies_dropped, g_m2a.drops_kept_page, g_m2a.readback_dropped, g_m2a.front_dropped,
                g_m2a.copies_written_after_loss, g_m2a.front_written, g_m2a.from_record, g_m2a.replays_avoided,
                g_m2a.pauses_avoided, g_m2a.fallbacks_ignored, g_m2a.clearing_resolves, g_m2a.writeouts_ignored, g_m2a.held_dropped, g_m2a.lost_clears,
                g_m2a.dclear_dropped, owned_newest_.size(), g_m2a.records_noted, g_m2a.records_superseded, g_m2a.records_evicted, m2a_loaded_from_record_,
                m2a_loaded_no_record_, g_m2a.depth_from_record, g_m2a.depth_from_dead_clear, truth_images_alive, truth_images_.size(), g_m2a.latch_released, g_m2a.latch_kept_no_edram, M2dEdramLessDrops(), M2dEdramLessBootDrops());
    {
      // Session 92, M2b: the owned entry's window line (M0 "truth entries" stays at its site: it reads 0 here because TruthRecordDraw is not called).
      uint64_t claims = 0, lean = 0, repeat = 0;
      uint32_t surfaces_new = 0, would_alias = 0, table_full = 0;
      OwnedEntryCounters(claims, lean, repeat, surfaces_new, would_alias, table_full, true);
      uint32_t truth_alive = 0;
      for (const TruthImage& t : truth_images_) truth_alive += t.image != VK_NULL_HANDLE ? 1u : 0u;
      REXGPU_INFO("[native][owned-only] frames {}-{}: owned entry {}; surfaces {} ({} new keys this window; {} keys sharing a base with an earlier key [would-have-aliased, upper bound]; "
                  "table full {}); claims: {} full / {} lean / {} repeat; truth images {} (of {} truth slots)",
                  st.first_frame, st.first_frame + st.frames - 1, owned_surf_mode_ ? "on" : "OFF", owned_surfaces_.size(), surfaces_new, would_alias, table_full, claims, lean, repeat,
                  truth_alive, truth_images_.size());
    }
    OwnedPendPrint(st.first_frame, st.frames);  // (Session 92, M4f-B: the "[native][pend]" line)
    REXGPU_INFO("[native][owned-only] frames {}-{}: top draw drops [{}]", st.first_frame, st.first_frame + st.frames - 1, top(g_m2a.draw_groups, 12));
    REXGPU_INFO("[native][owned-only] frames {}-{}: copies dropped by source [{}]; replays / pauses avoided by why [{}]", st.first_frame, st.first_frame + st.frames - 1,
                top(g_m2a.copy_groups, 12), top(g_m2a.avoided_why, 10));
    REXGPU_INFO("[native][owned-only] frames {}-{}: passes owned as dead (never committed, every copy dead) {}; copies filled (defined transparent black, M2c) {} (CPU read-backs filled {}); filled by source [{}]; copies written as a composite (larger than their pass's box; M2d) {} (CPU read-backs {}, lower part filled {})", st.first_frame,
                st.first_frame + st.frames - 1, g_m2a.dead_owned, g_m2a.copies_filled, g_m2a.readback_filled, top(g_m2a.fill_groups, 8), g_m2a.composites, g_m2a.composite_readback, g_m2a.composite_lower_filled);
  }
  g_m2a = M2aStats();
  m2a_loaded_from_record_ = m2a_loaded_no_record_ = 0;
  g_s92_held_not_aging = 0;
  g_s92 = S92Stats();
  st = OwnedFrameStats();
}

void NativeRenderer::OwnedRefreshFrame() {
  g_m2d_draw_index = 0;  // (Session 92, M2d: the order log's per-frame draw index)
  // Once a frame, at its end: the switch (a change applies from the next frame) and the census log window.
  // (Session 92, M4f-C) under THE CUT there is no old path to draw a frame the owned frame does not: the owned frame is held on, flips of the switch are ignored (counted, logged once)
  const bool cut_on = rex::graphics::free_counters::g_cut.active;
  static bool s_switched_off = false;  // (counted per switch-off, not per frame)
  if (cut_on && !REXCVAR_GET(gpu_native_owned_frame)) {
    if (!s_switched_off) {
      s_switched_off = true;
      ++rex::graphics::free_counters::g_cut.owned_flips_ignored;
      REXGPU_INFO("[d3d][cut] frame {}: gpu_native_owned_frame was switched off - ignored under the cut (the owned frame is held on; no old path to draw it)", frame_);
    }
  } else {
    s_switched_off = false;
  }
  const bool on = REXCVAR_GET(gpu_native_owned_frame) || cut_on;
  if (on != owned_frame_on_) OwnedFrameFlip(on);
  owned_only_latched_ = REXCVAR_GET(gpu_native_owned_only);  // (latched at the frame boundary: a mid-frame change never splits a frame)
  g_m2d_edram_less = owned_only_latched_ && M2dEdramLess();  // (the cache's own decision: it made no EDRAM buffer; review E2)
  M2dRendererFrameSeen();  // (review E5: from here on a stock copy dropped is not a boot one)
  if (!OwnedOnlyWanted() && !owned_newest_.empty()) owned_newest_.clear();  // (review C7: records pinned after a release / a switch-off)
  {
    // Session 92, M2b: the slot table of the next frame - the surface registry under the latch, the truth images otherwise. A change of table (the latch on or released)
    // leaves no pass, held clear or record that carries a slot of the other table: they end as is (their images are not touched).
    const bool surf_mode = OwnedOnlyWanted();
    if (surf_mode != owned_surf_mode_) {
      REXGPU_INFO("[native][owned-only] frame {}: the slot table is the {} from the next frame ({} surface keys, {} parked passes and {} records dropped)", frame_,
                  surf_mode ? "surface registry" : "truth images", owned_surfaces_.size(), owned_parked_.size(), owned_newest_.size());
      owned_surf_mode_ = surf_mode;
      if (surf_mode) {
        ++g_draw_rec_generation;  // (Session 95, r3cmd: a record holds slots of the registry)
        owned_surfaces_.clear();
        owned_dirty_.clear();  // (review F2: truth-slot entries hold owned images and compare with registry slots - nothing reads truth images under the latch)
        owned_clean_pending_.clear();
      }
      owned_parked_.clear();
      owned_newest_.clear();
      owned_held_.draws.clear();
      owned_held_.info = ClearsInfo();
      owned_depth_clears_.clear();
      owned_depth_dropped_.clear();
      OwnedPendClearReset();  // (Session 92, M4f-B: the slots of the old table mean something else)
      ++truth_claim_epoch_;
    }
  }
  if (g_m2d_edram_less && owned_only_latched_ && owned_frame_on_ && (owned_failed_ || truth_failed_) && !owned_only_released_logged_) {
    owned_only_released_logged_ = true;  // (Session 92, M2d: no EDRAM buffer, so no truth path to hand over to - counted once, the frame goes on owned-only and drops what it cannot take)
    ++g_m2a.latch_kept_no_edram;
    if (truth_failed_ && !truth_fail_reason_.empty()) ui::vulkan::GpuFatalExit(truth_fail_reason_);  // Session 95 (PLAIN, review): TRUE NATIVE has no other path - a named end, not a black frame
    REXGPU_ERROR("[native][owned-only] frame {}: a hard failure ({}) would release the owned-only latch, but there is no EDRAM buffer (TRUE NATIVE) and so no truth path - the latch is KEPT, the "
                 "frame goes on owned-only, dropping what it cannot take (counted)", frame_, owned_failed_ ? "owned images failed" : "the truth objects failed");
  }
  if (!g_m2d_edram_less && !rex::graphics::free_counters::g_cut.active && owned_only_latched_ && owned_frame_on_ && (owned_failed_ || truth_failed_) && !owned_only_released_logged_) {  // (M4f-C: never released under the cut)
    owned_only_released_logged_ = true;  // (Session 92, M2a: the hard failure releases the latch for the session - the old path is the only safety)
    ++g_m2a.latch_released;
    REXGPU_ERROR("[native][owned-only] frame {}: the owned-only latch is RELEASED for the session ({}) - the truth path is the only safety left", frame_,
                 owned_failed_ ? "owned images failed" : "the truth objects failed");
  }
  // The log: the cvar > 0 turns the per-pass lines on for the frames the segment log covers (the routes set that live);
  // set live to another value > 0 it logs that many frames from the next one.
  const int32_t n = REXCVAR_GET(gpu_native_owned_frame_log);
  if (owned_frame_log_seen_ == INT32_MIN) {
    owned_frame_log_seen_ = n;
  } else if (n != owned_frame_log_seen_) {
    owned_frame_log_seen_ = n;
    if (n > 0) owned_frame_log_left_ = n;
  }
  owned_frame_logging_ = owned_frame_on_ && n > 0 && (segment_log_frames_ > 0 || owned_frame_log_left_ > 0);
  if (owned_frame_log_left_ > 0) --owned_frame_log_left_;
}

void NativeRenderer::OwnedFrameFlip(bool on) {
  // Between frames (OwnedOnFrameEnd ended the pending pass already - replayed if it was not committed).
  REXGPU_INFO("[native][owned-frame] the owned frame turns {} from frame {}", on ? "on" : "off", frame_);
  if (!on) {
    OwnedFrameSummary();
    owned_alpha_keys_.clear();  // (the learned alpha flags go with the owned frame)
    // Off: every owned state goes back to the truth path - the dirty tiles' pixels copied in; owned pages stay mapped
    // until the game's next copy into them (no write-back: nothing reads the texture until then).
    if (owned_pending_.valid) OwnedReplayPending("the owned frame turns off");
    owned_newest_.clear();  // (Session 92, M2a: the surfaces' newest records go with the owned frame)
    OwnedPruneDirty();
    for (OwnedDirty& e : owned_dirty_) OwnedMaterialize(e, e.tiles);
    OwnedPruneDirty();
    uint32_t kept = 0;
    for (const OwnedPage& pg : owned_pages_) kept += pg.frame_owned ? 1u : 0u;
    owned_frame_stats_.flip_kept += kept;
    REXGPU_INFO("[native][owned-frame] flip off: {} owned pages kept until the game copies into them", kept);
  } else {
    // On: the deferred copies of earlier frames are run first (a carried copy of a page the owned frame is about to own
    // must not write its texture after the substitution), and the CPU-write watch is in place.
    TruthFlushDeferred(kDeferFlushFrameEnd);
    owned_paused_.clear();
  }
  owned_frame_on_ = on;
  if (on) OwnedWatchRegister();
  owned_frame_passes_.clear();
  owned_frame_last_ = -1;
}

void NativeRenderer::OwnedNoteEndedClears() {
  owned_clears_ended_ = false;
  if (owned_surf_mode_ || OwnedOnly()) return;  // (Session 92, M2b: no truth pass exists under the latch - no clears tiles)
  if (!owned_frame_on_ || owned_replaying_ || !truth_ || !replacing_) return;
  const auto& pc = truth_pass_clears_;
  if (!pc.valid || pc.area == 0 || truth_pass_color_ < 0 || size_t(truth_pass_color_) >= truth_images_.size()) return;
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  if (pc.rect[0] != 0 || pc.rect[1] != 0 || pc.rect[2] <= 0 || pc.rect[3] <= 0) return;
  const uint32_t w = uint32_t(pc.rect[2]) / scale_x, h = uint32_t(pc.rect[3]) / scale_y;
  const TruthImage& t = truth_images_[truth_pass_color_];
  std::bitset<2048> tiles = EdramTiles(t.base, t.pitch_tiles, t.is_64bpp(), 0, 0, w, h);
  if (pc.uses_depth && truth_pass_depth_ >= 0 && size_t(truth_pass_depth_) < truth_images_.size()) {
    const TruthImage& d = truth_images_[truth_pass_depth_];
    tiles |= EdramTiles(d.base, t.pitch_tiles, false, 0, 0, w, h);
  }
  owned_clears_tiles_ = tiles;
  owned_clears_submission_ = command_processor_.GetCurrentSubmission();
  owned_clears_ended_ = true;
}

void NativeRenderer::OwnedClearsTouched() { owned_clears_ended_ = false; }

void NativeRenderer::OwnedCheckSwapPage(uint32_t page) {
  if (!owned_frame_on_) return;
  page &= 0x1FFFF;
  for (const OwnedPage& pg : owned_pages_) {
    if (pg.page != page) continue;
    if (OwnedChains()) {  // (nothing is banned: counted and logged)
      ++owned_frame_stats_.ch_swap_unserved;
      {  // (review A2: remembered, so the next copy into the page runs from the truth path)
        bool found = false;
        for (auto& u : owned_unserved_pages_) {
          if (u.first == page) {
            u.second = frame_ + 120;
            found = true;
          }
        }
        if (!found && owned_unserved_pages_.size() < 16) owned_unserved_pages_.push_back({page, frame_ + 120});
      }
      static uint32_t swap_logged = 0;
      if (swap_logged < 12) {
        ++swap_logged;
        REXGPU_INFO("[native][owned] frame {}: the swap presents owned page 0x{:05X} that the hook cannot serve (pass base {} pitch {} ps {:016X}); not banned", frame_, page,
                    pg.key.base, pg.key.pitch, pg.key.ps);
      }
      continue;
    }
    OwnedPresentAdd(pg.key, "the swap presents an owned page it cannot serve");
    REXGPU_WARN("[native][owned] frame {}: the swap presents owned page 0x{:05X}: its pass (base {} pitch {} ps {:016X}) "
                "is refused from now on (present)", frame_, page, pg.key.base, pg.key.pitch, pg.key.ps);
  }
}

void NativeRenderer::OwnedNoteFrontBuffer(uint32_t frontbuffer_ptr) {
  const uint32_t page = (frontbuffer_ptr >> 12) & 0x1FFFF;
  if (!page) return;
  for (const uint32_t f : owned_front_pages_) {
    if (f == page) return;
  }
  owned_front_pages_[owned_front_next_++ & 3] = page;
}

void NativeRenderer::OwnedMarkLeftOut(const OwnedPage& page) {
  // (With the owned frame on: the page's memory is GPU-written as far as the shared memory goes - the CPU-write
  // invalidation is not widened over it, so a write next to it does not drop the page.)
  if (!owned_frame_on_) return;
  if (SharedMemory* const shared_memory = command_processor_.NativeSharedMemory()) {
    // (Only the pages the extent holds whole: a page it shares with other data - a vertex buffer the cover estimate reads -
    // is not marked GPU-written; the estimate refuses such data, gpu_native_own_steps blurs lost their cover to it.)
    const uint32_t first = (page.extent_start + 4095u) & ~4095u, last = (page.extent_start + page.extent_length) & ~4095u;
    if (last > first) shared_memory->MarkRangeGpuWrittenQuiet(first, last - first);
  }
}

void NativeRenderer::OwnedWatchRegister() {
  if (!owned_frame_on_ || owned_watch_handle_) return;
  SharedMemory* const shared_memory = command_processor_.NativeSharedMemory();
  if (!shared_memory) return;
  // (A CPU write into a page's memory - from a guest thread, in the global critical region: the ranges it hits are noted
  // for the render thread, which drops the pages. Writes by the GPU (the resolves) are the copies' business.)
  owned_watch_handle_ = shared_memory->RegisterGlobalWatch(
      [](const std::unique_lock<std::recursive_mutex>&, void* context, uint32_t address_first, uint32_t address_last,
         bool invalidated_by_gpu) {
        if (invalidated_by_gpu) return;
        NativeRenderer* const self = static_cast<NativeRenderer*>(context);
        std::lock_guard<std::mutex> lock(self->owned_watch_mutex_);
        for (const auto& e : self->owned_watch_extents_) {
          if (address_first < e.second && e.first <= address_last) {
            self->owned_watch_ranges_.emplace_back(e.first, e.second);
            self->owned_watch_pending_.store(true, std::memory_order_release);
          }
        }
      },
      this);
}

void NativeRenderer::OwnedWatchSync() {
  if (!owned_watch_handle_) return;  // (the list mode has no watch)
  std::lock_guard<std::mutex> lock(owned_watch_mutex_);
  owned_watch_extents_.clear();
  for (const OwnedPage& p : owned_pages_) {
    if (p.frame_owned) owned_watch_extents_.emplace_back(p.extent_start, p.extent_start + p.extent_length);
  }
}

void NativeRenderer::OwnedDrainCpuWrites() {
  std::vector<std::pair<uint32_t, uint32_t>> ranges;
  {
    std::lock_guard<std::mutex> lock(owned_watch_mutex_);
    ranges.swap(owned_watch_ranges_);
    owned_watch_pending_.store(false, std::memory_order_release);
  }
  if (ranges.empty() || owned_pages_.empty()) return;
  const size_t before = owned_pages_.size();
  owned_pages_.erase(std::remove_if(owned_pages_.begin(), owned_pages_.end(),
                                    [&](const OwnedPage& p) {
                                      if (!p.frame_owned) return false;
                                      const uint64_t start = p.extent_start, end = start + p.extent_length;
                                      for (const auto& r : ranges) {
                                        if (start == r.first && end == r.second) {
                                          static uint32_t logged = 0;
                                          if (logged < 32) {
                                            ++logged;
                                            REXGPU_INFO("[native][owned] frame {}: the CPU wrote page 0x{:05X}: no longer "
                                                        "ours", frame_, p.page);
                                          }
                                          return true;
                                        }
                                      }
                                      return false;
                                    }),
                     owned_pages_.end());
  if (owned_pages_.size() != before) {
    command_processor_.NativeOwnedPagesChanged();
    OwnedWatchSync();
  }
}

// ---- the extension points (Session 82): each later piece fills in its own .cpp -----------------------------------

void NativeRenderer::OwnedDestroy() {
  const ui::vulkan::VulkanDevice* vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (OwnedViewEntry& v : owned_views_) ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, v.view);
  owned_views_.clear();
  ++owned_view_generation_;  // (Session 95, r6tex review: no memo answers with a destroyed view)
  for (OwnedImage& o : owned_images_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, o.framebuffer);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, o.framebuffer_depth);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, o.framebuffer_depth_up);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.array_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_sample_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, o.depth_stencil_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, o.image);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, o.depth);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, o.memory);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, o.depth_memory);
  }
  owned_images_.clear();
  for (OwnedScratchDepth& d : owned_scratch_depths_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, d.view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, d.image);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, d.memory);
  }
  owned_scratch_depths_.clear();
  DecoupleCompareDestroy();  // Session 94
  owned_bytes_ = 0;
  if (owned_watch_handle_ && command_processor_.NativeSharedMemory()) {
    command_processor_.NativeSharedMemory()->UnregisterGlobalWatch(owned_watch_handle_);
  }
  owned_watch_handle_ = nullptr;
  for (OwnedPageImage& q : owned_page_images_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device, q.storage_view);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, q.image);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, q.memory);
  }
  owned_page_images_.clear();
  for (OwnedFeedback& f : owned_feedback_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, f.image);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, f.memory);
  }
  owned_feedback_.clear();
  // (Its sets - the fill sources, the page images' storage - go with the pools.)
  for (VkDescriptorPool pool : owned_pools_) dfn.vkDestroyDescriptorPool(device, pool, nullptr);
  owned_pools_.clear();
  owned_spare_sets_.clear();
  owned_wb_spare_sets_.clear();
  owned_pool_ = VK_NULL_HANDLE;
  owned_pages_.clear();
  owned_dirty_.clear();
  owned_newest_.clear();
  for (auto& passes : owned_render_passes_) {
    for (VkRenderPass& rp : passes) ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device, rp);
  }
}

bool NativeRenderer::OwnedChains() const { return REXCVAR_GET(gpu_native_owned_frame_chains); }

// ---- Session 92, M2a: the owned-only latch (gpu_native_owned_only; docs\native\d3d\S92_M2_DESIGN.md section 4) -------------------------------------

bool NativeRenderer::OwnedOnly() const { return owned_surf_mode_; }

// Session 95: what the resolution row (gpu_native_truth_draw_height) makes of the owned scene pass under TRUE NATIVE: its height over the whole size, within the ratio's 0.5..1.
// 1 where the old truth rule gave no draw height (DLSS, DLAA, EDRAM renaming, an uneven scale).
double OwnedRowRatio(uint32_t scale_x, uint32_t scale_y) {
  const int32_t rows = REXCVAR_GET(gpu_native_truth_draw_height);
  if (rows <= 0 || !REXCVAR_GET(gpu_native_owned_only) || scale_x != scale_y || uint32_t(rows) >= 720u * scale_y || REXCVAR_GET(gpu_edram_rename) || REXCVAR_GET(gpu_dlss) ||
      REXCVAR_GET(gpu_dlaa)) {
    return 1.0;
  }
  return std::max(0.5, double(rows) / (720.0 * double(scale_y)));
}

bool NativeRenderer::OwnedOnlyWanted() const {
  // The cvar, with the owned frame and its chains on (TRUE NATIVE). A hard owned failure (owned images cannot be made, the truth objects failed)
  // releases it - the old path is the only safety left for that; OwnedRefreshFrame logs the release loudly, once.
  // (Session 92, M2d: with no EDRAM buffer the truth path cannot take over - a hard failure is counted and the frame goes on owned-only, dropping.)
  return owned_only_latched_ && owned_frame_on_ && OwnedChains() && (g_m2d_edram_less || rex::graphics::free_counters::g_cut.active || (!owned_failed_ && !truth_failed_));
}

void NativeRenderer::OwnedNoteClaimDrop(uint32_t truth_reason, const Draw& draw) {
  ++g_m2a.claims_dropped;
  M2aGroup(g_m2a.draw_groups, fmt::format("claim refused ({}): ps {:016X} vs {:016X}", TruthReasonName(truth_reason), draw.pixel_shader_hash, draw.vertex_shader_hash));
  static uint32_t logged = 0;
  if (logged < 24) {
    ++logged;
    REXGPU_INFO("[native][owned-only] frame {}: claim refused ({}), claimed and dropped: vs {:016X} ps {:016X} primitive {} supported {} pitch {} msaa {} colour 0x{:08X} "
                "depth 0x{:08X} masks 0x{:X} depth test {} write {} stencil {} fast clear {} shader {}",
                frame_, TruthReasonName(truth_reason), draw.vertex_shader_hash, draw.pixel_shader_hash, draw.guest_primitive, draw.raster.supported,
                draw.surface_info & 0x3FFF, (draw.surface_info >> 16) & 3, draw.color_info, draw.depth_info, draw.color_mask_all, draw.raster.depth_test,
                draw.raster.depth_write, draw.raster.stencil, draw.fast_clear, draw.pixel_shader != VK_NULL_HANDLE);
  }
}

void NativeRenderer::OwnedNoteFallbackIgnored(uint32_t truth_reason) {
  ++g_m2a.fallbacks_ignored;
  if (std::string_view(TruthReasonName(truth_reason)) == "clearing resolve") ++g_m2a.clearing_resolves;
  static uint32_t logged = 0;
  if (logged < 8) {
    ++logged;
    REXGPU_INFO("[native][owned-only] frame {}: a truth fallback ({}) is ignored - the frame goes on", frame_, TruthReasonName(truth_reason));
  }
}

void NativeRenderer::OwnedDropDraw(const Draw& draw) {
  // A draw the owned frame did not take (or whose claim was refused): nothing is recorded, no pass is touched. Skipped (not dropped): no pipeline yet
  // (as the truth path skips it) and a draw of no pixels. Else counted by the reason OwnedTakeDrawImpl noted and the draw's signature.
  serve_pending_.clear();
  serve_rename_.valid = false;
  const std::string detail = owned_frame_pending_detail_;
  const uint8_t reason = owned_frame_pending_reason_;
  owned_frame_pending_reason_ = kOwnedReasonNone;
  owned_frame_pending_detail_.clear();
  const bool refused_claim = draw.claim_refused != 0;
  if (!refused_claim && draw.pipeline == VK_NULL_HANDLE && !draw.d3d_record) {  // (review D1: a refused claim has no pipeline too - it is a drop)
    ++g_m2a.draws_skipped_pipeline;
    return;
  }
  int32_t rect[4] = {0, 0, 0, 0};
  if (!refused_claim) {
    TruthDrawRect(draw, rect);
    if (rect[2] <= rect[0] || rect[3] <= rect[1]) {
      ++g_m2a.draws_skipped_empty;
      return;
    }
  }
  if (draw.fast_clear && draw.truth_depth >= 0 && draw.raster.depth_write) OwnedMarkDepthStale(draw.truth_depth);  // (review D8: a dropped depth-writing clear)
  ++g_m2a.draws_dropped;
  ++g_m2a.draw_reasons[refused_claim ? 15 : std::min<uint32_t>(reason, 14u)];
  if (draw.d3d_record) {
    ++g_m2a.lost_clears;
    D3dNoteClearLost("owned-only drop");
  }
  const std::string why = refused_claim ? fmt::format("claim/{}", TruthReasonName(draw.claim_refused))
                                        : fmt::format("{}/{}", OwnedReasonName(reason), owned_flushing_held_ ? std::string("flushed clear") : detail);
  const std::string sig = fmt::format("{} ps {:016X} vs {:016X} cls {} blend {:08X} z {}{} st {} mask {:X} fc {} rect {},{}-{},{}", why, draw.pixel_shader_hash,
                                      draw.vertex_shader_hash, draw.truth_class, draw.blend_control, draw.raster.depth_test ? "t" : "-",
                                      draw.raster.depth_write ? "w" : "-", draw.raster.stencil, draw.color_write_mask & 0xF, draw.fast_clear, rect[0],
                                      rect[1], rect[2], rect[3]);
  M2aGroup(g_m2a.draw_groups, sig);
  static uint32_t logged = 0;
  if (logged < 24) {
    ++logged;
    REXGPU_INFO("[native][owned-only] frame {}: draw dropped: {} (surface colour 0x{:08X} depth 0x{:08X} pitch {}; pass pending {} base {} pitch {}, {} records)", frame_,
                sig, draw.color_info, draw.depth_info, draw.surface_info & 0x3FFF, owned_pending_.valid, owned_pending_.base, owned_pending_.pitch_tiles,
                owned_newest_.size());
  }
}

void NativeRenderer::OwnedRecordsTrim(const char* why) {
  // At most 24 records (an image each); the oldest by its last draw goes first. Also the budget trims them (OwnedEvictFor).
  (void)why;
  while (owned_newest_.size() > 24) {
    size_t oldest = 0;
    for (size_t i = 1; i < owned_newest_.size(); ++i) {
      if (owned_newest_[i].last_tick < owned_newest_[oldest].last_tick) oldest = i;
    }
    owned_newest_.erase(owned_newest_.begin() + oldest);
    ++g_m2a.records_evicted;
  }
}

void NativeRenderer::OwnedNoteNewest(const OwnedPass& qin) {
  // A pass that ends (its copy served or not) is its surface's newest record: its image, box, depth / stencil and the tick of its last draw, kept as a
  // committed pass a later copy or start takes up. A record of the same image is replaced; an older record of the same colour surface loses its colour, of
  // the same depth slot its depth (E1 and the copy code read the flags); both gone = erased. The image is held (OwnedImageHeld) until then.
  if (qin.image < 0 || size_t(qin.image) >= owned_images_.size() || !owned_images_[qin.image].width) return;
  OwnedPass q = qin;
  q.valid = true;
  q.committed = true;
  q.open = false;
  q.draws.clear();
  q.crowd_calls.clear();
  q.serves.clear();
  q.held_clears.clear();
  q.kept_sampled.clear();
  q.census = -1;  // (a record outlives the frame whose census it pointed into)
  q.chain_parked = false;
  if (!q.last_tick) q.last_tick = ++owned_tick_;
  for (size_t i = 0; i < owned_newest_.size();) {
    OwnedPass& r = owned_newest_[i];
    if (r.image == q.image) {
      owned_newest_.erase(owned_newest_.begin() + i);
      continue;
    }
    // (Review D3: the notes come in any order - the frame end notes parked passes newest first - so the NEWER of the two by tick keeps the surface /
    // the depth slot, whichever is noted last.)
    // (Only when both sides are live: a side that already lost its colour / depth cannot take it from the other.)
    if (r.base == q.base && r.pitch_tiles == q.pitch_tiles && r.kind == q.kind) {
      if (!q.rec_colour_gone && !r.rec_colour_gone) {
        if (r.last_tick > q.last_tick) q.rec_colour_gone = true;
        else r.rec_colour_gone = true;
      } else if (!q.rec_colour_gone) {
        // r already lost it
      }
    }
    if (q.depth && r.depth && r.depth_slot == q.depth_slot && !q.rec_depth_gone && !r.rec_depth_gone) {
      if (r.depth_tick > q.depth_tick) q.rec_depth_gone = true;
      else r.rec_depth_gone = true;
    }
    if (r.rec_colour_gone && (!r.depth || r.rec_depth_gone)) {
      owned_newest_.erase(owned_newest_.begin() + i);
      ++g_m2a.records_superseded;
      continue;
    }
    ++i;
  }
  if (q.rec_colour_gone && (!q.depth || q.rec_depth_gone)) {  // (an older pass noted late: nothing of it is the newest)
    ++g_m2a.records_superseded;
    return;
  }
  owned_newest_.push_back(std::move(q));
  ++g_m2a.records_noted;
  OwnedRecordsTrim("noted");
}

void NativeRenderer::OwnedMarkDepthStale(int16_t depth_slot) {
  if (depth_slot < 0) return;
  for (OwnedPass& r : owned_newest_) {
    if (r.depth && r.depth_slot == depth_slot) r.rec_depth_gone = true;
  }
  for (OwnedPass& r : owned_parked_) {
    if (r.depth && r.depth_slot == depth_slot) r.rec_depth_gone = true;
  }
  if (owned_pending_.valid && owned_pending_.depth && owned_pending_.depth_slot == depth_slot) owned_pending_.rec_depth_gone = true;
}

void NativeRenderer::OwnedSupersedeRecords(uint32_t base, uint32_t pitch, uint32_t kind, int16_t depth_slot) {
  // Session 92, M2a (review D4): a pass of this surface starts - the older records' colour (and the depth slot it keeps) is no longer the newest. A record
  // with nothing newest left is let go.
  if (!OwnedOnly()) return;
  for (size_t i = 0; i < owned_newest_.size();) {
    OwnedPass& r = owned_newest_[i];
    if (r.base == base && r.pitch_tiles == pitch && r.kind == kind) r.rec_colour_gone = true;
    if (depth_slot >= 0 && r.depth && r.depth_slot == depth_slot) r.rec_depth_gone = true;
    if (r.rec_colour_gone && (!r.depth || r.rec_depth_gone)) {
      owned_newest_.erase(owned_newest_.begin() + i);
      ++g_m2a.records_superseded;
      continue;
    }
    ++i;
  }
}

// Session 92, part D1 (TRUE NATIVE): an owned pass (pending or parked) keeps the stencil of this depth slot and its box holds the rect - its stencil is
// the surface's there (a stencil draw is not refused for lack of a truth clear).
// Session 92 (census only, TRUE NATIVE): a stencil draw recorded by the truth path whose covered tiles are not all stencil_valid & owned of its
// depth image - the Session 47 test ClaimTruth applies, redone after OwnedDirtyBeforeDraw. Counts; changes nothing.
void NativeRenderer::OwnedCountUnprovenStencil(const Draw& draw) {
  if (!owned_frame_on_ || !OwnedChains() || draw.truth_depth < 0 || size_t(draw.truth_depth) >= truth_images_.size()) return;
  int32_t rect[4];
  TruthDrawRect(draw, rect);
  uint32_t scale_x = 1, scale_y = 1;
  command_processor_.NativeResolutionScale(scale_x, scale_y);
  const TruthImage& d_image = truth_images_[draw.truth_depth];
  const uint32_t gx0 = uint32_t(std::max(rect[0], 0)) / scale_x, gy0 = uint32_t(std::max(rect[1], 0)) / scale_y;
  const uint32_t gx1 = (uint32_t(std::max(rect[2], rect[0])) + scale_x - 1) / scale_x;
  const uint32_t image_rows = (2048u + d_image.row_tiles() - 1) / d_image.row_tiles() * 16;
  const uint32_t gy1 = std::min((uint32_t(std::max(rect[3], rect[1])) + scale_y - 1) / scale_y, image_rows);
  if (gx1 <= gx0 || gy1 <= gy0) return;
  const std::bitset<2048> tiles = EdramTiles(d_image.base, d_image.pitch_tiles, false, gx0, gy0, gx1, gy1);
  if ((tiles & ~(d_image.stencil_valid & d_image.owned)).any()) ++g_s92.unproven_stencil;
}

bool NativeRenderer::OwnedHoldsStencil(int16_t depth_slot, uint32_t pitch_tiles, const int32_t rect[4]) {
  if (!owned_frame_on_ || !OwnedChains()) return false;
  const auto holds = [&](const OwnedPass& q) {
    return q.valid && q.depth && q.stencil && q.depth_slot == depth_slot && q.pitch_tiles == pitch_tiles && rect[0] >= 0 && rect[1] >= 0 &&
           rect[2] <= int32_t(q.host_width) && rect[3] <= int32_t(q.host_height);
  };
  bool yes = holds(owned_pending_);
  for (size_t i = 0; !yes && i < owned_parked_.size(); ++i) yes = holds(owned_parked_[i]);
  if (yes) ++g_s92.stencil_claims;
  return yes;
}

}  // namespace rex::graphics::vulkan::native
