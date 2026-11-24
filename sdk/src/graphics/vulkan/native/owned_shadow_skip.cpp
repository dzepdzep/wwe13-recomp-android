// wwe2k14 patch (Session 95, item K): see owned_shadow_skip.h.
#include "owned_shadow_skip.h"

#include <algorithm>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(gpu_native_owned_shadow_skip, false, "GPU/Native",
                    "Session 95 (item K, off by default; LOW sets it with gpu_shadow_half_rate): in a frame where gpu_shadow_half_rate skipped every "
                    "wrestler shadow map, the passes that read only those maps (the coverage reductions 901118D1 / 0DC910FA and the two blurs "
                    "7C650AA1 / 694A6D14) are skipped too - draws and copy - and keep last frame's pages (never two frames in a row). Without "
                    "gpu_shadow_half_rate nothing is skipped");

REXCVAR_DEFINE_BOOL(gpu_native_owned_strip_half, false, "GPU/Native",
                    "Session 95 (resolves, off by default; LOW sets it): the entrance's light-cube strip passes (ps 8479D8DA into a 384x64 target, 5 passes with 6 "
                    "copies each) are drawn every other frame - the draws are skipped and every copy of those passes dropped on the frames between, "
                    "the cubes keep last frame's faces (one frame old lighting on the wrestlers)");

REXCVAR_DEFINE_BOOL(gpu_native_owned_ring_half, false, "GPU/Native",
                    "Session 95 (resolves, off by default; LOW sets it with the shadow levers): the ring's shadow map (ps D03E7D4C, 512x256) and the passes that read only "
                    "it (two reductions and a blur) are drawn every other frame - draws and copies of those passes are dropped on the frames between, the pages keep "
                    "last frame's (this map is the wrestlers' RIM-LIGHT map, read by 9 wrestler materials - it lags the pose by one frame; it is not the ring's shadow)");

REXCVAR_DEFINE_BOOL(gpu_native_half_phase, false, "GPU/Native",
                    "Session 95 (r7 contract 3, off by default): one phase for the half-rate passes. The wrestler shadow maps draw on even shadow frames; the light-cube strip, "
                    "the ring chain and the crowd pose passes are skipped only on even shadow frames (never two in a row), so the frames that skip the maps carry the "
                    "others' work (the peak-frame saving is not measured; odd frames may become the new peak - compare both parities). Needs the typed cut (CutPreDrawSkip) and the half-rate "
                    "levers it phases; changes nothing alone for the picture");

namespace rex::graphics::vulkan {
namespace shadow_skip {

namespace {

constexpr uint64_t kReduceInv = 0x901118D1EF082BECull;  // side_blur4_mask_inv
constexpr uint64_t kReduce = 0x0DC910FA641514C0ull;     // side_blur4_mask
constexpr uint64_t kBlur10 = 0x7C650AA1C792DFF6ull;     // side_blur10
constexpr uint64_t kBlur5 = 0x694A6D148DC33CFFull;      // side_blur5

uint64_t g_phase_n[4][2] = {};  // [what][frame parity]: decisions that drew
uint64_t g_phase_s[4][2] = {};  // ... that skipped
bool g_phase_logged = false;

struct Class {
  uint64_t ps;
  uint64_t last_drawn;  // frame + 1 of the last frame a pass of this shader was drawn (0 = never)
};

struct State {
  Class classes[4] = {{kReduceInv, 0}, {kReduce, 0}, {kBlur10, 0}, {kBlur5, 0}};
  std::vector<uint32_t> skipped_keys;  // targets whose draws were skipped and whose copy has not come yet
  uint32_t open_key = 0;               // the pass being decided (draws of one pass share it until its copy)
  bool open = false, open_skip = false;
  uint64_t draws_skipped = 0, draws_drawn = 0, copies_dropped = 0, copies_kept_clear = 0;
  uint64_t miss_size = 0, miss_mixed = 0, miss_prev = 0, miss_no_skip = 0;
  uint64_t frames_engaged = 0;
  bool engaged_this_frame = false, logged = false;
  // Session 95 (resolves): the strips
  uint64_t strip_last_drawn = UINT64_MAX - 1, strip_decided = UINT64_MAX;  // frame of the last drawn strip frame; frame the decision was made for
  bool strip_skip = false, strip_logged = false;
  uint64_t strip_draws_skipped = 0, strip_draws_drawn = 0, strip_frames_skipped = 0;
  // the ring chain
  uint64_t ring_last_drawn = UINT64_MAX - 1, ring_decided = UINT64_MAX, dep_last[3] = {UINT64_MAX - 1, UINT64_MAX - 1, UINT64_MAX - 1};
  bool ring_skip = false, ring_logged = false;
  uint64_t ring_skipped = 0, ring_drawn = 0, ring_dep_skipped = 0, ring_dep_drawn = 0, ring_frames_skipped = 0;
};
State g;

Class* Find(uint64_t ps) {
  for (Class& c : g.classes) {
    if (c.ps == ps) return &c;
  }
  return nullptr;
}

// The chain's targets (base / pitch in 80-pixel tiles / colour format), read in the log of frame 4139 (out\s95\sf0\gpuprof): the wrestler
// maps' reductions 128x128 on base 2032 pitch 2, 32x32 on base 2046 pitch 1, the blurs 512x512 on base 0 pitch 7 format 5 (k_16_16_16_16).
// The second map's chain (reductions 128x64 / 32x16 on 2040/2 and 2047/1) has the same shaders but reads the second map (PS D03E7D4C,
// not half-rate); gpu_native_owned_ring_half skips it (and its chain) on alternate frames.
bool TargetOk(uint64_t ps, uint32_t color_info, uint32_t surface_info) {
  const uint32_t base = color_info & 0xFFF, fmt = (color_info >> 16) & 0xF, tiles = std::max(1u, ((surface_info & 0x3FFF) + 79) / 80);
  if (ps == kReduceInv || ps == kReduce) return (base == 2032 && tiles == 2) || (base == 2046 && tiles == 1);
  return base == 0 && tiles == 7 && fmt == 5;
}

}  // namespace

bool Wanted() { return REXCVAR_GET(gpu_native_owned_shadow_skip); }

bool RingWanted() { return REXCVAR_GET(gpu_native_owned_ring_half); }

bool HalfPhase() { return REXCVAR_GET(gpu_native_half_phase); }

void NotePhase(int what, uint64_t frame, bool drawn) {
  if (!HalfPhase()) return;
  if (!g_phase_logged) {
    g_phase_logged = true;
    REXGPU_INFO("[native][half-phase] engaged (frame {}): shadow maps draw on even shadow frames; strip, ring and pose passes are skipped only on even shadow frames", frame);
  }
  ++(drawn ? g_phase_n : g_phase_s)[what][frame & 1];
}

bool RingPreDraw(uint64_t ps_hash, uint32_t key, uint32_t color_info, uint32_t surface_info, uint64_t frame) {
  constexpr uint64_t kRingPs = 0xD03E7D4C719B0EDBull;    // the 512x256 map pass of the match / Bryan frames (out\s95\sf0\gpuprof\wwe2k14_059.log, frame 10521)
  constexpr uint64_t kRingBlur = 0x632189BAFAE233F0ull;  // its blur, 512x256
  const uint32_t base = color_info & 0xFFF, tiles = std::max(1u, ((surface_info & 0x3FFF) + 79) / 80);
  int dep = -1;
  if (ps_hash == kRingPs) {
    if (base != 0 || tiles != 7) return false;
    if (g.ring_decided != frame) {  // the frame's first ring draw decides for the whole chain
      g.ring_decided = frame;
      g.ring_skip = (g.ring_last_drawn + 1 == frame) && (!HalfPhase() || !(frame & 1));
      NotePhase(2, frame, !g.ring_skip);
      if (g.ring_skip) {
        ++g.ring_frames_skipped;
      } else {
        g.ring_last_drawn = frame;
      }
    }
    if (!g.ring_skip) {
      ++g.ring_drawn;
      return false;
    }
  } else {
    if ((ps_hash == kReduceInv && base == 2040 && tiles == 2)) dep = 0;
    else if (ps_hash == kReduce && base == 2047 && tiles == 1) dep = 1;
    else if (ps_hash == kRingBlur && base == 0 && tiles == 7) dep = 2;
    if (dep < 0) return false;
    if (g.ring_decided != frame || !g.ring_skip || g.dep_last[dep] + 1 != frame) {  // the ring was drawn (or not seen) this frame, or this pass was not drawn the frame before
      g.dep_last[dep] = frame;
      ++g.ring_dep_drawn;
      return false;
    }
  }
  if (std::find(g.skipped_keys.begin(), g.skipped_keys.end(), key) == g.skipped_keys.end()) g.skipped_keys.push_back(key);
  g.engaged_this_frame = true;
  if (dep >= 0) ++g.ring_dep_skipped; else ++g.ring_skipped;
  if (!g.ring_logged) {
    g.ring_logged = true;
    REXGPU_INFO("[native][ring-half] first ring-chain pass skipped (frame {}): ps {:016X} target key {:08X} base {} pitch {}; its copies are dropped, the pages keep last frame's", frame, ps_hash, key, base, tiles);
  }
  return true;
}

bool StripWanted() { return REXCVAR_GET(gpu_native_owned_strip_half); }

bool StripPreDraw(uint64_t ps_hash, uint32_t key, uint32_t color_info, uint32_t surface_info, uint64_t frame) {
  constexpr uint64_t kStripPs = 0x8479D8DA0BB01D6Bull;  // read in Bryan's frame 5690 (out\s95\sf0\gpuprof\wwe2k14_059.log): 5 passes, surface base 0 pitch 5 (384x64)
  if (ps_hash != kStripPs) return false;
  const uint32_t base = color_info & 0xFFF, tiles = std::max(1u, ((surface_info & 0x3FFF) + 79) / 80);
  if (base != 0 || tiles != 5) return false;
  if (g.strip_decided != frame) {  // the frame's first strip draw decides for every strip pass of the frame
    g.strip_decided = frame;
    g.strip_skip = (g.strip_last_drawn + 1 == frame) && (!HalfPhase() || !(frame & 1));  // drawn the frame before: this one is skipped (half-phase: only on even frames)
    NotePhase(1, frame, !g.strip_skip);
    if (g.strip_skip) {
      ++g.strip_frames_skipped;
    } else {
      g.strip_last_drawn = frame;
    }
  }
  if (!g.strip_skip) {
    ++g.strip_draws_drawn;
    return false;
  }
  if (std::find(g.skipped_keys.begin(), g.skipped_keys.end(), key) == g.skipped_keys.end()) g.skipped_keys.push_back(key);
  g.engaged_this_frame = true;
  ++g.strip_draws_skipped;
  if (!g.strip_logged) {
    g.strip_logged = true;
    REXGPU_INFO("[native][strip-half] first strip pass skipped (frame {}): ps {:016X} target key {:08X}; its copies are dropped, the cubes keep last frame's faces", frame, ps_hash, key);
  }
  return true;
}

void NoteDrawn(uint32_t key) {
  if (g.skipped_keys.empty()) return;
  const auto it = std::find(g.skipped_keys.begin(), g.skipped_keys.end(), key);
  if (it != g.skipped_keys.end()) g.skipped_keys.erase(it);
}

void NoteMap(uint32_t key, bool skipped) {
  if (!skipped) {
    NoteDrawn(key);
    return;
  }
  if (std::find(g.skipped_keys.begin(), g.skipped_keys.end(), key) == g.skipped_keys.end()) g.skipped_keys.push_back(key);
}

bool PreDraw(uint64_t ps_hash, uint32_t key, uint32_t color_info, uint32_t surface_info, uint64_t frame, uint32_t maps_drawn,
             uint32_t maps_skipped) {
  Class* c = Find(ps_hash);
  if (!c) return false;
  if (g.open && g.open_key == key) return g.open_skip;  // the pass was decided at its first draw
  // a new pass: decide
  g.open = true;
  g.open_key = key;
  g.open_skip = false;
  if (!TargetOk(ps_hash, color_info, surface_info)) {
    ++g.miss_size;
  } else if (maps_skipped == 0) {
    ++g.miss_no_skip;
  } else if (maps_drawn != 0) {
    ++g.miss_mixed;
  } else if (c->last_drawn != frame) {  // last_drawn = frame + 1 of the last drawn frame: must be the frame before this one
    ++g.miss_prev;
  } else {
    g.open_skip = true;
  }
  if (g.open_skip) {
    if (std::find(g.skipped_keys.begin(), g.skipped_keys.end(), key) == g.skipped_keys.end()) g.skipped_keys.push_back(key);
    g.engaged_this_frame = true;
    ++g.draws_skipped;
    if (!g.logged) {
      g.logged = true;
      REXGPU_INFO("[native][shadow-skip] first dependent pass skipped: ps {:016X} target key {:08X} base {} pitch {} (frame {}; every wrestler map skipped this frame)",
                  ps_hash, key, color_info & 0xFFF, surface_info & 0x3FFF, frame);
    }
    return true;
  }
  c->last_drawn = frame + 1;
  ++g.draws_drawn;
  NoteDrawn(key);
  return false;
}

bool DropCopy(uint32_t key, bool clears) {
  if (g.open && g.open_key == key) g.open = false;  // the pass ends at its copy
  if (std::find(g.skipped_keys.begin(), g.skipped_keys.end(), key) == g.skipped_keys.end()) return false;
  if (clears) {  // a copy that also clears is kept (a later surface may rely on the clear)
    ++g.copies_kept_clear;
    return false;
  }
  ++g.copies_dropped;
  return true;
}

void FrameEnd(uint64_t frame) {
  g.open = false;  // a pass left without its copy: the next frame starts afresh
  g.skipped_keys.clear();
  if (g.engaged_this_frame) ++g.frames_engaged;
  g.engaged_this_frame = false;
  bool phase_any = false;
  for (int w = 0; w < 4; ++w) phase_any |= (g_phase_n[w][0] | g_phase_n[w][1] | g_phase_s[w][0] | g_phase_s[w][1]) != 0;
  if (frame % 600 == 0 && HalfPhase() && phase_any) {  // (no all-zero windows: a line then proves nothing)
    static const char* kName[4] = {"shadow", "strip", "ring", "pose"};
    for (int w = 0; w < 4; ++w) {
      REXGPU_INFO("[native][half-phase] frames to {}: {} drawn even {} / odd {}, skipped even {} / odd {}", frame, kName[w], g_phase_n[w][0], g_phase_n[w][1],
                  g_phase_s[w][0], g_phase_s[w][1]);
    }
    for (auto& a : g_phase_n) a[0] = a[1] = 0;
    for (auto& a : g_phase_s) a[0] = a[1] = 0;
  }
  if (frame % 600 == 0 && (g.ring_skipped || g.ring_drawn || g.ring_dep_drawn)) {
    REXGPU_INFO("[native][ring-half] frames to {}: ring map draws skipped {} / drawn {}, dependent passes skipped {} / drawn {}, frames skipped {}", frame, g.ring_skipped,
                g.ring_drawn, g.ring_dep_skipped, g.ring_dep_drawn, g.ring_frames_skipped);
    g.ring_skipped = g.ring_drawn = g.ring_dep_skipped = g.ring_dep_drawn = g.ring_frames_skipped = 0;
  }
  if (frame % 600 == 0 && (g.strip_draws_skipped || g.strip_draws_drawn)) {
    REXGPU_INFO("[native][strip-half] frames to {}: strip draws skipped {} / drawn {}, frames skipped {}; copies dropped so far {}", frame, g.strip_draws_skipped,
                g.strip_draws_drawn, g.strip_frames_skipped, g.copies_dropped);
    g.strip_draws_skipped = g.strip_draws_drawn = g.strip_frames_skipped = 0;
  }
  if (frame % 600 == 0 && (g.draws_skipped || g.draws_drawn || g.miss_size || g.miss_no_skip || g.miss_mixed || g.miss_prev)) {
    REXGPU_INFO("[native][shadow-skip] frames to {}: engaged frames {}; passes skipped {} / drawn {}; copies dropped {}, kept with a clear {}; "
                "drawn because: other target {}, no map skipped {}, mixed frame {}, not drawn the frame before {}",
                frame, g.frames_engaged, g.draws_skipped, g.draws_drawn, g.copies_dropped, g.copies_kept_clear, g.miss_size, g.miss_no_skip,
                g.miss_mixed, g.miss_prev);
    g.draws_skipped = g.draws_drawn = g.copies_dropped = g.copies_kept_clear = 0;
    g.miss_size = g.miss_no_skip = g.miss_mixed = g.miss_prev = 0;
    g.frames_engaged = 0;
  }
}

}  // namespace shadow_skip
}  // namespace rex::graphics::vulkan
