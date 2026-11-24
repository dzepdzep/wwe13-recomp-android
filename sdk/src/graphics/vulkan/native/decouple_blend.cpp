// wwe2k14 patch (Session 94, idea 1 phase 1B): the draw match across two guest ticks and the blended vertex constants of view 1 (the in-between; view 0 is the tick's own picture). See decouple_blend.h.
//
// Per typed draw (command thread), while the blend is active (gpu_d3d_decouple bit 3 and the game's frame word = gpu_d3d_decouple_word):
//   match     only draws of OUR vertex shaders are matched: key = vertex / pixel shader hash, the pass (colour 0 / depth / surface words), the dword 0 of the vertex fetch constants the
//             shader reads, the index DMA base and count (the census' key, phase0_census.cpp). The previous tick's draws with the same key are the candidates: one = it, several =
//             the one whose blendable registers are nearest.
//   camera    c56-c59 of the draw's bank, hashed by value, only in our shaders that read it as a matrix. The first matched draw carrying a block decides it: lerp(previous, this, 0.5)
//             in EVERY such draw carrying that value, unless the camera moved more than gpu_d3d_decouple_cam_guard (a cut: no blend). An unmatched draw does not decide.
//   matrices  the world rows c76-c78 or the bone palette c76-c161 (shader_swap::VertexBlendRegisters), per matrix (3 rows), lerped with the matched draw's, unless the rotation part
//             (x y z of the rows) moved more than gpu_d3d_decouple_rot_guard or the translation (w) more than gpu_d3d_decouple_move_guard (a cut or a snap: that matrix stays as in this tick).
//   others    nothing: an unmatched or translated draw uses this tick's constants in both views.
// What was blended is returned as a copy of the draw's bank (view 1's); the caller (D3dDirectAccess::TypedDraw) hands it to the command processor, which writes it as
// block 1 of the draw's vertex float buffer (UpdateBindings; block 0 = the draw's own bank = view 0 = today's picture).
#include "decouple_blend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/vulkan/shader.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>

#include "owned_record.h"
#include "shader_swap.h"

REXCVAR_DECLARE(int32_t, gpu_d3d_decouple);

REXCVAR_DEFINE_INT32(gpu_d3d_decouple_word, 30, "GPU/Native",
                     "Session 94 (gpu_d3d_decouple phase 1B): the blend and the two-step present run only while the game's frame-rate word (guest 0x8392DC40: 60, or 30 in "
                     "classic entrances; the address is the project's, src/frame_word.h) equals this: 30 = classic entrances shown at 60 (default, mode 1). 0 = always "
                     "(also matches: 60 simulation ticks shown as 120 on a 120 Hz screen). Read live");
REXCVAR_DEFINE_DOUBLE(gpu_d3d_decouple_cam_guard, 0.25, "GPU/Native",
                      "Session 94 (gpu_d3d_decouple phase 1B): a camera block (c56-c59) whose largest element moved more than this fraction of the block's largest element "
                      "since the previous tick is a camera cut: no blend (view 1 = view 0). Read live");
REXCVAR_DEFINE_DOUBLE(gpu_d3d_decouple_rot_guard, 0.30, "GPU/Native",
                      "Session 94 (gpu_d3d_decouple phase 1B): a world / bone matrix of one of our vertex shaders whose rotation part (x y z of its 3 rows) moved more than "
                      "this per element since the previous tick (~17 degrees) is not blended. Read live");
REXCVAR_DEFINE_DOUBLE(gpu_d3d_decouple_move_guard, 64.0, "GPU/Native",
                      "Session 94 (gpu_d3d_decouple phase 1B): a world / bone matrix whose translation (w of its 3 rows) moved more than this (world units a tick; they look like cm, src/cinecam.h: "
                      "unconfirmed) since the previous tick is not blended. Session 95: 2.0 -> 64: 2 cm a tick is slower than walking at 30 ticks a second, so the guard cut every moving bone (run 6588: "
                      "cut 90042 of ours 186648, 90012 of them by move); 64 = about 19 m/s at 30 ticks, faster than a limb, and the window line's group moves show what is really there. Read live");

REXCVAR_DEFINE_DOUBLE(gpu_d3d_decouple_wild_scale, 8.0, "GPU/Native",
                      "Session 95 (F5): a matrix group that moved more than this times gpu_d3d_decouple_move_guard (or rotated more than 1.0 an element, or is not finite) is a teleport or a stale "
                      "palette slot, not a bone: it is held at this tick's value and does not veto the draw's blend (a group between the guards does). Read live");

namespace rex::graphics::vulkan {

namespace {

constexpr uint32_t kCamReg = 56;      // c56..c59: the view-projection
constexpr uint32_t kMatrixReg = 76;   // c76..: the world rows / the bone palette
constexpr float kMoveBins[8] = {2.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f, 512.0f};  // (the last bin of State::move_bin: 512 and over)
constexpr uint32_t kFrameWordAddress = 0x8392DC40u;  // the game's frame-rate word (src/frame_word.h); big endian in guest memory

inline uint64_t Mix(uint64_t a, uint64_t b) {
  a ^= b + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2);
  return a;
}
inline float F(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
inline uint32_t Bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
}
inline uint64_t BlockHash(const uint32_t* w, uint32_t count) {
  uint64_t h = 0x51ED270Bull;
  for (uint32_t i = 0; i < count; ++i) h = Mix(h, w[i]);
  return h;
}

struct VsInfo {
  bool ours = false;
  bool cam = false;           // the shader's register mask includes c56-c59 (it reads the camera block as a matrix)
  std::vector<uint8_t> regs;  // the registers blended per draw (the shader's mask without c56-c59), ascending
};
struct OursRec {
  uint64_t key = 0;
  uint32_t offset = 0;  // into the tick's blob (4 words a register of VsInfo::regs)
  uint32_t count = 0;   // registers
  bool has_cam = false;
  uint32_t cam[16];
};
struct CamEntry {
  uint32_t blend[16];
  bool blended = false;
  bool decided = false;  // blended, or a cut: no later draw tries again (an unmatched first draw does not decide)
};

struct State {
  // the tick's tables
  std::vector<OursRec> cur_recs, prev_recs;
  std::unordered_map<uint64_t, std::vector<uint32_t>> cur_by_key, prev_by_key;  // key -> indices into the recs
  std::vector<uint32_t> cur_blob, prev_blob;
  std::unordered_map<uint64_t, CamEntry> cam_cur;  // camera block value -> its blend this tick
  std::unordered_map<uint64_t, VsInfo> vs_info;
  uint64_t info_generation = 0;
  // the gate
  bool tick_active = false;  // the tick in progress is blended
  bool tick_decided = false;
  bool prev_valid = false;   // the previous tick was blended (its tables are real)
  bool last_tick_active = false;
  // counters (window)
  uint64_t ticks = 0, active_ticks = 0, draws = 0, ours = 0, ours_blended = 0, ours_unmatched = 0, cuts = 0, cuts_rot = 0, cuts_move = 0, wild_groups = 0, crowd_cam_blended = 0, crowd_cam_other = 0, cam_blocks = 0, cam_blocks_blended = 0, cam_cuts = 0,
           cam_replaced = 0, cam_unmatched = 0, returned = 0;
  uint64_t cut_group[15] = {};  // (F5) the group index (bone slot, c76 + 3 x index; 14 = 14 and above) of each draw-level veto
  uint64_t move_bin[9] = {};    // (Session 95, F2) matrix groups by the translation they moved this tick (kMoveBins): what the move guard is tuned from
  uint64_t window_start_ticks = 0;
  std::array<uint32_t, 1024> scratch;
};
State g;

bool FrameWordActive(rex::memory::Memory* memory) {
  const int32_t want = REXCVAR_GET(gpu_d3d_decouple_word);
  if (want <= 0) return true;
  if (!memory) return true;
  const uint32_t* p = memory->TranslateVirtual<const uint32_t*>(kFrameWordAddress);
  if (!p) return true;
  const uint32_t word = __builtin_bswap32(*p);
  return int32_t(word) == want;
}

const VsInfo& InfoFor(uint64_t vs_hash) {
  const uint64_t generation = shader_swap::TranslationGeneration();
  if (generation != g.info_generation) {  // (a registration or a skip list changed: ask again)
    g.vs_info.clear();
    g.info_generation = generation;
  }
  auto it = g.vs_info.find(vs_hash);
  if (it != g.vs_info.end()) return it->second;
  VsInfo info;
  uint64_t mask[4];
  if (shader_swap::VertexBlendRegisters(vs_hash, mask)) {
    for (uint32_t r = 0; r < 256; ++r) {
      if ((mask[r >> 6] >> (r & 63)) & 1) {
        if (r >= kCamReg && r < kCamReg + 4) {
          info.cam = true;
          continue;
        }
        info.regs.push_back(uint8_t(r));
      }
    }
    info.ours = true;
  }
  return g.vs_info.emplace(vs_hash, std::move(info)).first->second;
}

}  // namespace

const uint32_t* DecoupleBlendDraw(rex::memory::Memory* memory, const OwnedRecordState& s, const VulkanShader* vs, bool crowd_member) {
  if (!(REXCVAR_GET(gpu_d3d_decouple) & 8) || s.inline_draw || !s.banks_valid) return nullptr;
  if (!g.tick_decided) {
    g.tick_decided = true;
    g.tick_active = FrameWordActive(memory);
  }
  if (!g.tick_active) return nullptr;
  ++g.draws;
  const uint64_t vs_hash = s.shader[0].hash, ps_hash = s.shader[1].hash;
  const VsInfo& info = InfoFor(vs_hash);
  if (!info.ours || crowd_member) return nullptr;  // (translated draws and crowd members: this tick's constants in both views; only OUR shaders are matched)
  ++g.ours;
  // ---- the key
  uint64_t key = Mix(Mix(vs_hash, ps_hash), Mix(Mix(s.color_info[0].value, s.depth_info.value), s.surface_info.value));
  if (vs) {
    for (const auto& binding : vs->vertex_bindings()) {
      if (binding.fetch_constant >= 96) continue;
      key = Mix(key, s.fetch[binding.fetch_constant * 2]);
    }
  }
  key = Mix(Mix(key, s.dma_base), s.draw_initiator.num_indices);

  // ---- this draw's record
  OursRec rec;
  rec.key = key;
  rec.offset = uint32_t(g.cur_blob.size());
  rec.count = uint32_t(info.regs.size());
  for (uint8_t r : info.regs) g.cur_blob.insert(g.cur_blob.end(), s.vs_float + 4 * r, s.vs_float + 4 * r + 4);
  if (info.cam) {
    rec.has_cam = true;
    std::memcpy(rec.cam, s.vs_float + 4 * kCamReg, sizeof(rec.cam));
  }
  const uint32_t* const cur_regs = g.cur_blob.data() + rec.offset;

  // ---- the match: the draws of the previous tick with the same key; one = it, several = the one whose registers are nearest (a second identical mesh may have changed place)
  const OursRec* prev = nullptr;
  const uint32_t* prev_regs = nullptr;
  if (g.prev_valid) {
    const auto it = g.prev_by_key.find(key);
    if (it != g.prev_by_key.end()) {
      float best = INFINITY;
      for (uint32_t index : it->second) {
        const OursRec& cand = g.prev_recs[index];
        if (cand.count != rec.count) continue;
        float distance = 0.0f;
        const uint32_t* cand_regs = g.prev_blob.data() + cand.offset;
        if (it->second.size() > 1) {
          for (uint32_t i = 0; i < rec.count * 4; ++i) distance += std::fabs(F(cand_regs[i]) - F(cur_regs[i]));
          if (cand.has_cam && rec.has_cam) {
            for (uint32_t i = 0; i < 16; ++i) distance += std::fabs(F(cand.cam[i]) - F(rec.cam[i]));
          }
        }
        if (distance < best) {
          best = distance;
          prev = &cand;
          prev_regs = cand_regs;
        }
      }
    }
  }
  const uint32_t rec_index = uint32_t(g.cur_recs.size());
  g.cur_recs.push_back(rec);
  g.cur_by_key[key].push_back(rec_index);
  if (!prev) ++g.ours_unmatched;

  bool changed = false;
  const CamEntry* cam = nullptr;
  // ---- the camera, by value; replaced only in our shaders that read it as a matrix
  if (info.cam) {
    const uint64_t ch = BlockHash(rec.cam, 16);
    auto cam_it = g.cam_cur.find(ch);
    if (cam_it == g.cam_cur.end()) {
      CamEntry fresh;
      std::memcpy(fresh.blend, rec.cam, sizeof(fresh.blend));
      ++g.cam_blocks;
      cam_it = g.cam_cur.emplace(ch, fresh).first;
    }
    CamEntry& entry = cam_it->second;
    if (!entry.decided) {
      bool zero = true;
      for (uint32_t i = 0; i < 16; ++i) zero &= rec.cam[i] == 0;
      if (zero) {
        entry.decided = true;
      } else if (prev && prev->has_cam) {
        const uint32_t* a = prev->cam;
        float max_cur = 0.0f, max_delta = 0.0f;
        bool finite = true;
        for (uint32_t i = 0; i < 16; ++i) {
          const float x = F(a[i]), y = F(rec.cam[i]);
          finite &= std::isfinite(x) && std::isfinite(y);
          max_cur = std::max(max_cur, std::fabs(y));
          max_delta = std::max(max_delta, std::fabs(y - x));
        }
        entry.decided = true;
        if (finite && max_delta <= REXCVAR_GET(gpu_d3d_decouple_cam_guard) * std::max(max_cur, 1e-6f)) {
          for (uint32_t i = 0; i < 16; ++i) entry.blend[i] = Bits(0.5f * (F(a[i]) + F(rec.cam[i])));
          entry.blended = max_delta > 0.0f;
          if (entry.blended) ++g.cam_blocks_blended;
        } else {
          ++g.cam_cuts;
        }
      } else {
        ++g.cam_unmatched;  // (not decided: a later draw carrying the same block with a partner may still blend it)
      }
    }
    cam = &entry;
    changed |= entry.blended;
  }
  if (!changed && !prev) return nullptr;

  uint32_t* out = g.scratch.data();
  std::memcpy(out, s.vs_float, 1024 * sizeof(uint32_t));
  if (cam && cam->blended) {
    std::memcpy(out + 4 * kCamReg, cam->blend, sizeof(cam->blend));
    ++g.cam_replaced;
  }
  if (prev && rec.count) {
    const double rot_guard = REXCVAR_GET(gpu_d3d_decouple_rot_guard), move_guard = REXCVAR_GET(gpu_d3d_decouple_move_guard);
    const uint32_t* a = prev_regs;
    const uint32_t* b = cur_regs;
    // Session 95 (F5): a draw blends ALL its matrix groups or none (a cut bone next to lerped neighbours would tear one skinned mesh). Pass 0 decides (every group is looked at, so the
    // counters see them all), pass 1 writes. A skinned shader's mask is the whole palette c76-c161, and a slot the mesh never reaches may hold stale data that differs wildly from tick to tick:
    // a group that moved a "wild" amount (not finite, rotation over 1.0 = beyond 60 degrees an element, translation over gpu_d3d_decouple_wild_scale x the move guard: a teleport, not a
    // bone) does not veto the draw; it is held at this tick's value, the same answer in every draw that carries that value (a wrestler's skin / gear / hair agree). A group between the guard
    // and "wild" is a real cut and vetoes the draw. The group index of each veto is counted (the summary line: cut groups).
    const double wild_move = move_guard * REXCVAR_GET(gpu_d3d_decouple_wild_scale);
    bool any = false, cut = false, cut_rot = false, cut_move = false;
    for (int pass = 0; pass < 2 && !cut; ++pass) {
      for (uint32_t i = 0; i < rec.count;) {
        uint32_t group = 1;
        const uint8_t first = info.regs[i];
        if (first >= kMatrixReg) {
          const uint32_t base = kMatrixReg + ((first - kMatrixReg) / 3) * 3;
          group = 0;
          while (i + group < rec.count && info.regs[i + group] >= base && info.regs[i + group] < base + 3) ++group;
        }
        float rot = 0.0f, move = 0.0f;
        bool finite = true, differs = false;
        for (uint32_t g_i = 0; g_i < group; ++g_i) {
          for (uint32_t c = 0; c < 4; ++c) {
            const float x = F(a[(i + g_i) * 4 + c]), y = F(b[(i + g_i) * 4 + c]);
            finite &= std::isfinite(x) && std::isfinite(y);
            differs |= x != y;
            (c < 3 ? rot : move) = std::max(c < 3 ? rot : move, std::fabs(y - x));
          }
        }
        const bool wild = !finite || rot > 1.0f || move > wild_move;
        const bool ok = finite && rot <= rot_guard && move <= move_guard;
        if (pass == 0) {
          if (finite && first >= kMatrixReg) {
            uint32_t bin = 0;
            while (bin < 8 && move >= kMoveBins[bin]) ++bin;
            ++g.move_bin[bin];
          }
          if (wild) {
            ++g.wild_groups;
          } else if (!ok) {
            cut = true;
            cut_rot |= rot > rot_guard;
            cut_move |= move > move_guard;
            const uint32_t group_index = first >= kMatrixReg ? (first - kMatrixReg) / 3 : 15;
            ++g.cut_group[group_index < 15 ? group_index : 14];
          }
        } else if (!wild) {
          for (uint32_t g_i = 0; g_i < group; ++g_i) {
            for (uint32_t c = 0; c < 4; ++c) {
              const float x = F(a[(i + g_i) * 4 + c]), y = F(b[(i + g_i) * 4 + c]);
              out[4 * info.regs[i + g_i] + c] = Bits(0.5f * (x + y));
            }
          }
          any |= differs;
        }
        i += group;
      }
    }
    if (cut) {
      ++g.cuts;
      g.cuts_rot += cut_rot;
      g.cuts_move += cut_move;
    }
    if (any) {
      ++g.ours_blended;
      changed = true;
    }
  }
  if (!changed) return nullptr;
  ++g.returned;
  return out;
}

bool DecoupleBlendCamera(const uint32_t* vs_float, float out[16]) {
  if (!(REXCVAR_GET(gpu_d3d_decouple) & 8) || !g.tick_decided || !g.tick_active) return false;
  const uint64_t ch = BlockHash(vs_float + 4 * kCamReg, 16);
  const auto it = g.cam_cur.find(ch);
  if (it == g.cam_cur.end() || !it->second.blended) {
    ++g.crowd_cam_other;
    return false;
  }
  std::memcpy(out, it->second.blend, sizeof(it->second.blend));
  ++g.crowd_cam_blended;
  return true;
}

void DecoupleBlendFrameEnd() {
  if (!(REXCVAR_GET(gpu_d3d_decouple) & 8)) return;
  ++g.ticks;
  g.last_tick_active = g.tick_decided && g.tick_active;
  if (g.last_tick_active) ++g.active_ticks;
  // this tick's tables become the previous tick's (only a blended tick recorded any; after an idle tick the previous tables are not real)
  g.prev_valid = g.last_tick_active;
  std::swap(g.prev_recs, g.cur_recs);
  std::swap(g.prev_by_key, g.cur_by_key);
  std::swap(g.prev_blob, g.cur_blob);
  g.cur_recs.clear();
  g.cur_by_key.clear();
  g.cur_blob.clear();
  g.cam_cur.clear();
  g.tick_decided = false;
  g.tick_active = false;
}

bool DecoupleBlendTickWasActive() { return g.last_tick_active; }

std::string DecoupleBlendSummary(uint64_t frames, uint32_t views, uint64_t presents, uint64_t two_step_ticks) {
  if (!(REXCVAR_GET(gpu_d3d_decouple) & 8)) return std::string();
  if (g.ticks - g.window_start_ticks < 600) return std::string();
  const uint64_t ticks = g.ticks - g.window_start_ticks;
  std::string cut_groups;
  for (uint32_t i = 0; i < 15; ++i) {
    if (g.cut_group[i]) cut_groups += fmt::format("{}{}:{}", cut_groups.empty() ? "" : " ", i, g.cut_group[i]);
  }
  if (cut_groups.empty()) cut_groups = "-";
  std::string move_bins;
  for (uint32_t i = 0; i < 9; ++i) {
    move_bins += i < 8 ? fmt::format("{}<{}:{}", i ? " " : "", kMoveBins[i], g.move_bin[i]) : fmt::format(" over:{}", g.move_bin[i]);
  }
  std::string line = fmt::format(
      "decouple: frames {}, views {}, ticks {} (blended {}), blended draws {} of ours {} (unmatched {}, cut {} [rotation {}, move {}; wild groups held {}; cut groups {}]; crowd camera blended {} (identity {}); draws given a view 1 bank {} of {}), camera blocks {} (blended {}, cut {}, "
      "unmatched {}), camera blocks replaced {} draws, presents per tick {:.2f}, group moves a tick {}",
      frames, views, ticks, g.active_ticks, g.ours_blended, g.ours, g.ours_unmatched, g.cuts, g.cuts_rot, g.cuts_move, g.wild_groups, cut_groups, g.crowd_cam_blended, g.crowd_cam_other, g.returned, g.draws, g.cam_blocks, g.cam_blocks_blended, g.cam_cuts, g.cam_unmatched,
      g.cam_replaced, ticks ? double(presents) / double(ticks) : 0.0, move_bins);
  (void)two_step_ticks;
  g.window_start_ticks = g.ticks;
  std::fill(std::begin(g.cut_group), std::end(g.cut_group), uint64_t(0));
  std::fill(std::begin(g.move_bin), std::end(g.move_bin), uint64_t(0));
  g.active_ticks = g.draws = g.ours = g.ours_blended = g.ours_unmatched = g.cuts = g.cuts_rot = g.cuts_move = g.wild_groups = g.crowd_cam_blended = g.crowd_cam_other = g.cam_blocks = g.cam_blocks_blended = g.cam_cuts = g.cam_replaced = g.cam_unmatched = g.returned = 0;
  return line;
}

}  // namespace rex::graphics::vulkan
