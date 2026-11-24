// wwe2k14 patch (Session 95, item N2): the scout, the job bodies and the result slots. See owned_scout.h for the mechanism; out\s95\read_N.md section 3.
//
// Threads: the command thread (CP) calls DrainEnter / BeginItem / NotePresent / DrawItem / Lookup; ONE scout thread ("GPU Jobs scout") walks the queue ahead;
// the pool's workers run JobBody. What each owns:
//   CP:     Cp (plain counters), the pair table's writes, the sync package's writes (only while pkg_ready is false).
//   scout:  its private OwnedRecordState, position, snapshot ring claims, batch; writes the slot's `pending` key.
//   jobs:   a snapshot (read), the slot's payload + final key (write), the snapshot's busy flag (clear).
// A draw's key = bit 63 | the queue item's start position (dwords, u32) << 16 | the draw's ordinal in the queue item. Slot index = hash(key).
#include "owned_scout.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define REX_SCOUT_PAUSE() _mm_pause()
#else
#define REX_SCOUT_PAUSE() std::this_thread::yield()
#endif
#include <mutex>
#include <string>
#include <cstdio>
#include <thread>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/d3d_noring.h>
#include <rex/graphics/free_counters.h>
#include <rex/graphics/job_pool.h>
#include <rex/graphics/pipeline/shader/interpreter.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/hash.h>
#include <rex/logging.h>
#include <rex/thread.h>

#include "owned_record.h"

REXCVAR_DECLARE(bool, gpu_interp_memo);
REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95)

REXCVAR_DEFINE_INT32(gpu_jobs, 0, "GPU/Native",
                     "Session 95 (item N, the scalable job system), bits: 1 = the job pool (workers sized from the process affinity mask: logical processors - 6, 0..14), "
                     "2 = the scout + pure jobs (cover / bounds rectangles computed ahead of the command thread; the command thread uses a result only when its key and "
                     "input fingerprint match and never waits). 0 = nothing runs, no thread is started. Needs the cut (TRUE NATIVE). [jobs] lines.");
REXCVAR_DEFINE_INT32(gpu_jobs_workers, 0, "GPU/Native",
                     "Session 95: the number of ACTIVE pool workers (live: workers above it park). 0 = from the process affinity mask. A 2-core / 4-thread machine gets 0 workers: "
                     "the scout is off and everything runs inline. Negative = 0 workers (the W=0 arm of a scaling curve).");
REXCVAR_DEFINE_BOOL(gpu_jobs_verify, false, "GPU/Native",
                    "Session 95: with gpu_jobs bit 2, the command thread computes every cover / bounds rectangle inline as well and compares it with the job's (the inline "
                    "result is used). Mismatches are counted and the first 8 logged; 0 are required.");

namespace rex::graphics::vulkan::scout {

bool g_on = false;
static bool g_cp_active_plain = false;  // command thread: DrainEnter / a mid-drain switch-on set it, DrainLeave clears it (also when g_on went off meanwhile)

namespace {

namespace jb = rex::graphics::jobs;

constexpr uint64_t kKeyBit = 1ull << 63;
constexpr uint64_t kPendingBit = 1ull << 62;
constexpr uint32_t kSlotCount = 8192;
constexpr uint32_t kSnapCount = 256;
constexpr uint32_t kPairCount = 4096;
constexpr uint32_t kResyncPresents = 120;
constexpr uint32_t kWindowFrames = 600;

// ---- slots
enum : uint32_t {
  kWFlags, kWCoverLine, kWCover0, kWBounds0 = kWCover0 + 4, kWFpLo = kWBounds0 + 4, kWFpHi, kWEpoch, kWCount
};
constexpr uint32_t kFCoverDone = 1, kFCoverOk = 2, kFBoundsDone = 4, kFBoundsOk = 8;

struct Slot {
  std::atomic<uint64_t> key{0};
  std::atomic<uint32_t> w[kWCount];
};

// ---- snapshots: a copy of the typed state a job works from
struct Snap {
  OwnedRecordState st;
  std::atomic<uint32_t> busy{0};
  uint64_t key = 0;
  const Shader* vs = nullptr;
  const Shader* ps = nullptr;
  uint32_t want = 0;   // 1 cover, 2 bounds, 4 owned-step cover
  uint32_t facts = 0;  // the Pair.flags facts: 8 pixel shader kills, 16 vertex shader interpretable, 32 vertex shader memexport, 64 interpreter memo on
};

// ---- the shader pair table: written by the command thread (the claim's own tests), read by the scout
struct Pair {
  std::atomic<uint64_t> vs{0};
  std::atomic<uint64_t> ps{0};
  std::atomic<const Shader*> vsp{nullptr};
  std::atomic<const Shader*> psp{nullptr};
  std::atomic<uint32_t> flags{0};  // 1 owned-step cover, 2 truth wants cover, 4 bounds vertex shader, 8 ps kills pixels, 16 vs interpretable, 32 vs memexport, 64 memo on (read by the command thread's Note, copied so a job touches no Shader state but its immutable microcode and maps)
};

inline uint32_t PairIndex(uint64_t vs, uint64_t ps) {
  const uint64_t h = (vs * 0x9E3779B97F4A7C15ull) ^ (ps * 0xC2B2AE3D27D4EB4Full);
  return uint32_t(h >> 40) & (kPairCount - 1);
}
inline uint32_t SlotIndex(uint64_t key) { return uint32_t((key * 0x9E3779B97F4A7C15ull) >> 40) & (kSlotCount - 1); }

struct CpStats {
  uint64_t used = 0, late = 0, empty = 0, torn = 0, stale = 0, no_key = 0;
  uint64_t cover_used = 0, bounds_used = 0, verify_checked = 0, verify_mismatch = 0;
  uint64_t la_items = 0, la_sum = 0, la_min_frame = UINT64_MAX;  // lookahead dwords: this frame
  uint64_t la_frames = 0, la_avg_sum = 0, la_min_sum = 0;         // ... summed over the window's frames
  uint64_t lai_sum = 0, lai_min_frame = UINT64_MAX, lai_avg_sum = 0, lai_min_sum = 0;  // lookahead items
  uint64_t frame_first = 0;
};

struct Global {
  std::atomic<bool> running{false};  // the scout may work
  bool verify = false;               // command thread
  std::atomic<D3dNoringShared*> queue{nullptr};
  std::atomic<rex::memory::Memory*> memory{nullptr};
  std::atomic<bool> cp_active{false};
  std::atomic<uint32_t> guest_epoch{0};  // bumped by the command thread when guest memory may have changed under a job
  std::atomic<uint32_t> cp_seq{0};      // packet / register items the command thread has begun
  std::atomic<uint32_t> scout_seq{0};   // ... the scout has applied (rebased at a sync)
  // the sync package
  std::atomic<bool> need_sync{true};
  std::atomic<bool> pkg_ready{false};
  uint32_t pkg_pos = 0, pkg_seq = 0;
  std::unique_ptr<OwnedRecordState> pkg_state;
  // tables
  std::unique_ptr<Slot[]> slots;
  std::unique_ptr<Snap[]> snaps;
  std::unique_ptr<Pair[]> pairs;
  // scout thread parking
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> parked{false};
  std::atomic<bool> thread_made{false};
  // scout-side counters (relaxed)
  std::atomic<uint64_t> s_items{0}, s_draws{0}, s_jobs{0}, s_batch_items{0}, s_resyncs{0}, s_desync{0}, s_behind{0}, s_unknown{0}, s_notwanted{0}, s_ring_full{0},
      s_inline_runs{0}, s_pending_skip{0};
  std::atomic<uint64_t> j_cover{0}, j_bounds{0}, j_cover_ok{0}, j_bounds_ok{0};
  CpStats cp;
  // the current draw (command thread)
  uint32_t item_pos = 0, draw_ord = 0;
  uint64_t cur_key = 0;
  bool announced = false;
  uint32_t last_active = UINT32_MAX;
  bool last_scout = false;
  uint64_t frames = 0;
};
Global& G() {
  static Global* g = [] {
    Global* x = new Global();  // (never destroyed: the scout and the workers may still run at exit)
    x->pkg_state.reset(new OwnedRecordState());
    x->slots.reset(new Slot[kSlotCount]);
    for (uint32_t i = 0; i < kSlotCount; ++i)
      for (uint32_t k = 0; k < kWCount; ++k) x->slots[i].w[k].store(0, std::memory_order_relaxed);
    x->snaps.reset(new Snap[kSnapCount]);
    x->pairs.reset(new Pair[kPairCount]);
    return x;
  }();
  return *g;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The inputs of a cover / bounds computation from a typed state: the same fields as OwnedFrontCoverInputs (owned_front_draw.cpp), without its counter.
void MakeInputs(const OwnedRecordState& s, CoverInputs& in) {
  in.interp_regs = nullptr;
  in.initiator = s.draw_initiator;
  in.sc_mode = s.sc_mode_cntl;
  in.reset_indx = s.reset_indx;
  in.surface_info = s.surface_info;
  in.color_control = s.color_control;
  in.window_offset = s.window_offset;
  in.vte = s.vte_cntl;
  in.clip = s.clip_cntl;
  in.su_vtx_cntl = s.su_vtx_cntl;
  in.indx_offset = s.indx_offset;
  in.min_indx = s.min_vtx_indx;
  in.max_indx = s.max_vtx_indx;
  in.vport[0] = s.vport[0];
  in.vport[1] = s.vport[1];
  in.vport[2] = s.vport[2];
  in.vport[3] = s.vport[3];
  draw_util::GetScissor(s.window_scissor_tl, s.window_scissor_br, s.window_offset, s.screen_scissor_tl, s.screen_scissor_br, s.surface_info.surface_pitch,
                        in.scissor);
  in.vs_const = s.vs_const.value;
  in.floats = s.vs_float;
  in.bool_loop = s.bool_loop;
  in.fetch = s.fetch;
}

// What a result depends on, hashed: the draw's words, the shaders, the constants the vertex shader reads (its float bitmap rows, the bool / loop words) and the
// vertex fetch constants it names. NOT the bytes of guest memory (see the hazard note in the hand-back).
uint64_t Fingerprint(const OwnedRecordState& s, const CoverInputs& in, const Shader& vs, const Shader* ps) {
  uint32_t words[48 + 1 + 4 * 512 + 40 + 192];
  uint32_t n = 0;
  const auto put = [&](uint32_t v) { words[n++] = v; };
  put(in.initiator.value), put(in.sc_mode.value), put(in.reset_indx.value), put(in.surface_info.value), put(in.color_control.value);
  put(in.window_offset.value), put(in.vte.value), put(in.clip.value), put(in.su_vtx_cntl.value), put(in.indx_offset.value);
  put(in.min_indx.value), put(in.max_indx.value);
  for (uint32_t i = 0; i < 4; ++i) put(std::bit_cast<uint32_t>(in.vport[i]));
  put(in.scissor.offset[0]), put(in.scissor.offset[1]), put(in.scissor.extent[0]), put(in.scissor.extent[1]);
  put(in.vs_const), put(s.dma_base), put(s.dma_size.value);
  const uint64_t vh = vs.ucode_data_hash(), ph = ps ? ps->ucode_data_hash() : 0;
  put(uint32_t(vh)), put(uint32_t(vh >> 32)), put(uint32_t(ph)), put(uint32_t(ph >> 32));
  const Shader::ConstantRegisterMap& map = vs.constant_register_map();
  if (map.float_dynamic_addressing) {
    std::memcpy(words + n, in.floats, sizeof(uint32_t) * 4 * 512);
    n += 4 * 512;
  } else {
    reg::SQ_VS_CONST base{};
    base.value = in.vs_const;
    for (uint32_t i = 0; i < 4; ++i) {
      uint64_t entry = map.float_bitmap[i];
      while (entry) {
        const uint32_t index = uint32_t(std::countr_zero(entry));
        entry &= entry - 1;
        const uint32_t reg_index = (i << 6) + index + base.base;
        if (reg_index >= 512) continue;
        std::memcpy(words + n, in.floats + 4 * size_t(reg_index), sizeof(uint32_t) * 4);
        n += 4;
      }
    }
  }
  std::memcpy(words + n, in.bool_loop, sizeof(uint32_t) * 40);
  n += 40;
  for (uint32_t i = 0; i < 96; ++i) {
    if (!(map.vertex_fetch_bitmap[i >> 5] & (uint32_t(1) << (i & 31)))) continue;
    if (n + 2 > sizeof(words) / sizeof(words[0])) break;
    words[n++] = in.fetch[2 * i];
    words[n++] = in.fetch[2 * i + 1];
  }
  return XXH3_64bits(words, size_t(n) * sizeof(uint32_t)) | 1;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The job bodies: NativeCoverRect / NativeBoundsRect (vulkan\command_processor.cpp), typed branch, on a worker. Same arithmetic and the same refusals; differences:
// no interpreter memo and no bounds cache (a fresh interpretation is what a memo hit stands for), no counters, no logs, and a refusal is a code (the
// command thread replays it into NativeCoverReject so its log line still appears). gpu_jobs_verify compares the two.
struct IndexInfo {
  uint32_t guest_base = 0, count = 0;
  xenos::Endian endianness = xenos::Endian::kNone;
  xenos::IndexFormat format = xenos::IndexFormat::kInt16;
};

IndexInfo IndexInfoOf(const OwnedRecordState& s) {  // as D3dDirectAccess::CutDraw builds it
  const uint32_t isz = s.draw_initiator.index_size == xenos::IndexFormat::kInt16 ? 2u : 4u;
  IndexInfo ii;
  ii.guest_base = s.dma_base & ~(isz - 1);
  ii.endianness = s.dma_size.swap_mode;
  ii.format = s.draw_initiator.index_size;
  ii.count = s.draw_initiator.num_indices;
  return ii;
}

ShaderInterpreter& InterpFor(rex::memory::Memory* mem) {
  struct Tls {
    std::unique_ptr<RegisterFile> regs;
    std::unique_ptr<ShaderInterpreter> interp;
  };
  thread_local Tls t;
  if (!t.interp) {
    t.regs = std::make_unique<RegisterFile>();  // (never read: every call sets a constant source)
    t.interp = std::make_unique<ShaderInterpreter>(*t.regs, *mem);
  }
  return *t.interp;
}

#define SCOUT_REJECT() (*line = int32_t(__LINE__), false)

bool CoverRect(rex::memory::Memory* mem, const CoverInputs& in, xenos::PrimitiveType prim_type, const Shader& vertex_shader, const Shader* pixel_shader,
               int32_t rect[4], bool owned_step, const IndexInfo* ibi, uint32_t facts, int32_t* line) {
  ShaderInterpreter::ConstantSource const_source;
  const_source.floats = in.floats;
  const_source.bool_loop = in.bool_loop;
  const_source.fetch = in.fetch;
  const_source.vs_const = in.vs_const;
  const bool strip = owned_step && prim_type == xenos::PrimitiveType::kTriangleStrip;
  const bool quad = prim_type == xenos::PrimitiveType::kQuadList || strip;
  if ((!quad && prim_type != xenos::PrimitiveType::kRectangleList) || !pixel_shader) return SCOUT_REJECT();
  const uint32_t vertex_count = quad ? 4 : 3;
  const auto initiator = in.initiator;
  const bool strip_indexed = strip && initiator.source_select == xenos::SourceSelect::kDMA && ibi && ibi->count >= 4;
  if ((initiator.source_select != xenos::SourceSelect::kAutoIndex && !strip_indexed) || initiator.num_indices != vertex_count) return SCOUT_REJECT();
  uint32_t strip_indices[4] = {0, 1, 2, 3};
  if (strip_indexed) {
    const bool wide = ibi->format == xenos::IndexFormat::kInt32;
    xenos::Endian endian = ibi->endianness;
    if (!wide && endian != xenos::Endian::kNone && endian != xenos::Endian::k8in16) {
      endian = endian == xenos::Endian::k8in32 ? xenos::Endian::k8in16 : xenos::Endian::kNone;
    }
    const bool reset = in.sc_mode.multi_prim_ib_ena;
    const uint32_t reset_guest = xenos::GpuSwap(in.reset_indx.reset_indx, endian);
    const uint8_t* data = mem->TranslatePhysical<const uint8_t*>(ibi->guest_base & 0x1FFFFFFFu);
    if (!data) return SCOUT_REJECT();
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t v = 0;
      if (wide) {
        std::memcpy(&v, data + size_t(i) * 4, 4);
      } else {
        uint16_t h;
        std::memcpy(&h, data + size_t(i) * 2, 2);
        v = h;
      }
      if (reset && v == reset_guest) return SCOUT_REJECT();
      strip_indices[i] = wide ? xenos::GpuSwap(v, endian) : xenos::GpuSwap(uint16_t(v), endian);
    }
  }
  const auto surface_info = in.surface_info;
  if (surface_info.msaa_samples != xenos::MsaaSamples::k1X || !surface_info.surface_pitch) return SCOUT_REJECT();
  const auto color_control = in.color_control;
  if (color_control.alpha_test_enable || color_control.alpha_to_mask_enable || (facts & 8)) return SCOUT_REJECT();
  const auto window_offset = in.window_offset;
  const auto vte = in.vte;
  if (window_offset.window_x_offset || window_offset.window_y_offset || (!vte.vtx_xy_fmt && !vte.vtx_w0_fmt) || !(facts & 16)) {
    return SCOUT_REJECT();
  }
  const auto mode_cntl = in.sc_mode;
  const auto clip_cntl = in.clip;
  const bool culled = mode_cntl.cull_front || mode_cntl.cull_back;
  if ((culled && !owned_step) || (mode_cntl.cull_front && mode_cntl.cull_back) || (clip_cntl.ucp_ena && !clip_cntl.clip_disable)) return SCOUT_REJECT();
  struct PositionSink : ShaderInterpreter::ExportSink {
    void Export(ucode::ExportRegister export_register, const float* value, uint32_t value_mask) override {
      if (export_register != ucode::ExportRegister::kVSPosition) return;
      for (uint32_t i = 0; i < 4; ++i) {
        if (value_mask & (1u << i)) position[i] = value[i];
      }
      written |= value_mask;
    }
    float position[4] = {};
    uint32_t written = 0;
  } sink;
  ShaderInterpreter& interpreter = InterpFor(mem);
  interpreter.SetRegisterSource(nullptr);
  interpreter.SetConstantSource(&const_source);
  interpreter.SetShader(xenos::ShaderType::kVertex, vertex_shader.ucode_dwords());
  interpreter.SetExportSink(&sink);
  const double scale_x = vte.vport_x_scale_ena ? double(in.vport[0]) : 1.0;
  const double scale_y = vte.vport_y_scale_ena ? double(in.vport[2]) : 1.0;
  const double offset_x = vte.vport_x_offset_ena ? double(in.vport[1]) : 0.0;
  const double offset_y = vte.vport_y_offset_ena ? double(in.vport[3]) : 0.0;
  const uint32_t index_offset = in.indx_offset.indx_offset;
  const uint32_t index_min = in.min_indx.min_indx;
  const uint32_t index_max = in.max_indx.max_indx;
  double px[4] = {}, py[4] = {};
  bool ok = true;
  for (uint32_t i = 0; i < vertex_count && ok; ++i) {
    std::memset(interpreter.temp_registers(), 0, sizeof(float) * 4 * xenos::kMaxShaderTempRegisters);
    const uint32_t raw = strip ? strip_indices[i] : i;
    const uint32_t index = std::min(std::max((raw + index_offset) & xenos::kVertexIndexMask, index_min), index_max);
    interpreter.temp_registers()[0] = float(index);
    sink.written = 0;
    interpreter.Execute();
    const double w = vte.vtx_xy_fmt ? 1.0 : double(sink.position[3]);
    ok = (sink.written & 0b11) == 0b11 && (vte.vtx_xy_fmt || (sink.written & 0b1000)) && w > 0.0 && std::isfinite(w);
    if (ok && !clip_cntl.clip_disable) {
      const double z = double(sink.position[2]), zw = double(sink.position[3]);
      ok = (sink.written & 0b0100) && std::isfinite(z) && z <= zw && z >= (clip_cntl.dx_clip_space_def ? 0.0 : -zw);
    }
    if (!ok) break;
    px[i] = double(sink.position[0]) / w * scale_x + offset_x;
    py[i] = double(sink.position[1]) / w * scale_y + offset_y;
  }
  interpreter.SetExportSink(nullptr);
  interpreter.SetConstantSource(nullptr);
  if (!ok) return SCOUT_REJECT();
  double min_x = DBL_MAX, max_x = -DBL_MAX, min_y = DBL_MAX, max_y = -DBL_MAX;
  for (uint32_t i = 0; i < vertex_count; ++i) {
    min_x = std::min(min_x, px[i]), max_x = std::max(max_x, px[i]);
    min_y = std::min(min_y, py[i]), max_y = std::max(max_y, py[i]);
  }
  const auto on = [](double a, double b) { return std::abs(a - b) <= 1.0e-4; };
  if (strip) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (!(on(px[i], min_x) || on(px[i], max_x)) || !(on(py[i], min_y) || on(py[i], max_y))) return SCOUT_REJECT();
    }
    if (on(px[1], px[2]) || on(py[1], py[2]) || on(px[0], px[3]) || on(py[0], py[3]) || on(px[0], px[1]) == on(py[0], py[1]) || on(min_x, max_x) || on(min_y, max_y)) {
      return SCOUT_REJECT();
    }
  } else if (quad) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (!(on(px[i], min_x) || on(px[i], max_x)) || !(on(py[i], min_y) || on(py[i], max_y))) return SCOUT_REJECT();
      const uint32_t j = (i + 1) & 3;
      if (on(px[i], px[j]) == on(py[i], py[j])) return SCOUT_REJECT();
    }
  } else {
    for (uint32_t i = 0; i < vertex_count; ++i) {
      if (!(on(px[i], min_x) || on(px[i], max_x)) || !(on(py[i], min_y) || on(py[i], max_y))) return SCOUT_REJECT();
      for (uint32_t j = 0; j < i; ++j) {
        if (on(px[i], px[j]) && on(py[i], py[j])) return SCOUT_REJECT();
      }
    }
  }
  if (culled) {
    const double cross = (px[0] * py[1] - px[1] * py[0]) + (px[1] * py[2] - px[2] * py[1]) + (px[2] * py[0] - px[0] * py[2]);
    const bool ccw = -0.5 * cross > 0.0;
    const bool front = mode_cntl.face ? !ccw : ccw;
    if (std::abs(cross) < 1.0e-6 || (mode_cntl.cull_back && !front) || (mode_cntl.cull_front && front)) return SCOUT_REJECT();
  }
  const double centre = in.su_vtx_cntl.pix_center == xenos::PixelCenter::kD3DZero ? 0.0 : 0.5;
  int32_t edges[4];
  const double values[4] = {min_x, min_y, max_x, max_y};
  for (uint32_t i = 0; i < 4; ++i) {
    const double x = values[i] - centre;
    const double distance = std::abs(x - std::round(x));
    if ((distance > 1.0e-6 && distance < 1.0e-3) || x < -1.0 || x > 8192.0) return SCOUT_REJECT();
    edges[i] = std::max(0, int32_t(std::ceil(x - 1.0e-6)));
  }
  const draw_util::Scissor& scissor = in.scissor;
  rect[0] = std::max(edges[0], int32_t(scissor.offset[0]));
  rect[1] = std::max(edges[1], int32_t(scissor.offset[1]));
  rect[2] = std::min(edges[2], int32_t(scissor.offset[0] + scissor.extent[0]));
  rect[3] = std::min(edges[3], int32_t(scissor.offset[1] + scissor.extent[1]));
  return (rect[2] > rect[0] && rect[3] > rect[1]) || SCOUT_REJECT();
}

// Returns the outcome as TruthNoteBounds counts it: 0 made (rect filled), 2 refused (rect zero).
bool BoundsRect(rex::memory::Memory* mem, const CoverInputs& in, xenos::PrimitiveType prim_type, const Shader& vertex_shader, uint32_t index_count,
                const IndexInfo* ibi, uint32_t facts, int32_t rect[4]) {
  std::fill(rect, rect + 4, 0);
  ShaderInterpreter::ConstantSource const_source;
  const_source.floats = in.floats;
  const_source.bool_loop = in.bool_loop;
  const_source.fetch = in.fetch;
  const_source.vs_const = in.vs_const;
  using PT = xenos::PrimitiveType;
  if (prim_type != PT::kTriangleList && prim_type != PT::kTriangleStrip && prim_type != PT::kTriangleFan && prim_type != PT::kQuadList) return false;
  const auto vte = in.vte;
  const auto window_offset = in.window_offset;
  const auto initiator = in.initiator;
  const bool indexed = initiator.source_select == xenos::SourceSelect::kDMA;
  if (window_offset.window_x_offset || window_offset.window_y_offset || !(facts & 16) ||
      (facts & 32) || !index_count || index_count > 8192 || (!indexed && initiator.source_select != xenos::SourceSelect::kAutoIndex) ||
      (indexed && (!ibi || ibi->count < index_count))) {
    return false;
  }
  const uint32_t index_size = indexed && ibi->format == xenos::IndexFormat::kInt32 ? 4u : 2u;
  // (the refusals the inline function makes while it forms its cache key: a vertex buffer over 1 MB or out of the physical space, too many buffers, an index range out of it)
  const Shader::ConstantRegisterMap& map = vertex_shader.constant_register_map();
  uint32_t part_count = (map.float_dynamic_addressing || !(facts & 64)) ? 5u : 2u;  // (2 only when the inline function's memo keys are in use: gpu_interp_memo)
  for (uint32_t i = 0; i < 96; ++i) {
    if (!(map.vertex_fetch_bitmap[i >> 5] & (uint32_t(1) << (i & 31)))) continue;
    xenos::xe_gpu_vertex_fetch_t fetch;
    std::memcpy(&fetch, in.fetch + 2 * size_t(i), sizeof(fetch));
    const uint32_t address = (uint32_t(fetch.address) << 2) & 0x1FFFFFFFu, bytes = uint32_t(fetch.size) << 2;
    if (!(part_count < 15 && bytes <= (1u << 20) && uint64_t(address) + bytes <= 0x20000000u)) return false;
    ++part_count;
  }
  if (indexed) {
    const uint32_t address = ibi->guest_base & 0x1FFFFFFFu, bytes = index_count * index_size;
    if (!(part_count < 16 && uint64_t(address) + bytes <= 0x20000000u)) return false;
  }
  std::vector<uint32_t> indices;
  indices.reserve(index_count);
  const uint32_t index_offset = in.indx_offset.indx_offset;
  const uint32_t index_min = in.min_indx.min_indx;
  const uint32_t index_max = in.max_indx.max_indx;
  xenos::Endian index_endian = indexed ? ibi->endianness : xenos::Endian::kNone;
  if (index_size == 2 && index_endian != xenos::Endian::kNone && index_endian != xenos::Endian::k8in16) {
    index_endian = index_endian == xenos::Endian::k8in32 ? xenos::Endian::k8in16 : xenos::Endian::kNone;
  }
  bool reset_enabled = false;
  uint32_t reset_guest_endian = 0;
  if (indexed && in.sc_mode.multi_prim_ib_ena && (prim_type == PT::kTriangleStrip || prim_type == PT::kTriangleFan)) {
    reset_guest_endian = xenos::GpuSwap(in.reset_indx.reset_indx, index_endian);
    reset_enabled = index_size == 4 || reset_guest_endian <= UINT16_MAX;
  }
  const uint8_t* index_data = indexed ? mem->TranslatePhysical<const uint8_t*>(ibi->guest_base & 0x1FFFFFFFu) : nullptr;
  for (uint32_t i = 0; i < index_count; ++i) {
    uint32_t raw = i;
    if (indexed) {
      if (index_size == 4) {
        uint32_t v;
        std::memcpy(&v, index_data + size_t(i) * 4, 4);
        if (reset_enabled && v == reset_guest_endian) continue;
        raw = xenos::GpuSwap(v, index_endian);
      } else {
        uint16_t v;
        std::memcpy(&v, index_data + size_t(i) * 2, 2);
        if (reset_enabled && v == reset_guest_endian) continue;
        raw = xenos::GpuSwap(v, index_endian);
      }
    }
    indices.push_back(std::min(std::max((raw + index_offset) & xenos::kVertexIndexMask, index_min), index_max));
  }
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  bool ok = !indices.empty() && indices.size() <= 2048;
  double min_x = DBL_MAX, max_x = -DBL_MAX, min_y = DBL_MAX, max_y = -DBL_MAX;
  if (ok) {
    ShaderInterpreter& interpreter = InterpFor(mem);
    interpreter.SetRegisterSource(nullptr);
    interpreter.SetConstantSource(&const_source);
    struct PositionSink : ShaderInterpreter::ExportSink {
      explicit PositionSink(const ShaderInterpreter& i) : interpreter(i) {}
      void Export(ucode::ExportRegister export_register, const float* value, uint32_t value_mask) override {
        if (export_register != ucode::ExportRegister::kVSPosition) return;
        for (uint32_t i = 0; i < 4; ++i) {
          if (value_mask & (1u << i)) position[i] = value[i];
        }
        written |= value_mask;
        last_step = interpreter.steps();
      }
      const ShaderInterpreter& interpreter;
      float position[4] = {};
      uint32_t written = 0;
      uint32_t last_step = 0;
    } sink(interpreter);
    interpreter.SetShader(xenos::ShaderType::kVertex, vertex_shader.ucode_dwords());
    interpreter.SetExportSink(&sink);
    const double scale_x = vte.vport_x_scale_ena ? double(in.vport[0]) : 1.0;
    const double scale_y = vte.vport_y_scale_ena ? double(in.vport[2]) : 1.0;
    const double offset_x = vte.vport_x_offset_ena ? double(in.vport[1]) : 0.0;
    const double offset_y = vte.vport_y_offset_ena ? double(in.vport[3]) : 0.0;
    interpreter.reset_fetches_out_of_range();
    interpreter.set_stop_after_steps(UINT32_MAX);
    bool first = true;
    for (const uint32_t index : indices) {
      std::memset(interpreter.temp_registers(), 0, sizeof(float) * 4 * xenos::kMaxShaderTempRegisters);
      interpreter.temp_registers()[0] = float(index);
      sink.written = 0;
      interpreter.Execute();
      if (first) {
        first = false;
        if (!interpreter.flow_was_dynamic() && sink.written) interpreter.set_stop_after_steps(sink.last_step + 1);
      }
      const double w_host = vte.vtx_w0_fmt ? double(sink.position[3]) : 1.0 / double(sink.position[3]);
      ok = (sink.written & 0b1011) == 0b1011 && w_host > 0.0 && std::isfinite(w_host) && std::isfinite(sink.position[0]) && std::isfinite(sink.position[1]);
      if (!ok) break;
      const double w = vte.vtx_xy_fmt ? 1.0 : w_host;
      const double px = double(sink.position[0]) / w * scale_x + offset_x;
      const double py = double(sink.position[1]) / w * scale_y + offset_y;
      min_x = std::min(min_x, px), max_x = std::max(max_x, px);
      min_y = std::min(min_y, py), max_y = std::max(max_y, py);
    }
    ok = ok && !interpreter.fetches_out_of_range();
    interpreter.set_stop_after_steps(UINT32_MAX);
    interpreter.SetExportSink(nullptr);
    interpreter.SetConstantSource(nullptr);
  }
  ok = ok && min_x >= -16384.0 && max_x <= 16384.0 && min_y >= -16384.0 && max_y <= 16384.0;
  if (ok) {
    rect[0] = std::max(0, int32_t(std::floor(min_x - 2.0)));
    rect[1] = std::max(0, int32_t(std::floor(min_y - 2.0)));
    rect[2] = std::max(rect[0], int32_t(std::ceil(max_x + 2.0)));
    rect[3] = std::max(rect[1], int32_t(std::ceil(max_y + 2.0)));
    ok = rect[2] > rect[0] && rect[3] > rect[1];
  }
  if (!ok) std::fill(rect, rect + 4, 0);
  return ok;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The job: for each snapshot of the range compute what was asked, publish the slot, free the snapshot.
void JobBody(void*, uint32_t first, uint32_t count) {
  Global& g = G();
  rex::memory::Memory* mem = g.memory.load(std::memory_order_acquire);
  for (uint32_t n = 0; n < count; ++n) {
    Snap& sn = g.snaps[(first + n) & (kSnapCount - 1)];
    const uint32_t epoch = g.guest_epoch.load(std::memory_order_acquire);  // (before any guest memory is read)
    const OwnedRecordState& st = sn.st;
    CoverInputs in;
    MakeInputs(st, in);
    const xenos::PrimitiveType prim = st.draw_initiator.prim_type;
    const IndexInfo ii = IndexInfoOf(st);
    uint32_t flags = 0;
    int32_t cover[4] = {}, bounds[4] = {}, line = 0;
    if (mem) {
      if (sn.want & 1) {
        flags |= kFCoverDone;
        if (CoverRect(mem, in, prim, *sn.vs, sn.ps, cover, (sn.want & 4) != 0, &ii, sn.facts, &line)) flags |= kFCoverOk;
        else std::fill(cover, cover + 4, 0);
        g.j_cover.fetch_add(1, std::memory_order_relaxed);
        if (flags & kFCoverOk) g.j_cover_ok.fetch_add(1, std::memory_order_relaxed);
      }
      if (sn.want & 2) {
        flags |= kFBoundsDone;
        if (BoundsRect(mem, in, prim, *sn.vs, st.draw_initiator.num_indices, &ii, sn.facts, bounds)) flags |= kFBoundsOk;
        g.j_bounds.fetch_add(1, std::memory_order_relaxed);
        if (flags & kFBoundsOk) g.j_bounds_ok.fetch_add(1, std::memory_order_relaxed);
      }
    }
    const uint64_t fp = Fingerprint(st, in, *sn.vs, sn.ps);
    Slot& sl = g.slots[SlotIndex(sn.key)];
    if (mem && sl.key.load(std::memory_order_relaxed) == (sn.key | kPendingBit)) {
      sl.w[kWFlags].store(flags, std::memory_order_relaxed);
      sl.w[kWCoverLine].store(uint32_t(line), std::memory_order_relaxed);
      for (uint32_t k = 0; k < 4; ++k) {
        sl.w[kWCover0 + k].store(uint32_t(cover[k]), std::memory_order_relaxed);
        sl.w[kWBounds0 + k].store(uint32_t(bounds[k]), std::memory_order_relaxed);
      }
      sl.w[kWFpLo].store(uint32_t(fp), std::memory_order_relaxed);
      sl.w[kWFpHi].store(uint32_t(fp >> 32), std::memory_order_relaxed);
      sl.w[kWEpoch].store(epoch, std::memory_order_relaxed);
      sl.key.store(sn.key, std::memory_order_release);
    }
    sn.busy.store(0, std::memory_order_release);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The scout.
class Scout {
 public:
  void Main() {
    rex::thread::set_current_thread_name("GPU Jobs scout");
    jb::ConfigureWorkerThread();
    jb::Pool& pool = jb::Pool::Get();
    who_ = pool.RegisterThread();
    kind_ = pool.RegisterKind("cover+bounds");
    Global& g = G();
    uint32_t idle = 0;
    for (;;) {
      if (!g.running.load(std::memory_order_acquire) || !g.queue.load(std::memory_order_acquire) || !g.cp_active.load(std::memory_order_acquire)) {
        FlushBatch();
        std::unique_lock<std::mutex> lock(g.mu);
        g.parked.store(true, std::memory_order_seq_cst);
        g.cv.wait_for(lock, std::chrono::milliseconds(2), [&] {
          return g.running.load(std::memory_order_acquire) && g.cp_active.load(std::memory_order_acquire);
        });
        g.parked.store(false, std::memory_order_seq_cst);
        idle = 0;
        continue;
      }
      if (Step()) {
        idle = 0;
        continue;
      }
      ++idle;
      if (idle < 4096) {  // (~100-300 us of polling the tail while the command thread drains: the next items come within a few draws)
        REX_SCOUT_PAUSE();
      } else {
        std::unique_lock<std::mutex> lock(g.mu);
        g.cv.wait_for(lock, std::chrono::microseconds(100));
      }
    }
  }

 private:
  // One queue item. True = progress.
  bool Step() {
    Global& g = G();
    D3dNoringShared* q = g.queue.load(std::memory_order_acquire);
    rex::memory::Memory* mem = g.memory.load(std::memory_order_acquire);
    if (!q || !mem) return false;
    if (!synced_) {
      if (g.pkg_ready.load(std::memory_order_acquire)) {
        pos_ = g.pkg_pos;
        std::memcpy(&st_, g.pkg_state.get(), sizeof(st_));
        g.scout_seq.store(g.pkg_seq, std::memory_order_relaxed);
        g.pkg_ready.store(false, std::memory_order_release);
        synced_ = true;
        presents_ = 0;
        g.s_resyncs.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      g.need_sync.store(true, std::memory_order_release);
      return false;
    }
    const uint32_t tail = q->tail.load(std::memory_order_acquire);
    if (pos_ == tail) {
      FlushBatch();
      return false;
    }
    const uint32_t cap = q->capacity, mask = cap - 1;
    const uint32_t p = pos_ & mask;
    const uint32_t w0 = q->storage[p];
    const uint32_t kind = w0 & 0xFF, total = w0 >> 8;
    if (int32_t(tail - pos_) < 0 || total == 0 || total > cap - p || (kind != kNrPad && total < kD3dNoringHeaderDwords) || total > uint32_t(tail - pos_)) {
      return Desync();
    }
    cp_head_ = q->head.load(std::memory_order_acquire);
    item_start_ = pos_;
    ord_ = 0;
    if (kind == kNrPackets || kind == kNrRegs) {
      payload_.assign(q->storage + p + kD3dNoringHeaderDwords, q->storage + p + total);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    // the producer reuses space once the command thread has passed it AND it has wrapped round: what was read is good only while the tail is far from this item
    if (uint32_t(q->tail.load(std::memory_order_acquire) - pos_) > cap - cap / 4) return Desync();
    if (kind == kNrPackets || kind == kNrRegs) {
      OwnedScoutApplyHooks hooks;
      hooks.draw = &Scout::DrawThunk;
      hooks.ctx = this;
      OwnedRecordScoutApply(st_, mem, kind, payload_.data(), uint32_t(payload_.size()), hooks);
      g.scout_seq.fetch_add(1, std::memory_order_relaxed);
      g.s_items.fetch_add(1, std::memory_order_relaxed);
    } else if (kind == kNrPresent) {
      if (++presents_ >= kResyncPresents) {  // (a periodic re-base on the command thread's own state)
        synced_ = false;
        g.need_sync.store(true, std::memory_order_release);
      }
      FlushBatch();
    }
    pos_ += total;
    return true;
  }

  bool Desync() {
    Global& g = G();
    g.s_desync.fetch_add(1, std::memory_order_relaxed);
    FlushBatch();
    synced_ = false;
    g.need_sync.store(true, std::memory_order_release);
    return false;
  }

  static void DrawThunk(void* ctx, const OwnedRecordState& st) { static_cast<Scout*>(ctx)->OnDraw(st); }

  void OnDraw(const OwnedRecordState& st) {
    Global& g = G();
    const uint32_t ord = ord_++;
    if (int32_t(item_start_ - cp_head_) <= 0) {  // the command thread is on / past this item
      g.s_behind.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    g.s_draws.fetch_add(1, std::memory_order_relaxed);
    if (st.draw_initiator.source_select != xenos::SourceSelect::kDMA) return;
    const uint64_t vh = st.shader[0].hash, ph = st.shader[1].dwords ? st.shader[1].hash : 0;
    Pair& pr = g.pairs[PairIndex(vh, ph)];
    const Pair* hit = nullptr;
    for (uint32_t probe = 0; probe < 16; ++probe) {
      const Pair& c = g.pairs[(PairIndex(vh, ph) + probe) & (kPairCount - 1)];
      const uint64_t cv = c.vs.load(std::memory_order_acquire);
      if (cv == 0) break;
      if (cv == vh && c.ps.load(std::memory_order_relaxed) == ph) {
        hit = &c;
        break;
      }
    }
    (void)pr;
    if (!hit) {
      g.s_unknown.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const uint32_t f = hit->flags.load(std::memory_order_relaxed);
    const bool owned = (f & 1) != 0, truth = (f & 2) != 0, bvs = (f & 4) != 0;
    const xenos::PrimitiveType prim = st.draw_initiator.prim_type;
    const bool strip = prim == xenos::PrimitiveType::kTriangleStrip;
    const uint32_t n_idx = st.draw_initiator.num_indices;
    // (the cover function refuses anything else at its first lines: a quad list of 4, a rectangle list of 3, an owned step's strip of 4 indices)
    const bool cover_shape = (prim == xenos::PrimitiveType::kQuadList && n_idx == 4) || (prim == xenos::PrimitiveType::kRectangleList && n_idx == 3) ||
                             (owned && strip && n_idx == 4);
    const Shader* ps = hit->psp.load(std::memory_order_relaxed);
    const bool want_cover = ps && cover_shape && (truth || owned);
    const bool want_bounds = bvs && (prim == xenos::PrimitiveType::kTriangleList || strip || prim == xenos::PrimitiveType::kTriangleFan ||
                                     prim == xenos::PrimitiveType::kQuadList);
    if (!want_cover && !want_bounds) {
      g.s_notwanted.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const uint64_t key = kKeyBit | (uint64_t(item_start_) << 16) | std::min<uint32_t>(ord, 0xFFFF);
    Slot& slot = g.slots[SlotIndex(key)];
    if (slot.key.load(std::memory_order_acquire) & kPendingBit) {  // (a job of another draw may still write this slot: one writer per slot)
      g.s_pending_skip.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    Snap& sn = g.snaps[next_snap_ & (kSnapCount - 1)];
    if (sn.busy.load(std::memory_order_acquire)) {
      g.s_ring_full.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    sn.busy.store(1, std::memory_order_relaxed);
    std::memcpy(&sn.st, &st, sizeof(OwnedRecordState));
    sn.key = key;
    sn.facts = f & (8u | 16u | 32u | 64u);
    sn.vs = hit->vsp.load(std::memory_order_relaxed);
    sn.ps = ps;
    sn.want = (want_cover ? 1u : 0u) | (want_bounds ? 2u : 0u) | (owned ? 4u : 0u);
    slot.key.store(key | kPendingBit, std::memory_order_release);
    if (batch_n_ == 0) batch_first_ = next_snap_;
    ++batch_n_;
    ++next_snap_;
    if (batch_n_ >= jb::Pool::Get().Batch(kind_)) FlushBatch();
  }

  void FlushBatch() {
    if (!batch_n_) return;
    Global& g = G();
    jb::Job job;
    job.fn = &JobBody;
    job.ctx = nullptr;
    job.first = batch_first_;
    job.count = batch_n_;
    job.kind = kind_;
    jb::Pool& pool = jb::Pool::Get();
    g.s_jobs.fetch_add(1, std::memory_order_relaxed);
    g.s_batch_items.fetch_add(batch_n_, std::memory_order_relaxed);
    if (!pool.Submit(who_, job)) {
      g.s_inline_runs.fetch_add(1, std::memory_order_relaxed);
      pool.RunInline(job);
    }
    batch_n_ = 0;
  }

  OwnedRecordState st_;
  uint32_t pos_ = 0, cp_head_ = 0, item_start_ = 0, ord_ = 0, presents_ = 0;
  bool synced_ = false;
  uint32_t who_ = jb::kInvalid, kind_ = 0;
  uint32_t next_snap_ = 0, batch_first_ = 0, batch_n_ = 0;
  std::vector<uint32_t> payload_;
};

void StartThread() {
  Global& g = G();
  if (g.thread_made.exchange(true)) return;
  std::thread([] {
    static Scout* scout = new Scout();
    scout->Main();
  }).detach();
}

void WakeScout() {
  Global& g = G();
  if (g.parked.load(std::memory_order_seq_cst)) g.cv.notify_one();  // (no lock: the scout's waits time out, a missed wake costs at most 2 ms)
}

void PrintWindow() {
  Global& g = G();
  CpStats& c = g.cp;
  jb::PoolStats ps;
  jb::Pool::Get().TakeStats(ps);
  const double fr = double(std::max<uint64_t>(1, c.la_frames));
  double busy = 0;
  uint32_t workers = 0;
  for (uint32_t w = 0; w < ps.workers_active && w < jb::kMaxWorkers; ++w, ++workers) busy += double(ps.worker_busy_ns[w]);
  const double busy_pct = workers && ps.window_ns ? 100.0 * busy / (double(ps.window_ns) * workers) : 0.0;
  std::string kinds;
  for (uint32_t k = 0; k < ps.kinds; ++k) {
    const jb::PoolStats::Kind& kd = ps.kind[k];
    char b[160];
    std::snprintf(b, sizeof b, "%s%s jobs %llu, batch %u, %.2f us an item (%u ns)", kinds.empty() ? "" : "; ", kd.name, (unsigned long long)kd.jobs, kd.batch,
                  kd.sampled_items ? double(kd.ns) / double(kd.sampled_items) / 1000.0 : 0.0, kd.ema_ns_per_item);
    kinds += b;
  }
  const uint64_t jc = g.j_cover.exchange(0), jb_ = g.j_bounds.exchange(0), jco = g.j_cover_ok.exchange(0), jbo = g.j_bounds_ok.exchange(0);
  REXGPU_INFO(
      "[jobs] frames {}-{}: slots used {} (cover {}, bounds {}) late {} empty {} (stale {}, torn {}, no key {}); lookahead dwords avg {:.0f} min {:.0f}, items avg {:.1f} min {:.1f}; "
      "scout items {}, draws {} (behind the command thread {}, shader pair unknown {}, not wanted {}), jobs {} ({} items, {} run inline, ring full {}, slot pending {}), resyncs {}, desyncs {}; "
      "job results cover {} ({} ok) bounds {} ({} ok); {}; steals {}, worker busy {:.0f}%; verify checked {} mismatches {}",
      c.frame_first, g.frames, c.used, c.cover_used, c.bounds_used, c.late, c.empty, c.stale, c.torn, c.no_key, double(c.la_avg_sum) / fr, double(c.la_min_sum) / fr,
      double(c.lai_avg_sum) / fr, double(c.lai_min_sum) / fr, g.s_items.exchange(0), g.s_draws.exchange(0), g.s_behind.exchange(0), g.s_unknown.exchange(0),
      g.s_notwanted.exchange(0), g.s_jobs.exchange(0), g.s_batch_items.exchange(0), g.s_inline_runs.exchange(0), g.s_ring_full.exchange(0), g.s_pending_skip.exchange(0), g.s_resyncs.exchange(0),
      g.s_desync.exchange(0), jc, jco, jb_, jbo, kinds.empty() ? "no job kinds yet" : kinds.c_str(), ps.steals, busy_pct, c.verify_checked, c.verify_mismatch);
  const uint64_t mism = c.verify_mismatch;
  c = CpStats();
  c.verify_mismatch = mism;  // (kept across windows)
  c.frame_first = g.frames;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// command thread
void DrainEnter(D3dNoringShared* queue, rex::memory::Memory* memory) {
  if (!g_on) return;
  Global& g = G();
  g.queue.store(queue, std::memory_order_release);
  g.memory.store(memory, std::memory_order_release);
  g.cp_active.store(true, std::memory_order_release);
  g_cp_active_plain = true;
  if (g.running.load(std::memory_order_relaxed)) WakeScout();
}

void DrainLeave() {
  if (!g_cp_active_plain) return;  // (also after a mid-drain switch-off: the scout must not keep reading a queue nobody drains)
  g_cp_active_plain = false;
  G().cp_active.store(false, std::memory_order_release);
}

void BeginItem(uint32_t head) {
  if (!g_on) return;
  Global& g = G();
  CpStats& c = g.cp;
  g.item_pos = head;
  g.draw_ord = 0;
  g.cur_key = 0;
  const uint32_t seq = g.cp_seq.fetch_add(1, std::memory_order_relaxed) + 1;
  D3dNoringShared* q = g.queue.load(std::memory_order_relaxed);
  if (q) {
    const uint64_t la = uint32_t(q->tail.load(std::memory_order_relaxed) - head);
    ++c.la_items;
    c.la_sum += la;
    c.la_min_frame = std::min(c.la_min_frame, la);
    if (g.running.load(std::memory_order_relaxed)) {
      const int32_t ahead = int32_t(g.scout_seq.load(std::memory_order_relaxed) - seq);
      const uint64_t ai = ahead > 0 ? uint64_t(ahead) : 0;
      c.lai_sum += ai;
      c.lai_min_frame = std::min(c.lai_min_frame, ai);
    }
  }
  if (g.running.load(std::memory_order_relaxed) && g.need_sync.load(std::memory_order_acquire) && !g.pkg_ready.load(std::memory_order_acquire)) {
    OwnedRecordScoutPrepare();
    std::memcpy(g.pkg_state.get(), &OwnedRecordStateNow(), sizeof(OwnedRecordState));
    g.pkg_pos = head;
    g.pkg_seq = seq - 1;
    g.need_sync.store(false, std::memory_order_relaxed);
    g.pkg_ready.store(true, std::memory_order_release);
  }
}

void NotePresent(D3dNoringShared* queue, rex::memory::Memory* memory) {
  if (!g_on && !REXCVAR_GET(gpu_jobs)) return;  // (the default path: no table is touched)
  Global& g = G();
  const int32_t bits = REXCVAR_GET(gpu_jobs);
  const bool want_on = bits != 0 && free_counters::g_cut.active;
  if (want_on != g_on) {
    g_on = want_on;
    if (!want_on) {
      g.running.store(false, std::memory_order_release);
    } else {  // (switched on in the middle of a drain: DrainEnter did not run for it)
      g.queue.store(queue, std::memory_order_release);
      g.memory.store(memory, std::memory_order_release);
      g.cp_active.store(true, std::memory_order_release);
      g_cp_active_plain = true;
    }
  }
  if (!g_on) return;
  ++g.frames;
  jb::Pool::Get().SetSampling((g.frames & 7) == 0);  // (job timing for the EMA: 1 frame in 8)
  g.verify = REXCVAR_GET(gpu_jobs_verify);
  jb::Pool& pool = jb::Pool::Get();
  pool.Configure(REXCVAR_GET(gpu_jobs_workers));
  const uint32_t active = pool.Active();
  // (one worker beside the scout gains nothing on a small machine: the scout needs 2, unless the count is forced)
  const bool scout_on = (bits & 2) != 0 && active > 0 && (active >= 2 || REXCVAR_GET(gpu_jobs_workers) > 0);
  if (scout_on != g.last_scout || active != g.last_active) {
    g.last_scout = scout_on;
    g.last_active = active;
    REXGPU_INFO("[jobs] pool: {} workers (lp {}, busy 6, spin {} us, scout {})", active, jb::Pool::LogicalProcessors(), pool.SpinMicros(), scout_on ? "on" : "off");
  }
  if (scout_on && !g.running.load(std::memory_order_relaxed)) {
    StartThread();
    g.need_sync.store(true, std::memory_order_release);
    g.running.store(true, std::memory_order_release);
  } else if (!scout_on && g.running.load(std::memory_order_relaxed)) {
    g.running.store(false, std::memory_order_release);
  }
  CpStats& c = g.cp;
  if (c.la_items) {
    ++c.la_frames;
    c.la_avg_sum += c.la_sum / c.la_items;
    c.la_min_sum += c.la_min_frame == UINT64_MAX ? 0 : c.la_min_frame;
    c.lai_avg_sum += c.lai_sum / c.la_items;
    c.lai_min_sum += c.lai_min_frame == UINT64_MAX ? 0 : c.lai_min_frame;
  }
  c.la_items = c.la_sum = c.lai_sum = 0;
  c.la_min_frame = c.lai_min_frame = UINT64_MAX;
  if (g.frames - c.frame_first >= kWindowFrames) PrintWindow();
}

void NoteGuestWrite() {
  if (g_on) G().guest_epoch.fetch_add(1, std::memory_order_acq_rel);
}

void Quiesce() {
  Global& g = G();
  g_on = false;
  g_cp_active_plain = false;
  g.running.store(false, std::memory_order_release);
  g.cp_active.store(false, std::memory_order_release);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  for (bool busy = true; busy && std::chrono::steady_clock::now() < deadline;) {
    busy = false;
    for (uint32_t i = 0; i < kSnapCount; ++i) busy |= g.snaps[i].busy.load(std::memory_order_acquire) != 0;
    if (busy) std::this_thread::yield();
  }
  g.queue.store(nullptr, std::memory_order_release);
  g.memory.store(nullptr, std::memory_order_release);  // (a job still running publishes nothing and frees its snapshot)
}

void DrawItem() {
  if (!g_on) return;
  Global& g = G();
  g.cur_key = kKeyBit | (uint64_t(g.item_pos) << 16) | std::min<uint32_t>(g.draw_ord, 0xFFFF);
  ++g.draw_ord;
}
void ClearKey() {
  if (g_on) G().cur_key = 0;
}
uint64_t CurrentKey() { return g_on ? G().cur_key : 0; }

// ---- Lookup
Lookup::Lookup(const OwnedRecordState* s, bool crowd_member, const Shader& vs, const Shader* ps) : s_(s), vs_(&vs), ps_(ps) {
  active_ = g_on && s && !crowd_member && !s->crowd_member && !s->inline_draw && G().running.load(std::memory_order_relaxed);
  if (active_ && !G().cur_key) {
    active_ = false;
    ++G().cp.no_key;
  }
}

void Lookup::Note(bool owned_cover, bool truth_wants_cover, bool bounds_vs) {
  if (!active_) return;
  Global& g = G();
  const uint64_t vh = vs_->ucode_data_hash(), ph = ps_ ? ps_->ucode_data_hash() : 0;
  if (!vs_->is_ucode_analyzed()) return;
  // what the job bodies would otherwise ask the Shader objects (read here, on the command thread)
  const uint32_t flags = (owned_cover ? 1u : 0u) | (truth_wants_cover ? 2u : 0u) | (bounds_vs ? 4u : 0u) | (ps_ && ps_->kills_pixels() ? 8u : 0u) |
                         (ShaderInterpreter::CanInterpretShader(*vs_) ? 16u : 0u) | (!vs_->cf_memexport_info().empty() ? 32u : 0u) | ((REXCVAR_GET(gpu_interp_memo) || (REXCVAR_GET(gpu_d3d_wall) & 64u)) ? 64u : 0u);
  for (uint32_t probe = 0; probe < 16; ++probe) {
    Pair& c = g.pairs[(PairIndex(vh, ph) + probe) & (kPairCount - 1)];
    const uint64_t cv = c.vs.load(std::memory_order_relaxed);
    if (cv == vh && c.ps.load(std::memory_order_relaxed) == ph) {
      if (c.flags.load(std::memory_order_relaxed) != flags) c.flags.store(flags, std::memory_order_relaxed);
      return;
    }
    if (cv == 0) {  // an empty entry: the only writer is this thread
      c.ps.store(ph, std::memory_order_relaxed);
      c.vsp.store(vs_, std::memory_order_relaxed);
      c.psp.store(ps_, std::memory_order_relaxed);
      c.flags.store(flags, std::memory_order_relaxed);
      c.vs.store(vh, std::memory_order_release);
      return;
    }
  }
}

bool Lookup::Fetch() {
  if (fetched_) return hit_;
  fetched_ = true;
  Global& g = G();
  CpStats& c = g.cp;
  const uint64_t key = g.cur_key;
  Slot& sl = g.slots[SlotIndex(key)];
  const uint64_t k1 = sl.key.load(std::memory_order_acquire);
  if (k1 != key) {
    ++(k1 == (key | kPendingBit) ? c.late : c.empty);
    return false;
  }
  uint32_t w[kWCount];
  for (uint32_t i = 0; i < kWCount; ++i) w[i] = sl.w[i].load(std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_acquire);
  if (sl.key.load(std::memory_order_relaxed) != k1) {
    ++c.torn;
    return false;
  }
  CoverInputs in;
  MakeInputs(*s_, in);
  const uint64_t fp = Fingerprint(*s_, in, *vs_, ps_);
  if (w[kWEpoch] != g.guest_epoch.load(std::memory_order_acquire) || fp != (uint64_t(w[kWFpLo]) | (uint64_t(w[kWFpHi]) << 32))) {
    ++c.stale;
    return false;
  }
  cover_job_ = (w[kWFlags] & kFCoverDone) != 0;
  cover_ok_ = (w[kWFlags] & kFCoverOk) != 0;
  bounds_job_ = (w[kWFlags] & kFBoundsDone) != 0;
  bounds_ok_ = (w[kWFlags] & kFBoundsOk) != 0;
  cover_line_ = int32_t(w[kWCoverLine]);
  for (uint32_t k = 0; k < 4; ++k) {
    cover_rect_[k] = int32_t(w[kWCover0 + k]);
    bounds_rect_[k] = int32_t(w[kWBounds0 + k]);
  }
  bounds_outcome_ = bounds_ok_ ? 0 : 2;
  hit_ = true;
  ++c.used;
  return true;
}

bool Lookup::Cover(bool& ok, int32_t rect[4], int32_t& reject_line) {
  if (!active_ || !Fetch() || !cover_job_) return false;
  if (G().verify) return false;
  ok = cover_ok_;
  reject_line = cover_line_;
  std::copy(cover_rect_, cover_rect_ + 4, rect);
  ++G().cp.cover_used;
  return true;
}

static void LogMismatch(const char* what, bool a_ok, const int32_t* a, bool b_ok, const int32_t* b) {
  static uint32_t logged = 0;
  if (logged++ >= 8) return;
  REXGPU_WARN("[jobs] VERIFY MISMATCH {}: job ok {} rect {} {} {} {}, inline ok {} rect {} {} {} {}", what, a_ok, a[0], a[1], a[2], a[3], b_ok, b[0], b[1], b[2], b[3]);
}

void Lookup::VerifyCover(bool ok, const int32_t rect[4]) {
  if (!active_ || !hit_ || !cover_job_ || !G().verify) return;
  CpStats& c = G().cp;
  ++c.verify_checked;
  if (ok != cover_ok_ || (ok && !std::equal(rect, rect + 4, cover_rect_))) {
    ++c.verify_mismatch;
    LogMismatch("cover", cover_ok_, cover_rect_, ok, rect);
  }
}

int Lookup::Bounds(int32_t rect[4]) {
  if (!active_ || !Fetch() || !bounds_job_) return -1;
  if (G().verify) return -1;
  std::copy(bounds_rect_, bounds_rect_ + 4, rect);
  ++G().cp.bounds_used;
  return bounds_outcome_;
}

void Lookup::VerifyBounds(bool ok, const int32_t rect[4]) {
  if (!active_ || !hit_ || !bounds_job_ || !G().verify) return;
  CpStats& c = G().cp;
  ++c.verify_checked;
  if (ok != bounds_ok_ || (ok && !std::equal(rect, rect + 4, bounds_rect_))) {
    ++c.verify_mismatch;
    LogMismatch("bounds", bounds_ok_, bounds_rect_, ok, rect);
  }
}

}  // namespace rex::graphics::vulkan::scout
