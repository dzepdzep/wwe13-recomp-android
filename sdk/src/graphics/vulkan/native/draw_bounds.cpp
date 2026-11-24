// wwe2k14 patch (Session 93): see draw_bounds.h. The code is the census' ScanMesh / BuildBox / BoxTest, moved here as they were.
#include "draw_bounds.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <rex/system/xmemory.h>

#include "owned_record.h"

namespace rex::graphics::vulkan::draw_bounds {

namespace {
constexpr uint64_t kPhysicalBytes = 0x20000000ull;
inline float F(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
}  // namespace

ScanStatus ScanIndices(rex::memory::Memory* mem, const OwnedRecordState& s, uint32_t& budget, IndexScan& out) {
  const bool is16 = s.draw_initiator.index_size == xenos::IndexFormat::kInt16;
  const uint32_t isz = is16 ? 2 : 4;
  uint32_t n = s.draw_initiator.num_indices;
  const uint32_t avail = s.dma_size.num_words * (4 / isz);
  if (n > avail) n = avail;
  const xenos::Endian endian = s.dma_size.swap_mode;
  if (is16 ? (endian != xenos::Endian::kNone && endian != xenos::Endian::k8in16) : false) return ScanStatus::kUnsupported;
  if (!n || n > 2000000) return ScanStatus::kUnsupported;
  if (uint64_t(s.dma_base & ~(isz - 1)) + uint64_t(n) * isz > kPhysicalBytes) return ScanStatus::kUnsupported;  // (outside the 512 MiB physical space)
  if (budget < n) return ScanStatus::kBudget;
  budget -= n;
  out.n = n;
  uint32_t mn = 0xFFFFFFFFu, mx = 0;
  bool resets = false;
  const uint32_t base = s.dma_base & ~(isz - 1);
  // The primitive reset index is skipped only when the reset is on (multi_prim_ib_ena) and the value equals reset_indx (24 bits; the low 16 for 16-bit indices).
  const bool reset_on = s.sc_mode_cntl.multi_prim_ib_ena != 0;
  const uint32_t reset16 = s.reset_indx.value & 0xFFFFu, reset32 = s.reset_indx.value & 0xFFFFFFu;
  if (is16) {
    const uint16_t* p = mem->TranslatePhysical<const uint16_t*>(base);
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t v = xenos::GpuSwap(p[i], endian);
      if (reset_on && v == reset16) {
        resets = true;
        continue;
      }
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
  } else {
    const uint32_t* p = mem->TranslatePhysical<const uint32_t*>(base);
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t v = xenos::GpuSwap(p[i], endian);
      if (reset_on && (v & 0xFFFFFFu) == reset32) {
        resets = true;
        continue;
      }
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
  }
  if (mn == 0xFFFFFFFFu) return ScanStatus::kEmpty;
  out.min_i = mn;
  out.max_i = mx;
  out.resets = resets;
  return ScanStatus::kOk;
}

BoxStatus BuildBox(rex::memory::Memory* mem, uint32_t min_i, uint32_t max_i, uint32_t vb_start_byte, uint32_t vb_size_bytes, uint32_t stride_words, uint32_t base,
                   xenos::Endian endian, uint32_t& budget, Box& out, uint32_t& vertices_read) {
  vertices_read = 0;
  const uint32_t first = min_i + base, last = max_i + base;
  const uint64_t stride = uint64_t(stride_words) * 4;
  if (last < first || uint64_t(last) * stride + 16 > vb_size_bytes) return BoxStatus::kRejected;
  if (uint64_t(vb_start_byte) + vb_size_bytes > kPhysicalBytes) return BoxStatus::kRejected;
  const uint32_t count = last - first + 1;
  if (count > 200000) return BoxStatus::kRejected;
  if (budget < count) return BoxStatus::kBudget;
  budget -= count;
  vertices_read = count;
  float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
  bool w_ok = true;
  const uint8_t* vb = mem->TranslatePhysical<const uint8_t*>(vb_start_byte);
  for (uint32_t v = first; v <= last; ++v) {
    uint32_t w[4];
    std::memcpy(w, vb + uint64_t(v) * stride, 16);
    for (int k = 0; k < 4; ++k) w[k] = xenos::GpuSwap(w[k], endian);
    const float x = F(w[0]), y = F(w[1]), z = F(w[2]), wf = F(w[3]);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(wf)) return BoxStatus::kRejected;
    if (wf != 1.0f) w_ok = false;
    lo[0] = std::min(lo[0], x);
    lo[1] = std::min(lo[1], y);
    lo[2] = std::min(lo[2], z);
    hi[0] = std::max(hi[0], x);
    hi[1] = std::max(hi[1], y);
    hi[2] = std::max(hi[2], z);
  }
  std::memcpy(out.lo, lo, sizeof(lo));
  std::memcpy(out.hi, hi, sizeof(hi));
  return w_ok ? BoxStatus::kOk : BoxStatus::kWNotOne;
}

int BoxTest(const uint32_t* vs_float, const Box& m, const float* host) {
  const float* c76 = reinterpret_cast<const float*>(vs_float + 4 * 76);
  const float* c77 = reinterpret_cast<const float*>(vs_float + 4 * 77);
  const float* c78 = reinterpret_cast<const float*>(vs_float + 4 * 78);
  const float* cam[4];
  for (int k = 0; k < 4; ++k) cam[k] = reinterpret_cast<const float*>(vs_float + 4 * (56 + k));
  bool all_behind = true;
  bool out[5] = {true, true, true, true, true};  // x < -w, x > w, y < -w, y > w, z > w
  const float margin = 1.01f;
  for (int c = 0; c < 8; ++c) {
    const float p[4] = {(c & 1) ? m.hi[0] : m.lo[0], (c & 2) ? m.hi[1] : m.lo[1], (c & 4) ? m.hi[2] : m.lo[2], 1.0f};
    const float wx = p[0] * c76[0] + p[1] * c76[1] + p[2] * c76[2] + p[3] * c76[3];
    const float wy = p[0] * c77[0] + p[1] * c77[1] + p[2] * c77[2] + p[3] * c77[3];
    const float wz = p[0] * c78[0] + p[1] * c78[1] + p[2] * c78[2] + p[3] * c78[3];
    float q[4];
    for (int k = 0; k < 4; ++k) q[k] = wx * cam[k][0] + wy * cam[k][1] + wz * cam[k][2] + cam[k][3];
    if (host) {
      q[0] = q[0] * host[0] + q[3] * host[2];
      q[1] = q[1] * host[1] + q[3] * host[3];
    }
    if (!(q[3] <= 0.0f)) all_behind = false;
    const bool pos = q[3] > 0.0f;
    out[0] &= pos && q[0] < -q[3] * margin;
    out[1] &= pos && q[0] > q[3] * margin;
    out[2] &= pos && q[1] < -q[3] * margin;
    out[3] &= pos && q[1] > q[3] * margin;
    out[4] &= pos && q[2] > q[3] * margin;
  }
  if (all_behind) return 2;
  if (out[0] || out[1] || out[2] || out[3]) return 1;
  if (out[4]) return 3;
  return 0;
}

}  // namespace rex::graphics::vulkan::draw_bounds
