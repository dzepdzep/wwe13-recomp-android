// wwe2k14 - Session 100, strip-360 B3 (STRIP360_PLAN.md): in owner mode (the new renderer draws; the old command processor draws and copies
// nothing, gpu_no_draw) the library's Clear (wrapper 0x82B9CDD8) and Resolve (wrapper 0x82B976A8) are replaced by the device bookkeeping they
// leave behind, as OwnerNullDraw does for draws (d3d_layer.h). No packets, no ring memory, no record: the reader has already taken the call
// (OnClear / OnResolve run first in the same hooks, d3d_layer_copy.h). The device effects come from docs\systems\owner_mode_duties.md
// sections 1.1 / 1.2 and 4, re-read from the listings 2026-11-11 (Session 100):
//
//   clear    0x82B9CCA0 builds the rectangle (RT0's size, else the depth surface's) and calls the core 0x82B9C620 once. The core returns
//            at once, writing nothing, when there is no depth surface and the flags name nothing but depth / stencil (L 0x82B9C664-
//            0x82B9C674), or when the rectangle clipped by the viewport and the scissor is empty (L 0x82B9C75C-0x82B9C768). Otherwise it
//            ORs main-mask bits 19 and 20 (L 0x82B9CB30-0x82B9CB3C: the next draw re-binds both shaders), runs one pass per colour target
//            named and bound (or one depth / stencil pass), each pass calling 0x82B99D18 when byte 11071 & 2 or byte 11072 & 0xC0
//            (L 0x82B9C320-0x82B9C348; 0x82B99D18 stores byte 11072 &= 0x3F, byte 11071 &= 0xFC, misc mask +32 top two bits cleared) and
//            drawing its rectangle (0x82B9BB60: fetch mask +24 |= 0x80000000, texture slot 0, L 0x82B9BDF4-0x82B9BDFC); it ends with
//            byte 11069 |= 4 and main mask +16 |= 0x20000 (L 0x82B9CC7C-0x82B9CC90). The float masks +0 / +8 are never touched.
//   resolve  the wrapper calls the core 0x82B96550 with r6 = r7 = 0. Locals: L136 = (flags & 0x8000) and the global 0x835CC108 is 0
//            (L 0x82B965B0-0x82B965CC), L144 = (flags & 0x4000) and the global 0x835CC104 is 0 (L 0x82B96588-0x82B965AC); both are 0 in the
//            common case and anything else is left to the original. With both 0: byte 11070 &= ~8 (L 0x82B96EC8-0x82B96F10); a depth-
//            clearing resolve (flags & 0x200, r26 at L 0x82B96A48) byte 11069 |= 4 and +16 |= 0x20000 (L 0x82B96FC8-0x82B96FFC);
//            0x82B99D18 (L 0x82B97018); +24 |= 0x80000000 and +16 |= 0x80000 (L 0x82B97408-0x82B9742C, r21 = r7 = 0); then, when the
//            global 0x835CC100 is 0, byte 11072 |= 0xC0 and +32 |= 0xC000000000000000 (L 0x82B975E0-0x82B97604: the test there is on
//            r26 = L144, reloaded at L 0x82B970D4, NOT the depth-clear flag: a depth-clearing resolve arms it too); +10776 = +10788 = 0
//            (L 0x82B9760C-0x82B97610); the destination texture's +8 and +12 = the device's issued fence +11036 (L 0x82B97630-0x82B97644).
//            Fence 0 (boot only) appends to a list instead: left to the original.
// Main / misc mask bits besides the ones above are left alone: the next null draw clears all five masks (OwnerNullDraw), and only bits
// 17-20 (the shader bind), the fetch mask's low word (the reader's texture slots) and the misc top bits (0x82B99D18's pending rule) are read
// before that. Ring memory the originals allocate (0x82B9D7B8) and their packets are not needed by anyone in owner mode.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(wwe2k14_owner_null_copy, true, "wwe2k14",
                    "Session 100 (strip-360 B3): in owner mode the library's clears and resolves leave only their device bookkeeping (no "
                    "packets for the old command processor); false: the originals run as before");

namespace wwe2k14::owner_null_copy {

namespace dl = wwe2k14::d3d_layer;

enum Bail : uint32_t { kRects = 0, kTiling, kHiz, kNoSurface, kResSource, kResParams, kResFlags, kResFence, kResShape, kBailCount };
inline constexpr const char* kBailNames[kBailCount] = {"clear with rectangles", "tiling / extents mode", "HiZ not plain", "clear with no surface",
                                                       "resolve source",        "resolve params",       "resolve flags 0x4000/0x8000",
                                                       "resolve fence 0",       "resolve destination shape"};

struct Counters {
  std::atomic<uint64_t> clears{0}, clears_empty{0}, resolves{0};
  std::array<std::atomic<uint64_t>, kBailCount> bails{};
};
inline Counters g_c;

inline bool Active() {
  return REXCVAR_GET(wwe2k14_rdr_owner) && REXCVAR_GET(wwe2k14_d3d_layer_owner_null) && REXCVAR_GET(wwe2k14_owner_null_copy);
}

inline bool Left(Bail why) {
  static std::array<std::atomic<bool>, kBailCount> logged{};
  if (!logged[why].exchange(true, std::memory_order_relaxed)) {
    REXLOG_INFO("[owner][copy] first clear / resolve left to the original: {}", kBailNames[why]);
  }
  g_c.bails[why].fetch_add(1, std::memory_order_relaxed);
  return false;
}

// 0x82B99D18's device stores (L 0x82B99D90-0x82B99DB0).
inline void PendingFlush(uint8_t* base, uint32_t dev) {
  base[dev + 11072] = uint8_t(base[dev + 11072] & 0x3F);
  base[dev + 11071] = uint8_t(base[dev + 11071] & 0xFC);
  dl::St64(base, dev + 32, dl::Ld64(base, dev + 32) & 0x3FFFFFFFFFFFFFFFull);
}

inline int32_t Trunc(uint32_t float_bits) {
  float f;
  std::memcpy(&f, &float_bits, 4);
  return int32_t(f);  // fctiwz
}

// The Clear wrapper 0x82B9CDD8 (r3 device, r5 rectangles, r6 flags). True = done (the original must not run).
inline bool Clear(PPCContext& ctx, uint8_t* base) {
  if (!Active()) return false;
  const uint32_t dev = ctx.r3.u32, rects = ctx.r5.u32, flags = ctx.r6.u32;
  if (rects) return Left(kRects);
  if (base[dev + 11068] & 0x31) return Left(kTiling);  // (0x82B9CCA0's tile-size rectangle, the extents recording, inside tiling)
  if (!wwe2k14::d3d_hiz::PlainCall(base, dev)) return Left(kHiz);
  const uint32_t rt0 = dl::Ld32(base, dev + 12816), depth_surface = dl::Ld32(base, dev + 12832);
  const uint32_t size_surface = rt0 ? rt0 : depth_surface;
  if (!size_surface) return Left(kNoSurface);
  // The core's early returns: nothing written.
  uint32_t eff = flags;
  if (!depth_surface) eff &= ~0xF0u;
  const uint32_t size_word = dl::Ld32(base, size_surface + 36);
  const int32_t width = int32_t((size_word >> 18) & 0x3FFF) + 1, height = int32_t((size_word >> 3) & 0x7FFF) + 1;
  const int32_t vx = Trunc(dl::Ld32(base, dev + 13024)), vy = Trunc(dl::Ld32(base, dev + 13028));
  const int32_t vw = Trunc(dl::Ld32(base, dev + 13032)), vh = Trunc(dl::Ld32(base, dev + 13036));
  int32_t x0 = std::max(0, vx), y0 = std::max(0, vy), x1 = std::min(width, vx + vw), y1 = std::min(height, vy + vh);
  if (dl::Ld32(base, dev + 12288)) {
    x0 = std::max(x0, int32_t(dl::Ld32(base, dev + 13052)));
    y0 = std::max(y0, int32_t(dl::Ld32(base, dev + 13056)));
    x1 = std::min(x1, int32_t(dl::Ld32(base, dev + 13060)));
    y1 = std::min(y1, int32_t(dl::Ld32(base, dev + 13064)));
  }
  // (L 0x82B9C664-0x82B9C674: only with no depth surface does a call naming nothing but depth / stencil return at once; with one, flags 0
  // still make the single depth / stencil pass below)
  if ((!depth_surface && !eff) || x1 <= x0 || y1 <= y0) {
    g_c.clears_empty.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  // The passes (0x82B9CB44-0x82B9CC48): no colour flag = one depth / stencil pass; else one per named target that has a surface.
  bool any_pass = !(eff & 0xF);
  for (uint32_t i = 0; i < 4 && !any_pass; ++i) any_pass = (eff & (1u << i)) && dl::Ld32(base, dev + 12816 + 4 * i);
  dl::St64(base, dev + 16, dl::Ld64(base, dev + 16) | 0x180000ull);
  if (any_pass) {
    if ((base[dev + 11071] & 2) || (base[dev + 11072] & 0xC0)) PendingFlush(base, dev);  // (the first pass consumes it)
    dl::St64(base, dev + 24, dl::Ld64(base, dev + 24) | 0x80000000ull);
  }
  base[dev + 11069] = uint8_t(base[dev + 11069] | 4);
  dl::St64(base, dev + 16, dl::Ld64(base, dev + 16) | 0x20000ull);
  g_c.clears.fetch_add(1, std::memory_order_relaxed);
  return true;
}

// The Resolve wrapper 0x82B976A8 (r3 device, r4 flags, r6 destination texture; the caller's r1 + 100 a params pointer). True = done.
inline bool Resolve(PPCContext& ctx, uint8_t* base) {
  if (!Active()) return false;
  const uint32_t dev = ctx.r3.u32, flags = ctx.r4.u32, dt = ctx.r6.u32;
  if (base[dev + 11068] & 0x31) return Left(kTiling);
  if (!wwe2k14::d3d_hiz::PlainCall(base, dev)) return Left(kHiz);
  if (!dt || (flags & 7) > 4) return Left(kResSource);
  if (dl::Ld32(base, ctx.r1.u32 + 100)) return Left(kResParams);
  if (((flags & 0x8000u) && dl::Ld32(base, 0x835CC108u) == 0) || ((flags & 0x4000u) && dl::Ld32(base, 0x835CC104u) == 0)) return Left(kResFlags);
  // (a 3D destination or format 61: the paths the old layer never described; left to the original)
  if (((dl::Ld32(base, dt + 48) >> 9) & 3) == 2 || (dl::Ld32(base, dt + 32) & 0x3F) == 61) return Left(kResShape);
  const uint32_t fence = dl::Ld32(base, dev + 11036);
  if (!fence) return Left(kResFence);

  base[dev + 11070] = uint8_t(base[dev + 11070] & 0xF7);
  if (flags & 0x200u) {
    base[dev + 11069] = uint8_t(base[dev + 11069] | 4);
    dl::St64(base, dev + 16, dl::Ld64(base, dev + 16) | 0x20000ull);
  }
  PendingFlush(base, dev);
  dl::St64(base, dev + 24, dl::Ld64(base, dev + 24) | 0x80000000ull);
  dl::St64(base, dev + 16, dl::Ld64(base, dev + 16) | 0x80000ull);
  if (dl::Ld32(base, 0x835CC100u) == 0) {
    base[dev + 11072] = uint8_t(base[dev + 11072] | 0xC0);
    dl::St64(base, dev + 32, dl::Ld64(base, dev + 32) | 0xC000000000000000ull);
  }
  dl::St32(base, dev + 10776, 0);
  dl::St32(base, dev + 10788, 0);
  dl::St32(base, dt + 8, fence);
  dl::St32(base, dt + 12, fence);
  g_c.resolves.fetch_add(1, std::memory_order_relaxed);
  return true;
}

}  // namespace wwe2k14::owner_null_copy
