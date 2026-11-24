/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/format/ucode.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/shared_memory.h>
#include <rex/graphics/util/draw_extent_estimator.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/perf/timeline.h>
#include <rex/ui/graphics_util.h>

REXCVAR_DEFINE_BOOL(execute_unclipped_draw_vs_on_cpu, false, "GPU",
                    "Execute unclipped draw vertex shader on CPU");

REXCVAR_DEFINE_BOOL(execute_unclipped_draw_vs_on_cpu_with_scissor, false, "GPU",
                    "Execute unclipped draw VS on CPU with scissor");

// wwe2k14 patch (Session 7, EXTENTS-SHADOW): CPU-computed screen extents for predicated tiling (see
// ScreenExtentsEstimator / ScreenExtentsShadow at the end of this file). All latched at frame boundaries (the
// guest swap), so they can be changed mid-run.
REXCVAR_DEFINE_STRING(gpu_extents_shaders, "", "GPU",
                      "Screen extents: comma-separated vertex shader ucode hashes (hex, as in the frame anatomy "
                      "CSV) whose draws get CPU-computed screen extents while gpu_extents_mode is not 0");
REXCVAR_DEFINE_INT32(gpu_extents_mode, 0, "GPU",
                     "Screen extents: 0 off; 1 shadow: compute conservative screen extents for the draws of "
                     "gpu_extents_shaders in the extents pass (tile 0) and count what predicated tiling would skip "
                     "in tiles 1-3 ([s7][EXTENTS] summaries), but keep reporting full-screen extents to the game; "
                     "2 report the computed extents (only together with gpu_extents_allow_write, otherwise like 1)");
REXCVAR_DEFINE_BOOL(gpu_extents_allow_write, false, "GPU",
                    "Screen extents: let gpu_extents_mode 2 report computed extents to the game (changes which "
                    "draws the game skips in tiles 1-3). Off: mode 2 behaves like mode 1");
REXCVAR_DEFINE_INT32(gpu_extents_mode_verify, 0, "GPU",
                     "Screen extents: N > 0 = every N-th computed draw is also run vertex by vertex through the "
                     "shader interpreter, and every vertex must lie inside the computed extents (mismatches logged "
                     "as [s7][EXTENTS-SHADOW], counts in the [s7][EXTENTS] summary). Slow; 0 = off");

// DEFINE_bool(
//     execute_unclipped_draw_vs_on_cpu, true,
//     "Execute the vertex shader for draws with clipping disabled, primarily "
//     "screen-space draws (such as clears), on the CPU when possible to estimate "
//     "the extent of the EDRAM involved in the draw.\n"
//     "Enabling this may significantly improve GPU performance as otherwise up "
//     "to the entire EDRAM may be considered used in draws without clipping, "
//     "potentially resulting in spurious EDRAM range ownership transfer round "
//     "trips between host render targets.\n"
//     "Also, on hosts where certain render target formats have to be emulated in "
//     "a lossy way (for instance, 16-bit fixed-point via 16-bit floating-point), "
//     "this prevents corruption of other render targets located after the "
//     "current ones in the EDRAM by lossy range ownership transfers done for "
//     "those draws.",
//     "GPU");
// DEFINE_bool(
//     execute_unclipped_draw_vs_on_cpu_with_scissor, false,
//     "Don't restrict the usage of execute_unclipped_draw_vs_on_cpu to only "
//     "non-scissored draws (with the right and the bottom sides of the scissor "
//     "rectangle at 8192 or beyond) even though if the scissor rectangle is "
//     "present, it's usually sufficient for esimating the height of the render "
//     "target.\n"
//     "Enabling this may cause excessive processing of vertices on the CPU, as "
//     "some games draw rectangles (for their UI, for instance) without clipping, "
//     "but with a proper scissor rectangle.",
//     "GPU");

namespace rex::graphics {

void DrawExtentEstimator::PositionYExportSink::Export(ucode::ExportRegister export_register,
                                                      const float* value, uint32_t value_mask) {
  if (export_register == ucode::ExportRegister::kVSPosition) {
    if (value_mask & 0b0010) {
      position_y_ = value[1];
    }
    if (value_mask & 0b1000) {
      position_w_ = value[3];
    }
  } else if (export_register == ucode::ExportRegister::kVSPointSizeEdgeFlagKillVertex) {
    if (value_mask & 0b0001) {
      point_size_ = value[0];
    }
    if (value_mask & 0b0100) {
      vertex_kill_ = rex::memory::Reinterpret<uint32_t>(value[2]);
    }
  }
}

uint32_t DrawExtentEstimator::EstimateVertexMaxY(const Shader& vertex_shader) {
  SCOPE_profile_cpu_f("gpu");

  const RegisterFile& regs = Regs();

  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  if (!vgt_draw_initiator.num_indices) {
    return 0;
  }
  if (vgt_draw_initiator.source_select != xenos::SourceSelect::kDMA &&
      vgt_draw_initiator.source_select != xenos::SourceSelect::kAutoIndex) {
    // TODO(Triang3l): Support immediate indices.
    return xenos::kTexture2DCubeMaxWidthHeight;
  }

  // Not reproducing tessellation.
  if (xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type) &&
      regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select ==
          xenos::VGTOutputPath::kTessellationEnable) {
    return xenos::kTexture2DCubeMaxWidthHeight;
  }

  assert_true(vertex_shader.type() == xenos::ShaderType::kVertex);
  assert_true(vertex_shader.is_ucode_analyzed());
  if (!ShaderInterpreter::CanInterpretShader(vertex_shader)) {
    return xenos::kTexture2DCubeMaxWidthHeight;
  }

  auto vgt_dma_size = regs.Get<reg::VGT_DMA_SIZE>();
  union {
    const void* index_buffer;
    const uint16_t* index_buffer_16;
    const uint32_t* index_buffer_32;
  };
  xenos::Endian index_endian = vgt_dma_size.swap_mode;
  if (vgt_draw_initiator.source_select == xenos::SourceSelect::kDMA) {
    xenos::IndexFormat index_format = vgt_draw_initiator.index_size;
    uint32_t index_buffer_base = regs[XE_GPU_REG_VGT_DMA_BASE];
    if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16) {
      // Handle the index endianness to same way as the PrimitiveProcessor.
      if (index_endian == xenos::Endian::k8in32) {
        index_endian = xenos::Endian::k8in16;
      } else if (index_endian == xenos::Endian::k16in32) {
        index_endian = xenos::Endian::kNone;
      }
      index_buffer_base &= ~uint32_t(sizeof(uint16_t) - 1);
    } else {
      assert_true(vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32);
      index_buffer_base &= ~uint32_t(sizeof(uint32_t) - 1);
    }
    index_buffer = memory_.TranslatePhysical(index_buffer_base);
  }
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  uint32_t reset_index = regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx;
  uint32_t index_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
  uint32_t min_index = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;
  uint32_t max_index = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;

  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  float viewport_y_scale =
      pa_cl_vte_cntl.vport_y_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE) : 1.0f;
  float viewport_y_offset =
      pa_cl_vte_cntl.vport_y_offset_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET) : 0.0f;

  int32_t point_vertex_min_diameter_float = 0;
  int32_t point_vertex_max_diameter_float = 0;
  float point_constant_radius_y = 0.0f;
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    point_vertex_min_diameter_float =
        rex::memory::Reinterpret<int32_t>(float(pa_su_point_minmax.min_size) * (2.0f / 16.0f));
    point_vertex_max_diameter_float =
        rex::memory::Reinterpret<int32_t>(float(pa_su_point_minmax.max_size) * (2.0f / 16.0f));
    point_constant_radius_y = float(regs.Get<reg::PA_SU_POINT_SIZE>().height) * (1.0f / 16.0f);
  }

  float max_y = -FLT_MAX;

  shader_interpreter_.SetShader(vertex_shader);

  PositionYExportSink position_y_export_sink;
  shader_interpreter_.SetExportSink(&position_y_export_sink);
  for (uint32_t i = 0; i < vgt_draw_initiator.num_indices; ++i) {
    uint32_t vertex_index;
    if (vgt_draw_initiator.source_select == xenos::SourceSelect::kDMA) {
      if (i < vgt_dma_size.num_words) {
        if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16) {
          vertex_index = index_buffer_16[i];
        } else {
          vertex_index = index_buffer_32[i];
        }
        // The Xenos only uses 24 bits of the index (reset_indx is 24-bit).
        vertex_index = xenos::GpuSwap(vertex_index, index_endian) & 0xFFFFFF;
      } else {
        vertex_index = 0;
      }
      if (pa_su_sc_mode_cntl.multi_prim_ib_ena && vertex_index == reset_index) {
        continue;
      }
    } else {
      assert_true(vgt_draw_initiator.source_select == xenos::SourceSelect::kAutoIndex);
      vertex_index = i;
    }
    vertex_index =
        std::min(max_index, std::max(min_index, (vertex_index + index_offset) & 0xFFFFFF));

    position_y_export_sink.Reset();

    shader_interpreter_.temp_registers()[0] = float(vertex_index);
    shader_interpreter_.Execute();

    if (position_y_export_sink.vertex_kill().has_value() &&
        (position_y_export_sink.vertex_kill().value() & ~(UINT32_C(1) << 31))) {
      continue;
    }
    if (!position_y_export_sink.position_y().has_value()) {
      continue;
    }
    float vertex_y = position_y_export_sink.position_y().value();
    if (!pa_cl_vte_cntl.vtx_xy_fmt) {
      if (!position_y_export_sink.position_w().has_value()) {
        continue;
      }
      vertex_y /= position_y_export_sink.position_w().value();
    }

    vertex_y = vertex_y * viewport_y_scale + viewport_y_offset;

    if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
      float point_radius_y;
      if (position_y_export_sink.point_size().has_value()) {
        // Vertex-specified diameter. Clamped effectively as a signed integer in
        // the hardware, -NaN, -Infinity ... -0 to the minimum, +Infinity, +NaN
        // to the maximum.
        point_radius_y = 0.5f * rex::memory::Reinterpret<float>(std::min(
                                    point_vertex_max_diameter_float,
                                    std::max(point_vertex_min_diameter_float,
                                             rex::memory::Reinterpret<int32_t>(
                                                 position_y_export_sink.point_size().value()))));
      } else {
        // Constant radius.
        point_radius_y = point_constant_radius_y;
      }
      vertex_y += point_radius_y;
    }

    // std::max is `a < b ? b : a`, thus in case of NaN, the first argument is
    // always returned - max_y, which is initialized to a normalized value.
    max_y = std::max(max_y, vertex_y);
  }
  shader_interpreter_.SetExportSink(nullptr);

  int32_t max_y_24p8 = ui::FloatToD3D11Fixed16p8(max_y);
  // 16p8 range is -32768 to 32767+255/256, but it's stored as uint32_t here,
  // as 24p8, so overflowing up to -8388608 to 8388608+255/256 is safe. The
  // range of the window offset plus the half-pixel offset is -16384 to 16384.5,
  // so it's safe to add both - adding it will neither move the 16p8 clamping
  // bounds -32768 and 32767+255/256 into the 0...8192 screen space range, nor
  // cause 24p8 overflow.
  if (regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero) {
    max_y_24p8 += 128;
  }
  if (pa_su_sc_mode_cntl.vtx_window_offset_enable) {
    max_y_24p8 += regs.Get<reg::PA_SC_WINDOW_OFFSET>().window_y_offset * 256;
  }
  // Top-left rule - .5 exclusive without MSAA, 1. exclusive with MSAA.
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  return (uint32_t(std::max(int32_t(0), max_y_24p8)) +
          ((rb_surface_info.msaa_samples == xenos::MsaaSamples::k1X) ? 127 : 255)) >>
         8;
}

uint32_t DrawExtentEstimator::EstimateMaxY(bool try_to_estimate_vertex_max_y,
                                           const Shader& vertex_shader) {
  SCOPE_profile_cpu_f("gpu");

  const RegisterFile& regs = Regs();

  auto pa_sc_window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
  int32_t window_y_offset = pa_sc_window_offset.window_y_offset;

  // Scissor.
  auto pa_sc_window_scissor_br = regs.Get<reg::PA_SC_WINDOW_SCISSOR_BR>();
  int32_t scissor_bottom = int32_t(pa_sc_window_scissor_br.br_y);
  bool scissor_window_offset = !regs.Get<reg::PA_SC_WINDOW_SCISSOR_TL>().window_offset_disable;
  if (scissor_window_offset) {
    scissor_bottom += window_y_offset;
  }
  auto pa_sc_screen_scissor_br = regs.Get<reg::PA_SC_SCREEN_SCISSOR_BR>();
  scissor_bottom = std::min(scissor_bottom, int32_t(pa_sc_screen_scissor_br.br_y));
  uint32_t max_y = uint32_t(std::max(scissor_bottom, int32_t(0)));

  if (regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable) {
    // Actual extent from the vertices.
    if (try_to_estimate_vertex_max_y && REXCVAR_GET(execute_unclipped_draw_vs_on_cpu)) {
      bool estimate_vertex_max_y;
      if (REXCVAR_GET(execute_unclipped_draw_vs_on_cpu_with_scissor)) {
        estimate_vertex_max_y = true;
      } else {
        estimate_vertex_max_y = false;
        if (scissor_bottom >= xenos::kTexture2DCubeMaxWidthHeight) {
          // Handle just the usual special 8192x8192 case in Direct3D 9 - 8192
          // may be a normal render target height (80x8192 is well within the
          // EDRAM size, for instance), no need to process the vertices on the
          // CPU in this case.
          int32_t scissor_right = int32_t(pa_sc_window_scissor_br.br_x);
          if (scissor_window_offset) {
            scissor_right += pa_sc_window_offset.window_x_offset;
          }
          scissor_right = std::min(scissor_right, int32_t(pa_sc_screen_scissor_br.br_x));
          if (scissor_right >= xenos::kTexture2DCubeMaxWidthHeight) {
            estimate_vertex_max_y = true;
          }
        }
      }
      if (estimate_vertex_max_y) {
        max_y = std::min(max_y, EstimateVertexMaxY(vertex_shader));
      }
    }
  } else {
    // Viewport. Though the Xenos itself doesn't have an implicit viewport
    // scissor (it's set by Direct3D 9 when a viewport is used), on hosts, it
    // usually exists and can't be disabled.
    auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
    float viewport_bottom = 0.0f;
    // First calculate all the integer.0 or integer.5 offsetting exactly at full
    // precision.
    if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().vtx_window_offset_enable) {
      viewport_bottom += float(window_y_offset);
    }
    if (regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero) {
      viewport_bottom += 0.5f;
    }
    // Then apply the floating-point viewport offset.
    if (pa_cl_vte_cntl.vport_y_offset_ena) {
      viewport_bottom += regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET);
    }
    viewport_bottom += pa_cl_vte_cntl.vport_y_scale_ena
                           ? std::abs(regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE))
                           : 1.0f;
    // Using floor, or, rather, truncation (because maxing with zero anyway)
    // similar to how viewport scissoring behaves on real AMD, Intel and Nvidia
    // GPUs on Direct3D 12 (but not WARP), also like in
    // draw_util::GetHostViewportInfo.
    // max(0.0f, viewport_bottom) to drop NaN and < 0 - max picks the first
    // argument in the !(a < b) case (always for NaN), min as float (max_y is
    // well below 2^24) to safely drop very large values.
    max_y = uint32_t(std::min(float(max_y), std::max(0.0f, viewport_bottom)));
  }

  return max_y;
}

// =================================================================================================================
// wwe2k14 patch (Session 7, EXTENTS-SHADOW): CPU-computed screen extents for predicated tiling, shadow mode.
//
// Why. WWE 2K14 draws its main 3D scene in four 320-pixel tiles (predicated tiling). In tile 0 the game wraps each
// draw in SCREEN_EXT_INIT ... draw ... SCREEN_EXT_RPT (EVENT_WRITE_EXT); the GPU reports the draw's screen extents
// and the game turns them into per-tile masks, so tiles 1-3 skip draws that don't touch them. ReXGlue reports
// full-screen extents, so every draw runs in every tile (docs/research/predicated_tiling.md). Here the command
// processor computes a conservative bounding box for the draws of selected vertex shaders (the crowd) and, in
// mode 1, only counts what tiles 1-3 would skip; the game still gets the full-screen values.
//
// How (ScreenExtentsEstimator::Estimate).
// - The vertex shader runs once per draw, not once per vertex, in affine arithmetic: every value is
//   c + sum(k_i * e_i) + r * e', where each e_i in [-1, 1] is a named input (the vertex index, or one component
//   of one vertex fetch, whose range over all the vertices of the draw is read from guest memory) and r >= 0 is
//   an anonymous term covering everything else (products of inputs, the float32 rounding of every operation,
//   denormal flushing). Additions and multiplications by known values are exact in this form, so a position
//   computed as view-projection * (world * fetched position) - the crowd shaders C1028A64EFCF8229 and
//   183CC64A4C4E96AC - keeps its exact linear dependence on the vertex data. Anything non-linear (min/max that
//   isn't decided, cnd on an unknown condition, rcp, ...) becomes an interval; anything not modelled (texture
//   fetches, kill, cube, ...) becomes "unbounded", which only matters if the position depends on it.
// - Values that depend only on constants are exact points, computed with float32 operations in the host shader's
//   order (correctly rounded add/mul, as Vulkan requires), so comparisons and selections on them match the GPU.
// - Only instructions the position can depend on are evaluated (a backward liveness pass per shader, done once);
//   the others write "unbounded" to their destinations, so a liveness mistake can only cause a fallback. For the
//   crowd that is the position fetch, the world and view-projection dp4s and one cndeq (the lighting is skipped).
// - Semantics follow the host translator (spirv_translator_alu.cpp, spirv_translator_fetch.cpp, the vertex index
//   path in spirv_translator.cpp): the SM3 "+-0 * x = 0" rule, max(a, a) as a move, the vertex index offset, 24-bit
//   mask and clamp, primitive reset, fetch addressing and format conversion, float constants read from the
//   register file like the Vulkan backend (c0-c255 at SHADER_CONSTANT_000), control flow like ShaderInterpreter.
// - Screen X = (x / w) * PA_CL_VPORT_XSCALE + XOFFSET (Y alike). x / w over the box of inputs is a
//   linear-fractional function; with w > 0 on the whole box its maximum is at a vertex of the box, found with
//   Dinkelbach's iteration and certified (g(t) = x0 - t w0 + sum |x_i - t w_i| <= 0 proves x / w <= t everywhere).
// - Then one pixel of margin (half-pixel offset, rasterizer snapping, MSAA sample positions), 8-pixel cells (the
//   unit EVENT_WRITE_EXT uses; the game reads pixels 8 * min ... 8 * max + 7), one more cell on every side, clamped
//   to 0-1023 (the game compares unsigned, so a negative minimum must never be written).
//
// Why the result is never smaller than what the host rasterizes.
// - The box holds every vertex's inputs: the ranges are the min / max over exactly the vertices the draw fetches
//   (same index list, offset, clamp and reset handling as the host; fetches outside the fetch constant's buffer,
//   which the host may not have uploaded, give up).
// - Affine arithmetic is inclusion-preserving: for every vertex there is a choice of the e's in [-1, 1] that gives
//   that vertex's value of every intermediate, including the position; the bounds of x / w are taken over all
//   choices, so they contain every vertex. float32 rounding is covered by the 2^-23 relative term added to every
//   operation (float32 rounding is at most 2^-24 relative) plus 1.2e-38 absolute (denormal flushing).
// - A triangle (or quad / polygon) whose vertices all have w > 0 projects to the convex hull of its projected
//   vertices; clipping, culling, the scissor and depth only remove samples. W <= 0 anywhere (or too close to 0)
//   gives up.
// - Everything else gives up and reports full screen (ScreenExtentsFallback): data the GPU wrote this session
//   (guest memory may be stale), non-polygon primitives, X/Y pre-divided by W, a non-zero window offset,
//   data-dependent control flow, loops, relative register addressing, memexport, unknown opcodes.
// gpu_extents_mode_verify checks the claim at runtime against ShaderInterpreter, vertex by vertex.
// =================================================================================================================

namespace {

constexpr uint32_t kExtentsSymbolCount = 24;
// float32 rounding of one operation, relative to its result's magnitude (round to nearest is at most 2^-24).
constexpr double kExtentsFloatRound = 1.0 / 8388608.0;
// Absolute slack for denormal flushing (FLT_MIN is 1.1755e-38).
constexpr double kExtentsFloatMin = 1.2e-38;
// Relative slack for the double arithmetic of the bounds themselves.
constexpr double kExtentsDoubleSlack = 1.0 / 1125899906842624.0;  // 2^-50
// Relative error of fetched integer / packed formats converted and normalized in float32 on the host.
constexpr double kExtentsFetchRound = 1.0 / 2097152.0;  // 2^-21
// Magnitudes past this are treated as unbounded (float32 overflow).
constexpr double kExtentsHuge = 1.0e30;
constexpr uint32_t kExtentsMaxCfSteps = 4096;
constexpr uint32_t kExtentsTempRegisters = 64;

// One component of one register in affine form (see the comment above).
struct Affine {
  // Nothing is known (NaN, infinity, or an operation that isn't modelled).
  bool top;
  // Exactly the vertex index the host shader starts with in r0.x (only copies keep this).
  bool vertex_index;
  // Named inputs with a coefficient.
  uint32_t mask;
  double c;
  // Radius of the anonymous term, >= 0.
  double err;
  double coef[kExtentsSymbolCount];
};

void AffineTop(Affine& v) {
  v.top = true;
  v.vertex_index = false;
  v.mask = 0;
  v.c = 0.0;
  v.err = 0.0;
}

void AffinePoint(Affine& v, double value) {
  v.top = false;
  v.vertex_index = false;
  v.mask = 0;
  v.c = value;
  v.err = 0.0;
}

bool AffineIsPoint(const Affine& v) { return !v.top && !v.mask && v.err == 0.0; }

void AffineAssign(Affine& dst, const Affine& src) {
  if (&dst == &src) {
    return;
  }
  dst.top = src.top;
  dst.vertex_index = src.vertex_index;
  dst.mask = src.mask;
  dst.c = src.c;
  dst.err = src.err;
  for (uint32_t bits = src.mask; bits; bits &= bits - 1) {
    uint32_t i = uint32_t(std::countr_zero(bits));
    dst.coef[i] = src.coef[i];
  }
}

double AffineRadius(const Affine& v) {
  double r = v.err;
  for (uint32_t bits = v.mask; bits; bits &= bits - 1) {
    r += std::abs(v.coef[std::countr_zero(bits)]);
  }
  return r + (std::abs(v.c) + r) * kExtentsDoubleSlack;
}

void AffineBounds(const Affine& v, double& lo, double& hi) {
  double r = AffineRadius(v);
  lo = v.c - r;
  hi = v.c + r;
}

void AffineInterval(Affine& v, double lo, double hi) {
  if (!(lo <= hi) || !(std::abs(lo) <= kExtentsHuge) || !(std::abs(hi) <= kExtentsHuge)) {
    AffineTop(v);
    return;
  }
  if (lo == hi && double(float(lo)) == lo) {
    AffinePoint(v, lo);
    return;
  }
  v.top = false;
  v.vertex_index = false;
  v.mask = 0;
  v.c = 0.5 * lo + 0.5 * hi;
  v.err = 0.5 * hi - 0.5 * lo;
  v.err += (std::abs(v.c) + v.err) * kExtentsDoubleSlack + kExtentsFloatMin;
}

// A float32 value as the GPU has it: non-finite -> unbounded; a denormal may be flushed to 0 -> [0, value].
void AffineFromFloat(Affine& v, float value) {
  if (!std::isfinite(value)) {
    AffineTop(v);
    return;
  }
  if (value != 0.0f && std::abs(value) < FLT_MIN) {
    AffineInterval(v, std::min(0.0, double(value)), std::max(0.0, double(value)));
    return;
  }
  AffinePoint(v, double(value));
}

// After an arithmetic operation: adds the float32 rounding of the result, gives up on huge magnitudes.
void AffineFinish(Affine& v) {
  double r = v.err;
  for (uint32_t bits = v.mask; bits; bits &= bits - 1) {
    r += std::abs(v.coef[std::countr_zero(bits)]);
  }
  double magnitude = std::abs(v.c) + r;
  if (!(magnitude <= kExtentsHuge)) {
    AffineTop(v);
    return;
  }
  v.top = false;
  v.vertex_index = false;
  v.err += magnitude * kExtentsFloatRound + kExtentsFloatMin;
}

// out may alias a or b in every operation below.

void AffineAdd(const Affine& a, const Affine& b, bool subtract, Affine& out) {
  if (a.top || b.top) {
    AffineTop(out);
    return;
  }
  if (AffineIsPoint(a) && AffineIsPoint(b)) {
    float fa = float(a.c), fb = float(b.c);
    float r = subtract ? fa - fb : fa + fb;
    AffineFromFloat(out, r);
    return;
  }
  const double s = subtract ? -1.0 : 1.0;
  const uint32_t am = a.mask, bm = b.mask;
  const double c = a.c + s * b.c;
  const double err = a.err + b.err;
  for (uint32_t bits = am | bm; bits; bits &= bits - 1) {
    uint32_t i = uint32_t(std::countr_zero(bits));
    double ca = ((am >> i) & 1) ? a.coef[i] : 0.0;
    double cb = ((bm >> i) & 1) ? b.coef[i] : 0.0;
    out.coef[i] = ca + s * cb;
  }
  out.mask = am | bm;
  out.c = c;
  out.err = err;
  AffineFinish(out);
}

// zero_rule: Shader Model 3, +-0 (or a denormal) * anything = +0 (the host applies it unless both operands are
// the same register component).
void AffineMul(const Affine& a, const Affine& b, bool zero_rule, Affine& out) {
  if (zero_rule && ((AffineIsPoint(a) && a.c == 0.0) || (AffineIsPoint(b) && b.c == 0.0))) {
    AffinePoint(out, 0.0);
    return;
  }
  if (a.top || b.top) {
    AffineTop(out);
    return;
  }
  if (AffineIsPoint(a) && AffineIsPoint(b)) {
    float r = float(a.c) * float(b.c);
    AffineFromFloat(out, r);
    return;
  }
  const uint32_t am = a.mask, bm = b.mask;
  const double ac = a.c, bc = b.c, ae = a.err, be = b.err;
  const double ra = AffineRadius(a), rb = AffineRadius(b);
  for (uint32_t bits = am | bm; bits; bits &= bits - 1) {
    uint32_t i = uint32_t(std::countr_zero(bits));
    double ca = ((am >> i) & 1) ? a.coef[i] : 0.0;
    double cb = ((bm >> i) & 1) ? b.coef[i] : 0.0;
    out.coef[i] = ac * cb + bc * ca;
  }
  out.mask = am | bm;
  out.c = ac * bc;
  // x0 y0 + sum (x0 y_i + y0 x_i) e_i + x0 * (y's anonymous term) + y0 * (x's) + (x - x0)(y - y0), the last
  // bounded by radius(x) * radius(y); plus an operand flushed to 0 (the product's change is below FLT_MIN times
  // the other operand).
  out.err = std::abs(ac) * be + std::abs(bc) * ae + ra * rb +
            kExtentsFloatMin * ((std::abs(ac) + ra) + (std::abs(bc) + rb));
  AffineFinish(out);
}

void AffineNegate(Affine& v) {
  v.vertex_index = false;
  if (v.top) {
    return;
  }
  v.c = -v.c;
  for (uint32_t bits = v.mask; bits; bits &= bits - 1) {
    uint32_t i = uint32_t(std::countr_zero(bits));
    v.coef[i] = -v.coef[i];
  }
}

void AffineAbs(Affine& v) {
  v.vertex_index = false;
  if (v.top) {
    return;
  }
  if (AffineIsPoint(v)) {
    v.c = std::abs(v.c);
    return;
  }
  double lo, hi;
  AffineBounds(v, lo, hi);
  if (lo >= 0.0) {
    return;
  }
  if (hi <= 0.0) {
    AffineNegate(v);
    return;
  }
  AffineInterval(v, 0.0, std::max(-lo, hi));
}

void AffineSaturate(Affine& v) {
  if (v.top) {
    // NaN saturates to a value in [0, 1] too.
    AffineInterval(v, 0.0, 1.0);
    return;
  }
  if (AffineIsPoint(v)) {
    AffinePoint(v, double(std::min(std::max(float(v.c), 0.0f), 1.0f)));
    return;
  }
  double lo, hi;
  AffineBounds(v, lo, hi);
  if (lo >= 0.0 && hi <= 1.0) {
    v.vertex_index = false;
    return;
  }
  AffineInterval(v, std::min(std::max(lo, 0.0), 1.0), std::min(std::max(hi, 0.0), 1.0));
}

void AffineHull(const Affine& a, const Affine& b, Affine& out) {
  if (a.top || b.top) {
    AffineTop(out);
    return;
  }
  if (AffineIsPoint(a) && AffineIsPoint(b) && a.c == b.c) {
    AffinePoint(out, a.c);
    return;
  }
  double alo, ahi, blo, bhi;
  AffineBounds(a, alo, ahi);
  AffineBounds(b, blo, bhi);
  AffineInterval(out, std::min(alo, blo), std::max(ahi, bhi));
}

// Shader Model 3 max / min (a >= b ? a : b, a < b ? a : b) of different operands.
void AffineMaxMin(const Affine& a, const Affine& b, bool is_max, Affine& out) {
  if (a.top || b.top) {
    AffineTop(out);
    return;
  }
  if (AffineIsPoint(a) && AffineIsPoint(b)) {
    bool pick_a = is_max ? (a.c >= b.c) : (a.c < b.c);
    AffinePoint(out, pick_a ? a.c : b.c);
    return;
  }
  double alo, ahi, blo, bhi;
  AffineBounds(a, alo, ahi);
  AffineBounds(b, blo, bhi);
  if (is_max) {
    if (alo >= bhi) {
      AffineAssign(out, a);
    } else if (blo > ahi) {
      AffineAssign(out, b);
    } else {
      AffineInterval(out, std::max(alo, blo), std::max(ahi, bhi));
      return;
    }
  } else {
    if (ahi < blo) {
      AffineAssign(out, a);
    } else if (alo >= bhi) {
      AffineAssign(out, b);
    } else {
      AffineInterval(out, std::min(alo, blo), std::min(ahi, bhi));
      return;
    }
  }
  out.vertex_index = false;
}

enum class AffineCmp { kEq, kNe, kGt, kGe };

// 1: true for every value, 0: false for every value, -1: unknown.
int AffineCompare(const Affine& a, const Affine& b, AffineCmp cmp) {
  if (a.top || b.top) {
    return -1;
  }
  if (AffineIsPoint(a) && AffineIsPoint(b)) {
    switch (cmp) {
      case AffineCmp::kEq:
        return a.c == b.c;
      case AffineCmp::kNe:
        return a.c != b.c;
      case AffineCmp::kGt:
        return a.c > b.c;
      case AffineCmp::kGe:
        return a.c >= b.c;
    }
  }
  double alo, ahi, blo, bhi;
  AffineBounds(a, alo, ahi);
  AffineBounds(b, blo, bhi);
  bool disjoint = ahi < blo || bhi < alo;
  switch (cmp) {
    case AffineCmp::kEq:
      return disjoint ? 0 : -1;
    case AffineCmp::kNe:
      return disjoint ? 1 : -1;
    case AffineCmp::kGt:
      return alo > bhi ? 1 : (ahi <= blo ? 0 : -1);
    case AffineCmp::kGe:
      return alo >= bhi ? 1 : (ahi < blo ? 0 : -1);
  }
  return -1;
}

void AffineFromDecision(Affine& out, int decision, double if_true, double if_false) {
  if (decision < 0) {
    AffineInterval(out, std::min(if_true, if_false), std::max(if_true, if_false));
  } else {
    AffinePoint(out, decision ? if_true : if_false);
  }
}

enum class AffineRounding { kFloor, kTrunc, kFract };

void AffineRound(const Affine& a, AffineRounding kind, Affine& out) {
  if (a.top) {
    AffineTop(out);
    return;
  }
  if (AffineIsPoint(a)) {
    float f = float(a.c);
    float r;
    switch (kind) {
      case AffineRounding::kFloor:
        r = std::floor(f);
        break;
      case AffineRounding::kTrunc:
        r = std::trunc(f);
        break;
      default: {
        float f_floor = std::floor(f);
        r = f - f_floor;
      } break;
    }
    AffineFromFloat(out, r);
    return;
  }
  double lo, hi;
  AffineBounds(a, lo, hi);
  switch (kind) {
    case AffineRounding::kFloor:
      AffineInterval(out, std::floor(lo), std::floor(hi));
      break;
    case AffineRounding::kTrunc:
      AffineInterval(out, std::trunc(lo), std::trunc(hi));
      break;
    default:
      AffineInterval(out, 0.0, 1.0);
      break;
  }
}

enum class AffineFunction { kExp2, kLog2, kRcp, kRsq, kSqrt, kSin, kCos };

// The host's transcendental functions aren't correctly rounded, so even exact inputs give a (small) interval.
// The host calls GLSL.std.450 Exp2 / Log2 / InverseSqrt / Sqrt / Sin / Cos and divides 1 / x; Vulkan's precision
// rules for those (core spec, "Precision and Operation of SPIR-V Instructions"):
// - exp2: 3 + 2 |x| ULP. ulp(v) <= |v| 2^-23 + 2^-149 (denormals), so the error is at most |f| rel + absolute with
//   rel = (3 + 2 M) 2^-23 and absolute (3 + 2 M) 2^-149, M = max |x| over the range (4 + 2 M is used; a flushed
//   denormal result, off by at most 2^-126, is inside the 1e-30 floor).
// - log2: 3 ULP outside [0.5, 2]; inside it, an absolute error below 2^-21 (the result is near 0 there).
// - 1 / x: 2.5 ULP for |x| in [2^-126, 2^126]. Always the case when this returns a value: magnitudes past 1e30 are
//   already unbounded, and |x| < 2^-126 gives a result past 1e30, which AffineInterval turns into unbounded.
// - inversesqrt: 2 ULP; sqrt: inherited from 1 / inversesqrt (about 4.5 ULP).
// - sin / cos: 2^-11 absolute inside [-pi, pi]; nothing is promised outside it, so that gives up (unbounded).
// Default: rel = 2^-16 (128 ULP, which also covers the double evaluation here), absolute 1e-30.
// f is monotone on [lo, hi] (sin / cos use [-1, 1]), so every exact result v is in [rlo, rhi], and the host's is in
// [v - |v| rel - absolute, v + |v| rel + absolute]. v - |v| rel is concave and v + |v| rel convex in v, so their
// extremes over [rlo, rhi] are at the ends: the bounds below hold every host result, for any rel >= 0.
void AffineTranscendental(const Affine& a, AffineFunction function, Affine& out) {
  if (a.top) {
    AffineTop(out);
    return;
  }
  double lo, hi;
  if (AffineIsPoint(a)) {
    lo = hi = a.c;
  } else {
    AffineBounds(a, lo, hi);
  }
  double rlo = 0.0, rhi = 0.0, rel = 1.0 / 65536.0, absolute = 1.0e-30;
  bool ok = true;
  switch (function) {
    case AffineFunction::kExp2:
      ok = hi <= 127.0;
      if (ok) {
        rlo = std::exp2(lo);
        rhi = std::exp2(hi);
        double ulps = 4.0 + 2.0 * std::max(std::abs(lo), std::abs(hi));
        rel = std::max(rel, ulps * (1.0 / 8388608.0));
        absolute += ulps * 1.401298464324817e-45;  // 2^-149
      }
      break;
    case AffineFunction::kLog2:
      ok = lo > 0.0;
      if (ok) {
        rlo = std::log2(lo);
        rhi = std::log2(hi);
        if (lo <= 2.0 && hi >= 0.5) {
          absolute = 1.0 / 1048576.0;  // 2^-20, twice the 2^-21 allowed inside [0.5, 2]
        }
      }
      break;
    case AffineFunction::kRcp:
      ok = lo > 0.0 || hi < 0.0;
      if (ok) {
        rlo = 1.0 / hi;
        rhi = 1.0 / lo;
      }
      break;
    case AffineFunction::kRsq:
      ok = lo > 0.0;
      if (ok) {
        rlo = 1.0 / std::sqrt(hi);
        rhi = 1.0 / std::sqrt(lo);
      }
      break;
    case AffineFunction::kSqrt:
      ok = lo >= 0.0;
      if (ok) {
        rlo = std::sqrt(lo);
        rhi = std::sqrt(hi);
      }
      break;
    case AffineFunction::kSin:
    case AffineFunction::kCos:
      ok = std::abs(lo) <= 3.14159 && std::abs(hi) <= 3.14159;
      if (ok) {
        if (lo == hi) {
          rlo = rhi = function == AffineFunction::kSin ? std::sin(lo) : std::cos(lo);
        } else {
          rlo = -1.0;
          rhi = 1.0;
        }
        absolute = 1.0 / 1024.0;  // twice the 2^-11 allowed
      }
      break;
  }
  if (!ok || !std::isfinite(rlo) || !std::isfinite(rhi)) {
    AffineTop(out);
    return;
  }
  double out_lo = std::min(rlo - std::abs(rlo) * rel, rhi - std::abs(rhi) * rel) - absolute;
  double out_hi = std::max(rlo + std::abs(rlo) * rel, rhi + std::abs(rhi) * rel) + absolute;
  AffineInterval(out, out_lo, out_hi);
}

// Bounds of (a0 + sum a_i e_i) / (b0 + sum b_i e_i) over e in [-1, 1]^n, with b0 - sum |b_i| > 0: the maximum
// (Dinkelbach's iteration from e = 0), certified by g(t) = a0 - t b0 + sum |a_i - t b_i| <= 0, which holds exactly
// when the ratio is <= t for every e.
bool MaxLinearFractional(const double* a, const double* b, uint32_t n, double a0, double b0, double& out) {
  double t = a0 / b0;
  for (uint32_t iteration = 0; iteration < 64; ++iteration) {
    double numerator = a0, denominator = b0;
    for (uint32_t i = 0; i < n; ++i) {
      double s = (a[i] - t * b[i]) >= 0.0 ? 1.0 : -1.0;
      numerator += s * a[i];
      denominator += s * b[i];
    }
    double t_next = numerator / denominator;
    if (!(t_next > t)) {
      break;
    }
    t = t_next;
  }
  double slack = 1.0e-9 * (1.0 + std::abs(t));
  for (uint32_t attempt = 0; attempt < 4; ++attempt, slack *= 1024.0) {
    double t_up = t + slack;
    double g = a0 - t_up * b0;
    for (uint32_t i = 0; i < n; ++i) {
      g += std::abs(a[i] - t_up * b[i]);
    }
    if (g <= 0.0) {
      out = t_up;
      return std::isfinite(t_up);
    }
  }
  return false;
}

// IEEE half to float, like GLSL unpackHalf2x16 on the host (not the Xenos extended range).
double HalfToDouble(uint32_t half) {
  uint32_t exponent = (half >> 10) & 0x1F;
  uint32_t mantissa = half & 0x3FF;
  double value;
  if (exponent == 0) {
    value = std::ldexp(double(mantissa), -24);
  } else if (exponent == 31) {
    value = mantissa ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
  } else {
    value = std::ldexp(double(mantissa | 0x400), int(exponent) - 25);
  }
  return (half & 0x8000) ? -value : value;
}

bool IsFloatVertexFormat(xenos::VertexFormat format) {
  switch (format) {
    case xenos::VertexFormat::k_32_FLOAT:
    case xenos::VertexFormat::k_32_32_FLOAT:
    case xenos::VertexFormat::k_32_32_32_FLOAT:
    case xenos::VertexFormat::k_32_32_32_32_FLOAT:
    case xenos::VertexFormat::k_16_16_FLOAT:
    case xenos::VertexFormat::k_16_16_16_16_FLOAT:
      return true;
    default:
      return false;
  }
}

// The host's conversion (spirv_translator_fetch.cpp) of one vertex's words (already endian-swapped, indexed by
// word) to the used components of the format. Exact for the float formats; the others carry float32 rounding.
bool DecodeVertexFormat(const ParsedVertexFetchInstruction::Attributes& attributes, uint32_t used_components,
                        const uint32_t* words, double* out) {
  switch (attributes.data_format) {
    case xenos::VertexFormat::k_32_FLOAT:
    case xenos::VertexFormat::k_32_32_FLOAT:
    case xenos::VertexFormat::k_32_32_32_FLOAT:
    case xenos::VertexFormat::k_32_32_32_32_FLOAT:
      for (uint32_t j = 0; j < 4; ++j) {
        if (used_components & (1u << j)) {
          float value;
          std::memcpy(&value, &words[j], sizeof(value));
          out[j] = double(value);
        }
      }
      break;
    case xenos::VertexFormat::k_16_16_FLOAT:
    case xenos::VertexFormat::k_16_16_16_16_FLOAT:
      for (uint32_t j = 0; j < 4; ++j) {
        if (used_components & (1u << j)) {
          out[j] = HalfToDouble((words[j >> 1] >> ((j & 1) * 16)) & 0xFFFF);
        }
      }
      break;
    case xenos::VertexFormat::k_32:
    case xenos::VertexFormat::k_32_32:
    case xenos::VertexFormat::k_32_32_32_32:
      for (uint32_t j = 0; j < 4; ++j) {
        if (!(used_components & (1u << j))) {
          continue;
        }
        double value = attributes.is_signed ? double(int32_t(words[j])) : double(words[j]);
        if (!attributes.is_integer) {
          if (attributes.is_signed) {
            value = attributes.signed_rf_mode == xenos::SignedRepeatingFractionMode::kNoZero
                        ? (value + 0.5) / 2147483647.5
                        : value / 2147483647.0;
          } else {
            value /= 4294967295.0;
          }
        }
        out[j] = value;
      }
      break;
    case xenos::VertexFormat::k_8_8_8_8:
    case xenos::VertexFormat::k_2_10_10_10:
    case xenos::VertexFormat::k_10_11_11:
    case xenos::VertexFormat::k_11_11_10:
    case xenos::VertexFormat::k_16_16:
    case xenos::VertexFormat::k_16_16_16_16: {
      uint32_t widths[4] = {}, offsets[4] = {}, word_index[4] = {};
      switch (attributes.data_format) {
        case xenos::VertexFormat::k_8_8_8_8:
          widths[0] = widths[1] = widths[2] = widths[3] = 8;
          offsets[1] = 8;
          offsets[2] = 16;
          offsets[3] = 24;
          break;
        case xenos::VertexFormat::k_2_10_10_10:
          widths[0] = widths[1] = widths[2] = 10;
          widths[3] = 2;
          offsets[1] = 10;
          offsets[2] = 20;
          offsets[3] = 30;
          break;
        case xenos::VertexFormat::k_10_11_11:
          widths[0] = widths[1] = 11;
          widths[2] = 10;
          offsets[1] = 11;
          offsets[2] = 22;
          break;
        case xenos::VertexFormat::k_11_11_10:
          widths[0] = 10;
          widths[1] = widths[2] = 11;
          offsets[1] = 10;
          offsets[2] = 21;
          break;
        case xenos::VertexFormat::k_16_16:
          widths[0] = widths[1] = 16;
          offsets[1] = 16;
          break;
        default:  // k_16_16_16_16
          widths[0] = widths[1] = widths[2] = widths[3] = 16;
          offsets[1] = offsets[3] = 16;
          word_index[2] = word_index[3] = 1;
          break;
      }
      for (uint32_t j = 0; j < 4; ++j) {
        if (!(used_components & (1u << j))) {
          continue;
        }
        uint32_t width = widths[j];
        if (!width) {
          return false;
        }
        uint32_t raw = (words[word_index[j]] >> offsets[j]) & ((width < 32) ? ((1u << width) - 1) : ~0u);
        double value;
        if (attributes.is_signed) {
          int32_t extended = int32_t(raw << (32 - width)) >> (32 - width);
          value = double(extended);
        } else {
          value = double(raw);
        }
        if (!attributes.is_integer) {
          if (attributes.is_signed) {
            double scale_inv = double((1u << (width - 1)) - 1);
            if (attributes.signed_rf_mode == xenos::SignedRepeatingFractionMode::kNoZero) {
              scale_inv += 0.5;
              value = value / scale_inv + 0.5 / scale_inv;
            } else {
              value = std::max(value / scale_inv, -1.0);
            }
          } else {
            value /= double((uint64_t(1) << width) - 1);
          }
        }
        out[j] = value;
      }
    } break;
    default:
      return false;
  }
  if (attributes.exp_adjust) {
    for (uint32_t j = 0; j < 4; ++j) {
      if (used_components & (1u << j)) {
        out[j] = std::ldexp(out[j], attributes.exp_adjust);
      }
    }
  }
  return true;
}

// A vertex shader parsed once (by ucode hash), the way the host translator sees it.
struct ExtentsProgram {
  enum class SlotKind : uint8_t { kNone, kAlu, kVertexFetch, kTextureFetch };
  struct Slot {
    SlotKind kind = SlotKind::kNone;
    uint32_t index = 0;
  };
  // Static reason the shader can't be evaluated (kNone if it can).
  ScreenExtentsFallback fallback = ScreenExtentsFallback::kNone;
  uint32_t cf_count = 0;
  // One per ucode instruction (3 dwords) address.
  std::vector<Slot> slots;
  std::vector<ParsedAluInstruction> alu;
  std::vector<ParsedVertexFetchInstruction> vfetch;
  // For each vfetch: the full fetch it takes its address from (itself for a full fetch).
  std::vector<uint32_t> vfetch_full;
  std::vector<ParsedTextureFetchInstruction> tfetch;
  // Per instruction address: whether the position can depend on it (MarkNeededInstructions). The others aren't
  // evaluated: they write "unbounded" to their destinations, so if this analysis were wrong the position would
  // come out unbounded (a fallback), never wrong.
  std::vector<uint8_t> needed;
  // ShaderInterpreter (gpu_extents_mode_verify) emulates this shader like the host does: no abs() on constants
  // (it ignores it), no packed / integer vertex formats (it decodes them differently), no relative constants.
  bool verify_comparable = true;
  // Highest float constant read with absolute addressing.
  uint32_t max_constant = 0;
};

// Backward liveness over the instructions in static control flow order (the order they run in without jumps or
// calls; with those, everything is needed). Live: register components (64 x 4 bits) and the previous scalar.
// Needed: writes the position, the predicate or a0 (always, they're rare and affect control flow / addressing),
// or something live. A needed instruction's register operands become live; an unconditional write (not in a
// conditional exec, not predicated) ends a component's liveness. A needed vfetch_mini makes its vfetch_full needed
// (the full fetch checks the index).
void MarkNeededInstructions(ExtentsProgram& program, const std::vector<std::pair<uint32_t, bool>>& order,
                            bool has_jumps) {
  program.needed.assign(program.slots.size(), has_jumps ? 1 : 0);
  if (has_jumps) {
    return;
  }
  uint64_t live[4] = {};  // bit r of live[c]: register r component c
  bool live_ps = false;
  bool everything_live = false;
  std::vector<uint8_t> full_needed_by_mini(program.vfetch.size(), 0);
  // components: the operand's components the operation reads (bit i = GetComponent(i)).
  auto make_operand_live = [&](const InstructionOperand& operand, uint32_t components) {
    if (operand.storage_source != InstructionStorageSource::kRegister) {
      return;
    }
    if (operand.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute ||
        operand.storage_index >= kExtentsTempRegisters) {
      everything_live = true;
      return;
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (!(components & (1u << i))) {
        continue;
      }
      SwizzleSource source = operand.GetComponent(i);
      if (source >= SwizzleSource::kX && source <= SwizzleSource::kW) {
        live[uint32_t(source)] |= uint64_t(1) << operand.storage_index;
      }
    }
  };
  // Whether the result writes a live component; kills them if unconditional.
  auto result_live = [&](const InstructionResult& result, bool& writes_position) -> bool {
    uint32_t mask = result.GetUsedWriteMask();
    if (!mask) {
      return false;
    }
    if (result.storage_target == InstructionStorageTarget::kPosition) {
      writes_position = true;
      return true;
    }
    if (result.storage_target != InstructionStorageTarget::kRegister) {
      return false;
    }
    if (result.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute ||
        result.storage_index >= kExtentsTempRegisters) {
      return true;
    }
    if (everything_live) {
      return true;
    }
    for (uint32_t c = 0; c < 4; ++c) {
      if ((mask & (1u << c)) && ((live[c] >> result.storage_index) & 1)) {
        return true;
      }
    }
    return false;
  };
  auto kill = [&](const InstructionResult& result) {
    uint32_t mask = result.GetUsedWriteMask();
    if (result.storage_target != InstructionStorageTarget::kRegister ||
        result.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute ||
        result.storage_index >= kExtentsTempRegisters) {
      return;
    }
    for (uint32_t c = 0; c < 4; ++c) {
      if (mask & (1u << c)) {
        live[c] &= ~(uint64_t(1) << result.storage_index);
      }
    }
  };
  for (size_t i = order.size(); i-- > 0;) {
    const uint32_t address = order[i].first;
    const bool conditional_exec = order[i].second;
    const ExtentsProgram::Slot& slot = program.slots[address];
    switch (slot.kind) {
      case ExtentsProgram::SlotKind::kAlu: {
        const ParsedAluInstruction& instr = program.alu[slot.index];
        if (instr.IsNop()) {
          break;
        }
        // The two halves separately: a half that isn't needed may compute from registers that aren't live (they
        // hold correct values or "unbounded", never wrong ones), its destination isn't live either.
        bool writes_position = false;
        bool vector_needed = result_live(instr.vector_and_constant_result, writes_position);
        bool scalar_needed = result_live(instr.scalar_result, writes_position);
        const bool writes_ps = instr.scalar_opcode != ucode::AluScalarOpcode::kRetainPrev;
        scalar_needed = scalar_needed || (writes_ps && live_ps);
        switch (instr.vector_opcode) {
          case ucode::AluVectorOpcode::kSetpEqPush:
          case ucode::AluVectorOpcode::kSetpNePush:
          case ucode::AluVectorOpcode::kSetpGtPush:
          case ucode::AluVectorOpcode::kSetpGePush:
          case ucode::AluVectorOpcode::kMaxA:
            vector_needed = true;
            break;
          default:
            break;
        }
        switch (instr.scalar_opcode) {
          case ucode::AluScalarOpcode::kSetpEq:
          case ucode::AluScalarOpcode::kSetpNe:
          case ucode::AluScalarOpcode::kSetpGt:
          case ucode::AluScalarOpcode::kSetpGe:
          case ucode::AluScalarOpcode::kSetpInv:
          case ucode::AluScalarOpcode::kSetpPop:
          case ucode::AluScalarOpcode::kSetpClr:
          case ucode::AluScalarOpcode::kSetpRstr:
          case ucode::AluScalarOpcode::kMaxAs:
          case ucode::AluScalarOpcode::kMaxAsf:
            scalar_needed = true;
            break;
          default:
            break;
        }
        if (!vector_needed && !scalar_needed) {
          break;
        }
        program.needed[address] = 1;
        // A needed instruction runs whole, so both halves' destinations are written.
        if (!conditional_exec && !instr.is_predicated) {
          kill(instr.vector_and_constant_result);
          kill(instr.scalar_result);
          if (writes_ps) {
            live_ps = false;
          }
        }
        if (vector_needed) {
          uint32_t used = instr.vector_and_constant_result.GetUsedResultComponents();
          for (uint32_t j = 0; j < instr.vector_operand_count; ++j) {
            uint32_t read;
            switch (instr.vector_opcode) {
              case ucode::AluVectorOpcode::kDp4:
                read = 0b1111;
                break;
              case ucode::AluVectorOpcode::kDp3:
                read = 0b0111;
                break;
              case ucode::AluVectorOpcode::kDp2Add:
                read = j < 2 ? 0b0011 : 0b0001;
                break;
              case ucode::AluVectorOpcode::kMaxA:
                read = used | (j == 0 ? 0b1000 : 0b0000);
                break;
              case ucode::AluVectorOpcode::kAdd:
              case ucode::AluVectorOpcode::kMul:
              case ucode::AluVectorOpcode::kMad:
              case ucode::AluVectorOpcode::kMax:
              case ucode::AluVectorOpcode::kMin:
              case ucode::AluVectorOpcode::kSeq:
              case ucode::AluVectorOpcode::kSgt:
              case ucode::AluVectorOpcode::kSge:
              case ucode::AluVectorOpcode::kSne:
              case ucode::AluVectorOpcode::kFrc:
              case ucode::AluVectorOpcode::kTrunc:
              case ucode::AluVectorOpcode::kFloor:
              case ucode::AluVectorOpcode::kCndEq:
              case ucode::AluVectorOpcode::kCndGe:
              case ucode::AluVectorOpcode::kCndGt:
                read = used;
                break;
              default:
                read = 0b1111;
                break;
            }
            make_operand_live(instr.vector_operands[j], read);
          }
        }
        if (scalar_needed) {
          if (instr.scalar_operand_count == 1) {
            make_operand_live(instr.scalar_operands[0], 0b0011);
          } else {
            for (uint32_t j = 0; j < instr.scalar_operand_count; ++j) {
              make_operand_live(instr.scalar_operands[j], 0b0001);
            }
          }
          switch (instr.scalar_opcode) {
            case ucode::AluScalarOpcode::kAddsPrev:
            case ucode::AluScalarOpcode::kSubsPrev:
            case ucode::AluScalarOpcode::kMulsPrev:
            case ucode::AluScalarOpcode::kMulsPrev2:
            case ucode::AluScalarOpcode::kRetainPrev:
              live_ps = true;
              break;
            default:
              break;
          }
        }
      } break;
      case ExtentsProgram::SlotKind::kVertexFetch: {
        const ParsedVertexFetchInstruction& fetch = program.vfetch[slot.index];
        bool writes_position = false;
        bool needed = result_live(fetch.result, writes_position) || full_needed_by_mini[slot.index];
        if (!needed) {
          break;
        }
        program.needed[address] = 1;
        if (!conditional_exec && !fetch.is_predicated) {
          kill(fetch.result);
        }
        if (fetch.is_mini_fetch) {
          uint32_t full = program.vfetch_full[slot.index];
          if (full < full_needed_by_mini.size()) {
            full_needed_by_mini[full] = 1;
          }
        } else {
          make_operand_live(fetch.operands[0], 0b0001);
        }
      } break;
      case ExtentsProgram::SlotKind::kTextureFetch: {
        const ParsedTextureFetchInstruction& fetch = program.tfetch[slot.index];
        bool writes_position = false;
        if (fetch.has_result() && result_live(fetch.result, writes_position)) {
          program.needed[address] = 1;
          // The result is unbounded anyway; its operands don't matter.
        }
      } break;
      default:
        break;
    }
  }
}

void PrepareExtentsProgram(const Shader& shader, ExtentsProgram& program) {
  if (shader.memexport_eM_written()) {
    program.fallback = ScreenExtentsFallback::kShaderUnsupported;
    return;
  }
  const uint32_t* ucode = shader.ucode_dwords();
  const size_t dword_count = shader.ucode_dword_count();
  program.cf_count = shader.cf_pair_index_bound() * 2;
  program.slots.resize(dword_count / 3);
  if (uint64_t(shader.cf_pair_index_bound()) * 3 > dword_count) {
    program.fallback = ScreenExtentsFallback::kShaderUnsupported;
    return;
  }
  ucode::VertexFetchInstruction previous_full;
  std::memset(&previous_full, 0, sizeof(previous_full));
  uint32_t previous_full_ordinal = UINT32_MAX;
  // Instruction addresses in static order, and whether their exec is conditional (for MarkNeededInstructions).
  std::vector<std::pair<uint32_t, bool>> order;
  bool has_jumps = false;
  auto check_operand = [&program](const InstructionOperand& operand) {
    if (operand.storage_source == InstructionStorageSource::kConstantFloat) {
      if (operand.is_absolute_value ||
          operand.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute) {
        program.verify_comparable = false;
      } else {
        program.max_constant = std::max(program.max_constant, operand.storage_index);
      }
    }
  };
  for (uint32_t cf_index = 0; cf_index < program.cf_count; ++cf_index) {
    ucode::ControlFlowInstruction cf_ab[2];
    ucode::UnpackControlFlowInstructions(ucode + 3 * (cf_index >> 1), cf_ab);
    const ucode::ControlFlowInstruction& cf = cf_ab[cf_index & 1];
    ucode::ControlFlowOpcode opcode = cf.opcode();
    if (opcode == ucode::ControlFlowOpcode::kLoopStart || opcode == ucode::ControlFlowOpcode::kLoopEnd) {
      program.fallback = ScreenExtentsFallback::kShaderUnsupported;
      return;
    }
    if (opcode == ucode::ControlFlowOpcode::kCondCall || opcode == ucode::ControlFlowOpcode::kReturn ||
        opcode == ucode::ControlFlowOpcode::kCondJmp) {
      has_jumps = true;
    }
    if (!ucode::IsControlFlowOpcodeExec(opcode)) {
      continue;
    }
    const bool conditional_exec = opcode != ucode::ControlFlowOpcode::kExec &&
                                  opcode != ucode::ControlFlowOpcode::kExecEnd;
    const ucode::ControlFlowExecInstruction& exec = cf.exec;
    for (uint32_t k = 0; k < exec.count(); ++k) {
      uint32_t address = exec.address() + k;
      if (address >= program.slots.size() || program.slots[address].kind != ExtentsProgram::SlotKind::kNone) {
        // Out of the ucode, or an instruction shared by two execs (the static parse order would be ambiguous).
        program.fallback = ScreenExtentsFallback::kShaderUnsupported;
        return;
      }
      order.emplace_back(address, conditional_exec);
      ExtentsProgram::Slot& slot = program.slots[address];
      const uint32_t* op = ucode + 3 * address;
      if ((exec.sequence() >> (2 * k)) & 0b01) {
        const auto& fetch = *reinterpret_cast<const ucode::FetchInstruction*>(op);
        if (fetch.opcode() == ucode::FetchOpcode::kVertexFetch) {
          ParsedVertexFetchInstruction parsed;
          uint32_t ordinal = uint32_t(program.vfetch.size());
          if (ParseVertexFetchInstruction(fetch.vertex_fetch(), previous_full, parsed)) {
            previous_full = fetch.vertex_fetch();
            previous_full_ordinal = ordinal;
          }
          if (!IsFloatVertexFormat(parsed.attributes.data_format)) {
            program.verify_comparable = false;
          }
          slot.kind = ExtentsProgram::SlotKind::kVertexFetch;
          slot.index = ordinal;
          program.vfetch.push_back(parsed);
          program.vfetch_full.push_back(previous_full_ordinal);
        } else {
          ParsedTextureFetchInstruction parsed;
          ParseTextureFetchInstruction(fetch.texture_fetch(), parsed);
          slot.kind = ExtentsProgram::SlotKind::kTextureFetch;
          slot.index = uint32_t(program.tfetch.size());
          program.tfetch.push_back(parsed);
        }
      } else {
        ParsedAluInstruction parsed;
        ParseAluInstruction(*reinterpret_cast<const ucode::AluInstruction*>(op), xenos::ShaderType::kVertex,
                            parsed);
        for (uint32_t i = 0; i < parsed.vector_operand_count; ++i) {
          check_operand(parsed.vector_operands[i]);
        }
        for (uint32_t i = 0; i < parsed.scalar_operand_count; ++i) {
          check_operand(parsed.scalar_operands[i]);
        }
        slot.kind = ExtentsProgram::SlotKind::kAlu;
        slot.index = uint32_t(program.alu.size());
        program.alu.push_back(parsed);
      }
    }
  }
  MarkNeededInstructions(program, order, has_jumps);
}

class ExtentsPositionSink : public ShaderInterpreter::ExportSink {
 public:
  void Export(ucode::ExportRegister export_register, const float* value, uint32_t value_mask) override {
    if (export_register != ucode::ExportRegister::kVSPosition) {
      return;
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (value_mask & (1u << i)) {
        position[i] = value[i];
      }
    }
    written |= value_mask;
  }
  float position[4] = {};
  uint32_t written = 0;
};

}  // namespace

const char* GetScreenExtentsFallbackName(ScreenExtentsFallback reason) {
  switch (reason) {
    case ScreenExtentsFallback::kNone:
      return "none";
    case ScreenExtentsFallback::kShaderNotAnalyzed:
      return "not_analyzed";
    case ScreenExtentsFallback::kShaderUnsupported:
      return "shader_unsupported";
    case ScreenExtentsFallback::kDataDependentFlow:
      return "data_dependent_flow";
    case ScreenExtentsFallback::kShaderTooLong:
      return "shader_too_long";
    case ScreenExtentsFallback::kIndexSource:
      return "index_source";
    case ScreenExtentsFallback::kTessellation:
      return "tessellation";
    case ScreenExtentsFallback::kPrimitiveType:
      return "primitive_type";
    case ScreenExtentsFallback::kVteFormat:
      return "vte_format";
    case ScreenExtentsFallback::kWindowOffset:
      return "window_offset";
    case ScreenExtentsFallback::kConstantsLayout:
      return "constants_layout";
    case ScreenExtentsFallback::kIndexMemory:
      return "index_memory";
    case ScreenExtentsFallback::kVertexMemory:
      return "vertex_memory";
    case ScreenExtentsFallback::kFetchIndex:
      return "fetch_index";
    case ScreenExtentsFallback::kGpuWritten:
      return "gpu_written";
    case ScreenExtentsFallback::kNoSharedMemory:
      return "no_shared_memory";
    case ScreenExtentsFallback::kNoPosition:
      return "no_position";
    case ScreenExtentsFallback::kUnbounded:
      return "unbounded";
    case ScreenExtentsFallback::kNearPlane:
      return "near_plane";
    case ScreenExtentsFallback::kNotListed:
      return "not_listed";
    case ScreenExtentsFallback::kNotIssued:
      return "not_issued";
    case ScreenExtentsFallback::kCopy:
      return "copy";
    case ScreenExtentsFallback::kNoDraw:
      return "no_draw";
    case ScreenExtentsFallback::kNoInit:
      return "no_init";
    default:
      return "?";
  }
}

struct ScreenExtentsEstimator::Impl {
  Impl(const RegisterFile& register_file, const memory::Memory& memory)
      : regs(register_file), memory(memory), interpreter(register_file, memory) {
    AffinePoint(zero, 0.0);
    AffineTop(top);
  }

  const RegisterFile& regs;
  const memory::Memory& memory;
  ShaderInterpreter interpreter;

  std::unordered_map<uint64_t, std::unique_ptr<ExtentsProgram>> programs;

  // The draw being estimated (kept for Verify).
  const ExtentsProgram* program = nullptr;
  std::vector<uint32_t> indices;
  double viewport_scale[2] = {}, viewport_offset[2] = {};
  bool last_estimated = false;

  // Evaluation state.
  SharedMemory* shared_memory = nullptr;
  ScreenExtentsFallback failure = ScreenExtentsFallback::kNone;
  Affine registers[kExtentsTempRegisters][4];
  uint64_t registers_written = 0;
  Affine previous_scalar;
  Affine position[4];
  uint32_t position_written = 0;
  // 0 false, 1 true, 2 unknown.
  uint32_t predicate = 0;
  bool a0_known = true;
  int32_t a0 = 0;
  uint32_t next_symbol = 0;
  uint32_t last_full_fetch = UINT32_MAX;
  Affine zero;
  Affine top;
  Affine operand_scratch[3][4];
  Affine vector_result[4];
  Affine scalar_result;
  Affine temp[3];
  Affine store_scratch[4];
  Affine fetch_values[4];

  void Fail(ScreenExtentsFallback reason) {
    if (failure == ScreenExtentsFallback::kNone) {
      failure = reason;
    }
  }

  const ExtentsProgram& GetProgram(const Shader& shader) {
    auto it = programs.find(shader.ucode_data_hash());
    if (it != programs.end()) {
      return *it->second;
    }
    auto program_new = std::make_unique<ExtentsProgram>();
    PrepareExtentsProgram(shader, *program_new);
    const ExtentsProgram& result = *program_new;
    programs.emplace(shader.ucode_data_hash(), std::move(program_new));
    return result;
  }

  bool BoolConstant(uint32_t address) const {
    return (regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 + (address >> 5)] >> (address & 31)) & 1;
  }

  const Affine& Load(const InstructionOperand& operand, uint32_t component, Affine& scratch) {
    SwizzleSource source = operand.GetComponent(component);
    const Affine* base;
    if (source == SwizzleSource::k0 || source == SwizzleSource::k1) {
      AffinePoint(scratch, source == SwizzleSource::k1 ? 1.0 : 0.0);
      base = &scratch;
    } else {
      uint32_t source_component = uint32_t(source) - uint32_t(SwizzleSource::kX);
      switch (operand.storage_source) {
        case InstructionStorageSource::kRegister: {
          if (operand.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute ||
              operand.storage_index >= kExtentsTempRegisters) {
            Fail(ScreenExtentsFallback::kShaderUnsupported);
            return top;
          }
          base = ((registers_written >> operand.storage_index) & 1)
                     ? &registers[operand.storage_index][source_component]
                     : &zero;
        } break;
        case InstructionStorageSource::kConstantFloat: {
          int32_t index = int32_t(operand.storage_index);
          if (operand.storage_addressing_mode == InstructionStorageAddressingMode::kAddressRegisterRelative) {
            if (!a0_known) {
              Fail(ScreenExtentsFallback::kConstantsLayout);
              return top;
            }
            index += a0;
          } else if (operand.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute) {
            Fail(ScreenExtentsFallback::kShaderUnsupported);
            return top;
          }
          if (index < 0 || index > 255) {
            Fail(ScreenExtentsFallback::kConstantsLayout);
            return top;
          }
          float value;
          uint32_t bits = regs[XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * uint32_t(index) + source_component];
          std::memcpy(&value, &bits, sizeof(value));
          AffineFromFloat(scratch, value);
          base = &scratch;
        } break;
        default:
          Fail(ScreenExtentsFallback::kShaderUnsupported);
          return top;
      }
    }
    if (!operand.is_absolute_value && !operand.is_negated) {
      return *base;
    }
    AffineAssign(scratch, *base);
    if (operand.is_absolute_value) {
      AffineAbs(scratch);
    }
    if (operand.is_negated) {
      AffineNegate(scratch);
    }
    return scratch;
  }

  // values[i]: the operation's result component i (X...W); nullptr where not computed.
  void Store(const InstructionResult& result, const Affine* const* values) {
    uint32_t write_mask = result.GetUsedWriteMask();
    if (!write_mask) {
      return;
    }
    for (uint32_t i = 0; i < 4; ++i) {
      if (!(write_mask & (1u << i))) {
        continue;
      }
      SwizzleSource source = result.components[i];
      if (source >= SwizzleSource::kX && source <= SwizzleSource::kW) {
        const Affine* value = values[uint32_t(source) - uint32_t(SwizzleSource::kX)];
        if (!value) {
          Fail(ScreenExtentsFallback::kShaderUnsupported);
          return;
        }
        AffineAssign(store_scratch[i], *value);
      } else {
        AffinePoint(store_scratch[i], source == SwizzleSource::k1 ? 1.0 : 0.0);
      }
      if (result.is_clamped) {
        AffineSaturate(store_scratch[i]);
      }
    }
    switch (result.storage_target) {
      case InstructionStorageTarget::kRegister: {
        if (result.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute ||
            result.storage_index >= kExtentsTempRegisters) {
          Fail(ScreenExtentsFallback::kShaderUnsupported);
          return;
        }
        uint32_t index = result.storage_index;
        if (!((registers_written >> index) & 1)) {
          registers_written |= uint64_t(1) << index;
          for (uint32_t i = 0; i < 4; ++i) {
            AffinePoint(registers[index][i], 0.0);
          }
        }
        for (uint32_t i = 0; i < 4; ++i) {
          if (write_mask & (1u << i)) {
            AffineAssign(registers[index][i], store_scratch[i]);
          }
        }
      } break;
      case InstructionStorageTarget::kPosition:
        for (uint32_t i = 0; i < 4; ++i) {
          if (write_mask & (1u << i)) {
            AffineAssign(position[i], store_scratch[i]);
          }
        }
        position_written |= write_mask;
        break;
      default:
        // Interpolators, point size / edge flag / kill (kill is ignored: drawing more vertices is
        // conservative). Memexport shaders are rejected when the program is prepared.
        break;
    }
  }

  bool CheckPredicate(bool is_predicated, bool condition, bool& run) {
    run = true;
    if (!is_predicated) {
      return true;
    }
    if (predicate > 1) {
      Fail(ScreenExtentsFallback::kDataDependentFlow);
      return false;
    }
    run = (predicate == 1) == condition;
    return true;
  }

  void SetA0(const Affine& value, bool round_half) {
    if (!AffineIsPoint(value)) {
      a0_known = false;
      return;
    }
    float f = float(value.c);
    if (round_half) {
      f = f + 0.5f;
    }
    f = std::floor(f);
    a0 = int32_t(std::min(std::max(f, -256.0f), 255.0f));
    a0_known = true;
  }

  // Returns whether vector_result holds the operation's result components.
  bool ExecVector(const ParsedAluInstruction& instr, uint32_t used) {
    const InstructionOperand* operands = instr.vector_operands;
    uint32_t identical = instr.vector_operand_count >= 2 ? operands[0].GetIdenticalComponents(operands[1]) : 0;
    switch (instr.vector_opcode) {
      case ucode::AluVectorOpcode::kAdd:
      case ucode::AluVectorOpcode::kMul:
      case ucode::AluVectorOpcode::kMad:
      case ucode::AluVectorOpcode::kSeq:
      case ucode::AluVectorOpcode::kSgt:
      case ucode::AluVectorOpcode::kSge:
      case ucode::AluVectorOpcode::kSne:
      case ucode::AluVectorOpcode::kFrc:
      case ucode::AluVectorOpcode::kTrunc:
      case ucode::AluVectorOpcode::kFloor:
      case ucode::AluVectorOpcode::kCndEq:
      case ucode::AluVectorOpcode::kCndGe:
      case ucode::AluVectorOpcode::kCndGt:
      case ucode::AluVectorOpcode::kMax:
      case ucode::AluVectorOpcode::kMin:
      case ucode::AluVectorOpcode::kMaxA: {
        if (instr.vector_opcode == ucode::AluVectorOpcode::kMaxA) {
          // a0 = (int)clamp(floor(src0.w + 0.5), -256, 255).
          SetA0(Load(operands[0], 3, operand_scratch[0][3]), true);
        }
        for (uint32_t c = 0; c < 4; ++c) {
          if (!(used & (1u << c))) {
            continue;
          }
          Affine& out = vector_result[c];
          const Affine& a = Load(operands[0], c, operand_scratch[0][c]);
          switch (instr.vector_opcode) {
            case ucode::AluVectorOpcode::kAdd:
              AffineAdd(a, Load(operands[1], c, operand_scratch[1][c]), false, out);
              break;
            case ucode::AluVectorOpcode::kMul:
              AffineMul(a, Load(operands[1], c, operand_scratch[1][c]), !((identical >> c) & 1), out);
              break;
            case ucode::AluVectorOpcode::kMad:
              AffineMul(a, Load(operands[1], c, operand_scratch[1][c]), !((identical >> c) & 1), temp[0]);
              AffineAdd(temp[0], Load(operands[2], c, operand_scratch[2][c]), false, out);
              break;
            case ucode::AluVectorOpcode::kSeq:
            case ucode::AluVectorOpcode::kSgt:
            case ucode::AluVectorOpcode::kSge:
            case ucode::AluVectorOpcode::kSne: {
              AffineCmp cmp = instr.vector_opcode == ucode::AluVectorOpcode::kSeq   ? AffineCmp::kEq
                              : instr.vector_opcode == ucode::AluVectorOpcode::kSgt ? AffineCmp::kGt
                              : instr.vector_opcode == ucode::AluVectorOpcode::kSge ? AffineCmp::kGe
                                                                                    : AffineCmp::kNe;
              int decision = AffineCompare(a, Load(operands[1], c, operand_scratch[1][c]), cmp);
              AffineFromDecision(out, decision, 1.0, 0.0);
            } break;
            case ucode::AluVectorOpcode::kFrc:
              AffineRound(a, AffineRounding::kFract, out);
              break;
            case ucode::AluVectorOpcode::kTrunc:
              AffineRound(a, AffineRounding::kTrunc, out);
              break;
            case ucode::AluVectorOpcode::kFloor:
              AffineRound(a, AffineRounding::kFloor, out);
              break;
            case ucode::AluVectorOpcode::kCndEq:
            case ucode::AluVectorOpcode::kCndGe:
            case ucode::AluVectorOpcode::kCndGt: {
              AffineCmp cmp = instr.vector_opcode == ucode::AluVectorOpcode::kCndEq   ? AffineCmp::kEq
                              : instr.vector_opcode == ucode::AluVectorOpcode::kCndGe ? AffineCmp::kGe
                                                                                      : AffineCmp::kGt;
              int decision = AffineCompare(a, zero, cmp);
              const Affine& if_true = Load(operands[1], c, operand_scratch[1][c]);
              const Affine& if_false = Load(operands[2], c, operand_scratch[2][c]);
              if (decision < 0) {
                AffineHull(if_true, if_false, out);
              } else {
                AffineAssign(out, decision ? if_true : if_false);
              }
            } break;
            default: {
              // max / min / maxa: max(a, a) is a move.
              if ((identical >> c) & 1) {
                AffineAssign(out, a);
              } else {
                AffineMaxMin(a, Load(operands[1], c, operand_scratch[1][c]),
                             instr.vector_opcode != ucode::AluVectorOpcode::kMin, out);
              }
            } break;
          }
        }
        return true;
      }
      case ucode::AluVectorOpcode::kDp4:
      case ucode::AluVectorOpcode::kDp3:
      case ucode::AluVectorOpcode::kDp2Add: {
        uint32_t count = instr.vector_opcode == ucode::AluVectorOpcode::kDp4   ? 4
                         : instr.vector_opcode == ucode::AluVectorOpcode::kDp3 ? 3
                                                                               : 2;
        // ((p0 + p1) + p2) + p3, like the host (no OpDot, no contraction).
        for (uint32_t j = 0; j < count; ++j) {
          const Affine& a = Load(operands[0], j, operand_scratch[0][j]);
          const Affine& b = Load(operands[1], j, operand_scratch[1][j]);
          if (!j) {
            AffineMul(a, b, !((identical >> j) & 1), temp[1]);
          } else {
            AffineMul(a, b, !((identical >> j) & 1), temp[0]);
            AffineAdd(temp[1], temp[0], false, temp[1]);
          }
        }
        if (instr.vector_opcode == ucode::AluVectorOpcode::kDp2Add) {
          AffineAdd(temp[1], Load(operands[2], 0, operand_scratch[2][0]), false, temp[1]);
        }
        for (uint32_t c = 0; c < 4; ++c) {
          if (used & (1u << c)) {
            AffineAssign(vector_result[c], temp[1]);
          }
        }
        return true;
      }
      case ucode::AluVectorOpcode::kMax4: {
        bool all_points = true, any_top = false;
        double best_lo = -DBL_MAX, best_hi = -DBL_MAX, best_point = -DBL_MAX;
        for (uint32_t j = 0; j < 4; ++j) {
          const Affine& a = Load(operands[0], j, operand_scratch[0][j]);
          if (a.top) {
            any_top = true;
            break;
          }
          double lo, hi;
          AffineBounds(a, lo, hi);
          all_points = all_points && AffineIsPoint(a);
          best_point = std::max(best_point, a.c);
          best_lo = std::max(best_lo, lo);
          best_hi = std::max(best_hi, hi);
        }
        if (any_top) {
          AffineTop(temp[1]);
        } else if (all_points) {
          AffinePoint(temp[1], best_point);
        } else {
          AffineInterval(temp[1], best_lo, best_hi);
        }
        for (uint32_t c = 0; c < 4; ++c) {
          if (used & (1u << c)) {
            AffineAssign(vector_result[c], temp[1]);
          }
        }
        return true;
      }
      case ucode::AluVectorOpcode::kSetpEqPush:
      case ucode::AluVectorOpcode::kSetpNePush:
      case ucode::AluVectorOpcode::kSetpGtPush:
      case ucode::AluVectorOpcode::kSetpGePush:
        // Not modelled: the predicate becomes unknown (an error only if something depends on it).
        predicate = 2;
        [[fallthrough]];
      case ucode::AluVectorOpcode::kKillEq:
      case ucode::AluVectorOpcode::kKillGt:
      case ucode::AluVectorOpcode::kKillGe:
      case ucode::AluVectorOpcode::kKillNe:
      case ucode::AluVectorOpcode::kDst:
      case ucode::AluVectorOpcode::kCube:
        for (uint32_t c = 0; c < 4; ++c) {
          if (used & (1u << c)) {
            AffineTop(vector_result[c]);
          }
        }
        return true;
      default:
        Fail(ScreenExtentsFallback::kShaderUnsupported);
        return false;
    }
  }

  // Sets scalar_result (not for retain_prev).
  void ExecScalar(const ParsedAluInstruction& instr) {
    const InstructionOperand* operands = instr.scalar_operands;
    const Affine* a = &zero;
    const Affine* b = &zero;
    if (instr.scalar_operand_count == 1) {
      a = &Load(operands[0], 0, operand_scratch[0][0]);
      b = &Load(operands[0], 1, operand_scratch[0][1]);
    } else if (instr.scalar_operand_count == 2) {
      a = &Load(operands[0], 0, operand_scratch[0][0]);
      b = &Load(operands[1], 0, operand_scratch[1][0]);
    }
    bool same_xy =
        instr.scalar_operand_count == 1 && operands[0].GetComponent(0) == operands[0].GetComponent(1);
    Affine& out = scalar_result;
    switch (instr.scalar_opcode) {
      case ucode::AluScalarOpcode::kAdds:
      case ucode::AluScalarOpcode::kSubs:
        AffineAdd(*a, *b, instr.scalar_opcode == ucode::AluScalarOpcode::kSubs, out);
        break;
      case ucode::AluScalarOpcode::kAddsPrev:
      case ucode::AluScalarOpcode::kSubsPrev:
        AffineAdd(*a, previous_scalar, instr.scalar_opcode == ucode::AluScalarOpcode::kSubsPrev, out);
        break;
      case ucode::AluScalarOpcode::kMuls:
        AffineMul(*a, *b, !same_xy, out);
        break;
      case ucode::AluScalarOpcode::kMulsPrev:
        AffineMul(*a, previous_scalar, true, out);
        break;
      case ucode::AluScalarOpcode::kMulsPrev2: {
        // ps != -FLT_MAX && isfinite(ps) && isfinite(src.b) && src.b > 0 ? src.a * ps : -FLT_MAX.
        if (AffineIsPoint(*a) && AffineIsPoint(*b) && AffineIsPoint(previous_scalar)) {
          float ps = float(previous_scalar.c), fb = float(b->c);
          if (ps != -FLT_MAX && std::isfinite(ps) && std::isfinite(fb) && fb > 0.0f) {
            AffineMul(same_xy ? *b : *a, previous_scalar, true, out);
          } else {
            AffinePoint(out, -FLT_MAX);
          }
        } else {
          AffineTop(out);
        }
      } break;
      case ucode::AluScalarOpcode::kMaxs:
      case ucode::AluScalarOpcode::kMins:
      case ucode::AluScalarOpcode::kMaxAs:
      case ucode::AluScalarOpcode::kMaxAsf:
        if (instr.scalar_opcode == ucode::AluScalarOpcode::kMaxAs ||
            instr.scalar_opcode == ucode::AluScalarOpcode::kMaxAsf) {
          SetA0(*a, instr.scalar_opcode == ucode::AluScalarOpcode::kMaxAs);
        }
        if (same_xy) {
          AffineAssign(out, *a);
        } else {
          AffineMaxMin(*a, *b, instr.scalar_opcode != ucode::AluScalarOpcode::kMins, out);
        }
        break;
      case ucode::AluScalarOpcode::kSeqs:
      case ucode::AluScalarOpcode::kSgts:
      case ucode::AluScalarOpcode::kSges:
      case ucode::AluScalarOpcode::kSnes: {
        AffineCmp cmp = instr.scalar_opcode == ucode::AluScalarOpcode::kSeqs   ? AffineCmp::kEq
                        : instr.scalar_opcode == ucode::AluScalarOpcode::kSgts ? AffineCmp::kGt
                        : instr.scalar_opcode == ucode::AluScalarOpcode::kSges ? AffineCmp::kGe
                                                                               : AffineCmp::kNe;
        AffineFromDecision(out, AffineCompare(*a, zero, cmp), 1.0, 0.0);
      } break;
      case ucode::AluScalarOpcode::kFrcs:
        AffineRound(*a, AffineRounding::kFract, out);
        break;
      case ucode::AluScalarOpcode::kTruncs:
        AffineRound(*a, AffineRounding::kTrunc, out);
        break;
      case ucode::AluScalarOpcode::kFloors:
        AffineRound(*a, AffineRounding::kFloor, out);
        break;
      case ucode::AluScalarOpcode::kExp:
        AffineTranscendental(*a, AffineFunction::kExp2, out);
        break;
      case ucode::AluScalarOpcode::kLog:
      case ucode::AluScalarOpcode::kLogc:
        AffineTranscendental(*a, AffineFunction::kLog2, out);
        break;
      case ucode::AluScalarOpcode::kRcp:
      case ucode::AluScalarOpcode::kRcpc:
      case ucode::AluScalarOpcode::kRcpf:
        AffineTranscendental(*a, AffineFunction::kRcp, out);
        break;
      case ucode::AluScalarOpcode::kRsq:
      case ucode::AluScalarOpcode::kRsqc:
      case ucode::AluScalarOpcode::kRsqf:
        AffineTranscendental(*a, AffineFunction::kRsq, out);
        break;
      case ucode::AluScalarOpcode::kSqrt:
        AffineTranscendental(*a, AffineFunction::kSqrt, out);
        break;
      case ucode::AluScalarOpcode::kSin:
        AffineTranscendental(*a, AffineFunction::kSin, out);
        break;
      case ucode::AluScalarOpcode::kCos:
        AffineTranscendental(*a, AffineFunction::kCos, out);
        break;
      case ucode::AluScalarOpcode::kSetpEq:
      case ucode::AluScalarOpcode::kSetpNe:
      case ucode::AluScalarOpcode::kSetpGt:
      case ucode::AluScalarOpcode::kSetpGe: {
        AffineCmp cmp = instr.scalar_opcode == ucode::AluScalarOpcode::kSetpEq   ? AffineCmp::kEq
                        : instr.scalar_opcode == ucode::AluScalarOpcode::kSetpNe ? AffineCmp::kNe
                        : instr.scalar_opcode == ucode::AluScalarOpcode::kSetpGt ? AffineCmp::kGt
                                                                                 : AffineCmp::kGe;
        int decision = AffineCompare(*a, zero, cmp);
        predicate = decision < 0 ? 2 : uint32_t(decision);
        AffineFromDecision(out, decision, 0.0, 1.0);
      } break;
      case ucode::AluScalarOpcode::kSetpInv:
      case ucode::AluScalarOpcode::kSetpPop:
      case ucode::AluScalarOpcode::kSetpRstr:
        if (AffineIsPoint(*a)) {
          float fa = float(a->c);
          float result;
          bool p;
          if (instr.scalar_opcode == ucode::AluScalarOpcode::kSetpInv) {
            p = fa == 1.0f;
            result = p ? 0.0f : (fa == 0.0f ? 1.0f : fa);
          } else if (instr.scalar_opcode == ucode::AluScalarOpcode::kSetpPop) {
            float minus_1 = fa - 1.0f;
            p = minus_1 <= 0.0f;
            result = p ? 0.0f : minus_1;
          } else {
            p = fa == 0.0f;
            result = p ? 0.0f : fa;
          }
          predicate = p ? 1 : 0;
          AffineFromFloat(out, result);
        } else {
          predicate = 2;
          AffineTop(out);
        }
        break;
      case ucode::AluScalarOpcode::kSetpClr:
        predicate = 0;
        AffinePoint(out, FLT_MAX);
        break;
      case ucode::AluScalarOpcode::kKillsEq:
      case ucode::AluScalarOpcode::kKillsGt:
      case ucode::AluScalarOpcode::kKillsGe:
      case ucode::AluScalarOpcode::kKillsNe:
      case ucode::AluScalarOpcode::kKillsOne:
        AffinePoint(out, 0.0);
        break;
      case ucode::AluScalarOpcode::kMulsc0:
      case ucode::AluScalarOpcode::kMulsc1:
        AffineMul(*a, *b, !(operands[0].GetIdenticalComponents(operands[1]) & 1), out);
        break;
      case ucode::AluScalarOpcode::kAddsc0:
      case ucode::AluScalarOpcode::kAddsc1:
      case ucode::AluScalarOpcode::kSubsc0:
      case ucode::AluScalarOpcode::kSubsc1:
        AffineAdd(*a, *b,
                  instr.scalar_opcode == ucode::AluScalarOpcode::kSubsc0 ||
                      instr.scalar_opcode == ucode::AluScalarOpcode::kSubsc1,
                  out);
        break;
      default:
        Fail(ScreenExtentsFallback::kShaderUnsupported);
        break;
    }
  }

  void ExecAlu(const ParsedAluInstruction& instr) {
    if (instr.IsNop()) {
      return;
    }
    bool run;
    if (!CheckPredicate(instr.is_predicated, instr.predicate_condition, run) || !run) {
      return;
    }
    // Both operations read their operands before either result is stored, like the host.
    uint32_t used = instr.vector_and_constant_result.GetUsedResultComponents();
    bool vector_values = false;
    if (used || ucode::GetAluVectorOpcodeInfo(instr.vector_opcode).changed_state) {
      vector_values = ExecVector(instr, used);
    }
    bool retain_prev = instr.scalar_opcode == ucode::AluScalarOpcode::kRetainPrev;
    if (!retain_prev) {
      ExecScalar(instr);
    }
    if (failure != ScreenExtentsFallback::kNone) {
      return;
    }
    const Affine* vector_values_ptr[4];
    for (uint32_t c = 0; c < 4; ++c) {
      vector_values_ptr[c] = (vector_values && (used & (1u << c))) ? &vector_result[c] : nullptr;
    }
    Store(instr.vector_and_constant_result, vector_values_ptr);
    if (!retain_prev) {
      AffineAssign(previous_scalar, scalar_result);
    }
    const Affine* scalar_value = retain_prev ? &previous_scalar : &scalar_result;
    const Affine* scalar_values_ptr[4] = {scalar_value, scalar_value, scalar_value, scalar_value};
    Store(instr.scalar_result, scalar_values_ptr);
  }

  void ExecVertexFetch(uint32_t ordinal) {
    const ExtentsProgram& p = *program;
    const ParsedVertexFetchInstruction& fetch = p.vfetch[ordinal];
    bool run;
    if (!CheckPredicate(fetch.is_predicated, fetch.predicate_condition, run) || !run) {
      return;
    }
    if (!fetch.is_mini_fetch) {
      if (fetch.attributes.stride) {
        // The index must be the vertex index the shader started with (instancing or computed indices would need
        // the index ranges to be derived from the arithmetic).
        const InstructionOperand& index_operand = fetch.operands[0];
        SwizzleSource source = index_operand.GetComponent(0);
        if (index_operand.storage_addressing_mode != InstructionStorageAddressingMode::kAbsolute ||
            index_operand.storage_index >= kExtentsTempRegisters || source > SwizzleSource::kW ||
            !((registers_written >> index_operand.storage_index) & 1) ||
            !registers[index_operand.storage_index][uint32_t(source)].vertex_index) {
          Fail(ScreenExtentsFallback::kFetchIndex);
          return;
        }
      }
      last_full_fetch = ordinal;
    } else if (p.vfetch_full[ordinal] == UINT32_MAX || last_full_fetch != p.vfetch_full[ordinal]) {
      // The host takes the address of the full fetch executed last; the parse assumed the static one.
      Fail(ScreenExtentsFallback::kShaderUnsupported);
      return;
    }
    const Affine* values_ptr[4] = {};
    if (!FetchBounds(fetch, values_ptr)) {
      return;
    }
    Store(fetch.result, values_ptr);
  }

  // The fetched components' ranges over every vertex of the draw.
  bool FetchBounds(const ParsedVertexFetchInstruction& fetch, const Affine** values_ptr) {
    for (uint32_t j = 0; j < 4; ++j) {
      AffinePoint(fetch_values[j], 0.0);
      values_ptr[j] = &fetch_values[j];
    }
    if (fetch.operands[1].storage_index >= xenos::kVertexFetchConstantCount) {
      Fail(ScreenExtentsFallback::kVertexMemory);
      return false;
    }
    xenos::xe_gpu_vertex_fetch_t fetch_constant = regs.GetVertexFetch(fetch.operands[1].storage_index);
    // kInvalidVertex is drawn with gpu_allow_invalid_fetch_constants (the draw was issued, so the host took it).
    if (fetch_constant.type != xenos::FetchConstantType::kVertex &&
        fetch_constant.type != xenos::FetchConstantType::kInvalidVertex) {
      Fail(ScreenExtentsFallback::kVertexMemory);
      return false;
    }
    const ParsedVertexFetchInstruction::Attributes& attributes = fetch.attributes;
    uint32_t used = fetch.result.GetUsedResultComponents();
    uint32_t format_components = (1u << xenos::GetVertexFormatComponentCount(attributes.data_format)) - 1;
    uint32_t used_format = used & format_components;
    uint32_t needed_words = xenos::GetVertexFormatNeededWords(attributes.data_format, used);
    if (!needed_words || !used_format) {
      return true;
    }
    double lo[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
    double hi[4] = {-DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX};
    bool non_finite[4] = {};
    const bool inexact = !IsFloatVertexFormat(attributes.data_format);
    const int64_t buffer_first = int64_t(fetch_constant.address);
    const int64_t buffer_end = buffer_first + int64_t(fetch_constant.size);
    const int64_t memory_end_dwords = int64_t(SharedMemory::kBufferSize >> 2);
    int64_t min_dword = INT64_MAX, max_dword = INT64_MIN;
    const uint32_t* memory_dwords = reinterpret_cast<const uint32_t*>(memory.physical_membase());
    size_t vertex_count = attributes.stride ? indices.size() : 1;
    for (size_t v = 0; v < vertex_count; ++v) {
      int64_t element = buffer_first + (attributes.stride ? int64_t(indices[v]) * attributes.stride : 0);
      uint32_t words[4] = {};
      for (uint32_t w = 0; w < 4; ++w) {
        if (!(needed_words & (1u << w))) {
          continue;
        }
        int64_t dword = element + attributes.offset + w;
        // Outside the fetch constant's range the host buffer may not hold guest data (only that range is
        // requested), so the guest memory read here could differ from what the host draws.
        if (dword < buffer_first || dword >= buffer_end || dword >= memory_end_dwords) {
          Fail(ScreenExtentsFallback::kVertexMemory);
          return false;
        }
        min_dword = std::min(min_dword, dword);
        max_dword = std::max(max_dword, dword);
        words[w] = xenos::GpuSwap(memory_dwords[dword], fetch_constant.endian);
      }
      double decoded[4] = {};
      if (!DecodeVertexFormat(attributes, used_format, words, decoded)) {
        Fail(ScreenExtentsFallback::kShaderUnsupported);
        return false;
      }
      for (uint32_t j = 0; j < 4; ++j) {
        if (!(used_format & (1u << j))) {
          continue;
        }
        if (!std::isfinite(decoded[j])) {
          non_finite[j] = true;
          continue;
        }
        lo[j] = std::min(lo[j], decoded[j]);
        hi[j] = std::max(hi[j], decoded[j]);
      }
    }
    if (max_dword >= min_dword &&
        shared_memory->IsRangeGpuWritten(uint32_t(min_dword << 2), uint32_t((max_dword - min_dword + 1) << 2))) {
      Fail(ScreenExtentsFallback::kGpuWritten);
      return false;
    }
    for (uint32_t j = 0; j < 4; ++j) {
      if (!(used_format & (1u << j))) {
        continue;
      }
      Affine& value = fetch_values[j];
      if (non_finite[j] || lo[j] > hi[j]) {
        AffineTop(value);
        continue;
      }
      if (lo[j] == hi[j] && !inexact) {
        AffineFromFloat(value, float(lo[j]));
        continue;
      }
      NewInput(value, lo[j], hi[j], inexact);
    }
    return true;
  }

  // A named input with the range [lo, hi] (an anonymous one when the names run out).
  void NewInput(Affine& value, double lo, double hi, bool inexact) {
    double center = 0.5 * lo + 0.5 * hi;
    double radius = 0.5 * hi - 0.5 * lo;
    double slack = (std::abs(center) + radius) * (inexact ? kExtentsFetchRound : kExtentsDoubleSlack) +
                   kExtentsFloatMin;
    value.top = false;
    value.vertex_index = false;
    value.c = center;
    if (radius > 0.0 && next_symbol < kExtentsSymbolCount) {
      value.mask = 1u << next_symbol;
      value.coef[next_symbol] = radius;
      ++next_symbol;
      value.err = slack;
    } else {
      value.mask = 0;
      value.err = radius + slack;
    }
  }

  void ExecTextureFetch(uint32_t ordinal) {
    const ParsedTextureFetchInstruction& fetch = program->tfetch[ordinal];
    bool run;
    if (!CheckPredicate(fetch.is_predicated, fetch.predicate_condition, run) || !run) {
      return;
    }
    if (!fetch.has_result()) {
      return;
    }
    // Texture data isn't modelled: unbounded (fails only if the position depends on it).
    const Affine* values_ptr[4] = {&top, &top, &top, &top};
    Store(fetch.result, values_ptr);
  }

  // An instruction the position doesn't depend on (ExtentsProgram::needed): its destinations become unbounded
  // instead of being computed (whether or not it would run), the previous scalar too.
  void ExecNotNeeded(const ExtentsProgram::Slot& slot) {
    const Affine* tops[4] = {&top, &top, &top, &top};
    switch (slot.kind) {
      case ExtentsProgram::SlotKind::kAlu: {
        const ParsedAluInstruction& instr = program->alu[slot.index];
        if (instr.IsNop()) {
          return;
        }
        Store(instr.vector_and_constant_result, tops);
        Store(instr.scalar_result, tops);
        if (instr.scalar_opcode != ucode::AluScalarOpcode::kRetainPrev) {
          AffineTop(previous_scalar);
        }
      } break;
      case ExtentsProgram::SlotKind::kVertexFetch: {
        const ParsedVertexFetchInstruction& fetch = program->vfetch[slot.index];
        if (!fetch.is_mini_fetch) {
          last_full_fetch = slot.index;
        }
        Store(fetch.result, tops);
      } break;
      case ExtentsProgram::SlotKind::kTextureFetch: {
        const ParsedTextureFetchInstruction& fetch = program->tfetch[slot.index];
        if (fetch.has_result()) {
          Store(fetch.result, tops);
        }
      } break;
      default:
        Fail(ScreenExtentsFallback::kShaderUnsupported);
        break;
    }
  }

  // Runs the program's control flow like ShaderInterpreter::Execute (and the host translator).
  bool Run(const uint32_t* ucode) {
    const ExtentsProgram& p = *program;
    uint32_t call_return[4];
    uint32_t call_depth = 0;
    uint32_t steps = 0;
    for (uint32_t cf_index = 0;;) {
      if (cf_index >= p.cf_count) {
        Fail(ScreenExtentsFallback::kShaderUnsupported);
        return false;
      }
      if (++steps > kExtentsMaxCfSteps) {
        Fail(ScreenExtentsFallback::kShaderTooLong);
        return false;
      }
      uint32_t cf_next = cf_index + 1;
      ucode::ControlFlowInstruction cf_ab[2];
      ucode::UnpackControlFlowInstructions(ucode + 3 * (cf_index >> 1), cf_ab);
      const ucode::ControlFlowInstruction& cf = cf_ab[cf_index & 1];
      ucode::ControlFlowOpcode opcode = cf.opcode();
      switch (opcode) {
        case ucode::ControlFlowOpcode::kNop:
        case ucode::ControlFlowOpcode::kAlloc:
        case ucode::ControlFlowOpcode::kMarkVsFetchDone:
          break;
        case ucode::ControlFlowOpcode::kExec:
        case ucode::ControlFlowOpcode::kExecEnd:
        case ucode::ControlFlowOpcode::kCondExec:
        case ucode::ControlFlowOpcode::kCondExecEnd:
        case ucode::ControlFlowOpcode::kCondExecPred:
        case ucode::ControlFlowOpcode::kCondExecPredEnd:
        case ucode::ControlFlowOpcode::kCondExecPredClean:
        case ucode::ControlFlowOpcode::kCondExecPredCleanEnd: {
          bool run = true;
          switch (opcode) {
            case ucode::ControlFlowOpcode::kCondExec:
            case ucode::ControlFlowOpcode::kCondExecEnd:
            case ucode::ControlFlowOpcode::kCondExecPredClean:
            case ucode::ControlFlowOpcode::kCondExecPredCleanEnd:
              run = BoolConstant(cf.cond_exec.bool_address()) == cf.cond_exec.condition();
              break;
            case ucode::ControlFlowOpcode::kCondExecPred:
            case ucode::ControlFlowOpcode::kCondExecPredEnd:
              if (!CheckPredicate(true, cf.cond_exec_pred.condition(), run)) {
                return false;
              }
              break;
            default:
              break;
          }
          if (!run) {
            // A skipped conditional exec_end doesn't end the shader (as in ShaderInterpreter and the host).
            break;
          }
          const ucode::ControlFlowExecInstruction& exec = cf.exec;
          for (uint32_t k = 0; k < exec.count(); ++k) {
            const ExtentsProgram::Slot& slot = p.slots[exec.address() + k];
            if (!p.needed[exec.address() + k]) {
              ExecNotNeeded(slot);
            } else {
              switch (slot.kind) {
                case ExtentsProgram::SlotKind::kAlu:
                  ExecAlu(p.alu[slot.index]);
                  break;
                case ExtentsProgram::SlotKind::kVertexFetch:
                  ExecVertexFetch(slot.index);
                  break;
                case ExtentsProgram::SlotKind::kTextureFetch:
                  ExecTextureFetch(slot.index);
                  break;
                default:
                  Fail(ScreenExtentsFallback::kShaderUnsupported);
                  break;
              }
            }
            if (failure != ScreenExtentsFallback::kNone) {
              return false;
            }
          }
          if (ucode::DoesControlFlowOpcodeEndShader(opcode)) {
            return true;
          }
        } break;
        case ucode::ControlFlowOpcode::kCondCall: {
          const ucode::ControlFlowCondCallInstruction& call = cf.cond_call;
          bool take = true;
          if (!call.is_unconditional()) {
            if (call.is_predicated()) {
              if (!CheckPredicate(true, call.condition(), take)) {
                return false;
              }
            } else {
              take = BoolConstant(call.bool_address()) == call.condition();
            }
          }
          if (take) {
            if (call_depth >= 4) {
              Fail(ScreenExtentsFallback::kShaderUnsupported);
              return false;
            }
            call_return[call_depth++] = cf_index + 1;
            cf_next = call.address();
          }
        } break;
        case ucode::ControlFlowOpcode::kReturn:
          if (call_depth) {
            cf_next = call_return[--call_depth];
          }
          break;
        case ucode::ControlFlowOpcode::kCondJmp: {
          const ucode::ControlFlowCondJmpInstruction& jump = cf.cond_jmp;
          bool take = true;
          if (!jump.is_unconditional()) {
            if (jump.is_predicated()) {
              if (!CheckPredicate(true, jump.condition(), take)) {
                return false;
              }
            } else {
              take = BoolConstant(jump.bool_address()) == jump.condition();
            }
          }
          if (take) {
            cf_next = jump.address();
          }
        } break;
        default:
          // Loops (rejected when preparing) and unknown opcodes.
          Fail(ScreenExtentsFallback::kShaderUnsupported);
          return false;
      }
      cf_index = cf_next;
    }
  }

  // The vertex indices the host shader gets, in draw order (primitive reset indices left out): the guest index
  // (endian-swapped), + VGT_INDX_OFFSET, masked to 24 bits, clamped to VGT_MIN/MAX_VTX_INDX.
  bool BuildIndices(const reg::VGT_DRAW_INITIATOR& initiator) {
    indices.clear();
    uint32_t count = initiator.num_indices;
    if (initiator.source_select == xenos::SourceSelect::kDMA) {
      // As in PrimitiveProcessor: no more indices than the index buffer holds.
      count = std::min(count, uint32_t(regs.Get<reg::VGT_DMA_SIZE>().num_words));
    }
    const uint32_t index_offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
    const uint32_t index_min = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;
    const uint32_t index_max = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;
    auto add = [&](uint32_t guest_index) {
      uint32_t index = (guest_index + index_offset) & xenos::kVertexIndexMask;
      indices.push_back(std::min(std::max(index, index_min), index_max));
    };
    if (initiator.source_select == xenos::SourceSelect::kAutoIndex) {
      indices.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        add(i);
      }
      return true;
    }
    if (initiator.source_select != xenos::SourceSelect::kDMA) {
      Fail(ScreenExtentsFallback::kIndexSource);
      return false;
    }
    const bool is_32bit = initiator.index_size == xenos::IndexFormat::kInt32;
    xenos::Endian endian = regs.Get<reg::VGT_DMA_SIZE>().swap_mode;
    if (!is_32bit) {
      // As in PrimitiveProcessor.
      if (endian == xenos::Endian::k8in32) {
        endian = xenos::Endian::k8in16;
      } else if (endian == xenos::Endian::k16in32) {
        endian = xenos::Endian::kNone;
      }
    }
    const uint32_t index_size_log2 = is_32bit ? 2 : 1;
    const uint32_t base = regs[XE_GPU_REG_VGT_DMA_BASE] & ~((uint32_t(1) << index_size_log2) - 1);
    const uint64_t bytes = uint64_t(count) << index_size_log2;
    if (base > SharedMemory::kBufferSize || SharedMemory::kBufferSize - base < bytes) {
      Fail(ScreenExtentsFallback::kIndexMemory);
      return false;
    }
    if (bytes && shared_memory->IsRangeGpuWritten(base, uint32_t(bytes))) {
      Fail(ScreenExtentsFallback::kGpuWritten);
      return false;
    }
    // Primitive reset, as PrimitiveProcessor enables it.
    bool reset_enabled = false;
    uint32_t reset_index = regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx;
    if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().multi_prim_ib_ena) {
      switch (initiator.prim_type) {
        case xenos::PrimitiveType::kLineStrip:
        case xenos::PrimitiveType::kTriangleFan:
        case xenos::PrimitiveType::kTriangleStrip:
        case xenos::PrimitiveType::kLineLoop:
        case xenos::PrimitiveType::kQuadStrip:
        case xenos::PrimitiveType::kPolygon:
        case xenos::PrimitiveType::k2DLineStrip:
        case xenos::PrimitiveType::k2DTriStrip:
          reset_enabled = is_32bit || reset_index <= UINT16_MAX;
          break;
        default:
          break;
      }
    }
    indices.reserve(count);
    const uint8_t* data = memory.TranslatePhysical<const uint8_t*>(base);
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t guest_index;
      if (is_32bit) {
        uint32_t raw;
        std::memcpy(&raw, data + 4 * i, sizeof(raw));
        guest_index = xenos::GpuSwap(raw, endian);
        if (reset_enabled && (guest_index & xenos::kVertexIndexMask) == reset_index) {
          continue;
        }
      } else {
        uint16_t raw;
        std::memcpy(&raw, data + 2 * i, sizeof(raw));
        guest_index = xenos::GpuSwap(uint32_t(raw), endian) & 0xFFFF;
        if (reset_enabled && guest_index == reset_index) {
          continue;
        }
      }
      add(guest_index);
    }
    return true;
  }

  void Estimate(const Shader& vertex_shader, ScreenExtentsEstimate& out) {
    out = ScreenExtentsEstimate();
    failure = ScreenExtentsFallback::kNone;
    last_estimated = false;
    program = nullptr;
    auto fail = [&](ScreenExtentsFallback reason) {
      out.ok = false;
      out.fallback = reason;
    };
    if (!vertex_shader.is_ucode_analyzed()) {
      fail(ScreenExtentsFallback::kShaderNotAnalyzed);
      return;
    }
    if (!shared_memory) {
      fail(ScreenExtentsFallback::kNoSharedMemory);
      return;
    }
    const ExtentsProgram& p = GetProgram(vertex_shader);
    if (p.fallback != ScreenExtentsFallback::kNone) {
      fail(p.fallback);
      return;
    }
    program = &p;

    // Draw state.
    auto initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
    switch (initiator.prim_type) {
      case xenos::PrimitiveType::kTriangleList:
      case xenos::PrimitiveType::kTriangleFan:
      case xenos::PrimitiveType::kTriangleStrip:
      case xenos::PrimitiveType::kQuadList:
      case xenos::PrimitiveType::kQuadStrip:
      case xenos::PrimitiveType::kPolygon:
        break;
      default:
        fail(ScreenExtentsFallback::kPrimitiveType);
        return;
    }
    if (xenos::IsMajorModeExplicit(initiator.major_mode, initiator.prim_type) &&
        regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select == xenos::VGTOutputPath::kTessellationEnable) {
      fail(ScreenExtentsFallback::kTessellation);
      return;
    }
    auto mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
    if (mode_cntl.poly_mode == xenos::PolygonModeEnable::kDualMode &&
        (mode_cntl.polymode_front_ptype != xenos::PolygonType::kTriangles ||
         mode_cntl.polymode_back_ptype != xenos::PolygonType::kTriangles)) {
      fail(ScreenExtentsFallback::kPrimitiveType);
      return;
    }
    auto vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
    if (vte_cntl.vtx_xy_fmt || !vte_cntl.vtx_w0_fmt) {
      fail(ScreenExtentsFallback::kVteFormat);
      return;
    }
    auto window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
    if (window_offset.window_x_offset || window_offset.window_y_offset) {
      fail(ScreenExtentsFallback::kWindowOffset);
      return;
    }
    if (regs.Get<reg::SQ_VS_CONST>().base) {
      fail(ScreenExtentsFallback::kConstantsLayout);
      return;
    }
    viewport_scale[0] = vte_cntl.vport_x_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XSCALE) : 1.0;
    viewport_scale[1] = vte_cntl.vport_y_scale_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YSCALE) : 1.0;
    viewport_offset[0] = vte_cntl.vport_x_offset_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_XOFFSET) : 0.0;
    viewport_offset[1] = vte_cntl.vport_y_offset_ena ? regs.Get<float>(XE_GPU_REG_PA_CL_VPORT_YOFFSET) : 0.0;
    for (uint32_t i = 0; i < 2; ++i) {
      if (!std::isfinite(viewport_scale[i]) || !std::isfinite(viewport_offset[i])) {
        fail(ScreenExtentsFallback::kUnbounded);
        return;
      }
    }

    if (!BuildIndices(initiator)) {
      fail(failure);
      return;
    }
    out.vertex_count = uint32_t(indices.size());
    last_estimated = true;
    if (indices.empty()) {
      // Nothing is drawn: extents that touch no tile.
      out.ok = true;
      out.empty = true;
      out.min_x = out.min_y = 1023;
      out.max_x = out.max_y = 0;
      return;
    }

    // Evaluate the shader over all vertices at once.
    registers_written = 1;
    uint32_t index_lo = *std::min_element(indices.begin(), indices.end());
    uint32_t index_hi = *std::max_element(indices.begin(), indices.end());
    next_symbol = 0;
    Affine& vertex_index = registers[0][0];
    if (index_lo == index_hi) {
      AffinePoint(vertex_index, double(index_lo));
    } else {
      NewInput(vertex_index, double(index_lo), double(index_hi), false);
    }
    vertex_index.vertex_index = true;
    for (uint32_t i = 1; i < 4; ++i) {
      AffinePoint(registers[0][i], 0.0);
    }
    AffinePoint(previous_scalar, 0.0);
    position_written = 0;
    predicate = 0;
    a0_known = true;
    a0 = 0;
    last_full_fetch = UINT32_MAX;
    if (!Run(vertex_shader.ucode_dwords())) {
      fail(failure != ScreenExtentsFallback::kNone ? failure : ScreenExtentsFallback::kShaderUnsupported);
      return;
    }
    if ((position_written & 0b1011) != 0b1011) {
      fail(ScreenExtentsFallback::kNoPosition);
      return;
    }
    const Affine& x = position[0];
    const Affine& y = position[1];
    const Affine& w = position[3];
    if (x.top || y.top || w.top) {
      fail(ScreenExtentsFallback::kUnbounded);
      return;
    }

    // x / w and y / w over the inputs' box. Each value's anonymous term is an input of its own.
    double ndc[2][2];
    for (uint32_t axis = 0; axis < 2; ++axis) {
      const Affine& numerator = axis ? y : x;
      double a[kExtentsSymbolCount + 2], b[kExtentsSymbolCount + 2];
      uint32_t n = 0;
      for (uint32_t bits = numerator.mask | w.mask; bits; bits &= bits - 1) {
        uint32_t i = uint32_t(std::countr_zero(bits));
        a[n] = ((numerator.mask >> i) & 1) ? numerator.coef[i] : 0.0;
        b[n] = ((w.mask >> i) & 1) ? w.coef[i] : 0.0;
        ++n;
      }
      a[n] = numerator.err;
      b[n] = 0.0;
      ++n;
      a[n] = 0.0;
      b[n] = w.err;
      ++n;
      double w_radius = 0.0;
      for (uint32_t i = 0; i < n; ++i) {
        w_radius += std::abs(b[i]);
      }
      double w_lo = w.c - w_radius;
      // W must be positive for every vertex, with a margin (near the eye plane x / w explodes, and the GPU's
      // rounding of W matters).
      if (!(w_lo > 1.0e-6 * (std::abs(w.c) + w_radius)) || !(w_lo > 1.0e-20)) {
        fail(ScreenExtentsFallback::kNearPlane);
        return;
      }
      double hi, negated_lo;
      if (!MaxLinearFractional(a, b, n, numerator.c, w.c, hi)) {
        fail(ScreenExtentsFallback::kUnbounded);
        return;
      }
      for (uint32_t i = 0; i < n; ++i) {
        a[i] = -a[i];
      }
      if (!MaxLinearFractional(a, b, n, -numerator.c, w.c, negated_lo)) {
        fail(ScreenExtentsFallback::kUnbounded);
        return;
      }
      ndc[axis][0] = -negated_lo;
      ndc[axis][1] = hi;
    }

    // Screen pixels, then cells.
    double screen[2][2];
    for (uint32_t axis = 0; axis < 2; ++axis) {
      double s0 = ndc[axis][0] * viewport_scale[axis] + viewport_offset[axis];
      double s1 = ndc[axis][1] * viewport_scale[axis] + viewport_offset[axis];
      screen[axis][0] = std::min(s0, s1);
      screen[axis][1] = std::max(s0, s1);
      if (!std::isfinite(screen[axis][0]) || !std::isfinite(screen[axis][1])) {
        fail(ScreenExtentsFallback::kUnbounded);
        return;
      }
    }
    out.screen_min_x = screen[0][0];
    out.screen_max_x = screen[0][1];
    out.screen_min_y = screen[1][0];
    out.screen_max_y = screen[1][1];
    // One pixel of margin, then one cell on each side; clamped (the game compares the values unsigned).
    auto cell_min = [](double pixels) {
      double cell = std::floor((pixels - 1.0) * 0.125) - 1.0;
      return uint16_t(std::min(std::max(cell, 0.0), 1023.0));
    };
    auto cell_max = [](double pixels) {
      double cell = std::floor((pixels + 1.0) * 0.125) + 1.0;
      return uint16_t(std::min(std::max(cell, 0.0), 1023.0));
    };
    out.min_x = cell_min(screen[0][0]);
    out.max_x = cell_max(screen[0][1]);
    out.min_y = cell_min(screen[1][0]);
    out.max_y = cell_max(screen[1][1]);
    out.ok = true;
  }

  ScreenExtentsEstimator::VerifyResult Verify(const Shader& vertex_shader, const ScreenExtentsEstimate& estimate,
                                              double exact_out[4], uint32_t& bad_vertex_out) {
    exact_out[0] = exact_out[2] = DBL_MAX;
    exact_out[1] = exact_out[3] = -DBL_MAX;
    bad_vertex_out = UINT32_MAX;
    if (!last_estimated || !program || !estimate.ok || estimate.empty || !program->verify_comparable) {
      return VerifyResult::kNotComparable;
    }
    // ShaderInterpreter returns 0 for constants past SQ_VS_CONST's size; the host doesn't check it.
    if (program->max_constant > regs.Get<reg::SQ_VS_CONST>().size) {
      return VerifyResult::kNotComparable;
    }
    interpreter.SetShader(xenos::ShaderType::kVertex, vertex_shader.ucode_dwords());
    ExtentsPositionSink sink;
    interpreter.SetExportSink(&sink);
    VerifyResult result = VerifyResult::kContained;
    for (uint32_t index : indices) {
      // The host starts every vertex with zeroed registers and the index in r0.x.
      std::memset(interpreter.temp_registers(), 0, sizeof(float) * 4 * xenos::kMaxShaderTempRegisters);
      interpreter.temp_registers()[0] = float(index);
      sink.written = 0;
      interpreter.Execute();
      double x = sink.position[0], y = sink.position[1], w = sink.position[3];
      bool contained = (sink.written & 0b1011) == 0b1011 && w > 0.0;
      if (contained) {
        double px = x / w * viewport_scale[0] + viewport_offset[0];
        double py = y / w * viewport_scale[1] + viewport_offset[1];
        exact_out[0] = std::min(exact_out[0], px);
        exact_out[1] = std::max(exact_out[1], px);
        exact_out[2] = std::min(exact_out[2], py);
        exact_out[3] = std::max(exact_out[3], py);
        // float32 evaluation in a different order: allow a thousandth of a pixel.
        double tolerance_x = 1.0e-3 + 1.0e-6 * std::abs(px);
        double tolerance_y = 1.0e-3 + 1.0e-6 * std::abs(py);
        contained = px >= estimate.screen_min_x - tolerance_x && px <= estimate.screen_max_x + tolerance_x &&
                    py >= estimate.screen_min_y - tolerance_y && py <= estimate.screen_max_y + tolerance_y;
      }
      if (!contained && result == VerifyResult::kContained) {
        result = VerifyResult::kMismatch;
        bad_vertex_out = index;
      }
    }
    interpreter.SetExportSink(nullptr);
    return result;
  }
};

ScreenExtentsEstimator::ScreenExtentsEstimator(const RegisterFile& register_file,
                                               const memory::Memory& memory)
    : impl_(std::make_unique<Impl>(register_file, memory)) {}

ScreenExtentsEstimator::~ScreenExtentsEstimator() = default;

void ScreenExtentsEstimator::Estimate(const Shader& vertex_shader, SharedMemory* shared_memory,
                                      ScreenExtentsEstimate& out) {
  impl_->shared_memory = shared_memory;
  impl_->Estimate(vertex_shader, out);
}

ScreenExtentsEstimator::VerifyResult ScreenExtentsEstimator::Verify(const Shader& vertex_shader,
                                                                    const ScreenExtentsEstimate& estimate,
                                                                    double exact_out[4],
                                                                    uint32_t& bad_vertex_out) {
  return impl_->Verify(vertex_shader, estimate, exact_out, bad_vertex_out);
}

// ---------------------------------------------------------------------------------------------------------------
// ScreenExtentsShadow: where the estimates are used, and what they would save.
//
// Passes. The tile a pass replays comes from the bin select the D3D worker writes before each replay
// (docs/research/predicated_tiling.md 1.3): 3 << (2 t) for tile t, plus bit 31 in tile 0, the extents pass (only
// there do the predicated SCREEN_EXT_RPT / SCREEN_EXT_INIT packets run); all ones outside tiling. A tiling bracket
// starts when tile 0's pass starts.
// Windows. SCREEN_EXT_INIT (in a pass with bit 31) opens a window; every draw packet in it is accumulated (a draw
// of a shader not listed, a resolve, or a draw the backend didn't draw makes the window full-screen); SCREEN_EXT_RPT
// reports it. The game wraps each draw in its own window.
// Would-skip. At the end of a bracket, each tile's rectangle comes from its pass's window offset (BeginTiling sets
// PA_SC_WINDOW_OFFSET to minus the tile origin): x1 = -offset x, y1 = -offset y; x2 = the next tile's x1 in the
// same row (8192 for the last one, larger than the real 1280, so the count can only be too low); y2 alike. A
// report skips tile t when the game's rule (sub_82CC95F0) finds no overlap: tile x1 < 8 (max x + 1) and tile x2 >
// 8 min x, and the same for y.
// ---------------------------------------------------------------------------------------------------------------

namespace {

constexpr int kExtentsPassUntiled = -1;
constexpr int kExtentsPassUnknown = -2;
constexpr uint32_t kExtentsSummaryFrames = 300;
constexpr uint32_t kExtentsMaxMismatchLogs = 32;

int ExtentsTileOfSelect(uint32_t select_lo) {
  if (select_lo == UINT32_MAX) {
    return kExtentsPassUntiled;
  }
  if (select_lo & 0x7FFFFF00u) {
    return kExtentsPassUnknown;
  }
  for (int tile = 0; tile < 4; ++tile) {
    if ((select_lo & 0xFFu) == (3u << (2 * tile))) {
      return tile;
    }
  }
  return kExtentsPassUnknown;
}

}  // namespace

struct ScreenExtentsShadow::Impl {
  Impl(const RegisterFile& register_file, const memory::Memory& memory)
      : regs(register_file), estimator(register_file, memory) {}

  static constexpr size_t kReasons = size_t(ScreenExtentsFallback::kCount);

  struct Counters {
    uint64_t estimated = 0;  // listed draws estimated (extents pass, inside a window)
    uint64_t estimate_ok = 0;
    uint64_t estimate_empty = 0;
    uint64_t estimate_fallback[kReasons] = {};
    uint64_t estimate_ns = 0;
    uint64_t estimate_ns_max = 0;
    uint64_t vertices = 0;
    uint64_t listed_outside_window = 0;  // listed draws in the extents pass with no window open
    uint64_t listed_other_pass = 0;      // listed draws in passes without bit 31 (tiles 1-3), not estimated
    uint64_t inits = 0;
    uint64_t inits_ignored = 0;  // SCREEN_EXT_INIT in a pass without bit 31
    uint64_t reports = 0;
    uint64_t reports_tile0 = 0;
    uint64_t reports_untiled = 0;
    uint64_t reports_other = 0;
    uint64_t reports_computed = 0;
    uint64_t report_fallback[kReasons] = {};
    uint64_t reports_written = 0;  // mode 2 with gpu_extents_allow_write
    uint64_t brackets = 0;
    uint64_t bracket_reports = 0;  // reports in tile 0 passes of finished brackets
    uint64_t bracket_reports_computed = 0;
    uint64_t tile_passes[4] = {};
    uint64_t tile_listed_draws[4] = {};
    uint64_t would_skip[4] = {};
    uint64_t verify_checked = 0;
    uint64_t verify_mismatches = 0;
    uint64_t verify_not_comparable = 0;
    double verify_slack_px = 0.0;  // sum over checked draws of the four sides' distance to the exact bounds

    void Add(const Counters& o) {
      estimated += o.estimated;
      estimate_ok += o.estimate_ok;
      estimate_empty += o.estimate_empty;
      for (size_t i = 0; i < kReasons; ++i) {
        estimate_fallback[i] += o.estimate_fallback[i];
        report_fallback[i] += o.report_fallback[i];
      }
      estimate_ns += o.estimate_ns;
      estimate_ns_max = std::max(estimate_ns_max, o.estimate_ns_max);
      vertices += o.vertices;
      listed_outside_window += o.listed_outside_window;
      listed_other_pass += o.listed_other_pass;
      inits += o.inits;
      inits_ignored += o.inits_ignored;
      reports += o.reports;
      reports_tile0 += o.reports_tile0;
      reports_untiled += o.reports_untiled;
      reports_other += o.reports_other;
      reports_computed += o.reports_computed;
      reports_written += o.reports_written;
      brackets += o.brackets;
      bracket_reports += o.bracket_reports;
      bracket_reports_computed += o.bracket_reports_computed;
      for (size_t t = 0; t < 4; ++t) {
        tile_passes[t] += o.tile_passes[t];
        tile_listed_draws[t] += o.tile_listed_draws[t];
        would_skip[t] += o.would_skip[t];
      }
      verify_checked += o.verify_checked;
      verify_mismatches += o.verify_mismatches;
      verify_not_comparable += o.verify_not_comparable;
      verify_slack_px += o.verify_slack_px;
    }
  };

  struct Report {
    uint16_t min_x, max_x, min_y, max_y;
  };

  struct Bracket {
    bool any = false;
    uint32_t reports = 0;  // in tile 0's pass
    std::vector<Report> computed;
    bool tile_seen[4] = {};
    bool tile_from_listed[4] = {};
    int32_t tile_x[4] = {};  // -PA_SC_WINDOW_OFFSET (see OnDraw)
    int32_t tile_y[4] = {};
    uint32_t tile_listed_draws[4] = {};
  };

  const RegisterFile& regs;
  ScreenExtentsEstimator estimator;

  // Settings, latched at frame boundaries.
  int32_t mode = 0;
  bool allow_write = false;
  int32_t verify_every = 0;
  std::string shaders_setting;
  std::vector<uint64_t> shaders;

  // Pass.
  bool select_valid = false;
  uint64_t last_select = 0;
  int pass_tile = kExtentsPassUntiled;
  bool pass_extents = true;

  // The SCREEN_EXT_INIT ... SCREEN_EXT_RPT window.
  bool window_open = false;
  bool window_has_draw = false;
  ScreenExtentsFallback window_fallback = ScreenExtentsFallback::kNone;
  bool window_empty = true;
  Report window_bounds = {};

  Bracket bracket;
  std::string last_tiles;  // the last finished bracket's tile rectangles, for the summary

  Counters frame_counters;
  Counters summary_counters;
  uint32_t summary_first_frame = 0;
  uint32_t summary_frames = 0;
  int32_t summary_mode = 0;
  uint64_t verify_counter = 0;
  uint64_t mismatches_logged = 0;

  bool IsListed(uint64_t hash) const {
    return std::find(shaders.begin(), shaders.end(), hash) != shaders.end();
  }

  void ParseShaders(const std::string& setting) {
    shaders_setting = setting;
    shaders.clear();
    size_t position = 0;
    while (position <= setting.size()) {
      size_t end = setting.find(',', position);
      if (end == std::string::npos) {
        end = setting.size();
      }
      std::string item = setting.substr(position, end - position);
      item.erase(std::remove_if(item.begin(), item.end(), [](char c) { return c == ' ' || c == '\t'; }),
                 item.end());
      if (!item.empty()) {
        uint64_t hash = std::strtoull(item.c_str(), nullptr, 16);
        if (hash) {
          shaders.push_back(hash);
        }
      }
      position = end + 1;
    }
  }

  void NotePass(uint64_t select) {
    if (select_valid && select == last_select) {
      return;
    }
    uint32_t select_lo = uint32_t(select);
    int tile = ExtentsTileOfSelect(select_lo);
    bool extents = (select_lo & 0x80000000u) != 0;
    if (tile == 0 && pass_tile != 0) {
      FinishBracket();
    }
    if (tile != pass_tile || extents != pass_extents) {
      // A window doesn't survive a pass without bit 31 (the predicated INIT / RPT packets don't run there). It
      // does survive untiled -> tile 0 (BeginTiling's unpredicated SCREEN_EXT_INIT may come before the select
      // changes).
      if (!extents) {
        window_open = false;
      }
      if (tile >= 0) {
        ++frame_counters.tile_passes[tile];
      }
    }
    pass_tile = tile;
    pass_extents = extents;
    last_select = select;
    select_valid = true;
  }

  void WindowFail(ScreenExtentsFallback reason) {
    if (window_fallback == ScreenExtentsFallback::kNone) {
      window_fallback = reason;
    }
  }

  void FinishBracket() {
    if (!bracket.any) {
      return;
    }
    Counters& counters = frame_counters;
    ++counters.brackets;
    counters.bracket_reports += bracket.reports;
    counters.bracket_reports_computed += bracket.computed.size();
    int32_t x1[4], x2[4], y1[4], y2[4];
    fmt::memory_buffer tiles;
    for (int t = 0; t < 4; ++t) {
      if (!bracket.tile_seen[t]) {
        continue;
      }
      x1[t] = bracket.tile_x[t];
      y1[t] = bracket.tile_y[t];
      x2[t] = 8192;
      y2[t] = 8192;
      for (int u = 0; u < 4; ++u) {
        if (u == t || !bracket.tile_seen[u]) {
          continue;
        }
        if (bracket.tile_y[u] == y1[t] && bracket.tile_x[u] > x1[t]) {
          x2[t] = std::min(x2[t], bracket.tile_x[u]);
        }
        if (bracket.tile_x[u] == x1[t] && bracket.tile_y[u] > y1[t]) {
          y2[t] = std::min(y2[t], bracket.tile_y[u]);
        }
      }
      counters.tile_listed_draws[t] += bracket.tile_listed_draws[t];
      fmt::format_to(std::back_inserter(tiles), "{}{}:{}-{}x{}-{}", tiles.size() ? " " : "", t, x1[t], x2[t],
                     y1[t], y2[t]);
    }
    for (const Report& report : bracket.computed) {
      for (int t = 0; t < 4; ++t) {
        if (!bracket.tile_seen[t]) {
          continue;
        }
        bool touched = x1[t] < 8 * (int32_t(report.max_x) + 1) && x2[t] > 8 * int32_t(report.min_x) &&
                       y1[t] < 8 * (int32_t(report.max_y) + 1) && y2[t] > 8 * int32_t(report.min_y);
        if (!touched) {
          ++counters.would_skip[t];
        }
      }
    }
    last_tiles.assign(tiles.data(), tiles.size());
    bracket.any = false;
    bracket.reports = 0;
    bracket.computed.clear();
    for (int t = 0; t < 4; ++t) {
      bracket.tile_seen[t] = false;
      bracket.tile_from_listed[t] = false;
      bracket.tile_listed_draws[t] = 0;
    }
  }

  void OnDraw(const Shader* vertex_shader, uint64_t select, bool issued, bool is_copy,
              SharedMemory* shared_memory, DrawRow& row) {
    NotePass(select);
    const bool listed = vertex_shader && !is_copy && IsListed(vertex_shader->ucode_data_hash());
    if (pass_tile >= 0) {
      bracket.any = true;
      // The tile's origin: at its pass's first draw, replaced at its first listed draw (a draw before the pass's
      // own window offset write, like a clear D3D issues right before tile 0, would see a stale offset).
      if (!bracket.tile_seen[pass_tile] || (listed && !bracket.tile_from_listed[pass_tile])) {
        auto window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
        bracket.tile_seen[pass_tile] = true;
        bracket.tile_from_listed[pass_tile] = listed;
        bracket.tile_x[pass_tile] = -int32_t(window_offset.window_x_offset);
        bracket.tile_y[pass_tile] = -int32_t(window_offset.window_y_offset);
      }
      if (listed && issued) {
        ++bracket.tile_listed_draws[pass_tile];
      }
    }
    if (!pass_extents) {
      if (listed) {
        ++frame_counters.listed_other_pass;
      }
      return;
    }
    if (!window_open) {
      if (listed && issued) {
        ++frame_counters.listed_outside_window;
      }
      return;
    }
    window_has_draw = true;
    if (is_copy) {
      WindowFail(ScreenExtentsFallback::kCopy);
      return;
    }
    if (!issued) {
      WindowFail(ScreenExtentsFallback::kNotIssued);
      return;
    }
    if (!listed) {
      WindowFail(ScreenExtentsFallback::kNotListed);
      return;
    }
    // The estimate (timed: the CPU cost of the method).
    ScreenExtentsEstimate estimate;
    auto start = std::chrono::steady_clock::now();
    estimator.Estimate(*vertex_shader, shared_memory, estimate);
    uint64_t ns = uint64_t(std::max<int64_t>(
        0, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
               .count()));
    Counters& counters = frame_counters;
    ++counters.estimated;
    counters.estimate_ns += ns;
    counters.estimate_ns_max = std::max(counters.estimate_ns_max, ns);
    counters.vertices += estimate.vertex_count;
    row.attempted = true;
    row.ok = estimate.ok;
    row.fallback = estimate.fallback;
    row.us = float(double(ns) / 1000.0);
    if (!estimate.ok) {
      ++counters.estimate_fallback[size_t(estimate.fallback)];
      WindowFail(estimate.fallback);
      return;
    }
    ++counters.estimate_ok;
    row.min_x = estimate.min_x;
    row.max_x = estimate.max_x;
    row.min_y = estimate.min_y;
    row.max_y = estimate.max_y;
    if (estimate.empty) {
      ++counters.estimate_empty;
    } else if (window_empty) {
      window_empty = false;
      window_bounds = {estimate.min_x, estimate.max_x, estimate.min_y, estimate.max_y};
    } else {
      window_bounds.min_x = std::min(window_bounds.min_x, estimate.min_x);
      window_bounds.max_x = std::max(window_bounds.max_x, estimate.max_x);
      window_bounds.min_y = std::min(window_bounds.min_y, estimate.min_y);
      window_bounds.max_y = std::max(window_bounds.max_y, estimate.max_y);
    }
    // gpu_extents_mode_verify: the same register state, vertex by vertex through ShaderInterpreter.
    if (verify_every > 0 && !estimate.empty && (++verify_counter % uint64_t(verify_every)) == 0) {
      double exact[4];
      uint32_t bad_vertex;
      switch (estimator.Verify(*vertex_shader, estimate, exact, bad_vertex)) {
        case ScreenExtentsEstimator::VerifyResult::kContained:
          ++counters.verify_checked;
          counters.verify_slack_px += (exact[0] - estimate.screen_min_x) + (estimate.screen_max_x - exact[1]) +
                                      (exact[2] - estimate.screen_min_y) + (estimate.screen_max_y - exact[3]);
          break;
        case ScreenExtentsEstimator::VerifyResult::kMismatch:
          ++counters.verify_checked;
          ++counters.verify_mismatches;
          if (mismatches_logged < kExtentsMaxMismatchLogs) {
            ++mismatches_logged;
            REXGPU_WARN(
                "[s7][EXTENTS-SHADOW] verify mismatch: vertex shader {:016X}, {} vertices, vertex index {} outside "
                "the estimate: estimated x {:.3f}...{:.3f} y {:.3f}...{:.3f}, vertices x {:.3f}...{:.3f} y "
                "{:.3f}...{:.3f} (cells x {}-{} y {}-{})",
                vertex_shader->ucode_data_hash(), estimate.vertex_count, bad_vertex, estimate.screen_min_x,
                estimate.screen_max_x, estimate.screen_min_y, estimate.screen_max_y, exact[0], exact[1],
                exact[2], exact[3], estimate.min_x, estimate.max_x, estimate.min_y, estimate.max_y);
          }
          break;
        default:
          ++counters.verify_not_comparable;
          break;
      }
    }
  }

  void OnInit(uint64_t select) {
    NotePass(select);
    ++frame_counters.inits;
    if (!pass_extents) {
      ++frame_counters.inits_ignored;
      return;
    }
    window_open = true;
    window_has_draw = false;
    window_fallback = ScreenExtentsFallback::kNone;
    window_empty = true;
  }

  bool OnReport(uint64_t select, uint16_t extents_out[6]) {
    NotePass(select);
    Counters& counters = frame_counters;
    ++counters.reports;
    if (pass_tile == 0) {
      ++counters.reports_tile0;
    } else if (pass_tile == kExtentsPassUntiled) {
      ++counters.reports_untiled;
    } else {
      ++counters.reports_other;
    }
    ScreenExtentsFallback why = ScreenExtentsFallback::kNone;
    if (!window_open) {
      why = ScreenExtentsFallback::kNoInit;
    } else if (!window_has_draw) {
      why = ScreenExtentsFallback::kNoDraw;
    } else {
      why = window_fallback;
    }
    Report report = window_empty ? Report{1023, 0, 1023, 0} : window_bounds;
    if (pass_tile == 0) {
      bracket.any = true;
      ++bracket.reports;
    }
    // The hardware keeps accumulating until the next SCREEN_EXT_INIT (the game sends one right after each
    // report), so the window stays open.
    if (why != ScreenExtentsFallback::kNone) {
      ++counters.report_fallback[size_t(why)];
      return false;
    }
    ++counters.reports_computed;
    if (pass_tile == 0) {
      bracket.computed.push_back(report);
    }
    if (mode == 2 && allow_write) {
      extents_out[0] = report.min_x;
      extents_out[1] = report.max_x;
      extents_out[2] = report.min_y;
      extents_out[3] = report.max_y;
      // Z as the full-screen fake has it.
      extents_out[4] = 0;
      extents_out[5] = 1;
      ++counters.reports_written;
      return true;
    }
    return false;
  }

  static void AppendReasons(fmt::memory_buffer& text, const uint64_t* reasons, double frames) {
    bool any = false;
    for (size_t i = 1; i < kReasons; ++i) {
      if (!reasons[i]) {
        continue;
      }
      fmt::format_to(std::back_inserter(text), "{}{}={:.1f}", any ? ", " : " [",
                     GetScreenExtentsFallbackName(ScreenExtentsFallback(i)), double(reasons[i]) / frames);
      any = true;
    }
    if (any) {
      text.push_back(']');
    }
  }

  void LogSummary(uint32_t last_frame) {
    const Counters& s = summary_counters;
    const double frames = double(std::max<uint32_t>(summary_frames, 1));
    fmt::memory_buffer text;
    fmt::format_to(std::back_inserter(text),
                   "[s7][EXTENTS] frames {}-{} ({} frames, mode {}{}, {} shader(s)), per frame: estimated {:.1f} (ok "
                   "{:.1f}, empty {:.1f}, fallback {:.1f}",
                   summary_first_frame, last_frame, summary_frames, summary_mode,
                   (summary_mode == 2 && !allow_write) ? " = shadow, writes not allowed" : "", shaders.size(),
                   double(s.estimated) / frames, double(s.estimate_ok) / frames,
                   double(s.estimate_empty) / frames, double(s.estimated - s.estimate_ok) / frames);
    AppendReasons(text, s.estimate_fallback, frames);
    fmt::format_to(std::back_inserter(text),
                   "), cost {:.3f} ms ({:.2f} us/draw, max {:.1f} us, {:.0f} vertices/draw); reports {:.1f} (tile 0 "
                   "{:.1f}, untiled {:.1f}, other {:.1f}), computed {:.1f}",
                   double(s.estimate_ns) / 1.0e6 / frames,
                   s.estimated ? double(s.estimate_ns) / 1000.0 / double(s.estimated) : 0.0,
                   double(s.estimate_ns_max) / 1000.0, s.estimated ? double(s.vertices) / double(s.estimated) : 0.0,
                   double(s.reports) / frames, double(s.reports_tile0) / frames, double(s.reports_untiled) / frames,
                   double(s.reports_other) / frames, double(s.reports_computed) / frames);
    AppendReasons(text, s.report_fallback, frames);
    fmt::format_to(std::back_inserter(text),
                   ", written {:.1f}; inits {:.1f} (outside the extents pass {:.1f}); listed draws outside a window "
                   "{:.1f}, in other passes {:.1f}; brackets {:.2f} (tile passes {:.2f}/{:.2f}/{:.2f}/{:.2f}, tiles "
                   "{}), listed draws per tile {:.1f}/{:.1f}/{:.1f}/{:.1f}, tile 0 reports {:.1f} (computed {:.1f}), "
                   "would skip: tile 1 {:.1f}, tile 2 {:.1f}, tile 3 {:.1f} (outside tile 0 {:.1f})",
                   double(s.reports_written) / frames, double(s.inits) / frames, double(s.inits_ignored) / frames,
                   double(s.listed_outside_window) / frames, double(s.listed_other_pass) / frames,
                   double(s.brackets) / frames, double(s.tile_passes[0]) / frames, double(s.tile_passes[1]) / frames,
                   double(s.tile_passes[2]) / frames, double(s.tile_passes[3]) / frames,
                   last_tiles.empty() ? std::string("none") : last_tiles, double(s.tile_listed_draws[0]) / frames,
                   double(s.tile_listed_draws[1]) / frames, double(s.tile_listed_draws[2]) / frames,
                   double(s.tile_listed_draws[3]) / frames, double(s.bracket_reports) / frames,
                   double(s.bracket_reports_computed) / frames, double(s.would_skip[1]) / frames,
                   double(s.would_skip[2]) / frames, double(s.would_skip[3]) / frames,
                   double(s.would_skip[0]) / frames);
    if (verify_every > 0 || s.verify_checked || s.verify_not_comparable) {
      fmt::format_to(std::back_inserter(text),
                     "; verify: checked {}, mismatches {}, not comparable {}, mean slack {:.2f} px per side",
                     s.verify_checked, s.verify_mismatches, s.verify_not_comparable,
                     (s.verify_checked - s.verify_mismatches)
                         ? s.verify_slack_px / (4.0 * double(s.verify_checked - s.verify_mismatches))
                         : 0.0);
    }
    REXGPU_INFO("{}", std::string_view(text.data(), text.size()));
  }

  bool EndFrame(uint32_t frame) {
    FinishBracket();
    window_open = false;
    select_valid = false;
    pass_tile = kExtentsPassUntiled;
    pass_extents = true;
    if (mode != 0) {
      Counters& f = frame_counters;
      // Fixed layout:
      //   cpf_ext   a = listed draws estimated with computed extents, b = estimated but fallen back to full
      //             screen, c = estimate time (microseconds)
      //   cpf_extw  a, b, c = reports whose computed extents miss tile 1, 2, 3 (would be skipped there)
      //   cpf_extr  a = SCREEN_EXT_RPT packets, b = of those, with computed extents, c = listed draws in tile 1
      if (perf::timeline::Enabled()) {
        perf::timeline::Record("cpf_ext", uint32_t(f.estimate_ok), uint32_t(f.estimated - f.estimate_ok),
                               uint32_t(f.estimate_ns / 1000));
        perf::timeline::Record("cpf_extw", uint32_t(f.would_skip[1]), uint32_t(f.would_skip[2]),
                               uint32_t(f.would_skip[3]));
        perf::timeline::Record("cpf_extr", uint32_t(f.reports), uint32_t(f.reports_computed),
                               uint32_t(f.tile_listed_draws[1]));
      }
      if (!summary_frames) {
        summary_first_frame = frame;
        summary_mode = mode;
      }
      summary_counters.Add(f);
      ++summary_frames;
    }
    frame_counters = Counters();

    // Latch the settings for the next frame.
    int32_t new_mode = std::min(std::max(REXCVAR_GET(gpu_extents_mode), int32_t(0)), int32_t(2));
    if (summary_frames && (summary_frames >= kExtentsSummaryFrames || new_mode != summary_mode)) {
      LogSummary(frame);
      summary_counters = Counters();
      summary_frames = 0;
    }
    if (new_mode != mode) {
      REXGPU_INFO("[s7][EXTENTS] gpu_extents_mode {} -> {} at frame {}", mode, new_mode, frame);
    }
    mode = new_mode;
    allow_write = REXCVAR_GET(gpu_extents_allow_write);
    verify_every = std::max(REXCVAR_GET(gpu_extents_mode_verify), int32_t(0));
    // The list is only needed while a mode is on. Copied through the cvar registry, which holds the registry lock
    // that SetFlagByName (console, settings overlay, input-script "cvar" command) assigns the string under; the
    // storage itself (REXCVAR_GET) can be reallocated by another thread while it is read.
    if (mode != 0) {
      const std::string setting = rex::cvar::Query<std::string>("gpu_extents_shaders");
      if (setting != shaders_setting) {
        ParseShaders(setting);
        REXGPU_INFO("[s7][EXTENTS] gpu_extents_shaders: {} shader(s) from \"{}\"", shaders.size(), setting);
      }
    }
    return mode != 0;
  }
};

ScreenExtentsShadow::ScreenExtentsShadow(const RegisterFile& register_file, const memory::Memory& memory)
    : impl_(std::make_unique<Impl>(register_file, memory)) {}

ScreenExtentsShadow::~ScreenExtentsShadow() = default;

bool ScreenExtentsShadow::Requested() { return REXCVAR_GET(gpu_extents_mode) != 0; }

bool ScreenExtentsShadow::EndFrame(uint32_t frame) { return impl_->EndFrame(frame); }

void ScreenExtentsShadow::OnDraw(const Shader* vertex_shader, uint64_t bin_select, bool issued, bool is_copy,
                                 SharedMemory* shared_memory, DrawRow& row) {
  impl_->OnDraw(vertex_shader, bin_select, issued, is_copy, shared_memory, row);
}

void ScreenExtentsShadow::OnScreenExtentsInit(uint64_t bin_select) { impl_->OnInit(bin_select); }

bool ScreenExtentsShadow::OnScreenExtentsReport(uint64_t bin_select, uint16_t extents_out[6]) {
  return impl_->OnReport(bin_select, extents_out);
}

}  // namespace rex::graphics
