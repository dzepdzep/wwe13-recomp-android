#pragma once
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

#include <cstdint>
#include <memory>
#include <optional>

#include <rex/graphics/pipeline/shader/interpreter.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/register_file.h>
#include <rex/memory.h>

namespace rex::graphics {

class DrawExtentEstimator {
 public:
  DrawExtentEstimator(const RegisterFile& register_file, const memory::Memory& memory)
      : register_file_(register_file),
        memory_(memory),
        shader_interpreter_(register_file, memory) {}

  // wwe2k14 patch (Session 87, D3D layer stage 3): the record's register image while a record-fed draw runs (null: the
  // file it was made with); forwarded to its interpreter.
  void SetRegisterSource(const RegisterFile* source) {
    register_source_ = source;
    shader_interpreter_.SetRegisterSource(source);
  }

  // The shader must have its ucode analyzed.
  uint32_t EstimateVertexMaxY(const Shader& vertex_shader);
  uint32_t EstimateMaxY(bool try_to_estimate_vertex_max_y, const Shader& vertex_shader);

 private:
  class PositionYExportSink : public ShaderInterpreter::ExportSink {
   public:
    void Export(ucode::ExportRegister export_register, const float* value,
                uint32_t value_mask) override;

    void Reset() {
      position_y_.reset();
      position_w_.reset();
      point_size_.reset();
      vertex_kill_.reset();
    }

    const std::optional<float>& position_y() const { return position_y_; }
    const std::optional<float>& position_w() const { return position_w_; }
    const std::optional<float>& point_size() const { return point_size_; }
    const std::optional<uint32_t>& vertex_kill() const { return vertex_kill_; }

   private:
    std::optional<float> position_y_;
    std::optional<float> position_w_;
    std::optional<float> point_size_;
    std::optional<uint32_t> vertex_kill_;
  };

  const RegisterFile& register_file_;
  const RegisterFile* register_source_ = nullptr;  // (Session 87, stage 3)
  const RegisterFile& Regs() const { return register_source_ ? *register_source_ : register_file_; }
  const memory::Memory& memory_;

  ShaderInterpreter shader_interpreter_;
};

// wwe2k14 patch (Session 7, EXTENTS-SHADOW): CPU-computed screen extents for the EVENT_WRITE_EXT reports of
// predicated tiling (docs/research/predicated_tiling.md, option (b), stage 1: shadow mode). Settings
// gpu_extents_mode, gpu_extents_shaders, gpu_extents_allow_write and gpu_extents_mode_verify are defined in
// draw_extent_estimator.cpp, which also describes the method and why the bounds are conservative.

class SharedMemory;

// Why a draw (or a report) got no computed extents. The [s7][EXTENTS] summary and the frame-anatomy ext_why
// column print GetScreenExtentsFallbackName.
enum class ScreenExtentsFallback : uint32_t {
  kNone,
  // ScreenExtentsEstimator (one draw).
  kShaderNotAnalyzed,  // vertex shader ucode not analyzed by the backend
  kShaderUnsupported,  // loops, calls too deep, relative register addressing, memexport, unknown opcode
  kDataDependentFlow,  // control flow or a predicated instruction depends on a per-vertex predicate
  kShaderTooLong,      // control flow step budget exceeded
  kIndexSource,        // immediate indices or an unknown index source
  kTessellation,       // tessellated draw
  kPrimitiveType,      // points, lines, rectangles, 2D types, polygon mode with points or lines
  kVteFormat,          // PA_CL_VTE_CNTL: X/Y pre-divided by W, or W given as 1/W
  kWindowOffset,       // non-zero PA_SC_WINDOW_OFFSET (only the tile 0 / untiled case is handled)
  kConstantsLayout,    // SQ_VS_CONST base not 0, or a relative constant index out of range or unknown
  kIndexMemory,        // index buffer outside guest physical memory
  kVertexMemory,       // vertex fetch constant not a vertex one, or a fetch outside its buffer or memory
  kFetchIndex,         // a vertex fetch index that is not the untouched vertex index
  kGpuWritten,         // index or vertex data in pages the GPU wrote (guest memory may be stale)
  kNoSharedMemory,     // the backend can't tell whether data was GPU-written
  kNoPosition,         // position X, Y or W never exported
  kUnbounded,          // position X, Y or W not bounded (depends on data the estimate can't bound)
  kNearPlane,          // W can be <= 0 (or too close to 0) for some vertex
  // ScreenExtentsShadow (one report: the SCREEN_EXT_INIT ... SCREEN_EXT_RPT window).
  kNotListed,   // a draw in the window has a vertex shader that is not in gpu_extents_shaders
  kNotIssued,   // a draw packet in the window that the backend didn't draw (IssueDraw not called or failed)
  kCopy,        // a resolve in the window
  kNoDraw,      // no draw packet in the window
  kNoInit,      // no SCREEN_EXT_INIT seen since the shadow was enabled (or in this pass)
  kCount,
};
const char* GetScreenExtentsFallbackName(ScreenExtentsFallback reason);

struct ScreenExtentsEstimate {
  // ok: every vertex of the draw is inside the bounds below; ok && empty: the draw has no vertices.
  bool ok = false;
  bool empty = false;
  ScreenExtentsFallback fallback = ScreenExtentsFallback::kNone;
  // What EVENT_WRITE_EXT would report: inclusive 8-pixel cells (pixels 8 * min ... 8 * max + 7), widened by one
  // pixel and then one cell on every side, clamped to 0...1023.
  uint16_t min_x = 0, max_x = 0, min_y = 0, max_y = 0;
  // The bounds in screen pixels before the widening (continuous coordinates, no half-pixel offset), for
  // gpu_extents_mode_verify.
  double screen_min_x = 0.0, screen_max_x = 0.0, screen_min_y = 0.0, screen_max_y = 0.0;
  // Vertex indices evaluated (reset indices excluded).
  uint32_t vertex_count = 0;
};

class ScreenExtentsEstimator {
 public:
  ScreenExtentsEstimator(const RegisterFile& register_file, const memory::Memory& memory);
  ~ScreenExtentsEstimator();

  // Bounds the screen-space position of every vertex of the current draw (register state as at the draw).
  // shared_memory (may be null: then nothing is estimated) tells whether index or vertex data is GPU-written.
  void Estimate(const Shader& vertex_shader, SharedMemory* shared_memory, ScreenExtentsEstimate& out);

  enum class VerifyResult { kContained, kMismatch, kNotComparable };
  // gpu_extents_mode_verify: runs ShaderInterpreter vertex by vertex on the draw last passed to Estimate (same
  // register state) and checks that every vertex is inside the estimate's unwidened screen bounds. exact_out =
  // the vertices' own bounds (min x, max x, min y, max y, pixels); bad_vertex_out = the first vertex index
  // outside them.
  VerifyResult Verify(const Shader& vertex_shader, const ScreenExtentsEstimate& estimate, double exact_out[4],
                      uint32_t& bad_vertex_out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Tracks SCREEN_EXT_INIT, draws and SCREEN_EXT_RPT on the command processor thread, estimates the draws of the
// listed shaders in the extents pass (tile 0), counts what predicated tiling would skip in tiles 1-3 and logs
// [s7][EXTENTS] summaries. Created by the command processor the first time gpu_extents_mode is not 0.
class ScreenExtentsShadow {
 public:
  ScreenExtentsShadow(const RegisterFile& register_file, const memory::Memory& memory);
  ~ScreenExtentsShadow();

  // Whether gpu_extents_mode asks for the shadow (one setting read; for the frame boundary while no tracker
  // exists).
  static bool Requested();

  // At each frame boundary (the guest swap): closes the frame's tiling brackets, records the per-frame timeline
  // events, logs the summary about every 300 frames and latches the settings for the next frame. Returns whether
  // the hooks below must be called during the next frame (gpu_extents_mode not 0).
  bool EndFrame(uint32_t frame);

  // What OnDraw did, for the frame-anatomy row.
  struct DrawRow {
    bool attempted = false;
    bool ok = false;
    uint16_t min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    ScreenExtentsFallback fallback = ScreenExtentsFallback::kNone;
    float us = 0.0f;
  };
  // Every draw packet the command processor executed. issued: IssueDraw was called and succeeded; is_copy: it
  // was a resolve.
  void OnDraw(const Shader* vertex_shader, uint64_t bin_select, bool issued, bool is_copy,
              SharedMemory* shared_memory, DrawRow& row);
  // EVENT_WRITE with SCREEN_EXT_INIT.
  void OnScreenExtentsInit(uint64_t bin_select);
  // EVENT_WRITE_EXT. Returns true and the six values to write (min x, max x, min y, max y, min z, max z) only in
  // gpu_extents_mode 2 with gpu_extents_allow_write and computed extents; otherwise the caller writes its
  // full-screen values.
  bool OnScreenExtentsReport(uint64_t bin_select, uint16_t extents_out[6]);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rex::graphics
