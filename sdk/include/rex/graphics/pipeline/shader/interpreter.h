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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <rex/assert.h>
#include <rex/graphics/format/ucode.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/xenos.h>
#include <rex/memory.h>

namespace rex::graphics {

class ShaderInterpreter {
 public:
  ShaderInterpreter(const RegisterFile& register_file, const memory::Memory& memory)
      : register_file_(register_file), memory_(memory) {}
  // wwe2k14 patch (Session 87, D3D layer stage 3): the interpreter is created once, on whichever register file the caller
  // had; a record-fed draw sets the record's image (null = the file it was made with) before each use.
  void SetRegisterSource(const RegisterFile* source) { register_source_ = source; }
  // wwe2k14 patch (Session 92, M4f-B): the constants and fetch words as arrays (the typed front end has no register file): float constants as 512 vec4 (the register
  // layout: vertex 0-255 then pixel 256-511), bool constants (8 words) then loop constants (32 words), 32 fetch constants of 6 words, and the SQ_VS_CONST / SQ_PS_CONST
  // words. nullptr = read the register source (stock, unchanged). EVERY user of a shared interpreter sets or clears it on entry.
  struct ConstantSource {
    const uint32_t* floats = nullptr;
    const uint32_t* bool_loop = nullptr;
    const uint32_t* fetch = nullptr;
    const void* vertex_memory = nullptr;  // (Session 95, item P) vertex fetch addresses are offsets in this block, not guest physical addresses; nullptr = guest memory
    uint32_t vs_const = 0, ps_const = 0;
  };
  void SetConstantSource(const ConstantSource* source) { const_source_ = source; }

  class ExportSink {
   public:
    virtual ~ExportSink() = default;
    virtual void AllocExport(ucode::AllocType /*type*/, uint32_t /*size*/) {}
    virtual void Export(ucode::ExportRegister /*export_register*/, const float* /*value*/,
                        uint32_t /*value_mask*/) {}
  };

  ExportSink* GetExportSink() const { return export_sink_; }
  void SetExportSink(ExportSink* new_export_sink) { export_sink_ = new_export_sink; }

  // wwe2k14 patch (Session 80): vertex fetch dwords read outside their buffer since the last reset (the host's
  // translated fetch reads memory there, this returns 0).
  uint32_t fetches_out_of_range() const { return fetches_out_of_range_; }
  void reset_fetches_out_of_range() { fetches_out_of_range_ = 0; }
  // wwe2k14 patch (Session 80, NativeBoundsRect): the exec slots (fetch / ALU) issued so far in this Execute, counted
  // whether or not a predicate skipped them (the step of an export is steps() while it runs); Execute returns once
  // stop_after_steps slots are issued (UINT32_MAX: never - the caller resets it); flow_was_dynamic: this Execute went
  // through anything that can issue a different sequence of slots for another vertex (a predicate on a slot or on the
  // control flow, a loop).
  uint32_t steps() const { return steps_; }
  void set_stop_after_steps(uint32_t steps) { stop_after_steps_ = steps; }
  // wwe2k14 patch (Session 95): this thread's interpreters decode unsigned packed vertex components at their offsets whatever
  // gpu_interp_packed_exact says (the diagnostic's vertex truth: it must stand for what the shader read).
  static void SetExactPackedFetch(bool exact);
  bool flow_was_dynamic() const { return flow_dynamic_; }
  const float* temp_registers() const { return &temp_registers_[0][0]; }
  float* temp_registers() { return &temp_registers_[0][0]; }

  static bool CanInterpretShader(const Shader& shader) {
    assert_true(shader.is_ucode_analyzed());
    // Texture instructions are not very common in vertex shaders (and not used
    // in Direct3D 9's internal rectangles such as clears) and are extremely
    // complex, not implemented.
    if (shader.uses_texture_fetch_instruction_results()) {
      return false;
    }
    return true;
  }
  void SetShader(xenos::ShaderType shader_type, const uint32_t* ucode) {
    shader_type_ = shader_type;
    ucode_ = ucode;
  }
  void SetShader(const Shader& shader) {
    assert_true(CanInterpretShader(shader));
    SetShader(shader.type(), shader.ucode_dwords());
  }

  void Execute();

 private:
  struct State {
    ucode::VertexFetchInstruction vfetch_full_last;
    uint32_t vfetch_address_dwords;
    float previous_scalar;
    uint32_t call_stack_depth;
    uint32_t call_return_addresses[4];
    uint32_t loop_stack_depth;
    xenos::LoopConstant loop_constants[4];
    uint32_t loop_iterators[4];
    int32_t address_register;
    bool predicate;

    void Reset() { std::memset(this, 0, sizeof(*this)); }

    int32_t GetLoopAddress() const {
      assert_true(loop_stack_depth && loop_stack_depth < 4);
      if (!loop_stack_depth || loop_stack_depth >= 4) {
        return 0;
      }
      xenos::LoopConstant loop_constant = loop_constants[loop_stack_depth];
      // Clamp to the real range specified in the IPR2015-00325 sequencer
      // specification.
      // https://portal.unifiedpatents.com/ptab/case/IPR2015-00325
      return std::min(INT32_C(256),
                      std::max(INT32_C(-256), int32_t(int32_t(loop_iterators[loop_stack_depth]) *
                                                          loop_constant.step +
                                                      loop_constant.start)));
    }
  };

  static float FlushDenormal(float value) {
    uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
    bits &= (bits & UINT32_C(0x7F800000)) ? ~UINT32_C(0) : (UINT32_C(1) << 31);
    return *reinterpret_cast<const float*>(&bits);
  }

  uint32_t GetTempRegisterIndex(uint32_t address, bool is_relative) const {
    return (int32_t(address) + (is_relative ? state_.GetLoopAddress() : 0)) &
           ((UINT32_C(1) << xenos::kMaxShaderTempRegistersLog2) - 1);
  }
  const float* GetTempRegister(uint32_t address, bool is_relative) const {
    return temp_registers_[GetTempRegisterIndex(address, is_relative)];
  }
  float* GetTempRegister(uint32_t address, bool is_relative) {
    return temp_registers_[GetTempRegisterIndex(address, is_relative)];
  }
  const std::array<float, 4> GetFloatConstant(uint32_t address, bool is_relative,
                                              bool relative_address_is_a0) const;

  void ExecuteAluInstruction(ucode::AluInstruction instr);
  void StoreFetchResult(uint32_t dest, bool is_dest_relative, uint32_t swizzle, const float* value);
  void ExecuteVertexFetchInstruction(ucode::VertexFetchInstruction instr);

  const RegisterFile& register_file_;
  const RegisterFile* register_source_ = nullptr;  // (Session 87, stage 3)
  const ConstantSource* const_source_ = nullptr;   // (Session 92, M4f-B)
  const RegisterFile& Regs() const { return register_source_ ? *register_source_ : register_file_; }
  const memory::Memory& memory_;

  ExportSink* export_sink_ = nullptr;
  // wwe2k14 patch (Session 80): vertex fetch dwords outside their fetch constant's range (read as 0 here).
  uint32_t fetches_out_of_range_ = 0;
  uint32_t steps_ = 0;  // (Session 80: see steps())
  uint32_t stop_after_steps_ = UINT32_MAX;
  bool flow_dynamic_ = false;

  xenos::ShaderType shader_type_ = xenos::ShaderType::kVertex;
  const uint32_t* ucode_ = nullptr;

  // For both inputs and locals.
  float temp_registers_[xenos::kMaxShaderTempRegisters][4];

  State state_;
};

}  // namespace rex::graphics
