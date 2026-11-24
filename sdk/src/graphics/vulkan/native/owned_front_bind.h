// wwe2k14 patch (Session 92, M4c; docs\native\d3d\S92_M3_M5_DESIGN.md section 2 "M4c", contract S92_M4C_CONTRACT.md): the shader constants and
// the binding contents built from the record decoder's typed state, as a SHADOW.
//
// For each draw M4b built (owned_front_draw.cpp), OwnedFrontBindShadow builds in HOST memory:
//   - the XeSystemConstants block (the parts UpdateSystemConstantValuesFull fills) from the typed fields,
//   - the vertex / pixel float constant banks through the shader's constant map, the bool / loop constants and the 32 fetch constants,
//   - the texture keys (and with them the texture signs) of the used fetch constants, and the vertex buffer ranges, from the fetch words,
// and compares them with what the OLD path used for the same draw: the system block it computed (system_constants_, uploaded unchanged into
// the ring), the banks it gathered from the register image, the texture cache's active bindings, the processor's vertex buffer states.
// It changes nothing the old path sees: no ring upload, no descriptor allocation or write, no texture cache load / upload / watch, no sampler
// creation, no RequestRange. Latch: gpu_d3d_owned_front bit 2 (value 4; with bits 0 + 1: value 7).
//
// The processor's private members are reached by OwnedFrontAccess (owned_front_draw.cpp), which gathers them into OwnedBindInputs; this file
// is pure logic over plain data.
#pragma once

#include <cstdint>
#include <string>

#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/primitive_processor.h>
#include <rex/graphics/registers.h>

#include "owned_record.h"

namespace rex::graphics::vulkan {

struct OwnedBindInputs {
  const OwnedRecordState* s = nullptr;  // the typed state at this draw (banks included)
  uint64_t frame = 0;
  uint32_t kind = 0;  // OwnedFrontClaimKind
  uint64_t vs_hash = 0, ps_hash = 0;
  const Shader* vs = nullptr;
  const Shader* ps = nullptr;

  // ---- what M4b derived on the way (the typed side's own values)
  bool polygonal = false;
  PrimitiveProcessor::ProcessingResult primitive{};
  bool shader_32bit_index_dma = false;
  float ndc_scale[3] = {};
  float ndc_offset[3] = {};
  uint32_t xy_extent[2] = {};
  reg::RB_DEPTHCONTROL depth_control{};
  uint32_t color_mask = 0;
  uint32_t used_texture_mask = 0;
  uint32_t scale_x = 1, scale_y = 1;

  // ---- render target cache / device state the old derivation read
  bool fsi = false, host_rt = false, rg16_truncated = false, rgba16_truncated = false, native_truth = false, gamma_as_unorm16 = false;
  bool image_view_format_swizzle = true;
  // The texture cache's own decisions that are not a function of the fetch words (used textures only): resolution-scaled textures, and
  // (without imageViewFormatSwizzle) the host swizzle. Taken as inputs, counted.
  uint32_t scaled_from_cache = 0;
  uint32_t host_swizzle_from_cache[32] = {};

  // ---- the stage binding lists the old path bound (as the draw's translations gave them)
  struct TextureBinding {
    uint8_t fetch_constant;
    bool is_signed;
  };
  TextureBinding textures[2][32] = {};  // [0] vertex, [1] pixel
  uint32_t texture_count[2] = {};
  uint32_t sampler_count[2] = {};

  // ---- what the old path's constants were made from, for this draw. The ring itself is not readable from the access struct
  // (ConstantsDynamicOffsets, which holds the mappings, is defined in vk\command_processor.cpp: hand-back with the exact lines); its bytes
  // are these sources uploaded unchanged: the system block is the processor's system_constants_ (memcpy'd into the ring), the banks are
  // the draw's register image (the old gather).
  const SpirvShaderTranslator::SystemConstants* old_sys = nullptr;
  const uint32_t* old_vs_float = nullptr;   // 256 * 4 words (register image, SHADER_CONSTANT_000_X)
  const uint32_t* old_ps_float = nullptr;   // 256 * 4 words (SHADER_CONSTANT_256_X)
  const uint32_t* old_bool_loop = nullptr;  // 8 + 32 words
  const uint32_t* old_fetch = nullptr;      // 32 * 6 words

  // ---- the texture cache's active binding per fetch constant (the old path's RequestTextures result)
  struct ActiveTexture {
    bool valid;
    uint32_t base_page, format, width, height;
  };
  ActiveTexture active[32] = {};


  // ---- M4c views and samplers (gathered by OwnedFrontAccess with find-only lookups into the texture cache)
  struct ViewCheck {
    bool compared = false;  // false: not compared (3D sampled as 1D/2D: the old path creates that view lazily)
    uint64_t old_view = 0, typed_view = 0;
  };
  ViewCheck view[2][32] = {};
  uint32_t skipped_3d_views = 0;
  uint32_t skipped_native_views = 0;  // the old path binds the native renderer's owned / served view
  struct SamplerCheck {
    uint32_t old_params = 0, typed_params = 0;
    uint64_t old_sampler = 0, typed_sampler = 0;
  };
  SamplerCheck sampler[2][32] = {};
  uint32_t sampler_checked[2] = {};  // entries of sampler[stage] filled
  uint32_t sampler_old_count[2] = {};  // the old path's list sizes

  // ---- the old path's vertex buffer states (96 vertex fetch constants)
  struct VertexState {
    uint32_t address, size;
  };
  VertexState vertex_state[96] = {};
};

// M4f-A: BuildSystemConstants over `in` into `out` (the processor's system_constants_).
void OwnedFrontBindBuildSystem(const OwnedBindInputs& in, SpirvShaderTranslator::SystemConstants& out);
// Builds and compares one draw. Returns false when it could not build (the reason is counted in the window).
bool OwnedFrontBindShadow(const OwnedBindInputs& in);
// Counted by the caller for a draw whose bind shadow was not attempted (reason names as the window line prints them).
void OwnedFrontBindNote(const char* reason);
// The layout lookup result M4b made: found by the counts (true) or fell back to the old description's (false).
void OwnedFrontBindNoteLayout(bool found_find_only, bool matches_old);
// The `[d3d][front] M4c` window line for frames [first, now].
void OwnedFrontBindPrintWindow(uint64_t first_frame, uint64_t now_frame);
// M4e: the counters of an inline draw's claim are their own (called by the draw shadow around it); the summary text goes in the M4e window line and resets them.
void OwnedFrontBindBucket(bool inline_draws);
std::string OwnedFrontBindInlineSummary();

}  // namespace rex::graphics::vulkan
