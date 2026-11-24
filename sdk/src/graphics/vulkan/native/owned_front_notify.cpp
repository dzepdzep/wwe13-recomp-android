// wwe2k14 patch (Session 87 stage 3 / Session 92 M4f-C): the notifications a stored block of constants makes (D3dFeedAccess::Notify) and the cut's processor-side
// effects (D3dFeedAccess::Cut*). Moved out of d3d_feed.cpp (the old D3D feed, left out of the REX_TRUE_NATIVE_ONLY proof build) so the decoder-as-executor can
// make them without it. SDK-internal.
#include "d3d_feed.h"

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/cvar.h>
#include <rex/graphics/d3d_feed.h>
#include <rex/graphics/d3d_noring.h>
#include <rex/graphics/free_counters.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/register_file.h>
#include <rex/graphics/registers.h>

#include <algorithm>
#include <cstdint>

REXCVAR_DECLARE(bool, gpu_native_frame_draw);

REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)

namespace rex::graphics::vulkan {

namespace {
constexpr uint32_t kConstLo = 0x4000;  // floats start

bool RangeHasUsage(const uint64_t* usage_map, uint32_t first_constant, uint32_t last_constant) {
  // (WriteRegistersFromMem's lambda: any bit of [first, last] set in the 256-bit map.)
  if (first_constant > last_constant) return false;
  const uint32_t first_word = first_constant >> 6, last_word = last_constant >> 6;
  const uint32_t first_bit = first_constant & 63, last_bit = last_constant & 63;
  if (first_word == last_word) {
    const uint32_t bit_count = last_bit - first_bit + 1;
    const uint64_t mask = bit_count == 64 ? ~uint64_t(0) : ((uint64_t(1) << bit_count) - 1) << first_bit;
    return (usage_map[first_word] & mask) != 0;
  }
  if (usage_map[first_word] & (~uint64_t(0) << first_bit)) return true;
  for (uint32_t word = first_word + 1; word < last_word; ++word) {
    if (usage_map[word]) return true;
  }
  const uint64_t last_mask = last_bit == 63 ? ~uint64_t(0) : ((uint64_t(1) << (last_bit + 1)) - 1);
  return (usage_map[last_word] & last_mask) != 0;
}
}  // namespace

// What VulkanCommandProcessor::WriteRegistersFromMem / WriteRegister notify for a piece of one class, made from the record's
// block (the register file is not touched).
void D3dFeedAccess::Notify(VCP& cp, Piece piece, uint32_t first, uint32_t count, bool changed) {
  using SPT = SpirvShaderTranslator;
  const uint32_t last = first + count - 1;
  switch (piece) {
    case Piece::kFloat: {
      cp.RepeatNoteRegisters(first, count);  // (floats: any write counts)
      if (cp.frame_open_) {
        const uint32_t first_constant = (first - kConstLo) >> 2;
        const uint32_t last_constant = (last - kConstLo) >> 2;
        if (first_constant < 256 &&
            RangeHasUsage(cp.current_float_constant_map_vertex_, first_constant, std::min(last_constant, 255u))) {
          cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SPT::kConstantBufferFloatVertex);
        }
        if (last_constant >= 256 && RangeHasUsage(cp.current_float_constant_map_pixel_,
                                                  first_constant >= 256 ? first_constant - 256 : 0, last_constant - 256)) {
          cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SPT::kConstantBufferFloatPixel);
        }
      }
    } break;
    case Piece::kFetch: {
      if (changed) cp.RepeatNoteRegisters(first, count);
      cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SPT::kConstantBufferFetch);
      const uint32_t first_dword = first - 0x4800, last_dword = last - 0x4800;
      if (first_dword < 6) rex::graphics::free_counters::NoteFetch0Write();  // (Session 92, M3b review P5)
      if (cp.texture_cache_) cp.texture_cache_->TextureFetchConstantsWritten(first_dword / 6, last_dword / 6);
      cp.InvalidateVertexBufferResidencyRange(first_dword / 2, last_dword / 2);
    } break;
    case Piece::kBoolLoop:
      if (changed) cp.RepeatNoteRegisters(first, count);
      cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SPT::kConstantBufferBoolLoop);
      break;
    case Piece::kOther:
      if (changed) cp.RepeatNoteRegisters(first, count);
      break;
  }
}
// ---- Session 92, M4f-C: the cut (see d3d_feed.h). The decoder stores a block into its typed state, then asks for the notifications: they are Notify's
// own (the one body above), preceded by the fetch source (so the texture cache and the object table read the typed fetch words) and followed by the
// census of what was really marked.
void D3dFeedAccess::CutFetchSource(CommandProcessor& base, const uint32_t* words) {
  VCP& cp = static_cast<VCP&>(base);
  if (cp.texture_cache_) cp.texture_cache_->SetFetchSource(words);
}

void D3dFeedAccess::CutNotify(CommandProcessor& base, Piece piece, uint32_t first, uint32_t count, bool changed, const uint32_t* fetch_words) {
  using SPT = SpirvShaderTranslator;
  namespace fc = rex::graphics::free_counters;
  VCP& cp = static_cast<VCP&>(base);
  if (piece == Piece::kOther) return;  // (the old Notify's kOther made only RepeatNoteRegisters: the decoder's dirty epochs replace it)
  if (fetch_words && cp.texture_cache_) cp.texture_cache_->SetFetchSource(fetch_words);
  const uint32_t before = cp.current_constant_buffers_up_to_date_;
  Notify(cp, piece, first, count, changed);
  const uint32_t marked = before & ~cp.current_constant_buffers_up_to_date_;  // the buffers that were up to date and are not any more
  fc::CutCounters& c = fc::g_cut;
  switch (piece) {
    case Piece::kFloat:
      ++c.float_runs;
      c.float_marks_vs += (marked >> SPT::kConstantBufferFloatVertex) & 1u;
      c.float_marks_ps += (marked >> SPT::kConstantBufferFloatPixel) & 1u;
      break;
    case Piece::kFetch:
      ++c.fetch_runs;
      c.fetch_marks += (marked >> SPT::kConstantBufferFetch) & 1u;
      c.fetch_texture_notes += cp.texture_cache_ ? 1u : 0u;  // (TextureFetchConstantsWritten(first, last) + the object table's OnFetchWritten)
      ++c.fetch_vertex_ranges;                               // (InvalidateVertexBufferResidencyRange)
      break;
    case Piece::kBoolLoop:
      ++c.bool_runs;
      c.bool_marks += (marked >> SPT::kConstantBufferBoolLoop) & 1u;
      break;
    default:
      break;
  }
}

// (Session 95, gpu_d3d_wall bit 4) A fetch run with the words that really differ: `changed_count` dwords from `changed_first`, 0 when the run wrote what the bank held. What the fetch words feed is
// marked for those only - the fetch constant buffer (when any differs), the texture slots, the vertex streams. The object table's Bind bookkeeping runs for the whole run (a Bind belongs to the write);
// a slot whose object changed with equal words, or any write after an object was defined or destroyed, is out of sync all the same.
void D3dFeedAccess::CutFetchChanged(CommandProcessor& base, uint32_t first, uint32_t count, uint32_t changed_first, uint32_t changed_count, const uint32_t* fetch_words) {
  using SPT = SpirvShaderTranslator;
  namespace fc = rex::graphics::free_counters;
  VCP& cp = static_cast<VCP&>(base);
  if (cp.texture_cache_) cp.texture_cache_->SetFetchSource(fetch_words);
  fc::CutCounters& c = fc::g_cut;
  ++c.fetch_runs;
  const uint32_t first_dword = first - 0x4800, last_dword = first_dword + count - 1;
  if (first_dword < 6) rex::graphics::free_counters::NoteFetch0Write();  // (Session 92, M3b review P5)
  if (changed_count) {
    cp.RepeatNoteRegisters(changed_first, changed_count);
    c.fetch_marks += (cp.current_constant_buffers_up_to_date_ >> SPT::kConstantBufferFetch) & 1u;
    cp.current_constant_buffers_up_to_date_ &= ~(UINT32_C(1) << SPT::kConstantBufferFetch);
    const uint32_t changed_first_dword = changed_first - 0x4800, changed_last_dword = changed_first_dword + changed_count - 1;
    cp.InvalidateVertexBufferResidencyRange(changed_first_dword / 2, changed_last_dword / 2);
    ++c.fetch_vertex_ranges;
  }
  if (cp.texture_cache_) {
    const uint32_t changed_first_slot = changed_count ? (changed_first - 0x4800) / 6 : UINT32_MAX;
    const uint32_t changed_last_slot = changed_count ? (changed_first - 0x4800 + changed_count - 1) / 6 : 0;
    cp.texture_cache_->TextureFetchConstantsChanged(first_dword / 6, last_dword / 6, changed_first_slot, changed_last_slot);
    ++c.fetch_texture_notes;
  }
}

uint64_t D3dFeedAccess::CutLoadShader(CommandProcessor& base, uint32_t type, uint32_t address, const uint32_t* microcode, uint32_t dwords) {
  VCP& cp = static_cast<VCP&>(base);
  // (Session 95, gpu_d3d_wall bit 32) ~470 shader items a frame for a few dozen shaders, each hashing its microcode: the first item of a frame for an address hashes and looks it up, the frame's others
  // take its answer (a shader in use is not rewritten inside a frame; the next frame hashes again, and a cache clear ends the answers).
  struct KeptShader {
    uint64_t frame = UINT64_MAX, cache_gen = 0;
    uint32_t address = 0, dwords = 0, type = 0;
    Shader* shader = nullptr;
  };
  static KeptShader kept[1024];
  KeptShader& k = kept[((address >> 2) * 0x9E3779B1u ^ dwords * 0x85EBCA77u ^ type) >> 22];
  const bool keep = address && (REXCVAR_GET(gpu_d3d_wall) & 32u);
  Shader* shader;
  if (keep && k.frame == cp.frame_current_ && k.cache_gen == cp.lean_cache_gen_ && k.address == address && k.dwords == dwords && k.type == type) {
    shader = k.shader;
  } else {
    // (the token handler's SHADER item: LoadShader keyed by the microcode itself, the active shader of the stage)
    shader = cp.LoadShader(type == 0 ? xenos::ShaderType::kVertex : xenos::ShaderType::kPixel, address, microcode, dwords);
    k.frame = cp.frame_current_;
    k.cache_gen = cp.lean_cache_gen_;
    k.address = address;
    k.dwords = dwords;
    k.type = type;
    k.shader = shader;
  }
  if (type == 0) {
    cp.active_vertex_shader_ = shader;
  } else {
    cp.active_pixel_shader_ = shader;
  }
  return shader ? shader->ucode_data_hash() : 0;
}

bool D3dFeedAccess::CutReady(CommandProcessor& base) {
  VCP& cp = static_cast<VCP&>(base);
  return cp.native_ && cp.native_->enabled();
}

uint64_t D3dFeedAccess::CutFrame(CommandProcessor& base) { return static_cast<VCP&>(base).frame_current_; }

void D3dFeedAccess::CutBinOp(CommandProcessor& base, uint32_t op, uint32_t a) {
  VCP& cp = static_cast<VCP&>(base);
  // The tiling predication state, as the four SET_BIN_* packet cases set it (each side has its own) - the Registers item's code in d3d_noring.cpp.
  const auto apply = [&] {
    if (op == d3d_noring_field::kRegsOpBinMaskLo) {
      cp.bin_mask_ = (cp.bin_mask_ & 0xFFFFFFFF00000000ull) | a;
      cp.BinMaskWritten("mask_lo");
    } else if (op == d3d_noring_field::kRegsOpBinMaskHi) {
      cp.bin_mask_ = (cp.bin_mask_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(a) << 32);
      cp.BinMaskWritten("mask_hi");
    } else {
      const uint64_t old_select = cp.bin_select_;
      if (op == d3d_noring_field::kRegsOpBinSelectLo) {
        cp.bin_select_ = (cp.bin_select_ & 0xFFFFFFFF00000000ull) | a;
        cp.BinSelectWritten("sel_lo", old_select);
      } else {
        cp.bin_select_ = (cp.bin_select_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(a) << 32);
        cp.BinSelectWritten("sel_hi", old_select);
      }
    }
  };
  if (cp.nf_ && cp.nf_role_ == VCP::NfRole::kAnswers) {
    if (REXCVAR_GET(gpu_native_frame_draw)) {  // (the drawing side's predication first, then the answers side's)
      cp.NfEnterDraw();
      apply();
      cp.NfLeaveDraw();
    }
    apply();
  } else {
    apply();
  }
}

}  // namespace rex::graphics::vulkan
