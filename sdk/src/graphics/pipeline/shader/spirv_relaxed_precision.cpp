// wwe2k14 patch (Session 73, gpu_native_ps_relaxed_precision): see spirv_relaxed_precision.h.

#include "spirv_relaxed_precision.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>

#define SPV_ENABLE_UTILITY_CODE
#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/spirv.hpp>

#ifndef REX_SPIRV_RELAXED_PRECISION_STANDALONE
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(gpu_native_ps_relaxed_precision, false, "GPU/Native",
                    "Session 73 (phones): mark the 32-bit float maths and texture sample results of the translated "
                    "pixel shaders RelaxedPrecision (mediump), so Adreno / Mali may run them at 16 bits; texture "
                    "coordinates, indices, depth and integer conversions stay full precision. Desktop drivers ignore "
                    "the decoration. Applies to shader modules made after it is set");
REXCVAR_DEFINE_STRING(gpu_native_ps_relaxed_only, "", "GPU/Native",
                      "Session 95 (r3gpu, a look lever, default empty): with gpu_native_ps_relaxed_precision (and gpu_native_ps_half), only these pixel shaders (comma-separated hex ucode hashes) are relaxed: the "
                      "wrestler materials can go to 16-bit maths without the arena's bright lamps. Empty = every shader the other switches allow. The list narrows the set before the character rule: a listed wrestler shader stays full precision unless gpu_native_ps_full_precision_characters is false");
REXCVAR_DEFINE_BOOL(gpu_native_ps_full_precision_characters, true, "GPU/Native",
                    "Session 73: with gpu_native_ps_relaxed_precision, the characters' pixel shaders (paired with a "
                    "gpu_dlss_objmv_shaders vertex shader, or listed in gpu_native_ps_full_precision_shaders) stay at "
                    "full precision (half precision darkened the wrestlers' skin)");
// The pixel shaders seen with the character vertex shaders in Session 73's runs (Lesnar, Bryan, the s21_lb replay's
// entrances), for the very first draws of an install before a pair is noted. Then DA494C9383E18961: no character
// vertex shader draws it, but at half precision it alone tints the wrestlers' skin (darker, more orange; bisected on
// the s21_lb replay's match shot, fast-forwarded runs 4741-4745: skin region 14,433 -> 4 pixels off by more than 16).
REXCVAR_DEFINE_STRING(gpu_native_ps_full_precision_shaders,
                      "046D67063E250B56,1B66FF87A6433D53,1D20CB5D36B02E5B,2B8A869CF01395DB,3C06B3329BE81996,"
                      "4C9D02186B6E5E9D,58782AADE08EE063,741439F55F5B15D4,91BF021CA7C0C1A5,925085B612A44F10,"
                      "9864A5F6683152D7,9C2FD0E90E71F4A3,9F8F910305279A11,AFB4E9E073971D7A,B08B5E734D48F5D0,"
                      "B1F58C0AA0E3C83A,BBB2E028DCF36E8D,C61723668154F870,CA3E51C5BA993041,D5604D4251D60877,"
                      "E86BD728B5D916BA,F75AE782D8A42A61,FEE45804EDD0EC05,DA494C9383E18961",
                      "GPU/Native",
                      "Session 73: pixel shaders gpu_native_ps_relaxed_precision leaves at full precision (comma-separated "
                      "hex ucode hashes)");
#endif

namespace rex::graphics::spirv_relaxed_precision {

namespace {

constexpr uint32_t kAnyElement = UINT32_MAX;

struct Inst {
  uint32_t offset;
  uint32_t opcode;
  uint32_t word_count;
  bool has_type;
  bool has_result;
};

bool IsAnnotation(uint32_t op) {
  switch (op) {
    case spv::OpDecorate:
    case spv::OpMemberDecorate:
    case spv::OpDecorationGroup:
    case spv::OpGroupDecorate:
    case spv::OpGroupMemberDecorate:
    case spv::OpDecorateId:
    case spv::OpDecorateString:
    case spv::OpMemberDecorateString:
      return true;
    default:
      return false;
  }
}

// Sections 1-7 of the logical layout (before the annotations).
bool IsPreamble(uint32_t op) {
  switch (op) {
    case spv::OpCapability:
    case spv::OpExtension:
    case spv::OpExtInstImport:
    case spv::OpMemoryModel:
    case spv::OpEntryPoint:
    case spv::OpExecutionMode:
    case spv::OpExecutionModeId:
    case spv::OpString:
    case spv::OpSourceExtension:
    case spv::OpSource:
    case spv::OpSourceContinued:
    case spv::OpName:
    case spv::OpMemberName:
    case spv::OpModuleProcessed:
      return true;
    default:
      return false;
  }
}

bool IsImageSample(uint32_t op) { return op >= spv::OpImageSampleImplicitLod && op <= spv::OpImageSampleProjDrefExplicitLod; }

// Image instructions whose operands after the image (coordinate, Dref, LOD / bias / gradients / offsets / sample)
// must stay full precision.
bool IsImageWithCoordinate(uint32_t op) {
  switch (op) {
    case spv::OpImageSampleImplicitLod:
    case spv::OpImageSampleExplicitLod:
    case spv::OpImageSampleDrefImplicitLod:
    case spv::OpImageSampleDrefExplicitLod:
    case spv::OpImageSampleProjImplicitLod:
    case spv::OpImageSampleProjExplicitLod:
    case spv::OpImageSampleProjDrefImplicitLod:
    case spv::OpImageSampleProjDrefExplicitLod:
    case spv::OpImageFetch:
    case spv::OpImageGather:
    case spv::OpImageDrefGather:
    case spv::OpImageRead:
    case spv::OpImageQueryLod:
    case spv::OpImageSparseSampleImplicitLod:
    case spv::OpImageSparseSampleExplicitLod:
    case spv::OpImageSparseSampleDrefImplicitLod:
    case spv::OpImageSparseSampleDrefExplicitLod:
    case spv::OpImageSparseSampleProjImplicitLod:
    case spv::OpImageSparseSampleProjExplicitLod:
    case spv::OpImageSparseSampleProjDrefImplicitLod:
    case spv::OpImageSparseSampleProjDrefExplicitLod:
    case spv::OpImageSparseFetch:
    case spv::OpImageSparseGather:
    case spv::OpImageSparseDrefGather:
    case spv::OpImageSparseRead:
    case spv::OpImageTexelPointer:
      return true;
    default:
      return false;
  }
}

bool IsAccessChain(uint32_t op) {
  return op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain || op == spv::OpPtrAccessChain ||
         op == spv::OpInBoundsPtrAccessChain;
}

bool IsDecoratedArithmetic(uint32_t op) {
  switch (op) {
    case spv::OpFNegate:
    case spv::OpFAdd:
    case spv::OpFSub:
    case spv::OpFMul:
    case spv::OpFDiv:
    case spv::OpFRem:
    case spv::OpFMod:
    case spv::OpVectorTimesScalar:
    case spv::OpMatrixTimesScalar:
    case spv::OpVectorTimesMatrix:
    case spv::OpMatrixTimesVector:
    case spv::OpMatrixTimesMatrix:
    case spv::OpDot:
      return true;
    default:
      return IsImageSample(op);
  }
}

// GLSL.std.450 instructions that are safe at fp16: range-preserving or simple products. Left out: the
// transcendentals and roots (Exp / Exp2 / Log / Log2 / Pow / Sqrt / InverseSqrt / trig / Atan2: fp16 range and the
// drivers' half-precision approximations), Length / Distance / Normalize (an internal sum of squares overflows fp16
// for components above ~181), Refract (a sqrt), the rounding family (Floor / Ceil / Fract / Trunc / Round: mostly
// applied to large values - wrapping, time, packing - where an 11-bit mantissa is wrong, and cheap anyway),
// Modf / Frexp / Ldexp, Determinant / MatrixInverse, Pack / Unpack and the interpolation functions.
bool IsDecoratedGlsl(uint32_t instruction) {
  switch (instruction) {
    case GLSLstd450FAbs:
    case GLSLstd450FSign:
    case GLSLstd450FMin:
    case GLSLstd450FMax:
    case GLSLstd450FClamp:
    case GLSLstd450NMin:
    case GLSLstd450NMax:
    case GLSLstd450NClamp:
    case GLSLstd450FMix:
    case GLSLstd450Step:
    case GLSLstd450SmoothStep:
    case GLSLstd450Fma:
    case GLSLstd450Cross:
    case GLSLstd450FaceForward:
    case GLSLstd450Reflect:
      return true;
    default:
      return false;
  }
}

}  // namespace

bool AddRelaxedPrecision(const uint32_t* words, size_t word_count, std::vector<uint32_t>& out, Stats* stats_out) {
  Stats stats;
  if (stats_out) *stats_out = stats;
  if (!words || word_count < 5 || words[0] != spv::MagicNumber) return false;
  const uint32_t bound = words[3];
  if (bound == 0 || bound > (1u << 24)) return false;

  // Parse.
  std::vector<Inst> insts;
  insts.reserve(word_count / 4);
  for (size_t at = 5; at < word_count;) {
    const uint32_t wc = words[at] >> 16, op = words[at] & 0xFFFF;
    if (!wc || at + wc > word_count) return false;
    bool has_result = false, has_type = false;
    spv::HasResultAndType(spv::Op(op), &has_result, &has_type);
    if (wc < 1u + uint32_t(has_result) + uint32_t(has_type)) return false;
    insts.push_back({uint32_t(at), op, wc, has_type, has_result});
    at += wc;
  }

  std::vector<int32_t> def(bound, -1);  // the instruction defining an id
  std::vector<uint8_t> f32_type(bound, 0), int_type(bound, 0), pointer_type(bound, 0), relaxed(bound, 0),
      depth_var(bound, 0);
  std::vector<uint32_t> int_constant_value(bound, kAnyElement);
  uint32_t glsl_set = 0;
  size_t entry_points = 0, last_annotation = SIZE_MAX, first_after_preamble = SIZE_MAX;
  // Functions: parameters in order, return values, calls.
  std::map<uint32_t, std::vector<uint32_t>> function_params, function_returns;
  std::vector<size_t> calls, stores, copies;
  std::vector<uint32_t> seeds;  // ids to keep full precision from the start
  uint32_t current_function = 0;
  // Blocks (for the reaching stores of Function variables).
  struct Block {
    size_t begin, end;  // the OpLabel, the terminator
    std::vector<uint32_t> successors;  // labels
    std::vector<int32_t> predecessors;
  };
  std::vector<Block> blocks;
  std::vector<int32_t> label_block(bound, -1), inst_block(insts.size(), -1);
  std::vector<uint8_t> function_var(bound, 0);
  int32_t current_block = -1;

  for (size_t i = 0; i < insts.size(); ++i) {
    const Inst& in = insts[i];
    const uint32_t* w = words + in.offset;
    if (first_after_preamble == SIZE_MAX && !IsPreamble(in.opcode)) first_after_preamble = i;
    if (IsAnnotation(in.opcode)) last_annotation = i;
    if (in.has_result) {
      const uint32_t id = w[in.has_type ? 2 : 1];
      if (id >= bound) return false;
      if (def[id] < 0) def[id] = int32_t(i);
    }
    if (in.opcode == spv::OpLabel) {
      current_block = int32_t(blocks.size());
      blocks.push_back({i, i, {}, {}});
      label_block[w[1]] = current_block;
    }
    inst_block[i] = current_block;
    switch (in.opcode) {
      case spv::OpBranch:
        if (current_block >= 0 && in.word_count >= 2) blocks[current_block].successors = {w[1]};
        break;
      case spv::OpBranchConditional:
        if (current_block >= 0 && in.word_count >= 4) blocks[current_block].successors = {w[2], w[3]};
        break;
      case spv::OpSwitch:
        if (current_block >= 0 && in.word_count >= 3) {
          // (selector, default, then literal / label pairs; a 64-bit selector has two-word literals)
          uint32_t literal_words = 1;
          if (w[1] < bound && def[w[1]] >= 0) {
            const uint32_t type = words[insts[def[w[1]]].offset + 1];
            if (type < bound && def[type] >= 0 && insts[def[type]].opcode == spv::OpTypeInt &&
                words[insts[def[type]].offset + 2] == 64) {
              literal_words = 2;
            }
          }
          auto& s = blocks[current_block].successors;
          s = {w[2]};
          for (uint32_t k = 3 + literal_words; k < in.word_count; k += literal_words + 1) s.push_back(w[k]);
        }
        break;
      case spv::OpVariable:
        if (in.word_count >= 4 && w[3] == spv::StorageClassFunction) function_var[w[2]] = 1;
        break;
      default:
        break;
    }
    if (current_block >= 0) blocks[current_block].end = i;
    switch (in.opcode) {
      case spv::OpEntryPoint:
        if (in.word_count < 3 || w[1] != spv::ExecutionModelFragment) return false;  // fragment-only modules
        ++entry_points;
        break;
      case spv::OpExtInstImport:
        if (in.word_count >= 3 && std::strncmp(reinterpret_cast<const char*>(w + 2), "GLSL.std.450",
                                               (in.word_count - 2) * 4) == 0) {
          glsl_set = w[1];
        }
        break;
      case spv::OpTypeFloat:
        if (in.word_count >= 3 && w[2] == 32) f32_type[w[1]] = 1;
        break;
      case spv::OpTypeInt:
        int_type[w[1]] = 1;
        break;
      case spv::OpTypePointer:
        pointer_type[w[1]] = 1;
        break;
      case spv::OpTypeVector:
      case spv::OpTypeMatrix:
        if (in.word_count >= 3 && w[2] < bound) {
          f32_type[w[1]] = f32_type[w[2]];
          int_type[w[1]] = int_type[w[2]];
        }
        break;
      case spv::OpConstant:
        if (in.word_count >= 4 && w[1] < bound && int_type[w[1]] && w[2] < bound) int_constant_value[w[2]] = w[3];
        break;
      case spv::OpDecorate:
        if (in.word_count >= 3 && w[1] < bound) {
          if (w[2] == spv::DecorationRelaxedPrecision) relaxed[w[1]] = 1;
          if (in.word_count >= 4 && w[2] == spv::DecorationBuiltIn && w[3] == spv::BuiltInFragDepth) {
            depth_var[w[1]] = 1;
          }
        }
        break;
      case spv::OpFunction:
        current_function = w[2];
        break;
      case spv::OpFunctionParameter:
        function_params[current_function].push_back(w[2]);
        break;
      case spv::OpReturnValue:
        if (in.word_count >= 2) function_returns[current_function].push_back(w[1]);
        break;
      case spv::OpFunctionEnd:
        current_function = 0;
        current_block = -1;
        break;
      case spv::OpFunctionCall:
        calls.push_back(i);
        break;
      case spv::OpStore:
        if (in.word_count >= 3) stores.push_back(i);
        break;
      case spv::OpCopyMemory:
        if (in.word_count >= 3) copies.push_back(i);
        break;
      default:
        break;
    }
    // Sinks: operands that must be computed at full precision.
    if (IsImageWithCoordinate(in.opcode) && in.has_result) {
      for (uint32_t k = 4; k < in.word_count; ++k) seeds.push_back(w[k]);  // after the (sampled) image
    } else if (in.opcode == spv::OpImageWrite) {
      for (uint32_t k = 2; k < in.word_count; ++k) seeds.push_back(w[k]);  // coordinate, texel, operands
    } else if (IsAccessChain(in.opcode)) {
      for (uint32_t k = 4; k < in.word_count; ++k) seeds.push_back(w[k]);  // indices
    } else if (in.opcode == spv::OpVectorExtractDynamic && in.word_count >= 5) {
      seeds.push_back(w[4]);
    } else if (in.opcode == spv::OpVectorInsertDynamic && in.word_count >= 6) {
      seeds.push_back(w[5]);
    } else if (in.opcode == spv::OpBitcast && in.word_count >= 4) {
      seeds.push_back(w[3]);  // bit-exact
    } else if (in.has_type && in.has_result && w[1] < bound && int_type[w[1]]) {
      // An integer value (made from floats by conversions, packing, ...): what it is made of stays full precision.
      seeds.push_back(w[2]);
    }
  }
  if (!entry_points) return false;

  // A pointer to (base, outermost index): the register array's elements are told apart by their constant index.
  // depth = the number of indices from the base (0 = the whole variable, 1 = a whole element).
  auto resolve_depth = [&](uint32_t pointer, uint32_t& element, uint32_t& depth) -> uint32_t {
    element = kAnyElement;
    depth = 0;
    for (int guard = 0; guard < 64 && pointer < bound && def[pointer] >= 0; ++guard) {
      const Inst& d = insts[def[pointer]];
      const uint32_t* w = words + d.offset;
      if (IsAccessChain(d.opcode) && d.word_count >= 4) {
        if (d.opcode == spv::OpPtrAccessChain || d.opcode == spv::OpInBoundsPtrAccessChain) {
          element = kAnyElement;
          depth += 8;  // (never a whole element)
        } else {
          element = d.word_count >= 5 && w[4] < bound ? int_constant_value[w[4]] : kAnyElement;
          depth += d.word_count - 4;
        }
        pointer = w[3];
      } else if (d.opcode == spv::OpCopyObject && d.word_count >= 4) {
        pointer = w[3];
      } else {
        break;
      }
    }
    return pointer;
  };
  auto resolve = [&](uint32_t pointer, uint32_t& element) -> uint32_t {
    uint32_t depth;
    return resolve_depth(pointer, element, depth);
  };

  // Function variables whose pointers are only loaded from, stored to and indexed (the translator's registers and
  // its other locals): their loads are followed to the stores that reach them through the blocks, so a guest
  // register used for a coordinate first and for a colour later keeps the colour maths relaxed. Any other use of
  // such a pointer (a call argument, a copy, ...) leaves the variable to the flow-insensitive rule below.
  std::vector<uint8_t> tracked(function_var);
  for (size_t i = 0; i < insts.size(); ++i) {
    if (inst_block[i] < 0) continue;
    const Inst& in = insts[i];
    const uint32_t* w = words + in.offset;
    const uint32_t first = 1 + uint32_t(in.has_type) + uint32_t(in.has_result);
    for (uint32_t k = first; k < in.word_count; ++k) {
      if ((in.opcode == spv::OpLoad && k == 3) || (in.opcode == spv::OpStore && k == 1) ||
          (IsAccessChain(in.opcode) && k == 3) || (in.opcode == spv::OpCopyObject && k == 3)) {
        continue;
      }
      const uint32_t id = w[k];
      if (id >= bound || def[id] < 0) continue;
      const Inst& d = insts[def[id]];
      if (!d.has_type || words[d.offset + 1] >= bound || !pointer_type[words[d.offset + 1]]) continue;
      uint32_t element;
      const uint32_t base = resolve(id, element);
      if (base < bound) tracked[base] = 0;
    }
  }
  // The stores to tracked variables (slots) and, per block, which of them reach its start.
  struct Slot {
    size_t inst;
    uint32_t base, element, depth;
  };
  std::vector<Slot> slots;
  std::vector<int32_t> inst_slot(insts.size(), -1);
  for (size_t si : stores) {
    uint32_t element, depth;
    const uint32_t base = resolve_depth(words[insts[si].offset + 1], element, depth);
    if (base < bound && tracked[base] && inst_block[si] >= 0) {
      inst_slot[si] = int32_t(slots.size());
      slots.push_back({si, base, element, depth});
    }
  }
  const size_t slot_words = (slots.size() + 63) / 64;
  std::map<uint32_t, std::vector<uint32_t>> base_slots;
  for (uint32_t s = 0; s < slots.size(); ++s) base_slots[slots[s].base].push_back(s);
  using Bits = std::vector<uint64_t>;
  auto transfer = [&](Bits& set, size_t i) {
    const int32_t s = inst_slot[i];
    if (s < 0) return;
    const Slot& slot = slots[s];
    if (slot.depth == 0 || (slot.depth == 1 && slot.element != kAnyElement)) {
      for (uint32_t o : base_slots[slot.base]) {
        if (slot.depth == 0 || slots[o].element == slot.element) set[o >> 6] &= ~(uint64_t(1) << (o & 63));
      }
    }
    set[s >> 6] |= uint64_t(1) << (s & 63);
  };
  std::vector<Bits> block_in(blocks.size(), Bits(slot_words, 0)), block_out(blocks.size(), Bits(slot_words, 0));
  if (slot_words) {
    for (size_t b = 0; b < blocks.size(); ++b) {
      for (uint32_t label : blocks[b].successors) {
        if (label < bound && label_block[label] >= 0) blocks[label_block[label]].predecessors.push_back(int32_t(b));
      }
    }
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t b = 0; b < blocks.size(); ++b) {
        Bits in_set(slot_words, 0);
        for (int32_t p : blocks[b].predecessors) {
          for (size_t k = 0; k < slot_words; ++k) in_set[k] |= block_out[p][k];
        }
        Bits out_set = in_set;
        for (size_t i = blocks[b].begin; i <= blocks[b].end; ++i) transfer(out_set, i);
        if (out_set != block_out[b] || in_set != block_in[b]) {
          block_in[b] = std::move(in_set);
          block_out[b] = std::move(out_set);
          changed = true;
        }
      }
    }
  }
  // The stores to a tracked variable's element that may reach the load at instruction i.
  auto reaching_stores = [&](size_t i, uint32_t base, uint32_t element, auto&& fn) {
    const int32_t b = inst_block[i];
    if (b < 0 || !slot_words) return;
    Bits set = block_in[b];
    for (size_t k = blocks[b].begin; k < i; ++k) transfer(set, k);
    for (uint32_t s : base_slots[base]) {
      if (!(set[s >> 6] >> (s & 63) & 1)) continue;
      if (element == kAnyElement || slots[s].element == kAnyElement || slots[s].element == element) {
        fn(slots[s].inst);
      }
    }
  };

  // Stores to FragDepth.
  for (size_t si : stores) {
    const uint32_t* w = words + insts[si].offset;
    uint32_t element;
    const uint32_t base = resolve(w[1], element);
    if (base < bound && depth_var[base]) seeds.push_back(w[2]);
  }

  // The backward walk over values, through Function / Private memory (loads to the stores of the same element).
  std::vector<uint8_t> keep(bound, 0);
  std::vector<uint32_t> work;
  auto push = [&](uint32_t id) {
    if (id < bound && !keep[id]) {
      keep[id] = 1;
      work.push_back(id);
    }
  };
  for (uint32_t id : seeds) push(id);
  struct Taint {
    bool any = false;
    std::set<uint32_t> elements;
  };
  std::map<uint32_t, Taint> tainted;
  bool all_memory = false;
  // A pointer parameter stands for the pointer arguments at its position in every call (glslang passes function
  // arguments that way); an access chain inside the callee then counts as the whole argument.
  std::map<uint32_t, std::pair<uint32_t, size_t>> param_position;  // parameter -> (function, position)
  for (const auto& [function, params] : function_params) {
    for (size_t p = 0; p < params.size(); ++p) param_position[params[p]] = {function, p};
  }
  auto for_each_argument = [&](uint32_t param, auto&& fn) {
    const auto pp = param_position.find(param);
    if (pp == param_position.end()) return;
    for (size_t ci : calls) {
      const uint32_t* c = words + insts[ci].offset;
      if (c[3] == pp->second.first && 4 + pp->second.second < insts[ci].word_count) fn(c[4 + pp->second.second]);
    }
  };
  auto is_param = [&](uint32_t base) {
    return base < bound && def[base] >= 0 && insts[def[base]].opcode == spv::OpFunctionParameter;
  };
  std::function<bool(uint32_t, int)> taint = [&](uint32_t pointer, int depth) -> bool {
    uint32_t element;
    const uint32_t base = resolve(pointer, element);
    if (is_param(base) && depth < 16) {
      bool changed = false;
      for_each_argument(base, [&](uint32_t argument) { changed |= taint(argument, depth + 1); });
      return changed;
    }
    if (base >= bound || def[base] < 0 || insts[def[base]].opcode != spv::OpVariable) {
      const bool changed = !all_memory;
      all_memory = true;  // not traceable: every store counts
      return changed;
    }
    Taint& t = tainted[base];
    if (element == kAnyElement) {
      const bool changed = !t.any;
      t.any = true;
      return changed;
    }
    return t.elements.insert(element).second;
  };
  std::function<bool(uint32_t, int)> is_tainted = [&](uint32_t pointer, int depth) -> bool {
    if (all_memory) return true;
    uint32_t element;
    const uint32_t base = resolve(pointer, element);
    if (is_param(base)) {
      if (depth >= 16) return true;
      bool any = false;
      for_each_argument(base, [&](uint32_t argument) { any = any || is_tainted(argument, depth + 1); });
      return any;
    }
    auto it = tainted.find(base);
    if (it == tainted.end()) return false;
    if (it->second.any || element == kAnyElement) return true;  // (an entry exists once something is tainted)
    return it->second.elements.count(element) != 0;
  };
  std::vector<uint8_t> store_done(insts.size(), 0);
  for (bool memory_changed = true; memory_changed;) {
    memory_changed = false;
    while (!work.empty()) {
      const uint32_t id = work.back();
      work.pop_back();
      if (def[id] < 0) continue;
      const Inst& d = insts[def[id]];
      const uint32_t* w = words + d.offset;
      const uint32_t first = 1 + uint32_t(d.has_type) + uint32_t(d.has_result);
      switch (d.opcode) {
        case spv::OpLoad:
          if (d.word_count >= 4) {
            uint32_t element;
            const uint32_t base = resolve(w[3], element);
            if (base < bound && tracked[base]) {
              reaching_stores(size_t(def[id]), base, element,
                              [&](size_t si) { push(words[insts[si].offset + 2]); });
            } else {
              memory_changed |= taint(w[3], 0);
            }
            push(w[3]);  // (the pointer's access chain: its indices are sinks anyway)
          }
          break;
        case spv::OpSelect:
          // The two values; the condition is a comparison (fine at fp16: only a pick between full-precision values
          // can change near its threshold).
          if (d.word_count >= 6) {
            push(w[4]);
            push(w[5]);
          }
          break;
        case spv::OpFunctionParameter:
          // The argument at the same position in every call of this function.
          for (const auto& [function, params] : function_params) {
            auto p = std::find(params.begin(), params.end(), id);
            if (p == params.end()) continue;
            const size_t position = size_t(p - params.begin());
            for (size_t ci : calls) {
              const uint32_t* c = words + insts[ci].offset;
              if (c[3] == function && 4 + position < insts[ci].word_count) push(c[4 + position]);
            }
          }
          if (w[1] < bound && pointer_type[w[1]]) memory_changed |= taint(id, 0);  // (whatever it points to)
          break;
        case spv::OpFunctionCall:
          for (uint32_t k = 4; k < d.word_count; ++k) push(w[k]);
          if (auto r = function_returns.find(w[3]); r != function_returns.end()) {
            for (uint32_t v : r->second) push(v);
          }
          break;
        case spv::OpVariable:
        case spv::OpLabel:
        case spv::OpFunction:
          break;
        default:
          // Every operand word (literals too: an id that happens to equal a literal is only kept full precision).
          for (uint32_t k = first; k < d.word_count; ++k) push(w[k]);
          break;
      }
    }
    // Stores (and copies) into tainted memory: their values flow into a sink too.
    for (size_t si : stores) {
      if (store_done[si]) continue;
      const uint32_t* w = words + insts[si].offset;
      if (is_tainted(w[1], 0)) {
        store_done[si] = 1;
        push(w[2]);
      }
    }
    for (size_t ci : copies) {
      if (store_done[ci]) continue;
      const uint32_t* w = words + insts[ci].offset;
      if (is_tainted(w[1], 0)) {
        store_done[ci] = 1;
        memory_changed |= taint(w[2], 0);
      }
    }
    memory_changed |= !work.empty();
  }

  // Decorate.
  std::vector<uint32_t> targets;
  for (const Inst& in : insts) {
    if (!in.has_type || !in.has_result) continue;
    const uint32_t* w = words + in.offset;
    bool candidate = IsDecoratedArithmetic(in.opcode);
    if (in.opcode == spv::OpExtInst && in.word_count >= 5 && glsl_set && w[3] == glsl_set) {
      candidate = IsDecoratedGlsl(w[4]);
    }
    if (!candidate || w[1] >= bound || !f32_type[w[1]]) continue;
    ++stats.candidates;
    const uint32_t id = w[2];
    if (relaxed[id]) {
      ++stats.already_relaxed;
    } else if (keep[id]) {
      ++stats.kept_full;
    } else {
      targets.push_back(id);
    }
  }
  stats.decorated = uint32_t(targets.size());
  if (stats_out) *stats_out = stats;
  if (targets.empty()) return false;

  // After the last annotation, or (none) after the preamble / debug sections.
  size_t insert_word;
  if (last_annotation != SIZE_MAX) {
    insert_word = insts[last_annotation].offset + insts[last_annotation].word_count;
  } else if (first_after_preamble != SIZE_MAX) {
    insert_word = insts[first_after_preamble].offset;
  } else {
    insert_word = word_count;
  }
  out.clear();
  out.reserve(word_count + targets.size() * 3);
  out.insert(out.end(), words, words + insert_word);
  for (uint32_t id : targets) {
    out.push_back((3u << 16) | uint32_t(spv::OpDecorate));
    out.push_back(id);
    out.push_back(uint32_t(spv::DecorationRelaxedPrecision));
  }
  out.insert(out.end(), words + insert_word, words + word_count);
  return true;
}

#ifndef REX_SPIRV_RELAXED_PRECISION_STANDALONE
namespace {
std::vector<uint64_t> ParseHashes(const std::string& list) {
  std::vector<uint64_t> out;
  size_t pos = 0;
  while (pos < list.size()) {
    size_t end = list.find(',', pos);
    if (end == std::string::npos) end = list.size();
    const std::string item = list.substr(pos, end - pos);
    if (!item.empty()) out.push_back(std::strtoull(item.c_str(), nullptr, 16));
    pos = end + 1;
  }
  return out;
}

struct CharacterShaders {
  std::shared_mutex mutex;
  std::unordered_set<uint64_t> pixel;    // full precision
  std::vector<uint64_t> vertex;          // gpu_dlss_objmv_shaders
  std::atomic<bool> ready{false};
  void Ensure() {
    if (ready.load(std::memory_order_acquire)) return;
    std::unique_lock lock(mutex);
    if (ready.load(std::memory_order_relaxed)) return;
    vertex = ParseHashes(rex::cvar::GetFlagByName("gpu_dlss_objmv_shaders"));
    for (uint64_t h : ParseHashes(REXCVAR_GET(gpu_native_ps_full_precision_shaders))) pixel.insert(h);
    ready.store(true, std::memory_order_release);
  }
};
CharacterShaders& Characters() {
  static CharacterShaders characters;
  return characters;
}
}  // namespace

void NoteShaderPair(uint64_t vertex_shader_hash, uint64_t pixel_shader_hash) {
  if (!pixel_shader_hash || !REXCVAR_GET(gpu_native_ps_relaxed_precision) ||
      !REXCVAR_GET(gpu_native_ps_full_precision_characters)) {
    return;
  }
  CharacterShaders& c = Characters();
  c.Ensure();
  if (std::find(c.vertex.begin(), c.vertex.end(), vertex_shader_hash) == c.vertex.end()) return;
  {
    std::shared_lock lock(c.mutex);
    if (c.pixel.count(pixel_shader_hash)) return;
  }
  std::unique_lock lock(c.mutex);
  if (c.pixel.insert(pixel_shader_hash).second) {
    REXGPU_INFO("[fp16] pixel shader {:016X} drawn with character vertex shader {:016X}: full precision",
                pixel_shader_hash, vertex_shader_hash);
  }
}

bool IsFullPrecision(uint64_t pixel_shader_hash) {
  if (!REXCVAR_GET(gpu_native_ps_full_precision_characters)) return false;
  CharacterShaders& c = Characters();
  c.Ensure();
  std::shared_lock lock(c.mutex);
  return c.pixel.count(pixel_shader_hash) != 0;
}

bool ApplyIfEnabled(const uint32_t* words, size_t word_count, uint64_t shader_hash, uint64_t modification,
                    std::vector<uint32_t>& out) {
  if (!REXCVAR_GET(gpu_native_ps_relaxed_precision)) return false;
  {  // (Session 95, r3gpu) the list, read once
    static const std::unordered_set<uint64_t> only = [] {
      std::unordered_set<uint64_t> set;
      const std::string text = REXCVAR_GET(gpu_native_ps_relaxed_only);
      for (size_t at = 0; at < text.size();) {
        size_t end = text.find(',', at);
        if (end == std::string::npos) end = text.size();
        set.insert(std::strtoull(text.substr(at, end - at).c_str(), nullptr, 16));
        at = end + 1;
      }
      if (!set.empty()) REXGPU_INFO("[fp16] gpu_native_ps_relaxed_only: {} pixel shaders", set.size());
      return set;
    }();
    if (!only.empty() && !only.count(shader_hash)) return false;
  }
  if (IsFullPrecision(shader_hash)) {
    static std::atomic<uint32_t> kept{0};
    if (kept.fetch_add(1) < 32) {
      REXGPU_INFO("[fp16] pixel shader {:016X} ({:016X}): a character's, kept at full precision", shader_hash,
                  modification);
    }
    return false;
  }
  Stats stats;
  const bool applied = AddRelaxedPrecision(words, word_count, out, &stats);
  // Each shader once (a module is made once per translation): the first 16 on their own, then a total every 64.
  static std::atomic<uint32_t> shaders{0};
  static std::atomic<uint64_t> decorated{0}, candidates{0};
  const uint32_t n = shaders.fetch_add(1) + 1;
  const uint64_t total_decorated = decorated.fetch_add(stats.decorated) + stats.decorated;
  const uint64_t total_candidates = candidates.fetch_add(stats.candidates) + stats.candidates;
  if (n <= 16) {
    REXGPU_INFO("[fp16] pixel shader {:016X} ({:016X}): {} of {} float results RelaxedPrecision ({} kept full "
                "precision: coordinates, indices, depth, integers)",
                shader_hash, modification, stats.decorated, stats.candidates, stats.kept_full);
  } else if ((n & 63) == 0) {
    REXGPU_INFO("[fp16] {} pixel shaders so far: {} of {} float results RelaxedPrecision", n, total_decorated,
                total_candidates);
  }
  return applied;
}
#endif

}  // namespace rex::graphics::spirv_relaxed_precision
