// wwe2k14 patch (Session 95, the diagnostic run): see diag_spirv.h.
#include "diag_spirv.h"

#include <algorithm>
#include <map>
#include <utility>

namespace rex::graphics::vulkan::diag {
namespace {

// SPIR-V opcodes and enums used here (from the specification; no header, so the test tool builds this file alone).
enum : uint32_t {
  kOpName = 5, kOpMemberName = 6, kOpEntryPoint = 15, kOpTypeVoid = 19, kOpTypeInt = 21, kOpTypeFloat = 22, kOpTypeVector = 23,
  kOpTypeMatrix = 24, kOpTypeArray = 28, kOpTypeStruct = 30, kOpTypePointer = 32, kOpTypeFunction = 33, kOpConstant = 43,
  kOpFunction = 54, kOpFunctionEnd = 56, kOpVariable = 59, kOpLoad = 61, kOpStore = 62, kOpAccessChain = 65, kOpDecorate = 71,
  kOpMemberDecorate = 72, kOpDecorationGroup = 73, kOpGroupDecorate = 74, kOpGroupMemberDecorate = 75, kOpDecorateId = 332,
  kOpDecorateString = 5632, kOpMemberDecorateString = 5633, kOpCompositeConstruct = 80, kOpConvertUToF = 112, kOpBitcast = 124,
  kOpFAdd = 129, kOpFSub = 131, kOpFMul = 133, kOpShiftRightLogical = 194, kOpBitwiseXor = 198, kOpBitwiseAnd = 199, kOpReturn = 253,
  kOpCapability = 17, kOpMemoryModel = 14, kOpExecutionMode = 16, kOpExtInstImport = 11, kOpLabel = 248,
};
enum : uint32_t {
  kStorageInput = 1, kStorageOutput = 3, kDecBuiltIn = 11, kDecFlat = 14, kDecLocation = 30, kBuiltInPosition = 0, kBuiltInPointSize = 1,
  kBuiltInVertexIndex = 42, kExecVertex = 0, kExecFragment = 4,
};

inline uint32_t Header(uint32_t op, uint32_t words) { return op | (words << 16); }
void Emit(std::vector<uint32_t>& out, uint32_t op, std::initializer_list<uint32_t> operands) {
  out.push_back(Header(op, uint32_t(operands.size()) + 1));
  out.insert(out.end(), operands.begin(), operands.end());
}

struct Parsed {
  const uint32_t* w;
  size_t n;
  uint32_t bound;
  size_t entry_at = 0;  // the OpEntryPoint's word
  uint32_t entry_id = 0;
  size_t annotations_end = 0, globals_end = 0;  // insert decorations at the first, types / variables at the second
  size_t struct_at = 0;
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> pointers;  // id -> storage class, pointee
  std::map<uint32_t, uint32_t> constants;                       // id -> first literal
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> arrays;     // id -> element, length id
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> matrices;   // id -> column, column count
  std::map<uint32_t, uint32_t> variables;                       // id -> pointer type
  std::map<uint32_t, uint32_t> locations;                       // variable id -> location
  std::map<uint32_t, uint32_t> struct_sizes;                    // struct id -> member count
  std::map<uint32_t, size_t> struct_offsets;                    // struct id -> its word
  uint32_t t_float = 0, t_uint = 0, t_int = 0, t_vec4 = 0, t_uvec4 = 0;
  uint32_t position_var = 0, position_struct = 0, position_member = 0;
  uint32_t point_var = 0, point_struct = 0, point_member = 0;
  uint32_t index_var = 0;
  std::vector<size_t> returns;  // the entry point's OpReturn words
};

bool Parse(Parsed& p, std::string* why) {
  const auto fail = [&](const char* text) {
    if (why) *why = text;
    return false;
  };
  if (p.n < 5 || p.w[0] != 0x07230203u) return fail("not a SPIR-V module");
  p.bound = p.w[3];
  uint32_t entries = 0;
  std::map<uint32_t, bool> position_decorated, point_decorated, index_decorated;
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> position_members, point_members;
  bool in_entry = false;
  size_t at = 5;
  while (at < p.n) {
    const uint32_t op = p.w[at] & 0xFFFFu, len = p.w[at] >> 16;
    if (!len || at + len > p.n) return fail("a malformed instruction");
    const uint32_t* a = p.w + at + 1;
    switch (op) {
      case kOpEntryPoint:
        ++entries;
        if (a[0] == kExecVertex) {
          p.entry_at = at;
          p.entry_id = a[1];
        }
        break;
      case kOpDecorate:
        p.annotations_end = at + len;
        if (a[1] == kDecBuiltIn) {
          if (a[2] == kBuiltInPosition) position_decorated[a[0]] = true;
          if (a[2] == kBuiltInPointSize) point_decorated[a[0]] = true;
          if (a[2] == kBuiltInVertexIndex) index_decorated[a[0]] = true;
        } else if (a[1] == kDecLocation) {
          p.locations[a[0]] = a[2];
        }
        break;
      case kOpMemberDecorate:
        p.annotations_end = at + len;
        if (a[2] == kDecBuiltIn) {
          if (a[3] == kBuiltInPosition) position_members[a[0]] = {a[0], a[1]};
          if (a[3] == kBuiltInPointSize) point_members[a[0]] = {a[0], a[1]};
        } else if (a[2] == kDecLocation) {
          // a location inside a block: not one of the translator's or our shaders' forms
          return fail("output locations inside a block");
        }
        break;
      case kOpDecorateId: case kOpDecorateString: case kOpMemberDecorateString: case kOpDecorationGroup: case kOpGroupDecorate:
      case kOpGroupMemberDecorate:
        p.annotations_end = at + len;
        break;
      case kOpTypeFloat:
        if (a[1] == 32) p.t_float = a[0];
        break;
      case kOpTypeInt:
        if (a[1] == 32) (a[2] ? p.t_int : p.t_uint) = a[0];
        break;
      case kOpTypeVector:
        if (a[2] == 4 && a[1] == p.t_float && p.t_float) p.t_vec4 = a[0];
        if (a[2] == 4 && a[1] == p.t_uint && p.t_uint) p.t_uvec4 = a[0];
        break;
      case kOpTypeMatrix:
        p.matrices[a[0]] = {a[1], a[2]};
        break;
      case kOpTypeArray:
        p.arrays[a[0]] = {a[1], a[2]};
        break;
      case kOpTypeStruct:
        p.struct_sizes[a[0]] = len - 2;
        p.struct_offsets[a[0]] = at;
        break;
      case kOpTypePointer:
        p.pointers[a[0]] = {a[1], a[2]};
        break;
      case kOpConstant:
        p.constants[a[1]] = a[2];
        break;
      case kOpVariable:
        if (!in_entry && !p.globals_end) p.variables[a[1]] = a[0];
        break;
      case kOpFunction:
        if (!p.globals_end) p.globals_end = at;
        in_entry = a[1] == p.entry_id;
        break;
      case kOpFunctionEnd:
        in_entry = false;
        break;
      case kOpReturn:
        if (in_entry) p.returns.push_back(at);
        break;
      default:
        break;
    }
    at += len;
  }
  if (entries != 1 || !p.entry_id) return fail("not a single vertex entry point");
  if (p.returns.empty()) return fail("the entry point has no OpReturn");
  if (!p.globals_end) return fail("no function");
  // Which output carries gl_Position / gl_PointSize, which input gl_VertexIndex.
  for (const auto& [id, type] : p.variables) {
    const auto ptr = p.pointers.find(type);
    if (ptr == p.pointers.end()) continue;
    const uint32_t storage = ptr->second.first, pointee = ptr->second.second;
    if (storage == kStorageOutput) {
      if (position_decorated.count(id)) p.position_var = id;
      if (point_decorated.count(id)) p.point_var = id;
      if (position_members.count(pointee)) {
        p.position_var = id, p.position_struct = pointee, p.position_member = position_members[pointee].second;
      }
      if (point_members.count(pointee)) {
        p.point_var = id, p.point_struct = pointee, p.point_member = point_members[pointee].second;
      }
    } else if (storage == kStorageInput && index_decorated.count(id)) {
      p.index_var = id;
    }
  }
  if (!p.position_var) return fail("no gl_Position output");
  if (p.position_struct) p.struct_at = p.struct_offsets[p.position_struct];
  if (!p.annotations_end) {  // no decoration at all: right before the first type
    return fail("no decorations");
  }
  return true;
}

// How many locations an output of this type takes.
uint32_t LocationsOf(const Parsed& p, uint32_t type) {
  const auto arr = p.arrays.find(type);
  if (arr != p.arrays.end()) {
    const auto len = p.constants.find(arr->second.second);
    return (len != p.constants.end() ? len->second : 1u) * LocationsOf(p, arr->second.first);
  }
  const auto mat = p.matrices.find(type);
  if (mat != p.matrices.end()) return mat->second.second;
  return 1;
}

}  // namespace

bool PatchPositionCapture(const uint32_t* words, size_t count, uint32_t max_locations, PositionPatch& out, std::string* why) {
  Parsed p{words, count};
  if (!Parse(p, why)) return false;
  uint32_t first_free = 0;
  for (const auto& [var, location] : p.locations) {
    const auto v = p.variables.find(var);
    if (v == p.variables.end()) continue;
    const auto ptr = p.pointers.find(v->second);
    if (ptr == p.pointers.end() || ptr->second.first != kStorageOutput) continue;
    first_free = std::max(first_free, location + LocationsOf(p, ptr->second.second));
  }
  if (first_free + 2 > max_locations) {
    if (why) *why = "no free output location";
    return false;
  }
  uint32_t next = p.bound;
  const auto id = [&] { return next++; };

  // New global declarations.
  std::vector<uint32_t> types, decorations;
  uint32_t t_float = p.t_float, t_uint = p.t_uint, t_vec4 = p.t_vec4, t_uvec4 = p.t_uvec4;
  if (!t_float) Emit(types, kOpTypeFloat, {t_float = id(), 32});
  if (!t_uint) Emit(types, kOpTypeInt, {t_uint = id(), 32, 0});
  if (!t_vec4) Emit(types, kOpTypeVector, {t_vec4 = id(), t_float, 4});
  if (!t_uvec4) Emit(types, kOpTypeVector, {t_uvec4 = id(), t_uint, 4});
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> pointer_of;
  for (const auto& [ptr_id, sc] : p.pointers) pointer_of.emplace(sc, ptr_id);
  const auto pointer = [&](uint32_t storage, uint32_t pointee) {
    const auto it = pointer_of.find({storage, pointee});
    if (it != pointer_of.end()) return it->second;
    const uint32_t ptr = id();
    Emit(types, kOpTypePointer, {ptr, storage, pointee});
    pointer_of[{storage, pointee}] = ptr;
    return ptr;
  };
  const auto uint_constant = [&](uint32_t value) {
    const uint32_t c = id();
    Emit(types, kOpConstant, {t_uint, c, value});
    return c;
  };
  const auto float_constant = [&](float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t c = id();
    Emit(types, kOpConstant, {t_float, c, bits});
    return c;
  };
  const uint32_t ptr_out_uvec4 = pointer(kStorageOutput, t_uvec4), ptr_out_uint = pointer(kStorageOutput, t_uint);
  const uint32_t ptr_out_vec4 = pointer(kStorageOutput, t_vec4), ptr_out_float = pointer(kStorageOutput, t_float);
  const uint32_t c16 = uint_constant(16), c8 = uint_constant(8), c255 = uint_constant(255), cffff = uint_constant(0xFFFFu);
  const uint32_t f_half = float_constant(0.5f), f_scale = float_constant(1.0f / 128.0f), f_one = float_constant(1.0f), f_zero = float_constant(0.0f);
  const uint32_t position_index = p.position_struct ? uint_constant(p.position_member) : 0;

  std::vector<uint32_t> extra_interface;
  const uint32_t out_bits = id(), out_index = id();
  Emit(types, kOpVariable, {ptr_out_uvec4, out_bits, kStorageOutput});
  Emit(types, kOpVariable, {ptr_out_uint, out_index, kStorageOutput});
  Emit(decorations, kOpDecorate, {out_bits, kDecLocation, first_free});
  Emit(decorations, kOpDecorate, {out_index, kDecLocation, first_free + 1});
  Emit(decorations, kOpDecorate, {out_bits, kDecFlat});
  Emit(decorations, kOpDecorate, {out_index, kDecFlat});
  extra_interface.push_back(out_bits);
  extra_interface.push_back(out_index);

  // gl_VertexIndex: the module's own, or a new input. Its type: int or uint.
  uint32_t index_var = p.index_var, index_type = 0;
  if (index_var) {
    index_type = p.pointers[p.variables[index_var]].second;
  } else {
    index_var = id();
    index_type = t_uint;
    Emit(types, kOpVariable, {pointer(kStorageInput, t_uint), index_var, kStorageInput});
    Emit(decorations, kOpDecorate, {index_var, kDecBuiltIn, kBuiltInVertexIndex});
    extra_interface.push_back(index_var);
  }

  // gl_PointSize: the module's own; a new last member of the gl_PerVertex block; or a new output beside a loose gl_Position.
  uint32_t point_index = 0, point_var = p.point_var;
  bool extend_struct = false;
  if (p.point_var) {
    if (p.point_struct) point_index = uint_constant(p.point_member);
  } else if (p.position_struct) {
    extend_struct = true;
    point_var = p.position_var;
    point_index = uint_constant(p.struct_sizes[p.position_struct]);
    Emit(decorations, kOpMemberDecorate, {p.position_struct, p.struct_sizes[p.position_struct], kDecBuiltIn, kBuiltInPointSize});
  } else {
    point_var = id();
    Emit(types, kOpVariable, {ptr_out_float, point_var, kStorageOutput});
    Emit(decorations, kOpDecorate, {point_var, kDecBuiltIn, kBuiltInPointSize});
    extra_interface.push_back(point_var);
  }

  // Rebuild, instruction by instruction.
  std::vector<uint32_t> o;
  o.reserve(count + 512);
  o.insert(o.end(), words, words + 5);
  size_t at = 5;
  size_t next_return = 0;
  while (at < count) {
    const uint32_t len = words[at] >> 16, op = words[at] & 0xFFFFu;
    if (at == p.annotations_end) o.insert(o.end(), decorations.begin(), decorations.end());
    if (at == p.globals_end) o.insert(o.end(), types.begin(), types.end());
    if (at == p.entry_at) {
      o.push_back(Header(kOpEntryPoint, len + uint32_t(extra_interface.size())));
      o.insert(o.end(), words + at + 1, words + at + len);
      o.insert(o.end(), extra_interface.begin(), extra_interface.end());
    } else if (extend_struct && at == p.struct_at) {
      o.push_back(Header(kOpTypeStruct, len + 1));
      o.insert(o.end(), words + at + 1, words + at + len);
      o.push_back(t_float);
    } else if (op == kOpReturn && next_return < p.returns.size() && p.returns[next_return] == at) {
      ++next_return;
      uint32_t position_ptr = p.position_var;
      if (p.position_struct) {
        position_ptr = id();
        Emit(o, kOpAccessChain, {ptr_out_vec4, position_ptr, p.position_var, position_index});
      }
      const uint32_t position = id(), bits = id();
      Emit(o, kOpLoad, {t_vec4, position, position_ptr});
      Emit(o, kOpBitcast, {t_uvec4, bits, position});
      Emit(o, kOpStore, {out_bits, bits});
      uint32_t raw = id();
      Emit(o, kOpLoad, {index_type, raw, index_var});
      if (index_type != t_uint) {
        const uint32_t as_uint = id();
        Emit(o, kOpBitcast, {t_uint, as_uint, raw});
        raw = as_uint;
      }
      Emit(o, kOpStore, {out_index, raw});
      // slot = (raw ^ (raw >> 16)) & 0xFFFF; the pixel (slot & 255, slot >> 8); clip position ((x + 0.5) / 128 - 1, ..., 0, 1)
      const uint32_t high = id(), folded = id(), slot = id(), x = id(), y = id(), fx = id(), fy = id();
      Emit(o, kOpShiftRightLogical, {t_uint, high, raw, c16});
      Emit(o, kOpBitwiseXor, {t_uint, folded, raw, high});
      Emit(o, kOpBitwiseAnd, {t_uint, slot, folded, cffff});
      Emit(o, kOpBitwiseAnd, {t_uint, x, slot, c255});
      Emit(o, kOpShiftRightLogical, {t_uint, y, slot, c8});
      Emit(o, kOpConvertUToF, {t_float, fx, x});
      Emit(o, kOpConvertUToF, {t_float, fy, y});
      const auto to_clip = [&](uint32_t value) {
        const uint32_t a = id(), b = id(), c = id();
        Emit(o, kOpFAdd, {t_float, a, value, f_half});
        Emit(o, kOpFMul, {t_float, b, a, f_scale});
        Emit(o, kOpFSub, {t_float, c, b, f_one});
        return c;
      };
      const uint32_t cx = to_clip(fx), cy = to_clip(fy), clip = id();
      Emit(o, kOpCompositeConstruct, {t_vec4, clip, cx, cy, f_zero, f_one});
      Emit(o, kOpStore, {position_ptr, clip});
      uint32_t point_ptr = point_var;
      if (point_index) {
        point_ptr = id();
        Emit(o, kOpAccessChain, {ptr_out_float, point_ptr, point_var, point_index});
      }
      Emit(o, kOpStore, {point_ptr, f_one});
      o.push_back(words[at]);  // OpReturn
    } else {
      o.insert(o.end(), words + at, words + at + len);
    }
    at += len;
  }
  o[3] = next;
  out.words = std::move(o);
  out.location = first_free;
  return true;
}

namespace {

// A small fragment module, written out by hand. ids: 1 main, 2 void, 3 fn, 4 label, then the shader's own.
void FragmentHeader(std::vector<uint32_t>& o, uint32_t bound) {
  o = {0x07230203u, 0x00010000u, 0x57323134u /* "W214": our generator tag */, bound, 0};
  Emit(o, kOpCapability, {1});        // Shader
  Emit(o, kOpMemoryModel, {0, 1});    // Logical GLSL450
}
void Words(std::vector<uint32_t>& o, std::initializer_list<uint32_t> w) { o.insert(o.end(), w.begin(), w.end()); }

}  // namespace

std::vector<uint32_t> CaptureFragmentModule(uint32_t location) {
  // in: 5 pos_in (location N), 6 id_in (N + 1); out: 7 pos_out (0), 8 id_out (1)
  enum : uint32_t {
    kMain = 1, kVoid, kFn, kLabel, kPosIn, kIdIn, kPosOut, kIdOut, kUint, kUvec4, kPtrInV, kPtrInU, kPtrOutV, kPtrOutU, kV, kU, kBound
  };
  std::vector<uint32_t> o;
  FragmentHeader(o, kBound);
  // OpEntryPoint Fragment %main "main" + interface
  o.push_back(Header(kOpEntryPoint, 3 + 2 + 4));
  Words(o, {kExecFragment, kMain, 0x6E69616Du /* "main" */, 0u, kPosIn, kIdIn, kPosOut, kIdOut});
  Emit(o, kOpExecutionMode, {kMain, 7});  // OriginUpperLeft
  Emit(o, kOpDecorate, {kPosIn, kDecLocation, location});
  Emit(o, kOpDecorate, {kIdIn, kDecLocation, location + 1});
  Emit(o, kOpDecorate, {kPosIn, kDecFlat});
  Emit(o, kOpDecorate, {kIdIn, kDecFlat});
  Emit(o, kOpDecorate, {kPosOut, kDecLocation, 0});
  Emit(o, kOpDecorate, {kIdOut, kDecLocation, 1});
  Emit(o, kOpTypeVoid, {kVoid});
  Emit(o, kOpTypeFunction, {kFn, kVoid});
  Emit(o, kOpTypeInt, {kUint, 32, 0});
  Emit(o, kOpTypeVector, {kUvec4, kUint, 4});
  Emit(o, kOpTypePointer, {kPtrInV, kStorageInput, kUvec4});
  Emit(o, kOpTypePointer, {kPtrInU, kStorageInput, kUint});
  Emit(o, kOpTypePointer, {kPtrOutV, kStorageOutput, kUvec4});
  Emit(o, kOpTypePointer, {kPtrOutU, kStorageOutput, kUint});
  Emit(o, kOpVariable, {kPtrInV, kPosIn, kStorageInput});
  Emit(o, kOpVariable, {kPtrInU, kIdIn, kStorageInput});
  Emit(o, kOpVariable, {kPtrOutV, kPosOut, kStorageOutput});
  Emit(o, kOpVariable, {kPtrOutU, kIdOut, kStorageOutput});
  Emit(o, kOpFunction, {kVoid, kMain, 0, kFn});
  Emit(o, kOpLabel, {kLabel});
  Emit(o, kOpLoad, {kUvec4, kV, kPosIn});
  Emit(o, kOpStore, {kPosOut, kV});
  Emit(o, kOpLoad, {kUint, kU, kIdIn});
  Emit(o, kOpStore, {kIdOut, kU});
  Emit(o, kOpReturn, {});
  Emit(o, kOpFunctionEnd, {});
  return o;
}

std::vector<uint32_t> SoloFragmentModule() {
  enum : uint32_t { kMain = 1, kVoid, kFn, kLabel, kFloat, kVec4, kPtrOut, kOut, kOne, kWhite, kBound };
  std::vector<uint32_t> o;
  FragmentHeader(o, kBound);
  o.push_back(Header(kOpEntryPoint, 3 + 2 + 1));
  Words(o, {kExecFragment, kMain, 0x6E69616Du, 0u, kOut});
  Emit(o, kOpExecutionMode, {kMain, 7});
  Emit(o, kOpDecorate, {kOut, kDecLocation, 0});
  Emit(o, kOpTypeVoid, {kVoid});
  Emit(o, kOpTypeFunction, {kFn, kVoid});
  Emit(o, kOpTypeFloat, {kFloat, 32});
  Emit(o, kOpTypeVector, {kVec4, kFloat, 4});
  Emit(o, kOpTypePointer, {kPtrOut, kStorageOutput, kVec4});
  Emit(o, kOpVariable, {kPtrOut, kOut, kStorageOutput});
  Emit(o, kOpConstant, {kFloat, kOne, 0x3F800000u});
  Emit(o, 44 /* OpConstantComposite */, {kVec4, kWhite, kOne, kOne, kOne, kOne});
  Emit(o, kOpFunction, {kVoid, kMain, 0, kFn});
  Emit(o, kOpLabel, {kLabel});
  Emit(o, kOpStore, {kOut, kWhite});
  Emit(o, kOpReturn, {});
  Emit(o, kOpFunctionEnd, {});
  return o;
}

}  // namespace rex::graphics::vulkan::diag
