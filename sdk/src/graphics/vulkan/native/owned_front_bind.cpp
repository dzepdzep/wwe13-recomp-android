// wwe2k14 patch (Session 92, M4c; contract S92_M4C_CONTRACT.md): the shader constants and the binding contents from the record decoder's typed
// state, as a SHADOW (owned_front_bind.h). Pure logic over plain data; the processor's private members are gathered by OwnedFrontAccess
// (owned_front_draw.cpp).
//
// What is built, from which typed fields (OwnedRecordState):
//   system constants   BuildSystemConstants = UpdateSystemConstantValuesFull restated over the typed fields (surface info, colour control,
//                      depth info, VTE / clip control, point size, VGT words, user clip planes, tessellation levels, blend colour...) plus the
//                      draw derivations M4b made (primitive result, viewport NDC, normalized masks); the texture part from the used fetch
//                      constants (signs: texture_util::SwizzleSigns of the typed fetch words).
//   float banks        GatherFloat: s.vs_float / s.ps_float through the shader's constant map (packed in register order, as the ring holds them).
//   bool / loop, fetch s.bool_loop, s.fetch as they are.
//   texture keys       TypedBinding: the fetch constant's texture key (and, with --gpu_d3d_textures=2, the bound game object's), compared with the
//                      texture cache's active binding (GetActiveTextureInfo).
//   vertex ranges      the vertex fetch words' (address, size), compared with the processor's vertex buffer states.
// Compared with the old path's bytes / states; mismatches counted by block and field, the first of each logged (cap 96).
//
//   views, samplers    OwnedFrontAccess (owned_front_draw.cpp) derives the texture key and host swizzle of each used fetch constant, finds the texture and its
//                      view find-only (FindTextureOnly / FindView) and the sampler by its typed parameters; compared with the raw views and the
//                      samplers the old path bound.
// NOT built here: the descriptor sets / dynamic offsets (they exist only by uploading into the ring and writing descriptors, which a shadow must not
// do); M4b copies them, M4f builds them when it owns the draw.
#include "owned_front_bind.h"

#include <memory>
#include <rex/cvar.h>
#include <rex/graphics/d3d_tex.h>
#include <rex/graphics/pipeline/render_target/cache.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/util/draw.h>
#include <rex/logging.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

REXCVAR_DECLARE(bool, gpu_depth_of_field);
REXCVAR_DECLARE(bool, gpu_allow_invalid_fetch_constants);

REXCVAR_DECLARE(uint32_t, gpu_d3d_wall);  // (Session 95: d3d_noring.cpp)

namespace rex::graphics {
bool IsSelfConsistentInvalidTextureFetch(const xenos::xe_gpu_texture_fetch_t& fetch);
}

namespace rex::graphics::vulkan {

namespace {

using SC = SpirvShaderTranslator::SystemConstants;
using ST = SpirvShaderTranslator;

// ---------------------------------------------------------------------------------------------------------------------------------
// the window
struct Window {
  uint64_t built = 0;     // draws the typed side built constants and bindings for
  uint64_t compared = 0;  // ... and compared with the old path's
  uint64_t mismatch_draws = 0;
  std::map<std::string, uint64_t> by_field;       // "block.field" -> draws
  std::map<std::string, uint64_t> not_built;      // reason -> draws
  uint64_t bytes_sys = 0, bytes_float = 0;        // compared bytes (the bar's "built" evidence)
  uint64_t textures_checked = 0, vfetch_checked = 0, views_checked = 0, samplers_checked = 0, skipped_3d_views = 0;
  uint64_t scaled_from_cache = 0;                 // draws whose texture resolution-scaled bits came from the texture cache (scale > 1)
  uint64_t fsi_masked = 0, crowd_c76_skipped = 0, banks_skipped = 0, native_views = 0;  // see the window line
  uint64_t history_only = 0;                     // draws where only bytes the shaders do not read (gated out) differed from the ring
  uint64_t layout_found = 0, layout_fallback = 0, layout_differs = 0;
  uint64_t build_ns = 0;
};
Window g_w;
Window g_winl;                // M4e: the inline draws' window (swapped with g_w around their claims)
bool g_inline_bucket = false;  // ... and the log keys of their mismatches are their own
std::set<std::string> g_logged_fields;
uint32_t g_logged = 0;

bool g_ds_enabled = false;  // BuildSystemConstants: depth_stencil_enabled (the FSI rule)
SC g_sc{};  // the typed side's persistent block (the old path's system_constants_ is zeroed at setup and written field by field)

inline uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Cmp {
  const OwnedBindInputs& in;
  bool any = false;
  void Note(const std::string& key, const char* what, uint64_t old_v, uint64_t typed_v) {
    any = true;
    ++g_w.by_field[key];
    if (g_logged < 96 && g_logged_fields.insert(g_inline_bucket ? "inline " + key : key).second) {
      ++g_logged;
      REXGPU_INFO("[d3d][front] {} mismatch #{}: {} ({}) old {:X} typed {:X} (frame {}, claim {}, vertex shader {:016X}, pixel shader {:016X})",
                  g_inline_bucket ? "M4e inline (M4c builders)" : "M4c", g_logged, key, what, old_v, typed_v, in.frame, in.kind == 0 ? "full" : in.kind == 1 ? "lean" : "repeat", in.vs_hash, in.ps_hash);
    }
  }
};

// ---------------------------------------------------------------------------------------------------------------------------------
// system constants

struct SysField {
  const char* name;
  uint32_t offset, size;
};
#define SF(m) {#m, uint32_t(offsetof(SC, m)), uint32_t(sizeof(SC::m))}
const SysField kSysFields[] = {
    SF(flags), SF(vertex_index_load_address), SF(vertex_index_endian), SF(line_loop_closing_index), SF(vertex_base_index),
    SF(vertex_index_reset), SF(compute_memexport_vertex_count), SF(user_clip_planes), SF(ndc_scale), SF(point_vertex_diameter_min),
    SF(ndc_offset), SF(point_vertex_diameter_max), SF(point_constant_diameter), SF(point_screen_diameter_to_ndc_radius),
    SF(textures_resolution_scaled), SF(texture_swizzled_signs), SF(texture_swizzles), SF(alpha_test_reference), SF(alpha_to_mask),
    SF(edram_32bpp_tile_pitch_dwords_scaled), SF(edram_depth_base_dwords_scaled), SF(color_exp_bias), SF(edram_poly_offset_front_scale),
    SF(edram_poly_offset_back_scale), SF(edram_poly_offset_front_offset), SF(edram_poly_offset_back_offset), SF(edram_stencil_front),
    SF(edram_stencil_back), SF(edram_rt_base_dwords_scaled), SF(edram_rt_format_flags), SF(edram_rt_blend_factors_ops),
    SF(edram_rt_keep_mask), SF(edram_rt_clamp), SF(edram_blend_constant), SF(vertex_index_min), SF(vertex_index_max),
    SF(tessellation_factor_range_min), SF(tessellation_factor_range_max),
};
#undef SF

// The texture key / signs of one fetch constant as the texture cache derives them (TextureCache::BindingInfoFromFetchConstant), from the
// typed fetch words. Only the pieces the compare and the system constants use.
struct TypedKey {
  bool valid = false;
  uint32_t base_page = 0, mip_page = 0, w_minus_1 = 0, h_minus_1 = 0, d_minus_1 = 0, pitch = 0, mip_max_level = 0;
  uint32_t dimension = 0, format = 0, endianness = 0;
  bool tiled = false, packed = false;
  bool operator==(const TypedKey& o) const {
    return valid == o.valid && (!valid || (base_page == o.base_page && mip_page == o.mip_page && w_minus_1 == o.w_minus_1 &&
                                           h_minus_1 == o.h_minus_1 && d_minus_1 == o.d_minus_1 && pitch == o.pitch &&
                                           mip_max_level == o.mip_max_level && dimension == o.dimension && format == o.format &&
                                           endianness == o.endianness && tiled == o.tiled && packed == o.packed));
  }
};

constexpr uint8_t kUnsignedSigns = uint8_t(xenos::TextureSign::kUnsigned) * uint8_t(0b01010101);

TypedKey KeyFromFetch(const xenos::xe_gpu_texture_fetch_t& fetch) {
  TypedKey k;
  switch (fetch.type) {
    case xenos::FetchConstantType::kTexture:
      break;
    case xenos::FetchConstantType::kInvalidTexture:
      if (REXCVAR_GET(gpu_allow_invalid_fetch_constants) || IsSelfConsistentInvalidTextureFetch(fetch)) break;
      return k;
    default:
      return k;
  }
  uint32_t w, h, d, base_page, mip_page, mip_max_level;
  texture_util::GetSubresourcesFromFetchConstant(fetch, &w, &h, &d, &base_page, &mip_page, nullptr, &mip_max_level);
  if (base_page == 0 && mip_page == 0) return k;
  if (fetch.dimension == xenos::DataDimension::k1D) {
    if (w >= xenos::kTexture2DCubeMaxWidthHeight || fetch.tiled || fetch.packed_mips) return k;
  }
  k.valid = true;
  k.base_page = base_page;
  k.mip_page = mip_page;
  k.w_minus_1 = w;
  k.h_minus_1 = h;
  k.d_minus_1 = d;
  k.pitch = fetch.pitch;
  k.mip_max_level = mip_max_level;
  k.dimension = uint32_t(fetch.dimension);
  k.format = uint32_t(GetBaseFormat(fetch.format));
  k.endianness = uint32_t(fetch.endianness);
  k.tiled = fetch.tiled;
  k.packed = fetch.packed_mips;
  return k;
}

// The key and signs the cache binds for slot `index`: the bound game object's (--gpu_d3d_textures=2) when it has one, else the fetch
// constant's (TextureCache::RequestTextures' derive_key, minus its counters). Find-only: the table is read, never written.
struct TypedBinding {
  TypedKey key;
  uint8_t signs = kUnsignedSigns;
};
TypedBinding BindingForFull(uint32_t index, const uint32_t* words6);
// (Session 95, gpu_d3d_wall bit 16) The answer is a function of the six words and the object behind the slot's id: kept per (id, words) while no object was defined or destroyed (thread-local: the
// scout's workers build too). The key derivation was ~3% of the command thread, for the same few fetch constants frame after frame.
TypedBinding BindingFor(uint32_t index, const uint32_t* words6) {
  const bool ids = d3d_tex::g_active.load(std::memory_order_relaxed) && d3d_tex::Enabled() && d3d_tex::UseIds();
  if (!(REXCVAR_GET(gpu_d3d_wall) & 16u) || !ids || d3d_tex::VerifyKeys()) return BindingForFull(index, words6);
  d3d_tex::Table& table = d3d_tex::Table::Get();
  uint32_t id = table.BoundId(index);
  if (!id) {
    bool never = true;
    id = table.Recall(index, words6, never);
  }
  struct Kept {
    uint64_t generation = 0;
    uint32_t id = 0;
    uint32_t words[6] = {};
    TypedBinding binding;
  };
  thread_local std::unique_ptr<Kept[]> kept;
  if (!kept) kept = std::make_unique<Kept[]>(1024);
  uint32_t h = id * 0x9E3779B1u;
  for (uint32_t k = 0; k < 6; ++k) h = (h ^ words6[k]) * 0x85EBCA77u;
  Kept& e = kept[h >> 22];
  const uint64_t generation = d3d_tex::ObjectsGeneration();
  if (e.generation == generation && e.id == id && !std::memcmp(e.words, words6, sizeof(e.words))) return e.binding;
  const TypedBinding b = BindingForFull(index, words6);
  e.generation = generation;
  e.id = id;
  std::memcpy(e.words, words6, sizeof(e.words));
  e.binding = b;
  return b;
}
TypedBinding BindingForFull(uint32_t index, const uint32_t* words6) {
  xenos::xe_gpu_texture_fetch_t fetch;
  std::memcpy(&fetch, words6, sizeof(fetch));
  TypedBinding b;
  TypedKey fetch_key = KeyFromFetch(fetch);
  const uint8_t fetch_signs = fetch_key.valid ? texture_util::SwizzleSigns(fetch) : kUnsignedSigns;
  b.key = fetch_key;
  b.signs = fetch_signs;
  const bool ids = d3d_tex::g_active.load(std::memory_order_relaxed) && d3d_tex::Enabled() && d3d_tex::UseIds();
  if (!ids) return b;
  d3d_tex::Table& table = d3d_tex::Table::Get();
  uint32_t id = table.BoundId(index);
  if (!id) {
    bool never = true;
    id = table.Recall(index, words6, never);
    if (!id) return b;
  }
  d3d_tex::Entry e;
  if (!table.Find(id, e)) return b;
  if (fetch.type != xenos::FetchConstantType::kTexture) return b;
  xenos::xe_gpu_texture_fetch_t object_fetch;
  std::memcpy(&object_fetch, e.words, sizeof(object_fetch));
  object_fetch.dword_4 = fetch.dword_4;  // the view bits are the slot's
  const TypedKey object_key = KeyFromFetch(object_fetch);
  if (!object_key.valid) return b;
  if (d3d_tex::VerifyKeys() && !(object_key == fetch_key)) return b;  // the fetch constant wins (and the signs are the fetch's anyway)
  b.key = object_key;
  b.signs = texture_util::SwizzleSigns(fetch);
  return b;
}

void BuildSystemConstants(const OwnedBindInputs& in, SC& sc, bool& scaled_from_cache) {
  const OwnedRecordState& s = *in.s;
  const PrimitiveProcessor::ProcessingResult& p = in.primitive;
  const reg::RB_DEPTHCONTROL& ndc = in.depth_control;
  const uint32_t scale_x = in.scale_x, scale_y = in.scale_y;

  reg::RB_COLOR_INFO color_infos[xenos::kMaxColorRenderTargets];
  float rt_clamp[4][4] = {};
  uint32_t rt_keep_masks[4][2] = {};
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    color_infos[i] = s.color_info[i];
    if (in.fsi) {
      RenderTargetCache::GetPSIColorFormatInfo(color_infos[i].color_format, (in.color_mask >> (i * 4)) & 0b1111, rt_clamp[i][0], rt_clamp[i][1],
                                               rt_clamp[i][2], rt_clamp[i][3], rt_keep_masks[i][0], rt_keep_masks[i][1]);
    }
  }
  bool depth_stencil_enabled = ndc.stencil_enable || ndc.z_enable;
  if (in.fsi && depth_stencil_enabled) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (s.depth_info.depth_base == color_infos[i].color_base && (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX)) {
        depth_stencil_enabled = false;
        break;
      }
    }
  }
  g_ds_enabled = depth_stencil_enabled;

  // ---- flags
  uint32_t flags = 0;
  if (in.shader_32bit_index_dma) flags |= ST::kSysFlag_VertexIndexLoad;
  if (p.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA ||
      p.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA ||
      p.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted) {
    flags |= ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad;
    if (s.draw_initiator.index_size == xenos::IndexFormat::kInt32) flags |= ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit;
    const bool reset = p.host_primitive_reset_enabled && p.host_vertex_shader_type != Shader::HostVertexShaderType::kPointListAsTriangleStrip &&
                       p.host_vertex_shader_type != Shader::HostVertexShaderType::kRectangleListAsTriangleStrip;
    if (reset) flags |= ST::kSysFlag_ComputeOrPrimitiveVertexIndexReset;
  }
  if (p.host_vertex_shader_type == Shader::HostVertexShaderType::kTriangleDomainPatchIndexed) flags |= ST::kSysFlag_ComputeMemExportPatchIndexInRegister1;
  if (p.host_vertex_shader_type == Shader::HostVertexShaderType::kTriangleDomainCPIndexed) {
    flags |= ST::kSysFlag_ComputeMemExportTriangleCPIndexed;
  } else if (p.host_vertex_shader_type == Shader::HostVertexShaderType::kQuadDomainCPIndexed) {
    flags |= ST::kSysFlag_ComputeMemExportQuadCPIndexed;
  }
  if (p.tessellation_mode == xenos::TessellationMode::kAdaptive &&
      (p.host_vertex_shader_type == Shader::HostVertexShaderType::kTriangleDomainPatchIndexed ||
       p.host_vertex_shader_type == Shader::HostVertexShaderType::kQuadDomainPatchIndexed)) {
    flags |= ST::kSysFlag_ComputeMemExportPatchIndexFromInvocation;
  }
  if (s.vte_cntl.vtx_xy_fmt) flags |= ST::kSysFlag_XYDividedByW;
  if (s.vte_cntl.vtx_z_fmt) flags |= ST::kSysFlag_ZDividedByW;
  if (s.vte_cntl.vtx_w0_fmt) flags |= ST::kSysFlag_WNotReciprocal;
  if (in.polygonal) flags |= ST::kSysFlag_PrimitivePolygonal;
  if (draw_util::IsPrimitiveLine(s.output_path_cntl.path_select == xenos::VGTOutputPath::kTessellationEnable, s.draw_initiator.prim_type)) {
    flags |= ST::kSysFlag_PrimitiveLine;
  }
  flags |= uint32_t(s.surface_info.msaa_samples) << ST::kSysFlag_MsaaSamples_Shift;
  if (s.depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) flags |= ST::kSysFlag_DepthFloat24;
  const xenos::CompareFunction alpha_test_function = s.color_control.alpha_test_enable ? s.color_control.alpha_func : xenos::CompareFunction::kAlways;
  flags |= uint32_t(alpha_test_function) << ST::kSysFlag_AlphaPassIfLess_Shift;
  if (!in.gamma_as_unorm16) {
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (color_infos[i].color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) flags |= ST::kSysFlag_ConvertColor0ToGamma << i;
    }
  }
  if (in.fsi && depth_stencil_enabled) {
    flags |= ST::kSysFlag_FSIDepthStencil;
    if (ndc.z_enable) {
      flags |= uint32_t(ndc.zfunc) << ST::kSysFlag_FSIDepthPassIfLess_Shift;
      if (ndc.z_write_enable) flags |= ST::kSysFlag_FSIDepthWrite;
    } else {
      flags |= ST::kSysFlag_FSIDepthPassIfLess | ST::kSysFlag_FSIDepthPassIfEqual | ST::kSysFlag_FSIDepthPassIfGreater;
    }
    if (ndc.stencil_enable) flags |= ST::kSysFlag_FSIStencilTest;
    if (alpha_test_function == xenos::CompareFunction::kAlways && !s.color_control.alpha_to_mask_enable) flags |= ST::kSysFlag_FSIDepthStencilEarlyWrite;
  }
  sc.flags = flags;

  if (flags & (ST::kSysFlag_VertexIndexLoad | ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad)) sc.vertex_index_load_address = p.guest_index_base;
  sc.vertex_index_reset = (flags & ST::kSysFlag_ComputeOrPrimitiveVertexIndexReset) ? s.reset_indx.reset_indx : 0;
  sc.compute_memexport_vertex_count = 0;

  xenos::Endian guest_index_endian = s.dma_size.swap_mode;
  if (s.draw_initiator.index_size == xenos::IndexFormat::kInt16 && guest_index_endian != xenos::Endian::kNone &&
      guest_index_endian != xenos::Endian::k8in16) {
    guest_index_endian = guest_index_endian == xenos::Endian::k8in32 ? xenos::Endian::k8in16 : xenos::Endian::kNone;
  }
  sc.vertex_index_endian = (flags & ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad) ? guest_index_endian : p.host_shader_index_endian;
  sc.line_loop_closing_index = p.line_loop_closing_index;
  sc.vertex_base_index = int32_t(s.indx_offset.value);
  sc.vertex_index_min = s.min_vtx_indx.value;
  sc.vertex_index_max = s.max_vtx_indx.value;

  // user clip planes: the planes the control word enables, packed
  if (!s.clip_cntl.clip_disable) {
    float* write = sc.user_clip_planes[0];
    uint32_t remaining = s.clip_cntl.ucp_ena;
    while (remaining) {
      const uint32_t plane = uint32_t(std::countr_zero(remaining));
      remaining &= remaining - 1;
      std::memcpy(write, &s.user_clip[plane * 4], 4 * sizeof(float));
      write += 4;
    }
  }
  sc.tessellation_factor_range_min = s.hos_tess[1] + 1.0f;  // (hos_tess: max, min)
  sc.tessellation_factor_range_max = s.hos_tess[0] + 1.0f;
  for (uint32_t i = 0; i < 3; ++i) {
    sc.ndc_scale[i] = in.ndc_scale[i];
    sc.ndc_offset[i] = in.ndc_offset[i];
  }

  // point size
  if (s.draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    sc.point_vertex_diameter_min = float(s.point_minmax.min_size) * (2.0f / 16.0f);
    sc.point_vertex_diameter_max = float(s.point_minmax.max_size) * (2.0f / 16.0f);
    sc.point_constant_diameter[0] = float(s.point_size.width) * (2.0f / 16.0f);
    sc.point_constant_diameter[1] = float(s.point_size.height) * (2.0f / 16.0f);
    sc.point_screen_diameter_to_ndc_radius[0] = float(scale_x) / std::max(in.xy_extent[0], uint32_t(1));
    sc.point_screen_diameter_to_ndc_radius[1] = float(scale_y) / std::max(in.xy_extent[1], uint32_t(1));
  }

  // textures: the signs from the typed fetch words; the resolution-scaled bit of a texture is a texture cache decision (a scaled resolve
  // exists only at a draw resolution scale above 1): 0 at scale 1, else the cache's own answer (an input, counted)
  uint32_t textures_resolution_scaled = 0;
  scaled_from_cache = false;
  {
    uint32_t remaining = in.used_texture_mask;
    while (remaining) {
      const uint32_t idx = uint32_t(std::countr_zero(remaining));
      remaining &= remaining - 1;
      const TypedBinding b = BindingFor(idx, &s.fetch[6 * idx]);
      uint32_t& signs_uint = sc.texture_swizzled_signs[idx >> 2];
      const uint32_t shift = 8 * (idx & 3);
      signs_uint = (signs_uint & ~(0xFFu << shift)) | (uint32_t(b.signs) << shift);
      if (scale_x > 1 || scale_y > 1) {
        textures_resolution_scaled |= in.scaled_from_cache & (1u << idx);
        scaled_from_cache = true;
      }
      if (!in.image_view_format_swizzle) {
        uint32_t& swz = sc.texture_swizzles[idx >> 1];
        const uint32_t sh = 12 * (idx & 1);
        swz = (swz & ~(0xFFFu << sh)) | (in.host_swizzle_from_cache[idx] << sh);
      }
    }
  }
  sc.textures_resolution_scaled = textures_resolution_scaled;

  sc.alpha_test_reference = s.alpha_ref;
  sc.alpha_to_mask = s.color_control.alpha_to_mask_enable ? (s.color_control.value >> 24) | (UINT32_C(1) << 8) : 0;

  const uint32_t edram_tile_dwords_scaled = xenos::kEdramTileWidthSamples * xenos::kEdramTileHeightSamples * (scale_x * scale_y);
  if (in.fsi) {
    sc.edram_32bpp_tile_pitch_dwords_scaled = ((s.surface_info.surface_pitch * (s.surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 2 : 1)) +
                                               (xenos::kEdramTileWidthSamples - 1)) /
                                              xenos::kEdramTileWidthSamples * edram_tile_dwords_scaled;
  }
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    const reg::RB_COLOR_INFO ci = color_infos[i];
    int32_t exp_bias = ci.color_exp_bias;
    if (in.host_rt && !in.native_truth &&
        ((ci.color_format == xenos::ColorRenderTargetFormat::k_16_16 && !in.rg16_truncated) ||
         (ci.color_format == xenos::ColorRenderTargetFormat::k_16_16_16_16 && !in.rgba16_truncated))) {
      exp_bias -= 5;
    }
    int32_t scale_bits = int32_t(UINT32_C(0x3F800000) + (uint32_t(exp_bias) << 23));
    std::memcpy(&sc.color_exp_bias[i], &scale_bits, sizeof(float));
    if (in.fsi) {
      sc.edram_rt_keep_mask[i][0] = rt_keep_masks[i][0];
      sc.edram_rt_keep_mask[i][1] = rt_keep_masks[i][1];
      if (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX) {
        sc.edram_rt_base_dwords_scaled[i] = ci.color_base * edram_tile_dwords_scaled;
        sc.edram_rt_format_flags[i] = RenderTargetCache::AddPSIColorFormatFlags(ci.color_format);
        sc.edram_rt_blend_factors_ops[i] = s.blend[i].value & 0x1FFF1FFF;
        std::memcpy(sc.edram_rt_clamp[i], rt_clamp[i], 4 * sizeof(float));
      }
    }
  }
  if (in.fsi) {
    sc.edram_depth_base_dwords_scaled = s.depth_info.depth_base * edram_tile_dwords_scaled;
    float fs = 0, fo = 0, bs = 0, bo = 0;
    if (in.polygonal) {
      if (s.sc_mode_cntl.poly_offset_front_enable) fs = s.poly_offset[0], fo = s.poly_offset[1];
      if (s.sc_mode_cntl.poly_offset_back_enable) bs = s.poly_offset[2], bo = s.poly_offset[3];
    } else if (s.sc_mode_cntl.poly_offset_para_enable) {
      fs = s.poly_offset[0], fo = s.poly_offset[1];
      bs = fs, bo = fo;
    }
    const float factor = xenos::kPolygonOffsetScaleSubpixelUnit * std::max(scale_x, scale_y);
    sc.edram_poly_offset_front_scale = fs * factor;
    sc.edram_poly_offset_front_offset = fo;
    sc.edram_poly_offset_back_scale = bs * factor;
    sc.edram_poly_offset_back_offset = bo;
    if (depth_stencil_enabled && ndc.stencil_enable) {
      sc.edram_stencil_front[0] = s.stencil_front.value & 0xFFFFFF;
      sc.edram_stencil_front[1] = (ndc.value >> 8) & ((1 << 12) - 1);
      if (in.polygonal && ndc.backface_enable) {
        sc.edram_stencil_back[0] = s.stencil_back.value & 0xFFFFFF;
        sc.edram_stencil_back[1] = (ndc.value >> 20) & ((1 << 12) - 1);
      } else {
        std::memcpy(sc.edram_stencil_back, sc.edram_stencil_front, 2 * sizeof(uint32_t));
      }
    }
    for (uint32_t i = 0; i < 4; ++i) sc.edram_blend_constant[i] = s.blend_color[i];
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// banks

// The packed float constants of a shader's usage bitmap (the gather UpdateBindings does): constant n of the stage at words[n * 4], in
// ascending order. Returns bytes written; `regs_out` gets the register index of each packed constant.
uint32_t GatherFloat(uint8_t* out, uint16_t* regs_out, const uint32_t* words, const uint64_t* bitmap) {
  uint8_t* const start = out;
  uint32_t n = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint64_t bits = bitmap[i];
    while (bits) {
      const uint32_t index = uint32_t(std::countr_zero(bits));
      bits &= bits - 1;
      std::memcpy(out, words + (i << 8) + (index << 2), 16);
      out += 16;
      regs_out[n++] = uint16_t((i << 6) + index);
    }
  }
  return uint32_t(out - start);
}

// gpu_depth_of_field = false: the old path zeroes three combine constants in the gathered pixel bytes (UpdateBindings' zero_depth_of_field)
void ZeroDepthOfField(uint8_t* base, const Shader* ps) {
  if (!ps || REXCVAR_GET(gpu_depth_of_field)) return;
  const uint64_t hash = ps->ucode_data_hash();
  if (hash != 0x1233F68C15FDF948ull && hash != 0x4E2CD8031D93DE3Eull) return;
  const Shader::ConstantRegisterMap& map = ps->constant_register_map();
  const auto rank = [&](uint32_t reg) -> int {
    if (!(map.float_bitmap[reg >> 6] & (1ull << (reg & 63)))) return -1;
    uint32_t n = 0;
    for (uint32_t i = 0; i < (reg >> 6); ++i) n += uint32_t(std::popcount(map.float_bitmap[i]));
    n += uint32_t(std::popcount(map.float_bitmap[reg >> 6] & ((1ull << (reg & 63)) - 1)));
    return int(n);
  };
  const int r20 = rank(20), r21 = rank(21);
  if (r20 < 0 || r21 < 0) return;
  const float zero = 0.0f;
  std::memcpy(base + size_t(r20) * 16 + 0, &zero, 4);
  std::memcpy(base + size_t(r20) * 16 + 12, &zero, 4);
  std::memcpy(base + size_t(r21) * 16 + 8, &zero, 4);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// M4f-A: the typed side's system constants written into the caller's block (the processor's own system_constants_), field by field as the old
// UpdateSystemConstantValuesFull does. The same BuildSystemConstants the shadow compares (0 mismatches).
void OwnedFrontBindBuildSystem(const OwnedBindInputs& in, SpirvShaderTranslator::SystemConstants& out) {
  bool scaled = false;
  BuildSystemConstants(in, out, scaled);
}

void OwnedFrontBindNote(const char* reason) { ++g_w.not_built[reason]; }

void OwnedFrontBindNoteLayout(bool found, bool matches_old) {
  if (!found) {
    ++g_w.layout_fallback;
  } else {
    ++g_w.layout_found;
    if (!matches_old) ++g_w.layout_differs;
  }
}

bool OwnedFrontBindShadow(const OwnedBindInputs& in) {
  const uint64_t t0 = NowNs();
  const OwnedRecordState& s = *in.s;
  ++g_w.built;
  Cmp c{in};

  // ---- the system constants
  bool scaled_from_cache = false;
  BuildSystemConstants(in, g_sc, scaled_from_cache);
  // (Session 93) the storage window's base is the host's per-draw choice, not register state: the compare carries the old block's.
  if (in.old_sys) g_sc.shared_memory_window_base_dwords = in.old_sys->shared_memory_window_base_dwords;
  if (scaled_from_cache) ++g_w.scaled_from_cache;

  // ---- vertex buffer ranges (old: the processor's states; typed: the vertex fetch words)
  if (in.vs) {
    const Shader::ConstantRegisterMap& map = in.vs->constant_register_map();
    for (uint32_t i = 0; i < 3; ++i) {
      uint32_t bits = map.vertex_fetch_bitmap[i];
      while (bits) {
        const uint32_t j = uint32_t(std::countr_zero(bits));
        bits &= bits - 1;
        const uint32_t v = i * 32 + j;
        xenos::xe_gpu_vertex_fetch_t vf;
        std::memcpy(&vf, s.fetch + 2 * v, sizeof(vf));
        if (vf.type != xenos::FetchConstantType::kVertex &&
            !(vf.type == xenos::FetchConstantType::kInvalidVertex && REXCVAR_GET(gpu_allow_invalid_fetch_constants))) {
          continue;  // (the old path refuses such a draw)
        }
        ++g_w.vfetch_checked;
        const OwnedBindInputs::VertexState& old = in.vertex_state[v];
        if (old.address != vf.address) c.Note("vertex_range.address", "vertex fetch constant address in dwords", old.address, vf.address);
        if (old.size != vf.size) c.Note("vertex_range.size", "vertex fetch constant size in dwords", old.size, vf.size);
      }
    }
  }

  // ---- the used textures: the typed key against the cache's active binding
  for (uint32_t st = 0; st < 2; ++st) {
    for (uint32_t t = 0; t < in.texture_count[st]; ++t) {
      const uint32_t idx = in.textures[st][t].fetch_constant & 31;
      const TypedBinding b = BindingFor(idx, &s.fetch[6 * idx]);
      const OwnedBindInputs::ActiveTexture& a = in.active[idx];
      ++g_w.textures_checked;
      if (a.valid != b.key.valid) {
        c.Note("texture.valid", "the fetch constant's texture key is valid", a.valid, b.key.valid);
        continue;
      }
      if (!a.valid) continue;
      if (a.base_page != b.key.base_page) c.Note("texture.base_page", "texture base page", a.base_page, b.key.base_page);
      if (a.format != b.key.format) c.Note("texture.format", "texture format", a.format, b.key.format);
      if (a.width != b.key.w_minus_1 + 1) c.Note("texture.width", "texture width", a.width, b.key.w_minus_1 + 1);
      if (a.height != b.key.h_minus_1 + 1) c.Note("texture.height", "texture height", a.height, b.key.h_minus_1 + 1);
    }
  }

  // ---- image views and samplers: the typed lookup against what the old path bound
  for (uint32_t st = 0; st < 2; ++st) {
    for (uint32_t t = 0; t < in.texture_count[st]; ++t) {
      const OwnedBindInputs::ViewCheck& v = in.view[st][t];
      if (!v.compared) continue;
      ++g_w.views_checked;
      if (v.old_view != v.typed_view) c.Note("view.image_view", "image view handle the shader texture binding gets", v.old_view, v.typed_view);
    }
    if (in.sampler_checked[st] != in.sampler_old_count[st]) c.Note("sampler.count", "sampler count", in.sampler_old_count[st], in.sampler_checked[st]);
    for (uint32_t i = 0; i < in.sampler_checked[st]; ++i) {
      const OwnedBindInputs::SamplerCheck& sm = in.sampler[st][i];
      ++g_w.samplers_checked;
      if (sm.old_params != sm.typed_params) c.Note("sampler.parameters", "sampler parameters (packed)", sm.old_params, sm.typed_params);
      if (sm.old_sampler != sm.typed_sampler) c.Note("sampler.handle", "VkSampler handle", sm.old_sampler, sm.typed_sampler);
    }
  }
  g_w.skipped_3d_views += in.skipped_3d_views;
  g_w.native_views += in.skipped_native_views;

  // ---- the old path's constants: system block, float banks, bool / loop, fetch
  if (!in.old_sys || !in.old_vs_float || !in.old_ps_float || !in.old_bool_loop || !in.old_fetch) {
    ++g_w.not_built["old path's constant sources unavailable"];
  } else {
    ++g_w.compared;
    // system constants: byte-wise, with the fields the shaders do not read for this draw gated out (they keep an earlier draw's value in both
    // blocks: user clip planes beyond the enabled ones, point sizes of a non-point draw, the signs of unused textures, ...)
    {
      uint8_t gate[sizeof(SC)];
      std::memset(gate, 1, sizeof(gate));
      const auto clear = [&](uint32_t off, uint32_t n) { std::memset(gate + off, 0, n); };
      const SC& sc = g_sc;
      if (!(sc.flags & (ST::kSysFlag_VertexIndexLoad | ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad))) {
        clear(offsetof(SC, vertex_index_load_address), sizeof(sc.vertex_index_load_address));
      }
      {
        const uint32_t planes = s.clip_cntl.clip_disable ? 0 : uint32_t(std::popcount(uint32_t(s.clip_cntl.ucp_ena)));
        for (uint32_t p = planes; p < 6; ++p) clear(offsetof(SC, user_clip_planes) + p * 16, 16);
      }
      if (s.draw_initiator.prim_type != xenos::PrimitiveType::kPointList) {
        clear(offsetof(SC, point_vertex_diameter_min), 4);
        clear(offsetof(SC, point_vertex_diameter_max), 4);
        clear(offsetof(SC, point_constant_diameter), sizeof(sc.point_constant_diameter));
        clear(offsetof(SC, point_screen_diameter_to_ndc_radius), sizeof(sc.point_screen_diameter_to_ndc_radius));
      }
      for (uint32_t i = 0; i < 32; ++i) {
        if (!(in.used_texture_mask & (1u << i))) clear(offsetof(SC, texture_swizzled_signs) + i, 1);
      }
      // (the host swizzles are the texture cache's; not written at all with imageViewFormatSwizzle, then both sides are zero)
      if (!in.image_view_format_swizzle) clear(offsetof(SC, texture_swizzles), sizeof(sc.texture_swizzles));
      if (in.fsi) {
        // Only what the old path writes conditionally (it keeps an earlier draw's value otherwise): a render target with both keep masks
        // all-ones (its base, format flags, blend, clamp), and the stencil words unless depth-stencil and stencil are both enabled.
        // Gate value 2 = FSI-masked (counted apart from the history-only bytes).
        const auto fsi_clear = [&](uint32_t off, uint32_t n) { std::memset(gate + off, 2, n); };
        for (uint32_t i = 0; i < 4; ++i) {
          if (sc.edram_rt_keep_mask[i][0] == UINT32_MAX && sc.edram_rt_keep_mask[i][1] == UINT32_MAX) {
            fsi_clear(offsetof(SC, edram_rt_base_dwords_scaled) + 4 * i, 4);
            fsi_clear(offsetof(SC, edram_rt_format_flags) + 4 * i, 4);
            fsi_clear(offsetof(SC, edram_rt_blend_factors_ops) + 4 * i, 4);
            fsi_clear(offsetof(SC, edram_rt_clamp) + 16 * i, 16);
          }
        }
        if (!(g_ds_enabled && in.depth_control.stencil_enable)) {
          fsi_clear(offsetof(SC, edram_stencil_front), sizeof(sc.edram_stencil_front));
          fsi_clear(offsetof(SC, edram_stencil_back), sizeof(sc.edram_stencil_back));
        }
      }
      const uint8_t* old_b = reinterpret_cast<const uint8_t*>(in.old_sys);
      const uint8_t* new_b = reinterpret_cast<const uint8_t*>(&sc);
      g_w.bytes_sys += sizeof(SC);
      if (std::memcmp(old_b, new_b, sizeof(SC)) != 0) {
        bool gated_differs = false, fsi_masked_differs = false;
        for (const SysField& f : kSysFields) {
          bool diff = false;
          for (uint32_t b = f.offset; b < f.offset + f.size; ++b) {
            if (old_b[b] == new_b[b]) continue;
            if (gate[b] == 1) {
              diff = true;
            } else if (gate[b] == 2) {
              fsi_masked_differs = true;
            }
          }
          if (!diff) continue;
          gated_differs = true;
          uint64_t ov = 0, nv = 0;
          std::memcpy(&ov, old_b + f.offset, std::min<uint32_t>(f.size, 8));
          std::memcpy(&nv, new_b + f.offset, std::min<uint32_t>(f.size, 8));
          c.Note(std::string("sys.") + f.name, "system constants field (first 8 bytes)", ov, nv);
        }
        if (!gated_differs) ++(fsi_masked_differs ? g_w.fsi_masked : g_w.history_only);
      }
    }

    // float banks: the constants the shader uses, gathered in register order from the typed bank and from the register image
    uint32_t crowd_c76 = 0;
    const auto compare_floats = [&](const Shader* shader, const uint32_t* typed_words, const uint32_t* old_words, const char* block,
                                    bool pixel) {
      if (!shader) return;
      if (!s.banks_valid) {  // the snapshot was taken without the banks (the latch changed between the queue item and the claim)
        ++g_w.banks_skipped;
        return;
      }
      const Shader::ConstantRegisterMap& map = shader->constant_register_map();
      if (!map.float_count) return;
      alignas(16) uint8_t built[sizeof(float) * 4 * 256];
      alignas(16) uint8_t old_b[sizeof(float) * 4 * 256];
      uint16_t regs[256];
      const uint32_t n = GatherFloat(built, regs, typed_words, map.float_bitmap);
      GatherFloat(old_b, regs, old_words, map.float_bitmap);
      if (pixel) {
        ZeroDepthOfField(built, shader);
        ZeroDepthOfField(old_b, shader);
      }
      g_w.bytes_float += n;
      if (std::memcmp(old_b, built, n) == 0) return;
      for (uint32_t k = 0; k < n / 16; ++k) {
        if (std::memcmp(old_b + k * 16, built + k * 16, 16) == 0) continue;
        if (!pixel && regs[k] >= 76 && regs[k] <= 78 && in.kind != 0 && !s.crowd_member) {  // crowd-merge members write c76-c78 into the image, one snapshot (M4d)
          ++crowd_c76;
          continue;
        }
        uint64_t ov, nv;
        std::memcpy(&ov, old_b + k * 16, 8);
        std::memcpy(&nv, built + k * 16, 8);
        c.Note(std::string(block) + ".c" + std::to_string(regs[k]), "float constant (first 8 bytes of 16)", ov, nv);
      }
    };
    compare_floats(in.vs, s.vs_float, in.old_vs_float, "vs_float", false);
    if (crowd_c76) ++g_w.crowd_c76_skipped;
    compare_floats(in.ps, s.ps_float, in.old_ps_float, "ps_float", true);

    for (uint32_t w = 0; w < 40 && s.banks_valid; ++w) {
      if (in.old_bool_loop[w] != s.bool_loop[w]) c.Note(w < 8 ? "bool_loop.bool" : "bool_loop.loop", "bool / loop constant word", in.old_bool_loop[w], s.bool_loop[w]);
    }
    // (words 190-191, slot 31's tail, hold the inline path's staging address: the decoder leaves them out of its own compare too)
    for (uint32_t w = 0; w < 190; ++w) {
      if (in.old_fetch[w] != s.fetch[w]) c.Note("fetch.slot" + std::to_string(w / 6), "fetch constant word", in.old_fetch[w], s.fetch[w]);
    }
  }
  if (c.any) ++g_w.mismatch_draws;
  g_w.build_ns += NowNs() - t0;
  return true;
}

void OwnedFrontBindPrintWindow(uint64_t first_frame, uint64_t now_frame) {
  const Window& w = g_w;
  std::vector<std::pair<std::string, uint64_t>> v(w.by_field.begin(), w.by_field.end());
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string top;
  for (size_t i = 0; i < v.size() && i < 8; ++i) top += (top.empty() ? "" : ", ") + v[i].first + " " + std::to_string(v[i].second);
  if (top.empty()) top = "none";
  std::string nb;
  for (const auto& r : w.not_built) nb += (nb.empty() ? "" : ", ") + r.first + " " + std::to_string(r.second);
  if (nb.empty()) nb = "none";
  const uint64_t frames = std::max<uint64_t>(now_frame - first_frame, 1);
  REXGPU_INFO(
      "[d3d][front] M4c frames {}-{}: draws built {}, compared with the ring {}, draws with a mismatch {}; mismatches by block.field (top 8): {}; not "
      "built / not compared: {}; bytes compared: system {} float {}; texture keys checked {}, vertex ranges checked {}, image views checked {} (3D-as-2D not compared {}), samplers checked {}; texture resolution-scaled "
      "bits taken from the texture cache (draw scale > 1) in {} draws; draws where only ungated history bytes differed {} (only FSI-gated bytes {}); views owned or served by the native renderer, not compared {}; crowd-member vs_float c76-c78 differences skipped in {} draws (until M4d); draws whose snapshot had no banks {}; layout found find-only {} "
      "(differs from the old {}), fell back to the old description's {}; descriptor sets and dynamic offsets: copied from the old description (M4f builds them); "
      "build+compare {:.1f} us a frame",
      first_frame, now_frame, w.built, w.compared, w.mismatch_draws, top, nb, w.bytes_sys, w.bytes_float, w.textures_checked, w.vfetch_checked, w.views_checked, w.skipped_3d_views, w.samplers_checked,
      w.scaled_from_cache, w.history_only, w.fsi_masked, w.native_views, w.crowd_c76_skipped, w.banks_skipped, w.layout_found, w.layout_differs, w.layout_fallback, double(w.build_ns) / 1000.0 / double(frames));
  g_w = Window{};
}

// M4e (bit 4): an inline draw's constants and bindings are counted in their own window (the M4c counters swapped for the call).
void OwnedFrontBindBucket(bool inline_draws) {
  std::swap(g_w, g_winl);
  g_inline_bucket = inline_draws;
}

std::string OwnedFrontBindInlineSummary() {
  const Window& w = g_winl;
  std::vector<std::pair<std::string, uint64_t>> v(w.by_field.begin(), w.by_field.end());
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string top;
  for (size_t i = 0; i < v.size() && i < 8; ++i) top += (top.empty() ? "" : ", ") + v[i].first + " " + std::to_string(v[i].second);
  if (top.empty()) top = "none";
  std::string nb;
  for (const auto& r : w.not_built) nb += (nb.empty() ? "" : ", ") + r.first + " " + std::to_string(r.second);
  if (nb.empty()) nb = "none";
  char buf[160];
  std::snprintf(buf, sizeof buf, "build+compare %.1f us a window", double(w.build_ns) / 1000.0);
  std::string out = "built " + std::to_string(w.built) + ", compared " + std::to_string(w.compared) + ", draws with a mismatch " + std::to_string(w.mismatch_draws) +
                    " (by block.field, top 8: " + top + "), not built / not compared: " + nb + ", bytes compared system " + std::to_string(w.bytes_sys) +
                    " float " + std::to_string(w.bytes_float) + ", texture keys " + std::to_string(w.textures_checked) + ", vertex ranges " +
                    std::to_string(w.vfetch_checked) + ", views " + std::to_string(w.views_checked) + ", samplers " + std::to_string(w.samplers_checked) + ", " + buf;
  g_winl = Window{};
  return out;
}

}  // namespace rex::graphics::vulkan
