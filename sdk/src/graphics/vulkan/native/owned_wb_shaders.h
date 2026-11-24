#pragma once
// wwe2k14 patch (Session 92, M2d, Part 2): the write-back's compute shader sources, moved out of native_truth.cpp unchanged (the truth path and the owned write-back both
// build their pipelines from them; the owned file is owned_wb_objects.cpp). Docs: docs\native\d3d\S92_M2_DESIGN.md section 3 (M2d).

namespace rex::graphics::vulkan::native {

// Session 47 (M6): a texture's texels from an image, packed into a buffer (one dword a texel, rows of `width`) for a
// copy into the texture: exactly what the store, the resolve's copy and the texture load would give. Modes: 0
// k_8_8_8_8 from RGBA8 (bit 4: red and blue swapped, the copy's swap), 1 k_16_16 / 3 k_16_16_16_16 (two dwords a
// texel) from RGBA16_SNORM (the resolve's full copy reads the -32...32 values and packs them as 16-bit unorm:
// resolve_full_32bpp / 64bpp), 2 k_24_8_FLOAT depth (20e4, as the store rounds it, decoded to a 32-bit float as the
// texture load does).
inline constexpr char kTruthFillComputeShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer Out { uint texels[]; };
layout(set = 0, binding = 1) uniform sampler2D source;
layout(push_constant) uniform Constants {
  uint origin_x;  // the image's first pixel
  uint origin_y;
  uint width;
  uint height;
  uint mode;
  uint gap;  // Session 47: the half-pixel offset gap (host pixels, x | y << 8) the copy fills from the first covered row / column
  uint dst_x;  // Session 52 (TEXOUT): where texel (0, 0) goes in the texture
  uint dst_y;
} c;
#if TEXOUT == 1
layout(set = 1, binding = 0, r32ui) uniform writeonly uimage2D texture_out;
#elif TEXOUT == 2
layout(set = 1, binding = 0, rgba16ui) uniform writeonly uimage2D texture_out;
#elif TEXOUT == 3
layout(set = 1, binding = 0, r8ui) uniform writeonly uimage2D texture_out;  // Session 59: k_8
#elif TEXOUT == 4
layout(set = 1, binding = 0, r16ui) uniform writeonly uimage2D texture_out;  // Session 59: k_4_4_4_4
#endif
uvec2 Gap(uvec2 p) { return max(p, uvec2(c.gap & 0xFFu, c.gap >> 8)); }
uint DepthToFloat24(float z) {
  z = clamp(z, 0.0, 1.0);
  uint u = floatBitsToUint(z);
  uint b = u < 0x38800000u ? ((u & 0x7FFFFFu) | 0x800000u) >> min(113u - (u >> 23), 24u) : u - (112u << 23);
  b += 3u + ((b >> 3) & 1u);
  return (b >> 3) & 0xFFFFFFu;
}
uint Float24ToFloat32(uint depth) {
  uint mantissa = depth & 1048575u;
  uint exponent = depth >> 20u;
  if (exponent == 0u) {
    if (mantissa == 0u) return 0u;
    uint msb = uint(findMSB(mantissa));
    exponent = msb + 4294967277u;
    mantissa = (mantissa << (20u - msb)) & 1048575u;
  }
  return ((exponent + 112u) << 23u) | (mantissa << 3u);
}
void main() {
  uvec2 g = gl_GlobalInvocationID.xy;
  if (g.x >= c.width || g.y >= c.height) return;
  vec4 raw = texelFetch(source, ivec2(Gap(g) + uvec2(c.origin_x, c.origin_y)), 0);
  uint mode = c.mode & 0xFu, value;
  if (mode == 0u) {
    uvec4 u = uvec4(round(clamp(raw, 0.0, 1.0) * 255.0));
    if ((c.mode & 16u) != 0u) u = u.bgra;
    value = u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
  } else if (mode == 1u || mode == 3u) {
    // The resolve's full copy: the 16-bit values (-32...32) to floats, then packed as 16-bit unorm.
    vec4 s = vec4(ivec4(round(clamp(raw, -1.0, 1.0) * 32767.0)));
    precise vec4 f = s * 0.000976592302;
    precise vec4 scaled = clamp(max(f, vec4(-1.0)), 0.0, 1.0) * 65535.0;
    precise vec4 biased = scaled + 0.5;
    uvec4 u = uvec4(biased);
    value = u.r | (u.g << 16);
    if (mode == 3u) {
#if TEXOUT == 2
      imageStore(texture_out, ivec2(g + uvec2(c.dst_x, c.dst_y)), u);
#elif !defined(TEXOUT)
      texels[(g.y * c.width + g.x) * 2u] = value;
      texels[(g.y * c.width + g.x) * 2u + 1u] = u.b | (u.a << 16);
#endif
      return;
    }
  } else if (mode == 4u || mode == 5u) {
    // Several texels a dword (k_8: 4, k_4_4_4_4: 2): this invocation's texels along x. The full copy's 8-bit values to
    // floats (x 1/255) packed as 8 / 4-bit unorm; k_8 takes red; k_4_4_4_4 is R | G << 4 | B << 8 | A << 12 in memory,
    // which the texture load turns into B4G4R4A4 (A | R << 4 | G << 8 | B << 12).
#if TEXOUT == 3 || TEXOUT == 4
    // Session 59 (gpu_native_truth_fill_direct_small): one texel an invocation, into the texture's R8_UINT / R16_UINT
    // view - the same value as below, in texture form.
    {
      precise vec4 f = vec4(uvec4(round(clamp(raw, 0.0, 1.0) * 255.0))) * 0.00392156886;
      if ((c.mode & 16u) != 0u) f = f.bgra;
      uint texel_value;
      if (mode == 4u) {
        precise float b = clamp(f.r, 0.0, 1.0) * 255.0 + 0.5;
        texel_value = uint(b) & 0xFFu;
      } else {
        precise vec4 n = clamp(f, 0.0, 1.0) * 15.0 + 0.5;
        uvec4 u = uvec4(n) & 0xFu;
        texel_value = u.a | (u.r << 4) | (u.g << 8) | (u.b << 12);
      }
      imageStore(texture_out, ivec2(g + uvec2(c.dst_x, c.dst_y)), uvec4(texel_value));
      return;
    }
#endif
    uint per = mode == 4u ? 4u : 2u;
    if (g.x * per >= c.width) return;
    value = 0u;
    for (uint k = 0u; k < per; ++k) {
      vec4 texel = texelFetch(source, ivec2(Gap(uvec2(g.x * per + k, g.y)) + uvec2(c.origin_x, c.origin_y)), 0);
      precise vec4 f = vec4(uvec4(round(clamp(texel, 0.0, 1.0) * 255.0))) * 0.00392156886;
      if ((c.mode & 16u) != 0u) f = f.bgra;
      if (mode == 4u) {
        precise float b = clamp(f.r, 0.0, 1.0) * 255.0 + 0.5;
        value |= (uint(b) & 0xFFu) << (8u * k);
      } else {
        precise vec4 n = clamp(f, 0.0, 1.0) * 15.0 + 0.5;
        uvec4 u = uvec4(n) & 0xFu;
        value |= (u.a | (u.r << 4) | (u.g << 8) | (u.b << 12)) << (16u * k);
      }
    }
    texels[g.y * (c.width / per) + g.x] = value;
    return;
  } else {
    value = Float24ToFloat32(DepthToFloat24(raw.r));
  }
#if TEXOUT == 1
  imageStore(texture_out, ivec2(g + uvec2(c.dst_x, c.dst_y)), uvec4(value));
#elif !defined(TEXOUT)
  texels[g.y * c.width + g.x] = value;
#endif
}
)";

// Session 52 (gpu_native_truth_direct_resolve_scaled_full): a full k_16_16 (mode 1, 32 bits a texel) / k_16_16_16_16
// (mode 3, 64 bits) colour copy at a resolution scale from a truth image (RGBA16_SNORM) into the scaled resolve buffer.
// Each texel's value as kTruthFillComputeShader packs it (the resolve's full copy: the 16-bit values -32...32 to floats,
// packed as 16-bit unorm - the texture's bits, which the texture load gives from memory by the endian swap alone), the
// 32-bit endian swap, at its place in the scaled buffer as the scaled texture load reads it
// (texture_load_scaled_direct.cs.glsl): 16-byte groups of 4 (32 bits) / 2 (64 bits) host texels; group (gx, y) of host
// texels is guest group (gx / scale_x, y / scale_y) at its guest 2D tiled offset times scale_x * scale_y, plus the
// sub-group ((gx % scale_x) * scale_y + y % scale_y) x 16 bytes. One host texel an invocation.
inline constexpr char kNativeResolveScaledFullComputeShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer Out { uint texels[]; };
layout(set = 0, binding = 1) uniform sampler2D source;
layout(set = 1, binding = 0, std430) writeonly buffer DestBuffer { uint dest[]; };
#ifdef FULL_DEPTH
layout(set = 0, binding = 2) uniform usampler2D stencil_source;  // Session 59: the depth image's stencil
#endif
#if defined(FULL_TEXOUT) && FULL_TEXOUT == 2
layout(set = 2, binding = 0, rgba16ui) uniform writeonly uimage2D texture_out;
#elif defined(FULL_TEXOUT)
layout(set = 2, binding = 0, r32ui) uniform writeonly uimage2D texture_out;
#endif
layout(push_constant) uniform Constants {
  uint origin_x;  // the copy's first pixel in the image (host)
  uint origin_y;
  uint width;  // host texels
  uint height;
  uint mode;
  uint gap;  // the half-pixel offset gap (host pixels, x | y << 8)
  uint dest_x0;  // the destination's first texel (guest, relative to the binding's start)
  uint dest_y0;
  uint pitch;  // guest texels, aligned to 32
  uint info;  // endian | scale_x << 4 | scale_y << 8
#ifdef FULL_TEXOUT
  uint tex_x0;  // Session 59: the copy's first host texel in the texture (an atlas part's offset included)
  uint tex_y0;
#endif
} c;
int XeTextureTiledOffset2D(ivec2 p, uint pitch_aligned, uint bpb_log2) {
  int macro = ((p.x >> 5) + (p.y >> 5) * int(pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  int micro = ((p.x & 7) + ((p.y & 14) << 2)) << bpb_log2;
  int offset = macro + ((micro & -16) << 1) + (micro & 15) + ((p.y & 1) << 4);
  return ((offset & -512) << 3) + ((p.y & 16) << 7) + ((offset & 448) << 2) +
         (((((p.y & 8) >> 2) + (p.x >> 3)) & 3) << 6) + (offset & 63);
}
uint XeEndianSwap32(uint value, uint endian) {
  if (endian == 1u || endian == 2u) value = ((value & 16711935u) << 8u) | ((value & 4278255360u) >> 8u);
  if (endian == 2u || endian == 3u) value = (value << 16u) | (value >> 16u);
  return value;
}
#ifdef FULL_DEPTH
// Session 59 (gpu_native_truth_direct_resolve_depth at a resolution scale): one host texel an invocation of a 32bpp
// depth copy - the dword the truth store writes (DepthToEdram << 8 | the stencil byte: the image's, mode bit 1, or bits
// 8-15; mode bit 0: 20e4), at its place in the scaled buffer as mode 1's 32bpp texels, endian swapped.
uint DepthToEdram(float z, bool is_float) {
  z = clamp(z, 0.0, 1.0);
  if (!is_float) {
    precise float scaled = z * 16777215.0;
    return uint(roundEven(scaled));
  }
  uint u = floatBitsToUint(z);
  uint b = u < 0x38800000u ? ((u & 0x7FFFFFu) | 0x800000u) >> min(113u - (u >> 23), 24u) : u - (112u << 23);
  b += 3u + ((b >> 3) & 1u);
  return (b >> 3) & 0xFFFFFFu;
}
void main() {
  uvec2 g = gl_GlobalInvocationID.xy;
  if (g.x >= c.width || g.y >= c.height) return;
  ivec2 q = ivec2(max(g, uvec2(c.gap & 0xFFu, c.gap >> 8)) + uvec2(c.origin_x, c.origin_y));
  float z = texelFetch(source, q, 0).r;
  uint st = (c.mode & 2u) != 0u ? (texelFetch(stencil_source, q, 0).r & 0xFFu) : ((c.mode >> 8) & 0xFFu);
  uint value = (DepthToEdram(z, (c.mode & 1u) != 0u) << 8) | st;
  uint endian = c.info & 7u, sx = (c.info >> 4) & 15u, sy = (c.info >> 8) & 15u;
  uint hx = c.dest_x0 * sx + g.x, hy = c.dest_y0 * sy + g.y;
  uint gsx = hx >> 2u;
  uint gux = gsx / sx, guy = hy / sy;
  uint offset = uint(XeTextureTiledOffset2D(ivec2(int(gux << 2u), int(guy)), c.pitch, 2u)) * (sx * sy) +
                (((((gsx - gux * sx) * sy + (hy - guy * sy)) << 2u) + (hx & 3u)) << 2u);
  dest[offset >> 2u] = XeEndianSwap32(value, endian);
}
#else
void main() {
  uvec2 g = gl_GlobalInvocationID.xy;
  uint small_mode = c.mode & 15u;
  if (small_mode == 4u || small_mode == 5u) {
    // Session 59 (gpu_native_truth_direct_resolve_scaled_small): one dword an invocation of a full k_8 (4 host texels)
    // / k_4_4_4_4 (2) copy from an RGBA8 image - the texels as the 1x direct copy packs them (memory form), at their
    // place in the scaled buffer as the stock scaled copy writes it (resolve_full_8bpp / 16bpp_scaled): 8-texel units
    // (8 / 16 bytes) of one host row; unit (ux, y) of host texels is guest unit (ux / scale_x, y / scale_y) at its guest
    // 2D tiled offset times scale_x * scale_y, plus the sub-unit ((ux % scale_x) * scale_y + y % scale_y) x 8 texels.
    uint per = small_mode == 4u ? 4u : 2u;
    if (g.x * per >= c.width || g.y >= c.height) return;
    uint value = 0u;
    for (uint k = 0u; k < per; ++k) {
      uvec2 pk = max(uvec2(g.x * per + k, g.y), uvec2(c.gap & 0xFFu, c.gap >> 8));
      vec4 texel = texelFetch(source, ivec2(pk + uvec2(c.origin_x, c.origin_y)), 0);
      precise vec4 f = vec4(uvec4(round(clamp(texel, 0.0, 1.0) * 255.0))) * 0.00392156886;
      if ((c.mode & 16u) != 0u) f = f.bgra;
      if (small_mode == 4u) {
        precise float b = clamp(f.r, 0.0, 1.0) * 255.0 + 0.5;
        value |= (uint(b) & 0xFFu) << (8u * k);
      } else {
        precise vec4 n = clamp(f, 0.0, 1.0) * 15.0 + 0.5;
        uvec4 u = uvec4(n) & 0xFu;
        value |= (u.r | (u.g << 4) | (u.b << 8) | (u.a << 12)) << (16u * k);
      }
    }
    uint endian = c.info & 7u, sx = (c.info >> 4) & 15u, sy = (c.info >> 8) & 15u;
    uint bpb_log2 = small_mode == 4u ? 0u : 1u;
    uint hx = c.dest_x0 * sx + g.x * per, hy = c.dest_y0 * sy + g.y;
    uint unit = hx >> 3u;
    uint gux = unit / sx, guy = hy / sy;
    uint offset = uint(XeTextureTiledOffset2D(ivec2(int(gux << 3u), int(guy)), c.pitch, bpb_log2)) * (sx * sy) +
                  (((((unit - gux * sx) * sy + (hy - guy * sy)) << 3u) + (hx & 7u)) << bpb_log2);
    // (Review: as the stock 8 / 16bpp copies - k_8 never swapped, k_4_4_4_4 only 8-in-16.)
    dest[offset >> 2u] = XeEndianSwap32(value, small_mode == 4u || endian != 1u ? 0u : 1u);
    return;
  }
  if (g.x >= c.width || g.y >= c.height) return;
  uvec2 p = max(g, uvec2(c.gap & 0xFFu, c.gap >> 8));
  vec4 raw = texelFetch(source, ivec2(p + uvec2(c.origin_x, c.origin_y)), 0);
  vec4 s = vec4(ivec4(round(clamp(raw, -1.0, 1.0) * 32767.0)));
  precise vec4 f = s * 0.000976592302;
  precise vec4 scaled = clamp(max(f, vec4(-1.0)), 0.0, 1.0) * 65535.0;
  precise vec4 biased = scaled + 0.5;
  uvec4 u = uvec4(biased);
  uint endian = c.info & 7u, sx = (c.info >> 4) & 15u, sy = (c.info >> 8) & 15u;
  bool bpb64 = c.mode == 3u;
  uint bpb_log2 = bpb64 ? 3u : 2u, group_log2 = bpb64 ? 1u : 2u;
  uint hx = c.dest_x0 * sx + g.x, hy = c.dest_y0 * sy + g.y;
  uint gsx = hx >> group_log2;
  uint gux = gsx / sx, guy = hy / sy;
  uint offset = uint(XeTextureTiledOffset2D(ivec2(int(gux << group_log2), int(guy)), c.pitch, bpb_log2)) * (sx * sy) +
                (((((gsx - gux * sx) * sy + (hy - guy * sy)) << group_log2) + (hx & ((1u << group_log2) - 1u)))
                 << bpb_log2);
  uint index = offset >> 2u;
  dest[index] = XeEndianSwap32(u.r | (u.g << 16), endian);
  if (bpb64) dest[index + 1u] = XeEndianSwap32(u.b | (u.a << 16), endian);
#if defined(FULL_TEXOUT) && FULL_TEXOUT == 2
  imageStore(texture_out, ivec2(c.tex_x0 + g.x, c.tex_y0 + g.y), u);  // (Session 59: the bits before the swap)
#elif defined(FULL_TEXOUT)
  imageStore(texture_out, ivec2(c.tex_x0 + g.x, c.tex_y0 + g.y), uvec4(u.r | (u.g << 16), 0u, 0u, 0u));
#endif
}
#endif
)";

// Session 59 (gpu_native_truth_direct_resolve_full_1x): one invocation per dword of a full k_8 (4 texels) /
// k_4_4_4_4 (2 texels) copy at 1x: the texels as kTruthFillComputeShader modes 4 / 5 convert them, in memory form
// (k_4_4_4_4: R | G << 4 | B << 8 | A << 12), at dest_base + their 2D tiled offset, 32-bit endian swapped.
inline constexpr char kNativeResolveFull1xComputeShader[] = R"(
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer Out { uint texels[]; };
layout(set = 0, binding = 1) uniform sampler2D source;
layout(set = 1, binding = 0, std430) buffer DestBuffer { uint dest[]; };
#ifdef FULL1X_DEPTH
layout(set = 0, binding = 2) uniform usampler2D stencil_source;  // Session 59: the depth image's stencil
#endif
#if defined(FULL1X_TEXOUT) && FULL1X_TEXOUT == 2
layout(set = 2, binding = 0, r16ui) uniform writeonly uimage2D texture_out;
#elif defined(FULL1X_TEXOUT)
layout(set = 2, binding = 0, r8ui) uniform writeonly uimage2D texture_out;
#endif
layout(push_constant) uniform Constants {
  uint origin_x;  // the image pixel of the copy's first texel
  uint origin_y;
  uint width;  // texels
  uint height;
  uint mode;  // 4 k_8, 5 k_4_4_4_4; bit 4: red and blue swapped
  uint dest_base;  // bytes, relative to the binding
  uint dest_x0;  // the copy's first texel in the destination
  uint dest_y0;
  uint pitch;  // texels, aligned to 32
  uint endian;
#ifdef FULL1X_TEXOUT
  uint tex_x0;  // Session 59: the copy's first texel in the texture
  uint tex_y0;
#endif
} c;
int XeTextureTiledOffset2D(ivec2 p, uint pitch_aligned, uint bpb_log2) {
  int macro = ((p.x >> 5) + (p.y >> 5) * int(pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  int micro = ((p.x & 7) + ((p.y & 14) << 2)) << bpb_log2;
  int offset = macro + ((micro & -16) << 1) + (micro & 15) + ((p.y & 1) << 4);
  return ((offset & -512) << 3) + ((p.y & 16) << 7) + ((offset & 448) << 2) +
         (((((p.y & 8) >> 2) + (p.x >> 3)) & 3) << 6) + (offset & 63);
}
uint XeEndianSwap32(uint value, uint endian) {
  if (endian == 1u || endian == 2u) value = ((value & 16711935u) << 8u) | ((value & 4278255360u) >> 8u);
  if (endian == 2u || endian == 3u) value = (value << 16u) | (value >> 16u);
  return value;
}
#ifdef FULL1X_DEPTH
// Session 59 (gpu_native_truth_direct_resolve_depth): one invocation per texel of a 32bpp depth copy - the EDRAM dword
// the truth store writes (kTruthStoreComputeShader's DepthToEdram << 8 | the stencil byte: the image's, mode bit 1, or
// bits 8-15) at dest_base + its 2D tiled offset, endian swapped (the stock fast 32bpp copy of those dwords). Mode bit 0:
// 20e4 depth.
uint DepthToEdram(float z, bool is_float) {
  z = clamp(z, 0.0, 1.0);
  if (!is_float) {
    precise float scaled = z * 16777215.0;
    return uint(roundEven(scaled));
  }
  uint u = floatBitsToUint(z);
  uint b = u < 0x38800000u ? ((u & 0x7FFFFFu) | 0x800000u) >> min(113u - (u >> 23), 24u) : u - (112u << 23);
  b += 3u + ((b >> 3) & 1u);
  return (b >> 3) & 0xFFFFFFu;
}
void main() {
  uvec2 g = gl_GlobalInvocationID.xy;
  if (g.x >= c.width || g.y >= c.height) return;
  ivec2 p = ivec2(uvec2(c.origin_x, c.origin_y) + g);
  float z = texelFetch(source, p, 0).r;
  uint s = (c.mode & 2u) != 0u ? (texelFetch(stencil_source, p, 0).r & 0xFFu) : ((c.mode >> 8) & 0xFFu);
  uint value = (DepthToEdram(z, (c.mode & 1u) != 0u) << 8) | s;
  uint offset = c.dest_base + uint(XeTextureTiledOffset2D(ivec2(int(c.dest_x0 + g.x), int(c.dest_y0 + g.y)), c.pitch, 2u));
  dest[offset >> 2u] = XeEndianSwap32(value, c.endian);
}
#else
void main() {
  uvec2 g = gl_GlobalInvocationID.xy;
  uint mode = c.mode & 15u;
  uint per = mode == 4u ? 4u : 2u;
  if (g.x * per >= c.width || g.y >= c.height) return;
  uint value = 0u;
  for (uint k = 0u; k < per; ++k) {
    vec4 texel = texelFetch(source, ivec2(uvec2(c.origin_x + g.x * per + k, c.origin_y + g.y)), 0);
    precise vec4 f = vec4(uvec4(round(clamp(texel, 0.0, 1.0) * 255.0))) * 0.00392156886;
    if ((c.mode & 16u) != 0u) f = f.bgra;
    uint texel_bits;
    if (mode == 4u) {
      precise float b = clamp(f.r, 0.0, 1.0) * 255.0 + 0.5;
      value |= (uint(b) & 0xFFu) << (8u * k);
      texel_bits = uint(b) & 0xFFu;
    } else {
      precise vec4 n = clamp(f, 0.0, 1.0) * 15.0 + 0.5;
      uvec4 u = uvec4(n) & 0xFu;
      value |= (u.r | (u.g << 4) | (u.b << 8) | (u.a << 12)) << (16u * k);
      texel_bits = u.a | (u.r << 4) | (u.g << 8) | (u.b << 12);  // (the texture's B4G4R4A4, as the load gives it)
    }
#ifdef FULL1X_TEXOUT
    imageStore(texture_out, ivec2(int(c.tex_x0 + g.x * per + k), int(c.tex_y0 + g.y)), uvec4(texel_bits, 0u, 0u, 0u));
#endif
  }
  uint offset = c.dest_base +
                uint(XeTextureTiledOffset2D(ivec2(int(c.dest_x0 + g.x * per), int(c.dest_y0 + g.y)), c.pitch,
                                            mode == 4u ? 0u : 1u));
  // (Review: as the stock resolve_full_8bpp / 16bpp - k_8 never swapped, k_4_4_4_4 only 8-in-16.)
  dest[offset >> 2u] = XeEndianSwap32(value, mode == 4u || c.endian != 1u ? 0u : 1u);
}
#endif
)";

}  // namespace rex::graphics::vulkan::native
