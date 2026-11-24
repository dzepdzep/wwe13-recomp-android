/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#version 460

// wwe2k14 patch (Session 7c, TEXDIRECT): direct-to-image variants of the
// resolution-scaled 32bpb and 64bpb texture load shaders
// (vulkan_spirv/texture_load_32bpb_scaled_cs.h, texture_load_64bpb_scaled_cs.h,
// generated from Xenia's texture_load.xesli / texture_load_32bpb.xesli /
// texture_load_64bpb.xesli / texture_address.xesli). Used by
// VulkanTextureCache::LoadTextureDataFromResidentMemoryImpl with
// vulkan_texture_load_direct for single-level, single-layer textures loaded
// from the scaled resolve buffer.
//
// Everything up to and including the two 16-byte source loads and their endian
// swaps is a statement-by-statement transcription of those shaders' SPIR-V (the
// disassembly in the generated headers), so the two uvec4 values per thread
// are the same words the buffer variants store. The only difference is where
// they go: the buffer variants store them to the scratch buffer at
// host_offset + row * host_pitch + block_x * bytes_per_block (the row being
// block_index.z * size_blocks.y + block_index.y), from where
// vkCmdCopyBufferToImage copies texel (block_x, row) of a 1-level, 1-layer
// image bit for bit; here each block is stored to texel
// (block_x, block_index.y) of the image through a view with a size-matched
// unsigned integer format, which stores the raw bits (R32_UINT for 32bpb - one
// 32-bit word per texel; R16G16B16A16_UINT for 64bpb - the two 32-bit words of
// a block as four 16-bit halves, low half first, which is the byte order of
// the block in the buffer on a little-endian host). The buffer variants also
// write the blocks past the image width (the scratch row pitch is rounded up
// to the 8 or 4 blocks a thread writes), which the copy never reads; those are
// skipped here. Only depth 1 (dispatch Z = 1, block_index.z = 0) is supported.
//
// Generated header (no glslangValidator in the SDK build; built from the
// SDK's glslang submodule out of tree):
//   glslangValidator -V --target-env vulkan1.0 -DXE_TEXTURE_LOAD_DIRECT_32BPB
//     texture_load_scaled_direct.cs.glsl -> vulkan_spirv/texture_load_32bpb_scaled_direct_cs.h
//   glslangValidator -V --target-env vulkan1.0 -DXE_TEXTURE_LOAD_DIRECT_64BPB
//     texture_load_scaled_direct.cs.glsl -> vulkan_spirv/texture_load_64bpb_scaled_direct_cs.h
// (the header wraps the -H disassembly in #if 0 and the binary as a uint32_t
// array, like the `xb buildshaders` headers).

#if defined(XE_TEXTURE_LOAD_DIRECT_32BPB) == defined(XE_TEXTURE_LOAD_DIRECT_64BPB)
#error Define exactly one of XE_TEXTURE_LOAD_DIRECT_32BPB and XE_TEXTURE_LOAD_DIRECT_64BPB.
#endif
// wwe2k14 patch (Session 22, TEXDEPTH): XE_TEXTURE_LOAD_DIRECT_DEPTH_FLOAT (with _32BPB) = the direct variant of
// the scaled 24-bit float depth load shader (vulkan_spirv/texture_load_depth_float_scaled_cs.h, k_24_8_FLOAT ->
// R32_SFLOAT). Its SPIR-V differs from the 32bpb one only by a 24-bit float -> 32-bit float conversion of each
// endian-swapped 16-byte group before the store (XeDepthFloat24To32 below); the load info (bytes per block, blocks
// per thread) is the same. Generated with tools\perf\gen_spirv_header.py (which reproduces the two Session 7c
// headers bit for bit):
//   -DXE_TEXTURE_LOAD_DIRECT_32BPB -DXE_TEXTURE_LOAD_DIRECT_DEPTH_FLOAT
//     -> vulkan_spirv/texture_load_depth_float_scaled_direct_cs.h
#if defined(XE_TEXTURE_LOAD_DIRECT_DEPTH_FLOAT) && !defined(XE_TEXTURE_LOAD_DIRECT_32BPB)
#error XE_TEXTURE_LOAD_DIRECT_DEPTH_FLOAT needs XE_TEXTURE_LOAD_DIRECT_32BPB.
#endif

layout(local_size_x = 4, local_size_y = 32, local_size_z = 1) in;

// TextureCache::LoadConstants.
layout(push_constant) uniform XeTextureLoadConstants {
  uint xe_texture_load_is_tiled_3d_endian_scale;
  uint xe_texture_load_guest_offset;
  uint xe_texture_load_guest_pitch_aligned;
  uint xe_texture_load_guest_z_stride_block_rows_aligned;
  uvec3 xe_texture_load_size_blocks;
  uint xe_texture_load_host_offset;
  uint xe_texture_load_host_pitch;
  uint xe_texture_load_height_texels;
};

// The scaled resolve buffer range of the texture (same binding as the buffer
// variants: set 1).
layout(std430, set = 1, binding = 0) readonly buffer XeTextureLoadSource {
  uvec4 xe_texture_load_source[];
};

// The texture's level 0, layer 0 (set 0, where the buffer variants have the
// destination buffer).
#ifdef XE_TEXTURE_LOAD_DIRECT_32BPB
layout(r32ui, set = 0, binding = 0) uniform writeonly uimage2D xe_texture_load_dest;
#else
layout(rgba16ui, set = 0, binding = 0) uniform writeonly uimage2D xe_texture_load_dest;
#endif

#ifdef XE_TEXTURE_LOAD_DIRECT_32BPB
const uint kBytesPerBlockLog2 = 2u;
// Blocks per thread: 8 (two 16-byte groups of 4).
const uint kBlocksPerThreadLog2 = 3u;
// Blocks per 16-byte group: 4.
const uint kBlocksPerGroupLog2 = 2u;
#else
const uint kBytesPerBlockLog2 = 3u;
// Blocks per thread: 4 (two 16-byte groups of 2).
const uint kBlocksPerThreadLog2 = 2u;
// Blocks per 16-byte group: 2.
const uint kBlocksPerGroupLog2 = 1u;
#endif

int XeTextureTiledOffset2D(ivec2 p, uint pitch_aligned, uint bpb_log2) {
  int macro = ((p.x >> 5) + (p.y >> 5) * int(pitch_aligned >> 5u)) << (bpb_log2 + 7u);
  int micro = ((p.x & 7) + ((p.y & 14) << 2)) << bpb_log2;
  int offset = macro + ((micro & -16) << 1) + (micro & 15) + ((p.y & 1) << 4);
  return ((offset & -512) << 3) + ((p.y & 16) << 7) + ((offset & 448) << 2) +
         (((((p.y & 8) >> 2) + (p.x >> 3)) & 3) << 6) + (offset & 63);
}

int XeTextureTiledOffset3D(ivec3 p, uint pitch_aligned, uint height_aligned, uint bpb_log2) {
  int macro_outer =
      ((p.y >> 4) + (p.z >> 2) * int(height_aligned >> 4u)) * int(pitch_aligned >> 5u);
  int macro = ((((p.x >> 5) + macro_outer) << (bpb_log2 + 6u)) & 268435455) << 1;
  int micro = (((p.x & 7) + ((p.y & 6) << 2)) << (bpb_log2 + 6u)) >> 6;
  int offset_outer = ((p.y >> 3) + (p.z >> 2)) & 1;
  int offset1 = offset_outer + ((((p.x >> 3) + (offset_outer << 1)) & 3) << 1);
  int offset2 = ((macro + (micro & -16)) << 1) + (micro & 15) +
                ((p.z & 3) << (bpb_log2 + 6u)) + ((p.y & 1) << 4);
  int address = (offset1 & 1) << 3;
  address += (offset2 >> 6) & 7;
  address <<= 3;
  address += offset1 & -2;
  address <<= 2;
  address += offset2 & -512;
  address <<= 3;
  address += offset2 & 63;
  return address;
}

uvec4 XeEndianSwap32(uvec4 value, uint endian) {
  // xenos::Endian: 1 = 8in16, 2 = 8in32, 3 = 16in32.
  if (endian == 1u || endian == 2u) {
    value = ((value & 16711935u) << 8u) | ((value & 4278255360u) >> 8u);
  }
  if (endian == 2u || endian == 3u) {
    value = (value << 16u) | (value >> 16u);
  }
  return value;
}

#ifdef XE_TEXTURE_LOAD_DIRECT_DEPTH_FLOAT
// Statement by statement from texture_load_depth_float_scaled_cs.h (the ids of its first copy, %7000 .. %24661):
// the guest's 24-bit float depth (20e4, the top 24 bits of the word) to a 32-bit float, denormals normalized,
// 0 kept as 0. Integer operations only (no float rounding or NaN handling involved).
uvec4 XeDepthFloat24To32(uvec4 value) {
  uvec4 depth = value >> 8u;                                       // %7000
  uvec4 mantissa = depth & uvec4(1048575u);                        // %12837
  uvec4 exponent = depth >> 20u;                                   // %22636
  bvec4 is_denormal = equal(exponent, uvec4(0u));                  // %16585
  uvec4 msb = uvec4(findMSB(mantissa));                            // %11339, %10773
  uvec4 shift = uvec4(20u) - msb;                                  // %6266
  uvec4 exponent_denormal = msb + uvec4(4294967277u);              // %8720
  exponent = mix(exponent, exponent_denormal, is_denormal);        // %10351
  uvec4 mantissa_normalized = (mantissa << shift) & uvec4(1048575u);  // %23252, %18842
  mantissa = mix(mantissa, mantissa_normalized, is_denormal);      // %10910
  uvec4 result = ((exponent + uvec4(112u)) << 23u) | (mantissa << 3u);  // %24569, %20351, %16294, %22396
  return mix(result, uvec4(0u), equal(depth, uvec4(0u)));          // %12703, %24661
}
#endif

void XeStoreGroup(uint first_block_x, uint row, uint width_blocks, uvec4 blocks) {
#ifdef XE_TEXTURE_LOAD_DIRECT_32BPB
  for (uint i = 0u; i < 4u; ++i) {
    uint x = first_block_x + i;
    if (x < width_blocks) {
      imageStore(xe_texture_load_dest, ivec2(int(x), int(row)), uvec4(blocks[i], 0u, 0u, 0u));
    }
  }
#else
  if (first_block_x < width_blocks) {
    imageStore(xe_texture_load_dest, ivec2(int(first_block_x), int(row)),
               uvec4(blocks.x & 65535u, blocks.x >> 16u, blocks.y & 65535u, blocks.y >> 16u));
  }
  if (first_block_x + 1u < width_blocks) {
    imageStore(xe_texture_load_dest, ivec2(int(first_block_x + 1u), int(row)),
               uvec4(blocks.z & 65535u, blocks.z >> 16u, blocks.w & 65535u, blocks.w >> 16u));
  }
#endif
}

void main() {
  uint flags = xe_texture_load_is_tiled_3d_endian_scale;
  bool is_3d_tiled = (flags & 2u) != 0u;
  uint endian = (flags >> 2u) & 3u;
  uvec2 resolution_scale = (uvec2(flags) >> uvec2(4u, 7u)) & 7u;

  uvec3 block_index = gl_GlobalInvocationID << uvec3(kBlocksPerThreadLog2, 0u, 0u);
  if (any(greaterThanEqual(block_index.xy, xe_texture_load_size_blocks.xy))) {
    return;
  }

  // Which host 16-byte group of the scaled guest 16-byte group this is.
  uvec2 group_scaled = uvec2(block_index.x >> kBlocksPerGroupLog2, block_index.y);
  uvec2 group_unscaled = group_scaled / resolution_scale;
  uvec3 guest_block_index =
      uvec3(group_unscaled.x << kBlocksPerGroupLog2, group_unscaled.y, block_index.z);
  uint guest_offset_unscaled;
  if (is_3d_tiled) {
    guest_offset_unscaled = uint(XeTextureTiledOffset3D(
        ivec3(guest_block_index), xe_texture_load_guest_pitch_aligned,
        xe_texture_load_guest_z_stride_block_rows_aligned, kBytesPerBlockLog2));
  } else {
    guest_offset_unscaled =
        uint(XeTextureTiledOffset2D(ivec2(guest_block_index.xy),
                                    xe_texture_load_guest_pitch_aligned, kBytesPerBlockLog2));
  }
  uvec2 group_sub = group_scaled - group_unscaled * resolution_scale;
  uint source_offset =
      guest_offset_unscaled * (resolution_scale.x * resolution_scale.y) +
      (((((group_sub.x * resolution_scale.y + group_sub.y) << kBlocksPerGroupLog2) +
         (block_index.x & ((1u << kBlocksPerGroupLog2) - 1u))) << kBytesPerBlockLog2));
  uint source_index = (xe_texture_load_guest_offset + source_offset) >> 4u;
  uvec4 blocks0 = XeEndianSwap32(xe_texture_load_source[source_index], endian);

  // The second 16-byte group.
  uint source_step;
  if (resolution_scale.x > 1u) {
    uint group_sub_x = group_scaled.x - (group_scaled.x / resolution_scale.x) * resolution_scale.x;
    if (group_sub_x + 1u == resolution_scale.x) {
      source_step = 32u * resolution_scale.x - (group_sub_x << 4u);
    } else {
      source_step = 16u;
    }
  } else {
    source_step = 32u;
  }
  uvec4 blocks1 = XeEndianSwap32(
      xe_texture_load_source[source_index + ((source_step * resolution_scale.y) >> 4u)], endian);

#ifdef XE_TEXTURE_LOAD_DIRECT_DEPTH_FLOAT
  blocks0 = XeDepthFloat24To32(blocks0);
  blocks1 = XeDepthFloat24To32(blocks1);
#endif

  uint width_blocks = xe_texture_load_size_blocks.x;
  XeStoreGroup(block_index.x, block_index.y, width_blocks, blocks0);
  XeStoreGroup(block_index.x + (1u << kBlocksPerGroupLog2), block_index.y, width_blocks, blocks1);
}
