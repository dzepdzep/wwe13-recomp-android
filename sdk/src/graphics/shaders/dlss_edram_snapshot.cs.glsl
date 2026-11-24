// wwe2k14 patch (Session 24, DLSS): a copy of a 32bpp single-sample colour surface straight from the EDRAM buffer
// (render_target_path_vulkan=fsi: the EDRAM buffer is the render target). Used right after the post chain's last
// full-screen pass, before the HUD draws: the frame's picture without the HUD, for DLSS.
//
// The address of host pixel (x, y) of the surface, in dwords (spirv_translator_rb.cpp FSI_LoadEdramOffsets, colour,
// 32bpp, 1 sample): tiles of tile_w x tile_h pixels (80 x 16 samples x the resolution scale), row-major at the
// surface's pitch in tiles; inside a tile row-major; + the surface base, wrapped at the EDRAM size.
//   set 0: the EDRAM buffer (storage buffer);
//   set 1: the output, R32_UINT storage view (the dword as stored: k_8_8_8_8 = R | G << 8 | B << 16 | A << 24).
//
// Generated header: glslangValidator -V --target-env vulkan1.0 -S comp (glslang from the SDK's submodule, built out
// of tree: out\glslang-build), see vulkan_spirv/dlss_edram_snapshot_cs.h.

#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer EdramBuffer { uint edram[]; };
layout(set = 1, binding = 0, r32ui) uniform writeonly uimage2D dest_image;

layout(push_constant) uniform Constants {
  uint base_dwords;        // surface base (tiles) x tile size (dwords)
  uint pitch_tile_dwords;  // the surface pitch in tiles x tile size (one row of tiles)
  uint tile_w;             // 80 x resolution scale x
  uint tile_h;             // 16 x resolution scale y
  uint width;              // host pixels to copy
  uint height;
  uint wrap_dwords;        // 2048 tiles x tile size
  uint flags;              // bit 0: swap the red and blue bytes
} c;

void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= c.width || p.y >= c.height) return;
  uint tile_x = p.x / c.tile_w, tile_y = p.y / c.tile_h;
  uint offset = tile_y * c.pitch_tile_dwords + tile_x * (c.tile_w * c.tile_h) + (p.y % c.tile_h) * c.tile_w +
                p.x % c.tile_w;
  uint v = edram[(c.base_dwords + offset) % c.wrap_dwords];
  if ((c.flags & 1u) != 0u) v = (v & 0xFF00FF00u) | ((v >> 16) & 0xFFu) | ((v & 0xFFu) << 16);
  imageStore(dest_image, ivec2(p), uvec4(v, 0u, 0u, 0u));
}
