// wwe2k14 patch (Session 41, FASTCLEAR): the fill of an exact cheap clear (gpu_fast_clear). The clear's own draw has
// just drawn one pixel (its shaders, the EDRAM packing and the depth conversion all as for any pixel); this copies
// that pixel's sample 0 dword to every sample of the clear's rectangle in one surface (colour or depth / stencil),
// keeping the bits the clear does not write (keep_mask: e.g. the stencil byte of a depth-only clear).
//
// The address of a sample, in dwords (spirv_translator_rb.cpp FSI_LoadEdramOffsets, 32bpp): sample coordinates are
// host pixels x the samples per axis (2x MSAA: 2 rows, 4x: 2 x 2); tiles of tile_w x tile_h samples (80 x 16 x the
// resolution scale), row-major at the surface's pitch in tiles, inside a tile row-major in two halves of tile_w / 2
// columns; a depth / stencil surface has the halves swapped (columns 40-79 first); + the surface base, wrapped at the
// EDRAM size.
//   set 0: the EDRAM buffer (storage buffer).
//
// Generated header: py tools\perf\gen_spirv_header.py (glslangValidator -V --target-env vulkan1.0 -S comp), see
// vulkan_spirv/edram_fill_cs.h.

#version 460

// 64 samples of one row per group: a row of a half-tile is contiguous (40 x the scale dwords), so a wave stores whole
// runs of dwords.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer EdramBuffer { uint edram[]; };

layout(push_constant) uniform Constants {
  uint base_dwords;        // surface base (tiles) x tile size (dwords)
  uint pitch_tile_dwords;  // the surface pitch in tiles x tile size (one row of tiles)
  uint tile_w;             // 80 x resolution scale x
  uint tile_h;             // 16 x resolution scale y
  uint x0;                 // the rectangle, in samples (host pixels x samples per axis)
  uint y0;
  uint width;
  uint height;
  uint src_dword;          // the drawn pixel's sample 0 (absolute dword index, wrapped)
  uint keep_mask;          // bits of every dword the clear leaves as they are
  uint wrap_dwords;        // 2048 tiles x tile size
  uint flags;              // bit 0: depth / stencil surface (halves swapped); bit 1: a 64bpp colour surface
} c;

void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= c.width || p.y >= c.height) return;
  uint sx = c.x0 + p.x, sy = c.y0 + p.y;
  uint half_w = c.tile_w >> 1u;
  if ((c.flags & 2u) != 0u) {
    // wwe2k14 patch (Session 45, NATIVE ALL): a 64bpp colour surface (FSI_LoadEdramOffsets): tiles of tile_w / 2 x
    // tile_h samples of two dwords each, rows of tiles at pitch_tile_dwords (the 64bpp pitch); both dwords of the
    // drawn pixel's sample 0 (src_dword, src_dword + 1).
    uint offset64 = (sy / c.tile_h) * c.pitch_tile_dwords + (sx / half_w) * (c.tile_w * c.tile_h) +
                    (sy % c.tile_h) * c.tile_w + (sx % half_w) * 2u;
    uint address64 = (c.base_dwords + offset64) % c.wrap_dwords;
    uint v0 = edram[c.src_dword], v1 = edram[(c.src_dword + 1u) % c.wrap_dwords];
    edram[address64] = v0;
    edram[(address64 + 1u) % c.wrap_dwords] = v1;
    return;
  }
  uint half_index = sx / half_w;
  bool second_half = (half_index & 1u) != 0u;
  bool depth = (c.flags & 1u) != 0u;
  uint offset = (sy / c.tile_h) * c.pitch_tile_dwords + (half_index >> 1u) * (c.tile_w * c.tile_h) +
                (sy % c.tile_h) * c.tile_w + sx % half_w + ((second_half != depth) ? half_w : 0u);
  uint address = (c.base_dwords + offset) % c.wrap_dwords;
  // Every invocation stores the same value: the source dword (inside the rectangle) is rewritten unchanged.
  uint v = edram[c.src_dword];
  if (c.keep_mask == 0u) {
    edram[address] = v;  // the whole dword: a store only
  } else {
    edram[address] = (edram[address] & c.keep_mask) | (v & ~c.keep_mask);
  }
}
