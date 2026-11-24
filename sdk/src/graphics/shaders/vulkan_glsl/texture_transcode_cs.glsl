// wwe2k14 patch (Session 93, ASTC / EAC transcode): texture load shaders for GPUs without BC support. They do what
// texture_load_64bpb / texture_load_128bpb do (endian swap, untiling into the scratch buffer, the same push constants and
// the same two descriptor sets) and, on the way, re-encode the BC blocks into a block format that Mali / Adreno /
// PowerVR do support:
//   TRANSCODE_BC1_ASTC   BC1 (DXT1)        -> ASTC 4x4 LDR  (128-bit blocks)  4 guest blocks per invocation
//   TRANSCODE_BC4_EAC    BC4 (DXT5A)       -> EAC R11       (64-bit blocks)   4 guest blocks per invocation
//   TRANSCODE_BC5_EAC    BC5 (DXN)         -> EAC RG11      (128-bit blocks)  2 guest blocks per invocation
// Test-only variants (tools/native/compute_test/transcode_test.py): COPY64 / COPY128 = the plain copy (proves the
// addressing against the shipped shaders), TEST_ONE_SET = both buffers in set 0 (the offline runner has one set; the
// game binds the destination at set 0 and the source at set 1, exactly like the shipped load shaders).
// Written from the Khronos Data Format Specification (EAC alpha / Unsigned R11 EAC, and the ASTC chapter: block mode,
// color endpoint modes 8 and 12, integer sequence encoding, endpoint and weight unquantization, dual plane decoding);
// no third-party encoder code. The untiling arithmetic is that of the shipped 64bpb / 128bpb shaders (read from their
// SPIR-V disassembly in vulkan_spirv/).
#version 450

// wwe2k14 patch (Session 95, BC1_EMU_BC3): a PC test of what the ASTC transcode does to a 3-colour BC1 block. A PC has no ASTC, so this variant of
// TRANSCODE_BC1_ASTC writes the same texels as a BC3 block (16 bytes, the same load path): RGB from the block's own colours, the transparent texels'
// RGB as the transcode gives them (the first endpoint, or with BC1_ALPHA_FIX the level nearest the opaque mean), alpha 0 there. Middle colour approximate (BC3 has
// the 2/3 levels only).
#if defined(TRANSCODE_BC1_ASTC) || defined(COPY64) || defined(TRANSCODE_BC4_EAC)
#define SRC_BPB_LOG2 3
#define XSHIFT 2
#else
#define SRC_BPB_LOG2 4
#define XSHIFT 1
#endif
#if defined(TRANSCODE_BC1_ASTC)
#define DST_BLOCK_BYTES 16
#elif defined(TRANSCODE_BC4_EAC)
#define DST_BLOCK_BYTES 8
#elif defined(TRANSCODE_BC5_EAC)
#define DST_BLOCK_BYTES 16
#elif defined(COPY64)
#define DST_BLOCK_BYTES 8
#else
#define DST_BLOCK_BYTES 16
#endif

layout(local_size_x = 4, local_size_y = 32, local_size_z = 1) in;

layout(push_constant) uniform Constants {
  uint is_tiled_3d_endian_scale;
  uint guest_offset;
  uint guest_pitch_aligned;
  uint guest_z_stride_block_rows_aligned;
  uvec3 size_blocks;
  uint host_offset;
  uint host_pitch;
  uint height_texels;
} pc;

#ifdef TEST_ONE_SET
layout(set = 0, binding = 0) buffer Dest { uvec4 dst[]; };
layout(set = 0, binding = 1) readonly buffer Source { uvec4 src[]; };
#else
layout(set = 0, binding = 0) writeonly buffer Dest { uvec4 dst[]; };
layout(set = 1, binding = 0) readonly buffer Source { uvec4 src[]; };
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Guest addressing (same arithmetic as the shipped load shaders; B = log2 of the guest block size in bytes).
// ---------------------------------------------------------------------------------------------------------------------

int TiledOffset2D(int x, int y, uint pitch_blocks_aligned) {
  const int B = SRC_BPB_LOG2;
  int macro = ((x >> 5) + (y >> 5) * int(pitch_blocks_aligned >> 5)) << (B + 7);
  int micro = ((x & 7) + ((y & 14) << 2)) << B;
  int o = macro + ((micro & -16) << 1) + (micro & 15) + ((y & 1) << 4);
  return ((o & -512) << 3) + ((y & 16) << 7) + ((o & 448) << 2) +
         ((((y & 8) >> 2) + (x >> 3)) & 3) * 64 + (o & 63);
}

int TiledOffset3D(int x, int y, int z, uint pitch_blocks_aligned, uint z_stride_rows_aligned) {
  const int B = SRC_BPB_LOG2;
  int zq = z >> 2;
  int macro_rows = (y >> 4) + zq * int(z_stride_rows_aligned >> 4);
  int macro = ((x >> 5) + macro_rows * int(pitch_blocks_aligned >> 5));
  int m1 = ((macro << (B + 6)) & 0x0FFFFFFF) << 1;
  int micro = ((((x & 7) + ((y & 6) << 2))) << (B + 6)) >> 6;
  int bank = ((y >> 3) + zq) & 1;
  int sel = bank + ((((x >> 3) + (bank << 1)) & 3) << 1);
  int o = (((m1 + (micro & -16)) << 1) + (micro & 15)) + ((z & 3) << (B + 6)) + ((y & 1) << 4);
  int s = ((((sel & 1) << 3) + ((o >> 6) & 7)) << 3) + (sel & -2);
  return (((s << 2) + (o & -512)) << 3) + (o & 63);
}

// ---------------------------------------------------------------------------------------------------------------------
uvec4 EndianSwap(uvec4 v, uint endian) {
  if (endian == 1u || endian == 2u) {
    v = ((v & 0x00FF00FFu) << 8) | ((v & 0xFF00FF00u) >> 8);
  }
  if (endian == 2u || endian == 3u) {
    v = (v << 16) | (v >> 16);
  }
  return v;
}

void PutBits(inout uint blk[4], uint pos, uint n, uint v) {
  uint w = pos >> 5;
  uint o = pos & 31u;
  blk[w] |= v << o;
  if (o + n > 32u) {
    blk[w + 1u] |= v >> (32u - o);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// BC1 -> ASTC 4x4 LDR.
// ---------------------------------------------------------------------------------------------------------------------
#ifdef TRANSCODE_BC1_ASTC

const uint kTritTable[61] = uint[61](
0x04020100u, 0x09080605u, 0x1211100Au, 0x18161514u, 0x07031A19u, 0x1B17130Bu,
0x200E0D0Cu, 0x25242221u, 0x2A292826u, 0x34323130u, 0x39383635u, 0x2B27233Au,
0x2C3B3733u, 0x41402E2Du, 0x46454442u, 0x504A4948u, 0x55545251u, 0x5A595856u,
0x534B4743u, 0x4D4C5B57u, 0x8281804Eu, 0x88868584u, 0x91908A89u, 0x96959492u,
0x839A9998u, 0x97938B87u, 0x8E8D8C9Bu, 0xA4A2A1A0u, 0xA9A8A6A5u, 0xB2B1B0AAu,
0xB8B6B5B4u, 0xA7A3BAB9u, 0xBBB7B3ABu, 0xC0AEADACu, 0xC5C4C2C1u, 0xCAC9C8C6u,
0xD4D2D1D0u, 0xD9D8D6D5u, 0xCBC7C3DAu, 0xCCDBD7D3u, 0x6160CECDu, 0x66656462u,
0x706A6968u, 0x75747271u, 0x7A797876u, 0x736B6763u, 0x6D6C7B77u, 0xE2E1E06Eu,
0xE8E6E5E4u, 0xF1F0EAE9u, 0xF6F5F4F2u, 0xE3FAF9F8u, 0xF7F3EBE7u, 0xEEEDECFBu,
0x3C1E1D1Cu, 0x5D5C3E3Du, 0x9E9D9C5Eu, 0xDCBEBDBCu, 0x3F1FDEDDu, 0xDFBF9F5Fu,
0x007E7D7Cu
);

uint TritCode(uint v) {
  return (kTritTable[v >> 2] >> ((v & 3u) * 8u)) & 255u;
}

uvec3 Expand565(uint c) {
  uint r = (c >> 11) & 31u, g = (c >> 5) & 63u, b = c & 31u;
  return uvec3((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2));
}

uint Unq7(uint q) { return (q << 1) | (q >> 6); }

// The 7-bit value whose unquantized 8-bit value is nearest to v.
uint Quant7(uint v) {
  uint q = min((v + 1u) >> 1, 127u);
  uint best = q;
  uint best_err = uint(abs(int(Unq7(q)) - int(v)));
  if (q > 0u) {
    uint e = uint(abs(int(Unq7(q - 1u)) - int(v)));
    if (e < best_err) { best = q - 1u; best_err = e; }
  }
  if (q < 127u) {
    uint e = uint(abs(int(Unq7(q + 1u)) - int(v)));
    if (e < best_err) { best = q + 1u; best_err = e; }
  }
  return best;
}

// Block mode 0x042: 4x4 weight grid, weight range 0..3 (2 bits), single plane. CEM 8 (RGB direct), 8-bit endpoints.
// Block mode 0x451: 4x4 grid, weight range 0..2 (one trit), DUAL plane (plane 0 = RGB, plane 1 = A, CCS = 3).
//   CEM 12 (RGBA direct), 7-bit endpoints (the largest range that fits: 128 - 19 config - 52 weight bits = 57 bits).
#ifdef BC1_EMU_BC3
uvec4 TranscodeBC1(uint w0, uint w1) {
  uint c0 = w0 & 0xFFFFu;
  uint c1 = w0 >> 16;
  if (c0 > c1) {
    return uvec4(0x0000FFFFu, 0u, w0, w1);  // alpha 255 everywhere (a0 = a1 = 255), the colour block as it is
  }
  uint tc = 0u;  // BC3 colour index of a transparent texel: 0 = c0
#ifdef BC1_ALPHA_FIX
  uint osum = 0u;
  uint ocnt = 0u;
  for (uint t = 0u; t < 16u; ++t) {
    uint idx = (w1 >> (2u * t)) & 3u;
    if (idx != 3u) {
      osum += idx == 1u ? 2u : (idx == 2u ? 1u : 0u);
      ocnt += 1u;
    }
  }
  uint level = ocnt == 0u ? 1u : min((2u * osum + ocnt) / (2u * ocnt), 2u);
  tc = level == 2u ? 1u : (level == 1u ? 2u : 0u);
#endif
  uint colour = 0u;
  uint alo = 0u;  // alpha index bits 0..31 (3 bits a texel)
  uint ahi = 0u;  // bits 32..47
  for (uint t = 0u; t < 16u; ++t) {
    uint idx = (w1 >> (2u * t)) & 3u;
    bool transparent = idx == 3u;
    colour |= (transparent ? tc : idx) << (2u * t);
    uint a = transparent ? 1u : 0u;  // a0 = 255 (index 0), a1 = 0 (index 1)
    uint pos = 3u * t;
    if (pos < 32u) { alo |= a << pos; }
    if (pos >= 32u) { ahi |= a << (pos - 32u); }
  }
  // a0 > a1 = the eight-level mode: 0 and 1 are a0 and a1.
  return uvec4(0x000000FFu | (alo << 16), (alo >> 16) | (ahi << 16), w0, colour);
}
#else
uvec4 TranscodeBC1(uint w0, uint w1) {
  uint c0 = w0 & 0xFFFFu;
  uint c1 = w0 >> 16;
  uvec3 e0 = Expand565(c0);
  uvec3 e1 = Expand565(c1);
  uint blk[4] = uint[4](0u, 0u, 0u, 0u);
  if (c0 > c1) {
    // Opaque four-colour block: colours 0, 1, 2/3 c0 + 1/3 c1, 1/3 c0 + 2/3 c1. The ASTC weights 0, 21, 43, 64 (of 64)
    // are the nearest to 0, 1/3, 2/3, 1 (max error 1.33 of 255 at a full-range pair). CEM 8 needs sum(e1) >= sum(e0),
    // else the decoder applies blue contraction: swap the endpoints and mirror the weights.
    bool swp = (e0.x + e0.y + e0.z) > (e1.x + e1.y + e1.z);
    uvec3 lo = swp ? e1 : e0;
    uvec3 hi = swp ? e0 : e1;
    blk[0] = 0x042u | (8u << 13);
    PutBits(blk, 17u, 8u, lo.x);
    PutBits(blk, 25u, 8u, hi.x);
    PutBits(blk, 33u, 8u, lo.y);
    PutBits(blk, 41u, 8u, hi.y);
    PutBits(blk, 49u, 8u, lo.z);
    PutBits(blk, 57u, 8u, hi.z);
    uint wbits = 0u;
    for (uint t = 0u; t < 16u; ++t) {
      uint idx = (w1 >> (2u * t)) & 3u;
      uint q = idx == 0u ? 0u : (idx == 1u ? 3u : (idx == 2u ? 1u : 2u));
      if (swp) { q = 3u - q; }
      wbits |= q << (2u * t);
    }
    blk[3] = bitfieldReverse(wbits);  // weight bit k goes to block bit 127 - k
  } else {
    // Three colours + transparent black (c0 <= c1): colours c0, c1, (c0 + c1) / 2, alpha 0 (the colour of a
    // transparent texel is not kept: it takes the colour of weight 0). Dual plane: plane 0 trit 0 / 1 / 2 =
    // E0 / middle / E1 (weights 0 / 32 / 64), plane 1 trit 0 / 2 = alpha 0 / 255 (alpha endpoints 0 and 127 -> 255).
    uvec3 q0 = uvec3(Quant7(e0.x), Quant7(e0.y), Quant7(e0.z));
    uvec3 q1 = uvec3(Quant7(e1.x), Quant7(e1.y), Quant7(e1.z));
    uint s0 = Unq7(q0.x) + Unq7(q0.y) + Unq7(q0.z);
    uint s1 = Unq7(q1.x) + Unq7(q1.y) + Unq7(q1.z);
    bool swp = s0 > s1;
    uvec3 lo = swp ? q1 : q0;
    uvec3 hi = swp ? q0 : q1;
    blk[0] = 0x451u | (12u << 13);
    PutBits(blk, 17u, 7u, lo.x);
    PutBits(blk, 24u, 7u, hi.x);
    PutBits(blk, 31u, 7u, lo.y);
    PutBits(blk, 38u, 7u, hi.y);
    PutBits(blk, 45u, 7u, lo.z);
    PutBits(blk, 52u, 7u, hi.z);
    PutBits(blk, 59u, 7u, 0u);
    PutBits(blk, 66u, 7u, 127u);
    uint trit[32];
    // Session 94 (BC1_ALPHA_FIX): a transparent texel (alpha 0) takes the colour level nearest the mean of the block's
    // opaque texels instead of weight 0, so bilinear / premultiplied edges do not pull the darker endpoint in.
    uint tc = 1u;
#ifdef BC1_ALPHA_FIX
    {
      uint osum = 0u;
      uint ocnt = 0u;
      for (uint t = 0u; t < 16u; ++t) {
        uint idx = (w1 >> (2u * t)) & 3u;
        if (idx != 3u) {
          uint c = idx == 1u ? 2u : (idx == 2u ? 1u : 0u);
          osum += swp ? 2u - c : c;
          ocnt += 1u;
        }
      }
      tc = ocnt == 0u ? 1u : min((2u * osum + ocnt) / (2u * ocnt), 2u);
    }
#else
    tc = swp ? 2u : 0u;  // identical to the Session 93 output (weight 0 = guest endpoint c0, mirrored when swapped)
#endif
    for (uint t = 0u; t < 16u; ++t) {
      uint idx = (w1 >> (2u * t)) & 3u;
      uint c = idx == 1u ? 2u : (idx == 2u ? 1u : 0u);
      if (swp) { c = 2u - c; }
      if (idx == 3u) { c = tc; }
      trit[2u * t] = c;
      trit[2u * t + 1u] = idx == 3u ? 0u : 2u;
    }
    uint slo = 0u;
    uint shi = 0u;
    for (uint g = 0u; g < 7u; ++g) {
      uint v;
      uint nbits;
      if (g < 6u) {
        v = trit[5u * g] + 3u * trit[5u * g + 1u] + 9u * trit[5u * g + 2u] + 27u * trit[5u * g + 3u] +
            81u * trit[5u * g + 4u];
        nbits = 8u;
      } else {
        v = trit[30] + 3u * trit[31];
        nbits = 4u;
      }
      uint code = TritCode(v) & ((1u << nbits) - 1u);
      uint pos = 8u * g;
      if (pos < 32u) {
        slo |= code << pos;
      } else {
        shi |= code << (pos - 32u);
      }
    }
    blk[3] = bitfieldReverse(slo);
    blk[2] |= bitfieldReverse(shi) | (3u << 10);  // weight bits 32..51 -> block bits 95..76, CCS (3) at bits 75:74
  }
  return uvec4(blk[0], blk[1], blk[2], blk[3]);
}
#endif  // BC1_EMU_BC3
#endif

// ---------------------------------------------------------------------------------------------------------------------
// BC4 -> EAC R11 (one 64-bit block in, one out). EAC has only 16 fixed modifier tables, so the 8 evenly spaced BC4
// levels cannot be copied: a small search picks (table, multiplier, base) per block for the levels the block really uses.
// ---------------------------------------------------------------------------------------------------------------------
#if defined(TRANSCODE_BC4_EAC) || defined(TRANSCODE_BC5_EAC)

const int kEacMod[128] = int[128](
    -3, -6, -9, -15, 2, 5, 8, 14,
    -3, -7, -10, -13, 2, 6, 9, 12,
    -2, -5, -8, -13, 1, 4, 7, 12,
    -2, -4, -6, -13, 1, 3, 5, 12,
    -3, -6, -8, -12, 2, 5, 7, 11,
    -3, -7, -9, -11, 2, 6, 8, 10,
    -4, -7, -8, -11, 3, 6, 7, 10,
    -3, -5, -8, -11, 2, 4, 7, 10,
    -2, -6, -8, -10, 1, 5, 7, 9,
    -2, -5, -8, -10, 1, 4, 7, 9,
    -2, -4, -8, -10, 1, 3, 7, 9,
    -2, -5, -7, -10, 1, 4, 6, 9,
    -3, -4, -7, -10, 2, 3, 6, 9,
    -1, -2, -3, -10, 0, 1, 2, 9,
    -4, -6, -8, -9, 3, 5, 7, 8,
    -3, -5, -7, -9, 2, 4, 6, 8);

uint BC4Index(uint w0, uint w1, uint t) {
  uint p = 3u * t;
  if (p + 3u <= 16u) { return (w0 >> (16u + p)) & 7u; }
  if (p >= 16u) { return (w1 >> (p - 16u)) & 7u; }
  return ((w0 >> 31) | ((w1 & 3u) << 1)) & 7u;  // t == 5 straddles the two words
}

uint ByteSwap32(uint v) {
  return ((v & 0xFFu) << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
}

// Returns the EAC block as two memory-order dwords (the 64-bit big-endian word of the spec, bytes q0..q7).
uvec2 TranscodeBC4(uint w0, uint w1) {
  float r0 = float(w0 & 255u);
  float r1 = float((w0 >> 8) & 255u);
  float lv[8];
  lv[0] = r0;
  lv[1] = r1;
  if (r0 > r1) {
    for (int k = 2; k < 8; ++k) { lv[k] = (float(8 - k) * r0 + float(k - 1) * r1) / 7.0; }
  } else {
    for (int k = 2; k < 6; ++k) { lv[k] = (float(6 - k) * r0 + float(k - 1) * r1) / 5.0; }
    lv[6] = 0.0;
    lv[7] = 255.0;
  }
  uint bidx[16];
  uint cnt[8] = uint[8](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  for (uint t = 0u; t < 16u; ++t) {
    uint i = BC4Index(w0, w1, t);
    bidx[t] = i;
    cnt[i] += 1u;
  }
  // Targets in the 11-bit domain of EAC (sampled as value / 2047, BC4 as value / 255).
  float tg[8];
  float tmin = 1e9;
  float tmax = -1e9;
  for (int i = 0; i < 8; ++i) {
    tg[i] = lv[i] * (2047.0 / 255.0);
    if (cnt[i] != 0u) { tmin = min(tmin, tg[i]); tmax = max(tmax, tg[i]); }
  }
  float span = tmax - tmin;
  float best_err = 1e30;
  int best_table = 13;
  int best_mult = 0;
  int best_base = 0;
  // For each table and each multiplier from 0.45x to 1.6x of the one that spans the block's range (plus the multiplier 0 =
  // step 1 for tiny ranges), three starting bases (centred on the used levels, lowest level on the table's lowest entry,
  // highest on its highest), each refined up to twice (every used level moves to its nearest table level, the base moves
  // by half the mean plus a quarter of the min+max residual; levels clamped to 0 / 2047 do not vote). The error is the
  // sum of squares plus 400x the worst squared error. The evenly spaced BC4 levels match no table, so this is a fit, not a
  // copy: up to 16 tables x ~8 multipliers x 3 bases x 3 passes (~1000 evaluations of at most 8 x 8 comparisons) per
  // block; offline max error ~13 (the exhaustive minimax over 16 x 16 x 256 reaches ~12.2 on the worst blocks).
  for (int table = 0; table < 16; ++table) {
    int mmin = kEacMod[table * 8 + 3];
    int mmax = kEacMod[table * 8 + 7];
    float mc = span / (8.0 * float(mmax - mmin));
    int mlo = span <= 48.0 ? 0 : max(int(floor(mc * 0.45)), 1);
    int mhi = min(int(ceil(mc * 1.6)), 15);
    for (int m = mlo; m <= mhi; ++m) {
      float step = m == 0 ? 1.0 : 8.0 * float(m);
      float center = 0.5 * (tmin + tmax) - 0.5 * step * float(mmin + mmax);
      for (int sb = 0; sb < 3; ++sb) {
      float c0 = sb == 0 ? center : (sb == 1 ? tmin - step * float(mmin) : tmax - step * float(mmax));
      int base = clamp(int(floor((c0 - 4.0) / 8.0 + 0.5)), 0, 255);
      for (int it = 0; it < 3; ++it) {
        float B = float(base * 8 + 4);
        float err = 0.0;
        float maxd = 0.0;
        float rsum = 0.0;
        float rcnt = 0.0;
        float rlo = 1e9;
        float rhi = -1e9;
        for (int i = 0; i < 8; ++i) {
          if (cnt[i] == 0u) { continue; }
          float bd = 1e30;
          float resid = 0.0;
          bool clamped = true;
          for (int j = 0; j < 8; ++j) {
            float raw = B + step * float(kEacMod[table * 8 + j]);
            float d = clamp(raw, 0.0, 2047.0) - tg[i];
            if (d * d < bd) {
              bd = d * d;
              resid = tg[i] - raw;
              clamped = raw <= 0.0 || raw >= 2047.0;
            }
          }
          err += float(cnt[i]) * bd;
          maxd = max(maxd, bd);
          if (!clamped) {
            rsum += float(cnt[i]) * resid;
            rcnt += float(cnt[i]);
            rlo = min(rlo, resid);
            rhi = max(rhi, resid);
          }
        }
        err += 400.0 * maxd;  // Session 94: favour a smaller worst-case texel error (a level 11-bit units off shows as a band)
        if (err < best_err) {
          best_err = err;
          best_table = table;
          best_mult = m;
          best_base = base;
        }
        if (it == 2 || rcnt == 0.0) { break; }
        int next_base = clamp(int(floor((B + 0.5 * (rsum / rcnt + 0.5 * (rlo + rhi)) - 4.0) / 8.0 + 0.5)), 0, 255);
        if (next_base == base) { break; }
        base = next_base;
      }
      }
    }
  }
  float step = best_mult == 0 ? 1.0 : 8.0 * float(best_mult);
  float B = float(best_base * 8 + 4);
  uint emap[8];
  for (int i = 0; i < 8; ++i) {
    float bd = 1e30;
    uint bj = 0u;
    for (int j = 0; j < 8; ++j) {
      float d = abs(clamp(B + step * float(kEacMod[best_table * 8 + j]), 0.0, 2047.0) - tg[i]);
      if (d < bd) { bd = d; bj = uint(j); }
    }
    emap[i] = bj;
  }
  uint flo = 0u;  // bits 31..0 of the 48 index bits
  uint fhi = 0u;  // bits 47..32
  for (uint t = 0u; t < 16u; ++t) {
    uint x = t & 3u;
    uint y = t >> 2;
    uint k = x * 4u + y;  // EAC pixels are numbered column by column (a, b, c, d = first column)
    uint pos = 45u - 3u * k;
    uint v = emap[bidx[t]];
    if (pos < 32u) { flo |= v << pos; }
    if (pos + 3u > 32u) {
      fhi |= pos >= 32u ? (v << (pos - 32u)) : (v >> (32u - pos));
    }
  }
  uint hi32 = (uint(best_base) << 24) | (uint(best_mult) << 20) | (uint(best_table) << 16) | fhi;
  return uvec2(ByteSwap32(hi32), ByteSwap32(flo));
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
void main() {
  uvec3 g = gl_GlobalInvocationID;
  uint endian = (pc.is_tiled_3d_endian_scale >> 2) & 3u;
  bool is_tiled = (pc.is_tiled_3d_endian_scale & 1u) != 0u;
  bool is_3d = (pc.is_tiled_3d_endian_scale & 2u) != 0u;
  uvec3 gs = uvec3(g.x << XSHIFT, g.y, g.z);
  if (any(greaterThanEqual(gs.xy, pc.size_blocks.xy))) {
    return;
  }
  ivec3 p = ivec3(gs);
  int host = p.x * DST_BLOCK_BYTES + (p.z * int(pc.size_blocks.y) + p.y) * int(pc.host_pitch);
  uint di = (uint(host) + pc.host_offset) >> 4;
  int off;
  if (is_tiled) {
    off = is_3d ? TiledOffset3D(p.x, p.y, p.z, pc.guest_pitch_aligned, pc.guest_z_stride_block_rows_aligned)
                : TiledOffset2D(p.x, p.y, pc.guest_pitch_aligned);
  } else {
    off = (p.z * int(pc.guest_z_stride_block_rows_aligned) + p.y) * int(pc.guest_pitch_aligned) +
          (p.x << SRC_BPB_LOG2);
  }
  uint si = uint(int(pc.guest_offset) + off) >> 4;
  uvec4 a = EndianSwap(src[si], endian);
  uvec4 b = EndianSwap(src[si + (is_tiled ? 2u : 1u)], endian);
#if defined(COPY64) || defined(COPY128)
  dst[di] = a;
  dst[di + 1u] = b;
#elif defined(TRANSCODE_BC1_ASTC)
  dst[di] = TranscodeBC1(a.x, a.y);
  dst[di + 1u] = TranscodeBC1(a.z, a.w);
  dst[di + 2u] = TranscodeBC1(b.x, b.y);
  dst[di + 3u] = TranscodeBC1(b.z, b.w);
#elif defined(TRANSCODE_BC4_EAC)
  uvec2 e0 = TranscodeBC4(a.x, a.y);
  uvec2 e1 = TranscodeBC4(a.z, a.w);
  uvec2 e2 = TranscodeBC4(b.x, b.y);
  uvec2 e3 = TranscodeBC4(b.z, b.w);
  dst[di] = uvec4(e0, e1);
  dst[di + 1u] = uvec4(e2, e3);
#elif defined(TRANSCODE_BC5_EAC)
  dst[di] = uvec4(TranscodeBC4(a.x, a.y), TranscodeBC4(a.z, a.w));
  dst[di + 1u] = uvec4(TranscodeBC4(b.x, b.y), TranscodeBC4(b.z, b.w));
#endif
}
