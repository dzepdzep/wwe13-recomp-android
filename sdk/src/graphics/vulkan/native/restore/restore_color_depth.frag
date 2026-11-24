#version 450
// wwe2k14 patch (Session 87, stage 4): the colour + depth restore (the game's yPostBase, guest pixel shader
// 9A836B68E3F8F0B0, 21 dwords: colour target 0 = texture fetch constant 1 at the first interpolator's xy, the fragment
// depth = fetch constant 0's x, saturated) as our own fragment shader, the last pixel shader of the bench frames that was
// still the SDK translator's. Our own code, written from the shader's microcode listing and checked against the
// translator's NATIVE module (tools\restore_check): the same operations in the same order - the fetch constant's size
// nudge on the coordinate, the LOD bias and the exponent adjust read from the fetch constant as the translator reads
// them, NClamp for the depth, then the colour epilogue (src\scene\xenos_ps.glsl's, copied below so the SDK does not
// depend on the game's folders).
//
// Registered (RegisterNativeRestoreShader with colour fetch constant 1, depth 0) as a shader swap with kFlagOwnEpilogue
// when gpu_native_restore_shader is on. SPIR-V: restore_color_depth.h (glslangValidator -V --target-env vulkan1.0 -S
// frag); regenerate with tools\restore_check\build.ps1.

layout(set = 1, binding = 0, std140) uniform XeSystemConstants {
  layout(offset = 0) uint flags;  // bits 16-18: alpha test pass if less / equal / greater; bit 19: colour 0 to gamma
  layout(offset = 272) float alpha_test_reference;
  layout(offset = 288) vec4 color_exp_bias;
} xe_system;

layout(set = 1, binding = 4, std140) uniform XeFetchConstants { uvec4 fetch_constants[48]; } xe_fetch;

layout(set = 3, binding = 1000) uniform texture2DArray t_depth;  // fetch constant 0 (x = the depth)
layout(set = 3, binding = 1001) uniform texture2DArray t_color;  // fetch constant 1
layout(set = 3, binding = 1200) uniform sampler s_depth;
layout(set = 3, binding = 1201) uniform sampler s_color;

layout(location = 0) in vec4 in_uv;  // xy: the coordinate of both textures
layout(location = 0) out vec4 out_color;

layout(constant_id = 0) const uint kXeColorQuantize = 0u;

vec4 XeQuantize(vec4 c) {
  if (kXeColorQuantize != 0u) c = mix(c, vec4(0.0), isnan(c));
  if (kXeColorQuantize == 1u) {
    precise vec4 scaled = clamp(c, 0.0, 1.0) * 255.0;
    precise vec4 level = trunc(scaled + 0.5);
    return level / 255.0;
  }
  if (kXeColorQuantize == 2u) {
    precise vec4 scaled = clamp(c, -32.0, 32.0) * (32767.0 / 32.0);
    precise vec4 level = trunc(scaled + mix(vec4(0.5), vec4(-0.5), lessThan(scaled, vec4(0.0))));
    return level / 32767.0;
  }
  return c;
}

vec3 XeLinearToPWLGamma(vec3 x) {
  x = clamp(x, 0.0, 1.0);
  bvec3 p3 = greaterThanEqual(x, vec3(512.0 / 1023.0));
  bvec3 p2 = greaterThanEqual(x, vec3(128.0 / 1023.0));
  bvec3 p1 = greaterThanEqual(x, vec3(64.0 / 1023.0));
  vec3 scale = mix(mix(vec3(1023.0), vec3(1023.0 / 2.0), p1), mix(vec3(1023.0 / 4.0), vec3(1023.0 / 8.0), p3), p2);
  vec3 offset = mix(mix(vec3(0.0), vec3(32.0 / 255.0), p1), mix(vec3(64.0 / 255.0), vec3(128.0 / 255.0), p3), p2);
  precise vec3 scaled = x * scale;
  precise vec3 gamma = trunc(scaled) * (1.0 / 255.0) + offset;
  return gamma;
}

void XeOutputColor(vec4 color) {
  uint test = bitfieldExtract(xe_system.flags, 16, 3);
  if (test != 7u) {
    float a = color.a, ref = xe_system.alpha_test_reference;
    bool pass = test == 5u ? !(a == ref)
                           : ((test & 1u) != 0u && a < ref) || ((test & 2u) != 0u && a == ref) ||
                                 ((test & 4u) != 0u && a > ref);
    if (!pass) discard;
  }
  precise vec4 biased = color * xe_system.color_exp_bias.x;
  if ((xe_system.flags & (1u << 19)) != 0u) biased.rgb = XeLinearToPWLGamma(biased.rgb);
  out_color = XeQuantize(biased);
}

void main() {
  // Fetch constant 1 (the colour): its size is dword 2 (fetch_constants[2].x), its LOD bias and exponent adjust dword 4
  // (fetch_constants[2].z). Fetch constant 0 (the depth): size fetch_constants[0].z, bias / adjust fetch_constants[1].x.
  precise float color_nudge_x = 0.00146484375 / float(bitfieldExtract(xe_fetch.fetch_constants[2].x, 0, 13) + 1u);
  precise float color_u = in_uv.x + color_nudge_x;
  precise float color_nudge_y = 0.00146484375 / float(bitfieldExtract(xe_fetch.fetch_constants[2].x, 13, 13) + 1u);
  precise float color_v = in_uv.y + color_nudge_y;
  int color_word = int(xe_fetch.fetch_constants[2].z);
  precise float color_bias = float(bitfieldExtract(color_word, 12, 10)) * 0.03125;
  vec4 sampled = texture(sampler2DArray(t_color, s_color), vec3(color_u, color_v, 0.0), color_bias);
  float color_scale = ldexp(1.0, bitfieldExtract(color_word, 13, 6));
  precise vec4 color = sampled * color_scale;

  precise float depth_nudge_x = 0.00146484375 / float(bitfieldExtract(xe_fetch.fetch_constants[0].z, 0, 13) + 1u);
  precise float depth_u = in_uv.x + depth_nudge_x;
  precise float depth_nudge_y = 0.00146484375 / float(bitfieldExtract(xe_fetch.fetch_constants[0].z, 13, 13) + 1u);
  precise float depth_v = in_uv.y + depth_nudge_y;
  int depth_word = int(xe_fetch.fetch_constants[1].x);
  precise float depth_bias = float(bitfieldExtract(depth_word, 12, 10)) * 0.03125;
  precise float depth =
      texture(sampler2DArray(t_depth, s_depth), vec3(depth_u, depth_v, 0.0), depth_bias).x * ldexp(1.0, bitfieldExtract(depth_word, 13, 6));
  gl_FragDepth = isnan(depth) ? 0.0 : clamp(depth, 0.0, 1.0);  // (the translator's NClamp: a NaN gives the low bound)
  XeOutputColor(color);
}
