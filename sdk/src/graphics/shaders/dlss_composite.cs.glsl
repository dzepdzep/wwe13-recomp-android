// wwe2k14 patch (Session 24, DLSS): the last pass of a DLSS frame at the swap. Writes the presenter's guest output
// image (output size) from:
//   set 0: the guest output image (storage, A2B10G10R10);
//   set 1: DLSS's output (output size);
//   set 2: the frame's final picture after the swap's gamma ramp (render size: what DLSS would otherwise get);
//   set 3: the frame's picture before the HUD, after the same gamma ramp (render size; = set 2 without a HUD image).
// Modes:
//   0: plain bilinear upscale of the final picture (the comparison "without DLSS", and frames DLSS can't take);
//   1: Catmull-Rom (bicubic) upscale of the final picture;
//   2: DLSS's output as is;
//   (2 and 3 with bias > 0: inside the character mask in set 5, blended toward the current frame's Catmull-Rom
//    upscale: the manual "bias current colour", Session 24 rung 2a)
//   3: DLSS's output + the HUD: the difference between the final picture and the pre-HUD picture, both upscaled with
//      Catmull-Rom (0 where there is no HUD, so the scene is pure DLSS; the HUD never goes through DLSS's history);
//   4: debug: the HUD difference as grey (black = no HUD) over a dimmed bilinear picture;
//   5: debug: the motion vectors DLSS gets (render pixels; 0.5 grey = still, red / green = x / y, hud_gain px =
//      full colour);
//   6: debug: the depth DLSS gets (the reversed hardware depth: white = near);
//   7: debug: the character mask (set 5 holds it then) in red over the dimmed picture.
//
// Generated header: glslangValidator -V --target-env vulkan1.0 -S comp (glslang from the SDK's submodule, built out
// of tree: out\glslang-build), see vulkan_spirv/dlss_composite_cs.h.

#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgb10_a2) uniform writeonly image2D out_image;
layout(set = 1, binding = 0) uniform sampler2D dlss_image;
layout(set = 2, binding = 0) uniform sampler2D final_image;
layout(set = 3, binding = 0) uniform sampler2D prehud_image;
layout(set = 4, binding = 0) uniform sampler2D motion_image;  // render size, RG16F (debug views)
layout(set = 5, binding = 0) uniform sampler2D depth_image;   // render size, R32F (debug views)

layout(push_constant) uniform Constants {
  uvec2 out_size;   // output pixels
  uvec2 low_size;   // render pixels (the final / pre-HUD images; may be smaller than their allocation)
  vec2 low_scale;   // low_size / the final image's allocated size (UV scale)
  vec2 pre_scale;   // low_size / the pre-HUD image's allocated size (UV scale)
  uint mode;
  float hud_gain;   // multiplies the HUD difference (1 = as drawn)
  float bias;       // modes 2 / 3 (Session 24 rung 2a): blend toward this frame's own upscale by mask x bias
  float pad;
  vec2 motion_bias; // modes 2 / 3: also blend where |motion| (render px) goes from .x to .y (0, 0 = off)
} c;

// Catmull-Rom from 16 texel fetches; uv in render pixels (texel centres at +0.5).
vec3 Cubic(sampler2D s, vec2 p, uvec2 size) {
  vec2 t = p - 0.5;
  vec2 f = fract(t);
  ivec2 b = ivec2(floor(t)) - 1;
  vec2 f2 = f * f, f3 = f2 * f;
  vec2 w0 = -0.5 * f3 + f2 - 0.5 * f;
  vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
  vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  vec2 w3 = 0.5 * f3 - 0.5 * f2;
  vec2 w[4] = vec2[4](w0, w1, w2, w3);
  ivec2 hi = ivec2(size) - 1;
  vec3 sum = vec3(0.0);
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      ivec2 q = clamp(b + ivec2(i, j), ivec2(0), hi);
      sum += texelFetch(s, q, 0).rgb * (w[i].x * w[j].y);
    }
  }
  return sum;
}

void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (any(greaterThanEqual(p, c.out_size))) return;
  vec2 uv = (vec2(p) + 0.5) / vec2(c.out_size);  // 0..1 over the output
  vec2 lp = uv * vec2(c.low_size);              // render pixels
  vec3 rgb;
  if (c.mode == 0u) {
    rgb = texture(final_image, uv * c.low_scale).rgb;
  } else if (c.mode == 1u) {
    rgb = Cubic(final_image, lp, c.low_size);
  } else if (c.mode == 2u || c.mode == 3u) {
    // DLSS's output; with bias (set 5 holds the character mask then), its history is traded for this frame's
    // own upscale on the characters (no smearing, less smoothing).
    rgb = texelFetch(dlss_image, ivec2(p), 0).rgb;
    float m = c.bias > 0.0 ? clamp(texture(depth_image, uv).r * c.bias, 0.0, 1.0) : 0.0;
    if (c.motion_bias.y > 0.0) {
      // Fast-moving pixels (the game's own motion blur is in DLSS's input there): this frame's upscale.
      m = max(m, smoothstep(c.motion_bias.x, c.motion_bias.y, length(texture(motion_image, uv).rg)));
    }
    {
      if (m > 0.0) {
        vec3 cur = c.mode == 3u ? Cubic(prehud_image, lp, c.low_size) : Cubic(final_image, lp, c.low_size);
        rgb = mix(rgb, cur, m);
      }
    }
    if (c.mode == 3u) {
      rgb += (Cubic(final_image, lp, c.low_size) - Cubic(prehud_image, lp, c.low_size)) * c.hud_gain;
    }
  } else if (c.mode == 5u) {
    vec2 mv = texelFetch(motion_image, ivec2(lp), 0).rg;
    rgb = vec3(clamp(0.5 + mv / (2.0 * c.hud_gain), 0.0, 1.0), 0.5);  // hud_gain = pixels for full colour
  } else if (c.mode == 7u) {
    float m = texelFetch(depth_image, ivec2(lp), 0).r;
    rgb = texture(final_image, uv * c.low_scale).rgb * 0.5 + vec3(0.5 * m, 0.0, 0.0);
  } else if (c.mode == 6u) {
    float dev = texelFetch(depth_image, ivec2(lp), 0).r;
    rgb = vec3(1.0 - clamp(-log2(max(dev, 1e-7)) / 16.0, 0.0, 1.0));
  } else {
    vec3 hud = abs(texture(final_image, uv * c.low_scale).rgb - texture(prehud_image, uv * c.pre_scale).rgb);
    rgb = texture(final_image, uv * c.low_scale).rgb * 0.25 + vec3(min(1.0, 4.0 * max(hud.r, max(hud.g, hud.b))));
  }
  imageStore(out_image, ivec2(p), vec4(clamp(rgb, 0.0, 1.0), 1.0));
}
