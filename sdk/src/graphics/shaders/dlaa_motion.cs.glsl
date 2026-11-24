// wwe2k14 patch (Session 18, DLAA phase 1): camera motion vectors from the scene depth resolve.
//
// Bindings (all through the texture cache's direct-load storage set layout: one storage image at binding 0):
//   set 0: the scene depth resolve's cached texture, R32_UINT view (the guest word of each texel after the load's
//          endian swap: D24FS8 = 20e4 float depth in the top 24 bits, reversed: near = 1);
//   set 1: an output image, R32_UINT view: a debug view's colours, or (DLAA) the hardware depth as float bits;
//   set 2: our motion image, RG16F: where this pixel's surface was last frame, minus where it is now, in host
//          pixels (the DLSS convention: current -> previous), jitter removed.
// Camera only: everything is treated as static (wrestlers, crowd and cloth get the camera's motion).
//
// Generated header: glslangValidator -V --target-env vulkan1.0 -S comp (glslang from the SDK's submodule, built out
// of tree: out\glslang-build), see vulkan_spirv/dlaa_motion_cs.h.

#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D depth_image;
layout(set = 1, binding = 0, r32ui) uniform uimage2D color_image;
layout(set = 2, binding = 0, rg16f) uniform writeonly image2D motion_image;

layout(push_constant) uniform Constants {
  // Row-vector affine matrix: this frame's view space -> last frame's view space (inverse(V_now) * V_prev).
  mat4 reproject;
  vec2 tan_half_fov;   // x, y: tan of half the horizontal / vertical field of view
  vec2 near_far;       // the camera's near / far planes (view units)
  vec2 jitter;         // this frame's jitter (host pixels), removed from the pixel position
  uint size;           // width | height << 16 (host pixels)
  uint mode;           // bits 0-1: debug 0 off, 1 motion as colour, 2 depth as grey; bits 2-3: depth byte order;
                       // bit 4: the colour image is B8G8R8A8 (else R8G8B8A8); bit 5: (no debug) write the
                       // hardware depth (0..1, reversed) as float bits to set 1 (DLAA's depth input);
                       // bits 8-15: motion colour scale (px)
} c;

float F24ToFloat(uint v) {
  uint e = v >> 20, m = v & 0xFFFFFu;
  return e == 0u ? float(m) / 1048576.0 * exp2(-15.0) : (1.0 + float(m) / 1048576.0) * exp2(float(e) - 15.0);
}

uint ByteSwap(uint v) {
  return (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
}

uint PackColor(vec3 rgb) {
  uvec3 b = uvec3(clamp(rgb, 0.0, 1.0) * 255.0 + 0.5);
  if ((c.mode & 16u) != 0u) b = b.bgr;
  return b.x | (b.y << 8) | (b.z << 16) | (255u << 24);
}

void main() {
  uvec2 size = uvec2(c.size & 0xFFFFu, c.size >> 16);
  uvec2 p = gl_GlobalInvocationID.xy;
  if (any(greaterThanEqual(p, size))) return;
  uint word = imageLoad(depth_image, ivec2(p)).r;
  uint order = (c.mode >> 2) & 3u;
  if (order == 1u) word = ByteSwap(word);
  uint d24 = order == 2u ? (word & 0xFFFFFFu) : (word >> 8);
  float dev = F24ToFloat(d24);
  float d = clamp(1.0 - dev, 0.0, 1.0);  // reversed depth -> 0 = near
  float n = c.near_far.x, f = c.near_far.y;
  float z = n * f / (f - d * (f - n));   // distance along the view direction
  // This pixel without the jitter, in [-1, 1] (y down, like the view's y).
  vec2 ndc = ((vec2(p) + 0.5 - c.jitter) / vec2(size)) * 2.0 - 1.0;
  vec4 view_now = vec4(ndc * c.tan_half_fov * z, z, 1.0);
  vec4 view_prev = view_now * c.reproject;  // row vector
  vec2 ndc_prev = view_prev.z > 1e-4 ? view_prev.xy / (view_prev.z * c.tan_half_fov) : ndc;
  vec2 motion = (ndc_prev - ndc) * 0.5 * vec2(size);  // host pixels, current -> previous
  imageStore(motion_image, ivec2(p), vec4(motion, 0.0, 0.0));
  uint debug = c.mode & 3u;
  if (debug == 0u && (c.mode & 32u) != 0u) {
    imageStore(color_image, ivec2(p), uvec4(floatBitsToUint(dev), 0u, 0u, 0u));
  } else if (debug == 1u) {
    float scale = float(max((c.mode >> 8) & 0xFFu, 1u));
    imageStore(color_image, ivec2(p), uvec4(PackColor(vec3(0.5 + motion / (2.0 * scale), 0.5)), 0u, 0u, 0u));
  } else if (debug == 2u) {
    float g = 1.0 - clamp(log(max(z, n) / n) / log(8000.0 / n), 0.0, 1.0);  // white = near, black = 80 m
    imageStore(color_image, ivec2(p), uvec4(PackColor(vec3(g)), 0u, 0u, 0u));
  }
}
