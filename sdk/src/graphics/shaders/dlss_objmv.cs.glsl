// wwe2k14 patch (Session 24, OBJMV): motion vectors for the characters (wrestlers, referee), for DLSS / DLAA.
//
// The character draws' vertex shaders run with their gl_Position captured by transform feedback (one vec4 per
// vertex of every triangle, in primitive order), this frame's draw into `cur`, the same draw of last frame (matched by
// shader and order) into `prev`. One workgroup of 64 threads per triangle rasterizes it over the scene's pixels (the
// threads share the bounding box's pixels: big close-up triangles stay fast; no graphics pipeline:
// nothing of the emulated render state is touched): where the triangle covers a pixel's (unjittered) sample and its
// view depth there matches the scene depth (the visible surface), the pixel's motion becomes the screen-space move of
// that surface point from last frame (perspective-correct, the camera included) and the mask gets 1.
//   set 0: this frame's captured positions (storage buffer, vec4 per vertex);
//   set 1: last frame's (same layout);
//   set 2: the scene depth DLSS gets (R32_UINT view: the float bits of the reversed hardware depth, 1 = near);
//   set 3: the motion image (RG16F, host pixels, current -> previous, as the camera pass writes it);
//   set 4: the character mask (R8: 1 where a character surface was found).
//
// Generated header: glslangValidator -V --target-env vulkan1.0 -S comp (glslang from the SDK's submodule, built out
// of tree: out\glslang-build), see vulkan_spirv/dlss_objmv_cs.h.

#version 460

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer CurPositions { vec4 cur[]; };
layout(set = 1, binding = 0, std430) readonly buffer PrevPositions { vec4 prev[]; };
layout(set = 2, binding = 0, r32ui) uniform readonly uimage2D depth_image;
layout(set = 3, binding = 0, rg16f) uniform writeonly image2D motion_image;
layout(set = 4, binding = 0, r8) uniform writeonly image2D mask_image;

layout(push_constant) uniform Constants {
  vec4 viewport;    // the draw's host viewport without the jitter: x, y, width, height (host pixels)
  vec2 jitter;      // this frame's jitter (host pixels)
  vec2 near_far;    // the camera's near / far (for the scene depth -> view z)
  uint cur_first;   // the draw's first vertex in cur[]
  uint prev_first;  // and in prev[]
  uint triangles;
  uint size;        // render width | height << 16
  float tolerance;  // accepted |view z - scene z| / scene z
  uint flags;       // bit 0: write motion, bit 1: write the mask
  uint max_pixels;  // triangles with a larger pixel bounding box are skipped
  uint groups_x;    // workgroups per dispatch row (triangle = row * groups_x + column)
} c;

float Edge(vec2 a, vec2 b, vec2 p) { return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x); }

vec2 ToScreen(vec4 clip) { return c.viewport.xy + (clip.xy / clip.w * 0.5 + 0.5) * c.viewport.zw; }

void main() {
  uint t = gl_WorkGroupID.y * c.groups_x + gl_WorkGroupID.x;
  if (t >= c.triangles) return;
  uint ci = c.cur_first + 3u * t, pi = c.prev_first + 3u * t;
  vec4 a = cur[ci], b = cur[ci + 1u], d = cur[ci + 2u];
  vec4 pa = prev[pi], pb = prev[pi + 1u], pd = prev[pi + 2u];
  // Near-plane crossings: skip (the camera pass keeps those pixels).
  if (a.w <= 1e-3 || b.w <= 1e-3 || d.w <= 1e-3) return;
  vec2 sa = ToScreen(a), sb = ToScreen(b), sd = ToScreen(d);
  float area = Edge(sa, sb, sd);
  if (abs(area) < 1e-8) return;
  uvec2 size = uvec2(c.size & 0xFFFFu, c.size >> 16);
  // Pixel p samples the unjittered position p + 0.5 - jitter.
  vec2 lo = min(sa, min(sb, sd)) - 0.5 + c.jitter, hi = max(sa, max(sb, sd)) - 0.5 + c.jitter;
  ivec2 p0 = max(ivec2(ceil(lo)), ivec2(0));
  ivec2 p1 = min(ivec2(floor(hi)), ivec2(size) - 1);
  if (p1.x < p0.x || p1.y < p0.y) return;
  if (uint(p1.x - p0.x + 1) * uint(p1.y - p0.y + 1) > c.max_pixels) return;
  float n = c.near_far.x, f = c.near_far.y;
  uint bw = uint(p1.x - p0.x + 1), area_px = bw * uint(p1.y - p0.y + 1);
  for (uint i = gl_LocalInvocationIndex; i < area_px; i += 64u) {
    {
      int x = p0.x + int(i % bw), y = p0.y + int(i / bw);
      vec2 u = vec2(x, y) + 0.5 - c.jitter;
      float w0 = Edge(sb, sd, u) / area, w1 = Edge(sd, sa, u) / area;
      float w2 = 1.0 - w0 - w1;
      if (w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5) continue;
      // Perspective-correct weights; the surface's view depth is the interpolated clip w.
      vec3 pw = vec3(w0 / a.w, w1 / b.w, w2 / d.w);
      float inv_w = pw.x + pw.y + pw.z;
      float z = 1.0 / inv_w;
      pw *= z;
      // The scene's visible surface here (the same decode as the camera pass).
      float dev = uintBitsToFloat(imageLoad(depth_image, ivec2(x, y)).r);
      float dd = clamp(1.0 - dev, 0.0, 1.0);
      float z_scene = n * f / (f - dd * (f - n));
      if (abs(z - z_scene) > c.tolerance * z_scene) continue;
      vec4 pp = pw.x * pa + pw.y * pb + pw.z * pd;
      if (pp.w <= 1e-3) continue;
      if ((c.flags & 1u) != 0u) imageStore(motion_image, ivec2(x, y), vec4(ToScreen(pp) - u, 0.0, 0.0));
      if ((c.flags & 2u) != 0u) imageStore(mask_image, ivec2(x, y), vec4(1.0));
    }
  }
}
