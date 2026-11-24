#version 450

// wwe2k14 patch (Session 94, present_pre_rotate): the final-pass variant of guest_output_triangle_strip_rect_vs
// (same rectangle push constants at offset 0, same 4-vertex strip). The swapchain image is in the panel native
// orientation, the picture is laid out in the window (logical) one: the position is rotated by the display transform
// (ROT, a specialization constant: 0 none, 1 = clip rotated by -90 degrees, 2 = 180, 3 = +90 degrees) and the logical
// framebuffer coordinate is passed on at location 0 in place of gl_FragCoord (the pixel shaders are patched at run time
// to read that instead of the built-in; they compute their texture coordinates from it).
layout(constant_id = 0) const uint ROT = 0u;
layout(constant_id = 1) const float LOGICAL_WIDTH = 1.0;   // logical viewport size in pixels
layout(constant_id = 2) const float LOGICAL_HEIGHT = 1.0;

layout(push_constant) uniform PushConstants {
  vec2 rect_offset;                    // NDC of the logical viewport
  vec2 rect_size;
} pc;

layout(location = 0) out vec4 logical_frag_coord;

void main() {
  vec2 t = vec2((uvec2(uint(gl_VertexIndex)) >> uvec2(0u, 1u)) & uvec2(1u));
  vec2 p = pc.rect_offset + t * pc.rect_size;
  logical_frag_coord = vec4((p * 0.5 + 0.5) * vec2(LOGICAL_WIDTH, LOGICAL_HEIGHT), 0.0, 1.0);
  vec2 q = p;
  if (ROT == 1u) {
    q = vec2(p.y, -p.x);
  } else if (ROT == 2u) {
    q = vec2(-p.x, -p.y);
  } else if (ROT == 3u) {
    q = vec2(-p.y, p.x);
  }
  gl_Position = vec4(q, 0.0, 1.0);
}
