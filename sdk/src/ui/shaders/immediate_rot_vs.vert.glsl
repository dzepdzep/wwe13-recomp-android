#version 450

// wwe2k14 patch (Session 94, present_pre_rotate): immediate_vs (the UI drawer's vertex shader) with the display-transform
// rotation of the final pass (see guest_output_triangle_strip_rect_rot_vs.vert.glsl; rot 0 is the unrotated shader).
layout(push_constant) uniform PushConstants {
  vec2 coordinate_space_size_inv;
  uint rot;
} pc;

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;
layout(location = 2) in vec4 in_color;
layout(location = 0) out vec2 out_texcoord;
layout(location = 1) out vec4 out_color;

void main() {
  out_texcoord = in_texcoord;
  out_color = in_color;
  vec2 p = in_position * pc.coordinate_space_size_inv * 2.0 - 1.0;
  vec2 q = p;
  if (pc.rot == 1u) {
    q = vec2(p.y, -p.x);
  } else if (pc.rot == 2u) {
    q = vec2(-p.x, -p.y);
  } else if (pc.rot == 3u) {
    q = vec2(-p.y, p.x);
  }
  gl_Position = vec4(q, 0.0, 1.0);
}
