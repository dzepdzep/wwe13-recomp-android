// wwe2k14 patch (Session 95, P2b of out\s95\android_structural.md; gpu_native_index_stitch): a triangle strip with reset values as ONE strip without any, so it can be drawn with primitive restart off.
// Pure code (no Vulkan, no guest memory): tools\owned\stitch_test.cpp compares it with what restart draws.
//   - Each run of elements between reset values is a strip of its own. A run of fewer than 3 elements makes no triangle under restart and is dropped.
//   - Runs are joined by repeating the last element of what is written and the first of the next run, so every triangle of the join has two equal indices (zero area, no fragments: the native
//     pipelines are always VK_POLYGON_MODE_FILL). An extra repeat keeps each run on an even position: an even triangle of a strip keeps its vertex order, so winding and the provoking vertex are the
//     ones the run has under restart.
//   - No run of 3 or more: one triangle of three equal indices, so the draw still has a count and still draws nothing.
#pragma once

#include <cstdint>
#include <vector>

namespace rex::graphics::vulkan::native {

// `in`: `count` host-order elements. Fills `out`; false when every element is the reset value (nothing to draw from).
template <typename T>
bool StitchTriangleStrip(const T* in, uint32_t count, T reset, std::vector<T>& out) {
  out.clear();
  bool any = false;
  T first = 0;
  for (uint32_t i = 0; i < count;) {
    if (in[i] == reset) {
      ++i;
      continue;
    }
    uint32_t end = i + 1;
    while (end < count && in[end] != reset) ++end;
    if (!any) {
      any = true;
      first = in[i];
    }
    if (end - i >= 3) {
      if (!out.empty()) {
        const T last = out.back();
        out.push_back(last);
        if (out.size() % 2 == 0) out.push_back(last);  // (an odd length before the join: one more, or the run would start on an odd position)
        out.push_back(in[i]);
      }
      out.insert(out.end(), in + i, in + end);
    }
    i = end;
  }
  if (!any) return false;
  if (out.empty()) out.assign(3, first);
  return true;
}

}  // namespace rex::graphics::vulkan::native
