#pragma once
/**
 * @file        rex/ui/surface_android.h
 * @brief       Android ANativeWindow surface (VK_KHR_android_surface)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * wwe2k14 Android port. The window is SDL's (SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER) and stays owned by SDL.
 */

#include <android/native_window.h>

#include <rex/ui/surface.h>

namespace rex {
namespace ui {

class AndroidNativeWindowSurface final : public Surface {
 public:
  explicit AndroidNativeWindowSurface(ANativeWindow* window) : window_(window) {
    // wwe2k14 patch (Session 92): SDL may release its reference on the Java thread; hold our own.
    if (window_) ANativeWindow_acquire(window_);
  }
  ~AndroidNativeWindowSurface() override {
    if (window_) ANativeWindow_release(window_);
  }

  TypeIndex GetType() const override { return kTypeIndex_AndroidNativeWindow; }
  ANativeWindow* window() const { return window_; }

 protected:
  bool GetSizeImpl(uint32_t& width_out, uint32_t& height_out) const override {
    int32_t width = window_ ? ANativeWindow_getWidth(window_) : 0;
    int32_t height = window_ ? ANativeWindow_getHeight(window_) : 0;
    if (width <= 0 || height <= 0) {
      width_out = 0;
      height_out = 0;
      return false;
    }
    width_out = uint32_t(width);
    height_out = uint32_t(height);
    return true;
  }

 private:
  ANativeWindow* window_;
};

}  // namespace ui
}  // namespace rex
