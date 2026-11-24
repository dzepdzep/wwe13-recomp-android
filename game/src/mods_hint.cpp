// wwe13 - on-screen text drawn over the game:
// - the R3 mods hint on the character select screen (only when a mod adds superstars), next to the game's own button
//   legends at the bottom (A SELECT, B BACK, HELP MENU), without a box;
// - the FPS counter (launcher setting "Show an FPS counter in the game": WWE13_SHOW_FPS=1), the game's own frames
//   per second, top left.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <imgui.h>
#include <rex/ui/imgui_dialog.h>

bool wwe13_roster_hint(int* page, int* pages);  // roster_mods.cpp
uint32_t wwe13_game_frames();                   // roster_mods.cpp

namespace wwe13 {
namespace {
// White text with a dark drop shadow, as the game's legends.
void ShadowText(ImDrawList* dl, ImFont* font, float size, ImVec2 at, const char* text) {
  const float d = size * 0.08f;
  dl->AddText(font, size, ImVec2(at.x + d, at.y + d), IM_COL32(0, 0, 0, 200), text);
  dl->AddText(font, size, at, IM_COL32(255, 255, 255, 255), text);
}

bool FpsOn() {
  static const bool on = [] {
    const char* v = std::getenv("WWE13_SHOW_FPS");
    return v && std::strcmp(v, "1") == 0;
  }();
  return on;
}
}  // namespace

class OverlayText final : public rex::ui::ImGuiDialog {
 public:
  explicit OverlayText(rex::ui::ImGuiDrawer* drawer) : rex::ui::ImGuiDialog(drawer) {}

 protected:
  void OnDraw(ImGuiIO& io) override {
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImFont* font = ImGui::GetFont();
    const float w = io.DisplaySize.x, h = io.DisplaySize.y;

    int page = 0, pages = 0;
    if (wwe13_roster_hint(&page, &pages)) {
      char label[48];
      if (page == 0)
        std::snprintf(label, sizeof label, pages == 1 ? "MODS" : "MODS (%d PAGES)", pages);
      else if (page < pages)
        std::snprintf(label, sizeof label, "NEXT PAGE (%d/%d)", page, pages);
      else
        std::snprintf(label, sizeof label, page == 1 ? "WWE ROSTER" : "WWE ROSTER (%d/%d)", page, pages);
      // (the legends' row: its text centre is at 90.3% of the height; HELP MENU ends at about 66% of the width)
      const float size = h * 0.03f, cy = h * 0.903f, r = h * 0.0145f;
      const ImVec2 c(w * 0.70f, cy);
      dl->AddCircleFilled(ImVec2(c.x + r * 0.1f, c.y + r * 0.1f), r, IM_COL32(0, 0, 0, 160), 24);
      dl->AddCircleFilled(c, r, IM_COL32(70, 70, 70, 255), 24);
      dl->AddCircle(c, r, IM_COL32(220, 220, 220, 255), 24, h * 0.0022f);
      const float isz = r * 1.05f;
      const ImVec2 iw = font->CalcTextSizeA(isz, 1e9f, 0.0f, "RT");
      dl->AddText(font, isz, ImVec2(c.x - iw.x * 0.5f, c.y - iw.y * 0.5f), IM_COL32(255, 255, 255, 255), "RT");
      ShadowText(dl, font, size, ImVec2(c.x + r * 1.6f, cy - size * 0.5f), label);
    }

    if (FpsOn()) {
      const auto now = std::chrono::steady_clock::now();
      const uint32_t frames = wwe13_game_frames();
      if (last_.time_since_epoch().count() == 0) {
        last_ = now;
        last_frames_ = frames;
      }
      const double secs = std::chrono::duration<double>(now - last_).count();
      if (secs >= 0.5) {
        fps_ = int((frames - last_frames_) / secs + 0.5);
        last_ = now;
        last_frames_ = frames;
      }
      char text[24];
      std::snprintf(text, sizeof text, "%d FPS", fps_);
      ShadowText(dl, font, h * 0.032f, ImVec2(w * 0.012f, h * 0.012f), text);
    }
  }

 private:
  std::chrono::steady_clock::time_point last_{};
  uint32_t last_frames_ = 0;
  int fps_ = 0;
};

rex::ui::ImGuiDialog* CreateModsHint(rex::ui::ImGuiDrawer* drawer) { return new OverlayText(drawer); }
}  // namespace wwe13
