// wwe13 - app on the new ReXGlue SDK (Android port, slim).
#pragma once

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string_view>

#if defined(__ANDROID__)
#include <SDL3/SDL_messagebox.h>
#endif

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/rex_app.h>
#include <rex/ui/imgui_dialog.h>

#include "game_version_check.h"
#include "scene_aa.h"
#include "scene_resolution.h"
#include "shader_cache_seed.h"

namespace rex::ui { class ImGuiDialog; class ImGuiDrawer; }
namespace wwe13 {
rex::ui::ImGuiDialog* CreateModsHint(rex::ui::ImGuiDrawer* drawer);
void InitializeEdramTilesForSceneResolution();
void ApplyNativeRendererSettings();
void StartFrameStats();
}

class Wwe13App : public rex::ReXApp {
 public:
  using rex::ReXApp::ReXApp;

  static std::unique_ptr<rex::ui::WindowedApp> Create(rex::ui::WindowedAppContext& ctx) {
    return std::unique_ptr<Wwe13App>(new Wwe13App(ctx, "wwe13", PPCImageConfig));
  }


  void OnConfigurePaths(rex::PathConfig& paths) override {
    game_data_root_ = paths.game_data_root;
    user_data_root_ = paths.user_data_root;
    game_file_check_result_ = wwe13::CheckSupportedGameFiles(paths.game_data_root);
  }

  void OnPreSetup(rex::RuntimeConfig& config) override {
    (void)config;
    wwe13::InitializeSceneResolutionOption();
    wwe13::InitializeEdramTilesForSceneResolution();
    wwe13::InitializeSceneAaOption();
    wwe13::SeedShaderCache(user_data_root_);
    const char* env = std::getenv("WWE13_DEPTH_F24_CONVERT");
    if (env == nullptr || std::strcmp(env, "0") != 0) {
      rex::cvar::SetFlagByName("depth_float24_convert_in_pixel_shader", "true");
    }
    wwe13::ApplyNativeRendererSettings();
    wwe13::StartFrameStats();
  }

  void OnCreateDialogs(rex::ui::ImGuiDrawer* drawer) override { mods_hint_.reset(wwe13::CreateModsHint(drawer)); }

  void OnPreLaunchModule() override {
    if (game_file_check_result_ == wwe13::GameFileCheckResult::kSupported) return;
    wwe13::ShowUnsupportedGameFilesMessage(game_file_check_result_);
#if defined(__ANDROID__)
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "WWE '13",
                             "The game files are not a supported WWE '13 title update 2.0.1.0 set.",
                             nullptr);
#endif
  }

 private:
  std::unique_ptr<rex::ui::ImGuiDialog> mods_hint_;
  std::filesystem::path game_data_root_;
  std::filesystem::path user_data_root_;
  wwe13::GameFileCheckResult game_file_check_result_ = wwe13::GameFileCheckResult::kSupported;
};
