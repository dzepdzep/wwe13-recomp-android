/**
 * @file        ui/windowed_app_main_sdl.cpp
 * @brief       Entry point for windowed applications (SDL3 windowing)
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ui/windowed_app.h>
#include <rex/ui/windowed_app_context_sdl.h>

#if REX_PLATFORM_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objbase.h>
#include <shellapi.h>
#endif

#if REX_PLATFORM_ANDROID
#include <android/log.h>
#include <dlfcn.h>
#include <sys/system_properties.h>
#include <cstring>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <SDL3/SDL_system.h>
#include <adrenotools/driver.h>
#include <rex/system.h>
namespace rex::ui::vulkan {
using VulkanLoaderOverride = void* (*)();
void SetVulkanLoaderOverride(VulkanLoaderOverride override_fn);
}  // namespace rex::ui::vulkan
#endif

namespace {

int RunWindowedApp(int argc, char** argv) {
  auto remaining = rex::cvar::Init(argc, argv);
  rex::cvar::ApplyEnvironment();
  rex::InitLoggingEarly();

  int result;
  {
    rex::ui::SDLWindowedAppContext app_context;
    if (!app_context.Initialize()) {
      return EXIT_FAILURE;
    }

#if REX_PLATFORM_WIN32
    // Apartment-threaded COM for shell dialogs.
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) {
      return EXIT_FAILURE;
    }
#endif

    std::unique_ptr<rex::ui::WindowedApp> app = rex::ui::GetWindowedAppCreator()(app_context);

    // Match remaining positional args to the app's expected options.
    const auto& option_names = app->GetPositionalOptions();
    std::map<std::string, std::string> parsed;
    size_t count = std::min(remaining.size(), option_names.size());
    for (size_t i = 0; i < count; ++i) {
      parsed[option_names[i]] = remaining[i];
    }
    app->SetParsedArguments(std::move(parsed));

    result = app->OnInitialize() ? app_context.RunMainMessageLoop() : EXIT_FAILURE;

    app->InvokeOnDestroy();
  }

#if REX_PLATFORM_WIN32
  CoUninitialize();
#endif

  return result;
}

#if REX_PLATFORM_WIN32
// Convert wide argv from CommandLineToArgvW to UTF-8 for cvar::Init.
std::vector<std::string> WideArgsToUtf8(int argc, wchar_t** wargv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    std::wstring wide(wargv[i]);
    if (wide.empty()) {
      args.emplace_back();
      continue;
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr,
                                   0, nullptr, nullptr);
    std::string utf8(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), utf8.data(), size,
                        nullptr, nullptr);
    args.push_back(std::move(utf8));
  }
  return args;
}
#endif

}  // namespace

#if REX_PLATFORM_WIN32

int WINAPI wWinMain(HINSTANCE hinstance, HINSTANCE hinstance_prev, LPWSTR command_line,
                    int show_cmd) {
  (void)hinstance;
  (void)hinstance_prev;
  (void)command_line;
  (void)show_cmd;

  int wargc = 0;
  wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
  auto utf8_args = WideArgsToUtf8(wargc, wargv);
  LocalFree(wargv);

  std::vector<char*> argv_ptrs;
  argv_ptrs.reserve(utf8_args.size());
  for (auto& s : utf8_args) {
    argv_ptrs.push_back(s.data());
  }
  return RunWindowedApp(static_cast<int>(argv_ptrs.size()), argv_ptrs.data());
}

#elif REX_PLATFORM_ANDROID

// WWE13 Android port: SDL's Java activity loads libmain.so and calls SDL_main on its own thread.
// The app's external files folder holds the game files ("WWE 13"), user data, logs, mods and the
// launcher-written option files (env.txt / launch_options.txt and their *_user.txt companions).
namespace {
std::string g_vk_driver, g_vk_hook_dir, g_vk_tmp_dir;
void* OpenCustomVulkanDriver() {
  if (g_vk_driver.empty()) return nullptr;
  const std::filesystem::path driver_path(g_vk_driver);
  const std::string driver_dir = driver_path.parent_path().string() + "/";
  void* handle = adrenotools_open_libvulkan(RTLD_NOW, ADRENOTOOLS_DRIVER_CUSTOM, g_vk_tmp_dir.c_str(),
                                            g_vk_hook_dir.c_str(), driver_dir.c_str(),
                                            driver_path.filename().c_str(), nullptr, nullptr);
  __android_log_print(handle ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "rexglue",
                      "GPU driver %s: %s", g_vk_driver.c_str(), handle ? "loaded via adrenotools" : "FAILED");
  return handle;
}
std::string Trim(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}
std::vector<std::string> ReadLines(const std::filesystem::path& p) {
  std::vector<std::string> out;
  if (FILE* f = std::fopen(p.c_str(), "r")) {
    char line[2048];
    while (std::fgets(line, sizeof(line), f)) {
      std::string s = Trim(line);
      if (!s.empty() && s[0] != '#') out.push_back(std::move(s));
    }
    std::fclose(f);
  }
  return out;
}
void SelectCustomDriver() {
  const char* internal = SDL_GetAndroidInternalStoragePath();
  if (!internal) return;
  std::error_code ec;
  const std::filesystem::path drivers = std::filesystem::path(internal) / "drivers";
  auto cur = ReadLines(drivers / "current.txt");
  const std::string chosen = cur.empty() ? std::string() : cur[0];
  if (chosen == "system") return;
  for (auto& dir : std::filesystem::directory_iterator(drivers, ec)) {
    if (!dir.is_directory() || (!chosen.empty() && dir.path().filename() != chosen)) continue;
    std::string library;
    if (FILE* meta = std::fopen((dir.path() / "meta.json").c_str(), "r")) {
      std::string json;
      char buf[1024];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof(buf), meta)) > 0) json.append(buf, n);
      std::fclose(meta);
      const size_t key = json.find("\"libraryName\"");
      if (key != std::string::npos) {
        const size_t q1 = json.find('"', json.find(':', key));
        const size_t q2 = q1 == std::string::npos ? q1 : json.find('"', q1 + 1);
        if (q2 != std::string::npos) library = json.substr(q1 + 1, q2 - q1 - 1);
      }
    }
    if (!library.empty() && library.find('/') == std::string::npos && std::filesystem::exists(dir.path() / library)) {
      g_vk_driver = (dir.path() / library).string();
      return;
    }
    for (auto& file : std::filesystem::directory_iterator(dir.path(), ec)) {
      const std::string n = file.path().filename().string();
      if ((n.starts_with("libvulkan") || n.starts_with("vulkan.")) && n.ends_with(".so")) {
        g_vk_driver = file.path().string();
        return;
      }
    }
  }
}
}  // namespace

extern "C" __attribute__((visibility("default"))) int SDL_main(int argc, char* argv[]) {
  const char* external = SDL_GetAndroidExternalStoragePath();
  if (const char* files_dir = std::getenv("WWE2K14_FILES_DIR"); files_dir && *files_dir) external = files_dir;
  if (!external) return EXIT_FAILURE;
  const std::filesystem::path root(external);
  std::error_code ec;
  std::filesystem::create_directories(root / "WWE 13", ec);
  std::filesystem::create_directories(root / "userdata", ec);
  std::filesystem::create_directories(root / "logs", ec);
  chdir(root.c_str());
  setenv("HOME", root.c_str(), 1);

  std::vector<std::string> args_storage{
      argc > 0 ? argv[0] : "wwe13",
      "--game_data_root=" + (root / "WWE 13").string(),
      "--xmp_music_dir=" + (root / "music").string(),  // WWE13: the launcher's custom music folder
      "--gpu_plugin=xenos",
      "--async_shader_compilation=false",
      "--vulkan_async_skip_incomplete_frames=false",
      "--fullscreen=true",
      "--vulkan_allow_present_mode_immediate=false",
      "--log_verbose=false",
      "--mnk_mode=false",
      "--user_data_root=" + (root / "userdata").string(),
      // WWE13: the native renderer defaults (also in wwe13j/src/native_settings.cpp). As arguments, because the GPU
      // plugin's settings only exist once it is loaded; launch_options*.txt lines come later and override them.
      "--gpu_native=2",
      "--gpu_fast_clear=true",
      "--gpu_native_truth=true",
      "--gpu_native_half2_skip_gpr_draws=true",
      "--gpu_native_half2=2",
      "--gpu_native_lean_draws=true",
      "--gpu_native_trust_geometry_pages=true",
      "--gpu_native_truth_skip_writeback=true",
      "--vulkan_merge_adjacent_barriers=true",
      "--gpu_native_truth_direct_resolve_scaled=true",
      "--gpu_native_truth_fill_direct=true",
      "--vulkan_texture_load_direct_unscaled=true",
      "--vulkan_texture_load_direct_unscaled_any_page=true",
      "--gpu_native_truth_direct_resolve_scaled_full=true",
      "--gpu_native_truth_direct_resolve_scaled_texture=true",
      "--gpu_native_truth_defer_dest_use=true",
      "--gpu_native_truth_texture_fill_scaled=true",
      "--gpu_native_truth_texture_fill_scaled_kinds=32",
      "--gpu_native_truth_batch_resolves=true",
      "--gpu_native_truth_batch_rename=true",
      "--gpu_native_truth_lazy_pass_deps=true",
      "--gpu_native_truth_lazy_pass_style=1",
      "--gpu_native_truth_rename_depth=true",
      "--vulkan_texture_sample_written_early=true",
      "--gpu_native_lean_benign_writes=true",
      "--gpu_native_truth_skip_writeback_scaled=true",
      "--gpu_native_truth_defer_dest_use_scaled=true",
      "--gpu_native_truth_batch_fills=true",
      "--gpu_native_truth_direct_resolve_scaled_full_texture=true",
      "--gpu_native_truth_direct_resolve_full_1x=true",
      "--vulkan_texture_native_storage_small=true",
      "--gpu_native_truth_fill_direct_small=true",
      "--gpu_native_truth_fill_direct_depth=true",
      "--gpu_native_truth_direct_resolve_depth=true",
      "--gpu_native_truth_direct_resolve_scaled_small=true",
      "--vulkan_uniform_pool_device_local=true",
      "--vulkan_device_local_upload_min_heap_mb=128",
      "--gpu_native_truth_host_rt=true",
      "--gpu_native_truth_pass_dont_care=true",
      "--gpu_native_truth_color_none=true",
      "--gpu_native_vertex_endian_spec=true",
      "--gpu_native_ps_texture_spec=true",
      "--gpu_native_ps_mul_mode=2",
      "--gpu_native_ps_implicit_lod=true",
      "--gpu_native_fma=3",
      "--gpu_native_truth_drop_dead_resolves=1",
      "--gpu_native_vs_mul_mode=1",
      "--gpu_native_vs_mul_fetch_finite=true",
      "--gpu_native_ps_epilogue_spec=true",
      "--gpu_native_shader_swap=true",
      "--gpu_native_truth_skip_restores=true",
      "--gpu_native_truth_skip_repeat_resolves=true",
      "--gpu_native_truth_fill_rename_check=true",
      "--gpu_scene_crowd_cull=2",
      "--gpu_scene_crowd_thin=100",
      "--readback_resolve_batch=true",
      "--readback_resolve_batch_min_bytes=16385",
      "--vulkan_require_vertex_pipeline_stores_and_atomics=false",
      "--vulkan_require_fill_mode_non_solid=false",
      "--vulkan_require_geometry_shader=false",
      "--vulkan_repeat_draw_fast=true",
      "--vulkan_repeat_draw_merge=true",
      // WWE13: D5E2FCC9F76BC40C = the stands crowd member mesh (1679 draws/frame in an entrance shot), C1028A64EFCF8229
      // the next most repeated (491); 183CC64A4C4E96AC / 5344BF064B17B7C2 are 2K14's.
      "--vulkan_repeat_draw_merge_shaders=183CC64A4C4E96AC,5344BF064B17B7C2,D5E2FCC9F76BC40C,C1028A64EFCF8229",
      "--wwe2k14_d3d_layer=true",
      "--wwe2k14_d3d_copy=false",
      "--wwe2k14_d3d_inline=true",
      "--gpu_wait_reg_mem_spin=true",
      "--vulkan_async_submit=true",
      "--vulkan_submit_on_wait=true",
      "--vulkan_deferred_descriptor_writes=true",
      "--gpu_type0_bulk_writes=true",
      "--gpu_type0_block_split=true",
      "--gpu_request_ranges_fast=true",
      "--vulkan_reuse_texture_bindings=true",
      "--vulkan_sampler_param_cache=true",
      "--vulkan_constants_dynamic_offsets=true",
      "--gpu_rt_cache_skip_unchanged=true",
      "--vulkan_viewport_info_cache=true",
      "--vulkan_vertex_residency_cache_off=true",
      "--vulkan_async_swap=true",
      "--vulkan_pipeline_no_submit_wait=true",
      "--gpu_request_textures_skip_unchanged=true",
      "--gpu_upload_skip_unchanged=true",
      "--vulkan_texture_load_direct=true",
      "--gpu_native_lean_multi_base=true",
      "--gpu_native_lean_state_cache=true",
      "--gpu_make_valid_skip_arm=true",
      "--vulkan_float_constant_runs=true",
      "--gpu_native_owned_frame=true",
      "--gpu_native_owned_only=true",
      "--gpu_native_owned_frame_tiles_memo=true",
      "--gpu_d3d_noring=1",
      "--wwe2k14_d3d_noring=3",
      "--gpu_skip_noop_draws=true",
      "--vulkan_upload_hoist=2",
      "--vulkan_texture_load_eager=true",
      "--vulkan_texture_load_direct_depth=true",
  };
  // Mods enabled in the launcher: mods/enabled.txt lists mod folder names in load order.
  {
    std::string overlays;
    for (auto& s : ReadLines(root / "mods" / "enabled.txt")) {
      if (s.find('/') != std::string::npos) continue;
      if (!overlays.empty()) overlays += ';';
      overlays += (root / "mods" / s).string();
    }
    if (!overlays.empty()) setenv("REX_MOD_OVERLAYS", overlays.c_str(), 1);
  }
  for (const char* env_name : {"env.txt", "env_user.txt"}) {
    for (auto& s : ReadLines(root / env_name)) {
      auto eq = s.find('=');
      if (eq != std::string::npos) setenv(s.substr(0, eq).c_str(), s.substr(eq + 1).c_str(), 1);
    }
  }
  for (const char* options_name : {"launch_options.txt", "launch_options_user.txt"}) {
    for (auto& s : ReadLines(root / options_name)) {
      if (s.starts_with("--ignore_offset_for_ranged_allocations") || s.starts_with("--debug_ui")) continue;
      args_storage.push_back(s);
    }
  }
  // Custom GPU driver (Mesa Turnip): <internal>/drivers/<name>/, chosen in drivers/current.txt.
  if (const char* forced = std::getenv("REX_VULKAN_DRIVER"); forced && *forced) g_vk_driver = forced;
  else SelectCustomDriver();
  // WWE13: Turnip on Adreno 6xx (Snapdragon 865 / 870 / 888, Adreno 650/660) loses push constants with its "shared
  // consts" (the EDRAM dump/resolve shaders then see zero pitches: a black or garbled picture). Per-stage push constants
  // fix it (tested on a Retroid Pocket Flip 2, Adreno 650: correct picture, demo match 22 -> 49 fps vs the stock driver).
  // Still open: Turnip on the Flip 2 then hits "GPU faulted or hung" (VK_ERROR_DEVICE_LOST) a minute or two into matches
  // (not fixed by nolrz / sysmem / full barriers between passes; TU_DEBUG flushall+syncdraw avoids it at ~20 fps). The
  // launcher therefore defaults Adreno 6xx to the phone's own driver.
  // Only Adreno 6xx, and only when the user did not set TU_DEBUG themselves (env_user.txt).
  if (!g_vk_driver.empty() && !std::getenv("TU_DEBUG")) {
    char model[64] = {};
    if (FILE* f = std::fopen("/sys/class/kgsl/kgsl-3d0/gpu_model", "r")) {
      if (!std::fgets(model, sizeof(model), f)) model[0] = 0;
      std::fclose(f);
    }
    if (!model[0]) {  // gpu_model unreadable: the Snapdragon platforms with an Adreno 6xx
      char plat[PROP_VALUE_MAX] = {};
      __system_property_get("ro.board.platform", plat);
      for (const char* p : {"kona", "lahaina", "msmnile", "lito"})
        if (std::strcmp(plat, p) == 0) std::snprintf(model, sizeof(model), "Adreno6xx (%s)", plat);
    }
    if (std::strncmp(model, "Adreno6", 7) == 0) {
      setenv("TU_DEBUG", "push_consts_per_stage", 1);
      __android_log_print(ANDROID_LOG_INFO, "wwe13", "Turnip on %s: TU_DEBUG=push_consts_per_stage", model);
    } else {
      __android_log_print(ANDROID_LOG_INFO, "wwe13", "GPU model '%s': Turnip defaults", model);
    }
  }
  {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&SDL_main), &info) && info.dli_fname)
      g_vk_hook_dir = std::filesystem::path(info.dli_fname).parent_path().string() + "/";
    if (const char* cache = SDL_GetAndroidCachePath()) {
      std::filesystem::path tmp = std::filesystem::path(cache) / "adrenotools";
      std::filesystem::create_directories(tmp, ec);
      g_vk_tmp_dir = tmp.string() + "/";
    }
  }
  if (!g_vk_driver.empty()) rex::ui::vulkan::SetVulkanLoaderOverride(&OpenCustomVulkanDriver);

  std::vector<char*> args;
  for (auto& s : args_storage) args.push_back(s.data());
  args.push_back(nullptr);

  if (!rex::InitializeAndroidSystemForApplicationContext()) {
    return EXIT_FAILURE;
  }
  int result = RunWindowedApp(int(args.size() - 1), args.data());
  rex::ShutdownAndroidSystem();
  std::_Exit(result);
}

#else

int main(int argc, char* argv[]) {
  return RunWindowedApp(argc, argv);
}

#endif
