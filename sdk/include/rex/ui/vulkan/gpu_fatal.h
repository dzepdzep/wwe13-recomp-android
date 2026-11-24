// wwe2k14 patch (Session 95, PLAIN): a GPU that lacks something the renderer needs and has no fallback for ends the run with a
// NAMED reason instead of a black frame. Header-only (inline) so the UI library (device creation) and the GPU library (the
// truth objects) both use it without an exported symbol; it keeps no state but a function-static "already asked".
//
// What it does: logs "FATAL: GPU lacks <what>", appends the same line to logs/startup.txt (the game's black box file, which the
// phone launcher reads), writes logs/exit_request.txt (the game's Java side polls it and brings the launcher back), waits
// 3 s for that, then _Exit(3). Only when the cvar gpu_fatal_exit says so: -1 (default) = Android only, 0 = never, 1 = always
// (a PC test of the path). The logs folder is <current directory>/logs, as the black box's is.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>

namespace rex::ui::vulkan {

inline void GpuFatalExit(const std::string& what) {
  static bool asked = false;
  if (asked) return;
  asked = true;
  const std::string mode = rex::cvar::GetFlagByName("gpu_fatal_exit");
  const bool android = REX_PLATFORM_ANDROID != 0;
  const bool on = mode == "1" || (mode != "0" && android);
  const std::string line = "FATAL: GPU lacks " + what;
  if (!on) {
    REXLOG_ERROR("[quirks] {} (gpu_fatal_exit={}: no exit requested)", line, mode);
    return;
  }
  REXLOG_ERROR("[quirks] {} - ending the run (gpu_fatal_exit={})", line, mode);
  std::error_code ec;
  const std::filesystem::path logs = std::filesystem::current_path(ec) / "logs";
  std::filesystem::create_directories(logs, ec);
  if (FILE* f = std::fopen((logs / "startup.txt").string().c_str(), "ab")) {
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }
  if (FILE* f = std::fopen((logs / "exit_request.txt").string().c_str(), "wb")) {
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::_Exit(3);
}

}  // namespace rex::ui::vulkan
