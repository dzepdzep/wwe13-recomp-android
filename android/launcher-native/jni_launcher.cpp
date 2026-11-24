#include <filesystem>
#include <cstdio>
#include <algorithm>
// JNI bridge between the Android launcher activity (Java) and the shared WWE '13 launcher core (launcher2/core).
#include <jni.h>

#include <atomic>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "launcher_core.h"

namespace lc = wwe13::launcher;
namespace fs = std::filesystem;

namespace {
lc::Paths g_paths;
std::mutex g_progress_mutex;
uint64_t g_done = 0, g_total = 0;
std::string g_label;
lc::CancelFlag g_cancel{false};

void Progress(uint64_t done, uint64_t total, const std::string& label) {
  std::lock_guard<std::mutex> lock(g_progress_mutex);
  g_done = done;
  g_total = total;
  g_label = label;
}
void ResetProgress() {
  g_cancel = false;
  Progress(0, 0, "");
}
std::string Str(JNIEnv* env, jstring s) {
  if (!s) return {};
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string r(c ? c : "");
  if (c) env->ReleaseStringUTFChars(s, c);
  return r;
}
jstring J(JNIEnv* env, const std::string& s) { return env->NewStringUTF(s.c_str()); }
jstring R(JNIEnv* env, const lc::Result& r) { return J(env, r.ok ? std::string() : (r.error.empty() ? "Failed." : r.error)); }
const char* StateName(lc::FileState s) {
  switch (s) {
    case lc::FileState::kFound: return "found";
    case lc::FileState::kWrongVersion: return "wrong";
    case lc::FileState::kCorrupt: return "corrupt";
    default: return "missing";
  }
}
}  // namespace

#define FN(ret, name) extern "C" JNIEXPORT ret JNICALL Java_com_toni_wwe13_LauncherNative_##name

FN(void, init)(JNIEnv* env, jclass, jstring root) {
  const fs::path r = Str(env, root);
  g_paths.exe_dir = r;
  g_paths.game_exe = r / "wwe13";
  g_paths.settings_file = r / "wwe13-enhanced.ini";
  g_paths.userdata_dir = r / "userdata";
  g_paths.music_dir = r / "music";
  g_paths.logs_dir = r / "logs";
  g_paths.backups_dir = r / "backups";
  g_paths.default_game_folder = r / "WWE 13";
  std::error_code ec;
  for (const auto& d : {g_paths.userdata_dir, g_paths.music_dir, g_paths.logs_dir, g_paths.backups_dir,
                        g_paths.default_game_folder}) {
    fs::create_directories(d, ec);
  }
}

// "base|tu|dlc_installed|dlc_known\ndetail"
FN(jstring, status)(JNIEnv* env, jclass) {
  lc::GameFilesStatus s = lc::CheckGameFolder(g_paths.default_game_folder, g_paths);
  std::ostringstream o;
  o << StateName(s.base_game) << "|" << StateName(s.title_update) << "|" << s.dlc_installed << "|"
    << s.dlc_known << "\n" << s.detail;
  return J(env, o.str());
}

FN(jstring, progress)(JNIEnv* env, jclass) {
  std::lock_guard<std::mutex> lock(g_progress_mutex);
  return J(env, std::to_string(g_done) + "|" + std::to_string(g_total) + "|" + g_label);
}
FN(void, cancel)(JNIEnv*, jclass) { g_cancel = true; }

FN(jstring, extractIso)(JNIEnv* env, jclass, jstring path) {
  ResetProgress();
  return R(env, lc::ExtractDiscImage(Str(env, path), g_paths.default_game_folder, Progress, g_cancel));
}

// Returns "" on success, otherwise the error. kind_out[0] receives "tu", "dlc" or "unknown".
FN(jstring, inspectPackage)(JNIEnv* env, jclass, jstring path) {
  auto info = lc::InspectPackage(Str(env, path));
  if (!info) return J(env, "unknown|");
  const char* kind = info->kind == lc::PackageKind::kTitleUpdate ? "tu"
                     : info->kind == lc::PackageKind::kDlc       ? "dlc"
                                                                 : "unknown";
  return J(env, std::string(kind) + "|" + info->display_name);
}

FN(jstring, importPackage)(JNIEnv* env, jclass, jstring path) {
  ResetProgress();
  return R(env, lc::ImportPackage(Str(env, path), g_paths, g_paths.default_game_folder, Progress, g_cancel));
}

FN(jstring, importCreations)(JNIEnv* env, jclass, jobjectArray paths, jstring pack_name, jboolean replace) {
  ResetProgress();
  std::vector<fs::path> files;
  for (jsize i = 0; i < env->GetArrayLength(paths); ++i) {
    files.emplace_back(Str(env, static_cast<jstring>(env->GetObjectArrayElement(paths, i))));
  }
  return R(env, lc::ImportCreations(g_paths, files, replace, Progress, g_cancel, Str(env, pack_name)));
}

// One line per creation: "kind|display name|package name"
FN(jstring, listCreations)(JNIEnv* env, jclass) {
  std::ostringstream o;
  for (const auto& c : lc::ListInstalledCreations(g_paths)) {
    static const char* kinds[] = {"Superstar", "Entrance", "Arena", "Logos", "Save"};
    o << kinds[int(c.kind)] << "|" << c.display_name << "|" << c.package_name << "\n";
  }
  return J(env, o.str());
}
FN(jstring, removeCreation)(JNIEnv* env, jclass, jstring package_name) {
  return R(env, lc::RemoveCreation(g_paths, Str(env, package_name)));
}

// One line per song: "path|title|format|seconds"
FN(jstring, listSongs)(JNIEnv* env, jclass) {
  std::ostringstream o;
  for (const auto& s : lc::ListSongs(g_paths)) {
    o << s.file.string() << "|" << s.title << "|" << s.format << "|" << int(s.seconds) << "\n";
  }
  return J(env, o.str());
}
FN(jstring, removeSong)(JNIEnv* env, jclass, jstring path) {
  return R(env, lc::RemoveSong(g_paths, Str(env, path)));
}

FN(jint, getLanguage)(JNIEnv*, jclass) { return lc::LoadGameLanguage(g_paths).id; }
FN(jstring, setLanguage)(JNIEnv* env, jclass, jint id) {
  return R(env, lc::SaveGameLanguage(g_paths, id));
}

FN(jstring, backupSaves)(JNIEnv* env, jclass) {
  lc::BackupInfo info;
  return R(env, lc::BackupSaves(g_paths, false, &info));
}
// One line per backup: "path|when|bytes|auto"
FN(jstring, listBackups)(JNIEnv* env, jclass) {
  std::ostringstream o;
  for (const auto& b : lc::ListBackups(g_paths)) {
    o << b.file.string() << "|" << b.when << "|" << b.bytes << "|" << (b.automatic ? 1 : 0) << "\n";
  }
  return J(env, o.str());
}
FN(jstring, exportSaves)(JNIEnv* env, jclass, jstring path) {
  return R(env, lc::ExportSaves(g_paths, Str(env, path)));
}
// Changes whenever a save file is added, removed or rewritten; "" when there are no saves.
FN(jstring, savesFingerprint)(JNIEnv* env, jclass) {
  namespace fs = std::filesystem;
  const fs::path account = g_paths.userdata_dir / "B13EBABEBABEBABE";
  std::vector<std::string> items;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(account, fs::directory_options::skip_permission_denied, ec), end;
       !ec && it != end; it.increment(ec)) {
    std::error_code e2;
    if (!it->is_regular_file(e2) || e2) continue;
    const auto size = it->file_size(e2);
    const long long time = (long long)it->last_write_time(e2).time_since_epoch().count();
    items.push_back(fs::relative(it->path(), account, e2).string() + ":" + std::to_string(size) + ":" +
                    std::to_string(time));
  }
  if (items.empty()) return J(env, "");
  std::sort(items.begin(), items.end());
  uint64_t h = 1469598103934665603ull;
  for (const auto& item : items) {
    for (unsigned char c : item) h = (h ^ c) * 1099511628211ull;
    h = (h ^ '\n') * 1099511628211ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
  return J(env, buf);
}
FN(jstring, restoreBackup)(JNIEnv* env, jclass, jstring path) {
  return R(env, lc::RestoreBackup(g_paths, Str(env, path)));
}
