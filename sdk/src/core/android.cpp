/**
 * @file        rex/core/android.cpp
 * @brief       Android system glue: API level, start-up / shut-down of the Android pieces
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * wwe2k14 Android port. Xenia declared these (rex/platform.h, rex/system.h, rex/filesystem.h) without an Android
 * implementation in this tree. Game files reach the runtime as plain paths (an ISO in the app's folder, or a file
 * descriptor from Android's file picker as /proc/self/fd/N), so content:// URIs are not opened here.
 */

#include <rex/platform.h>
#if REX_PLATFORM_ANDROID

#include <cstdlib>
#include <string_view>

#include <android/api-level.h>
#include <sys/system_properties.h>

#include <rex/filesystem.h>
#include <rex/memory/utils.h>
#include <rex/system.h>
#include <rex/thread.h>

namespace rex {

int GetAndroidApiLevel() {
  static const int level = [] {
#if __ANDROID_API__ >= 29
    return android_get_device_api_level();
#else
    // android_get_device_api_level() is only in the API 29+ headers; read the property it reads.
    char value[PROP_VALUE_MAX] = {};
    if (__system_property_get("ro.build.version.sdk", value) > 0) {
      return std::atoi(value);
    }
    return __ANDROID_API__;
#endif
  }();
  return level;
}

bool InitializeAndroidSystemForApplicationContext() {
  memory::AndroidInitialize();
  thread::AndroidInitialize();
  filesystem::AndroidInitialize();
  return true;
}

void ShutdownAndroidSystem() {
  filesystem::AndroidShutdown();
  thread::AndroidShutdown();
  memory::AndroidShutdown();
}

namespace filesystem {

void AndroidInitialize() {}
void AndroidShutdown() {}

bool IsAndroidContentUri(const std::string_view source) {
  return source.starts_with("content://");
}

int OpenAndroidContentFileDescriptor(const std::string_view, const char*) {
  // Not supported: pass /proc/self/fd/N paths instead (see the file comment).
  return -1;
}

}  // namespace filesystem
}  // namespace rex

#endif  // REX_PLATFORM_ANDROID
