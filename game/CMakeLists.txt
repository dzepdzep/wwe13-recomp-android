cmake_minimum_required(VERSION 3.25)
project(wwe13 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cross builds: run codegen with the host rexglue (tools/host_rexglue.sh), not the Android one.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/generated/rexglue.cmake" _rexglue_cmake)
if(ANDROID)
  string(REPLACE "$<TARGET_FILE:rex::rexglue> codegen" "${CMAKE_CURRENT_SOURCE_DIR}/tools/host_rexglue.sh codegen" _rexglue_cmake "${_rexglue_cmake}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rexglue_host.cmake" "${_rexglue_cmake}")
include("${CMAKE_CURRENT_BINARY_DIR}/rexglue_host.cmake")

find_package(Git QUIET)
set(WWE13_BUILD_COMMIT unknown)
set(WWE13_BUILD_DIRTY unknown)
set(WWE13_SDK_COMMIT unknown)
set(WWE13_SDK_DIRTY unknown)
if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE WWE13_BUILD_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE WWE13_GIT_RESULT ERROR_QUIET)
  if(NOT WWE13_GIT_RESULT EQUAL 0)
    set(WWE13_BUILD_COMMIT unknown)
  else()
    execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE WWE13_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE WWE13_GIT_STATUS_RESULT ERROR_QUIET)
    if(WWE13_GIT_STATUS_RESULT EQUAL 0)
      if(WWE13_GIT_STATUS STREQUAL "")
        set(WWE13_BUILD_DIRTY no)
      else()
        set(WWE13_BUILD_DIRTY yes)
      endif()
    endif()
  endif()

  if(NOT DEFINED WWE13_SDK_ROOT OR WWE13_SDK_ROOT STREQUAL "")
    set(WWE13_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../tools/rexglue")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${WWE13_SDK_ROOT}"
    OUTPUT_VARIABLE WWE13_SDK_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE WWE13_SDK_GIT_RESULT ERROR_QUIET)
  if(WWE13_SDK_GIT_RESULT EQUAL 0)
    execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain
      WORKING_DIRECTORY "${WWE13_SDK_ROOT}"
      OUTPUT_VARIABLE WWE13_SDK_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE WWE13_SDK_GIT_STATUS_RESULT ERROR_QUIET)
    if(WWE13_SDK_GIT_STATUS_RESULT EQUAL 0)
      if(WWE13_SDK_GIT_STATUS STREQUAL "")
        set(WWE13_SDK_DIRTY no)
      else()
        set(WWE13_SDK_DIRTY yes)
      endif()
    endif()
  endif()
endif()
include(cmake/wwe13_version.cmake)
string(TIMESTAMP WWE13_BUILD_DATE "%Y-%m-%dT%H:%M:%SZ" UTC)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(src/wwe13_build_info.h.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/wwe13_build_info.h" @ONLY)

set(WWE13_SOURCES
  src/main.cpp
  src/game_version_check.cpp
  src/present_interval.cpp
  src/crowd_options.cpp
  src/heap_diag.cpp
  src/training.cpp
  src/crash_log.cpp
  src/pac_diag.cpp
  src/audio_capture.cpp
  src/unlock_all.cpp
  src/roster_mods.cpp
  src/mods_hint.cpp
  src/cloth_handshake.cpp
  src/scene_resolution.cpp
  src/scene_aa.cpp
  src/shader_cache_seed.cpp
  src/native_settings.cpp
  src/coherency_stub.cpp
  src/dlist_tripwire.cpp
  src/touch_pad.cpp
  src/ps_prompts.cpp
  src/d3d_layer_glue.cpp
  src/tiling_hooks.cpp
  src/perf_hint.cpp)
if(ANDROID)
  # Android port: the game is libmain.so, loaded by the SDLActivity-based APK.
  add_library(wwe13 SHARED ${WWE13_SOURCES})
  set_target_properties(wwe13 PROPERTIES OUTPUT_NAME main)
  target_link_options(wwe13 PRIVATE -Wl,--no-undefined -Wl,-Bsymbolic)
  # SDL3 lives in librexruntime.so (one SDL instance per process).
  target_include_directories(wwe13 PRIVATE $<TARGET_PROPERTY:SDL3::SDL3,INTERFACE_INCLUDE_DIRECTORIES>)
else()
  add_executable(wwe13 ${WWE13_SOURCES})
endif()
target_include_directories(wwe13 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
rexglue_setup_target(wwe13)
target_include_directories(wwe13 PRIVATE "${REXSDK_DIR}/thirdparty/imgui")
if(ANDROID)
  target_link_libraries(wwe13 PRIVATE adrenotools)
endif()
if(WIN32)
  target_link_libraries(wwe13 PRIVATE advapi32)
  # Link map next to the exe: symbol -> object resolution for Windows-only link checks.
  target_link_options(wwe13 PRIVATE "LINKER:/map:$<TARGET_FILE_DIR:wwe13>/wwe13.map")
endif()
