# rexui

# Core UI sources
set(REXUI_CORE_SOURCES
    graphics_upload_buffer_pool.cpp
    graphics_util.cpp
    image_decode.cpp
    imgui_dialog.cpp
    imgui_drawer.cpp
    immediate_drawer.cpp
    keybinds.cpp
    menu_item.cpp
    presenter.cpp
    renderdoc_api.cpp
    sdl_virtual_key.cpp
    window.cpp
    window_sdl.cpp
    windowed_app.cpp
    windowed_app_context.cpp
    windowed_app_context_sdl.cpp
)

# Optional overlay dialogs
set(REXUI_OVERLAY_SOURCES
    overlay/achievement_icon_cache.cpp
    overlay/achievement_toast.cpp
    overlay/achievements_overlay.cpp
    overlay/debug_overlay.cpp
    overlay/console_overlay.cpp
    overlay/console_commands.cpp
    overlay/settings_overlay.cpp
)

# Vulkan UI backend
set(REXUI_VULKAN_SOURCES
    vulkan/linked_type_descriptor_set_allocator.cpp
    vulkan/vulkan_moltenvk.cpp
    vulkan/single_layout_descriptor_set_pool.cpp
    vulkan/spirv_tools_context.cpp
    vulkan/ui_samplers.cpp
    vulkan/vulkan_device.cpp
    vulkan/vulkan_immediate_drawer.cpp
    vulkan/vulkan_instance.cpp
    vulkan/streamline.cpp
    vulkan/fsr.cpp
    vulkan/gpu_quirks.cpp
    vulkan/gpu_quirks_apply.cpp
    vulkan/vulkan_mem_alloc.cpp
    vulkan/vulkan_presenter.cpp
    vulkan/vulkan_provider.cpp
    vulkan/vulkan_submission_tracker.cpp
    vulkan/vulkan_upload_buffer_pool.cpp
    vulkan/vulkan_util.cpp
)

# D3D12 UI backend
if(REXGLUE_USE_D3D12)
    set(REXUI_D3D12_SOURCES
        d3d12/d3d12_cpu_descriptor_pool.cpp
        d3d12/d3d12_descriptor_heap_pool.cpp
        d3d12/d3d12_immediate_drawer.cpp
        d3d12/d3d12_presenter.cpp
        d3d12/d3d12_provider.cpp
        d3d12/d3d12_submission_tracker.cpp
        d3d12/d3d12_upload_buffer_pool.cpp
        d3d12/d3d12_util.cpp
    )
endif()

# Platform-specific sources (windowing itself is SDL3, platform-independent)
if(WIN32)
    set(REXUI_PLATFORM_SOURCES
        surface_win.cpp
    )
elseif(APPLE)
    set(REXUI_PLATFORM_SOURCES
        surface_mac.cpp
    )
elseif(ANDROID)
    # wwe2k14 Android port: AndroidNativeWindowSurface is header-only (rex/ui/surface_android.h).
    set(REXUI_PLATFORM_SOURCES)
else()
    set(REXUI_PLATFORM_SOURCES
        surface_gnulinux.cpp
    )
endif()

# Combine sources
set(REXUI_SOURCES
    ${REXUI_CORE_SOURCES}
    ${REXUI_OVERLAY_SOURCES}
    ${REXUI_PLATFORM_SOURCES}
)

if(REXGLUE_USE_VULKAN)
    list(APPEND REXUI_SOURCES ${REXUI_VULKAN_SOURCES})
endif()

if(REXGLUE_USE_D3D12)
    list(APPEND REXUI_SOURCES ${REXUI_D3D12_SOURCES})
endif()

add_library(rexui OBJECT ${REXUI_SOURCES})
add_library(rex::ui ALIAS rexui)

rex_target_pch(rexui <rex/math.h>)

target_include_directories(rexui PUBLIC
    $<BUILD_INTERFACE:${REXGLUE_ROOT}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(rexui PRIVATE
    ${PROJECT_BINARY_DIR}/include  # Generated macOS Vulkan path data
    ${REXGLUE_ROOT}/src  # For ui/shaders/bytecode includes
    ${REXGLUE_ROOT}/thirdparty/stb  # stb_image.h for icon decoding
)

target_link_libraries(rexui PUBLIC
    rexcore
    renderdoc
    imgui::imgui
    SDL3::SDL3
)
target_link_libraries(rexui PRIVATE utf8cpp)

if(REXGLUE_FIDELITYFX_SOURCE_DIR AND EXISTS "${REXGLUE_FIDELITYFX_SOURCE_DIR}/sdk/include")
    target_include_directories(rexui PRIVATE
        "${REXGLUE_FIDELITYFX_SOURCE_DIR}/sdk/include"
        "${REXGLUE_FIDELITYFX_SOURCE_DIR}/ffx-api/include"
    )
    target_compile_definitions(rexui PRIVATE REX_HAS_FIDELITYFX_SDK=1)
endif()

# wwe2k14 patch (Session 18, DLAA): NVIDIA Streamline headers from the user's own download next to the SDK source
# (thirdparty/streamline/sdk, not part of the project); the runtime DLLs are loaded at run time from a folder the
# user names (vulkan_streamline_dir). Without the headers, streamline.cpp builds as stubs.
set(REXUI_STREAMLINE_DIR "${REXGLUE_ROOT}/../streamline/sdk" CACHE PATH "NVIDIA Streamline SDK (headers only)")
if(EXISTS "${REXUI_STREAMLINE_DIR}/include/sl.h")
    target_include_directories(rexui PRIVATE "${REXUI_STREAMLINE_DIR}/include")
    target_compile_definitions(rexui PRIVATE REX_HAS_STREAMLINE=1)
    message(STATUS "rexui: Streamline headers found at ${REXUI_STREAMLINE_DIR}/include (DLAA support)")
endif()

# wwe2k14 patch (Session 33, FSR): AMD FidelityFX ffx-api headers (MIT) from thirdparty/FidelityFX-SDK next to the
# SDK source; the runtime (amd_fidelityfx_vk.dll, built from the same SDK) is loaded at run time from the exe folder.
# Without the headers, fsr.cpp builds as stubs.
set(REXUI_FFX_API_DIR "${REXGLUE_ROOT}/../FidelityFX-SDK/ffx-api" CACHE PATH "AMD FidelityFX ffx-api (headers only)")
if(WIN32 AND EXISTS "${REXUI_FFX_API_DIR}/include/ffx_api/ffx_api.h")
    target_include_directories(rexui PRIVATE "${REXUI_FFX_API_DIR}/include")
    target_compile_definitions(rexui PRIVATE REX_HAS_FFX_API=1)
    message(STATUS "rexui: FidelityFX ffx-api headers found at ${REXUI_FFX_API_DIR}/include (FSR 3 support)")
endif()

set(REXUI_HAS_FIDELITYFX_RUNTIME OFF)
if(TARGET amd_fidelityfx_vk)
    target_link_libraries(rexui PRIVATE amd_fidelityfx_vk)
    set(REXUI_HAS_FIDELITYFX_RUNTIME ON)
endif()
if(TARGET amd_fidelityfx_dx12)
    target_link_libraries(rexui PRIVATE amd_fidelityfx_dx12)
    set(REXUI_HAS_FIDELITYFX_RUNTIME ON)
endif()
if(REXUI_HAS_FIDELITYFX_RUNTIME)
    target_compile_definitions(rexui PRIVATE REX_HAS_FIDELITYFX_RUNTIME=1)
endif()

if(REXGLUE_USE_VULKAN)
    target_link_libraries(rexui PUBLIC
        Vulkan::Headers
        GPUOpen::VulkanMemoryAllocator
        spirv-tools-headers
    )
endif()

# Platform-specific dependencies
if(ANDROID)
    # wwe2k14 (Android port): ANativeWindow (libandroid) and logcat (liblog); no X11 / Wayland.
    if(NOT TARGET adrenotools)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/libadrenotools ${CMAKE_BINARY_DIR}/libadrenotools)
    endif()
    target_link_libraries(rexui PUBLIC android log adrenotools)
elseif(UNIX AND NOT APPLE AND NOT REXGLUE_HOST_TOOLS_ONLY)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(X11_XCB REQUIRED x11-xcb)
    # Required, not optional: vulkan/api.h pulls in <wayland-client.h> on every
    # Linux build.
    pkg_check_modules(WAYLAND REQUIRED wayland-client)

    target_include_directories(rexui PUBLIC
        ${X11_XCB_INCLUDE_DIRS}
        ${WAYLAND_INCLUDE_DIRS}
    )
    target_link_libraries(rexui PUBLIC
        ${X11_XCB_LIBRARIES}
        ${WAYLAND_LIBRARIES}
    )
elseif(APPLE)
    target_link_libraries(rexui PRIVATE "-framework CoreFoundation")
endif()

if(WIN32)
    target_link_libraries(rexui PUBLIC dxgi dxguid)
endif()

# D3D12 backend dependencies
if(REXGLUE_USE_D3D12)
    target_link_libraries(rexui PUBLIC dxc-headers)
endif()

