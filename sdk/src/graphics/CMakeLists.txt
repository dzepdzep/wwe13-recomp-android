# rexgpu-xenos: Xenos GPU emulation plugin (ported from Xenia)
# Namespace: rex::graphics
#
# Runtime-loaded via rex::system::LoadGpuPlugin; consumers never link this
# target. The Xenos format layer is compiled into rexcore.

set(REXGPU_XENOS_SOURCES
    plugin_main.cpp

    # Core GPU types
    flags.cpp
    register_file.cpp
    registers.cpp
    pipeline/shader/shader.cpp
    format/ucode.cpp
    sampler_info.cpp

    # GPU system
    graphics_system.cpp
    command_processor.cpp
    d3d_copy.cpp  # wwe2k14 patch (Session 87, D3D layer F2)
    d3d_inline_draw.cpp
    d3d_noring.cpp  # wwe2k14 patch (Session 87, D3D layer stage 2)
    d3d_tex.cpp  # wwe2k14 patch (Session 87, D3D layer stage 4, textures)
    job_pool.cpp  # wwe2k14 patch (Session 95, item N1): the GPU job pool
    half2.cpp
    half2_log.cpp
    util/draw_extent_estimator.cpp
    util/draw.cpp
    packet_disassembler.cpp
    primitive_processor.cpp
    pipeline/render_target/cache.cpp
    shared_memory.cpp
    pipeline/texture/cache.cpp
    pipeline/shader/interpreter.cpp
    pipeline/shader/translator.cpp
    pipeline/shader/translation_store.cpp  # wwe2k14 patch (Session 92, M6)
    pipeline/shader/translator_disasm.cpp
)

# SPIR-V shader translation
set(REXGPU_XENOS_SPIRV_SOURCES
    pipeline/shader/spirv_builder.cpp
    pipeline/shader/spirv.cpp
    pipeline/shader/spirv_relaxed_precision.cpp
    pipeline/shader/spirv_translator.cpp
    pipeline/shader/spirv_translator_alu.cpp
    pipeline/shader/spirv_translator_fetch.cpp
    pipeline/shader/spirv_translator_memexport.cpp
    pipeline/shader/spirv_translator_rb.cpp
)

# Vulkan backend
set(REXGPU_XENOS_VULKAN_SOURCES
    vulkan/graphics_system.cpp
    vulkan/deferred_command_buffer.cpp
    vulkan/command_processor.cpp
    vulkan/disk_pipeline_cache.cpp  # wwe2k14 patch (Session 63)
    vulkan/pipeline_cache.cpp
    vulkan/m6_precompile.cpp  # wwe2k14 patch (Session 92, M6)
    vulkan/primitive_processor.cpp
    vulkan/render_target_cache.cpp
    vulkan/shader.cpp
    vulkan/shared_memory.cpp
    vulkan/texture_cache.cpp
    vulkan/native/native_renderer.cpp
    vulkan/native/native_truth.cpp
    vulkan/native/inline_ring.cpp
    vulkan/native/index_pool.cpp  # wwe2k14 patch (Session 95, Lane B1): the index pool (was vertex_mirror.cpp)
    vulkan/native/skin_once.cpp  # wwe2k14 patch (Session 95): each wrestler vertex skinned once a frame (gpu_native_skin_once)
    vulkan/native/vertex_input.cpp  # wwe2k14 patch (Session 95): the attribute draws' state and [vinput] line (gpu_native_vertex_input)
    vulkan/native/vertex_mirror.cpp  # wwe2k14 patch (Session 95): the host-order vertex pool, gpu_native_skin_once's source only
    vulkan/native/own_crowd.cpp
    vulkan/native/owned_copy.cpp
    vulkan/native/owned_copy_lean.cpp
    vulkan/native/owned_restore_skip.cpp  # wwe2k14 patch (Session 95): gpu_owned_restore_skip
    vulkan/native/owned_graph.cpp  # wwe2k14 patch (Session 95, item 4: frame graph lite)
    vulkan/native/owned_decouple.cpp  # wwe2k14 patch (Session 94, idea 1 phase 1A: gpu_d3d_decouple)
    vulkan/native/decouple_blend.cpp  # wwe2k14 patch (Session 94, phase 1B): the draw match across ticks + the blended view 0 constants
    vulkan/native/owned_entry.cpp  # wwe2k14 patch (Session 92, M2b)
    vulkan/native/owned_feedback.cpp
    vulkan/native/owned_shadow_skip.cpp  # wwe2k14 patch (Session 95, item K): gpu_native_owned_shadow_skip
    vulkan/native/owned_post.cpp
    vulkan/native/owned_reader.cpp
    vulkan/native/owned_scene.cpp
    vulkan/native/owned_start.cpp
    vulkan/native/owned_steps.cpp
    vulkan/native/d3d_copy_records.cpp  # wwe2k14 patch (Session 87, D3D layer F2)
    vulkan/native/owned_resolve.cpp  # wwe2k14 patch (Session 92, M3a)
    vulkan/native/owned_record.cpp  # wwe2k14 patch (Session 92, M4a)
    vulkan/native/owned_scout.cpp  # wwe2k14 patch (Session 95, item N2): the scout + cover / bounds jobs (gpu_jobs)
    vulkan/native/diag_layer.cpp  # wwe2k14 patch (Session 95): the diagnostic run's GPU side (gpu_diag_layer)
    vulkan/native/diag_spirv.cpp  # wwe2k14 patch (Session 95): its SPIR-V passes
    vulkan/native/phase0_census.cpp  # wwe2k14 patch (Session 93): the log-only phase 0 census (gpu_d3d_phase0_census)
    vulkan/native/draw_bounds.cpp  # wwe2k14 patch (Session 93): mesh bounds + the box test (census, cull_ours)
    vulkan/native/cull_ours.cpp  # wwe2k14 patch (Session 93): gpu_d3d_cull_ours, our off-screen draws are not drawn
    vulkan/native/owned_front_own.cpp  # wwe2k14 patch (Session 92, M4f-C): the typed draw + the cut's entries
    vulkan/native/owned_front_notify.cpp  # wwe2k14 patch (Session 92, M4f-C): the notifications + the cut's processor effects (out of d3d_feed.cpp)
    vulkan/native/owned_front_draw.cpp  # wwe2k14 patch (Session 92, M4b)
    vulkan/native/draw_slot.cpp  # wwe2k14 patch (Session 95): gpu_d3d_draw_slot
    vulkan/native/owned_front_bind.cpp  # wwe2k14 patch (Session 92, M4c)
    vulkan/native/owned_wb_objects.cpp  # wwe2k14 patch (Session 92, M2d)
    vulkan/native/owned_claim.cpp  # wwe2k14 patch (Session 92, M2d)
    vulkan/native/owned_tno.cpp  # wwe2k14 patch (Session 92, M2d, empty unless REX_TRUE_NATIVE_ONLY)
    vulkan/native/shader_swap.cpp
)

# D3D12 backend
if(REXGLUE_USE_D3D12)
    # DXBC third-party dependency
    set(REXGPU_XENOS_DXBC_THIRDPARTY_SOURCES
        ${REXGLUE_ROOT}/thirdparty/dxbc/DXBCChecksum.cpp
    )

    # DXBC shader translation (required for D3D12)
    set(REXGPU_XENOS_DXBC_SOURCES
        pipeline/shader/dxbc.cpp
        pipeline/shader/dxbc_translator.cpp
        pipeline/shader/dxbc_translator_alu.cpp
        pipeline/shader/dxbc_translator_fetch.cpp
        pipeline/shader/dxbc_translator_memexport.cpp
        pipeline/shader/dxbc_translator_om.cpp
    )

    # D3D12 GPU backend
    set(REXGPU_XENOS_D3D12_SOURCES
        d3d12/graphics_system.cpp
        d3d12/command_processor.cpp
        d3d12/primitive_processor.cpp
        d3d12/render_target_cache.cpp
        d3d12/shader.cpp
        d3d12/shared_memory.cpp
        d3d12/texture_cache.cpp
        d3d12/deferred_command_list.cpp
        d3d12/pipeline_cache.cpp
    )
endif()

if(REXGLUE_USE_VULKAN)
    list(APPEND REXGPU_XENOS_SOURCES ${REXGPU_XENOS_SPIRV_SOURCES} ${REXGPU_XENOS_VULKAN_SOURCES})
    # wwe2k14 patch (Session 92, M4f-C): the old record-fed draw path - the D3D feed (register image) and the direct lean handler - is left out of the TRUE NATIVE proof build
    if(NOT REX_TRUE_NATIVE_ONLY)
        list(APPEND REXGPU_XENOS_SOURCES vulkan/native/d3d_direct.cpp vulkan/native/d3d_feed.cpp)
    endif()
endif()

if(REXGLUE_USE_D3D12)
    list(APPEND REXGPU_XENOS_SOURCES
        ${REXGPU_XENOS_DXBC_THIRDPARTY_SOURCES}
        ${REXGPU_XENOS_DXBC_SOURCES}
        ${REXGPU_XENOS_D3D12_SOURCES})
endif()

add_library(rexgpu-xenos SHARED ${REXGPU_XENOS_SOURCES})
add_library(rex::gpu-xenos ALIAS rexgpu-xenos)

set_target_properties(rexgpu-xenos PROPERTIES
    EXPORT_NAME gpu-xenos
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(rexgpu-xenos PRIVATE
    ${REXGLUE_ROOT}/thirdparty/renderdoc
    ${REXGLUE_ROOT}  # For thirdparty/dxbc/DXBCChecksum.h include
)

target_compile_options(rexgpu-xenos PRIVATE -Wno-unused-variable)

rex_target_pch(rexgpu-xenos
    <mutex>
    <unordered_map>
    <rex/math.h>
    <rex/memory.h>
    <rex/thread/mutex.h>
    <rex/system/xmemory.h>
    <rex/graphics/xenos.h>
    <rex/graphics/registers.h>
    <rex/graphics/format/ucode.h>
    <rex/graphics/pipeline/shader/shader.h>
)

if(REXGPU_XENOS_DXBC_THIRDPARTY_SOURCES)
    set_source_files_properties(${REXGPU_XENOS_DXBC_THIRDPARTY_SOURCES}
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
endif()

# Everything PRIVATE: the plugin has no consumers. Never link the OBJECT
# libraries (rexcore/rexui) here; rexruntime already contains those objects
# and relinking them would duplicate global state.
target_link_libraries(rexgpu-xenos PRIVATE
    rexruntime
    xxHash::xxhash
)

if(REXGLUE_USE_VULKAN)
    target_link_libraries(rexgpu-xenos PRIVATE
        Vulkan::Headers
        glslang::SPIRV
        GPUOpen::VulkanMemoryAllocator
        glslang
        MachineIndependent
        GenericCodeGen
        OSDependent
        OGLCompiler
    )
    # wwe2k14 patch (Session 61, vulkan_spirv_optimize): SPIRV-Tools' optimiser, built in (thirdparty/CMakeLists.txt).
    if(TARGET SPIRV-Tools-opt)
        target_link_libraries(rexgpu-xenos PRIVATE SPIRV-Tools-opt)
        target_compile_definitions(rexgpu-xenos PRIVATE REX_HAS_SPIRV_OPT=1)
    endif()
endif()

if(REXGLUE_USE_D3D12)
    target_link_libraries(rexgpu-xenos PRIVATE
        d3d12
        dxgi
        dxguid
    )
endif()
