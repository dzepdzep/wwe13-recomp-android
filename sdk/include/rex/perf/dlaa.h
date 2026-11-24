/**
 * @file        perf/dlaa.h
 * @brief       The game camera of each frame, from the app to the GPU plugin (wwe2k14 patch, Session 18, DLAA)
 *
 * The Vulkan command processor registers a sink at SetupContext; the app calls SubmitCamera at the game's camera
 * hand-off (any thread). Exported from rexruntime, like perf/flight.h, so the GPU plugin and the app share it
 * without the app including the command processor's headers.
 */
#pragma once

#include <cstdint>

namespace rex::perf::dlaa {

// view: world -> view, row vectors (x right, y down, z forward); fov in degrees; frame = guest swaps before it.
using CameraFn = void (*)(void* user, uint64_t frame, const float view[16], float fov_degrees, float near_z,
                          float far_z);

// nullptr unregisters (only if `user` matches the registered one).
void SetCameraSink(CameraFn fn, void* user);
// Forwards to the sink; false if none is registered.
bool SubmitCamera(uint64_t frame, const float view[16], float fov_degrees, float near_z, float far_z);

}  // namespace rex::perf::dlaa
