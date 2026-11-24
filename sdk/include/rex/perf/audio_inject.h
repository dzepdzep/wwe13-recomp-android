/**
 * @file        perf/audio_inject.h
 * @brief       Add the app's own stereo audio to the guest's mix (wwe2k14 patch, Session 101 lane 288)
 *
 * The SDL audio driver calls Inject from SubmitFrame (the guest's audio thread), on the host copy of the frame the
 * guest submitted, BEFORE the tap (perf/audio_tap.h), the stereo fold, the output gain and `audio_mute`. So the app's
 * music is part of the game's mix: it is captured by the tap, silenced by `audio_mute`, and it stops while the audio
 * system is paused (no guest frames are submitted then, so the callback is not called).
 *
 * The callback is given two zeroed host-endian float blocks (L, R), `samples` long (256 on the 360 path), at
 * `frequency` Hz, and ADDS what it has into them. It returns true when it wrote anything. The SDK then adds the blocks
 * into the frame's FL and FR planes (big-endian floats), unclamped (the fold clamps). The callback runs on the
 * audio thread: it must not allocate, lock or block (a lock-free ring read is the intended use). With no callback
 * registered Inject costs one atomic load. Exported from rexruntime, like perf/audio_tap.h.
 */
#pragma once

#include <cstdint>

namespace rex::perf::audio {

using InjectFn = bool (*)(void* user, float* left, float* right, uint32_t samples, uint32_t frequency);

// Register the (single) injector. nullptr unregisters, only if `user` matches the registered one. The first registration
// is safe at any time; unregister / re-register only while the guest is not mixing (the audio thread reads the function
// and its user pointer without a lock).
void SetInject(InjectFn fn, void* user);

// `frame_be`: `channels` planes of `samples` big-endian floats, FL then FR first. Returns true if anything was added.
bool Inject(void* frame_be, uint32_t samples, uint32_t channels, uint32_t frequency);

}  // namespace rex::perf::audio