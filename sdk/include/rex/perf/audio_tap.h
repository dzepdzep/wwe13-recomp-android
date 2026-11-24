/**
 * @file        perf/audio_tap.h
 * @brief       A copy of each audio frame the guest submits (wwe2k14 patch, Session 18: sound for re-shoots)
 *
 * The SDL audio driver calls Tap from SubmitFrame (the guest's audio thread) with the frame exactly as the guest
 * wrote it: `channels` blocks of `samples` big-endian floats, one block per channel (FL, FR, C, LFE, SL, SR), at
 * `frequency` Hz. The app registers a sink only while it records (default: none, so Tap costs one atomic load).
 * Exported from rexruntime, like perf/flight.h.
 */
#pragma once

#include <cstdint>

namespace rex::perf::audio {

using TapFn = void (*)(void* user, const void* frame_be, uint32_t samples, uint32_t channels, uint32_t frequency);

// nullptr unregisters (only if `user` matches the registered one).
void SetTap(TapFn fn, void* user);
void Tap(const void* frame_be, uint32_t samples, uint32_t channels, uint32_t frequency);

}  // namespace rex::perf::audio
