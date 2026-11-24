/**
 ******************************************************************************
 * ReXGlue runtime - custom (XMP) music playback
 *
 * Decodes Xbox 360 "user playlist" songs (the files a title enumerates from
 * its music folder) and mixes them into the host audio output.
 ******************************************************************************
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rex::audio {

// Decoded PCM, always stereo float at the host output rate.
struct CustomMusicTrack {
  std::vector<float> samples;  // interleaved LR
  uint32_t sample_rate = 0;    // rate of the source file
  uint32_t channels = 0;       // source channel count
  uint64_t duration_ms = 0;
  std::string decoder;         // "dr_mp3" / "dr_flac" / "dr_wav" / "stb_vorbis"
};

// Decodes a file to stereo float PCM at `target_rate`. Runs on whatever
// thread calls it (the caller puts it on a worker). Returns nullptr and fills
// `error` on failure.
std::unique_ptr<CustomMusicTrack> DecodeCustomMusicFile(const std::filesystem::path& path,
                                                        uint32_t target_rate, std::string* error);

  // Plays at most one custom song at a time, mixed on top of guest audio.
  //
  // Threading: the public methods are called from the XMP app (guest thread) and
  // the disk/SDL audio callback. The worker thread owns decoding; Play() only
  // hands the request over (never decodes on the guest thread) and the mix
  // callback only reads the lock-protected decoded samples.
class CustomMusicPlayer {
 public:
  static CustomMusicPlayer& instance();

  void Shutdown();

  // Hands the decode request to the worker; decoding never runs on the
  // calling (guest) thread. Returns false only if the player is shutting down
  // or the request could not be queued; a decode failure is reported later in
  // the log and surfaces as the state going back to idle.
  bool Play(const std::filesystem::path& path, const std::string& name);
  void Stop();
  void Pause();
  void Continue();

  bool IsPlaying() const { return state_.load(std::memory_order_relaxed) == kPlaying; }
  bool IsPaused() const { return state_.load(std::memory_order_relaxed) == kPaused; }
  // True while decoded samples are being published/played (or loading). Lets
  // the caller tell "Play() accepted" from "Play() rejected because shutdown".
  bool IsActive() const {
    uint32_t state = state_.load(std::memory_order_relaxed);
    return state != kIdle && state != kFinished;
  }

  void SetVolume(float volume) {
    volume_.store(std::clamp(volume, 0.0f, 1.0f), std::memory_order_relaxed);
  }
  float volume() const { return volume_.load(std::memory_order_relaxed); }

  const std::string& name() const { return name_; }
  const std::string& decoder() const { return decoder_; }
  uint32_t sample_rate() const { return sample_rate_.load(std::memory_order_relaxed); }
  uint64_t duration_ms() const { return duration_ms_.load(std::memory_order_relaxed); }

  // Mixes up to `frames` stereo frames from the song into `output` (interleaved
  // at `output_rate`, `output_channels` channels per frame). Called from the
  // audio device callback. Adds the song on top of whatever is already in the
  // buffer; does nothing when idle. `output_channels` must be >= 2; the song
  // goes to front-left/front-right.
  void MixFrames(float* output, uint32_t frames, uint32_t output_rate, uint32_t output_channels);

 private:
  enum State : uint32_t {
    kIdle = 0,
    kLoading = 3,
    kPlaying = 1,
    kPaused = 2,
    kFinished = 4,
  };

  CustomMusicPlayer() = default;
  ~CustomMusicPlayer();
  CustomMusicPlayer(const CustomMusicPlayer&) = delete;
  CustomMusicPlayer& operator=(const CustomMusicPlayer&) = delete;

  void WorkerMain();

  mutable std::mutex mutex_;
  std::thread worker_;
  bool worker_running_ = false;
  std::atomic<uint32_t> state_{kIdle};

  // Request handed to the worker.
  std::filesystem::path request_path_;
  std::string request_name_;
  bool request_pending_ = false;
  // Bumped by every Play()/Stop() so the worker can drop the result of a
  // decode that has been superseded while it was running.
  uint64_t request_serial_ = 0;

  // Decoded samples (owned by the worker thread object, read under mutex_).
  std::shared_ptr<CustomMusicTrack> track_;
  size_t cursor_ = 0;  // in track samples

  std::atomic<float> volume_{1.0f};
  std::atomic<uint32_t> sample_rate_{0};  // source file rate (status/log only)
  // Host device rate the track is resampled to; learned from MixFrames.
  std::atomic<uint32_t> output_rate_{48000};
  std::atomic<uint64_t> duration_ms_{0};
  std::string name_;
  std::string decoder_;

  std::atomic<bool> shutdown_{false};
  std::atomic<uint64_t> playback_generation_{0};
};

}  // namespace rex::audio
