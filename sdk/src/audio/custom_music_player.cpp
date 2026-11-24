/**
 ******************************************************************************
 * ReXGlue runtime - custom (XMP) music playback
 *
 * See custom_music_player.h. The decoder backends are the single-file
 * libraries vendored under thirdparty/decoders; FFmpeg is not used because
 * the vendored FFmpeg build enables only the XMA/WMA-Pro + MP3 decoders (no
 * FLAC/Vorbis decoders, no OGG/WAV demuxers), and extending it would mean
 * regenerating its per-platform configs and list overlays.
 ******************************************************************************
 */

#include <rex/audio/custom_music_player.h>
#include <rex/perf/audio_inject.h>
#include <vector>
#include <algorithm>

#include <algorithm>
#include <cstring>

#include <rex/filesystem.h>
#include <rex/platform.h>
#include <rex/logging.h>

#include <cstdio>

// The decoder implementations live in thirdparty/decoders/decoder_impl.cpp;
// only the declarations are needed here.
#include <decoders/dr_mp3.h>
#include <decoders/dr_flac.h>
#include <decoders/dr_wav.h>
#define STB_VORBIS_HEADER_ONLY
#include <decoders/stb_vorbis.c>

namespace rex::audio {

namespace {

const char* FormatName(const std::filesystem::path& path) {
  auto extension = rex::path_to_utf8(path.extension());
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (extension == ".mp3") {
    return "mp3";
  }
  if (extension == ".flac") {
    return "flac";
  }
  if (extension == ".wav") {
    return "wav";
  }
  if (extension == ".ogg") {
    return "ogg";
  }
  return "unknown";
}

// A decoded track that can be read one frame at a time regardless of the
// source codec/sample rate. Each backend implements its own reader; the
// player resamples to the host rate once at load time so the mix loop stays
// branch-free.
struct TrackReader {
  virtual ~TrackReader() = default;
  virtual uint32_t channels() const = 0;
  virtual uint32_t sample_rate() const = 0;
  // Reads up to `frames` interleaved frames into out (channels * frames).
  virtual size_t ReadFrames(float* out, size_t frames) = 0;
};

struct Mp3Reader : TrackReader {
  drmp3 decoder{};
  uint32_t rate = 0;
  uint32_t channel_count = 0;
  ~Mp3Reader() override { drmp3_uninit(&decoder); }
  uint32_t channels() const override { return channel_count; }
  uint32_t sample_rate() const override { return rate; }
  size_t ReadFrames(float* out, size_t frames) override {
    return static_cast<size_t>(
        drmp3_read_pcm_frames_f32(&decoder, static_cast<drmp3_uint64>(frames), out));
  }
};

struct FlacReader : TrackReader {
  drflac* decoder = nullptr;
  ~FlacReader() override {
    if (decoder) {
      drflac_close(decoder);
    }
  }
  uint32_t channels() const override { return decoder->channels; }
  uint32_t sample_rate() const override { return decoder->sampleRate; }
  size_t ReadFrames(float* out, size_t frames) override {
    return static_cast<size_t>(
        drflac_read_pcm_frames_f32(decoder, static_cast<drflac_uint64>(frames), out));
  }
};

struct WavReader : TrackReader {
  drwav decoder{};
  ~WavReader() override { drwav_uninit(&decoder); }
  uint32_t channels() const override { return decoder.channels; }
  uint32_t sample_rate() const override { return decoder.sampleRate; }
  size_t ReadFrames(float* out, size_t frames) override {
    return static_cast<size_t>(
        drwav_read_pcm_frames_f32(&decoder, static_cast<drwav_uint64>(frames), out));
  }
};

struct VorbisReader : TrackReader {
  stb_vorbis* decoder = nullptr;
  uint32_t rate = 0;
  uint32_t channel_count = 0;
  ~VorbisReader() override {
    if (decoder) {
      stb_vorbis_close(decoder);
    }
  }
  uint32_t channels() const override { return channel_count; }
  uint32_t sample_rate() const override { return rate; }
  size_t ReadFrames(float* out, size_t frames) override {
    if (channel_count == 0) {
      return 0;
    }
    // stb_vorbis returns interleaved samples, capped at INT_MAX; a chunk per
    // call keeps the return value well inside int range.
    size_t total_frames = 0;
    while (total_frames < frames) {
      size_t chunk = std::min<size_t>(frames - total_frames, 1u << 20);
      int samples = stb_vorbis_get_samples_float_interleaved(
          decoder, static_cast<int>(channel_count), out + total_frames * channel_count,
          static_cast<int>(chunk * channel_count));
      if (samples <= 0) {
        break;
      }
      total_frames += static_cast<size_t>(samples);
      if (static_cast<size_t>(samples) < chunk) {
        break;
      }
    }
    return total_frames;
  }
};

// Linear-interpolating stereo resampler.
void ResampleToStereo(const std::vector<float>& input, uint32_t in_channels, uint32_t in_rate,
                      uint32_t out_rate, std::vector<float>* output) {
  const size_t input_frames = in_channels ? input.size() / in_channels : 0;
  if (input_frames == 0 || in_rate == 0 || out_rate == 0) {
    output->clear();
    return;
  }
  const double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);
  const size_t output_frames = static_cast<size_t>(
      std::max(1.0, static_cast<double>(input_frames) * ratio));
  output->resize(output_frames * 2);

  for (size_t frame = 0; frame < output_frames; ++frame) {
    const double source_position = static_cast<double>(frame) / ratio;
    const size_t index0 = static_cast<size_t>(source_position);
    const size_t index1 = std::min(index0 + 1, input_frames - 1);
    const float fraction = static_cast<float>(source_position - static_cast<double>(index0));

    float left = 0.0f;
    float right = 0.0f;
    for (uint32_t channel = 0; channel < in_channels; ++channel) {
      const float sample0 = input[index0 * in_channels + channel];
      const float sample1 = input[index1 * in_channels + channel];
      const float sample = sample0 + (sample1 - sample0) * fraction;
      if (channel == 0) {
        left = sample;
      } else if (channel == 1) {
        right = sample;
      } else {
        // Fold extra channels symmetrically into both sides.
        left += sample * 0.5f;
        right += sample * 0.5f;
      }
    }
    (*output)[frame * 2] = left;
    (*output)[frame * 2 + 1] = right;
  }
}

std::unique_ptr<TrackReader> OpenReader(const std::filesystem::path& path, std::string* error) {
#if REX_PLATFORM_WIN32
  // Wide paths: song names often contain characters outside the ANSI code page.
  const std::wstring filename = path.wstring();
#define REX_MUSIC_OPEN(fn) fn##_w
#else
  const std::string filename = rex::path_to_utf8(path);
#define REX_MUSIC_OPEN(fn) fn
#endif
  const char* format = FormatName(path);

  if (std::strcmp(format, "mp3") == 0) {
    auto reader = std::make_unique<Mp3Reader>();
    if (!REX_MUSIC_OPEN(drmp3_init_file)(&reader->decoder, filename.c_str(), nullptr)) {
      *error = "dr_mp3 failed to open";
      return nullptr;
    }
    reader->rate = reader->decoder.sampleRate;
    reader->channel_count = reader->decoder.channels;
    return reader;
  }
  if (std::strcmp(format, "flac") == 0) {
    auto reader = std::make_unique<FlacReader>();
    reader->decoder = REX_MUSIC_OPEN(drflac_open_file)(filename.c_str(), nullptr);
    if (!reader->decoder) {
      *error = "dr_flac failed to open";
      return nullptr;
    }
    return reader;
  }
  if (std::strcmp(format, "wav") == 0) {
    auto reader = std::make_unique<WavReader>();
    if (!REX_MUSIC_OPEN(drwav_init_file)(&reader->decoder, filename.c_str(), nullptr)) {
      *error = "dr_wav failed to open";
      return nullptr;
    }
    return reader;
  }
  if (std::strcmp(format, "ogg") == 0) {
    auto reader = std::make_unique<VorbisReader>();
    int vorbis_error = 0;
#if REX_PLATFORM_WIN32
    FILE* file = _wfopen(filename.c_str(), L"rb");
    reader->decoder = file ? stb_vorbis_open_file(file, 1, &vorbis_error, nullptr) : nullptr;
    if (file && !reader->decoder) {
      std::fclose(file);
    }
#else
    reader->decoder = stb_vorbis_open_filename(filename.c_str(), &vorbis_error, nullptr);
#endif
    if (!reader->decoder) {
      *error = "stb_vorbis failed to open (" + std::to_string(vorbis_error) + ")";
      return nullptr;
    }
    stb_vorbis_info info = stb_vorbis_get_info(reader->decoder);
    reader->rate = info.sample_rate;
    reader->channel_count = info.channels;
    return reader;
  }

  *error = std::string("unsupported extension ") + format;
  return nullptr;
}
#undef REX_MUSIC_OPEN

}  // namespace

std::unique_ptr<CustomMusicTrack> DecodeCustomMusicFile(const std::filesystem::path& path,
                                                        uint32_t target_rate, std::string* error) {
  auto reader = OpenReader(path, error);
  if (!reader) {
    return nullptr;
  }

  const uint32_t channels = reader->channels();
  const uint32_t rate = reader->sample_rate();
  if (channels == 0 || rate == 0) {
    *error = "decoder reported zero channels or sample rate";
    return nullptr;
  }

  std::vector<float> pcm;
  const size_t kChunkFrames = 1u << 14;
  std::vector<float> chunk(kChunkFrames * channels);
  for (;;) {
    const size_t frames = reader->ReadFrames(chunk.data(), kChunkFrames);
    if (frames == 0) {
      break;
    }
    pcm.insert(pcm.end(), chunk.begin(), chunk.begin() + frames * channels);
  }

  if (pcm.empty()) {
    *error = "decoded zero samples";
    return nullptr;
  }

  auto track = std::make_unique<CustomMusicTrack>();
  track->sample_rate = rate;
  track->channels = channels;
  ResampleToStereo(pcm, channels, rate, target_rate, &track->samples);
  const uint64_t frames = track->samples.size() / 2;
  track->duration_ms = frames * 1000 / target_rate;

  // Name the backend instead of the container for the log.
  const char* format = FormatName(path);
  if (std::strcmp(format, "ogg") == 0) {
    track->decoder = "stb_vorbis";
  } else if (std::strcmp(format, "flac") == 0) {
    track->decoder = "dr_flac";
  } else if (std::strcmp(format, "wav") == 0) {
    track->decoder = "dr_wav";
  } else if (std::strcmp(format, "mp3") == 0) {
    track->decoder = "dr_mp3";
  } else {
    track->decoder = format;
  }
  return track;
}

CustomMusicPlayer& CustomMusicPlayer::instance() {
  static CustomMusicPlayer player;
  return player;
}

CustomMusicPlayer::~CustomMusicPlayer() { Shutdown(); }

void CustomMusicPlayer::Shutdown() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!worker_running_) {
      return;
    }
    shutdown_.store(true, std::memory_order_relaxed);
  }
  // Wake the worker out of any wait and join it.
  if (worker_.joinable()) {
    worker_.join();
  }
  std::lock_guard<std::mutex> guard(mutex_);
  worker_running_ = false;
  track_.reset();
}

void CustomMusicPlayer::WorkerMain() {
  for (;;) {
    std::filesystem::path path;
    std::string name;
    uint64_t serial = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (!shutdown_.load(std::memory_order_relaxed) && !request_pending_) {
        // Short slices: the guest expects the song to start right after the
        // play request, so the worst-case handoff delay stays small.
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        lock.lock();
      }
      if (shutdown_.load(std::memory_order_relaxed)) {
        return;
      }
      path = request_path_;
      name = request_name_;
      serial = request_serial_;
      request_pending_ = false;
      state_.store(kLoading, std::memory_order_relaxed);
    }

    std::string error;
    std::unique_ptr<CustomMusicTrack> track;
    try {
      track = DecodeCustomMusicFile(path, output_rate_.load(std::memory_order_relaxed), &error);
    } catch (const std::exception& e) {
      error = std::string("exception: ") + e.what();
    }
    if (!track) {
      REXLOG_ERROR("custom music: failed to decode '{}': {}", rex::path_to_utf8(path), error);
      std::lock_guard<std::mutex> guard(mutex_);
      // A newer request may have arrived while decoding; only publish the
      // failure if it did not.
      if (!request_pending_ && serial == request_serial_) {
        state_.store(kIdle, std::memory_order_relaxed);
      }
      continue;
    }

    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (shutdown_.load(std::memory_order_relaxed)) {
        return;
      }
      if (request_pending_ || serial != request_serial_) {
        // A newer song (or a stop) is waiting; the next loop iteration handles
        // it and the stale decode is dropped.
        continue;
      }
      name_ = name;
      decoder_ = track->decoder;
      sample_rate_.store(track->sample_rate, std::memory_order_relaxed);
      duration_ms_.store(track->duration_ms, std::memory_order_relaxed);
      REXLOG_INFO("custom music: play '{}' path={} decoder={} source_rate={}Hz duration={}ms",
                  name, rex::path_to_utf8(path), track->decoder, track->sample_rate, track->duration_ms);
      track_ = std::move(track);
      cursor_ = 0;
      state_.store(kPlaying, std::memory_order_relaxed);
    }
  }
}

namespace {
// WWE13 (jsdk port): the songs join the guest's mix through the audio inject registry (perf/audio_inject.h) instead
// of the old SDK's driver hook.
bool InjectCustomMusic(void* user, float* left, float* right, uint32_t samples, uint32_t frequency) {
  auto* player = static_cast<CustomMusicPlayer*>(user);
  if (!player->IsPlaying() || samples == 0 || samples > 4096) return false;
  thread_local std::vector<float> interleaved;
  if (interleaved.size() < size_t(samples) * 2) interleaved.resize(size_t(samples) * 2);
  std::fill(interleaved.begin(), interleaved.begin() + size_t(samples) * 2, 0.0f);
  player->MixFrames(interleaved.data(), samples, frequency, 2);
  bool any = false;
  for (uint32_t i = 0; i < samples; ++i) {
    const float l = interleaved[2 * i], r = interleaved[2 * i + 1];
    if (l != 0.0f || r != 0.0f) any = true;
    left[i] += l;
    right[i] += r;
  }
  return any;
}
}  // namespace

bool CustomMusicPlayer::Play(const std::filesystem::path& path, const std::string& name) {
  {
    static std::once_flag inject_once;
    std::call_once(inject_once, [this] { rex::perf::audio::SetInject(&InjectCustomMusic, this); });
  }
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (shutdown_.load(std::memory_order_relaxed)) {
      return false;
    }
    request_path_ = path;
    request_name_ = name;
    request_pending_ = true;
    ++request_serial_;
    if (worker_running_) {
      // Already running; the loop will pick the request up.
      return true;
    }
    worker_running_ = true;
    // Releasing the lock before starting the thread: WorkerMain takes it.
  }
  worker_ = std::thread([this]() { WorkerMain(); });
  return true;
}

void CustomMusicPlayer::Stop() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (state_.load(std::memory_order_relaxed) != kIdle || track_) {
    REXLOG_INFO("custom music: stop '{}'", name_);
  }
  // Bump the serial so a decode in flight is not published after the stop.
  ++request_serial_;
  track_.reset();
  cursor_ = 0;
  state_.store(kIdle, std::memory_order_relaxed);
  request_pending_ = false;
}

void CustomMusicPlayer::Pause() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (state_.load(std::memory_order_relaxed) == kPlaying) {
    REXLOG_INFO("custom music: pause '{}'", name_);
    state_.store(kPaused, std::memory_order_relaxed);
  }
}

void CustomMusicPlayer::Continue() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (state_.load(std::memory_order_relaxed) == kPaused) {
    REXLOG_INFO("custom music: continue '{}'", name_);
    state_.store(kPlaying, std::memory_order_relaxed);
  }
}

void CustomMusicPlayer::MixFrames(float* output, uint32_t frames, uint32_t output_rate,
                                  uint32_t output_channels) {
  if (frames == 0 || !output || output_channels < 2) {
    return;
  }
  if (output_rate != 0) {
    output_rate_.store(output_rate, std::memory_order_relaxed);
  }
  const uint32_t state = state_.load(std::memory_order_relaxed);
  if (state != kPlaying) {
    return;
  }
  if (sample_rate_.load(std::memory_order_relaxed) == 0) {
    return;
  }
  const float volume = volume_.load(std::memory_order_relaxed);
  if (volume <= 0.0f) {
    return;
  }

  std::lock_guard<std::mutex> guard(mutex_);
  if (!track_ || state_.load(std::memory_order_relaxed) != kPlaying) {
    return;
  }
  const std::vector<float>& samples = track_->samples;
  const size_t total_frames = samples.size() / 2;
  for (uint32_t frame = 0; frame < frames; ++frame) {
    if (cursor_ >= total_frames) {
      state_.store(kFinished, std::memory_order_relaxed);
      REXLOG_INFO("custom music: '{}' finished", name_);
      break;
    }
    float* out = output + static_cast<size_t>(frame) * output_channels;
    out[0] += samples[cursor_ * 2 + 0] * volume;
    out[1] += samples[cursor_ * 2 + 1] * volume;
    ++cursor_;
  }
}

}  // namespace rex::audio
