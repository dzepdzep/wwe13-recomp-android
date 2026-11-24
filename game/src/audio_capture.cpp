// wwe13 - audio capture for testing (WWE13_CAPTURE_AUDIO=1 in env_user.txt): the game's whole sound mix as it goes
// to the speakers is written to capture/audio-<date>-<time>.wav (stereo, 16-bit), so sounds can be checked without
// listening (impacts, announcer, music). The header is kept valid every second, so a killed run leaves a playable
// file. Stops after 20 minutes. The log names the file and the wall-clock time of its first sample (to line it up
// with a screen recording).
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <rex/logging.h>
#include <rex/perf/audio_tap.h>

namespace {
struct Capture {
  FILE* f = nullptr;
  uint32_t rate = 0;
  uint64_t frames = 0;  // stereo sample frames written
  uint64_t since_header = 0;
  bool done = false;
  std::vector<int16_t> buf;
};
Capture g_cap;

void WriteHeader(Capture& c) {
  const uint32_t data = uint32_t(c.frames * 4);
  struct {
    char riff[4]; uint32_t size; char wave[4]; char fmt[4]; uint32_t fmt_size; uint16_t tag, ch; uint32_t rate, bps;
    uint16_t align, bits; char data[4]; uint32_t data_size;
  } h = {{'R', 'I', 'F', 'F'}, 36 + data, {'W', 'A', 'V', 'E'}, {'f', 'm', 't', ' '}, 16, 1, 2, c.rate, c.rate * 4, 4, 16,
         {'d', 'a', 't', 'a'}, data};
  const long pos = ftell(c.f);
  fseek(c.f, 0, SEEK_SET);
  fwrite(&h, sizeof h, 1, c.f);
  fseek(c.f, pos, SEEK_SET);
  fflush(c.f);
}

float BeFloat(const uint8_t* p) {
  const uint32_t v = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}

void Tap(void*, const void* frame_be, uint32_t samples, uint32_t channels, uint32_t frequency) {
  Capture& c = g_cap;
  if (c.done) return;
  static const bool on = [] {
    const char* v = std::getenv("WWE13_CAPTURE_AUDIO");
    return v && *v == '1';
  }();
  if (!on) {
    c.done = true;
    return;
  }
  if (!c.f) {
    mkdir("capture", 0775);
    char stamp[32], path[96];
    const time_t now = time(nullptr);
    struct tm tmv{};
    localtime_r(&now, &tmv);
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tmv);
    snprintf(path, sizeof path, "capture/audio-%s.wav", stamp);
    c.f = fopen(path, "wb+");
    if (!c.f) {
      c.done = true;
      return;
    }
    c.rate = frequency;
    WriteHeader(c);
    fseek(c.f, 44, SEEK_SET);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
    REXLOG_INFO("[wwe13-capture] audio to {} ({} Hz, {} channels), first sample at {} ms since 1970", path, frequency,
                channels, ms);
  }
  const auto* p = static_cast<const uint8_t*>(frame_be);
  c.buf.resize(size_t(samples) * 2);
  for (uint32_t i = 0; i < samples; ++i) {
    auto ch = [&](uint32_t k) { return k < channels ? BeFloat(p + (size_t(k) * samples + i) * 4) : 0.0f; };
    // FL FR C LFE SL SR -> stereo (as the SDL fold)
    float l = ch(0) + 0.707f * ch(2) + 0.707f * ch(4);
    float r = ch(1) + 0.707f * ch(2) + 0.707f * ch(5);
    l = std::clamp(l, -1.0f, 1.0f);
    r = std::clamp(r, -1.0f, 1.0f);
    c.buf[i * 2] = int16_t(l * 32767.0f);
    c.buf[i * 2 + 1] = int16_t(r * 32767.0f);
  }
  fwrite(c.buf.data(), 2, c.buf.size(), c.f);
  c.frames += samples;
  c.since_header += samples;
  if (c.since_header >= c.rate) {
    c.since_header = 0;
    WriteHeader(c);
  }
  if (c.frames >= uint64_t(c.rate) * 60 * 20) {
    WriteHeader(c);
    fclose(c.f);
    c.f = nullptr;
    c.done = true;
    REXLOG_INFO("[wwe13-capture] audio stopped after 20 minutes");
  }
}

struct Install {
  Install() {
    rex::perf::audio::SetTap(Tap, &g_cap);  // (the variable is read at the first frame: env_user.txt is set by then)
  }
} g_install;
}  // namespace
