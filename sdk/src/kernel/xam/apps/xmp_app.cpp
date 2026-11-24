/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/kernel/xam/apps/xmp_app.h>
#include <rex/audio/custom_music_player.h>
#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/runtime.h>
#include <rex/string.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xenumerator.h>
#include <rex/system/xthread.h>
#include <rex/thread.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <system_error>

REXCVAR_DEFINE_STRING(xmp_music_dir, "", "XMP",
                      "Host folder containing custom XMP music (empty uses <exe>/music)");

namespace rex {
namespace kernel {
namespace xam {
using namespace rex::system;
using namespace rex::system::xam;
namespace apps {
using namespace rex::system;

namespace {

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

rex::audio::CustomMusicPlayer& Player() {
  return rex::audio::CustomMusicPlayer::instance();
}

bool XmpStateProbeEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("REX_XMP_STATE_PROBE");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

// Notification traces fire on every XMP state/behaviour/controller change; keep them bounded even
// with the probe on (first 64 lines, then 1 in 100).
bool XmpStateProbeNotifyLog() {
  if (!XmpStateProbeEnabled()) {
    return false;
  }
  static std::atomic<uint64_t> notify_log_count{0};
  const uint64_t count = notify_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
  return count <= 64 || count % 100 == 0;
}

}  // namespace

XmpApp::XmpApp(KernelState* kernel_state)
    : App(kernel_state, 0xFA),
      state_(State::kIdle),
      playback_client_(PlaybackClient::kTitle),
      playback_mode_(PlaybackMode::kUnknown),
      repeat_mode_(RepeatMode::kUnknown),
      unknown_flags_(0),
      volume_(1.0f),
      active_playlist_(nullptr),
      active_song_index_(0),
      next_playlist_handle_(1),
      next_song_handle_(1) {}

X_HRESULT XmpApp::XMPGetStatus(uint32_t state_ptr) {
  if (!XThread::GetCurrentThread()->main_thread()) {
    // Some stupid games will hammer this on a thread - induce a delay
    // here to keep from starving real threads.
    rex::thread::Sleep(std::chrono::milliseconds(1));
  }

  REXKRNL_TRACE("XMPGetStatus({:08X})", state_ptr);
  // A custom song can end (or fail to decode) without any guest message; make
  // the reported state truthful instead of stuck at kPlaying.
  if (active_custom_song_.song && !Player().IsActive()) {
    REXKRNL_INFO("XMP custom song '{}' ended", rex::string::to_utf8(active_custom_song_.song->name));
    active_custom_song_ = {};
    state_ = State::kIdle;
  }
  memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(state_ptr),
                                   static_cast<uint32_t>(state_));
  if (XmpStateProbeEnabled()) {
    const uint64_t count = get_status_log_count_.load(std::memory_order_relaxed);
    if (count <= 3 || count % 1000 == 0) {
      REXKRNL_INFO("[xmp-state-probe] XMPGetStatus state={} custom_active={}",
                   static_cast<uint32_t>(state_), active_custom_song_.song != nullptr);
    }
  }
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPCreateTitlePlaylist(uint32_t songs_ptr, uint32_t song_count,
                                         uint32_t playlist_name_ptr,
                                         const std::u16string& playlist_name, uint32_t flags,
                                         uint32_t out_song_handles, uint32_t out_playlist_handle) {
  REXKRNL_DEBUG(
      "XMPCreateTitlePlaylist({:08X}, {:08X}, {:08X}({}), {:08X}, {:08X}, "
      "{:08X})",
      songs_ptr, song_count, playlist_name_ptr, rex::string::to_utf8(playlist_name), flags,
      out_song_handles, out_playlist_handle);
  auto playlist = std::make_unique<Playlist>();
  playlist->handle = ++next_playlist_handle_;
  playlist->name = playlist_name;
  playlist->flags = flags;
  if (songs_ptr) {
    for (uint32_t i = 0; i < song_count; ++i) {
      auto song = std::make_unique<Song>();
      song->handle = ++next_song_handle_;
      uint8_t* song_base = memory_->TranslateVirtual(songs_ptr + (i * 36));
      song->file_path = memory::load_and_swap<std::u16string>(
          memory_->TranslateVirtual(memory::load_and_swap<uint32_t>(song_base + 0)));
      song->name = memory::load_and_swap<std::u16string>(
          memory_->TranslateVirtual(memory::load_and_swap<uint32_t>(song_base + 4)));
      song->artist = memory::load_and_swap<std::u16string>(
          memory_->TranslateVirtual(memory::load_and_swap<uint32_t>(song_base + 8)));
      song->album = memory::load_and_swap<std::u16string>(
          memory_->TranslateVirtual(memory::load_and_swap<uint32_t>(song_base + 12)));
      song->album_artist = memory::load_and_swap<std::u16string>(
          memory_->TranslateVirtual(memory::load_and_swap<uint32_t>(song_base + 16)));
      song->genre = memory::load_and_swap<std::u16string>(
          memory_->TranslateVirtual(memory::load_and_swap<uint32_t>(song_base + 20)));
      song->track_number = memory::load_and_swap<uint32_t>(song_base + 24);
      song->duration_ms = memory::load_and_swap<uint32_t>(song_base + 28);
      song->format = static_cast<Song::Format>(memory::load_and_swap<uint32_t>(song_base + 32));
      if (out_song_handles) {
        memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(out_song_handles + (i * 4)),
                                         song->handle);
      }
      playlist->songs.emplace_back(std::move(song));
    }
  }
  if (out_playlist_handle) {
    memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(out_playlist_handle),
                                     playlist->handle);
  }

  auto global_lock = global_critical_region_.Acquire();
  playlists_.insert({playlist->handle, playlist.get()});
  playlist.release();
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPDeleteTitlePlaylist(uint32_t playlist_handle) {
  REXKRNL_DEBUG("XMPDeleteTitlePlaylist({:08X})", playlist_handle);
  auto global_lock = global_critical_region_.Acquire();
  auto it = playlists_.find(playlist_handle);
  if (it == playlists_.end()) {
    REXKRNL_ERROR("Playlist {:08X} not found", playlist_handle);
    return X_E_NOTFOUND;
  }
  auto playlist = it->second;
  if (playlist == active_playlist_) {
    XMPStop(0);
  }
  playlists_.erase(it);
  delete playlist;
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPPlayTitlePlaylist(uint32_t playlist_handle, uint32_t song_handle) {
  REXKRNL_DEBUG("XMPPlayTitlePlaylist({:08X}, {:08X})", playlist_handle, song_handle);
  Playlist* playlist = nullptr;
  {
    auto global_lock = global_critical_region_.Acquire();
    auto it = playlists_.find(playlist_handle);
    if (it == playlists_.end()) {
      REXKRNL_ERROR("Playlist {:08X} not found", playlist_handle);
      return X_E_NOTFOUND;
    }
    playlist = it->second;
  }

  if (playback_client_ == PlaybackClient::kSystem) {
    REXKRNL_WARN("XMPPlayTitlePlaylist: System playback is enabled!");
    return X_E_SUCCESS;
  }

  // Start playlist?
  REXKRNL_WARN("Playlist playback not supported");
  active_playlist_ = playlist;
  active_song_index_ = 0;
  state_ = State::kPlaying;
  OnStateChanged();
  if (XmpStateProbeNotifyLog()) {
    REXKRNL_INFO("[xmp-state-probe] notify 0x0a000002 param=1 (playlist play)");
  }
  kernel_state_->BroadcastNotification(kMsgPlaybackBehaviorChanged, 1);
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPContinue() {
  REXKRNL_DEBUG("XMPContinue()");
  if (state_ == State::kPaused) {
    state_ = State::kPlaying;
    Player().Continue();
  }
  OnStateChanged();
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPStop(uint32_t unk) {
  assert_zero(unk);
  REXKRNL_DEBUG("XMPStop({:08X})", unk);
  active_playlist_ = nullptr;  // ?
  active_song_index_ = 0;
  active_custom_song_ = {};
  Player().Stop();
  state_ = State::kIdle;
  OnStateChanged();
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPPause() {
  REXKRNL_DEBUG("XMPPause()");
  if (state_ == State::kPlaying) {
    state_ = State::kPaused;
    Player().Pause();
  }
  OnStateChanged();
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPPlayCustomSong(const CustomMusicSong* song) {
  if (!song) {
    return X_E_NOTFOUND;
  }
  active_playlist_ = nullptr;
  active_song_index_ = 0;
  active_custom_song_.song = song;
  active_custom_song_.start_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()
                                                                .time_since_epoch())
          .count());
  Player().SetVolume(volume_);
  if (!Player().Play(song->path, rex::string::to_utf8(song->name))) {
    REXKRNL_ERROR("XMP custom play failed to start for '{}' ({})", rex::string::to_utf8(song->name),
                  rex::path_to_utf8(song->path));
    active_custom_song_ = {};
    state_ = State::kIdle;
    OnStateChanged();
    return X_E_FAIL;
  }
  state_ = State::kPlaying;
  OnStateChanged();
  if (XmpStateProbeNotifyLog()) {
    REXKRNL_INFO("[xmp-state-probe] notify 0x0a000002 param=1 (custom play)");
  }
  kernel_state_->BroadcastNotification(kMsgPlaybackBehaviorChanged, 1);
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPNext() {
  REXKRNL_DEBUG("XMPNext()");
  if (!active_playlist_) {
    return X_E_NOTFOUND;
  }
  state_ = State::kPlaying;
  active_song_index_ = (active_song_index_ + 1) % active_playlist_->songs.size();
  OnStateChanged();
  return X_E_SUCCESS;
}

X_HRESULT XmpApp::XMPPrevious() {
  REXKRNL_DEBUG("XMPPrevious()");
  if (!active_playlist_) {
    return X_E_NOTFOUND;
  }
  state_ = State::kPlaying;
  if (!active_song_index_) {
    active_song_index_ = static_cast<int>(active_playlist_->songs.size()) - 1;
  } else {
    --active_song_index_;
  }
  OnStateChanged();
  return X_E_SUCCESS;
}

void XmpApp::OnStateChanged() {
  if (XmpStateProbeNotifyLog()) {
    REXKRNL_INFO("[xmp-state-probe] notify 0x0a000001 param={}",
                 static_cast<uint32_t>(state_));
  }
  kernel_state_->BroadcastNotification(kMsgStateChanged, static_cast<uint32_t>(state_));
}

void XmpApp::EnsureCustomMusicLoaded() {
  std::call_once(custom_music_once_, [this]() {
    try {
      const std::string configured_directory = REXCVAR_GET(xmp_music_dir);
      std::filesystem::path music_directory =
          configured_directory.empty() ? rex::filesystem::GetExecutableFolder() / "music"
                                       : rex::to_path(configured_directory);
      std::error_code absolute_error;
      auto absolute_directory = std::filesystem::absolute(music_directory, absolute_error);
      if (!absolute_error) {
        music_directory = std::move(absolute_directory);
      }

      struct Candidate {
        std::filesystem::path path;
        std::string filename;
        std::string sort_key;
      };
      std::vector<Candidate> candidates;
      std::error_code directory_error;
      if (std::filesystem::is_directory(music_directory, directory_error) && !directory_error) {
        std::filesystem::directory_iterator iterator(music_directory, directory_error);
        const std::filesystem::directory_iterator end;
        while (!directory_error && iterator != end) {
          std::error_code file_error;
          if (iterator->is_regular_file(file_error) && !file_error) {
            const auto extension = LowerAscii(rex::path_to_utf8(iterator->path().extension()));
            if (extension == ".mp3" || extension == ".ogg" || extension == ".flac" ||
                extension == ".wav") {
              auto filename = rex::path_to_utf8(iterator->path().filename());
              candidates.push_back(
                  {iterator->path(), filename, LowerAscii(std::move(filename))});
            }
          }
          iterator.increment(directory_error);
        }
      }

      std::sort(candidates.begin(), candidates.end(), [](const Candidate& left,
                                                         const Candidate& right) {
        if (left.sort_key != right.sort_key) {
          return left.sort_key < right.sort_key;
        }
        return left.filename < right.filename;
      });

      custom_music_songs_.reserve(candidates.size());
      for (const auto& candidate : candidates) {
        auto name = rex::path_to_utf16(candidate.path.stem());
        if (name.size() > 19) {
          name.resize(19);
          if (!name.empty() && name.back() >= 0xD800 && name.back() <= 0xDBFF) {
            name.pop_back();
          }
        }
        const uint32_t format = LowerAscii(rex::path_to_utf8(candidate.path.extension())) == ".mp3" ? 1 : 0;
        custom_music_songs_.push_back(
            {static_cast<uint32_t>(custom_music_songs_.size() + 1), candidate.path,
             std::move(name), format});
      }

      REXKRNL_INFO("custom music folder {}: {} songs", rex::path_to_utf8(music_directory),
                   custom_music_songs_.size());
    } catch (const std::exception& e) {
      // A song file name or folder the host cannot handle must never take the game down.
      custom_music_songs_.clear();
      REXKRNL_ERROR("custom music folder scan failed: {}", e.what());
    }
  });
}

void XmpApp::AppendCustomMusicItems(XStaticUntypedEnumerator* enumerator) {
  EnsureCustomMusicLoaded();
  for (const auto& song : custom_music_songs_) {
    auto* record = enumerator->AppendItem();
    std::memset(record, 0, 820);
    memory::store_and_swap<uint32_t>(record + 0, song.handle);
    memory::store_and_swap<std::u16string>(record + 572, song.name);
    memory::store_and_swap<uint32_t>(record + 772, 0);
    memory::store_and_swap<uint32_t>(record + 776, 0);
    memory::store_and_swap<uint32_t>(record + 780, song.format);
  }
}

X_HRESULT XmpApp::DispatchMessageSync(uint32_t message, uint32_t buffer_ptr,
                                      uint32_t buffer_length) {
  // NOTE: buffer_length may be zero or valid.
  bool log_message = true;
  if (message == 0x00070009) {
    const auto count = get_status_log_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    log_message = count <= 3 || count % 1000 == 0;
  } else if (message == 0x0007001B) {
    const auto count =
        get_playback_controller_log_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    log_message = count <= 3 || count % 1000 == 0;
  }
  if (log_message) {
    REXKRNL_INFO("XMP message app={:#x} msg={:#x} buffer_length={}", app_id_, message,
                 buffer_length);
  }

  auto buffer = memory_->TranslateVirtual(buffer_ptr);
  switch (message) {
    case 0x00070025: {
      assert_true(!buffer_length || buffer_length == 12);
      return X_E_SUCCESS;
    }
    case 0x0007003B: {
      assert_true(!buffer_length || buffer_length == 824);
      auto* record = buffer + 4;
      const uint32_t song_handle = memory::load_and_swap<uint32_t>(record + 0);
      const auto song_name = memory::load_and_swap<std::u16string>(record + 572);
      EnsureCustomMusicLoaded();

      // Name first: the folder contents are enumerated by us and the handles
      // the guest remembers go stale when the folder changes.
      const CustomMusicSong* selected_song = nullptr;
      for (const auto& song : custom_music_songs_) {
        if (song.name == song_name) {
          selected_song = &song;
          break;
        }
      }
      if (!selected_song) {
        for (const auto& song : custom_music_songs_) {
          if (song.handle == song_handle) {
            selected_song = &song;
            break;
          }
        }
      }
      if (!selected_song) {
        REXKRNL_ERROR("XMP play request for unknown song '{}' (handle {:08X})",
                      rex::string::to_utf8(song_name), song_handle);
        return X_E_NOTFOUND;
      }
      REXKRNL_INFO("XMP play request: {} ({}) handle={:08X}", rex::string::to_utf8(song_name),
                   rex::path_to_utf8(selected_song->path), song_handle);
      return XMPPlayCustomSong(selected_song);
    }
    case 0x00070002: {
      assert_true(!buffer_length || buffer_length == 12);
      uint32_t xmp_client = memory::load_and_swap<uint32_t>(buffer + 0);
      uint32_t storage_ptr = memory::load_and_swap<uint32_t>(buffer + 4);
      uint32_t song_handle = memory::load_and_swap<uint32_t>(buffer + 8);  // 0?
      uint32_t playlist_handle =
          memory::load_and_swap<uint32_t>(memory_->TranslateVirtual(storage_ptr));
      assert_true(xmp_client == 0x00000002);
      return XMPPlayTitlePlaylist(playlist_handle, song_handle);
    }
    case 0x00070003: {
      assert_true(!buffer_length || buffer_length == 4);
      uint32_t xmp_client = memory::load_and_swap<uint32_t>(buffer + 0);
      assert_true(xmp_client == 0x00000002);
      return XMPContinue();
    }
    case 0x00070004: {
      assert_true(!buffer_length || buffer_length == 8);
      uint32_t xmp_client = memory::load_and_swap<uint32_t>(buffer + 0);
      uint32_t unk = memory::load_and_swap<uint32_t>(buffer + 4);
      assert_true(xmp_client == 0x00000002);
      return XMPStop(unk);
    }
    case 0x00070005: {
      assert_true(!buffer_length || buffer_length == 4);
      uint32_t xmp_client = memory::load_and_swap<uint32_t>(buffer + 0);
      assert_true(xmp_client == 0x00000002);
      return XMPPause();
    }
    case 0x00070006: {
      assert_true(!buffer_length || buffer_length == 4);
      uint32_t xmp_client = memory::load_and_swap<uint32_t>(buffer + 0);
      assert_true(xmp_client == 0x00000002);
      return XMPNext();
    }
    case 0x00070007: {
      assert_true(!buffer_length || buffer_length == 4);
      uint32_t xmp_client = memory::load_and_swap<uint32_t>(buffer + 0);
      assert_true(xmp_client == 0x00000002);
      return XMPPrevious();
    }
    case 0x00070008: {
      assert_true(!buffer_length || buffer_length == 16);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> playback_mode;
        rex::be<uint32_t> repeat_mode;
        rex::be<uint32_t> flags;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 16);

      assert_true(args->xmp_client == 0x00000002 || args->xmp_client == 0x00000000);
      REXKRNL_DEBUG("XMPSetPlaybackBehavior({:08X}, {:08X}, {:08X})", uint32_t(args->playback_mode),
                    uint32_t(args->repeat_mode), uint32_t(args->flags));
      playback_mode_ = static_cast<PlaybackMode>(uint32_t(args->playback_mode));
      repeat_mode_ = static_cast<RepeatMode>(uint32_t(args->repeat_mode));
      unknown_flags_ = args->flags;
      if (XmpStateProbeNotifyLog()) {
        REXKRNL_INFO("[xmp-state-probe] notify 0x0a000002 param=0 (playback behavior set)");
      }
      kernel_state_->BroadcastNotification(kMsgPlaybackBehaviorChanged, 0);
      return X_E_SUCCESS;
    }
    case 0x00070009: {
      assert_true(!buffer_length || buffer_length == 8);
      uint32_t xmp_client = memory::load_and_swap<uint32_t>(buffer + 0);
      uint32_t state_ptr = memory::load_and_swap<uint32_t>(buffer + 4);  // out ptr to 4b - expect 0
      assert_true(xmp_client == 0x00000002);
      return XMPGetStatus(state_ptr);
    }
    case 0x0007000B: {
      assert_true(!buffer_length || buffer_length == 8);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> volume_ptr;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 8);

      assert_true(args->xmp_client == 0x00000002);
      REXKRNL_DEBUG("XMPGetVolume({:08X})", uint32_t(args->volume_ptr));
      memory::store_and_swap<float>(memory_->TranslateVirtual(args->volume_ptr), volume_);
      if (XmpStateProbeEnabled()) {
        static std::atomic<uint64_t> get_volume_probe_count{0};
        const auto count = get_volume_probe_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 3 || count % 1000 == 0) {
          REXKRNL_INFO("[xmp-state-probe] XMPGetVolume value={:g}", volume_);
        }
      }
      return X_E_SUCCESS;
    }
    case 0x0007000C: {
      assert_true(!buffer_length || buffer_length == 8);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<float> value;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 8);

      assert_true(args->xmp_client == 0x00000002);
      REXKRNL_INFO("XMPSetVolume({:g})", float(args->value));
      volume_ = args->value;
      Player().SetVolume(volume_);
      return X_E_SUCCESS;
    }
    case 0x0007000D: {
      assert_true(!buffer_length || buffer_length == 36);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> storage_ptr;
        rex::be<uint32_t> storage_size;
        rex::be<uint32_t> songs_ptr;
        rex::be<uint32_t> song_count;
        rex::be<uint32_t> playlist_name_ptr;
        rex::be<uint32_t> flags;
        rex::be<uint32_t> song_handles_ptr;
        rex::be<uint32_t> playlist_handle_ptr;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 36);

      memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(args->playlist_handle_ptr),
                                       args->storage_ptr);
      assert_true(args->xmp_client == 0x00000002 || args->xmp_client == 0x00000000);
      std::u16string playlist_name;
      if (!args->playlist_name_ptr) {
        playlist_name = u"";
      } else {
        playlist_name = memory::load_and_swap<std::u16string>(
            memory_->TranslateVirtual(args->playlist_name_ptr));
      }
      // dummy_alloc_ptr is the result of a XamAlloc of storage_size.
      assert_true(uint32_t(args->storage_size) == 4 + uint32_t(args->song_count) * 128);
      return XMPCreateTitlePlaylist(args->songs_ptr, args->song_count, args->playlist_name_ptr,
                                    playlist_name, args->flags, args->song_handles_ptr,
                                    args->storage_ptr);
    }
    case 0x0007000E: {
      assert_true(!buffer_length || buffer_length == 12);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> unk_ptr;  // 0
        rex::be<uint32_t> info_ptr;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 12);

      auto info = memory_->TranslateVirtual(args->info_ptr);
      assert_true(args->xmp_client == 0x00000002);
      assert_zero(args->unk_ptr);
      // Layout (guest caller sub_82CBBA00, generated/wwe13_recomp.89.cpp:47214):
      // {client, 0, info_ptr}. The playlist path below writes the handle at
      // +0 and the string fields at +4+572; mirror that for the custom song.
      if (active_custom_song_.song) {
        const CustomMusicSong* song = active_custom_song_.song;
        memory::store_and_swap<uint32_t>(info + 0, song->handle);
        memory::store_and_swap<std::u16string>(info + 4 + 572, song->name);
        REXKRNL_INFO("XMP custom song info: '{}' (handle {:08X})",
                     rex::string::to_utf8(song->name), song->handle);
        return X_E_SUCCESS;
      }
      if (!active_playlist_) {
        REXKRNL_ERROR("XMPGetInfo?({:08X}, {:08X}): no active song", uint32_t(args->unk_ptr),
                      uint32_t(args->info_ptr));
        return X_E_FAIL;
      }
      auto& song = active_playlist_->songs[active_song_index_];
      memory::store_and_swap<uint32_t>(info + 0, song->handle);
      memory::store_and_swap<std::u16string>(info + 4 + 572 + 0, song->name);
      memory::store_and_swap<std::u16string>(info + 4 + 572 + 40, song->artist);
      memory::store_and_swap<std::u16string>(info + 4 + 572 + 80, song->album);
      memory::store_and_swap<std::u16string>(info + 4 + 572 + 120, song->album_artist);
      memory::store_and_swap<std::u16string>(info + 4 + 572 + 160, song->genre);
      memory::store_and_swap<uint32_t>(info + 4 + 572 + 200, song->track_number);
      memory::store_and_swap<uint32_t>(info + 4 + 572 + 204, song->duration_ms);
      memory::store_and_swap<uint32_t>(info + 4 + 572 + 208, static_cast<uint32_t>(song->format));
      return X_E_SUCCESS;
    }
    case 0x00070013: {
      assert_true(!buffer_length || buffer_length == 8);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> storage_ptr;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 8);

      uint32_t playlist_handle =
          memory::load_and_swap<uint32_t>(memory_->TranslateVirtual(args->storage_ptr));
      assert_true(args->xmp_client == 0x00000002 || args->xmp_client == 0x00000000);
      return XMPDeleteTitlePlaylist(playlist_handle);
    }
    case 0x0007001A: {
      // XMPSetPlaybackController
      assert_true(!buffer_length || buffer_length == 12);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> controller;
        rex::be<uint32_t> playback_client;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 12);

      assert_true((args->xmp_client == 0x00000002 && args->controller == 0x00000000) ||
                  (args->xmp_client == 0x00000000 && args->controller == 0x00000001));
      REXKRNL_DEBUG("XMPSetPlaybackController({:08X}, {:08X})", uint32_t(args->controller),
                    uint32_t(args->playback_client));

      playback_client_ = PlaybackClient(uint32_t(args->playback_client));
      if (XmpStateProbeNotifyLog()) {
        REXKRNL_INFO(
            "[xmp-state-probe] XMPSetPlaybackController controller={} client={} -> notify 0x0a000003 param={}",
            uint32_t(args->controller), uint32_t(args->playback_client),
            static_cast<uint32_t>(!args->playback_client));
      }
      kernel_state_->BroadcastNotification(kMsgPlaybackControllerChanged, !args->playback_client);
      return X_E_SUCCESS;
    }
    case 0x0007001B: {
      // XMPGetPlaybackController
      assert_true(!buffer_length || buffer_length == 12);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> controller_ptr;
        rex::be<uint32_t> locked_ptr;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 12);

      assert_true(args->xmp_client == 0x00000002);
      REXKRNL_DEBUG("XMPGetPlaybackController({:08X}, {:08X}, {:08X})", uint32_t(args->xmp_client),
                    uint32_t(args->controller_ptr), uint32_t(args->locked_ptr));
      // Report the playback state stored by XMPSetPlaybackController instead of a
      // hardcoded idle (0, 0): the guest poll chain (sub_829BCF48) treats the pair
      // as the live controller state and only sees "not active" while both words
      // are zero. The controller word is not tracked by this app and the title's
      // own setter calls always pass controller == 0, so it stays 0.
      memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(args->controller_ptr), 0);
      memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(args->locked_ptr),
                                       static_cast<uint32_t>(playback_client_));
      if (XmpStateProbeEnabled()) {
        const uint64_t count =
            get_playback_controller_log_count_.load(std::memory_order_relaxed);
        if (count <= 3 || count % 1000 == 0) {
          REXKRNL_INFO(
              "[xmp-state-probe] XMPGetPlaybackController controller={} locked={} client={}", 0,
              static_cast<uint32_t>(playback_client_), static_cast<uint32_t>(playback_client_));
        }
      }

      if (!XThread::GetCurrentThread()->main_thread()) {
        // Atrain spawns a thread 82437FD0 to call this in a tight loop forever.
        rex::thread::Sleep(std::chrono::milliseconds(10));
      }

      return X_E_SUCCESS;
    }
    case 0x00070029: {
      // XMPGetPlaybackBehavior
      assert_true(!buffer_length || buffer_length == 16);
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> playback_mode_ptr;
        rex::be<uint32_t> repeat_mode_ptr;
        rex::be<uint32_t> unk3_ptr;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 16);

      assert_true(args->xmp_client == 0x00000002 || args->xmp_client == 0x00000000);
      REXKRNL_DEBUG("XMPGetPlaybackBehavior({:08X}, {:08X}, {:08X})",
                    uint32_t(args->playback_mode_ptr), uint32_t(args->repeat_mode_ptr),
                    uint32_t(args->unk3_ptr));
      if (args->playback_mode_ptr) {
        memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(args->playback_mode_ptr),
                                         static_cast<uint32_t>(playback_mode_));
      }
      if (args->repeat_mode_ptr) {
        memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(args->repeat_mode_ptr),
                                         static_cast<uint32_t>(repeat_mode_));
      }
      if (args->unk3_ptr) {
        memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(args->unk3_ptr), unknown_flags_);
      }
      return X_E_SUCCESS;
    }
    case 0x0007002E: {
      assert_true(!buffer_length || buffer_length == 12);
      // Query of size for XamAlloc - the result of the alloc is passed to
      // 0x0007000D.
      struct {
        rex::be<uint32_t> xmp_client;
        rex::be<uint32_t> song_count;
        rex::be<uint32_t> size_ptr;
      }* args = memory_->TranslateVirtual<decltype(args)>(buffer_ptr);
      static_assert_size(decltype(*args), 12);

      assert_true(args->xmp_client == 0x00000002 || args->xmp_client == 0x00000000);
      // We don't use the storage, so just fudge the number.
      memory::store_and_swap<uint32_t>(memory_->TranslateVirtual(args->size_ptr),
                                       4 + uint32_t(args->song_count) * 128);
      return X_E_SUCCESS;
    }
    case 0x0007003D: {
      // XMPCaptureOutput - not sure how this works :/
      REXKRNL_DEBUG("XMPCaptureOutput(...)");
      assert_always("XMP output not unimplemented");
      return X_E_FAIL;
    }
  }
  REXKRNL_ERROR(
      "Unimplemented XMP message app={:08X}, msg={:08X}, arg1={:08X}, "
      "arg2={:08X}",
      app_id(), message, buffer_ptr, buffer_length);
  return X_E_FAIL;
}

}  // namespace apps
}  // namespace xam
}  // namespace kernel
}  // namespace rex
