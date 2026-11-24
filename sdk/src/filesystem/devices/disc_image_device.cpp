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

#include <rex/filesystem/devices/disc_image_device.h>
#include <rex/filesystem/devices/disc_image_entry.h>

#include <algorithm>
#include <vector>

#include <fmt/format.h>

#include <rex/det.h>  // wwe2k14 patch (Session 77): det_fs_constant
#include <rex/literals.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>

namespace rex::filesystem {

using namespace rex::literals;

const size_t kXESectorSize = 2_KiB;

DiscImageDevice::DiscImageDevice(const std::string_view mount_path,
                                 const std::filesystem::path& host_path)
    : Device(mount_path), name_("GDFX"), host_path_(host_path) {}

DiscImageDevice::~DiscImageDevice() = default;

bool DiscImageDevice::Initialize() {
  mmap_ = memory::MappedMemory::Open(host_path_, memory::MappedMemory::Mode::kRead);
  if (!mmap_) {
    REXFS_ERROR("Disc image could not be mapped");
    return false;
  }

  ParseState state = {};
  state.ptr = mmap_->data();
  state.size = mmap_->size();
  auto result = Verify(&state);
  if (result != Error::kSuccess) {
    REXFS_ERROR("Failed to verify disc image header: {}", static_cast<int>(result));
    return false;
  }

  result = ReadAllEntries(&state, state.ptr + state.root_offset);
  if (result != Error::kSuccess) {
    REXFS_ERROR("Failed to read all GDFX entries: {} (disc image {}: {})", static_cast<int>(result),
                host_path_.string(), parse_error_.empty() ? "?" : parse_error_);
    return false;
  }
  // WWE 2K14 Session 26: an image cut short (an interrupted copy) lists files it doesn't hold; refuse it here instead
  // of failing on a read in the middle of the game.
  if (files_past_end_) {
    REXFS_ERROR("Disc image {} is incomplete: {} of {} files lie beyond its end ({} bytes); copy it again",
                host_path_.string(), files_past_end_, file_count_, state.size);
    return false;
  }
  REXFS_INFO("Disc image {}: {} files, {} bytes, game partition at {:#x}", host_path_.string(), file_count_,
             total_file_size_, state.game_offset);

  return true;
}

void DiscImageDevice::Dump(string::StringBuffer* string_buffer) {
  auto global_lock = global_critical_region_.Acquire();
  string_buffer->AppendFormat(
      "{}: {} files, {} bytes (game_offset={:#x}, root_sector={}, root_size={}, host_size={})\n",
      mount_path(), file_count_, total_file_size_, disc_info_.game_offset, disc_info_.root_sector,
      disc_info_.root_size, disc_info_.host_size);
}

Entry* DiscImageDevice::ResolvePath(const std::string_view path) {
  // The filesystem will have stripped our prefix off already, so the path will
  // be in the form:
  // some\PATH.foo
  REXFS_DEBUG("DiscImageDevice::ResolvePath({})", path);
  return root_entry_->ResolvePath(path);
}

DiscImageDevice::Error DiscImageDevice::Verify(ParseState* state) {
  // Find sector 32 of the game partition - try at a few points.
  static const size_t likely_offsets[] = {
      0x00000000, 0x0000FB20, 0x00020600, 0x02080000, 0x0FD90000,
  };
  bool magic_found = false;
  for (size_t n = 0; n < rex::countof(likely_offsets); n++) {
    state->game_offset = likely_offsets[n];
    if (VerifyMagic(state, state->game_offset + (32 * kXESectorSize))) {
      magic_found = true;
      break;
    }
  }
  if (!magic_found) {
    // File doesn't have the magic values - likely not a real GDFX source.
    return Error::kErrorFileMismatch;
  }

  // Read sector 32 to get FS state.
  if (state->size < state->game_offset + (32 * kXESectorSize)) {
    return Error::kErrorReadError;
  }
  uint8_t* fs_ptr = state->ptr + state->game_offset + (32 * kXESectorSize);
  state->root_sector = memory::load<uint32_t>(fs_ptr + 20);
  state->root_size = memory::load<uint32_t>(fs_ptr + 24);
  state->root_offset = state->game_offset + (state->root_sector * kXESectorSize);
  if (state->root_size < 13 || state->root_size > 32_MiB) {
    return Error::kErrorDamagedFile;
  }
  if (state->root_offset + state->root_size > state->size) {  // Session 26: the root table must be inside the image
    return Error::kErrorDamagedFile;
  }

  disc_info_.game_offset = state->game_offset;
  disc_info_.root_sector = state->root_sector;
  disc_info_.root_size = state->root_size;
  disc_info_.host_size = state->size;

  return Error::kSuccess;
}

bool DiscImageDevice::VerifyMagic(ParseState* state, size_t offset) {
  if (offset + 28 > state->size) {  // Session 26: the magic and the root sector / size after it
    return false;
  }

  // Simple check to see if the given offset contains the magic value.
  return std::memcmp(state->ptr + offset, "MICROSOFT*XBOX*MEDIA", 20) == 0;
}

DiscImageDevice::Error DiscImageDevice::ReadAllEntries(ParseState* state,
                                                       const uint8_t* root_buffer) {
  auto root_entry = new DiscImageEntry(this, nullptr, "", mmap_.get());
  root_entry->attributes_ = kFileAttributeDirectory;
  root_entry_ = std::unique_ptr<Entry>(root_entry);

  (void)root_buffer;  // Session 26: ReadTable checks the root table's bounds itself
  if (!ReadTable(state, state->root_offset, state->root_size, root_entry, 0)) {
    return Error::kErrorDamagedFile;
  }

  return Error::kSuccess;
}

// WWE 2K14 Session 26: records why the image is refused (logged once by Initialize). Out of line, so the recursive
// functions keep small stack frames.
__attribute__((noinline)) bool DiscImageDevice::Damaged(const char* what, size_t a, size_t b) {
  if (parse_error_.empty()) {
    parse_error_ = fmt::format("damaged directory table: {} ({}, {})", what, a, b);
  }
  return false;
}

bool DiscImageDevice::ReadTable(ParseState* state, size_t offset, size_t size, DiscImageEntry* parent,
                                uint32_t depth) {
  if (offset + size > state->size || size > 32_MiB) {
    return Damaged("a table lies beyond the end of the image", offset, size);
  }
  // A table parsed twice means two folders share it, or a folder contains itself.
  if (std::find(tables_seen_.begin(), tables_seen_.end(), offset) != tables_seen_.end()) {
    return Damaged("a table is used twice", offset, size);
  }
  tables_seen_.push_back(offset);
  const uint8_t* buffer = state->ptr + offset;
  // An empty folder may be stored as a table of padding (0xFF) instead of a zero length: no entries.
  if (size >= 4 && memory::load<uint32_t>(buffer) == 0xFFFFFFFFu) {
    return true;
  }
  std::vector<uint8_t> visited(size / 4 + 1, 0);
  return ReadEntry(state, buffer, size, 0, parent, depth, visited);
}

bool DiscImageDevice::ReadEntry(ParseState* state, const uint8_t* buffer, size_t dir_size,
                                uint16_t entry_ordinal, DiscImageEntry* parent, uint32_t depth,
                                std::vector<uint8_t>& visited) {
  // Session 26: a real disc's trees are balanced (height ~17 for the largest possible table) and folders nest a few
  // levels, so 1024 is far above any valid image and far below the main thread's stack.
  const size_t entry_offset = size_t(entry_ordinal) * 4;
  if (depth > 1024) {
    return Damaged("too deep", depth, dir_size);
  }
  if (entry_offset + 14 > dir_size) {
    return Damaged("an entry lies outside its table", entry_ordinal, dir_size);
  }
  if (visited[entry_ordinal]) {
    return Damaged("an entry is linked twice", entry_ordinal, dir_size);
  }
  visited[entry_ordinal] = 1;
  const uint8_t* p = buffer + entry_offset;

  uint16_t node_l = memory::load<uint16_t>(p + 0);
  uint16_t node_r = memory::load<uint16_t>(p + 2);
  size_t sector = memory::load<uint32_t>(p + 4);
  size_t length = memory::load<uint32_t>(p + 8);
  uint8_t attributes = memory::load<uint8_t>(p + 12);
  uint8_t name_length = memory::load<uint8_t>(p + 13);
  auto name_buffer = reinterpret_cast<const char*>(p + 14);
  if (entry_offset + 14 + name_length > dir_size) {
    return Damaged("a name runs past its table", entry_ordinal, dir_size);
  }

  if (node_l && !ReadEntry(state, buffer, dir_size, node_l, parent, depth + 1, visited)) {
    return false;
  }

  auto name = std::string(name_buffer, name_length);

  auto entry = DiscImageEntry::Create(this, parent, name, mmap_.get());
  // wwe2k14 patch (Session 77): det_fs_constant - a read-only folder's attributes (host_path_entry.cpp: folder 0x10,
  // file 0x81), not the image's byte (a disc's, or 0x21 / 0x11 in an image tools\iso\make_xiso.py packed).
  if (det::FsConstant()) {
    entry->attributes_ = (attributes & kFileAttributeDirectory) ? uint32_t(kFileAttributeDirectory)
                                                                : uint32_t(kFileAttributeNormal | kFileAttributeReadOnly);
  } else {
    entry->attributes_ = attributes | kFileAttributeReadOnly;
  }
  entry->size_ = length;
  entry->allocation_size_ = rex::round_up(length, bytes_per_sector());
  // (Session 77, review: det_fs_constant - a folder's size is 0, as in a folder dump; its table length stays local.)
  if (det::FsConstant() && (attributes & kFileAttributeDirectory)) entry->size_ = 0, entry->allocation_size_ = 0;

  // Set to January 1, 1970 (UTC) in 100-nanosecond intervals
  entry->create_timestamp_ = 10000 * 11644473600000LL;
  entry->access_timestamp_ = 10000 * 11644473600000LL;
  entry->write_timestamp_ = 10000 * 11644473600000LL;

  if (attributes & kFileAttributeDirectory) {
    // Folder.
    entry->data_offset_ = 0;
    entry->data_size_ = 0;
    if (length) {
      // Not a leaf - read in children (Session 26: ReadTable checks the whole table is inside the image).
      if (!ReadTable(state, state->game_offset + (sector * kXESectorSize), length, entry.get(), depth + 1)) {
        return false;
      }
    }
  } else {
    // File.
    entry->data_offset_ = state->game_offset + (sector * kXESectorSize);
    entry->data_size_ = length;
    ++file_count_;
    total_file_size_ += length;
    if (length && entry->data_offset_ + length > state->size) {
      ++files_past_end_;  // Session 26: Initialize refuses the image (a truncated copy)
    }
  }

  // Add to parent.
  parent->children_.emplace_back(std::move(entry));

  // Read next file in the list.
  if (node_r && !ReadEntry(state, buffer, dir_size, node_r, parent, depth + 1, visited)) {
    return false;
  }

  return true;
}

}  // namespace rex::filesystem
