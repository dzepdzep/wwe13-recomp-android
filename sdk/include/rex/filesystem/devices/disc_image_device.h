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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rex/det.h>  // wwe2k14 patch (Session 77): det_fs_constant
#include <rex/filesystem/device.h>
#include <rex/memory/mapped_memory.h>

namespace rex::filesystem {

class DiscImageEntry;

class DiscImageDevice : public Device {
 public:
  DiscImageDevice(const std::string_view mount_path, const std::filesystem::path& host_path);
  ~DiscImageDevice() override;

  bool Initialize() override;
  void Dump(string::StringBuffer* string_buffer) override;
  Entry* ResolvePath(const std::string_view path) override;

  struct DiscInfo {
    size_t game_offset;
    size_t root_sector;
    size_t root_size;
    size_t host_size;
  };

  const Entry* root() const { return root_entry_.get(); }
  uint64_t file_count() const { return file_count_; }
  uint64_t total_file_size() const { return total_file_size_; }
  const DiscInfo& disc_info() const { return disc_info_; }

  const std::string& name() const override { return name_; }
  uint32_t attributes() const override { return 0; }
  uint32_t component_name_max_length() const override { return 255; }

  // wwe2k14 patch (Session 77): det_fs_constant - a folder's constant volume size (host_path_device.h), not the
  // image file's.
  uint32_t total_allocation_units() const override {
    if (det::FsConstant()) return 128 * 1024;
    return uint32_t(mmap_->size() / sectors_per_allocation_unit() / bytes_per_sector());
  }
  uint32_t available_allocation_units() const override { return det::FsConstant() ? 128 * 1024 : 0; }
  uint32_t sectors_per_allocation_unit() const override { return 1; }
  uint32_t bytes_per_sector() const override { return 0x200; }

 private:
  enum class Error {
    kSuccess = 0,
    kErrorOutOfMemory = -1,
    kErrorReadError = -10,
    kErrorFileMismatch = -30,
    kErrorDamagedFile = -31,
  };

  std::string name_;
  std::filesystem::path host_path_;
  std::unique_ptr<Entry> root_entry_;
  std::unique_ptr<memory::MappedMemory> mmap_;
  DiscInfo disc_info_{};
  uint64_t file_count_ = 0;
  uint64_t total_file_size_ = 0;

  typedef struct {
    uint8_t* ptr;
    size_t size;         // Size (bytes) of total image.
    size_t game_offset;  // Offset (bytes) of game partition.
    size_t root_sector;  // Offset (sector) of root.
    size_t root_offset;  // Offset (bytes) of root.
    size_t root_size;    // Size (bytes) of root.
  } ParseState;

  Error Verify(ParseState* state);
  bool VerifyMagic(ParseState* state, size_t offset);
  Error ReadAllEntries(ParseState* state, const uint8_t* root_buffer);
  // WWE 2K14 Session 26: a damaged image fails the mount instead of reading outside a table, looping or running out
  // of stack. ReadTable: one directory table (inside the image, parsed once); ReadEntry: one entry of it (inside the
  // table, visited once); depth counts both tree links and folder nesting. No logging inside the recursion (small
  // stack frames): Damaged() records the reason, Initialize logs it.
  bool ReadTable(ParseState* state, size_t offset, size_t size, DiscImageEntry* parent, uint32_t depth);
  bool ReadEntry(ParseState* state, const uint8_t* buffer, size_t dir_size, uint16_t entry_ordinal,
                 DiscImageEntry* parent, uint32_t depth, std::vector<uint8_t>& visited);
  bool Damaged(const char* what, size_t a, size_t b);
  uint64_t files_past_end_ = 0;  // files whose data lies beyond the end of the image (a truncated copy)
  std::string parse_error_;
  std::vector<size_t> tables_seen_;  // offsets of the directory tables already parsed
};

}  // namespace rex::filesystem
