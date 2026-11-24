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

#include <rex/filesystem/devices/host_path_entry.h>

#include <algorithm>

#include <rex/filesystem.h>
#include <rex/filesystem/devices/host_path_device.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/string/utf8.h>

namespace rex::filesystem {

HostPathDevice::HostPathDevice(const std::string_view mount_path,
                               const std::filesystem::path& host_path, bool read_only,
                               bool allow_share_delete)
    : Device(mount_path),
      name_("STFS"),
      host_path_(host_path),
      read_only_(read_only),
      allow_share_delete_(allow_share_delete) {}

HostPathDevice::~HostPathDevice() = default;

bool HostPathDevice::Initialize() {
  if (!std::filesystem::exists(host_path_)) {
    if (!read_only_) {
      // Create the path.
      std::filesystem::create_directories(host_path_);
    } else {
      REXFS_ERROR("Host path does not exist");
      return false;
    }
  }

  auto root_entry = new HostPathEntry(this, nullptr, "", host_path_);
  root_entry->attributes_ = kFileAttributeDirectory;
  root_entry_ = std::unique_ptr<Entry>(root_entry);
  PopulateEntry(root_entry);

  return true;
}

void HostPathDevice::Dump(string::StringBuffer* string_buffer) {
  auto global_lock = global_critical_region_.Acquire();
  root_entry_->Dump(string_buffer, 0);
}

Entry* HostPathDevice::ResolvePath(const std::string_view path) {
  // The filesystem will have stripped our prefix off already, so the path will
  // be in the form:
  // some\PATH.foo
  auto* resolved = root_entry_->ResolvePath(path);
  if (resolved) {
    return resolved;
  }

  // Fallback to a lazy case-insensitive host lookup when an entry is missing
  // from the in-memory tree (for example because casing differs on Linux).
  auto* current_entry = static_cast<HostPathEntry*>(root_entry_.get());
  for (const auto& part : rex::string::utf8_split_path(path)) {
    if (part.empty()) {
      continue;
    }

    auto* child = current_entry->GetChild(part);
    if (!child) {
      // Stat the exact name first: enumerating a directory of hundreds of
      // archives costs milliseconds a walk. Only casing mismatch reaches
      // the scan below.
      const auto exact_path = current_entry->host_path() / rex::to_path(part);
      rex::filesystem::FileInfo exact_info;
      if (rex::filesystem::GetInfo(exact_path, &exact_info)) {
        auto* exact_child = HostPathEntry::Create(this, current_entry, exact_path, exact_info);
        if (!exact_child) {
          return nullptr;
        }
        current_entry->children_.push_back(std::unique_ptr<Entry>(exact_child));
        current_entry = static_cast<HostPathEntry*>(exact_child);
        continue;
      }

      auto child_infos = rex::filesystem::ListFiles(current_entry->host_path());
      auto match = std::find_if(child_infos.begin(), child_infos.end(), [&](const auto& info) {
        return rex::string::utf8_equal_case(rex::path_to_utf8(info.name), part);
      });
      if (match == child_infos.end()) {
        return nullptr;
      }

      auto new_child = HostPathEntry::Create(this, current_entry,
                                             current_entry->host_path() / match->name, *match);
      if (!new_child) {
        return nullptr;
      }
      child = new_child;
      current_entry->children_.push_back(std::unique_ptr<Entry>(new_child));
    }

    current_entry = static_cast<HostPathEntry*>(child);
  }

  return current_entry;
}

void HostPathDevice::PopulateEntry(HostPathEntry* parent_entry) {
  auto child_infos = rex::filesystem::ListFiles(parent_entry->host_path());
  for (auto& child_info : child_infos) {
    auto child = HostPathEntry::Create(this, parent_entry,
                                       parent_entry->host_path() / child_info.name, child_info);
    parent_entry->children_.push_back(std::unique_ptr<Entry>(child));

    if (child_info.type == rex::filesystem::FileInfo::Type::kDirectory) {
      PopulateEntry(child);
    }
  }
}

size_t HostPathDevice::ApplyOverlay(const std::filesystem::path& overlay_root, bool replace_only) {
  auto global_lock = global_critical_region_.Acquire();
  if (!root_entry_ || !std::filesystem::is_directory(overlay_root)) {
    return 0;
  }
  return OverlayEntry(static_cast<HostPathEntry*>(root_entry_.get()), overlay_root, replace_only);
}

size_t HostPathDevice::OverlayEntry(HostPathEntry* parent_entry,
                                    const std::filesystem::path& overlay_dir, bool replace_only) {
  size_t applied = 0;
  for (auto& info : rex::filesystem::ListFiles(overlay_dir)) {
    const std::string name = rex::path_to_utf8(info.name);
    auto& children = parent_entry->children_;
    auto it = std::find_if(children.begin(), children.end(), [&](const auto& child) {
      return rex::string::utf8_equal_case(child->name(), name);
    });
    const std::filesystem::path overlay_path = overlay_dir / info.name;
    if (replace_only && it == children.end()) continue;
    if (info.type == rex::filesystem::FileInfo::Type::kDirectory) {
      HostPathEntry* dir_entry;
      if (it != children.end() && ((*it)->attributes() & kFileAttributeDirectory)) {
        dir_entry = static_cast<HostPathEntry*>(it->get());
      } else {
        dir_entry = HostPathEntry::Create(this, parent_entry, overlay_path, info);
        if (!dir_entry) continue;
        if (it != children.end()) {
          it->reset(dir_entry);
        } else {
          children.push_back(std::unique_ptr<Entry>(dir_entry));
        }
      }
      applied += OverlayEntry(dir_entry, overlay_path, replace_only);
      continue;
    }
    auto* file_entry = HostPathEntry::Create(this, parent_entry, overlay_path, info);
    if (!file_entry) continue;
    if (it != children.end()) {
      // Keep the game's own spelling of the name.
      file_entry->name_ = (*it)->name();
      it->reset(file_entry);
    } else {
      children.push_back(std::unique_ptr<Entry>(file_entry));
    }
    ++applied;
  }
  return applied;
}

}  // namespace rex::filesystem
