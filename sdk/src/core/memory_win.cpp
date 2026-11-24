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

#include "platform_win.h"

#include <rex/cvar.h>
#include <rex/memory/utils.h>

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP | WINAPI_PARTITION_SYSTEM | \
                            WINAPI_PARTITION_GAMES)
#define REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS
#endif

// wwe2k14 patch (Session 95 review): test only - behaves as a Windows without the two calls below, to run the refusal the player sees.
REXCVAR_DEFINE_BOOL(memory_test_no_placeholder_views, false, "Memory",
                    "Session 95 test switch: act as a Windows without VirtualAlloc2 / MapViewOfFile3, so the start stops at the memory refusal message. Default off");

namespace rex {
namespace memory {

size_t page_size() {
  static size_t value = 0;
  if (!value) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    value = si.dwPageSize;
  }
  return value;
}

size_t allocation_granularity() {
  static size_t value = 0;
  if (!value) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    value = si.dwAllocationGranularity;
  }
  return value;
}

// wwe2k14 patch (Session 95, flat guest memory): see utils.h. Desktop builds take the placeholder route below.
size_t view_offset_granularity() {
  return page_size();
}

DWORD ToWin32ProtectFlags(PageAccess access) {
  switch (access) {
    case PageAccess::kNoAccess:
      return PAGE_NOACCESS;
    case PageAccess::kReadOnly:
      return PAGE_READONLY;
    case PageAccess::kReadWrite:
      return PAGE_READWRITE;
    case PageAccess::kExecuteReadOnly:
      return PAGE_EXECUTE_READ;
    case PageAccess::kExecuteReadWrite:
      return PAGE_EXECUTE_READWRITE;
    default:
      assert_unhandled_case(access);
      return PAGE_NOACCESS;
  }
}

PageAccess ToXeniaProtectFlags(DWORD access) {
  if (access & PAGE_GUARD) {
    // Strip the page guard flag for now...
    access &= ~PAGE_GUARD;
  }

  switch (access) {
    case PAGE_NOACCESS:
      return PageAccess::kNoAccess;
    case PAGE_READONLY:
      return PageAccess::kReadOnly;
    case PAGE_READWRITE:
      return PageAccess::kReadWrite;
    case PAGE_EXECUTE_READ:
      return PageAccess::kExecuteReadOnly;
    case PAGE_EXECUTE_READWRITE:
      return PageAccess::kExecuteReadWrite;
    default:
      return PageAccess::kNoAccess;
  }
}

bool IsWritableExecutableMemorySupported() {
#ifdef REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS
  return true;
#else
  // To test FromApp functions on desktop, undefine
  // REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS and link to WindowsApp.lib.
  return false;
#endif
}

void* AllocFixed(void* base_address, size_t length, AllocationType allocation_type,
                 PageAccess access) {
  DWORD alloc_type = 0;
  switch (allocation_type) {
    case AllocationType::kReserve:
      alloc_type = MEM_RESERVE;
      break;
    case AllocationType::kCommit:
      alloc_type = MEM_COMMIT;
      break;
    case AllocationType::kReserveCommit:
      alloc_type = MEM_RESERVE | MEM_COMMIT;
      break;
    default:
      assert_unhandled_case(allocation_type);
      break;
  }
  DWORD protect = ToWin32ProtectFlags(access);
#ifdef REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS
  return VirtualAlloc(base_address, length, alloc_type, protect);
#else
  return VirtualAllocFromApp(base_address, length, ULONG(alloc_type), ULONG(protect));
#endif
}

bool DeallocFixed(void* base_address, size_t length, DeallocationType deallocation_type) {
  DWORD free_type = 0;
  switch (deallocation_type) {
    case DeallocationType::kRelease:
      free_type = MEM_RELEASE;
      length = 0;
      break;
    case DeallocationType::kDecommit:
      free_type = MEM_DECOMMIT;
      break;
    default:
      assert_unhandled_case(deallocation_type);
      break;
  }
  return VirtualFree(base_address, length, free_type) ? true : false;
}

bool Protect(void* base_address, size_t length, PageAccess access, PageAccess* out_old_access) {
  if (out_old_access) {
    *out_old_access = PageAccess::kNoAccess;
  }
  DWORD new_protect = ToWin32ProtectFlags(access);
#ifdef REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS
  DWORD old_protect = 0;
  BOOL result = VirtualProtect(base_address, length, new_protect, &old_protect);
#else
  ULONG old_protect = 0;
  BOOL result = VirtualProtectFromApp(base_address, length, ULONG(new_protect), &old_protect);
#endif
  if (!result) {
    return false;
  }
  if (out_old_access) {
    *out_old_access = ToXeniaProtectFlags(DWORD(old_protect));
  }
  return true;
}

bool QueryProtect(void* base_address, size_t& length, PageAccess& access_out) {
  access_out = PageAccess::kNoAccess;

  MEMORY_BASIC_INFORMATION info;
  ZeroMemory(&info, sizeof(info));

  SIZE_T result = VirtualQuery(base_address, &info, sizeof(info));
  if (!result) {
    return false;
  }

  length = info.RegionSize;
  access_out = ToXeniaProtectFlags(info.Protect);
  return true;
}

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path& path, size_t length,
                                          PageAccess access, bool commit) {
  DWORD protect = ToWin32ProtectFlags(access) | (commit ? SEC_COMMIT : SEC_RESERVE);
  auto full_path = "Local" / path;
#ifdef REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS
  HANDLE h =
      CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, protect, static_cast<DWORD>(length >> 32),
                         static_cast<DWORD>(length), full_path.c_str());
#else
  HANDLE h = CreateFileMappingFromApp(INVALID_HANDLE_VALUE, nullptr, ULONG(protect),
                                      ULONG64(length), full_path.c_str());
#endif
  if (!h) {
    return kFileMappingHandleInvalid;
  }
  return reinterpret_cast<FileMappingHandle>(h);
}

void CloseFileMappingHandle(FileMappingHandle handle, const std::filesystem::path& path) {
  CloseHandle(reinterpret_cast<HANDLE>(handle));
}

#ifdef REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS
// wwe2k14 patch (Session 95, flat guest memory): a view at a file offset that is not a multiple of 64 KB. MapViewOfFileEx
// refuses one; MapViewOfFile3 takes one only when it replaces a placeholder (Microsoft's MEM_REPLACE_PLACEHOLDER notes:
// the 64 KB rules for Offset and BaseAddress do not apply; proved on a pagefile SEC_RESERVE section with commit, 4 KB
// protection and unmap: out\s95\mem\ph_probe.cpp). Both calls exist from Windows 10 1803: looked up, so the exe still
// loads on an older Windows and Memory::Initialize can say why it stopped.
static void* MapFileViewAtPageOffset(HANDLE handle, void* base_address, size_t length, PageAccess access,
                                     size_t file_offset) {
  using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
  using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);
  const HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll");
  if (!kernelbase) {
    return nullptr;
  }
  const auto virtual_alloc2 =
      reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelbase, "VirtualAlloc2"));
  const auto map_view_of_file3 =
      reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(kernelbase, "MapViewOfFile3"));
  if (!virtual_alloc2 || !map_view_of_file3 || REXCVAR_GET(memory_test_no_placeholder_views)) {
    return nullptr;
  }
  const HANDLE process = GetCurrentProcess();
  void* placeholder = virtual_alloc2(process, base_address, length,
                                     MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
  if (!placeholder) {
    return nullptr;
  }
  void* mapping = map_view_of_file3(handle, process, placeholder, ULONG64(file_offset), length,
                                    MEM_REPLACE_PLACEHOLDER, ULONG(ToWin32ProtectFlags(access)), nullptr, 0);
  if (!mapping) {
    VirtualFree(placeholder, 0, MEM_RELEASE);
    return nullptr;
  }
  return mapping;
}
#endif

void* MapFileView(FileMappingHandle handle, void* base_address, size_t length, PageAccess access,
                  size_t file_offset) {
#ifdef REX_BASE_MEMORY_WIN_USE_DESKTOP_FUNCTIONS
  if (file_offset & (allocation_granularity() - 1)) {
    return MapFileViewAtPageOffset(reinterpret_cast<HANDLE>(handle), base_address, length, access, file_offset);
  }
  DWORD target_address_low = static_cast<DWORD>(file_offset);
  DWORD target_address_high = static_cast<DWORD>(file_offset >> 32);
  DWORD file_access = 0;
  switch (access) {
    case PageAccess::kReadOnly:
      file_access = FILE_MAP_READ;
      break;
    case PageAccess::kReadWrite:
      file_access = FILE_MAP_ALL_ACCESS;
      break;
    case PageAccess::kExecuteReadOnly:
      file_access = FILE_MAP_READ | FILE_MAP_EXECUTE;
      break;
    case PageAccess::kExecuteReadWrite:
      file_access = FILE_MAP_ALL_ACCESS | FILE_MAP_EXECUTE;
      break;
    case PageAccess::kNoAccess:
    default:
      assert_unhandled_case(access);
      return nullptr;
  }
  return MapViewOfFileEx(reinterpret_cast<HANDLE>(handle), file_access, target_address_high,
                         target_address_low, length, base_address);
#else
  // VirtualAlloc2FromApp and MapViewOfFile3FromApp were added in 10.0.17134.0.
  // https://docs.microsoft.com/en-us/uwp/win32-and-com/win32-apis
  HANDLE process = GetCurrentProcess();
  void* placeholder =
      VirtualAlloc2FromApp(process, base_address, length, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                           PAGE_NOACCESS, nullptr, 0);
  if (!placeholder) {
    return nullptr;
  }
  void* mapping = MapViewOfFile3FromApp(reinterpret_cast<HANDLE>(handle), process, placeholder,
                                        ULONG64(file_offset), length, MEM_REPLACE_PLACEHOLDER,
                                        ULONG(ToWin32ProtectFlags(access)), nullptr, 0);
  if (!mapping) {
    VirtualFree(placeholder, length, MEM_RELEASE);
    return nullptr;
  }
  return mapping;
#endif
}

bool UnmapFileView(FileMappingHandle handle, void* base_address, size_t length) {
  return UnmapViewOfFile(base_address) ? true : false;
}

}  // namespace memory
}  // namespace rex
