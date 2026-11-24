/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/det.h>  // wwe2k14 patch (Session 15): det_frames
#include <optional>
#include <unordered_map>  // wwe2k14 patch (Session 100, tier E0)
#include <vector>

#include <rex/chrono/clock.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/util/string_utils.h>  // For TranslateAnsiStringAddress
#include <rex/system/xobject.h>
// #include <rex/kernel/xboxkrnl/private.h>  // TODO: JIT only
#include <rex/system/xenumerator.h>
#include <rex/system/xevent.h>
#include <rex/system/xfile.h>
#include <rex/system/xmodule.h>
#include <rex/system/xmutant.h>
#include <rex/system/xnotifylistener.h>
#include <rex/system/xsemaphore.h>
#include <rex/system/xsymboliclink.h>
#include <rex/system/xthread.h>

namespace rex::system {

XObject::XObject(Type type) : kernel_state_(nullptr), pointer_ref_count_(1), type_(type) {
  handles_.reserve(10);
}

XObject::XObject(KernelState* kernel_state, Type type)
    : kernel_state_(kernel_state),
      type_(type),
      pointer_ref_count_(1),
      guest_object_ptr_(0),
      allocated_guest_object_(false) {
  handles_.reserve(10);

  // TODO: Assert kernel_state != nullptr in this constructor.
  if (kernel_state) {
    kernel_state->object_table()->AddHandle(this, nullptr);
  }
}

XObject::~XObject() {
  // TODO: these are being asserted true... find out why
  // assert_true(handles_.empty());
  // assert_zero(pointer_ref_count_);

  if (allocated_guest_object_) {
    uint32_t ptr = guest_object_ptr_ - sizeof(X_OBJECT_HEADER);
    auto header = memory()->TranslateVirtual<X_OBJECT_HEADER*>(ptr);

    // Free the object creation info
    if (header->object_type_ptr) {
      memory()->SystemHeapFree(header->object_type_ptr);
    }

    memory()->SystemHeapFree(ptr);
  }
}

rex::Runtime* XObject::emulator() const {
  return kernel_state_->emulator_;
}
KernelState* XObject::kernel_state() const {
  return kernel_state_;
}
rex::memory::Memory* XObject::memory() const {
  return kernel_state_->memory();
}

XObject::Type XObject::type() const {
  return type_;
}

void XObject::RetainHandle() {
  kernel_state_->object_table()->RetainHandle(handles_[0]);
}

bool XObject::ReleaseHandle() {
  // FIXME: Return true when handle is actually released.
  return kernel_state_->object_table()->ReleaseHandle(handles_[0]) == X_STATUS_SUCCESS;
}

void XObject::Retain() {
  ++pointer_ref_count_;
}

void XObject::Release() {
  if (--pointer_ref_count_ == 0) {
    delete this;
  }
}

X_STATUS XObject::Delete() {
  if (kernel_state_ == nullptr) {
    // Fake return value for api-scanner
    return X_STATUS_SUCCESS;
  } else {
    if (!name_.empty()) {
      kernel_state_->object_table()->RemoveNameMapping(name_);
    }
    return kernel_state_->object_table()->RemoveHandle(handles_[0]);
  }
}

bool XObject::SaveObject(stream::ByteStream* stream) {
  stream->Write<uint32_t>(allocated_guest_object_);
  stream->Write<uint32_t>(guest_object_ptr_);

  stream->Write(uint32_t(handles_.size()));
  stream->Write(&handles_[0], handles_.size() * sizeof(X_HANDLE));

  return true;
}

bool XObject::RestoreObject(stream::ByteStream* stream) {
  allocated_guest_object_ = stream->Read<uint32_t>() > 0;
  guest_object_ptr_ = stream->Read<uint32_t>();

  handles_.resize(stream->Read<uint32_t>());
  stream->Read(&handles_[0], handles_.size() * sizeof(X_HANDLE));

  // Restore our pointer to our handles in the object table.
  for (size_t i = 0; i < handles_.size(); i++) {
    kernel_state_->object_table()->RestoreHandle(handles_[i], this);
  }

  return true;
}

object_ref<XObject> XObject::Restore(KernelState* kernel_state, Type type,
                                     stream::ByteStream* stream) {
  switch (type) {
    case Type::Enumerator:
      break;
    case Type::Event:
      return XEvent::Restore(kernel_state, stream);
    case Type::File:
      return XFile::Restore(kernel_state, stream);
    case Type::IOCompletion:
      break;
    case Type::Module:
      return XModule::Restore(kernel_state, stream);
    case Type::Mutant:
      return XMutant::Restore(kernel_state, stream);
    case Type::NotifyListener:
      return XNotifyListener::Restore(kernel_state, stream);
    case Type::Semaphore:
      return XSemaphore::Restore(kernel_state, stream);
    case Type::Session:
      break;
    case Type::Socket:
      break;
    case Type::SymbolicLink:
      return XSymbolicLink::Restore(kernel_state, stream);
    case Type::Thread:
      return XThread::Restore(kernel_state, stream);
    case Type::Timer:
      break;
    case Type::Undefined:
      break;
  }

  assert_always("No restore handler exists for this object!");
  return nullptr;
}

void XObject::SetAttributes(uint32_t obj_attributes_ptr) {
  if (!obj_attributes_ptr) {
    return;
  }

  auto name = util::TranslateAnsiStringAddress(
      memory(),
      memory::load_and_swap<uint32_t>(memory()->TranslateVirtual(obj_attributes_ptr + 4)));
  if (!name.empty()) {
    name_ = std::string(name);
    kernel_state_->object_table()->AddNameMapping(name_, handles_[0]);
  }
}

uint32_t XObject::TimeoutTicksToMs(int64_t timeout_ticks) {
  if (timeout_ticks > 0) {
    // Absolute time, based on January 1, 1601.
    // TODO(benvanik): convert time to relative time.
    assert_always();
    return 0;
  } else if (timeout_ticks < 0) {
    // Relative time.
    return (uint32_t)(-timeout_ticks / 10000);  // Ticks -> MS
  } else {
    return 0;
  }
}

X_STATUS XObject::Wait(uint32_t wait_reason, uint32_t processor_mode, uint32_t alertable,
                       uint64_t* opt_timeout) {
  auto wait_handle = GetWaitHandle();
  if (!wait_handle) {
    // Object doesn't support waiting.
    return X_STATUS_SUCCESS;
  }

  auto timeout_ms = opt_timeout ? std::chrono::milliseconds(chrono::Clock::ScaleGuestDurationMillis(
                                      TimeoutTicksToMs(*opt_timeout)))
                                : std::chrono::milliseconds::max();

  XThread::CheckTitleTermination();
  rex::thread::WaitResult result;
  // wwe2k14 patch (Session 15): det_serial: try the wait without blocking; if it would block, give the baton up
  // until the object is signalled (checked without consuming it), an APC is pending (alertable), or the timeout's
  // det vblank has come.
  if (det::SerialEnabled() && det::SerialIsGuestThread()) {
    const bool infinite = timeout_ms == std::chrono::milliseconds::max();
    const uint64_t target =
        infinite ? ~0ull : det::Vblanks() + det::VblanksForMs(uint64_t(timeout_ms.count()));
    // Session 77: the wait handle and the waiting thread (not the native handle): POSIX has no kernel object to
    // query, and a mutant is judged for this thread, not for the scheduler thread that asks.
    const std::thread::id waiter = std::this_thread::get_id();
    auto* thread = XThread::GetCurrentThread();
    // Session 77: waiting on a guest thread: ready as soon as its det slot has exited. Its host handle is signalled
    // only at the very end of its exit, after it gave the baton up, so "signalled yet?" depended on host speed (a PC
    // and a phone could pick a different next thread). Once ready, the rest of the exit needs no baton: wait for the
    // host signal here, holding the baton.
    auto* xthread = dynamic_cast<XThread*>(this);
    const int target_slot = xthread ? xthread->det_slot() : -1;
    for (;;) {
      result = rex::thread::Wait(wait_handle, alertable ? true : false, std::chrono::milliseconds(0));
      if (result != rex::thread::WaitResult::kTimeout || timeout_ms.count() == 0) break;
      if (target_slot >= 0 && det::SerialSlotExited(target_slot)) {
        result = rex::thread::Wait(wait_handle, false, std::chrono::milliseconds::max());
        break;
      }
      if (!infinite && det::Vblanks() >= target) break;
      det::SerialBlock(
          [wait_handle, waiter, thread, alertable, target_slot] {
            return (target_slot >= 0 && det::SerialSlotExitedLocked(target_slot)) ||
                   det::HandleSignaled(wait_handle, waiter) || (alertable && thread && thread->HasUserApcs());
          },
          target);
    }
  } else
  // wwe2k14 patch (Session 15): det_frames: a blocking wait counts the guest thread as blocked; a finite timeout
  // ends at a det vblank (counted from now), not in host time.
  if (det::Enabled() && det::IsGuestThread() && timeout_ms.count() != 0) {
    det::WaitScope scope;
    if (timeout_ms == std::chrono::milliseconds::max()) {
      result = rex::thread::Wait(wait_handle, alertable ? true : false, timeout_ms);
    } else {
      const uint64_t target = det::Vblanks() + det::VblanksForMs(uint64_t(timeout_ms.count()));
      do {
        result = rex::thread::Wait(wait_handle, alertable ? true : false, std::chrono::milliseconds(1));
      } while (result == rex::thread::WaitResult::kTimeout && det::Vblanks() < target);
    }
  } else {
    result = rex::thread::Wait(wait_handle, alertable ? true : false, timeout_ms);
  }
  XThread::CheckTitleTermination();
  switch (result) {
    case rex::thread::WaitResult::kSuccess:
      WaitCallback();
      return X_STATUS_SUCCESS;
    case rex::thread::WaitResult::kUserCallback:
      // Or X_STATUS_ALERTED?
      return X_STATUS_USER_APC;
    case rex::thread::WaitResult::kTimeout:
      rex::thread::MaybeYield();
      return X_STATUS_TIMEOUT;
    default:
    case rex::thread::WaitResult::kAbandoned:
    case rex::thread::WaitResult::kFailed:
      return X_STATUS_ABANDONED_WAIT_0;
  }
}

X_STATUS XObject::SignalAndWait(XObject* signal_object, XObject* wait_object, uint32_t wait_reason,
                                uint32_t processor_mode, uint32_t alertable,
                                uint64_t* opt_timeout) {
  auto timeout_ms = opt_timeout ? std::chrono::milliseconds(chrono::Clock::ScaleGuestDurationMillis(
                                      TimeoutTicksToMs(*opt_timeout)))
                                : std::chrono::milliseconds::max();

  // wwe2k14 patch (Session 15): det_serial: signal, try the wait, then the scheduler loop as in Wait.
  if (det::SerialEnabled() && det::SerialIsGuestThread()) {
    auto r = rex::thread::SignalAndWait(signal_object->GetWaitHandle(), wait_object->GetWaitHandle(),
                                        alertable ? true : false, std::chrono::milliseconds(0));
    const bool infinite = timeout_ms == std::chrono::milliseconds::max();
    const uint64_t target =
        infinite ? ~0ull : det::Vblanks() + det::VblanksForMs(uint64_t(timeout_ms.count()));
    auto* wait_handle = wait_object->GetWaitHandle();
    const std::thread::id waiter = std::this_thread::get_id();  // Session 77: see Wait
    auto* thread = XThread::GetCurrentThread();
    while (r == rex::thread::WaitResult::kTimeout && timeout_ms.count() != 0 &&
           (infinite || det::Vblanks() < target)) {
      det::SerialBlock(
          [wait_handle, waiter, thread, alertable] {
            return det::HandleSignaled(wait_handle, waiter) ||
                   (alertable && thread && thread->HasUserApcs());
          },
          target);
      r = rex::thread::Wait(wait_object->GetWaitHandle(), alertable ? true : false, std::chrono::milliseconds(0));
    }
    switch (r) {
      case rex::thread::WaitResult::kSuccess:
        wait_object->WaitCallback();
        return X_STATUS_SUCCESS;
      case rex::thread::WaitResult::kUserCallback:
        return X_STATUS_USER_APC;
      case rex::thread::WaitResult::kTimeout:
        return X_STATUS_TIMEOUT;
      default:
        return X_STATUS_ABANDONED_WAIT_0;
    }
  }
  // wwe2k14 patch (Session 15): det_frames: counted as blocked (timeouts stay host time here: rare).
  std::optional<det::WaitScope> det_scope;
  if (det::Enabled() && det::IsGuestThread() && timeout_ms.count() != 0) det_scope.emplace();
  auto result =
      rex::thread::SignalAndWait(signal_object->GetWaitHandle(), wait_object->GetWaitHandle(),
                                 alertable ? true : false, timeout_ms);
  switch (result) {
    case rex::thread::WaitResult::kSuccess:
      wait_object->WaitCallback();
      return X_STATUS_SUCCESS;
    case rex::thread::WaitResult::kUserCallback:
      // Or X_STATUS_ALERTED?
      return X_STATUS_USER_APC;
    case rex::thread::WaitResult::kTimeout:
      rex::thread::MaybeYield();
      return X_STATUS_TIMEOUT;
    default:
    case rex::thread::WaitResult::kAbandoned:
    case rex::thread::WaitResult::kFailed:
      return X_STATUS_ABANDONED_WAIT_0;
  }
}

X_STATUS XObject::WaitMultiple(uint32_t count, XObject** objects, uint32_t wait_type,
                               uint32_t wait_reason, uint32_t processor_mode, uint32_t alertable,
                               uint64_t* opt_timeout) {
  std::vector<rex::thread::WaitHandle*> wait_handles(count);
  for (size_t i = 0; i < count; ++i) {
    wait_handles[i] = objects[i]->GetWaitHandle();
    assert_not_null(wait_handles[i]);
  }

  auto timeout_ms = opt_timeout ? std::chrono::milliseconds(chrono::Clock::ScaleGuestDurationMillis(
                                      TimeoutTicksToMs(*opt_timeout)))
                                : std::chrono::milliseconds::max();

  XThread::CheckTitleTermination();
  // wwe2k14 patch (Session 15): det_serial: as XObject::Wait, over all the objects (any / all).
  if (det::SerialEnabled() && det::SerialIsGuestThread()) {
    const bool infinite = timeout_ms == std::chrono::milliseconds::max();
    const uint64_t target =
        infinite ? ~0ull : det::Vblanks() + det::VblanksForMs(uint64_t(timeout_ms.count()));
    const std::thread::id waiter = std::this_thread::get_id();  // Session 77: see Wait
    auto* thread = XThread::GetCurrentThread();
    const bool any = wait_type != 0;
    for (;;) {
      if (any) {
        auto copy = wait_handles;
        auto r = rex::thread::WaitAny(std::move(copy), alertable ? true : false, std::chrono::milliseconds(0));
        if (r.first != rex::thread::WaitResult::kTimeout || timeout_ms.count() == 0 ||
            (!infinite && det::Vblanks() >= target)) {
          XThread::CheckTitleTermination();
          switch (r.first) {
            case rex::thread::WaitResult::kSuccess:
              objects[r.second]->WaitCallback();
              return X_STATUS(r.second);
            case rex::thread::WaitResult::kUserCallback:
              return X_STATUS_USER_APC;
            case rex::thread::WaitResult::kTimeout:
              return X_STATUS_TIMEOUT;
            case rex::thread::WaitResult::kAbandoned:
              return X_STATUS(X_STATUS_ABANDONED_WAIT_0 + r.second);
            default:
              return X_STATUS_UNSUCCESSFUL;
          }
        }
      } else {
        auto copy = wait_handles;
        auto r = rex::thread::WaitAll(std::move(copy), alertable ? true : false, std::chrono::milliseconds(0));
        if (r != rex::thread::WaitResult::kTimeout || timeout_ms.count() == 0 ||
            (!infinite && det::Vblanks() >= target)) {
          XThread::CheckTitleTermination();
          switch (r) {
            case rex::thread::WaitResult::kSuccess:
              for (uint32_t i = 0; i < count; i++) objects[i]->WaitCallback();
              return X_STATUS_SUCCESS;
            case rex::thread::WaitResult::kUserCallback:
              return X_STATUS_USER_APC;
            case rex::thread::WaitResult::kTimeout:
              return X_STATUS_TIMEOUT;
            default:
              return X_STATUS_UNSUCCESSFUL;
          }
        }
      }
      det::SerialBlock(
          [wait_handles, waiter, thread, alertable, any] {
            if (alertable && thread && thread->HasUserApcs()) return true;
            bool all = true;
            for (auto* h : wait_handles) {
              const bool sig = det::HandleSignaled(h, waiter);
              if (any && sig) return true;
              all = all && sig;
            }
            return !any && all;
          },
          target);
    }
  }
  // wwe2k14 patch (Session 15): det_frames: counted as blocked; a finite timeout ends at a det vblank.
  std::optional<det::WaitScope> det_scope;
  const bool det_on = det::Enabled() && det::IsGuestThread() && timeout_ms.count() != 0;
  if (det_on) det_scope.emplace();
  const bool det_timed = det_on && timeout_ms != std::chrono::milliseconds::max();
  const uint64_t det_target =
      det_timed ? det::Vblanks() + det::VblanksForMs(uint64_t(timeout_ms.count())) : 0;
  if (det_timed) timeout_ms = std::chrono::milliseconds(1);
  if (wait_type) {
    auto handles_copy = wait_handles;
    auto result =
        rex::thread::WaitAny(std::move(wait_handles), alertable ? true : false, timeout_ms);
    while (det_timed && result.first == rex::thread::WaitResult::kTimeout && det::Vblanks() < det_target) {
      auto again = handles_copy;
      result = rex::thread::WaitAny(std::move(again), alertable ? true : false, timeout_ms);
    }
    XThread::CheckTitleTermination();
    switch (result.first) {
      case rex::thread::WaitResult::kSuccess:
        objects[result.second]->WaitCallback();

        return X_STATUS(result.second);
      case rex::thread::WaitResult::kUserCallback:
        // Or X_STATUS_ALERTED?
        return X_STATUS_USER_APC;
      case rex::thread::WaitResult::kTimeout:
        rex::thread::MaybeYield();
        return X_STATUS_TIMEOUT;
      default:
      case rex::thread::WaitResult::kAbandoned:
        return X_STATUS(X_STATUS_ABANDONED_WAIT_0 + result.second);
      case rex::thread::WaitResult::kFailed:
        return X_STATUS_UNSUCCESSFUL;
    }
  } else {
    auto handles_copy = wait_handles;
    auto result =
        rex::thread::WaitAll(std::move(wait_handles), alertable ? true : false, timeout_ms);
    while (det_timed && result == rex::thread::WaitResult::kTimeout && det::Vblanks() < det_target) {
      auto again = handles_copy;
      result = rex::thread::WaitAll(std::move(again), alertable ? true : false, timeout_ms);
    }
    XThread::CheckTitleTermination();
    switch (result) {
      case rex::thread::WaitResult::kSuccess:
        for (uint32_t i = 0; i < count; i++) {
          objects[i]->WaitCallback();
        }

        return X_STATUS_SUCCESS;
      case rex::thread::WaitResult::kUserCallback:
        // Or X_STATUS_ALERTED?
        return X_STATUS_USER_APC;
      case rex::thread::WaitResult::kTimeout:
        rex::thread::MaybeYield();
        return X_STATUS_TIMEOUT;
      default:
      case rex::thread::WaitResult::kAbandoned:
      case rex::thread::WaitResult::kFailed:
        return X_STATUS_ABANDONED_WAIT_0;
    }
  }
}

uint8_t* XObject::CreateNative(uint32_t size) {
  auto global_lock = rex::thread::global_critical_region::AcquireDirect();

  uint32_t total_size = size + sizeof(X_OBJECT_HEADER);

  auto mem = memory()->SystemHeapAlloc(total_size);
  if (!mem) {
    // Out of memory!
    return nullptr;
  }

  allocated_guest_object_ = true;
  memory()->Zero(mem, total_size);
  SetNativePointer(mem + sizeof(X_OBJECT_HEADER), true);

  auto header = memory()->TranslateVirtual<X_OBJECT_HEADER*>(mem);

  auto object_type = memory()->SystemHeapAlloc(sizeof(X_OBJECT_TYPE));
  if (object_type) {
    // Set it up in the header.
    // Some kernel method is accessing this struct and dereferencing a member
    // @ offset 0x14
    header->object_type_ptr = object_type;
  }

  return memory()->TranslateVirtual(guest_object_ptr_);
}

void XObject::SetNativePointer(uint32_t native_ptr, bool uninitialized) {
  auto global_lock = rex::thread::global_critical_region::AcquireDirect();

  // If hit: We've already setup the native ptr with CreateNative!
  assert_zero(guest_object_ptr_);

  auto header = kernel_state_->memory()->TranslateVirtual<X_DISPATCH_HEADER*>(native_ptr);

  // Memory uninitialized, so don't bother with the check.
  if (!uninitialized) {
    assert_true(!(header->wait_list_blink & 0x1));
  }

  // Stash pointer in struct.
  // FIXME: This assumes the object has a dispatch header (some don't!)
  StashHandle(header, handle());

  guest_object_ptr_ = native_ptr;
}

// wwe2k14 patch (Session 100, strip-360 tier E0): guest address of each lazily bound dispatcher header -> the handle of its host object
// (read and written under the global critical region, GetNativeObject).
static std::unordered_map<uint32_t, uint32_t>& NativeBinds() {
  static std::unordered_map<uint32_t, uint32_t> binds;
  return binds;
}
static void RecordNativeBind(uint32_t guest, uint32_t handle) { NativeBinds()[guest] = handle; }

object_ref<XObject> XObject::GetNativeObject(KernelState* kernel_state, void* native_ptr,
                                             int32_t as_type) {
  assert_not_null(native_ptr);

  // Unfortunately the XDK seems to inline some KeInitialize calls, meaning
  // we never see it and just randomly start getting passed events/timers/etc.
  // Luckily it seems like all other calls (Set/Reset/Wait/etc) are used and
  // we don't have to worry about PPC code poking the struct. Because of that,
  // we init on first use, store our handle in the struct, and dereference it
  // each time.
  // We identify this by setting wait_list_flink to a magic value. When set,
  // wait_list_blink will hold a handle to our object.

  auto global_lock = rex::thread::global_critical_region::AcquireDirect();

  auto header = reinterpret_cast<X_DISPATCH_HEADER*>(native_ptr);

  if (as_type == -1) {
    as_type = header->type;
  }

  if (header->wait_list_flink == kXObjSignature) {
    // Already initialized.
    // TODO: assert if the type of the object != as_type
    uint32_t handle = header->wait_list_blink;
    auto object = kernel_state->object_table()->LookupObject<XObject>(handle);

    // TODO(benvanik): assert nothing has been changed in the struct.
    return object;
  } else {
    // wwe2k14 patch (Session 100, strip-360 tier E0, docs\native\mem_tier_e_kernel_objects.md): every lazy bind of a guest-embedded
    // dispatcher header is recorded by guest address. A second bind of an address whose first host object is still alive means the guest
    // re-initialised the header inline (the signature overwritten) and the first host object is orphaned: its waiters would never wake.
    // Logged (the first 32 binds, every re-bind); behaviour unchanged.
    {
      static uint32_t logged = 0;
      const uint32_t guest = kernel_state->memory()->HostToGuestVirtual(native_ptr);
      auto& binds = NativeBinds();
      auto it = binds.find(guest);
      if (it != binds.end() && kernel_state->object_table()->LookupObject<XObject>(it->second)) {
        REXSYS_WARN("[wwe2k14][kobj] re-bind of guest header {:08X} (type {}): its first host object (handle {:08X}) is still alive and is "
                    "now orphaned",
                    guest, as_type, it->second);
      } else if (logged < 32) {
        ++logged;
        REXSYS_INFO("[wwe2k14][kobj] lazy bind #{} of guest header {:08X} (type {})", logged, guest, as_type);
      }
    }
    // First use, create new.
    // https://www.nirsoft.net/kernel_struct/vista/KOBJECTS.html
    XObject* object = nullptr;
    switch (as_type) {
      case 0:  // EventNotificationObject
      case 1:  // EventSynchronizationObject
      {
        auto ev = new XEvent(kernel_state);
        ev->InitializeNative(native_ptr, header);
        object = ev;
      } break;
      case 2:  // MutantObject
      {
        auto mutant = new XMutant(kernel_state);
        mutant->InitializeNative(native_ptr, header);
        object = mutant;
      } break;
      case 5:  // SemaphoreObject
      {
        auto sem = new XSemaphore(kernel_state);
        auto success = sem->InitializeNative(native_ptr, header);
        // Can't report failure to the guest at late initialization:
        assert_true(success);
        object = sem;
      } break;
      case 3:   // ProcessObject
      case 4:   // QueueObject
      case 6:   // ThreadObject
      case 7:   // GateObject
      case 8:   // TimerNotificationObject
      case 9:   // TimerSynchronizationObject
      case 18:  // ApcObject
      case 19:  // DpcObject
      case 20:  // DeviceQueueObject
      case 21:  // EventPairObject
      case 22:  // InterruptObject
      case 23:  // ProfileObject
      case 24:  // ThreadedDpcObject
      default:
        assert_always();
        return NULL;
    }

    // Stash pointer in struct.
    // FIXME: This assumes the object contains a dispatch header (some don't!)
    StashHandle(header, object->handle());
    RecordNativeBind(kernel_state->memory()->HostToGuestVirtual(native_ptr), object->handle());  // wwe2k14 tier E0

    return object_ref<XObject>(object);
  }
}

}  // namespace rex::system
