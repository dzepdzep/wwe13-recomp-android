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

#pragma once

#include "rex/system/function_dispatcher.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rex/graphics/register_file.h>
#include <rex/kernel.h>
#include <rex/memory.h>
#include <rex/system/interfaces/graphics.h>
#include <rex/system/xthread.h>
#include <rex/thread/mutex.h>
#include <rex/ui/graphics_provider.h>
#include <rex/ui/presenter.h>

// Forward declarations
namespace rex {
class Memory;
namespace stream {
class ByteStream;
}  // namespace stream
}  // namespace rex

namespace rex::ui {
class WindowedAppContext;
}  // namespace rex::ui

namespace rex::graphics {

class CommandProcessor;

class GraphicsSystem : public system::IGraphicsSystem {
 public:
  virtual ~GraphicsSystem();

  virtual std::string name() const = 0;

  memory::Memory* memory() const { return memory_; }
  runtime::FunctionDispatcher* function_dispatcher() const { return function_dispatcher_; }
  system::KernelState* kernel_state() const { return kernel_state_; }
  ::rex::ui::GraphicsProvider* provider() const override { return provider_.get(); }
  ::rex::ui::Presenter* presenter() const override { return presenter_.get(); }

  X_STATUS SetupPresentation(::rex::ui::WindowedAppContext* app_context) override;
  X_STATUS SetupGuestGpu(runtime::FunctionDispatcher* function_dispatcher,
                         system::KernelState* kernel_state) override;
  bool has_presentation() const override { return presenter_ != nullptr; }
  void Shutdown() override;

  // wwe2k14 patch (Session 101, lane 233): RuntimeConfig::guest_gpu_only. Set before SetupGuestGpu. SetupGuestGpu then registers the MMIO
  // range and starts the VSync thread only: provider(), presenter() and command_processor() stay null for the system's whole life, provided SetupPresentation was not called before this flag
  // was set (Runtime::Setup sets it first, and skips SetupPresentation when it is on).
  void SetGuestGpuOnly(bool guest_gpu_only) override { guest_gpu_only_ = guest_gpu_only; }
  bool guest_gpu_only() const { return guest_gpu_only_; }

  // wwe2k14 patch (Session 101, lane 260): RuntimeConfig::guest_vblank_ours. Set before SetupGuestGpu: no VSync thread is started; the app dispatches the
  // vblank interrupt itself with the callback the game registered (VdSetGraphicsInterruptCallback -> SetInterruptCallback), read back below.
  void SetGuestVblankOurs(bool guest_vblank_ours) override { guest_vblank_ours_ = guest_vblank_ours; }
  bool guest_vblank_ours() const { return guest_vblank_ours_; }
  uint32_t interrupt_callback() const { return interrupt_callback_.load(std::memory_order_acquire); }
  uint32_t interrupt_callback_data() const { return interrupt_callback_data_.load(std::memory_order_relaxed); }

  // May be called from any thread any number of times, even during recovery
  // from a device loss.
  void OnHostGpuLossFromAnyThread(bool is_responsible);

  RegisterFile* register_file() { return &register_file_; }
  CommandProcessor* command_processor() const { return command_processor_.get(); }

  void InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) override;
  void EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) override;

  void SetInterruptCallback(uint32_t callback, uint32_t user_data) override;
  void DispatchInterruptCallback(uint32_t source, uint32_t cpu);

  virtual void ClearCaches();
  virtual void InvalidateGpuMemory();

  void InitializeShaderStorage(const std::filesystem::path& cache_root, uint32_t title_id,
                               bool blocking) override;

  bool is_paused() const { return paused_; }
  void Pause();
  void Resume();

  bool Save(::rex::stream::ByteStream* stream);
  bool Restore(::rex::stream::ByteStream* stream);

 protected:
  GraphicsSystem();

  // Backends build their provider here. Called lazily from either setup
  // entry point; with_presentation is false only on headless guest-GPU paths.
  virtual void CreateProvider(bool with_presentation) = 0;

  virtual std::unique_ptr<CommandProcessor> CreateCommandProcessor() = 0;

  static uint32_t ReadRegisterThunk(void* ppc_context, GraphicsSystem* gs, uint32_t addr);
  static void WriteRegisterThunk(void* ppc_context, GraphicsSystem* gs, uint32_t addr,
                                 uint32_t value);

  uint32_t ReadRegister(uint32_t addr);
  void WriteRegister(uint32_t addr, uint32_t value);

  // wwe2k14 patch (Session 6, O1): the arguments only go into the timeline's "vblank" event: a =
  // microseconds after its deadline the vblank fired, b = 1 if the precise timer did the wait
  // before it (0 = Sleep(1) polling), c = microseconds of the precise timer's final spin before it.
  void MarkVblank(uint32_t late_us = 0, uint32_t precise_timer = 0, uint32_t spin_us = 0);

  memory::Memory* memory_ = nullptr;
  runtime::FunctionDispatcher* function_dispatcher_ = nullptr;
  system::KernelState* kernel_state_ = nullptr;
  ::rex::ui::WindowedAppContext* app_context_ = nullptr;
  std::unique_ptr<::rex::ui::GraphicsProvider> provider_;
  bool provider_supports_presentation_ = false;
  bool guest_gpu_only_ = false;  // wwe2k14 patch (Session 101, lane 233)
  bool guest_vblank_ours_ = false;  // wwe2k14 patch (Session 101, lane 260)

  // wwe2k14 patch (Session 95, RACE R3): written by a guest thread, read by the vsync and command threads. Data first, callback
  // published with release (graphics_system.cpp). atomic<uint32_t> keeps the size and alignment of uint32_t.
  std::atomic<uint32_t> interrupt_callback_{0};
  std::atomic<uint32_t> interrupt_callback_data_{0};

  std::atomic<bool> vsync_worker_running_;
  system::object_ref<system::XHostThread> vsync_worker_thread_;

  RegisterFile register_file_;
  std::unique_ptr<CommandProcessor> command_processor_;

  bool paused_ = false;

 private:
  std::unique_ptr<::rex::ui::Presenter> presenter_;

  std::atomic_flag host_gpu_loss_reported_;
};

}  // namespace rex::graphics
