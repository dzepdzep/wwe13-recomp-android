// WWE13 (Android, Turnip on KGSL): a non-blocking fence status.
//
// On the KGSL kernel driver a timestamp wait with a timeout of 0 ms means "wait forever" (adreno_drawctxt_wait), and the
// Turnip build in use passes a zero relative timeout straight through: vkGetFenceStatus and vkWaitForFences(timeout 0)
// - the "is it done yet?" polls - block until the GPU finishes that submission. The command thread polled after every
// submission and the presenter's refresh tracker before every swap: a profile of the command thread (off-CPU included)
// had 52 % of its wall time in kgsl_syncobj_wait, CPU and GPU running in turn instead of side by side.
//
// Poll() answers without blocking: an unknown fence is handed to a watcher thread that waits for it (blocking is fine
// there) and reports it done. A fence the caller resets or destroys must be Forget()-ed first (the watcher may be
// waiting on it). REX_FENCE_ORACLE=0 turns it off (Poll then asks the driver as before).
#pragma once

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_set>

#include <rex/ui/vulkan/device.h>

namespace rex::ui::vulkan::fence_oracle {

inline bool Enabled() {
#if REX_PLATFORM_ANDROID
  static const bool on = [] {
    const char* v = std::getenv("REX_FENCE_ORACLE");
    return !(v && std::strcmp(v, "0") == 0);
  }();
  return on;
#else
  return false;
#endif
}

class Oracle {
 public:
  static Oracle& Get() {
    static Oracle* oracle = new Oracle();  // (never destroyed: the thread outlives static destruction order)
    return *oracle;
  }

  // True when the fence is signaled; never blocks while it is not.
  bool Poll(const VulkanDevice* device, VkFence fence) {
    if (!Enabled()) {
      return device->functions().vkGetFenceStatus(device->device(), fence) == VK_SUCCESS;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (done_.erase(fence)) {
      return true;
    }
    if (queued_.insert(fence).second) {
      if (!device_) {
        device_ = device;
        thread_ = std::thread([this] { Run(); });
        thread_.detach();
      }
      queue_.push_back(fence);
      cv_.notify_one();
    }
    return false;
  }

  // Before a fence is reset or destroyed (or re-used after the caller waited for it itself).
  void Forget(VkFence fence) {
    if (!Enabled() || fence == VK_NULL_HANDLE) return;
    std::unique_lock<std::mutex> lock(mutex_);
    if (queued_.empty() && done_.empty()) return;
    // (erased first: a watcher still waiting on it gives up at its next 1 s step)
    if (queued_.erase(fence)) {
      for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (*it == fence) {
          queue_.erase(it);
          break;
        }
      }
    }
    done_.erase(fence);
    idle_cv_.wait(lock, [&] { return current_ != fence; });
  }

 private:
  void Run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      cv_.wait(lock, [&] { return !queue_.empty(); });
      const VkFence fence = queue_.front();
      queue_.pop_front();
      current_ = fence;
      lock.unlock();
      VkResult result;
      do {
        result = device_->functions().vkWaitForFences(device_->device(), 1, &fence, VK_TRUE, 1000000000ull);
      } while (result == VK_TIMEOUT && StillWanted(fence));
      lock.lock();
      current_ = VK_NULL_HANDLE;
      if (queued_.erase(fence) && result == VK_SUCCESS) {
        done_.insert(fence);
      }
      idle_cv_.notify_all();
    }
  }

  bool StillWanted(VkFence fence) {
    std::lock_guard<std::mutex> lock(mutex_);
    return queued_.count(fence) != 0;
  }

  std::mutex mutex_;
  std::condition_variable cv_, idle_cv_;
  std::deque<VkFence> queue_;
  std::unordered_set<VkFence> queued_, done_;
  VkFence current_ = VK_NULL_HANDLE;
  const VulkanDevice* device_ = nullptr;
  std::thread thread_;
};

}  // namespace rex::ui::vulkan::fence_oracle
