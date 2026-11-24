#pragma once
/**
 * @file        ui/vulkan/queue_submit_order.h
 * @brief       Queue submission order across threads (wwe2k14 patch, Session 6, DRAIN)
 *
 * The GPU command processor hands its submissions (replay of the recorded commands + vkQueueSubmit) to its
 * submit thread (vulkan_async_submit, and with vulkan_async_swap the swap submission as well). Other users of
 * the same VkQueue that rely on queue submission order for synchronizing with that work - the presenter, which
 * reads the guest output image written by the swap submission and signals a fence that must cover it - must
 * then not submit before that submission is on the queue.
 *
 * The producer gives such a submission a ticket (an increasing number). The submitting thread retires tickets
 * in order with MarkSubmitted, right after the vkQueueSubmit and with the queue still acquired. A dependent
 * submitter either calls AwaitSubmitted(ticket) before its own vkQueueSubmit (with the queue not acquired), or
 * hands a fence-only signal over with SignalFenceAfter so it does not have to wait at all.
 *
 * Header-only: used both by rexruntime (the presenter) and by the GPU plugin (the command processor). Shared
 * through std::shared_ptr so it outlives either of them.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <rex/perf/timeline.h>
#include <rex/ui/vulkan/device.h>

namespace rex {
namespace ui {
namespace vulkan {

class VulkanQueueSubmitOrder {
 public:
  // AwaitSubmitted calls that actually had to block, since the last TakeWaitStats.
  struct WaitStats {
    uint64_t count = 0;
    // Waits on the producer's own thread (for example, the presenter painting in the guest output thread
    // immediately after the refresh, PaintMode::kGuestOutputThreadImmediately) - those still stall it.
    uint64_t count_on_producer_thread = 0;
    uint64_t total_ns = 0;
    uint64_t max_ns = 0;
    // Fence signals handed over to the submitting thread / submitted by the caller itself.
    uint64_t fence_signals_handed_over = 0;
    uint64_t fence_signals_inline = 0;
  };

  // The thread giving out tickets, only to tell its own waits apart in the statistics.
  void SetProducerThread(std::thread::id thread_id) {
    producer_thread_.store(thread_id, std::memory_order_relaxed);
  }

  // Blocks until the submission with `ticket` is on the queue (its vkQueueSubmit has returned, successfully or
  // not - on failure the dependent submission will fail or report device loss by itself). Must not be called
  // with the queue acquired, as the submitting thread needs it.
  void AwaitSubmitted(uint64_t ticket) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (submitted_ >= ticket) {
      return;
    }
    perf::timeline::Scope timeline_scope("pwait_b", "pwait_e", uint32_t(ticket));
    std::chrono::steady_clock::time_point wait_start = std::chrono::steady_clock::now();
    submitted_cv_.wait(lock, [this, ticket] { return submitted_ >= ticket; });
    uint64_t wait_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - wait_start)
                                    .count());
    ++wait_stats_.count;
    if (std::this_thread::get_id() == producer_thread_.load(std::memory_order_relaxed)) {
      ++wait_stats_.count_on_producer_thread;
    }
    wait_stats_.total_ns += wait_ns;
    wait_stats_.max_ns = std::max(wait_stats_.max_ns, wait_ns);
  }

  // Returns true if the submitting thread will signal `fence` with a fence-only vkQueueSubmit right after the
  // submission with `ticket` (in the same queue acquisition, so the signal covers it). Returns false if that
  // submission is already on the queue - the caller then submits the signal itself, which is after it in
  // submission order anyway. The fence must be unsignaled and not used otherwise until signaled.
  bool SignalFenceAfter(uint64_t ticket, VkFence fence) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (submitted_ >= ticket) {
      ++wait_stats_.fence_signals_inline;
      return false;
    }
    fence_signals_.emplace_back(ticket, fence);
    ++wait_stats_.fence_signals_handed_over;
    return true;
  }

  // Submitting thread: call with `queue` acquired, right after the vkQueueSubmit of the submission with
  // `ticket` (tickets in increasing order), whether it has succeeded or not, so waiters never wait forever.
  // Submits the fence signals handed over for it (and for any earlier ticket). Returns the number of
  // fence-only submissions that have failed.
  // wwe2k14 patch (Session 7, SUBMITFIX): those are kept, in order, for RetryFailedFenceSignals unless the device
  // is lost; `last_error` (optional) receives the result of the failure.
  uint32_t MarkSubmitted(uint64_t ticket, const VulkanDevice::Functions& dfn, VkQueue queue,
                         VkResult* last_error = nullptr) {
    // Take the signals in the same critical section as the ticket is retired, so a signal is either taken
    // here or refused by SignalFenceAfter (and submitted by its caller), never lost.
    fence_signals_submitting_.clear();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      submitted_ = std::max(submitted_, ticket);
      auto fence_signals_kept_end = fence_signals_.begin();
      for (auto it = fence_signals_.begin(); it != fence_signals_.end(); ++it) {
        if (it->first <= ticket) {
          fence_signals_submitting_.push_back(it->second);
        } else {
          *(fence_signals_kept_end++) = *it;
        }
      }
      fence_signals_.erase(fence_signals_kept_end, fence_signals_.end());
      // wwe2k14 patch (Session 73): still unsubmitted until the loop below has submitted them.
      fence_signals_unsubmitted_.insert(fence_signals_unsubmitted_.end(), fence_signals_submitting_.begin(),
                                        fence_signals_submitting_.end());
    }
    uint32_t failed_count = 0;
    for (VkFence fence : fence_signals_submitting_) {
      // wwe2k14 patch (Session 7, SUBMITFIX): the caller of SignalFenceAfter already counts the fence as pending
      // (VulkanSubmissionTracker), so a wait on it would never end if the signal were just dropped - keep it for
      // RetryFailedFenceSignals (unless the device is lost: dropped as before), and after a failure the later
      // ones too, so the fences are still signaled in the order they were handed over in (a later signal makes
      // the tracker take the earlier fences as done).
      if (!fence_signals_failed_.empty()) {
        fence_signals_failed_.push_back(fence);
        ++failed_count;
        continue;
      }
      // "Fence signal operations that are defined by vkQueueSubmit additionally include in the first
      //  synchronization scope all commands that occur earlier in submission order."
      VkResult result = dfn.vkQueueSubmit(queue, 0, nullptr, fence);
      if (result != VK_SUCCESS) {
        ++failed_count;
        if (last_error) {
          *last_error = result;
        }
        if (result != VK_ERROR_DEVICE_LOST) {
          fence_signals_failed_.push_back(fence);
        }
      }
    }
    UpdateUnsubmitted();
    submitted_cv_.notify_all();
    return failed_count;
  }

  // wwe2k14 patch (Session 7, SUBMITFIX): submitting thread, with `queue` still acquired since the MarkSubmitted
  // that has returned failures (so that no fence signal submitted directly by a caller of SignalFenceAfter, refused
  // for the retired ticket, gets onto the queue ahead of them): submits the kept fence signals again, in order,
  // until one fails. Returns the number not submitted - kept for another retry, or dropped if the device is lost;
  // `last_error` receives the result of the failure.
  uint32_t RetryFailedFenceSignals(const VulkanDevice::Functions& dfn, VkQueue queue,
                                   VkResult& last_error) {
    size_t signaled_count = 0;
    VkResult result = VK_SUCCESS;
    for (; signaled_count < fence_signals_failed_.size(); ++signaled_count) {
      result = dfn.vkQueueSubmit(queue, 0, nullptr, fence_signals_failed_[signaled_count]);
      if (result != VK_SUCCESS) {
        break;
      }
    }
    fence_signals_failed_.erase(fence_signals_failed_.begin(),
                                fence_signals_failed_.begin() + signaled_count);
    UpdateUnsubmitted();
    submitted_cv_.notify_all();
    uint32_t failed_count = uint32_t(fence_signals_failed_.size());
    if (result != VK_SUCCESS) {
      last_error = result;
      if (result == VK_ERROR_DEVICE_LOST) {
        fence_signals_failed_.clear();
      }
    }
    return failed_count;
  }

  // wwe2k14 patch (Session 7, SUBMITFIX review): submitting thread, when it stops retrying: forgets the kept fence
  // signals (those fences stay unsignaled, as before SUBMITFIX), so later signals are no longer held behind them.
  void DropFailedFenceSignals() {
    fence_signals_failed_.clear();
    UpdateUnsubmitted();
    submitted_cv_.notify_all();
  }

  // wwe2k14 patch (Session 73): true while a fence handed over by SignalFenceAfter is not on the queue yet. Until then
  // no other thread may wait on or query it: vkQueueSubmit's fence is externally synchronized (validation
  // UNASSIGNED-Threading-MultipleThreads), and the Adreno driver never wakes a wait on a fence it has not received.
  bool FenceSignalPending(VkFence fence) {
    std::lock_guard<std::mutex> lock(mutex_);
    return FenceSignalPendingLocked(fence);
  }
  // Blocks until `fence` (handed over by SignalFenceAfter) is on the queue, or has been dropped.
  void AwaitFenceSignalSubmitted(VkFence fence) {
    std::unique_lock<std::mutex> lock(mutex_);
    submitted_cv_.wait(lock, [this, fence] { return !FenceSignalPendingLocked(fence); });
  }

  // Submitting thread: the taken fence signals not submitted are those kept for a retry.
  void UpdateUnsubmitted() {
    std::lock_guard<std::mutex> lock(mutex_);
    fence_signals_unsubmitted_ = fence_signals_failed_;
  }
  bool FenceSignalPendingLocked(VkFence fence) const {
    for (const auto& signal : fence_signals_) {
      if (signal.second == fence) return true;
    }
    return std::find(fence_signals_unsubmitted_.begin(), fence_signals_unsubmitted_.end(), fence) !=
           fence_signals_unsubmitted_.end();
  }

  WaitStats TakeWaitStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    WaitStats wait_stats = wait_stats_;
    wait_stats_ = WaitStats();
    return wait_stats;
  }

 private:
  std::mutex mutex_;
  std::condition_variable submitted_cv_;
  // Last retired ticket (0 - none).
  uint64_t submitted_ = 0;
  // <Ticket, fence> handed over by SignalFenceAfter, not submitted yet.
  std::vector<std::pair<uint64_t, VkFence>> fence_signals_;
  // Only accessed by the submitting thread (MarkSubmitted).
  std::vector<VkFence> fence_signals_submitting_;
  // wwe2k14 patch (Session 7, SUBMITFIX): fence signals whose vkQueueSubmit has failed, in handover order, for
  // RetryFailedFenceSignals. Only accessed by the submitting thread.
  std::vector<VkFence> fence_signals_failed_;
  // wwe2k14 patch (Session 73): taken by MarkSubmitted but not on the queue yet (under mutex_).
  std::vector<VkFence> fence_signals_unsubmitted_;
  std::atomic<std::thread::id> producer_thread_{};
  WaitStats wait_stats_;
};

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
