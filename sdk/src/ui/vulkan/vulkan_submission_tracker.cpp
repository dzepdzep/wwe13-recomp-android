/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <atomic>
#include <chrono>
#include <cstdint>

#include <rex/assert.h>
#include <rex/perf/hang_beats.h>  // wwe2k14 patch (Session 95, HANG)
#include <rex/logging.h>  // wwe2k14 patch (Session 93, log only)
#include <rex/ui/vulkan/queue_submit_order.h>  // wwe2k14 patch (Session 73)
#include <rex/ui/vulkan/fence_oracle.h>  // WWE13
#include <rex/ui/vulkan/submission_tracker.h>
#include <rex/ui/vulkan/util.h>

namespace rex {
namespace ui {
namespace vulkan {

VulkanSubmissionTracker::FenceAcquisition::~FenceAcquisition() {
  if (!submission_tracker_) {
    // Dropped submission or left after std::move.
    return;
  }
  assert_true(submission_tracker_->fence_acquired_ == fence_);
  if (fence_ != VK_NULL_HANDLE) {
    if (signal_failed_) {
      // Left in the unsignaled state.
      submission_tracker_->fences_reclaimed_.push_back(fence_);
    } else {
      // Left in the pending state.
      submission_tracker_->fences_pending_.emplace_back(submission_tracker_->submission_current_,
                                                        fence_);
    }
    submission_tracker_->fence_acquired_ = VK_NULL_HANDLE;
  }
  ++submission_tracker_->submission_current_;
}

void VulkanSubmissionTracker::Shutdown() {
  AwaitAllSubmissionsCompletion();
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  for (VkFence fence : fences_reclaimed_) {
    fence_oracle::Oracle::Get().Forget(fence);
    dfn.vkDestroyFence(device, fence, nullptr);
  }
  fences_reclaimed_.clear();
  for (const std::pair<uint64_t, VkFence>& fence_pair : fences_pending_) {
    fence_oracle::Oracle::Get().Forget(fence_pair.second);
    dfn.vkDestroyFence(device, fence_pair.second, nullptr);
  }
  fences_pending_.clear();
  assert_true(fence_acquired_ == VK_NULL_HANDLE);
  util::DestroyAndNullHandle(dfn.vkDestroyFence, device, fence_acquired_);
}

void VulkanSubmissionTracker::FenceAcquisition::SubmissionFailedOrDropped() {
  if (!submission_tracker_) {
    return;
  }
  assert_true(submission_tracker_->fence_acquired_ == fence_);
  if (fence_ != VK_NULL_HANDLE) {
    submission_tracker_->fences_reclaimed_.push_back(fence_);
  }
  submission_tracker_->fence_acquired_ = VK_NULL_HANDLE;
  fence_ = VK_NULL_HANDLE;
  // No submission acquisition from now on, don't increment the current
  // submission index as well.
  submission_tracker_ = VK_NULL_HANDLE;
}

bool VulkanSubmissionTracker::FenceHandoverPending(VkFence fence, bool wait) {
  for (auto it = handed_over_.begin(); it != handed_over_.end(); ++it) {
    if (it->first != fence) continue;
    if (it->second && it->second->FenceSignalPending(fence)) {
      if (!wait) return true;
      it->second->AwaitFenceSignalSubmitted(fence);
    }
    handed_over_.erase(it);
    return false;
  }
  return false;
}

uint64_t VulkanSubmissionTracker::UpdateAndGetCompletedSubmission() {
  if (!fences_pending_.empty()) {
    const VulkanDevice::Functions& dfn = vulkan_device_->functions();
    const VkDevice device = vulkan_device_->device();
    while (!fences_pending_.empty()) {
      const std::pair<uint64_t, VkFence>& pending_pair = fences_pending_.front();
      assert_true(pending_pair.first > submission_completed_on_gpu_);
      if (FenceHandoverPending(pending_pair.second, false) ||  // wwe2k14 patch (Session 73)
          !fence_oracle::Oracle::Get().Poll(vulkan_device_, pending_pair.second)) {  // WWE13 (fence_oracle.h)
        break;
      }
      fences_reclaimed_.push_back(pending_pair.second);
      submission_completed_on_gpu_ = pending_pair.first;
      fences_pending_.pop_front();
    }
  }
  return submission_completed_on_gpu_;
}

bool VulkanSubmissionTracker::AwaitSubmissionCompletion(uint64_t submission_index) {
  // The tracker itself can't give a submission index for a submission that
  // hasn't even started being recorded yet, the client has provided a
  // completely invalid value or has done overly optimistic math if such an
  // index has been obtained somehow.
  assert_true(submission_index <= submission_current_);
  // Waiting for the current submission is fine if there was a failure or a
  // refusal to submit, and the submission index wasn't incremented, but still
  // need to release objects referenced in the dropped submission (while
  // shutting down, for instance - in this case, waiting for the last successful
  // submission, which could have also referenced the objects from the new
  // submission - we can't know since the client has already overwritten its
  // last usage index, would correctly ensure that GPU usage of the objects is
  // not pending). Waiting for successful submissions, but failed signals, will
  // result in a true race condition, however, but waiting for the closest
  // successful signal is the best approximation - also retrying to signal in
  // this case.
  // Go from the most recent to wait only for one fence, which includes all the
  // preceding ones.
  // "Fence signal operations that are defined by vkQueueSubmit additionally
  // include in the first synchronization scope all commands that occur earlier
  // in submission order."
  size_t reclaim_end = fences_pending_.size();
  if (reclaim_end) {
    const VulkanDevice::Functions& dfn = vulkan_device_->functions();
    const VkDevice device = vulkan_device_->device();
    while (reclaim_end) {
      const std::pair<uint64_t, VkFence>& pending_pair = fences_pending_[reclaim_end - 1];
      assert_true(pending_pair.first > submission_completed_on_gpu_);
      // wwe2k14 patch (Session 73): a handed-over fence not on the queue yet: waited for (if this is a wait) until it
      // is, else not queried.
      if (FenceHandoverPending(pending_pair.second, pending_pair.first <= submission_index)) {
        --reclaim_end;
        continue;
      }
      if (pending_pair.first <= submission_index) {
        // Wait if requested.
#if REX_PLATFORM_ANDROID
        // wwe2k14 Android port: Adreno returns VK_TIMEOUT from an infinite wait early (see the command processor's
        // AwaitSubmissionCompletion); wait in 1 s steps.
        VkResult wait_result;
        // wwe2k14 patch (Session 93, log only; the Adreno 720 stock-driver freeze): a wait still running at 1 s, then every 5 s (64 lines).
        const auto stuck_start = std::chrono::steady_clock::now();
        int64_t stuck_next_ms = 1000;
        do {
          wait_result = dfn.vkWaitForFences(device, 1, &pending_pair.second, VK_TRUE, 1000000000ull);
          // wwe2k14 patch (Session 95, HANG): a wait the guard can name.
          if (wait_result == VK_TIMEOUT) {
            rex::perf::hang_beats::Get().fence_steps.fetch_add(1, std::memory_order_relaxed);
            const int64_t stuck_ms = int64_t(
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - stuck_start).count());
            static std::atomic<uint32_t> stuck_lines{0};
            if (stuck_ms >= stuck_next_ms) {
              stuck_next_ms = stuck_ms + 5000;
              if (stuck_lines.fetch_add(1, std::memory_order_relaxed) < 64) {
                REXLOG_WARN("[vk][wait][stuck] a submission tracker's fence wait: still waiting after {} ms (awaited submission {}, "
                            "this fence's {}, completed {}): the GPU has not finished it",
                            stuck_ms, submission_index, pending_pair.first, submission_completed_on_gpu_);
              }
            }
          }
        } while (wait_result == VK_TIMEOUT);
        if (wait_result == VK_SUCCESS) {
          break;
        }
#else
        if (dfn.vkWaitForFences(device, 1, &pending_pair.second, VK_TRUE, UINT64_MAX) ==
            VK_SUCCESS) {
          break;
        }
#endif
      }
      // Just refresh the completed submission.
      if (fence_oracle::Oracle::Get().Poll(vulkan_device_, pending_pair.second)) {  // WWE13 (fence_oracle.h)
        break;
      }
      --reclaim_end;
    }
    if (reclaim_end) {
      submission_completed_on_gpu_ = fences_pending_[reclaim_end - 1].first;
      for (; reclaim_end; --reclaim_end) {
        fences_reclaimed_.push_back(fences_pending_.front().second);
        fences_pending_.pop_front();
      }
    }
  }
  return submission_completed_on_gpu_ == submission_index;
}

VulkanSubmissionTracker::FenceAcquisition
VulkanSubmissionTracker::AcquireFenceToAdvanceSubmission() {
  assert_true(fence_acquired_ == VK_NULL_HANDLE);
  // Reclaim fences if the client only gets the completed submission index or
  // awaits in special cases such as shutdown.
  UpdateAndGetCompletedSubmission();
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  if (!fences_reclaimed_.empty()) {
    VkFence reclaimed_fence = fences_reclaimed_.back();
    fence_oracle::Oracle::Get().Forget(reclaimed_fence);  // WWE13 (fence_oracle.h)
    if (dfn.vkResetFences(device, 1, &reclaimed_fence) == VK_SUCCESS) {
      fence_acquired_ = fences_reclaimed_.back();
      fences_reclaimed_.pop_back();
    }
  }
  if (fence_acquired_ == VK_NULL_HANDLE) {
    VkFenceCreateInfo fence_create_info;
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_create_info.pNext = nullptr;
    fence_create_info.flags = 0;
    // May fail, a null fence is handled in FenceAcquisition.
    dfn.vkCreateFence(device, &fence_create_info, nullptr, &fence_acquired_);
  }
  return FenceAcquisition(*this, fence_acquired_);
}

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
