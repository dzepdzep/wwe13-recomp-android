// wwe2k14 patch (Session 92, M3a): a resolve record reaches the owned frame without the register file or the render target cache.
//
// VulkanCommandProcessor::OwnedResolveRecord is what VulkanRenderTargetCache::Resolve does for a copy the owned frame serves
// (render_target_cache.cpp, "Copying" and the owned_wb destination block), taken apart from the cache: OwnedTakeResolve (served from a pass,
// a record, a defined fill, or dropped and counted), the page marks of a served copy, and - for a copy whose bytes the CPU reads back (an armed
// write-back) - the destination (the 1x shared memory range or the scaled resolve buffer range, a transient descriptor) and the dispatch
// (OwnedWriteBackRun), then the marks the cache made. Not here: the cache's stock copy paths (the truth images / the EDRAM), the cache's
// clear (the owned images are cleared by the game's clear records; the cache's host render targets / EDRAM are not read under the owned-only
// latch - counted as clears_skipped) and the texture-only record machinery.
//
// Called from IssueCopyFrom with the ResolveInfo that D3dCopyAccess::Resolve built from the record (draw_util::GetResolveInfoFromInput).
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/cvar.h>
#include <rex/graphics/free_counters.h>
#include <rex/graphics/shared_memory.h>
#include <rex/logging.h>

#include <algorithm>
#include <cstdint>

REXCVAR_DECLARE(bool, gpu_native_truth_defer_dest_use);         // (render_target_cache.cpp)
REXCVAR_DECLARE(bool, gpu_native_truth_defer_dest_use_scaled);  // (render_target_cache.cpp)

namespace rex::graphics::vulkan {

bool VulkanCommandProcessor::OwnedResolveRecord(const draw_util::ResolveInfo& resolve_info, uint32_t& written_address_out,
                                                uint32_t& written_length_out) {
  namespace fc = rex::graphics::free_counters;
  fc::OwnedResolveCounters& c = fc::g_or;
  written_address_out = 0;
  written_length_out = 0;
  ++c.calls;
  if (c.calls % 1200 == 0) {
    REXGPU_INFO("[d3d][owned-resolve] {} record resolves: served from the owned frame {}, written into memory from our image {} (write-backs that "
                "wrote nothing {}), dropped {} (the owned frame did not take them), nothing to copy {}, read back (queued or immediate) {}, "
                "clearing resolves whose cache clear was skipped {}, EDRAM renames skipped {}, write-back without a copy shader {}, scaled range failures {}, "
                "shadow-chain copy skips by source (0-3 colour, 4 depth) {}/{}/{}/{}/{}",
                c.calls, c.served, c.written, c.wb_failed, c.not_taken, c.nothing, c.read_back, c.clears_skipped, c.rename_skipped,
                c.wb_no_shader, c.range_failed, c.chain_by_src[0], c.chain_by_src[1], c.chain_by_src[2], c.chain_by_src[3], c.chain_by_src[4]);
  }

  // Nothing to copy / clear.
  if (!resolve_info.coordinate_info.width_div_8 || !resolve_info.height_div_8) {
    ++c.nothing;
    return true;
  }

  const uint32_t scale_x = texture_cache_->draw_resolution_scale_x(), scale_y = texture_cache_->draw_resolution_scale_y();
  const bool draw_resolution_scaled = texture_cache_->IsDrawResolutionScaled();
  bool copied = false;
  if (resolve_info.copy_dest_extent_length) {
    draw_util::ResolveCopyShaderConstants copy_shader_constants;
    uint32_t copy_group_count_x, copy_group_count_y;
    const draw_util::ResolveCopyShaderIndex copy_shader =
        resolve_info.GetCopyShader(scale_x, scale_y, copy_shader_constants, copy_group_count_x, copy_group_count_y);
    const bool owned_taken = NativeOwnedResolve(resolve_info, copy_shader != draw_util::ResolveCopyShaderIndex::kUnknown);
    const bool owned_wb = owned_taken && NativeOwnedWriteBackArmed();
    if (owned_taken && !owned_wb) {
      // A served copy: our image holds it, the page's samplers are given that image. (At a resolution scale its pages marked scaled as the game's
      // copy marks them; the range's textures invalidated only when a page is newly marked. A derived cube's face at 1x marks its range resolved.)
      ++c.served;
      copied = true;
      if (draw_resolution_scaled) {
        texture_cache_->MarkRangeScaledResolved(resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length, true);
      } else if (NativeOwnedMarkResolved()) {
        texture_cache_->MarkRangeAsResolved(resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length);
      }
    } else if (!owned_taken) {
      // The owned frame did not take the copy (no frame to serve it: the native renderer is not replacing). Under the owned-only latch there is no
      // other way to run it (no stock copy from the EDRAM / the truth images): dropped, counted.
      ++c.not_taken;
      if (c.not_taken <= 12) {
        REXGPU_ERROR("[d3d][owned-resolve] frame {}: a resolve record was NOT TAKEN by the owned frame (lost): source {} colour base {} depth base {} dest 0x{:08X}",
                     frame_current_, uint32_t(resolve_info.rb_copy_control.copy_src_select), uint32_t(resolve_info.color_original_base),
                     uint32_t(resolve_info.depth_original_base), resolve_info.copy_dest_base);
      }
      copied = true;
    } else if (copy_shader == draw_util::ResolveCopyShaderIndex::kUnknown) {
      // (Armed, but a copy shape the destination block cannot run: as the cache, nothing is written; OwnedTakeResolve's next call says so loudly.)
      ++c.wb_no_shader;
    } else {
      // The write-back's destination block (the cache's, without the stock copy paths and the truth-image machinery).
      const draw_util::ResolveCopyShaderInfo& copy_shader_info = draw_util::resolve_copy_shader_info[size_t(copy_shader)];
      const uint32_t copy_dest_range_unscaled =
          resolve_info.copy_dest_extent_start - resolve_info.copy_dest_base + resolve_info.copy_dest_extent_length;
      uint64_t copy_dest_base = resolve_info.copy_dest_base;
      uint64_t copy_dest_range_length = copy_dest_range_unscaled;
      uint64_t copy_dest_use_start = resolve_info.copy_dest_extent_start;
      uint64_t copy_dest_use_length = resolve_info.copy_dest_extent_length;
      if (draw_resolution_scaled) {
        if (!texture_cache_->GetScaledResolveRange(resolve_info.copy_dest_base, copy_dest_range_unscaled, copy_shader_info.dest_bpe_log2,
                                                   copy_dest_base, copy_dest_range_length) ||
            !texture_cache_->GetScaledResolveRange(resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length,
                                                   copy_shader_info.dest_bpe_log2, copy_dest_use_start, copy_dest_use_length)) {
          ++c.range_failed;
          REXGPU_ERROR("[d3d][owned-resolve] failed to map the scaled resolve destination range (base={:08X}, length={:08X})",
                       resolve_info.copy_dest_base, copy_dest_range_unscaled);
          return false;
        }
      }
      // A new resolve for the native renderer's bookkeeping (texture-only records it overlaps are decided before it writes).
      NativeTruthBeforeResolveWrite(resolve_info);
      bool copy_dest_committed;
      if (draw_resolution_scaled) {
        copy_dest_committed =
            texture_cache_->CommitScaledResolveRange(resolve_info.copy_dest_base, copy_dest_range_unscaled, copy_shader_info.dest_bpe_log2);
      } else {
        SharedMemory::RamReaderScope ram_reader(*shared_memory_, SharedMemory::RamReader::kResolveDest);
        copy_dest_committed = shared_memory_->RequestRange(resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length);
      }
      if (!copy_dest_committed) {
        REXGPU_ERROR("[d3d][owned-resolve] failed to obtain the resolve destination memory region");
      } else {
        const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
        const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
        const VkDevice device = vulkan_device->device();
        const VkDescriptorSet descriptor_set_dest = AllocateSingleTransientDescriptor(SingleTransientDescriptorLayout::kStorageBufferCompute);
        if (descriptor_set_dest != VK_NULL_HANDLE) {
          VkDescriptorBufferInfo dest_buffer_info;
          dest_buffer_info.buffer = draw_resolution_scaled ? texture_cache_->scaled_resolve_buffer() : shared_memory_->buffer();
          dest_buffer_info.offset = copy_dest_base;
          dest_buffer_info.range = copy_dest_range_length;
          VkWriteDescriptorSet write;
          write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          write.pNext = nullptr;
          write.dstSet = descriptor_set_dest;
          write.dstBinding = 0;
          write.dstArrayElement = 0;
          write.descriptorCount = 1;
          write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          write.pImageInfo = nullptr;
          write.pBufferInfo = &dest_buffer_info;
          write.pTexelBufferView = nullptr;
          dfn.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

          // The destination's use (as the cache): held until a dispatch writes it (dropped if none does), else at once.
          if (draw_resolution_scaled && REXCVAR_GET(gpu_native_truth_defer_dest_use_scaled) && NativeTruthActive()) {
            NativeHoldScaledResolveDestUse(copy_dest_use_start, copy_dest_use_length);
          } else if (draw_resolution_scaled) {
            texture_cache_->UseScaledResolveBufferForWrite(copy_dest_use_start, copy_dest_use_length);
          } else if (REXCVAR_GET(gpu_native_truth_defer_dest_use) && NativeTruthActive()) {
            NativeHoldResolveDestUse(uint32_t(copy_dest_use_start), uint32_t(copy_dest_use_length));
          } else {
            shared_memory_->Use(VulkanSharedMemory::Usage::kComputeWrite,
                                std::pair<uint32_t, uint32_t>(uint32_t(copy_dest_use_start), uint32_t(copy_dest_use_length)));
          }
          // The write-back: the owned image's bytes into the destination. One that cannot run writes nothing (never a stock copy: it would read
          // EDRAM tiles that never got this pass); the owned frame counted it.
          draw_util::ResolveCopyShaderConstants wb_constants = copy_shader_constants;
          if (!draw_resolution_scaled) wb_constants.dest_base -= uint32_t(dest_buffer_info.offset);
          NativeTruthBeforeMemoryWrite(resolve_info);  // (M3a review P4: texture-only records this copy overlaps go into memory BEFORE our bytes, as the cache did)
          const bool wrote = NativeOwnedWriteBackRun(resolve_info, wb_constants, descriptor_set_dest, copy_group_count_x, copy_group_count_y);
          NativeDropResolveDestUse();  // (nothing else wrote the destination)
          if (wrote) {
            ++c.written;
            texture_cache_->MarkRangeAsResolved(resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length);
          } else {
            ++c.wb_failed;
            if (draw_resolution_scaled) {
              texture_cache_->MarkRangeScaledResolved(resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length, true);
            }
          }
          if (!draw_resolution_scaled) {
            shared_memory_->NoteResolveWritten(resolve_info.copy_dest_extent_start, resolve_info.copy_dest_extent_length);
          }
          texture_cache_->EagerNoteResolve();
          NativeAfterResolveMarked();
          NativeTruthAfterResolveMarked(resolve_info);
          NativeOwnedMarkResolved();  // (consumes a derived cube's flag)
          if (wrote) {
            written_address_out = resolve_info.copy_dest_extent_start;
            written_length_out = resolve_info.copy_dest_extent_length;
            if (native_resolve_readback_) ++c.read_back;
          }
          copied = true;
        }
      }
    }
  } else {
    copied = true;
  }

  // Clearing: the cache's clear works on its own host render targets / the EDRAM buffer, which the owned frame neither draws into nor reads under
  // the owned-only latch; the owned images are cleared by the game's clear records. Skipped, counted.
  if (resolve_info.IsClearingDepth() || resolve_info.IsClearingColor()) {
    ++c.clears_skipped;
    if (c.clears_skipped <= 12) {  // (M3a review P3: named - the bar is 0)
      REXGPU_INFO("[d3d][owned-resolve] frame {}: a CLEARING resolve (no pending clear exists): source {} colour base {} depth base {} clear depth {} colour {}",
                  frame_current_, uint32_t(resolve_info.rb_copy_control.copy_src_select), uint32_t(resolve_info.color_original_base),
                  uint32_t(resolve_info.depth_original_base), resolve_info.IsClearingDepth(), resolve_info.IsClearingColor());
    }
  }
  return copied;
}

}  // namespace rex::graphics::vulkan
