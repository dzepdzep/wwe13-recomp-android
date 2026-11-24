/**
 * @file        graphics/vulkan/disk_pipeline_cache.h
 * @brief       A VkPipelineCache kept on disk (wwe2k14 Android port)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * The PC drivers keep their own shader cache, so a pipeline compiled once comes back fast in the next run. The
 * Adreno 830 driver on Jarrod's phone keeps none: every run compiled every pipeline again (phone runs 010-020: the
 * first match seconds crawl, a 4.4 s stall at the bell, the character-portrait stall). This keeps one VkPipelineCache
 * in <cache>/shaders/shareable/TITLE.vkpc for the game's per-shader pipelines (the emulated path's and NATIVE's): loaded
 * when the shader storage opens, saved in the background after new pipelines (the game hard-exits on close). The
 * driver checks the file's header itself (vendor, device, driver UUID) and ignores data from another device or driver.
 * vulkan_disk_pipeline_cache: default on on Android, off elsewhere (the PC drivers already cache).
 */

#pragma once

#include <filesystem>

#include <rex/ui/vulkan/api.h>

namespace rex::ui::vulkan {
class VulkanDevice;
}

namespace rex::graphics::vulkan::disk_pipeline_cache {

// Opens (or re-opens) the cache from `file`. No-op when the setting is off.
void Open(const ui::vulkan::VulkanDevice* device, const std::filesystem::path& file);
// The cache for vkCreate*Pipelines, or VK_NULL_HANDLE (off / not open).
VkPipelineCache Get();
// After a pipeline was created with Get(): saves in the background now and then.
void NoteCreated();
// Writes the cache now (blocking) and destroys it.
void Close();

}  // namespace rex::graphics::vulkan::disk_pipeline_cache
