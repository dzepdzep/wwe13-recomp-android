/**
 * @file        graphics/vulkan/spirv_opt_store.h
 * @brief       The precompile's optimised SPIR-V, kept on disk (wwe2k14 Session 70)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 *
 * SPIRV-Tools' performance passes (vulkan_spirv_optimize) make the translated shaders ~76% of their size and are
 * exact here (Session 70 check_item s70_spirv_opt: pictures 6 / 6, resolve hashes 467 / 467, gameplay identical;
 * Brock's entrance up to -14% GPU on the 4090), but cost ~75 ms of CPU a shader - too slow for the game itself. The
 * launcher's precompile (gpu_native_precompile + vulkan_spirv_optimize_collect) pays that once: every module it makes
 * from code not stored yet is optimised on a pool of threads (one per CPU thread) and stored in
 * <cache>/shaders/shareable/TITLE.spvopt, keyed by the XXH3 of the translator's own output. From then on every run
 * makes its modules from the stored optimised code (vulkan_spirv_optimize_store, default on) at no cost; code not
 * stored stays unoptimised. Made from the player's own game files, in their own folder: never shipped.
 * File: 'XSPO', version; records {XXH3 of the input, XXH3 of the words, word count, words}; a bad record ends it.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace rex::graphics::vulkan::spirv_opt_store {

// Loads (or re-opens) the store from `file`.
void Open(const std::filesystem::path& file);
// Waits for nothing: queued jobs are dropped (the precompile waits for Pending() == 0 before it quits).
void Close();
// The optimised words stored for this input (the XXH3 of the translator's output) into `out`; false if none.
bool Find(uint64_t key, std::vector<uint32_t>& out);
// The precompile: optimises these words on the pool and stores the result (nothing if stored or queued already).
void Collect(uint64_t key, const uint32_t* words, size_t count);
// Jobs queued or running; shaders this run optimised and stored.
uint32_t Pending();
uint32_t Collected();

}  // namespace rex::graphics::vulkan::spirv_opt_store
