#pragma once
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

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/graphics/pipeline/texture/cache.h>
#include <rex/graphics/vulkan/shader.h>
#include <rex/graphics/vulkan/shared_memory.h>
#include <rex/hash.h>
#include <rex/ui/vulkan/mem_alloc.h>
#include <rex/ui/vulkan/single_layout_descriptor_set_pool.h>

namespace rex::graphics::vulkan {

class VulkanCommandProcessor;
struct OwnedFrontAccess;  // wwe2k14 patch (Session 92, M4c)

class VulkanTextureCache final : public TextureCache {
 public:
  // wwe2k14 patch (Session 92, M4c): find-only reads of the binding state for the front end shadow. A friend declaration only.
  friend struct OwnedFrontAccess;
  // Sampler parameters that can be directly converted to a host sampler or used
  // for checking whether samplers bindings are up to date.
  union SamplerParameters {
    uint32_t value;
    struct {
      xenos::ClampMode clamp_x : 3;         // 3
      xenos::ClampMode clamp_y : 3;         // 6
      xenos::ClampMode clamp_z : 3;         // 9
      xenos::BorderColor border_color : 2;  // 11
      uint32_t mag_linear : 1;              // 12
      uint32_t min_linear : 1;              // 13
      uint32_t mip_linear : 1;              // 14
      xenos::AnisoFilter aniso_filter : 3;  // 17
      uint32_t mip_min_level : 4;           // 21
      uint32_t mip_base_map : 1;            // 22
      // Maximum mip level is in the texture resource itself, but mip_base_map
      // can be used to limit fetching to mip_min_level.
    };

    SamplerParameters() : value(0) { static_assert_size(*this, sizeof(value)); }
    struct Hasher {
      size_t operator()(const SamplerParameters& parameters) const {
        return std::hash<uint32_t>{}(parameters.value);
      }
    };
    bool operator==(const SamplerParameters& parameters) const { return value == parameters.value; }
    bool operator!=(const SamplerParameters& parameters) const { return value != parameters.value; }
  };

  // Transient descriptor set layouts must be initialized in the command
  // processor.
  static std::unique_ptr<VulkanTextureCache> Create(
      const RegisterFile& register_file, VulkanSharedMemory& shared_memory,
      uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
      VulkanCommandProcessor& command_processor,
      VkPipelineStageFlags guest_shader_pipeline_stages) {
    std::unique_ptr<VulkanTextureCache> texture_cache(new VulkanTextureCache(
        register_file, shared_memory, draw_resolution_scale_x, draw_resolution_scale_y,
        command_processor, guest_shader_pipeline_stages));
    if (!texture_cache->Initialize()) {
      return nullptr;
    }
    return std::move(texture_cache);
  }

  ~VulkanTextureCache();

  void BeginSubmission(uint64_t new_submission_index) override;
  void BeginFrame() override;
  void EndFrame();

  // Must be called within a frame - creates and untiles textures needed by
  // shaders, and enqueues transitioning them into the sampled usage. This may
  // bind compute pipelines (notifying the command processor about that), and
  // also since it may insert deferred barriers, before flushing the barriers
  // preceding host GPU work.
  void RequestTextures(uint32_t used_texture_mask) override;

  // wwe2k14 patch (Session 18, DLAA): a resolve target's cached texture as a compute storage image (barrier to
  // GENERAL pushed; usage kComputeStorageWrite, so the next sampling transitions it back). Needs a texture created
  // with the direct-load storage usage and an R32_UINT view (vulkan_texture_load_direct, scaled 32bpb).
  struct DlaaStorage {
    VkImage image = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;   // direct-load layout: the R32_UINT storage view of level 0
    VkFormat host_format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;          // host pixels
    const char* failure = nullptr;
  };
  bool DlaaResolveStorage(uint32_t base_address, uint32_t pitch_texels, uint32_t width, uint32_t height,
                          xenos::TextureFormat format, xenos::Endian endian, DlaaStorage& out);
  // The same texture as a copy destination (barrier to TRANSFER_DST_OPTIMAL pushed; usage kTransferDestination,
  // so the next sampling transitions it back); no storage view needed. out.set stays VK_NULL_HANDLE.
  // general: GENERAL layout instead (read and written by copies; usage kComputeStorageWrite: the caller must end
  // with a barrier from its last access to COMPUTE_SHADER / SHADER_WRITE in GENERAL, which the next sampling
  // transition chains from).
  bool DlaaResolveCopyTarget(uint32_t base_address, uint32_t pitch_texels, uint32_t width, uint32_t height,
                             xenos::TextureFormat format, xenos::Endian endian, DlaaStorage& out,
                             bool general = false);
  // Session 24: initializes the direct-load objects on first use (at resolution_scale 1 no scaled texture does).
  VkDescriptorSetLayout dlaa_storage_set_layout() {
    EnsureDirectLoadInitialized();
    return direct_load_descriptor_set_layout_;
  }
  // A set of that layout for one of our own images (a storage view in GENERAL layout); VK_NULL_HANDLE on failure.
  VkDescriptorSet DlaaStorageSetForView(VkImageView view);

  VkImageView GetActiveBindingOrNullImageView(uint32_t fetch_constant_index,
                                              xenos::FetchOpDimension dimension, bool is_signed);

  // wwe2k14 patch (Session 6, CP6): with vulkan_sampler_param_cache, the result comes from a cache keyed on
  // the value of every input (see GetSamplerParametersCached).
  // wwe2k14 patch (Session 6, CP5): incremented for a fetch constant every time its binding is updated
  // (UpdateTextureBindingsImpl - the only place the binding state that GetActiveBindingOrNullImageView
  // reads for it changes, also called before any texture is destroyed), so a texture descriptor written
  // for it stays what would be written now while the generation is the same. Except for 3D textures
  // sampled as 1D/2D (see GetTextureBindings3DMask), whose view is created lazily by
  // GetActiveBindingOrNullImageView itself.
  uint32_t GetTextureBindingGeneration(uint32_t fetch_constant_index) const {
    return texture_binding_generations_[fetch_constant_index];
  }
  // wwe2k14 patch (Session 6, CP5): fetch constants currently bound to a 3D texture.
  uint32_t GetTextureBindings3DMask() const { return texture_bindings_3d_mask_; }

  SamplerParameters GetSamplerParameters(const VulkanShader::SamplerBinding& binding) const;

  // Must be called for every used sampler at least once in a single submission,
  // and a submission must be open for this to be callable.
  // Returns:
  // - The sampler, if obtained successfully - and increases its last usage
  //   submission index - and has_overflown_out = false.
  // - VK_NULL_HANDLE and has_overflown_out = true if there's a total sampler
  //   count overflow in a submission that potentially hasn't completed yet.
  // - VK_NULL_HANDLE and has_overflown_out = false in case of a general failure
  //   to create a sampler.
  // wwe2k14 patch (Session 6, CP6): with vulkan_sampler_param_cache, repeated calls within one submission
  // return the sampler without the samplers_ lookup.
  VkSampler UseSampler(SamplerParameters parameters, bool& has_overflown_out);
  // Returns the submission index to await (may be the current submission in
  // case of an overflow within a single submission - in this case, it must be
  // ended, and a new one must be started) in case of sampler count overflow, so
  // samplers may be freed, and UseSamplers may take their slots.
  uint64_t GetSubmissionToAwaitOnSamplerOverflow(uint32_t overflowed_sampler_count) const;

  // Returns the 2D view of the front buffer texture (for fragment shader
  // reading - the barrier will be pushed in the command processor if needed),
  // or VK_NULL_HANDLE in case of failure. May call LoadTextureData.
  // If swap_source_needs_rb_swap_out is not nullptr, writes whether the final
  // guest-to-host swizzle requires swapping red and blue (R <- B, B <- R) with
  // green preserved, which is needed by the presentation fallback path on
  // devices without imageViewFormatSwizzle.
  VkImageView RequestSwapTexture(uint32_t& width_scaled_out, uint32_t& height_scaled_out,
                                 xenos::TextureFormat& format_out,
                                 uint32_t* width_unscaled_out = nullptr,
                                 uint32_t* height_unscaled_out = nullptr,
                                 bool* swap_source_needs_rb_swap_out = nullptr);

  bool GetScaledResolveRange(uint32_t start_unscaled, uint32_t length_unscaled,
                             uint32_t length_scaled_alignment_log2, uint64_t& start_scaled_out,
                             uint64_t& length_scaled_out) const;

  // wwe2k14 patch (Session 7c, TEXDIRECT): called at every frame boundary (VulkanCommandProcessor::
  // CollectFrameCounters, before the cpf* events and cp_swap): compares the vulkan_texture_load_direct_verify
  // readbacks of completed submissions, records cpf_texdirect and logs the [s7][TEXDIRECT] summary.
  void RecordDirectLoadFrameCounters(uint64_t frame);

  bool CommitScaledResolveRange(uint32_t start_unscaled, uint32_t length_unscaled,
                                uint32_t length_scaled_alignment_log2 = 0) {
    return EnsureScaledResolveMemoryCommitted(start_unscaled, length_unscaled,
                                              length_scaled_alignment_log2);
  }
  VkBuffer scaled_resolve_buffer() const { return scaled_resolve_buffer_; }
  void UseScaledResolveBufferForRead();
  // wwe2k14 patch (Session 22, EAGER): a resolve was recorded (after MarkRangeAsResolved). See
  // vulkan_texture_load_eager in texture_cache.cpp.
  void EagerNoteResolve();
  void UseScaledResolveBufferForWrite(uint64_t written_start_scaled,
                                      uint64_t written_length_scaled);

  // wwe2k14 patch (Session 7c, TEXLOAD): measurement of texture loads. TexLoadFrameBoundary: at every frame boundary
  // (VulkanCommandProcessor::CollectFrameCounters) - the cpf_texload timeline event and the [s7][TEXLOAD] summary
  // about every 300 frames. TexLoadAnatomyFrameBoundary: at every frame-anatomy frame boundary
  // (VulkanCommandProcessor::AnatomyGpuFrameBoundary, same arguments) - one row per load of the captured frames to
  // <anatomy>_texloads.csv.
  void TexLoadFrameBoundary(uint64_t frame);
  void TexLoadAnatomyFrameBoundary(const std::string& rows_path, int64_t frame);

  // wwe2k14 patch (Session 45, NATIVE FRAME): running counts of texture loads (any path), and of loads of textures
  // over ranges a resolve wrote (resolution-scaled, or a 1x page MarkRangeAsResolved marked) with the bytes they
  // wrote, for the native renderer's frame census (command processor thread only).
  struct NativeLoadCounters {
    uint64_t loads = 0, resolved_loads = 0, resolved_bytes = 0;
  };
  const NativeLoadCounters& native_load_counters() const { return native_load_counters_; }

  // wwe2k14 patch (Session 45, NATIVE TEXWRITE, step 2 N3a): the native renderer's resolves also fill the 1x texture
  // they feed, so it is not loaded again. The k_8_8_8_8 2D tiled single-level texture last loaded directly over
  // base_page (learned from the direct 1x loads), when it still exists, is up to date, has its R32_UINT storage view
  // and matches the resolve (pitch in 32-texel units, endianness): prepared for a compute write that keeps its other
  // texels (usage kComputeStorageWrite, layout GENERAL); returns its storage descriptor set (binding 0), its size and a
  // handle for NativeTextureResolved. VK_NULL_HANDLE when there is none.
  // (Session 59: `format` k_16_16 / k_16_16_16_16 too - their R32_UINT / R16G16B16A16_UINT storage views.)
  // (Session 79, allow_layer with gpu_native_truth_fill_cube_faces: or one face of such a cube, base_page the face's
  // start page - the set is the face's; the cube must be up to date before, the write covers one face only.)
  VkDescriptorSet NativeTextureForResolve(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness,
                                          uint32_t& width_out, uint32_t& height_out, void*& handle_out, bool allow_scaled = false,
                                          xenos::TextureFormat format = xenos::TextureFormat::k_8_8_8_8,
                                          bool allow_layer = false, int32_t* layer_out = nullptr);
  // After the resolve's MarkRangeAsResolved (which made the texture outdated): it holds what the memory holds.
  void NativeTextureResolved(void* handle);
  // wwe2k14 patch (Session 47, NATIVE TRUTH M6): the 1x 2D tiled single-level texture of `format` last loaded (directly
  // or by copy) over base_page, when it still exists, is up to date and matches the resolve (pitch in 32-texel units,
  // endianness): its image as a copy destination (barrier to TRANSFER_DST_OPTIMAL pushed; usage kTransferDestination,
  // so the next sampling transitions it back), its size and a handle for NativeTextureResolved. False when there is
  // none (`why` says why).
  // (Session 47, step 2: or the start page of one of its mip levels - not in the packed tail - with that level's pitch;
  // level_out says which, width_out / height_out are that level's.)
  // (Session 52, gpu_native_truth_fill_direct: storage_bytes 4 / 8 - Session 59: or 1 / 2 - and storage_set_out - for a single-level texture with
  // the direct load's raw-bits storage view of that size, the texture is prepared for a compute write into it (layout
  // GENERAL) and *storage_set_out is its storage set; otherwise as before, *storage_set_out VK_NULL_HANDLE.)
  bool NativeTextureForCopy(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness, xenos::TextureFormat format,
                            VkImage& image_out, uint32_t& width_out, uint32_t& height_out, uint32_t& level_out,
                            bool& has_mips_out, void*& handle_out, const char*& why, uint32_t storage_bytes = 0,
                            VkDescriptorSet* storage_set_out = nullptr);
  // wwe2k14 patch (Session 82, P2, G4): where a copy into the texture whose base page (or mip level start page) is
  // base_page lands, from the key a load of it taught (NativeLearnTextureKey) - nothing of the texture is touched.
  bool NativeResolveTarget(uint32_t base_page, uint32_t pitch_div_32, uint32_t endianness, xenos::TextureFormat format,
                           uint32_t& texture_page_out, uint32_t& width_out, uint32_t& height_out, uint32_t& levels_out,
                           uint32_t& level_out, bool& scaled_out, uint32_t& extent_bytes_out);
  // (Session 82, P2, G7: the same for a copy into a cube's face - the face's start page: the cube's base page, face size,
  // face count, the face (array layer), scaled flag, the cube's guest bytes.)
  bool NativeResolveFace(uint32_t face_page, uint32_t endianness, xenos::TextureFormat format,
                         uint32_t& texture_page_out, uint32_t& width_out, uint32_t& height_out, uint32_t& layers_out,
                         uint32_t& layer_out, bool& scaled_out, uint32_t& extent_bytes_out);
  VkDescriptorSetLayout native_texture_set_layout() const { return direct_load_descriptor_set_layout_; }
  // wwe2k14 patch (Session 52, gpu_native_truth_cover_outdated): the rectangle (host texels) the next
  // NativeTextureForResolve / NativeTextureForCopy call's write covers; a single-level texture the write covers whole
  // need not be up to date first (every texel is written). Consumed by that call.
  void NativeSetCover(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    native_cover_[0] = x;
    native_cover_[1] = y;
    native_cover_[2] = w;
    native_cover_[3] = h;
  }
  uint32_t native_cover_[4] = {1, 1, 0, 0};
  // wwe2k14 patch (Session 52, gpu_native_truth_batch_resolves): while set, NativeTextureForResolve keeps the
  // texture's barrier (to its compute-write layout) here instead of pushing it - a deferred resolve records it with
  // its batch.
  struct NativeHeldBarrier {
    bool valid = false;
    VkImage image = VK_NULL_HANDLE;
    VkPipelineStageFlags src_stage = 0;
    VkAccessFlags src_access = 0;
    VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  };
  bool native_hold_barrier_ = false;
  // wwe2k14 patch (Session 80, gpu_native_truth_serve_copies): bindings whose view is now another image's.
  uint32_t native_served_mask_ = 0;
  // wwe2k14 patch (Session 80, gpu_native_own_steps): bindings given an owned step's image (RequestTextures).
  uint32_t native_owned_mask_ = 0;
  bool native_last_request_served_ = false;
  NativeHeldBarrier native_held_barrier_;
  // (Taken at the start of the call - an early return must not leave it for the next one.)
  std::array<uint32_t, 4> NativeTakeCover() {
    const std::array<uint32_t, 4> cover = {native_cover_[0], native_cover_[1], native_cover_[2], native_cover_[3]};
    native_cover_[0] = native_cover_[1] = 1;
    native_cover_[2] = native_cover_[3] = 0;
    return cover;
  }
  bool NativeCoversWhole(const TextureKey& key, const std::array<uint32_t, 4>& cover) const {
    const uint32_t w = key.GetWidth() * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
    const uint32_t h = key.GetHeight() * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
    return !key.mip_max_level && cover[0] == 0 && cover[1] == 0 && cover[2] >= w && cover[3] >= h;
  }
  // wwe2k14 patch (Session 49, gpu_native_truth_texture_fill_lazy): whether the texture the last NativeTextureForCopy
  // returned had been sampled since it was last written (filled, loaded).
  bool native_last_copy_sampled() const { return native_last_copy_sampled_; }
  // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_copies): whether the texture the last NativeTextureForResolve
  // / NativeTextureForCopy returned was used for nothing but compute writes since its last one (not sampled, not swapped,
  // not loaded): an earlier native copy into it that this write covers was read by nothing.
  bool native_last_copy_untouched() const { return native_last_copy_untouched_; }
  // wwe2k14 patch (Session 49, NATIVE TRUTH skip write-back): a texture (a handle from the two calls above) that a
  // compute shader can read back texel by texel - a single-level 2D tiled 1x texture with the direct load's raw-bits
  // storage view (R32_UINT: 32 bits a texel, R16G16B16A16_UINT: 64): its bytes a texel, 0 if none.
  // (Session 56: allow_scaled - a scaled resolve texture too, gpu_native_truth_skip_writeback_scaled.)
  // (Session 79, layer >= 0: a face of a cube NativeTextureForResolve fills - gpu_native_truth_fill_cube_faces.)
  // (Session 80, allow_depth: a k_24_8_FLOAT texture's R32_UINT view too - the 32-bit floats the direct load made of
  // its 20e4 depth, for a reader that converts them back: gpu_native_truth_skip_writeback_depth.)
  uint32_t NativeTextureReadable(void* handle, bool allow_scaled = false, int32_t layer = -1,
                                 bool allow_depth = false) const;
  // Its storage descriptor set, the texture prepared for a compute read (layout GENERAL, usage kComputeStorageWrite
  // so its next sampling transitions it back); VK_NULL_HANDLE if it has none. And pinning (no eviction).
  // (Session 79, layer >= 0: the cube face's.)
  VkDescriptorSet NativeTextureForRead(void* handle, uint32_t& old_usage_out, bool allow_scaled = false,
                                       int32_t layer = -1, bool allow_depth = false);
  // After the read's dispatch (review: a draw may sample it next, with the layout its bindings were made for): back to
  // the usage and layout it had (nothing if it was a compute storage texture or undefined).
  void NativeTextureEndRead(void* handle, uint32_t old_usage);
  // wwe2k14 patch (Session 56, vulkan_texture_sample_written_early): the textures written by compute or a copy since
  // the last call are moved to the guest shader sampled usage now (their barriers pushed to the command processor),
  // so a draw sampling one later in a render pass needs no barrier there (which would end the pass) - except those in
  // `keep` (handles: a pending write still needs them), which stay listed. Returns how many were moved.
  uint32_t SampleWrittenTextures(void* const* keep, size_t keep_count);
  // wwe2k14 patch (Session 56): see TextureCache::BeforeScaledResolveRead.
  void BeforeScaledResolveRead(uint32_t start, uint32_t length) override;
  void NativeTexturePin(void* handle, bool pin);
  // wwe2k14 patch (Session 80, gpu_native_truth_serve_copies): the used bindings served from an image, for a draw that
  // is not a truth draw after all - their textures' own views again (generations bumped), the textures taken to the
  // sampled usage as RequestTextures takes them (a copy still pending into one flushed first, its barrier pushed).
  // Returns whether any binding was served.
  bool NativeUnserve(uint32_t used_texture_mask);
  bool NativeServed(uint32_t used_texture_mask) const { return (native_served_mask_ & used_texture_mask) != 0; }
  // wwe2k14 patch (Session 80, gpu_native_own_steps): the owned pages changed - the unchanged-bindings shortcut of
  // RequestTextures runs its loops again (a binding of a page that became ours keeps no stale view; review).
  void NativeOwnedChanged() { ++texture_usage_changes_; }
  // (Whether the last texture request served a used binding: a repeat of that draw, which requests nothing, takes the
  // full path - its served image is made readable by its own request.)
  bool native_last_request_served() const { return native_last_request_served_; }

 protected:
  bool native_last_copy_sampled_ = true;  // wwe2k14 patch (Session 49)
  bool native_last_copy_untouched_ = false;  // wwe2k14 patch (Session 80)
  bool IsSignedVersionSeparateForFormat(TextureKey key) const override;
  bool IsScaledResolveSupportedForFormat(TextureKey key) const override;
  uint32_t GetHostFormatSwizzle(TextureKey key) const override;

  uint32_t GetMaxHostTextureWidthHeight(xenos::DataDimension dimension) const override;
  uint32_t GetMaxHostTextureDepthOrArraySize(xenos::DataDimension dimension) const override;

  std::unique_ptr<Texture> CreateTexture(TextureKey key) override;

  bool EnsureScaledResolveMemoryCommitted(uint32_t start_unscaled, uint32_t length_unscaled,
                                          uint32_t length_scaled_alignment_log2 = 0) override;

  bool LoadTextureDataFromResidentMemoryImpl(Texture& texture, bool load_base,
                                             bool load_mips) override;

  void UpdateTextureBindingsImpl(uint32_t fetch_constant_mask) override;

 private:
  enum LoadDescriptorSetIndex {
    kLoadDescriptorSetIndexDestination,
    kLoadDescriptorSetIndexSource,
    kLoadDescriptorSetCount,
  };

  struct HostFormat {
    LoadShaderIndex load_shader;
    // Do NOT add integer formats to this - they are not filterable, can only be
    // read with ImageFetch, not ImageSample! If any game is seen using
    // num_format 1 for fixed-point formats (for floating-point, it's normally
    // set to 1 though), add a constant buffer containing multipliers for the
    // textures and multiplication to the tfetch implementation.
    VkFormat format;
    // Whether the format is block-compressed on the host (the host block size
    // matches the guest format block size in this case), and isn't decompressed
    // on load.
    bool block_compressed;

    // Set up dynamically based on what's supported by the device.
    bool linear_filterable;
  };

  struct HostFormatPair {
    HostFormat format_unsigned;
    HostFormat format_signed;
    // Mapping of Xenos swizzle components to Vulkan format components.
    uint32_t swizzle;
    // Whether the unsigned and the signed formats are compatible for one image
    // and the same image data (on a portability subset device, this should also
    // take imageViewFormatReinterpretation into account).
    bool unsigned_signed_compatible;
  };

  class VulkanTexture final : public Texture {
   public:
    enum class Usage {
      kUndefined,
      kTransferDestination,
      kGuestShaderSampled,
      kSwapSampled,
      // wwe2k14 patch (Session 7c, TEXDIRECT): written by a direct load compute dispatch (GENERAL layout).
      kComputeStorageWrite,
    };

    // wwe2k14 patch (Session 7c, TEXDIRECT): how the image was created for vulkan_texture_load_direct.
    enum class DirectLoadCreation : uint8_t {
      // The setting was off or the texture is not resolution-scaled - no storage usage.
      kNotRequested,
      // Requested, but the key or the formats are not eligible (levels, layers, packed base, load shader, host
      // format not in the bit-exact list or without storage image support) - no storage usage.
      kUnsupported,
      // Requested, but direct loads are unavailable (their initialization failed) - no storage usage.
      kUnavailable,
      // Eligible, but the device rejected the image with the extra usage and flags
      // (vkGetPhysicalDeviceImageFormatProperties failed, or the image exceeds the limits it returned), or creating
      // it with them failed and it was created without them (review fix) - no storage usage.
      kDeviceRejected,
      // Created with the storage usage and the direct_load_format() view format.
      kCapable,
    };

    // Takes ownership of the image and its memory.
    explicit VulkanTexture(VulkanTextureCache& texture_cache, const TextureKey& key, VkImage image,
                           VmaAllocation allocation, bool track_usage = true);
    ~VulkanTexture();

    VkImage image() const { return image_; }

    // Doesn't transition (the caller must insert the barrier).
    Usage SetUsage(Usage new_usage) {
      Usage old_usage = usage_;
      usage_ = new_usage;
      // wwe2k14 patch (Session 7, TEXREQ): see request_textures_loop_state_.
      if (old_usage != new_usage) {
        VulkanTextureCache& cache = static_cast<VulkanTextureCache&>(texture_cache());
        ++cache.texture_usage_changes_;
        // wwe2k14 patch (Session 56): written by compute or a copy - see SampleWrittenTextures.
        if (cache.track_written_ &&
            (new_usage == Usage::kComputeStorageWrite || new_usage == Usage::kTransferDestination) &&
            !in_written_list) {
          in_written_list = true;
          cache.written_textures_.push_back(this);
        }
      }
      return old_usage;
    }
    Usage usage() const { return usage_; }  // wwe2k14 patch (Session 56)
    // wwe2k14 patch (Session 80, gpu_native_truth_serve_copies): GetView's component swizzle, for a view of another
    // image in this texture's place.
    static VkComponentSwizzle NativeComponentSwizzle(uint32_t host_swizzle, uint32_t component) {
      return GetComponentSwizzle(host_swizzle, component);
    }
    bool in_written_list = false;  // wwe2k14 patch (Session 56): in written_textures_
    // wwe2k14 patch (Session 80, gpu_native_truth_drop_dead_copies): really used since the last native write - sampled
    // by a draw, the swap or an eager load, or loaded from memory (not SampleWrittenTextures' layout change).
    bool native_used = true;

    VkImageView GetView(bool is_signed, uint32_t host_swizzle, bool is_array = true);
    // wwe2k14 patch (Session 92, M4c): GetView's lookup only - VK_NULL_HANDLE if the view does not exist yet (never creates).
    VkImageView FindView(bool is_signed, uint32_t host_swizzle, bool is_array = true) const;
    VkImageView GetOrCreate3DAs2DImageView(bool is_signed, uint32_t host_swizzle);

    // wwe2k14 patch (Session 7c, TEXDIRECT).
    DirectLoadCreation direct_load_creation() const { return direct_load_creation_; }
    // The unsigned integer format of the storage view (R32_UINT or R16G16B16A16_UINT), or VK_FORMAT_UNDEFINED.
    VkFormat direct_load_format() const { return direct_load_format_; }
    void SetDirectLoad(DirectLoadCreation creation, VkFormat format) {
      direct_load_creation_ = creation;
      direct_load_format_ = format;
    }
    // Storage image descriptor set (direct load layout, set 0) for level 0, layer 0 through the
    // direct_load_format() view. Created on first use and never rewritten while the texture exists, so it can
    // be bound in any number of submissions. VK_NULL_HANDLE on failure - which is remembered (review fix): after a
    // failure, it returns VK_NULL_HANDLE without trying (and logging) again for this texture.
    VkDescriptorSet GetDirectLoadDescriptorSet();
    // wwe2k14 patch (Session 79, gpu_native_truth_fill_cube_faces): the same for array layer `layer` of level 0 (layer
    // 0: GetDirectLoadDescriptorSet) - a 2D view of the layer, created on first use, kept while the texture exists.
    static constexpr uint32_t kNativeLayerViews = 6;
    VkDescriptorSet GetLayerStorageDescriptorSet(uint32_t layer);
    // wwe2k14 patch (Session 7c, TEXLOAD): whether the last load into the image took its scratch buffer from the
    // vulkan_texture_load_scratch_ring (LoadTextureDataFromResidentMemoryImpl: a later load into the image while it
    // is still a transfer destination needs the image write-after-write barrier if either load used the ring).
    bool last_load_used_scratch_ring() const { return last_load_used_scratch_ring_; }
    void SetLastLoadUsedScratchRing(bool used) { last_load_used_scratch_ring_ = used; }
    // wwe2k14 patch (Session 22, EAGER): the frame of the texture's last early load (UINT64_MAX: none).
    uint64_t eager_loaded_frame() const { return eager_loaded_frame_; }
    void SetEagerLoadedFrame(uint64_t frame) { eager_loaded_frame_ = frame; }
    // wwe2k14 patch (Session 44, S0b ceiling): whether a direct 1x load has filled the image once
    // (vulkan_texture_load_skip_debug skips the later ones).
    bool direct_loaded_once() const { return direct_loaded_once_; }
    void SetDirectLoadedOnce() { direct_loaded_once_ = true; }

   private:
    union ViewKey {
      uint32_t key;
      struct {
        uint32_t is_signed_separate_view : 1;
        uint32_t host_swizzle : 12;
        uint32_t is_array : 1;
      };

      ViewKey() : key(0) { static_assert_size(*this, sizeof(key)); }

      struct Hasher {
        size_t operator()(const ViewKey& key) const {
          return std::hash<decltype(key.key)>{}(key.key);
        }
      };
      bool operator==(const ViewKey& other_key) const { return key == other_key.key; }
      bool operator!=(const ViewKey& other_key) const { return !(*this == other_key); }
    };

    static constexpr VkComponentSwizzle GetComponentSwizzle(uint32_t texture_swizzle,
                                                            uint32_t component_index) {
      xenos::XE_GPU_TEXTURE_SWIZZLE texture_component_swizzle =
          xenos::XE_GPU_TEXTURE_SWIZZLE((texture_swizzle >> (3 * component_index)) & 0b111);
      if (texture_component_swizzle == xenos::XE_GPU_TEXTURE_SWIZZLE(component_index)) {
        // The portability subset requires all swizzles to be IDENTITY, return
        // IDENTITY specifically, not R, G, B, A.
        return VK_COMPONENT_SWIZZLE_IDENTITY;
      }
      switch (texture_component_swizzle) {
        case xenos::XE_GPU_TEXTURE_SWIZZLE_R:
          return VK_COMPONENT_SWIZZLE_R;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_G:
          return VK_COMPONENT_SWIZZLE_G;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_B:
          return VK_COMPONENT_SWIZZLE_B;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_A:
          return VK_COMPONENT_SWIZZLE_A;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_0:
          return VK_COMPONENT_SWIZZLE_ZERO;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_1:
          return VK_COMPONENT_SWIZZLE_ONE;
        default:
          // An invalid value.
          return VK_COMPONENT_SWIZZLE_IDENTITY;
      }
    }

    VkImage image_;
    VmaAllocation allocation_;

    Usage usage_ = Usage::kUndefined;
    bool last_load_used_scratch_ring_ = false;  // wwe2k14 patch (Session 7c, TEXLOAD)
    uint64_t eager_loaded_frame_ = UINT64_MAX;  // wwe2k14 patch (Session 22, EAGER)

    std::unordered_map<ViewKey, VkImageView, ViewKey::Hasher> views_;
    std::unique_ptr<VulkanTexture> texture_3d_as_2d_;
    VkImageView image_view_3d_as_2d_unsigned_ = VK_NULL_HANDLE;
    VkImageView image_view_3d_as_2d_signed_ = VK_NULL_HANDLE;

    // wwe2k14 patch (Session 7c, TEXDIRECT).
    DirectLoadCreation direct_load_creation_ = DirectLoadCreation::kNotRequested;
    bool direct_loaded_once_ = false;  // wwe2k14 patch (Session 44)
    VkFormat direct_load_format_ = VK_FORMAT_UNDEFINED;
    VkImageView direct_load_view_ = VK_NULL_HANDLE;
    // Index in VulkanTextureCache::direct_load_descriptor_pool_, SIZE_MAX if not allocated.
    size_t direct_load_descriptor_index_ = SIZE_MAX;
    // Creating the storage view or allocating the descriptor set failed once - not tried again.
    bool direct_load_descriptor_failed_ = false;
    // wwe2k14 patch (Session 79): layers 1-5 (index layer - 1), as direct_load_view_ / _descriptor_index_.
    VkImageView layer_storage_views_[kNativeLayerViews - 1] = {};
    size_t layer_storage_indices_[kNativeLayerViews - 1] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
  };

  struct VulkanTextureBinding {
    VkImageView image_view_unsigned;
    VkImageView image_view_signed;

    VulkanTextureBinding() { Reset(); }

    void Reset() {
      image_view_unsigned = VK_NULL_HANDLE;
      image_view_signed = VK_NULL_HANDLE;
    }
  };

  struct Sampler {
    VkSampler sampler;
    bool uses_custom_border_color;
    uint64_t last_usage_submission;
    std::pair<const SamplerParameters, Sampler>* used_previous;
    std::pair<const SamplerParameters, Sampler>* used_next;
  };

  static constexpr bool AreDimensionsCompatible(xenos::FetchOpDimension binding_dimension,
                                                xenos::DataDimension resource_dimension) {
    switch (binding_dimension) {
      case xenos::FetchOpDimension::k1D:
      case xenos::FetchOpDimension::k2D:
        return resource_dimension == xenos::DataDimension::k1D ||
               resource_dimension == xenos::DataDimension::k2DOrStacked ||
               resource_dimension == xenos::DataDimension::k3D;
      case xenos::FetchOpDimension::k3DOrStacked:
        return resource_dimension == xenos::DataDimension::k3D;
      case xenos::FetchOpDimension::kCube:
        return resource_dimension == xenos::DataDimension::kCube;
      default:
        return false;
    }
  }

  explicit VulkanTextureCache(const RegisterFile& register_file, VulkanSharedMemory& shared_memory,
                              uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
                              VulkanCommandProcessor& command_processor,
                              VkPipelineStageFlags guest_shader_pipeline_stages);

  bool Initialize();
  bool InitializeScaledResolveBuffer();
  void ShutdownScaledResolveBuffer();

  const HostFormatPair& GetHostFormatPair(TextureKey key) const;

  void GetTextureUsageMasks(VulkanTexture::Usage usage, VkPipelineStageFlags& stage_mask,
                            VkAccessFlags& access_mask, VkImageLayout& layout);
  bool EnsureScaledResolveBufferAllocated(uint64_t start_scaled, uint64_t length_scaled);
  void GetScaledResolveUsageMasks(VkPipelineStageFlags& stage_mask_out,
                                  VkAccessFlags& access_mask_out, bool write) const;

  xenos::ClampMode NormalizeClampMode(xenos::ClampMode clamp_mode) const;

  // wwe2k14 patch (Session 6, CP6): sampler parameter and sampler lookup caches (vulkan_sampler_param_cache,
  // vulkan_sampler_param_cache_verify). GetSamplerParametersForFetch and UseSamplerUncached are the original
  // GetSamplerParameters and UseSampler bodies.
  SamplerParameters GetSamplerParametersForFetch(const xenos::xe_gpu_texture_fetch_t& fetch,
                                                 const VulkanShader::SamplerBinding& binding) const;
  SamplerParameters GetSamplerParametersCached(const xenos::xe_gpu_texture_fetch_t& fetch,
                                               const VulkanShader::SamplerBinding& binding,
                                               bool verify) const;
  static uint32_t GetSamplerParameterCacheExtraKey(const VulkanShader::SamplerBinding& binding);
  VkSampler UseSamplerUncached(SamplerParameters parameters, bool& has_overflown_out);
  void ReportSamplerCacheStats();

  // wwe2k14 patch (Session 7c, TEXDIRECT): vulkan_texture_load_direct. Loads of resolution-scaled textures with
  // one mip level and one layer that use the 32bpb or 64bpb load shader (and need no float conversion pass)
  // write the texels with a compute shader directly into the image, through a storage view with a size-matched
  // unsigned integer format, instead of compute -> scratch buffer -> vkCmdCopyBufferToImage. See
  // shaders/texture_load_scaled_direct.cs.glsl for why the texel bits are the same.
  enum DirectLoadShader : uint32_t {
    kDirectLoadShader32bpb,
    kDirectLoadShader64bpb,
    kDirectLoadShaderDepthFloat,  // wwe2k14 patch (Session 22, TEXDEPTH)
    kDirectLoadShaderCount,
  };
  // Why an eligible-looking load (scaled resolve source, setting on) took the buffer + copy path.
  enum DirectLoadFallback : uint32_t {
    // More than one mip level, or the base level not being loaded.
    kDirectLoadFallbackLevels,
    // Not a single-layer 1D / 2D texture (arrays, 3D, cube).
    kDirectLoadFallbackLayers,
    // The base level is stored in a packed mip tail.
    kDirectLoadFallbackPacked,
    // Load shader other than 32bpb / 64bpb, block-compressed host format, float conversion pass, host format
    // not in the bit-exact list, or no storage image support for the image's formats.
    kDirectLoadFallbackFormat,
    // Review fix (was counted as format): eligible, but the device rejected the image with the storage usage
    // (VulkanTexture::DirectLoadCreation::kDeviceRejected - image format properties, their limits, or the image
    // creation with them failed).
    kDirectLoadFallbackDevice,
    // Review fix (was counted as format): direct loads are unavailable - their initialization failed
    // (VulkanTexture::DirectLoadCreation::kUnavailable).
    kDirectLoadFallbackUnavailable,
    // The image was created while vulkan_texture_load_direct was off, so it has no storage usage.
    kDirectLoadFallbackNoStorage,
    // Creating the storage view or its descriptor failed (now or in an earlier load of the texture).
    kDirectLoadFallbackDescriptor,
    kDirectLoadFallbackCount,
  };
  struct DirectLoadStats {
    uint32_t direct[kDirectLoadShaderCount];
    // Bytes of texels written by direct loads.
    uint64_t direct_bytes;
    uint32_t fallbacks[kDirectLoadFallbackCount];
    // Scaled textures created while the setting was on, by VulkanTexture::DirectLoadCreation: with the storage
    // usage (kCapable) / without: not eligible (kUnsupported), rejected by the device (kDeviceRejected), direct
    // loads unavailable (kUnavailable).
    uint32_t created_capable;
    uint32_t created_unsupported;
    uint32_t created_rejected;
    uint32_t created_unavailable;
    // vulkan_texture_load_direct_verify.
    uint32_t verify_recorded;
    uint32_t verify_checked;
    uint32_t verify_mismatches;
    // Not compared: no free readback slot, readback buffer creation failed, or the submission's commands were
    // dropped (async submit).
    uint32_t verify_dropped;
    // Review fix: command processor thread time of the comparisons (CheckDirectLoadVerifies), microseconds - the
    // cost verify mode adds to the frame boundary.
    uint64_t verify_compare_us;

    uint32_t direct_total() const {
      uint32_t total = 0;
      for (uint32_t count : direct) {
        total += count;
      }
      return total;
    }
    uint32_t fallbacks_total() const {
      uint32_t total = 0;
      for (uint32_t count : fallbacks) {
        total += count;
      }
      return total;
    }
    void Add(const DirectLoadStats& other) {
      for (uint32_t i = 0; i < kDirectLoadShaderCount; ++i) {
        direct[i] += other.direct[i];
      }
      direct_bytes += other.direct_bytes;
      for (uint32_t i = 0; i < kDirectLoadFallbackCount; ++i) {
        fallbacks[i] += other.fallbacks[i];
      }
      created_capable += other.created_capable;
      created_unsupported += other.created_unsupported;
      created_rejected += other.created_rejected;
      created_unavailable += other.created_unavailable;
      verify_recorded += other.verify_recorded;
      verify_checked += other.verify_checked;
      verify_mismatches += other.verify_mismatches;
      verify_dropped += other.verify_dropped;
      verify_compare_us += other.verify_compare_us;
    }
  };
  // A readback of one verified direct load: [0, half_size) - the buffer + copy path's load shader output
  // (row pitch row_pitch), [half_size, 2 * half_size) - the directly written image, copied with the same row
  // pitch. Rows are compared over width_scaled texels.
  struct DirectLoadVerifySlot {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    const uint8_t* mapping = nullptr;
    bool coherent = false;
    bool pending = false;
    uint64_t submission = 0;
    uint64_t lost_upload_count = 0;
    uint64_t frame = 0;
    VkDeviceSize half_size = 0;
    uint32_t row_pitch = 0;
    uint32_t bytes_per_texel = 0;
    uint32_t width_scaled = 0;
    uint32_t height_scaled = 0;
    // For the mismatch log (addresses, sizes and formats only).
    uint32_t base_address = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    xenos::TextureFormat guest_format = xenos::TextureFormat(0);
    VkFormat host_format = VK_FORMAT_UNDEFINED;
    uint32_t load_flags = 0;
    bool control = false;
  };
  static constexpr uint32_t kDirectLoadVerifySlotsMax = 8;

  // The view format of the storage view for a load shader, VK_FORMAT_UNDEFINED if not covered.
  static VkFormat GetDirectLoadViewFormat(LoadShaderIndex load_shader);
  // wwe2k14 patch (Session 22, TEXDEPTH): the direct load shader for a load shader GetDirectLoadViewFormat accepts.
  static DirectLoadShader GetDirectLoadShader(LoadShaderIndex load_shader) {
    switch (load_shader) {
      case kLoadShaderIndex64bpb:
        return kDirectLoadShader64bpb;
      case kLoadShaderIndexDepthFloat:
        return kDirectLoadShaderDepthFloat;
      default:
        return kDirectLoadShader32bpb;
    }
  }
  // Whether a host format can be loaded through the GetDirectLoadViewFormat view of its load shader bit-exactly
  // (same texel size and plain memory layout), and supports storage images with optimal tiling here.
  bool IsDirectLoadHostFormatSupported(VkFormat format, LoadShaderIndex load_shader) const;
  // For CreateTexture: the storage view format if a texture with the key and the image formats can be loaded
  // directly, VK_FORMAT_UNDEFINED otherwise. Direct loads must be initialized (EnsureDirectLoadInitialized).
  VkFormat GetDirectLoadFormatForNewTexture(TextureKey key, const VkFormat* formats,
                                            uint32_t format_count) const;
  // Review fix: initialized lazily, on the first creation of a scaled texture with vulkan_texture_load_direct on
  // (one attempt), so with the setting off nothing is created or logged. Whether direct loads are available.
  bool EnsureDirectLoadInitialized();
  bool InitializeDirectLoad();
  void ShutdownDirectLoad();
  // Review fix: destroys the readback buffers of the verify slots not waiting for a submission (all of them once
  // their submissions have completed), when vulkan_texture_load_direct_verify is off.
  void ReleaseDirectLoadVerifySlots();
  // Records the direct load of the base level. Fails (before recording anything) only where the buffer + copy
  // path would fail too (source range, transient descriptor).
  bool LoadTextureDataDirect(VulkanTexture& texture, DirectLoadShader shader,
                             LoadShaderIndex load_shader, VkDescriptorSet descriptor_set_dest,
                             bool is_3d_tiling);
  // Records the reference dispatch (the buffer + copy path's shader into a readback slot) and the copy of
  // the directly written image for vulkan_texture_load_direct_verify. The direct load's source descriptor
  // set and constants are reused.
  void RecordDirectLoadVerify(VulkanTexture& texture, LoadShaderIndex load_shader,
                              VkDescriptorSet descriptor_set_source, const LoadConstants& load_constants,
                              uint32_t group_count_x, uint32_t group_count_y, uint32_t bytes_per_texel);
  void CheckDirectLoadVerifies();

  bool direct_load_initialize_attempted_ = false;
  VkDescriptorSetLayout direct_load_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout direct_load_pipeline_layout_ = VK_NULL_HANDLE;
  std::array<VkPipeline, kDirectLoadShaderCount> direct_load_pipelines_{};
  std::unique_ptr<ui::vulkan::SingleLayoutDescriptorSetPool> direct_load_descriptor_pool_;
  PFN_vkGetPhysicalDeviceImageFormatProperties direct_load_get_image_format_properties_ = nullptr;
  // Formats with VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT in optimalTilingFeatures, among the ones considered.
  std::vector<VkFormat> direct_load_storage_formats_;
  DirectLoadStats direct_load_frame_stats_ = {};
  uint64_t direct_load_skipped_debug_ = 0;  // wwe2k14 patch (Session 44): vulkan_texture_load_skip_debug, run total
  DirectLoadStats direct_load_window_stats_ = {};
  DirectLoadStats direct_load_run_stats_ = {};
  uint64_t direct_load_window_first_frame_ = UINT64_MAX;
  // Verify selection: per frame, direct loads before the first verified one (rotates so that different loads
  // of the frame get verified), and verifies recorded this frame.
  uint32_t direct_load_verify_skip_ = 0;
  uint32_t direct_load_frame_verifies_ = 0;
  std::array<DirectLoadVerifySlot, kDirectLoadVerifySlotsMax> direct_load_verify_slots_;
  uint32_t direct_load_verify_mismatches_logged_ = 0;
  // Images whose creation with the storage usage failed and that were created without it (logged up to 4).
  uint32_t direct_load_create_failures_logged_ = 0;

  struct SamplerParameterCacheEntry {
    // The 6 fetch constant dwords.
    uint64_t key_fetch[3];
    // GetSamplerParameterCacheExtraKey (always has the top bit set, so a zeroed entry never matches).
    uint32_t key_extra;
    SamplerParameters parameters;
  };
  static_assert(sizeof(SamplerParameterCacheEntry) == 32);
  // 2-way set-associative, one cache line per set.
  struct alignas(64) SamplerParameterCacheSet {
    SamplerParameterCacheEntry entries[2];
  };
  static constexpr uint32_t kSamplerParameterCacheSetCountLog2 = 12;

  struct SamplerUseCacheEntry {
    // Submission in which UseSampler returned `sampler` for `parameters` (0 - never, submission indices start
    // at 1).
    uint64_t submission;
    VkSampler sampler;
    SamplerParameters parameters;
  };
  static constexpr uint32_t kSamplerUseCacheSizeLog2 = 8;

  struct SamplerCacheStats {
    // Since the last summary.
    uint64_t parameter_lookups;
    uint64_t parameter_hits;
    uint64_t parameter_verified;
    uint64_t parameter_mismatches;
    uint64_t use_calls;
    uint64_t use_hits;
    uint64_t use_verified;
    uint64_t use_mismatches;
    uint32_t frames;
    // Since startup.
    uint64_t parameter_mismatches_total;
    uint64_t use_mismatches_total;
    uint32_t parameter_mismatches_logged;
    uint32_t use_mismatches_logged;
  };

  VulkanCommandProcessor& command_processor_;
  VkPipelineStageFlags guest_shader_pipeline_stages_;

  // Using the Vulkan Memory Allocator because texture count in games is
  // naturally pretty much unbounded, while Vulkan implementations, especially
  // on Windows versions before 10, may have an allocation count limit as low as
  // 4096.
  VmaAllocator vma_allocator_ = VK_NULL_HANDLE;

  static const HostFormatPair kBestHostFormats[64];
  static const HostFormatPair kHostFormatGBGRUnaligned;
  static const HostFormatPair kHostFormatBGRGUnaligned;
  static const HostFormatPair kHostFormatDXT1Unaligned;
  static const HostFormatPair kHostFormatDXT2_3Unaligned;
  static const HostFormatPair kHostFormatDXT4_5Unaligned;
  static const HostFormatPair kHostFormatDXNUnaligned;
  static const HostFormatPair kHostFormatDXT5AUnaligned;
  HostFormatPair host_formats_[64];

  VkPipelineLayout load_pipeline_layout_ = VK_NULL_HANDLE;
  std::array<VkPipeline, kLoadShaderCount> load_pipelines_{};
  std::array<VkPipeline, kLoadShaderCount> load_pipelines_scaled_{};

  // If both images can be placed in the same allocation, it's one allocation,
  // otherwise it's two separate.
  std::array<VkDeviceMemory, 2> null_images_memory_{};
  VkImage null_image_2d_array_cube_ = VK_NULL_HANDLE;
  VkImage null_image_3d_ = VK_NULL_HANDLE;
  VkImageView null_image_view_2d_array_ = VK_NULL_HANDLE;
  VkImageView null_image_view_cube_ = VK_NULL_HANDLE;
  VkImageView null_image_view_3d_ = VK_NULL_HANDLE;
  bool null_images_cleared_ = false;

  std::array<VulkanTextureBinding, xenos::kTextureFetchConstantCount> vulkan_texture_bindings_;
  // wwe2k14 patch (Session 6, CP5): see GetTextureBindingGeneration and GetTextureBindings3DMask.
  std::array<uint32_t, xenos::kTextureFetchConstantCount> texture_binding_generations_{};
  uint32_t texture_bindings_3d_mask_ = 0;

  // wwe2k14 patch (Session 7, TEXREQ): for gpu_request_textures_skip_unchanged. The number of times any
  // texture's usage changed (VulkanTexture::SetUsage - the only writer of a texture's usage), and the state
  // in which the per-draw loops of RequestTextures last ran in full (recorded whatever the settings).
  uint64_t texture_usage_changes_ = 0;
  std::vector<VulkanTexture*> written_textures_;  // wwe2k14 patch (Session 56): see SampleWrittenTextures
  bool track_written_ = false;  // vulkan_texture_sample_written_early, latched in BeginFrame (review: off = no list)
  struct RequestTexturesLoopState {
    bool valid = false;
    uint32_t used_texture_mask = 0;
    uint64_t bindings_version = 0;
    uint64_t submission = 0;
    uint64_t usage_changes = 0;
  };
  RequestTexturesLoopState request_textures_loop_state_;

  // Unsupported texture formats used during this frame (for research and
  // testing).
  enum : uint8_t {
    kUnsupportedResourceBit = 1,
    kUnsupportedUnormBit = kUnsupportedResourceBit << 1,
    kUnsupportedSnormBit = kUnsupportedUnormBit << 1,
  };
  uint8_t unsupported_format_features_used_[64] = {};

  uint32_t sampler_max_count_;

  xenos::AnisoFilter max_anisotropy_;

  std::unordered_map<SamplerParameters, Sampler, SamplerParameters::Hasher> samplers_;
  std::pair<const SamplerParameters, Sampler>* sampler_used_first_ = nullptr;
  std::pair<const SamplerParameters, Sampler>* sampler_used_last_ = nullptr;
  uint32_t custom_border_color_sampler_count_ = 0;

  VkBuffer scaled_resolve_buffer_ = VK_NULL_HANDLE;
  uint64_t scaled_resolve_buffer_size_ = 0;
  bool scaled_resolve_buffer_sparse_ = false;
  uint32_t scaled_resolve_buffer_memory_type_ = UINT32_MAX;
  std::vector<VkDeviceMemory> scaled_resolve_buffer_memory_;
  uint32_t scaled_resolve_sparse_granularity_log2_ = UINT32_MAX;
  std::vector<uint64_t> scaled_resolve_sparse_allocated_;
  bool scaled_resolve_last_usage_write_ = false;
  std::pair<uint64_t, uint64_t> scaled_resolve_last_written_range_{0, 0};
  // wwe2k14 patch (Session 22, RESWAW): the non-empty ranges resolves wrote since the last barrier that ordered the
  // scaled resolve buffer's writes (one entry unless vulkan_resolve_skip_disjoint_waw skipped a barrier), and counts.
  std::vector<std::pair<uint64_t, uint64_t>> scaled_resolve_unordered_writes_;
  uint64_t resolve_waw_skipped_ = 0;
  uint64_t resolve_waw_kept_ = 0;

  // wwe2k14 patch (Session 22, EAGER): early texture loads (vulkan_texture_load_eager). Command processor thread
  // only. A plan entry is (the frame's resolve count when the texture was loaded late, the texture), learned in
  // one frame (eager_plan_next_) and used in the next (eager_plan_prev_, in resolve order; a destroyed texture's
  // entry becomes nullptr). eager_pending_: textures to load at the next draw.
  void EagerRotate(uint64_t frame);
  void EagerRecordLoad(VulkanTexture& texture);
  // wwe2k14 patch (Session 87, step 3; vulkan_texture_load_early_cpu): textures the CPU rewrites every frame (the entrance tron
  // video's planes) are loaded at the frame's first draw, not at their use - where the load ends the scene's render pass (three
  // splits a frame in Brock's and Bryan's entrances). Learned like the eager plan: a texture loaded late last frame, invalidated
  // by the CPU only, is due at the first draw of this one; the watch re-arms at the load, so a later CPU write reloads it at its
  // use, as without this setting.
  std::vector<VulkanTexture*> cpu_plan_prev_;
  std::vector<VulkanTexture*> cpu_plan_next_;
  std::vector<VulkanTexture*> cpu_pending_;
  std::vector<VulkanTexture*> cpu_used_;  // planned textures a draw sampled this frame (only these stay planned)
  bool cpu_loading_ = false;
  struct CpuEarlyStats {
    uint64_t loads = 0, learned = 0, reloaded = 0, already_valid = 0, failed = 0;
  } cpu_early_stats_;
  void EagerForget(const VulkanTexture* texture);
  uint64_t eager_frame_ = UINT64_MAX;
  uint32_t eager_resolves_ = 0;
  uint32_t eager_draws_since_resolve_ = 0;
  bool eager_loading_ = false;
  size_t eager_plan_cursor_ = 0;
  std::vector<std::pair<uint32_t, VulkanTexture*>> eager_plan_prev_;
  std::vector<std::pair<uint32_t, VulkanTexture*>> eager_plan_next_;
  std::vector<VulkanTexture*> eager_pending_;
  struct EagerStats {
    uint64_t loads = 0;          // early loads done
    uint64_t learned = 0;        // late loads recorded for the next frame
    uint64_t reloaded = 0;       // textures loaded again at their use after an early load in the same frame
    uint64_t already_valid = 0;  // due, but not outdated (nothing to load)
    uint64_t failed = 0;
  } eager_stats_;

  // wwe2k14 patch (Session 6, CP6): see the types above. Allocated on first use, zero-initialized; only
  // accessed on the command processor thread.
  mutable std::unique_ptr<SamplerParameterCacheSet[]> sampler_parameter_cache_;
  std::unique_ptr<SamplerUseCacheEntry[]> sampler_use_cache_;
  mutable SamplerCacheStats sampler_cache_stats_ = {};

  NativeLoadCounters native_load_counters_;  // wwe2k14 patch (Session 45, NATIVE FRAME)
  // wwe2k14 patch (Session 45, NATIVE TEXWRITE): base page -> the key of the last direct 1x load there.
  std::unordered_map<uint32_t, TextureKey> native_resolve_keys_;
  // wwe2k14 patch (Session 47, NATIVE TRUTH M6): a mip level's start page -> the texture's key, the level and the level's
  // pitch in 32-texel units (learned with native_resolve_keys_, gpu_native_truth only).
  struct NativeMipKey {
    TextureKey key;
    uint32_t level;
    uint32_t pitch_div_32;
  };
  std::unordered_map<uint32_t, NativeMipKey> native_resolve_mip_keys_;
  // wwe2k14 patch (Session 79, gpu_native_truth_fill_cube_faces): a cube face's start page -> the cube's key and the
  // face (array layer), for the 1x tiled single-level k_8_8_8_8 cubes with the R32_UINT storage view.
  struct NativeLayerKey {
    TextureKey key;
    uint32_t layer;
  };
  std::unordered_map<uint32_t, NativeLayerKey> native_resolve_layer_keys_;
  void NativeLearnTextureKey(const VulkanTexture& texture);

  // wwe2k14 patch (Session 7c, TEXLOAD): texture load counts (command processor thread only). The frame part goes to
  // cpf_texload, the window part to the [s7][TEXLOAD] summary.
  struct TexLoadStats {
    // This frame.
    uint32_t frame_loads = 0;
    uint32_t frame_loads_scaled = 0;
    uint64_t frame_bytes = 0;  // written to images by the copies
    // The summary window (from window_first_frame).
    uint64_t window_first_frame = UINT64_MAX;
    uint64_t loads = 0;
    uint64_t loads_scaled = 0;
    uint64_t bytes = 0;
    uint64_t float_converts = 0;
    uint64_t invalidated_gpu = 0;   // a resolve or memory export wrote the texture's memory since its last load
    uint64_t invalidated_cpu = 0;   // the CPU did (both may be counted for one load)
    uint64_t first_loads = 0;       // the texture was never loaded before
    uint64_t transitions = 0;       // the pre-copy barrier transitioned a used image (layout not undefined)
    uint64_t rewrite_all = 0;       // of those, the load rewrites every subresource (could discard)
    uint64_t discards = 0;          // of those, discarded with vulkan_texture_load_discard_old
    uint64_t ring_loads = 0;        // loads with a vulkan_texture_load_scratch_ring entry
    uint64_t ring_waw_barriers = 0;  // image write-after-write barriers (this or the last load used the ring)
    // VulkanCommandProcessor::ScratchRingStats at the window start.
    uint64_t ring_acquisitions_start = 0;
    uint64_t ring_barriers_start = 0;
    uint64_t ring_pre_transitions_start = 0;
    uint64_t ring_grows_start = 0;
    uint64_t ring_releases_start = 0;
  };
  TexLoadStats texload_stats_;
  // The running frame-anatomy capture's texture load CSV (null: no capture), and its rows of the current frame.
  std::FILE* texload_csv_ = nullptr;
  std::string texload_csv_rows_path_;  // the anatomy rows CSV it belongs to
  std::string texload_csv_path_;
  std::string texload_csv_text_;
  uint32_t texload_csv_frame_ = 0;
  void TexLoadCsvFlush();
};

}  // namespace rex::graphics::vulkan
