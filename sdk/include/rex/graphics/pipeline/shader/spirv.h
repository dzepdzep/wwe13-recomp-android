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

#include <atomic>
#include <cstring>
#include <vector>

#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/xenos.h>

namespace rex::graphics {

class SpirvShader : public Shader {
 public:
  class SpirvTranslation : public Translation {
   public:
    explicit SpirvTranslation(SpirvShader& shader, uint64_t modification)
        : Translation(shader, modification) {}
  };

  explicit SpirvShader(xenos::ShaderType shader_type, uint64_t ucode_data_hash,
                       const uint32_t* ucode_dwords, size_t ucode_dword_count,
                       std::endian ucode_source_endian = std::endian::big);

  // Resource bindings are gathered after the successful translation of any
  // modification for simplicity of translation (and they don't depend on
  // modification bits).

  struct TextureBinding {
    uint32_t fetch_constant : 5;
    // Stacked and 3D are separate TextureBindings.
    xenos::FetchOpDimension dimension : 2;
    uint32_t is_signed : 1;
  };
  // Safe to hash and compare with memcmp for layout hashing.
  const std::vector<TextureBinding>& GetTextureBindingsAfterTranslation() const {
    return texture_bindings_;
  }
  const uint32_t GetUsedTextureMaskAfterTranslation() const { return used_texture_mask_; }

  struct SamplerBinding {
    uint32_t fetch_constant : 5;
    xenos::TextureFilter mag_filter : 2;
    xenos::TextureFilter min_filter : 2;
    xenos::TextureFilter mip_filter : 2;
    xenos::AnisoFilter aniso_filter : 3;
  };
  const std::vector<SamplerBinding>& GetSamplerBindingsAfterTranslation() const {
    return sampler_bindings_;
  }

  // wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation): the bindings a translation would have set up,
  // taken from a registered replacement (Shader::ApplyRegisteredAnalysis has the shader-level part). Samplers carry no
  // filter overrides (use the fetch constant). Once the shader has bindings from a translation, this does nothing.
  void ApplyRegisteredBindings(const RegisteredAnalysis& analysis) {
    if (bindings_setup_entered_.test_and_set(std::memory_order_relaxed)) {
      return;
    }
    texture_bindings_.clear();
    sampler_bindings_.clear();
    used_texture_mask_ = 0;
    for (const RegisteredAnalysis::Texture& texture : analysis.textures) {
      TextureBinding& binding = texture_bindings_.emplace_back();
      std::memset(&binding, 0, sizeof(binding));  // (stable for the layout hash, as the translator's)
      binding.fetch_constant = texture.fetch_constant;
      binding.dimension = texture.dimension;
      binding.is_signed = texture.is_signed ? 1 : 0;
      used_texture_mask_ |= UINT32_C(1) << texture.fetch_constant;
    }
    for (const RegisteredAnalysis::Sampler& sampler : analysis.samplers) {
      SamplerBinding& binding = sampler_bindings_.emplace_back();
      binding.fetch_constant = sampler.fetch_constant;
      binding.mag_filter = xenos::TextureFilter(sampler.mag_filter);
      binding.min_filter = xenos::TextureFilter(sampler.min_filter);
      binding.mip_filter = xenos::TextureFilter(sampler.mip_filter);
      binding.aniso_filter = xenos::AnisoFilter(sampler.aniso_filter);
    }
  }
  // Pixel shaders (gpu_native_swap_skip_translation_pixel): the bindings of the shader's REPLACEMENT translation - the images
  // and samplers its SPIR-V declares, in the canonical order. They belong to the replacement translation (native_ext has
  // kNativeExtSwap), not to the shader: GetTextureBindingsAfterTranslation and the other getters stay the emulated
  // translation's (the translator's), and a reader that knows which translation its draw uses asks for that one's.
  bool has_replacement_bindings() const { return has_replacement_bindings_; }
  const std::vector<TextureBinding>& TextureBindingsFor(bool replacement) const {
    return replacement ? replacement_texture_bindings_ : texture_bindings_;
  }
  const std::vector<SamplerBinding>& SamplerBindingsFor(bool replacement) const {
    return replacement ? replacement_sampler_bindings_ : sampler_bindings_;
  }
  uint32_t UsedTextureMaskFor(bool replacement) const {
    return replacement ? replacement_used_texture_mask_ : used_texture_mask_;
  }
  void SetReplacementBindings(const RegisteredAnalysis& analysis) {
    replacement_texture_bindings_.clear();
    replacement_sampler_bindings_.clear();
    replacement_used_texture_mask_ = 0;
    for (const RegisteredAnalysis::Texture& texture : analysis.textures) {
      TextureBinding& binding = replacement_texture_bindings_.emplace_back();
      std::memset(&binding, 0, sizeof(binding));
      binding.fetch_constant = texture.fetch_constant;
      binding.dimension = texture.dimension;
      binding.is_signed = texture.is_signed ? 1 : 0;
      replacement_used_texture_mask_ |= UINT32_C(1) << texture.fetch_constant;
    }
    for (const RegisteredAnalysis::Sampler& sampler : analysis.samplers) {
      SamplerBinding& binding = replacement_sampler_bindings_.emplace_back();
      binding.fetch_constant = sampler.fetch_constant;
      binding.mag_filter = xenos::TextureFilter(sampler.mag_filter);
      binding.min_filter = xenos::TextureFilter(sampler.min_filter);
      binding.mip_filter = xenos::TextureFilter(sampler.mip_filter);
      binding.aniso_filter = xenos::AnisoFilter(sampler.aniso_filter);
    }
    has_replacement_bindings_ = true;
  }
  // Forget the registered bindings (a real translation sets them up again). The replacement's are not these.
  void DropRegisteredBindings() {
    texture_bindings_.clear();
    sampler_bindings_.clear();
    used_texture_mask_ = 0;
    bindings_setup_entered_.clear(std::memory_order_relaxed);
  }

 protected:
  Translation* CreateTranslationInstance(uint64_t modification) override;

 private:
  friend class SpirvShaderTranslator;

  std::atomic_flag bindings_setup_entered_ = ATOMIC_FLAG_INIT;
  std::vector<TextureBinding> texture_bindings_;
  std::vector<SamplerBinding> sampler_bindings_;
  uint32_t used_texture_mask_ = 0;
  // wwe2k14 patch (Session 87): the replacement translation's bindings, see SetReplacementBindings.
  bool has_replacement_bindings_ = false;
  std::vector<TextureBinding> replacement_texture_bindings_;
  std::vector<SamplerBinding> replacement_sampler_bindings_;
  uint32_t replacement_used_texture_mask_ = 0;
};

}  // namespace rex::graphics
