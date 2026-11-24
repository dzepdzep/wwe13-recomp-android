/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <cinttypes>
#include <cstring>
#include <utility>

#include <fmt/format.h>

#include <rex/filesystem.h>
#include <rex/graphics/format/ucode.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/math.h>
#include <rex/memory.h>
#include <rex/string.h>

namespace rex::graphics {
using namespace ucode;

Shader::Shader(xenos::ShaderType shader_type, uint64_t ucode_data_hash,
               const uint32_t* ucode_dwords, size_t ucode_dword_count,
               std::endian ucode_source_endian)
    : shader_type_(shader_type), ucode_data_hash_(ucode_data_hash) {
  // We keep ucode data in host native format so it's easier to work with.
  ucode_data_.resize(ucode_dword_count);
  if (std::endian::native != ucode_source_endian) {
    memory::copy_and_swap(ucode_data_.data(), ucode_dwords, ucode_dword_count);
  } else {
    std::memcpy(ucode_data_.data(), ucode_dwords, sizeof(uint32_t) * ucode_dword_count);
  }
}

Shader::~Shader() {
  for (auto it : translations_) {
    delete it.second;
  }
}

std::string Shader::Translation::GetTranslatedBinaryString() const {
  std::string result;
  result.resize(translated_binary_.size());
  std::memcpy(const_cast<char*>(result.data()), translated_binary_.data(),
              translated_binary_.size());
  return result;
}

std::pair<std::filesystem::path, std::filesystem::path> Shader::Translation::Dump(
    const std::filesystem::path& base_path, const char* path_prefix) const {
  if (!is_valid()) {
    return std::make_pair(std::filesystem::path(), std::filesystem::path());
  }

  std::filesystem::path path = base_path;
  // Ensure target path exists.
  std::filesystem::path target_path = base_path;
  if (!target_path.empty()) {
    target_path = std::filesystem::absolute(target_path);
    std::filesystem::create_directories(target_path);
  }

  const char* type_extension = shader().type() == xenos::ShaderType::kVertex ? "vert" : "frag";

  std::filesystem::path binary_path =
      target_path / fmt::format("shader_{:016X}_{:016X}{}.{}.bin.{}", shader().ucode_data_hash(), modification(),
                                native_ext_ ? fmt::format("_e{:08X}", native_ext_) : std::string(), path_prefix,
                                type_extension);
  FILE* binary_file = filesystem::OpenFile(binary_path, "wb");
  if (binary_file) {
    fwrite(translated_binary_.data(), sizeof(*translated_binary_.data()), translated_binary_.size(),
           binary_file);
    fclose(binary_file);
  }

  std::filesystem::path disasm_path;
  if (!host_disassembly_.empty()) {
    disasm_path =
        target_path / fmt::format("shader_{:016X}_{:016X}.{}.{}", shader().ucode_data_hash(),
                                  modification(), path_prefix, type_extension);
    FILE* disasm_file = filesystem::OpenFile(disasm_path, "w");
    if (disasm_file) {
      fwrite(host_disassembly_.data(), sizeof(*host_disassembly_.data()), host_disassembly_.size(),
             disasm_file);
      fclose(disasm_file);
    }
  }

  return std::make_pair(std::move(binary_path), std::move(disasm_path));
}

Shader::Translation* Shader::GetOrCreateTranslation(uint64_t modification, bool* is_new) {
  auto it = translations_.find(modification);
  if (it != translations_.end()) {
    if (is_new) {
      *is_new = false;
    }
    return it->second;
  }
  Translation* translation = CreateTranslationInstance(modification);
  translations_.emplace(modification, translation);
  if (is_new) {
    *is_new = true;
  }
  return translation;
}

Shader::Translation* Shader::GetOrCreateNativeTranslation(uint64_t modification, uint32_t native_ext, bool* is_new) {
  if (!native_ext) {
    return GetOrCreateTranslation(modification, is_new);
  }
  const uint64_t key = TranslationKey(modification, native_ext);
  auto it = translations_.find(key);
  if (it != translations_.end()) {
    if (is_new) {
      *is_new = false;
    }
    return it->second;
  }
  Translation* translation = CreateTranslationInstance(modification);
  translation->native_ext_ = native_ext;
  translations_.emplace(key, translation);
  if (is_new) {
    *is_new = true;
  }
  return translation;
}

void Shader::DestroyTranslation(uint64_t modification) {
  auto it = translations_.find(modification);
  if (it == translations_.end()) {
    return;
  }
  delete it->second;
  translations_.erase(it);
}

std::pair<std::filesystem::path, std::filesystem::path> Shader::DumpUcode(
    const std::filesystem::path& base_path) const {
  // Ensure target path exists.
  std::filesystem::path target_path = base_path;
  if (!target_path.empty()) {
    target_path = std::filesystem::absolute(target_path);
    std::filesystem::create_directories(target_path);
  }

  const char* type_extension = type() == xenos::ShaderType::kVertex ? "vert" : "frag";

  std::filesystem::path binary_path =
      target_path / fmt::format("shader_{:016X}.ucode.bin.{}", ucode_data_hash(), type_extension);
  FILE* binary_file = filesystem::OpenFile(binary_path, "wb");
  if (binary_file) {
    fwrite(ucode_data().data(), sizeof(*ucode_data().data()), ucode_data().size(), binary_file);
    fclose(binary_file);
  }

  std::filesystem::path disasm_path;
  if (is_ucode_analyzed()) {
    disasm_path =
        target_path / fmt::format("shader_{:016X}.ucode.{}", ucode_data_hash(), type_extension);
    FILE* disasm_file = filesystem::OpenFile(disasm_path, "w");
    if (disasm_file) {
      fwrite(ucode_disassembly().data(), sizeof(*ucode_disassembly().data()),
             ucode_disassembly().size(), disasm_file);
      fclose(disasm_file);
    }
  }

  return std::make_pair(std::move(binary_path), std::move(disasm_path));
}

Shader::Translation* Shader::CreateTranslationInstance(uint64_t modification) {
  // Default implementation for simple cases like ucode disassembly.
  return new Translation(*this, modification);
}


// wwe2k14 patch (Session 87, stage 4, gpu_native_swap_skip_translation): see the header.
void Shader::ApplyRegisteredAnalysis(const RegisteredAnalysis& analysis) {
  if (is_ucode_analyzed_) {
    return;
  }
  const bool vertex = type() == xenos::ShaderType::kVertex;
  constant_register_map_ = {};
  if (analysis.float_dynamic_addressing) {
    constant_register_map_.float_dynamic_addressing = true;
    constant_register_map_.float_count = 256;
    std::memset(constant_register_map_.float_bitmap, UINT8_MAX, sizeof(constant_register_map_.float_bitmap));
  } else {
    constant_register_map_.float_count = 0;
    for (int i = 0; i < 4; ++i) {
      constant_register_map_.float_bitmap[i] = analysis.float_bitmap[i];
      constant_register_map_.float_count += rex::bit_count(analysis.float_bitmap[i]);
    }
  }
  std::memcpy(constant_register_map_.bool_bitmap, analysis.bool_bitmap, sizeof(analysis.bool_bitmap));
  constant_register_map_.loop_bitmap = analysis.loop_bitmap;
  vertex_bindings_.clear();
  if (vertex && analysis.stream_fetch_constant < 96) {
    VertexBinding& binding = vertex_bindings_.emplace_back();
    binding.binding_index = 0;
    binding.fetch_constant = analysis.stream_fetch_constant;
    binding.stride_words = analysis.stream_stride_words;
    constant_register_map_.vertex_fetch_bitmap[analysis.stream_fetch_constant / 32] |=
        uint32_t(1) << (analysis.stream_fetch_constant % 32);
  }
  texture_bindings_.clear();
  uint32_t unique_bindings = 0;
  for (const RegisteredAnalysis::Texture& texture : analysis.textures) {
    TextureBinding binding;
    binding.binding_index = size_t(-1);
    binding.fetch_constant = texture.fetch_constant;
    for (const TextureBinding& earlier : texture_bindings_) {
      if (earlier.fetch_constant == texture.fetch_constant) {
        binding.binding_index = earlier.binding_index;
        break;
      }
    }
    if (binding.binding_index == size_t(-1)) {
      binding.binding_index = unique_bindings++;
    }
    binding.fetch_instr.dimension = texture.dimension;
    texture_bindings_.push_back(std::move(binding));
  }
  label_addresses_.clear();
  cf_pair_index_bound_ = 0;
  register_static_address_bound_ = analysis.register_static_address_bound;
  writes_interpolators_ = analysis.writes_interpolators;
  writes_point_size_edge_flag_kill_vertex_ = 0;
  writes_color_targets_ = analysis.writes_color_targets;
  uses_register_dynamic_addressing_ = false;
  kills_pixels_ = analysis.kills_pixels;
  uses_texture_fetch_instruction_results_ = !analysis.textures.empty();
  writes_depth_ = analysis.writes_depth;
  cf_memexport_info_.clear();
  memexport_eM_written_ = 0;
  memexport_eM_potentially_written_before_end_ = 0;
  memexport_stream_constants_.clear();
  ucode_disassembly_.clear();
  analysis_registered_ = true;
  is_ucode_analyzed_ = true;
  // wwe2k14 patch (Session 95, Lane C): this analysis has no control flow, so a census of it would read "no vertex fetch". The full analysis (PrepareFullAnalysis) gives the real form.
  vertex_input_form_ = {};
  vertex_input_form_.refusal = "registered-analysis";
}

void Shader::DropRegisteredAnalysis() {
  if (!analysis_registered_) {
    return;
  }
  vertex_bindings_.clear();
  texture_bindings_.clear();
  constant_register_map_ = {};
  label_addresses_.clear();
  cf_pair_index_bound_ = 0;
  register_static_address_bound_ = 0;
  writes_interpolators_ = 0;
  writes_point_size_edge_flag_kill_vertex_ = 0;
  writes_color_targets_ = 0;
  uses_register_dynamic_addressing_ = false;
  kills_pixels_ = false;
  uses_texture_fetch_instruction_results_ = false;
  writes_depth_ = false;
  cf_memexport_info_.clear();
  memexport_eM_written_ = 0;
  memexport_eM_potentially_written_before_end_ = 0;
  memexport_stream_constants_.clear();
  ucode_disassembly_.clear();
  analysis_registered_ = false;
  is_ucode_analyzed_ = false;
  vertex_input_form_ = {};  // (Session 95, Lane C)
}

}  // namespace rex::graphics
