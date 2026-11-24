#pragma once
// Session 87 stage 4: the tfetch sampler filter overrides of the registered shaders (the translator's sampler_bindings: mag, min,
// mip as xenos::TextureFilter, aniso as xenos::AnisoFilter; 3 / 3 / 3 / 7 = follow the fetch constant), recorded by
// tools/sampler_record/record.py. A replacement's sampler applies them (SpirvShader::SetReplacementBindings). Hashes and numbers
// only: an annotation of the shaders, no microcode.
#include <cstdint>

namespace rex::graphics::vulkan::shader_swap {

struct SamplerOverrideRow {
  uint64_t hash;
  uint8_t vertex;  // 1 = a vertex shader
  uint8_t fetch_constant;
  uint8_t mag_filter, min_filter, mip_filter, aniso_filter;
};

inline constexpr SamplerOverrideRow kSamplerOverrideRows[] = {
    {0x022F374B74F15579ull, 0, 0, 3, 3, 3, 0},
    {0x0D34F45940731536ull, 1, 16, 3, 3, 3, 0},
    {0x0E2A3DF5BAA2CA18ull, 1, 16, 3, 3, 3, 0},
    {0x1233F68C15FDF948ull, 0, 0, 3, 3, 3, 0},
    {0x1B66FF87A6433D53ull, 0, 12, 3, 3, 3, 0},
    {0x1D20CB5D36B02E5Bull, 0, 12, 3, 3, 3, 0},
    {0x1D3A5C6685E70D68ull, 1, 17, 3, 3, 3, 0},
    {0x20395ADD7E10418Full, 1, 16, 3, 3, 3, 0},
    {0x3C06B3329BE81996ull, 0, 12, 3, 3, 3, 0},
    {0x4C9D02186B6E5E9Dull, 0, 12, 3, 3, 3, 0},
    {0x4E2CD8031D93DE3Eull, 0, 0, 3, 3, 3, 0},
    {0x5344BF064B17B7C2ull, 1, 16, 3, 3, 3, 0},
    {0x91BF021CA7C0C1A5ull, 0, 12, 3, 3, 3, 0},
    {0x925085B612A44F10ull, 0, 12, 3, 3, 3, 0},
    {0x94B69B274F560003ull, 1, 16, 3, 3, 3, 0},
    {0x94FE8F169F50CF85ull, 1, 16, 3, 3, 3, 0},
    {0x9B46FDF960C4CC1Full, 0, 12, 3, 3, 3, 0},
    {0x9C2FD0E90E71F4A3ull, 0, 12, 3, 3, 3, 0},
    {0x9F8F910305279A11ull, 0, 12, 3, 3, 3, 0},
    {0xA0A9844B8A45CC6Bull, 0, 12, 3, 3, 3, 0},
    {0xB08B5E734D48F5D0ull, 0, 12, 3, 3, 3, 0},
    {0xB1F58C0AA0E3C83Aull, 0, 12, 3, 3, 3, 0},
    {0xB63CD5FCA34C147Dull, 0, 12, 3, 3, 3, 0},
    {0xBBB2E028DCF36E8Dull, 0, 12, 3, 3, 3, 0},
    {0xCAA5A55D7869AA08ull, 1, 16, 3, 3, 3, 0},
    {0xE86BD728B5D916BAull, 0, 12, 3, 3, 3, 0},
    {0xE98D0C3605046302ull, 0, 12, 3, 3, 3, 0},
    {0xED3793930D3A2C1Dull, 0, 12, 3, 3, 3, 0},
    {0xEFCB021A03BA3E6Aull, 1, 16, 3, 3, 3, 0},
    {0xF235532B75570EAFull, 0, 0, 3, 3, 3, 0},
    {0xF75AE782D8A42A61ull, 0, 12, 3, 3, 3, 0},
    {0xFEE45804EDD0EC05ull, 0, 12, 3, 3, 3, 0},
};

}  // namespace rex::graphics::vulkan::shader_swap
