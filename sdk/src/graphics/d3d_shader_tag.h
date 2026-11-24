#pragma once
// wwe2k14 patch (Session 90, TRUE NATIVE memory): the microcode address of a D3D-layer shader load (IM_LOAD word, record SHADER
// item, inline token). Bit 31 set = a GUEST VIRTUAL address (the game exe keeps its patched vertex-shader copies in the virtual
// heap, so the layer takes no physical pool); clear = physical, as the library's IM_LOAD. A physical IM_LOAD address never has
// bit 31 (29-bit physical space). The address only locates the bytes: the shader cache is keyed by the microcode itself.
#include <cstdint>

#include <rex/system/xmemory.h>

namespace rex::graphics {

inline constexpr uint32_t kD3dShaderVirtualTag = 0x80000000u;

inline uint32_t* D3dShaderMicrocodePtrOf(memory::Memory* memory, uint32_t address) {
  return (address & kD3dShaderVirtualTag) ? memory->TranslateVirtual<uint32_t*>(address & ~kD3dShaderVirtualTag)
                                          : memory->TranslatePhysical<uint32_t*>(address);
}

// half2_ucode_: equality key against the library's draw packets (physical, masked to 29 bits); a tagged address keeps its
// bit 31 so it never equals one.
inline uint32_t D3dShaderHalf2Key(uint32_t address) {
  return (address & kD3dShaderVirtualTag) ? address : (address & 0x1FFFFFFFu);
}

}  // namespace rex::graphics
