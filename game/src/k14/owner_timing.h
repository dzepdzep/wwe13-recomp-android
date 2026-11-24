// WWE '13 port: owner mode is not part of this port.
#pragma once
namespace wwe2k14::owner_timing {
inline bool Fence(PPCContext&, uint8_t*) { return false; }
inline bool Interrupt(PPCContext&, uint8_t*) { return false; }
inline bool Flip(PPCContext&, uint8_t*) { return false; }
inline bool Submit(PPCContext&, uint8_t*) { return false; }
}  // namespace wwe2k14::owner_timing
