// WWE '13 port: stage 4 (texture objects) hooks game-side texture code; not ported yet.
#pragma once
#include <rex/cvar.h>
REXCVAR_DECLARE(int32_t, wwe2k14_d3d_textures);
namespace wwe2k14::d3d_tex {
inline void OnDraw(uint8_t*, uint32_t, uint64_t) {}
}
