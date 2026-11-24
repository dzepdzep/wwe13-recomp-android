// wwe2k14 patch (Session 87, D3D layer family F4): the token's draw item taken straight into the native renderer.
//
// The D3D layer's token (CommandProcessor::ExecutePacketType3_D3D_TOKEN) already carries the game's own knowledge of what
// changed (register runs and shader loads, applied to the register file as before) and, per draw chunk, one DRAW item.
// With --gpu_d3d_direct on, the DRAW item is not handed to the stock draw packet path (ExecutePacketType3_DRAW_INDX ->
// ExecutePacketType3Draw -> IssueDrawLean / IssueDraw, whose first act is to infer from the register writes what the
// record already said) but to the backend's handler registered here: it takes the library-draw knowledge from the record
// (a token IS a library indexed draw: no Half 2 attribution needed), classifies the dirty state as the writes noted it,
// and draws through a lean-draw variant with caches keyed by what the record says changed (vertex ranges stamped against
// the shared memory's page-validity epoch). A draw it cannot take goes the stock way, counted per reason.
//
// Declarations only: the base command processor never names a backend type (the pattern of d3d_copy.h).
#pragma once

#include <cstdint>

namespace rex::memory {
class RingBuffer;
}

namespace rex::graphics {

class CommandProcessor;

// The handler consumes the DRAW item's four payload words (viz query word, VGT_DRAW_INITIATOR, DMA base, DMA size) and
// returns true, or leaves the reader exactly where it was and returns false (the stock path then runs).
using D3dDirectDrawHandler = bool (*)(CommandProcessor& cp, rex::memory::RingBuffer* reader);

inline D3dDirectDrawHandler& D3dDirectDrawHandlerSlot() {
  static D3dDirectDrawHandler handler = nullptr;
  return handler;
}
inline void D3dDirectSetDrawHandler(D3dDirectDrawHandler handler) { D3dDirectDrawHandlerSlot() = handler; }

// Called by the token handler for each DRAW item on the drawing side. False: not consumed (switch off, no backend handler,
// or a draw the direct path leaves to the stock path).
inline bool D3dDirectDrawItem(CommandProcessor& cp, rex::memory::RingBuffer* reader) {
  const D3dDirectDrawHandler handler = D3dDirectDrawHandlerSlot();
  return handler != nullptr && handler(cp, reader);
}

}  // namespace rex::graphics
