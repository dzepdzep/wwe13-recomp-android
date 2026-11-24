// wwe2k14 patch (Session 87, D3D layer family F2): the private copy-family packet. The game's D3D device hooks (clear,
// resolve, GPR split, tiling) write ONE small type-3 packet into the ring that carries a record built from the device
// object's own state; the command processor hands it to the backend, which executes it into the owned frame's clear /
// copy entry points. No register decode, no PM4 of the library's own writers.
//
// Shared by the SDK and the game exe (plain constants and field indexes only; no SDK types).
//
// Packet: type-3, opcode kD3dCopyOpcode (0x7D; F1's token is 0x7C), payload (count dwords after the header):
//   [0] kD3dCopyMagic   [1] sequence (counts up, checked: a gap is logged loudly)   [2] kind   [3] sem_dwords
//   [4 ...] the semantic block (kind-specific, dword indexes below)
// All payload words are big-endian in guest memory like every ring word (the reader byte-swaps); floats travel as their
// IEEE bit patterns.
#pragma once

#include <cstdint>

namespace rex::memory {
class RingBuffer;
}

namespace rex::graphics {

inline constexpr uint32_t kD3dCopyOpcode = 0x7D;
inline constexpr uint32_t kD3dCopyMagic = 0xD3D20001u;
inline constexpr uint32_t kD3dCopyHeaderDwords = 4;
inline constexpr uint32_t kD3dCopyMaxSemDwords = 64;

// The layout version the capability cvar gpu_d3d_copy_abi reports (an int). The game exe keeps its own copy of this
// layout (src\d3d_copy_layout.h) and takes the records only when the cvar equals the number it was written for - the
// exe never includes this header, so a changed layout never forces a re-pin.
inline constexpr int32_t kD3dCopyAbi = 3;

enum D3dCopyKind : uint32_t {
  kD3dCopyClear = 1,
  kD3dCopyGprSplit = 2,  // census only: the 360's shader register split, dropped
  kD3dCopyResolve = 3,
  kD3dCopyKindCount
};

// Clear semantic block (kD3dCopyClear). One pass = one colour target (or depth only) and its final rectangle, clipped
// exactly as the library's clear clips it (the surface or tile size, the viewport, the scissor when enabled).
namespace d3d_clear_sem {
inline constexpr uint32_t kFlags = 0;       // the call's flags: bits 0-3 colour targets, 0x10 depth, 0x20 stencil
inline constexpr uint32_t kColor = 1;       // 4 floats r, g, b, a (words 1-4)
inline constexpr uint32_t kDepth = 5;       // float
inline constexpr uint32_t kStencil = 6;
inline constexpr uint32_t kSurfaceInfo = 7;  // device+10368: RB_SURFACE_INFO shadow (pitch in pixels bits 0-13, MSAA 16-17)
inline constexpr uint32_t kDepthInfo = 8;    // device+10376: RB_DEPTH_INFO shadow (0 when no depth surface)
inline constexpr uint32_t kColorInfo = 9;    // the pass colour target's RB_COLOR_INFO (surface+28, format fixed as the clear does); 0: none
inline constexpr uint32_t kRect = 10;        // x0 y0 x1 y1 in guest pixels (words 10-13)
inline constexpr uint32_t kPassRt = 14;      // 0-3 the colour target, 0xFFFFFFFF depth only
inline constexpr uint32_t kVtxCntl = 15;     // device+10688: PA_SU_VTX_CNTL (the half-pixel offset of the clear's vertices)
// Appended (the framing is by count, so an older executor ignores them): the colour state a DEPTH-ONLY clear inherits - the library's
// depth-only branch writes no RB_COLOR_MASK / RB_BLENDCONTROL0 / RB_COLORCONTROL, the draw runs with what the last flush left,
// which is the device's shadow of them (device+10460 / +10552 / +10556).
inline constexpr uint32_t kColorMask = 16;
inline constexpr uint32_t kBlendControl = 17;
inline constexpr uint32_t kColorControl = 18;
inline constexpr uint32_t kDwords = 16;           // the required part; kDwordsFull with the inherited colour state
inline constexpr uint32_t kDwordsFull = 19;
}  // namespace d3d_clear_sem

// Resolve semantic block (kD3dCopyResolve): the values the library's resolve (0x82CAEA30) leaves in its device object
// as register images, computed by the game's record builder from the call's arguments and the destination texture
// header, plus the surface words of the device and the rectangle (already clipped / aligned as the library does).
namespace d3d_resolve_sem {
inline constexpr uint32_t kFlags = 0;       // the call's flags (sample bits added as the library adds them)
inline constexpr uint32_t kCopyControl = 1;  // RB_COPY_CONTROL image
inline constexpr uint32_t kDestBase = 2;     // RB_COPY_DEST_BASE
inline constexpr uint32_t kDestPitch = 3;    // RB_COPY_DEST_PITCH
inline constexpr uint32_t kDestInfo = 4;     // RB_COPY_DEST_INFO
inline constexpr uint32_t kDepthClear = 5;   // RB_DEPTH_CLEAR
inline constexpr uint32_t kColorClear = 6;   // RB_COLOR_CLEAR
inline constexpr uint32_t kColorClearLo = 7;  // RB_COLOR_CLEAR_LO
inline constexpr uint32_t kSurfaceInfo = 8;  // device+10368
inline constexpr uint32_t kColorInfo = 9;    // RB_COLOR_INFO of the copied target (device+10372 / 10376 + 4n)
inline constexpr uint32_t kDepthInfo = 10;   // device+10376
inline constexpr uint32_t kRect = 11;        // x0 y0 x1 y1 (words 11-14)
inline constexpr uint32_t kVtxCntl = 15;     // device+10688: PA_SU_VTX_CNTL
inline constexpr uint32_t kDwords = 16;
}  // namespace d3d_resolve_sem

// Hook-side counters the backend prints in its census (reasons the game side did not take a call are the game's own).
struct D3dCopyCensus {
  uint64_t packets = 0, clears = 0, clears_held = 0, clears_lost = 0, gpr_splits = 0, bad = 0;
  uint64_t clears_stock = 0, clears_stock_hold_off = 0, clears_stock_refused = 0;
  uint64_t resolves = 0, resolves_owned = 0, resolves_empty = 0, resolves_stock_clear = 0, resolves_stock_refused = 0,
           resolves_stock_other = 0;
  uint64_t resolves_issued = 0, resolves_det_skipped = 0;  // (Session 87 review: every resolve is IssueCopy's now)
};

class CommandProcessor;

// The backend registers its executor once (a static initialiser in its own file); the base command processor's
// type-3 dispatch calls D3dCopyPacket for opcode kD3dCopyOpcode (src/graphics/d3d_copy.cpp): it reads and checks the
// payload (magic, sequence) and hands it to the registered handler. Only the command thread calls either.
using D3dCopyHandler = bool (*)(CommandProcessor& cp, uint32_t kind, const uint32_t* sem, uint32_t sem_count);
void D3dCopySetHandler(D3dCopyHandler handler);
bool D3dCopyPacket(CommandProcessor& cp, rex::memory::RingBuffer* reader, uint32_t count);

}  // namespace rex::graphics
