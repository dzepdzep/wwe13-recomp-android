// WWE '13: build the 3D scene without MSAA as ONE tile instead of predicated-tiling strips (port of WWE 2K14's
// tiling_hooks.h, --wwe2k14_force_no_msaa). The engine's scene-target builder sub_82B0AD40 (2K14 sub_82C22AF8) fills the
// 256-byte layout at scene object + 12 (width, height, MSAA type, tile count, tiles); here, right after that, the layout
// becomes the engine's own 1-tile 1x layout, and the "type 0 = no scene targets" gate is kept open so the targets are
// still made (the earlier "AA off" attempt in WWE '13 lost its scene targets - corrupt picture). Default on with the
// native renderer: it is what lets the no-ring levels run (wwe2k14_d3d_noring needs one tile). WWE13_NO_MSAA=0 = off.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "rex_compat.h"
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/system/kernel_state.h>
#include <rex/runtime.h>

REXCVAR_DEFINE_BOOL(wwe2k14_force_no_msaa, true, "wwe2k14",
                    "WWE13: the scene as one tile without MSAA (2K14's switch name; the no-ring levels check it)");

namespace {
constexpr uint32_t kLayoutWidth = 12, kLayoutHeight = 16, kLayoutType = 20, kLayoutTiles = 24, kLayoutTile0 = 28;
uint32_t Ld(uint32_t a) { return rex::memory::load_and_swap<uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual<const uint8_t*>(a)); }
void St(uint32_t a, uint32_t v) { rex::memory::store_and_swap<uint32_t>(REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(a), v); }
thread_local uint32_t t_forced_object = 0;
bool Enabled() {
  static const bool on = [] {
    const char* v = std::getenv("WWE13_NO_MSAA");
    if (v && std::strcmp(v, "0") == 0) rex::cvar::SetFlagByName("wwe2k14_force_no_msaa", "false");
    return REXCVAR_GET(wwe2k14_force_no_msaa);
  }();
  return on;
}
}  // namespace

void wwe13_tiling_layout_chosen(PPCRegister& r1, PPCRegister& r31) {
  (void)r1;
  t_forced_object = 0;
  if (!Enabled()) return;
  const uint32_t object = r31.u32;
  const uint32_t w = Ld(object + kLayoutWidth), h = Ld(object + kLayoutHeight), type = Ld(object + kLayoutType);
  if (type == 0 || int32_t(w) <= 0 || int32_t(h) <= 0) return;
  const uint64_t tiles = (uint64_t(w) + 79) / 80 * ((uint64_t(h) + 15) / 16);
  if (2 * tiles > 2048) return;  // 1x colour + depth must fit the EDRAM
  const uint32_t old_tiles = Ld(object + kLayoutTiles);
  St(object + kLayoutType, 0);
  St(object + kLayoutTiles, 1);
  St(object + kLayoutTile0 + 0, 0);
  St(object + kLayoutTile0 + 4, 0);
  St(object + kLayoutTile0 + 8, w);
  St(object + kLayoutTile0 + 12, h);
  t_forced_object = object;
  REXLOG_INFO("[wwe13-tiling] scene object 0x{:08X}: {}x{} MSAA type {} in {} tile(s) -> 1x, 1 tile", object, w, h, type,
              old_tiles);
}

void wwe13_tiling_create_gate(PPCRegister& r11, PPCRegister& r31) {
  if (t_forced_object == r31.u32 && r11.u32 == 0) r11.u64 = 1;
}
