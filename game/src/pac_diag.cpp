// wwe13 - pac part guard: the game unpacks a .pac part (byte-pair compressed) into a buffer its caller sized, without
// checking that the part fits (sub_82B1E360: pac, dest, dest size, part index, entry name). A part bigger than the
// buffer overwrote the stack and crashed (a merged DLC chEtc). Such a part is now named in the log and skipped (the
// call reports failure, as for a missing part) instead of crashing the game.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <rex/logging.h>
#include <rex/ppc/context.h>
#include "rex_compat.h"

extern "C" PPC_FUNC(__imp__sub_82B1E360);
extern "C" PPC_FUNC(__imp__sub_82B1E228);
void wwe13_ps_prompts_part(uint8_t* base, uint32_t dest, uint32_t size);  // ps_prompts.cpp
extern "C" PPC_FUNC(sub_82B1E360) {
  const uint32_t dest = ctx.r4.u32;
  const uint32_t pac = ctx.r3.u32, dest_size = ctx.r5.u32, index = ctx.r6.u32, name = ctx.r7.u32;
  if (pac && dest_size && index != 0xFFFFFFFFu) {
    PPCContext c = ctx;
    c.r3.u64 = pac;
    c.r4.u64 = index;
    __imp__sub_82B1E228(c, base);  // (the part's unpacked size)
    const uint32_t size = c.r3.u32;
    if (size > dest_size) {
      static std::vector<uint64_t> said;
      const uint64_t key = (uint64_t(name) << 32) | index;
      bool seen = false;
      for (uint64_t k : said) seen |= k == key;
      if (!seen && said.size() < 200) {
        said.push_back(key);
        std::string n;
        for (uint32_t i = 0; name && i < 16; ++i) {
          const char ch = char(PPC_LOAD_U8(name + i));
          if (!ch) break;
          n += ch;
        }
        REXLOG_WARN("[wwe13-pac] part {} of '{}' unpacks to {} bytes, its buffer holds {}: skipped (caller {:08x})",
                    index, n, size, dest_size, uint32_t(ctx.lr));
      }
      ctx.r3.u64 = 0;
      return;
    }
  }
  __imp__sub_82B1E360(ctx, base);
  if (pac && dest && ctx.r3.u32 && index != 0xFFFFFFFFu) {  // PlayStation icons on stored parts too
    PPCContext c = ctx;
    c.r3.u64 = pac;
    c.r4.u64 = index;
    __imp__sub_82B1E228(c, base);
    wwe13_ps_prompts_part(base, dest, c.r3.u32);
  }
}
