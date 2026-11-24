// "Unlock all superstars" (launcher setting, WWE13_UNLOCK_ALL=1): while the character select screen builds its lists
// (sub_82839468, roster_mods.cpp), the game's "is this superstar unlocked" check (sub_82288448(id)) answers yes for
// every superstar with a record. Only there: forcing the general unlockables check (sub_82288BD8) everywhere made the
// game announce the unlocks at boot ("Booker T has been unlocked") and would have saved them. Nothing is written to
// the save; turning the setting off locks them again.
#include <atomic>
#include <cstdlib>
#include <cstring>

#include <rex/logging.h>
#include "rex_compat.h"

bool wwe13_select_visible_frames(uint32_t frames);  // roster_mods.cpp
namespace {
bool UnlockAll() {
  static const bool on = [] {
    const char* v = std::getenv("WWE13_UNLOCK_ALL");
    const bool o = v && std::strcmp(v, "1") == 0;
    if (o) REXLOG_INFO("[wwe13-unlock] all superstars selectable (WWE13_UNLOCK_ALL=1; the save is not changed)");
    return o;
  }();
  return on;
}
thread_local int t_select_build = 0;
}  // namespace

void wwe13_unlock_select_build(bool begin) { t_select_build += begin ? 1 : -1; }

namespace {
// (only while the select screen builds: answering yes anywhere else - even the half second after leaving it - made
// the menus announce the unlocks, "Booker T has been unlocked")
bool OnSelectScreen() { return t_select_build > 0; }
// (the cells' lock icons ask sub_82288448 while the screen is up: the grid was seen this frame or the one before)
bool wwe13_select_up() { return wwe13_select_visible_frames(2); }
}  // namespace

// The general unlockables check (n 0-128), forced only on the select screen: the cells' lock icons ask it there.
extern "C" PPC_FUNC(__imp__sub_82288BD8);
extern "C" PPC_FUNC(sub_82288BD8) {
  const int32_t n = int32_t(ctx.r3.u32);
  __imp__sub_82288BD8(ctx, base);
  if (ctx.r3.u32 == 0 && n >= 0 && n < 129 && UnlockAll() && OnSelectScreen()) ctx.r3.u64 = 1;
}

extern "C" PPC_FUNC(__imp__sub_82288448);
extern "C" PPC_FUNC(sub_82288448) {
  const int32_t id = int32_t(ctx.r3.u32);
  __imp__sub_82288448(ctx, base);
  if (ctx.r3.u32 == 0 && (OnSelectScreen() || wwe13_select_up()) && id >= 0 && id < 500 && UnlockAll()) {
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 3) REXLOG_INFO("[wwe13-unlock] superstar {} offered on the select screen", id);
    ctx.r3.u64 = 1;
  }
}
