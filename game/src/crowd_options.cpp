// Crowd options (Android performance work).
//
// The stands crowd is drawn member by member: the block draw sub_8249E108 (frame 368, record scratch r1+192, member
// matrix r1+128) sets each group's mesh (sub_82AA0998 SetStreamSource, sub_82AA0A68 SetIndices) and then, per visible
// member, the member's world matrix as VS constants c76-c78 (sub_82AA2240, r4 = 76, r6 = 3) and one draw
// (sub_82AA0B28). Distant members are flat cards drawn by the card loop sub_82499190 (frame 224, scratch r1+80,
// matrix r1+112), one strip each. The same layout as WWE 2K14's audience_draw_block / card loop (sub_82518340 /
// sub_82512F70 there; see the WWE 2K14 recomp's scene_crowd.h), so the same techniques apply.
//
// WWE13_CROWD=0 hides the crowd (both functions skipped): for measuring what the crowd costs and as a speed option.
//
// Crowd batching (WWE13_CROWD_BATCH, default 1; 2K14's wwe2k14_scene_crowd 1): inside the block draw, each mesh group's
// first member is recorded as the game does; the other members' world matrices (c76-c78, 12 dwords) are collected
// instead of recorded (their op-21 constants and op-15 draw skipped) and handed to the SDK as one run
// (CommandProcessor::SceneCrowdPush). When the command processor executes the first member's draw it adds the others
// as instances of that draw (gpu_scene_crowd_batch): one host draw per group instead of one per member, and the game's
// display list is shorter. The SDK stands down on its own when a list is replayed twice (scene_crowd_blocked).

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <rex/logging.h>
#include <rex/ppc/context.h>
#include "rex_compat.h"

namespace {
bool CrowdHidden() {
  static const bool hidden = [] {
    const char* value = std::getenv("WWE13_CROWD");
    const bool h = value != nullptr && std::strcmp(value, "0") == 0;
    if (h) REXLOG_INFO("[wwe13-crowd] crowd hidden (WWE13_CROWD=0)");
    return h;
  }();
  return hidden;
}
}  // namespace

void wwe13_crowd_push(const uint32_t* first, const uint32_t* others, uint32_t count);
bool wwe13_crowd_blocked();

namespace {
bool CrowdBatch() {
  static const bool on = [] {
    const char* value = std::getenv("WWE13_CROWD_BATCH");
    return !(value != nullptr && std::strcmp(value, "0") == 0);
  }();
  return on;
}
struct CrowdRecord {
  bool in_block = false;
  bool have_first = false;
  bool skip_draw = false;
  uint32_t first[12] = {};
  std::vector<uint32_t> others;
  void Flush() {
    if (have_first && !others.empty()) {
      static bool said = false;
      if (!said) {
        said = true;
        REXLOG_INFO("[wwe13-crowd] batching: first run of {} members pushed", others.size() / 12 + 1);
      }
      wwe13_crowd_push(first, others.data(), uint32_t(others.size() / 12));
    }
    have_first = false;
    skip_draw = false;
    others.clear();
  }
};
thread_local CrowdRecord t_crowd;
}  // namespace

extern "C" PPC_FUNC(__imp__sub_8249E108);
extern "C" PPC_FUNC(sub_8249E108) {
  if (CrowdHidden()) return;
  static bool said = false;
  if (!said) {
    said = true;
    REXLOG_INFO("[wwe13-crowd] block draw seen (batch {}, blocked {})", CrowdBatch(), wwe13_crowd_blocked());
  }
  if (!CrowdBatch() || wwe13_crowd_blocked()) {
    __imp__sub_8249E108(ctx, base);
    return;
  }
  t_crowd.in_block = true;
  t_crowd.have_first = false;
  t_crowd.others.clear();
  __imp__sub_8249E108(ctx, base);
  t_crowd.Flush();
  t_crowd.in_block = false;
}

// SetStreamSource (a mesh group starts): the previous group's run goes out.
extern "C" PPC_FUNC(__imp__sub_82AA0998);
extern "C" PPC_FUNC(sub_82AA0998) {
  if (t_crowd.in_block) t_crowd.Flush();
  __imp__sub_82AA0998(ctx, base);
}

// Record "set VS float constants" (r4 first register, r5 data, r6 count of vec4s).
extern "C" PPC_FUNC(__imp__sub_82AA2240);
extern "C" PPC_FUNC(sub_82AA2240) {
  if (t_crowd.in_block && ctx.r4.u32 == 76 && ctx.r6.u32 == 3) {
    uint32_t m[12];
    for (uint32_t i = 0; i < 12; ++i) m[i] = PPC_LOAD_U32(ctx.r5.u32 + 4 * i);
    if (!t_crowd.have_first) {
      std::memcpy(t_crowd.first, m, sizeof(m));
      t_crowd.have_first = true;
    } else {
      t_crowd.others.insert(t_crowd.others.end(), m, m + 12);
      t_crowd.skip_draw = true;
      return;
    }
  }
  __imp__sub_82AA2240(ctx, base);
}

// Record a draw: a collected member's draw is not recorded.
extern "C" PPC_FUNC(__imp__sub_82AA0B28);
extern "C" PPC_FUNC(sub_82AA0B28) {
  if (t_crowd.in_block && t_crowd.skip_draw) {
    t_crowd.skip_draw = false;
    return;
  }
  __imp__sub_82AA0B28(ctx, base);
}

extern "C" PPC_FUNC(__imp__sub_82499190);
extern "C" PPC_FUNC(sub_82499190) {
  if (CrowdHidden()) return;
  if (CrowdBatch() && !wwe13_crowd_blocked() && !t_crowd.in_block) {  // the card loop: the same per-member pattern
    t_crowd.in_block = true;
    t_crowd.have_first = false;
    t_crowd.others.clear();
    __imp__sub_82499190(ctx, base);
    t_crowd.Flush();
    t_crowd.in_block = false;
    return;
  }
  __imp__sub_82499190(ctx, base);
}
