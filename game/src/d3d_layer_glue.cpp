// WWE '13: 2K14's D3D layer (Half 2 + layer F1-F3 + no-ring), ported from Jarrodfeng/wwe2k14-recomp (BSD) into src/k14.
// WWE '13 links the same Xbox D3D library: code addresses are 2K14's minus 0x1184E0 (instruction streams compared:
// 0x82CA4000-0x82CD1000 match 90-100%), data addresses mapped from the library's own references (tools13/dmap.py).
// Every switch is 2K14's (wwe2k14_d3d_layer / _copy / _inline / _noring); all default off.
#include "rex_compat.h"
#include <sched.h>
namespace wwe13_perf {
void AddWait(int64_t ns);
void OnSwap();
int AbPhase(const char* what);
}
#include <chrono>
#include <thread>
#include <cstdlib>
#include <rex/logging.h>
#include <rex/cvar.h>
#include <rex/perf/timeline.h>
#include <rex/runtime.h>
#include <rex/system/xtypes.h>
#include <rex/kernel/xboxkrnl/video.h>

REXCVAR_DEFINE_BOOL(wwe2k14_rdr_readback_defer, false, "wwe2k14", "(2K14 adapter reader; not in the WWE '13 port)");
REXCVAR_DEFINE_INT32(wwe2k14_d3d_textures, 0, "wwe2k14", "(2K14 stage 4 texture objects; not in the WWE '13 port)");
REXCVAR_DEFINE_BOOL(wwe2k14_rdr_owner, false, "wwe2k14", "(2K14 owner mode; not in the WWE '13 port)");

#include "k14/half2_hooks.h"
#include "k14/owner_timing.h"
#include "k14/d3d_textures.h"

// VdSwap: the guest frame counter the layer reads, and the no-ring consumer's swap words (2K14 kernel_overrides.h).
namespace rex::kernel::xboxkrnl {
void VdSwap_entry(mapped_void buffer_ptr, mapped_void fetch_ptr, mapped_void unk2, mapped_void unk3,
                  mapped_void unk4, mapped_u32 frontbuffer_ptr, mapped_u32 texture_format_ptr,
                  mapped_u32 color_space_ptr, mapped_u32 width, mapped_u32 height);
}
namespace wwe13 {
inline void VdSwapOverride(mapped_void buffer_ptr, mapped_void fetch_ptr, mapped_void unk2,
                           mapped_void unk3, mapped_void unk4, mapped_u32 frontbuffer_ptr,
                           mapped_u32 texture_format_ptr, mapped_u32 color_space_ptr, mapped_u32 width,
                           mapped_u32 height) {
  wwe2k14::g_guest_frame.fetch_add(1, std::memory_order_release);
  rex::perf::timeline::Record("vdswap");
  wwe13_perf::OnSwap();
  // WWE13_PIN_RENDER=<cpu list, e.g. 7>: the presenting (render) thread is pinned there once (experiment).
  static thread_local bool pin_done = false;
  if (!pin_done) {
    pin_done = true;
    if (const char* v = std::getenv("WWE13_PIN_RENDER"); v && *v) {
      cpu_set_t set;
      CPU_ZERO(&set);
      for (const char* c = v; *c; ++c) {
        if (*c >= '0' && *c <= '9') CPU_SET(*c - '0', &set);
      }
      const int r = sched_setaffinity(0, sizeof(set), &set);
      REXLOG_INFO("[wwe13-pin] render thread pinned to {} ({})", v, r);
    }
  }
  rex::kernel::xboxkrnl::VdSwap_entry(buffer_ptr, fetch_ptr, unk2, unk3, unk4, frontbuffer_ptr, texture_format_ptr,
                                      color_space_ptr, width, height);
  wwe2k14::d3d_noring::AfterVdSwap(REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(0), buffer_ptr.guest_address());
}
}  // namespace wwe13
REX_HOOK(__imp__VdSwap, wwe13::VdSwapOverride)

// One more frame in flight (WWE13_FRAME_QUEUE=2): the D3D present (sub_82B9B2E0) blocks on the fence it inserted at the
// previous present ([device+15344], BlockOnFence sub_82B9D540 at 0x82B9B9C4), so the game cannot build frame N+1 until
// the command processor has finished frame N. On the 360 the GPU front end is quick; here the command thread needs
// ~10 ms a frame, which serialises with the game's own ~10 ms. With 2 the present waits for the fence of the frame
// before the previous one instead.
REX_EXTERN(__imp__sub_82B9D540);
REX_HOOK_RAW(sub_82B9D540) {
  static const int depth = [] {
    const char* v = std::getenv("WWE13_FRAME_QUEUE");
    return v ? std::atoi(v) : 1;
  }();
  rex::perf::timeline::Scope wwe13_tl_bof("bof_b", "bof_e", uint32_t(ctx.lr), ctx.r4.u32);
  if (depth >= 2 && uint32_t(ctx.lr) == 0x82B9B9C8u) {
    static uint32_t older = 0;
    const uint32_t previous = ctx.r4.u32;  // the previous present's fence
    ctx.r4.u64 = older;                    // wait for the one before it (0 = no wait)
    older = previous;
  }
  __imp__sub_82B9D540(ctx, base);
}

// The D3D library's wait loops (BlockOnFence, the present's flip wait, ...) call this one iteration body (sub_82B99A70:
// a short delay loop + the GPU-hang checks; r3 != 0 = keep waiting) back to back: the render thread burned ~40% of its
// CPU time spinning here (profile: __imp__sub_82B9D540 24%, __imp__sub_82B99A70 18%) while it waited for the command
// thread. A short sleep per iteration frees the core (less heat, more room for the command thread).
// WWE13_SPIN_SLEEP_US=<us> (default 40; 0 = spin as the original). In-run A/B (WWE13_AB_PERIOD_S): same fps either way
// (58.55 vs 58.47, 58.03 vs 57.94 over ~9500 frames each). (Sleeping in the command thread's WAIT_REG_MEM polls too did
// cost: -0.36 fps, 23% more late frames - those stay yield-spins.)
REX_EXTERN(__imp__sub_82B99A70);
REX_HOOK_RAW(sub_82B99A70) {
  static const int sleep_us = [] {
    const char* v = std::getenv("WWE13_SPIN_SLEEP_US");
    return v ? std::atoi(v) : 40;
  }();
  const auto t0 = std::chrono::steady_clock::now();
  __imp__sub_82B99A70(ctx, base);
  // (with WWE13_AB_PERIOD_S: A = spin, B = sleep WWE13_SPIN_SLEEP_US or 40)
  const int ab = wwe13_perf::AbPhase("spin");
  const int us = ab < 0 ? sleep_us : ab == 1 ? (sleep_us > 0 ? sleep_us : 40) : 0;
  if (us > 0 && ctx.r3.u32 != 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  }
  wwe13_perf::AddWait(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

// For crowd_options.cpp (the crowd runs are pushed to the SDK's command processor).
void wwe13_crowd_push(const uint32_t* first, const uint32_t* others, uint32_t count) {
  if (auto* cp = wwe2k14::half2::Cp()) {
    if (!cp->scene_crowd_blocked()) cp->SceneCrowdPush(first, others, count, false);
  }
}
bool wwe13_crowd_blocked() {
  auto* cp = wwe2k14::half2::Cp();
  return !cp || cp->scene_crowd_blocked();
}
