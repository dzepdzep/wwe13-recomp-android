/**
 * wwe2k14 patch (Session 15, determinism): deterministic frames mode (det_frames, default off).
 *
 * Why: a guest advances frames on the vblank, while its other threads (loaders, decoders, workers) and the GPU
 * finish their work in host wall time. How much work lands in which frame then depends on host timing, so a title
 * that does anything per frame (RNG draws, animation steps, staged setup) runs differently every time, and more so
 * the faster the frames go. With det_frames on:
 *   - quiescent vblank: the vblank thread fires the next vblank only once every guest thread is blocked (in a kernel
 *     wait, a sleep, or a spin wait the title marks with WaitScope) and the command processor is idle or waiting,
 *     and that state has held for det_frames_settle_us; so each frame contains all the work its inputs allow;
 *   - virtual time: guest timebase / system time reads return a time derived from the det vblank count (60 Hz) plus
 *     a tiny per-thread step per read, so time-budgeted guest code sees the same time every run;
 *   - timed guest waits and sleeps end at vblank boundaries (their length counted in det vblanks), not in host time.
 * Threads are counted only while they are guest threads (XThread); host threads never block a vblank.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <thread>

namespace rex::thread {
class WaitHandle;
}

namespace rex::det {

bool Enabled();

// Guest thread lifetime (XThread::Execute).
void GuestThreadStarted();
void GuestThreadExiting();
bool IsGuestThread();

// wwe2k14 patch (Session 77, cross-device tester check, XDET_PLAN P7 / gap 11): det_time_constant (with det_frames)
// - the virtual clocks start from constant bases instead of the host's uptime and date (timebase 1 s, uptime 1000 ms,
// system time 2013-10-29 00:00 UTC = the game's release), so no tester's clock or date reaches the guest; and guest
// code the SDK calls on a host thread (interrupt callbacks, the audio worker's client callbacks: GuestCallbackScope)
// reads the virtual clocks too. Off: unchanged.
bool TimeConstant();
uint64_t ConstantSystemTimeBase();
// A guest thread, or (det_time_constant) guest code running inside a GuestCallbackScope on a host thread.
bool ReadsVirtualTime();
struct GuestCallbackScope {
  GuestCallbackScope();
  ~GuestCallbackScope();
  GuestCallbackScope(const GuestCallbackScope&) = delete;
  GuestCallbackScope& operator=(const GuestCallbackScope&) = delete;
};
// det_fs_constant (gap 10, no det_frames needed): the file system answers what a folder dump and a disc image of the
// same disc, and any machine's save folder, would otherwise answer differently with constants: file times
// kConstantFileTime (1970-01-01, what a disc image already reports), a disc image's attributes and volume size as a
// read-only folder's. Off: unchanged.
bool FsConstant();
inline constexpr uint64_t kConstantFileTime = 116444736000000000ull;

// A guest thread blocks (no effect on host threads or with det_frames off).
void EnterWait();
void LeaveWait();
struct WaitScope {
  WaitScope() { EnterWait(); }
  ~WaitScope() { LeaveWait(); }
  WaitScope(const WaitScope&) = delete;
  WaitScope& operator=(const WaitScope&) = delete;
};

// Command processor: idle (no commands) or blocked in a WAIT_REG_MEM poll.
void SetCpWaiting(bool waiting);

// Det vblank count and waiting on it (the caller counts as blocked while waiting).
uint64_t Vblanks();
void NoteVblank();
void WaitVblanks(uint64_t target);
// Det vblanks for a guest timeout in milliseconds (at least 1).
uint64_t VblanksForMs(uint64_t ms);
// wwe2k14 patch (Session 77, cross-device): the vblank thread counts every vblank that fired on a timeout (det_serial's
// det_serial_vblank_timeout_ms, det_frames' quiescence timeout), so the det hash CSV can mark it even when the log
// line is filtered or cut. And the effective renderer (what the backend actually chose after its feature checks, e.g.
// "vulkan rt=host" when the device has no fragment shader interlock), set by the backend, read by the CSV writer.
void NoteVblankTimeout();
uint64_t VblankTimeouts();
void SetRendererInfo(const std::string& info);
std::string RendererInfo();
// wwe2k14 patch (Session 27.5): the virtual time rate (det_frames_time_hz as applied: 60 until it first changes) and
// the virtual time since the first det vblank in microseconds (for host code that must follow guest time, e.g. a
// video decoder standing in for the guest's own).
int32_t TimeHz();
uint64_t VirtualMicros();

// Vblank thread: blocks until the guest is quiescent (as above) for settle_us, or timeout_ms passed (returns false).
// 0 = the det_frames_settle_us / det_frames_timeout_ms settings.
bool WaitQuiescent(uint32_t settle_us, uint32_t timeout_ms);
int32_t RunningGuestThreads();
// Host thread ids of the running guest threads, and the CP state (for the timeout log).
std::string DescribeRunning();

// Frame barrier: the guest's frame thread calls FrameBarrier() at each present (VdSwap); it counts as blocked and
// returns when the vblank thread, at the next quiescent point, calls ReleaseBarrier(). At that point the vblank thread
// fires a vblank unless one already fired since the previous barrier (VblankDueAtBarrier() == false), so time moves
// at least one 60 Hz step per frame even when the guest does not pace on the vblank.
void FrameBarrier();
bool BarrierWaiting();
bool VblankDueAtBarrier();
void ReleaseBarrier();

// Virtual guest time (det_frames on, guest thread): timebase ticks and 100 ns system-time offset.
uint64_t VirtualGuestTicks(uint64_t real_ticks, uint64_t guest_tick_frequency);
uint64_t VirtualSystemTimeOffset(uint64_t real_offset);
uint32_t VirtualUptimeMillis(uint32_t real_ms);
// Work that must happen exactly at det vblanks (the vblank thread, at idle, right after NoteVblank): e.g. the
// kernel's KeTimeStampBundle, which a host timer otherwise rewrites every millisecond.
void AddVblankHook(std::function<void()> hook);
void RunVblankHooks();

// Command processor state as SetCpWaiting left it.
bool CpWaiting();

// det_serial (needs det_frames): a deterministic cooperative scheduler. One guest thread runs at a time; it gives
// the baton up only when it blocks (SerialBlock), and the next ready thread is picked round-robin in creation order.
// A blocked thread is ready when its deadline (det vblank) has come or its ready() test says so (evaluated by the
// scheduler; must not consume anything: use HandleSignaled). Before every decision the command processor is drained
// and pending external requests (host threads that must run guest code: GPU interrupts) run one at a time.
bool SerialEnabled();
// guest = false for host workers that take part (the audio worker): they never take the first turn, and vblanks
// start only once a guest thread exists.
int SerialRegisterThread(bool runnable, bool guest = true);  // at XThread creation (the creator's turn): a slot in creation order
void SerialMarkRunnable(int slot);        // a thread created suspended is resumed
void SerialThreadStart(int slot);  // at XThread::Execute: wait for the first turn
void SerialThreadExit();
bool SerialIsGuestThread();
void SerialBlock(std::function<bool()> ready, uint64_t deadline_vblank);
// cp_request: the command processor asks: it counts as waiting from the moment the request is queued (atomically,
// under the scheduler lock) until fn has run (cleared before the scheduler continues).
void SerialRunExternal(const std::function<void()>& fn, bool cp_request = false);
bool SerialWaitIdle(uint32_t timeout_ms);  // vblank thread: until nothing is runnable (true) or timeout
void SerialAfterVblank();                  // vblank thread: schedule after the vblank work
void NoteCpProgress();
// Host workers that change guest memory on their own register a "drained" test (checked before every decision).
// wwe2k14 patch (Session 77): returns an id for RemoveDrainCheck (a worker that shuts down must unregister: a stale
// check would keep the scheduler waiting forever, or call into a destroyed object).
int AddDrainCheck(std::function<bool()> drained);
void RemoveDrainCheck(int id);
// wwe2k14 patch (Session 77): a slot's thread has given its slot up (SerialThreadExit / SerialThreadKilled). The
// Locked form is for ready() predicates (the scheduler calls them holding its lock).
bool SerialSlotExited(int slot);
bool SerialSlotExitedLocked(int slot);
// wwe2k14 patch (Session 77): another thread terminated this slot's thread (it never returns from SerialBlock):
// retire the slot (and pass the baton on if it held it).
void SerialThreadKilled(int slot);
// wwe2k14 patch (Session 77): with det_serial_trace_detail, the scheduler trace notes before each thread pick what
// the command processor was waiting on (the function the CP registers: e.g. "w1:00001A40" = an idle ring at that read
// pointer, "w2:ADDR=VALUE" = WAIT_REG_MEM on memory), and TraceTag text (the CP tags each interrupt request with its
// ring position) - to find which wait lets host timing decide where GPU interrupts run.
void SetCpDescribe(std::function<std::string()> describe);
void TraceTag(const std::string& tag);
bool TraceDetail();
// Audio under det_serial: the worker waits for pumps (AudioWaitPump), runs its client callbacks once per pump and
// reports AudioPumpDone; the vblank thread asks for this vblank's pumps at idle and waits for them.
void AudioSetWorker(bool present);
void AudioAddPumps(uint64_t n);
bool AudioWaitPump(uint32_t timeout_ms);
void AudioPumpDone();
void RunAudioForVblank();   // vblank thread: add this vblank's pumps (the worker's slot becomes ready)
bool AudioPumpsPending();
uint64_t CpProgress();
// Signalled now, without consuming it (Windows: NtQueryEvent / NtQuerySemaphore / NtQueryMutant / NtQueryTimer).
bool HandleSignaled(void* native_handle);
// wwe2k14 patch (Session 77): the same for a wait by thread `waiter`, on every platform: Windows forwards to the
// native-handle version; POSIX asks the SDK's own wait object (rex::thread::PeekSignaled; it has no kernel object to
// query, and a mutant must be judged for the waiter, not for the scheduler thread asking).
bool HandleSignaled(rex::thread::WaitHandle* wait_handle, std::thread::id waiter);

}  // namespace rex::det
