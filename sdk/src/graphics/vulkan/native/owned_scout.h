// wwe2k14 patch (Session 95, item N2; out\s95\read_N.md section 3): the SCOUT and the result slots.
//
// A scout thread reads the host queue AHEAD of the command processor on a private copy of the decoder's typed state (OwnedRecordState: pure stores, no
// notifications, no guest stores, no LoadShader), and for the draws whose cover rectangle / vertex bounds the claim will want it snapshots the state and
// queues a pure job on the pool (job_pool.h). The job writes its result into a keyed slot ring. The command processor, at the SAME point of its own
// path (DrawFromBase), asks "is the slot ready and does its key + fingerprint match this draw?": yes = it uses the result, no = it computes inline as it
// always did. IT NEVER WAITS. A late job, a thin lookahead, 0 workers, a stale state: all are the existing code path.
//
// Latch: gpu_jobs bit 2 (value 2) with the pool at >= 1 worker; bit 1 (value 1) makes the pool alone. gpu_jobs_workers (live) sets the active worker count
// (0 = from the process affinity mask). gpu_jobs_verify: the command processor computes inline as well and compares (the inline result is used).
// Off (gpu_jobs 0) = no thread is started and every hook below returns at its first line.
#pragma once

#include <cstdint>

namespace rex::memory {
class Memory;
}

namespace rex::graphics {
struct D3dNoringShared;
class Shader;
}  // namespace rex::graphics

namespace rex::graphics::vulkan {

struct OwnedRecordState;

namespace scout {

// Command thread. True while the hooks are live (gpu_jobs != 0 and the cut is latched).
extern bool g_on;

// d3d_noring.cpp (through OwnedRecordHooks): around the drain loop, per packet / register item (head = the item's start in dwords), per present.
void DrainEnter(D3dNoringShared* queue, rex::memory::Memory* memory);
void DrainLeave();
void BeginItem(uint32_t head);
void NotePresent(D3dNoringShared* queue, rex::memory::Memory* memory);
// Guest memory may have changed under a job (an interrupt item ran, a read-back / write-back wrote it): results made before this are stale.
void NoteGuestWrite();
// ShutdownContext: stop the scout and the jobs touching the queue / guest memory (bounded wait, ~100 ms).
void Quiesce();

// owned_record.cpp DecodeToken: a token DRAW item is about to be handed to the draw entry; the key names it. ClearKey after the entry returned.
void DrawItem();
void ClearKey();
uint64_t CurrentKey();

// DrawFromBase (owned_front_own.cpp): one per typed draw. Inactive (every method returns "miss" at once) when the hooks are off, there is no state, or the
// draw is a crowd member / inline draw. Methods are called in the order Note, Cover, VerifyCover, Bounds, VerifyBounds.
class Lookup {
 public:
  Lookup(const OwnedRecordState* s, bool crowd_member, const Shader& vs, const Shader* ps);
  bool active() const { return active_; }
  // What the claim asked for, from its own tests: the scout wants it for the next draws of this shader pair.
  void Note(bool owned_cover, bool truth_wants_cover, bool bounds_vs);
  // Cover rectangle: true = answered by a job (ok / rect / reject line, the inline function's own meaning); false = compute inline.
  bool Cover(bool& ok, int32_t rect[4], int32_t& reject_line);
  void VerifyCover(bool ok, const int32_t rect[4]);
  // Vertex bounds: -1 = compute inline; else the outcome (0 made, 2 refused; rect zero when refused).
  int Bounds(int32_t rect[4]);
  void VerifyBounds(bool ok, const int32_t rect[4]);

 private:
  bool Fetch();
  const OwnedRecordState* s_;
  const Shader* vs_;
  const Shader* ps_;
  bool active_ = false, fetched_ = false, hit_ = false;
  bool cover_job_ = false, cover_ok_ = false, bounds_job_ = false, bounds_ok_ = false;
  int32_t cover_rect_[4] = {}, bounds_rect_[4] = {};
  int32_t cover_line_ = 0;
  int bounds_outcome_ = 2;
};

}  // namespace scout
}  // namespace rex::graphics::vulkan
