// wwe2k14 patch (Session 48, HALF2): the command processor's side of the game's library draws (half2.cpp).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <rex/graphics/command_processor.h>
#include <rex/graphics/register_file.h>

namespace rex::graphics {

struct CommandProcessor::Half2State {
  // The queue of library draws, in the order the game's writers ran: a ring the game reserves at its tail (under the
  // mutex) and the command processor consumes from its head.
  static constexpr uint32_t kSlots = 8192;
  std::unique_ptr<Half2LibraryDraw[]> slots{new Half2LibraryDraw[kSlots]};
  std::mutex producer_mutex;
  std::atomic<uint64_t> tail{0};  // next slot the game reserves
  uint64_t head = 0;              // next slot the command processor looks at (its thread only)
  std::atomic<uint64_t> head_published{0};  // head as the producer may read it (Session 50: without the mutex)
  std::atomic<uint64_t> dropped{0};  // reservations refused (the queue was full)
  // The last matched draw (a tiling replay of its packets compares against it again).
  Half2LibraryDraw last;
  bool last_valid = false;
  // Per-frame census (command processor thread).
  struct Census {
    uint32_t cp_draws = 0;          // draw packets executed
    uint32_t matched = 0;           // of those, attributed to a library draw
    uint32_t replays = 0;           // of those, a second execution of the last matched draw's packets
    uint32_t via_ib = 0;            // of those, inside an indirect buffer a library draw's packets called
    uint32_t unmatched_ib = 0;      // draw packets no library draw wrote, inside an indirect buffer
    uint32_t unmatched_ring = 0;    // the same, in the ring
    uint32_t library_skipped = 0;   // library draws passed over with no draw packet matched
    uint32_t library_pending = 0;   // library draws queued, not yet reached, at the frame boundary
    uint32_t skipped_packets = 0;   // draw packets the bin test skipped (their library draw consumed)
    uint32_t stale = 0;             // library draws in an executed segment with no draw packet matched (dropped)
    uint32_t waits = 0;             // a draw packet waited for the game to finish its entry
    uint32_t wait_timeouts = 0;
    uint32_t mismatched_draws = 0;  // matched draws with at least one register different
    uint64_t mismatched_values = 0;
    uint32_t inline_vs_same = 0;    // inline-loaded vertex shaders found equal to the object's microcode
    uint32_t fields_compared = 0;   // draw-description fields compared (indexed draws)
    uint32_t unanalysed = 0;        // matched draws without both shaders analysed (compared in full)
    uint64_t unused_values = 0;     // differences in constants the bound shaders do not read
    uint32_t owned_writes = 0;      // packets writing the float constants the shadow lacks
    uint32_t owned_writes_in_draws = 0;  // of those, inside the last matched library draw's packets
    uint32_t kind_matched[5] = {};  // by kind: 1 indexed, 2 inline, 3 clear, 4 resolve
  };
  Census frame;
  Census window;  // summed over gpu_native_half2_log_frames frames
  uint32_t window_frames = 0;
  // Registers that differed (register -> count), over the window; float groups as 0x4000 + 16 x group.
  std::unordered_map<uint32_t, uint32_t> mismatch_registers;
  std::unordered_map<uint32_t, uint32_t> unused_registers;  // differences the shaders do not read
  // Draw packets no library draw wrote, by 64 KB region of guest physical memory (address >> 16), over the window.
  std::unordered_map<uint32_t, uint32_t> unmatched_regions;
  uint32_t unmatched_logged = 0;
  uint32_t owned_logged = 0;
  uint32_t inline_logged = 0;
  uint32_t mismatches_logged = 0;
  std::unordered_map<uint32_t, uint32_t> logged_per_register;  // at most 3 logged lines a register
  std::atomic<uint32_t> frames_seen{0};  // read by the game's threads (queued_frame)
};

// wwe2k14 patch (Session 51, STATE LOG, half2_log.cpp): each library call's own packet bytes, in call order. A ring of
// words: records of a 6-word header (kind, seq low / high, pieces, payload words, flags), the pieces (guest physical
// begin, words - more than one when the library kicked the segment inside the call) and the payload (the packets as
// the library wrote them, guest byte order). One producer at a time (the mutex), the command
// processor thread consumes.
struct CommandProcessor::Half2Log {
  static constexpr uint32_t kWords = 1u << 22;  // 16 MB
  static constexpr uint32_t kHeaderWords = 6;
  static constexpr uint32_t kMaxPieces = 8;  // a call's packets in up to 8 ranges (segment kicks inside the call)
  static constexpr uint32_t kFlagSplit = 1;  // the call's packets are not one range: no payload, the state is lost
  static constexpr uint32_t kFlagContinues = 2;  // the call goes on in the next record (a kick inside it)
  std::unique_ptr<uint32_t[]> ring{new uint32_t[kWords]};
  std::mutex mutex;
  std::atomic<uint64_t> write{0};          // words published
  std::atomic<uint64_t> read_published{0};  // words consumed (the producer refuses when the ring would overflow)
  std::atomic<uint64_t> next_seq{1};
  std::atomic<uint64_t> lost{0};  // calls whose packets are not in the log (refused reservations, full ring, splits)
  // Producer census (atomics: the game's threads), read and reset at the command processor's frame boundary.
  std::atomic<uint64_t> records{0}, payload_words{0}, nested{0}, splits{0}, ring_full{0}, copy_ns{0};
  std::atomic<uint64_t> kicks{0}, multi_piece{0};
  std::atomic<uint32_t> split_logged{0};
  std::atomic<uint64_t> gap_records{0}, gap_words{0}, kicks_outside{0}, jumps{0};

  // Consumer (command processor thread).
  uint64_t read = 0;
  uint64_t last_seq = 0;      // the last record opened (review 1: a hole after it = a loss)
  bool was_logging = false;   // (review 2) logging at the last frame boundary
  bool seeded = false;
  bool open = false;          // a record is open (its header read)
  uint64_t cur_seq = 0;
  uint32_t cur_kind = 0, cur_begin = 0, cur_words = 0, cur_pos = 0;
  uint32_t cur_pieces = 0, cur_piece_begin[kMaxPieces] = {}, cur_piece_words[kMaxPieces] = {};
  uint32_t cur_flags = 0;
  // The segment executor (gpu_native_detached 1, command processor thread).
  struct Exec {
    uint64_t segments = 0, log_words = 0, memory_head = 0, memory_between = 0, memory_tail = 0, records = 0;
    uint64_t records_by_kind[6] = {}, split_records = 0, stale_records = 0, waits = 0, wait_timeouts = 0;
    uint64_t not_logged_segments = 0, failed = 0, exec_ns = 0;
    // Step 3 (gpu_native_detached 2).
    uint64_t detached_frames = 0, self_records = 0, self_words = 0, self_ns = 0, skipped_words = 0, fallbacks[4] = {};
    uint64_t swaps = 0, skip_missed = 0;
  };
  Exec exec;
  std::vector<uint32_t> scratch;  // a record's words, contiguous, for the packet executor
  uint32_t blocked_segments = 0;  // segments in a row the next record did not belong to
  // Step 3: the no-GPU mode (set by the game at a present), a switch back's drain and skip, the fallback hysteresis.
  std::atomic<bool> detached{false};
  std::atomic<uint64_t> drain_until_seq{0};  // after a switch back: records below it ran / run from the log
  std::atomic<uint32_t> skip_until{0};       // the write position at the switch back (guest physical)
  std::atomic<uint32_t> fallback_frames{0};  // > 0: the game keeps (turns) the mode off
  std::atomic<uint64_t> switches_on{0}, switches_off{0}, ring_waits_full{0};
  uint64_t self_last_seq = 0;
  uint64_t swaps_seen = 0;
  int32_t last_detached_mode = 0;
  uint64_t pace_mark = 0, pace_mark_older = 0;  // (game's render thread) the log's write position at the last present while detached
  std::atomic<uint64_t> pace_us{0}, pace_timeouts{0};
  std::atomic<uint32_t> jump_logged{0};
  uint32_t fallback_logged = 0;
  uint64_t cur_payload = 0;   // ring index of the open record's first payload word
  std::unique_ptr<uint32_t[]> values{new uint32_t[RegisterFile::kRegisterCount]};
  std::unique_ptr<uint64_t[]> touched{new uint64_t[(RegisterFile::kRegisterCount + 63) / 64]};
  uint64_t bin_select = 0, bin_mask = 0;
  uint32_t ucode[2] = {};     // IM_LOAD microcode (guest physical, 0 = inline)
  uint32_t inline_vs_at = 0;  // the last IM_LOAD_IMMEDIATE vertex shader's packet (guest physical)
  uint32_t shader_at[2] = {};  // the last VS / PS load packet (diagnostic)
  struct Census {
    uint64_t applied_packets = 0, applied_words = 0, predicated_skipped = 0, ib_packets = 0, ib_followed = 0;
    uint64_t ib_not_followed = 0;
    uint32_t at_calls = 0, compared = 0, via_ib = 0, not_ready = 0, seq_gaps = 0, out_of_range = 0, behind = 0;
    uint32_t seeds = 0, differing_draws = 0, shader_differs = 0, seq_holes = 0;
    uint64_t differing_values = 0, compared_values = 0, apply_ns = 0;
  };
  Census window;
  uint32_t window_frames = 0;
  std::array<uint32_t, 128> opcodes{};                  // type-3 opcodes applied, over the window
  std::unordered_map<uint32_t, uint32_t> diff_registers;  // register -> draws it differed in, over the window
  std::unordered_map<uint32_t, uint32_t> diff_logged;
  uint32_t lines_logged = 0;
  uint32_t dumps = 0;  // diagnostic record dumps (once per run)
  uint32_t kind2_dumps = 0;
  uint32_t type1_dumps = 0;
  uint32_t ib_logged = 0;
  std::unordered_set<uint32_t> ib_seen;
};

}  // namespace rex::graphics
