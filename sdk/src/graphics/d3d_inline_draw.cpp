// wwe2k14 patch (Session 87, D3D LAYER F3): executes the game's inline-vertex draw records (see
// include\rex\graphics\d3d_inline_draw.h). New file; the only edits elsewhere are the three declarations in
// command_processor.h, the opcode constant and the dispatch case in command_processor.cpp, and this file's CMake line.
#include <algorithm>
#include <atomic>
#include <rex/graphics/d3d_tex.h>
#include <cstring>
#include <bit>
#include <memory>
#include <mutex>
#include <vector>

#include <rex/cvar.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/d3d_inline_draw.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/memory/ring_buffer.h>
#include <rex/memory/utils.h>
#include <rex/system/xmemory.h>
#include "d3d_shader_tag.h"
#include "vulkan/native/owned_record.h"  // (Session 92, M4e)

REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);  // (d3d_noring.cpp) bit 4 = M4e

// The capability marker the game exe looks up by name (rex::cvar::GetFlagInfo) before it takes an inline draw: a build
// without it never gets a token. Read-only.
REXCVAR_DEFINE_INT32(gpu_d3d_inline_abi, 1, "GPU/Native",
                     "Session 87 (D3D layer F3): the inline-draw token ABI this build executes (opcode 0x7E, payload magic "
                     "0xD3D30001 + sequence + the record inline). Read-only marker, do not set")
    .range(1, 1);

// wwe2k14 patch (Session 93, idea 5 piece 5): inline vertex bytes on pages of their own.
REXCVAR_DEFINE_BOOL(gpu_d3d_inline_host_ring, false, "GPU/Native",
                    "Session 93 (idea 5): an inline draw's carried vertex bytes are staged at the start of a host page of their own (the next "
                    "draw on the next page), so the shared memory upload of each draw finds its pages untouched by any command recorded "
                    "earlier in the submission and goes into the hoist stream (vulkan_upload_hoist 2: no per-draw barrier) instead of in "
                    "place. The staging bytes, the shaders and the bindings are unchanged. [d3d][inline] ring counters. false = packed "
                    "16-byte staging as before (default). Live")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {

namespace {

using d3d::InlineDrawRecord;

// Vertex bytes of records that carry them are put here before their draw: guest physical memory of ours (a ring; the
// shared memory uploads from it exactly as from the game's own memory, and a later overwrite of a spot is the CPU write
// the write watches already handle).
struct InlineStaging {
  uint32_t virtual_base = 0;
  uint32_t size = 0;
  uint32_t offset = 0;
  bool failed = false;
};
InlineStaging g_staging;
constexpr uint32_t kStagingBytes = 1u << 20;

// The census (command thread only): one line per gpu_d3d_layer_log_frames-like window; here a fixed 600 draws' worth of
// frames is not known, so it is kept per call count and printed on the swap by D3dInlineCensus().
struct InlineCensus {
  std::atomic<uint64_t> draws{0}, replays{0}, vertices{0}, vertex_bytes{0}, runs{0}, registers{0}, shaders_set{0},
      staged{0}, guest_memory{0}, bad{0}, tiled{0}, ring_draws{0}, ring_wraps{0}, ring_pages{0};
  std::atomic<uint32_t> last_sequence{0};
  std::atomic<uint32_t> window_start{0};
};
InlineCensus g_census;

// (Session 92, M4f-B) Where the record's vertex bytes are: the game's own allocation, or ours (copied into the staging ring). One body: the old path calls it
// from ExecuteD3dInlineDraw; the decoder's executor will (M4f-C). Returns the guest physical address of the vertices; `ok` false = the draw is dropped
// (counted bad).
uint32_t InlineStage(memory::Memory* memory, const InlineDrawRecord& rec, bool& ok) {
  InlineCensus& c = g_census;
  ok = true;
  const uint32_t bytes = rec.vertex_count * rec.stride_bytes;
  if ((rec.flags & d3d::kInlineUseGuestVertexMemory) && rec.guest_vertex_address) {
    c.guest_memory.fetch_add(1, std::memory_order_relaxed);
    return rec.guest_vertex_address;  // (already the physical form the library forms)
  }
  InlineStaging& s = g_staging;
  if (!s.virtual_base && !s.failed) {
    // (Session 90: `virtual_base` now holds a PHYSICAL address, out of the SDK's own system reserve - not the game's pool.)
    uint32_t address = 0;
    if (memory::D3dLayerTakePhysicalReserve(kStagingBytes, &address)) {
      s.virtual_base = address;
      s.size = kStagingBytes;
      memory::D3dLayerNoteGuestMemory("inline vertex staging", kStagingBytes, false, address);
    } else {
      s.failed = true;
      REXGPU_ERROR("[d3d] inline vertex staging could not be allocated: inline draws with carried bytes are dropped");
    }
  }
  if (!s.virtual_base || rec.vertices.size() < bytes || bytes > s.size) {
    c.bad.fetch_add(1, std::memory_order_relaxed);
    ok = false;
    return 0;
  }
  uint32_t use = (s.offset + 15u) & ~15u;
  if (REXCVAR_GET(gpu_d3d_inline_host_ring)) {
    // (Session 93) The draw's bytes start a host page: the pages of an earlier draw (still in the submission) are never touched again, so this
    // draw's upload is hoistable. Ring wraps are counted: past the end the pages of the oldest draws come round again (their uploads then stay in place).
    const uint32_t page = std::max<uint32_t>(4096u, uint32_t(memory::page_size()));
    const auto align_page = [&](uint32_t offset) { return ((s.virtual_base + offset + page - 1) & ~(page - 1)) - s.virtual_base; };
    use = align_page(s.offset);
    if (use + bytes > s.size) {
      use = align_page(0);
      c.ring_wraps.fetch_add(1, std::memory_order_relaxed);
    }
    if (use + bytes > s.size) use = 0;  // (a ring smaller than a page and a draw: as before)
    c.ring_draws.fetch_add(1, std::memory_order_relaxed);
    c.ring_pages.fetch_add((bytes + page - 1) / page, std::memory_order_relaxed);
  } else if (use + bytes > s.size) {
    use = 0;  // wrap: an older draw's bytes are long consumed (its upload was recorded in order)
  }
  s.offset = use + bytes;
  uint8_t* host = memory->TranslateVirtual<uint8_t*>(0xA0000000u + s.virtual_base + use)  /* the 64 KB view: a WATCHED view, so the write invalidates the shared memory (the physical view is not watched) */;
  std::memcpy(host, rec.vertices.data(), bytes);
  c.staged.fetch_add(1, std::memory_order_relaxed);
  return s.virtual_base + use;
}

}  // namespace

// (Session 92, M4f-B) d3d_direct.cpp: the inline draw through the typed draw; false = the old path draws it (counted there by reason).
namespace vulkan {
bool OwnedFrontInlineDraw(CommandProcessor& base, uint32_t viz_query, uint32_t initiator, bool& drawn);
}

// The token (integrated by F1): the record travels INLINE in the guest command memory (so the tiling worker's per-strip
// replays of a recorded segment find it; nothing is stored host side and the exe calls no SDK function). Payload, big-endian
// words as every packet: [0] magic 0xD3D30001, [1] sequence, [2] flags, [3] primitive, [4] vertex count, [5] stride bytes,
// [6] viz word, [7-10] slot 31 words 0-3, [11] runs, [12] values, [13] vs address, [14] vs dwords, [15] vs inline dwords,
// [16-18] the same for the pixel shader, [19] guest vertex address, [20] vertex bytes; then 3 words per run (first register,
// count, value offset), the values, the vs and ps inline words, the vertex bytes padded to 4. The inline words and the vertex
// bytes are raw guest memory bytes: stored as the bytes are (read back swapped twice).
bool CommandProcessor::ExecutePacketType3_D3D_INLINE(memory::RingBuffer* reader, uint32_t packet, uint32_t count) {
  d3d_tex::WriterScope tex_writer(d3d_tex::kWInline, nf_role_ != NfRole::kAnswers);  // wwe2k14 patch (Session 87, stage 4)
  InlineCensus& c = g_census;
  const size_t payload_end = (reader->read_offset() + size_t(count) * sizeof(uint32_t)) % size_t(reader->capacity());
  auto fail = [&](const char* what, uint32_t a, uint32_t b) {
    REXGPU_ERROR("[d3d] inline token error: {} ({:08X}, {:08X}); the token ({} dwords) is skipped", what, a, b, count);
    c.bad.fetch_add(1, std::memory_order_relaxed);
    reader->set_read_offset(payload_end);
    return true;
  };
  constexpr uint32_t kFixed = 21;
  if (count < kFixed) return fail("short token", count, 0);
  uint32_t w[kFixed];
  for (uint32_t i = 0; i < kFixed; ++i) w[i] = reader->ReadAndSwap<uint32_t>();
  if (w[0] != d3d::kInlineTokenMagic) return fail("bad magic", w[0], w[1]);
  InlineDrawRecord rec;
  rec.sequence = w[1];
  rec.flags = w[2];
  rec.primitive_type = w[3];
  rec.vertex_count = w[4];
  rec.stride_bytes = w[5];
  rec.viz_query_word = w[6];
  for (uint32_t i = 0; i < 4; ++i) rec.slot31_words[i] = w[7 + i];
  const uint32_t nruns = w[11], nvalues = w[12];
  rec.vertex_shader.guest_address = w[13];
  rec.vertex_shader.dword_count = w[14];
  const uint32_t vs_inline = w[15];
  rec.pixel_shader.guest_address = w[16];
  rec.pixel_shader.dword_count = w[17];
  const uint32_t ps_inline = w[18];
  rec.guest_vertex_address = w[19];
  const uint32_t vbytes = w[20];
  const uint64_t need = uint64_t(kFixed) + 3ull * nruns + nvalues + vs_inline + ps_inline + ((uint64_t(vbytes) + 3) >> 2);
  if (need != count) return fail("record size does not match its counts", uint32_t(need), count);
  rec.runs.resize(nruns);
  for (uint32_t i = 0; i < nruns; ++i) {
    rec.runs[i].first_register = reader->ReadAndSwap<uint32_t>();
    rec.runs[i].count = reader->ReadAndSwap<uint32_t>();
    rec.runs[i].value_offset = reader->ReadAndSwap<uint32_t>();
  }
  rec.values.resize(nvalues);
  for (uint32_t i = 0; i < nvalues; ++i) rec.values[i] = reader->ReadAndSwap<uint32_t>();
  auto raw = [&](uint32_t n, std::vector<uint32_t>& out) {
    out.resize(n);
    for (uint32_t i = 0; i < n; ++i) out[i] = std::byteswap(reader->ReadAndSwap<uint32_t>());
  };
  raw(vs_inline, rec.vertex_shader.inline_words);
  raw(ps_inline, rec.pixel_shader.inline_words);
  // (Session 87 review item 6: a shader named by no address must have its microcode in the token.)
  if ((rec.vertex_shader.dword_count && !rec.vertex_shader.guest_address && rec.vertex_shader.dword_count > vs_inline) ||
      (rec.pixel_shader.dword_count && !rec.pixel_shader.guest_address && rec.pixel_shader.dword_count > ps_inline)) {
    return fail("a shader without an address carries fewer words than it claims", rec.vertex_shader.dword_count,
                rec.pixel_shader.dword_count);
  }
  std::vector<uint32_t> vwords;
  raw((vbytes + 3) >> 2, vwords);
  rec.vertices.resize(vbytes);
  if (vbytes) std::memcpy(rec.vertices.data(), vwords.data(), vbytes);
  const uint32_t last = c.last_sequence.exchange(rec.sequence, std::memory_order_relaxed);
  if (last && int32_t(rec.sequence - last) <= 0) c.replays.fetch_add(1, std::memory_order_relaxed);
  return ExecuteD3dInlineDraw(rec);
}

bool CommandProcessor::ExecuteD3dInlineDraw(const d3d::InlineDrawRecord& rec) {
  InlineCensus& c = g_census;
  c.draws.fetch_add(1, std::memory_order_relaxed);
  c.vertices.fetch_add(rec.vertex_count, std::memory_order_relaxed);
  c.vertex_bytes.fetch_add(rec.vertices.size(), std::memory_order_relaxed);
  if (rec.flags & d3d::kInlineTiled) c.tiled.fetch_add(1, std::memory_order_relaxed);

  // Registers are applied a run at a time, as the record token's RUN items are (ExecutePacketType0Body on the run's words as a
  // reader sees them: the feed's Run when an image scope wants it, else the block write WriteRegisterRangeFromRing - the same
  // notifications a type-0 packet gives, ONE per fetch-constant range, not one per register: the six per-register writes of a
  // slot used to notify the texture cache six times, and every notification after the first cleared the slot's object id).
  auto apply_block = [&](uint32_t first, const uint32_t* values, uint32_t count) {
    if (!count) return;  // (a RingBuffer of 0 bytes divides by zero; a 0-count run writes nothing)
    static thread_local std::vector<uint32_t> big_endian;
    big_endian.resize(count);
    for (uint32_t i = 0; i < count; ++i) big_endian[i] = std::byteswap(values[i]);
    memory::RingBuffer block(reinterpret_cast<uint8_t*>(big_endian.data()), size_t(count) * sizeof(uint32_t));
    block.set_write_offset(size_t(count) * sizeof(uint32_t));
    ExecutePacketType0Body(&block, first, count, 0);
  };
  // 1. The state the flush would have written.
  for (const d3d::RegisterRun& run : rec.runs) {
    if (size_t(run.value_offset) + run.count > rec.values.size()) {
      REXGPU_ERROR("[d3d] inline record {}: a register run past its values", rec.sequence);
      c.bad.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    apply_block(run.first_register, rec.values.data() + run.value_offset, run.count);
    c.runs.fetch_add(1, std::memory_order_relaxed);
    c.registers.fetch_add(run.count, std::memory_order_relaxed);
  }

  // 2. The shaders, as the IM_LOAD / IM_LOAD_IMMEDIATE handlers set them.
  auto set_shader = [&](xenos::ShaderType type, const d3d::ShaderBind& bind) {
    if (!bind.present()) return;
    const uint32_t slot = type == xenos::ShaderType::kVertex ? 0 : 1;
    Shader* shader;
    if (bind.guest_address) {
      shader = LoadShader(type, bind.guest_address, D3dShaderMicrocodePtrOf(memory_, bind.guest_address), bind.dword_count);
      half2_ucode_[slot] = D3dShaderHalf2Key(bind.guest_address);
    } else {
      shader = LoadShader(type, 0, bind.inline_words.data(), bind.dword_count);
      half2_ucode_[slot] = 0;
    }
    half2_shader_at_[slot] = half2_packet_at_;
    (type == xenos::ShaderType::kVertex ? active_vertex_shader_ : active_pixel_shader_) = shader;
    RepeatNotePacket(xenos::PM4_IM_LOAD);  // (Session 87 review item 7: a shader load ends a repeat, as the IM_LOAD packet does)
    c.shaders_set.fetch_add(1, std::memory_order_relaxed);
  };
  if (rec.flags & d3d::kInlineHasVertexShader) set_shader(xenos::ShaderType::kVertex, rec.vertex_shader);
  if (rec.flags & d3d::kInlineHasPixelShader) set_shader(xenos::ShaderType::kPixel, rec.pixel_shader);

  // 3. Where the vertex bytes are: the game's own allocation, or ours.
  const uint32_t bytes = rec.vertex_count * rec.stride_bytes;
  const uint32_t dwords = bytes >> 2;
  bool stage_ok = true;
  const uint32_t physical = InlineStage(memory_, rec, stage_ok);
  if (!stage_ok) return true;

  // 4. VGT_INDX_OFFSET = 0, then slot 31 (stream 0): the shadow's words 0-3, and the address / size words the library
  // forms (type 3 = vertex fetch; endian 2, size in dwords at bit 2, the 0x10000000 tag).
  const uint32_t zero = 0;
  apply_block(XE_GPU_REG_VGT_INDX_OFFSET, &zero, 1);
  const uint32_t fetch_first = XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 6 * 31;
  const uint32_t slot31[6] = {rec.slot31_words[0], rec.slot31_words[1], rec.slot31_words[2], rec.slot31_words[3],
                              (physical & ~3u) | 3u, 0x10000002u | ((dwords & 0xFFFFFFu) << 2)};
  apply_block(fetch_first, slot31, 6);  // (one block: one fetch notification for slot 31, as a record run gives)
  // SHADER_CONSTANT_FLUSH_FETCH_0..2 (0x5000-0x5002): the library writes three zero dwords here after the fetch constants.
  // The register file only stores them (SDK: nothing reads them) but the lean path counts the write: kept identical.
  const uint32_t zeros[3] = {0, 0, 0};
  apply_block(0x5000, zeros, 3);

  // 5. The draw: auto-indexed (source select 2), the primitive type and the vertex count in the initiator. The shared
  // draw body reads the one initiator word from a reader; it is that word in a 16-byte buffer, nothing else is parsed.
  const uint32_t initiator = (rec.primitive_type & 0x3Fu) | (2u << 6) | (rec.vertex_count << 16);
  alignas(16) uint8_t word[16] = {};
  memory::store_and_swap<uint32_t>(word, initiator);
  memory::RingBuffer reader(word, sizeof(word));
  reader.set_write_offset(sizeof(uint32_t));
  // (Session 92, M4e, gpu_d3d_owned_front bit 4) A SHADOW: the claim of this draw also builds its description from the typed state of this record (the decoder took
  // the snapshot ahead; the staged vertex address is this path's own, handed in as an input) and compares. The draw itself is unchanged.
  const bool owned_front_inline = (REXCVAR_GET(gpu_d3d_owned_front) & 16u) != 0;
  if (owned_front_inline) vulkan::OwnedRecordInlineBegin(rec.sequence, rec.viz_query_word, initiator, slot31[4], slot31[5]);
  // (Session 92, M4f-B, bit 6) The typed state draws it (staging above is the one body; the staged address is slot 31's words 4-5 in the snapshot); a draw the
  // typed side cannot take goes the old way, counted by reason.
  bool drawn = false;
#if REX_TRUE_NATIVE_ONLY
  // (M4f-C, the proof build: the old inline tail has no typed hook here - the cut's executor draws inline records; this path is never reached under the cut)
  {
    static uint64_t dead_inline = 0;  // (counted: an old-path inline draw in the proof build is a dead drop)
    if (++dead_inline == 1) REXGPU_ERROR("[d3d] an inline draw reached the old inline tail in the TRUE NATIVE proof build: dropped");
  }
  if (false) {
#else
  if (!(owned_front_inline && (REXCVAR_GET(gpu_d3d_owned_front) & 96u) == 96u && vulkan::OwnedFrontInlineDraw(*this, rec.viz_query_word, initiator, drawn))) {
#endif
    drawn = ExecutePacketType3Draw(&reader, 0xC0012200u, "D3D_INLINE_DRAW", rec.viz_query_word, 1);
  }
  if (owned_front_inline) vulkan::OwnedRecordInlineEnd();
  return drawn;
}

void CommandProcessor::D3dInlineCensusFrame(uint32_t frame) {
  InlineCensus& c = g_census;
  if (frame - c.window_start.load(std::memory_order_relaxed) < 600) return;
  const uint64_t draws = c.draws.exchange(0), replays = c.replays.exchange(0), vertices = c.vertices.exchange(0),
                 vbytes = c.vertex_bytes.exchange(0), runs = c.runs.exchange(0), regs = c.registers.exchange(0),
                 shaders = c.shaders_set.exchange(0), staged = c.staged.exchange(0), guest = c.guest_memory.exchange(0),
                 bad = c.bad.exchange(0), tiled = c.tiled.exchange(0), ring_draws = c.ring_draws.exchange(0),
                 ring_wraps = c.ring_wraps.exchange(0), ring_pages = c.ring_pages.exchange(0);
  if (ring_draws) {
    REXLOG_INFO("[d3d][inline] ring (gpu_d3d_inline_host_ring) frames {}-{}: {} inline draws staged on pages of their own ({} pages carried), "
                "ring wraps {} (staging {} KB)",
                c.window_start.load(), frame, ring_draws, ring_pages, ring_wraps, g_staging.size >> 10);
  }
  if (draws || bad) {
    REXLOG_INFO("[d3d] frames {}-{}: inline draws {} ({} sequence repeats = tiling replays, {} recorded in tiled mode), "
                "{} vertices, {} vertex bytes carried ({} staged, {} read from the game's memory); {} register runs "
                "({} registers), {} shaders set; errors {}",
                c.window_start.load(), frame, draws, replays, tiled, vertices, vbytes, staged, guest, runs, regs, shaders,
                bad);
  }
  c.window_start.store(frame, std::memory_order_relaxed);
}

}  // namespace rex::graphics

namespace rex::graphics {
uint32_t D3dInlineStageWords(memory::Memory* memory, const uint32_t* w, uint32_t count, bool& ok) {
  ok = false;
  constexpr uint32_t kFixed = 21;
  if (count < kFixed) return 0;
  d3d::InlineDrawRecord rec;
  rec.flags = w[2];
  rec.vertex_count = w[4];
  rec.stride_bytes = w[5];
  rec.guest_vertex_address = w[19];
  const uint32_t nruns = w[11], nvalues = w[12], vs_inline = w[15], ps_inline = w[18], vbytes = w[20];
  const uint64_t words_before = uint64_t(kFixed) + 3ull * nruns + nvalues + vs_inline + ps_inline;
  if (words_before + ((uint64_t(vbytes) + 3) >> 2) != count) return 0;
  const uint32_t* vwords = w + words_before;
  rec.vertices.resize(vbytes);
  for (uint32_t i = 0; i * 4 < vbytes; ++i) {
    const uint32_t raw = std::byteswap(vwords[i]);  // (the vertex bytes are raw guest bytes stored swapped twice)
    std::memcpy(rec.vertices.data() + i * 4, &raw, std::min<uint32_t>(4, vbytes - i * 4));
  }
  return InlineStage(memory, rec, ok);
}

bool D3dInlineVertexWords(const uint32_t* w, uint32_t count, const uint32_t*& words, uint32_t& bytes, bool& guest_memory) {
  constexpr uint32_t kFixed = 21;
  if (count < kFixed) return false;
  const uint32_t nruns = w[11], nvalues = w[12], vs_inline = w[15], ps_inline = w[18], vbytes = w[20];
  const uint64_t words_before = uint64_t(kFixed) + 3ull * nruns + nvalues + vs_inline + ps_inline;
  if (words_before + ((uint64_t(vbytes) + 3) >> 2) != count) return false;
  guest_memory = (w[2] & d3d::kInlineUseGuestVertexMemory) && w[19];
  const uint64_t need = uint64_t(w[4]) * w[5];
  if (!guest_memory && need > vbytes) return false;
  words = w + words_before;
  bytes = uint32_t(need);
  return true;
}
}  // namespace rex::graphics
