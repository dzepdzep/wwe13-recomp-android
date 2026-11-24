// wwe2k14 patch (Session 87, D3D layer family F2): the copy-family record packet's base-side reader (see
// include/rex/graphics/d3d_copy.h). It only frames the packet: the executor is the backend's.
#include <rex/graphics/d3d_copy.h>
#include <rex/graphics/d3d_tex.h>

#include <vector>

#include <rex/logging.h>
#include <rex/memory/ring_buffer.h>

namespace rex::graphics {

namespace {
D3dCopyHandler g_handler = nullptr;
uint32_t g_expected_seq = 0;  // the next sequence number (0: none seen yet)
}  // namespace

void D3dCopySetHandler(D3dCopyHandler handler) { g_handler = handler; }

bool D3dCopyPacket(CommandProcessor& cp, rex::memory::RingBuffer* reader, uint32_t count) {
  d3d_tex::WriterScope tex_writer(d3d_tex::kWCopy, false);  /* (a copy token has no Bind to end) */  // wwe2k14 patch (Session 87, stage 4)
  static thread_local std::vector<uint32_t> payload;
  payload.resize(count);
  for (uint32_t i = 0; i < count; ++i) payload[i] = reader->ReadAndSwap<uint32_t>();
  if (count < kD3dCopyHeaderDwords || payload[0] != kD3dCopyMagic ||
      count != kD3dCopyHeaderDwords + payload[3]) {
    REXGPU_ERROR("[d3d] copy packet malformed: {} dwords, magic {:08X}, kind {}, sem {}", count, count ? payload[0] : 0,
                 count > 2 ? payload[2] : 0, count > 3 ? payload[3] : 0);
    return true;  // (consumed; the record is lost, which the backend's census cannot see: loud here)
  }
  if (g_expected_seq && payload[1] != g_expected_seq) {
    REXGPU_ERROR("[d3d] copy packet sequence {} where {} was expected (a record was skipped or replayed)", payload[1],
                 g_expected_seq);
  }
  g_expected_seq = payload[1] + 1;
  if (!g_handler) {
    REXGPU_ERROR("[d3d] copy packet (kind {}) with no backend handler: dropped", payload[2]);
    return true;
  }
  g_handler(cp, payload[2], payload.data() + kD3dCopyHeaderDwords, payload[3]);
  return true;  // (the record is consumed either way; a record the executor could not take is counted by it, loudly)
}

}  // namespace rex::graphics
