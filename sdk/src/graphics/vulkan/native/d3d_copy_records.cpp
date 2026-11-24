// wwe2k14 patch (Session 87, D3D layer family F2): the copy-family records (include/rex/graphics/d3d_copy.h) executed
// into the owned frame. The game's Clear arrives as a record built from the D3D device object (the call's arguments,
// the bound surfaces' words, the device's surface shadows, the clipped rectangle) - no packets of the library's clear
// writer, no register decode, no FastClearDetect. NativeRenderer::D3dClear builds the clear's Draw (the fields the
// owned frame's clear path reads) and offers it to OwnedTakeDraw exactly as a claimed fast clear is offered.
//
// What is NOT here, by design of the contract: any fallback that replays the library's own packets. A clear the owned
// frame does not take is counted and logged as LOST (D3dNoteClearLost); the census line says how many. The clear has no
// pipeline, so the truth path cannot draw it either (see the hand-back: the missing piece is a clear of our own).
#include <rex/graphics/vulkan/native/native_renderer.h>

#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/d3d_copy.h>
#include <rex/graphics/free_counters.h>
#include "owned_dead_clear.h"
#include "owned_record.h"
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/memory/ring_buffer.h>
#include <rex/memory/utils.h>
#include <rex/system/xmemory.h>

#include <algorithm>
#include <bit>
#include <cstdint>

REXCVAR_DECLARE(bool, gpu_det_skip_host_draws);  // (command_processor.cpp)
REXCVAR_DECLARE(uint32_t, gpu_d3d_owned_front);  // (d3d_noring.cpp) bit 4 = M4e
// wwe2k14 patch (Session 92, M3a): a resolve record goes to the owned frame without register writes. Effective only with the owned-only latch
// (NativeRenderer::OwnedOnly()) and the native renderer on; default false = the register path below, byte for byte.
REXCVAR_DEFINE_BOOL(gpu_d3d_owned_resolve, false, "GPU/Native",
                    "Session 92 (M3a): a resolve record is run by IssueCopyFrom from its words (no register writes, no IssueCopy register reads) and "
                    "served by the owned frame (VulkanCommandProcessor::OwnedResolveRecord) instead of the render target cache. Needs "
                    "gpu_native_owned_only; otherwise the register path runs");
// Session 87 (diagnosis): a depth-only record clear is drawn with the colour mask 0 and the depth-only EDRAM mode (the reviewed,
// exact form); false = the colour-and-depth form with mask 0xFFFF (what the owned frame's clear hold was built on).
REXCVAR_DEFINE_BOOL(gpu_d3d_copy_clear_depthonly_inherit, true, "GPU/Native",
                    "Session 87 (D3D layer F2): a depth-only record clear runs with the colour mask / blend / colour control the "
                    "device had (carried in the record), as the library's depth-only packets did; false = the constant forms");
REXCVAR_DEFINE_BOOL(gpu_d3d_copy_clear_depthonly_exact, true, "GPU/Native",
                    "Session 87: a depth-only record clear writes RB_COLOR_MASK 0 and the depth-only EDRAM mode; false = colour mask 0xFFFF and "
                    "the colour+depth mode (diagnosis of the owned frame's pass starts)");

// The capability marker the game exe looks for (rex::cvar::GetFlagInfo): this SDK build executes the copy-family record.
REXCVAR_DEFINE_BOOL(gpu_d3d_copy_clear_hold, false, "GPU/Native",
                    "Session 87 (D3D layer F2): let the owned frame hold a record clear without a draw (NativeRenderer::D3dClear) "
                    "when it can; off (default): every record clear is drawn the library's way (registers + its clear shaders + "
                    "IssueDraw), counted as clear_stock");

static uint64_t g_boot_clears = 0;  // (Session 92, M2d: clears made before the native renderer was ever enabled - counted apart from clear_stock)

REXCVAR_DEFINE_INT32(gpu_d3d_copy_abi, rex::graphics::kD3dCopyAbi, "GPU/Native",
                     "Session 87 (D3D layer F2): the layout version of the copy-family record packet (opcode 0x7D: the "
                     "game's Clear and Resolve as records) this SDK build executes. Not a setting - the game exe takes "
                     "the records only when it equals the number its own copy of the layout was written for.");

namespace rex::graphics::vulkan {

// The friend of VulkanCommandProcessor (command_processor.h): the executor's way to the native renderer.
struct D3dCopyAccess {
  static bool Execute(CommandProcessor& base, uint32_t kind, const uint32_t* sem, uint32_t count);
  static bool Resolve(VulkanCommandProcessor& cp, const uint32_t* sem, uint32_t count);
  static bool ClearStock(VulkanCommandProcessor& cp, const uint32_t* sem, uint32_t count);
};

namespace {
D3dCopyCensus g_census;
uint64_t g_census_frame = 0;
bool g_registered = [] {
  D3dCopySetHandler(&D3dCopyAccess::Execute);
  return true;
}();

float F(uint32_t bits) { return std::bit_cast<float>(bits); }
}  // namespace

bool D3dCopyAccess::Execute(CommandProcessor& base, uint32_t kind, const uint32_t* sem, uint32_t count) {
  VulkanCommandProcessor& cp = static_cast<VulkanCommandProcessor&>(base);
  ++g_census.packets;
  if (g_census.packets - g_census_frame >= 1200) {  // (a census line every 1200 records; the game side prints its own)
    g_census_frame = g_census.packets;
    REXGPU_INFO("[d3d] copy records so far: {} packets, clears {} (boot clears {}, held by the owned frame {}, clear_stock {} [hold off {}, refused {}], lost {}), GPR splits "
                "dropped {}, bad {}",
                g_census.packets, g_census.clears, g_boot_clears, g_census.clears_held, g_census.clears_stock, g_census.clears_stock_hold_off, g_census.clears_stock_refused, g_census.clears_lost, g_census.gpr_splits,
                g_census.bad);
  }
  switch (kind) {
    case kD3dCopyClear: {
      ++g_census.clears;
      // (The stock way needs a submission only - the very first clear of a run comes before the native renderer exists.)
      if (!cp.BeginSubmission(true)) {
        ++g_census.clears_lost;
        REXGPU_ERROR("[d3d] clear record lost: no submission");
        return false;
      }
      // (Session 92, M2d) The run's first clear(s), before the native renderer is ever enabled, are BOOT clears: drawn the stock way, counted apart, so clear_stock
      // (and the free counter) read the clears after boot. (A run whose native renderer never comes up counts all of them as boot.)
      static bool s_native_seen = false;
      if (cp.native_ && cp.native_->enabled()) s_native_seen = true;
      if (!s_native_seen) {
        ++g_boot_clears;
        return ClearStock(cp, sem, count);
      }
      // The semantic hold (a pipeline-less held draw) is OFF by default: a held clear that the owned frame later flushes or
      // replays (a frame where it turns off, a pass it does not feed) cannot be drawn from the truth path, and drawing it
      // there would re-enter IssueDraw inside a draw being recorded. The stock way below gives every clear a real pipeline.
      // (Session 92, M4f-B, gpu_d3d_owned_front bits 32 + 64: the hold is effective whatever the cvar says, and a clear the owned frame refuses is PENDED by D3dClear
      // (OwnedPendClearMake), never drawn the library's way: ClearStock is reached only by the boot clears above. A clear D3dClear cannot pend - no renderer or owned
      // frame, a colour format without a class, no surface slot, a stencil-only clear - is LOST, counted and logged.)
      // (Review fix 1: only an owned-only frame of a running native renderer takes this way; a released latch, a side-car frame, a renderer off after boot keep the old hold-cvar / ClearStock path exactly.)
      const bool bit = native::OwnedPendBitOn();
      const bool native_up = cp.native_ && cp.native_->enabled();
      const bool b_on = bit && native_up && cp.native_->OwnedOnlyActive();
      if (free_counters::g_cut.active && !b_on) {  // (THE CUT: ClearStock is never run after boot; a clear the owned frame cannot take is LOST and counted)
        ++g_census.clears_lost;
        free_counters::NoteCutDrop("clear lost: not an owned-only frame under the cut (ClearStock is not run)");
        return true;
      }
      if (bit && !b_on) native::OwnedPendNoteRefusal(6);  // (counted: bit 64 on but not an owned-only frame - the old path runs)
      if ((REXCVAR_GET(gpu_d3d_copy_clear_hold) || b_on) && native_up) {
        const uint32_t empty0 = native::OwnedPendEmptyTotal();
        if (cp.native_->D3dClear(sem, count)) {
          if (native::OwnedPendEmptyTotal() == empty0) ++g_census.clears_held;  // (an empty rectangle clears nothing: counted on its own in the pend line, not as held)
          return true;
        }
        if (b_on) {
          ++g_census.clears_lost;
          cp.native_->D3dNoteClearLost("M4f-B: not takeable, not pendable");
          return false;
        }
        ++g_census.clears_stock_refused;
      } else {
        ++g_census.clears_stock_hold_off;
      }
      ++g_census.clears_stock;
      rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.clear_stock);
      return ClearStock(cp, sem, count);
    }
    case kD3dCopyGprSplit:
      ++g_census.gpr_splits;
      return true;
    case kD3dCopyResolve:
      return Resolve(cp, sem, count);
    default:
      ++g_census.bad;
      REXGPU_ERROR("[d3d] copy record of unknown kind {}", kind);
      return false;
  }
}

// The Clear record drawn the library's own way (listing 0x82CB4B00 / 0x82CB4560 / 0x82CB42F8 / 0x82CB4040, read in Session
// 87): the register images the library's clear writes into its stream, written straight into the register file (no
// packets), the library's two clear shaders (microcode in the game image: vertex 27 dwords at 0x82082A38, pixel 9 dwords at
// 0x82082A10), three vertices of 7 floats (x y z r g b a: the rectangle's corners as a rectangle list, the depth, the
// colour) in memory of ours behind vertex fetch constant 0, and the rectangle-list draw through the draw path (IssueDraw,
// via ExecutePacketType3Draw with the one initiator word). Everything it writes - the registers and the active shaders - is
// put back afterwards: the library's own clear ends by marking its clobbered state dirty so the game re-sends it, and with
// the writer skipped nothing is marked, so the register file and the shaders must look as before the clear.
// Generic (colour + optional depth / stencil) form only: a colour-only clear is drawn as a colour draw (the library packs
// the colour into a depth write for large colour-only rectangles, a hardware trick with the same pixels).
#if !REX_TRUE_NATIVE_ONLY
bool D3dCopyAccess::ClearStock(VulkanCommandProcessor& cp, const uint32_t* s, uint32_t count) {
  using namespace d3d_clear_sem;
  if (count < kDwords) {
    ++g_census.bad;
    return false;
  }
  // Vertex memory of ours: guest physical (the shared memory uploads from it as from the game's own), a small ring.
  static uint32_t staging_virtual = 0, staging_offset = 0;
  static bool staging_failed = false;
  constexpr uint32_t kStagingBytes = 64 * 1024;
  if (!staging_virtual && !staging_failed) {
    // (Session 90: `staging_virtual` holds a PHYSICAL address out of the SDK system reserve, not the game pool.)
    uint32_t address = 0;
    if (memory::D3dLayerTakePhysicalReserve(kStagingBytes, &address)) {
      staging_virtual = address;
      memory::D3dLayerNoteGuestMemory("clear vertex staging", kStagingBytes, false, address);
    } else {
      staging_failed = true;
    }
  }
  if (!staging_virtual) {
    ++g_census.clears_lost;
    REXGPU_ERROR("[d3d] clear record lost: no vertex staging memory");
    return false;
  }
  // The library's clear shaders, read from the image (guest byte order, as IM_LOAD_IMMEDIATE carries them).
  constexpr uint32_t kVertexShaderAddress = 0x82082A38u, kVertexShaderDwords = 27;
  constexpr uint32_t kPixelShaderAddress = 0x82082A10u, kPixelShaderDwords = 9;
  std::vector<uint32_t> vs_words(kVertexShaderDwords), ps_words(kPixelShaderDwords);
  std::memcpy(vs_words.data(), cp.memory_->TranslateVirtual<const uint8_t*>(kVertexShaderAddress), kVertexShaderDwords * 4);
  std::memcpy(ps_words.data(), cp.memory_->TranslateVirtual<const uint8_t*>(kPixelShaderAddress), kPixelShaderDwords * 4);

  const uint32_t flags = s[kFlags];
  const bool depth = (flags & 0x10) != 0, stencil = (flags & 0x20) != 0;
  const uint32_t surface_info = s[kSurfaceInfo];
  const uint32_t msaa = (surface_info >> 16) & 3;
  // The vertices (listing 0x82CB4040): the rectangle's corners (l,t) (r,t) (r,b) less the half pixel when the vertex rounding
  // is the D3D one (PA_SU_VTX_CNTL bit 0 clear), the depth, the colour.
  const float half = (s[kVtxCntl] & 1) ? 0.0f : 0.5f;
  const float x0 = float(int32_t(s[kRect + 0])) - half, y0 = float(int32_t(s[kRect + 1])) - half;
  const float x1 = float(int32_t(s[kRect + 2])) - half, y1 = float(int32_t(s[kRect + 3])) - half;
  const float z = depth ? F(s[kDepth]) : 0.0f;
  const float corners[3][2] = {{x0, y0}, {x1, y0}, {x1, y1}};
  uint32_t use = (staging_offset + 15u) & ~15u;
  if (use + 84 > kStagingBytes) use = 0;  // (an older clear's vertices are long consumed: its upload was recorded in order)
  staging_offset = use + 84;
  uint8_t* host = cp.memory_->TranslateVirtual<uint8_t*>(0xA0000000u + staging_virtual + use) /* a watched view (not the physical one) */;
  for (uint32_t v = 0; v < 3; ++v) {
    const float vertex[7] = {corners[v][0], corners[v][1], z, F(s[kColor + 0]), F(s[kColor + 1]), F(s[kColor + 2]),
                             F(s[kColor + 3])};
    for (uint32_t i = 0; i < 7; ++i) memory::store_and_swap<uint32_t>(host + (v * 7 + i) * 4, std::bit_cast<uint32_t>(vertex[i]));
  }
  const uint32_t physical = staging_virtual + use;

  // The registers (the library's fixed packets of 0x82CB4B00 / 0x82CB4560 / 0x82CB4040), remembered to be put back.
  // (Session 87, stage 3, gpu_d3d_own_copy: inside the image scope the values to put back are the image's, and WriteRegister
  // writes the image - VulkanCommandProcessor::WriteRegister -; outside it, the register file as before.)
  const RegisterFile& regs = cp.DrawRegs();
  std::vector<std::pair<uint32_t, uint32_t>> saved;
  const auto write = [&](uint32_t index, uint32_t value) {
    if (std::none_of(saved.begin(), saved.end(), [&](const auto& p) { return p.first == index; })) {
      saved.emplace_back(index, regs[index]);
    }
    cp.WriteRegister(index, value);
  };
  write(XE_GPU_REG_SQ_PROGRAM_CNTL, 0x10010001u);
  write(XE_GPU_REG_SQ_CONTEXT_MISC, 0);
  write(XE_GPU_REG_VGT_MAX_VTX_INDX, 0xFFFF);
  write(XE_GPU_REG_VGT_MIN_VTX_INDX, 0);
  write(XE_GPU_REG_VGT_INDX_OFFSET, 0);
  write(XE_GPU_REG_PA_SC_VIZ_QUERY, 0);
  write(XE_GPU_REG_PA_CL_CLIP_CNTL, 0x10000);
  write(XE_GPU_REG_PA_SU_SC_MODE_CNTL, 0x10000);
  write(XE_GPU_REG_PA_CL_VTE_CNTL, 0x300);
  write(XE_GPU_REG_PA_SC_AA_MASK, 0xFFFF);
  write(XE_GPU_REG_PA_SU_VTX_CNTL, s[kVtxCntl]);
  const bool depth_only = s[kPassRt] == 0xFFFFFFFFu;
  bool depth_only_mode = depth_only;  // (the depth-only EDRAM mode, unless the diagnosis form asks for the colour+depth one)
  if (depth_only && count >= kDwordsFull && REXCVAR_GET(gpu_d3d_copy_clear_depthonly_inherit)) {
    // The library's depth-only branch writes none of the colour state (listing 0x82CB4B00: the colour mask / blend / colour
    // control packets are in the colour-target branch only): its draw runs with what the flush before it left - the device's
    // shadows, carried in the record. That keeps the draw the same fast clear the truth path saw from the library's packets
    // (writes colour with whatever mask the scene had, colour info 0).
    write(XE_GPU_REG_RB_COLOR_MASK, s[kColorMask]);
    write(XE_GPU_REG_RB_COLORCONTROL, s[kColorControl]);
    write(XE_GPU_REG_RB_BLENDCONTROL0, s[kBlendControl]);
  } else {
    const bool depth_only_exact = depth_only && REXCVAR_GET(gpu_d3d_copy_clear_depthonly_exact);
    depth_only_mode = depth_only_exact;
    write(XE_GPU_REG_RB_COLOR_MASK, depth_only_exact ? 0u : 0xFFFFu);  // (depth-only: no colour target, nothing to write)
    write(XE_GPU_REG_RB_COLORCONTROL, 0);
    write(XE_GPU_REG_RB_BLENDCONTROL0, 0x00010001u);
  }
  if (stencil) write(XE_GPU_REG_RB_STENCILREFMASK, (s[kStencil] & 0xFF) | 0x00FF0000u);
  write(XE_GPU_REG_PA_SC_WINDOW_OFFSET, 0);
  write(XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL, 0);
  write(XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR, 0x20002000u);
  write(XE_GPU_REG_RB_SURFACE_INFO, surface_info);
  write(XE_GPU_REG_RB_COLOR_INFO, s[kColorInfo]);
  write(XE_GPU_REG_RB_DEPTH_INFO, depth || stencil ? s[kDepthInfo] : regs[XE_GPU_REG_RB_DEPTH_INFO]);
  write(XE_GPU_REG_PA_SC_AA_CONFIG, msaa == 0 ? 0u : msaa == 1 ? 0x8001u : 0xC003u);
  write(XE_GPU_REG_RB_DEPTHCONTROL, (depth ? 0x76u : 0u) | (stencil ? 0x8701u : 0u));
  write(XE_GPU_REG_RB_MODECONTROL, uint32_t(depth_only_mode ? xenos::EdramMode::kDepthOnly : xenos::EdramMode::kColorDepth));
  // Vertex fetch constant 0 (registers 0x4800-0x4805): the vertices, 21 dwords; and the three flush-fetch registers.
  write(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0, (physical & ~3u) | 3u);
  write(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + 1, 0x10000002u | (21u << 2));
  for (uint32_t i = 2; i < 6; ++i) write(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + i, 0);
  for (uint32_t i = 0; i < 3; ++i) write(0x5000 + i, 0);

  // The shaders, as the IM_LOAD_IMMEDIATE handler sets them (and the draw path's other users of them put back after).
  Shader* const old_vs = cp.active_vertex_shader_;
  Shader* const old_ps = cp.active_pixel_shader_;
  const uint32_t old_ucode[2] = {cp.half2_ucode_[0], cp.half2_ucode_[1]};
  const uint32_t old_at[2] = {cp.half2_shader_at_[0], cp.half2_shader_at_[1]};
  cp.active_vertex_shader_ = cp.LoadShader(xenos::ShaderType::kVertex, 0, vs_words.data(), kVertexShaderDwords);
  cp.active_pixel_shader_ = cp.LoadShader(xenos::ShaderType::kPixel, 0, ps_words.data(), kPixelShaderDwords);
  cp.half2_ucode_[0] = cp.half2_ucode_[1] = 0;

  // (Session 92, M4e, gpu_d3d_owned_front bit 4) A SHADOW: the effect this clear is about to have, read back from what was just written (the registers, the
  // staged vertices), is compared with the clear the typed state + the record describe. Reads only; the draw below is unchanged.
  if ((REXCVAR_GET(gpu_d3d_owned_front) & 16u) != 0) {
    OwnedClearOld old;
    old.surface_info = regs[XE_GPU_REG_RB_SURFACE_INFO];
    old.color_info = regs[XE_GPU_REG_RB_COLOR_INFO];
    old.depth_info = regs[XE_GPU_REG_RB_DEPTH_INFO];
    old.color_mask = regs[XE_GPU_REG_RB_COLOR_MASK];
    old.depth_control = regs[XE_GPU_REG_RB_DEPTHCONTROL];
    old.stencil_refmask = regs[XE_GPU_REG_RB_STENCILREFMASK];
    old.mode_control = regs[XE_GPU_REG_RB_MODECONTROL];
    old.aa_config = regs[XE_GPU_REG_PA_SC_AA_CONFIG];
    old.vtx_cntl = regs[XE_GPU_REG_PA_SU_VTX_CNTL];
    for (uint32_t i = 0; i < 21; ++i) old.vertex[i / 7][i % 7] = memory::load_and_swap<uint32_t>(host + i * 4);
    OwnedRecordClearShadow(s, count, old);
  }

  // The draw: three vertices, auto-index, the rectangle list (primitive 8).
  const uint32_t initiator = uint32_t(xenos::PrimitiveType::kRectangleList) | (2u << 6) | (3u << 16);
  alignas(16) uint8_t word[16] = {};
  memory::store_and_swap<uint32_t>(word, initiator);
  memory::RingBuffer reader(word, sizeof(word));
  reader.set_write_offset(sizeof(uint32_t));
  const bool ok = cp.ExecutePacketType3Draw(&reader, 0xC0012200u, "D3D_CLEAR", 0, 1);

  cp.active_vertex_shader_ = old_vs;
  cp.active_pixel_shader_ = old_ps;
  cp.half2_ucode_[0] = old_ucode[0];
  cp.half2_ucode_[1] = old_ucode[1];
  cp.half2_shader_at_[0] = old_at[0];
  cp.half2_shader_at_[1] = old_at[1];
  for (auto it = saved.rbegin(); it != saved.rend(); ++it) cp.WriteRegister(it->first, it->second);
  return ok;
}
#else
bool D3dCopyAccess::ClearStock(VulkanCommandProcessor&, const uint32_t*, uint32_t) {
  return false;  // (the TRUE NATIVE proof build has no library-way clear: a boot clear is lost, counted by the caller)
}
#endif  // !REX_TRUE_NATIVE_ONLY

// The Resolve record: a ResolveInfo from the record's values (draw_util::GetResolveInfoFromInput - no register file, no
// vertices, no scissor), the record's register images written into the register file (no packets), and IssueCopy run with
// the ResolveInfo override (the one thing no register holds, the rectangle, rides in that way). IssueCopy does every piece of
// per-copy work itself - the render target cache's Resolve (OnResolve, OwnedTakeResolve, the texture marks, the clear,
// AfterResolve), the load-turbo / dead-glare / shadow-chain skips, half-rate shadows, DLAA bookkeeping, the readback history
// and the GPU marks - so there is no owned branch here (Session 87 review items 1 and 2: the first version took the owned
// copy itself and skipped that work, and a refused take ran OnResolve / OwnedTakeResolve twice).
bool D3dCopyAccess::Resolve(VulkanCommandProcessor& cp, const uint32_t* s, uint32_t count) {
  using namespace d3d_resolve_sem;
  ++g_census.resolves;
  if (count < kDwords) {
    ++g_census.bad;
    REXGPU_ERROR("[d3d] resolve record of {} dwords (expected {})", count, kDwords);
    return false;
  }
  if (REXCVAR_GET(gpu_det_skip_host_draws)) {  // (as IssueCopy's own skip: no host copy in the determinism no-draw runs)
    ++g_census.resolves_det_skipped;
    return true;
  }
  draw_util::D3dResolveInput in;
  in.copy_control = s[kCopyControl];
  in.dest_base = s[kDestBase];
  in.dest_pitch = s[kDestPitch];
  in.dest_info = s[kDestInfo];
  in.depth_clear = s[kDepthClear];
  in.color_clear = s[kColorClear];
  in.color_clear_lo = s[kColorClearLo];
  in.surface_info = s[kSurfaceInfo];
  in.color_info = s[kColorInfo];
  in.depth_info = s[kDepthInfo];
  in.vtx_cntl = s[kVtxCntl];
  in.x0 = int32_t(s[kRect + 0]);
  in.y0 = int32_t(s[kRect + 1]);
  in.x1 = int32_t(s[kRect + 2]);
  in.y1 = int32_t(s[kRect + 3]);
  if ((REXCVAR_GET(gpu_d3d_owned_front) & 16u) != 0) {  // (Session 92, M4e shadow: the words M3 feeds IssueCopyFrom against the typed state's view)
    OwnedResolveOld old;
    old.copy_control = in.copy_control;
    old.dest_base = in.dest_base;
    old.dest_pitch = in.dest_pitch;
    old.dest_info = in.dest_info;
    old.surface_info = in.surface_info;
    old.color_info = in.color_info;
    old.depth_info = in.depth_info;
    old.mode_control = uint32_t(xenos::EdramMode::kCopy);  // (the CopyWords the owned way builds)
    old.vtx_cntl = in.vtx_cntl;
    OwnedRecordResolveShadow(s, count, old);
  }
  if (!cp.BeginSubmission(true)) return false;
  VulkanRenderTargetCache& rtc = *cp.render_target_cache_;
  // (Session 92, M3a) The owned frame's way: IssueCopyFrom from the record's words and its ResolveInfo, no register written.
  const bool owned_way = REXCVAR_GET(gpu_d3d_owned_resolve) && cp.native_ && cp.native_->enabled() && cp.native_->OwnedResolveReady();
  if (free_counters::g_cut.active && !owned_way) {  // (THE CUT: the register path is never run; a resolve the owned way cannot take is DROPPED and counted)
    free_counters::NoteCutDrop("resolve dropped: the owned resolve is not ready (the register path is not run under the cut)");
    return true;
  }
  const uint32_t sx = rtc.draw_resolution_scale_x(), sy = rtc.draw_resolution_scale_y();
  draw_util::ResolveInfo info;
  if (!draw_util::GetResolveInfoFromInput(in, sx, sy, rtc.IsFixedRG16TruncatedToMinus1To1(),
                                          rtc.IsFixedRGBA16TruncatedToMinus1To1(), info)) {
    ++g_census.resolves_empty;  // (nothing to copy or clear: as the library's own resolve of an empty rectangle)
    return true;
  }
  if (owned_way) {
    CopyWords w;
    w.copy_control = in.copy_control;
    w.dest_base = in.dest_base;
    w.dest_pitch = in.dest_pitch;
    w.dest_info = in.dest_info;
    w.surface_info = in.surface_info;
    w.color_info = in.color_info;
    w.depth_info = in.depth_info;
    w.mode_control = uint32_t(xenos::EdramMode::kCopy);  // (the library's resolve runs in the copy EDRAM mode)
    const bool ok = cp.IssueCopyFrom(w, &info);
    ++g_census.resolves_issued;
    return ok;
  }
  rex::graphics::free_counters::Bump(rex::graphics::free_counters::g_c.resolves_regs);  // (Session 92, M3a: log-only counter)
  const uint32_t src = info.rb_copy_control.copy_src_select;
  cp.WriteRegister(XE_GPU_REG_RB_SURFACE_INFO, in.surface_info);
  // (The colour target's info register only for a colour source: a depth resolve (source 4) must not zero RB_COLOR3_INFO.)
  if (src < 4) cp.WriteRegister(reg::RB_COLOR_INFO::rt_register_indices[src], in.color_info);
  cp.WriteRegister(XE_GPU_REG_RB_DEPTH_INFO, in.depth_info);
  cp.WriteRegister(XE_GPU_REG_RB_COPY_CONTROL, in.copy_control);
  cp.WriteRegister(XE_GPU_REG_RB_COPY_DEST_BASE, in.dest_base);
  cp.WriteRegister(XE_GPU_REG_RB_COPY_DEST_PITCH, in.dest_pitch);
  cp.WriteRegister(XE_GPU_REG_RB_COPY_DEST_INFO, in.dest_info);
  cp.WriteRegister(XE_GPU_REG_RB_DEPTH_CLEAR, in.depth_clear);
  cp.WriteRegister(XE_GPU_REG_RB_COLOR_CLEAR, in.color_clear);
  cp.WriteRegister(XE_GPU_REG_RB_COLOR_CLEAR_LO, in.color_clear_lo);
  cp.WriteRegister(XE_GPU_REG_PA_SU_VTX_CNTL, in.vtx_cntl);
  draw_util::SetResolveInfoOverride(&info);
  const bool ok = cp.IssueCopy();
  draw_util::SetResolveInfoOverride(nullptr);
  // The library's own resolve ends by zeroing the copy control and the copy destination info (listing 0x82CAEA30's tail): so
  // the next copy-reading code sees no stale copy.
  cp.WriteRegister(XE_GPU_REG_RB_COPY_CONTROL, 0);
  cp.WriteRegister(XE_GPU_REG_RB_COPY_DEST_INFO, 0);
  ++g_census.resolves_issued;
  return ok;
}


// wwe2k14 patch (Session 92, M4f-C): the cut's executor runs a clear / resolve record from its host-order block (no packet, no reader): the same
// executor the packet handler reaches (D3dCopyAccess::Execute).
bool OwnedExecCopy(CommandProcessor& cp, uint32_t kind, const uint32_t* sem, uint32_t sem_dwords) {
  return D3dCopyAccess::Execute(cp, kind, sem, sem_dwords);
}

}  // namespace rex::graphics::vulkan

namespace rex::graphics::vulkan::native {

namespace {
uint64_t g_lost_total = 0, g_lost_window = 0;
}

void NativeRenderer::D3dNoteClearLost(const char* where) {
  ++g_lost_total;
  ++g_lost_window;
  if (g_lost_total <= 12 || (g_lost_total & (g_lost_total - 1)) == 0) {
    REXGPU_ERROR("[d3d] frame {}: a record clear was LOST ({}); {} so far. The owned frame did not take it and the "
                 "truth path has no pipeline for it",
                 frame_, where, g_lost_total);
  }
}

bool NativeRenderer::D3dClear(const uint32_t* sem, uint32_t count) {
  using namespace rex::graphics::d3d_clear_sem;
  const bool pend = OwnedPendBitOn() && OwnedOnly();  // (Session 92, M4f-B, review fix 1: only in an owned-only frame; a refusal below is counted as unpendable, the clear is lost - ClearStock is not reached)
  if (count < kDwords) return false;
  // (The owned frame's conditions, as OwnedTakeDrawImpl tests them first; it tests them again.)
  if (!enabled_ || !truth_ || !replacing_ || !owned_frame_on_) {
    if (pend) OwnedPendNoteRefusal(0);
    return false;
  }

  const uint32_t flags = sem[kFlags];
  const uint32_t surface_info = sem[kSurfaceInfo];
  const bool colour = sem[kPassRt] != 0xFFFFFFFFu;
  const bool depth = (flags & 0x10) != 0;
  const bool stencil = (flags & 0x20) != 0;
  const uint32_t msaa = (surface_info >> 16) & 3;  // 0 1x, 1 2x, 2 4x
  const uint32_t sx = msaa >= 2 ? 2 : 1, sy = msaa >= 1 ? 2 : 1;
  const uint32_t pitch_tiles = std::max(1u, ((surface_info & 0x3FFF) * sx + 79) / 80);

  Draw d;
  d.fast_clear = true;
  d.truth = true;
  d.post = true;
  d.owned = true;
  d.d3d_record = true;
  d.screen_space = true;
  d.raster.supported = true;
  d.surface_info = surface_info;
  d.color_info = colour ? sem[kColorInfo] : 0;
  d.depth_info = depth || stencil ? sem[kDepthInfo] : 0;
  d.color_write_mask = colour ? 0xFu : 0;
  d.color_mask_all = d.color_write_mask;
  d.blend_control = 0x00010001u;  // one, zero: written as it is (what ClearsAccumulate asks of a clear)
  d.clear_color_known = colour;
  for (uint32_t i = 0; i < 4; ++i) d.clear_color[i] = F(sem[kColor + i]);
  // The depth a clear writes whole: always passing, written, no bias. Stencil: replace, every bit, one reference.
  d.raster.depth_test = d.raster.depth_write = depth;
  d.raster.depth_compare = VK_COMPARE_OP_ALWAYS;
  d.clear_depth = depth ? F(sem[kDepth]) : -1.0f;
  if (stencil) {
    d.raster.stencil = true;
    const uint32_t face = uint32_t(VK_STENCIL_OP_REPLACE) << 3 | uint32_t(VK_COMPARE_OP_ALWAYS) << 9;
    d.raster.stencil_ops = face | face << 12;
    const uint8_t value = uint8_t(sem[kStencil] & 0xFF);
    d.stencil_ref = value;
    for (uint32_t i = 0; i < 2; ++i) {
      d.stencil_refs[i] = value;
      d.stencil_read_masks[i] = 0xFF;
      d.stencil_write_masks[i] = 0xFF;
    }
  }
  d.truth_sx = uint8_t(sx);
  d.truth_sy = uint8_t(sy);
  // The rectangle in guest samples (the fill's coordinates), the scissor wide open (TruthDrawRect narrows to the fill).
  d.fill_rect[0] = sem[kRect + 0] * sx;
  d.fill_rect[1] = sem[kRect + 1] * sy;
  d.fill_rect[2] = sem[kRect + 2] * sx;
  d.fill_rect[3] = sem[kRect + 3] * sy;
  if (d.fill_rect[2] <= d.fill_rect[0] || d.fill_rect[3] <= d.fill_rect[1]) {
    if (!pend) return false;
    OwnedPendNoteRefusal(5);  // (an empty rectangle clears nothing: handled, as the library's own clear of it draws nothing)
    return true;
  }
  d.fill_pitch_tiles = pitch_tiles;
  d.scissor.offset = {0, 0};
  d.scissor.extent = {16384u, 16384u};
  // (Session 92, M2b: SurfaceSlot = the surface registry under the latch, the truth images otherwise.)
  // The truth images this clear belongs to, as ClaimTruth finds them (colour by its class and format, depth/stencil kind 3).
  d.truth_class = uint8_t(colour ? TruthClass(d) : 1u);
  if (colour) {
    if (!TruthClass(d)) {  // (a colour format the truth images have no class for)
      if (pend) OwnedPendNoteRefusal(2);
      return false;
    }
    d.color_quantize = TruthClass(d);
    d.truth_color = SurfaceSlot(d.color_info & 0xFFF, pitch_tiles, TruthClass(d), (d.color_info >> 16) & 0xF);
    d.fill_color_base = d.color_info & 0xFFF;
    if (d.truth_color < 0) {
      if (pend) OwnedPendNoteRefusal(3);
      return false;
    }
  }
  if (depth || stencil) {
    d.truth_depth = SurfaceSlot(d.depth_info & 0xFFF, pitch_tiles, 3, (d.depth_info >> 16) & 1);
    d.fill_depth_base = d.depth_info & 0xFFF;
    if (d.truth_depth < 0) {
      if (pend) OwnedPendNoteRefusal(3);
      return false;
    }
  }
  // (A listed clear shader passes its colour through unquantised-idle: ClearsAccumulate's "quantized" flag; ours is exact.)
  d.pixel_ext = SpirvShaderTranslator::kNativeExtEpilogueIdle;
  d.truth_epoch = truth_claim_epoch_;
  if (OwnedTakeDraw(d)) return true;
  if (!pend) return false;
  // (Session 92, M4f-B) Not taken: the clear is the surface's PENDING clear (OwnedPendClearMake ends what holds the surface first, so no later draw lands in a pass that lacks the clear).
  if (OwnedPendClearMake(d, kPendRefusal, true)) return true;
  OwnedPendNoteRefusal(!colour && !depth ? 1 : 4);
  return false;
}

}  // namespace rex::graphics::vulkan::native
