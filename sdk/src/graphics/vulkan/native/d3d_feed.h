// wwe2k14 patch (Session 87, D3D layer stage 3; docs\native\d3d\S3_CONTRACT.md in the game repository): the backend's
// side of the record-fed draw. See include\rex\graphics\d3d_feed.h for the mechanism and native/d3d_feed.cpp for the code.
// Command processor thread only. Included by native/d3d_direct.cpp (the draw handler) and native/d3d_feed.cpp.
#pragma once

#include <cstdint>

#include <rex/graphics/vulkan/command_processor.h>

namespace rex::graphics::vulkan {

// The DRAW item's words.
struct D3dFeedItem {
  uint32_t viz_query;
  uint32_t initiator;  // VGT_DRAW_INITIATOR
  uint32_t dma_base;   // VGT_DMA_BASE
  uint32_t dma_size;   // VGT_DMA_SIZE
};

struct D3dFeedCensus {
  // runs
  uint64_t runs_fed = 0;          // register runs stored into the image (the register file not written)
  uint64_t registers_fed = 0;
  uint64_t runs_stock = 0;        // runs the stock type-0 path wrote (switch off for the moment, or not modelled)
  uint64_t runs_unmodelled = 0;   // ... of those, registers the image does not model (counted by the first register)
  uint64_t rebaselines = 0;       // the image re-copied from the register file (first use, or after a foreign packet)
  uint64_t foreign = 0;           // foreign packets that found the image valid
  uint64_t flushes = 0;           // flushes that copied pending ranges into the register file
  uint64_t flushed_registers = 0;
  uint64_t flush_by_packet = 0;   // ... of those, caused by a foreign packet
  uint64_t flush_by_draw = 0;     // ... caused by a draw handed back to the stock path
  // draws
  uint64_t draws_direct = 0;      // draws taken by the lean variant reading the image
  uint64_t fb_crowd = 0;          // handed back: the draw is a crowd run's first member (its code reads the register file)
  uint64_t crowd_hits = 0;        // draws that are a crowd run's first member (any mode; the image is stale after each)
  uint64_t draws_seen = 0;        // draw items seen with the feed on (own or verify)
  uint64_t fb_pass = 0;           // handed back: a special mode (the stock lean path decides)
  uint64_t fb_template = 0;       // handed back: the lean template did not hold (IssueDraw runs, after a flush)
  uint64_t copy_scopes = 0;         // copy / clear / inline tokens run in an image scope (gpu_d3d_own_copy)
  uint64_t draws_crowd_image = 0;  // crowd runs' first members fed from the record too (gpu_d3d_own_crowd): the members replayed on the image
  uint64_t reg_rmw_fed = 0;         // Session 90: REG_RMW packets applied on the image (no register-file catch-up)
  uint64_t scoped_writes = 0;       // single register writes the crowd code made into the image
  uint64_t scoped_group_registers = 0;  // registers AppendCrowdInstances stored into the image (marked pending)
  uint64_t foreign_by_kind[1024] = {};  // foreign packets that found the image valid, by type << 8 | opcode
  uint64_t flush_by_kind[1024] = {};    // ... and the flushes they caused
  uint32_t flush_first_frame[1024] = {}, flush_last_frame[1024] = {};  // ... the frames of the first and the last flush of the kind
  uint32_t flush_logged = 0;            // first-flush lines printed
  uint64_t draws_full_image = 0; // drawn by the full derivation reading the image (gpu_d3d_own_full): fed from the record too
  uint64_t full_image_pass = 0;   // ... of those, by the stock lean path (a special mode)
  uint64_t full_image_template = 0;  // ... and by IssueDraw (the template did not hold)
  uint64_t fb_copy_mode = 0;      // handed back: the image says edram copy mode (IssueCopy reads the register file)
  uint64_t fb_mode = 0;          // handed back: feeding switched off for this draw
  uint64_t verify_checked = 0;
  uint64_t verify_mismatches = 0;  // draws with at least one register of the image differing from the register file
  uint64_t verify_registers = 0;
  uint32_t verify_logged = 0;
};

struct D3dFeedAccess {
  using VCP = VulkanCommandProcessor;
  enum class Mode { kOff, kOwn, kVerify };

  // The mode for the next run / draw: cvars, and the modes the fed path leaves to the stock one.
  static Mode ModeNow(VCP& cp);
  // The register classes the notifications differ by, and the run apply (members: they touch the processor's state).
  enum class Piece { kFloat, kFetch, kBoolLoop, kOther };
  static void Notify(VCP& cp, Piece piece, uint32_t first, uint32_t count, bool changed);
  static void Apply(VCP& cp, uint32_t first, uint32_t n, const uint32_t* values, bool notify, bool track);
  // The handlers registered in rex/graphics/d3d_feed.h.
  static bool Run(CommandProcessor& cp, rex::memory::RingBuffer* reader, uint32_t first, uint32_t n);
  static void Flush(CommandProcessor& cp);
  static void Foreign(CommandProcessor& cp);
  static bool ScopedWrite(CommandProcessor& cp, uint32_t index, uint32_t value);
  static bool ScopeWanted(CommandProcessor& cp);
  static bool ScopeBegin(CommandProcessor& cp);
  static bool RegRmw(CommandProcessor& cp, uint32_t rmw_info, uint32_t and_mask, uint32_t or_mask);
  static void ScopeEnd(CommandProcessor& cp);
  static uint32_t* ImageValues();
  static void ImageDirty(uint32_t first, uint32_t count);

  // The draw side. BeginDraw makes the image valid, stores the draw's three words into it and (mode kOwn) points the
  // derivations' register source at it; EndDraw points them back. Both can be called twice.
  static void BeginDraw(VCP& cp, const D3dFeedItem& item, Mode mode);
  static void EndDraw(VCP& cp);
  // kVerify: after the stock draw, compares the image with the register file over every range the derivations read.
  static void VerifyDraw(VCP& cp);
  // This draw (its c76-c78 in the image, after BeginDraw) is a crowd run's first member, or a crowd verify is active: the
  // crowd code reads and writes the register file around it.
  static bool CrowdMatch(VCP& cp);
  static D3dFeedCensus& Census();
  static void Print(VCP& cp);

  // ---- Session 92, M4f-C (gpu_d3d_owned_front bit 128, the cut): the processor-side effects of the record decoder acting as the executor
  // (native/owned_record.cpp). The feed is OFF under the cut (ModeNow = kOff: no image, D3dFeedState() 0); these touch the processor's state
  // through this friend only, in the same places and order as the feed's Notify / the token handler's SHADER item / the Registers item.
  // Notify made from a block stored into the decoder's typed state (`fetch_words` = the typed fetch constants: the texture cache and the object
  // table read them through the fetch source). Counts the constant buffers it really marks out of date (free_counters::g_cut).
  static void CutNotify(CommandProcessor& cp, Piece piece, uint32_t first, uint32_t count, bool changed, const uint32_t* fetch_words);
  static void CutFetchChanged(CommandProcessor& cp, uint32_t first, uint32_t count, uint32_t changed_first, uint32_t changed_count, const uint32_t* fetch_words);  // (Session 95, gpu_d3d_wall bit 4)
  // SHADER item: LoadShader once, the active shader. Returns the shader's microcode hash (ucode_data_hash), 0 when the load failed.
  static uint64_t CutLoadShader(CommandProcessor& cp, uint32_t type, uint32_t address, const uint32_t* microcode, uint32_t dwords);
  // The native renderer is enabled (draws before it are dropped and counted by the executor).
  static bool CutReady(CommandProcessor& cp);
  static uint64_t CutFrame(CommandProcessor& cp);  // (Session 95 r5) the frame being decoded
  // The tiling bin ops (ops 3-6 of a Registers item) on the processor's predication state, both sides of the native frame as the old path did.
  static void CutBinOp(CommandProcessor& cp, uint32_t op, uint32_t a);
  // The texture cache's fetch source = `words` (the typed state's fetch constants) for the cut session.
  static void CutFetchSource(CommandProcessor& cp, const uint32_t* words);
};

}  // namespace rex::graphics::vulkan
