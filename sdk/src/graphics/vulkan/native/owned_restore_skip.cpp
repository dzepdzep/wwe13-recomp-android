// wwe2k14 patch (Session 95, gpu_owned_restore_skip): the entrance frame's restore and the draws the game repeats after it.
//
// Bryan's frame (Session 95 skin1 run 007 frame 7492, the segment log): the scene pass, a copy of colour and depth, 11 draws, the far blur reads that picture, then a whole-screen restore puts
// the scene back and the SAME 11 draws are drawn again before the wrestlers. Restore + repeat leaves what the image held before them, so neither has to run.
// Bit 1: watch only: the draws after a copy and the ones after the next restore compared (state, geometry, bindings, the constant bytes the GPU reads), counted per window.
// Bit 2: watch only: each render pass's inputs against the same pass of the frame before.
// Bit 4: the skip. Learned (8 whole repeats, the pages the restore read never read again before the game replaced them; a differing frame does not unlearn it, it costs nothing more than
//        the old move: the same one copy): at the copy's next draw the pass stays in its image and
//        the pages get a copy of their own (OwnedLeanDetach); at the restore the pages on the image move the same way and the restore is recorded into the same image, as drawn as ever. The
//        draws go on being compared; when the last one matches, the image already holds what restore + repeat would leave: their draws are turned off in the stream (their state stays),
//        and at the submission's end the pages' copy too. Anything else on the way (a draw that differs, a read of the pages by anything but the restore, a restore that does not cover):
//        nothing is turned off, and the stream as recorded is the game's frame: a wrong guess costs copies, never a pixel. The one case the stream cannot prove: a read of those pages in a
//        LATER frame before the game replaces them (held past the frame): watched while learning, and if it ever happens counted and logged, and the skip stands down.
// Rule 6: our own images; nothing here models the EDRAM.
// State for a job system: all of it is the owned frame's (one thread, the decoder's side).

#include <rex/graphics/vulkan/native/native_renderer.h>

#include <algorithm>
#include <cstring>

#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/logging.h>

REXCVAR_DECLARE(int32_t, gpu_native_truth_pass_log);
REXCVAR_DEFINE_INT32(gpu_owned_restore_skip, 0, "GPU/Native",
                     "Session 95 (default 0 = off): a restore whose following draws repeat the draws since the last copy leaves what the image held before them. Bit 1: compare and log; "
                     "bit 2: log each pass's inputs against the frame before; bit 4: the restore, the repeated draws and the copy for the restore turned off in the stream when proven (needs "
                     "gpu_native_owned_frame_continue_copy and the owned-only frame). 5 = the skip; 0 = today's frame. Bit 16 (Session 95 r5, a look lever): the same draws in another order count as a repeat though they blend, in every restore list of every scene, not the entrance's 11 draws alone: the "
                     "out of order count of the [native][lean][rs] line says how often it decided");

namespace rex::graphics::vulkan::native {

namespace {
constexpr size_t kMaxSeq = 48;
constexpr uint32_t kGoodToArm = 8;
uint32_t CopyBit(uint32_t copy_count) { return 1u << std::min(copy_count, 31u); }
struct Hash {
  uint64_t h = 0x9E3779B97F4A7C15ull;
  template <typename T>
  void add(const T& v) {
    uint64_t x = 0;
    std::memcpy(&x, &v, std::min(sizeof(T), sizeof(x)));
    h = (h ^ x) * 0xFF51AFD7ED558CCDull;
    h ^= h >> 32;
  }
};
// The vertex float bank of a shader with dynamic addressing (all 256 vec4). The game leaves stack leftovers in four words of it, new at every draw (runs 003, 005, the only words of the 11 draws that
// differ): all of c28 (a position at times, pointer bits at others), c185.y, c185.z. Our own shaders of these draws read c56-59, c72, c209, c254, c255 and the bones c[76 + 3n]; c185.x is the only part of c185 read
// anywhere (vs_ch_gear). Every other word has to be the same, bit for bit.
bool SameWords(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint64_t& leftovers) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] == b[i]) continue;
    if (a.size() != 1024 || (i / 4 != 28 && i != 185 * 4 + 1 && i != 185 * 4 + 2)) return false;
    ++leftovers;
  }
  return true;
}
}  // namespace

NativeRenderer::OwnedRsSig NativeRenderer::OwnedRsSigOf(const Draw& draw, const int32_t rect[4]) {
  OwnedRsSig s;
  s.vs = draw.vertex_shader_hash;
  s.ps = draw.pixel_shader_hash;
  Hash st;
  st.add(draw.vertex_shader_hash);
  st.add(draw.pixel_shader_hash);
  st.add(draw.pipeline);
  st.add(draw.layout);
  st.add(draw.vertex_modification);
  st.add(draw.pixel_modification);
  st.add(draw.color_write_mask);
  st.add(draw.color_mask_all);
  st.add(draw.blend_control);
  for (const float f : draw.blend_constants) st.add(f);
  st.add(draw.raster.cull_mode);
  st.add(draw.raster.depth_test);
  st.add(draw.raster.depth_write);
  st.add(draw.raster.depth_compare);
  st.add(draw.raster.depth_bias_constant);
  st.add(draw.raster.depth_bias_slope);
  st.add(draw.raster.stencil);
  st.add(draw.raster.stencil_ops);
  for (uint32_t f = 0; f < 2; ++f) {
    st.add(draw.stencil_refs[f]);
    st.add(draw.stencil_read_masks[f]);
    st.add(draw.stencil_write_masks[f]);
  }
  st.add(draw.viewport.x);
  st.add(draw.viewport.y);
  st.add(draw.viewport.width);
  st.add(draw.viewport.height);
  st.add(draw.viewport.minDepth);
  st.add(draw.viewport.maxDepth);
  for (uint32_t i = 0; i < 4; ++i) st.add(rect[i]);
  s.state = st.h;
  Hash g;
  g.add(draw.indexed);
  g.add(draw.index_buffer);
  g.add(draw.index_offset);
  g.add(draw.index_type);
  g.add(draw.vertex_count);
  g.add(draw.instance_count);
  for (uint32_t i = 0; i < draw.vb_count; ++i) {
    g.add(draw.vb[i]);
    g.add(draw.vb_offsets[i]);
  }
  g.add(draw.vertex_offset);
  s.geometry = g.h;
  Hash b;
  for (uint32_t i = 0; i < SpirvShaderTranslator::kDescriptorSetCount; ++i) {
    if (i == SpirvShaderTranslator::kDescriptorSetConstants || i >= SpirvShaderTranslator::kDescriptorSetMutableLayoutsStart || !(draw.sets_bound & (1u << i))) continue;
    b.add(draw.sets[i]);
  }
  s.sets = b.h;
  s.consts = command_processor_.OwnedConstantsHash();
  {
    size_t size = 0;
    const uint8_t* words = command_processor_.OwnedConstantsBytes(SpirvShaderTranslator::kConstantBufferFloatVertex, size);
    if (words) s.vertex_floats.assign(reinterpret_cast<const uint32_t*>(words), reinterpret_cast<const uint32_t*>(words) + size / 4);
  }
  // (Not plain: what the signature cannot vouch for. A page of ours can change between the two draws; the textures' descriptor set is made per draw, so the used fetch constants in s.consts stand for it.)
  // (Opaque, depth tested and written: the order of such draws does not change what they leave, but for two at the same depth.)
  s.opaque = (draw.blend_control == 0x00010001u) && draw.raster.depth_test && draw.raster.depth_write &&
             (draw.raster.depth_compare == VK_COMPARE_OP_LESS || draw.raster.depth_compare == VK_COMPARE_OP_LESS_OR_EQUAL);
  s.plain = !draw.own_skip_call && !draw.memexport && !draw.occlusion_query && !draw.raster.stencil && s.consts != 0 && (draw.restore || owned_sampled_images_.empty());
  if (REXCVAR_GET(gpu_owned_restore_skip) & 8) {
    for (uint32_t i = 0; i < 5; ++i) {
      size_t size = 0;
      const uint8_t* bytes = command_processor_.OwnedConstantsBytes(i, size);
      if (bytes) s.bytes[i].assign(bytes, bytes + size);
    }
    for (uint32_t i = 0; i < SpirvShaderTranslator::kDescriptorSetCount && i < 8; ++i) {
      if (i != SpirvShaderTranslator::kDescriptorSetConstants && (draw.sets_bound & (1u << i))) s.set_list[i] = draw.sets[i];
    }
  }
  return s;
}

void NativeRenderer::OwnedRsNote(const OwnedPass& p, const Draw& draw, const int32_t rect[4]) {
  const int32_t mode = REXCVAR_GET(gpu_owned_restore_skip);
  if (!mode) return;
  if (!draw.restore && !owned_rs_compare_ && owned_rs_last_restore_ + 240 <= frame_) return;
  OwnedRsStats& st = owned_rs_stats_;
  if (owned_rs_compare_) {
    const OwnedRsSig s = OwnedRsSigOf(draw, rect);
    // (The game sorts the two lists apart at times: the same draws in another order are the same list if every one of them is opaque.)
    const auto same_draw = [&](const OwnedRsSig& r) { uint64_t leftovers = 0; return s.state == r.state && s.geometry == r.geometry && s.sets == r.sets && s.consts == r.consts && SameWords(s.vertex_floats, r.vertex_floats, leftovers); };
    size_t pick = 0;
    while (pick < owned_rs_ref_.size() && owned_rs_ref_[pick].used) ++pick;
    // (Reordering is only allowed when EVERY draw of the list is opaque, depth tested and written with LESS / LEQUAL: one blended draw in it makes the order the picture.)
    // (Session 95 r5, bit 16, a look lever: the entrance's 11 draws are alpha blended with depth writes, compare GREATER_OR_EQUAL (blend 07060706), and Bryan's wide shot lists them in another order
    // than the blur's list: the image keeps the blur list's order. Only where two of them overlap with alpha under 1 can a pixel differ. Off: as before.)
    const bool any_order = (REXCVAR_GET(gpu_owned_restore_skip) & 16) != 0;
    const bool all_opaque = std::all_of(owned_rs_ref_.begin(), owned_rs_ref_.end(), [](const OwnedRsSig& x) { return x.opaque; });
    if (pick < owned_rs_ref_.size() && !same_draw(owned_rs_ref_[pick]) && (any_order || (s.opaque && all_opaque))) {
      for (size_t i = 0; i < owned_rs_ref_.size(); ++i) {
        const OwnedRsSig& c = owned_rs_ref_[i];
        if (c.used || (!any_order && !c.opaque) || !same_draw(c)) continue;
        pick = i;
        ++st.out_of_order;
        break;
      }
    }
    if (pick >= owned_rs_ref_.size()) pick = owned_rs_ref_.size() - 1;
    OwnedRsSig& r = owned_rs_ref_[pick];
    const bool same_state = s.state == r.state, same_geometry = s.geometry == r.geometry, same_sets = s.sets == r.sets,
               same_consts = s.consts == r.consts && SameWords(s.vertex_floats, r.vertex_floats, st.near_words);
    if (same_state && same_geometry && same_sets && same_consts && s.plain && !r.used) {
      r.used = true;
      if (++owned_rs_at_ == owned_rs_ref_.size()) {
        ++st.full;
        owned_rs_compare_ = false;
        owned_lean_learn_[owned_rs_learn_entry_].repeats |= owned_rs_learn_bit_;
        ++owned_rs_good_;
        OwnedRsArm& a = owned_rs_arm_;
        if (a.armed && a.restore_seen && a.nop_begin && a.bit == owned_rs_learn_bit_ && a.pass_image == p.image) {
          DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
          cb.FlushHeldCommands();
          if (cb.NopDraws(a.nop_begin, cb.stream_offset(), true)) {
            a.proved = true;
            ++st.skipped;
            static bool first = false;
            if (!first) {
              first = true;
              REXGPU_INFO("[native][lean][rs] frame {}: first restore and {} repeated draws turned off in the stream; the pages' copy goes at the submission's end", frame_, owned_rs_ref_.size());
            }
          } else {
            ++st.nop_failed;
          }
        }
      }
    } else {
      ++st.differ;
      st.differ_at += owned_rs_at_;
      st.differ_state += !same_state;
      st.differ_geometry += !same_geometry;
      st.differ_sets += !same_sets;
      st.differ_consts += !same_consts;
      st.differ_plain += !s.plain;
      if ((REXCVAR_GET(gpu_owned_restore_skip) & 8) && st.lines < 12 && same_state && same_geometry) {
        std::string detail;
        for (uint32_t i = 0; i < 5; ++i) {
          const std::vector<uint8_t>&x = s.bytes[i], &y = r.bytes[i];
          uint32_t count = 0;
          std::string first;
          const size_t n = std::min(x.size(), y.size()) / 4;
          for (size_t d = 0; d < n; ++d) {
            uint32_t a, b;
            std::memcpy(&a, x.data() + d * 4, 4);
            std::memcpy(&b, y.data() + d * 4, 4);
            if (a == b) continue;
            float fa, fb;
            std::memcpy(&fa, &a, 4);
            std::memcpy(&fb, &b, 4);
            if (count < 10) first += fmt::format(" [{}] {:08X}({:g}) was {:08X}({:g})", d, a, fa, b, fb);
            ++count;
          }
          detail += fmt::format(" | buffer {} size {}/{} words differing {}{}", i, x.size(), y.size(), count, first);
        }
        std::string sets_differ;
        for (uint32_t i = 0; i < 8; ++i) {
          if (s.set_list[i] != r.set_list[i]) sets_differ += fmt::format(" {}", i);
        }
        REXGPU_INFO("[native][lean][rs] frame {} detail: sets differing:{}{}", frame_, sets_differ, detail);
      }
      if (REXCVAR_GET(gpu_native_truth_pass_log) > 0) {  // (Session 95 r5: which draw of the list keeps it from going out of order)
        std::string list;
        for (const OwnedRsSig& x : owned_rs_ref_) list += fmt::format(" {:08X}/{:08X}:{}{}", uint32_t(x.vs >> 32), uint32_t(x.ps >> 32), x.opaque ? 'o' : 'b', x.used ? 'u' : '-');
        REXGPU_INFO("[native][lean][rs] frame {}: differs at {} of {}: this draw {:08X}/{:08X} opaque {} blend {:08X} depth test {} write {} compare {}; list{}", frame_, owned_rs_at_, owned_rs_ref_.size(),
                    uint32_t(draw.vertex_shader_hash >> 32), uint32_t(draw.pixel_shader_hash >> 32), s.opaque, draw.blend_control, draw.raster.depth_test, draw.raster.depth_write, uint32_t(draw.raster.depth_compare), list);
      }
      if (st.lines < 12 && same_state && same_geometry) {
        ++st.lines;
        REXGPU_INFO("[native][lean][rs] frame {}: draw {} of {} after the restore differs (state {} geometry {} sets {} constants {} plain {}), vs {:016X} ps {:016X}", frame_, owned_rs_at_,
                    owned_rs_ref_.size(), !same_state, !same_geometry, !same_sets, !same_consts, s.plain, draw.vertex_shader_hash, draw.pixel_shader_hash);
      }
      owned_rs_compare_ = false;
    }
    return;
  }
  if (draw.restore) {
    ++st.restores;
    owned_rs_last_restore_ = frame_ + 1;
    st.ref_draws += owned_rs_seq_.size();
    // (Only a restore that rewrites the whole image from the pages the list of draws began on says anything about that list.)
    const int16_t left = owned_rs_arm_.armed ? owned_rs_arm_.image : owned_rs_left_image_;
    const bool from_left = owned_sampled_images_.size() == 1 && owned_sampled_images_[0] == left;
    const bool covers = OwnedLeanCovers(p, draw, rect);
    st.not_covered += !covers;
    st.other_pages += !from_left;
    if (from_left) owned_rs_stale_image_ = left;
    if (covers && from_left && !owned_rs_seq_.empty() && owned_rs_seq_.size() < kMaxSeq) {
      owned_rs_ref_ = owned_rs_seq_;
      owned_rs_at_ = 0;
      owned_rs_compare_ = true;
    }
    owned_rs_seq_.clear();
    return;
  }
  if (p.draws_since_color == 1) {
    owned_rs_seq_.clear();
    owned_rs_learn_bit_ = CopyBit(p.copy_count);
    owned_rs_learn_entry_ = OwnedLeanEntry(p);
  }
  // (Not in a scene with no restores: the first restore after a time is only counted; the list starts with the next frame's copy.)
  // (Once a list is learned and armed, only the lists of a copy that has repeats are signed, plus a few frames in 64 to find new ones: the signature is a 4 KB vector and a hash a draw.)
  const bool skip_sign = owned_rs_good_ >= kGoodToArm && !(owned_lean_learn_[owned_rs_learn_entry_].repeats & owned_rs_learn_bit_) && (frame_ & 63) >= 2;
  if (owned_rs_seq_.size() < kMaxSeq && owned_rs_last_restore_ + 240 > frame_ && !skip_sign) owned_rs_seq_.push_back(OwnedRsSigOf(draw, rect));
}

bool NativeRenderer::OwnedRsGoOn(OwnedPass& p, const Draw& draw, const int32_t rect[4]) {
  if (!(REXCVAR_GET(gpu_owned_restore_skip) & 4) || !OwnedOnly()) return false;
  OwnedRsStats& st = owned_rs_stats_;
  OwnedRsArm& a = owned_rs_arm_;
  const size_t learn = OwnedLeanEntry(p);
  const uint32_t bit = CopyBit(p.copy_count);
  if (!a.armed) {
    if (draw.restore || draw.repeat || owned_rs_good_ < kGoodToArm || !(owned_lean_learn_[learn].repeats & bit)) return false;
    if (!OwnedLeanDetach(p, learn, bit)) return false;
    const OwnedLeanPending q = owned_lean_pending_.back();
    owned_lean_pending_.pop_back();  // (this copy's fate is decided here, not by "replaced unread")
    a = OwnedRsArm();
    a.armed = true;
    a.image = q.image;
    a.bit = bit;
    a.pass_image = p.image;
    a.copy_begin = q.begin;
    a.copy_end = q.end;
    a.submission = q.submission;
    ++st.armed;
    return true;
  }
  if (a.restore_seen || !draw.restore) return false;
  // The restore: stays in the image if it is the one the list of draws expects; the pages on the image (the far blur's copy) take a copy as any detach, turned off when replaced unread.
  // (The list's own copy, not the pass's count now: the copy for the blur comes after the draws, with no draw after it, and leaves the list as it was. Session 95, stack 2: `bit` here disarmed every frame.)
  if (a.bit != owned_rs_learn_bit_ || a.pass_image != p.image || !OwnedLeanCovers(p, draw, rect) || owned_sampled_images_.size() != 1 || owned_sampled_images_[0] != a.image || !OwnedLeanDetach(p, learn, bit)) {
    ++st.disarmed;
    a = OwnedRsArm();
    return false;
  }
  a.restore_seen = true;
  ++st.stayed;
  return true;
}

void NativeRenderer::OwnedRsMoved(int16_t old_image) { owned_rs_left_image_ = old_image; }

void NativeRenderer::OwnedRsNopBegin() {
  OwnedRsArm& a = owned_rs_arm_;
  if (!a.restore_seen || a.nop_begin) return;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  cb.FlushHeldCommands();
  a.nop_begin = cb.stream_offset();
}

void NativeRenderer::OwnedRsAfterDraw(const Draw& draw) {
  OwnedRsArm& a = owned_rs_arm_;
  if (!a.armed || draw.restore) return;
  if (std::find(owned_sampled_images_.begin(), owned_sampled_images_.end(), a.image) != owned_sampled_images_.end()) {
    a.other_read = true;
    ++owned_rs_stats_.other_reads;
  }
}

void NativeRenderer::OwnedRsPageRead(uint32_t page) {
  const auto stale = std::find_if(owned_rs_stale_.begin(), owned_rs_stale_.end(), [&](const std::pair<uint32_t, int16_t>& e) { return e.first == page; });
  if (stale == owned_rs_stale_.end()) return;
  // (Made this frame, or now on another image: the game replaced it.)
  const int16_t image = stale->second;
  if (std::any_of(owned_pages_.begin(), owned_pages_.end(), [&](const OwnedPage& q) { return q.page == page && (q.frame == frame_ || q.image != image); })) {
    owned_rs_stale_.erase(stale);
    return;
  }
  OwnedRsStats& st = owned_rs_stats_;
  ++st.stale_reads;
  if (st.stale_reads <= 4) REXGPU_INFO("[native][lean][rs] frame {}: page 0x{:X}, left by a restore, read before the game replaced it: the skip stands down until it is proven again", frame_, page);
  owned_rs_good_ = 0;
  owned_rs_stale_.clear();
}

void NativeRenderer::OwnedRsPagesChanged() {
  if (owned_rs_stale_.empty()) return;
  owned_rs_stale_.erase(std::remove_if(owned_rs_stale_.begin(), owned_rs_stale_.end(),
                                       [&](const std::pair<uint32_t, int16_t>& e) { return std::none_of(owned_pages_.begin(), owned_pages_.end(), [&](const OwnedPage& q) { return q.page == e.first && q.image == e.second; }); }),
                        owned_rs_stale_.end());
}

void NativeRenderer::OwnedRsSubmissionEnd() {
  OwnedRsArm& a = owned_rs_arm_;
  if (!a.armed) return;
  OwnedRsStats& st = owned_rs_stats_;
  DeferredCommandBuffer& cb = command_processor_.deferred_command_buffer();
  if (a.proved && !a.other_read && a.submission == command_processor_.GetCurrentSubmission() && a.copy_end <= cb.stream_offset() && cb.NopRange(a.copy_begin, a.copy_end)) {
    ++st.copies_off;
  } else if (a.proved) {
    ++st.nop_failed;
  }
  a = OwnedRsArm();
}

void NativeRenderer::OwnedRsPassNote(const OwnedPass& p, const Draw& draw, const int32_t rect[4]) {
  if (!(REXCVAR_GET(gpu_owned_restore_skip) & 2)) return;
  if (p.timer_draws == 0 || owned_rs_passes_.empty()) {
    OwnedRsPass n;
    n.ps = draw.pixel_shader_hash;
    n.width = p.host_width;
    n.height = p.host_height;
    owned_rs_passes_.push_back(n);
  }
  OwnedRsPass& e = owned_rs_passes_.back();
  const OwnedRsSig s = OwnedRsSigOf(draw, rect);
  uint64_t vf = 0;
  for (size_t i = 0; i < s.vertex_floats.size(); ++i) {
    if (s.vertex_floats.size() == 1024 && (i / 4 == 28 || i == 185 * 4 + 1 || i == 185 * 4 + 2)) continue;
    vf = (vf ^ s.vertex_floats[i]) * 0xFF51AFD7ED558CCDull;
    vf ^= vf >> 29;
  }
  ++e.draws;
  Hash a, b;
  a.add(e.all);
  a.add(s.state);
  a.add(s.geometry);
  a.add(s.sets);
  a.add(s.consts);
  a.add(vf);
  b.add(e.no_sets);
  b.add(s.state);
  b.add(s.geometry);
  b.add(s.consts);
  b.add(vf);
  e.all = a.h;
  e.no_sets = b.h;
}

void NativeRenderer::OwnedRsFrame() {
  if (!REXCVAR_GET(gpu_owned_restore_skip)) return;
  OwnedRsStats& st = owned_rs_stats_;
  owned_rs_compare_ = false;
  owned_rs_seq_.clear();
  // (Pages a restore read and that stay held past this frame: read before they are replaced, the skip is wrong for them.)
  // (Kept past the frame: a page stays watched until the game replaces it, which OwnedRsPagesChanged / OwnedRsPageRead see; capped.)
  if (owned_rs_stale_image_ >= 0) {
    for (const OwnedPage& q : owned_pages_) {
      if (q.image != owned_rs_stale_image_ || owned_rs_stale_.size() >= 4096) continue;
      const auto has = std::find_if(owned_rs_stale_.begin(), owned_rs_stale_.end(), [&](const std::pair<uint32_t, int16_t>& e) { return e.first == q.page; });
      if (has == owned_rs_stale_.end()) owned_rs_stale_.emplace_back(q.page, q.image);
      else has->second = q.image;
    }
  }
  owned_rs_stale_image_ = -1;
  owned_rs_left_image_ = -1;
  if (REXCVAR_GET(gpu_owned_restore_skip) & 2) {
    for (size_t i = 0; i < owned_rs_passes_.size(); ++i) {
      const OwnedRsPass& c = owned_rs_passes_[i];
      auto it = std::find_if(owned_rs_tally_.begin(), owned_rs_tally_.end(), [&](const OwnedRsTally& t) { return t.ps == c.ps && t.width == c.width && t.height == c.height; });
      if (it == owned_rs_tally_.end()) {
        OwnedRsTally t;
        t.ps = c.ps;
        t.width = c.width;
        t.height = c.height;
        owned_rs_tally_.push_back(t);
        it = owned_rs_tally_.end() - 1;
      }
      ++it->passes;
      const bool have = i < owned_rs_passes_prev_.size() && owned_rs_passes_prev_[i].ps == c.ps && owned_rs_passes_prev_[i].draws == c.draws && owned_rs_passes_prev_[i].width == c.width;
      if (!have) {
        ++it->unmatched;
      } else {
        it->same += owned_rs_passes_prev_[i].all == c.all;
        it->same_no_sets += owned_rs_passes_prev_[i].no_sets == c.no_sets;
      }
    }
    owned_rs_passes_prev_.swap(owned_rs_passes_);
    owned_rs_passes_.clear();
  }
  // (Session 95, review r5, 5: bit 16 can change a blended picture only where it decided: its count, fine enough to set against a shot's frame.)
  if ((REXCVAR_GET(gpu_owned_restore_skip) & 16) && st.out_of_order != st.out_of_order_logged && frame_ / 60 != st.out_of_order_slot) {
    st.out_of_order_slot = frame_ / 60;
    REXGPU_INFO("[native][lean][rs] frame {}: out of order {} since the last such line", frame_, st.out_of_order - st.out_of_order_logged);
    st.out_of_order_logged = st.out_of_order;
  }
  if (++st.frames < 600) return;
  if (REXCVAR_GET(gpu_owned_restore_skip) & 2) {
    std::sort(owned_rs_tally_.begin(), owned_rs_tally_.end(), [](const OwnedRsTally& x, const OwnedRsTally& y) { return x.passes > y.passes; });
    for (size_t i = 0; i < std::min<size_t>(owned_rs_tally_.size(), 40); ++i) {
      const OwnedRsTally& t = owned_rs_tally_[i];
      REXGPU_INFO("[native][lean][rs] frame {}: passes ps {:016X} {}x{}: {:.2f} a frame; the same as the frame before {:.2f}, without bindings {:.2f}, not paired {:.2f}", frame_, t.ps, t.width, t.height,
                  t.passes / 600.0, t.same / 600.0, t.same_no_sets / 600.0, t.unmatched / 600.0);
    }
    owned_rs_tally_.clear();
  }
  const double f = 1.0 / double(st.frames);
  REXGPU_INFO("[native][lean][rs] frame {}: per frame: restores recorded {:.2f} (draws before them {:.1f}; not covering {:.2f}, other pages {:.2f}), whole repeats {:.2f}, differing {:.2f} (at draw {:.1f}; state {:.2f} "
              "geometry {:.2f} sets {:.2f} constants {:.2f} not plain {:.2f}); armed {:.2f}, stayed for the restore {:.2f}, skipped {:.2f}, copies off {:.2f}, disarmed {:.2f}, nop failed {:.2f}, other reads {:.2f}, stale reads {}, out of order {}",
              frame_, st.restores * f, st.restores ? double(st.ref_draws) / st.restores : 0.0, st.not_covered * f, st.other_pages * f, st.full * f, st.differ * f, st.differ ? double(st.differ_at) / st.differ : 0.0,
              st.differ_state * f, st.differ_geometry * f, st.differ_sets * f, st.differ_consts * f, st.differ_plain * f, st.armed * f, st.stayed * f, st.skipped * f, st.copies_off * f, st.disarmed * f,
              st.nop_failed * f, st.other_reads * f, st.stale_reads, st.out_of_order);
  const uint32_t lines = st.lines;
  st = OwnedRsStats();
  st.lines = lines;
}

}  // namespace rex::graphics::vulkan::native
