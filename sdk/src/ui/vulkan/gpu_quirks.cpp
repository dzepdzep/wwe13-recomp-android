// wwe2k14 patch (Session 93, idea 6): the driver quirks table. See gpu_quirks.h.

#include "gpu_quirks.h"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace rex::ui::vulkan::quirks {

namespace {

std::string Lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool Contains(const std::string& haystack, const char* needle) {
  return Lower(haystack).find(needle) != std::string::npos;
}

std::string Fmt(const char* format, ...) {
  char buffer[512];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  return buffer;
}

// Arm "rNNpM" out of a driverInfo string such as "v1.r44p1-01eac0.abc" (matrix 2.0: parse driverInfo first).
bool ParseArmRp(const std::string& info, int& r, int& p) {
  for (size_t i = 0; i + 3 < info.size(); ++i) {
    if (info[i] != 'r' || !std::isdigit(static_cast<unsigned char>(info[i + 1]))) continue;
    size_t j = i + 1;
    int rv = 0;
    while (j < info.size() && std::isdigit(static_cast<unsigned char>(info[j]))) rv = rv * 10 + (info[j++] - '0');
    if (j >= info.size() || info[j] != 'p' || j + 1 >= info.size() || !std::isdigit(static_cast<unsigned char>(info[j + 1]))) continue;
    ++j;
    int pv = 0;
    while (j < info.size() && std::isdigit(static_cast<unsigned char>(info[j]))) pv = pv * 10 + (info[j++] - '0');
    r = rv;
    p = pv;
    return true;
  }
  return false;
}

bool IsTurnip(const DeviceInfo& info) {
  return info.driver_id == kDriverMesaTurnip || Contains(info.device_name, "turnip") ||
         Contains(info.driver_name, "turnip");
}

bool IsQualcommProprietary(const DeviceInfo& info) {
  return info.vendor_id == kVendorQualcomm && !IsTurnip(info);
}

// (r, p) <= (r2, p2)
bool RpLe(int r, int p, int r2, int p2) { return r < r2 || (r == r2 && p <= p2); }

bool IsMobileVendor(uint32_t vendor) {
  return vendor == kVendorQualcomm || vendor == kVendorArm || vendor == kVendorImagination ||
         vendor == kVendorSamsung;
}

void AddInfo(Result& r, const char* rule, std::string text) {
  Decision d;
  d.kind = Decision::kInfo;
  d.rule = rule;
  d.reason = std::move(text);
  r.decisions.push_back(std::move(d));
}

void AddSet(Result& r, const char* rule, const char* cvar, const char* value, std::string reason,
            bool advisory = false, bool init_only = false) {
  Decision d;
  d.kind = Decision::kSetCvar;
  d.rule = rule;
  d.cvar = cvar;
  d.value = value;
  d.reason = std::move(reason);
  d.advisory = advisory;
  d.init_only = init_only;
  r.decisions.push_back(std::move(d));
}

}  // namespace

const char* GpuClassName(GpuClass c) {
  switch (c) {
    case GpuClass::kDesktop: return "desktop";
    case GpuClass::kIgpu: return "igpu";
    case GpuClass::kPhone: return "phone";
    case GpuClass::kCpu: return "cpu";
  }
  return "desktop";
}

const char* DriverIdName(uint32_t id) {
  switch (id) {
    case kDriverUnknown: return "UNKNOWN";
    case kDriverAmdProprietary: return "AMD_PROPRIETARY";
    case kDriverAmdOpenSource: return "AMD_OPEN_SOURCE";
    case kDriverMesaRadv: return "MESA_RADV";
    case kDriverNvidiaProprietary: return "NVIDIA_PROPRIETARY";
    case kDriverIntelProprietaryWindows: return "INTEL_PROPRIETARY_WINDOWS";
    case kDriverIntelOpenSourceMesa: return "INTEL_OPEN_SOURCE_MESA";
    case kDriverImaginationProprietary: return "IMAGINATION_PROPRIETARY";
    case kDriverQualcommProprietary: return "QUALCOMM_PROPRIETARY";
    case kDriverArmProprietary: return "ARM_PROPRIETARY";
    case kDriverMesaLlvmpipe: return "MESA_LLVMPIPE";
    case kDriverMoltenVk: return "MOLTENVK";
    case kDriverGoogleSwiftshader: return "GOOGLE_SWIFTSHADER";
    case kDriverMesaTurnip: return "MESA_TURNIP";
    case kDriverSamsungProprietary: return "SAMSUNG_PROPRIETARY";
    default: return "OTHER";
  }
}

uint32_t EffectiveDriverId(const DeviceInfo& info) {
  if (info.driver_id != kDriverUnknown) return info.driver_id;
  if (IsTurnip(info)) return kDriverMesaTurnip;
  switch (info.vendor_id) {
    case kVendorQualcomm: return kDriverQualcommProprietary;
    case kVendorArm: return kDriverArmProprietary;
    case kVendorImagination: return kDriverImaginationProprietary;
    case kVendorSamsung: return kDriverSamsungProprietary;
    case kVendorNvidia: return kDriverNvidiaProprietary;
    case kVendorAmd: return kDriverAmdProprietary;
    case kVendorIntel: return kDriverIntelProprietaryWindows;
    default: return kDriverUnknown;
  }
}

DriverVersion DecodeDriverVersion(const DeviceInfo& info) {
  DriverVersion v;
  const uint32_t raw = info.driver_version;
  const uint32_t id = EffectiveDriverId(info);
  if (info.vendor_id == kVendorArm && id == kDriverArmProprietary) {
    // Matrix 2.0: driverInfo "rNNpM" first, then VK_MAKE_VERSION(rN, pM, 0); a non-zero patch field = a hash.
    int r, p;
    if (ParseArmRp(info.driver_info, r, p)) {
      v.arm_r = r;
      v.arm_p = p;
    } else if ((raw & 0xFFF) == 0 && (raw >> 22) < 100) {
      v.arm_r = int(raw >> 22);
      v.arm_p = int((raw >> 12) & 0x3FF);
    }
    if (v.arm_r >= 0) {
      v.major = uint32_t(v.arm_r);
      v.minor = uint32_t(v.arm_p);
      v.text = Fmt("r%dp%d", v.arm_r, v.arm_p);
    } else {
      v.text = Fmt("unknown(0x%X)", raw);
    }
    return v;
  }
  if (id == kDriverNvidiaProprietary) {  // 10/8/8/6 bits
    v.major = raw >> 22;
    v.minor = (raw >> 14) & 0xFF;
    v.patch = (raw >> 6) & 0xFF;
    v.build = raw & 0x3F;
    v.text = Fmt("%u.%u.%u.%u", v.major, v.minor, v.patch, v.build);
    return v;
  }
  if (id == kDriverIntelProprietaryWindows) {  // 101.2112 = the last two fields of 31.0.101.2112
    v.major = raw >> 14;
    v.minor = raw & 0x3FFF;
    v.text = Fmt("%u.%u", v.major, v.minor);
    return v;
  }
  if (id == kDriverImaginationProprietary) {  // a changeset number
    v.major = raw;
    v.text = Fmt("%u", raw);
    return v;
  }
  // Qualcomm proprietary (bit 31 set -> major 512), Turnip / Mesa, AMD, Samsung, others: the standard split.
  v.major = raw >> 22;
  v.minor = (raw >> 12) & 0x3FF;
  v.patch = raw & 0xFFF;
  v.text = Fmt("%u.%u.%u", v.major, v.minor, v.patch);
  return v;
}

// Session 95 (PLAIN review #6): a device that renders on the CPU: deviceType CPU, or llvmpipe / SwiftShader whatever it reports.
static bool IsCpuRenderer(const DeviceInfo& info) {
  const uint32_t id = EffectiveDriverId(info);
  return info.device_type == kTypeCpu || id == kDriverMesaLlvmpipe || id == kDriverGoogleSwiftshader ||
         Contains(info.device_name, "llvmpipe") || Contains(info.device_name, "swiftshader");
}

GpuClass Classify(const DeviceInfo& info) {
  if (IsCpuRenderer(info) || Contains(info.device_name, "gfxstream") || Contains(info.device_name, "venus")) {
    return GpuClass::kCpu;
  }
  // Session 95 (PLAIN): on Android every GPU that is not a CPU renderer is a phone, whatever its vendor or deviceType says
  // (an unknown vendor used to fall to kIgpu / kDesktop and so to the PC tiers). Not before the CPU block above:
  // gfxstream, SwiftShader and venus (emulators) stay CPU class. deviceType VIRTUAL_GPU is a phone here too (review #6);
  // on a PC it stays CPU class, as before.
  if (info.is_android) return GpuClass::kPhone;
  if (info.device_type == kTypeVirtual) return GpuClass::kCpu;
  if (info.device_type == kTypeDiscrete) return GpuClass::kDesktop;
  if (info.device_type == kTypeIntegrated) {
    if (IsMobileVendor(info.vendor_id)) return GpuClass::kPhone;  // also Adreno X1-85 laptops (matrix 4)
    if (info.vendor_id == kVendorIntel || info.vendor_id == kVendorAmd) return GpuClass::kIgpu;
    return info.is_android ? GpuClass::kPhone : GpuClass::kIgpu;  // other vendors: Android = phone
  }
  // deviceType OTHER: no information, go by the vendor.
  if (IsMobileVendor(info.vendor_id)) return GpuClass::kPhone;
  if (info.vendor_id == kVendorNvidia) return GpuClass::kDesktop;
  return GpuClass::kIgpu;
}

// Session 95 (PLAIN): the GPU families whose paths were run on real phones: Qualcomm stock, Turnip, Arm, Imagination.
bool IsKnownFamily(const DeviceInfo& info) {
  return info.vendor_id == kVendorQualcomm || info.vendor_id == kVendorArm || info.vendor_id == kVendorImagination ||
         IsTurnip(info);
}

Result Evaluate(const DeviceInfo& info, const Options& options) {
  Result r;
  r.version = DecodeDriverVersion(info);
  r.driver_id = EffectiveDriverId(info);
  r.cls = Classify(info);
  const DriverVersion& ver = r.version;
  const bool qcom = IsQualcommProprietary(info);
  const bool turnip = IsTurnip(info);
  const bool arm = info.vendor_id == kVendorArm && r.driver_id == kDriverArmProprietary;

  // Rule 3 (matrix): Arm erratum 3786926, dynamic rendering reuses stale state. r49p1-r49p2, r50p0-r51p0 [S13].
  if (arm && ver.arm_r >= 0) {
    const bool bad = (RpLe(49, 1, ver.arm_r, ver.arm_p) && RpLe(ver.arm_r, ver.arm_p, 49, 2)) ||
                     (RpLe(50, 0, ver.arm_r, ver.arm_p) && RpLe(ver.arm_r, ver.arm_p, 51, 0));
    if (bad) {
      AddSet(r, "mali_dynamic_rendering_errata", "vulkan_dynamic_rendering", "false",
             Fmt("Arm erratum 3786926 (vkCmdBeginRendering reuses stale state) on %s [S13, matrix 3 #3]",
                 ver.text.c_str()));
    }
  }

  // Session 94 (matrix 3 rule 4): classic render passes on every Qualcomm / Arm / Imagination GPU, behind its own switch.
  // (Rule 3 above already set it for the Arm erratum versions; a second decision for the same cvar is skipped.)
  if (info.vendor_id == kVendorQualcomm || info.vendor_id == kVendorArm || info.vendor_id == kVendorImagination) {
    // (its own rule name: a gpu_quirks_disable of the Arm erratum rule does not silence this one; a double set of the same value is harmless)
    if (options.tiler_classic_passes) {
      AddSet(r, "tiler_classic_passes", "vulkan_dynamic_rendering", "false",
             "tile-based GPU: classic render passes (matrix 3 rule 4; gpu_quirks_tiler_classic_passes, an A/B candidate)");
    } else {
      AddInfo(r, "tiler_classic_passes",
              "not applied (gpu_quirks_tiler_classic_passes=false): classic render passes would be used on this tile-based GPU "
              "(matrix 3 rule 4)");
    }
  }

  // Rule 6: custom border colours broken on Qualcomm proprietary and Turnip (yuzu/Eden removed the extension) [S4].
  if (qcom || turnip) {
    AddSet(r, "qcom_custom_border_color", "vulkan_allow_custom_border_color", "false",
           "VK_EXT_custom_border_color broken on Qualcomm / Turnip (yuzu, Eden drop it; a blanket claim, unmeasured "
           "here); the texture cache then uses its standard-colour fallback (we use it for two YCbCr-black borders "
           "only) [S4 483-488, matrix 3 #6; Session 95 Lane D: not advisory, the fallback is the supported path]");
  }

  // Session 95 (Lane D): sparse shared memory. The stock Adreno 830 is clean with it on (pilot_verdict.md:18); Turnip is the
  // measured exception, so it goes off there (a full 512 MiB buffer instead). Elsewhere the strict bit is logged so a report
  // shows what the driver promised: with residencyNonResidentStrict, a read of an unbound page returns 0 instead of undefined.
  if (turnip) {
    AddSet(r, "turnip_no_sparse", "vulkan_sparse_shared_memory", "false",
           "Mesa Turnip: sparse shared memory off, a full non-sparse buffer is created instead (Session 95 Lane D)");
  } else {
    AddInfo(r, "sparse_shared_memory",
            Fmt("vulkan_sparse_shared_memory is not changed by a rule (on by default; used only if the device has sparseResidencyBuffer, the [sparse] line says): residencyNonResidentStrict=%s (%s)",
                info.residency_non_resident_strict ? "true" : "false",
                info.residency_non_resident_strict ? "unbound sparse pages read as 0"
                                                   : "unbound sparse pages are undefined when read"));
  }
  // Rule 7: timeline semaphores. This SDK does not create any (grep of src and include), so nothing to switch off.
  if (qcom || turnip) {
    AddInfo(r, "qcom_timeline_semaphores",
            "timeline semaphores are unreliable on Qualcomm / Turnip [S4 933-940]; not used by this SDK, nothing to "
            "switch off (fences + binary semaphores only)");
  }

  // Rule 14: Qualcomm compiles pipelines serially in yuzu [S4 637-641]. Unmeasured for us -> advisory.
  if (qcom || turnip) {
    AddSet(r, "qcom_pipeline_threads", "vulkan_pipeline_creation_threads", "1",
           "yuzu compiles serially on Qualcomm, Dolphin turns multithreaded precompilation off on Android "
           "[S4 637-641, S1 #47, matrix 3 #14]; would slow the first-run compile on a fast phone: unmeasured",
           /*advisory=*/true);
  }

  // Session 95 (P8 part 1, out\s95\android_structural.md): attribute vertex input (gpu_native_vertex_input=2) with primitive restart off for index
  // buffers that hold no reset value, on class phone, behind the master switch gpu_quirks_vertex_input_phone (default off for now). Every PC (iGPU included, after the
  // Surface's paired A B B A of Session 95, stack 3) gets it from the game's TRUE NATIVE list (src\d3d_owned.h), not from this table.
  if (r.cls == GpuClass::kPhone) {
    if (options.vertex_input_phone) {
      const char* why = "phone-class GPU: vertex and index buffers read as real attributes (the 128 MiB storage window no longer limits vertex reads), "
                        "no primitive restart on draws without reset values (P8 part 1)";
      AddSet(r, "vertex_input_phone", "gpu_native_vertex_input", "2", why, false, true);
      AddSet(r, "vertex_input_phone", "gpu_native_index_no_restart", "true", why, false, true);
    } else {
      AddInfo(r, "vertex_input_phone",
              "not applied (gpu_quirks_vertex_input_phone=false): this phone-class GPU keeps reading vertices from the storage windows "
              "(gpu_native_vertex_input=2 and gpu_native_index_no_restart=true are the rule)");
    }
  } else if (r.cls == GpuClass::kIgpu) {
    AddInfo(r, "vertex_input_phone", "iGPU class: no rule here; the game's TRUE NATIVE list sets gpu_native_vertex_input on every PC (Session 95)");
  }

  // R2 Q5: geometry shaders avoided on phone tilers behind its own switch.
  if (r.cls == GpuClass::kPhone && info.geometry_shader) {
    if (options.mobile_no_gs) {
      const char* why = "phone-class tiler: geometry shaders avoided, primitive types expanded in the vertex shader "
                        "(R2 Q5, rank 2 Mali cost)";
      AddSet(r, "mobile_no_gs", "vulkan_force_expand_rectangle_lists_in_vs", "true", why, false, true);
      AddSet(r, "mobile_no_gs", "vulkan_force_expand_point_sprites_in_vs", "true", why, false, true);
      AddSet(r, "mobile_no_gs", "vulkan_force_convert_quad_lists_to_triangle_lists", "true", why, false, true);
      // Session 94 (nogs): TRUE NATIVE's typed draws take the VS-expanded host types with no geometry shader.
      AddSet(r, "mobile_no_gs", "gpu_native_vs_expand", "true", why, false, true);
    } else {
      AddInfo(r, "mobile_no_gs",
              "not applied (gpu_quirks_mobile_no_gs=false): geometry shaders stay in use on this GPU. Since Session 94 "
              "(gpu_native_vs_expand) TRUE NATIVE's typed draws take the VS-expanded host types, so the switch is "
              "available; it is held off by default until the milestone check passes");
    }
  }

  // Session 94 review (nogs): a GPU without the geometryShader feature already gets the vertex-shader expansion from the
  // primitive processor (vulkan\primitive_processor.cpp:55-63); without gpu_native_vs_expand those draws leave the native path.
  if (!info.geometry_shader) {
    AddSet(r, "no_gs_vs_expand", "gpu_native_vs_expand", "true",
           "no geometryShader feature: the primitive processor expands point / rectangle lists in the vertex shader; TRUE "
           "NATIVE's typed draws take those expanded types (Session 94)",
           false, true);
  }

  // Item 2: robustBufferAccess stays on on Arm (errata 3779191 / 3781554 make SSBO reads speculative) [S13].
  if (arm) {
    AddInfo(r, "arm_robust_buffer_access",
            "keep vulkan_robust_buffer_access=true on Arm: errata 3779191 / 3781554 (speculative SSBO reads, vertices "
            "shaded in groups of 4) [S13]; off only as a perf A/B");
  }

  // Item 8: extended dynamic state allow bits (nothing in the SDK uses EDS yet; idea 7 reads gpu_quirks_eds_allowed).
  {
    int bits = kEdsAll;
    std::string why;
    auto drop = [&](int bit, const char* what) {
      if (bits & bit) {
        bits &= ~bit;
        if (!why.empty()) why += "; ";
        why += what;
      }
    };
    if (qcom) {
      drop(kEds1, "Qualcomm: EDS1 broken [S4 490-495]");
      drop(kVertexInput, "Qualcomm: vertex input dynamic state broken [S4 617-623]");
      if (ver.minor >= 676 && ver.minor <= 679) drop(kEds2, "Qualcomm 512.676-679: EDS2 broken [S4 571-581]");
    }
    if (arm) drop(kEds1, "Arm: EDS1 broken [S4 524-530]");
    if (arm && ver.arm_r >= 38 && RpLe(ver.arm_r, ver.arm_p, 44, 0)) {
      if (!why.empty()) why += "; ";
      why += "Arm r38p0-r44p0: erratum 3809538, re-set every dynamic state after each bind [S13]";
    }
    if (info.vendor_id == kVendorSamsung) drop(kEds3ColorBlend, "Samsung: EDS3 colour blend broken [S5]");
    if (info.vendor_id == kVendorAmd && r.driver_id != kDriverMesaRadv)
      drop(kEds3ColorBlend, "AMD: EDS3 colour blend broken [S4 597-650, S5]");
    if (r.driver_id == kDriverIntelProprietaryWindows) drop(kVertexInput, "Intel Windows: VIDS disabled [S4 653-670]");
    if (r.driver_id == kDriverNvidiaProprietary) {
      const bool old = ver.major < 580 || (ver.major == 580 && (ver.minor < 119 || (ver.minor == 119 && ver.patch < 2)));
      if (old) drop(kVertexInput, "NVIDIA < 580.119.02: VIDS off [S5]");
    }
    r.eds_allowed = bits;
    if (bits != kEdsAll) {
      AddSet(r, "eds_block_list", "gpu_quirks_eds_allowed", Fmt("%d", bits).c_str(),
             why + " [matrix 3 #8]");
    } else {
      AddInfo(r, "eds_block_list", "gpu_quirks_eds_allowed=31 (nothing blocked for this device) [matrix 3 #8]");
    }
  }

  // Item 1: storage buffer limits and what the window split would be (logged; vulkan_shared_memory_window is
  // another builder's switch and is not touched here).
  if (info.max_storage_buffer_range) {
    constexpr uint64_t kShared = 512ull << 20;
    const uint64_t range = info.max_storage_buffer_range;
    // Session 95 (Lane D): the same arithmetic as command_processor.cpp's [storage-window] block (window = the largest power
    // of two within the limit, a stride of a quarter window, one storage buffer bound). The old text counted
    // ceil(512 / range) disjoint windows, which no code path uses.
    uint64_t w = kShared;
    while (w > range) w >>= 1;
    std::string plan;
    if (range >= kShared) {
      plan = "one 512 MiB binding, no window";
    } else if (w < (64ull << 20)) {
      plan = "below the 64 MiB the windows need: the split stays";
    } else {
      const uint64_t stride = w / 4;
      plan = Fmt("%llu windows of %llu MiB every %llu MiB, one storage buffer",
                 (unsigned long long)((kShared - w) / stride + 1), (unsigned long long)(w >> 20),
                 (unsigned long long)(stride >> 20));
    }
    AddInfo(r, "storage_limits",
            Fmt("real device limits: maxStorageBufferRange=%llu MiB maxPerStageDescriptorStorageBuffers=%u maxDescriptorSetStorageBuffersDynamic=%u "
                "minStorageBufferOffsetAlignment=%llu maxPushConstantsSize=%u; 512 MiB shared memory -> "
                "vulkan_shared_memory_window=auto gives (by this real limit; vulkan_emulate_limits is applied later): %s [matrix 1, 3 #1]",
                (unsigned long long)(range >> 20), info.max_per_stage_storage_buffers, info.max_dynamic_storage_buffers,
                (unsigned long long)info.min_storage_buffer_offset_alignment, info.max_push_constants, plan.c_str()));
  }
  // Items 9 / 10: formats, logged only (the depth format and BC decisions are already made per format query).
  if (info.formats.known) {
    std::string gaps;
    auto gap = [&](bool ok, const char* name) {
      if (!ok) {
        if (!gaps.empty()) gaps += ", ";
        gaps += name;
      }
    };
    gap(info.formats.d24s8, "no D24S8 (host depth is D32S8)");
    gap(info.formats.d32s8, "no D32S8");
    gap(info.formats.rg16_unorm_linear, "RG16 UNORM not linear-filterable");
    gap(info.formats.rgba16_unorm_linear, "RGBA16 UNORM not linear-filterable");
    gap(info.formats.r32f_linear, "R32F not linear-filterable");
    gap(info.formats.bc1_sampled, "no BCn (GPU decode / transcode path)");
    AddInfo(r, "formats",
            gaps.empty() ? std::string("all checked formats supported (D24S8, D32S8, RG16/RGBA16 UNORM and R32F linear, BC1)")
                         : gaps + " [matrix 3 #5, #9, #10]");
  }

  // Session 95 (PLAIN): gpu_plain_mode. auto = plain only on a phone-class GPU of no known family, or on a virtual GPU on Android
  // (review #6: any vendor, no real phone ran those paths; a CPU renderer stays out first); on = always (LADDER rung 1 and test runs);
  // off = never. The cvars are forced in gpu_quirks_apply.cpp (ForcePlain).
  {
    const bool known = IsKnownFamily(info);
    const bool virtual_gpu = info.is_android && info.device_type == kTypeVirtual && !IsCpuRenderer(info);
    if (options.plain_mode == "on") {
      r.plain = true;
      r.plain_why = "gpu_plain_mode=on";
    } else if (options.plain_mode == "off") {
      r.plain_why = "gpu_plain_mode=off";
    } else if (virtual_gpu) {
      r.plain = true;
      r.plain_why = "gpu_plain_mode=auto: virtual GPU on Android (vendor " + Fmt("0x%04X", info.vendor_id) + ")";
    } else if (r.cls == GpuClass::kPhone && !known) {
      r.plain = true;
      r.plain_why = "gpu_plain_mode=auto: phone-class GPU of no known family (vendor " + Fmt("0x%04X", info.vendor_id) + ")";
    } else {
      r.plain_why = std::string("gpu_plain_mode=auto: ") + (known ? "known family" : "not a phone-class GPU") + ", flags unchanged";
    }
  }
  return r;
}

namespace {

// "a.b.c.d" -> pieces; false when anything is not a number.
bool SplitDots(const std::string& s, std::vector<uint32_t>& out) {
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find('.', start);
    if (end == std::string::npos) end = s.size();
    if (end == start) return false;
    char* stop = nullptr;
    const std::string piece = s.substr(start, end - start);
    const unsigned long v = std::strtoul(piece.c_str(), &stop, 10);
    if (!stop || *stop) return false;
    out.push_back(uint32_t(v));
    start = end + 1;
  }
  return !out.empty();
}

}  // namespace

bool ParseEmulateGpu(const std::string& spec, DeviceInfo& info, std::string& error) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= spec.size()) {
    size_t colon = spec.find(':', start);
    if (colon == std::string::npos) colon = spec.size();
    parts.push_back(spec.substr(start, colon - start));
    start = colon + 1;
  }
  std::string type;
  if (parts.size() >= 4) {
    const std::string last = Lower(parts.back());
    if (last == "integrated" || last == "discrete" || last == "cpu") {
      type = last;
      parts.pop_back();
    }
  }
  if (parts.size() < 3 || parts[0].empty()) {
    error = "expected vendorhex:driverver:name[:type]";
    return false;
  }
  for (size_t i = 3; i < parts.size(); ++i) parts[2] += ":" + parts[i];
  if (parts[2].empty()) {
    error = "empty device name";
    return false;
  }
  char* stop = nullptr;
  const unsigned long vendor = std::strtoul(parts[0].c_str(), &stop, 16);
  if (!stop || *stop || parts[0].empty()) {
    error = "vendor is not hex";
    return false;
  }
  DeviceInfo e = info;
  e.vendor_id = uint32_t(vendor);
  e.device_name = parts[2];
  e.driver_name.clear();
  e.driver_info.clear();
  e.driver_id = kDriverUnknown;
  e.device_id = 0;
  e.driver_id = Contains(e.device_name, "turnip") ? kDriverMesaTurnip : kDriverUnknown;
  e.driver_id = EffectiveDriverId(e);
  const std::string& dv = parts[1];
  if (!dv.empty() && dv[0] == 'r' && dv.find('p') != std::string::npos) {
    int rv, pv;
    if (!ParseArmRp(dv, rv, pv)) {
      error = "bad rNNpM";
      return false;
    }
    e.driver_info = dv;
    e.driver_version = (uint32_t(rv) << 22) | (uint32_t(pv) << 12);
  } else if (dv.find('.') != std::string::npos) {
    std::vector<uint32_t> p;
    if (!SplitDots(dv, p)) {
      error = "bad dotted driver version";
      return false;
    }
    if (e.driver_id == kDriverIntelProprietaryWindows && p.size() == 2) {
      e.driver_version = (p[0] << 14) | (p[1] & 0x3FFF);
    } else if (e.driver_id == kDriverNvidiaProprietary && p.size() == 4) {
      e.driver_version = (p[0] << 22) | (p[1] << 14) | (p[2] << 6) | (p[3] & 0x3F);
    } else if (p.size() == 3) {
      e.driver_version = (p[0] << 22) | (p[1] << 12) | (p[2] & 0xFFF);  // 512 << 22 = bit 31 set (Qualcomm)
    } else {
      error = "dotted version needs 3 parts (4 for NVIDIA, 2 for Intel Windows)";
      return false;
    }
  } else {
    char* s2 = nullptr;
    const unsigned long v = std::strtoul(dv.c_str(), &s2, 0);
    if (!s2 || *s2 || dv.empty()) {
      error = "bad driver version";
      return false;
    }
    e.driver_version = uint32_t(v);
  }
  if (type == "integrated") e.device_type = kTypeIntegrated;
  else if (type == "discrete") e.device_type = kTypeDiscrete;
  else if (type == "cpu") e.device_type = kTypeCpu;
  else if (type.empty()) {
    if (e.vendor_id == kVendorNvidia || (e.vendor_id == kVendorAmd && Contains(e.device_name, "rx ")))
      e.device_type = kTypeDiscrete;
    else
      e.device_type = kTypeIntegrated;
  } else {
    error = "type must be integrated | discrete | cpu";
    return false;
  }
  info = e;
  return true;
}

std::string HeaderLine(const DeviceInfo& info, const Result& result) {
  return Fmt("[quirks] GPU %s vendor 0x%04X driver id=%u (%s) %s class %s", info.device_name.c_str(), info.vendor_id,
             result.driver_id, DriverIdName(result.driver_id), result.version.text.c_str(), GpuClassName(result.cls));
}

}  // namespace rex::ui::vulkan::quirks
