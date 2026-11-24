// wwe2k14 patch (Session 93, idea 6): the driver quirks table.
//
// Pure logic, no Vulkan and no cvar calls, so it can be tested offline (tools\native\quirks_test in the game repo
// compiles gpu_quirks.cpp alone). The device code (vulkan_device.cpp) fills a DeviceInfo from the real device,
// calls Evaluate(), and gpu_quirks_apply.cpp turns the decisions into cvar values and log lines.
//
// Sources: docs\research\gpu_matrix_s93.md (the game repo). Source tags [Sn] in the reasons point to its section 5.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rex::ui::vulkan::quirks {

enum Vendor : uint32_t {
  kVendorAmd = 0x1002,
  kVendorSamsung = 0x144D,
  kVendorNvidia = 0x10DE,
  kVendorIntel = 0x8086,
  kVendorQualcomm = 0x5143,
  kVendorArm = 0x13B5,
  kVendorImagination = 0x1010,
};

// VkDriverId values (Khronos reference page; the matrix's section 2.0, [S10]).
enum DriverId : uint32_t {
  kDriverUnknown = 0,
  kDriverAmdProprietary = 1,
  kDriverAmdOpenSource = 2,
  kDriverMesaRadv = 3,
  kDriverNvidiaProprietary = 4,
  kDriverIntelProprietaryWindows = 5,
  kDriverIntelOpenSourceMesa = 6,
  kDriverImaginationProprietary = 7,
  kDriverQualcommProprietary = 8,
  kDriverArmProprietary = 9,
  kDriverMesaLlvmpipe = 13,
  kDriverMoltenVk = 14,
  kDriverGoogleSwiftshader = 16,
  kDriverMesaTurnip = 18,
  kDriverSamsungProprietary = 21,
};

// VkPhysicalDeviceType values.
enum DeviceType : uint32_t { kTypeOther = 0, kTypeIntegrated = 1, kTypeDiscrete = 2, kTypeVirtual = 3, kTypeCpu = 4 };

enum class GpuClass { kDesktop, kIgpu, kPhone, kCpu };
const char* GpuClassName(GpuClass c);

// Format support the device code queried (optimal tiling). Only logged, no rule depends on it yet.
struct FormatSupport {
  bool known = false;
  bool d24s8 = false;               // depth/stencil attachment + sampled
  bool d32s8 = false;
  bool rg16_unorm_linear = false;   // R16G16 UNORM sampled + linear filter
  bool rgba16_unorm_linear = false;
  bool r32f_linear = false;
  bool bc1_sampled = false;
};

struct DeviceInfo {
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t device_type = kTypeOther;
  uint32_t driver_id = kDriverUnknown;  // VkDriverId, 0 = unknown / no VK_KHR_driver_properties
  uint32_t driver_version = 0;          // VkPhysicalDeviceProperties::driverVersion
  uint32_t api_version = 0;
  std::string device_name;
  std::string driver_name;
  std::string driver_info;
  bool is_android = false;
  // Limits (the matrix's item 1).
  uint64_t max_storage_buffer_range = 0;
  uint32_t max_per_stage_storage_buffers = 0;
  uint32_t max_dynamic_storage_buffers = 0;
  uint64_t min_storage_buffer_offset_alignment = 0;
  uint32_t max_push_constants = 0;
  bool geometry_shader = true;
  // Session 95 (Lane D): VkPhysicalDeviceSparseProperties::residencyNonResidentStrict (unbound sparse pages read 0).
  bool residency_non_resident_strict = false;
  // Memory (heaps, bytes).
  uint64_t device_local_bytes = 0;
  bool device_local_host_visible = false;
  FormatSupport formats;
};

struct DriverVersion {
  uint32_t major = 0, minor = 0, patch = 0, build = 0;
  int arm_r = -1, arm_p = -1;  // Arm "rNNpM", -1 = not known
  std::string text;
};

// The vendor's own split of driverVersion (the matrix's section 2.0).
DriverVersion DecodeDriverVersion(const DeviceInfo& info);
const char* DriverIdName(uint32_t id);
// driver_id when known, else inferred from the vendor (and "turnip" in the name).
uint32_t EffectiveDriverId(const DeviceInfo& info);
GpuClass Classify(const DeviceInfo& info);

// A rule's outcome. kSetCvar = set `cvar` to `value` if it is still at its default; kInfo = logged only.
struct Decision {
  enum Kind { kSetCvar, kInfo } kind = kInfo;
  std::string rule;
  std::string cvar;
  std::string value;
  std::string reason;      // why, with source tags
  bool advisory = false;   // unmeasured: applied only with gpu_quirks_advisory=true
  bool init_only = false;  // the cvar is an init-only flag (set through the lifecycle override)
};

// Bits of gpu_quirks_eds_allowed (the matrix's item 8). The pipeline code reads them when it adopts the extensions.
enum EdsBits : int {
  kEds1 = 1,
  kEds2 = 2,
  kEds3 = 4,
  kVertexInput = 8,
  kEds3ColorBlend = 16,
  kEdsAll = 31,
};

struct Options {
  bool mobile_no_gs = false;  // gpu_quirks_mobile_no_gs
  bool tiler_classic_passes = false;  // gpu_quirks_tiler_classic_passes (Session 94)
  std::string plain_mode = "auto";    // gpu_plain_mode (Session 95): auto | on | off
  bool vertex_input_phone = false;    // gpu_quirks_vertex_input_phone (Session 95, P8 part 1)
};

struct Result {
  DriverVersion version;
  uint32_t driver_id = 0;
  GpuClass cls = GpuClass::kDesktop;
  int eds_allowed = kEdsAll;
  bool plain = false;       // Session 95 (PLAIN): force the boring flag set (gpu_quirks_apply.cpp ForcePlain)
  std::string plain_why;
  std::vector<Decision> decisions;
};

// Session 95 (PLAIN): Qualcomm stock, Turnip, Arm, Imagination = the families run on real phones.
bool IsKnownFamily(const DeviceInfo& info);

Result Evaluate(const DeviceInfo& info, const Options& options);

// "vendorhex:driverver:name[:type]" (vulkan_emulate_gpu). driverver = a raw number (0x.. or decimal driverVersion),
// "rNNpM" (Arm), or dotted ("512.800.55" Qualcomm / Mesa / standard split, "101.2112" Intel, "580.119.2.0" NVIDIA).
// type = integrated | discrete | cpu (default: guessed from the vendor and the name). Overwrites the identity fields
// of `info` (vendor, device name, driver version / info / id, type). Returns false (and `error`) when malformed.
bool ParseEmulateGpu(const std::string& spec, DeviceInfo& info, std::string& error);

// "[quirks] GPU <name> vendor 0x.... driver id=<n> (<NAME>) <version> class <class>"
std::string HeaderLine(const DeviceInfo& info, const Result& result);

}  // namespace rex::ui::vulkan::quirks
