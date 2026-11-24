// WWE '13 port: 2K14's renderer adapter reader is not part of this port (always off).
#pragma once
#include <cstdint>
#include <rex/cvar.h>
#include <rex/ppc/context.h>
REXCVAR_DECLARE(bool, wwe2k14_rdr_readback_defer);
namespace wwe2k14::rdr_reader {
inline bool ReaderOn() { return false; }
struct Times { uint64_t old_draws = 0, old_copies = 0; };
inline Times g_times;
struct ScopedNs { explicit ScopedNs(uint64_t&) {} };
inline void OnFloatFlush(uint64_t, uint32_t) {}
inline bool OnDrawIndexed(PPCContext&, uint8_t*) { return false; }
inline void OnBeginVertices(PPCContext&, uint8_t*) {}
inline void OnEndVertices(PPCContext&, uint8_t*) {}
inline void OnGammaRamp(PPCContext&, uint8_t*) {}
inline void OnPresent(PPCContext&, uint8_t*) {}
inline void KoLatchAtPresent() {}
inline void OnClear(PPCContext&, uint8_t*) {}
inline void OnResolve(PPCContext&, uint8_t*) {}
}  // namespace wwe2k14::rdr_reader
namespace wwe2k14::thumb_defer {
inline void OnPresent(PPCContext&, uint8_t*) {}
}
