/*
 * Compilation unit for the vendored single-file audio decoders.
 *
 * The decoder headers are implementation-inlined C; this TU exists so they
 * compile exactly once with warnings off (see thirdparty/CMakeLists.txt).
 */

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// stb_vorbis.c wraps its own declarations/definitions in extern "C".
// Define the implementation here directly (not via STB_VORBIS_HEADER_ONLY)
// so the header half is emitted exactly once, from this TU.
#include "stb_vorbis.c"
