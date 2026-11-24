# Vendored single-header audio decoders

Brought in for the XMP custom-music player (`src/audio/custom_music_player.cpp`).
Rationale: the vendored FFmpeg build only enables the XMA/WMA-Pro and MP3
decoders (plus the MP3 demuxer). Enabling FLAC/Vorbis decoders and the
OGG/WAV demuxers there means regenerating the per-platform
`config_*.h`, `codec_list.c`/`demuxer_list.c` overlays and extending the
explicit source lists in `thirdparty/CMakeLists.txt` - a large, cross-platform
build change for four small formats. Single-file decoders are contained to one
new static library and are trivially available on both Linux and Windows.

| File | Upstream | Version | sha256 | Licence |
|---|---|---|---|---|
| `dr_mp3.h` | https://github.com/mackron/dr_libs (`dr_mp3.h`, master 2026-09-29) | v0.7.4 | `997b7ee18de6e6b81e2a83f1ea9fc62aef25c62b28d48db95635f49e65de0a2f` | public domain / MIT-0 (choice) |
| `dr_flac.h` | https://github.com/mackron/dr_libs (`dr_flac.h`, master 2026-09-29) | v0.13.4 | `111144e778f55738db6851cb226015c419e00d04b916a09506d4856d9cff945c` | public domain / MIT-0 (choice) |
| `dr_wav.h` | https://github.com/mackron/dr_libs (`dr_wav.h`, master 2026-09-29) | v0.14.6 | `03e70c1a2d9787cd7ed3e966c075bea7bac6373f759db9cd7ca9ccdfc4ec4493` | public domain / MIT-0 (choice) |
| `stb_vorbis.c` | https://github.com/nothings/stb (`stb_vorbis.c`, master 2026-09-29) | v1.22 | `4c7cb2ff1f7011e9d67950446b7eb9ca044f2e464d76bfbb0b84dd2e23e65636` | MIT / public domain (choice) |

Full licence texts are at the bottom of each file, unmodified.

The versions are the ones in the upstream files themselves: dr_libs master
does not carry a fixed release tag in the header banner beyond
`v0.7.4 - TBD` / `v0.13.4 - TBD` / `v0.14.6 - TBD`, and stb_vorbis.c says `v1.22`.
