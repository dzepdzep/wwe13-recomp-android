# WWE '13 Recomp for Android

An Android port of the **WWE '13 static recompilation** ([PromethenDev/wwe13-recomp](https://github.com/PromethenDev/wwe13-recomp)).
The Xbox 360 game code is translated ahead of time to native ARM64 code, so this is not an emulator. It runs at 60 FPS on
a Snapdragon 8 Gen 2 handheld (AYN Odin 2 Portal).

**No game files are included.** You need your own copy of the Xbox 360 version of WWE '13.

## What works
- All main modes tested: exhibition matches (one-on-one, tag, ladder, steel cage, Hell in a Cell, Royal Rumble), Attitude
  Era, WWE Universe.
- Creations: Create-a-Superstar, entrances, Special Moves, Create-a-Story, Create-an-Arena.
- **Custom entrance music:** add your own MP3, FLAC, OGG or WAV files in the launcher, then pick them in the game under
  Creations → Entrance → Finalize → Music → User Playlist.
- Title update 2.0.1.0 and the DLC packs (if you own them).
- Controllers (Bluetooth or built-in handheld controls).
- **Touch controls** made for WWE '13: GRAPPLE, STRIKE, WHIP / PIN, FINISHER, REVERSAL, RUN (tap to switch on or off),
  CLIMB / TAG, LIMB / ITEM, a taunt d-pad and a small right stick. They hide while a controller is connected (launcher
  setting: Automatic / Always on / Off, plus opacity and vibration). Tap ✎ in the game's top-right corner to move and
  resize the buttons.

## Requirements
- An Android device with a Snapdragon (Adreno) GPU, Android 13 or newer, 64-bit. Tested on the AYN Odin 2 Portal.
- About 9 GB of free space for the installed game (plus room for your disc image while installing).
- Your own WWE '13 **Xbox 360 disc image** (`.iso`) and **title update 2.0.1.0**. DLC is optional.

## Install
1. Download the APK from [Releases](../../releases) and install it.
2. Open **WWE '13**. In the launcher, tap **Install from my disc image (.iso)…** and pick your image. The game is copied
   into the app, so you can delete the image afterwards.
3. Tap **Add title update or DLC…** and pick the title update (and any DLC you own).
4. Tap **Play**.

## Building from source
This is for developers; the build is not simple. You need a Linux host with clang/LLVM 18–20, CMake, Ninja, Java 17,
the Android SDK (platform 35, build-tools 35.0.0) and NDK r27.

1. `tools/fetch_thirdparty.sh` fills `sdk/thirdparty/` (ReXGlue SDK v0.10.1 submodules and libadrenotools).
2. Build the SDK for the host (`sdk/`, the `rexglue` codegen tool), then run the codegen on your own `default.xex` with
   `game/wwe13_config.toml` (see `game/tools/host_rexglue.sh`). This writes `game/generated/`; generated code is not
   in this repo because it is made from the game.
3. Build the SDK and `game/` for Android arm64 with `android/android-arm64-j.cmake` (set `ANDROID_NDK` and
   `ANDROID_NDK_HOST`), and the launcher library in `android/launcher-native/`.
4. Make the APK: `ANDROID_SDK=… EXTRA_LIBS="librexruntime.so librexgpu-xenos.so" android/apk/build_apk.sh libmain.so out.apk`.
   `EXTRAS=1` builds the development version with the unfinished features (mods, save sync, training mode).

## Layout
- `game/` – the WWE '13 project: game hooks and fixes (`src/`), config, tools.
- `sdk/` – a fork of the ReXGlue SDK with the Android port (Vulkan on Adreno, audio, input, custom music).
- `android/` – the launcher app (Java), the APK build script, the native launcher library and toolchain files.
- `tools/` – helper scripts.

## Credits
This port stands on the work of many people. Thank you!

- **[PromethenDev/wwe13-recomp](https://github.com/PromethenDev/wwe13-recomp)** (BSD-3-Clause) – the WWE '13 recompilation
  this port is built on: the game project, fixes and the launcher core (`game/launcher2`).
- **[ReXGlue SDK](https://github.com/rexglue/rexglue-sdk)** by Tom Clay and contributors (BSD-3-Clause) – the Xbox 360
  recompilation toolkit and runtime (`sdk/` is a fork of v0.10.1).
- **[Xenia](https://xenia.jp)** by Ben Vanik and the Xenia project contributors (BSD-3-Clause) – ReXGlue's runtime is based
  in part on Xenia.
- **[Jarrodfeng/wwe2k14-recomp](https://github.com/Jarrodfeng/wwe2k14-recomp)** (BSD) – its D3D layer and SDK work
  (Half 2, no-ring command path, Android port pieces) were ported into this project.
- **[HollywoodAkeem/rexglue-sdk-yukes](https://github.com/HollywoodAkeem/rexglue-sdk-yukes)** – the ReXGlue fork with the
  Yuke's engine fixes that wwe13-recomp was built with.
- **[libadrenotools](https://github.com/bylaws/libadrenotools)** by Billy Laws (BSD-2-Clause) – custom GPU driver loading.
- **[Mesa / Turnip](https://mesa3d.org)** (MIT) – the open-source Adreno Vulkan driver bundled in the APK.
- **[dr_libs](https://github.com/mackron/dr_libs)** by David Reid (MIT-0 / public domain) and
  **[stb_vorbis](https://github.com/nothings/stb)** by Sean Barrett (MIT / public domain) – custom music decoding.
- **[FFmpeg](https://ffmpeg.org)** (LGPL-2.1; source used: [wmarti/FFmpeg](https://github.com/wmarti/FFmpeg),
  branch `xenia-ffmpeg-canary-full`) – XMA audio decoding, built into `librexruntime.so`.
- **[SDL3](https://libsdl.org)** (zlib), **[Vulkan headers/loader, glslang, SPIRV-Tools](https://github.com/KhronosGroup)**,
  **[fmt](https://github.com/fmtlib/fmt)**, **[spdlog](https://github.com/gabime/spdlog)**, **[miniz](https://github.com/richgel999/miniz)**,
  **[xxHash](https://github.com/Cyan4973/xxHash)**, **[toml++](https://github.com/marzer/tomlplusplus)** and the other libraries
  in `sdk/thirdparty/` – each under its own licence.

## Legal
This project contains no game code or assets from WWE '13. It is not affiliated with or endorsed by WWE, 2K, THQ, Yuke's
or Microsoft. WWE '13 and all related names and logos are trademarks of their owners. Use only game files you own.
The recompiled code in the APK is generated from the game's executable and only works with your own copy of the game.
