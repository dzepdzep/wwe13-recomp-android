# k14port

Ports the WWE 2K14 superstars WWE '13 doesn't have into a WWE '13 mod folder, on the device, from the player's own
discs (their installed WWE '13 files and their WWE 2K14 disc image). The launcher runs it as `libk14port.so` from
its native library folder ("Port from my WWE 2K14 disc"). Nothing from either game is in this tool or its output
until it runs on the player's discs.

    k14port --w13 <WWE '13 folder> --iso <WWE 2K14 .iso> --out <mod folder> [--no-titantrons] [--ids 360,361]

Steps (the same as the PC tools in tools/pac, tools/sound, tools/video): models and select pictures, names
(string.pac), titantrons (Bink 2 KB2f -> KB2d, kb2d_enc), entrance music (Wwise carrier numbers), entrances
(EVP/EVZ with the motion names cut to 5 characters), movesets and ratings (roster.txt fields 9 and 10).

Build: `./build.sh` (this machine) or `NDK=<ndk> OUT=libk14port.so ./build.sh` (Android arm64).

Third-party code: `lzxd.c` and `mspack/` are libmspack (LGPL 2.1, Stuart Caie), with a block-length hook added;
`bink2_decode.c`/`.h` and `bink2/` are the LGPL 2.1+ Bink 2 decoder (Paul B Mahol's FFmpeg patch, from the
WWE12-PC project). Their notices are in the files.
