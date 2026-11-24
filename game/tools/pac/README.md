# WWE '13 .pac tools

Build the LZX helpers (libmspack from jsdk/thirdparty):
    L=../../../jsdk/thirdparty/libmspack/libmspack/mspack
    gcc -O2 -DPROBE -I$L -o xlzx  xlzx.c $L/lzxd.c
    gcc -O2 -DPROBE -I$L -o xlzxd xlzx.c lzxd_dbg.c   # lzxd_dbg.c = lzxd.c printing block lengths

Use:
    python3 pactool.py list   ch188.pac
    python3 pactool.py unpack ch188.pac out/

Layers: XCTD/LZX (unxctd.py) -> EPK8 directory -> PACH archives -> BPE entries (bpe.py).
