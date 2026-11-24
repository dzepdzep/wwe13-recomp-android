#!/bin/sh
# k14port for this machine (cc/c++) or, with NDK=<ndk dir>, for Android arm64 (as libk14port.so: an executable the
# launcher runs from its native library folder).
set -e
cd "$(dirname "$0")"
OUT=${OUT:-./k14port}
if [ -n "$NDK" ]; then
  T=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
  R=$(ls -d $NDK/toolchains/llvm/prebuilt/*/lib/clang/* | head -1)
  CC="$T/clang --target=aarch64-none-linux-android33 -resource-dir=$R -fPIE"; CXX="$T/clang++ --target=aarch64-none-linux-android33 -resource-dir=$R -fPIE -stdlib=libc++"
  EXTRA="-static-libstdc++ -pie -fuse-ld=lld --rtlib=compiler-rt --unwindlib=libunwind -Wl,-z,max-page-size=16384"
else
  CC=${CC:-cc}; CXX=${CXX:-c++}; EXTRA=""
fi
O=$(mktemp -d)
$CC -O2 -c -Imspack -w lzxd.c -o $O/lzxd.o
$CC -O2 -c -Imspack -w mspack/system.c -o $O/system.o
$CC -O2 -c -I. -w -DKB2D_LIB kb2d_enc_lib.c -o $O/kb2d.o
$CXX -O2 -std=c++17 -c -Imspack k14port_io.cpp -o $O/io.o
$CXX -O2 -std=c++17 -c -Imspack k14port.cpp -o $O/main.o
$CXX $EXTRA -o "$OUT" $O/*.o -lz -lm
rm -rf $O
echo "built $OUT"
