#!/usr/bin/env bash
# Builds WWE13-Android.apk from libmain.so (no Gradle: aapt2 + javac + d8 + apksigner).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SO=${1:?path to libmain.so}
OUT=${2:-$HERE/out/WWE13-Android.apk}
SDK=${ANDROID_SDK:?set ANDROID_SDK to your Android SDK (platforms/android-35, build-tools/35.0.0)}
ANDROID_JAR=$SDK/platforms/android-35/android.jar
D8=$SDK/build-tools/35.0.0/lib/d8.jar
APKSIGNER=$SDK/build-tools/35.0.0/lib/apksigner.jar
AAPT2=${AAPT2:-$SDK/build-tools/35.0.0/aapt2}
SDLJAVA=$HERE/../../sdk/thirdparty/sdl3/android-project/app/src/main/java
W=$HERE/out/work
rm -rf "$W"; mkdir -p "$W/classes" "$W/dex" "$W/res" "$W/apk/lib/arm64-v8a"

"$AAPT2" compile --dir "$HERE/res" -o "$W/res/res.zip"
"$AAPT2" link -o "$W/base.apk" -I "$ANDROID_JAR" --manifest "$HERE/AndroidManifest.xml" "$W/res/res.zip"

# EXTRAS=1: the development build with mods, custom music, save sync and the testing options (Features.java).
cp -r "$HERE/java" "$W/src"
[ "${EXTRAS:-0}" = 1 ] && sed -i 's/EXTRAS = false/EXTRAS = true/' "$W/src/com/toni/wwe13/Features.java"
javac -nowarn -encoding UTF-8 --release 8 -cp "$ANDROID_JAR" -d "$W/classes" \
  $(find "$SDLJAVA" "$W/src" -name '*.java') 2>&1 | grep -v "^Note:\|warning: \[options\]\|JAVA_TOOL_OPTIONS" || true
[ -f "$W/classes/com/toni/wwe13/LauncherActivity.class" ] || { echo "javac failed"; exit 1; }
java -cp "$D8" com.android.tools.r8.D8 --release --min-api 33 --lib "$ANDROID_JAR" --output "$W/dex" \
  $(find "$W/classes" -name '*.class')

cp "$W/base.apk" "$W/unaligned.apk"
cp "$W/dex/classes.dex" "$W/apk/classes.dex"
cp "$SO" "$W/apk/lib/arm64-v8a/libmain.so"
# adrenotools hook libraries (custom GPU drivers), built next to libmain.so.
for hook in ${HOOK_DIR:-$HERE/../../sdk/out/android-arm64-v8a}/lib{main_hook,file_redirect_hook,gsl_alloc_hook,hook_impl}.so; do
  cp "$hook" "$W/apk/lib/arm64-v8a/"
done
cp ${LAUNCHER_BUILD:-$HERE/../launcher-native/build}/libwwe13launcher.so "$W/apk/lib/arm64-v8a/"
# New SDK: shared runtime + GPU plugin next to libmain.so (EXTRA_LIBS="a.so b.so").
for extra in ${EXTRA_LIBS:-}; do
  cp "$extra" "$W/apk/lib/arm64-v8a/"
  llvm-strip-18 --strip-debug "$W/apk/lib/arm64-v8a/$(basename "$extra")"
done
llvm-strip-18 --strip-debug "$W/apk/lib/arm64-v8a/libmain.so" "$W/apk/lib/arm64-v8a/libwwe13launcher.so"
# Bundled first-run files (optional, not in the repo): a Turnip GPU driver (Mesa, MIT) and a prebuilt shader cache.
# ASSETS=<dir> with drivers/libvulkan_freedreno.so and/or shader-cache/*.
A=${ASSETS:-$HERE/../assets}
if [ -f "$A/drivers/libvulkan_freedreno.so" ]; then
  mkdir -p "$W/apk/assets/drivers"; cp "$A/drivers/libvulkan_freedreno.so" "$W/apk/assets/drivers/libvulkan_freedreno.bin"
fi
if [ -d "$A/shader-cache" ]; then mkdir -p "$W/apk/assets/shader-cache"; cp "$A"/shader-cache/* "$W/apk/assets/shader-cache/"; fi
python3 -I "$HERE/zipalign.py" "$W/unaligned.apk" "$W/apk" "$W/aligned.apk"

KS=$HERE/debug.keystore
if [ ! -f "$KS" ]; then
  keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias wwe13 \
    -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=WWE13 Android port" >/dev/null 2>&1
fi
mkdir -p "$(dirname "$OUT")"
java -jar "$APKSIGNER" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android \
  --ks-key-alias wwe13 --min-sdk-version 33 --out "$OUT" "$W/aligned.apk"
ls -la "$OUT"
