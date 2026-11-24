#!/usr/bin/env bash
# Fills sdk/thirdparty/ with the libraries the SDK builds against (not stored in this repo):
#   - the submodules of the official ReXGlue SDK at v0.10.1 (546a082), the release this SDK fork is based on;
#   - libadrenotools (custom GPU driver loading on Android), commit 8fae8ce, with its submodules.
# Files already in sdk/thirdparty/ (this fork's own, e.g. decoders/) are kept.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TP=$ROOT/sdk/thirdparty
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone -q https://github.com/rexglue/rexglue-sdk.git "$TMP/rexglue"
git -C "$TMP/rexglue" checkout -q 546a082a4356173635f6c40510a41ff53e518b5a
git -C "$TMP/rexglue" submodule update -q --init --recursive
cp -rn "$TMP/rexglue/thirdparty/." "$TP/"

git clone -q https://github.com/bylaws/libadrenotools.git "$TP/libadrenotools"
git -C "$TP/libadrenotools" checkout -q 8fae8ce254dfc1344527e05301e43f37dea2df80
git -C "$TP/libadrenotools" submodule update -q --init --recursive
echo "sdk/thirdparty is ready."
