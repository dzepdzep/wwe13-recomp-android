#!/usr/bin/env bash
# Runs the ReXGlue codegen built from ../../sdk on a Linux host without X11: libxstub.so (built from hostfix/xstub.c with
# `cc -shared -fPIC -o hostfix/libxstub.so hostfix/xstub.c`) stubs the three xcb/X11 calls the tool links against.
HERE=$(cd "$(dirname "$0")" && pwd)
LD_PRELOAD=$HERE/hostfix/libxstub.so exec "${REXGLUE:-$HERE/../../sdk/out/linux-amd64/rexglue}" "$@"
