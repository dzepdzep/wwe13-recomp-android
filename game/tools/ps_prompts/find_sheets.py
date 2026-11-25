#!/usr/bin/env python3
"""Finds textures with Xbox button icons in unpacked pac folders (from the user's own disc).

  find_sheets.py <out_png_dir> <dir>...
Scans every file for DDS textures (DXT1/DXT3/DXT5/A8R8G8B8), decodes them and keeps the ones that hold the four Xbox
face-button colours (green A, red B, blue X, yellow Y) as solid round marks. Writes a PNG of each hit and prints
size, format and the FNV-1a hash of the pixel data (the key ps_prompts.cpp matches on).
"""
import os, struct, sys
import numpy as np
from PIL import Image


def fnv(b):
    h = 0xcbf29ce484222325
    for c in b:
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv_np(b):
    # fast enough for a few MB
    h = 0xcbf29ce484222325
    for c in bytes(b):
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def rgb565(v):
    r = ((v >> 11) & 31).astype(np.int32); g = ((v >> 5) & 63).astype(np.int32); b = (v & 31).astype(np.int32)
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)


def decode_bc(data, w, h, fmt):
    bw, bh = max(1, w // 4), max(1, h // 4)
    bs = 8 if fmt == b'DXT1' else 16
    n = bw * bh
    blk = np.frombuffer(data[:n * bs], np.uint8).reshape(n, bs)
    cb = blk[:, -8:]
    c0 = cb[:, 0].astype(np.uint16) | (cb[:, 1].astype(np.uint16) << 8)
    c1 = cb[:, 2].astype(np.uint16) | (cb[:, 3].astype(np.uint16) << 8)
    e0, e1 = rgb565(c0), rgb565(c1)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (2 * e0 + e1) // 3, (e0 + e1) // 2)
    p3 = np.where(four, (e0 + 2 * e1) // 3, 0)
    pal = np.stack([e0, e1, p2, p3], 1)  # n,4,3
    bits = cb[:, 4].astype(np.uint32) | (cb[:, 5].astype(np.uint32) << 8) | (cb[:, 6].astype(np.uint32) << 16) | (cb[:, 7].astype(np.uint32) << 24)
    idx = np.stack([(bits >> (2 * i)) & 3 for i in range(16)], 1)
    rgb = np.take_along_axis(pal, idx[:, :, None].astype(np.int64).repeat(3, 2), 1)  # n,16,3
    if fmt == b'DXT1':
        a = np.where((~four) & (idx == 3), 0, 255)
    elif fmt == b'DXT3':
        ab = blk[:, :8]
        a = np.stack([((ab[:, i // 2] >> (4 * (i % 2))) & 15) * 17 for i in range(16)], 1)
    else:
        a0 = blk[:, 0].astype(np.int32); a1 = blk[:, 1].astype(np.int32)
        lv = [a0, a1]
        for i in range(1, 7):
            lv.append(np.where(a0 > a1, ((7 - i) * a0 + i * a1) // 7, ((5 - i) * a0 + i * a1) // 5 if i < 5 else (0 if i == 5 else 255)))
        lv = np.stack(lv, 1)
        ab = np.zeros(n, np.uint64)
        for i in range(6):
            ab |= blk[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
        ai = np.stack([((ab >> np.uint64(3 * i)) & np.uint64(7)).astype(np.int64) for i in range(16)], 1)
        a = np.take_along_axis(lv, ai, 1)
    px = np.concatenate([rgb, a[:, :, None]], 2).astype(np.uint8).reshape(bh, bw, 4, 4, 4)
    return px.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)


def xbox_score(img):
    r, g, b, a = [img[:, :, i].astype(int) for i in range(4)]
    vis = a > 128
    def frac(m):
        return int((m & vis).sum())
    green = frac((g > 120) & (g > r + 50) & (g > b + 50))
    red = frac((r > 150) & (r > g + 80) & (r > b + 80))
    blue = frac((b > 150) & (b > r + 60) & (b > g + 20))
    yellow = frac((r > 170) & (g > 140) & (b < 90))
    return green, red, blue, yellow


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    seen = set()
    for root in sys.argv[2:]:
        for dp, dn, fn in os.walk(root):
            if '.orig' in dp:
                continue
            for f in fn:
                p = os.path.join(dp, f)
                d = open(p, 'rb').read()
                i = 0
                while True:
                    i = d.find(b'DDS |\x00\x00\x00', i)
                    if i < 0:
                        break
                    hd = d[i:i + 128]
                    i += 4
                    if len(hd) < 128:
                        break
                    h, w = struct.unpack('<II', hd[12:20])
                    fourcc = hd[84:88]
                    if fourcc not in (b'DXT1', b'DXT3', b'DXT5') or w < 64 or h < 64 or w > 4096 or h > 4096:
                        continue
                    size = w * h // (2 if fourcc == b'DXT1' else 1)
                    px = d[i - 4 + 128:i - 4 + 128 + size]
                    if len(px) < size:
                        continue
                    key = (w, h, px[:4096], len(px))
                    if key in seen:
                        continue
                    seen.add(key)
                    img = decode_bc(px, w, h, fourcc)
                    s = xbox_score(img)
                    if min(s) >= 20:
                        hsh = fnv(px)
                        name = '%s_%dx%d_%s_%016x.png' % (os.path.relpath(p, root).replace('/', '_'), w, h, fourcc.decode(), hsh)
                        Image.fromarray(img).save(os.path.join(out, name))
                        print(name, 'off', hex(i - 4), 'score', s, flush=True)


if __name__ == '__main__':
    main()
