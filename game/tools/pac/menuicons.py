"""Select-screen pictures for a ported wrestler: copies the wrestler's portraits (SSFA/SSFB/SSFC/<id>) and grid icon
(entry <id> of the icon pack MENU/SSFD) from WWE 2K14's pac/menu/menuHD.pac into WWE '13's.

WWE '13 menuHD: SSFA/SSFB = big renders (512x512 DDS), SSFC = small (256x256), all BPE and keyed by the character id;
MENU/SSFD = PACH(1) of a BPE'd PACH of per-id icons (DDS, 0x1080 bytes; offsets from the end of the PACH table,
16-byte aligned). WWE 2K14 has the same layout (its SSFD inner pack is ZLIB).

  python3 menuicons.py <w13 menuHD.pac> <2k14 menuHD.pac> <out menuHD.pac> <id> [<id> ...]
Then: plist.py replace <plist360.def> <plist360.arc> 'pac\\menu\\menuHD.pac' <out> <out.arc>
"""
import os, struct, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pactool import load, epk8_entries
from bpe import bpe_expand
from bpec import bpe_wrap_c
from epkadd import add_entries


def entry(d, g, n):
    for gg, nn, o, s, _, _ in epk8_entries(d):
        if gg == g and nn == n:
            return d[o:o + s]
    return None


def trim_bpe(b):
    return b[:16 + struct.unpack('<I', b[8:12])[0]] if b[:4] == b'BPE ' else b


def unpack_any(b):
    """BPE or ZLIB part -> bytes."""
    if b[:4] == b'BPE ':
        usz = struct.unpack('<I', b[12:16])[0]
        out, _ = bpe_expand(b[16:16 + struct.unpack('<I', b[8:12])[0]], usz, big_endian_size=False)
        return out
    if b[:4] == b'ZLIB':
        return zlib.decompress(b[16:])
    return b


def pach_parse(d):
    n = struct.unpack('<I', d[4:8])[0]
    ds = 8 + 12 * n
    return [(pid, d[ds + o:ds + o + s]) for pid, o, s in (struct.unpack('<III', d[8 + 12 * j:20 + 12 * j]) for j in range(n))]


def pach_build(parts, align=16):
    n = len(parts)
    ds = 8 + 12 * n
    hdr = bytearray(b'PACH' + struct.pack('<I', n))
    body = bytearray()
    ents = []
    for pid, b in parts:
        body += b'\0' * ((-(ds + len(body))) % align)
        ents.append((pid, len(body), len(b)))
        body += b
    for pid, o, s in ents:
        hdr += struct.pack('<III', pid, o, s)
    return bytes(hdr + body)


def main():
    w13, k14, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    ids = [int(x) for x in sys.argv[4:]]
    adds = []
    for i in ids:
        name = f'{i:04d}'
        for g in ('SSFA', 'SSFB', 'SSFC'):
            b = entry(k14, g, name)
            if b is None:
                print(f'2K14 has no {g}/{name}')
                continue
            adds.append(((g, name), trim_bpe(b)))
    icons = dict(pach_parse(unpack_any(pach_parse(entry(w13, 'MENU', 'SSFD'))[0][1])))
    k14_icons = dict(pach_parse(unpack_any(pach_parse(entry(k14, 'MENU', 'SSFD'))[0][1])))
    for i in ids:
        if i in k14_icons:
            icons[i] = k14_icons[i]
        else:
            print(f'2K14 has no grid icon for {i}')
    inner = pach_build(sorted(icons.items()))
    adds.append((('MENU', 'SSFD'), pach_build([(0, bpe_wrap_c(inner))], align=4)))
    open(out, 'wb').write(add_entries(w13, adds))
    print(f'{out}: pictures for {ids}')


if __name__ == '__main__':
    main()
