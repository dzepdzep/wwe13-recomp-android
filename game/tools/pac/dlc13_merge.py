"""Merges the mod's changes into the newest WWE '13 DLC's full copies of misc/m3/chEtc.
With WWE '13 DLC installed the game reads dlc1:\\pac\\misc_dlc03.pac, ch\\chEtc_dlc03.pac (and m3_dlc03.pac) instead
of the game's own misc/chEtc/m3, so the mod's ported superstars and moves would be lost. The merged copies go to
<mod>/dlc/pac/... (laid over the DLC package by the runtime, see content_manager.cpp).
  dlc13_merge.py <base dir> <mod dir> <dlc dir> <out dir>
    base: the game's misc.pac m3.pac chEtc.pac; mod: the mod's; dlc: misc_dlc03.pac m3_dlc03.pac chEtc_dlc03.pac
Moves both the mod (from 2K14) and the DLC add under the same id keep the mod's version (its motions are the
ones in the mod's mot07/08).
"""
import os, sys, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newmoves as N
import prm_rebuild as P
import pactool
import bpec
import bpe_patch
SHRINK = True
ONLY = ('DAT', 'PRO')  # (CHAR entries taken from the mod)
from pactool import load


def waze(base, mod, dlc):
    def recs(d):
        u, e = N.entry(d, 'MOVS', 'WAZE'); n = struct.unpack('<I', u[4:8])[0]
        return u, e, n, N.waze_records(u)
    rid = N.waze_id
    _, _, _, rb = recs(base); _, _, _, rm = recs(mod); du, de, dn, rd = recs(dlc)
    bids = {rid(r) for r in rb}
    madd = {rid(r): r for r in rm if rid(r) not in bids}
    assert all(rid(r) in {rid(x) for x in rm} for r in rb), 'mod removed records?'
    out, seen, over = [], set(), 0
    for r in rd:
        i = rid(r)
        if i in madd: r = madd[i]; over += 1
        out.append(r); seen.add(i)
    new = [r for i, r in madd.items() if i not in seen]
    out += new
    u = N.waze_build(du, out)
    print(f'WAZE: dlc {dn}, mod adds {len(madd)} ({over} replace the dlc\'s, {len(new)} new) -> {len(out)}')
    return N.compact(N.replace_entry(dlc, de, u))


def append_entry(d, group, name, blob):
    """The entry's new bytes go after the end of the file and its directory record points there; every other entry
    keeps its offset (a merged misc whose later entries had moved crashed the game at match load; a grown last entry,
    m3's PRM, was fine)."""
    d = bytearray(d)
    hits = [(g, n, off, size, gi) for g, n, off, size, k, sub, gk, gi in pactool.walk(bytes(d)) if (g, n) == (group, name)]
    assert len(hits) == 1
    _, _, off, size, gi = hits[0]
    rec = name.encode('latin1').ljust(4)[:4] + struct.pack('<II', (off - 0x4000) // 0x800, size // 0x100)
    at = bytes(d[0x800:0x4000]).find(rec)
    assert at >= 0 and bytes(d[0x800:0x4000]).count(rec) == 1, 'directory record not found'
    d += b'\0' * (-len(d) % 0x800)
    new_off = len(d)
    d += blob + b'\0' * (-len(blob) % 0x800)
    d[0x800 + at + 4:0x800 + at + 12] = struct.pack('<II', (new_off - 0x4000) // 0x800, (len(blob) + 0xFF) // 0x100)
    d[8:12] = struct.pack('<I', len(d) - 0x4000)  # (header: data length)
    return bytes(d)


def prm(base, mod, dlc, out_path):
    _, _, _, Ab = P.subs(base, 'PRM'); _, _, _, Am = P.subs(mod, 'PRM')
    x, e, top, Ad = P.subs(dlc, 'PRM')
    Rb, Rm, Rd = P.records(Ab), P.records(Am), P.records(Ad)
    stats = {}
    # (sub 4's start field depends on the layout: compared without it)
    norm = lambda k, v: (v[0][:4] + v[0][10:], v[1][:4] + v[1][10:], v[2], v[3]) if k == 4 else v
    for k in (0, 1, 2, 4, 6):
        n = 0
        for key, v in Rm[k].items():
            if key not in Rb[k]:
                Rd[k][key] = v; n += 1
            elif norm(k, v) != norm(k, Rb[k][key]):
                raise SystemExit(f'mod changed an original record in sub {k}: {key.hex()}')
        stats[k] = n
    for k in (3, 7):  # (the mod leaves these as the original)
        assert Am[k] == Ab[k], f'mod changed sub {k}'
    print('PRM: mod records laid into the dlc table', stats)
    B = P.build(Rd, Ad)
    p0 = N.pach_build([(p, N.bpe_wrap_c(B[p]) if len(B[p]) else N.bpe_wrap(b'')) for p in sorted(B)], align=16)
    parts = list(top); parts[0] = (0, p0)
    open(out_path, 'wb').write(N.compact(N.replace_entry(dlc, e, N.pach_build(parts, align=4))))


def chetc(base_p, mod_p, dlc_p, out_dir):
    """Unpacks the dlc chEtc, puts in every part the mod changed (the dlc must not have changed it too), repacks."""
    import subprocess, tempfile, shutil, hashlib
    def parts(d):
        o = {}
        for g, n, off, size, k, sub, gk, gi in pactool.walk(d):
            if sub is not None:
                for eid, b in sub: o[(g, n, eid)] = b
            else: o[(g, n, -1)] = d[off:off + size]
        return o
    B, M, D = parts(load(base_p)), parts(load(mod_p)), parts(load(dlc_p))
    ch = {k: v for k, v in M.items() if B.get(k) != v and k[1] in ONLY}
    for k in ch:
        assert k in D and D[k] == B.get(k), f'conflict in {k}'
    tmp = tempfile.mkdtemp()
    subprocess.run([sys.executable, pactool.__file__, 'unpack', dlc_p, tmp], check=True, stdout=subprocess.DEVNULL)
    import json
    man = json.load(open(os.path.join(tmp, 'manifest.json')))
    n = 0
    for gnum, g in enumerate(man['groups']):
        for e in g['entries']:
            if e['type'] != 'pach': continue
            ef = os.path.join(tmp, f'{gnum:02d}_{g["tag"]}', e['name'] or '_')
            for pid, fn in e['parts']:
                k = (g['tag'], e['name'], pid)
                if k in ch:
                    open(os.path.join(ef, fn), 'wb').write(ch[k]); n += 1
                elif k == ('CHCM', 'DAT', 21) and SHRINK:
                    # (the game crashed reading a merged copy larger than the DLC's own: one stored part is
                    # byte-pair compressed so the file stays within the original size)
                    raw = open(os.path.join(ef, fn), 'rb').read()
                    open(os.path.join(ef, fn), 'wb').write(bpec.bpe_wrap_c(raw))
                    os.unlink(os.path.join(ef, '.orig', fn)) if os.path.exists(os.path.join(ef, '.orig', fn)) else None
    assert n == len(ch), (n, len(ch))
    os.makedirs(os.path.join(out_dir, 'ch'), exist_ok=True)
    out = os.path.join(out_dir, 'ch', os.path.basename(dlc_p))
    subprocess.run([sys.executable, pactool.__file__, 'pack', tmp, out], check=True)
    shutil.rmtree(tmp)
    print(f'chEtc: {n} parts from the mod')


if __name__ == '__main__':
    b, m, d, o = sys.argv[1:5]
    os.makedirs(o, exist_ok=True)
    misc = waze(load(f'{b}/misc.pac'), load(f'{m}/misc.pac'), load(f'{d}/misc_dlc03.pac'))
    open(f'{o}/misc_dlc03.pac', 'wb').write(misc)
    prm(load(f'{b}/m3.pac'), load(f'{m}/m3.pac'), load(f'{d}/m3_dlc03.pac'), f'{o}/m3_dlc03.pac')
    chetc_inplace(f'{b}/chEtc.pac', f'{m}/chEtc.pac', f'{d}/chEtc_dlc03.pac', o)


def chetc_inplace(base_p, mod_p, dlc_p, out_dir, only=('DAT', 'PRO')):
    """The game reads the DLC chEtc's entries with the DLC's own sizes (a bigger merged entry was read cut short and
    its last parts unpacked garbage over the stack). So the mod's parts are written into the DLC file in place: each
    entry keeps its offset and its room (to the next 0x800), and fits because unused slots get the smallest blank record."""
    import struct
    B, M = {}, {}
    for d, o in ((load(base_p), B), (load(mod_p), M)):
        for g, n, off, size, k, sub, gk, gi in pactool.walk(d):
            if sub is not None:
                for eid, b in sub: o[(g, n, eid)] = b
    dd = bytearray(load(dlc_p))
    total = 0
    # Unused slots (DAT name "0", "DLC <n>", "dummy_<n>", "MOTION TEST <n>"): see below.
    import re
    dat = {}
    for g, n, off, size, k, sub, gk, gi in pactool.walk(bytes(dd)):
        if (g, n) == ('CHAR', 'DAT'): dat = dict(sub)
    o = next(v for v in dat.values() if b'DREW' in v).find(b'DREW') if any(b'DREW' in v for v in dat.values()) else 58
    modids = {eid for (g, n, eid), v in M.items() if g == 'CHAR' and v != B.get((g, n, eid))}
    unused = sorted(e for e, v in dat.items() if e not in modids and
                    re.fullmatch(rb'0|DLC \d+|dummy_\d+|MOTION TEST \d+', v[o:o + 24].split(b'\0')[0]))
    for g, n, off, size, k, sub, gk, gi in list(pactool.walk(bytes(dd))):
        if sub is None or n not in only or g != 'CHAR':
            continue
        raws = pactool.pach_raw(bytes(dd[off:off + size]))
        rawd = dict(raws); plain0 = dict(sub)
        # Unused slots get the smallest blank record of the entry, with their own id put in (bytes patched in the
        # compressed stream); kept as they are when that is not smaller.
        slot_blobs = {}
        blanks = [e for e in unused if re.fullmatch(rb'0', dat[e][o:o + 24].split(b'\0')[0])]
        from bpe import bpe_expand
        # (where a blank record holds its own id: the offsets at which every blank one does)
        L = min(len(plain0[b]) for b in blanks)
        ids_at = [i for i in range(L - 1) if all(plain0[b][i:i + 2] == struct.pack('<H', b) for b in blanks)]
        for e in unused:
            best = None
            for tid in sorted(blanks, key=lambda x: len(rawd[x]))[:40]:
                if best is not None and len(rawd[tid]) >= len(best): break
                t = plain0[tid]
                ch = {}
                for i in ids_at: ch[i] = e & 0xFF; ch[i + 1] = e >> 8
                ch = {i: v for i, v in ch.items() if t[i] != v}
                want = bytearray(t)
                for i, v in ch.items(): want[i] = v
                nb = bpe_patch.patch(rawd[tid], ch)
                if nb is None: nb = bpec.bpe_wrap_c(bytes(want))  # (packed again when the bytes are inside pairs)
                assert bpe_expand(nb[16:], len(t), False)[0] == bytes(want)
                if best is None or len(nb) < len(best): best = nb
            if best is not None and len(best) < len(rawd[e]):
                slot_blobs[e] = best
        parts, changed = [], 0
        for eid, stored in raws:
            if eid in unused and eid in slot_blobs:
                parts.append((eid, slot_blobs[eid])); continue
            m = M.get((g, n, eid))
            if m is not None and m != B.get((g, n, eid)):
                assert dict(sub)[eid] == B.get((g, n, eid)), f'dlc changed {g}/{n}/{eid} too'
                stored = bpec.bpe_wrap_c(m); changed += 1
            parts.append((eid, stored))
        # (every part keeps its own bytes: parts sharing their bytes made the game report a disc read error)
        hdr = bytearray(b'PACH' + struct.pack('<I', len(parts))); body = bytearray(); tab = []
        for eid, b in parts:
            tab.append((eid, len(body), len(b)))
            body += b; body += b'\0' * (-len(body) % 4)
        for eid, po, pl in tab:
            hdr += struct.pack('<III', eid, po, pl)
        blob = bytes(hdr + body)
        room = (size + 0x7FF) // 0x800 * 0x800  # (the game reads the entry with room to the next 0x800)
        assert len(blob) <= room, f'{g}/{n}: {len(blob)} > {room}'
        dd[off:off + room] = blob + b'\0' * (room - len(blob))
        if len(blob) > size:  # the directory's size (0x100 units) grows within the room
            rec = n.encode().ljust(4)[:4] + struct.pack('<II', (off - 0x4000) // 0x800, size // 0x100)
            at = bytes(dd[0x800:0x4000]).find(rec)
            assert at >= 0 and bytes(dd[0x800:0x4000]).count(rec) == 1, 'directory entry not found'
            dd[0x800 + at + 8:0x800 + at + 12] = struct.pack('<I', (len(blob) + 0xFF) // 0x100)
        print(f'chEtc {g}/{n}: id at {ids_at}; {changed} parts from the mod, {len(unused)} unused slots made blank ({len(slot_blobs)}), {len(blob)} of {size} bytes')
        total += changed
    os.makedirs(os.path.join(out_dir, 'ch'), exist_ok=True)
    open(os.path.join(out_dir, 'ch', os.path.basename(dlc_p)), 'wb').write(bytes(dd))
    return total


def catalog_patch(arc, pairs):
    """info/catalog_hd.arc and catalog_sd.arc are the DLC's pac directory cache (as plist360.arc for the disc's pacs:
    [ff ff][u16 BE record] then big-endian [tag][u16 dword count << 4][u16 kind][u32 0] groups of [name][u32 offset
    (0x800 units)][u32 size (0x100 units)]). The game reads a DLC pac's entries from here, not from the pac (a merged
    m3 was read with the original PRM size and crashed). pairs: [(original pac bytes, merged pac bytes)]; the record
    that lists exactly the original's entries gets the merged one's."""
    a = bytearray(arc)
    for orig, new in pairs:
        def ents(d):
            return [(g, n, (o - 0x4000) // 0x800, s // 0x100) for g, n, o, s, _, _ in pactool.epk8_entries(d)]
        eo, en = ents(orig), ents(new)
        assert [(g, n) for g, n, _, _ in eo] == [(g, n) for g, n, _, _ in en]
        hit = 0
        r = a.find(b'\xff\xff')
        while r >= 0:
            nx = a.find(b'\xff\xff', r + 4)
            end = nx if nx > 0 else len(a)
            p = r + 4; found = []
            while p + 12 <= end:
                tag = a[p:p + 4].decode('latin1').strip(); cnt, kind = struct.unpack('>HH', a[p + 4:p + 8]); p += 12
                nlen = 8 if kind & 0x1000 else 4
                for _ in range((cnt >> 4) * 4 // (nlen + 8)):
                    name = a[p:p + nlen].decode('latin1').strip('\x00 ')
                    o, sz = struct.unpack('>II', a[p + nlen:p + nlen + 8])
                    found.append((tag, name, o, sz, p + nlen)); p += nlen + 8
            if [(t, n, o, z) for t, n, o, z, _ in found] == eo:
                for (t, n, o, z, at), (_, _, no, nz) in zip(found, en):
                    a[at:at + 8] = struct.pack('>II', no, nz)
                hit += 1
            r = nx
        assert hit == 1, f'record not found ({hit})'
    return bytes(a)
