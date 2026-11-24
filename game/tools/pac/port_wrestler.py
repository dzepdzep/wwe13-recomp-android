#!/usr/bin/env python3
"""Ports WWE 2K14 superstars into WWE '13 as a mod folder (one command for any number of them).

  python3 port_wrestler.py --w13 <dir> --k14 <dir> --out <mod dir> [--names names.json] [--copy 322] <id> [<id> ...]

--w13: WWE '13 files: pac/ch/chEtc.pac, pac/ch/ch<copy>.pac (template attire layout), pac/menu/menuHD.pac,
       plist360.def/.arc, plist360_4x3.def/.arc (all from the game folder).
--k14: WWE 2K14 files from the user's disc: pac/ch/ch<id>.pac per superstar, pac/menu/menuHD.pac.
--names: {"360": ["DAMIEN SANDOW", "SANDOW"], ...} (full and short name). Without it, names come from 2K14 itself: its
         chEtc CHAR/DAT record holds string ids (u16 at 0x0a full name = 60000+id, 0x0c short name = 62000+id) that
         are in string.pac (SDB banks: PACH of ZLIB tables [u32 0][u32 n][n x (offset, length, string id)]).
--copy: the WWE '13 superstar whose record (stats, moves, entrance) the new ones start from (default 322).

For each id: ch<id>.pac is built from the 2K14 attires (2K14 parts are already BPE in the same format, copied as
they are; parts only WWE '13 has, like 0100.pach, come from the template attire), a chEtc DAT/PRO record (copy of
<copy> with the id and name), the 2K14 select-screen pictures (menuicons.py), the plist360 records, and a
roster.txt line. The id stays the 2K14 id (free in WWE '13: 359-399 and the other unused ids).
"""
import argparse, json, os, shutil, struct, subprocess, sys, unicodedata, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pactool import load, epk8_entries
import plist
from menuicons import pach_build, unpack_any, trim_bpe
from epkadd import add_entries
from bpec import bpe_wrap_c


def epk8_write(groups):
    """groups: [(tag, kind, [(name, blob)])] -> EPK8 bytes (entries 0x800-aligned)."""
    data = bytearray()
    directory = bytearray()
    for tag, kind, ents in groups:
        nlen = 8 if kind & 1 else 4
        body = bytearray()
        for name, blob in ents:
            data += b'\0' * (-len(data) % 0x800)
            off = len(data)
            data += blob
            body += name.encode('latin1').ljust(nlen, b' ')[:nlen] + struct.pack('<II', off // 0x800, (len(blob) + 0xFF) // 0x100)
        directory += tag.encode('latin1').ljust(4, b' ') + struct.pack('<HHI', len(body) // 4, kind, 0) + body
    data += b'\0' * (-len(data) % 0x800)
    out = bytearray(0x4000)
    out[0:16] = b'EPK8' + struct.pack('<III', 44, len(data), 7)
    out[0x800:0x800 + len(directory)] = directory
    return bytes(out + data)


def raw_pach(blob):
    """PACH -> [(id, stored bytes)] (offsets from the end of the table)."""
    n = struct.unpack('<I', blob[4:8])[0]
    ds = 8 + 12 * n
    return [(pid, blob[ds + o:ds + o + s]) for pid, o, s in (struct.unpack('<III', blob[8 + 12 * j:20 + 12 * j]) for j in range(n))]


def k14_names(k14_dir, ids, lang='00'):
    """{id: (full, short)} from WWE 2K14's chEtc + string.pac (ASCII: WWE '13's name fonts)."""
    etc, strpac = os.path.join(k14_dir, 'pac', 'ch', 'chEtc.pac'), os.path.join(k14_dir, 'pac', 'string.pac')
    if not (os.path.exists(etc) and os.path.exists(strpac)):
        return {}
    d = load(etc)
    dat = next(dict(raw_pach(d[o:o + s])) for g, n, o, s, _, _ in epk8_entries(d) if g == 'CHAR' and n == 'DAT')
    want = {}
    for cid in ids:
        if cid in dat:
            want[cid] = struct.unpack('<HH', unpack_any(dat[cid])[10:14])
    sids = {x for v in want.values() for x in v}
    found = {}
    d = load(strpac)
    for g, n, o, s, _, _ in epk8_entries(d):
        if g != 'SDB' or n[2:] != lang:
            continue
        for _, b in raw_pach(d[o:o + s]):
            if b[:4] == b'ZLIB':
                u = zlib.decompressobj().decompress(b[16:16 + struct.unpack('<I', b[8:12])[0]])
            else:
                u = unpack_any(b)
            cnt = struct.unpack('<I', u[4:8])[0] if len(u) >= 8 else 0
            if not 0 < cnt < 100000 or 8 + 12 * cnt > len(u):
                continue
            for j in range(cnt):
                off, ln, sid = struct.unpack('<III', u[8 + 12 * j:20 + 12 * j])
                if sid in sids and sid not in found:
                    found[sid] = u[off:off + ln].split(b'\0')[0].decode('utf-8', 'replace')
    ascii_ = lambda t: unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode().upper()
    return {cid: (ascii_(found[f]), ascii_(found[sh])) for cid, (f, sh) in want.items() if f in found and sh in found}


def k14_titantrons(k14_dir, ids):
    """{id: 2K14 titantron number} from the 2K14 chEtc CHAR/PRO record (u16 at 0x242, entrance slot 1; 255 = none)."""
    etc = os.path.join(k14_dir, 'pac', 'ch', 'chEtc.pac')
    if not os.path.exists(etc):
        return {}
    d = load(etc)
    pro = next(dict(raw_pach(d[o:o + s])) for g, n, o, s, _, _ in epk8_entries(d) if g == 'CHAR' and n == 'PRO')
    out = {}
    for cid in ids:
        if cid in pro:
            t = struct.unpack('<H', unpack_any(pro[cid])[0x242:0x244])[0]
            if t not in (0, 255, 0xFFFF):
                out[cid] = t
    return out


def port_titantrons(k14_dir, out, ids, enc, w13_movies=None):
    """Converts each superstar's 2K14 titantron (Bink 2 KB2f) to WWE '13's KB2d with kb2d_enc, as
    movies/titantron/<n>.bik2 under a number WWE '13 doesn't use (the superstar's id; a titantron shared by
    several – The Shield's – takes the first one's). Returns {id: n}."""
    used = set()
    if w13_movies and os.path.isdir(w13_movies):
        used = {int(f[:-5]) for f in os.listdir(w13_movies) if f.endswith('.bik2') and f[:-5].isdigit()}
    tdir = os.path.join(out, 'movies', 'titantron')
    os.makedirs(tdir, exist_ok=True)
    num_of_tid, result = {}, {}
    for cid, tid in k14_titantrons(k14_dir, ids).items():
        src = os.path.join(k14_dir, 'movies', 'titantron', f'{tid}.bik2')
        if not os.path.exists(src):
            print(f'{cid}: no 2K14 titantron {tid}.bik2')
            continue
        if tid not in num_of_tid:
            n = cid
            while n in used or n in num_of_tid.values():
                n += 1000  # (WWE '13 numbers are < 1000)
            r = subprocess.run([enc, src, os.path.join(tdir, f'{n}.bik2')], capture_output=True, text=True)
            if r.returncode:
                print(f'{cid}: titantron {tid} not converted: {r.stderr.strip()}')
                continue
            num_of_tid[tid] = n
            print(f'{cid}: titantron {tid} -> {n}.bik2 ({r.stderr.strip().split(", ")[-2]})')
        result[cid] = num_of_tid[tid]
    return result


# Parts taken from the WWE '13 template attire, not from 2K14 (as in the tested Sandow port): 0x32 (shared table),
# 0x64 (PACH that 2K14 lacks), 0x6e (YANM animation, 2K14's is a newer version), 0x9c40 (DUMY).
KEEP_W13 = (0x32, 0x64, 0x6e, 0x9c40)


def build_ch(k14_pac, template_pac, cid):
    """WWE '13 ch<cid>.pac from a 2K14 one: attire names 000<cid><xx>, 2K14 parts (stored bytes) + KEEP_W13 parts."""
    k = load(k14_pac)
    t = load(template_pac)
    tmpl = dict(raw_pach(next(t[o:o + s] for g, n, o, s, _, _ in epk8_entries(t) if g == 'EMD')))
    ents = []
    for g, n, o, s, _, _ in epk8_entries(k):
        if g != 'EMD':
            continue
        parts = dict(raw_pach(k[o:o + s]))
        for pid in KEEP_W13:
            if pid in tmpl:
                parts[pid] = tmpl[pid]
        pach = pach_build([(pid, trim_bpe(b)) for pid, b in sorted(parts.items())], align=4)
        ents.append((f'000{cid}{n[-2:]}' if len(n) == 8 else n, pach))
    if not ents:
        raise SystemExit(f'{k14_pac}: no attires')
    return epk8_write([('EMD', 0x11, ents)])


def chetc_with(chetc, cid_names, copy_id):
    """chEtc.pac with DAT/PRO records for the new ids (copies of copy_id with the id and names)."""
    d = load(chetc)
    adds = []
    for g, n, o, s, kind, gi in epk8_entries(d):
        if g != 'CHAR' or n not in ('DAT', 'PRO'):
            continue
        parts = dict(raw_pach(d[o:o + s]))
        src = unpack_any(parts[copy_id])
        for cid, (full, short) in cid_names.items():
            rec = bytearray(src)
            if n == 'DAT':
                rec[0x102:0x10a] = struct.pack('<4H', *(cid * 10 + j for j in range(1, 5)))  # (DSBD string ids)
                for off in (0x38, 0x10a, 0x110):
                    rec[off:off + 2] = struct.pack('<H', cid)
                for off, ln, v in ((0x3a, 0x44, full), (0x7e, 0x5c, full), (0xda, 0x26, short)):
                    rec[off:off + ln] = v.encode('latin1', 'replace').ljust(ln, b'\0')[:ln]
            parts[cid] = bpe_wrap_c(bytes(rec))
        # (unchanged parts keep their stored bytes)
        new = pach_build(sorted((pid, trim_bpe(b) if b[:4] == b'BPE ' else b) for pid, b in parts.items()), align=4)
        adds.append(((g, n), new))
    return add_entries(d, adds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w13', required=True)
    ap.add_argument('--k14', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--names')
    ap.add_argument('--copy', type=int, default=322)
    ap.add_argument('--kb2d-enc', help='kb2d_enc (tools/video): also convert the 2K14 titantrons')
    ap.add_argument('--w13-movies', help="WWE '13 movies/titantron folder (to avoid its numbers)")
    ap.add_argument('ids', type=int, nargs='+')
    a = ap.parse_args()
    names = json.load(open(a.names)) if a.names else {}
    w, k, out = a.w13, a.k14, a.out
    for cid, nm in k14_names(k, a.ids).items():
        names.setdefault(str(cid), list(nm))
    os.makedirs(os.path.join(out, 'pac', 'ch'), exist_ok=True)
    os.makedirs(os.path.join(out, 'pac', 'menu'), exist_ok=True)
    cid_names = {}
    done = []
    for cid in a.ids:
        src = os.path.join(k, 'pac', 'ch', f'ch{cid}.pac')
        if not os.path.exists(src):
            print(f'{cid}: no {src}, skipped')
            continue
        dst = os.path.join(out, 'pac', 'ch', f'ch{cid}.pac')
        open(dst, 'wb').write(build_ch(src, os.path.join(w, 'pac', 'ch', f'ch{a.copy}.pac'), cid))
        nm = names.get(str(cid)) or [f'SUPERSTAR {cid}', f'SUPERSTAR {cid}']
        cid_names[cid] = (nm[0], nm[1] if len(nm) > 1 else nm[0])
        done.append(cid)
        print(f'{cid}: ch{cid}.pac ({os.path.getsize(dst)} bytes) – {cid_names[cid][0]}')
    if not done:
        raise SystemExit('nothing ported')
    chetc_out = os.path.join(out, 'pac', 'ch', 'chEtc.pac')
    open(chetc_out, 'wb').write(chetc_with(os.path.join(w, 'pac', 'ch', 'chEtc.pac'), cid_names, a.copy))
    # select-screen pictures
    menu_out = os.path.join(out, 'pac', 'menu', 'menuHD.pac')
    sys.argv = ['menuicons', os.path.join(w, 'pac', 'menu', 'menuHD.pac'), os.path.join(k, 'pac', 'menu', 'menuHD.pac'),
                menu_out] + [str(c) for c in done]
    import menuicons
    menuicons.main()
    # plist360 (+ 4x3): new pacs added, chEtc patched, menuHD regenerated (HD set only)
    for suffix in ('', '_4x3'):
        dfile, afile = os.path.join(w, f'plist360{suffix}.def'), os.path.join(w, f'plist360{suffix}.arc')
        odef, oarc = os.path.join(out, f'plist360{suffix}.def'), os.path.join(out, f'plist360{suffix}.arc')
        shutil.copyfile(dfile, odef)
        tmp = oarc + '.tmp'
        plist.patch(odef, afile, 'pac\\ch\\chEtc.pac', chetc_out, tmp)
        for cid in done:
            pac, built = f'pac\\ch\\ch{cid}.pac', os.path.join(out, 'pac', 'ch', f'ch{cid}.pac')
            if pac.lower() in open(odef, 'rb').read().decode('latin1').lower().split('\r\n'):
                plist.replace(odef, tmp, pac, built, tmp)  # (WWE '13 has a pac for this id already: the mod's replaces it)
            else:
                plist.add(odef, tmp, pac, built, odef, tmp)
        names_in_def = open(odef, 'rb').read().decode('latin1').lower().split('\r\n')
        if 'pac\\menu\\menuhd.pac' in names_in_def:
            plist.replace(odef, tmp, 'pac\\menu\\menuHD.pac', menu_out, tmp)
        os.replace(tmp, oarc)
    trons = port_titantrons(k, out, done, a.kb2d_enc, a.w13_movies) if a.kb2d_enc else {}
    with open(os.path.join(out, 'roster.txt'), 'w') as f:
        f.write('# ported from WWE 2K14: id|copy record of|full name|short name|DSBD id (hex)|titantron\n')
        for cid in done:
            line = f'{cid}|{a.copy}|{cid_names[cid][0]}|{cid_names[cid][1]}|{cid * 10 + 1:x}'
            f.write(line + (f'|{trons[cid]}' if cid in trons else '') + '\n')
    print(f'{out}: {len(done)} superstar(s)')


if __name__ == '__main__':
    main()
