#!/usr/bin/env python3
"""WWE '13 .pac unpacker (personal modding tool for files from the user's own disc).

Layers: XCTD (0x0FF512ED, LZX) -> EPK8 directory -> PACH archives -> BPE-compressed entries.
  python3 pactool.py list   <file.pac>
  python3 pactool.py unpack <file.pac> <outdir>
  python3 pactool.py pack   <outdir> <new.pac>     (rebuilds from an unpacked folder; parts are stored uncompressed,
                                                    which the game accepts – tested with chEtc.pac)
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unxctd import unpack as xctd_unpack
from bpe import bpe_expand, bpe_wrap
from bpec import bpe_wrap_c

MAGICS = {b'JBOY': 'model', b'YANM': 'anim', b'BPE ': 'bpe', b'PACH': 'pach', b'EPK8': 'epk8', b'XPR2': 'tex'}

def load(path):
    d = open(path, 'rb').read()
    if d[:4] == b'\x0f\xf5\x12\xed':
        tmp = '/tmp/_pactool.dec'
        xctd_unpack(path, tmp)
        d = open(tmp, 'rb').read()
    return d

def epk8_entries(d):
    """Yields (group, name, offset, size). Offsets in 0x800 units from 0x4000, sizes in 0x100 units."""
    assert d[:4] in (b'EPK8', b'EPAC'), 'not EPK8/EPAC'
    epac = d[:4] == b'EPAC'  # WWE 2K14: u32 count (dwords), 4-char names only
    p = 0x800
    while p + 12 <= 0x4000 and d[p:p+4].strip(b'\0'):
        group = d[p:p+4].decode('latin1').strip()
        count, kind = struct.unpack('<HH', d[p+4:p+8])  # (EPAC kind 0/1, EPK8 0x10/0x11: bit 0 = 8-char names)
        p += 12
        gi = p
        nlen = 8 if kind & 1 else 4
        # (count is in 32-bit words: entries are a name + offset + size)
        for _ in range(count * 4 // (nlen + 8)):
            name = d[p:p+nlen].decode('latin1').strip(); p += nlen
            off, size = struct.unpack('<II', d[p:p+8]); p += 8
            yield group, name, 0x4000 + off * 0x800, size * 0x100, kind, gi

def pach_raw(blob):
    cnt = struct.unpack('<I', blob[4:8])[0]
    ents = [struct.unpack('<III', blob[8+12*i:20+12*i]) for i in range(cnt)]
    base = 8 + 12 * cnt
    return [(eid, blob[base+off:base+off+size]) for eid, off, size in ents]

def pach_entries(blob):
    """(id, bytes) of a PACH archive; BPE entries are expanded."""
    if blob[:4] != b'PACH':
        return None
    cnt = struct.unpack('<I', blob[4:8])[0]
    ents = [struct.unpack('<III', blob[8+12*i:20+12*i]) for i in range(cnt)]
    base = 8 + 12 * cnt
    out = []
    for eid, off, size in ents:
        e = blob[base+off:base+off+size]
        if e[:4] == b'BPE ':
            _, csz, usz = struct.unpack('<III', e[4:16])
            e, _ = bpe_expand(e[16:16+csz], usz, False)
        out.append((eid, e))
    return out

def kind(b):
    return MAGICS.get(b[:4], b[:4].decode('latin1') if all(32 <= c < 127 for c in b[:4]) else 'bin')

def walk(d):
    for group, name, off, size, gkind, gi in epk8_entries(d):
        blob = d[off:off+size]
        sub = pach_entries(blob)
        yield group, name, off, size, kind(blob), sub, gkind, gi

def pack(folder, out_path):
    """EPK8 header + directory, each entry a PACH of plain (uncompressed) parts, or the raw blob."""
    import json
    man = json.load(open(os.path.join(folder, 'manifest.json')))
    data = bytearray()
    directory = bytearray()
    for gnum, g in enumerate(man['groups']):
        ents = bytearray()
        nlen = 8 if g['kind'] & 1 else 4
        for e in g['entries']:
            ef = os.path.join(folder, f'{gnum:02d}_{g["tag"]}', e['name'] or '_')
            if e['type'] == 'pach':
                parts = []
                for pid, fn in e['parts']:
                    b = open(os.path.join(ef, fn), 'rb').read()
                    orig = os.path.join(ef, '.orig', fn)
                    if os.path.exists(orig) and open(orig, 'rb').read() == b:
                        b = open(orig + '.stored', 'rb').read()  # unchanged: the original stored (BPE) bytes
                    elif os.path.exists(orig + '.stored') and open(orig + '.stored', 'rb').read()[:4] == b'BPE ':
                        # changed: real byte-pair compression. (Plain parts and stored BPE both crashed the game at match
                        # load: its buffers seem sized for the original compressed size, so a part must not grow much.)
                        b = bpe_wrap_c(b)
                    parts.append((pid, b))
                hdr = bytearray(b'PACH' + struct.pack('<I', len(parts)))
                body = bytearray(); offs = []
                for pid, b in parts:
                    offs.append((pid, len(body), len(b)))
                    body += b
                    body += b'\0' * (-len(body) % 4)
                for pid, o, n in offs:
                    hdr += struct.pack('<III', pid, o, n)
                blob = bytes(hdr + body)
            else:
                blob = open(os.path.join(ef, e['file']), 'rb').read()
            data += b'\0' * (-len(data) % 0x800)
            off = len(data); data += blob
            size_units = (len(blob) + 0xFF) // 0x100
            ents += e['name'].encode('latin1').ljust(nlen, b' ')[:nlen] + struct.pack('<II', off // 0x800, size_units)
        directory += g['tag'].encode('latin1').ljust(4, b' ') + struct.pack('<HHI', len(ents) // 4, g['kind'], 0) + ents
    data += b'\0' * (-len(data) % 0x800)
    out = bytearray(0x4000)
    out[0:16] = man.get('magic', 'EPK8').encode() + struct.pack('<III', man['epk8_a'], len(data), man['epk8_c'])
    assert 0x800 + len(directory) <= 0x4000, 'directory too large'
    out[0x800:0x800 + len(directory)] = directory
    open(out_path, 'wb').write(bytes(out + data))

def main():
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == 'pack':
        return pack(path, sys.argv[3])
    d = load(path)
    import json
    man = {'magic': d[:4].decode('latin1'), 'epk8_a': struct.unpack('<I', d[4:8])[0], 'epk8_c': struct.unpack('<I', d[12:16])[0], 'groups': []}
    last_gi = None
    for group, name, off, size, k, sub, gkind, gi in walk(d):
        if cmd == 'list':
            print(f'{group}/{name}  {size:>9}  {k}' + (f'  ({len(sub)} entries: ' + ', '.join(f'{eid}:{kind(b)}:{len(b)}' for eid, b in sub[:8]) + (' ...' if len(sub) > 8 else '') + ')' if sub else ''))
        else:
            folder = os.path.join(sys.argv[3], f'{len(man["groups"]) - (gi == last_gi):02d}_{group}', name or '_')
            os.makedirs(folder, exist_ok=True)
            if gi != last_gi:
                man['groups'].append({'tag': group, 'kind': gkind, 'entries': []}); last_gi = gi
            if sub is not None:
                parts = []
                raws = dict(pach_raw(d[off:off+size]))
                os.makedirs(os.path.join(folder, '.orig'), exist_ok=True)
                for eid, b in sub:
                    fn = f'{eid:04d}.{kind(b)}'
                    open(os.path.join(folder, fn), 'wb').write(b); parts.append((eid, fn))
                    open(os.path.join(folder, '.orig', fn + '.stored'), 'wb').write(raws[eid])
                    open(os.path.join(folder, '.orig', fn), 'wb').write(b)
                man['groups'][-1]['entries'].append({'name': name, 'type': 'pach', 'parts': parts})
            else:
                fn = f'raw.{k}'
                open(os.path.join(folder, fn), 'wb').write(d[off:off+size])
                man['groups'][-1]['entries'].append({'name': name, 'type': 'raw', 'file': fn})
    if cmd == 'unpack':
        json.dump(man, open(os.path.join(sys.argv[3], 'manifest.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
