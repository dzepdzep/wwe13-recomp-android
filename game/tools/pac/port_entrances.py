"""Ports WWE 2K14 entrances (pac/evt/EntranceNN.pac EVP/EVZ <id> and its variants 10<id>..14<id>) into WWE '13's
EntranceNN.pac (same NN: WWE '13 looks an entrance up in the pac of its id range) and the mod's plist.
2K14's EVZ parts name their motion "25000<id>_A00_def"; WWE '13 only knows the 5-character "25000" form and hangs
on the longer one, so the names are cut to 5 characters (the rest of the data is the same format).
  port_entrances.py <w13 evt dir> <k14 evt dir> <mod dir (plist360*.def/.arc, pac/evt/ written)> <entrance id…>
Only ids WWE '13 doesn't have are added; returns the ids ported.
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import port_wrestler as P
from bpe import bpe_wrap
import epkadd, plist


def parts(u):
    k = struct.unpack('<I', u[4:8])[0]; base = 8 + 12 * k; r = []
    for i in range(k):
        pid, off, sz = struct.unpack('<3I', u[8 + 12 * i:20 + 12 * i]); r.append((pid, u[base + off:base + off + sz]))
    return r


def pach(ps):
    hdr = b'PACH' + struct.pack('<I', len(ps)); off = 0; body = b''
    for pid, b in ps:
        hdr += struct.pack('<3I', pid, off, len(b)); body += b; off += len(b)
    return hdr + body


def fix_evz(blob):
    out = []
    for pid, b in parts(P.unpack_any(blob)):
        b = bytearray(b)
        if len(b) >= 0x6c:
            nm = bytes(b[0x4c:0x6c]).split(b'\0')[0]
            if len(nm) > 5 and nm[:3] == b'250': b[0x4c:0x6c] = nm[:5].ljust(32, b'\0')
        out.append((pid, bytes(b)))
    return bpe_wrap(pach(out))


def main(w13_evt, k14_evt, mod, ids):
    ids = set(ids); done = []
    for nn in range(10):
        name = 'Entrance%02d.pac' % nn
        w, k = os.path.join(w13_evt, name), os.path.join(k14_evt, name)
        if not (os.path.exists(w) and os.path.exists(k)): continue
        wd = P.load(w); have = {n for g, n, o, s, _, _ in P.epk8_entries(wd)}
        kd = P.load(k); adds = []
        for g, n, o, s, _, _ in P.epk8_entries(kd):
            if g not in ('EVP', 'EVZ') or n in have or len(n) not in (3, 5) or int(n[-3:]) not in ids: continue
            blob = kd[o:o + s]
            adds.append(((g, n), fix_evz(blob) if g == 'EVZ' else blob))
        if not adds: continue
        got = sorted({int(n[-3:]) for (g, n), _ in adds}); done += got
        os.makedirs(os.path.join(mod, 'pac', 'evt'), exist_ok=True)
        out = os.path.join(mod, 'pac', 'evt', name)
        open(out, 'wb').write(bytes(epkadd.add_entries(wd, adds)))
        for d_, a_ in (('plist360.def', 'plist360.arc'), ('plist360_4x3.def', 'plist360_4x3.arc')):
            plist.replace(os.path.join(mod, d_), os.path.join(mod, a_), 'pac\\evt\\' + name, out, os.path.join(mod, a_))
        print(name, 'entrances', got, '(%d entries)' % len(adds))
    return done


if __name__ == '__main__':
    print(main(sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]]))
