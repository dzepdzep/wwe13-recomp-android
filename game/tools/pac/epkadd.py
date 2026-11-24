"""Add entries to an EPK8 pac without rebuilding it: the existing data stays where it is, new blobs are appended
(0x800-aligned) and the directory gets the new entries (sorted into their group).

  python3 epkadd.py <in.pac (EPK8 or XCTD)> <out.pac> <GROUP/NAME=file> [...]
"""
import struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pactool import load


def add_entries(d, adds):
    d = bytearray(d)
    assert d[:4] == b'EPK8'
    groups = []  # [tag, kind, [(name, off_units, size_units)]]
    p = 0x800
    while d[p:p+4].strip(b'\0'):
        tag = d[p:p+4].decode('latin1'); cnt, kind = struct.unpack('<HH', d[p+4:p+8]); p += 12
        nlen = 8 if kind & 1 else 4
        ents = []
        for _ in range(cnt * 4 // (nlen + 8)):
            ents.append((d[p:p+nlen].decode('latin1'), *struct.unpack('<II', d[p+nlen:p+nlen+8]))); p += nlen + 8
        groups.append([tag, kind, ents])
    for (gtag, name), blob in adds:
        g = next(g for g in groups if g[0].strip() == gtag)
        nlen = 8 if g[1] & 1 else 4
        name = name.ljust(nlen)[:nlen]
        d += b'\0' * (-len(d) % 0x800)
        off = (len(d) - 0x4000) // 0x800
        d += blob
        e = (name, off, (len(blob) + 0xFF) // 0x100)
        g[2] = [x for x in g[2] if x[0] != name] + [e]
        g[2].sort(key=lambda x: x[0])
    d += b'\0' * (-len(d) % 0x800)
    directory = bytearray()
    for tag, kind, ents in groups:
        nlen = 8 if kind & 1 else 4
        body = b''.join(n.encode('latin1') + struct.pack('<II', o, s) for n, o, s in ents)
        directory += tag.encode('latin1') + struct.pack('<HHI', len(body) // 4, kind, 0) + body
    assert 0x800 + len(directory) <= 0x4000, 'directory too large'
    d[0x800:0x4000] = directory + b'\0' * (0x3800 - len(directory))
    d[8:12] = struct.pack('<I', len(d) - 0x4000)
    return bytes(d)


if __name__ == '__main__':
    adds = []
    for a in sys.argv[3:]:
        k, f = a.split('=', 1)
        g, n = k.split('/', 1)
        adds.append(((g, n), open(f, 'rb').read()))
    open(sys.argv[2], 'wb').write(add_entries(load(sys.argv[1]), adds))
