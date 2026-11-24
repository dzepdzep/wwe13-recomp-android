"""PRM part 0 'PPLS' tables (subs 20 + 30 (same content), 31, 40), little-endian:
  'PPLS', u16 nKeys, nGroups, nRecords, 8;  nKeys x [cdabcdab, u32 key (variant << 16 | move id, 0 = continues the
  previous key), u8 groups, u8 flag, u16 0];  nGroups x [cdabcdab, u32 records];  nRecords x 56-byte records (key
  order, group order); then the rest of the sub (zero padding to the original size, or a 4-byte tail).
Keys are move ids, not positions, so a key's groups/records can be added on their own.
"""
import struct
RS = 56


def parse(b):
    a, g, c, w = struct.unpack('<4H', b[4:12]); p = 12
    L1 = [b[p + 12 * i + 4:p + 12 * i + 12] for i in range(a)]; p += 12 * a
    grp = [struct.unpack('<I', b[p + 8 * i + 4:p + 8 * i + 8])[0] for i in range(g)]; p += 8 * g
    ents = []; gi = ri = 0
    for e in L1:
        k = struct.unpack('<I', e[:4])[0]; n = e[4]
        gs = []
        for _ in range(n):
            m = grp[gi]; gi += 1
            gs.append(b[p + RS * ri:p + RS * (ri + m)]); ri += m
        ents.append([k, e[4:], gs])
    assert gi == g and ri == c
    return {'w': w, 'ents': ents, 'rest': b[p + RS * c:], 'size': len(b)}


def build(t, keep_size=True):
    E = t['ents']; gl = [g for _, _, gs in E for g in gs]
    h = b'PPLS' + struct.pack('<4H', len(E), len(gl), sum(len(g) // RS for g in gl), t['w'])
    h += b''.join(struct.pack('<2I', 0xabcdabcd, k) + bytes([len(gs)]) + f[1:] for k, f, gs in E)
    h += b''.join(struct.pack('<2I', 0xabcdabcd, len(g) // RS) for g in gl)
    body = h + b''.join(gl)
    rest = t['rest']
    if keep_size and len(body) + len(rest) > t['size'] and rest.strip(b'\0') == b'':
        rest = rest[:max(0, t['size'] - len(body))]
    if keep_size and len(body) + len(rest) < t['size'] and rest.strip(b'\0') == b'':
        rest = b'\0' * (t['size'] - len(body))
    return body + rest


def blocks(t):
    """[(key, [entries])]: a keyed entry plus the key-0 entries that follow it."""
    out = []
    for e in t['ents']:
        if e[0] == 0 and out: out[-1][1].append(e)
        else: out.append((e[0], [e]))
    return out


def add_missing(t, src, want):
    have = {k for k, _ in blocks(t)}; added = []
    for k, es in blocks(src):
        if k and k not in have and want(k):
            t['ents'].extend([list(e) for e in es]); added.append(k)
    return added
