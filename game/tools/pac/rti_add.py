"""Adds the RTI entries (PRM top part 8: 'RTI\\0', u32 LE count, 32-byte LE entries keyed by u32 variant<<16 | move id,
sorted) that WWE 2K14 (disc PRM and DLC5 PRM5) has and the given m3 lacks. Existing entries are left as they are:
replacing the whole table with 2K14's (which also changes 1691/1692/31018/50090) went with wrestlers floating and
strikes missing. FACD (part 7) is not touched: 2K14's entries point at 2K14 resources (none equal WWE '13's).
  rti_add.py <m3 in> <entry> <out> <2K14 m3.pac> <2K14 m3_dlc05.pac>
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newmoves as N, prm_rebuild as P
from pactool import load


def rti(b):
    n = struct.unpack('<I', b[4:8])[0]
    return {struct.unpack('<I', b[8 + 32 * i:12 + 32 * i])[0]: b[8 + 32 * i:40 + 32 * i] for i in range(n)}


def main(src, name, out, k14, dlc5):
    d = load(src); x, e, top, A = P.subs(d, name)
    mids = {struct.unpack('>H', A[0][i:i + 2])[0] for i in range(0, len(A[0]), 40)}
    parts = list(top); j = [n for n, (p, _) in enumerate(parts) if p == 8][0]
    raw = parts[j][1]; cur = N.unz(raw); R = rti(cur)
    add = {}
    for f, nm in ((k14, 'PRM'), (dlc5, 'PRM5')):
        for k, v in rti(N.unz(dict(P.subs(load(f), nm)[2])[8])).items():
            if k not in R and (k & 0xffff) in mids: add.setdefault(k, v)
    R.update(add)
    new = cur[:4] + struct.pack('<I', len(R)) + b''.join(R[k] for k in sorted(R))
    print('RTI: +%d entries' % len(add), sorted((k >> 16, k & 0xffff) for k in add))
    if raw[:4] == b'BPE ': new = N.bpe_wrap_c(new)
    parts[j] = (8, new)
    open(out, 'wb').write(N.compact(N.replace_entry(d, e, N.pach_build(parts, align=4))))


if __name__ == '__main__':
    main(*sys.argv[1:6])
