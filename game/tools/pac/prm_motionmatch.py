"""For moves whose motion in the mod comes from WWE 2K14 (or its add-ons) although WWE '13's move database already had
a record for the id (WWE '13 had the move entry but no motion of its own on the disc), the database records must be
2K14's: they carry the timing (length, hit frames) of the motion. With WWE '13's records Dusty's running grapple
(50527: 0x3a/0x30 frames vs 2K14's 0x1e/0x19) played out of step.
  prm_motionmatch.py <m3 in> <entry> <out> <ids> <src.pac:ENTRY> [...]   (first source with the id wins)
Replaces subs 0/1/2/4/6 records of each id (BE + LE twins).
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newmoves as N, prm_rebuild as P
from pactool import load


def main(src, name, out, idstr, specs):
    ids = [int(v) for v in idstr.split(',')]
    d = load(src); x, e, top, A = P.subs(d, name); R = P.records(A)
    S = []
    for spec in specs:
        path, ent = spec.rsplit(':', 1); _, _, _, B = P.subs(load(path), ent); S.append(P.records(B))
    mid = lambda k: struct.unpack('>H', k[:2])[0]
    done = []
    for i in ids:
        s = next((s for s in S if any(mid(k) == i for k in s[0])), None)
        if not s: print('no source for', i); continue
        for k in R:
            for key in [kk for kk in R[k] if mid(kk) == i]: del R[k][key]
            for key, v in s[k].items():
                if mid(key) == i: R[k][key] = v
        done.append(i)
    B = P.build(R, A)
    p0 = N.pach_build([(p, N.bpe_wrap_c(B[p]) if len(B[p]) else N.bpe_wrap(b'')) for p in sorted(B)], align=16)
    parts = list(top); parts[0] = (0, p0)
    open(out, 'wb').write(N.compact(N.replace_entry(d, e, N.pach_build(parts, align=4))))
    print('records from 2K14 for', len(done), 'ids')


if __name__ == '__main__':
    main(*sys.argv[1:5], sys.argv[5:])
