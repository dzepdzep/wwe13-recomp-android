"""Ports the per-move tables of the move database that newmoves/prm_rebuild left as WWE '13's: in PRM part 0 the
'PPLS' tables (subs 20/30, 21, 22, 31, 40: keyed by move id; 2K14 has entries for moves WWE '13 lacks - sub 31 is
empty in WWE '13 and holds only 2K14 moves such as 33904-33910), and the top-level parts 7 ('FACD', 64-byte
entries keyed by move) and 8 ('RTI', 32-byte entries keyed by (variant << 16 | move id): placement data).
2K14's tables are supersets of WWE '13's (every WWE '13 key is present; RTI: 1049 of 1053 entries identical).
  prm_tables.py <m3 in> <entry> <out> <2K14 disc m3.pac> <2K14 add-on m3_dlc05.pac>
Subs come from the add-on's PRM5 (its full, newest copy), part 7 from the disc PRM, part 8 from PRM5.
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newmoves as N, prm_rebuild as P
from pactool import load

SUBS = tuple(int(x) for x in os.environ.get('PRM_SUBS', '20,21,22,30,31,40').split(',') if x)
PARTS = tuple(int(x) for x in os.environ.get('PRM_PARTS', '7,8').split(',') if x)

def main(src, name, out, k14, dlc5):
    d = load(src)
    x, e, top, A = P.subs(d, name)
    _, _, ktop, _ = P.subs(load(k14), 'PRM')
    _, _, dtop, D = P.subs(load(dlc5), 'PRM5')
    for s in SUBS:
        print('sub %d: %d -> %d' % (s, len(A[s]), len(D[s])))
        A[s] = D[s]
    p0 = N.pach_build([(p, N.bpe_wrap_c(A[p]) if len(A[p]) else N.bpe_wrap(b'')) for p in sorted(A)], align=16)
    parts = list(top)
    if SUBS: parts[0] = (0, p0)
    K, DT = dict(ktop), dict(dtop)
    for i, srcp in ((7, K), (8, DT)):
        if i not in PARTS: continue
        j = [n for n, (pid, _) in enumerate(parts) if pid == i][0]
        old = parts[j][1]; new = srcp[i]
        if old[:4] == b'BPE ' and new[:4] != b'BPE ': new = N.bpe_wrap_c(new)
        print('part %d: %d -> %d bytes (%s)' % (i, len(N.unz(old)), len(N.unz(new)), new[:4]))
        parts[j] = (i, new)
    open(out, 'wb').write(N.compact(N.replace_entry(d, e, N.pach_build(parts, align=4))))

if __name__ == '__main__':
    main(*sys.argv[1:6])
