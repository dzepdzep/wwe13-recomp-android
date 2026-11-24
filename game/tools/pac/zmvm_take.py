"""Replaces the motion records of given move ids (every group/slot where the mod has them, role = slot parity) with
WWE 2K14's version of the same id (2K14 reworked some shared moves, e.g. 54101 One-handed Bulldog 3's victim part).
Pair with prm_motionmatch.py for the same ids (2K14's database records).
  zmvm_take.py <2K14 mot dir (mot07/08.pac)> <in dir> <out dir> <id,id,...>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newmoves import pach, zmvm_index, replace_entry, compact
from menuicons import pach_build
from pactool import load

kdir, src, out, ids = sys.argv[1], sys.argv[2], sys.argv[3], {int(v) for v in sys.argv[4].split(',')}
os.makedirs(out, exist_ok=True)
k = {}
for f in ('mot07', 'mot08'):
    d = load(os.path.join(kdir, f + '.pac'))
    for g, e in zmvm_index(d).items():
        for b, bb in pach(d[e[2]:e[2] + e[3]]):
            for x, xb in pach(bb):
                for m, r in pach(xb):
                    if m in ids: k.setdefault((m, x % 2), r)
print('2K14 records:', sorted(k))
for f in ('mot06', 'mot07', 'mot08'):
    d = load(os.path.join(src, f + '.pac')); n = 0
    for g, e in list(zmvm_index(d).items()):
        nb = []; mod = False
        for bank, bb in pach(d[e[2]:e[2] + e[3]]):
            xs = []
            for x, xb in pach(bb):
                ents = []
                for m, r in pach(xb):
                    if (m, x % 2) in k and r != k[(m, x % 2)]: r = k[(m, x % 2)]; n += 1; mod = True
                    ents.append((m, r))
                xs.append((x, pach_build(ents, align=4)))
            nb.append((bank, pach_build(xs, align=4)))
        if mod: d = replace_entry(d, e, pach_build(nb, align=4))
    print(f, 'replaced', n)
    open(os.path.join(out, f + '.pac'), 'wb').write(compact(d))
