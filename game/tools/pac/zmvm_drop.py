"""Removes the mod's (2K14) motion records for move ids that WWE '13's own add-on packs already animate.
WWE '13's add-ons (dlc_mot02/03, ZMOT groups DLC1/DLC2, own banks 52/64-66/80/81, slots 0/1) carry motions for ~90
ids together with matching move-database records. newmoves also shipped 2K14's motions for some of those ids; once
zmvm_place.py put them in readable slots they competed with the add-on's (Dusty's running grapple 50527 looked wrong).
  zmvm_drop.py <ids json: [[file, group, bank, slot, id], ...] (WWE '13 add-on index)> <w13 slot list json> <in dir> <out dir>
WWE '13's own disc entries (w13 slot list) are never removed.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newmoves import pach, zmvm_index, replace_entry, compact
from menuicons import pach_build
from pactool import load

dlc = {r[4] for r in json.load(open(sys.argv[1]))} - {65535}
own = {tuple(r) for r in json.load(open(sys.argv[2]))}
src, out = sys.argv[3], sys.argv[4]; os.makedirs(out, exist_ok=True)
for f in ('mot06', 'mot07', 'mot08'):
    d = load(os.path.join(src, f + '.pac')); changed = False; gone = set(); n = 0
    for g, e in list(zmvm_index(d).items()):
        nb = []; mod = False
        for bank, bb in pach(d[e[2]:e[2] + e[3]]):
            xs = []
            for x, xb in pach(bb):
                ents = [(m, r) for m, r in pach(xb) if not (m in dlc and (f, g, bank, x, m) not in own)]
                if len(ents) != len(pach(xb)): mod = True; n += len(pach(xb)) - len(ents); gone |= {m for m, _ in pach(xb)} - {m for m, _ in ents}
                if ents: xs.append((x, pach_build(ents, align=4)))
            nb.append((bank, pach_build(xs, align=4)))
        if mod: d = replace_entry(d, e, pach_build(nb, align=4)); changed = True
    print(f, 'removed', n, 'records of', len(gone), 'ids', sorted(gone))
    if changed: open(os.path.join(out, f + '.pac'), 'wb').write(compact(d))
