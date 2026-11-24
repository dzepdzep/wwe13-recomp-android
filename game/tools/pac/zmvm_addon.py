"""Ships WWE '13's own add-on motions (dlc_mot02/03, groups DLC1/DLC2, slots 0 attacker / 1 victim) in the mod's
base mot pacs, at the slots the slot rule gives, in place of the 2K14 motions newmoves put there for the same ids.
Those ids' move-database records are WWE '13's (written for the add-on motion), and the add-on groups may only be
loaded when a WWE '13 add-on superstar is in the match, so the mod's stars need the motion in the base pacs.
  zmvm_addon.py <add-on pac dir (dlc_mot02.pac, dlc_mot03.pac)> <w13 slot list json> <in dir> <out dir>
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newmoves import pach, zmvm_index, replace_entry, compact
from menuicons import pach_build
from pactool import load, epk8_entries

src_dir, slots, src, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
own = {tuple(r) for r in json.load(open(slots))}
add = {}
for f in ('dlc_mot02', 'dlc_mot03'):
    d = load(os.path.join(src_dir, f + '.pac'))
    for e in epk8_entries(d):
        for b, bb in pach(d[e[2]:e[2] + e[3]]):
            for x, xb in pach(bb):
                for m, r in pach(xb):
                    if m != 65535: add.setdefault((m, x % 2), r)
print('add-on motions:', len(add))
for f in ('mot06', 'mot07', 'mot08'):
    d = load(os.path.join(src, f + '.pac')); n = 0; ids = set()
    for g, e in list(zmvm_index(d).items()):
        nb = []; mod = False
        for bank, bb in pach(d[e[2]:e[2] + e[3]]):
            xs = []
            for x, xb in pach(bb):
                ents = []
                for m, r in pach(xb):
                    k = (m, x % 2)
                    if k in add and (f, g, bank, x, m) not in own and r != add[k]:
                        r = add[k]; n += 1; ids.add(m); mod = True
                    ents.append((m, r))
                xs.append((x, pach_build(ents, align=4)))
            nb.append((bank, pach_build(xs, align=4)))
        if mod: d = replace_entry(d, e, pach_build(nb, align=4))
    print(f, 'replaced', n, 'records of', len(ids), 'ids')
    open(os.path.join(out, f + '.pac'), 'wb').write(compact(d))
