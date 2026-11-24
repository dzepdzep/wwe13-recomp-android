"""Puts ported motions where WWE '13 looks for them.

A ZMVM group is PACH(bank) > PACH(slot) > PACH(move id) > record. In WWE '13 the slot of a move is
2 * (id // 100 - base) + role (role 0 = attacker, 1 = victim), with one base per bank (bank 45: 370, bank 56: 500,
bank 59: 550, ...). The exceptions are a few low ids (< 2000) kept in other banks' slots 0/1.
newmoves.py copied 2K14's slot numbers, and 2K14 numbers its slots differently (and the DLC packs from 0), so many
ported motions sat in slots the game never reads for their id (e.g. Dusty's taunts 37451/37453 in bank 45 slot 0
instead of 8; 505xx moves in bank 56 slot 0/1 instead of 10/11).

  zmvm_place.py <w13 slot list json> <in dir with mot06/07/08.pac> <out dir>
The json lists WWE '13's own entries as [file, group, bank, slot, id] (built from the original pacs); those are
never moved. Each other record goes to 2 * (id // 100 - base) + (slot % 2) of its bank.
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newmoves import pach, zmvm_index, replace_entry, compact
from menuicons import pach_build
from pactool import load


def bases(w13):
    c = collections.defaultdict(collections.Counter)
    for f, g, b, x, i in w13:
        if i != 65535 and i >= 2000: c[b][i // 100 - x // 2] += 1
    return {b: cc.most_common(1)[0][0] for b, cc in c.items()}


def fix(d, fname, base, own, log):
    changed = False
    for g, e in list(zmvm_index(d).items()):
        banks = pach(d[e[2]:e[2] + e[3]]); nb = []; mod = False
        for bank, bb in banks:
            if bank not in base:
                nb.append((bank, bb)); continue
            xs = collections.OrderedDict((x, dict(pach(xb))) for x, xb in pach(bb))
            moves = []
            for x, ents in xs.items():
                for mid in list(ents):
                    if mid == 65535 or mid < 2000 or (fname, g, bank, x, mid) in own: continue
                    t = 2 * (mid // 100 - base[bank]) + (x % 2)
                    if t < 0 or t == x: continue
                    moves.append((x, t, mid))
            if not moves:
                nb.append((bank, bb)); continue
            for x, t, mid in moves:
                rec = xs[x].pop(mid)
                tgt = xs.setdefault(t, {})
                if mid in tgt:
                    log.append(f'{fname} {g} bank {bank}: {mid} already in slot {t}, kept that one'); continue
                tgt[mid] = rec
                log.append(f'{fname} {g} bank {bank}: {mid} slot {x} -> {t}')
            nb.append((bank, pach_build(sorted((x, pach_build(sorted(v.items()), align=4)) for x, v in xs.items() if v), align=4)))
            mod = True
        if mod:
            d = replace_entry(d, e, pach_build(nb, align=4)); changed = True
    return compact(d) if changed else None


if __name__ == '__main__':
    w13 = json.load(open(sys.argv[1])); src, out = sys.argv[2], sys.argv[3]
    own = {tuple(r) for r in w13}; base = bases(w13)
    os.makedirs(out, exist_ok=True)
    for f in ('mot06', 'mot07', 'mot08'):
        log = []
        r = fix(load(os.path.join(src, f + '.pac')), f, base, own, log)
        print(f, len(log), 'moved'); print('\n'.join('  ' + l for l in log[:400]))
        if r: open(os.path.join(out, f + '.pac'), 'wb').write(r)
