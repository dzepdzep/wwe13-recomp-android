"""2K14 movesets for ported superstars as WWE '13 profile patches (roster.txt field 9, "off:value,…" hex).
WWE '13's CHAR/PRO (1180 bytes) and 2K14's (1276) hold the same move ids (shared numbering) in a different field
order. The field map is learned from the superstars both games have (both name their own id as entrance music):
for each WWE '13 u16 field below 0x240 the 2K14 field with the most equal non-zero values (>= 20 of ~48). A 2K14
value WWE '13 never uses anywhere in its profiles' move area (a 2K14-only move) is left out, so that field keeps
the copied wrestler's move.
  port_moves.py <w13 chEtc.pac> <k14 chEtc.pac> <roster.txt in> <roster.txt out> [newmoves.txt]
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import port_wrestler as P


def pros(path):
    d = P.load(path)
    pro = next(dict(P.raw_pach(d[o:o + s])) for g, n, o, s, _, _ in P.epk8_entries(d) if g == 'CHAR' and n == 'PRO')
    return {k: P.unpack_any(v) for k, v in pro.items()}


def main(w13_etc, k14_etc, rin, rout, extra=None):
    w, k = pros(w13_etc), pros(k14_etc)
    W = {i: struct.unpack('<590H', v[:1180]) for i, v in w.items() if len(v) >= 1180}
    K = {i: struct.unpack('<638H', v[:1276]) for i, v in k.items() if len(v) >= 1276}
    both = [i for i in W if i in K and 100 <= i < 400 and W[i][0x120] == i and K[i][0x120] == i]
    fmap = {}
    for wi in range(0x120):
        best = max(range(0x130), key=lambda ki: (sum(1 for i in both if W[i][wi] == K[i][ki] and W[i][wi]), -ki))
        if sum(1 for i in both if W[i][wi] == K[i][best] and W[i][wi]) >= 20: fmap[wi] = best
    known = set(); [known.update(W[i][:0x120]) for i in W]
    if extra: known.update(int(v) for v in open(extra).read().split())  # moves added by newmoves.py
    out, tot, ok = [], 0, 0
    for line in open(rin, encoding='latin1'):
        line = line.rstrip('\r\n')
        if line[:1].isdigit():
            f = line.split('|'); c = int(f[0])
            while len(f) < 8: f.append('')
            p = []
            for wi, ki in sorted(fmap.items()):
                v = K[c][ki]; tot += 1
                if v == 0 or v in known: p.append('%x:%x' % (wi * 2, v)); ok += 1
            line = '|'.join(f[:8] + [','.join(p)] + f[9:])
        elif line.startswith('#') and 'moveset' not in line:
            line += '|moveset (profile off:value, hex)'
        out.append(line)
    open(rout, 'w', encoding='latin1').write('\n'.join(out) + '\n')
    print('moveset fields: %d mapped, %d of %d values known to WWE \'13' % (len(fmap), ok, tot))


if __name__ == '__main__':
    main(*sys.argv[1:6])
