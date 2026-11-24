"""2K14 attributes for ported superstars as WWE '13 runtime-record patches (roster.txt field 10, "off:byte,…" hex).
Runtime record = chEtc CHAR/DAT without its 4-byte header (u16 big-endian). Learned from the superstars both games
have: DAT 0x04-0x11 (14 ratings) and its copy 0x12-0x1f <- 2K14 DAT 0x24-0x31; 0x112-0x116 <- 0x38-0x3c;
weight u16 0x34 <- 2K14 u16 0x16. A value outside what WWE '13 uses for that byte is left out.
  port_stats.py <w13 chEtc.pac> <k14 chEtc.pac> <roster.txt in> <roster.txt out>
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import port_wrestler as P

BYTES = [(w, 0x24 + (w - 0x04) % 14) for w in range(0x04, 0x20)] + [(0x112 + i, 0x38 + i) for i in range(5)]


def dats(path):
    d = P.load(path)
    dat = next(dict(P.raw_pach(d[o:o + s])) for g, n, o, s, _, _ in P.epk8_entries(d) if g == 'CHAR' and n == 'DAT')
    return {k: P.unpack_any(v) for k, v in dat.items()}


def main(w13_etc, k14_etc, rin, rout):
    w, k = dats(w13_etc), dats(k14_etc)
    seen = {wo: {v[wo] for v in w.values() if len(v) > wo} for wo, _ in BYTES}
    out = []
    for line in open(rin, encoding='latin1'):
        line = line.rstrip('\r\n')
        if line[:1].isdigit():
            f = line.split('|'); c = int(f[0]); d = k[c]
            while len(f) < 9: f.append('')
            p = ['%x:%x' % (wo - 4, d[ko]) for wo, ko in BYTES if d[ko] in seen[wo]]
            wt = struct.unpack('<H', d[0x16:0x18])[0]
            if 0 < wt < 0x4000: p += ['%x:%x' % (0x30, wt >> 8), '%x:%x' % (0x31, wt & 0xFF)]
            line = '|'.join(f[:9] + [','.join(p)])
        elif line.startswith('#') and 'attributes' not in line:
            line += '|attributes (record off:byte, hex)'
        out.append(line)
    open(rout, 'w', encoding='latin1').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:5])
