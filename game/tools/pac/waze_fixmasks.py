"""Repairs the category masks of WAZE records that newmoves/merge_waze took from WWE 2K14. merge_waze cut 2K14's
records at entry+40 + 160*i, but records start at entry+336 (= 40 + 2*160 - 24... the masks are the first 24 bytes
of a record), so each copied 160-byte piece carried one move's name/id/data and the NEXT 2K14 move's category masks;
the added moves ended up in the wrong Create-a-Moveset categories (Jab & Straight Punch: no signature flag).
Each record whose id is not one of WWE '13's own (base) gets its whole 160 bytes from the first 2K14 source that has
the id (when the rest differs, only the masks are taken). The same shift hit WWE '13's own moves right after an
inserted one: those get their masks back from the WWE '13 pac(s) given (first listed wins).
  waze_fixmasks.py <in.pac> <out.pac> <w13 base misc pac>[,<w13 add-on misc pac>...] <2K14 src pac>...
"""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newmoves as N
from pactool import load

BASE, RS = 336, 160

def recs(u):
    n = struct.unpack('<I', u[4:8])[0]
    return n, [u[BASE + RS * i:BASE + RS * (i + 1)] for i in range(n)]

def mid(r): return struct.unpack('<H', r[152:154])[0]

def main(src, out, w13, *k14):
    d = bytearray(load(src)); u, e = N.entry(bytes(d), 'MOVS', 'WAZE'); n, R = recs(u)
    own = {}
    for p in w13.split(','):
        for r in recs(N.entry(load(p), 'MOVS', 'WAZE')[0])[1]: own.setdefault(mid(r), r)
    K = {}
    for p in k14:
        for r in recs(N.entry(load(p), 'MOVS', 'WAZE')[0])[1]: K.setdefault(mid(r), r)
    fixed = same = differ = missing = ownfix = 0
    for i, r in enumerate(R):
        m = mid(r)
        if m in own and own[m][24:] == r[24:]:
            if own[m][:24] != r[:24]: R[i] = own[m]; ownfix += 1  # (WWE '13's own move after an inserted one)
            continue
        k = K.get(m)
        if k is None: missing += 1; continue
        if k[:24] == r[:24]: same += 1; continue
        if k[24:] != r[24:]: differ += 1; R[i] = k[:24] + r[24:]; continue  # (data kept, masks from 2K14)
        R[i] = k; fixed += 1
    body = u[:BASE] + b''.join(R) + u[BASE + RS * n:]
    d[e[2]:e[2] + len(u)] = body
    open(out, 'wb').write(d)
    print('2K14 moves: masks repaired %d, already right %d, masks repaired with other data kept %d, no 2K14 record %d; '
          "WWE '13 moves: masks restored %d" % (fixed, same, differ, missing, ownfix))

if __name__ == '__main__':
    main(*sys.argv[1:])
