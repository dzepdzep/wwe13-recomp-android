"""WAZE (misc MOVS/WAZE) record order. The game keeps the records at entry+336 (160 bytes each: 3 x u64 category
masks, name at +24, owner u16 at +128, id u16 at +152) and builds Create-a-Moveset's move lists only from the first
N records, N = the number of records with a name (sub_828DA118 counts them, sub_828D6160 scans that many). WWE '13
puts its named moves first (sorted by name) and its 434 unnamed ones after them; moves newmoves/dlc merging appended
after the unnamed block were never listed (Dusty's Bionic Elbow Stamp: his signature slot showed NONE).
This moves every named record in front of the unnamed ones (new ones at their place by name), same size, in place.
  waze_order.py <in.pac> <out.pac>
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newmoves as N
from pactool import load

BASE, RS = 336, 160

def key(name):
    s = re.sub(r"[^A-Z0-9 ]", "", name.upper())
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def main(src, out):
    d = bytearray(load(src)); u, e = N.entry(bytes(d), 'MOVS', 'WAZE')
    n = struct.unpack('<I', u[4:8])[0]
    recs = [u[BASE + RS * i:BASE + RS * (i + 1)] for i in range(n)]
    named = [r for r in recs if r[24]]
    first_blank = next(i for i, r in enumerate(recs) if not r[24])
    head = recs[:first_blank]  # (WWE '13's own sorted named block, kept as it is)
    late = [r for r in recs[first_blank:] if r[24]]
    blank = [r for r in recs if not r[24]]
    nm = lambda r: r[24:88].split(b'\0')[0].decode('latin1')
    out_named = list(head)
    for r in sorted(late, key=lambda r: key(nm(r))):
        k = key(nm(r)); j = len(out_named)
        while j > 0 and key(nm(out_named[j - 1])) > k: j -= 1
        out_named.insert(j, r)
    new = out_named + blank
    assert len(new) == n and sorted(new) == sorted(recs)
    body = u[:BASE] + b''.join(new) + u[BASE + RS * n:]
    assert len(body) == len(u)
    d[e[2]:e[2] + len(u)] = body
    open(out, 'wb').write(d)
    print('WAZE: %d records, %d named (%d moved in front of the %d unnamed)' % (n, len(named), len(late), len(blank)))

if __name__ == '__main__':
    main(*sys.argv[1:3])
