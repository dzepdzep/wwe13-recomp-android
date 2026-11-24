"""Ports WWE 2K14 entrance music into WWE '13's sound/Music.pck by re-pointing "carrier" music numbers: WWE '13
music numbers no superstar's profile uses (generic Create-an-Entrance tracks 519-578). Nothing is added to the banks
(the game's sound memory has no room to grow them): the carrier's sound object in Entrance_Music_M gets the 2K14
media id and streams it from the start (its in-bank prefetch part is dropped), and the 2K14 media (XMA, same Wwise
version 65) is added to the package's streamed files. A ported superstar's profile then names the carrier number.
  music_port2.py <w13 Music.pck> <out Music.pck> <map.json {carrier n: 2k14 music id}>
"""
import json, struct, sys
ARGS = sys.argv[1:]
sys.path.insert(0, '/sessions/rcw-01vrgngxxjbzh8hczr5zpvxm')
from bnk import fnv, chunks, hirc
from isofile import IsoFile
import importlib.util
spec = importlib.util.spec_from_file_location('mp', '/sessions/rcw-01vrgngxxjbzh8hczr5zpvxm/music_port.py'); mp = importlib.util.module_from_spec(spec)
sys.argv = ['x']; spec.loader.exec_module(mp)
CARRIERS = list(range(519, 579))
EV, MU = 0xf4757ccf, 0x9a4847cb

def sound_of(evb, E, mub, M, music):
    """the sound object id behind Play_MUS_<music>_0_0 (event -> play action -> container -> sound)"""
    try:
        return _sound_of(evb, E, mub, M, music)
    except (struct.error, KeyError):
        return None

def _sound_of(evb, E, mub, M, music):
    eid = fnv('Play_MUS_%04d_0_0' % music)
    if eid not in E: return None
    t, q, l = E[eid]; eb = evb[q + 5:q + 5 + l]
    aid = struct.unpack('>I', eb[8:12])[0]; t, q, l = E[aid]; tgt = struct.unpack('>I', evb[q + 5 + 6:q + 5 + 10])[0]
    if tgt not in M: return None
    t, q, l = M[tgt]; rb = mub[q + 5:q + 5 + l]
    for o in range(len(rb) - 3):
        sid = struct.unpack('>I', rb[o:o + 4])[0]
        if sid in M and M[sid][0] == 2: return sid
    return None

def main():
    w13, out, spec_ = ARGS[0], ARGS[1], json.load(open(ARGS[2]))
    pk = mp.read_pck(w13)
    ev13, mu13 = pk['banks'][EV], pk['banks'][MU]
    E13, M13 = hirc(ev13), hirc(mu13)
    E_S, M = IsoFile('/sound/Game_Sound_E_S.pck'), IsoFile('/sound/Game_Sound_M.pck')
    def lut(m):
        h = m.read(0, 8); hs = struct.unpack('>I', h[4:8])[0]; d = m.read(0, 8 + hs)
        ver, ls, bs, ss, es = struct.unpack('>5I', d[8:28]); p = 28 + ls
        banks = {e[0]: e for e in (struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0]))}; p += bs
        streams = {e[0]: e for e in (struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0]))}
        return banks, streams
    esb, _ = lut(E_S); mb, ms = lut(M)
    e = esb[0xe85dda21]; ev14 = E_S.read(e[3] * e[1], e[2]); H14 = hirc(ev14)
    def carrier_ok(n):
        """a carrier whose sound object is its own and streams from a prefetch part (type 2)"""
        sid = sound_of(ev13, E13, mu13, M13, n)
        if not sid: return None
        t, q, l = M13[sid]; b = mu13[q + 5:q + 5 + l]
        return sid if struct.unpack('>I', b[8:12])[0] == 2 else None
    if isinstance(spec_, list):  # [2K14 music ids]: pick the carriers
        free = [n for n in CARRIERS if carrier_ok(n)]
        mp_, used = {}, set()
        for m in spec_:
            while free and carrier_ok(free[0]) in used: free.pop(0)
            if not free: print('out of carriers at 2K14 %d' % m); break
            n = free.pop(0); mp_[n] = m; used.add(carrier_ok(n))
    else:
        mp_ = {int(k): v for k, v in spec_.items()}
    change, done = {}, {}  # WWE '13 sound id -> 2K14 media id; carrier -> 2K14 music id
    for n, m in sorted(mp_.items()):
        sid13 = carrier_ok(n)
        sid14 = sound_of(ev14, H14, ev14, H14, m)
        if not sid13 or not sid14 or sid13 in change:
            print('%d <- %d: not found (%s, %s)' % (n, m, sid13, sid14)); continue
        done[n] = m
        t, q, l = H14[sid14]; media = struct.unpack('>I', ev14[q + 5 + 12:q + 5 + 16])[0]
        change[sid13] = media
        se = ms[media]
        pk['streams'][media] = (lambda se=se: M.read(se[3] * se[1], se[2]))
        print('music %d <- 2K14 %d: sound %08x plays media %08x (%d bytes)' % (n, m, sid13, media, se[2]))
    # media bank: the carriers' sounds re-pointed, stream-only; their prefetch parts dropped
    objs = mp.objs_list(mu13); out_objs = []; drop = set()
    for t, b in objs:
        oid = struct.unpack('>I', b[:4])[0]
        if oid in change:
            drop.add(struct.unpack('>I', b[12:16])[0])
            b = b[:8] + struct.pack('>I', 1) + struct.pack('>II', change[oid], change[oid]) + b[28:]  # (streamed: no offset / in-memory size fields)
        out_objs.append((t, b))
    c = chunks(mu13); s0, ln0 = c[b'DIDX']; ds0 = c[b'DATA'][0]
    didx = [struct.unpack('>3I', mu13[s0 + i * 12:s0 + i * 12 + 12]) for i in range(ln0 // 12)]
    keep = [(mid, mu13[ds0 + off:ds0 + off + size]) for mid, off, size in didx if mid not in drop]
    pk['banks'][MU] = mp.rebuild_media(mu13, out_objs, keep)
    print('Entrance_Music_M %d -> %d bytes' % (len(mu13), len(pk['banks'][MU])))
    mp.write_pck(out, pk, w13)
    json.dump({str(n): m for n, m in done.items()}, open(out + '.json', 'w'))

if __name__ == '__main__':
    main()
