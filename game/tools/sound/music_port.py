"""Ports WWE 2K14 entrance music into WWE '13's sound/Music.pck (Wwise AKPK, bank version 65 in both games).
2K14's music objects (event Play_MUS_<m>_a_b -> actions -> random container -> sound -> streamed XMA media) are
copied as they are (same Wwise version; their parent actor-mixer 078ce9f2 is the same in both games) and the events
are renamed Play_MUS_<n>_a_b for the new music number n; the containers join the actor-mixer's children, the media's
prefetch part goes into Entrance_Music_M's DIDX/DATA and the whole media into the package's streamed files.
  music_port.py <w13 Music.pck> <out Music.pck> <map.json {n: 2k14 music id}>
"""
import json, struct, sys
ARGS = sys.argv[1:]
PREFETCH = False
# WWE '13 music numbers nobody's profile uses (generic Create-an-Entrance tracks): their prefetch parts make room
VICTIMS = list(range(519, 579)) + list(range(421, 468))
sys.path.insert(0, '/sessions/rcw-01vrgngxxjbzh8hczr5zpvxm')
from bnk import fnv, chunks, hirc
from isofile import IsoFile

def read_pck(path):
    f = open(path, 'rb'); h = f.read(8); hs = struct.unpack('>I', h[4:8])[0]; d = h + f.read(hs)
    ver, ls, bs, ss, es = struct.unpack('>5I', d[8:28]); p = 28
    lang = d[p:p + ls]; p += ls
    banks = [struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0])]; p += bs
    streams = [struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0])]; p += ss
    ext = d[p:p + es]
    def blob(e):
        f.seek(e[3] * e[1]); return f.read(e[2])
    return dict(ver=ver, lang=lang, ext=ext, banks={e[0]: blob(e) for e in banks}, streams={e[0]: (lambda e=e: blob(e)) for e in streams})

def write_pck(path, pk, src):
    """Keeps the original file's layout: every bank / stream stays at its offset (a changed bank must still fit its
    blocks), new streams go at the end, only the header's tables change."""
    import shutil
    f = open(src, 'rb'); h = f.read(8); hs = struct.unpack('>I', h[4:8])[0]; d = h + f.read(hs)
    ver, ls, bs, ss, es = struct.unpack('>5I', d[8:28]); p = 28 + ls
    obanks = {e[0]: e for e in (struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0]))}; p += bs
    ostreams = {e[0]: e for e in (struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0]))}
    first_data = min(e[3] * e[1] for e in list(obanks.values()) + list(ostreams.values()))
    shutil.copyfile(src, path)
    blk = 2048
    with open(path, 'r+b') as o:
        o.seek(0, 2); end = o.tell()
        bents = []
        for k, b in sorted(pk['banks'].items()):
            e = obanks[k]
            room = -(-e[2] // blk) * blk
            if len(b) > room:
                raise SystemExit('bank %08x grew past its blocks (%d > %d)' % (k, len(b), room))
            o.seek(e[3] * blk); o.write(b + b'\0' * (room - len(b)) if len(b) < e[2] else b)
            bents.append((k, blk, len(b), e[3], e[4]))
        sents = []
        for k, g in sorted(pk['streams'].items()):
            if k in ostreams:
                sents.append(ostreams[k]); continue
            b = g() if callable(g) else g
            pos = -(-end // blk); o.seek(pos * blk); o.write(b); end = pos * blk + len(b)
            sents.append((k, blk, len(b), pos, 0))
        bl = struct.pack('>I', len(bents)) + b''.join(struct.pack('>5I', *e) for e in bents)
        sl = struct.pack('>I', len(sents)) + b''.join(struct.pack('>5I', *e) for e in sorted(sents))
        hsize = 20 + ls + len(bl) + len(sl) + es
        head = b'AKPK' + struct.pack('>I', hsize) + struct.pack('>5I', ver, ls, len(bl), len(sl), es) + d[28:28 + ls] + bl + sl + pk['ext']
        if len(head) > first_data:
            raise SystemExit('header too big for the room before the data (%d > %d)' % (len(head), first_data))
        o.seek(0); o.write(head + b'\0' * (first_data - len(head)))

def rebuild_bank(d, hirc_objs, extra_didx=None):
    """bank d with its HIRC replaced by hirc_objs [(type, body)] and (for the media bank) DIDX/DATA extended."""
    c = chunks(d); out = b''
    p = 0
    data_add = b''
    while p < len(d):
        tag = d[p:p + 4]; ln = struct.unpack('>I', d[p + 4:p + 8])[0]; body = d[p + 8:p + 8 + ln]
        if tag == b'HIRC':
            body = struct.pack('>I', len(hirc_objs)) + b''.join(bytes([t]) + struct.pack('>I', len(b)) + b for t, b in hirc_objs)
        elif tag == b'DIDX' and extra_didx:
            dl = c[b'DATA'][1]; base = (dl + 15) & ~15; add = b''; dat = b''
            for mid, pre in extra_didx:
                off = base + len(dat); add += struct.pack('>3I', mid, off, len(pre)); dat += pre + b'\0' * (-len(pre) % 16)
            body += add; data_add = (b'\0' * (base - dl)) + dat
        elif tag == b'DATA' and extra_didx:
            body += data_add
        out += tag + struct.pack('>I', len(body)) + body
        p += 8 + ln
    return out

def rebuild_media(d, hirc_objs, media):
    """the media bank with its HIRC replaced and DIDX/DATA rebuilt from [(media id, prefetch bytes)]"""
    out = b''; p = 0
    didx = b''; data = b''
    for mid, b in media:
        data += b'\0' * (-len(data) % 16); didx += struct.pack('>3I', mid, len(data), len(b)); data += b
    while p < len(d):
        tag = d[p:p + 4]; ln = struct.unpack('>I', d[p + 4:p + 8])[0]; body = d[p + 8:p + 8 + ln]
        if tag == b'HIRC':
            body = struct.pack('>I', len(hirc_objs)) + b''.join(bytes([t]) + struct.pack('>I', len(b)) + b for t, b in hirc_objs)
        elif tag == b'DIDX':
            body = didx
        elif tag == b'DATA':
            body = data
        out += tag + struct.pack('>I', len(body)) + body
        p += 8 + ln
    return out

def objs_list(d):
    H = hirc(d); return [(t, d[q + 5:q + 5 + l]) for oid, (t, q, l) in sorted(H.items(), key=lambda x: x[1][1])]

def main():
    w13, out, mp = ARGS[0], ARGS[1], {int(k): v for k, v in json.load(open(ARGS[2])).items()}
    pk = read_pck(w13)
    EV, MU = 0xf4757ccf, 0x9a4847cb
    ev13, mu13 = pk['banks'][EV], pk['banks'][MU]
    ev_objs, mu_objs = objs_list(ev13), objs_list(mu13)
    ids13 = {struct.unpack('>I', b[:4])[0] for t, b in ev_objs + mu_objs}
    E_S, M = IsoFile('/sound/Game_Sound_E_S.pck'), IsoFile('/sound/Game_Sound_M.pck')
    # 2K14 banks / streams
    def lut(m):
        h = m.read(0, 8); hs = struct.unpack('>I', h[4:8])[0]; d = m.read(0, 8 + hs)
        ver, ls, bs, ss, es = struct.unpack('>5I', d[8:28]); p = 28 + ls
        banks = {e[0]: e for e in (struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0]))}; p += bs
        streams = {e[0]: e for e in (struct.unpack('>5I', d[p + 4 + i * 20:p + 24 + i * 20]) for i in range(struct.unpack('>I', d[p:p + 4])[0]))}
        return banks, streams
    esb, _ = lut(E_S); mb, ms = lut(M)
    e = esb[0xe85dda21]; ev14 = E_S.read(e[3] * e[1], e[2])
    e = mb[MU]; mu14 = M.read(e[3] * e[1], e[2])
    H14 = hirc(ev14)
    c = chunks(mu14); s, ln = c[b'DIDX']; ds = c[b'DATA'][0]
    didx14 = {struct.unpack('>I', mu14[s + i * 12:s + i * 12 + 4])[0]: struct.unpack('>3I', mu14[s + i * 12:s + i * 12 + 12]) for i in range(ln // 12)}
    body = lambda oid: (H14[oid][0], ev14[H14[oid][1] + 5:H14[oid][1] + 5 + H14[oid][2]])
    add_ev, add_mu, new_rs, didx_add, done = [], [], [], [], set()
    replace = {}  # (WWE '13 event id -> new body): a music number WWE '13 already has (the game only plays known ones)
    for n, m in sorted(mp.items()):
        for a in range(3):
            for b in range(3):
                neid = fnv('Play_MUS_%04d_%d_%d' % (n, a, b))
                eid = fnv('Play_MUS_%04d_%d_%d' % (m, a, b))
                if eid not in H14:
                    eid = fnv('Play_MUS_%04d_0_0' % m)
                    if neid not in ids13 or eid not in H14:
                        continue
                t, eb = body(eid)
                if neid in ids13:
                    replace[neid] = struct.pack('>I', neid) + eb[4:]
                else:
                    add_ev.append((t, struct.pack('>I', neid) + eb[4:]))
                for k in range(struct.unpack('>I', eb[4:8])[0]):
                    aid = struct.unpack('>I', eb[8 + 4 * k:12 + 4 * k])[0]
                    if aid in done or aid in ids13: continue
                    done.add(aid); ta, ab = body(aid)
                    if ab[-4:] == struct.pack('>I', 0xe85dda21):  # (play actions name the bank to load: 2K14's -> WWE '13's)
                        ab = ab[:-4] + struct.pack('>I', MU)
                    add_ev.append((ta, ab))
                    tgt = struct.unpack('>I', ab[6:10])[0]
                    if tgt in done or tgt in ids13 or tgt not in H14: continue
                    done.add(tgt); tr, rb = body(tgt); add_mu.append((tr, rb)); new_rs.append(tgt)
                    for o in range(0, len(rb) - 3):  # (the container's sounds)
                        cid = struct.unpack('>I', rb[o:o + 4])[0]
                        if cid in H14 and H14[cid][0] == 2 and cid not in done and cid not in ids13:
                            done.add(cid); ts, sb = body(cid)
                            media = struct.unpack('>I', sb[12:16])[0]
                            if PREFETCH and media in didx14:
                                _, off, size = didx14[media]; didx_add.append((media, mu14[ds + off:ds + off + size]))
                            else:  # streamed only (no prefetch part in the bank: its memory pool has little room)
                                sb = sb[:8] + struct.pack('>I', 1) + sb[12:20] + struct.pack('>II', 0, 0) + sb[28:]
                            add_mu.append((ts, sb))
                            se = ms[media]
                            pk['streams'][media] = (lambda se=se: M.read(se[3] * se[1], se[2]))
                            print('  %d: media %08x (%d bytes, prefetch %d)' % (n, media, se[2], didx14.get(media, (0, 0, 0))[2]))
    # the actor-mixer 078ce9f2 gets the new containers as children (a sorted u32 list after its u32 count)
    out_mu = []
    for t, b in mu_objs:
        if struct.unpack('>I', b[:4])[0] == 0x078ce9f2:
            for o in range(4, len(b) - 8, 1):
                cnt = struct.unpack('>I', b[o:o + 4])[0]
                if 100 < cnt < 2000 and o + 4 + 4 * cnt <= len(b):
                    kids = [struct.unpack('>I', b[o + 4 + 4 * i:o + 8 + 4 * i])[0] for i in range(cnt)]
                    if kids == sorted(kids) and all(k in ids13 for k in kids):
                        kids = sorted(kids + new_rs)
                        b = b[:o] + struct.pack('>I', len(kids)) + b''.join(struct.pack('>I', k) for k in kids) + b[o + 4 + 4 * cnt:]
                        print('actor-mixer children %d -> %d' % (cnt, len(kids)))
                        break
            else:
                raise SystemExit('children list not found')
        out_mu.append((t, b))
    ev_objs = [(t, replace.get(struct.unpack('>I', b[:4])[0], b)) for t, b in ev_objs]
    print('%d WWE 13 events now play 2K14 music' % len(replace))
    pk['banks'][EV] = rebuild_bank(ev13, ev_objs + add_ev)
    new_mu = rebuild_bank(mu13, out_mu + add_mu, didx_add)
    if len(new_mu) > len(mu13):
        # The game loads Entrance_Music_M into a pool sized for the original: it must not grow. Room comes from
        # dropping the in-bank prefetch part of WWE '13 sounds nobody's profile uses (they then stream from the
        # start, as the ported ones do).
        need = len(new_mu) - len(mu13)
        c = chunks(mu13); s0, ln0 = c[b'DIDX']; ds0 = c[b'DATA'][0]
        didx = [struct.unpack('>3I', mu13[s0 + i * 12:s0 + i * 12 + 12]) for i in range(ln0 // 12)]
        H13 = hirc(mu13); E13 = hirc(ev13)
        victims = []
        for vm in VICTIMS:
            eid = fnv('Play_MUS_%04d_0_0' % vm)
            if eid not in E13: continue
            t, q, l = E13[eid]; eb = ev13[q + 5:q + 5 + l]
            aid = struct.unpack('>I', eb[8:12])[0]; t, q, l = E13[aid]; tgt = struct.unpack('>I', ev13[q + 5 + 6:q + 5 + 10])[0]
            t, q, l = H13[tgt]; rb = mu13[q + 5:q + 5 + l]
            for o in range(len(rb) - 3):
                sid = struct.unpack('>I', rb[o:o + 4])[0]
                if sid in H13 and H13[sid][0] == 2:
                    victims.append(sid); break
            freed = sum(x[2] for x in didx if any(struct.unpack('>I', mu13[H13[v][1] + 5 + 12:H13[v][1] + 5 + 16])[0] == x[0] for v in victims))
            if freed - 12 * len(victims) > need + 64: break
        vmedia = {struct.unpack('>I', mu13[H13[v][1] + 5 + 12:H13[v][1] + 5 + 16])[0] for v in victims}
        out2 = []
        for t, b in out_mu:
            if struct.unpack('>I', b[:4])[0] in victims:
                b = b[:8] + struct.pack('>I', 1) + b[12:20] + struct.pack('>II', 0, 0) + b[28:]
            out2.append((t, b))
        keep = [(mid, mu13[ds0 + off:ds0 + off + size]) for mid, off, size in didx if mid not in vmedia]
        new_mu = rebuild_media(mu13, out2 + add_mu, keep + didx_add)
        print('media bank: %d sound(s) stream without prefetch now (%s) – %d -> %d bytes (original %d)' % (
            len(victims), ', '.join('%08x' % v for v in victims), len(mu13) + need, len(new_mu), len(mu13)))
        if len(new_mu) > len(mu13):
            raise SystemExit('media bank still too big')
    pk['banks'][MU] = new_mu
    write_pck(out, pk, w13)
    print('%s: %d events/actions, %d containers/sounds, %d media added' % (out, len(add_ev), len(add_mu), len(didx_add)))

if __name__ == '__main__':
    main()
