"""Static check of every move a ported superstar uses, so broken moves can be found without playing.

For each superstar in roster.txt (profile = the copied wrestler's WWE '13 profile + the roster's move overrides) and
each move field it checks:
  name     the move record (misc MOVS/WAZE, 160 bytes, name at +24, id u16 at +152): a move with no record is unknown
           to the game's menus.
  db       the move database (m3 MDAT/PRM part 0 sub 0) has the id.
  motion   ZMVM motion records exist for the id in a slot the game reads (slot = 2 * (id // 100 - base) + role, see
           zmvm_place.py) - attacker (even slot) and, when the database has a victim record (x = 1), victim (odd slot).
  timing   the motion comes from WWE 2K14 (WWE '13 has no motion of its own for the id), so the database records
           (subs 0/1/4/6) must be 2K14's (they carry the motion's length and hit frames; prm_motionmatch.py).
  links    database records of the id that name another move id (sub 0 +20, e.g. variants 5a00/5b00 -> 0x1916):
           that id must have a database record too.
Field labels come from what the stock roster uses in the same field (most common move names).
  move_audit.py <cfg.json> [superstar id ...]  -> report on stdout (and --json <file>)
cfg: {"w13_addon_slots": [json of WWE '13 add-on motions], "w13_chetc", "roster", "m3", "m3_entry", "waze_pacs": [...], "mot_pacs": [...], "w13_slots" (json),
      "k14_prm": [["path", "ENTRY"], ...] (first source wins)}
"""
import sys, os, json, struct, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prm_rebuild as P, newmoves as N
from pactool import load
from newmoves import pach, zmvm_index
import port_moves

MOVE_AREA = 0x240


def waze_names(paths):
    names = {}
    for p in paths:
        u, _ = N.entry(load(p), 'MOVS', 'WAZE'); n = struct.unpack('<I', u[4:8])[0]
        for r in N.waze_records(u):
            names.setdefault(N.waze_id(r), N.waze_name(r))
    return names


def motions(paths):
    m = collections.defaultdict(set)
    for p in paths:
        d = load(p); f = os.path.basename(p)
        for g, e in zmvm_index(d).items():
            for b, bb in pach(d[e[2]:e[2] + e[3]]):
                for x, xb in pach(bb):
                    for mid, _ in pach(xb): m[mid].add((b, x))
    return m


def bases(w13):
    c = collections.defaultdict(collections.Counter)
    for f, g, b, x, i in w13:
        if i != 65535 and i >= 2000: c[b][i // 100 - x // 2] += 1
    return {b: cc.most_common(1)[0][0] for b, cc in c.items()}


def main(cfg, only, jout=None):
    C = json.load(open(cfg))
    W = {i: struct.unpack('<590H', v[:1180]) for i, v in port_moves.pros(C['w13_chetc']).items() if len(v) >= 1180}
    names = waze_names(C['waze_pacs'])
    _, _, top, A = P.subs(load(C['m3']), C['m3_entry']); R = P.records(A)
    mid = lambda k: struct.unpack('>H', k[:2])[0]
    db = collections.defaultdict(dict)
    for k in R:
        for key, v in R[k].items(): db[mid(key)].setdefault(k, {})[key] = v
    K = []
    for p, e in C['k14_prm']:
        _, _, _, B = P.subs(load(p), e); KR = P.records(B); kd = collections.defaultdict(dict)
        for k in KR:
            for key, v in KR[k].items(): kd[mid(key)].setdefault(k, {})[key] = v
        K.append((os.path.basename(p), kd))
    w13 = json.load(open(C['w13_slots'])); native = {r[4] for r in w13}; base = bases(w13)
    addon = {r[4] for p in C.get('w13_addon_slots', []) for r in json.load(open(p))}  # WWE '13 add-on motions (own banks)
    mot = motions(C['mot_pacs'])
    # field labels: stock superstars' moves per field
    stock = [i for i in W if 100 <= i < 400 and W[i][0x120] == i]
    # moves the stock roster uses are animated by WWE '13 itself (some from files not indexed here, e.g. the
    # compressed disc dlc_mot01.pac): not checked for motions
    stock_ids = {W[i][f] for i in stock for f in range(MOVE_AREA // 2)}
    for p in C.get('w13_dlc_chetc', []):  # WWE '13 add-on superstars' profiles (their moves come with the add-ons)
        for v in port_moves.pros(p).values():
            if len(v) >= 1180: stock_ids |= set(struct.unpack('<590H', v[:1180])[:MOVE_AREA // 2])
    addon |= stock_ids
    label = {}
    for f in range(MOVE_AREA // 2):
        c = collections.Counter(names.get(W[i][f], '') for i in stock if W[i][f] >= 1000)
        c.pop('', None)
        label[f * 2] = ' / '.join(n for n, _ in c.most_common(3))
    # what is normal per field: share of the stock values that have a move record / a database record
    named = {f: (sum(1 for i in stock if W[i][f] in names) / max(1, sum(1 for i in stock if 1000 <= W[i][f] != 0xffff)))
             for f in range(MOVE_AREA // 2)}
    indb = {f: (sum(1 for i in stock if W[i][f] in db) / max(1, sum(1 for i in stock if 1000 <= W[i][f] != 0xffff)))
            for f in range(MOVE_AREA // 2)}
    # roster
    report = {}; summary = collections.Counter()
    for line in open(C['roster'], encoding='latin1'):
        if not line[:1].isdigit(): continue
        fl = line.rstrip('\r\n').split('|'); sid = int(fl[0])
        if only and sid not in only: continue
        prof = list(W[int(fl[1])])
        if len(fl) > 8 and fl[8]:
            for kv in fl[8].split(','):
                o, v = kv.split(':'); prof[int(o, 16) // 2] = int(v, 16)
        issues = []
        for f in range(MOVE_AREA // 2):
            v = prof[f]
            if v < 1000 or v == 0xffff: continue
            prob = []
            if v not in names and named[f] > 0.5: prob.append('no move record (WAZE)')
            if v not in db or 0 not in db[v]:
                if indb[f] > 0.5: prob.append('not in move database')
            else:
                roles = {struct.unpack('>H', k[2:4])[0] & 0xff for k in db[v][0]}
                b = None
                have = mot.get(v, set())
                for role in ([] if v in addon else sorted(r for r in roles if r in (0, 1))):
                    ok = [s for s in have if s[1] % 2 == role and s[0] in base and s[1] == 2 * (v // 100 - base[s[0]]) + role]
                    if not ok and not (v < 2000 and any(s[1] % 2 == role for s in have)):
                        prob.append('no %s motion in a readable slot%s' % ('attacker' if role == 0 else 'victim',
                                    (' (has %s)' % sorted(have)) if have else ''))
                if v not in native and v not in addon and have:
                    src = next(((n, kd[v]) for n, kd in K if 0 in kd.get(v, {})), None)
                    if src:
                        for k in (0, 1, 4, 6):
                            a = {kk: (vv[0] if k != 4 else (vv[0][:4] + vv[0][10:], vv[2])) for kk, vv in db[v].get(k, {}).items()}
                            bb = {kk: (vv[0] if k != 4 else (vv[0][:4] + vv[0][10:], vv[2])) for kk, vv in src[1].get(k, {}).items()}
                            if a != bb: prob.append('timing: sub %d differs from %s' % (k, src[0])); break
                for key, rec in ([] if v in addon else db[v][0].items()):
                    link = struct.unpack('>H', rec[0][20:22])[0]
                    if link and link != v and link < 0xff00 and link not in db: prob.append('links to unknown id %d' % link)
            if prob:
                issues.append({'field': '0x%x' % (f * 2), 'what': label[f * 2], 'id': v, 'name': names.get(v, '?'), 'problems': prob})
                for p in prob: summary[p.split(' (')[0].split(':')[0]] += 1
        report[sid] = {'name': fl[2], 'issues': issues}
    for sid, r in report.items():
        print('%d %s: %d problem move(s)' % (sid, r['name'], len(r['issues'])))
        for i in r['issues']:
            print('   %s [%s] %d %s: %s' % (i['field'], i['what'][:40], i['id'], i['name'], '; '.join(i['problems'])))
    print('summary:', dict(summary))
    if jout: json.dump({'labels': label, 'report': report}, open(jout, 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv[1:]; jout = None
    if '--json' in a: i = a.index('--json'); jout = a[i + 1]; del a[i:i + 2]
    main(a[0], {int(x) for x in a[1:]}, jout)
