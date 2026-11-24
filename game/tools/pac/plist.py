"""plist360.arc: the game's cache of every pac's directory ('PAC_LIST_ARC_EPAC_BIG'). The game takes entry offsets and
sizes from here, not from the pac itself, so a rebuilt pac whose entries moved or grew needs its record updated.

Layout: plist360.def lists the pacs (line 0 = header; pac i = line i+1). The .arc is a sequence of records
[ff ff][u16 BE pac index] followed by groups [4-char tag][u16 BE dword count << 4][u16 BE 0x1000 = 8-char names][u32 0] and big-endian
entries [8- or 4-char name][u32 offset (0x800 units)][u32 size (0x100 units)] - the pac's EPK8 directory, byte-swapped.

  python3 plist.py <plist360.def> <plist360.arc> <pac path as in .def, e.g. pac\\ch\\ch322.pac> <new.pac> <out.arc>
"""
import struct, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pactool import load, epk8_entries

def patch(def_path, arc_path, pac_name, new_pac, out_path):
    names = open(def_path, 'rb').read().decode('latin1').split('\r\n')
    idx = [n.lower() for n in names].index(pac_name.lower()) - 1
    a = bytearray(open(arc_path, 'rb').read())
    rec = a.find(b'\xff\xff' + struct.pack('>H', idx))
    assert rec >= 0, 'record not found'
    nxt = a.find(b'\xff\xff', rec + 4)
    new = {(g, n): ((o - 0x4000) // 0x800, s // 0x100) for g, n, o, s, _, _ in epk8_entries(load(new_pac))}
    p = rec + 4; changed = 0
    while p < (nxt if nxt > 0 else len(a)):
        tag = a[p:p+4].decode('latin1').strip(); cnt, kind = struct.unpack('>HH', a[p+4:p+8]); p += 12
        nlen = 8 if kind & 0x1000 else 4
        n_entries = (cnt >> 4) * 4 // (nlen + 8)
        for _ in range(n_entries):
            name = a[p:p+nlen].decode('latin1').strip('\x00 ')
            o, s = struct.unpack('>II', a[p+nlen:p+nlen+8])
            no, ns = new[(tag, name)]
            if (o, s) != (no, ns):
                print(f'  {tag}/{name}: offset {o:#x} -> {no:#x}, size {s:#x} -> {ns:#x}'); changed += 1
            a[p+nlen:p+nlen+8] = struct.pack('>II', no, ns)
            p += nlen + 8
    open(out_path, 'wb').write(a)
    print(f'{pac_name}: record {idx}, {changed} entr{"y" if changed == 1 else "ies"} updated')



def add(def_path, arc_path, pac_name, new_pac, out_def, out_arc):
    """A pac the game never had (a new wrestler's ch###.pac): appended to the .def and given a record at the end of the .arc."""
    text = open(def_path, 'rb').read().decode('latin1')
    names = text.split('\r\n')
    if pac_name.lower() in [n.lower() for n in names]:
        raise SystemExit(f'{pac_name} already listed')
    while names and names[-1] == '':
        names.pop()
    idx = len(names) - 1
    names.append(pac_name)
    open(out_def, 'wb').write(('\r\n'.join(names) + '\r\n').encode('latin1'))
    groups = {}
    for g, n, o, s, kind, gi in epk8_entries(load(new_pac)):
        groups.setdefault((gi, g, kind), []).append((n, (o - 0x4000) // 0x800, s // 0x100))
    rec = bytearray(b'\xff\xff' + struct.pack('>H', idx))
    for (gi, g, kind), ents in sorted(groups.items()):
        nlen = 8 if kind & 1 else 4
        rec += g.encode('latin1').ljust(4, b' ') + struct.pack('>HHI', (len(ents) * (nlen + 8) // 4) << 4, 0x1000 if nlen == 8 else 0, 0)
        for n, o, s in ents:
            rec += n.encode('latin1').ljust(nlen, b' ')[:nlen] + struct.pack('>II', o, s)
    a = open(arc_path, 'rb').read()
    assert a[-4:] == b'\xff\xff\xff\xff', 'no end marker'
    open(out_arc, 'wb').write(a[:-4] + bytes(rec) + a[-4:])  # (before the ff ff ff ff end marker)
    print(f'{pac_name}: added as record {idx} ({len(rec)} bytes)')



def replace(def_path, arc_path, pac_name, new_pac, out_arc):
    """A pac whose entries were added/removed: its whole record is regenerated."""
    names = open(def_path, 'rb').read().decode('latin1').split('\r\n')
    idx = [n.lower() for n in names].index(pac_name.lower()) - 1
    a = open(arc_path, 'rb').read()
    start = a.find(b'\xff\xff' + struct.pack('>H', idx))
    end = a.find(b'\xff\xff' + struct.pack('>H', idx + 1), start + 4)
    if end < 0:
        end = len(a) - 4
    groups = {}
    for g, n, o, s, kind, gi in epk8_entries(load(new_pac)):
        groups.setdefault((gi, g, kind), []).append((n, (o - 0x4000) // 0x800, s // 0x100))
    rec = bytearray(b'\xff\xff' + struct.pack('>H', idx))
    for (gi, g, kind), ents in sorted(groups.items()):
        nlen = 8 if kind & 1 else 4
        rec += g.encode('latin1').ljust(4, b' ') + struct.pack('>HHI', (len(ents) * (nlen + 8) // 4) << 4, 0x1000 if nlen == 8 else 0, 0)
        for n, o, s in ents:
            rec += n.encode('latin1').ljust(nlen, b' ')[:nlen] + struct.pack('>II', o, s)
    open(out_arc, 'wb').write(a[:start] + bytes(rec) + a[end:])
    print(f'{pac_name}: record {idx} regenerated ({end - start} -> {len(rec)} bytes)')


if __name__ == '__main__':
    if sys.argv[1] == 'replace':  # replace <def> <arc> <pac name> <new.pac> <out.arc>
        replace(*sys.argv[2:7])
        sys.exit(0)
    if sys.argv[1] == 'add':  # add <def> <arc> <pac name> <new.pac> <out.def> <out.arc>
        add(*sys.argv[2:8])
    else:
        patch(*sys.argv[1:6])
