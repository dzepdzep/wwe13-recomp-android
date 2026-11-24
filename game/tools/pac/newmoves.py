"""Adds 2K14-only moves to WWE '13: motions (ZMVM), the move database (m3 MDAT/PRM) and the move records (misc MOVS/WAZE).
  python3 newmoves.py <w13 pac dir> <k14 pac dir> <out pac dir> <id> [...]
      [--plist <dir with plist360.def/.arc and plist360_4x3.def/.arc>]   (written to <out>/plist360*.arc)
w13: mot06/07/08.pac, m3.pac, misc.pac (WWE '13); k14: mot07/08.pac, m3.pac, createmode.pac (WWE 2K14).

What goes where (WWE 2K14 and WWE '13 use the same formats and move ids):
- Motions: ZMVM groups are PACH(bank) > PACH(x: 0 attacker, 1 victim, 2.. phases) > PACH(move id) > BPE/ZLIB record.
  A move goes into the WWE '13 bank most of its 2K14 bank's other moves are in (new 2K14 banks: the bank of the
  nearest WWE '13 id), in every group that has that bank. Records are copied as they are.
- PRM part 0 (BE subs 0-7, LE copies 10-17): sub 0 (40-byte records), 1 (28), 4 (12), 6 (32) are sorted by
  (id, x); sub 4 is (id, x, u32 start, u16 count) into sub 5 (4-byte timed events), start counted from the first
  entry of the id's hundred (one segment per id // 100). Subs are stored BPE-compressed (the game reads the entry
  into a 4 MB heap).
- WAZE: 160-byte records added among the named ones (see waze_build), count at +4.
- plist360.arc caches every entry's offset and size (m3.pac and misc.pac included): a stale record makes the game
  read the old size (the PRM's parts 6-9 came out as garbage and the match load crashed).
"""
import sys,struct,bisect,collections,os,shutil
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from pactool import load, epk8_entries
from menuicons import unpack_any, pach_build
from bpe import bpe_wrap
from bpec import bpe_wrap_c

def pach(u):
  k=struct.unpack('<I',u[4:8])[0]; base=8+12*k
  return [(pid,u[base+o:base+o+s]) for pid,o,s in (struct.unpack('<3I',u[8+12*i:20+12*i]) for i in range(k))]
def unz(b): return unpack_any(b) if b[:4] in (b'BPE ',b'ZLIB') else b
def entry(d,g,n):
  e=[e for e in epk8_entries(d) if e[0]==g and e[1]==n][0]; return d[e[2]:e[2]+e[3]],e
def replace_entry(d,e,blob):
  d=bytearray(d); d+=b'\0'*(-len(d)%0x800); off=len(d); d+=blob; d+=b'\0'*(-len(d)%0x800)
  nlen=8 if e[4]&1 else 4
  # locate exact directory slot for this entry
  p=e[5]
  while True:
    if d[p:p+nlen].decode('latin1').strip()==e[1] and 0x4000+struct.unpack('<I',d[p+nlen:p+nlen+4])[0]*0x800==e[2]: break
    p+=nlen+8
  d[p+nlen:p+nlen+8]=struct.pack('<II',(off-0x4000)//0x800,(len(blob)+0xff)//0x100)
  return bytes(d)

def compact(d):
  """Lays the entries out again in directory order (drops the data replace_entry left behind)."""
  d=bytearray(d); out=bytearray(d[:0x4000]); p=0x800
  t0=0x4000+struct.unpack('<I',d[8:12])[0]; trailer=bytes(d[t0:t0+0x800])  # (the 'EOP5' block after the data)
  while p+12<=0x4000 and d[p:p+4].strip(b'\0'):
    cnt,kind=struct.unpack('<HH',d[p+4:p+8]); p+=12; nlen=8 if kind&1 else 4
    for _ in range(cnt*4//(nlen+8)):
      o,s=struct.unpack('<II',d[p+nlen:p+nlen+8]); blob=d[0x4000+o*0x800:0x4000+o*0x800+s*0x100]
      out+=b'\0'*(-len(out)%0x800); no=(len(out)-0x4000)//0x800; out+=blob
      out[p+nlen:p+nlen+8]=struct.pack('<II',no,s); p+=nlen+8
  end=len(out)-0x4000; end+=-end%0x800
  out[8:12]=struct.pack('<I',end); out+=b'\0'*(0x4000+end-len(out)); out+=trailer
  return bytes(out)

# ---------------- PRM ----------------
RS={0:40,1:28,4:12,6:32}
def prm_subs(d):
  x,e=entry(d,'MDAT','PRM'); top=pach(x); p0=pach(top[0][1])
  return x,e,top,{p:unz(b) for p,b in p0}
def keys(b,rs): return [b[i:i+4] for i in range(0,len(b),rs)]
def merge_prm(d13,d14,ids):
  x,e,top,A=prm_subs(d13); _,_,_,B=prm_subs(d14)
  ids=set(ids); added=collections.Counter()
  for k,rs in RS.items():
    a,al=bytearray(A[k]),bytearray(A[k+10]); b,bl=B[k],B[k+10]
    assert len(a)==len(al) and len(b)==len(bl)
    if k==4:
      s5,s5l=bytearray(A[5]),bytearray(A[15])
      # 2K14 absolute sub5 positions
      def absmap(b4):
        out={};cum=0;base=0
        for i in range(0,len(b4),12):
          mid,xx,st,n,_=struct.unpack('>HHIHH',b4[i:i+12])
          if st==0 and i: base=cum
          out[b4[i:i+4]]=(base+st,n); cum+=n
        return out
      ab14=absmap(b)
    new=[(b[i:i+rs],bl[i:i+rs]) for i in range(0,len(b),rs) if struct.unpack('>H',b[i:i+2])[0] in ids]
    for rec,recl in new:
      ka=keys(a,rs)
      if rec[:4] in ka: continue
      pos=bisect.bisect_left(ka,rec[:4])
      if k==4:
        # sub5 is kept in one segment per hundred ids (start counts from the segment's first entry)
        R=[struct.unpack('>HHIHH',a[i:i+12]) for i in range(0,len(a),12)]
        h=struct.unpack('>H',rec[:2])[0]//100
        blk=[i for i,r in enumerate(R) if r[0]//100==h]
        if blk:
          cum=sum(r[3] for r in R[:blk[0]]); seglen=sum(R[i][3] for i in blk)
        else:
          cum=sum(r[3] for r in R[:pos]); seglen=0
        st14,n=ab14[rec[:4]]
        ins=(cum+seglen)*4
        if blk: assert blk[0]<=pos<=blk[-1]+1
        s5[ins:ins]=B[5][st14*4:(st14+n)*4]; s5l[ins:ins]=B[15][st14*4:(st14+n)*4]
        rec=rec[:4]+struct.pack('>IH',seglen,n)+rec[10:12]
        recl=recl[:4]+struct.pack('<IH',seglen,n)+recl[10:12]
      a[pos*rs:pos*rs]=rec; al[pos*rs:pos*rs]=recl; added[k]+=1
    A[k],A[k+10]=bytes(a),bytes(al)
    if k==4: A[5],A[15]=bytes(s5),bytes(s5l)
  print('PRM records added',dict(added))
  p0=pach_build([(p,bpe_wrap_c(A[p]) if len(A[p]) else bpe_wrap(b'')) for p in sorted(A)],align=16)  # (compressed: the file is read into the match's memory)
  parts=[(p,b) for p,b in top]; parts[0]=(0,p0)
  blob=pach_build(parts,align=4)
  return replace_entry(d13,e,blob)

# ---------------- WAZE ----------------
# misc MOVS/WAZE: a 336-byte header (count at +4), then 160-byte records: 3 x u64 category masks (+0, which lists of
# Create-a-Moveset show the move), name (+24), owner superstar u16 (+128), id u16 (+152). The game lists only the first
# N records, N = the records with a name (sub_828DA118), so WWE '13 keeps its named moves first (by name) and its
# unnamed ones last. (The tools here once cut records at +40, 24 bytes off: added moves carried the next move's masks
# and sat after the unnamed block, never listed. waze_order.py / waze_fixmasks.py repair files made that way.)
WAZE_BASE, WAZE_RS = 336, 160
def waze_records(u):
  n=struct.unpack('<I',u[4:8])[0]
  return [u[WAZE_BASE+WAZE_RS*i:WAZE_BASE+WAZE_RS*(i+1)] for i in range(n)]
def waze_id(r): return struct.unpack('<H',r[152:154])[0]
def waze_name(r): return r[24:88].split(b'\0')[0].decode('latin1')
def waze_key(name):
  import re
  t=re.sub(r"[^A-Z0-9 ]","",name.upper())
  return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)",t)]
def waze_build(u,recs):
  """recs -> the WAZE entry: named records first (the original named block as it was, others placed by name), then
  the unnamed ones."""
  n0=struct.unpack('<I',u[4:8])[0]
  named=[r for r in recs if r[24]]; blank=[r for r in recs if not r[24]]
  first_blank=next((i for i,r in enumerate(recs) if not r[24]),len(recs))
  head=list(recs[:first_blank]); late=[r for r in recs[first_blank:] if r[24]]
  out=head
  for r in sorted(late,key=lambda r:waze_key(waze_name(r))):
    k=waze_key(waze_name(r)); j=len(out)
    while j>0 and waze_key(waze_name(out[j-1]))>k: j-=1
    out.insert(j,r)
  out=out+blank
  assert len(out)==len(recs)
  return u[:4]+struct.pack('<I',len(out))+u[8:WAZE_BASE]+b''.join(out)+u[WAZE_BASE+WAZE_RS*n0:]
def merge_waze(d13,d14,ids):
  u,e=entry(d13,'MOVS','WAZE'); v,_=entry(d14,'MOVS','WAZE')
  ra=waze_records(u); rb=waze_records(v)
  have={waze_id(r) for r in ra}
  add=[]; seen=set()
  for r in rb:
    m=waze_id(r)
    if m in ids and m not in have and m not in seen: add.append(r); seen.add(m)
  print('WAZE records added',len(add))
  return replace_entry(d13,e,waze_build(u,ra+add))

# ---------------- ZMVM ----------------
def zmvm_index(d):
  res={}
  for e in epk8_entries(d):
    if e[0]!='ZMVM': continue
    res[e[1]]=e
  return res
def merge_motions(w13dir,k14dir,outdir,ids):
  ids=set(ids)
  # 2K14 records: id -> {x: blob}; 2K14 bank of each id
  src=collections.defaultdict(dict); bank14={}; ids14=collections.defaultdict(set)
  for f in ('mot07','mot08'):
    d=load(os.path.join(k14dir,f+'.pac'))
    for g,e in zmvm_index(d).items():
      for bank,bb in pach(d[e[2]:e[2]+e[3]]):
        for x,xb in pach(bb):
          for mid,rec in pach(xb):
            ids14[bank].add(mid)
            if mid in ids:
              bank14[mid]=bank
              if x not in src[mid]: src[mid][x]=rec
  print('2K14 motions found for',len(src),'ids')
  pacs={f:load(os.path.join(w13dir,f+'.pac')) for f in ('mot06','mot07','mot08')}
  id2bank=collections.defaultdict(set)
  for f,d in pacs.items():
    for g,e in zmvm_index(d).items():
      for bank,bb in pach(d[e[2]:e[2]+e[3]]):
        for x,xb in pach(bb):
          for mid,_ in pach(xb): id2bank[mid].add(bank)
  known=sorted(id2bank); target={}
  for mid in src:
    # the '13 bank most of the 2K14 bank's other ids are in
    c=collections.Counter(b for o in ids14[bank14[mid]] if o in id2bank for b in id2bank[o])
    if c: target[mid]=c.most_common(1)[0][0]; continue
    i=bisect.bisect_left(known,mid)
    nb=min([n for n in (known[i-1] if i else None, known[i] if i<len(known) else None) if n is not None],key=lambda n:abs(n-mid))
    target[mid]=min(id2bank[nb])
    print('  new 2K14 bank',bank14[mid],'for',mid,'-> nearest id',nb,'bank',target[mid])
  outs={}
  for f,d in pacs.items():
    changed=False
    for g,e in list(zmvm_index(d).items()):
      banks=pach(d[e[2]:e[2]+e[3]]); mod=False; nb=[]
      for bank,bb in banks:
        mine=[m for m,t in target.items() if t==bank]
        if mine:
          xs=dict(pach(bb))
          for x in sorted(set(x for m in mine for x in src[m])):
            if x not in xs: xs[x]=pach_build([],align=4); print('  new x level',x,'in',f,g,bank)
            ents=dict(pach(xs[x]))
            for m in mine:
              if x in src[m] and m not in ents: ents[m]=src[m][x]
            xs[x]=pach_build(sorted(ents.items()),align=4)
          bb=pach_build(sorted(xs.items()),align=4); mod=True
        nb.append((bank,bb))
      if mod:
        d=replace_entry(d,e,pach_build(nb,align=4)); changed=True; print('  motions added to',f,g)
    if changed: outs[f]=d
  return outs,target

if __name__=='__main__':
  args=sys.argv[1:]; pl=None
  if '--plist' in args: i=args.index('--plist'); pl=args[i+1]; del args[i:i+2]
  w13,k14,out=args[:3]; ids=[int(a) for a in args[3:]]
  os.makedirs(out,exist_ok=True)
  outs,target=merge_motions(w13,k14,out,ids)
  ok=sorted(target)
  for f,d in outs.items(): open(os.path.join(out,f+'.pac'),'wb').write(compact(d))
  m3=merge_prm(load(os.path.join(w13,'m3.pac')),load(os.path.join(k14,'m3.pac')),ok)
  open(os.path.join(out,'m3.pac'),'wb').write(compact(m3))
  misc=merge_waze(load(os.path.join(w13,'misc.pac')),load(os.path.join(k14,'createmode.pac')),set(ok))
  open(os.path.join(out,'misc.pac'),'wb').write(compact(misc))
  open(os.path.join(out,'newmoves.txt'),'w').write('\n'.join(map(str,ok))+'\n')
  print('moves ported',len(ok))
  if pl:
    import plist
    names=[f for f in ('mot06','mot07','mot08','m3','misc') if os.path.exists(os.path.join(out,f+'.pac'))]
    for v in ('','_4x3'):
      arc=os.path.join(pl,'plist360%s.arc'%v)
      for i,f in enumerate(names):
        nxt=os.path.join(out,'plist360%s.arc.%d'%(v,i))
        plist.patch(os.path.join(pl,'plist360%s.def'%v),arc,'pac\\%s.pac'%f,os.path.join(out,f+'.pac'),nxt); arc=nxt
      shutil.copy(arc,os.path.join(out,'plist360%s.arc'%v))
