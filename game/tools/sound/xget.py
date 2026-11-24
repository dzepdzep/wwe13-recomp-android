import sys,struct,os
iso,out=sys.argv[1],sys.argv[2]; want=set(sys.argv[3:])
f=open(iso,'rb'); base=0x2080000
f.seek(base+32*2048); assert f.read(20)==b'MICROSOFT*XBOX*MEDIA'
root,rsize=struct.unpack('<II',f.read(8))
def entries(sec,size):
    f.seek(base+sec*2048); d=f.read(size); res=[]; seen=set(); stack=[0]
    while stack:
        o=stack.pop()
        if o in seen or o*4+14>len(d): continue
        seen.add(o)
        l,r,s,sz,a,n=struct.unpack('<HHIIBB',d[o*4:o*4+14])
        if l==0xFFFF: continue
        res.append((d[o*4+14:o*4+14+n].decode('latin1'),s,sz,a))
        if l: stack.append(l)
        if r: stack.append(r)
    return res
def find(path):
    sec,size=root,rsize
    for part in path.strip('/').split('/'):
        m=[e for e in entries(sec,size) if e[0].lower()==part.lower()]
        if not m: return None
        _,sec,size,a=m[0]
    return sec,size
for p in sorted(want):
    r=find(p)
    if not r: print('missing',p); continue
    dst=os.path.join(out,p.strip('/')); os.makedirs(os.path.dirname(dst),exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst)==r[1]: continue
    f.seek(base+r[0]*2048); open(dst,'wb').write(f.read(r[1])); print('ok',p,r[1])
