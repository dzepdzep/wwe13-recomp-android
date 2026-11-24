import struct
def fnv(s):
    h=2166136261
    for c in s.lower().encode(): h=(h*16777619)&0xffffffff; h^=c
    return h
def chunks(d):
    p=0; out={}
    while p<len(d):
        tag=d[p:p+4]; ln=struct.unpack('>I',d[p+4:p+8])[0]; out[tag]=(p+8,ln); p+=8+ln
    return out
def hirc(d, be=True):
    c=chunks(d); s,l=c[b'HIRC']; n=struct.unpack('>I',d[s:s+4])[0]; q=s+4; objs={}
    for i in range(n):
        t=d[q]; ln=struct.unpack('>I',d[q+1:q+5])[0]; oid=struct.unpack('>I',d[q+5:q+9])[0]
        objs[oid]=(t,q,ln); q+=5+ln
    return objs
