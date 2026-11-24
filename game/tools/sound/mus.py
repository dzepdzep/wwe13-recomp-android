import struct
from bnk import *
def trace(ev, mu, mid):
    """[(event name, [(action type, target, sound id, media id, sound body)])] for Play_MUS_<mid>_a_b"""
    E1=hirc(ev); M=hirc(mu); res=[]
    for a in range(3):
        for b in range(3):
            name='Play_MUS_%04d_%d_%d'%(mid,a,b); eid=fnv(name)
            if eid not in E1: continue
            t,q,ln=E1[eid]; body=ev[q+5:q+5+ln]; na=struct.unpack('>I',body[4:8])[0]; acts=[]
            for k in range(na):
                aid=struct.unpack('>I',body[8+4*k:12+4*k])[0]; t2,q2,l2=E1[aid]; ab=ev[q2+5:q2+5+l2]
                atype=struct.unpack('>H',ab[4:6])[0]; tgt=struct.unpack('>I',ab[6:10])[0]
                snd=None; media=None
                if tgt in M and M[tgt][0]==5:
                    t3,q3,l3=M[tgt]; rb=mu[q3+5:q3+5+l3]
                    # children list at the end: u32 count then ids (search for a sound child)
                    for o in range(len(rb)-8,8,-1):
                        cand=struct.unpack('>I',rb[o:o+4])[0]
                        if cand in M and M[cand][0]==2:
                            snd=cand; break
                    if snd:
                        t4,q4,l4=M[snd]; sb=mu[q4+5:q4+5+l4]; media=struct.unpack('>I',sb[12:16])[0]
                acts.append((hex(atype),'%08x'%tgt,'%08x'%(snd or 0),'%08x'%(media or 0)))
            res.append((name,acts))
    return res
