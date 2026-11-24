import struct,sys
def akpk(path):
    f=open(path,'rb'); h=f.read(8); assert h[:4]==b'AKPK'
    hs=struct.unpack('>I',h[4:8])[0]; d=h+f.read(hs)
    ver,lang_sz,bank_sz,stream_sz,ext_sz=struct.unpack('>5I',d[8:28]); p=28
    langs=d[p:p+lang_sz]; p+=lang_sz
    out={}
    for name,sz in (('bank',bank_sz),('stream',stream_sz)):
        n=struct.unpack('>I',d[p:p+4])[0]; ents=[]
        for i in range(n):
            ents.append(struct.unpack('>5I',d[p+4+i*20:p+24+i*20]))
        out[name]=ents; p+=sz
    return langs,out
if __name__=='__main__':
    langs,o=akpk(sys.argv[1]); print(langs)
    for k,v in o.items():
        print(k,len(v)); 
        for e in v[:8]: print('  id %08x block %d size %d off %d lang %d'%e)
