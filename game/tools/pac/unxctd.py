# Unpacks a WWE '13 XCTD (0x0FF512ED) .pac to its raw EPK8 data. Format (found here): 36-byte header, then chunks of
# [u16 BE compressed size][LZX data]; each group of chunks is an independent LZX stream (window 2^17, 32 KB frames)
# ending with a zero-length block; the group's output size is the sum of its LZX block lengths.
import struct,subprocess,sys,os
T=os.path.dirname(os.path.abspath(__file__))
def find_start(d):
    """the first chunk-walk start that ends exactly at EOF and whose first LZX block is a plausible frame
    (some headers also happen to walk to EOF from a wrong start)"""
    for start in _walk_starts(d):
        p=start; ch=[]
        while p<len(d) and len(ch)<4:
            c=struct.unpack('>H',d[p:p+2])[0]; p+=2
            if c==0: continue
            ch.append(d[p:p+c]); p+=c
        r=run('xlzxd',ch,4*32768,'/tmp/fs.bin').stderr.split()
        if len(r)>1 and r[0]=='BLK' and 0<int(r[1])<=32768: return start
    raise ValueError('no chunk layout found')
def _walk_starts(d):
    # the header length varies; the chunk walk [u16 BE size][data] must end exactly at the end of the file
    L=len(d)
    for start in range(16,8192):
        p=start
        while p+2<=L:
            c=struct.unpack('>H',d[p:p+2])[0]
            if c==0 and p>start and p+4<=L and 0<struct.unpack('>H',d[p+2:p+4])[0]<=0x9000:
                p+=2; continue  # (big files: a zero u16 before some chunks, e.g. at 0x80000 - skipped)
            if c==0 or c>0x9000: break
            p+=2+c
        if p==L: yield start
def chunks_of(d):
    assert d[:4]==b'\x0f\xf5\x12\xed', 'not an XCTD pac'
    p=find_start(d); ch=[]
    while p<len(d):
        c=struct.unpack('>H',d[p:p+2])[0]; p+=2
        if c==0: continue
        ch.append(d[p:p+c]); p+=c
    return ch
def run(tool,chunks,outlen,out='/tmp/q.bin'):
    open('/tmp/g.bin','wb').write(b''.join(chunks)+b'\0'*64)
    return subprocess.run([os.path.join(T,tool),'/tmp/g.bin','0','17',str(outlen),out],capture_output=True,text=True)
def unpack(path,outpath):
    ch=chunks_of(open(path,'rb').read())
    i=0; out=bytearray(); groups=0
    while i<len(ch):
        # the LZX block lengths from chunk i on (a stream may continue into the next group's bytes as garbage)
        r=run('xlzxd',ch[i:i+512],512*32768)
        cums=[]; c=0
        for x in r.stderr.split('\n'):
            if x.startswith('BLK'):
                v=int(x.split()[1])
                if v==0 or v>(1<<25): break  # (a block may span many 32 KB frames)
                c+=v; cums.append(c)
        # the group: the longest prefix of blocks that decodes exactly from ceil(n/32768) chunks, where that
        # chunk count matches (every chunk = one 32 KB output frame) - largest first
        best=None
        for n in sorted(set(cums),reverse=True):
            j=i+(n+32767)//32768
            if j>len(ch): continue
            rr=run('xlzx',ch[i:j],n,'/tmp/part.bin')
            if 'err' not in rr.stderr and rr.stdout.strip()==f'got {n}':
                # the next group must start cleanly (or this is the end)
                if j==len(ch) or run('xlzxd',ch[j:j+1],32768).stderr.startswith('BLK'):
                    best=(n,j); break
        assert best, ('no group at chunk',i)
        n,j=best
        run('xlzx',ch[i:j],n,'/tmp/part.bin'); part=open('/tmp/part.bin','rb').read()
        assert len(part)==n
        out+=part; groups+=1; i=j
    open(outpath,'wb').write(out); return groups,len(out)
if __name__=='__main__':
    print(unpack(sys.argv[1],sys.argv[2]))
