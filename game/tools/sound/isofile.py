import struct,os
ISO="/sessions/rcw-01vrgngxxjbzh8hczr5zpvxm/mnt/microsoft_xbox360_w/WWE 2K14 (USA, Europe).iso"
import sys
sys.argv=['x',ISO,'/tmp']
exec(open('/sessions/rcw-01vrgngxxjbzh8hczr5zpvxm/xget.py').read().split('for p in sorted(want)')[0])
class IsoFile:
    def __init__(self,path):
        sec,self.size=find(path); self.base=base+sec*2048
    def read(self,off,n):
        f.seek(self.base+off); return f.read(n)
