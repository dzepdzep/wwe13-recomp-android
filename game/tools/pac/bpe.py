import struct
def bpe_expand(src, out_size, big_endian_size=True):
    out=bytearray(); p=0; n=len(src)
    while p<n and len(out)<out_size:
        left=list(range(256)); right=[0]*256
        c=0
        while c<256:
            count=src[p]; p+=1
            if count>127:
                c+=count-127; count=0
            if c==256: break
            for _ in range(count+1):
                left[c]=src[p]; p+=1
                if left[c]!=c:
                    right[c]=src[p]; p+=1
                c+=1
        size=(src[p]<<8)|src[p+1] if big_endian_size else src[p]|(src[p+1]<<8); p+=2
        stack=[]
        i=0
        while i<size or stack:
            if stack: b=stack.pop()
            else: b=src[p]; p+=1; i+=1
            if left[b]==b: out.append(b)
            else:
                stack.append(right[b]); stack.append(left[b])
    return bytes(out), p

# The game's files use blocks of at most ~2.3 KB output (MOVS: 134,760 bytes in 60 blocks); its decoder likely has a
# small fixed buffer. A 64 KB stored block hung the game at character select, so stored blocks stay small.
BLOCK = 2048

def bpe_store(data):
    """A valid BPE stream with no pairs (every byte a literal): what the game's BPE reader expects, uncompressed.
    Blocks of up to 0xFFFF bytes: pair table = skip 128 (0xFF), byte 128 as itself (0x80), skip 127 (0xFE), u16 LE size, the bytes."""
    out = bytearray(); p = 0
    while p < len(data):
        n = min(BLOCK, len(data) - p)
        out += b'\xff\x80\xfe' + bytes((n & 0xFF, n >> 8)) + data[p:p+n]
        p += n
    return bytes(out)

def bpe_wrap(data):
    """'BPE ' header (as in the game's files) around a stored stream."""
    import struct
    body = bpe_store(data)
    return b'BPE ' + struct.pack('<III', 0x100, len(body), len(data)) + body
