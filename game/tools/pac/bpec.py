"""Byte-pair encoder (Philip Gage's algorithm), output readable by bpe.bpe_expand (u16 LE block sizes)."""
def _compress_block(buf, maxchars=220, threshold=3):
    left = list(range(256)); right = [0]*256
    buf = bytearray(buf)
    ever = set(buf)  # a byte value that was ever in the data stays a literal (it may still be referenced by pairs)
    while True:
        # count pairs
        counts = {}
        prev = None
        for i in range(len(buf)-1):
            k = (buf[i] << 8) | buf[i+1]
            counts[k] = counts.get(k, 0) + 1
        free = [c for c in range(256) if c not in ever and left[c] == c]
        if not counts or not free: break
        best = max(counts, key=counts.get)
        if counts[best] < threshold: break
        a, b = best >> 8, best & 0xFF
        code = free[0]
        out = bytearray(); i = 0; n = len(buf)
        while i < n:
            if i+1 < n and buf[i] == a and buf[i+1] == b:
                out.append(code); i += 2
            else:
                out.append(buf[i]); i += 1
        buf = out
        left[code] = a; right[code] = b; ever.add(code)
    # pair table, as bpe_expand reads it: a byte > 127 skips (byte-127) literal codes and then ONE entry follows;
    # a byte <= 127 is followed by (byte+1) entries; an entry = left code (+ right code if it is a pair).
    tab = bytearray(); c = 0
    def entry(k):
        tab.append(left[k])
        if left[k] != k: tab.append(right[k])
    while c < 256:
        if left[c] == c:
            r = 0
            while c + r < 256 and left[c + r] == c + r and r < 128: r += 1
            tab.append(r + 127); c += r
            if c == 256: break
            entry(c); c += 1
        else:
            n = 0
            while c + n < 256 and left[c + n] != c + n and n < 128: n += 1
            tab.append(n - 1)
            for k in range(c, c + n): entry(k)
            c += n
    return bytes(tab) + bytes((len(buf) & 0xFF, len(buf) >> 8)) + bytes(buf)

def bpe_compress(data, block=4096, maxchars=230):
    """Blocks end at `block` bytes or when `maxchars` distinct byte values are in them (Gage's rule), so every block
    keeps free codes for pairs. (MOVS: 98,788 bytes vs the game's own 98,890.)"""
    out = bytearray(); p = 0
    while p < len(data):
        seen = set(); q = p
        while q < len(data) and q - p < block:
            if data[q] not in seen:
                if len(seen) >= maxchars: break
                seen.add(data[q])
            q += 1
        out += _compress_block(data[p:q]); p = q
    return bytes(out)

def bpe_wrap_c(data, block=4096):
    import struct
    body = bpe_compress(data, block)
    return b'BPE ' + struct.pack('<III', 0x100, len(body), len(data)) + body
