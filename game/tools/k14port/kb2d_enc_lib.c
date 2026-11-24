/* kb2d_enc: re-encodes a Bink 2 movie (e.g. a WWE 2K14 KB2f titantron) as revision KB2d, the one WWE '13 plays.
 * Every frame is a key frame of intra macroblocks (the KB2d intra syntax: 4-bit quantizer, fixed-width DC
 * differences, band-coded AC). Decoding uses bink2_decode.c (LGPL 2.1+ Bink 2 decoder), included here.
 *   kb2d_enc <in.bik2> <out.bik2> [q 0-15 (default 7)] */
#include <stdio.h>
#include <wchar.h>
static int _wfopen_s(FILE **f, const wchar_t *p, const wchar_t *m) { (void)p; (void)m; *f = NULL; return 1; }
#define _fseeki64 fseeko
#define _ftelli64 ftello
#include "bink2_decode.c"

/* ---- bit writer (little-endian bit order, as the decoder reads) ---- */
typedef struct { uint8_t *buf; size_t cap, bits; } BW;
static void bw_put(BW *w, unsigned v, int n)
{
    for (int i = 0; i < n; i++) {
        size_t byte = w->bits >> 3;
        if (byte + 8 >= w->cap) { w->cap = w->cap * 2 + 4096; w->buf = realloc(w->buf, w->cap); memset(w->buf + byte, 0, w->cap - byte); }
        if ((v >> i) & 1) w->buf[byte] |= (uint8_t)(1u << (w->bits & 7));
        w->bits++;
    }
}
static void bw_align32(BW *w) { while (w->bits & 31) bw_put(w, 0, 1); }
static int bits_for(int v) { int n = 0; while (v >> n) n++; return n; }
static void put_width(BW *w, int n) { if (n >= 7) { bw_put(w, 7, 3); bw_put(w, n - 7, 2); } else bw_put(w, n, 3); }

/* ---- the decoder's IDCT as a matrix, and its inverse ---- */
static double fwd[64][64]; /* coef = fwd * pixels (pixels centred: the DC term carries the mean) */
static void make_fwd(void)
{
    double m[64][128];
    for (int k = 0; k < 64; k++) {
        float b[64] = { 0 };
        b[k] = 1.0f;
        for (int i = 0; i < 8; i++) bink2f_idct_1d(b + i, 8);
        for (int i = 0; i < 8; i++) bink2f_idct_1d(b + i * 8, 1);
        for (int p = 0; p < 64; p++) m[p][k] = b[p];
    }
    for (int i = 0; i < 64; i++) for (int j = 0; j < 64; j++) m[i][64 + j] = i == j;
    for (int col = 0; col < 64; col++) {
        int piv = col;
        for (int r = col + 1; r < 64; r++) if (fabs(m[r][col]) > fabs(m[piv][col])) piv = r;
        for (int j = 0; j < 128; j++) { double t = m[col][j]; m[col][j] = m[piv][j]; m[piv][j] = t; }
        double d = m[col][col];
        for (int j = 0; j < 128; j++) m[col][j] /= d;
        for (int r = 0; r < 64; r++) if (r != col) { double f = m[r][col]; if (f != 0) for (int j = 0; j < 128; j++) m[r][j] -= f * m[col][j]; }
    }
    for (int i = 0; i < 64; i++) for (int j = 0; j < 64; j++) fwd[i][j] = m[i][64 + j];
}
static void dct8(const uint8_t *src, int stride, double coef[64])
{
    double px[64];
    for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) px[y * 8 + x] = src[y * stride + x];
    for (int i = 0; i < 64; i++) { double s = 0; for (int p = 0; p < 64; p++) s += fwd[i][p] * px[p]; coef[i] = s; }
}

/* ---- one component of an intra macroblock: 16 (luma) or 4 (chroma) blocks ---- */
static int qlevel(double v) { return v >= 0 ? (int)(v + 0.5) : -(int)(-v + 0.5); }

static void encode_ac(BW *w, const uint8_t *scan, const uint8_t *bands, int lv[4][64])
{
    int idx = 1;
    for (int b = 0; idx < 64; b++) {
        const int last = bands[b];
        int rest = 0, mx = 0;
        for (int k = idx; k < 64; k++) for (int j = 0; j < 4; j++) rest |= lv[j][scan[k]];
        if (!rest) { bw_put(w, 1, 1); return; }
        bw_put(w, 0, 1);
        for (int k = idx; k <= last; k++) for (int j = 0; j < 4; j++) { int a = abs(lv[j][scan[k]]); if (a > mx) mx = a; }
        int n = bits_for(mx);
        put_width(w, n);
        if (!n) { idx = last + 1; continue; }
        for (; idx <= last; idx++) {
            for (int j = 0; j < 4; j++) bw_put(w, abs(lv[j][scan[idx]]), n);
            for (int j = 0; j < 4; j++) if (lv[j][scan[idx]]) bw_put(w, lv[j][scan[idx]] < 0, 1);
        }
    }
}

/* DCs: chooses the differences so that the decoder's prediction lands on the wanted values; writes them. */
static void encode_dc(BW *w, Bink2Context *c, int is_luma, int q, int flags, const double want[16])
{
    const int n = is_luma ? 16 : 4;
    const float dq = bink2f_dc_quant[q];
    const int first = (flags & 0x20) && (flags & 0x80);
    int r[16] = { 0 };
    float *dc = c->current_dc[c->mb_pos].dc[c->comp];
    for (int i = 0; i < n; i++) {
        float tdc[16];
        for (int k = 0; k < 16; k++) tdc[k] = k < i ? r[k] * dq : 0;
        if (first) tdc[0] -= 1024.0f; /* (bits >= 10 below: the extra DC isn't coded) */
        bink2f_predict_dc(c, is_luma, 0, 2047, flags, tdc);
        double t = want[i] < 0 ? 0 : want[i] > 2047 ? 2047 : want[i];
        int v = qlevel((t - dc[i]) / dq);
        if (v > 1023) v = 1023; if (v < -1023) v = -1023;
        r[i] = v;
    }
    int mx = 0;
    for (int i = 0; i < n; i++) if (abs(r[i]) > mx) mx = abs(r[i]);
    int bits = bits_for(mx);
    if (first && bits < 11 - q_dc_bits[q]) bits = 11 - q_dc_bits[q];
    if (bits > 10) bits = 10;
    put_width(w, bits);
    if (bits)
        for (int j = 0; j < n; j += 4) {
            for (int i = 0; i < 4; i++) bw_put(w, abs(r[i + j]), bits);
            for (int i = 0; i < 4; i++) if (r[i + j]) bw_put(w, r[i + j] < 0, 1);
        }
    /* the decoder's state: the final DCs */
    float tdc[16];
    for (int k = 0; k < 16; k++) tdc[k] = k < n ? r[k] * dq : 0;
    if (first) tdc[0] -= 1024.0f;
    bink2f_predict_dc(c, is_luma, 0, 2047, flags, tdc);
}

static void encode_luma(BW *w, Bink2Context *c, const uint8_t *src, int stride, int q, int flags)
{
    double coef[16][64], want[16];
    int lv[4][64];
    bw_put(w, q, 4);
    for (int k = 0; k < 16; k++) {
        const int pos = luma_repos[k];
        dct8(src + (pos & 3) * 8 + (pos >> 2) * 8 * stride, stride, coef[k]);
        want[k] = coef[k][0] * 8.0;
    }
    encode_dc(w, c, 1, q, flags, want);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            for (int p = 0; p < 64; p++)
                lv[j][p] = p ? qlevel(coef[i * 4 + j][p] / (bink2f_ac_quant[q] * bink2f_luma_intra_qmat[p])) : 0;
        for (int j = 0; j < 4; j++) for (int p = 1; p < 64; p++) { if (lv[j][p] > 1023) lv[j][p] = 1023; if (lv[j][p] < -1023) lv[j][p] = -1023; }
        encode_ac(w, bink2f_luma_scan, bink2d_bands_luma_intra, lv);
    }
}
static void encode_chroma(BW *w, Bink2Context *c, const uint8_t *src, int stride, int q, int flags)
{
    double coef[4][64], want[4];
    int lv[4][64];
    bw_put(w, q, 4);
    for (int j = 0; j < 4; j++) {
        dct8(src + (j & 1) * 8 + (j >> 1) * 8 * stride, stride, coef[j]);
        want[j] = coef[j][0] * 8.0;
    }
    encode_dc(w, c, 0, q, flags, want);
    for (int j = 0; j < 4; j++)
        for (int p = 0; p < 64; p++) {
            int v = p ? qlevel(coef[j][p] / (bink2f_ac_quant[q] * bink2f_chroma_qmat[p])) : 0;
            lv[j][p] = v > 1023 ? 1023 : v < -1023 ? -1023 : v;
        }
    encode_ac(w, bink2f_chroma_scan, bink2d_bands_chroma, lv);
}

/* one frame: planes padded to a multiple of 32 (luma) */
static void encode_frame(BW *w, Bink2Context *c, uint8_t *pl[3], int st[3], int W, int H, int q, int qc)
{
    const int hA = (H + 31) & ~31, s0 = (H / 2 + 16) & 0xFFFFFFE0;
    const int mbs = (W + 31) / 32;
    bw_put(w, 0x20, 32);           /* frame flags: DC fill threshold 0x20 (as WWE '13's), deblocking on */
    bw_put(w, 0, 32);              /* the second slice's offset (filled in below) */
    for (int s = 0; s < 2; s++) {
        const int start = s ? s0 : 0, end = s ? hA : s0;
        if (s) {
            bw_align32(w);
            uint32_t off = (uint32_t)(w->bits >> 3);
            memcpy(w->buf + 4, &off, 4);
        }
        memset(c->prev_dc, 0, mbs * sizeof(*c->prev_dc));
        memset(c->current_dc, 0, mbs * sizeof(*c->current_dc));
        for (int y = start; y < end; y += 32) {
            for (int x = 0; x < W; x += 32) {
                int flags = 0;
                if (y == start) flags |= 0x80;
                if (!x) flags |= 0x20;
                if (x == 32) flags |= 0x200;
                if (x + 32 >= W) flags |= 0x40;
                c->mb_pos = x / 32;
                c->current_dc[c->mb_pos].block_type = INTRA_BLOCK;
                c->comp = 0; encode_luma(w, c, pl[0] + y * st[0] + x, st[0], q, flags);
                c->comp = 1; encode_chroma(w, c, pl[2] + y / 2 * st[2] + x / 2, st[2], qc, flags); /* V */
                c->comp = 2; encode_chroma(w, c, pl[1] + y / 2 * st[1] + x / 2, st[1], qc, flags); /* U */
            }
            FFSWAP(DCPredict *, c->current_dc, c->prev_dc);
        }
    }
    bw_align32(w);
}

static uint8_t *read_all(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n); if (fread(b, 1, *n, f) != *n) { fclose(f); return NULL; } fclose(f); return b;
}
static void wl32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }

int kb2d_main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "kb2d_enc <in.bik2> <out.bik2> [q] [qc]\n"); return 1; }
    const int q = argc > 3 ? atoi(argv[3]) : 7, qc = argc > 4 ? atoi(argv[4]) : q;
    size_t n; uint8_t *in = read_all(argv[1], &n);
    WCHAR err[256];
    if (!in) { fprintf(stderr, "can't read %s\n", argv[1]); return 1; }
    Bink2Reader *r = bink2_open_memory(in, n, err, 256);
    if (!r) { fprintf(stderr, "%ls\n", err); return 1; }
    const int W = r->w, H = r->h, F = r->frames;
    const int WA = (W + 31) & ~31, HA = (H + 31) & ~31;
    uint8_t *pl[3]; int st[3] = { WA, WA / 2, WA / 2 };
    pl[0] = calloc(WA * HA, 1); pl[1] = calloc(WA * HA / 4, 1); pl[2] = calloc(WA * HA / 4, 1);
    make_fwd();
    Bink2Context c = { 0 };
    const int mbs = WA / 32;
    c.current_dc = calloc(mbs, sizeof(DCPredict)); c.prev_dc = calloc(mbs, sizeof(DCPredict));
    uint8_t **pk = calloc(F, sizeof(*pk)); uint32_t *ps = calloc(F, 4), mx = 0;
    size_t total = 0;
    for (int f = 0; f < F; f++) {
        if (!bink2_next(r)) fprintf(stderr, "frame %d: damaged\n", f);
        for (int p = 0; p < 3; p++) {
            const int pw = p ? W / 2 : W, ph = p ? H / 2 : H, pwa = p ? WA / 2 : WA, pha = p ? HA / 2 : HA;
            for (int y = 0; y < pha; y++) {
                const uint8_t *s = r->data[r->cur][p] + (y < ph ? y : ph - 1) * r->linesize[p];
                for (int x = 0; x < pwa; x++) pl[p][y * st[p] + x] = s[x < pw ? x : pw - 1];
            }
        }
        BW w = { 0 };
        encode_frame(&w, &c, pl, st, W, H, q, qc);
        pk[f] = w.buf; ps[f] = (uint32_t)(w.bits >> 3);
        if (ps[f] > mx) mx = ps[f];
        total += ps[f];
    }
    /* the file: 44-byte header, frame index, frames */
    const uint32_t data0 = 44 + 4 * (F + 1);
    FILE *o = fopen(argv[2], "wb");
    uint8_t h[44];
    memcpy(h, in, 44);
    h[3] = 'd';
    wl32(h + 4, (uint32_t)(data0 + total - 8));
    wl32(h + 8, F); wl32(h + 12, mx); wl32(h + 16, F);
    wl32(h + 36, 0); wl32(h + 40, 0); /* no flags, no audio */
    fwrite(h, 1, 44, o);
    uint32_t at = data0;
    for (int f = 0; f <= F; f++) { uint8_t b[4]; wl32(b, at | (f < F ? 1 : 0)); fwrite(b, 1, 4, o); if (f < F) at += ps[f]; }
    for (int f = 0; f < F; f++) fwrite(pk[f], 1, ps[f], o);
    fclose(o);
    fprintf(stderr, "%s: %dx%d, %d frames, q %d/%d, %zu bytes (largest frame %u)\n", argv[2], W, H, F, q, qc,
            (size_t)data0 + total, mx);
    return 0;
}
