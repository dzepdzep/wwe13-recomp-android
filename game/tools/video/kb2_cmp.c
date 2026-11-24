/* kb2_cmp <a.bik2> <b.bik2> [frame to dump]: decodes both (any Bink 2 revision) and prints the luma PSNR of b vs a;
 * dumps the frame as a.ppm / b.ppm (BGRA -> RGB). */
#include <stdio.h>
#include <wchar.h>
static int _wfopen_s(FILE **f, const wchar_t *p, const wchar_t *m) { (void)p; (void)m; *f = NULL; return 1; }
#define _fseeki64 fseeko
#define _ftelli64 ftello
#include "bink2_decode.c"
static uint8_t *rd(const char *p, size_t *n) { FILE *f = fopen(p, "rb"); fseek(f, 0, 2); *n = ftell(f); fseek(f, 0, 0); uint8_t *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b; }
static void ppm(const char *p, const uint8_t *bgra, int w, int h) { FILE *f = fopen(p, "wb"); fprintf(f, "P6 %d %d 255\n", w, h); for (int i = 0; i < w * h; i++) { fputc(bgra[i*4+2], f); fputc(bgra[i*4+1], f); fputc(bgra[i*4], f); } fclose(f); }
int main(int argc, char **argv)
{
    size_t na, nb; uint8_t *a = rd(argv[1], &na), *b = rd(argv[2], &nb); WCHAR e[256];
    Bink2Reader *ra = bink2_open_memory(a, na, e, 256), *rb = bink2_open_memory(b, nb, e, 256);
    if (!ra || !rb) { fprintf(stderr, "open: %ls\n", e); return 1; }
    int dump = argc > 3 ? atoi(argv[3]) : -1, W = ra->w, H = ra->h;
    double worst = 1e9, sum = 0; int bad = 0;
    uint8_t *ba = malloc(W * H * 4), *bb = malloc(W * H * 4);
    for (int f = 0; f < ra->frames && f < rb->frames; f++) {
        if (!bink2_next(ra) || !bink2_next(rb)) bad++;
        double se = 0;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { int d = ra->data[ra->cur][0][y * ra->linesize[0] + x] - rb->data[rb->cur][0][y * rb->linesize[0] + x]; se += d * d; }
        double psnr = se ? 10 * log10(255.0 * 255 * W * H / se) : 99;
        if (psnr < worst) worst = psnr; sum += psnr;
        if (f == dump) { bink2_bgra(ra, ba); bink2_bgra(rb, bb); ppm("a.ppm", ba, W, H); ppm("b.ppm", bb, W, H); }
    }
    printf("frames %d/%d, damaged %d, luma PSNR avg %.2f worst %.2f\n", ra->frames, rb->frames, bad, sum / ra->frames, worst);
    return 0;
}
