// Minimal LZX decode probe for Xbox 360 XCompress (0x0FF512ED) streams, using libmspack lzxd.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mspack.h"
#include "lzx.h"
struct mfile { unsigned char* buf; size_t len, pos; };
static struct mspack_file* m_open(struct mspack_system* s, const char* f, int m) { return NULL; }
static void m_close(struct mspack_file* f) {}
static int m_read(struct mspack_file* f, void* b, int n) { struct mfile* m=(struct mfile*)f; size_t r=m->len-m->pos; if((size_t)n<r) r=n; memcpy(b,m->buf+m->pos,r); m->pos+=r; return (int)r; }
static int m_write(struct mspack_file* f, void* b, int n) { struct mfile* m=(struct mfile*)f; size_t r=m->len-m->pos; if((size_t)n<r) r=n; memcpy(m->buf+m->pos,b,r); m->pos+=r; return (int)r; }
static int m_seek(struct mspack_file* f, off_t o, int mode) { return 0; }
static off_t m_tell(struct mspack_file* f) { return ((struct mfile*)f)->pos; }
static void m_msg(struct mspack_file* f, const char* fmt, ...) {}
static void* m_alloc(struct mspack_system* s, size_t n) { return malloc(n); }
static void m_free(void* p) { free(p); }
static void m_copy(void* a, void* b, size_t n) { memmove(b, a, n); }
static struct mspack_system sys = { m_open, m_close, m_read, m_write, m_seek, m_tell, m_msg, m_alloc, m_free, m_copy, NULL };
// returns bytes produced
size_t lzx_dec(unsigned char* in, size_t inlen, unsigned char* out, size_t outlen, int wbits, int frame, int reset) {
  struct mfile src={in,inlen,0}, dst={out,outlen,0};
  struct lzxd_stream* s = lzxd_init(&sys,(struct mspack_file*)&src,(struct mspack_file*)&dst,wbits,reset,frame,(off_t)outlen,0);
  if(!s) return 0;
  int r = lzxd_decompress(s,(off_t)outlen);
  size_t got = dst.pos;
  lzxd_free(s);
  if (r) fprintf(stderr, "lzxd err %d after %zu\n", r, got);
  return got;
}
#ifdef PROBE
int main(int argc,char**argv){
  FILE* f=fopen(argv[1],"rb"); fseek(f,0,2); long n=ftell(f); fseek(f,0,0); unsigned char* d=malloc(n); fread(d,1,n,f); fclose(f);
  int off=atoi(argv[2]), wbits=atoi(argv[3]); size_t outlen=strtoul(argv[4],0,0);
  unsigned char* o=calloc(1,outlen);
  size_t got=lzx_dec(d+off,n-off,o,outlen,wbits,0x8000,argc>6?atoi(argv[6]):0);
  printf("got %zu\n",got);
  FILE* w=fopen(argv[5],"wb"); fwrite(o,1,got,w); fclose(w);
}
#endif
