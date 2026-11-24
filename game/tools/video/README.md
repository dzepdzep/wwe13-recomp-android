kb2d_enc: re-encodes a Bink 2 movie (WWE 2K14 titantrons are revision KB2f) as KB2d, the revision WWE '13 plays.
All frames intra (key frames). Needs bink2_decode.c + bink2/ (the LGPL 2.1+ Bink 2 decoder with a KB2d description,
from the WWE12-PC project's port/src/modmaker/video) next to it; kept on the PC VM in ~/kb2.

  gcc -O2 -o kb2d_enc kb2d_enc.c -lm
  ./kb2d_enc <in.bik2> <out.bik2> [q (default 7)] [chroma q]
  ./kb2_cmp <original.bik2> <converted.bik2> [frame]   (PSNR via the reference decoders; dumps a.ppm / b.ppm)

Tested: 2K14 360.bik2 (Sandow) -> 11 MB KB2d, luma PSNR 35 dB, plays on WWE '13's arena screens on the Odin.
