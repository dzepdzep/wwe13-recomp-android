"""Copies base.apk entries and adds extra files: .so stored uncompressed and 16 KiB-aligned, others deflated."""
import sys, zipfile, os
src, extra_dir, dst = sys.argv[1:4]
def add_aligned(zout, arcname, data, align):
    info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    # Local header = 30 + len(name) + len(extra); pad extra so data starts aligned.
    offset = zout.fp.tell() + 30 + len(arcname.encode())
    pad = (-offset) % align
    info.extra = b'\0' * pad
    zout.writestr(info, data)
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, 'w') as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.compress_type == zipfile.ZIP_STORED:
            add_aligned(zout, item.filename, data, 4)
        else:
            zout.writestr(item, data, compress_type=zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(extra_dir):
        for f in sorted(files):
            p = os.path.join(root, f)
            arc = os.path.relpath(p, extra_dir).replace(os.sep, '/')
            data = open(p, 'rb').read()
            if arc.endswith('.so'):
                add_aligned(zout, arc, data, 16384)
            else:
                info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
                zout.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
