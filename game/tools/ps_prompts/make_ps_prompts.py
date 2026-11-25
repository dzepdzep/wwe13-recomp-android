#!/usr/bin/env python3
"""Draws PlayStation button icons (our own artwork, no game data) and encodes them as DXT5 blocks for the game's
button-icon sheets (256x512 DXT5 in pac/menu/font*.pac and 512x512 DXT5 in pac/game2dHD.pac). Writes src/ps_prompts_data.h:
RGBA patches (rectangle, replace or draw-over), which ps_prompts.cpp composites into the decompressed texture and
re-encodes as DXT5 at run time. Only these cells change; the rest of the sheet stays the game's.

  python3 make_ps_prompts.py <out header> [preview.png]
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 4  # supersampling
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Cells of the sheet (x0, y0, x1, y1), multiples of 4. Layout of the game's sheet:
# row 0: X Y B A, row 1: LS RS LS-click RS-click, rows 2-3: d-pad (kept), LB RB BACK, LT RT START, RS-left/right.
CELLS = {
    "square":   (0, 0, 44, 40),     # X
    "triangle": (64, 0, 108, 40),   # Y
    "circle":   (128, 0, 172, 40),  # B
    "cross":    (192, 0, 236, 40),  # A
    "L":        (0, 40, 48, 88),    # LS
    "R":        (64, 40, 112, 88),  # RS
    "L3":       (128, 40, 176, 88), # LS click
    "R3":       (192, 40, 240, 88), # RS click
    "L1":       (0, 180, 60, 212),  # LB
    "R1":       (76, 180, 136, 212),# RB
    "SHARE":    (152, 176, 200, 216),  # BACK
    "L2":       (0, 216, 32, 264),  # LT
    "R2":       (76, 216, 112, 264),# RT
    "OPTIONS":  (152, 220, 200, 260),  # START
    "R_LR":     (0, 304, 52, 352),  # RS with left/right arrows
}
SYMBOL = {"cross": (0x7C, 0xB2, 0xE8), "circle": (0xFF, 0x66, 0x66), "square": (0xF0, 0x8C, 0xD8),
          "triangle": (0x40, 0xE2, 0xA0)}


def face_button(d, w, h, kind):
    r = min(w, h) / 2 - 1.5 * SS
    cx, cy = w / 2, h / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 18, 22, 255), outline=(170, 170, 180, 255), width=int(1.5 * SS))
    s = r * 0.5
    col = SYMBOL[kind] + (255,)
    lw = int(r * 0.17)
    if kind == "cross":
        d.line([cx - s, cy - s, cx + s, cy + s], fill=col, width=lw)
        d.line([cx - s, cy + s, cx + s, cy - s], fill=col, width=lw)
    elif kind == "circle":
        d.ellipse([cx - s, cy - s, cx + s, cy + s], outline=col, width=lw)
    elif kind == "square":
        d.rectangle([cx - s * 0.9, cy - s * 0.9, cx + s * 0.9, cy + s * 0.9], outline=col, width=lw)
    else:
        d.polygon([(cx, cy - s * 1.05), (cx + s, cy + s * 0.7), (cx - s, cy + s * 0.7)], outline=col, width=lw)


def label(d, cx, cy, text, size, fill=(240, 240, 240, 255)):
    f = ImageFont.truetype(FONT, int(size))
    b = d.textbbox((0, 0), text, font=f)
    d.text((cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2), text, font=f, fill=fill)


def stick(d, w, h, text, arrows=False):
    r = min(w, h) / 2 - 2 * SS
    cx, cy = w / 2, h / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(28, 28, 32, 255), outline=(170, 170, 180, 255), width=int(1.5 * SS))
    d.ellipse([cx - r * 0.72, cy - r * 0.72, cx + r * 0.72, cy + r * 0.72], outline=(80, 80, 90, 255), width=int(SS))
    label(d, cx, cy, text, r * (0.85 if len(text) == 1 else 0.7))
    if arrows:
        red = (220, 30, 30, 255)
        a = r * 0.45
        d.polygon([(cx - r - a * 0.1, cy), (cx - r + a, cy - a), (cx - r + a, cy + a)], fill=red)
        d.polygon([(cx + r + a * 0.1, cy), (cx + r - a, cy - a), (cx + r - a, cy + a)], fill=red)


def pill(d, w, h, text, tall=False):
    m = 1.5 * SS
    rad = (w if tall else h) * 0.3
    d.rounded_rectangle([m, m, w - m, h - m], radius=rad, fill=(225, 225, 230, 255), outline=(90, 90, 100, 255),
                        width=int(1.5 * SS))
    label(d, w / 2, h / 2, text, min(w, h) * (0.42 if tall else 0.6), fill=(20, 20, 24, 255))


def small_button(d, w, h, kind):
    m = 2 * SS
    d.rounded_rectangle([m, h * 0.22, w - m, h * 0.78], radius=h * 0.25, fill=(225, 225, 230, 255),
                        outline=(90, 90, 100, 255), width=int(1.5 * SS))
    c = (20, 20, 24, 255)
    cx, cy = w / 2, h / 2
    if kind == "OPTIONS":  # three lines
        for k in (-1, 0, 1):
            d.line([cx - w * 0.16, cy + k * h * 0.1, cx + w * 0.16, cy + k * h * 0.1], fill=c, width=int(1.6 * SS))
    else:  # share: arrow up out of a tray
        d.line([cx, cy - h * 0.16, cx, cy + h * 0.06], fill=c, width=int(1.6 * SS))
        d.polygon([(cx, cy - h * 0.2), (cx - w * 0.08, cy - h * 0.08), (cx + w * 0.08, cy - h * 0.08)], fill=c)
        d.line([(cx - w * 0.15, cy - h * 0.02), (cx - w * 0.15, cy + h * 0.13), (cx + w * 0.15, cy + h * 0.13),
                (cx + w * 0.15, cy - h * 0.02)], fill=c, width=int(1.4 * SS))


def draw(name, w, h):
    im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    W, H = w * SS, h * SS
    if name in SYMBOL:
        face_button(d, W, H, name)
    elif name in ("L", "R", "L3", "R3"):
        stick(d, W, H, name)
    elif name == "R_LR":
        stick(d, W, H, "R", arrows=True)
    elif name in ("L1", "R1"):
        pill(d, W, H, name)
    elif name in ("L2", "R2"):
        pill(d, W, H, name, tall=True)
    else:
        small_button(d, W, H, name)
    return im.resize((w, h), Image.LANCZOS)


# ---- DXT5 encoding ----
def rgb565(c):
    return (int(c[0]) >> 3) << 11 | (int(c[1]) >> 2) << 5 | (int(c[2]) >> 3)


def unpack565(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return np.array([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], float)


def enc_block(px):  # px: 4x4x4 uint8
    p = px.reshape(16, 4).astype(float)
    a = p[:, 3]
    a0, a1 = int(a.max()), int(a.min())
    if a0 == a1:
        a1 = max(0, a0 - 1) if a0 > 0 else 0
        if a0 == 0: a0 = 1
    pal = [a0, a1] + [((7 - i) * a0 + i * a1) / 7 for i in range(1, 7)]
    ai = [int(np.argmin([abs(v - q) for q in pal])) for v in a]
    abits = 0
    for i, v in enumerate(ai): abits |= v << (3 * i)
    ab = bytes([a0, a1]) + abits.to_bytes(6, "little")
    rgb = p[:, :3]
    w = a > 16
    pts = rgb[w] if w.any() else rgb
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T) if len(pts) > 1 else np.eye(3)
    axis = np.linalg.eigh(cov)[1][:, -1] if np.ptp(pts) > 0 else np.array([1., 1, 1])
    t = (pts - mean) @ axis
    c0, c1 = mean + axis * t.max(), mean + axis * t.min()
    c0, c1 = np.clip(c0, 0, 255), np.clip(c1, 0, 255)
    v0, v1 = rgb565(c0), rgb565(c1)
    if v0 < v1: v0, v1 = v1, v0
    if v0 == v1:
        idx = [0] * 16
    else:
        e0, e1 = unpack565(v0), unpack565(v1)
        cp = [e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3]
        idx = [int(np.argmin([((c - q) ** 2).sum() for q in cp])) for c in rgb]
    cbits = 0
    for i, v in enumerate(idx): cbits |= v << (2 * i)
    return ab + v0.to_bytes(2, "little") + v1.to_bytes(2, "little") + cbits.to_bytes(4, "little")


def stick_center(w, h, text):
    """Overlay for the stick-direction icons: a new stick centre with L / R over the game's "LS" / "RS" text; the
    game's ring and red arrows around it stay."""
    im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    W, H = w * SS, h * SS
    r = H / 2 - 0.5 * SS
    d.rounded_rectangle([W / 2 - (W - H) / 2 - r, H / 2 - r, W / 2 + (W - H) / 2 + r, H / 2 + r], radius=r, fill=(30, 30, 34, 255), outline=(80, 80, 90, 255), width=int(SS))
    label(d, W / 2, H / 2, text, r * 1.15)
    return im.resize((w, h), Image.LANCZOS)


# The second sheet (pac/game2dHD.pac, 512x512): the same cells in its left half, plus LS / RS with direction arrows in
# three columns x six rows on the right (rows 0-2 LS, 3-5 RS).
SHEETS = [  # (width, height, FNV-1a 64 of the original pixel bytes)
    (256, 512, 0xf8416f9573d0bc05),
    (512, 512, 0xe31732e85301a980),
    (128, 64, 0xda6a4f053e0b2ea7),  # the controller picture of the superstar select screen (white Xbox 360 pad)
    (128, 64, 0xb9562e7041145719),  # the same screen's dark controller outline (CPU slot, bottom)
]


def controller_outline(w=128, h=64):
    """The dark version: our controller with its colours inverted to the game's style (black body, light rim and parts)."""
    from PIL import ImageFilter
    big = controller(w * 4, h * 4)
    a = big.getchannel("A")
    rgb = np.array(big.convert("RGB"), int)
    lum = rgb.mean(axis=2)
    # white body -> near black; grey rim / touch pad and dark parts -> light grey
    out = np.where(lum[..., None] > 225, 20, np.where(lum[..., None] > 170, 150, 200)).repeat(3, axis=2)
    inner = np.array(a.filter(ImageFilter.MinFilter(9)), int)
    rim = np.clip(np.array(a, int) - inner, 0, 255)[..., None] / 255.0
    out = (out * (1 - rim) + 230 * rim).astype(np.uint8)
    im = Image.fromarray(out).convert("RGBA")
    im.putalpha(a)
    return im.resize((w, h), Image.LANCZOS)


def controller(w=128, h=64):
    """A white PlayStation-style controller (two sticks side by side below a touch pad; d-pad and the four symbols)."""
    S = 8
    W, H = w * S, h * S
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    body, edge, dark = (238, 238, 240, 255), (150, 150, 158, 255), (34, 34, 38, 255)
    # grips and body
    for gx, k in ((0.255, -1), (0.745, 1)):
        g = [(W * (gx - 0.07 * k), H * 0.30), (W * (gx + 0.06 * k), H * 0.30), (W * (gx + 0.085 * k), H * 0.86),
             (W * (gx - 0.035 * k), H * 0.86)]
        d.polygon(g, fill=edge)
        d.ellipse([W * (gx + 0.025 * k) - W * 0.062, H * 0.70, W * (gx + 0.025 * k) + W * 0.062, H * 0.97], fill=edge)
        d.ellipse([W * (gx + 0.025 * k) - W * 0.055, H * 0.70, W * (gx + 0.025 * k) + W * 0.055, H * 0.94], fill=body)
        g2 = [(W * (gx - 0.065 * k), H * 0.30), (W * (gx + 0.055 * k), H * 0.30), (W * (gx + 0.078 * k), H * 0.84),
              (W * (gx - 0.028 * k), H * 0.84)]
        d.polygon(g2, fill=body)
    d.rounded_rectangle([W * 0.18, H * 0.10, W * 0.82, H * 0.66], radius=H * 0.20, fill=body, outline=edge, width=S)
    # touch pad
    d.rounded_rectangle([W * 0.38, H * 0.14, W * 0.62, H * 0.42], radius=S * 3, fill=(205, 205, 212, 255), outline=edge, width=S)
    # sticks
    for sx in (0.40, 0.60):
        d.ellipse([W * sx - H * 0.12, H * 0.48, W * sx + H * 0.12, H * 0.72], fill=dark)
        d.ellipse([W * sx - H * 0.07, H * 0.53, W * sx + H * 0.07, H * 0.67], fill=(70, 70, 76, 255))
    # d-pad
    cx, cy, a, b = W * 0.29, H * 0.33, H * 0.10, H * 0.035
    d.rectangle([cx - a, cy - b, cx + a, cy + b], fill=dark)
    d.rectangle([cx - b, cy - a, cx + b, cy + a], fill=dark)
    # face buttons
    cx, cy, o, r = W * 0.71, H * 0.33, H * 0.10, H * 0.052
    for kind, (dx, dy) in (("triangle", (0, -1)), ("circle", (1, 0)), ("cross", (0, 1)), ("square", (-1, 0))):
        x, y = cx + dx * o, cy + dy * o
        d.ellipse([x - r, y - r, x + r, y + r], fill=dark)
        col, s, lw = SYMBOL[kind] + (255,), r * 0.5, max(2, int(r * 0.22))
        if kind == "cross":
            d.line([x - s, y - s, x + s, y + s], fill=col, width=lw); d.line([x - s, y + s, x + s, y - s], fill=col, width=lw)
        elif kind == "circle":
            d.ellipse([x - s, y - s, x + s, y + s], outline=col, width=lw)
        elif kind == "square":
            d.rectangle([x - s * 0.85, y - s * 0.85, x + s * 0.85, y + s * 0.85], outline=col, width=lw)
        else:
            d.polygon([(x, y - s), (x + s, y + s * 0.7), (x - s, y + s * 0.7)], outline=col, width=lw)
    return im.resize((w, h), Image.LANCZOS)


def patches():
    """(sheet, x, y, w, h, mode, rgba) - mode 0 replaces the rectangle, 1 draws over the game's pixels."""
    out = []
    for name, (x0, y0, x1, y1) in CELLS.items():
        art = np.array(draw(name, x1 - x0, y1 - y0))
        out.append((0, x0, y0, x1 - x0, y1 - y0, 0, art))
        if name != "R_LR":
            out.append((1, x0, y0, x1 - x0, y1 - y0, 0, art))
    for row, cy in enumerate((21, 64, 108, 152, 196, 240)):
        for cx in (280, 344, 408):
            x0, y0 = (cx - 14) // 4 * 4, (cy - 14) // 4 * 4
            w = 28 if row < 3 else 32  # the game's "RS" sits a little further right
            art = np.array(stick_center(w, 28, "L" if row < 3 else "R"))
            out.append((1, x0, y0, w, 28, 1, art))
    out.append((2, 0, 0, 128, 64, 0, np.array(controller())))
    out.append((3, 0, 0, 128, 64, 0, np.array(controller_outline())))
    return out


def main():
    out = sys.argv[1]
    pv = [Image.new("RGBA", (w, h), (0, 0, 0, 0)) for w, h, _ in SHEETS]
    lines = ["// Generated by tools/ps_prompts/make_ps_prompts.py (our own PlayStation icon artwork). Do not edit.",
             "#pragma once", "#include <cstdint>", "namespace wwe13_ps {",
             "struct Sheet { uint32_t w, h; uint64_t hash; };",
             "// x, y, w, h in pixels (multiples of 4); mode 0 = replace, 1 = draw over; rgba = w*h*4 bytes, straight alpha",
             "struct Patch { uint8_t sheet; uint16_t x, y, w, h; uint8_t mode; const uint8_t* rgba; };",
             "static const Sheet kSheets[] = {"]
    lines += [f"  {{{w}, {h}, 0x{hsh:016x}ull}}," for w, h, hsh in SHEETS]
    lines.append("};")
    rows = []
    for k, (sh, x, y, w, h, mode, art) in enumerate(patches()):
        pv[sh].paste(Image.fromarray(art), (x, y), Image.fromarray(art))
        lines.append(f"static const uint8_t k_p{k}[] = {{" + ",".join(str(b) for b in art.astype(np.uint8).tobytes()) + "};")
        rows.append(f"  {{{sh}, {x}, {y}, {w}, {h}, {mode}, k_p{k}}},")
    lines.append("static const Patch kPatches[] = {")
    lines += rows
    lines += ["};", "}  // namespace wwe13_ps", ""]
    open(out, "w").write("\n".join(lines))
    if len(sys.argv) > 2:
        for i, im in enumerate(pv):
            im.save(sys.argv[2].replace(".png", f"_{i}.png"))


if __name__ == "__main__":
    main()
