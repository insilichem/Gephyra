"""
Renders the Gephyra application icon into src/gephyra/assets/.

    python3 tools/make_icons.py

Developer tool only (needs Pillow); the generated PNGs are committed and
loaded at runtime by tkinter, so Pillow is not a Gephyra dependency.

The motif matches the manual cover and the interface: a chain of cyan
nodes (water oxygens) joined by dashed hydrogen bonds, on the dark
background of the interface. Small sizes use solid bonds and fewer nodes,
because dashes and detail vanish at 16-32 px.
"""

import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "src", "gephyra", "assets")
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)

BG = (13, 17, 23, 255)          # interface dark background
BG_EDGE = (34, 211, 238, 70)    # faint cyan rim
CYAN = (34, 211, 238, 255)

# Chain of nodes in unit coordinates (y down), a bent wire like the cover motif.
NODES = [(0.20, 0.76), (0.36, 0.54), (0.58, 0.64), (0.80, 0.44), (0.62, 0.22), (0.38, 0.27)]
SMALL_NODES = [(0.22, 0.70), (0.45, 0.48), (0.72, 0.62), (0.70, 0.28)]


def _dashed(draw, p1, p2, width, dash, gap, fill):
    (x1, y1), (x2, y2) = p1, p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    t = 0.0
    while t < length:
        e = min(t + dash, length)
        draw.line([(x1 + ux * t, y1 + uy * t), (x1 + ux * e, y1 + uy * e)], fill=fill, width=width)
        t = e + gap


def render(size):
    scale = 8                                   # supersample, then downscale for smooth edges
    S = size * scale
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = S * 0.04
    radius = S * 0.22
    d.rounded_rectangle([pad, pad, S - pad, S - pad], radius=radius, fill=BG,
                        outline=BG_EDGE, width=max(1, int(S * 0.012)))

    small = size <= 32
    nodes = SMALL_NODES if small else NODES
    pts = [(x * S, y * S) for x, y in nodes]
    bond = int(S * (0.075 if small else 0.024))
    node_r = S * (0.10 if small else 0.052)
    for p1, p2 in zip(pts[:-1], pts[1:]):
        if small:
            d.line([p1, p2], fill=CYAN, width=bond)
        else:
            _dashed(d, p1, p2, bond, dash=S * 0.034, gap=S * 0.026, fill=CYAN)
    for x, y in pts:
        d.ellipse([x - node_r, y - node_r, x + node_r, y + node_r], fill=CYAN)
    return im.resize((size, size), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    for size in SIZES:
        path = os.path.join(OUT, f"gephyra-{size}.png")
        render(size).save(path, optimize=True)
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
