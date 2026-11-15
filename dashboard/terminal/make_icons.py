#!/data/venvs/jackthelearner/bin/python
"""make_icons.py — the terminal app's icon: a blue prompt chevron and a white
cursor on a near-black square. Drawn with shapes, so it needs no font. The
square is full-bleed (phones round the corners themselves) and the glyph sits
inside the middle 60%, the safe zone for maskable icons."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "www")
S = 1024                                            # draw large, shrink for smooth edges
im = Image.new("RGB", (S, S))
px = im.load()
for y in range(S):                                  # a soft top-to-bottom shade
    t = y / (S - 1)
    c = (round(34 - 22 * t), round(34 - 22 * t), round(38 - 24 * t))
    for x in range(S):
        px[x, y] = c
d = ImageDraw.Draw(im)
w = 74                                              # stroke width
pts = [(300, 330), (480, 512), (300, 694)]          # the chevron
d.line(pts, fill=(10, 132, 255), width=w, joint="curve")
for (x, y) in pts:                                  # round the ends and the elbow
    d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=(10, 132, 255))
d.rounded_rectangle((540, 640, 740, 640 + w), radius=w / 2, fill=(242, 242, 247))   # the cursor
for n in (180, 192, 512):
    im.resize((n, n), Image.LANCZOS).save(out / f"icon-{n}.png", optimize=True)
print("icons written to", out)
