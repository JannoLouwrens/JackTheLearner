"""B-7 band, read off Ni et al. 2022 (arXiv 2110.05038) appendix Fig. separate_vs_shared
(figs/results/pomdp/pendulumv/diagnosis.png) by pixel: axis calibrated on the plot's own
gridlines (y: -500/-1000/-1500 at rows 160/325/490; x: 10k..40k at cols 466/672/877/1083)."""
import sys, json
from PIL import Image
import numpy as np
im = np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(int)
r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
y2v = lambda y: -500 - (y - 160) * 500 / 165
x2p = lambda v: int(round(466 + (v - 10000) * (1083 - 466) / 30000))
line = (b > 200) & (r < 90) & (g < 90)            # the solid blue mean line
band = (b - r > 12) & (b - g > 12) & ~line           # the light blue shade (seed spread)
red = (r > 200) & (g < 90) & (b < 90)
out = {}
for v in list(range(10000, 50001, 2000)) + [49000]:
    x = min(x2p(v), 1286)
    ys = np.where(line[30:560, x])[0] + 30
    bs = np.where((band | line)[30:560, x])[0] + 30
    rs = np.where(red[30:560, x])[0] + 30
    out[v] = dict(mean=round(y2v(ys.mean()), 1) if len(ys) else None,
                  band_hi=round(y2v(bs.min()), 1) if len(bs) else None,
                  band_lo=round(y2v(bs.max()), 1) if len(bs) else None,
                  shared_mean=round(y2v(rs.mean()), 1) if len(rs) else None)
    print(v, out[v])
json.dump(out, open(sys.argv[2], 'w'), indent=1)
