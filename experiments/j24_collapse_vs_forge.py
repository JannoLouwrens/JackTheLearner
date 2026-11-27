"""J24 beside-reading (decides nothing): is the sighted cell's unlearning the
EYE's fault or the LEARNER's? Run the same per-window meals table over forge
round 1 rung M (old paint, 300k, five seeds a arm), where the plain-SAC
baseline (E-food, J24's learner: no resets) sits beside K1 (resets every 50k),
K2 (TQC), N3 and control C2.

A seed "collapses" when its mean meals/life over the last three 30k windows is
under 20% of its best of the first five. Prints per arm: collapsed/seeds and
(seed, early peak, late mean).

    python -m experiments.j24_collapse_vs_forge [/data/jack-logs/forge/r1/raw]
"""
from __future__ import annotations

import glob
import json
import sys

from experiments.j24_collapse_diag import summarize

WINDOW = 30_000


def table(raw_dir="/data/jack-logs/forge/r1/raw", rung="M"):
    agg = {}
    for f in sorted(glob.glob(f"{raw_dir}/{rung}_g*.json")):
        for name, c in json.load(open(f))["cells"].items():
            if "lives" not in c:
                continue
            m = [r["meals_per_life"] for r in summarize(c, WINDOW)]
            peak, late = max(m[:5]), sum(m[-3:]) / 3
            agg.setdefault(c["arm"], []).append((name.split("_")[1], round(peak, 1), round(late, 2), late < 0.2 * peak))
    return agg


def main(argv):
    for arm, rows in table(*argv).items():
        print(f"{arm:12s} collapsed {sum(r[3] for r in rows)}/{len(rows)}  "
              + "  ".join(f"{s} {p}->{l}" for s, p, l, _ in rows))


if __name__ == "__main__":
    main(sys.argv[1:])
