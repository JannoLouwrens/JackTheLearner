"""J24 beside-reading (decides nothing): WHY does the sighted cell unlearn eating?

Reads a home J24 raw file (`/data/jack-logs/j24/chains/j24_s94N.json`) and
prints, per cell and per 50k-decision window of lives:
  lives, decisions a life, floor meals a life, reward a life,
  energy drain per decision (basal_drain / decisions), and the policy's
  mean action std / saturation from the trace.

The question it separates: is the sighted cell's falling reward a learner
PROFITING FROM ENDING A LIFE (dying faster: drain per decision rises) or a
learner that STOPPED FINDING FOOD (drain per decision flat, intake falls)?
Not a verdict; the frozen read is `pilot_b2.j17_late_meals` on all nine cells.

    python -m experiments.j24_collapse_diag /data/jack-logs/j24/chains/j24_s943.json
"""
from __future__ import annotations

import collections
import json
import sys

WINDOW = 50_000


def windows(lives, window=WINDOW, n_windows=10):
    t = 0
    rows = collections.defaultdict(list)
    for life in lives:
        t += life["decisions"]
        rows[min((t - 1) // window, n_windows - 1)].append(life)
    return rows


def summarize(cell, window=WINDOW):
    out = []
    trace = {r["t"]: r for r in cell.get("trace", [])}
    for w, ls in sorted(windows(cell["lives"], window).items()):
        n = len(ls)
        dec = sum(l["decisions"] for l in ls)
        tr = [r for t, r in trace.items() if w * window < t <= (w + 1) * window]
        out.append({
            "from": w * window,
            "lives": n,
            "dec_per_life": dec / n,
            "meals_per_life": sum(l["floor_meals"] for l in ls) / n,
            "reward_per_life": sum(l["reward"] for l in ls) / n,
            "drain_per_kdec": 1000 * sum(l["basal_drain"] for l in ls) / dec,
            "intake_per_life": sum(l["intake"] for l in ls) / n,
            "std": (sum(sum(r["std"]) / len(r["std"]) for r in tr) / len(tr)) if tr else None,
            "sat": (sum(r["sat"] for r in tr) / len(tr)) if tr else None,
        })
    return out


def main(path):
    d = json.load(open(path))
    for name, cell in d["cells"].items():
        print(name)
        for r in summarize(cell):
            print("  {from:>6} lives {lives:2d}  dec/life {dec_per_life:5.0f}  meals {meals_per_life:4.2f}"
                  "  reward {reward_per_life:+7.1f}  drain/kdec {drain_per_kdec:.3f}"
                  "  intake {intake_per_life:.2f}  std {std:.3f}  sat {sat:.3f}".format(**r))


if __name__ == "__main__":
    main(sys.argv[1])
