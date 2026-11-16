"""Row 4d — J3 row 4 re-registered as a same-against-same reproduction.

THE RULE — row 4d, committed before any run (11-16 ~12:30 UTC).
Why. Rows 4 / 4b MISSED the band built from the public store's 10 finals
(4 of our 20 seeds solve, then collapse). Row 4c ran the store's own stack on
this box: 82.3 +/- 11.6 on 20 seeds, one collapse, two partials — it misses
that band too, so the band (store sd 4.2) understates the recipe's spread.
4c was VOID because its per-seed control clause breached (lr-0 seed 0 = 98.0:
a frozen init solves MountainCarContinuous). Row 4c's reading A says: re-register
row 4 as ours vs the published code run HERE, with a rule written before it
runs. This is that rule. Rows 4 / 4b stay MISSES; 4c stays VOID.

Question. Does OUR stack's Zoo PPO reproduce the PUBLISHED code's Zoo PPO on
MountainCarContinuous-v0, run on the same machine, fresh seeds?
Arms (same file `row4_stack.run_cell`, same settings, metric, 20,000 steps;
the interpreter is the only factor):
  r4dours      /data/venvs/jack-std: Python 3.12.13, SB3 2.9.2, gymnasium 1.4.0,
               torch 2.15.0+cpu, numpy 2.5.5. (Rows 4/4b ran the same SB3 and
               gymnasium on Kaggle's py3.13/torch 2.11 — a host difference,
               removed here by running both arms on the box.)
  r4dpub       /data/venvs/sb3-17: Python 3.9.25, SB3 1.7.0, gym 0.21.0,
               torch 1.12.0, numpy 1.23.1 (4c's stack).
  r4doursctrl, r4dpubctrl   the arms at learning_rate 0.0 (CONTROLS).
Seeds. Learners 20..39 (fresh: rows 4/4b/4c used 0..19); controls 20..29.
Metric. Per seed, mean of ep_info_buffer (last <= 100 episodes) at the end of
learn() — row 4's metric exactly. Collapse = metric < 60.
Primary (n = 20 per arm):
  HL = Hodges-Lehmann shift (median of all ours_i - pub_j); its 90% CI by the
  Moses / Mann-Whitney normal approximation (z = 1.645).
  Margin +/- 5.0 return points (the half-width of row 4b's band, 5.1, and the
  store's own sd, 4.2 — a difference smaller than the published spread).
  REPRODUCES  iff the HL 90% CI lies inside [-5, +5] AND Fisher two-sided on
              collapses (ours k/20 vs pub k/20) has p >= 0.05.
  DIFFERS     iff Mann-Whitney two-sided p < 0.05 OR that Fisher p < 0.05.
  INCONCLUSIVE otherwise -> ONE pre-registered extension: learner seeds
              40..59 on both arms, pooled n = 40 per arm, the same three
              readings; INCONCLUSIVE at n = 40 reads "not shown equivalent at
              40 seeds" = row 4d MISSES. No further extension.
Sizing, stated honestly. On the EXISTING data (rows 4+4b ours on Kaggle vs
4c pub on the box, 20 vs 20) the same statistic gives HL -1.3, 90% CI
[-6.6, +2.7] — width ~9, i.e. at n = 20 the CI fits in +/-5 only if HL is
within ~0.5 of zero; the n = 40 extension (width ~6.5) is what the margin is
sized for. That existing pair is not this test (different seeds and host for
ours); it is quoted only to show the margin was not set to be easy.
Control (stated on its distribution, the 4c lesson). Base rate of a solving
frozen init on record: 1 of 15 (ours 0/10, pub 1/5). VOID unless all 20
control cells complete AND at most 3 of the 20 reach >= 60 (P(>3 | 1/15) =
0.04) AND the learners' pooled >= 60 count (of 40) exceeds the controls'
(of 20) by one-sided Fisher p < 0.001 (the metric must measure learning).
Box, nice 19, 2 workers, one torch thread per cell, ~112 s wall a cell.
Analysis runs under jack-std (scipy).

Usage: jack-std/bin/python -m experiments.row4d run OUTDIR [--extend]
       jack-std/bin/python -m experiments.row4d verdict OUTDIR
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

PY = {"r4dours": "/data/venvs/jack-std/bin/python",
      "r4doursctrl": "/data/venvs/jack-std/bin/python",
      "r4dpub": "/data/venvs/sb3-17/bin/python",
      "r4dpubctrl": "/data/venvs/sb3-17/bin/python"}
LEARN_SEEDS = tuple(range(20, 40))
EXT_SEEDS = tuple(range(40, 60))
CTRL_SEEDS = tuple(range(20, 30))
MARGIN = 5.0
COLLAPSE = 60.0
Z90 = 1.645


def cells(extend: bool = False) -> list:
    learn = LEARN_SEEDS + (EXT_SEEDS if extend else ())
    out = []
    for s in learn:                       # interleave so a partial run is balanced
        out += [("r4dours", s), ("r4dpub", s)]
    for s in CTRL_SEEDS:
        out += [("r4doursctrl", s), ("r4dpubctrl", s)]
    return out


def hl_ci(a: list, b: list, z: float = Z90):
    d = sorted(x - y for x in a for y in b)
    n1, n2, N = len(a), len(b), len(a) * len(b)
    k = math.floor(n1 * n2 / 2 - z * math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12))
    k = max(0, k)
    med = (d[(N - 1) // 2] + d[N // 2]) / 2
    return med, d[k], d[N - 1 - k]


def verdict(recs: list, extend: bool = False) -> dict:
    from scipy.stats import fisher_exact, mannwhitneyu
    by = {}
    for r in recs:
        by.setdefault(r["arm"], {})[r["seed"]] = r["mean_return"]
    learn = LEARN_SEEDS + (EXT_SEEDS if extend else ())
    out = {"n_per_arm": len(learn), "extended": extend}
    missing = [(a, s) for a, s in cells(extend) if s not in by.get(a, {})]
    if missing:
        out["reading"] = "INCOMPLETE"
        out["missing"] = missing
        return out
    ours = [by["r4dours"][s] for s in learn]
    pub = [by["r4dpub"][s] for s in learn]
    ctrl = [by[a][s] for a in ("r4doursctrl", "r4dpubctrl") for s in CTRL_SEEDS]
    ko, kp = sum(v < COLLAPSE for v in ours), sum(v < COLLAPSE for v in pub)
    hl, lo, hi = hl_ci(ours, pub)
    mw_p = float(mannwhitneyu(ours, pub, alternative="two-sided").pvalue)
    f_p = float(fisher_exact([[ko, len(ours) - ko], [kp, len(pub) - kp]])[1])
    c_hit = sum(v >= COLLAPSE for v in ctrl)
    l_hit = sum(v >= COLLAPSE for v in ours + pub)
    c_p = float(fisher_exact([[l_hit, len(ours + pub) - l_hit],
                              [c_hit, len(ctrl) - c_hit]], alternative="greater")[1])
    m = lambda v: round(sum(v) / len(v), 2)
    sd = lambda v: round(math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)), 2)
    out.update({
        "ours": {"per_seed": [round(v, 1) for v in ours], "mean": m(ours), "sd": sd(ours), "collapsed": ko},
        "pub": {"per_seed": [round(v, 1) for v in pub], "mean": m(pub), "sd": sd(pub), "collapsed": kp},
        "ctrl": {"per_seed": [round(v, 1) for v in ctrl], "ge60": c_hit},
        "hl": round(hl, 2), "hl_ci90": [round(lo, 2), round(hi, 2)],
        "mw_two_sided_p": round(mw_p, 4), "fisher_collapse_p": round(f_p, 4),
        "ctrl_vs_learners_fisher_p": c_p})
    if c_hit > 3 or c_p >= 0.001:
        out["reading"] = "VOID"
    elif -MARGIN <= lo and hi <= MARGIN and f_p >= 0.05:
        out["reading"] = "REPRODUCES"
    elif mw_p < 0.05 or f_p < 0.05:
        out["reading"] = "DIFFERS"
    else:
        out["reading"] = "MISSES (not shown equivalent at 40 seeds)" if extend else "INCONCLUSIVE"
    return out


def load(outdir: Path) -> list:
    return [json.loads(f.read_text()) for f in sorted(outdir.glob("r4d*.json"))]


def run_all(outdir: str, extend: bool = False, workers: int = 2) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells(extend) if not (out / f"{c[0]}_{c[1]}.json").exists()]
    running = []
    while todo or running:
        while todo and len(running) < workers:
            arm, seed = todo.pop(0)
            code = ("import json,sys; from pathlib import Path; "
                    "from experiments.row4_stack import run_cell; "
                    f"r = run_cell({arm!r}, {seed}, row='4d'); "
                    f"Path({str(out)!r}, '{arm}_{seed}.json').write_text(json.dumps(r)); "
                    "print(r['arm'], r['seed'], 'mean', round(r['mean_return'], 1), 'wall', r['wall_s'], flush=True)")
            p = subprocess.Popen(["nice", "-n", "19", PY[arm], "-c", code])
            running.append((p, arm, seed))
            print(f"START {arm} {seed} pid {p.pid}", flush=True)
        time.sleep(5)
        for item in list(running):
            p, arm, seed = item
            if p.poll() is not None:
                running.remove(item)
                print(f"END {arm} {seed} rc {p.returncode}", flush=True)
    v = verdict(load(out), extend)
    (out / ("verdict_ext.json" if extend else "verdict.json")).write_text(json.dumps(v, indent=1))
    print(json.dumps(v), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if len(a) >= 3 and a[1] == "run":
        run_all(a[2], extend="--extend" in a)
    elif len(a) >= 3 and a[1] == "verdict":
        print(json.dumps(verdict(load(Path(a[2])), "--extend" in a), indent=1))
    else:
        print(__doc__)
        sys.exit(2)
