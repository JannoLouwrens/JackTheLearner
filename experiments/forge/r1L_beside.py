"""Round 1-L's BESIDE readings and its KEEP RULE, as `rounds/r1L/spec.json`
froze them (52033df2) — written 2026-11-27 02:10 UTC, before any r1L raw file
was home (`/data/jack-logs/forge/r1L/raw` did not exist yet).

`r1L_read` (ORDERS S20) gives the harness verdict for K1 against the baseline,
the books and rung M's late-half rate per run. The frozen rule names two more
things that had no code:

  * the LONG-LIFE reading (`outcome.beside.readings[1]`): per run, the
    sustained rate over lives ENDING in (800k, 1M] against lives ending in
    (300k, 500k]. Sustained = the harness's own mark (`sustained_life`,
    ruling 22), as in `metric.py`. It decides nothing by test.
  * the KEEP RULE (`keep_rule`): only if the primary says K1 differs (look 1
    or 2), brain B's keep setting is whichever of K1/K2 has the higher
    late-half mean over seeds 1421-1430; closer than K1's own seed range
    (max - min of K1's ten rates) -> K1.

Everything is read THROUGH `r1L_read.read`, so each of its refusals (moved
spec, moved metric, a missing run, another pin/world/settings) applies here
first. This file adds no test and moves no line. Writes nothing: it prints.

Readings of the frozen text, each the conservative one (stated here so they
are on file before a number exists):
  - a run that is not whole (crash / cut / missing) reads 0.0 in both windows
    and in the mean, and is named (`outcome.crash`: it counts against its arm);
  - a window in which no life ends reads 0.0, with its life count 0 printed;
  - "closer than" is strict and K2 must also be HIGHER: K2 is kept only when
    mean(K2) - mean(K1) >= K1's range and > 0. Every other case, an exact tie
    included, is K1 (the record's);
  - NEEDS_LOOK_2 -> no keep setting YET; FUTILE / NOT_DISTINGUISHED -> none
    chosen by this round;
  - the keep rule reads tranche 1's seeds only, also at look 2.

    python -m experiments.forge.r1L_beside --tranches 1
"""
from __future__ import annotations

import hashlib
import json
import math
import sys

from experiments.forge import Refused, check_lock
from experiments.forge import r1L_read as R
from experiments.harness import sustained_life
from experiments.harness.looks import NEEDS_LOOK_2

MID, LAST = (300_000, 500_000), (800_000, 1_000_000)     # lives ENDING in (lo, hi]


def window_rate(lives: list, lo: float, hi: float) -> tuple:
    """(sustained rate, lives) over lives ending in (lo, hi]. `lives` as the
    kernel wrote them (dicts with decisions, cause, intake, basal_drain)."""
    cum, n, good = 0.0, 0, 0
    for l in lives:
        cum += float(l["decisions"])
        if cum > hi:
            break
        if cum > lo:
            n += 1
            good += bool(sustained_life(l["cause"] == "capped", l["intake"], l["basal_drain"]))
    return (good / n if n else 0.0), n


def long_life(files: dict, plan: dict, not_ok: dict) -> dict:
    """{arm: [{seed, mid, mid_lives, last, last_lives, status}, ...]} in the plan's order."""
    recs = {(rec["arm"], rec["seed"]): rec for p in files.values()
            for rec in (p.get("cells") or {}).values()}
    out = {}
    for arm, seeds in plan.items():
        rows = []
        for s in seeds:
            status = not_ok.get(f"{arm}_s{s}", "ok")
            if status == "ok":
                (m, mn), (l, ln) = (window_rate(recs[(arm, s)]["lives"], *w) for w in (MID, LAST))
            else:
                m = l = 0.0
                mn = ln = 0
            rows.append({"seed": s, "mid": m, "mid_lives": mn, "last": l, "last_lives": ln,
                         "status": status})
        out[arm] = rows
    return out


def keep(decision: str, k1: list, k2: list) -> dict:
    """The frozen keep rule on the ten late-half rates of each arm."""
    if len(k1) != len(k2) or not k1:
        raise Refused("keep rule: K1 and K2 must carry the same tranche-1 seeds")
    m1, m2 = sum(k1) / len(k1), sum(k2) / len(k2)
    rng = max(k1) - min(k1)
    out = {"mean_K1": m1, "mean_K2": m2, "K1_range": rng, "decision": decision}
    if decision == "A_DIFFERS":
        k2_wins = m2 > m1 and (m2 - m1) >= rng
        out["keep"] = "K2" if k2_wins else "K1"
        out["why"] = ("K2's mean is higher by at least K1's own seed range" if k2_wins
                      else "K1's mean is the higher (or equal)" if m1 >= m2
                      else "K2 is higher but closer than K1's own seed range: K1, the record's")
    elif decision == NEEDS_LOOK_2:
        out["keep"], out["why"] = None, "not yet: the primary needs look 2 (launch tranche 2)"
    else:
        out["keep"], out["why"] = None, f"none chosen by this round: the primary reads {decision}"
    return out


def read(spec: dict, files: dict, tranches, **kw) -> dict:
    """`r1L_read.read` (all its refusals) + the long-life reading + the keep rule."""
    out = R.read(spec, files, tranches, **kw)
    plan = R.manifest(spec, tranches)
    t1 = R.manifest(spec, [1])
    prim = spec["outcome"]["primary"]
    cands = [c["id"] for c in spec["candidates"]]
    if cands != [prim["a"], "K2"] or t1[cands[0]] != t1[cands[1]]:
        raise Refused("keep rule: the frozen rule compares K1 and K2 on tranche 1's seeds")
    n = len(t1[cands[0]])
    out["long_life"] = long_life(files, plan, out["status"])
    out["keep"] = keep(out["verdict"].decision, out["rate_M"]["K1"][:n], out["rate_M"]["K2"][:n])
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else math.nan


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranches", default="1")
    tr = [int(x) for x in ap.parse_args(argv).tranches.split(",")]
    sp = R.ROUND / "spec.json"
    try:
        check_lock(R.ROUND)   # ruling 130: no raw file is opened before the lock
        out = read(json.loads(sp.read_text()),
                   {f.name: json.loads(f.read_text()) for f in sorted(R.RAW.glob("L_g*.json"))},
                   tr, spec_sha=hashlib.sha256(sp.read_bytes()).hexdigest(),
                   freeze_sha=R.frozen_sha())
    except Refused as e:
        print(f"REFUSED: {e}")
        return 1
    v = out["verdict"]
    print(f"PRIMARY K1 vs E-food: {v.decision} at look {v.look}  p1 {v.p1:.4f}  p2 {v.p2:.4f}"
          f"  (n_perm {R.N_PERM}, seed {R.SEED}, pin {out['pin'][:10]})")
    print(f"not whole (read 0, counted): {out['status'] or 'none'}")
    print("LONG LIFE, deciding nothing by test — sustained rate, lives ending in "
          "(300k, 500k] -> (800k, 1M]:")
    for arm, rows in out["long_life"].items():
        up = sum(r["last"] > r["mid"] for r in rows)
        down = sum(r["last"] < r["mid"] for r in rows)
        print(f"  {arm:<11} mean {_mean([r['mid'] for r in rows]):.3f} -> "
              f"{_mean([r['last'] for r in rows]):.3f}  (runs up {up}, down {down}, "
              f"same {len(rows) - up - down})  "
              + " ".join(f"{r['mid']:.2f}>{r['last']:.2f}" for r in rows))
    k = out["keep"]
    print(f"KEEP RULE: late-half mean K1 {k['mean_K1']:.3f}, K2 {k['mean_K2']:.3f}, "
          f"K1's seed range {k['K1_range']:.3f} -> keep {k['keep']} ({k['why']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
