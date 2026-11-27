"""Same against same through `harness.looks.compare_looks` at TEN runs an arm
(Auditor 11-27 finding 2: the wrapper that will read round 1-L's look 1 had
only planted tests; `same_vs_same` covers the frozen `compare` at twenty).

RULE (written before the run, 2026-11-27 02:09 UTC):
  Same generator, scenarios and seeds as `harness/same_vs_same.py` (rep r of
  scenario i: seed 10_000*i + r, 20 runs drawn per arm in the same order);
  the wrapper is given ONLY the first ten runs of each arm, n_perm 2000 and
  the same generator, exactly as round 1-L's reader calls it. Both arms come
  from one generator, so every A_DIFFERS is a false alarm.
  Reps: the first 150 of S1 and of S2 (300; the box's 15-CPU-minute line, not
  the 1,000 of check #1 — this is a SAMPLE and is reported as one).
  Read:
    (a) every rep's p1 equals, bit for bit, the p1 of the same rep in check
        #1's recorded 1,000-rep run (`svs2_S*.jsonl`), and the wrapper's
        verdict is that record's look-1 outcome (A_DIFFERS at look 1 ->
        A_DIFFERS; FUTILE -> FUTILE; anything at look 2 -> NEEDS_LOOK_2).
        One mismatch = FAIL: the wrapper is not the frozen look 1.
    (b) look-1 false alarms / reps, with its Clopper-Pearson 95% interval,
        beside the stated per-look rate 0.025. FAIL if the interval's LOWER
        end is above 0.025. A sample this size cannot prove the rate; with
        (a) holding on every rep, the 1,000-rep record's look-1 rate is the
        wrapper's, and that figure is printed beside it.
  Nothing here edits or re-judges the harness.

Run:  python -m experiments.forge.looks_fa run OUT.jsonl [reps]
      python -m experiments.forge.looks_fa tally OUT.jsonl
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from experiments.harness import LOOK_ALPHA, LOOKS
from experiments.harness.looks import NEEDS_LOOK_2, compare_looks
from experiments.harness.same_vs_same import N_PERM, SCEN, _cp, synth_run

RECORD = "/data/jack-logs/harness/svs2_{}.jsonl"
REPS = 150


def expected(rec: dict) -> str:
    """The look-1 outcome of a recorded 20-run `compare` verdict."""
    return rec["d"] if rec["look"] == 1 else NEEDS_LOOK_2


def record(scen: str, path: str = RECORD) -> dict:
    return {x["r"]: x for x in map(json.loads, open(path.format(scen))) if "r" in x}


def one(scen: str, r: int):
    s = SCEN[scen]
    rng = np.random.default_rng(10_000 * s["idx"] + r)
    a = [synth_run(rng, s) for _ in range(LOOKS[1])]
    b = [synth_run(rng, s) for _ in range(LOOKS[1])]
    return compare_looks(a[:LOOKS[0]], b[:LOOKS[0]], N_PERM, rng)


def run(out: str, reps: int = REPS):
    with open(out, "a") as f:
        for scen in SCEN:
            rec = record(scen)
            for r in range(reps):
                t0 = time.time()
                v = one(scen, r)
                f.write(json.dumps({"scen": scen, "r": r, "d": v.decision, "look": v.look,
                                    "p1": v.p1, "rec_p1": rec[r]["p1"],
                                    "rec_d": expected(rec[r]),
                                    "s": round(time.time() - t0, 2)}) + "\n")
                f.flush()


def tally(rows: list, records: dict | None = None) -> dict:
    """`rows`: the run's lines. `records`: {scen: {r: recorded row}} for the
    1,000-rep look-1 rate printed beside (None in tests)."""
    n = len(rows)
    mism = [(x["scen"], x["r"]) for x in rows
            if x["p1"] != x["rec_p1"] or x["d"] != x["rec_d"] or x["look"] != 1]
    fa = sum(x["d"] == "A_DIFFERS" for x in rows)
    lo, hi = _cp(fa, n) if n else (0.0, 1.0)
    out = {"n": n, "false_alarms": fa, "rate": fa / n if n else float("nan"),
           "cp95": (float(lo), float(hi)), "futile": sum(x["d"] == "FUTILE" for x in rows),
           "needs_look_2": sum(x["d"] == NEEDS_LOOK_2 for x in rows), "mismatch": mism}
    if records:
        full = [x for rec in records.values() for x in rec.values()]
        out["record_look1"] = (sum(x["d"] == "A_DIFFERS" and x["look"] == 1 for x in full),
                               len(full))
    out["verdict"] = ("FAIL (the wrapper is not the frozen look 1)" if mism or not n
                      else "FAIL (false-alarm rate above the stated 0.025)" if lo > LOOK_ALPHA
                      else "HOLDS (sample)")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else REPS)
    else:
        t = tally([json.loads(l) for l in open(sys.argv[2])], {s: record(s) for s in SCEN})
        print(json.dumps(t, indent=1))
