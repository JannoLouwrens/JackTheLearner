"""J4 check #1 — same against same, through the real pipeline (PUBLISHED_PRIORS §8).

Why: the released AdaStop package rejected 10-12% of same-against-same
comparisons at a stated 5%. A comparison tool that has not passed this is
not believed here.

RULE (committed before any run, 2026-11-16):
  Pipeline: synthetic per-life logs -> `outcome` (ruling 19) -> `compare`
  (two looks at 10 and 20 runs/arm, 0.025 each, futility at look 1,
  permutation log-rank with n_perm = 2000, the default the real comparison
  uses). Both arms are drawn from the SAME generator, so every rejection is
  a false alarm.
  Scenarios (both must pass):
    S1 "mixed"  — heterogeneous learners (learning scale k ~ lognormal),
                  life lengths 300-2000 decisions in steps of 100, meals
                  Poisson(8 (1 - exp(-cum/k))), 1% crash per life, W=3,
                  T=150,000.
    S2 "ties"   — every life exactly 1,000 decisions (crossing times tie
                  on a 1,000 grid), same learners slower, W=3, T=80,000:
                  heavy censoring and heavy ties.
  Repetitions: 1,000 per scenario, rep r uses seed 10_000*scen + r.
  PASS: false-alarm rate (A_DIFFERS / reps) <= 0.050 (the stated rate:
  two looks x 0.025) in EACH scenario. Reported beside it: per-look rates,
  futility rate, Clopper-Pearson 95% interval, censoring fraction.
  Control that MUST FAIL: the same p-values judged with no correction for
  the two looks (0.05 at each look) — its false-alarm rate must exceed
  0.050 in at least one scenario, or this check cannot tell a correct
  stopping rule from a broken one, and the check is VOID.
  Thresholds are never moved after a run; a FAIL is fixed in the harness.

Run:  python -m experiments.harness.same_vs_same run S1 OUT.jsonl [reps]
      python -m experiments.harness.same_vs_same tally OUT_S1.jsonl OUT_S2.jsonl
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
from scipy import stats

from experiments.harness import Outcome, analysis_hash, compare, outcome

SCEN = {
    "S1": dict(idx=1, W=3, T=150_000, crash=0.01, life=(3, 20), k_mu=math.log(60_000)),
    "S2": dict(idx=2, W=3, T=80_000, crash=0.0, life=(10, 10), k_mu=math.log(250_000)),
}
REPS, N_PERM, RATE = 1000, 2000, 0.050


def synth_run(rng, s) -> Outcome:
    k = math.exp(rng.normal(s["k_mu"], 0.7))
    lives, cum = [], 0
    while cum <= s["T"]:
        if rng.random() < s["crash"]:
            lives.append("crash")
            break
        d = 100 * int(rng.integers(s["life"][0], s["life"][1] + 1))
        cum += d
        lives.append((d, int(rng.poisson(8 * (1 - math.exp(-cum / k))))))
    return outcome(lives, s["W"], s["T"])


def run(scen: str, out: str, reps: int = REPS):
    s = SCEN[scen]
    with open(out, "a") as f:
        f.write(json.dumps({"header": scen, "hash": analysis_hash(), "n_perm": N_PERM,
                            "reps": reps}) + "\n")
        for r in range(reps):
            rng = np.random.default_rng(10_000 * s["idx"] + r)
            t0 = time.time()
            a = [synth_run(rng, s) for _ in range(20)]
            b = [synth_run(rng, s) for _ in range(20)]
            v = compare(a, b, n_perm=N_PERM, rng=rng)
            f.write(json.dumps({"r": r, "d": v.decision, "look": v.look, "p1": v.p1,
                                "p2": None if math.isnan(v.p2) else v.p2,
                                "ev": sum(o.event for o in a + b),
                                "fut1": sum(o.event for o in a[:10]) <= 1
                                and sum(o.event for o in b[:10]) <= 1,
                                "s": round(time.time() - t0, 2)}) + "\n")
            f.flush()


def _cp(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def tally(paths):
    ok, ctrl_fail, done, rows = True, False, True, []
    for p in paths:
        recs = [json.loads(l) for l in open(p)]
        head = [x for x in recs if "header" in x][-1]
        rs = [x for x in recs if "r" in x]
        n = len(rs)
        fa = sum(x["d"] == "A_DIFFERS" for x in rs)
        l1 = sum(x["d"] == "A_DIFFERS" and x["look"] == 1 for x in rs)
        fut = sum(x["d"] == "FUTILE" for x in rs)
        # control: 0.05 per look, same p-values, same futility rule
        cfa = sum(x["p1"] <= 0.05 or (x["p2"] is not None and x["p2"] <= 0.05) for x in rs)
        cens = 1 - sum(x["ev"] for x in rs) / (40 * n)
        lo, hi = _cp(fa, n)
        passed = fa / n <= RATE
        done &= n >= REPS
        ok &= passed
        ctrl_fail |= cfa / n > RATE
        rows.append(f"{head['header']}: n={n} false-alarm {fa}/{n}={fa/n:.3f} "
                    f"(CP95 {lo:.3f}-{hi:.3f}; look1 {l1/n:.3f}, look2 {(fa-l1)/n:.3f}) "
                    f"futile {fut/n:.3f} censored {cens:.2f} | control(0.05/look) "
                    f"{cfa/n:.3f} | {('PASS' if passed else 'FAIL') if n >= REPS else 'INCOMPLETE'} hash {head['hash'][:12]}")
    print("\n".join(rows))
    verdict = "INCOMPLETE" if not done else "VOID (control did not fail)" if not ctrl_fail else ("PASS" if ok else "FAIL")
    print("CHECK #1:", verdict)
    return verdict


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else REPS)
    else:
        tally(sys.argv[2:])
