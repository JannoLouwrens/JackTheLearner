"""SH.02 option-(a) DIAGNOSTIC — is the learner above a NON-DEGENERATE null?

NOT A SPEC. NOT A PILOT. NOTHING HERE TOUCHES `experiments/ledger.json`, and
`SH.02`'s registered `HEADROOM` VOID, its provisional gates and its `run()`
refusal all stand exactly as they were. This module exists because the Review
FULL of 2026-09-20 authorised precisely one builder act on `SH.02` before the
world-edit window opens, and named its limits in the same breath (queue row
`sh02-null-saturation`, THE BUNDLED RULING):

    "The one thing the builder MAY do meanwhile, and its limits. `SH.02` may be
    run under (a) as a DIAGNOSTIC ONLY — scored against the random walk,
    reported in the run record, and NEVER as the registered gate. It costs CPU
    this project has spare, it measures whether the learner clears a
    non-degenerate comparator at all, and it tells us before the expensive
    window opens whether (b) is worth the certificates it will bill. A
    diagnostic that is labelled a diagnostic is not a re-pointed null."

Option (b) — the matched outward impulse at spawn — is the ADOPTED repair and
it is a VENUE change: it bills the 21 `playground.py` certificates and is bound
to `w1-world-edit-window`, the most expensive and still-undesigned instrument
this project owns. This file answers the one question the ruling said should be
answered for free first.

WHY THE PILOT'S SCALARS CANNOT ANSWER IT. `/data/sh02_pilot_seed90.json` records
LEVELS (`frac_shelt_learn` 0.0136, `frac_shelt_rand` 0.3639) but no per-life
arrays, and option (a)'s statistic is a Welch z over per-life sheltered
fractions. A difference of levels is not a z: the gate `Z_MIN` 3.0 is about
variance as well as means, and the pilot's own `z_shelter` -377.72 is the
project's standing example of a z that is arithmetic rather than effect size
(2 twin eval lives, zero variance, the 1e-9 denominator floor). So the contrast
has to be re-run to be read honestly.

## PRE-REGISTRATION — written and committed BEFORE the run, per law 1

ENVELOPE: exactly the pilot's. Seed 90, `n_decisions` 3000 per arm, the same
`_run_arm` / `_eval_lives` / `_fracs` / `_welch_z` functions the registered gate
uses, imported from the spec module rather than reimplemented (the
`_split_foreclosed` rule: two readers of one quantity share code or they drift).

ARMS: `learner`, `random`, `twin`. The twin is here ONLY as the KNOWN-ANSWER
CONTROL and its contrast is not the question. All six of `SH.02`'s declared
`IMPL_DEPS` — `thermal.py`, `w0.py`, `playground.py`, `cores.py`,
`survival.py`, `drives.py` — are byte-unchanged since the pilot commit
`8abfa70` (verified with `git log 8abfa70..HEAD -- <path>`: zero commits on
each), and the two commits that did touch the spec file are docstring-only. The
arms are deterministically seeded. **This run must therefore REPRODUCE the
pilot's recorded scalars, and if it does not, this file reports UNREPRODUCED
and no number in it is evidence** — the rule LESSONS.md states one level up for
at-chance controls: an instrument that cannot hit a known answer does not get to
report a new one.

    KNOWN ANSWERS (from the pilot record, seed 90, N=3000/arm, commit 8abfa70)
      frac_shelt_learn  0.0136      frac_shelt_twin  1.0000
      frac_shelt_rand   0.3639      rand_frozen      25 (of 26 lives)
      z_shelter        -377.72      lives: learner 32, twin 15, random 26

THE STATISTIC — option (a) as the ruling describes it, and nothing else:

    z_rand = welch(per-life sheltered fraction | learner eval lives,
                   per-life sheltered fraction | random  eval lives)

scored on the SAME gate-eligible set (`_eval_lives`: late and complete) the
registered gate uses. `Z_MIN` is the registry's 3.0 and DOES NOT MOVE here or
anywhere; it is quoted, never rewritten.

THREE PRE-REGISTERED READINGS, declared before any number exists:

  CLEARS         z_rand >= +3.0. Option (a) has a live claim after all, and the
                 only thing between it and a gate is the desk's refusal — which
                 was on PRINCIPLE (it leaves three degenerate arms at the
                 ceiling and looks away from them), not on arithmetic. That
                 would be worth saying out loud, because it changes the price of
                 waiting for the window.
  DEAD           z_rand <= -3.0. The learner is BELOW a non-degenerate
                 comparator by the same bar the claim would have to clear. Then
                 (a) is arithmetically dead as well as refused, and — the part
                 that bears on (b) — the question the window would be asked to
                 fix is not only "the null has no headroom" but "the learner is
                 beaten by a random walk in this venue". A matched outward
                 impulse removes the twin's free 1.0000; it does not obviously
                 move a learner that is already outside.
  INDETERMINATE  |z_rand| < 3.0. Neither arm is above the other at this bar and
                 the diagnostic has no news; say so rather than reading the
                 sign.

FORECAST, recorded so the run can contradict it: **DEAD.** The pilot's levels
are 0.0136 against 0.3639, a factor of 26.8, and `SH.02`'s own docstring already
carries the hypothesis that the learner "is the one arm that leaves". A forecast
is not a finding and this one is written down for exactly one reason — so that
if the run returns CLEARS or INDETERMINATE, nobody can claim afterwards that it
came out as expected.

WHAT THIS FILE MAY NOT BE USED FOR, restated because the ruling restates it:
`z_rand` may not become a conjunct, may not replace `z_shelter`, and may not
appear in `_check`. Re-pointing the registered null is the redesign the desk
refused; this is a measurement taken before an expensive decision.

Run it:  /data/venvs/jackthelearner/bin/python -m experiments.sh02_random_comparator_diag
"""
from __future__ import annotations

import json
import time

import numpy as np

from .tests.sh_02_born_sheltered import (Z_MIN, _eval_lives, _fracs,
                                         _run_arm, _welch_z)

ARTIFACT = "/data/sh02_random_comparator_diag_seed90.json"

SEED = 90
N_DEC = 3000          # the pilot's envelope, not the spec's N_DECISIONS
BAR = Z_MIN           # 3.0, quoted from the spec. It does not move.

# The pilot's recorded scalars and the tolerance each is checked at. The
# tolerance is the precision the pilot RECORDED at, not a band chosen to pass:
# the artifact stores 4 decimal places, so 5e-5 is exact agreement there, and
# integer counts are checked exactly.
KNOWN = {"frac_shelt_learn": (0.0136, 5e-5),
         "frac_shelt_twin": (1.0, 5e-5),
         "frac_shelt_rand": (0.3639, 5e-5),
         "z_shelter": (-377.7245, 5e-3)}
KNOWN_INT = {"rand_frozen": 25, "n_lives_learn": 32, "n_lives_twin": 15,
             "n_lives_rand": 26}


def main() -> dict:
    t0 = time.perf_counter()
    out: dict = {"seed": SEED, "n_decisions": N_DEC, "bar": BAR}

    arms = {}
    for mode in ("learner", "random", "twin"):
        t = time.perf_counter()
        arms[mode] = _run_arm(SEED, mode, n_decisions=N_DEC)
        print(f"  {mode:8} {time.perf_counter() - t:7.1f}s "
              f"{len(arms[mode]['lives']):3} lives", flush=True)

    ev = {m: _eval_lives(r, n_decisions=N_DEC) for m, r in arms.items()}
    fr = {m: _fracs(v) for m, v in ev.items()}

    out.update({
        # the known-answer half
        "frac_shelt_learn": float(fr["learner"].mean()) if len(fr["learner"]) else 0.0,
        "frac_shelt_twin": float(fr["twin"].mean()) if len(fr["twin"]) else 0.0,
        "frac_shelt_rand": float(fr["random"].mean()) if len(fr["random"]) else 0.0,
        "z_shelter": _welch_z(fr["learner"], fr["twin"]),
        "rand_frozen": int(sum(L["frozen"] for L in arms["random"]["lives"])),
        "n_lives_learn": len(arms["learner"]["lives"]),
        "n_lives_twin": len(arms["twin"]["lives"]),
        "n_lives_rand": len(arms["random"]["lives"]),
        # the question
        "z_rand": _welch_z(fr["learner"], fr["random"]),
        "n_eval_lives_learn": len(ev["learner"]),
        "n_eval_lives_rand": len(ev["random"]),
        "n_eval_lives_twin": len(ev["twin"]),
        "sd_frac_learn": float(fr["learner"].std(ddof=1)) if len(fr["learner"]) > 1 else 0.0,
        "sd_frac_rand": float(fr["random"].std(ddof=1)) if len(fr["random"]) > 1 else 0.0,
        "optimiser_steps": arms["learner"]["optimiser_steps"],
        "wall_s": time.perf_counter() - t0,
    })

    # ── the known-answer control decides whether anything else may be read ──
    misses = []
    for k, (want, tol) in KNOWN.items():
        got = out[k]
        if abs(got - want) > tol:
            misses.append(f"{k} {got:.6g} vs pilot {want:.6g} (tol {tol:g})")
    for k, want in KNOWN_INT.items():
        if out[k] != want:
            misses.append(f"{k} {out[k]} vs pilot {want}")
    out["reproduced"] = not misses
    out["misses"] = misses

    if misses:
        out["reading"] = "UNREPRODUCED"
    elif out["z_rand"] >= BAR:
        out["reading"] = "CLEARS"
    elif out["z_rand"] <= -BAR:
        out["reading"] = "DEAD"
    else:
        out["reading"] = "INDETERMINATE"

    with open(ARTIFACT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print()
    print(f"  READING: {out['reading']}")
    if misses:
        print("  the known-answer control MISSED — no number here is evidence:")
        for m in misses:
            print("    -", m)
    else:
        print("  known-answer control: REPRODUCED "
              f"(learn {out['frac_shelt_learn']:.4f} / twin "
              f"{out['frac_shelt_twin']:.4f} / rand "
              f"{out['frac_shelt_rand']:.4f}, frozen {out['rand_frozen']})")
    print(f"  z_rand {out['z_rand']:+.4f} against the quoted bar "
          f"{BAR:+.1f}  (learner {out['frac_shelt_learn']:.4f} "
          f"+/- {out['sd_frac_learn']:.4f} over {out['n_eval_lives_learn']} "
          f"eval lives  vs  random {out['frac_shelt_rand']:.4f} +/- "
          f"{out['sd_frac_rand']:.4f} over {out['n_eval_lives_rand']})")
    print(f"  wrote {ARTIFACT}  ({out['wall_s']:.0f}s)")
    print("  NO LEDGER ROW WAS WRITTEN. SH.02 stays pilot-blocked and its "
          "HEADROOM VOID stands.")
    return out


if __name__ == "__main__":
    main()
