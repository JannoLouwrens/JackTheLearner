"""PS.05 known-answer control — is `far` illegible at this venue, or is the
ESTIMATOR the thing that cannot read it?

## WHY THIS EXISTS, AND WHO ORDERED IT

`ps05-legibility-holdout-is-a-band-lottery` (DISPOSITIONED, DUE 2026-10-02,
execution the builder's) names three candidate readings for PS.05 attempt 1's
sole fired conjunct — `probe_r2` **-0.3775 +- 0.6053** against
`PROBE_R2_MIN` 0.35, every world gate green on every seed:

    (a) the estimator — trip-level holdout with 4 test units makes the
        headline a lottery over which bands the test trips occupy;
    (b) genuine range limit — odour legibility may honestly die past ~5 m at
        this LAMBDA_M and noise floor, and D_LEG (1, 6) m spans past the
        sense's edge;
    (c) probe capacity, the sibling's reading.

THE PS-FAMILY LEGIBILITY RULING (Review DAILY 2026-09-25) added the sentence
this file answers, verbatim from the row: *"the ruling does NOT decide your
reading (b) genuine range limit — it holds that (b) is not READABLE off this
estimator's output until the estimator clears a known-answer control."*

So (b) is unreadable until somebody measures what a BY-CONSTRUCTION-LEGIBLE
channel reads on THE SAME ROWS. That measurement is this file. It is a
DIAGNOSTIC: it writes no ledger row, moves no bar in either direction, and
registers no conjunct. Precedents for the form: `t301_shuffle_probe.py`,
`lg03_blind_twin_probe.py`, `sm03_readout_sweep_probe.py`.

## WHAT MAKES THE REFERENCE BY-CONSTRUCTION LEGIBLE

`odour.StaticField` is an analytic law, not a learned sensor (odour.py:173):

    C = A0 * strength * exp(-d / LAMBDA_M) + N(0, NOISE_SIGMA)

with `A0` 1.0, `LAMBDA_M` 2.0 m, `NOISE_SIGMA` 1e-3. PS.05's legibility trips
place ONE food source at the target and record `OdourSensor.obs` — 12 floats,
`[left C x4, right C x4, d(mean)/dt x4]` — as the LAST block of the feature
vector. Food is channel 0, so the two live concentration columns are indices
`KIN_DIM + NEED_DIM + 0` = 54 (left) and `+ 4` = 58 (right), and the target is
`y = remaining / D_LEG[1]`.

Inverting the law needs no fitting at all:

    d_hat = -LAMBDA_M * ln( mean(C_left, C_right) )      # metres
    y_hat = d_hat / D_LEG[1]

That is a ZERO-PARAMETER reference. Its only bias is bilateral: the mean of
two exponentials exceeds the exponential of the mean distance by convexity,
which reads `d_hat` low by ~`HEAD_SEP_M^2 / (8 * LAMBDA_M)` ~ 0.003 m — four
orders under the test-band spread. Three readings are reported beside it, and
the split is recorded for every one of them (`a-quoted-diagnostic-number-
becomes-a-calibration-constant`, LESSONS 2026-09-25: record the fit/eval split
WHEN the number is measured, or the floor registered against it inherits the
leak):

  * `r2_inv`     ZERO parameters, fit on NOTHING, scored on the test trips.
  * `r2_fit1d`   y ~ a + b*u on the TRAIN trips only, u = -ln(C_mean);
                 scored on the test trips. This is PS.09's honest form —
                 direction and cut learned on train rows (`ps09-known-answer-
                 floor-was-calibrated-on-an-oracle-cut`).
  * `r2_oracle`  the SAME 1-D fit taken ON THE TEST ROWS. Reported only to
                 price the leak that defect was about. It is not a result.

## THE REFERENCE CARRIES ITS OWN MUST-FAIL CONTROL

A known-answer control with no control of its own is an assertion. `decay`,
`smoke` and `water` have NO source in this fixture, so their columns are pure
`N(0, 1e-3)`. The identical arithmetic on each must FAIL — if a decoy channel
also reads the sign, the reference is reading something other than the odour
law and every number above is void. Reported as `r2_inv_decay/smoke/water`.

## AND THE REPLAY IS CHECKED AGAINST THE LEDGER BEFORE ANY REFERENCE IS READ

This file calls PS.05's own `_collect` and `_score` — the committed functions,
not a re-implementation — so the rows are THE ROWS. The proof is that the
registered probe reproduces: per-seed `probe_r2` must come back
**-1.126 / +0.356 / -0.362** (the row's re-derived per-seed values, quoted in
the queue row; the ledger stores the mean -0.377524). A replay that does not
reproduce those digits is a broken replay and NOTHING else it prints may be
quoted. This is the same discipline the reference is here to impose, pointed
at the instrument that imposes it.

## PRE-REGISTERED FORECAST, WRITTEN BEFORE THE SCRIPT WAS RUN ONCE

Committed in its own commit ahead of any number, the `SH.02` diagnostic's form
(builder, 2026-09-26). I forecast:

  * **`r2_inv` >= 0.95 on ALL THREE seeds**, including seed 0, whose test
    bands (0.61-1.21 in y, i.e. 3.7-7.3 m) the queue row identifies as the
    far-heavy draw that sent the headline to -1.126. Reason: at 7.3 m,
    `C = 6.6e-3` against a 1e-3 noise floor, so the law is still invertible
    with SNR ~7 at the worst row, and R^2 against the TEST mean is insensitive
    to which band the test trips occupy once the estimate is near-exact.
  * **`r2_inv_decay`, `r2_inv_smoke`, `r2_inv_water` all strongly negative.**
  * **`r2_oracle` - `r2_fit1d` small** (< 0.05). PS.09's leak came from a
    54-feature fit memorising trip identity; a 1-D fit has almost nothing to
    memorise, so this venue should NOT show PS.09's oracle gap. If it DOES,
    that is a finding about the holdout rather than the estimator and it
    argues for the row's reading (a).

**IF THE FORECAST HOLDS, reading (b) IS REFUTED AT THIS VENUE ON
MEASUREMENT** — `far` is legible en route to a reader that knows the world's
own law, the sense carries it at the full D_LEG range, and what failed is the
200-feature RFF+ridge probe PS.05 registered. **IF `r2_inv` IS LOW TOO**, (b)
survives and the sense's edge is the finding; the spec should then measure
WHERE legibility ends rather than average over it, which is the row's own
alternative.

**WHAT THIS FILE MAY NOT DO, and the boundary is the reason it is a probe and
not an edit.** `PROBE_R2_MIN` 0.35 does not move in either direction. Nothing
here can rescue PS.05's FAIL: a reference reading the target does not make the
registered probe read it. Recalibrating any floor against these numbers is a
THRESHOLD decision and belongs to the Review (`1^13`: *"if you find yourself
wanting to move a bar to make one of these pass, stop and route it back"*).
"""
from __future__ import annotations

import json
import math

import numpy as np

from .. import odour
from ..tests import ps_05_far_is_a_price as ps05

SEEDS = (0, 1, 2)

# The per-seed registered readings this replay must reproduce before any
# reference number below may be quoted (queue row, re-derived from attempt 1).
LEDGER_PER_SEED = (-1.126, 0.356, -0.362)
REPLAY_TOL = 5e-3

# Column layout, derived from the spec's own constants rather than typed.
_ODOUR0 = ps05.KIN_DIM + 9                      # needs.NEED_DIM == 9
_KIN_NEED = _ODOUR0                             # the amputation suffix cut


def _channel_pair(ch: str) -> tuple:
    """(left, right) column indices for one odour channel."""
    i = odour.CHANNEL_INDEX[ch]
    return _ODOUR0 + i, _ODOUR0 + odour.C + i


def _split(trips: list) -> tuple:
    """PS.05's own held-out-BY-TRIP split, not a re-derivation of it."""
    return trips[:-ps05.N_TEST_TRIPS], trips[-ps05.N_TEST_TRIPS:]


def _u(rows: list, ch: str) -> tuple:
    """u = -LAMBDA_M * ln(C_mean) in metres, and the count of rows whose
    concentration was non-positive. The count is REPORTED, never clamped
    silently: an epsilon clamp is a domain assertion wearing a safety
    guard's costume (LESSONS, 2026-09-27)."""
    li, ri = _channel_pair(ch)
    c = np.array([0.5 * (r[0][li] + r[0][ri]) for r in rows])
    n_bad = int((c <= 0.0).sum())
    safe = np.where(c > 0.0, c, np.nan)
    return -odour.LAMBDA_M * np.log(safe), n_bad


def _y(rows: list) -> np.ndarray:
    return np.array([r[1] for r in rows])


def _drop_nan(u: np.ndarray, y: np.ndarray) -> tuple:
    ok = np.isfinite(u)
    return u[ok], y[ok], int((~ok).sum())


def _seed_report(seed: int) -> dict:
    d = ps05._collect(seed)
    if "refused" in d or d.get("dead", 0.0) > 0.0:
        return {"seed": seed, "usable": False,
                "why": "borrow refused" if "refused" in d else "dead events"}

    trips = [t["rows"] for t in d["leg"] if t["rows"]]
    tr, te = _split(trips)
    tr_rows = [r for t in tr for r in t]
    te_rows = [r for t in te for r in t]
    n_cols = ps05.KIN_DIM + 9 + ps05.ODOUR_DIM

    # ── the replay's own known-answer control: the registered probe ──
    r2_reg = ps05._score(trips, n_cols)
    r2_amp = ps05._score(trips, _KIN_NEED)

    # ── the by-construction reference on the LIVE channel ──
    u_tr, n_bad_tr = _u(tr_rows, "food")
    u_te, n_bad_te = _u(te_rows, "food")
    y_tr, y_te = _y(tr_rows), _y(te_rows)
    u_tr, y_tr, n_nan_tr = _drop_nan(u_tr, y_tr)
    u_te_f, y_te_f, n_nan_te = _drop_nan(u_te, y_te)

    D_MAX = ps05.D_LEG[1]
    r2_inv = ps05._r2(y_te_f, u_te_f / D_MAX)

    def _fit1d(uf, yf, ue):
        A = np.stack([np.ones_like(uf), uf], axis=1)
        coef, *_ = np.linalg.lstsq(A, yf, rcond=None)
        return coef, coef[0] + coef[1] * ue

    coef_tr, pred_tr_fit = _fit1d(u_tr, y_tr, u_te_f)
    r2_fit1d = ps05._r2(y_te_f, pred_tr_fit)
    coef_or, pred_or = _fit1d(u_te_f, y_te_f, u_te_f)
    r2_oracle = ps05._r2(y_te_f, pred_or)

    # ── the reference's own must-fail control: sourceless channels ──
    decoys = {}
    for ch in ("decay", "smoke", "water"):
        ud, _ = _u(te_rows, ch)
        ud, yd, _ = _drop_nan(ud, _y(te_rows))
        decoys[ch] = (ps05._r2(yd, ud / D_MAX) if ud.size else float("nan"),
                      int(ud.size))

    return {
        "seed": seed, "usable": True,
        "n_trips": len(trips), "n_train_rows": len(tr_rows),
        "n_test_rows": len(te_rows),
        "test_y_min": float(y_te.min()), "test_y_max": float(y_te.max()),
        "train_y_min": float(np.array([r[1] for r in tr_rows]).min()),
        "train_y_max": float(np.array([r[1] for r in tr_rows]).max()),
        "probe_r2_replay": r2_reg,
        "control_r2_replay": r2_amp,
        "r2_inv": r2_inv,
        "r2_fit1d": r2_fit1d,
        "r2_oracle": r2_oracle,
        "oracle_gap": r2_oracle - r2_fit1d,
        "fit1d_coef": [float(c) for c in coef_tr],
        "n_nonpositive_train": n_bad_tr, "n_nonpositive_test": n_bad_te,
        "n_dropped_train": n_nan_tr, "n_dropped_test": n_nan_te,
        "decoys": {k: {"r2": v[0], "n": v[1]} for k, v in decoys.items()},
    }


def main() -> None:
    out = [_seed_report(s) for s in SEEDS]
    print("PS.05 ODOUR REFERENCE PROBE — known-answer control for the "
          "legibility conjunct")
    print("=" * 78)

    ok = [r for r in out if r.get("usable")]
    print("\nREPLAY CHECK (must reproduce before anything else is quoted)")
    faithful = True
    for r, want in zip(out, LEDGER_PER_SEED):
        if not r.get("usable"):
            print(f"  seed {r['seed']}  UNUSABLE — {r['why']}")
            faithful = False
            continue
        got = r["probe_r2_replay"]
        good = abs(got - want) <= REPLAY_TOL
        faithful &= good
        print(f"  seed {r['seed']}  probe_r2 replay {got:+.4f}  "
              f"vs row {want:+.4f}  "
              f"{'MATCH' if good else 'MISMATCH — DO NOT QUOTE'}")
    if ok:
        print(f"  replay mean {np.mean([r['probe_r2_replay'] for r in ok]):+.6f}"
              f"  vs ledger probe_r2 -0.377524")
    print(f"  REPLAY {'FAITHFUL' if faithful else 'BROKEN'}")

    print("\nTHE REFERENCE — by-construction inversion of C = exp(-d/LAMBDA_M)")
    for r in ok:
        print(f"  seed {r['seed']}: r2_inv {r['r2_inv']:+.4f} (0 params)  "
              f"r2_fit1d {r['r2_fit1d']:+.4f} (train-fit)  "
              f"r2_oracle {r['r2_oracle']:+.4f}  "
              f"gap {r['oracle_gap']:+.4f}")
        print(f"           registered probe {r['probe_r2_replay']:+.4f}  "
              f"amputated {r['control_r2_replay']:+.4f}  "
              f"test y [{r['test_y_min']:.3f}, {r['test_y_max']:.3f}]  "
              f"train y [{r['train_y_min']:.3f}, {r['train_y_max']:.3f}]")
        print(f"           rows train/test {r['n_train_rows']}/"
              f"{r['n_test_rows']}  non-positive C "
              f"{r['n_nonpositive_train']}/{r['n_nonpositive_test']}  "
              f"dropped {r['n_dropped_train']}/{r['n_dropped_test']}")

    print("\nTHE REFERENCE'S MUST-FAIL CONTROL — sourceless channels")
    for r in ok:
        ds = "  ".join(f"{k} {v['r2']:+.3f}" for k, v in r["decoys"].items())
        print(f"  seed {r['seed']}: {ds}")

    if ok:
        print("\nWORST SEED (the discipline PS.05's own gate uses)")
        print(f"  r2_inv    min {min(r['r2_inv'] for r in ok):+.4f}")
        print(f"  r2_fit1d  min {min(r['r2_fit1d'] for r in ok):+.4f}")
        print(f"  probe_r2  min {min(r['probe_r2_replay'] for r in ok):+.4f}"
              f"   vs PROBE_R2_MIN {ps05.PROBE_R2_MIN}")

    print("\nJSON")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
