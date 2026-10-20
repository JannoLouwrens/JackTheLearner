"""PS.09 re-based known-answer reference — can an HONESTLY-BASED reference
read the worth-it sign at this venue well enough to clear the UNMOVED
`KA_SIGNAL_MIN` 0.90, or does the floor return to the owner?

## WHO ORDERED IT, AND WHAT IT DISCHARGES

`ps09-known-answer-floor-was-calibrated-on-an-oracle-cut` (DISPOSITIONED
2026-10-03, execution the builder's, DUE 2026-10-19). THE KNOWN-ANSWER
REFERENCE RULING's order, verbatim: *"what `DUE:` buys is the re-based
reference MEASURED at `PS.09`'s venue, honest split, parameter count declared,
read against the unmoved 0.90 — with both branches pre-committed above so
neither is a surprise that gets argued with afterwards."* The ruling refused to
carry `PS.05`'s 0.9994 across venues: *"recovering `d_hat` for free does not
make the sign free, because the sign needs the cost model too."* This file is
that measurement. It is a DIAGNOSTIC in the family idiom
(`ps05_odour_reference_probe.py`, `t301_shuffle_probe.py`): it writes no
ledger row, moves no bar in either direction, and registers no conjunct.

## THE BASIS, CHOSEN BY THE RULING'S PREFERENCE ORDER — AND WHY FULL (i)
## IS NOT AVAILABLE FOR A SIGN TARGET

The ruling's order of strength: (i) the venue's declared law inverted, ZERO
fitted parameters; (ii) a fixed-basis fit on named columns with its split
declared; (iii) a search, train-confined. At `PS.09`:

  * The SENSE side has a declared law: `C = A0 * k * exp(-d / LAMBDA_M)`
    (`odour.py:173`, `A0` 1.0, `LAMBDA_M` 2.0). Inverting it is free:
    `d_hat = -LAMBDA_M * ln(C / A0)` — exact for k=1 and reading
    `d - LAMBDA_M*ln k` for k=2. ZERO parameters. This is basis (i).
  * The PRICE side has NO declared law. The trip's cost is the REAL
    `NeedLayer` integration of gear-scaled thrash — a measured consequence of
    the body's physics (PS.05 had to MEASURE the price curve; `p_ref` is
    re-measured per seed). No closed form exists in the repo to invert, so a
    zero-parameter sign reference would have to invent one, which is a basis
    pretending to be a law.

So the strongest HONESTLY AVAILABLE reference is a hybrid, and its parameter
count is declared here as the ruling requires: **basis (i) for distance
recovery (0 parameters) composed with an affine pricing model
`net_hat = a + b * d_hat` fit on the TRAIN rows only (2 fitted parameters,
`np.linalg.lstsq`)** — the venue's measured payoff/price line, which is the
one thing the ruling says the sign genuinely needs. The split is the spec's
own held-out-BY-TRIP split (`trips[:-N_TEST_TRIPS]`), byte-identical to what
the registered probe and the old cut reference are judged on, and the sign is
read with the spec's own `_bal_acc` on the same held-out rows. Note the
algebra: `d_hat` is affine in `ln C`, so this reference is exactly an affine
fit in log-concentration — the basis the 09-27 `PS.05` addendum showed is the
law-matched one (*"the target is a log of the channel"*; here the latent
distance is, and net is affine in distance to first order).

## WHAT IS REPORTED BESIDE IT, AND WHY EACH

  * `ka_cut_replay`, `probe_replay` — the spec's own `_ka_ref_acc` and
    `_score`, recomputed on the same trips: THE REPLAY RECEIPT. The ledger
    stores means/stds only (`aggregate-hides-worst-seed`), so the receipt is
    that the recomputed per-seed values aggregate to the recorded
    `ka_ref_acc` 0.806667 +/- 0.0573488 and `probe_bal_acc`
    0.546667 +/- 0.0754247 (attempt 2, 2026-09-25T08:18:29). A replay that
    does not reproduce those digits is broken and NOTHING else here may be
    quoted.
  * `ka_rebased_oracle` — the SAME 2-parameter fit taken ON the held-out
    rows. Reported only to price this basis's structural leak (forecast:
    small; a 2-parameter fit has almost nothing to memorise). Not a result.
  * `ka_rebased_trip` — the same honest reference with per-row predictions
    averaged per trip before the sign is read (the 09-25 diagnostic reported
    a per-trip variant; disclosed for comparability, decides nothing).
  * `ka_raw_honest` — the same 2-parameter fit on RAW C instead of
    `d_hat`. The basis-mismatch comparison the 09-27 lesson asks for: two
    references with the same split and different bases are two different
    claims about the venue.
  * `monotone_ceiling` — the spec's cut search fit ON the held-out rows:
    the saturating argmax the ruling REFUTED as a reference, reported here
    ONLY as a ceiling diagnostic for monotone-in-C readers on this draw
    (the seed-1 datum read 1.0000). If the ceiling is high while the honest
    reference is low, the miss is boundary PLACEMENT; if the ceiling itself
    is under 0.90, NO monotone reader can clear the floor on that draw,
    however it is fit. Oracle information, decides nothing, calibrates
    nothing.
  * DECOYS — the identical honest pipeline on the sourceless `decay`,
    `smoke`, `water` channels (pure clipped `N(0, 1e-3)` in this fixture:
    the only `odour.Source` is the offer's food channel,
    `ps_09_worth_it_is_real.py:444`). The reference's own must-fail
    control: if a decoy reads the sign at >= 0.70, the reference is reading
    something other than the odour law and every number above is void.
    Non-positive concentrations drop out of `ln` and the surviving count is
    DISCLOSED, never clamped silently.

## PRE-REGISTERED FORECAST, WRITTEN BEFORE THE SCRIPT WAS RUN ONCE

  * Replay FAITHFUL: recomputed means/stds match the ledger to 1e-5.
  * `ka_rebased_honest` lands NEAR the honest cut's 0.75-0.87 band, and the
    worst seed is BELOW 0.90 — **I forecast branch (B)**. Reason: an affine
    zero-crossing is placed by regression on net VALUES, not optimised for
    sign accuracy, so it should sit near, not above, an accuracy-optimised
    train cut; and 0.90 on a 5/5-class test draw allows at most one trip's
    rows wrong.
  * `|ka_rebased_oracle - ka_rebased_honest| <= 0.10` on every seed (2
    parameters have almost nothing to memorise). If it is LARGE, that is a
    holdout finding, not an estimator finding, and it must be said.
  * `monotone_ceiling >= 0.90` on at least two seeds (the seed-1 datum was
    1.0000): the signal is separable and the honest question is placement.
  * Every decoy < 0.70 (expected ~0.5), with ~half its rows dropped by the
    sign filter.

## THE TWO BRANCHES, PRE-COMMITTED BY THE RULING — NEITHER IS A SURPRISE

  * **(A) `ka_rebased_honest >= 0.90` on ALL THREE seeds**: the floor is
    clearable by an honestly-based reference and nothing was ever wrong but
    the basis. The follow-on (separate act): re-base the spec's KA reference
    to this form with its parameter count declared beside the floor —
    strengthening, no bar moved.
  * **(B) the worst seed is BELOW 0.90**: `KA_SIGNAL_MIN` 0.90 sits above
    what ANY honest reference measured at this venue can read (cut search
    0.75-0.87; law-inverted affine as measured here), and the floor RETURNS
    TO THE OWNER as a bar move no desk may make — with this file's numbers
    as the evidence and NO replacement floor proposed off these rows, which
    would be the original defect one measurement later.

## WHAT THIS FILE MAY NOT DO

`KA_SIGNAL_MIN` 0.90, `KA_GAP_MAX` 0.10, `ACC_MIN` 0.65 and every other bar
are byte-unmoved. Nothing here can rescue PS.09 (VOID a2) or any sibling's
FAIL: a reference reading the sign does not make the registered probe read
it. Any floor recalibration is a THRESHOLD decision and is the owner's.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .. import needs, odour
from ..tests import ps_09_worth_it_is_real as ps09

SEEDS = (0, 1, 2)
ARTIFACT = "/data/ps09_rebased_reference_probe.json"

# The recorded aggregates this replay must reproduce (attempt 2, VOID,
# 2026-09-25T08:18:29; the ledger stores means/stds only).
LEDGER_KA = (0.806667, 0.0573488)
LEDGER_PROBE = (0.546667, 0.0754247)
REPLAY_TOL = 1e-5

DECOY_MAX = 0.70        # a decoy at or above this voids the reference
_BASE = ps09.KIN_DIM + needs.NEED_DIM   # odour block offset (spec's own)


def _scalar(rows, ch: str) -> np.ndarray:
    """Per-row mean of the L/R concentration dims for one channel — the
    spec's `_ka_scalar`, generalised to the decoy channels."""
    i = odour.CHANNEL_INDEX[ch]
    a = np.asarray(rows)
    return 0.5 * (a[:, _BASE + i] + a[:, _BASE + odour.C + i])


def _rows_y(trips) -> tuple:
    rows = [r for t in trips for r in t[0]]
    y = np.array([t[1] for t in trips for _ in t[0]])
    return rows, y


def _dhat(c: np.ndarray) -> tuple:
    """Basis (i): invert the declared law. Non-positive C is dropped and
    counted, never clamped (an epsilon clamp is a domain assertion wearing
    a safety guard's costume — LESSONS 2026-09-27)."""
    n_bad = int((c <= 0.0).sum())
    safe = np.where(c > 0.0, c, np.nan)
    return -odour.LAMBDA_M * np.log(safe / odour.A0), n_bad


def _affine_sign(u_fit, y_fit, u_eval, y_eval) -> tuple:
    """net_hat = a + b*u, 2 parameters, lstsq on the fit rows; balanced sign
    accuracy on the eval rows via the spec's own `_bal_acc`. NaN rows (from
    the ln domain filter) are dropped per side and the counts returned."""
    okf, oke = np.isfinite(u_fit), np.isfinite(u_eval)
    uf, yf = u_fit[okf], y_fit[okf]
    ue, ye = u_eval[oke], y_eval[oke]
    if uf.size < 2 or ue.size == 0 or np.unique(uf).size < 2:
        return float("nan"), (None, None), int(okf.sum()), int(oke.sum())
    A = np.stack([np.ones_like(uf), uf], axis=1)
    coef, *_ = np.linalg.lstsq(A, yf, rcond=None)
    pred = coef[0] + coef[1] * ue
    return (ps09._bal_acc(ye, pred), (float(coef[0]), float(coef[1])),
            int(okf.sum()), int(oke.sum()))


def _oracle_cut_acc(x: np.ndarray, y: np.ndarray) -> float:
    """The spec's cut search fit AND scored on the same rows — the
    saturating argmax, reported ONLY as the monotone ceiling diagnostic."""
    xs = np.unique(x)
    cuts = np.concatenate([[xs[0] - 1.0], (xs[:-1] + xs[1:]) / 2.0,
                           [xs[-1] + 1.0]])
    best = -1.0
    for direction in (1.0, -1.0):
        for cut in cuts:
            best = max(best, ps09._bal_acc(y, direction * (x - cut)))
    return best


def _seed_report(seed: int) -> dict:
    d = ps09._collect(seed)
    if "refused" in d or d.get("no_food") or d.get("dead", 0.0) > 0.0:
        return {"seed": seed, "usable": False, "why": str(sorted(d))}

    trips = [(t["rows"], t["net"]) for t in d["leg"] if t["rows"]]
    if len(trips) != ps09.N_LEG:
        return {"seed": seed, "usable": False,
                "why": f"{len(trips)} trips != N_LEG"}
    tr, te = trips[:-ps09.N_TEST_TRIPS], trips[-ps09.N_TEST_TRIPS:]
    tr_rows, y_tr = _rows_y(tr)
    te_rows, y_te = _rows_y(te)
    n_cols = ps09.KIN_DIM + needs.NEED_DIM + ps09.ODOUR_DIM

    # ── replay receipts: the spec's own instruments on the same trips ──
    ka_cut_replay = ps09._ka_ref_acc(trips)
    probe_replay = ps09._score(trips, n_cols)

    # ── the re-based reference: law inversion + 2-parameter pricing ──
    u_tr, bad_tr = _dhat(_scalar(tr_rows, "food"))
    u_te, bad_te = _dhat(_scalar(te_rows, "food"))
    honest, coef, n_fit, n_eval = _affine_sign(u_tr, y_tr, u_te, y_te)
    oracle, _, _, _ = _affine_sign(u_te, y_te, u_te, y_te)
    raw_tr = _scalar(tr_rows, "food").astype(float)
    raw_te = _scalar(te_rows, "food").astype(float)
    raw_honest, _, _, _ = _affine_sign(raw_tr, y_tr, raw_te, y_te)

    # per-trip variant: mean prediction per held-out trip, sign vs trip net
    trip_acc = float("nan")
    if coef[0] is not None:
        preds, ys = [], []
        for rows, net in te:
            u, _ = _dhat(_scalar(rows, "food"))
            u = u[np.isfinite(u)]
            if u.size:
                preds.append(coef[0] + coef[1] * float(u.mean()))
                ys.append(net)
        if preds:
            trip_acc = ps09._bal_acc(np.array(ys), np.array(preds))

    ceiling = _oracle_cut_acc(_scalar(te_rows, "food"), y_te)

    decoys = {}
    for ch in ("decay", "smoke", "water"):
        ud_tr, _ = _dhat(_scalar(tr_rows, ch))
        ud_te, _ = _dhat(_scalar(te_rows, ch))
        acc, _, nf, ne = _affine_sign(ud_tr, y_tr, ud_te, y_te)
        decoys[ch] = {"acc": acc, "n_fit": nf, "n_eval": ne}

    return {
        "seed": seed, "usable": True,
        "n_train_rows": len(tr_rows), "n_test_rows": len(te_rows),
        "n_pos_test": int((y_te > 0).sum()), "n_neg_test": int((y_te <= 0).sum()),
        "ka_cut_replay": float(ka_cut_replay),
        "probe_replay": float(probe_replay),
        "ka_rebased_honest": float(honest),
        "ka_rebased_oracle": float(oracle),
        "oracle_gap": float(oracle - honest),
        "ka_rebased_trip": float(trip_acc),
        "ka_raw_honest": float(raw_honest),
        "monotone_ceiling": float(ceiling),
        "coef": list(coef),
        "n_nonpositive_train": bad_tr, "n_nonpositive_test": bad_te,
        "decoys": decoys,
    }


def main() -> None:
    seeds = tuple(int(a) for a in sys.argv[1:]) or SEEDS
    try:
        with open(ARTIFACT) as fh:
            merged = {r["seed"]: r for r in json.load(fh)}
    except (OSError, ValueError):
        merged = {}
    for s in seeds:
        merged[s] = _seed_report(s)
    out = [merged[s] for s in sorted(merged)]
    with open(ARTIFACT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    ok = [r for r in out if r.get("usable")]
    print("PS.09 RE-BASED KNOWN-ANSWER REFERENCE — basis (i) inversion + "
          "2-parameter pricing")
    print("=" * 78)
    for r in out:
        if not r.get("usable"):
            print(f"  seed {r['seed']}  UNUSABLE — {r['why']}")
    if len(ok) == len(SEEDS) and all(r["seed"] in SEEDS for r in ok):
        print("\nREPLAY CHECK (must reproduce before anything else is quoted)")
        faithful = True
        for name, key, (want_m, want_s) in (
                ("ka_ref_acc", "ka_cut_replay", LEDGER_KA),
                ("probe_bal_acc", "probe_replay", LEDGER_PROBE)):
            vals = [r[key] for r in ok]
            m, s = float(np.mean(vals)), float(np.std(vals))
            good = abs(m - want_m) <= REPLAY_TOL and abs(s - want_s) <= REPLAY_TOL
            faithful &= good
            print(f"  {name}: recomputed {m:.6f} +/- {s:.6f}  vs ledger "
                  f"{want_m} +/- {want_s}  "
                  f"{'MATCH' if good else 'MISMATCH — DO NOT QUOTE'}")
        print(f"  REPLAY {'FAITHFUL' if faithful else 'BROKEN'}")

    print("\nTHE RE-BASED REFERENCE (honest: fit on train rows only; "
          "2 parameters declared)")
    for r in ok:
        print(f"  seed {r['seed']}: honest {r['ka_rebased_honest']:.4f}  "
              f"oracle {r['ka_rebased_oracle']:.4f}  "
              f"gap {r['oracle_gap']:+.4f}  trip-mean {r['ka_rebased_trip']:.4f}  "
              f"rawC {r['ka_raw_honest']:.4f}  "
              f"ceiling {r['monotone_ceiling']:.4f}  "
              f"cut-replay {r['ka_cut_replay']:.4f}")
    print("\nDECOYS (must fail: each < %.2f, surviving n disclosed)" % DECOY_MAX)
    for r in ok:
        ds = "  ".join(f"{ch} {v['acc']:.3f} (n {v['n_fit']}/{v['n_eval']})"
                       for ch, v in r["decoys"].items())
        print(f"  seed {r['seed']}: {ds}")

    if ok:
        worst = min(r["ka_rebased_honest"] for r in ok)
        decoy_ok = all(not np.isfinite(v["acc"]) or v["acc"] < DECOY_MAX
                       for r in ok for v in r["decoys"].values())
        print(f"\nWORST SEED honest reference: {worst:.4f}  vs UNMOVED "
              f"KA_SIGNAL_MIN {ps09.KA_SIGNAL_MIN}")
        print(f"DECOY CONTROL: {'FIRED AS REQUIRED' if decoy_ok else 'A DECOY READ THE SIGN — REFERENCE VOID'}")
        if len(ok) == len(SEEDS):
            branch = "A" if worst >= ps09.KA_SIGNAL_MIN else "B"
            print(f"BRANCH ({branch}): " + (
                "the floor is clearable by an honestly-based reference"
                if branch == "A" else
                "the floor sits above what any honest reference reads here "
                "— it returns to the owner"))
    print(f"\nARTIFACT {ARTIFACT}")


if __name__ == "__main__":
    main()
