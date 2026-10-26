"""PS.08 known-answer calibration pair — the drain-rate channel, measured
under the honest/oracle discipline rules 1-3 require, with NO floor declared.

## WHY THIS EXISTS, AND WHO ORDERED IT

THE KNOWN-ANSWER REFERENCE RULING (Review DAILY 2026-10-03), clause (d):
`PS.08` *"stays held one more step, and not on the coupling, which is now
closed. It is held on rule 3 and on nothing else: its candidate channel has
never been measured under the honest/oracle pair the other two specs now have
in the table above, and this ruling has just forbidden declaring a floor
without that pair. That measurement is cheap and is ordered on the sibling
row's own clock, not here."* The sibling row is
`ps08-amputation-control-out-reads-the-probe-intero-is-not-clock-like`
(DISPOSITIONED, DUE 2026-10-31). This file is that measurement.

Clause (d) also pins the channel: *"part 1's known-answer channel and part 2's
blinded quantity SHALL be the same quantity"* — and part 2's blinded quantity
is the e/w DRAIN RATE. So the candidate measured here is the drain rate, not a
convenient substitute.

It is a DIAGNOSTIC in the `ps05_odour_reference_probe.py` /
`ps09_rebased_reference_probe.py` form: it writes no ledger row, moves no bar
in either direction, registers no conjunct, and — explicitly — DECLARES NO
FLOOR. The floor declaration is held behind `D51` (class goal, decide_by
2026-11-03): *"a floor may not be declared off this spec's own rows while that
answer is open"* (the row's WAITS-ON note, 2026-10-20). Measuring the pair is
the prerequisite D51's answer will be applied to; it is not the application.

## THE REPLAY CONTRACT, AND WHERE ITS TARGETS COME FROM

This file calls PS.08's own `_collect` and `_speed_score` — the committed
functions, not re-implementations — and runs the FULL per-seed pipeline (cost
half included), because the PART 2 RECORD measured that a leg-only run sees a
different microsleep stream than the registered run. Before any reference
number is quoted, the registered probe and the rate-blind control must
reproduce.

The ledger stores means and stds, not per-seed values. The per-seed multisets
are DERIVED here, before running, from the quantisation the rig forces:
balanced accuracy on 8 test trips split 4/4 is a multiple of 1/16, so a
3-seed multiset is three sixteenths summing to 3*mean. For the live row
(attempt 3, `ran_at` 2026-09-28T11:24:35, clean stamp):

  probe_bal_acc   mean 0.583333 std 0.0589256  ->  {8,10,10}/16
                  = {0.500, 0.625, 0.625}   (unique at that std among
                  sixteenth-triples summing to 28/16; nearest rivals
                  {8,8,12}/16 std 0.1179 and {9,9,10}/16 std 0.0295)
  control_bal_acc mean 0.750000 std 0.176777   ->  {8,14,14}/16
                  = {0.500, 0.875, 0.875}   (rivals {10,12,14}/16 std
                  0.102, {8,12,16}/16 std 0.204)
  control_caught  mean 0.333333 = exactly one seed at or under 0.60 —
                  consistent: the 0.500 seed.

A replay whose per-seed multisets do not match EXACTLY is a broken replay and
NOTHING else this file prints may be quoted.

## THE CANDIDATE REFERENCE: the in-bout drain rate, pre-roll-free

Each leg trip's rows are taken at decisions 4 and 8 of each of 3 bouts —
within-bout PAIRS over an identical 0.8 s push window. So the venue hands us
a pure rate read on a fixed window, with the random pre-roll CANCELLED by
construction (both rows of a pair sit after the same pre-roll):

    stat(trip) = mean over bouts k of
                 (e[dec4,k] - e[dec8,k]) + (w[dec4,k] - w[dec8,k])

e and w are NAMED columns 0 and 1 of the intero block (`needs.NeedLayer.obs`:
[e, w, p, T_norm, f, c, i, d, pain]). Scored through the registered probe's
own fixed form (`_speed_score`'s): threshold = TRAIN median, sign fit on
TRAIN, balanced accuracy on the held-out 8 trips — the same split, same trip
order, same draw as the registered probe. **Declared parameter count: 2**
(sign, threshold), both fit on train only. Basis (ii) of rule 3; basis (i) is
unavailable because inverting M = M_BASAL + kappa_act*p_mech returns p_mech,
and p_mech -> class needs a cut the law does not supply.

Reported beside it, never gated:

  * `sincestart` — the same arithmetic on (e_start - e_row)+(w_start - w_row)
    against the frozen trip-start block: the claim's own priced quantity read
    early, but pre-roll-CONTAMINATED (0-8 s of uncounted random work/rest sits
    inside the delta). Secondary on purpose.
  * `oracle` cells — for each candidate, the ceiling: best (sign, cut) chosen
    ON THE TEST STATS (every midpoint + outer cuts, both signs). This is
    PS.09's oracle-cut defect, priced deliberately: leak = oracle - honest.
    It is not a result; it is what D51's venue question needs a number for.

## THE REFERENCE'S MUST-FAIL CONTROLS, and the lottery cells

Two decoys run the IDENTICAL form on channels the push cannot write with load
content BY CONSTRUCTION: `clock_d` (column 7, d — a pure function of t) and
`clock_c` (column 5, c), both as since-start deltas, i.e. genuine clock
content through the same arithmetic. If a decoy matches or beats the primary
reference's honest cell, the reference is void at this venue.

And THREE LOTTERY CELLS, forced by what the ledger already shows. The
rate-blind control — by construction a function of the per-trip (t0, e0, w0)
jitter draws alone — reads {0.500, 0.875, 0.875}: the class separates from
the JITTER DRAW on two of three seeds. Mechanism hypothesis, stated before
measuring: the class is a deterministic function of trip index j
(`LEG_CYCLE` makes heavy = odd j) while (t0, e0, w0) are pseudo-random
functions of j — a 24-sample correlation lottery between a periodic label and
a hashed jitter. The cells: the raw draws t0, e0, w0 (recomputed exactly as
`_collect` draws them, alignment asserted against each trip's own `heavy`
field), each scored through the same threshold form.

## PRE-REGISTERED FORECAST, written before this script was run once

  1. REPLAY: both multisets reproduce exactly.
  2. PRIMARY (in-bout drain): honest >= 0.75 on at least 2 of 3 seeds and
     worst seed >= 0.625 (mechanism: a limb blocked by mass does less
     |tau*omega| work in the fixed window — attempt 1's own finding, read
     this time through a clean rate). PRE-COMMITTED INTERPRETATION if it
     lands BELOW the registered probe's own worst seed (0.500-0.625 band):
     the drain-rate channel is not demonstrably legible at the LEG venue's
     early window, the clause-(d) coupling cannot be satisfied THERE, and
     that is a venue finding for the row and for D51's calibration-venue
     question — never a reason to move a bar or to swap the channel silently.
  3. ORACLE LEAK <= 0.125 (one 8-trip quantum) on every seed for this
     2-parameter form. A larger leak is the holdout-lottery reading and
     disqualifies these rows as a floor venue regardless of D51's answer.
  4. DECOYS strictly below the primary's honest cell on every seed.
  5. LOTTERY: on the two 0.875-control seeds, at least one of (t0, e0, w0)
     reads >= 0.75 on test through the same form — i.e. the control's
     reading is the fixture's periodic-label-vs-hashed-jitter correlation,
     not interoception. If all three read <= 0.625 there, that mechanism is
     REFUTED and the control's 0.875 needs a different explanation — a
     finding either way, and the second one is stranger.

## WHAT THIS FILE MAY NOT DO

`ACC_MIN` 0.70, `SHUF_ACC_MAX` 0.60, `CONTROL_ACC_MAX` 0.60,
`CONTROL_MARGIN_MIN` 0.10 do not move in either direction. No floor is
declared on any cell below (D51 is open). Nothing here can rescue PS.08's
FAIL: a reference reading the class does not make the registered probe read
it. Recalibrating any floor against these numbers is a THRESHOLD decision and
belongs to the Review and D51's owner.
"""
from __future__ import annotations

import json

import numpy as np

from .. import needs
from ..tests import ps_08_heavy_is_a_price as ps08

SEEDS = (0, 1, 2)
ARTIFACT = "/data/ps08_drainrate_reference_probe.json"

# Derived in the docstring from the live row's mean/std and the 1/16 quantum.
PROBE_MULTISET = (0.500, 0.625, 0.625)
CONTROL_MULTISET = (0.500, 0.875, 0.875)
REPLAY_TOL = 1e-9

# needs.NeedLayer.obs: [e, w, p, T_norm, f, c, i, d, pain]
COL_E, COL_W, COL_C, COL_D = 0, 1, 5, 7


def _pairs(rows: list) -> list:
    """Within-bout (dec4, dec8) row pairs; the rig appends them in order."""
    if len(rows) != 2 * ps08.N_LEG_BOUTS:
        raise RuntimeError(f"leg trip has {len(rows)} rows, not "
                           f"{2 * ps08.N_LEG_BOUTS}")
    return [(rows[2 * k], rows[2 * k + 1]) for k in range(ps08.N_LEG_BOUTS)]


def _stat_inbout(trip_rows: list) -> float:
    return float(np.mean([(a[COL_E] - b[COL_E]) + (a[COL_W] - b[COL_W])
                          for a, b in _pairs(trip_rows)]))


def _stat_sincestart(trip_rows: list, blind_row) -> float:
    return float(np.mean([(blind_row[COL_E] - r[COL_E])
                          + (blind_row[COL_W] - r[COL_W])
                          for r in trip_rows]))


def _stat_clock(trip_rows: list, blind_row, col: int) -> float:
    return float(np.mean([abs(r[col] - blind_row[col]) for r in trip_rows]))


def _honest(s_tr, y_tr, s_te, y_te) -> float:
    """The registered probe's own form (`_speed_score`): threshold = train
    median, sign fit on train, balanced accuracy on the held-out trips.
    2 parameters, both from train."""
    thr = float(np.median(s_tr))
    sign = max((1.0, -1.0),
               key=lambda s: ps08._bal_acc(y_tr, s * (s_tr - thr)))
    return ps08._bal_acc(y_te, sign * (s_te - thr))


def _oracle(s_te, y_te) -> float:
    """The ceiling: best (sign, cut) chosen ON the test stats — every
    midpoint plus outer cuts, both signs. PS.09's oracle-cut defect, run on
    purpose to price the leak. Not a result."""
    xs = np.sort(np.unique(s_te))
    cuts = ([float(xs[0] - 1.0)]
            + [float(0.5 * (xs[i] + xs[i + 1])) for i in range(len(xs) - 1)]
            + [float(xs[-1] + 1.0)])
    return max(ps08._bal_acc(y_te, sign * (s_te - cut))
               for cut in cuts for sign in (1.0, -1.0))


def _cells(stats: np.ndarray, ys: np.ndarray) -> dict:
    s_tr, s_te = stats[:-ps08.N_TEST_TRIPS], stats[-ps08.N_TEST_TRIPS:]
    y_tr, y_te = ys[:-ps08.N_TEST_TRIPS], ys[-ps08.N_TEST_TRIPS:]
    honest = _honest(s_tr, y_tr, s_te, y_te)
    oracle = _oracle(s_te, y_te)
    return {"honest": honest, "oracle": oracle, "leak": oracle - honest,
            "n_params": 2}


def _jitter_draws(seed: int, n: int) -> list:
    """(t0, e0, w0) per leg trip, recomputed exactly as `_collect` draws
    them (the same RandomState seed and the same call order)."""
    out = []
    for j in range(n):
        rng = np.random.RandomState(seed * 6421 + 17 * j + 5)
        t0 = needs.DAY_S + float(rng.uniform(0.0, ps08.LEG_T_JITTER_S))
        e0 = float(rng.uniform(*ps08.LEG_E0))
        w0 = float(rng.uniform(*ps08.LEG_W0))
        out.append((t0, e0, w0))
    return out


def _seed_report(seed: int) -> dict:
    d = ps08._collect(seed)
    if "refused" in d or d.get("no_object") or d.get("bad_lane") \
            or d.get("dead", 0.0) > 0.0:
        return {"seed": seed, "usable": False,
                "why": ("borrow refused" if "refused" in d else
                        "dead events" if d.get("dead", 0.0) > 0.0 else
                        "rig refused")}

    leg = [t for t in d["leg"] if t["rows"]]
    trips = [(t["rows"], 1.0 if t["heavy"] == 1.0 else -1.0) for t in leg]
    blind = [(t["rows_blind"], y) for (_, y), t in zip(trips, leg)]
    ys = np.array([y for _, y in trips])

    # ── the replay: the committed functions on the committed slices ──
    probe_replay = ps08._speed_score(trips, ps08.QVEL_SLICE)
    control_replay = ps08._speed_score(blind, ps08.CLOCK_SLICE)

    # ── the candidate reference and its companions ──
    s_inbout = np.array([_stat_inbout(r) for r, _ in trips])
    s_since = np.array([_stat_sincestart(t["rows"], t["rows_blind"][0])
                        for t in leg])
    s_clk_d = np.array([_stat_clock(t["rows"], t["rows_blind"][0], COL_D)
                        for t in leg])
    s_clk_c = np.array([_stat_clock(t["rows"], t["rows_blind"][0], COL_C)
                        for t in leg])

    # ── the lottery cells: the raw jitter draws, alignment asserted ──
    draws = _jitter_draws(seed, ps08.N_LEG)
    want_y = np.array([1.0 if ps08.LEG_CYCLE[j % len(ps08.LEG_CYCLE)]
                       >= ps08.HEAVY_MIN_IDX else -1.0
                       for j in range(ps08.N_LEG)])
    if len(leg) != ps08.N_LEG or not np.array_equal(want_y, ys):
        raise RuntimeError("leg trip order does not match the fixture's "
                           "j -> class map; lottery cells would misalign")
    s_t0 = np.array([t for t, _, _ in draws])
    s_e0 = np.array([e for _, e, _ in draws])
    s_w0 = np.array([w for _, _, w in draws])

    return {
        "seed": seed, "usable": True, "n_trips": len(trips),
        "probe_replay": probe_replay, "control_replay": control_replay,
        "inbout": _cells(s_inbout, ys),
        "sincestart": _cells(s_since, ys),
        "clock_d": _cells(s_clk_d, ys),
        "clock_c": _cells(s_clk_c, ys),
        "lottery_t0": _cells(s_t0, ys),
        "lottery_e0": _cells(s_e0, ys),
        "lottery_w0": _cells(s_w0, ys),
        "stat_inbout_by_class": {
            "light_mean": float(np.mean(s_inbout[ys < 0])),
            "heavy_mean": float(np.mean(s_inbout[ys > 0]))},
    }


def main() -> None:
    out = [_seed_report(s) for s in SEEDS]
    print("PS.08 DRAIN-RATE REFERENCE PROBE — the honest/oracle pair clause "
          "(d) holds the row on")
    print("=" * 78)

    ok = [r for r in out if r.get("usable")]
    print("\nREPLAY CHECK (must reproduce before anything else is quoted)")
    faithful = len(ok) == len(SEEDS)
    for r in out:
        if not r.get("usable"):
            print(f"  seed {r['seed']}  UNUSABLE — {r['why']}")
            continue
        print(f"  seed {r['seed']}  probe {r['probe_replay']:.4f}  "
              f"control {r['control_replay']:.4f}")
    if ok:
        got_p = sorted(round(r["probe_replay"], 6) for r in ok)
        got_c = sorted(round(r["control_replay"], 6) for r in ok)
        p_ok = all(abs(a - b) <= REPLAY_TOL
                   for a, b in zip(got_p, sorted(PROBE_MULTISET)))
        c_ok = all(abs(a - b) <= REPLAY_TOL
                   for a, b in zip(got_c, sorted(CONTROL_MULTISET)))
        faithful &= p_ok and c_ok
        print(f"  probe multiset   {got_p} vs derived "
              f"{sorted(PROBE_MULTISET)}  "
              f"{'MATCH' if p_ok else 'MISMATCH — DO NOT QUOTE'}")
        print(f"  control multiset {got_c} vs derived "
              f"{sorted(CONTROL_MULTISET)}  "
              f"{'MATCH' if c_ok else 'MISMATCH — DO NOT QUOTE'}")
    print(f"  REPLAY {'FAITHFUL' if faithful else 'BROKEN'}")

    def row(r, key):
        c = r[key]
        return (f"honest {c['honest']:.4f}  oracle {c['oracle']:.4f}  "
                f"leak {c['leak']:+.4f}")

    print("\nTHE REFERENCE — in-bout e/w drain rate (2 named columns, "
          "2 train-fit params)")
    for r in ok:
        print(f"  seed {r['seed']}: {row(r, 'inbout')}   "
              f"[stat light {r['stat_inbout_by_class']['light_mean']:.5f} "
              f"vs heavy {r['stat_inbout_by_class']['heavy_mean']:.5f}]")

    print("\nSECONDARY — since-start e/w delta (pre-roll-contaminated, "
          "reported only)")
    for r in ok:
        print(f"  seed {r['seed']}: {row(r, 'sincestart')}")

    print("\nMUST-FAIL DECOYS — clock channels through the identical form")
    for r in ok:
        print(f"  seed {r['seed']}: d {row(r, 'clock_d')}   |   "
              f"c {row(r, 'clock_c')}")

    print("\nLOTTERY CELLS — the raw jitter draws vs the periodic class")
    for r in ok:
        print(f"  seed {r['seed']}: t0 {r['lottery_t0']['honest']:.4f}  "
              f"e0 {r['lottery_e0']['honest']:.4f}  "
              f"w0 {r['lottery_w0']['honest']:.4f}   (honest cells; "
              f"oracles {r['lottery_t0']['oracle']:.3f}/"
              f"{r['lottery_e0']['oracle']:.3f}/"
              f"{r['lottery_w0']['oracle']:.3f})")

    if ok:
        print("\nWORST SEED (the discipline the family's own gates use)")
        print(f"  inbout honest  min {min(r['inbout']['honest'] for r in ok):.4f}")
        print(f"  inbout leak    max {max(r['inbout']['leak'] for r in ok):+.4f}")
        print(f"  probe replay   min {min(r['probe_replay'] for r in ok):.4f}"
              f"   vs ACC_MIN {ps08.ACC_MIN} (unmoved, not gated here)")

    print("\nNO FLOOR IS DECLARED ON ANY CELL ABOVE — D51 (decide_by "
          "2026-11-03) holds that act.")

    with open(ARTIFACT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nARTIFACT {ARTIFACT}")
    print("\nJSON")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
