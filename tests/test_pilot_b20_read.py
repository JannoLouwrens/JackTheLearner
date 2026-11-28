"""pilot_b20_read: refuses a draft / missing pin; exact Boschloo equals scipy's;
planted HOLDS, NOT HELD (the 15-of-20 line and the strict > 2 x B bar), a lost
seed (refused until declared, then counted not-above), a wrong-pin file.

    python tests/test_pilot_b20_read.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import pilot_b20_read as R  # noqa: E402

PIN = "f" * 40


def refuses(fn, needle):
    try:
        fn()
    except R.Refused as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"did not refuse ({needle})")


def cell(meals: int) -> dict:
    """20 lives of 50k decisions, `meals` each (late = lives starting >= 500k)."""
    lives = [{"decisions": 50_000, "cause": "energy", "floor_meals": meals, "apples": 0}
             for _ in range(20)]
    return {"decisions": R.T, "lives": lives}


def plant(d: Path, sighted: dict, blind: int = 3, drop=(), pin=PIN):
    for unit, cl in R.pack().items():
        cells = {}
        for arm, s in cl:
            if f"{arm}_s{s}" in drop:
                continue
            cells[f"{arm}_s{s}"] = cell(blind if arm == R.BLIND else sighted[s])
        (d / f"b20_s{unit}.json").write_text(json.dumps({"pin": pin, "cells": cells}))


def sighted_with(n_above: int, at_bar=0) -> dict:
    """bar = 2 x 3 = 6: n_above seeds eat 10, `at_bar` eat exactly 6 (NOT above), rest 2."""
    seeds = list(R.SEEDS)
    out = {s: 2 for s in seeds}
    for s in seeds[:n_above]:
        out[s] = 10
    for s in seeds[n_above:n_above + at_bar]:
        out[s] = 6
    return out


def test_refuses_a_draft_and_a_bad_pin():
    refuses(lambda: R.main([]), "needs --pin")
    refuses(lambda: R.check_pin("521f15eb"), "not FROZEN")   # the reader's own commit: still a DRAFT (HEAD freezes)
    refuses(lambda: R.check_pin("e797d367"), "not FROZEN")   # the real draft commit
    refuses(lambda: R.check_pin("deadbeef" * 5), "not a commit")


def test_boschloo_equals_scipy():
    import numpy as np
    from scipy.stats import binom, boschloo_exact
    P = R.boschloo_two_sided(15)
    assert abs(P[4, 0] - 0.043) < 5e-4, P[4, 0]          # harness check #12's known case
    for a, b in ((7, 2), (10, 9), (15, 3), (1, 0)):
        s = boschloo_exact([[a, b], [15 - a, 15 - b]]).pvalue
        assert abs(P[a, b] - s) < 1e-5, (a, b, P[a, b], s)
    n, pa, pb = 8, 0.9, 0.2                                # power by scipy table by table
    want = sum(binom.pmf(a, n, pa) * binom.pmf(b, n, pb)
               for a in range(n + 1) for b in range(n + 1)
               if 0 < a + b < 2 * n
               and boschloo_exact([[a, b], [n - a, n - b]]).pvalue <= R.ALPHA)
    assert abs(R.power(n, pa, pb) - want) < 1e-6, (R.power(n, pa, pb), want)
    assert R.n_needed(0.5, 0.5) is None and np.isfinite(R.power(5, 0.5, 0.5))


def test_planted_holds_and_not_held():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        plant(d, sighted_with(15))
        r = R.read(R.load(PIN, root=d))
        assert r["verdict"] == "HOLDS" and r["k"] == 15 and r["bar"] == 6.0, r["k"]
        assert r["blind_mean"] == 3.0 and r["blind_n"] == 5
        assert r["windows"][f"{R.SIGHTED}_s2001"] == [10.0] * 20
        assert r["n_table"]["0.3"]["n_per_arm"] == R.n_needed(0.75, 0.45)
        plant(d, sighted_with(14, at_bar=1))                # 14 above + one exactly AT 2 x B
        r = R.read(R.load(PIN, root=d))
        assert r["verdict"] == "NOT HELD AT 1M" and r["k"] == 14, r["k"]
        hold = R.HOLD_NEED                                  # mutant: a 14-of-20 line passes it
        R.HOLD_NEED = 14
        try:
            assert R.read(R.load(PIN, root=d))["verdict"] == "HOLDS"
        finally:
            R.HOLD_NEED = hold


def test_lost_seed_and_wrong_pin():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        gone = f"{R.SIGHTED}_s2001"                          # one of the 15 eaters
        plant(d, sighted_with(15), drop=(gone,))
        refuses(lambda: R.load(PIN, root=d), "missing")
        r = R.read(R.load(PIN, lost=[gone], root=d))
        assert r["lost"] == [gone] and r["late_meals"][gone] == "LOST"
        assert r["k"] == 14 and r["verdict"] == "NOT HELD AT 1M"   # lost = not above
        assert r["sighted_spread"]["n"] == 19
        refuses(lambda: R.load(PIN, lost=[gone, f"{R.SIGHTED}_s2002"], root=d), "home whole")
        refuses(lambda: R.load(PIN, lost=["B20-bins16_s1999"], root=d), "no registered run")
        plant(d, sighted_with(15), pin="0" * 40)
        refuses(lambda: R.load(PIN, root=d), "not the freeze")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
