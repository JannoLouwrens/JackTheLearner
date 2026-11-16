"""Planted logs for `experiments/pilot_b_report.py` — each of the rule's four
answers, the bars' edges, and the refusals. Run: python -m tests.test_pilot_b_report
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from experiments import pilot_b as P
from experiments import pilot_b_report as R

T = R.T
SUS = dict(decisions=9000, seconds=1800.0, floor_meals=23, apples=0, cause="capped",
           intake=7.13, basal_drain=3.0, sustained=True, reward=2000.0)
STARVE = dict(decisions=2500, seconds=500.0, floor_meals=0, apples=0, cause="energy",
              intake=0.0, basal_drain=0.83, sustained=False, reward=-99.0)
# eats 6 meals, dies anyway: crosses theta = 5, never the sustained mark
SNACK = dict(decisions=4000, seconds=800.0, floor_meals=6, apples=0, cause="energy",
             intake=1.0, basal_drain=1.33, sustained=False, reward=400.0)
# capped but under-fed (the planted disagreement of test_22): not sustained
THIN = dict(SUS, floor_meals=8, intake=2.4, sustained=False)


def rec(arm, seed, head=(), lead=0):
    """`lead` starving lives, then `head`, then starving lives up to exactly T,
    the last one cut off by T ("unfinished")."""
    lives = [dict(STARVE) for _ in range(lead)] + [dict(l) for l in head]
    used = sum(l["decisions"] for l in lives)
    while T - used > STARVE["decisions"]:
        lives.append(dict(STARVE))
        used += STARVE["decisions"]
    lives.append(dict(STARVE, decisions=T - used, cause="unfinished"))
    return {"arm": arm, "seed": seed, "T": T, "rc": 0, "lives": lives}


def book(b_heads=None, floor_heads=None, lead=0):
    """All 35 registered runs; `b_heads`/`floor_heads`: {run_id: head lives}."""
    b_heads, floor_heads = b_heads or {}, floor_heads or {}
    runs = {}
    for arm, seed in P.all_cells():
        k = R.rid(arm, seed)
        runs[k] = rec(arm, seed, {**b_heads, **floor_heads}.get(k, ()), lead)
    return runs


def read(runs):
    return R.reading({k: R.score(v) for k, v in runs.items()})


def b_ids(n):
    return [R.rid("b", s) for s in P.B_SEEDS[:n]]


def test_nobody_eats_reads_does_not_learn():
    r = read(book())
    assert r["reading"] == "DOES NOT LEARN AT THIS BUDGET"
    assert r["n_b"] == {1: 0, 2: 0, 3: 0} and r["floor_gate"] and r["W"] is None


def test_five_of_twenty_is_sustained_and_four_is_not():
    r5 = read(book({k: [SUS] for k in b_ids(5)}))
    assert r5["reading"] == "SUSTAINED" and r5["n_b"][1] == 5 and r5["W"] == 1
    r4 = read(book({k: [SUS] for k in b_ids(4)}))
    assert r4["reading"] == "DOES NOT LEARN AT THIS BUDGET" and r4["n_b"][1] == 4


def test_W_is_the_largest_with_five_and_one_break_resets_it():
    heads = {k: [SUS, SUS, SUS] for k in b_ids(5)}
    heads.update({k: [SUS, SUS, STARVE, SUS] for k in [R.rid("b", s) for s in P.B_SEEDS[5:9]]})
    r = read(book(heads))
    assert r["n_b"] == {1: 9, 2: 9, 3: 5} and r["W"] == 3
    assert r["median_crossing"] == 27000 and r["T_decisive"] == 150_000


def test_late_crossing_keeps_T_at_300k():
    r = read(book({k: [SUS] for k in b_ids(6)}, lead=60))   # crossing at 159,000
    assert r["reading"] == "SUSTAINED" and r["median_crossing"] == 159_000
    assert r["T_decisive"] == 300_000


def test_floor_gate_one_allowed_two_void_even_if_b_sustains():
    b = {k: [SUS] for k in b_ids(10)}
    one = read(book(b, {R.rid("held8", 301): [SUS]}))
    assert one["reading"] == "SUSTAINED" and one["floor_successes"] == 1
    two = read(book(b, {R.rid("held8", 301): [SUS], R.rid("blr0", 303): [SUS]}))
    assert two["reading"] == "VOID" and not two["floor_gate"]
    assert two["floor_by_arm"] == {"noop": 0, "held8": 1, "blr0": 1}


def test_eats_but_starves_needs_five_at_W3():
    r = read(book({k: [SNACK, SNACK, SNACK] for k in b_ids(5)}))
    assert r["reading"] == "EATS BUT STARVES" and r["n_theta"][3] == 5 and r["n_b"][1] == 0
    assert r["W"] is None
    r = read(book({k: [SNACK, SNACK, SNACK] for k in b_ids(4)}))
    assert r["reading"] == "DOES NOT LEARN AT THIS BUDGET"


def test_capped_but_underfed_and_cut_off_lives_do_not_cross():
    # seed 315's shape: eating at 3x drain when T cuts the life -> unfinished, never sustained
    runs = book({k: [THIN] for k in b_ids(20)})
    for k in b_ids(20):
        last = runs[k]["lives"][-1]
        last.update(floor_meals=24, intake=7.44, basal_drain=2.31, sustained=True)
    r = read(runs)
    assert r["n_b"][1] == 0 and r["reading"] != "SUSTAINED"
    assert R.run_stats(runs[b_ids(1)[0]])["unfinished"][1] == 24


def test_missing_or_extra_run_gives_no_reading():
    runs = book({k: [SUS] for k in b_ids(20)})
    short = dict(runs)
    del short[R.rid("blr0", 305)]
    for bad in (short, {**runs, "b_s999": rec("b", 999)}):
        try:
            read(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a reading was given on books that do not balance")
    md, complete = R.render(short, {k: "x.json" for k in short}, [])
    assert not complete and "PARTIAL" in md and "Reading by the rule" not in md
    md, complete = R.render(runs, {k: "x.json" for k in runs}, ["prereg moved"])
    assert not complete and "Reading by the rule" not in md and "prereg moved" in md
    md, complete = R.render(runs, {k: "x.json" for k in runs}, [])
    assert complete and "Reading by the rule: **SUSTAINED**" in md


def test_a_crashed_cell_is_an_outcome_not_a_hole():
    runs = book({k: [SUS] for k in b_ids(5)})
    k = b_ids(1)[0]
    runs[k] = {"arm": "b", "seed": 301, "lives": [], "rc": 1, "error": "rc=1: boom"}
    r = read(runs)                      # still 35 runs: the crash is counted, against B
    assert r["n_b"][1] == 4 and r["reading"] == "DOES NOT LEARN AT THIS BUDGET"
    assert R.crashed(runs[k]) and "**CRASH**" in R.render(runs, {}, [])[0]
    short = rec("b", 302)
    short["lives"] = short["lives"][:10]          # a cell that stopped early, rc 0
    assert R.crashed(short) and R.lives_for(short, "sustained")[-1] == "crash"


def test_a_run_in_two_kernel_files_is_refused():
    with tempfile.TemporaryDirectory() as d:
        cell = {"cells": {"b_s301": rec("b", 301)}}
        (Path(d) / "k0.json").write_text(json.dumps(cell))
        assert set(R.load(d)[0]) == {"b_s301"}
        (Path(d) / "k1.json").write_text(json.dumps(cell))
        try:
            R.load(d)
        except ValueError:
            pass
        else:
            raise AssertionError("the same run id from two files must refuse")


def test_frozen_checks_name_missing_runs_and_pass_full_books():
    full = book()
    assert R.frozen_checks(full) == [], R.frozen_checks(full)
    short = dict(full)
    del short[R.rid("blr0", 305)]
    bad = R.frozen_checks(short)
    assert len(bad) == 2 and all("blr0_s305" in b for b in bad)


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # report every failure, then exit 1
                fails += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if fails else 0)
