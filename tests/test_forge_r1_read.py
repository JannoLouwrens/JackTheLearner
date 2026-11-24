"""Planted tests for round 1's reader (F3 / ORDERS J18): raw kernel files -> rows.

Every expected number is worked by hand from the frozen rule's words (late
half, sustained = capped AND intake >= drain, crash/cut = 0), not read back
from the code. Written and run BEFORE the judge lock and before any raw file
of the round was opened.
Run: /data/venvs/jackthelearner/bin/python -m tests.test_forge_r1_read
"""
import copy
import json
import subprocess as sp
import tempfile
from pathlib import Path

from experiments.forge import Refused, climb
from experiments.forge import r1_read as R
from experiments.forge import r1_run
from experiments.forge.metric import metric_hash

SPEC = json.loads((R.ROUND / "spec.json").read_text())
SHA = "s" * 64
PIN = "b" * 40
T = SPEC["rungs"]["S"]["decisions"]            # 150,000
SEEDS = SPEC["rungs"]["S"]["seeds"]
ARMS = [SPEC["baseline"], *SPEC["twins"], *[c["id"] for c in SPEC["candidates"]]]
WORLD = {"difficulty": 0.75, "reward": "dr100-eat", "random_layout": True,
         "lazy_contact_audio": True, "spawn_clear": 1.55, "move_cost_dial": True,
         "native_substep": True, "eye": "rays16", "food_floor_only": True,
         "eye_frame": "body", "motor_frame": "body"}


def raises(fn, text=""):
    try:
        fn()
    except Refused as e:
        assert text in str(e), f"refused for another reason: {e}"
        return True
    raise AssertionError("was not refused")


def life(decisions, cause="starved", intake=0.0, drain=1.0, meals=0):
    return {"decisions": decisions, "seconds": decisions / 5, "floor_meals": meals, "apples": 0,
            "cause": cause, "intake": intake, "basal_drain": drain,
            "sustained": cause == "capped" and intake >= drain, "reward": 0.0}


FED = dict(cause="capped", intake=3.0, drain=1.5, meals=6)        # a sustained life
LONG = dict(cause="capped", intake=1.0, drain=1.5, meals=2)       # long, under-fed
# 16 lives of 9,000 = 144,000, + 6,016 unfinished = 150,016 (> T: the tail is not counted).
# Late lives END after 75,000: lives 9..16 = 8 late lives.
NONE = [life(9000, "starved")] * 16
EARLY = [life(9000, **FED)] * 8 + [life(9000, "starved")] * 8       # ate only in the early half
HALF = [life(9000, "starved")] * 8 + [life(9000, **FED)] * 4 + [life(9000, **LONG)] * 4
ALLW = [life(9000, "starved")] * 8 + [life(9000, **FED)] * 8


def rec(arm, seed, lives=NONE, **over):
    r = {"arm": arm, "seed": seed, "T": T, "decisions": T + 16, "wall_s": 9000.0, "rc": 0,
         "world": dict(WORLD),
         "forge": {"round": "r1", "arm": arm, "rung": "S", "code_head": PIN,
                   "settings": R.rule_settings(SPEC, arm)},
         "lives": list(lives) + [life(6016, "unfinished", 9.0, 1.0, 9)]}
    r.update(over)
    return r


def files_of(lives_by_arm=None, per=4, dur=7200.0):
    """All 30 cells of rung S, seed-major, `per` to a kernel file."""
    lives_by_arm = lives_by_arm or {}
    cells = [(a, s) for s in SEEDS for a in ARMS]
    out = {}
    for j in range(0, len(cells), per):
        out[f"S_g{j // per}.json"] = {
            "round": "r1", "rung": "S", "pin": PIN, "spec_sha256": SHA,
            "kernel": {"duration_s": dur, "charge_seconds": 0.0, "job_id": "x"},
            "cells": {f"{a}_s{s}": rec(a, s, lives_by_arm.get(a, NONE))
                      for a, s in cells[j:j + per]}}
    return out


def score(files, prior=()):
    return R.score_rung(SPEC, "S", files, list(prior), spec_sha=SHA)


def by(rows, arm):
    return [r for r in rows if r["arm"] == arm]


def test_planted_eater_scores_by_hand():
    rows = score(files_of({"K2": HALF, "K1": ALLW}))
    assert len(rows) == 30 and {r["status"] for r in rows} == {"ok"}
    for r in by(rows, "K2"):             # 4 sustained of 8 late lives; meals (4*6 + 4*2) / 8 = 4
        assert (r["score"], r["late_lives"], r["late_sustained"], r["tiebreak"]) == (0.5, 8, 4, 4.0)
    for r in by(rows, "K1"):             # 8 of 8; 6 meals a life
        assert (r["score"], r["tiebreak"]) == (1.0, 6.0)
    assert all(r["score"] == 0.0 and r["tiebreak"] == 0.0 for r in by(rows, "E-food"))
    assert all(r["metric_sha256"] == SPEC["outcome"]["metric_sha256"] and r["code_head"] == PIN
               for r in rows)


def test_long_life_without_eating_and_early_eating_score_zero():
    rows = score(files_of({"K3": [life(9000, **LONG)] * 16, "K5": EARLY}))
    assert all(r["score"] == 0.0 for r in by(rows, "K3"))       # capped every life, never fed enough
    assert all(r["tiebreak"] == 2.0 for r in by(rows, "K3"))    # the tiebreak still sees its meals
    assert all(r["score"] == 0.0 and r["tiebreak"] == 0.0 for r in by(rows, "K5"))


def test_unfinished_tail_is_never_counted():
    # the unfinished life carries 9 meals and intake 9 > drain 1; it must add nothing
    rows = score(files_of({"N3": ALLW}))
    assert all(r["late_lives"] == 8 and r["tiebreak"] == 6.0 for r in by(rows, "N3"))


def test_crash_cut_missing_and_short_runs_score_zero_and_stay_in():
    f = files_of({"K6": ALLW})
    g = f["S_g1.json"]["cells"]                  # g1 = K3 K5 K6 N3 on seed 1401
    assert "K6_s1401" in g
    g["K6_s1401"] = {"arm": "K6", "seed": 1401, "lives": ALLW, "cut": True,
                     "decisions_reached": 144000, "from_partial": True, "rc": -9}
    f["S_g4.json"]["cells"]["K6_s1402"]["error"] = "rc=1: boom"
    f["S_g4.json"]["cells"]["K6_s1402"]["rc"] = 1
    f["S_g6.json"]["cells"]["K6_s1403"]["decisions"] = T - 32
    f["S_g6.json"]["cells"]["N3_s1403"] = {"arm": "N3", "seed": 1403, "lives": [], "not_found": True}
    rows = score(f)
    assert [(r["status"], r["score"]) for r in by(rows, "K6")] == [("cut", 0.0), ("crash", 0.0), ("cut", 0.0)]
    assert [r["status"] for r in by(rows, "N3")] == ["ok", "ok", "missing"]
    nan = files_of()
    nan["S_g0.json"]["cells"]["K1_s1401"]["lives"][3] = life(float("nan"))
    assert by(score(nan), "K1")[0]["status"] == "crash"
    got = climb(SPEC, rows, "S")                 # 2 runs lost: 0 mean, and K6 does not climb
    assert got.means["K6"] == 0.0 and "K6" not in got.climbers


def test_a_kernel_where_nothing_finished_is_refused_not_scored():
    f = files_of()
    for k, r in f["S_g7.json"]["cells"].items():
        f["S_g7.json"]["cells"][k] = {"arm": r["arm"], "seed": r["seed"], "lives": [], "not_found": True}
    raises(lambda: score(f), "cut kernel")


def test_cells_must_be_exactly_the_rules():
    f = files_of()
    del f["S_g7.json"]                                           # the half-size kernel absent
    raises(lambda: score(f), "runs missing")
    f = files_of()
    del f["S_g2.json"]["cells"]["C1_s1401"]
    raises(lambda: score(f), "runs missing")
    f = files_of()
    f["S_g9.json"] = copy.deepcopy(f["S_g0.json"])
    raises(lambda: score(f), "two kernel files")
    f = files_of()
    f["S_g0.json"]["cells"]["K9_s1401"] = {**rec("K1", 1401), "arm": "K9"}
    raises(lambda: score(f), "not a cell")
    f = files_of()
    f["S_g0.json"]["cells"]["K1_s1411"] = rec("K1", 1411)        # a rung-M seed
    raises(lambda: score(f), "not a cell")
    raises(lambda: score({}), "no kernel file")


def test_one_commit_one_spec_one_rung():
    f = files_of()
    f["S_g3.json"]["pin"] = "c" * 40
    raises(lambda: score(f), "one pinned commit")
    f = files_of()
    f["S_g3.json"]["cells"]["K1_s1402"]["forge"]["code_head"] = "c" * 40
    raises(lambda: score(f), "not the pin")
    f = files_of()
    f["S_g3.json"]["spec_sha256"] = "t" * 64
    raises(lambda: score(f), "another spec")
    f = files_of()
    f["S_g3.json"]["rung"] = "M"
    raises(lambda: score(f), "not round r1 rung S")
    f = files_of()
    f["S_g3.json"]["cells"]["K1_s1402"]["forge"]["rung"] = None  # a smoke run
    raises(lambda: score(f), "not a rung-S run")
    f = files_of()
    del f["S_g3.json"]["kernel"]["duration_s"]
    raises(lambda: score(f), "no kernel duration")


def test_a_run_with_other_settings_or_another_world_is_refused():
    f = files_of()
    f["S_g1.json"]["cells"]["K5_s1401"]["forge"]["settings"]["gradient_steps"] = 32  # = the baseline
    raises(lambda: score(f), "rule's settings")
    f = files_of()
    f["S_g1.json"]["cells"]["K5_s1401"]["world"]["difficulty"] = 0.5
    raises(lambda: score(f), "another world")
    f = files_of()
    f["S_g1.json"]["cells"]["K5_s1401"]["world"]["eye_frame"] = "world"
    raises(lambda: score(f), "another world")


def test_rule_settings_are_the_kernels_for_every_arm():
    for a in ARMS:
        assert R.rule_settings(SPEC, a) == r1_run.arm_settings(SPEC, a), a
    assert R.rule_settings(SPEC, "K5")["gradient_steps"] == 3
    assert R.rule_settings(SPEC, "E-food-lr0")["lr"] == 0.0
    assert R.rule_settings(SPEC, "K1")["param_resets"]["S"] == [50000, 100000]


def test_a_rung_is_read_once_and_the_metric_is_the_frozen_one():
    rows = score(files_of())
    raises(lambda: score(files_of(), prior=rows), "read once")
    raises(lambda: R.score_rung(SPEC, "S", files_of(), [], spec_sha=SHA, code_hash="0" * 64),
           "frozen hash")
    assert metric_hash() == SPEC["outcome"]["metric_sha256"]


def test_cost_is_the_kernels_duration():
    rows = score(files_of(dur=7200.0))                    # 8 kernels x 2 h
    assert abs(sum(r["kernel_hours"] for r in rows) - 16.0) < 1e-6
    assert abs(sum(r["core_hours"] for r in rows) - 60.0) < 1e-6     # 30 runs x 2 h each
    g7 = [r for r in rows if r["kernel"] == "S_g7.json"]
    assert len(g7) == 2 and all(r["kernel_hours"] == 1.0 for r in g7)


def test_rows_go_through_the_ladder_and_the_tiebreak_only_orders_equals():
    # K2 0.5, K1 0.5 (more meals), K3 0.5 (more meals still), K5 0.5 fewest meals: 8 in -> 3 climb
    more = [life(9000, "starved")] * 8 + [life(9000, **FED)] * 4 + [life(9000, "capped", 1.0, 1.5, 4)] * 4
    most = [life(9000, "starved")] * 8 + [life(9000, **FED)] * 4 + [life(9000, "capped", 1.0, 1.5, 5)] * 4
    few = [life(9000, "starved")] * 8 + [life(9000, **FED)] * 4 + [life(9000, "starved")] * 4
    rows = score(files_of({"K2": HALF, "K1": more, "K3": most, "K5": few}))
    got = climb(SPEC, rows, "S")
    assert (got.tiebreaks["K5"], got.tiebreaks["K2"], got.tiebreaks["K1"], got.tiebreaks["K3"]) == (3.0, 4.0, 5.0, 5.5)
    assert sorted(got.climbers) == ["K1", "K2", "K3"] and "tiebreak" in got.reasons["K5"]
    # the baseline eating MORE meals at an equal score still blocks everyone: score alone
    rows = score(files_of({"K2": HALF, "E-food": HALF}))
    assert climb(SPEC, rows, "S").climbers == []
    assert R.rung_arms(SPEC, [], "S") == ARMS


def test_read_refuses_without_the_lock_before_opening_a_raw_file():
    with tempfile.TemporaryDirectory() as t:
        rd, raw = Path(t) / "experiments/forge/rounds/r1", Path(t) / "raw"
        rd.mkdir(parents=True)
        raw.mkdir()
        sp.run(["git", "init", "-q", t], check=True)
        (rd / "spec.json").write_text(json.dumps(SPEC))
        (raw / "S_g0.json").write_text("{ not json: opening this file raises ValueError")
        raises(lambda: R.read("S", rd, raw), "no judge lock")
        assert not (rd / "results.jsonl").exists()


if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f()
            print("PASS", n)
        except Exception as e:   # noqa: BLE001
            bad += 1
            print("FAIL", n, repr(e))
    print(f"{len(tests) - bad}/{len(tests)} passed")
    sys.exit(1 if bad else 0)
