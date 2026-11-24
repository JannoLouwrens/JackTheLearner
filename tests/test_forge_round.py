"""Planted tests for the forge's round runner and scoreboard (docs/FORGE.md, F1).

Every expected value is written from the charter's rule text, not read back
from the code; each test can fail.
Run: /data/venvs/jackthelearner/bin/python -m tests.test_forge_round
"""
import copy
import json
import math
import tempfile
from pathlib import Path

from experiments.forge import Refused, climb, load_round, source_yield, validate_spec
from experiments.forge.metric import metric_hash, sustained_rate
from scripts import forge_scoreboard

H = "a" * 64          # the planted frozen metric hash
S_SEEDS, M_SEEDS = [1, 2, 3], [11, 12, 13, 14, 15]


def raises(fn, text=""):
    try:
        fn()
    except Refused as e:
        assert text in str(e), f"refused for another reason: {e}"
        return True
    raise AssertionError("was not refused")


def cand(i, source="published"):
    return {"id": i, "source": source, "prior": "§19 planted", "twin": "lr0"}


def spec_of(cands, budget=(100, 100)):
    return {"round": "t", "question": "planted", "baseline": "base", "twins": ["lr0"],
            "candidates": cands,
            "rungs": {"S": {"decisions": 50_000, "seeds": S_SEEDS},
                      "M": {"decisions": 300_000, "seeds": M_SEEDS},
                      "L": {"decisions": 1_000_000, "seeds": list(range(21, 31))}},
            "outcome": {"metric": "sustained_rate", "requires_eating": True, "metric_sha256": H},
            "budget": {"core_hours": budget[0], "kernel_hours": budget[1]}}


def rows_of(scores, rung="S", cost=0.1, h=H):
    """scores: {arm: number or [per-seed numbers]}"""
    seeds = S_SEEDS if rung == "S" else M_SEEDS
    out = []
    for arm, v in scores.items():
        vs = v if isinstance(v, list) else [v] * len(seeds)
        out += [dict(arm=arm, rung=rung, seed=s, score=x, core_hours=cost, kernel_hours=cost,
                     metric_sha256=h) for s, x in zip(seeds, vs)]
    return out


BASE = {"base": 0.10, "lr0": 0.05}
THREE = [cand("win"), cand("null", "inventor"), cand("ctl", "control")]


def run(spec, rows, rung="S"):
    return climb(spec, rows, rung, code_hash=H)


# ---- the five planted cases of order F1 -------------------------------------
def test_planted_winner_climbs():
    got = run(spec_of(THREE), rows_of({**BASE, "win": 0.6, "null": 0.1, "ctl": 0.1}))
    assert got.status == "OK" and got.climbers == ["win"]
    assert math.isclose(got.means["win"], 0.6) and math.isclose(got.control_mean, 0.1)


def test_planted_null_does_not_climb():
    # every candidate sits on the baseline: nobody climbs, whatever the top third is
    got = run(spec_of(THREE), rows_of({**BASE, "win": 0.1, "null": 0.1, "ctl": 0.1}))
    assert got.climbers == []
    # and a null beside a real winner stays down
    got = run(spec_of(THREE), rows_of({**BASE, "win": 0.6, "null": 0.1, "ctl": 0.1}))
    assert "null" not in got.climbers and got.reasons["null"] != "CLIMBS"


def test_candidate_equal_to_baseline_does_not_climb():
    # top of the rung, above its twin and above plain variation — and exactly the baseline
    sc = {"base": 0.3, "lr0": 0.05, "win": 0.3, "null": 0.1, "ctl": 0.1}
    got = run(spec_of(THREE), rows_of(sc))
    assert got.climbers == [] and "baseline" in got.reasons["win"]
    assert run(spec_of(THREE), rows_of({**sc, "win": 0.31})).climbers == ["win"]


def test_candidate_equal_to_control_does_not_climb():
    # 'win' beats the baseline and its twin, and is alone in the top third —
    # but plain variation did exactly as well, so it has shown nothing.
    rows = rows_of({**BASE, "win": 0.4, "ctl": 0.4, "ctl2": 0.4, "x1": 0.0, "x2": 0.0, "x3": 0.0})
    cands = [cand("win"), cand("ctl", "control"), cand("ctl2", "control"),
             cand("x1"), cand("x2"), cand("x3")]
    got = run(spec_of(cands), rows)
    assert "win" not in got.climbers
    # one notch above plain variation and it climbs (the clause is not always-false)
    rows = rows_of({**BASE, "win": 0.5, "ctl": 0.4, "ctl2": 0.4, "x1": 0.0, "x2": 0.0, "x3": 0.0})
    assert "win" in run(spec_of(cands), rows).climbers


def test_round_over_budget_stops():
    rows = rows_of({**BASE, "win": 0.6, "null": 0.1, "ctl": 0.1}, cost=1.0)   # 15 rows = 15 h
    got = run(spec_of(THREE, budget=(14.9, 100)), rows)
    assert got.status == "STOPPED" and got.climbers == []
    got = run(spec_of(THREE, budget=(100, 14.9)), rows)                       # kernel line too
    assert got.status == "STOPPED" and got.climbers == []
    assert run(spec_of(THREE, budget=(15, 15)), rows).climbers == ["win"]      # exactly on budget runs


def test_other_metric_hash_is_refused():
    rows = rows_of({**BASE, "win": 0.6, "null": 0.1, "ctl": 0.1})
    rows[-1]["metric_sha256"] = "b" * 64                  # one row scored by other code
    raises(lambda: run(spec_of(THREE), rows), "other metric code")
    good = rows_of({**BASE, "win": 0.6, "null": 0.1, "ctl": 0.1})
    raises(lambda: climb(spec_of(THREE), good, "S", code_hash="c" * 64), "on disk differs")
    # the real code on disk is not the planted hash either
    raises(lambda: climb(spec_of(THREE), good, "S"), "on disk differs")


# ---- the rest of the rule ---------------------------------------------------
def test_must_beat_never_trains_twin():
    got = run(spec_of(THREE), rows_of({"base": 0.1, "lr0": 0.7, "win": 0.6, "null": 0.1, "ctl": 0.1}))
    assert got.climbers == [] and "twin" in got.reasons["win"]


def test_top_third_and_tie_across_the_cut():
    ids = ["a", "b", "c", "d", "e", "f"]
    cands = [cand(i) for i in ids] + [cand("ctl", "control")]          # 7 in -> ceil(7/3) = 3
    sc = {**BASE, "a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5, "f": 0.4, "ctl": 0.2}
    assert run(spec_of(cands), rows_of(sc)).climbers == ["a", "b", "c"]
    sc["d"] = 0.7                                                     # c ties the first arm below the cut
    assert run(spec_of(cands), rows_of(sc)).climbers == ["a", "b"]


def test_tiebreak_orders_equal_scores_only_and_never_beats_the_baseline():
    ids = ["a", "b", "c", "d"]
    cands = [cand(i) for i in ids] + [cand("ctl", "control")]          # 5 in -> ceil(5/3) = 2
    sc = {**BASE, "a": 0.9, "b": 0.7, "c": 0.7, "d": 0.2, "ctl": 0.2}

    def rows(tb):
        out = rows_of(sc)
        for r in out:
            r["tiebreak"] = tb.get(r["arm"], 0.0)
        return out
    assert run(spec_of(cands), rows({})).climbers == ["a"]              # b, c tied across the cut
    got = run(spec_of(cands), rows({"b": 1.0, "c": 3.0}))
    assert got.climbers == ["a", "c"] and "tiebreak" in got.reasons["b"]
    assert run(spec_of(cands), rows({"b": 3.0, "c": 3.0})).climbers == ["a"]   # tied on both: drops
    # a lower score with a huge tiebreak is still a lower score
    assert run(spec_of(cands), rows({"d": 99.0})).climbers == ["a"]
    # equal to the baseline / twin / plain variation on the score: no tiebreak buys the climb
    eq = {"base": 0.5, "lr0": 0.05, "win": 0.5, "null": 0.1, "ctl": 0.1}
    r3 = rows_of(eq)
    for r in r3:
        r["tiebreak"] = 9.0 if r["arm"] == "win" else 0.0
    assert run(spec_of(THREE), r3).climbers == []
    eq = {**BASE, "win": 0.4, "null": 0.1, "ctl": 0.4}
    r3 = rows_of(eq)
    for r in r3:
        r["tiebreak"] = 9.0 if r["arm"] == "win" else 0.0
    assert "plain variation" in run(spec_of(THREE), r3).reasons["win"]
    r3[0]["tiebreak"] = float("nan")                                   # a bad tiebreak is 0, not a crash
    run(spec_of(THREE), r3)


def test_control_candidate_can_climb():
    got = run(spec_of(THREE), rows_of({**BASE, "win": 0.1, "null": 0.1, "ctl": 0.5}))
    assert got.climbers == ["ctl"]


def test_crash_counts_against_and_missing_run_is_incomplete():
    rows = rows_of({**BASE, "win": [0.9, float("nan"), None], "null": 0.1, "ctl": 0.1})
    got = run(spec_of(THREE), rows)
    assert math.isclose(got.means["win"], 0.3)             # 0.9 + 0 + 0 over 3 seeds
    rows = [r for r in rows_of({**BASE, "win": 0.9, "null": 0.1, "ctl": 0.1})
            if not (r["arm"] == "win" and r["seed"] == 3)]
    got = run(spec_of(THREE), rows)
    assert got.climbers == [] and got.reasons["win"].startswith("INCOMPLETE")
    rows = [r for r in rows_of({**BASE, "win": 0.9, "null": 0.1, "ctl": 0.1})
            if not (r["arm"] == "base" and r["seed"] == 3)]
    raises(lambda: run(spec_of(THREE), rows), "baseline")


def test_duplicate_unknown_and_unnamed_seed_refused():
    rows = rows_of({**BASE, "win": 0.6, "null": 0.1, "ctl": 0.1})
    raises(lambda: run(spec_of(THREE), rows + [rows[0]]), "duplicate")
    raises(lambda: run(spec_of(THREE), rows + [dict(rows[0], arm="ghost")]), "never named")
    raises(lambda: run(spec_of(THREE), rows + [dict(rows[0], seed=99)]), "seed the rule never named")


def test_medium_rung_takes_only_climbers_and_at_most_two_go_long():
    ids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    cands = [cand(i) for i in ids] + [cand("ctl", "control")]          # 9 in -> 3 climb
    s = rows_of({**BASE, **{i: 0.9 - 0.1 * n for n, i in enumerate(ids)}, "ctl": 0.05})
    spec = spec_of(cands)
    assert run(spec, s).climbers == ["a", "b", "c"]
    m = rows_of({**BASE, "a": 0.5, "b": 0.6, "c": 0.7}, rung="M")
    got = run(spec, s + m, "M")
    assert got.entrants == ["a", "b", "c"]
    assert got.yardstick == "ctl" and got.climbers == [], "the control yardstick has not run at M"
    assert got.reasons["c"].startswith("INCOMPLETE")
    got = run(spec, s + m + rows_of({"ctl": 0.65}, rung="M"), "M")
    assert got.climbers == ["c"] and math.isclose(got.control_mean, 0.65)   # ceil(3/3) = 1
    got = run(spec, s + m + rows_of({"ctl": 0.7}, rung="M"), "M")
    assert got.climbers == [], "equal to the yardstick does not go long; the yardstick cannot climb"
    # plain variation ahead of 'b': the control climbs, 'b' does not, and no yardstick is needed
    s2 = rows_of({**BASE, **{i: 0.9 - 0.1 * n for n, i in enumerate(ids)}, "ctl": 0.85})
    assert run(spec, s2).climbers == ["a", "ctl"]
    m2 = rows_of({**BASE, "ctl": 0.2, "a": 0.5}, rung="M")
    got = run(spec, s2 + m2, "M")
    assert got.yardstick is None and got.climbers == ["a"]            # ceil(2/3) = 1 goes long
    raises(lambda: run(spec, s2 + m2, "L"), "not a climbing rung")


def test_at_most_two_go_long():
    ids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ctl = [f"k{i}" for i in range(11)]                                # 19 in -> ceil(19/3) = 7 climb
    spec = spec_of([cand(i) for i in ids] + [cand(k, "control") for k in ctl])
    s = rows_of({**BASE, **{i: 0.9 - 0.01 * n for n, i in enumerate(ids)}, **{k: 0.01 for k in ctl}})
    assert run(spec, s).climbers == ids[:7]
    m = rows_of({**BASE, **{i: 0.9 - 0.01 * n for n, i in enumerate(ids[:7])}, "k0": 0.2}, rung="M")
    got = run(spec, s + m, "M")
    assert got.yardstick == "k0" and got.climbers == ["a", "b"]       # ceil(7/3) = 3, capped at 2


def test_spec_rules():
    ok = spec_of(THREE)
    validate_spec(ok)

    def broken(edit, text):
        s = copy.deepcopy(ok)
        edit(s)
        raises(lambda: validate_spec(s), text)

    broken(lambda s: s["candidates"][0].update(prior=" "), "no prior")
    broken(lambda s: s["candidates"][0].update(source="guess"), "source")
    broken(lambda s: s["candidates"].pop(), "no control-source")
    broken(lambda s: s["outcome"].update(requires_eating=False), "require eating")
    broken(lambda s: s["outcome"].pop("metric_sha256"), "metric_sha256")
    broken(lambda s: s["budget"].pop("kernel_hours"), "budget.kernel_hours")
    broken(lambda s: s["rungs"]["M"].update(seeds=[1, 12, 13]), "reused across rungs")
    broken(lambda s: s["candidates"].extend(cand(f"n{i}") for i in range(8)), "more than 8")
    eight = spec_of([cand(f"n{i}") for i in range(8)] + [cand("ctl", "control")])
    validate_spec(eight)                                   # 8 new + the control is allowed


def test_source_yield_names_the_source_that_does_not_pay():
    y = source_yield(spec_of(THREE), rows_of({**BASE, "win": 0.6, "null": 0.1, "ctl": 0.1}),
                     code_hash=H)["sources"]
    assert y["published"] == {"in": 1, "M": 1, "L": 0, "paying": True}
    assert y["inventor"] == {"in": 1, "M": 0, "L": 0, "paying": False}
    assert y["control"]["M"] == 0
    assert source_yield(spec_of(THREE), [], code_hash=H)["sources"]["published"]["paying"] is None


# ---- the metric requires eating ---------------------------------------------
def test_metric_long_life_without_eating_scores_zero():
    starving = [(1000, True, 0.0, 5.0)] * 10               # capped lives, no intake
    fed = [(1000, True, 6.0, 5.0)] * 10
    died_fed = [(1000, False, 6.0, 5.0)] * 10              # ate but died early
    assert sustained_rate(starving, 10_000) == 0.0
    assert sustained_rate(fed, 10_000) == 1.0
    assert sustained_rate(died_fed, 10_000) == 0.0


def test_metric_counts_late_lives_only_and_crash_is_zero():
    early_good = [(1000, True, 6.0, 5.0)] * 5 + [(1000, True, 0.0, 5.0)] * 5
    assert sustained_rate(early_good, 10_000) == 0.0       # lives ending after 5,000: all starved
    late_good = early_good[::-1]
    assert sustained_rate(late_good, 10_000) == 1.0
    mixed = [(1000, True, 0.0, 5.0)] * 6 + [(1000, True, 6.0, 5.0)] * 2 + [(1000, True, 0.0, 5.0)] * 2
    assert math.isclose(sustained_rate(mixed, 10_000), 2 / 5)
    assert sustained_rate([(1000, True, 6.0, 5.0)] * 9 + ["crash"], 10_000) == 0.0
    assert sustained_rate([], 10_000) == 0.0
    assert sustained_rate([(1000, True, 6.0, 5.0)] * 20, 10_000) == 1.0   # lives past the rung ignored


def test_metric_hash_moves_with_either_file():
    with tempfile.TemporaryDirectory() as t:
        a, b = Path(t) / "a.py", Path(t) / "b.py"
        a.write_text("x = 1\n")
        b.write_text("y = 1\n")
        h0 = metric_hash((a, b))
        b.write_text("y = 2\n")
        h1 = metric_hash((a, b))
        a.write_text("x = 2\n")
        assert len({h0, h1, metric_hash((a, b))}) == 3
    assert len(metric_hash()) == 64


# ---- files on disk and the scoreboard ---------------------------------------
def test_unfrozen_rule_is_refused():
    with tempfile.TemporaryDirectory() as t:
        raises(lambda: load_round(t), "no frozen rule")
        forge_scoreboard.synthetic_round(Path(t))
        raises(lambda: load_round(Path(t) / "dry"), "not committed")   # a temp file is not in git
        spec, rows = load_round(Path(t) / "dry", require_committed=False)
        assert len(rows) == 15 and spec["round"] == "dry"


def test_dry_round_renders():
    with tempfile.TemporaryDirectory() as t:
        root = forge_scoreboard.synthetic_round(Path(t))
        text = forge_scoreboard.render(root, Path(t) / "none.json", require_committed=False)
    line = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("| ")}
    assert "CLIMBS" in line["`winner`"] and "+0.5" in line["`winner`"]
    assert "CLIMBS" not in line["`null`"] and "CLIMBS" not in line["`ctl`"]
    assert line["published"].split("|")[2:5] == [" 1 ", " 1 ", " 0 "] and "yes" in line["published"]
    assert "**NO**" in line["inventor"]
    assert "1.50 of 10 core-h" in text and "SYNTHETIC" in text and "No reading yet." in text


def test_refused_round_is_shown_as_refused_not_dropped():
    with tempfile.TemporaryDirectory() as t:
        root = forge_scoreboard.synthetic_round(Path(t))
        p = root / "dry" / "results.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        rows[0]["metric_sha256"] = "0" * 64
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        text = forge_scoreboard.render(root, Path(t) / "none.json", require_committed=False)
    assert "REFUSED" in text and "CLIMBS" not in text


def test_real_scoreboard_is_current():
    assert forge_scoreboard.main(["--check"]) == 0


def _lock_repo(t):
    """A temp repo shaped like ours: harness, registry, ladder rule, metric,
    a round with its spec and reader; committed, then locked and committed."""
    import subprocess as sp
    from experiments.forge import write_lock
    r = Path(t)
    for n in ("experiments/harness/__init__.py", "experiments/registry.py",
              "experiments/forge/__init__.py", "experiments/forge/metric.py",
              "experiments/forge/rounds/r9/spec.json", "experiments/forge/r9_read.py",
              "experiments/unrelated.py"):
        (r / n).parent.mkdir(parents=True, exist_ok=True)
        (r / n).write_text(f"# {n}\n")
    git = lambda *a: sp.run(["git", "-C", t, "-c", "user.email=t@t", "-c", "user.name=t", *a],  # noqa: E731
                            check=True, capture_output=True)
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "freeze")
    rd = r / "experiments/forge/rounds/r9"
    lock = write_lock(rd, [r / "experiments/forge/r9_read.py"])
    git("add", str(rd / "judge_lock.json"))
    git("commit", "-qm", "lock")
    return r, rd, lock, git


def test_judge_lock_refuses_a_planted_edited_judge():
    from experiments.forge import check_lock, write_lock
    with tempfile.TemporaryDirectory() as t:
        r, rd, lock, git = _lock_repo(t)
        assert set(lock["files"]) == {
            "experiments/harness/__init__.py", "experiments/registry.py",
            "experiments/forge/__init__.py", "experiments/forge/metric.py",
            "experiments/forge/rounds/r9/spec.json", "experiments/forge/r9_read.py"}
        check_lock(rd)                                   # control: untouched judge passes
        (r / "experiments/unrelated.py").write_text("x = 1\n")
        check_lock(rd)                                   # a non-judge edit is not a judge edit
        raises(lambda: write_lock(rd, ["experiments/forge/r9_read.py"]), "locked once")
        for n in ("experiments/harness/__init__.py",     # plant 1: the harness
                  "experiments/forge/r9_read.py"):       # plant 2: the reader's own file
            f = r / n
            good = f.read_text()
            f.write_text(good + "THRESHOLD = 0.0\n")
            raises(lambda: check_lock(rd), "judge moved")
            git("commit", "-qam", "edit committed")      # committing the edit does not launder it
            raises(lambda: check_lock(rd), "judge moved")
            f.write_text(good)
            git("commit", "-qam", "revert")
            check_lock(rd)
        lp = rd / "judge_lock.json"                      # plant 3: re-hash the lock to the edit
        doc = json.loads(lp.read_text())
        doc["files"]["experiments/registry.py"] = "0" * 64
        lp.write_text(json.dumps(doc))
        raises(lambda: check_lock(rd), "edited after")
        git("commit", "-qam", "relock")
        raises(lambda: check_lock(rd), "edited after")


def test_no_result_is_read_without_the_lock():
    import subprocess as sp
    from experiments.forge import check_lock
    with tempfile.TemporaryDirectory() as t:
        rd = Path(t) / "rounds/r8"
        rd.mkdir(parents=True)
        sp.run(["git", "init", "-q", t], check=True)
        raises(lambda: check_lock(rd), "no judge lock")


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
