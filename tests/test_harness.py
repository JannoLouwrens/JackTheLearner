"""J4 harness self-checks #10 and #12 (PUBLISHED_PRIORS §8 "Reproduce first").

Each assertion can fail: the expected values are written from the rule text
and the record's published numbers, not read back from the code.
Run: /data/venvs/jackthelearner/bin/python -m tests.test_harness   (no pytest in the venv)
"""
import math

from experiments.harness import (Outcome, boschloo_p, fisher_p, logrank_stat,
                                 outcome, perm_logrank)


# ---- #12: tools on the record's known case ---------------------------------
def test_12_fisher_known_case():
    assert round(fisher_p(4, 15, 0, 15), 3) == 0.100


def test_12_boschloo_known_case():
    assert round(boschloo_p(4, 15, 0, 15), 3) == 0.043


def test_12_arm_order_does_not_matter():
    assert math.isclose(boschloo_p(0, 15, 4, 15), boschloo_p(4, 15, 0, 15))


# ---- #10: outcome code on synthetic logs, edge cases -----------------------
W, T = 3, 10_000


def test_10_crosses_at_end_of_first_good_window():
    lives = [(1000, 0), (1000, 5), (1000, 5), (1000, 5), (1000, 9)]
    # windows: (0,5,5)=3.33 no; (5,5,5)=5.0 yes -> end of life 4 = 4000
    assert outcome(lives, W, T) == Outcome(4000.0, True)


def test_10_exact_threshold_counts():
    assert outcome([(10, 5)] * 3, W, T) == Outcome(30.0, True)


def test_10_just_below_threshold_never_crosses():
    assert outcome([(10, 4.999)] * 50, W, T) == Outcome(float(T), False)


def test_10_mean_not_every_life():
    # 0, 0, 15 averages 5: the window MEAN decides, not each life
    assert outcome([(10, 0), (10, 0), (10, 15)], W, T) == Outcome(30.0, True)


def test_10_fewer_than_W_lives_censored():
    assert outcome([(10, 99), (10, 99)], W, T) == Outcome(float(T), False)


def test_10_crossing_at_exactly_T_counts_past_T_does_not():
    assert outcome([(T / 3, 5)] * 3, W, T).event
    lives = [(4000, 5), (4000, 5), (2001, 5)]
    assert outcome(lives, W, T) == Outcome(float(T), False)


def test_10_crash_and_nan_are_censored_at_T():
    good = [(10, 9)] * 3
    assert outcome(["crash"] + good, W, T) == Outcome(float(T), False)
    assert outcome([(10, 9), (10, float("nan"))] + good, W, T) == Outcome(float(T), False)
    assert outcome([(10, 9), (float("nan"), 9)] + good, W, T).event is False


def test_10_window_does_not_reach_back_across_a_crash():
    # lives after a crash never count, even if the window would close
    assert outcome([(10, 9), (10, 9), "crash", (10, 9)], W, T).event is False


def test_10_W1_first_good_life():
    assert outcome([(7, 1), (7, 6)], 1, T) == Outcome(14.0, True)


def test_10_bad_args_raise():
    for w, t in [(0, T), (W, 0)]:
        try:
            outcome([(1, 1)], w, t)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# ---- the test statistic behaves (not a numbered check; guards #1 later) ----
def test_logrank_identical_arms_zero_and_separated_arms_small_p():
    fast = [Outcome(1000.0 + i, True) for i in range(10)]
    slow = [Outcome(float(T), False) for _ in range(10)]
    assert logrank_stat(fast, list(fast)) < 1e-9
    assert logrank_stat(slow, slow) == 0.0
    assert perm_logrank(fast, slow, n_perm=500, rng=0) < 0.01
    assert perm_logrank(fast, list(fast), n_perm=200, rng=0) > 0.5


def test_compare_two_looks_and_futility():
    from experiments.harness import compare
    fast = [Outcome(1000.0 + i, True) for i in range(20)]
    slow = [Outcome(float(T), False) for _ in range(20)]
    v = compare(fast, slow, n_perm=500, rng=0)
    assert v.decision == "A_DIFFERS" and v.look == 1
    assert compare(slow, list(slow), n_perm=100, rng=0).decision == "FUTILE"
    v = compare(fast, list(fast), n_perm=200, rng=0)
    assert v.decision == "NOT_DISTINGUISHED" and v.look == 2


# ---- #13: runs analysed = runs launched ------------------------------------
def test_13_books_balance():
    from experiments.harness import check_runs_accounted
    o = Outcome(float(T), False)
    check_runs_accounted({"A": [0, 1], "B": [0, 1]},
                         {"A": {0: o, 1: o}, "B": {0: o, 1: o}})


def test_13_dropped_extra_duplicate_and_missing_arm_all_raise():
    from experiments.harness import check_runs_accounted
    o = Outcome(float(T), False)
    bad = [({"A": [0, 1]}, {"A": {0: o}}),                 # a run dropped
           ({"A": [0]}, {"A": {0: o, 7: o}}),              # a run never launched
           ({"A": [0, 0]}, {"A": {0: o}}),                 # duplicate seed
           ({"A": [0], "B": [0]}, {"A": {0: o}})]          # a whole arm missing
    for launched, analysed in bad:
        try:
            check_runs_accounted(launched, analysed)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {launched} vs {analysed}")


# ---- #3: calibration arms at floor and ceiling ------------------------------
def _calib_runs(kind, n, seed):
    """Synthetic per-life logs through the real `outcome`: floor = the record's
    random arm (0-0.2 meals/life -> Poisson 0.1), ceiling = its scripted
    forager (24-27 meals/life). 1,000-decision lives, W=3, T=60,000."""
    import numpy as np
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n):
        if kind == "floor":
            lives = [(1000, int(rng.poisson(0.1))) for _ in range(60)]
        else:
            lives = [(1000, int(rng.integers(24, 28))) for _ in range(60)]
        runs.append(outcome(lives, 3, 60_000))
    return runs


def test_3_gate_valid_on_record_like_arms():
    from experiments.harness.calibration import calibration_gate
    floor = {a: _calib_runs("floor", 3, i) for i, a in enumerate(("random", "statue", "twin"))}
    floor["shuffled"] = _calib_runs("floor", 5, 9)
    g = calibration_gate(floor, {"scripted": _calib_runs("ceiling", 3, 4)})
    assert g.valid and g.floor_successes == 0 and g.ceiling_misses == 0, g


def test_3_gate_voids_two_floor_successes_or_one_ceiling_miss():
    from experiments.harness.calibration import calibration_gate
    hit, miss = Outcome(3000.0, True), Outcome(60_000.0, False)
    ceil = {"scripted": [hit] * 3}
    assert calibration_gate({"random": [hit, miss, miss]}, ceil).valid          # 1 in total: allowed
    assert not calibration_gate({"random": [hit, miss], "statue": [hit]}, ceil).valid  # 2 across arms
    assert not calibration_gate({"random": [miss] * 3}, {"scripted": [hit, miss, hit]}).valid


def test_3_gate_voids_missing_or_empty_arms():
    from experiments.harness.calibration import calibration_gate
    hit, miss = Outcome(3000.0, True), Outcome(60_000.0, False)
    assert not calibration_gate({}, {"scripted": [hit]}).valid
    assert not calibration_gate({"random": [miss]}, {}).valid
    assert not calibration_gate({"random": []}, {"scripted": [hit]}).valid


def test_3_pipeline_separates_ceiling_from_floor_and_floor_is_futile():
    from experiments.harness import compare
    floor, ceil = _calib_runs("floor", 20, 11), _calib_runs("ceiling", 20, 12)
    v = compare(ceil, floor, n_perm=500, rng=0)
    assert v.decision == "A_DIFFERS" and v.look == 1, v
    assert compare(floor, _calib_runs("floor", 20, 13), n_perm=200, rng=0).decision == "FUTILE"



# ---- #11: analysis frozen and hashed; `judge` refuses without it -----------
def _prereg_setup():
    import shutil, tempfile
    from pathlib import Path
    from experiments.harness import prereg
    d = Path(tempfile.mkdtemp())
    for n in prereg.ANALYSIS_FILES:
        shutil.copy(prereg.HERE / n, d / n)
    ids = {"A": [f"a{i}" for i in range(20)], "B": [f"b{i}" for i in range(20)]}
    runs = {arm: {rid: [(1000, 6)] * 3 for rid in ids[arm]} for arm in ids}
    pre = d / "prereg.json"
    prereg.freeze(pre, question="A vs B", launched=ids, W=3, T=60_000, n_perm=200, root=d)
    return prereg, d, pre, runs


def test_11_frozen_judge_runs():
    prereg, d, pre, runs = _prereg_setup()
    v = prereg.judge(pre, runs, require_committed=False, root=d)
    assert v.decision == "NOT_DISTINGUISHED"   # identical arms, all cross at 3000


def test_11_refuses_when_analysis_code_moved():
    prereg, d, pre, runs = _prereg_setup()
    with open(d / "__init__.py", "a") as f:
        f.write("\n# one byte of drift\n")
    try:
        prereg.judge(pre, runs, require_committed=False, root=d)
    except prereg.Refused as e:
        assert "__init__.py" in str(e)
    else:
        raise AssertionError("moved analysis code was not refused")


def test_11_refuses_without_preregistration_or_uncommitted():
    prereg, d, pre, runs = _prereg_setup()
    for kw, path in ((dict(require_committed=False), d / "missing.json"),
                     (dict(require_committed=True), pre)):   # temp file: not in git
        try:
            prereg.judge(path, runs, root=d, **kw)
        except prereg.Refused:
            pass
        else:
            raise AssertionError(f"judge ran without a committed pre-registration ({path})")


def test_11_runs_must_balance_the_frozen_manifest():
    prereg, d, pre, runs = _prereg_setup()
    del runs["B"]["b7"]
    try:
        prereg.judge(pre, runs, require_committed=False, root=d)
    except ValueError as e:
        assert "b7" in str(e)
    else:
        raise AssertionError("a dropped run was not caught")



def test_logrank_all_tied_identical_arms_not_distinguished():
    # every run in both arms crosses at 3000: zero log-rank variance (SciPy NaN)
    from experiments.harness import compare, logrank_stat, perm_logrank
    tied = [Outcome(3000.0, True)] * 20
    assert logrank_stat(tied, tied) == 0.0
    assert perm_logrank(tied[:10], tied[:10], 200, 0) == 1.0
    assert compare(tied, list(tied), n_perm=200, rng=0).decision == "NOT_DISTINGUISHED"

# ---- #10 re-run on ruling 22's sustained mark (J8 item 3) ------------------
from experiments.harness import outcome_sustained, sustained_life  # noqa: E402

CAP = 9000


def _s(d=CAP, capped=True, intake=1.0, drain=0.5):
    return (d, capped, intake, drain)


def test_22_life_needs_cap_and_intake_at_least_drain():
    assert sustained_life(True, 3.0, 3.0)            # exactly at drain counts
    assert not sustained_life(True, 2.999, 3.0)      # just below does not
    assert not sustained_life(False, 99.0, 3.0)      # died rich: not sustained
    assert not sustained_life(True, float("nan"), 3.0)
    assert not sustained_life(True, 3.0, float("nan"))
    assert not sustained_life(1, 3.0, 1.0)           # capped must be True itself


def test_22_crosses_at_end_of_Wth_consecutive():
    lives = [_s(), _s(intake=0.1), _s(), _s(), _s()]
    # streak: 1, 0, 1, 2, 3 -> crosses at the end of life 5
    assert outcome_sustained(lives, 3, 10 * CAP) == Outcome(5.0 * CAP, True)


def test_22_every_life_not_window_mean():
    # theta-style mean would pass (two rich lives carry a poor one); this must not
    lives = [_s(intake=9.0), _s(intake=0.0), _s(intake=9.0)] * 3
    assert outcome_sustained(lives, 3, 100 * CAP).event is False


def test_22_a_death_breaks_the_streak():
    lives = [_s(), _s(), (500, False, 0.0, 0.1), _s(), _s()]
    assert outcome_sustained(lives, 3, 100 * CAP).event is False
    assert outcome_sustained(lives + [_s()], 3, 100 * CAP) == Outcome(5 * CAP + 500.0, True)


def test_22_T_bound_crash_nan_and_few_lives():
    assert outcome_sustained([_s()] * 3, 3, 3 * CAP).event          # exactly T counts
    assert outcome_sustained([_s()] * 3, 3, 3 * CAP - 1).event is False
    assert outcome_sustained([_s()] * 2, 3, 10 * CAP).event is False
    assert outcome_sustained([_s(), "crash", _s(), _s()], 3, 10 * CAP).event is False
    assert outcome_sustained([_s(), (float("nan"), True, 1, 0), _s(), _s()], 3,
                             10 * CAP).event is False


def test_22_planted_known_positive_and_theta_disagree():
    # f2-like capped lives that ate below drain: theta says crossed, sustained says no
    f2 = [(CAP, True, 2.4, 3.0, 30)] * 3
    assert outcome([(l[0], l[4]) for l in f2], 3, 10 * CAP).event
    assert outcome_sustained([l[:4] for l in f2], 3, 10 * CAP).event is False
    # row-e-like fed lives: both marks cross
    e = [(CAP, True, 4.3, 3.0, 53)] * 3
    assert outcome_sustained([l[:4] for l in e], 3, 10 * CAP) == Outcome(3.0 * CAP, True)


def test_22_bad_args_raise():
    for w, t in [(0, CAP), (3, 0)]:
        try:
            outcome_sustained([_s()], w, t)
        except ValueError:
            continue
        raise AssertionError("bad args accepted")


def test_22_judge_scores_the_frozen_mark():
    prereg, d, pre, _ = _prereg_setup()
    ids = {"A": [f"a{i}" for i in range(20)], "B": [f"b{i}" for i in range(20)]}
    fed = [(9000, True, 4.0, 3.0)] * 3          # sustained
    hungry = [(9000, True, 2.0, 3.0)] * 3       # capped, below drain
    runs = {"A": {r: fed for r in ids["A"]}, "B": {r: hungry for r in ids["B"]}}
    prereg.freeze(pre, question="fed vs hungry", launched=ids, W=3, T=60_000,
                  n_perm=200, root=d, mark="sustained")
    assert prereg.judge(pre, runs, require_committed=False, root=d).decision == "A_DIFFERS"
    runs["A"] = {r: hungry for r in ids["A"]}   # control: both hungry -> nothing crosses
    assert prereg.judge(pre, runs, require_committed=False, root=d).decision == "FUTILE"


def test_pilot_b_kernels_cover_the_registered_35_runs_once():
    """J8 item 4: the kernel split launches exactly the rule's runs — B x20
    (301-320), noop/held8/blr0 x5 (301-305) — each once, <= 4 per kernel,
    and the job body the kernel ships is valid Python with the pins in it."""
    from experiments import pilot_b as p
    g = p.kernel_groups()
    flat = [c for v in g.values() for c in v]
    assert len(flat) == len(set(flat)) == 35
    assert sorted(flat) == sorted(p.all_cells())
    assert {s for a, s in flat if a == "b"} == set(range(301, 321))
    for arm in p.FLOOR_ARMS:
        assert {s for a, s in flat if a == arm} == set(range(301, 306))
    assert all(len(v) <= 4 for v in g.values())
    assert all(sum(a == "blr0" for a, _ in v) <= 2 for v in g.values())
    body = p.job_body(g["k0"])
    compile(body, "job", "exec")
    assert "sbx-rl==0.28.0" in body and "300000" in body
    try:
        p.run_cell("nope", 1, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown arm must refuse")


# ---- S20 / ruling 123: look 1 read with ten runs home -----------------------
def _mixed(seed, n, rate):
    import numpy as np
    r = np.random.default_rng(seed)
    return [Outcome(float(r.integers(1000, 50_000)), True) if r.random() < rate
            else Outcome(float(T * 10), False) for _ in range(n)]


def test_s20_look1_p_same_at_10_and_20_runs():
    from experiments.harness import compare
    from experiments.harness.looks import compare_looks
    a, b = _mixed(1, 20, 0.6), _mixed(2, 20, 0.5)
    full = compare(a, b, n_perm=300, rng=7)
    for k in (10, 13, 19):
        v = compare_looks(a[:k], b[:k], n_perm=300, rng=7)
        assert v.p1 == full.p1 and v.look == 1, (k, v, full)
    assert compare_looks(a, b, n_perm=300, rng=7) == full   # >= 20: compare itself


def test_s20_planted_look1_winner_stops_at_10():
    from experiments.harness.looks import compare_looks
    fast = [Outcome(1000.0 + i, True) for i in range(10)]
    slow = [Outcome(float(T), False) for _ in range(10)]
    v = compare_looks(fast, slow, n_perm=500, rng=0)
    assert v.decision == "A_DIFFERS" and v.look == 1 and v.p1 <= 0.025
    assert compare_looks(slow, list(slow), n_perm=100, rng=0).decision == "FUTILE"


def test_s20_null_pair_at_10_needs_look_2_and_under_10_raises():
    from experiments.harness.looks import compare_looks
    fast = [Outcome(1000.0 + i, True) for i in range(10)]
    v = compare_looks(fast, list(fast), n_perm=200, rng=0)
    assert v.decision == "NEEDS_LOOK_2" and v.look == 1 and math.isnan(v.p2)
    for k in (0, 9):
        try:
            compare_looks(fast[:k], fast, n_perm=50, rng=0)
        except ValueError:
            continue
        raise AssertionError(f"{k} runs an arm took a look")


def test_s20_judge_looks_passes_through_and_keeps_refusals():
    from experiments.harness.looks import judge_looks
    prereg, d, pre, runs = _prereg_setup()
    full = prereg.judge(pre, runs, require_committed=False, root=d)
    assert judge_looks(pre, runs, require_committed=False, root=d) == full
    ten = {arm: {rid: runs[arm][rid] for rid in list(runs[arm])[:10]} for arm in runs}
    v = judge_looks(pre, ten, require_committed=False, root=d)
    assert v.decision == "NEEDS_LOOK_2" and v.p1 == full.p1   # identical arms
    gap = {arm: dict(r) for arm, r in ten.items()}
    del gap["B"]["b3"]
    gap["B"]["b12"] = runs["B"]["b12"]
    for bad in (gap, {"A": ten["A"], "B": dict(list(ten["B"].items())[:9])},
                {**ten, "C": {"c0": [(1000, 6)] * 3}}):
        try:
            judge_looks(pre, bad, require_committed=False, root=d)
        except (prereg.Refused, ValueError):
            continue
        raise AssertionError(f"judge_looks read an unbalanced look 1: {list(bad)}")


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

