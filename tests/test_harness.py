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
