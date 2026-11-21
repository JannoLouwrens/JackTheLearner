"""J10 why step 2: the fork job is the row's own segment path with only the reseed added."""
from experiments import j10_fork as F


def test_job_body_compiles_and_mounts_s1_seg2():
    b = F.job_body(1)
    compile(b, "job", "exec")
    assert "FORK = 1" in b and "train_segment(1, out, _p, 3)" in b
    assert "/kaggle/input/jack-cpu-1795167808" in b and "sbx-rl==0.28.0" in b
    assert F.job_body(2) != b


def test_fork_outside_rule_refused():
    for f in (0, 3):
        try:
            F.job_body(f)
        except ValueError:
            continue
        raise AssertionError(f)


def test_reading_planted():
    L, K, B = {"R": 581, "fell": 10}, {"R": 5800, "fell": 0}, {"R": 3000, "fell": 5}
    assert F.reading({1: L, 2: L})["reading"] == "RECIPE"
    assert F.reading({1: K, 2: K})["reading"] == "CHAOS"
    assert F.reading({1: K, 2: L})["reading"] == "MIXED"
    assert F.reading({1: B, 2: B})["reading"] == "MIXED"
    assert F.reading({1: L})["reading"] == "INCOMPLETE"
    assert F.reading({1: {"R": 999, "fell": 7}, 2: L})["per_fork"][1] == "BETWEEN"
