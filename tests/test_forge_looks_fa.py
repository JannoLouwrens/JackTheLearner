"""Planted tests for the ten-run false-alarm sample's tally
(`experiments/forge/looks_fa.py`). Run:
/data/venvs/jackthelearner/bin/python -m tests.test_forge_looks_fa
"""
from experiments.forge import looks_fa as F


def row(r, d="NEEDS_LOOK_2", p1=0.5, rec_p1=None, rec_d=None, look=1):
    return {"scen": "S1", "r": r, "d": d, "look": look, "p1": p1,
            "rec_p1": p1 if rec_p1 is None else rec_p1, "rec_d": d if rec_d is None else rec_d}


def test_clean_sample_holds_and_counts():
    rows = [row(r) for r in range(296)] + [row(296 + r, "A_DIFFERS", 0.01) for r in range(3)] \
        + [row(299, "FUTILE")]
    t = F.tally(rows)
    assert t["verdict"] == "HOLDS (sample)" and t["n"] == 300
    assert (t["false_alarms"], t["futile"], t["needs_look_2"]) == (3, 1, 296)
    assert t["cp95"][0] < 0.01 < t["cp95"][1]


def test_one_moved_p_fails():
    rows = [row(r) for r in range(299)] + [row(299, p1=0.5, rec_p1=0.5000001)]
    assert F.tally(rows)["verdict"].startswith("FAIL (the wrapper")


def test_a_verdict_that_is_not_the_records_look_1_fails():
    rows = [row(r) for r in range(299)] + [row(299, "NEEDS_LOOK_2", rec_d="A_DIFFERS")]
    assert F.tally(rows)["mismatch"] == [("S1", 299)]
    rows = [row(r) for r in range(299)] + [row(299, look=2)]
    assert F.tally(rows)["verdict"].startswith("FAIL (the wrapper")


def test_too_many_false_alarms_fail():
    rows = [row(r) for r in range(270)] + [row(270 + r, "A_DIFFERS", 0.01) for r in range(30)]
    assert F.tally(rows)["verdict"].startswith("FAIL (false-alarm")   # 0.10, CP low end > 0.025


def test_empty_is_not_a_pass():
    assert F.tally([])["verdict"].startswith("FAIL")


def test_expected_maps_a_20_run_verdict_to_its_look_1():
    assert F.expected({"d": "A_DIFFERS", "look": 1}) == "A_DIFFERS"
    assert F.expected({"d": "FUTILE", "look": 1}) == "FUTILE"
    assert F.expected({"d": "A_DIFFERS", "look": 2}) == "NEEDS_LOOK_2"
    assert F.expected({"d": "NOT_DISTINGUISHED", "look": 2}) == "NEEDS_LOOK_2"


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:
                fails += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if fails else 0)
