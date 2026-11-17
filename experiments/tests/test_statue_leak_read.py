"""Planted traces for statue_leak_probe.verdict (the READING RULE).
Each planted cause must be named; a mismatched replay must be VOID."""
from experiments.statue_leak_probe import verdict


def _res(root1, food1, mism=()):
    t0 = {"k": 0, "tag": "spawn", "root": [0, 0, 1], "foods": {"f": [1.6, 0]},
          "dist": {"f": 1.6}, "ate": {"f": 0}}
    t1 = {"k": 10, "tag": "tick", "root": [0, 0, 1], "foods": {"f": [1.6, 0]},
          "dist": {"f": 1.6}, "ate": {"f": 0}}
    t2 = {"k": 20, "tag": "meal", "root": root1, "foods": {"f": food1},
          "dist": {"f": 0.3}, "ate": {"f": 1}}
    return {"mismatches": list(mism), "trace": [t0, t1, t2]}


def test_slide():
    v = verdict(_res([1.0, 0, 1], [1.6, 0]))
    assert v["verdict"] == "slide" and v["root_mv"] == 1.0 and v["food_mv"] == 0


def test_shove():
    assert verdict(_res([0, 0, 1], [0.4, 0]))["verdict"] == "shove"


def test_both():
    assert verdict(_res([0.6, 0, 1], [1.0, 0]))["verdict"] == "both"


def test_control_mismatch_is_void():
    v = verdict(_res([1.0, 0, 1], [1.6, 0], mism=[{"life": 3}]))
    assert v["verdict"] == "VOID"


def main():
    for f in (test_slide, test_shove, test_both, test_control_mismatch_is_void):
        f()
        print("ok", f.__name__)
    print("PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
