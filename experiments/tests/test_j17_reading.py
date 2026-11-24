"""J17 (3): THE RULE's reading (pilot_b2.j17_readings), planted before any J17
kernel is home. Rule: docs/research/prereg/row128_vs_rays16.md FROZEN 98e22036.
Synthetic run records only: nothing here reads or writes a real run."""
import math
import sys

from experiments import pilot_b2 as P
from experiments import pilot_b_report as R

L = 3000                       # 100 lives x 3,000 decisions = 300,000


def _life(meals, cause="starved", decisions=L, fed=False):
    return {"decisions": decisions, "cause": cause, "floor_meals": meals, "apples": 0,
            "sustained": fed, "intake": 2.0 if fed else 0.0, "basal_drain": 1.0,
            "seconds": decisions / 10.0, "reward": float(meals)}


def run(early, late=None, rc=0):
    """early meals/life for lives starting < 150k, late for >= 150k."""
    late = early if late is None else late
    lives = [_life(early if i * L < P.J17_LATE_FROM else late) for i in range(100)]
    return {"lives": lives, "rc": rc, "T": P.J17_T}


def sustained_run():
    lives = [_life(9, "capped", 9000, fed=True) for _ in range(33)]
    lives.append(_life(0, "unfinished", 3000))
    return {"lives": lives, "rc": 0, "T": P.J17_T}


def book(row, rays, blind, lr0=None):
    vals = {"K-row128": row, "K-rays16": rays, "K-row128-blind": blind,
            "K-lr0-row128": lr0 or [run(0.1)] * 5}
    out = {}
    for a, v in vals.items():
        for s, r in zip(P.J17_SEEDS, v):
            out[R.rid(a, s)] = r if isinstance(r, dict) else run(r)
    return out


def test_helps_needs_both_clauses():
    res = P.j17_readings(book([3, 3.1, 3.2, 3.3, 3.4], [1] * 5, [0.5] * 5))
    assert res["U_rays16"] == 25 and res["U_blind"] == 25
    assert res["clause"] == "row128 HELPS"
    assert res["verdict"] == "neither eats"        # nobody crossed W=2


def test_width_not_sight():
    res = P.j17_readings(book([3] * 5, [1] * 5, [3] * 5))
    assert res["U_blind"] == 12.5 and res["clause"] == "width, not sight"


def test_reached_then_lost_reads_by_the_late_half():
    # row128 eats 5 a life early and 0 late; rays16 eats 1 throughout.
    res = P.j17_readings(book([run(5, 0)] * 5, [1] * 5, [0] * 5))
    assert res["late"]["K-row128"] == [0.0] * 5
    assert res["U_rays16"] == 0 and res["clause"].startswith("no difference")


def test_late_window_boundary_is_150k():
    r = run(0, 0)
    for i, life in enumerate(r["lives"]):
        start = i * L
        life["floor_meals"] = 7 if start == 150_000 else (99 if start == 147_000 else 0)
    assert abs(P.j17_late_meals(r) - 7 / 50) < 1e-12   # 50 lives start >= 150k


def test_mwu_needs_four_seed_pairs():
    res = P.j17_readings(book([10, 10, 10, 2, 2], [1, 1, 1, 3, 3], [0] * 5))
    assert res["U_rays16"] == 21 and res["pairs_rays16"] == 3
    assert res["clause"].startswith("no difference")


def test_nan_late_never_helps_row128():
    empty = {"lives": [_life(0, "unfinished", P.J17_T)], "rc": 0, "T": P.J17_T}
    assert math.isnan(P.j17_late_meals(empty))
    res = P.j17_readings(book([3] * 5, [empty] * 5, [0] * 5))
    assert res["U_rays16"] == 0


def test_blind_sustained_voids():
    res = P.j17_readings(book([sustained_run()] * 5, [1] * 5, [sustained_run()] * 3 + [0, 0]))
    assert res["verdict"] == "VOID — broken"


def test_lr0_crossing_voids():
    res = P.j17_readings(book([3] * 5, [1] * 5, [0] * 5, lr0=[sustained_run()] * 2 + [run(0)] * 3))
    assert res["verdict"] == "VOID — broken"


def test_row128_eats_verdict():
    res = P.j17_readings(book([sustained_run()] * 4 + [3], [1] * 5, [0] * 5))
    assert res["arms"]["K-row128"] == "SUSTAINED"
    assert res["verdict"] == "row128 eats"
    assert "row128 eats" in P.j17_md(res)


def test_missing_run_raises():
    b = book([3] * 5, [1] * 5, [0] * 5)
    b.pop(R.rid("K-rays16", 923))
    try:
        P.j17_readings(b)
    except ValueError:
        return
    raise AssertionError("a missing run must raise")


if __name__ == "__main__":       # no venv on the box has pytest (INBOX 11-21 07:25)
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)


def test_late_from_beyond_the_run_raises():
    """A mark the run never reaches is a mis-set reader, not "no late meals": it must raise,
    not return nan (which reads as NOT EATING / never-helps downstream). Planted: 99k run."""
    short = {"lives": [_life(5) for _ in range(33)], "rc": 0, "T": 99_000}   # 99,000 decisions
    for bad in (99_000, 150_000):
        try:
            P.j17_late_meals(short, late_from=bad)
        except ValueError:
            continue
        raise AssertionError(f"late_from {bad} past a 99k run returned instead of raising")
    assert P.j17_late_meals(short, late_from=50_000) == 5.0
