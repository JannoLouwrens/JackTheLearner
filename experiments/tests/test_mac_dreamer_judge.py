"""Row 10's judge (experiments/brain_a/mac_dreamer.py) on planted series — no sheeprl needed.

Each verdict the rule names must be reachable: PASS, DIVERGENCE (ratio and non-finite), VOID (REP differs;
CTRL cannot fail), and the no-MPS apparatus check."""
import math
import random

from experiments.brain_a import mac_dreamer as M

STEPS = 100


def _curve(seed, scale=1.0, offset=0.0):
    r = random.Random(seed)
    return [offset + 5.0 * math.exp(-i / 30) + scale * r.gauss(0, 0.3) for i in range(STEPS)]


def _base():
    ref = {"Loss/reward": _curve(0), "gradnorm/actor": _curve(10)}
    return {
        "REF": ref,
        "REP": {k: list(v) for k, v in ref.items()},
        "FL1": {"Loss/reward": _curve(1), "gradnorm/actor": _curve(11)},
        "FL2": {"Loss/reward": _curve(2), "gradnorm/actor": _curve(12)},
        "CTRL": {"Loss/reward": [5.0] * STEPS, "gradnorm/actor": _curve(13)},   # reward never learns
    }


def test_pass_within_floor():
    s = _base()
    s["TEST"] = {"Loss/reward": _curve(3), "gradnorm/actor": _curve(14)}
    v = M.judge(s, STEPS)
    assert v["rep_identical"] and v["ctrl_fails"], v
    assert v["verdict"] == "PASS", v


def test_divergence_by_offset():
    s = _base()
    s["TEST"] = {"Loss/reward": _curve(3, offset=1.0), "gradnorm/actor": _curve(14)}
    v = M.judge(s, STEPS)
    assert v["verdict"] == "DIVERGENCE" and v["test_over"] == ["Loss/reward"], v


def test_divergence_by_nan():
    s = _base()
    t = _curve(3)
    t[57] = float("nan")
    s["TEST"] = {"Loss/reward": t, "gradnorm/actor": _curve(14)}
    assert M.judge(s, STEPS)["verdict"] == "DIVERGENCE"


def test_void_when_rep_differs():
    s = _base()
    s["REP"]["gradnorm/actor"][3] += 1e-12
    s["TEST"] = {"Loss/reward": _curve(3), "gradnorm/actor": _curve(14)}
    assert M.judge(s, STEPS)["verdict"] == "VOID"


def test_void_when_control_cannot_fail():
    s = _base()
    s["CTRL"] = {"Loss/reward": _curve(4), "gradnorm/actor": _curve(13)}
    s["TEST"] = {"Loss/reward": _curve(3), "gradnorm/actor": _curve(14)}
    assert M.judge(s, STEPS)["verdict"] == "VOID"


def test_apparatus_only_without_test():
    v = M.judge(_base(), STEPS)
    assert v["verdict"].startswith("APPARATUS_OK"), v


def test_zero_floor_counts_any_distance_as_inf():
    assert M.ratios({"a": 0.0, "b": 1e-9}, {"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": math.inf}
