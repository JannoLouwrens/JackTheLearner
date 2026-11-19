"""Planted tests for experiments/needs_grid_row.py: the grid's rules and the frozen verdict."""
import numpy as np
import pytest

from experiments import needs_grid_row as m


def test_grid_rules():
    g = m.Grid()
    assert g.pos == (5, 5) and g.obs().shape == (40,)
    h0 = g.h.copy()
    g.step(0)                                   # up, into an empty cell: only decay
    assert np.allclose(g.h, h0 + g.res[g.pos] - m.DECAY)
    g.pos = (1, 1)
    g.step(0)                                   # into the wall: does not move
    assert g.pos == (1, 1)
    assert g.res[0].sum() == 0 and g.res[:, 0].sum() == 0
    # each corner cell carries only its own resource, peak at the walkable corner
    assert np.argmax(g.res[1, 1]) == 2 and np.argmax(g.res[9, 9]) == 0


def test_drive_reward_sign():
    lo, hi = np.full(4, 0.5), np.full(4, 1.0)
    assert m.drive(lo) - m.drive(hi) > 0       # moving toward h* is rewarded
    assert np.all(m.drive_each(lo) - m.drive_each(hi) > 0)


def _runs(meds, hidden_wins=10):
    runs = []
    for cell, (kind, anneal) in m.CELLS.items():
        for s in m.SEEDS:
            d = meds[cell]
            if cell == "hidden@5000" and s >= hidden_wins:
                d = 0.0
            runs.append({"kind": kind, "anneal": anneal, "seed": s, "delta": d})
    return runs


GOOD = {"dqn@5000": 1.0, "gmq@5000": 0.5, "dqn@1": 3.0, "gmq@1": 0.6, "hidden@5000": 4.0, "frozen@5000": 15.0}


def test_judge_pass_and_planted_failures():
    j = m.judge(_runs(GOOD))
    assert (j["R1"], j["R2"], j["R3"]) == ("PASS", "PASS", "MET")
    assert m.judge(_runs({**GOOD, "frozen@5000": 4.0}))["verdict"].startswith("VOID")
    assert m.judge(_runs({**GOOD, "gmq@5000": 1.2}))["R1"] == "FAIL"     # gmq not better
    assert m.judge(_runs({**GOOD, "dqn@1": 1.0}))["R2"] == "FAIL"
    assert m.judge(_runs({**GOOD, "hidden@5000": 1.2}))["R3"].startswith("MISS")
    # hidden median high but only 7 of 10 pairs worse: still a miss
    assert m.judge(_runs(GOOD, hidden_wins=7))["R3"].startswith("MISS")
    assert m.judge(_runs(GOOD)[:-1])["verdict"] == "INCOMPLETE"


def test_kernel_body_refuses_unknown_cell():
    assert "dqn@5000" in m.kernel_body(["dqn@5000"], "t")
    with pytest.raises(SystemExit):
        m.kernel_body(["dqn@500"], "t")
