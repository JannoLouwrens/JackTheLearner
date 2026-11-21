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


# ---- NH-2 (J14 (c), ruling 70): one change, own seeds, own tag, the gate in code -----------------
# sha256 of kernel_body(...) computed from the file at 8f13ab28, BEFORE the NH-2 code was added
NH1_BODY_SHA = "a1dc433542035153ea1eb6091d5aba0058d26693bd2b909c77a0e7ce8b600e82"


def test_nh1_path_is_byte_identical():
    import hashlib
    cells = ["dqn@5000", "gmq@5000", "frozen@5000"]
    assert hashlib.sha256(m.kernel_body(cells, "needs_grid_x").encode()).hexdigest() == NH1_BODY_SHA
    assert m.MUS == [(10, 10), (10, 0), (0, 0), (0, 10)] and m.SEEDS == list(range(10))
    assert "MUS_INSIDE" not in m.kernel_body(cells, "t") and "NH2" not in m.kernel_body(cells, "t")
    # the default map is still NH-1's: the peak is in the wall, the best walkable cell gets e^-1 of it
    g = m.Grid()
    assert abs(g.res[9, 9, 0] * 2 * np.pi - np.exp(-1.0)) < 1e-9


def test_nh2_moves_only_mu():
    import ast
    g = m.Grid(m.MUS_INSIDE)
    assert abs(g.res[9, 9, 0] * 2 * np.pi - 1.0) < 1e-9          # the peak ON the walkable corner
    assert np.argmax(g.res[1, 1]) == 2 and g.res[0].sum() == 0   # same resource order, walls still empty
    b = m.kernel_body_nh2(m.NH2_CELLS, "nh2_grid_t")
    ast.parse(b)
    assert "mus=MUS_INSIDE" in b and "for s in NH2_SEEDS" in b and "for s in SEEDS" not in b
    assert m.NH2_SEEDS == list(range(10, 20)) and not set(m.NH2_SEEDS) & set(m.SEEDS)
    with pytest.raises(SystemExit):                                # the whole row or nothing
        m.kernel_body_nh2(["dqn@5000"], "t")


def test_nh2_judge_reads_its_own_seeds_only():
    def runs(seeds, meds=GOOD):
        return [{"kind": k, "anneal": a, "seed": s, "delta": meds[c]}
                for c, (k, a) in m.CELLS.items() for s in seeds]
    j = m.judge(runs(m.NH2_SEEDS), m.NH2_SEEDS)
    assert (j["R1"], j["R2"], j["R3"]) == ("PASS", "PASS", "MET")
    # NH-1's seeds do not score under NH-2, NH-2's do not score under NH-1, and a mix is not a row
    assert m.judge(runs(m.SEEDS), m.NH2_SEEDS)["verdict"] == "INCOMPLETE"
    assert m.judge(runs(m.NH2_SEEDS))["verdict"] == "INCOMPLETE"
    assert m.judge(runs(m.SEEDS) + runs(m.NH2_SEEDS), m.NH2_SEEDS)["verdict"] == "INCOMPLETE"
    # the bands did not move: NH-1's measured medians fail NH-2's R1 and R2 as they failed NH-1's
    jb = m.judge(runs(m.NH2_SEEDS, {**GOOD, "dqn@5000": 16.8, "gmq@5000": 8.2, "gmq@1": 5.4}), m.NH2_SEEDS)
    assert jb["R1"] == "FAIL" and jb["R2"] == "FAIL"


def test_nh2_gate(tmp_path):
    import json

    def put(seed, delta, **kw):
        r = {"kind": "dqn", "anneal": 5000, "seed": seed, "steps": m.STEPS, "delta": delta, "mus": m.MUS_INSIDE}
        r.update(kw)
        json.dump(r, open(tmp_path / f"pilot_mu_inside_dqn5000_s{seed}.json", "w"))
    assert m.nh2_gate(str(tmp_path))["open"] is False               # nothing home: closed, and says WAIT
    assert m.nh2_gate(str(tmp_path))["reading"].startswith("WAIT")
    put(2, 1.2)
    assert m.nh2_gate(str(tmp_path))["open"] is False               # one seed is not both
    put(5, 4.99)
    assert m.nh2_gate(str(tmp_path))["open"] is True
    put(5, 5.0)                                                     # the rule says < 5.0
    g = m.nh2_gate(str(tmp_path))
    assert g["open"] is False and g["reading"].startswith("NO NH-2")
    put(5, 1.0, mus=m.MUS)                                          # a frozen-mu run is not the pilot
    assert m.nh2_gate(str(tmp_path))["open"] is False
    put(5, 1.0, steps=1500)                                         # nor is a short one
    assert m.nh2_gate(str(tmp_path))["open"] is False
