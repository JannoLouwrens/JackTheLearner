"""run_b's training trace (11-19, J15 E2c, before seed 801).

Known answers, each with a control that must fail:
- trace_window: pinned actions read sat 1.0 / std 0; uniform(-0.5, 0.5)
  actions read sat 0 and std ~0.29 (control: the two differ).
- READS ONLY: arm K's world, 600 decisions with training switched on early
  (WARMUP 100): trace on and trace off give byte-identical lives. Control:
  the same comparison against another seed DIFFERS (the check can fail).
- The trace has one row per TRACE_EVERY window, and rows after warm-up carry
  SAC's actor_loss / critic_loss / ent_coef (0.1 fixed) as finite numbers.
- trace_row (11-21, Next-up 3): a learner key that collides with a window key
  (SBX's ``train/std``) is stored as ``train_std``; the window's per-dim
  action ``std`` survives in every row. Planted: last = {"std": 0.7, ...}.
  Control: the pre-fix row build (``row.update(last)``) loses the window std
  on the same input — the check can fail.
"""
import json
import math

import numpy as np

from experiments import pilot_b as B
from experiments import pilot_b2 as P


def test_window():
    w = B.trace_window(np.ones((50, 2)))
    assert w["sat"] == 1.0 and w["std"] == [0.0, 0.0]
    u = B.trace_window(np.random.default_rng(0).uniform(-0.5, 0.5, (4000, 2)))
    assert u["sat"] == 0.0 and all(abs(s - 0.2887) < 0.01 for s in u["std"])


def _old_row(t, acts, last):
    """The pre-11-21 row build, kept only as this test's control."""
    row = {"t": int(t), **B.trace_window(acts)}
    row.update({k: round(float(v), 5) for k, v in last.items()})
    return row


def test_row_keeps_both_std():
    acts = np.random.default_rng(1).uniform(-0.5, 0.5, (400, 2))
    last = {"std": 0.7, "actor_loss": -1.25, "ent_coef": 0.1}
    r = B.trace_row(400, acts, last)
    assert isinstance(r["std"], list) and len(r["std"]) == 2, r
    assert r["std"] == B.trace_window(acts)["std"], r
    assert r["train_std"] == 0.7 and r["actor_loss"] == -1.25, r
    assert "train_actor_loss" not in r and "train_ent_coef" not in r, r
    old = _old_row(400, acts, last)
    assert old["std"] == 0.7, "control: the old build must lose the window std"
    assert r["std"] != old["std"]


def test_reads_only():
    R = P.PILOT_RULE
    world = P.pilot_world("K", True, R["eye_frame"], R["motor_frame"], R["native_substep"])
    old = (B.WARMUP, B.TRACE_EVERY)
    B.WARMUP, B.TRACE_EVERY = 100, 200
    try:
        on = B.run_b(801, 600, world=world, trace=True)
        off = B.run_b(801, 600, world=world)
        other = B.run_b(802, 600, world=world)
    finally:
        B.WARMUP, B.TRACE_EVERY = old
    lv = lambda r: json.dumps(r["lives"] + [r["decisions"]], sort_keys=True)
    assert lv(on) == lv(off), "the trace changed what B did"
    assert lv(on) != lv(other), "control: the comparison cannot fail"
    assert "trace" not in off
    tr = on["trace"]
    assert [r["t"] for r in tr] == [200, 400, 600], tr
    for r in tr:
        assert isinstance(r["std"], list) and len(r["std"]) == len(r["mean"]), r
    for r in tr[1:]:
        for k in ("actor_loss", "critic_loss", "ent_coef"):
            assert r[k] is not None and math.isfinite(r[k]), (k, r)
        assert abs(r["ent_coef"] - 0.1) < 1e-6
    return tr


def main():
    test_window()
    test_row_keeps_both_std()
    tr = test_reads_only()
    print("trace rows:", tr)
    print("test_pilot_b_trace PASS")


if __name__ == "__main__":
    main()
