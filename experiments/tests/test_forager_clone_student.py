"""J15 E2b: the student, its trainer, its test cell and its scorer (11-18, before any clone ran).

Known answers, each with a control that must fail:
- permute_pairs keeps each (distance, texture) pair together and only reorders them
  (control: the identity row is unchanged by no permutation, changed by one).
- the trainer learns a planted mapping: on actions that are a fixed function of the row,
  the clone arm's held-out MSE falls far below the permuted arm's (which cannot see
  where anything is).
- the test cell drives f4's world through gate 2's loop with the clone as policy and
  labels the lives by student.
- score(): planted cells give PASS / FAIL / VOID exactly as the frozen rule says."""
import tempfile
from pathlib import Path

import numpy as np

from experiments import forager_clone as C


def test_permute_pairs():
    rng = np.random.default_rng(0)
    x = np.stack([np.arange(256, dtype=np.float32)] * 4)
    y = C.permute_pairs(x, rng)
    assert y.shape == x.shape
    for r in y:
        p = r.reshape(128, 2)
        assert np.all(p[:, 1] == p[:, 0] + 1) and np.all(p[:, 0] % 2 == 0)   # pairs intact
        assert sorted(r.tolist()) == list(range(256))
    assert not np.array_equal(y[0], x[0])                                   # control: it moved
    assert not np.array_equal(y[0], y[1])                                   # fresh per row
    one = C.permute_pairs(x[0], rng)
    assert one.shape == (256,)


def test_trainer_learns_planted_map():
    rng = np.random.default_rng(1)
    X = rng.random((6000, 256)).astype(np.float32)
    W = rng.normal(size=(256, 8)).astype(np.float32) / 8
    A = np.tanh(X @ W - (X @ W).mean(0)).astype(np.float32)
    _, rc = C.train_clone(X, A, 0, "clone", epochs=6)
    _, rp = C.train_clone(X, A, 0, "perm", epochs=6)
    assert rc["hold_mse"][-1] < 0.5 * rp["hold_mse"][-1], (rc["hold_mse"], rp["hold_mse"])
    assert rc["hold_mse"][-1] < rc["hold_mse"][0]


def test_cell_runs_student_in_f4_world():
    X = np.random.default_rng(2).random((512, 256)).astype(np.float32)
    A = np.zeros((512, 8), np.float32)
    sd, _ = C.train_clone(X, A, 1, "clone", epochs=1)
    with tempfile.TemporaryDirectory() as d:
        np.savez_compressed(C.weights_path("clone", 1, Path(d)), **sd)
        res = C.test_cell("clone", 1, 741, lives=1, cap=40, root=Path(d))
    assert len(res) == 1 and res[0]["row"] == "clone1" and res[0]["decisions"] <= 40
    assert res[0]["seed"] == 741


def _cells(meals: dict) -> dict:
    cells = {}
    for row, m in meals.items():
        cells[row] = {"row": row, "lives": [{"seed": s, "floor_meals": m, "row": row}
                                            for s in C.TEST_SEEDS for _ in range(2)]}
    return cells


def test_score_planted():
    base = {"b40": 0}
    good = {f"clone{k}": 6 for k in C.CLONE_SEEDS}
    bad = {f"perm{k}": 1 for k in C.CLONE_SEEDS}
    assert C.score(_cells({**base, **good, **bad}))["verdict"] == "PASS"
    miss = {f"clone{k}": (6 if k == 0 else 4) for k in C.CLONE_SEEDS}
    assert C.score(_cells({**base, **miss, **bad}))["verdict"] == "FAIL"
    leaky = {f"perm{k}": 6 for k in C.CLONE_SEEDS}                       # control passes
    assert C.score(_cells({**base, **good, **leaky}))["verdict"] == "VOID"


def main():
    test_permute_pairs()
    test_trainer_learns_planted_map()
    test_cell_runs_student_in_f4_world()
    test_score_planted()
    print("test_forager_clone_student PASS")


if __name__ == "__main__":
    main()
