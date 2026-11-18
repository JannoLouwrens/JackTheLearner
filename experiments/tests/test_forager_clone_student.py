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


def test_kernel_pieces():
    """The E2b test kernel: 140 distinct cells; refuses missing weights, and
    refuses present-but-uncommitted ones with a different reason (the pass
    case — tracked files — is not exercised until the students exist); the
    job body compiles; a failing cell is kept with its error, not dropped."""
    cells = C.test_cells()
    assert len(cells) == len(set(cells)) == 6 * 20 + 20
    with tempfile.TemporaryDirectory() as d:
        bad = C.weights_ready(Path(d))
        assert len(bad) == 6 and all(b.startswith("missing") for b in bad)
        for a in C.ARMS:
            for k in C.CLONE_SEEDS:
                np.savez_compressed(C.weights_path(a, k, Path(d)), w=np.zeros(1))
        bad = C.weights_ready(Path(d))
        assert len(bad) == 6 and all(b.startswith("untracked") for b in bad)
    compile(C._JOB.replace("__PINS__", "[]"), "job", "exec")
    with tempfile.TemporaryDirectory() as d:
        r = C.run_pool([("clone", 9, 741)], d, workers=1)       # no clone_s9 weights
    rec = r["clone9_s741"]
    assert rec["rc"] != 0 and rec["lives"] == [] and "clone_s9" in rec["error"]


def test_full_kernel_pieces():
    """The FULL kernel (collect + manifest check + train + test): the hash
    sees the bytes (identical arrays match, one changed action differs —
    the control); a missing seed is reported, not skipped; the manifest
    writer refuses an incomplete collection; with no committed manifest
    the kernel refuses; the job compiles; the subprocess path collect-to ->
    train-one -> cell (with a weights root) runs end to end on a tiny cap."""
    import subprocess
    import sys
    rng = np.random.default_rng(3)
    z = {"x": rng.random((50, 256)).astype(np.float32),
         "a": rng.random((50, 8)).astype(np.float32)}
    with tempfile.TemporaryDirectory() as d:
        np.savez_compressed(Path(d) / "teacher_s701.npz", **z)
        z2 = {"x": z["x"], "a": z["a"].copy()}
        z2["a"][7, 3] += 1e-6
        np.savez_compressed(Path(d) / "teacher_s702.npz", **z2)
        man = {"701": {"sha256": C.teacher_hash(z), "n": 50},
               "702": {"sha256": C.teacher_hash(z), "n": 50},
               "703": {"sha256": C.teacher_hash(z), "n": 50}}
        chk = C.check_manifest(d, man, seeds=(701, 702, 703))
        assert chk["701"] == "match" and chk["702"].startswith("differs") and chk["703"] == "missing"
        try:
            C.write_manifest(Path(d))
            raise AssertionError("manifest written from 2 of 40 seeds")
        except RuntimeError as e:
            assert "38 teacher seeds missing" in str(e)
    if not C.TEACHER_MANIFEST.exists():
        assert C.full_ready() and "no teacher manifest" in C.full_ready()[0]
        try:
            C.kernel_full()
            raise AssertionError("full kernel launched without a manifest")
        except RuntimeError as e:
            assert "refused" in str(e)
    compile(C._JOB_FULL.replace("__PINS__", "[]"), "job", "exec")
    py, m = sys.executable, "experiments.forager_clone"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "w").mkdir()
        r = C.collect_seed(701, lives=1, cap=30)          # real teacher, tiny cap
        np.savez_compressed(d / "teacher_s701.npz", **r)
        rc = subprocess.run([py, "-c", (
            "import sys; from experiments import forager_clone as C;"
            "C.load_teacher.__defaults__ = (C.OUT, (701,));"
            "C.train_clone.__defaults__ = (1,);"
            "sys.exit(C.main(['train-one', 'clone', '0', sys.argv[1], sys.argv[2]]))"),
            str(d), str(d / "w")], cwd=C.REPO).returncode
        assert rc == 0
        assert C.weights_path("clone", 0, d / "w").exists() and (d / "w" / "clone_s0.json").exists()
        res = C.test_cell("clone", 0, 741, lives=1, cap=40, root=d / "w")
        assert len(res) == 1 and res[0]["row"] == "clone0"
        # run_pool hands its root to the cell subprocess: weights absent there,
        # present in the default root's absence too -> the error names THIS root
        (d / "empty_root").mkdir()
        cells = C.run_pool([("clone", 0, 741)], d / "cells", workers=1, root=d / "empty_root")
    rec = cells["clone0_s741"]
    assert rec["rc"] != 0 and "empty_root" in rec["error"], rec


def main():
    test_permute_pairs()
    test_trainer_learns_planted_map()
    test_cell_runs_student_in_f4_world()
    test_score_planted()
    test_kernel_pieces()
    test_full_kernel_pieces()
    print("test_forager_clone_student PASS")


if __name__ == "__main__":
    main()
