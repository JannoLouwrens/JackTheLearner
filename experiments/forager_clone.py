"""forager_clone.py — J15: the learnability proof, part 2 (ORDERS J15).

PLAN stage 1, part "Learning to eat", line "a copy of it inside his own
network, and a learner given the direction to food". Step 2 -> 3. Prior:
behaviour cloning a scripted policy into a small MLP is a known-answer task;
a learner handed the bearing to food is the record's standard "is it the eye
or the learner" control. Nothing here is a part of him: the forager is an
adult's ruler, the clone is a probe of whether HIS network shape can hold it.

ORDERS: "Not before J12 (2) has a body-frame row. Until then: write the two
rules and the data collector only." This file is that and nothing else —
no clone is trained, no E2c run starts, until J12 gate 2 is in WORLD_CARD.
If gate 2 FAILS at d0 this order becomes "say why" and E2b/E2c do not run.

=============================================================================
RULE E2b — the forager copied into brain B's actor network.
Written 2026-11-18 09:3x UTC, before any data is collected or any clone runs.
=============================================================================
Teacher  gate 2's row f4: `eye_forager.EyeForager` unchanged, in
         `eye_forager.make_env(seed, 0.75, row="f4")` (body-frame row128 eye,
         body-frame motor, lazy_contact_audio, native_substep, d0).
Data     `collect`: the teacher's own lives on TRAINING seeds 701..740 (fresh:
         no row, pilot or LAB has used 7xx), 3 lives a seed, life cap 9,000;
         every decision stores (row = o[:256], teacher action a[0:8]) plus,
         as a DIAGNOSTIC ONLY (never a clone input), whether the teacher held
         a target that decision.
Student  brain B's actor shape as the J11 pilot builds it (SBX SAC
         MlpPolicy, net_arch [256, 256] library default): an MLP 256 -> 256
         -> 256 -> 8, tanh output, trained from scratch by mean-squared error
         on the teacher's actions. Input: the body-frame row (256 numbers)
         ONLY — no proprio, no needs, no memory (ORDERS). Adam 1e-3, batch
         256, 20 epochs, 10% held out for the loss report; 3 training
         seeds (clone seeds 0, 1, 2) — the clone is judged per clone seed.
Test     each clone drives the body on FRESH seeds 741..760, d0, 10 lives a
         seed, cap 9,000 — gate 2's bookkeeping (`eye_forager.run_seed`'s
         loop) with the clone as the policy.
Bar      UNMOVED, gate 2's: mean floor meals/life >= 5.0 AND > b4's mean on
         the same seed (white random, rerun on 741..760), on >= 18 of 20
         seeds. E2b PASSES iff >= 2 of the 3 clone seeds pass.
Control  MUST FAIL: the same student trained on the same data with the 128
         (distance, texture) pairs of every input permuted by a fresh
         permutation each sample (the g4 twin's corruption, at training and
         at test). If it passes on >= 2 of 3 clone seeds, E2b is VOID.
Disclosed before the run: the teacher is STATEFUL (a dead-reckoned target,
         bans, cooldowns, a wander hold of 25 decisions); a memoryless clone
         of it can only copy the part of its policy the current row
         determines. A miss with the target-held fraction high is "the
         forager's memory, not the eye" — and says brain B needs memory
         (§5's recurrent rows), not that the eye is blind. That reading is
         written now so a miss cannot be explained afterwards.
Extension  NONE.

=============================================================================
RULE E2c — brain B given the direction to food.
=============================================================================
World    the J11 pilot's arm-K world (food_floor_only, spawn_clear 1.55, d0,
         drain ramp K, body-frame eye AND motor — the default proposed in
         INBOX 11-18 07:40 unless the Director rules otherwise first),
         native_substep on.
Learner  the pilot's B0 SAC unchanged, its observation extended by TWO
         numbers: the bearing (body frame, radians / pi) and the distance
         (metres / 10, clipped to 1) to the nearest uneaten floor food,
         computed from the simulator — a handed sense, scored-and-
         ineligible, never his.
Size     T = 300,000 decisions, 5 seeds (seeds 801..805), J11's frozen
         readings (sustained / eats but starves / slides) and J11's floor
         clause, read from PILOT_B2.md's rule unchanged.
Control  MUST FAIL: the same arm with the two numbers replaced by the bearing
         and distance of a random fixed point re-drawn each life.
Reading  pass (sustained on >= 3 of 5, control < 3) = the task is learnable
         by B when the direction is given -> a miss of B0 in J11 is the eye.
         miss = B cannot learn the task even told where food is -> the
         learner or the world, not the eye. Reported, decides no seat.
Extension  NONE. E2c's code (the two-number wrapper) is written when J11's
         rule is frozen — it inherits that rule's readings verbatim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import eye_forager as EF  # noqa: E402

TRAIN_SEEDS = tuple(range(701, 741))
TEST_SEEDS = tuple(range(741, 761))
TRAIN_LIVES = 3
D0 = 0.75
N_IN = 2 * EF.N_ROW                       # the row: 128 (distance, texture)
OUT = Path("/data/jack-logs/forager_clone")


def collect_seed(seed: int, lives: int = TRAIN_LIVES, cap: int = 9000,
                 d: float = D0) -> dict:
    """The teacher f4's own lives on one training seed: (row, action, held)."""
    env = EF.make_env(seed, d, row="f4")
    o, info = env.reset(seed=seed)
    pol = EF.make_policy("f4", seed)
    X, A, H, meals = [], [], [], []
    for _life in range(lives):
        pol.new_life()
        ate0 = dict(info["ate_total"])
        n, term = 0, False
        while n < cap:
            row = np.asarray(o[:N_IN], dtype=np.float32).copy()
            a = np.asarray(pol.act(o), dtype=np.float32)
            X.append(row)
            A.append(a.copy())
            H.append(pol.target is not None)
            o, _r, term, _tr, info = env.step(a)
            n += 1
            if term:
                break
        ate = info["ate_total"]
        meals.append(int(ate.get("obj0", 0) - ate0.get("obj0", 0)
                         + ate.get("obj1", 0) - ate0.get("obj1", 0)))
        if not term:
            env.w.respawn()
            env._pending_obs = None
        o, info = env.reset()
    return {"x": np.stack(X), "a": np.stack(A), "held": np.array(H, bool),
            "meals": np.array(meals, int), "seed": seed}


def collect(seeds=TRAIN_SEEDS, out: Path = OUT, lives: int = TRAIN_LIVES) -> list:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seeds:
        r = collect_seed(s, lives)
        p = out / f"teacher_s{s}.npz"
        np.savez_compressed(p, **r)
        print(f"seed {s}: {len(r['x'])} decisions, meals/life {r['meals'].tolist()}, "
              f"target held {r['held'].mean():.3f}", flush=True)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# E2b: the student, its training, and its test (written 2026-11-18 12:xx,
# after gate 2 PASSED at d0, before any teacher data existed on disk).
# ---------------------------------------------------------------------------
CLONE_SEEDS = (0, 1, 2)
ARMS = ("clone", "perm")                  # perm = the control that must fail
HIDDEN = (256, 256)                       # SBX SAC MlpPolicy default actor
EPOCHS, BATCH, LR, HOLDOUT = 20, 256, 1e-3, 0.10
TEST_LIVES = 10
WEIGHTS = REPO / "experiments" / "forager_clone_weights"


def _mlp():
    import torch.nn as nn
    h1, h2 = HIDDEN
    return nn.Sequential(nn.Linear(N_IN, h1), nn.ReLU(), nn.Linear(h1, h2),
                         nn.ReLU(), nn.Linear(h2, 8), nn.Tanh())


def permute_pairs(x: np.ndarray, rng) -> np.ndarray:
    """The g4 twin's corruption: the 128 (distance, texture) pairs of each
    row reordered by a fresh permutation per row. x: (B, 256) or (256,)."""
    one = x.ndim == 1
    xb = x.reshape(-1, EF.N_ROW, 2)
    idx = np.argsort(rng.random(xb.shape[:2]), axis=1)
    out = np.take_along_axis(xb, idx[:, :, None], axis=1).reshape(-1, N_IN)
    return out[0] if one else out


def load_teacher(data_dir: Path = OUT, seeds=TRAIN_SEEDS):
    X, A = [], []
    for s_ in seeds:
        z = np.load(Path(data_dir) / f"teacher_s{s_}.npz")
        X.append(z["x"])
        A.append(z["a"])
    return np.concatenate(X).astype(np.float32), np.concatenate(A).astype(np.float32)


def train_clone(X: np.ndarray, A: np.ndarray, clone_seed: int, arm: str,
                epochs: int = EPOCHS) -> tuple:
    """MSE behaviour cloning, from scratch. arm 'perm' permutes every input
    afresh every time it is seen. Returns (state_dict as numpy, report)."""
    import torch
    torch.manual_seed(clone_seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(10_000 + clone_seed)
    n = len(X)
    order = rng.permutation(n)
    n_hold = int(round(HOLDOUT * n))
    hold, tr = order[:n_hold], order[n_hold:]
    net = _mlp()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = torch.nn.MSELoss()
    Xh = X[hold] if arm == "clone" else permute_pairs(X[hold], rng)
    Xh_t, Ah_t = torch.from_numpy(Xh), torch.from_numpy(A[hold])
    rep = {"arm": arm, "clone_seed": clone_seed, "n_train": int(len(tr)),
           "n_hold": int(n_hold), "hold_mse": []}
    for _ep in range(epochs):
        net.train()
        ep_order = rng.permutation(tr)
        for i in range(0, len(ep_order), BATCH):
            b = ep_order[i:i + BATCH]
            xb = X[b] if arm == "clone" else permute_pairs(X[b], rng)
            loss = lossf(net(torch.from_numpy(xb)), torch.from_numpy(A[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            rep["hold_mse"].append(round(float(lossf(net(Xh_t), Ah_t)), 6))
    sd = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    return sd, rep


class ClonePolicy:
    """The trained student as a policy: body-frame row in, action out. No
    memory (new_life does nothing); 'perm' sees a freshly permuted row."""

    def __init__(self, weights: dict, arm: str, seed: int):
        import torch
        torch.set_num_threads(1)
        self.net = _mlp()
        self.net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()})
        self.net.eval()
        self.arm = arm
        self.rng = np.random.default_rng(seed)
        self.target = None                 # collector/diagnostic interface

    def new_life(self):
        pass

    def act(self, o):
        import torch
        x = np.asarray(o[:N_IN], dtype=np.float32)
        if self.arm == "perm":
            x = permute_pairs(x, self.rng).astype(np.float32)
        with torch.no_grad():
            return self.net(torch.from_numpy(x)[None])[0].numpy().astype(np.float32)


def weights_path(arm: str, clone_seed: int, root: Path = WEIGHTS) -> Path:
    return Path(root) / f"{arm}_s{clone_seed}.npz"


def test_cell(arm: str, clone_seed: int, seed: int, lives: int = TEST_LIVES,
              cap: int = 9000, root: Path = WEIGHTS) -> list:
    """One test seed for one student (or the b4 floor when arm == 'b4'),
    through gate 2's own life loop in f4's world."""
    if arm == "b4":
        res = EF.run_seed("b4", seed, D0, lives=lives, cap=cap)
    else:
        w = dict(np.load(weights_path(arm, clone_seed, root)))
        pol = ClonePolicy(w, arm, seed)
        res = EF.run_seed("f4", seed, D0, lives=lives, cap=cap, policy=pol)
    for r in res:
        r["row"], r["arm"], r["clone_seed"] = f"{arm}{clone_seed}", arm, clone_seed
    return res


def test_cells() -> list:
    """Every (arm, clone_seed, seed) of the E2b test; b4 once per seed."""
    cells = [(a, k, s_) for a in ARMS for k in CLONE_SEEDS for s_ in TEST_SEEDS]
    return cells + [("b4", 0, s_) for s_ in TEST_SEEDS]


def score(cells: dict) -> dict:
    """The frozen bar per student, then E2b's verdict (>= 2 of 3 clone seeds
    pass; VOID if the permuted control passes on >= 2 of 3)."""
    from experiments import world_card as WC
    by = {}
    for rec in cells.values():
        by.setdefault(rec["row"], []).extend(rec["lives"])
    b = by.get("b40", [])
    for lv in b:
        lv["row"] = "b4"
    out = {}
    for a in ARMS:
        for k in CLONE_SEEDS:
            lv = [dict(x, row=f"{a}{k}") for x in by.get(f"{a}{k}", [])]
            out[f"{a}{k}"] = WC.score(lv, b, TEST_SEEDS)
    npass = {a: sum(bool(out[f"{a}{k}"]["passes"]) for k in CLONE_SEEDS) for a in ARMS}
    verdict = ("VOID" if npass["perm"] >= 2 else
               "PASS" if npass["clone"] >= 2 else "FAIL")
    return {"students": out, "passing": npass, "verdict": verdict}


# ---------------------------------------------------------------------------
# E2b test kernel (11-18 12:4x, before any student was trained): one Kaggle
# CPU session runs test_cells() — 6 students x 20 seeds + b4 x 20 = 140
# cells of 10 lives — through the same per-cell subprocess pool gate 2 used.
# ---------------------------------------------------------------------------
KERNEL_EST_HOURS = 6.0       # gate 2: 80 cells in 3.25 h; 140 cells, worst case all f4-long
TEST_OUT = OUT / "test.json"

_JOB = r'''
import subprocess as _s, sys as _y
_s.run([_y.executable, "-m", "pip", "install", "-q"] + __PINS__, check=True)
import json, os
os.chdir("/tmp/jack")
from experiments.forager_clone import run_pool, test_cells
r = run_pool(test_cells(), os.environ["JACK_OUT"])
json.dump({"cells": r}, open(os.path.join(os.environ["JACK_OUT"], "clone_test.json"), "w"))
print("DONE", flush=True)
'''


def run_pool(cells: list, outdir, workers: int = EF.WORKERS) -> dict:
    """Each (arm, clone_seed, seed) cell in its own `cell` subprocess; failed
    cells are kept with their error, never dropped."""
    import subprocess
    import time
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    todo, live, res = list(cells), [], {}
    while todo or live:
        while todo and len(live) < workers:
            arm, k, seed = todo.pop(0)
            f = outdir / f"{arm}{k}_s{seed}.jsonl"
            err = open(str(f) + ".err", "w")
            live.append((arm, k, seed, f, err, subprocess.Popen(
                [sys.executable, "-m", "experiments.forager_clone", "cell", arm,
                 str(k), str(seed), str(f)],
                cwd=REPO, stdout=subprocess.DEVNULL, stderr=err)))
        time.sleep(2)
        for item in list(live):
            arm, k, seed, f, err, p = item
            if p.poll() is None:
                continue
            live.remove(item)
            err.close()
            lives = ([json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                     if f.exists() else [])
            rec = {"row": f"{arm}{k}", "seed": seed, "lives": lives, "rc": p.returncode}
            if p.returncode != 0:
                rec["error"] = Path(str(f) + ".err").read_text()[-800:]
            res[f"{arm}{k}_s{seed}"] = rec
            print(f"{arm}{k} s{seed}: rc {p.returncode}, {len(lives)} lives", flush=True)
    return res


def weights_ready(root: Path = WEIGHTS) -> list:
    """Reasons the kernel must not launch: a student file missing, or present
    but not committed (the kernel clones the pushed HEAD, not this disk)."""
    import subprocess
    bad = []
    for a in ARMS:
        for k in CLONE_SEEDS:
            w = weights_path(a, k, root)
            if not w.exists():
                bad.append(f"missing {w.name}")
                continue
            r = subprocess.run(["git", "ls-files", "--error-unmatch", str(w)],
                               cwd=REPO, capture_output=True)
            if r.returncode != 0:
                bad.append(f"untracked {w.name}")
    return bad


def kernel() -> dict:
    from experiments.gpu import build_job, result_json, submit
    bad = weights_ready()
    if bad:
        raise RuntimeError("E2b test kernel refused: " + "; ".join(bad))
    body = _JOB.replace("__PINS__", repr(EF.KERNEL_PINS))
    sub = submit(build_job(body), prefer="kaggle-cpu", est_hours=KERNEL_EST_HOURS,
                 timeout_s=EF.REMOTE_TIMEOUT_S, fetch=["clone_test.json"])
    if not sub.ok:
        raise RuntimeError(f"forager_clone test kernel: {sub.message}")
    r = result_json(sub, "clone_test.json")
    TEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUT.write_text(json.dumps(r, indent=1) + "\n")
    return r


def main(argv):
    if argv and argv[0] == "train":            # train [epochs] — all 6 students
        ep = int(argv[1]) if len(argv) > 1 else EPOCHS
        X, A = load_teacher()
        print(f"teacher: {len(X)} decisions", flush=True)
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            for k in CLONE_SEEDS:
                sd, rep = train_clone(X, A, k, arm, epochs=ep)
                np.savez_compressed(weights_path(arm, k), **sd)
                print(json.dumps(rep), flush=True)
        return 0
    if argv and argv[0] == "cell":             # cell <arm> <clone_seed> <seed> <out>
        from experiments.sim_speed_kcpu import _compat_multiray
        _compat_multiray()
        res = test_cell(argv[1], int(argv[2]), int(argv[3]))
        with open(argv[4], "a") as fh:
            fh.write("\n".join(json.dumps(r) for r in res) + "\n")
        return 0
    if argv and argv[0] == "kernel":           # kernel — the E2b test, one Kaggle CPU session
        r = kernel()
        print(json.dumps({k: len(v["lives"]) for k, v in r["cells"].items()}))
        return 0
    if argv and argv[0] == "score":            # score [test.json]
        p = Path(argv[1]) if len(argv) > 1 else TEST_OUT
        print(json.dumps(score(json.loads(p.read_text())["cells"]), indent=1))
        return 0
    if not argv or argv[0] not in ("collect", "smoke"):
        print("usage: forager_clone.py collect [seed ...] | smoke | train [epochs] | cell ... | kernel | score [file]")
        return 2
    if argv[0] == "smoke":
        r = collect_seed(153, lives=1, cap=300)
        print(f"smoke: x {r['x'].shape} a {r['a'].shape} held {r['held'].mean():.3f}")
        return 0
    seeds = tuple(int(s) for s in argv[1:]) or TRAIN_SEEDS
    collect(seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
