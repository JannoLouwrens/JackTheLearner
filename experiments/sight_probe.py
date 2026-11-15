"""Sight probe (OUTSIDER 11-15 E1): can his seated senses locate floor food?

LAB, not a spec: writes no ledger row. Run by J1 to decide what the forager
race (J1 step 2/3) is measuring before step 3 spends 4x the lives on it.

RULES, declared before any cell (commit before the run):
  data      W0(seed, lethal=False). Random policy, each action held for
            HOLD=5 decisions (0.2 s each). The body is re-spawned uniformly on
            a legal cell every RESPAWN=40 decisions so positions cover the arena.
  target    world-frame direction (cos, sin) from the body to the NEAREST
            floor food (obj0/obj1, `w.drives._food`) — rays are world-frame
            (`w0.py:219`), so a world-frame target is what sight could carry.
  inputs    (A) vision only (32), (B) the whole observation minus language and
            placebo (vision+audio+touch+proprio+needs).
  fit       train on seed 153, test on seed 154 (a different mutated world).
            Ridge (lambda chosen on a 20% split of TRAIN only) and a 2x64 MLP
            (torch, 400 Adam epochs, fixed seed). Score = held-out R2 averaged
            over cos and sin.
  arms      rays as seated (body origin + 0.05 m) vs rays from z = 0.12 m.
  control   the same fit with the target rows SHUFFLED within the train set
            must score R2 <= 0.05 on test (else the probe leaks).
  pre-run   (11-15 13:4x, 200 decisions, s153) the body origin is 0.65 m only
            at spawn; it settles to median 0.068 m (p10 .066, p90 .164), so the
            seated rays fire at ~0.12 m most of a life and hit floor food on
            10% of decisions. The OUTSIDER's "0 of 615" read unsettled spawns.
  reading   (OUTSIDER's prediction) seated R2 near 0, low rays > 0.5.
            seated best R2 > 0.3 -> sight already carries food direction; the
            OUTSIDER's section 1 is wrong and the brain race stands.
            seated best R2 < 0.1 and low-ray > 0.3 -> the race is run blind;
            report to the Director with these numbers.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "disabled")

import numpy as np  # noqa: E402

from experiments import w0 as W0MOD  # noqa: E402
from experiments.w0 import W0, random_action  # noqa: E402

HOLD = 5
RESPAWN = 40
LOW_Z = 0.12
KEYS = ("vision", "audio", "touch", "proprio", "needs")


def _w(seed: int) -> W0:
    from experiments.forager_race import _cal  # the race's own j0/alpha
    cal = _cal()
    return W0(seed=seed, j0=cal["j0"], alpha=cal["alpha"], lethal=False)


def collect(seed: int, n: int, low: bool) -> dict:
    w = _w(seed)
    if low:
        orig = W0MOD.RAY_Z_OFFSET

        def _vision_low(self=w):
            z = float(self.data.xpos[self.rover_bid][2])
            W0MOD.RAY_Z_OFFSET = LOW_Z - z
            try:
                return W0._vision(self)
            finally:
                W0MOD.RAY_Z_OFFSET = orig
        w._vision = _vision_low
    rng = np.random.RandomState(seed + 7)
    gids = [w.drives._food[k][0] for k in ("obj0", "obj1") if k in w.drives._food]
    X, Y, D, HIT = [], [], [], 0
    a = random_action(rng)
    for t in range(n):
        if t % RESPAWN == 0:
            w.respawn()
        if t % HOLD == 0:
            a = random_action(rng)
        obs = w.observe()
        xy = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
        best = None
        for g in gids:
            v = np.array(w.data.geom_xpos[g][:2], dtype=float) - xy
            if best is None or np.linalg.norm(v) < np.linalg.norm(best):
                best = v
        r = float(np.linalg.norm(best)) + 1e-9
        X.append(np.concatenate([obs[k] for k in KEYS]).astype(np.float32))
        Y.append([best[0] / r, best[1] / r])
        D.append(r)
        HIT += int(_ray_hits(w, gids, low))
        w.decide(a)
    return {"X": np.array(X), "Y": np.array(Y, dtype=np.float32),
            "D": np.array(D), "hit_frac": HIT / n}


def _ray_hits(w, gids, low) -> bool:
    """Does any of the 16 rays (as the arm casts them) hit a floor food geom?"""
    pos = np.array(w.data.xpos[w.rover_bid], dtype=float)
    pos[2] = LOW_Z if low else pos[2] + W0MOD.RAY_Z_OFFSET
    gid = np.zeros(1, dtype=np.int32)
    for k in range(W0MOD.N_RAYS):
        d = w.mujoco.mj_ray(w.model, w.data, pos, w._ray_dirs[k],
                            W0MOD._RAY_GROUPS, 1, -1, gid)
        if 0 <= d <= W0MOD.RAY_MAX_RANGE and int(gid[0]) in gids:
            return True
    return False


def r2(yt, yp):
    ss = ((yt - yp) ** 2).sum(0)
    st = ((yt - yt.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss / np.maximum(st, 1e-12)))


def ridge(Xtr, Ytr, Xte, Yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    k = int(0.8 * len(A))
    best = (None, -1e9)
    for lam in (0.1, 1, 10, 100, 1000):
        W = np.linalg.solve(A[:k].T @ A[:k] + lam * np.eye(A.shape[1]), A[:k].T @ Ytr[:k])
        s = r2(Ytr[k:], A[k:] @ W)
        if s > best[1]:
            best = (lam, s)
    lam = best[0]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
    return r2(Yte, B @ W)


def mlp(Xtr, Ytr, Xte, Yte):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(2)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    T = torch.tensor(Ytr)
    net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(400):
        opt.zero_grad()
        loss = ((net(A) - T) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return r2(Yte, net(B).numpy())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    out = {"n_per_seed": n, "hold": HOLD, "respawn": RESPAWN, "low_z": LOW_Z}
    t0 = time.time()
    for arm, low in (("seated", False), ("low_0.12", True)):
        tr, te = collect(153, n, low), collect(154, n, low)
        nv = 32
        res = {"ray_hits_food_frac_train": tr["hit_frac"],
               "ray_hits_food_frac_test": te["hit_frac"],
               "median_dist_m": float(np.median(tr["D"]))}
        for name, sl in (("vision", slice(0, nv)), ("all", slice(None))):
            res[f"ridge_{name}"] = ridge(tr["X"][:, sl], tr["Y"], te["X"][:, sl], te["Y"])
            res[f"mlp_{name}"] = mlp(tr["X"][:, sl], tr["Y"], te["X"][:, sl], te["Y"])
        perm = np.random.RandomState(1).permutation(len(tr["Y"]))
        res["control_shuffled_mlp_all"] = mlp(tr["X"], tr["Y"][perm], te["X"], te["Y"])
        out[arm] = res
        print(arm, json.dumps(res), f"{time.time() - t0:.0f}s", flush=True)
    path = os.environ.get("SIGHT_OUT", "/data/tmp/sight_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
