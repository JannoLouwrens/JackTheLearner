#!/usr/bin/env python
"""T2.14 rig probe — is imitation-from-real-mocap a WELL-POSED task here?

Not a claim and not a gate. It answers one question before any gate is frozen:
for each candidate action target, are the two registry nulls (mean-action,
1-NN retrieval) INFORMATIVE, and does a linear readout already solve it? A
target that ridge nails is a target that measures arithmetic, not imitation.

Splits are CLIP-DISJOINT. Splitting by frame would let 1-NN retrieve the
adjacent frame of the same clip, which is leakage wearing a null's clothes.
"""
import os
import sys

REPO = "/home/opc/jackthelearner"
os.chdir(REPO)
sys.path.insert(0, REPO)

import json
from pathlib import Path

import numpy as np

CACHE = Path("/data/jack-data/retarget-cache")
DT = 0.02
KP, KD = 10.0, 1.0


def load_clips(limit=None):
    """(pos, vel) per cached clip, newest-format only, clip identity kept."""
    out = []
    for f in sorted(CACHE.glob("cmu_*.npz")):
        d = np.load(f)
        if not {"positions", "velocities"} <= set(d.files):
            continue
        p, v = d["positions"].astype(np.float64), d["velocities"].astype(np.float64)
        if p.ndim != 2 or p.shape[1] != 17 or len(p) < 4:
            continue
        out.append((f.stem, p, v))
        if limit and len(out) >= limit:
            break
    return out


def build(clips, target):
    """obs = (pos[t-1], vel[t-1]); label depends on `target`. Clip ids returned
    so the split can be clip-disjoint."""
    X, Y, G = [], [], []
    for i, (name, p, v) in enumerate(clips):
        o = np.concatenate([p[:-1], v[:-1]], axis=1)          # (T-1, 34)
        if target == "pd":            # the SHIPPED loader's action
            y = np.clip(KP * (p[1:] - p[:-1]) - KD * v[:-1], -0.4, 0.4)
        elif target == "nextpose":    # the loader's other documented mode
            y = p[1:]
        elif target == "delta":       # next pose relative to current
            y = p[1:] - p[:-1]
        else:
            raise ValueError(target)
        X.append(o); Y.append(y); G.append(np.full(len(o), i))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def zstats(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1e-6
    return mu, sd


def nn_mse(Xtr, Ytr, Xte, Yte):
    mu, sd = zstats(Xtr)
    A = ((Xtr - mu) / sd).astype(np.float32)
    B = ((Xte - mu) / sd).astype(np.float32)
    a2 = (A * A).sum(1)
    pred = np.empty_like(Yte)
    for i in range(0, len(B), 512):
        b = B[i:i + 512]
        d = a2[None, :] - 2.0 * (b @ A.T)
        pred[i:i + 512] = Ytr[d.argmin(1)]
    return float(((pred - Yte) ** 2).mean())


def ridge_mse(Xtr, Ytr, Xte, Yte, grid=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)):
    mu, sd = zstats(Xtr)
    A = ((Xtr - mu) / sd)
    B = ((Xte - mu) / sd)
    val = np.arange(len(A)) % 5 == 0
    fit = ~val

    def solve(X, Y, l2):
        return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ Y)

    ym = Ytr[fit].mean(0)
    best, best_l2 = np.inf, None
    for l2 in grid:
        W = solve(A[fit], Ytr[fit] - ym, l2)
        e = float(((A[val] @ W + ym - Ytr[val]) ** 2).mean())
        if e < best:
            best, best_l2 = e, l2
    ym = Ytr.mean(0)
    W = solve(A, Ytr - ym, best_l2)
    return float(((B @ W + ym - Yte) ** 2).mean()), best_l2


def main():
    clips = load_clips()
    print(f"clips: {len(clips)}  frames: {sum(len(p) for _, p, _ in clips)}")
    rng = np.random.RandomState(0)
    order = rng.permutation(len(clips))
    n_tr = int(0.75 * len(clips))
    tr_ids, te_ids = set(order[:n_tr].tolist()), set(order[n_tr:].tolist())

    report = {"n_clips": len(clips), "targets": {}}
    for target in ("pd", "nextpose", "delta"):
        X, Y, G = build(clips, target)
        m_tr = np.isin(G, list(tr_ids))
        m_te = ~m_tr
        Xtr, Ytr, Xte, Yte = X[m_tr], Y[m_tr], X[m_te], Y[m_te]
        # subsample train for the NN null's O(n*m) — declared, not hidden
        sub = np.random.RandomState(1).choice(len(Xtr), min(20000, len(Xtr)),
                                              replace=False)
        Xtr_s, Ytr_s = Xtr[sub], Ytr[sub]
        sub_te = np.random.RandomState(2).choice(len(Xte), min(4000, len(Xte)),
                                                 replace=False)
        Xte_s, Yte_s = Xte[sub_te], Yte[sub_te]

        mse_mean = float(((Ytr.mean(0) - Yte_s) ** 2).mean())
        mse_nn = nn_mse(Xtr_s, Ytr_s, Xte_s, Yte_s)
        mse_ridge, l2 = ridge_mse(Xtr_s, Ytr_s, Xte_s, Yte_s)
        sat = float((np.abs(Y) >= 0.4 - 1e-9).mean()) if target == "pd" else 0.0

        row = {
            "n_train_frames": int(m_tr.sum()), "n_test_frames": int(m_te.sum()),
            "mse_mean": round(mse_mean, 6),
            "mse_nn": round(mse_nn, 6),
            "mse_ridge": round(mse_ridge, 6),
            "ridge_l2": l2,
            "nn_over_mean": round(mse_nn / mse_mean, 4),
            "ridge_over_nn": round(mse_ridge / mse_nn, 4),
            "saturated_frac": round(sat, 4),
            "label_std": round(float(Y.std()), 6),
        }
        report["targets"][target] = row
        print(f"\n--- target={target}")
        for k, v in row.items():
            print(f"    {k:18s} {v}")
    Path("/data/t214_probe.json").write_text(json.dumps(report, indent=1))
    print("\nwrote /data/t214_probe.json")


if __name__ == "__main__":
    main()
