"""Needs-hidden known-answer row (PUBLISHED_PRIORS §18 item 4; ORDERS Next-up 2).

RULE — written and committed BEFORE any cell runs (ruling 33's form).

Source, read from the primary text this time (arXiv e-print 2204.06608,
`rldm.tex` + figures; Dulberg, Dubey, Berwian & Cohen, RLDM 2022 — the
short version of the PNAS 2023 paper): a grid of N = 4 overlaid resource
maps, each a 2D Gaussian (Sigma = I) centred on one corner; the agent sees a
3x3 egocentric slice of all 4 maps plus its 4 internal stats h; actions =
4 cardinal moves; each step h_i += resource_i at the NEXT cell, and every
h_i decays by 0.004; set-points 5, start 0.5, centre start; ONE life of
30,000 steps, learning online. Reward (HRRL, Keramati & Gutkin 2014):
r = D(H_t) - D(H_t+1), D = (sum |h* - h|^n)^(1/m), (n, m) = (4, 2).
DQN: MLP ReLU, Adam lr 1e-3, gamma 0.5, buffer 30k, target sync 200,
batch 512, one update per env step, eps 1 -> 0.01 linear over A steps.
GmQ: one DQN per stat, each rewarded by its own D(h_i), greedy action =
argmax_a sum_i Q_i(a). Score: Delta = mean over t in [15k, 30k) of
sum_i |h* - h_i|  (lower is better).

Published numbers — read off the paper's FIGURES (box plots, n = 50), so
every band is our reading, +-0.1:
  Fig 3a (eps-annealing A):      DQN median [IQR]        GmQ median [IQR]
     A = 1                        3.2  [1.9, 6.0]         0.6  [0.5, 0.7]
     A = 5000                     0.95 [0.75, 1.35]       0.5  [0.42, 0.6]
     A = 10000                    0.8  [0.7, 0.95]        0.6  [0.5, 0.75]
  Fig 2b (DQN, gamma 0.5):        1.1  [0.9, 1.55]  (A not stated)
The paper has NO needs-hidden arm. The needs-hidden prediction is EVAAA's
(NeurIPS 2025, DreamerV3, four needs: the needs-hidden arm fails to learn;
figures only, not verified) carried to this grid — that part is TEST, not
REPRODUCE, and is said so wherever it is reported.

Ours beside theirs (every setting not in the text is ours, and listed):
| setting | published | ours |
|---|---|---|
| grid size | not in text; Table 1 means at {0,10}; Fig 1a draws 9x9 walkable inside a 1-cell wall (measured off the PNG: cells ~24.5 px, panel ~251 px) | 11x11 map, coordinates 0..10, outer ring = wall (no resource, cannot enter), walkable 9x9 |
| start cell | "centre" | (5, 5) |
| "normalized" Gaussian | pdf or sum-normalised — unstated | the pdf: N(x; mu, I) = exp(-d^2/2)/(2 pi) on every cell incl. walls (walls read 0 in the view) |
| view beyond the map | unstated | zeros |
| hidden layers | "1024" / "500" units; 1.09e6 params | two hidden layers: DQN 40-1024-1024-4 (1,095,684), GmQ 4 x 40-500-500-4 (1,092,016) — the only depth matching the stated count |
| TD loss | "standard" | Huber (smooth L1), as Mnih 2013/2015 |
| updates before buffer fills | unstated | start once the buffer holds one batch (512) |
| stat bounds | unstated | none (h may go negative or above h*) |
| device | unstated | CUDA when present (T4 on Kaggle); box smoke = 117 s per 1,500 DQN steps on one ARM core, so a 30k CPU run is ~55 min and the 60 runs are a GPU kernel |
| seeds | n = 50 | n = 10 per cell (CPU budget); bands are judged on medians, which 10 seeds estimate to ~+-0.15 at these IQRs (inference) |

Cells (seeds 0-9 each, all at gamma 0.5):
  dqn@5000, gmq@5000            — R1, the reproduction
  dqn@1,    gmq@1               — R2, the exploration row
  hidden@5000                   — R3, DQN with the 4 stats replaced by zeros in
                                  its input (width kept); reward unchanged
  frozen@5000                   — CONTROL: DQN with lr = 0 (never trains)

Verdict, frozen now:
  CONTROL: frozen@5000 median Delta > 5.0, else the whole row is VOID (a Delta
    the untrained net reaches cannot show learning).
  R1 PASS iff dqn@5000 median in [0.65, 1.45] AND gmq@5000 median in
    [0.32, 0.70] AND gmq@5000 median < dqn@5000 median. (Each band = the
    figure's IQR widened by the 0.1 reading error.)
  R2 PASS iff dqn@1 median >= 1.8 AND gmq@1 median <= 0.80 (direction +
    the published quartile on the far side, widened by 0.1).
  R3 (TEST, reported): EVAAA's prediction "needs-hidden fails" is MET iff
    hidden@5000 median > 1.45 (above R1's DQN band) AND hidden > dqn@5000 on
    >= 8 of 10 seed pairs. Otherwise it is a named miss: "on this grid a
    memoryless DQN without its stats still holds homeostasis to Delta = x".
    R3 is read only if R1 PASSES; if R1 fails, R3 is reported, not read.
  No extension seeds; no threshold moves after the first cell.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

N_RES = 4
SIZE = 11
MUS = [(10, 10), (10, 0), (0, 0), (0, 10)]   # Fig 1a: res1 bottom-right, res2 bottom-left, res3 top-left, res4 top-right (row, col)
H_STAR = 5.0
H0 = 0.5
DECAY = 0.004
N_EXP, M_EXP = 4, 2
STEPS = 30_000
T1, T2 = 15_000, 30_000
GAMMA = 0.5
LR = 1e-3
BUF = 30_000
TARGET_SYNC = 200
BATCH = 512
EPS0, EPS1 = 1.0, 0.01
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# NH-1 addendum pilot only (not the frozen row): the figure's peak sits on the walkable corner
MUS_INSIDE = [(9, 9), (9, 1), (1, 1), (1, 9)]
CELLS = {
    "dqn@5000": ("dqn", 5000), "gmq@5000": ("gmq", 5000),
    "dqn@1": ("dqn", 1), "gmq@1": ("gmq", 1),
    "hidden@5000": ("hidden", 5000), "frozen@5000": ("frozen", 5000),
}
SEEDS = list(range(10))
OUT = "/data/jack-logs/needs_grid_row"


class Grid:
    """Dulberg's homeostatic grid. obs = 3x3x4 view (36) + 4 stats = 40."""

    def __init__(self, mus=None):
        rr, cc = np.mgrid[0:SIZE, 0:SIZE]
        self.res = np.stack([np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / 2.0) / (2 * np.pi)
                             for r, c in (mus or MUS)], -1)
        self.wall = np.zeros((SIZE, SIZE), bool)
        self.wall[0, :] = self.wall[-1, :] = self.wall[:, 0] = self.wall[:, -1] = True
        self.res[self.wall] = 0.0
        self.pos = (SIZE // 2, SIZE // 2)
        self.h = np.full(N_RES, H0)

    def view(self):
        r, c = self.pos
        v = np.zeros((3, 3, N_RES))
        for i in range(3):
            for j in range(3):
                y, x = r + i - 1, c + j - 1
                if 0 <= y < SIZE and 0 <= x < SIZE:
                    v[i, j] = self.res[y, x]
        return v.ravel()

    def obs(self):
        return np.concatenate([self.view(), self.h]).astype(np.float32)

    def step(self, a):
        r, c = self.pos
        y, x = r + MOVES[a][0], c + MOVES[a][1]
        if not self.wall[y, x]:
            self.pos = (y, x)
        h0 = self.h.copy()
        self.h = self.h + self.res[self.pos] - DECAY
        return h0, self.h.copy()


def drive(h):
    return float(np.sum(np.abs(H_STAR - h) ** N_EXP) ** (1.0 / M_EXP))


def drive_each(h):
    return np.abs(H_STAR - h) ** (N_EXP / M_EXP)


def mlp(width, hidden):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 4))


def run_one(kind: str, anneal: int, seed: int, steps: int = STEPS, t1: int = T1, t2: int = T2,
            mus=None) -> dict:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    env = Grid(mus)
    k = N_RES if kind == "gmq" else 1
    hidden = 500 if kind == "gmq" else 1024
    nets = [mlp(40, hidden).to(dev) for _ in range(k)]
    tgts = [mlp(40, hidden).to(dev) for _ in range(k)]
    for n, t in zip(nets, tgts):
        t.load_state_dict(n.state_dict())
    params = [p for n in nets for p in n.parameters()]
    opt = torch.optim.Adam(params, lr=0.0 if kind == "frozen" else LR)
    S = np.zeros((BUF, 40), np.float32); S2 = np.zeros((BUF, 40), np.float32)
    A = np.zeros(BUF, np.int64); R = np.zeros((BUF, k), np.float32)
    hide = kind == "hidden"

    def seen(o):
        if hide:
            o = o.copy(); o[36:] = 0.0
        return o

    o = seen(env.obs())
    devs = []
    for t in range(steps):
        eps = EPS1 if anneal <= 1 else max(EPS1, EPS0 + (EPS1 - EPS0) * t / anneal)
        if rng.random() < eps:
            a = int(rng.integers(4))
        else:
            with torch.no_grad():
                q = sum(n(torch.from_numpy(o)[None].to(dev)) for n in nets)
            a = int(q.argmax())
        h0, h1 = env.step(a)
        if kind == "gmq":
            r = drive_each(h0) - drive_each(h1)
        else:
            r = np.array([drive(h0) - drive(h1)])
        o2 = seen(env.obs())
        i = t % BUF
        S[i], A[i], R[i], S2[i] = o, a, r, o2
        o = o2
        if t1 <= t < t2:
            devs.append(float(np.abs(H_STAR - h1).sum()))
        n_have = min(t + 1, BUF)
        if n_have >= BATCH and kind != "frozen":
            idx = rng.integers(n_have, size=BATCH)
            s, s2 = torch.from_numpy(S[idx]).to(dev), torch.from_numpy(S2[idx]).to(dev)
            aa = torch.from_numpy(A[idx])[:, None].to(dev)
            rr = torch.from_numpy(R[idx]).to(dev)
            loss = 0.0
            for j, (n, tg) in enumerate(zip(nets, tgts)):
                qsa = n(s).gather(1, aa).squeeze(1)
                with torch.no_grad():
                    y = rr[:, j] + GAMMA * tg(s2).max(1).values
                loss = loss + F.smooth_l1_loss(qsa, y)
            opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % TARGET_SYNC == 0:
            for n, tg in zip(nets, tgts):
                tg.load_state_dict(n.state_dict())
    return {"kind": kind, "anneal": anneal, "seed": seed, "steps": steps,
            "delta": float(np.mean(devs)) if devs else None, "h_final": env.h.tolist()}


def judge(runs: list) -> dict:
    by = {}
    for r in runs:
        by.setdefault(f"{r['kind']}@{r['anneal']}", {})[r["seed"]] = r["delta"]
    med = {c: float(np.median(list(v.values()))) for c, v in by.items()}
    out = {"medians": med, "n": {c: len(v) for c, v in by.items()}}
    if any(len(by.get(c, {})) < len(SEEDS) for c in CELLS):
        out["verdict"] = "INCOMPLETE"
        return out
    if med["frozen@5000"] <= 5.0:
        out["verdict"] = "VOID (control reached Delta <= 5.0)"
        return out
    r1 = 0.65 <= med["dqn@5000"] <= 1.45 and 0.32 <= med["gmq@5000"] <= 0.70 and med["gmq@5000"] < med["dqn@5000"]
    r2 = med["dqn@1"] >= 1.8 and med["gmq@1"] <= 0.80
    wins = sum(by["hidden@5000"][s] > by["dqn@5000"][s] for s in SEEDS)
    r3 = med["hidden@5000"] > 1.45 and wins >= 8
    out.update(R1="PASS" if r1 else "FAIL", R2="PASS" if r2 else "FAIL",
               R3=("MET" if r3 else "MISS") + ("" if r1 else " (reported, not read: R1 failed)"),
               R3_seed_pairs=f"{wins}/10", verdict="SCORED")
    return out


_KERNEL_BODY = r'''
import json, sys, time
from experiments.needs_grid_row import CELLS, SEEDS, run_one
plan = __PLAN__
runs = []
for cell in plan:
    kind, anneal = CELLS[cell]
    for s in SEEDS:
        t0 = time.time(); r = run_one(kind, anneal, s); r["wall_s"] = round(time.time() - t0, 1)
        runs.append(r); print("CELL", cell, s, r["delta"], r["wall_s"], flush=True)
        json.dump({"runs": runs}, open(__TAG__ + ".json", "w"))
print("DONE", flush=True)
'''


def kernel_body(cells: list, tag: str) -> str:
    for c in cells:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c}; known: {list(CELLS)}")
    return _KERNEL_BODY.replace("__PLAN__", repr(list(cells))).replace("__TAG__", repr(tag))


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "smoke":   # NOT a cell: 1,500 steps, Delta window [500, 1500); timing only
        for kind in ("dqn", "gmq", "hidden", "frozen"):
            t0 = time.time(); r = run_one(kind, 1000, 99, steps=1500, t1=500, t2=1500)
            print("SMOKE", kind, round(r["delta"], 3), f"{time.time() - t0:.1f}s", flush=True)
        return 0
    if cmd == "pilot" and len(argv) > 2:   # NOT the row: NH-1 addendum, mu ON the walkable corner
        seed = int(argv[2])
        t0 = time.time(); r = run_one("dqn", 5000, seed, mus=MUS_INSIDE); r["wall_s"] = round(time.time() - t0, 1)
        r["mus"] = MUS_INSIDE
        os.makedirs(OUT, exist_ok=True)
        json.dump(r, open(os.path.join(OUT, f"pilot_mu_inside_dqn5000_s{seed}.json"), "w"))
        print("PILOT", seed, r["delta"], r["h_final"], r["wall_s"], flush=True)
        return 0
    if cmd == "score":
        import glob
        runs = []
        for p in sorted(glob.glob(os.path.join(OUT, "needs_grid_*.json"))):
            runs += json.load(open(p)).get("runs", [])
        print(json.dumps(judge(runs), indent=1))
        return 0
    if cmd == "kernel" and len(argv) > 2:   # e.g. kernel dqn@5000 gmq@5000 frozen@5000
        from experiments.gpu import build_job, result_json, submit
        cells = argv[2:]
        tag = "needs_grid_" + "_".join(c.replace("@", "") for c in cells)
        body = kernel_body(cells, tag)
        os.makedirs(OUT, exist_ok=True)
        sub = submit(build_job(body), prefer="kaggle", est_hours=0.3 + 0.35 * len(cells),
                     timeout_s=int(3600 * (1 + 0.5 * len(cells))), fetch=[tag + ".json"])
        print("SUBMIT", sub.backend, sub.ok, sub.job_id, sub.message, flush=True)
        if not sub.ok:
            return 1
        dst = os.path.join(OUT, tag + ".json")
        open(dst, "w").write(json.dumps(result_json(sub, tag + ".json")))
        print("WROTE", dst, flush=True)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
