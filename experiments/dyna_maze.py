"""Idle-slot reproduction — Dyna maze (PUBLISHED_PRIORS §6 "Reproduce first", row 1).

What it checks: offline updates ("sleep" — learning while not acting) speed
learning, the most reliable gain on record (§6 "Sleep"). Published: Sutton &
Barto 2nd ed., Example 8.1 / Figure 8.2 — 6×9 Dyna maze, γ 0.95, α 0.1,
ε 0.1, Q₀ = 0, reward 1 at the goal, 30 repetitions; the text: "the
nonplanning agent took about 25 episodes to reach (ε-)optimal performance,
whereas the n = 5 agent took about five episodes, and the n = 50 agent took
only three episodes." Shortest path = 14 steps.

CODE, UNMODIFIED: ShangtongZhang/reinforcement-learning-an-introduction
@96bc203617a70eba6b7784a8f876aaf218b2c914, `chapter08/maze.py` — its own
`Maze`, `DynaParams`, `TrivialModel`, `dyna_q`, loaded by path from
/data/jack-src/rlai (the commit is checked before running). This file only
loops seeds and scores; it is not a learner.

Ours beside theirs: (a) the book uses 30 repetitions, the repo's
`figure_8_2` uses 10 — we use the book's 30; (b) we seed numpy per run
(`np.random.seed(run)`) so a run is reproducible, the repo seeds nothing;
(c) we also run a never-trains twin. Nothing else differs.

RULE (ruling 33's form; committed before the first run):
  seeds      runs 0–29 (np.random.seed(run) before each (run, n) pair), n ∈ {0, 5, 50},
             50 episodes each, a fresh Q and model per (run, n).
  metric     E(n) = the first episode e (1-indexed) such that the 30-run mean
             steps-per-episode is ≤ 21 (1.5 × the 14-step optimum) at e AND at
             each of the 4 episodes after it (5 in a row), i.e. "reached
             ε-optimal and stayed". Never reached → E = ∞.
  accept     REPRODUCES iff E(0) ∈ [15, 40] AND E(5) ∈ [3, 8] AND E(50) ∈ [2, 5]
             AND E(50) ≤ E(5) < E(0). (Bands are wide because the published
             numbers are prose read off a figure: "about 25 / about five / three".)
  control    never-trains twin: n = 50, α = 0 (Q never moves; the policy stays
             uniform-random-with-ties), same 30 seeds. It must have E = ∞ and
             a mean of the last 10 episodes > 100 steps, else the metric is
             VOID (a random walk would be scoring as "learned").
  extension  ONE, named now: if exactly one of the three bands misses, rerun with
             runs 30–59 and judge the pooled 60-run curve. Two or more misses →
             MISS, no extension. A miss means our harness or the reading is
             wrong: find why, never widen a band.
  compute    tabular, CPU seconds-minutes, box, nice 19.

Run:  $PY -m experiments.dyna_maze [--runs 0-29] [--out F]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np

SRC = "/data/jack-src/rlai"
COMMIT = "96bc203617a70eba6b7784a8f876aaf218b2c914"
NS = (0, 5, 50)
EPISODES = 50
THRESH = 21.0
RUN_IN_A_ROW = 5
BANDS = {0: (15, 40), 5: (3, 8), 50: (2, 5)}


def load_upstream():
    head = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"upstream at {head}, rule pins {COMMIT}")
    spec = importlib.util.spec_from_file_location("rlai_maze", f"{SRC}/chapter08/maze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def curve(mod, n: int, runs, alpha: float = 0.1) -> np.ndarray:
    maze = mod.Maze()
    steps = np.zeros((len(runs), EPISODES))
    for i, run in enumerate(runs):
        np.random.seed(run)
        p = mod.DynaParams()
        p.planning_steps = n
        p.alpha = alpha
        q = np.zeros(maze.q_size)
        model = mod.TrivialModel()
        if alpha == 0.0:
            maze.max_steps = 5000  # a random walk can be long; cap only the twin
        for ep in range(EPISODES):
            steps[i, ep] = mod.dyna_q(q, model, maze, p)
        maze.max_steps = float("inf")
    return steps


def first_settled(mean: np.ndarray) -> float:
    ok = mean <= THRESH
    for e in range(len(mean) - RUN_IN_A_ROW + 1):
        if ok[e:e + RUN_IN_A_ROW].all():
            return e + 1
    return float("inf")


def judge(E: dict, control_E: float, control_tail: float) -> str:
    if not (control_E == float("inf") and control_tail > 100):
        return "VOID"
    inband = all(BANDS[n][0] <= E[n] <= BANDS[n][1] for n in NS)
    order = E[50] <= E[5] < E[0]
    return "REPRODUCES" if inband and order else "MISSES"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="0-29")
    ap.add_argument("--out", default="/data/jack-logs/dyna_maze/result.json")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.runs.split("-"))
    runs = list(range(lo, hi + 1))
    mod = load_upstream()
    t0 = time.process_time()
    res = {"commit": COMMIT, "runs": [lo, hi], "curves": {}, "E": {}}
    for n in NS:
        m = curve(mod, n, runs).mean(0)
        res["curves"][str(n)] = m.round(2).tolist()
        res["E"][str(n)] = first_settled(m)
        print(f"n={n:>2}: E={res['E'][str(n)]}  ep1={m[0]:.0f} ep2={m[1]:.0f} ep5={m[4]:.1f} "
              f"ep25={m[24]:.1f} last10={m[-10:].mean():.1f}", flush=True)
    c = curve(mod, 50, runs, alpha=0.0).mean(0)
    res["control"] = {"E": first_settled(c), "last10": float(c[-10:].mean()),
                      "curve": c.round(1).tolist()}
    print(f"control n=50 alpha=0: E={res['control']['E']} last10={res['control']['last10']:.0f}")
    E = {n: res["E"][str(n)] for n in NS}
    res["verdict"] = judge(E, res["control"]["E"], res["control"]["last10"])
    res["cpu_s"] = round(time.process_time() - t0, 1)
    print("VERDICT", res["verdict"], f"cpu {res['cpu_s']} s")
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
