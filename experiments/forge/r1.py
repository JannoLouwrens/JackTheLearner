"""Round 1 of the forge (F3 / ORDERS J18, ruling 69): the KEEP round on E-food.

The rule itself is `experiments/forge/rounds/r1/spec.json`; its commit is the
freeze. This file only holds what the spec's numbers were DERIVED from, so a
test can re-derive them (tests/test_forge_r1_spec.py):

  control_draw(n)   plain variation near the baseline, the box and generator
                    FORGE_POOL.md states for C1/C2 (`random.Random(
                    "forge-control-<n>")`), drawn in the order lr, batch, tau,
                    warm-up.
  reset_points(T)   K1's resets for a rung of T decisions: every RESET_EVERY
                    decisions, strictly inside the run.

Nothing here trains anything.
"""
from __future__ import annotations

import math
import random

RESET_EVERY = 50_000   # K1: >= 1 reset inside rung S (J18). 50k = just over the
                       # slowest first capped life in E2c (18-47k), and before
                       # E2c's earliest loss (last capped life 81-189k).
BOX = {"lr": (1.5e-4, 6e-4), "batch_size": (128, 256, 512),
       "tau": (0.0025, 0.01), "learning_starts": (2500, 5000, 10000)}


def control_draw(n: int) -> dict:
    r = random.Random(f"forge-control-{n}")
    lr = math.exp(r.uniform(math.log(BOX["lr"][0]), math.log(BOX["lr"][1])))
    batch = r.choice(BOX["batch_size"])
    tau = math.exp(r.uniform(math.log(BOX["tau"][0]), math.log(BOX["tau"][1])))
    warm = r.choice(BOX["learning_starts"])
    return {"lr": float(f"{lr:.6g}"), "batch_size": batch,
            "tau": float(f"{tau:.6g}"), "learning_starts": warm}


def reset_points(T: int, every: int = RESET_EVERY) -> list:
    return list(range(every, int(T), every))
