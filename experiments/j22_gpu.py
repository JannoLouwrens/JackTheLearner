"""J22 (2) timing cell, GPU half: arm h4 (conv eye + shift) on a free T4.

Why: the box half (`j22_timing.py`, 11-24 13:40) read 6.1 learning
decisions/s by wall but ~0.7 per CPU-second (a subtraction of two separate
runs, so each run's JIT compile is in it and the read is NOT clean); a
100k cell is 4.3 h by the wall read and much more by the CPU read. h4 is a
JAX learner with convolutions, the shape DrQ's runs had on GPU. Same job as
`j17_h0_gpu` (its pinned jax/flax/optax trio, its PILOT_RULE world: body eye,
body motor, native substep), with h4's cells:
  cuda single T = 5,000 (warmup) and 10,000 (5,000 learning) for H-conv-row128,
  then the THREE pilot cells at once at 10,000 (sighted, blind, shuffled:
  the pilot runs them on ONE machine).
Read written before it runs (`read()`): the venue is Colab T4 iff the trio's
100k estimate (warmup + 95k at the slowest cell's learning rate) <= 8 h;
otherwise the INBOX gets the numbers and the box/Kaggle CPU prices are
compared. No row, no ledger: a timing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import j17_h0_gpu as G

OUT = Path("/data/jack-logs/j22/j22_gpu_timing.json")
ARMS = ("H-conv-row128", "H-conv-row128-blind", "H-conv-row128-shuffled")
FILE = "j22_gpu_timing.json"


def job_body() -> str:
    from experiments.pilot_b import KERNEL_PINS
    body = (G._JOB.replace("__PINS__", repr(list(KERNEL_PINS)))
            .replace("__SEED__", str(G.SEED)).replace("j17_h0_gpu_timing.json", FILE)
            .replace('["__ARM__", "__BLIND__"]', repr(list(ARMS)))
            .replace('"cuda_T10000_pair"', '"cuda_T10000_trio"')
            .replace("__ARM__", ARMS[0]))
    assert "__" + "BLIND__" not in body and repr(list(ARMS)) in body
    return body


def read(r: dict) -> dict:
    out = {}
    lr = G.learn_rate(r, "cuda_T5000_single", "cuda_T10000_single", ARMS[0])
    try:
        w5 = r["cuda_T5000_single"]["result"]["wall_s"][ARMS[0]]
    except (KeyError, TypeError):
        w5 = None
    out["single"] = {"learn_dps": lr, "warmup_s": w5,
                     "est_100k_h": (w5 + 95_000 / lr) / 3600 if lr and w5 is not None else None}
    try:
        trio = r["cuda_T10000_trio"]["result"]["wall_s"]
        lr3 = 5000.0 / (max(trio.values()) - w5)
        est3 = (w5 + 95_000 / lr3) / 3600
        out["trio"] = {"cells": sorted(trio), "learn_dps": lr3, "est_100k_h": est3,
                       "fits_8h": est3 <= 8.0}
    except (KeyError, TypeError, ZeroDivisionError, ValueError):
        out["trio"] = None
    return out


def launch() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(job_body()), prefer="kaggle", est_hours=1.0,
                 timeout_s=4 * 3600, fetch=[FILE])
    if not (sub.ok and FILE in sub.artifacts):
        raise RuntimeError(f"j22 GPU timing: {sub.message}")
    r = result_json(sub, FILE)
    r["read"] = read(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["read"], indent=1), flush=True)
    return r


if __name__ == "__main__":
    if sys.argv[1:] == ["launch"]:
        launch()
    elif sys.argv[1:] == ["read"]:
        print(json.dumps(read(json.loads(OUT.read_text())), indent=1))
    elif sys.argv[1:] != ["pilot"]:
        print("usage: python -m experiments.j22_gpu launch|read|pilot")


# ---- J22 (3) the pilot itself (11-24 14:30, after the timing read: cuda single
# 54.3 learning dec/s, trio 12.5/s -> 100k ~ 2.15 h, fits 8 h -> T4 is the venue).
# Three cells on ONE machine, seed 923 (J17_HUNT_SEED), T = J17_HUNT_T (100k),
# PILOT_RULE's world: h4 sighted / blind twin / shuffled twin. Same job shape as
# j17_h0_gpu.pair_body (pinned jax trio, cuda, no XLA preallocation). Read by
# `j17_onset hunt` + `j17_late_meals(rec, 50_000)`; branch (4) is ruled in ORDERS.
PILOT_OUT = Path("/data/jack-logs/j22/j22_pilot_h4.json")


def pilot_body() -> str:
    from experiments.pilot_b2 import J17_HUNT_SEED, J17_HUNT_T, PILOT_RULE, pilot_job_body
    cells = [(a, J17_HUNT_SEED) for a in ARMS]
    rule = ("J22 h4 PILOT on GPU 11-24: conv eye + random shift (PUBLISHED_PRIORS "
            "§12/§24, DrQ), sighted/blind/shuffled, 1 seed, no verdict; world = "
            + PILOT_RULE["rule"])
    body = pilot_job_body(cells, PILOT_RULE["food_floor_only"], J17_HUNT_T, rule,
                          PILOT_RULE["eye_frame"], PILOT_RULE["motor_frame"],
                          PILOT_RULE["native_substep"])
    first = body.index("\n", body.index("check=True)")) + 1   # after the pins install
    return G._GPU_ENV + body[:first] + G._TRIO + body[first:]


def launch_pilot() -> dict:
    from experiments.gpu import build_job, result_json, submit
    sub = submit(build_job(pilot_body()), prefer="colab", est_hours=2.5,
                 timeout_s=int(3.9 * 3600), fetch=["pilot_b2_cells.json"])
    if not (sub.ok and "pilot_b2_cells.json" in sub.artifacts):
        raise RuntimeError(f"j22 h4 GPU pilot: {sub.message}")
    r = result_json(sub, "pilot_b2_cells.json")
    PILOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    PILOT_OUT.write_text(json.dumps(r, indent=1) + "\n")
    for k, rec in r["cells"].items():
        print(f"{k}: {len(rec['lives'])} lives" + (" (cut)" if rec.get("cut") else ""), flush=True)
    return r


if __name__ == "__main__" and sys.argv[1:] == ["pilot"]:
    launch_pilot()
