"""SM.02 pre-pilot-2 CPU learnability check (LESSONS rule, 2026-08-20).

Trains the gate-critical no-smell arms at FULL production budget with the
potential-based shaping repair, on CPU, and prints trained-vs-random eval —
the two numbers that decide whether pilot 2 is worth its GPU hours:
  nosmell/vis  must approach LEARN_VIS_FRAC (0.6) * random
  nosmell/occ  must approach LEARN_OCC_FRAC (0.85) * random
Usage: python /home/opc/jackthelearner/experiments/sm02_learnability_check.py {vis|occ}
(absolute path on purpose — the chdir/sys.path pin below makes the launch
cwd irrelevant, which `python -m` cannot: -m resolves the package BEFORE any
line of this file runs. LESSONS: detached scripts launched from /data died
at import three times before this pin existed.)
"""
import os
import sys

REPO = "/home/opc/jackthelearner"
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import json
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.tests import sm_02_smell_finds_occluded as sm

cond = sys.argv[1]
seed = 90
print(f"START cond={cond} seed={seed} n_train={sm.N_TRAIN} "
      f"pid={os.getpid()}", flush=True)
rig = sm._Rig(seed, occluded=(cond == "occ"))
t0 = time.time()
net = sm._train_arm(rig, "nosmell", seed, sm.N_TRAIN, torch, nn, "cpu")
train_s = time.time() - t0
ev = sm._eval_arm(rig, net, "nosmell", seed, sm.N_EVAL,
                  torch=torch, dev="cpu")
rnd = sm._eval_arm(rig, None, "nosmell", seed, sm.N_EVAL,
                   rng_random=np.random.RandomState(seed * 41))
out = {"cond": cond, "seed": seed, "train_s": round(train_s, 1),
       "trained": ev, "random": rnd,
       "ratio": ev["t_mean"] / rnd["t_mean"]}
print(json.dumps(out, indent=1))
with open(f"/data/sm02_learnability_{cond}.json", "w") as f:
    json.dump(out, f, indent=1)
