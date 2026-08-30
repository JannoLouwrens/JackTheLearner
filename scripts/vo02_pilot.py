#!/usr/bin/env python
"""VO.02's learnability + envelope pilot.

Answers the only two questions that decide whether VO.02 is dispatchable:
  1. LEARNABILITY — does a pair separated by a real acoustic channel get off
     chance at the provisional 600x64 envelope? If not, the envelope is wrong
     and the budget declaration would be a guess.
  2. THROUGHPUT — measured wall-clock per arm, so the TIER is declared from
     measurement and never from the registry's optimism (T3.06's precedent).

It also produces the RIG arms (`muted`, `scrambled`) the provisional bars are
allowed to be frozen from. The trained arm's numbers are recorded and
disclosed, and NO bar may be moved to fit them.

Detached-safe: cwd and sys.path are pinned before any repo import.
"""
import json
import os
import sys
import time

REPO = "/home/opc/jackthelearner"
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import experiments.tests.vo_02_two_jacks_signal as M  # noqa: E402

SEED = int(os.environ.get("VO02_SEED", "0"))
OUT = os.environ.get("VO02_OUT", "/data/vo02_pilot_seed%d.json" % SEED)

res = {"seed": SEED, "n_updates": M.N_UPDATES, "batch": M.BATCH,
       "episodes_per_arm": M.N_UPDATES * M.BATCH, "n_eval": M.N_EVAL,
       "n_cic": M.N_CIC, "n_perm": M.N_PERM, "timings_s": {}}


def stage(name, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    res["timings_s"][name] = round(dt, 1)
    res.update(out)
    print("[%7.1fs] %-10s %s" % (dt, name,
                                 {k: round(v, 4) for k, v in out.items()}),
          flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)


print("VO.02 pilot seed=%d  %d episodes/arm  -> %s" %
      (SEED, M.N_UPDATES * M.BATCH, OUT), flush=True)

# the rig instruments first: if the estimator or the channel is dead here,
# nothing downstream is worth the wall-clock.
stage("urn", lambda: M._urn_game(SEED))
stage("probe", lambda: M._probe(SEED))
stage("level", lambda: M._level(SEED))

# the nulls before the claim, deliberately: the bars this pilot is allowed to
# freeze come from these, and reading them first means they were not chosen
# with the claim arm's number already on the screen.
stage("muted", lambda: M._arm(SEED, "muted"))
stage("untrained", lambda: M._arm(SEED, "untrained"))
stage("scrambled", lambda: M._arm(SEED, "scrambled"))
stage("trained", lambda: M._arm(SEED, "trained"))

res["total_s"] = round(sum(res["timings_s"].values()), 1)
res["projected_3seed_h"] = round(res["total_s"] * 3 / 3600.0, 2)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, sort_keys=True)
print("TOTAL %.1f s ; projected 3-seed registered run %.2f h"
      % (res["total_s"], res["projected_3seed_h"]), flush=True)
print("chance=%.3f  trained_coord=%.4f  muted_coord=%.4f  scrambled_coord=%.4f"
      % (M.CHANCE, res["trained_coord"], res["muted_coord"],
         res["scrambled_coord"]), flush=True)
print("mi_ear=%.4f floor=%.4f | cic=%.4f floor=%.4f"
      % (res["trained_mi_ear"], res["trained_mi_perm_p95"],
         res["trained_cic"], res["trained_cic_perm_p95"]), flush=True)
