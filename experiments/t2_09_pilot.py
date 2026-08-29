"""T2.09 pilot — seed 90, the grid that freezes the bars.

Runs BOTH candidate arms on BOTH worlds so `_CLAIM_ARM` is chosen here, before
any official seed, and the registered run only tests the named arm on unseen
seeds 0/1/2. Writes /data/t2_09_pilot_seed90.json.

Detached-script rule (LESSONS): a /data-rooted setsid script dies at import
unless it pins chdir + sys.path itself.
"""
import json
import os
import sys
import time

REPO = "/home/opc/jackthelearner"
os.chdir(REPO)
sys.path.insert(0, REPO)

from experiments.tests import t2_09_noisy_tv_control as T  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 90
OUT = f"/data/t2_09_pilot_seed{SEED}.json"

GRID = [("icm", True), ("rnd", True), ("disagree", True),
        ("zero", True), ("random", True),
        ("rnd", False), ("disagree", False), ("zero", False)]

res, t_all = {}, time.time()
for arm, noisy in GRID:
    t0 = time.time()
    m = T._life(SEED, arm, noisy)
    m["wall_s"] = round(time.time() - t0, 1)
    res[f"{arm}_{'noisy' if noisy else 'static'}"] = m
    print(f"{arm:9} {'noisy' if noisy else 'static':6} {m['wall_s']:7.1f}s  "
          f"dwell={m['dwell_share']:.4f} cov={m['coverage']:.4f} "
          f"ratio={m['panel_reward_ratio']:.3f} decay={m['reward_decay']:.3f}",
          flush=True)
    json.dump({"seed": SEED, "n_decisions": T.N_DECISIONS, "arms": res,
               "total_wall_s": round(time.time() - t_all, 1)},
              open(OUT, "w"), indent=2)

print(f"\nDONE {round(time.time() - t_all, 1)}s -> {OUT}", flush=True)
