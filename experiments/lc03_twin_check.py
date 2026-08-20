"""LC.03 rig re-derivation check — is the frozen twin's +158 s life_gain the
exploration schedule, and does a constant std kill it?

THE FAULT UNDER TEST (attempt 1 VOID, 2026-08-14, control (c) fired):
run_survival's exploration std decays linearly over the RUN (0.5 -> 0.1,
frac = k/n_decisions), and drives price actuator power at up to 3x basal
drain. So every `policy="core"` run — learning or not — gets quieter and
therefore longer-lived as the run proceeds: the life_gain ruler reads the
schedule, not learning. Predicts exactly what the VOID run measured:
untrained twins +158..190 s (tight seed spread: schedule-driven), random
null +6.5 +/- 12 (constant activity, stationary), statue +0.01 (stationary).

TWO ARMS, PRE-REGISTERED BARS (set before launch, 2026-08-20; the SM.02
learnability-check pattern — a 10-minute CPU answer before a 15-hour run):
  "schedule"  dreamer-xs twin, seed 90, pilot envelope (12k decisions,
              e0=0.3), the CURRENT schedule (0.5, 0.1).
              REPRODUCE bar: life_gain >= +20.0 s (the e0=1.0 run read
              +158; scaled by e0=0.3 the expectation is ~47 s; the null's
              seed noise scales to ~4 s).
  "constant"  same twin, same seed, same envelope, explore_std=(0.3, 0.3)
              (the old schedule's time-mean, so total exploration matches).
              FIX bar: |life_gain| <= 10.0 s.
BOTH bars met -> the constant-std repair in lc_03_survival_screening.py
stands; relaunch the registered run. REPRODUCE fails -> the diagnosis is
wrong; STOP — do not relaunch, re-derive again. FIX fails while REPRODUCE
passes -> a second nonstationarity exists; find it before any relaunch.
"""
import json
import os
import sys
import time

os.chdir("/home/opc/jackthelearner")
sys.path.insert(0, "/home/opc/jackthelearner")

from experiments.protocol import borrow_metrics          # noqa: E402
from experiments.survival import run_survival            # noqa: E402

SEED = 90                      # the pilot's seed
N_DEC = 12_000                 # the pilot envelope
E0 = 0.3
ARM = "dreamer-xs"             # the twin the VOID named: +158.4 +/- 2.0 s
OUT = "/data/lc03_twin_check.json"
BAR_REPRODUCE = 20.0
BAR_FIX = 10.0


def main() -> None:
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    assert b.ok, f"PS.01 borrow refused: {b.refusal}"
    j0, alpha = b.values["j0_ms"], b.values["alpha"]
    print(f"borrowed j0={j0} alpha={alpha}", flush=True)

    res: dict = {"seed": SEED, "n_decisions": N_DEC, "e0": E0, "arm": ARM,
                 "bars": {"reproduce_min": BAR_REPRODUCE,
                          "fix_abs_max": BAR_FIX}}
    for name, std in (("schedule", (0.5, 0.1)), ("constant", (0.3, 0.3))):
        t0 = time.time()
        r = run_survival(SEED, j0=j0, alpha=alpha, e0=E0,
                         n_decisions=N_DEC, policy="core", arm=ARM,
                         train=False, explore_std=std)
        spans = r["life_spans"]
        third = len(spans) // 3
        res[name] = {
            "explore_std": list(std),
            "life_gain": r["life_gain"],
            "mean_life_s": r["mean_life_s"],
            "mean_life_first_third": (sum(spans[:third]) / third
                                      if third else 0.0),
            "mean_life_last_third": (sum(spans[-third:]) / third
                                     if third else 0.0),
            "n_lives": r["n_lives"],
            "life_spans": spans,
            "wall_s": round(time.time() - t0, 1),
        }
        print(f"{name}: life_gain {r['life_gain']:+.2f} s  "
              f"mean_life {r['mean_life_s']:.1f} s  "
              f"n_lives {r['n_lives']:.0f}  "
              f"({res[name]['wall_s']:.0f} s wall)", flush=True)
        with open(OUT, "w") as f:      # partial write survives a kill
            json.dump(res, f, indent=1)

    res["verdict"] = {
        "reproduced": bool(res["schedule"]["life_gain"] >= BAR_REPRODUCE),
        "fixed": bool(abs(res["constant"]["life_gain"]) <= BAR_FIX),
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("verdict:", res["verdict"], flush=True)


if __name__ == "__main__":
    main()
