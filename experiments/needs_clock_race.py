"""J6(b) — the needs-clock race: shipped W0 needs vs the declared k = 72 clock.

Ruling 1 (ORDERS J6, 11-13): W0 has ONE declared clock, k = 72. Thermal obeys
it; hunger, thirst and the two sleep taus do not (W.7 smoke s0, 2685dd9).
No world file is edited until an arm WINS this race.

Arms (constants patched in-process on `experiments.needs`; every reader takes
them as module globals at call time):
  shipped  B_E 1/1800  B_W 1/450   TAU_WAKE 700  TAU_SLEEP 160
  k72      B_E 1/25200 B_W 1/3600  TAU_WAKE 910  TAU_SLEEP 210

DECLARED RULE (written before any run, seeds 0-4):
  k72 ships iff (1) W.2 check (c) passes on every seed (c_pass == 1), and
  (2) on every NE.01 `ok_*` clause its seed-mean is >= shipped's seed-mean
  (no worse), and (3) NE.01's pooled max cause share is no higher than
  shipped's. Decisions-per-life of NE.01's random lives is REPORTED for both
  arms (the cost), not gated.

APPARATUS NOTE (found by arithmetic, before running): W.2's harness horizons
are sized from the SHIPPED constants — DEPRIVED_HORIZON_S = 3000 s ("≫ any
single-need deadline in W0") is below k72's thirst deadline 3720 s and hunger deadline 25500 s (the
watered life in W.2(c) runs on the same horizon), and
WAKE_FIT_S 3500 s is 5 x tau_wake(700) but only 3.8 x 910. Unpatched, W.2
would truncate the k72 lives and score the arm FAIL for the harness's sake.
Per arm the horizons are therefore scaled to the arm's own constants
(deprived = 1.5 x the longer single-need deadline; wake fit = 5 x tau_wake;
sleep fit = 7.5 x tau_sleep). Bars are untouched. W.2 itself needs the same
re-registration before k72 can ship (say so in the commit that ships it).

Usage (nice 19, <= 2 workers; ~40 min/NE.01 seed on this box per the
11-13 ledger row, so run detached):
  python -m experiments.needs_clock_race run  ARM SPEC SEED   # one cell -> /data/jack-logs/needs_race/
  python -m experiments.needs_clock_race tally
  python -m experiments.needs_clock_race check                # patch + horizon sanity, seconds
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

OUT = Path("/data/jack-logs/needs_race")
SEEDS = (0, 1, 2, 3, 4)
ARMS = {
    "shipped": {"B_E": 1 / 1800.0, "B_W": 1 / 450.0, "TAU_WAKE": 700.0, "TAU_SLEEP": 160.0},
    "k72": {"B_E": 1 / 25200.0, "B_W": 1 / 3600.0, "TAU_WAKE": 910.0, "TAU_SLEEP": 210.0},
}
SPECS = {"W.2": "experiments.tests.w_2_needs_ledger",
         "NE.01": "experiments.tests.ne_01_nobody_survives_by_accident"}


def _apply(arm: str):
    N = importlib.import_module("experiments.needs")
    for k, v in ARMS[arm].items():
        setattr(N, k, v)
    mods = {s: importlib.reload(importlib.import_module(p)) for s, p in SPECS.items()}
    w2 = mods["W.2"]
    # W.2 drives BOTH the deprived (thirst) and the watered (hunger) life on
    # DEPRIVED_HORIZON_S, so it must cover the longer of the two deadlines.
    deadline = max(1.0 / N.B_W + N.DEATH_W_S, 1.0 / N.B_E + N.DEATH_E_S)
    w2.DEPRIVED_HORIZON_S = max(3000.0, 1.5 * deadline)
    w2.WAKE_FIT_S = max(3500.0, 5.0 * N.TAU_WAKE)
    w2.SLEEP_FIT_S = max(1200.0, 7.5 * N.TAU_SLEEP)
    return N, mods


def check() -> dict:
    out = {}
    for arm in ARMS:
        N, mods = _apply(arm)
        w2 = mods["W.2"]
        out[arm] = {"B_E": N.B_E, "B_W": N.B_W, "TAU_WAKE": N.TAU_WAKE,
                    "TAU_SLEEP": N.TAU_SLEEP,
                    "thirst_deadline_s": 1 / N.B_W + N.DEATH_W_S,
                    "hunger_deadline_s": 1 / N.B_E + N.DEATH_E_S,
                    "deprived_horizon_s": w2.DEPRIVED_HORIZON_S,
                    "wake_fit_s": w2.WAKE_FIT_S, "sleep_fit_s": w2.SLEEP_FIT_S}
    return out


def run_cell(arm: str, spec: str, seed: int) -> Path:
    _, mods = _apply(arm)
    m = mods[spec]._experiment(int(seed))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{arm}_{spec}_s{seed}.json"
    p.write_text(json.dumps({"arm": arm, "spec": spec, "seed": int(seed),
                             "constants": ARMS[arm], "metrics": m},
                            default=lambda o: getattr(o, "tolist", lambda: str(o))()))
    return p


def tally() -> dict:
    cells = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    res: dict = {}
    for arm in ARMS:
        ne = [c["metrics"] for c in cells if c["arm"] == arm and c["spec"] == "NE.01"]
        w2 = [c["metrics"] for c in cells if c["arm"] == arm and c["spec"] == "W.2"]
        r = {"n_ne01": len(ne), "n_w2": len(w2),
             "w2_c_pass_all": bool(w2) and all(m.get("c_pass") == 1.0 for m in w2)}
        if ne:
            keys = sorted({k for m in ne for k in m if k.startswith("ok_")})
            r["ok"] = {k: sum(float(m.get(k, 0)) for m in ne) / len(ne) for k in keys}
            causes = [k for k in ne[0] if k.startswith("deaths_") and "microsleep" not in k]
            tot = {k: sum(float(m[k]) for m in ne) for k in causes}
            s = sum(tot.values())
            r["max_cause_share"] = max(tot.values()) / s if s else 1.0
            r["random_death_dec_min"] = min(float(m["death_dec_min"]) for m in ne)
            r["random_death_dec_max"] = max(float(m["death_dec_max"]) for m in ne)
        res[arm] = r
    a, b = res.get("shipped", {}), res.get("k72", {})
    complete = all(res[x]["n_ne01"] == len(SEEDS) and res[x]["n_w2"] == len(SEEDS) for x in res)
    if complete:
        no_worse = all(b["ok"].get(k, 0) >= v for k, v in a["ok"].items())
        res["k72_ships"] = bool(b["w2_c_pass_all"] and no_worse
                                and b["max_cause_share"] <= a["max_cause_share"])
    else:
        res["k72_ships"] = None   # incomplete: no verdict
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "run":
        print(run_cell(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    elif cmd == "tally":
        print(json.dumps(tally(), indent=1))
    else:
        print(json.dumps(check(), indent=1))
