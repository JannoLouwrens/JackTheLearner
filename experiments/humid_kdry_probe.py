"""J6 order step (d) pre-declaration probe: hum34 with K_DRY re-derived. No ledger, no race verdict.

WHY: hum34 (humid_race) is the only arm since sweat whose W.2(c) hot route passes 5/5. It lost W.2(a)
(a_err_w 0.158 vs INTEGRATOR_TOL 0.01) because shipped K_DRY = M_BASAL/(T_SETPOINT - 30) no longer
pins the dry open statue at 37 C when T_DAY = 34. ORDERS J6(d) asks for "K_DRY re-derived". This probe
measures what that one re-derivation costs and buys BEFORE an arm is declared:

  hum34k = hum34 + K_DRY := M_BASAL/(T_SETPOINT - T_DAY) (= 100/3) and C_EFF := K_DRY*TAU_T
           (needs.py's own definitions, re-evaluated at the arm's T_DAY; TAU_T keeps its meaning).

  route    : humid_race._route on the 8 thermal routes (same table as humid_race check).
  W.2 cells: W.2's own _experiment on seeds given, through humid_race's apply + the patch.
  apparatus: hum34 W.2 s0 run here must equal /data/jack-logs/humid_race/hum34_W.2_s0.json.

Usage: python -m experiments.humid_kdry_probe routes | cell ARM SEED | apparatus
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import heat_race as H
from experiments import humid_race as HR

OUT = Path("/data/jack-logs/humid_kdry_probe")
ARMS = ("hum34", "hum34k")


def _apply(arm: str):
    N, mods = HR._apply("hum34")
    if arm == "hum34k":
        N.K_DRY = N.M_BASAL / (N.T_SETPOINT - N.T_DAY)
        N.C_EFF = N.K_DRY * N.TAU_T
    elif arm != "hum34":
        raise KeyError(arm)
    return N, mods


def routes() -> dict:
    out = {}
    for arm in ARMS:
        N, _ = _apply(arm)
        day, night, rest, full = N.T_DAY, N.T_DAY - N.DELTA_T_NIGHT, N.M_BASAL, 3.0 * N.M_BASAL
        r = {"K_DRY": round(N.K_DRY, 3), "C_EFF": round(N.C_EFF, 1)}
        for name, M, wet, occ, te in (
                ("rest_open_day", rest, 0, 0, day), ("rest_roof_day", rest, 0, 1, day),
                ("rest_soak_day", rest, 1, 0, day), ("full_open_day", full, 0, 0, day),
                ("full_roof_day", full, 0, 1, day), ("rest_soak_night", rest, 1, 0, night),
                ("rest_roof_night", rest, 0, 1, night), ("rest_open_night", rest, 0, 0, night)):
            r[name] = HR._route(N, M, wet, occ, te)
        out[arm] = r
    H._apply("shipped")
    return out


def cell(arm: str, seed: int) -> Path:
    _, mods = _apply(arm)
    m = mods["W.2"]._experiment(int(seed))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{arm}_W.2_s{seed}.json"
    p.write_text(json.dumps({"arm": arm, "spec": "W.2", "seed": int(seed), "metrics": m},
                            default=H._jsonable))
    return p


def apparatus() -> dict:
    a, b = OUT / "hum34_W.2_s0.json", HR.OUT / "hum34_W.2_s0.json"
    ma, mb = json.loads(a.read_text())["metrics"], json.loads(b.read_text())["metrics"]
    bad = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
    return {"compared": len(set(ma) | set(mb)), "mismatched": bad[:8], "ok": not bad}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "routes":
        print(json.dumps(routes(), indent=1))
    elif cmd == "cell":
        print(cell(sys.argv[2], int(sys.argv[3])))
    elif cmd == "apparatus":
        print(json.dumps(apparatus()))
