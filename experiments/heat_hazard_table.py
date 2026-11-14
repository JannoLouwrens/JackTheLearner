"""J6(c) pre-declaration measurement: which heat hazard CAN kill a sweating Jack in W0?

No ledger row, no world file edited. Pure arithmetic on `experiments.needs`'s own
`thermal_step` under the heat race's `sweat_paid` patch (experiments.heat_race._apply).

WHY: the water race (40d47e4) found sweat makes W.2(c)'s roof route unreachable
(peak 37.34 C). Before declaring a "real heat hazard" arm, measure which hazard can
reach T_HOT_DEATH (40 C) at all, so the next arm is not raced blind.

Humidity model (declared, not in W0 yet): the evaporative ceiling scales with the
skin-to-air vapour-pressure gradient, f(RH, T_env) = max(0, (p_sat(35 C) -
RH * p_sat(T_env)) / p_sat(35 C)), Magnus formula; the sweat clamp E_SW_MAX and the
wet-skin channel are both multiplied by f. Biology anchor: Vecellio et al. 2022
(J Appl Physiol 132:340) put the young-adult uncompensable limit near 31 C
wet-bulb at light activity, i.e. ~36 C/80% RH or ~40 C/50%, far below Sherwood &
Huber's 35 C theoretical bound.

Usage: python -m experiments.heat_hazard_table      (seconds; prints JSON)
"""
from __future__ import annotations

import json
import math

from experiments import heat_race as H


def p_sat(t_c: float) -> float:          # kPa, Magnus
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def f_humid(rh: float, t_env: float) -> float:
    return max(0.0, (p_sat(35.0) - rh * p_sat(t_env)) / p_sat(35.0))


def time_to_death(N, M: float, occ: float, t_env: float, rh: float,
                  horizon_s: float = 3600.0, dt: float = 0.2) -> dict:
    g = H.G_SW_PER_BASAL * N.M_BASAL
    cap = H.E_SW_MAX_PER_BASAL * N.M_BASAL * f_humid(rh, t_env)
    T, t, above, t_max = 37.0, 0.0, 0.0, 37.0
    k = N.k_eff(0.0, occ)
    while t < horizon_s:
        e = min(cap, g * max(0.0, T - N.T_SETPOINT))
        T += (M - k * (T - t_env) - e) / N.C_EFF * dt
        t += dt
        t_max = max(t_max, T)
        above = above + dt if T > N.T_HOT_DEATH else 0.0
        if above >= N.DEATH_T_S:
            return {"dies": 1, "death_s": round(t, 1), "t_max_c": round(t_max, 2)}
    return {"dies": 0, "death_s": None, "t_max_c": round(t_max, 2)}


def main() -> dict:
    N, _ = H._apply("sweat_paid")
    m_full = N.M_BASAL + N.M_BASAL * 2.0      # kappa_act * p_max = 2 * M_BASAL
    out = {"M_full_power_W": m_full, "k_open": N.k_eff(0.0, 0.0),
           "k_roof": N.k_eff(0.0, 1.0), "E_SW_MAX_W": H.E_SW_MAX_PER_BASAL * N.M_BASAL,
           "T_DAY": N.T_DAY, "rows": []}
    for t_env in (30.0, 35.0, 40.0, 45.0, 50.0):
        for rh in (0.2, 0.5, 0.8, 1.0):
            for M, act in ((N.M_BASAL, "rest"), (m_full, "full")):
                for occ, where in ((0.0, "open"), (1.0, "roof")):
                    r = time_to_death(N, M, occ, t_env, rh)
                    out["rows"].append({"T_env": t_env, "RH": rh, "act": act,
                                        "where": where,
                                        "f_evap": round(f_humid(rh, t_env), 3), **r})
    # Dry-air ambient needed for death at full power in the open (no humidity).
    lo, hi = 30.0, 120.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if time_to_death(N, m_full, 0.0, mid, 0.0)["dies"] else (mid, hi)
    out["dry_open_full_power_lethal_T_env_c"] = round(hi, 2)
    H._apply("shipped")
    return out


if __name__ == "__main__":
    res = main()
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("T_env RH   act  where f_evap dies death_s t_max")
    for r in res["rows"]:
        print(f"{r['T_env']:5.0f} {r['RH']:.1f} {r['act']:>5} {r['where']:>5} {r['f_evap']:6.3f}"
              f" {r['dies']:4d} {str(r['death_s']):>7} {r['t_max_c']:6.2f}")
