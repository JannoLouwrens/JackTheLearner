"""J6(c) humidity arm — pre-registration arithmetic (route table), no cells, no ledger.

WHY (docs/research/NEEDS_CLOCK_RACE.md, 11-14 03:55): sweat made W0's only heat
death unreachable (W.2(c) `c_hot_reachable` 1 -> 0 on 10/10 sweat seeds), and
`heat_hazard_table` showed the only physical route back to heat death for a
sweating body is HUMIDITY capping evaporation. Before a race is declared, this
module builds the candidate humidity worlds on the water race's `swp_k72` body
and measures, through the arm's OWN patched `needs.thermal_step` (sweat clamp
AND wet-skin channel both scaled by f), every thermal route W.2 and W.1 probe:
which routes kill, which refuge survives, and what each candidate costs.

Humidity model (heat_hazard_table's, declared there):
  f(RH, T_env) = max(0, (p_sat(35) - RH * p_sat(T_env)) / p_sat(35)), Magnus.
  Local RH rises toward saturation under a seal (the occupant's own vapour has
  nowhere to go): RH_local = RH_OPEN + (RH_SEALED - RH_OPEN) * sky_occlusion.
  `needs.thermal_step` keeps its exact signature (W.1 inspects it).

Candidate arms (both on swp_k72 = sweat_paid + k72 clock):
  hum30   T_DAY 30 (shipped), RH_OPEN 0.5, RH_SEALED 1.0.  No other constant moves.
  hum34   T_DAY 34, DELTA_T_NIGHT 14 (night stays 20 C), RH_OPEN 0.5, RH_SEALED 1.0.
          Costs: W.2's docstring fixed point K_DRY = M_BASAL/(T_SETPOINT - T_DAY)
          no longer holds (a dry statue in the open settles above 37 C), and
          NE.01's DELTA_T_NIGHT calibration (12 -> 10) is re-opened.

Routes (fixed ambient, 3000 s, death = DEATH_T_S continuous dwell past a threshold):
  rest_open_day / rest_roof_day / rest_soak_day   statue, wet 0/0/1, occ 0/1/0
  full_open_day / full_roof_day                   M = 3 x basal (kappa_act*p_max = 2 M_BASAL)
  rest_soak_night (W.2's cold route) / rest_roof_night / rest_open_night

Usage: python -m experiments.humid_race check      (seconds; prints JSON)
"""
from __future__ import annotations

import json
import sys

from experiments import heat_race as H
from experiments import water_clock_race as WR
from experiments.heat_hazard_table import f_humid

HUMID = {"hum30": {"T_DAY": 30.0, "DELTA_T_NIGHT": 10.0, "RH_OPEN": 0.5, "RH_SEALED": 1.0},
         "hum34": {"T_DAY": 34.0, "DELTA_T_NIGHT": 14.0, "RH_OPEN": 0.5, "RH_SEALED": 1.0}}
ARMS = ("swp_k72",) + tuple(HUMID)


def _apply(arm: str):
    """swp_k72's body + clocks, then (humid arms) the humidity field on the
    shipped thermal_step signature. Horizons of W.2 come from WR._apply."""
    N, mods = WR._apply("swp_k72")
    if arm in HUMID:
        h = HUMID[arm]
        N.T_DAY, N.DELTA_T_NIGHT = h["T_DAY"], h["DELTA_T_NIGHT"]
        dry_e_evap = N.e_evap               # heat_race's sweat_paid e_evap

        def thermal_step(T, p_mech, skin_wetness, sky_occlusion, T_env, dt,
                         kappa_act, wind_ms=0.0, shiver_on=True):
            rh = h["RH_OPEN"] + (h["RH_SEALED"] - h["RH_OPEN"]) * sky_occlusion
            m = N.metabolic_rate(p_mech, T, kappa_act, shiver_on)
            dT = (m - N.k_eff(skin_wetness, sky_occlusion, wind_ms) * (T - T_env)
                  - f_humid(rh, T_env) * dry_e_evap(skin_wetness, T)) / N.C_EFF
            return T + dT * dt
        N.thermal_step = thermal_step
    return N, mods


def _route(N, M: float, wet: float, occ: float, T_env: float,
           horizon_s: float = 3000.0, dt: float = 0.2) -> dict:
    kappa = 1.0                              # p_mech carries the extra watts
    T, t, hot, cold, lo, hi = 37.0, 0.0, 0.0, 0.0, 37.0, 37.0
    while t < horizon_s:
        T = N.thermal_step(T, M - N.M_BASAL, wet, occ, T_env, dt, kappa)
        t += dt
        lo, hi = min(lo, T), max(hi, T)
        hot = hot + dt if T >= N.T_HOT_DEATH else 0.0
        cold = cold + dt if T <= N.T_COLD_DEATH else 0.0
        if hot >= N.DEATH_T_S or cold >= N.DEATH_T_S:
            return {"dies": "hyperthermia" if hot else "hypothermia",
                    "t_s": round(t, 1), "t_min": round(lo, 2), "t_max": round(hi, 2)}
    return {"dies": "", "t_s": None, "t_min": round(lo, 2), "t_max": round(hi, 2)}


def check() -> dict:
    out = {}
    for arm in ARMS:
        N, _ = _apply(arm)
        day, night = N.T_DAY, N.T_DAY - N.DELTA_T_NIGHT
        rest, full = N.M_BASAL, 3.0 * N.M_BASAL
        out[arm] = {
            "T_DAY": day, "T_NIGHT": night,
            "rest_open_day": _route(N, rest, 0.0, 0.0, day),
            "rest_roof_day": _route(N, rest, 0.0, 1.0, day),
            "rest_soak_day": _route(N, rest, 1.0, 0.0, day),
            "full_open_day": _route(N, full, 0.0, 0.0, day),
            "full_roof_day": _route(N, full, 0.0, 1.0, day),
            "rest_soak_night": _route(N, rest, 1.0, 0.0, night),
            "rest_roof_night": _route(N, rest, 0.0, 1.0, night),
            "rest_open_night": _route(N, rest, 0.0, 0.0, night),
        }
    H._apply("shipped")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        print(json.dumps(check(), indent=1))
    else:
        sys.exit(f"unknown command {cmd!r}")
