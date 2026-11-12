"""J6: NE.01's own sheltered-night fixture (same pose search, same _night) under
the shipped OCC_CUT law vs the sky-temperature law priced in shelter_law_price.py.
Diagnostic only: writes no ledger. Usage: python -m experiments.shelter_law_night SEED"""
import json, sys
import numpy as np
from experiments import needs as N
from experiments.tests import ne_01_nobody_survives_by_accident as NE

ORIG = N.thermal_step
FR = N.H_R / (N.H_R + N.H_C_STILL)
TSKY = lambda Ta: 0.0552 * (Ta + 273.15) ** 1.5 - 273.15     # Swinbank 1963
TA_NIGHT = 27.79   # night air that keeps the OPEN night at T_env_eff 20 C (price.py)
LOG = []

def cut_step(T, p, wet, occ, T_env, dt, kappa, wind_ms=0.0, shiver_on=True):
    LOG.append(occ)
    return ORIG(T, p, wet, occ, T_env, dt, kappa, wind_ms, shiver_on)

def sky_step(T, p, wet, occ, T_env, dt, kappa, wind_ms=0.0, shiver_on=True):
    LOG.append(occ)
    if T_env < N.T_DAY - 1e-9:   # night: the roof hides the cold sky
        T_env = TA_NIGHT - (1.0 - occ) * FR * (TA_NIGHT - TSKY(TA_NIGHT))
    return ORIG(T, p, wet, 0.0, T_env, dt, kappa, wind_ms, shiver_on)

def main(seed):
    pose = NE._find_shelter_pose(seed)
    out = {"seed": seed, "pose": [float(v) for v in pose["xy"]], "occ_probe": pose["occ_probe"]}
    for name, fn in (("cut", cut_step), ("sky", sky_step)):
        LOG.clear(); N.thermal_step = fn
        r = NE._night(seed, pose["xy"], 23, coarse=False)
        o = np.array(LOG)
        out[name] = dict(T_dawn=round(r["T_dawn"], 3), cost=round(r["cost"], 4), alive=r["alive"],
                         occ_mean=round(r["occ_mean"], 3), occ_q=[round(float(q), 3) for q in np.quantile(o, [.1, .5, .9])],
                         ok=int(r["alive"] == 1 and r["occ_mean"] >= NE.SHELTER_OCC_MIN and r["cost"] <= NE.SHELTER_COST_MAX))
    N.thermal_step = ORIG
    print(json.dumps(out), flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]))
