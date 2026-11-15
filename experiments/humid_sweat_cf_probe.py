"""J6 order step (d) probe: W.2(a) on hum34 read against the SWEATING closed form. No ledger, no verdict.

WHY: hum34 keeps the hot route (W.2(c)) but loses W.2(a) (a_err_w .158 > INTEGRATOR_TOL .01): the open dry
statue no longer sits at 37 C when T_DAY = 34, so it sweats and thirst drains faster than the plain
w = 1 - B_W t. Re-deriving K_DRY repairs (a) but kills the hot route (humid_kdry_probe, 3de7a5d). This probe
measures the other route BEFORE anything is re-registered: keep hum34's world, change (a)'s reference.

The sweating closed form (bar INTEGRATOR_TOL unmoved, same window, same deprived life):
  above 37 C, wet = 0, occ = 0, p_mech = 0, T_env = T_DAY, the arm's thermal step is LINEAR in T while
  g*(T-37) < cap:   dT/dt = A - Bk (T - 37)   ->   T(t) = T* - (T* - 37) exp(-Bk t),  T* = 37 + A/Bk.
  A and Bk are read off the arm's OWN N.thermal_step at two temperatures (linearity asserted at a third).
  w(t) = 1 - B_W [ t + C_SW ( (T*-37) t - (T*-37)(1 - exp(-Bk t)) / Bk ) ]
  e(t) = 1 - B_E t   (no shiver above 37, M = M_BASAL)      p(t) = plain form (no thermal term)
Also reported: "trace" = the same integral on the statue's OWN measured T(t) (circular; diagnostic only).
Control that must fail: the plain form on hum34 (it is what W.2 measures today: .158).
Reference that must be byte-same: on shipped, T* = 37 and the sweating form collapses to the plain one.

Usage: python -m experiments.humid_sweat_cf_probe ARM SEED [SEED ...]     ARM in {hum34, shipped}
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from experiments import heat_race as H
from experiments import humid_race as HR

OUT = Path("/data/jack-logs/humid_sweat_cf_probe")


def _apply(arm: str):
    if arm == "hum34":
        return HR._apply("hum34")
    if arm == "shipped":
        return H._apply("shipped")
    raise KeyError(arm)


def _lin(N):
    """dT/dt = A - Bk (T-37) for the dry open statue by day, from the arm's own step."""
    def rate(T):
        return N.thermal_step(T, 0.0, 0.0, 0.0, N.T_DAY, 1.0, 1.0) - T
    r37, r38, r39 = rate(37.0), rate(38.0), rate(39.0)
    A, Bk = r37, r37 - r38
    nonlin = abs((r38 - r39) - Bk)
    return A, Bk, nonlin


def cell(arm: str, seed: int) -> dict:
    N, mods = _apply(arm)
    W = mods["W.2"]
    A, Bk, nonlin = _lin(N)
    Tst = 37.0 + (A / Bk if Bk > 0 else 0.0)
    rig = W.Rig(seed)
    lay = rig.layer(seed)
    p0 = lay.state.p
    t, win, dt = 0.0, W.CLOSED_FORM_WINDOW_S, W.DT_S
    plain_w = sweat_w = trace_w = sweat_e = sweat_p = 0.0
    trace_int = 0.0                       # ∫ max(0, T-37) on the statue's own T
    t_hi = lay.state.T
    while not lay.dead and t < win - 1e-9:
        T_prev = lay.state.T
        rig.place("land")
        lay.set_asleep(False)
        lay.begin_decision()
        lay.substep(rig.model, rig.data, dt)
        lay.decide()
        t += dt
        s = lay.state
        t_hi = max(t_hi, s.T)
        trace_int += max(0.0, T_prev - N.T_SETPOINT) * dt
        dT0 = Tst - 37.0
        cf_int = dT0 * t - (dT0 * (1.0 - math.exp(-Bk * t)) / Bk if Bk > 0 else 0.0)
        plain_w = max(plain_w, abs(s.w - (1.0 - N.B_W * t)))
        sweat_w = max(sweat_w, abs(s.w - (1.0 - N.B_W * (t + N.C_SW * cf_int))))
        trace_w = max(trace_w, abs(s.w - (1.0 - N.B_W * (t + N.C_SW * trace_int))))
        sweat_e = max(sweat_e, abs(s.e - (1.0 - N.B_E * t)))
        sweat_p = max(sweat_p, abs(s.p - (1.0 - (1.0 - p0) * math.exp(-t / N.TAU_WAKE))))
    r = {"arm": arm, "seed": seed, "window_s": t, "dead_in_window": float(lay.dead),
         "A": A, "Bk": Bk, "nonlin": nonlin, "T_star": Tst, "T_max_meas": t_hi,
         "C_SW": N.C_SW, "plain_err_w": plain_w, "sweat_err_w": sweat_w,
         "trace_err_w": trace_w, "err_e": sweat_e, "err_p": sweat_p,
         "a_worst_sweat": max(sweat_w, sweat_e, sweat_p),
         "a_pass_sweat": float(max(sweat_w, sweat_e, sweat_p) <= W.INTEGRATOR_TOL),
         "a_pass_plain": float(max(plain_w, sweat_e, sweat_p) <= W.INTEGRATOR_TOL)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{arm}_s{seed}.json").write_text(json.dumps(r, default=H._jsonable))
    return r


if __name__ == "__main__":
    arm = sys.argv[1]
    for sd in sys.argv[2:]:
        print(json.dumps(cell(arm, int(sd))), flush=True)
