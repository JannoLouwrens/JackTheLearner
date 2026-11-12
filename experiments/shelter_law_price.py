"""J6 pricing: current OCC_CUT law vs sky-temperature (Swinbank) shelter law. Pure ODE, no world."""
import os, sys
os.chdir('/home/opc/jackthelearner'); sys.path.insert(0, '/home/opc/jackthelearner')
from experiments import needs as N
dt = 0.2
def night(occ, law, T0=37.0, Ta=None):
    T = T0
    for _ in range(int(N.NIGHT_S / dt)):
        if law == 'cut':
            T = N.thermal_step(T, 0.0, 0.0, occ, N.T_DAY - N.DELTA_T_NIGHT, dt, 0.0)
        else:
            Tenv = Ta - (1 - occ) * FR * (Ta - tsky(Ta))
            T = N.thermal_step(T, 0.0, 0.0, 0.0, Tenv, dt, 0.0)
    return T
def day_eq(occ):  # current law, resting, by day: equilibrium (no shiver above 37)
    return N.T_DAY + N.M_BASAL / N.k_eff(0, occ)
FR = N.H_R / (N.H_R + N.H_C_STILL)
tsky = lambda Ta: 0.0552 * (Ta + 273.15) ** 1.5 - 273.15   # Swinbank 1963, clear sky
# night air chosen so the OPEN night is unchanged (T_env_eff = 20 C, cost 0.498 calibration kept)
lo, hi = 10.0, 40.0
for _ in range(60):
    m = (lo + hi) / 2
    (lo, hi) = (m, hi) if m - FR * (m - tsky(m)) < N.T_DAY - N.DELTA_T_NIGHT else (lo, m)
Ta = (lo + hi) / 2
print(f"NIGHT_S={N.NIGHT_S} f_rad={FR:.3f}  night air for unchanged open night: {Ta:.2f} C, sky {tsky(Ta):.2f} C")
print("occ   cut:T_dawn cost   sky:T_dawn cost   cut:day_eq_rest")
for occ in (0, .2, .4, .5, .6, .7, .741, .8, .9, 1.0):
    a = night(occ, 'cut'); b = night(occ, 'sky', Ta=Ta)
    print(f"{occ:4.2f}  {a:6.2f} {N.delta_T(a):.3f}      {b:6.2f} {N.delta_T(b):.3f}      {day_eq(occ):6.2f}")
