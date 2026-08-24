"""NE.01 dev pilot — NOT the registered run. Measures: throughput, random-agent
death time+cause, occlusion lying under the platform, night-cost sweep."""
import sys, time, math
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import numpy as np
import mujoco
from playground import make_playground, PlaygroundParams, LADDER_X, LADDER_Y, humanoid_index
from experiments import needs
from experiments.needs import (NeedLayer, NeedState, thermal_step, metabolic_rate,
                               delta_T, T_SETPOINT, T_DAY, DAY_S, NIGHT_S, K_DRY)

J0, ALPHA, PMAX = 2.23677, 0.027222, 1407.854719
POOL = (2.6, -2.4, 0.9, 0.0)

# ── 1. night-cost sweep, pure ODE (asleep, statue, dry) ─────────────────
print("== night-cost sweep (dawn delta_T after 400 s asleep, from 37 C) ==")
for dtn in (4.0, 6.0, 8.0, 10.0, 12.0):
    for occ in (0.0, 0.45, 0.55):
        T = 37.0
        for _ in range(2000):
            T = thermal_step(T, 0.0, 0.0, occ, T_DAY - dtn, 0.2, 0.0)
        print(f"  dTn={dtn:4.1f} occ={occ:.2f}  T_dawn={T:6.2f}  deltaT={delta_T(T):.3f}")

# ── 2. random agent: death time + cause, throughput ─────────────────────
print("\n== random lives ==")
p = PlaygroundParams(seed=0)
for life in range(3):
    model, data, water = make_playground(p, with_water=True, with_humanoid=True)
    layer = NeedLayer(model, j0=J0, alpha=ALPHA, p_max=PMAX, pool=POOL, seed=life)
    rng = np.random.RandomState(100 + life)
    dt = float(model.opt.timestep)
    fs = max(1, int(round(0.2 / dt)))
    t0 = time.time()
    k = 0
    while not layer.dead and k < 6000:
        layer.begin_decision()
        ctrl = rng.uniform(-0.4, 0.4, model.nu) * layer.gear_scale()
        if layer.microsleep_zeroed():
            ctrl = ctrl * 0.0
        for _ in range(fs):
            data.ctrl[:] = ctrl
            water.apply(model, data)
            mujoco.mj_step(model, data)
            layer.substep(model, data, dt)
        layer.decide()
        k += 1
        if k in (500, 1500, 3000):
            s = layer.state
            print(f"   k={k}: e={s.e:.2f} w={s.w:.2f} p={s.p:.2f} T={s.T:.1f} "
                  f"f={s.f:.2f} i={s.i:.2f} pow={layer.last_power_w:.0f}W gear={layer.gear_scale():.2f}")
    wall = time.time() - t0
    rec = layer.death_record
    print(f"  life {life}: dead={layer.dead} k={k} t={layer.t:.0f}s "
          f"cause={rec['cause'] if rec else None} wall={wall:.1f}s ({k/wall:.0f} dec/s)")

# ── 3. occlusion lying under the platform, and in the open ──────────────
print("\n== occlusion probe ==")
model, data, water = make_playground(p, with_water=True, with_humanoid=True)
ix = humanoid_index(model)
qadr, dadr, nv = ix["qposadr"], ix["dofadr"], ix.get("nv", 23)
layer = NeedLayer(model, j0=J0, alpha=ALPHA, p_max=PMAX, pool=POOL, seed=0)
for label, (x, y) in (("under-platform", (LADDER_X, LADDER_Y + 0.45)),
                      ("open", (0.0, 1.5))):
    data.qpos[qadr:qadr+3] = (x, y, 0.3)
    data.qpos[qadr+3:qadr+7] = (0.7071, 0.0, 0.7071, 0.0)   # lying on back
    data.qvel[dadr:dadr+len(data.qvel[dadr:])] = 0.0
    mujoco.mj_forward(model, data)
    occ = layer._sky_occlusion(model, data)
    print(f"  {label}: occ={occ:.3f}  head_z={data.geom_xpos[int(model.geom('head').id)][2]:.2f}")
