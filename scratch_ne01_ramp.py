"""Pilot 3: grid-search a lying pose under the ramp for occ >= 0.4 after settling."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
from playground import make_playground, PlaygroundParams, humanoid_index
from experiments.needs import NeedLayer

for seed in (0, 1, 2):
    p = PlaygroundParams(seed=seed)
    if seed > 0:
        p = p.mutate(np.random.RandomState(seed))
    model, data, water = make_playground(p, with_water=True, with_humanoid=True)
    ix = humanoid_index(model)
    qadr = ix["qposadr"]
    layer = NeedLayer(model, j0=2.24, alpha=0.027, p_max=1408.0, pool=(2.6,-2.4,p.pool_size,0.0), seed=0)
    head_id = int(model.geom("head").id)
    th = math.radians(p.ramp_angle_deg)
    best = (0.0, None)
    q0 = data.qpos.copy(); v0 = data.qvel.copy()
    for dx in np.arange(-1.1, 1.11, 0.15):
        for dy in (1.4, 1.7, 2.0, 2.3, 2.6):
            for quat in ((0.7071,0,0.7071,0), (0.7071,0,-0.7071,0), (0.5,0.5,0.5,0.5)):
                data.qpos[:] = q0; data.qvel[:] = 0.0
                data.qpos[qadr:qadr+3] = (-2.5+dx, dy, 0.25)
                data.qpos[qadr+3:qadr+7] = quat
                mujoco.mj_forward(model, data)
                for _ in range(300):     # settle 1.5 s
                    data.ctrl[:] = 0.0
                    mujoco.mj_step(model, data)
                occ = layer._sky_occlusion(model, data)
                if occ > best[0]:
                    best = (occ, (-2.5+dx, dy, quat, float(data.geom_xpos[head_id][2])))
    print(f"seed {seed} (ramp {p.ramp_angle_deg:.1f} deg): best occ={best[0]:.3f} at {best[1]}")
