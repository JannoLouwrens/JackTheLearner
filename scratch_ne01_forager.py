"""Pilot 4: drink teleport + mouth-gated food serve mechanics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
from playground import make_playground, PlaygroundParams, humanoid_index
from experiments.needs import NeedLayer

p = PlaygroundParams(seed=0)
model, data, water = make_playground(p, with_water=True, with_humanoid=True)
ix = humanoid_index(model)
qadr = ix["qposadr"]
layer = NeedLayer(model, j0=2.24, alpha=0.027, p_max=1408.0, pool=(2.6,-2.4,p.pool_size,0.0), seed=0)
head_id = int(model.geom("head").id)
dt = float(model.opt.timestep); fs = int(round(0.2/dt))

def decide_n(n, ctrl_fn=None):
    for _ in range(n):
        layer.begin_decision()
        ctrl = ctrl_fn() if ctrl_fn else np.zeros(model.nu)
        for _ in range(fs):
            data.ctrl[:] = ctrl
            water.apply(model, data)
            mujoco.mj_step(model, data)
            layer.substep(model, data, dt)
        layer.decide()

# 1. drink: teleport lying onto the pool
data.qpos[qadr:qadr+3] = (2.6, -2.4, 0.30)
data.qpos[qadr+3:qadr+7] = (0.7071, 0, 0.7071, 0)
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
decide_n(15)
print(f"drink: drank_total={layer.drank_total} wet={layer.state.skin_wetness:.3f} "
      f"head_z={data.geom_xpos[head_id][2]:.2f} dead={layer.dead} w={layer.state.w:.2f}")

# 2. teleport back to home ground, serve obj0 above the head
data.qpos[qadr:qadr+3] = (0.0, -1.6, 0.30)
data.qpos[qadr+3:qadr+7] = (0.7071, 0, 0.7071, 0)
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)
decide_n(5)
hp = np.array(data.geom_xpos[head_id])
for name in ("obj0", "apple"):
    bid = int(model.body(name).id)
    jadr = int(model.body_jntadr[bid])
    qa = int(model.jnt_qposadr[jadr])
    data.qpos[qa:qa+3] = (hp[0], hp[1], hp[2] + 0.25)
    data.qpos[qa+3:qa+7] = (1,0,0,0)
    dof = int(model.jnt_dofadr[jadr]); data.qvel[dof:dof+6] = 0.0
mujoco.mj_forward(model, data)
e0 = layer.state.e
layer.state = layer.state.__class__(**{**vars(layer.state), "e": 0.4})
decide_n(20)
print(f"eat: ate={dict(layer.ate_total)} e 0.40 -> {layer.state.e:.3f} wet={layer.state.skin_wetness:.3f}")
