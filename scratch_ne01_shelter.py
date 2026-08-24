"""Pilot 2: sheltered sleeping pose = lying under platform with objects flanked."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mujoco
from playground import make_playground, PlaygroundParams, LADDER_X, LADDER_Y, humanoid_index
from experiments.needs import NeedLayer

for seed in (0, 1, 2):
    p = PlaygroundParams(seed=seed)
    if seed > 0:
        p = p.mutate(np.random.RandomState(seed))
    model, data, water = make_playground(p, with_water=True, with_humanoid=True)
    ix = humanoid_index(model)
    qadr, dadr = ix["qposadr"], ix["dofadr"]
    layer = NeedLayer(model, j0=2.24, alpha=0.027, p_max=1408.0, pool=(2.6,-2.4,p.pool_size,0.0), seed=0)
    hx, hy = LADDER_X, LADDER_Y + 0.45          # platform center
    data.qpos[qadr:qadr+3] = (hx, hy, 0.3)
    data.qpos[qadr+3:qadr+7] = (0.7071, 0.0, 0.7071, 0.0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    head_id = int(model.geom("head").id)
    hp = np.array(data.geom_xpos[head_id])
    print(f"seed {seed}: head at {hp.round(2)}, plat_h={p.ladder_height:.2f}")
    occ0 = layer._sky_occlusion(model, data)
    # flank the head with all objects, ring at 0.45 m, resting on floor
    n = 0
    for i in range(p.n_objects):
        try:
            bid = int(model.body(f"obj{i}").id)
        except Exception:
            continue
        jadr = int(model.body_jntadr[bid])
        qa = int(model.jnt_qposadr[jadr])
        ang = 2*np.pi*n/max(1,p.n_objects)
        # objects placed beside the head, snug ring
        data.qpos[qa:qa+3] = (hp[0]+0.40*np.cos(ang), hp[1]+0.40*np.sin(ang), 0.35)
        data.qpos[qa+3:qa+7] = (1,0,0,0)
        n += 1
    mujoco.mj_forward(model, data)
    occ1 = layer._sky_occlusion(model, data)
    # let it settle 2 s and re-measure (objects fall to rest)
    for _ in range(400):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    occ2 = layer._sky_occlusion(model, data)
    hp2 = np.array(data.geom_xpos[head_id])
    print(f"  occ platform-only={occ0:.3f}  +objects placed={occ1:.3f}  settled 2s={occ2:.3f} head_z={hp2[2]:.2f}")
