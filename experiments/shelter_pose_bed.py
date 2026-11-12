"""J6: trace NE.01's chosen shelter pose through _night's own bed-down (NIGHT_SETTLE_DEC awake
decisions, then asleep). Diagnostic only, no ledger. Usage: python experiments/shelter_pose_bed.py SEED"""
import sys, numpy as np
sys.path.insert(0, "/home/opc/jackthelearner")
from experiments.tests import ne_01_nobody_survives_by_accident as NE
from experiments import needs
from playground import humanoid_index
seed = int(sys.argv[1])
pose = NE._find_shelter_pose(seed)
print("pose", pose, flush=True)
p, model, data, water, pool = NE._build(seed)
qadr = humanoid_index(model)["qposadr"]
layer = NE._layer(model, pool, seed * 733 + 23)
layer.t = needs.DAY_S; layer.state = needs.NeedState(p=0.70)
NE._place(model, data, qadr, (pose["xy"][0], pose["xy"][1], 0.25))
root0 = data.qpos[qadr:qadr+3].copy() if np.isscalar(qadr) else None
occ = []
for i in range(NE.NIGHT_SETTLE_DEC + 60):
    if i == NE.NIGHT_SETTLE_DEC: layer.set_asleep(True)
    NE._step_decision(model, data, water, layer, np.zeros(model.nu))
    occ.append(layer.last_occlusion)
    if i < 12 or i % 10 == 0:
        print(i, round(layer.last_occlusion, 3), "root", np.round(np.asarray(data.qpos)[np.atleast_1d(qadr)[0]:np.atleast_1d(qadr)[0]+3] if not isinstance(qadr, dict) else 0, 3), np.round(data.xpos[1], 3) if False else "", flush=True)
