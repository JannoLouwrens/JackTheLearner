"""J6(d) diagnosis: why do some NE.01 random lives outlive the gate-2 window?
Usage: python experiments/random_life_drink_probe.py ARM SEED LIFE DEC_CAP  (nice 19)
Replays ONE life of ne_01._random_lives under an arm's _apply, byte-same rng
seeds, and logs every 500 decisions: drank_total, w, root xy, Chebyshev distance past the pool edge (<0 = inside).
No ledger, no threshold."""
import os, sys, json
REPO = "/home/opc/jackthelearner"
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
from experiments import humid_race as HR
arm, seed, life, dcap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
_, mods = HR._apply(arm)
ne = mods["NE.01"]
cap = min(ne._windows()["life_cap"], dcap)
p, model, data, water, pool = ne._build(seed)
layer = ne._layer(model, pool, seed * 131 + life)
null = ne._DisabledNeed() if life == 0 else None
rng = np.random.RandomState(seed * 9973 + life * 17)
from playground import humanoid_index
qadr = humanoid_index(model)["qposadr"]
k = 0; rows = []; drink_dec = []
while not layer.dead and k < cap:
    d0 = layer.drank_total
    ctrl = rng.uniform(-ne.CTRL_SCALE, ne.CTRL_SCALE, model.nu) * layer.gear_scale()
    ne._step_decision(model, data, water, layer, ctrl, null=null)
    if layer.drank_total > d0: drink_dec.append(k)
    k += 1
    if k % 500 == 0 or layer.dead:
        x, y = data.qpos[qadr:qadr+2]
        rows.append({"k": k, "drank": int(layer.drank_total), "w": round(float(layer.state.w), 3),
                     "xy": [round(float(x), 2), round(float(y), 2)],
                     "pool_d": round(float(max(abs(x-pool[0]), abs(y-pool[1])) - pool[2]), 2)})
        print(json.dumps(rows[-1]), flush=True)
out = {"arm": arm, "seed": seed, "life": life, "k": k, "dead": int(layer.dead),
       "cause": (layer.death_record or {}).get("cause", "none"), "drank": int(layer.drank_total),
       "drink_dec": drink_dec, "pool": list(pool), "rows": rows}
open(f"/data/tmp/j6d/{arm}_s{seed}_l{life}.json", "w").write(json.dumps(out))
print("DONE", out["k"], out["dead"], out["cause"], out["drank"], drink_dec[:20], flush=True)
