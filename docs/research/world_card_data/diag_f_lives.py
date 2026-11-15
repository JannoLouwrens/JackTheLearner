"""Diagnostic (smoke seed, NOT evidence): why does row f eat 12 meals in life 0 and ~2.5 after?
mode persist = registered f (memory carried across lives); mode fresh = a new HandForager each life."""
import sys, json, os, numpy as np
sys.path.insert(0, "/home/opc/jackthelearner"); os.chdir("/home/opc/jackthelearner")
from experiments import world_card as wc
from experiments.jack_env import JackEnv
mode, seed, lives, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
env = JackEnv(seed=seed); o, info = env.reset(seed=seed)
w = env.w; m = w.model
pol = wc.HandForager(seed)
def foodstate():
    r = {}
    for nme in ("obj0", "obj1"):
        try: g = int(m.geom(nme).id)
        except Exception: continue
        p = w.data.geom_xpos[g]
        r[nme] = {"tex": round(float(w._tex[g]), 4), "xyz": [round(float(x), 2) for x in p]}
    return r
for life in range(lives):
    if mode == "fresh" and life > 0: pol = wc.HandForager(seed * 1000 + life)
    pol.new_life()
    ate0 = dict(info["ate_total"]); n = 0; t0 = w.sim_seconds; start = foodstate()
    chase = 0; seen_food = 0; ftex = {v["tex"] for v in start.values()}
    while n < wc.LIFE_CAP_DECISIONS:
        a = pol.act(o)
        if pol.target is not None: chase += 1
        vis = np.asarray(o[:32]).reshape(16, 2)
        if any(round(float(t), 4) in ftex for t in vis[:, 1]): seen_food += 1
        o, r, term, trunc, info = env.step(a); n += 1
        if term: break
    ate = {k: info["ate_total"][k] - ate0.get(k, 0) for k in info["ate_total"]}
    rec = {"mode": mode, "seed": seed, "life": life, "life_s": round(w.sim_seconds - t0, 1), "ate": ate,
           "cause": info.get("death_cause") if term else "censored",
           "frac_chasing": round(chase / n, 3), "frac_food_tex_on_a_ray": round(seen_food / n, 3),
           "food_start": start, "food_end": foodstate(),
           "pol_food": sorted(pol.food), "pol_banned": sorted(pol.banned), "pol_miss": pol.miss, "pol_food_miss": pol.food_miss}
    open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
    if term: o, info = env.reset()
    else:
        w.respawn(); env._pending_obs = None; o, info = env.reset()
