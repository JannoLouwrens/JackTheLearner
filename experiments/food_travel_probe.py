"""food_travel_probe.py — does a floor food move on its own after a random
layout places it, or only when his body touches it? (J11, the statue leak's
lead: in seed 601 life 64 the eaten food obj1 travelled 1.9 m in 10
decisions, to the body and past it.) Learner-free; no ledger write.

Method: the card's world (pilot_b2.make_world(seed), d 0.75, random_layout,
spawn_clear 1.55 m), worlds 601-608. For N placements per world: call W0's
own `respawn()` (exactly what every rebirth does: lay the foods out, clear,
draw the body), then step DoNothing (zero action) for 50 decisions. Record,
per floor food: spawn distance to the body root, its xy and z at k = 0, 1,
5, 10, 50, and its deepest contact with ground at k = 0 (after mj_forward).
    nice -n 19 $PY -m experiments.food_travel_probe [n_per_world]
Output: /data/jack-logs/j11/food_travel.json

READING RULE (written 11-18 01:20, BEFORE any trace exists; `read`):
travel = a food's xy displacement k 0 -> 50. FAR = born >= 4 m from the body
root (no body part reaches 4 m), NEAR = < 2.2 m.
  "self-moving" (world fault in the layout, body not needed)
      if >= 10% of FAR foods travel >= 0.5 m;
  "body-pushed" (only contact moves foods)
      if <= 1% of FAR foods travel >= 0.5 m AND >= 1 NEAR food travels >= 0.5 m;
  "still" if <= 1% FAR and no NEAR food travels >= 0.5 m;
  "mixed" otherwise.
VOID if fewer than 100 FAR foods or fewer than 30 NEAR foods were sampled.
Also printed: travel quantiles FAR/NEAR, and for movers the k at which most
travel happened and the ground penetration at placement (a food dropped
inside a bump is ejected; one on a slope rolls).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

SEEDS = tuple(range(601, 609))
KS = (0, 1, 5, 10, 50)
OUT = Path("/data/jack-logs/j11/food_travel.json")
FAR, NEAR, MOVE = 4.0, 2.2, 0.5


def probe(seed: int, n: int) -> list:
    from experiments.pilot_b2 import make_world
    from experiments.jack_env import FLOOR_FOODS
    env = make_world(seed)
    env.reset(seed=seed)
    w = env.w
    m, d = w.model, w.data
    names = [x for x in getattr(env, "floor_foods", FLOOR_FOODS) if x in w.drives._food]
    bid = {x: m.body(x).id for x in names}
    gid = {x: int(m.geom(x).id) for x in names}
    ground = set(int(g) for g in w.ground_gids)
    rows = []
    for p in range(n):
        if p > 0:
            w.respawn()
        w.mujoco.mj_forward(m, d)
        pen = {x: 0.0 for x in names}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for x in names:
                if gid[x] in (g1, g2) and ((g2 if g1 == gid[x] else g1) in ground):
                    pen[x] = min(pen[x], float(c.dist))
        root = d.xpos[w.rover_bid][:2].copy()
        tr = {x: {} for x in names}
        for k in range(KS[-1] + 1):
            if k in KS:
                for x in names:
                    tr[x][k] = [float(v) for v in d.xpos[bid[x]][:3]]
            if k < KS[-1]:
                w.decide(np.zeros(8))
                if w.died_this_decision:
                    break
        for x in names:
            if 0 not in tr[x]:
                continue
            xy0 = np.array(tr[x][0][:2])
            kmax = max(tr[x])
            rows.append({
                "seed": seed, "p": p, "food": x,
                "spawn_d": float(np.hypot(*(xy0 - root))),
                "pen0": pen[x], "kmax": kmax, "trace": tr[x],
                "travel": float(np.hypot(*(np.array(tr[x][kmax][:2]) - xy0))),
            })
    return rows


def verdict(rows: list) -> dict:
    far = [r for r in rows if r["spawn_d"] >= FAR]
    near = [r for r in rows if r["spawn_d"] < NEAR]
    if len(far) < 100 or len(near) < 30:
        return {"verdict": "VOID", "far": len(far), "near": len(near)}
    fm = sum(r["travel"] >= MOVE for r in far)
    nm = sum(r["travel"] >= MOVE for r in near)
    ff = fm / len(far)
    if ff >= 0.10:
        v = "self-moving"
    elif ff <= 0.01 and nm >= 1:
        v = "body-pushed"
    elif ff <= 0.01:
        v = "still"
    else:
        v = "mixed"
    q = lambda a: [round(float(np.quantile(a, t)), 3) for t in (0.5, 0.9, 0.99, 1.0)]
    movers = [r for r in rows if r["travel"] >= MOVE]

    def when(r):
        t = r["trace"]
        ks = sorted(int(k) for k in t)
        seg = [(ks[i], float(np.hypot(t[str(ks[i])][0] - t[str(ks[i - 1])][0],
                                      t[str(ks[i])][1] - t[str(ks[i - 1])][1])))
               for i in range(1, len(ks))]
        return max(seg, key=lambda s: s[1])[0] if seg else None
    return {"verdict": v, "far": len(far), "far_movers": fm, "far_frac": round(ff, 4),
            "near": len(near), "near_movers": nm,
            "travel_q50_90_99_max": {"far": q([r["travel"] for r in far]),
                                     "near": q([r["travel"] for r in near])},
            "movers": [{"seed": r["seed"], "p": r["p"], "food": r["food"],
                        "spawn_d": round(r["spawn_d"], 2), "travel": round(r["travel"], 2),
                        "pen0": round(r["pen0"], 4), "by_k": when(r)}
                       for r in movers][:40],
            "pen0_q": {"movers": q([r["pen0"] for r in movers]) if movers else None,
                       "all": q([r["pen0"] for r in rows])}}


def main(n: int) -> None:
    rows = []
    for s in SEEDS:
        rows += probe(s, n)
        print(f"seed {s} rows {len(rows)}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"n_per_world": n, "rows": rows}) + "\n")
    print(json.dumps(verdict(json.loads(OUT.read_text())["rows"]), indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "read":
        path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
        print(json.dumps(verdict(json.loads(path.read_text())["rows"]), indent=1))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
