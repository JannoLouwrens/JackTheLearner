"""statue_leak_probe.py — why a do-nothing body born 1.577 m from food lived
fed (J11, floor card: c0 noop-k1 AND noop-k0, seed 601, life 64, 36 floor
meals, spawn 1.577 m > SPAWN_CLEAR_R 1.55 m). Learner-free; no ledger write.

Question: is the R = 1.55 m reach (measured over 250-decision lives,
/data/jack-logs/j11/reach_probe.py) too short because (a) the body keeps
sliding over a 9,000-decision life, or (b) a food moves to him (foods are
free bodies: "where eating shoves them", jack_env OPTIONS random_layout), or
(c) something else (a food re-placed by random_layout at a rebirth? — no:
the statue never dies inside the life, so no re-placement happens in it).

Method: replay the card cell exactly (pilot_b2.make_world(601), DoNothing,
PilotLives cap 9,000) (run 2: WITHOUT native_substep — it drifted 1-2 decisions in run 1) to life 64; SELF-CHECK every finished
life's (decisions, floor meals) against the card's c0.json row — any
mismatch is printed and the replay is void. In life 64 log, every 10
decisions and at every meal: body root (x, y, z), each floor food's (x, y),
nearest food distance, energy.
    nice -n 19 $PY -m experiments.statue_leak_probe   (~30 CPU-min)
Output: /data/jack-logs/j11/statue_leak_601_64.json

READING RULE (written 11-17 22:45, BEFORE the trace exists; `read`):
    $PY -m experiments.statue_leak_probe read [path]
VOID if any life mismatched the card or the life has no floor meal. Else,
at the FIRST meal, for the food eaten (the one whose ate count rose):
gap0 = its spawn-time distance, root_mv = body xy travel since spawn,
food_mv = that food's xy travel since spawn. Verdict:
  "slide" (cause a: R too short)  if root_mv >= 2 * food_mv
  "shove" (cause b: a food came)  if food_mv >= 2 * root_mv
  "both"                          otherwise.
Also printed: max root travel over the whole life (what a re-measured R
must exceed if "slide"), and the decision of the first meal.
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

SEED, LIFE, ARM = 601, 64, "noop-k1"
CARD = Path("/data/jack-logs/pilot_b2/c0.json")
OUT = Path("/data/jack-logs/j11/statue_leak_601_64_r2.json")
# run 1 (/data/jack-logs/j11/statue_leak_601_64.json): VOID by the rule (lives 51/54 off by 1-2
# decisions under native_substep) AND its "root" was the apple (qpos[:3]); life 64 itself matched
# (9000 dec, 36 meals); obj1 eaten every ~250 decisions (k=49,300,551,801) = the respawn period.


def main():
    from experiments.jack_env import (JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R,
                                      FLOOR_FOODS)
    from experiments.pilot_b import DIFFICULTY, REWARD, LIFE_CAP, PilotLives
    from experiments.world_card import DoNothing
    card = next(r for r in json.loads(CARD.read_text())["cells"].values()
                if r["arm"] == ARM and r["seed"] == SEED)["lives"]
    env = JackEnv(seed=SEED, difficulty=DIFFICULTY, reward=REWARD,
                  random_needs=NEEDS_S_TRP, random_layout=True,
                  lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R,
                  move_cost_dial=False)   # as the card ran it: native_substep drifted 1-2 decisions in lives 51/54 (run 1 VOID)
    w = PilotLives(env, cap=LIFE_CAP)
    pol = DoNothing(SEED)
    o, info = w.reset(seed=SEED)
    pol.new_life()
    mism, trace, meals_prev = [], [], 0

    def snap(k, tag):
        e = w.env.w
        root = [float(x) for x in e.data.xpos[e.rover_bid][:3]]   # his body (qpos[:3] is the APPLE's freejoint — run 1 read that, 11-17 23:50)
        foods = {n: [float(x) for x in e.data.xpos[e.model.body(n).id][:2]]
                 for n in getattr(w.env, "floor_foods", FLOOR_FOODS)
                 if n in e.drives._food}
        d = {n: float(np.hypot(root[0] - xy[0], root[1] - xy[1]))
             for n, xy in foods.items()}
        ate = dict(e.drives.ate_total)
        trace.append({"k": k, "tag": tag, "root": root, "foods": foods,
                      "dist": d, "ate": ate, "e": float(e.drives.state.e)})
        return sum(ate.get(n, 0) for n in foods)

    k = 0
    while len(w.lives) <= LIFE:
        in_life = len(w.lives) == LIFE
        if in_life and k == 0:
            meals_prev = snap(0, "spawn")
            spawn_d = float(getattr(w.env, "last_spawn_food_dist", float("nan")))
        o, r, term, trunc, info = w.step(pol.act(o))
        if in_life:
            k += 1
            m = sum(w.env.w.drives.ate_total.get(n, 0)
                    for n in getattr(w.env, "floor_foods", FLOOR_FOODS))
            if m != meals_prev or k % 10 == 0:
                meals_prev = snap(k, "meal" if m != meals_prev else "tick")
        if term or trunc:
            i = len(w.lives) - 1
            got = (w.lives[i]["decisions"], w.lives[i]["floor_meals"])
            want = (card[i]["decisions"], card[i]["floor_meals"])
            if got != want:
                mism.append({"life": i, "got": got, "card": want})
                print("MISMATCH", i, got, want, flush=True)
            if i % 8 == 0:
                print(f"life {i} ok={got == want} {got}", flush=True)
            o, info = w.reset()
            pol.new_life()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = {"seed": SEED, "life": LIFE, "arm": ARM,
           "spawn_dist_replay": spawn_d, "spawn_dist_card": card[LIFE]["spawn_dist"],
           "life_row": w.lives[LIFE], "mismatches": mism, "trace": trace}
    OUT.write_text(json.dumps(res) + "\n")
    first = next((t for t in trace if t["tag"] == "meal"), None)
    print("DONE mismatches", len(mism), "spawn", spawn_d,
          "first meal", None if first is None else
          {"k": first["k"], "root": first["root"], "dist": first["dist"]},
          "spawn root", trace[0]["root"], "spawn dist", trace[0]["dist"], flush=True)


def _xy(a):
    return np.asarray(a[:2], dtype=float)


def verdict(res):
    """Apply the READING RULE (docstring) to a probe result dict."""
    if res.get("mismatches"):
        return {"verdict": "VOID", "why": "replay mismatched the card"}
    tr = res["trace"]
    t0 = tr[0]
    first = next((t for t in tr if t["tag"] == "meal"), None)
    if first is None:
        return {"verdict": "VOID", "why": "no floor meal in the life"}
    prev = max((t for t in tr if t["k"] < first["k"]), key=lambda t: t["k"])
    eaten = [n for n in first["ate"]
             if first["ate"].get(n, 0) > prev["ate"].get(n, 0)
             and n in t0["foods"]]
    if not eaten:
        return {"verdict": "VOID", "why": "meal food not identified"}
    n = eaten[0]
    root_mv = float(np.linalg.norm(_xy(first["root"]) - _xy(t0["root"])))
    food_mv = float(np.linalg.norm(_xy(first["foods"][n]) - _xy(t0["foods"][n])))
    max_root = max(float(np.linalg.norm(_xy(t["root"]) - _xy(t0["root"])))
                   for t in tr)
    if root_mv >= 2 * food_mv:
        v = "slide"
    elif food_mv >= 2 * root_mv:
        v = "shove"
    else:
        v = "both"
    return {"verdict": v, "food": n, "first_meal_k": first["k"],
            "gap0": t0["dist"][n], "root_mv": root_mv, "food_mv": food_mv,
            "max_root_travel": max_root}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "read":
        path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
        print(json.dumps(verdict(json.loads(path.read_text())), indent=1))
    else:
        main()
