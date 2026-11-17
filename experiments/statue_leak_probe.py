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
PilotLives cap 9,000) with native_substep=True (identical world, J6 2b,
tests/test_native_substep_env.py) to life 64; SELF-CHECK every finished
life's (decisions, floor meals) against the card's c0.json row — any
mismatch is printed and the replay is void. In life 64 log, every 10
decisions and at every meal: body root (x, y, z), each floor food's (x, y),
nearest food distance, energy.
    nice -n 19 $PY -m experiments.statue_leak_probe   (~30 CPU-min)
Output: /data/jack-logs/j11/statue_leak_601_64.json
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
OUT = Path("/data/jack-logs/j11/statue_leak_601_64.json")


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
                  move_cost_dial=False, native_substep=True)
    w = PilotLives(env, cap=LIFE_CAP)
    pol = DoNothing(SEED)
    o, info = w.reset(seed=SEED)
    pol.new_life()
    mism, trace, meals_prev = [], [], 0

    def snap(k, tag):
        e = w.env.w
        root = [float(x) for x in e.data.qpos[:3]]
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


if __name__ == "__main__":
    main()
