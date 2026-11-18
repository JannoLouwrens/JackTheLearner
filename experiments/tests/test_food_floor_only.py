"""test_food_floor_only.py — proof of jack_env `food_floor_only` (J11).

    nice -n 19 $PY -m experiments.tests.test_food_floor_only

(1) IDENTITY: food_floor_only=False == the option absent, byte for byte,
    300 random-action decisions incl. deaths, the J11 pilot world, 2 seeds.
(2) PLANTED: on, 60 layouts x 2 worlds (W0's own respawn): no floor food's
    xy inside a non-floor ground footprint, and no food moves >= 0.5 m in the
    first 10 decisions of a do-nothing body when born >= 4 m from it.
(3) CONTROL (must fail the planted clause): off, same layouts — at least one
    food laid inside a footprint (the ramp/stairs fault the option removes).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R  # noqa: E402
from experiments.tests.test_jack_env_cures import stream  # noqa: E402

WORLD = dict(difficulty=0.75, reward="dr100-eat", random_needs=NEEDS_S_TRP,
             random_layout=True, lazy_contact_audio=True, spawn_clear=SPAWN_CLEAR_R)


def layouts(seed, n, on):
    env = JackEnv(seed=seed, food_floor_only=on, **WORLD)
    env.reset(seed=seed)
    w = env.w
    m, d = w.model, w.data
    names = [x for x in env.floor_foods if x in w.drives._food]
    inside, far_moves, far_n = 0, 0, 0
    for p in range(n):
        if p:
            w.respawn()
        xy = np.array([d.xpos[m.body(x).id][:2] for x in names])
        mask = env._flat_floor_mask(xy)
        inside += int((~mask).sum())
        root = d.xpos[w.rover_bid][:2].copy()
        for _ in range(10):
            w.decide(np.zeros(8))
            if w.died_this_decision:
                break
        xy1 = np.array([d.xpos[m.body(x).id][:2] for x in names])
        far = np.linalg.norm(xy - root, axis=1) >= 4.0
        far_n += int(far.sum())
        far_moves += int((np.linalg.norm(xy1 - xy, axis=1)[far] >= 0.5).sum())
    return inside, far_moves, far_n


def main():
    ok = True
    for s in (402, 403):
        same = stream(s, 300, **WORLD) == stream(s, 300, food_floor_only=False, **WORLD)
        print(f"(1) seed {s} identity {same}", flush=True)
        ok &= same
    tot = {True: [0, 0, 0], False: [0, 0, 0]}
    for on in (True, False):
        for s in (601, 602):
            r = layouts(s, 60, on)
            tot[on] = [a + b for a, b in zip(tot[on], r)]
            print(f"   on={on} seed {s} inside={r[0]} far movers={r[1]}/{r[2]}", flush=True)
    planted = tot[True][0] == 0 and tot[True][1] == 0 and tot[True][2] >= 50
    control = tot[False][0] >= 1
    print(f"(2) planted inside={tot[True][0]} far movers={tot[True][1]}/{tot[True][2]} -> {planted}")
    print(f"(3) control inside={tot[False][0]} far movers={tot[False][1]}/{tot[False][2]} -> {control}")
    ok &= planted and control
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
