"""Proof: experiments/dial_curriculum.py (the dial's success gate, Next-up 7).

    python -m experiments.tests.test_dial_curriculum [N=150]

(a) CONTROLLER == AVALON'S CODE. A verbatim transcription of
    godot_gym.py L623-633 + world_generator.py L561 (lo = 0, the only lo
    Avalon runs: L420) is driven by the same 20,000 random success flags as
    SuccessGatedDial(0, 1, 0.01): ceilings equal at every step (|diff| = 0).
    Draws lie in [lo, ceiling] and average (lo + ceiling) / 2 within 0.01
    (10,000 draws at a fixed ceiling). Uniform mode ignores success; fixed
    mode always returns lo. CONTROL that must fail: the same transcription
    with the success sign flipped differs.
(b) PLANTED, in his world (pilot world + spawn_clear + move_cost_dial on,
    seed 153, PilotLives cap 20 decisions so lives end by the cap):
    success_fn = always True, step 0.05, 9 lives -> the ceiling climbs
    0.75, 0.80 ... 1.00 and stays clipped; every life after the first is a
    NEW world (seed differs) whose difficulty equals its draw, drawn inside
    [0.75, ceiling-at-draw]; its extra floor foods = dial_extra_foods(d),
    each floor meal = dial_meal(NU_FLOORFOOD, d), move_cost_scale =
    dial_move_cost(d), read off the rebuilt world. CONTROL that must fail
    the climb: success_fn = always False -> the ceiling never leaves 0.75,
    every d = 0.75.
(c) IDENTITY OF A REBUILD: the world the wrapper rebuilds at (d, seed) steps
    exactly like a freshly constructed JackEnv(seed, difficulty=d) under the
    same N random actions (obs, reward, termination) — no state of the old
    world leaks into the new one. CONTROL that must fail: a fresh world at
    seed + 1.
Also printed: the CPU cost of one rebuild.

RESULT (2026-11-17 22:50, box, nice 19, N=3000, 313 CPU-s): PASS.
 (a) max |diff| 0 over 20,000 lives (ceiling roamed 0.00-1.00); sign-flipped
     control differs by 1. Draws at ceiling .9: min .7500 max .9000 mean
     .8253. Uniform ceiling stays .75 after 500 updates; fixed draws {.75}.
 (b) always-success: ceiling at each draw .75 .80 .85 .90 .95 1.0 1.0 1.0 1.0;
     d .750 .753 .803 .889 .812 .774 .811 .831 .986; world checks 9/9;
     9/9 distinct world seeds. Control always-failure: ceilings {.75}, d {.75},
     world checks 6/6. Rebuild CPU median 44 ms (~2-3 decisions' worth).
 (c) d .83 seed 9001: identical, 3,002 rows, 1 death; d 1.0 seed 9002:
     identical, 3,003 rows, 2 deaths; seed+1 controls differ on both.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from experiments.dial_curriculum import DialCurriculum, SuccessGatedDial


def avalon_ceilings(flags, step, flip=False):
    """godot_gym.py L623-633 then world_generator.py L561, min_difficulty 0."""
    wrapper_d, gen_d, out = 0.0, 0.0, []
    for s in flags:
        ok = (not s) if flip else s
        if ok == 1:
            wrapper_d += step
        elif ok == 0:
            wrapper_d -= step
        wrapper_d = max(min(wrapper_d, 1.0), 0.0)
        gen_d = max(0.0, wrapper_d)
        out.append(gen_d)
    return out


def part_a():
    rng = np.random.default_rng(7)
    # successes more likely than not, so the ceiling roams the whole range
    flags = [int(x) for x in rng.random(20000) < 0.52]
    ours = SuccessGatedDial(0.0, 1.0, 0.01)
    mine = [ours.update(bool(f)) for f in flags]
    ref = avalon_ceilings(flags, 0.01)
    diff = max(abs(a - b) for a, b in zip(mine, ref))
    ctl = max(abs(a - b) for a, b in zip(mine, avalon_ceilings(flags, 0.01, flip=True)))
    print(f"(a) ceilings vs Avalon: max |diff| {diff:.3g} over {len(flags)} lives "
          f"(range seen {min(ref):.2f}-{max(ref):.2f}); control (sign flipped) {ctl:.3g}")
    g = SuccessGatedDial(0.75, 1.0, 0.01, initial=0.9, rng=np.random.default_rng(1))
    dr = np.array([g.draw() for _ in range(10000)])
    u = SuccessGatedDial(0.75, 1.0, 0.01, mode="uniform")
    for f in flags[:500]:
        u.update(bool(f))
    fx = SuccessGatedDial(0.75, 1.0, 0.01, mode="fixed")
    fxd = {fx.draw() for _ in range(100)}
    ok_draw = dr.min() >= 0.75 and dr.max() <= 0.9 and abs(dr.mean() - 0.825) < 0.01
    print(f"    draws at ceiling .9: min {dr.min():.4f} max {dr.max():.4f} mean {dr.mean():.4f}; "
          f"uniform ceiling after 500 updates {u.ceiling}; fixed draws {fxd}")
    return diff == 0.0 and ctl > 0.0 and ok_draw and u.ceiling == 0.75 and fxd == {0.75}


def pilot_world(seed, d=0.75):
    from experiments.jack_env import JackEnv, NEEDS_S_TRP, SPAWN_CLEAR_R
    from experiments.pilot_b import REWARD
    return JackEnv(seed=seed, difficulty=d, reward=REWARD, random_needs=NEEDS_S_TRP,
                   random_layout=True, lazy_contact_audio=True,
                   spawn_clear=SPAWN_CLEAR_R, move_cost_dial=True)


def run_lives(success_fn, lives, seed=153, cap=20):
    from experiments.drives import NU_FLOORFOOD
    from experiments.jack_env import (EXTRA_FOODS, dial_extra_foods, dial_meal,
                                      dial_move_cost)
    from experiments.pilot_b import PilotLives
    env = DialCurriculum(PilotLives(pilot_world(seed), cap=cap), seed=seed,
                         step=0.05, success_fn=success_fn)
    act = np.random.default_rng(seed)
    checks, rebuild_s = [], []
    for k in range(lives):
        t = time.process_time()
        env.reset()
        if k:
            rebuild_s.append(time.process_time() - t)
        je, cur = env.jenv, env._current
        d = cur["d"]
        extra = sum(n in je.w.drives._food for n in EXTRA_FOODS)
        meals = {n: je.w.drives._food[n][1] for n in ("obj0", "obj1") if n in je.w.drives._food}
        dd = 1.0 if je.difficulty is None else je.difficulty
        ok = (abs(dd - d) < 1e-12 and je._seed == cur["world_seed"]
              and 0.75 <= d <= cur["ceiling"] + 1e-12
              and extra == (dial_extra_foods(d) if d < 1.0 else 0)
              and all(abs(v - (dial_meal(NU_FLOORFOOD, d) if d < 1.0 else NU_FLOORFOOD)) < 1e-12
                      for v in meals.values())
              and abs(je.move_cost_scale - (dial_move_cost(d) if d < 1.0 else 1.0)) < 1e-12)
        checks.append(ok)
        while True:
            _o, _r, term, trunc, _i = env.step(act.uniform(-1, 1, je.action_space.shape))
            if term or trunc:
                break
    return env, checks, rebuild_s


def part_b():
    env, chk, rb = run_lives(lambda row: True, 9)
    ceil = [round(h["ceiling"], 4) for h in env.history]
    seeds = [h["world_seed"] for h in env.history]
    ds = [round(h["d"], 3) for h in env.history]
    want = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0]
    climb = ceil == want and len(set(seeds)) == len(seeds) and all(chk)
    print(f"(b) always-success: ceiling at each draw {ceil}; d {ds}; "
          f"world checks {sum(chk)}/{len(chk)}; distinct world seeds {len(set(seeds))}/{len(seeds)}")
    cenv, cchk, _ = run_lives(lambda row: False, 6)
    cceil = {round(h["ceiling"], 4) for h in cenv.history}
    cds = {round(h["d"], 6) for h in cenv.history}
    held = cceil == {0.75} and cds == {0.75} and all(cchk)
    print(f"    control always-failure: ceilings {cceil}, d {cds} (must not climb); "
          f"world checks {sum(cchk)}/{len(cchk)}")
    print(f"    rebuild CPU: median {1000 * np.median(rb):.0f} ms (n={len(rb)})")
    return climb and held


def stream(env, n, seed):
    a = np.random.default_rng(seed)
    o, _ = env.reset()
    rows = [o.copy()]
    for _ in range(n):
        o, r, term, trunc, _ = env.step(a.uniform(-1, 1, env.action_space.shape).astype(np.float32))
        rows.append(np.concatenate([o, [r, float(term)]]))
        if term:
            o, _ = env.reset()
            rows.append(o.copy())
    return rows


def same(x, y):
    return len(x) == len(y) and all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(x, y))


def part_c(n):
    from experiments.pilot_b import PilotLives
    ok_all = True
    for d, ws in ((0.83, 9001), (1.0, 9002)):
        old = pilot_world(153)
        stream(old, 60, 5)                 # the old world has lived a while
        w = DialCurriculum(PilotLives(old, cap=10 ** 9), seed=153)
        w.set_world(d, ws)
        a = stream(old, n, 11)             # step the rebuilt JackEnv itself
        b = stream(pilot_world(ws, d), n, 11)
        c = stream(pilot_world(ws + 1, d), n, 11)
        ok = same(a, b) and not same(a, c)
        ok_all &= ok
        print(f"(c) rebuild at d {d} seed {ws}: identical to a fresh world {same(a, b)} "
              f"({len(a)} rows, {sum(r[-1] for r in a if r.shape[0] > old.observation_space.shape[0])} deaths); "
              f"control seed+1 identical {same(a, c)}")
    return ok_all


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    t = time.process_time()
    ra, rb, rc = part_a(), part_b(), part_c(n)
    print(f"a {ra}  b {rb}  c {rc}  ({time.process_time() - t:.0f} CPU-s)")
    print("PASS" if ra and rb and rc else "FAIL")
    return 0 if ra and rb and rc else 1


if __name__ == "__main__":
    sys.exit(main())
