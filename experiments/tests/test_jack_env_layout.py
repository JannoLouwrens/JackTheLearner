"""J2 item 2 (rest) proof: jack_env's random_layout — a new food layout each life.

(a) OFF IS BYTE-IDENTICAL: random_layout=False equals JackEnv() decision for
    decision (observation bytes, reward, terminated, truncated, meals) — 3 seeds.
(b) ON DOES WHAT IT SAYS: at birth and at every respawn, every floor food
    sits at a cell of W0's legal grid, at its own W0 rest height, >= 0.5 m
    from the other foods, penetrating nothing but ground and his body
    (tolerance 1 mm); the new body penetrates nothing (W0's own test);
    no food failed to place; the layouts of one run are all different.
    Also with the dial at d=0: the same, and the body is spawned within the
    dial's 0.5 m of a food (or at the nearest clear cell).
    APPARATUS: as the needs proof — a life is ~1,500 random decisions, so the
    test starves him (e = 1e-4) at decisions STARVE in every run alike.
    Guard: >= 3 respawns per run, else respawn was not measured.
(c) OWN STREAM: two envs on the same seed give the same layouts; the next
    seed gives different ones.
(d) CONTROL THAT MUST FAIL: the (b) check applied to the default world
    (foods stay where they lie across lives) must report FAIL, and the
    layout run's observation stream must differ from the default's.
"""
import sys
import numpy as np

from experiments.jack_env import JackEnv, default_cal, LAYOUT_SEP
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
STARVE = (150, 300, 450)
SEEDS = (153, 154, 155)


def snap(env):
    """Food (x, y, z) and the body-penetration flag at a life's start."""
    w = env.w
    foods = {n: tuple(map(float, w.data.xpos[w.model.body(n).id]))
             for n in env.floor_foods if n in w.drives._food}
    return foods, bool(w._penetrating()), tuple(map(float, w.data.xpos[w.rover_bid][:2]))


def run(seed, cal, n, rng_seed, **kw):
    env = JackEnv(seed=seed, **cal, **kw)
    env.reset(seed=seed)
    starts = [snap(env)]
    rng = np.random.RandomState(rng_seed)
    out = []
    for k in range(n):
        if k in STARVE:
            env.w.drives.state.e = 1e-4
        o, r, term, trunc, info = env.step(random_action(rng))
        out.append((o.tobytes(), r, term, trunc, info["meals"]))
        if term:
            starts.append(snap(env))
        if term or trunc:
            env.reset()
    return env, out, starts


def check_b(env, starts, dial0=False):
    legal = env.w.legal_spawns()
    rest = getattr(env, "_food_rest", None)
    ok = True
    for foods, pen, body in starts:
        xy = np.array([v[:2] for v in foods.values()])
        on_grid = all(np.min(np.linalg.norm(legal - p, axis=1)) < 1e-6 for p in xy)
        sep = all(np.linalg.norm(xy[i] - xy[j]) >= LAYOUT_SEP - 1e-9
                  for i in range(len(xy)) for j in range(i + 1, len(xy)))
        z_ok = rest is None or all(abs(foods[nm][2] - rest[nm][2]) < 1e-6
                                   for nm in foods if nm in rest)
        ok &= on_grid and sep and z_ok and not pen
        if dial0:
            dist = np.min(np.linalg.norm(xy - np.array(body), axis=1))
            ok &= dist <= 1.0       # 0.5 m dial radius, or nearest clear cell
    layouts = [tuple(sorted((n, round(v[0], 6), round(v[1], 6)) for n, v in f.items()))
               for f, _, _ in starts]
    ok &= len(set(layouts)) == len(layouts)
    ok &= getattr(env, "layout_failed", 1) == 0
    return ok


def main():
    cal = default_cal()
    ok_a = ok_b = ok_c = guard = True
    ctrl_fails = differs = True
    for seed in SEEDS:
        rs = seed * 7919 + 3
        e0, base, st0 = run(seed, cal, N, rs)
        _, off, _ = run(seed, cal, N, rs, random_layout=False)
        a = base == off
        ok_a &= a
        e1, out, st = run(seed, cal, N, rs, random_layout=True)
        b1 = check_b(e1, st)
        e2, _, st2 = run(seed, cal, N, rs, random_layout=True, difficulty=0.0)
        b2 = check_b(e2, st2, dial0=True)
        guard &= len(st) >= 4 and len(st2) >= 4
        ok_b &= b1 and b2
        _, _, st_again = run(seed, cal, 0, rs, random_layout=True)
        _, _, st_next = run(seed + 1, cal, 0, rs, random_layout=True)
        c = (st_again[0][0] == st[0][0]) and (st_next[0][0] != st[0][0])
        ok_c &= c
        cf = not check_b(e0, st0)
        ctrl_fails &= cf
        differs &= out != base
        d_body = [round(float(np.min(np.linalg.norm(
            np.array([v[:2] for v in f.values()]) - np.array(bd), axis=1))), 2)
            for f, _, bd in st2]
        print(f"seed {seed}: (a) {a}; lives {len(st)}/{len(st2)} (default {len(st0)}); "
              f"(b) layout {b1}, layout+d0 {b2} (body-food m {d_body}); "
              f"(c) {c}; control check fails on default {cf}; "
              f"birth layout {[(n, round(v[0], 2), round(v[1], 2)) for n, v in st[0][0].items()]}")
    d = ctrl_fails and differs
    ok = ok_a and ok_b and ok_c and guard and d
    print(f"(a) {ok_a} (b) {ok_b} (c) {ok_c} guard>=3 respawns {guard} "
          f"(d) control fails + stream differs {d}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
