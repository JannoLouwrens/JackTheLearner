"""J2 item 1 (part) proof: jack_env's difficulty dial — meal size + regrowth.

(a) OFF IS BYTE-IDENTICAL: difficulty=1.0 equals difficulty=None (today's
    world) decision for decision (observation bytes, reward, terminated,
    truncated, meals) — 3 seeds.
(b) d=0 and d=0.5 do what they say: floor-food meal size is
    1.0 + d*(0.08-1.0) (apple untouched), and after every floor eat event the
    regrowth timer left is within (d*66.9 - 0.2, d*66.9] s; at d=1 the same
    measurement reads (66.7, 66.9]. Pooled guard: at least one floor eat at
    every d, else (b) measured nothing.
(c) Control that must fail: d=0 does NOT reproduce today's world on a seed
    where food is eaten (else the dial changes nothing).
"""
import sys
import numpy as np

from experiments.drives import NU_FLOORFOOD, NU_APPLE, RESPAWN_FLOORFOOD_S
from experiments.jack_env import JackEnv, default_cal, FLOOR_FOODS
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEEDS = (153, 154, 155)


def run(seed, cal, n, rng_seed, difficulty):
    env = JackEnv(seed=seed, **cal, difficulty=difficulty)
    env.reset(seed=seed)
    dl = env.w.drives
    nus = {k: v[1] for k, v in dl._food.items()}
    rng = np.random.RandomState(rng_seed)
    out, left = [], []
    for _ in range(n):
        before = dict(dl.ate_total)
        o, r, term, trunc, info = env.step(random_action(rng))
        for k in FLOOR_FOODS:
            if k in dl.ate_total and dl.ate_total[k] > before[k]:
                left.append(dl._respawn_at[k] - dl.t)
        out.append((o.tobytes(), r, term, trunc, info["meals"]))
        if term or trunc:
            env.reset()
    return out, nus, left


def main():
    cal = default_cal()
    ok, ctrl_differs = True, False
    eats = {0.0: 0, 0.5: 0, 1.0: 0}
    for s in SEEDS:
        rs = s * 6553 + 11
        base, nus0, _ = run(s, cal, N, rs, None)
        one, _, left1 = run(s, cal, N, rs, 1.0)
        a = base == one
        b = True
        for d, (out, nus, left) in ((1.0, (one, nus0, left1)),
                                    (0.5, run(s, cal, N, rs, 0.5)),
                                    (0.0, run(s, cal, N, rs, 0.0))):
            want = 1.0 + d * (NU_FLOORFOOD - 1.0)
            b &= all(abs(nus[k] - want) < 1e-12 for k in FLOOR_FOODS if k in nus)
            b &= ("apple" not in nus) or nus["apple"] == NU_APPLE
            R = d * RESPAWN_FLOORFOOD_S
            b &= all(R - 0.2 - 1e-9 < x <= R + 1e-9 for x in left)
            eats[d] += len(left)
            if d == 0.0:
                same = all(x[0] == y[0] for x, y in zip(base, out))
                if base[-1][4] > 0 or out[-1][4] > 0:
                    ctrl_differs |= not same
                print(f"seed {s}: meals in {N} decisions d=1 {base[-1][4]}, "
                      f"d=0 {out[-1][4]}; regrowth left at d=0 "
                      f"{['%.2f' % x for x in left][:6]}")
        print(f"seed {s}: (a) d=1.0==None {'PASS' if a else 'FAIL'}; "
              f"(b) meal size + regrowth {'PASS' if b else 'FAIL'}")
        ok &= a and b
    print("floor eats seen per d (guard for (b)):", eats)
    print("(c) control d=0 differs from today's world:", ctrl_differs)
    ok &= all(v > 0 for v in eats.values()) and ctrl_differs
    print("PROOF:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
