"""J2 item 1 (food count) proof: jack_env's difficulty dial adds floor foods.

(a) OFF IS BYTE-IDENTICAL: difficulty=1.0 equals difficulty=None decision for
    decision (observation bytes, reward, terminated, truncated, meals), and
    the drive layer's food set is exactly {apple, obj0, obj1} in both.
(b) The count is what it says: at d = 0, 0.25, 0.6, 0.9 the food set is
    {apple, obj0, obj1} + those of the first round-half-up(3*(1-d)) of
    obj2..obj4 that the seed's world carries (W0 draws 4 or 6 objects per
    seed: up to 5 / 4 / 3 / 2 floor foods; seed 155 has no obj4), each extra's meal = dial_meal(0.08, d), and
    every extra eat event leaves its regrowth timer in (d*66.9-0.2, d*66.9].
    Pooled guard: at least one EXTRA-food eat event at d=0 across seeds, else
    (b)'s eat clause measured nothing.
(c) Control that must fail: with the extras stripped back out at d=0 (the
    previous commit's world), the extras are never eaten and the
    decision stream differs from the d=0 run on some seed — else adding them
    changed nothing.
"""
import sys
import numpy as np

from experiments.drives import NU_FLOORFOOD, RESPAWN_FLOORFOOD_S
from experiments.jack_env import (JackEnv, default_cal, EXTRA_FOODS,
                                  dial_meal, dial_extra_foods)
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEEDS = (153, 154, 155)


def run(seed, cal, n, rng_seed, difficulty, strip=False):
    env = JackEnv(seed=seed, **cal, difficulty=difficulty)
    if strip:
        # control: the world without the food-count part (extras unregistered
        # before the first decision; birth placement then saw them, which
        # only moves the start — the stream is still compared)
        for k in EXTRA_FOODS:
            env.w.drives._food.pop(k, None)
        env.floor_foods = tuple(k for k in env.floor_foods if k not in EXTRA_FOODS)
    env.reset()
    dl = env.w.drives
    foods = dict(dl._food)
    rng = np.random.RandomState(rng_seed)
    out, left, extra_eats = [], [], 0
    for _ in range(n):
        before = dict(dl.ate_total)
        o, r, term, trunc, info = env.step(random_action(rng))
        for k in EXTRA_FOODS:
            if k in dl._food and dl.ate_total[k] > before.get(k, 0):
                extra_eats += 1
                left.append(dl._respawn_at[k] - dl.t)
        out.append((o.tobytes(), r, term, trunc, info["meals"]))
        if term or trunc:
            env.reset()
    run.last_ate = dict(dl.ate_total)
    return out, foods, left, extra_eats


def main():
    cal = default_cal()
    ok, ctrl_differs, ctrl_extra = True, False, 0
    pooled_d0 = 0
    for s in SEEDS:
        rs = s * 6553 + 11
        base, f_none, _, _ = run(s, cal, N, rs, None)
        one, f_one, _, _ = run(s, cal, N, rs, 1.0)
        e0 = JackEnv(seed=s, **cal)
        names = {e0.w.model.geom(i).name for i in range(e0.w.model.ngeom)}
        a = base == one and set(f_none) == set(f_one) == {"apple", "obj0", "obj1"}
        b = True
        for d in (0.0, 0.25, 0.6, 0.9):
            out, foods, left, ex = run(s, cal, N, rs, d)
            k = dial_extra_foods(d)
            have = [e for e in EXTRA_FOODS[:k] if e in names]
            want = {"apple", "obj0", "obj1", *have}
            b &= set(foods) == want and k == {0.0: 3, 0.25: 2, 0.6: 1, 0.9: 0}[d]
            b &= all(abs(foods[e][1] - dial_meal(NU_FLOORFOOD, d)) < 1e-12
                     for e in have)
            R = d * RESPAWN_FLOORFOOD_S
            b &= all(R - 0.2 - 1e-9 < x <= R + 1e-9 for x in left)
            if d == 0.0:
                pooled_d0 += ex
                d0_ate = run.last_ate
                c_out, _, _, c_ex = run(s, cal, N, rs, 0.0, strip=True)
                c_ate = run.last_ate
                ctrl_extra += c_ex
                ctrl_differs |= any(x != y for x, y in zip(out, c_out))
                print(f"seed {s}: d=0 meals {out[-1][4]} (extra-food eats "
                      f"{ex}); stripped control meals {c_out[-1][4]} "
                      f"(extra eats {c_ex}); d=1 meals {base[-1][4]}")
                print(f"   per food d=0 {d0_ate}; stripped {c_ate}")
        print(f"seed {s}: (a) d=1.0==None, foods unchanged "
              f"{'PASS' if a else 'FAIL'}; (b) count/meal/regrowth "
              f"{'PASS' if b else 'FAIL'}")
        ok &= a and b
    print("extra-food eats at d=0 pooled (guard for (b)):", pooled_d0)
    print("(c) control: stripped extras eaten", ctrl_extra,
          "times; stream differs from d=0:", ctrl_differs)
    ok &= pooled_d0 > 0 and ctrl_extra == 0 and ctrl_differs
    print("PROOF:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
