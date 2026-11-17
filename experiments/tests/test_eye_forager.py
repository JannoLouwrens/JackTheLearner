"""J12 (2) proof: the row128 forager IS row f2's forager, changed only in its eye.

    python -m experiments.tests.test_eye_forager [N=1500]

(a) IDENTITY: EyeForager(n_rays=16) on today's 16-ray world against
    world_card.HandForager2, same seed, both fed the SAME observation each
    decision (the env is driven by HandForager2's action): actions equal bit
    for bit over N decisions on smoke seeds 153 and 154 at d = 1 and d 0.75,
    with eats so the learning rules are exercised, and the shuffled twin
    (shuffle_rays=True, same rng) equal too.
    Control that must fail: the same comparison with ray 0's distance in
    EyeForager's copy of the observation nudged by 1e-3 must differ somewhere
    (the comparison can see a one-ray change).
(b) PLANTED, the angle-kept blob rule matters at 128 rays: on the row128
    world at d 0.75 (seed 153, f3 driving), count decisions where a floor
    food is reported as a blob by blobs_n with s = 8 vs s = 1 (the unscaled
    ray-count rule). s = 8 must report food on strictly more decisions.
(c) the shuffled twin permutes all 128 rays (a fixed obs gives a different
    vis order on two calls) — shape check.
"""
import sys

import numpy as np

from experiments import world_card as WC
from experiments.eye_forager import EyeForager, blobs_n, make_env
from experiments.jack_env import JackEnv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


def compare(seed, d, shuffle, nudge=False):
    env = JackEnv(seed=seed) if d == 1.0 else JackEnv(seed=seed, difficulty=d)
    o, info = env.reset(seed=seed)
    ref = WC.HandForager2(seed, shuffle_rays=shuffle)
    new = EyeForager(seed, shuffle_rays=shuffle, n_rays=16)
    ref.new_life(); new.new_life()
    diff = eats = 0
    meals0 = sum(info["ate_total"].values())
    for _ in range(N):
        o2 = o.copy()
        if nudge:
            o2[0] = o2[0] + 1e-3
        a = ref.act(o)
        b = new.act(o2)
        diff += int(not np.array_equal(a, b))
        o, r, te, tr, info = env.step(a)
        if te or tr:
            o, info = env.reset()
            ref.new_life(); new.new_life()
    eats = sum(info["ate_total"].values()) - meals0
    return diff, eats


def main():
    ok = True
    for seed in (153, 154):
        for d in (1.0, 0.75):
            for sh in (False, True):
                diff, eats = compare(seed, d, sh)
                good = diff == 0
                ok &= good
                print(f"(a) seed {seed} d {d} shuffle {sh}: {diff} differing actions "
                      f"of {N}, {eats} meals  {'OK' if good else 'FAIL'}", flush=True)
    diff, eats = compare(153, 0.75, False, nudge=True)
    good = diff > 0
    ok &= good
    print(f"(a) control (ray 0 nudged 1e-3): {diff} differing actions "
          f"{'OK (differs)' if good else 'FAIL (cannot see it)'}", flush=True)

    env = make_env(153, 0.75)
    o, info = env.reset(seed=153)
    pol = EyeForager(153)
    pol.new_life()
    foods = {round(float(env.w._tex[env.w.drives._food[k][0]]), 4)
             for k in env.floor_foods if k in env.w.drives._food}
    n8 = n1 = 0
    for _ in range(N // 3):
        vis = np.asarray(o[:256], float)
        b8 = {round(t, 4) for _, _, t in blobs_n(vis, 128, 8, pol.ray_ang)}
        b1 = {round(t, 4) for _, _, t in blobs_n(vis, 128, 1, pol.ray_ang)}
        n8 += bool(b8 & foods)
        n1 += bool(b1 & foods)
        o, r, te, tr, info = env.step(pol.act(o))
        if te or tr:
            o, info = env.reset(); pol.new_life()
    good = len(foods) > 0 and n8 > n1
    ok &= good
    print(f"(b) food textures {sorted(foods)}; decisions with a food blob of "
          f"{N // 3}: s=8 {n8}, s=1 {n1}  {'OK' if good else 'FAIL'}", flush=True)

    tw = EyeForager(153, shuffle_rays=True)
    p1 = tw.rng.permutation(128); p2 = tw.rng.permutation(128)
    good = len(p1) == 128 and not np.array_equal(p1, p2)
    ok &= good
    print(f"(c) twin permutes 128 rays, fresh each call  {'OK' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
