"""J15 (1) proof: f5's every action is a function of the CURRENT row only.

    python -m experiments.tests.test_f5 [N=400]

(a) REPLAY: one life's rows on smoke seed 153 at d0 (f5 driving, gate 3's
    world), N decisions. The same row handed twice gives the same action; the
    N rows replayed in a shuffled order through ONE used policy object give,
    row for row, the actions of the life; a fresh object gives them too; the
    policy object's attributes are unchanged by acting.
    Control that must fail: f4 (`EyeForager`, which remembers) put through the
    same shuffled replay must differ from its own life on at least one row —
    the replay can see memory.
(b) ROW ONLY: every observation entry after the 256 row floats replaced by
    noise -> the same actions.
(c) PLANTED rows, known answers: a wide-enough near object at 90 deg left is
    pushed at; a post narrower than F5_THIN_M is ignored; a strip FARTHER
    than its flanks is ignored (and the same run NEARER is pushed at); with
    nothing kept, the push is at the middle of the widest open gap.
(d) the shuffled twin g5 is not f5: on the life's rows its actions differ.
"""
import math
import sys

import numpy as np

from experiments import world_card as WC
from experiments.eye_forager import (EyeForager, RowForager, f5_act, make_env)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def life_rows(pol, n):
    env = make_env(153, 0.75, row="f5")
    o, info = env.reset(seed=153)
    pol.new_life()
    obs, acts = [], []
    for _ in range(n):
        a = pol.act(o)
        obs.append(np.array(o, dtype=float)); acts.append(np.array(a))
        o, r, te, tr, info = env.step(a)
        if te or tr:
            break
    return obs, acts


def planted(runs, base_d=0.9, base_t=0.24):
    """A 128-ray row: far wall everywhere, then (first_ray, n_rays, d, tex)."""
    v = np.empty(256)
    v[0::2], v[1::2] = base_d, base_t
    for k0, m, d, t in runs:
        for k in range(k0, k0 + m):
            v[2 * (k % 128)], v[2 * (k % 128) + 1] = d, t
    return v


def ang_of(a):
    return math.degrees(math.atan2(a[7], a[6])) % 360.0


def main():
    ok = True
    pol = RowForager(153)
    obs, acts = life_rows(pol, N)
    n = len(obs)
    twice = sum(int(not np.array_equal(pol.act(o), pol.act(o))) for o in obs)
    order = np.random.RandomState(7).permutation(n)
    before = {k: v for k, v in vars(pol).items()}
    replay = sum(int(not np.array_equal(pol.act(obs[i]), acts[i])) for i in order)
    fresh = RowForager(999)
    fresh_d = sum(int(not np.array_equal(fresh.act(obs[i]), acts[i])) for i in order)
    same_state = before == vars(pol)
    moved = len({tuple(np.round(a[6:8], 6)) for a in acts})
    good = twice == 0 and replay == 0 and fresh_d == 0 and same_state and moved > 5
    ok &= good
    print(f"(a) {n} rows, {moved} distinct pushes: same-row-twice diffs {twice}, "
          f"shuffled replay diffs {replay}, fresh object diffs {fresh_d}, "
          f"attributes unchanged {same_state}  {'OK' if good else 'FAIL'}", flush=True)

    f4 = EyeForager(153)
    obs4, acts4 = life_rows(f4, N)
    f4b = EyeForager(153); f4b.new_life()
    order4 = np.random.RandomState(7).permutation(len(obs4))
    d4 = sum(int(not np.array_equal(f4b.act(obs4[i]), acts4[i])) for i in order4)
    good = d4 > 0
    ok &= good
    print(f"(a) control, f4 under the same shuffled replay: {d4} of {len(obs4)} "
          f"differ  {'OK (memory seen)' if good else 'FAIL (replay is blind)'}", flush=True)

    rng = np.random.RandomState(3)
    nb = 0
    for o, a in zip(obs, acts):
        o2 = o.copy(); o2[256:] = rng.uniform(-1, 1, o2.shape[0] - 256)
        nb += int(not np.array_equal(pol.act(o2), a))
    good = nb == 0
    ok &= good
    print(f"(b) non-row entries replaced by noise: {nb} differing actions  "
          f"{'OK' if good else 'FAIL'}", flush=True)

    # (c) planted rows. ray k sits at 2.8125*k deg; a 4-ray run 30..33 centres 88.6
    a1 = ang_of(f5_act(planted([(30, 4, 0.2, 0.5)])))            # 4 rays, 1.6 m: 0.31 m wide
    a2 = ang_of(f5_act(planted([(30, 4, 0.2, 0.5), (90, 1, 0.1, 0.6)])))   # + a 0.04 m post, nearer
    thin_alone = ang_of(f5_act(planted([(90, 1, 0.1, 0.6)])))
    far_strip = planted([(0, 128, 0.3, 0.24), (60, 4, 0.8, 0.5)])  # strip BEHIND its flanks
    a3 = ang_of(f5_act(far_strip))
    gap = planted([(0, 64, 0.2, 0.24)])                          # open rays 64..127 -> mid 95.5
    a4 = ang_of(f5_act(gap))
    c1 = abs(a1 - 88.59) < 0.1
    c2 = abs(a2 - 88.59) < 0.1
    # post at ray 90 (253.1 deg) is not a target; the gap rule then sees 127 open
    # rays 91..89 whose middle is ray 26 = 73.1 deg, straight away from the post
    c3 = abs(thin_alone - 2.8125 * 26) < 0.1
    c4 = abs(a3 - 2.8125 * 61.5) < 0.1                           # strip not a target: gap rule finds the far rays
    c4 = c4 and abs(ang_of(f5_act(planted([(0, 128, 0.3, 0.24), (60, 4, 0.1, 0.5)])))
                    - 2.8125 * 61.5) < 0.1                       # ...and a NEAR run there IS pushed at
    c5 = abs(a4 - 2.8125 * 95.5) < 0.1
    good = c1 and c2 and c3 and c5 and c4
    ok &= good
    print(f"(c) planted: object {a1:.1f} deg (88.6) {c1}; nearer thin post ignored "
          f"{a2:.1f} {c2}; post alone -> gap {thin_alone:.1f} (73.1) {c3}; far strip {a3:.1f} {c4}; "
          f"widest gap {a4:.1f} (268.6) {c5}  {'OK' if good else 'FAIL'}", flush=True)

    g5 = RowForager(153, shuffle_rays=True)
    dg = sum(int(not np.array_equal(g5.act(o), a)) for o, a in zip(obs, acts))
    good = dg > n // 2
    ok &= good
    print(f"(d) g5 differs from f5 on {dg} of {n} rows  {'OK' if good else 'FAIL'}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
