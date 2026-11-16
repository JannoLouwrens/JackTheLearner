"""J6(1) proof: `make_vec` (n JackEnvs in n processes) changes nothing in the world.

(1) Identity: 3 seeds in one 3-process AsyncVectorEnv, each worker fed its own
    fixed random-action stream, vs a single `JackEnv` with the same seed and
    the same actions, manual reset on each life end. Every step: observation
    (the next life's first after a death), reward, terminated, truncated,
    and at life ends `info["final_obs"]` == the single env's terminal
    observation — exact (np.array_equal), with terminal_obs "reborn" AND
    "dying", and with a cap (max_episode_decisions) so truncation is crossed.
(2) Control that must FAIL: the same comparison with worker 0's action stream
    shifted by one seed must differ.
(3) Speed: decisions per wall-second and per CPU-second (process CPU of the
    parent + children), single env vs make_vec(n) for the n asked.

argv: [N steps] [j0,alpha] [speed n]. The calibration is given by hand while
LC.02 is stale (default_cal() refuses; J9's edit) — identity does not depend
on its value: both sides get the same one.

Box results 11-16 13:41 (300 steps, e0 0.05, cap 40; 144 life/cap ends per
mode): identity 6/6 PASS (reborn and dying), controls 2/2 differ.
Speed, 4 shared ARM cores at load 3.8 (other runs) — a floor, not the
vector's number: single 35.3 dec/wall-s (35.9/CPU-s); make_vec(2) 56.2
dec/wall-s (33.3/CPU-s, worker start-up included) = 1.59x wall, per-CPU
unchanged. The 4-process row belongs on a Kaggle CPU kernel (4 free cores).
"""
import os
import sys
import time
import resource

import numpy as np

from experiments.jack_env import JackEnv, make_vec
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEEDS = (153, 154, 155)


def actions(rng_seed, n):
    rng = np.random.RandomState(rng_seed)
    return [random_action(rng) for _ in range(n)]


def single(seed, acts, kw):
    env = JackEnv(seed=seed, **kw)
    o, _ = env.reset(seed=seed)
    out = [("reset", o)]
    for a in acts:
        o, r, te, tr, _ = env.step(a)
        fin = None
        if te or tr:
            fin = o
            o, _ = env.reset()
        out.append((o, r, te, tr, fin))
    return out


def vec(seeds, acts_per, kw, n):
    ve = make_vec(len(seeds), seeds=seeds, **kw)
    try:
        o, _ = ve.reset(seed=list(seeds))
        outs = [[("reset", o[i])] for i in range(len(seeds))]
        for t in range(n):
            a = np.stack([acts_per[i][t] for i in range(len(seeds))])
            o, r, te, tr, info = ve.step(a)
            for i in range(len(seeds)):
                fin = None
                if te[i] or tr[i]:
                    fin = info["final_obs"][i]
                outs[i].append((o[i], float(r[i]), bool(te[i]), bool(tr[i]), fin))
        return outs
    finally:
        ve.close()


def same(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        xr, yr = isinstance(x[0], str), isinstance(y[0], str)
        if xr or yr:
            if not (xr and yr and np.array_equal(x[1], y[1])):
                return False
            continue
        if not np.array_equal(x[0], y[0]):
            return False
        if x[1] != y[1] or x[2] != y[2] or x[3] != y[3]:
            return False
        if (x[4] is None) != (y[4] is None):
            return False
        if x[4] is not None and not np.array_equal(x[4], y[4]):
            return False
    return True


def cpu_s():
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def main():
    j0, alpha = (float(x) for x in (sys.argv[2] if len(sys.argv) > 2
                                    else "1.0,1.0").split(","))
    base = {"j0": j0, "alpha": alpha, "e0": 0.05}
    ok = True
    for mode in ("reborn", "dying"):
        kw = dict(base, terminal_obs=mode, max_episode_decisions=40)
        acts = [actions(s * 6553 + 11, N) for s in SEEDS]
        v = vec(SEEDS, acts, kw, N)
        ends = 0
        for i, s in enumerate(SEEDS):
            ref = single(s, acts[i], kw)
            ends += sum(1 for x in ref[1:] if x[4] is not None)
            good = same(v[i], ref)
            print(f"identity {mode} seed {s}: {'PASS' if good else 'FAIL'}")
            ok &= good
        print(f"  life/cap ends crossed ({mode}): {ends}")
        ok &= ends > 0
        # control: worker 0 fed a different stream must NOT match its reference
        acts_c = [actions(SEEDS[0] * 6553 + 12, N)] + acts[1:]
        vc = vec(SEEDS, acts_c, kw, N)
        ctrl_differs = not same(vc[0], single(SEEDS[0], acts[0], kw))
        print(f"control {mode} (shifted stream differs): "
              f"{'PASS' if ctrl_differs else 'FAIL'}")
        ok &= ctrl_differs
    print("PROOF:", "PASS" if ok else "FAIL")

    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    T = 400
    kw = dict(base)
    acts = actions(7, T)
    c0, t0 = cpu_s(), time.time()
    single(SEEDS[0], acts, kw)
    w1, c1 = time.time() - t0, cpu_s() - c0
    seeds = [SEEDS[0] + k for k in range(n)]
    ap = [actions(7 + k, T) for k in range(n)]
    c0, t0 = cpu_s(), time.time()
    vec(seeds, ap, kw, T)
    wn, cn = time.time() - t0, cpu_s() - c0
    print(f"speed single: {T / w1:.1f} dec/wall-s, {T / max(c1, 1e-9):.1f} dec/CPU-s")
    print(f"speed vec{n}: {n * T / wn:.1f} dec/wall-s, {n * T / max(cn, 1e-9):.1f} "
          f"dec/CPU-s (includes worker start-up; load {os.getloadavg()[0]:.1f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
