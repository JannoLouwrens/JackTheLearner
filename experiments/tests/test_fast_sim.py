"""J6(2a) proof: `JackEnv(lazy_contact_audio=True)` changes nothing in the world.

(1) Identity: per seed, the shipped env and the lazy env get the same fixed
    random-action stream; every step the observation, reward, terminated,
    truncated must be exactly equal (np.array_equal / ==), AND every audio
    event the synth makes (t, pair, centroid pos, force — recorded by wrapping
    `_make_event` on both) must be exactly equal, in order. Life ends are
    crossed (e0 small, cap) and the env is reset there, as a learner would.
(2) Control that must FAIL: the lazy env fed the action stream of another
    seed must differ from the shipped env.
(3) Speed: process CPU-seconds per decision, shipped vs lazy, same seed and
    actions (run back to back in one process; box load shared).

argv: [N steps] [j0,alpha]. Hand calibration (identity is cal-independent).

Box results 11-16 15:40 (300 steps x 3 seeds, e0 0.05, cap 40, load ~1.6):
identity 3/3 PASS (144 life/cap ends; 29,805 audio events equal field for
field), control differs. Speed (2a alone): shipped 40.8 dec/CPU-s, lazy 51.8
dec/CPU-s = 1.27x (30-step smoke at load 1.5: 38.1 -> 47.1, 1.24x).
"""
import sys
import time

import numpy as np

from experiments.jack_env import JackEnv
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEEDS = (153, 154, 155)


def actions(rng_seed, n):
    rng = np.random.RandomState(rng_seed)
    return [random_action(rng) for _ in range(n)]


def _hook(env, log):
    s = env.w.synth
    orig = s._make_event

    def rec(t, pair, pos, force):
        log.append((t, pair, np.array(pos), force))
        return orig(t, pair, pos, force)
    s._make_event = rec


def run(seed, acts, kw):
    env = JackEnv(seed=seed, **kw)
    o, _ = env.reset(seed=seed)
    ev = []
    _hook(env, ev)
    out = [o]
    t0 = time.process_time()
    for a in acts:
        o, r, te, tr, _ = env.step(a)
        out.append((o, r, te, tr))
        if te or tr:
            o, _ = env.reset()
            _hook(env, ev)
            out.append(o)
    return out, ev, time.process_time() - t0


def same(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                    and np.array_equal(x, y)):
                return False
            continue
        if not np.array_equal(x[0], y[0]) or x[1:] != y[1:]:
            return False
    return True


def same_ev(a, b):
    return len(a) == len(b) and all(
        x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2]) and x[3] == y[3]
        for x, y in zip(a, b))


def main():
    j0, alpha = (float(x) for x in (sys.argv[2] if len(sys.argv) > 2
                                    else "1.0,1.0").split(","))
    kw = {"j0": j0, "alpha": alpha, "e0": 0.05, "max_episode_decisions": 40}
    ok, t_ref, t_lazy, nev = True, 0.0, 0.0, 0
    for s in SEEDS:
        acts = actions(s * 6553 + 11, N)
        ref, ev_r, tr_ = run(s, acts, kw)
        lz, ev_l, tl_ = run(s, acts, dict(kw, lazy_contact_audio=True))
        t_ref, t_lazy, nev = t_ref + tr_, t_lazy + tl_, nev + len(ev_r)
        g = same(ref, lz) and same_ev(ev_r, ev_l)
        print(f"identity seed {s}: {'PASS' if g else 'FAIL'} "
              f"({len(ev_r)} audio events, {len(ref) - N - 1} life/cap ends)")
        ok &= g and len(ev_r) > 0
    ref, ev_r, _ = run(SEEDS[0], actions(SEEDS[0] * 6553 + 11, N), kw)
    lz, ev_l, _ = run(SEEDS[0], actions(SEEDS[1] * 6553 + 11, N),
                      dict(kw, lazy_contact_audio=True))
    ctrl_differs = not (same(ref, lz) and same_ev(ev_r, ev_l))
    print(f"control (shifted actions) differs: {ctrl_differs}")
    ok &= ctrl_differs
    d = len(SEEDS) * N
    print(f"speed: shipped {d / t_ref:.1f} dec/CPU-s, lazy {d / t_lazy:.1f} "
          f"dec/CPU-s ({t_ref / t_lazy:.2f}x); {nev} events total")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
