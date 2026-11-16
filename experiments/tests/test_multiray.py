"""J6(3) proof: `JackEnv(multiray_vision=True)` changes nothing in the world.

(1) Identity: per seed, the shipped env and the multiray env get the same
    fixed random-action stream; every step the observation, reward,
    terminated, truncated must be exactly equal (np.array_equal / ==). Life
    ends are crossed (e0 small, cap) and reset as a learner would. Seeds
    153-155 in the default world, plus seed 156 in J8's pilot world
    (difficulty 0.75, random_layout, random_needs 1.0) with
    terminal_obs="dying" (observe_pure must restore the stream through the
    patched eye too).
(2) The panel's noise draw is exercised: the number of rays that hit the
    noise panel is counted (each one consumes a world random draw); the run
    must see at least one, or the identity says nothing about the stream.
(3) Control that must FAIL: the multiray env fed another seed's actions
    must differ from the shipped env.
(4) Speed: process CPU-seconds per decision, shipped vs multiray.

argv: [N steps]. Hand calibration j0 = alpha = 1 (identity is cal-independent).

Box results 11-16 17:25 (300 steps x 4 worlds, load ~1): identity 4/4 PASS
(212 life/cap ends; 824 panel-ray draws consumed identically), control
differs. Speed: shipped 39.3, multiray 39.2 dec/CPU-s (0.999x). `run stale`:
0 lines (jack_env / fast_sim are no certificate's dependency). Raw ray check beforehand: 2,400 rays, mj_multiRay == mj_ray loop in
distance and geom id bit for bit; `_vision` 0.057 ms vs mj_multiRay call
0.022 ms — under 0.3% of a decision, so not a speed lever for the 16-ray eye.
"""
import sys
import time

import numpy as np

from experiments.jack_env import JackEnv
from experiments.w0 import random_action
import experiments.fast_sim as FS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BASE = {"j0": 1.0, "alpha": 1.0, "e0": 0.05, "max_episode_decisions": 40}
PILOT = {"j0": 1.0, "alpha": 1.0, "max_episode_decisions": 40,
         "difficulty": 0.75, "random_layout": True, "random_needs": 1.0,
         "terminal_obs": "dying"}
WORLDS = [(153, BASE), (154, BASE), (155, BASE), (156, PILOT)]

PANEL_HITS = [0]
_orig = FS._multiray_vision


def _counting(self):
    out = _orig(self)
    if self.panel_gid >= 0:
        PANEL_HITS[0] += int(np.sum(self._mr_gid == self.panel_gid))
    return out


FS._multiray_vision = _counting


def actions(rng_seed, n):
    rng = np.random.RandomState(rng_seed)
    return [random_action(rng) for _ in range(n)]


def run(seed, acts, kw):
    env = JackEnv(seed=seed, **kw)
    o, _ = env.reset(seed=seed)
    out = [o]
    t0 = time.process_time()
    for a in acts:
        o, r, te, tr, _ = env.step(a)
        out.append((o, r, te, tr))
        if te or tr:
            o, _ = env.reset()
            out.append(o)
    return out, time.process_time() - t0


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


def main():
    ok, t_ref, t_mr = True, 0.0, 0.0
    for s, kw in WORLDS:
        acts = actions(s * 6553 + 11, N)
        ref, a = run(s, acts, kw)
        before = PANEL_HITS[0]
        mr, b = run(s, acts, dict(kw, multiray_vision=True))
        t_ref, t_mr = t_ref + a, t_mr + b
        g = same(ref, mr)
        print(f"identity seed {s} ({'pilot' if kw is PILOT else 'default'}): "
              f"{'PASS' if g else 'FAIL'} ({len(ref) - N - 1} life/cap ends, "
              f"{PANEL_HITS[0] - before} panel-ray draws)")
        ok &= g
    print(f"panel draws exercised: {PANEL_HITS[0]}")
    ok &= PANEL_HITS[0] > 0
    s, kw = WORLDS[0]
    ref, _ = run(s, actions(s * 6553 + 11, N), kw)
    mr, _ = run(s, actions(WORLDS[1][0] * 6553 + 11, N),
                dict(kw, multiray_vision=True))
    ctrl = not same(ref, mr)
    print(f"control (shifted actions) differs: {ctrl}")
    ok &= ctrl
    d = len(WORLDS) * N
    print(f"speed: shipped {d / t_ref:.1f} dec/CPU-s, multiray {d / t_mr:.1f} "
          f"dec/CPU-s ({t_ref / t_mr:.3f}x)")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
