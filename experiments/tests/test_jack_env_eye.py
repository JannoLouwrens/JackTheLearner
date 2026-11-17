"""J12 (1) proof: jack_env's `eye` option ("rays16" default, "row128").

    python -m experiments.tests.test_jack_env_eye [N=200]

(a) IDENTITY: JackEnv(eye="rays16") == JackEnv() byte for byte (obs, reward,
    terminated, truncated) on seeds 153-155 and J8's pilot world (seed 156,
    d 0.75, random_layout, random_needs 1.0, terminal_obs "dying"), random
    actions, life ends crossed. Control: another action seed differs.
(b) PLANTED, the row is the old eye plus 112 rays between: same seed, same
    actions, row128 vs rays16. The eye moves no body, so touch, proprio and
    needs must be equal every step, and row ray 8j must read today's ray j's
    distance exactly (texture too, except on the noise panel, whose draw
    order changes). Control that must fail: row ray 8j+4 (half-way between)
    differs from ray j somewhere.
(c) PLANTED, what the row adds: entry rung (d 0.75, random_layout, the
    shipped calibration so the body lives and settles), seeds 157-158, share of decisions on
    which ANY ray hits a floor food — rays16 (= row rays 8j) vs row128. Must
    be strictly more for row128. Shape: vision 256 wide, obs space matches,
    and terminal_obs "dying" (observe_pure through the patched eye) runs.
(d) COST: CPU ms per look, W0's 16-ray `_vision` vs the 128-ray row.

RESULT (2026-11-17, box, nice 19, `200`, 1.9 CPU-min): PASS.
 (a) 4/4 identical, controls differ.  (b) 4/4: other senses equal, ray 8j
 distance equal, texture equal off the panel; ray 8j+4 differs.
 (c) 400 decisions: any ray on a floor food rays16 .287, row128 .905 (the
 LAB's decoder read .528 / .909).  (d) 0.058 vs 0.089 CPU-ms a look —
 +0.03 ms on a ~20 ms decision (0.15%).
 First (c) used the hand calibration j0 = alpha = 1 + random_needs: the body
 died every ~3 decisions, never settled to eye height, and NO eye saw food
 (0/0) — a test apparatus error, fixed by the shipped calibration.
"""
import sys
import time

import numpy as np

from experiments.jack_env import JackEnv
from experiments.w0 import random_action

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BASE = {"j0": 1.0, "alpha": 1.0, "e0": 0.05, "max_episode_decisions": 40}
PILOT = {"j0": 1.0, "alpha": 1.0, "max_episode_decisions": 40,
         "difficulty": 0.75, "random_layout": True, "random_needs": 1.0,
         "terminal_obs": "dying"}
CASES = [(153, BASE), (154, BASE), (155, BASE), (156, PILOT)]


def roll(env, seed, act_seed, n, hook=None):
    rng = np.random.RandomState(act_seed)
    o, _ = env.reset(seed=seed)
    out = [o.copy()]
    for _ in range(n):
        o, r, te, tr, _ = env.step(random_action(rng))
        out.append((o.copy(), r, te, tr))
        if hook:
            hook(env)
        if te or tr:
            o, _ = env.reset()
            out.append(o.copy())
    return out


def same(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                return False
        elif not (np.array_equal(x[0], y[0]) and x[1:] == y[1:]):
            return False
    return True


def main():
    ok = True
    # (a)
    for seed, kw in CASES:
        ref = roll(JackEnv(seed=seed, **kw), seed, seed + 1000, N)
        got = roll(JackEnv(seed=seed, eye="rays16", **kw), seed, seed + 1000, N)
        ctl = roll(JackEnv(seed=seed, **kw), seed, seed + 2000, N)
        ida, ctla = same(ref, got), not same(ref, ctl)
        print(f"(a) seed {seed}: identical={ida} control differs={ctla}")
        ok &= ida and ctla
    # (b) + (c)
    hits16 = hits128 = steps = 0
    for seed, kw in CASES:
        e16 = JackEnv(seed=seed, **kw)
        e128 = JackEnv(seed=seed, eye="row128", **kw)
        assert e128.observation_space.shape[0] == e16.observation_space.shape[0] + 224
        foods = {gid for name, (gid, _) in e128.w.drives._food.items()
                 if name in e128.floor_foods}
        panel = e128.w.panel_gid
        rng16, rng128 = np.random.RandomState(seed + 3000), np.random.RandomState(seed + 3000)
        o16, _ = e16.reset(seed=seed)
        o128, _ = e128.reset(seed=seed)
        eq_rest = eq_dist = eq_tex = True
        mid_differs = False
        for t in range(N + 1):
            v16, rest16 = o16[:32], o16[32:]
            v128, rest128 = o128[:256], o128[256:]
            assert v128.shape[0] == 256
            eq_rest &= np.array_equal(rest16, rest128)
            eq_dist &= np.array_equal(v16[0::2], v128[0::2][0::8])
            gid8 = e128.w._row_gid[0::8]
            nonpanel = gid8 != panel
            eq_tex &= np.array_equal(v16[1::2][nonpanel], v128[1::2][0::8][nonpanel])
            mid_differs |= not np.array_equal(v16[0::2], v128[0::2][4::8])
            if t == N:
                break
            a = random_action(rng16)
            random_action(rng128)
            o16, _, te, tr, _ = e16.step(a)
            o128, _, te2, tr2, _ = e128.step(a)
            assert (te, tr) == (te2, tr2)
            if te or tr:
                o16, _ = e16.reset()
                o128, _ = e128.reset()
        print(f"(b) seed {seed}: other senses equal={eq_rest} ray8j distance "
              f"equal={eq_dist} texture equal (off panel)={eq_tex} | control "
              f"ray8j+4 differs={mid_differs}")
        ok &= eq_rest and eq_dist and eq_tex and mid_differs
    for seed in (157, 158):
        env = JackEnv(seed=seed, eye="row128", difficulty=0.75, random_layout=True)
        foods = [env.w.drives._food[n][0] for n in env.floor_foods
                 if n in env.w.drives._food]
        rng = np.random.RandomState(seed + 4000)
        env.reset(seed=seed)
        for _ in range(N):
            _, _, te, tr, _ = env.step(random_action(rng))
            g = env.w._row_gid
            hits16 += bool(np.isin(g[0::8], foods).any())
            hits128 += bool(np.isin(g, foods).any())
            steps += 1
            if te or tr:
                env.reset()
    more = hits128 > hits16
    print(f"(c) d 0.75, {steps} decisions: any ray on a floor food "
          f"rays16 {hits16/steps:.3f} vs row128 {hits128/steps:.3f} -> more={more}")
    ok &= more
    # (d)
    env = JackEnv(seed=153, eye="row128", **BASE)
    w = env.w
    from experiments.fast_sim import _row_vision
    cls_vision = type(w)._vision
    for name, f in (("rays16", lambda: cls_vision(w)), ("row128", lambda: _row_vision(w))):
        f()
        t0 = time.process_time()
        for _ in range(2000):
            f()
        print(f"(d) {name}: {1000 * (time.process_time() - t0) / 2000:.4f} CPU-ms a look")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
