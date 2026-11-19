"""Proof: the dial race's two missing parts in experiments/dial_curriculum.py
(rule docs/research/prereg/dial_gated_vs_uniform.md; ORDERS JACK Next-up 1).

    python -m experiments.tests.test_dial_race_parts

(a) SHAM COIN. sham_success_fn(seed, p): over 20,000 lives the up-rate is
    within 0.01 of p (p = 0.3, 0.7); the SAME sequence whether every row is
    sustained or none is (it never reads the life); seed s twice = identical,
    seed s+1 differs. CONTROL that must fail "ignores the row": `sustained`.
(b) COMMON EVAL WORLDS. eval_plan(s, n) has n*2*20 entries, every seed in
    [40,000, 41,000), 20 per d per point; same s -> same sha (whatever arm
    calls it: the arm is not an argument); s+1 -> another sha.
(c) EVAL LIVES IN HIS WORLD (pilot world, body eye/motor, food_floor_only,
    native_substep; cap 15): every life is built at its d (read off the env)
    and its seed; the same act twice -> identical rows.
(d) LEARNING PAUSED, NOTHING LEAKED (Brain B, learning_starts 100, T 400,
    eval every 100, 1 life per d, cap 10): the actor's parameters, the replay
    buffer's position and VecNormalize's obs count after learn() are
    IDENTICAL with the eval callback and without it, and 4 points x 2 lives
    were played. CONTROL that must fail: a planted leaky eval that plays its
    lives through the TRAINING env — parameters or buffer differ.
Runs in /data/venvs/jack-std (sbx/jax): nice -n 19 taskset -c 3.

RESULT (2026-11-19 08:43, box, ~70 CPU-s): PASS.
 (a) rates .3019 / .7006; row-blind True; seed+1 differs; control reads row.
 (b) 480 entries, all in band, 20 per (point, d); sha 521 214e83e2bf05ca19
     twice, 522 ddff965b53a30749.
 (c) built (.75, 40011) (None, 40012) (.75, 40013); identical rows x2.
 (d) none vs eval: buffer 416 = 416, obs count 417 = 417, actor params
     identical, 8 eval lives at points 0-3 (plan sha e1ca19b77a0f0ad4).
     CONTROL leaky: obs count 457, params differ -> caught.
 First run FAILED (d) on the test's own count (expected 3 points; T 400 /
 every 100 = 4) — the expectation was fixed, nothing in the code moved.
"""
from __future__ import annotations

import sys

import numpy as np

from experiments import dial_curriculum as dc


def world():
    from experiments.pilot_b2 import pilot_world
    return pilot_world("K", True, "body", "body", True)


def part_a() -> bool:
    ok = True
    for p in (0.3, 0.7):
        c = dc.sham_success_fn(521, p)
        rate = np.mean([c({"sustained": False}) for _ in range(20_000)])
        print(f"  (a) p={p}: rate {rate:.4f}")
        ok &= abs(rate - p) < 0.01
    a = dc.sham_success_fn(522, 0.5); b = dc.sham_success_fn(522, 0.5)
    yes = [a({"sustained": True}) for _ in range(500)]
    no = [b({"sustained": False}) for _ in range(500)]
    c = dc.sham_success_fn(523, 0.5)
    other = [c({"sustained": True}) for _ in range(500)]
    ctrl = [dc.sustained({"sustained": True}), dc.sustained({"sustained": False})]
    print(f"  (a) row-blind {yes == no}; seed+1 differs {yes != other}; "
          f"CONTROL sustained reads the row {ctrl[0] != ctrl[1]}")
    return ok and yes == no and yes != other and ctrl[0] != ctrl[1]


def part_b() -> bool:
    p = dc.eval_plan(521, 12)
    q = dc.eval_plan(521, 12)
    r = dc.eval_plan(522, 12)
    band = all(dc.EVAL_SEED_LO <= s < dc.EVAL_SEED_HI for _, _, s in p)
    per = {(pt, d): 0 for pt, d, _ in p}
    for pt, d, _ in p:
        per[(pt, d)] += 1
    ok = (len(p) == 480 and band and set(per.values()) == {20}
          and dc.plan_sha(p) == dc.plan_sha(q) and dc.plan_sha(p) != dc.plan_sha(r))
    print(f"  (b) n {len(p)} band {band} per(point,d) {set(per.values())} "
          f"sha 521 {dc.plan_sha(p)} again {dc.plan_sha(q)} 522 {dc.plan_sha(r)}")
    return ok


def part_c() -> bool:
    W = world()
    ents = [(0, 0.75, 40_011), (0, 1.0, 40_012), (1, 0.75, 40_013)]
    built = [(dc.build_at(W, s, d).difficulty, dc.build_at(W, s, d)._seed) for _, d, s in ents]
    want = [(0.75, 40_011), (None, 40_012), (0.75, 40_013)]
    act = lambda o: np.full(8, 0.3)
    r1 = dc.eval_lives(act, W, ents, cap=15)
    r2 = dc.eval_lives(act, W, ents, cap=15)
    print(f"  (c) built {built}; rows {len(r1)} identical {r1 == r2}; "
          f"decisions {[r['decisions'] for r in r1]}")
    return built == want and r1 == r2 and len(r1) == 3


def _train(cb_kind: str):
    import jax
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from experiments.pilot_b import PilotLives, make_b
    W = world()
    venv = VecNormalize(DummyVecEnv([lambda: PilotLives(W(153))]), norm_obs=True,
                        norm_reward=False, clip_obs=10.0)
    venv.seed(153)
    m = make_b(venv, 153, 400)
    m.learning_starts = 100
    cb = None
    if cb_kind == "eval":
        cb = dc.make_eval_callback(venv, W, 521, 400, every=100, per_d=1, cap=10)
    elif cb_kind == "leaky":
        from stable_baselines3.common.callbacks import BaseCallback

        class Leaky(BaseCallback):           # PLANTED: eval through the training env
            def _on_step(self):
                if self.num_timesteps % 100 == 0:
                    for _ in range(10):
                        venv.step_async(np.array([[0.3] * 8])); venv.step_wait()
                return True
        cb = Leaky()
    m.learn(total_timesteps=400, progress_bar=False, callback=cb)
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(m.policy.actor_state.params)]
    return leaves, int(m.replay_buffer.pos), float(venv.obs_rms.count), cb


def part_d() -> bool:
    p0, b0, n0, _ = _train("none")
    p1, b1, n1, cb = _train("eval")
    p2, b2, n2, _ = _train("leaky")
    same = lambda a, b: len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    played = len(cb.rows)
    print(f"  (d) none: buf {b0} obs_n {n0}; eval: buf {b1} obs_n {n1} params same "
          f"{same(p0, p1)} eval lives {played} pts {sorted({r['point'] for r in cb.rows})} sha {cb.sha}")
    print(f"  (d) CONTROL leaky: buf {b2} obs_n {n2} params same {same(p0, p2)}")
    ok = same(p0, p1) and b0 == b1 and n0 == n1 and played == 8
    ctrl_fails = not (same(p0, p2) and b0 == b2 and n0 == n2)
    return ok and ctrl_fails


def main(argv) -> int:
    res = {}
    for k, f in (("a", part_a), ("b", part_b), ("c", part_c), ("d", part_d)):
        res[k] = bool(f())
        print(f"({k}) {'PASS' if res[k] else 'FAIL'}")
    allok = all(res.values())
    print("PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
