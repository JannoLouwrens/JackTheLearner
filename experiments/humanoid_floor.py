"""J10 (1) — the Gymnasium Humanoid-v5 floor (PUBLISHED_PRIORS §4 "Reproduce first").

Two policies that cannot learn, on the UNMODIFIED Gymnasium Humanoid-v5
(gymnasium 1.4.0, mujoco 3.17.0, venv /data/venvs/jack-std, all env defaults:
healthy_reward 5, terminate when torso z leaves (1.0, 2.0), frame_skip 5,
reset noise 1e-2, cap 1,000 steps):

  random — `env.action_space.sample()` each step (action space seeded per episode)
  zero   — all-zero action each step

These are the floor every walking row (J10 (2), (3)) is read against: a
learner that does not beat the zero-action return has learned nothing about
standing up.

RULE (committed before the first run, J10 (1)):
  episodes   100 per policy, `env.reset(seed=s)` for s = 0..99, one env.
  metric     mean undiscounted episode return and mean episode length.
  reference  §4 / PUBLISHED_EVIDENCE_2026-11-15_BUILD.md: random 110 ± 30 in
             23.5 steps; zero 199 ± 3 in 40 steps. HONEST LABEL: that line
             says "(measured here)" — it is an earlier on-box measurement by
             the research desk, not a literature figure. This row therefore
             checks that our install reproduces it, and it is the floor; it
             is not a published-number reproduction and is never cited as one.
  accept     REPRODUCES iff random mean return in [80, 140] AND zero mean
             return in [196, 202] (the reference's ± read as the band on the
             100-episode mean; for random this is ±30 on a mean whose SE is
             ~3, i.e. loose, said here so nobody reads it as tight).
             MISSES otherwise -> find why (version, reset noise, termination
             band) before (2) is run.
  control    none applies: neither policy has parameters to train, so there
             is no never-trains twin. Sanity printed beside it: the zero
             policy must beat random (if not, the reward is not the one §4
             describes).
  compute    ~1 CPU-second; run on the box under nice 19 (the order's "Kaggle
             CPU only" is for (2)'s training kernel; all 4 Kaggle CPU sessions
             were busy with J8 when this ran).

Run:  /data/venvs/jack-std/bin/python -m experiments.humanoid_floor [--out F]
"""
from __future__ import annotations

import json
import sys

import numpy as np

EPISODES = 100
BANDS = {"random": (80.0, 140.0), "zero": (196.0, 202.0)}


def run_policy(policy: str, episodes: int = EPISODES) -> dict:
    import gymnasium as gym
    env = gym.make("Humanoid-v5")
    rets, lens = [], []
    for s in range(episodes):
        obs, _ = env.reset(seed=s)
        env.action_space.seed(s)
        zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        ret, n, done = 0.0, 0, False
        while not done:
            a = env.action_space.sample() if policy == "random" else zero
            obs, r, term, trunc, _ = env.step(a)
            ret += float(r)
            n += 1
            done = term or trunc
        rets.append(ret)
        lens.append(n)
    env.close()
    r = np.asarray(rets)
    return {"policy": policy, "episodes": episodes,
            "return_mean": float(r.mean()), "return_sd": float(r.std(ddof=1)),
            "return_se": float(r.std(ddof=1) / np.sqrt(len(r))),
            "len_mean": float(np.mean(lens)), "len_min": int(min(lens)),
            "len_max": int(max(lens))}


def main(argv) -> int:
    import argparse
    import gymnasium
    import mujoco
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rows = {k: run_policy(k) for k in ("random", "zero")}
    ok = {k: BANDS[k][0] <= rows[k]["return_mean"] <= BANDS[k][1] for k in rows}
    verdict = "REPRODUCES" if all(ok.values()) else "MISSES"
    sanity = rows["zero"]["return_mean"] > rows["random"]["return_mean"]
    res = {"versions": {"gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__},
           "rows": rows, "in_band": ok, "verdict": verdict, "sanity_zero_beats_random": sanity}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
