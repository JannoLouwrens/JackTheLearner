"""Shared by test_jack_env_paint: the geom-paint streams whose sha256 is
pinned there (taken from HEAD b06d830b's code BEFORE the paint option was
written, so "paint='geom' is today's world" is checked against yesterday's
code, not against itself)."""
import hashlib

import numpy as np

PILOT = dict(difficulty=0.75, reward="dr100-eat", random_layout=True,
             lazy_contact_audio=True, food_floor_only=True, eye_frame="body",
             motor_frame="body", native_substep=True)
BASE = {"j0": 1.0, "alpha": 1.0, "e0": 0.05, "max_episode_decisions": 40}
GEOM_CASES = (("rays16-base-153", 153, dict(BASE)),
              ("row128-pilot-941", 941, dict(PILOT, eye="row128")),
              ("row128-pilot-942", 942, dict(PILOT, eye="row128")),
              ("row128-pilot-943", 943, dict(PILOT, eye="row128")))


def pilot_kw():
    from experiments.jack_env import NEEDS_S_TRP, SPAWN_CLEAR_R
    return dict(PILOT, random_needs=NEEDS_S_TRP, spawn_clear=SPAWN_CLEAR_R)


def stream_sha(env, seed: int, n: int, act_seed: int) -> str:
    from experiments.w0 import random_action
    rng = np.random.RandomState(act_seed)
    h = hashlib.sha256()
    o, _ = env.reset(seed=seed)
    h.update(o.tobytes())
    for _ in range(n):
        o, r, te, tr, _ = env.step(random_action(rng))
        h.update(o.tobytes())
        h.update(np.float64(r).tobytes())
        h.update(bytes([te, tr]))
        if te or tr:
            o, _ = env.reset()
            h.update(o.tobytes())
    return h.hexdigest()


def geom_stream_sha(n: int, **extra) -> dict:
    from experiments.jack_env import JackEnv
    out = {}
    for name, seed, kw in GEOM_CASES:
        if "difficulty" in kw:
            kw = dict(pilot_kw(), eye=kw["eye"])
        out[name] = stream_sha(JackEnv(seed=seed, **kw, **extra), seed, n, seed + 1000)
    return out
