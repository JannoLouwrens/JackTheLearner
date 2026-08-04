"""T0.07 — measure real throughput before believing any compute estimate.

Not a pass/fail claim about quality; a measurement that every later plan depends
on. The pipeline review measured ~27 env-steps/s and traced it to 6,095 ATen
dispatches per B=1 forward — the rollout loop is CPU-dispatch-bound, not FLOP-
bound. That number decides whether a T4 helps at all: if the bottleneck is Python
dispatch on one environment, a GPU sits idle waiting and the honest answer is to
vectorise before renting anything.

So this records three figures and their ratios:
  bare      — MuJoCo physics alone, no policy. The ceiling.
  policy    — physics plus a full brain forward. What training actually costs.
  vectorised— N environments stepped together, amortising Python overhead.

The pass condition is only that the measurement is sound (stable across repeats
and the environment is genuinely stepping); the numbers themselves are recorded
for planning. A test that "fails" because a box is slow would be measuring the
box, not the code.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
STEPS = 300
N_VEC = 8


def _setup():
    os.environ.setdefault("MUJOCO_GL", "disabled")
    sys.path.insert(0, str(REPO))


def _bare(steps: int = STEPS) -> float:
    import gymnasium as gym
    e = gym.make("Humanoid-v5"); e.reset(seed=0)
    a = e.action_space.sample()
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, term, trunc, _ = e.step(a)
        if term or trunc:
            e.reset()
    return steps / (time.perf_counter() - t0)


def _vectorised(steps: int = STEPS // 4, n: int = N_VEC) -> float:
    import gymnasium as gym
    v = gym.make_vec("Humanoid-v5", num_envs=n, vectorization_mode="sync")
    v.reset(seed=0)
    a = v.action_space.sample()
    t0 = time.perf_counter()
    for _ in range(steps):
        v.step(a)
    return (steps * n) / (time.perf_counter() - t0)


def _with_policy(steps: int = 60) -> tuple[float, int]:
    import gymnasium as gym, numpy as np, torch
    from UnifiedBrain import UnifiedBrain, UnifiedBrainConfig
    from VirtualWorld import apply_action

    cfg = UnifiedBrainConfig()
    for flag in ("llm_enabled", "enable_intrinsic_motivation"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    brain = UnifiedBrain(cfg).eval()
    e = gym.make("Humanoid-v5"); u = e.unwrapped
    e.reset(seed=0)
    obs = torch.zeros(1, cfg.obs_dim)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            act = brain(obs)["actions"]
            # Chunked head: take the first step of the chunk, width-checked.
            a = act.reshape(-1)[: u.model.nu].double().numpy()
            apply_action(u.data, u.model, a)
            _, _, term, trunc, _ = e.step(np.zeros(u.model.nu))
            if term or trunc:
                e.reset()
    return steps / (time.perf_counter() - t0), int(cfg.obs_dim)


def _experiment(seed: int) -> dict:
    _setup()
    bare_a = _bare()
    bare_b = _bare()          # repeat: a measurement that will not repeat is not one
    vec = _vectorised()
    pol, obs_dim = _with_policy()

    spread = abs(bare_a - bare_b) / max(bare_a, bare_b)
    return {
        "bare_steps_per_s": round((bare_a + bare_b) / 2, 1),
        "bare_repeat_spread": round(spread, 4),
        "vectorised_steps_per_s": round(vec, 1),
        "vectorised_speedup": round(vec / ((bare_a + bare_b) / 2), 2),
        "policy_steps_per_s": round(pol, 2),
        "policy_slowdown_vs_bare": round(((bare_a + bare_b) / 2) / max(pol, 1e-9), 1),
        "hours_for_2M_steps_policy": round(2_000_000 / max(pol, 1e-9) / 3600, 1),
        "obs_dim": obs_dim,
    }


def _check(m: dict, _c: dict) -> bool:
    # Sound measurement, not a performance bar: repeats agree within 25% and the
    # sim genuinely ran.
    return m["bare_repeat_spread"] < 0.25 and m["bare_steps_per_s"] > 1 and m["policy_steps_per_s"] > 0


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.07"], _experiment, _check, ledger=ledger)
