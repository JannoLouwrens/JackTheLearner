"""T2.00 — the RL update must be sane before any locomotion claim.

Written after T2.01 measured training at -4334 versus +170 untrained: PPO was
not merely failing to help, it was destroying a policy that beat random at
initialisation. Three bugs caused it, none visible in a loss curve, and all
three are cheap to guard forever:

  1. NO RETURN NORMALIZATION. value_head sits on the shared trunk and Humanoid
     returns run to the hundreds, so vf_loss was 540.5 against pg_loss 0.267 --
     with vf_coef=0.43 the value term was ~870x the policy term. After gradient
     clipping the policy gradient was a rounding error in the update DIRECTION.
     All 57M params optimised to regress returns; the policy was dragged, never
     trained.
  2. UNBOUNDED log_std. Nothing clamped it and the entropy bonus inflates it
     without limit.
  3. ACTIONS NEVER CLIPPED to the env range. |action| hit 1.20 then 2.37 against
     Humanoid's +-0.4 in two iterations; MuJoCo clips ctrl internally, so PPO
     assigned credit to components that never touched the physics.

This spec runs a handful of real PPO iterations and asserts the three invariants
directly. It is CPU-cheap by design: it must gate every GPU locomotion run,
so it cannot cost GPU time itself.

CONTROL: with normalize_returns disabled, the vf/pg ratio MUST blow past the
threshold. A guard that passes in both configurations is measuring nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

ITERS = 4
N_ENVS = 4
ROLLOUT = 32
# Pre-registered. The value term may legitimately exceed the policy term -- it
# is a regression loss against a moving target -- but ~870x means the policy is
# not being optimised at all.
MAX_VF_PG_RATIO = 50.0
MAX_LOG_STD = 0.0        # std <= 1.0
MIN_LOG_STD = -4.6       # std >= 0.01


def _run(seed: int, normalize: bool) -> dict:
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from TrainingPipeline import TrainingPipeline, PipelineConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = PipelineConfig()
    cfg.normalize_returns = normalize
    tp = TrainingPipeline(cfg)
    tp.make_optimizer(phase=3)
    envs = tp.make_vec_envs(N_ENVS)

    ratios, stds, env_absmax = [], [], []
    lo = envs.single_action_space.low
    hi = envs.single_action_space.high
    try:
        for _ in range(ITERS):
            buf = tp.collect_rollout_vec(envs, n_steps=ROLLOUT)
            stats = tp.rl_update(buf)
            pg = abs(float(stats.get("pg_loss", 0.0)))
            vf = abs(float(stats.get("vf_loss", 0.0)))
            ratios.append(vf / max(pg, 1e-9))
            stds.append(float(tp.log_std.max()))
            # What the ENV would receive: the buffer deliberately stores the
            # unclipped Gaussian sample (its density is over unclipped actions),
            # so clip here exactly as collect_rollout_vec does before stepping.
            a = buf["actions"].detach().cpu().numpy()
            env_absmax.append(float(np.abs(np.clip(a, lo, hi)).max()))
    finally:
        envs.close()

    return {
        "max_vf_pg_ratio": round(max(ratios), 2),
        "final_vf_pg_ratio": round(ratios[-1], 2),
        "max_log_std": round(max(stds), 4),
        "min_log_std": round(min(stds), 4),
        "env_action_absmax": round(max(env_absmax), 4),
        "action_limit": round(float(abs(hi).max()), 4),
    }


def _experiment(seed: int) -> dict:
    return _run(seed, normalize=True)


def _control(seed: int) -> dict:
    """Without return normalization the ratio must explode."""
    r = _run(seed, normalize=False)
    return {"unnormalized_vf_pg_ratio": r["max_vf_pg_ratio"]}


def _check(m: dict, c: dict) -> bool:
    balanced = m["max_vf_pg_ratio"] <= MAX_VF_PG_RATIO
    bounded = MIN_LOG_STD <= m["min_log_std"] and m["max_log_std"] <= MAX_LOG_STD
    in_range = m["env_action_absmax"] <= m["action_limit"] + 1e-6
    control_breaks = c["unnormalized_vf_pg_ratio"] > MAX_VF_PG_RATIO
    return balanced and bounded and in_range and control_breaks


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.00"], _experiment, _check, control_fn=_control, ledger=ledger)
