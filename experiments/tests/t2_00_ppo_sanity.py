"""T2.00 v2 — the RL update must be sane before any locomotion claim.

Written after T2.01 measured training at -4334 versus +170 untrained: PPO was
not merely failing to help, it was destroying a policy that beat random at
initialisation. Three bugs caused it, none visible in a loss curve, and all
three are cheap to guard forever:

  1. VALUE-TERM DOMINATION. value_head sits on the shared trunk and Humanoid
     returns run to the hundreds, so without return normalization the value
     term was ~870x the policy term -- all 57M params optimised to regress
     returns; the policy was dragged, never trained.
  2. UNBOUNDED log_std. Nothing clamped it and the entropy bonus inflates it
     without limit.
  3. ACTIONS NEVER CLIPPED to the env range. |action| hit 1.20 then 2.37 against
     Humanoid's +-0.4 in two iterations; MuJoCo clips ctrl internally, so PPO
     assigned credit to components that never touched the physics.

v1 -> v2, and why the FAIL that forced it stays in the ledger's history: v1
gated domination on the vf/pg LOSS ratio. That metric has a hidden dependence
on minibatch geometry -- pg_loss at an unmoved policy is ~0 BY CONSTRUCTION
(normalized advantages, ratio=1), so its magnitude tracks how far the policy
drifts within the update, and ppo_minibatch 64->512 halved the gradient steps,
halved the drift, and tripped the gate at 178.57 with nothing wrong. A probe of
the true quantity -- the gradient norm each term contributes to the shared
trunk -- read a healthy 1.9-2.8x at every minibatch size. v2 gates on that
gradient-norm ratio, measured inside rl_update itself (term_grad_diag=True) on
the first real minibatch of every iteration, so the guard reads the production
code path, not a reimplementation. The loss ratio is still recorded as a
diagnostic; it no longer gates.

This spec runs a handful of real PPO iterations and asserts the invariants
directly. It is CPU-cheap by design: it must gate every GPU locomotion run,
so it cannot cost GPU time itself.

CONTROL: with normalize_returns disabled, the GRAD ratio MUST blow past the
threshold (returns in the hundreds inflate the value gradient by the same
factor). A guard that passes in both configurations is measuring nothing.
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
# Pre-registered for v2. Healthy balance measured 1.9-2.8x across minibatch
# sizes 64/128/512; the un-normalized pathology inflates the value gradient by
# the return scale (~hundreds). 25 sits an order of magnitude above healthy
# and an order below the pathology.
MAX_VF_PG_GRAD_RATIO = 25.0
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

    grad_ratios, loss_ratios, stds, env_absmax = [], [], [], []
    lo = envs.single_action_space.low
    hi = envs.single_action_space.high
    try:
        for _ in range(ITERS):
            buf = tp.collect_rollout_vec(envs, n_steps=ROLLOUT)
            stats = tp.rl_update(buf, term_grad_diag=True)
            grad_ratios.append(stats["grad_vf_norm"]
                               / max(stats["grad_pg_norm"], 1e-9))
            pg = abs(float(stats.get("pg_loss", 0.0)))
            vf = abs(float(stats.get("vf_loss", 0.0)))
            loss_ratios.append(vf / max(pg, 1e-9))
            stds.append(float(tp.log_std.max()))
            # What the ENV would receive: the buffer deliberately stores the
            # unclipped Gaussian sample (its density is over unclipped actions),
            # so clip here exactly as collect_rollout_vec does before stepping.
            a = buf["actions"].detach().cpu().numpy()
            env_absmax.append(float(np.abs(np.clip(a, lo, hi)).max()))
    finally:
        envs.close()

    return {
        "max_vf_pg_grad_ratio": round(max(grad_ratios), 2),
        "final_vf_pg_grad_ratio": round(grad_ratios[-1], 2),
        "max_vf_pg_loss_ratio": round(max(loss_ratios), 2),   # diagnostic only
        "max_log_std": round(max(stds), 4),
        "min_log_std": round(min(stds), 4),
        "env_action_absmax": round(max(env_absmax), 4),
        "action_limit": round(float(abs(hi).max()), 4),
    }


def _experiment(seed: int) -> dict:
    return _run(seed, normalize=True)


def _control(seed: int) -> dict:
    """Without return normalization the value gradient must dominate."""
    r = _run(seed, normalize=False)
    return {"unnormalized_grad_ratio": r["max_vf_pg_grad_ratio"]}


def _check(m: dict, c: dict) -> bool:
    balanced = m["max_vf_pg_grad_ratio"] <= MAX_VF_PG_GRAD_RATIO
    bounded = MIN_LOG_STD <= m["min_log_std"] and m["max_log_std"] <= MAX_LOG_STD
    in_range = m["env_action_absmax"] <= m["action_limit"] + 1e-6
    control_breaks = c["unnormalized_grad_ratio"] > MAX_VF_PG_GRAD_RATIO
    return balanced and bounded and in_range and control_breaks


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.00"], _experiment, _check, control_fn=_control, ledger=ledger)
