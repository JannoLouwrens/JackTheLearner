"""T0.14 — evaluation must actually be deterministic, and the obs contract must hold.

Written after the most expensive bug in the project. TrainingPipeline never
called .eval() or .train(), so 36 nn.Dropout modules at p=0.1 were live during
rollout, during the PPO update, and during what the code calls "deterministic"
evaluation. Measured on the real pipeline: two forwards of the SAME state
differed by 42% of the policy mean's own magnitude, 66% for the value, and the
PPO importance ratio at ZERO policy change put ~20% of samples outside
clip_range=0.3 — the update was clipping against its own sampling noise.

It survived four T2.01 runs, a T2.02 architecture comparison, and an
owner-facing recommendation, because the baseline it was compared against
(SB3) disables training mode automatically and has no dropout. One arm ran with
42% injected action noise and the other with none, and the gap was attributed
to architecture.

The class of bug is "a silent default the library does for you and your code
does not", and it is invisible by inspection — there is no wrong line to read.
So this spec asserts the OBSERVABLE property instead of the implementation:
forward the same input twice and demand bit-identity.

Checks:
  1. After collect_rollout_vec, the model is in eval mode.
  2. After rl_update, it is back in train mode (dropout belongs in the
     gradient path and nowhere else).
  3. In eval mode, two forwards of one state are BIT-IDENTICAL — policy mean
     and value both.
  4. config.mujoco_obs_dim equals what the environment actually emits.
     Humanoid-v5 emits 348; the config said 376, the v4 value, so 28 zeros
     were padded into every observation.

CONTROL: with the model forced into train mode, check 3 MUST fail. A
determinism test that passes in both modes is asserting nothing — which is
precisely how this went unnoticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

MAX_EVAL_DRIFT = 0.0        # bit-identity, not "small"


def _forward_twice(tp, train_mode: bool):
    """Two forwards of one state. Probe dimension comes from the RUNNING
    normaliser, not from config: the two disagree (that is check 4), and a
    probe sized from config crashes here instead of failing check 4 cleanly.
    Each check must be able to fail on its own terms.
    """
    import numpy as np
    import torch

    tp.model.train(train_mode)
    dim = getattr(getattr(tp, "obs_mean", None), "shape", [tp.config.mujoco_obs_dim])[0]
    obs = np.zeros(int(dim), dtype=np.float32)
    ot = torch.tensor(tp.normalize_obs(obs), dtype=torch.float32,
                      device=tp.device).unsqueeze(0)
    with torch.no_grad():
        o1 = tp.model(tp.project_obs(ot))
        o2 = tp.model(tp.project_obs(ot))
        a1, a2 = tp.policy_mean(o1), tp.policy_mean(o2)
        v1, v2 = o1["value"].reshape(-1), o2["value"].reshape(-1)
    scale = max(float(a1.abs().mean()), 1e-9)
    return (float((a1 - a2).abs().max()) / scale,
            float((v1 - v2).abs().max()) / max(float(v1.abs().mean()), 1e-9))


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    import gymnasium as gym
    import numpy as np
    import torch
    from TrainingPipeline import TrainingPipeline, PipelineConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    tp = TrainingPipeline(PipelineConfig())
    tp.make_optimizer(phase=3)

    envs = tp.make_vec_envs(2)
    try:
        buf = tp.collect_rollout_vec(envs, n_steps=8)
        eval_mode_after_rollout = not tp.model.training
        tp.rl_update(buf)
        train_mode_after_update = tp.model.training
    finally:
        envs.close()

    act_drift, val_drift = _forward_twice(tp, train_mode=False)

    env = gym.make("Humanoid-v5")
    env_obs_dim = int(env.observation_space.shape[0])
    env.close()

    return {
        "n_dropout_modules": sum(1 for m in tp.model.modules()
                                 if isinstance(m, torch.nn.Dropout)),
        "eval_mode_after_rollout": eval_mode_after_rollout,
        "train_mode_after_update": train_mode_after_update,
        "eval_action_drift": round(act_drift, 9),
        "eval_value_drift": round(val_drift, 9),
        "config_obs_dim": int(tp.config.mujoco_obs_dim),
        "env_obs_dim": env_obs_dim,
        "obs_dim_matches": tp.config.mujoco_obs_dim == env_obs_dim,
    }


def _control(seed: int) -> dict:
    """Forced into TRAIN mode, the determinism check must fail.

    Without this the spec would pass on a model that has no dropout at all,
    or on one where .eval() silently does nothing — asserting a property that
    cannot be violated is the failure mode this whole spec exists to catch.
    """
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from TrainingPipeline import TrainingPipeline, PipelineConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    tp = TrainingPipeline(PipelineConfig())
    act_drift, _ = _forward_twice(tp, train_mode=True)
    return {"train_mode_action_drift": round(act_drift, 6)}


def _check(m: dict, c: dict) -> bool:
    return (m["eval_mode_after_rollout"]
            and m["train_mode_after_update"]
            and m["eval_action_drift"] <= MAX_EVAL_DRIFT
            and m["eval_value_drift"] <= MAX_EVAL_DRIFT
            and m["obs_dim_matches"]
            # the control MUST be non-deterministic, or nothing is being tested
            and c["train_mode_action_drift"] > 1e-3)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.14"], _experiment, _check, control_fn=_control, ledger=ledger)
