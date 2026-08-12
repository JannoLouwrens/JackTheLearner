"""T0.25 — a value head is a baseline only if subtracting it removes the advantage.

The claim is checkable in closed form, which is why this is a Tier-0 harness
spec and not an argument. Build a reward sequence, compute its EXACT value
function under the same boundary and done conventions the GAE recursion uses,
hand that value function to the critic slot, and ask what is left:

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)  ==  0   for all t
    => advantages == 0

Any advantage estimator that does not produce zeros here is not subtracting a
baseline; it is subtracting something else, and the policy gradient it feeds
PPO is a discounted reward sum with an offset — REINFORCE, wearing an
actor-critic's clothes.

WHY THIS RUNS AT TWO NORMALISER STATES. The pipeline trains the critic on
returns that have been divided by a running return-std (`normalize_returns`),
so the critic emits V/scale. GAE's delta adds RAW rewards to that. At a FRESH
pipeline the running scale is exactly 1.0 and the two unit systems agree, so a
perfect critic cancels and the defect is invisible. It only appears once the
normaliser has seen data. A single-regime version of this test would have
passed on a broken estimator, which is the whole reason the metric is a MAX
over regimes rather than a number from a fresh instance.

METRIC, pre-registered: residual_ratio = std(adv | perfect V) / std(adv | V=0).
The denominator is the null — no critic at all, advantages equal to the raw
lambda-discounted reward sums. A correct estimator scores 0.0; the null scores
1.0 by construction. PASS needs max over regimes < 0.02 (float-noise slack).

CONTROL: the pre-fix recursion, kept executable — the critic's output used
verbatim against raw rewards. On this same fixture it must NOT cancel. If the
control ever cancels, the fixture stopped reproducing the defect and this spec
guards nothing.
"""
from __future__ import annotations

import torch

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

T_STEPS = 64
N_ENVS = 4
ALIVE_REWARD = 5.0        # Humanoid-v5's healthy_reward, the regime that broke
DONE_RATE = 0.02
MAX_RESIDUAL = 0.02


def _fixture(seed: int):
    """Humanoid-shaped rewards with sparse episode boundaries."""
    g = torch.Generator().manual_seed(1000 + seed)
    rewards = ALIVE_REWARD + 0.1 * torch.randn(T_STEPS, N_ENVS, generator=g)
    dones = (torch.rand(T_STEPS, N_ENVS, generator=g) < DONE_RATE).float()
    dones[-1] = 0.0            # the last row bootstraps from 0 either way
    return rewards, dones


def _true_value(rewards: torch.Tensor, dones: torch.Tensor, gamma: float):
    """The EXACT value function of this trajectory under GAE's own conventions.

    V(s_t) = r_t + gamma * (1 - d_t) * V(s_{t+1}),  V(s_T) = 0.
    Derived from the recursion in `compute_gae`, not from a textbook, so a
    disagreement about conventions cannot be mistaken for a defect.
    """
    V = torch.zeros_like(rewards)
    run = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        run = rewards[t] + gamma * (1 - dones[t]) * run
        V[t] = run
    return V


def _prefix_gae(tp, rewards, dones, old_values):
    """THE PRE-FIX RECURSION, verbatim (TrainingPipeline.py @ 2026-08-11).

    The critic's output enters delta in whatever units the critic emits, while
    the rewards stay raw. Kept executable as this spec's control: a guard whose
    control is a paraphrase guards a paraphrase.
    """
    gamma = tp.config.gamma
    gae_lambda = tp.config.gae_lambda
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        next_value = old_values[t + 1] if t < T - 1 else torch.zeros_like(rewards[0])
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - old_values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    returns = advantages + old_values
    if getattr(tp.config, "normalize_returns", True):
        batch_var = returns.detach().var()
        n = returns.numel()
        tp.ret_count += n
        w = n / tp.ret_count
        tp.ret_var = (1 - w) * tp.ret_var + w * batch_var
        scale = torch.sqrt(tp.ret_var + 1e-8).clamp(min=1e-3)
        returns = returns / scale
        advantages = advantages / scale
    return advantages


def _pipeline():
    from TrainingPipeline import TrainingPipeline, PipelineConfig
    return TrainingPipeline(PipelineConfig())


def _scale_of(tp) -> torch.Tensor:
    """The normaliser the critic's targets were last divided by."""
    if not getattr(tp.config, "normalize_returns", True):
        return torch.ones(())
    return torch.sqrt(tp.ret_var + 1e-8).clamp(min=1e-3)


def _residuals(seed: int, gae_fn) -> dict:
    """residual_ratio at a fresh normaliser and at a warmed one.

    `gae_fn(tp, rewards, dones, values) -> advantages`. The normaliser state is
    snapshotted and restored around every call, so the perfect-critic arm and
    the V=0 null arm are measured against the SAME scale — otherwise the ratio
    would be reporting normaliser drift, not cancellation.
    """
    tp = _pipeline()
    rewards, dones = _fixture(seed)
    V = _true_value(rewards, dones, tp.config.gamma)
    out = {}

    for regime in ("fresh", "warmed"):
        if regime == "warmed":
            # One ordinary update's worth of statistics: an untrained critic
            # (V=0) on this same rollout. This is exactly how the running scale
            # leaves 1.0 in a real run.
            gae_fn(tp, rewards, dones, torch.zeros_like(rewards))

        saved = (tp.ret_var.clone(), tp.ret_count)
        scale_in = _scale_of(tp)
        # What a critic trained to convergence on normalised returns emits.
        adv_perfect = gae_fn(tp, rewards, dones, V / scale_in)
        tp.ret_var, tp.ret_count = saved[0].clone(), saved[1]
        adv_null = gae_fn(tp, rewards, dones, torch.zeros_like(rewards))
        tp.ret_var, tp.ret_count = saved[0].clone(), saved[1]

        out[f"{regime}_scale"] = round(float(scale_in), 4)
        out[f"{regime}_null_adv_std"] = round(float(adv_null.std()), 4)
        out[f"{regime}_residual_ratio"] = round(
            float(adv_perfect.std() / adv_null.std().clamp(min=1e-8)), 5)

    out["max_residual_ratio"] = max(out["fresh_residual_ratio"],
                                    out["warmed_residual_ratio"])
    return out


def _experiment(seed: int) -> dict:
    def production(tp, rewards, dones, values):
        adv, _returns, _v = tp.compute_gae(rewards, dones, values)
        return adv
    return _residuals(seed, production)


def _control(seed: int) -> dict:
    return _residuals(seed, _prefix_gae)


def _check(m: dict, c: dict) -> bool:
    return (m["max_residual_ratio"] < MAX_RESIDUAL
            and c["max_residual_ratio"] >= MAX_RESIDUAL)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.25"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
