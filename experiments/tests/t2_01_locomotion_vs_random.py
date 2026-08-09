"""T2.01 — the trained policy must beat random actions by more than luck.

This is the first spec in the ladder whose passing run produces a Jack worth
keeping: a policy that moves the humanoid better than noise does. Everything
before it proved machinery; this is the first capability.

The bar is deliberately modest — beat RANDOM, not walk well — because its job is
to prove the training loop trains, end to end, on the real environment: batched
rollout (collect_rollout_vec) -> vectorised GAE -> PPO update -> a return that
climbs. T2.02 raises the bar to the honest 140K-param MLP baseline afterwards.

Statistics, pre-registered:
  - 3 seeds trained independently; evaluation is deterministic (mean action).
  - sigma = max(std of trained means across seeds, std of random episode
    returns) — whichever noise source is larger bounds the claim.
  - PASS needs (trained_mean - random_mean) >= 5 * sigma, per the spec, AND
    every individual seed beating the random mean (no seed carried by another).
  - CONTROL: the untrained network, same architecture, must NOT clear the same
    bar. If it does, the bar measures architecture bias, not learning.

Wall-clock is bounded per seed rather than step-bounded: the budget is gpu<2h
and a fixed step count at unknown VM throughput would either waste quota or
overrun it. The actual env-steps completed are recorded in the metrics, so the
result is attributable regardless — which is how v2's 105K steps/seed became
visible as a second, independent problem alongside the runaway policy.

T1.08's noise-floor discipline applies: the seed spread is IN the check, not a
footnote.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID

SEEDS = [0, 1, 2]
# v2 completed only 105,472 env-steps per seed (~80 steps/s). MuJoCo is not the
# bottleneck -- 1024 env-steps of Humanoid costs it ~0.5s against ~13s measured
# per iteration -- the PPO update was: 16 minibatches x 5 epochs of batch 64,
# ~12s of the 13. Two changes follow: ppo_minibatch=512 in the pipeline (same
# total sample-passes, GPU actually utilised) and N_ENVS 8->32 (the rollout
# forward is batched over envs). COMPUTE BUDGET changes, not threshold changes:
# the 5-sigma bar, the control, and the all-seeds rule below are untouched.
N_ENVS = 32
ROLLOUT_STEPS = 128          # per env per iteration -> 4096-sample PPO batches
# v4: 30 -> 110. v3 was the first HEALTHY run (every seed beat random, curve
# still climbing at cutoff) and failed only the effect size: 2.21 sigma at
# 192K steps/seed against the 5-sigma bar. The pre-registered branch for a
# CLIMBING curve is more compute, so the spec moves to the gpu<8h budget
# class. ~850K steps/seed at the measured ~128 env-steps/s. The 5-sigma bar,
# the control, and the all-seeds rule are untouched -- if the curve rises and
# the bar still fails at 850K steps, that is a real sample-efficiency verdict
# on the architecture, and the D1 trunk question comes next.
TRAIN_MINUTES_PER_SEED = 110
EVAL_EPISODES = 5
RANDOM_EPISODES = 10
MIN_SIGMA_ADVANTAGE = 5.0

JOB = r'''
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "gymnasium[mujoco]"],
        check=True)

import json, time, numpy as np, torch
from TrainingPipeline import TrainingPipeline, PipelineConfig

def eval_policy(tp, episodes, random_actions=False):
    """Deterministic evaluation: mean action, no exploration noise."""
    env = tp.make_env()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            if random_actions:
                act = env.action_space.sample()
            else:
                # Same bounded mean the training rollout uses, then clipped like
                # every other action reaching an environment. The v2 run
                # evaluated the RAW head output, so it fed actions of magnitude
                # 40+ into env.step and measured a bang-bang policy nobody had
                # trained.
                #
                # tp.act_deterministic, NOT a local re-implementation of the
                # forward. This function used to call tp.model(...) directly and
                # never set eval mode, so both the untrained control (fresh
                # nn.Module defaults to training=True) and the trained arm
                # (rl_update leaves train mode on) evaluated with 36 dropout
                # modules live -- 103.6% drift between two forwards of one
                # identical state, measured on this net. T0.14 fixed the
                # pipeline's internals and could not reach this file. T0.16
                # guards the composition.
                act = tp.act_deterministic(obs)
            act = np.clip(act, env.action_space.low, env.action_space.high)
            obs, r, term, trunc, _ = env.step(act)
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return returns

def train_one(seed, minutes):
    torch.manual_seed(seed); np.random.seed(seed)
    tp = TrainingPipeline(PipelineConfig())
    tp.make_optimizer(phase=3)

    untrained = eval_policy(tp, __EVAL_EPS__)          # control: same arch, no training

    envs = tp.make_vec_envs(__N_ENVS__)
    deadline = time.time() + minutes * 60
    iters = steps = 0
    curve = []
    while time.time() < deadline:
        buf = tp.collect_rollout_vec(envs, n_steps=__ROLLOUT__)
        stats = tp.rl_update(buf)
        iters += 1
        steps += __ROLLOUT__ * __N_ENVS__
        # WITHOUT THIS the run reports only endpoints, and a catastrophic
        # failure (T2.01 v1: trained -4334 vs untrained +170) cannot be located
        # in time. Track what actually diagnoses PPO collapse: rollout reward,
        # the learned action std (entropy bonus can inflate it without bound),
        # and action magnitude versus the env's own action range.
        if iters % 5 == 1 or iters < 5:
            curve.append({
                "iter": iters, "steps": steps,
                "mean_reward": float(buf["rewards"].mean()),
                "action_std": float(tp.log_std.exp().mean()),
                "action_absmax": float(buf["actions"].abs().max()),
                "value_mean": float(buf["values"].mean()),
                **{k: round(float(v), 4) for k, v in list(stats.items())[:3]},
            })
    envs.close()

    trained = eval_policy(tp, __EVAL_EPS__)
    return {"seed": seed, "iters": iters, "env_steps": steps, "curve": curve,
            "untrained_returns": untrained, "trained_returns": trained,
            "trained_mean": float(np.mean(trained)),
            "untrained_mean": float(np.mean(untrained))}

t0 = time.time()
tp0 = TrainingPipeline(PipelineConfig())
random_returns = eval_policy(tp0, __RANDOM_EPS__, random_actions=True)
del tp0

out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "random_returns": random_returns,
       "seeds": [train_one(s, __MINUTES__) for s in __SEEDS__],
       "wall_minutes": round((time.time() - t0) / 60, 1)}
import os as _o
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t201.json"), "w"), indent=1)
print("DONE", json.dumps({k: out[k] for k in ("gpu", "wall_minutes")}), flush=True)
'''


def _submit() -> dict:
    body = (JOB.replace("__SEEDS__", repr(SEEDS))
               .replace("__N_ENVS__", repr(N_ENVS))
               .replace("__ROLLOUT__", repr(ROLLOUT_STEPS))
               .replace("__MINUTES__", repr(TRAIN_MINUTES_PER_SEED))
               .replace("__EVAL_EPS__", repr(EVAL_EPISODES))
               .replace("__RANDOM_EPS__", repr(RANDOM_EPISODES)))
    job = build_job(body)
    # timeout_s stays below the runner's own gpu<8h child timeout (36000s), or
    # the parent kills a job that was about to hand back finished science.
    res = submit(job, prefer="kaggle", est_hours=6.5, timeout_s=32000,
                 fetch=["t201.json"])
    if not res.ok:
        raise RuntimeError(f"GPU job failed on {res.backend}: {res.message}")
    path = res.artifacts.get("t201.json")
    if not path:
        raise RuntimeError(f"no artifact from {res.backend}. message={res.message!r} "
                           f"stdout_tail={res.stdout[-400:]!r}")
    d = json.loads(Path(path).read_text())
    d["backend"] = res.backend
    return d


_CACHE: dict = {}


def _stats(vals):
    n = len(vals)
    m = sum(vals) / n
    return m, (sum((v - m) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5


def _experiment(seed: int) -> dict:
    # ONE submission for the whole spec. run_spec calls _experiment once per
    # declared seed (3), but the JOB trains all three seeds internally — an
    # unguarded _submit() here launched a second identical 5.5h kernel the
    # moment the first one's artifact landed (2026-08-07, ~11 GPU-hours of
    # redundant compute; v3 only escaped because JACK_REUSE_KERNEL happened to
    # pin all three calls to one kernel).
    if not _CACHE:
        _CACHE.update(_submit())
    rnd_mean, rnd_std = _stats(_CACHE["random_returns"])
    trained_means = [s["trained_mean"] for s in _CACHE["seeds"]]
    tr_mean, tr_std = _stats(trained_means)
    sigma = max(tr_std, rnd_std, 1e-6)
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "wall_minutes": _CACHE["wall_minutes"],
        "env_steps_per_seed": [s["env_steps"] for s in _CACHE["seeds"]],
        "curve_seed0": _CACHE["seeds"][0].get("curve", [])[:8],
        "random_mean": round(rnd_mean, 1), "random_std": round(rnd_std, 2),
        "trained_means": [round(m, 1) for m in trained_means],
        "trained_mean": round(tr_mean, 1), "trained_std": round(tr_std, 2),
        "sigma_used": round(sigma, 2),
        "sigma_advantage": round((tr_mean - rnd_mean) / sigma, 2),
        "all_seeds_beat_random": all(m > rnd_mean for m in trained_means),
    }


def _control(seed: int) -> dict:
    """Untrained same-architecture network must NOT clear the bar."""
    rnd_mean, rnd_std = _stats(_CACHE["random_returns"])
    unt_means = [s["untrained_mean"] for s in _CACHE["seeds"]]
    u_mean, u_std = _stats(unt_means)
    sigma = max(u_std, rnd_std, 1e-6)
    return {"untrained_mean": round(u_mean, 1),
            "untrained_sigma_advantage": round((u_mean - rnd_mean) / sigma, 2)}


def _check(m: dict, c: dict) -> bool:
    return (m["sigma_advantage"] >= MIN_SIGMA_ADVANTAGE
            and m["all_seeds_beat_random"]
            and c["untrained_sigma_advantage"] < MIN_SIGMA_ADVANTAGE)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.01"], _experiment, _check, control_fn=_control, ledger=ledger)
