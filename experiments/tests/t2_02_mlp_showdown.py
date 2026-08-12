"""T2.02 — the chosen architecture vs the honest MLP baseline, matched steps.

This is the arbitration run that T2.01 v4 made necessary. The facts going in,
so the next reader does not need the journal:

  - CORRECTED 2026-08-12. This paragraph used to cite "4.06 sigma, trained
    means [249.6, 292.7, 240.8] -> 261.0" as the fact going in. Those numbers
    come from the run T0.14 INVALIDATED (36 dropout modules live during eval,
    obs padded 376 vs the env's 348); the ledger's standing instruction is
    never to cite them as architecture evidence, and this file was still doing
    it. The valid measurement is T2.01 v4, Kaggle P100, 2026-08-10, 692,224
    env-steps/seed: FAILED at 1.19 sigma vs random (bar 5), trained means
    [231.9, 384.5, 155.3] -> 257.2, and the curve had PLATEAUED by ~300K steps
    at mean_reward ~5.1 — Humanoid-v5's healthy_reward of 5.0 and little else.
    Seed 2's trained policy (155.3) scored BELOW its own untrained control
    (186.0).
  - AND THE PREMISE HAS MOVED. T0.25 (2026-08-12) found why: GAE subtracted a
    baseline that was ~28x too small, because the critic is trained on
    normalised returns and emits V/scale while delta added raw rewards. PPO was
    running as REINFORCE with a batch-mean baseline. Fixed in 08444b2. So the
    transformer arm's 1.19 sigma is a measurement of a broken estimator, not of
    the architecture, and THIS SPEC MUST NOT BE RUN until T2.01 has re-run
    post-fix: its validity gate needs BOTH arms to clear 3 sigma vs random, and
    a pre-fix transformer arm at 1.19 sigma would VOID the run and spend ~7
    Kaggle-hours to arbitrate nothing. Re-read the numbers above from the
    ledger before submitting.
  - The local CPU probe (/tmp/mlp_probe.json, journaled 2026-08-07): a
    54K-param SB3 PPO MLP at the SAME 704,512 steps/seed reached
    [583.9, 546.8, 461.4] -> 530.7, 12.3 sigma by T2.01's own metric. Double
    the transformer's return. Strong prior, but not a ledger claim — no
    pre-registration, no control, different hardware. This spec is that claim
    done properly.

TWO DESIGN CHOICES WORTH DEFENDING:

The baseline is SB3 PPO with RL-Zoo3-flavoured hyperparameters, NOT our
pipeline with an MLP bolted in. Deliberate: D2 says choose what is best
proven, and the honest question is "does the chosen architecture + pipeline
beat standard practice at equal experience", not "does it beat a strawman
sharing our pipeline's weaknesses". If standard practice wins, the kill
criterion says use standard practice.

The comparison is at MATCHED steps (the transformer's achieved count, target
~640K/seed), not the registry's original 2M. Throughput math, measured not
guessed: the transformer trains at ~106 env-steps/s on the P100 (v4), so
2M x 3 seeds is ~15.7h for that arm alone — over the 9h session cap and half
the 30h weekly quota to re-measure a plateau v4 already established. The
hypothesis ("beats a ~140K MLP at equal environment steps") and the kill
criterion are untouched; only the step count at which equality holds moved,
and it moved INTO the transformer's measured plateau, where more steps were
demonstrably not going to help it. RL-Zoo3's published 2M numbers (sac 6232,
td3 5567, tqc 7239) remain in the registry as context for how far EITHER arm
is from competent locomotion.

Pre-registered, before the run:
  - 3 seeds per arm, one kernel, same machine. MLP trains to >=90% of its
    seed's transformer step count or the run is VOID (throughput assumption
    failed), not a verdict.
  - Validity gate (T1.02's lesson): BOTH arms must beat random by >=3 sigma
    (sigma = max(arm seed std, random episode std)) or the run is VOID — two
    non-learners cannot arbitrate an architecture question.
  - PASS (transformer survives) needs BOTH:
      (tr_mean - mlp_mean) >= 2 * max(tr_std, mlp_std)  AND
      the transformer wins in every seed pairing.
    Anything else is "the MLP matches or wins" — FAIL, and the kill criterion
    fires: the transformer policy loses locomotion to the MLP.
  - CONTROL: untrained versions of both arms must NOT clear the 3-sigma
    learning gate. If an untrained net "learns", the gate measures
    architecture bias, not learning.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

SEEDS = [0, 1, 2]
N_ENVS = 32
ROLLOUT_STEPS = 128
TRAIN_MINUTES_PER_SEED = 100   # transformer arm: ~640K steps at v4's 106 steps/s
MLP_MINUTES_CAP = 45           # safety cap; probe needed ~27 min/seed on a slower CPU
EVAL_EPISODES = 5
RANDOM_EPISODES = 10
MIN_LEARN_SIGMA = 3.0          # validity gate, both arms, vs random
MIN_PAIR_SIGMA = 2.0           # transformer must beat MLP by this to survive
MIN_STEP_MATCH = 0.9           # MLP steps / transformer steps, else VOID

JOB = r'''
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q",
         "gymnasium[mujoco]", "stable-baselines3>=2.3"], check=True)

import json, os, time, numpy as np, torch
import gymnasium as gym
from TrainingPipeline import TrainingPipeline, PipelineConfig
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

OUT = os.path.join(os.environ["JACK_OUT"], "t202.json")


# ---------------- transformer arm: verbatim T2.01 v4 training ----------------

def eval_policy(tp, episodes, random_actions=False):
    env = tp.make_env()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            if random_actions:
                act = env.action_space.sample()
            else:
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


def train_transformer(seed, minutes):
    torch.manual_seed(seed); np.random.seed(seed)
    tp = TrainingPipeline(PipelineConfig())
    tp.make_optimizer(phase=3)
    untrained = eval_policy(tp, __EVAL_EPS__)
    envs = tp.make_vec_envs(__N_ENVS__)
    deadline = time.time() + minutes * 60
    iters = steps = 0
    curve = []
    while time.time() < deadline:
        buf = tp.collect_rollout_vec(envs, n_steps=__ROLLOUT__)
        stats = tp.rl_update(buf)
        iters += 1
        steps += __ROLLOUT__ * __N_ENVS__
        if iters % 20 == 1:
            curve.append({"iter": iters, "steps": steps,
                          "mean_reward": float(buf["rewards"].mean()),
                          "action_std": float(tp.log_std.exp().mean())})
    envs.close()
    trained = eval_policy(tp, __EVAL_EPS__)
    return {"seed": seed, "env_steps": steps, "curve": curve,
            "untrained_returns": untrained, "trained_returns": trained,
            "untrained_mean": float(np.mean(untrained)),
            "trained_mean": float(np.mean(trained))}


# ---------------- MLP arm: the local probe, run properly ----------------

class RawRewardWrapper(gym.Wrapper):
    """VecNormalize rescales rewards for training; evaluation must not see that."""
    def step(self, a):
        o, r, te, tr, i = self.env.step(a)
        i["raw_reward"] = float(r)
        return o, r, te, tr, i


class TimeCap(BaseCallback):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline
    def _on_step(self):
        return time.time() < self.deadline


def eval_mlp(model, vec, episodes):
    was_training, was_norm = vec.training, vec.norm_reward
    vec.training = False; vec.norm_reward = False
    returns = []
    for _ in range(episodes):
        obs = vec.reset()
        done, total = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec.step(act)
            total += float(infos[0].get("raw_reward", 0.0))
            done = bool(dones[0])
        returns.append(total)
    vec.training = was_training; vec.norm_reward = was_norm
    return returns


def train_mlp(seed, target_steps):
    vec = VecNormalize(DummyVecEnv([lambda: RawRewardWrapper(gym.make("Humanoid-v5"))]),
                       norm_obs=True, norm_reward=True)
    # net_arch [128,128] puts actor+critic at ~132K params — the spec's "~140K".
    # The probe's 54K default already doubled the transformer; this is the
    # bigger benefit-of-the-doubt version the spec names.
    model = PPO("MlpPolicy", vec, seed=seed, verbose=0, device="cpu",
                policy_kwargs={"net_arch": [128, 128]},
                n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4,
                gamma=0.95, gae_lambda=0.9, clip_range=0.3, ent_coef=0.002)
    n_params = sum(p.numel() for p in model.policy.parameters())
    untrained = eval_mlp(model, vec, __EVAL_EPS__)
    model.learn(total_timesteps=int(target_steps),
                callback=TimeCap(time.time() + __MLP_CAP__ * 60),
                progress_bar=False)
    trained = eval_mlp(model, vec, __EVAL_EPS__)
    steps_done = int(model.num_timesteps)
    vec.close()
    return {"seed": seed, "params": int(n_params), "env_steps": steps_done,
            "untrained_returns": untrained, "trained_returns": trained,
            "untrained_mean": float(np.mean(untrained)),
            "trained_mean": float(np.mean(trained))}


t0 = time.time()
tp0 = TrainingPipeline(PipelineConfig())
random_returns = eval_policy(tp0, __RANDOM_EPS__, random_actions=True)
del tp0

out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "random_returns": random_returns, "seeds": []}
for s in __SEEDS__:
    tr = train_transformer(s, __MINUTES__)
    mlp = train_mlp(s, tr["env_steps"])
    out["seeds"].append({"transformer": tr, "mlp": mlp})
    out["wall_minutes"] = round((time.time() - t0) / 60, 1)
    # Partial dump per seed: a timeout should cost the last seed, not the run.
    json.dump(out, open(OUT, "w"), indent=1)
    print("SEED", s, "tr", round(tr["trained_mean"], 1), "@", tr["env_steps"],
          "mlp", round(mlp["trained_mean"], 1), "@", mlp["env_steps"], flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print("DONE", json.dumps({k: out[k] for k in ("gpu", "wall_minutes")}), flush=True)
'''


def _submit() -> dict:
    body = (JOB.replace("__SEEDS__", repr(SEEDS))
               .replace("__N_ENVS__", repr(N_ENVS))
               .replace("__ROLLOUT__", repr(ROLLOUT_STEPS))
               .replace("__MINUTES__", repr(TRAIN_MINUTES_PER_SEED))
               .replace("__MLP_CAP__", repr(MLP_MINUTES_CAP))
               .replace("__EVAL_EPS__", repr(EVAL_EPISODES))
               .replace("__RANDOM_EPS__", repr(RANDOM_EPISODES)))
    job = build_job(body)
    # Expected ~6.7h (3 x 100min transformer + 3 x ~30min MLP); timeout below
    # the runner's gpu<8h child timeout (36000s) per T2.01's lesson.
    res = submit(job, prefer="kaggle", est_hours=7.0, timeout_s=32000,
                 fetch=["t202.json"])
    if not res.ok:
        raise RuntimeError(f"GPU job failed on {res.backend}: {res.message}")
    path = res.artifacts.get("t202.json")
    if not path:
        raise RuntimeError(f"no artifact from {res.backend}. message={res.message!r} "
                           f"stdout_tail={res.stdout[-400:]!r}")
    d = json.loads(Path(path).read_text())
    d["backend"] = res.backend
    return d


# ONE submission per spec: run_spec calls _experiment once per seed, and the
# kernel trains all seeds. See T2.01's 11-GPU-hour scar.
_CACHE: dict = {}


def _stats(vals):
    n = len(vals)
    m = sum(vals) / n
    return m, (sum((v - m) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit())
    rnd_mean, rnd_std = _stats(_CACHE["random_returns"])
    tr_means = [p["transformer"]["trained_mean"] for p in _CACHE["seeds"]]
    mlp_means = [p["mlp"]["trained_mean"] for p in _CACHE["seeds"]]
    tr_mean, tr_std = _stats(tr_means)
    mlp_mean, mlp_std = _stats(mlp_means)
    step_match = min(p["mlp"]["env_steps"] / max(p["transformer"]["env_steps"], 1)
                     for p in _CACHE["seeds"])
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "wall_minutes": _CACHE["wall_minutes"],
        "transformer_steps": [p["transformer"]["env_steps"] for p in _CACHE["seeds"]],
        "mlp_steps": [p["mlp"]["env_steps"] for p in _CACHE["seeds"]],
        "mlp_params": _CACHE["seeds"][0]["mlp"]["params"],
        "step_match_ratio": round(step_match, 3),
        "random_mean": round(rnd_mean, 1), "random_std": round(rnd_std, 2),
        "transformer_means": [round(m, 1) for m in tr_means],
        "mlp_means": [round(m, 1) for m in mlp_means],
        "transformer_mean": round(tr_mean, 1), "transformer_std": round(tr_std, 2),
        "mlp_mean": round(mlp_mean, 1), "mlp_std": round(mlp_std, 2),
        "tr_sigma_vs_random": round((tr_mean - rnd_mean) / max(tr_std, rnd_std, 1e-6), 2),
        "mlp_sigma_vs_random": round((mlp_mean - rnd_mean) / max(mlp_std, rnd_std, 1e-6), 2),
        "pair_sigma_advantage": round((tr_mean - mlp_mean) / max(tr_std, mlp_std, 1e-6), 2),
        "transformer_wins_all_seeds": all(t > m for t, m in zip(tr_means, mlp_means)),
    }


def _control(seed: int) -> dict:
    """Untrained versions of both arms must not clear the learning gate."""
    rnd_mean, rnd_std = _stats(_CACHE["random_returns"])
    utr = [p["transformer"]["untrained_mean"] for p in _CACHE["seeds"]]
    uml = [p["mlp"]["untrained_mean"] for p in _CACHE["seeds"]]
    utr_m, utr_s = _stats(utr)
    uml_m, uml_s = _stats(uml)
    return {
        "untrained_tr_mean": round(utr_m, 1),
        "untrained_mlp_mean": round(uml_m, 1),
        "untrained_tr_sigma": round((utr_m - rnd_mean) / max(utr_s, rnd_std, 1e-6), 2),
        "untrained_mlp_sigma": round((uml_m - rnd_mean) / max(uml_s, rnd_std, 1e-6), 2),
    }


def _check(m: dict, c: dict) -> Status | bool:
    # These three paths are VOID, not FAIL: the run could not test the claim.
    # They returned a bare `False` until 2026-08-09, and run_spec maps False ->
    # FAIL "pre-registered threshold not met" — which fires this spec's `kills`
    # field ("the transformer policy") off a comparison that explicitly refused
    # to arbitrate. That corruption had to be repaired by hand once already.
    if m["step_match_ratio"] < MIN_STEP_MATCH:
        m["verdict"] = (f"VOID — MLP reached only {m['step_match_ratio']:.0%} of the "
                        "transformer's steps; the comparison is not at equal "
                        "experience. Raise MLP_MINUTES_CAP, do not compare.")
        return Status.VOID
    if m["tr_sigma_vs_random"] < MIN_LEARN_SIGMA or m["mlp_sigma_vs_random"] < MIN_LEARN_SIGMA:
        m["verdict"] = ("VOID — an arm failed the 3-sigma learning gate vs random "
                        f"(tr {m['tr_sigma_vs_random']}, mlp {m['mlp_sigma_vs_random']}). "
                        "Two non-learners cannot arbitrate the architecture.")
        return Status.VOID
    if c["untrained_tr_sigma"] >= MIN_LEARN_SIGMA or c["untrained_mlp_sigma"] >= MIN_LEARN_SIGMA:
        # An untrained net clearing the gate means the gate measures bias, not
        # learning — the measurement is an artifact, so there is nothing to
        # refute. VOID for the same reason as the two above.
        m["verdict"] = ("VOID — an UNTRAINED net cleared the 3-sigma learning gate "
                        f"(tr {c['untrained_tr_sigma']}, mlp {c['untrained_mlp_sigma']}). "
                        "The gate is measuring architectural bias, not learning.")
        return Status.VOID
    return (m["pair_sigma_advantage"] >= MIN_PAIR_SIGMA
            and m["transformer_wins_all_seeds"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.02"], _experiment, _check, control_fn=_control, ledger=ledger)
