"""Harness self-check #8, second half (J4/§8), paid on J8's Brain B learner:
reward forever -> the trained critic sits at its fixed point 1/(1-gamma)
(entropy-corrected), and a life cap must not pull it down.

Rule (written and committed BEFORE the registered run, 11-16 19:35 UTC):
  World: a constant observation (zeros, the pilot world's observation shape),
  the pilot world's action space, reward +1 on every decision, never a death;
  a cap every CAP decisions reported as TRUNCATED (the pilot's convention).
  Learner: `pilot_b.make_b` UNCHANGED (SBX SAC, gamma 0.995, tau 0.005,
  ent_coef 0.1 fixed, gSDE, layer-norm 256x256, warm-up 5,000, 32/32),
  run for N decisions, seed 301.
  Prediction: SBX's target is r + gamma (1-done) (min Q'(s',a') - alpha logpi(a'|s')),
  so with done == 0 always the fixed point is
      Q* = (1 - gamma * alpha * E[logpi]) / (1 - gamma)       (= 200 when E[logpi] = 0)
  with E[logpi] measured from 4,096 actor samples at the end of the run.
  Measured: mean over the same 4,096 samples of min(Q1, Q2)(s, a), current params.
  PASS iff |Q_measured / Q* - 1| <= 0.05.
  Control that MUST fail: the identical run with the cap reported as
  TERMINATED (the time-limit bug). Its fixed point is
      Q*_ctrl = (1 - g' alpha E[logpi]) / (1 - g'),  g' = gamma (1 - 1/CAP)
  (~86 at CAP 150), so the main check must FAIL on it; if it passes, VOID.
  Reported beside, not gated: whether the control lands within 5% of Q*_ctrl.
  N and CAP are fixed here: N = 200,000 (the Polyak/gamma time constant is
  ~1/((1-gamma) tau) = 40,000 updates; 200k = 5 constants -> residual < 1%),
  CAP = 150. If the main run misses because it had not converged, that is a
  FAIL of this check at this N, reported as such — N is not raised after
  seeing the number without a re-registration commit.

  Where it runs (probe 11-16 19:35, 12,000 decisions = 7,000 updates, NOT a
  verdict): 210 CPU-s -> ~30 ms per update on the box, so N = 200k is ~1.7
  CPU-h per arm, ~3.3 for both: over the box's 15 CPU-min line -> one Kaggle
  CPU kernel, both arms. Probe read Q 50.3 vs Q* 308.8 (E[logpi] -5.46): far
  from converged at 7k updates, as the time constant predicts. The process
  used ~2.4 cores despite `_single_thread` (setdefault does not override an
  inherited XLA_FLAGS / thread pool) — run it under taskset on the box.

Run: nice -n 19 /data/venvs/jack-std/bin/python -m experiments.harness.check8_fixed_point [main|control|both]
Out: /data/jack-logs/harness/check8_fixed_point.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N = 200_000
CAP = 150
SEED = 301
SAMPLES = 4096
TOL = 0.05
OUT = Path("/data/jack-logs/harness/check8_fixed_point.json")


def _run(control: bool, n: int = N) -> dict:
    from experiments import pilot_b as pb
    pb._single_thread()
    import gymnasium as gym
    import numpy as np
    import jax
    import jax.numpy as jnp
    from stable_baselines3.common.vec_env import DummyVecEnv

    world = pb.make_world(SEED)
    obs_shape, act_space = world.observation_space.shape, world.action_space
    try:
        world.close()
    except Exception:
        pass

    class Forever(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, obs_shape, np.float32)
        action_space = act_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return np.zeros(obs_shape, np.float32), {}

        def step(self, action):
            self.t += 1
            end = self.t >= CAP
            term, trunc = (end, False) if control else (False, end)
            return np.zeros(obs_shape, np.float32), 1.0, term, trunc, {}

    venv = DummyVecEnv([Forever])
    model = pb.make_b(venv, SEED, n)
    t0 = time.process_time()
    model.learn(total_timesteps=n)
    cpu = time.process_time() - t0

    pol = model.policy
    obs = jnp.zeros((SAMPLES,) + obs_shape, jnp.float32)
    dist = pol.actor_state.apply_fn(pol.actor_state.params, obs)
    acts = dist.sample(seed=jax.random.PRNGKey(SEED))
    logp = float(jnp.mean(dist.log_prob(acts)))
    q = pol.qf_state.apply_fn(pol.qf_state.params, obs, acts,
                              rngs={"dropout": jax.random.PRNGKey(0)})
    qmin = float(jnp.mean(jnp.min(q, axis=0)))
    alpha, g = 0.1, pb.GAMMA
    q_star = (1 - g * alpha * logp) / (1 - g)
    gc = g * (1 - 1 / CAP)
    q_star_ctrl = (1 - gc * alpha * logp) / (1 - gc)
    ratio = qmin / q_star
    return {"control": control, "N": n, "CAP": CAP, "seed": SEED,
            "cpu_s": round(cpu, 1), "E_logpi": round(logp, 4),
            "Q_measured": round(qmin, 3), "Q_star": round(q_star, 3),
            "ratio": round(ratio, 4), "within_tol": abs(ratio - 1) <= TOL,
            "Q_star_ctrl": round(q_star_ctrl, 3),
            "ratio_vs_ctrl_pred": round(qmin / q_star_ctrl, 4)}


def main(argv) -> int:
    which = argv[1] if len(argv) > 1 else "both"
    n = int(argv[2]) if len(argv) > 2 else N   # anything but N is a PROBE, never the verdict
    res = json.loads(OUT.read_text()) if OUT.exists() and n == N else {}
    for arm in (["main", "control"] if which == "both" else [which]):
        r = _run(arm == "control", n)
        print(json.dumps(r), flush=True)
        if n == N:
            res[arm] = r
    if n != N:
        print("PROBE (N != registered N): no verdict")
        return 0
    if "main" in res and "control" in res:
        m, c = res["main"]["within_tol"], res["control"]["within_tol"]
        res["verdict"] = "VOID" if c else ("PASS" if m else "FAIL")
        print("VERDICT", res["verdict"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
