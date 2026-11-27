"""J29 / KNOWN_ANSWERS row B-7 (frozen c848d888): the twin `B7flat` that MUST FAIL.

Memoryless Brain B (`brain_b.make_row1("ours")`'s recipe: SBX 0.28 SAC + LayerNorm, the
rl-zoo3 2.9.2 `Pendulum-v1` entry = lr 1e-3, every other setting SB3/SBX default) on the
SAME velocity-only Pendulum-V wrapper as `brain_b_mem`, 51,000 env steps, seeds 0-3, the
same eval schedule (20 deterministic episodes at every 1,000 env steps from 2,000 to
51,000) and the same seed score (`brain_b_mem.b7_seed_score`, mean over [40k, 50k]).
VOID (not a memory test as built) if the 4-seed mean is >= -190.9.

Twin beside the arm — every difference, said:
| setting            | B7mem (Ni's recipe)                  | B7flat (Brain B, Zoo)             |
| memory             | LSTM-128 over (prev action, r, obs)  | none: acts on the 1-d velocity    |
| start              | 5 random episodes, then 1,000 updates| learning_starts 100 (SBX default) |
| updates            | 1 a step                             | 1 a step (train_freq 1)           |
| gamma / batch / lr | 0.9 / 32 episodes / 3e-4             | 0.99 / 256 / 1e-3 (Zoo)           |
| action             | tanh in [-1,1] -> torque 2a          | identical (action_space [-1,1])   |
| eval env           | PendulumV(), unseeded                | PendulumV(), reset seeded 10_000+seed once |
The twin keeps Brain B's own settings on purpose: the question is "does the B we already
have solve Pendulum-V without memory?", not "does Ni's learner without its LSTM".

Run: python -m experiments.b7_flat --seed S --out /data/jack-logs/j29/b7flat_sS.json
"""
import argparse, json, os, time
import numpy as np
import gymnasium as gym

from experiments.brain_b_mem import PendulumV, b7_seed_score

STEPS, EVAL_EVERY, EVAL_FROM, EVAL_EPS = 51_000, 1_000, 2_000, 20


class PendulumVUnit(gym.Wrapper):
    """PendulumV with its true action range declared ([-1,1]; the wrapper scales by 2),
    so an SB3-family learner samples and squashes into the range B7mem acts in."""
    def __init__(self):
        super().__init__(PendulumV())
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = self.env.observation_space


def make_flat(seed, **over):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from sbx import SAC
    from stable_baselines3.common.monitor import Monitor
    kw = dict(learning_rate=1e-3, seed=seed, policy_kwargs=dict(layer_norm=True))
    kw.update(over)
    return SAC("MlpPolicy", Monitor(PendulumVUnit()), **kw)


def evaluate(model, env, n):
    rets = []
    for _ in range(n):
        o, _ = env.reset(); done, R = False, 0.0
        while not done:
            a, _ = model.predict(o, deterministic=True)
            o, r, te, tr, _ = env.step(a); R += r; done = te or tr
        rets.append(R)
    return float(np.mean(rets))


def run_seed(seed, out=None, steps=STEPS, every=EVAL_EVERY, start=EVAL_FROM, eps=EVAL_EPS, **over):
    from stable_baselines3.common.callbacks import BaseCallback
    model = make_flat(seed, **over)
    ev_env = PendulumVUnit(); ev_env.reset(seed=10_000 + seed)
    evals, t0 = [], time.process_time()

    class Marks(BaseCallback):
        def _on_step(self):
            n = self.num_timesteps
            if n >= start and n % every == 0:
                evals.append(dict(step=n, ret=evaluate(self.model, ev_env, eps),
                                  cpu_s=round(time.process_time() - t0, 1)))
                _write(out, seed, False, evals)
            return True

    model.learn(total_timesteps=steps, callback=Marks())
    _write(out, seed, True, evals)
    return evals


def _write(out, seed, complete, evals):
    if not out:
        return
    import sbx, stable_baselines3
    rec = dict(arm="B7flat", seed=seed, recipe="brain_b ours: SBX SAC lr 1e-3 layer_norm, PendulumV",
               sbx=sbx.__version__, sb3=stable_baselines3.__version__, complete=complete, evals=evals)
    if complete and evals:
        rec["seed_score"] = b7_seed_score(evals)
    json.dump(rec, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_seed(a.seed, a.out)
