"""official_speed.py — the official-JAX arm of the Mac first-day row 9, automated (ORDERS "Next up" 2).

Row 9 (PUBLISHED_PRIORS.md §10 "Reproduce first"): "Dreamer steps/s on JAX, sheeprl, sheeprl-MPS at 1M then
12M, same script on T4". The sheeprl arms are `mac_dreamer.py speed`; this is the JAX arm, so the three
arms of the row can be read side by side from one `scripts/mac_first_day.py --only 9` run.

Source: danijar/dreamerv3 @ e01491f (MIT), code only, never weights; the agent starts from a fresh init.
Built exactly as `g1_probe.py` builds it: the official `configs.yaml` (`defaults` + the size preset),
the official `make_agent`, the official `agent.train(carry, data)` with the per-step seed from
`agent._seeds` — what `agent.stream()` hands it. Nothing inside the agent is touched.

Ours beside theirs (beside the official `embodied.run.train` on `dmc_proprio`):
| setting         | official                                | here                                              |
|-----------------|-----------------------------------------|---------------------------------------------------|
| task            | DMC walker, proprio vector               | a dummy 30-number vector, 6 continuous actions    |
|                 |                                          | (= mac_dreamer.py's OBS_DIM/ACT_DIM, so the arms  |
|                 |                                          | time the same shapes)                             |
| batches         | sampled from replay, async               | K = 16 fixed numpy batches (seed 1234), cycled    |
| compute dtype   | bfloat16 (`jax.compute_dtype` default)   | float32 by default (the sheeprl arm is float32);  |
|                 |                                          | `--dtype bfloat16` times the shipped setting      |
| jax platform    | cuda                                     | cpu (`--platform` to change it)                   |
| size            | `dmc_proprio` = size1m                   | `--size 1m|12m` (the official presets verbatim)   |
| everything else | batch 16 x 64 (+1 context), losses, optimiser, `jax.jit: True` — unchanged      |
A "grad step" is one `agent.train` call (world model + actor + critic on one batch), the unit of row 9.

`--warm` calls first (`make_agent` AOT-compiles the train step, so `compile_s` = the first call only), then
`--timed` calls ending with `jax.block_until_ready` on the params; prints one JSON line last.

    /data/venvs/jack-dv3/bin/python -m experiments.brain_a.official_speed \
        --src /data/src/dreamerv3_e01491f --size 1m --warm 5 --timed 10
Interpreter: a venv with the pin's requirements (box: /data/venvs/jack-dv3, README §0.2b).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

OBS_DIM, ACT_DIM, K_BATCHES = 30, 6, 16  # must equal mac_dreamer.py's (mac_first_day m9 asserts it)


def build_env_class(obs_dim=OBS_DIM, act_dim=ACT_DIM):
    import elements
    import embodied
    import numpy as np

    class DummyVec(embodied.Env):
        def __init__(self, task, length=100):
            del task
            self.length, self.count, self.done = length, 0, False

        @property
        def obs_space(self):
            return {"vector": elements.Space(np.float32, (obs_dim,)), "reward": elements.Space(np.float32),
                    "is_first": elements.Space(bool), "is_last": elements.Space(bool),
                    "is_terminal": elements.Space(bool)}

        @property
        def act_space(self):
            return {"reset": elements.Space(bool), "action": elements.Space(np.float32, (act_dim,), -1.0, 1.0)}

        def step(self, action):  # never stepped here; present so make_env is a real Env
            if action.pop("reset") or self.done:
                self.count, self.done = 0, False
                return self._obs(0.0, is_first=True)
            self.count += 1
            self.done = self.count >= self.length
            return self._obs(0.0, is_last=self.done, is_terminal=self.done)

        def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
            return dict(vector=np.zeros(obs_dim, np.float32), reward=np.float32(reward), is_first=is_first,
                        is_last=is_last, is_terminal=is_terminal)

    return DummyVec


def speed(src, size="1m", warm=500, timed=500, seed=0, dtype="float32", platform="cpu"):
    sys.path.insert(0, str(pathlib.Path(src).resolve()))
    import elements
    import jax
    import numpy as np
    import ruamel.yaml as yaml
    from dreamerv3 import main as dv3main
    from embodied.jax import internal

    Env = build_env_class()
    dv3main.make_env = lambda config, index, **kw: dv3main.wrap_env(Env(config.task), config)
    folder = pathlib.Path(dv3main.__file__).parent
    configs = yaml.YAML(typ="safe").load((folder / "configs.yaml").read_text())
    config = elements.Config(configs["defaults"]).update(configs[f"size{size}"])
    config = elements.Flags(config).parse([
        "--task", "jackvec_v0", "--seed", str(seed), "--jax.compute_dtype", dtype,
        "--jax.platform", platform, "--jax.jit", "True", "--logdir", tempfile.mkdtemp(prefix="offspeed_")])
    agent = dv3main.make_agent(config)
    B, T = config.batch_size, config.batch_length + config.replay_context
    rng = np.random.default_rng(1234)
    batches = []
    for _ in range(K_BATCHES):
        b = {}
        for k, sp in agent.spaces.items():
            shape = (B, T) + tuple(sp.shape)
            if sp.dtype == bool:
                v = np.zeros(shape, bool)
                if k == "is_first":
                    v[:, 0] = True
            elif np.issubdtype(sp.dtype, np.floating):
                v = rng.uniform(-1.0 if k in ("action", "vector") else 0.0, 1.0, shape).astype(sp.dtype)
            else:
                v = rng.integers(0, 2**31 - 1, shape).astype(sp.dtype)
            b[k] = v
        batches.append(b)
    carry = agent.init_train(B)

    def step(i):
        nonlocal carry
        data = internal.device_put(batches[i % K_BATCHES], agent.train_sharded)
        data["seed"] = agent._seeds(i, agent.train_mirrored)
        carry, _, mets = agent.train(carry, data)
        return mets

    t0 = time.perf_counter()
    mets = step(0)
    jax.block_until_ready(agent.params)
    t1 = time.perf_counter()
    for i in range(1, warm):
        mets = step(i)
    jax.block_until_ready(agent.params)
    t2 = time.perf_counter()
    for i in range(warm, warm + timed):
        mets = step(i)
    jax.block_until_ready(agent.params)
    t3 = time.perf_counter()
    finite = all(bool(np.all(np.isfinite(np.asarray(v)))) for v in mets.values()
                 if np.asarray(v).dtype.kind == "f")
    # agent.params also holds optimiser state, the slow critic and return norms: count only the seven
    # trained modules (size1m: 641,641 = mac_dreamer's sheeprl count, the same network shapes)
    n = int(sum(np.asarray(v).size for k, v in agent.params.items()
                if k.split("/")[0] not in ("opt", "slowval", "slowval_count", "retnorm")))
    n_all = int(sum(np.asarray(v).size for v in agent.params.values()))
    from experiments.brain_a.official_debug import versions
    return {"status": "RAN", "arm": "official-jax", "size": size, "platform": platform, "dtype": dtype,
            "params": n, "params_incl_opt_state": n_all, "B": B, "T": T, "warm": warm, "timed": timed, "compile_s": round(t1 - t0, 2),
            "warm_s": round(t2 - t1, 2), "timed_s": round(t3 - t2, 2),
            "grad_steps_per_s": round(timed / (t3 - t2), 3), "ms_per_step": round(1e3 * (t3 - t2) / timed, 2),
            "metrics_finite": finite, "devices": [str(d) for d in jax.devices()], "versions": versions()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/src/dreamerv3_e01491f")
    ap.add_argument("--size", choices=["1m", "12m"], default="1m")
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--timed", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--platform", default="cpu")
    a = ap.parse_args(argv)
    assert a.warm >= 1 and a.timed >= 1
    rec = speed(a.src, a.size, a.warm, a.timed, a.seed, a.dtype, a.platform)
    print(json.dumps(rec))
    return 0 if rec.get("metrics_finite") else 3


if __name__ == "__main__":
    REPO = pathlib.Path(__file__).resolve().parents[2]
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    sys.exit(main())
