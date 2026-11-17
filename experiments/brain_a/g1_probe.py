"""MLX port step 0.4a (G1 pre-probe): is the OFFICIAL agent's train step repeatable when called
directly on a fixed batch — no env threads, no replay, no `embodied.run.train`?

Source: danijar/dreamerv3 @ e01491f (MIT), code only. Report: PUBLISHED_EVIDENCE_2026-11-15_PORT.md
build-plan row 0.4 / gate G1 ("the golden file must be identical on two runs").
0.2b measured that the training LOOP is not repeatable (async batches); this asks whether the
agent's own `train()` is, before any file-noise patch is written. Nothing inside the agent touched.

Same config as 0.2b (`defaults debug`, float32, Dummy74 obs/act spaces, seed 0). The batch is a
fixed numpy draw (seed 1234) over `agent.spaces`; the batch is device_put and the per-step seed is
`agent._seeds(i, ...)`, exactly what `agent.stream()` does. N train calls; dumps sha256 of every param (by name) plus
the metrics each call returns. Run twice in separate processes and compare the JSONs.
Not yet: file noise for the two samplers, compiler off, intermediates (the rest of 0.4).

    /data/venvs/jack-dv3/bin/python -m experiments.brain_a.g1_probe \
        --src /data/src/dreamerv3_e01491f --out /data/jack-logs/mlx/g1_probe_r1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(pathlib.Path(a.src).resolve()))
    import elements
    import numpy as np
    import ruamel.yaml as yaml
    from dreamerv3 import main as dv3main
    from embodied.jax import internal
    from experiments.brain_a.official_debug import build_env_class, versions

    Dummy74 = build_env_class()
    dv3main.make_env = lambda config, index, **kw: dv3main.wrap_env(Dummy74(config.task), config)
    folder = pathlib.Path(dv3main.__file__).parent
    configs = yaml.YAML(typ="safe").load((folder / "configs.yaml").read_text())
    config = elements.Config(configs["defaults"]).update(configs["debug"])
    config = elements.Flags(config).parse([
        "--task", "jack74_v0", "--seed", "0", "--jax.compute_dtype", "float32",
        "--logdir", tempfile.mkdtemp(prefix="g1probe_")])
    agent = dv3main.make_agent(config)

    B, T = config.batch_size, config.batch_length + config.replay_context
    rng = np.random.default_rng(1234)
    batch = {}
    for k, sp in agent.spaces.items():
        shape = (B, T) + tuple(sp.shape)
        if sp.dtype == bool:
            v = np.zeros(shape, bool)
            if k == "is_first":
                v[:, 0] = True
        elif np.issubdtype(sp.dtype, np.floating):
            lo = -1.0 if k in ("action", "vector") else 0.0
            v = rng.uniform(lo, 1.0, shape).astype(sp.dtype)
        else:
            v = rng.integers(0, 2**31 - 1, shape).astype(sp.dtype)  # e.g. stepid
        batch[k] = v

    carry = agent.init_train(B)
    mets_all = []
    for i in range(a.steps):
        data = internal.device_put({k: v.copy() for k, v in batch.items()}, agent.train_sharded)
        data["seed"] = agent._seeds(i, agent.train_mirrored)
        carry, _, mets = agent.train(carry, data)
        mets_all.append({k: float(np.asarray(v).mean()) for k, v in mets.items()
                         if np.asarray(v).dtype.kind == "f"})
    import jax
    params = {k: hashlib.sha256(np.asarray(jax.device_get(v)).tobytes()).hexdigest()[:16]
              for k, v in sorted(agent.params.items())}
    rec = {"versions": versions(), "steps": a.steps, "batch_keys": sorted(batch),
           "B": B, "T": T, "n_params": len(params), "params": params, "metrics": mets_all}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print("G1_PROBE", a.out, "n_params", len(params), "metric calls", len(mets_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
