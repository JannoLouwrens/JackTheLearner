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
0.4c iii: `--noise-file` (file_noise.py) feeds the two samplers from an .npz, eager only.
0.4d: `--dump-step N --dump X.npz` records call N's grads by name, loss and aux outs (install_capture);
`--compare-npz A B` checks two dumps bit for bit.
0.4c: `--jit False` = compiler off; needs `PYTHONPATH=/data/tmp/ninjax362` (ninjax 3.6.2) + the AOT skip below.

0.4b controls (must DIFFER from the seed-0 run, or the identity above proves nothing):
`--seed 1` (agent seed: init weights + seed stream) and `--step-seed-offset 1000` (same init,
only the per-step seeds handed to train() shifted — shows train() actually consumes its seed,
so identity is not the trivial output of a seed-blind step). `--compare A B` prints counts.

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
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-seed-offset", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--jit", choices=["True", "False"], default="True",
                    help="0.4c: the official `jax.jit` flag; False = compiler off")
    ap.add_argument("--disable-jit", action="store_true",
                    help="0.4c ii: config jit kept True, train() called under jax.disable_jit()")
    ap.add_argument("--trace", default="pol/head/action/mean,slowval",
                    help="0.4c: comma list of param-name substrings hashed after EVERY step")
    ap.add_argument("--noise-file", help="0.4c iii: .npz noise for the two samplers (made if absent)")
    ap.add_argument("--noise-seed", type=int, default=0, help="seed used only when making the file")
    ap.add_argument("--dump-step", type=int, default=None,
                    help="0.4d: 1-based train call whose intermediates are dumped (eager only)")
    ap.add_argument("--dump", help="0.4d: .npz path for the intermediates")
    ap.add_argument("--compare-npz", nargs=2, metavar=("A", "B"))
    a = ap.parse_args(argv)
    if a.compare_npz:
        return compare_npz(*a.compare_npz)
    if a.compare:
        return compare(*a.compare)
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
        "--task", "jack74_v0", "--seed", str(a.seed), "--jax.compute_dtype", "float32",
        "--jax.jit", a.jit,
        "--logdir", tempfile.mkdtemp(prefix="g1probe_")])
    if a.jit == "False":
        # 0.4c ii patch (2 methods, listed as a difference): official `jit: False` sets
        # jax_disable_jit globally (internal.py:36) but agent.py:457/467 still AOT-lower the
        # train/report fns, which JAX refuses with jit disabled. Skip the AOT step so the
        # jax.jit-wrapped fns run eagerly. Nothing else inside the agent touched.
        from embodied.jax import agent as ejagent
        ejagent.Agent._compile_train = lambda self: None
        ejagent.Agent._compile_report = lambda self: None
    noise = None
    if a.noise_file:
        assert a.jit == "False", "file noise needs --jit False (a Python cursor under jit is baked in)"
        from embodied.jax import outs as ejouts
        from experiments.brain_a.file_noise import FileNoise
        noise = FileNoise(a.noise_file, seed=a.noise_seed).install(ejouts)
    cap = None
    if a.dump_step:
        assert a.jit == "False", "intermediates need --jit False (concrete arrays)"
        cap = install_capture()
    agent = dv3main.make_agent(config)
    noise_at_init = noise.snapshot() if noise else None

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

    import jax
    tr = [t for t in a.trace.split(",") if t]

    def traced():
        return {k: hashlib.sha256(np.asarray(jax.device_get(v)).tobytes()).hexdigest()[:16]
                for k, v in sorted(agent.params.items()) if any(t in k for t in tr)}

    carry = agent.init_train(B)
    mets_all, trace, draws = [], [traced()], []
    for i in range(a.steps):
        data = internal.device_put({k: v.copy() for k, v in batch.items()}, agent.train_sharded)
        data["seed"] = agent._seeds(i + a.step_seed_offset, agent.train_mirrored)
        if cap is not None:
            cap["on"] = (i + 1 == a.dump_step)
        if a.disable_jit:
            with jax.disable_jit():
                carry, _, mets = agent.train(carry, data)
        else:
            carry, _, mets = agent.train(carry, data)
        mets_all.append({k: float(np.asarray(v).mean()) for k, v in mets.items()
                         if np.asarray(v).dtype.kind == "f"})
        trace.append(traced())
        if noise:
            draws.append(noise.snapshot())
    if cap is not None:
        np.savez(a.dump, **cap["arrays"])
        print("DUMP", a.dump, "arrays", len(cap["arrays"]))
    # first step after which each traced param differs from its init (None = never moved)
    first_move = {k: next((i for i, h in enumerate(trace) if h[k] != trace[0][k]), None)
                  for k in trace[0]}
    params = {k: hashlib.sha256(np.asarray(jax.device_get(v)).tobytes()).hexdigest()[:16]
              for k, v in sorted(agent.params.items())}
    rec = {"versions": versions(), "steps": a.steps, "seed": a.seed,
           "step_seed_offset": a.step_seed_offset, "jit": a.jit, "disable_jit": a.disable_jit,
           "first_move": first_move, "batch_keys": sorted(batch),
           "B": B, "T": T, "noise_file": a.noise_file, "noise_at_init": noise_at_init, "noise_draws": draws, "n_params": len(params), "params": params, "metrics": mets_all}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print("G1_PROBE", a.out, "n_params", len(params), "metric calls", len(mets_all))
    for k, v in first_move.items():
        print("FIRST_MOVE", k, v)
    return 0


def install_capture():
    """0.4d: wrap `ninjax.grad` as seen by `embodied/jax/opt.py` (a module-attribute proxy; the
    official file is not edited). On the armed call it records, as concrete eager arrays: every
    gradient by param name (`grad/<name>`), the scaled loss (`loss/total_scaled`), and the agent
    loss's aux `outs` — encoder tokens, the posterior features `repfeat` (deter/stoch/logit) and the
    per-element losses (`outs/losses/<k>`). Not reached without editing the agent: the imagined
    features and returns (inside `Agent.loss`, not returned)."""
    import numpy as np
    import jax
    import ninjax
    from embodied.jax import opt as ejopt
    cap = {"on": False, "arrays": {}}

    def flat(prefix, tree):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            key = prefix + "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
            cap["arrays"][key] = np.asarray(jax.device_get(leaf))

    def grad(fn, modules, has_aux=False, **kw):
        inner = ninjax.grad(fn, modules, has_aux=has_aux, **kw)

        def run(*args, **kwargs):
            out = inner(*args, **kwargs)
            if cap["on"] and not ninjax.creating():
                loss, params, grads, aux = out
                cap["arrays"]["loss/total_scaled"] = np.asarray(jax.device_get(loss))
                flat("grad/", grads)
                if isinstance(aux, tuple) and len(aux) == 4 and isinstance(aux[2], dict):
                    flat("outs/", aux[2])
            return out
        return run

    class Proxy:
        def __getattr__(self, k):
            return grad if k == "grad" else getattr(ninjax, k)
    ejopt.nj = Proxy()
    return cap


def compare_npz(pa, pb):
    """0.4d: arrays equal bit for bit between two dumps; IDENTICAL iff same keys and all equal."""
    import numpy as np
    A, B = np.load(pa), np.load(pb)
    keys = sorted(set(A.files) | set(B.files))
    eq = sum(k in A.files and k in B.files and A[k].shape == B[k].shape
             and A[k].tobytes() == B[k].tobytes() for k in keys)
    groups = {}
    for k in keys:
        groups.setdefault(k.split("/")[0], []).append(k)
    for g, ks in sorted(groups.items()):
        ge = sum(k in A.files and k in B.files and A[k].tobytes() == B[k].tobytes() for k in ks)
        print(f"  {g}: {ge}/{len(ks)} equal")
    print(f"COMPARE_NPZ arrays_equal {eq}/{len(keys)} -> " + ("IDENTICAL" if eq == len(keys) else "DIFFER"))
    return 0


def compare(pa, pb):
    """Counts of equal params / metric values between two probe JSONs; IDENTICAL iff all equal."""
    A, B = (json.loads(pathlib.Path(p).read_text()) for p in (pa, pb))
    keys = sorted(set(A["params"]) | set(B["params"]))
    peq = sum(A["params"].get(k) == B["params"].get(k) for k in keys)
    mv = [(x.get(k), y.get(k)) for x, y in zip(A["metrics"], B["metrics"])
          for k in sorted(set(x) | set(y))]
    meq = sum(u == v for u, v in mv)
    same = peq == len(keys) and meq == len(mv) and len(A["metrics"]) == len(B["metrics"])
    print(f"COMPARE params_equal {peq}/{len(keys)} metrics_equal {meq}/{len(mv)} -> "
          + ("IDENTICAL" if same else "DIFFER"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
