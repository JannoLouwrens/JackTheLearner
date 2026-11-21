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
0.4d ii: `--dump-step 1,2,20` dumps several calls (keys prefixed `c<N>/`), and the probe FAILS (rc 3,
`ZERO_GRAD_CHECK FAIL`) if any gradient is exactly zero in EVERY dumped call unless named in `--allow-zero`
— 0.4d found the `dyn/prior*` grads all-zero at call 20 (free-nats floor), a hole a golden file
must not carry silently.
0.4d iii: `--set agent.opt.warmup=0` passes an official config flag through (listed difference).
0.4c: `--jit False` = compiler off; needs `PYTHONPATH=/data/tools/ninjax362` (ninjax 3.6.2) + the AOT skip below.

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
    ap.add_argument("--dump-step", default=None,
                    help="0.4d: 1-based train call(s), comma list, whose intermediates are dumped (eager only)")
    ap.add_argument("--allow-zero", default="",
                    help="0.4d ii: comma list of grad names allowed to be all-zero in every dumped call")
    ap.add_argument("--dump", help="0.4d: .npz path for the intermediates")
    ap.add_argument("--compare-npz", nargs=2, metavar=("A", "B"))
    ap.add_argument("--preset", choices=["debug", "size1m"], default="debug",
                    help="2.1: `debug` (G1, unchanged) or `defaults size1m` (+ debug's jax cpu/prealloc only)")
    ap.add_argument("--noise-n", type=int, default=None,
                    help="2.1: draws per kind when MAKING the noise file (size1m needs ~2.3M uniform a call)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="0.4d iii: extra official config flag, e.g. agent.opt.warmup=0 (a LISTED difference)")
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
    config = elements.Config(configs["defaults"]).update(configs[a.preset])
    # 2.1 size1m: the published preset on top of defaults; the only extra flags are the golden-file
    # rule's CPU + no-prealloc (what `debug` sets under jax:) — listed differences, nothing else.
    cpu = ["--jax.platform", "cpu", "--jax.prealloc", "False"] if a.preset == "size1m" else []
    config = elements.Flags(config).parse(cpu + [
        "--task", "jack74_v0", "--seed", str(a.seed), "--jax.compute_dtype", "float32",
        "--jax.jit", a.jit,
        "--logdir", tempfile.mkdtemp(prefix="g1probe_")]
        + [x for kv in a.set for x in ("--" + kv.split("=", 1)[0], kv.split("=", 1)[1])])
    if a.set:
        print("CONFIG_OVERRIDES", a.set, "agent.opt.warmup", config.agent.opt.warmup, flush=True)
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
        kw = {"n": a.noise_n} if a.noise_n else {}
        noise = FileNoise(a.noise_file, seed=a.noise_seed, **kw).install(ejouts)
    cap = None
    dump_steps = sorted({int(x) for x in a.dump_step.split(",")}) if a.dump_step else []
    if dump_steps:
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
            cap["on"] = (i + 1) in dump_steps
            cap["t"] = 0
            cap["prefix"] = f"c{i + 1}/" if len(dump_steps) > 1 else ""
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
        print("DUMP", a.dump, "arrays", len(cap["arrays"]), "calls", dump_steps,
              "skipped_tracers", cap.get("skipped", 0),
              "prerun_calls_not_stored", cap.get("prerun", 0))
        zero_rc = zero_grad_check(cap["arrays"], dump_steps,
                                  {x for x in a.allow_zero.split(",") if x})
    # first step after which each traced param differs from its init (None = never moved)
    first_move = {k: next((i for i, h in enumerate(trace) if h[k] != trace[0][k]), None)
                  for k in trace[0]}
    params = {k: hashlib.sha256(np.asarray(jax.device_get(v)).tobytes()).hexdigest()[:16]
              for k, v in sorted(agent.params.items())}
    rec = {"versions": versions(), "preset": a.preset, "steps": a.steps, "seed": a.seed,
           "step_seed_offset": a.step_seed_offset, "jit": a.jit, "disable_jit": a.disable_jit,
           "first_move": first_move, "batch_keys": sorted(batch),
           "B": B, "T": T, "noise_file": a.noise_file, "noise_at_init": noise_at_init, "noise_draws": draws, "n_params": len(params), "params": params, "metrics": mets_all}
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print("G1_PROBE", a.out, "n_params", len(params), "metric calls", len(mets_all))
    for k, v in first_move.items():
        print("FIRST_MOVE", k, v)
    return zero_rc if cap is not None else 0


def zero_grad_check(arrays, dump_steps, allow):
    """0.4d ii: per dumped call, the grads that are exactly zero everywhere; FAIL (3) if a grad is
    all-zero in every dumped call and not allowed. A golden file with such a grad cannot catch a
    wrong backward through it."""
    import numpy as np
    multi = len(dump_steps) > 1
    zero_in = {}
    for k, v in arrays.items():
        rest = k.split("/", 1)[1] if multi else k
        if not rest.startswith("grad/"):
            continue
        name = rest[len("grad/"):]
        zero_in.setdefault(name, [])
        if not np.any(v):
            zero_in[name].append(int(k.split("/", 1)[0][1:]) if multi else dump_steps[0])
    for c in dump_steps:
        zs = sorted(n for n, cs in zero_in.items() if c in cs)
        print(f"ZERO_GRADS call {c}: {len(zs)}/{len(zero_in)}", " ".join(zs))
    always = sorted(n for n, cs in zero_in.items() if len(cs) == len(dump_steps))
    bad = [n for n in always if n not in allow]
    print("ZERO_GRAD_CHECK", "FAIL" if bad else "PASS", f"always-zero {len(always)}",
          f"unallowed {len(bad)}", " ".join(bad))
    return 3 if bad else 0


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
                pre = cap.get("prefix", "")
                cap["arrays"][pre + "loss/total_scaled"] = np.asarray(jax.device_get(loss))
                flat(pre + "grad/", grads)
                if isinstance(aux, tuple) and len(aux) == 4 and isinstance(aux[2], dict):
                    flat(pre + "outs/", aux[2])
            return out
        return run

    class Proxy:
        def __getattr__(self, k):
            return grad if k == "grad" else getattr(ninjax, k)
    ejopt.nj = Proxy()
    install_model_capture(cap)
    return cap


def install_model_capture(cap):
    """2.1 (golden v1): the intermediates rungs 6, 7 and 9 read, recorded on the armed call by
    wrapping three official callables from outside (class attribute / module global swapped; no
    official file edited; listed difference). Values inside the loss are JVP tracers under
    `value_and_grad`; their primal is concrete when the compiler is off, and that primal is stored.
    - `RSSM._observe`, every step t of the posterior scan (rungs 6, 7): `rssm/t<ttt>/in_deter`,
      `in_stoch`, `in_action/<k>`, `in_tokens`, `in_reset` and the step's `out/{deter,stoch,logit}`.
    - `RSSM.imagine` (multi-step call; rung 9): `imag/start/<k>`, `imag/feat/<k>`, `imag/act/<k>`.
    - `agent.imag_loss` (rung 9): inputs `imagloss/in_rew`, `in_con`, outputs `imagloss/loss/<k>`
      and `imagloss/outs/<k>` (lambda-returns `ret` among them).
    A value that is still a tracer after unwrapping is NOT stored and is counted in
    `cap["skipped"]`; the probe prints that count, and a nonzero count means the file is partial."""
    import numpy as np
    import jax
    import ninjax
    from dreamerv3 import rssm as rssm_mod
    from dreamerv3 import agent as agent_mod
    cap.setdefault("skipped", 0)
    cap["t"] = 0

    def conc(x):
        while isinstance(x, jax.core.Tracer) and hasattr(x, "primal"):
            x = x.primal
        return None if isinstance(x, jax.core.Tracer) else np.asarray(x)

    def put(key, tree):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            sub = "/".join(str(getattr(q, "key", getattr(q, "idx", q))) for q in path)
            v = conc(leaf)
            if v is None:
                cap["skipped"] += 1
            else:
                cap["arrays"][cap.get("prefix", "") + key + ("/" + sub if sub else "")] = v

    cap.setdefault("prerun", 0)

    def armed():
        # NAMED 11-21 (was "a second posterior scan"): ninjax runs a function once more as a
        # state-tracking PRERUN before the real pass — `nj.grad` over the whole loss and
        # `nj.scan` over step 0 (`_prerun`, ninjax 3.6.2 l.450) — inside a context with
        # ignore=True and its own seed, so its samples differ. Those calls are not the training
        # computation; they are counted in `cap["prerun"]` and never stored.
        if not (cap["on"] and not ninjax.creating()):
            return False
        if ninjax.context().ignore:
            cap["prerun"] += 1
            return False
        return True

    orig_obs = rssm_mod.RSSM._observe

    def _observe(self, carry, tokens, action, reset, training):
        out = orig_obs(self, carry, tokens, action, reset, training)
        if armed():
            k = f"rssm/t{cap['t']:03d}/"
            put(k + "in_deter", carry["deter"]); put(k + "in_stoch", carry["stoch"])
            put(k + "in_action", action); put(k + "in_tokens", tokens); put(k + "in_reset", reset)
            put(k + "out", out[1][1])
            cap["t"] += 1
        return out
    rssm_mod.RSSM._observe = _observe

    orig_imag = rssm_mod.RSSM.imagine

    def imagine(self, carry, policy, length, training, single=False):
        out = orig_imag(self, carry, policy, length, training, single)
        if armed() and not single:
            put("imag/start", carry); put("imag/feat", out[1]); put("imag/act", out[2])
        return out
    rssm_mod.RSSM.imagine = imagine

    orig_il = agent_mod.imag_loss

    def imag_loss(act, rew, con, *args, **kw):
        out = orig_il(act, rew, con, *args, **kw)
        if armed():
            put("imagloss/in_rew", rew); put("imagloss/in_con", con)
            put("imagloss/loss", out[0]); put("imagloss/outs", out[1])
        return out
    agent_mod.imag_loss = imag_loss


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
