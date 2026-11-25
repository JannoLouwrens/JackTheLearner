"""MLX port of DreamerV3 — rung 8a (PORT report step 2.4, first world-model loss row):
the dynamics and representation losses with free bits, over rung 7's 64-step pass at size1m.

Source: PORT report ladder row 8 "World-model losses — relative 1e-5 on means — golden: yes";
step 2.4 "Encoder, decoder, heads; world-model losses". This file is the dyn/rep part only
(the prediction losses need the heads — the next row). Official code `danijar/dreamerv3` at
e01491f, `dreamerv3/rssm.py` `RSSM.loss` L120-133 and `_dist` L173-176:
  dyn = Agg(OneHot(sg(post), unimix), 1, sum).kl(Agg(OneHot(prior, unimix)))
  rep = Agg(OneHot(post, unimix)).kl(Agg(OneHot(sg(prior), unimix)))
  if free_nats: dyn, rep = max(dyn, free_nats), max(rep, free_nats)     (free_nats 1.0)
  -> [B, T] each; prior = `_prior(feat['deter'])` over the whole pass.
Run end to end: the official `rssm.loss` (its own observe scan + prior + KL), under
`jax.disable_jit()`, same params (same PRNGKey(0) creation pass as rung 7's golden — checked
equal to it) and the same file noise (same file, same cursor bookkeeping — checked: its post
and prior equal rung 7's golden). A second official RSSM with free_nats= on the SAME params
FSPLIT=7.5 (chosen after the first run measured raw KL 2.45-11.63, median 7.62, all above 1.0)
makes the max() bind on part of the elements, so the floor's code path is compared too.

"Ours beside theirs": NumPy float64 reference vs official float32 (COMPUTE_DTYPE f32 as
A1-A3; published bf16). Widths are size1m's (rung 7's SIZE1M); inputs are the toy's (tokens
12, toy action space, batch 4) — rung 7's listed difference, unchanged. Tolerance: ladder row
8 says "relative 1e-5 on means" — read here as |port − ref| / (|ref| + 1) ≤ 1e-5 on the
[B,T] mean AND ≤ 1e-4 (rung 7's tolerance) per element; the per-element bound is ours.

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \
      experiments/brain_a/rung8a.py --vs-official [--dump-golden]
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung8a.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if "--preset" not in sys.argv:                 # rung 8a runs at size1m only (rung 7 reads argv)
    sys.argv += ["--preset", "size1m"]
import rung7  # noqa: E402
from rung1 import OFFICIAL, REF_kl  # noqa: E402
from rung6 import TOY, B, CONT, DISC, CONT_DIM, DISC_N  # noqa: E402

T = rung7.T
FREE = 1.0
FSPLIT = 7.5   # a second official pass, same params + noise: the floor must bind on some elements
TOL_MEAN = 1e-5
TOL_EL = 1e-4
GOLDEN8 = os.path.join(HERE, "rung8a_dynrep_golden_size1m_e01491f.npz")
assert TOY["deter"] == 512, "rung 8a is a size1m rung"


def close(name, got, ref):
    got, ref = np.float64(np.asarray(got)), np.float64(ref)
    if got.shape != ref.shape:
        return False, f"{name}: shape {got.shape} != {ref.shape}"
    el = float((np.abs(got - ref) / (np.abs(ref) + 1)).max())
    mn = float(abs(got.mean() - ref.mean()) / (abs(ref.mean()) + 1))
    ok = bool(np.all(np.isfinite(got)) and el <= TOL_EL and mn <= TOL_MEAN)
    return ok, f"{name}: mean {mn:.3g}, element {el:.3g} x(|ref|+1) {'ok' if ok else 'OVER'}"


def REF_dynrep(post, prior, free=FREE, mix=TOY["unimix"]):
    """f64: dyn and rep are the same VALUE (stop-gradients differ only in the backward pass)."""
    v, raw = REF_kl(np.float64(post), np.float64(prior), mix, free)
    return dict(dyn=v, rep=v, raw=raw)


def floor_share(raw):
    return float((raw > FREE).mean())


# ---------------------------------------------------------------- official
def main_official():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax.nets as nets
    import embodied.jax.outs as outs
    nets.COMPUTE_DTYPE = jnp.float32
    from dreamerv3.rssm import RSSM
    from file_noise import FileNoise
    noise = FileNoise(rung7.NOISE, seed=7, n=rung7.NOISE_N).install(outs)   # rung 7's file + seed
    act_space = {CONT: elements.Space(np.float32, (CONT_DIM,), -1, 1),
                 DISC: elements.Space(np.int32, (), 0, DISC_N)}
    I = rung7.inputs()
    rssm = RSSM(act_space, name="rssm", **TOY)                        # free_nats default 1.0
    assert rssm.free_nats == FREE
    rssmS = RSSM(act_space, name="rssm", free_nats=FSPLIT, **TOY)     # same params, floor binds
    rows = []

    def put(ok, d):
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    def run(model):
        def f(tokens, action, reset):
            carry = model.initial(B)
            _, _, losses, feat, mets = model.loss(carry, tokens, action, reset, False)
            return dict(dyn=losses["dyn"], rep=losses["rep"], post=feat["logit"],
                        prior=model._prior(feat["deter"]), stoch=feat["stoch"],
                        dyn_ent=mets["dyn_ent"], rep_ent=mets["rep_ent"])
        return f

    J = lambda d: jax.tree.map(jnp.asarray, d)  # noqa: E731
    args = (jnp.asarray(I["tokens"]), J(I["action"]), jnp.asarray(I["reset"]))
    with jax.disable_jit():
        state, _ = nj.pure(run(rssm))({}, *args, create=True, seed=jax.random.PRNGKey(0))
        cur = dict(noise.cur)                     # the split pass must read the SAME uniforms
        _, out = nj.pure(run(rssm))(state, *args, seed=jax.random.PRNGKey(1))
        noise.cur.update(cur)
        _, out0 = nj.pure(run(rssmS))(state, *args, seed=jax.random.PRNGKey(1))
    out = {k: np.asarray(v) for k, v in out.items()}
    out0 = {k: np.asarray(v) for k, v in out0.items()}
    P7, _, _, g7 = rung7._load_golden()
    same_p = all(np.array_equal(np.asarray(state[f"rssm/{k}"], np.float32), v) for k, v in P7.items())
    put(same_p and len(P7) == len(state), f"params == rung 7 size1m golden's ({len(P7)} arrays, bit for bit)")
    for k in ("post", "prior", "stoch"):
        d = float(np.abs(out[k] - g7[k]).max())
        put(d == 0.0, f"official loss pass {k} == rung 7 golden (max |d| {d:.3g})")
    put(bool(np.array_equal(out["post"], out0["post"])), f"free_nats={FSPLIT} pass sees the same post (cursor rewound)")
    put(out["dyn"].shape == (B, T) and out["rep"].shape == (B, T), f"dyn/rep shape [B,T] = {out['dyn'].shape}")
    ref = REF_dynrep(g7["post"], g7["prior"])
    refS = REF_dynrep(g7["post"], g7["prior"], FSPLIT)
    sh = floor_share(ref["raw"])
    print(f"        raw KL: min {ref['raw'].min():.4g}, median {np.median(ref['raw']):.4g}, max "
          f"{ref['raw'].max():.4g}; above the 1.0 floor {sh:.1%}")
    for k in ("dyn", "rep"):
        put(*close(f"{k} (free 1) f32 vs f64", out[k], ref[k]))
        put(*close(f"{k} (free {FSPLIT}) f32 vs f64", out0[k], refS[k]))
    # at size1m init every raw KL is above the published floor 1.0 (measured 11-25: min 2.45),
    # so free 1 alone never exercises the max(); the FSPLIT pass must put elements on both sides
    shS = float((ref["raw"] > FSPLIT).mean())
    put(0.05 < shS < 0.95, f"free {FSPLIT} splits the elements ({shS:.1%} above, need 5-95%)")
    # planted wrong references must each FAIL against the official raw and floored values
    mix = TOY["unimix"]
    mut = {
        "no unimix": REF_kl(np.float64(g7["post"]), np.float64(g7["prior"]), 0.0, FSPLIT),
        "free bits ignored": REF_kl(np.float64(g7["post"]), np.float64(g7["prior"]), mix, 0.0),
        "KL direction swapped": REF_kl(np.float64(g7["prior"]), np.float64(g7["post"]), mix, FSPLIT),
        "free bits on the T-mean, not per step": (np.maximum(REF_kl(np.float64(g7["post"]),
            np.float64(g7["prior"]), mix, 0.0)[1].mean(1, keepdims=True), FSPLIT).repeat(T, 1), None),
        "prior from the posterior's deter one step late": REF_kl(np.float64(g7["post"]),
            np.float64(np.concatenate([g7["prior"][:, :1], g7["prior"][:, :-1]], 1)), mix, FSPLIT),
    }
    for name, (v, raw) in mut.items():
        bad = not close("m", out0["dyn"], v)[0]       # each mutant at FSPLIT vs the official FSPLIT pass
        put(bad, f"planted '{name}': {'CAUGHT' if bad else 'MISSED'}")
    print(f"        official dyn_ent {float(out['dyn_ent']):.6g}, rep_ent {float(out['rep_ent']):.6g}")
    if "--dump-golden" in sys.argv:
        np.savez(GOLDEN8, dyn=out["dyn"], rep=out["rep"], dyn_split=out0["dyn"], rep_split=out0["rep"],
                 free_split=np.float32(FSPLIT))
        print(f"golden written: {GOLDEN8}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f, size1m, T {T}): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv[1:]:
        sys.exit(main_official())
    print("port tests: not yet written (next commit)")
    sys.exit(1)
