"""MLX port of DreamerV3 — rung 7 references (PORT report step 2.3): the 64-step memory pass.

Source: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table row
2.3: "64-step memory pass with a sampler that reads file noise or forces golden
samples — Rung 7 passes"; ladder row 7: "64-step memory pass with file noise,
1e-4; no sample flips (on a near-tie, feed the golden sample)". Official code
`danijar/dreamerv3` at e01491f, `dreamerv3/rssm.py` `RSSM.observe` (L58-70,
`nj.scan` of `_observe` over axis 1) and `embodied/jax/outs.py`:
  OneHot(logit, unimix).sample -> Categorical: probs = softmax(logit);
  probs = (1-unimix)*probs + unimix/classes; logits = log(probs); sample =
  jax.random.categorical = argmax(logits + Gumbel). With `file_noise.FileNoise`
  installed the Gumbel is -log(-log(u)), u = the file's uniforms in call order,
  clipped to [1e-7, 1-1e-7] (file_noise.py, a listed difference).
The one-step maths (core, posterior, prior, reset masking) is rung 6's
`REF_step` (reused); this rung adds the sampler and the carry across 64 steps.

How the official pass is run: `rssm.observe(..., single=False)` — the real scan
path — under `jax.disable_jit()` so `jax.lax.scan` runs as a Python loop and
the file cursor advances per step (as golden v1 ran, `--jit False`). The
sampler wrapper logs every call's cursor and whether ninjax marks it a PRERUN
(`context().ignore`, ninjax 3.6.2 `_prerun` runs step 0 once more before the
scan, and `nj.pure(create=True)` runs the whole pass once to make params). The
reference consumes the uniforms at the cursors of the REAL calls; how many
draws the prerun/creation calls eat is printed — the port must skip the same.

"Ours beside theirs": NumPy float64 vs official float32 (COMPUTE_DTYPE f32 as
A1-A3; published bf16). Toy size as rung 6 (deter 64, hidden 32, stoch 4x8,
blocks 8), batch 4, T = 64 (the ladder's length). Resets: every row at t=0;
row 1 again at t=20; row 2 at t=40; row 3 never again (in-batch control).
Near-tie rule: where the f64 reference's top-two (logit+gumbel) margin is
< 1e-4 the GOLDEN sample is fed and counted; any other disagreement is a flip.

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \
      experiments/brain_a/rung7.py --vs-official [--dump-golden]

STATE 11-21 17:10 (J16 row 2.3, slot 1): REFERENCES HOLD, --vs-official 15/15
(deter 8.5e-8, post 3.7e-7, prior 6.9e-7 x(|ref|+1) over all 64 steps, no
drift growth; 0 sample flips of 1,024 latents, 0 near-ties; reset at t=20 ==
zero-carry step, row 3 control differs; constant-noise control flips 760;
planted no-unimix / noise one call late / prior-sampled / resets-ignored each
CAUGHT). Sampler bookkeeping the port must copy: the creation pass eats 65
calls, the scoring pass 1 PRERUN call (128 uniforms, ninjax `_prerun` of step
0) before the 64 real ones. Golden `rung7_scan_golden_e01491f.npz` (sha256
dd3af760...): params, inputs, the 64 real calls' uniforms, official outputs.
Next: port stub + rung 7 tests committed FAILING, then the port.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL  # noqa: E402
import rung6  # noqa: E402
from rung6 import TOY, B, TOKDIM, CONT, DISC, CONT_DIM, DISC_N  # noqa: E402

T = 64
TOL7 = 1e-4                       # ladder row 7
TIE = 1e-4
RESETS = {0: [0, 1, 2, 3], 20: [1], 40: [2]}
GOLDEN = os.path.join(HERE, "rung7_scan_golden_e01491f.npz")
NOISE = "/data/jack-logs/mlx/rung7_noise.npz"
NOISE_N = 20_000


def close7(name, got, ref, tol=TOL7):
    got = np.float64(np.asarray(got)); ref = np.float64(ref)
    if got.shape != ref.shape:
        return False, f"{name}: shape {got.shape} != {ref.shape}"
    err = np.abs(got - ref) / (np.abs(ref) + 1.0)
    worst = float(err.max()) if err.size else 0.0
    ok = bool(np.all(np.isfinite(got)) and worst <= tol)
    return ok, f"{name}: worst {worst:.3g} x(|ref|+1) {'ok' if ok else f'OVER {tol:g}'}"


def inputs(seed=20261122):
    r = np.random.default_rng(seed)
    reset = np.zeros((B, T), bool)
    for t, rows in RESETS.items():
        reset[rows, t] = True
    return dict(tokens=r.normal(0, 1, (B, T, TOKDIM)).astype(np.float32),
                action={CONT: r.uniform(-3, 3, (B, T, CONT_DIM)).astype(np.float32),
                        DISC: r.integers(0, DISC_N, (B, T)).astype(np.int32)},
                reset=reset)


def unimix_logits(logit, unimix=TOY["unimix"]):
    z = logit - logit.max(-1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    if unimix:
        p = (1 - unimix) * p + unimix / p.shape[-1]
    return np.log(p)


def REF_sample(logit, u, golden_idx=None, unimix=TOY["unimix"]):
    """argmax(unimix-logits + Gumbel(u)); near-ties (margin < TIE) take the golden index.
    Returns (one-hot, n_ties)."""
    g = -np.log(-np.log(np.clip(u.astype(np.float64), 1e-7, 1 - 1e-7)))
    s = unimix_logits(logit, unimix) + g
    idx = s.argmax(-1)
    srt = np.sort(s, -1)
    tie = (srt[..., -1] - srt[..., -2]) < TIE
    if golden_idx is not None:
        idx = np.where(tie, golden_idx, idx)
    return np.eye(logit.shape[-1])[idx], int(tie.sum())


def REF_observe(P, I, U, golden_stoch=None, c=TOY, unimix=TOY["unimix"], use_reset=True,
                feed_prior=False):
    """64 `_observe` steps from the zero carry; U[t] = the uniforms the step-t sampler reads.
    Returns dict of [B, T, ...] arrays deter, post, prior, stoch, and the tie count."""
    carry = dict(deter=np.zeros((B, c["deter"])), stoch=np.zeros((B, c["stoch"], c["classes"])))
    out = {k: [] for k in ("deter", "post", "prior", "stoch")}
    ties = 0
    for t in range(T):
        act = {CONT: I["action"][CONT][:, t].astype(np.float64), DISC: I["action"][DISC][:, t]}
        reset = I["reset"][:, t] if use_reset else np.zeros(B, bool)
        r = rung6.REF_step(P, carry, I["tokens"][:, t].astype(np.float64), act, reset, c)
        gi = None if golden_stoch is None else golden_stoch[:, t].argmax(-1)
        logit = r["prior"] if feed_prior else r["post"]
        st, n = REF_sample(logit, U[t], gi, unimix)
        ties += n
        carry = dict(deter=r["deter"], stoch=st)
        for k, v in (("deter", r["deter"]), ("post", r["post"]), ("prior", r["prior"]), ("stoch", st)):
            out[k].append(v)
    res = {k: np.stack(v, 1) for k, v in out.items()}
    res["ties"] = ties
    return res


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
    nets.COMPUTE_DTYPE = jnp.float32            # 32-true, as A1-A3 (published default bf16)
    from dreamerv3.rssm import RSSM
    from file_noise import FileNoise
    os.makedirs(os.path.dirname(NOISE), exist_ok=True)
    noise = FileNoise(NOISE, seed=7, n=NOISE_N).install(outs)
    log = []                                     # (cursor_start, n, prerun?) per sampler call
    inner = outs.Categorical.sample

    def logged(self, seed, shape=()):
        c0 = noise.cur["uniform"]
        y = inner(self, seed, shape)
        log.append((c0, noise.cur["uniform"] - c0, bool(nj.context().ignore) or nj.creating()))
        return y
    outs.Categorical.sample = logged

    act_space = {CONT: elements.Space(np.float32, (CONT_DIM,), -1, 1),
                 DISC: elements.Space(np.int32, (), 0, DISC_N)}
    I = inputs()
    rssm = RSSM(act_space, name="rssm", **TOY)
    rows = []

    def put(ok, d):
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    def run(tokens, action, reset):
        carry = rssm.initial(B)
        carry, entries, feat = rssm.observe(carry, tokens, action, reset, False)
        return dict(deter=feat["deter"], post=feat["logit"], stoch=feat["stoch"],
                    prior=rssm._prior(feat["deter"]))

    J = lambda d: jax.tree.map(jnp.asarray, d)  # noqa: E731
    args = (jnp.asarray(I["tokens"]), J(I["action"]), jnp.asarray(I["reset"]))
    with jax.disable_jit():
        state, _ = nj.pure(run)({}, *args, create=True, seed=jax.random.PRNGKey(0))
        n_create = len(log)
        log.clear()
        c_start = noise.cur["uniform"]
        _, out = nj.pure(run)(state, *args, seed=jax.random.PRNGKey(1))
    out = {k: np.asarray(v) for k, v in out.items()}
    real = [(c, n) for c, n, pre in log if not pre]
    pre = [(c, n) for c, n, p in log if p]
    per = B * TOY["stoch"] * TOY["classes"]
    print(f"sampler calls: creation pass {n_create}; scoring pass {len(log)} = {len(pre)} prerun "
          f"(+{sum(n for _, n in pre)} uniforms) + {len(real)} real; cursor at pass start {c_start}")
    put(len(real) == T and all(n == per for _, n in real)
        and all(real[i + 1][0] == real[i][0] + per for i in range(T - 1)),
        f"real sampler calls: {len(real)} == T, each {per} uniforms, contiguous")
    U = np.stack([noise.uniform[c:c + per].reshape(B, TOY["stoch"], TOY["classes"]) for c, _ in real])
    P = {k.split("rssm/", 1)[1]: np.asarray(v, np.float64) for k, v in state.items()}
    ref = REF_observe(P, I, U, golden_stoch=out["stoch"])
    for k in ("deter", "post", "prior"):
        put(*close7(f"{k} [B,T] f32 vs f64 (64 steps)", out[k], ref[k]))
    drift = [float(np.abs(out["deter"][:, t] - ref["deter"][:, t]).max()) for t in (0, 15, 31, 63)]
    print("        deter |d| at t=0,15,31,63: " + ", ".join(f"{d:.2g}" for d in drift))
    flips = int((out["stoch"].argmax(-1) != ref["stoch"].argmax(-1)).sum())
    nlat = B * T * TOY["stoch"]
    put(flips == 0 and ref["ties"] <= nlat // 100,
        f"samples: {flips} flips of {nlat} latents; near-ties fed golden {ref['ties']} (<= 1%)")
    st = out["stoch"]
    put(bool(np.all(st.sum(-1) == 1) and np.all((st == 0) | (st == 1))), "official samples one-hot")
    # reset: row 1 at t=20 restarts from zero carry -> its step-20 deter equals the step
    # computed from a zero carry; row 3 (no reset at 20) must not (control).
    z = dict(deter=np.zeros((B, TOY["deter"])), stoch=np.zeros((B, TOY["stoch"], TOY["classes"])))
    act20 = {CONT: I["action"][CONT][:, 20].astype(np.float64), DISC: I["action"][DISC][:, 20]}
    r0 = rung6.REF_step(P, z, I["tokens"][:, 20].astype(np.float64), act20, np.ones(B, bool))
    # deter alone is weak here: from a zero carry at init (zero biases) deter' is exactly 0 in
    # both; the posterior logit reads the tokens and is not, so it is the real check.
    for k in ("deter", "post"):
        put(*close7(f"row 1 at its t=20 reset == zero-carry step ({k})", out[k][1, 20], r0[k][1]))
    pz = float(np.abs(r0["post"][1]).max())
    put(pz > 1e-2, f"zero-carry post is non-trivial (max |logit| {pz:.3g} > 1e-2)")
    d3 = float(np.abs(out["post"][3, 20] - r0["post"][3]).max())
    put(d3 > 1e-3, f"control: row 3 (no reset at 20) post differs from the zero-carry step ({d3:.3g} > 1e-3)")
    # the sampler matters: an argmax (no noise) pass must NOT match the official samples
    am = REF_observe(P, I, np.full_like(U, 0.5))
    fl = int((am["stoch"].argmax(-1) != out["stoch"].argmax(-1)).sum())
    put(fl > nlat // 10, f"control: constant-noise pass flips {fl}/{nlat} samples (> 10%)")
    # planted wrong references must each FAIL
    mutants = {
        "no unimix in the sampler": dict(unimix=0.0),
        "noise read one call late": dict(U=np.concatenate([U[1:], U[:1]])),
        "posterior sampled from the prior logit": dict(feed_prior=True),
        "resets after t=0 ignored": dict(use_reset=False),
    }
    for mname, kw in mutants.items():
        UU = kw.pop("U", U)
        m = REF_observe(P, I, UU, **kw)
        mfl = int((m["stoch"].argmax(-1) != out["stoch"].argmax(-1)).sum())
        caught = mfl > 0 or not all(close7(k, out[k], m[k])[0] for k in ("deter", "post"))
        put(caught, f"planted '{mname}': {'CAUGHT' if caught else 'MISSED'} ({mfl} flips)")
    if "--dump-golden" in sys.argv:
        np.savez(GOLDEN, **{f"param/{k}": np.asarray(state[f"rssm/{k}"], np.float32) for k in P},
                 **{f"out/{k}": v for k, v in out.items()}, uniform=U.astype(np.float32),
                 **{f"in/action_{k}": v for k, v in I["action"].items()},
                 **{"in/tokens": I["tokens"], "in/reset": I["reset"]})
        print(f"golden written: {GOLDEN}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f, toy, T {T}): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv[1:]:
        sys.exit(main_official())
    print("rung 7 port tests not written yet (next slot)")
    sys.exit(2)
