"""MLX port of DreamerV3 — rung 6 references (PORT report step 2.2): one recurrent step.

Source: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md` step table row
2.2: "One recurrent step: core, posterior, prior, reset masking — Rung 6
passes". Official code `danijar/dreamerv3` at e01491f
(`/data/src/dreamerv3_e01491f`), `dreamerv3/rssm.py`:

  _observe  L72-90: deter, stoch, action masked to 0 where `reset`
            (nets.mask = where(~reset, x, 0), reset broadcast over trailing
            dims); action = DictConcat(act_space, 1) (sorted keys; discrete ->
            one_hot(classes), continuous -> as is) then masked again;
            deter = _core(...); x = concat([deter, tokens]); obslayers x
            (Linear hidden -> Norm rms -> act); logit = obslogit Linear
            (stoch*classes, outscale) reshaped [stoch, classes]; stoch =
            OneHot(logit, unimix).sample — NOT in this rung (rung 7's sampler).
  _core     L139-162: stoch flattened; action /= max(1, |action|) (stop-grad);
            x0/x1/x2 = Linear(hidden)->Norm->act of deter/stoch/action;
            x = concat(x0,x1,x2) repeated over g=blocks groups, each group
            prefixed with its slice of deter, flattened; dynlayers x
            (BlockLinear(deter, g) -> Norm -> act); dyngru BlockLinear(3*deter,
            g); gates split PER GROUP into reset/cand/update; reset=sigmoid,
            cand=tanh(reset*cand), update=sigmoid(update-1);
            deter' = update*cand + (1-update)*deter.
  _prior    L164-169: imglayers x (Linear hidden -> Norm -> act) -> priorlogit.
  nets      Linear: x @ kernel + bias. BlockLinear: kernel [g, in/g, out/g],
            einsum per group + bias. Norm 'rms': x * rsqrt(mean(x^2) + 1e-4)
            * scale (computed in float32). act 'gelu' = jax.nn.gelu, whose
            default is approximate=True (the tanh form) — the reference uses
            the tanh form, not erf.

"Ours beside theirs": the reference is NumPy float64; the official run is
float32 (nets.COMPUTE_DTYPE set to float32, as A1-A3 ran `32-true`; the
published default is bfloat16 — a listed difference). Sizes are toy (deter
64, hidden 32, stoch 4x8, blocks 8, dynlayers 1, obslayers 1, imglayers 2)
so the check runs in seconds; the maths is the same code path as size1m.
Actions: one continuous key (3 dims, values beyond +-1 so the max(1,|a|)
squash binds) and one discrete key (4 classes) -> both DictConcat branches.
Reset: rows 0 and 2 of a batch of 4 reset (rung 6's "reset masking");
rows 1 and 3 carry a non-zero state — the in-batch control.

Tolerance as rung 1: |official - ref| <= 1e-6 * (|ref| + 1)... except that
float32 vs float64 through ~6 layers of norm/gelu drifts above 1e-6; the
tolerance used here is stated per check below and is NOT the port's: the
port is read against the official float32 GOLDEN this mode dumps, as rung 3.

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \
      experiments/brain_a/rung6.py --vs-official [--dump-golden]
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung6.py

STATE 11-21 16:45 (J16 row 2.2, slot 1): REFERENCES HOLD, --vs-official
12/12 (9 agreement checks incl. the reset-row and in-batch control, 3
planted wrong references each CAUGHT: erf gelu, flat GRU gate split, no
action squash). A first draft of the squash mutant was itself a no-op
(a*max(1,|a|) re-squashes to sign(a)) and was MISSED — fixed to skip the
division. Golden: `rung6_step_golden_e01491f.npz` (official f32 jit outputs
+ params + inputs). Next: port stub in mlx_maths + tests committed FAILING.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL, close  # noqa: E402

TOY = dict(deter=64, hidden=32, stoch=4, classes=8, blocks=8, dynlayers=1, obslayers=1,
           imglayers=2, norm="rms", act="gelu", unimix=0.01, outscale=1.0)
B = 4
TOKDIM = 12
CONT, DISC = "a_move", "b_pick"       # sorted order: continuous first, then discrete
CONT_DIM, DISC_N = 3, 4
RESET = np.array([True, False, True, False])
EPS = 1e-4
GOLDEN = os.path.join(HERE, "rung6_step_golden_e01491f.npz")


# ---------------------------------------------------------------- references (float64)
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rmsnorm(x, scale):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + EPS) * scale


def lin(P, name, x):
    return x @ P[f"{name}/kernel"] + P[f"{name}/bias"]


def blocklin(P, name, x, g):
    k = P[f"{name}/kernel"]                       # [g, in/g, out/g]
    xg = x.reshape(*x.shape[:-1], g, x.shape[-1] // g)
    y = np.einsum("...ki,kio->...ko", xg, k).reshape(*x.shape[:-1], -1)
    return y + P[f"{name}/bias"]


def layer(P, name, x):
    return gelu(rmsnorm(lin(P, name, x), P[f"{name}norm/scale"]))


def REF_core(P, deter, stoch, action, c=TOY, squash=True):
    g = c["blocks"]
    stoch = stoch.reshape(stoch.shape[0], -1)
    if squash:                                    # False only for the planted mutant
        action = action / np.maximum(1.0, np.abs(action))
    x = np.concatenate([layer(P, "dynin0", deter), layer(P, "dynin1", stoch),
                        layer(P, "dynin2", action)], -1)
    x = np.repeat(x[..., None, :], g, -2)
    dg = deter.reshape(*deter.shape[:-1], g, -1)
    x = np.concatenate([dg, x], -1).reshape(*deter.shape[:-1], -1)
    for i in range(c["dynlayers"]):
        x = gelu(rmsnorm(blocklin(P, f"dynhid{i}", x, g), P[f"dynhid{i}norm/scale"]))
    x = blocklin(P, "dyngru", x, g)
    xg = x.reshape(*x.shape[:-1], g, -1)
    r, cand, u = [a.reshape(*x.shape[:-1], -1) for a in np.split(xg, 3, -1)]
    r = sigmoid(r)
    cand = np.tanh(r * cand)
    u = sigmoid(u - 1.0)
    return u * cand + (1.0 - u) * deter


def REF_actemb(action):
    """DictConcat: sorted keys; continuous as is; discrete -> one-hot."""
    return np.concatenate([action[CONT], np.eye(DISC_N)[action[DISC]]], -1)


def REF_prior(P, deter, c=TOY):
    x = deter
    for i in range(c["imglayers"]):
        x = layer(P, f"prior{i}", x)
    return lin(P, "priorlogit", x).reshape(*x.shape[:-1], c["stoch"], c["classes"])


def REF_step(P, carry, tokens, action, reset, c=TOY):
    """One `_observe` step up to the posterior logit, plus `_prior(deter')`.
    Returns dict(masked_deter, actemb, deter, post, prior)."""
    keep = (~reset).astype(np.float64)
    deter = carry["deter"] * keep[:, None]
    stoch = carry["stoch"] * keep[:, None, None]
    a = REF_actemb(action) * keep[:, None]
    deter2 = REF_core(P, deter, stoch, a, c)
    x = np.concatenate([deter2, tokens], -1)
    for i in range(c["obslayers"]):
        x = layer(P, f"obs{i}", x)
    post = lin(P, "obslogit", x).reshape(B, c["stoch"], c["classes"])
    return dict(masked_deter=deter, actemb=a, deter=deter2, post=post, prior=REF_prior(P, deter2, c))


def inputs(seed=20261121):
    r = np.random.default_rng(seed)
    st = np.zeros((B, TOY["stoch"], TOY["classes"]))
    st[np.arange(B)[:, None], np.arange(TOY["stoch"]), r.integers(0, TOY["classes"], (B, TOY["stoch"]))] = 1
    return dict(
        carry=dict(deter=np.tanh(r.normal(0, 1, (B, TOY["deter"]))).astype(np.float32),
                   stoch=st.astype(np.float32)),
        tokens=r.normal(0, 1, (B, TOKDIM)).astype(np.float32),
        action={CONT: r.uniform(-3, 3, (B, CONT_DIM)).astype(np.float32),
                DISC: r.integers(0, DISC_N, (B,)).astype(np.int32)},
        reset=RESET.copy())


# ---------------------------------------------------------------- official
def main_official():
    """REF_step against the official RSSM._observe / _prior at e01491f (float32 CPU, jit)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax.nets as nets
    nets.COMPUTE_DTYPE = jnp.float32            # 32-true, as A1-A3 (published default bf16)
    from dreamerv3.rssm import RSSM
    act_space = {CONT: elements.Space(np.float32, (CONT_DIM,), -1, 1),
                 DISC: elements.Space(np.int32, (), 0, DISC_N)}
    I = inputs()
    rssm = RSSM(act_space, name="rssm", **TOY)
    rows = []

    def chk(name, got, ref, tol):
        ok, d = close(name, np.asarray(got, np.float64), ref, tol)
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    def step(carry, tokens, action, reset):
        keep = ~reset
        deter, stoch, act = nets.mask((carry["deter"], carry["stoch"], action), keep)
        actemb = nets.mask(nets.DictConcat(act_space, 1)(act), keep)
        carry2, (entry, feat) = rssm._observe(carry, tokens, action, reset, False)
        prior = rssm._prior(feat["deter"])
        return dict(masked_deter=deter, actemb=actemb, deter=feat["deter"], post=feat["logit"],
                    prior=prior, stoch=feat["stoch"])

    J = lambda d: jax.tree.map(jnp.asarray, d)  # noqa: E731
    args = (J(I["carry"]), jnp.asarray(I["tokens"]), J(I["action"]), jnp.asarray(I["reset"]))
    state, _ = nj.pure(step)({}, *args, create=True, seed=jax.random.PRNGKey(0))
    fn = jax.jit(lambda s, *a: nj.pure(step)(s, *a, seed=jax.random.PRNGKey(1)))
    _, out = fn(state, *args)
    P = {k.split("rssm/", 1)[1]: np.asarray(v, np.float64) for k, v in state.items()}
    print(f"official params: {len(P)} arrays: {sorted(P)}")
    ref = REF_step(P, {k: v.astype(np.float64) for k, v in I["carry"].items()},
                   I["tokens"].astype(np.float64),
                   {CONT: I["action"][CONT].astype(np.float64), DISC: I["action"][DISC]}, I["reset"])
    chk("masked deter (exact)", out["masked_deter"], ref["masked_deter"], 0.0)
    chk("action embedding (exact)", out["actemb"], ref["actemb"], 0.0)
    for k in ("deter", "post", "prior"):
        chk(f"{k} (jit f32 vs f64)", out[k], ref[k], 1.0)
    # reset rows: a reset row's step must not depend on its carry — the official
    # output for rows 0/2 is equal to the reference run from a ZERO carry.
    z = {k: np.zeros_like(v, np.float64) for k, v in I["carry"].items()}
    ref0 = REF_step(P, z, I["tokens"].astype(np.float64),
                    {CONT: I["action"][CONT].astype(np.float64), DISC: I["action"][DISC]},
                    np.ones(B, bool))
    for k in ("deter", "post"):
        chk(f"reset rows {k} == from-zero carry", np.asarray(out[k])[RESET], ref0[k][RESET], 1.0)
    # in-batch control: a kept row must NOT equal its from-zero version (else the
    # mask check above could not fail)
    d = float(np.abs(np.asarray(out["deter"])[~RESET] - ref0["deter"][~RESET]).max())
    ok = d > 1e-3
    rows.append((ok, f"control: kept rows differ from the zero-carry step (max |d| {d:.3g} > 1e-3)"))
    print(("  ok  " if ok else "  XX  ") + rows[-1][1])
    st = np.asarray(out["stoch"])
    ok = bool(np.all(st.sum(-1) == 1) and np.all((st == 0) | (st == 1)))
    rows.append((ok, "posterior sample is one-hot per latent (values NOT compared: rung 7's sampler)"))
    print(("  ok  " if ok else "  XX  ") + rows[-1][1])
    # planted: three wrong references must each FAIL against the official output,
    # else the agreement above could not have caught them.
    from math import erf
    g = globals()
    keep_gelu, keep_core = g["gelu"], g["REF_core"]

    def flat_gates(P, deter, stoch, action, c=TOY):   # gates split flat, not per group
        gg = c["blocks"]
        st = stoch.reshape(stoch.shape[0], -1)
        a = action / np.maximum(1.0, np.abs(action))
        x = np.concatenate([layer(P, "dynin0", deter), layer(P, "dynin1", st), layer(P, "dynin2", a)], -1)
        x = np.repeat(x[..., None, :], gg, -2)
        x = np.concatenate([deter.reshape(*deter.shape[:-1], gg, -1), x], -1).reshape(*deter.shape[:-1], -1)
        for i in range(c["dynlayers"]):
            x = gelu(rmsnorm(blocklin(P, f"dynhid{i}", x, gg), P[f"dynhid{i}norm/scale"]))
        r, cand, u = np.split(blocklin(P, "dyngru", x, gg), 3, -1)
        u = sigmoid(u - 1.0)
        return u * np.tanh(sigmoid(r) * cand) + (1.0 - u) * deter
    mutants = {
        "gelu as erf (exact) form": dict(gelu=np.vectorize(lambda v: 0.5 * v * (1 + erf(v / np.sqrt(2))))),
        "GRU gates split flat, not per block": dict(REF_core=flat_gates),
        "no action squash max(1,|a|)": dict(REF_core=lambda P, d, s, a, c=TOY: keep_core(P, d, s, a, c, squash=False)),
    }
    for mname, patch in mutants.items():
        g.update(patch)
        try:
            mref = REF_step(P, {k: v.astype(np.float64) for k, v in I["carry"].items()},
                            I["tokens"].astype(np.float64),
                            {CONT: I["action"][CONT].astype(np.float64), DISC: I["action"][DISC]}, I["reset"])
            caught = not all(close(k, out[k], mref[k], 1.0)[0] for k in ("deter", "post", "prior"))
        finally:
            g["gelu"], g["REF_core"] = keep_gelu, keep_core
        rows.append((caught, f"planted '{mname}': {'CAUGHT' if caught else 'MISSED'}"))
        print(("  ok  " if caught else "  XX  ") + rows[-1][1])
    if "--dump-golden" in sys.argv:
        np.savez(GOLDEN, **{f"param/{k}": np.asarray(state[f"rssm/{k}"], np.float32) for k in P},
                 **{f"out/{k}": np.asarray(v) for k, v in out.items()},
                 **{f"in/carry_{k}": v for k, v in I["carry"].items()},
                 **{f"in/action_{k}": v for k, v in I["action"].items()},
                 **{"in/tokens": I["tokens"], "in/reset": I["reset"]})
        print(f"golden written: {GOLDEN}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f, toy {TOY}): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


# ---------------------------------------------------------------- port tests (MLX)
# Two tests, each run eager AND under mx.compile:
#   step_ref    port vs REF_step (float64) on the golden's params and inputs, 1e-6 x(|ref|+1)
#   step_golden port vs the official f32 jit output dumped in the golden, same tolerance
# plus, inside each: reset rows equal the port run from a zero carry, and kept
# rows differ from it (the control that makes the mask check able to fail).
ACT_SPEC = {CONT: 0, DISC: DISC_N}


def _load_golden():
    G = np.load(GOLDEN)
    P = {k[len("param/"):]: G[k] for k in G.files if k.startswith("param/")}
    I = dict(carry=dict(deter=G["in/carry_deter"], stoch=G["in/carry_stoch"]),
             tokens=G["in/tokens"], reset=G["in/reset"].astype(bool),
             action={CONT: G[f"in/action_{CONT}"], DISC: G[f"in/action_{DISC}"]})
    out = {k[len("out/"):]: G[k] for k in G.files if k.startswith("out/")}
    return P, I, out


def _port_run(M, mx, P, I, compiled, reset=None):
    reset = I["reset"] if reset is None else reset
    MP = {k: mx.array(v) for k, v in P.items()}
    args = (dict(deter=mx.array(I["carry"]["deter"]), stoch=mx.array(I["carry"]["stoch"])),
            mx.array(I["tokens"]), {k: mx.array(v) for k, v in I["action"].items()},
            mx.array(reset))
    f = lambda p, c, t, a, r: M.rssm_step(p, c, t, a, r, TOY, ACT_SPEC)  # noqa: E731
    if compiled:
        f = mx.compile(f)
    out = f(MP, *args)
    return {k: np.asarray(v) for k, v in out.items()}


def t_step(M, mx, compiled, against):
    P, I, gold = _load_golden()
    got = _port_run(M, mx, P, I, compiled)
    if against == "ref":
        P64 = {k: v.astype(np.float64) for k, v in P.items()}
        tgt = REF_step(P64, {k: v.astype(np.float64) for k, v in I["carry"].items()},
                       I["tokens"].astype(np.float64),
                       {CONT: I["action"][CONT].astype(np.float64), DISC: I["action"][DISC]}, I["reset"])
    else:
        tgt = {k: gold[k] for k in ("masked_deter", "actemb", "deter", "post", "prior")}
    rows = [close(f"{k} vs {against.upper()}", got[k], tgt[k], 1.0) for k in
            ("masked_deter", "actemb", "deter", "post", "prior")]
    z = dict(I, carry={k: np.zeros_like(v) for k, v in I["carry"].items()})
    got0 = _port_run(M, mx, P, z, compiled, reset=np.ones(B, bool))
    R = I["reset"]
    rows += [close(f"reset rows {k} == zero-carry port step", got[k][R], got0[k][R], 1.0)
             for k in ("deter", "post")]
    d = float(np.abs(got["deter"][~R] - got0["deter"][~R]).max())
    rows.append((d > 1e-3, f"control: kept rows differ from the zero-carry step (max |d| {d:.3g} > 1e-3)"))
    return rows


TESTS = [("step_ref", lambda M, mx, c: t_step(M, mx, c, "ref")),
         ("step_golden", lambda M, mx, c: t_step(M, mx, c, "golden"))]


def _mlx():
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"   # mlx_smoke.py: mx.compile needs it
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    import mlx.core as mx
    return mx


def main_mlx():
    mx = _mlx()
    import mlx_maths as M
    n = 0
    for name, t in TESTS:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c)
                r[mode] = all(o for o, _ in rows)
                det = [d for o, d in rows if not o][:4] or ["all rows ok"]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                r[mode], det = False, [f"{type(e).__name__}: {e}"[:200]]
            if mode == "eager" or not r[mode]:
                for d in det:
                    print(f"    {mode[0]}:", d)
        n += r["eager"] and r["compiled"]
        print(f"{name:12s} eager {'PASS' if r['eager'] else 'FAIL'}  compiled {'PASS' if r['compiled'] else 'FAIL'}")
    print(f"RUNG 6: {n}/{len(TESTS)} tests pass eager AND compiled (mlx {mx.__version__}) -> "
          f"{'PASS' if n == len(TESTS) else 'FAIL'}")
    return 0 if n == len(TESTS) else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv[1:]:
        sys.exit(main_official())
    sys.exit(main_mlx())
