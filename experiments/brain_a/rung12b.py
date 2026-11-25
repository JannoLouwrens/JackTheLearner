"""MLX port of DreamerV3 — rung 12, second part (ORDERS J25): the policy heads' logp + entropy and
the policy loss term. REFERENCES ONLY in this commit (port tests + module are the next steps).

Official code `danijar/dreamerv3` at e01491f:
  `configs.yaml` policy {minstd 0.1, maxstd 1.0, outscale 0.01, unimix 0.01};
      policy_dist_cont bounded_normal, policy_dist_disc categorical; imag_loss actent 3e-4.
  `embodied/jax/heads.py` Head.bounded_normal: mean = tanh(m), std = (hi-lo) sigmoid(s + 2) + lo;
      Head.__call__ wraps in Agg(sum over the space's dims) when the space has a shape.
      Head.categorical: outs.Categorical(logits) — NO unimix (only `onehot` passes self.unimix),
      so the published unimix 0.01 does not touch a discrete policy. Planted below.
  `embodied/jax/outs.py` Normal.logp = norm.logpdf, entropy 0.5 log(2 pi std^2) + 0.5;
      Categorical.logp = log_softmax[event], entropy -(p log p).
  `agent.py imag_loss` L412-416: logpi = sum_k logp_k(sg(act_k))[:, :-1]; ents likewise;
      policy_loss = sg(weight[:, :-1]) * -(logpi * sg(adv_normed) + actent * sum(ents)).

What is compared: NumPy float64 vs the official Head + outs + loss expression run for real (f32;
the Linear layers tapped so the head's pre-activations are the test input), on a two-key action
space (continuous [3] + discrete 5 classes), [32, 16] steps. Outputs: logp, entropy per key, the
loss; and the GRADIENT of sum(loss) w.r.t. every pre-activation (mean, stddev, logits) — f64 central
differences vs official jax.grad (logpi and the entropy carry it; adv and weight are sg). Planted
wrong references must each be CAUGHT.
"Ours beside theirs": f64 vs f32; heads' Linear layers replaced by the given pre-activations (the
MLP trunk is rung 8's machinery); adv, weight synthetic seeded (rungs 11, 12a pin them).
Tolerance as rung 11: |d|/(|ref|+1) <= 1e-4.

  nice -n 19 /data/venvs/jack-dv3/bin/python -I experiments/brain_a/rung12b.py --vs-official
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL  # noqa: E402

MINSTD, MAXSTD, UNIMIX, ACTENT = 0.1, 1.0, 0.01, 3e-4
N, T1, A, K = 32, 16, 3, 5
TOL = 1e-4
GOLDEN12B = os.path.join(HERE, "rung12b_policy_golden_e01491f.npz")


def inputs(seed=121):
    r = np.random.default_rng(seed)
    f = np.float32
    return dict(
        m=f(1.5 * r.standard_normal((N, T1, A))),
        s=f(2.0 * r.standard_normal((N, T1, A))),
        logits=f(2.0 * r.standard_normal((N, T1, K))),
        act_c=f(np.clip(r.standard_normal((N, T1, A)), -1, 1)),
        act_d=r.integers(0, K, (N, T1)).astype(np.int32),
        adv=f(r.standard_normal((N, T1 - 1))),
        weight=f(np.cumprod(r.uniform(0.9, 1.0, (N, T1)), 1)),
    )


def _lse(x):
    mx = x.max(-1, keepdims=True)
    return mx + np.log(np.exp(x - mx).sum(-1, keepdims=True))


def REF_policy(inp, minstd=MINSTD, maxstd=MAXSTD, actent=ACTENT, cat_unimix=0.0, std_shift=2.0,
               no_tanh=False, ent_sign=1.0, mean_not_sum=False, keep_last=False):
    """Float64 policy heads + loss -> dict(logp_c, ent_c, logp_d, ent_d, loss)."""
    m, s, lg = (np.float64(inp[k]) for k in ("m", "s", "logits"))
    mean = m if no_tanh else np.tanh(m)
    std = (maxstd - minstd) / (1 + np.exp(-(s + std_shift))) + minstd
    x = np.float64(inp["act_c"])
    lpc = -0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
    enc = 0.5 * np.log(2 * np.pi * std ** 2) + 0.5
    agg = (lambda v: v.mean(-1)) if mean_not_sum else (lambda v: v.sum(-1))
    lpc, enc = agg(lpc), agg(enc)
    if cat_unimix:
        p = np.exp(lg - _lse(lg))
        lg = np.log((1 - cat_unimix) * p + cat_unimix / K)
    lsm = lg - _lse(lg)
    lpd = np.take_along_axis(lsm, np.int64(inp["act_d"])[..., None], -1)[..., 0]
    end = -(np.exp(lsm) * lsm).sum(-1)
    sl = slice(None, -1)
    if keep_last:
        sl = slice(1, None)
    logpi = lpc[:, sl] + lpd[:, sl]
    ent = enc[:, sl] + end[:, sl]
    w = np.float64(inp["weight"])[:, :-1]
    loss = w * -(logpi * np.float64(inp["adv"]) + actent * ent_sign * ent)
    return dict(logp_c=lpc, ent_c=enc, logp_d=lpd, ent_d=end, loss=loss)


def REF_grads(inp, h=1e-6, **kw):
    """Central differences of sum(loss) in f64 w.r.t. m, s, logits. Each element enters one loss
    term only, so one vectorised +-h perturbation of the whole array gives every partial."""
    base = {k: np.float64(v) if k in ("m", "s", "logits") else v for k, v in inp.items()}
    out = {}
    for k in ("m", "s", "logits"):
        g = np.zeros_like(base[k])
        for j in range(base[k].shape[-1]):
            up, dn = dict(base), dict(base)
            e = np.zeros_like(base[k]); e[..., j] = h
            up[k], dn[k] = base[k] + e, base[k] - e
            lu = REF_policy(up, **kw)["loss"]; ld = REF_policy(dn, **kw)["loss"]
            # element [n, t, j] feeds loss[n, t] (t < T1-1 for the default slicing)
            d = np.zeros(base[k].shape[:2])
            d[:, :-1] = (lu - ld) / (2 * h)
            g[..., j] = d
        out[k] = g
    return out


PLANTED = {
    "unimix 0.01 applied to the categorical policy": dict(cat_unimix=0.01),
    "minstd 0 (std unbounded below)": dict(minstd=0.0),
    "no +2 shift inside the stddev sigmoid": dict(std_shift=0.0),
    "mean without tanh": dict(no_tanh=True),
    "entropy subtracted (sign flipped)": dict(ent_sign=-1.0),
    "actent 3e-3": dict(actent=3e-3),
    "logp/ent sliced [1:] not [:-1]": dict(keep_last=True),
    "continuous dims averaged not summed": dict(mean_not_sum=True),
}


def worst(got, ref):
    w = 0.0
    for k, r in ref.items():
        g, r = np.float64(got[k]), np.float64(r)
        w = max(w, float((np.abs(g - r) / (np.abs(r) + 1)).max()))
    return w


def main_official():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import elements
    from embodied.jax import heads, nets
    nets.COMPUTE_DTYPE = jnp.float32  # as rung 11: f32 compute, the port's dtype
    inp = inputs()
    sg = jax.lax.stop_gradient

    PRE = {}
    # Tap the heads' Linear layers: Head.sub returns the given pre-activation (patched on Head
    # itself — a subclass would need its ninjax fields re-declared). Head's own maths runs.
    heads.Head.sub = lambda self, name, make=None, *a, **k: (lambda x: PRE[name])
    Tap = heads.Head

    def fwd(m, s, lg):
        PRE.clear(); PRE.update(mean=m, stddev=s)
        kw = dict(minstd=MINSTD, maxstd=MAXSTD, unimix=UNIMIX, outscale=0.01)
        pc = Tap(elements.Space(np.float32, (A,), -1.0, 1.0), "bounded_normal", **kw, name="pc")(
            jnp.zeros((N, T1, 1), jnp.float32))
        PRE.clear(); PRE.update(logits=lg)
        pd = Tap(elements.Space(np.int32, (), 0, K), "categorical", **kw, name="pd")(
            jnp.zeros((N, T1, 1), jnp.float32))
        policy = {"c": pc, "d": pd}
        act = {"c": jnp.asarray(inp["act_c"]), "d": jnp.asarray(inp["act_d"])}
        logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
        ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
        loss = sg(jnp.asarray(inp["weight"])[:, :-1]) * -(
            logpi * sg(jnp.asarray(inp["adv"])) + ACTENT * sum(ents.values()))
        return loss, dict(logp_c=pc.logp(act["c"]), ent_c=pc.entropy(),
                          logp_d=pd.logp(act["d"]), ent_d=pd.entropy(), loss=loss)

    def total(m, s, lg):
        return fwd(m, s, lg)[0].sum()

    args = tuple(jnp.asarray(inp[k]) for k in ("m", "s", "logits"))
    with jax.disable_jit():
        _, out = nj.pure(lambda *a: fwd(*a)[1])({}, *args, create=True, seed=jax.random.PRNGKey(0))
        _, gr = nj.pure(lambda *a: jax.grad(total, argnums=(0, 1, 2))(*a))(
            {}, *args, create=True, seed=jax.random.PRNGKey(0))
    got = {k: np.asarray(v) for k, v in out.items()}
    ggot = dict(zip(("m", "s", "logits"), (np.asarray(g) for g in gr)))
    rows = []

    def put(ok, d):
        rows.append(bool(ok)); print(("  ok  " if ok else "  XX  ") + d)
    ref = REF_policy(inp)
    put(got["logp_c"].shape == (N, T1) and got["logp_d"].shape == (N, T1),
        f"shapes logp_c {got['logp_c'].shape} (Agg summed the 3 dims), logp_d {got['logp_d'].shape}")
    w = worst(got, ref)
    put(w <= TOL, f"logp, entropy (both keys) + loss, f32 vs f64 worst {w:.3g} x(|ref|+1)")
    gref = REF_grads(inp)
    gw = worst(ggot, gref)
    gmax = max(float(np.abs(v).max()) for v in gref.values())
    put(gw <= TOL, f"grad sum(loss) wrt mean/stddev/logits pre-acts: jax.grad vs f64 diffs worst "
        f"{gw:.3g} (max|g| {gmax:.3g})")
    put(all(float(np.abs(ggot[k][:, -1]).max()) == 0.0 for k in ggot),
        "last step carries no gradient (sliced [:, :-1])")
    for name, kw in PLANTED.items():
        c = worst(got, REF_policy(inp, **kw)) > TOL
        put(c, f"planted '{name}': {'CAUGHT' if c else 'MISSED'}")
    if "--dump-golden" in sys.argv:
        np.savez_compressed(GOLDEN12B, **{f"in/{k}": v for k, v in inp.items()},
                            **{f"out/{k}": v for k, v in got.items()},
                            **{f"grad/{k}": v for k, v in ggot.items()})
        print("        dumped", GOLDEN12B)
    n = len(rows); p = sum(rows)
    print(f"rung12b references vs official e01491f: {p}/{n} {'HOLD' if p == n else 'BROKEN'}")
    return 0 if p == n else 1


# ---------------------------------------------------------------- port tests (MLX)
# `mlx_maths.policy_loss(m, s, logits, act_c, act_d, adv, weight, minstd, maxstd, actent)`
#   -> (loss [N, T1-1], dict(logp_c, ent_c, logp_d, ent_d) [N, T1]): bounded_normal (summed over
#   the action dims) + categorical (no unimix) heads from their pre-activations, and imag_loss's
#   policy term. Each test eager AND under mx.compile:
#   policy_ref     forward vs REF_policy (float64)
#   policy_golden  forward vs the official f32 dump
#   policy_grad    mx.grad of sum(loss) wrt (m, s, logits) vs official jax.grad AND f64 diffs


def _fwd(M, mx, compiled, inp):
    def f(m, s, lg):
        loss, d = M.policy_loss(m, s, lg, mx.array(inp["act_c"]), mx.array(inp["act_d"]),
                                mx.array(inp["adv"]), mx.array(inp["weight"]), MINSTD, MAXSTD, ACTENT)
        return loss, d["logp_c"], d["ent_c"], d["logp_d"], d["ent_d"]
    f = mx.compile(f) if compiled else f
    o = f(*(mx.array(inp[k]) for k in ("m", "s", "logits")))
    return dict(zip(("loss", "logp_c", "ent_c", "logp_d", "ent_d"), (np.asarray(v) for v in o)))


def _grad(M, mx, compiled, inp):
    def tot(m, s, lg):
        return M.policy_loss(m, s, lg, mx.array(inp["act_c"]), mx.array(inp["act_d"]),
                             mx.array(inp["adv"]), mx.array(inp["weight"]), MINSTD, MAXSTD,
                             ACTENT)[0].sum()
    g = mx.grad(tot, argnums=(0, 1, 2))
    g = mx.compile(g) if compiled else g
    o = g(*(mx.array(inp[k]) for k in ("m", "s", "logits")))
    return dict(zip(("m", "s", "logits"), (np.asarray(v) for v in o)))


def _golden():
    G = np.load(GOLDEN12B)
    inp = {k[3:]: G[k] for k in G.files if k.startswith("in/")}
    return inp, {k[4:]: G[k] for k in G.files if k.startswith("out/")}, \
        {k[5:]: G[k] for k in G.files if k.startswith("grad/")}


def t_ref(M, mx, compiled):
    inp = inputs()
    w = worst(_fwd(M, mx, compiled, inp), REF_policy(inp))
    return [(w <= TOL, f"logp/ent/loss vs REF: worst {w:.3g} x(|ref|+1)")]


def t_golden(M, mx, compiled):
    inp, out, _ = _golden()
    w = worst(_fwd(M, mx, compiled, inp), out)
    return [(w <= TOL, f"logp/ent/loss vs GOLDEN: worst {w:.3g} x(|ref|+1)")]


def t_grad(M, mx, compiled):
    inp, _, gg = _golden()
    got = _grad(M, mx, compiled, inp)
    w1 = worst(got, gg)
    w2 = worst(got, REF_grads(inp))
    return [(w1 <= TOL, f"grads vs official jax.grad: worst {w1:.3g}"),
            (w2 <= TOL, f"grads vs f64 differences: worst {w2:.3g}")]


TESTS = [("policy_ref", t_ref), ("policy_golden", t_golden), ("policy_grad", t_grad)]


def run_tests(M, mx, tests=TESTS, quiet=False):
    n = 0
    for name, t in tests:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c)
                r[mode] = all(o for o, _ in rows)
                det = [d for _, d in rows]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                r[mode], det = False, [f"{type(e).__name__}: {e}"[:200]]
            if not quiet and (mode == "eager" or not r[mode]):
                for d in det:
                    print(f"    {mode[0]}:", d)
        n += r["eager"] and r["compiled"]
        if not quiet:
            print(f"{name:15s} eager {'PASS' if r['eager'] else 'FAIL'}  compiled {'PASS' if r['compiled'] else 'FAIL'}")
    return n


def main_mlx():
    import rung6
    mx = rung6._mlx()
    import mlx_maths as M
    n = run_tests(M, mx)
    print(f"RUNG 12b: {n}/{len(TESTS)} tests pass eager AND compiled (mlx {mx.__version__}) -> "
          f"{'PASS' if n == len(TESTS) else 'FAIL'}")
    return 0 if n == len(TESTS) else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv:
        sys.exit(main_official())
    sys.exit(main_mlx())
