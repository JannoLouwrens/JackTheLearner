"""MLX port of DreamerV3 — rung 1 tests (PORT report step 1.2), pure maths.

Source: `docs/research/PUBLISHED_EVIDENCE_2026-11-15_PORT.md`, parity ladder
rung 1: "Pure maths: squashing, bins, two-hot, 1% mix, KL with free bits,
sampling gradient, normal and coin-flip losses — 1e-6; uniform and mirrored
decode exactly 0, also compiled — no golden file (NumPy reference)". Step 1.2's
done-test: "tests exist and fail" (the module, `mlx_maths.py`, is a stub until
step 1.3). Contract lines: `dreamer_port_spec.md` §(2) items 11, 21, 22, 24,
27–30, 32 (official `danijar/dreamerv3` at e01491f, `/data/src/dreamerv3_e01491f`).

Three modes, three interpreters:

  # 1. the references ARE the official code (JAX 0.4.33 CPU, float32):
  nice -n 19 /data/venvs/jack-dv3/bin/python -I experiments/brain_a/rung1.py --vs-official
  # 2. the port against the references, eager and mx.compile'd (exit 0 = rung 1 PASS):
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung1.py
  # 3. known-positives: each planted wrong implementation must FAIL its test:
  taskset -c 0 nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung1.py --planted

What "1e-6" means here (the report does not say relative to what; rung 10's
lesson, `port_gate.py`, is that a global norm hides wrong small terms): each
element must satisfy |port - ref| <= 1e-6 * (|ref| + s), with the floor `s`
stated per test — s = 1e-30 (pure relative) where a small value is the point
(symlog/symexp near 0: catches log(1+x) for log1p), s = 1 for log-densities,
losses, KLs and weights that are sums of O(1) terms, and for the two-hot decode
the scale is the condition of the sum, sum_i |p_i * bin_i|, because the bins
reach +-4.85e8 and the value is their cancellation. References are NumPy
float64 computed from the SAME float32 inputs the port sees. Gradients are
checked at the same tolerance against analytic float64 gradients.
The exact-zero clauses use `==`, not a tolerance.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = "/data/src/dreamerv3_e01491f"
TOL = 1e-6

# ---------------------------------------------------------------- references
# NumPy float64, written from the official lines cited; `--vs-official` checks
# every one of them against the official JAX function on the same inputs.


def REF_symlog(x):
    x = np.float64(x)
    return np.sign(x) * np.log1p(np.abs(x))


def REF_symexp(x):
    x = np.float64(x)
    return np.sign(x) * np.expm1(np.abs(x))


def OFFICIAL_BINS():
    """The official float32 bins, bit for bit (`rung1_bins_e01491f.json`,
    written by the official heads.py L136–139 on JAX 0.4.33 CPU). The bins are
    built in float32 there (`linspace(..., dtype=f32)`), so they sit up to
    1.24e-6 (relative) off the float64 values `REF_bins` gives — measured by
    `--vs-official` on 11-19. Parity is with the official code, so rung 1's
    bins reference is THIS array; `REF_bins` stays as the float64 contract the
    official array must lie within 2e-6 of (checked)."""
    with open(os.path.join(HERE, "rung1_bins_e01491f.json")) as f:
        h = json.load(f)["uint32_hex"]
    return np.array([int(x, 16) for x in h], np.uint32).view(np.float32)


def REF_bins(n=255):
    if n % 2 == 1:
        half = REF_symexp(np.linspace(-20, 0, (n - 1) // 2 + 1))
        return np.concatenate([half, -half[:-1][::-1]])
    half = REF_symexp(np.linspace(-20, 0, n // 2))
    return np.concatenate([half, -half[::-1]])


def REF_twohot_target(target, bins):
    t = np.float64(target)[..., None]
    b = np.float64(bins)
    n = len(b)
    below = np.clip((b <= t).sum(-1) - 1, 0, n - 1)
    above = np.clip(n - (b > t).sum(-1), 0, n - 1)
    t = t[..., 0]
    equal = below == above
    db = np.where(equal, 1.0, np.abs(b[below] - t))
    da = np.where(equal, 1.0, np.abs(b[above] - t))
    tot = db + da
    out = np.zeros(t.shape + (n,))
    np.put_along_axis(out, below[..., None], (da / tot)[..., None], -1)
    # below == above: both weights land on the same bin and sum (one_hot + one_hot)
    prev = np.take_along_axis(out, above[..., None], -1)
    np.put_along_axis(out, above[..., None], prev + (db / tot)[..., None], -1)
    return out


def _log_softmax(z):
    z = np.float64(z)
    m = z.max(-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))


def REF_twohot_loss(logits, target, bins):
    return -(REF_twohot_target(target, bins) * _log_softmax(logits)).sum(-1)


def REF_twohot_decode(logits, bins):
    p = np.exp(_log_softmax(logits))
    return (p * np.float64(bins)).sum(-1), (np.abs(p * np.float64(bins))).sum(-1)


def REF_unimix_logits(logits, mix):
    p = np.exp(_log_softmax(logits))
    return np.log((1 - mix) * p + mix / p.shape[-1])


def REF_kl(post, prior, mix, free):
    lp = _log_softmax(REF_unimix_logits(post, mix))
    lq = _log_softmax(REF_unimix_logits(prior, mix))
    kl = (np.exp(lp) * (lp - lq)).sum(-1).sum(-1)
    return np.maximum(kl, free), kl


def _num_grad(f, x, h=1e-6):
    """Central differences in float64 (error ~h^2 * f''' ~ 1e-11 here)."""
    x = np.float64(x).copy()
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def REF_kl_grads(post, prior, mix, free):
    """d sum(dyn)/d prior and d sum(rep)/d post (the two non-zero ones).
    dyn = KL[sg(post) || prior], rep = KL[post || sg(prior)] — same value,
    opposite stop-gradients (rssm.py L123–124)."""
    d_prior = _num_grad(lambda q: REF_kl(post, q, mix, free)[0].sum(), prior)
    d_post = _num_grad(lambda p: REF_kl(p, prior, mix, free)[0].sum(), post)
    return d_prior, d_post


def REF_st_grad(logits, w):
    """d sum(w * (onehot + p - sg(p))) / d logits = J_softmax^T w."""
    p = np.exp(_log_softmax(logits))
    w = np.float64(w)
    return p * (w - (p * w).sum(-1, keepdims=True))


def REF_bounded_normal(m, s, lo, hi):
    m = np.float64(m); s = np.float64(s)
    return np.tanh(m), (hi - lo) / (1 + np.exp(-(s + 2.0))) + lo


def REF_normal_logp(x, mu, sd):
    x = np.float64(x); mu = np.float64(mu); sd = np.float64(sd)
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def REF_normal_entropy(sd):
    return 0.5 * np.log(2 * np.pi * np.float64(sd) ** 2) + 0.5


def REF_logp_grads(x, m, s, lo, hi):
    """d sum(logp(x | bounded_normal(m, s))) / d m, / d s — the actor's
    logpi(sg(a)) backward (0.4d iii: the term a global gate cannot see)."""
    x = np.float64(x)
    mu, sd = REF_bounded_normal(m, s, lo, hi)
    z = (x - mu) / sd
    dmu = z / sd
    dsd = z * z / sd - 1 / sd
    sig = (sd - lo) / (hi - lo)
    return dmu * (1 - mu ** 2), dsd * (hi - lo) * sig * (1 - sig)


def COND_logp_grads(x, m, s, lo, hi):
    """The scale each actor gradient is judged against: |ref| + |upstream|_summed x
    max|local derivative| (tanh' <= 1, ((hi-lo) sigmoid)' <= (hi-lo)/4).
    Why (measured 11-19, `--vs-official`): float32 forms 1 - y^2 from y =
    tanh(m) with an error of ~1 ulp of 1, times the upstream dlogp/dmu. At
    m = 8 XLA's float32 tanh returns exactly 1.0 (nearest float32 0.99999976)
    and the official gradient is 0 where float64 says 2.2e-6: against |ref| + 1
    the official code itself misses 1e-6 (by 2.2e-6; at m = 6 by 1.1e-6). It
    is within 4.5e-7 of this scale. Same reasoning as the decode's condition."""
    # Summed magnitudes, so neither x - mu (z = 0 when the action equals the
    # mean) nor z^2 - 1 (the std term) can cancel the scale to zero.
    mu, sd = REF_bounded_normal(m, s, lo, hi)
    a = (np.abs(np.float64(x)) + np.abs(mu)) / sd
    return a / sd, (a * a + 1) / sd * (hi - lo) / 4


def REF_binary_logp(logit, event):
    l = np.float64(logit); e = np.float64(event)
    ls = -np.logaddexp(0, -l)       # log sigmoid(l), stable
    lns = -np.logaddexp(0, l)       # log sigmoid(-l)
    return e * ls + (1 - e) * lns


# ---------------------------------------------------------------- inputs
# float32 inputs, fixed; both sides read exactly these numbers.

def inputs():
    rng = np.random.default_rng(20261119)
    f = np.float32
    tiny = np.array([0, 1e-30, 1e-12, 1e-8, 1e-6, 1e-3], f)
    big = np.array([0.5, 1, 2, 10, 1e3, 1e6, 4.85e8, 1e30], f)
    x_symlog = np.concatenate([tiny, -tiny, big, -big, rng.normal(0, 30, 64).astype(f)])
    x_symexp = np.concatenate([tiny, -tiny, np.array([0.5, 1, 5, 15, 20, 30, 80], f),
                               -np.array([0.5, 1, 5, 15, 20, 30, 80], f),
                               rng.uniform(-20, 20, 64).astype(f)])
    bins32 = OFFICIAL_BINS()
    on_bins = bins32[[0, 1, 63, 126, 127, 128, 200, 253, 254]]
    targets = np.concatenate([
        on_bins, np.array([0, 1e-7, -1e-7, 0.3, -0.3, 1.7, 123.4, -98765.4, 1e9, -1e9, 3e38], f),
        (rng.normal(0, 1, 32) * 10 ** rng.uniform(-3, 6, 32)).astype(f)]).astype(f)
    logits_tw = rng.normal(0, 1, (len(targets), 255)).astype(f)
    logits_dec = np.concatenate([
        rng.normal(0, 1, (16, 255)), rng.normal(0, 4, (8, 255)),
        np.tile(np.where(np.arange(255) == 140, 30.0, 0.0), (1, 1))]).astype(f)
    half = rng.normal(0, 2, (6, 127))
    mirrored = np.concatenate([half, rng.normal(0, 2, (6, 1)), half[:, ::-1]], 1).astype(f)
    uniform = np.zeros((3, 255), f)
    um_logits = np.concatenate([rng.normal(0, 1, (8, 4)), rng.normal(0, 20, (8, 4)),
                                np.array([[50, -50, 0, 0]])]).astype(f)
    post = rng.normal(0, 1.5, (4, 32, 4)).astype(f)
    prior = (post + rng.normal(0, 1.5, (4, 32, 4))).astype(f)
    prior[0] = post[0] + rng.normal(0, 0.05, (32, 4)).astype(f)  # KL < free: clamp, zero grad
    idx = rng.integers(0, 4, (6, 32)).astype(np.int32)
    st_logits = rng.normal(0, 2, (6, 32, 4)).astype(f)
    st_w = rng.normal(0, 1, (6, 32, 4)).astype(f)
    m_raw = np.concatenate([rng.normal(0, 1, 40), [0, 3, -3, 8]]).astype(f)
    s_raw = np.concatenate([rng.normal(0, 2, 40), [0, -6, 6, -2]]).astype(f)
    act = np.concatenate([rng.normal(0, 1, 40), [0, 0.9, -1.5, 2.5]]).astype(f)
    bl = np.concatenate([rng.normal(0, 3, 40), [0, 20, -20, 50, -50, 100, -100]]).astype(f)
    be = (rng.uniform(0, 1, len(bl)) < 0.5).astype(f)
    be[-6:] = [1, 0, 0, 1, 1, 0]
    return dict(x_symlog=x_symlog, x_symexp=x_symexp, bins32=bins32, targets=targets,
                logits_tw=logits_tw, logits_dec=logits_dec, mirrored=mirrored, uniform=uniform,
                um_logits=um_logits, post=post, prior=prior, idx=idx, st_logits=st_logits,
                st_w=st_w, m_raw=m_raw, s_raw=s_raw, act=act, bl=bl, be=be)


def close(name, got, ref, s, scale=None):
    got = np.float64(np.asarray(got)); ref = np.float64(ref)
    if got.shape != ref.shape:
        return False, f"{name}: shape {got.shape} != {ref.shape}"
    sc = (np.abs(ref) if scale is None else np.float64(scale)) + s
    with np.errstate(invalid="ignore"):
        err = np.abs(got - ref) / sc
    err = np.where(got == ref, 0.0, err)          # exact match at a zero scale is 0, not 0/0
    err = np.where(np.isnan(err), np.inf, err)
    worst = float(err.max()) if err.size else 0.0
    ok = bool(np.all(np.isfinite(got)) and worst <= TOL)
    lab = f"(|ref|+{s:g})" if scale is None else "(cond)"
    return ok, f"{name}: worst {worst:.3g} x{lab} {'ok' if ok else 'OVER 1e-6'}"


# ---------------------------------------------------------------- the tests
# Each takes (M, mx, compiled) and returns a list of (ok, detail).

def _wrap(mx, fn, compiled):
    return mx.compile(fn) if compiled else fn


def t_symlog(M, mx, c, I):
    x = mx.array(I["x_symlog"]); y = mx.array(I["x_symexp"])
    a = _wrap(mx, M.symlog, c)(x); b = _wrap(mx, M.symexp, c)(y)
    return [close("symlog", a, REF_symlog(I["x_symlog"]), 1e-30),
            close("symexp", b, REF_symexp(I["x_symexp"]), 1e-30)]


def t_bins(M, mx, c, I):
    b = np.asarray(M.bins(255))
    out = [close("bins(255) vs official f32", b, OFFICIAL_BINS(), 1e-30)]
    out.append((b.dtype == np.float32, f"bins dtype {b.dtype} (float32 required)"))
    out.append((bool(np.all(b[::-1] == -b)), "bins mirrored exactly (b[::-1] == -b)"))
    return out


def t_twohot(M, mx, c, I):
    b = mx.array(I["bins32"]); t = mx.array(I["targets"]); lg = mx.array(I["logits_tw"])
    w = _wrap(mx, lambda t: M.twohot_target(t, b), c)(t)
    loss = _wrap(mx, lambda l, t: M.twohot_loss(l, t, b), c)(lg, t)
    wr = REF_twohot_target(I["targets"], I["bins32"])
    out = [close("twohot weights", w, wr, 1.0),
           close("twohot loss", loss, REF_twohot_loss(I["logits_tw"], I["targets"], I["bins32"]), 1.0)]
    out.append((bool(np.all(np.abs(np.asarray(w).sum(-1) - 1) <= 1e-6)), "twohot weights sum to 1"))
    return out


def t_decode(M, mx, c, I):
    b = mx.array(I["bins32"])
    dec = _wrap(mx, lambda l: M.twohot_decode(l, b), c)
    ref, cond = REF_twohot_decode(I["logits_dec"], I["bins32"])
    out = [close("decode", dec(mx.array(I["logits_dec"])), ref, 0.0, scale=cond)]
    u = np.asarray(dec(mx.array(I["uniform"])))
    m = np.asarray(dec(mx.array(I["mirrored"])))
    out.append((bool(np.all(u == 0.0)), f"uniform decode == 0 exactly: {u.tolist()}"))
    out.append((bool(np.all(m == 0.0)), f"mirrored decode == 0 exactly: {m.tolist()}"))
    return out


def t_unimix(M, mx, c, I):
    a = _wrap(mx, lambda l: M.unimix_logits(l, 0.01), c)(mx.array(I["um_logits"]))
    return [close("unimix 1%", a, REF_unimix_logits(I["um_logits"], 0.01), 1.0)]


def t_kl(M, mx, c, I):
    P, Q = mx.array(I["post"]), mx.array(I["prior"])
    f = _wrap(mx, lambda p, q: M.kl_dyn_rep(p, q, 0.01, 1.0), c)
    dyn, rep = f(P, Q)
    ref, raw = REF_kl(I["post"], I["prior"], 0.01, 1.0)
    out = [close("kl dyn value", dyn, ref, 1.0), close("kl rep value", rep, ref, 1.0)]
    g_dyn = _wrap(mx, mx.grad(lambda p, q: M.kl_dyn_rep(p, q, 0.01, 1.0)[0].sum(), argnums=(0, 1)), c)(P, Q)
    g_rep = _wrap(mx, mx.grad(lambda p, q: M.kl_dyn_rep(p, q, 0.01, 1.0)[1].sum(), argnums=(0, 1)), c)(P, Q)
    rq, rp = REF_kl_grads(I["post"], I["prior"], 0.01, 1.0)
    out.append((bool(np.all(np.asarray(g_dyn[0]) == 0)), "d dyn / d post == 0 (sg(post))"))
    out.append((bool(np.all(np.asarray(g_rep[1]) == 0)), "d rep / d prior == 0 (sg(prior))"))
    out.append(close("d dyn / d prior", g_dyn[1], rq, 1.0))
    out.append(close("d rep / d post", g_rep[0], rp, 1.0))
    clamped = raw < 1.0
    out.append((bool(clamped[0] and np.all(np.asarray(g_rep[0])[0] == 0)),
                f"free bits: raw KL {raw[0]:.4f} < 1 -> zero gradient"))
    return out


def t_st(M, mx, c, I):
    i, L, W = mx.array(I["idx"]), mx.array(I["st_logits"]), mx.array(I["st_w"])
    v = np.asarray(_wrap(mx, M.straight_through, c)(i, L))
    onehot = np.eye(4, dtype=np.float32)[I["idx"]]
    out = [(bool(np.all(v == onehot)), "straight-through forward == onehot exactly")]
    g = _wrap(mx, mx.grad(lambda l: (M.straight_through(i, l) * W).sum()), c)(L)
    out.append(close("straight-through grad", g, REF_st_grad(I["st_logits"], I["st_w"]), 1.0))
    return out


def t_normal(M, mx, c, I):
    m, s, x = mx.array(I["m_raw"]), mx.array(I["s_raw"]), mx.array(I["act"])
    mu, sd = _wrap(mx, lambda m, s: M.bounded_normal(m, s, 0.1, 1.0), c)(m, s)
    rmu, rsd = REF_bounded_normal(I["m_raw"], I["s_raw"], 0.1, 1.0)
    out = [close("actor mean tanh", mu, rmu, 1.0), close("actor std", sd, rsd, 1.0)]
    lp = _wrap(mx, M.normal_logp, c)(x, mx.array(rmu.astype(np.float32)), mx.array(rsd.astype(np.float32)))
    out.append(close("normal logp", lp, REF_normal_logp(I["act"], rmu.astype(np.float32), rsd.astype(np.float32)), 1.0))
    en = _wrap(mx, M.normal_entropy, c)(mx.array(rsd.astype(np.float32)))
    out.append(close("normal entropy", en, REF_normal_entropy(rsd.astype(np.float32)), 1.0))

    def tot(m, s):
        mu, sd = M.bounded_normal(m, s, 0.1, 1.0)
        return M.normal_logp(x, mu, sd).sum()
    gm, gs = _wrap(mx, mx.grad(tot, argnums=(0, 1)), c)(m, s)
    rgm, rgs = REF_logp_grads(I["act"], I["m_raw"], I["s_raw"], 0.1, 1.0)
    cm, cs = COND_logp_grads(I["act"], I["m_raw"], I["s_raw"], 0.1, 1.0)
    out += [close("d logp / d mean_raw", gm, rgm, 0.0, scale=np.abs(rgm) + cm),
            close("d logp / d std_raw", gs, rgs, 0.0, scale=np.abs(rgs) + cs)]
    return out


def t_binary(M, mx, c, I):
    a = _wrap(mx, M.binary_logp, c)(mx.array(I["bl"]), mx.array(I["be"]))
    return [close("binary logp", a, REF_binary_logp(I["bl"], I["be"]), 1.0)]


TESTS = [("squash", t_symlog), ("bins", t_bins), ("twohot", t_twohot), ("decode", t_decode),
         ("unimix", t_unimix), ("kl", t_kl), ("straight_through", t_st),
         ("normal", t_normal), ("binary", t_binary)]


def run_tests(M, mx, I, verbose=True):
    """Returns {test: {"eager": ok, "compiled": ok, "detail": [...]}}."""
    res = {}
    for name, t in TESTS:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c, I)
                ok = all(o for o, _ in rows)
                det = [d for _, d in rows]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                ok, det = False, [f"{type(e).__name__}: {e}"[:200]]
            r[mode] = ok
            r[mode + "_detail"] = det
        res[name] = r
        if verbose:
            print(f"{name:17s} eager {'PASS' if r['eager'] else 'FAIL'}  "
                  f"compiled {'PASS' if r['compiled'] else 'FAIL'}")
            for d in r["eager_detail"]:
                print("    e:", d)
            if r["compiled_detail"] != r["eager_detail"]:
                for d in r["compiled_detail"]:
                    print("    c:", d)
    return res


# ---------------------------------------------------------------- planted
# Each a standalone WRONG MLX implementation of one function — a bug class the
# report or an earlier port names. Its test must FAIL (eager or compiled).

def planted(mx):
    import types
    sm = lambda l: mx.softmax(l, axis=-1)  # noqa: E731
    P = {}
    P["squash"] = ("symlog via log(1+|x|), not log1p", dict(
        symlog=lambda x: mx.sign(x) * mx.log(1 + mx.abs(x)),
        symexp=lambda x: mx.sign(x) * (mx.exp(mx.abs(x)) - 1)))
    P["bins"] = ("256 bins from 128-point halves (even branch for an odd head)", dict(
        bins=lambda n: mx.array(REF_bins(256)[:255].astype(np.float32))))
    P["twohot"] = ("two-hot weights linear in symlog space, not raw", dict(
        twohot_target=lambda t, b: mx.array(REF_twohot_target(
            REF_symlog(np.asarray(t)), REF_symlog(np.asarray(b))).astype(np.float32)),
        twohot_loss=lambda l, t, b: -(mx.array(REF_twohot_target(
            REF_symlog(np.asarray(t)), REF_symlog(np.asarray(b))).astype(np.float32))
            * (l - mx.logsumexp(l, axis=-1, keepdims=True))).sum(-1)))
    P["decode"] = ("naive left-to-right sum(p * bins)", dict(
        twohot_decode=lambda l, b: (sm(l) * b).sum(-1)))
    P["unimix"] = ("mix applied to logits, not probabilities", dict(
        unimix_logits=lambda l, m: (1 - m) * l))

    def kl_nosg(p, q, m, f):
        lp = mx.log(0.99 * sm(p) + 0.01 / 4); lq = mx.log(0.99 * sm(q) + 0.01 / 4)
        k = mx.maximum((mx.exp(lp) * (lp - lq)).sum(-1).sum(-1), f)
        return k, k
    P["kl"] = ("dyn and rep without their stop-gradients", dict(kl_dyn_rep=kl_nosg))

    def st_bad(i, l):
        oh = (mx.arange(l.shape[-1]) == i[..., None]).astype(mx.float32)
        return (oh + sm(l)) - mx.stop_gradient(sm(l))
    P["straight_through"] = ("(onehot + p) - sg(p): forward not exactly onehot", dict(
        straight_through=st_bad))

    P["normal"] = ("std without the +2 shift", dict(
        bounded_normal=lambda m, s, lo, hi: (mx.tanh(m), (hi - lo) * mx.sigmoid(s) + lo),
        normal_logp=lambda x, mu, sd: -0.5 * ((x - mu) / sd) ** 2 - mx.log(sd) - 0.5 * np.log(2 * np.pi),
        normal_entropy=lambda sd: 0.5 * mx.log(2 * np.pi * sd * sd) + 0.5))
    P["binary"] = ("log(sigmoid(l)) — underflows to -inf at |l| = 100", dict(
        binary_logp=lambda l, e: e * mx.log(mx.sigmoid(l)) + (1 - e) * mx.log(mx.sigmoid(-l))))
    out = {}
    for test, (why, fns) in P.items():
        out[test] = (why, types.SimpleNamespace(**fns))
    return out


def main_mlx(planted_mode):
    gxx = "/opt/rh/gcc-toolset-15/root/usr/bin"   # mlx_smoke.py: mx.compile needs it
    if os.path.isdir(gxx) and not os.environ.get("PATH", "").startswith(gxx):
        os.environ["PATH"] = gxx + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TMPDIR", "/data/tmp")
    import mlx.core as mx
    I = inputs()
    if not planted_mode:
        sys.path.insert(0, HERE)
        import mlx_maths as M
        res = run_tests(M, mx, I)
        npass = sum(r["eager"] and r["compiled"] for r in res.values())
        print(f"RUNG 1: {npass}/{len(res)} tests pass eager AND compiled "
              f"(mlx {mx.__version__}) -> {'PASS' if npass == len(res) else 'FAIL'}")
        return 0 if npass == len(res) else 1
    caught = 0
    P = planted(mx)
    for test, (why, M) in P.items():
        t = dict(TESTS)[test]
        fails = []
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c, I)
                bad = [d for o, d in rows if not o]
            except Exception as e:  # noqa: BLE001
                bad = [f"{type(e).__name__}: {e}"[:160]]
            if bad:
                fails.append(f"{mode}: {bad[0]}")
        ok = bool(fails)
        caught += ok
        print(f"planted {test:17s} [{why}] -> {'CAUGHT' if ok else 'MISSED'}")
        for f in fails:
            print("    ", f)
    # Report-only probe for step 1.3: the official symmetric sum (outs.py
    # L298–302) ported literally — does mx.compile keep it exactly 0? (JAX jit
    # does not on this box: issue #245.) Not a plant: either answer is allowed.
    def sym(l, b):
        p = mx.softmax(l, axis=-1); m = (l.shape[-1] - 1) // 2
        return (p[..., m:m + 1] * b[m:m + 1]).sum(-1) + (
            (p[..., :m] * b[:m])[..., ::-1] + p[..., m + 1:] * b[m + 1:]).sum(-1)
    b32 = mx.array(I["bins32"])
    for nm in ("uniform", "mirrored"):
        e = np.asarray(sym(mx.array(I[nm]), b32)).tolist()
        c = np.asarray(mx.compile(sym)(mx.array(I[nm]), b32)).tolist()
        print(f"probe official symmetric sum, {nm}: eager {e} compiled {c}")
    print(f"PLANTED: {caught}/{len(P)} caught -> {'OK' if caught == len(P) else 'HARNESS BLIND'}")
    return 0 if caught == len(P) else 1


# ---------------------------------------------------------------- vs official

def main_official():
    """Every REF_* against the official JAX code at e01491f, float32 CPU, on
    the same inputs. The tolerance is the test's own; a REF that the official
    float32 code misses by more than rounding is a wrong reference."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")
    from embodied.jax import nets, outs
    I = inputs()
    f32 = jnp.float32
    rows = []

    def chk(name, got, ref, s, scale=None):
        ok, d = close(name, np.asarray(got), ref, s, scale)
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    chk("symlog", nets.symlog(jnp.array(I["x_symlog"])), REF_symlog(I["x_symlog"]), 1e-30)
    chk("symexp", nets.symexp(jnp.array(I["x_symexp"])), REF_symexp(I["x_symexp"]), 1e-30)
    half = nets.symexp(jnp.linspace(-20, 0, 128, dtype=f32))
    obins = np.asarray(jnp.concatenate([half, -half[:-1][::-1]], 0))   # heads.py L137–139
    same = bool(np.all(obins.view(np.uint32) == OFFICIAL_BINS().view(np.uint32)))
    rows.append((same, "official bins == pinned rung1_bins_e01491f.json bit for bit"))
    print(("  ok  " if same else "  XX  ") + rows[-1][1])
    rb = REF_bins(255); nz = rb != 0
    rel = float(np.max(np.abs(obins[nz] - rb[nz]) / np.abs(rb[nz])))
    nd = int(np.sum(obins != rb.astype(np.float32)))
    rows.append((rel <= 2e-6, f"official f32 bins vs float64 REF_bins: max rel {rel:.3g} "
                              f"(<= 2e-6), {nd}/255 differ from REF cast to f32"))
    print(("  ok  " if rows[-1][0] else "  XX  ") + rows[-1][1])
    b = jnp.array(I["bins32"])
    th = outs.TwoHot(jnp.array(I["logits_tw"]), b)
    chk("twohot loss", th.loss(jnp.array(I["targets"])),
        REF_twohot_loss(I["logits_tw"], I["targets"], I["bins32"]), 1.0)
    # weights: the official loss with one-hot logits at each bin recovers them
    ref, cond = REF_twohot_decode(I["logits_dec"], I["bins32"])
    chk("decode", outs.TwoHot(jnp.array(I["logits_dec"]), b).pred(), ref, 0.0, scale=cond)
    for nm in ("uniform", "mirrored"):
        v = np.asarray(outs.TwoHot(jnp.array(I[nm]), b).pred())
        ok = bool(np.all(v == 0)); rows.append((ok, f"official {nm} decode == 0: {v.tolist()}"))
        print(("  ok  " if ok else "  XX  ") + rows[-1][1])
        vj = np.asarray(jax.jit(lambda l: outs.TwoHot(l, b).pred())(jnp.array(I[nm])))
        rows.append((True, f"official {nm} decode under jax.jit: {vj.tolist()} (issue #245 says jit may miss 0)"))
        print("  ..  " + rows[-1][1])
    chk("unimix 1%", outs.Categorical(jnp.array(I["um_logits"]), 0.01).logits,
        REF_unimix_logits(I["um_logits"], 0.01), 1.0)

    def dist(l):
        return outs.Agg(outs.OneHot(l, 0.01), 1, jnp.sum)
    sg = jax.lax.stop_gradient

    def dynrep(p, q):
        dyn = jnp.maximum(dist(sg(p)).kl(dist(q)), 1.0)    # rssm.py L123–129
        rep = jnp.maximum(dist(p).kl(dist(sg(q))), 1.0)
        return dyn, rep
    P, Q = jnp.array(I["post"]), jnp.array(I["prior"])
    dyn, rep = dynrep(P, Q)
    r, raw = REF_kl(I["post"], I["prior"], 0.01, 1.0)
    chk("kl dyn", dyn, r, 1.0); chk("kl rep", rep, r, 1.0)
    gd = jax.grad(lambda p, q: dynrep(p, q)[0].sum(), argnums=(0, 1))(P, Q)
    gr = jax.grad(lambda p, q: dynrep(p, q)[1].sum(), argnums=(0, 1))(P, Q)
    rq, rp = REF_kl_grads(I["post"], I["prior"], 0.01, 1.0)
    chk("d dyn/d prior", gd[1], rq, 1.0); chk("d rep/d post", gr[0], rp, 1.0)
    z = bool(np.all(np.asarray(gd[0]) == 0) and np.all(np.asarray(gr[1]) == 0))
    rows.append((z, "official stop-gradient grads exactly 0")); print(("  ok  " if z else "  XX  ") + rows[-1][1])
    oh = outs.OneHot(jnp.array(I["st_logits"]))
    v = np.asarray(oh._onehot_with_grad(jnp.array(I["idx"])))
    z = bool(np.all(v == np.eye(4, dtype=np.float32)[I["idx"]]))
    rows.append((z, "official straight-through forward == onehot")); print(("  ok  " if z else "  XX  ") + rows[-1][1])
    W = jnp.array(I["st_w"])
    g = jax.grad(lambda l: (outs.OneHot(l)._onehot_with_grad(jnp.array(I["idx"])) * W).sum())(jnp.array(I["st_logits"]))
    chk("straight-through grad", g, REF_st_grad(I["st_logits"], I["st_w"]), 1.0)

    def bn(m, s):   # heads.py L146–155 with minstd 0.1, maxstd 1.0
        sd = (1.0 - 0.1) * jax.nn.sigmoid(s + 2.0) + 0.1
        return outs.Normal(jnp.tanh(m), sd)
    m, s, x = jnp.array(I["m_raw"]), jnp.array(I["s_raw"]), jnp.array(I["act"])
    d = bn(m, s)
    rmu, rsd = REF_bounded_normal(I["m_raw"], I["s_raw"], 0.1, 1.0)
    chk("actor mean", d.mean, rmu, 1.0); chk("actor std", d.stddev, rsd, 1.0)
    d2 = outs.Normal(jnp.array(rmu.astype(np.float32)), jnp.array(rsd.astype(np.float32)))
    chk("normal logp", d2.logp(x), REF_normal_logp(I["act"], rmu.astype(np.float32), rsd.astype(np.float32)), 1.0)
    chk("normal entropy", d2.entropy(), REF_normal_entropy(rsd.astype(np.float32)), 1.0)
    gm, gs = jax.grad(lambda m, s: bn(m, s).logp(x).sum(), argnums=(0, 1))(m, s)
    rgm, rgs = REF_logp_grads(I["act"], I["m_raw"], I["s_raw"], 0.1, 1.0)
    cm, cs = COND_logp_grads(I["act"], I["m_raw"], I["s_raw"], 0.1, 1.0)
    chk("d logp/d mean_raw", gm, rgm, 0.0, scale=np.abs(rgm) + cm)
    chk("d logp/d std_raw", gs, rgs, 0.0, scale=np.abs(rgs) + cs)
    chk("binary logp", outs.Binary(jnp.array(I["bl"])).logp(jnp.array(I["be"])),
        REF_binary_logp(I["bl"], I["be"]), 1.0)
    n_ok = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f): {n_ok}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n_ok == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--vs-official" in a:
        sys.exit(main_official())
    sys.exit(main_mlx("--planted" in a))
