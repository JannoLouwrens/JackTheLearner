"""MLX port of DreamerV3 — rung 11 references (ORDERS J25, PORT report step 2.5 second part):
the critic loss, size1m.

Official code `danijar/dreamerv3` at e01491f:
  `agent.py` L65-68 val = MLPHead(scalar, **value) (layers 3, units 1024 -> size1m 64, silu, rms,
      symexp_twohot, outscale 0.0, bins 255); slowval = SlowModel(MLPHead(.., name='slowval'),
      source=val, rate 0.02, every 1); valnorm = Normalize(impl 'none').
  `imag_loss` L382-438 (value part): val = value.pred() (valnorm none: offset 0, scale 1);
      tarval = val (config slowtar False); ret = lambda_return(0, 1-con, rew, tarval, tarval, disc, lam)
      tar_padded = [ret, 0]; losses['value'] = sg(weight[:, :-1]) *
      (value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
  `repl_loss` L448-480: same with real is_last / is_terminal / reward, boot, disc 1-1/333,
      weight ~last -> losses['repval'].
  `agent.loss` L241: critic total = mean(value) * 1.0 + mean(repval) * 0.3 (loss_scales).
  `train` L141 + `utils.py` SlowModel.update: after the step, slow = mix*src + (1-mix)*slow,
      mix = rate where count % every == 0.

What is compared (NumPy float64 vs the official f32 run, the official functions run for real with
REAL value / slowvalue heads and the REAL valnorm; only the policy and the ret/adv normalisers are
stubbed — rung 12):
  (1) published init: value logits all zero (outscale 0.0) -> pred 0, loss log 255 per term;
  (2) NOISED val and slowval params (seeded N(0, 0.3), different seeds — rung 8b's scar: at init the
      head is blind): val pred, ret, value [N,H] and repval [B,K-1] losses;
  (3) GRADIENT of the critic total w.r.t. every val parameter (official jax.grad) vs float64 central
      differences on a sample of elements, and EXACTLY zero on every slowval parameter (the slow
      head is only ever under sg) — the forward value cannot see a wrongly routed gradient;
  (4) one SlowModel.update vs the float64 EMA;
  planted wrong references, each must be CAUGHT.

Inputs: imagination feat = rung 10 golden's policy-path rollout (N 32, H+1 16, deter 512 + stoch
32x4), rew / con = rung 10's ret_in; replay feat = rung 7 size1m golden's last K = 8 steps, with
rung 10's last / term / rrew / rboot (boot = the imagined return at each start in the agent; here a
fixed array, as rung 10). "Ours beside theirs": f64 vs f32 (published bf16); size1m units 64; N 32
starts (published B*T); valnorm 'none' is the published config, so the critic sees raw returns.
Tolerance as rung 8b: element |d|/(|ref|+1) <= 1e-4, means <= 1e-5; gradients relative 1e-3 of
the sampled element's magnitude + 1e-6 (f32 run vs f64 differences).

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \
      experiments/brain_a/rung11.py --vs-official [--dump-golden]
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL, REF_bins, REF_twohot_loss, REF_twohot_decode  # noqa: E402
from rung10 import REF_lambda_return  # noqa: E402

GOLDEN7 = os.path.join(HERE, "rung7_scan_golden_size1m_e01491f.npz")
GOLDEN10 = os.path.join(HERE, "rung10_imag_golden_size1m_e01491f.npz")
GOLDEN11 = os.path.join(HERE, "rung11_critic_golden_size1m_e01491f.npz")
UNITS, LAYERS, NBINS = 64, 3, 255
K, H = 8, 15
HORIZON, LAM, SLOWREG, REPVAL = 333, 0.95, 1.0, 0.3
RATE, EVERY = 0.02, 1
NOISE_SD = 0.3
HEAD_SD = 0.03
TOL_EL, TOL_MEAN, TOL_G = 1e-4, 1e-5, 1e-3


def close(name, got, ref, el_tol=TOL_EL, mean_tol=TOL_MEAN):
    got, ref = np.float64(np.asarray(got)), np.float64(ref)
    if got.shape != ref.shape:
        return False, f"{name}: shape {got.shape} != {ref.shape}"
    el = float((np.abs(got - ref) / (np.abs(ref) + 1)).max())
    mn = float(abs(got.mean() - ref.mean()) / (abs(ref.mean()) + 1))
    ok = bool(np.all(np.isfinite(got)) and el <= el_tol and mn <= mean_tol)
    return ok, f"{name}: mean {mn:.3g}, element {el:.3g} x(|ref|+1) {'ok' if ok else 'OVER'}"


def feat2tensor(deter, stoch):
    return np.concatenate([deter, stoch.reshape(*stoch.shape[:2], -1)], -1).astype(np.float32)


def inputs():
    G10, G7 = np.load(GOLDEN10), np.load(GOLDEN7)
    fi = feat2tensor(G10["policy/feat/deter"], G10["policy/feat/stoch"])
    fr = feat2tensor(G7["out/deter"][:, -K:], G7["out/stoch"][:, -K:])
    R = {k[len("ret_in/"):]: G10[k] for k in G10.files if k.startswith("ret_in/")}
    return fi, fr, R


def _silu(x):
    return x / (1 + np.exp(-x))


def REF_logits(P, prefix, x, layers=LAYERS, eps=1e-4, act=_silu):
    f = lambda k: np.asarray(P[f"{prefix}/{k}"], np.float64)  # noqa: E731
    x = np.asarray(x, np.float64)
    for i in range(layers):
        x = x @ f(f"mlp/linear{i}/kernel") + f(f"mlp/linear{i}/bias")
        x = act(x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * f(f"mlp/norm{i}/scale"))
    W, b = f("head/logits/kernel"), f("head/logits/bias")
    return x @ W.reshape(W.shape[0], -1) + b.reshape(-1)


def REF_critic(P, fi, fr, R, bins=None, slowreg=SLOWREG, layers=LAYERS, pad=True, slowtar_src="slowval",
               repl_weight="last", tar_from="val", fixed=None):
    """Float64 critic: value [N,H], repval [B,K-1], total, and the preds. `fixed` = a previous
    result whose ret / rret / slow / slow_r are used as the (stop-gradient) targets — the official
    sg(tar_padded) and sg(slowvalue.pred()): a derivative moves the logits, not the targets."""
    b = REF_bins(NBINS) if bins is None else bins
    lv_i, ls_i = REF_logits(P, "val", fi, layers), REF_logits(P, slowtar_src, fi, layers)
    lv_r, ls_r = REF_logits(P, "val", fr, layers), REF_logits(P, slowtar_src, fr, layers)
    val_i, slow_i = REF_twohot_decode(lv_i, b)[0], REF_twohot_decode(ls_i, b)[0]
    val_r, slow_r = REF_twohot_decode(lv_r, b)[0], REF_twohot_decode(ls_r, b)[0]
    tar_i = val_i if tar_from == "val" else slow_i
    tar_r = val_r if tar_from == "val" else slow_r
    con = np.float64(R["con"])
    weight = np.cumprod(con, 1)                                      # contdisc: disc 1
    ret = REF_lambda_return(np.zeros_like(con), 1 - con, R["rew"], tar_i, tar_i, 1.0, LAM)
    if fixed is not None:
        ret, slow_i = fixed["ret"], fixed["slow"]
    tp = np.concatenate([ret, 0 * ret[:, -1:]], 1) if pad else np.concatenate([ret, ret[:, -1:]], 1)
    value = (weight * (REF_twohot_loss(lv_i, tp, b) + slowreg * REF_twohot_loss(lv_i, slow_i, b)))[:, :-1]
    rret = REF_lambda_return(R["last"], R["term"], R["rrew"], tar_r, R["rboot"], 1 - 1 / HORIZON, LAM)
    if fixed is not None:
        ret, rret, slow_i, slow_r = fixed["ret"], fixed["rret"], fixed["slow"], fixed["slow_r"]
    rp = np.concatenate([rret, 0 * rret[:, -1:]], 1)
    w = np.float64(~R["last"]) if repl_weight == "last" else np.float64(~R["term"])
    repval = (w * (REF_twohot_loss(lv_r, rp, b) + slowreg * REF_twohot_loss(lv_r, slow_r, b)))[:, :-1]
    return dict(value=value, repval=repval, total=value.mean() + REPVAL * repval.mean(),
                val=val_i, slow=slow_i, slow_r=slow_r, ret=ret, rret=rret)


def noised(shapes, seed):
    """Seeded N(0, 0.3) in sorted key order (+1 on norm scales); the OUTPUT kernel at N(0, HEAD_SD):
    at 0.3 its 255 logits spread so wide that the decoded values reach 1e7 (std 6.7e6 measured
    11-25) — a regime a critic never sits in, where f32 softmax noise alone fills the tolerance."""
    r = np.random.default_rng(seed)
    return {k: np.float32(r.normal(0, HEAD_SD if "head/logits/kernel" in k else NOISE_SD, shapes[k])
                          + (1.0 if k.endswith("scale") else 0.0))
            for k in sorted(shapes)}


def grad_sample(P, n_per=3, seed=11):
    """(key, index) pairs: n_per random elements of every val parameter."""
    r = np.random.default_rng(seed)
    out = []
    for k in sorted(P):
        if k.startswith("val/"):
            for _ in range(n_per):
                out.append((k, tuple(int(r.integers(s)) for s in P[k].shape)))
    return out


def REF_grad(P, fi, fr, R, sample, h=1e-5, sg=True):
    base = REF_critic(P, fi, fr, R) if sg else None
    g = []
    for k, i in sample:
        Pp = dict(P); Pm = dict(P)
        Pp[k] = np.float64(P[k]).copy(); Pp[k][i] += h
        Pm[k] = np.float64(P[k]).copy(); Pm[k][i] -= h
        g.append((REF_critic(Pp, fi, fr, R, fixed=base)["total"]
                  - REF_critic(Pm, fi, fr, R, fixed=base)["total"]) / (2 * h))
    return np.array(g)


def main_official():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax
    import embodied.jax.nets as nets
    nets.COMPUTE_DTYPE = jnp.float32
    from dreamerv3 import agent as A
    scalar = elements.Space(np.float32, ())
    kw = dict(layers=LAYERS, units=UNITS, act="silu", norm="rms", output="symexp_twohot",
              outscale=0.0, winit="trunc_normal_in", bins=NBINS)
    valm = embodied.jax.MLPHead(scalar, **kw, name="val")
    slowm = embodied.jax.MLPHead(scalar, **kw, name="slowval")
    valnorm = embodied.jax.Normalize(impl="none", rate=0.01, limit=1e-8, name="valnorm")
    fi, fr, R = inputs()
    rows = []

    def put(ok, d):
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    class Pol:
        def __init__(self, n): self.n = n
        def logp(self, x): return jnp.zeros(self.n)
        def entropy(self): return jnp.zeros(self.n)

    class Norm:
        def stats(self): return jnp.float32(0), jnp.float32(1)
        def __call__(self, x, update): return jnp.float32(0), jnp.float32(1)

    S = R["rew"].shape
    J = jnp.asarray

    def run(fi, fr, rew, con, last, term, rrew, rboot):
        v_i, s_i = valm(fi, 2), slowm(fi, 2)
        li, oi, _ = A.imag_loss({"a": jnp.zeros(S)}, rew, con, {"a": Pol(S)}, v_i, s_i, Norm(), valnorm,
                                Norm(), update=True, contdisc=True, horizon=HORIZON, slowtar=False,
                                lam=LAM, actent=3e-4, slowreg=SLOWREG)
        v_r, s_r = valm(fr, 2), slowm(fr, 2)
        lr, orr, _ = A.repl_loss(last, term, rrew, rboot, v_r, s_r, valnorm, update=True,
                                 slowreg=SLOWREG, slowtar=False, horizon=HORIZON, lam=LAM)
        total = li["value"].mean(1).mean() * 1.0 + lr["repval"].mean() * REPVAL
        return dict(value=li["value"], repval=lr["repval"], total=total, val=v_i.pred(), slow=s_i.pred(),
                    ret=oi["ret"], rret=orr["ret"], logits=v_i.logits)

    args = tuple(J(x) for x in (fi, fr, R["rew"], R["con"], R["last"], R["term"], R["rrew"], R["rboot"]))
    with jax.disable_jit():
        state, _ = nj.pure(run)({}, *args, create=True, seed=jax.random.PRNGKey(0))
        _, out0 = nj.pure(run)(state, *args, seed=jax.random.PRNGKey(1))
        shapes = {k: np.shape(v) for k, v in state.items()}
        Pn = {**noised({k: s for k, s in shapes.items() if k.startswith("val/")}, 7),
              **noised({k: s for k, s in shapes.items() if k.startswith("slowval/")}, 8)}
        put(set(Pn) == set(state), f"param keys: {len(state)} (val + slowval, valnorm 'none' holds none)")
        stN = {k: J(v) for k, v in Pn.items()}
        _, outN = nj.pure(run)(stN, *args, seed=jax.random.PRNGKey(1))
        tot = lambda st: nj.pure(run)(st, *args, seed=jax.random.PRNGKey(1))[1]["total"]  # noqa: E731
        G = jax.grad(tot)(stN)
    out0 = jax.tree.map(np.asarray, out0)
    outN = jax.tree.map(np.asarray, outN)
    G = {k: np.asarray(v) for k, v in G.items()}
    P0 = {k: np.asarray(v) for k, v in state.items()}
    print("        params:", ", ".join(f"{k} {tuple(v.shape)}" for k, v in sorted(P0.items()) if k.startswith("val/")))
    N = fi.shape[0]
    put(out0["value"].shape == (N, H) and out0["repval"].shape == (R["last"].shape[0], K - 1),
        f"loss shapes value {out0['value'].shape}, repval {out0['repval'].shape}")
    z = float(np.abs(out0["logits"]).max())
    put(z == 0.0 and float(np.abs(out0["val"]).max()) == 0.0,
        f"published init: value logits all zero, pred 0 (max |logit| {z:.3g}; outscale 0.0)")
    ref0 = REF_critic(P0, fi, fr, R)
    for k in ("value", "repval"):
        put(*close(f"published init: {k} f32 vs f64", out0[k], ref0[k]))
    refN = REF_critic(Pn, fi, fr, R)
    for k in ("val", "slow", "ret", "rret", "value", "repval"):
        put(*close(f"noised params: {k} f32 vs f64", outN[k], refN[k]))
    put(*close("noised params: critic total f32 vs f64", outN["total"], refN["total"]))
    sp = float(np.std(outN["val"]))
    put(sp > 1.0, f"noised val pred varies over [N,H+1] (std {sp:.3g}) — the pass is not blind")
    # (3) gradients
    smp = grad_sample(Pn)
    gr = REF_grad(Pn, fi, fr, R, smp)
    go = np.array([G[k][i] for k, i in smp])
    gel = np.abs(go - gr) / (np.abs(gr) + 1e-6)
    gok = bool(np.all(np.isfinite(go)) and (np.abs(go - gr) <= TOL_G * np.abs(gr) + 1e-6).all())
    put(gok, f"grad of critic total, {len(smp)} val elements (every param, 3 each) f32 vs f64 central "
             f"differences: worst rel {gel.max():.3g} {'ok' if gok else 'OVER'}")
    for j in np.argsort(-gel)[:4]:
        print(f"        grad {smp[j][0]}{list(smp[j][1])}: official {go[j]:.6g}, f64 {gr[j]:.6g}")
    nz = float(np.abs(go).min())
    put(nz > 0, f"sampled val grads non-zero (min |g| {nz:.3g}) — the gradient check is not blind")
    gbad = REF_grad(Pn, fi, fr, R, smp, sg=False)
    cb = not bool((np.abs(go - gbad) <= TOL_G * np.abs(gbad) + 1e-6).all())
    put(cb, f"planted grad 'value target not under sg (ret moves with val)': {'CAUGHT' if cb else 'MISSED'}")
    sz = max(float(np.abs(v).max()) for k, v in G.items() if k.startswith("slowval/"))
    put(sz == 0.0, f"slowval grads exactly 0 on every param (max |g| {sz:.3g}) — slow head only under sg")
    # (4) SlowModel.update, official
    src_m = embodied.jax.MLPHead(scalar, **kw, name="src")
    dst_m = embodied.jax.SlowModel(embodied.jax.MLPHead(scalar, **kw, name="dst"), source=src_m,
                                   rate=RATE, every=EVERY)

    def upd(x):
        src_m(x, 2); dst_m(x, 2); dst_m.update()
        return 0
    Ps = noised({"src/" + k[4:]: s for k, s in shapes.items() if k.startswith("val/")}, 9)
    Pd = noised({"dst/" + k[4:]: s for k, s in shapes.items() if k.startswith("val/")}, 10)
    with jax.disable_jit():
        st2 = {**{k: J(v) for k, v in {**Ps, **Pd}.items()}, "dst_count/value": J(np.int32(0))}
        st2, _ = nj.pure(upd)(st2, J(fi[:2, :2]), seed=jax.random.PRNGKey(2))
    st2 = {k: np.asarray(v) for k, v in st2.items()}
    ref_upd = {k: RATE * np.float64(Ps["src/" + k[4:]]) + (1 - RATE) * np.float64(Pd[k]) for k in Pd}
    worst = max(float(np.abs(st2[k] - ref_upd[k]).max() / (np.abs(ref_upd[k]).max() + 1)) for k in Pd)
    put(worst <= 1e-6 and int(st2.get("dst_count/value", -1)) == 1,
        f"SlowModel.update: dst = 0.02 src + 0.98 dst on {len(Pd)} params (worst {worst:.3g}), count -> 1")
    srcmove = max(float(np.abs(st2[k] - Ps[k]).max()) for k in Ps)
    put(srcmove == 0.0, "SlowModel.update leaves the source untouched")
    # planted wrong references
    mut = {
        "slowreg term dropped": dict(slowreg=0.0),
        "two hidden layers, not three": dict(layers=2),
        "bins linear, not symexp": dict(bins=np.linspace(-REF_bins(NBINS).max(), REF_bins(NBINS).max(), NBINS)),
        "slowreg regularises toward val's own pred": dict(slowtar_src="val"),
        "repval weighted by ~is_terminal, not ~is_last": dict(repl_weight="term"),
        "returns bootstrap from slowval (slowtar True)": dict(tar_from="slow"),
    }
    for name, kw_ in mut.items():
        m = REF_critic(Pn, fi, fr, R, **kw_)
        bad = not (close("m", outN["value"], m["value"])[0] and close("m", outN["repval"], m["repval"])[0])
        put(bad, f"planted '{name}': {'CAUGHT' if bad else 'MISSED'}")
    if "--dump-golden" in sys.argv:
        np.savez_compressed(GOLDEN11, **{f"out0/{k}": v for k, v in out0.items()},
                            **{f"outN/{k}": v for k, v in outN.items()},
                            **{f"grad/{k}": v for k, v in G.items()},
                            **{f"param0/{k}": v for k, v in P0.items()},
                            **{f"slowupd/{k}": v for k, v in st2.items() if k.startswith("dst/")})
        print(f"golden written: {GOLDEN11}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f, size1m, N {N}, H {H}): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


# ---------------------------------------------------------------- port tests (MLX)
# `mlx_maths.critic_loss(P, fi, rew, con, fr, last, term, rrew, rboot, horizon, lam, slowreg, repval)`
#   -> dict(value [N,H], repval [B,K-1], total (scalar), val [N,H+1], slow [N,H+1], ret [N,H],
#   rret [B,K-1]) — val/slowval MLPHeads (3 layers) on feat2tensor'd inputs, imag_loss's and
#   repl_loss's value terms with valnorm 'none', total = mean(value) + repval * mean(repval).
#   P keys are the official state's ("val/mlp/linear0/kernel", "slowval/...").
# Each test eager AND under mx.compile:
#   critic_ref     port vs REF_critic (float64) at the noised params
#   critic_golden  port vs the official f32 dump at published init (out0) and noised (outN)
#   critic_grad    mx.grad of total w.r.t. every val param vs the official jax.grad dump (per param
#                  max |d| <= 1e-3 x max |g_official|), every slowval grad exactly 0; control: the
#                  grad with the slowreg term dropped differs
#   slow_update    mlx_maths.slow_update (rung 3's) on the val params vs the official SlowModel dump
TOL_GP = 1e-3


def _golden():
    G = np.load(GOLDEN11)
    P0 = {k[len("param0/"):]: G[k] for k in G.files if k.startswith("param0/")}
    shapes = {k: v.shape for k, v in P0.items()}
    Pn = {**noised({k: s for k, s in shapes.items() if k.startswith("val/")}, 7),
          **noised({k: s for k, s in shapes.items() if k.startswith("slowval/")}, 8)}
    return G, P0, Pn


def _port(M, mx, P, compiled, slowreg=SLOWREG):
    fi, fr, R = inputs()
    A = mx.array
    MP = {k: A(v) for k, v in P.items()}
    args = (A(fi), A(R["rew"]), A(R["con"]), A(fr), A(R["last"]), A(R["term"]), A(R["rrew"]), A(R["rboot"]))
    f = lambda P_, *a: M.critic_loss(P_, *a, HORIZON, LAM, slowreg, REPVAL)  # noqa: E731
    out = (mx.compile(f) if compiled else f)(MP, *args)
    return {k: np.asarray(v) for k, v in out.items()}, MP, args, f


def t_critic(M, mx, compiled, against):
    G, P0, Pn = _golden()
    fi, fr, R = inputs()
    rows = []
    o, *_ = _port(M, mx, Pn, compiled)
    if against == "ref":
        r = REF_critic(Pn, fi, fr, R)
        for k in ("val", "slow", "ret", "rret", "value", "repval", "total"):
            rows.append(close(f"noised {k} vs REF", o[k], r[k]))
    else:
        o0, *_ = _port(M, mx, P0, compiled)
        for k in ("value", "repval", "total", "val"):
            rows.append(close(f"published init {k} vs GOLDEN", o0[k], G[f"out0/{k}"]))
        for k in ("val", "slow", "ret", "rret", "value", "repval", "total"):
            rows.append(close(f"noised {k} vs GOLDEN", o[k], G[f"outN/{k}"]))
    return rows


def t_grad(M, mx, compiled):
    G, _, Pn = _golden()
    rows = []

    def grads(slowreg):
        _, MP, args, f = _port(M, mx, Pn, False, slowreg)
        g = mx.grad(lambda P_: f(P_, *args)["total"])
        g = mx.compile(g) if compiled else g
        return {k: np.asarray(v) for k, v in g(MP).items()}
    g = grads(SLOWREG)
    worst, keyw = 0.0, ""
    for k in sorted(k for k in G.files if k.startswith("grad/val/")):
        name = k[len("grad/"):]
        ref = np.float64(G[k])
        d = float(np.abs(np.float64(g[name]) - ref).max() / (np.abs(ref).max() + 1e-12))
        if d > worst:
            worst, keyw = d, name
    rows.append((np.isfinite(worst) and worst <= TOL_GP,
                 f"grad of total, 11 val params vs GOLDEN jax.grad: worst {worst:.3g} x max|g| ({keyw})"))
    sz = max(float(np.abs(v).max()) for k, v in g.items() if k.startswith("slowval/"))
    rows.append((sz == 0.0, f"slowval grads exactly 0 (max |g| {sz:.3g})"))
    g0 = grads(0.0)
    d = max(float(np.abs(g0[k] - g[k]).max()) for k in g if k.startswith("val/"))
    rows.append((d > 1e-5, f"control: slowreg 0 changes the val grads ({d:.3g} > 1e-5)"))
    return rows


def t_slow(M, mx, compiled):
    G, _, _ = _golden()
    shapes = {k[len("param0/val/"):]: G[k].shape for k in G.files if k.startswith("param0/val/")}
    Ps = noised({"src/" + k: s for k, s in shapes.items()}, 9)
    Pd = noised({"dst/" + k: s for k, s in shapes.items()}, 10)
    f = lambda s, d, c: M.slow_update(s, d, c, RATE, EVERY)  # noqa: E731
    f = mx.compile(f) if compiled else f
    rows = []
    for k in sorted(Pd):
        new, cnt = f(mx.array(Ps["src/" + k[4:]]), mx.array(Pd[k]), mx.array(np.int32(0)))
        rows.append(close(f"slow {k} vs GOLDEN", np.asarray(new), G[f"slowupd/{k}"], 1e-6, 1e-6))
    rows.append((int(np.asarray(cnt)) == 1, "count -> 1"))
    return rows


TESTS = [("critic_ref", lambda M, mx, c: t_critic(M, mx, c, "ref")),
         ("critic_golden", lambda M, mx, c: t_critic(M, mx, c, "golden")),
         ("critic_grad", t_grad),
         ("slow_update", t_slow)]


def main_mlx():
    import rung6
    mx = rung6._mlx()
    import mlx_maths as M
    n = 0
    for name, t in TESTS:
        r = {}
        for mode, c in (("eager", False), ("compiled", True)):
            try:
                rows = t(M, mx, c)
                r[mode] = all(o for o, _ in rows)
                det = [d for o, d in rows if not o][:4] if not r[mode] else [d for _, d in rows]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                r[mode], det = False, [f"{type(e).__name__}: {e}"[:200]]
            if mode == "eager" or not r[mode]:
                for d in det:
                    print(f"    {mode[0]}:", d)
        n += r["eager"] and r["compiled"]
        print(f"{name:14s} eager {'PASS' if r['eager'] else 'FAIL'}  compiled {'PASS' if r['compiled'] else 'FAIL'}")
    print(f"RUNG 11: {n}/{len(TESTS)} tests pass eager AND compiled (mlx {mx.__version__}) -> "
          f"{'PASS' if n == len(TESTS) else 'FAIL'}")
    return 0 if n == len(TESTS) else 1


def main_planted():
    """Each planted bug is patched into the real port; the tests must FAIL eager."""
    import rung6
    mx = rung6._mlx()
    import mlx_maths as M
    names = ("twohot_target", "lambda_return", "_mlp_head_n", "critic_loss", "_ACT")
    keep = {k: getattr(M, k) for k in names}

    SG = mx.stop_gradient

    def tt_nosg(target, b):                 # rung 1's twohot_target, the WEIGHTS left carrying grads
        n = b.shape[-1]
        t = target.astype(mx.float32)[..., None]
        ti = SG(t)                          # bin indices are integers: MLX cannot VJP a gather index
        below = mx.clip((b <= ti).astype(mx.int32).sum(-1) - 1, 0, n - 1)
        above = mx.clip(n - (b > ti).astype(mx.int32).sum(-1), 0, n - 1)
        eq = below == above
        one = mx.array(1.0, mx.float32)
        db = mx.where(eq, one, mx.abs(b[below] - t[..., 0]))
        da = mx.where(eq, one, mx.abs(b[above] - t[..., 0]))
        return (M._onehot(below, n) * (da / (db + da))[..., None] + M._onehot(above, n) * (db / (db + da))[..., None])

    def nosg_critic(P, fi, rew, con, fr, last, term, rrew, rboot, horizon, lam, slowreg, repval):
        try:
            mx.stop_gradient = lambda x: x
            M.twohot_target = tt_nosg
            return keep["critic_loss"](P, fi, rew, con, fr, last, term, rrew, rboot, horizon, lam, slowreg, repval)
        finally:
            mx.stop_gradient = SG
            M.twohot_target = keep["twohot_target"]

    bugs = {
        "no stop-gradient anywhere in the critic (targets carry grads)": dict(critic_loss=nosg_critic),
        "lam on the bootstrap, not the continuation": dict(
            lambda_return=lambda la, te, r, v, b, d, lam: keep["lambda_return"](la, te, r, v, b, d, 1 - lam)),
        "two hidden layers, not three": dict(_mlp_head_n=lambda P, pre, x, out, n: keep["_mlp_head_n"](P, pre, x, out, n - 1)),
        "slowreg term dropped": dict(critic_loss=lambda P, *a: keep["critic_loss"](P, *a[:10], 0.0, a[11])),
        "repval scale 1.0, not 0.3": dict(critic_loss=lambda P, *a: keep["critic_loss"](P, *a[:11], 1.0)),
        "relu for silu": dict(_ACT={**keep["_ACT"], "silu": lambda x: mx.maximum(x, 0)}),
    }
    caught = 0
    for name, patch in bugs.items():
        for k, v in patch.items():
            setattr(M, k, v)
        try:
            bad = [d for _, t in TESTS for o, d in t(M, mx, False) if not o]
        except Exception as e:  # noqa: BLE001
            bad = [f"crash {type(e).__name__}: {e}"]
        finally:
            for k, v in keep.items():
                setattr(M, k, v)
        caught += bool(bad)
        print(f"planted [{name}] -> {'CAUGHT' if bad else 'MISSED'}" + (f"  ({bad[0][:90]})" if bad else ""))
    ctl = all(o for _, t in TESTS for o, _ in t(M, mx, False))
    print(f"PLANTED: {caught}/{len(bugs)} caught, unpatched port {'passes' if ctl else 'FAILS'} -> "
          f"{'OK' if caught == len(bugs) and ctl else 'HARNESS BLIND'}")
    return 0 if caught == len(bugs) and ctl else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv:
        sys.exit(main_official())
    if "--planted" in sys.argv[1:]:
        sys.exit(main_planted())
    sys.exit(main_mlx())
