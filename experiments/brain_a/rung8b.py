"""MLX port of DreamerV3 — rung 8b (PORT report step 2.4, second world-model loss row):
the reward and continue prediction losses, over rung 7's size1m feature pass. REFERENCES ONLY
in this commit (NumPy float64 vs the official code); the MLX module and its tests are next.

Source: PORT report ladder row 8 "World-model losses — relative 1e-5 on means — golden: yes".
Official code `danijar/dreamerv3` at e01491f, `dreamerv3/agent.py` L51-58 and L172-177:
  feat2tensor = concat(deter, stoch.reshape(..., -1))
  rew = MLPHead(scalar, **rewhead)  -> TwoHot(logits, symexp bins 255);  loss = -logp(reward)
  con = MLPHead(binary, **conhead)  -> Binary(logit);  con = f32(~is_terminal) * (1 - 1/horizon)
  rewhead {layers 1, units 1024 (size1m: 64), silu, rms, symexp_twohot, outscale 0.0, bins 255}
  conhead {layers 1, units 1024 (size1m: 64), silu, rms, binary, outscale 1.0}
  MLP layer: Linear -> Norm('rms', eps 1e-4, scale) -> act;   Head: one Linear, kernel x outscale.

The scar this rung is built around: rewhead's outscale 0.0 makes the reward logits all ZERO at
init, so at published init the rew loss is log(255) whatever the MLP or the input — a golden at
init is blind to the whole head. So the official heads are run TWICE on the same inputs:
  (i)  published init — asserts the zero logits / log 255, and compares con;
  (ii) every head parameter (kernels, biases, norm scales) replaced by seeded N(0, 0.3) noise —
       the comparison that can fail on any parameter path.
Planted wrong references must each be CAUGHT by (ii).

"Ours beside theirs": NumPy float64 vs official float32 (COMPUTE_DTYPE f32, as rungs 6-8a;
published bf16). Inputs: rung 7's size1m golden deter/stoch (deter 512 + stoch 32x4 = 640),
reward and is_terminal drawn here from a fixed seed (rewards span 0, +-small, +-large; ~10%
terminal). The decoder's vector loss is rung 8c, not here. Tolerance as 8a: element
|d|/(|ref|+1) <= 1e-4 and the [B,T] mean <= 1e-5.

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \
      experiments/brain_a/rung8b.py --vs-official [--dump-golden]
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL, REF_bins, REF_twohot_loss, REF_binary_logp  # noqa: E402

GOLDEN7 = os.path.join(HERE, "rung7_scan_golden_size1m_e01491f.npz")
GOLDEN8 = os.path.join(HERE, "rung8b_rewcon_golden_size1m_e01491f.npz")
UNITS = 64            # size1m: .*\.units: 64
HORIZON = 333
TOL_MEAN = 1e-5
TOL_EL = 1e-4
NOISE_SD = 0.3


def close(name, got, ref):
    got, ref = np.float64(np.asarray(got)), np.float64(ref)
    if got.shape != ref.shape:
        return False, f"{name}: shape {got.shape} != {ref.shape}"
    el = float((np.abs(got - ref) / (np.abs(ref) + 1)).max())
    mn = float(abs(got.mean() - ref.mean()) / (abs(ref.mean()) + 1))
    ok = bool(np.all(np.isfinite(got)) and el <= TOL_EL and mn <= TOL_MEAN)
    return ok, f"{name}: mean {mn:.3g}, element {el:.3g} x(|ref|+1) {'ok' if ok else 'OVER'}"


def inputs():
    G = np.load(GOLDEN7)
    deter, stoch = G["out/deter"], G["out/stoch"]
    feat = np.concatenate([deter, stoch.reshape(*stoch.shape[:2], -1)], -1).astype(np.float32)
    B, T = feat.shape[:2]
    r = np.random.default_rng(20261125)
    rew = r.choice([0.0, 0.0, 1.0, -1.0, 0.03, 7.5, -250.0, 1e4], size=(B, T)) * r.uniform(0.5, 1.5, (B, T))
    term = r.uniform(size=(B, T)) < 0.1
    return feat, rew.astype(np.float32), term


def con_target(term):
    return np.float64(~term) * (1 - 1 / HORIZON)


def _silu(x):
    return x / (1 + np.exp(-x))


def REF_mlp_head(P, prefix, x, out, eps=1e-4, rms=True, act=_silu):
    f = lambda k: np.asarray(P[f"{prefix}/{k}"], np.float64)  # noqa: E731  (np.float64(arr) scalarises (1,))
    x = np.asarray(x, np.float64) @ f("mlp/linear0/kernel") + f("mlp/linear0/bias")
    if rms:
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * f("mlp/norm0/scale")
    x = act(x)
    W, b = f(f"head/{out}/kernel"), f(f"head/{out}/bias")
    return x @ W.reshape(W.shape[0], -1) + b.reshape(-1)


def REF_losses(P, feat, rew, term, **kw):
    lr = REF_mlp_head(P, "rew", feat, "logits", **kw)
    lc = REF_mlp_head(P, "con", feat, "logit", **kw)[..., 0]
    return dict(rew=REF_twohot_loss(lr, rew, REF_bins(255)), con=-REF_binary_logp(lc, con_target(term)),
                rew_logits=lr, con_logit=lc)


def noised_params(shapes, seed=7):
    """Pass (ii)'s params: seeded N(0, 0.3) in SORTED key order (+1 on norm scales), so the MLX
    tests regenerate them from the shapes instead of the golden carrying 400 KB of noise."""
    r = np.random.default_rng(seed)
    return {k: np.float32(r.normal(0, NOISE_SD, shapes[k]) + (1.0 if k.endswith("scale") else 0.0))
            for k in sorted(shapes)}


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
    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    kw = dict(layers=1, units=UNITS, act="silu", norm="rms", winit="trunc_normal_in")
    rewm = embodied.jax.MLPHead(scalar, output="symexp_twohot", outscale=0.0, bins=255, **kw, name="rew")
    conm = embodied.jax.MLPHead(binary, output="binary", outscale=1.0, **kw, name="con")
    feat, rew, term = inputs()
    rows = []

    def put(ok, d):
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    def run(feat, rew, con):
        r, c = rewm(feat, 2), conm(feat, 2)
        return dict(rew=r.loss(rew), con=c.loss(con), rew_logits=r.logits, con_logit=c.logit)

    args = (jnp.asarray(feat), jnp.asarray(rew), jnp.asarray(np.float32(con_target(term))))
    with jax.disable_jit():
        state, _ = nj.pure(run)({}, *args, create=True, seed=jax.random.PRNGKey(0))
        _, out0 = nj.pure(run)(state, *args, seed=jax.random.PRNGKey(1))
        stateN = noised_params({k: np.shape(v) for k, v in state.items()})
        _, outN = nj.pure(run)({k: jnp.asarray(v) for k, v in stateN.items()}, *args,
                               seed=jax.random.PRNGKey(1))
    out0 = {k: np.asarray(v) for k, v in out0.items()}
    outN = {k: np.asarray(v) for k, v in outN.items()}
    P0 = {k: np.asarray(v) for k, v in state.items()}
    print("        params:", ", ".join(f"{k} {tuple(v.shape)}" for k, v in sorted(P0.items())))
    B, T = rew.shape
    put(out0["rew"].shape == (B, T) and out0["con"].shape == (B, T), f"loss shapes [B,T] = {out0['rew'].shape}")
    z = float(np.abs(out0["rew_logits"]).max())
    put(z == 0.0, f"published init: rew logits all zero (max |logit| {z:.3g}; outscale 0.0)")
    put(*close("published init: rew loss == log 255", out0["rew"], np.full((B, T), np.log(255.0))))
    ref0 = REF_losses(P0, feat, rew, term)
    put(*close("published init: con f32 vs f64", out0["con"], ref0["con"]))
    refN = REF_losses(stateN, feat, rew, term)
    for k in ("rew_logits", "con_logit", "rew", "con"):
        put(*close(f"noised params: {k} f32 vs f64", outN[k], refN[k]))
    spread = float(np.std(outN["rew"]))
    put(spread > 0.1, f"noised rew loss varies over [B,T] (std {spread:.3g}) — the pass is not blind")
    mut = {
        "no rms norm": REF_losses(stateN, feat, rew, term, rms=False),
        "relu for silu": REF_losses(stateN, feat, rew, term, act=lambda x: np.maximum(x, 0)),
        "norm eps 1.0 for 1e-4": REF_losses(stateN, feat, rew, term, eps=1.0),
        "con target without the horizon discount": dict(
            REF_losses(stateN, feat, rew, term), con=-REF_binary_logp(refN["con_logit"], np.float64(~term))),
        "rew bins linear, not symexp": dict(refN, rew=REF_twohot_loss(
            refN["rew_logits"], rew, np.linspace(-REF_bins(255).max(), REF_bins(255).max(), 255))),
        "stoch before deter in feat": REF_losses(stateN, np.concatenate([feat[..., 512:], feat[..., :512]], -1),
                                                 rew, term),
    }
    for name, m in mut.items():
        bad = not (close("m", outN["rew"], m["rew"])[0] and close("m", outN["con"], m["con"])[0])
        put(bad, f"planted '{name}': {'CAUGHT' if bad else 'MISSED'}")
    if "--dump-golden" in sys.argv:
        np.savez(GOLDEN8, rew=outN["rew"], con=outN["con"], rew0=out0["rew"], con0=out0["con"],
                 reward=rew, term=term,
                 **{f"param0/{k}": v for k, v in P0.items()})
        print(f"golden written: {GOLDEN8}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f, size1m): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


if __name__ == "__main__":
    if "--vs-official" in sys.argv:
        sys.exit(main_official())
    print("rung 8b: references only so far; run with --vs-official")
    sys.exit(2)
