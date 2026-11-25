"""MLX port of DreamerV3 — rung 8c (PORT report step 2.4, third world-model loss row):
the decoder's VECTOR reconstruction losses, over rung 7's size1m feature pass. References
(NumPy float64 vs the official code) `971816b9`; port tests below (`mlx_maths.dec_vec_loss`),
committed failing before the module.

Source: PORT report ladder row 8 "World-model losses — relative 1e-5 on means — golden: yes".
Official code `danijar/dreamerv3` at e01491f:
  rssm.py L253-306 Decoder (vector branch):
    inp = concat([stoch.reshape(BT, -1), deter.reshape(BT, -1)])      <- STOCH FIRST, unlike
                                                                         agent.feat2tensor (deter first)
    x = MLP(layers 3, units 1024 (size1m: 64), act silu, norm rms)(inp) -> reshape [B,T,units]
    outs = DictHead(veckeys, 'symlog_mse' (continuous) | 'categorical' (discrete), outscale 1.0)
  heads.py L127 symlog_mse: pred = Linear(space.shape)(x); MSE(pred, squash=symlog)
  heads.py L89  Agg(output, len(space.shape), sum)  when space.shape != ()
  outs.py  L138 MSE.loss = square(pred - sg(symlog(target)))
  agent.py L170-181: losses[key] = recon.loss(sg(value)) for every key in dec_space -> [B,T].
  configs.yaml dec.simple {layers 3, units 1024, act silu, norm rms, outscale 1.0,
  winit trunc_normal_in, bspace 8}; size1m .*\\.units 64 (depth 4 is the image branch's).

Obs space used: DMC walker's PROPRIO keys as dm_control publishes them — orientations (14,),
height () (a scalar key: the head has no Agg, the Linear's unit shape is ()), velocity (9,).
Not covered here, said now: the 'categorical' branch (discrete vector keys — none in DMC
proprio) and the whole image branch (BlockLinear + conv; a later rung).

As 8b, two passes on the same inputs: (i) published init (outscale 1.0, so not blind — but
biases are zero and norm scales one, so those paths are only tested by (ii)); (ii) every
decoder parameter replaced by seeded N(0, 0.3) noise (+1 on norm scales). Targets span the
symlog's linear and log regions (|x| from 0.01 to ~300), so a missing squash shows.

"Ours beside theirs": NumPy float64 vs official float32 (COMPUTE_DTYPE f32, as rungs 6-8b;
published bf16). Inputs: rung 7's size1m golden deter [B,T,512] / stoch [B,T,32,4]; targets
drawn here from a fixed seed. Tolerance as 8a/8b: element |d|/(|ref|+1) <= 1e-4, [B,T] mean <= 1e-5.

  PYTHONPATH=/data/tools/ninjax362 nice -n 19 /data/venvs/jack-dv3/bin/python -I \\
      experiments/brain_a/rung8c.py --vs-official [--dump-golden]
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rung1 import OFFICIAL  # noqa: E402
from rung8b import GOLDEN7, close, noised_params, _silu  # noqa: E402

GOLDEN8C = os.path.join(HERE, "rung8c_dec_golden_size1m_e01491f.npz")
UNITS = 64            # size1m: .*\.units: 64
LAYERS = 3
VEC = {"orientations": (14,), "height": (), "velocity": (9,)}   # walker proprio, dm_control order


def feats():
    G = np.load(GOLDEN7)
    return G["out/deter"].astype(np.float32), G["out/stoch"].astype(np.float32)


def targets(B, T):
    r = np.random.default_rng(20261126)
    out = {}
    for k, shp in VEC.items():
        mag = 10.0 ** r.uniform(-2, 2.5, (B, T, *shp))
        out[k] = np.float32(r.choice([-1.0, 1.0], (B, T, *shp)) * mag)
    return out


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def REF_dec(P, deter, stoch, tgt, *, eps=1e-4, rms=True, act=_silu, layers=LAYERS, order="stoch_first",
            squash=symlog, agg=np.sum):
    f = lambda k: np.asarray(P[f"dec/{k}"], np.float64)  # noqa: E731
    B, T = deter.shape[:2]
    s, d = np.float64(stoch).reshape(B * T, -1), np.float64(deter).reshape(B * T, -1)
    x = np.concatenate([s, d] if order == "stoch_first" else [d, s], -1)
    for i in range(layers):
        x = x @ f(f"mlp/linear{i}/kernel") + f(f"mlp/linear{i}/bias")
        if rms:
            x = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * f(f"mlp/norm{i}/scale")
        x = act(x)
    x = x.reshape(B, T, -1)
    out = {}
    for k, shp in VEC.items():
        W, b = f(f"vec/{k}/pred/kernel"), f(f"vec/{k}/pred/bias")
        pred = (x @ W.reshape(W.shape[0], -1) + b.reshape(-1)).reshape(B, T, *shp)
        sq = np.square(pred - squash(np.float64(tgt[k])))
        out[k] = agg(sq, axis=tuple(range(2, sq.ndim))) if shp else sq
        out[f"pred/{k}"] = pred
    return out


def main_official():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, OFFICIAL)
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax.nets as nets
    from dreamerv3 import rssm
    nets.COMPUTE_DTYPE = jnp.float32
    space = {k: elements.Space(np.float32, s) for k, s in VEC.items()}
    dec = rssm.Decoder(space, depth=4, mults=(2, 3, 4, 4), layers=LAYERS, units=UNITS, act="silu",
                       norm="rms", outscale=1.0, winit="trunc_normal_in", outer=False, kernel=5,
                       bspace=8, strided=False, name="dec")
    deter, stoch = feats()
    B, T = deter.shape[:2]
    tgt = targets(B, T)
    rows = []

    def put(ok, d):
        rows.append((ok, d)); print(("  ok  " if ok else "  XX  ") + d)

    def run(deter, stoch, tgt):
        reset = jnp.zeros((B, T), bool)
        _, _, recons = dec({}, dict(deter=deter, stoch=stoch), reset, True)
        out = {k: recons[k].loss(tgt[k]) for k in VEC}
        out.update({f"pred/{k}": recons[k].pred() for k in VEC})
        return out

    args = (jnp.asarray(deter), jnp.asarray(stoch), {k: jnp.asarray(v) for k, v in tgt.items()})
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
    want = {f"dec/mlp/{n}{i}/{p}" for i in range(LAYERS) for n, p in (("linear", "kernel"), ("linear", "bias"),
                                                                       ("norm", "scale"))}
    want |= {f"dec/vec/{k}/pred/{p}" for k in VEC for p in ("kernel", "bias")}
    put(set(P0) == want, f"param names = the reference's {len(want)} (extra {sorted(set(P0) - want)}, "
                         f"missing {sorted(want - set(P0))})")
    put(P0["dec/mlp/linear0/kernel"].shape[0] == 512 + 128, "MLP input width 640 = stoch 128 + deter 512")
    put(all(out0[k].shape == (B, T) for k in VEC), f"loss shapes [B,T] = {out0['height'].shape} for all keys")
    for tag, P, o in (("published init", P0, out0), ("noised params", stateN, outN)):
        ref = REF_dec(P, deter, stoch, tgt)
        for k in VEC:
            put(*close(f"{tag}: {k} loss f32 vs f64", o[k], ref[k]))
            put(*close(f"{tag}: {k} pred f32 vs f64", o[f"pred/{k}"], ref[f"pred/{k}"]))
    refN = REF_dec(stateN, deter, stoch, tgt)
    sd = float(np.std(outN["velocity"]))
    put(sd > 0.1, f"noised velocity loss varies over [B,T] (std {sd:.3g}) — the pass is not blind")
    mut = {
        "deter before stoch (feat2tensor's order)": REF_dec(stateN, deter, stoch, tgt, order="deter_first"),
        "no symlog on the target": REF_dec(stateN, deter, stoch, tgt, squash=lambda x: x),
        "mean over dims, not sum": REF_dec(stateN, deter, stoch, tgt, agg=np.mean),
        "relu for silu": REF_dec(stateN, deter, stoch, tgt, act=lambda x: np.maximum(x, 0)),
        "no rms norm": REF_dec(stateN, deter, stoch, tgt, rms=False),
        "norm eps 1.0 for 1e-4": REF_dec(stateN, deter, stoch, tgt, eps=1.0),
        "symexp of pred vs raw target": {k: (np.square(np.sign(refN[f"pred/{k}"]) * np.expm1(np.minimum(np.abs(
            refN[f"pred/{k}"]), 30)) - np.float64(tgt[k]))) for k in VEC},
    }
    mut["symexp of pred vs raw target"] = {
        k: (v.sum(axis=tuple(range(2, v.ndim))) if VEC[k] else v) for k, v in mut["symexp of pred vs raw target"].items()}
    for name, m in mut.items():
        bad = not all(close("m", outN[k], m[k])[0] for k in VEC)
        put(bad, f"planted '{name}': {'CAUGHT' if bad else 'MISSED'}")
    # 'layers 2 for 3' needs a 2-layer param set: drop linear2/norm2 and feed linear1's width (64 = 64)
    P2 = {k: v for k, v in stateN.items() if "linear2" not in k and "norm2" not in k}
    m2 = REF_dec(P2, deter, stoch, tgt, layers=2)
    bad = not all(close("m", outN[k], m2[k])[0] for k in VEC)
    put(bad, f"planted 'layers 2 for 3': {'CAUGHT' if bad else 'MISSED'}")
    if "--dump-golden" in sys.argv:
        np.savez(GOLDEN8C, **{f"lossN/{k}": outN[k] for k in VEC}, **{f"loss0/{k}": out0[k] for k in VEC},
                 **{f"tgt/{k}": v for k, v in tgt.items()}, **{f"param0/{k}": v for k, v in P0.items()})
        print(f"golden written: {GOLDEN8C}")
    n = sum(o for o, _ in rows)
    print(f"VS-OFFICIAL (jax {jax.__version__}, e01491f, size1m): {n}/{len(rows)} agree -> "
          f"{'REFERENCES HOLD' if n == len(rows) else 'A REFERENCE OR A TOLERANCE IS WRONG'}")
    return 0 if n == len(rows) else 1


# ---------------------------------------------------------------- port tests (MLX)
# `mlx_maths.dec_vec_loss(P, deter, stoch, targets, spaces, layers)` -> dict(<key>: loss [B,T],
# "pred/<key>": [B,T,*shape]) — the decoder's vector branch: dec_inp (stoch FIRST), MLP
# (linear/rms/silu x layers), one Linear per key, MSE vs symlog(target), summed over the key's dims.
# P keys are the official state's ("dec/mlp/linear0/kernel", "dec/vec/height/pred/bias", ...).
# Each test eager AND under mx.compile, on rung 7's size1m deter/stoch and this rung's targets,
# at BOTH param sets:
#   dec_ref     port vs REF_dec (float64) at the same params, losses and preds
#   dec_golden  port vs the official f32 dump (loss0 published init; lossN noised)
# plus: the noised velocity loss varies (not blind); a control with every target's sign flipped
# must differ (the target is read, and read through an odd squash).
# Patch points the planted harness uses: dec_inp, symlog, _rms, _ACT, _vec_agg.

def _golden():
    G = np.load(GOLDEN8C)
    P0 = {k[len("param0/"):]: G[k] for k in G.files if k.startswith("param0/")}
    PN = noised_params({k: v.shape for k, v in P0.items()})
    return G, P0, PN


def _port(M, mx, P, compiled, tgt):
    deter, stoch = feats()
    MP = {k: mx.array(v) for k, v in P.items()}
    T = {k: mx.array(v) for k, v in tgt.items()}
    f = lambda p, d, s, t: M.dec_vec_loss(p, d, s, t, VEC, LAYERS)  # noqa: E731
    if compiled:
        f = mx.compile(f)
    return {k: np.asarray(v) for k, v in f(MP, mx.array(deter), mx.array(stoch), T).items()}


def t_dec(M, mx, compiled, against):
    G, P0, PN = _golden()
    deter, stoch = feats()
    tgt = targets(*deter.shape[:2])
    rows = [(all(np.array_equal(tgt[k], G[f"tgt/{k}"]) for k in VEC),
             "targets() reproduce the golden's targets bit for bit")]
    for tag, P, gk in (("published init", P0, "loss0"), ("noised", PN, "lossN")):
        got = _port(M, mx, P, compiled, tgt)
        if against == "ref":
            ref = REF_dec(P, deter, stoch, tgt)
            for k in VEC:
                rows.append(close(f"{tag} {k} loss vs REF", got[k], ref[k]))
                rows.append(close(f"{tag} {k} pred vs REF", got[f"pred/{k}"], ref[f"pred/{k}"]))
        else:
            for k in VEC:
                rows.append(close(f"{tag} {k} loss vs GOLDEN", got[k], G[f"{gk}/{k}"]))
        if gk == "lossN":
            sd = float(np.std(got["velocity"]))
            rows.append((sd > 0.1, f"noised velocity loss varies (std {sd:.3g} > 0.1)"))
            c = _port(M, mx, P, compiled, {k: -v for k, v in tgt.items()})
            d = float(max(np.abs(c[k] - got[k]).max() for k in VEC))
            rows.append((d > 1e-3, f"control: sign-flipped targets differ ({d:.3g} > 1e-3)"))
    return rows


TESTS = [("dec_ref", lambda M, mx, c: t_dec(M, mx, c, "ref")),
         ("dec_golden", lambda M, mx, c: t_dec(M, mx, c, "golden"))]


def _mlx():
    import rung6
    return rung6._mlx()


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
                det = [d for o, d in rows if not o][:4] if not r[mode] else [d for _, d in rows]
            except Exception as e:  # noqa: BLE001 — a stub or crash is a FAIL, reported
                r[mode], det = False, [f"{type(e).__name__}: {e}"[:200]]
            if mode == "eager" or not r[mode]:
                for d in det:
                    print(f"    {mode[0]}:", d)
        n += r["eager"] and r["compiled"]
        print(f"{name:14s} eager {'PASS' if r['eager'] else 'FAIL'}  compiled {'PASS' if r['compiled'] else 'FAIL'}")
    print(f"RUNG 8c: {n}/{len(TESTS)} tests pass eager AND compiled (mlx {mx.__version__}) -> "
          f"{'PASS' if n == len(TESTS) else 'FAIL'}")
    return 0 if n == len(TESTS) else 1


def main_planted():
    """Each planted bug is patched into the real port; the tests must FAIL eager."""
    mx = _mlx()
    import mlx_maths as M
    names = ("dec_inp", "symlog", "_rms", "_ACT", "_vec_agg")
    keep = {k: getattr(M, k) for k in names}
    bugs = {
        "deter before stoch (feat2tensor's order)": dict(dec_inp=lambda d, s: M.feat2tensor(d, s)),
        "no symlog on the target": dict(symlog=lambda x: x),
        "mean over dims, not sum": dict(_vec_agg=lambda x, axes: mx.mean(x, axes) if axes else x),
        "relu for silu": dict(_ACT={**keep["_ACT"], "silu": lambda x: mx.maximum(x, 0.0)}),
        "no rms norm": dict(_rms=lambda x, s: x * s),
        "norm eps 1.0 for 1e-4": dict(_rms=lambda x, s: x * mx.rsqrt(mx.mean(x * x, -1, keepdims=True) + 1.0) * s),
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
        print(f"planted [{name}] -> {'CAUGHT' if bad else 'MISSED'}" + (f"  ({bad[0][:80]})" if bad else ""))
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
