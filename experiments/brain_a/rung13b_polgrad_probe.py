"""Probe (not a test) for rung 13b-grads read 2: why the 13 `pol/` gradients miss at calls 6/8 only.
Policy gradient = d/dθ mean(sg(w) * -(logp(θ) * sg(adv) + actent * ent(θ))), adv = (ret - val[:, :-1]) / scale.
The forward of every piece is certified (12b, imagloss-vq vs an f64 truth), so the gradient is linear in adv:
this probe swaps ONLY adv's source and reads port_gate on the 13 pol tensors:
  port   — adv from the port's f32 val decode (rung 13b-grads as read);
  f64    — adv from an f64 decode of the port's own val logits and an f64 lambda return (the truth);
  gret   — ret = the golden's dumped imagloss/outs/ret, val from the port (the official's own ret).
Also prints |port ret - golden ret| and the share of |adv| carried by the elements where ret differs > 1e-3.
  /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung13b_polgrad_probe.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TMPDIR", "/data/tmp")
import mlx.core as mx  # noqa: E402
import mlx_maths as M  # noqa: E402
import port_gate  # noqa: E402
from rung13b import GOLDEN_A, R13, CALLS  # noqa: E402

G, R = np.load(GOLDEN_A), np.load(R13)
b = M.bins(255)
b64 = np.array(b).astype(np.float64)


def dec64(L):
    L = np.asarray(L, np.float64)
    L = L - L.max(-1, keepdims=True)
    p = np.exp(L)
    p /= p.sum(-1, keepdims=True)
    m = 127
    return p[..., m] * b64[m] + ((p[..., m + 1:] - p[..., :m][..., ::-1]) * b64[m + 1:]).sum(-1)


def lret64(con, rew, val, lam=0.95):
    live = con[:, 1:]
    interm = rew[:, 1:] + (1 - lam) * live * val[:, 1:]
    rets = [val[:, -1]]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * lam * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


def perc(x, q):
    v = np.sort(x.reshape(-1))
    pos = q / 100.0 * (v.size - 1)
    i0 = int(np.floor(pos)); i1 = min(i0 + 1, v.size - 1); fr = pos - i0
    return v[i0] * (1 - fr) + v[i1] * fr


for c in CALLS:
    g = f"c{c}/imag/"
    d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
    s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
    a = np.concatenate([G[g + "act/action"], np.zeros_like(G[g + "act/action"][:, :1])], 1)
    rw, cn = G[f"c{c}/imagloss/in_rew"], G[f"c{c}/imagloss/in_con"]
    lo0, hi0 = float(R[f"c{c}/pre/retnorm/lo/value"]), float(R[f"c{c}/pre/retnorm/hi/value"])
    Pv = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/val/")}
    x = M.feat2tensor(mx.array(d), mx.array(s))
    lv = M._mlp_head_n(Pv, "val", x, "logits", 3)
    val32 = np.array(M.twohot_decode(lv, b)).astype(np.float64)
    val64 = dec64(np.array(lv))
    ret32 = np.array(M.lambda_return(mx.zeros_like(mx.array(cn)), 1.0 - mx.array(cn), mx.array(rw),
                                     M.twohot_decode(lv, b), M.twohot_decode(lv, b), 1.0, 0.95)).astype(np.float64)
    ret64 = lret64(cn.astype(np.float64), rw.astype(np.float64), val64)
    gret = G[f"c{c}/imagloss/outs/ret"].astype(np.float64)

    def adv_of(ret, val):
        lo = 0.99 * lo0 + 0.01 * perc(ret, 5.0); hi = 0.99 * hi0 + 0.01 * perc(ret, 95.0)
        return (ret - val[:, :-1]) / max(1.0, hi - lo)

    advs = dict(port=adv_of(ret32, val32), f64=adv_of(ret64, val64), gret=adv_of(gret, val32))
    dr = np.abs(ret32 - gret)
    big = dr > 1e-3
    print(f"call {c}: |port ret - golden ret| max {dr.max():.3e}, > 1e-3 on {big.sum()} of {dr.size}; "
          f"|val32 - val64| max {np.abs(val32 - val64).max():.3e}; |adv| mean {np.abs(advs['f64']).mean():.3e}; "
          f"share of sum|adv_port - adv_f64| on those: "
          f"{np.abs(advs['port'] - advs['f64'])[big].sum() / max(np.abs(advs['port'] - advs['f64']).sum(), 1e-30):.2f}")
    Pp = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/pol/")}
    gold = {f"c{c}/grad/{k[len(f'c{c}/pre/'):]}": R[f"c{c}/grad/{k[len(f'c{c}/pre/'):]}"] for k in R.files
            if k.startswith(f"c{c}/pre/pol/")}
    w = mx.array(np.cumprod(cn, 1).astype(np.float32))
    for tag, adv in advs.items():
        def loss(P):
            m = M._mlp_head_n(P, "pol", x, "action/mean", 3)
            sd = M._mlp_head_n(P, "pol", x, "action/stddev", 3)
            z = mx.zeros((*m.shape[:-1], 1), mx.float32)
            pl, _ = M.policy_loss(m, sd, z, mx.array(a), mx.zeros(m.shape[:-1], mx.int32),
                                  mx.array(adv.astype(np.float32)), w, 0.1, 1.0, 3e-4)
            return pl.mean()
        gr = mx.grad(loss)(Pp)
        port = {f"c{c}/grad/{k}": np.array(v) for k, v in gr.items()}
        port_gate.report(f"call {c} pol adv={tag}", gold, port)

# PART 2: distance to the TRUTH — the f64-adv gradient (same f32 logp path) is the reference; port (f32 adv)
# and official golden are each compared against it, per pol tensor (rel = ||x - t|| / ||t||).
print("PART 2: rel to the f64-adv gradient, per call: official golden vs port (median / max over 13 pol tensors)")
for c in CALLS:
    g = f"c{c}/imag/"
    d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
    s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
    a = np.concatenate([G[g + "act/action"], np.zeros_like(G[g + "act/action"][:, :1])], 1)
    rw, cn = G[f"c{c}/imagloss/in_rew"], G[f"c{c}/imagloss/in_con"]
    lo0, hi0 = float(R[f"c{c}/pre/retnorm/lo/value"]), float(R[f"c{c}/pre/retnorm/hi/value"])
    Pv = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/val/")}
    x = M.feat2tensor(mx.array(d), mx.array(s))
    lv = M._mlp_head_n(Pv, "val", x, "logits", 3)
    v32 = M.twohot_decode(lv, b)
    val32 = np.array(v32).astype(np.float64)
    val64 = dec64(np.array(lv))
    ret32 = np.array(M.lambda_return(mx.zeros_like(mx.array(cn)), 1.0 - mx.array(cn), mx.array(rw), v32, v32,
                                     1.0, 0.95)).astype(np.float64)
    ret64 = lret64(cn.astype(np.float64), rw.astype(np.float64), val64)

    def adv_of(ret, val):
        lo = 0.99 * lo0 + 0.01 * perc(ret, 5.0); hi = 0.99 * hi0 + 0.01 * perc(ret, 95.0)
        return (ret - val[:, :-1]) / max(1.0, hi - lo)
    Pp = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/pol/")}
    w = mx.array(np.cumprod(cn, 1).astype(np.float32))
    out = {}
    for tag, adv in (("port", adv_of(ret32, val32)), ("f64", adv_of(ret64, val64))):
        def loss(P):
            m = M._mlp_head_n(P, "pol", x, "action/mean", 3)
            sd = M._mlp_head_n(P, "pol", x, "action/stddev", 3)
            z = mx.zeros((*m.shape[:-1], 1), mx.float32)
            pl, _ = M.policy_loss(m, sd, z, mx.array(a), mx.zeros(m.shape[:-1], mx.int32),
                                  mx.array(adv.astype(np.float32)), w, 0.1, 1.0, 3e-4)
            return pl.mean()
        out[tag] = {k: np.array(v, np.float64).ravel() for k, v in mx.grad(loss)(Pp).items()}
    ro, rp = [], []
    for k, t in out["f64"].items():
        nt = np.linalg.norm(t)
        if nt == 0:
            continue
        ro.append(np.linalg.norm(R[f"c{c}/grad/{k}"].astype(np.float64).ravel() - t) / nt)
        rp.append(np.linalg.norm(out["port"][k] - t) / nt)
    if ro:
        print(f"  call {c}: official {np.median(ro):.3e} / {max(ro):.3e}   port {np.median(rp):.3e} / {max(rp):.3e}")
