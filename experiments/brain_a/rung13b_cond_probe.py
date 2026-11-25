# Rung 13b-imagloss, second diagnosis (11-25 20:50): is EVERY rew/val/ret miss inside the f32 decode's own
# conditioning? kappa(elem) = sum_i ulp_f32(p_i) * |bin_i| (one rounding step of each prob, summed) — the size
# of a decode change that one-ulp disagreements in the softmax can produce. For ret (linear in val with weights
# in [0,1] over the remaining horizon) the bound used is the sum of kappa_val over t..15.
# Run: nice -n 19 /data/venvs/jack-mlx/bin/python experiments/brain_a/rung13b_cond_probe.py
import sys, numpy as np
sys.path.insert(0, '/home/opc/jackthelearner/experiments/brain_a'); sys.argv = ['x']
import rung13b as r
import mlx.core as mx, mlx_maths as M
G, R = np.load(r.GOLDEN_A), np.load(r.R13); b = np.abs(np.array(M.bins(255)).astype(np.float64)); B = M.bins(255)
def kappa(L):
    L = L.astype(np.float32); p = np.exp(L - L.max(-1, keepdims=True)); p = (p / p.sum(-1, keepdims=True)).astype(np.float32)
    return (np.spacing(p).astype(np.float64) * b).sum(-1)
for c in (1, 2, 6, 8):
    P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}
    g = f"c{c}/imag/"
    d = np.concatenate([G[g+"start/deter"][:, None], G[g+"feat/deter"]], 1); s = np.concatenate([G[g+"start/stoch"][:, None], G[g+"feat/stoch"]], 1)
    x = M.feat2tensor(mx.array(d), mx.array(s))
    Lr = np.array(M._mlp_head(P("rew"), "rew", x, "logits")); Lv = np.array(M._mlp_head_n(P("val"), "val", x, "logits", 3))
    hr = np.array(M.twohot_decode(mx.array(Lr), B)); val = np.array(M.twohot_decode(mx.array(Lv), B))
    kr, kv = kappa(Lr), kappa(Lv)
    gr = G[f"c{c}/imagloss/in_rew"]; gc = G[f"c{c}/imagloss/in_con"]
    ret = np.array(M.lambda_return(mx.zeros_like(mx.array(gc)), 1.0 - mx.array(gc), mx.array(gr), mx.array(val), mx.array(val), 1.0, 0.95))
    gret = G[f"c{c}/imagloss/outs/ret"]; kret = np.cumsum(kv[:, ::-1], 1)[:, ::-1][:, 1:]
    for nm, a, ref, k in (("rew", hr, gr, kr), ("ret", ret, gret, kret)):
        dd = np.abs(a - ref); tol = 1e-5 * (np.abs(ref) + 1); miss = dd > tol
        inside = dd <= tol + 4 * k
        ill = 4 * k > tol
        print(f"call {c} {nm}: miss {miss.sum()}, of them inside tol+4kappa {(miss & inside).sum()}; "
              f"ill-conditioned elems {ill.sum()} of {ill.size}; miss on WELL-conditioned {(miss & ~ill).sum()}; "
              f"max d/(tol+4k) {np.max(dd / (tol + 4 * k)):.3f}")
