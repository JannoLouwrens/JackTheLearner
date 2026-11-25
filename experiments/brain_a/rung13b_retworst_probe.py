# Rung 13b-imagloss-sym lead (11-25 21:30): the port's worst ret deviation from the f64 truth is 13x the official's
# at call 8 (0.382 vs 0.0287). Where is it, and is it the val decode's f32 error carried through the recursion?
# Run: nice -n 19 /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung13b_retworst_probe.py
import sys, numpy as np
sys.path.insert(0, '/home/opc/jackthelearner/experiments/brain_a'); sys.argv = ['x']
import rung13b as r
import mlx.core as mx, mlx_maths as M
G, R = np.load(r.GOLDEN_A), np.load(r.R13); B = M.bins(255); b64 = np.array(B).astype(np.float64)
def dec64(L):
    L = L.astype(np.float64); p = np.exp(L - L.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    return p[..., 127] * b64[127] + ((p[..., 128:] - p[..., :127][..., ::-1]) * b64[128:]).sum(-1)
def lret(con, rew, val, lam=0.95):
    live = con[:, 1:]; interm = rew[:, 1:] + (1 - lam) * live * val[:, 1:]; rets = [val[:, -1]]
    for t in reversed(range(live.shape[1])): rets.append(interm[:, t] + live[:, t] * lam * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)
for c in (6, 8):
    P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}
    g = f"c{c}/imag/"
    d = np.concatenate([G[g+"start/deter"][:, None], G[g+"feat/deter"]], 1); s = np.concatenate([G[g+"start/stoch"][:, None], G[g+"feat/stoch"]], 1)
    x = M.feat2tensor(mx.array(d), mx.array(s))
    Lv = np.array(M._mlp_head_n(P("val"), "val", x, "logits", 3)); v32 = np.array(M.twohot_decode(mx.array(Lv), B)); v64 = dec64(Lv)
    gr = G[f"c{c}/imagloss/in_rew"].astype(np.float64); gc = G[f"c{c}/imagloss/in_con"].astype(np.float64); go = G[f"c{c}/imagloss/outs/ret"]
    t64 = lret(gc, gr, v64); p64 = lret(gc, gr, v32.astype(np.float64))   # port's f32 vals, f64 recursion
    port = np.array(M.lambda_return(mx.zeros_like(mx.array(gc, mx.float32)), 1 - mx.array(gc, mx.float32), mx.array(gr, mx.float32), mx.array(v32), mx.array(v32), 1.0, 0.95))
    ep, eo = np.abs(port - t64), np.abs(go - t64)
    i, t = np.unravel_index(ep.argmax(), ep.shape)
    print(f"call {c}: worst port elem ({i},{t}) port {port[i,t]:.4f} official {go[i,t]:.4f} truth {t64[i,t]:.4f} | "
          f"port-vals-f64-recursion {p64[i,t]:.4f} (so recursion rounding = {abs(port[i,t]-p64[i,t]):.2e})")
    print(f"   val f32-f64 err along row {i}, t..15: " + " ".join(f"{e:+.3f}" for e in (v32 - v64)[i, t:]))
    print(f"   |ret| at worst: {abs(t64[i,t]):.3f}; err stats port p50 {np.median(ep):.2e} p99 {np.percentile(ep,99):.2e} max {ep.max():.3f} | "
          f"official p50 {np.median(eo):.2e} p99 {np.percentile(eo,99):.2e} max {eo.max():.3f}")
    print(f"   val decode err (all elems) p50 {np.median(np.abs(v32-v64)):.2e} p99 {np.percentile(np.abs(v32-v64),99):.2e} max {np.abs(v32-v64).max():.3f}; "
          f"top-bin p max {np.exp(Lv - Lv.max(-1,keepdims=True))[..., -1].max():.2e}")
