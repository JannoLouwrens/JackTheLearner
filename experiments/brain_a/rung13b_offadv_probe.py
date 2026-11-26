"""Probe (not a test) for re-registering rung 13b-grads' pol clause (J25, 11-26): can the port's POLICY BACKWARD
be held to the published gate when it is fed the OFFICIAL's own f32 advantage?
Why: wheel 13u read the miss as mlx's f32 softmax rounding — at call 2 |val_mlx - val_rounded| 0.584 while a decode
of the correctly rounded f64 probs sits 1.1e-3 from f64; the two-hot decode (bins to 4.85e8, |val| ~1e-3) turns
ulps into 0.2-0.58 jumps, and adv ~3e-3. Any f32 build (the official included) draws its own adv lottery, so
"port <= 1.25x official vs an f64 truth" compares two draws. The official's adv is not dumped, but its per-element
policy loss is (golden `c<N>/imagloss/loss/policy`): loss = w * -(logpi * adv + actent * ent), so
    adv_off = -(loss / w + actent * ent) / logpi        (w > 0; where w == 0 the term has no gradient)
with logpi/ent from the port's own f32 path (certified: 12b, imagloss-vq). Fed adv_off, the port's pol gradient
must equal the official's R13 `c<N>/grad/pol/*` to f32 backward precision — a 1e-4 published gate, no ratio.
Prints per call: back-solve residual (loss rebuilt from adv_off vs golden), port_gate on the 13 pol tensors, and a
CONTROL that must FAIL: adv_off shifted one step in time.
  /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung13b_offadv_probe.py
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
ACTENT = 3e-4
print(f"mlx {mx.__version__}")
for c in CALLS:
    g = f"c{c}/imag/"
    d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
    s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
    a = mx.array(np.concatenate([G[g + "act/action"], np.zeros_like(G[g + "act/action"][:, :1])], 1))
    cn = G[f"c{c}/imagloss/in_con"]
    x = M.feat2tensor(mx.array(d), mx.array(s))
    Pp = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/pol/")}
    gold = {f"c{c}/grad/{k}": R[f"c{c}/grad/{k}"] for k in Pp}
    w32 = np.cumprod(cn, 1).astype(np.float32)
    w = mx.array(w32)

    def heads(P):
        m = M._mlp_head_n(P, "pol", x, "action/mean", 3)
        sd = M._mlp_head_n(P, "pol", x, "action/stddev", 3)
        return m, sd, mx.zeros((*m.shape[:-1], 1), mx.float32)

    m, sd, z = heads(Pp)
    zi = mx.zeros(m.shape[:-1], mx.int32)
    _, aux = M.policy_loss(m, sd, z, a, zi, mx.zeros((m.shape[0], m.shape[1] - 1), mx.float32), w, 0.1, 1.0, ACTENT)
    lp = (np.array(aux["logp_c"]) + np.array(aux["logp_d"]))[:, :-1].astype(np.float64)
    en = (np.array(aux["ent_c"]) + np.array(aux["ent_d"]))[:, :-1].astype(np.float64)
    L = G[f"c{c}/imagloss/loss/policy"].astype(np.float64)
    wv = w32[:, :-1].astype(np.float64)
    live = wv > 0
    adv = np.zeros_like(L)
    adv[live] = -(L[live] / wv[live] + ACTENT * en[live]) / lp[live]
    rebuilt = wv * -(lp * adv + ACTENT * en)
    res = np.abs(rebuilt - L).max() / max(np.abs(L).max(), 1e-30)
    print(f"call {c}: live {int(live.sum())}/{live.size}, |logpi| min {np.abs(lp[live]).min():.2e}, |adv_off| mean "
          f"{np.abs(adv[live]).mean():.3e}, back-solve residual (max |rebuilt-L| / max|L|) {res:.1e}")

    def grads(adv_np):
        def loss(P):
            mm, ss, zz = heads(P)
            pl, _ = M.policy_loss(mm, ss, zz, a, zi, mx.array(adv_np.astype(np.float32)), w, 0.1, 1.0, ACTENT)
            return pl.mean()
        return {f"c{c}/grad/{k}": np.array(v) for k, v in mx.grad(loss)(Pp).items()}

    port_gate.report(f"call {c} pol adv=official", gold, grads(adv))
    port_gate.report(f"CONTROL call {c} pol adv=official rolled 1 step", gold, grads(np.roll(adv, 1, axis=1)))
