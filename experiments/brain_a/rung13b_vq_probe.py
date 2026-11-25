"""Rung 13b-imagloss-vq first-read diagnosis (11-25): where is the port's value-loss error vs the f64 truth born?
Splits V into its pieces (logsumexp, the two twohot NLLs) and compares each f32 piece to f64 at call 2/6/8."""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TMPDIR", "/data/tmp")
import mlx.core as mx  # noqa: E402
import mlx_maths as M  # noqa: E402
from rung13b import GOLDEN_A, R13  # noqa: E402

G, R = np.load(GOLDEN_A), np.load(R13)
b = M.bins(255)
q = lambda e: "/".join(f"{np.percentile(np.abs(e), k):.1e}" for k in (50, 90, 99))  # noqa: E731
for c in (2, 6, 8):
    P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}  # noqa: E731
    g = f"c{c}/imag/"
    d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
    s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
    x = M.feat2tensor(mx.array(d), mx.array(s))
    lv = M._mlp_head_n(P("val"), "val", x, "logits", 3)
    L = np.array(lv, np.float64)
    m_ = L.max(-1, keepdims=True)
    lse64 = (m_ + np.log(np.exp(L - m_).sum(-1, keepdims=True)))[..., 0]
    lse_mx = np.array(mx.logsumexp(lv, axis=-1))
    Lf = np.array(lv)
    mf = Lf.max(-1, keepdims=True)
    lse_jaxstyle = (mf + np.log(np.exp(Lf - mf).sum(-1, keepdims=True, dtype=np.float32)))[..., 0]
    print(f"call {c}: |lse| median {np.median(np.abs(lse64)):.2f}; lse err vs f64: mx.logsumexp {q(lse_mx - lse64)}"
          f"  numpy-f32 max-shift {q(lse_jaxstyle - lse64)}")
    lsm_mx = np.array(lv - mx.logsumexp(lv, axis=-1, keepdims=True))
    lsm_ls = np.array(mx.log(mx.softmax(lv, axis=-1)))
    lsm_sub = np.array((lv - mx.max(lv, -1, keepdims=True)) - mx.log(mx.sum(mx.exp(lv - mx.max(lv, -1, keepdims=True)), -1, keepdims=True)))
    ref = L - lse64[..., None]
    print(f"        log_softmax err (all 255 bins): L-lse {q(lsm_mx - ref)}  shift-first {q(lsm_sub - ref)}  log(softmax) {q(lsm_ls - ref)}")

# Step-by-step at call 2 (11-25 22:35): which primitive carries the lse error?
c = 2
P = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/val/")}
g = f"c{c}/imag/"
d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
lv = M._mlp_head_n(P, "val", M.feat2tensor(mx.array(d), mx.array(s)), "logits", 3)
Lf = np.array(lv)
mf = Lf.max(-1, keepdims=True)
sh = (Lf - mf).astype(np.float32)                       # exact in f32 (Sterbenz) for these ranges
print("step: shifted range", float(sh.min()), float(sh.max()), "; |L| median", float(np.median(np.abs(Lf))))
e_mx = np.array(mx.exp(mx.array(sh))).astype(np.float64)
e64 = np.exp(sh.astype(np.float64))
print("step: exp rel err", q((e_mx - e64) / e64))
S64 = e64.sum(-1)
S_mx = np.array(mx.sum(mx.array(e64.astype(np.float32)), -1)).astype(np.float64)
print("step: sum rel err (exact-ish inputs)", q((S_mx - S64) / S64))
lg = np.array(mx.log(mx.array(S64.astype(np.float32)))).astype(np.float64)
print("step: log abs err", q(lg - np.log(S64)), " log(S) median", float(np.median(np.log(S64))))
tot = mf[..., 0].astype(np.float64) + np.log(S64)
print("step: f32 rounding of lse itself (half-ulp floor)", q(tot.astype(np.float32).astype(np.float64) - tot))
