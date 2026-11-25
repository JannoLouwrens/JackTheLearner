"""Probe for `rung13b.py --repval` read 1 (11-25): official repval sits ~1e-3 (q99) off our f64 truth at calls
2/6/8 while the port sits ~2e-5. Which teacher-forced input differs from the official's? Each variant recomputes the
f64 truth from the port's own logits and prints the official's |err| quantiles against it."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TMPDIR", "/data/tmp")
import mlx.core as mx  # noqa: E402
import mlx_maths as M  # noqa: E402
import rung13b as r  # noqa: E402

G, R = np.load(r.GOLDEN_A), np.load(r.R13)
bt = r.rebuild_batch()
b = M.bins(255)
b64 = np.array(b).astype(np.float64)


def lsm64(L):
    L = np.asarray(L, np.float64); m = L.max(-1, keepdims=True)
    return L - m - np.log(np.exp(L - m).sum(-1, keepdims=True))


def dec64(L):
    p = np.exp(lsm64(L))
    return p[..., 127] * b64[127] + ((p[..., 128:] - p[..., :127][..., ::-1]) * b64[128:]).sum(-1)


def nll64(L, tgt):
    n = b64.size
    t = np.asarray(tgt, np.float64)[..., None]
    lo = np.clip((b64 <= t).sum(-1) - 1, 0, n - 1); hi = np.clip(n - (b64 > t).sum(-1), 0, n - 1)
    eq = lo == hi; tt = t[..., 0]
    db = np.where(eq, 1.0, np.abs(b64[lo] - tt)); da = np.where(eq, 1.0, np.abs(b64[hi] - tt))
    lp = lsm64(L)
    return -(da / (db + da) * np.take_along_axis(lp, lo[..., None], -1)[..., 0]
             + db / (db + da) * np.take_along_axis(lp, hi[..., None], -1)[..., 0])


def lret64(rew, boot, disc=1 - 1 / 333, lam=0.95):
    live = np.full(rew[:, 1:].shape, disc); cont = np.full(rew[:, 1:].shape, lam)
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    rets = [boot[:, -1]]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


for c in (2, 6, 8):
    P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}  # noqa
    d, s = G[f"c{c}/outs/repfeat/deter"], G[f"c{c}/outs/repfeat/stoch"]
    x = M.feat2tensor(mx.array(d), mx.array(s))
    lv = np.array(M._mlp_head_n(P("val"), "val", x, "logits", 3))
    ls = np.array(M._mlp_head_n(P("slowval"), "slowval", x, "logits", 3))
    val, slow = dec64(lv), dec64(ls)
    ret0 = G[f"c{c}/imagloss/outs/ret"][:, 0].astype(np.float64)
    off = G[f"c{c}/outs/losses/repval"].astype(np.float64)
    rw = bt["reward"].astype(np.float64)
    V = {"base rew[1:] boot ret0(B,K)": (rw[:, 1:], ret0.reshape(16, 64)),
         "rew[:-1]": (rw[:, :-1], ret0.reshape(16, 64)),
         "boot (K,B).T": (rw[:, 1:], ret0.reshape(64, 16).T),
         "boot = val": (rw[:, 1:], val),
         "boot = slow": (rw[:, 1:], slow),
         "rew 0": (0 * rw[:, 1:], ret0.reshape(16, 64))}
    for name, (rr, bb) in V.items():
        t = nll64(lv[:, :-1], lret64(rr, bb)) + nll64(lv[:, :-1], slow[:, :-1])
        e = np.abs(off - t)
        print(f"c{c} {name:28s} official |err| q50/90/99 " + "/".join(f"{np.percentile(e, k):.1e}" for k in (50, 90, 99)))

# where the official's tail sits (base inputs), call 8
if "--where" in sys.argv:
    c = 8
    P = lambda h: {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/{h}/")}  # noqa
    x = M.feat2tensor(mx.array(G[f"c{c}/outs/repfeat/deter"]), mx.array(G[f"c{c}/outs/repfeat/stoch"]))
    lv = np.array(M._mlp_head_n(P("val"), "val", x, "logits", 3))
    ls = np.array(M._mlp_head_n(P("slowval"), "slowval", x, "logits", 3))
    slow = dec64(ls)
    ret = lret64(bt["reward"][:, 1:].astype(np.float64), G[f"c{c}/imagloss/outs/ret"][:, 0].reshape(16, 64).astype(np.float64))
    t1, t2 = nll64(lv[:, :-1], ret), nll64(lv[:, :-1], slow[:, :-1])
    off = G[f"c{c}/outs/losses/repval"].astype(np.float64)
    e = np.abs(off - t1 - t2)
    print("err by t (max over B), every 8th:", " ".join(f"{e[:, t].max():.1e}" for t in range(0, 63, 8)))
    print("err by b (max over t):", " ".join(f"{e[i].max():.1e}" for i in range(16)))
    i, t = np.unravel_index(e.argmax(), e.shape)
    print("worst", i, t, "ret", ret[i, t], "slow", slow[i, t], "nll ret", t1[i, t], "nll slow", t2[i, t], "off", off[i, t])
    # sensitivity: official ret error needed
    for dr in (1e-6, 1e-5, 1e-4):
        print(f"  d nll / ret+{dr:g}:", abs(nll64(lv[i:i+1, t:t+1], ret[i:i+1, t:t+1] + dr) - t1[i, t])[0, 0])
