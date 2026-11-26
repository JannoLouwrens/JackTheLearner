"""Probe (not a test) for wheel 13's rung13b_grads FAIL (J25, 11-26): is the policy-gradient miss on x86 mlx
f32 softmax rounding of the value head's tail probabilities?
Why it could be: the 255 two-hot bins reach |4.85e8| while |val| is ~1e-3 at call 2, so ONE ulp on one tail
probability swings the decoded value by up to 0.226 (box measurement, 11-26); the policy gradient is linear in
adv ~3e-4. On the box mlx's f32 softmax equals the correctly rounded f32 of the f64 softmax (decode |mlx - f64|
5.7e-4 = |rounded - f64|); the wheel read |val32 - val64| 0.58 at call 2.
Prints, per call: ulp distance of mlx f32 softmax vs the correctly rounded f64 softmax (max, count of probs off),
|val_mlx - val_rounded| and |val_mlx - val_f64|. If the wheel's probs sit a few ulp off and val_mlx - val_rounded
carries the 0.58, the reading is rounding, not the port's maths.
  /data/venvs/jack-mlx/bin/python -I experiments/brain_a/rung13b_softmax_ulp_probe.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TMPDIR", "/data/tmp")
import mlx.core as mx  # noqa: E402
import mlx_maths as M  # noqa: E402
from rung13b import GOLDEN_A, R13, CALLS  # noqa: E402

G, R = np.load(GOLDEN_A), np.load(R13)
b = M.bins(255)
b64 = np.array(b).astype(np.float64)
b32 = b64.astype(np.float32)


def dec(p, bb, dt):
    p, bb, m = p.astype(dt), bb.astype(dt), 127
    return p[..., m] * bb[m] + ((p[..., m + 1:] - p[..., :m][..., ::-1]) * bb[m + 1:]).sum(-1, dtype=dt)


print(f"max |bin| {np.abs(b64).max():.6e}; mlx {mx.__version__}")
for c in CALLS:
    g = f"c{c}/imag/"
    d = np.concatenate([G[g + "start/deter"][:, None], G[g + "feat/deter"]], 1)
    s = np.concatenate([G[g + "start/stoch"][:, None], G[g + "feat/stoch"]], 1)
    Pv = {k[len(f"c{c}/pre/"):]: mx.array(R[k]) for k in R.files if k.startswith(f"c{c}/pre/val/")}
    x = M.feat2tensor(mx.array(d), mx.array(s))
    lv = M._mlp_head_n(Pv, "val", x, "logits", 3)
    v_mlx = np.array(M.twohot_decode(lv, b)).astype(np.float64)
    L = np.array(lv).astype(np.float64)
    L -= L.max(-1, keepdims=True)
    p64 = np.exp(L)
    p64 /= p64.sum(-1, keepdims=True)
    pr = p64.astype(np.float32)
    pm = np.array(mx.softmax(lv.astype(mx.float32), axis=-1))
    ulp = np.abs(pm.view(np.int32).astype(np.int64) - pr.view(np.int32).astype(np.int64))
    v_rnd = dec(pr, b32, np.float32).astype(np.float64)
    v64 = dec(p64, b64, np.float64)
    print(f"call {c}: softmax ulp vs rounded-f64 max {ulp.max()} off {int((ulp > 0).sum())} of {ulp.size} | "
          f"|val_mlx - val_rounded| max {np.abs(v_mlx - v_rnd).max():.3e} | |val_mlx - val_f64| max "
          f"{np.abs(v_mlx - v64).max():.3e} | |val_rounded - val_f64| max {np.abs(v_rnd - v64).max():.3e} | "
          f"|val| max {np.abs(v64).max():.3e}", flush=True)
