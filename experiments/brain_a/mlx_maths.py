"""MLX port of DreamerV3 — rung 1's maths module (PORT report step 1.3).

The contract each function must meet is the official code at e01491f
(`/data/src/dreamerv3_e01491f`), restated as NumPy float64 in `rung1.py`
(`REF_*`) and checked against the official JAX functions by
`rung1.py --vs-official` (25/25). Inputs and outputs are `mx.array` float32.
Done-test: `rung1.py` exits 0 — 9/9 tests eager AND under `mx.compile`.

Every function is a line-for-line port of the official line cited above it.
Ours beside theirs (one row per deviation; anything not listed is the same
formula in the same order):

  | item | official (e01491f) | port | why |
  |------|--------------------|------|-----|
  | two-hot decode | symmetric sum `(p1*b1)[::-1] + p3*b3` (outs.py L298–302) | `(p3 - p1[::-1]) * b3`, the mirrored probabilities cancelled BEFORE the multiply | the official form is exactly 0 eager but +0.164 under `mx.compile` (−0.17 under `jax.jit`, issue #245) on mirrored logits — measured 11-19 by `rung1.py --planted`'s probe. Identical in exact arithmetic because the bins are mirrored bit for bit (b1[::-1] == −b3, asserted by `bins` and by the test); the difference of two equal float32 numbers is exactly 0 whatever the compiler fuses |
  | bins grid | `jnp.linspace(-20, 0, 128, dtype=f32)` | `_jnp_linspace`: start·fma(−i, f32(1/div), 1) + stop·i·f32(1/div), the fma done as one float64 rounding on the CPU stream | what XLA:CPU actually runs (iota/div → iota·reciprocal, then LLVM fma contraction). `mx.linspace` and the jaxpr computed op by op each miss JAX's grid at 41/128 points (1 ulp of 20) → bins 1.92e-6 off official, over 1e-6; this form misses 0 |
  | two-hot decode | `self.unsquash(wavg)` | no unsquash argument | the reward/value heads call TwoHot with the default identity (heads.py L141); squashing lives in the bins |
  | binary logp | `jax.nn.log_sigmoid(l)` | `-logaddexp(0, -l)` | MLX 0.32 has no log_sigmoid; this is the same stable softplus form JAX uses (`-softplus(-x)`) |
  | normal logp | `jax.scipy.stats.norm.logpdf` | the closed form `-z²/2 - log σ - log √(2π)` | MLX has no scipy.stats; same formula (rung1's REF checks it to 1e-6) |
  | straight-through | `jax.nn.one_hot(index, K)` | `arange(K) == index[..., None]` cast to float32 | MLX has no one_hot; same values |
"""
from __future__ import annotations

import math

import mlx.core as mx

f32 = mx.float32
_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


# nets.py L59–64
def symlog(x):
    return mx.sign(x) * mx.log1p(mx.abs(x))


def symexp(x):
    return mx.sign(x) * mx.expm1(mx.abs(x))


def _jnp_linspace(start, stop, num):
    """`jnp.linspace(start, stop, num, dtype=f32)` bit for bit as XLA:CPU runs
    it at JAX 0.4.33. The jaxpr is start*(1 - iota/div) + stop*(iota/div); XLA
    rewrites iota/div as iota * f32(1/div) and LLVM contracts 1 - iota*r into
    one fused multiply-add, so each point is start * fma(-iota, r, 1) (+ stop *
    iota*r). Measured 11-19 on the 128-point grid: the jaxpr computed op by op
    misses JAX's grid at 41 points, `mx.linspace` at 41, this form at 0 —
    and those 1-ulp-of-20 misses become 1.92e-6 on the bins, over rung 1's
    1e-6. MLX has no elementwise fma: an f32 x f32 product is exact in
    float64, so 1 - iota*r is rounded once (float64 on the CPU stream; the
    bins are built once at init, as in the official head)."""
    div = num - 1
    r = mx.array(1.0 / div, f32).astype(mx.float64, stream=mx.cpu)
    i = mx.arange(div, dtype=f32).astype(mx.float64, stream=mx.cpu)
    one_minus = mx.subtract(1.0, i * r, stream=mx.cpu).astype(f32, stream=mx.cpu)
    stepf = mx.arange(div, dtype=f32) * mx.array(1.0 / div, f32)
    out = mx.array(start, f32) * one_minus + mx.array(stop, f32) * stepf
    return mx.concatenate([out, mx.array([stop], f32)], 0)


# heads.py L136–143: symexp(linspace(-20, 0, ., f32)), mirrored; float32
def bins(n: int = 255):
    if n % 2 == 1:
        half = symexp(_jnp_linspace(-20.0, 0.0, (n - 1) // 2 + 1))
        return mx.concatenate([half, -half[:-1][::-1]], 0)
    half = symexp(_jnp_linspace(-20.0, 0.0, n // 2))
    return mx.concatenate([half, -half[::-1]], 0)


def _onehot(index, k):
    return (mx.arange(k) == index[..., None]).astype(f32)


# outs.py L311–326: two-hot weights in RAW space (squash = identity)
def twohot_target(target, b):
    target = mx.stop_gradient(target.astype(f32))
    n = b.shape[-1]
    t = target[..., None]
    below = (b <= t).astype(mx.int32).sum(-1) - 1
    above = n - (b > t).astype(mx.int32).sum(-1)
    below = mx.clip(below, 0, n - 1)
    above = mx.clip(above, 0, n - 1)
    equal = below == above
    one = mx.array(1.0, f32)
    dist_to_below = mx.where(equal, one, mx.abs(b[below] - target))
    dist_to_above = mx.where(equal, one, mx.abs(b[above] - target))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    return (_onehot(below, n) * weight_below[..., None]
            + _onehot(above, n) * weight_above[..., None])


# outs.py L327–330
def twohot_loss(logits, target, b):
    logits = logits.astype(f32)
    w = twohot_target(target, b)
    log_pred = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(w * log_pred).sum(-1)


# outs.py L285–309, the odd/even split kept; see the table for the one change
def twohot_decode(logits, b):
    probs = mx.softmax(logits.astype(f32), axis=-1)
    n = logits.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1, p2, p3 = probs[..., :m], probs[..., m:m + 1], probs[..., m + 1:]
        b2, b3 = b[..., m:m + 1], b[..., m + 1:]
        return (p2 * b2).sum(-1) + ((p3 - p1[..., ::-1]) * b3).sum(-1)
    p1, p2 = probs[..., :n // 2], probs[..., n // 2:]
    b2 = b[..., n // 2:]
    return ((p2 - p1[..., ::-1]) * b2).sum(-1)


# outs.py L210–217: log((1-mix)*softmax + mix*uniform)
def unimix_logits(logits, mix):
    logits = logits.astype(f32)
    if not mix:
        return logits
    probs = mx.softmax(logits, axis=-1)
    uniform = mx.ones_like(probs) / probs.shape[-1]
    return mx.log((1 - mix) * probs + mix * uniform)


def _cat_kl(a, b):
    """outs.py L232–236: Categorical.kl on (already unimixed) logits."""
    logprob = a - mx.logsumexp(a, axis=-1, keepdims=True)
    logother = b - mx.logsumexp(b, axis=-1, keepdims=True)
    prob = mx.softmax(a, axis=-1)
    return (prob * (logprob - logother)).sum(-1)


# rssm.py L120–133 + L173–176: Agg(OneHot(., unimix), 1, sum); max(., free)
def kl_dyn_rep(post, prior, mix, free):
    sg = mx.stop_gradient
    p, q = unimix_logits(post, mix), unimix_logits(prior, mix)
    psg, qsg = unimix_logits(sg(post), mix), unimix_logits(sg(prior), mix)
    dyn = mx.maximum(_cat_kl(psg, q).sum(-1), free)
    rep = mx.maximum(_cat_kl(p, qsg).sum(-1), free)
    return dyn, rep


# outs.py L265–270: sg(onehot) + (probs - sg(probs))
def straight_through(index, logits):
    probs = mx.softmax(logits.astype(f32), axis=-1)
    value = mx.stop_gradient(_onehot(index, logits.shape[-1]))
    return value + (probs - mx.stop_gradient(probs))


# heads.py L146–155
def bounded_normal(m, s, lo, hi):
    return mx.tanh(m), (hi - lo) * mx.sigmoid(s + 2.0) + lo


# outs.py L161–176
def normal_logp(event, mean, std):
    z = (event.astype(f32) - mean) / std
    return -0.5 * z * z - mx.log(std) - _LOG_SQRT_2PI


def normal_entropy(std):
    return 0.5 * mx.log(2 * math.pi * mx.square(std)) + 0.5


# outs.py L189–199
def binary_logp(logit, event):
    event = event.astype(f32)
    logp = -mx.logaddexp(0.0, -logit)
    lognotp = -mx.logaddexp(0.0, logit)
    return event * logp + (1 - event) * lognotp


# ---------------------------------------------------------------- rung 2 (step 1.4)
# STUBS until the fill: tests (`rung2.py`) are committed while these FAIL.
# agent.py L482-490
def lambda_return(last, term, rew, val, boot, disc, lam):
    raise NotImplementedError("rung 2: lambda_return")


# utils.py L84-88: jnp.percentile, method 'linear' (MLX has none: sort + interpolate)
def percentile(x, q):
    raise NotImplementedError("rung 2: percentile")


# utils.py L44-56, L90-91 (impl 'perc')
def perc_update(lo, hi, corr, x, rate, perclo, perchi, debias):
    raise NotImplementedError("rung 2: perc_update")


# utils.py L58-73 (impl 'perc'): sg(lo * corr), sg(max(limit, (hi - lo) * corr))
def perc_stats(lo, hi, corr, rate, limit, debias):
    raise NotImplementedError("rung 2: perc_stats")


# utils.py L113-119
def slow_update(src, dst, count, rate, every):
    raise NotImplementedError("rung 2: slow_update")
