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
# Filled 11-20 after `rung2.py` was committed failing against stubs (d5d840d1).
# Ours beside theirs:
#   percentile  MLX has none: flat sort + linear interpolation at rank
#               q/100*(n-1), the rank and weight computed in Python float64
#               (shapes are static) — equals jnp.percentile under jax.jit
#               (the eager JAX call misses q=100 by one ulp of the rank;
#               the agent is compiled, so the compiled value is the target).
#   everything else line for line from the cited official lines.

# agent.py L482-490
def lambda_return(last, term, rew, val, boot, disc, lam):
    rets = [boot[:, -1]]
    live = (1 - term.astype(f32))[:, 1:] * disc
    cont = (1 - last.astype(f32))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return mx.stack(list(reversed(rets))[:-1], 1)


# utils.py L84-88: jnp.percentile, method 'linear'
def percentile(x, q):
    s = mx.sort(x.astype(f32).reshape(-1))
    n = s.shape[0]
    rank = float(q) / 100.0 * (n - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, n - 1)
    w = rank - lo
    if w == 0.0:
        return s[lo]
    return s[lo] * (1.0 - w) + s[hi] * w


# utils.py L44-56, L90-91 (impl 'perc')
def perc_update(lo, hi, corr, x, rate, perclo, perchi, debias):
    x = mx.stop_gradient(x.astype(f32))
    lo = (1 - rate) * lo + rate * mx.stop_gradient(percentile(x, perclo))
    hi = (1 - rate) * hi + rate * mx.stop_gradient(percentile(x, perchi))
    if debias:
        corr = (1 - rate) * corr + rate * 1.0
    return lo, hi, corr


# utils.py L58-73 (impl 'perc')
def perc_stats(lo, hi, corr, rate, limit, debias):
    c = 1.0 / mx.maximum(rate, corr) if debias else 1.0
    lo, hi = lo * c, hi * c
    return mx.stop_gradient(lo), mx.stop_gradient(mx.maximum(limit, hi - lo))


# utils.py L113-119
def slow_update(src, dst, count, rate, every):
    mix = mx.where(count % every == 0, rate, 0.0)
    return mix * src + (1 - mix) * dst, count + 1


# ---------------------------------------------------------------- rung 3 (step 1.5)
# agent.py L342-380 `_make_opt` + embodied/jax/opt.py: clip_by_agc(agc, pmin
# 1e-3) -> scale_by_rms(beta2, eps) -> scale_by_momentum(beta1) -> (wd 0) ->
# optax.scale_by_learning_rate(warm-up). state = (step, nu dict, mu dict),
# step an int32 mx.array (counts updates done; optax's schedule count = step).
# Ours beside theirs:
#   | warm-up lr | optax linear_schedule (count/steps, (init-end)*frac+end) | `_ramp`: fma(-lr, fma(-count, f32(1/warmup), 1), lr), each fma one float64 rounding | what XLA:CPU runs under jit (rung3.ramp, measured bit-exact); the agent trains jitted |
#   | pow        | jnp `f32(beta) ** step` | mx.power(f32(beta), f32(step)) | MLX has no integer_pow; read to 1e-6 by rung 3 |

_PMIN = 1e-3


def _ramp(lr, warmup, count):
    if not warmup:
        return mx.array(lr, f32)
    L = mx.array(lr, f32).astype(mx.float64, stream=mx.cpu)
    r = (mx.array(1.0, f32) / mx.array(float(warmup), f32)).astype(mx.float64, stream=mx.cpu)
    c = mx.minimum(count, warmup).astype(f32).astype(mx.float64, stream=mx.cpu)
    frac = mx.subtract(1.0, c * r, stream=mx.cpu).astype(f32, stream=mx.cpu)
    lr_t = (-L * frac.astype(mx.float64, stream=mx.cpu) + L).astype(f32, stream=mx.cpu)
    return mx.where(count < warmup, lr_t, mx.array(lr, f32))


def opt_init(params):
    z = {k: mx.zeros(v.shape, f32) for k, v in params.items()}
    return (mx.array(0, mx.int32), z, {k: mx.zeros(v.shape, f32) for k, v in params.items()})


def opt_update(params, grads, state, lr, warmup, agc, eps, beta1, beta2):
    """-> (params', state', updates); updates = the lr-scaled step, pre-store."""
    count, nu, mu = state
    step = count + 1
    lr_t = _ramp(lr, warmup, count)
    # opt.py casts (1 - decay) from Python float64 to f32; bias correction in f32
    b1, b2 = mx.array(beta1, f32), mx.array(beta2, f32)
    a1, a2 = mx.array(1 - beta1, f32), mx.array(1 - beta2, f32)
    bc1 = 1 - mx.power(b1, step.astype(f32))
    bc2 = 1 - mx.power(b2, step.astype(f32))
    newp, nnu, nmu, upd = {}, {}, {}, {}
    for k in params:
        p, u = params[k], grads[k]
        # clip_by_agc: u * 1/max(1, |u| / (clip * max(pmin, |p|)))
        un = mx.sqrt(mx.sum(u * u))
        pn = mx.sqrt(mx.sum(p * p))
        u = u * (1 / mx.maximum(1, un / (agc * mx.maximum(_PMIN, pn))))
        nnu[k] = b2 * nu[k] + a2 * u * u
        u = u / (mx.sqrt(nnu[k] / bc2) + eps)
        nmu[k] = b1 * mu[k] + a1 * u
        u = nmu[k] / bc1
        upd[k] = -lr_t * u
        newp[k] = p + upd[k]
    return newp, (step, nnu, nmu), upd


# ---------------------------------------------------------------- rung 6: one RSSM step
# Contract: `rung6.py` REF_step (NumPy float64), itself checked 12/12 against
# the official rssm.py `_observe`/`_core`/`_prior` at e01491f by
# `rung6.py --vs-official`. Done-test: `rung6.py` exits 0, eager AND compiled.
def rssm_step(P, carry, tokens, action, reset, cfg, act_spec):
    """P: {name: mx.array} official param names (without the 'rssm/' prefix);
    carry: dict(deter [B,D], stoch [B,S,C]); tokens [B,T]; action: {key: array};
    reset [B] bool; act_spec: {key: classes (int>0) or 0 for continuous}.
    -> dict(masked_deter, actemb, deter, post, prior) — post/prior are logits
    [B,S,C]; the posterior SAMPLE is rung 7's."""
    g = cfg["blocks"]
    keep = mx.logical_not(reset).astype(f32)
    # rssm.py L72-90 _observe: nets.mask = where(~reset, x, 0)
    deter = carry["deter"] * keep[:, None]
    stoch = carry["stoch"] * keep[:, None, None]
    parts = []
    for k in sorted(act_spec):                    # nets.DictConcat: sorted keys
        a = action[k]
        if act_spec[k]:
            a = (mx.arange(act_spec[k])[None, :] == a[:, None]).astype(f32)
        parts.append(a.astype(f32))
    actemb = mx.concatenate(parts, -1) * keep[:, None]
    deter2 = _core(P, deter, stoch, actemb, cfg)
    x = mx.concatenate([deter2, tokens.astype(f32)], -1)
    for i in range(cfg["obslayers"]):
        x = _layer(P, f"obs{i}", x, cfg["act"])
    post = _lin(P, "obslogit", x).reshape(x.shape[0], cfg["stoch"], cfg["classes"])
    x = deter2                                     # rssm.py L164-169 _prior
    for i in range(cfg["imglayers"]):
        x = _layer(P, f"prior{i}", x, cfg["act"])
    prior = _lin(P, "priorlogit", x).reshape(x.shape[0], cfg["stoch"], cfg["classes"])
    return dict(masked_deter=deter, actemb=actemb, deter=deter2, post=post, prior=prior)


# ---------------------------------------------------------------- rung 7: 64-step memory pass
# Contract: `rung7.py` REF_observe (NumPy float64), itself checked 15/15 against
# the official RSSM.observe scan at e01491f by `rung7.py --vs-official`.
# Done-test: `rung7.py` exits 0, eager AND compiled.
def rssm_observe(P, carry, tokens, action, reset, uniforms, cfg, act_spec):
    """tokens [B,T,K]; action {key: [B,T,...]}; reset [B,T] bool; uniforms [T,B,S,C] =
    the uniforms the step-t posterior sampler reads (prerun draws already skipped).
    -> dict(deter [B,T,D], post, prior, stoch [B,T,S,C]) — stoch one-hot samples."""
    T = tokens.shape[1]
    mix = cfg["unimix"]
    out = {k: [] for k in ("deter", "post", "prior", "stoch")}
    for t in range(T):                             # nj.scan over axis 1 (rssm.py L58-70)
        r = rssm_step(P, carry, tokens[:, t], {k: v[:, t] for k, v in action.items()},
                      reset[:, t], cfg, act_spec)
        st = sample_onehot(r["post"], uniforms[t], mix)
        carry = dict(deter=r["deter"], stoch=st)
        for k, v in (("deter", r["deter"]), ("post", r["post"]), ("prior", r["prior"]), ("stoch", st)):
            out[k].append(v)
    return {k: mx.stack(v, 1) for k, v in out.items()}


def rssm_loss(P, carry, tokens, action, reset, uniforms, cfg, act_spec, free):
    """rssm.py L120-133: observe scan, prior = `_prior(feat['deter'])` over the pass (the scan's
    per-step prior is rssm_step's `_prior` on the step's new deter — reused, not recomputed), then
    dyn = KL(sg(post) || prior), rep = KL(post || sg(prior)), each max(., free) per [B,T]
    element. Ours beside theirs: `free=0` skips nothing (max(kl, 0) == kl since KL >= 0); the
    official `if self.free_nats:` skips the max — same values. dyn_ent/rep_ent metrics not ported."""
    out = rssm_observe(P, carry, tokens, action, reset, uniforms, cfg, act_spec)
    dyn, rep = kl_dyn_rep(out["post"], out["prior"], cfg["unimix"], free)
    return dict(dyn=dyn, rep=rep, **out)


def sample_onehot(logit, u, mix):
    """outs.OneHot(logit, unimix).sample with file noise: argmax(log((1-mix)softmax + mix/C)
    + Gumbel(u)), u clipped to [1e-7, 1-1e-7] as file_noise.py. Ours beside theirs: the
    official draws Gumbel from jax.random (here from the golden's uniforms); no
    straight-through gradient (no training at this rung)."""
    p = mx.softmax(logit.astype(f32), -1)
    if mix:
        p = (1.0 - mix) * p + mix / logit.shape[-1]
    g = -mx.log(-mx.log(mx.clip(u.astype(f32), 1e-7, 1.0 - 1e-7)))
    idx = mx.argmax(mx.log(p) + g, -1)
    return (mx.arange(logit.shape[-1]) == idx[..., None]).astype(f32)


_GELU_C = math.sqrt(2.0 / math.pi)


def _gelu(x):                                      # jax.nn.gelu default approximate=True
    return 0.5 * x * (1.0 + mx.tanh(_GELU_C * (x + 0.044715 * x * x * x)))


def _silu(x):                                      # jax.nn.silu = x * sigmoid(x)
    return x * mx.sigmoid(x)


# nets.act by name. Ours beside theirs: official nets.Linear/BlockLinear read `act` from
# the config (defaults' silu; the toy rungs set gelu); the port used to hardcode gelu, which
# failed rung 7 at size1m with 406 flips (11-24) — it now reads cfg["act"] like rung6.ACT.
_ACT = {"gelu": _gelu, "silu": _silu}


def _rms(x, scale):                                # nets.Norm 'rms', eps 1e-4, in f32
    return x * mx.rsqrt(mx.mean(x * x, -1, keepdims=True) + 1e-4) * scale


def _lin(P, name, x):
    return x @ P[f"{name}/kernel"] + P[f"{name}/bias"]


def _blocklin(P, name, x, g):                      # nets.BlockLinear: kernel [g, in/g, out/g]
    k = P[f"{name}/kernel"]
    lead = x.shape[:-1]
    xg = x.reshape(*lead, g, x.shape[-1] // g)
    y = mx.einsum("...ki,kio->...ko", xg, k).reshape(*lead, -1)
    return y + P[f"{name}/bias"]


def _layer(P, name, x, act):
    return _ACT[act](_rms(_lin(P, name, x), P[f"{name}norm/scale"]))


def _core(P, deter, stoch, action, cfg):          # rssm.py L139-162
    g = cfg["blocks"]
    lead = deter.shape[:-1]
    stoch = stoch.reshape(stoch.shape[0], -1)
    action = action / mx.maximum(1.0, mx.abs(action))
    a = cfg["act"]
    x = mx.concatenate([_layer(P, "dynin0", deter, a), _layer(P, "dynin1", stoch, a),
                        _layer(P, "dynin2", action, a)], -1)
    x = mx.repeat(mx.expand_dims(x, -2), g, -2)
    x = mx.concatenate([deter.reshape(*lead, g, -1), x], -1).reshape(*lead, -1)
    for i in range(cfg["dynlayers"]):
        x = _ACT[a](_rms(_blocklin(P, f"dynhid{i}", x, g), P[f"dynhid{i}norm/scale"]))
    x = _blocklin(P, "dyngru", x, g)
    xg = x.reshape(*lead, g, -1)
    n = xg.shape[-1] // 3                           # gates split PER GROUP
    r, cand, u = (xg[..., j * n:(j + 1) * n].reshape(*lead, -1) for j in range(3))
    r = mx.sigmoid(r)
    cand = mx.tanh(r * cand)
    u = mx.sigmoid(u - 1.0)
    return u * cand + (1.0 - u) * deter


# ---------------------------------------------------------------- rung 8b (step 2.4, row 8)
# Reward + continue heads and their losses (agent.py L51-58, L172-177; heads.py MLPHead;
# outs.py TwoHot / Binary). Filled 11-25 after `rung8b.py` was committed failing (9db7bb98).
# Ours beside theirs:
#   compute dtype f32 (published bf16), as rungs 6-8a.
#   one MLP layer read by name `{prefix}/mlp/linear0`, `{prefix}/mlp/norm0` — the published
#     rewhead/conhead have layers 1; deeper heads are not ported (no row needs them yet).
#   outscale is an init setting, not a forward one: the port reads whatever kernel it is
#     given (published init's zero reward kernel gives the zero logits / log 255 the test asserts).

# agent.py L172-173: feat2tensor = concat(deter, stoch flattened)
def feat2tensor(deter, stoch):
    return mx.concatenate([deter, stoch.reshape(*stoch.shape[:-2], -1)], -1)


# agent.py L57: con = f32(~is_terminal) * (1 - 1/horizon)
def con_target(term, horizon):
    return (1.0 - term.astype(f32)) * (1.0 - 1.0 / horizon)


def _mlp_head(P, prefix, x, out):
    x = _ACT["silu"](_rms(_lin(P, f"{prefix}/mlp/linear0", x), P[f"{prefix}/mlp/norm0/scale"]))
    W, b = P[f"{prefix}/head/{out}/kernel"], P[f"{prefix}/head/{out}/bias"]
    return x @ W.reshape(W.shape[0], -1) + b.reshape(-1)


def rewcon_loss(P, deter, stoch, reward, term, horizon):
    """agent.py L51-58: rew = -TwoHot(rewhead(feat), symexp bins 255).logp(reward);
    con = -Binary(conhead(feat)).logp(con_target). Returns per-[B,T] losses and the logits."""
    feat = feat2tensor(deter.astype(f32), stoch.astype(f32))
    rl = _mlp_head(P, "rew", feat, "logits")
    cl = _mlp_head(P, "con", feat, "logit")[..., 0]
    return dict(rew=twohot_loss(rl, reward.astype(f32), bins(255)),
                con=-binary_logp(cl, con_target(term, horizon)), rew_logits=rl, con_logit=cl)


# ---------------------------------------------------------------- rung 8c (step 2.4, row 8)
# Decoder VECTOR branch and its losses (rssm.py L253-306; heads.py symlog_mse + Agg; outs.py MSE;
# agent.py L178-181). Filled 11-25 after `rung8c.py` was committed failing (963d32f1).
# Ours beside theirs:
#   compute dtype f32 (published bf16), as rungs 6-8b.
#   continuous keys only ('symlog_mse'); the 'categorical' branch for discrete vector keys and
#     the whole image branch (BlockLinear + conv) are not ported — no row needs them yet.
#   the sg() on the target is a no-op here (no gradient is taken in this rung).

# rssm.py L293-295: the decoder concatenates STOCH first, then deter (feat2tensor is the reverse)
def dec_inp(deter, stoch):
    return mx.concatenate([stoch.reshape(*stoch.shape[:-2], -1), deter], -1)


# heads.py L89 Agg(output, len(space.shape), sum); a () space has no Agg
def _vec_agg(x, axes):
    return mx.sum(x, axes) if axes else x


def dec_vec_loss(P, deter, stoch, targets, spaces, layers):
    """Per key: square(Linear_k(MLP(dec_inp)) - symlog(target)), summed over the key's dims ->
    [B,T]; also "pred/<key>" [B,T,*shape]. MLP = layers x (linear -> rms -> silu), act silu."""
    x = dec_inp(deter.astype(f32), stoch.astype(f32))
    for i in range(layers):
        x = _ACT["silu"](_rms(_lin(P, f"dec/mlp/linear{i}", x), P[f"dec/mlp/norm{i}/scale"]))
    out = {}
    for k, shp in spaces.items():
        W, b = P[f"dec/vec/{k}/pred/kernel"], P[f"dec/vec/{k}/pred/bias"]
        pred = (x @ W.reshape(W.shape[0], -1) + b.reshape(-1)).reshape(*x.shape[:-1], *shp)
        sq = mx.square(pred - symlog(targets[k].astype(f32)))
        out[k] = _vec_agg(sq, list(range(2, sq.ndim)))
        out[f"pred/{k}"] = pred
    return out
