"""MLX port of DreamerV3 — rung 1's maths module (PORT report step 1.3). STUB.

Step 1.2 (this commit) writes rung 1's tests (`rung1.py`) while every function
here raises NotImplementedError, so the tests exist and FAIL. Step 1.3 fills
these in until `rung1.py` passes eager AND compiled.

The contract each function must meet is the official code at e01491f
(`/data/src/dreamerv3_e01491f`), restated as NumPy float64 in `rung1.py`
(`REF_*`) and checked against the official JAX functions by
`rung1.py --vs-official`. Inputs and outputs are `mx.array` float32.

Ours beside theirs (filled in at step 1.3, one row per deviation):

  | item | official (e01491f) | port | why |
  |------|--------------------|------|-----|
  | (none yet) | | | |
"""
from __future__ import annotations


def _todo(*_a, **_k):
    raise NotImplementedError("rung 1 maths: PORT step 1.3")


# nets.py L59–64
symlog = _todo
symexp = _todo
# heads.py L132–144: symexp(linspace(-20, 0, 128)) mirrored, 255 bins, float32
bins = _todo                 # bins(n: int) -> mx.array (n,)
# outs.py L311–330: two-hot target weights in RAW space, then -sum t*log_softmax
twohot_target = _todo        # twohot_target(target (...,), bins (n,)) -> (..., n)
twohot_loss = _todo          # twohot_loss(logits (..., n), target (...,), bins) -> (...,)
# outs.py L285–309: symmetric sum, exactly 0 for uniform / mirrored probs
twohot_decode = _todo        # twohot_decode(logits (..., n), bins) -> (...,)
# outs.py L210–217: log((1-mix)*softmax + mix/K)
unimix_logits = _todo        # unimix_logits(logits (..., K), mix) -> (..., K)
# rssm.py L120–133 + L173–176: OneHot(unimix) KL summed over latents, max(., free)
kl_dyn_rep = _todo           # kl_dyn_rep(post (..., L, K), prior (..., L, K), mix, free) -> (dyn, rep)
# outs.py L265–270: onehot(index) + p - sg(p)
straight_through = _todo     # straight_through(index (...,) int, logits (..., K)) -> (..., K)
# heads.py L146–155: Normal(tanh(m), (hi-lo)*sigmoid(s+2)+lo)
bounded_normal = _todo       # bounded_normal(m, s, lo, hi) -> (mean, std)
# outs.py L161–176
normal_logp = _todo          # normal_logp(event, mean, std) -> elementwise
normal_entropy = _todo       # normal_entropy(std) -> elementwise
# outs.py L189–199
binary_logp = _todo          # binary_logp(logit, event) -> elementwise
