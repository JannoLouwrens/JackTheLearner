"""A separate first layer per sense before fusion, for brain B's critic --
J17 hunt h3 (built 11-22 while h2 runs; no run yet).

Published source (PUBLISHED_PRIORS §24 arm (d)): Lee et al. 2018, "Making
Sense of Vision and Touch" (arXiv 1810.10191) -- each sense gets its own
encoder, fused into "a compact and multimodal representation"; DreamerV3
(2301.04104) "normalization, balancing" names the same concern. Abstracts
only were read (§24 "Not verified"); this is the arm's SHAPE, not their code.

Why this arm: §24's check of sbx's critic -- at init ~86% of every first-layer
unit's input variance is eye (256 of 298 inputs), and the LayerNorm comes
AFTER that mix, so it rescales the joint sum and cannot rebalance the blocks.
Here the first hidden layer is split: obs[:, :V] (vision; SENSES order in
jack_env puts vision first, 256 wide for row128) -> Dense(n/2) -> LayerNorm,
and everything else (audio, touch, proprio, needs + action) -> Dense(n/2) ->
LayerNorm; concat -> ReLU; then sbx's own remaining hidden layers and output
Dense unchanged. Same width (n = net_arch[0]), so only the first layer's
connectivity and its per-sense normalisation move against J17's K-row128.

Differences from the published designs (stated, not cured):
  * Lee 2018 uses a CNN for images, a causal conv for force/torque and a
    self-supervised fused representation; ours is ONE Dense layer per block,
    trained only by the critic's TD loss;
  * two blocks (vision / the rest), not one encoder per sense;
  * critic only; the actor is sbx's default (as OFN in h2);
  * the action joins the non-vision block;
  * per-block LayerNorm is our reading of "balancing" -- no source pins it.
An inventor's choice in its details; it runs as a hunt arm beside its own
blind twin, a pilot (one seed), never as a claim.
"""
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from sbx.common.policies import Flatten

VISION_DIMS = 256        # row128 eye width; jack_env SENSES puts vision first


class SplitContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1
    vision_dims: int = VISION_DIMS

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        v, rest = x[..., :self.vision_dims], x[..., self.vision_dims:]
        rest = jnp.concatenate([rest, action], -1)
        half = self.net_arch[0] // 2
        hv = nn.LayerNorm(name="ln_vision")(nn.Dense(half, name="enc_vision")(v))
        hr = nn.LayerNorm(name="ln_rest")(nn.Dense(self.net_arch[0] - half, name="enc_rest")(rest))
        x = self.activation_fn(jnp.concatenate([hv, hr], -1))
        for n_units in self.net_arch[1:]:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        return nn.Dense(self.output_dim)(x)


class SplitVectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        vmap_critic = nn.vmap(
            SplitContinuousCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        return vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            output_dim=self.output_dim,
        )(obs, action)
