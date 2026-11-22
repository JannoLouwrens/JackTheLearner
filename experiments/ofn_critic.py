"""Output feature normalisation (OFN) for brain B's critic -- J17 hunt h2.

Published recipe (PUBLISHED_PRIORS §24 arm (c)): Hussing et al. 2024,
"Dissecting Deep RL with High Update Ratios: Combatting Value Divergence"
(arXiv 2403.05996, main.tex §"Limiting gradient explosion via unit ball
normalization", read from the e-print source 11-22): "project the output
features of the critic encoder onto the unit ball using the function
f(x) = x / ||x||_2 ... where x is the output of our encoder phi(s, a)";
"We solely apply OFN to the critic"; "a one-line change in implementation".
Bjorck et al. 2022 (2110.11222) is the earlier penultimate-normalisation
source they cite.

This is sbx 0.x's own ContinuousCritic / VectorCritic (jack-std venv), copied,
with that one line before the output Dense. Differences from Hussing's recipe
(stated, not cured):
  * our critic keeps its LayerNorm per hidden layer (theirs is a plain 2x256
    ReLU MLP) -- so only ONE thing moves against J17's K-row128;
  * x is taken AFTER the last hidden ReLU (our reading of "output of our
    encoder"); the paper's placement relative to the activation is not
    spelled out beyond that;
  * a numerical guard: ||x|| is sqrt(sum x^2 + 1e-12), so an all-zero ReLU
    output gives 0 features instead of NaN (not in their formula);
  * update-to-data ratio 1 (theirs: high UTD; the paper's subject).
"""
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from sbx.common.policies import Flatten

EPS2 = 1e-12


def unit_ball(x):
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + EPS2)


class OFNContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        x = unit_ball(x)                     # Hussing 2024 OFN: the one line
        return nn.Dense(self.output_dim)(x)


class OFNVectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        vmap_critic = nn.vmap(
            OFNContinuousCritic,
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


CRITICS = {"ofn": OFNVectorCritic}
