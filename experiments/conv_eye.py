"""J22 arm h4: his ray eye fed the way the record feeds an image to a replay
learner -- a conv encoder + random shift (DrQ). Built 11-24, before any run.

Published source (PUBLISHED_PRIORS §12, §24): Kostrikov, Yarats & Fergus 2020,
"Image Augmentation Is All You Need" (DrQ, arXiv 2004.13649) and Yarats et al.
2021 (DrQ-v2, 2107.09645): SAC on raw pixels fails (SAC-pixel walker 42 vs
948 from state at 500k); a small conv encoder trained by the critic loss only,
with random shift of the image, repairs it (DrQ 921). The record's encoder is
2-D; his eye is ONE level row of 128 rays x 2 channels [distance/8 m,
material], interleaved per ray (`fast_sim._row_vision`: out[0::2] = d,
out[1::2] = t), azimuth 2*pi*k/128 -- a FULL CIRCLE. So it is fed as a 1-D
image of 128 positions x 2 channels, and the shift is a ROLL (a full circle
has no edge to replicate; rolling it is what turning on the spot does).

The arm (ORDERS J22 (1)):
  encoder  obs[:, :256] -> (B, 128, 2) -> Conv1d(32, k3, stride 2) -> ReLU ->
           3 x [Conv1d(32, k3, stride 1) -> ReLU] (VALID padding, 128 -> 63 ->
           61 -> 59 -> 57) -> flatten (1,824) -> Dense 50 -> LayerNorm -> tanh
  join     body/needs numbers (obs[:, 256:]) are concatenated AFTER the encoder
  critic   ONE encoder shared by both Q heads (outside the vmap, as DrQ's
           critic.encoder); each head = sbx's own ContinuousCritic on
           [features, rest, action], J17's net_arch [256, 256] + LayerNorm
  actor    sbx's own SquashedGaussianActor on [stop_gradient(enc(v)), rest];
           its copy of the encoder params is overwritten from the critic's
           before every actor update, and stop_gradient gives it zero grad,
           so Adam leaves it where the critic put it (tested)
  shift    in the critic update, obs and next_obs each get their OWN integer
           roll in [-6, 6] rays per sample; the actor update rolls its obs
           once and the actor and the critic read that same rolled obs. The
           body/needs numbers are never moved (tested). Acting is unshifted.

Ours beside theirs (DrQ, Kostrikov 2020, Table/Appendix; v2 where it differs):
  * image 84x84x(3 frames x 3 RGB)  | ours 128 rays x 2 channels, one frame
    (no frame stack: J17's arm K has none; his needs carry memory of nothing)
  * Conv2d 3x3                      | ours Conv1d k3 (the eye is 1-D)
  * pad 4 px + random crop 84 (=+-4)| ours roll +-6 of 128 rays (+-4.7 % of
    the row; the order's number; DrQ's +-4 of 84 = +-4.8 %)
  * K=2 target / M=2 current augs   | ours 1 + 1 (= DrQ-v2, and the order)
  * critic tau 0.01 (encoder inside) | ours sbx tau 0.005 for everything
    (J17 arm K's setting kept; ONE key changes against J17)
  * actor lr / critic lr 1e-3, batch 512 | ours J17's 3e-4,
    batch 256, 1 update a decision, gamma 0.995, auto alpha (J17 arm K)
  * obs scaling: pixels / 255       | ours VecNormalize per-input running
    mean/std (J17 arm K), so a roll mixes rays normalised with different
    statistics -- disclosed, not cured
  * target encoder = Polyak copy    | ours the same (it lives in the critic's
    target params, updated by sbx's soft_update)
  * actor: separate MLP, 1024 wide  | ours sbx's actor, J17's [256, 256]
An arm of a pilot (ORDERS J22 (3)), one seed, never a claim on its own.
Read 11-24 (ruling 101): NOT SHOWN -- one seed, 100k, twins 0.31 / 0.87
late meals a life against sighted 0.56; and a +-6-ray roll of a ring is a
TURN with the action unmoved (PUBLISHED_PRIORS §27), not DrQ's
optimality-invariant shift. Never read as "B does not learn from his eye".
(11-25: two DrQ settings the official config does not have -- an encoder
tau 0.05 and a walker lr 2e-4 -- were removed from the table above, §27.)
"""
from __future__ import annotations

from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from sbx import SAC
from sbx.common.policies import ContinuousCritic, Flatten, SquashedGaussianActor

N_RAYS, N_CH = 128, 2
VISION_DIMS = N_RAYS * N_CH     # vision is first in jack_env's SENSES order
SHIFT = 6
FEATURE_DIM = 50


class RayEncoder(nn.Module):
    @nn.compact
    def __call__(self, v: jnp.ndarray) -> jnp.ndarray:
        x = v.reshape(v.shape[:-1] + (N_RAYS, N_CH))
        for i, stride in enumerate((2, 1, 1, 1)):
            x = nn.relu(nn.Conv(32, (3,), strides=(stride,), padding="VALID", name=f"conv{i}")(x))
        x = x.reshape(x.shape[:-2] + (-1,))
        return jnp.tanh(nn.LayerNorm(name="ln")(nn.Dense(FEATURE_DIM, name="fc")(x)))


def split(x: jnp.ndarray):
    x = Flatten()(x) if x.ndim > 2 else x
    return x[..., :VISION_DIMS], x[..., VISION_DIMS:]


class ConvEyeActor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def get_std(self):
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        v, rest = split(x)
        f = jax.lax.stop_gradient(RayEncoder(name="encoder")(v))
        return SquashedGaussianActor(net_arch=self.net_arch, action_dim=self.action_dim,
                                     log_std_min=self.log_std_min, log_std_max=self.log_std_max,
                                     activation_fn=self.activation_fn, name="head")(
            jnp.concatenate([f, rest], -1))


class ConvEyeVectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        v, rest = split(obs)
        h = jnp.concatenate([RayEncoder(name="encoder")(v), rest], -1)
        heads = nn.vmap(ContinuousCritic, variable_axes={"params": 0},
                        split_rngs={"params": True, "dropout": True},
                        in_axes=None, out_axes=0, axis_size=self.n_critics)
        return heads(use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate,
                     net_arch=self.net_arch, activation_fn=self.activation_fn,
                     output_dim=self.output_dim, name="heads")(h, action)


def roll_rays(obs: jnp.ndarray, key: jax.Array, shift: int = SHIFT) -> jnp.ndarray:
    """Each sample's ray row rolled by its own integer in [-shift, shift];
    the numbers after the eye are returned untouched."""
    v, rest = obs[..., :VISION_DIMS], obs[..., VISION_DIMS:]
    s = jax.random.randint(key, (obs.shape[0],), -shift, shift + 1)
    rays = v.reshape(obs.shape[0], N_RAYS, N_CH)
    idx = (jnp.arange(N_RAYS)[None, :] - s[:, None]) % N_RAYS
    rays = jnp.take_along_axis(rays, idx[..., None], axis=1)
    return jnp.concatenate([rays.reshape(obs.shape[0], VISION_DIMS), rest], -1)


def sync_encoder(actor_state, qf_state):
    """The actor's encoder copy <- the critic's online encoder."""
    p = jax.tree_util.tree_map(lambda x: x, actor_state.params)
    p = {**p, "params": {**p["params"], "encoder": qf_state.params["params"]["encoder"]}}
    return actor_state.replace(params=p)


class ConvEyeSAC(SAC):
    """sbx SAC with DrQ's random shift and the actor reading the critic's
    encoder. Only `update_critic` and `update_actor_and_temperature` change;
    sbx's jitted bodies are called unchanged on rolled observations."""

    @staticmethod
    @jax.jit
    def update_critic(actor_state, qf_state, ent_coef_state, observations, actions,
                      next_observations, rewards, dones, discounts, key):
        key, k_obs, k_next = jax.random.split(key, 3)
        return SAC.update_critic(actor_state, qf_state, ent_coef_state,
                                 roll_rays(observations, k_obs), actions,
                                 roll_rays(next_observations, k_next),
                                 rewards, dones, discounts, key)

    @classmethod
    def update_actor_and_temperature(cls, actor_state, qf_state, ent_coef_state,
                                     observations, target_entropy, key):
        key, k_obs = jax.random.split(key)
        actor_state = sync_encoder(actor_state, qf_state)
        return SAC.update_actor_and_temperature.__func__(
            cls, actor_state, qf_state, ent_coef_state,
            roll_rays(observations, k_obs), target_entropy, key)


POLICY_KWARGS = dict(actor_class=ConvEyeActor, vector_critic_class=ConvEyeVectorCritic)
