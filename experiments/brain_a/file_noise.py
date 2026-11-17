"""MLX port step 0.4c(iii): noise from a file for the official agent's two samplers.

Report row 0.4 ("noise from file"): the port can only be checked bit-for-bit against the official
code if both consume the SAME noise; JAX's threefry stream is not something the MLX side should have
to re-implement, so the official samplers are patched to read their base draws, in call order, from
an .npz file the MLX side will read too. Source: danijar/dreamerv3 @ e01491f, `embodied/jax/outs.py`.

The two samplers (every stochastic draw in a `defaults debug` train step goes through them):
  - `outs.Normal.sample` (the continuous policy, `policy_dist_cont: bounded_normal`):
    official `jax.random.normal(seed) * stddev + mean`  ->  `normal[cursor] * stddev + mean`.
  - `outs.Categorical.sample` (RSSM stoch latents via OneHot; a discrete policy):
    official `jax.random.categorical(seed, logits)`, which IS argmax(logits + Gumbel(seed)) ->
    argmax(logits + (-log(-log(u)))) with `u = uniform[cursor]` clipped to [1e-7, 1 - 1e-7].
DIFFERENCES from the official code (listed for the port's contract): the base draws come from the
file instead of threefry (same distributions, different numbers); the Gumbel transform is written
out instead of `jax.random.gumbel`'s; the `seed` argument is ignored. `Binary.sample` and dropout are
not patched — the debug train step does not call them (asserted by the per-step draw counts).
Works only eagerly (`--jit False`): a Python cursor under `jax.jit` would be baked in at trace time.
"""
from __future__ import annotations

import pathlib

import numpy as np

N_DEFAULT = 4_000_000


class FileNoise:
    def __init__(self, path, seed=0, n=N_DEFAULT):
        path = pathlib.Path(path)
        if not path.exists():
            rng = np.random.default_rng(seed)
            np.savez(path, uniform=rng.random(n, dtype=np.float32),
                     normal=rng.standard_normal(n, dtype=np.float32), seed=np.int64(seed))
        z = np.load(path)
        self.path, self.uniform, self.normal = str(path), z["uniform"], z["normal"]
        self.cur = {"uniform": 0, "normal": 0}
        self.calls = {"uniform": 0, "normal": 0}

    def take(self, kind, shape):
        n = int(np.prod(shape)) if len(shape) else 1
        arr = self.uniform if kind == "uniform" else self.normal
        c = self.cur[kind]
        if c + n > arr.size:
            raise RuntimeError(f"noise file {self.path} exhausted: {kind} needs {c + n} > {arr.size}")
        self.cur[kind] = c + n
        self.calls[kind] += 1
        return arr[c:c + n].reshape(shape)

    def snapshot(self):
        return {"cur": dict(self.cur), "calls": dict(self.calls)}

    def install(self, outs):
        import jax.numpy as jnp
        noise = self

        def normal_sample(self, seed, shape=()):
            eps = jnp.asarray(noise.take("normal", tuple(shape) + tuple(self.mean.shape)))
            return eps * self.stddev + self.mean

        def categorical_sample(self, seed, shape=()):
            full = tuple(shape) + tuple(self.logits.shape)
            u = np.clip(noise.take("uniform", full), 1e-7, 1 - 1e-7)
            g = jnp.asarray(-np.log(-np.log(u)), jnp.float32)
            return jnp.argmax(self.logits + g, axis=-1)

        outs.Normal.sample = normal_sample
        outs.Categorical.sample = categorical_sample
        return self
