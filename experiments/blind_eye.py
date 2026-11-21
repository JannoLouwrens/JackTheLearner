"""J17's must-fail arm: an eye with its width and no content.

`BlindEye(env)` hands the learner the world's observation with every VISION
number set to zero -- the same shape, the same audio / touch / proprio /
needs, the same world underneath (nothing in the simulation is touched: the
eye is still cast, its numbers are just not passed on). Rule:
docs/research/prereg/row128_vs_rays16.md (frozen, ruling 66), arm
`K-row128-blind`: if row128 beats 16 rays but NOT this arm, the gain was the
wider input, not sight.

Vision is the FIRST block of JackEnv's observation (jack_env.SENSES order);
its width is read from the world's own observe_pure(), never assumed.
"""
from __future__ import annotations

import numpy as np

from experiments.pilot_b import _Wrapper  # gymnasium.Wrapper


class BlindEye(_Wrapper):
    # pilot_b.PilotLives reads env.w / counters and WRITES these three through
    # whatever wraps the world (found 11-19 in handed_sense.HandedSense): reads
    # fall through, the named writes are forwarded, nothing else.
    FORWARD_SET = ("_pending_obs", "_truncated", "_died")

    def __init__(self, env):
        from experiments.jack_env import SENSES
        if SENSES[0] != "vision":
            raise RuntimeError("BlindEye: vision is no longer the first sense block")
        super().__init__(env)
        self.n_vision = None

    def __getattr__(self, name):
        if name == "env" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        if name in BlindEye.FORWARD_SET and "env" in self.__dict__:
            setattr(self.env, name, value)
        else:
            object.__setattr__(self, name, value)

    def _blind(self, o):
        if self.n_vision is None:
            # observe_pure, NOT observe: observe() draws from the world's random
            # stream (placebo, noise panel) and one extra draw would make this
            # arm's world differ from K-row128's (the planted test's control).
            self.n_vision = int(np.asarray(self.env.w.observe_pure()["vision"]).size)
            if not 0 < self.n_vision < o.shape[0]:
                raise RuntimeError(f"BlindEye: vision width {self.n_vision} vs obs {o.shape[0]}")
        o = np.array(o, dtype=np.float32, copy=True)
        o[:self.n_vision] = 0.0
        return o

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        return self._blind(o), info

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        return self._blind(o), r, term, trunc, info
