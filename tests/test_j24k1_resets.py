"""J24 follow-up DRAFT (prereg/eye_bins16_k1.md): the reset knob.

Planted: a K1 learner resets its nets at the named decision count and a J24
learner does not; J24's arms stay byte-identical; K1 arms differ only by
param_resets. Run under /data/venvs/jack-std (sbx)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import jax
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments import pilot_b, pilot_b2


def _leaves(model):
    return np.concatenate([np.ravel(x) for x in
                           jax.tree_util.tree_leaves(model.policy.actor_state.params)])


def _model(resets):
    venv = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    return pilot_b.make_b(venv, 0, 10_000, param_resets=resets)


def test_default_hands_sbx_no_resets():
    m = _model(None)
    assert not m.param_resets and m.reset_idx == 0


def test_k1_resets_at_named_point_and_j24_does_not():
    w = pilot_b.WARMUP
    k1, base = _model([w + 64]), _model(None)
    k1.learn(w + 32); base.learn(w + 32)
    assert k1.reset_idx == 0
    before = _leaves(k1)
    k1.learn(64, reset_num_timesteps=False); base.learn(64, reset_num_timesteps=False)
    assert k1.reset_idx == 1 and base.reset_idx == 0
    # a reset re-initialises: actor weights jump far more than 2 train calls move them
    assert np.abs(_leaves(k1) - before).mean() > 0.01


def test_arms():
    assert set(pilot_b2.J24K1_ARMS) == {"K1-bins16", "K1-bins16-blind", "K1-bins16-shuffled"}
    assert pilot_b2.J24K1_RESETS == tuple(range(50_000, 500_000, 50_000))
    for a, c in pilot_b2.J24K1_ARMS.items():
        j = pilot_b2.J24_ARMS["K-" + a[3:]]
        assert "param_resets" not in j
        assert {k: v for k, v in c.items() if k != "param_resets"} == j
        assert pilot_b2.arm_cfg(a)["param_resets"] == list(range(50_000, 500_000, 50_000))


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print("PASS", n)
