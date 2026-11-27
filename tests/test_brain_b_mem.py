"""J29: the port's apparatus checks (not a reading). Run: python -m tests.test_brain_b_mem"""
import numpy as np, torch
from experiments.brain_b_mem import PendulumV, BrainBMem


def test_obs_is_velocity_only():
    env = PendulumV(); o, _ = env.reset(seed=3)
    assert o.shape == (1,)
    th, thdot = env.unwrapped.state
    assert abs(o[0] - thdot) < 1e-6, (o, thdot)   # planted: a full-obs env would fail the shape


def test_separate_encoders_and_update_moves_both():
    b = BrainBMem(0, dict(__import__("experiments.brain_b_mem", fromlist=["CFG"]).CFG, batch=2))
    assert b.actor.enc is not b.critic.enc and b.actor.enc.rnn is not b.critic.enc.rnn
    env = PendulumV()
    for _ in range(2):
        b.rollout(env, random=True)
    assert b.eps[0]["o"].shape == (201, 1, 1) and b.eps[0]["a"].shape == (200, 1, 1)
    a0 = [p.clone() for p in b.actor.parameters()]; c0 = [p.clone() for p in b.critic.parameters()]
    t0 = [p.clone() for p in b.critic_t.parameters()]
    b.update()
    assert any(not torch.equal(x, p) for x, p in zip(a0, b.actor.parameters()))
    assert any(not torch.equal(x, p) for x, p in zip(c0, b.critic.parameters()))
    # Polyak: target moved by tau, not copied
    d = max((p - x).abs().max().item() for x, p in zip(t0, b.critic_t.parameters()))
    assert 0 < d < 0.01, d


def test_policy_rollout_bounded():
    b = BrainBMem(1); env = PendulumV()
    ret, n = b.rollout(env, deterministic=True, learn=False)
    assert n == 200 and -16.3 * 200 <= ret <= 0


if __name__ == "__main__":
    for f in [test_obs_is_velocity_only, test_separate_encoders_and_update_moves_both, test_policy_rollout_bounded]:
        f(); print("PASS", f.__name__)
