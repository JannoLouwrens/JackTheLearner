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


def test_run_loop_schedule_matches_theirs():
    """Their learner order at a tiny size: 1 random episode -> 200 updates; 10 iterations of
    (1 rollout, 200 updates); eval when steps//200 hits a new multiple of 5 -> steps 1000, 2000."""
    from experiments import brain_b_mem as m
    c = dict(m.CFG, obs_emb=4, act_emb=2, rew_emb=2, rnn=4, layers=(4, 4), batch=2,
             init_eps=1, iters=10, eval_eps=1)
    calls = []
    orig = m.BrainBMem.update
    m.BrainBMem.update = lambda self: calls.append(1)        # count updates, skip the maths
    try:
        ev = m.run_seed(0, None, c)
    finally:
        m.BrainBMem.update = orig
    assert [e["step"] for e in ev] == [1000, 2000], ev
    assert len(calls) == 200 + 10 * 200, len(calls)


def test_b7_score_window():
    from experiments.brain_b_mem import b7_seed_score
    ev = [dict(step=s, ret=-1000.0 if s < 40000 or s > 50000 else -170.0) for s in range(2000, 51001, 1000)]
    assert b7_seed_score(ev) == -170.0                      # 51k point and pre-40k excluded
    try:
        b7_seed_score([e for e in ev if e["step"] != 45000]); raise AssertionError("missing point accepted")
    except ValueError:
        pass


def test_carry_is_bit_equal():
    """A seed cut twice and resumed from its carry file (fresh objects each time) equals the
    unbroken seed exactly: same curve, same final weights. Planted: resuming with the torch RNG
    state dropped must differ (the check can fail)."""
    import os, tempfile
    from experiments import brain_b_mem as m
    c = dict(m.CFG, obs_emb=4, act_emb=2, rew_emb=2, rnn=4, layers=(4, 4), batch=2,
             init_eps=1, iters=10, eval_eps=1)
    def final(seed, carry=None, cuts=(), drop_torch_rng=False):
        for s in cuts:
            m.run_seed(seed, None, c, carry=carry, stop_at=s)
            if drop_torch_rng:
                st = torch.load(carry, weights_only=False); st["torch_rng"] = torch.manual_seed(999).get_state()
                torch.save(st, carry)
        holder = {}
        orig = m.BrainBMem.__init__
        def spy(self, *a, **k):
            orig(self, *a, **k); holder["b"] = self
        m.BrainBMem.__init__ = spy
        try:
            ev = m.run_seed(seed, None, c, carry=carry)
        finally:
            m.BrainBMem.__init__ = orig
        return ev, [p.detach().clone() for p in holder["b"].actor.parameters()]
    ev0, w0 = final(0)
    with tempfile.TemporaryDirectory() as d:
        ev1, w1 = final(0, os.path.join(d, "c.pt"), cuts=(600, 1400))
    assert [e["ret"] for e in ev0] == [e["ret"] for e in ev1] and len(ev0) == 2, (ev0, ev1)
    assert all(torch.equal(a, b) for a, b in zip(w0, w1))
    with tempfile.TemporaryDirectory() as d:
        ev2, w2 = final(0, os.path.join(d, "c.pt"), cuts=(600,), drop_torch_rng=True)
    assert not all(torch.equal(a, b) for a, b in zip(w0, w2)), "planted RNG drop not caught"


def test_deadline_cut_is_bit_equal():
    """A wall-clock deadline cut (already past: cuts at the first boundary, steps 200) and resumed
    equals the unbroken seed; a deadline cut without a carry path refuses."""
    import os, tempfile, json
    from experiments import brain_b_mem as m
    c = dict(m.CFG, obs_emb=4, act_emb=2, rew_emb=2, rnn=4, layers=(4, 4), batch=2,
             init_eps=1, iters=10, eval_eps=1)
    ev0 = m.run_seed(0, None, c)
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "c.pt")
        part = m.run_seed(0, None, c, carry=cp, deadline=0.0)
        st = torch.load(cp, weights_only=False)
        assert st["steps"] == 200 and part == [], (st["steps"], part)
        ev1 = m.run_seed(0, None, c, carry=cp)
    assert [e["ret"] for e in ev0] == [e["ret"] for e in ev1] and len(ev0) == 2
    try:
        m.run_seed(0, None, c, deadline=0.0); raise AssertionError("cut without carry accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    for f in [test_obs_is_velocity_only, test_separate_encoders_and_update_moves_both, test_policy_rollout_bounded,
              test_run_loop_schedule_matches_theirs, test_b7_score_window,
              test_carry_is_bit_equal, test_deadline_cut_is_bit_equal]:
        f(); print("PASS", f.__name__)
