"""J15 E2c: the two-number wrapper (11-18, before J11's rule; no learner runs).

Known answers, each with a control that must fail:
- encode(): planted geometry. Food 1 m ahead at yaw 0 -> bearing 0, distance 0.1;
  turn the body +90 deg -> bearing -0.5 (radpi) / sin -1 (cossin); 12 m -> distance
  clipped to 1; no food -> (0, 1). The +-pi jump of radpi is there (2 sides of the
  seam differ by ~2) and cossin's is not (control: the proposal's reason holds).
- On the pilot's arm-K world (food_floor_only, body eye + body motor): the default
  wrapper leaves his own senses byte-identical to the unwrapped env on the same
  seed and actions; the food arm's numbers equal an independent recomputation from
  JackEnv._food_xy (all foods uneaten at birth); the random point holds within a
  life, is re-drawn at a new life, survives a truncation; zero is zero.
"""
import math

import numpy as np

from experiments import handed_sense as H


def test_encode_planted():
    e = H.encode((1.0, 0.0), (0.0, 0.0), 0.0, "radpi")
    assert np.allclose(e, [0.0, 0.1]), e
    e = H.encode((1.0, 0.0), (0.0, 0.0), math.pi / 2, "radpi")
    assert np.allclose(e, [-0.5, 0.1]), e
    c = H.encode((1.0, 0.0), (0.0, 0.0), math.pi / 2, "cossin")
    assert np.allclose(c, [0.0, -1.0, 0.1], atol=1e-6), c
    assert np.allclose(H.encode((12.0, 0.0), (0, 0), 0.0, "radpi"), [0.0, 1.0])
    assert np.allclose(H.encode(None, (0, 0), 0.0, "radpi"), [0.0, 1.0])
    assert np.allclose(H.encode(None, (0, 0), 0.0, "cossin"), [0.0, 0.0, 1.0])
    a = H.encode((-1.0, 1e-3), (0, 0), 0.0, "radpi")
    b = H.encode((-1.0, -1e-3), (0, 0), 0.0, "radpi")
    assert abs(a[0] - b[0]) > 1.9                       # radpi jumps at the seam
    a = H.encode((-1.0, 1e-3), (0, 0), 0.0, "cossin")
    b = H.encode((-1.0, -1e-3), (0, 0), 0.0, "cossin")
    assert np.abs(a - b).max() < 0.01                   # cossin does not


def _world(seed=811):
    from experiments.pilot_b2 import pilot_world
    return pilot_world("K", True, eye_frame="body", motor_frame="body")(seed)


def test_on_pilot_world():
    from experiments.fast_sim import body_yaw
    raw = _world()
    env = H.HandedSense(_world(), "food", "radpi", seed=811)
    o0, _ = raw.reset(seed=811)
    o1, _ = env.reset(seed=811)
    assert o1.shape[0] == o0.shape[0] + 2 == env.observation_space.shape[0]
    assert np.array_equal(o1[:-2], o0)                  # his senses untouched
    w = env.env.w
    body = np.array(w.data.xpos[w.rover_bid][:2])
    f = env.env._food_xy()
    assert len(H.uneaten_food_xy(env.env)) == len(f) > 0
    near = f[np.argmin(((f - body) ** 2).sum(1))]
    b = math.atan2(near[1] - body[1], near[0] - body[0]) - body_yaw(w)
    b = (b + math.pi) % (2 * math.pi) - math.pi
    assert np.allclose(o1[-2:], [b / math.pi, min(np.hypot(*(near - body)) / 10, 1)], atol=1e-5)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.uniform(-1, 1, raw.action_space.shape)
        p, *_ = raw.step(a)
        q, *_ = env.step(a)
        assert np.array_equal(q[:-2], p)                # identical stream, same actions

    rp = H.HandedSense(_world(), "random_point", "cossin", seed=811)
    rp.reset(seed=811)
    pt = rp.point.copy()
    for _ in range(3):
        rp.step(np.zeros(rp.action_space.shape))
    assert np.array_equal(rp.point, pt)                 # held within a life
    rp.env._truncated, rp.env._died = True, False
    rp.reset()
    assert np.array_equal(rp.point, pt)                 # a truncation keeps it
    rp.env._truncated, rp.env._died = False, True
    rp.reset()
    assert not np.array_equal(rp.point, pt)             # control: a new life re-draws
    f = rp.env._food_xy()
    assert np.all(rp.point >= f.min(0) - 1e-9) and np.all(rp.point <= f.max(0) + 1e-9)

    z = H.HandedSense(_world(), "zero", "radpi", seed=811)
    oz, _ = z.reset(seed=811)
    assert np.array_equal(oz[-2:], [0.0, 0.0]) and np.array_equal(oz[:-2], o0)


def test_under_pilot_lives():
    """11-19: run_b wraps every world in PilotLives, which reads env.w and
    writes env._pending_obs/_truncated. Through HandedSense the stream must be
    the bare world's (senses, reward, ends, lives) and the writes must land on
    JackEnv. Control: a wrapper that does not forward (gymnasium's own) must
    fail at the first reset."""
    from experiments.pilot_b import PilotLives, _Wrapper
    raw = PilotLives(_world(), cap=50)
    hs = H.HandedSense(_world(), "food", "cossin", seed=801)
    env = PilotLives(hs, cap=50)
    o, _ = env.reset(seed=811)
    p, _ = raw.reset(seed=811)
    assert np.array_equal(o[:-3], p)
    for i in range(160):
        a = np.random.default_rng(i).uniform(-1, 1, env.action_space.shape)
        o, r, t, tr, _ = env.step(a)
        q, r2, t2, tr2, _ = raw.step(a)
        assert np.array_equal(o[:-3], q) and (r, t, tr) == (r2, t2, tr2)
        if t or tr:
            o, _ = env.reset()
            raw.reset()
    assert len(env.lives) == len(raw.lives) >= 2      # cap 50 -> lives end
    assert "_truncated" not in hs.__dict__ and "_pending_obs" not in hs.__dict__
    assert hs.env._truncated is False

    class Bare(_Wrapper):                               # control: no forwarding
        pass
    try:
        PilotLives(Bare(_world()), cap=50).reset(seed=811)
    except AttributeError:
        pass
    else:
        raise AssertionError("control: a non-forwarding wrapper did not fail")


def main():
    test_encode_planted()
    test_on_pilot_world()
    test_under_pilot_lives()
    print("test_handed_sense PASS")


if __name__ == "__main__":
    main()
