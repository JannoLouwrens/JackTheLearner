"""J29: B7flat twin apparatus checks (not a reading). Run: python -m tests.test_b7_flat"""
import numpy as np
from experiments import b7_flat as f


def test_same_task_as_mem_arm():
    e = f.PendulumVUnit(); o, _ = e.reset(seed=1)
    assert o.shape == (1,) and abs(o[0] - e.unwrapped.state[1]) < 1e-6   # velocity only
    assert e.action_space.low[0] == -1.0 and e.action_space.high[0] == 1.0
    e.step(np.array([1.0], np.float32))
    assert abs(e.unwrapped.last_u - 2.0) < 1e-6                         # a=1 -> torque 2, as B7mem


def test_recipe_is_brain_b_ours():
    m = f.make_flat(0)
    assert m.learning_rate == 1e-3 and m.policy_kwargs.get("layer_norm") is True
    assert type(m).__module__.startswith("sbx")


def test_schedule_and_score_window():
    ev = f.run_seed(0, None, steps=600, every=200, start=400, eps=1)
    assert [e["step"] for e in ev] == [400, 600]
    full = [dict(step=s, ret=-1.0) for s in range(2000, 51001, 1000)]
    assert f.b7_seed_score(full) == -1.0 and len(full) == 50
    try:
        f.b7_seed_score(full[:-12]); raise AssertionError("window with a missing point accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
