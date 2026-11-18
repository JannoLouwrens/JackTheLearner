"""J15: the clone's data collector records the teacher's own lives (11-18, before any data).

Known answer: on smoke seed 153 (d0, 1 life, cap 300) the collector's stored actions,
replayed open-loop into a fresh f4 world, reproduce the stored rows bit for bit, and
its meal count equals `eye_forager.run_seed`'s for the same life. Control that must
differ: the g4 twin (rays permuted) on the same seed stores different actions.
Seeds pinned: training 701..740 and test 741..760 never overlap gate 2's 261..280."""
import numpy as np

from experiments import eye_forager as EF
from experiments import forager_clone as C


def test_seeds_fresh():
    assert not set(C.TRAIN_SEEDS) & set(C.TEST_SEEDS)
    assert not (set(C.TRAIN_SEEDS) | set(C.TEST_SEEDS)) & set(EF.GATE2_SEEDS + EF.GATE_SEEDS)
    assert C.N_IN == 256


def test_collector_is_the_teacher():
    r = C.collect_seed(153, lives=1, cap=300)
    assert r["x"].shape == (300, 256) and r["a"].shape == (300, 8)
    ref = EF.run_seed("f4", 153, C.D0, lives=1, cap=300)[0]
    assert int(r["meals"][0]) == ref["floor_meals"], (r["meals"], ref)
    env = EF.make_env(153, C.D0, row="f4")          # open-loop replay
    o, _ = env.reset(seed=153)
    for t in range(300):
        assert np.array_equal(np.asarray(o[:256], np.float32), r["x"][t]), t
        o, *_ = env.step(r["a"][t])
    # control: the permuted twin's actions differ from the teacher's
    env = EF.make_env(153, C.D0, row="g4")
    o, _ = env.reset(seed=153)
    pol = EF.make_policy("g4", 153)
    pol.new_life()
    a_g = []
    for _t in range(50):
        a = np.asarray(pol.act(o), np.float32)
        a_g.append(a)
        o, *_ = env.step(a)
    assert not np.array_equal(np.stack(a_g), r["a"][:50])


def main():
    test_seeds_fresh()
    test_collector_is_the_teacher()
    print("test_forager_clone PASS")


if __name__ == "__main__":
    main()
