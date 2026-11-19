"""J15 (4) E2c kernel parts (11-19, before seed 801; no registered run here).

Known answers, each with a control that must fail:
- E2C_ARMS are arm K exactly plus `handed` (food / zero / random_point); the
  pilot's own arms and 40-run packing are untouched (p0..p9 as before).
- e2c packing: 15 runs, every seed carries all three arms, 4 kernels.
- e2c_kernel REFUSES while J11's forty runs are not all home (control: with a
  stubbed "nothing missing" the guard opens).
- A 300-decision smoke of run_pilot_cell("E-food") on the frozen frames: B
  sees 3 more numbers than arm K, and the record names the handed sense.
"""
import numpy as np

from experiments import pilot_b2 as P


def test_arms_and_packing():
    for m in ("food", "zero", "random_point"):
        c = dict(P.E2C_ARMS[f"E-{m}"])
        assert c.pop("handed") == m and c.pop("trace") is True
        assert c == P.PILOT_ARMS["K"]
    assert not any("trace" in c for c in P.PILOT_ARMS.values())
    assert set(P.PILOT_ARMS) == set(P.PILOT_LEARNERS + P.PILOT_FLOORS)
    assert sorted(P.pilot_groups(P.PILOT_RULE["seeds"])) == [f"p{j}" for j in range(10)]
    g = P.e2c_groups()
    cells = [c for k in sorted(g) for c in g[k]]
    assert sorted(g) == ["e0", "e1", "e2", "e3"] and len(cells) == 15 == len(set(cells))
    for s in P.E2C_SEEDS:
        assert {a for a, t in cells if t == s} == set(P.E2C_ARMS)


def test_guard():
    real = P.e2c_ready
    if real():                                   # today: J11 not home -> refuse
        try:
            P.e2c_kernel("e0")
        except RuntimeError as e:
            assert "not home" in str(e)
        else:
            raise AssertionError("e2c_kernel launched before J11 was home")
    called = {}
    P.e2c_ready = lambda: []                      # control: the guard opens
    real_pk = P.pilot_kernel
    P.pilot_kernel = lambda name, rule, cells=None, out_name=None: called.update(
        name=name, rule=rule, cells=cells, out=out_name)
    try:
        P.e2c_kernel("e1")
    finally:
        P.e2c_ready, P.pilot_kernel = real, real_pk
    assert called["cells"] == P.e2c_groups()["e1"] and called["out"] == "e2c_e1.json"
    r = called["rule"]
    assert (r["eye_frame"], r["motor_frame"], r["native_substep"], r["food_floor_only"]) == \
        ("body", "body", True, True) and "E2c" in r["rule"]


def test_smoke():
    R = P.PILOT_RULE
    kw = dict(eye_frame=R["eye_frame"], motor_frame=R["motor_frame"],
              native_substep=R["native_substep"])
    e = P.pilot_world("E-food", True, **kw)(801)
    k = P.pilot_world("K", True, **kw)(801)
    assert e.observation_space.shape[0] == k.observation_space.shape[0] + 3
    rec = P.run_pilot_cell("E-food", 801, True, T=300, **kw)
    assert rec["handed"] == "food" and rec["handed_encoding"] == "cossin"
    assert rec["pilot_arm"] == "E-food" and rec["trace_every"] == 5000
    assert rec["trace"] == []                    # 300 < one window


def main():
    test_arms_and_packing()
    test_guard()
    test_smoke()
    print("test_e2c_kernel PASS")


if __name__ == "__main__":
    main()
