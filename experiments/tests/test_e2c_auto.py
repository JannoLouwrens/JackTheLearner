"""J15 ruling 57: E2c's auto twins (11-20, before any of their seeds).

Known answers, each with a control that must fail:
- E-food-auto / E-zero-auto are E-food / E-zero exactly plus ent_coef="auto";
  the fixed arms' packing e2c_groups() is byte-identical to before the edit
  (sha256 of its repr pinned 11-20 from the pre-edit code), and E2C_ARMS too.
- Auto packing: 10 runs, a0..a2, every seed carries both twins.
- ent_coef reaches SAC: a fixed arm's first 2,000 decisions (training on from
  decision 100) are identical with the default and with ent_coef=0.1 passed
  explicitly, and its trace reads 0.1; the auto arm's trace reads a LEARNED
  coefficient != 0.1 and its lives DIFFER (control: the knob is not inert).
- e2c_kernel("a0") submits the auto cells with the frozen frames.
"""
import hashlib
import json

from experiments import pilot_b as B
from experiments import pilot_b2 as P

PRE_EDIT_GROUPS = "0f01eedd56ec1998"
PRE_EDIT_ARMS = "86e9688f28b38280"


def test_arms_and_packing():
    assert hashlib.sha256(repr(P.e2c_groups()).encode()).hexdigest()[:16] == PRE_EDIT_GROUPS
    assert hashlib.sha256(json.dumps(P.E2C_ARMS, sort_keys=True).encode()
                          ).hexdigest()[:16] == PRE_EDIT_ARMS
    for m in ("food", "zero"):
        c = dict(P.E2C_AUTO_ARMS[f"E-{m}-auto"])
        assert c.pop("ent_coef") == "auto" and c == P.E2C_ARMS[f"E-{m}"]
    g = P.e2c_auto_groups()
    cells = [c for k in sorted(g) for c in g[k]]
    assert sorted(g) == ["a0", "a1", "a2"] and len(cells) == 10 == len(set(cells))
    for s in P.E2C_SEEDS:
        assert {a for a, t in cells if t == s} == set(P.E2C_AUTO_ARMS)
    assert not set(P.E2C_AUTO_ARMS) & (set(P.E2C_ARMS) | set(P.PILOT_ARMS))


def test_ent_coef_reaches_sac():
    R = P.PILOT_RULE
    kw = dict(eye_frame=R["eye_frame"], motor_frame=R["motor_frame"],
              native_substep=R["native_substep"])
    old = (B.WARMUP, B.TRACE_EVERY)
    B.WARMUP, B.TRACE_EVERY = 100, 1000
    try:
        fx = P.run_pilot_cell("E-food", 801, True, T=2000, **kw)
        world = P.pilot_world("E-food", True, **kw)
        fx2 = B.run_b(801, 2000, world=world, trace=True, ent_coef=0.1)
        au = P.run_pilot_cell("E-food-auto", 801, True, T=2000, **kw)
    finally:
        B.WARMUP, B.TRACE_EVERY = old
    lv = lambda r: json.dumps(r["lives"] + [r["decisions"]], sort_keys=True)
    assert lv(fx) == lv(fx2), "the fixed arm's lives changed"
    assert "ent_coef_setting" not in fx and au["ent_coef_setting"] == "auto"
    assert all(abs(r["ent_coef"] - 0.1) < 1e-6 for r in fx["trace"])
    a = [r["ent_coef"] for r in au["trace"]]
    assert all(x is not None and abs(x - 0.1) > 1e-3 for x in a), a
    assert lv(au) != lv(fx), "control: auto changed nothing"
    return a


def test_kernel_routing():
    called = {}
    real_r, real_pk = P.e2c_ready, P.pilot_kernel
    P.e2c_ready = lambda: []
    P.pilot_kernel = lambda name, rule, cells=None, out_name=None: called.update(
        rule=rule, cells=cells, out=out_name)
    try:
        P.e2c_kernel("a0")
    finally:
        P.e2c_ready, P.pilot_kernel = real_r, real_pk
    assert called["cells"] == P.e2c_auto_groups()["a0"] and called["out"] == "e2c_a0.json"
    r = called["rule"]
    assert (r["eye_frame"], r["motor_frame"], r["native_substep"], r["food_floor_only"]) == \
        ("body", "body", True, True) and "ruling 57" in r["rule"]
    assert "ruling 57" not in P.PILOT_RULE.get("rule", "")


def main():
    test_arms_and_packing()
    test_kernel_routing()
    print("auto ent_coef trace:", test_ent_coef_reaches_sac())
    print("test_e2c_auto PASS")


if __name__ == "__main__":
    main()
