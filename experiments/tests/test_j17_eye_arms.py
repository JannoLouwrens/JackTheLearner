"""J17: the eye race's arms (rule prereg/row128_vs_rays16.md, frozen by ruling 66).

Known answers, each with a control that must fail:
- pilot 2's and E2c's arms and packing are byte-identical to before the J17
  edit (sha256 pinned from the pre-edit code, the same pins test_e2c_auto has
  carried since 11-20, plus pilot_groups and PILOT_ARMS pinned here).
- The four arms are arm K (K-lr0 for the floor) + ent_coef auto + an eye; 20
  runs, r0..r4, every kernel one seed's four arms.
- The blind arm's eye carries WIDTH and NO CONTENT: obs width equals
  K-row128's, every vision number 0, and every other number, reward, death
  and meal over a fixed action sequence is identical to K-row128's world
  (control: K-row128's own vision is NOT all zero, and a wrapper that drew
  from the world's random stream would break the identity).
- make_world() with no eye is the old call: same obs as before for arm K.
"""
import hashlib
import json

import numpy as np

from experiments import pilot_b2 as P

PRE_E2C_GROUPS = "0f01eedd56ec1998"     # test_e2c_auto's pins (11-20)
PRE_E2C_ARMS = "86e9688f28b38280"
PRE_PILOT_GROUPS = "3b76abbf8dbb67c5"             # pinned 11-21 from the code at 98e22036, before the J17 edit
PRE_PILOT_ARMS = "486781eca32d72dc"
PRE_AUTO_GROUPS = "434dad6eb7968328"


def _h(x):
    return hashlib.sha256(x.encode()).hexdigest()[:16]


def test_old_groups_byte_identical():
    assert _h(repr(P.e2c_groups())) == PRE_E2C_GROUPS
    assert _h(json.dumps(P.E2C_ARMS, sort_keys=True)) == PRE_E2C_ARMS
    assert _h(repr(P.pilot_groups(P.PILOT_RULE["seeds"]))) == PRE_PILOT_GROUPS
    assert _h(json.dumps(P.PILOT_ARMS, sort_keys=True)) == PRE_PILOT_ARMS
    assert _h(repr(P.e2c_auto_groups())) == PRE_AUTO_GROUPS
    assert not set(P.J17_ARMS) & (set(P.PILOT_ARMS) | set(P.E2C_ARMS) | set(P.E2C_AUTO_ARMS))
    # control: the pin can tell a changed packing from the frozen one
    assert _h(repr(P.j17_groups())) != PRE_PILOT_GROUPS


def test_arms_are_K_plus_auto_plus_an_eye():
    eyes = {"K-rays16": "rays16", "K-row128": "row128", "K-row128-blind": "row128", "K-lr0-row128": "row128"}
    assert set(P.J17_ARMS) == set(eyes)
    for arm, eye in eyes.items():
        c = dict(P.J17_ARMS[arm])
        assert c.pop("eye") == eye and c.pop("ent_coef") == "auto" and c.pop("trace") is True
        assert c.pop("blind", False) is (arm == "K-row128-blind")
        assert c == P.PILOT_ARMS["K-lr0" if arm == "K-lr0-row128" else "K"], arm
        assert "handed" not in c                       # no handed sense on any arm
    assert P.J17_ARMS["K-lr0-row128"]["lr"] == 0.0
    assert P.J17_SEEDS == (921, 922, 923, 924, 925) and P.J17_T == 300_000 == P.PILOT_T
    assert P.J17_LATE_FROM == 150_000
    g = P.j17_groups()
    cells = [c for k in sorted(g) for c in g[k]]
    assert sorted(g) == ["r0", "r1", "r2", "r3", "r4"] and len(cells) == 20 == len(set(cells))
    for k, s in zip(sorted(g), P.J17_SEEDS):
        assert [a for a, _ in g[k]] == list(P.J17_ARMS) and {t for _, t in g[k]} == {s}
    used = set(P.PILOT_RULE["seeds"]) | set(P.E2C_SEEDS)
    assert not set(P.J17_SEEDS) & used


SMOKE_SEED = 931      # never 921-925: the rule's seeds are not stepped before the row


def _world(arm, seed=SMOKE_SEED):
    R = P.PILOT_RULE
    return P.pilot_world(arm, R["food_floor_only"], R["eye_frame"], R["motor_frame"], R["native_substep"])(seed)


def test_blind_eye_is_width_without_content(n=2000):
    see, blind, rays = _world("K-row128"), _world("K-row128-blind"), _world("K-rays16")
    o1, _ = see.reset(seed=SMOKE_SEED)
    o2, _ = blind.reset(seed=SMOKE_SEED)
    o3, _ = rays.reset(seed=SMOKE_SEED)
    nv = blind.n_vision
    assert nv == 256 and o1.shape == o2.shape and o3.shape[0] == o1.shape[0] - 256 + 32
    rng = np.random.RandomState(0)
    seen_nonzero, deaths, meals = 0, 0, 0
    for t in range(n):
        a = rng.uniform(-1, 1, see.action_space.shape).astype(np.float32)
        r1 = see.step(a)
        r2 = blind.step(a)
        assert not r2[0][:nv].any(), t                                   # no content
        assert np.array_equal(r1[0][nv:], r2[0][nv:]), t                 # every other number identical
        assert r1[1] == r2[1] and r1[2] == r2[2] and r1[3] == r2[3], t   # the same world underneath
        assert r1[4]["meals"] == r2[4]["meals"] and r1[4]["decisions"] == r2[4]["decisions"]
        seen_nonzero += int(r1[0][:nv].any())
        if r1[2] or r1[3]:
            deaths += 1
            o1, _ = see.reset()
            o2, _ = blind.reset()
            assert not o2[:nv].any() and np.array_equal(o1[nv:], o2[nv:])
        meals = r1[4]["meals"]
    # the reset path too, whether or not a life ended inside n (lives here outlast 2k decisions)
    o1, _ = see.reset()
    o2, _ = blind.reset()
    assert not o2[:nv].any() and np.array_equal(o1[nv:], o2[nv:]) and o1[:nv].any()
    # control: the seeing arm's eye DOES carry content (else "all zero" proves nothing)
    assert seen_nonzero > 0.9 * n, seen_nonzero
    return {"n_vision": nv, "obs": int(o1.shape[0]), "rays16_obs": int(o3.shape[0]),
            "seen_nonzero": seen_nonzero, "resets": deaths, "meals": meals}


def test_default_eye_call_unchanged():
    # arm K (no eye key) and K-rays16 (eye named) see the same world and the same numbers
    k, r = _world("K"), _world("K-rays16")
    a0, _ = k.reset(seed=SMOKE_SEED)
    b0, _ = r.reset(seed=SMOKE_SEED)
    assert np.array_equal(a0, b0)
    rng = np.random.RandomState(1)
    for _ in range(300):
        a = rng.uniform(-1, 1, k.action_space.shape).astype(np.float32)
        x, y = k.step(a), r.step(a)
        assert np.array_equal(x[0], y[0]) and x[1] == y[1]
    assert "eye" not in P.arm_cfg("K")


def test_kernel_routing():
    called = {}
    real = P.pilot_kernel
    P.pilot_kernel = lambda name, rule, cells=None, out_name=None: called.update(
        name=name, rule=rule, cells=cells, out=out_name)
    try:
        P.j17_kernel("r3")
    finally:
        P.pilot_kernel = real
    assert called["cells"] == P.j17_groups()["r3"] and called["out"] == "j17_r3.json"
    r = called["rule"]
    assert (r["eye_frame"], r["motor_frame"], r["native_substep"], r["food_floor_only"]) == \
        ("body", "body", True, True) and "ruling 66" in r["rule"]
    assert "ruling 66" not in P.PILOT_RULE.get("rule", "")
    body = P.pilot_job_body(called["cells"], True, P.J17_T, r["rule"], "body", "body", True)
    assert "('K-row128-blind', 924)" in body and "300000" in body


def test_multi_ray_takes_either_mujoco_signature():
    """Kaggle pins mujoco 3.17.0 (mj_multiRay has a `normal` argument), the box runs
    3.2.3 (it has not). Found 11-21 by running these arms under 3.17.0: the row eye
    died at construction. Both signatures, through fakes; control: the old direct
    11-argument call is refused by the new signature."""
    import mujoco
    from experiments import fast_sim as F
    calls = []

    def old(m, d, pnt, vec, groups, flg, excl, gid, dist, nray, cutoff):
        calls.append(("old", nray, flg, excl))
    old.__doc__ = "mj_multiRay(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, dist, nray, cutoff)"

    def new(m, d, pnt, vec, groups, flg, excl, gid, dist, normal, nray, cutoff):
        assert normal is None
        calls.append(("new", nray, flg, excl))
    new.__doc__ = "mj_multiRay(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, dist, normal: NDArray | None, nray, cutoff)"
    real, real_flag = mujoco.mj_multiRay, F._MR_HAS_NORMAL
    try:
        for fake in (old, new):
            mujoco.mj_multiRay, F._MR_HAS_NORMAL = fake, None
            F._multi_ray("m", "d", "p", "v", "g", "gid", "dist", 128)
        assert calls == [("old", 128, 1, -1), ("new", 128, 1, -1)]
        try:
            new("m", "d", "p", "v", "g", 1, -1, "gid", "dist", 128, 1e10)   # the pre-fix call
            raise AssertionError("control: the old call fits the new signature")
        except TypeError:
            pass
    finally:
        mujoco.mj_multiRay, F._MR_HAS_NORMAL = real, real_flag
    # and the real one, whichever version this interpreter has: the row eye builds and sees
    o, _ = _world("K-row128").reset(seed=SMOKE_SEED)
    assert o[:256].any()
    # first look, pinned from mujoco 3.2.3 AND 3.17.0 (bit-identical on both, 11-21)
    assert abs(float(o[:256].sum()) - 137.463484) < 1e-4, float(o[:256].sum())
    return mujoco.__version__


def test_learner_cells_run_end_to_end(T=1500):
    """Needs sbx (run with /data/venvs/jack-std/bin/python). Every arm through the
    kernel's own cell function on a NON-rule seed: the eye and auto alpha reach SAC,
    the blind arm trains on a 298-wide input, and lr 0 stays lr 0."""
    from experiments import pilot_b as B
    R = P.PILOT_RULE
    kw = dict(eye_frame=R["eye_frame"], motor_frame=R["motor_frame"], native_substep=R["native_substep"])
    old = (B.WARMUP, B.TRACE_EVERY)
    B.WARMUP, B.TRACE_EVERY = 100, 500
    out = {}
    try:
        for arm in P.J17_ARMS:
            rec = P.run_pilot_cell(arm, SMOKE_SEED, True, T=T, **kw)
            # decisions overshoot T by under one 8-step batch (1,504 for 1,500)
            assert rec["T"] == T and T <= rec["decisions"] < T + 16, rec["decisions"]
            assert rec["pilot_arm"] == arm and rec["ent_coef_setting"] == "auto"
            assert rec["lr"] == (0.0 if arm == "K-lr0-row128" else 3e-4), rec["lr"]
            assert "error" not in rec and len(rec["trace"]) == T // B.TRACE_EVERY
            assert rec["eye"] == P.J17_ARMS[arm]["eye"] and rec["blind"] is (arm == "K-row128-blind")
            assert rec["trace"] and "handed" not in rec
            out[arm] = [round(r["ent_coef"], 4) for r in rec["trace"] if r.get("ent_coef") is not None]
    finally:
        B.WARMUP, B.TRACE_EVERY = old
    # auto alpha MOVES on the training arms and cannot move at lr 0 (the control)
    for arm in ("K-rays16", "K-row128", "K-row128-blind"):
        assert len(set(out[arm])) > 1, (arm, out[arm])
    return out


def main():
    test_old_groups_byte_identical()
    test_arms_are_K_plus_auto_plus_an_eye()
    test_default_eye_call_unchanged()
    test_kernel_routing()
    print("multi-ray under mujoco", test_multi_ray_takes_either_mujoco_signature())
    print("blind eye:", test_blind_eye_is_width_without_content())
    try:
        import sbx  # noqa: F401
    except ImportError:
        print("learner cells: SKIPPED (no sbx in this venv; run with /data/venvs/jack-std/bin/python)")
    else:
        print("learner cells, alpha trace:", test_learner_cells_run_end_to_end())
    print("test_j17_eye_arms PASS")


if __name__ == "__main__":
    main()
