"""J24 (3): brain B's three cells on the kind paint's binned eye (ORDERS J24).

Known answers, each with a control that must fail:
- every older arm dict and packing is byte-identical (test_j17_eye_arms's
  pins, re-asserted here) and make_world() with no paint is the old call;
- the three arms are J17's K-row128 setting with ONLY the eye changed
  (eye="bins16", paint="kind"), blind = J17's blind wrapper;
- K-bins16 vision is 80 wide (5 kinds x 16 bins) and sees things;
- blind: same width, every bin zero, every other number / reward / meal over
  a fixed action sequence identical to K-bins16's world;
- shuffled: each step's 16 bin COLUMNS are one permutation of the sighted
  step's columns (every kind moved together), every other number identical,
  the world underneath the same; control: the permutation is not the
  identity on most steps that see anything (else "shuffled" is the sighted
  arm), and a per-kind independent shuffle would break the column identity.
- end to end (jack-std only, sbx): each arm's cell runs through
  run_pilot_cell and records eye/paint/shuffled.
Seeds: SMOKE 951, never the rule's 941-943 (not stepped before the row).
"""
import numpy as np

from experiments import pilot_b2 as P
from experiments.tests import test_j17_eye_arms as J17

SMOKE_SEED = 951


def _world(arm, seed=SMOKE_SEED):
    R = P.PILOT_RULE
    return P.pilot_world(arm, R["food_floor_only"], R["eye_frame"], R["motor_frame"],
                         R["native_substep"])(seed)


def test_old_arms_untouched():
    J17.test_old_groups_byte_identical()
    J17.test_arms_are_K_plus_auto_plus_an_eye()
    assert not set(P.J24_ARMS) & (set(P.PILOT_ARMS) | set(P.E2C_ARMS) | set(P.E2C_AUTO_ARMS)
                                  | set(P.J17_ARMS) | set(P.J17_HUNT_ARMS))
    assert "paint" not in P.arm_cfg("K-row128") and "paint" not in P.arm_cfg("K")


def test_arms_are_K_row128_with_only_the_eye_changed():
    assert list(P.J24_ARMS) == ["K-bins16", "K-bins16-blind", "K-bins16-shuffled"]
    base = {"K-bins16": "K-row128", "K-bins16-blind": "K-row128-blind",
            "K-bins16-shuffled": "K-row128"}
    for arm, b in base.items():
        c = dict(P.J24_ARMS[arm])
        assert c.pop("eye") == "bins16" and c.pop("paint") == "kind"
        assert c.pop("shuffled", None) == ("bins" if arm == "K-bins16-shuffled" else None)
        ref = dict(P.J17_ARMS[b])
        ref.pop("eye")
        assert c == ref, arm
        assert "handed" not in c and c["ent_coef"] == "auto" and c["lr"] is None
    assert P.J24_SEEDS == (941, 942, 943) and P.J24_T == 500_000
    assert SMOKE_SEED not in P.J24_SEEDS


def _pair_run(a_arm, b_arm, n, check):
    A, B = _world(a_arm), _world(b_arm)
    oa, _ = A.reset(seed=SMOKE_SEED)
    ob, _ = B.reset(seed=SMOKE_SEED)
    nv = B.n_vision if B.n_vision else 80
    check(oa, ob, nv, -1)
    rng = np.random.RandomState(0)
    for t in range(n):
        a = rng.uniform(-1, 1, A.action_space.shape).astype(np.float32)
        ra, rb = A.step(a), B.step(a)
        assert np.array_equal(ra[0][nv:], rb[0][nv:]), t               # every other number identical
        assert ra[1] == rb[1] and ra[2] == rb[2] and ra[3] == rb[3], t # the same world underneath
        assert ra[4]["meals"] == rb[4]["meals"]
        check(ra[0], rb[0], nv, t)
        if ra[2] or ra[3]:
            oa, _ = A.reset()
            ob, _ = B.reset()
            check(oa, ob, nv, t)


def test_sighted_and_blind(n=1500):
    seen = []

    def check(a, b, nv, t):
        assert nv == 80 and a.shape == b.shape
        assert not b[:nv].any(), t
        seen.append(bool(a[:nv].any()))
    _pair_run("K-bins16", "K-bins16-blind", n, check)
    # control: the sighted bins DO carry content, in [0, 1]
    assert sum(seen) > 0.9 * len(seen), sum(seen)


def test_shuffled_moves_whole_bins(n=1500):
    from experiments.fast_sim import BINS, N_KINDS
    moved, looked = [0], [0]

    def check(a, b, nv, t):
        assert nv == N_KINDS * BINS == 80
        A = a[:nv].reshape(N_KINDS, BINS)
        S = b[:nv].reshape(N_KINDS, BINS)
        # each shuffled column is some sighted column, kinds together: same multiset of columns
        ca = sorted(map(tuple, A.T.tolist()))
        cs = sorted(map(tuple, S.T.tolist()))
        assert ca == cs, t
        if len(set(ca)) > 1:
            looked[0] += 1
            moved[0] += int(not np.array_equal(A, S))
    _pair_run("K-bins16", "K-bins16-shuffled", n, check)
    assert looked[0] > 0.5 * n and moved[0] > 0.9 * looked[0], (looked, moved)
    # control: a per-kind independent shuffle is NOT what the twin does — it fails the column check
    rng = np.random.default_rng(0)
    A = np.arange(N_KINDS * BINS, dtype=float).reshape(N_KINDS, BINS)
    bad = np.stack([row[rng.permutation(BINS)] for row in A])
    assert sorted(map(tuple, A.T.tolist())) != sorted(map(tuple, bad.T.tolist()))


def test_learner_cells_run_end_to_end(T=600):
    """Needs sbx (/data/venvs/jack-std/bin/python). Each J24 arm through the
    kernel's own cell function on a NON-rule seed: an 80-bin eye reaches SAC."""
    try:
        import sbx  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("sbx not in this venv (run under jack-std)")
    from experiments import pilot_b as B
    R = P.PILOT_RULE
    kw = dict(eye_frame=R["eye_frame"], motor_frame=R["motor_frame"],
              native_substep=R["native_substep"])
    old = (B.WARMUP, B.TRACE_EVERY)
    B.WARMUP, B.TRACE_EVERY = 100, 300
    try:
        for arm in P.J24_ARMS:
            rec = P.run_pilot_cell(arm, SMOKE_SEED, True, T=T, **kw)
            assert "error" not in rec and T <= rec["decisions"] < T + 16, rec.get("error")
            assert rec["pilot_arm"] == arm and rec["ent_coef_setting"] == "auto"
            assert rec["eye"] == "bins16" and rec["paint"] == "kind"
            assert rec["blind"] is (arm == "K-bins16-blind")
            assert rec.get("shuffled") == ("bins" if arm == "K-bins16-shuffled" else None)
    finally:
        B.WARMUP, B.TRACE_EVERY = old


def main():
    """PYTHONPATH=. /data/venvs/jack-std/bin/python -m experiments.tests.test_j24_arms"""
    import sys
    import traceback
    ok = True
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            ok = False
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
