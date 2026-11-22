"""J17 critic hunt (11-22): the SimBa arm pair, built beside J17's frozen arms.

Known answers, each with a control that must fail:
- J17's four arms, pilot 2's and E2c's arms/packing are byte-identical
  (J17_ARMS sha pinned here from the code at 2de6b1ea, before this edit).
- make_b() with no policy args hands sbx the OLD call: "MlpPolicy",
  layer_norm + 2x256 (captured by a stub SAC); control: a hunt arm's kwargs
  reach the stub as "SimbaPolicy" + SimBa's sizes + use_sde False.
- run_pilot_cell forwards policy keys ONLY for hunt arms (stub run_b);
  control: a J17 arm forwards none.
- The hunt pair differs ONLY in blind (planted: dropping blind makes the
  pair identical, so the twin is the one-variable control).
"""
import hashlib
import json

from experiments import pilot_b as B
from experiments import pilot_b2 as P

PRE_J17_ARMS = "5fa1c32fcdfb545b"               # from 2de6b1ea, before this edit


def _h(x):
    return hashlib.sha256(x.encode()).hexdigest()[:16]


def test_frozen_arms_unchanged():
    assert _h(json.dumps(P.J17_ARMS, sort_keys=True)) == PRE_J17_ARMS
    assert not set(P.J17_HUNT_ARMS) & (set(P.J17_ARMS) | set(P.PILOT_ARMS)
                                        | set(P.E2C_ARMS) | set(P.E2C_AUTO_ARMS))
    for d in (P.J17_ARMS, P.PILOT_ARMS, P.E2C_ARMS, P.E2C_AUTO_ARMS):
        for c in d.values():                 # floor arms' "policy" is their scripted floor
            if c["kind"] != "floor":
                assert not {"policy", "policy_kwargs", "use_sde"} & set(c)


class _Patch:
    """Set obj.name = value for the block, restore after (no pytest here)."""
    def __init__(self, obj, name, value):
        self.o, self.n, self.v = obj, name, value

    def __enter__(self):
        self.old = getattr(self.o, self.n)
        setattr(self.o, self.n, self.v)

    def __exit__(self, *a):
        setattr(self.o, self.n, self.old)


def _stub_sbx(got):
    import sys
    import types

    class SAC:
        def __init__(self, policy, venv, **kw):
            got.update(policy=policy, **kw)
    return _Patch(sys.modules, "sbx", types.SimpleNamespace(SAC=SAC)) if "sbx" in sys.modules \
        else _Insert(sys.modules, "sbx", types.SimpleNamespace(SAC=SAC))


class _Insert(_Patch):
    def __enter__(self):
        self.o[self.n] = self.v

    def __exit__(self, *a):
        self.o.pop(self.n, None)


class _PatchItem(_Patch):
    def __enter__(self):
        self.old = self.o[self.n]
        self.o[self.n] = self.v

    def __exit__(self, *a):
        self.o[self.n] = self.old


def test_make_b_default_is_old_call():
    got = {}
    with _stub_sbx(got):
        _make_b_checks(got)
    return "default = MlpPolicy LN 2x256 gSDE; hunt arm = SimbaPolicy pi128 qf512x2 no gSDE"


def _make_b_checks(got):
    B.make_b(None, 1, 1000)
    assert got["policy"] == "MlpPolicy"
    assert got["policy_kwargs"] == dict(layer_norm=True, net_arch=[256, 256])
    assert got["use_sde"] is True and got["ent_coef"] == 0.1
    c = P.J17_HUNT_ARMS["H-simba-row128"]
    got.clear()
    B.make_b(None, 1, 1000, use_sde=c["use_sde"], ent_coef="auto",
             policy=c["policy"], policy_kwargs=c["policy_kwargs"])
    assert got["policy"] == "SimbaPolicy" and got["use_sde"] is False      # control differs
    assert got["policy_kwargs"] == dict(net_arch=dict(pi=[128], qf=[512, 512]))


def test_cell_forwards_policy_only_for_hunt_arms():
    seen = {}

    def fake_run_b(seed, T, **kw):
        seen.clear(); seen.update(kw)
        return {"lives": []}
    with _Patch(B, "run_b", fake_run_b), _Patch(P, "pilot_world", lambda *a, **k: None):
        P.run_pilot_cell("H-simba-row128-blind", 923, True, T=10)
        assert seen["policy"] == "SimbaPolicy" and seen["use_sde"] is False
        assert seen["ent_coef"] == "auto" and seen["trace"] is True
        P.run_pilot_cell("K-row128", 923, True, T=10)                       # control
        assert not {"policy", "policy_kwargs", "use_sde"} & set(seen)
    return "hunt arm forwards policy/use_sde; J17 arm forwards none"


def test_pair_differs_only_in_blind():
    a = dict(P.J17_HUNT_ARMS["H-simba-row128"])
    b = dict(P.J17_HUNT_ARMS["H-simba-row128-blind"])
    assert b.pop("blind") is True and "blind" not in a
    assert a == b
    k = {x: y for x, y in a.items() if x not in ("policy", "policy_kwargs", "use_sde")}
    assert k == {x: y for x, y in P.J17_ARMS["K-row128"].items()}   # = J17's sighted arm + SimBa
    assert P.J17_HUNT_SEED == 923 and P.J17_HUNT_T == 100_000
    return "pair = J17 K-row128 + SimBa, twin differs only in blind"


def main():
    test_frozen_arms_unchanged()
    print("frozen arms: J17/pilot/E2c byte-identical, no policy keys")
    print("make_b:", test_make_b_default_is_old_call())
    print("run_pilot_cell:", test_cell_forwards_policy_only_for_hunt_arms())
    print("pair:", test_pair_differs_only_in_blind())
    print("test_j17_hunt_arms PASS")


if __name__ == "__main__":
    main()
