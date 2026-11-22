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
    return _PatchItem(sys.modules, "sbx", types.SimpleNamespace(SAC=SAC)) if "sbx" in sys.modules \
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


def test_h1_pair_is_j17_plus_adamw_only():
    """h1 (11-22): the AdamW-only pair = J17's own sighted/blind arms + ONE key."""
    for h, k in (("H-wd-row128", "K-row128"), ("H-wd-row128-blind", "K-row128-blind")):
        c = dict(P.J17_HUNT_ARMS[h])
        assert c.pop("optimizer") == "adamw"
        assert c == P.J17_ARMS[k]
    assert P.J17_HUNT_GROUPS == {"h0": ("H-simba-row128", "H-simba-row128-blind"),
                                 "h1": ("H-wd-row128", "H-wd-row128-blind"),
                                 "h2": ("H-ofn-row128", "H-ofn-row128-blind")}   # h2 added 11-22
    got = {}

    def fake_pilot_kernel(name, rule, cells=None, out_name=None):
        got[name] = (cells, out_name)
    with _Patch(P, "pilot_kernel", fake_pilot_kernel):
        P.j17_hunt_kernel("h0"); P.j17_hunt_kernel("h1")
    assert got["h0"] == ([("H-simba-row128", 923), ("H-simba-row128-blind", 923)],
                         "j17_hunt_h0.json")                     # h0 = what was launched
    assert got["h1"] == ([("H-wd-row128", 923), ("H-wd-row128-blind", 923)],
                         "j17_hunt_h1.json")
    seen = {}

    def fake_run_b(seed, T, **kw):
        seen.clear(); seen.update(kw)
        return {"lives": []}
    with _Patch(B, "run_b", fake_run_b), _Patch(P, "pilot_world", lambda *a, **k: None):
        P.run_pilot_cell("H-wd-row128", 923, True, T=10)
        assert seen["optimizer"] == "adamw" and "policy" not in seen
        P.run_pilot_cell("K-row128", 923, True, T=10)                      # control
        assert "optimizer" not in seen
    got2 = {}
    with _stub_sbx(got2):
        B.make_b(None, 1, 1000)
        assert "optimizer_class" not in got2["policy_kwargs"]               # default unchanged
        got2.clear()
        B.make_b(None, 1, 1000, ent_coef="auto", optimizer="adamw")
        import optax
        assert got2["policy_kwargs"] == dict(layer_norm=True, net_arch=[256, 256],
                                             optimizer_class=optax.adamw)
        assert got2["policy"] == "MlpPolicy" and got2["use_sde"] is True
    return "h1 = J17 K-row128 pair + adamw only; h0 cells as launched; forwarded only for H-wd"


def test_h1_real_critic_decays():
    """Real sbx (jack-std): with a ZERO gradient, the critic's adamw step moves
    every weight toward 0 by lr*1e-4*w (the update, unapplied); control: MlpPolicy's adam moves nothing."""
    try:
        import gymnasium as gym
        import jax
        import jax.numpy as jnp
        import optax
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        return "SKIPPED (no sbx in this venv)"
    out = {}
    for opt in (None, "adamw"):
        venv = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        m = B.make_b(venv, 0, 1000, optimizer=opt)
        st = m.policy.qf_state
        zero = jax.tree_util.tree_map(jnp.zeros_like, st.params)
        upd, _ = st.tx.update(zero, st.opt_state, st.params)
        w0 = jax.tree_util.tree_leaves(st.params)
        u = jax.tree_util.tree_leaves(upd)
        out[opt] = max(float(jnp.max(jnp.abs(x))) for x in u)
        if opt == "adamw":                  # the update itself (w*(1-3e-8) rounds in f32)
            for w, x in zip(w0, u):
                assert jnp.allclose(x, -3e-4 * 1e-4 * w, rtol=1e-4, atol=0), "not lr*wd*w"
    assert out[None] == 0.0 and out["adamw"] > 0.0, out
    return f"zero-grad step: adam moves {out[None]:.1e}, adamw moves {out['adamw']:.1e}"


def test_h2_pair_is_j17_plus_ofn_only():
    """h2 (11-22): the OFN pair = J17's own sighted/blind arms + ONE key; forwarded
    only for H-ofn; make_b hands sbx the OFN class only when asked."""
    for h, k in (("H-ofn-row128", "K-row128"), ("H-ofn-row128-blind", "K-row128-blind")):
        c = dict(P.J17_HUNT_ARMS[h])
        assert c.pop("critic") == "ofn"
        assert c == P.J17_ARMS[k]
    got = {}

    def fake_pilot_kernel(name, rule, cells=None, out_name=None):
        got[name] = (cells, out_name)
    with _Patch(P, "pilot_kernel", fake_pilot_kernel):
        P.j17_hunt_kernel("h2")
    assert got["h2"] == ([("H-ofn-row128", 923), ("H-ofn-row128-blind", 923)],
                         "j17_hunt_h2.json")
    seen = {}

    def fake_run_b(seed, T, **kw):
        seen.clear(); seen.update(kw)
        return {"lives": []}
    with _Patch(B, "run_b", fake_run_b), _Patch(P, "pilot_world", lambda *a, **k: None):
        P.run_pilot_cell("H-ofn-row128-blind", 923, True, T=10)
        assert seen["critic"] == "ofn" and "optimizer" not in seen and "policy" not in seen
        P.run_pilot_cell("H-wd-row128", 923, True, T=10)                   # control
        assert "critic" not in seen
    try:
        from experiments.ofn_critic import OFNVectorCritic
    except ImportError:
        return "h2 arms PASS; make_b class check SKIPPED (no sbx in this venv)"
    got2 = {}
    with _stub_sbx(got2):
        B.make_b(None, 1, 1000)
        assert "vector_critic_class" not in got2["policy_kwargs"]          # default unchanged
        got2.clear()
        B.make_b(None, 1, 1000, ent_coef="auto", critic="ofn")
        assert got2["policy_kwargs"] == dict(layer_norm=True, net_arch=[256, 256],
                                             vector_critic_class=OFNVectorCritic)
    return "h2 = J17 K-row128 pair + critic ofn only; forwarded only for H-ofn"


def test_h2_real_critic_is_bounded():
    """Real sbx: with unit-norm features, every Q obeys |Q - b| <= ||W_out||
    however large the input; control: sbx's own critic breaks that bound once
    the input is scaled x1e4 (its features grow with the input)."""
    try:
        import gymnasium as gym
        import jax.numpy as jnp
        import numpy as np
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        return "SKIPPED (no sbx in this venv)"
    rng = np.random.default_rng(0)
    out = {}
    for crit in (None, "ofn"):
        venv = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        m = B.make_b(venv, 0, 1000, critic=crit)
        st = m.policy.qf_state
        obs = jnp.asarray(rng.normal(size=(64, 3)) * 1e4, jnp.float32)
        act = jnp.asarray(rng.uniform(-1, 1, size=(64, 1)), jnp.float32)
        q = st.apply_fn(st.params, obs, act)                   # (n_critics, 64, 1)
        p = st.params["params"]["VmapOFNContinuousCritic_0" if crit else
                                "VmapContinuousCritic_0"]
        last = p[sorted(k for k in p if k.startswith("Dense_"))[-1]]
        bound = jnp.linalg.norm(last["kernel"][:, :, 0], axis=-1)[:, None, None]
        excess = float(jnp.max(jnp.abs(q - last["bias"][:, None, :]) - bound))
        out[crit] = excess
    assert out["ofn"] <= 1e-5, out
    assert out[None] > 0.0, out                                # control must break it
    return f"|Q-b| - ||W||: ofn {out['ofn']:.1e} (<= 0), sbx default {out[None]:.1e} (> 0)"


def main():
    test_frozen_arms_unchanged()
    print("frozen arms: J17/pilot/E2c byte-identical, no policy keys")
    print("make_b:", test_make_b_default_is_old_call())
    print("run_pilot_cell:", test_cell_forwards_policy_only_for_hunt_arms())
    print("pair:", test_pair_differs_only_in_blind())
    print("h1:", test_h1_pair_is_j17_plus_adamw_only())
    print("h1 real:", test_h1_real_critic_decays())
    print("h2:", test_h2_pair_is_j17_plus_ofn_only())
    print("h2 real:", test_h2_real_critic_is_bounded())
    print("test_j17_hunt_arms PASS")


if __name__ == "__main__":
    main()
