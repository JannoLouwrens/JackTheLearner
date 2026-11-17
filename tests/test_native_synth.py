"""J6 (2b) piece 4: C `jn_synth_step` (+ the shipped Python event builder for
the pairs it fires) == `fast_sim._lazy_step` (itself proven == the shipped
`ContactAudioSynth.step`, experiments/tests/test_fast_sim.py).

Shadow harness on live JackEnv sub-steps (shipped cal, lazy_contact_audio).
C state (`_prev_pairs`, `_last_fired`) is loaded ONCE per synth object and
then evolves alone across every sub-step, decision and reset. After every
sub-step: the events each side produced (every field, exact) and the state.

Cases: seeds 153-155 x {random actions; body set on a live floor food every
10 decisions (new pairs: him x food)}. Must be exercised: events fired, and
pairs skipped by the refractory window (counted by a control with the window
set to 0 — which therefore must fire MORE events: the control that must fail).
"""
import mujoco
import numpy as np

from experiments.jack_env import JackEnv
from experiments.native_substep import NativeSynth

SEEDS = (153, 154, 155)
DECISIONS = 150
FIELDS = ("t", "geom1", "geom2", "voiced_geom", "force", "amp", "azimuth",
          "lateral", "elevation", "distance")


def _same(a, b):
    return (len(a) == len(b) and all(
        all(getattr(x, k) == getattr(y, k) for k in FIELDS)
        and np.array_equal(x.pos, y.pos) for x, y in zip(a, b)))


def _drop_on_food(env):
    w, L = env.w, env.w.drives
    for name, (gid, _) in L._food.items():
        if L._respawn_at[name] <= L.t and w.data.geom_xpos[gid][2] < 0.5:
            adr = int(w.model.jnt_qposadr[w.model.body_jntadr[w.rover_bid]])
            w.data.qpos[adr:adr + 2] = w.data.geom_xpos[gid][:2]
            w.data.qvel[:] = 0.0
            mujoco.mj_forward(w.model, w.data)
            return


def _run(seed, feed=False):
    env = JackEnv(seed=seed, lazy_contact_audio=True)
    env.reset(seed=seed)
    st = dict(calls=0, bad=0, events=0, ctrl_extra=0, resets=0, fields=set())
    hooked = set()

    def hook(S):
        if id(S) in hooked:
            return
        hooked.add(id(S))
        ns, ctrl = NativeSynth(S), NativeSynth(S)
        ctrl.cfg.refr = 0.0
        orig = S.step

        def step(data):
            n0 = len(S.events)
            orig(data)
            py = S.events[n0:]
            ns.events, ctrl.events = [], []
            ns.step(data); ctrl.step(data)
            d = ns.diff() + ([] if _same(ns.events, py) else ["events"])
            st["calls"] += 1
            st["bad"] += bool(d)
            st["fields"].update(d)
            st["events"] += len(py)
            st["ctrl_extra"] += len(ctrl.events) - len(py)
        S.step = step
    hook(env.w.synth)
    rng = np.random.default_rng(seed)
    for i in range(DECISIONS):
        if feed and i % 10 == 0:
            _drop_on_food(env)
        a = (np.zeros(env.action_space.shape) if feed
             else rng.uniform(-1, 1, env.action_space.shape))
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            st["resets"] += 1
            env.reset()
            hook(env.w.synth)
    return st


def test_native_synth_identical():
    for name, feed in (("world", False), ("eat", True)):
        for s in SEEDS:
            st = _run(s, feed)
            print(name, s, st, flush=True)
            assert st["calls"] >= DECISIONS * 40
            assert st["bad"] == 0, st["fields"]
            assert st["events"] > 0
            assert st["ctrl_extra"] > 0          # refractory exercised; control fails


if __name__ == "__main__":
    test_native_synth_identical()
    print("PASS")
