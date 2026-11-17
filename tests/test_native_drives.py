"""J6 (2b) piece 2: C `jn_drives_substep` == `drives.DriveLayer.substep`.

Shadow harness on live JackEnv sub-steps (shipped cal, lazy_contact_audio).
At the FIRST sub-step of every decision the C struct is loaded from the
Python layer; it then evolves on its own through all 40 sub-steps and every
field (6 accumulators, w, t, n_onsets, contact edge, prev speed, submersion
clock, wet flag, every food timer and meal count) is compared with exact
float equality after EVERY sub-step.

Cases (each must exercise its branch, asserted):
  seeds 153-155, today's world, random actions        — power/rest/impact
  seeds 153-155, zero action, body set on a live food every 10 decisions — eating + timers
  seeds 153-155, body dropped in the pool region, zero action, the drive
    layer's surface raised 0 -> 0.5 m in Python AND C (W0's floor covers the
    pool, so in today's world no geom can be below 0)       — wet + drowning
Control (must DIFFER): the same C with the plain-order norm instead of
the box BLAS's FMA order — proves the harness sees last-bit float error.
"""
import mujoco
import numpy as np

from experiments.jack_env import JackEnv
from experiments.native_substep import NativeDrives
from experiments import w0 as w0mod

SEEDS = (153, 154, 155)
DECISIONS = 150


def _drop_in_pool(env):
    x, y, half, surf = env.w.drives.pool
    _drop_at(env, x, y)


def _drop_on_food(env):
    w, L = env.w, env.w.drives
    for name, (gid, _) in L._food.items():
        if L._respawn_at[name] <= L.t and w.data.geom_xpos[gid][2] < 0.5:
            x, y = w.data.geom_xpos[gid][:2]
            _drop_at(env, x, y)
            return


def _drop_at(env, x, y):
    w = env.w
    adr = int(w.model.jnt_qposadr[w.model.body_jntadr[w.rover_bid]])
    w.data.qpos[adr:adr + 2] = (x, y)
    w.data.qvel[:] = 0.0
    mujoco.mj_forward(w.model, w.data)


def _run(seed, *, difficulty=None, pool=False, still=False, feed=False):
    env = JackEnv(seed=seed, lazy_contact_audio=True, difficulty=difficulty)
    env.reset(seed=seed)
    st = dict(calls=0, bad=0, ctrl_bad=0, eats=0, onsets=0, wet=0, drown=0,
              rest=0, fields=set())
    shadows = {}

    def hook(layer):
        if id(layer) in shadows:
            return
        if pool:   # W0's floor plane covers the pool (no pit): no geom can
            x, y, half, _ = layer.pool   # get below z = 0, so the branch is
            layer.pool = (x, y, half, 0.5)   # planted in BOTH implementations
        nd, ctrl = NativeDrives(layer), NativeDrives(layer, fma_norm=False)
        orig = layer.substep

        def sub(model, data, dt):
            if layer._dt_acc == 0.0:
                nd.load(); ctrl.load()
            ate0, on0 = sum(layer.ate_total.values()), layer.n_onsets
            orig(model, data, dt)
            nd.substep(data, dt); ctrl.substep(data, dt)
            d = nd.diff()
            st["calls"] += 1
            st["bad"] += bool(d)
            st["fields"].update(d)
            st["ctrl_bad"] += bool(ctrl.diff())
            st["eats"] += sum(layer.ate_total.values()) > ate0
            st["onsets"] += layer.n_onsets > on0
            st["wet"] += bool(layer._wet_in)
            st["drown"] += layer._drown_dt > 0
            st["rest"] += layer._rest_dt > 0
        layer.substep = sub
        shadows[id(layer)] = sub
    hook(env.w.drives)
    if pool:
        _drop_in_pool(env)
    rng = np.random.default_rng(seed)
    for i in range(DECISIONS):
        if feed and i % 10 == 0:
            _drop_on_food(env)
        a = (np.zeros(env.action_space.shape) if (pool or still)
             else rng.uniform(-1, 1, env.action_space.shape))
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
            hook(env.w.drives)
            if pool:
                _drop_in_pool(env)
    return st


CASES = (("world", {}, ("onsets", "rest")),
         ("eat", {"still": True, "feed": True}, ("eats",)),
         ("pool", {"pool": True}, ("wet", "drown")))


def test_native_drives_identical():
    for name, kw, need in CASES:
        for s in SEEDS:
            st = _run(s, **kw)
            print(name, s, st)
            assert st["calls"] >= DECISIONS * 40
            assert st["bad"] == 0, st["fields"]
            for k in need:
                assert st[k] > 0, (name, s, k)       # the branch was exercised
            assert st["ctrl_bad"] > 0                # the control must fail


if __name__ == "__main__":
    test_native_drives_identical()
    print("PASS")
