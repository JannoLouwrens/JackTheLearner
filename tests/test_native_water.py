"""J6 (2b) piece 3: C `jn_water_apply` == `playground.Water.apply`.

Shadow harness on live JackEnv sub-steps (shipped cal, lazy_contact_audio).
The C state (`_last`, `_written`, `_last_any`) is loaded from the Python
Water ONCE when the env is built and then evolves on its own for the whole
run. At every sub-step: snapshot `xfrc_applied`, run Python, restore, run
C, compare the whole `xfrc_applied` array AND the state with exact float
equality; restore, run the control, compare; leave Python's result in place.

Cases (each must exercise its branch, asserted):
  world  seeds 153-155, random actions                     — today's world
  held   seeds 153-155, body dropped at the pool centre, zero action,
         surface 0 (W0's floor covers the pool): parts of him inside the
         pool box touch the floor and are HELD DRY              — held branch
  float  same drop, the Water's surface raised 0 -> 0.5 m and its dry-geom
         set recomputed for that surface, in Python (C loads from it) —
         buoyancy + quadratic + linear drag, the additive write
Control (must DIFFER): the plain-order norm instead of the box BLAS's FMA.
"""
import mujoco
import numpy as np

from experiments.jack_env import JackEnv
from experiments.native_substep import NativeWater

SEEDS = (153, 154, 155)
DECISIONS = 150


def _raise(water, surf):
    m = water.model
    water.surface_z = surf
    water._dry_geoms = set()
    for gid in range(m.ngeom):          # the constructor's rule, new surface
        if m.geom_bodyid[gid] != 0:
            continue
        if m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0:
            continue
        gz = float(m.geom_pos[gid][2])
        top = gz if m.geom_type[gid] == 0 else gz + float(m.geom_size[gid][2])
        if top >= surf - 1e-3:
            water._dry_geoms.add(gid)


def _drop_in_pool(env):
    w, W = env.w, env.w.water
    adr = int(w.model.jnt_qposadr[w.model.body_jntadr[w.rover_bid]])
    w.data.qpos[adr:adr + 2] = (W.x, W.y)
    w.data.qvel[:] = 0.0
    mujoco.mj_forward(w.model, w.data)


def _run(seed, *, drop=False, surf=None):
    env = JackEnv(seed=seed, lazy_contact_audio=True)
    env.reset(seed=seed)
    st = dict(calls=0, bad=0, ctrl_bad=0, applied=0, held=0, fields=set())
    hooked = set()

    def hook(W):
        if id(W) in hooked:
            return
        hooked.add(id(W))
        if surf is not None:
            _raise(W, surf)
        nw, ctrl = NativeWater(W), NativeWater(W, fma_norm=False)
        orig = W.apply

        def apply(model, data):
            m = model
            for bid in range(1, m.nbody):
                r = float(W._radius[bid])
                if r > 0 and m.body_mass[bid] > 0 and W.in_pool(data.xipos[bid], r):
                    if bid in W._held_dry(data):
                        st["held"] += 1
                        break
            pre = data.xfrc_applied.copy()
            orig(model, data)
            py = data.xfrc_applied.copy()
            data.xfrc_applied[:] = pre
            nw.apply(data)
            d = nw.diff() + ([] if np.array_equal(data.xfrc_applied, py) else ["xfrc"])
            data.xfrc_applied[:] = pre
            ctrl.apply(data)
            cb = bool(ctrl.diff()) or not np.array_equal(data.xfrc_applied, py)
            data.xfrc_applied[:] = py
            st["calls"] += 1
            st["bad"] += bool(d)
            st["fields"].update(d)
            st["ctrl_bad"] += cb
            st["applied"] += bool(W._last_any)
        W.apply = apply
    hook(env.w.water)
    if drop:
        _drop_in_pool(env)
    rng = np.random.default_rng(seed)
    for _ in range(DECISIONS):
        a = (np.zeros(env.action_space.shape) if drop
             else rng.uniform(-1, 1, env.action_space.shape))
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
            hook(env.w.water)
            if drop:
                _drop_in_pool(env)
    return st


CASES = (("world", {}, ()),
         ("held", {"drop": True}, ("held",)),
         ("float", {"drop": True, "surf": 0.5}, ("applied",)))


def test_native_water_identical():
    for name, kw, need in CASES:
        for s in SEEDS:
            st = _run(s, **kw)
            print(name, s, st, flush=True)
            assert st["calls"] >= DECISIONS * 40
            assert st["bad"] == 0, st["fields"]
            for k in need:
                assert st[k] > 0, (name, s, k)       # the branch was exercised
            if name == "float":
                assert st["ctrl_bad"] > 0            # the control must fail


if __name__ == "__main__":
    test_native_water_identical()
    print("PASS")
