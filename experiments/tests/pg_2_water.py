"""PG.2 — the water must be real water, checked against theory.

Archimedes gives an exact prediction with no free parameters: a sphere of
density ratio r floats with submerged fraction r. So this spec does not ask
"does it look wet" — it computes the equilibrium depth analytically and demands
the simulation land there.

That standard earned its keep. Four bugs were found this way, and every one of
them produced water that looked entirely convincing:

  1. Radius derived from body_inertia. For a sphere I = 2/5 m r^2, so sqrt(I)
     scales with sqrt(MASS): denser bodies got a larger inferred radius and
     therefore MORE buoyancy. Measured, rho=0.8 floated at +0.21 while rho=0.3
     sat at -0.03 — the physics ran backwards.
  2. No linear damping. Quadratic drag vanishes as v->0, so nothing settled: a
     float still oscillated at |vz| ~ 0.9 after 40,000 steps.
  3. in_pool() tested the body CENTRE against the surface, so any partially
     submerged body whose centre rode above the waterline received ZERO
     buoyancy — i.e. everything with density < 0.5, exactly the things meant to
     float. The submerged-fraction maths was correct all along; the gate in
     front of it silently returned zero.
  4. Sampling a genuinely bobbing float at one instant instead of time-averaging.

CONTROL: with buoyancy disabled every density must sink. Without that, "it
floats" could mean the body is resting on the pool floor.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# This spec certifies a property of the WORLD, so the world hashes into
# impl_sha. Change playground.py and this certificate goes stale loudly
# instead of standing over a world it no longer describes.
IMPL_DEPS = ["playground.py"]

REPO = Path(__file__).resolve().parents[2]

DENSITY_RATIOS = [0.2, 0.3, 0.5, 0.8]
RADIUS = 0.15
SETTLE_STEPS = 4000
WINDOW_STEPS = 3000
MAX_DEPTH_ERROR_FRAC = 0.10      # of the sphere radius; the spec's ±10%


def _expected_z(ratio: float, r: float) -> float:
    """Centre height at which submerged fraction equals the density ratio.

    Spherical cap: frac(h) = h^2 (3r - h) / (4 r^3), h = depth of the lowest
    point below the surface. Solved numerically because the cubic's closed form
    adds nothing here.
    """
    import numpy as np
    ds = np.linspace(-r, r, 20001)
    frac = np.clip(((ds + r) ** 2 * (3 * r - (ds + r))) / (4 * r ** 3), 0, 1)
    return float(-ds[int(np.argmin(np.abs(frac - ratio)))])


def _float_depth(ratio: float, buoyancy: bool = True) -> tuple:
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    from playground import Water, WATER_DENSITY

    vol = 4.0 / 3.0 * math.pi * RADIUS ** 3
    mass = ratio * WATER_DENSITY * vol
    xml = f"""<mujoco><option timestep="0.005" gravity="0 0 -9.81"/><worldbody>
      <geom type="plane" pos="0 0 -2" size="5 5 0.1"/>
      <body pos="0 0 0.5"><freejoint/>
      <geom type="sphere" size="{RADIUS}" mass="{mass}"/></body>
      </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    w = Water(m, x=0.0, y=0.0, half=2.0, depth=2.0)
    w.enabled = buoyancy
    for _ in range(SETTLE_STEPS):
        d.xfrc_applied[:] = 0
        w.apply(m, d)
        mujoco.mj_step(m, d)
    zs = []
    for _ in range(WINDOW_STEPS):
        d.xfrc_applied[:] = 0
        w.apply(m, d)
        mujoco.mj_step(m, d)
        zs.append(float(d.qpos[2]))
    # Time-averaged: a real float bobs, so an instantaneous sample is the wrong
    # measurement, not a tighter one.
    return float(np.mean(zs)), float(np.std(zs))


def _experiment(seed: int) -> dict:
    out = {}
    worst = 0.0
    for r in DENSITY_RATIOS:
        z, bob = _float_depth(r)
        exp = _expected_z(r, RADIUS)
        err = abs(z - exp) / RADIUS
        worst = max(worst, err)
        out[f"z_rho{r}"] = round(z, 4)
        out[f"expected_rho{r}"] = round(exp, 4)
        out[f"err_rho{r}"] = round(err, 4)
        out[f"bob_rho{r}"] = round(bob, 5)
    out["worst_depth_error_frac"] = round(worst, 4)
    return out


def _control(seed: int) -> dict:
    """No buoyancy: everything sinks. Otherwise 'floats' may mean 'sits on the floor'."""
    sunk = [_float_depth(r, buoyancy=False)[0] for r in (0.2, 0.5)]
    return {"no_buoyancy_z": [round(z, 3) for z in sunk],
            "no_buoyancy_max_z": round(max(sunk), 3)}


def _check(m: dict, c: dict) -> bool:
    return (m["worst_depth_error_frac"] <= MAX_DEPTH_ERROR_FRAC
            and c["no_buoyancy_max_z"] < -1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.2"], _experiment, _check, control_fn=_control, ledger=ledger)
