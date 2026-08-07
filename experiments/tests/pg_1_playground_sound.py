"""PG.1 — the playground must build and obey physics before anything learns in it.

A broken world teaches broken lessons, and every curiosity claim in Tier 5 rests
on this one. The checks are the cheapest possible and each corresponds to a way
a procedural MJCF world silently goes wrong:

  builds        every landmark geom present (a typo in a template yields a world
                missing its ladder, and nothing downstream would say so)
  mutates       ACCEL-style edits stay valid MJCF — the open-ended loop mutates
                this world thousands of times unattended
  settles       energy bounded at rest; objects do not jitter or launch
  friction      a box slides iff tan(theta) > mu, i.e. the ramp is a real ramp
                and not a conveyor belt

The friction check is the one with teeth. It compares against the analytic
prediction rather than "looks fine": at mu=0.9 a box must HOLD on a 15-degree
ramp (tan 15 = 0.27) and must SLIDE on a 50-degree one (tan 50 = 1.19). A world
where everything slides, or nothing does, passes a visual inspection.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

LANDMARKS = ["floor", "ramp", "stair0", "rung0", "platform", "apple",
             "pool_water", "pool_floor", "welded_block", "noise_panel",
             "seesaw_plank"]
SETTLE_STEPS = 400
MAX_REST_VEL = 0.05          # m/s at rest
N_MUTATIONS = 12


def _slides(angle_deg: float, mu: float, steps: int = 600) -> float:
    """Distance a box travels down a ramp of this angle. Analytic check.

    Two things here are easy to get wrong and both were:

    MJCF's default angle unit is DEGREES (compiler angle="degree"). Passing
    math.radians() into euler= therefore built a 0.87-degree "50-degree" ramp,
    and nothing slid on anything -- the frictionless control crept 0.0455m when
    it should have run metres. A control that fails alongside the experiment is
    what exposed this; had it been absent, "nothing slides" would have read as a
    plausible high-friction world.

    The slider is a flat SLAB, not a cube. A cube topples once tan(theta)
    exceeds width/height = 1, i.e. above 45 degrees, so a 50-degree cube test
    would measure tumbling and call it sliding. At 0.12 x 0.04 the slab needs
    tan(theta) > 3 (71.6 degrees) to tip, leaving 50 degrees an honest slide.
    """
    import mujoco
    import numpy as np

    xml = f"""<mujoco><option timestep="0.005" gravity="0 0 -9.81"/><worldbody>
      <geom name="ramp" type="box" pos="0 0 0" size="3 1 0.02"
            euler="0 {-angle_deg} 0" friction="{mu} 0.005 0.0001"/>
      <body pos="0 0 0.12" euler="0 {-angle_deg} 0"><freejoint/>
      <geom type="box" size="0.12 0.12 0.04" mass="1"
            friction="{mu} 0.005 0.0001"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    # Seat the box on the incline before measuring, or the drop dominates.
    for _ in range(200):
        mujoco.mj_step(m, d)
    x0 = float(d.qpos[0])
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return abs(float(d.qpos[0]) - x0)


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    from playground import PlaygroundParams, build_mjcf, make_playground

    p = PlaygroundParams(seed=seed)
    model, data, water = make_playground(p)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(model.ngeom)}
    missing = [g for g in LANDMARKS if g not in names]

    for _ in range(SETTLE_STEPS):
        data.xfrc_applied[:] = 0
        if water:
            water.apply(model, data)
        mujoco.mj_step(model, data)
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    rest_vel = float(np.abs(data.qvel).max()) if model.nv else 0.0

    # Mutation robustness: the open-ended loop edits this world unattended.
    rng = np.random.RandomState(seed + 7)
    mutated_ok = 0
    q = p
    for _ in range(N_MUTATIONS):
        q = q.mutate(rng)
        try:
            mujoco.MjModel.from_xml_string(build_mjcf(q))
            mutated_ok += 1
        except Exception:
            pass

    shallow = _slides(15.0, mu=0.9)     # tan15=0.27 < 0.9 -> must HOLD
    steep = _slides(50.0, mu=0.9)       # tan50=1.19 > 0.9 -> must SLIDE
    return {
        "geoms": int(model.ngeom), "bodies": int(model.nbody),
        "missing_landmarks": ",".join(missing) or "none",
        "finite_after_settle": finite,
        "max_rest_velocity": round(rest_vel, 5),
        "mutations_valid": mutated_ok, "mutations_tried": N_MUTATIONS,
        "slide_15deg": round(shallow, 4),
        "slide_50deg": round(steep, 4),
        "friction_discriminates": round(steep / max(shallow, 1e-4), 1),
    }


def _control(seed: int) -> dict:
    """Frictionless: the shallow ramp that HELD must now slide.

    Without this, "the box held" could just mean the box was stuck on geometry
    rather than obeying friction.
    """
    sys.path.insert(0, str(REPO))
    return {"frictionless_slide_15deg": round(_slides(15.0, mu=0.0), 4)}


def _check(m: dict, c: dict) -> bool:
    return (m["missing_landmarks"] == "none"
            and m["finite_after_settle"]
            and m["max_rest_velocity"] <= MAX_REST_VEL
            and m["mutations_valid"] == m["mutations_tried"]
            and m["slide_15deg"] < 0.05          # holds
            and m["slide_50deg"] > 0.5           # slides
            and c["frictionless_slide_15deg"] > 0.5)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.1"], _experiment, _check, control_fn=_control, ledger=ledger)
