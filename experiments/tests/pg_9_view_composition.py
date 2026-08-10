"""PG.9 — The eye's view is not mostly obstacle.

WHY THIS EXISTS. PG.6 certified the playground eye five separate times — radius
R^2 0.97, bearing 1.27 deg, both nulls dead, the out-of-FOV control failing
exactly as it should — while the camera stared into a ladder 0.8 m from its
lens and a quarter of Jack's visual field was rungs. Every number was honest.
Not one of them was about the view.

A human rendered one frame and saw it in a second. The ladder could not,
because of 153 specs exactly one rendered an image at all. This spec is that
second of looking, made into a number that runs every time.

WHAT IT MEASURES, AND THE ATTEMPT THAT FAILED FIRST. Per-geom frame coverage
does not work: in the bad view no single geom exceeded 15% of the frame — the
ladder only reached 25.9% once rails and rungs were summed — so any per-geom
threshold passes the very view that motivated the spec. Grouping geoms into
structures by name prefix works but needs a hand-written list of what counts as
a structure, and it would not survive a world nobody has designed yet.

What makes an occluder harmful is that it is CLOSE. The ladder sat 0.8 m away;
the arena walls sit 5 m away and are backdrop, not obstruction. So the measure
is the fraction of the frame nearer than 1 m, read from the depth buffer. It
needs no names, and it will work unchanged on the jungle — which matters,
because the jungle is by definition cluttered.

THE CONTROL IS HISTORY, NOT INVENTION. The bad pose is re-rendered and required
to FAIL this spec's own threshold. A view-quality test that cannot flag the view
that caused it to be written is decoration. Its numbers are on record:

    pose                          near<1 m    floor
    (0, -3.4) north, behind ladder  22.2%     51.2%   <- must FAIL
    (-1.6, -3.4) yaw 20 deg          0.0%     61.8%   <- current, must PASS

THRESHOLDS ARE PRE-REGISTERED INTO THE GAP. 5% and 35% sit in empty space
between the two measurements rather than beside either. If a future camera
fails this, move the camera — do not move the number. The whole point of the
spec is that the framing is checkable at all.

NOTE WHAT THIS DOES NOT DO. It is not eyes. It catches one class of blind spot —
a view obstructed by something near — and says nothing about whether the scene
is meaningful, lit, or pointed at anything worth seeing. The general lesson
stands: the ladder verifies what someone thought to measure.
"""

from __future__ import annotations

import numpy as np

from ..render import ensure_gl, view_diagnostics

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import playground as pg  # noqa: E402

from ..protocol import Ledger, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402

# The camera pose is part of the world contract, so this spec's verdict depends
# on playground.py exactly as PG.6's does.
IMPL_DEPS = ["playground.py"]

NEAR_M = 1.0
NEAR_FRAC_GATE = 0.05      # frame fraction nearer than NEAR_M
FLOOR_FRAC_GATE = 0.35     # workspace visibility
CONTROL_MIN_NEAR = 0.15    # the bad pose must be caught this decisively

# The pose as it stood on 2026-08-09, kept verbatim as the control.
BAD_POS = (0.0, -3.4, 1.10)
BAD_XYAXES = (1.0, 0.0, 0.0, 0.0, 0.35, 0.94)


def _measure(seed: int, pos=None, xyaxes=None) -> dict:
    """Render the playground eye and report what the frame is made of.

    Restores the world contract in a finally block: this spec temporarily
    rewrites module-level camera constants to render the control pose, and
    leaking that into another spec in the same process would silently move
    every downstream visual measurement.
    """
    keep = (pg.EYE_POS, pg.EYE_XYAXES)
    try:
        if pos is not None:
            pg.EYE_POS, pg.EYE_XYAXES = pos, xyaxes
        model, data, _ = pg.make_playground(
            pg.PlaygroundParams(seed=seed), with_water=False)
        mujoco.mj_forward(model, data)
        return view_diagnostics(model, data, camera="eye", near_m=NEAR_M)
    finally:
        pg.EYE_POS, pg.EYE_XYAXES = keep


def _experiment(seed: int) -> dict:
    v = _measure(seed)
    m = {
        "near_field_frac": round(v["near_field_frac"], 4),
        "floor_frac": round(v["floor_frac"], 4),
        "median_depth_m": round(v["median_depth_m"], 3),
    }
    m["seed_gates_ok"] = float(
        m["near_field_frac"] < NEAR_FRAC_GATE
        and m["floor_frac"] >= FLOOR_FRAC_GATE)
    return m


def _control(seed: int) -> dict:
    """The pose that started this. It must fail the gate it inspired."""
    v = _measure(seed, BAD_POS, BAD_XYAXES)
    caught = float(v["near_field_frac"] >= CONTROL_MIN_NEAR)
    return {
        "bad_pose_near_field_frac": round(v["near_field_frac"], 4),
        "bad_pose_floor_frac": round(v["floor_frac"], 4),
        "control_seed_gates_ok": caught,
    }


def _check(m: dict, c: dict) -> bool:
    return (m["seed_gates_ok"] == 1.0
            and m["near_field_frac"] < NEAR_FRAC_GATE
            and m["floor_frac"] >= FLOOR_FRAC_GATE
            and c["control_seed_gates_ok"] == 1.0
            and c["bad_pose_near_field_frac"] >= CONTROL_MIN_NEAR)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.9"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
