"""PG.3 — the ladder must be climbable in principle, and falls must be clean.

GOAL.md's founding image is a ladder with an apple on top: Jack must be able to
climb it, fall off it, and come back. Before any learning claim touches that
ladder, the FIXTURE itself must be certified: can MuJoCo adhesion actuators
(the stand-in for grip — Humanoid-v5 has ball hands and no fingers) actually
support a body's weight on THESE rungs, and does a fall leave the episode
stream intact? If not, every curiosity-climbs-the-ladder spec upstream is
untestable and the playground needs redesign, not training.

The rig is deliberately minimal — a certification jig, not a humanoid: a 30 kg
torso hanging from two arms, each a reach(y)+lift(z) slide pair ending in an
adhesion hand. Hands grip rungs from BELOW, so gravity is opposed by adhesion
alone: with the gain at zero there is nothing — no friction path, no resting
contact — that can hold the hang, which is what makes the null a real null.
The scripted sequence is real climbing: hang, release one hand, swing it
around the rung (reach out, lift, reach in, press), re-grip, same for the
other hand, then pull the body up one rung spacing.

Declared rig conveniences (this spec certifies the LADDER + adhesion physics,
not aerodynamics): the free joint carries damping=10 so the pendulum swing of
a point-gripped body settles within a phase, and the torso geom is masked out
of ladder collisions (contype 1 vs the ladder's 4) because the rig's torso
hangs in the rung plane where a real climber's body would be in front of it.
Hands collide with everything (contype 5). Neither convenience can fake the
claim: adhesion still carries the full 302 N or it does not.

Three sub-claims, all pre-registered:
  hold    hanging from both hands moves the torso < 8 cm over the settle phase
  ascend  the scripted sequence gains >= 0.7 rung spacings of torso height,
          across three rung spacings (0.26/0.30/0.34 m) spanning the middle of
          the mutation range — "climbable" must survive the world mutating
  fall    releasing both hands mid-air: states stay finite, the body comes to
          rest on the floor, a mid-fall snapshot restores bit-for-bit into a
          fresh MjData (resumable), and after mj_resetData the rig grips and
          holds again (the episode stream continues after a fall)

CONTROL: the identical script with adhesion never energised must slip — the
torso drops during the very first hang, gains nothing, and ends the script on
the floor. The drop threshold is 0.15 m, not free-fall distance, because the
ungripped hands land and REST on the rung below (plain normal contact), which
is legitimate physics and exactly the distinction under test: a body can rest
on a rung without grip, but it cannot HOLD a hang or ascend. If the control
climbs, the rig is standing on geometry and adhesion proved nothing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

HAND_R = 0.04
RUNG_R = 0.028                  # playground.py rung capsule size
GRIP_OFF = HAND_R + RUNG_R      # hand-centre to rung-axis distance at touch
LADDER_Y = -2.6                 # playground.py ladder plane
SPACINGS = (0.30, 0.26, 0.34)   # per-seed rung spacing, mid-mutation-range
ADHESION_GAIN = 900.0           # N per hand; body weight is ~302 N
DT = 0.005

HOLD_DRIFT_MAX = 0.08           # m, pre-registered
ASCENT_FRAC_MIN = 0.70          # of one rung spacing
FALL_REST_SPEED_MAX = 0.5      # m/s on climber dofs after the fall settles
FALL_REST_TORSO_Z_MAX = 0.35    # m: torso geom centre must end near the floor
RESUME_DEV_MAX = 1e-6           # snapshot restore must be deterministic
CONTROL_DROP_MIN = 0.15         # m the null must slip during the first hang
                                # (ungripped hands come to REST on the rung
                                # below — resting is allowed, holding is not)


def _build(seed: int, spacing: float):
    """Playground + climber rig. Returns (model, ids, rung_zs, grip_rung_idx)."""
    sys.path.insert(0, str(REPO))
    import mujoco
    from playground import PlaygroundParams, build_mjcf

    p = PlaygroundParams(seed=seed, ladder_rung_spacing=spacing)
    xml = build_mjcf(p)
    # Ladder geoms into their own contact class (4): hands (5) grip them, the
    # rig torso (1) passes through them, the floor (1) still catches everything.
    xml = re.sub(r'(<geom name="(?:rung\d+|ladder_rail[LR])" )',
                 r'\1contype="4" conaffinity="4" ', xml)

    rungs = [(i + 1) * spacing for i in range(p.ladder_rungs)
             if (i + 1) * spacing < p.ladder_height]
    # Grip the highest rung <= 1.31 m that still has a rung above it: high
    # enough that the hanging torso clears the floor, low enough to ascend.
    k = max(i for i, z in enumerate(rungs) if z <= 1.31 and i + 1 < len(rungs))
    z0 = rungs[k] - GRIP_OFF - 0.003        # 3 mm slack, inside the 15 mm margin

    def arm(side: str, x: float) -> str:
        return (f'<body name="arm{side}" pos="{x} 0 0">'
                f'<joint name="reach{side}" type="slide" axis="0 1 0" range="-0.25 0.05" damping="40"/>'
                f'<joint name="lift{side}" type="slide" axis="0 0 1" range="-0.2 0.55" damping="40"/>'
                f'<geom name="hand{side}" type="sphere" size="{HAND_R}" mass="0.4" '
                f'margin="0.015" gap="0.015" contype="5" conaffinity="5" '
                f'friction="1.2 0.05 0.001" rgba="0.9 0.7 0.5 1"/></body>')

    climber = (
        f'<body name="climber" pos="0 {LADDER_Y} {z0:.4f}">'
        f'<joint name="climber_root" type="free" damping="10"/>'
        f'<geom name="climber_torso" type="capsule" fromto="0 0 -0.65 0 0 -0.25" '
        f'size="0.07" mass="30" contype="1" conaffinity="1" rgba="0.3 0.5 0.8 1"/>'
        f'{arm("L", -0.10)}{arm("R", 0.10)}</body>')
    actuators = (
        '<actuator>'
        '<position name="a_reachL" joint="reachL" kp="1500" ctrlrange="-0.25 0.05" forcerange="-400 400"/>'
        '<position name="a_liftL" joint="liftL" kp="3000" ctrlrange="-0.2 0.55" forcerange="-600 600"/>'
        '<position name="a_reachR" joint="reachR" kp="1500" ctrlrange="-0.25 0.05" forcerange="-400 400"/>'
        '<position name="a_liftR" joint="liftR" kp="3000" ctrlrange="-0.2 0.55" forcerange="-600 600"/>'
        f'<adhesion name="a_adhL" body="armL" ctrlrange="0 1" gain="{ADHESION_GAIN}"/>'
        f'<adhesion name="a_adhR" body="armR" ctrlrange="0 1" gain="{ADHESION_GAIN}"/>'
        '</actuator>')
    xml = xml.replace("</worldbody>", climber + "\n  </worldbody>")
    xml = xml.replace("</mujoco>", actuators + "\n</mujoco>")

    model = mujoco.MjModel.from_xml_string(xml)
    ids = {n: model.actuator(f"a_{n}").id
           for n in ("reachL", "liftL", "reachR", "liftR", "adhL", "adhR")}
    return model, ids, rungs, k


def _phase(model, data, ids, targets: dict, steps: int) -> None:
    """Ramp the named actuators linearly from their current ctrl to targets."""
    import mujoco
    start = {n: float(data.ctrl[ids[n]]) for n in targets}
    for t in range(steps):
        f = (t + 1) / steps
        for n, tgt in targets.items():
            data.ctrl[ids[n]] = start[n] + f * (tgt - start[n])
        mujoco.mj_step(model, data)


def _climber_dofs(model) -> list:
    dofs = []
    for jn in ("climber_root", "reachL", "liftL", "reachR", "liftR"):
        j = model.joint(jn)
        dofs.extend(range(int(j.dofadr[0]),
                          int(j.dofadr[0]) + (6 if jn == "climber_root" else 1)))
    return dofs


def _advance_hand(model, data, ids, side: str, spacing: float) -> None:
    """Move one hand around its rung to the next one up and re-grip it.

    The press target overshoots the touch point by 0.15 m because single-hand
    support sags the torso (weight/kp ~ 0.1 m) and the pendulum tilt dips the
    raised hand a few more cm; the overshoot turns all of that into a bounded
    press force (kp * err <= ~300 N) against the rung's underside instead of a
    missed grip. After adhesion is back on, the target relaxes to +0.03 so the
    other hand's phase does not carry the press as extra load.
    """
    adh, reach, lift = f"adh{side}", f"reach{side}", f"lift{side}"
    data.ctrl[ids[adh]] = 0.0                                # let go
    _phase(model, data, ids, {reach: -0.12}, 250)            # swing out of the rung plane
    _phase(model, data, ids, {lift: spacing - 0.02}, 400)    # rise past the gripped rung
    _phase(model, data, ids, {reach: 0.0}, 250)              # swing back under the next rung
    _phase(model, data, ids, {lift: spacing + 0.15}, 300)    # press into its underside
    data.ctrl[ids[adh]] = 1.0                                # grip
    _phase(model, data, ids, {}, 150)
    _phase(model, data, ids, {lift: spacing + 0.03}, 150)    # relax the press


def _run_script(seed: int, spacing: float, adhesion_on: bool) -> dict:
    import mujoco
    import numpy as np

    model, ids, rungs, k = _build(seed, spacing)
    data = mujoco.MjData(model)
    dofs = _climber_dofs(model)
    torso = model.geom("climber_torso").id

    def torso_z() -> float:
        return float(data.geom_xpos[torso][2])

    grip = 1.0 if adhesion_on else 0.0
    mujoco.mj_forward(model, data)      # geom_xpos is zeros until a forward pass
    z_initial = torso_z()

    # ── hang: both hands on rung k, full weight on adhesion ─────────────
    data.ctrl[ids["adhL"]] = data.ctrl[ids["adhR"]] = grip
    _phase(model, data, ids, {}, 300)
    z_hold_a = torso_z()
    _phase(model, data, ids, {}, 300)
    z_hold_b = torso_z()
    hold_drift = abs(z_hold_b - z_hold_a)
    hold_drop = z_initial - z_hold_b

    # ── one rung up: hand over hand, then pull ──────────────────────────
    _advance_hand(model, data, ids, "R", spacing)
    _advance_hand(model, data, ids, "L", spacing)
    _phase(model, data, ids, {"liftL": 0.0, "liftR": 0.0}, 500)
    zs = []
    for _ in range(200):
        mujoco.mj_step(model, data)
        zs.append(torso_z())
    z_top = float(np.mean(zs[-100:]))
    ascent = z_top - z_hold_b

    # Both hands actually on rung k+1? (supporting evidence, not the metric)
    next_rung = model.geom(f"rung{k + 1}").id
    hands = {model.geom("handL").id, model.geom("handR").id}
    gripped = set()
    for c in data.contact:
        pair = {int(c.geom1), int(c.geom2)}
        if next_rung in pair:
            gripped |= pair & hands
    out = {
        "spacing": spacing,
        "start_rung_z": round(rungs[k], 3),
        "hold_drift_m": round(hold_drift, 4),
        "hold_drop_m": round(hold_drop, 4),
        "ascent_m": round(ascent, 4),
        "ascent_frac": round(ascent / spacing, 4),
        "hands_on_next_rung": len(gripped),
        "final_torso_z": round(torso_z(), 4),
    }
    if not adhesion_on:
        return out

    # ── the fall: release everything mid-air ────────────────────────────
    data.ctrl[ids["adhL"]] = data.ctrl[ids["adhR"]] = 0.0
    data.ctrl[ids["reachL"]] = data.ctrl[ids["reachR"]] = -0.15  # clear the rungs
    for _ in range(40):
        mujoco.mj_step(model, data)
    snap = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(), "warm": data.qacc_warmstart.copy(),
            "time": float(data.time)}
    for _ in range(250):
        mujoco.mj_step(model, data)
    ref = (data.qpos.copy(), data.qvel.copy())

    # Resume: the mid-fall snapshot must continue identically in a fresh MjData.
    d2 = mujoco.MjData(model)
    d2.qpos[:] = snap["qpos"]; d2.qvel[:] = snap["qvel"]
    d2.ctrl[:] = snap["ctrl"]; d2.qacc_warmstart[:] = snap["warm"]
    d2.time = snap["time"]
    mujoco.mj_forward(model, d2)
    for _ in range(250):
        mujoco.mj_step(model, d2)
    resume_dev = float(max(np.abs(ref[0] - d2.qpos).max(),
                           np.abs(ref[1] - d2.qvel).max()))

    # Let the crash finish and settle.
    speeds = []
    for _ in range(650):
        mujoco.mj_step(model, data)
        speeds.append(float(np.abs(data.qvel[dofs]).max()))
    fall_finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    out.update({
        "fall_finite": int(fall_finite),
        "fall_rest_speed": round(max(speeds[-100:]), 4),
        "fall_rest_torso_z": round(torso_z(), 4),
        "resume_max_dev": resume_dev,
    })

    # ── the stream continues: reset and grip again ──────────────────────
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    data.ctrl[ids["adhL"]] = data.ctrl[ids["adhR"]] = 1.0
    _phase(model, data, ids, {}, 300)
    za = torso_z()
    _phase(model, data, ids, {}, 300)
    out["regrip_after_reset_drift_m"] = round(abs(torso_z() - za), 4)
    return out


def _experiment(seed: int) -> dict:
    return _run_script(seed, SPACINGS[seed % len(SPACINGS)], adhesion_on=True)


def _control(seed: int) -> dict:
    """Zero adhesion, identical script: the hang must slip and nothing ascends."""
    return _run_script(seed, SPACINGS[seed % len(SPACINGS)], adhesion_on=False)


def _check(m: dict, c: dict) -> bool:
    return (m["hold_drift_m"] <= HOLD_DRIFT_MAX
            and m["ascent_frac"] >= ASCENT_FRAC_MIN
            and m["fall_finite"] == 1
            and m["fall_rest_speed"] <= FALL_REST_SPEED_MAX
            and m["fall_rest_torso_z"] <= FALL_REST_TORSO_Z_MAX
            and m["resume_max_dev"] <= RESUME_DEV_MAX
            and m["regrip_after_reset_drift_m"] <= HOLD_DRIFT_MAX
            and c["hold_drop_m"] >= CONTROL_DROP_MIN
            and c["final_torso_z"] <= FALL_REST_TORSO_Z_MAX
            and c["ascent_frac"] <= 0.10)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.3"], _experiment, _check, control_fn=_control, ledger=ledger)
