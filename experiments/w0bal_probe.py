"""w0bal_probe.py — the W0.BAL body bakeoff, run as a DESK PROBE (24th audit B4).

NOT a spec and NOT an adoption. Precedent: D8's four scratch probes
(2026-08-14). This runs the instrument pre-registered in
INTEGRATION_QUEUE.md "W0.BAL" and preserved verbatim in DECISIONS_NEEDED.md
D9, attaches the numbers to D9, and leaves the A/B/C choice on the owner's
desk. Arms B and C change the world contract (PG.3 inherited geometry,
BA.01/PS.02/PS.03 certificates), which is the owner's call per D8.

PRE-REGISTRATION (verbatim from W0.BAL, 2026-08-09):
  arms    A  the rover as built (this is also the NULL, measured at
             upright_cos -0.041 in LC.02's ledger entry)
          B  bounded restoring torque on the torso, gated on floor contact
             EXACTLY as the drive is (same `_grounded()` check, applied per
             substep, zero contribution once the feet leave the ground)
          C  wide base + lowered COM: the spherical foot becomes a plinth and
             the mass moves into it until the rig is statically stable under
             the 600 N drive
  metric  upright_frac  (fraction of decisions with upright_cos >= 0.7)
          hand_reach_z_max (highest world z any hand geom attains)
  policy  identical uniform-random actions: same per-seed sequence for all
          three arms (RandomState(12345 + seed))
  budget  3 seeds x 500 decisions, same mutated worlds (seed 0 nursery,
          seeds 1-2 one ACCEL mutation, exactly as W0 builds them)
  kill    no arm reaches a hand above the first rung (z = rung_spacing,
          per-seed, worlds mutate the spacing) -> the ladder branch moves to
          a different body, not a better rig

ARM B's free parameters, chosen BEFORE running and recorded here because the
pre-registration says "bounded" without pricing the bound. The worst-case
gravity toppling torque of the as-built body is m_torso * g * lever =
30 kg * 9.81 * 0.30 m ~= 88 N-m (torso COM 0.30 m above the foot centre).
  KP   = 120 N-m at 90 deg tilt (tau = KP * |z_body x z_world|, so righting
         authority exists from prone but only just: bound > gravity, not >>)
  KD   = 15 N-m-s (damping on the world-frame angular velocity, xy only)
  TMAX = 120 N-m hard clip; yaw component zeroed (a righting mechanism must
         not hand out free turning authority)

ARM C's arithmetic, so "statically stable under the 600 N drive" is a
computed property, not a hope: plinth box 0.35 x 0.35 x 0.05 half-extents,
bottom at body z -0.64 (identical rest height to the sphere it replaces, so
ROVER_REST_Z and `_place` stay valid); masses swap, torso 30->2, base 2->30.
COM sits ~0.085 m above the floor; tipping margin m*g*half_width =
32.8 * 9.81 * 0.35 ~= 113 N-m against drive * COM height = 600 * 0.085
~= 51 N-m. Contact classes, friction, arm geometry and TOUCH_GEOMS names are
untouched.

Artifact: experiments/artifacts/w0bal_bakeoff.json. Wall cost: minutes, CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.w0 import W0, random_action  # noqa: E402

SEEDS = (0, 1, 2)
DECISIONS = 500
UPRIGHT_BAR = 0.7
J0, ALPHA = 1.0, 0.01          # LC.02's timing-only convention; nothing here
                               # reads e/i/w as a quantity either

KP, KD, TMAX = 120.0, 15.0, 120.0

_FOOT_SPHERE = ('<geom name="rover_foot" type="sphere" pos="0 0 -0.550" '
                'size="0.09" mass="2"')
_FOOT_PLINTH = ('<geom name="rover_foot" type="box" pos="0 0 -0.590" '
                'size="0.35 0.35 0.05" mass="30"')
_TORSO_HEAVY = 'size="0.07" mass="30"'
_TORSO_LIGHT = 'size="0.07" mass="2"'


class _RightingStepper:
    """Proxy over the mujoco module for arm B.

    `W0.decide` calls `self.mujoco.mj_step` once per substep, after it has
    evaluated the drive gate; this proxy applies the bounded righting torque
    through `xfrc_applied[:, 3:6]` under the SAME `_grounded()` gate and then
    steps. Torque channel is disjoint from the drive's force channel
    (`xfrc_applied[:, :2]`), so the two cheats cannot overwrite each other.
    """

    def __init__(self, mj):
        self._mj = mj
        self.w = None

    def __getattr__(self, name):
        return getattr(self._mj, name)

    def mj_step(self, model, data):
        w = self.w
        if w is not None:
            bid = w.rover_bid
            if w._grounded():
                mat = np.array(data.xmat[bid], dtype=float).reshape(3, 3)
                zb = mat[:, 2]
                err = np.cross(zb, np.array([0.0, 0.0, 1.0]))
                da = w.ix["root_dofadr"]
                w_body = np.array(data.qvel[da + 3:da + 6], dtype=float)
                w_world = mat @ w_body
                tau = KP * err - KD * w_world
                tau[2] = 0.0
                n = float(np.linalg.norm(tau))
                if n > TMAX:
                    tau *= TMAX / n
                data.xfrc_applied[bid, 3:6] = tau
            else:
                data.xfrc_applied[bid, 3:6] = 0.0
        self._mj.mj_step(model, data)


def _build(arm: str, seed: int) -> W0:
    import playground as pg

    if arm == "C":
        orig = pg._rover_fragments

        def patched(spawn):
            body, act = orig(spawn)
            if _TORSO_HEAVY not in body or _FOOT_SPHERE not in body:
                raise RuntimeError("arm C patch found nothing to replace — "
                                   "the rover XML moved; refusing to run arm A "
                                   "under a C label")
            body = body.replace(_TORSO_HEAVY, _TORSO_LIGHT)
            body = body.replace(_FOOT_SPHERE, _FOOT_PLINTH)
            return body, act

        pg._rover_fragments = patched
        try:
            w = W0(seed=seed, j0=J0, alpha=ALPHA)
        finally:
            pg._rover_fragments = orig
        # prove the patch took: the foot geom must be a box (mjGEOM_BOX = 6)
        import mujoco
        gid = w.ix["geom"]["rover_foot"]
        if int(w.model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise RuntimeError("arm C world does not carry the plinth foot")
        return w

    w = W0(seed=seed, j0=J0, alpha=ALPHA)
    if arm == "B":
        proxy = _RightingStepper(w.mujoco)
        proxy.w = w
        w.mujoco = proxy
    return w


def _run(arm: str, seed: int) -> dict:
    w = _build(arm, seed)
    rng = np.random.RandomState(12345 + seed)   # identical sequence per seed,
                                                # all arms — F4's discipline
    gL = w.ix["geom"]["handL"]
    gR = w.ix["geom"]["handR"]
    upright_hits = 0
    hand_z_max = -np.inf
    cos_trace_tail = []
    for k in range(DECISIONS):
        w.decide(random_action(rng))
        cos = float(w.data.xmat[w.rover_bid][8])
        if cos >= UPRIGHT_BAR:
            upright_hits += 1
        hz = max(float(w.data.geom_xpos[gL][2]), float(w.data.geom_xpos[gR][2]))
        hand_z_max = max(hand_z_max, hz)
        if k >= DECISIONS - 50:
            cos_trace_tail.append(cos)
    r = w.report()
    first_rung_z = float(w.params.ladder_rung_spacing)
    return {
        "arm": arm, "seed": seed,
        "upright_frac": upright_hits / DECISIONS,
        "hand_reach_z_max": float(hand_z_max),
        "first_rung_z": first_rung_z,
        "hand_above_first_rung": bool(hand_z_max > first_rung_z),
        "upright_cos_final": float(r["upright_cos"]),
        "upright_cos_tail_mean": float(np.mean(cos_trace_tail)),
        "torso_z_final": float(r["torso_z"]),
        "drive_gate_frac": float(r["drive_gate_frac"]),
    }


def main() -> None:
    rows = []
    for arm in ("A", "B", "C"):
        for seed in SEEDS:
            row = _run(arm, seed)
            rows.append(row)
            print(f"arm {row['arm']} seed {row['seed']}: "
                  f"upright_frac {row['upright_frac']:.3f}  "
                  f"hand_z_max {row['hand_reach_z_max']:.3f}  "
                  f"rung1 {row['first_rung_z']:.3f}  "
                  f"above={row['hand_above_first_rung']}", flush=True)
    kill = not any(r["hand_above_first_rung"] for r in rows)
    out = {
        "preregistration": "INTEGRATION_QUEUE.md W0.BAL / DECISIONS_NEEDED.md D9",
        "authority": "24th audit B4 — run, attach to D9, ADOPT NOTHING",
        "arm_B_params": {"KP": KP, "KD": KD, "TMAX": TMAX},
        "seeds": list(SEEDS), "decisions": DECISIONS,
        "upright_bar": UPRIGHT_BAR,
        "rows": rows,
        "kill_criterion_fired": kill,
    }
    path = REPO / "experiments" / "artifacts" / "w0bal_bakeoff.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"kill_criterion_fired={kill}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
