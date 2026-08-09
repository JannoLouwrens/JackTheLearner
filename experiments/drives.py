"""The drive layer: energy, integrity, wetness — the substrate PS.01 calibrates.

`docs/research/PURPOSE_AND_SCAFFOLDING.md` §2.2-2.5 specifies three scalars in
[0, 1] that persist across an unbroken life, an integrator that couples them to
MuJoCo's own quantities, and a *soft* incapacity (never a termination) so the
life has no episode boundaries — an episode boundary is a free teleport back to
a good state, i.e. an experimenter-supplied curriculum.

    h = (e, i, w)   e energy     1 = fed       0 = starving
                    i integrity  1 = unhurt    0 = wrecked
                    w wetness    0 = dry       1 = soaked   (setpoint 0)

This module is the substrate, not a claim. **Every constant below is a PROPOSAL
until PS.01 replaces it with a measurement** (the spec's own `notes`), and two of
them — `J0` and `alpha` — have no default at all and must be passed in, because
a default there would be a number nobody measured wearing the costume of one.
`LESSONS.md`: "A default of zero is not 'unknown'."

Ownership, deliberately: this layer does NOT step the physics. The caller owns
`mj_step`; it calls `substep()` after each one and `decide()` at the end of a
decision. Two reasons, both scars. A layer that owned the loop would be a second
copy of the stepping code the pipeline already has ("two kernels
re-implementing one operation is the defect"), and a layer that could not be
driven substep-by-substep could not be measured against a world it did not also
control.

Units are SI throughout and `dt` is always seconds of simulated time, so nothing
here depends on `frame_skip`. That is not cosmetic: `J` is a genuine impulse
(N·s, force integrated over the decision) rather than a per-decision force sum,
precisely so a `frame_skip` change cannot move the damage threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

# ── §2.2 proposals: energy ──────────────────────────────────────────────
BASAL_B = 1.0 / 600.0        # s^-1   a resting body empties in 10 minutes
KAPPA = 1.67e-5              # J^-1   ~200 W of mechanical work roughly triples b
NU_APPLE = 0.50              # energy restored by the platform apple
NU_FLOORFOOD = 0.08          # energy restored by one floor food
RESPAWN_APPLE_S = 120.0
RESPAWN_FLOORFOOD_S = 90.0

# ── §2.2 proposals: integrity ───────────────────────────────────────────
RHO_HEAL = 1.0 / 900.0       # s^-1   full heal in 15 minutes of rest
DROWN_RATE = 0.05            # s^-1   while the head has been under > DROWN_DELAY
DROWN_DELAY_S = 8.0
# ‖qvel‖ below which the body counts as resting and heals. Not given in §2.2.
# Taken from PG.8's measured settle: 10 s at zero control leaves |qvel|max at
# 0.011 rad/s against its 0.5 gate, a 45x margin — so 0.5 separates "settled"
# from "moving" on this body by measurement rather than by taste.
Q_REST = 0.5

# ── §2.2 proposals: wetness ─────────────────────────────────────────────
WET_TAU_DRY_S = 120.0        # §2.2 gives the decay constant
WET_TAU_WET_S = 120.0        # §2.2 does not give the rise constant; symmetric,
                             # and PS.01 gates nothing on w — it is logged only.

# ── §2.5: the drive function, K&G form at the (1, 1, 0) setpoint ────────
LAMBDA = (1.0, 1.0, 0.3)
N_EXP, M_EXP = 4.0, 2.0      # §2.5 defaults; PS.01 reports them, does not fit them

# ── §2.2: weakness, the soft incapacity that replaces termination ───────
GEAR_FLOOR = 0.4             # gear_scale = 0.4 + 0.6 * min(e, i); starving is
                             # never an absorbing trap he cannot climb out of
SIGMA0 = 0.0                 # ctrl_noise = SIGMA0 * (1 - i). §2.2 states the
                             # form and gives no magnitude; PS.01 runs an open-
                             # loop random policy where injected ctrl noise
                             # would be indistinguishable from the policy, so it
                             # is 0 here and must be set before any ARM runs.

DRIVE_DIM = 6                # §2.4: [e, i, w, d(h), edot, idot]

FOOD_GEOMS = {"apple": NU_APPLE, "obj0": NU_FLOORFOOD, "obj1": NU_FLOORFOOD}
RESPAWN_S = {"apple": RESPAWN_APPLE_S,
             "obj0": RESPAWN_FLOORFOOD_S, "obj1": RESPAWN_FLOORFOOD_S}
# §2.2 writes "torso + head". In Humanoid-v5 the head is a GEOM on the torso
# body, not a body of its own, so `cfrc_ext` has no head row to sum and "torso"
# already carries the head's external force. Naming both would have summed the
# torso twice. Checked against the live model rather than transcribed.
IMPACT_BODIES = ("torso",)
HEAD_GEOM = "head"                    # for the drowning test, §2.2


def drive(e: float, i: float, w: float) -> float:
    """d(h) — distance from the (1, 1, 0) setpoint, §2.5."""
    terms = (LAMBDA[0] * abs(1.0 - e) ** N_EXP
             + LAMBDA[1] * abs(1.0 - i) ** N_EXP
             + LAMBDA[2] * abs(0.0 - w) ** N_EXP)
    return float(terms ** (1.0 / M_EXP))


def gear_scale(e: float, i: float) -> float:
    """Weakness, §2.2. Nothing terminates; incapacity is soft and reversible."""
    return GEAR_FLOOR + (1.0 - GEAR_FLOOR) * min(e, i)


@dataclass
class DriveState:
    e: float = 1.0
    i: float = 1.0
    w: float = 0.0

    def d(self) -> float:
        return drive(self.e, self.i, self.w)


class DriveLayer:
    """The integrator. Caller owns `mj_step`; this owns `h`.

    Usage, one decision:

        layer.begin_decision()
        for _ in range(frame_skip):
            data.ctrl[:] = ctrl * layer.gear_scale()
            mujoco.mj_step(model, data)
            layer.substep(model, data, model.opt.timestep)
        h = layer.decide()

    `j0` is the impulse below which contact is NORMAL and costs nothing, and
    `alpha` converts the excess into integrity. Both are measured by PS.01 and
    neither has a default: a wrong number here is silent, and silence is the one
    failure mode this repo has paid for most.
    """

    def __init__(self, model, *, j0: float, alpha: float,
                 pool: Optional[tuple] = None, state: Optional[DriveState] = None):
        if j0 is None or alpha is None:
            raise ValueError("j0 and alpha must be measured (PS.01), not defaulted")
        self.model = model
        self.j0 = float(j0)
        self.alpha = float(alpha)
        # (x, y, half, surface_z) of the pool region, or None for a dry world.
        self.pool = pool
        self.state = state or DriveState()

        self._body_ids = _humanoid_bodies(model)
        self._impact_ids = [model.body(n).id for n in IMPACT_BODIES]
        self._geom_of_body = {g: int(model.geom_bodyid[g]) for g in range(model.ngeom)}
        self._jack_geoms = {g for g, b in self._geom_of_body.items()
                            if b in self._body_ids}
        self._head_geoms = [int(model.geom(HEAD_GEOM).id)]
        self._food = {}
        for name, nu in FOOD_GEOMS.items():
            try:
                self._food[name] = (int(model.geom(name).id), nu)
            except (KeyError, ValueError):
                pass                       # a mutated world may not carry it
        ix = _humanoid_index(model)
        self._dofadr, self._ndof = ix

        self.t = 0.0
        self._respawn_at = {name: 0.0 for name in self._food}
        self._submerged_since: Optional[float] = None
        self._reset_decision()
        # Diagnostics the caller may read after any decision.
        self.last_j = 0.0
        self.last_power_w = 0.0
        self.ate_total = {name: 0 for name in self._food}

    # ── the decision cycle ──────────────────────────────────────────────
    def begin_decision(self) -> None:
        self._reset_decision()

    def _reset_decision(self) -> None:
        self._j_acc = 0.0
        self._power_dt = 0.0
        self._dt_acc = 0.0
        self._ate = 0.0
        self._rest_dt = 0.0
        self._drown_dt = 0.0

    def gear_scale(self) -> float:
        return gear_scale(self.state.e, self.state.i)

    def ctrl_noise_scale(self) -> float:
        return SIGMA0 * (1.0 - self.state.i)

    def substep(self, model, data, dt: float) -> None:
        """Accumulate everything that is a function of the physics, per mj_step.

        Energy is integrated HERE rather than at the decision boundary because
        mechanical power is a substep quantity; sampling it once per decision
        would alias a 200 Hz signal at 40 Hz. Integrity is accumulated here and
        applied in `decide()` because §2.2 defines J over the decision.
        """
        d, n = self._dofadr, self._ndof
        tau = np.asarray(data.qfrc_actuator[d + 6:d + n])
        omega = np.asarray(data.qvel[d + 6:d + n])
        power = float(np.abs(tau * omega).sum())          # W, mechanical
        self._power_dt += power * dt
        self._dt_acc += dt

        f = np.asarray(data.cfrc_ext[self._impact_ids])   # (2, 6) force+torque
        self._j_acc += float(np.linalg.norm(f)) * dt      # N·s, a real impulse

        qvel = np.asarray(data.qvel[d:d + n])
        if float(np.linalg.norm(qvel)) < Q_REST:
            self._rest_dt += dt

        # eating: a physical contact between one of Jack's geoms and a food geom
        for name, (gid, nu) in self._food.items():
            if self.t + self._dt_acc < self._respawn_at[name]:
                continue
            if _in_contact(data, self._jack_geoms, gid):
                self._ate += nu
                self._respawn_at[name] = self.t + self._dt_acc + RESPAWN_S[name]
                self.ate_total[name] += 1

        # drowning: the head geom below the pool surface for > DROWN_DELAY_S
        if self.pool is not None and self._head_under(data):
            if self._submerged_since is None:
                self._submerged_since = self.t + self._dt_acc
            elif (self.t + self._dt_acc) - self._submerged_since > DROWN_DELAY_S:
                self._drown_dt += dt
        else:
            self._submerged_since = None

        self._wet_dt = dt
        self._wet_in = self._any_geom_in_water(data)
        self._integrate_wet(dt)

    def decide(self) -> DriveState:
        """Close the decision: apply §2.2's three update rules and clip."""
        dt = self._dt_acc
        s = self.state
        j = self._j_acc
        power_mean = self._power_dt / dt if dt > 0 else 0.0

        e = s.e - (BASAL_B * dt + KAPPA * self._power_dt) + self._ate
        i = (s.i
             - self.alpha * max(0.0, j - self.j0)
             - DROWN_RATE * self._drown_dt
             + RHO_HEAL * self._rest_dt)
        self.state = DriveState(e=float(np.clip(e, 0.0, 1.0)),
                                i=float(np.clip(i, 0.0, 1.0)),
                                w=float(np.clip(s.w, 0.0, 1.0)))
        self.t += dt
        self.last_j = j
        self.last_power_w = power_mean
        self._reset_decision()
        return self.state

    # ── observation, §2.4 ───────────────────────────────────────────────
    def obs(self, prev: Optional[DriveState] = None, dt: float = 1.0) -> np.ndarray:
        """[e, i, w, d(h), edot, idot]. Concatenated OUTSIDE `humanoid_obs`, so
        the 348 that PG.8 asserts against gymnasium stays a 348 (§2.4)."""
        s = self.state
        p = prev or s
        v = np.array([s.e, s.i, s.w, s.d(),
                      (s.e - p.e) / dt, (s.i - p.i) / dt], dtype=np.float64)
        if v.shape[0] != DRIVE_DIM:
            raise RuntimeError(f"drive obs is {v.shape[0]}, not {DRIVE_DIM}")
        return v

    # ── internals ───────────────────────────────────────────────────────
    def _integrate_wet(self, dt: float) -> None:
        w = self.state.w
        if self._wet_in:
            w += (1.0 - w) * dt / WET_TAU_WET_S
        else:
            w -= w * dt / WET_TAU_DRY_S
        self.state = replace(self.state, w=float(np.clip(w, 0.0, 1.0)))

    def _head_under(self, data) -> bool:
        if self.pool is None:
            return False
        x, y, half, surf = self.pool
        for g in self._head_geoms:
            p = data.geom_xpos[g]
            if abs(p[0] - x) <= half and abs(p[1] - y) <= half and p[2] < surf:
                return True
        return False

    def _any_geom_in_water(self, data) -> bool:
        if self.pool is None:
            return False
        x, y, half, surf = self.pool
        for g in self._jack_geoms:
            p = data.geom_xpos[g]
            if abs(p[0] - x) <= half and abs(p[1] - y) <= half and p[2] < surf:
                return True
        return False


def _in_contact(data, geoms: set, gid: int) -> bool:
    for k in range(int(data.ncon)):
        c = data.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 == gid and g2 in geoms) or (g2 == gid and g1 in geoms):
            return True
    return False


def _humanoid_bodies(model) -> set:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from playground import humanoid_body_ids
    return set(humanoid_body_ids(model))


def _humanoid_index(model) -> tuple:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from playground import HUMANOID_NV, humanoid_index
    return humanoid_index(model)["dofadr"], HUMANOID_NV
