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

Four are no longer proposals. `J0` and `alpha` were measured by PS.01
(2026-08-10T05:29, held out on five fresh fall runs). `KAPPA`,
`RESPAWN_FLOORFOOD_S` and `RESPAWN_APPLE_S` were re-derived by PS.01 unit (a)
the same day, on held-out seeds 3-5, after the shipped `KAPPA` was measured to
make the world unsurvivable for *any* policy. Their derivations are inline
below and the criterion was committed before it was solved.

Ownership, deliberately: this layer does NOT step the physics. The caller owns
`mj_step`; it calls `substep()` after each one and `decide()` at the end of a
decision. Two reasons, both scars. A layer that owned the loop would be a second
copy of the stepping code the pipeline already has ("two kernels
re-implementing one operation is the defect"), and a layer that could not be
driven substep-by-substep could not be measured against a world it did not also
control.

Units are SI throughout and `dt` is always seconds of simulated time, so nothing
here depends on `frame_skip`.

**`J_t` IS NO LONGER AN IMPULSE (2026-08-10).** It was
`sum_substeps ||cfrc_ext[torso]|| * dt` — a genuine N·s impulse — until the
`PS.01/J` and `PS.01/J2` bakeoffs measured that formulation **at chance** for
telling a fall from ordinary ground contact (AUC 0.520 against a shuffled null
of 0.4966 ± 0.0122; two of its proposed repairs scored *below* chance).
Thirteen channels competed over two rounds and `impact_speed` won at 0.973 AUC,
+10.32 sigma, 2.66 sigma clear of the runner-up:

    J_t = the root's linear SPEED one substep before contact ONSET, maximised
          over the onsets that occurred during this decision (0 if none).

Two consequences worth knowing before re-opening it. (1) The channel is
**kinematic**, so this layer no longer reads `cfrc_ext` at all — which retires
the `mj_rnePostConstraint` staleness caveat that used to make `j` wrong in any
caller that batched the call (see `w0.py`'s `step()` and `LESSONS.md`). It also
means a `frame_skip` change still cannot move the threshold: a speed is a state
variable, not an accumulation. (2) The dimensional change is absorbed by
`alpha`, which PS.01 calibrates against the same 1.8 m platform fall §2.2 always
named. Full evidence: `docs/DECISIONS_RESOLVED.md`,
`experiments/bakeoffs/ps01_impulse*.py`, `PURPOSE_AND_SCAFFOLDING.md` §2.2.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

# ── §2.2 proposals: energy ──────────────────────────────────────────────
BASAL_B = 1.0 / 600.0        # s^-1   a resting body empties in 10 minutes
NU_APPLE = 0.50              # energy restored by the platform apple
NU_FLOORFOOD = 0.08          # energy restored by one floor food

# ── MEASURED, not proposed: the energy economy (PS.01 unit (a), 2026-08-10) ──
# `experiments/calibrations/ps01_energy.py` — criterion committed unrun in
# 92aae6f, solved on HELD-OUT seeds 3-5, full derivation in
# PURPOSE_AND_SCAFFOLDING.md 2.3. In one line: this body's mechanical power was
# never measured before the constants were written down.
#
#   P_bar(1) = 1434.8 +- 22.2 W   a full-strength random policy, seeds 3-5
#
# KAPPA was 1.67e-5, defined by 2.2's sentence "vigorous activity (~200 W)
# roughly triples b". 200 W is a HUMAN premise; Humanoid-v5's actuators deliver
# 7.17x it under the very policy the drain is priced against, so the shipped
# kappa made constant activity cost 15.4x basal and NO policy of any competence
# could have been fed (a perfect forager harvested 0.23x of what it cost). The
# sentence is kept and the number re-derived from the measured body:
#   KAPPA = (3 - 1) * BASAL_B / P_bar(1),  i.e. drain(1) = 3 x basal exactly.
KAPPA = 2.3231e-06           # J^-1   §2.2's "roughly triples b", re-derived
# Supply rates follow from the criterion, which moves the RESPAWN PERIOD and
# never the per-item value — nu_apple/nu_floor is the climb-vs-forage incentive
# ratio and 2.3 calls it load-bearing.
#   C2: floor supply S_f = min(PAL*b, b + KAPPA*P_bar(0.25)) = 2.3914e-3 /s,
#       the SMALLER of a biological anchor (PAL 1.7) and the journal's
#       duty-cycle anchor, i.e. the harsher world. It funds a duty cycle of
#       D* = 0.217: an agent acting a fifth of the time subsists on the floor.
#   C3: S_f is 2.09x short of constant activity, so the ladder still decides
#       between subsisting and acting -- 2.3's intent, priced against the drain
#       an acting body pays instead of against basal.
#   C1: every food perfectly harvested must feed a fully active body that misses
#       one respawn in five: S_max >= drain(1)/0.8 = 6.25e-3 /s.
RESPAWN_APPLE_S = 129.6      # was 120.0 — C1 closes the gap C2/C3 leave
RESPAWN_FLOORFOOD_S = 66.9   # was  90.0 — C2

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
# The impact channel used to name a SENSOR BODY (`IMPACT_BODIES = ("torso",)`,
# §2.2's "torso + head"). `PS.01/J2` retired it: a 1.8 m platform fall lands on
# the FEET and `cfrc_ext[torso]` reads identically zero for the 0.30 s after
# contact onset, so the torso sensor and a whole-body landing are on different
# bodies. The winning channel reads the free root's velocity instead, which no
# body choice can get wrong.
HEAD_GEOM = "head"                    # for the drowning test, §2.2


@dataclass(frozen=True)
class BodyRef:
    """WHICH body this layer is integrating, resolved against the live model.

    The layer was written for Humanoid-v5 and every id it needed was looked up
    by a humanoid name. `LEARNING_CORE.md` §5.0 then put the LC bakeoff on the
    climber-rover, and the choice was: a second integrator for the second body,
    or one integrator that is told which body it is holding. `LESSONS.md` —
    *"two kernels re-implementing one operation is the defect"* — settles it.
    The humanoid path is byte-identical to what it was; the rover supplies its
    own ids through `playground.rover_index`.

    `head_geoms` is the drowning probe, not an anatomical claim: on the rover
    the torso capsule's own geom is the highest point of the body, and on the
    humanoid `head` is a geom on the torso body for the same reason.

    There is deliberately no `impact` field. It named the bodies whose
    `cfrc_ext` rows were the impact channel, and `PS.01/J2` measured that
    channel at chance; the winner is the free root's arrival speed, which
    `dofadr` already locates on either body.
    """
    bodies: frozenset       # every body id that is Jack — the eating/water test
    head_geoms: tuple       # geom ids submerged == drowning
    dofadr: int             # first dof of the free root
    ndof: int               # free root (6) + actuated dofs


def humanoid_body_ref(model) -> BodyRef:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from playground import HUMANOID_NV, humanoid_body_ids, humanoid_index
    return BodyRef(bodies=frozenset(humanoid_body_ids(model)),
                   head_geoms=(int(model.geom(HEAD_GEOM).id),),
                   dofadr=humanoid_index(model)["dofadr"], ndof=HUMANOID_NV)


def rover_body_ref(model) -> BodyRef:
    """The climber-rover: 3 bodies, 10 dofs (free root + 4 arm slides)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from playground import rover_index
    ix = rover_index(model)
    return BodyRef(bodies=frozenset(ix["body"].values()),
                   head_geoms=(ix["geom"]["rover_torso"],),
                   dofadr=ix["root_dofadr"], ndof=10)


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

    **No `mj_rnePostConstraint` is required.** It used to be, and the
    requirement was load-bearing and expensive: `cfrc_ext` is filled by that
    call and not by `mj_step` (the PG.8 lesson), so a caller that batched it to
    once per decision — as `w0.py` deliberately does, because the per-substep
    call costs ~15-25% throughput and dropped 4 of LC.02's 5 arms below the 5.0
    floor — silently fed `substep` the PREVIOUS decision's contact state. That
    hazard is **gone**: `PS.01/J2` replaced the force channel with the root's
    arrival speed, and `qvel` is current after every `mj_step`. One measured
    decision retired a documented instrumentation trap, which is worth more
    than the AUC it was chosen for.

    `j0` is the arrival speed (m/s) below which contact is NORMAL and costs
    nothing — the 95th percentile of ordinary ground contact — and `alpha`
    converts the excess into integrity. Both are measured by PS.01 and neither
    has a default: a wrong number here is silent, and silence is the one
    failure mode this repo has paid for most.
    """

    def __init__(self, model, *, j0: float, alpha: float,
                 pool: Optional[tuple] = None, state: Optional[DriveState] = None,
                 body: Optional[BodyRef] = None):
        if j0 is None or alpha is None:
            raise ValueError("j0 and alpha must be measured (PS.01), not defaulted")
        self.model = model
        self.j0 = float(j0)
        self.alpha = float(alpha)
        # (x, y, half, surface_z) of the pool region, or None for a dry world.
        self.pool = pool
        self.state = state or DriveState()

        self.body = body or humanoid_body_ref(model)
        self._body_ids = set(self.body.bodies)
        self._geom_of_body = {g: int(model.geom_bodyid[g]) for g in range(model.ngeom)}
        self._jack_geoms = {g for g, b in self._geom_of_body.items()
                            if b in self._body_ids}
        self._head_geoms = list(self.body.head_geoms)
        self._jack_mask = np.zeros(model.ngeom, dtype=bool)
        self._jack_mask[list(self._jack_geoms)] = True
        self._jack_gids = np.array(sorted(self._jack_geoms), dtype=int)
        self._food = {}
        for name, nu in FOOD_GEOMS.items():
            try:
                self._food[name] = (int(model.geom(name).id), nu)
            except (KeyError, ValueError):
                pass                       # a mutated world may not carry it
        self._dofadr, self._ndof = self.body.dofadr, self.body.ndof

        self.t = 0.0
        self._respawn_at = {name: 0.0 for name in self._food}
        self._submerged_since: Optional[float] = None
        # Contact-onset state is LIFE-level, not decision-level: a landing that
        # straddles a decision boundary is one landing, and resetting either of
        # these in `begin_decision` would count it twice.
        self._touching_world = False
        self._prev_speed: Optional[float] = None
        self._reset_decision()
        # Diagnostics the caller may read after any decision.
        self.last_j = 0.0
        self.last_power_w = 0.0
        self.last_dt = 0.0
        self.last_rest_dt = 0.0
        self.n_onsets = 0
        self.ate_total = {name: 0 for name in self._food}

    # ── the decision cycle ──────────────────────────────────────────────
    def begin_decision(self) -> None:
        self._reset_decision()

    def _reset_decision(self) -> None:
        self._j_max = 0.0
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

        qvel = np.asarray(data.qvel[d:d + n])
        if float(np.linalg.norm(qvel)) < Q_REST:
            self._rest_dt += dt

        # ONE vectorised pass over `data.contact.geom` per substep, read by both
        # the impact channel and eating. Identical semantics to a Python scan —
        # LC.02 measured the old per-food-item form at 5.35 ms of a 20.4 ms
        # decision, and a throughput floor that a lazy inner loop can fail is a
        # floor on the loop, not the core.
        partners = _contact_partners(data, self._jack_mask)

        # impact, §2.2 as decided by PS.01/J2: the root's linear speed one
        # substep BEFORE contact onset. `partners` may contain Jack's own geoms
        # (he folds onto himself constantly under any policy, and that is not a
        # landing), so WORLD contact is a partner that is not his — the same
        # label-free predicate the bakeoff scored, which never sees which regime
        # it is in. Only the False->True edge counts: while he lies on the floor
        # the flag stays True and no speed is read, which is why lying there
        # cannot manufacture damage however long it lies.
        speed = float(np.linalg.norm(qvel[:3]))
        touching = bool(partners - self._jack_geoms)
        if touching and not self._touching_world:
            arrival = speed if self._prev_speed is None else self._prev_speed
            self._j_max = max(self._j_max, arrival)
            self.n_onsets += 1
        self._touching_world = touching
        self._prev_speed = speed

        # eating: a physical contact between one of Jack's geoms and a food geom.
        for name, (gid, nu) in self._food.items():
            if self.t + self._dt_acc < self._respawn_at[name]:
                continue
            if gid in partners:
                self._ate += nu
                self._respawn_at[name] = (self.t + self._dt_acc
                                          + RESPAWN_S[name])
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
        j = self._j_max          # m/s, the worst arrival speed of this decision
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
        # Published because PS.01 has to report the drain it measured rather
        # than the drain §2.2 guessed, and a caller that must reach into
        # `_rest_dt` to do that is a caller reading a private accumulator after
        # `_reset_decision` has already been scheduled to clear it.
        self.last_dt = dt
        self.last_rest_dt = self._rest_dt
        self._reset_decision()
        return self.state

    # ── W0-2: a new body, without a new world ───────────────────────────
    def new_body(self, state: Optional[DriveState] = None) -> None:
        """Death: reset everything that belongs to the BODY, nothing else.

        `LEARNING_CORE.md` §5.0 W0-2 asks for death and a respawn that is *not a
        free teleport to a good state*. The body/world split is what makes that
        precise, and it is the whole reason this is a method rather than a fresh
        `DriveLayer`:

          BODY, reset here      e, i, w; the contact-onset edge detector; the
                                submersion clock; the part-decision accumulators
          WORLD, deliberately   `self.t` (the world clock) and `_respawn_at`
          NOT reset             (the food regrowth timers), and `ate_total`
                                (a cumulative diagnostic, not a body state)

        Reset the food timers too and every death would hand him a freshly
        stocked larder — an experimenter-supplied curriculum arriving through
        the back door, which is exactly the objection the random respawn exists
        to answer. `state` overrides the (1, 1, 0) setpoint; W0's tests use it
        to build a deliberately drifting world as the positive control for a
        trend detector, and nothing in the ladder may use it to make an arm
        survive.
        """
        self.state = state or DriveState()
        self._touching_world = False
        self._prev_speed = None
        self._submerged_since = None
        self._reset_decision()

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
        p = np.asarray(data.geom_xpos)[self._jack_gids]
        return bool(np.any((np.abs(p[:, 0] - x) <= half)
                           & (np.abs(p[:, 1] - y) <= half)
                           & (p[:, 2] < surf)))


def _contact_partners(data, mask: np.ndarray) -> set:
    """Geom ids currently touching any geom selected by `mask`.

    One numpy pass over `data.contact.geom` (an (ncon, 2) array since MuJoCo
    3.x) instead of one Python loop per query. `_in_contact` below is kept
    because it is the readable single-query form and is still the right tool
    outside a substep loop.
    """
    n = int(data.ncon)
    if n == 0:
        return set()
    g = np.asarray(data.contact.geom[:n])
    a, b = g[:, 0], g[:, 1]
    return set(b[mask[a]].tolist()) | set(a[mask[b]].tolist())


def _in_contact(data, geoms: set, gid: int) -> bool:
    for k in range(int(data.ncon)):
        c = data.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 == gid and g2 in geoms) or (g2 == gid and g1 in geoms):
            return True
    return False


