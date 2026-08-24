"""The seven-need integrator: NEEDS_AND_DEATH §2.3's suite. The substrate NE.01 calibrates.

    h = (e, w, p, T, f, c, i)

      e  energy        1 = fed          0 = starving        setpoint 1     LETHAL
      w  water         1 = hydrated     0 = dehydrated      setpoint 1     LETHAL
      p  sleep         0 = rested       1 = must sleep      setpoint 0     lethal INDIRECTLY
      T  core temp     degrees C                            setpoint 37    LETHAL BOTH SIDES
      f  fatigue       0 = fresh        1 = spent           setpoint 0     not lethal
      c  social        1 = contented    0 = isolated        setpoint 1     not lethal
      i  integrity     1 = unhurt       0 = wrecked         setpoint 1     LETHAL

This module is the substrate, not a claim. **Every constant below is a PROPOSAL
until NE.01 replaces it with a measurement** (the spec's own `kills` field says
exactly that). Constants inherited from a measurement say where the measurement
lives; constants that are design choices say what chose them. `drives.py` (the
three-need PS §2.2 suite) is NOT touched — its certificates (PS.01, the LC
family, W0) stand on its exact bytes, and LESSONS.md prices what an additive
edit to a declared dependency costs. This file imports drives' pure, shared
pieces (BodyRef, the contact scan, Q_REST) and re-implements the need dynamics
to §2.3's letter.

Ownership is inherited verbatim from `DriveLayer`: the CALLER owns `mj_step`;
this layer owns `h`. `begin_decision()` / `substep()` per mj_step /
`decide()` at the decision boundary. Units are SI, `dt` is simulated seconds,
nothing depends on `frame_skip`.

DECLARED DIVERGENCES from `drives.py`, each per §2.3's letter:
  * EATING IS MOUTH-GATED. §2.3: "mouth geom contacts food". DriveLayer fed on
    any-body contact; here only `BodyRef.head_geoms` (the humanoid's head geom,
    the rover's torso capsule) can eat or drink.
  * DROWNING IS A DEATH, NOT A DRAIN. §2.5: head submerged > 20 s kills,
    routed through i (i := 0, cause recorded "drowning"). PS's 8 s delay +
    0.05/s drain is retired here.
  * WETNESS IS NOT A NEED. §2.2 demoted it: `skin_wetness` multiplies heat
    loss (kappa_wet) and drives evaporative cooling. Same 120 s time constants
    as PS measured nothing against (they were logged-only there too).

ONE KNOWN SPEC-VS-ARITHMETIC DISCREPANCY, flagged rather than silently fixed:
NE.01's registry prose says the do-nothing statue "dies of starvation". Under
§2.3's own constants the statue's water empties at 450 s and kills at 570 s,
long before energy's 1,800 + 300 s — the statue dies of DEHYDRATION. The
control's purpose (doing nothing must be lethal) is unaffected; the cause word
must be corrected at NE.01 implementation time, before the registered run.

Sleep-time compression (§9): `sleep_coarse_step()` advances the needs without
physics at a coarse internal dt. NE.01 must verify the coarse thermal
trajectory matches the fine one within 0.2 C over a night before any sleep
arm may buy speed with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .drives import (BodyRef, Q_REST, _contact_partners,  # shared kernel, not copied
                     humanoid_body_ref, rover_body_ref)

__all__ = [
    "NeedState", "NeedLayer", "need_drive", "delta_T", "t_norm",
    "NEED_DIM", "LAMBDA", "N_EXP", "M_EXP", "RHO_ALIVE",
    "humanoid_body_ref", "rover_body_ref",
]

# ── the clock (§2.3): one decision = 0.2 s; a sim-day = 1,200 s ─────────
DAY_S = 800.0
NIGHT_S = 400.0
SIM_DAY_S = DAY_S + NIGHT_S

# ── depletion rates (§2.2 table, given in the doc) ──────────────────────
B_E = 1.0 / 1800.0           # s^-1  energy empties in 30 min at basal
B_W = 1.0 / 450.0            # s^-1  water ~3 drinks per sim-day
B_C = 1.0 / 3600.0           # s^-1  social: 3 sim-days of solitude to bottom

# ── sleep: Borbely process S only, §2.3 (constants given) ───────────────
TAU_WAKE = 700.0             # s     0.05 -> 0.70 over one 800 s day
TAU_SLEEP = 160.0            # s     0.70 -> 0.06 over one 400 s night
# ratio 4.375:1 vs the canonical 4.33:1 (Daan 1984 via Borbely & Achermann
# 1999) — the 1 % deviation is deliberate and recorded in NE.01's notes.

# ── metabolic hub (§2.3). M_BASAL is a nominal scale: e-drain reads M/M_BASAL
# and the thermal ODE reads M/k_dry, so only RATIOS matter. 100 W is the
# human resting figure, kept for legibility.
M_BASAL = 100.0              # W, nominal
SHIVER_CAP = 2.0 * M_BASAL   # §2.3: "capped at 2*M_basal" -> M at most 3x basal
C_SH = SHIVER_CAP / 6.0      # W/C  PROPOSAL: cap reached exactly at delta_T's
                             # cold saturation (31 C, 6 C of deviation)
C_SW = 0.5                   # 1/C  PROPOSAL: water drain doubles exactly at
                             # delta_T's hot saturation (39 C, 2 C of deviation)

# ── thermal ODE (§2.3). PROPOSALS, sized so a resting dry agent in the open
# by day sits at the 37 C setpoint (k_dry = M_basal / (37 - T_day)), and a
# night in the open equilibrates ~3.0 C cold (dawn delta_T ~0.50, mid-band
# of NE.01's 0.3-0.6 gate, no frostbite) — "survivable once, costly", the
# §2.3 pedagogy. DELTA_T_NIGHT was CALIBRATED 12 -> 10 by NE.01's assigned
# pre-run sweep (2026-08-24): 12 put the open-night cost at 0.598, on the
# gate's own edge, where physics drift decides the verdict instead of the
# world; 10 reads 0.498. The 5-point table ships in NE.01's metrics.
T_SETPOINT = 37.0
T_DAY = 30.0                 # C  PROPOSAL
DELTA_T_NIGHT = 10.0         # C  calibrated by NE.01's sweep (see above)
T_WATER = 24.0               # C  PROPOSAL: the pool is cool, so soaking at
                             # night is the fastest available way to die
TAU_T = 240.0                # s  §2.3 gives it: the thermoneutral time constant
K_DRY = M_BASAL / (T_SETPOINT - T_DAY)      # W/C = 100/7
C_EFF = K_DRY * TAU_T        # J/C, §2.3: "so the time constant is exactly TAU_T"
KAPPA_WET = 4.0              # PROPOSAL: soaked skin loses heat 5x dry —
                             # bounded stand-in for water's far higher
                             # conductance; NE.01 measures what it does
OCC_CUT = 0.7                # §2.3 gives it: full occlusion cuts k_eff by 70 %
C_EV = 0.5 * M_BASAL         # W  PROPOSAL: full-wet evaporation sheds half
                             # basal at thermoneutral, more when sweating

# ── fatigue (§2.3). TAU_F_FALL given; TAU_F_RISE proposed so sustained max
# power drives f -> 1 exactly (steady state f* = (P/P_max)*TAU_F_FALL/TAU_F_RISE).
TAU_F_FALL = 60.0            # s   given: recovery in ~a minute of rest
TAU_F_RISE = 60.0            # s   PROPOSAL

# ── integrity (PS §2.2 verbatim + §2.3's three additions) ───────────────
RHO_HEAL = 1.0 / 900.0       # s^-1  inherited: full heal in 15 min of rest
T_INJURY_BAND = 5.0          # C     §2.3: damage when |T - 37| > 5
ALPHA_T = 1.0 / 600.0        # (C*s)^-1  PROPOSAL: at 28 C (4 C beyond the
                             # band) i drains at 0.4/min — a graded warning en
                             # route to the 20 s hypothermia death, never the
                             # faster killer

# ── death (§2.5, given). Grace windows exist because an instantaneous lethal
# bound is unlearnable: 300 s of starvation is 1,500 decisions.
DEATH_E_S = 300.0
DEATH_W_S = 120.0
DEATH_T_S = 20.0
T_COLD_DEATH = 28.0          # C  INCAPACITATION threshold (Swiss HT III), not
T_HOT_DEATH = 40.0           # C  a survival bound — §1.2's binding constraint:
                             # W0 has no medicine, so unconscious = dead. Never
                             # call the 9-vs-3 margin a SURVIVAL asymmetry.
DROWN_DEATH_S = 20.0         # s  head under this long kills (routed through i)

# ── microsleep (§2.5): sleep kills indirectly and legibly ───────────────
MS_P_FLOOR = 0.98            # p at which microsleeps become possible (given)
MS_P_ONSET_MAX = 0.05        # PROPOSAL: onset prob/decision at p = 1.0 (one
                             # microsleep ~every 4 s of maximal debt)
MS_DUR_S = (1.0, 2.0)        # §2.5 gives it: action zeroed for 1-2 s
MS_DEATH_WINDOW_S = 10.0     # §2.5: deaths_with_microsleep_within_10s

# ── food & drink. nu per item INHERITED from drives.py (the apple:floor
# ratio is the climb-vs-forage incentive and PS §2.3 calls it load-bearing).
# Respawn periods are PS.01's measured values RETIMED x3 — §0.1 row 1: "the
# constants were tuned for a 10-minute metabolic clock; §2.3 retimes them
# against a sim-day". b_e is exactly 3x slower than PS's basal (1/1800 vs
# 1/600) and drain(full activity) = 3x basal in both economies, so scaling
# the respawn periods by 3 preserves PS.01's C1-C3 supply-vs-drain criteria
# unchanged. NE.01 measures whether that scaling survives contact.
NU_APPLE = 0.50
NU_FLOORFOOD = 0.08
RESPAWN_APPLE_S = 3.0 * 129.6        # = 388.8
RESPAWN_FLOORFOOD_S = 3.0 * 66.9     # = 200.7
FOOD = {"apple": (NU_APPLE, RESPAWN_APPLE_S),
        "obj0": (NU_FLOORFOOD, RESPAWN_FLOORFOOD_S),
        "obj1": (NU_FLOORFOOD, RESPAWN_FLOORFOOD_S)}
NU_DRINK = 0.9               # PROPOSAL: ~one drink refills a 450 s tank, ~3
                             # drinks per sim-day at the doc's stated cadence
DRINK_REFRACTORY_S = 30.0    # PROPOSAL: a gulp, not a tap — drive reduction
                             # already pays zero at w = 1, this only stops the
                             # event counter saturating while he wades
DRINK_DEPTH_M = 0.25         # PROPOSAL: mouth within this of the surface

# ── social (§4.1/§4.3, given). Restorations are RECORDED WORLD EVENTS the
# caller reports (G-A rule: if it is not in the diary it did not happen);
# this layer only does the arithmetic.
NU_SOCIAL = {"proximity": 0.02, "conversation": 0.15,
             "helped": 0.25, "helping": 0.25}
BETA_BOUT = 0.6              # within-bout geometric decay (given)
BOUT_GAP_S = 300.0           # a bout ends after this long with no event (given)

# ── the drive function (§2.6, given) ────────────────────────────────────
LAMBDA = {"e": 1.0, "w": 1.0, "p": 0.5, "T": 1.0, "f": 0.3, "c": 0.3, "i": 1.0}
N_EXP, M_EXP = 4.0, 2.0      # Keramati & Gutkin: n > m >= 1 is what makes a
                             # fixed intake worth MORE when more deprived.
                             # NE.01 may retune INSIDE the constraint only.
RHO_ALIVE = 1.0 / 6000.0     # "one unit per sim-day survived" (§2.6). The
                             # reward itself is the arms' to build, not ours.

# Two of the three published sources misprint the inequality (NE.00's notes);
# a check binds the next author where prose cannot (LESSONS.md).
assert N_EXP > M_EXP >= 1.0, "K&G constraint n > m >= 1 violated"

# ── weakness (§2.3): gear_scale = 0.5 + 0.5*(1-f)*min(e,i) ─────────────
GEAR_FLOOR = 0.5

# ── observation (§2.4b, given): nine floats, appended OUTSIDE humanoid_obs ──
NEED_DIM = 9                 # [e, w, p, T_norm, f, c, i, d(h), pain]

# ── sky occlusion (§2.3, given): 9 upward rays from the head geom ───────
SKY_N_RAYS = 9
_SKY_CONE_DEG = 30.0         # PROPOSAL: zenith + 8 rays on a 30-degree cone
SKY_RAY_Z_OFFSET = 0.10      # m above the head geom, clearing its own radius


def _sky_dirs() -> np.ndarray:
    z = math.radians(_SKY_CONE_DEG)
    dirs = [(0.0, 0.0, 1.0)]
    for k in range(SKY_N_RAYS - 1):
        a = 2.0 * math.pi * k / (SKY_N_RAYS - 1)
        dirs.append((math.sin(z) * math.cos(a), math.sin(z) * math.sin(a),
                     math.cos(z)))
    return np.array(dirs, dtype=np.float64)


_SKY_DIRS = _sky_dirs()


# ── pure pieces, importable without a model ─────────────────────────────
def delta_T(T: float) -> float:
    """delta_T in [0,1] — ASYMMETRIC on purpose (§2.2): saturates at 31 C
    (3 C before cold death) and 39 C (1 C before hot death), so the drive
    still has gradient where a policy can act on it."""
    if T < T_SETPOINT:
        return min(1.0, (T_SETPOINT - T) / 6.0)
    return min(1.0, (T - T_SETPOINT) / 2.0)


def t_norm(T: float) -> float:
    """The OBSERVATION carries the sign delta_T discards (§2.4b): the drive
    does not care which way you are dying, the policy must."""
    if T < T_SETPOINT:
        return float(np.clip((T - T_SETPOINT) / 6.0, -1.0, 1.0))
    return float(np.clip((T - T_SETPOINT) / 2.0, -1.0, 1.0))


def deltas(e, w, p, T, f, c, i) -> dict:
    return {"e": 1.0 - e, "w": 1.0 - w, "p": p, "T": delta_T(T),
            "f": f, "c": 1.0 - c, "i": 1.0 - i}


def need_drive(e, w, p, T, f, c, i) -> float:
    """d(h) = (sum_k lambda_k * delta_k^n)^(1/m), n=4, m=2 (§2.6)."""
    d = deltas(e, w, p, T, f, c, i)
    s = sum(LAMBDA[k] * d[k] ** N_EXP for k in LAMBDA)
    return float(s ** (1.0 / M_EXP))


def metabolic_rate(p_mech: float, T: float, kappa_act: float) -> float:
    """M = M_basal + kappa_act*P_mech + shiver (§2.3). Shivering is capped at
    2x basal, so deep cold at most TRIPLES the drain — the '3x' in the §2.2
    interaction table is this cap, not a tuned cross-term."""
    shiver = min(SHIVER_CAP, C_SH * max(0.0, T_SETPOINT - T))
    return M_BASAL + kappa_act * p_mech + shiver


def k_eff(skin_wetness: float, sky_occlusion: float) -> float:
    return K_DRY * (1.0 + KAPPA_WET * skin_wetness) * (1.0 - OCC_CUT * sky_occlusion)


def e_evap(skin_wetness: float, T: float) -> float:
    """Evaporative cooling NEEDS wet skin (§2.3's letter): a hot DRY agent
    sheds heat only through k_eff — the pool is the remedy for heat, which is
    what gives PG.2's water a warm-weather stake to match its cold one."""
    return C_EV * skin_wetness * (1.0 + C_SW * max(0.0, T - T_SETPOINT))


def thermal_step(T: float, p_mech: float, skin_wetness: float,
                 sky_occlusion: float, T_env: float, dt: float,
                 kappa_act: float) -> float:
    """One Euler step of C_eff*dT = M - k_eff*(T - T_env) - E_evap (§2.3).
    Pure, so NE.01's §9 coarse-vs-fine check can drive it at any dt."""
    m = metabolic_rate(p_mech, T, kappa_act)
    dT = (m - k_eff(skin_wetness, sky_occlusion) * (T - T_env)
          - e_evap(skin_wetness, T)) / C_EFF
    return T + dT * dt


@dataclass
class NeedState:
    e: float = 1.0
    w: float = 1.0
    p: float = 0.05          # §2.3's worked day starts at 0.05, not 0
    T: float = T_SETPOINT
    f: float = 0.0
    c: float = 1.0
    i: float = 1.0
    skin_wetness: float = 0.0    # NOT a need — a thermal multiplier (§2.2)

    def d(self) -> float:
        return need_drive(self.e, self.w, self.p, self.T, self.f, self.c, self.i)


class NeedLayer:
    """The §2.3 integrator. Caller owns `mj_step`; this owns `h`.

    Usage, one decision (identical contract to `DriveLayer`):

        layer.begin_decision()
        for _ in range(frame_skip):
            ctrl_eff = ctrl * layer.gear_scale()
            if layer.microsleep_zeroed():
                ctrl_eff = 0.0 * ctrl_eff        # §2.5: action zeroed 1-2 s
            data.ctrl[:] = ctrl_eff
            mujoco.mj_step(model, data)
            layer.substep(model, data, model.opt.timestep)
        h = layer.decide()
        if layer.dead: ...                       # world handles the respawn

    `j0`/`alpha` are PS.01's measured impact constants (borrow_metrics
    "PS.01"); `p_max` is the body's full-power mechanical output — for the
    humanoid PS.01 unit (a) measured P_bar(1) = 1434.8 W. None has a default:
    a wrong number here is silent (drives.py's own rule). `kappa_act` is
    derived, not passed: 2*M_BASAL/p_max, so constant full power exactly
    triples M — the same criterion PS.01's re-derived KAPPA satisfies.

    Sleep is a caller decision (`set_asleep`): the world or the policy says
    when he lies down; this layer says what it does to him. Microsleeps are
    the layer's own (they are physiology, not policy).
    """

    def __init__(self, model, *, j0: float, alpha: float, p_max: float,
                 pool: Optional[tuple] = None, body: Optional[BodyRef] = None,
                 state: Optional[NeedState] = None, seed: int = 0):
        for name, v in (("j0", j0), ("alpha", alpha), ("p_max", p_max)):
            if v is None:
                raise ValueError(f"{name} must be measured, not defaulted")
        self.model = model
        self.j0, self.alpha = float(j0), float(alpha)
        self.p_max = float(p_max)
        self.kappa_act = 2.0 * M_BASAL / self.p_max
        self.pool = pool                    # (x, y, half, surface_z) or None
        self.state = state or NeedState()
        self.rng = np.random.RandomState(seed)

        self.body = body or humanoid_body_ref(model)
        self._body_ids = set(self.body.bodies)
        geom_of_body = {g: int(model.geom_bodyid[g]) for g in range(model.ngeom)}
        self._jack_geoms = {g for g, b in geom_of_body.items()
                            if b in self._body_ids}
        self._jack_mask = np.zeros(model.ngeom, dtype=bool)
        self._jack_mask[list(self._jack_geoms)] = True
        self._jack_gids = np.array(sorted(self._jack_geoms), dtype=int)
        # The mouth (§2.3: "mouth geom contacts food"): head_geoms by intent —
        # the humanoid's head, the rover's torso capsule (its highest geom).
        self._mouth_geoms = set(int(g) for g in self.body.head_geoms)
        self._mouth_mask = np.zeros(model.ngeom, dtype=bool)
        self._mouth_mask[list(self._mouth_geoms)] = True

        self._food = {}
        for name, (nu, respawn) in FOOD.items():
            try:
                self._food[name] = (int(model.geom(name).id), nu, respawn)
            except (KeyError, ValueError):
                pass                        # a mutated world may not carry it
        self._dofadr, self._ndof = self.body.dofadr, self.body.ndof

        # ── world state (survives new_body, exactly DriveLayer's split) ──
        self.t = 0.0
        self._respawn_at = {name: 0.0 for name in self._food}
        self.ate_total = {name: 0 for name in self._food}
        self.drank_total = 0

        self._mj = None                     # lazy mujoco handle (rays only)
        self._geomid = np.zeros(1, dtype=np.int32)
        self._reset_body_state()
        self._reset_decision()

        # Diagnostics the caller may read after any decision.
        self.last_j = 0.0
        self.last_power_w = 0.0
        self.last_dt = 0.0
        self.last_rest_dt = 0.0
        self.last_occlusion = 0.0
        self.last_pain = 0.0
        self.n_onsets = 0

    # ── the clock (§2.3) ────────────────────────────────────────────────
    def is_night(self, t: Optional[float] = None) -> bool:
        return ((self.t if t is None else t) % SIM_DAY_S) >= DAY_S

    def T_env(self, t: Optional[float] = None) -> float:
        return T_DAY - (DELTA_T_NIGHT if self.is_night(t) else 0.0)

    # ── caller-facing physiology ────────────────────────────────────────
    def gear_scale(self) -> float:
        """§2.3: 0.5 + 0.5*(1-f)*min(e, i) — PS's weakness floor, now
        fatigue-gated. Nothing terminates; incapacity is soft."""
        s = self.state
        return GEAR_FLOOR + (1.0 - GEAR_FLOOR) * (1.0 - s.f) * min(s.e, s.i)

    def microsleep_zeroed(self) -> bool:
        """True while a microsleep holds the actuators at zero. The caller
        multiplies ctrl by 0; the layer never touches `data`."""
        return self._ms_until > self.t

    def set_asleep(self, asleep: bool) -> None:
        """The caller's sleep switch. Onset sets f -> 0 outright (§2.3)."""
        if asleep and not self.asleep:
            self.state = replace(self.state, f=0.0)
        self.asleep = bool(asleep)

    def social_event(self, channel: str) -> float:
        """Apply one RECORDED restoration event (§4.1's four channels) with
        §4.3's within-bout decay: nu * beta^k, bout ends after 300 s quiet.
        Returns the delta actually applied. The diary write is the CALLER's
        duty (G-A: if it is not in the diary, it did not happen)."""
        nu = NU_SOCIAL[channel]
        if self.t - self._last_social_t > BOUT_GAP_S:
            self._bout_k = 0
        dc = nu * (BETA_BOUT ** self._bout_k)
        self._bout_k += 1
        self._last_social_t = self.t
        c = float(np.clip(self.state.c + dc, 0.0, 1.0))
        applied = c - self.state.c
        self.state = replace(self.state, c=c)
        return applied

    # ── the decision cycle ──────────────────────────────────────────────
    def begin_decision(self) -> None:
        self._reset_decision()

    def _reset_decision(self) -> None:
        self._j_max = 0.0
        self._power_dt = 0.0
        self._dt_acc = 0.0
        self._ate = 0.0
        self._drank = 0.0
        self._rest_dt = 0.0
        self._occl_done = False

    def _reset_body_state(self) -> None:
        self.asleep = False
        self.dead = False
        self.death_record: Optional[dict] = None
        self._touching_world = False
        self._prev_speed: Optional[float] = None
        self._submerged_since: Optional[float] = None
        self._t_e0 = 0.0
        self._t_w0 = 0.0
        self._t_cold = 0.0
        self._t_hot = 0.0
        self._drink_ready_at = 0.0
        self._ms_until = -1.0
        self._ms_last_end = -np.inf
        self._bout_k = 0
        self._last_social_t = -np.inf
        self._occlusion = 0.0

    def substep(self, model, data, dt: float) -> None:
        """Everything that is a function of the physics, per mj_step.
        Mechanical power is integrated here (a 200 Hz signal sampled at 40 Hz
        would alias — drives.py's reasoning, unchanged)."""
        d, n = self._dofadr, self._ndof
        tau = np.asarray(data.qfrc_actuator[d + 6:d + n])
        omega = np.asarray(data.qvel[d + 6:d + n])
        self._power_dt += float(np.abs(tau * omega).sum()) * dt
        self._dt_acc += dt

        qvel = np.asarray(data.qvel[d:d + n])
        if float(np.linalg.norm(qvel)) < Q_REST:
            self._rest_dt += dt

        # Impact channel — PS.01/J2's winner, verbatim from drives.py: the
        # root's linear speed one substep before a False->True world-contact
        # edge. Lying on the floor cannot manufacture damage.
        partners = _contact_partners(data, self._jack_mask)
        speed = float(np.linalg.norm(qvel[:3]))
        touching = bool(partners - self._jack_geoms)
        if touching and not self._touching_world:
            arrival = speed if self._prev_speed is None else self._prev_speed
            self._j_max = max(self._j_max, arrival)
            self.n_onsets += 1
        self._touching_world = touching
        self._prev_speed = speed

        # Eating: MOUTH contact with a live food geom (§2.3).
        mouth_partners = _contact_partners(data, self._mouth_mask)
        now = self.t + self._dt_acc
        for name, (gid, nu, respawn) in self._food.items():
            if now < self._respawn_at[name]:
                continue
            if gid in mouth_partners:
                self._ate += nu
                self._respawn_at[name] = now + respawn
                self.ate_total[name] += 1

        # Drinking: a discrete event — mouth in the pool region, near the
        # surface, off refractory. An event, not a rate: restoration is a
        # world event the accounting identity can audit (§0.1, G-A).
        if self.pool is not None and now >= self._drink_ready_at:
            if self._mouth_near_surface(data):
                self._drank += NU_DRINK
                self._drink_ready_at = now + DRINK_REFRACTORY_S
                self.drank_total += 1

        # Submersion clock (drowning, §2.5) and skin wetness (thermal).
        if self.pool is not None and self._head_under(data):
            if self._submerged_since is None:
                self._submerged_since = now
        else:
            self._submerged_since = None
        self._integrate_wet(dt, self._any_geom_in_water(data))

        # Sky occlusion: 9 rays, cast ONCE per decision (first substep).
        if not self._occl_done:
            self._occlusion = self._sky_occlusion(model, data)
            self._occl_done = True

    def decide(self) -> NeedState:
        """Close the decision: §2.3's update rules, §2.5's deaths, clip."""
        dt = self._dt_acc
        s = self.state
        if self.dead or dt <= 0.0:
            self._reset_decision()
            return s
        p_mean = self._power_dt / dt
        occl = self._occlusion
        t_env = self.T_env()

        # All cross-terms read the DECISION-START state, then h updates as one
        # block — no within-decision ordering artifacts.
        m_rate = metabolic_rate(p_mean, s.T, self.kappa_act)
        e = s.e - (m_rate / M_BASAL) * B_E * dt + self._ate
        w = s.w - (1.0 + C_SW * max(0.0, s.T - T_SETPOINT)) * B_W * dt + self._drank
        T = thermal_step(s.T, p_mean, s.skin_wetness, occl, t_env, dt,
                         self.kappa_act)
        if self.asleep:
            p = s.p * math.exp(-dt / TAU_SLEEP)
        else:
            p = s.p + (1.0 - s.p) * (1.0 - math.exp(-dt / TAU_WAKE))
        f = s.f + (p_mean / (self.p_max * TAU_F_RISE) - s.f / TAU_F_FALL) * dt
        c = s.c - B_C * dt
        therm_excess = max(0.0, abs(s.T - T_SETPOINT) - T_INJURY_BAND)
        i = (s.i
             - self.alpha * max(0.0, self._j_max - self.j0)
             - ALPHA_T * therm_excess * dt
             + RHO_HEAL * self._rest_dt)

        new = NeedState(e=float(np.clip(e, 0.0, 1.0)),
                        w=float(np.clip(w, 0.0, 1.0)),
                        p=float(np.clip(p, 0.0, 1.0)),
                        T=float(T),
                        f=float(np.clip(f, 0.0, 1.0)),
                        c=float(np.clip(c, 0.0, 1.0)),
                        i=float(np.clip(i, 0.0, 1.0)),
                        skin_wetness=s.skin_wetness)
        self.last_pain = max(0.0, s.i - new.i)      # phasic, rectified (§2.4b)
        self.state = new
        self.t += dt
        self.last_j = self._j_max
        self.last_power_w = p_mean
        self.last_dt = dt
        self.last_rest_dt = self._rest_dt
        self.last_occlusion = occl

        self._update_microsleep(dt)
        self._check_death(dt)
        self._reset_decision()
        return self.state

    # ── death (§2.5) ────────────────────────────────────────────────────
    def _check_death(self, dt: float) -> None:
        s = self.state
        self._t_e0 = self._t_e0 + dt if s.e <= 0.0 else 0.0
        self._t_w0 = self._t_w0 + dt if s.w <= 0.0 else 0.0
        self._t_cold = self._t_cold + dt if s.T <= T_COLD_DEATH else 0.0
        self._t_hot = self._t_hot + dt if s.T >= T_HOT_DEATH else 0.0

        cause = None
        if (self._submerged_since is not None
                and self.t - self._submerged_since > DROWN_DEATH_S):
            self.state = replace(s, i=0.0)          # routed through i (§2.5)
            cause = "drowning"
        elif s.i <= 0.0:
            cause = "injury"
        elif self._t_w0 >= DEATH_W_S:
            cause = "dehydration"
        elif self._t_e0 >= DEATH_E_S:
            cause = "starvation"
        elif self._t_cold >= DEATH_T_S:
            cause = "hypothermia"
        elif self._t_hot >= DEATH_T_S:
            cause = "hyperthermia"
        if cause is not None:
            self.dead = True
            self.death_record = {
                "cause": cause, "t": self.t,
                "microsleep_within_10s":
                    (self.t - self._ms_last_end) <= MS_DEATH_WINDOW_S,
                "state": vars(self.state).copy(),
            }

    # ── microsleep (§2.5): physiology, owned here, seeded ───────────────
    def _update_microsleep(self, dt: float) -> None:
        if self.asleep:
            return
        if self.microsleep_zeroed():
            return
        if self._ms_until > 0 and self.t >= self._ms_until:
            self._ms_last_end = self._ms_until
            self._ms_until = -1.0
        p = self.state.p
        if p >= MS_P_FLOOR:
            prob = MS_P_ONSET_MAX * (p - MS_P_FLOOR) / (1.0 - MS_P_FLOOR)
            if self.rng.rand() < prob:
                dur = self.rng.uniform(*MS_DUR_S)
                self._ms_until = self.t + dur

    # ── sleep-time compression (§9's coarse path) ───────────────────────
    def sleep_coarse_step(self, dt: float, dt_int: float = 1.0) -> NeedState:
        """Advance a SLEEPING, motionless body `dt` sim-seconds with no
        physics: P_mech = 0, at rest (heals), eats and drinks nothing, keeps
        cooling, keeps draining. Internally sub-cycles the thermal ODE at
        `dt_int` so the trajectory NE.01's §9 gate compares against the
        fine-step one is an integration, not one giant Euler leap."""
        if not self.asleep:
            raise RuntimeError("coarse stepping is for sleep only (§9)")
        s = self.state
        remaining = float(dt)
        T, e, w, i = s.T, s.e, s.w, s.i
        while remaining > 1e-12 and not self.dead:
            h = min(dt_int, remaining)
            t_env = self.T_env()
            m_rate = metabolic_rate(0.0, T, self.kappa_act)
            e -= (m_rate / M_BASAL) * B_E * h
            w -= (1.0 + C_SW * max(0.0, T - T_SETPOINT)) * B_W * h
            therm_excess = max(0.0, abs(T - T_SETPOINT) - T_INJURY_BAND)
            i += (RHO_HEAL - ALPHA_T * therm_excess) * h
            T = thermal_step(T, 0.0, s.skin_wetness, self._occlusion,
                             t_env, h, self.kappa_act)
            self.state = NeedState(
                e=float(np.clip(e, 0.0, 1.0)), w=float(np.clip(w, 0.0, 1.0)),
                p=float(self.state.p * math.exp(-h / TAU_SLEEP)),
                T=float(T), f=0.0, c=float(np.clip(s.c - B_C * h, 0.0, 1.0)),
                i=float(np.clip(i, 0.0, 1.0)), skin_wetness=s.skin_wetness)
            self.t += h
            self._check_death(h)
            remaining -= h
        return self.state

    # ── W0-2: a new body, without a new world (drives.py's split, verbatim) ──
    def new_body(self, state: Optional[NeedState] = None) -> None:
        """Death: reset everything that belongs to the BODY, nothing else.
        The world clock, the food respawn timers and the cumulative counters
        survive — resetting them would hand every death a freshly stocked
        larder (drives.new_body's reasoning, inherited whole)."""
        self.state = state or NeedState()
        self._reset_body_state()
        self._reset_decision()

    # ── observation (§2.4b): nine floats, the null gets them too ────────
    def obs(self) -> np.ndarray:
        s = self.state
        v = np.array([s.e, s.w, s.p, t_norm(s.T), s.f, s.c, s.i,
                      s.d(), self.last_pain], dtype=np.float64)
        if v.shape[0] != NEED_DIM:
            raise RuntimeError(f"need obs is {v.shape[0]}, not {NEED_DIM}")
        return v

    # ── internals ───────────────────────────────────────────────────────
    def _integrate_wet(self, dt: float, in_water: bool) -> None:
        w = self.state.skin_wetness
        if in_water:
            w += (1.0 - w) * dt / 120.0     # PS's tau, inherited (logged-only
        else:                               # there; load-bearing here via k_eff)
            w -= w * dt / 120.0
        self.state = replace(self.state,
                             skin_wetness=float(np.clip(w, 0.0, 1.0)))

    def _mouth_near_surface(self, data) -> bool:
        x, y, half, surf = self.pool
        for g in self._mouth_geoms:
            p = data.geom_xpos[g]
            if (abs(p[0] - x) <= half and abs(p[1] - y) <= half
                    and abs(p[2] - surf) <= DRINK_DEPTH_M):
                return True
        return False

    def _head_under(self, data) -> bool:
        x, y, half, surf = self.pool
        for g in self._mouth_geoms:
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

    def _sky_occlusion(self, model, data) -> float:
        """Fraction of 9 upward rays from the head that hit something (§2.3).
        Geometric, unlabelled: no shelter zone, no tagged geom — pushed
        objects occlude sky, so shelter is CONSTRUCTIBLE and ungameable by
        going to a place. A ray that hits Jack's own geom counts as OPEN
        (declared: a hand over his head is not a roof; casting past his own
        body would need an N-bounce ray MuJoCo does not offer)."""
        if self._mj is None:
            import mujoco               # compute-only; callers who RENDER own
            self._mj = mujoco           # the ensure_gl()-before-import rule
        head = int(next(iter(self._mouth_geoms)))
        pnt = np.array(data.geom_xpos[head], dtype=np.float64)
        pnt[2] += SKY_RAY_Z_OFFSET
        hits = 0
        for k in range(SKY_N_RAYS):
            dist = self._mj.mj_ray(model, data, pnt, _SKY_DIRS[k],
                                   None, 1, -1, self._geomid)
            gid = int(self._geomid[0])
            if dist >= 0 and gid >= 0 and gid not in self._jack_geoms:
                hits += 1
        return hits / float(SKY_N_RAYS)


# ── self-test: known answers from the doc's own worked numbers ──────────
def _self_test() -> int:
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}  {detail}")
        if not ok:
            failures.append(name)

    # 1. Sleep pressure, §2.3's worked day: 0.05 -> ~0.70 over 800 s awake,
    #    back to ~0.06 over 400 s asleep.
    p = 0.05
    for _ in range(4000):                      # 800 s at 0.2 s decisions
        p = p + (1 - p) * (1 - math.exp(-0.2 / TAU_WAKE))
    check("sleep: day 0.05 -> ~0.70", abs(p - 0.697) < 0.01, f"p={p:.4f}")
    for _ in range(2000):                      # 400 s asleep
        p = p * math.exp(-0.2 / TAU_SLEEP)
    check("sleep: night -> ~0.057", abs(p - 0.057) < 0.005, f"p={p:.4f}")

    # 2. delta_T asymmetry: saturates at 31 C and 39 C; sign in t_norm.
    check("delta_T saturates at 31 C", delta_T(31.0) == 1.0 and delta_T(31.5) < 1.0)
    check("delta_T saturates at 39 C", delta_T(39.0) == 1.0 and delta_T(38.5) < 1.0)
    check("t_norm signed", t_norm(34.0) < 0 < t_norm(38.0))

    # 3. K&G deprivation direction (the misprint guard, NE.00's notes): a
    #    fixed intake must be worth MORE when more deprived.
    lo = need_drive(0.2, 1, 0, T_SETPOINT, 0, 1, 1) - need_drive(0.3, 1, 0, T_SETPOINT, 0, 1, 1)
    hi = need_drive(0.8, 1, 0, T_SETPOINT, 0, 1, 1) - need_drive(0.9, 1, 0, T_SETPOINT, 0, 1, 1)
    check("K&G: intake worth more when deprived", lo > hi > 0,
          f"deprived={lo:.4f} sated={hi:.4f}")

    # 4. The two capped couplings: deep cold triples M; 39 C doubles water drain.
    check("shiver cap: M(rest, 25 C) = 3x basal",
          abs(metabolic_rate(0.0, 25.0, 0.0) - 3 * M_BASAL) < 1e-9)
    drain = 1.0 + C_SW * (39.0 - T_SETPOINT)
    check("sweat: water drain at 39 C = 2x", abs(drain - 2.0) < 1e-9)

    # 5. Thermal equilibria (the proposals' own design criteria): resting dry
    #    day -> 37 C; night open equilibrates cold but above the 32 C frostbite
    #    band; a night in the pool region (wet) falls below it.
    T = 37.0
    for _ in range(5000):
        T = thermal_step(T, 0.0, 0.0, 0.0, T_DAY, 0.2, 0.0)
    check("day rest equilibrium ~37 C", abs(T - 37.0) < 0.05, f"T={T:.2f}")
    T = 37.0
    for _ in range(5000):
        T = thermal_step(T, 0.0, 0.0, 0.0, T_DAY - DELTA_T_NIGHT, 0.2, 0.0)
    check("night open: cold but no frostbite", 32.0 < T < 35.0, f"T={T:.2f}")
    Tw = 37.0
    for _ in range(5000):
        Tw = thermal_step(Tw, 0.0, 1.0, 0.0, T_DAY - DELTA_T_NIGHT, 0.2, 0.0)
    check("night soaked: below frostbite band", Tw < 32.0, f"T={Tw:.2f}")
    check("wet colder than dry", Tw < T)

    # 6. Statue arithmetic (the flagged NE.01 discrepancy, pinned by a check):
    #    water empties at 450 s and kills at 570 s — BEFORE starvation's
    #    1,800 + 300 s. The registry's "dies of starvation" prose is wrong.
    t_dehydr = 450.0 + DEATH_W_S
    t_starve = 1800.0 + DEATH_E_S
    check("statue dies of DEHYDRATION first", t_dehydr < t_starve,
          f"{t_dehydr:.0f}s vs {t_starve:.0f}s")

    # 7. Coarse-vs-fine thermal agreement over one 400 s night (statue, dry,
    #    open sky) — §9's gate is 0.2 C; the integrator itself must be far
    #    inside it so NE.01 measures the SLEEP path, not Euler error.
    Tf = 34.0
    for _ in range(2000):
        Tf = thermal_step(Tf, 0.0, 0.0, 0.0, T_DAY - DELTA_T_NIGHT, 0.2, 0.0)
    Tc = 34.0
    for _ in range(400):
        Tc = thermal_step(Tc, 0.0, 0.0, 0.0, T_DAY - DELTA_T_NIGHT, 1.0, 0.0)
    check("coarse(1 s) vs fine(0.2 s) night < 0.02 C", abs(Tf - Tc) < 0.02,
          f"|dT|={abs(Tf - Tc):.4f}")

    # 8. Live smoke on the playground humanoid: 100 random decisions — obs is
    #    9 finite floats, occlusion in [0,1], needs move, nothing crashes.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from playground import make_playground, PlaygroundParams
    p = PlaygroundParams()
    model, data, water = make_playground(p, with_humanoid=True)
    layer = NeedLayer(model, j0=1.0, alpha=0.1, p_max=1434.8,  # SMOKE ONLY —
                      pool=(2.6, -2.4, p.pool_size, 0.0), seed=0)
    # the registered NE.01 run borrows PS.01's j0/alpha from the ledger.
    rng = np.random.RandomState(0)
    obs0 = layer.obs()
    for _ in range(100):
        layer.begin_decision()
        ctrl = rng.uniform(-0.4, 0.4, model.nu) * layer.gear_scale()
        if layer.microsleep_zeroed():
            ctrl *= 0.0
        for _ in range(40):
            data.ctrl[:] = ctrl
            import mujoco
            mujoco.mj_step(model, data)
            if water is not None:
                water.apply(model, data)
            layer.substep(model, data, model.opt.timestep)
        layer.decide()
        if layer.dead:
            break
    o = layer.obs()
    check("smoke: obs is 9 finite floats",
          o.shape == (NEED_DIM,) and bool(np.all(np.isfinite(o))))
    check("smoke: occlusion in [0,1]", 0.0 <= layer.last_occlusion <= 1.0,
          f"occ={layer.last_occlusion:.2f}")
    check("smoke: needs moved", bool(np.any(o != obs0)),
          f"e={layer.state.e:.4f} p={layer.state.p:.4f} T={layer.state.T:.2f}")
    check("smoke: clock advanced ~20 s", abs(layer.t - 100 * 0.2) < 1e-6,
          f"t={layer.t:.1f}")

    # 9. Death machinery: pin w at 0 -> dehydration after 120 s; microsleep
    #    bookkeeping fields present in the record.
    layer2 = NeedLayer(model, j0=1.0, alpha=0.1, p_max=1434.8, seed=1)
    layer2.state = replace(layer2.state, w=0.0)
    n = 0
    while not layer2.dead and n < 2000:
        layer2._dt_acc = 0.2                # pure-integrator drive: no physics,
        layer2._occl_done = True            # statue semantics
        layer2.decide()
        layer2.state = replace(layer2.state, w=0.0)
        n += 1
    check("death: w=0 kills at 120 s",
          layer2.dead and layer2.death_record["cause"] == "dehydration"
          and abs(layer2.death_record["t"] - DEATH_W_S) <= 0.4,
          f"t={layer2.death_record['t'] if layer2.death_record else None}")
    check("death record carries microsleep flag",
          layer2.death_record is not None
          and "microsleep_within_10s" in layer2.death_record)

    # 10. Microsleeps fire at pinned p=1 and last 1-2 s.
    layer3 = NeedLayer(model, j0=1.0, alpha=0.1, p_max=1434.8, seed=2)
    events = 0
    for _ in range(2000):
        layer3._dt_acc = 0.2
        layer3._occl_done = True
        layer3.state = replace(layer3.state, p=1.0, w=1.0, e=1.0)
        was = layer3.microsleep_zeroed()
        layer3.decide()
        if layer3.microsleep_zeroed() and not was:
            events += 1
    check("microsleep: fires at p=1", events > 5, f"onsets={events} in 400 s")

    print(f"\n{'PASS' if not failures else 'FAIL'}: "
          f"{len(failures)} failure(s) of the needs-integrator self-test")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
