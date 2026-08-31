"""LT.01 — The Ladder Test is measurable: null floor and un-gameable rise.

Stage 0 of the Ladder Test (docs/research/CURIOSITY_BAKEOFF.md §2.4-2.7, §5):
before a single arm is trained, certify that the test itself is honest and the
target reachable. Four questions, all measured, none argued:

  1. NULL FLOOR IS ZERO: a free-roaming random climber-rover produces zero
     ENGAGED attempts (ladder-supported rise >= 0.25 m under the full h(t)
     conjunction) in 3 seeds x 3,000 decisions (600 s of life each).
  2. RAW HEIGHT IS GAMEABLE: the same random agent reaches >= 0.6 m of torso
     rise with NO ladder involvement (stairs, ramp, tumbles) — so absolute
     torso z cannot be the metric, and the three-clause h(t) is load-bearing.
  3. THE BOOTSTRAP EXISTS: from the ladder base, a genuine weight-bearing hang
     (contact AND airborne, held >= 0.5 s, ladder bearing >= 0.5 x body
     weight, rise >= 0.25 m) occurs in 1-5%% of 3 s random bursts — so the
     first success is reachable by chance and learning progress has something
     to select over (no Go-Explore archive needed).
  4. NO ALTERNATE ROUTE: neither the random agent nor a greedy
     height-maximising oracle WITH ADHESION DISABLED ever reaches the platform
     — so a SUCCESS in LT.03+ can only be climbing.

THE h(t) CONJUNCTION (§2.4, frozen 2026-08-09 against measurement — the two
rejected definitions scored 0.55 and 0.063 false-positive under random action):

    h(t) = z(torso) - z_rest   iff  (i)  CLIMB x LADDER contact at t
                                    (ii) no body x GROUND contact at t
                                    (iii) (i)&(ii) held continuously >= 0.5 s
                                         AND the ladder's vertical force on
                                         the body >= 0.5 x body weight
    else 0

  LADDER = {rung*, ladder_railL, ladder_railR}   CLIMB = {handL, handR, foot}
  GROUND = {floor, ramp, stair*, seesaw_plank, poolwall*, pool_floor, obj*,
            welded_block, platform}       (resolved by NAME against the live
            model, so the sets survive world mutation)
  ATTEMPT: maximal interval bracketed by CLIMB x LADDER contact, gaps < 3 s.
  ENGAGED: attempt with max h >= 0.25 m.
  PLATFORM REACH: torso z >= ladder_height - 0.15 AND torso xy inside the
  platform footprint; ladder-supported only if >= 3 distinct rungs were
  contacted in the bracketing attempt.

z_rest and body weight are measured per world at build time (pilot: 0.360 m,
322 N), never assumed. The vertical-force SIGN CONVENTION is calibrated per
world against the foot-on-floor rest contact (the settled body must read
~ +1.0 body weight of floor support); a rig that cannot reproduce its own
weight does not get to certify a zero.

WORLDS: per-seed `PlaygroundParams().mutate()` at the documented strength
0.15 (§2.2: no arm ever sees a hand-picked world; G4: LT.01 re-measures both
ceilings per world mutation). The body is `playground._rover_fragments` — the
PG.3-certified arms/adhesion/contact classes unchanged, W0's action mapping
(6 ctrl + gated 600 N drive) without W0's needs machinery: no gear weakness,
no death, reward identically zero. An unbroken life, per §2.1.

THRESHOLDS are the registry's, fixed from the 2026-08-09 pilot BEFORE this
implementation existed (G6). The envelope equals the pilot's envelope
(3 x 3,000 free-roam decisions; 810 bursts vs the pilot's 800), so no
extreme-value bar is being read at a larger n than froze it (the T3.06
lesson, checked by construction).

PRE-REGISTERED GATES, in test order:

  VOID — an instrument fault is not a world reading (T0.22):
    V1 non-finite qpos/qvel in any phase             "non-finite physics"
    V2 force calibration: at rest, neither sign convention recovers body
       weight from floor contacts within 30%%        "force sign uncalibrated"
    V3 the SCRIPTED HANG (deterministic grip-and-pull at the base, PG.3's
       certified physics) fails to register an ENGAGED attempt through the
       full h(t) conjunction — then the free-roam zero is a dead instrument,
       not a null floor (the at-chance-control lesson: a zero must carry
       proof its instrument was alive).             "h(t) never fired on a
                                                     scripted hang"
    V4 the oracle control never gains >= 0.25 m of rise in some seed — an
       optimiser that never moved uphill cannot certify the absence of an
       alternate route.                             "oracle never climbed"

  CLAIM — PASS iff ALL FOUR, else FAIL with `claim_branch` naming the first
  falsified clause (the BA.03 one-bit-verdict lesson):
    C1 null_engaged_attempts == 0 on every seed (mean of a non-negative
       count is 0 iff every seed is 0)
    C2 >= 0.6 m of non-ladder torso rise reached in at least one seed
       (per-seed indicator; the pooled-max reading of the hypothesis)
    C3 pooled P(hang | 3 s burst) within [0.01, 0.05] under the adopted
       persistent+load-bearing definition (equal bursts per seed, so the
       mean of per-seed rates IS the pooled rate)
    C4 platform never reached by a non-ladder route: free-roam random
       (metric) AND the adhesion-disabled oracle (control) both zero

  The registry's control direction is C4's second conjunct: the oracle must
  FAIL to reach the platform. Its aliveness is V4, per T0.22.

ORACLE (control): greedy random-shooting height maximisation — K=8 candidate
actions (plus hold-previous), 1 s lookahead by state save/restore (PG.3's
snapshot pattern), score = max torso z over the rollout, adhesion channels
forced to zero everywhere, 20 s of decisions from each of four sites (ladder
base, stairs base, ramp base, seesaw/object field). Site starts are the
strongest honest attack on "no alternate route": a myopic optimiser is not
asked to cross the arena, it is handed each candidate route's doorstep.

PILOT RECORD (seed 90, disjoint from recorded seeds 0-2; mechanics and
runtime only — every claim bar above was frozen 2026-08-09, nothing here
tuned one; 2026-08-31, this commit):
  settle: z_rest 0.3896 m, weight 321.77 N, floor support 321.77 N (+1.000 W,
  sign convention: rotated force acts on geom2) — V2's calibration recovers
  weight exactly. Scripted hang: engaged 1.0, h_max 0.3701 m — V3's
  conjunction fires end-to-end, and it is what surfaced the SIGN finding in
  `_ladder_force_z`'s docstring (hanging dead still, the signed contact
  force reads -1556 N; the first draft gated signed-positive and scored the
  certified hang 0.0). Bursts, 60: p_inst 0.100, p_persist 0.050, p_load
  0.0333 (2/60), best hang rise 0.5101 m. Oracle, 4 sites x 30 decisions:
  rise 0.2761 m, platform 0 (V4's floor clears at a third of the registered
  envelope). Free-roam, 500 decisions: 0 engaged, 0 platform — and
  NON-LADDER RISE 0.0099 m: the body tips over within seconds (torso z
  ~0.08) and travels by dragging (3.29 m in 60 s, mobility probe), never
  regaining standing. That is W0.BAL's measured 0.002-0.004 upright fraction
  showing up in this rig, and it says C2 (the 08-09 pilot's 1.007 m
  free-roam z ceiling) may not reproduce on the as-built body. The gate
  stays exactly as pre-registered — dropping the 0.6 m clause after seeing
  this pilot would be post-hoc threshold selection, and a FAIL on that
  branch is a real measurement feeding the body-has-no-chair fork
  (PROGRESS.md 08-31, FOR THE OWNER §1). Runtime: ~11 min/seed both phases,
  inside CPU_LONG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# This spec certifies properties of the WORLD and the body; both live in
# playground.py, so it hashes into impl_sha (PG.3's precedent).
IMPL_DEPS = ["playground.py"]

REPO = Path(__file__).resolve().parents[2]

# ── envelope (the pilot's, deliberately: no bar is read at a new n) ────────
N_DECISIONS = 3000          # free-roam decisions per seed (600 s of life)
N_BURSTS = 270              # per seed -> 810 total (falsified_by names 800)
BURST_DECISIONS = 15        # 3 s per burst
ORACLE_DECISIONS = 100      # 20 s of real decisions per site
ORACLE_K = 8                # random candidates per decision (+ hold-previous)
ORACLE_HORIZON = 5          # 1 s lookahead per candidate
SUBSTEPS = 40               # 40 x 0.005 s = 0.2 s per decision (w0.py's)
MUTATE_STRENGTH = 0.15      # §2.2's documented per-seed world mutation

# ── pre-registered thresholds (2026-08-09 pilot; registry LT.01) ───────────
ENGAGED_RISE = 0.25         # m, an attempt becomes ENGAGED
PERSIST_S = 0.5             # s, clause (iii) persistence
LOAD_FRAC = 0.5             # x body weight, clause (iii) load-bearing
GAP_S = 3.0                 # s, attempt bracketing gap
NONLADDER_RISE_BAR = 0.6    # m, C2: raw height must be this gameable
HANG_BAND = (0.01, 0.05)    # C3: pooled P(hang) must land in 1-5%
SUCCESS_MARGIN = 0.15       # platform reach: z >= ladder_height - 0.15
SUCCESS_RUNGS = 3           # distinct rungs for a ladder-supported arrival
ORACLE_ALIVE_RISE = 0.25    # m, V4: the oracle must at least climb this
CALIB_TOL = 0.30            # V2: floor support within 30% of body weight

GRIP_OFF = 0.045 + 0.028    # hand radius + rung radius (playground.py)


class _Rig:
    """One built world + the h(t) instrument. Everything resolved by name
    against the live model (T0.14's lesson: no declared constant is trusted)."""

    def __init__(self, seed: int):
        sys.path.insert(0, str(REPO))
        import mujoco
        import playground as pg

        self.mujoco = mujoco
        self.pg = pg
        self.params = pg.PlaygroundParams().mutate(
            np.random.RandomState(seed), strength=MUTATE_STRENGTH)
        self.model, self.data, self.water = pg.make_playground(
            self.params, with_rover=True)
        self.ix = pg.rover_index(self.model)
        assert self.model.nu == pg.ROVER_NU

        names = {gid: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                 for gid in range(self.model.ngeom)}
        self.ladder = np.array([g for g, n in names.items() if n and (
            n.startswith("rung") or n.startswith("ladder_rail"))])
        self.rungs = np.array([g for g, n in names.items()
                               if n and n.startswith("rung")])
        self.climb = np.array([g for g, n in names.items()
                               if n in ("handL", "handR", "rover_foot")])
        self.body = np.array([g for g, n in names.items()
                              if n in ("rover_torso", "rover_foot", "handL", "handR")])
        self.ground = np.array([g for g, n in names.items() if n and (
            n == "floor" or n == "ramp" or n.startswith("stair")
            or n == "seesaw_plank" or n.startswith("poolwall")
            or n == "pool_floor" or (n.startswith("obj") and n != "")
            or n == "welded_block" or n == "platform")])
        self.gate_ground = np.array(sorted(self.ix["ground_geoms"]))
        self.torso_gid = self.ix["geom"]["rover_torso"]
        self.rover_bid = self.ix["body"]["rover"]
        self.weight = float(self.model.body_subtreemass[self.rover_bid]) * 9.81

        self.lo = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=float)
        self.hi = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=float)
        lx, ly = pg.LADDER_X, pg.LADDER_Y
        self.ladder_base = np.array([lx, ly])
        self.platform_xy = np.array([lx, ly + 0.45])
        self.panel_xy = np.array([0.0, self.params.arena_size - 0.1])
        self.platform_z = self.params.ladder_height - SUCCESS_MARGIN
        self.rung_zs = [(i + 1) * self.params.ladder_rung_spacing
                        for i in range(self.params.ladder_rungs)
                        if (i + 1) * self.params.ladder_rung_spacing
                        < self.params.ladder_height]

        # Measured at build time, per world: settle standing, read the torso.
        self.z_rest, self.calib = self._settle_and_calibrate()
        self.reset_meter()

    # ── build-time measurement ────────────────────────────────────────────
    def _settle_and_calibrate(self):
        """1 s of zero action, then z_rest and the contact-force convention.

        The sign question ("does mj_contactForce report the force on geom1 or
        geom2?") is answered by the world, not the docs: at rest the floor
        must push the body UP by ~one body weight. `sign=+1` means the
        rotated force vector acts on the contact's geom2."""
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        for _ in range(5 * SUBSTEPS):
            if self.water is not None:
                self.water.apply(self.model, self.data)
            self.mujoco.mj_step(self.model, self.data)
        z_rest = float(self.data.geom_xpos[self.torso_gid][2])

        floor_like = set(self.gate_ground.tolist())
        body = set(self.body.tolist())
        up = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if not ((g1 in floor_like and g2 in body)
                    or (g2 in floor_like and g1 in body)):
                continue
            f6 = np.zeros(6)
            self.mujoco.mj_contactForce(self.model, self.data, i, f6)
            world = c.frame.reshape(3, 3).T @ f6[:3]
            up += float(world[2]) if g2 in body else -float(world[2])
        # `up` under the geom2 convention; the true support is +weight.
        for sign in (1.0, -1.0):
            if abs(sign * up - self.weight) / self.weight <= CALIB_TOL:
                return z_rest, {"sign": sign, "support_n": sign * up, "ok": 1.0}
        return z_rest, {"sign": 1.0, "support_n": up, "ok": 0.0}

    # ── the meter ─────────────────────────────────────────────────────────
    def reset_meter(self):
        self.streak = 0.0
        self.h_max = 0.0
        self.rise_inst_max = -np.inf     # clauses (i)+(ii) only
        self.rise_persist_max = -np.inf  # + persistence, no load clause
        self.nonladder_z_max = -np.inf
        self.attempts = []               # finalized [H, n_rungs]
        self.cur = None                  # open attempt {H, rungs, last_t}
        self.t = 0.0
        self.platform_hits = 0
        self.platform_nonladder = 0

    def _pairs(self):
        n = self.data.ncon
        if n == 0:
            return np.zeros((0, 2), dtype=int)
        return np.array(self.data.contact.geom[:n], dtype=int)

    def _any_pair(self, pairs, a: np.ndarray, b: np.ndarray) -> bool:
        if pairs.shape[0] == 0:
            return False
        m = ((np.isin(pairs[:, 0], a) & np.isin(pairs[:, 1], b))
             | (np.isin(pairs[:, 1], a) & np.isin(pairs[:, 0], b)))
        return bool(m.any())

    def grounded(self, pairs=None) -> bool:
        pairs = self._pairs() if pairs is None else pairs
        return self._any_pair(pairs, self.body, self.gate_ground)

    def _ladder_force_z(self) -> float:
        """Net vertical CONSTRAINT force the ladder exerts on the body, N.

        The load-bearing clause gates on this value's MAGNITUDE, and the sign
        matters enough to record why: `mj_contactForce` reports the contact
        reaction only, never the adhesion actuator's pull. A hand RESTING on
        top of a rung with the body's weight on it reads ~+1.0 W (up). A hand
        GRIPPING a rung from below — PG.3's certified hang, the mechanism the
        whole Ladder Test exists to detect — reads ~-(grip + weight) DOWN,
        because the rung's reaction opposes the adhesion pulling the hand
        into it (measured on the scripted hang: -1556 N while hanging dead
        still). A signed >= +0.5 W gate would therefore reject certified
        climbing while accepting only rung-top mantles. While clauses (i) and
        (ii) hold — on the ladder, off the ground — for 0.5 s continuously,
        a net interface force of >= 0.5 W in either direction is only
        sustainable if the ladder is holding the body (nothing else is in
        contact to push against; free fall would break the contact within
        ~0.15 s), so the magnitude is the sign-invariant witness."""
        out = 0.0
        climb = set(self.climb.tolist())
        ladder = set(self.ladder.tolist())
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 in climb and g2 in ladder:
                body_is_2 = False
            elif g2 in climb and g1 in ladder:
                body_is_2 = True
            else:
                continue
            f6 = np.zeros(6)
            self.mujoco.mj_contactForce(self.model, self.data, i, f6)
            world = c.frame.reshape(3, 3).T @ f6[:3]
            out += self.calib["sign"] * (float(world[2]) if body_is_2
                                         else -float(world[2]))
        return out

    def substep(self, dt: float):
        """Advance the instrument one physics step (call after mj_step)."""
        self.t += dt
        pairs = self._pairs()
        lad = self._any_pair(pairs, self.climb, self.ladder)
        gnd = self._any_pair(pairs, self.body, self.ground)
        tz = float(self.data.geom_xpos[self.torso_gid][2])

        if lad:
            if self.cur is None or (self.t - self.cur["last_t"]) >= GAP_S:
                if self.cur is not None:
                    self.attempts.append(
                        [self.cur["H"], len(self.cur["rungs"])])
                self.cur = {"H": 0.0, "rungs": set(), "last_t": self.t}
            self.cur["last_t"] = self.t
            if pairs.shape[0]:
                m = (np.isin(pairs[:, 0], self.rungs)
                     & np.isin(pairs[:, 1], self.climb))
                m |= (np.isin(pairs[:, 1], self.rungs)
                      & np.isin(pairs[:, 0], self.climb))
                for r in pairs[m].ravel():
                    if r in self.rungs:
                        self.cur["rungs"].add(int(r))
        else:
            self.nonladder_z_max = max(self.nonladder_z_max, tz)

        airborne_grip = lad and not gnd
        self.streak = self.streak + dt if airborne_grip else 0.0
        rise = tz - self.z_rest
        if airborne_grip:
            self.rise_inst_max = max(self.rise_inst_max, rise)
            if self.streak >= PERSIST_S:
                self.rise_persist_max = max(self.rise_persist_max, rise)
                if abs(self._ladder_force_z()) >= LOAD_FRAC * self.weight:
                    h = rise
                    self.h_max = max(self.h_max, h)
                    if self.cur is not None:
                        self.cur["H"] = max(self.cur["H"], h)

        # Platform arrival, attributed to its bracketing attempt (or none).
        if tz >= self.platform_z:
            xy = self.data.geom_xpos[self.torso_gid][:2]
            if (abs(xy[0] - self.platform_xy[0]) <= 0.45
                    and abs(xy[1] - self.platform_xy[1]) <= 0.45):
                self.platform_hits += 1
                n_rungs = (len(self.cur["rungs"]) if self.cur is not None
                           and (self.t - self.cur["last_t"]) < GAP_S else 0)
                if n_rungs < SUCCESS_RUNGS:
                    self.platform_nonladder += 1

    def finalize(self):
        if self.cur is not None:
            self.attempts.append([self.cur["H"], len(self.cur["rungs"])])
            self.cur = None

    def engaged(self) -> int:
        self.finalize()
        return sum(1 for H, _ in self.attempts if H >= ENGAGED_RISE)

    # ── stepping ──────────────────────────────────────────────────────────
    def decide(self, action: np.ndarray, meter: bool = True):
        """W0's action mapping without W0's needs: 6 ctrl + gated drive."""
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        ctrl = self.lo + (a[:6] * 0.5 + 0.5) * (self.hi - self.lo)
        force = a[6:8] * self.pg.ROVER_DRIVE_FORCE
        dt = float(self.model.opt.timestep)
        for _ in range(SUBSTEPS):
            self.data.ctrl[:] = ctrl
            gate = self.grounded()
            self.data.xfrc_applied[self.rover_bid, :2] = force if gate else 0.0
            if self.water is not None:
                self.water.apply(self.model, self.data)
            self.mujoco.mj_step(self.model, self.data)
            if meter:
                self.substep(dt)
        self.data.xfrc_applied[self.rover_bid, :2] = 0.0

    def teleport(self, x: float, y: float, reset: bool = True):
        """Place the rover standing at (x, y), world otherwise reset."""
        if reset:
            self.mujoco.mj_resetData(self.model, self.data)
        q, d = self.ix["root_qposadr"], self.ix["root_dofadr"]
        self.data.qpos[q:q + 3] = (x, y, self.pg.ROVER_REST_Z + 0.01)
        self.data.qpos[q + 3:q + 7] = (1, 0, 0, 0)
        self.data.qvel[d:d + 6] = 0.0
        for n, adr in self.ix["jnt_qposadr"].items():
            self.data.qpos[adr] = 0.0
        for n, adr in self.ix["jnt_dofadr"].items():
            self.data.qvel[adr] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[self.rover_bid, :] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> dict:
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                "ctrl": self.data.ctrl.copy(), "time": float(self.data.time),
                "warm": self.data.qacc_warmstart.copy()}

    def restore(self, s: dict):
        self.data.qpos[:] = s["qpos"]; self.data.qvel[:] = s["qvel"]
        self.data.ctrl[:] = s["ctrl"]; self.data.qacc_warmstart[:] = s["warm"]
        self.data.time = s["time"]
        self.mujoco.mj_forward(self.model, self.data)

    def finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all()
                    and np.isfinite(self.data.qvel).all())


# ── phases ─────────────────────────────────────────────────────────────────
def _free_roam(rig: _Rig, rng: np.random.RandomState) -> dict:
    rig.teleport(*rig.params.spawn()[:2])
    rig.reset_meter()
    near_ladder = near_panel = rung_dec = 0
    for _ in range(N_DECISIONS):
        rig.decide(rng.uniform(-1.0, 1.0, size=8))
        xy = rig.data.geom_xpos[rig.torso_gid][:2]
        near_ladder += int(np.linalg.norm(xy - rig.ladder_base) <= 1.0)
        near_panel += int(np.linalg.norm(xy - rig.panel_xy) <= 2.0)
        rung_dec += int(rig.cur is not None
                        and (rig.t - rig.cur["last_t"]) < 0.2)
    engaged = rig.engaged()
    return {
        "null_engaged_attempts": engaged,
        "free_h_max": round(rig.h_max, 4),
        "nonladder_rise_max": round(rig.nonladder_z_max - rig.z_rest, 4),
        "nonladder_rise_ge_bar": float(
            rig.nonladder_z_max - rig.z_rest >= NONLADDER_RISE_BAR),
        "near_ladder_frac": round(near_ladder / N_DECISIONS, 4),
        "panel_dwell": round(near_panel / N_DECISIONS, 4),
        "rung_contact_frac": round(rung_dec / N_DECISIONS, 4),
        "platform_nonladder": rig.platform_nonladder,
        "free_finite": float(rig.finite()),
    }


def _bursts(rig: _Rig, rng: np.random.RandomState) -> dict:
    """From the ladder base, hands at the rung plane: 3 s random bursts."""
    n_inst = n_persist = n_load = 0
    best = -np.inf
    for _ in range(N_BURSTS):
        rig.teleport(rig.pg.LADDER_X, rig.pg.LADDER_Y + 0.05)
        rig.reset_meter()
        for _ in range(BURST_DECISIONS):
            rig.decide(rng.uniform(-1.0, 1.0, size=8))
        n_inst += int(rig.rise_inst_max >= ENGAGED_RISE)
        n_persist += int(rig.rise_persist_max >= ENGAGED_RISE)
        n_load += int(rig.h_max >= ENGAGED_RISE)
        best = max(best, rig.h_max)
    ok = rig.finite()
    return {
        "p_hang_inst": round(n_inst / N_BURSTS, 4),
        "p_hang_persist": round(n_persist / N_BURSTS, 4),
        "p_hang": round(n_load / N_BURSTS, 4),
        "burst_rise_ceiling": round(best, 4),
        "burst_finite": float(ok),
    }


def _scripted_hang(rig: _Rig) -> dict:
    """V3: PG.3's certified physics driven through THIS instrument.

    Grip a rung near 1.0 m with both hands, adhesion on, pull the lift slides
    to their floor. The body must leave the ground and hang load-bearing long
    enough that the full h(t) conjunction — and the ENGAGED counter — fire.
    If they do not, every zero this spec reports is unattributable."""
    target = min((z for z in rig.rung_zs if 0.85 <= z <= 1.25),
                 key=lambda z: abs(z - 1.0), default=None)
    if target is None:                      # mutation left no graspable rung
        return {"hang_check_engaged": 0.0, "hang_check_h": 0.0,
                "hang_finite": 0.0}
    rig.teleport(rig.pg.LADDER_X, rig.pg.LADDER_Y)
    rig.reset_meter()
    origin_z = float(rig.data.qpos[rig.ix["root_qposadr"] + 2])
    lift = float(np.clip(target - GRIP_OFF + 0.005 - origin_z, -0.2, 0.55))

    def act(reach, lift_v, adh):
        # inverse of decide()'s mapping: ctrl -> action in [-1, 1]
        a = np.zeros(8)
        spans = [("reachL", reach), ("liftL", lift_v),
                 ("reachR", reach), ("liftR", lift_v)]
        for k, (name, v) in enumerate(spans):
            i = rig.ix["act"][name]
            a[k] = 2.0 * (v - rig.lo[i]) / (rig.hi[i] - rig.lo[i]) - 1.0
        a[4] = a[5] = adh
        return np.clip(a, -1.0, 1.0)

    for _ in range(5):                       # reach the rung plane, hands up
        rig.decide(act(0.0, lift, -1.0))
    for _ in range(5):                       # grip
        rig.decide(act(0.0, lift, 1.0))
    steps = 10
    for s in range(steps):                   # pull to the slide floor
        f = (s + 1) / steps
        rig.decide(act(0.0, lift + f * (-0.2 - lift), 1.0))
    for _ in range(10):                      # hold the hang for 2 s
        rig.decide(act(0.0, -0.2, 1.0))
    return {"hang_check_engaged": float(rig.engaged() >= 1),
            "hang_check_h": round(rig.h_max, 4),
            "hang_finite": float(rig.finite())}


def _oracle(rig: _Rig, rng: np.random.RandomState) -> dict:
    """The control: greedy height maximisation, adhesion DISABLED."""
    a_size = rig.params.arena_size
    sites = [
        (rig.pg.LADDER_X, rig.pg.LADDER_Y + 0.05),        # the ladder, gripless
        (1.7, 2.2),                                        # stairs base
        (-3.4, 2.0),                                       # ramp base
        (-2.0, -0.6),                                      # seesaw / objects
    ]
    best_rise = -np.inf
    platform = 0
    finite = 1.0
    for sx, sy in sites:
        sx = float(np.clip(sx, -a_size + 0.4, a_size - 0.4))
        sy = float(np.clip(sy, -a_size + 0.4, a_size - 0.4))
        rig.teleport(sx, sy)
        rig.reset_meter()
        prev = np.zeros(8)
        for _ in range(ORACLE_DECISIONS):
            snap = rig.snapshot()
            best_a, best_s = None, -np.inf
            cands = [prev] + [rng.uniform(-1.0, 1.0, size=8)
                              for _ in range(ORACLE_K)]
            for a in cands:
                a = a.copy()
                a[4] = a[5] = -1.0                        # adhesion OFF
                rig.restore(snap)
                score = -np.inf
                for _ in range(ORACLE_HORIZON):
                    rig.decide(a, meter=False)
                    score = max(score,
                                float(rig.data.geom_xpos[rig.torso_gid][2]))
                if score > best_s:
                    best_s, best_a = score, a
            rig.restore(snap)
            rig.decide(best_a)
            prev = best_a
        best_rise = max(best_rise,
                        float(rig.nonladder_z_max - rig.z_rest),
                        float(rig.h_max))
        platform += rig.platform_hits
        finite = min(finite, float(rig.finite()))
    return {"oracle_rise_max": round(best_rise, 4),
            "oracle_alive": float(best_rise >= ORACLE_ALIVE_RISE),
            "oracle_platform": platform,
            "oracle_finite": finite}


# ── the spec's three callables ─────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    rig = _Rig(seed)
    rng = np.random.RandomState(10_000 + seed)
    out = {"z_rest": round(rig.z_rest, 4), "body_weight_n": round(rig.weight, 2),
           "force_calib_ok": rig.calib["ok"],
           "floor_support_n": round(rig.calib["support_n"], 2),
           "ladder_height": rig.params.ladder_height,
           "rung_spacing": round(rig.params.ladder_rung_spacing, 4)}
    out.update(_free_roam(rig, rng))
    out.update(_bursts(rig, rng))
    out.update(_scripted_hang(rig))
    return out


def _control(seed: int) -> dict:
    rig = _Rig(seed)
    out = {"force_calib_ok": rig.calib["ok"]}
    out.update(_oracle(rig, np.random.RandomState(20_000 + seed)))
    return out


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # V1 — physics finite in every phase, every seed
    if (m.get("free_finite", 0.0) != 1.0 or m.get("burst_finite", 0.0) != 1.0
            or m.get("hang_finite", 0.0) != 1.0
            or c.get("oracle_finite", 0.0) != 1.0):
        return _void(m, "non-finite physics")
    # V2 — the force instrument reproduced body weight at rest
    if m.get("force_calib_ok", 0.0) != 1.0 or c.get("force_calib_ok", 0.0) != 1.0:
        return _void(m, "force sign uncalibrated: rest support != body weight")
    # V3 — the zero's instrument fired on a scripted hang
    if m.get("hang_check_engaged", 0.0) != 1.0:
        return _void(m, "h(t) never fired on a scripted hang; the free-roam "
                        "zero is a dead instrument, not a null floor")
    # V4 — the oracle moved uphill in every seed
    if c.get("oracle_alive", 0.0) != 1.0:
        return _void(m, "oracle never climbed; absence of an alternate route "
                        "is unmeasured")
    # THE CLAIM — four clauses, first falsified branch named
    if m["null_engaged_attempts"] != 0.0:
        m["claim_branch"] = "null floor is not zero: random action produced " \
                            "engaged attempts"
        return False
    if m["nonladder_rise_ge_bar"] <= 0.0:
        m["claim_branch"] = "no non-ladder route reached 0.6 m rise: raw " \
                            "torso z is less gameable than the pilot measured"
        return False
    if m["p_hang"] <= 0.0:
        m["claim_branch"] = "no bootstrap: P(hang) = 0 across all bursts"
        return False
    if not (HANG_BAND[0] <= m["p_hang"] <= HANG_BAND[1]):
        m["claim_branch"] = "hang rate outside the 1-5% band"
        return False
    if m["platform_nonladder"] != 0.0 or c["oracle_platform"] != 0.0:
        m["claim_branch"] = "alternate route reached the platform"
        return False
    m["claim_branch"] = "null floor zero, raw z gameable, bootstrap in band, " \
                        "no alternate route"
    return True


def run(ledger: Ledger | None = None):
    import os
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))
    return run_spec(BY_ID["LT.01"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


def _pilot():
    """Seed 90, mechanics and runtime only (disjoint from recorded seeds 0-2;
    W0.DIAG's pilot idiom). Prints JSON, records NOTHING, reads no gate —
    every claim bar was frozen 2026-08-09."""
    import json
    import time as _time

    rig = _Rig(90)
    rng = np.random.RandomState(7)
    out = {"z_rest": rig.z_rest, "weight": rig.weight, "calib": rig.calib,
           "rungs": rig.rung_zs}

    t0 = _time.time()
    out["hang"] = _scripted_hang(rig)
    out["t_hang_s"] = round(_time.time() - t0, 1)

    t0 = _time.time()
    global N_DECISIONS, N_BURSTS, ORACLE_DECISIONS
    full_nd, full_nb, full_od = N_DECISIONS, N_BURSTS, ORACLE_DECISIONS
    N_DECISIONS, N_BURSTS, ORACLE_DECISIONS = 500, 60, 30
    try:
        out["free"] = _free_roam(rig, rng)
        out["t_free_s"] = round(_time.time() - t0, 1)
        t0 = _time.time()
        out["bursts"] = _bursts(rig, rng)
        out["t_bursts_s"] = round(_time.time() - t0, 1)
        t0 = _time.time()
        out["oracle"] = _oracle(rig, rng)
        out["t_oracle_s"] = round(_time.time() - t0, 1)
    finally:
        N_DECISIONS, N_BURSTS, ORACLE_DECISIONS = full_nd, full_nb, full_od
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _pilot()
