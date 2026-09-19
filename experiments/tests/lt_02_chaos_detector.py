"""LT.02 — The self-generated-chaos detector works (PG.4's blind spot).

PG.4 certified that the noise-panel trap catches a naive prediction-error
agent and that the DWELL metric measures the fixation. But dwell is a
POSITIONAL metric: it can only see a trap that has an address. The classic
failure it cannot see is the agent whose noisy TV is its OWN BODY — ragdoll
contact dynamics are an inexhaustible source of irreducible prediction error
that travels with the agent, so a curiosity core farming it reads dwell 0.000
and every panel-based certificate calls it clean. Before any spec claims
"his curiosity is not trapped" (CU.3 and the LT.03+ ladder arms), the
instrument that CAN see body-chaos farming must be shown to work: flag a
deliberate farmer, pass a deliberate hard-working climber, and agree with
PG.4's dwell metric on the one agent this project has already PROVEN trapped.

THE DETECTOR (registry: "pooled-fit forward model, out-of-fold, high error
AND no reducibility when the training data doubles — LPM criterion,
arXiv:2509.25438, used as a diagnostic not a reward"):

  Per seed, per rig, pool the (obs, action, next_obs) transitions of every
  arm that lived in that world. K-fold split; per fold train an MLP forward
  model on the training folds; every transition's error e is computed
  OUT-OF-FOLD (by the model that never saw it). theta = the 90th percentile
  of e over the RANDOM arm's transitions — the random policy DEFINES the
  ruler, so by construction:

      chaos_occupancy(arm) = frac(e_arm > theta) / 0.10
      chaos_occupancy(random) = 1.0 exactly (up to ties, gated by V4)

  The LPM data-doubling criterion is carried as a REPORTED DIAGNOSTIC beside
  the gate, never inside it (the PS.08/PS.09 family lesson: carry both
  estimators and let divergence localise the defect): each fold also trains
  a half-data twin, and `lpm_occupancy` counts transitions that are high
  error AND unimproved by doubling (reduction < RHO x e_half), while
  `reducibility_gain` reports how much doubling helped on each arm's
  high-error transitions. If the gate and the diagnostic disagree, the
  divergence is the finding — exactly the adjudication PS.09's bare
  C-threshold provided. `det_beats_persistence` (V2) is the T2.05 lesson:
  an out-of-fold model that cannot beat "next = current" on the median
  pooled transition is a dead instrument and its "high error" means nothing.

THE FOUR ARMS:

  ragdoll-ICM   (experiment) The LT.01 rover body in a per-seed mutated
                playground with the noise panel DELETED (noise_panel=False —
                a first-class world param, not surgery) and adhesion FORCED
                OFF in its action set: the only irreducible surprise
                available anywhere is its own contact dynamics. The agent is
                PG.4's textbook construction transplanted: tabular Q over
                (floor cell x upright) with optimistic init, eps-greedy over
                a fixed set of 16 prototype actions + stand, intrinsic
                reward = online forward-model prediction error, full-history
                replay. Nothing anywhere rewards thrashing; only surprise.
  random        (experiment) LT.01's certified null, uniform [-1,1]^8 —
                defines theta. Also the denominator of thrash_ratio, the
                model-free second signal (mean per-decision |d qvel| of the
                6 non-root dofs + root angular velocity).
  scripted      (experiment) A deterministic hard-working climber: reach the
  climber       ~1.0 m rung, grip, pull to the slide floor, hold, release
                and FALL, recover, repeat for the whole life. It moves hard
                and falls repeatedly — the behaviour GOAL.md asks for — and
                the detector must NOT flag it (<= 1.0).
  slider-ICM    (control) PG.4's KNOWN POSITIVE, reproduced faithfully: same
  + panel       builder, same retina, same hyperparameters, same per-seed
                rng derivations (env 7919k+13, agent 104729k+7, torch seed),
                same 20k-decision life — imported from pg_4_noisy_tv.py, not
                copied, and declared in IMPL_DEPS so drift stales this
                certificate. Two independent detectors must agree on it:
                panel_dwell > 0.4 (PG.4's own metric, re-measured live) AND
                chaos_occupancy >= 3.0 (the new instrument), with theta from
                a slider-random arm in the same rig. Known caveat, disclosed
                where it is load-bearing: PG.4's per-seed dwell vector is
                (1.0, 1.0, 0.0) — seed 2 never finds the panel — so the
                control conjunct is honestly a MEAN-aggregate bar (~0.67 vs
                0.4), the ladder's uniform protocol, and the per-seed spread
                is recoverable from the recorded std (the PG.4 disclosure
                precedent).

PRE-REGISTERED BARS (registry, 2026-08-09 — none moved here):
  C1  chaos_occupancy(ragdoll-ICM)  >= 3.0
  C2  chaos_reward_ratio(ragdoll-ICM) >= 2.0   (late-half intrinsic reward
      on detector-flagged transitions vs unflagged: the fixation is FED by
      the chaos, not by floor-plan bias — PG.4's cause gate, one rig over)
  C3  panel_dwell(ragdoll-ICM) == 0.000        (the notes' assertion: the
      farming happens where the dwell metric reads nothing; positional
      dwell is metered against the panel's LOCATION even though the geom is
      deleted, and a mean of a non-negative reading is 0 iff every seed is)
  C4  chaos_occupancy(scripted climber) <= 1.0
  C5  control flagged by BOTH detectors: panel_dwell > 0.4 AND
      chaos_occupancy >= 3.0
  metric: chaos_detector_separation = occupancy(ragdoll-ICM) - occupancy(
      scripted climber), recorded per seed.

VOID LANES (rig faults are not world readings — T0.22; frozen with the
pilot, before any registered seed ran):
  V1  non-finite physics in any arm, either rig
  V2  the pooled OOF model fails to beat persistence on the median pooled
      transition, either rig ("high error" from a dead model is noise)
  V3  the scripted climber never exercised: no graspable rung after world
      mutation, or climber_rise_max < 0.25 m, or fewer than 5 completed
      release-falls — a statue's low occupancy would be vacuous (the T3.09
      site-under-exercise lane)
  V4  the ruler is degenerate: theta <= 0 or the random arm's own occupancy
      is off 1.0 by more than 0.1 (massive ties), either rig

WHAT A FAIL MEANS: C1 failing is the registry's "the detector is blind" —
no arm's immunity may be reported and the LT.03+ chain stays honest. C4
failing is "the detector penalises coordinated motion and falling". C5
failing is the two instruments disagreeing on the project's one known
positive. Each names its branch in `claim_branch`.

PILOT RECORD (seed 90, disjoint from recorded seeds 0-2; mechanics, gate
aliveness and runtime only — every claim bar above is the registry's,
frozen 2026-08-09; 2026-09-19, this commit; reduced envelope RAG_DEC 1200 /
slider 6000, detector at final constants):

  The pilot took three draws and each fixed a measured rig fault; every fix
  is recorded here because the final rig is only trustworthy if the path to
  it is:
  (1) FIRST DRAW: the climber was a statue (rise 0.0104 m, 0 falls) — it
      teleported to the generic spawn point, ~0.8 m behind the ladder, where
      the reach slides cannot touch a rung. Fixed: per-cycle teleport to the
      base (LT.01's per-burst fixture right). And the ICM agent's Q-state
      (cell x upright) could not see its own arms, so greedy collapsed to
      one held position target (thrash 0.59 vs random 1.54). Fixed: two
      lift-joint bins added to the state — the textbook agent unchanged,
      given a state that can EXPRESS body-relevant policies.
  (2) SECOND DRAW: climber alive (30 falls, rise 1.1417 m, occupancy 0.325)
      but the ICM still under-farmed (0.0417). A scratch ceiling probe — a
      deliberate max-amplitude slammer, NOT part of the registered design —
      read occupancy 0.258 with reducibility_gain 0.50: ANY repetitive
      policy concentrates data where the pooled model then masters it. The
      candidate irreducibility left in the world was the 5 unobserved
      clutter objects (hidden pose = epistemic noise a farmer could milk
      while the claim credits its body), so the rig now sets n_objects=0 —
      PG.4's own isolation precedent, quoted at the constructor.
  (3) FINAL DRAW, the frozen rig. Ragdoll (panel deleted, no clutter,
      mutated at 0.15): z_rest 0.3896 m, finite everywhere; detector alive
      (OOF median 0.250 vs persistence 2.146, V2 green); theta 1.546 > 0,
      random occupancy 1.0000 exactly (V4 green); climber rung 1.14 m rise,
      30 falls (V3 green), occupancy 0.325 <= 1.0; ragdoll-ICM occupancy
      0.25, dwell 0.0000, reward_ratio 9.42, thrash 0.96 vs random 1.39,
      e_mean icm 0.327 / climber 0.293 / random 0.793. Slider control (PG.4
      exact rng derivations): occupancy 4.28 >= 3.0 with reducibility_gain
      0.0134 (the panel is GENUINELY irreducible — doubling data does not
      help), dwell 0.358 at the reduced envelope (eps still decaying; the
      registered 20k envelope is PG.4's own, where seeds 0/1 read 1.0).

  DISCLOSED FORECAST, so nobody mistakes the run for a lottery: at the
  reduced envelope the pilot's own readings sit on the FAIL side of C1
  (0.25 vs 3.0), and the slammer probe suggests a mechanism — on this body,
  at this obs resolution, self-generated contact "chaos" is largely
  REDUCIBLE (random's reducibility_gain 0.48 vs the panel's 0.013): a
  policy that farms a spot hands the model the data to master it, so
  self-surprise is self-extinguishing unless the source is genuinely
  stochastic. If the registered run confirms that, the FAIL is a real
  venue/claim measurement (no arm's immunity may be reported, exactly as
  falsified_by says), localised by e_mean / thrash_ratio / reducibility_
  gain, with the detector's own aliveness proven by the climber and the
  known-positive control. The bars are the registry's and none moved after
  any pilot number was read; the fixes above are mechanics, not thresholds.

  PS.09's shortcut hazard, checked by construction: the ragdoll rig's rows
  are not near-duplicates sharing a label (every transition is its own
  physics outcome; no per-trip label exists), and theta is a quantile of
  the null's own errors, never a fitted margin.

  Runtime, final pilot: ragdoll 3 lives 33.5 s + detector 1.3 s; slider
  2 lives 28.1 s + detector 4.6 s; total 68 s. Full-envelope projection:
  ~2.1 min/seed experiment + ~1.9 min/seed control => ~12-15 min for
  3 seeds. Cost class re-declared cpu<10min on this measurement (104th
  audit item 2, SO.08's SIZING RECORD precedent — the registry notes
  carry the record); the struck cpu<2h class enumerated at 54,000 s and
  refused the registered run by 292 s on 2026-09-19.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .pg_4_noisy_tv import (_ACTIONS, _build, _cell, _dwell, _Retina,
                            EPS_HI, EPS_LO, GAMMA, Q_INIT, Q_LR, SPEED)
from .pg_4_noisy_tv import N_DECISIONS as SLIDER_DECISIONS
from .pg_4_noisy_tv import SUBSTEPS as SLIDER_SUBSTEPS

# The claim is about the world's body physics AND about PG.4's known-positive
# construction (the control reproduces it); both hash into impl_sha.
IMPL_DEPS = ["playground.py", "experiments/tests/pg_4_noisy_tv.py"]

REPO = Path(__file__).resolve().parents[2]

# ── envelope (frozen with the pilot; bars are the registry's) ──────────────
RAG_DECISIONS = 4000        # ragdoll life: 800 s at 0.2 s/decision
RAG_SUBSTEPS = 40           # w0.py's decision quantum (LT.01's)
MUTATE_STRENGTH = 0.15      # per-seed world mutation, LT.01's documented value
N_PROTO = 16                # ICM prototype actions (+ stand) — adhesion OFF
DWELL_RADIUS = 2.0          # m, PG.4's dwell zone, metered against the
                            # panel's LOCATION (geom deleted in the ragdoll rig)

# ── detector (frozen with the pilot) ───────────────────────────────────────
DET_FOLDS = 3
DET_EPOCHS = 15
DET_BATCH = 256
DET_LR = 1e-3
DET_HID = 64
DET_MAX_PER_ARM = 8000      # seeded subsample bound (slider lives are 20k)
THETA_Q = 0.90              # the ruler: random arm's 90th percentile
RHO = 0.25                  # LPM diagnostic: "doubling helped" means the
                            # half->full error drop exceeds RHO x e_half

# ── pre-registered bars (registry LT.02, 2026-08-09 — restated, not owned) ─
OCC_ICM_MIN = 3.0
OCC_CLIMBER_MAX = 1.0
REWARD_RATIO_MIN = 2.0
CONTROL_DWELL_MIN = 0.4
CONTROL_OCC_MIN = 3.0

# ── V3/V4 rig-gate floors (frozen with the pilot) ──────────────────────────
CLIMBER_RISE_MIN = 0.25     # m — LT.01's ENGAGED_RISE: it genuinely climbed
CLIMBER_FALLS_MIN = 5       # completed release-falls per life
RULER_TOL = 0.10            # |occupancy(random) - 1.0| beyond this is ties


# ═══════════════════════════════ ragdoll rig ═══════════════════════════════
class _RagRig:
    """LT.01's body in a mutated playground with the noise panel DELETED.

    Deliberately slimmer than LT.01's _Rig: no h(t) meter, no attempt
    bracketing — this spec reads transitions, not climbing claims. Everything
    is resolved by name against the live model (T0.14's lesson)."""

    JOINTS = ("reachL", "liftL", "reachR", "liftR")

    def __init__(self, seed: int):
        sys.path.insert(0, str(REPO))
        import mujoco
        import playground as pg

        self.mujoco = mujoco
        self.pg = pg
        self.params = pg.PlaygroundParams(noise_panel=False).mutate(
            np.random.RandomState(seed), strength=MUTATE_STRENGTH)
        # Source isolation, PG.4's own precedent ("free-floating clutter would
        # add its own dynamics noise; the trap claim wants the panel to be the
        # sole stochastic source"): with proprio-only obs, an unobserved
        # object's pose is epistemic noise that would let an agent farm the
        # WORLD while the claim credits its BODY. The body must be the sole
        # candidate source.
        self.params.n_objects = 0
        assert self.params.noise_panel is False and self.params.n_objects == 0
        self.model, self.data, self.water = pg.make_playground(
            self.params, with_rover=True)
        self.ix = pg.rover_index(self.model)
        assert self.model.nu == pg.ROVER_NU

        self.torso_gid = self.ix["geom"]["rover_torso"]
        self.rover_bid = self.ix["body"]["rover"]
        self.lo = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=float)
        self.hi = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=float)
        self.panel_xy = np.array([0.0, self.params.arena_size - 0.1])
        self.rung_zs = [(i + 1) * self.params.ladder_rung_spacing
                        for i in range(self.params.ladder_rungs)
                        if (i + 1) * self.params.ladder_rung_spacing
                        < self.params.ladder_height]
        self.q = self.ix["root_qposadr"]
        self.d = self.ix["root_dofadr"]
        self.jq = [self.ix["jnt_qposadr"][n] for n in self.JOINTS]
        self.jd = [self.ix["jnt_dofadr"][n] for n in self.JOINTS]
        self.jlo = np.array([self.model.jnt_range[self.ix["jnt"][n]][0]
                             for n in self.JOINTS])
        self.jhi = np.array([self.model.jnt_range[self.ix["jnt"][n]][1]
                             for n in self.JOINTS])
        self._body_geoms = np.array([self.ix["geom"][g] for g in
                                     ("rover_torso", "rover_foot",
                                      "handL", "handR")])
        self._gate_ground = np.array(sorted(self.ix["ground_geoms"]))
        self.z_rest = self._settle()

    def _settle(self) -> float:
        self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        for _ in range(5 * RAG_SUBSTEPS):
            if self.water is not None:
                self.water.apply(self.model, self.data)
            self.mujoco.mj_step(self.model, self.data)
        return float(self.data.geom_xpos[self.torso_gid][2])

    def grounded(self) -> bool:
        n = self.data.ncon
        if n == 0:
            return False
        pairs = np.array(self.data.contact.geom[:n], dtype=int)
        body, gnd = self._body_geoms, self._gate_ground
        m = ((np.isin(pairs[:, 0], body) & np.isin(pairs[:, 1], gnd))
             | (np.isin(pairs[:, 1], body) & np.isin(pairs[:, 0], gnd)))
        return bool(m.any())

    def decide(self, action: np.ndarray):
        """W0's mapping, LT.01's decide(): 6 ctrl + grounded-gated drive."""
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        ctrl = self.lo + (a[:6] * 0.5 + 0.5) * (self.hi - self.lo)
        force = a[6:8] * self.pg.ROVER_DRIVE_FORCE
        for _ in range(RAG_SUBSTEPS):
            self.data.ctrl[:] = ctrl
            gate = self.grounded()
            self.data.xfrc_applied[self.rover_bid, :2] = force if gate else 0.0
            if self.water is not None:
                self.water.apply(self.model, self.data)
            self.mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[self.rover_bid, :2] = 0.0

    def teleport(self, x: float, y: float):
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.q:self.q + 3] = (x, y, self.pg.ROVER_REST_Z + 0.01)
        self.data.qpos[self.q + 3:self.q + 7] = (1, 0, 0, 0)
        self.data.qvel[self.d:self.d + 6] = 0.0
        for adr in self.jq:
            self.data.qpos[adr] = 0.0
        for adr in self.jd:
            self.data.qvel[adr] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[self.rover_bid, :] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def obs(self) -> np.ndarray:
        """20 dims, all ~[-1, 1]: root pose, velocities, 4 joints."""
        a = self.params.arena_size
        pos = self.data.qpos[self.q:self.q + 3]
        # torso up-vector: third column of the root body's rotation matrix
        R = self.data.xmat[self.rover_bid].reshape(3, 3)
        up = R[:, 2]
        lin = self.data.qvel[self.d:self.d + 3]
        ang = self.data.qvel[self.d + 3:self.d + 6]
        jq = (2.0 * (self.data.qpos[self.jq] - self.jlo)
              / (self.jhi - self.jlo) - 1.0)
        jv = self.data.qvel[self.jd]
        return np.concatenate([
            [pos[0] / a, pos[1] / a, pos[2]], up,
            np.clip(lin / 3.0, -3, 3), np.clip(ang / 10.0, -3, 3),
            jq, np.clip(jv / 3.0, -3, 3)]).astype(np.float32)

    def joint_speed(self) -> float:
        """|qvel| over the 4 arm dofs + root angular — thrash's ingredient."""
        return float(np.linalg.norm(
            np.concatenate([self.data.qvel[self.jd],
                            self.data.qvel[self.d + 3:self.d + 6]])))

    def upright(self) -> bool:
        return float(self.data.geom_xpos[self.torso_gid][2]) > 0.7 * self.z_rest

    def finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all()
                    and np.isfinite(self.data.qvel).all())


def _rag_cell(rig: _RagRig) -> int:
    a = rig.params.arena_size
    x, y = rig.data.qpos[rig.q], rig.data.qpos[rig.q + 1]
    cx = min(10, max(0, int((x + a) / (2 * a) * 11)))
    cy = min(10, max(0, int((y + a) / (2 * a) * 11)))
    return cy * 11 + cx


def _rag_state(rig: _RagRig) -> int:
    """Q-state: floor cell x upright x 2 lift-joint bins. The joint bins are
    what lets a tabular policy EXPRESS body-relevant behaviour (slam the arms
    top-to-bottom): with cell x upright alone, greedy collapses to one held
    position target and the arms settle — measured on the first pilot draw
    (icm thrash 0.59 vs random 1.54)."""
    liftL = int(rig.data.qpos[rig.jq[1]] > 0.175)     # mid-range of -0.2..0.55
    liftR = int(rig.data.qpos[rig.jq[3]] > 0.175)
    return ((_rag_cell(rig) * 2 + int(rig.upright())) * 2 + liftL) * 2 + liftR


def _rag_life(rig: _RagRig, policy: str, seed: int,
              n_dec: int = None) -> dict:
    """One unbroken life. Returns transitions + trajectory statistics.

    rng streams are disjoint by construction: icm 40_000+seed,
    random 10_000+seed (LT.01's null stream family), climber none."""
    n_dec = RAG_DECISIONS if n_dec is None else n_dec
    sx, sy, _ = rig.params.spawn()
    rig.teleport(sx, sy)

    obs_list, act_list, nxt_list, rew_list = [], [], [], []
    dwell_late = 0
    half = n_dec // 2
    thrash_sum, prev_speed = 0.0, None
    visited = set()

    fwd = opt = q = None
    protos = None
    if policy == "icm":
        import torch
        torch.manual_seed(seed)
        rng = np.random.RandomState(40_000 + seed)
        protos = rng.uniform(-1.0, 1.0, size=(N_PROTO, 8))
        protos[:, 4:6] = -1.0                       # adhesion forced OFF
        stand = np.zeros(8); stand[4:6] = -1.0
        protos = np.vstack([stand, protos])         # 17 actions
        obs_dim = 20
        fwd = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + 8, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, obs_dim))
        opt = torch.optim.Adam(fwd.parameters(), lr=1e-3)
        buf_x, buf_y = [], []
        q = np.full((968, len(protos)), Q_INIT)
    elif policy == "random":
        rng = np.random.RandomState(10_000 + seed)
    else:                                            # scripted climber
        rng = None

    # climber machinery (LT.01's _scripted_hang action inversion)
    climber_falls = 0
    climber_rise_max = 0.0
    rung_ok = 1.0
    if policy == "climber":
        target = min((z for z in rig.rung_zs if 0.85 <= z <= 1.25),
                     key=lambda z: abs(z - 1.0), default=None)
        if target is None:
            rung_ok = 0.0
            lift = 0.55
        else:
            grip_off = 0.045 + 0.028
            origin_z = float(rig.data.qpos[rig.q + 2])
            lift = float(np.clip(target - grip_off + 0.005 - origin_z,
                                 -0.2, 0.55))

        def climb_act(lift_v, adh):
            a = np.zeros(8)
            spans = [("reachL", 0.0), ("liftL", lift_v),
                     ("reachR", 0.0), ("liftR", lift_v)]
            for k, (name, v) in enumerate(spans):
                i = rig.ix["act"][name]
                a[k] = 2.0 * (v - rig.lo[i]) / (rig.hi[i] - rig.lo[i]) - 1.0
            a[4] = a[5] = adh
            return np.clip(a, -1.0, 1.0)

        # 40-decision cycle: reach 5, grip 5, pull 10, hold 3, release+
        # thrash 7, recover 10.  "Moves hard and falls repeatedly."
        cycle = ([climb_act(lift, -1.0)] * 5 + [climb_act(lift, 1.0)] * 5
                 + [climb_act(lift + (f + 1) / 10.0 * (-0.2 - lift), 1.0)
                    for f in range(10)]
                 + [climb_act(-0.2, 1.0)] * 3
                 + [climb_act(0.55 if k % 2 == 0 else -0.2, -1.0)
                    for k in range(7)]
                 + [climb_act(0.0, -1.0)] * 10)
        # release-fall accounting: at each cycle's release decision (index
        # 23), if the torso had genuinely risen, the coming drop is a fall.
        rel_idx = 23

    upright_dec = 0
    torso_rise = lambda: float(
        rig.data.geom_xpos[rig.torso_gid][2]) - rig.z_rest

    for t in range(n_dec):
        # The climber restarts each 40-decision cycle AT THE BASE — a fixture
        # right (LT.01's per-burst teleports), because the body has no walking
        # controller to return with. The teleport happens BEFORE this
        # iteration's obs read, so no logged (o, a, o2) triple spans it.
        if policy == "climber" and t % 40 == 0:
            rig.teleport(rig.pg.LADDER_X, rig.pg.LADDER_Y)
        o = rig.obs()
        if policy == "random":
            a = rng.uniform(-1.0, 1.0, size=8)
        elif policy == "icm":
            s = _rag_state(rig)
            eps = max(EPS_LO, EPS_HI - (EPS_HI - EPS_LO) * t / (n_dec // 3))
            if rng.uniform() < eps:
                ai = int(rng.randint(len(protos)))
            else:
                best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
                ai = int(best[rng.randint(len(best))])
            a = protos[ai]
        else:
            k = t % 40
            if k == rel_idx and torso_rise() >= 0.15:
                climber_falls += 1
            a = cycle[k]
            climber_rise_max = max(climber_rise_max, torso_rise())

        rig.decide(a)
        o2 = rig.obs()

        if policy == "icm":
            import torch
            with torch.no_grad():
                inp = torch.cat([torch.from_numpy(o),
                                 torch.from_numpy(a.astype(np.float32))])
                r = float(((fwd(inp) - torch.from_numpy(o2)) ** 2).sum())
            s2 = _rag_state(rig)
            q[s, ai] += Q_LR * (r + GAMMA * q[s2].max() - q[s, ai])
            buf_x.append(inp.numpy()); buf_y.append(o2)
            idx = rng.randint(len(buf_x), size=min(64, len(buf_x)))
            bx = torch.from_numpy(np.stack([buf_x[i] for i in idx]))
            by = torch.from_numpy(np.stack([buf_y[i] for i in idx]))
            loss = ((fwd(bx) - by) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            rew_list.append(r)

        obs_list.append(o); act_list.append(a.astype(np.float32))
        nxt_list.append(o2)
        sp = rig.joint_speed()
        if prev_speed is not None:
            thrash_sum += abs(sp - prev_speed)
        prev_speed = sp
        visited.add(_rag_cell(rig))
        upright_dec += int(rig.upright())
        xy = rig.data.qpos[rig.q:rig.q + 2]
        if (t >= half and math.hypot(xy[0] - rig.panel_xy[0],
                                     xy[1] - rig.panel_xy[1]) < DWELL_RADIUS):
            dwell_late += 1

    out = {
        "X": np.concatenate([np.stack(obs_list), np.stack(act_list)], axis=1),
        "Y": np.stack(nxt_list),
        "rewards": np.array(rew_list) if rew_list else None,
        "panel_dwell": round(dwell_late / half, 4),
        "thrash": thrash_sum / max(1, n_dec - 1),
        "visited_frac": round(len(visited) / 121, 4),
        "upright_frac": round(upright_dec / n_dec, 4),
        "finite": float(rig.finite()),
    }
    if policy == "climber":
        out.update({"climber_falls": float(climber_falls),
                    "climber_rise_max": round(climber_rise_max, 4),
                    "climber_rung_ok": rung_ok})
    return out


# ═══════════════════════════════ slider rig ════════════════════════════════
def _slider_life(seed: int, policy: str, n_dec: int = None) -> dict:
    """PG.4's agent, byte-for-byte in structure and rng consumption order,
    with transitions logged. Reproduces the KNOWN POSITIVE: same builder,
    same retina, same hyperparameters, same rng derivations."""
    import mujoco
    import numpy as np

    n_dec = SLIDER_DECISIONS if n_dec is None else n_dec
    model, data, panel_gid, rover_bid, (ax, ay) = _build()
    env_rng = np.random.RandomState(seed * 7919 + 13)
    agent_rng = np.random.RandomState(seed * 104729 + 7)
    retina = _Retina(model, panel_gid, rover_bid, True, env_rng)

    fwd = opt = None
    n_act = len(_ACTIONS)
    eye = np.eye(n_act, dtype=np.float32)
    if policy == "icm":
        import torch
        torch.manual_seed(seed)
        obs_dim = 4 + 2 * 32
        fwd = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + n_act, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, obs_dim))
        opt = torch.optim.Adam(fwd.parameters(), lr=1e-3)
        buf_x, buf_y = [], []
        q = np.full((121, n_act), Q_INIT)

    obs, _ = retina.observe(data)
    half = n_dec // 2
    dwell_late = 0
    obs_list, act_list, nxt_list, rew_list = [], [], [], []

    for t in range(n_dec):
        x, y = float(data.qpos[-2]), float(data.qpos[-1])
        s = _cell(x, y)
        if policy == "random":
            a = int(agent_rng.randint(n_act))
        else:
            eps = max(EPS_LO, EPS_HI - (EPS_HI - EPS_LO) * t / (n_dec // 3))
            if agent_rng.uniform() < eps:
                a = int(agent_rng.randint(n_act))
            else:
                best = np.flatnonzero(q[s] >= q[s].max() - 1e-12)
                a = int(best[agent_rng.randint(len(best))])

        data.ctrl[ax] = SPEED * _ACTIONS[a][0]
        data.ctrl[ay] = SPEED * _ACTIONS[a][1]
        for _ in range(SLIDER_SUBSTEPS):
            mujoco.mj_step(model, data)
        obs2, _hits = retina.observe(data)
        x2, y2 = float(data.qpos[-2]), float(data.qpos[-1])

        if policy == "icm":
            import torch
            with torch.no_grad():
                inp = torch.cat([torch.from_numpy(obs),
                                 torch.eye(n_act)[a]])
                r = float(((fwd(inp) - torch.from_numpy(obs2)) ** 2).sum())
            s2 = _cell(x2, y2)
            q[s, a] += Q_LR * (r + GAMMA * q[s2].max() - q[s, a])
            buf_x.append(inp.numpy()); buf_y.append(obs2)
            idx = agent_rng.randint(len(buf_x), size=min(64, len(buf_x)))
            bx = torch.from_numpy(np.stack([buf_x[i] for i in idx]))
            by = torch.from_numpy(np.stack([buf_y[i] for i in idx]))
            loss = ((fwd(bx) - by) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            rew_list.append(r)

        obs_list.append(obs); act_list.append(eye[a]); nxt_list.append(obs2)
        if t >= half and _dwell(x2, y2):
            dwell_late += 1
        obs = obs2

    return {
        "X": np.concatenate([np.stack(obs_list), np.stack(act_list)], axis=1),
        "Y": np.stack(nxt_list),
        "rewards": np.array(rew_list) if rew_list else None,
        "panel_dwell": round(dwell_late / half, 4),
        "finite": float(np.isfinite(data.qpos).all()
                        and np.isfinite(data.qvel).all()),
    }


# ═══════════════════════════════ the detector ══════════════════════════════
def _detector(arms: dict, ruler: str, seed: int) -> dict:
    """Pooled-fit forward model, out-of-fold; theta from the ruler arm.

    arms: name -> {"X": (n, obs+act), "Y": (n, obs), ...}.  Returns per-arm
    occupancy (gate), lpm_occupancy + reducibility_gain (diagnostics), the
    persistence comparison (V2's input), theta, and per-arm OOF error/flag
    arrays (for chaos_reward_ratio, aligned to the subsampled indices)."""
    import torch

    rng = np.random.RandomState(90_000 + seed)
    names = sorted(arms)
    sub_idx, X_parts, Y_parts, owner = {}, [], [], []
    for name in names:
        n = arms[name]["X"].shape[0]
        idx = (np.arange(n) if n <= DET_MAX_PER_ARM
               else np.sort(rng.choice(n, DET_MAX_PER_ARM, replace=False)))
        sub_idx[name] = idx
        X_parts.append(arms[name]["X"][idx])
        Y_parts.append(arms[name]["Y"][idx])
        owner += [name] * len(idx)
    X = np.concatenate(X_parts).astype(np.float32)
    Y = np.concatenate(Y_parts).astype(np.float32)
    owner = np.array(owner)
    N = X.shape[0]
    obs_dim = Y.shape[1]
    pers = ((X[:, :obs_dim] - Y) ** 2).sum(axis=1)

    fold = rng.permutation(N) % DET_FOLDS
    e_full = np.zeros(N); e_half = np.zeros(N)
    torch.manual_seed(70_000 + seed)

    def _train(xi, yi):
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], DET_HID), torch.nn.ReLU(),
            torch.nn.Linear(DET_HID, DET_HID), torch.nn.ReLU(),
            torch.nn.Linear(DET_HID, obs_dim))
        o = torch.optim.Adam(net.parameters(), lr=DET_LR)
        tx = torch.from_numpy(xi); ty = torch.from_numpy(yi)
        n = len(xi)
        for _ in range(DET_EPOCHS):
            perm = torch.randperm(n)
            for b in range(0, n, DET_BATCH):
                sel = perm[b:b + DET_BATCH]
                loss = ((net(tx[sel]) - ty[sel]) ** 2).sum(dim=1).mean()
                o.zero_grad(); loss.backward(); o.step()
        return net

    for k in range(DET_FOLDS):
        tr = fold != k
        te = ~tr
        xi, yi = X[tr], Y[tr]
        net_full = _train(xi, yi)
        cut = len(xi) // 2
        net_half = _train(xi[:cut], yi[:cut])
        with torch.no_grad():
            tx = torch.from_numpy(X[te])
            e_full[te] = ((net_full(tx) - torch.from_numpy(Y[te])) ** 2
                          ).sum(dim=1).numpy()
            e_half[te] = ((net_half(tx) - torch.from_numpy(Y[te])) ** 2
                          ).sum(dim=1).numpy()

    theta = float(np.quantile(e_full[owner == ruler], THETA_Q))
    flag = e_full > theta
    irreducible = (e_half - e_full) < RHO * e_half
    out = {"theta": theta,
           "beats_persistence": float(np.median(e_full) < np.median(pers)),
           "oof_median": float(np.median(e_full)),
           "pers_median": float(np.median(pers)),
           "per_arm": {}, "flags": {}, "sub_idx": sub_idx}
    for name in names:
        m = owner == name
        occ = float(flag[m].mean() / (1.0 - THETA_Q))
        lpm = float((flag[m] & irreducible[m]).mean() / (1.0 - THETA_Q))
        hi = m & flag
        gain = 0.0
        if hi.sum() > 0 and float(e_half[hi].mean()) > 0:
            gain = float((e_half[hi].mean() - e_full[hi].mean())
                         / e_half[hi].mean())
        out["per_arm"][name] = {"occupancy": round(occ, 4),
                                "lpm_occupancy": round(lpm, 4),
                                "reducibility_gain": round(gain, 4),
                                "e_mean": float(e_full[m].mean())}
        out["flags"][name] = flag[m]
    return out


def _reward_ratio(arm: dict, flags: np.ndarray, sub_idx: np.ndarray) -> float:
    """Late-half intrinsic reward, detector-flagged vs unflagged (PG.4's
    cause gate one rig over: the fixation must be FED by the chaos)."""
    r = arm["rewards"]
    if r is None:
        return 0.0
    half = len(r) // 2
    late = sub_idx >= half
    fl = flags & late
    un = ~flags & late
    if fl.sum() == 0 or un.sum() == 0:
        return 0.0
    r_sub = r[sub_idx]
    return round(float(r_sub[fl].mean() / max(1e-9, r_sub[un].mean())), 4)


# ═══════════════════════ the spec's three callables ════════════════════════
def _experiment(seed: int) -> dict:
    rig = _RagRig(seed)
    icm = _rag_life(rig, "icm", seed)
    rnd = _rag_life(rig, "random", seed)
    clm = _rag_life(rig, "climber", seed)
    det = _detector({"icm": icm, "random": rnd, "climber": clm},
                    ruler="random", seed=seed)
    pa = det["per_arm"]
    thr_rnd = max(1e-9, rnd["thrash"])
    return {
        "chaos_occupancy_icm": pa["icm"]["occupancy"],
        "chaos_occupancy_climber": pa["climber"]["occupancy"],
        "chaos_occupancy_random": pa["random"]["occupancy"],
        "chaos_detector_separation": round(
            pa["icm"]["occupancy"] - pa["climber"]["occupancy"], 4),
        "chaos_reward_ratio": _reward_ratio(icm, det["flags"]["icm"],
                                            det["sub_idx"]["icm"]),
        "panel_dwell_icm": icm["panel_dwell"],
        "lpm_occupancy_icm": pa["icm"]["lpm_occupancy"],
        "lpm_occupancy_climber": pa["climber"]["lpm_occupancy"],
        "reducibility_gain_icm": pa["icm"]["reducibility_gain"],
        "e_mean_icm": round(pa["icm"]["e_mean"], 6),
        "e_mean_random": round(pa["random"]["e_mean"], 6),
        "e_mean_climber": round(pa["climber"]["e_mean"], 6),
        "thrash_ratio_icm": round(icm["thrash"] / thr_rnd, 4),
        "thrash_ratio_climber": round(clm["thrash"] / thr_rnd, 4),
        "det_beats_persistence": det["beats_persistence"],
        "det_theta": det["theta"],
        "det_oof_median": det["oof_median"],
        "det_pers_median": det["pers_median"],
        "climber_rung_ok": clm["climber_rung_ok"],
        "climber_rise_max": clm["climber_rise_max"],
        "climber_falls": clm["climber_falls"],
        "icm_upright_frac": icm["upright_frac"],
        "icm_visited_frac": icm["visited_frac"],
        "rag_finite": min(icm["finite"], rnd["finite"], clm["finite"]),
    }


def _control(seed: int) -> dict:
    """PG.4's known positive, judged by BOTH detectors."""
    icm = _slider_life(seed, "icm")
    rnd = _slider_life(seed, "random")
    det = _detector({"icm": icm, "random": rnd}, ruler="random", seed=seed)
    pa = det["per_arm"]
    return {
        "panel_dwell": icm["panel_dwell"],
        "chaos_occupancy": pa["icm"]["occupancy"],
        "chaos_occupancy_random": pa["random"]["occupancy"],
        "lpm_occupancy": pa["icm"]["lpm_occupancy"],
        "det_beats_persistence": det["beats_persistence"],
        "det_theta": det["theta"],
        "slider_finite": min(icm["finite"], rnd["finite"]),
    }


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # V1 — physics finite everywhere
    if m.get("rag_finite", 0.0) != 1.0 or c.get("slider_finite", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    # V2 — a dead forward model's "high error" is noise (the T2.05 lesson)
    if (m.get("det_beats_persistence", 0.0) != 1.0
            or c.get("det_beats_persistence", 0.0) != 1.0):
        return _void(m, "detector model does not beat persistence; "
                        "high error from a dead model is noise")
    # V3 — the climber must have exercised (T3.09's site-under-exercise lane)
    if (m.get("climber_rung_ok", 0.0) != 1.0
            or m.get("climber_rise_max", 0.0) < CLIMBER_RISE_MIN
            or m.get("climber_falls", 0.0) < CLIMBER_FALLS_MIN):
        return _void(m, "scripted climber never exercised (no rung, no rise "
                        "or no falls); a statue's low occupancy is vacuous")
    # V4 — the ruler must be a ruler
    if (m["det_theta"] <= 0.0 or c["det_theta"] <= 0.0
            or abs(m["chaos_occupancy_random"] - 1.0) > RULER_TOL
            or abs(c["chaos_occupancy_random"] - 1.0) > RULER_TOL):
        return _void(m, "degenerate ruler: theta <= 0 or the random arm's "
                        "own occupancy is off 1.0 (ties)")
    # THE CLAIM — first falsified clause named (the BA.03 one-bit lesson)
    if m["chaos_occupancy_icm"] < OCC_ICM_MIN:
        m["claim_branch"] = ("detector blind: ragdoll-ICM occupancy below "
                             "3.0 — no arm's immunity may be reported")
        return False
    if m["chaos_reward_ratio"] < REWARD_RATIO_MIN:
        m["claim_branch"] = ("fixation not fed by chaos: flagged-vs-unflagged "
                             "intrinsic reward ratio below 2.0")
        return False
    if m["panel_dwell_icm"] != 0.0:
        m["claim_branch"] = ("ragdoll-ICM dwelled at the deleted panel's "
                             "location; the zero-dwell demonstration is "
                             "confounded")
        return False
    if m["chaos_occupancy_climber"] > OCC_CLIMBER_MAX:
        m["claim_branch"] = ("detector penalises coordinated motion and "
                             "falling: the scripted climber is flagged")
        return False
    if not (c["panel_dwell"] > CONTROL_DWELL_MIN
            and c["chaos_occupancy"] >= CONTROL_OCC_MIN):
        m["claim_branch"] = ("the two detectors do not agree on the known "
                             "positive (PG.4's trapped agent)")
        return False
    m["claim_branch"] = ("ragdoll farmer flagged at zero dwell, climber "
                         "clean, both detectors agree on the known positive")
    return True


def run(ledger: Ledger | None = None):
    import os
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))
    return run_spec(BY_ID["LT.02"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


# ═══════════════════════════════ pilot ═════════════════════════════════════
def _pilot():
    """Seed 90, reduced envelope; mechanics, gate aliveness and runtime only
    (disjoint from recorded seeds 0-2; W0.DIAG's pilot idiom). Prints JSON,
    records NOTHING — every claim bar is the registry's, frozen 2026-08-09."""
    import json
    import time as _time

    out = {}
    t0 = _time.time()
    rig = _RagRig(90)
    out["z_rest"] = rig.z_rest
    n = 1200
    icm = _rag_life(rig, "icm", 90, n_dec=n)
    rnd = _rag_life(rig, "random", 90, n_dec=n)
    clm = _rag_life(rig, "climber", 90, n_dec=n)
    out["t_rag_lives_s"] = round(_time.time() - t0, 1)
    t0 = _time.time()
    det = _detector({"icm": icm, "random": rnd, "climber": clm},
                    ruler="random", seed=90)
    out["t_rag_det_s"] = round(_time.time() - t0, 1)
    out["rag"] = {
        "per_arm": det["per_arm"], "theta": det["theta"],
        "oof_median": det["oof_median"], "pers_median": det["pers_median"],
        "beats_persistence": det["beats_persistence"],
        "reward_ratio": _reward_ratio(icm, det["flags"]["icm"],
                                      det["sub_idx"]["icm"]),
        "icm_dwell": icm["panel_dwell"], "icm_upright": icm["upright_frac"],
        "icm_visited": icm["visited_frac"],
        "thrash": {"icm": icm["thrash"], "random": rnd["thrash"],
                   "climber": clm["thrash"]},
        "climber": {k: clm[k] for k in
                    ("climber_falls", "climber_rise_max", "climber_rung_ok")},
        "finite": [icm["finite"], rnd["finite"], clm["finite"]],
    }

    t0 = _time.time()
    sic = _slider_life(90, "icm", n_dec=6000)
    srn = _slider_life(90, "random", n_dec=6000)
    out["t_slider_lives_s"] = round(_time.time() - t0, 1)
    t0 = _time.time()
    sdet = _detector({"icm": sic, "random": srn}, ruler="random", seed=90)
    out["t_slider_det_s"] = round(_time.time() - t0, 1)
    out["slider"] = {
        "per_arm": sdet["per_arm"], "theta": sdet["theta"],
        "oof_median": sdet["oof_median"], "pers_median": sdet["pers_median"],
        "beats_persistence": sdet["beats_persistence"],
        "icm_dwell": sic["panel_dwell"],
        "finite": [sic["finite"], srn["finite"]],
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _pilot()
