"""w0.py — World Zero: the body, the senses, and one decision of Jack's life.

`LEARNING_CORE.md` §5.0 lists six things the learning-core bakeoff needs from
the world, and W0 is the smallest world that supplies them. This module is the
substrate for LC.02 (throughput), LC.03 (screening) and LC.04/LC.05 (the
arbitration). It is deliberately NOT a claim: nothing here is measured, and the
only thing that may assert a capability is a spec in `experiments/tests/`.

    W0-1  needs        `drives.DriveLayer` on the rover, e/i/w + d(h)
    W0-2  death        `lethal=True`: e or i reaching 0 ends the life, and the
                       body reappears at a UNIFORMLY RANDOM legal spawn drawn
                       from `legal_spawns()`. Certified by XL.00.
    W0-3  cross-life   `diary=EpisodicMemory(...)`: the store is never reset by
                       death and every row it writes carries `meta["life"]`.
                       The substrate is ME.10's; XL.00 certifies it crosses
                       death here.
    W0-4  observation  the six-modality dict below
    W0-5  noise panel  `playground.PlaygroundParams.noise_panel`, unchanged
    W0-6  zero reward  there is no reward function in this file. Grep it.

**The body is the climber-rover** (`playground._rover_fragments`), 8 action
dimensions, for the three reasons §5.0 gives — not the humanoid.

THE OBSERVATION, and why each width is what it is. The keys and dimensions are
`cores.MODALITIES` verbatim, because F3 requires an identical observation for
every arm and the arms were built against that table in LC.01:

  vision    32  16 rays x [distance, texture]. PG.4's `_Retina`, generalised:
                the ray mask now excludes the body by RENDER GROUP (the rover's
                own geoms are group 3) instead of by a single `bodyexclude` id,
                because the rover is three bodies and PG.4's rover was one — a
                single exclude id would have let the arms occlude the eyes.
  audio     16  8 log-spaced bands x 2 ears, from `ContactAudio`'s own events
                and its own pan convention (gL = sqrt((1-p)/2)). Not a rendered
                waveform: rendering 0.3 s of 16 kHz stereo per decision would
                make hearing cost more than the learner, and the band energies
                are what a cochlea delivers anyway. The mapping from event to
                band uses `synth.fundamental(voiced_geom)`, which is the same
                function PG.5 certified.
  touch      8  4 geoms (torso, foot, handL, handR) x [log-force, contact flag]
  proprio   12  4 arm slide positions, 4 velocities, torso z, torso upright
                cosine, root vx, root vy
  needs      6  `drives.DriveLayer.obs` — [e, i, w, d(h), edot, idot]
  language  32  ABSENT, and absent as an INPUT CONDITION, not as zeros: the key
                is reported in `dropped` and the core substitutes its learned
                `missing` embedding (LC.01's U3). W0 has no talker in it yet;
                the parent-LLM voice is GOAL.md's, not this iteration's. Zeros
                would have been a silent lie the cores could not distinguish
                from a silent speaker.
  placebo    6  matched dimension, matched statistics, zero information. Carried
                into the world because LC.01 admitted the arms WITH it, and an
                arm whose input width changed between admission and scoring was
                not the arm that was admitted.

THE DRIVE IS THE ONE CHEAT AND IT IS GATED. Two of the eight action dimensions
are a world-frame horizontal force on the torso, bounded at 600 N and applied
only while some rover geom touches floor, ramp or stair. It grants locomotion —
T2.01's problem, not the learning rule's — and cannot contribute one newton off
the ground, so ladder height is still earned by the arms. `drive_gate_frac` is
reported every run: if it ever reads 1.0 the gate is not gating.

DEATH IS OPT-IN, AND THAT IS NOT TIMIDITY. `lethal` defaults to False because
`LC.02`'s certificate is a measurement of ONE UNBROKEN LIFE (its own hypothesis
says so) and flipping the default would retroactively change what that ledger
entry measured without changing a line of its test. A caller that wants W0-2
asks for it; `LC.03` asks for it.

WHAT THIS MODULE DOES NOT DO: it does not train, it does not reward, and it does
not decide anything. `LC.02` measures how fast it runs. It now ENDS A LIFE when
asked to, and that is the one behaviour here that a spec certifies (XL.00)
rather than merely uses.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ContactAudio import MIN_DISTANCE, ContactAudioSynth      # noqa: E402
from experiments import drives                                 # noqa: E402
from experiments.cores import MODALITIES, PLACEBO_DIM, PLACEBO_KEY  # noqa: E402

# ── the decision, fixed here so no caller can quietly change the unit ───────
SUBSTEPS = 40                       # 40 x 0.005 s = 0.2 simulated seconds
DECISIONS_PER_SIM_S = 5.0           # LEARNING_CORE.md §5.1's accounting unit
SIM_S_PER_DECISION = 1.0 / DECISIONS_PER_SIM_S

# ── the senses ─────────────────────────────────────────────────────────────
N_RAYS = MODALITIES["vision"] // 2          # 16 rays x [distance, texture]
RAY_MAX_RANGE = 8.0                         # m; beyond this a ray reads "empty"
RAY_Z_OFFSET = 0.05                         # above the body origin
N_BANDS = MODALITIES["audio"] // 2          # 8 bands x 2 ears
BAND_LO_HZ, BAND_HI_HZ = 80.0, 4000.0       # ContactAudio's own fundamental clip
TOUCH_GEOMS = ("rover_torso", "rover_foot", "handL", "handR")
TOUCH_FORCE_SCALE = 100.0                   # N, the log-compression knee

# The pool, as `playground.make_playground` builds it.
POOL_XY = (2.6, -2.4)

# Ray group mask: the rover's own geoms are render group 3 and are invisible to
# its eyes; everything else in the world (groups 0-2) is visible.
_RAY_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)


# ── W0-2: death, and a respawn that is not a free teleport ─────────────────
SPAWN_GRID = 25              # candidates per axis; 625 poses probed once, cached
SPAWN_MARGIN = 0.75          # m inside the arena edge, so no spawn is off-world
SPAWN_PENETRATION = 1e-3     # m; a resting body deeper than this is INSIDE
                             # something and that pose is not a legal spawn
MIN_LEGAL_SPAWNS = 100       # a world offering fewer is not one you can respawn
                             # uniformly IN — refuse rather than sample from 3
DEATH_FLOOR = 0.0            # e or i at the clip floor. `drives` clips to
                             # [0, 1], so this is reached, never crossed.


def band_edges() -> np.ndarray:
    """Log-spaced band boundaries over ContactAudio's fundamental range."""
    return np.geomspace(BAND_LO_HZ, BAND_HI_HZ, N_BANDS + 1)


def uniform_legal_spawn(legal: np.ndarray, rng: np.random.RandomState,
                        death_xy: Tuple[float, float]) -> Tuple[float, float]:
    """§5.0 W0-2's rule: uniform over the legal set, blind to where he died.

    `death_xy` is in the signature and deliberately unused. The alternative —
    a sampler that cannot see the death site — makes the independence claim
    true by a type signature rather than by measurement, and XL.00's positive
    control (`spawn_at_death`, same signature, uses it) could not exist. A
    property that no control can violate is not a property this repo may claim.
    """
    k = int(rng.randint(len(legal)))
    return float(legal[k][0]), float(legal[k][1])


class W0:
    """One rover, one playground, one unbroken life.

    Usage, one decision:

        w = W0(seed=0, j0=..., alpha=...)
        obs = w.observe()
        w.decide(action)           # action: 8 floats in [-1, 1]

    `observe()` is pure. `decide()` advances 0.2 simulated seconds.
    """

    #: keys the core must treat as an input CONDITION, not as data (LC.01 U3)
    DROPPED: Tuple[str, ...] = ("language",)

    def __init__(self, seed: int = 0, *, j0: float, alpha: float,
                 params: Optional[object] = None, mutate: bool = True,
                 lethal: bool = False, diary: Optional[object] = None,
                 spawn_sampler=None):
        import mujoco
        import playground as pg

        self.mujoco = mujoco
        self.pg = pg
        p = params
        if p is None:
            p = pg.PlaygroundParams(seed=seed)
            if mutate and seed:
                # F4: the world varies with the seed and identically across
                # arms. Seed 0 is the nursery; seeds 1+ take one ACCEL edit,
                # exactly as PG.8 does — three runs under three seed integers
                # in one world would not be three seeds.
                p = p.mutate(np.random.RandomState(seed))
        self.params = p
        self.model, self.data, self.water = pg.make_playground(
            p, with_water=True, with_rover=True)
        self.ix = pg.rover_index(self.model)

        # F5: every width is asserted against the LIVE model, never a constant.
        if int(self.model.nu) != pg.ROVER_NU:
            raise RuntimeError(f"rover nu is {self.model.nu}, not {pg.ROVER_NU}")
        self.action_dim = pg.ROVER_ACTION_DIM

        self.rover_bid = self.ix["body"]["rover"]
        self.touch_gids = [self.ix["geom"][n] for n in TOUCH_GEOMS]
        self.body_gids = {g for g in range(self.model.ngeom)
                          if int(self.model.geom_bodyid[g]) in
                          set(self.ix["body"].values())}
        self.ground_gids = self.ix["ground_geoms"]
        self._body_mask = np.zeros(self.model.ngeom, dtype=bool)
        self._body_mask[list(self.body_gids)] = True
        self._ground_mask = np.zeros(self.model.ngeom, dtype=bool)
        self._ground_mask[list(self.ground_gids)] = True

        self.synth = ContactAudioSynth(self.model)
        self._edges = band_edges()
        self._fund = np.array([self.synth.fundamental(g)
                               for g in range(self.model.ngeom)])
        self._band_of = np.clip(
            np.searchsorted(self._edges, self._fund, side="right") - 1,
            0, N_BANDS - 1)
        self._audio = np.zeros(MODALITIES["audio"], dtype=np.float32)
        self.audio_events_total = 0
        try:
            self.panel_gid = int(self.model.geom("noise_panel").id)
        except (KeyError, ValueError):
            self.panel_gid = -1            # a mutated world may drop the panel

        self.drives = drives.DriveLayer(
            self.model, j0=j0, alpha=alpha,
            pool=(POOL_XY[0], POOL_XY[1], p.pool_size, 0.0),
            body=drives.rover_body_ref(self.model))
        self._prev_drive = drives.DriveState()

        self._ray_dirs = np.array(
            [[math.cos(2 * math.pi * k / N_RAYS),
              math.sin(2 * math.pi * k / N_RAYS), 0.0] for k in range(N_RAYS)])
        self._geomid = np.zeros(1, dtype=np.int32)
        # Stable per-geom texture in [0.1, 0.9] — PG.4's convention, so the
        # noise panel is the only thing whose texture moves.
        self._tex = ((np.arange(self.model.ngeom) * 0.37) % 0.8) + 0.1
        self._rng = np.random.RandomState(seed * 7919 + 13)

        self.decisions = 0
        self.sim_seconds = 0.0
        self.drive_gate_open = 0

        # ── W0-2/W0-3 state ────────────────────────────────────────────
        self.lethal = bool(lethal)
        self.diary = diary
        self.spawn_sampler = spawn_sampler or uniform_legal_spawn
        # A SEPARATE stream from `self._rng`. `_rng` drives the noise panel's
        # texture and the placebo channel, i.e. what he SEES; drawing respawns
        # from it would make the sensory noise of life k+1 a function of how
        # many times he had died, which is a channel from the death counter into
        # the observation that no arm should be able to read.
        self._spawn_rng = np.random.RandomState(seed * 104729 + 7)
        self._legal: Optional[np.ndarray] = None
        self.life = 0
        self.deaths = 0
        self.died_this_decision = False
        self.last_death_cause = ""
        self._life_started_at = 0.0
        self.life_lengths: list = []        # sim-seconds, one per COMPLETED life
        self.death_sites: list = []         # (x, y) where each life ended
        self.spawn_sites: list = []         # (x, y) where the next one began

        mujoco.mj_forward(self.model, self.data)
        self.mujoco.mj_rnePostConstraint(self.model, self.data)

    # ── the senses ──────────────────────────────────────────────────────
    def _vision(self) -> np.ndarray:
        pos = np.array(self.data.xpos[self.rover_bid], dtype=float)
        pos[2] += RAY_Z_OFFSET
        out = np.empty(MODALITIES["vision"], dtype=np.float32)
        for k in range(N_RAYS):
            dist = self.mujoco.mj_ray(self.model, self.data, pos,
                                      self._ray_dirs[k], _RAY_GROUPS, 1,
                                      -1, self._geomid)
            gid = int(self._geomid[0])
            if dist < 0 or dist > RAY_MAX_RANGE or gid < 0:
                d, t = 1.0, 0.0
            else:
                d = dist / RAY_MAX_RANGE
                t = float(self._tex[gid])
                if gid == self.panel_gid:
                    # PG.4's noisy TV, with its R_RESOLVE acuity falloff: the
                    # texture is irreducibly unpredictable and gets sharper as
                    # he approaches. W0-5.
                    amp = min(1.0, 2.5 / max(dist, 1e-6))
                    t = 0.5 + amp * (float(self._rng.uniform()) - 0.5)
            out[2 * k] = d
            out[2 * k + 1] = t
        return out

    def _touch(self) -> np.ndarray:
        out = np.zeros(MODALITIES["touch"], dtype=np.float32)
        f6 = np.zeros(6)
        for k in range(int(self.data.ncon)):
            con = self.data.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            for gid in (g1, g2):
                if gid not in self.touch_gids:
                    continue
                j = self.touch_gids.index(gid)
                self.mujoco.mj_contactForce(self.model, self.data, k, f6)
                out[2 * j] += abs(float(f6[0]))
                out[2 * j + 1] = 1.0
        for j in range(len(TOUCH_GEOMS)):
            out[2 * j] = math.log1p(out[2 * j]) / math.log1p(TOUCH_FORCE_SCALE)
        return out

    def _proprio(self) -> np.ndarray:
        qa = self.ix["jnt_qposadr"]
        da = self.ix["jnt_dofadr"]
        names = ("reachL", "liftL", "reachR", "liftR")
        pos = [float(self.data.qpos[qa[n]]) for n in names]
        vel = [float(self.data.qvel[da[n]]) for n in names]
        root = self.ix["root_dofadr"]
        out = np.array(pos + vel + [
            float(self.data.xpos[self.rover_bid][2]),
            float(self.data.xmat[self.rover_bid][8]),      # upright cosine
            float(self.data.qvel[root]), float(self.data.qvel[root + 1]),
        ], dtype=np.float32)
        if out.shape[0] != MODALITIES["proprio"]:
            raise RuntimeError(f"proprio is {out.shape[0]}, not "
                               f"{MODALITIES['proprio']}")
        return out

    def _needs(self) -> np.ndarray:
        return self.drives.obs(self._prev_drive,
                               dt=SIM_S_PER_DECISION).astype(np.float32)

    def observe(self) -> Dict[str, np.ndarray]:
        """The W0-4 dict. `language` is present-and-dropped, never zero-filled."""
        obs = {
            "vision": self._vision(),
            "audio": self._audio.copy(),
            "touch": self._touch(),
            "proprio": self._proprio(),
            "needs": self._needs(),
            # Shape-correct so a caller can stack it; the core is told to drop
            # it and substitutes its learned `missing` embedding instead.
            "language": np.zeros(MODALITIES["language"], dtype=np.float32),
            PLACEBO_KEY: self._rng.randn(PLACEBO_DIM).astype(np.float32),
        }
        for k, v in obs.items():
            if v.shape[0] != (PLACEBO_DIM if k == PLACEBO_KEY else MODALITIES[k]):
                raise RuntimeError(f"modality {k} is {v.shape[0]} wide")
        return obs

    # ── one decision ────────────────────────────────────────────────────
    def decide(self, action: np.ndarray) -> None:
        """Apply an 8-vector and advance 0.2 simulated seconds.

        action[0:4]  arm slide targets, mapped to each actuator's ctrlrange
        action[4:6]  adhesion, mapped to [0, 1]
        action[6:8]  the gated horizontal drive, in [-1, 1] x 600 N
        """
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if a.shape[0] != self.action_dim:
            raise RuntimeError(f"action is {a.shape[0]}, not {self.action_dim}")

        gear = self.drives.gear_scale()          # §2.2 weakness, not a reward
        lo = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=float)
        hi = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=float)
        ctrl = lo + (a[:6] * 0.5 + 0.5) * (hi - lo)
        ctrl[:4] *= gear

        force = a[6:8] * self.pg.ROVER_DRIVE_FORCE * gear
        self.drives.begin_decision()
        dt = float(self.model.opt.timestep)
        n_gated = 0
        for _ in range(SUBSTEPS):
            self.data.ctrl[:] = ctrl
            gate = self._grounded()
            n_gated += int(gate)
            self.data.xfrc_applied[self.rover_bid, :2] = force if gate else 0.0
            if self.water is not None:
                self.water.apply(self.model, self.data)
            self.mujoco.mj_step(self.model, self.data)
            self.synth.step(self.data)
            self.drives.substep(self.model, self.data, dt)
        self.data.xfrc_applied[self.rover_bid, :2] = 0.0
        self.mujoco.mj_rnePostConstraint(self.model, self.data)
        # This call fills `cfrc_ext` for the OBSERVATION only. It used to carry
        # a known, deliberate staleness for the drive layer as well: `cfrc_ext`
        # is filled by mj_rnePostConstraint (the PG.8 lesson) and never by
        # mj_step, so every `drives.substep()` above read the PREVIOUS
        # decision's contact state for its impact-impulse accumulation. The
        # correct per-substep call was tried (2026-08-09), cost ~15-25%
        # throughput and dropped 4 of 5 LC.02 arms below the 5.0 floor, and was
        # reverted; it was safe only because LC.02 never reads `j`.
        # RESOLVED 2026-08-10: `PS.01/J2` replaced §2.2's force channel with the
        # root's arrival speed, so `drives.substep` no longer touches
        # `cfrc_ext` at all. `j` is now correct in THIS loop, at no throughput
        # cost, and PS.01 no longer needs a stepping loop of its own to get it.

        self._prev_drive = drives.DriveState(**vars(self.drives.state))
        self.drives.decide()
        self._audio = self._drain_audio()
        self.decisions += 1
        self.drive_gate_open += n_gated
        self.sim_seconds += SUBSTEPS * dt

        # W0-2. Checked at the decision boundary, AFTER the clock advances, so
        # the recorded life length is the time he actually lived.
        self.died_this_decision = False
        if self.lethal:
            cause = self.death_cause()
            if cause:
                self._die(cause)

    # ── W0-2: death and respawn ─────────────────────────────────────────
    def death_cause(self) -> str:
        """"energy" | "integrity" | "" — the empty string is alive.

        A string, not a bool, because "he died" and "he starved" are different
        facts and an aggregate death count would hide a world where every death
        has one cause. XL.00 reports the split.
        """
        s = self.drives.state
        if s.e <= DEATH_FLOOR:
            return "energy"
        if s.i <= DEATH_FLOOR:
            return "integrity"
        return ""

    def legal_spawns(self) -> np.ndarray:
        """(N, 2) legal respawn poses, computed ONCE against the live model.

        Legal = the body, placed upright at rest height with its arms at zero,
        penetrates nothing that is not ground. That is a geometric property of
        THIS world, so it is derived from the model rather than declared: a
        hand-written spawn list would be a constant that a mutated world could
        silently invalidate, which is the T0.14 mistake.

        Deliberately NOT excluded: the ladder base. §5.0 forbids respawning *at*
        a useful location, not respawning uniformly over a set that contains one
        cell near it — carving out "good" cells would be the experimenter
        shaping the curriculum in the opposite direction.
        """
        if self._legal is not None:
            return self._legal
        a = float(self.params.arena_size) - SPAWN_MARGIN
        axis = np.linspace(-a, a, SPAWN_GRID)
        qpos0, qvel0 = self.data.qpos.copy(), self.data.qvel.copy()
        ok = []
        for x in axis:
            for y in axis:
                self._place(float(x), float(y))
                self.mujoco.mj_forward(self.model, self.data)
                if not self._penetrating():
                    ok.append((float(x), float(y)))
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = qvel0
        self.mujoco.mj_forward(self.model, self.data)
        if len(ok) < MIN_LEGAL_SPAWNS:
            raise RuntimeError(
                f"only {len(ok)} legal spawns in this world (need "
                f"{MIN_LEGAL_SPAWNS}); 'uniformly random legal spawn' over a "
                f"handful of poses is a fixed spawn wearing a random costume")
        self._legal = np.asarray(ok, dtype=float)
        return self._legal

    def _place(self, x: float, y: float) -> None:
        """Move the BODY to (x, y) at rest height, upright, arms zeroed, at rest.

        Note what is not here: `mj_resetData`. Resetting the whole world on
        death would put the objects back, refill the food and rewind the clock —
        the free teleport to a good state that §5.0 names as the thing W0-2 must
        not be.
        """
        qa, da = self.ix["root_qposadr"], self.ix["root_dofadr"]
        self.data.qpos[qa:qa + 3] = (x, y, self.pg.ROVER_REST_Z + 0.01)
        self.data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[da:da + 6] = 0.0
        for n in ("reachL", "liftL", "reachR", "liftR"):
            self.data.qpos[self.ix["jnt_qposadr"][n]] = 0.0
            self.data.qvel[self.ix["jnt_dofadr"][n]] = 0.0
        self.data.qacc[:] = 0.0
        self.data.xfrc_applied[self.rover_bid, :] = 0.0

    def _penetrating(self) -> bool:
        """Is any rover geom inside a non-ground geom by more than the tolerance?

        Rover-rover pairs are excluded: the arms fold against the torso in every
        pose and that is the body's own geometry, not the world's.
        """
        for k in range(int(self.data.ncon)):
            con = self.data.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            mine = (g1 in self.body_gids, g2 in self.body_gids)
            if not any(mine) or all(mine):
                continue
            other = g2 if mine[0] else g1
            if other in self.ground_gids:
                continue
            if float(con.dist) < -SPAWN_PENETRATION:
                return True
        return False

    def respawn(self, at: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """Put a new body in the world. Returns the (x, y) it was placed at."""
        legal = self.legal_spawns()
        death_xy = (float(self.data.xpos[self.rover_bid][0]),
                    float(self.data.xpos[self.rover_bid][1]))
        x, y = (at if at is not None
                else self.spawn_sampler(legal, self._spawn_rng, death_xy))
        self._place(float(x), float(y))
        self.mujoco.mj_forward(self.model, self.data)
        self.mujoco.mj_rnePostConstraint(self.model, self.data)
        self.drives.new_body()
        self._prev_drive = drives.DriveState()
        # The old body's last sounds do not follow the new one into the world.
        self.synth.events = []
        self._audio = np.zeros(MODALITIES["audio"], dtype=np.float32)
        return float(x), float(y)

    def _die(self, cause: str) -> None:
        death_xy = (float(self.data.xpos[self.rover_bid][0]),
                    float(self.data.xpos[self.rover_bid][1]))
        self.life_lengths.append(self.sim_seconds - self._life_started_at)
        self.death_sites.append(death_xy)
        if self.diary is not None:
            # W0-3: the row is written by the WORLD, carries the life index, and
            # is never removed by what follows. `did`, because dying is
            # something that happened to him, not something he was told.
            self.diary.record(
                "did", "jack",
                f"life ended, {cause} gone, after "
                f"{self.life_lengths[-1]:.1f} seconds",
                importance=10.0,
                meta={"life": self.life, "cause": cause,
                      "sim_s": self.sim_seconds,
                      "x": death_xy[0], "y": death_xy[1]})
        self.deaths += 1
        self.last_death_cause = cause
        self.died_this_decision = True
        self.life += 1
        self.spawn_sites.append(self.respawn())
        self._life_started_at = self.sim_seconds

    def _grounded(self) -> bool:
        """Is any rover geom touching floor, ramp or stair? The drive's gate.

        Evaluated every substep — 40 times per decision — so it is one numpy
        pass over `data.contact.geom`, not a Python loop. The readable loop
        form cost 1.32 ms of a 20.4 ms decision when LC.02 first profiled it.
        """
        n = int(self.data.ncon)
        if n == 0:
            return False
        g = np.asarray(self.data.contact.geom[:n])
        a, b = g[:, 0], g[:, 1]
        return bool(np.any((self._body_mask[a] & self._ground_mask[b])
                           | (self._body_mask[b] & self._ground_mask[a])))

    def _drain_audio(self) -> np.ndarray:
        """Events fired during this decision -> 8 bands x 2 ears.

        Listener at the torso, facing the body's +x axis. ContactAudio's own
        conventions throughout (`ContactAudio.py:26-29`): p = -sin(azimuth),
        gL = sqrt((1-p)/2), gR = sqrt((1+p)/2), 1/distance attenuation with a
        MIN_DISTANCE floor. Energy is amplitude squared, which is what makes
        band energies additive across simultaneous events.
        """
        out = np.zeros(MODALITIES["audio"], dtype=np.float32)
        # The synth appends to `events` forever. A life is meant to run for
        # hours, so the queue is DRAINED here rather than indexed into: the
        # cumulative count is kept separately and the list cannot grow without
        # bound. ("Bound the thing you are optimising" applies to a log too.)
        new = self.synth.events
        self.synth.events = []
        self.audio_events_total += len(new)
        if not new:
            return out
        pos = np.array(self.data.xpos[self.rover_bid], dtype=float)
        mat = np.array(self.data.xmat[self.rover_bid], dtype=float).reshape(3, 3)
        yaw = math.atan2(mat[1, 0], mat[0, 0])
        self.synth.set_listener(pos, yaw)
        for ev in new:
            p = -math.sin(ev.azimuth)
            gl = math.sqrt(max(0.0, (1.0 - p) / 2.0))
            gr = math.sqrt(max(0.0, (1.0 + p) / 2.0))
            att = 1.0 / max(ev.distance, MIN_DISTANCE)
            e = (ev.amp * att) ** 2
            b = int(self._band_of[ev.voiced_geom])
            out[2 * b] += e * gl * gl
            out[2 * b + 1] += e * gr * gr
        return out

    # ── read-only reporting ─────────────────────────────────────────────
    def report(self) -> Dict[str, float]:
        """Everything a spec may want about the run that is not a claim."""
        return {
            "decisions": float(self.decisions),
            "sim_seconds": float(self.sim_seconds),
            "drive_gate_frac": (self.drive_gate_open
                                / max(1, self.decisions * SUBSTEPS)),
            "audio_events": float(self.audio_events_total),
            "torso_z": float(self.data.xpos[self.rover_bid][2]),
            "upright_cos": float(self.data.xmat[self.rover_bid][8]),
            "energy": float(self.drives.state.e),
            "integrity": float(self.drives.state.i),
            "lethal": float(self.lethal),
            "deaths": float(self.deaths),
            "life": float(self.life),
            "life_s": float(self.sim_seconds - self._life_started_at),
        }


def random_action(rng: np.random.RandomState, n: int = 8) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, n)
