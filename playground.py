"""Jack's playground: the world he learns by living in.

Serves GOAL.md. Nobody ships "humanoid nursery with a ladder and a pool", so it
is built here — procedurally, from a parameter vector, because that vector is
also the mutation space for the open-ended loop (ACCEL-style, arXiv:2203.01302):
the world grows with him.

What is in it and why each piece earns its place:

  ramp / stairs   the learnable precursors. Learning progress will find these
                  long before the ladder, and that ordering IS the emergent
                  curriculum ladder spec CU.2 tests for.
  ladder          the owner's brief, literally: a ladder with a reward at the
                  top. Humanoid-v5 has ball hands and no fingers, so grip is
                  impossible by geometry — MuJoCo ADHESION actuators (native
                  since 2.2) give a controllable "hold" scalar instead. Falling
                  off is not failure, it is the data.
  pool            MuJoCo has no fluid volume, only a global medium. So water is
                  a REGION plus a passive-force callback: buoyancy on submerged
                  geom volume and quadratic drag. He must struggle and learn.
  objects         balls, boxes and a cylinder with randomised mass/friction —
                  the affordance substrate (push? lift? roll?) for CU.6.
  seesaw          a hinged plank: a dynamic affordance, unlike the static ramp.
  noise panel     MANDATORY FIXTURE, not scenery. A wall patch whose texture
                  re-randomises every step. A prediction-error agent must get
                  trapped here (PG.4), and every curiosity claim must report its
                  dwell time near it. Without a working trap, "his curiosity is
                  real" is unfalsifiable.
  walls           the nursery is a room. Load-bearing for the noisy-TV fixture:
                  the panel must differ from its surroundings ONLY in texture
                  (see the wall comment in build_mjcf).
  JACK            `with_humanoid=True` puts the Humanoid-v5 body IN the room,
                  spawned within reach of the ladder. Until 2026-08-09 that
                  argument existed and did nothing, and nothing in the repo
                  ever passed True: PG.1-PG.7 all passed on an EMPTY world.
                  See PG.8 and the LESSONS.md entry it produced.

Deliberately NOT here: anything requiring a resident GPU, and any reward
function. The playground has no goals. Goals come from Jack.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

WATER_DENSITY = 1000.0      # kg/m^3
GRAVITY = 9.81

# The ladder's footprint, in one place so a spec can aim at it rather than
# re-deriving a literal that build_mjcf might later move.
LADDER_X, LADDER_Y = 0.0, -2.6
LADDER_HALF_WIDTH = 0.25            # rail offset from LADDER_X

# Where Jack starts: this far from the ladder base on the +y side (open floor —
# the ladder sits near the arena's -y edge), at Humanoid-v5's own torso height.
SPAWN_OFFSET_Y = 1.0
SPAWN_Z = 1.4

# Humanoid-v5 as gymnasium ships it: 13 bodies at or below the torso, 24 qpos,
# 23 dof, 17 motors, and the 348-dim observation PipelineConfig expects. These
# are asserted against the live model, not trusted.
HUMANOID_NBODY = 13
HUMANOID_NQ = 24
HUMANOID_NV = 23
HUMANOID_NU = 17
HUMANOID_OBS_DIM = 348


@dataclass
class PlaygroundParams:
    """The mutation space. Every field is something the open-ended loop may edit.

    Defaults are the "nursery" setting: gentle ramp, climbable stairs, a ladder
    that is hard but not absurd, shallow water.
    """
    ramp_angle_deg: float = 15.0
    stair_count: int = 4
    stair_rise: float = 0.18
    stair_run: float = 0.30
    ladder_rungs: int = 6
    ladder_rung_spacing: float = 0.30
    ladder_height: float = 1.8
    pool_size: float = 1.5
    pool_depth: float = 0.6
    n_objects: int = 5
    object_mass_range: tuple = (0.2, 3.0)
    object_size_range: tuple = (0.06, 0.18)
    seesaw: bool = True
    noise_panel: bool = True
    arena_size: float = 6.0
    seed: int = 0
    # Jack's spawn point. None means "derive it from the ladder base" — the
    # world may mutate the ladder, and he must still start within reach of it.
    # An explicit tuple is how PG.8's control puts him outside the arena.
    humanoid_spawn: Optional[tuple] = None

    def spawn(self) -> tuple:
        """(x, y, z) of the torso at t=0. Tracks the ladder unless overridden."""
        if self.humanoid_spawn is not None:
            return tuple(float(v) for v in self.humanoid_spawn)
        return (LADDER_X, LADDER_Y + SPAWN_OFFSET_Y, SPAWN_Z)

    def mutate(self, rng: np.random.RandomState, strength: float = 0.15) -> "PlaygroundParams":
        """One ACCEL-style edit: perturb toward a neighbouring world.

        Small edits, not resampling — the point is a world adjacent to the one
        he has partly mastered, which is what makes the curriculum climb rather
        than jump to noise.
        """
        def jitter(v, lo, hi):
            return float(np.clip(v * (1 + rng.uniform(-strength, strength)), lo, hi))

        return PlaygroundParams(
            ramp_angle_deg=jitter(self.ramp_angle_deg, 5.0, 40.0),
            stair_count=int(np.clip(self.stair_count + rng.randint(-1, 2), 2, 8)),
            stair_rise=jitter(self.stair_rise, 0.08, 0.35),
            stair_run=self.stair_run,
            ladder_rungs=int(np.clip(self.ladder_rungs + rng.randint(-1, 2), 3, 10)),
            ladder_rung_spacing=jitter(self.ladder_rung_spacing, 0.20, 0.45),
            ladder_height=self.ladder_height,
            pool_size=jitter(self.pool_size, 0.8, 3.0),
            pool_depth=jitter(self.pool_depth, 0.2, 1.2),
            n_objects=int(np.clip(self.n_objects + rng.randint(-1, 2), 2, 10)),
            object_mass_range=self.object_mass_range,
            object_size_range=self.object_size_range,
            seesaw=self.seesaw, noise_panel=self.noise_panel,
            arena_size=self.arena_size, seed=int(rng.randint(0, 10_000)),
            humanoid_spawn=self.humanoid_spawn,
        )


def humanoid_source_xml() -> Path:
    """Path to gymnasium's Humanoid asset — the SOURCE OF TRUTH for the body.

    Deliberately not a copy in this repo. The pipeline trains on `Humanoid-v5`;
    if the playground's Jack were a hand-transcribed twin, the two could drift
    apart silently and every playground result would be about a body nobody
    trains. Reading the shipped asset makes them the same body by construction,
    and PG.8 asserts the compiled constants match `gym.make("Humanoid-v5")`.
    """
    import gymnasium.envs.mujoco as gm
    return Path(gm.__file__).resolve().parent / "assets" / "humanoid.xml"


def _humanoid_fragments(spawn: tuple) -> tuple:
    """Humanoid-v5's torso subtree, tendons and motors, in RADIANS.

    Two hazards, both of the kind this repo has already been bitten by:

    ANGLE UNITS. gymnasium's humanoid.xml declares `<compiler angle="degree">`;
    this playground declares radian, and one document has one compiler. Splicing
    the text verbatim would reinterpret every hinge `range` as radians — a
    -160..-2 knee becomes -9169..-115 degrees, i.e. unlimited — and nothing
    would error. That is the MJCF-degrees bug that broke PG.1's ramp, one level
    up. So the 17 hinge ranges are converted here; every other attribute in the
    file (pos, axis, fromto, quat, size, gear, ctrlrange) is unitless.

    DEFAULTS. The humanoid's `<default>` sets condim=1, margin, armature and a
    motor ctrlrange. Merged into the playground's root default those would
    silently re-specify the ramp, the pool walls and the noise panel — PG.1-PG.7
    measure that geometry. They go into a named class instead, reaching the
    humanoid through childclass= and the motors through class=, and touching
    nothing else. `material="geom"` is dropped from the class (it names an asset
    we do not import); the class's own rgba already colours the body.
    """
    root = ET.parse(humanoid_source_xml()).getroot()
    if root.find("compiler").get("angle") != "degree":
        raise RuntimeError(
            "gymnasium's humanoid.xml is no longer in degrees; the radian "
            "conversion in _humanoid_fragments is now wrong. Re-run PG.8.")

    torso = root.find("worldbody/body[@name='torso']")
    n_converted = 0
    for j in torso.iter("joint"):
        if j.get("type") == "hinge" and j.get("range"):
            lo, hi = (math.radians(float(v)) for v in j.get("range").split())
            j.set("range", f"{lo:.12f} {hi:.12f}")
            n_converted += 1
    if n_converted != HUMANOID_NU:
        raise RuntimeError(f"expected {HUMANOID_NU} limited hinges, "
                           f"converted {n_converted}")

    torso.set("pos", "{:.6f} {:.6f} {:.6f}".format(*spawn))
    torso.set("childclass", "humanoid")
    actuator = root.find("actuator")
    for motor in actuator:
        motor.set("class", "humanoid")

    default = ('<default class="humanoid">'
               '<joint armature="1" damping="1" limited="true"/>'
               '<geom conaffinity="1" condim="1" contype="1" margin="0.001" '
               'rgba="0.8 0.6 0.4 1"/>'
               '<motor ctrllimited="true" ctrlrange="-.4 .4"/>'
               '</default>')
    dump = lambda e: ET.tostring(e, encoding="unicode")
    return default, dump(torso), dump(root.find("tendon")), dump(actuator)


def build_mjcf(p: PlaygroundParams, with_humanoid: bool = False) -> str:
    """Emit the playground as MJCF XML.

    Kept as plain string templating rather than dm_control.mjcf: the artifact is
    then a file a person can read, diff and hand to a bug report — the same
    reason the ledger is a JSON file in git.
    """
    rng = np.random.RandomState(p.seed)
    a = p.arena_size

    # ── ramp ────────────────────────────────────────────────────────────
    th = math.radians(p.ramp_angle_deg)
    ramp_len = 2.0
    ramp = (f'<geom name="ramp" type="box" pos="-2.5 2.0 {ramp_len*math.sin(th)/2:.3f}" '
            f'size="{ramp_len/2:.3f} 0.8 0.02" euler="0 {-th:.4f} 0" '
            f'rgba="0.55 0.45 0.35 1" friction="0.9 0.05 0.001"/>')

    # ── stairs ──────────────────────────────────────────────────────────
    stairs = []
    for i in range(p.stair_count):
        h = (i + 1) * p.stair_rise
        stairs.append(
            f'<geom name="stair{i}" type="box" pos="{2.2 + i*p.stair_run:.3f} 2.2 {h/2:.3f}" '
            f'size="{p.stair_run/2:.3f} 0.7 {h/2:.3f}" rgba="0.5 0.5 0.55 1"/>')

    # ── ladder + the apple ──────────────────────────────────────────────
    lx, ly, w = LADDER_X, LADDER_Y, LADDER_HALF_WIDTH
    ladder = [
        f'<geom name="ladder_railL" type="capsule" fromto="{lx-w} {ly} 0 {lx-w} {ly} {p.ladder_height}" size="0.035" rgba="0.6 0.4 0.2 1"/>',
        f'<geom name="ladder_railR" type="capsule" fromto="{lx+w} {ly} 0 {lx+w} {ly} {p.ladder_height}" size="0.035" rgba="0.6 0.4 0.2 1"/>',
    ]
    for i in range(p.ladder_rungs):
        z = (i + 1) * p.ladder_rung_spacing
        if z >= p.ladder_height:
            break
        ladder.append(
            f'<geom name="rung{i}" type="capsule" fromto="{lx-w} {ly} {z:.3f} {lx+w} {ly} {z:.3f}" '
            f'size="0.028" rgba="0.65 0.45 0.25 1"/>')
    # The platform, and the apple on top of it. The apple carries NO reward —
    # it is an object like any other. If Jack climbs for it, that must come from
    # curiosity, not from a number we planted.
    ladder.append(
        f'<geom name="platform" type="box" pos="{lx} {ly+0.45} {p.ladder_height:.3f}" '
        f'size="0.45 0.45 0.03" rgba="0.5 0.35 0.2 1"/>')
    apple = (f'<body name="apple" pos="{lx} {ly+0.45} {p.ladder_height+0.09:.3f}">'
             f'<freejoint/><geom name="apple" type="sphere" size="0.06" mass="0.15" '
             f'rgba="0.85 0.15 0.15 1"/></body>')

    # ── pool: a visual-only box; the physics is the passive callback ─────
    px, py = 2.6, -2.4
    pool = (f'<geom name="pool_water" type="box" pos="{px} {py} {-p.pool_depth/2:.3f}" '
            f'size="{p.pool_size:.3f} {p.pool_size:.3f} {p.pool_depth/2:.3f}" '
            f'rgba="0.2 0.45 0.75 0.35" contype="0" conaffinity="0" group="2"/>')
    # Walls so the basin exists physically; the floor is cut by the pit depth.
    pool_walls = []
    for i, (dx, dy, sx, sy) in enumerate([
            (p.pool_size, 0, 0.05, p.pool_size), (-p.pool_size, 0, 0.05, p.pool_size),
            (0, p.pool_size, p.pool_size, 0.05), (0, -p.pool_size, p.pool_size, 0.05)]):
        pool_walls.append(
            f'<geom name="poolwall{i}" type="box" pos="{px+dx:.3f} {py+dy:.3f} {-p.pool_depth/2:.3f}" '
            f'size="{sx:.3f} {sy:.3f} {p.pool_depth/2:.3f}" rgba="0.4 0.4 0.45 1"/>')
    pool_floor = (f'<geom name="pool_floor" type="box" pos="{px} {py} {-p.pool_depth-0.02:.3f}" '
                  f'size="{p.pool_size:.3f} {p.pool_size:.3f} 0.02" rgba="0.35 0.35 0.4 1"/>')

    # ── objects: the affordance substrate ───────────────────────────────
    objects = []
    for i in range(p.n_objects):
        kind = rng.choice(["sphere", "box", "cylinder"])
        size = rng.uniform(*p.object_size_range)
        mass = rng.uniform(*p.object_mass_range)
        ox, oy = rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.5)
        geom_size = {"sphere": f"{size:.3f}",
                     "box": f"{size:.3f} {size:.3f} {size:.3f}",
                     "cylinder": f"{size:.3f} {size:.3f}"}[kind]
        objects.append(
            f'<body name="obj{i}" pos="{ox:.3f} {oy:.3f} {size+0.05:.3f}"><freejoint/>'
            f'<geom name="obj{i}" type="{kind}" size="{geom_size}" mass="{mass:.3f}" '
            f'friction="{rng.uniform(0.4, 1.2):.3f} 0.05 0.001" '
            f'rgba="{rng.uniform(0.3,0.9):.2f} {rng.uniform(0.3,0.9):.2f} {rng.uniform(0.3,0.9):.2f} 1"/></body>')

    # An immovable welded block: the CONTROL for affordance learning (CU.6).
    # If Jack's affordance model calls this pushable, it learned actions, not
    # interactions.
    objects.append('<geom name="welded_block" type="box" pos="-1.5 -1.5 0.15" '
                   'size="0.15 0.15 0.15" rgba="0.3 0.3 0.3 1"/>')

    # ── seesaw: a dynamic affordance ────────────────────────────────────
    seesaw = ""
    if p.seesaw:
        seesaw = ('<geom name="fulcrum" type="cylinder" pos="-2.5 -0.5 0.1" size="0.08 0.5" '
                  'euler="1.5708 0 0" rgba="0.4 0.4 0.4 1"/>'
                  '<body name="seesaw" pos="-2.5 -0.5 0.22">'
                  '<joint name="seesaw_hinge" type="hinge" axis="0 1 0" damping="0.5"/>'
                  '<geom name="seesaw_plank" type="box" size="1.0 0.25 0.03" mass="2.0" '
                  'rgba="0.6 0.5 0.3 1"/></body>')

    # ── perimeter walls: the nursery is a ROOM ──────────────────────────
    # These earn their place through the noisy-TV fixture: the panel must be a
    # patch of WALL that differs only in texture. Free-floating at the arena
    # edge, its silhouette against empty space is itself irreducibly hard to
    # predict — PG.4 measured a static-texture control fixating at 0.725 dwell
    # on the silhouette alone, which would make the trap measure geometry, not
    # noise. Embedded flush in a wall, the only surprise left is the texture.
    walls = []
    for i, (wx, wy, sx, sy) in enumerate([
            (0, a, a, 0.05), (0, -a, a, 0.05),
            (a, 0, 0.05, a), (-a, 0, 0.05, a)]):
        walls.append(
            f'<geom name="wall{i}" type="box" pos="{wx} {wy} 1.25" '
            f'size="{sx} {sy} 1.25" rgba="0.75 0.73 0.68 1"/>')

    # ── the noisy-TV panel: a fixture, not scenery ──────────────────────
    noise = ""
    if p.noise_panel:
        noise = (f'<geom name="noise_panel" type="box" pos="0 {a-0.1:.2f} 1.2" '
                 f'size="0.9 0.05 0.9" rgba="0.5 0.5 0.5 1" material="noise_mat"/>')

    # ── Jack ────────────────────────────────────────────────────────────
    # Off by default so that PG.1-PG.7's fixtures still compile the world they
    # certified. Every spec about an AGENT must pass True; PG.8 is the gate.
    hum_default = hum_body = hum_tendon = hum_actuator = ""
    if with_humanoid:
        hum_default, hum_body, hum_tendon, hum_actuator = _humanoid_fragments(p.spawn())
        hum_default = f"<default>{hum_default}</default>"

    return f"""<mujoco model="jack_playground">
  <compiler angle="radian" coordinate="local"/>
  {hum_default}
  <option timestep="0.005" gravity="0 0 -{GRAVITY}" integrator="RK4"/>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.1 0.15 0.3" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.3 0.25" rgb2="0.3 0.35 0.3" width="300" height="300"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <texture name="noise_tex" type="2d" builtin="flat" rgb1="0.5 0.5 0.5" width="64" height="64"/>
    <material name="noise_mat" texture="noise_tex"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="{a} {a} 0.1" material="grid_mat" friction="1.0 0.05 0.001"/>
    {ramp}
    {''.join(stairs)}
    {''.join(ladder)}
    {apple}
    {pool}
    {''.join(pool_walls)}
    {pool_floor}
    {''.join(objects)}
    {seesaw}
    {''.join(walls)}
    {noise}
    {hum_body}
  </worldbody>
  {hum_tendon}
  {hum_actuator}
</mujoco>
"""


class Water:
    """Buoyancy and drag for the pool region, applied as a passive force.

    MuJoCo's `density`/`viscosity` are GLOBAL medium properties — set them and
    Jack swims through the air too. So water is applied per-geom, per-step, only
    below the surface:

        F_buoyancy = rho * V_submerged * g   (up)
        F_drag     = -0.5 * rho * Cd * A * |v| * v

    Submerged volume uses the geom's bounding sphere clipped to the surface —
    a spherical-cap approximation. It is not exact and does not need to be: the
    claim under test (PG.2) is that a body of density ratio r floats at depth
    ~r within 10%, which a cap approximation satisfies.
    """

    def __init__(self, model, x: float, y: float, half: float, depth: float,
                 drag_coef: float = 1.2, linear_drag: float = 6.0):
        self.x, self.y, self.half, self.depth = x, y, half, depth
        self.surface_z = 0.0        # pool is dug into the floor plane
        self.cd = drag_coef
        # Linear (viscous/wave-making) damping, 1/s. NOT cosmetic: with
        # quadratic drag alone the damping force vanishes as v->0, so a float
        # never settles — measured, a rho=0.3 sphere was still oscillating at
        # |vz| ~ 0.4-0.9 after 40,000 steps (200 s). Real water damps a bobbing
        # body through wave-making and added mass, which are linear in v at
        # small amplitude. Without this term PG.2's equilibrium-depth claim is
        # not merely inaccurate, it is unmeasurable.
        self.c_lin = linear_drag
        self.enabled = True
        self.model = model
        # Per-body effective radius from GEOM GEOMETRY, precomputed once.
        #
        # The first implementation derived it from body_inertia, which is wrong
        # in a way that inverts the physics: for a sphere I = 2/5 m r^2, so
        # sqrt(I) scales with sqrt(MASS). Denser bodies got a larger inferred
        # radius and therefore MORE buoyancy — measured, a rho=0.8 ball floated
        # at z=+0.21 while a rho=0.3 ball sat at z=-0.03. Geometry must come
        # from geometry.
        self._radius = np.zeros(model.nbody)
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            gtype = model.geom_type[gid]
            gs = model.geom_size[gid]
            if gtype == 2:                       # sphere
                r = gs[0]
            elif gtype in (3, 5):                # capsule, cylinder
                r = (gs[0] ** 2 * max(gs[1], gs[0])) ** (1.0 / 3.0)
            elif gtype == 6:                     # box: equivalent-volume sphere
                r = ((3.0 / (4.0 * math.pi)) * 8.0 * gs[0] * gs[1] * gs[2]) ** (1.0 / 3.0)
            else:
                continue
            self._radius[bid] = max(self._radius[bid], float(r))

    def in_pool(self, pos, radius: float = 0.0) -> bool:
        """Is ANY part of the body in the water?

        The first version tested the body's CENTRE against the surface, which
        silently zeroed buoyancy for every partially-submerged body whose centre
        floated above the waterline — i.e. everything with density ratio < 0.5,
        exactly the things that are supposed to float. Measured: a rho=0.3
        sphere at its correct equilibrium depth received 0.0000 N of buoyancy
        against 41.6 N of weight, so it sank until its centre went under. The
        submerged-fraction maths was right; this gate in front of it was wrong.
        """
        return (abs(pos[0] - self.x) < self.half + radius
                and abs(pos[1] - self.y) < self.half + radius
                and pos[2] - radius < self.surface_z)

    def apply(self, model, data) -> None:
        """Add buoyancy + drag to xfrc_applied. Call every step, before mj_step."""
        if not self.enabled:
            return
        for bid in range(1, model.nbody):
            pos = data.xipos[bid]
            r = float(self._radius[bid])
            if r <= 0 or model.body_mass[bid] <= 0:
                continue
            if not self.in_pool(pos, r):
                continue
            # Spherical-cap submerged fraction.
            d = self.surface_z - pos[2]          # depth of centre below surface
            if d >= r:
                frac = 1.0
            elif d <= -r:
                continue
            else:
                h = d + r
                frac = (h * h * (3 * r - h)) / (4.0 * r ** 3)
            vol = (4.0 / 3.0) * math.pi * r ** 3 * frac
            buoy = WATER_DENSITY * vol * GRAVITY

            vel = data.cvel[bid][3:6]
            speed = float(np.linalg.norm(vel))
            area = math.pi * r * r * frac
            drag = -0.5 * WATER_DENSITY * self.cd * area * speed * vel
            drag = drag - self.c_lin * WATER_DENSITY * vol * vel

            data.xfrc_applied[bid][:3] = drag
            data.xfrc_applied[bid][2] += buoy


def humanoid_body_ids(model) -> list:
    """Body ids of the torso subtree, in id order == Humanoid-v5's body order.

    MuJoCo numbers bodies depth-first in XML order, so a spliced subtree is
    contiguous and internally ordered exactly as it is in the standalone model.
    That is what lets `humanoid_obs` slice cinert/cvel/cfrc_ext by this list and
    reproduce `HumanoidEnv._get_obs` — asserted, not assumed, by PG.8.
    """
    torso = model.body("torso").id
    out = []
    for b in range(model.nbody):
        anc = b
        while anc > 0 and anc != torso:
            anc = int(model.body_parentid[anc])
        if anc == torso:
            out.append(b)
    return out


def humanoid_index(model) -> dict:
    """Where Jack lives inside the playground's shared qpos/qvel/body arrays.

    The playground's other free bodies (apple, obj0-4) and the seesaw hinge
    occupy addresses too, so nothing here may be assumed to start at 0 or 1 the
    way it does in the single-body Humanoid-v5 model.
    """
    root = model.joint("root")
    qadr, dadr = int(root.qposadr[0]), int(root.dofadr[0])
    bodies = humanoid_body_ids(model)
    if len(bodies) != HUMANOID_NBODY:
        raise RuntimeError(f"expected {HUMANOID_NBODY} humanoid bodies, "
                           f"found {len(bodies)}")
    # The subtree's joints must be contiguous from the free root, or the slices
    # below would silently read a seesaw or an apple into Jack's proprioception.
    jids = sorted(int(j) for b in bodies
                  for j in range(int(model.body_jntadr[b]),
                                 int(model.body_jntadr[b]) + int(model.body_jntnum[b]))
                  if model.body_jntnum[b] > 0)
    if jids != list(range(jids[0], jids[0] + HUMANOID_NU + 1)):
        raise RuntimeError("humanoid joints are not contiguous in the model")
    return {"qposadr": qadr, "dofadr": dadr, "bodies": bodies}


def humanoid_obs(model, data) -> np.ndarray:
    """The Humanoid-v5 observation, emitted from inside the playground.

    Byte-for-byte the concatenation `HumanoidEnv._get_obs` builds (gymnasium
    1.1.1), restricted to Jack's own addresses:

        qpos[2:24] 22 | qvel 23 | cinert 130 | cvel 78 | qfrc_actuator[6:] 17
        | cfrc_ext 78   = 348

    Written as a function of the model rather than a copied constant because
    the 376-vs-348 bug (T0.14) came from exactly one constant being copied from
    a version that no longer applied.
    """
    ix = humanoid_index(model)
    q, d, b = ix["qposadr"], ix["dofadr"], ix["bodies"]
    obs = np.concatenate([
        data.qpos[q + 2:q + HUMANOID_NQ],
        data.qvel[d:d + HUMANOID_NV],
        data.cinert[b].flatten(),
        data.cvel[b].flatten(),
        data.qfrc_actuator[d + 6:d + HUMANOID_NV],
        data.cfrc_ext[b].flatten(),
    ])
    if obs.shape[0] != HUMANOID_OBS_DIM:
        raise RuntimeError(f"observation is {obs.shape[0]}, "
                           f"not {HUMANOID_OBS_DIM}")
    return obs


def step(model, data, ctrl=None, frame_skip: int = 5, water: Optional["Water"] = None):
    """Advance the world one DECISION — the playground's only stepping kernel.

    Bare `mujoco.mj_step` is not enough, and the gap is silent. MuJoCo populates
    `data.cfrc_ext` in `mj_rnePostConstraint`, which `mj_step` does not call;
    gymnasium's `MujocoEnv._step_mujoco_simulation` calls it after every
    `frame_skip` block, which is the only reason `HumanoidEnv._get_obs` has
    contact forces in it at all. Every playground caller stepped with bare
    `mj_step`, so **78 of `humanoid_obs`'s 348 columns were identically zero in
    this world** — measured: max |playground obs - Humanoid-v5 obs| on a
    floor-contact state is 114.97 without the call and 0.0061 with it.

    PG.8 could not see it. Its obs-equivalence check compares at z = 4 m, above
    the walls, deliberately contact-free "so cfrc_ext is zero on both sides" —
    the one state in which the 78 columns cannot tell a live channel from a dead
    one. (`LESSONS.md`: an assertion made against a saturated quantity cannot
    fail.) PG.8 now also compares in contact, and the no-`rne` path is its
    control.

    So there is one stepping function, it matches gymnasium's, and anything that
    wants Jack's observation goes through it — "two kernels re-implementing one
    operation is the defect".
    """
    import mujoco

    if ctrl is not None:
        data.ctrl[:] = ctrl
    for _ in range(frame_skip):
        if water is not None:
            water.apply(model, data)
        mujoco.mj_step(model, data)
    mujoco.mj_rnePostConstraint(model, data)


def make_playground(params: Optional[PlaygroundParams] = None,
                    with_water: bool = True, with_humanoid: bool = False):
    """Build the world and return (model, data, water). CPU-only, no rendering."""
    import mujoco

    p = params or PlaygroundParams()
    xml = build_mjcf(p, with_humanoid=with_humanoid)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    water = None
    if with_water:
        water = Water(model, x=2.6, y=-2.4, half=p.pool_size, depth=p.pool_depth)
    return model, data, water
