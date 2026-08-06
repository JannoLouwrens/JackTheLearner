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
  noise panel     MANDATORY FIXTURE, not scenery. A wall whose texture
                  re-randomises every step. A prediction-error agent must get
                  trapped here (PG.4), and every curiosity claim must report its
                  dwell time near it. Without a working trap, "his curiosity is
                  real" is unfalsifiable.

Deliberately NOT here: anything requiring a resident GPU, and any reward
function. The playground has no goals. Goals come from Jack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

WATER_DENSITY = 1000.0      # kg/m^3
GRAVITY = 9.81


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
        )


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
    lx, ly = 0.0, -2.6
    ladder = [
        f'<geom name="ladder_railL" type="capsule" fromto="{lx-0.25} {ly} 0 {lx-0.25} {ly} {p.ladder_height}" size="0.035" rgba="0.6 0.4 0.2 1"/>',
        f'<geom name="ladder_railR" type="capsule" fromto="{lx+0.25} {ly} 0 {lx+0.25} {ly} {p.ladder_height}" size="0.035" rgba="0.6 0.4 0.2 1"/>',
    ]
    for i in range(p.ladder_rungs):
        z = (i + 1) * p.ladder_rung_spacing
        if z >= p.ladder_height:
            break
        ladder.append(
            f'<geom name="rung{i}" type="capsule" fromto="{lx-0.25} {ly} {z:.3f} {lx+0.25} {ly} {z:.3f}" '
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

    # ── the noisy-TV panel: a fixture, not scenery ──────────────────────
    noise = ""
    if p.noise_panel:
        noise = (f'<geom name="noise_panel" type="box" pos="0 {a-0.1:.2f} 1.2" '
                 f'size="0.9 0.05 0.9" rgba="0.5 0.5 0.5 1" material="noise_mat"/>')

    return f"""<mujoco model="jack_playground">
  <compiler angle="radian" coordinate="local"/>
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
    {noise}
  </worldbody>
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


def make_playground(params: Optional[PlaygroundParams] = None, with_water: bool = True):
    """Build the world and return (model, data, water). CPU-only, no rendering."""
    import mujoco

    p = params or PlaygroundParams()
    xml = build_mjcf(p)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    water = None
    if with_water:
        water = Water(model, x=2.6, y=-2.4, half=p.pool_size, depth=p.pool_depth)
    return model, data, water
