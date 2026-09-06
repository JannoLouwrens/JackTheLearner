"""PG.8 — Jack is IN the playground and can act in it.

PG.1-PG.7 all PASS and every one of them is honest. They certify the WORLD:
friction discriminates hold-from-slide 1751x, water floats spheres at the exact
Archimedes depth, contact audio pans to the true bearing, the ladder is
climbable. Then a research agent listed the model's bodies and found
`[world, apple, obj0-4, seesaw]` with `nu = 0`. `build_mjcf` took a
`with_humanoid` argument that was never referenced and that nothing in the repo
ever passed True. PG.3 climbs the ladder with what its own docstring calls "a
certification jig, not a humanoid".

So the ladder stood, the apple sat on top, the pool held water, and there was
nobody there to climb, swim or fall. Seven green fixtures composed into an empty
room, and no amount of scrutinising them individually would have shown it —
each certifies a PROPERTY of the world, none certifies that anything LIVES in
it. That is the whole distance between a green ladder and GOAL.md.

This spec closes it, and it is deliberately a fixture spec: no policy, no
learning, no reward. It asserts that the body Jack will inhabit is (1) present,
(2) the SAME body `TrainingPipeline` trains on, (3) numerically stable in this
world, (4) emitting the observation the pipeline expects, (5) capable of
applying force, and (6) standing within an unobstructed reach of the ladder.

Six pre-registered checks, thresholds fixed before the run (measured margins in
brackets, from the calibration probe on 2026-08-09):

  present     `torso` exists, its subtree is 13 bodies, nu == 17, and all 17
              motors drive humanoid joints.                          [exact]
  fidelity    body_mass, body_inertia, jnt_range, actuator_gear and
              actuator_ctrlrange agree with `gym.make("Humanoid-v5")` to
              <= 1e-9. This is the real guard: gymnasium's humanoid.xml is in
              DEGREES and this world is in RADIANS, so a verbatim splice turns
              a -160..-2 deg knee into an effectively unlimited joint with
              nothing erroring — the MJCF-degrees bug that broke PG.1's ramp,
              one level up. The conversion lives in playground._humanoid_
              fragments; this check is what makes it a claim.       [dev 0.0]
  observation `humanoid_obs` is 348-long, equals PipelineConfig.mujoco_obs_dim,
              and is bit-equivalent (<= 1e-9) to `HumanoidEnv._get_obs` on a
              matched, contact-free state — AND agrees to <= 0.05 on a matched
              state IN CONTACT, where the 78 cfrc_ext columns are nonzero.
              The contact half was added 2026-08-09 and it found a real defect:
              cfrc_ext is filled by `mj_rnePostConstraint`, `mj_step` does not
              call it, and no playground caller did — so 78 of the 348 columns
              were identically zero in this world (dev 114.97 without the call,
              0.0061 with). The contact-free state was the one place those
              columns could not tell live from dead. A dimension count alone
              would pass on a permuted or wrongly-sliced vector, and the
              playground shares qpos with the apple, five objects and the
              seesaw, so slicing is exactly where this would go wrong.
                                       [dev 4.7e-16 free; 0.0061 vs 115 contact]
  settles     2000 steps (10 s) at zero control: states stay finite, MuJoCo
              raises no warning, and |qvel| on Jack's dofs falls below 0.5.
              He falls over — that is correct, Humanoid-v5 spawns standing and
              unbalanced — the claim is that the physics stays sane.
                                                             [qvel 0.011, 45x]
  actuated    from the settled state, a fixed +-0.4 drive on all 17 motors
              moves his qpos by >= 0.10 rad more than the zero-control
              continuation of the SAME state.                 [2.45-2.71, 24x]
  reachable   after settling, the horizontal distance to the ladder base is
              <= 1.5 m AND a ray from the torso to the nearest rung hits a
              ladder geom first — nothing is in the way.  [1.04 m, hits rung0]

CONTROL (the spec's declared one): the identical humanoid spawned OUTSIDE the
arena must fail `reachable` — both halves. Without it "reachable" could be an
assertion about the ladder's coordinates that holds wherever Jack is.

NULL: the playground exactly as it stood this morning, `with_humanoid=False`.
It must fail every check — no torso, nu == 0, no observation. It is carried in
the experiment metrics rather than in the control because the control tests the
reach metric and the null tests the whole spec; conflating them would let one
of the two go unmeasured.

Seeds vary the WORLD and the initial pose, not just the RNG label: seed 0 is the
nursery default, seeds 1+ take one ACCEL-style `mutate` step, and every seed
applies Humanoid-v5's own +-0.01 reset noise to qpos/qvel. Three identical runs
under three different seed integers would not be three seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# This spec certifies a property of the WORLD, so the world hashes into
# impl_sha. Change playground.py and this certificate goes stale loudly
# instead of standing over a world it no longer describes.
IMPL_DEPS = ["playground.py", "TrainingPipeline.py"]  # TrainingPipeline undeclared until 2026-09-06 (78th audit finding 1.1)

REPO = Path(__file__).resolve().parents[2]

MODEL_DEV_MAX = 1e-9        # playground Jack vs gym.make("Humanoid-v5")
OBS_DEV_MAX = 1e-9          # humanoid_obs vs HumanoidEnv._get_obs, no contact
# ── STRENGTHENED 2026-08-09 (T1.02 precedent: strengthen only) ──────────
# The contact-free comparison above is the only state in which cfrc_ext is zero
# on BOTH sides, so the 78 columns it contributes were compared 0 == 0 and could
# not distinguish a live channel from a dead one. They WERE dead: MuJoCo fills
# cfrc_ext in `mj_rnePostConstraint`, which `mj_step` does not call and every
# playground caller therefore never called. Measured on a floor-contact state:
# 114.97 deviation without the call, 0.00611 with it. The residual 0.00611 is
# real physics — the playground floor declares friction 1/0.05/0.001 (PG.1
# measures that) against Humanoid-v5's 1/0.1/0.1, and the contact sets differ 10
# vs 9 — so the threshold is set an order of magnitude above the measured
# residual and four below the broken path.
CONTACT_OBS_DEV_MAX = 0.05
CONTACT_OBS_DEV_MIN_BROKEN = 1.0   # the no-rne control must blow past this
SETTLE_STEPS = 2000         # 10 s at the playground's 0.005 timestep
SETTLE_QVEL_MAX = 0.5       # rad/s or m/s on Jack's dofs
DRIVE_DIVERGENCE_MIN = 0.10  # rad of qpos, driven vs zero-control
REACH_DIST_MAX = 1.5        # m, torso to ladder base, horizontal
RESET_NOISE = 0.01          # Humanoid-v5's own reset_noise_scale
OUTSIDE_SPAWN = (8.0, 8.0, 1.4)   # beyond the arena walls (MuJoCo planes are
                                  # infinite in collision, so he still lands on
                                  # a floor and settles — only reach changes)


def _params(seed: int):
    sys.path.insert(0, str(REPO))
    import numpy as np
    from playground import PlaygroundParams

    p = PlaygroundParams(seed=seed)
    if seed > 0:
        p = p.mutate(np.random.RandomState(seed))
    return p


def _reset(model, data, seed: int):
    """Humanoid-v5's reset: qpos0/qvel0 plus uniform +-reset_noise_scale."""
    import mujoco
    import numpy as np
    from playground import humanoid_index

    mujoco.mj_resetData(model, data)
    ix = humanoid_index(model)
    q, d = ix["qposadr"], ix["dofadr"]
    rng = np.random.RandomState(1000 + seed)
    data.qpos[q:q + 24] += rng.uniform(-RESET_NOISE, RESET_NOISE, 24)
    data.qvel[d:d + 23] += rng.uniform(-RESET_NOISE, RESET_NOISE, 23)
    mujoco.mj_forward(model, data)
    return ix


def _nearest_rung(model, data, z: float):
    """(geom id, xpos) of the rung closest to height z. Rungs span the rails,
    so their centre is a point ON the ladder — the rails themselves are at
    x = +-0.25 and a ray at the ladder's centre-line passes clean between them.
    """
    import numpy as np

    best = None
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith("rung"):
            dz = abs(float(data.geom_xpos[g][2]) - z)
            if best is None or dz < best[0]:
                best = (dz, g, np.array(data.geom_xpos[g], dtype=np.float64))
    return best[1], best[2]


def _reach(model, data) -> dict:
    """Horizontal distance to the ladder base, and what blocks the line to it.

    The ray starts 0.45 m along its own direction so it leaves Jack's own body
    (torso capsule + arms span roughly 0.35 m) instead of instantly hitting his
    chest; mj_ray's bodyexclude takes one body and he is thirteen.
    """
    import mujoco
    import numpy as np
    from playground import LADDER_X, LADDER_Y

    torso = model.body("torso").id
    src = np.array(data.xpos[torso], dtype=np.float64)
    dist = float(np.hypot(src[0] - LADDER_X, src[1] - LADDER_Y))

    _, rung = _nearest_rung(model, data, float(src[2]))
    vec = rung - src
    vec = vec / float(np.linalg.norm(vec))
    gid = np.zeros(1, dtype=np.int32)
    mujoco.mj_ray(model, data, src + 0.45 * vec, vec, None, 1, -1, gid)
    hit = model.geom(int(gid[0])).name if int(gid[0]) >= 0 else ""
    return {
        "dist_to_ladder_m": round(dist, 4),
        "ray_first_hit": hit,
        "ray_hits_ladder": int(hit.startswith("rung")
                               or hit.startswith("ladder_rail")),
    }


def _obs_equivalence(model, data, seed: int) -> float:
    """Max |playground obs - Humanoid-v5 obs| on a matched contact-free state.

    Lifted to z = 4 m: above the walls, so cfrc_ext is zero on both sides and
    the comparison is of the state-derived 270 columns, not of two different
    contact sets. The xy offset is deliberately NOT matched — Jack stands
    somewhere else in a bigger room, and the observation must not care.
    """
    import gymnasium as gym
    import mujoco
    import numpy as np
    from playground import humanoid_index, humanoid_obs

    env = gym.make("Humanoid-v5")
    try:
        rm, rd = env.unwrapped.model, env.unwrapped.data
        mujoco.mj_resetData(rm, rd)
        rng = np.random.RandomState(seed)
        rd.qpos[:] = rm.qpos0
        rd.qpos[2] = 4.0
        rd.qpos[7:] = rng.uniform(-0.2, 0.2, rm.nq - 7)
        rd.qvel[:] = rng.uniform(-0.3, 0.3, rm.nv)
        mujoco.mj_forward(rm, rd)
        ref = env.unwrapped._get_obs()

        ix = humanoid_index(model)
        q, d = ix["qposadr"], ix["dofadr"]
        mujoco.mj_resetData(model, data)
        data.qpos[q:q + 24] = rd.qpos
        data.qpos[q] += 1.7                 # a different place in the room
        data.qpos[q + 1] -= 2.3
        data.qvel[d:d + 23] = rd.qvel
        mujoco.mj_forward(model, data)
        return float(np.abs(humanoid_obs(model, data) - ref).max())
    finally:
        env.close()


def _contact_obs_equivalence(model, data, seed: int) -> dict:
    """The same comparison, IN CONTACT — where the 78 cfrc_ext columns are live.

    Settles gymnasium's own Humanoid on its own floor, copies that state into the
    playground (offset in xy, because the observation must not care where in the
    room he is), and compares `humanoid_obs` against `_get_obs` twice: once
    exactly as every playground caller used to produce it (`mj_forward` only) and
    once through the shared kernel. The first IS the control — it is not a
    tidied restatement of the bug, it is the bug, and it must fail.
    """
    import gymnasium as gym
    import mujoco
    import numpy as np
    from playground import humanoid_index, humanoid_obs

    env = gym.make("Humanoid-v5")
    try:
        rm, rd = env.unwrapped.model, env.unwrapped.data
        mujoco.mj_resetData(rm, rd)
        rng = np.random.RandomState(500 + seed)
        rd.qpos[:] = rm.qpos0
        rd.qpos[7:] += rng.uniform(-RESET_NOISE, RESET_NOISE, rm.nq - 7)
        for _ in range(600):                    # 3 s: he falls and lies in contact
            rd.ctrl[:] = 0.0
            mujoco.mj_step(rm, rd)
        mujoco.mj_rnePostConstraint(rm, rd)
        ref = env.unwrapped._get_obs()

        ix = humanoid_index(model)
        q, d = ix["qposadr"], ix["dofadr"]
        mujoco.mj_resetData(model, data)
        data.qpos[q:q + 24] = rd.qpos
        data.qvel[d:d + 23] = rd.qvel
        # a different place in the room, and one with nothing else in it
        cx, cy = _clear_floor_spot(model, data, q)
        broken = float(np.abs(humanoid_obs(model, data) - ref).max())
        mujoco.mj_rnePostConstraint(model, data)
        fixed = humanoid_obs(model, data)
        from playground import humanoid_body_ids
        non_floor = _foreign_contacts(model, data, set(humanoid_body_ids(model)),
                                      int(model.geom("floor").id))
        return {
            "contact_ncon_ref": int(rd.ncon),
            "contact_ncon_pg": int(data.ncon),
            "contact_non_floor_pg": non_floor,
            "contact_spot_xy": [round(cx, 2), round(cy, 2)],
            "contact_cfrc_norm_ref": round(float(np.linalg.norm(rd.cfrc_ext)), 4),
            "contact_obs_dev": round(float(np.abs(fixed - ref).max()), 6),
            "contact_obs_dev_no_rne": round(broken, 4),
        }
    finally:
        env.close()


def _foreign_contacts(model, data, jack: set, floor: int) -> int:
    """Contacts between Jack and something that is neither Jack nor the floor.

    The obvious "any contact not involving the floor" counts two things that are
    not defects: `apple` resting on `platform` (two OTHER bodies, nothing to do
    with him) and `left_foot`/`right_foot` against `butt`, which are Jack's own
    self-collisions and are present in Humanoid-v5 too. Counting those made
    every one of 625 candidate spots look contaminated and the search fell back
    to the arena centre, i.e. into the ladder. The quantity that matters is
    FOREIGN contact.
    """
    n = 0
    for k in range(int(data.ncon)):
        g1, g2 = int(data.contact[k].geom1), int(data.contact[k].geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        j1, j2 = b1 in jack, b2 in jack
        if j1 == j2:                       # both his, or neither his
            continue
        other = g2 if j1 else g1
        if other != floor:
            n += 1
    return n


def _clear_floor_spot(model, data, q: int) -> tuple:
    """Place Jack's already-set pose where the ONLY thing he touches is the floor.

    Not decoration. The first version of this check offset him by a fixed
    (+1.7, -2.3) and seed 1 — a MUTATED world — dropped an object there, so his
    contact set was 10 against Humanoid-v5's 8 and the deviation read 2.24
    instead of 0.004. That is a real world difference, not a defect in
    `humanoid_obs`, and gating on it would have failed the check for a reason it
    does not test. Every seed mutates the furniture, so the spot cannot be
    written down.

    It is searched rather than computed from geom sizes because the property
    that matters — "no contact except the floor" — is one MuJoCo can answer
    exactly, while any bounding-box estimate of it is an approximation that
    would silently drift as new geometry is added. `contact_non_floor_pg` is
    gated, so a world with nowhere clear left is a red ledger entry rather than
    a quietly contaminated comparison.
    """
    import mujoco
    import numpy as np

    from playground import humanoid_body_ids

    floor = int(model.geom("floor").id)
    jack = set(humanoid_body_ids(model))
    half = float(model.geom_size[floor][0]) - 1.0
    grid = np.linspace(-half, half, 25)
    # Search outward from the arena centre: the furniture clusters there, but a
    # spot flush against a wall is worse, so nearest-first with a hard reject is
    # the right order.
    cands = sorted(((float(x), float(y)) for x in grid for y in grid),
                   key=lambda xy: xy[0] ** 2 + xy[1] ** 2)
    best = cands[0]
    for x, y in cands:
        data.qpos[q], data.qpos[q + 1] = x, y
        mujoco.mj_forward(model, data)
        if _foreign_contacts(model, data, jack, floor) == 0 and int(data.ncon) > 0:
            return x, y
    data.qpos[q], data.qpos[q + 1] = best
    mujoco.mj_forward(model, data)
    return best


def _model_fidelity(model) -> float:
    """Max deviation of Jack's compiled constants from Humanoid-v5's own.

    Six arrays, chosen because each is destroyed by a different splicing
    mistake: jnt_range by the degree/radian bug, dof_armature and jnt_stiffness
    by a `<default>` that failed to reach the subtree, body_mass/body_inertia
    by a lost `inertiafromgeom`, gear and ctrlrange by a motor default landing
    in the wrong class.
    """
    import gymnasium as gym
    import numpy as np
    from playground import humanoid_body_ids, humanoid_index

    env = gym.make("Humanoid-v5")
    try:
        rm = env.unwrapped.model
        bodies = humanoid_body_ids(model)
        d = humanoid_index(model)["dofadr"]
        jids = [model.joint(rm.joint(k).name).id for k in range(rm.njnt)]
        aids = [model.actuator(rm.actuator(k).name).id for k in range(rm.nu)]
        pairs = [
            (model.body_mass[bodies], rm.body_mass[1:]),
            (model.body_inertia[bodies], rm.body_inertia[1:]),
            (model.jnt_range[jids], rm.jnt_range),
            (model.jnt_stiffness[jids], rm.jnt_stiffness),
            (model.dof_armature[d:d + 23], rm.dof_armature),
            (model.dof_damping[d:d + 23], rm.dof_damping),
            (model.actuator_gear[aids], rm.actuator_gear),
            (model.actuator_ctrlrange[aids], rm.actuator_ctrlrange),
        ]
        return max(float(np.abs(np.asarray(a) - np.asarray(b)).max())
                   for a, b in pairs)
    finally:
        env.close()


def _run(seed: int, spawn=None) -> dict:
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    from dataclasses import replace

    from playground import (HUMANOID_NBODY, HUMANOID_NU, HUMANOID_OBS_DIM,
                            humanoid_body_ids, humanoid_obs, make_playground)
    from playground import step as pg_step
    from TrainingPipeline import PipelineConfig

    p = _params(seed)
    if spawn is not None:
        p = replace(p, humanoid_spawn=spawn)
    model, data, _ = make_playground(p, with_humanoid=True)

    bodies = set(humanoid_body_ids(model))
    motors_on_humanoid = sum(
        1 for a in range(model.nu)
        if int(model.jnt_bodyid[int(model.actuator_trnid[a][0])]) in bodies)

    ix = _reset(model, data, seed)
    q, d = ix["qposadr"], ix["dofadr"]
    spawn_contacts = sum(
        1 for i in range(int(data.ncon))
        if int(model.geom_bodyid[data.contact[i].geom1]) in bodies
        or int(model.geom_bodyid[data.contact[i].geom2]) in bodies)

    obs = humanoid_obs(model, data)
    out = {
        "has_humanoid": int(any(model.body(b).name == "torso"
                                for b in range(model.nbody))),
        "n_humanoid_bodies": len(bodies),
        "spawn_xyz": [round(v, 3) for v in p.spawn()],
        "nu": int(model.nu),
        "motors_on_humanoid": motors_on_humanoid,
        "obs_dim": int(obs.shape[0]),
        "pipeline_obs_dim": int(PipelineConfig().mujoco_obs_dim),
        "obs_max_dev_vs_v5": _obs_equivalence(model, data, seed),
        **_contact_obs_equivalence(model, data, seed),
        "model_max_dev_vs_v5": _model_fidelity(model),
        "spawn_contacts": spawn_contacts,
        "world_mutated": int(seed > 0),
    }

    # ── settle: 10 s of nothing, and the physics must stay sane ─────────
    # Through the shared kernel (playground.step), not a bare mj_step loop, so
    # `settle_obs_finite` below reads an observation whose contact columns are
    # actually populated. Same 2000 physics steps; 400 decisions of frame_skip 5.
    ix = _reset(model, data, seed)
    for _ in range(SETTLE_STEPS // 5):
        pg_step(model, data, frame_skip=5)
    out.update({
        "settle_finite": int(bool(np.isfinite(data.qpos).all()
                                  and np.isfinite(data.qvel).all())),
        "settle_qvel_max": round(float(np.abs(data.qvel[d:d + 23]).max()), 5),
        "mujoco_warnings": int(np.asarray(data.warning.number).sum()),
        "settle_obs_finite": int(bool(np.isfinite(humanoid_obs(model, data)).all())),
    })
    out.update(_reach(model, data))

    # ── actuated: same settled state, driven vs left alone ──────────────
    snap = (data.qpos.copy(), data.qvel.copy(), data.qacc_warmstart.copy())

    def roll(ctrl):
        d2 = mujoco.MjData(model)
        d2.qpos[:], d2.qvel[:], d2.qacc_warmstart[:] = snap
        mujoco.mj_forward(model, d2)
        d2.ctrl[:] = ctrl
        for _ in range(200):
            mujoco.mj_step(model, d2)
        return d2.qpos[q:q + 24].copy(), float(np.abs(d2.qfrc_actuator).max())

    drive = np.random.RandomState(seed).choice([-0.4, 0.4], size=model.nu)
    driven, drive_force = roll(drive)
    idle, idle_force = roll(np.zeros(model.nu))
    out.update({
        "drive_divergence_rad": round(float(np.abs(driven - idle).max()), 5),
        "drive_qfrc_max": round(drive_force, 3),
        "idle_qfrc_max": round(idle_force, 6),
    })
    return out


def _null(seed: int) -> dict:
    """The playground exactly as it stood before PG.8: nobody is in it."""
    sys.path.insert(0, str(REPO))
    from playground import make_playground

    model, _, _ = make_playground(_params(seed), with_humanoid=False)
    names = {model.body(b).name for b in range(model.nbody)}
    return {
        "null_nu": int(model.nu),
        "null_has_humanoid": int("torso" in names),
        "null_bodies": sorted(names),
    }


def _experiment(seed: int) -> dict:
    m = _run(seed)
    m.update(_null(seed))
    return m


def _control(seed: int) -> dict:
    """Same Jack, spawned across the room. `reachable` must fail, both halves.

    Everything else about him is unchanged, so a control that also passed
    would prove the reach metric reads the ladder's coordinates rather than
    his position.
    """
    return _run(seed, spawn=OUTSIDE_SPAWN)


def _check(m: dict, c: dict) -> bool:
    return (
        # present
        m["has_humanoid"] == 1
        and m["n_humanoid_bodies"] == 13
        and m["nu"] == 17
        and m["motors_on_humanoid"] == 17
        # fidelity — the degrees/radians guard
        and m["model_max_dev_vs_v5"] <= MODEL_DEV_MAX
        # observation
        and m["obs_dim"] == 348
        and m["obs_dim"] == m["pipeline_obs_dim"]
        and m["obs_max_dev_vs_v5"] <= OBS_DEV_MAX
        # observation, IN CONTACT — the 78 cfrc_ext columns must be live, and
        # the path that left them dead must be visibly worse
        and m["contact_cfrc_norm_ref"] > 0.0
        and m["contact_non_floor_pg"] == 0
        and m["contact_obs_dev"] <= CONTACT_OBS_DEV_MAX
        and m["contact_obs_dev_no_rne"] >= CONTACT_OBS_DEV_MIN_BROKEN
        # settles
        and m["spawn_contacts"] == 0
        and m["settle_finite"] == 1
        and m["settle_obs_finite"] == 1
        and m["mujoco_warnings"] == 0
        and m["settle_qvel_max"] <= SETTLE_QVEL_MAX
        # actuated
        and m["drive_divergence_rad"] >= DRIVE_DIVERGENCE_MIN
        and m["drive_qfrc_max"] > 0.0
        # reachable
        and m["dist_to_ladder_m"] <= REACH_DIST_MAX
        and m["ray_hits_ladder"] == 1
        # the null: the world as it was must fail everything
        and m["null_nu"] == 0
        and m["null_has_humanoid"] == 0
        # the control: reach must be a statement about where he is
        and c["dist_to_ladder_m"] > REACH_DIST_MAX
        and c["ray_hits_ladder"] == 0
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.8"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
