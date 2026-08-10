"""PS.01 — the drive layer is a real control problem, and a statue loses.

`docs/research/PURPOSE_AND_SCAFFOLDING.md` §2.2-2.3 proposes a homeostatic drive:
energy that depletes with mechanical work and is restored by eating, integrity
that is damaged by impact and heals with rest, and a soft incapacity instead of
a termination. Every number in it is a PROPOSAL, and this spec's own `notes` say
so: *"Every number in 2.2 is a PROPOSAL until this spec replaces it with a
measurement."* Nothing has trained under it and nothing may, until it is shown
to be a control problem at all — a drive that never moves cannot pressure a
policy, and a drive that flatlines in a minute cannot be learned under.

## v2, 2026-08-10 — THE PROBE was redesigned, not the world

Attempt 1 (05:29) and attempt 2 (07:33) are FAILs and stay in the ledger's
history. Attempt 2's three surviving failures were diagnosed as **one defect,
and it was in the instrument**: the probe could not produce the events the gates
were about. `INTEGRATION_QUEUE.md`'s TOP entry is the redesign, cross-checked
against `NEEDS_AND_DEATH.md` §G-B and `LESSONS.md` on 2026-08-10; the four
changes are implemented here.

    clause                measured, attempt 2   why no constant could fix it
    spread_i >= 0.30      2.96e-5               a random policy never climbs, so
                                                it never falls from height, and
                                                never holds still, so rho never
                                                heals — while the SAME integrator
                                                scored 0.161 on a held-out
                                                platform fall. The channel is
                                                live; the probe could not reach it.
    ok_random_survives    0.0                   a random policy is not a forager
                                                (1.0 items in 600 s). Demanding
                                                that flailing beat resting is
                                                demanding kappa = 0.
    ok_statue_starves     4.35e-14              the statue dies at t = 1/b = 600 s
                                                and the window was 600 s. Its
                                                pre-registered failure was
                                                scheduled at the last sample.

1. **The range is measured over a MIXED FIXTURE probe** (`probe="mixed"`):
   repeating cycles of random action, a scripted release from platform height
   (the same drop `_params(fall=True)` builds), and scripted rest. This is a
   fixture probing the INTEGRATOR, not a claim about a policy — it certifies
   that `i` has usable range over behaviours the world ADMITS, which is what
   "the drive is a control problem" means. And the events are gated, not just
   the range: `n_damaging >= 5` and `n_rest_decisions >= 100` are PASS
   conditions, so a probe that failed to exercise the variable is a red entry
   rather than a confident 2.96e-5.
2. **The domination clause compares the statue against a FORAGER FIXTURE**
   (`probe="forager"`): a body acting at the derived duty cycle `D* = 0.217`
   which harvests both floor foods every time they respawn, run through the
   real `DriveLayer`. It must not starve; the statue must. This verifies unit
   (a)'s C2 on the shipped path instead of in arithmetic, and it is the honest
   form of §5 G-B's question — *is the dark room beaten by some behaviour this
   world admits*, not *is it beaten by flailing*. The fixture abstracts
   LOCOMOTION and nothing else (food is placed on him when it respawns, because
   PS.01 runs before anything trains and there is no locomotion controller to
   call); it pays the real drain through the real integrator.
3. **The window strictly contains the control's death.** 3,000 -> 4,500
   decisions (900 s = 1.5 x `1/b`), and `statue_death_s < 0.8 x horizon` is a
   gate. A control designed to fail at the boundary of the window cannot be
   observed failing.
4. **Every drain rate is reported next to the regime it was measured in.**
   Attempt 1 recorded `mean_power_w = 293` and `frac_e_zero = 0.848` in one
   entry and nobody joined them: that power was a STARVING body's, because
   `gear_scale = 0.4 + 0.6*min(e, i)` sat at its 0.4 floor for 85% of the run,
   and §2.3 then exonerated `kappa` on it. v2 records
   `mean_power_w_full_strength` (`e = i` pinned at 1) beside it and prices
   subsistence against THAT, so the confound is a field in the record rather
   than something a reader has to notice.

WHAT IT MEASURES, in the order the numbers depend on each other.

1. **`J_0`** — the threshold below which contact is ordinary and costs nothing.
   §2.2's channel `J_t` was decided by bakeoff on 2026-08-10 (`PS.01/J`,
   `PS.01/J2`, `docs/DECISIONS_RESOLVED.md`): the root's linear SPEED one
   substep before contact onset, 0.973 AUC at separating a platform fall from
   an ordinary collapse, against a force-integral formulation measured at
   chance. `J_0` is the 95th percentile of that quantity over decisions in
   which contact ONSET occurred, in the GROUND regime — Jack at his ordinary
   spawn under a random policy, i.e. exactly the ordinary contact a fall has to
   be told apart from.

   The population is the per-DECISION value, not the per-onset value, because
   the per-decision value is what the integrator compares against `j0`. A
   threshold calibrated on a statistic the shipped path never computes is the
   T0.16 failure mode.

2. **`alpha`** — calibrated so a fall from the ladder platform costs 0.15
   integrity, which is §2.2's own instruction. The fall regime is the
   bakeoff's: released beside the platform at `ladder_height + SPAWN_Z`, so his
   torso falls the full 1.8 m of ladder onto the floor. `alpha` is set from the
   MEDIAN total excess `sum_t max(0, J_t - J_0)` over the whole fall episode,
   not from the first landing alone, because "a fall costs 0.15" is a claim
   about the fall — the bounce and the roll are part of it.

3. **The fall cost, HELD OUT.** `alpha` is derived on calibration runs and
   verified on five FRESH fall runs with different rng, driven through the real
   `DriveLayer` rather than through the arithmetic that produced `alpha`.
   Median `1 - min(i)` must land in [0.10, 0.20]. This is the check that can
   catch a calibration that does not survive clipping, healing, or the
   integrator's own decision boundaries — the difference between the formula
   and the shipped path.

4. **The dynamic range**, the spec's headline metric: 4,500 decisions (900
   simulated seconds) of an unbroken life under the MIXED probe, with `e` and
   `i` logged every decision, plus the event gates of change 1.

5. **Subsistence arithmetic**, §2.3, priced at full strength: floor food alone
   must exceed basal and fall short of the drain of a constantly-acting body,
   `b + kappa * P_bar(1)`.

6. **The statue is dominated.** A do-nothing policy over the same 4,500
   decisions, its death timed and required to land inside the window, against
   the forager fixture of change 2.

THE NULL — the drive integrator disabled — runs ON THE SAME ROLLOUT as the live
arm rather than in a second world. A disabled second layer observes every
substep of the live arm's physics and is reduced by the same percentile code, so
`null_spread == 0` says the reduction reads the layer and not the world. It
costs no extra physics, which is why it can be afforded at all.

THE CONTROL is the do-nothing policy and it MUST fail: best integrity, it must
never reach food, and it must be SEEN to die inside the window. If doing nothing
were survivable indefinitely, the dark room is a stable optimum and no
homeostatic arm could be interpreted (§5 G-B).

WHAT THIS CANNOT DO. It cannot say the drive makes anything learn — that is
LC.03 and the §4 bakeoff. It says the drive is a control problem, and if it is
not, it kills §2.2-2.3's numbers rather than the idea (the spec's `kills`).
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# This spec certifies a property of the WORLD, so the world hashes into
# impl_sha. Change playground.py and this certificate goes stale loudly
# instead of standing over a world it no longer describes.
IMPL_DEPS = ["playground.py"]

REPO = Path(__file__).resolve().parents[2]

# ── the rollout, fixed before the run ───────────────────────────────────
SIM_S_PER_DECISION = 0.2      # w0.py's accounting unit, and the bakeoff's
CTRL_SCALE = 0.4              # the random policy, as scored in PS.01/J and J2
N_DECISIONS = 4500            # v2: 900 simulated seconds = 1.5 x the statue's 1/b
HORIZON_S = N_DECISIONS * SIM_S_PER_DECISION
N_CALIB_DECISIONS = 60        # the bakeoff's: a 1.8 m fall lands in under 1 s
N_CALIB_RUNS = 10             # per regime, per seed — the bakeoff's N_RUNS
N_VERIFY_RUNS = 5             # HELD-OUT fall runs, different rng from calibration
N_POWER_DECISIONS = 400       # ps01_energy.py's window for P_bar, unchanged

# ── the MIXED probe's cycle (v2 change 1) ───────────────────────────────
# One cycle = 75 s: flail, be dropped from the platform, then lie still. Twelve
# cycles fill the 900 s life exactly. The drop is what moves `i` down (nothing a
# random policy does can), the rest is what moves it back up (rho heals only
# below Q_REST), and the random segment is what the ORIGINAL probe was.
CYCLE_RANDOM = 100            # 20 s of random action on the ground
CYCLE_DROP = 75               # 15 s from the release: fall, land, tumble
CYCLE_REST = 200              # 40 s of zero control -> rho heals
MIN_DAMAGING = 5              # the probe must have produced impacts above j0
MIN_REST_DECISIONS = 100      # ...and decisions spent actually at rest
REST_FRACTION = 0.5           # a decision counts as resting if the majority of
                              # its simulated time was below Q_REST

# ── the FORAGER fixture (v2 change 2) ───────────────────────────────────
# D* = (S_f - b) / (kappa * P_bar(1)) — the duty cycle floor food funds, derived
# and printed by `experiments/calibrations/ps01_energy.py` under unit (a)'s C2.
FORAGER_DUTY = 0.217
FLOOR_FOODS = ("obj0", "obj1")
FOOD_SERVE_DZ = 0.30          # dropped onto his torso from 30 cm: a real contact
FOOD_SERVE_DX = 0.12          # ...the two of them apart, so they do not collide

# ── the pre-registered gates (registry.PS.01 hypothesis + falsified_by) ──
SPREAD_MIN = 0.30             # p90 - p10 of e and of i over the life
FALL_COST_TARGET = 0.15       # §2.2: "alpha calibrated so a fall ... costs 0.15"
FALL_COST_LO, FALL_COST_HI = 0.10, 0.20
J0_PERCENTILE = 95.0          # §2.2, verbatim
ALIVE_AT_S = 60.0             # "always flatlines at zero within a minute" is the
                              # falsifier's own timescale, not one invented here
STATUE_DEATH_FRAC = 0.8       # v2 change 3: the control's death must be OBSERVED
NEVER_S = HORIZON_S * 10.0    # sentinel for "e never reached 0" — a real number
                              # rather than an inf, because the ledger is JSON and
                              # a gate that cannot be serialised cannot be audited
POOL_XY = (2.6, -2.4)         # as `playground.make_playground` builds it
POOL_SURFACE_Z = 0.0

_CALIB_CACHE: dict = {}


# ── world construction ──────────────────────────────────────────────────
def _params(seed: int, fall: bool = False):
    """Seed 0 is the nursery default; seeds 1+ take one ACCEL mutate step.

    Same convention as PG.8, and for its reason: three identical worlds under
    three different seed integers would not be three seeds. `mutate` leaves
    `ladder_height` alone, so the fall height — the thing `alpha` is calibrated
    against — is the same 1.8 m in every seed.
    """
    import dataclasses

    import numpy as np
    sys.path.insert(0, str(REPO))
    from playground import (LADDER_X, LADDER_Y, SPAWN_OFFSET_Y, SPAWN_Z,
                            PlaygroundParams)

    p = PlaygroundParams(seed=seed)
    if seed > 0:
        p = p.mutate(np.random.RandomState(seed))
    if fall:
        # Beside the platform (which sits at LADDER_Y + 0.45), at its height, so
        # he free-falls the ladder's full height onto the floor. The bakeoff's
        # FALL regime, unchanged.
        spawn = (LADDER_X, LADDER_Y + 0.45 + 0.9, p.ladder_height + SPAWN_Z)
    else:
        spawn = (LADDER_X, LADDER_Y + SPAWN_OFFSET_Y, SPAWN_Z)
    return dataclasses.replace(p, humanoid_spawn=spawn)


def _drop_point(seed: int) -> tuple:
    """Where the MIXED probe releases him from — the calibration's fall spawn.

    Read off `_params(fall=True)` rather than recomputed, so the drop the range
    is measured over and the drop `alpha` is calibrated against cannot drift
    apart.
    """
    return tuple(_params(seed, fall=True).humanoid_spawn)


def _build(seed: int, fall: bool = False):
    sys.path.insert(0, str(REPO))
    from playground import make_playground

    p = _params(seed, fall)
    model, data, water = make_playground(p, with_water=True, with_humanoid=True)
    pool = (POOL_XY[0], POOL_XY[1], p.pool_size, POOL_SURFACE_Z)
    return model, data, water, pool


class _DisabledDrive:
    """The null: an integrator that is fed every substep and never integrates.

    It rides the live arm's rollout, so its constancy is a statement about the
    integrator rather than about a second, quieter world.
    """

    def __init__(self):
        self.e, self.i, self.w = 1.0, 1.0, 0.0

    def begin_decision(self):
        pass

    def substep(self, model, data, dt):
        pass

    def decide(self):
        return self


# ── the probes ──────────────────────────────────────────────────────────
def _mixed_schedule(n: int) -> list:
    """Per-decision mode for the mixed fixture. `drop` is the release decision;
    the rest of the drop segment is random action, exactly as the calibration's
    fall regime runs."""
    cycle = (["random"] * CYCLE_RANDOM
             + ["drop"] + ["random"] * (CYCLE_DROP - 1)
             + ["rest"] * CYCLE_REST)
    reps = n // len(cycle) + 1
    return (cycle * reps)[:n]


def _release(data, qadr: int, dadr: int, ndof: int, xyz) -> None:
    """Put him back at platform height, upright, at rest. Only the humanoid's
    own addresses are touched — the apple, the objects and the seesaw keep
    whatever state the life has already put them in."""
    data.qpos[qadr:qadr + 3] = xyz
    data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dadr:dadr + ndof] = 0.0


def _floor_food_joints(model) -> dict:
    """qpos/dof addresses of the free joint of each floor food, if it exists."""
    out = {}
    for name in FLOOR_FOODS:
        try:
            bid = int(model.body(name).id)
        except (KeyError, ValueError):
            continue
        jadr = int(model.body_jntadr[bid])
        if jadr < 0 or int(model.body_jntnum[bid]) < 1:
            continue
        out[name] = (int(model.jnt_qposadr[jadr]), int(model.jnt_dofadr[jadr]))
    return out


def _serve(data, joints: dict, torso_id: int, name: str, dx: float) -> None:
    """Place a floor food just above his torso, at rest.

    This is the fixture's ONE abstraction and it is deliberately narrow: it
    stands in for locomotion, which does not exist yet, and for nothing else.
    Eating still has to happen through `DriveLayer.substep`'s real contact test,
    the respawn timer is still the shipped one, and the energy the fixture pays
    for acting is the drain the shipped integrator computes.
    """
    import numpy as np
    qa, da = joints[name]
    p = np.asarray(data.xpos[torso_id])
    data.qpos[qa:qa + 3] = (p[0] + dx, p[1], p[2] + FOOD_SERVE_DZ)
    data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[da:da + 6] = 0.0


# ── one life ────────────────────────────────────────────────────────────
def _life(world_seed: int, rng_seed: int, *, j0: float, alpha: float,
          probe: str, n_decisions: int, fall: bool = False,
          harvest: bool = False) -> dict:
    """Run one unbroken life and return everything measured from it.

    `probe` selects the policy fixture:
      `random`   — random action every decision (the calibration regimes)
      `statue`   — zero control every decision (the CONTROL)
      `mixed`    — v2's range probe: random / released-from-platform / rest
      `forager`  — v2's domination fixture: duty `D*`, floor food harvested
      `power`    — random at duty 1 with `e = i` pinned at 1, for `P_bar(1)`

    `harvest=True` also returns the per-decision impact channel for every
    decision in which a contact ONSET occurred — the population `J_0` is a
    percentile of.
    """
    import mujoco
    import numpy as np
    sys.path.insert(0, str(REPO))
    from experiments import drives
    from playground import HUMANOID_NV, humanoid_index

    model, data, water, pool = _build(world_seed, fall)
    layer = drives.DriveLayer(model, j0=j0, alpha=alpha, pool=pool)
    null = _DisabledDrive()
    rng = np.random.RandomState(rng_seed)
    dt = float(model.opt.timestep)
    frame_skip = max(1, int(round(SIM_S_PER_DECISION / dt)))

    ix = humanoid_index(model)
    qadr, dadr = ix["qposadr"], ix["dofadr"]
    torso_id = int(model.body("torso").id)
    sched = _mixed_schedule(n_decisions) if probe == "mixed" else None
    drop_xyz = _drop_point(world_seed) if probe == "mixed" else None
    joints = _floor_food_joints(model) if probe == "forager" else {}
    due = {name: 0.0 for name in joints}
    served = {name: 0 for name in joints}

    E, I, W, NE, NI = [], [], [], [], []
    js, power_dt, rest_dt, total_dt = [], 0.0, 0.0, 0.0
    n_rest_decisions = n_acted = 0
    t_e_zero_s = NEVER_S
    for k in range(n_decisions):
        if probe == "power":
            # `gear_scale` stays 1.0, so this is the drain the food has to fund
            # rather than the drain of a body the shortfall has already starved.
            layer.state = drives.DriveState(e=1.0, i=1.0, w=layer.state.w)
        mode = sched[k] if sched is not None else None
        if mode == "drop":
            _release(data, qadr, dadr, HUMANOID_NV, drop_xyz)
        if probe == "forager":
            for name in joints:
                if layer.t >= due[name]:
                    _serve(data, joints, torso_id, name,
                           FOOD_SERVE_DX if name == FLOOR_FOODS[0] else -FOOD_SERVE_DX)
                    served[name] += 1

        if probe == "statue":
            act = False
        elif probe == "forager":
            act = bool(rng.random_sample() < FORAGER_DUTY)
        elif probe == "mixed":
            act = mode != "rest"
        else:
            act = True
        n_acted += int(act)

        layer.begin_decision()
        null.begin_decision()
        raw = (rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu) if act
               else np.zeros(model.nu))
        ctrl = raw * layer.gear_scale()          # §2.2's weakness, applied
        n0 = layer.n_onsets
        ate0 = dict(layer.ate_total)
        for _ in range(frame_skip):
            data.ctrl[:model.nu] = ctrl
            water.apply(model, data)
            mujoco.mj_step(model, data)
            layer.substep(model, data, dt)
            null.substep(model, data, dt)
        h = layer.decide()
        null.decide()
        power_dt += layer.last_power_w * layer.last_dt
        rest_dt += layer.last_rest_dt
        total_dt += layer.last_dt
        if layer.last_rest_dt > REST_FRACTION * layer.last_dt:
            n_rest_decisions += 1
        if layer.n_onsets > n0:
            js.append(layer.last_j)
        for name in joints:
            if layer.ate_total[name] > ate0[name]:
                due[name] = layer.t + drives.RESPAWN_S[name]
        if h.e <= 0.0 and t_e_zero_s >= NEVER_S:
            t_e_zero_s = (k + 1) * SIM_S_PER_DECISION
        E.append(h.e); I.append(h.i); W.append(h.w)
        NE.append(null.e); NI.append(null.i)

    E, I, W = np.array(E), np.array(I), np.array(W)
    k60 = min(len(E) - 1, int(round(ALIVE_AT_S / SIM_S_PER_DECISION)) - 1)
    out = {
        "spread_e": float(np.percentile(E, 90) - np.percentile(E, 10)),
        "spread_i": float(np.percentile(I, 90) - np.percentile(I, 10)),
        "e_min": float(E.min()), "i_min": float(I.min()),
        "e_final": float(E[-1]), "i_final": float(I[-1]),
        "e_at_60s": float(E[k60]), "i_at_60s": float(I[k60]),
        "frac_e_zero": float((E <= 0.0).mean()),
        "frac_i_zero": float((I <= 0.0).mean()),
        "frac_e_one": float((E >= 1.0).mean()),
        "w_max": float(W.max()),
        "null_spread_e": float(np.percentile(NE, 90) - np.percentile(NE, 10)),
        "null_spread_i": float(np.percentile(NI, 90) - np.percentile(NI, 10)),
        "mean_power_w": float(power_dt / total_dt) if total_dt else 0.0,
        "rest_frac": float(rest_dt / total_dt) if total_dt else 0.0,
        "n_rest_decisions": int(n_rest_decisions),
        "acted_frac": float(n_acted / n_decisions),
        "t_e_zero_s": float(t_e_zero_s),
        "n_onsets": int(layer.n_onsets),
        "n_damaging": int(sum(1 for v in js if v > j0)),
        "ate_total": int(sum(layer.ate_total.values())),
        "ate_floor": int(layer.ate_total.get("obj0", 0)
                         + layer.ate_total.get("obj1", 0)),
        "ate_apple": int(layer.ate_total.get("apple", 0)),
        "n_served": int(sum(served.values())),
        "all_finite": int(bool(np.isfinite(E).all() and np.isfinite(I).all()
                               and np.isfinite(data.qpos).all())),
        "fall_cost": float(1.0 - I.min()),
    }
    if harvest:
        out["js"] = js
    return out


# ── phase 1+2: calibration, cached per seed ─────────────────────────────
def _calibrate(seed: int) -> dict:
    """Measure `J_0` and `alpha` from the two labelled regimes.

    Cached because `run_spec` calls the experiment and the control once each per
    seed and both need the SAME calibration for that seed — a control run under
    a different `alpha` would not be a control.
    """
    if seed in _CALIB_CACHE:
        return _CALIB_CACHE[seed]
    import numpy as np

    ground = []
    for r in range(N_CALIB_RUNS):
        ground += _life(seed * 101 + r, seed * 9973 + r * 17,
                        j0=float("inf"), alpha=0.0, probe="random",
                        n_decisions=N_CALIB_DECISIONS, fall=False,
                        harvest=True)["js"]
    j0 = float(np.percentile(ground, J0_PERCENTILE))

    firsts, excesses = [], []
    for r in range(N_CALIB_RUNS):
        js = _life(seed * 101 + r, seed * 9973 + r * 17 + 1,
                   j0=float("inf"), alpha=0.0, probe="random",
                   n_decisions=N_CALIB_DECISIONS, fall=True,
                   harvest=True)["js"]
        if not js:
            continue
        firsts.append(js[0])
        excesses.append(sum(max(0.0, v - j0) for v in js))
    excess_med = float(np.median(excesses)) if excesses else 0.0
    alpha = FALL_COST_TARGET / excess_med if excess_med > 0 else 0.0

    out = {
        "j0_ms": j0,
        "alpha": alpha,
        "ground_onset_decisions": len(ground),
        "ground_j_p50": float(np.percentile(ground, 50)),
        "ground_j_max": float(np.max(ground)) if ground else 0.0,
        "fall_first_j_med": float(np.median(firsts)) if firsts else 0.0,
        "fall_excess_med": excess_med,
        "fall_runs_with_contact": len(excesses),
    }
    _CALIB_CACHE[seed] = out
    return out


# ── the experiment ──────────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    import numpy as np

    cal = _calibrate(seed)
    j0, alpha = cal["j0_ms"], cal["alpha"]

    # 3. the fall cost, on runs the calibration never saw, through the real layer
    costs = [_life(seed * 101 + N_CALIB_RUNS + r,
                   seed * 9973 + (N_CALIB_RUNS + r) * 17 + 1,
                   j0=j0, alpha=alpha, probe="random",
                   n_decisions=N_CALIB_DECISIONS, fall=True)["fall_cost"]
             for r in range(N_VERIFY_RUNS)]

    # 4. the life, over the MIXED probe (v2 change 1)
    m = _life(seed, seed * 31 + 7, j0=j0, alpha=alpha, probe="mixed",
              n_decisions=N_DECISIONS)
    m.update(cal)
    m["fall_cost_med"] = float(np.median(costs))
    m["fall_cost_min"] = float(np.min(costs))
    m["fall_cost_max"] = float(np.max(costs))

    # 4b. the drain at FULL STRENGTH (v2 change 4) — the number the supply has
    #     to fund, recorded beside the number the probe happened to produce.
    p = _life(seed, seed * 7717 + 3, j0=j0, alpha=alpha, probe="power",
              n_decisions=N_POWER_DECISIONS)
    m["mean_power_w_full_strength"] = p["mean_power_w"]

    # 5. subsistence, §2.3 — priced against the full-strength drain
    sys.path.insert(0, str(REPO))
    from experiments import drives
    supply = 2.0 * drives.NU_FLOORFOOD / drives.RESPAWN_FLOORFOOD_S
    m["floor_supply_rate"] = supply
    m["basal_rate"] = drives.BASAL_B
    m["active_drain_rate"] = drives.BASAL_B + drives.KAPPA * m["mean_power_w"]
    m["full_strength_drain_rate"] = (drives.BASAL_B
                                     + drives.KAPPA * m["mean_power_w_full_strength"])
    m["drive_dynamic_range"] = min(m["spread_e"], m["spread_i"])

    # 6. the FORAGER fixture (v2 change 2) — the statue's live opponent
    f = _life(seed, seed * 613 + 11, j0=j0, alpha=alpha, probe="forager",
              n_decisions=N_DECISIONS)
    m["forager_e_min"] = f["e_min"]
    m["forager_e_final"] = f["e_final"]
    m["forager_i_min"] = f["i_min"]
    m["forager_ate_floor"] = f["ate_floor"]
    m["forager_power_w"] = f["mean_power_w"]
    m["forager_acted_frac"] = f["acted_frac"]
    m["forager_drain_rate"] = drives.BASAL_B + drives.KAPPA * f["mean_power_w"]
    m["forager_served"] = f["n_served"]

    # per-seed conjunctions, so `_aggregate`'s mean over seeds cannot let one
    # seed's failure hide inside another seed's margin
    m["ok_spread"] = int(m["spread_e"] >= SPREAD_MIN and m["spread_i"] >= SPREAD_MIN)
    m["ok_probe_exercised"] = int(m["n_damaging"] >= MIN_DAMAGING
                                  and m["n_rest_decisions"] >= MIN_REST_DECISIONS)
    m["ok_alive_60s"] = int(m["e_at_60s"] > 0.0 and m["i_at_60s"] > 0.0)
    m["ok_fall_cost"] = int(FALL_COST_LO <= m["fall_cost_med"] <= FALL_COST_HI)
    m["ok_subsistence"] = int(supply >= drives.BASAL_B
                              and supply < m["full_strength_drain_rate"])
    m["ok_null_flat"] = int(m["null_spread_e"] == 0.0 and m["null_spread_i"] == 0.0)
    m["ok_forager_survives"] = int(f["e_min"] > 0.0)
    return m


def _control(seed: int) -> dict:
    """The do-nothing policy. It must fail: best integrity, it never eats, and
    its death must be OBSERVED inside the window rather than scheduled at its
    edge (v2 change 3).

    Same world, same seed, same calibration — only the actions are gone.
    """
    cal = _calibrate(seed)
    m = _life(seed, seed * 31 + 7, j0=cal["j0_ms"], alpha=cal["alpha"],
              probe="statue", n_decisions=N_DECISIONS)
    m.update(cal)
    m["statue_death_s"] = m["t_e_zero_s"]
    m["ok_statue_starves"] = int(m["e_min"] <= 0.0)
    m["ok_statue_never_ate"] = int(m["ate_total"] == 0)
    m["ok_statue_death_observed"] = int(m["t_e_zero_s"]
                                        < STATUE_DEATH_FRAC * HORIZON_S)
    return m


def _check(m: dict, c: dict) -> bool:
    return bool(
        # the rollout is readable at all
        m["all_finite"] == 1
        and c["all_finite"] == 1
        # the calibration produced a channel that separates the regimes
        and m["fall_first_j_med"] > m["j0_ms"]
        and m["alpha"] > 0.0
        # 3. a fall from the platform costs 0.10-0.20 integrity, held out
        and m["ok_fall_cost"] == 1.0
        # 4. both drives traverse a usable range, on every seed — and the probe
        #    must have PRODUCED the events the range is about, or a small
        #    spread is an unmeasured variable rather than an inert one
        and m["ok_probe_exercised"] == 1.0
        and m["ok_spread"] == 1.0
        and m["ok_alive_60s"] == 1.0
        # 5. floor food subsists a partly-active body and does not fund a
        #    constantly-active one, priced at FULL STRENGTH
        and m["ok_subsistence"] == 1.0
        # 6. the statue is strictly dominated: its energy reaches the weakness
        #    floor inside the window and the forager fixture's never does
        and c["ok_statue_starves"] == 1.0
        and c["ok_statue_death_observed"] == 1.0
        and m["ok_forager_survives"] == 1.0
        # the CONTROL must fail on its own pre-registered side
        and c["i_min"] >= m["i_min"]
        and c["ok_statue_never_ate"] == 1.0
        # the NULL: a disabled integrator does not move, on the same physics
        and m["ok_null_flat"] == 1.0
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PS.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
