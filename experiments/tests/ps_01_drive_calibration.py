"""PS.01 — the drive layer is a real control problem, and a statue loses.

`PURPOSE_AND_SCAFFOLDING.md` §2.2-2.3 proposes a homeostatic drive: energy that
depletes with mechanical work and is restored by eating, integrity that is
damaged by impact and heals with rest, and a soft incapacity instead of a
termination. Every number in it is a PROPOSAL, and this spec's own `notes` say
so: *"Every number in 2.2 is a PROPOSAL until this spec replaces it with a
measurement."* Nothing has trained under it and nothing may, until it is shown
to be a control problem at all — a drive that never moves cannot pressure a
policy, and a drive that flatlines in a minute cannot be learned under.

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

4. **The dynamic range**, the spec's headline metric: 3,000 decisions (600
   simulated seconds) of an unbroken life under a random policy, with `e` and
   `i` logged every decision. The registry pre-registers `p90 - p10 >= 0.3` for
   both, and the falsifier pre-registers the two ways that can be uninteresting
   — a drive that never depletes, and one that flatlines at zero within a
   minute.

5. **Subsistence arithmetic**, §2.3: two floor foods supply
   `2 x 0.08 / 90 = 1.78e-3` energy per second against a basal drain of
   `1.67e-3`, so resting on the floor is survivable and acting is not. Checked
   against the MEASURED active drain rate (`b + kappa * P`, with `P` the mean
   mechanical power this rollout actually produced), not against §2.2's
   parenthetical guess at what activity costs.

6. **The statue is dominated.** A do-nothing policy over the same 3,000
   decisions. Pre-registered: its energy reaches the weakness floor while an
   active random policy's does not.

THE NULL — the drive integrator disabled — runs ON THE SAME ROLLOUT as the
random arm rather than in a second world. A disabled second layer observes
every substep of the live arm's physics and is reduced by the same percentile
code, so `null_spread == 0` says the reduction reads the layer and not the
world. It costs no extra physics, which is why it can be afforded at all.

THE CONTROL is the do-nothing policy and it MUST fail: best integrity, and it
must never reach food. If doing nothing were survivable indefinitely, the dark
room is a stable optimum and no homeostatic arm could be interpreted (§5 G-B).

WHAT THIS CANNOT DO. It cannot say the drive makes anything learn — that is
LC.03 and the §4 bakeoff. It says the drive is a control problem, and if it is
not, it kills §2.2-2.3's numbers rather than the idea (the spec's `kills`).
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]

# ── the rollout, fixed before the run ───────────────────────────────────
SIM_S_PER_DECISION = 0.2      # w0.py's accounting unit, and the bakeoff's
CTRL_SCALE = 0.4              # the random policy, as scored in PS.01/J and J2
N_DECISIONS = 3000            # the registry's own number: 600 simulated seconds
N_CALIB_DECISIONS = 60        # the bakeoff's: a 1.8 m fall lands in under 1 s
N_CALIB_RUNS = 10             # per regime, per seed — the bakeoff's N_RUNS
N_VERIFY_RUNS = 5             # HELD-OUT fall runs, different rng from calibration

# ── the pre-registered gates (registry.PS.01 hypothesis + falsified_by) ──
SPREAD_MIN = 0.30             # p90 - p10 of e and of i over the 3000 decisions
FALL_COST_TARGET = 0.15       # §2.2: "alpha calibrated so a fall ... costs 0.15"
FALL_COST_LO, FALL_COST_HI = 0.10, 0.20
J0_PERCENTILE = 95.0          # §2.2, verbatim
ALIVE_AT_S = 60.0             # "always flatlines at zero within a minute" is the
                              # falsifier's own timescale, not one invented here
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


# ── one life ────────────────────────────────────────────────────────────
def _life(world_seed: int, rng_seed: int, *, j0: float, alpha: float,
          statue: bool, n_decisions: int, fall: bool = False,
          harvest: bool = False) -> dict:
    """Run one unbroken life and return everything measured from it.

    `harvest=True` also returns the per-decision impact channel for every
    decision in which a contact ONSET occurred — the population `J_0` is a
    percentile of.
    """
    import mujoco
    import numpy as np
    sys.path.insert(0, str(REPO))
    from experiments import drives

    model, data, water, pool = _build(world_seed, fall)
    layer = drives.DriveLayer(model, j0=j0, alpha=alpha, pool=pool)
    null = _DisabledDrive()
    rng = np.random.RandomState(rng_seed)
    dt = float(model.opt.timestep)
    frame_skip = max(1, int(round(SIM_S_PER_DECISION / dt)))

    E, I, W, NE, NI = [], [], [], [], []
    js, power_dt, rest_dt, total_dt = [], 0.0, 0.0, 0.0
    for _ in range(n_decisions):
        layer.begin_decision()
        null.begin_decision()
        raw = (np.zeros(model.nu) if statue
               else rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu))
        ctrl = raw * layer.gear_scale()          # §2.2's weakness, applied
        n0 = layer.n_onsets
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
        if layer.n_onsets > n0:
            js.append(layer.last_j)
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
        "n_onsets": int(layer.n_onsets),
        "n_damaging": int(sum(1 for v in js if v > j0)),
        "ate_total": int(sum(layer.ate_total.values())),
        "ate_floor": int(layer.ate_total.get("obj0", 0)
                         + layer.ate_total.get("obj1", 0)),
        "ate_apple": int(layer.ate_total.get("apple", 0)),
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
                        j0=float("inf"), alpha=0.0, statue=False,
                        n_decisions=N_CALIB_DECISIONS, fall=False,
                        harvest=True)["js"]
    j0 = float(np.percentile(ground, J0_PERCENTILE))

    firsts, excesses = [], []
    for r in range(N_CALIB_RUNS):
        js = _life(seed * 101 + r, seed * 9973 + r * 17 + 1,
                   j0=float("inf"), alpha=0.0, statue=False,
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
                   j0=j0, alpha=alpha, statue=False,
                   n_decisions=N_CALIB_DECISIONS, fall=True)["fall_cost"]
             for r in range(N_VERIFY_RUNS)]

    # 4. the life
    m = _life(seed, seed * 31 + 7, j0=j0, alpha=alpha, statue=False,
              n_decisions=N_DECISIONS)
    m.update(cal)
    m["fall_cost_med"] = float(np.median(costs))
    m["fall_cost_min"] = float(np.min(costs))
    m["fall_cost_max"] = float(np.max(costs))

    # 5. subsistence, §2.3 — measured against the drain this rollout produced
    sys.path.insert(0, str(REPO))
    from experiments import drives
    supply = 2.0 * drives.NU_FLOORFOOD / drives.RESPAWN_FLOORFOOD_S
    m["floor_supply_rate"] = supply
    m["basal_rate"] = drives.BASAL_B
    m["active_drain_rate"] = drives.BASAL_B + drives.KAPPA * m["mean_power_w"]
    m["drive_dynamic_range"] = min(m["spread_e"], m["spread_i"])

    # per-seed conjunctions, so `_aggregate`'s mean over seeds cannot let one
    # seed's failure hide inside another seed's margin
    m["ok_spread"] = int(m["spread_e"] >= SPREAD_MIN and m["spread_i"] >= SPREAD_MIN)
    m["ok_alive_60s"] = int(m["e_at_60s"] > 0.0 and m["i_at_60s"] > 0.0)
    m["ok_fall_cost"] = int(FALL_COST_LO <= m["fall_cost_med"] <= FALL_COST_HI)
    m["ok_subsistence"] = int(supply >= drives.BASAL_B
                              and supply < m["active_drain_rate"])
    m["ok_null_flat"] = int(m["null_spread_e"] == 0.0 and m["null_spread_i"] == 0.0)
    m["ok_random_survives"] = int(m["e_min"] > 0.0)
    return m


def _control(seed: int) -> dict:
    """The do-nothing policy. It must fail: best integrity, and it never eats.

    Same world, same seed, same calibration — only the actions are gone.
    """
    cal = _calibrate(seed)
    m = _life(seed, seed * 31 + 7, j0=cal["j0_ms"], alpha=cal["alpha"],
              statue=True, n_decisions=N_DECISIONS)
    m.update(cal)
    m["ok_statue_starves"] = int(m["e_min"] <= 0.0)
    m["ok_statue_never_ate"] = int(m["ate_total"] == 0)
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
        # 4. both drives traverse a usable range, on every seed
        and m["ok_spread"] == 1.0
        and m["ok_alive_60s"] == 1.0
        # 5. floor food subsists at rest and not in activity
        and m["ok_subsistence"] == 1.0
        # 6. the statue is strictly dominated: its energy reaches the weakness
        #    floor and the active policy's does not
        and c["ok_statue_starves"] == 1.0
        and m["ok_random_survives"] == 1.0
        # the CONTROL must fail on its own pre-registered side
        and c["i_min"] >= m["i_min"]
        and c["ok_statue_never_ate"] == 1.0
        # the NULL: a disabled integrator does not move, on the same physics
        and m["ok_null_flat"] == 1.0
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PS.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
