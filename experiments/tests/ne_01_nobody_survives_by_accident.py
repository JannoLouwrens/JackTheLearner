"""NE.01 — the needs are a real control problem: nobody survives by accident.

`experiments/needs.py` is the §2.3 seven-need integrator, built 2026-08-24 with
every constant marked PROPOSAL. This spec is what turns those proposals into
measurements (its own `kills` field: "Every constant in 2.3 is a PROPOSAL until
this spec replaces it with a measurement"). It certifies a property of the
WORLD + integrator pair, before anything trains under it.

## The pre-registered gates (registry NE.01, unchanged)

1. RANGE     every need's delta traverses >= 0.3 (p90-p10), none pinned.
2. RANDOM    a random agent dies within 300-6,000 decisions, every life.
3. STATUE    (control i) doing nothing is lethal, cause RECORDED — under
             §2.3's own constants that cause is DEHYDRATION at ~570 s (450 s
             tank + 120 s grace), NOT starvation (1,800 + 300 s); the registry
             prose was corrected 2026-08-24 pre-run per the build-time flag.
4. FORAGER   (control ii) a scripted competent forager survives >= 3 sim-days.
5. CAUSES    no single need causes > 60% of pooled random deaths.
6. NIGHT     a night asleep in the open costs 0.3-0.6 and is survivable once.
7. SHELTER   a night at sky_occlusion >= 0.4 is nearly free.
Plus §9: the coarse sleep path matches fine physics within 0.2 C over a night.

## Pre-registered DEFINITIONS the gates read (fixed before the run)

* NIGHT COST = `delta_T` at dawn, i.e. the doc's "integrity-equivalent drive"
  (`NEEDS_AND_DEATH` line ~1099) taken literally: lambda_T == lambda_i == 1.0
  and both enter d(h) through the same exponent, so the integrity loss that
  would produce the same drive contribution as a thermal deviation IS delta_T
  itself. The cost is deliberately NOT d(dawn) - d(nightfall): water drains
  0.89 tank/night under ANY roof (400 s * b_w), so total-drive cost is
  dominated by a term shelter cannot touch and would read ~0.85 everywhere —
  a gate on it could not fail in the direction the falsifier names ("shelter
  makes no measurable difference"). The thermal channel is the TEACHABLE part
  of the night; the shared drains are recorded beside it, not gated.
* NEARLY FREE (gate 7) = dawn delta_T <= 0.2 AND <= 0.5 x the open-night
  cost, at a realized sky_occlusion >= 0.4. Fixed here, before the run.
* RANGE (gate 1) is measured over the BEHAVIOURS THE WORLD ADMITS — the max
  per-trajectory p90-p10 across this spec's own probes (random lives, the
  drop/rest probe, the night probes, the forager), in delta coordinates
  (`needs.deltas`, all in [0,1]). This is PS.01 v2's precedent verbatim: a
  random policy cannot fall from a platform it never climbs and cannot spend
  three days draining `c` inside a 100 s life; v2's redesign established that
  "usable range" means range over admitted behaviours, with EVENT GATES so an
  unexercised channel is a loud red rather than a quiet small spread. The
  event gates here: the drop probe must produce >= 2 damaging impacts, the
  forager must actually eat (>= 5 meals), drink (>= 3) and sleep (>= 1,000
  decisions). PINNED = a need whose delta sits at one clip bound in > 95% of
  all pooled samples.

## Fixture abstractions, declared (the PS.01 v2 idiom: narrow, and named)

The scripted forager abstracts LOCOMOTION and nothing else — no locomotion
controller exists at this rung (T2.01 is FAIL). Food is served onto his mouth
when due (eating still happens through the layer's real mouth-contact test,
respawn clocks and all), he is teleported to the pool to drink (the drink
still fires through the real mouth-near-surface test) and teleported to his
sleeping spot at night. Teleports zero qvel, so they cannot manufacture
impact damage. Each apple is paid for with a 10 s full-power action burst —
the climb's energy bill, since the serve waives the climb itself (PG.3 proves
the climb is a behaviour this world admits). Awake and idle he acts at duty
0.10 (declared: below PS.01's floor-food D* = 0.217 because this forager also
harvests apples; the §2.3 economy funds it — supply ~1.6 e/day vs ~0.8
drained).

## What the pilots already measured (2026-08-24, pre-registration, recorded
## so the run's numbers land against stated expectations)

* Random flailing at CTRL_SCALE 0.4 produces 360-990 W of mechanical power;
  kappa_act (sized so full power triples M, §2.3's own criterion) then puts
  the dry thermal equilibrium at 40.5-46.8 C against T_DAY = 30 and
  k_dry = 14.29 W/C. Three pilot lives died of HYPERTHERMIA at 92-109 s.
  Gate 2 is comfortably met; gate 5 is expected to FAIL with the T-channel
  near 100% — which is precisely the distortion `NEEDS_AND_DEATH` §9
  pre-registered this gate to bound ("temperature is roughly 20x more
  dangerous, relative to hunger, than it is for a human — the largest single
  distortion in the suite"). If it fails, that is a MEASUREMENT about the
  §2.3 parameterisation (the spec's `kills`), not a harness fault, and the
  redesign routes through the Review, not an argument here.
* The occlusion model has a lethal interior: k_eff = k_dry(1 - 0.7*occ) cuts
  heat LOSS only (there is no sun load), so at occ = 1.0 a sleeping body
  (M = 100 W through 4.29 W/C) equilibrates ~23 C above ambient — a sealed
  roof cooks him in ~3 min, and BY DAY any occ > ~0.43 at rest crosses the
  40 C line. Shelter has an optimal depth. The sheltered-night fixture
  therefore searches for a pose in the occ 0.5-0.9 band (under the ramp —
  a lean-to the world already admits; poses from 0.11 to 1.00 exist there),
  and the forager shelters only AFTER nightfall, napping in the open by day
  (day-open at rest equilibrates exactly 37 C). Both facts are recorded in
  the metrics; neither is patched here.
* DELTA_T_NIGHT was calibrated 12 -> 10 by the doc's own assignment ("NE.01
  calibrates DELTA_T_NIGHT against its 0.3-0.6 night-cost gate"): 12 put the
  open-night cost at 0.598 — on the gate's edge, where integrator drift
  rather than the world decides — 10 reads 0.498, mid-band. The 5-point
  sweep table ships in the metrics (`sweep_open_cost_dtn*`).

## Provenance

j0/alpha are PS.01's measured impact constants and p_max is its P_bar(1),
read live via `borrow_metrics` (never pasted — T0.14's scar). A refusal is
Status.VOID: an uncalibrated integrity channel refutes nothing. A fresh
J_0 (p95 of the impact channel over random-life contact onsets, the "normal
locomotion" regime) is measured and RECORDED beside the borrowed one, per the
spec's notes; the borrowed one is the one in use.

THE NULL — the integrator disabled — rides random life 0's own rollout: fed
every substep, it must not move (every spread 0, no death), so a nonzero
spread is a statement about the reduction, not about a second quieter world.

WHAT THIS CANNOT SAY: that needs teach anything (NE.03), that any need earns
its parameters (NE.02), or that the compression of human timescales into a
1,200 s sim-day is harmless (§9 declares it unsettled; gate 5 only bounds it).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID

# This spec certifies a property of the WORLD + integrator pair: both hash
# into impl_sha so the certificate goes stale loudly when either changes.
IMPL_DEPS = ["playground.py", "experiments/needs.py", "experiments/drives.py"]

REPO = Path(__file__).resolve().parents[2]

# ── the rollout, fixed before the run ───────────────────────────────────
SIM_S_PER_DECISION = 0.2
CTRL_SCALE = 0.4              # PS.01's random policy, the regime j0 describes
N_RANDOM_LIVES = 5
RANDOM_LIFE_CAP = 6000        # gate 2's own upper bound: one full sim-day
DEATH_MIN_DEC, DEATH_MAX_DEC = 300, 6000
N_PUSH_DECISIONS = 1500       # the random object-pushing occlusion probe
FORAGER_DECISIONS = 18000     # 3 sim-days, gate 4 verbatim
FORAGER_DUTY = 0.10
APPLE_BURST_DEC = 50          # 10 s full-power: the climb's energy bill
NIGHT_DECISIONS = 2000        # one 400 s night
NIGHT_SETTLE_DEC = 25         # awake bed-down shared by fine and coarse twins
OPEN_SPOT = (-0.8, -1.0)      # flat ground: no ramp, stairs, ladder or pool
LYING_QUAT = (0.7071, 0.0, 0.7071, 0.0)
HOME_SPOT = (0.0, -1.6, 0.30)

# ── the pre-registered thresholds (registry NE.01 + definitions above) ──
SPREAD_MIN = 0.30
PINNED_FRAC = 0.95
CAUSE_SHARE_MAX = 0.60
NIGHT_COST_LO, NIGHT_COST_HI = 0.30, 0.60
SHELTER_OCC_MIN = 0.40
SHELTER_COST_MAX = 0.20       # "nearly free", quantified pre-run
SHELTER_RATIO_MAX = 0.50
COARSE_FINE_T_MAX = 0.20      # §9's gate, verbatim
OPEN_OCC_MAX = 0.15           # instrument gate: an "open" night must be open
MIN_DAMAGING = 2              # the drop probe must have produced its events
MIN_MEALS, MIN_DRINKS, MIN_SLEEP_DEC = 5, 3, 1000
STATUE_DEATH_FRAC = 0.8       # the control's death must be OBSERVED (PS.01 v2)
SWEEP_DTN = (6.0, 8.0, 10.0, 12.0, 14.0)
CAUSES = ("dehydration", "starvation", "hyperthermia", "hypothermia",
          "injury", "drowning")

_BORROW_CACHE: dict = {}


def _borrowed():
    """PS.01's j0/alpha/P_bar(1), read once per process, refusal preserved."""
    if "b" not in _BORROW_CACHE:
        _BORROW_CACHE["b"] = borrow_metrics(
            "PS.01", ("j0_ms", "alpha", "mean_power_w_full_strength"))
    return _BORROW_CACHE["b"]


# ── world construction (PS.01's seed convention: seed 0 nursery, 1+ mutate) ──
def _params(seed: int):
    import dataclasses

    import numpy as np
    sys.path.insert(0, str(REPO))
    from playground import PlaygroundParams

    p = PlaygroundParams(seed=seed)
    if seed > 0:
        p = p.mutate(np.random.RandomState(seed))
    return p


def _build(seed: int):
    sys.path.insert(0, str(REPO))
    from playground import make_playground

    p = _params(seed)
    model, data, water = make_playground(p, with_water=True, with_humanoid=True)
    pool = (2.6, -2.4, p.pool_size, 0.0)
    return p, model, data, water, pool


def _layer(model, pool, seed):
    from experiments import needs

    b = _borrowed()
    return needs.NeedLayer(model, j0=b.values["j0_ms"], alpha=b.values["alpha"],
                           p_max=b.values["mean_power_w_full_strength"],
                           pool=pool, seed=seed)


def _place(model, data, qadr, xyz, quat=LYING_QUAT):
    """Teleport the humanoid root, at rest. qvel zeroed: a teleport must not
    manufacture an impact (the j channel reads arrival speed at contact onset)."""
    import mujoco
    data.qpos[qadr:qadr + 3] = xyz
    data.qpos[qadr + 3:qadr + 7] = quat
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


class _DisabledNeed:
    """The null: fed every substep, never integrates. Rides the live rollout,
    and its variables are SAMPLED per decision like the live layer's (PS.01's
    null), so `null_spread == 0` is a statement about the reduction path, not
    a constant the test wrote down."""

    def __init__(self):
        self.e, self.w, self.p, self.T = 1.0, 1.0, 0.05, 37.0
        self.f, self.c, self.i = 0.0, 1.0, 1.0
        self.dead = False
        self.rows = []

    def begin_decision(self):
        pass

    def substep(self, model, data, dt):
        pass

    def decide(self):
        self.rows.append([self.e, self.w, self.p, self.T,
                          self.f, self.c, self.i])
        return self


def _step_decision(model, data, water, layer, ctrl, null=None):
    """One decision: caller-owned mj_step loop, the NeedLayer contract."""
    import mujoco
    dt = float(model.opt.timestep)
    fs = max(1, int(round(SIM_S_PER_DECISION / dt)))
    layer.begin_decision()
    if null is not None:
        null.begin_decision()
    if layer.microsleep_zeroed():
        ctrl = ctrl * 0.0
    for _ in range(fs):
        data.ctrl[:] = ctrl
        water.apply(model, data)
        mujoco.mj_step(model, data)
        layer.substep(model, data, dt)
        if null is not None:
            null.substep(model, data, dt)
    layer.decide()
    if null is not None:
        null.decide()


def _deltas_row(layer):
    from experiments import needs
    s = layer.state
    d = needs.deltas(s.e, s.w, s.p, s.T, s.f, s.c, s.i)
    return [d[k] for k in "ewpTfci"]


# ── probe 1: random lives (gates 2 and 5, j0 harvest, null) ─────────────
def _random_lives(seed: int) -> dict:
    import numpy as np

    deaths, js, occs, trajs = [], [], [], []
    null_dead = 0
    for life in range(N_RANDOM_LIVES):
        p, model, data, water, pool = _build(seed)
        layer = _layer(model, pool, seed * 131 + life)
        null = _DisabledNeed() if life == 0 else None
        rng = np.random.RandomState(seed * 9973 + life * 17)
        rows = []
        k = 0
        while not layer.dead and k < RANDOM_LIFE_CAP:
            n0 = layer.n_onsets
            ctrl = rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu) * layer.gear_scale()
            _step_decision(model, data, water, layer, ctrl, null=null)
            if layer.n_onsets > n0:
                js.append(layer.last_j)
            occs.append(layer.last_occlusion)
            rows.append(_deltas_row(layer))
            k += 1
        rec = layer.death_record
        deaths.append({
            "died": int(layer.dead), "decisions": k,
            "cause": rec["cause"] if rec else "none",
            "ms10": int(bool(rec and rec["microsleep_within_10s"]))})
        trajs.append(np.array(rows))
        if null is not None:
            null_dead = int(null.dead)
            nr = np.array(null.rows)
            null_spread = float(max(np.percentile(nr[:, j], 90)
                                    - np.percentile(nr[:, j], 10)
                                    for j in range(nr.shape[1])))
        del model, data, water, layer
    return {"deaths": deaths, "js": js, "occs": occs, "trajs": trajs,
            "null_dead": null_dead, "null_spread": null_spread}


# ── probe 2: drop/rest (exercises i; PS.01 v2's mixed-fixture reasoning) ──
def _drop_probe(seed: int) -> dict:
    import numpy as np
    sys.path.insert(0, str(REPO))
    from playground import LADDER_X, LADDER_Y, SPAWN_Z, humanoid_index

    p, model, data, water, pool = _build(seed)
    layer = _layer(model, pool, seed * 977 + 5)
    rng = np.random.RandomState(seed * 31 + 3)
    qadr = humanoid_index(model)["qposadr"]
    b = _borrowed()
    j0 = b.values["j0_ms"]
    drop_xyz = (LADDER_X, LADDER_Y + 0.45 + 0.9, p.ladder_height + SPAWN_Z)

    rows, n_damaging, n_rest = [], 0, 0
    sched = (["drop"] + ["random"] * 74 + ["rest"] * 200) * 3
    for mode in sched:
        if layer.dead:
            break
        if mode == "drop":
            _place(model, data, qadr, drop_xyz, quat=(1.0, 0.0, 0.0, 0.0))
        act = mode != "rest"
        n0 = layer.n_onsets
        ctrl = (rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu) * layer.gear_scale()
                if act else np.zeros(model.nu))
        _step_decision(model, data, water, layer, ctrl)
        if layer.n_onsets > n0 and layer.last_j > j0:
            n_damaging += 1
        if layer.last_rest_dt > 0.5 * layer.last_dt:
            n_rest += 1
        rows.append(_deltas_row(layer))
    return {"traj": np.array(rows), "n_damaging": n_damaging, "n_rest": n_rest}


# ── probe 3: the sheltered sleeping pose the world admits ───────────────
def _find_shelter_pose(seed: int) -> dict:
    """Grid-search lying poses under the ramp for realized occlusion.

    The target is the occ 0.5-0.9 BAND, not the max: the occlusion model's
    lethal interior (docstring) makes occ ~1.0 a cooked corpse, and the pilot
    measured poses from 0.11 to 1.00 here. Also returns the maximum found —
    the constructibility record."""
    import numpy as np

    p, model, data, water, pool = _build(seed)
    sys.path.insert(0, str(REPO))
    from playground import humanoid_index
    qadr = humanoid_index(model)["qposadr"]
    layer = _layer(model, pool, 0)

    best_band, best_max = None, (0.0, None)
    q0 = data.qpos.copy()
    for dx in np.arange(-0.6, 0.61, 0.2):
        for dy in (1.2, 1.4, 1.6):
            data.qpos[:] = q0
            _place(model, data, qadr, (-2.7 + dx, dy, 0.25))
            for _ in range(200):                    # settle 1 s
                data.ctrl[:] = 0.0
                import mujoco
                mujoco.mj_step(model, data)
            occ = layer._sky_occlusion(model, data)
            if occ > best_max[0]:
                best_max = (occ, (-2.7 + dx, dy))
            if 0.5 <= occ <= 0.9:
                score = -abs(occ - 0.7)
                if best_band is None or score > best_band[0]:
                    best_band = (score, occ, (-2.7 + dx, dy))
    del model, data, water
    if best_band is not None:
        return {"xy": best_band[2], "occ_probe": best_band[1],
                "occ_max_reachable": best_max[0]}
    return {"xy": best_max[1], "occ_probe": best_max[0],
            "occ_max_reachable": best_max[0]}


# ── probe 4: the nights (gates 6 and 7, §9 coarse-vs-fine) ──────────────
def _night(seed: int, xy, rng_salt: int, coarse: bool) -> dict:
    """One 400 s night asleep at `xy`, from a sated nightfall state.

    Fine: NIGHT_SETTLE_DEC awake decisions (bed-down, occlusion cast), then
    asleep under full physics. Coarse: the SAME bed-down, then
    `sleep_coarse_step` for the remainder — §9's question is exactly whether
    that path changes the physics."""
    import numpy as np
    from experiments import needs
    sys.path.insert(0, str(REPO))
    from playground import humanoid_index

    p, model, data, water, pool = _build(seed)
    qadr = humanoid_index(model)["qposadr"]
    layer = _layer(model, pool, seed * 733 + rng_salt)
    layer.t = needs.DAY_S                            # nightfall
    layer.state = needs.NeedState(p=0.70)            # §2.3's worked dusk
    _place(model, data, qadr, (xy[0], xy[1], 0.25))

    rows, occs = [], []
    for _ in range(NIGHT_SETTLE_DEC):
        _step_decision(model, data, water, layer, np.zeros(model.nu))
        rows.append(_deltas_row(layer))
        occs.append(layer.last_occlusion)
    layer.set_asleep(True)
    if coarse:
        remaining = needs.NIGHT_S - NIGHT_SETTLE_DEC * SIM_S_PER_DECISION
        layer.sleep_coarse_step(remaining, dt_int=1.0)
    else:
        n = NIGHT_DECISIONS - NIGHT_SETTLE_DEC
        for _ in range(n):
            if layer.dead:
                break
            _step_decision(model, data, water, layer, np.zeros(model.nu))
            rows.append(_deltas_row(layer))
            occs.append(layer.last_occlusion)
    s = layer.state
    out = {"T_dawn": s.T, "cost": needs.delta_T(s.T), "alive": int(not layer.dead),
           "e_dawn": s.e, "w_dawn": s.w, "occ_mean": float(np.mean(occs)),
           "traj": np.array(rows) if rows else np.zeros((0, 7)),
           "cause": layer.death_record["cause"] if layer.death_record else "none"}
    del model, data, water
    return out


def _sweep_table() -> dict:
    """The 5-point DELTA_T_NIGHT calibration sweep, pure ODE (asleep, dry,
    open, from 37 C) — the table that chose the shipped constant."""
    from experiments import needs
    out = {}
    for dtn in SWEEP_DTN:
        T = 37.0
        for _ in range(2000):
            T = needs.thermal_step(T, 0.0, 0.0, 0.0, needs.T_DAY - dtn,
                                   SIM_S_PER_DECISION, 0.0)
        out[f"sweep_open_cost_dtn{int(dtn)}"] = needs.delta_T(T)
    return out


# ── probe 5: the scripted competent forager (gate 4, exercises e/w/p/c) ──
def _forager(seed: int, shelter_xy) -> dict:
    import numpy as np
    from experiments import needs
    sys.path.insert(0, str(REPO))
    from playground import humanoid_index

    p, model, data, water, pool = _build(seed)
    qadr = humanoid_index(model)["qposadr"]
    layer = _layer(model, pool, seed * 389 + 7)
    rng = np.random.RandomState(seed * 613 + 11)
    head_id = int(model.geom("head").id)

    joints = {}
    for name in ("obj0", "obj1", "apple"):
        try:
            bid = int(model.body(name).id)
        except (KeyError, ValueError):
            continue
        jadr = int(model.body_jntadr[bid])
        if jadr < 0:
            continue
        joints[name] = (int(model.jnt_qposadr[jadr]), int(model.jnt_dofadr[jadr]))

    def serve(name):
        qa, da = joints[name]
        hp = np.asarray(data.geom_xpos[head_id])
        data.qpos[qa:qa + 3] = (hp[0], hp[1], hp[2] + 0.25)
        data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[da:da + 6] = 0.0

    due = {name: 0.0 for name in joints}
    ate0 = dict(layer.ate_total)
    rows, occs_sleep = [], []
    asleep = False
    at_shelter = False
    pool_visit = 0
    burst = 0
    sleep_dec = 0
    _place(model, data, qadr, HOME_SPOT)
    for k in range(FORAGER_DECISIONS):
        if layer.dead:
            break
        s = layer.state
        want_sleep = s.p > 0.6 or layer.is_night()
        if want_sleep and not asleep:
            # shelter only at night — by day the roof is the hotter place
            # (the occlusion model's lethal interior, docstring), so day
            # naps happen in the open at home, where rest sits at 37 C.
            if layer.is_night():
                _place(model, data, qadr, (shelter_xy[0], shelter_xy[1], 0.25))
                at_shelter = True
            layer.set_asleep(True)
            asleep, pool_visit = True, 0
        elif asleep and layer.is_night() and not at_shelter:
            # a day nap that ran into nightfall: relocate under the roof
            _place(model, data, qadr, (shelter_xy[0], shelter_xy[1], 0.25))
            at_shelter = True
        elif not want_sleep and asleep:
            layer.set_asleep(False)
            _place(model, data, qadr, HOME_SPOT)
            asleep, at_shelter = False, False

        ctrl = np.zeros(model.nu)
        if asleep:
            sleep_dec += 1
            occs_sleep.append(layer.last_occlusion)
        else:
            if pool_visit > 0:
                pool_visit -= 1
                if pool_visit == 0:
                    _place(model, data, qadr, HOME_SPOT)
            elif s.w < 0.6 and layer.t >= layer._drink_ready_at:
                _place(model, data, qadr, (pool[0], pool[1], 0.30))
                pool_visit = 10
            elif burst > 0:
                burst -= 1
                ctrl = rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu) * layer.gear_scale()
                if burst == 0 and "apple" in joints:
                    serve("apple")
            elif ("apple" in joints and layer.t >= due["apple"] and s.e < 0.9):
                burst = APPLE_BURST_DEC          # pay the climb before the apple
            else:
                for name in ("obj0", "obj1"):
                    if name in joints and layer.t >= due[name] and s.e < 0.95:
                        serve(name)
                if rng.random_sample() < FORAGER_DUTY:
                    ctrl = rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu) * layer.gear_scale()
        _step_decision(model, data, water, layer, ctrl)
        for name in joints:
            if layer.ate_total[name] > ate0[name]:
                due[name] = layer.t + needs.FOOD[name][1]
                ate0[name] = layer.ate_total[name]
        rows.append(_deltas_row(layer))

    out = {
        "survived_dec": len(rows) if not layer.dead else int(layer.t / SIM_S_PER_DECISION),
        "alive": int(not layer.dead),
        "cause": layer.death_record["cause"] if layer.death_record else "none",
        "meals": int(sum(layer.ate_total.values())),
        "drinks": int(layer.drank_total),
        "sleep_dec": sleep_dec,
        "sleep_occ_mean": float(np.mean(occs_sleep)) if occs_sleep else 0.0,
        "traj": np.array(rows),
    }
    del model, data, water
    return out


# ── probe 6: occlusion reachable by random pushing (the notes' clause) ──
def _push_probe(seed: int) -> dict:
    import numpy as np
    sys.path.insert(0, str(REPO))
    from playground import humanoid_index

    p, model, data, water, pool = _build(seed)
    qadr = humanoid_index(model)["qposadr"]
    layer = _layer(model, pool, seed * 271 + 9)
    rng = np.random.RandomState(seed * 4409 + 1)
    _place(model, data, qadr, (0.0, 0.5, 0.9), quat=(1.0, 0.0, 0.0, 0.0))
    occs = []
    for _ in range(N_PUSH_DECISIONS):
        if layer.dead:
            layer.new_body()                     # deaths here are not counted
        ctrl = rng.uniform(-CTRL_SCALE, CTRL_SCALE, model.nu) * layer.gear_scale()
        _step_decision(model, data, water, layer, ctrl)
        occs.append(layer.last_occlusion)
    occs = np.array(occs)
    del model, data, water
    return {"push_occ_max": float(occs.max()),
            "push_occ_frac_pos": float((occs > 0.0).mean())}


# ── the experiment ──────────────────────────────────────────────────────
NEED_KEYS = tuple("ewpTfci")


def _spreads(trajs) -> dict:
    """Per-need spread = max over probe trajectories of within-trajectory
    p90-p10, in delta coordinates. Pinned = one clip bound > 95% of pooled."""
    import numpy as np
    per_need, pinned = {}, {}
    pooled = np.concatenate([t for t in trajs if len(t)], axis=0)
    for j, k in enumerate(NEED_KEYS):
        best = 0.0
        for t in trajs:
            if len(t) < 10:
                continue
            best = max(best, float(np.percentile(t[:, j], 90)
                                   - np.percentile(t[:, j], 10)))
        per_need[k] = best
        col = pooled[:, j]
        pinned[k] = int(max((col <= 0.0).mean(), (col >= 1.0).mean())
                        > PINNED_FRAC)
    return {"per_need": per_need, "pinned": pinned}


def _experiment(seed: int) -> dict:
    import numpy as np

    b = _borrowed()
    if not b.ok:
        return {"void": f"VOID: {b.refusal}", "all_finite": 0, **b.provenance}

    rnd = _random_lives(seed)
    drop = _drop_probe(seed)
    pose = _find_shelter_pose(seed)
    night_open = _night(seed, OPEN_SPOT, 21, coarse=False)
    night_open_coarse = _night(seed, OPEN_SPOT, 21, coarse=True)
    night_shel = _night(seed, pose["xy"], 23, coarse=False)
    for_ = _forager(seed, pose["xy"])
    push = _push_probe(seed)

    trajs = (rnd["trajs"] + [drop["traj"], night_open["traj"],
                             night_shel["traj"], for_["traj"]])
    sp = _spreads(trajs)
    min_spread = min(sp["per_need"].values())

    deaths = rnd["deaths"]
    n_died = sum(d["died"] for d in deaths)
    in_window = sum(d["died"] and DEATH_MIN_DEC <= d["decisions"] <= DEATH_MAX_DEC
                    for d in deaths)
    dtimes = [d["decisions"] for d in deaths if d["died"]]
    cause_counts = {c: sum(d["cause"] == c for d in deaths) for c in CAUSES}

    js = rnd["js"]
    m = {
        "all_finite": int(all(np.isfinite(t).all() for t in trajs if len(t))),
        # gate 1 — range over admitted behaviours
        "min_need_spread": min_spread,
        **{f"spread_{k}": v for k, v in sp["per_need"].items()},
        "n_pinned": int(sum(sp["pinned"].values())),
        "ok_spread": int(min_spread >= SPREAD_MIN
                         and sum(sp["pinned"].values()) == 0),
        # gate 2 — random deaths inside the window
        "n_random_died": n_died,
        "n_random_in_window": in_window,
        "death_dec_min": float(min(dtimes)) if dtimes else -1.0,
        "death_dec_max": float(max(dtimes)) if dtimes else -1.0,
        "ok_random_deaths": int(n_died == N_RANDOM_LIVES
                                and in_window == N_RANDOM_LIVES),
        # gate 5 — pooled cause counts (shares computed in _check from means)
        **{f"deaths_{c}": n for c, n in cause_counts.items()},
        "deaths_with_microsleep_within_10s": sum(d["ms10"] for d in deaths),
        # gates 6/7 — the nights
        "night_open_cost": night_open["cost"],
        "night_open_T_dawn": night_open["T_dawn"],
        "night_open_alive": night_open["alive"],
        "night_open_occ": night_open["occ_mean"],
        "night_open_e_dawn": night_open["e_dawn"],
        "night_open_w_dawn": night_open["w_dawn"],
        "night_shelter_cost": night_shel["cost"],
        "night_shelter_T_dawn": night_shel["T_dawn"],
        "night_shelter_alive": night_shel["alive"],
        "night_shelter_occ": night_shel["occ_mean"],
        "ok_night_open": int(night_open["alive"] == 1
                             and night_open["occ_mean"] <= OPEN_OCC_MAX
                             and NIGHT_COST_LO <= night_open["cost"] <= NIGHT_COST_HI),
        "ok_night_shelter": int(
            night_shel["alive"] == 1
            and night_shel["occ_mean"] >= SHELTER_OCC_MIN
            and night_shel["cost"] <= SHELTER_COST_MAX
            and night_shel["cost"] <= SHELTER_RATIO_MAX * max(night_open["cost"], 1e-9)),
        # §9 — the coarse sleep path may not change the physics
        "coarse_fine_dT": abs(night_open["T_dawn"] - night_open_coarse["T_dawn"]),
        "ok_coarse_fine": int(abs(night_open["T_dawn"]
                                  - night_open_coarse["T_dawn"]) <= COARSE_FINE_T_MAX),
        # gate 4 — the forager (control ii lives in the experiment, PS.01's
        # pattern: run_spec admits one control_fn and the statue is it)
        "forager_survived_dec": for_["survived_dec"],
        "forager_alive": for_["alive"],
        "forager_meals": for_["meals"],
        "forager_drinks": for_["drinks"],
        "forager_sleep_dec": for_["sleep_dec"],
        "forager_sleep_occ": for_["sleep_occ_mean"],
        "ok_forager": int(for_["alive"] == 1
                          and for_["survived_dec"] >= FORAGER_DECISIONS),
        "ok_forager_exercised": int(for_["meals"] >= MIN_MEALS
                                    and for_["drinks"] >= MIN_DRINKS
                                    and for_["sleep_dec"] >= MIN_SLEEP_DEC),
        # instrument-alive event gates (PS.01 v2's medicine)
        "n_damaging": drop["n_damaging"],
        "n_rest_decisions": drop["n_rest"],
        "ok_probe_exercised": int(drop["n_damaging"] >= MIN_DAMAGING),
        # the null rode random life 0
        "null_spread": rnd["null_spread"],
        "null_dead": rnd["null_dead"],
        "ok_null_flat": int(rnd["null_spread"] == 0.0 and rnd["null_dead"] == 0),
        # recorded, not gated
        "j0_p95_measured": float(np.percentile(js, 95)) if js else -1.0,
        "j0_borrowed": b.values["j0_ms"],
        "alpha_borrowed": b.values["alpha"],
        "p_max_borrowed": b.values["mean_power_w_full_strength"],
        "shelter_occ_probe": pose["occ_probe"],
        "occ_max_reachable": pose["occ_max_reachable"],
        **push,
        **_sweep_table(),
        **{k: str(v) for k, v in b.provenance.items()},
    }
    m["need_dynamic_range_x_death_spread"] = (
        min_spread * (in_window / float(N_RANDOM_LIVES)))
    return m


def _control(seed: int) -> dict:
    """Control (i): the DO-NOTHING statue. It must die, with the cause
    RECORDED — §2.3's constants say dehydration at ~570 s — never eating,
    never drinking, at best integrity. Observed inside the window, not
    scheduled at its edge (PS.01 v2 change 3)."""
    import numpy as np

    b = _borrowed()
    if not b.ok:
        return {"void": f"VOID: {b.refusal}", "all_finite": 0}

    p, model, data, water, pool = _build(seed)
    layer = _layer(model, pool, seed * 547 + 13)
    rows, i_min = [], 1.0
    k = 0
    while not layer.dead and k < RANDOM_LIFE_CAP:
        _step_decision(model, data, water, layer, np.zeros(model.nu))
        rows.append(_deltas_row(layer))
        i_min = min(i_min, layer.state.i)
        k += 1
    rec = layer.death_record
    m = {
        "all_finite": int(np.isfinite(np.array(rows)).all()),
        "statue_died": int(layer.dead),
        "statue_death_dec": k,
        "statue_death_s": layer.t,
        "statue_cause_recorded": int(bool(rec and rec["cause"] in CAUSES)),
        "statue_cause_dehydration": int(bool(rec and rec["cause"] == "dehydration")),
        "statue_i_min": float(i_min),
        "statue_ate": int(sum(layer.ate_total.values())),
        "statue_drank": int(layer.drank_total),
        "ok_statue_death_observed": int(layer.dead
                                        and k < STATUE_DEATH_FRAC * RANDOM_LIFE_CAP),
    }
    del model, data, water
    return m


def _check(m: dict, c: dict):
    from ..protocol import _declares_void
    if _declares_void(m) or _declares_void(c):
        return Status.VOID
    # gate 5's pooled share, exact from per-seed means (equal lives per seed)
    total = sum(m.get(f"deaths_{k}", 0.0) for k in CAUSES)
    max_share = (max(m.get(f"deaths_{k}", 0.0) for k in CAUSES) / total
                 if total > 0 else 1.0)
    return bool(
        m["all_finite"] == 1 and c["all_finite"] == 1
        # 1. every need traverses a usable range, no need pinned, every seed
        and m["ok_spread"] == 1.0
        and m["ok_probe_exercised"] == 1.0
        and m["ok_forager_exercised"] == 1.0
        # 2. a random agent dies, inside [300, 6000], every life, every seed
        and m["ok_random_deaths"] == 1.0
        # 5. no single need above 60% of pooled random deaths
        and max_share <= CAUSE_SHARE_MAX
        # 6. a night in the open costs 0.3-0.6 and is survivable once
        and m["ok_night_open"] == 1.0
        # 7. a night at occ >= 0.4 is nearly free
        and m["ok_night_shelter"] == 1.0
        # §9: the coarse sleep path does not change the physics
        and m["ok_coarse_fine"] == 1.0
        # 4. the hand-written oracle survives three sim-days
        and m["ok_forager"] == 1.0
        # 3. the CONTROL fails on its pre-registered side: doing nothing is
        # lethal, cause recorded, observed, at best integrity, zero intake
        and c["statue_died"] == 1.0
        and c["statue_cause_recorded"] == 1.0
        and c["ok_statue_death_observed"] == 1.0
        and c["statue_ate"] == 0.0 and c["statue_drank"] == 0.0
        and c["statue_i_min"] >= 0.9
        # the null: a disabled integrator does not move on the same physics
        and m["ok_null_flat"] == 1.0
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["NE.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
