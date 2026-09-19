"""PS.08 — Heavy is a price: moving mass costs need-currency before anyone
has to learn it.

HYPOTHESIS (registry, unchanged). The venue charges for moving mass, in the
currency the needs already speak: a scripted transport policy driven to
displace playground objects of >= 4 pre-registered masses (inside the
EXISTING object_mass_range 0.2-3.0 kg, playground.py:123 — no new world
content) over an identical displacement pays a need-cost (integrated drive
drain read off the existing needs.py channels: M = M_BASAL + kappa_act*p_mech,
needs.py:245, with fatigue riding the same p_mean, needs.py:513 — no new
instrument) that increases MONOTONICALLY in object mass on every seed; the
light-vs-heavy cost gap exceeds the venue's outcome-metric quantum; and the
load is LEGIBLE beforehand — a probe on the sensory vector predicts the mass
class above chance early in the push, while there is still time to decline
the load.

## THE FIXTURE: BOUT-PUSHING, BECAUSE THIS BODY HAS NO WALK AND NO GRIP

The body has no transport controller, so a "transport policy" is scripted the
way PS.05 scripted walking — except that here NOTHING may be virtual, because
the mass must enter the accounting through real physics or the spec is the
tautology its own registry clause forbids. The idiom is the BOUT: teleport
the lying body (qvel zeroed — a teleport must not manufacture an impact,
PS.06) to a fixed offset BEHIND the object along the push bearing, then run
BOUT_DEC decisions of seeded thrash (PUSH_SCALE, gear-scaled — the NE.01
caller's contract). The flailing limbs strike the object; momentum transfer
and floor friction — mujoco's, not ours — decide how far it slides. A trip
is a FIXED schedule of N_BOUTS such bouts, each re-centring the object to
the lane start and the body to the standard offset behind it, so every bout
is an independent draw of "this body shoves this mass" and the trip's
displacement is the SUM of per-bout distances moved. Two design choices
were forced by pilot measurement, not taste: (a) the schedule is FIXED, not
push-until-D — a hitting-time design was piloted and its stopping-time
noise swamped the mass signal (gap/quantum 1.3-2.0 across three rig
variants), while the fixed schedule sums near-iid per-bout increments; (b)
displacement is the DISTANCE MOVED per bout (norm), not a bearing
projection — the legs reach past the near face and strike both directions,
and signed increments cancel for exactly the masses that resist most
(measured: 3.0 kg summed 0.057 m of projection from 0.49 m of motion).
Mass resists motion in ANY direction; the norm is the honest coupling. The
trip must ACHIEVE at least D_PUSH cumulative displacement (rig gate — a
BODY finding when it cannot), and the trip's outcome is the endpoint drain
of the layer's own channels, (e_start-e_end)+(w_start-w_end), PRORATED to
exactly D_PUSH of displacement: cost per registered distance, PS.05's
accounting plus one honest division, no new meter.

WHY THE COST IS MONOTONE IF THE WORLD REALLY CHARGES: a heavier object takes
less displacement from the same strike (impulse J gives it J/m of speed, and
sliding friction mu*m*g takes it away over (J/m)^2/(2*mu*g) metres), so the
same commanded thrash buys FEWER metres, and the registered displacement is
priced at more integrated M per metre — basal AND activity drain both accrue
on the layer's own clock while the mass refuses to move. Nothing in this
file computes any of that: it is measured or it is absent, and "absent" is
the registry's first falsified_by branch, honestly reachable.

## THE TAUTOLOGY GUARD (registry, rig): p_mech IS MEASURED, NEVER SYNTHESISED

The power the accounting reads is `NeedLayer`'s own |tau*omega| integral over
the body's actuators (needs.py), exactly as every PS sibling reads it. Mass
enters this file in ONE place: `model.body_mass`/`body_inertia` for the
pushed object (inertia scaled with mass — fixed geometry, uniform density).
No constant here is a function of the nominal mass; no fixture line computes
work as mu*m*g*d; nothing writes a need channel (the PS.09 rule). The guard's
TEETH are the registry's null, the EQUALISED-MASS TWIN: the identical push
protocol with the object's TRUE mass pinned to MASSES[0] whatever its nominal
label says. If cost still rises with the nominal label, the rise was schedule
or fixture, not mass — the twin's cost-vs-nominal-mass curve must be FLAT
(spread strictly under the very threshold the live gap must clear).

## THE THREE WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The monotone cost is a schedule artifact** (later trips tired, a drift
   in the fixture, a lane obstruction growing with trip index). Caught by
   the equalised-mass twin above — same schedule, same lane, same trip
   order, mass pinned light: its curve must be flat. Every trip also starts
   from a standardised fresh body and a re-parked lane (fixture rights, the
   PS.09 comparable-larders reasoning), so no state crosses trips.
2. **The gap is smaller than the venue can resolve.** The QUANTUM (LC.03/
   DP.04 resolution lesson) is the largest within-mass repeat spread of trip
   cost — the same mass re-pushed under different ctrl draws on a fresh
   body, pure rig noise. The light-vs-heavy gap must clear QUANTUM_MULT x
   that floor AND an absolute pre-registered floor, per seed.
3. **The probe reads the episode clock or the trip's identity, not a
   sense.** Legibility rows are taken EARLY (decisions 4 and 8 inside each
   of N_LEG_BOUTS=3 bouts — 12.5% of a cost trip's 24-bout schedule, while
   there is still time to decline the load) on dedicated short trips with
   jittered night start, jittered e0/w0 and an uncounted random work/rest
   PRE-ROLL (PS.05's fake-3 machinery) so no interoceptive offset encodes
   the trip. Rows are held out BY TRIP, the shuffled pairing must collapse
   to chance, and the registry's control — THE SENSORY AMPUTATION — drops
   the load-bearing channels: the KINEMATIC block (pose+velocity: the limbs
   feeling the object resist) AND the PAIN channel (impact against more
   mass is a real nociceptive signal), leaving the 8 clock-like
   interoceptive dims, on which the IDENTICAL probe form must FAIL.
   Feature order is therefore [intero-sans-pain 8 | pain 1 | kin 45] so the
   amputation is a clean prefix keep.

## THE PS.09 LESSON, CARRIED (LESSONS.md 2026-09-19, binding on this probe)

PS.09's probe read 0.78 on its pilot and 0.547 registered: 5 near-duplicate
sniff rows/trip let a 200-feature RFF+ridge fit memorise trip identity while
a bare single-channel threshold read the same held-out sign at 1.00. This
spec applies the lesson STRUCTURALLY, not decoratively: (a) rows within a
trip VARY — they are snapshots of a body mid-thrash at two decisions in
each of three bouts, not near-copies of one stillness; (b) THE REGISTERED
PROBE IS THE CHEAPEST READER THAT COULD POSSIBLY WORK — a fixed-form
single-statistic threshold (per trip, the mean over rows of mean |qvel|:
limbs blocked by mass move slower; sign and threshold fit on train trips
only, the form frozen at registration) — because PS.09 measured that in
this near-duplicate-row regime the high-capacity estimator is the thing
that breaks, and gating on it measures the probe, not the venue; (c) the
family's RFF+ridge idiom rides BESIDE it as diag_rff_acc, REPORTED and
NEVER gated — the known-answer pairing PS.09 had to reconstruct from a
seed-1 introspection after the fact, so a divergence between the two is on
the record from the first run.

## PRE-REGISTERED, before the first registered run (T3.06 lesson)

MASSES = (0.2, 0.9, 1.7, 3.0) kg — spanning the full object_mass_range,
15x; the spacing fork was decided by piloting BOTH arms (see the MASSES
constant). The pushed object is obj2 — non-food BY CONSTRUCTION (needs.py
registers only obj0/obj1 as food, needs.py:161), so the mouth-gated contact
scan can never pay the pusher; obj2 exists on every registered seed and the
pilot seed (n_objects mutates within 4-6 from the default 5) and a mutated
world without it VOIDs on the rig. The lane: every bout starts the object
at PUSH_START (-0.3, 0.6), the body OFFSET=1.30 m up-lane on +x, and a trip
must accumulate D_PUSH = 0.4 m of real displacement over its fixed 24-bout
schedule — the geometry chosen against the world's fixed-structure map so
that the object's wandering path AND the lying body's full envelope (geoms
spanning ~[-1.22, +0.19] m around the root) clear the ramp (-2.5, 2.0),
seesaw (-2.5, -0.5), welded block (-1.5, -1.5), ladder (0, -2.6), stairs
(2.2, 2.2) and the pool (2.6, -2.4) at every mutated pool_size; a rig
assertion refuses any lane that could enter a pool footprint. The body is
placed LEGS-toward-the-object: under LYING_QUAT the body's long reach (the
legs, ~1.22 m from the root) extends toward -x, so the -x bearing is the
side the body can actually strike — measured, not assumed (the +x side has
0.19 m of reach and a smoke bout at that geometry produced near-zero
contact). Every other object is parked at the far
corner and drinking is parked exactly as in PS.09 (the w price still
accrues; only unoffered payoffs are parked). Cost trips run at fresh
e0 = w0 = 1.0 (payoffs are zero, so nothing clips). All gates below were
frozen after a seed-90 pilot (disjoint from registered seeds 0/1/2 — the
PG.6/SM.01/PS.02 convention); the pilot reading sits beside each gate.
Deaths, a stale calibration, an object the body cannot displace the
registered distance within BOUT_MAX bouts (the registry names this branch:
a BODY finding, never a world verdict — PS.03 measured this rover toppling
under sustained push, so displacement is EARNED at the rig, never assumed),
a teleport that moves the object (placement work is unpaid work — the
PS.06 rule extended to the object, gated at OBJ_STILL_MAX), stray
consumption, or a starved row set are RIG findings: Status.VOID, never
FAIL (T0.22). FAIL is reserved for the three falsified_by branches: mass
is free, sub-quantum gap, illegible load.

PILOT (seed 90, disjoint from registered seeds 0/1/2; final fixture,
verified by an independent re-run — deterministic to 1e-15 on the cost
half): every gate green, _check True. Cost per registered 0.4 m:
0.00877 / 0.01477 / 0.01744 / 0.02067 at 0.2/0.9/1.7/3.0 kg — strictly
monotone; gap 0.0119 = 3.35x the quantum 0.00355 and 1.49x the absolute
floor. Equalised-mass twin: 0.00877 at every nominal mass, spread 5e-15
— the same commanded schedule, same draws, mass pinned light, and the
"price curve" vanishes to machine epsilon, so the live curve is the mass
being charged for. Distance actually bought by the same thrash: 7.94 m
at 0.2 kg vs 3.25 m at 3.0 kg. Legibility: probe_bal_acc 0.875 vs the
0.70 bar (16 train trips 8/8, 8 test 4/4), shuffled pairing 0.425,
amputated clock-only control 0.50 (margin 0.375), fresh_frac 1.008.
diag_rff_acc (reported, ungated): 0.5625 — the family's RFF idiom
collapses HERE TOO exactly as PS.09 measured, while the single-statistic
threshold reads the class; carrying both is the point. Two rig facts
paid for by earlier pilot iterations and kept as constants' comments:
OFFSET 1.42 left contact too rare (6 hit-bouts in 21) and PUSH_SCALE
0.4 delivered lottery kicks plus injury deaths inside one trip. The
mass-spacing fork was decided by piloting BOTH arms (see MASSES).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco                                              # noqa: E402

from .. import needs                                       # noqa: E402
from ..protocol import Ledger, Status, borrow_metrics, run_spec  # noqa: E402
from ..registry import BY_ID                               # noqa: E402

# The claim is about the WORLD's needs accounting and the body that pays it.
IMPL_DEPS = ["experiments/needs.py", "playground.py"]

# ── the rollout, fixed before the run (PS siblings' constants where shared) ─
SIM_S = 0.2                  # s per decision (NE.01's cadence)
S_WORK = 0.4                 # ctrl scale, probe/preroll: PS.01's regime
PUSH_SCALE = 0.3             # ctrl scale DURING push bouts: at 0.4 the flail
                             # delivers lottery kicks (a lucky strike sent
                             # the 3.0 kg box the whole lane in 3 bouts) and
                             # enough impact damage to kill by injury inside
                             # one long trip; the price must come from many
                             # small shoves, not from rare violence
PROBE_SEED = 4242            # ONE commanded sequence for P_REF, everywhere
REF_PROBE_DEC = 50           # 10 s fresh reference probe -> alive-proof
LYING_QUAT = (0.7071, 0.0, 0.7071, 0.0)
PUSH_OBJ = "obj2"            # non-food by construction (needs.py:161)
MASSES = (0.2, 0.9, 1.7, 3.0)    # kg — inside object_mass_range 0.2-3.0.
                             # BOTH arms of the spacing fork were PILOTED
                             # (2026-09-19): log-even spacing (0.2/0.5/1.2/
                             # 3.0) evens the middle cost gaps (0.0029/
                             # 0.0038/0.0051) but KILLS the probe (0.125 —
                             # the 0.5-vs-1.2 class boundary sits where
                             # |qvel| no longer separates), while this
                             # spacing reads probe 0.875 with middle cost
                             # gaps 0.0060/0.0027/0.0032 against quantum
                             # 0.00355. The venue's price curve has a
                             # measured shelf at 0.9-1.2 kg, so the m2->m3
                             # gap sits UNDER the quantum: strict 4-point
                             # monotonicity carries a real per-seed flip
                             # risk, partly damped by the paired ctrl draws
                             # (same stream across masses within a rep).
                             # DISCLOSED, not repaired: the repair was
                             # measured to cost the probe.
HEAVY_MIN_IDX = 2            # mass class: index >= this is HEAVY
D_PUSH = 0.4                 # m: every trip must ACHIEVE at least this much
                             # cumulative displacement (rig gate), and the
                             # trip's cost is prorated to exactly this much
PUSH_START = (-0.3, 0.6)     # lane origin; bearing is -x (see the docstring)
PUSH_U = (-1.0, 0.0)         # the push bearing, fixed
OFFSET = 1.30                # body root this far behind the object, on the
                             # +x side: the legs (the lying body's long
                             # reach, extending ~1.22 m from the root toward
                             # -x) start adjacent to the object surface and
                             # kick it down the lane; measured on seed 90,
                             # OFFSET 1.42 leaves contact too rare (6 hit-
                             # bouts in 21) and 1.30 hits on most bouts
OBJ_STILL_MAX = 1e-3         # m/s: placement must not accelerate the object
BOUT_DEC = 10                # 2 s of thrash per bout, then re-place
N_BOUTS = 24                 # FIXED schedule per trip — no early stop: a
                             # hitting-time design ("push until D") was
                             # measured on the pilot and its stopping-time
                             # noise swamped the mass signal (gap/quantum
                             # 1.3-2.0 across three rig variants); the fixed
                             # schedule sums near-iid per-bout increments
N_REP = 4                    # trips per registered mass, live and twin
E0_COST = 1.0                # cost trips start fresh: no payoff, no clip
W0_COST = 1.0
PARK_XY = (2.8, 2.8)         # far-corner park for every non-pushed object
PARK_DY = -0.45              # spacing between parked objects
N_LEG = 24                   # single-bout legibility trips; held out BY TRIP
N_TEST_TRIPS = 8
LEG_CYCLE = (0, 2, 1, 3)     # mass-index cycle -> exact class balance
N_LEG_BOUTS = 3              # a leg trip pushes this many bouts (12.5% of a
                             # cost trip's schedule — still "early")
ROW_DECS = (4, 8)            # snapshot decisions inside EACH leg bout: the
                             # first pilot took all rows in bout 1, where a
                             # cold-started body has often not yet engaged
                             # the object, and the probe read chance
LEG_T_JITTER_S = 250.0       # trip start time ~ DAY_S + U(0, this)
LEG_E0 = (0.60, 0.90)        # jittered starts: intero cannot encode the trip
LEG_W0 = (0.70, 1.00)
PREROLL_MAX_S = 8.0          # uncounted random work/rest before a leg trip
P_PREROLL_WORK = 0.6
INTERO_CLOCK_DIM = 8         # [e, w, p, T_norm, f, c, i, d(h)] — clock-like
PAIN_DIM = 1                 # pain is LOAD-BEARING here: amputated
KIN_DIM = 45                 # qpos[2:24] 22 + qvel 23 — the felt limbs

# ── pre-registered gates (final pilot, seed 90, reading beside each) ────
FRESH_FRAC_MIN = 0.50        # alive-proof: P_REF >= this x p_max   (1.008)
GAP_ABS_MIN = 0.008          # light-vs-heavy cost gap, absolute  (0.0119)
                             # NOTE: an earlier draft of this file carried
                             # 0.015 beside annotations from a superseded
                             # fixture variant (hitting-time, D_PUSH 0.8);
                             # the floor was re-frozen at ~2.3x the FINAL
                             # fixture's measured quantum BEFORE anything
                             # was committed or run registered — this is
                             # gate FREEZING against the fixture that will
                             # run, not a weakening of a registered bar
QUANTUM_MULT = 2.0           # ...and >= this x the measured quantum
                             #        (q = 0.00355; gap / q = 3.35)
ACC_MIN = 0.70               # the headline: load legible early    (0.875)
SHUF_ACC_MAX = 0.60          # trip-level shuffled pairing, mean of 20
                             #                              (pilot 0.425)
CONTROL_ACC_MAX = 0.60       # amputated probe must fail...  (pilot 0.50)
CONTROL_MARGIN_MIN = 0.10    # ...and by a margin           (pilot 0.375)
MIN_CLASS_TRAIN = 6          # leg trips per class, train            (8/8)
MIN_CLASS_TEST = 3           # ...and test                           (4/4)

# ── the probe: random Fourier features + ridge, one fixed draw ──────────
N_RFF = 200
RFF_SEED = 20260919
RIDGE_LAMBDA = 1.0
N_SHUFFLE = 20

_CACHE: dict = {}


def _calibration() -> tuple:
    """PS.01's j0/alpha/p_max, or a refusal. This spec has no defaults."""
    b = borrow_metrics("PS.01", ("j0_ms", "alpha",
                                 "mean_power_w_full_strength"))
    if not b.ok:
        return None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values, b.provenance


def _build(seed: int):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from playground import PlaygroundParams, make_playground

    vals, _ = _calibration()
    p = PlaygroundParams(seed=seed)
    if seed > 0:
        p = p.mutate(np.random.RandomState(seed))
    model, data, water = make_playground(p, with_water=True,
                                         with_humanoid=True)
    pool = (2.6, -2.4, p.pool_size, 0.0)
    layer = needs.NeedLayer(model, j0=vals["j0_ms"], alpha=vals["alpha"],
                            p_max=vals["mean_power_w_full_strength"],
                            pool=pool, seed=seed)
    layer.t = needs.DAY_S            # nightfall — PS.05/PS.06's venue
    # Rig assertion (pilot repair 1): the whole lane, endpoint included, must
    # sit clear of the pool's mutated footprint — a wet object pays the
    # skin-wetness multiplier and the cost stops being about mass.
    lane_x = PUSH_START[0] + (D_PUSH + 1.0)
    if (abs(lane_x - 2.6) < p.pool_size + 0.5
            and abs(PUSH_START[1] - (-2.4)) < p.pool_size + 0.5):
        raise RuntimeError("push lane intersects the pool footprint")
    return model, data, water, layer, p


def _object_joints(model) -> dict:
    """qpos/dof addresses of every free-jointed objN (PS.09's helper,
    generalised to the whole affordance substrate)."""
    out = {}
    for i in range(12):
        name = f"obj{i}"
        try:
            bid = int(model.body(name).id)
        except (KeyError, ValueError):
            continue
        jadr = int(model.body_jntadr[bid])
        if jadr < 0 or int(model.body_jntnum[bid]) < 1:
            continue
        out[name] = (int(model.jnt_qposadr[jadr]), int(model.jnt_dofadr[jadr]),
                     bid)
    return out


def _move_free(data, qa: int, da: int, xyz) -> None:
    data.qpos[qa:qa + 3] = xyz
    data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[da:da + 6] = 0.0


def _set_mass(model, data, bid: int, orig_mass: float, orig_inertia,
              m: float) -> None:
    """The ONE place mass enters this file: the model's own arrays, inertia
    scaled with mass (fixed geometry, uniform density). mj_setConst refreshes
    the derived quantities the solver reads."""
    scale = m / orig_mass
    model.body_mass[bid] = m
    model.body_inertia[bid] = np.asarray(orig_inertia) * scale
    mujoco.mj_setConst(model, data)


def _place(model, data, xyz):
    """Teleport the humanoid root, at rest, in the STANDARD lying posture
    (joint angles reset to the model default — a bout must start from an
    identical body, or posture drift correlates bout outcomes; qvel zeroed:
    a teleport must not manufacture an impact — PS.06)."""
    from playground import HUMANOID_NQ, humanoid_index
    q = humanoid_index(model)["qposadr"]
    data.qpos[q + 7:q + HUMANOID_NQ] = model.qpos0[q + 7:q + HUMANOID_NQ]
    data.qpos[q:q + 3] = xyz
    data.qpos[q + 3:q + 7] = LYING_QUAT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _step(model, data, water, layer, ctrl):
    """One decision under the NeedLayer contract (PS.05's loop)."""
    dt = float(model.opt.timestep)
    fs = max(1, int(round(SIM_S / dt)))
    layer.begin_decision()
    if layer.microsleep_zeroed():
        ctrl = ctrl * 0.0
    for _ in range(fs):
        data.ctrl[:] = ctrl
        water.apply(model, data)
        mujoco.mj_step(model, data)
        layer.substep(model, data, dt)
    layer.decide()


def _reset_trip(model, data, layer, joints, t0: float, e0: float,
                w0: float) -> None:
    """A fresh body, a re-parked lane, a re-armed larder guard. All fixture
    rights: the claim is about the world's per-trip pricing, and comparable
    trips need comparable bodies AND comparable lanes (PS.09's reasoning)."""
    gid = int(model.geom(PUSH_OBJ).id)
    z0 = float(np.max(model.geom_size[gid])) + 0.02
    i_park = 0
    for name, (qa, da, _bid) in joints.items():
        if name == PUSH_OBJ:
            _move_free(data, qa, da, (*PUSH_START, z0))
        else:
            _move_free(data, qa, da,
                       (PARK_XY[0], PARK_XY[1] + PARK_DY * i_park, 0.15))
            i_park += 1
    _place(model, data, (PUSH_START[0] - OFFSET * PUSH_U[0],
                         PUSH_START[1] - OFFSET * PUSH_U[1], 0.30))
    # A fresh body per trip includes NOT being dead: the layer's own
    # resurrect, called BEFORE the drink-park it would otherwise undo. A
    # trip that dies still VOIDs the seed (dead_events); this only stops a
    # single death poisoning every later trip's record.
    layer._reset_body_state()
    layer.state = replace(needs.NeedState(), e=float(e0), w=float(w0))
    layer.t = float(t0)
    layer._respawn_at = {name: 0.0 for name in layer._food}
    # Drinking is PARKED exactly as in PS.09 (fixture right): the w PRICE
    # still accrues; only the unoffered payoff is parked.
    layer._drink_ready_at = float("inf")


def _ref_power(model, data, water, layer, joints) -> float:
    """P_REF: mean achieved power under ONE fixed commanded sequence from
    the standard pose on a fresh body (PS.05's probe, verbatim)."""
    _reset_trip(model, data, layer, joints, needs.DAY_S, 1.0, 1.0)
    rng = np.random.RandomState(PROBE_SEED)
    ps = []
    for _ in range(REF_PROBE_DEC):
        ctrl = rng.uniform(-S_WORK, S_WORK, model.nu) * layer.gear_scale()
        _step(model, data, water, layer, ctrl)
        ps.append(layer.last_power_w)
    return float(np.mean(ps))


def _features(model, data, layer) -> np.ndarray:
    """[intero-sans-pain 8 | pain 1 | kin 45]: the clock-like interoceptive
    block FIRST so the amputation control is a clean prefix keep — pain and
    the kinematics are the load-bearing senses here (see the docstring)."""
    from playground import HUMANOID_NQ, humanoid_index
    ix = humanoid_index(model)
    q, d = ix["qposadr"], ix["dofadr"]
    kin = np.concatenate([data.qpos[q + 2:q + HUMANOID_NQ],
                          data.qvel[d:d + 23]])
    ob = layer.obs()
    v = np.concatenate([ob[:INTERO_CLOCK_DIM], ob[INTERO_CLOCK_DIM:], kin])
    if v.shape[0] != INTERO_CLOCK_DIM + PAIN_DIM + KIN_DIM:
        raise RuntimeError(f"feature vector is {v.shape[0]}, not "
                           f"{INTERO_CLOCK_DIM + PAIN_DIM + KIN_DIM}")
    return v


def _obj_state(model, data, joints) -> tuple:
    qa, da, _bid = joints[PUSH_OBJ]
    xy = np.array(data.qpos[qa:qa + 2], dtype=float)
    speed = float(np.linalg.norm(data.qvel[da:da + 3]))
    return xy, speed


def _push_trip(model, data, water, layer, joints, mass_idx: int,
               rng: np.random.RandomState, *, equalise: bool,
               leg: bool = False, t0: float = None, e0: float = E0_COST,
               w0: float = W0_COST) -> dict:
    """One transport trip for MASSES[mass_idx]. Live and twin trips push
    until progress >= D_PUSH (or the BOUT_MAX budget dies — a BODY finding).
    Leg trips run exactly ONE bout and take the early-push rows. Cost is the
    endpoint drain of the layer's own channels over the trip."""
    qa, da, bid = joints[PUSH_OBJ]
    orig_m = _push_trip._orig[0]
    orig_I = _push_trip._orig[1]
    true_m = MASSES[0] if equalise else MASSES[mass_idx]
    _set_mass(model, data, bid, orig_m, orig_I, true_m)
    _reset_trip(model, data, layer, joints,
                needs.DAY_S if t0 is None else t0, e0, w0)
    ate0 = sum(layer.ate_total.values())
    drank0 = layer.drank_total

    if leg:
        # The uncounted pre-roll (PS.05's fake-3): random work/rest at the
        # standard spot so pose/fatigue/temperature start at random offsets.
        for _ in range(int(rng.uniform(0.0, PREROLL_MAX_S) / SIM_S)):
            ctrl = (rng.uniform(-S_WORK, S_WORK, model.nu)
                    * layer.gear_scale()
                    if rng.rand() < P_PREROLL_WORK else np.zeros(model.nu))
            _step(model, data, water, layer, ctrl)
            if layer.dead:
                return {"dead": 1.0}

    qa2, da2, _ = joints[PUSH_OBJ]
    gid = int(model.geom(PUSH_OBJ).id)
    z0 = float(np.max(model.geom_size[gid])) + 0.02
    u = np.array(PUSH_U, dtype=float)
    start_xy = np.array(PUSH_START, dtype=float)
    e_start, w_start = layer.state.e, layer.state.w

    rows = []
    s_total = 0.0
    still_ok = 1.0
    n_bouts = N_LEG_BOUTS if leg else N_BOUTS
    for bout in range(n_bouts):
        # Re-centre the OBJECT to the lane start (at rest) and the BODY to
        # the standard posture behind it: each bout is an independent draw of
        # "this body shoves this mass", and the trip's displacement is the
        # SUM of the per-bout displacements. Both teleports are fixture
        # rights (the PS.09 parking class) and both are gated below against
        # manufacturing motion.
        _move_free(data, qa2, da2, (*PUSH_START, z0))
        _place(model, data, (PUSH_START[0] + OFFSET * (-u[0]),
                             PUSH_START[1] + OFFSET * (-u[1]), 0.30))
        _, speed_after = _obj_state(model, data, joints)
        if speed_after > OBJ_STILL_MAX:
            still_ok = 0.0
        for dec in range(BOUT_DEC):
            ctrl = rng.uniform(-PUSH_SCALE, PUSH_SCALE,
                               model.nu) * layer.gear_scale()
            _step(model, data, water, layer, ctrl)
            if layer.dead:
                return {"dead": 1.0}
            if leg and (dec + 1) in ROW_DECS:
                rows.append(_features(model, data, layer))
        obj_xy, _ = _obj_state(model, data, joints)
        # Displacement is the DISTANCE MOVED, not a projection: the body's
        # legs reach past the object's near face and strike it in both
        # directions, and signed increments cancel for exactly the masses
        # that resist most (measured on the pilot: 3.0 kg summed to 0.057 m
        # of projection from 0.49 m of actual motion). Mass resists being
        # moved in ANY direction; the norm is the honest coupling.
        s_total += float(np.linalg.norm(obj_xy - start_xy))

    cost = (e_start - layer.state.e) + (w_start - layer.state.w)
    displaced = float(s_total >= D_PUSH)
    # The outcome: the registered displacement priced at this trip's
    # measured cost-per-metre over the FIXED schedule.
    cost_d = cost * D_PUSH / max(s_total, 1e-9) if displaced else float("nan")
    return {"dead": 0.0, "cost": float(cost), "cost_d": float(cost_d),
            "s_total": float(s_total), "displaced": displaced,
            "still_ok": still_ok,
            "stray_eats": float(sum(layer.ate_total.values()) - ate0),
            "stray_drinks": float(layer.drank_total - drank0),
            "rows": rows,
            "heavy": float(mass_idx >= HEAVY_MIN_IDX)}


_push_trip._orig = None


def _fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    rng = np.random.RandomState(RFF_SEED)
    d = Ztr.shape[1]
    W = rng.randn(d, N_RFF) / math.sqrt(d)
    b = rng.uniform(0.0, 2.0 * math.pi, N_RFF)
    Ptr = np.cos(Ztr @ W + b) * math.sqrt(2.0 / N_RFF)
    Pte = np.cos(Zte @ W + b) * math.sqrt(2.0 / N_RFF)
    ybar = ytr.mean()
    A = Ptr.T @ Ptr + RIDGE_LAMBDA * np.eye(N_RFF)
    beta = np.linalg.solve(A, Ptr.T @ (ytr - ybar))
    return Pte @ beta + ybar


def _bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy over rows: chance = 0.5 under any class mix, and a
    majority-class predictor scores 0.5 (PS.09's guard)."""
    t, p = y_true > 0.0, y_pred > 0.0
    pos, neg = t, ~t
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.0
    return 0.5 * (float((p[pos]).mean()) + float((~p[neg]).mean()))


def _score(trips, n_cols: int, shuffle: bool = False) -> float:
    """Held-out-BY-TRIP balanced accuracy on the first n_cols features; the
    probe regresses the class in {-1,+1} and is judged on the sign. Each
    trip is (rows, y). shuffle=True: mean over N_SHUFFLE TRIP-LEVEL
    permutations of the train pairing (PS.09's honest null for rows that
    share a label)."""
    tr, te = trips[:-N_TEST_TRIPS], trips[-N_TEST_TRIPS:]
    Xtr = np.array([r[:n_cols] for t in tr for r in t[0]])
    Xte = np.array([r[:n_cols] for t in te for r in t[0]])
    yte = np.array([t[1] for t in te for _ in t[0]])
    if not shuffle:
        ytr = np.array([t[1] for t in tr for _ in t[0]])
        return _bal_acc(yte, _fit_predict(Xtr, ytr, Xte))
    ys = np.array([t[1] for t in tr])
    scores = []
    for kk in range(N_SHUFFLE):
        perm = np.random.RandomState(RFF_SEED + 1 + kk).permutation(ys)
        ytr = np.array([perm[i] for i, t in enumerate(tr) for _ in t[0]])
        scores.append(_bal_acc(yte, _fit_predict(Xtr, ytr, Xte)))
    return float(np.mean(scores))


QVEL_SLICE = slice(INTERO_CLOCK_DIM + PAIN_DIM + 22, None)   # the 23 qvel
CLOCK_SLICE = slice(0, INTERO_CLOCK_DIM)                     # amputated form


def _speed_score(trips, sl: slice, shuffle: bool = False) -> float:
    """THE REGISTERED PROBE: a fixed-form single-statistic threshold on the
    sensory rows — per trip, the mean over rows of mean |x[sl]|; the sign
    and threshold (train median) are fit on the TRAIN trips only; balanced
    accuracy on the held-out trips. The form is fixed at registration; only
    its two scalars are learned. shuffle=True: mean over N_SHUFFLE
    trip-level permutations of the train labels (the honest null). The
    amputated control runs the IDENTICAL form on the clock-like block."""
    def stat(t):
        return float(np.mean([np.mean(np.abs(r[sl])) for r in t[0]]))
    tr, te = trips[:-N_TEST_TRIPS], trips[-N_TEST_TRIPS:]
    s_tr = np.array([stat(t) for t in tr])
    s_te = np.array([stat(t) for t in te])
    thr = float(np.median(s_tr))
    yte = np.array([t[1] for t in te])
    ys = np.array([t[1] for t in tr])

    def acc_for(ytr):
        sign = max((1.0, -1.0), key=lambda s: _bal_acc(ytr, s * (s_tr - thr)))
        return _bal_acc(yte, sign * (s_te - thr))

    if not shuffle:
        return acc_for(ys)
    return float(np.mean([
        acc_for(np.random.RandomState(RFF_SEED + 1 + k).permutation(ys))
        for k in range(N_SHUFFLE)]))


def _collect(seed: int) -> dict:
    """Every simulation this spec needs, once. Cached: the control re-scores
    the same rows on the clock-like prefix (PS.09)."""
    if seed in _CACHE:
        return _CACHE[seed]
    vals, prov = _calibration()
    if vals is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]
    try:
        model, data, water, layer, _p = _build(seed)
    except RuntimeError:
        _CACHE[seed] = {"bad_lane": True}
        return _CACHE[seed]
    joints = _object_joints(model)
    if PUSH_OBJ not in joints:
        _CACHE[seed] = {"no_object": True}
        return _CACHE[seed]
    bid = joints[PUSH_OBJ][2]
    _push_trip._orig = (float(model.body_mass[bid]),
                        np.array(model.body_inertia[bid], dtype=float))
    p_ref = _ref_power(model, data, water, layer, joints)

    dead = 0.0
    live = {i: [] for i in range(len(MASSES))}
    twin = {i: [] for i in range(len(MASSES))}
    for rep in range(N_REP):
        for i in range(len(MASSES)):
            # PAIRED draws: the same ctrl stream for every mass within a rep
            # (and its twin), so a rep is a within-draw mass comparison.
            rng_l = np.random.RandomState(seed * 977 + rep * 31)
            r = _push_trip(model, data, water, layer, joints, i, rng_l,
                           equalise=False)
            rng_t = np.random.RandomState(seed * 977 + rep * 31)
            t = _push_trip(model, data, water, layer, joints, i, rng_t,
                           equalise=True)
            dead += r.get("dead", 0.0) + t.get("dead", 0.0)
            if dead == 0.0:
                live[i].append(r)
                twin[i].append(t)

    leg = []
    for j in range(N_LEG):
        rng = np.random.RandomState(seed * 6421 + 17 * j + 5)
        i = LEG_CYCLE[j % len(LEG_CYCLE)]
        t0 = needs.DAY_S + float(rng.uniform(0.0, LEG_T_JITTER_S))
        e0 = float(rng.uniform(*LEG_E0))
        w0 = float(rng.uniform(*LEG_W0))
        r = _push_trip(model, data, water, layer, joints, i, rng,
                       equalise=False, leg=True, t0=t0, e0=e0, w0=w0)
        dead += r.get("dead", 0.0)
        if dead == 0.0:
            leg.append(r)

    _CACHE[seed] = {"p_ref": p_ref, "p_max":
                    vals["mean_power_w_full_strength"], "dead": dead,
                    "live": live, "twin": twin, "leg": leg}
    del model, data, water
    return _CACHE[seed]


def _experiment(seed: int) -> dict:
    d = _collect(seed)
    if "refused" in d:
        return {"borrow_ok": 0.0, "rig_ok": 0.0, "seed_gates_ok": 0.0}
    if d.get("no_object") or d.get("bad_lane"):
        return {"borrow_ok": 1.0, "rig_ok": 0.0, "seed_gates_ok": 0.0,
                "no_object": float(bool(d.get("no_object"))),
                "bad_lane": float(bool(d.get("bad_lane")))}
    if d["dead"] > 0.0:
        return {"borrow_ok": 1.0, "rig_ok": 0.0, "seed_gates_ok": 0.0,
                "dead_events": float(d["dead"])}

    live, twin, leg = d["live"], d["twin"], d["leg"]
    idx = range(len(MASSES))
    all_trips = ([t for i in idx for t in live[i]]
                 + [t for i in idx for t in twin[i]] + leg)
    displaced_ok = float(all(t["displaced"] == 1.0
                             for i in idx for t in live[i] + twin[i]))
    still_ok = float(all(t["still_ok"] == 1.0 for t in all_trips))
    stray_ok = float(all(t["stray_eats"] == 0.0 and t["stray_drinks"] == 0.0
                         for t in all_trips))

    cost_mean = [float(np.mean([t["cost_d"] for t in live[i]])) for i in idx]
    cost_std = [float(np.std([t["cost_d"] for t in live[i]])) for i in idx]
    twin_mean = [float(np.mean([t["cost_d"] for t in twin[i]])) for i in idx]
    s_mean = [float(np.mean([t["s_total"] for t in live[i]])) for i in idx]
    quantum = max(max(cost_std), 1e-9)
    thresh = max(GAP_ABS_MIN, QUANTUM_MULT * quantum)
    gap = cost_mean[-1] - cost_mean[0]
    mono_ok = float(all(cost_mean[i + 1] > cost_mean[i]
                        for i in range(len(MASSES) - 1)))
    twin_spread = max(twin_mean) - min(twin_mean)
    # The twin must FAIL the very bar the live gap must clear (docstring).
    twin_flat_ok = float(twin_spread < thresh)

    trips = [(t["rows"], 1.0 if t["heavy"] == 1.0 else -1.0)
             for t in leg if t["rows"]]
    tr, te = trips[:-N_TEST_TRIPS], trips[-N_TEST_TRIPS:]
    n_h_tr = sum(1 for t in tr if t[1] > 0.0)
    n_l_tr = sum(1 for t in tr if t[1] <= 0.0)
    n_h_te = sum(1 for t in te if t[1] > 0.0)
    n_l_te = sum(1 for t in te if t[1] <= 0.0)
    n_cols = INTERO_CLOCK_DIM + PAIN_DIM + KIN_DIM
    leg_ok = float(len(trips) == N_LEG
                   and min(n_h_tr, n_l_tr) >= MIN_CLASS_TRAIN
                   and min(n_h_te, n_l_te) >= MIN_CLASS_TEST)
    acc = _speed_score(trips, QVEL_SLICE) if leg_ok else 0.0
    acc_shuf = _speed_score(trips, QVEL_SLICE, shuffle=True) if leg_ok else 0.0
    diag = _score(trips, n_cols) if leg_ok else 0.0

    m = {
        "borrow_ok": 1.0,
        # rig: could this run test the claim at all? (VOID territory)
        "dead_events": 0.0,
        "fresh_frac": d["p_ref"] / d["p_max"],
        "displaced_ok": displaced_ok,
        "still_ok": still_ok,
        "stray_ok": stray_ok,
        # the cost table
        "cost_m1": cost_mean[0], "cost_m2": cost_mean[1],
        "cost_m3": cost_mean[2], "cost_m4": cost_mean[3],
        "twin_m1": twin_mean[0], "twin_m2": twin_mean[1],
        "twin_m3": twin_mean[2], "twin_m4": twin_mean[3],
        "s_m1": s_mean[0], "s_m4": s_mean[3],
        "quantum": quantum,
        "gap": gap,
        "mono_ok": mono_ok,
        "twin_spread": twin_spread,
        "twin_flat_ok": twin_flat_ok,
        # legibility
        "n_heavy_train": float(n_h_tr), "n_light_train": float(n_l_tr),
        "n_heavy_test": float(n_h_te), "n_light_test": float(n_l_te),
        "leg_ok": leg_ok,
        "probe_bal_acc": acc,
        "shuffled_bal_acc": acc_shuf,
        "diag_rff_acc": diag,        # the family RFF idiom: REPORTED, ungated
        "n_features": float(n_cols),
    }
    m["rig_ok"] = float(
        m["fresh_frac"] >= FRESH_FRAC_MIN
        and m["displaced_ok"] == 1.0
        and m["still_ok"] == 1.0
        and m["stray_ok"] == 1.0)
    m["seed_gates_ok"] = float(
        m["rig_ok"] == 1.0
        and m["mono_ok"] == 1.0
        and m["gap"] >= thresh
        and m["twin_flat_ok"] == 1.0
        and m["leg_ok"] == 1.0
        and m["probe_bal_acc"] >= ACC_MIN
        and m["shuffled_bal_acc"] <= SHUF_ACC_MAX)
    return m


def _control(seed: int) -> dict:
    """THE SENSORY AMPUTATION (PS.02's control, reused): the same rows with
    the load-bearing channels — pain and the kinematic block — deleted,
    keeping only the 8 clock-like interoceptive dims. If the probe still
    reads the class it was reading the episode clock, and the senses earned
    nothing."""
    d = _collect(seed)
    if ("refused" in d or d.get("no_object") or d.get("bad_lane")
            or d["dead"] > 0.0):
        return {"control_bal_acc": 0.0, "control_caught": 0.0}
    trips = [(t["rows"], 1.0 if t["heavy"] == 1.0 else -1.0)
             for t in d["leg"] if t["rows"]]
    if len(trips) < N_TEST_TRIPS + 2:
        return {"control_bal_acc": 0.0, "control_caught": 0.0}
    acc_c = _speed_score(trips, CLOCK_SLICE)
    return {"control_bal_acc": acc_c,
            "control_n_features": float(INTERO_CLOCK_DIM),
            "control_caught": float(acc_c <= CONTROL_ACC_MAX)}


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        # An uncalibrated world refutes nothing. VOID, never FAIL (T0.22).
        return Status.VOID
    if m.get("rig_ok", 0.0) != 1.0:
        # A dead body, an undisplaced mass, a popped placement or stray
        # consumption is a BODY/rig finding — the run could not test the
        # claim (registry, verbatim).
        return Status.VOID
    world = bool(m["mono_ok"] == 1.0
                 and m["gap"] >= max(GAP_ABS_MIN, QUANTUM_MULT * m["quantum"])
                 and m["twin_flat_ok"] == 1.0)
    if not world:
        # A falsified_by branch fired: mass is free, non-monotone, or the
        # gap is under the quantum. A real red, not a rig one.
        return False
    if m.get("leg_ok", 0.0) != 1.0:
        # The world half held but the probe's row set starved — the
        # legibility instrument could not run. VOID, never FAIL (T0.22).
        return Status.VOID
    return bool(
        m["seed_gates_ok"] == 1.0
        and m["probe_bal_acc"] >= ACC_MIN
        and m["shuffled_bal_acc"] <= SHUF_ACC_MAX
        and c["control_caught"] == 1.0
        and c["control_bal_acc"] <= CONTROL_ACC_MAX
        and (m["probe_bal_acc"] - c["control_bal_acc"]) >= CONTROL_MARGIN_MIN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PS.08"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
