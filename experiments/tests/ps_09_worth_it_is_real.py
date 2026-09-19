"""PS.09 — Worth-it is a real distinction: the world pays and charges in one
currency, and some trips do not pay.

HYPOTHESIS (registry, unchanged). W0's ledger makes net value computable and
NON-VACUOUS: resource payoffs and traversal prices are the SAME need-currency
(floor food restores nu = 0.08, needs.py:157, while PS.05 MEASURED the
traversal price at 0.016-0.069 over 1.5-6 m — the payoff sits INSIDE the
price span, so a break-even exists in-world at registration time), and across
>= 4 pre-registered (payoff, distance) offers straddling that break-even, a
scripted collect-trip's NET need-gain (restored minus drained, both read off
the existing needs.py channels — no new instrument) is POSITIVE for at least
one registered offer and NEGATIVE for at least one, on every seed, each
side's margin exceeding the venue's outcome-metric quantum; and the sign is
LEGIBLE beforehand — a probe on the PRE-DEPARTURE sensory vector predicts the
trip's net sign above chance before the first step.

## THE FIXTURE — PS.05's, plus a real collect

Everything PS.05 pre-registered about traversal is inherited byte-for-byte:
the body has no walking controller, so a trip of distance d is scripted
walking effort (S_WORK thrash, gear-scaled) whose REAL metabolic price the
REAL NeedLayer integrates, while a virtual position advances at the
power-coupled speed v = V_WALK * clip(p / P_REF, 0, V_CLIP_MULT). The venue
is night (PS.06's finding: K_DRY makes daylight exertion cook the body).
What PS.09 adds is the ARRIVAL: the trip ends with the body actually EATING.

  * EATING IS REAL, NEVER SYNTHESISED (the PS.08 tautology-guard rule,
    applied to payoffs): at arrival the fixture serves a floor food by
    placing it just above the head — PS.01's `_serve` precedent, verbatim in
    spirit: "eating still has to happen through the layer's real contact
    test". The payoff enters `e` only via `NeedLayer.substep`'s mouth-gated
    contact scan (needs.py:474) and `decide()`'s own `+ self._ate`. Nothing
    in this file writes a need channel.
  * THE OFFER is (k, d): k floor foods (nu = 0.08 each, needs.py:157) at
    distance d. The arrival dwell is structurally IDENTICAL across k — always
    EAT_SLOTS serve-slots of EAT_DEC decisions; a slot beyond k serves
    nothing — so the k=2 payoff differs from k=1 by the eaten nu alone,
    never by dwell time.
  * FOODS ARE PARKED at a far corner at every trip reset (fixture right, the
    same class as serving): obj0/obj1 spawn wherever the world mutation put
    them, and a food that happened to sit at SPOT would feed the walking
    thrash mid-trip. Parking + the stray-consumption rig gate (zero eats
    before arrival, zero drinks ever) make the payoff attributable to the
    offer. Respawn timers are reset per trip for the same reason the body
    is: comparable trips need comparable larders.
  * NET = (e_end - e_start) + (w_end - w_start), snapshot at DEPARTURE
    (after the pre-departure sniffs) and after the arrival dwell — endpoint
    deltas of the layer's own channels, no new meter (PS.05's accounting,
    extended through the dwell so the price of stopping to eat is paid).

## THE FOUR WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The negative branch is a payoff-accounting artifact** — anything that
   under-credits the payoff (a clipped e, a failed eat, a dwell charged only
   to far offers) would manufacture "not worth it". Caught by the registry's
   null, THE FREE-LUNCH TWIN: the same offers with traversal instantaneous
   (PS.05's teleport twin) and the SAME fixed arrival dwell + eat. Under it
   net = payoff minus ~nothing, so the negative branch must VANISH: every
   twin offer must read positive by a pre-registered floor. (Clipping is
   also excluded at the rig: e0 bands are set so e_end < 1 by margin.)
2. **The eat itself is fake** — a food teleported into the mouth might be
   "eaten" by fixture arithmetic. It is not: the rig gate requires exactly k
   eat EVENTS on the layer's own `ate_total` counter per trip (mouth-gated,
   needs.py:29), and a trip that cannot collect VOIDs the run on the rig —
   a BODY/rig finding, never a world verdict (registry, verbatim).
3. **The margins are smaller than the venue can resolve.** The QUANTUM
   (LC.03/DP.04 resolution lesson) is the largest within-offer repeat spread
   of NET — the same offer re-run under different ctrl draws on a fresh
   body, pure rig noise (no discrete drain events can enter: eating is the
   measured payoff itself, drinking is gated to zero, and p stays far below
   the microsleep floor on a fresh body inside one night). Each side's
   witness margin must clear QUANTUM_MULT x that floor AND an absolute floor.
4. **The probe reads a schedule artifact, not a sense.** Pre-departure rows
   carry no en-route clock (there is no elapsed time yet), but a lazy rig
   would let trip index leak through state drift. Killed at the rig
   (PS.05's fake-3 machinery, inherited): jittered night start time,
   jittered e0/w0, and a random-duration uncounted PRE-ROLL of work/rest so
   pose/fatigue/temperature start at random offsets. What remains
   offer-bearing is the declared sense: the offer emits odour
   (odour.StaticField at the target, strength = k, SM.01's certified
   sensor, nose noise on), and the odour block sits LAST in the feature
   vector so the registry's control — THE SENSORY AMPUTATION — is a clean
   suffix drop. The amputated probe must FAIL, and the shuffled pairing
   must collapse to chance.

## LEGIBILITY: WHAT THE ROWS ARE, AND THE DISCLOSED VENUE FACT

Per legibility trip: an offer drawn from three pre-registered STRATA in a
fixed cycle (NEG, PN, NEG, PF — see constants; the strata bracket the
break-even PS.05's price curve locates near d* ~ 6 m for k=1, and exist so
both classes populate train AND test — the PS.05 band-lottery lesson applied
at the rig, never at the gate). y is the trip's own MEASURED net; rows are
honestly labeled whatever a mutated world's prices do to the strata's
intent, and a class starved below the pre-registered minima is a VOID
(starved instrument), never a quiet re-draw. Before departure the agent
stands at SPOT and takes N_SNIFF sniff rows (feature vector: 45 kinematic +
9 interoceptive + 12 odour dims, odour last), heading = the offer's bearing.
Held out BY TRIP; the probe is PS.05's exact scorer (200 random Fourier
features + ridge, one fixed draw) regressing net, judged on BALANCED sign
accuracy (chance = 0.5 under any class mix; a majority-class probe scores
0.5). DISCLOSED VENUE FACT: one static sniff cannot separate source
strength from distance (C = k*exp(-d/LAMBDA)), so offers in the k=1-far /
k=2-farther concentration overlap are genuinely ambiguous to this nose;
the bar prices that in — it is a fact about W0's smell, reported, not
hidden.

## PRE-REGISTERED, before the first registered run (T3.06 lesson)

Registered offers OFFERS = ((1, 1.5), (2, 6.0), (1, 7.5), (1, 9.0)) —
(k, metres). Expected signs at registration (PS.05's measured curve + the
dwell price): +, +, -, - — but the CLAIM binds only "at least one each
side, margins over quantum", so a mutated world may relocate the break-even
without voiding the question. d = 9 m exceeds PS.05's proven 6 m reach and
re-earns it here at the rig gate (registry, verbatim); C(9 m) = 0.011 for
k=1, 11x the 1e-3 nose noise. All gates below were frozen after a seed-90
pilot (disjoint from registered seeds 0/1/2 — the PG.6/SM.01/PS.02
convention); the pilot reading sits beside each gate. Deaths, a stale
calibration, an unreached or uneaten offer, stray consumption, or a starved
row set are RIG findings: Status.VOID, never FAIL (T0.22). FAIL is reserved
for the four falsified_by branches: no positive offer, no negative offer,
sub-quantum margins, illegible sign.

PILOT (seed 90, 2026-09-19, final — after the two rig repairs below): nets
by offer (mean over 3 reps) +0.0418 / +0.0607 / -0.0420 / -0.0613; twin
nets +0.0462 / +0.1262 / +0.0462 / +0.0462 (every twin positive: the
negative branch vanishes under free traversal); quantum (max within-offer
repeat std) 0.00357, so the weakest witness margin is 11.7x the quantum;
fresh_frac 0.978; probe balanced sign accuracy 0.78, shuffled pairing
0.488, amputated control 0.52 (margin 0.26); train classes 11 NEG / 11
POS, test 5/5. The two repairs the pilot forced, both at the RIG (gates
untouched): (1) a 30 cm food drop missed a thrash-displaced head on 2 of
32 leg trips and one head thrashed into the mutated pool's region and
DRANK (+0.9-worth of unsolicited w-payoff on a foraging row) — three
poisoned labels; the serve now INTERSECTS the head geom inside a fixed
4-slot retry dwell (structurally identical across trips and k, stragglers
parked so a retry cannot overshoot k), and drinking is parked at trip
reset exactly as the foods are. (2) The 8/10-correct test-trip ceiling is
the disclosed venue fact in action: both misses sit in the k-d
concentration overlap where a static exponential field is genuinely
ambiguous (|grad C|/C = 1/LAMBDA regardless of strength, so no local sniff
separates a big far offer from a small near one — only the offer
distribution's statistics do).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco                                              # noqa: E402

from .. import needs, odour                                # noqa: E402
from ..protocol import Ledger, Status, borrow_metrics, run_spec  # noqa: E402
from ..registry import BY_ID                               # noqa: E402

# The claim is about the WORLD's pricing and paying in one currency, and the
# sense that carries the offer.
IMPL_DEPS = ["experiments/needs.py", "experiments/odour.py", "playground.py"]

# ── the rollout, fixed before the run (PS.05's constants where shared) ──
SIM_S = 0.2                  # s per decision (NE.01's cadence)
S_WORK = 0.4                 # ctrl scale: PS.01's calibrated regime
PROBE_SEED = 4242            # ONE commanded sequence for P_REF, everywhere
REF_PROBE_DEC = 50           # 10 s fresh reference probe -> P_REF
V_WALK = 0.4                 # m/s, the registered walking pace
V_CLIP_MULT = 1.25           # v may not exceed this x V_WALK
REACH_EPS = 0.15             # m, arrival radius
OFFERS = ((1, 1.5), (2, 6.0), (1, 7.5), (1, 9.0))   # (k foods, metres)
N_REP = 3                    # trips per registered offer
OFFER_BUDGET_MULT = 2.5      # alive-proof window, offer trips (PS.05)
LEG_BUDGET_MULT = 3.0        # ...leg trips start at jittered-low e0: slower
TWIN_SETTLE_DEC = 20         # 4 s fixed teleport-arrival interval
EAT_SLOTS = 4                # arrival dwell: always this many serve-slots
EAT_DEC = 10                 # 2 s per slot; a slot re-serves while ate < k
FLOOR_FOODS = ("obj0", "obj1")   # nu = 0.08 each (needs.py FOOD)
PARK_XY = ((2.8, 2.8), (2.8, 2.2))   # per-trip food park, far from SPOT
E0_OFFER = 0.70              # offer-trip start energy: payoff cannot clip
W0_OFFER = 1.00              # water only drains — no clip risk at the top
N_LEG = 32                   # legibility trips; held out BY TRIP
N_TEST_TRIPS = 10
STRATA = {"NEG": (1, 7.0, 9.0),      # (k, d_lo, d_hi) — expected negative
          "PN":  (1, 1.5, 4.0),      # positive-near
          "PF":  (2, 5.0, 9.0)}      # positive-far: worth going far FOR
CYCLE = ("NEG", "PN", "NEG", "PF")   # fixed stratum cycle -> ~50/50 classes
N_SNIFF = 5                  # pre-departure sniff rows per leg trip
LEG_T_JITTER_S = 250.0       # trip start time ~ DAY_S + U(0, this)
LEG_E0 = (0.60, 0.80)        # e0 band: k=2 payoff still cannot clip at 1.0
LEG_W0 = (0.70, 1.0)
PREROLL_MAX_S = 20.0         # uncounted random work/rest before a leg trip
P_PREROLL_WORK = 0.6
LYING_QUAT = (0.7071, 0.0, 0.7071, 0.0)
SPOT = (-0.8, -1.0, 0.30)    # NE.01's open flat ground (PS.05's SPOT)
FOOD_SERVE_DZ = 0.30         # served from 30 cm above the head (PS.01)
KIN_DIM = 45                 # qpos[2:24] + qvel[23]: pose+velocity only
ODOUR_DIM = 12               # OdourSensor.obs: 4 ch x [L, R] + 4 derivs

# ── pre-registered gates (final pilot, seed 90, reading beside each) ────
FRESH_FRAC_MIN = 0.50        # alive-proof: P_REF >= this x p_max     (0.978)
NET_ABS_MIN = 0.008          # each side's witness margin, absolute
                             #             (pilot +0.0607 pos, 0.0420 neg)
QUANTUM_MULT = 2.0           # ...and >= this x the measured quantum
                             #        (q = 0.00357; neg margin / q = 11.7)
TWIN_NET_MIN = 0.030         # free lunch: EVERY twin offer net >= this
                             #                        (pilot min +0.0462)
ACC_MIN = 0.65               # the headline: sign legible pre-departure
                             #      (pilot 0.78; the overlap band is the
                             #       honest ceiling — see the pilot note)
SHUF_ACC_MAX = 0.60          # trip-level shuffled pairing, mean of 20
                             #                              (pilot 0.488)
CONTROL_ACC_MAX = 0.60       # amputated probe must fail...    (pilot 0.52)
CONTROL_MARGIN_MIN = 0.10    # ...and by a margin              (pilot 0.26)
MIN_CLASS_TRAIN = 6          # measured-sign trips per class, train  (11/11)
MIN_CLASS_TEST = 3           # ...and test                            (5/5)

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
    return model, data, water, layer


def _food_joints(model) -> dict:
    """qpos/dof addresses of each floor food's free joint (PS.01's helper)."""
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


def _move_food(data, joints: dict, name: str, xyz) -> None:
    qa, da = joints[name]
    data.qpos[qa:qa + 3] = xyz
    data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[da:da + 6] = 0.0


def _place(model, data, xyz):
    """Teleport the humanoid root, at rest (qvel zeroed: a teleport must not
    manufacture an impact — PS.06)."""
    from playground import humanoid_index
    q = humanoid_index(model)["qposadr"]
    data.qpos[q:q + 3] = xyz
    data.qpos[q + 3:q + 7] = LYING_QUAT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _reset_trip(model, data, layer, joints, t0: float, e0: float,
                w0: float) -> None:
    """A fresh body at the standard spot, at night; foods PARKED far away;
    respawn timers cleared. All three are fixture rights: the claim is about
    the world's per-offer pricing and paying, and comparable offers need
    comparable bodies AND comparable larders (see the fixture section)."""
    _place(model, data, SPOT)
    for i, name in enumerate(joints):
        _move_food(data, joints, name, (*PARK_XY[i], 0.15))
    mujoco.mj_forward(model, data)
    layer.state = replace(needs.NeedState(), e=float(e0), w=float(w0))
    layer.t = float(t0)
    layer._respawn_at = {name: 0.0 for name in layer._food}
    # Drinking is PARKED exactly as the foods are (fixture right): the offer
    # is food, and a head that thrashes into a mutated pool's region would
    # collect an unsolicited w-payoff that corrupts the trip's net (measured:
    # one leg trip in the seed-90 pilot, +0.9-worth of drink on a foraging
    # row). The w PRICE still accrues; only the unoffered payoff is parked.
    layer._drink_ready_at = float("inf")


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


def _features(model, data, layer, odour_obs: np.ndarray) -> np.ndarray:
    """Pose + velocity, then interoception, then the ODOUR BLOCK LAST so the
    amputation control is a clean suffix drop (PS.05, verbatim)."""
    from playground import HUMANOID_NQ, humanoid_index
    ix = humanoid_index(model)
    q, d = ix["qposadr"], ix["dofadr"]
    kin = np.concatenate([data.qpos[q + 2:q + HUMANOID_NQ],
                          data.qvel[d:d + 23]])
    v = np.concatenate([kin, layer.obs(), odour_obs])
    if v.shape[0] != KIN_DIM + needs.NEED_DIM + ODOUR_DIM:
        raise RuntimeError(f"feature vector is {v.shape[0]}, not "
                           f"{KIN_DIM + needs.NEED_DIM + ODOUR_DIM}")
    return v


def _head_pos(model, data, layer) -> np.ndarray:
    gid = int(next(iter(layer._mouth_geoms)))
    return np.array(data.geom_xpos[gid], dtype=float)


def _trip(model, data, water, layer, joints, k: int, d: float,
          bearing: float, p_ref: float, rng: np.random.RandomState, *,
          teleport: bool, sniff: bool = False, t0: float = None,
          e0: float = E0_OFFER, w0: float = W0_OFFER, nose_rng=None,
          budget_mult: float = OFFER_BUDGET_MULT):
    """One collect trip for offer (k, d). Returns the trip's NET need-gain
    (endpoint deltas, departure to end-of-dwell), whether the target was
    REACHED, whether exactly k foods were EATEN through the layer's own
    mouth gate, stray-consumption flags, and — when sniffing — the
    pre-departure legibility rows."""
    _reset_trip(model, data, layer, joints,
                needs.DAY_S if t0 is None else t0, e0, w0)
    rows = []
    if sniff:
        # The uncounted pre-roll (fake 4): random work/rest offsets
        # pose/fatigue/temperature before any row is taken.
        for _ in range(int(rng.uniform(0.0, PREROLL_MAX_S) / SIM_S)):
            ctrl = (rng.uniform(-S_WORK, S_WORK, model.nu)
                    * layer.gear_scale()
                    if rng.rand() < P_PREROLL_WORK else np.zeros(model.nu))
            _step(model, data, water, layer, ctrl)
            if layer.dead:
                return {"dead": 1.0}
    pos = np.array(SPOT[:2], dtype=float)
    tgt = pos + d * np.array([math.cos(bearing), math.sin(bearing)])

    if sniff:
        field = odour.StaticField([odour.Source("offer", "food",
                                                (tgt[0], tgt[1], 0.3),
                                                strength=float(k))])
        sensor = odour.OdourSensor(field)
        for _ in range(N_SNIFF):
            _step(model, data, water, layer, np.zeros(model.nu))
            if layer.dead:
                return {"dead": 1.0}
            ob = sensor.obs(_head_pos(model, data, layer), bearing, layer.t,
                            rng=nose_rng)
            rows.append(_features(model, data, layer, ob))

    # DEPARTURE: the trip's accounting window opens here.
    e_start, w_start = layer.state.e, layer.state.w
    ate_start = sum(layer.ate_total.values())
    drank_start = layer.drank_total

    reached = 0.0
    if teleport:
        pos = tgt.copy()
        reached = 1.0
        for _ in range(TWIN_SETTLE_DEC):
            _step(model, data, water, layer, np.zeros(model.nu))
            if layer.dead:
                return {"dead": 1.0}
    else:
        n_max = int(math.ceil((budget_mult * d / V_WALK) / SIM_S))
        for _ in range(n_max):
            ctrl = rng.uniform(-S_WORK, S_WORK, model.nu) * layer.gear_scale()
            _step(model, data, water, layer, ctrl)
            if layer.dead:
                return {"dead": 1.0}
            v = V_WALK * min(max(layer.last_power_w / p_ref, 0.0),
                             V_CLIP_MULT)
            vec = tgt - pos
            dist = float(np.linalg.norm(vec))
            step_len = min(v * SIM_S, dist)
            if dist > 1e-9:
                pos = pos + vec / dist * step_len
            if float(np.linalg.norm(tgt - pos)) <= REACH_EPS:
                reached = 1.0
                break
    stray_eats = float(sum(layer.ate_total.values()) - ate_start)

    # ARRIVAL DWELL: EAT_SLOTS serve-slots, ALWAYS all of them, so the dwell
    # is structurally identical across trips and across k (fake 1). A slot
    # re-serves the next uneaten food INTERSECTING the head geom (guaranteed
    # mouth contact at the next scan — the seed-90 pilot measured a 30 cm
    # drop missing a thrash-displaced head on 2 of 32 trips, and a missed
    # eat mislabels the trip's net), alternating foods so a respawn
    # refractory never blocks a retry.
    names = list(joints)
    served = 0
    for slot in range(EAT_SLOTS):
        # Park every food first: a served-but-missed straggler lying against
        # the head must not be eaten by accident in a later slot (it would
        # overshoot k), then serve the next one if the count is still short.
        for i, name in enumerate(names):
            _move_food(data, joints, name, (*PARK_XY[i], 0.15))
        eaten = sum(layer.ate_total.values()) - ate_start
        if reached == 1.0 and eaten < k:
            head = _head_pos(model, data, layer)
            r_head = float(model.geom_size[
                int(next(iter(layer._mouth_geoms)))][0])
            _move_food(data, joints, names[served % len(names)],
                       (head[0], head[1], head[2] + 0.5 * r_head))
            served += 1
        mujoco.mj_forward(model, data)
        for _ in range(EAT_DEC):
            _step(model, data, water, layer, np.zeros(model.nu))
            if layer.dead:
                return {"dead": 1.0}
    ate = float(sum(layer.ate_total.values()) - ate_start)
    net = (layer.state.e - e_start) + (layer.state.w - w_start)
    return {"dead": 0.0, "net": float(net), "reached": reached,
            "ate_ok": float(reached == 1.0 and ate == float(k)
                            and stray_eats == 0.0),
            "no_drink": float(layer.drank_total == drank_start),
            "e_end": float(layer.state.e), "rows": rows}


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
    """Balanced sign accuracy over rows: chance = 0.5 under any class mix,
    and a majority-class predictor scores 0.5 (the class-imbalance guard)."""
    t, p = y_true > 0.0, y_pred > 0.0
    pos, neg = t, ~t
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.0
    return 0.5 * (float((p[pos]).mean()) + float((~p[neg]).mean()))


def _score(trips, n_cols: int, shuffle: bool = False) -> float:
    """Held-out-BY-TRIP balanced sign accuracy on the first n_cols features;
    the probe regresses net (PS.05's scorer) and is judged on the sign. Each
    trip is (rows, y). shuffle=True: mean over N_SHUFFLE TRIP-LEVEL
    permutations of the train pairing — rows keep their trip structure and
    trips swap labels, the honest null for rows that share a label."""
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


def _collect(seed: int) -> dict:
    """Every simulation this spec needs, once. Cached: the control re-scores
    the same rows minus the odour suffix (PS.05)."""
    if seed in _CACHE:
        return _CACHE[seed]
    vals, prov = _calibration()
    if vals is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]
    model, data, water, layer = _build(seed)
    joints = _food_joints(model)
    if len(joints) < len(FLOOR_FOODS):
        # A mutated world without its floor foods cannot pose the offer.
        _CACHE[seed] = {"no_food": True}
        return _CACHE[seed]
    p_ref = _ref_power(model, data, water, layer, joints)

    dead = 0.0
    live = {o: [] for o in OFFERS}
    twin = {o: [] for o in OFFERS}
    for rep in range(N_REP):
        for j, (k, d) in enumerate(OFFERS):
            bearing = float(np.random.RandomState(
                seed * 7919 + rep * 101 + j).uniform(0.0, 2.0 * math.pi))
            r = _trip(model, data, water, layer, joints, k, d, bearing,
                      p_ref, np.random.RandomState(seed * 977 + rep * 31 + j),
                      teleport=False)
            t = _trip(model, data, water, layer, joints, k, d, bearing,
                      p_ref, np.random.RandomState(seed * 977 + rep * 31 + j),
                      teleport=True)
            dead += r.get("dead", 0.0) + t.get("dead", 0.0)
            if dead == 0.0:
                live[(k, d)].append(r)
                twin[(k, d)].append(t)

    leg = []
    for i in range(N_LEG):
        rng = np.random.RandomState(seed * 6421 + 17 * i + 5)
        k, dlo, dhi = STRATA[CYCLE[i % len(CYCLE)]]
        d = float(rng.uniform(dlo, dhi))
        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        t0 = needs.DAY_S + float(rng.uniform(0.0, LEG_T_JITTER_S))
        e0 = float(rng.uniform(*LEG_E0))
        w0 = float(rng.uniform(*LEG_W0))
        r = _trip(model, data, water, layer, joints, k, d, bearing, p_ref,
                  rng, teleport=False, sniff=True, t0=t0, e0=e0, w0=w0,
                  nose_rng=np.random.RandomState(seed * 331 + i),
                  budget_mult=LEG_BUDGET_MULT)
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
    if d.get("no_food"):
        return {"borrow_ok": 1.0, "rig_ok": 0.0, "seed_gates_ok": 0.0,
                "no_food": 1.0}
    if d["dead"] > 0.0:
        return {"borrow_ok": 1.0, "rig_ok": 0.0, "seed_gates_ok": 0.0,
                "dead_events": float(d["dead"])}

    live, twin, leg = d["live"], d["twin"], d["leg"]
    all_trips = ([t for o in OFFERS for t in live[o]]
                 + [t for o in OFFERS for t in twin[o]] + leg)
    reach_ok = float(all(t["reached"] == 1.0 for t in all_trips))
    ate_ok = float(all(t["ate_ok"] == 1.0 for t in all_trips))
    no_drink = float(all(t["no_drink"] == 1.0 for t in all_trips))
    e_clip_ok = float(all(t["e_end"] < 0.999 for t in all_trips))

    net_mean = [float(np.mean([t["net"] for t in live[o]])) for o in OFFERS]
    net_std = [float(np.std([t["net"] for t in live[o]])) for o in OFFERS]
    twin_mean = [float(np.mean([t["net"] for t in twin[o]])) for o in OFFERS]
    quantum = max(max(net_std), 1e-9)
    thresh = max(NET_ABS_MIN, QUANTUM_MULT * quantum)
    pos_net_max = max(net_mean)
    neg_net_min = min(net_mean)
    sign_split_ok = float(pos_net_max >= thresh and neg_net_min <= -thresh)
    twin_net_min = min(twin_mean)

    trips = [(t["rows"], t["net"]) for t in leg if t["rows"]]
    tr, te = trips[:-N_TEST_TRIPS], trips[-N_TEST_TRIPS:]
    n_pos_tr = sum(1 for t in tr if t[1] > 0.0)
    n_neg_tr = sum(1 for t in tr if t[1] <= 0.0)
    n_pos_te = sum(1 for t in te if t[1] > 0.0)
    n_neg_te = sum(1 for t in te if t[1] <= 0.0)
    n_cols = KIN_DIM + needs.NEED_DIM + ODOUR_DIM
    leg_ok = float(len(trips) == N_LEG
                   and min(n_pos_tr, n_neg_tr) >= MIN_CLASS_TRAIN
                   and min(n_pos_te, n_neg_te) >= MIN_CLASS_TEST)
    acc = _score(trips, n_cols) if leg_ok else 0.0
    acc_shuf = _score(trips, n_cols, shuffle=True) if leg_ok else 0.0

    m = {
        "borrow_ok": 1.0,
        # rig: could this run test the claim at all? (VOID territory)
        "dead_events": 0.0,
        "fresh_frac": d["p_ref"] / d["p_max"],
        "reach_ok": reach_ok,
        "ate_ok": ate_ok,
        "no_drink": no_drink,
        "e_clip_ok": e_clip_ok,
        # the offer table
        "net_o1": net_mean[0], "net_o2": net_mean[1],
        "net_o3": net_mean[2], "net_o4": net_mean[3],
        "twin_o1": twin_mean[0], "twin_o2": twin_mean[1],
        "twin_o3": twin_mean[2], "twin_o4": twin_mean[3],
        "quantum": quantum,
        "pos_net_max": pos_net_max,
        "neg_net_min": neg_net_min,
        "sign_split_ok": sign_split_ok,
        "twin_net_min": twin_net_min,
        # legibility
        "n_pos_train": float(n_pos_tr), "n_neg_train": float(n_neg_tr),
        "n_pos_test": float(n_pos_te), "n_neg_test": float(n_neg_te),
        "leg_ok": leg_ok,
        "probe_bal_acc": acc,
        "shuffled_bal_acc": acc_shuf,
        "n_features": float(n_cols),
    }
    m["rig_ok"] = float(
        m["fresh_frac"] >= FRESH_FRAC_MIN
        and m["reach_ok"] == 1.0 and m["ate_ok"] == 1.0
        and m["no_drink"] == 1.0 and m["e_clip_ok"] == 1.0)
    m["seed_gates_ok"] = float(
        m["rig_ok"] == 1.0
        and m["sign_split_ok"] == 1.0
        and m["twin_net_min"] >= TWIN_NET_MIN
        and m["leg_ok"] == 1.0
        and m["probe_bal_acc"] >= ACC_MIN
        and m["shuffled_bal_acc"] <= SHUF_ACC_MAX)
    return m


def _control(seed: int) -> dict:
    """THE SENSORY AMPUTATION (PS.02's control, reused): the same rows with
    the odour suffix — the offer-bearing channels — deleted. If the probe
    still reads the sign it was reading a schedule artifact, and the sense
    earned nothing."""
    d = _collect(seed)
    if "refused" in d or d.get("no_food") or d["dead"] > 0.0:
        return {"control_bal_acc": 0.0, "control_caught": 0.0}
    trips = [(t["rows"], t["net"]) for t in d["leg"] if t["rows"]]
    if len(trips) < N_TEST_TRIPS + 2:
        return {"control_bal_acc": 0.0, "control_caught": 0.0}
    acc_c = _score(trips, KIN_DIM + needs.NEED_DIM)
    return {"control_bal_acc": acc_c,
            "control_n_features": float(KIN_DIM + needs.NEED_DIM),
            "control_caught": float(acc_c <= CONTROL_ACC_MAX)}


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        # An uncalibrated world refutes nothing. VOID, never FAIL (T0.22).
        return Status.VOID
    if m.get("rig_ok", 0.0) != 1.0:
        # A dead body, an unreached or uneaten offer, stray consumption or
        # a clipped payoff is a BODY/rig finding — the run could not test
        # the claim (registry).
        return Status.VOID
    world = bool(m["sign_split_ok"] == 1.0
                 and m["twin_net_min"] >= TWIN_NET_MIN)
    if not world:
        # A falsified_by branch fired: no positive offer, no negative offer,
        # or margins under the quantum. A real red, not a rig one.
        return False
    if m.get("leg_ok", 0.0) != 1.0:
        # The world half held but the probe's row set starved a class — the
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
    return run_spec(BY_ID["PS.09"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
