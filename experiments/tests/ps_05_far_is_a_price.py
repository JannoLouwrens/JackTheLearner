"""PS.05 — Far is a price: distance costs need-currency before anyone has to
learn it.

HYPOTHESIS (registry, unchanged). W0 charges for traversal, in the currency
the needs already speak: a scripted traversal policy driven to identical
resources placed at >= 4 pre-registered distances pays a need-cost (integrated
drive drain read off the existing needs.py channels — no new instrument) that
increases MONOTONICALLY in distance on every seed; the near-vs-far cost gap
exceeds the venue's outcome-metric quantum; and distance-to-resource is
LEGIBLE beforehand — a probe on the sensory vector predicts it well above
chance while the agent is still en route.

## THE TRAVERSAL FIXTURE, AND WHY IT ABSTRACTS EXACTLY ONE THING

This body has no walking controller (T2.01 is settled FAIL) and tips over in
seconds, so the fixture abstracts LOCOMOTION and nothing else — PS.01's
forager-fixture precedent, verbatim: *"the fixture abstracts LOCOMOTION and
nothing else; it pays the real drain through the real integrator."* A trip of
registered distance d is: the body exerts the scripted walking effort (the
same S_WORK regime PS.01 calibrated and PS.06 inherited, scaled by
gear_scale() per the caller's contract), the REAL NeedLayer integrates the
REAL metabolic price of that work, and a virtual position advances toward the
resource at a speed COUPLED TO THE BODY'S ACHIEVED POWER:

    v = V_WALK * clip(p_decision / P_REF, 0, V_CLIP_MULT)

P_REF is the seed's own fresh reference power (measured, PS.06's probe idiom,
never a constant). The coupling is what makes the rig gate an alive-proof
rather than arithmetic: a body too weak to exert moves slower and can fail to
REACH inside the trip budget, which VOIDs the run on the rig and records a
BODY finding, never a world verdict (registry, verbatim). Walking effort is
thrash at S_WORK — in this venue thrash counts as work (PS.06's rig note),
and what is charged for it is needs.py's own arithmetic:

    DECAY ACCOUNTING (pre-registered; needs.py constants, none moved):
      energy  de/dt = -(M / M_BASAL) * B_E,  M = M_BASAL + kappa_act * p_mech
              basal B_E = 1/1800 s^-1; at the measured walking power the
              multiplier is ~2.7x basal (kappa_act = 2*M_BASAL/p_max)
      water   dw/dt = -B_W * (1 + C_SW * max(T - 37, 0));  B_W = 1/450 s^-1
      cost of one trip = (e_start - e_end) + (w_start - w_end), trip start to
      ARRIVAL — endpoint deltas of the layer's own channels, no new meter.

## THE VENUE IS NIGHT — PS.06's finding, inherited not rediscovered

K_DRY makes basal metabolism exactly thermoneutral at T_DAY, so sustained
exertion in daylight cooks the body (PS.06 measured its frozen twin dying of
hyperthermia mid-sustain). The whole protocol runs at night (layer.t = DAY_S,
NE.01's nightfall idiom): T_env drops 10 C and the walking effort sheds its
heat. The claim is about distance pricing, not heat; night isolates it.

## THE THREE WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The cost is a clock artifact, not a price of distance** — anything that
   accrues per trip regardless of traversal (reset residue, arrival
   bookkeeping, world time) would read as a distance price. Caught by the
   registry's null, THE TELEPORT TWIN: the same trip schedule, same resets,
   same arrival bookkeeping, with traversal made instantaneous — the virtual
   position jumps to the resource and the body rests through one FIXED
   arrival interval, identical across distances, so the twin's cost is a
   live nonzero reading whose FLATNESS is the assertion rather than a zero
   the instrument never touched. A monotone twin curve is the artifact
   detected; the twin's spread across distances must sit under the quantum.

2. **The gap is smaller than the venue can resolve.** The outcome metric is
   an integrated drain; its QUANTUM (the LC.03/DP.04 resolution lesson,
   applied at registration) is the largest of the within-distance repeat
   spreads — the same trip re-run under different ctrl draws on a fresh
   body, so the spread is pure rig noise. Discrete drain events cannot enter
   a trip by construction (no eating, no drinking, and p stays far below the
   MS_P_FLOOR = 0.98 microsleep threshold on a fresh body inside one night),
   so repeat spread IS the quantum here. The near-vs-far gap must clear
   QUANTUM_MULT x that measured floor AND an absolute pre-registered floor.

3. **The probe reads the episode clock, not a sense.** Every trip ends when
   it ends, so elapsed time correlates with distance covered. Removed at the
   RIG, not by weakening the gate (PS.06's SETTLE_DEC reasoning): legibility
   trips draw their distance CONTINUOUSLY from D_LEG (so total distance is
   not recoverable from time), start at a JITTERED night time, start from
   JITTERED e0/w0 (so the energy level does not encode elapsed time), and —
   the two repairs the seed-90 pilots forced — begin with a RANDOM-DURATION
   PRE-ROLL of uncounted work/rest (U(0, PREROLL_MAX_S)) AND take their rows
   on a MEANDERING SURVEY rather than a straight-line approach. Both were
   measured in, not guessed: pilot 1's amputated probe read 0.378 against
   the 0.30 cap because a state reset to fixed f0/p0/T0 makes fatigue,
   sleep pressure and temperature perfect within-trip clocks; pilot 2, with
   the pre-roll alone, still read 0.466, and the residual is STRUCTURAL —
   on a straight-line approach with d_total ~ U(D_LEG), censoring makes
   E[remaining | elapsed] linear (E[d | d > vt] - vt), so ANY elapsed-time
   clock predicts remaining and the drain channels ARE clocks; that is the
   claim's own currency and cannot be hidden. So the coupling is broken at
   the path instead: legibility trips walk persistent random-walk headings
   (redrawn with prob P_TURN per decision) for a FIXED decision budget, so
   remaining distance is non-monotone in time and elapsed time carries
   essentially nothing about it — while the odour channel reads it
   directly. A forager's real path meanders; "en route" it remains. What
   remains distance-bearing is the declared sense: the resource emits odour
   (odour.StaticField, C = exp(-d / LAMBDA_M), SM.01's certified sensor,
   nose noise on), and the odour block sits LAST in the feature vector so
   the registry's control — THE SENSORY AMPUTATION, the same rows with the
   distance-bearing channels removed — is a clean suffix drop. The amputated
   probe must FAIL, and the shuffled pairing must collapse to chance.

## LEGIBILITY: WHAT THE ROWS ARE

Per legibility trip: d ~ U(D_LEG), random bearing, jittered start state as
above. En route, every SNAP_EVERY decisions (after a short skip), snapshot
the sensory vector [45 kinematic dims + NEED_DIM interoceptive + 12 odour
dims, odour last] and record y = remaining distance / D_LEG[1]. Held out BY
TRIP (test trips never share rows with train trips — rows inside one trip
are autocorrelated; PS.02's held-out-by-run reasoning). The probe is generic
on purpose: 200 random Fourier features + ridge, one fixed draw, applied
identically to experiment, control and shuffled null (PS.06's exact scorer).

## PRE-REGISTERED, before the first registered run (T3.06 lesson)

Registered distances D_COST = (1.5, 3.0, 4.5, 6.0) m — inside the odour
sense's legible range (C(6 m) = 0.050 >> the 1e-3 nose noise) and inside one
night at the registered pace. V_WALK = 0.4 m/s, V_CLIP_MULT = 1.25,
TRIP_BUDGET_MULT = 2.5 (the alive-proof window), REACH_EPS = 0.15 m.
N_REP = 3 trips per distance. All gates below were set with margin after a
seed-90 pilot (disjoint from registered seeds 0/1/2 — the PG.6/SM.01/PS.02
convention); pilot numbers are quoted beside each gate and in
docs/LOOP_JOURNAL.md under this spec's pre-registration.

Deaths anywhere in the protocol, a body that cannot exert at the registered
scale, a trip that cannot reach its target inside the budget, or a starved
row set are RIG findings: Status.VOID, never FAIL — a run that could not
test the claim refutes nothing (T0.22). FAIL is reserved for the three
falsified_by branches: flat/non-monotone cost, sub-quantum gap, illegible
distance.
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

# The claim is about the WORLD's pricing of distance and the senses that
# carry it.
IMPL_DEPS = ["experiments/needs.py", "experiments/odour.py", "playground.py"]

# ── the rollout, fixed before the run ───────────────────────────────────
SIM_S = 0.2                  # s per decision (NE.01's cadence)
S_WORK = 0.4                 # ctrl scale: PS.01's calibrated regime (PS.06)
PROBE_SEED = 4242            # ONE commanded sequence for P_REF, everywhere
REF_PROBE_DEC = 50           # 10 s fresh reference probe -> P_REF
V_WALK = 0.4                 # m/s, the registered walking pace
V_CLIP_MULT = 1.25           # v may not exceed this x V_WALK
TRIP_BUDGET_MULT = 2.5       # alive-proof window: budget = this x nominal
REACH_EPS = 0.15             # m, arrival radius
D_COST = (1.5, 3.0, 4.5, 6.0)   # m, the registered distances
N_REP = 3                    # trips per registered distance
TWIN_SETTLE_DEC = 20         # 4 s fixed arrival interval, identical across d
N_LEG_TRIPS = 14             # legibility trips; held out BY TRIP
N_TEST_TRIPS = 4
D_LEG = (1.0, 6.0)           # legibility distance draw (continuous)
LEG_T_JITTER_S = 250.0       # trip start time ~ DAY_S + U(0, this)
LEG_EW_JITTER = (0.70, 1.0)  # e0, w0 ~ U(band): the clock-killers (fake 3)
PREROLL_MAX_S = 20.0         # uncounted random work/rest before a
P_PREROLL_WORK = 0.6         # legibility trip: offsets f/p/T/pose (fake 3)
LEG_TRIP_DEC = 90            # fixed survey budget, 18 s — ends by CLOCK,
                             # never by arrival, so duration is uninformative
P_TURN = 0.15                # heading redraw prob per decision (fake 3)
SNAP_EVERY = 5               # snapshot cadence, decisions (1 s)
SNAP_SKIP = 3                # skip the first decisions of a trip
LYING_QUAT = (0.7071, 0.0, 0.7071, 0.0)
SPOT = (-0.8, -1.0, 0.30)    # NE.01's open flat ground (PS.06's SPOT)
KIN_DIM = 45                 # qpos[2:24] + qvel[23]: pose+velocity only
ODOUR_DIM = 12               # OdourSensor.obs: 4 ch x [L, R] + 4 derivs

# ── pre-registered gates (final pilot, seed 90, reading beside each) ────
FRESH_FRAC_MIN = 0.50        # alive-proof: P_REF >= this x p_max     (0.982)
GAP_ABS_MIN = 0.015          # near-vs-far cost gap, absolute floor  (0.0535)
QUANTUM_MULT = 2.0           # ...and >= this x the measured quantum
                             #                (q = 0.00157, gap/q = 34.0)
TWIN_COST_MIN = 0.005        # the twin's meter is alive             (0.0112)
TWIN_FLAT_MAX_Q = 1.0        # twin spread across d <= this x quantum
                             #                            (3.6e-6 vs q)
PROBE_R2_MIN = 0.35          # the headline: legible en route         (0.707)
SHUFFLED_R2_MAX = 0.05       # declared null pairing, mean of 20     (-0.140)
CONTROL_R2_MAX = 0.30        # amputated probe must fail...           (0.014)
CONTROL_MARGIN_MIN = 0.15    # ...and by a margin                     (0.693)
ROWS_TEST_MIN = 12           # instrument not starved                    (72)
Y_SPREAD_MIN = 0.30          # rows actually span distance (max-min y) (1.39)

# ── the probe: random Fourier features + ridge, one fixed draw ──────────
N_RFF = 200
RFF_SEED = 20260919
RIDGE_LAMBDA = 1.0
N_SHUFFLE = 20               # mean over permutations (PS.06's reasoning)

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
    layer.t = needs.DAY_S            # nightfall — see the venue section
    return model, data, water, layer


def _place(model, data, xyz):
    """Teleport the humanoid root, at rest (qvel zeroed: a teleport must not
    manufacture an impact — PS.06)."""
    from playground import humanoid_index
    q = humanoid_index(model)["qposadr"]
    data.qpos[q:q + 3] = xyz
    data.qpos[q + 3:q + 7] = LYING_QUAT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _reset_trip(model, data, layer, t0: float, e0: float = 1.0,
                w0: float = 1.0) -> None:
    """A fresh body at the standard spot, at night, per-trip. Resetting the
    layer's state between trips is a fixture right: the claim is about the
    world's per-trip pricing, not a life narrative, and comparable trips
    need comparable bodies (PS.06 resets per session for the same reason)."""
    _place(model, data, SPOT)
    layer.state = replace(needs.NeedState(), e=float(e0), w=float(w0))
    layer.t = float(t0)


def _step(model, data, water, layer, ctrl):
    """One decision under the NeedLayer contract (PS.06's loop)."""
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


def _ref_power(model, data, water, layer) -> float:
    """P_REF: mean achieved power under ONE fixed commanded sequence from
    the standard pose on a fresh body (PS.06's capability probe, shortened).
    Commands are scaled by gear_scale() per the caller's contract."""
    _reset_trip(model, data, layer, needs.DAY_S)
    rng = np.random.RandomState(PROBE_SEED)
    ps = []
    for _ in range(REF_PROBE_DEC):
        ctrl = rng.uniform(-S_WORK, S_WORK, model.nu) * layer.gear_scale()
        _step(model, data, water, layer, ctrl)
        ps.append(layer.last_power_w)
    return float(np.mean(ps))


def _features(model, data, layer, odour_obs: np.ndarray) -> np.ndarray:
    """Pose + velocity, then interoception, then the ODOUR BLOCK LAST so the
    amputation control is a clean suffix drop of the distance-bearing
    channels."""
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


def _trip(model, data, water, layer, d: float, bearing: float, p_ref: float,
          rng: np.random.RandomState, *, teleport: bool,
          collect: bool = False, t0: float = None, e0: float = 1.0,
          w0: float = 1.0, nose_rng=None):
    """One trip to a resource at distance d. Returns a dict with the trip's
    integrated need-cost (endpoint deltas, trip start to arrival), whether
    the target was REACHED inside the budget, and — when collecting — the
    legibility rows gathered en route.

    Live: scripted walking effort every decision; the virtual position
    advances at the power-coupled speed. Teleport twin: the position jumps
    to the target and the body rests through TWIN_SETTLE_DEC decisions —
    one fixed arrival interval, identical across distances.
    """
    _reset_trip(model, data, layer, needs.DAY_S if t0 is None else t0,
                e0=e0, w0=w0)
    if collect:
        # The clock-killing pre-roll (fake 3): a random-duration burn-in of
        # work/rest, uncounted, so f/p/T/pose start at random offsets.
        for _ in range(int(rng.uniform(0.0, PREROLL_MAX_S) / SIM_S)):
            ctrl = (rng.uniform(-S_WORK, S_WORK, model.nu)
                    * layer.gear_scale()
                    if rng.rand() < P_PREROLL_WORK else np.zeros(model.nu))
            _step(model, data, water, layer, ctrl)
            if layer.dead:
                return {"dead": 1.0}
    pos = np.array(SPOT[:2], dtype=float)
    tgt = pos + d * np.array([math.cos(bearing), math.sin(bearing)])
    e_start, w_start = layer.state.e, layer.state.w
    rows = []

    if teleport:
        for _ in range(TWIN_SETTLE_DEC):
            _step(model, data, water, layer, np.zeros(model.nu))
            if layer.dead:
                return {"dead": 1.0}
        cost = (e_start - layer.state.e) + (w_start - layer.state.w)
        return {"dead": 0.0, "cost": float(cost), "reached": 1.0,
                "n_dec": float(TWIN_SETTLE_DEC), "rows": rows}

    field = odour.StaticField([odour.Source("resource", "food",
                                            (tgt[0], tgt[1], 0.3))])
    sensor = odour.OdourSensor(field)
    n_max = LEG_TRIP_DEC if collect \
        else int(math.ceil((TRIP_BUDGET_MULT * d / V_WALK) / SIM_S))
    heading = bearing
    reached = 0.0
    for k in range(n_max):
        ctrl = rng.uniform(-S_WORK, S_WORK, model.nu) * layer.gear_scale()
        _step(model, data, water, layer, ctrl)
        if layer.dead:
            return {"dead": 1.0}
        v = V_WALK * min(max(layer.last_power_w / p_ref, 0.0), V_CLIP_MULT)
        if collect:
            # The meandering survey (fake 3): persistent random headings,
            # fixed budget — remaining distance is non-monotone in time.
            if rng.rand() < P_TURN:
                heading = float(rng.uniform(0.0, 2.0 * math.pi))
            pos = pos + v * SIM_S * np.array([math.cos(heading),
                                              math.sin(heading)])
            remaining = float(np.linalg.norm(tgt - pos))
            if k >= SNAP_SKIP and (k - SNAP_SKIP) % SNAP_EVERY == 0:
                ob = sensor.obs((pos[0], pos[1], 0.3), heading, layer.t,
                                rng=nose_rng)
                rows.append((_features(model, data, layer, ob),
                             remaining / D_LEG[1]))
            continue
        vec = tgt - pos
        dist = float(np.linalg.norm(vec))
        step_len = min(v * SIM_S, dist)
        if dist > 1e-9:
            pos = pos + vec / dist * step_len
        if float(np.linalg.norm(tgt - pos)) <= REACH_EPS:
            reached = 1.0
            break
    cost = (e_start - layer.state.e) + (w_start - layer.state.w)
    return {"dead": 0.0, "cost": float(cost), "reached": reached,
            "n_dec": float(k + 1), "rows": rows}


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


def _r2(y, yhat) -> float:
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / max(sst, 1e-12)


def _score(trips, n_cols: int, shuffle: bool = False) -> float:
    """Held-out-BY-TRIP R^2 on the first n_cols features; test set is the
    last N_TEST_TRIPS trips. shuffle=True: mean over N_SHUFFLE independent
    permutations of the train pairing (PS.06's scorer, verbatim)."""
    tr = trips[:-N_TEST_TRIPS]
    te = trips[-N_TEST_TRIPS:]
    Xtr = np.array([r[0][:n_cols] for t in tr for r in t])
    ytr = np.array([r[1] for t in tr for r in t])
    Xte = np.array([r[0][:n_cols] for t in te for r in t])
    yte = np.array([r[1] for t in te for r in t])
    if not shuffle:
        return _r2(yte, _fit_predict(Xtr, ytr, Xte))
    scores = [_r2(yte, _fit_predict(
        Xtr, np.random.RandomState(RFF_SEED + 1 + k).permutation(ytr), Xte))
        for k in range(N_SHUFFLE)]
    return float(np.mean(scores))


def _collect(seed: int) -> dict:
    """Every simulation this spec needs, once. Cached: the control re-scores
    the same rows minus the odour suffix — re-simulating would risk the arms
    differing by something other than the sense (PS.06)."""
    if seed in _CACHE:
        return _CACHE[seed]
    vals, prov = _calibration()
    if vals is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]
    model, data, water, layer = _build(seed)
    p_ref = _ref_power(model, data, water, layer)

    dead = 0.0
    live = {d: [] for d in D_COST}      # cost trips, live arm
    twin = {d: [] for d in D_COST}      # cost trips, teleport twin
    for rep in range(N_REP):
        for j, d in enumerate(D_COST):
            bearing = float(np.random.RandomState(
                seed * 7919 + rep * 101 + j).uniform(0.0, 2.0 * math.pi))
            r = _trip(model, data, water, layer, d, bearing, p_ref,
                      np.random.RandomState(seed * 977 + rep * 31 + j),
                      teleport=False)
            t = _trip(model, data, water, layer, d, bearing, p_ref,
                      np.random.RandomState(seed * 977 + rep * 31 + j),
                      teleport=True)
            dead += r.get("dead", 0.0) + t.get("dead", 0.0)
            if dead == 0.0:
                live[d].append(r)
                twin[d].append(t)

    leg = []
    for i in range(N_LEG_TRIPS):
        rng = np.random.RandomState(seed * 6421 + 17 * i + 5)
        d = float(rng.uniform(*D_LEG))
        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        t0 = needs.DAY_S + float(rng.uniform(0.0, LEG_T_JITTER_S))
        e0 = float(rng.uniform(*LEG_EW_JITTER))
        w0 = float(rng.uniform(*LEG_EW_JITTER))
        r = _trip(model, data, water, layer, d, bearing, p_ref, rng,
                  teleport=False, collect=True, t0=t0, e0=e0, w0=w0,
                  nose_rng=np.random.RandomState(seed * 331 + i))
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
    if d["dead"] > 0.0:
        return {"borrow_ok": 1.0, "rig_ok": 0.0, "seed_gates_ok": 0.0,
                "dead_events": float(d["dead"])}

    live, twin, leg = d["live"], d["twin"], d["leg"]
    reach_ok = float(all(t["reached"] == 1.0
                         for dd in D_COST for t in live[dd]))
    cost_mean = [float(np.mean([t["cost"] for t in live[dd]]))
                 for dd in D_COST]
    cost_std = [float(np.std([t["cost"] for t in live[dd]]))
                for dd in D_COST]
    twin_mean = [float(np.mean([t["cost"] for t in twin[dd]]))
                 for dd in D_COST]
    # The quantum: the largest within-distance repeat spread (see fake 2).
    quantum = max(max(cost_std), 1e-9)
    gap = cost_mean[-1] - cost_mean[0]
    monotone = float(all(cost_mean[j + 1] > cost_mean[j]
                         for j in range(len(D_COST) - 1)))
    twin_spread = max(twin_mean) - min(twin_mean)
    twin_min = min(twin_mean)

    trips = [t["rows"] for t in leg if t["rows"]]
    n_cols = KIN_DIM + needs.NEED_DIM + ODOUR_DIM
    n_test = sum(len(t) for t in trips[-N_TEST_TRIPS:])
    n_train = sum(len(t) for t in trips[:-N_TEST_TRIPS])
    ys = [r[1] for t in trips for r in t]
    y_spread = float(max(ys) - min(ys)) if ys else 0.0
    r2 = _score(trips, n_cols)
    r2_shuf = _score(trips, n_cols, shuffle=True)

    m = {
        "borrow_ok": 1.0,
        # rig: could this run test the claim at all? (VOID territory)
        "dead_events": 0.0,
        "fresh_frac": d["p_ref"] / d["p_max"],
        "reach_ok": reach_ok,
        "n_rows_train": float(n_train),
        "n_rows_test": float(n_test),
        "y_spread": y_spread,
        # the cost curve
        "cost_d1": cost_mean[0], "cost_d2": cost_mean[1],
        "cost_d3": cost_mean[2], "cost_d4": cost_mean[3],
        "quantum": quantum,
        "gap": gap,
        "monotone_ok": monotone,
        "twin_cost_min": twin_min,
        "twin_spread": twin_spread,
        # the headline
        "probe_r2": r2,
        "shuffled_r2": r2_shuf,
        "n_features": float(n_cols),
    }
    m["rig_ok"] = float(
        m["fresh_frac"] >= FRESH_FRAC_MIN
        and m["reach_ok"] == 1.0
        and m["n_rows_test"] >= ROWS_TEST_MIN
        and m["y_spread"] >= Y_SPREAD_MIN)
    m["seed_gates_ok"] = float(
        m["rig_ok"] == 1.0
        and m["monotone_ok"] == 1.0
        and m["gap"] >= max(GAP_ABS_MIN, QUANTUM_MULT * m["quantum"])
        and m["twin_cost_min"] >= TWIN_COST_MIN
        and m["twin_spread"] <= TWIN_FLAT_MAX_Q * m["quantum"]
        and m["probe_r2"] >= PROBE_R2_MIN
        and m["shuffled_r2"] <= SHUFFLED_R2_MAX)
    return m


def _control(seed: int) -> dict:
    """THE SENSORY AMPUTATION (PS.02's control, reused): the same rows with
    the odour suffix — the distance-bearing channels — deleted. If the probe
    still works it was reading the episode clock, and the sense earned
    nothing."""
    d = _collect(seed)
    if "refused" in d or d["dead"] > 0.0:
        return {"control_r2": 0.0, "control_caught": 0.0}
    trips = [t["rows"] for t in d["leg"] if t["rows"]]
    if len(trips) < N_TEST_TRIPS + 2:
        return {"control_r2": 0.0, "control_caught": 0.0}
    r2c = _score(trips, KIN_DIM + needs.NEED_DIM)
    return {"control_r2": r2c,
            "control_n_features": float(KIN_DIM + needs.NEED_DIM),
            "control_caught": float(r2c <= CONTROL_R2_MAX)}


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        # An uncalibrated world refutes nothing. VOID, never FAIL (T0.22).
        return Status.VOID
    if m.get("rig_ok", 0.0) != 1.0:
        # A dead body, an unreached target or a starved row set is a
        # BODY/rig finding — the run could not test the claim (registry).
        return Status.VOID
    return bool(
        m["seed_gates_ok"] == 1.0
        and m["probe_r2"] >= PROBE_R2_MIN
        and m["shuffled_r2"] <= SHUFFLED_R2_MAX
        and c["control_caught"] == 1.0
        and c["control_r2"] <= CONTROL_R2_MAX
        and (m["probe_r2"] - c["control_r2"]) >= CONTROL_MARGIN_MIN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PS.05"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
