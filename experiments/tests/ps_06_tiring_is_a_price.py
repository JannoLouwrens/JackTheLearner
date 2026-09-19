"""PS.06 — Tiring is a price: sustained exertion drains the body's capability
before anyone has to learn it.

HYPOTHESIS (registry, unchanged). W0's needs layer charges for work in a
currency the body must repay: a scripted sustained-exertion policy drives
fatigue f (needs.py — f rises with mechanical power p_mean, falls with rest,
taus 60 s/60 s) high enough on every seed that the weakness gate
gear_scale = 0.5 + 0.5*(1-f)*min(e,i) measurably cuts what the SAME ctrl
commands achieve; rest REPAYS it at the pre-registered TAU_F_FALL timescale
(distinguishing tiredness from damage); and the imminent droop is LEGIBLE
beforehand — a probe on the sensory vector predicts the next capability
deficit well above chance.

## THE VENUE IS NIGHT, AND THE PILOT FOUND WHY THAT IS NOT A CONVENIENCE

The first dry protocol ran in daylight and the FATIGUE-FROZEN twin DIED OF
HYPERTHERMIA mid-sustain while the live arm survived. That is a real finding
about this world, recorded here because no ledger row will ever carry it:
K_DRY makes basal metabolism exactly thermoneutral at T_DAY, so ALL exertion
heat is transient headroom, and the fatigue gate — by throttling applied
torque as f accrues — is the only thing that stops a working body from
cooking itself. Freeze f and the ungated thrash crosses T_HOT_DEATH inside
90 s. Fatigue is load-bearing thermal protection in W0, not decoration.

For THIS spec, though, a null that dies cannot serve as a flatness baseline,
so the whole protocol runs at night (layer.t starts at DAY_S — NE.01's
nightfall idiom): T_env drops 10 C, the extra conductive headroom sheds the
exertion heat, and both arms survive the identical schedule. The registered
claim is about fatigue pricing capability, not about heat; the night venue
isolates exactly that.

## THE INSTRUMENT: POSE-RESET PROBES, BECAUSE THRASH POWER IS NOISY

"What the same ctrl achieves" is measured by a CAPABILITY PROBE: teleport the
body to a standard lying pose at a fixed spot (qvel zeroed — a teleport must
not manufacture an impact), then replay an IDENTICAL commanded ctrl sequence
(one fixed rng draw, PROBE_SEED) and read mean |tau*omega| power. The pilot
measured why nothing weaker works: free-running window-mean thrash power
carries +-10-15% posture-regime noise (the body settles into different
tangles), which swamps the gear signal across lives — three successive
probe designs on free-running windows scored R^2 <= 0.1 while the pose-reset
probe's repeat deviation on the frozen twin is ~4%. The probe holds posture
fixed so physiology is the only free variable.

The commanded pattern is multiplied by layer.gear_scale() before application
— that is the CALLER'S CONTRACT with the needs layer (NE.01's idiom), i.e.
exactly how weakness is enforced in this venue. Identical commands, spent
body, less achieved output: that is the claim, measured.

## THE THREE WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The droop is energy drain or damage wearing fatigue's name.** Caught by
   the registry's null, the FATIGUE-FROZEN TWIN: an identical world, seed and
   commanded schedule with f clamped to 0 after every decision. gear_scale
   couples min(e,i) too, so the twin's droop is exactly the non-fatigue share;
   the claim gates on the DIFFERENCE (live gap minus frozen gap) and on the
   twin staying near-flat. And rest must repay on TAU_F_FALL's clock — the
   measured f-decay time constant must land in TAU_FIT_BAND, where damage
   (RHO_HEAL, 15 min) and energy (no rest recovery at all) cannot follow.

2. **The gap is smaller than the venue can resolve.** The outcome metric is a
   probe-power ratio; its quantum is the frozen twin's fresh repeat deviation
   — the same probe run twice on a body whose physiology does not move, so
   the deviation is pure instrument noise (the live arm's repeat pair is NOT
   usable for this: its second probe reads lower because PROBING ITSELF
   TIRES, which is reported as live_fresh_dev and is evidence, not noise).
   The fatigue gap must clear QUANTUM_MULT x that measured floor AND an
   absolute pre-registered floor, per seed.

3. **The probe reads the clock or the visible droop, not a sense.** Caught by
   the registry's declared control, THE SENSORY AMPUTATION: the same rows
   with the interoceptive block (needs.obs(), the NEED_DIM suffix) deleted.
   The kinematic block (pose + velocity, what an outside observer sees) stays
   — deliberately, because a probe that only had interoception to read could
   not fail this way. Legibility rows are collected at varying f under
   SEEDED RANDOM work/rest segments, so f is history-dependent and no episode
   clock reconstructs it; the shuffled pairing must collapse to chance.

## LEGIBILITY: WHAT THE ROWS ARE

Per session (a fresh night, fresh body): a session-fresh reference probe,
then alternating seeded work/rest segments; after each segment, snapshot the
sensory vector (45 kinematic dims + NEED_DIM interoceptive dims, suffix
last so the control is a clean suffix drop), then run a capability probe.
Target y = probe power / session-fresh power: the capability the body is
about to demonstrate, normalised within-session so nothing cross-session
leaks in. Held out BY SESSION (train sessions never share rows with test
sessions) — rows inside one session are autocorrelated and a row split
would report memorisation as generalisation (PS.02's reasoning, inherited).
The probe is generic on purpose: 200 random Fourier features + ridge, one
fixed draw, applied identically to experiment, control and shuffled null.

## PRE-REGISTERED, before the first registered run (T3.06 lesson)

Exertion scale S_WORK = 0.4 — the regime PS.01's p_max describes (its
"full strength" is CTRL_SCALE 0.4; the pilot measured scale 1.0 thrash at
2.0x p_max, which is outside the calibrated regime and cooks the body even
at night). Sustain 90 s (~1.5 tau: f reaches ~0.5 with gear feedback), rest
120 s (2 tau: ~86% repaid). All gates below were set with margin after a
seed-90 pilot (disjoint from registered seeds 0/1/2, the PG.6/SM.01/PS.02
convention); pilot numbers are quoted beside each gate and in
docs/LOOP_JOURNAL.md under this spec's pre-registration.

Deaths anywhere in the protocol, a body that cannot exert at the registered
scale, a twin whose freeze did not hold, or a starved row set are RIG
findings: Status.VOID, never FAIL — a run that could not test the claim
refutes nothing (T0.22). FAIL is reserved for the four falsified_by branches:
decorative tiredness, sub-quantum gap, no rest repayment on the registered
timescale, illegible droop.
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

# The claim is about the WORLD's needs layer and the body that pays it.
IMPL_DEPS = ["experiments/needs.py", "playground.py"]

# ── the rollout, fixed before the run ───────────────────────────────────
SIM_S = 0.2                  # s per decision (NE.01's cadence)
S_WORK = 0.4                 # ctrl scale everywhere: PS.01's calibrated regime
PROBE_DEC = 75               # 15 s capability probe (protocol)
SESS_PROBE_DEC = 30          # 6 s capability probe (legibility sessions)
PROBE_SEED = 4242            # ONE commanded sequence, identical everywhere
SUSTAIN_DEC = 450            # 90 s sustained exertion (~1.5 * TAU_F_RISE)
REST_DEC = 600               # 120 s rest (2 * TAU_F_FALL)
N_SESSIONS = 5               # legibility sessions; held out BY SESSION
N_TEST_SESSIONS = 2
SESSION_S = 350.0            # s per session, inside one 400 s night
SEG_S = (6.0, 18.0)          # work/rest segment duration draw
P_WORK = 0.55                # P(segment is work)
SETTLE_DEC = 10              # 2 s of zero ctrl before every snapshot: the
                             # row is taken at momentary stillness, so the
                             # kinematic block cannot carry the CURRENT droop
                             # (velocity ~ applied torque ~ gear) and both
                             # arms face the actual question — is the FUTURE
                             # deficit knowable before it is visible. The
                             # seed-90 pilot without this read control_r2
                             # 0.40: the leak the registry's control clause
                             # names, removed at the rig, not by weakening
                             # the gate. f decays x0.967 in the settle —
                             # negligible against TAU_F_FALL.
LYING_QUAT = (0.7071, 0.0, 0.7071, 0.0)
SPOT = (-0.8, -1.0, 0.30)    # NE.01's open flat ground, clear of props
KIN_DIM = 45                 # qpos[2:24] 22 + qvel 23 — pose+velocity only:
                             # no actuator forces, which would hand the
                             # control the applied torque for free

# ── pre-registered gates (final pilot, seed 90, reading beside each) ────
FRESH_FRAC_MIN = 0.50        # alive-proof: fresh probe >= this x p_max (0.837)
F_SPENT_MIN = 0.35           # tiredness accrued, not decorative        (0.512)
GAP_ABS_MIN = 0.10           # fatigue-attributable capability gap      (0.276)
QUANTUM_MULT = 2.0           # ...and >= this x the measured quantum (q=0.0343)
FROZEN_SPENT_MIN = 0.75      # the twin must stay near-flat             (0.851)
REC_GAIN_MIN = 0.08          # rest repays capability, twin-differenced (0.293)
TAU_FIT_BAND = (45.0, 75.0)  # measured f-decay tau vs registered 60     (59.9)
PROBE_R2_MIN = 0.35          # the headline: legible beforehand         (0.674)
SHUFFLED_R2_MAX = 0.05       # declared null pairing, mean of 20       (-0.176)
CONTROL_R2_MAX = 0.30        # amputated probe must fail...             (0.101)
CONTROL_MARGIN_MIN = 0.15    # ...and by a margin, not by rounding      (0.573)
ROWS_TEST_MIN = 20           # instrument not starved                      (35)
F_ROW_MAX_MIN = 0.35         # sessions actually explored fatigue       (0.475)

# ── the probe: random Fourier features + ridge, one fixed draw ──────────
N_RFF = 200
RFF_SEED = 20260919
RIDGE_LAMBDA = 1.0
N_SHUFFLE = 20               # the shuffled null is a MEAN over permutations:
                             # at ~60 train rows a single permutation is a
                             # lottery (the seed-90 pilot drew +0.25 on one
                             # draw and -0.1 on others), and a null that can
                             # pass or fail on its own draw gates nothing

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
    layer.t = needs.DAY_S            # nightfall — see the venue section above
    return model, data, water, layer


def _place(model, data, xyz):
    """Teleport the humanoid root, at rest. qvel zeroed: a teleport must not
    manufacture an impact (the j channel reads arrival speed at onset)."""
    from playground import humanoid_index
    q = humanoid_index(model)["qposadr"]
    data.qpos[q:q + 3] = xyz
    data.qpos[q + 3:q + 7] = LYING_QUAT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _step(model, data, water, layer, ctrl, freeze: bool):
    """One decision under the NeedLayer contract; the frozen twin's f is
    clamped AFTER decide, so every gear_scale() read sees f = 0."""
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
    if freeze:
        layer.state = replace(layer.state, f=0.0)


def _capability_probe(model, data, water, layer, freeze: bool,
                      n_dec: int) -> float:
    """Mean achieved mechanical power under ONE fixed commanded sequence from
    a standard pose. The commands are scaled by gear_scale() per the caller's
    contract — that is the mechanism under test, not a confound."""
    _place(model, data, SPOT)
    rng = np.random.RandomState(PROBE_SEED)
    ps = []
    for _ in range(n_dec):
        ctrl = rng.uniform(-S_WORK, S_WORK, model.nu) * layer.gear_scale()
        _step(model, data, water, layer, ctrl, freeze)
        ps.append(layer.last_power_w)
    return float(np.mean(ps))


def _protocol(seed: int, freeze: bool) -> dict:
    """fresh x2 -> sustain -> spent -> rest (f tracked) -> recovered."""
    model, data, water, layer = _build(seed)
    rng = np.random.RandomState(seed * 977 + 3)
    p_fresh1 = _capability_probe(model, data, water, layer, freeze, PROBE_DEC)
    p_fresh2 = _capability_probe(model, data, water, layer, freeze, PROBE_DEC)
    t_max = layer.state.T
    twin_f_max = layer.state.f if freeze else 0.0
    for _ in range(SUSTAIN_DEC):
        ctrl = rng.uniform(-S_WORK, S_WORK, model.nu) * layer.gear_scale()
        _step(model, data, water, layer, ctrl, freeze)
        t_max = max(t_max, layer.state.T)
        if freeze:
            twin_f_max = max(twin_f_max, layer.state.f)
        if layer.dead:
            return {"dead": 1.0, "cause": layer.death_record["cause"]}
    f_spent = layer.state.f
    p_spent = _capability_probe(model, data, water, layer, freeze, PROBE_DEC)
    t_max = max(t_max, layer.state.T)
    f_track = []
    for _ in range(REST_DEC):
        _step(model, data, water, layer, np.zeros(model.nu), freeze)
        f_track.append(layer.state.f)
        if layer.dead:
            return {"dead": 1.0, "cause": layer.death_record["cause"]}
    f_rest = layer.state.f
    p_rec = _capability_probe(model, data, water, layer, freeze, PROBE_DEC)
    out = {"dead": 0.0, "p_fresh1": p_fresh1, "p_fresh2": p_fresh2,
           "p_spent": p_spent, "p_rec": p_rec, "f_spent": f_spent,
           "f_rest": f_rest, "t_max": t_max, "e_end": layer.state.e,
           "i_end": layer.state.i, "twin_f_max": twin_f_max}
    if not freeze:
        # f-decay time constant during rest, log-linear fit — the clock that
        # separates tiredness (TAU_F_FALL) from damage (RHO_HEAL) and energy.
        ft = np.asarray(f_track)
        t = np.arange(1, len(ft) + 1) * SIM_S
        mask = ft > 0.02
        out["tau_fit_s"] = (float(-1.0 / np.polyfit(t[mask],
                                                    np.log(ft[mask]), 1)[0])
                            if mask.sum() > 20 else float("nan"))
    del model, data, water
    return out


def _features(model, data, layer) -> np.ndarray:
    """Pose + velocity (what an observer sees), then the interoceptive block
    LAST so the amputation control is a clean suffix drop."""
    from playground import HUMANOID_NQ, humanoid_index
    ix = humanoid_index(model)
    q, d = ix["qposadr"], ix["dofadr"]
    kin = np.concatenate([data.qpos[q + 2:q + HUMANOID_NQ],
                          data.qvel[d:d + 23]])
    v = np.concatenate([kin, layer.obs()])
    if v.shape[0] != KIN_DIM + needs.NEED_DIM:
        raise RuntimeError(f"feature vector is {v.shape[0]}, "
                           f"not {KIN_DIM + needs.NEED_DIM}")
    return v


def _session(seed: int, salt: int):
    """One legibility session: fresh reference, then (segment -> snapshot ->
    probe) cycles until the night budget is spent. Returns rows or None on a
    death (a rig event, surfaced by the caller as VOID)."""
    model, data, water, layer = _build(seed)
    rng = np.random.RandomState(seed * 7919 + 101 + salt)
    fresh = _capability_probe(model, data, water, layer, False, SESS_PROBE_DEC)
    if fresh <= 0.0:
        return None
    rows = []
    budget_end = needs.DAY_S + SESSION_S
    while layer.t < budget_end - (SESS_PROBE_DEC + 5) * SIM_S:
        work = rng.rand() < P_WORK
        for _ in range(int(rng.uniform(*SEG_S) / SIM_S)):
            ctrl = (rng.uniform(-S_WORK, S_WORK, model.nu)
                    * layer.gear_scale() if work else np.zeros(model.nu))
            _step(model, data, water, layer, ctrl, False)
            if layer.dead:
                return None
        for _ in range(SETTLE_DEC):
            _step(model, data, water, layer, np.zeros(model.nu), False)
            if layer.dead:
                return None
        x = _features(model, data, layer)
        y = _capability_probe(model, data, water, layer, False,
                              SESS_PROBE_DEC) / fresh
        if layer.dead:
            return None
        rows.append((x, float(y), float(layer.state.f)))
    del model, data, water
    return rows


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


def _score(sessions, n_cols: int, shuffle: bool = False) -> float:
    """Held-out-BY-SESSION R^2 for the first n_cols features. The test set is
    always the last N_TEST_SESSIONS sessions. With shuffle=True, the MEAN
    over N_SHUFFLE independent permutations of the train pairing."""
    tr = sessions[:-N_TEST_SESSIONS]
    te = sessions[-N_TEST_SESSIONS:]
    Xtr = np.array([r[0][:n_cols] for s in tr for r in s])
    ytr = np.array([r[1] for s in tr for r in s])
    Xte = np.array([r[0][:n_cols] for s in te for r in s])
    yte = np.array([r[1] for s in te for r in s])
    if not shuffle:
        return _r2(yte, _fit_predict(Xtr, ytr, Xte))
    scores = [_r2(yte, _fit_predict(
        Xtr, np.random.RandomState(RFF_SEED + 1 + k).permutation(ytr), Xte))
        for k in range(N_SHUFFLE)]
    return float(np.mean(scores))


def _collect(seed: int) -> dict:
    """Every simulation this spec needs, once. Cached: the control re-scores
    the same rows minus the interoceptive suffix — re-simulating would risk
    the arms differing by something other than the sense."""
    if seed in _CACHE:
        return _CACHE[seed]
    vals, prov = _calibration()
    if vals is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]
    live = _protocol(seed, freeze=False)
    frozen = _protocol(seed, freeze=True)
    sessions = [_session(seed, s) for s in range(N_SESSIONS)]
    _CACHE[seed] = {"live": live, "frozen": frozen, "sessions": sessions,
                    "p_max": vals["mean_power_w_full_strength"]}
    return _CACHE[seed]


def _experiment(seed: int) -> dict:
    d = _collect(seed)
    if "refused" in d:
        return {"borrow_ok": 0.0, "rig_ok": 0.0, "seed_gates_ok": 0.0}
    live, frozen, sessions = d["live"], d["frozen"], d["sessions"]

    dead = live.get("dead", 1.0) + frozen.get("dead", 1.0) \
        + float(sum(s is None for s in sessions))
    if dead > 0.0:
        return {"borrow_ok": 1.0, "rig_ok": 0.0, "seed_gates_ok": 0.0,
                "dead_events": float(dead)}

    fresh_live = live["p_fresh1"]
    fresh_frozen = frozen["p_fresh1"]
    # The quantum: the frozen twin's repeat deviation — same probe twice on a
    # body whose physiology does not move (see fake-path 2 above).
    quantum = abs(frozen["p_fresh1"] - frozen["p_fresh2"]) \
        / max(fresh_frozen, 1e-9)
    live_fresh_dev = abs(live["p_fresh1"] - live["p_fresh2"]) \
        / max(fresh_live, 1e-9)

    live_spent = live["p_spent"] / fresh_live
    frozen_spent = frozen["p_spent"] / fresh_frozen
    fatigue_gap = frozen_spent - live_spent
    live_rec_gain = live["p_rec"] / fresh_live - live_spent
    frozen_rec_gain = frozen["p_rec"] / fresh_frozen - frozen_spent
    rec_gain_diff = live_rec_gain - frozen_rec_gain

    good = [s for s in sessions if s]
    n_cols = KIN_DIM + needs.NEED_DIM
    n_test = sum(len(s) for s in good[-N_TEST_SESSIONS:])
    n_train = sum(len(s) for s in good[:-N_TEST_SESSIONS])
    f_row_max = max(r[2] for s in good for r in s)
    r2 = _score(good, n_cols)
    r2_shuf = _score(good, n_cols, shuffle=True)

    m = {
        "borrow_ok": 1.0,
        # rig: could this run test the claim at all? (VOID territory)
        "dead_events": 0.0,
        "fresh_frac": fresh_live / d["p_max"],
        "twin_f_max": frozen["twin_f_max"],
        "n_rows_train": float(n_train),
        "n_rows_test": float(n_test),
        "f_row_max": f_row_max,
        # the protocol's numbers
        "quantum": quantum,
        "live_fresh_dev": live_fresh_dev,
        "f_spent": live["f_spent"],
        "f_after_rest": live["f_rest"],
        "live_spent_ratio": live_spent,
        "frozen_spent_ratio": frozen_spent,
        "fatigue_gap": fatigue_gap,
        "live_rec_gain": live_rec_gain,
        "frozen_rec_gain": frozen_rec_gain,
        "rec_gain_diff": rec_gain_diff,
        "tau_fit_s": live["tau_fit_s"],
        "t_max_protocol": max(live["t_max"], frozen["t_max"]),
        "e_end_protocol": live["e_end"],
        "i_end_protocol": live["i_end"],
        # the headline
        "probe_r2": r2,
        "shuffled_r2": r2_shuf,
        "n_features": float(n_cols),
    }
    m["rig_ok"] = float(
        m["fresh_frac"] >= FRESH_FRAC_MIN
        and m["twin_f_max"] == 0.0
        and m["n_rows_test"] >= ROWS_TEST_MIN
        and m["f_row_max"] >= F_ROW_MAX_MIN)
    m["seed_gates_ok"] = float(
        m["rig_ok"] == 1.0
        and m["f_spent"] >= F_SPENT_MIN
        and m["fatigue_gap"] >= max(GAP_ABS_MIN, QUANTUM_MULT * m["quantum"])
        and m["frozen_spent_ratio"] >= FROZEN_SPENT_MIN
        and m["rec_gain_diff"] >= REC_GAIN_MIN
        and TAU_FIT_BAND[0] <= m["tau_fit_s"] <= TAU_FIT_BAND[1]
        and m["probe_r2"] >= PROBE_R2_MIN
        and m["shuffled_r2"] <= SHUFFLED_R2_MAX)
    return m


def _control(seed: int) -> dict:
    """THE SENSORY AMPUTATION (PS.02's control, reused): the same rows with
    the interoceptive suffix deleted. If the probe still works it was reading
    the clock or the visible droop, and the sense earned nothing."""
    d = _collect(seed)
    if "refused" in d:
        return {"control_r2": 0.0, "control_caught": 0.0}
    good = [s for s in d["sessions"] if s]
    if len(good) < N_SESSIONS:
        return {"control_r2": 0.0, "control_caught": 0.0}
    r2c = _score(good, KIN_DIM)
    return {"control_r2": r2c,
            "control_n_features": float(KIN_DIM),
            "control_caught": float(r2c <= CONTROL_R2_MAX)}


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        # An uncalibrated world refutes nothing. VOID, never FAIL (T0.22).
        return Status.VOID
    if m.get("rig_ok", 0.0) != 1.0:
        # A dead body, a broken freeze or a starved row set is a BODY/rig
        # finding — the run could not test the claim (registry, verbatim).
        return Status.VOID
    return bool(
        m["seed_gates_ok"] == 1.0
        and m["probe_r2"] >= PROBE_R2_MIN
        and m["shuffled_r2"] <= SHUFFLED_R2_MAX
        and c["control_caught"] == 1.0
        and c["control_r2"] <= CONTROL_R2_MAX
        and (m["probe_r2"] - c["control_r2"]) >= CONTROL_MARGIN_MIN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PS.06"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
