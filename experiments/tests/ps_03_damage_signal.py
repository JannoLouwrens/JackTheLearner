"""PS.03 — Damage is a signal, not just an ending.

HYPOTHESIS (registry, unchanged). Harm produces a GRADED, sensed damage signal
that precedes death, and a single exposure is enough to shift behaviour away
from its cause.

DESIGN, pre-registered in docs/LOOP_JOURNAL.md (2026-08-13 ~00:4x UTC) before
this file existed, measured constants and all. The rig is a wrapper over W0
(`thermal.py` precedent — w0.py and playground.py untouched): two PERCEPT-FREE
sites H and T (the `fire_xy` precedent: no geom, so no vision ray, no sound and
no touch can find them), mirrored about the arena's x-axis in a clear region
DERIVED from the live world's legal-spawn set (the PG.8 rule: when a fixture
needs a clean configuration, derive it from the model and gate on the
derivation having worked). Entering H costs

    delta_i = min(DI_CAP, ETA * v_entry)        one prick per entry,
                                                refractory until exit

applied to `drives.DriveLayer.state` AFTER the decision, so the next
observation's needs channel carries it as a negative `idot` — the damage is
SENSED through the same 6-float interoceptive vector everything else uses, not
through a wrapper flag. T is byte-identical and harmless.

WHY THE SITES ARE PERCEPT-FREE, and what that buys. TA.01's twin design exists
to catch "he learned to avoid novelty, not injury". Here novelty has nothing to
key on — the sites emit nothing — so a percept-keyed novelty control would be
VACUOUS (reasoned in the pre-registration, before implementation). What remains
falsifiable is PLACE specificity, and the twin carries it: avoidance that
transfers to T is fear of the neighbourhood, not of the harm.

THE PROTOCOL is forced-approach trials (free roaming was REJECTED BY
MEASUREMENT in the pre-registration pilot: a random walk's two-zone occupancy
at these ranges is 0.000). Each trial: restore the world to one snapshot, place
the rover D0 from the target site on its outward side, aim at it with drawn
heading noise, drive. PAIRED DRAWS: the post phase re-uses the baseline phase's
(bearing, noise) draws verbatim, and the world snapshot makes trials
deterministic, so the pre-to-post delta is attributable to the learner's state
and not to fresh binomial noise. (This is a matched-pairs design, not the
fixed-shuffle-draw defect in LESSONS — there is no permutation null here.)

    baseline   learner OFF, empty map: P(enter H), P(enter T)   n=N_TRIALS each
    exposure   learner ON, one straight approach into H — exactly ONE felt prick
    post       learner FROZEN at one exposure: same draws, P(enter H), P(enter T)

    one_exposure_avoidance_delta = P(enter H | baseline) - P(enter H | post)

THE LEARNER is the component under test, and it is deliberately minimal:
place-keyed one-shot aversion. On FELT damage — read from the OBSERVATION's
`idot`, never from wrapper internals — it deposits a Gaussian kernel
(SIGMA_PLACE) at the current xy scaled by the felt magnitude; a reflex turns
the drive away when the heading's LOOKAHEAD point is averse. Position comes
from the harness (the scripted approach policy is world-authored either way);
the sensed quantity the claim is about is the damage signal.

THE NULL (registry): the harmless variant, ETA = 0. Same world, same draws,
same exposure walk — nothing is felt, the map stays empty, and because the
paired trials are deterministic the null's post phase must reproduce baseline
EXACTLY. `null_delta` is 0.0 iff no aversion formed; it moves if damage is
applied despite ETA=0, or if the learner keys on anything other than felt
damage (touch, novelty, entry itself). Discriminating, not saturated.

THE CONTROL THAT MUST FAIL (registry's twin clause, made executable): the
GLOBAL-FEAR learner — kernel width SIGMA_GLOBAL >= the site separation — must
transfer its avoidance to T. It is the demonstration that the twin gate CAN
detect transfer of the size the experiment produces (LESSONS: the control's
power is a parameter of the design, and it must be shown, not assumed).

GATE ANCHORS, chosen against the T2.08 lesson (a floor anchored to the
experiment pilot's bulk is a second bet on the pilot's draw):
  * the headline gates are RELATIVE (post <= half of baseline; delta large
    enough that a quarter of trials changed outcome) — invariant to a
    family-level shift in absolute entry rates;
  * gradation gates are ratios (span >= 3x) over quantities the rig is
    REQUIRED (rig_ok, else VOID) to make reachable — the BA.01-v3 rule;
  * felt-ness reuses TA.01's exact exogenous ruler: the needs-channel move
    must be >= 3x what the ordinary passage of the same time costs.
VOID, never FAIL, is returned for rig degeneracy: no clean site pair in this
world, a baseline approach that cannot reach the site, an exposure that missed,
or a speed spread too narrow to show gradation on. FAIL is reserved for the
hypothesis: damage unsensed, ungraded, instantly lethal, or avoidance that
needs more than one exposure / transfers to the twin.

CUMULATIVE LETHALITY is arithmetic, TA.01's illness-half precedent: DI_CAP
survivable by construction (one worst prick leaves i >= 0.75 against
DEATH_FLOOR 0.0), ceil(1/DI_CAP) = 4 capped pricks without healing reach the
floor. Reported (`pricks_to_death`), and the single-exposure floor is gated
(`i_min_single` >= SURVIVAL_MARGIN) — "the only way to learn about a danger is
to die of it" is exactly what the falsified_by clause forbids.

PILOT: seed-90 world family, disjoint from the registered seeds 0/1/2 (PG.6 /
TA.01 / PS.02 precedent). Pilot numbers, 2026-08-13: baseline entry 10/10 on
both sites; post-exposure entry 0/10 on H and 10/10 (identical trajectories)
on T; one_exposure_avoidance_delta 1.0; null delta 0.0 with 0 felt events and
0 map points; global-fear control transfer 1.0; v_entry 0.416-2.331 (span
5.60x), delta_i 0.033-0.187 monotone, sensed_dev 0.0, felt_ratio_min 9.9e3,
i_min_single 0.743. Two rig facts the pilot forced, both now in the design:
the arm servos' snap at a=0 registered as phantom ~2.2 m/s falls (arms are
held at their spawn target, derived from the live ctrlrange), and at
DRIVE_SCALE 0.8 the walk's own hops crossed PS.01's j0 so the ETA=0 null FELT
THE WALK (null_delta read 1.0) — the approach was slowed until walking is not
falling, which is what makes the null's zero a measurement rather than a
mercy. Gates were set with margin BEFORE the registered run and must not move
after it (law 4).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

from .. import drives                                      # noqa: E402
from ..protocol import Ledger, Status, borrow_metrics, run_spec  # noqa: E402
from ..registry import BY_ID                               # noqa: E402
from ..w0 import (DEATH_FLOOR, SIM_S_PER_DECISION, SPAWN_GRID,   # noqa: E402
                  SPAWN_MARGIN, W0, POOL_XY)

# The claim is about the WORLD's damage law and the needs channel that carries
# it, so the certificate goes stale when either moves.
IMPL_DEPS = ["experiments/w0.py", "experiments/drives.py", "playground.py"]

# ── the hazard law (pre-registered 2026-08-13, before implementation) ───
ETA = 0.08                # integrity per (m/s) of entry speed
DI_CAP = 0.25             # worst single prick; 4 capped pricks reach the floor
R_ENTRY = 0.6             # m — inside this radius of a site is "in" it

# ── the rig (knobs named in the pre-registration's implementer task) ────
D0 = 1.0                  # m — trial start distance from the target site
HORIZON = 100             # decisions per trial (20 simulated seconds)
SETTLE = 5                # decisions of stillness before every approach
HEAD_SIGMA = 0.25         # rad — per-trial heading noise
APPROACH_SPREAD = 0.6     # rad — bearing sector half-width, outward side
DRIVE_SCALE = 0.05        # Part B approach drive: fast enough to prick well
                          # above FELT_IDOT, gentle enough that the walk's own
                          # hops stay under PS.01's j0 (an approach must not
                          # also be a fall, or the ETA=0 null feels the WALK)
# Part A's entry-speed ladder: (drive_scale, pulse decisions or None, start
# distance). The rover is a top-heavy pole that TOPPLES under any sustained
# forward push and enters at gravity's speed (~1.2 m/s) however gentle the
# drive — measured in the pilot — so the slow half starts just outside the
# boundary and lets a nudge topple it across early in the fall (0.4-0.7 m/s),
# while the fast half drives hard from the standard distance (up to ~2.3).
GRADE_ARMS = ((0.01, 2, 0.68), (0.01, 2, 0.75), (0.01, 2, 0.80),
              (0.01, 2, 0.88), (0.05, None, 1.0), (0.13, None, 1.0),
              (0.35, None, 1.0), (1.00, None, 1.0))
N_TRIALS = 10             # per cell (site x phase)
SEP_MIN = 3.0             # m — minimum H-T separation
CLEAR_R = 1.7             # m — radius around each site that must be all-legal
ABORT_DIST = D0 + 0.8     # a rover this far from the site has left the trial

# ── the learner (apparatus, fixed before the registered run) ────────────
SIGMA_PLACE = 0.6         # m — place-keyed kernel; << site separation
SIGMA_GLOBAL = 4.0        # m — the global-fear control; >= site separation
A_THRESH = 0.008          # aversion at the lookahead point that trips the reflex
LOOKAHEAD = 0.6           # m ahead of the rover along its heading
FELT_IDOT = -0.02         # obs idot at/below this is felt damage

# ── pre-registered gates (set with margin after the seed-90 pilot) ──────
RIG_BASE_MIN = 0.5        # baseline entry below this = broken approach -> VOID
RIG_MIN_ENCOUNTERS = 6    # Part A must land at least this many pricks -> VOID
RIG_V_SPAN_MIN = 3.0      # entry-speed spread the rig must produce -> VOID
DELTA_MIN = 0.30          # headline floor: >= 3 of 10 trials changed outcome
POST_OVER_BASE_MAX = 0.5  # post entry rate at most half of baseline
TWIN_BAND = 0.20          # |baseline_T - post_T| must stay inside (2 trials)
NULL_BAND = 0.10          # the ETA=0 null's |delta| ceiling (1 trial)
CTRL_TRANSFER_MIN = 0.30  # global fear must shift T by this much (must fail
                          # the twin criterion decisively, not by rounding)
DI_SPAN_MIN = 3.0         # graded: delivered delta_i spans >= 3x
FELT_RATIO_GATE = 3.0     # TA.01's ruler: d(h) move >= 3x the clock's
SENSED_DEV_MAX = 1e-3     # obs idot vs the drive state's actual move
SURVIVAL_MARGIN = 0.5     # i after any single exposure stays above this

_CACHE: dict = {}


# ── the hazard wrapper ──────────────────────────────────────────────────
class HazardWorld:
    """A `W0` with two percept-free sites over it. Owns no physics.

    `eta=0.0` is the registry's declared null: the same sites, the same entry
    detection, the same refractory logic — the damage term is multiplied by
    zero and nothing else differs.
    """

    def __init__(self, w0: W0, h_xy, t_xy, eta: float = ETA):
        self.w0 = w0
        self.sites = {"H": np.asarray(h_xy, float), "T": np.asarray(t_xy, float)}
        self.eta = float(eta)
        self.inside = {"H": False, "T": False}
        self.last_prick = None            # (v_entry, delta_i) or None
        # The action that HOLDS the arm slides at their spawn position (ctrl
        # target 0), derived from the live ctrlrange (T0.14: never a copied
        # constant). a=0 would command the mid-range target and the position
        # servos' snap bounces the torso hard enough to register as a FALL
        # (measured in the pilot: phantom ~2.2 m/s onsets with no drive).
        lo = np.asarray(w0.model.actuator_ctrlrange[:4, 0], float)
        hi = np.asarray(w0.model.actuator_ctrlrange[:4, 1], float)
        self.arm_hold = 2.0 * (0.0 - lo) / (hi - lo) - 1.0

    def xy(self) -> np.ndarray:
        return np.asarray(self.w0.data.xpos[self.w0.rover_bid][:2], float)

    def _speed_xy(self) -> float:
        d = self.w0.ix["root_dofadr"]
        return float(np.hypot(self.w0.data.qvel[d], self.w0.data.qvel[d + 1]))

    def reset_sites(self) -> None:
        self.inside = {"H": False, "T": False}
        self.last_prick = None

    def decide(self, action) -> None:
        self.w0.decide(action)
        self.last_prick = None
        xy = self.xy()
        for name, s in self.sites.items():
            ins = bool(np.hypot(xy[0] - s[0], xy[1] - s[1]) <= R_ENTRY)
            if ins and not self.inside[name] and name == "H":
                v = self._speed_xy()
                di = min(DI_CAP, self.eta * v)
                st = self.w0.drives.state
                self.w0.drives.state = replace(
                    st, i=float(np.clip(st.i - di, 0.0, 1.0)))
                self.last_prick = (v, di)
            self.inside[name] = ins


# ── the learner ─────────────────────────────────────────────────────────
class PlaceAversion:
    """One-shot place-keyed aversion. Reads FELT damage from the obs only."""

    def __init__(self, sigma: float):
        self.sigma = float(sigma)
        self.pts: list = []               # (xy, magnitude)

    def update(self, xy, magnitude: float) -> None:
        self.pts.append((np.asarray(xy, float).copy(), float(magnitude)))

    def value(self, xy) -> float:
        if not self.pts:
            return 0.0
        xy = np.asarray(xy, float)
        s2 = 2.0 * self.sigma * self.sigma
        return float(sum(m * math.exp(-float(((p - xy) ** 2).sum()) / s2)
                         for p, m in self.pts))


# ── site derivation: a clean mirrored pair from the LIVE world ──────────
def _site_pair(w0: W0):
    """(h_xy, t_xy) mirrored about y=0, or None if this world offers none.

    Clean = every legal-spawn grid cell within CLEAR_R of each site exists in
    the legal set and is outside the pool's expanded bounding box. Derived, not
    declared (PG.8), and its failure is a red VOID, not a fallback.
    """
    legal = w0.legal_spawns()
    a = float(w0.params.arena_size) - SPAWN_MARGIN
    axis = np.linspace(-a, a, SPAWN_GRID)
    step = axis[1] - axis[0]
    legal_set = {(round(float(x), 6), round(float(y), 6)) for x, y in legal}
    px, py = POOL_XY
    pool_r = float(w0.params.pool_size) + 0.9

    def clean(cx: float, cy: float) -> bool:
        for gx in axis:
            if abs(gx - cx) > CLEAR_R:
                continue
            for gy in axis:
                if (gx - cx) ** 2 + (gy - cy) ** 2 > CLEAR_R ** 2:
                    continue
                if (round(float(gx), 6), round(float(gy), 6)) not in legal_set:
                    return False
                if abs(gx - px) <= pool_r and abs(gy - py) <= pool_r:
                    return False
        return True

    best = None
    for cx in axis:
        for cy in axis:
            if cy < SEP_MIN / 2.0:
                continue
            # start poses sit D0 outward of each site and must be on-grid
            if abs(cx) > a - 0.2 or cy + D0 + 0.2 > a:
                continue
            if clean(cx, cy) and clean(cx, -cy):
                # prefer the tightest legal separation (shortest, best-mapped
                # corridor) and then the most central x
                key = (cy, abs(cx))
                if best is None or key < best[0]:
                    best = (key, (float(cx), float(cy)))
    if best is None:
        return None
    cx, cy = best[1]
    return (cx, cy), (cx, -cy)


# ── one forced-approach trial ───────────────────────────────────────────
def _restore(hw: HazardWorld, snap, start_xy) -> None:
    d = hw.w0.data
    d.qpos[:] = snap[0]
    d.qvel[:] = snap[1]
    d.qacc_warmstart[:] = 0.0
    d.xfrc_applied[:] = 0.0
    hw.w0.respawn(at=start_xy)            # places rover, resets drives + audio
    hw.reset_sites()


def _trial(hw: HazardWorld, snap, site_name: str, phi: float, eps: float,
           learner, learn: bool, drive_scale: float,
           pulse: int | None = None, d0: float = D0) -> dict:
    """One approach. Returns entries, felt events, and any prick encounter.

    `pulse=k` drives for the first k decisions only, then coasts/topples — the
    slow half of Part A's entry-speed ladder. `None` drives throughout.
    """
    site = hw.sites[site_name]
    twin = hw.sites["T" if site_name == "H" else "H"]
    outward = (site - twin) / float(np.linalg.norm(site - twin))
    c, s = math.cos(phi), math.sin(phi)
    off = np.array([c * outward[0] - s * outward[1],
                    s * outward[0] + c * outward[1]])
    start = site + d0 * off
    _restore(hw, snap, (float(start[0]), float(start[1])))

    psi = math.atan2(site[1] - start[1], site[0] - start[0]) + eps
    u = np.array([math.cos(psi), math.sin(psi)])

    # Settle: the 1 cm placement drop and the servo transient die out before
    # the approach starts, so the walk's own dynamics are all that is scored.
    for _ in range(SETTLE):
        a = np.zeros(8)
        a[0:4] = hw.arm_hold
        a[4:6] = -1.0
        hw.decide(a)

    entered = {"H": False, "T": False}
    felt_events = 0
    encounter = None                      # (v_entry, di, obs_delta_i, d_move,
                                          #  state_delta_i)
    pending = None
    for step in range(HORIZON):
        obs = hw.w0.observe()
        needs = obs["needs"]
        idot, d_h = float(needs[5]), float(needs[3])
        if pending is not None:
            v, di, d_before, i_before = pending
            obs_di = -idot * SIM_S_PER_DECISION
            state_di = i_before - float(hw.w0.drives.state.i)
            encounter = (v, di, obs_di, d_h - d_before, state_di)
            pending = None
        if learn and idot <= FELT_IDOT:
            learner.update(hw.xy(), -idot * SIM_S_PER_DECISION)
            felt_events += 1
        xy = hw.xy()
        vec = u
        if learner is not None and learner.value(xy + LOOKAHEAD * u) >= A_THRESH:
            vec = -u
        a = np.zeros(8)
        a[0:4] = hw.arm_hold              # arms stay at spawn pose
        a[4:6] = -1.0                     # adhesion off
        scale = drive_scale if (pulse is None or step < pulse) else 0.0
        a[6:8] = vec * scale
        i_before = float(hw.w0.drives.state.i)
        d_before = d_h
        hw.decide(a)
        for name in ("H", "T"):
            entered[name] = entered[name] or hw.inside[name]
        if hw.last_prick is not None:
            pending = (hw.last_prick[0], hw.last_prick[1], d_before, i_before)
        if entered[site_name]:
            break
        xy2 = hw.xy()
        if np.hypot(xy2[0] - site[0], xy2[1] - site[1]) > ABORT_DIST:
            break

    # A prick lands in the obs one decision AFTER it is applied; if the trial
    # ended on entry, take that one extra reading so the learner can feel it
    # and the encounter can be scored.
    if pending is not None:
        obs = hw.w0.observe()
        idot, d_h = float(obs["needs"][5]), float(obs["needs"][3])
        v, di, d_before, i_before = pending
        obs_di = -idot * SIM_S_PER_DECISION
        state_di = i_before - float(hw.w0.drives.state.i)
        encounter = (v, di, obs_di, d_h - d_before, state_di)
        if learn and idot <= FELT_IDOT:
            learner.update(hw.xy(), -idot * SIM_S_PER_DECISION)
            felt_events += 1

    return {"entered": entered, "felt": felt_events, "encounter": encounter,
            "i_end": float(hw.w0.drives.state.i)}


# ── the felt ruler (TA.01's, verbatim in spirit) ────────────────────────
def _clock_move() -> float:
    """What the ordinary passage of one decision costs in d(h): basal drain."""
    de = drives.BASAL_B * SIM_S_PER_DECISION
    return drives.drive(1.0 - de, 1.0, 0.0) - drives.drive(1.0, 1.0, 0.0)


# ── the phases, once per seed, shared by experiment and control ─────────
def _calibration():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _rate(results, site: str) -> float:
    return float(np.mean([r["entered"][site] for r in results]))


def _collect(seed: int) -> dict:
    if seed in _CACHE:
        return _CACHE[seed]
    j0, alpha, prov = _calibration()
    if j0 is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]

    w0 = W0(seed=seed, j0=j0, alpha=alpha, lethal=False)
    pair = _site_pair(w0)
    if pair is None:
        _CACHE[seed] = {"no_pair": True}
        return _CACHE[seed]
    hw = HazardWorld(w0, pair[0], pair[1])
    w0.mujoco.mj_forward(w0.model, w0.data)
    snap = (w0.data.qpos.copy(), w0.data.qvel.copy())

    rng = np.random.RandomState(seed * 3571 + 29)
    draws_h = [(float(rng.uniform(-APPROACH_SPREAD, APPROACH_SPREAD)),
                float(rng.normal(0.0, HEAD_SIGMA))) for _ in range(N_TRIALS)]
    draws_t = [(float(rng.uniform(-APPROACH_SPREAD, APPROACH_SPREAD)),
                float(rng.normal(0.0, HEAD_SIGMA))) for _ in range(N_TRIALS)]

    off = PlaceAversion(SIGMA_PLACE)      # empty forever: the naive policy

    # Part A — gradation: straight approaches over the (scale, pulse) ladder,
    # learner off. Certifies the delivered law on speeds this rig produces.
    grade = []
    for sc, pulse, d0 in GRADE_ARMS:
        r = _trial(hw, snap, "H", 0.0, 0.0, off, False, sc, pulse=pulse, d0=d0)
        if r["encounter"] is not None:
            grade.append(r["encounter"] + (r["i_end"],))

    # Part B — baseline, learner off.
    base_h = [_trial(hw, snap, "H", p, e, off, False, DRIVE_SCALE)
              for p, e in draws_h]
    base_t = [_trial(hw, snap, "T", p, e, off, False, DRIVE_SCALE)
              for p, e in draws_t]

    def expose_and_post(sigma: float, eta: float) -> dict:
        hw.eta = eta
        learner = PlaceAversion(sigma)
        exp = _trial(hw, snap, "H", 0.0, 0.0, learner, True, DRIVE_SCALE)
        post_h = [_trial(hw, snap, "H", p, e, learner, False, DRIVE_SCALE)
                  for p, e in draws_h]
        post_t = [_trial(hw, snap, "T", p, e, learner, False, DRIVE_SCALE)
                  for p, e in draws_t]
        hw.eta = ETA
        return {"exp": exp, "post_h": post_h, "post_t": post_t,
                "n_pts": len(learner.pts)}

    experiment = expose_and_post(SIGMA_PLACE, ETA)
    null = expose_and_post(SIGMA_PLACE, 0.0)
    control = expose_and_post(SIGMA_GLOBAL, ETA)

    _CACHE[seed] = {"grade": grade, "base_h": base_h, "base_t": base_t,
                    "experiment": experiment, "null": null, "control": control,
                    "prov": prov}
    return _CACHE[seed]


# ── the spec ────────────────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    d = _collect(seed)
    if "refused" in d:
        return {"borrow_ok": 0.0}
    if "no_pair" in d:
        return {"borrow_ok": 1.0, "rig_ok": 0.0, "site_pair_found": 0.0}

    grade = d["grade"]
    exp = d["experiment"]
    p_base_h, p_base_t = _rate(d["base_h"], "H"), _rate(d["base_t"], "T")
    p_post_h, p_post_t = _rate(exp["post_h"], "H"), _rate(exp["post_t"], "T")
    p_null_h = _rate(d["null"]["post_h"], "H")

    vs = np.array([g[0] for g in grade])
    dis = np.array([g[1] for g in grade])
    obs_dis = np.array([g[2] for g in grade])
    d_moves = np.array([g[3] for g in grade])
    state_dis = np.array([g[4] for g in grade])
    i_ends = np.array([g[5] for g in grade])

    order = np.argsort(vs)
    monotone = float(bool(np.all(np.diff(dis[order]) > -1e-12))) if len(grade) \
        else 0.0
    v_span = float(vs.max() / max(vs.min(), 1e-9)) if len(grade) else 0.0
    di_span = float(dis.max() / max(dis.min(), 1e-9)) if len(grade) else 0.0
    sensed_dev = float(np.max(np.abs(obs_dis - state_dis))) if len(grade) \
        else 1.0
    clock = _clock_move()
    felt_ratio_min = float(np.min(d_moves) / clock) if len(grade) else 0.0
    i_min_single = float(i_ends.min()) if len(grade) else 0.0

    delta = p_base_h - p_post_h
    m = {
        "borrow_ok": 1.0,
        "site_pair_found": 1.0,
        # rig health — VOID territory, not FAIL territory
        "base_entry_h": p_base_h,
        "base_entry_t": p_base_t,
        "n_grade_encounters": float(len(grade)),
        "v_span": v_span,
        "exposure_entered": float(exp["exp"]["entered"]["H"]),
        "exposure_felt": float(exp["exp"]["felt"]),
        # Part A — the graded, sensed, survivable law
        "di_span": di_span,
        "di_monotone_in_v": monotone,
        "sensed_dev": sensed_dev,
        "felt_ratio_min": felt_ratio_min,
        "i_min_single": i_min_single,
        "v_entry_min": float(vs.min()) if len(grade) else 0.0,
        "v_entry_max": float(vs.max()) if len(grade) else 0.0,
        "pricks_to_death": float(math.ceil((1.0 - DEATH_FLOOR) / DI_CAP)),
        # Part B — the headline
        "one_exposure_avoidance_delta": delta,
        "post_entry_h": p_post_h,
        "post_over_base": p_post_h / max(p_base_h, 1e-9),
        "twin_shift": p_base_t - p_post_t,
        "exposure_map_points": float(exp["n_pts"]),
        # the declared null
        "null_delta": p_base_h - p_null_h,
        "null_felt": float(d["null"]["exp"]["felt"]),
        "null_map_points": float(d["null"]["n_pts"]),
    }
    m["rig_ok"] = float(
        m["site_pair_found"] == 1.0
        and m["base_entry_h"] >= RIG_BASE_MIN
        and m["base_entry_t"] >= RIG_BASE_MIN
        and m["n_grade_encounters"] >= RIG_MIN_ENCOUNTERS
        and m["v_span"] >= RIG_V_SPAN_MIN
        and m["exposure_entered"] == 1.0
        and m["exposure_felt"] == 1.0)
    m["seed_gates_ok"] = float(
        m["rig_ok"] == 1.0
        # graded, sensed, precedes death
        and m["di_span"] >= DI_SPAN_MIN
        and m["di_monotone_in_v"] == 1.0
        and m["sensed_dev"] <= SENSED_DEV_MAX
        and m["felt_ratio_min"] >= FELT_RATIO_GATE
        and m["i_min_single"] >= SURVIVAL_MARGIN
        # one exposure shifts behaviour away from its cause
        and m["one_exposure_avoidance_delta"] >= DELTA_MIN
        and m["post_over_base"] <= POST_OVER_BASE_MAX
        # ...and not away from its harmless twin
        and abs(m["twin_shift"]) <= TWIN_BAND
        # ...and not without a cause
        and abs(m["null_delta"]) <= NULL_BAND
        and m["null_felt"] == 0.0
        and m["null_map_points"] == 0.0)
    return m


def _control(seed: int) -> dict:
    """GLOBAL FEAR: kernel width >= site separation. Its avoidance MUST
    transfer to the harmless twin — that is the twin gate demonstrating it can
    see transfer, and a global-fear arm that stayed site-specific would mean
    the twin criterion measures nothing."""
    d = _collect(seed)
    if "refused" in d or "no_pair" in d:
        return {"control_rig_ok": 0.0, "control_caught": 0.0}
    p_base_t = _rate(d["base_t"], "T")
    ctrl = d["control"]
    transfer = p_base_t - _rate(ctrl["post_t"], "T")
    c = {
        "control_rig_ok": float(ctrl["exp"]["entered"]["H"] == 1.0
                                and ctrl["exp"]["felt"] >= 1.0),
        "control_transfer_t": transfer,
        "control_post_h": _rate(ctrl["post_h"], "H"),
    }
    c["control_caught"] = float(transfer >= CTRL_TRANSFER_MIN
                                and transfer > TWIN_BAND)
    return c


def _check(m: dict, c: dict):
    if m.get("borrow_ok", 0.0) != 1.0:
        # An uncalibrated world refutes nothing (T0.22). VOID, never FAIL.
        return Status.VOID
    if m.get("rig_ok", 0.0) != 1.0 or c.get("control_rig_ok", 0.0) != 1.0:
        # The approach rig, not the hypothesis, failed to produce exposures.
        return Status.VOID
    return bool(
        m["seed_gates_ok"] == 1.0
        and m["one_exposure_avoidance_delta"] >= DELTA_MIN
        and m["post_over_base"] <= POST_OVER_BASE_MAX
        and abs(m["twin_shift"]) <= TWIN_BAND
        and abs(m["null_delta"]) <= NULL_BAND
        and c["control_caught"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PS.03"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
