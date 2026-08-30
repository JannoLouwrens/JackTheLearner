"""BA.03 — He braces against a surface: balance is used where direction still
has authority.

HYPOTHESIS (registry, unchanged). In a scenario where a graspable surface is
within reach, a learner given BA.01's vestibular channel PLACES ITS SUPPORT ON
THE LEAN SIDE and stays upright measurably longer than an identical learner
trained with the channel deleted (>= 3 sigma across seeds), and the gain
vanishes when the channel is replaced by matched-statistics noise.

WHY THIS SPEC EXISTS AND WHY IT IS NOT BA.02 AGAIN. BA.02 VOIDed its rig three
times at ~46 min a run, and its diagnosis (docstring, 2026-08-14) was not a
tuning miss: four scratch probes measured the ENVELOPE of the contrast BA.02
gates on at ~0.0-0.1 s ON OPEN GROUND, below the spec's own 0.20 s floor, for
every actuator group. D8 recorded that as "the rover has no actuation whose
useful effect depends on fall direction" and named `wall-brace` as option 3 —
the one scenario the four probes never tested. `ba03_wall_brace_probe.py`
tested it, and this spec is what that probe licenses.

## WHAT THE PROBE MEASURED, AND THE TWO TRAPS IT HANDS TO THIS FILE

Committed measurement (seed 90, `wall1` inner face, standoff 0.28 m, LATERAL
falls, N=10, 12.0 s horizon, hand-written ORACLE policies, no ledger row):

    hold           0.840 +- 0.058      out_gripboth  7.660 +- 0.685  <- best BLIND
    out_nogrip     0.860 +- 0.067      one-hand grip, labelling A    2.220 +- 0.351
                                       one-hand grip, labelling B    9.460 +- 0.538

    paired B - out_gripboth = +1.800 +- 0.538 s   (3.3 sigma)
    paired A - out_gripboth = -5.440 +- 0.341 s

  TRAP 1 — WHICH HAND IS THE RIGHT HAND IS NOT KNOWN TO THIS FILE. The probe's
  intuitive labelling (grip the hand on the LOW side) was 5.4 s WORSE than its
  opposite. That is the learner's discovery to make: nothing here encodes a
  side, and `brace_side_accuracy` is reported AGAINST THE LEAN SIDE exactly as
  the registry words it, with its SIGN a finding rather than a gate. What is
  gated is CONSISTENCY — that the braced side is a function of the lean side at
  all — because the registry's own second honest outcome is "the sensing arm
  wins but the brace lands on the wrong side as often as the right one".

  TRAP 2 — THE NULL IS 7.66 s, NOT 0.84. Against `random` or `hold` almost
  anything looks like balance. The blind twin here is a full linear reactive
  policy with the channel deleted, so the best fixed blind posture (extend and
  grip BOTH hands) is inside its class; and because CEM might simply fail to
  find it, the probe's hand-written `gripboth` posture is evaluated alongside
  as a REFERENCE, and a blind twin that falls far short of it VOIDs the run.
  A "gain" over a blind twin that never found its own best posture is an
  optimiser artifact wearing a sense's name.

  THE ARITHMETIC THAT PRECEDES BOTH (probe §1, generalised in LESSONS.md):
  `playground._rover_fragments` pins the arm bodies at body x = -+0.10 and puts
  both slides on y and z, so the arm-pair CoM sits at x == 0 for ALL reach and
  lift. Arm POSTURE has identically zero lateral authority. The actuator that
  can express "on the lean side" is ADHESION — 900 N at one hand and not the
  other is a moment about body-y — so adhesion is IN this spec's action space
  (BA.02 held it off) and the drive stays out (D8: directionally potent only in
  the harmful direction).

## ARMS — all trained/evaluated on the SAME worlds with PAIRED episode draws
## (site, tilt, kick, lean side identical across arms, episode for episode)

  vest      (experiment)  the full standardized 27-dim observation.
  deprived  (null twin)   graviceptive suffix pinned to its training mean
                          (zero after standardization). Identical architecture,
                          zero channel information, and NOT helpless: every
                          fixed posture and every blind reactive strategy —
                          including grip-both, the 7.66 s null — is in its
                          class.
  noise     (declared control, `_control`) suffix replaced each decision by
                          fresh N(0,1). Matched width, matched standardized
                          statistics, zero information. Its gain must VANISH.
  random    (null)        untrained policy class: a[:6] ~ U[-1,1] per decision.
  gripboth  (reference, REPORTED + rig-gated) the probe's hand-written best
                          fixed blind posture, replayed on the same packs. It
                          is the level the blind twin must reach.

## THE SECOND CONTROL THE REGISTRY MAKES MANDATORY: REMOVE THE SURFACE

The trained `vest` and `deprived` arms are re-evaluated on the SAME packs at an
open-ground site (BA.02's `_boundary_sites`-filtered spawn list, the cell
nearest the arena centre). D8's four probes put the open-ground directional
ceiling at ~0.0-0.1 s, so if the gain survives with no surface to brace
against, the rig is not measuring bracing and the run is VOID, not FAIL. Both
arms are equally out-of-distribution there, and the gate is on their
DIFFERENCE, which is what makes the transfer eval a fair test of the confound
rather than of the policies.

## GATES (PROVISIONAL until the seed-90 pilot freezes them — `_GATES_FROZEN`)

CLAIM (Status.FAIL when unmet):
  1. `gain_positive` == 1.0 and `t_gain` >= `T_GAIN_MIN` (3.0). The registry's
     own bar; it does not move.
  2. The matched-noise gain vanishes: `gain_noise` <= `NOISE_GAIN_FRAC_MAX` x
     `gain`, and `gain - gain_noise` >= `VEST_OVER_NOISE_MIN`.
  3. `brace_consistency` >= `BRACE_CONSISTENCY_MIN` — the braced side is a
     function of the lean side. Registry: brace-side accuracy is a REPORTED
     GATE, not a footnote.

RIG (Status.VOID — the world could not test the claim):
  4. The random policy topples on >= `TOPPLED_FRAC_MIN` of eval episodes and
     survives under `RANDOM_UP_FRAC_MAX` of the horizon.
  5. The best trained arm beats random by >= `IMPROVE_MARGIN_MIN`.
  6. The blind twin reaches the hand-written best fixed blind posture:
     `up_gripboth - up_deprived` <= `DEPRIVED_SHORTFALL_MAX`. (Trap 2.)
  7. The surface is load-bearing: `gain_nosurface` <= `NOSURF_GAIN_MAX`.
  8. A refused PS.01 borrow, or a non-finite statistic.

ANATOMY, REPORTED NEVER GATED (registry: "TWO CHANNELS SEPARATELY"). The
trained vest policy is re-evaluated on the paired eval packs with one
sub-block at a time pinned to its mean: touch, grav, canals, otoliths, vx/vy.
The linear-acceleration (otoliths) and angular-velocity (canals) contributions
are therefore reported apart, as BA.01's note and the registry require. A
brace carried wholly by one channel is a finding.

## SIZING IS PRE-REGISTERED AS A REQUIREMENT (registry, and the implementer
## may not skip it)

BA.02's scar: at k_fit = 3 against a paired per-episode sigma of 7.5 decisions
and a 1.375-decision signal, the CEM selection SNR was 0.32 and theta
random-walked — three VOIDs measured exactly that. The rule the registry
pre-registers is `k_fit >= (2*sigma/S)^2`, and `N_EVAL >= (sigma/SE_target)^2`,
sized against MEASURED noise, amending the TIER and never the thresholds.

CANDIDATES BELOW ARE SIZED FROM THE PROBE, which is the only measurement that
exists before the pilot: sigma_pair = 0.538 x sqrt(10) = 1.70 s, S = 1.80 s, so
(2 x 1.70 / 1.80)^2 = 3.6 -> `CEM_K_FIT` = 6 (1.7x margin), and
(1.70 / 0.25)^2 = 46 -> `N_EVAL` = 48 (SE 0.245 s). Those are ORACLE-arm
numbers; the pilot re-measures sigma from the TRAINED arms' own paired eval
deltas (`gain_se`, `sigma_pair_eval`) and the constants are frozen against
that, in a commit, before any registered run.

WHAT FAIL WOULD MEAN. Not "the learner was undertrained" — that is what the rig
gates are for. FAIL here means the channel buys nothing even where a surface
gives fall direction authority, D8's open-ground finding generalises, and the
honest status of balance-as-a-used-sense is 'sensed, unused' until a body with
directional catch authority exists (registry `kills`).

## PILOT RECORD — seed 90, ran 2026-08-30 13:15-15:00 UTC (6299.5 s wall, one
## seed, N_EVAL 48), artifact /data/ba03_pilot_seed90.json. GATES NOW FROZEN.

The pilot completed and then sat unharvested for eight hours: `_GATES_FROZEN`
stayed False, `_PILOT_OWED` went on saying "no pilot has been run: the artifact
does not exist" while the artifact existed on disk, and `run coverage` counted
BA.03 as pilot-owed shelf furniture. That is its own finding and it is fixed in
`coverage.py` (PILOT-HARVESTABLE) rather than only here.

THE RIG IS ALIVE ON EVERY CONJUNCT — the first time in the balance family.
BA.02 VOIDed three times with every arm sitting at random; this rig separates:

    up_random   2.1875 s   (18% of the 12.0 s horizon; toppled_frac 0.958)
    up_gripboth 7.4375 s   (the hand-written best fixed blind posture)
    up_deprived 10.6375 s  <- the blind twin BEAT the reference by 3.20 s
    up_vest     10.4000 s
    best_trained - up_random = 8.45 s   against IMPROVE_MARGIN_MIN 0.20
    deprived_shortfall = -3.20 s        against DEPRIVED_SHORTFALL_MAX 1.0

Trap 2 is closed by measurement, not assumption: CEM did not merely reach the
hand-written posture, it found a better one, so no "gain" here could be an
optimiser artifact. The surface is load-bearing exactly as D8 predicted —
removing it drops both arms to ~0.85 s and `gain_nosurface` reads 0.0042 s
against NOSURF_GAIN_MAX 0.30, which independently re-measures D8's open-ground
directional ceiling of ~0.0-0.1 s on this spec's own rig.

CEM SELECTION WORKS HERE, and that is BA.02's diagnostic run in reverse. BA.02
VOIDed because its matched-noise arm REPRODUCED the elite curve (order
statistics, not learning). Here the curves are vest 37.3 -> 50.8, deprived
27.0 -> 56.1, noise last 38.5: the blind twin ends 17.6 decisions above the
matched-noise arm trained under identical selection. Selection is
discriminating, so CEM_K_FIT = 6 is kept on measured evidence. The registry's
`(2*sigma/S)^2` form is NOT evaluable from this artifact — S is the elite-vs-
mean fitness gap and the pilot does not record it — and saying so is better
than reporting a number the artifact cannot support.

THE CLAIM READS NEGATIVE AT SEED 90, AND THE RUN IS BEING DISPATCHED ANYWAY.
`gain` = -0.2375 s (se 0.356), `gain_positive` 0.0: the vest arm is a fifth of
a standard error BELOW its blind twin. The matched-noise control behaves
emphatically (`gain_noise` -5.225 s, i.e. the noise arm loses 5.2 s to the
twin). The brace gate passes: `brace_consistency` 0.75 against 0.70, and the
composition matters — `brace_decisive_frac` is also 0.75 and
`brace_side_accuracy` is 0.0, so EVERY decisive brace went to the same side and
that side was the HIGH side, never the lean side. Consistency is ceilinged by
decisiveness (25% of episodes had |adh_L - adh_R| < ADH_SEP_MIN), not eroded by
inconsistency: the learner is perfectly consistent whenever it commits, and it
confirms the probe's Trap 1 — the intuitive labelling was backwards.

Seed 90 is disjoint from the registered 0/1/2, so it may size and it may not
decide. It forecasts FAIL and that forecast is recorded here BEFORE the run, so
the outcome cannot be narrated afterwards. Dispatching a spec whose pilot
predicts FAIL is not waste — refusing to is run-until-pass wearing thrift's
clothes. `balance` is a GOAL.md commitment with three declared specs and zero
passes, the rig is now demonstrably able to test the claim, and the registry
pre-registers what FAIL means here.

ANATOMY (reported, never gated) — and it is the most informative row. Pinning
one sub-block of the vest policy's input to its mean:

    touch     10.400 -> 3.0875   <- the whole policy
    grav      10.400 -> 10.7875
    canals    10.400 -> 10.9542
    otoliths  10.400 -> 10.3583
    vx/vy     10.400 -> 10.5500

The winning vest policy reads PLANTAR TOUCH and nothing vestibular; deleting
touch costs it 7.3 s, deleting any true vestibular block costs it nothing. And
the deprived twin, which has touch pinned too, still reaches 10.64 s by another
route. So the pilot's zero gain is not a failure to learn — it is two arms
finding equally good solutions, one of which happens to route through touch.

WHAT WAS SIZED, AND WHAT WAS NOT MOVED. No threshold moved. `N_EVAL` 48 -> 120,
derived (not fitted) from the measured `sigma_pair_eval` 2.4674 s — the probe's
48 was sized against an ORACLE sigma of 1.70 s and the trained arms are 45%
noisier. One gate was ADDED and it is strengthen-only: `HEADROOM_MIN_MULT`.

WHY THE NEW GATE EXISTS. Every rig gate above watches the RANDOM arm's distance
from the roof; none watched the NULL TWIN's. The twin sits at 10.6375 s of a
12.0 s horizon — 88.6% — so the largest gain physically available to the claim
is 1.3625 s, and at N_EVAL 48 the signal needed to clear T_GAIN_MIN was 1.068 s.
The claim had 1.28x the room it needed and no instrument in this file could say
so; a seed whose twin landed at 11.5 s would have been arithmetically incapable
of a PASS with all eight gates green. That is SH.02's saturated null (its
headroom VOID fired this morning at exactly 1.0000) and DP.04's unresolvable
statistic, arriving here third. `_rig` now requires
`HORIZON_S - up_deprived >= 2.0 * T_GAIN_MIN * gain_se`, VOID otherwise, and
reports `claim_headroom_s` / `claim_headroom_ratio` so the margin is visible
rather than derivable. The multiplier is the principled quantity and N_EVAL is
the consequence, in that order: sqrt(N) >= 2*3*2.4674/1.3625 = 10.87 -> N = 120,
which puts the pilot at ratio 2.016 — barely clearing a bar it did not set.

TIER RE-COST (the registry pre-authorises re-costing the TIER, never the
thresholds). 6299 s/seed at N_EVAL 48; N_EVAL 120 adds ~870 eval episodes
against ~5760 already run, so ~2.0 h/seed and ~6 h for three. That is outside
`CPU_LONG` (cpu<2h) — and `run.py` KILLS a child at the declared budget's
timeout, so leaving the label alone would have destroyed the run rather than
mislabelled it. Budget becomes `CPU_DAYS`.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

from ..protocol import Ledger, Status, borrow_metrics, run_spec   # noqa: E402
from ..registry import BY_ID                                      # noqa: E402
from ..w0 import W0, SIM_S_PER_DECISION                           # noqa: E402
# BA.01's rig constants and helpers, by reference (one definition of the fall).
from .ba_01_feels_the_fall import (GRAVITY, KICK_JIT, KICK_OMEGA_P,  # noqa: E402
                                   TOPPLE_UP, VEST_DIM,
                                   _boundary_sites, _tilt_quat)

# The claim goes stale when the world, the body, the drive layer or the sense's
# own defining rig moves.
IMPL_DEPS = ["experiments/w0.py", "playground.py", "experiments/drives.py",
             "experiments/tests/ba_01_feels_the_fall.py"]

# ── observation layout (BA.01's ordering, verbatim — BA.02's constants) ──
BLIND_DIM = 8                 # 4 arm slide positions + 4 slide velocities
TOUCH_DIM = 8
CH_DIM = TOUCH_DIM + VEST_DIM  # the graviceptive suffix BA.01 registered (19)
OBS_DIM = BLIND_DIM + CH_DIM   # 27
# ADHESION IS IN THE ACTION SPACE — the whole reason this spec is askable
# (probe §1: the slides have identically zero lateral authority; 900 N at one
# hand is a moment about body-y). The drive stays out: D8 measured it
# directionally potent only in the harmful direction.
ACT_DIM = 6                    # 4 slide targets + 2 adhesion commands
THETA_DIM = ACT_DIM * OBS_DIM + ACT_DIM   # 168 linear-policy parameters

# ── the scenario (PRE-REGISTERED; probe §2) ─────────────────────────────
WALL_GEOM = "wall1"           # the only large flat vertical surface in reach
STANDOFF = 0.28               # m from the wall face to the body axis
# BA.02 V3's tilt draw, by value: below ~4 deg the contact-solver floor and the
# damping-10 free joint quench any action's influence on fall time.
BA03_TILT0_LOG10_DEG = (0.6, 1.4)     # theta ~ 10^U[...]: 4 to 25 deg
# LATERAL falls only. A fall with no side cannot have a lean side to brace on,
# and the probe's v2 round (fore/aft) returned a confident negative for exactly
# that reason. `aim` is +-x; sides alternate so the eval set is balanced.
AIM_LEFT, AIM_RIGHT = math.pi, 0.0

# ── the rig envelope (PRE-REGISTERED) ───────────────────────────────────
HORIZON = 60                  # decisions = 12 s; braces run to the horizon
T_SETTLE = 3                  # hold decisions before the tilt+kick (BA.01)
N_STATS_EP = 12               # random-policy pre-pass sizing mu/sd + noise match

# ── the learner (PRE-REGISTERED; identical across arms by construction) ─
CEM_POP = 24
CEM_ELITE = 6
CEM_ITERS = 12
CEM_SIG_INIT = 0.5
CEM_SIG_FLOOR = 0.05

# ── gates. FROZEN against the seed-90 pilot (2026-08-30; see PILOT RECORD) ──
_GATES_FROZEN = True
_PILOT_ARTIFACT = "/data/ba03_pilot_seed90.json"

# PILOT-SIZED (the registry's sizing requirement; see the PILOT RECORD).
CEM_K_FIT = 6                 # kept: the pilot MEASURED selection working
N_EVAL = 120                  # was 48. Derived from the pilot's own sigma and
# ceiling headroom: sqrt(N) >= HEADROOM_MIN_MULT * T_GAIN_MIN * sigma_pair /
# (HORIZON_S - up_deprived) = 2*3*2.4674/1.3625 = 10.87 -> N >= 118.1 -> 120.
# The probe's 48 was sized against an ORACLE sigma of 1.70 s; the trained arms
# measure 2.4674 s. This is the registry's own sizing formula applied to the
# measured statistic, and it re-costs the TIER (CPU_LONG -> CPU_DAYS), never a
# threshold.

# THE REGISTRY'S BAR (constitutional here, does not move).
T_GAIN_MIN = 3.0
# Claim gates (FAIL). FROZEN 2026-08-30 — every one UNCHANGED from its
# pre-pilot candidate value; the pilot cleared each and moved none.
NOISE_GAIN_FRAC_MAX = 0.50    # gain_noise <= this fraction of gain (BA.02's)
VEST_OVER_NOISE_MIN = 0.20    # sim-s: gain - gain_noise floor      (BA.02's)
BRACE_CONSISTENCY_MIN = 0.70  # the braced side is a function of the lean side
ADH_SEP_MIN = 0.5             # |a[4] - a[5]| below this is "no side chosen"
# Rig gates (VOID, not FAIL — a world that could not test the claim).
TOPPLED_FRAC_MIN = 0.60       # random policy must actually fall
RANDOM_UP_FRAC_MAX = 0.80     # ...but not survive ~the horizon
IMPROVE_MARGIN_MIN = 0.20     # sim-s: best trained arm over random
DEPRIVED_SHORTFALL_MAX = 1.0  # sim-s: blind twin vs the hand-written gripboth
NOSURF_GAIN_MAX = 0.30        # sim-s: D8's open-ground ceiling is ~0.0-0.1
HEADROOM_MIN_MULT = 2.0       # NEW, strengthen-only: the room above the NULL
# TWIN must be at least this multiple of the signal the claim needs to clear
# T_GAIN_MIN. 2.0 is chosen on principle — a claim decided by the ceiling
# rather than by the sense is not a measurement — and NOT fitted: N_EVAL was
# then derived FROM it, which is why the pilot reads 2.02x rather than
# something comfortable. See RESOLUTION in `_rig` and LESSONS.md.

_CACHE: dict = {}


def _calibration() -> tuple:
    """PS.01's j0/alpha, or a refusal. W0 has no defaults for them (BA.01)."""
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _wall_face(w: W0) -> float:
    """The y of `wall1`'s inner face, read from the model, never a constant.

    `w.ix["geom"]` maps only the rover's own geoms, so the arena wall is
    resolved through MuJoCo's name table (probe `_wall_face`, verbatim)."""
    gid = w.mujoco.mj_name2id(w.model, w.mujoco.mjtObj.mjOBJ_GEOM, WALL_GEOM)
    if gid < 0:
        raise RuntimeError(f"no geom named {WALL_GEOM!r} in this world")
    return float(w.model.geom_pos[gid][1]) + float(w.model.geom_size[gid][1])


def _joint_action(w: W0, reach: float, lift: float,
                  adh_l: float = -1.0, adh_r: float = -1.0) -> np.ndarray:
    """Joint-space targets -> W0's 8-vector, through the LIVE ctrlrange.

    PS.03's phantom-servo scar: a = 0 is MID-range, not neutral, so a posture
    is never written as a raw action value. Drive dims stay 0 (genuinely none).
    """
    lo = np.asarray(w.model.actuator_ctrlrange[:4, 0], dtype=float)
    hi = np.asarray(w.model.actuator_ctrlrange[:4, 1], dtype=float)
    a = np.zeros(8)
    t = np.array([reach, lift, reach, lift])
    a[:4] = 2.0 * (t - lo) / (hi - lo) - 1.0
    a[4], a[5] = adh_l, adh_r
    return np.clip(a, -1.0, 1.0)


def _arm_hold(w: W0) -> np.ndarray:
    """The 8-vector that commands the SPAWN pose, adhesion off."""
    return _joint_action(w, 0.0, 0.0)


# The probe's best fixed BLIND posture, by value: extend toward the wall and
# grip BOTH hands. It encodes no side (that is Trap 1) and it is the 7.66 s
# null the blind twin must reach (Trap 2).
GRIPBOTH_REACH, GRIPBOTH_LIFT = -0.25, 0.10


def _brace_site(w: W0) -> tuple:
    """The wall site. Checked against the world's OWN legality predicate: it
    sits outside `legal_spawns()`, whose grid stops at arena_size -
    SPAWN_MARGIN, and SPAWN_MARGIN sizes the uniform-spawn probe rather than
    stating a law (probe §2)."""
    return (0.0, _wall_face(w) + STANDOFF)


def _site_is_legal(w: W0, site: tuple) -> bool:
    w._place(float(site[0]), float(site[1]))
    w.mujoco.mj_forward(w.model, w.data)
    return not w._penetrating()


def _open_site(w: W0) -> tuple:
    """The NO-SURFACE control's site: the open legal cell nearest the arena
    centre. `_boundary_sites` is BA.01's own definition of "against the
    geometry" — BA.02 V2(c) removed those cells because a body that topples
    against an obstacle and rests on it scores upright longer by leaning."""
    legal = np.asarray(w.legal_spawns(), dtype=float)
    bs = _boundary_sites(w)
    if len(bs):
        bset = {(round(float(x), 9), round(float(y), 9)) for x, y, _b in bs}
        keep = [i for i, (x, y) in enumerate(legal)
                if (round(float(x), 9), round(float(y), 9)) not in bset]
        if keep:
            legal = legal[keep]
    k = int(np.argmin(np.sum(legal ** 2, axis=1)))
    return (float(legal[k][0]), float(legal[k][1]))


def _draw_pack(rng: np.random.RandomState, left: bool) -> dict:
    """One LATERAL fall — everything that must be PAIRED across arms. The site
    is fixed (the scenario IS "a graspable surface within reach"); tilt, kick
    and lean side are the draw. Tilt/kick rules are BA.01's, verbatim."""
    theta = math.radians(10.0 ** rng.uniform(*BA03_TILT0_LOG10_DEG))
    mag = theta * KICK_OMEGA_P * 10.0 ** rng.uniform(*KICK_JIT)
    u = rng.randn(3)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    return {"aim": AIM_LEFT if left else AIM_RIGHT, "left": bool(left),
            "theta": theta, "kick": u * mag}


def _obs_row(w: W0, v_prev: np.ndarray) -> tuple:
    """BA.01's 27-dim row, ordering verbatim: blind(8) + touch(8) + vest(11)."""
    da = w.ix["root_dofadr"]
    xmat = w.data.xmat[w.rover_bid]
    R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    grav_body = -R.T @ np.array([0.0, 0.0, 1.0])
    canals = w.data.qvel[da + 3:da + 6].copy()
    v = w.data.qvel[da:da + 3].copy()
    a_world = (v - v_prev) / SIM_S_PER_DECISION
    otoliths = R.T @ (a_world - np.array([0.0, 0.0, -GRAVITY]))
    p = w._proprio()
    row = np.concatenate([p[:8], w._touch(),
                          grav_body, canals, otoliths, p[10:12]])
    assert row.shape[0] == OBS_DIM
    return row, v, float(xmat[8])


def _episode(w: W0, pack: dict, act_fn, hold: np.ndarray, site: tuple,
             horizon: int = HORIZON) -> tuple:
    """One brace attempt. Returns (decisions upright, rows, first_action).

    `first_action` is the 6-vector commanded at the FIRST decision after the
    kick — the brace decision, which `brace_side_accuracy` reads. `respawn`
    resets pose, arms, velocities and the drive state (a fresh body, gear 1.0).
    """
    mujoco = w.mujoco
    qa, da = w.ix["root_qposadr"], w.ix["root_dofadr"]
    w.respawn(at=site)
    for _ in range(T_SETTLE):
        w.decide(hold)
    q0 = w.data.qpos[qa + 3:qa + 7].copy()
    qt = _tilt_quat(pack["theta"], pack["aim"])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qt, q0)
    w.data.qpos[qa + 3:qa + 7] = out
    w.data.qvel[da:da + 6] = 0.0
    w.data.qvel[da + 3:da + 6] = pack["kick"]
    mujoco.mj_forward(w.model, w.data)

    v_prev = w.data.qvel[da:da + 3].copy()
    rows, first = [], None
    for t in range(horizon):
        row, v_prev, up = _obs_row(w, v_prev)
        if up < TOPPLE_UP:
            return t, rows, first
        rows.append(row)
        u = act_fn(row)
        if first is None:
            first = np.asarray(u, dtype=float).copy()
        a = np.array(hold)
        a[:6] = u
        w.decide(a)
    return horizon, rows, first


# ── the policy class and its conditions ─────────────────────────────────
def _policy(theta: np.ndarray, mu: np.ndarray, sd: np.ndarray, cond: str,
            noise_rng: np.random.RandomState | None):
    """cond: 'vest' full obs | 'deprived' suffix at its mean | 'noise' suffix
    i.i.d. N(0,1) per decision | 'ablate:<a>:<b>' pinning dims [a:b) of the
    STANDARDIZED row to zero (the anatomy evals). Identical architecture in
    every case — only the information in the suffix differs."""
    W = theta[:ACT_DIM * OBS_DIM].reshape(ACT_DIM, OBS_DIM)
    b = theta[ACT_DIM * OBS_DIM:]

    def act(row: np.ndarray) -> np.ndarray:
        z = (row - mu) / sd
        if cond == "deprived":
            z[BLIND_DIM:] = 0.0
        elif cond == "noise":
            z[BLIND_DIM:] = noise_rng.randn(CH_DIM)
        elif cond.startswith("ablate:"):
            _, a0, b0 = cond.split(":")
            z[int(a0):int(b0)] = 0.0
        return np.tanh(W @ z + b)
    return act


def _random_policy(rng: np.random.RandomState):
    def act(row: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, ACT_DIM)
    return act


def _gripboth_policy(w: W0):
    """The probe's hand-written best fixed BLIND posture, as a 6-vector."""
    a = _joint_action(w, GRIPBOTH_REACH, GRIPBOTH_LIFT, adh_l=1.0, adh_r=1.0)
    u = a[:6].copy()

    def act(row: np.ndarray) -> np.ndarray:
        return u
    return act


def _cem_step(w: W0, st: dict, hold: np.ndarray, site: tuple, mu, sd,
              cond: str, seed: int, it: int, pop, elite, k_fit,
              horizon) -> None:
    """One CEM iteration for one arm. BA.02 V2(a): `_collect` interleaves these
    so no arm trains on a systematically fresher world (W0 never resets the
    world — `_place` deliberately omits `mj_resetData` — so catchability
    drifts across a run, and v1 measured drift as a gain).

    Common draws per iteration: every candidate faces the same k_fit packs, and
    every ARM's `g` stream is seeded identically, so arms face the same packs
    and the same candidate perturbations. Lean sides alternate WITHIN each
    iteration's pack list, so no candidate is ranked on one side only."""
    g = st["g"]
    packs = [_draw_pack(g, left=(i % 2 == 0)) for i in range(k_fit)]
    cands = [st["theta"] + st["sig"] * g.randn(THETA_DIM) for _ in range(pop)]
    fits = []
    for ci, th in enumerate(cands):
        ups = []
        for ei, pack in enumerate(packs):
            nr = (np.random.RandomState(
                seed * 1_000_003 + it * 10_007 + ci * 101 + ei)
                  if cond == "noise" else None)
            up, _, _ = _episode(w, pack, _policy(th, mu, sd, cond, nr),
                                hold, site, horizon)
            ups.append(up)
        fits.append(float(np.mean(ups)))
    order = np.argsort(fits)[::-1]
    el = np.stack([cands[i] for i in order[:elite]])
    st["theta"] = el.mean(0)
    st["sig"] = np.maximum(el.std(0), CEM_SIG_FLOOR)
    st["curve"].append(float(np.mean([fits[i] for i in order[:elite]])))


def _collect(seed: int, iters=CEM_ITERS, pop=CEM_POP, elite=CEM_ELITE,
             k_fit=CEM_K_FIT, n_eval=N_EVAL, n_stats=N_STATS_EP,
             horizon=HORIZON) -> dict:
    """Everything this spec needs for one seed, once. Cached: `_control` reuses
    the same trained arms and the same paired eval draws (BA.01's pattern —
    re-simulating would let the two sides differ by something other than the
    channel)."""
    key = (seed, iters, pop, elite, k_fit, n_eval, n_stats, horizon)
    if key in _CACHE:
        return _CACHE[key]
    j0, alpha, prov = _calibration()
    if j0 is None:
        _CACHE[key] = {"refused": prov}
        return _CACHE[key]
    t0 = time.time()
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=False)
    site = _brace_site(w)
    site_legal = _site_is_legal(w, site)
    open_site = _open_site(w)
    hold = _arm_hold(w)

    # Stats pre-pass: mu/sd for standardization AND the matched-noise scale.
    srng = np.random.RandomState(seed * 613 + 7)
    rows_all = []
    for i in range(n_stats):
        pack = _draw_pack(srng, left=(i % 2 == 0))
        _, rows, _ = _episode(w, pack,
                              lambda r: srng.uniform(-1, 1, ACT_DIM),
                              hold, site, horizon)
        rows_all.extend(rows)
    X = np.asarray(rows_all)
    mu, sd = X.mean(0), X.std(0) + 1e-8

    # Interleaved CEM (BA.02 V2(a)): one iteration per arm in rotating order.
    tconds = ("vest", "deprived", "noise")
    st = {c: {"theta": np.zeros(THETA_DIM),
              "sig": np.full(THETA_DIM, CEM_SIG_INIT),
              "g": np.random.RandomState(seed * 9973 + 101),
              "curve": []} for c in tconds}
    for it in range(iters):
        r = it % len(tconds)
        for cond in tconds[r:] + tconds[:r]:
            _cem_step(w, st[cond], hold, site, mu, sd, cond, seed, it,
                      pop, elite, k_fit, horizon)
    arms = {c: st[c]["theta"] for c in tconds}
    curves = {c: st[c]["curve"] for c in tconds}

    # Paired eval draws, one list per seed, shared by every condition. Lean
    # sides alternate so `brace_side_accuracy` has a balanced denominator.
    erng = np.random.RandomState(seed * 271 + 17)
    packs = [_draw_pack(erng, left=(i % 2 == 0)) for i in range(n_eval)]
    rrng = np.random.RandomState(seed * 83 + 5)
    gb = _gripboth_policy(w)

    nb = BLIND_DIM
    blocks = {"touch": (nb, nb + 8), "grav": (nb + 8, nb + 11),
              "canals": (nb + 11, nb + 14), "otoliths": (nb + 14, nb + 17),
              "vxvy": (nb + 17, nb + 19)}

    def _act_for(c: str, ei: int):
        if c == "random":
            return _random_policy(rrng)
        if c == "gripboth":
            return gb
        if c.startswith("anat:"):
            a0, b0 = blocks[c[5:]]
            return _policy(arms["vest"], mu, sd, f"ablate:{a0}:{b0}", None)
        base = c[:-3] if c.endswith("_os") else c
        nr = (np.random.RandomState(seed * 41 + 900_000 + ei)
              if base == "noise" else None)
        return _policy(arms[base], mu, sd, base, nr)

    # Interleaved eval (BA.02 V2(b)): every condition runs per episode in
    # rotating order, so the drift differential between any two conditions is
    # bounded by per-episode drift, never per-block. `*_os` are the NO-SURFACE
    # control: the identical trained arms, the identical packs, open ground.
    econds = ["vest", "deprived", "noise", "random", "gripboth",
              "vest_os", "deprived_os"] + [f"anat:{n}" for n in blocks]
    ups: dict = {c: [] for c in econds}
    braces = []
    for ei, pack in enumerate(packs):
        r = ei % len(econds)
        for c in econds[r:] + econds[:r]:
            s = open_site if c.endswith("_os") else site
            up, _, first = _episode(w, pack, _act_for(c, ei), hold, s, horizon)
            ups[c].append(up * SIM_S_PER_DECISION)
            if c == "vest":
                braces.append((bool(pack["left"]), None if first is None
                               else float(first[4] - first[5])))

    ev = {c: np.asarray(ups[c], dtype=float) for c in econds}
    toppled_random = float(np.mean(ev["random"] <
                                   horizon * SIM_S_PER_DECISION - 1e-9))
    anatomy = {n: float(ev[f"anat:{n}"].mean()) for n in blocks}

    # THE BRACE SIDE (Trap 1). `d = adh_L - adh_R` at the first post-kick
    # decision. handL sits at body x = -0.10, so a LEFT (-x) lean is handL's
    # side. Accuracy is reported AGAINST THE LEAN SIDE exactly as the registry
    # words it; what is GATED is consistency, because which side wins is the
    # learner's finding and the probe's intuitive labelling was 5.4 s wrong.
    n_dec = sum(1 for _l, d in braces if d is not None and abs(d) >= ADH_SEP_MIN)
    n_lean = sum(1 for l, d in braces
                 if d is not None and abs(d) >= ADH_SEP_MIN and (d > 0) == l)
    n_hi = n_dec - n_lean
    n_b = max(len(braces), 1)

    # Drift recheck, REPORTED never gated (BA.02 V2(d)).
    n_re = min(8, n_eval)
    re_ups = [_episode(w, packs[ei], _act_for("vest", ei), hold, site,
                       horizon)[0] * SIM_S_PER_DECISION for ei in range(n_re)]
    drift_recheck = float(np.mean(re_ups) - float(np.mean(ups["vest"][:n_re])))

    _CACHE[key] = {"ev": ev, "curves": curves, "anatomy": anatomy,
                   "toppled_random": toppled_random, "prov": prov,
                   "drift_recheck": drift_recheck, "site": list(site),
                   "site_legal": bool(site_legal), "open_site": list(open_site),
                   "brace": {"decisive_frac": n_dec / n_b,
                             "accuracy": n_lean / n_b,
                             "consistency": max(n_lean, n_hi) / n_b},
                   "wall_s": time.time() - t0, "horizon": horizon,
                   "n_eval": n_eval, "k_fit": k_fit, "iters": iters}
    return _CACHE[key]


def _rig(c: dict) -> dict:
    """Per-seed rig health; the conjunction is VOID-gated in `_check`."""
    h_s = c["horizon"] * SIM_S_PER_DECISION
    ev = c["ev"]
    up_random = float(ev["random"].mean())
    up_deprived = float(ev["deprived"].mean())
    up_gripboth = float(ev["gripboth"].mean())
    best_trained = max(float(ev[k].mean())
                       for k in ("vest", "deprived", "noise"))
    # The NO-SURFACE control (registry-mandated): the same trained arms, the
    # same packs, open ground. D8's directional ceiling there is ~0.0-0.1 s.
    gain_os = float(ev["vest_os"].mean() - ev["deprived_os"].mean())
    # RESOLUTION (added 2026-08-30 from the seed-90 pilot; strengthen-only).
    # Every gate above watches how far the RANDOM arm sits from the roof and
    # none watches the NULL TWIN. The claim is a DIFFERENCE, so what bounds it
    # is the room left ABOVE the twin: the pilot put the blind twin at 10.6375 s
    # of a 12.0 s horizon, leaving 1.3625 s for a contrast that must clear
    # T_GAIN_MIN * gain_se. A twin near the roof makes the claim arithmetically
    # unreachable while every other gate here still reads green — SH.02's
    # saturated null and DP.04's unresolvable statistic, arriving on this rig.
    # Reported as well as gated, so a reader can see the margin, not infer it.
    delta = ev["vest"] - ev["deprived"]
    n_d = max(len(delta), 2)
    se_gain = float(delta.std(ddof=1)) / math.sqrt(n_d)
    headroom = h_s - up_deprived
    need = T_GAIN_MIN * se_gain
    ok = (c["site_legal"]
          and c["toppled_random"] >= TOPPLED_FRAC_MIN
          and up_random <= RANDOM_UP_FRAC_MAX * h_s
          and best_trained - up_random >= IMPROVE_MARGIN_MIN
          # Trap 2: a blind twin that never found its own best fixed posture
          # makes any "gain" an optimiser artifact wearing a sense's name.
          and up_gripboth - up_deprived <= DEPRIVED_SHORTFALL_MAX
          and gain_os <= NOSURF_GAIN_MAX
          and headroom >= HEADROOM_MIN_MULT * need)
    return {"toppled_frac_random": c["toppled_random"],
            "up_random": up_random, "up_gripboth": up_gripboth,
            "best_trained": best_trained,
            "deprived_shortfall": up_gripboth - up_deprived,
            "gain_nosurface": gain_os,
            "claim_headroom_s": headroom,
            "claim_headroom_ratio": headroom / need if need > 0 else float("inf"),
            "site_legal": 1.0 if c["site_legal"] else 0.0,
            "seed_rig_ok": 1.0 if ok else 0.0}


def _experiment(seed: int, **env) -> dict:
    c = _collect(seed, **env)
    if "refused" in c:
        return {"probe": "VOID", "gain": float("nan"), **c["refused"]}
    ev = c["ev"]
    up_v, up_d = float(ev["vest"].mean()), float(ev["deprived"].mean())
    delta = ev["vest"] - ev["deprived"]
    n = max(len(delta), 2)
    sig_pair = float(delta.std(ddof=1))
    out = {"up_vest": up_v, "up_deprived": up_d, "gain": up_v - up_d,
           "gain_positive": 1.0 if up_v - up_d > 0 else 0.0,
           # The sizing evidence the registry demands, measured on the TRAINED
           # arms rather than borrowed from the probe's oracles.
           "sigma_pair_eval": sig_pair, "gain_se": sig_pair / math.sqrt(n),
           "brace_side_accuracy": c["brace"]["accuracy"],
           "brace_consistency": c["brace"]["consistency"],
           "brace_decisive_frac": c["brace"]["decisive_frac"],
           **_rig(c),
           "up_vest_nosurface": float(ev["vest_os"].mean()),
           "up_deprived_nosurface": float(ev["deprived_os"].mean()),
           "vest_fit_first": c["curves"]["vest"][0],
           "vest_fit_last": c["curves"]["vest"][-1],
           "deprived_fit_first": c["curves"]["deprived"][0],
           "deprived_fit_last": c["curves"]["deprived"][-1],
           "drift_recheck": c["drift_recheck"], "wall_s": c["wall_s"]}
    for k, v in c["anatomy"].items():
        out[f"up_ablate_{k}"] = v
    return out


def _control(seed: int, **env) -> dict:
    """The declared control: matched-statistics noise in the channel. Its gain
    over the deprived twin must vanish."""
    c = _collect(seed, **env)
    if "refused" in c:
        return {"probe": "VOID", "gain_noise": float("nan"), **c["refused"]}
    up_n = float(c["ev"]["noise"].mean())
    up_d = float(c["ev"]["deprived"].mean())
    return {"up_noise": up_n, "gain_noise": up_n - up_d,
            "noise_fit_last": c["curves"]["noise"][-1],
            "seed_rig_ok": _rig(c)["seed_rig_ok"]}


def _declared_void(m: dict) -> bool:
    v = m.get("gain", m.get("gain_noise", float("nan")))
    return m.get("probe") == "VOID" or not np.isfinite(v)


def _check(m: dict, c: dict):
    if _declared_void(m) or _declared_void(c):
        return Status.VOID
    # Rig degeneracy is VOID, per seed: the mean of the conjunction is 1.0 only
    # when EVERY seed's world could have tested the claim.
    if m["seed_rig_ok"] < 1.0:
        return Status.VOID
    # The registry's bar: >= 3 sigma across seeds, every seed positive.
    # `_aggregate` hands the POPULATION std over n=3 seeds; the t statistic
    # wants the SAMPLE std over the seed mean: t = mean / (s/sqrt(n)) with
    # s = std_pop*sqrt(n/(n-1)), which reduces to mean*sqrt(2)/std_pop at n=3.
    t_gain = m["gain"] * math.sqrt(2.0) / max(m.get("gain_std", 0.0), 1e-9)
    ok = (m["gain_positive"] == 1.0
          and t_gain >= T_GAIN_MIN
          # the control must FAIL: the matched-noise gain vanishes
          and c["gain_noise"] <= NOISE_GAIN_FRAC_MAX * m["gain"]
          and m["gain"] - c["gain_noise"] >= VEST_OVER_NOISE_MIN
          # the registry's REPORTED GATE: the brace has a side, and that side
          # is a function of the lean side. Its SIGN is a finding, not a gate.
          and m["brace_consistency"] >= BRACE_CONSISTENCY_MIN)
    return Status.PASS if ok else Status.FAIL


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:                                  # pragma: no cover
        raise RuntimeError(
            "BA.03 gates are PROVISIONAL. Run the seed-90 pilot "
            "(`python -m experiments.tests.ba_03_braces_against_a_surface "
            f"pilot`), read {_PILOT_ARTIFACT}, size CEM_K_FIT/N_EVAL against "
            "the measured `sigma_pair_eval` per the registry's sizing "
            "requirement, freeze BRACE_CONSISTENCY_MIN / "
            "DEPRIVED_SHORTFALL_MAX / NOSURF_GAIN_MAX against it in a commit, "
            "then set _GATES_FROZEN = True. A gate fitted to the run it judges "
            "is not a gate.")
    return run_spec(BY_ID["BA.03"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


# ── smoke and pilot ─────────────────────────────────────────────────────
def _smoke():
    """Tiny envelope, every entry point once — including the `_check` path,
    which is where a shape error would otherwise wait for the registered run."""
    env = dict(iters=2, pop=6, elite=2, k_fit=2, n_eval=4, n_stats=2,
               horizon=24)
    m = _experiment(0, **env)
    c = _control(0, **env)
    print("smoke experiment:", json.dumps(m, indent=1, default=float))
    print("smoke control:", json.dumps(c, indent=1, default=float))
    print("smoke check path:", _check({**m, "gain_std": 1.0}, c))


def _pilot():
    """Seed 90 (disjoint from the registered 0/1/2), full envelope, JSON to
    stdout AND to `_PILOT_ARTIFACT`. Writes no ledger row."""
    t0 = time.time()
    m = _experiment(90)
    c = _control(90)
    out = {"seed": 90, "experiment": m, "control": c,
           "constants": {"iters": CEM_ITERS, "pop": CEM_POP,
                         "elite": CEM_ELITE, "k_fit": CEM_K_FIT,
                         "n_eval": N_EVAL, "horizon": HORIZON,
                         "standoff_m": STANDOFF},
           "pilot_wall_s": time.time() - t0}
    txt = json.dumps(out, default=float, indent=1)
    try:
        with open(_PILOT_ARTIFACT, "w") as fh:
            fh.write(txt)
    except OSError as exc:                       # pragma: no cover
        print(f"WARN: could not write {_PILOT_ARTIFACT}: {exc}")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _smoke()
    elif len(sys.argv) > 1 and sys.argv[1] == "pilot":
        _pilot()
    else:
        print(run().status)
