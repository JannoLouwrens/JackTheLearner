"""BA.01 — He feels himself falling before he falls.

HYPOTHESIS (registry, unchanged). Jack carries a sensed orientation signal —
gravity's direction in his own body frame — from which a linear probe recovers
tilt, and from which time-to-topple is predictable while he is still upright.

THE BODY IS AN INVERTED PENDULUM AND THAT IS THE POINT. W0.BAL (integration
queue, raised by LC.02's measurement 2026-08-09): a 30 kg capsule on a 0.09 m
spherical foot topples under random action within ~20 decisions and slides on
its side. This spec does not fix that — fixing it is a control problem the
LC bakeoff owns. It asks the question that must be answerable BEFORE any
balance controller could exist: while the body is still upright, do his senses
carry the fall that is coming? A creature that cannot feel falling cannot learn
to catch itself, however good its learning rule.

THE VESTIBULAR CHANNEL, REGISTERED AS BIOLOGY HANDS IT TO US (registry notes):
not one signal but two —

  otoliths  linear acceleration: specific force in the body frame,
            f = R^T (a_world - g). At rest it reads 9.81 body-up; in free
            fall it reads zero. This is what distinguishes FALLING from
            BEING CARRIED — a body moved at constant velocity feels rest.
  canals    angular velocity in the body (local) frame, straight from the
            free joint's rotational dofs.

plus gravity's direction in the body frame (the static orientation signal the
hypothesis names) and the root's horizontal velocity. Root vx/vy belong on the
balance side deliberately: on a pivoting foot the origin's horizontal velocity
is tilt rate times height — a control that kept them would read the fall
through kinematic coupling and the spec would be measuring bookkeeping.

TOUCH IS A BALANCE ORGAN TOO, AND THE PILOT PROVED IT. The seed-90 pilot put
the eight touch floats on the blind side and they alone scored AUC 0.94: the
foot's normal force unloads as the fall accelerates (N ~ m(g - a_z)), which is
plantar graviception — the same somatosensory balance contribution biology
gives the sole of the foot. A "control" keeping them would not be reading a
clock, it would be reading a second balance organ. So the ORIENTATION CHANNEL
this spec registers — and its control deletes — is the whole graviceptive
suffix: touch + vestibular, with the vestibular-proper numbers (auc_vest,
auc_grav, auc_canals, auc_otoliths, auc_touch) reported separately so the
claim's anatomy is visible and no single organ hides behind the block.

THE BLIND SIDE is what remains and it is measured blind, not assumed blind:
the four arm slide positions and their velocities read tilt at R^2 = 0.04 in
the pilot. Vision is excluded from BOTH sides: the rays rotate with the body,
so a tilted body sees tilted distances — a feature that smuggles orientation
in through the eyes would make the control unfailable. Needs are excluded
because they drain monotonically with time — feeding the probe a clock while
gating against a clock-reading control would be arming both sides of the same
test. The headline probe runs on the VESTIBULAR block itself, because the
hypothesis says "FROM WHICH": the claim is that the orientation signal carries
the fall, not that something in a wider bag of floats does (the pilot measured
the dilution directly: vestibular-alone 0.95 AUC, vestibular+touch 0.82,
all 27 floats 0.81 — a generic kernel probe pays for every uninformative
dimension it is handed). Touch is deleted ALONGSIDE the vestibular block in
the control and reported separately in the metrics.

## THE THREE WAYS THIS COULD BE FAKE, AND WHAT CATCHES EACH

1. **The probe reads the episode clock** — falls cluster at a characteristic
   time after spawn, so elapsed time alone predicts "topple soon". Caught
   twice, by design: the declared control deletes the balance suffix from the
   SAME rollouts (physics identical, decision for decision) and must fail; and
   the null predictor is the SAME probe machinery given elapsed time as its
   only feature — the headline must beat it by a pre-registered margin, not by
   rounding.

2. **The rig makes every episode identical** — if every fall shares one
   schedule (the zero-perturbation pilot measured exactly this: passive
   topple at ~10 decisions on most spawns), time-to-topple degenerates into
   the clock and the control could not have failed however honest the probe.
   So each episode draws a pre-registered LOG-uniform initial tilt (fall
   time goes as log(1/theta), so only a log draw spreads it) plus an
   angular-velocity kick, and respawns to a fresh site. Two statistics
   guard two different things and carry two names (v3): TF_FALL_SPREAD_MIN
   gates the spread of FALL times alone — the detector for THIS failure
   mode, fall dynamics degenerating onto one schedule — and
   TF_ABS_SPREAD_MIN gates the spread of ABSOLUTE topple times (hold +
   fall), which is what the clock null needs in order to be able to fail.
   Under v2's rig the absolute spread includes the rig's own uniform hold,
   so it can no longer see this failure mode — that discovery is v3's scar
   (see the V3 section). Rows are only eligible while upright cosine >=
   UPRIGHT_ROW — the claim is about feeling the fall EARLY, not about
   reading a body already at 45 degrees.

3. **The task is unscoreable and reports FAIL anyway** — too few topples, too
   few eligible rows, or a test set with one class. Those are rig failures,
   not refutations: they return Status.VOID (the T2.02 lesson — only a run
   that tested the claim may say FAIL).

## WHAT FAIL WOULD MEAN

FAIL is reserved for the sense failing: tilt unrecoverable by a linear probe,
the AUC no better than the clock, or the control passing (the "signal" was
never the balance channel). Then balance is not a sense he has, it is an
outcome he suffers — and every climbing and locomotion claim that assumes he
can tell up from down inherits that hole.

PILOT: seed 90, disjoint from the registered seeds 0/1/2 (PS.02, PG.6, SM.01
precedent). Gates were set with margin after the pilot; the pilot numbers are
pre-registered in docs/LOOP_JOURNAL.md under this spec before the recorded run.

## V2 (attempt 2, T1.02 precedent: strengthen only, v1's FAIL stays in history)

V1 recorded FAIL 2026-08-12T17:33 and the failure was the RIG'S, not clearly
the sense's: tilt_r2 0.9997 with the blind control at -0.08, but the
ELAPSED-TIME NULL scored 0.856 on the registered worlds (0.72 on the pilot
world) against the headline's 0.880 — margin 0.024 < the pre-registered 0.10.
tf_spread read 3.68 +/- 2.32 ACROSS seeds: world mutation changes spawn-site
statistics, and on some seeds every episode topples on nearly one schedule, so
the clock knew almost everything the vestibular channel knows. The spec's own
docstring names this rig failure mode (#2) and v1's rig only defended against
it with the tilt draw, which the spawn statistics of a mutated world can undo.

V2 removes the correlation by construction rather than by draw (the PS.02
order: derive the world from what the control must be able to detect — here,
what the clock null must NOT be able to see):

  HOLD-THEN-RELEASE. Each episode settles for T_SETTLE decisions, then pins
  the FULL root pose for a random hold of t_r ~ U{0..HOLD_MAX} decisions
  (arms keep moving under noise, so the blind block stays live), then applies
  the v1 tilt + a log-uniform-magnitude kick at release. Absolute topple time
  is now t_r + fall_time with t_r uniform, so episode-elapsed time carries
  almost no information about time-to-topple on ANY seed's world, whatever
  its spawn statistics. Rows before release are not scored on either side:
  before the kick exists, no sensor could know it — a label the sensors
  cannot know is censored, not asserted (the same principle as the survivor
  trim). Scored rows live in a uniform box — the first K_POST post-release
  decisions, at absolute time within [K_POST-1, HOLD_MAX] of the hold range —
  where t_r uniform makes P(y|t) flat by construction. Three pilot
  measurements forced the box and the two pins, in order: survivors running
  to the horizon made late time purely negative (P(y=1|t) 0.59 -> 0.00, raw
  t alone AUC 0.90); an orientation-only pin let arm noise DRIFT the body
  against structure, so survival rose 0.08 -> 0.70 with hold length and the
  clock read the outcome through the floor; and a fixed kick scale sent
  every episode down in ~7-12 decisions, starving the negative class. Each
  fix is documented at its site; no gate moved for any of them.

Two more v1 defects fixed, neither a threshold:

  THE SHUFFLE NULL WAS ONE DRAW, NOT A NULL. v1 used a single permutation from
  a FIXED seed, shared by all three spec seeds. A fixed permutation applied to
  similarly-ordered rows (same episode/time structure on similar worlds) is
  one correlated draw of a statistic with real variance — v1 measured
  0.063 +/- 0.018, consistently positive across seeds, the signature of that
  coincidence. v2 reports the MEAN over N_SHUF=8 permutations drawn from a
  seed-derived RNG. The gate is unchanged.

  RIG DEGENERACY IS VOID, PER SEED. v1 folded toppled_frac and tf_spread into
  the FAIL conjunction, contradicting its own docstring (#3: rig failures are
  VOID — only a run that tested the claim may say FAIL, the T2.02 lesson).
  v2 returns Status.VOID when any seed's world is degenerate (seed_rig_ok),
  which is STRICTER as a PASS bar than v1's aggregate-mean gates, and honest
  as a verdict: a world where every fall shares one schedule cannot test
  whether he feels falling. FAIL remains reserved for the sense failing.

All pre-registered thresholds are UNTOUCHED from v1. V2 pilot numbers (seed
90) are pre-registered in docs/LOOP_JOURNAL.md before the recorded run.

## V3 (attempt 3, T1.02 precedent: strengthen only; v2's PASS stays in history)

The overseer's 11th audit (RANK 1) found that v2 changed what TF_SPREAD_MIN
BOUNDS without moving its number. v2 redefined tf_spread from the spread of
FALL times to the spread of ABSOLUTE topple times, which include the rig's
own uniform hold t_r ~ U{0..40}: std(t_r) alone is 11.85 decisions against
the 2.5 gate, so a world with ZERO fall-time variance — failure mode #2 in
its purest form — would clear the gate 4.7x on the strength of the rig's own
RNG. The correct statistic, tf_fall_spread, was computed in the v2 diff and
left ungated. Law 4 protects the number; nothing protected the measurement.

V3 gates it: TF_FALL_SPREAD_MIN joins seed_rig_ok. Its value, 2.5, is the
value already in the file's history — v1's TF_SPREAD_MIN gated exactly this
quantity (no hold existed) and was set to 2.5 after the v1 pilot measured
fall-time spread at 5.69; v1's registered run read 3.68 +/- 2.32. It is NOT
chosen from v2's numbers. The absolute-spread statistic and its gate are
RENAMED (tf_abs_spread, TF_ABS_SPREAD_MIN; value byte-identical) so that one
name no longer carries two jobs: the absolute spread is what the clock null
needs to be able to fail; the fall spread is failure mode #2's detector. A
new gate on a previously ungated statistic is a strengthening — law 4
permits it — and it is pre-registered in docs/LOOP_JOURNAL.md with the
seed-90 pilot's tf_fall_spread beside it before the recorded run.

## V4 (attempt 4, T1.02 precedent: strengthen only; v3's VOID stays in history)

V3's new gate promptly took back v2's PASS — seed 2's world read
tf_fall_spread 1.49 — and the diagnosis (three measurements, journal
2026-08-12 ~20:45) says the gate was unreachable on OPEN GROUND by
construction, on every world: (1) the post-release fall-time bulk is
~7 +/- 1.5-2.2 decisions per world (q10-q90 = 4-10); v3's passing seeds
cleared 2.5 on 1-2 rare structure-outlier falls — tail lottery, and seed 2
just drew no outliers. (2) The slow side is FLOORED by the contact solver:
with zero arm noise and 0.1 deg tilt the body still falls in 9-10 decisions,
so open-ground fall times live in [3,10] and their std cannot exceed ~2.2.
(3) The v3 kick draw, independent of the tilt draw, erased the tilt spread —
a 0.6 rad/s kick is ~8.6 deg equivalent and overrides a 0.1 deg tilt.

V4 changes the RIG so fall dynamics genuinely spread; every gate is
byte-identical to v3's (nothing moved, nothing renamed):

  KICK IS TILT-PROPORTIONAL. |kick| = theta * KICK_OMEGA_P * 10^U[KICK_JIT],
  random direction, so the two-decade log-tilt spread survives into fall
  times instead of being overwritten by an independent kick.

  BOUNDARY SPAWNS. With probability P_STRUCT the episode spawns at a legal
  cell whose nearest ILLEGAL spawn-grid cell is within STRUCT_STEPS grid
  steps — "beside an obstacle", derived from the live model's own legal-spawn
  probe (the PG.8 reference-don't-transcribe rule; 42-65 such sites per
  world) — and the tilt's fall direction aims at that cell's bearing plus
  AIM_JITTER*randn. Falls then lean, slide and catch on world geometry,
  which is where v1's fall spread always came from.

  Rig health measured before registration at N_EP=120 on ALL FOUR worlds
  (rig-health quantities only — no sense gate was computed outside pilot
  seed 90): fall std 6.14-8.27, toppled 0.93-0.98, boundary sites 42-65.

  KNOWN RISK, gated by the UNMOVED control gates: struct episodes could leak
  structure proximity into the blind block (arm slides stalling on walls),
  which the control cap 0.70 / margin 0.15 exists to catch; the pilot must
  show the margin before the registered run.
"""
from __future__ import annotations

import math

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

from ..protocol import Ledger, Status, borrow_metrics, run_spec   # noqa: E402
from ..registry import BY_ID                                      # noqa: E402
from ..w0 import (W0, SIM_S_PER_DECISION,                         # noqa: E402
                  SPAWN_GRID, SPAWN_MARGIN)

# The claim is about the world's body and its senses, so it goes stale when
# either moves.
IMPL_DEPS = ["experiments/w0.py", "playground.py"]

# ── the rollouts ────────────────────────────────────────────────────────
# The rig was shaped by three pilot measurements, not by intention (the PS.02
# order: the control shaped the world). (1) A uniform tilt draw gave median
# topple at 5 decisions with std 1.5 — pendulum fall time goes as log(1/theta),
# so uniform theta is a clock and the AUC was unscoreable (7 negative rows).
# (2) The tilt floor is structural: from EXACT upright with zero velocity the
# contact solver injects ~0.8 deg within one decision, so every free-standing
# spawn falls by ~13 decisions regardless of how small the draw is. Fall times
# are therefore spread by drawing theta LOG-uniformly above that floor, and by
# a strong angular kick whose direction matters (a kick against the tilt swings
# the body through upright and buys it decisions — an energy state the
# vestibular channel reads and the clock cannot). (3) Spawn sites vary: some
# poses lean on structure and never topple, which is where the honest
# late-time negative rows come from — so every episode respawns.
N_EP_TRAIN = 84              # episodes the probes fit on
N_EP_TEST = 36               # episodes they are scored on — held out by EPISODE
HORIZON = 80                 # decisions = 16 s; worst case hold 40 + fall ~15
HOLD_MAX = 40                # v2: t_r ~ U{0..40} decisions of pinned-in-place
T_SETTLE = 3                 # v2: settle decisions before the hold pose is set
TILT0_LOG10_DEG = (-1.0, 1.15)   # theta ~ 10^U[...]: 0.1 to 14 deg
# V4: the kick is TILT-PROPORTIONAL. v2's independent log-uniform kick draw
# (10^U[-2.0,-0.22] rad/s) kept the negative class alive but ERASED the tilt
# spread it was meant to complement — a 0.6 rad/s kick is ~8.6 deg
# equivalent and overrides a 0.1 deg tilt, so fall times collapsed onto the
# kick's own schedule. |kick| = theta * KICK_OMEGA_P * 10^U[KICK_JIT],
# random direction: the perturbation ENERGY now spans the tilt draw's two
# decades and the jitter varies it another ~1.2 decades around that.
KICK_OMEGA_P = 3.5               # 1/s: kick rad/s per rad of drawn tilt
KICK_JIT = (-0.7, 0.5)           # log10 jitter around the proportional kick
# V4: boundary spawns. Fall-time spread on open ground is capped at ~2.2
# decisions by the contact solver's own floor (measured: 0.1 deg tilt at
# zero arm noise still falls in 9-10 decisions), so the fall-dynamics
# spread the tf_fall_spread gate demands must come from world GEOMETRY —
# falls that lean, slide and catch on structure. Sites are derived from the
# live model's legal-spawn grid, never written down (PG.8).
P_STRUCT = 0.65                  # P(episode spawns beside structure)
STRUCT_STEPS = 1.5               # grid steps to nearest illegal cell, max
AIM_JITTER = 0.4                 # rad, randn jitter on the aimed bearing
ARM_NOISE = 0.3              # slide actions ~ U[-1,1] * this; drive+adhesion 0
GRAVITY = 9.81
N_SHUF = 8                   # v2: permutations averaged for the shuffle null

# ── labels ──────────────────────────────────────────────────────────────
TOPPLE_UP = 0.5              # upright cosine below this = toppled (60 deg)
UPRIGHT_ROW = 0.9            # rows eligible for the AUC while above this
W_WARN = 5                   # decisions: "topples within the next 1.0 s"
# V2: scored rows live in the first K_POST decisions after release, at
# absolute episode time in [K_POST-1, HOLD_MAX]. Outside that box the class
# composition is position-biased by construction — survivors run to the
# horizon (the pilot measured P(y=1|t) falling 0.59 -> 0.00 with t, raw t
# alone scoring AUC 0.90), and early absolute times over-represent
# early-window rows. Inside it, t_r uniform makes the within-window position
# uniform at every absolute t, so P(y|t) is flat BY CONSTRUCTION and the
# clock null (and any diffusion clock in the blind block) has nothing to
# read. Rows outside the box are censored, not asserted.
K_POST = 12                  # scored post-release window, decisions

# ── the probes ──────────────────────────────────────────────────────────
N_RFF = 300                  # PS.02's generic probe, one fixed draw
RFF_SEED = 20260812
RIDGE_LAMBDA = 1.0
VEST_DIM = 11                # grav_body 3 + canals 3 + otoliths 3 + vx,vy
GRAV_DIM = 8 + VEST_DIM      # the orientation channel: touch + vestibular

# ── pre-registered gates (set with margin after the seed-90 pilot;
#    pilot numbers recorded beside each gate and in LOOP_JOURNAL.md) ─────
TOPPLED_FRAC_MIN = 0.60      # a world with nothing falling tests nothing (VOID)
# V3: two spreads, two jobs, two names (11th audit, RANK 1 — see docstring).
TF_ABS_SPREAD_MIN = 2.5      # decisions, std of ABSOLUTE topple times (hold +
                             # fall): the clock null must be able to fail.
                             # Renamed from TF_SPREAD_MIN, value untouched.
TF_FALL_SPREAD_MIN = 2.5     # decisions, std of FALL times alone: failure mode
                             # #2's detector. v1 gated this quantity at 2.5
                             # (pilot 5.69, registered 3.68 +/- 2.32); v2 left
                             # it ungated; v3 restores the gate at v1's value.
MIN_CLASS_ROWS = 25          # test rows per class, else unscoreable (VOID)
TILT_R2_MIN = 0.90           # linear probe recovers tilt-cosine (pilot 0.998)
TILT_SHUF_R2_MAX = 0.05      # shuffled pairing must collapse (pilot < 0)
TILT_CONTROL_R2_MAX = 0.30   # arm slides must not recover tilt (pilot 0.04)
AUC_MIN = 0.85               # the headline (pilot 0.95)
AUC_TIME_MARGIN_MIN = 0.10   # headline minus elapsed-time null (pilot ~0.23)
CONTROL_AUC_MAX = 0.70       # blind AUC cap (pilot 0.64)
CONTROL_MARGIN_MIN = 0.15    # headline minus blind AUC (pilot ~0.31)

_CACHE: dict = {}


def _calibration() -> tuple:
    """PS.01's j0/alpha, or a refusal. W0 has no defaults for them."""
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if not b.ok:
        return None, None, {**b.provenance, "borrow_refusal": b.refusal}
    return b.values["j0_ms"], b.values["alpha"], b.provenance


def _tilt_quat(theta: float, lean_bearing: float) -> np.ndarray:
    """World-frame tilt of `theta` rad whose LEAN points at `lean_bearing`.

    A rotation about the horizontal axis (cos phi, sin phi, 0) leans the
    body's up-axis toward (sin phi, -cos phi): Rodrigues with v = z_hat and
    u.v = 0 gives v' = z_hat cos(theta) + (u x z_hat) sin(theta), and
    u x z_hat = (sin phi, -cos phi, 0). So aiming the lean at bearing b
    means phi = b + pi/2. Verified numerically in the v4 pilot rig.
    """
    phi = lean_bearing + math.pi / 2.0
    s = math.sin(theta / 2.0)
    return np.array([math.cos(theta / 2.0),
                     math.cos(phi) * s, math.sin(phi) * s, 0.0])


def _spawn_grid(w: W0) -> tuple:
    """(legal, illegal, step) on the world's own spawn grid — the one
    derivation both the boundary sites and the T0.26 open site share."""
    a = float(w.params.arena_size) - SPAWN_MARGIN
    axis = np.linspace(-a, a, SPAWN_GRID)
    step = float(axis[1] - axis[0])
    legal = w.legal_spawns()
    legal_set = {(round(float(x), 9), round(float(y), 9)) for x, y in legal}
    illegal = np.array([(x, y) for x in axis for y in axis
                        if (round(float(x), 9), round(float(y), 9))
                        not in legal_set])
    return legal, illegal, step


def _boundary_sites(w: W0) -> np.ndarray:
    """(S, 3) rows of (x, y, bearing-to-structure), derived from the model.

    A boundary site is a LEGAL spawn cell whose nearest ILLEGAL cell on the
    same spawn grid lies within STRUCT_STEPS grid steps — "beside an
    obstacle" as the world's own legal-spawn probe defines obstacle, never a
    hand-written list (the PG.8 rule: when you can reference, reference).
    The bearing points at that nearest illegal cell, so an aimed tilt sends
    the fall INTO the geometry rather than merely near it.
    """
    legal, illegal, step = _spawn_grid(w)
    if len(illegal) == 0:
        return np.zeros((0, 3))
    sites = []
    for x, y in legal:
        d = np.hypot(illegal[:, 0] - x, illegal[:, 1] - y)
        k = int(np.argmin(d))
        if d[k] <= STRUCT_STEPS * step:
            b = math.atan2(illegal[k, 1] - y, illegal[k, 0] - x)
            sites.append((float(x), float(y), b))
    return np.asarray(sites, dtype=np.float64)


def _open_site(w: W0) -> np.ndarray:
    """The anti-boundary site: the legal cell FARTHEST from any illegal
    cell, so world geometry cannot vary a fall started there. Derived from
    the same grid as `_boundary_sites`, for T0.26's degenerate rig."""
    legal, illegal, _step = _spawn_grid(w)
    if len(illegal) == 0:
        x, y = legal[0]
        return np.asarray([[float(x), float(y), 0.0]])
    dmin = [float(np.min(np.hypot(illegal[:, 0] - x, illegal[:, 1] - y)))
            for x, y in legal]
    k = int(np.argmax(dmin))
    return np.asarray([[float(legal[k][0]), float(legal[k][1]), 0.0]])


def _episode(w: W0, rng: np.random.RandomState,
             sites: np.ndarray) -> dict:
    """One hold-then-release episode: respawn, hold upright, release, fall.

    Actions move ONLY the four arm slides. The drive dims are a world-frame
    force — a hidden push knob the blind features could read through its
    consequences — and adhesion glues hands; neither belongs in a passive
    balance measurement. Arm noise is kept so the blind block is a live signal
    the control probe genuinely gets to try: a control fed constants cannot
    fail meaningfully (PS.02: a control that cannot fail is not a control —
    the mirror rule: a control that cannot PASS proves nothing either).

    V2: for t_r ~ U{0..HOLD_MAX} decisions the root's ORIENTATION is pinned
    upright after every step (the structural ~0.8 deg/decision contact-solver
    tilt is undone; linear dofs stay free so the body settles honestly, arms
    keep diffusing). At release the v1 tilt + kick is applied. Rows are
    recorded only from release — before the kick exists no sensor could know
    it, so a pre-release label would be noise on both sides of the test.
    """
    mujoco = w.mujoco
    # V4: boundary spawn with an aimed fall, else uniform legal with a
    # uniform fall direction. The struct flag is reported, never gated.
    struct = bool(len(sites) > 0 and rng.rand() < P_STRUCT)
    if struct:
        k = int(rng.randint(len(sites)))
        w.respawn(at=(sites[k][0], sites[k][1]))
        aim = float(sites[k][2]) + float(rng.randn()) * AIM_JITTER
    else:
        w.respawn()
        aim = float(rng.uniform(0.0, 2.0 * math.pi))
    qa, da = w.ix["root_qposadr"], w.ix["root_dofadr"]
    t_r = int(rng.randint(0, HOLD_MAX + 1))
    # Settle: let the spawn's 1 cm drop dissipate with only orientation
    # pinned, so the held pose is a real resting state, not a hover.
    for _ in range(T_SETTLE):
        act = np.zeros(8)
        act[:4] = rng.uniform(-1.0, 1.0, 4) * ARM_NOISE
        w.decide(act)
        w.data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        w.data.qvel[da + 3:da + 6] = 0.0
        mujoco.mj_forward(w.model, w.data)
    # Hold: the FULL root pose is pinned — held in place, arms still moving.
    # The first pilot pinned orientation only, and the body drifted with the
    # arm noise: longer holds slid it against structure it could then lean
    # on, so survival rose 0.08 -> 0.70 across the t_r range and the episode
    # clock legitimately predicted the outcome. A hold that does not hold
    # position re-couples the clock to the world through the floor.
    q_hold = w.data.qpos[qa:qa + 7].copy()
    for _ in range(t_r):
        act = np.zeros(8)
        act[:4] = rng.uniform(-1.0, 1.0, 4) * ARM_NOISE
        w.decide(act)
        w.data.qpos[qa:qa + 7] = q_hold
        w.data.qvel[da:da + 6] = 0.0
        mujoco.mj_forward(w.model, w.data)
    # Release: tilt the root about the world frame (lean aimed per the spawn),
    # kick its angular velocity in proportion to the drawn tilt (V4).
    theta = math.radians(10.0 ** rng.uniform(*TILT0_LOG10_DEG))
    q0 = w.data.qpos[qa + 3:qa + 7].copy()
    qt = _tilt_quat(theta, aim)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qt, q0)
    w.data.qpos[qa + 3:qa + 7] = out
    w.data.qvel[da:da + 6] = 0.0
    mag = theta * KICK_OMEGA_P * 10.0 ** rng.uniform(*KICK_JIT)
    u = rng.randn(3)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    w.data.qvel[da + 3:da + 6] = u * mag
    mujoco.mj_forward(w.model, w.data)

    rows, uprights = [], []
    v_prev = w.data.qvel[da:da + 3].copy()
    t_f = None
    for t in range(HORIZON - T_SETTLE - t_r):
        xmat = w.data.xmat[w.rover_bid]
        R = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
        up = float(xmat[8])
        grav_body = -R.T @ np.array([0.0, 0.0, 1.0])       # gravity direction
        canals = w.data.qvel[da + 3:da + 6].copy()          # ang vel, body frame
        v = w.data.qvel[da:da + 3].copy()
        a_world = (v - v_prev) / SIM_S_PER_DECISION
        otoliths = R.T @ (a_world - np.array([0.0, 0.0, -GRAVITY]))
        v_prev = v
        p = w._proprio()          # 4 slide pos, 4 slide vel, z, upright, vx, vy
        touch = w._touch()
        blind = np.concatenate([p[:8], touch])
        balance = np.concatenate([grav_body, canals, otoliths, p[10:12]])
        assert balance.shape[0] == VEST_DIM
        rows.append(np.concatenate([blind, balance]))
        uprights.append(up)
        if up < TOPPLE_UP:
            t_f = t
            break
        act = np.zeros(8)
        act[:4] = rng.uniform(-1.0, 1.0, 4) * ARM_NOISE
        w.decide(act)
    return {"X": np.asarray(rows, dtype=np.float64),
            "upright": np.asarray(uprights, dtype=np.float64),
            "t_f": t_f, "t_r": t_r, "struct": struct}


def _collect(seed: int) -> dict:
    """Every simulation this spec needs, once. Cached: the control reuses it.

    The control is the SAME rollouts observed without the balance suffix —
    re-simulating would risk the two arms differing by something other than
    the sense, which is the only difference the spec is allowed to have.
    """
    if seed in _CACHE:
        return _CACHE[seed]
    j0, alpha, prov = _calibration()
    if j0 is None:
        _CACHE[seed] = {"refused": prov}
        return _CACHE[seed]
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=False)
    sites = _boundary_sites(w)          # V4: derived once per seed, cached
    rng = np.random.RandomState(seed * 7907 + 11)
    eps = [_episode(w, rng, sites) for _ in range(N_EP_TRAIN + N_EP_TEST)]
    _CACHE[seed] = {"eps": eps, "prov": prov,
                    "n_boundary_sites": float(len(sites))}
    return _CACHE[seed]


# ── probes: linear ridge (tilt), RFF ridge (topple), one fixed draw ─────
def _ridge_predict(Xtr, ytr, Xte, rff: bool) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if rff:
        rng = np.random.RandomState(RFF_SEED)
        d = Ztr.shape[1]
        W = rng.randn(d, N_RFF) / math.sqrt(d)
        b = rng.uniform(0.0, 2.0 * math.pi, N_RFF)
        Ztr = np.cos(Ztr @ W + b) * math.sqrt(2.0 / N_RFF)
        Zte = np.cos(Zte @ W + b) * math.sqrt(2.0 / N_RFF)
    ybar = ytr.mean()
    A = Ztr.T @ Ztr + RIDGE_LAMBDA * np.eye(Ztr.shape[1])
    beta = np.linalg.solve(A, Ztr.T @ (ytr - ybar))
    return Zte @ beta + ybar


def _r2(y, yhat) -> float:
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / max(sst, 1e-12)


def _auc(y, score) -> float:
    """Mann–Whitney rank AUC; nan when the test set has one class."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # midranks for ties
    allsc = np.concatenate([neg, pos])
    for v in np.unique(allsc):
        m = allsc == v
        ranks[m] = ranks[m].mean()
    r_pos = ranks[len(neg):].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _label_rows(ep: dict) -> tuple:
    """Eligible rows (still upright) with the 'topples within W_WARN' label.

    A surviving episode's last W_WARN rows are trimmed: they could belong to a
    topple just past the horizon, and a label that might be wrong is censored,
    not asserted (PS.02's nan-target lesson, one step earlier).
    """
    up = ep["upright"]
    n = len(up)
    t_f = ep["t_f"]
    last = n if t_f is not None else n - W_WARN
    last = min(last, K_POST)                      # V2: the scored window
    rows, ys, ts = [], [], []
    for t in range(max(last, 0)):
        t_abs = ep["t_r"] + t
        if not (K_POST - 1 <= t_abs <= HOLD_MAX):  # V2: the uniform box
            continue
        if up[t] < UPRIGHT_ROW:
            continue
        y = 1.0 if (t_f is not None and t_f - t <= W_WARN) else 0.0
        rows.append(ep["X"][t])
        ys.append(y)
        # The clock null reads ABSOLUTE episode time — settle and hold
        # included — because that is the clock a fake probe would be reading.
        ts.append((T_SETTLE + t_abs) * SIM_S_PER_DECISION)
    return rows, ys, ts


def _stack(eps: list) -> tuple:
    X, y, t = [], [], []
    for ep in eps:
        r, yy, tt = _label_rows(ep)
        X.extend(r); y.extend(yy); t.extend(tt)
    return (np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(t, dtype=np.float64)[:, None])


def _tilt_sets(eps: list) -> tuple:
    """All not-yet-toppled rows with the tilt COSINE as the target.

    The hypothesis asks a LINEAR probe to recover tilt, and the upright cosine
    is tilt in monotone units — the same information, bijectively. The angle
    itself goes as sqrt(1 - u) exactly where the rows cluster (near upright),
    so a linear probe of the ANGLE fails for the reader's reasons rather than
    the sensor's: the seed-90 pilot measured R^2 0.195 on the angle and 0.998
    on the cosine from the same features. Chosen after the pilot, disclosed
    here and in the journal pre-registration.
    """
    X, y = [], []
    for ep in eps:
        X.extend(ep["X"])
        y.extend(ep["upright"])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def rig_health(eps: list) -> dict:
    """The rig-health statistics and their per-seed gate, in ONE place.

    tf_abs_spread is the spread of ABSOLUTE topple times (hold + fall): the
    quantity that must be wide for the clock null to be able to fail. Under
    this rig it includes the hold's own uniform t_r, so it says nothing
    about fall dynamics. tf_fall_spread is the spread of FALL times alone —
    the detector for every episode toppling on one schedule (failure mode
    #2) — and v3 gates it in `rig_ok`.

    Extracted from `_evaluate` so T0.26 can drive the REAL statistic path on
    fixture rigs (a tidied restatement would pass while the shipped
    computation drifted — the T0.16 lesson).
    """
    t_fs = [T_SETTLE + ep["t_r"] + ep["t_f"]
            for ep in eps if ep["t_f"] is not None]
    falls = [ep["t_f"] for ep in eps if ep["t_f"] is not None]
    toppled_frac = len(t_fs) / len(eps)
    tf_abs_spread = float(np.std(t_fs)) if t_fs else 0.0
    tf_fall_spread = float(np.std(falls)) if falls else 0.0
    return {"toppled_frac": toppled_frac, "tf_abs_spread": tf_abs_spread,
            "tf_fall_spread": tf_fall_spread,
            "median_t_f": float(np.median(t_fs)) if t_fs else float("nan"),
            "rig_ok": 1.0 if (toppled_frac >= TOPPLED_FRAC_MIN
                              and tf_abs_spread >= TF_ABS_SPREAD_MIN
                              and tf_fall_spread >= TF_FALL_SPREAD_MIN)
            else 0.0}


def rollout_rig(world_seed: int, n_ep: int, degenerate: bool) -> list | None:
    """Episodes from the real rig, or from this spec's DECLARED degenerate rig.

    The degenerate rig is docstring failure mode #2 made executable — every
    episode topples on ONE schedule: a single fixed tilt (10^0.8 = 6.3 deg),
    zero kick, zero arm noise, zero aim jitter, every spawn at the model's
    own most-open cell (`_open_site`, where geometry cannot catch a fall).
    Only the hold t_r still varies, so ABSOLUTE topple times stay wide while
    fall dynamics carry no variance at all — precisely the world v2's
    abs-spread gate wrongly certified. (A first fixture that kept arm noise
    and uniform spawns measured tf_fall_spread 3.51 — uniform legal spawns
    land beside structure often enough to buy outlier falls, the v3 tail
    lottery in miniature — which is why the fixture pins BOTH.)

    T0.26 drives both branches through the same `_episode` the recorded
    runs use and asserts the rig-health gate in both directions: the broken
    world must score BELOW TF_FALL_SPREAD_MIN, the honest rig's bulk ABOVE
    it (reachability and inertness are two directions of one assertion —
    LESSONS, 2026-08-12). The degeneracy is declared HERE, in the artifact,
    never invented by the auditor (the LC.01 lesson). BA.01's own recorded
    runs never call this. Returns None on a PS.01 borrow refusal.
    """
    global TILT0_LOG10_DEG, KICK_JIT, KICK_OMEGA_P, ARM_NOISE, \
        P_STRUCT, AIM_JITTER
    j0, alpha, _prov = _calibration()
    if j0 is None:
        return None
    w = W0(seed=world_seed, j0=j0, alpha=alpha, lethal=False)
    rng = np.random.RandomState(world_seed * 104729 + 13)
    saved = (TILT0_LOG10_DEG, KICK_JIT, KICK_OMEGA_P, ARM_NOISE,
             P_STRUCT, AIM_JITTER)
    try:
        if degenerate:
            TILT0_LOG10_DEG, KICK_JIT, KICK_OMEGA_P = (0.8, 0.8), (0.0, 0.0), 0.0
            ARM_NOISE, P_STRUCT, AIM_JITTER = 0.0, 1.0, 0.0
            sites = _open_site(w)
        else:
            sites = _boundary_sites(w)
        return [_episode(w, rng, sites) for _ in range(n_ep)]
    finally:
        (TILT0_LOG10_DEG, KICK_JIT, KICK_OMEGA_P, ARM_NOISE,
         P_STRUCT, AIM_JITTER) = saved


def _evaluate(seed: int, blind: bool) -> dict:
    """The headline probe reads the graviceptive suffix; the control reads
    the blind prefix. Same rollouts, one slice apart — the only difference
    the spec allows."""
    c = _collect(seed)
    if "refused" in c:
        return {"probe": "VOID", **{k: float("nan") for k in ("auc",)},
                **c["refused"]}
    eps = c["eps"]
    tr, te = eps[:N_EP_TRAIN], eps[N_EP_TRAIN:]
    sl_x = slice(None, -GRAV_DIM) if blind else slice(-VEST_DIM, None)

    rig = rig_health(eps)

    Xtr, ytr, ttr = _stack(tr)
    Xte, yte, tte = _stack(te)
    n_pos, n_neg = int(yte.sum()), int((1 - yte).sum())

    out = {"toppled_frac": rig["toppled_frac"],
           "tf_abs_spread": rig["tf_abs_spread"],
           "tf_fall_spread": rig["tf_fall_spread"],
           "median_t_f": rig["median_t_f"],
           "n_rows_train": float(len(ytr)), "n_pos_test": float(n_pos),
           "n_neg_test": float(n_neg),
           # V4 rig descriptors, reported never gated: how much of the fall
           # spread rides on geometry, and whether this world offers any.
           "n_boundary_sites": c["n_boundary_sites"],
           "struct_frac": float(np.mean([ep["struct"] for ep in eps]))}
    if n_pos < MIN_CLASS_ROWS or n_neg < MIN_CLASS_ROWS or len(ytr) < 100:
        out["probe"] = "VOID"
        out["auc"] = float("nan")
        return out

    out["auc"] = _auc(yte, _ridge_predict(Xtr[:, sl_x], ytr, Xte[:, sl_x],
                                          rff=True))

    # Tilt: the linear probe of the hypothesis, on all not-toppled rows.
    Ttr_X, Ttr_y = _tilt_sets(tr)
    Tte_X, Tte_y = _tilt_sets(te)
    out["tilt_r2"] = _r2(Tte_y, _ridge_predict(Ttr_X[:, sl_x], Ttr_y,
                                               Tte_X[:, sl_x], rff=False))

    if not blind:
        # The elapsed-time null: the same machinery, the clock as its only
        # feature. What survivorship makes predictable, it gets for free.
        out["auc_time"] = _auc(yte, _ridge_predict(ttr, ytr, tte, rff=True))
        # Shuffled tilt pairing (chance for tilt, the registry's null).
        # V2: the MEAN over N_SHUF permutations from a seed-derived RNG. One
        # fixed permutation shared across seeds is a single correlated draw
        # of a statistic with real variance, not the null's value — v1
        # measured that draw at 0.063 +/- 0.018, consistently positive.
        prng = np.random.RandomState(RFF_SEED + 1 + 7919 * seed)
        shufs = []
        for _ in range(N_SHUF):
            sh = prng.permutation(len(Ttr_y))
            shufs.append(_r2(
                Tte_y, _ridge_predict(Ttr_X[:, sl_x], Ttr_y[sh],
                                      Tte_X[:, sl_x], rff=False)))
        out["tilt_r2_shuffled"] = float(np.mean(shufs))
        out["tilt_r2_shuffled_spread"] = float(np.std(shufs))
        # The organs of the graviceptive block, reported separately as the
        # registry demands — a system given only gravity's direction cannot
        # tell falling from being carried, and touch must not hide the
        # vestibular numbers (or vice versa). Reported, not gated.
        nb = Xtr.shape[1] - GRAV_DIM
        for name, sl in (("touch", slice(nb, nb + 8)),
                         ("vest", slice(nb + 8, None)),
                         ("grav", slice(nb + 8, nb + 11)),
                         ("canals", slice(nb + 11, nb + 14)),
                         ("otoliths", slice(nb + 14, nb + 17))):
            out[f"auc_{name}"] = _auc(
                yte, _ridge_predict(Xtr[:, sl], ytr, Xte[:, sl], rff=True))
        # V2 split: rig health (the world could test the claim) is separate
        # from the sense gates (the claim held), because they carry different
        # verdicts — a degenerate rig is VOID, a failed sense is FAIL.
        # V3 adds the fall-spread gate: a world whose falls all share one
        # schedule could not have tested the claim, however wide the hold
        # makes the absolute spread. The conjunction lives in `rig_health`,
        # where T0.26 gates it in both directions.
        out["seed_rig_ok"] = rig["rig_ok"]
        gates = (out["auc"] >= AUC_MIN
                 and out["auc"] - out["auc_time"] >= AUC_TIME_MARGIN_MIN
                 and out["tilt_r2"] >= TILT_R2_MIN)
        out["seed_gates_ok"] = 1.0 if gates else 0.0
    return out


def _experiment(seed: int) -> dict:
    return _evaluate(seed, blind=False)


def _control(seed: int) -> dict:
    """The declared control: the balance suffix deleted, physics identical."""
    return _evaluate(seed, blind=True)


def _declared_void(m: dict) -> bool:
    return m.get("probe") == "VOID" or not np.isfinite(m.get("auc", np.nan))


def _check(m: dict, c: dict):
    if _declared_void(m) or _declared_void(c):
        return Status.VOID
    # Rig degeneracy is VOID, per seed: a world where nothing topples, or
    # where every topple shares one schedule, could not have tested the claim
    # (docstring #3; the T2.02 lesson — only a run that tested it may FAIL).
    # seed_rig_ok is the per-seed conjunction, so its mean is 1.0 only when
    # EVERY seed's world was healthy — stricter than v1's aggregate means.
    if m["seed_rig_ok"] < 1.0:
        return Status.VOID
    ok = (m["seed_gates_ok"] == 1.0
          and m["tilt_r2"] >= TILT_R2_MIN
          and m["tilt_r2_shuffled"] <= TILT_SHUF_R2_MAX
          and m["auc"] >= AUC_MIN
          and m["auc"] - m["auc_time"] >= AUC_TIME_MARGIN_MIN
          # the control must FAIL, and by a margin, on both probes
          and c["tilt_r2"] <= TILT_CONTROL_R2_MAX
          and c["auc"] <= CONTROL_AUC_MAX
          and m["auc"] - c["auc"] >= CONTROL_MARGIN_MIN)
    return Status.PASS if ok else Status.FAIL


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["BA.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    print(run().status)
