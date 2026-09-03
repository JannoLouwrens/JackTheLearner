"""SO.02 — "I'm cold" is TRUE when he is cold.

GOAL.md gives Jack VOICE ("he must be able to make sound, not only receive
it") and NEEDS ("the needs ARE the curriculum"), and this spec is where the
two meet: when Jack's body is in trouble, the sound he makes must MEAN the
trouble — a listener with nothing but its ears must be able to tell WHICH need
is urgent and roughly HOW urgent, and Jack must shut up when nothing is wrong.
"Kills: language grounded in state, as distinct from language that
pattern-matches a situation."

WHAT THE SPEAKER IS, stated honestly up front. The utterance policy here is an
INNATE CRY REFLEX, designed, not learned — an infant's cry is not learned
either, and GOAL.md's biology shelf names "innate reflex priors" as
sanctioned. Nothing in this file claims the mapping was acquired; VO.02
already proved a learned sound-to-meaning mapping on this same channel. What
THIS spec measures, and what could genuinely fail, is the composed system's
TRUTHFULNESS end to end:

    certified need dynamics  ->  reflex  ->  4-D voice  ->  physics
        (thermal.py law,          (this      (ContactAudio:  (1/r, pan law,
         drives.py law)           file)       VO.01's rig)    calibrated room)
                                     ->  the ear  ->  a listener that decodes

Every stage past the reflex is a real acoustic channel: the emitter stands at
a RANDOM bearing and range (1.0-4.5 m) on each utterance so loudness can never
identify the call, the room carries real contact audio calibrated to VO.01's
declared difficulty (SIR_TARGET_DB over the voice's own level), and the
listener's entire input is `vo_01._features` of the received stereo. Whether
channel identity, urgency level, and the informative SILENCE of a content body
survive that channel is a measurement, not a construction.

THE NEEDS ARE THE CERTIFIED ONES, integrated by the certified laws. COLD is
body temperature under `thermal.py`'s pre-registered linear law
(dTb/dt = G_RATE * (T_eff - T_NEUTRAL); PS.02's certificate), driven by a
seeded ambient schedule of warm and cold segments — with a guard that forces a
warm segment when Tb approaches TB_LETHAL, because this is a talking spec, not
a dying one (the abstraction is "he reaches the fire"; LF.01 owns dying).
HUNGER is energy under `drives.py`'s basal drain (BASAL_B; PS.01's
calibration), with seeded eat events (NU_APPLE) when it runs low. Urgency:

    u_cold   = clip((TB_SHIVER - Tb) / (TB_SHIVER - TB_LETHAL), 0, 1)
    u_hunger = clip((E_CRY - e) / E_CRY, 0, 1)

TB_SHIVER is thermal.py's own "below this the cold is unmistakable in the
sense" — the threshold is the substrate's, not this file's. E_CRY is this
file's design constant, declared here.

THE REFLEX. At each decision moment (DT_S apart, N_MOMENTS of them): silent
when both urgencies are zero; otherwise cry with the urgent channel's FORM —
f0 and brightness are the channel cue (hunger high/bright, cold low/dark, ±
seeded jitter), DURATION encodes the urgency level, and AMPLITUDE IS FIXED,
deliberately: range is randomized per utterance, so a level code in loudness
would be destroyed by the channel and a listener reading loudness would be
reading distance. When both channels are urgent the louder need (argmax) is
voiced; `both_frac` is reported.

THE LISTENER. Features from every moment's received stereo — voiced moments
AND silent ones (the room alone), because "he is fine" is carried by the
ABSENCE of a cry and a listener that cannot read silence cannot use this
channel. Even moments train, odd moments test; every render is an independent
pose and room, so the split shares only the class-conditional feature
distribution, which is exactly what is under test. Three readouts, trained on
the train split only:

    class   ridge to one-hot {nominal, hungry, cold}, argmax  -> acc_class
    level   ridge to urgency, on voiced moments               -> r2_level
    MI      k-means codes (fit on train) vs class, plug-in MI with a
            permutation floor (VO.02's estimator, verbatim)   -> mi_ear

THE FLUENCY NULL (registry): "utterances sampled from the marginal
distribution at the same rate." Implemented as the exact null of the pairing:
the same rendered moments, the same firing rate, the same form marginal, with
the moment->render assignment PERMUTED before training. Every marginal
statistic survives; only the state->sound join is destroyed. Scored by the
same predicate as the claim (`_claim` does not know which arm it reads) and
must fail.

THE TWO REGISTRY CONTROLS, both in `_control`, both must land red:

  FREEZE. Needs clamped nominal for the whole life, same seeded poses and
  rooms, and he is MADE to talk at exactly the claim arm's firing moments with
  forms permuted among them. The claim-trained listener then decodes these
  renders, scored against the COUNTERFACTUAL labels (what the schedule would
  have said at those moments). If the voice is the carrier, this collapses to
  chance; if it stays up, the listener was reading pose, timing or room — the
  test was measuring nothing (law 2) and the run FAILs.

  SWAP. Same life, REAL needs, but "cold" is emitted for hunger and vice
  versa. The claim-trained listener MUST BE MISLED — decode the swapped
  channel, not the true one — "proving the channel is read and not merely
  present" (registry). A listener that still decodes the truth was never
  reading the sound.

THE HONESTY GATES. `fire_given_nominal` and `fire_given_urgent` are near-
invariants of the reflex as designed and are gated as composition checks (if
the urgency arithmetic or the schedule broke, they fire). The registry's named
falsifier — "correlated but ALWAYS firing (a thermostat that is always on is
not communicating)" — is `fire_frac`, gated on both sides: he must talk, and
he must not talk all the time. That number depends on the certified dynamics
meeting the schedule, not on the reflex alone.

VOID LANES, never FAIL (a rig that could not pose the question refutes
nothing): a class the life never visited (occupancy), urgency without spread
(a level readout with nothing to regress), a room at the wrong declared
difficulty (VO.01's two-sided SIR gate, re-measured over THIS rig's cry-form
marginal — quieting the room fails as loudly as drowning the voice), a dead
MI estimator (planted known-answer self-test), or non-finite features.

WHAT THIS SPEC DOES NOT CLAIM. Not that the cry was learned (it is declared
innate). Not words, not syntax — two need channels and a level is a cry, not
a sentence. Not interoception's dynamics (PS.01/PS.02 own those
certificates). And per the registry note, nothing here touches the diary:
this is the one legitimately generative channel because the claim is checked
against a live variable; speech about the PAST still quotes the diary
(ME.9's law, untouched).

PILOT PROTOCOL (SM.02/VO.02 idiom): `_GATES_FROZEN` starts False and `run()`
refuses. The pilot runs the full pipeline on seed 90 (never a claim seed);
bars marked PROVISIONAL below are frozen from its NULL and CONTROL readings
and from theory, never from the claim arm's numbers; the whole pilot is
disclosed in this docstring at freezing time.

PILOT RECORD — 2026-09-04, seed 90, full envelope, 6.8 s wall, artifact
/data/so02_pilot_seed90_v3.json, disclosed in full (claim arm included, so an
auditor can see what was on the screen when the bars froze):

    claim   acc_class 1.000  base_rate 0.422  mi_ear 1.553  floor_p95 0.097
            r2_level 0.870   fire_frac 0.578  fire|nominal 0.0  fire|urgent 1.0
            occ nom/hun/cold 0.422/0.244/0.333  u_std 0.221  both_frac 0.122
    null    acc 0.350 (base 0.422)  mi 0.056 vs floor 0.101  r2 -0.59  -> dead
    freeze  acc 0.500 vs chance 0.567  mi 0.046 vs floor 0.073        -> dead
    swap    misled 1.000  true-acc 0.000                       -> fully misled
    rig     SIR 6.09 dB vs declared 6.0 +/- 2   mi selftest 1.55 bits
            feats finite 1.0

Two earlier pilot iterations on the same seed found and fixed FIXTURE faults
before any gate froze, both disclosed: (1) the schedule's segment clock never
decremented, so one cold snap ran until the lethal guard fired and the life
stayed warm forever (occ_cold 0.072, byte-identical across two different
schedule parameterisations — the constancy was the tell); (2) CRY_AMP 0.5 put
the cry marginal at 10.66 dB over the room vs the declared 6 +/- 2, repaired
by the log amp law to 0.14 (ratio 0.585 = -4.66 dB), landing at 6.09.

FROZEN 2026-09-04 from that record: every PROVISIONAL bar STANDS exactly as
pre-registered — ACC_MARGIN 0.25 (the null cleared base by -0.072),
MI_MARGIN_BITS 0.25 (null MI 0.056 against its own 0.101 floor; VO.02's
precedent margin, same estimator, same codebook), R2_LEVEL_MIN 0.30
(pre-registered from VO.01's gated dur/f0 recovery, not from this pilot),
FREEZE_ACC_TOL 0.10 / FREEZE_MI_TOL 0.05 (freeze read -0.067 and -0.026),
MISLEAD_MIN 0.75 / SWAP_TRUE_MAX 0.25 (measured 1.000 / 0.000). No bar was
tightened toward the claim arm's numbers and none was loosened.
"""

from __future__ import annotations

import math

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402,F401  (must follow ensure_gl; V._world needs it)

import ContactAudio as CA  # noqa: E402

from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402
from .. import drives as DR  # noqa: E402
from .. import thermal as TH  # noqa: E402
from . import vo_01_voice_emission as V  # noqa: E402

# The claim is about the certified substrates, not only this file: the need
# laws are drives.py/thermal.py's, the channel is VO.01's world, room
# calibration, pose distribution, ear and feature extractor. Change any of
# them and this certificate goes stale loudly.
IMPL_DEPS = ["playground.py", "ContactAudio.py",
             "experiments/tests/vo_01_voice_emission.py",
             "experiments/thermal.py", "experiments/drives.py"]

# ── SM.02's idiom: run() refuses until the pilot has been read and this
# flag flipped. No bar below may ever be fitted to the claim arm's number.
_GATES_FROZEN = True            # frozen 2026-09-04 from the PILOT RECORD above

# ── the life ────────────────────────────────────────────────────────────
LIFE_S = 720.0
DT_S = 2.0
N_MOMENTS = int(LIFE_S / DT_S)          # 360 decision moments

# ambient schedule: long warm stretches, short cold snaps, and a guard that
# walks him to warmth before the cold can kill (LF.01 owns dying).
WARM_SEG_S = (60.0, 120.0)
COLD_SEG_S = (25.0, 60.0)
WARM_OFF_C = (15.0, 25.0)               # T_eff - T_NEUTRAL while warm
COLD_EFF_C = (-12.0, -2.0)              # T_eff while cold
TB_GUARD_C = 2.0                        # forced warm below TB_LETHAL + this

E0 = 0.90                               # starting energy
E_EAT_AT = 0.20                         # below this he starts looking for food
EAT_DELAY_S = (15.0, 45.0)              # ...and finds it after this long
E_CRY = 0.35                            # hunger is urgent below this energy

# ── the reflex ──────────────────────────────────────────────────────────
CLS_NOMINAL, CLS_HUNGRY, CLS_COLD = 0, 1, 2
N_CLASSES = 3
F0_FORM = {CLS_HUNGRY: +0.6, CLS_COLD: -0.6}    # hunger high, cold low
BR_FORM = {CLS_HUNGRY: +0.4, CLS_COLD: -0.4}
FORM_JITTER = 0.10                      # seeded, per utterance
CRY_AMP = 0.14                          # FIXED: loudness may not encode level.
                                        # 0.14 places the cry-form marginal at
                                        # VO.01's declared difficulty over this
                                        # rig's poses (first pilot read 10.66 dB
                                        # at 0.5 vs the declared 6 +/- 2; the
                                        # amp law is log, -4.66 dB = ratio
                                        # 0.585 = action 0.14). A loudness
                                        # constant is fixture calibration, not
                                        # a gate.
DUR_LO, DUR_SPAN = -0.5, 1.5            # dur action = DUR_LO + DUR_SPAN * u

# ── the channel (VO.01's rig, verbatim constants) ───────────────────────
BG_EVENTS_PER_EP = V.BG_EVENTS_PER_EP
RANGE_M = V.RANGE_M
N_LEVEL = 120                           # episodes for the achieved-SIR measure

# ── the listener ────────────────────────────────────────────────────────
RIDGE_ALPHA = 10.0
MI_CODES = 8
KMEANS_ITERS = 40
N_PERM = 200
EPS = 1e-12

# ── gates: theory/design-fixed ──────────────────────────────────────────
OCC_MIN = 0.10                  # every class must be lived, or VOID
U_STD_MIN = 0.10                # urgency must have spread, or level is vacuous
MI_SELFTEST_MIN = 1.20          # planted-perfect 3-class MI over its own floor
FIRE_NOM_MAX = 0.02             # he does not cry over nothing
FIRE_URG_MIN = 0.95             # he does cry when it is real
FIRE_FRAC_LO, FIRE_FRAC_HI = 0.10, 0.85   # the registry's thermostat falsifier
SEED_SPREAD_FACTOR = 1.5        # t3_06's exact all-seeds bound for n=3, ddof=0

# ── gates: PROVISIONAL until the pilot freezes them (see docstring) ─────
ACC_MARGIN = 0.25               # held-out class acc over the base rate
MI_MARGIN_BITS = 0.25           # I(class; ear code) over its permutation floor
R2_LEVEL_MIN = 0.30             # urgency recovery (VO.01's dur/f0 gated prior)
FREEZE_ACC_TOL = 0.10           # freeze control: decode of counterfactuals...
FREEZE_MI_TOL = 0.05            # ...and their MI must sit at the floor
MISLEAD_MIN = 0.75              # swap control: decoded-as-swapped rate
SWAP_TRUE_MAX = 0.25            # ...while the truth is gone from the decode


# ═══════════════════════════════════════════════════════════════════════
# THE LIFE: certified laws on a seeded schedule
# ═══════════════════════════════════════════════════════════════════════
def _schedule(seed: int) -> dict:
    """Integrate Tb (thermal.py's law) and e (drives.py's basal drain) over
    the seeded ambient/eat schedule. Returns per-moment classes and urgency."""
    rng = np.random.RandomState(seed * 9176 + 5)
    tb, e = TH.TB_HEALTHY, E0
    seg_left, seg_warm = 0.0, True
    t_eff = TH.T_NEUTRAL
    eat_at = None
    classes = np.zeros(N_MOMENTS, dtype=int)
    u = np.zeros(N_MOMENTS)
    u_c_all = np.zeros(N_MOMENTS)
    u_h_all = np.zeros(N_MOMENTS)
    for k in range(N_MOMENTS):
        t = k * DT_S
        seg_left -= DT_S
        if seg_left <= 0.0:
            seg_warm = not seg_warm
            if seg_warm:
                seg_left = rng.uniform(*WARM_SEG_S)
                t_eff = TH.T_NEUTRAL + rng.uniform(*WARM_OFF_C)
            else:
                seg_left = rng.uniform(*COLD_SEG_S)
                t_eff = rng.uniform(*COLD_EFF_C)
        # the guard: he walks to the fire before the cold can end the life
        if not seg_warm and tb <= TH.TB_LETHAL + TB_GUARD_C:
            seg_warm, seg_left = True, rng.uniform(*WARM_SEG_S)
            t_eff = TH.T_NEUTRAL + rng.uniform(*WARM_OFF_C)
        # thermal.py's declared law, forward Euler at the decision step
        tb = min(TH.TB_HEALTHY, tb + TH.G_RATE * (t_eff - TH.T_NEUTRAL) * DT_S)
        # drives.py's basal drain and a found meal
        e = max(0.0, e - DR.BASAL_B * DT_S)
        if e <= E_EAT_AT and eat_at is None:
            eat_at = t + rng.uniform(*EAT_DELAY_S)
        if eat_at is not None and t >= eat_at:
            e, eat_at = min(1.0, e + DR.NU_APPLE), None
        u_c = np.clip((TH.TB_SHIVER - tb) / (TH.TB_SHIVER - TH.TB_LETHAL), 0, 1)
        u_h = np.clip((E_CRY - e) / E_CRY, 0, 1)
        u_c_all[k], u_h_all[k] = u_c, u_h
        if u_c <= 0.0 and u_h <= 0.0:
            classes[k], u[k] = CLS_NOMINAL, 0.0
        elif u_h >= u_c:
            classes[k], u[k] = CLS_HUNGRY, u_h
        else:
            classes[k], u[k] = CLS_COLD, u_c
    both = float(np.mean((u_c_all > 0) & (u_h_all > 0)))
    return {"classes": classes, "u": u, "both_frac": both}


def _forms(seed: int, classes: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The reflex's 4-D actions, one row per moment (rows for silent moments
    exist but are never emitted). Jitter comes from its own stream so every
    arm that reuses these forms reuses them byte-identically."""
    rng = np.random.RandomState(seed * 33461 + 11)
    acts = np.zeros((N_MOMENTS, CA.VOICE_ACTION_DIM))
    for k in range(N_MOMENTS):
        c = int(classes[k])
        j0, j1 = rng.uniform(-FORM_JITTER, FORM_JITTER, size=2)
        if c == CLS_NOMINAL:
            continue
        acts[k] = [np.clip(F0_FORM[c] + j0, -1, 1),
                   np.clip(BR_FORM[c] + j1, -1, 1),
                   CRY_AMP,
                   np.clip(DUR_LO + DUR_SPAN * float(u[k]), -1, 1)]
    return acts


# ═══════════════════════════════════════════════════════════════════════
# THE CHANNEL: one render per moment, per-moment streams shared across arms
# ═══════════════════════════════════════════════════════════════════════
def _pose_bank(seed: int, model, data) -> list:
    """A clear-line pose and room density per moment, from a stream all arms
    share — so freeze and swap hear the same rooms at the same poses."""
    rng = np.random.RandomState(seed * 104729 + 29)
    bank = []
    tries = 0
    while len(bank) < N_MOMENTS:
        tries += 1
        if tries > 80 * N_MOMENTS:
            raise RuntimeError("SO.02: could not find clear emitter poses")
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(*RANGE_M)
        pos = np.array([V.HEAD[0] + r * math.cos(ang),
                        V.HEAD[1] + r * math.sin(ang), V.HEAD[2]])
        if V._hit_geom(model, data, pos, V.HEAD) != "":
            continue
        n_bg = int(rng.randint(*BG_EVENTS_PER_EP))
        bank.append((pos, n_bg))
    return bank


def _render_moment(model, data, bg, gain, seed, k, pose, n_bg, action):
    """One moment at the ear. Room choice and ear noise come from per-moment
    streams derived from (seed, k) only — identical across arms by
    construction, whatever fired."""
    rng_room = np.random.RandomState((seed * 1_000_003 + 7919 * k + 1) % (2**31))
    rng_ear = np.random.RandomState((seed * 1_000_003 + 7919 * k + 2) % (2**31))
    synth = V._episode_synth(model, V.HEAD, bg, rng_room, n_bg, gain)
    if action is not None:
        synth.emit_voice(V.T_VOICE, pose, action, data=data)
    ear = V._ear(synth, rng_ear)
    return V._features(ear, CA.SAMPLE_RATE)


def _render_arm(model, data, bg, gain, seed, bank, classes, acts,
                idx=None) -> np.ndarray:
    """Features for the given moments (all of them by default)."""
    idx = range(N_MOMENTS) if idx is None else idx
    feats = []
    for k in idx:
        pose, n_bg = bank[k]
        action = acts[k] if classes[k] != CLS_NOMINAL else None
        feats.append(_render_moment(model, data, bg, gain, seed, k,
                                    pose, n_bg, action))
    return np.array(feats)


# ═══════════════════════════════════════════════════════════════════════
# THE LISTENER
# ═══════════════════════════════════════════════════════════════════════
def _ridge_fit(xtr: np.ndarray, ytr: np.ndarray):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    a = np.hstack([(xtr - mu) / sd, np.ones((len(xtr), 1))])
    reg = RIDGE_ALPHA * np.eye(a.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(a.T @ a + reg, a.T @ ytr)
    return mu, sd, w


def _ridge_pred(fit, x: np.ndarray) -> np.ndarray:
    mu, sd, w = fit
    return np.hstack([(x - mu) / sd, np.ones((len(x), 1))]) @ w


def _kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Lloyd's with k-means++ init, written out so the quantiser is
    deterministic under this file's own seed (VO.02's reasoning)."""
    rng = np.random.RandomState(seed)
    c = [x[rng.randint(len(x))]]
    for _ in range(k - 1):
        d2 = np.min([np.sum((x - ci) ** 2, 1) for ci in c], axis=0)
        p = d2 / max(d2.sum(), EPS)
        c.append(x[rng.choice(len(x), p=p)])
    c = np.array(c)
    for _ in range(KMEANS_ITERS):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(lab == j):
                c[j] = x[lab == j].mean(0)
    return c


def _codes(cent: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.argmin(((x[:, None] - cent[None]) ** 2).sum(-1), axis=1)


def _plugin_mi(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> float:
    j = np.zeros((nx, ny))
    np.add.at(j, (x, y), 1.0)
    j /= max(j.sum(), EPS)
    px, py = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return float(np.sum(j[nz] * np.log2(j[nz] / np.maximum((px @ py)[nz], EPS))))


def _mi_floor(x, y, nx, ny, rng, n_perm=N_PERM) -> float:
    vals, xs = np.empty(n_perm), x.copy()
    for i in range(n_perm):
        rng.shuffle(xs)
        vals[i] = _plugin_mi(xs, y, nx, ny)
    return float(np.percentile(vals, 95))


def _mi_selftest(seed: int) -> float:
    """Planted known-answer: codes carrying the class exactly must read the
    class entropy over the floor; if not, the estimator is dead and every MI
    below is meaningless (VO.02's urn lesson, at this file's cheap scale)."""
    rng = np.random.RandomState(seed * 613 + 3)
    lab = rng.randint(0, N_CLASSES, size=200)
    mi = _plugin_mi(lab, lab, N_CLASSES, MI_CODES)
    return mi - _mi_floor(lab.copy(), lab, N_CLASSES, MI_CODES, rng)


# ═══════════════════════════════════════════════════════════════════════
# THE DECODE, one code path for every arm (no arm knows which it is)
# ═══════════════════════════════════════════════════════════════════════
def _decode(feats, classes, u):
    """Train on even moments, score on odd. Returns metrics + the fits, so
    the controls can interrogate the SAME trained listener."""
    n = len(feats)
    tr = np.arange(0, n, 2)
    te = np.arange(1, n, 2)
    onehot = np.eye(N_CLASSES)[classes]
    cls_fit = _ridge_fit(feats[tr], onehot[tr])
    pred = np.argmax(_ridge_pred(cls_fit, feats[te]), axis=1)
    acc = float(np.mean(pred == classes[te]))
    base = float(np.max(np.bincount(classes[te], minlength=N_CLASSES)) / len(te))
    fire_tr = tr[classes[tr] != CLS_NOMINAL]
    fire_te = te[classes[te] != CLS_NOMINAL]
    lvl_fit = _ridge_fit(feats[fire_tr], u[fire_tr, None])
    yhat = _ridge_pred(lvl_fit, feats[fire_te])[:, 0]
    y = u[fire_te]
    r2 = float(1.0 - np.sum((y - yhat) ** 2)
               / max(np.sum((y - y.mean()) ** 2), EPS))
    cent = _kmeans(feats[tr], MI_CODES, seed=1234)
    codes_te = _codes(cent, feats[te])
    rng = np.random.RandomState(4321)
    mi = _plugin_mi(classes[te], codes_te, N_CLASSES, MI_CODES)
    floor = _mi_floor(classes[te].copy(), codes_te, N_CLASSES, MI_CODES, rng)
    return ({"acc_class": acc, "base_rate": base, "r2_level": r2,
             "mi_ear": mi, "mi_perm_p95": floor},
            {"cls_fit": cls_fit, "cent": cent, "te": te, "tr": tr})


# ═══════════════════════════════════════════════════════════════════════
# THE ACHIEVED DIFFICULTY, over THIS rig's cry-form marginal
# ═══════════════════════════════════════════════════════════════════════
def _level(model, data, bg, gain, seed, bank, acts, fire_idx) -> float:
    """VO.01's declared SIR, re-measured over this spec's own poses and the
    cry forms actually emitted — not the calibration's episodes, which would
    re-derive the formula (VO.02's tripwire lesson)."""
    rng = np.random.RandomState(seed * 6700417 + 41)
    pick = rng.choice(fire_idx, size=min(N_LEVEL, len(fire_idx)), replace=False)
    v, b = [], []
    sr = CA.SAMPLE_RATE
    for k in pick:
        pose, n_bg = bank[k]
        rng_room = np.random.RandomState(
            (seed * 1_000_003 + 7919 * int(k) + 1) % (2**31))
        room = V._episode_synth(model, V.HEAD, bg, rng_room, n_bg, gain)
        b.append(V._rms(V._ear(room, None, mute_voice=True), sr))
        alone = CA.ContactAudioSynth(model)
        alone.set_listener(V.HEAD, V.HEAD_YAW)
        alone.emit_voice(V.T_VOICE, pose, acts[k], data=data)
        v.append(V._rms(V._ear(alone, None), sr))
    return float(20.0 * math.log10(max(np.mean(v), EPS) / max(np.mean(b), EPS)))


# ═══════════════════════════════════════════════════════════════════════
# THE EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════
_CACHE: dict = {}       # seed -> what _control needs from the claim arm


def _experiment(seed: int) -> dict:
    model, data = V._world(seed)
    bg = V._background(seed)
    gain = V._bg_gain(seed)
    sched = _schedule(seed)
    classes, u = sched["classes"], sched["u"]
    acts = _forms(seed, classes, u)
    bank = _pose_bank(seed, model, data)
    fire = classes != CLS_NOMINAL
    fire_idx = np.where(fire)[0]

    feats = _render_arm(model, data, bg, gain, seed, bank, classes, acts)
    m, fits = _decode(feats, classes, u)

    # honesty of the composed system, measured over the lived trajectory
    m["fire_frac"] = float(np.mean(fire))
    m["fire_given_nominal"] = float(np.mean(fire[classes == CLS_NOMINAL])) \
        if np.any(classes == CLS_NOMINAL) else 1.0
    m["fire_given_urgent"] = float(np.mean(fire[classes != CLS_NOMINAL])) \
        if np.any(classes != CLS_NOMINAL) else 0.0
    occ = np.bincount(classes, minlength=N_CLASSES) / float(N_MOMENTS)
    m["occ_min"] = float(occ.min())
    m["occ_nominal"], m["occ_hungry"], m["occ_cold"] = map(float, occ)
    m["u_std"] = float(np.std(u[fire])) if fire.any() else 0.0
    m["both_frac"] = sched["both_frac"]
    m["feats_finite"] = float(np.all(np.isfinite(feats)))

    # rig: the estimator's known answer, and the declared difficulty
    m["mi_selftest"] = _mi_selftest(seed)
    m["voice_to_background_db"] = _level(model, data, bg, gain, seed, bank,
                                         acts, fire_idx)

    # THE FLUENCY NULL: same renders, same rate, same marginal — the
    # moment->sound pairing permuted before training. Same decode path.
    rng = np.random.RandomState(seed * 7907 + 13)
    perm = rng.permutation(N_MOMENTS)
    nm, _ = _decode(feats[perm], classes, u)
    m["null_acc"] = nm["acc_class"]
    m["null_base_rate"] = nm["base_rate"]
    m["null_mi"] = nm["mi_ear"]
    m["null_mi_p95"] = nm["mi_perm_p95"]
    m["null_r2_level"] = nm["r2_level"]

    _CACHE[seed] = {"model": model, "data": data, "bg": bg, "gain": gain,
                    "bank": bank, "classes": classes, "u": u, "acts": acts,
                    "fits": fits, "fire_idx": fire_idx}
    return m


def _control(seed: int) -> dict:
    """The registry's two controls, both against the CLAIM-trained listener."""
    if seed not in _CACHE:                      # run_spec runs fn first; if a
        _experiment(seed)                       # future runner reorders, rebuild
    c = _CACHE[seed]
    model, data, bg, gain = c["model"], c["data"], c["bg"], c["gain"]
    bank, classes, u, acts = c["bank"], c["classes"], c["u"], c["acts"]
    fits, fire_idx = c["fits"], c["fire_idx"]
    te_fire = np.array([k for k in fits["te"] if classes[k] != CLS_NOMINAL])

    # FREEZE: nominal body, forced speech at the claim's moments, forms
    # permuted among them; decode scored against the counterfactual labels.
    rng = np.random.RandomState(seed * 51427 + 17)
    perm_within = fire_idx[rng.permutation(len(fire_idx))]
    frz_acts = acts.copy()
    frz_acts[fire_idx] = acts[perm_within]
    frz = _render_arm(model, data, bg, gain, seed, bank, classes, frz_acts,
                      idx=te_fire)
    pred = np.argmax(_ridge_pred(fits["cls_fit"], frz), axis=1)
    cf = classes[te_fire]                       # counterfactual labels
    freeze_acc = float(np.mean(pred == cf))
    counts = np.bincount(cf, minlength=N_CLASSES)
    freeze_chance = float(counts.max() / max(len(cf), 1))
    codes = _codes(fits["cent"], frz)
    rngf = np.random.RandomState(seed * 271 + 19)
    frz_mi = _plugin_mi(cf, codes, N_CLASSES, MI_CODES)
    frz_floor = _mi_floor(cf.copy(), codes, N_CLASSES, MI_CODES, rngf)

    # SWAP: real needs, the other channel's form. The listener must be misled.
    swp_acts = acts.copy()
    for k in fire_idx:
        cswap = CLS_COLD if classes[k] == CLS_HUNGRY else CLS_HUNGRY
        j0, j1 = swp_acts[k, 0] - F0_FORM[classes[k]], \
            swp_acts[k, 1] - BR_FORM[classes[k]]
        swp_acts[k, 0] = np.clip(F0_FORM[cswap] + j0, -1, 1)
        swp_acts[k, 1] = np.clip(BR_FORM[cswap] + j1, -1, 1)
    swp = _render_arm(model, data, bg, gain, seed, bank, classes, swp_acts,
                      idx=te_fire)
    spred = np.argmax(_ridge_pred(fits["cls_fit"], swp), axis=1)
    swapped = np.where(cf == CLS_HUNGRY, CLS_COLD, CLS_HUNGRY)
    return {"c_freeze_acc": freeze_acc, "c_freeze_chance": freeze_chance,
            "c_freeze_mi": frz_mi, "c_freeze_mi_p95": frz_floor,
            "c_swap_misled": float(np.mean(spred == swapped)),
            "c_swap_true_acc": float(np.mean(spred == cf)),
            "c_n_fire_te": float(len(te_fire))}


# ═══════════════════════════════════════════════════════════════════════
# THE VERDICT
# ═══════════════════════════════════════════════════════════════════════
def _worst_lo(d, key):
    return d[key] - SEED_SPREAD_FACTOR * d.get(key + "_std", 0.0)


def _worst_hi(d, key):
    return d[key] + SEED_SPREAD_FACTOR * d.get(key + "_std", 0.0)


def _claim(acc, base, mi, floor, r2) -> bool:
    """One predicate for claim and null; it never knows which arm it reads."""
    return bool(acc - base >= ACC_MARGIN
                and mi - floor >= MI_MARGIN_BITS
                and r2 >= R2_LEVEL_MIN)


def _check(m: dict, c: dict):
    # ── RIG: a life that never posed the question refutes nothing. VOID.
    rig_ok = (_worst_lo(m, "occ_min") >= OCC_MIN
              and _worst_lo(m, "u_std") >= U_STD_MIN
              and m["feats_finite"] >= 1.0
              and _worst_lo(m, "mi_selftest") >= MI_SELFTEST_MIN
              and abs(m["voice_to_background_db"] - V.SIR_TARGET_DB)
              <= V.SIR_TOL_DB
              and c.get("c_n_fire_te", 0.0) >= 20.0)
    if not rig_ok:
        return Status.VOID

    # ── THE NULL must fail the identical predicate.
    if _claim(m["null_acc"], m["null_base_rate"], m["null_mi"],
              m["null_mi_p95"], m["null_r2_level"]):
        return False

    # ── THE CONTROLS, both directions the registry demands.
    freeze_dead = (_worst_hi(c, "c_freeze_acc")
                   <= c["c_freeze_chance"] + FREEZE_ACC_TOL
                   and c["c_freeze_mi"] - c["c_freeze_mi_p95"]
                   <= FREEZE_MI_TOL)
    swap_misled = (_worst_lo(c, "c_swap_misled") >= MISLEAD_MIN
                   and _worst_hi(c, "c_swap_true_acc") <= SWAP_TRUE_MAX)
    if not (freeze_dead and swap_misled):
        return False

    # ── HONESTY: he talks when it is true and only then, and not always.
    honest = (m["fire_given_nominal"] <= FIRE_NOM_MAX
              and _worst_lo(m, "fire_given_urgent") >= FIRE_URG_MIN
              and _worst_lo(m, "fire_frac") >= FIRE_FRAC_LO
              and _worst_hi(m, "fire_frac") <= FIRE_FRAC_HI)
    if not honest:
        return False

    # ── THE CLAIM, worst seed.
    return _claim(_worst_lo(m, "acc_class"), _worst_hi(m, "base_rate"),
                  _worst_lo(m, "mi_ear"), _worst_hi(m, "mi_perm_p95"),
                  _worst_lo(m, "r2_level"))


def _dry() -> list:
    """Proves the verdict LOGIC; the pilot proves the rows are reachable."""
    def base(**kw):
        d = {"occ_min": 0.25, "u_std": 0.3, "feats_finite": 1.0,
             "mi_selftest": 1.5, "voice_to_background_db": V.SIR_TARGET_DB,
             "acc_class": 0.9, "base_rate": 0.4, "mi_ear": 1.1,
             "mi_perm_p95": 0.1, "r2_level": 0.6,
             "null_acc": 0.4, "null_base_rate": 0.4, "null_mi": 0.08,
             "null_mi_p95": 0.1, "null_r2_level": -0.1,
             "fire_frac": 0.5, "fire_given_nominal": 0.0,
             "fire_given_urgent": 1.0, "both_frac": 0.05}
        d.update(kw)
        return d

    ctl = {"c_freeze_acc": 0.5, "c_freeze_chance": 0.5, "c_freeze_mi": 0.05,
           "c_freeze_mi_p95": 0.1, "c_swap_misled": 0.95,
           "c_swap_true_acc": 0.03, "c_n_fire_te": 80.0}
    rows = [
        ("all green", base(), ctl, True),
        ("a class never lived -> VOID", base(occ_min=0.02), ctl, Status.VOID),
        ("urgency without spread -> VOID", base(u_std=0.01), ctl, Status.VOID),
        ("room at the wrong difficulty -> VOID",
         base(voice_to_background_db=V.SIR_TARGET_DB - 9), ctl, Status.VOID),
        ("MI estimator dead -> VOID", base(mi_selftest=0.2), ctl, Status.VOID),
        ("fluency null clears -> FAIL",
         base(null_acc=0.9, null_mi=1.0, null_r2_level=0.6), ctl, False),
        ("freeze control decodes counterfactuals -> FAIL", base(),
         {**ctl, "c_freeze_acc": 0.9, "c_freeze_mi": 0.9}, False),
        ("swap fails to mislead -> FAIL", base(),
         {**ctl, "c_swap_misled": 0.3, "c_swap_true_acc": 0.6}, False),
        ("the thermostat: always firing -> FAIL", base(fire_frac=0.97),
         ctl, False),
        ("cries over nothing -> FAIL", base(fire_given_nominal=0.3),
         ctl, False),
        ("silent when it is real -> FAIL", base(fire_given_urgent=0.5),
         ctl, False),
        ("class decode at base rate -> FAIL", base(acc_class=0.45),
         ctl, False),
        ("MI at the floor -> FAIL", base(mi_ear=0.2), ctl, False),
        ("level unrecoverable -> FAIL", base(r2_level=0.05), ctl, False),
    ]
    return [(name, _check(mm, cc), want,
             _check(mm, cc) == want) for name, mm, cc, want in rows]


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "SO.02 gates are provisional — run the pilot (seed 90), freeze "
            "the bars in this file from its NULL/CONTROL readings, then run "
            "(SM.02's _GATES_FROZEN idiom). Nothing here may be fitted to "
            "the claim arm's numbers.")
    return run_spec(BY_ID["SO.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    ok = True
    for name, got, want, good in _dry():
        ok &= good
        print(f"  [{'ok' if good else 'XX'}] {name}: got={got} want={want}")
    print("dry table:", "PASS" if ok else "FAIL")
