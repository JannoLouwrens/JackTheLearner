"""VO.02 — do two Jacks invent a signal?

GOAL.md lists VOICE as the one constitutional sense that is an EFFECTOR — "how
a creature acts on other creatures" — and lists OTHER MINDS as one of the three
expansions beyond the jungle. Both commitments read `0 pass`. VO.01 certified
the CHANNEL: an emission Jack makes arrives at another creature's ear as sound,
quieter with distance, muffled through a solid, mixed into the same stream as
everything else. It certified nothing about MEANING, and said so: *"Not that
the emission is a signal (nothing here learns), not that two agents coordinate
(VO.02)."*

This spec is that. Two Jacks, one world, a coordination problem neither can
solve alone, and NOTHING passing between them except air.

THE FAILURE THIS SPEC EXISTS TO PREVENT is named in its own registry entry and
it is this field's most common false positive (Lowe, Foerster, Boureau, Pineau
& Dauphin, *On the Pitfalls of Measuring Emergent Communication*, AAMAS 2019,
arXiv:1903.05168): **coordination rises while the signal carries nothing.** The
pair finds some other channel — position, timing, turn count, a shared trunk —
and every number goes up. Lowe et al. measured speaker consistency at 0.202 for
a trained 2x2 pair, 0.198 with the messages SCRAMBLED, and 0.171 with the
emission parameters NEVER TRAINED. A metric that cannot separate those three is
measuring the agents' shared machinery, not their communication.

So the claim here is deliberately not "they coordinated". It is:

    I(referent ; what actually arrived at the listener's ear) rises above the
    shuffled-channel floor, AND coordination rises with it.

Both conjuncts, or nothing. The measurement is taken AT THE EAR, downstream of
1/r attenuation, the pan law, the ray-cast, the ear's own noise floor and a
room full of real contact audio — never at the emitter's mouth, where a signal
that never survives the channel would score exactly the same.


## THE GAME

`N_STATES = 4` referents (Lowe's 4x4 game, and Barrett 2009's evidence that
basic reinforcement converges reliably at this size where 3x3 already fails
~9.6% of the time). One episode:

  1. A referent `r ~ U{0..3}` is shown to the EMITTER and to nobody else.
  2. The emitter stands at a random bearing and range on a verified-clear line
     to the listener's head — VO.01's set-A pose distribution, so the listener
     cannot identify the call by loudness or direction.
  3. The emitter vocalises: four continuous dimensions (f0, brightness,
     amplitude, duration) through `ContactAudio.emit_voice`. It is NOT choosing
     from a codebook we wrote. The sound itself is invented.
  4. The listener hears `[2, T]` stereo — the call, plus real contact events
     from real dropped objects at a calibrated level, plus its own noise floor
     — and nothing else. No referent, no emission vector, no event times, no
     pose. `vo_01._features` is the whole of its input, unchanged.
  5. The listener picks one of four acts. Payoff 1 if it matches `r`, else 0.
     BOTH agents receive it; neither observes anything the other saw.

"A coordination problem that pays only if they act differently" (the registry's
words) is structural here rather than incentivised: the emitter cannot act and
the listener cannot see. Two Jacks doing the same thing score chance, forever.
The MUTED arm is what proves it — see the rig gates.

The two agents hold NO shared parameters. Lowe et al.'s own localisation of the
artifact is that speaker consistency collapses from 0.510 to 0.124 (4x4) when
the emission and action networks are separated; this rig is separated from the
start, so that particular artifact cannot be present to collapse.

Nothing is differentiable through the channel, because the channel is a
renderer. Both agents learn by REINFORCE on the common payoff, which is the
honest situation for a creature: you find out whether the noise you made
worked, not what you should have made instead.


## THE FOUR ARMS — three of them mandatory nulls

  `trained`    the claim. Both heads learn.
  `untrained`  null (ii). The emission head is frozen at initialisation and
               never updated; the listener trains normally against it. This is
               the arm that catches "the listener learned the room".
  `muted`      null (iii). `mute_voice=True`; the listener hears the room and
               its own noise floor. Both heads train normally. A muted pair
               above chance means the referent is reaching the listener OUTSIDE
               the channel, which is an apparatus fault, not a refutation —
               so it VOIDs.
  `scrambled`  null (i), and this spec's `_control`. The emission is permuted
               BEFORE DELIVERY: the waveform the listener receives in episode i
               is the one rendered from episode j's emission, at episode j's
               pose, with episode j's room. Every marginal statistic the
               listener could exploit — level, bearing, timing, background
               density, the emitter's own action distribution — is preserved
               exactly. Only the pairing is destroyed. Both heads still train.

The scrambled arm is scored by the SAME predicate as the claim arm, on the same
keys, computed by the same code path (`_claim`). LESSONS.md, 2026-08-29: *a
control scored on a gate that mentions the control is a control that cannot
fail — and it reads exactly like a strong one.* `_claim` does not know which
arm it is looking at.


## THE MEASUREMENTS

**MI AT THE EAR — `mi_ear`, in bits.** On held-out episodes the received
features are quantised by a k-means codebook (`MI_CODES = 8`) fitted on a
DISJOINT half, and `I(referent ; code)` is the plug-in estimate on the other
half. Plug-in MI is biased upward at finite sample, and the bias is exactly the
size of the effect a weak signal would produce, so the number is never read
alone: `mi_perm_p95` is the 95th percentile of the same estimator over
`N_PERM = 200` shuffles of the referent labels within the held-out set. The
gate is on the DIFFERENCE. A floor computed from the same estimator on the same
sample absorbs the same bias.

**CAUSAL INFLUENCE — `cic`, in bits, and it is INTERVENTIONAL.** The registry
declares positive LISTENING as the control, not positive signalling, and cites
Lowe et al.'s finding that 89.3% (2x2), 97.9% (4x4) and 99.9% (8x8) of games
sat within 1.02x of the CIC minimum while *looking* like they communicated. A
correlational I(message; action) cannot separate those, because a listener that
ignores the sound and a listener that uses it both produce actions correlated
with the message whenever the pair coordinates at all.

So `cic` is measured by intervening. At each of `N_CIC` held-out poses, the
emitter's call for EVERY referent is rendered through that ONE pose, that ONE
room, that ONE background — and the listener's action distribution is read for
each. Position, timing, distance, turn count and room are held fixed by
construction; the only thing that varies is the sound. `cic` is
`I(intervened referent ; act)` averaged over poses. Its floor `cic_perm_p95`
re-samples the response rows ACROSS poses, which destroys the within-pose
sound->act dependence while preserving the marginal distribution of listener
responses exactly. Both are reported, always, never the value alone.

**THE HARNESS CHECK — a known-answer test for the estimator, run first.** The
registry's own staging note: *the floor of this literature is TABULAR: 2
agents, ZERO parameters, 2 states/2 signals/2 acts, four Polya urns, Roth-Erev
reinforcement, convergence to a signalling system with probability 1 (Argiento,
Pemantle, Skyrms & Volkov, Stoch. Proc. Appl. 2009), measured at ~0.2 s of one
CPU core.* That system is run here — with forgetting, which Barrett 2009 shows
fixes basic reinforcement's failure rate to 0% up to 32 symbols — and the MI
estimator is pointed at it. It must read ~1 bit on a converged 2x2 signalling
system and the floor on its scrambled twin. If it does not, the instrument is
broken and every number below is meaningless: the run VOIDs and says so.

That check costs a fifth of a second and it is the difference between "the MI
was at the floor" meaning *he did not signal* and meaning *we cannot measure
signalling*.

**THE CHANNEL IS ALIVE IN THIS RIG — `probe_r2_*`.** VO.01's certificate is
about VO.01's rig. Here a ridge probe on the received features must recover the
four emission dimensions on held-out episodes with uniform-random emissions,
and its MUTED twin must not. A dead channel VOIDs; it does not refute a claim
about signalling, because there was no signal to carry. (LESSONS.md: *an
at-chance control must carry proof its instrument was alive.*)

**THE DIFFICULTY IS DECLARED — `voice_to_background_db`.** VO.01's scar: a
recovery number means nothing without the level of the interference it was
measured against. The room is scaled per seed by VO.01's own `_bg_gain` to put
the voice `SIR_TARGET_DB` above it, and the achieved level is re-measured HERE,
over THIS rig's pose bank (not the calibration's own episodes, which would
re-derive the formula), and gated two-sided. Quieting the room to buy a PASS
fails as loudly as a room that drowns the voice.


## WHAT WOULD FALSIFY IT, stated before the run

`coord_without_mi = 1` is the registry's named false positive made into a
reported field: the coordination conjunct clears and the MI conjunct does not.
That is a FAIL, not a VOID, and the message names it. It is the outcome this
whole file is shaped around, and it is the one an emergent-communication paper
is most likely to report as a success.

The other honest FAIL is `coord` at chance — the pair never solved the game.
The other honest VOID is a dead instrument: the urn check, the probe, the level,
or a muted pair that solves the game anyway.


## WHAT THIS SPEC DOES NOT CLAIM

Not compositionality. A holistic protocol is what this size of game produces
and what the registry note expects; compositional structure needs a re-learning
bottleneck plus an expressivity constraint (FROZEN_VS_PLASTIC.md 10.6b) and is
a different spec. Not that the signal means anything to Jack outside this game.
Not that four referents is language. The claim is that two separate learners,
sharing no parameters, invented a sound-to-referent mapping that SURVIVES A
REAL ACOUSTIC CHANNEL — and that we can tell that apart from the three ways of
looking like it.

Depends on VO.01, whose world, room calibration, pose distribution, ear model
and feature extractor are imported rather than reimplemented — so this claim is
about the same certified channel, and it goes stale the moment that one does.


## PILOT RECORD — 2026-08-30, seed 0, full envelope, /data/vo02_pilot_seed0.json

Every arm at the registered 600x64 = 38,400 episodes. Disclosed in full,
including the claim arm, so an auditor can see exactly what was on the screen
when the bars below were frozen.

    arm         coord    mi_ear / floor      cic / floor
    trained     0.9962   1.5284 / 0.0652     1.9997 / 1.3713
    untrained   0.3962   0.0459 / 0.0628     0.5516 / 0.6243
    scrambled   0.2525   0.0395 / 0.0587     0.0049 / 0.0292
    muted       0.2737   0.0430 / 0.0606     0.0017 / 0.0679
    chance 0.250; urn 1.0 success / 0.9998 bits; probe R2 0.704, muted -0.414;
    voice_to_background 6.587 dB vs the declared 6.0 +/- 2.

**All three nulls are dead, and — the part that matters — they are dead in
DIFFERENT WAYS.** Scrambled and muted sit at chance with sub-floor information.
The UNTRAINED arm does not: it coordinates at 0.396, well above chance, because
a randomly-initialised emission head is still a FIXED RANDOM CODE, and a fixed
random code carries information the listener can learn. Its MI at the ear stays
at the floor (the four random calls are not separable through this room at this
level) while its CIC reaches 0.552 — under its own floor of 0.624, but not by
much. This is precisely the discrimination Lowe et al. report most metrics lack,
and this rig has it: three nulls, three distinguishable failure modes.

**ONE BAR WAS STRENGTHENED ON THAT BASIS, AND ONLY THAT ONE.** The provisional
`COORD_MIN = 0.55` / `COORD_MARGIN = 0.20` put the coordination gate at 0.45,
which is **0.054 above the untrained null**. That is too thin: a seed whose
random init happens to be more discriminable could clear a gate the fixed-code
null was supposed to be nowhere near. The bars are raised to 0.70 / 0.35, whose
justification is the NULL arm's 0.396 and nothing else. Law 4 permits
strengthening and forbids the reverse; no other bar moved, in either direction.
`MI_MARGIN_BITS` and `CIC_MARGIN_BITS` stand exactly as pre-registered before
the pilot ran.

**The CIC ceiling predicted by `_floor_selftest` was met almost exactly.** The
self-test said a perfect signalling system clears the across-pose CIC floor by
0.617-0.622 bits; the trained arm cleared it by 0.628 at a CIC of 1.9997 against
a theoretical maximum of 2.0. The estimator's ceiling is real and this arm is
sitting on it — which is why `CIC_MARGIN_BITS` must stay far below it.

**BUDGET, DECLARED FROM MEASUREMENT AND NOT FROM THE REGISTRY'S ESTIMATE.** One
seed cost 1,142.9 s wall-clock: 217-307 s per arm plus 13 s of rig instruments.
Three seeds project to **0.95 h**. The registry declared `Budget.GPU`; the
measurement says `Budget.CPU_LONG`, and the entry is amended to match. The time
is spent in `ContactAudio`'s numpy DSP and MuJoCo's ray casts, with two MLPs
totalling under 15K parameters — **a GPU would buy nothing here**, and leaving
the declaration at GPU would stock a queue class this spec can never honestly
spend a Kaggle hour on. T3.06 took the same correction the same day; the routing
lesson is that a declared attribute machinery consumes must match behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402,F401  (must follow ensure_gl; imported for _world)

import ContactAudio as CA  # noqa: E402

from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402
from . import vo_01_voice_emission as V  # noqa: E402

# The claim is about the certified channel, not only about this file. VO.01's
# module is a dependency in the same sense `playground.py` is: change the ear,
# the room calibration or the feature vector and this certificate must go stale
# loudly rather than stand over a channel it no longer describes.
IMPL_DEPS = ["playground.py", "ContactAudio.py",
             "experiments/tests/vo_01_voice_emission.py"]

# ── PROVISIONAL BARS ────────────────────────────────────────────────────
# SM.02's idiom, and T3.06/T2.09/T2.19's precedent: the bars below are sized
# from theory where theory fixes them (chance is 1/N_STATES; a converged 2x2
# signalling system carries exactly 1 bit) and from the PILOT's rig arms
# everywhere else. `run()` refuses until a pilot has been read and this flag is
# flipped in the file. No bar here was ever fitted to the claim arm's number.
_GATES_FROZEN = True            # frozen 2026-08-30 from the PILOT RECORD above

# ── the game ────────────────────────────────────────────────────────────
N_STATES = 4                    # referents; Lowe et al.'s 4x4 game
N_ACTS = N_STATES
CHANCE = 1.0 / N_STATES

# ── training ────────────────────────────────────────────────────────────
# PROVISIONAL — sized by the pilot at measured throughput, never estimated.
# The LC.03 budget scar and BA.02's note: re-cost the envelope in the pilot and
# amend the TIER, never the thresholds.
BATCH = 64
N_UPDATES = 600
EMIT_LR = 3e-3
LIST_LR = 1e-3
EMIT_HIDDEN = 16
LIST_HIDDEN = 64
LOGSTD_INIT = math.log(0.5)
LOGSTD_MIN = math.log(0.05)
ENT_COEF_EMIT = 1e-3
ENT_COEF_LIST = 1e-2
ENT_ANNEAL = 0.5                # entropy bonus decays to this fraction by the end
BASELINE_BETA = 0.02

# ── evaluation ──────────────────────────────────────────────────────────
N_EVAL = 800                    # held-out episodes: coordination and MI at the ear
N_CIC = 150                     # held-out poses for the interventional CIC
N_PROBE = 400                   # uniform-emission episodes for the liveness probe
N_LEVEL = 200                   # episodes for the achieved SIR measurement
MI_CODES = 8                    # k-means codebook over received features
KMEANS_ITERS = 40
N_PERM = 200                    # permutations behind every floor

# ── the tabular harness check (Skyrms' urns) ────────────────────────────
URN_STATES = 2
URN_SIGNALS = 2
URN_ACTS = 2
URN_PLAYS = 100_000
URN_FORGET = 1e-4               # Barrett 2009: forgetting fixes basic reinforcement
URN_TAIL = 5_000                # plays scored for convergence

# ── gates: theory-fixed ─────────────────────────────────────────────────
URN_SUCCESS_MIN = 0.95          # a converged 2x2 signalling system
URN_MI_MIN = 0.90               # ...carries ~1 bit; the estimator must see it
URN_MI_SCRAM_MAX = 0.10         # ...and must NOT see it in the scrambled twin
CHANCE_TOL = 0.06               # band around 1/N_STATES for an arm at chance

# ── gates: PROVISIONAL, to be frozen from the pilot's RIG arms ──────────
PROBE_R2_MIN = 0.50             # the channel is alive in THIS rig (VO.01: 0.50)
PROBE_MUTE_MAX = 0.10           # ...and the probe is not reading the room
# STRENGTHENED 2026-08-30 from 0.55/0.20, and the justification is the UNTRAINED
# NULL, not the claim arm: a frozen emission head is a fixed random code and
# coordinated at 0.396, leaving the old gate only 0.054 of clearance above a
# null it is supposed to be nowhere near. Law 4 permits strengthening only.
COORD_MIN = 0.70                # coordination, absolute
COORD_MARGIN = 0.35             # ...and above chance by this much
COORD_TSTAT_MIN = 3.0           # ...at >= 3 sigma across seeds
MI_MARGIN_BITS = 0.25           # I(referent; ear code) above its permutation floor
CIC_MARGIN_BITS = 0.15          # interventional influence above its floor
SEED_SPREAD_FACTOR = 1.5        # t3_06's exact all-seeds bound for n=3, ddof=0

# ── the floors' own known-answer bars, measured on PLANTED structure ────
# See `_floor_selftest`. These are properties of the ESTIMATOR, measured on
# synthetic data with a known answer, so freezing them costs nothing and tells
# nothing about any arm.
MI_FLOOR_COLLAPSE_MIN = 1.50    # planted-perfect measures 1.972 of a 2.0 ceiling
CIC_FLOOR_COLLAPSE_MIN = 0.40   # planted-perfect measures 0.617 of a 2.0 ceiling

EPS = 1e-12

# ── THE ARITHMETIC GUARD ────────────────────────────────────────────────
# A gate asking for more headroom than its own floor leaves available is
# UNSATISFIABLE BY ARITHMETIC — it cannot be cleared by a perfect result, so it
# is not a threshold, it is a refusal wearing one. T3.10 was piloted, dispatched
# and parked on 2026-08-30 with exactly this defect, discovered only after the
# GPU had run. It is cheap to make it impossible instead: the margins are
# checked against the collapse the floors actually permit, AT IMPORT, so a spec
# with an unsatisfiable gate cannot be registered, let alone dispatched.
assert MI_MARGIN_BITS < MI_FLOOR_COLLAPSE_MIN, (
    f"MI_MARGIN_BITS={MI_MARGIN_BITS} exceeds the {MI_FLOOR_COLLAPSE_MIN} bits "
    f"a perfect signalling system clears its own floor by — unsatisfiable")
assert CIC_MARGIN_BITS < CIC_FLOOR_COLLAPSE_MIN, (
    f"CIC_MARGIN_BITS={CIC_MARGIN_BITS} exceeds the {CIC_FLOOR_COLLAPSE_MIN} "
    f"bits a perfect signalling system clears its own floor by — unsatisfiable")


# ═══════════════════════════════════════════════════════════════════════
# THE ESTIMATOR, and the known-answer test that has to precede it
# ═══════════════════════════════════════════════════════════════════════
def _plugin_mi(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> float:
    """I(X;Y) in bits, plug-in, on two integer label vectors.

    Positively biased at finite sample — which is why nothing in this file ever
    reads it without the matching permutation floor from `_mi_floor`.
    """
    j = np.zeros((nx, ny), dtype=float)
    np.add.at(j, (x, y), 1.0)
    j /= max(j.sum(), EPS)
    px = j.sum(1, keepdims=True)
    py = j.sum(0, keepdims=True)
    nz = j > 0
    return float(np.sum(j[nz] * np.log2(j[nz] / np.maximum((px @ py)[nz], EPS))))


def _mi_floor(x, y, nx, ny, rng, n_perm=N_PERM) -> tuple:
    """(p95, mean) of the SAME estimator on the SAME sample with X shuffled.

    The bias the plug-in estimate carries is a function of the sample size and
    the alphabet sizes, both of which are held identical here. Reported as a
    pair because a floor whose mean and p95 are far apart is a floor computed on
    too little data to gate against.
    """
    vals = np.empty(n_perm)
    xs = x.copy()
    for i in range(n_perm):
        rng.shuffle(xs)
        vals[i] = _plugin_mi(xs, y, nx, ny)
    return float(np.percentile(vals, 95)), float(np.mean(vals))


def _kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Lloyd's algorithm with k-means++ init. Returns centroids.

    Written out rather than imported so the quantiser is deterministic under
    this file's own seed and carries no dependency the certificate does not
    declare.
    """
    rng = np.random.RandomState(seed)
    cent = [x[rng.randint(len(x))]]
    for _ in range(k - 1):
        d = np.min(((x[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        tot = d.sum()
        cent.append(x[rng.randint(len(x))] if tot <= EPS
                    else x[int(rng.choice(len(x), p=d / tot))])
    c = np.array(cent)
    for _ in range(KMEANS_ITERS):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j]
                        for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c


def _assign(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1).astype(int)


def _ear_mi(feats: np.ndarray, refs: np.ndarray, seed: int) -> dict:
    """I(referent ; received code) in bits, with its permutation floor.

    The codebook is fitted on the FIRST half and the MI is scored on the SECOND.
    A codebook fitted on the scored sample would let the quantiser memorise the
    partition, which is the same disease as a probe scored on its training set.
    """
    x = feats - feats.mean(0)
    x = x / (feats.std(0) + 1e-9)
    h = len(x) // 2
    cent = _kmeans(x[:h], MI_CODES, seed * 7919 + 3)
    code = _assign(x[h:], cent)
    r = refs[h:].astype(int)
    mi = _plugin_mi(r, code, N_STATES, MI_CODES)
    p95, mean = _mi_floor(r, code, N_STATES, MI_CODES,
                          np.random.RandomState(seed * 104729 + 5))
    used = len(np.unique(code))
    return {"mi_ear": mi, "mi_perm_p95": p95, "mi_perm_mean": mean,
            "mi_codes_used": float(used)}


# ── the tabular known-answer system ─────────────────────────────────────
def _urn_game(seed: int) -> dict:
    """Skyrms' signalling game on four Polya urns, Roth-Erev with forgetting.

    Sender urns: one per state, balls labelled by signal. Receiver urns: one per
    signal, balls labelled by act. Success reinforces the two balls drawn.
    Convergence to a signalling system has probability 1 at 2x2 (Argiento,
    Pemantle, Skyrms & Volkov 2009); forgetting extends that to larger games
    (Barrett 2009). Zero parameters, no gradients, ~0.2 s.

    Its purpose here is NOT to demonstrate signalling. It is to hand `_ear_mi`'s
    estimator a system whose answer is known — 1 bit — and a scrambled twin
    whose answer is known to be 0.
    """
    rng = np.random.RandomState(seed * 15485863 + 7)
    send = np.ones((URN_STATES, URN_SIGNALS))
    recv = np.ones((URN_SIGNALS, URN_ACTS))
    ok = np.zeros(URN_PLAYS, dtype=np.int8)
    st = np.empty(URN_PLAYS, dtype=int)
    sg = np.empty(URN_PLAYS, dtype=int)
    for t in range(URN_PLAYS):
        s = rng.randint(URN_STATES)
        p = send[s] / send[s].sum()
        m = int(rng.choice(URN_SIGNALS, p=p))
        q = recv[m] / recv[m].sum()
        a = int(rng.choice(URN_ACTS, p=q))
        st[t], sg[t] = s, m
        if a == s:
            ok[t] = 1
            send[s, m] += 1.0
            recv[m, a] += 1.0
        send *= (1.0 - URN_FORGET)
        recv *= (1.0 - URN_FORGET)
        np.maximum(send, 1e-6, out=send)
        np.maximum(recv, 1e-6, out=recv)
    tail = slice(URN_PLAYS - URN_TAIL, URN_PLAYS)
    s_t, g_t = st[tail], sg[tail]
    mi = _plugin_mi(s_t, g_t, URN_STATES, URN_SIGNALS)
    p95, _ = _mi_floor(s_t, g_t, URN_STATES, URN_SIGNALS,
                       np.random.RandomState(seed * 2749 + 13))
    # the scrambled twin: the same plays, the signal decoupled from the state
    scram = g_t.copy()
    np.random.RandomState(seed * 3571 + 17).shuffle(scram)
    mi_s = _plugin_mi(s_t, scram, URN_STATES, URN_SIGNALS)
    return {"urn_success": float(ok[tail].mean()),
            "urn_mi": mi, "urn_mi_floor": p95, "urn_mi_scram": mi_s}


# ═══════════════════════════════════════════════════════════════════════
# THE WORLD — VO.01's, imported, not reimplemented
# ═══════════════════════════════════════════════════════════════════════
_BANK: dict = {}


def _pose_bank(seed: int, n: int) -> list:
    """`n` episode conditions on a verified-clear mouth->ear line.

    One bank per seed, shared by every arm and by both halves of every
    evaluation, so `trained`, `untrained`, `muted` and `scrambled` see
    BYTE-IDENTICAL world conditions at the same episode index. Any difference
    between the arms is then the arm and not the draw.

    A condition fixes the pose, how many background contacts the room makes,
    WHICH real recorded contacts they are, and when they land — everything
    except what the emitter says.
    """
    key = (seed, n)
    if key in _BANK:
        return _BANK[key]
    model, data = V._world(seed)
    bg = V._background(seed)
    rng = np.random.RandomState(seed * 6151 + 23)
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > 40 * n:
            raise RuntimeError("VO.02: could not find clear emitter poses")
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(*V.RANGE_M)
        pos = np.array([V.HEAD[0] + r * math.cos(ang),
                        V.HEAD[1] + r * math.sin(ang), V.HEAD[2]])
        if V._hit_geom(model, data, pos, V.HEAD) != "":
            continue
        n_bg = int(rng.randint(*V.BG_EVENTS_PER_EP))
        idx = rng.choice(len(bg), size=min(n_bg, len(bg)), replace=False)
        times = rng.uniform(0.0, V.RENDER_S - 0.35, size=len(idx))
        out.append((pos, np.asarray(idx, dtype=int), times))
    _BANK[key] = out
    return out


def _synth_for(model, cond, bg, gain):
    """A synth at the certified head pose, preloaded with this condition's
    contacts. Deterministic in `cond` — `_episode_synth`'s job without its RNG,
    so an arm cannot differ from another arm by its draw."""
    pos, idx, times = cond
    synth = CA.ContactAudioSynth(model)
    synth.set_listener(V.HEAD, V.HEAD_YAW)
    for k, i in enumerate(idx):
        e = bg[int(i)]
        az, lat, el, dist = synth.localize(e.pos)
        synth.events.append(CA.AudioEvent(
            t=float(times[k]), geom1=e.geom1, geom2=e.geom2,
            voiced_geom=e.voiced_geom, pos=e.pos, force=e.force,
            amp=e.amp * gain, azimuth=az, lateral=lat, elevation=el,
            distance=dist))
    return synth


def _hear(model, data, cond, bg, gain, action, noise_rng, mute=False):
    """Render one episode and return what the listener is allowed to know.

    `V._features` is imported, not copied: the listener's input is the same
    138-dim log-band spectrogram VO.01 certified, and nothing else reaches it.
    """
    synth = _synth_for(model, cond, bg, gain)
    synth.emit_voice(V.T_VOICE, cond[0], action, data=data)
    ear = V._ear(synth, noise_rng, mute_voice=mute)
    return V._features(ear, CA.SAMPLE_RATE)


# ═══════════════════════════════════════════════════════════════════════
# THE TWO JACKS — no shared parameters, no gradient through the air
# ═══════════════════════════════════════════════════════════════════════
class _Emitter(nn.Module):
    """referent -> a 4-D vocalisation. Gaussian policy, tanh-squashed into the
    action box `ContactAudio` declares."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed * 433494437 + 1)
        self.net = nn.Sequential(
            nn.Linear(N_STATES, EMIT_HIDDEN), nn.Tanh(),
            nn.Linear(EMIT_HIDDEN, CA.VOICE_ACTION_DIM))
        self.log_std = nn.Parameter(torch.full((CA.VOICE_ACTION_DIM,),
                                               LOGSTD_INIT))

    def dist(self, r_onehot):
        mu = torch.tanh(self.net(r_onehot))
        std = torch.exp(self.log_std.clamp(min=LOGSTD_MIN, max=0.0))
        return torch.distributions.Normal(mu, std)


class _Listener(nn.Module):
    """received waveform features -> one of N_ACTS. Sees no referent, no pose,
    no emission vector, no event times."""

    def __init__(self, seed: int, n_feat: int):
        super().__init__()
        torch.manual_seed(seed * 2654435761 + 2)
        self.net = nn.Sequential(
            nn.Linear(n_feat, LIST_HIDDEN), nn.ReLU(),
            nn.Linear(LIST_HIDDEN, LIST_HIDDEN), nn.ReLU(),
            nn.Linear(LIST_HIDDEN, N_ACTS))

    def logits(self, x):
        return self.net(x)


def _onehot(r):
    return torch.eye(N_STATES)[torch.as_tensor(r, dtype=torch.long)]


_ARM_SALT = {"trained": 101, "untrained": 211, "muted": 307, "scrambled": 401}


def _arm(seed: int, arm: str, n_updates: int = N_UPDATES,
         batch: int = BATCH) -> dict:
    """Train one pair and evaluate it. `arm` in {trained, untrained, muted,
    scrambled}.

    Everything except the arm's own defining difference is identical across
    arms: the pose bank, the room gain, the background events, the network
    initialisations (seeded from `seed`, not from the arm), the optimiser, the
    schedule and the number of episodes.
    """
    model, data = V._world(seed)
    bg = V._background(seed)
    gain = V._bg_gain(seed)
    mute = (arm == "muted")
    scramble = (arm == "scrambled")
    freeze_emitter = (arm == "untrained")

    bank = _pose_bank(seed, N_EVAL + N_CIC + n_updates * batch)
    train_bank = bank[N_EVAL + N_CIC:]
    eval_bank = bank[:N_EVAL]
    cic_bank = bank[N_EVAL:N_EVAL + N_CIC]

    # NOT `hash(arm)`: Python randomises string hashing per process unless
    # PYTHONHASHSEED is pinned, so the same (seed, arm) would draw different
    # referents in a re-run and the matched-bank design would be a fiction.
    rng = np.random.RandomState(seed * 999331 + _ARM_SALT[arm])
    noise_rng = np.random.RandomState(seed * 7 + 101)

    emit = _Emitter(seed)
    n_feat = len(V._features(np.zeros((2, int(V.RENDER_S * CA.SAMPLE_RATE))),
                             CA.SAMPLE_RATE))
    listen = _Listener(seed, n_feat)
    opt_e = torch.optim.Adam(emit.parameters(), lr=EMIT_LR)
    opt_l = torch.optim.Adam(listen.parameters(), lr=LIST_LR)

    # feature standardisation, fitted ONCE on a warm-up batch of this arm's own
    # renders and frozen: a running normaliser would leak the referent
    # distribution's drift into the listener's input.
    warm = []
    for i in range(min(128, len(train_bank))):
        a = np.clip(rng.uniform(-1, 1, CA.VOICE_ACTION_DIM), -1, 1)
        warm.append(_hear(model, data, train_bank[i], bg, gain, a,
                          noise_rng, mute))
    warm = np.array(warm)
    f_mu, f_sd = warm.mean(0), warm.std(0) + 1e-6

    base = 0.0
    ptr = 0
    hist = []
    for u in range(n_updates):
        frac = u / max(n_updates - 1, 1)
        ent_e = ENT_COEF_EMIT * (1.0 - (1.0 - ENT_ANNEAL) * frac)
        ent_l = ENT_COEF_LIST * (1.0 - (1.0 - ENT_ANNEAL) * frac)

        refs = rng.randint(0, N_STATES, size=batch)
        oh = _onehot(refs)
        d = emit.dist(oh)
        acts = d.sample()
        logp_e = d.log_prob(acts).sum(-1)
        a_np = acts.detach().numpy()

        conds = [train_bank[(ptr + k) % len(train_bank)] for k in range(batch)]
        ptr += batch
        feats = np.array([_hear(model, data, conds[k], bg, gain, a_np[k],
                                noise_rng, mute) for k in range(batch)])
        if scramble:
            # null (i): the listener receives a real, fully-rendered episode —
            # just not THIS one. Level, bearing, room and emission statistics
            # are preserved exactly; only the pairing is destroyed.
            feats = feats[rng.permutation(batch)]

        x = torch.as_tensor((feats - f_mu) / f_sd, dtype=torch.float32)
        lg = listen.logits(x)
        pd = torch.distributions.Categorical(logits=lg)
        choice = pd.sample()
        rew = (choice.numpy() == refs).astype(np.float64)
        base += BASELINE_BETA * (rew.mean() - base)
        adv = torch.as_tensor(rew - base, dtype=torch.float32)

        opt_l.zero_grad()
        (-(adv * pd.log_prob(choice)).mean()
         - ent_l * pd.entropy().mean()).backward()
        opt_l.step()

        if not freeze_emitter:
            opt_e.zero_grad()
            (-(adv * logp_e).mean() - ent_e * d.entropy().sum(-1).mean()
             ).backward()
            opt_e.step()

        hist.append(float(rew.mean()))

    out = {f"{arm}_train_tail": float(np.mean(hist[-20:]))}

    # ── evaluation: held-out poses the training loop never touched ──────
    with torch.no_grad():
        refs = rng.randint(0, N_STATES, size=N_EVAL)
        d = emit.dist(_onehot(refs))
        a_np = d.sample().numpy()
        feats = np.array([_hear(model, data, eval_bank[k], bg, gain, a_np[k],
                                noise_rng, mute) for k in range(N_EVAL)])
        if scramble:
            feats = feats[rng.permutation(N_EVAL)]
        x = torch.as_tensor((feats - f_mu) / f_sd, dtype=torch.float32)
        pd = torch.distributions.Categorical(logits=listen.logits(x))
        choice = pd.sample().numpy()
        out[f"{arm}_coord"] = float((choice == refs).mean())

        mi = _ear_mi(feats, refs, seed)
        for k, v in mi.items():
            out[f"{arm}_{k}"] = v

        # ── the interventional CIC ──────────────────────────────────────
        # One pose, one room, one background; only the sound changes. Position,
        # timing, distance and turn count cannot carry the referent here
        # because they are held FIXED across the intervention.
        mus = torch.tanh(emit.net(_onehot(np.arange(N_STATES)))).numpy()
        rows = np.empty((N_CIC, N_STATES, N_ACTS))
        for i, cond in enumerate(cic_bank):
            fs = np.array([_hear(model, data, cond, bg, gain, mus[r],
                                 noise_rng, mute) for r in range(N_STATES)])
            if scramble:
                # the scrambled pair's delivery is decoupled at the ear, so its
                # intervention is decoupled too — the same permutation applied
                # at the same place, never a different measurement.
                fs = fs[rng.permutation(N_STATES)]
            xi = torch.as_tensor((fs - f_mu) / f_sd, dtype=torch.float32)
            rows[i] = torch.softmax(listen.logits(xi), dim=-1).numpy()
        out[f"{arm}_cic"] = _cic(rows)
        p95, mean = _cic_floor(rows, np.random.RandomState(seed * 5387 + 19))
        out[f"{arm}_cic_perm_p95"] = p95
        out[f"{arm}_cic_perm_mean"] = mean
    return out


def _cic(rows: np.ndarray) -> float:
    """I(intervened referent ; act) in bits, averaged over poses.

    `rows[i, r, a]` = P(act = a | pose i, the emitter's call for referent r).
    The referent is intervened on uniformly, so p(r) = 1/N_STATES exactly and
    no marginal has to be estimated.
    """
    p_ar = rows / np.maximum(rows.sum(-1, keepdims=True), EPS)
    p_a = p_ar.mean(1, keepdims=True)
    t = p_ar * np.log2(np.maximum(p_ar, EPS) / np.maximum(p_a, EPS))
    return float(t.sum(-1).mean())


def _cic_floor(rows: np.ndarray, rng, n_perm=N_PERM) -> tuple:
    """The floor for `_cic`: response rows re-sampled ACROSS poses.

    Permuting within a pose is not a floor — `_cic` is symmetric in `r`, so it
    would return the same number. This resamples each (pose, referent) cell's
    response from a different pose, which destroys the within-pose sound->act
    dependence while leaving the marginal distribution of listener responses
    exactly as measured.
    """
    n_pose, n_r, _ = rows.shape
    flat = rows.reshape(n_pose * n_r, -1)
    vals = np.empty(n_perm)
    for i in range(n_perm):
        pick = rng.randint(0, len(flat), size=n_pose * n_r)
        vals[i] = _cic(flat[pick].reshape(n_pose, n_r, -1))
    return float(np.percentile(vals, 95)), float(np.mean(vals))


def _floor_selftest() -> dict:
    """Are the two permutation floors FLOORS, or invariances of their own
    statistics?

    A permutation null is only a null if the permutation destroys the
    dependence the statistic is sensitive to. `_cic` is symmetric in the
    referent index, so the obvious shuffle — permute the referent labels within
    a pose — leaves it EXACTLY UNCHANGED: measured 2.0000, "floor" 2.0000,
    collapse 0.0000, and `cic - floor >= CIC_MARGIN_BITS` becomes unsatisfiable
    by arithmetic. That is why `_cic_floor` resamples ACROSS poses instead, and
    this function is the proof that the distinction is real rather than a
    comment.

    Both floors are pointed at PLANTED PERFECT structure — referent == code;
    every referent driving its own act at every pose — where the answer is
    known to be log2(N_STATES) = 2 bits. Each floor must sit far below it. A
    future edit that quietly reintroduces an invariance fails here, in the
    ledger, on synthetic data, before any GPU-hour or any arm is run.

    Measured 2026-08-30: MI collapses 1.972 of 2.0; CIC collapses 0.617 of 2.0
    at `N_CIC` poses. The CIC floor is much the more conservative of the two —
    resampling confident one-hot responses across poses often lands a
    near-injective assignment, which reads as influence — and 0.617 is
    therefore the CEILING on any CIC margin this rig can ever ask for.
    """
    n = N_STATES
    refs = np.repeat(np.arange(n), 100)
    mi = _plugin_mi(refs, refs.copy(), n, MI_CODES)
    mi_p95, _ = _mi_floor(refs, refs.copy(), n, MI_CODES,
                          np.random.RandomState(11))
    rows = np.tile(np.eye(n)[None], (N_CIC, 1, 1))
    cic = _cic(rows)
    cic_p95, _ = _cic_floor(rows, np.random.RandomState(13))
    # the invariance that would have been the bug, measured rather than argued
    bad = np.stack([r[np.random.RandomState(17 + i).permutation(n)]
                    for i, r in enumerate(rows)])
    return {"mi_floor_collapse": float(mi - mi_p95),
            "cic_floor_collapse": float(cic - cic_p95),
            "cic_within_pose_collapse": float(cic - _cic(bad))}


# ═══════════════════════════════════════════════════════════════════════
# THE RIG'S OWN INSTRUMENTS
# ═══════════════════════════════════════════════════════════════════════
def _probe(seed: int) -> dict:
    """Is the channel alive IN THIS RIG?

    VO.01's certificate is about VO.01's episodes. Here, uniform-random
    emissions over THIS pose bank, a ridge probe on the received features,
    scored on held-out episodes — and its muted twin, which must not recover
    anything. An at-chance MI with a dead probe is an apparatus outcome; an
    at-chance MI with a live probe is a finding about signalling.
    """
    model, data = V._world(seed)
    bg = V._background(seed)
    gain = V._bg_gain(seed)
    bank = _pose_bank(seed, N_PROBE)
    rng = np.random.RandomState(seed * 314159 + 29)
    noise_rng = np.random.RandomState(seed * 271828 + 31)
    acts = rng.uniform(-1.0, 1.0, size=(N_PROBE, CA.VOICE_ACTION_DIM))
    out = {}
    for tag, mute in (("probe", False), ("probe_mute", True)):
        f = np.array([_hear(model, data, bank[k], bg, gain, acts[k],
                            noise_rng, mute) for k in range(N_PROBE)])
        h = int(0.75 * N_PROBE)
        r2 = V._ridge_r2(f[:h], acts[:h], f[h:], acts[h:])
        out[f"{tag}_r2_mean"] = float(np.mean(r2))
        out[f"{tag}_r2_max"] = float(np.max(r2))
    return out


def _level(seed: int) -> dict:
    """The achieved voice-to-room level over THIS rig's pose bank.

    Not read off `_bg_gain`'s own calibration episodes, which would re-derive
    the formula and could only fire on an edit (VO.01's tripwire lesson). This
    is a measurement on the distribution the claim is actually scored over: if
    this rig's poses differ from the calibration's, the two-sided gate says so.
    """
    model, data = V._world(seed)
    bg = V._background(seed)
    gain = V._bg_gain(seed)
    bank = _pose_bank(seed, N_LEVEL)
    rng = np.random.RandomState(seed * 6700417 + 37)
    sr = CA.SAMPLE_RATE
    v, b = [], []
    for k in range(N_LEVEL):
        cond = bank[k]
        room = _synth_for(model, cond, bg, gain)
        b.append(V._rms(V._ear(room, None, mute_voice=True), sr))
        alone = CA.ContactAudioSynth(model)
        alone.set_listener(V.HEAD, V.HEAD_YAW)
        alone.emit_voice(V.T_VOICE, cond[0],
                         rng.uniform(-1.0, 1.0, CA.VOICE_ACTION_DIM), data=data)
        v.append(V._rms(V._ear(alone, None), sr))
    return {"voice_to_background_db":
            float(20.0 * math.log10(max(np.mean(v), EPS)
                                    / max(np.mean(b), EPS)))}


# ═══════════════════════════════════════════════════════════════════════
# THE EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════
def _experiment(seed: int) -> dict:
    m = {}
    m.update(_floor_selftest())        # are the floors floors, or invariances?
    m.update(_urn_game(seed))          # the estimator's known-answer test, first
    m.update(_probe(seed))             # the channel is alive in this rig
    m.update(_level(seed))             # ...at a declared difficulty
    m.update(_arm(seed, "trained"))    # the claim
    m.update(_arm(seed, "untrained"))  # null (ii)
    m.update(_arm(seed, "muted"))      # null (iii)
    # Restate the claim arm's numbers under the neutral names `_claim` reads,
    # so the predicate is blind to which arm produced them.
    for k in ("coord", "mi_ear", "mi_perm_p95", "cic", "cic_perm_p95"):
        m[k] = m[f"trained_{k}"]
    return m


def _control(seed: int) -> dict:
    """Null (i), and this spec's declared control: SCRAMBLED MESSAGES.

    It returns the SAME keys as `_experiment`'s claim arm and is scored by the
    SAME `_claim` predicate. If a scrambled pair clears the bar, the bar is
    measuring the room, the pose distribution or the listener's prior — not the
    signal — and this spec has no business recording a PASS.
    """
    m = _arm(seed, "scrambled")
    for k in ("coord", "mi_ear", "mi_perm_p95", "cic", "cic_perm_p95"):
        m[k] = m[f"scrambled_{k}"]
    return m


# ═══════════════════════════════════════════════════════════════════════
# THE PRE-REGISTERED CHECK
# ═══════════════════════════════════════════════════════════════════════
def _worst_lo(d, key):
    return d[key] - SEED_SPREAD_FACTOR * d.get(key + "_std", 0.0)


def _worst_hi(d, key):
    return d[key] + SEED_SPREAD_FACTOR * d.get(key + "_std", 0.0)


def _coord_ok(d) -> bool:
    """The coordination conjunct, on the worst seed and at >= 3 sigma."""
    std = d.get("coord_std", 0.0)
    t = (d["coord"] - CHANCE) * math.sqrt(3.0) / max(std, 1e-9)
    return bool(_worst_lo(d, "coord") >= COORD_MIN
                and _worst_lo(d, "coord") - CHANCE >= COORD_MARGIN
                and t >= COORD_TSTAT_MIN)


def _mi_ok(d) -> bool:
    """The information conjunct: MI at the EAR above its own permutation
    floor, and causal influence above its own. Both differences are taken on
    the worst seed."""
    mi = _worst_lo(d, "mi_ear") - _worst_hi(d, "mi_perm_p95")
    ci = _worst_lo(d, "cic") - _worst_hi(d, "cic_perm_p95")
    return bool(mi >= MI_MARGIN_BITS and ci >= CIC_MARGIN_BITS)


def _claim(d) -> bool:
    """The whole claim, in one predicate that does not know which arm it is
    looking at. Applied unchanged to the trained arm, to the untrained null and
    to the scrambled control."""
    return _coord_ok(d) and _mi_ok(d)


def _check(m: dict, c: dict):
    # ── RIG: an apparatus that cannot measure signalling must never record a
    # verdict about signalling. Each of these VOIDs, and none of them can fire
    # this spec's `kills` field.
    est_ok = (m["urn_success"] >= URN_SUCCESS_MIN
              and m["urn_mi"] >= URN_MI_MIN
              and m["urn_mi_scram"] <= URN_MI_SCRAM_MAX
              # ...and the floors those readings are gated against are floors,
              # not invariances of their own statistics. Without this, an
              # at-floor MI cannot be told apart from a floor that cannot move.
              and m["mi_floor_collapse"] >= MI_FLOOR_COLLAPSE_MIN
              and m["cic_floor_collapse"] >= CIC_FLOOR_COLLAPSE_MIN)
    chan_ok = (_worst_lo(m, "probe_r2_mean") >= PROBE_R2_MIN
               and _worst_hi(m, "probe_mute_r2_max") <= PROBE_MUTE_MAX)
    level_ok = abs(m["voice_to_background_db"] - V.SIR_TARGET_DB) <= V.SIR_TOL_DB
    # A muted pair above chance means the referent reaches the listener outside
    # the channel. That is a leak in the rig, not a refutation of signalling.
    leak_ok = _worst_hi(m, "muted_coord") <= CHANCE + CHANCE_TOL
    if not (est_ok and chan_ok and level_ok and leak_ok):
        return Status.VOID

    # ── NULLS: each must FAIL the identical predicate.
    if _claim({k[len("untrained_"):]: v for k, v in m.items()
               if k.startswith("untrained_")} | {"coord_std": m.get(
                   "untrained_coord_std", 0.0)}):
        return False
    if _claim(c):
        return False

    # ── THE CLAIM, both conjuncts. The registry's named false positive —
    # coordination without information at the ear — is a FAIL, and
    # `coord_without_mi` in the record says which branch fired.
    return bool(_coord_ok(m) and _mi_ok(m))


def _dry() -> list:
    """The verdict table. It proves the LOGIC of `_check`, and nothing about
    whether any row is reachable — LESSONS.md, 2026-08-29: *a `_dry()` table
    proves the verdict logic; it cannot prove the rows mean anything.* The
    thing that proves the rows mean anything is the pilot, which runs this
    whole pipeline at toy size.
    """
    def base(**kw):
        d = {"urn_success": 1.0, "urn_mi": 1.0, "urn_mi_scram": 0.0,
             "mi_floor_collapse": 1.97, "cic_floor_collapse": 0.62,
             "probe_r2_mean": 0.8, "probe_mute_r2_max": 0.0,
             "voice_to_background_db": V.SIR_TARGET_DB,
             "muted_coord": CHANCE, "coord": 0.9, "coord_std": 0.01,
             "mi_ear": 1.2, "mi_perm_p95": 0.1, "cic": 0.8,
             "cic_perm_p95": 0.1}
        for k in ("coord", "mi_ear", "mi_perm_p95", "cic", "cic_perm_p95"):
            d.setdefault("untrained_" + k, CHANCE if k == "coord" else 0.05)
        d["untrained_mi_perm_p95"] = 0.1
        d["untrained_cic_perm_p95"] = 0.1
        d.update(kw)
        return d

    ctl = {"coord": CHANCE, "coord_std": 0.01, "mi_ear": 0.09,
           "mi_perm_p95": 0.1, "cic": 0.05, "cic_perm_p95": 0.1}
    rows = [
        ("all green", base(), ctl, True),
        ("urn check dead -> VOID", base(urn_mi=0.2), ctl, Status.VOID),
        ("estimator sees a scrambled system -> VOID",
         base(urn_mi_scram=0.5), ctl, Status.VOID),
        # the defect this file's `_floor_selftest` exists to make impossible:
        # a "floor" that is an invariance of its own statistic reads 0.0
        # collapse, and every gate above it is unsatisfiable by arithmetic.
        ("CIC floor is an invariance, not a floor -> VOID",
         base(cic_floor_collapse=0.0), ctl, Status.VOID),
        ("MI floor cannot move -> VOID", base(mi_floor_collapse=0.2), ctl,
         Status.VOID),
        ("channel dead in rig -> VOID", base(probe_r2_mean=0.1), ctl,
         Status.VOID),
        ("probe reads the room -> VOID", base(probe_mute_r2_max=0.4), ctl,
         Status.VOID),
        ("room drowns the voice -> VOID",
         base(voice_to_background_db=V.SIR_TARGET_DB - 9), ctl, Status.VOID),
        ("muted pair solves it -> VOID", base(muted_coord=0.9), ctl,
         Status.VOID),
        ("THE named false positive: coord high, MI at floor -> FAIL",
         base(mi_ear=0.11, cic=0.06), ctl, False),
        ("coordination at chance -> FAIL", base(coord=CHANCE), ctl, False),
        ("scrambled control also clears -> FAIL", base(),
         {"coord": 0.9, "coord_std": 0.01, "mi_ear": 1.2,
          "mi_perm_p95": 0.1, "cic": 0.8, "cic_perm_p95": 0.1}, False),
        ("untrained emitter also clears -> FAIL",
         base(untrained_coord=0.9, untrained_coord_std=0.01,
              untrained_mi_ear=1.2, untrained_cic=0.8), ctl, False),
    ]
    out = []
    for name, m, c, want in rows:
        got = _check(m, c)
        out.append((name, got, want, got == want))
    return out


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "VO.02 gates are provisional — run the pilot, freeze the bars in "
            "this file from its RIG arms, then run (SM.02's _GATES_FROZEN "
            "idiom). Nothing here may be fitted to the trained arm.")
    return run_spec(BY_ID["VO.02"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    ok = True
    for name, got, want, good in _dry():
        ok &= good
        print(f"  [{'ok' if good else 'XX'}] {name}: got={got} want={want}")
    print("dry table:", "PASS" if ok else "FAIL")
