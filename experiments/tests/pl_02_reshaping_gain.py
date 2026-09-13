"""PL.02 — The RESHAPING test: does another sense change what an encoder computes?

THE CLAIM. A vision encoder trained JOINTLY with audio by cross-modal masked
prediction outperforms an A-only encoder of matched capacity WHEN BOTH ARE
EVALUATED ON VISION ALONE. The reshaping gain

    R = perf(M_AB | A only) - perf(U_A)

is positive, paired by seed, bootstrap CI excluding zero. This is the M3L
signature (arXiv:2311.00924) made into a metric, and since the PLASTIC-ONLY
decree (2026-08-09) it measures what the plastic path BUYS — it is the
decree's sole registered falsifier. A null R does not restore freezing (the
owner decreed the ENDS); it removes the arithmetic argument that has been
carried as if it were a measurement.

THE MODALITY PAIR, and why it is this one. A = the playground eye — PG.6's
certified protocol (object radius recoverable at R^2 >= 0.80 from raw pixels,
occlusion-rejected sampling, the canary/GL discipline), rendered at 64 px
under the ADOPTED eye quality (`experiments/eye_quality.py`, the PL.00
renderer bakeoff's winner — new visual work opts in; existing certificates
were not migrated). B = contact audio from `ContactAudio.py`'s modal family:
a struck object rings at f0 = clip(180 / char_size, 80, 4000) Hz
(`ContactAudioSynth.fundamental`), so over the probe band r in [0.06, 0.18] m
the fundamental spans 1000-3000 Hz and audio is PHYSICALLY monotone in the
attribute the probe reads. Stereo pan carries bearing, 1/dist carries range.
That is the teaching channel: sound knows the radius; does hearing it during
training change what the eye's encoder computes about radius afterwards?

perf(.) IS PRE-REGISTERED AS: ridge-probe (l2 = 1.0, PG.6's operating point)
R^2 for object RADIUS from the encoder's 64-d features, on held-out episodes
never seen in pretraining, VISION ALONE at test (the A-encoder never takes
audio as input in any arm — fusion lives in the decoder heads, so "evaluated
on A alone" is structural, not a masking convention). Bearing gain is
reported as a diagnostic and is NOT scored.

THE ARMS. One A-encoder architecture, one per-seed init shared by every arm
(same torch seed -> byte-identical initial tensors), matched optimiser steps,
matched batch:

    U_A       masked autoencoding on frames only. The A-only baseline.
    PLASTIC   joint cross-modal masked prediction: masked frame + (dropped
              half the time) audio; reconstruct masked pixels AND predict the
              audio features from the joint latent on audio-dropped rows.
    FROZEN    the registered null: M_AB whose A-encoder IS U_A's trained
              tensor, frozen — heads and audio encoder train, the A-features
              cannot move, so R = 0 EXACTLY, by construction. Scored and
              ineligible, never excluded. The harness ASSERTS the arithmetic
              (max |feature diff| < FEAT_TOL, |R_frozen| < FROZEN_TOL) and
              returns VOID if its own pairing is broken.
    SHUFFLED  the declared control: identical to PLASTIC but the audio drawn
              from a DIFFERENT episode (a fixed derangement of the pretext
              set) — correspondence destroyed, marginals and temporal
              statistics preserved. R must collapse to ~0. If shuffled-B
              reshapes A just as well, the gain is capacity or
              regularisation, not binding, and the run is VOID.

VERDICT ARITHMETIC, pre-registered (house worst-seed idiom):
    claim   PASS iff, on EVERY seed, the paired bootstrap 95% CI of
            R_plastic (resample the test episodes, recompute both probes'
            R^2 on the SAME resample, difference) lies above zero.
    control VOID (any seed) iff shuf_R's CI excludes zero from above AND
            shuf_R >= CONTROL_COLLAPSE_FRAC * plastic_R — "reshapes just as
            well" operationalised at half the plastic gain, chosen
            conservative in the refusing direction: a control that VOIDs a
            true claim costs a re-design; a control that waves through a
            capacity artifact costs a false certificate.
    null    VOID iff the frozen arm's R is not exactly zero (see FROZEN).

RIG GATES, each VOID (an invalid run is not evidence): the GL canary must
not move (frame-sum, PG.6's discipline); every trained arm's pretext loss
must fall below LEARN_DROP x its initial value (an arm that never learned
tests nothing); the audio channel itself must know the radius (ridge from
audio features -> radius R^2 >= AUDIO_TEACH_R2_MIN on the held-out set —
if the teacher is ignorant the premise is dead, not refuted); the probe
instrument must be alive (shuffled-label probe R^2 <= SHUF_LABEL_R2_MAX);
feature extraction must be deterministic (encode twice, identical); AND
the eye itself must still carry the attribute (82nd audit B4, 2026-09-07):
absolute radius R^2 >= EYE_RADIUS_R2_MIN, PG.6's own 0.80 bar, on the
coarse eye that PG.6's certificate predates. R is a difference, so a
blinded eye collapses both arms together and a null R would be
indistinguishable from a dead channel — on the PLASTIC-ONLY decree's sole
registered falsifier, that ambiguity must VOID, not FAIL.

THAT GATE READS THE RAW-PIXEL RIDGE, NOT U_A's FEATURES (Review DAILY
2026-09-11, `5e39771`, row `pl02-eye-gate-reads-the-encoder-not-the-eye`;
implemented by the builder 2026-09-12). `r2_raw_pixel` is a ridge at the
SAME operating point (l2 = L2) from the flattened RGB frames of THIS run's
own probe split to radius — PG.6's certified quantity, re-measured here
rather than inherited from the seed-90 probe. `EYE_RADIUS_R2_MIN` is
unmoved at 0.80 and the VOID semantics are unmoved; only the referent
changed, and `r2_ua` is NOT deleted — it stays a first-class recorded
metric on the row, which was the 82nd audit B4's other half.

WHY THE REFERENT MOVED, and the reason is algebraic rather than
interpretive, so it holds whichever way the verdict falls: **`r2_ua` is the
SUBTRAHEND in the claim's own effect size.** The spec computes
`R_pl = r2_pl - r2_ua`. A VOID gate requiring `r2_ua >= 0.80` therefore
requires the baseline to be near-saturated BEFORE the run is allowed to
count, which caps the largest reshaping gain the spec can ever report at
<= 0.20 — against an observed gain of 0.94 (SMOKE RECORD 2). As lettered
the gate does not test whether the eye is alive; it algebraically
suppresses the quantity it was added to guard, and it is un-clearable by
construction in exactly the regime the claim exists to test (audio
rescuing a weak encoder, where a weak `r2_ua` is the PREMISE and not the
fault). Under the T1.02 precedent that makes the EXPERIMENT wrong, which
is the only ground on which a gate may be re-aimed.

BOTH HOLES THE LETTERED GATE COVERED ARE CLOSED BY INSTRUMENTS THAT
ALREADY EXIST, so no conjunct is added: (a) dead channel — B4's actual
worry — is covered by the new referent, since a blind eye cannot produce a
0.93 raw-pixel ridge; (b) audio leaking into `r2_pl` is covered by the
spec's own declared SHUFFLED control, measured clean on the RGB@64 smoke
(shuffled_R -0.002328, CI excluding zero from BELOW, control_reshapes_too
0). Inventing a redundant control would be manufacturing rigour rather
than adding it.

THE NEW GATE IS LIVE, AND THAT IS SHOWN BY EXHIBITING IT FIRING rather
than by its currently passing (the 2026-09-12 lesson: a passing control
may be correct or unreachable, and those look identical from outside).
Replayed through `_check` on the smoke's own metric dict with
`r2_raw_pixel` set to the decomposition's MEASURED grey@64 ceiling
0.5614 -> VOID; at RGB@64's measured 0.9327 -> the branch is not taken and
the check proceeds. The two numbers are the same eye at the same
resolution differing only in the chromatic channel, so the gate's live
range is a real operating point of this rig and not a hypothetical. No
earlier branch reads `r2_raw_pixel`, so the branch is not dominated; it is
placed where the lettered gate stood, AFTER the teacher gate and BEFORE
the probe-leak gate, because an ignorant teacher is a deader premise than
a blind eye and should be the one reported.

OPERATING POINT, fixed a priori and validated for RIG ALIVENESS ONLY on
seed 90 (disjoint from the registered seeds 0/1/2) before the registered
run — not tuned to the verdict, which has no free threshold to tune (the
claim bar is "CI excludes zero", from the registry):
    64 px RGB frames; N_PRETEXT=1500; probe split 1000/600; 1200 Adam
    steps at 1e-3, batch 64; patch masking 8x8 patches at 35%; audio
    dropped on 50% of rows; audio features: 12 log band energies
    (800-3600 Hz, log-spaced) + log level + pan, z-scored on pretext
    stats.

SMOKE RECORD 1 (seed 90, 64 px GREY, ran 2026-09-07T13:09:58Z detached,
/data/pl02_smoke_seed90.log, 879 s billed): check -> VOID — the rig was
not alive at the first operating point. Two gates fired, honestly:
  - r2_ua -0.0039 vs EYE_RADIUS_R2_MIN 0.80 (the 82nd-audit B4 gate,
    firing on its first exercise). r2_frozen == r2_ua exactly and
    r2_plastic -0.0040 — all three probes equally dead — while
    loss_drop_ua 0.0083 says reconstruction was aced.
  - learn_ok 0: the plastic arm's combined loss ROSE (ratio 1.104) —
    shared root: predicting audio from a radius-blind latent is
    unlearnable, so the audio term cannot fall.
Gates that read clean: audio teacher 0.9997, canary, determinism,
shuffled-label probe 6e-5, frozen arithmetic exact.

THE DECOMPOSITION (pl02_rig_probe.py, seed 90, artifacts
/data/pl02_rig_probe.json + /data/pl02_rgb_probe.json +
/data/pl02_uargb_probe.json), per the PL.00 lesson — decompose before
blaming the substrate. Raw-pixel ridge ceilings at the coarse quality:
grey@64 0.5614, grey@96 0.6861, RGB@64 0.9327, RGB@96 0.9438 (spec's own
1000/600 split for RGB). PG.6's 0.80 certificate is an RGB raw-pixel
number; the first operating point silently discarded the chromatic
channel that carries most of the radius signal, capping the eye gate
below its own bar before any encoder ran — the same inherited-default
shape as PL.00's shadow pass, one layer up: nobody CHOSE grey, it
arrived as a .mean(axis=2) convenience. RGB restores the certified
channel at unchanged resolution and render cost; the switch is a rig
repair validated on the disjoint smoke seed, and no verdict threshold
moved in any direction.

SMOKE RECORD 2 (seed 90, 64 px RGB, ran 2026-09-07T14:40:47Z detached,
/data/pl02_smoke_rgb_seed90.log, 1011.56 s billed): check -> VOID — ONE
gate fired, and it is the routed one. r2_ua -0.001685 vs
EYE_RADIUS_R2_MIN 0.80 (the 82nd-audit B4 gate as lettered). Every other
instrument was alive and green: r2_plastic 0.941133 vs r2_frozen
-0.001685 (frozen arithmetic exact-zero intact), reshaping_gain_R
0.942818 with CI [0.930903, 0.960186] above zero, learn_ok 1 (plastic
pretext loss fell to 0.4751 of initial — SMOKE 1's rising-loss fault is
gone with the chromatic channel restored), audio teacher 0.999658,
canary/determinism clean, shuffled-label probe 8.2e-5, control clean
(shuffled_R -0.002328, CI [-0.003224, -0.001415] excludes zero from
BELOW; control_reshapes_too 0). This is the exact scenario the routing
priced: the gate-as-lettered VOIDs a run in which the claim's own
instruments measured a live reshaping gain of 0.94, because U_A's
masked-AE bottleneck never encodes radius at any tested budget (steps
probe COMPLETE, /data/pl02_steps_probe.json, checkpoints 1200..6000:
feat R^2 0.0192 / -0.0007 / -0.0035 / -0.0033 / -0.0032 while pretext
loss keeps falling — saturated, not data-starved) while the raw eye
reads 0.93 on the same episodes. The registered run stays blocked until
a smoke PASSES and `pl02-eye-gate-reads-the-encoder-not-the-eye`
(DUE 2026-09-09) rules what the gate reads. Nothing in this record moves
a threshold; the gate is untouched.

SMOKE RECORD 3 (seed 90, 64 px RGB, UNDER THE RE-AIMED GATE, ran
2026-09-12T23:11:16Z detached, /data/pl02_smoke_rawgate_seed90.log,
~1100 s): check -> **PASS**. The rig is alive and the registered run is
unblocked. `r2_raw_pixel` **0.924963** vs EYE_RADIUS_R2_MIN 0.80 — measured
on THIS run's own probe split, and 0.008 below the decomposition's
independently-sampled RGB@64 ceiling of 0.9327, which is the agreement two
different draws of the same sampler should show.

AND THE EDIT IS PROVED INERT ON EVERYTHING ELSE, which is the part worth
keeping: every metric SMOKE RECORD 2 published reproduces to the last
digit — reshaping_gain_R 0.942818, CI [0.930903, 0.960186], r2_plastic
0.941133, r2_ua -0.001685 (UNCHANGED, still recorded, no longer gating),
r2_frozen -0.001685, audio 0.999658, shuffled_label 8.2e-05, canary and
determinism clean, control shuffled_R -0.002328 with CI excluding zero from
BELOW and control_reshapes_too 0. The only new number is the one the ruling
added. `loss_drop_ua` 0.0041.

Housekeeping, disclosed: the process printed `XIO: fatal IO error 22` on
the shared X display AFTER `check -> True` was written — a teardown-order
artifact of releasing renderers at exit, not a render fault. Any frame
corruption from a poisoned display is what `canary_ok` exists to catch and
it read 1 on this run (PG.6's discipline, and the reason the renderers are
held for the process lifetime in the first place).

REGISTERED RUN, ATTEMPT 1 — **VOID** (ran 2026-09-13T00:17:57, commit
`c150187`, 2936.14 s CPU, seeds 0/1/2, peak RSS 1345.1 MB). The re-aimed eye
gate CLEARED on the registered seeds — `r2_raw_pixel` **0.929242 ± 0.003954**
vs 0.80, a third draw agreeing with the smoke's 0.924963 and the
decomposition's 0.9327 — and the run then died on the FIRST conjunct of
`_check`, the LEARN gate:

    learn_ok          0.666667 ± 0.471405   -> [1,1,0]
    shuffled_learn_ok 0.666667 ± 0.471405   -> [1,1,0]

Everything else read green and the claim's own instruments read high —
`reshaping_gain_R` 0.954619 with CI [0.943936, 0.971919] above zero and
`claim_ci_above_zero` 1.0 on every seed, `r2_plastic` 0.954440 vs `r2_ua`
-0.000179, `frozen_exact_zero` 1.0, control `shuffled_R` -0.004332 with its
CI excluding zero from BELOW and `control_reshapes_too` 0, canary 1,
`det_drift` 0.0, audio teacher 0.999616, `shuffled_label_r2` 7.3e-05. **None
of that is claimable: the run is VOID, and a VOID is not a near-miss PASS.**

WHICH ARM MISSED — deduced, because attempt 1's row could not say. `learn_ok`
is `all(last < LEARN_DROP*first)` over (U_A, PLASTIC, FROZEN) and only two of
those three ratios were recorded. At n=3 the worst seed obeys
|x-mu| <= sigma*sqrt(2), so `loss_drop_plastic` 0.4033 ± 0.0769 (worst
admissible 0.5120) and `loss_drop_ua` 0.0085 ± 0.0030 (worst 0.0127) both
clear 0.90 on EVERY seed. The implicated arms are therefore FROZEN and
SHUFFLED — precisely the two whose ratios were not emitted. Which seed, and
whether it is the same seed in both, is unknowable from that row.

**THE REPAIR IS DISCLOSURE, NOT A THRESHOLD** (builder, 2026-09-13, same shape
as the 09-12 `LG.03` ruling on `planner_calib_reach`: emit the number the run
already computes and throws away). `loss_drop_frozen` and `loss_drop_shuffled`
are now recorded. `LEARN_DROP` stays **0.90**, `learn_ok` and
`shuffled_learn_ok` are unchanged in definition and in effect, no verdict
threshold moves in either direction, and **attempt 2 is predicted VOID with
every attempt-1 number reproduced to the last digit** — `det_drift` 0.0 says
this rig is deterministic, and the two added metrics consume no RNG and enter
no gate. That prediction is the point: this re-run cannot be a run-until-pass,
because nothing in it can change the verdict. What it buys is an ATTRIBUTED
VOID instead of an anonymous one, and it discharges the staleness this edit
creates rather than leaving it for someone else.

Why the attribution matters more than it looks: FROZEN is the registered null
whose `R` is **zero by construction**, so its contribution to the verdict does
not depend on its loss having fallen — while its membership in `learn_ok` can
void the whole run. Whether that is the right membership is a GATE question
and it is NOT decided here; it is routed with the numbers attached once
attempt 2 says which arm and which seed.

WEIGHTS ARE PERSISTED — the 2026-09-07 standing rule (PROGRESS item 5):
PL.02 is an arena of the Vision-encoder seat (`experiments/champions.py`),
so every trained A-encoder's state_dict is written to
`experiments/artifacts/pl02_encoders_seed{n}.pt` (gitignored, on-box) and
the row records the path. Any future question about these encoders costs a
file read, not a retrain.

REUSE, declared: PG.6's geometry, occlusion-rejected sampler, ridge and
canary discipline are imported from `pg_6_playground_eyes.py` (declared in
IMPL_DEPS — the sibling-helper edge the 81st audit ordered tracked; the
transitive walker sees it as of `ded4219`). The eye here applies the
adopted coarse quality, which PG.6's certificate predates, so PG.6's _Eye
is not reused directly — the world contract (EYE_POS/EYE_XYAXES/EYE_FOVY)
is identical and comes from `playground.py` either way.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np

# ensure_gl() must precede the mujoco import (experiments/render.py: GLX
# under Xvfb; no libEGL/libOSMesa on this box).
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import playground as pg  # noqa: E402

from ContactAudio import (MODE_GAINS, MODE_RATIOS, SAMPLE_RATE, TAU0,  # noqa: E402
                          VOICE_SECONDS, ContactAudioSynth)
from ..eye_quality import apply_eye_quality  # noqa: E402
from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402
from . import pg_6_playground_eyes as pg6  # noqa: E402
from .pg_6_playground_eyes import (IN_FOV_MAX, _r2, _Ridge,  # noqa: E402
                                   _sample_unoccluded)

SPEC_ID = "PL.02"
IMPL_DEPS = ["playground.py", "ContactAudio.py",
             "experiments/eye_quality.py",
             "experiments/tests/pg_6_playground_eyes.py"]

RES = 64
N_PRETEXT = 1500
N_PROBE_TR, N_PROBE_TE = 1000, 600
STEPS, BATCH, LR = 1200, 64, 1e-3
Z_A, Z_B = 64, 32
PATCH, MASK_FRAC = 8, 0.35
AUDIO_DROP = 0.5
L2 = 1.0                       # PG.6's ridge operating point, unchanged
N_BOOT = 2000
CI_LO, CI_HI = 2.5, 97.5

LEARN_DROP = 0.90              # final pretext loss must be < this x initial
AUDIO_TEACH_R2_MIN = 0.50      # the teaching channel must know the radius
EYE_RADIUS_R2_MIN = 0.80       # PG.6's bar: the eye must carry the radius
SHUF_LABEL_R2_MAX = 0.10       # a probe that fits shuffled labels leaks
CONTROL_COLLAPSE_FRAC = 0.50   # "reshapes just as well" = half the gain
FROZEN_TOL = 1e-9
FEAT_TOL = 1e-5

N_BANDS = 12
BAND_LO_HZ, BAND_HI_HZ = 800.0, 3600.0

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def _torch():
    import torch
    torch.set_num_threads(2)   # four shared cores, paying tenants beside us
    return torch


# ── the eye, at the adopted quality ──────────────────────────────────────
_EYES: dict = {}


class _CoarseEye(pg6._Eye):
    """PG.6's eye with the PL.00-adopted render quality applied.

    The construction lines are repeated rather than inherited because
    `apply_eye_quality` must run BEFORE `mujoco.Renderer` allocates its
    offscreen framebuffers, and pg6._Eye does both inside one __init__.
    Everything behavioural (place/frame/truth/unoccluded/canary) is
    inherited, so the two eyes cannot drift in what they measure — only in
    what the render costs and which GL passes it pays for.
    """

    def __init__(self, seed: int, res: int = RES):
        params = pg.PlaygroundParams(seed=seed, n_objects=0)
        self.model, self.data, _ = pg.make_playground(
            params, with_water=False,
            probe_objects=(("probe0", 0.0, 0.0, 0.10),))
        apply_eye_quality(self.model)          # the one divergence from pg6
        self.gid = self.model.geom("probe0").id
        self.bid = self.model.body("probe0").id
        self.qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.bid]]
        self.r = mujoco.Renderer(self.model, height=res, width=res)
        self._canary = None
        self._canary = self.canary()


def get_eye(seed: int) -> _CoarseEye:
    # Held for the process lifetime: a GC'd Renderer poisons the shared X
    # display and the NEXT renderer returns plausible corrupt frames (PG.6).
    if seed not in _EYES:
        _EYES[seed] = _CoarseEye(seed)
    return _EYES[seed]


# ── the audio channel ────────────────────────────────────────────────────
def _f0_of_radius(eye: _CoarseEye, radius: float) -> float:
    """The fundamental ContactAudio would assign this geom at this radius.

    Read THROUGH ContactAudioSynth rather than re-deriving 180/r here, so if
    the modal family's size->pitch law ever changes, this spec follows it
    (ContactAudio.py is in IMPL_DEPS; a change stales this certificate)."""
    eye.model.geom_size[eye.gid, 0] = radius
    synth = ContactAudioSynth(eye.model)
    return synth.fundamental(eye.gid)


def _ring(f0: float) -> np.ndarray:
    """Mono modal ring, ContactAudio's synthesis family (free-bar partials)."""
    n = int(VOICE_SECONDS * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    sig = np.zeros(n)
    total = 0.0
    for ratio, gain in zip(MODE_RATIOS, MODE_GAINS):
        f = f0 * ratio
        if f >= 0.45 * SAMPLE_RATE:
            break
        sig += gain * np.exp(-t / (TAU0 / ratio)) * np.sin(2 * math.pi * f * t)
        total += gain
    return sig / max(total, 1e-12)


_BAND_EDGES = np.geomspace(BAND_LO_HZ, BAND_HI_HZ, N_BANDS + 1)


def _audio_feats(eye: _CoarseEye, radius: float, bearing_deg: float,
                 dist: float) -> np.ndarray:
    """14-d: 12 log band energies + log level + pan. Deterministic physics."""
    sig = _ring(_f0_of_radius(eye, radius))
    p = math.sin(math.radians(bearing_deg))          # pan toward the right
    gl_, gr_ = math.sqrt((1.0 - p) / 2.0), math.sqrt((1.0 + p) / 2.0)
    g = 1.0 / max(dist, 0.1)
    left, right = gl_ * g * sig, gr_ * g * sig
    mid = 0.5 * (left + right)
    spec = np.abs(np.fft.rfft(mid)) ** 2
    freqs = np.fft.rfftfreq(len(mid), 1.0 / SAMPLE_RATE)
    bands = np.empty(N_BANDS)
    for i in range(N_BANDS):
        m = (freqs >= _BAND_EDGES[i]) & (freqs < _BAND_EDGES[i + 1])
        bands[i] = math.log(float(spec[m].sum()) + 1e-12)
    el, er = float((left ** 2).sum()), float((right ** 2).sum())
    level = math.log(el + er + 1e-12)
    pan = (er - el) / max(er + el, 1e-12)
    return np.concatenate([bands, [level, pan]]).astype(np.float32)


# ── episodes ─────────────────────────────────────────────────────────────
def _episodes(eye: _CoarseEye, rng: np.random.RandomState, n: int) -> dict:
    frames = np.empty((n, RES, RES, 3), dtype=np.float32)
    radii = np.empty(n, dtype=np.float32)
    bearings = np.empty(n, dtype=np.float32)
    dists = np.empty(n, dtype=np.float32)
    audio = np.empty((n, N_BANDS + 2), dtype=np.float32)
    for i in range(n):
        b, d, r, _ = _sample_unoccluded(eye, rng, 0.0, IN_FOV_MAX, signed=True)
        frames[i] = eye.frame(b, d, r)               # RGB — see docstring
        radii[i], bearings[i], dists[i] = r, b, d
        audio[i] = _audio_feats(eye, r, b, d)
    return {"frames": frames, "radius": radii, "bearing": bearings,
            "dist": dists, "audio": audio}


def _nchw(torch, frames: np.ndarray):
    """(n, RES, RES, 3) float32 -> contiguous NCHW tensor."""
    return torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2)))


# ── models ───────────────────────────────────────────────────────────────
def _build_models(torch, seed: int):
    """One init per seed, shared by every arm: torch is reseeded to the same
    value before each construction, so the arms' initial A-encoders are
    byte-identical and R is paired by construction, not by hope."""
    nn = torch.nn
    torch.manual_seed(seed * 1009 + 7)

    side = RES // 16                # four stride-2 convs
    feat = 64 * side * side
    enc_a = nn.Sequential(
        nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(feat, Z_A))
    enc_b = nn.Sequential(nn.Linear(N_BANDS + 2, 64), nn.ReLU(),
                          nn.Linear(64, Z_B))
    dec = nn.Sequential(
        nn.Linear(Z_A + Z_B, feat), nn.ReLU(),
        nn.Unflatten(1, (64, side, side)),
        nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(16, 3, 4, 2, 1))
    aud_head = nn.Sequential(nn.Linear(Z_A + Z_B, 64), nn.ReLU(),
                             nn.Linear(64, N_BANDS + 2))
    return enc_a, enc_b, dec, aud_head


def _mask_frames(torch, x, rng):
    """Zero a fixed fraction of 8x8 patches; return (masked, patch mask)."""
    b = x.shape[0]
    g = RES // PATCH
    keep = torch.from_numpy(
        (rng.rand(b, g, g) >= MASK_FRAC).astype(np.float32))
    m = keep.repeat_interleave(PATCH, 1).repeat_interleave(PATCH, 2)
    return x * m.unsqueeze(1), (1.0 - m).unsqueeze(1)


def _pretrain(torch, arm: str, data: dict, seed: int, models) -> dict:
    """One arm's pretext run. Matched steps/batch/optimiser for every arm.

    arm in {"ua", "plastic", "frozen", "shuffled"}. `models` are freshly
    built with the shared per-seed init; for "frozen" the caller passes an
    A-encoder already trained by the "ua" arm, and it is excluded from the
    optimiser — the same frozen tensor, per the registered null."""
    nn = torch.nn
    enc_a, enc_b, dec, aud_head = models
    frames = _nchw(torch, data["frames"])
    aud = data["audio"]
    mu, sd = aud.mean(0), aud.std(0) + 1e-8
    aud_z = torch.from_numpy((aud - mu) / sd)
    n = frames.shape[0]

    rng = np.random.RandomState(seed * 31 + {"ua": 0, "plastic": 1,
                                             "frozen": 2, "shuffled": 3}[arm])
    if arm == "shuffled":
        # A fixed derangement: every row's audio comes from a DIFFERENT
        # episode. Marginals preserved, correspondence destroyed.
        perm = np.roll(rng.permutation(n), 1)
        aud_z = aud_z[torch.from_numpy(perm)]

    params = list(dec.parameters())
    if arm != "ua":
        params += list(enc_b.parameters()) + list(aud_head.parameters())
    if arm != "frozen":
        params = list(enc_a.parameters()) + params
    opt = torch.optim.Adam(params, lr=LR)
    torch.manual_seed(seed * 7919 + 11)

    first_loss, last_loss = None, None
    for step in range(STEPS):
        idx = torch.from_numpy(rng.randint(0, n, BATCH))
        x = frames[idx]
        xm, hole = _mask_frames(torch, x, rng)
        za = enc_a(xm)
        if arm == "ua":
            zb = torch.zeros(BATCH, Z_B)
        else:
            drop = torch.from_numpy(
                (rng.rand(BATCH) < AUDIO_DROP).astype(np.float32)).unsqueeze(1)
            zb = enc_b(aud_z[idx]) * (1.0 - drop)
        z = torch.cat([za, zb], dim=1)
        recon = dec(z)
        loss = ((recon - x) ** 2 * hole).sum() / hole.sum().clamp(min=1.0)
        if arm != "ua":
            pred = aud_head(z)
            # predict B only where B was withheld — cross-modal, not copy
            w = drop
            aud_loss = ((pred - aud_z[idx]) ** 2 * w).sum() / (
                w.sum() * (N_BANDS + 2)).clamp(min=1.0)
            loss = loss + aud_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
    return {"enc_a": enc_a, "first_loss": first_loss, "last_loss": last_loss,
            "aud_mu": mu, "aud_sd": sd}


def _features(torch, enc_a, frames: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = _nchw(torch, frames)
        out = []
        for i in range(0, x.shape[0], 256):
            out.append(enc_a(x[i:i + 256]).numpy())
    return np.concatenate(out).astype(np.float64)


def _probe_r2(Xtr, ytr, Xte, yte) -> tuple:
    fit = _Ridge(Xtr, l2=L2)
    pred = fit.predict(ytr, Xte)
    return _r2(yte, pred), pred


def _boot_ci(y, pred_a, pred_b, seed: int) -> tuple:
    """Paired bootstrap over test episodes of r2(pred_a) - r2(pred_b)."""
    rng = np.random.RandomState(seed * 613 + 29)
    n = len(y)
    diffs = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.randint(0, n, n)
        diffs[k] = _r2(y[idx], pred_a[idx]) - _r2(y[idx], pred_b[idx])
    return (float(np.percentile(diffs, CI_LO)),
            float(np.percentile(diffs, CI_HI)))


# ── the experiment ───────────────────────────────────────────────────────
_CACHE: dict = {}


def _fixture(seed: int) -> dict:
    """Rendered episodes + the U_A/plastic training results, cached so the
    declared control (_control) reuses the SAME data, init and baseline
    rather than re-rolling them — the pairing is the measurement."""
    if seed in _CACHE:
        return _CACHE[seed]
    torch = _torch()
    eye = get_eye(seed)
    canary0 = eye.canary()
    rng = np.random.RandomState(seed * 271 + 3)
    pretext = _episodes(eye, rng, N_PRETEXT)
    probe_tr = _episodes(eye, rng, N_PROBE_TR)
    probe_te = _episodes(eye, rng, N_PROBE_TE)
    _CACHE[seed] = {"eye": eye, "canary0": canary0, "pretext": pretext,
                    "probe_tr": probe_tr, "probe_te": probe_te}
    return _CACHE[seed]


def _raw_pixel_r2(fx: dict) -> float:
    """PG.6's certified quantity on THIS run's own probe split: radius from
    RAW flattened RGB pixels, same ridge operating point (l2 = L2).

    This is the referent of the eye-aliveness VOID gate as ruled on
    2026-09-11 — see the docstring. It touches no encoder and no arm, so it
    is a property of the eye and the sampler alone and cannot be moved by
    anything the pretext training does. Dual-form ridge: d = 3*RES^2 = 12288
    features against n = 1000 rows, so `_Ridge` solves the n x n system.
    """
    ytr = fx["probe_tr"]["radius"].astype(np.float64)
    yte = fx["probe_te"]["radius"].astype(np.float64)
    Xtr = fx["probe_tr"]["frames"].reshape(len(ytr), -1).astype(np.float64)
    Xte = fx["probe_te"]["frames"].reshape(len(yte), -1).astype(np.float64)
    r2, _ = _probe_r2(Xtr, ytr, Xte, yte)
    return r2


def _arm_r2(torch, fx: dict, res: dict) -> tuple:
    """(r2, test predictions) of one trained A-encoder on radius, A only."""
    Xtr = _features(torch, res["enc_a"], fx["probe_tr"]["frames"])
    Xte = _features(torch, res["enc_a"], fx["probe_te"]["frames"])
    r2, pred = _probe_r2(Xtr, fx["probe_tr"]["radius"].astype(np.float64),
                         Xte, fx["probe_te"]["radius"].astype(np.float64))
    return r2, pred, Xte


def _experiment(seed: int) -> dict:
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    torch = _torch()
    fx = _fixture(seed)
    yte = fx["probe_te"]["radius"].astype(np.float64)

    ua = _pretrain(torch, "ua", fx["pretext"], seed,
                   _build_models(torch, seed))
    plastic = _pretrain(torch, "plastic", fx["pretext"], seed,
                        _build_models(torch, seed))
    # FROZEN: fresh heads from the shared init, but the A-encoder IS U_A's
    # trained tensor — a deep copy, then excluded from the optimiser.
    import copy
    fr_models = _build_models(torch, seed)
    fr_models = (copy.deepcopy(ua["enc_a"]),) + fr_models[1:]
    for p in fr_models[0].parameters():
        p.requires_grad_(False)
    frozen = _pretrain(torch, "frozen", fx["pretext"], seed, fr_models)

    r2_ua, pred_ua, Xte_ua = _arm_r2(torch, fx, ua)
    r2_pl, pred_pl, _ = _arm_r2(torch, fx, plastic)
    r2_fr, pred_fr, Xte_fr = _arm_r2(torch, fx, frozen)

    R_pl = r2_pl - r2_ua
    lo_pl, hi_pl = _boot_ci(yte, pred_pl, pred_ua, seed)
    R_fr = r2_fr - r2_ua
    feat_diff = float(np.abs(Xte_fr - Xte_ua).max())

    # rig gates, measured
    det = float(np.abs(_features(torch, ua["enc_a"],
                                 fx["probe_te"]["frames"]) - Xte_ua).max())
    aud_tr = fx["probe_tr"]["audio"].astype(np.float64)
    aud_te = fx["probe_te"]["audio"].astype(np.float64)
    r2_aud, _ = _probe_r2(aud_tr, fx["probe_tr"]["radius"].astype(np.float64),
                          aud_te, yte)
    sh = np.random.RandomState(seed * 97 + 1).permutation(len(yte))
    r2_shuf_label, _ = _probe_r2(
        Xte_ua, yte[sh], Xte_ua, yte)  # fit on shuffled labels, score on true
    canary_ok = int(fx["eye"].canary() == fx["canary0"])
    learn_ok = int(all(r["last_loss"] < LEARN_DROP * r["first_loss"]
                       for r in (ua, plastic, frozen)))
    # the eye-aliveness gate's referent (ruled 2026-09-11) — no encoder in it
    r2_raw = _raw_pixel_r2(fx)

    # persist the trained A-encoders — the champion-arena standing rule
    ART_DIR.mkdir(exist_ok=True)
    art = ART_DIR / f"pl02_encoders_seed{seed}.pt"
    torch.save({"ua": ua["enc_a"].state_dict(),
                "plastic": plastic["enc_a"].state_dict(),
                "spec": SPEC_ID, "seed": seed,
                "note": "frozen arm's A-encoder == ua's, by construction"},
               art)
    art_sha = hashlib.sha256(art.read_bytes()).hexdigest()[:8]

    frozen_exact = int(abs(R_fr) < FROZEN_TOL and feat_diff < FEAT_TOL)
    claim_seed = int(lo_pl > 0.0)

    # Cached for the declared control: same U_A baseline, same pairing.
    fx["ua_res"], fx["pred_ua"] = ua, pred_ua
    fx["r2_ua"], fx["plastic_R"] = r2_ua, R_pl

    return {
        "seed": seed,
        "reshaping_gain_R": round(R_pl, 6),
        "R_ci_lo": round(lo_pl, 6), "R_ci_hi": round(hi_pl, 6),
        "r2_raw_pixel": round(r2_raw, 6),
        "r2_ua": round(r2_ua, 6), "r2_plastic": round(r2_pl, 6),
        "r2_frozen": round(r2_fr, 6),
        "frozen_R": R_fr, "frozen_feat_diff": feat_diff,
        "frozen_exact_zero": frozen_exact,
        "claim_ci_above_zero": claim_seed,
        "audio_channel_r2": round(r2_aud, 6),
        "shuffled_label_r2": round(r2_shuf_label, 6),
        "canary_ok": canary_ok, "learn_ok": learn_ok,
        "det_drift": det,
        "loss_drop_ua": round(ua["last_loss"] / max(ua["first_loss"], 1e-12), 4),
        "loss_drop_plastic": round(
            plastic["last_loss"] / max(plastic["first_loss"], 1e-12), 4),
        # DISCLOSURE, added 2026-09-13 after attempt 1 VOIDed on `learn_ok`
        # with no way to say WHICH arm missed: `learn_ok` quantifies over three
        # arms and only two of their ratios were recorded, so the run's own
        # verdict was unattributable from its own row. Emitting the third costs
        # nothing (it is computed above) and moves no gate — LEARN_DROP stays
        # 0.90 and `learn_ok` is unchanged.
        "loss_drop_frozen": round(
            frozen["last_loss"] / max(frozen["first_loss"], 1e-12), 4),
        "weights_artifact": str(art), "weights_sha8": art_sha,
    }


def _control(seed: int) -> dict:
    """SHUFFLED-PARTNER, the declared control. Same data, same init, same
    steps as the plastic arm — only the correspondence is destroyed. Its R
    must collapse to ~0; a shuffled partner that reshapes as well as the
    real one means the gain was capacity, and _check returns VOID."""
    torch = _torch()
    fx = _fixture(seed)
    yte = fx["probe_te"]["radius"].astype(np.float64)
    shuf = _pretrain(torch, "shuffled", fx["pretext"], seed,
                     _build_models(torch, seed))
    # The U_A baseline comes from the experiment's cache — the SAME trained
    # tensor and the SAME predictions, so the control's R is paired against
    # the identical baseline the claim's R was. run_spec runs experiments
    # before controls in one process; a missing cache is a harness fault and
    # should error loudly, not silently re-derive.
    r2_ua, pred_ua = fx["r2_ua"], fx["pred_ua"]
    r2_sh, pred_sh, _ = _arm_r2(torch, fx, shuf)
    R_sh = r2_sh - r2_ua
    lo, hi = _boot_ci(yte, pred_sh, pred_ua, seed + 500)
    # "reshapes just as well", per-seed, against the SAME seed's plastic R
    reshapes_too = int(lo > 0.0 and R_sh >= CONTROL_COLLAPSE_FRAC
                       * max(fx["plastic_R"], 1e-12))
    return {"seed": seed, "shuffled_R": round(R_sh, 6),
            "shuffled_ci_lo": round(lo, 6), "shuffled_ci_hi": round(hi, 6),
            "r2_ua_ctrl": round(r2_ua, 6), "r2_shuffled": round(r2_sh, 6),
            "control_reshapes_too": reshapes_too,
            # Same disclosure as the experiment's `loss_drop_frozen`: attempt 1
            # VOIDed with `shuffled_learn_ok` 0.667 and no recorded ratio to
            # say by how much or on which seed. The gate is unchanged.
            "loss_drop_shuffled": round(
                shuf["last_loss"] / max(shuf["first_loss"], 1e-12), 4),
            "shuffled_learn_ok": int(
                shuf["last_loss"] < LEARN_DROP * shuf["first_loss"])}


def _check(m: dict, c: dict):
    # ── RIG GATES: VOID, not FAIL — an invalid run is not evidence.
    if m["canary_ok"] < 1.0 or m["learn_ok"] < 1.0:
        return Status.VOID
    if c.get("shuffled_learn_ok", 1.0) < 1.0:
        return Status.VOID
    if m["det_drift"] > 0.0:
        return Status.VOID
    if m["audio_channel_r2"] < AUDIO_TEACH_R2_MIN:
        return Status.VOID          # the teacher is ignorant; premise dead
    if m["r2_raw_pixel"] < EYE_RADIUS_R2_MIN:
        return Status.VOID          # the eye is blind; R would be a
        # difference of two dead probes, not a reshaping measurement
        # (82nd audit B4 — the coarse eye postdates PG.6's certificate).
        # THE REFERENT IS THE RAW-PIXEL RIDGE, NOT `r2_ua` (Review DAILY
        # 2026-09-11, 5e39771): r2_ua is the SUBTRAHEND of the claim's own
        # effect size, so gating on it caps the reportable gain at <= 0.20
        # and is un-clearable in exactly the regime the claim tests. The
        # bar (0.80) and the VOID semantics are unmoved. `r2_ua` is still
        # recorded above and still appears on the row.
    if m["shuffled_label_r2"] > SHUF_LABEL_R2_MAX:
        return Status.VOID          # the probe instrument leaks
    # the analytic null must BE the arithmetic it claims to be
    if m["frozen_exact_zero"] < 1.0:
        return Status.VOID
    # ── THE DECLARED CONTROL: shuffled-B must not reshape "just as well".
    # `control_reshapes_too` is a per-seed indicator paired against the same
    # seed's plastic R; aggregation is mean-across-seeds, so ANY offending
    # seed pulls the mean above 0 and this fires — conservative in the
    # refusing direction, per the docstring.
    if c["control_reshapes_too"] > 0.0:
        return Status.VOID
    # ── THE CLAIM: every seed's paired bootstrap CI above zero.
    return bool(m["claim_ci_above_zero"] >= 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Rig-aliveness pass on seed 90 (disjoint from registered seeds).
        m = _experiment(90)
        c = _control(90)
        print({k: m[k] for k in sorted(m) if not k.startswith("weights")})
        print({k: c[k] for k in sorted(c)})
        print("check ->", _check(m, c))
    else:
        print(run())
