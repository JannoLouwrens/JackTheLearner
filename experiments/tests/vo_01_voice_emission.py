"""VO.01 — he can make a sound, and it is heard as a sound in the world.

GOAL.md lists VOICE among the constitutional senses, and it is the only one
that is an EFFECTOR: "how a creature acts on other creatures". Before any claim
about signalling, coordination or an invented protocol (VO.02, GEN.02/03) can
mean anything, one thing has to be true and it has never been measured here:
that an emission Jack produces ARRIVES AT ANOTHER CREATURE'S EAR as sound, and
gets there the way sound gets anywhere — quieter with distance, muffled through
a solid, and mixed into the same stream as everything else the world is doing.

The failure this spec exists to prevent is named in the registry: *a wire
between two brains wearing the word "voice"*. That wire is easy to build by
accident — pass the emission vector to the listener's observation and every
downstream communication experiment succeeds, having tested nothing. Here the
listener is given ONLY `[2, T]` stereo samples: the emission is synthesised by
`ContactAudio`, panned by the same law as a falling apple, attenuated by the
same 1/r, occluded by the same `mj_ray` that decides what the eye can see, and
summed into the same buffer as the contact events. Everything the probe knows,
it knows from the waveform.

WHAT IS MEASURED, and each gate's rival

  THE DIFFICULTY, which is now itself a gate. A recovery number means nothing
  without the level of the interference it was measured against, and v2's
  interference level was an undeclared constant: `BG_EVENTS_PER_EP = (2, 7)`,
  chosen by taste, which put the voice at **-4.36 dB** relative to the
  playground's own contact noise. Below the room, the four recovery gates were
  measuring auditory scene analysis, not the channel this spec claims. v3
  DERIVES the level instead: the mixed background is scaled per seed by
  `_bg_gain` to hit `SIR_TARGET_DB = +6` — audible over the room, nowhere near
  alone in it — and `voice_to_background_db` is reported and gated within
  +/-2 dB. The gate is TWO-SIDED on purpose: quieting the room to buy a PASS
  now fails exactly as loudly as a room that drowns the voice. The background
  is scaled, never removed; the four recovery gates below are unchanged from
  v1. (Pre-registered in docs/LOOP_JOURNAL.md, 2026-08-11, before any recovery
  number at the new level had been seen.)

  RECOVERY. The emission is four continuous action dimensions (f0, brightness,
  amplitude, duration) — a policy can drive it, and VO.01's emitter drives it
  with uniform noise because this spec is about the channel, not about what to
  say. Those four have to be INDEPENDENT to be worth four action dimensions,
  and the first recorded run proved they were not: the call was peak-normalised,
  so a bright call came out 3.8x quieter than a dark one at identical `amp`, and
  brightness recovery read 0.347 against a 0.50 gate. The emitter now normalises
  to constant RMS with Schroeder phases; `amp` means loudness and `brightness`
  means timbre. A ridge probe on a crude log-band spectrogram of the received stereo
  regresses those four back, TRAINED on one set of episodes and scored on
  HELD-OUT ones, with the emitter at a random bearing and a random distance in
  every episode so loudness alone can never identify the call, and with real
  contact audio from real dropped objects mixed into the same ears. Rival: the
  MUTED emitter, the same episodes, the same pipeline, the same background and
  the same noise floor. If the muted probe also recovers, the information was
  never in the sound.

  DISTANCE. The same reference call at six ranges on a verified-clear line;
  received RMS against the declared 1/max(r, 0.5). Taken alone this gate would
  be a TRIPWIRE — it re-derives the render's own multiplication and can only
  fire on an edit. Two things make it evidence: a render with 1/r removed is
  run as a control and the gate must catch it, and the INVERSE-SQUARE rival
  (energy, not pressure — the law a plausible implementation would have picked)
  is required to MISS by a margin.

  OCCLUSION, two-sided, as SM.01 gates it for smell and for the same reason.
  It must attenuate (or the ray-cast is a no-op and "voice crosses walls" is
  true by construction) AND it must not silence (or voice is just vision with
  extra steps). The occluder is `playground.py`'s `welded_block` and the
  geometry is SM.01's line, deliberately: it makes the two senses' occlusion
  claims one comparable statement rather than two anecdotes. Light is checked
  independently — it must NOT reach the hidden listener, and it must reach a
  second listener at the identical 2.0 m.

  MUFFLING. `OCC_TRANSMISSION` is frequency-dependent (the mass law: walls pass
  bass), so the received spectral centroid must FALL behind the block. This is
  the gate that could have been a tripwire — a check that merely re-derives the
  module's own formula can only ever fire on an edit (LESSONS.md, SM.01's
  falloff gate). It is a discrimination instead because it names its rival: a
  FLAT occluder, same transmission, no tilt, is rendered as a control and the
  centroid gate must miss it.

  HEARD THROUGH THE WALL. A probe trained on clear-line episodes at the low
  geometry is scored on occluded episodes at MATCHED distances and bearings, so
  the only difference between train and test is the block. f0 and duration are
  gated; BRIGHTNESS AND AMPLITUDE ARE REPORTED, NOT GATED, and the reason is
  the physics: a low-pass occluder changes spectral tilt systematically, so a
  probe that has only ever heard unoccluded calls should mis-read timbre
  through a wall. That is a prediction of the model, not a hole in the test —
  and if `occ_recov_r2_bright` ever came out HIGH, the occluder would not be
  filtering.

WHAT THIS SPEC DOES NOT CLAIM. Not that the emission is a signal (nothing here
learns), not that two agents coordinate (VO.02, blocked on a second Jack), and
not the decibels: `ContactAudio.OCC_TRANSMISSION` models transmission through a
solid and ignores diffraction, which real bass exploits to bend around a 0.3 m
block. The claim is the ORDERING — sound crosses what light does not — and the
mechanism, not the number.

Depends on PG.5, which certified that this synth's panning and its labels
describe the world; voice reuses that same pan law and adds nothing to it.
"""

from __future__ import annotations

import math

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import ContactAudio as CA  # noqa: E402
import playground as pg  # noqa: E402

from ..protocol import Ledger, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402

# The claim is about the WORLD and the synth, not only about this file: the
# occluder is `playground.py`'s welded block and every emission constant is
# `ContactAudio.py`'s. Change either and this certificate goes stale loudly
# instead of standing over a world it no longer describes.
IMPL_DEPS = ["playground.py", "ContactAudio.py"]

# ── the listener ────────────────────────────────────────────────────────
HEAD = (0.0, 0.0, 1.4)          # PG.5's certified listener pose
HEAD_YAW = 0.0

# ── episode timing ──────────────────────────────────────────────────────
T_VOICE = 0.50                  # when the mouth opens, s
RENDER_S = 1.60
WIN = (0.48, 1.28)              # the analysis window, s (0.80 s, > max duration)
EAR_NOISE_SIGMA = 1e-3          # the listener's own noise floor

# ── the probe episodes (set A: head height, clear line) ─────────────────
N_TRAIN = 300
N_TEST = 100
RANGE_M = (1.0, 4.5)            # inside the 6 m arena, outside arm's reach
BG_EVENTS_PER_EP = (2, 7)       # real contact audio, mixed into the same ears

# ── the interference level: DERIVED from a stated target, then GATED ────
# v2 recorded voice 0.0152 against background 0.0251 — a signal-to-interference
# ratio of -4.36 dB, i.e. the voice sat BELOW the playground's own contact
# noise, and at that ratio the recovery gates were measuring auditory scene
# analysis rather than the channel this spec claims. The constant that set it,
# `BG_EVENTS_PER_EP`, was chosen by taste and never derived, while every gate
# around it was reasoned about at length.
#
# The background is NOT removed — a clean synthetic channel would prove
# nothing about a world that is also making noise. Instead the mixed
# background is SCALED, per seed, to hit a stated SIR, and that SIR is then a
# reported metric with a TWO-SIDED gate: too quiet fails exactly as loudly as
# too loud, so the difficulty of this spec can never again be adjusted without
# the ledger showing it. Pre-registered in docs/LOOP_JOURNAL.md on 2026-08-11,
# before any recovery number at the new level had been seen; the four recovery
# gates below are unchanged from v1.
SIR_TARGET_DB = 6.0             # the voice audible over the room, not alone in it
SIR_TOL_DB = 2.0
N_CALIB = 60                    # calibration episodes, on their own RNG stream

# ── features and probe ──────────────────────────────────────────────────
# A crude log-band spectrogram — what a cochlea gives you before anything is
# learned. Sized against N_TRAIN, not against what would score best: 115
# features on 300 examples, ridge-regularised, standardised on the train split.
BAND_HZ = (60.0, 7500.0)
N_SBANDS = 14                   # bands per spectrogram frame
N_FRAMES = 8                    # 100 ms frames across the analysis window
N_BANDS = 24                    # full-window bands, for fine spectral detail
RIDGE_ALPHA = 10.0
EPS = 1e-12
DIMS = ("f0", "bright", "amp", "dur")

# ── the occlusion fixture: SM.01's line, SM.01's occluder ───────────────
BLOCK = np.array([-1.5, -1.5, 0.15])            # `welded_block`, half-extent 0.15
OCCLUDER_NAME = "welded_block"
L_HIDDEN = BLOCK + np.array([1.00, 0.0, 0.0])   # listener 1.0 m past the block
L_LIT = np.array([-2.5, -3.5, 0.15])            # listener on a clear line
OCC_D = 2.0                                     # mouth-to-ear range, both cases
OCC_JITTER = 0.04
N_OCC = 160                                     # matched occluded / clear pairs
OCC_D_RANGE = (1.6, 2.4)
REF_ACTION = (-0.05, 0.5, 0.6, 0.0)             # f0 ~ 226 Hz, bright, mid-length

# ── the distance ladder ─────────────────────────────────────────────────
DIST_LADDER = (0.6, 1.0, 1.6, 2.5, 3.5, 4.5)

# ── PRE-REGISTERED GATES ────────────────────────────────────────────────
R2_MIN_PER_DIM = 0.50       # half the variance of each emitted parameter
R2_MIN_MEAN = 0.60          # ...and the four together
MUTE_R2_MAX = 0.05          # the muted null must be at chance
MUTE_RMS_MULT = 2.0         # ...and, in a SILENT world, its ears at the floor.
                            # v1 applied this to the muted ear WITH background
                            # contact audio in it and recorded 0.0251 against a
                            # 0.002 gate — it was measuring how loud the
                            # playground is, which is not a claim about voice.
                            # The registry's control ("hears nothing above the
                            # noise floor") is about the VOICE, so the arm that
                            # tests it shuts the mouth AND empties the world;
                            # `mute_ear_rms` stays reported beside it.
DIST_LAW_TOL = 0.05         # received RMS vs the declared 1/max(r, 0.5)
DIST_INV2_DISCRIM_MIN = 0.10    # ...and the inverse-square rival must MISS
OCC_RATIO_MAX = 0.50        # the block really attenuates
OCC_RATIO_MIN = 0.02        # ...and does not silence
OCC_CENTROID_DROP_MIN = 0.15    # the muffling signature, which a flat gain lacks
OCC_R2_MIN = 0.50           # heard through the wall: f0 and duration
CLIP_FRAC_MAX = 1e-4        # the ears are not saturating


# ── world ───────────────────────────────────────────────────────────────
# Cached per seed: `_experiment` and `_control` both need the same world, and
# the background rollout is the only physics this spec runs.
_WORLD: dict = {}
_BG: dict = {}


def _world(seed: int):
    """The playground, unchanged. The occluder is its own welded block."""
    if seed not in _WORLD:
        p = pg.PlaygroundParams(seed=seed)
        model = mujoco.MjModel.from_xml_string(pg.build_mjcf(p))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        _WORLD[seed] = (model, data)
    return _WORLD[seed]


def _hit_geom(model, data, a, b) -> str:
    """Name of the first geom on the segment a->b, or "" if the line is clear.
    Independent of the synth's own occlusion call on purpose."""
    a = np.asarray(a, dtype=float)
    delta = np.asarray(b, dtype=float) - a
    dist = float(np.linalg.norm(delta))
    gid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, a, delta / dist, None, 1, -1, gid)
    if not (0.0 <= hit < dist - 1e-6) or gid[0] < 0:
        return ""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or "?"


def _free_bodies(model):
    out = []
    for bid in range(model.nbody):
        jadr = model.body_jntadr[bid]
        if jadr >= 0 and model.jnt_type[jadr] == 0:      # mjJNT_FREE
            out.append(bid)
    return out


def _background(seed):
    """A bank of REAL contact events: objects dropped in the real playground,
    heard from the head pose. Mixed into every episode so the probe has to pick
    the voice out of a world that is also making noise.

    Runs on its OWN `MjData`. The cached `data` is the pristine `mj_forward`
    state every line-of-sight check in this file is taken against, and a
    rollout that moved objects onto those lines would make the geometry
    assertions describe a world that no longer existed when the audio was
    rendered.
    """
    if seed in _BG:
        return _BG[seed]
    model, _ = _world(seed)
    data = mujoco.MjData(model)
    synth = CA.ContactAudioSynth(model)
    synth.set_listener(HEAD, HEAD_YAW)
    rng = np.random.RandomState(seed * 7919 + 11)
    for bid in _free_bodies(model):
        jadr = model.body_jntadr[bid]
        qadr = model.jnt_qposadr[jadr]
        vadr = model.jnt_dofadr[jadr]
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(1.2, 3.0)
        data.qpos[qadr:qadr + 3] = (r * math.cos(ang), r * math.sin(ang),
                                    rng.uniform(0.8, 1.6))
        data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        synth.step(data)
    _BG[seed] = synth.events
    return _BG[seed]


def _episode_synth(model, listener, bg, rng, n_bg, bg_gain=1.0):
    """A fresh synth at `listener`, preloaded with `n_bg` background contacts
    re-localized to this listener and scattered in time.

    `bg_gain` scales the contact events' amplitude — `render` applies `e.amp`
    linearly, so this is a level control on the room and nothing else: the same
    events, at the same times, from the same real drops, arriving quieter. It
    is derived by `_bg_gain` from `SIR_TARGET_DB`, never chosen.
    """
    synth = CA.ContactAudioSynth(model)
    synth.set_listener(listener, HEAD_YAW)
    if bg and n_bg > 0:
        for idx in rng.choice(len(bg), size=min(n_bg, len(bg)), replace=False):
            e = bg[int(idx)]
            az, lat, el, dist = synth.localize(e.pos)
            synth.events.append(CA.AudioEvent(
                t=float(rng.uniform(0.0, RENDER_S - 0.35)), geom1=e.geom1,
                geom2=e.geom2, voiced_geom=e.voiced_geom, pos=e.pos,
                force=e.force, amp=e.amp * bg_gain, azimuth=az, lateral=lat,
                elevation=el, distance=dist))
    return synth


def _ear(synth, rng=None, **render_kw) -> np.ndarray:
    """Render, then add the listener's own noise floor. `rng=None` means a
    CLEAN measurement of the render law (the distance/occlusion fixtures)."""
    ear = synth.render(RENDER_S, **render_kw)
    if rng is not None:
        ear = ear + rng.normal(0.0, EAR_NOISE_SIGMA, size=ear.shape)
    return ear


# ── what the listener may use: the waveform, and nothing else ───────────
def _window(ear, sr):
    return ear[:, int(WIN[0] * sr):int(WIN[1] * sr)]


def _bands(x, sr, n_bands) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    edges = np.geomspace(BAND_HZ[0], BAND_HZ[1], n_bands + 1)
    idx = np.clip(np.searchsorted(edges, freqs, side="right") - 1, -1, n_bands - 1)
    out = np.zeros(n_bands)
    sel = idx >= 0
    np.add.at(out, idx[sel], spec[sel])
    return out


def _features(ear, sr) -> np.ndarray:
    """Everything the listener is allowed to know: a log-band spectrogram of
    the received stereo, plus the two channel energies. No labels, no emission
    vector, no event times — the waveform and nothing else."""
    w = _window(ear, sr)
    mono = w[0] + w[1]
    frames = np.array_split(mono, N_FRAMES)
    spg = np.concatenate([_bands(f, sr, N_SBANDS) for f in frames])
    full = _bands(mono, sr, N_BANDS)
    chan = np.array([float(np.sum(w[0] ** 2)), float(np.sum(w[1] ** 2))])
    return np.log(np.concatenate([spg, full, chan]) + EPS)


def _centroid(ear, sr) -> float:
    """Energy-weighted mean frequency in the analysis band — the quantity that
    falls when a low-pass occluder is between mouth and ear, and does not when
    the occluder is a flat gain."""
    w = _window(ear, sr)
    mono = w[0] + w[1]
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    sel = (freqs >= BAND_HZ[0]) & (freqs < BAND_HZ[1])
    p = spec[sel]
    return float(np.sum(freqs[sel] * p) / max(np.sum(p), EPS))


def _rms(ear, sr) -> float:
    return float(np.sqrt(np.mean(_window(ear, sr) ** 2)))


# ── the room's level, measured against the voice rather than chosen ─────
_GAIN: dict = {}


def _bg_gain(seed: int) -> float:
    """The scale that puts the room `SIR_TARGET_DB` below the voice.

    Measured, not chosen. `N_CALIB` episodes drawn from set A's own pose
    distribution are rendered TWICE — once with the mouth open into an EMPTY
    world (the voice alone) and once with the mouth shut into the full
    background at unit gain (the room alone). Both renders are clean, so the
    ratio belongs to the two sources and not to the ear's noise floor. The gain
    then follows in closed form, because `render` scales a contact event
    linearly in `amp`.

    Its own RNG stream, and its own episodes: the calibration never sees a pose
    the probe is scored on.
    """
    if seed in _GAIN:
        return _GAIN[seed]
    model, data = _world(seed)
    bg = _background(seed)
    rng = np.random.RandomState(seed * 15485863 + 41)
    sr = CA.SAMPLE_RATE
    v, b = [], []
    tries = 0
    while len(v) < N_CALIB:
        tries += 1
        if tries > 40 * N_CALIB:
            raise RuntimeError("calibration: could not find clear emitter poses")
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(*RANGE_M)
        pos = np.array([HEAD[0] + r * math.cos(ang), HEAD[1] + r * math.sin(ang),
                        HEAD[2]])
        if _hit_geom(model, data, pos, HEAD) != "":
            continue
        action = rng.uniform(-1.0, 1.0, size=CA.VOICE_ACTION_DIM)
        n_bg = int(rng.randint(*BG_EVENTS_PER_EP))
        room = _episode_synth(model, HEAD, bg, rng, n_bg)
        b.append(_rms(_ear(room, None, mute_voice=True), sr))
        alone = CA.ContactAudioSynth(model)
        alone.set_listener(HEAD, HEAD_YAW)
        alone.emit_voice(T_VOICE, pos, action, data=data)
        v.append(_rms(_ear(alone, None), sr))
    ratio = float(np.mean(v)) / max(float(np.mean(b)), EPS)
    _GAIN[seed] = float(ratio / (10.0 ** (SIR_TARGET_DB / 20.0)))
    return _GAIN[seed]


# ── the probe ───────────────────────────────────────────────────────────
def _ridge_r2(xtr, ytr, xte, yte) -> np.ndarray:
    """Held-out R^2 per target dimension. Standardisation statistics come from
    the TRAIN split only; the test split never touches the fit."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    a = np.hstack([(xtr - mu) / sd, np.ones((len(xtr), 1))])
    b = np.hstack([(xte - mu) / sd, np.ones((len(xte), 1))])
    reg = RIDGE_ALPHA * np.eye(a.shape[1])
    reg[-1, -1] = 0.0                       # never shrink the intercept
    w = np.linalg.solve(a.T @ a + reg, a.T @ ytr)
    pred = b @ w
    ss_res = np.sum((yte - pred) ** 2, axis=0)
    ss_tot = np.sum((yte - yte.mean(0)) ** 2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, EPS)


def _set_a(seed, mute: bool):
    """Set A: head height, random bearing, random range, clear line of sight.
    Returns (features, actions, mean ear RMS)."""
    model, data = _world(seed)
    bg = _background(seed)
    gain = _bg_gain(seed)
    rng = np.random.RandomState(seed * 104729 + 17)
    sr = CA.SAMPLE_RATE
    feats, acts, rmss = [], [], []
    n = N_TRAIN + N_TEST
    tries = 0
    while len(acts) < n:
        tries += 1
        if tries > 40 * n:
            raise RuntimeError("set A: could not find clear emitter poses")
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(*RANGE_M)
        pos = np.array([HEAD[0] + r * math.cos(ang), HEAD[1] + r * math.sin(ang),
                        HEAD[2]])
        if _hit_geom(model, data, pos, HEAD) != "":
            continue
        action = rng.uniform(-1.0, 1.0, size=CA.VOICE_ACTION_DIM)
        n_bg = int(rng.randint(*BG_EVENTS_PER_EP))
        synth = _episode_synth(model, HEAD, bg, rng, n_bg, gain)
        synth.emit_voice(T_VOICE, pos, action, data=data)
        ear = _ear(synth, rng, mute_voice=mute)
        feats.append(_features(ear, sr))
        acts.append(action)
        rmss.append(_rms(ear, sr))
    return np.array(feats), np.array(acts), float(np.mean(rmss))


def _recovery(seed, mute: bool) -> dict:
    x, y, rms = _set_a(seed, mute)
    r2 = _ridge_r2(x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:])
    tag = "mute" if mute else "recov"
    out = {f"{tag}_r2_{d}": float(v) for d, v in zip(DIMS, r2)}
    out[f"{tag}_r2_mean"] = float(np.mean(r2))
    out[f"{tag}_r2_max"] = float(np.max(r2))
    out[f"{tag}_ear_rms"] = rms
    return out


def _silence(seed: int) -> dict:
    """The registry's control, correctly instrumented: mouth shut, world empty.
    What is left in the ear must be the ear's own noise and nothing else — and
    the SAME episodes with the mouth open must be loudly above it, or the
    comparison is between two silences."""
    model, data = _world(seed)
    rng = np.random.RandomState(seed * 2749 + 31)
    sr = CA.SAMPLE_RATE
    # N_CALIB episodes, not the 20 of v1: `voiced_silent_rms` is now the
    # numerator of a GATED quantity (`voice_to_background_db`), so its standard
    # error has to be small next to the +/-2 dB tolerance. More samples of the
    # same estimator — no gate moved.
    shut, open_ = [], []
    for _ in range(N_CALIB):
        ang, r = rng.uniform(-math.pi, math.pi), rng.uniform(*RANGE_M)
        pos = np.array([HEAD[0] + r * math.cos(ang), HEAD[1] + r * math.sin(ang),
                        HEAD[2]])
        if _hit_geom(model, data, pos, HEAD) != "":
            continue
        action = rng.uniform(-1.0, 1.0, size=CA.VOICE_ACTION_DIM)
        s = CA.ContactAudioSynth(model)
        s.set_listener(HEAD, HEAD_YAW)
        s.emit_voice(T_VOICE, pos, action, data=data)
        shut.append(_rms(_ear(s, rng, mute_voice=True), sr))
        open_.append(_rms(_ear(s, rng), sr))
    return {"mute_silent_rms": float(np.mean(shut)),
            "voiced_silent_rms": float(np.mean(open_))}


# ── distance: the declared law, measured at the ear ─────────────────────
def _distance(seed, voice_distance: bool = True) -> dict:
    model, data = _world(seed)
    sr = CA.SAMPLE_RATE
    rms, clear = [], True
    for d in DIST_LADDER:
        pos = np.array([HEAD[0] + d, HEAD[1], HEAD[2]])
        if _hit_geom(model, data, pos, HEAD) != "":
            clear = False
        s = CA.ContactAudioSynth(model)
        s.set_listener(HEAD, HEAD_YAW)
        s.emit_voice(T_VOICE, pos, REF_ACTION, data=data)
        rms.append(_rms(_ear(s, None, voice_distance=voice_distance), sr))
    rms = np.array(rms)
    d = np.array(DIST_LADDER)
    want = 1.0 / np.maximum(d, CA.MIN_DISTANCE)
    # Scale-free comparison: the law is about the SHAPE of the falloff, so both
    # curves are normalised at the nearest range before they are compared.
    got_n = rms / max(rms[0], EPS)
    dev = float(np.max(np.abs(got_n - want / want[0]) / (want / want[0])))
    inv2 = (1.0 / np.maximum(d, CA.MIN_DISTANCE) ** 2)
    dev2 = float(np.max(np.abs(got_n - inv2 / inv2[0]) / (inv2 / inv2[0])))
    tag = "" if voice_distance else "nodist_"
    return {f"{tag}dist_law_dev": dev,
            f"{tag}dist_dev_inverse_square": dev2,
            f"{tag}dist_monotone": float(bool(np.all(np.diff(rms) < 0.0))),
            f"{tag}dist_rms_near": float(rms[0]),
            f"{tag}dist_rms_far": float(rms[-1]),
            f"{tag}dist_line_clear": float(clear)}


# ── occlusion: two-sided, against independently-checked geometry ────────
def _occ_pairs(seed):
    """Matched (occluded, clear) emitter poses: same range, same jitter, one
    behind the block and one on an open line. Only the block differs."""
    model, data = _world(seed)
    rng = np.random.RandomState(seed * 6151 + 5)
    pairs = []
    tries = 0
    while len(pairs) < N_OCC:
        tries += 1
        if tries > 40 * N_OCC:
            raise RuntimeError("occlusion fixture: matched poses not found")
        d = rng.uniform(*OCC_D_RANGE)
        j = rng.uniform(-OCC_JITTER, OCC_JITTER, size=3)
        hid = L_HIDDEN + np.array([-d, 0.0, 0.0]) + j
        lit = L_LIT + np.array([0.0, d, 0.0]) + j
        if _hit_geom(model, data, hid, L_HIDDEN) != OCCLUDER_NAME:
            continue
        if _hit_geom(model, data, lit, L_LIT) != "":
            continue
        action = rng.uniform(-1.0, 1.0, size=CA.VOICE_ACTION_DIM)
        pairs.append((hid, lit, action))
    return pairs


def _occ_ref(seed, **render_kw) -> dict:
    """The reference call at exactly OCC_D, hidden and lit. Clean (no noise, no
    background) so the number is the occluder's and nothing else's."""
    model, data = _world(seed)
    sr = CA.SAMPLE_RATE
    out = {}
    for tag, listener, mouth in (
            ("occ", L_HIDDEN, L_HIDDEN + np.array([-OCC_D, 0.0, 0.0])),
            ("clear", L_LIT, L_LIT + np.array([0.0, OCC_D, 0.0]))):
        s = CA.ContactAudioSynth(model)
        s.set_listener(listener, HEAD_YAW)
        ev = s.emit_voice(T_VOICE, mouth, REF_ACTION, data=data)
        ear = _ear(s, None, **render_kw)
        out[f"ref_{tag}_rms"] = _rms(ear, sr)
        out[f"ref_{tag}_call_rms"] = float(np.sqrt(np.mean(
            ear[:, int(T_VOICE * sr):int((T_VOICE + ev.params.duration) * sr)] ** 2)))
        out[f"ref_{tag}_centroid"] = _centroid(ear, sr)
        out[f"ref_{tag}_flagged_occluded"] = float(ev.occluded)
    return out


def _occ_recovery(seed) -> dict:
    """Train the probe on the CLEAR half of the matched pairs, score it on the
    OCCLUDED half. Same ranges, same bearings, same calls — only the block."""
    model, data = _world(seed)
    bg = _background(seed)
    gain = _bg_gain(seed)
    rng = np.random.RandomState(seed * 3571 + 23)
    sr = CA.SAMPLE_RATE
    fx_c, fx_o, ys = [], [], []
    n_hidden_blocked = 0
    v_o, room_o = [], []
    for i, (hid, lit, action) in enumerate(_occ_pairs(seed)):
        n_bg = int(rng.randint(*BG_EVENTS_PER_EP))
        s_o = _episode_synth(model, L_HIDDEN, bg, rng, n_bg, gain)
        e_o = s_o.emit_voice(T_VOICE, hid, action, data=data)
        n_hidden_blocked += int(e_o.occluded)
        fx_o.append(_features(_ear(s_o, rng), sr))
        s_c = _episode_synth(model, L_LIT, bg, rng, n_bg, gain)
        s_c.emit_voice(T_VOICE, lit, action, data=data)
        fx_c.append(_features(_ear(s_c, rng), sr))
        ys.append(action)
        # REPORTED, NOT GATED. The clear-line SIR is set to target by
        # construction; behind the wall it is whatever the occluder leaves,
        # and that number is the context for `occ_recov_r2_*`. Measured on a
        # subsample because it costs two extra renders per pair.
        if i < N_CALIB:
            room_o.append(_rms(_ear(s_o, None, mute_voice=True), sr))
            alone = CA.ContactAudioSynth(model)
            alone.set_listener(L_HIDDEN, HEAD_YAW)
            alone.emit_voice(T_VOICE, hid, action, data=data)
            v_o.append(_rms(_ear(alone, None), sr))
    fx_c, fx_o, ys = np.array(fx_c), np.array(fx_o), np.array(ys)
    half = N_OCC // 2
    r2_occ = _ridge_r2(fx_c[:half], ys[:half], fx_o[half:], ys[half:])
    r2_clear = _ridge_r2(fx_c[:half], ys[:half], fx_c[half:], ys[half:])
    out = {f"occ_recov_r2_{d}": float(v) for d, v in zip(DIMS, r2_occ)}
    out.update({f"clear_recov_r2_{d}": float(v) for d, v in zip(DIMS, r2_clear)})
    out["occ_recov_r2_mean"] = float(np.mean(r2_occ))
    out["clear_recov_r2_mean"] = float(np.mean(r2_clear))
    out["occ_all_blocked"] = float(n_hidden_blocked == N_OCC)
    out["occ_voice_to_background_db"] = float(20.0 * math.log10(
        max(float(np.mean(v_o)), EPS) / max(float(np.mean(room_o)), EPS)))
    return out


# ── the experiment ──────────────────────────────────────────────────────
def _experiment(seed: int) -> dict:
    model, data = _world(seed)
    sr = CA.SAMPLE_RATE
    m: dict = {}

    # The geometry, checked by this file's own ray-caster rather than the
    # synth's: if the synth's occlusion call and its rendering shared a bug,
    # asking the synth whether the block is there would confirm the bug.
    mouth_hidden = L_HIDDEN + np.array([-OCC_D, 0.0, 0.0])
    mouth_lit = L_LIT + np.array([0.0, OCC_D, 0.0])
    hit = _hit_geom(model, data, mouth_hidden, L_HIDDEN)
    m["light_reaches_hidden"] = float(hit == "")
    m["hidden_occluder_is_block"] = float(hit == OCCLUDER_NAME)
    m["light_reaches_lit"] = float(_hit_geom(model, data, mouth_lit, L_LIT) == "")
    m["hidden_range_m"] = float(np.linalg.norm(mouth_hidden - L_HIDDEN))
    m["lit_range_m"] = float(np.linalg.norm(mouth_lit - L_LIT))
    m["ranges_match"] = float(abs(m["hidden_range_m"] - m["lit_range_m"]) < 1e-9)

    m.update(_recovery(seed, mute=False))
    m.update(_recovery(seed, mute=True))
    m.update(_silence(seed))
    m.update(_distance(seed))
    m.update(_occ_ref(seed))
    m.update(_occ_recovery(seed))

    m["occ_amp_ratio"] = float(m["ref_occ_rms"] / max(m["ref_clear_rms"], EPS))
    m["occ_centroid_drop"] = float(
        1.0 - m["ref_occ_centroid"] / max(m["ref_clear_centroid"], EPS))
    # REPORTED, NOT GATED. An SNR floor is a proxy for "he can still be heard",
    # and this spec measures that quantity DIRECTLY as occluded recovery with
    # the noise floor and the background present (LESSONS.md: measure the
    # quantity you are claiming, not a proxy that correlates with it). Taken
    # over the call's own extent, not the analysis window, because a signal's
    # level is not measured across an interval where it is absent.
    m["occ_snr"] = float(m["ref_occ_call_rms"] / EAR_NOISE_SIGMA)
    m["clear_snr"] = float(m["ref_clear_call_rms"] / EAR_NOISE_SIGMA)

    # THE DIFFICULTY OF THIS SPEC, MADE A MEASUREMENT. The voice alone (mouth
    # open, empty world) against the room alone (mouth shut, background
    # present), both over set A's pose distribution and both at the ear. v2
    # recorded -4.36 dB here without ever computing it; the gate below is
    # two-sided, so a future iteration cannot quiet the room to buy a PASS any
    # more than a loud room can hide the channel.
    m["bg_gain"] = _bg_gain(seed)
    m["voice_to_background_db"] = float(20.0 * math.log10(
        max(m["voiced_silent_rms"], EPS) / max(m["mute_ear_rms"], EPS)))

    # ears that are not saturating, and audio that is finite
    s = CA.ContactAudioSynth(model)
    s.set_listener(HEAD, HEAD_YAW)
    s.emit_voice(T_VOICE, np.array([HEAD[0] + 1.0, HEAD[1], HEAD[2]]),
                 (1.0, 1.0, 1.0, 1.0), data=data)
    loud = _ear(s, None)
    m["ear_finite"] = float(bool(np.all(np.isfinite(loud))))
    m["ear_peak"] = float(np.max(np.abs(loud)))
    m["ear_clip_frac"] = float(np.mean(np.abs(loud) >= 0.999999))

    # the registry's metric: recovery at the ear, discounted by how far the
    # measured attenuation strays from the law the fixture declares. 1.0 would
    # be perfect recovery through an exactly-declared channel.
    m["listener_recovery_x_attenuation_error"] = float(
        m["recov_r2_mean"] * (1.0 - min(1.0, m["dist_law_dev"] / DIST_LAW_TOL)))

    m["seed_gates_ok"] = float(
        # the spec is being run at the difficulty it declares — checked FIRST,
        # because every recovery number below is only meaningful at a stated
        # signal-to-interference ratio
        abs(m["voice_to_background_db"] - SIR_TARGET_DB) <= SIR_TOL_DB
        # the emission arrives, and every dimension of it survives the trip
        and all(m[f"recov_r2_{d}"] >= R2_MIN_PER_DIM for d in DIMS)
        and m["recov_r2_mean"] >= R2_MIN_MEAN
        # ...and the muted mouth is at chance, with ears at the noise floor
        and m["mute_r2_max"] <= MUTE_R2_MAX
        and m["mute_silent_rms"] <= MUTE_RMS_MULT * EAR_NOISE_SIGMA
        # ...and the same episodes with the mouth OPEN are far above it, or
        # the null passed because nothing was ever emitted
        and m["voiced_silent_rms"] >= 5.0 * m["mute_silent_rms"]
        # ...and it gets quieter the way the fixture says it does
        and m["dist_line_clear"] == 1.0
        and m["dist_monotone"] == 1.0
        and m["dist_law_dev"] <= DIST_LAW_TOL
        # ...and the inverse-square rival misses, so the gate discriminates
        and m["dist_dev_inverse_square"] >= DIST_INV2_DISCRIM_MIN
        # light does NOT reach the hidden listener, and it is the block
        and m["light_reaches_hidden"] == 0.0
        and m["hidden_occluder_is_block"] == 1.0
        and m["light_reaches_lit"] == 1.0
        and m["ranges_match"] == 1.0
        and m["ref_occ_flagged_occluded"] == 1.0
        and m["ref_clear_flagged_occluded"] == 0.0
        # ...the block attenuates, and does not silence
        and m["occ_amp_ratio"] <= OCC_RATIO_MAX
        and m["occ_amp_ratio"] >= OCC_RATIO_MIN
        # ...it MUFFLES: a low-pass, not a volume knob
        and m["occ_centroid_drop"] >= OCC_CENTROID_DROP_MIN
        # ...and he is still heard through it
        and m["occ_all_blocked"] == 1.0
        and m["occ_recov_r2_f0"] >= OCC_R2_MIN
        and m["occ_recov_r2_dur"] >= OCC_R2_MIN
        # ...on ears that are not clipping
        and m["ear_finite"] == 1.0
        and m["ear_peak"] <= 1.0
        and m["ear_clip_frac"] <= CLIP_FRAC_MAX)
    return m


# ── the controls: three sabotages, each aimed at one gate ───────────────
def _control(seed: int) -> dict:
    """Each arm disables one mechanism and nothing else; the gate that owns
    that mechanism must catch it. A gate no sabotage can trip is decorative
    (PG.5's precedent, and LESSONS.md on tripwires vs discriminations)."""
    c: dict = {}

    # 1. the wall does nothing
    r = _occ_ref(seed, voice_occlusion=False)
    ratio = r["ref_occ_rms"] / max(r["ref_clear_rms"], EPS)
    c["noocc_amp_ratio"] = float(ratio)
    c["control_occ_caught"] = float(ratio > OCC_RATIO_MAX)

    # 2. the wall is a volume knob, not a filter
    r = _occ_ref(seed, flat_occlusion=True)
    drop = 1.0 - r["ref_occ_centroid"] / max(r["ref_clear_centroid"], EPS)
    c["flat_centroid_drop"] = float(drop)
    c["control_flat_caught"] = float(drop < OCC_CENTROID_DROP_MIN)

    # 3. distance does nothing
    d = _distance(seed, voice_distance=False)
    c.update(d)
    c["control_dist_caught"] = float(d["nodist_dist_law_dev"] > DIST_LAW_TOL
                                     or d["nodist_dist_monotone"] == 0.0)
    return c


def _check(m: dict, c: dict) -> bool:
    return bool(
        m["seed_gates_ok"] == 1.0
        # the declared difficulty, restated at the aggregate
        and abs(m["voice_to_background_db"] - SIR_TARGET_DB) <= SIR_TOL_DB
        # the headline, restated at the aggregate so a lucky seed cannot carry it
        and all(m[f"recov_r2_{d}"] >= R2_MIN_PER_DIM for d in DIMS)
        and m["recov_r2_mean"] >= R2_MIN_MEAN
        and m["mute_r2_max"] <= MUTE_R2_MAX
        and m["mute_silent_rms"] <= MUTE_RMS_MULT * EAR_NOISE_SIGMA
        and m["dist_law_dev"] <= DIST_LAW_TOL
        and m["dist_dev_inverse_square"] >= DIST_INV2_DISCRIM_MIN
        and OCC_RATIO_MIN <= m["occ_amp_ratio"] <= OCC_RATIO_MAX
        and m["occ_centroid_drop"] >= OCC_CENTROID_DROP_MIN
        and m["occ_recov_r2_f0"] >= OCC_R2_MIN
        and m["occ_recov_r2_dur"] >= OCC_R2_MIN
        # every sabotage caught, on every seed
        and c["control_occ_caught"] == 1.0
        and c["control_flat_caught"] == 1.0
        and c["control_dist_caught"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["VO.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
