"""Procedural contact audio: the playground makes sound when things hit things.

Serves GOAL.md ("all senses, one brain") by giving Jack's world a hearable
physics channel. There is no audio hardware in the loop and no recorded samples:
sound is SYNTHESIZED from MuJoCo contact events, so every sound has exact
ground-truth provenance — which geom, where, how hard. That provenance is the
point: UB.4 ("hearing is load-bearing") needs audio whose localization labels
are certain, and PG.5 certifies this module before anything trains on it.

Method — modal resonance (van den Doel & Pai, "The sounds of physical shapes",
1998): a struck object rings as a bank of exponentially decaying sinusoids.
Here each geom's fundamental comes from its characteristic size (small things
ring high), partials follow free-bar ratios, and impact force sets amplitude.
Of a contacting pair, the SMALLER geom is voiced — the object rings, not the
floor. Surface (floor) modes are deliberately absent in v1.

Spatialization — constant-power stereo panning by the LATERAL angle of the
source in the listener frame, plus 1/distance attenuation. Panning encodes
left/right ONLY: front-back disambiguation needs ITD/spectral cues (future
work). Event labels nonetheless carry the full azimuth, elevation and distance,
so a learner may be trained on more than the pan can express.

Conventions (pre-registered, tested by PG.5):
  azimuth    atan2(rel . left, rel . forward), positive to the listener's LEFT
  lateral    asin(sin(azimuth)) — azimuth folded to [-pi/2, pi/2]
  pan p      -sin(azimuth), so p=+1 is hard RIGHT
  gains      gL = sqrt((1-p)/2), gR = sqrt((1+p)/2)  (constant power)
Decoding therefore is: p_hat = (ER-EL)/(EL+ER), lateral_hat = -asin(p_hat).

Voice (VO.01) — the same stream, from the other end. Jack VOCALISES by writing
a 4-D action (f0, brightness, amplitude, duration) into `emit_voice`, which is
synthesised as a harmonic call, panned and attenuated by the SAME laws as a
contact, and mixed into the SAME buffer. A listener therefore receives his
voice as stereo samples and nothing else — there is no side channel between two
brains wearing the word "voice". Occlusion is one `mj_ray` against the geometry
the eye uses, and it MUFFLES rather than silences (see OCC_TRANSMISSION).

Usage:
    synth = ContactAudioSynth(model)
    synth.set_listener(pos=[0,0,1.4], yaw=0.7)   # later: Jack's head pose
    for _ in range(steps):
        mujoco.mj_step(model, data)
        synth.step(data)                          # detect contact onsets
    stereo = synth.render(duration=2.0)           # [2, T] float in [-1, 1]
    labels = synth.events                         # ground truth per event
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
VOICE_SECONDS = 0.30        # ring length per event
MODE_RATIOS = (1.0, 2.76, 5.40, 8.93)     # free-bar partials
MODE_GAINS = (1.0, 0.5, 0.25, 0.125)
TAU0 = 0.06                 # fundamental decay time constant, s
REFRACTORY_S = 0.10         # per contact-pair retrigger guard
MIN_DISTANCE = 0.5          # attenuation floor, m
AMP_K = 0.03                # amp = min(1, AMP_K * sqrt(F_normal))

# ── VOICE: the effector half of hearing (VO.01) ─────────────────────────
# GOAL.md lists voice among the constitutional senses precisely because it is
# the only one that is an EFFECTOR: "how a creature acts on other creatures".
# A vocalisation is not a message in a side channel — it is synthesised here,
# spatialized by the same pan law as a contact, attenuated by the same 1/r, and
# it arrives at the listener as nothing but stereo samples. Whatever a listener
# knows about it, it knows through its ears.
#
# The emission is FOUR CONTINUOUS ACTION DIMENSIONS, so a policy can drive it:
#   a[0] -> f0          fundamental, log-spaced over VOICE_F0_HZ
#   a[1] -> brightness  spectral tilt: harmonic h has amplitude h**-tilt
#   a[2] -> amp         emitted amplitude, log-spaced over VOICE_AMP
#   a[3] -> duration    linear over VOICE_DUR_S
# Deliberately NOT a symbolic channel (registry VO.01): an emergent protocol
# has to survive distance, occlusion and the listener's own encoder.
VOICE_ACTION_DIM = 4
VOICE_F0_HZ = (80.0, 700.0)
VOICE_TILT = (3.0, 0.4)     # brightness 0 -> tilt 3.0 (dark); 1 -> 0.4 (bright)
VOICE_AMP = (0.05, 1.0)     # fraction of VOICE_RMS_FULL, NOT a peak
VOICE_DUR_S = (0.10, 0.60)
VOICE_HARMONICS = 24
VOICE_RAMP_S = 0.02         # raised-cosine attack/release, so duration is the
                            # envelope length rather than a decay constant

# THE CALL IS NORMALISED TO CONSTANT RMS, and this is a correction, not a
# detail. v1 scaled by `amp / sum(weights)`, which made LOUDNESS A FUNCTION OF
# TIMBRE: at identical `amp` a bright call came out 3.8x quieter at the mouth
# than a dark one (measured), so two action dimensions that are supposed to be
# independent were entangled, and bright calls sat systematically nearer the
# masking floor. VO.01's first recorded run reported brightness recovery of
# 0.347 against a 0.50 gate because of it. `amp` now means loudness and
# `brightness` means timbre, which is what a policy driving four dimensions is
# entitled to assume.
#
# VOICE_RMS_FULL is DERIVED, not chosen: the loudest call at the closest
# representable range must peak below full scale.
#     0.9 / (max crest factor 2.0 * max distance gain 1/MIN_DISTANCE = 2.0)
# Schroeder phases (Schroeder 1970, "Synthesis of low-peak-factor signals")
# are what make 2.0 available — with all harmonics phase-aligned the same call
# is a pulse train and crests at 4.1, which would clip at close range and break
# the linearity every attenuation measurement assumes.
VOICE_RMS_FULL = 0.225

# Occlusion, declared here and gated by VO.01. A solid between mouth and ear
# transmits sound — it does not stop it, which is the whole reason voice and
# smell are worth having when sight fails — and it transmits LOW frequencies
# better than high (the mass law). So a wall does not silence him; it MUFFLES
# him, and the received spectral centroid falling is the falsifiable signature
# that separates this from a flat volume knob. VO.01 carries a flat-occluder
# control that must miss that gate.
#   gain(f) = OCC_TRANSMISSION * min(1, (OCC_FREF_HZ / f) ** OCC_ALPHA)
# KNOWN MODEL GAP, stated rather than hidden: this is transmission only. Real
# low-frequency sound also DIFFRACTS around a small obstacle, so a 0.3 m block
# in the real world attenuates far less than this model says. VO.01 therefore
# claims the ordering (sound crosses what light does not) and not the decibels.
OCC_TRANSMISSION = 0.32
OCC_FREF_HZ = 250.0
OCC_ALPHA = 1.0


@dataclass
class VoiceParams:
    """Physical emission parameters, decoded from an action vector."""
    f0: float               # Hz
    brightness: float       # [0, 1]; 1 is bright
    amp: float              # [0, 1] peak amplitude at the mouth
    duration: float         # s

    @property
    def tilt(self) -> float:
        return VOICE_TILT[0] + (VOICE_TILT[1] - VOICE_TILT[0]) * self.brightness


def voice_params_from_action(action) -> VoiceParams:
    """Map a 4-D action in [-1, 1] to physical emission parameters.

    f0 and amp are LOG-spaced because both are perceived that way and because a
    linear map would spend most of its action range on differences no listener
    could resolve.
    """
    a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
    if a.size != VOICE_ACTION_DIM:
        raise ValueError(f"voice action must have {VOICE_ACTION_DIM} dims, got {a.size}")
    u = (a + 1.0) / 2.0
    f0 = VOICE_F0_HZ[0] * (VOICE_F0_HZ[1] / VOICE_F0_HZ[0]) ** u[0]
    amp = VOICE_AMP[0] * (VOICE_AMP[1] / VOICE_AMP[0]) ** u[2]
    dur = VOICE_DUR_S[0] + (VOICE_DUR_S[1] - VOICE_DUR_S[0]) * u[3]
    return VoiceParams(f0=float(f0), brightness=float(u[1]), amp=float(amp),
                       duration=float(dur))


@dataclass
class VoiceEvent:
    """One vocalisation, with its ground-truth localization labels."""
    t: float                # sim time, s
    pos: np.ndarray         # mouth position, world frame [3]
    action: np.ndarray      # the 4-D action that produced it
    params: VoiceParams
    azimuth: float          # rad, listener frame, positive left
    lateral: float          # rad, azimuth folded to [-pi/2, pi/2]
    elevation: float        # rad, positive up
    distance: float         # m
    occluded: bool          # is there a solid between mouth and ear?
    occluder_geom: int      # which one, or -1


@dataclass
class AudioEvent:
    """One contact onset, with its ground-truth localization labels."""
    t: float                # sim time, s
    geom1: int
    geom2: int
    voiced_geom: int        # the smaller geom of the pair — the one that rings
    pos: np.ndarray         # contact position, world frame [3]
    force: float            # solver normal force at onset, N
    amp: float              # synthesis amplitude in [0, 1]
    azimuth: float          # rad, listener frame, positive left
    lateral: float          # rad, azimuth folded to [-pi/2, pi/2]
    elevation: float        # rad, positive up
    distance: float         # m


class ContactAudioSynth:
    """Detects contact onsets and renders them as spatialized stereo audio."""

    def __init__(self, model, sample_rate: int = SAMPLE_RATE):
        self.model = model
        self.sr = sample_rate
        self.listener_pos = np.array([0.0, 0.0, 1.4])
        self.listener_yaw = 0.0
        self.events: List[AudioEvent] = []
        self.voice_events: List[VoiceEvent] = []
        self._prev_pairs: set = set()
        self._last_fired: Dict[Tuple[int, int], float] = {}
        # Characteristic size per geom: geometric mean of its nonzero size
        # entries. Crude but monotone in actual extent for every geom type,
        # which is all the fundamental needs.
        self._char_size = np.ones(model.ngeom)
        for gid in range(model.ngeom):
            gs = np.asarray(model.geom_size[gid], dtype=float)
            nz = gs[gs > 1e-6]
            self._char_size[gid] = float(np.exp(np.mean(np.log(nz)))) if len(nz) else 1.0

    def set_listener(self, pos, yaw: float) -> None:
        self.listener_pos = np.asarray(pos, dtype=float).copy()
        self.listener_yaw = float(yaw)

    def fundamental(self, gid: int) -> float:
        """Fundamental frequency of a geom's modal bank, Hz."""
        return float(np.clip(180.0 / self._char_size[gid], 80.0, 4000.0))

    def localize(self, pos) -> Tuple[float, float, float, float]:
        """(azimuth, lateral, elevation, distance) of a world point in the
        listener frame. ONE implementation, shared by contacts and by voice, so
        the two cannot drift into disagreeing about where the world is."""
        rel = np.asarray(pos, dtype=float) - self.listener_pos
        cy, sy = math.cos(self.listener_yaw), math.sin(self.listener_yaw)
        fwd = rel[0] * cy + rel[1] * sy
        left = -rel[0] * sy + rel[1] * cy
        azimuth = math.atan2(left, fwd)
        horiz = math.hypot(rel[0], rel[1])
        return (azimuth, math.asin(math.sin(azimuth)),
                math.atan2(rel[2], horiz), float(np.linalg.norm(rel)))

    # ── contact onset detection ──────────────────────────────────────────

    def step(self, data) -> None:
        """Call after each mj_step. Fires an event when a contact pair newly
        appears (and is past its refractory window). Resting/rolling contact
        persists as an active pair and does not retrigger."""
        import mujoco
        t = float(data.time)
        # Collect ALL of each pair's contact points this step before firing:
        # a box landing flat emits up to 4 corner contacts at once, and taking
        # the first corner as the source puts it up to half a diagonal off the
        # object — measured 0.26 m, a 9-degree bearing error at 1.5 m. The
        # acoustic source is the contact CENTROID.
        by_pair: Dict[Tuple[int, int], list] = {}
        for i in range(data.ncon):
            con = data.contact[i]
            pair = (min(con.geom1, con.geom2), max(con.geom1, con.geom2))
            f6 = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, f6)
            by_pair.setdefault(pair, []).append((np.array(con.pos), abs(float(f6[0]))))
        for pair, pts in by_pair.items():
            if pair in self._prev_pairs:
                continue
            if t - self._last_fired.get(pair, -1e9) < REFRACTORY_S:
                continue
            pos = np.mean([p for p, _ in pts], axis=0)
            force = float(sum(f for _, f in pts))
            self._last_fired[pair] = t
            self.events.append(self._make_event(t, pair, pos, force))
        self._prev_pairs = set(by_pair)

    def _make_event(self, t: float, pair: Tuple[int, int], pos: np.ndarray,
                    force: float) -> AudioEvent:
        g1, g2 = pair
        voiced = g1 if self._char_size[g1] <= self._char_size[g2] else g2
        azimuth, lateral, elevation, dist = self.localize(pos)
        amp = min(1.0, AMP_K * math.sqrt(max(force, 1.0)))
        return AudioEvent(t=t, geom1=g1, geom2=g2, voiced_geom=voiced, pos=pos,
                          force=force, amp=amp, azimuth=azimuth,
                          lateral=lateral, elevation=elevation, distance=dist)

    # ── voice: he makes a sound ──────────────────────────────────────────

    def emit_voice(self, t: float, pos, action, data=None) -> VoiceEvent:
        """Vocalise at `pos` with a 4-D action, into the SAME stereo stream.

        `data` (an `mjData` at the pose the emission happens in) enables the
        occlusion test: one `mj_ray` from mouth to ear, against the same
        geometry and the same `flg_static` the eye's ray-caster uses, so
        "sound crosses what light does not" is a measurement rather than a
        flag. Without `data` the emission is treated as unoccluded.
        """
        pos = np.asarray(pos, dtype=float).copy()
        azimuth, lateral, elevation, dist = self.localize(pos)
        occluded, gid = False, -1
        if data is not None:
            occluded, gid = self.occluded_by(data, pos)
        e = VoiceEvent(t=float(t), pos=pos,
                       action=np.asarray(action, dtype=float).reshape(-1).copy(),
                       params=voice_params_from_action(action),
                       azimuth=azimuth, lateral=lateral, elevation=elevation,
                       distance=dist, occluded=occluded, occluder_geom=gid)
        self.voice_events.append(e)
        return e

    def occluded_by(self, data, pos) -> Tuple[bool, int]:
        """(is the mouth->ear line blocked, by which geom). -1 when clear."""
        import mujoco
        a = np.asarray(pos, dtype=float)
        delta = self.listener_pos - a
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-9:
            return False, -1
        gid = np.zeros(1, dtype=np.int32)
        hit = mujoco.mj_ray(self.model, data, a, delta / dist, None, 1, -1, gid)
        if 0.0 <= hit < dist - 1e-6 and gid[0] >= 0:
            return True, int(gid[0])
        return False, -1

    def _voice_wave(self, e: VoiceEvent, occlusion: bool = True,
                    flat_occlusion: bool = False) -> np.ndarray:
        """Mono harmonic call for one vocalisation, occlusion applied per
        harmonic (that is where the muffling lives)."""
        p = e.params
        n = max(1, int(p.duration * self.sr))
        t = np.arange(n) / self.sr
        h = np.arange(1, VOICE_HARMONICS + 1, dtype=float)
        f = p.f0 * h
        keep = f < 0.45 * self.sr
        h, f = h[keep], f[keep]
        w = h ** (-p.tilt)

        # Constant-RMS normalisation, from the UNOCCLUDED weights: occlusion
        # has to attenuate, so it is applied after the call's level is fixed.
        scale = VOICE_RMS_FULL * p.amp / math.sqrt(max(np.sum(w ** 2) / 2.0, 1e-30))
        g = np.ones_like(w)
        if occlusion and e.occluded:
            g = np.full_like(w, OCC_TRANSMISSION) if flat_occlusion else (
                OCC_TRANSMISSION * np.minimum(1.0, (OCC_FREF_HZ / f) ** OCC_ALPHA))

        # Generalised Schroeder phases: phi_i = -2*pi * sum_{k<i} (h_i - h_k) p_k
        pw = w ** 2 / max(float(np.sum(w ** 2)), 1e-30)
        cum_p = np.concatenate([[0.0], np.cumsum(pw)[:-1]])
        cum_hp = np.concatenate([[0.0], np.cumsum(h * pw)[:-1]])
        phi = -2 * math.pi * (h * cum_p - cum_hp)

        sig = (scale * w * g) @ np.sin(
            2 * math.pi * f[:, None] * t[None, :] + phi[:, None])
        # raised-cosine attack/release, so `duration` IS the envelope length
        r = min(int(VOICE_RAMP_S * self.sr), n // 2)
        if r > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
            sig[:r] *= ramp
            sig[n - r:] *= ramp[::-1]
        return sig

    # ── synthesis ────────────────────────────────────────────────────────

    def _voice(self, e: AudioEvent) -> np.ndarray:
        """Mono modal ring for one event."""
        n = int(VOICE_SECONDS * self.sr)
        t = np.arange(n) / self.sr
        f0 = self.fundamental(e.voiced_geom)
        sig = np.zeros(n)
        total_gain = 0.0
        for ratio, gain in zip(MODE_RATIOS, MODE_GAINS):
            f = f0 * ratio
            if f >= 0.45 * self.sr:
                break
            sig += gain * np.exp(-t / (TAU0 / ratio)) * np.sin(2 * math.pi * f * t)
            total_gain += gain
        if total_gain > 0:
            sig *= e.amp / total_gain
        return sig

    def render(self, duration: float, mode: str = "stereo",
               pan_override: Optional[Dict[int, float]] = None,
               mute_voice: bool = False, voice_occlusion: bool = True,
               voice_distance: bool = True,
               flat_occlusion: bool = False) -> np.ndarray:
        """Render all events — contacts and vocalisations — into one stereo
        buffer [2, T]. One stream: a listener cannot tell them apart except by
        listening.

        The keyword arguments are NULL ARMS, and they exist only so a spec can
        sabotage a mechanism and prove its gate can report the bad case.
        mode="mono" duplicates the mid signal into both channels (bearing must
        become undecodable). pan_override maps event index -> pan in [-1, 1]
        (the shuffled-pan null). mute_voice renders the mouth shut (VO.01's
        null baseline). voice_occlusion=False lets the wall do nothing;
        voice_distance=False removes 1/r; flat_occlusion=True makes the wall a
        volume knob instead of a low-pass — each is a rival the corresponding
        VO.01 gate must catch.
        """
        T = int(duration * self.sr)
        buf = np.zeros((2, T))
        if not mute_voice:
            for e in self.voice_events:
                start = int(e.t * self.sr)
                if start >= T:
                    continue
                sig = self._voice_wave(e, occlusion=voice_occlusion,
                                       flat_occlusion=flat_occlusion)
                n = min(len(sig), T - start)
                if mode == "mono":
                    gl = gr = math.sqrt(0.5)
                else:
                    p = -math.sin(e.azimuth)
                    gl = math.sqrt((1.0 - p) / 2.0)
                    gr = math.sqrt((1.0 + p) / 2.0)
                g = 1.0 / max(e.distance, MIN_DISTANCE) if voice_distance else 1.0
                buf[0, start:start + n] += gl * g * sig[:n]
                buf[1, start:start + n] += gr * g * sig[:n]
        for i, e in enumerate(self.events):
            start = int(e.t * self.sr)
            if start >= T:
                continue
            sig = self._voice(e)
            n = min(len(sig), T - start)
            if mode == "mono":
                gl = gr = math.sqrt(0.5)
            else:
                p = -math.sin(e.azimuth)
                if pan_override is not None and i in pan_override:
                    p = float(np.clip(pan_override[i], -1.0, 1.0))
                gl = math.sqrt((1.0 - p) / 2.0)
                gr = math.sqrt((1.0 + p) / 2.0)
            g = 1.0 / max(e.distance, MIN_DISTANCE)
            buf[0, start:start + n] += gl * g * sig[:n]
            buf[1, start:start + n] += gr * g * sig[:n]
        return np.clip(buf, -1.0, 1.0)


def decode_lateral(stereo: np.ndarray) -> float:
    """Recover the lateral angle from a stereo window by channel energy ratio.
    Inverts the constant-power pan law; the ONLY information used is L/R energy."""
    el = float(np.sum(stereo[0] ** 2))
    er = float(np.sum(stereo[1] ** 2))
    if el + er <= 0:
        return 0.0
    p = (er - el) / (el + er)
    return -math.asin(max(-1.0, min(1.0, p)))
