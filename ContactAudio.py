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
        rel = pos - self.listener_pos
        cy, sy = math.cos(self.listener_yaw), math.sin(self.listener_yaw)
        fwd = rel[0] * cy + rel[1] * sy
        left = -rel[0] * sy + rel[1] * cy
        azimuth = math.atan2(left, fwd)
        dist = float(np.linalg.norm(rel))
        horiz = math.hypot(rel[0], rel[1])
        elevation = math.atan2(rel[2], horiz)
        amp = min(1.0, AMP_K * math.sqrt(max(force, 1.0)))
        return AudioEvent(t=t, geom1=g1, geom2=g2, voiced_geom=voiced, pos=pos,
                          force=force, amp=amp, azimuth=azimuth,
                          lateral=math.asin(math.sin(azimuth)),
                          elevation=elevation, distance=dist)

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
               pan_override: Optional[Dict[int, float]] = None) -> np.ndarray:
        """Render all events into a stereo buffer [2, T].

        mode="mono" duplicates the mid signal into both channels (a null arm:
        bearing must become undecodable). pan_override maps event index -> pan
        in [-1, 1], replacing the truth-derived pan (the shuffled-pan null arm).
        """
        T = int(duration * self.sr)
        buf = np.zeros((2, T))
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
