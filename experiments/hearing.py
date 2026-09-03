"""hearing.py — the audio front end and the A2 stem, shared by HR.7 and HR.6.

This module is the FORWARD CONTRACT between the bearing guard (HR.7) and the
representation bakeoff (HR.6): the stem HR.7 certifies must be byte-for-byte
the stem HR.6 trains, or the guard guards nothing. HR.6's A2 arm imports
`MelConvStem` from here; HR.7 declares this file in IMPL_DEPS so an edit to
the stem goes loudly stale on HR.7's certificate instead of silently invalid.

The representation (HEARING_BAKEOFF.md §4.2, arm A2, the incumbent from
UNIFIED_BRAIN.md §4): 2-channel log-mel, 64 bins, 25 ms window / 10 ms hop,
-> Conv2d stack -> 4 tokens. Token count 4 is HR.6's equalised budget
(arXiv:2601.16667 — unequal token budgets make a representation bakeoff a
comparison of token budgets).

TWO MEASURED TRAPS carried from HEARING_BAKEOFF.md §1.4.3 (2026-08-09, 108
PG.5-style drops on this box), stated here because this file is where a
refactor would silently reintroduce them:

  1. The pan law's log-domain interaural level difference is exactly
     atanh(p) (constant-power panning: gL=sqrt((1-p)/2), gR=sqrt((1+p)/2),
     so log gR - log gL = atanh(p), verified to machine precision). A LINEAR
     readout of bearing from log-mel saturates at the lateral extremes and
     scored 0.40 where the analytic tanh link scored 1.00. Any probe on this
     representation must be non-linear or must predict atanh(p) and invert.
  2. Pool bearing in the ENERGY domain, never by averaging per-bin LOG
     values: the log floor pins near-silent bins to zero ILD and drags the
     mean toward centre (measured 0.69 vs 1.00). `analytic_lateral` below is
     the correct pooling and doubles as HR.7's instrument-aliveness check.

The architectural failure mode both specs guard against: a stem whose first
op averages over the CHANNEL dimension IS the mono control, silently — one
line deleting Jack's only directional sense. `MelConvStem` keeps the two
channels as separate input planes of the first Conv2d for exactly this
reason; do not "simplify" them into a mean.
"""
from __future__ import annotations

import math

import numpy as np

SR = 16000            # ContactAudio.SAMPLE_RATE; asserted by callers, not here
N_MELS = 64
WIN_MS = 25.0
HOP_MS = 10.0
FMIN = 60.0           # below the synth's lowest fundamental band of interest
FMAX = 7600.0
LOG_FLOOR = 1e-6
N_TOKENS = 4
D_TOKEN = 64


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SR, n_fft: int | None = None,
                   n_mels: int = N_MELS, fmin: float = FMIN,
                   fmax: float = FMAX) -> np.ndarray:
    """Triangular mel filters (HTK scale), shape (n_mels, n_fft//2 + 1)."""
    if n_fft is None:
        n_fft = int(sr * WIN_MS / 1000.0)
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax),
                                   n_mels + 2))
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


def stereo_melpow(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """(2, T) waveform -> (2, N_MELS, frames) mel POWER (energy domain).

    Kept separate from the log so callers that pool bearing can pool here
    (trap 2 above). Hann-windowed frames, 25 ms / 10 ms.
    """
    assert audio.ndim == 2 and audio.shape[0] == 2, "expects (2, T) stereo"
    n_fft = int(sr * WIN_MS / 1000.0)
    hop = int(sr * HOP_MS / 1000.0)
    T = audio.shape[1]
    n_frames = max(1 + (T - n_fft) // hop, 1)
    window = np.hanning(n_fft)
    fb = mel_filterbank(sr, n_fft)
    out = np.zeros((2, N_MELS, n_frames))
    for ch in range(2):
        for t in range(n_frames):
            frame = audio[ch, t * hop:t * hop + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec = np.abs(np.fft.rfft(frame * window)) ** 2
            out[ch, :, t] = fb @ spec
    return out


def stereo_logmel(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """(2, T) waveform -> (2, N_MELS, frames) log-mel — the stem's input."""
    return np.log(stereo_melpow(audio, sr) + LOG_FLOOR)


def analytic_lateral(melpow: np.ndarray) -> float:
    """Lateral angle (rad) from mel POWER by energy-domain pooling + the
    exact atanh link. This is the correct decode from §1.4.3 (scored 1.00)
    and HR.7's instrument-aliveness reference: if THIS cannot read bearing
    from the mel of a stereo window, the front end or the window is broken
    and no stem may be blamed.

      pooled ILD = 0.5 * log(E_R / E_L) = atanh(p)   ->   p = tanh(ILD)
      pan p = -sin(azimuth)  (ContactAudio pan law)  ->  lateral = -asin(p)
    """
    e_l = float(melpow[0].sum())
    e_r = float(melpow[1].sum())
    if e_l <= 0.0 or e_r <= 0.0:
        return 0.0
    p = math.tanh(0.5 * math.log(e_r / e_l))
    return -math.asin(max(-1.0, min(1.0, p)))


_STEM_CLASS = None


def stem_class():
    """The A2 stem class. Torch is imported here, lazily, so listings that
    import this module for its constants do not pay for it."""
    global _STEM_CLASS
    if _STEM_CLASS is not None:
        return _STEM_CLASS
    import torch.nn as nn
    import torch.nn.functional as F

    class MelConvStem(nn.Module):
        """2-channel log-mel -> 4 tokens of 64 dims (~101K params).

        The two stereo channels enter as the two INPUT PLANES of conv1 —
        never averaged (see module docstring). Frequency is strided down
        64 -> 8, time is preserved and adaptive-pooled to N_TOKENS.
        """

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(2, 32, 3, stride=(2, 1), padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1), nn.ReLU(),
            )
            self.proj = nn.Linear(128, D_TOKEN)

        def forward(self, x):            # (B, 2, N_MELS, T)
            h = self.conv(x)             # (B, 128, N_MELS/8, T)
            h = h.mean(dim=2)            # pool FREQUENCY (not channels!)
            h = F.adaptive_avg_pool1d(h, N_TOKENS)   # (B, 128, N_TOKENS)
            return self.proj(h.transpose(1, 2))      # (B, N_TOKENS, D_TOKEN)

    _STEM_CLASS = MelConvStem
    return MelConvStem


def make_stem(seed: int):
    """The A2 stem at deterministic random init (HR.7 probes it untrained —
    the guard is architectural: does the wiring destroy bearing before
    training ever starts?). Returns an eval-mode torch module."""
    import torch

    torch.manual_seed(seed * 7919 + 11)
    stem = stem_class()()
    stem.eval()
    return stem
