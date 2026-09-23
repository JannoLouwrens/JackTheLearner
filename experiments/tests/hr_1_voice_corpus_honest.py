"""HR.1 — the voice corpus is honest before anyone is scored.

THE CLAIM (registry, verbatim intent): a speaker corpus exists ON THIS BOX with
>=8 enrolled and >=8 held-out UNKNOWN speakers, disjoint enrolment/test
utterances, CROSS-SESSION test material (cross-CHAPTER for LibriSpeech — the
registry notes say why), and a NOISE/REVERB stratum — such that no NON-VOCAL
channel cue can identify a speaker in either stratum. If a probe on channel
features alone reads a speaker above chance+5%, every speaker-ID number
downstream (HR.3, HR.4) is a microphone measurement, not a voice measurement,
and this spec's `kills` field fires exactly as written.

THE CORPUS: LibriSpeech dev-clean (CC-BY-4.0), fetched 2026-09-18 under the
owner's D19 ruling ("yes may download anything to /data", 2026-09-17), 40
speakers / ~350 MB extracted at CORPUS_ROOT. The tarball was byte-exact against
the registry's recorded Content-Length (337,926,286) and was deleted after
extraction per the tenant rule. 31 of 40 speakers carry >=2 chapters and are
eligible for cross-chapter enrolment; the rest can only serve in the UNKNOWN
pool. THE FETCH IS NOT PART OF THIS RUN — the spec is Budget.CPU (cpu<10min)
and certifies the corpus that exists; if the corpus is absent the run returns
VOID with the fetch instruction in its metrics, never a verdict.

THE DELIVERY CONTRACT — the fixture is code+data, and this module is the code.
`load_clip(path, ...)` is the ONLY sanctioned way to read this corpus, and
HR.2/HR.3/HR.4 must consume clips through it. It delivers: first <=6.0 s,
float32, DC-REMOVED and RMS-NORMALISED to TARGET_RMS. Two of the four channel
cues named in `falsified_by` (DC offset, clip loudness) are therefore killed BY
CONSTRUCTION — the probe still carries both features so the run PROVES they are
dead rather than asserting it. The other two (silence-segment spectrum, noise
floor) survive normalisation up to a common scale and are what this spec
actually measures.

THE PROBE (deliberately non-vocal): per delivered clip, 17 features —
12 log-spaced band energies (50 Hz–8 kHz) of the average power spectrum of the
QUIETEST 10% of 25 ms frames (the silence-segment spectrum), the noise-floor
level, an estimated speech-to-floor SNR, delivered DC, delivered RMS, and a
clipping fraction (flat-top artifact; survives normalisation). A multinomial
logistic probe (torch, deterministic per seed) trains on the enrolment clips
and is scored on: (a) CLEAN cross-chapter test clips, (b) the NOISE/REVERB
stratum of the same clips, (c) the LEAKY CONTROL — held-out clips drawn from
the ENROLMENT chapters themselves, i.e. same-session test material. The probe
training is IDENTICAL for all three reads; only the test material moves, so
the control isolates exactly the session variable the claim is about.

THE NOISE/REVERB STRATUM (registry: NOT optional): one synthetic room per seed
— a unit-energy exponential-decay RIR at RT60 0.4 s, the SAME room for every
speaker so the stratum adds no discriminative cue — then per-clip pink noise
at exactly 15 dB SNR (SVeritas' harsh setting), then redelivery through the
same normalisation contract. Synthesised with numpy/scipy alone, no download.

GATES, all pre-registered here and none movable:
  chance          = 1 / n_enrolled                (registry null_baseline)
  LEAK bar        = chance + 0.05                 (registry falsified_by)
  PASS            = probe accuracy <= LEAK bar in BOTH strata on EVERY seed,
                    i.e. min_channel_leak_margin >= 0.0
  CONTROL floor   = chance + 0.15: the same probe on same-session clips must
                    read AT OR ABOVE this on EVERY seed, else the run is VOID —
                    a leak detector that cannot see a planted leak has measured
                    nothing (LESSONS.md, T0.13; the at-chance-control rule,
                    24th audit: an at-chance experiment reading needs proof the
                    instrument was alive, and the control IS that proof).
  VOID, never FAIL, when the rig cannot testify: corpus absent, <8 enrolled
  after per-speaker minimums, <8 unknown, any non-finite audio, a noise
  stratum that did not measurably transform the audio, or the control floor
  above.

PRE-STATED EXPECTED VERDICT (the T2.05 discipline): FAIL on the clean stratum
is the LIKELIER branch. LibriVox readers record all their chapters on their own
constant equipment, so cross-chapter is cross-session in time but often NOT in
channel; if the silence-floor spectrum identifies readers above chance+5%,
that is a true fact about this corpus and it is exactly what this spec exists
to catch BEFORE HR.3 buys a speaker-ID number on it. A PASS would certify the
20/20 fixture as delivered; a FAIL kills HR.2-4 as registered and routes the
repair (channel equalisation in the delivery contract, or a different corpus)
through the Review — the bar itself never moves. Either branch is a real
measurement about the fixture, which is why the spec is tier 2 and cheap.

Gate purity (the LG.02 standard): every gated value is recorded per seed in
the row, verbatim, alongside the aggregated headline metric.

Artifact: /data/jack_corpora/librispeech/hr1_manifest.json — the split (per
seed), the delivery contract constants, and a corpus census (file count +
total bytes) so a later run can detect the corpus changed under the
certificate.

ARM (a) OF THE FIXTURE-REDESIGN DISPOSITION (Review DAILY 2026-09-23, row
`hr1-clean-stratum-is-a-microphone-measurement`; implemented by the builder
the same day). Attempt 2 FAILed on its pre-stated branch: the clean
cross-chapter stratum read 0.2375/0.3812/0.4268 against the 0.10 bar —
LibriVox equipment is per-reader constant, so cross-chapter is cross-session
in time but not in channel. The ordered repair: PER-CLIP QUIET-FLOOR SPECTRAL
WHITENING added to the delivery contract. `load_clip` now estimates the
noise-floor power spectrum from the quietest QUIET_FRAC of frames (the same
framing the probe uses), builds the zero-phase inverse filter (smoothed
WHITEN_SMOOTH_BINS bins, regularised WHITEN_REG relative to the mean floor
power — both constants declared here BEFORE the run and not revisited), and
applies it to the whole clip, then re-delivers. This flattens the
silence-segment SPECTRUM by construction; it deliberately does NOT touch the
floor LEVEL, the SNR, or the clipping fraction — those cues stay measurable,
which is what should keep the planted same-session leak control alive.
EVERY GATE IS UNCHANGED so the run can FAIL: the 0.10 clean bar, the 0.20
planted-leak floor (chance+0.15), the noise stratum, the 17 features, the
probe, the seeds. Two pre-registered outcomes, both results, neither moving
a threshold:
  - clean stratum below the 0.10 bar AND the planted leak still above its
    0.20 floor -> the confound is removable in-corpus; HR.2-HR.4 unblock on
    LibriSpeech and VCTK's 11.7 GB is not spent;
  - the leak control at or below 0.20 (the run returns VOID: instrument
    dead), OR the clean stratum still at/above 0.10 -> arm (a) is REFUTED,
    not a tuning opportunity. Do NOT re-tune the whitener; arm (c) — VCTK,
    genuinely multi-session — fires automatically per the disposition,
    without a further Review sitting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

CORPUS_ROOT = Path("/data/jack_corpora/librispeech/LibriSpeech/dev-clean")
MANIFEST = Path("/data/jack_corpora/librispeech/hr1_manifest.json")

SEEDS = (0, 1, 2)                 # registry: seeds=3
SR = 16000
CLIP_S = 6.0                      # delivered length cap
MIN_CLIP_S = 3.0                  # shorter files are not sampled
TARGET_RMS = 0.05                 # delivery contract
N_ENROLLED_TARGET = 20            # registry notes: 20/20 split of dev-clean
N_ENROLLED_MIN = 8                # hypothesis floor
N_UNKNOWN_MIN = 8                 # hypothesis floor
N_ENROL_CLIPS = 8                 # per speaker, from enrolment chapters
N_TEST_CLIPS = 8                  # per speaker, from DISJOINT chapters
N_CTRL_CLIPS = 8                  # per speaker, same-chapter held-out (control)
PER_SPK_MIN = 4                   # a speaker below this in enrol or test drops
CTRL_SPK_MIN = 2                  # min same-chapter held-out clips to join control
LEAK_EXCESS = 0.05                # falsified_by: "above chance+5%"
CONTROL_FLOOR_EXCESS = 0.15       # planted leak must read 3x the allowed excess
NOISE_SNR_DB = 15.0               # SVeritas' environmental setting
RT60_S = 0.4
N_BANDS = 12
QUIET_FRAC = 0.10
FRAME, HOP = 400, 160             # 25 ms / 10 ms at 16 kHz
PROBE_STEPS = 600
PROBE_LR = 0.05
PROBE_WD = 1e-3
WHITEN_REG = 1e-3                 # arm (a): rel. regularisation of the floor
WHITEN_SMOOTH_BINS = 5            # arm (a): floor-spectrum smoothing window

_FEAT_CACHE: dict[str, "object"] = {}   # path -> clean delivered features
_BUNDLE: dict | None = None


# ── the delivery contract ──────────────────────────────────────────────────

def load_clip(path: str | Path, max_s: float = CLIP_S):
    """The ONLY sanctioned reader for this corpus. DC-removed, RMS-normalised
    float32, first `max_s` seconds, QUIET-FLOOR SPECTRALLY WHITENED (arm (a),
    2026-09-23). HR.2/HR.3/HR.4 must use this."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(path), frames=int(max_s * SR), dtype="float32",
                    always_2d=False)
    if sr != SR:
        raise RuntimeError(f"{path}: sample rate {sr} != {SR}")
    x = np.asarray(x, dtype=np.float64)
    return deliver(_whiten_quiet_floor(deliver(x)))


def _whiten_quiet_floor(x):
    """Arm (a): flatten the clip's own silence-floor spectrum. Estimates the
    noise-floor power spectrum from the quietest QUIET_FRAC of frames (the
    probe's framing), builds a smoothed, regularised zero-phase inverse
    filter, applies it to the whole clip. Touches the floor's SHAPE only —
    level, SNR and clipping cues are left for the probe to measure."""
    import numpy as np
    n = (len(x) - FRAME) // HOP + 1
    if n < 10:
        return x
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx]
    energy = np.mean(frames * frames, axis=1)
    order = np.argsort(energy)
    n_quiet = max(5, int(QUIET_FRAC * n))
    win = np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(frames[order[:n_quiet]] * win, axis=1)) ** 2
    floor = np.mean(spec, axis=0)
    kernel = np.ones(WHITEN_SMOOTH_BINS) / WHITEN_SMOOTH_BINS
    floor = np.convolve(floor, kernel, mode="same")
    mean_floor = float(np.mean(floor))
    if mean_floor <= 0.0:
        return x
    gain = np.sqrt(mean_floor / (floor + WHITEN_REG * mean_floor))
    freqs_frame = np.fft.rfftfreq(FRAME, 1.0 / SR)
    freqs_clip = np.fft.rfftfreq(len(x), 1.0 / SR)
    X = np.fft.rfft(x) * np.interp(freqs_clip, freqs_frame, gain)
    y = np.fft.irfft(X, len(x))
    return y if np.all(np.isfinite(y)) else x


def deliver(x):
    """Normalisation half of the contract, applied to ANY audio entering the
    fixture (incl. the noise stratum after augmentation)."""
    import numpy as np
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x)))
    if rms > 1e-9:
        x = x * (TARGET_RMS / rms)
    return x


# ── the non-vocal channel probe features ───────────────────────────────────

def _features(x) -> "object":
    """17 non-vocal features of a DELIVERED clip. No pitch, no formants, no
    content — silence-floor spectrum, levels, and clipping only."""
    import numpy as np
    n = (len(x) - FRAME) // HOP + 1
    if n < 10:
        return None
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx]                                    # (n, FRAME)
    energy = np.mean(frames * frames, axis=1)
    order = np.argsort(energy)
    n_quiet = max(5, int(QUIET_FRAC * n))
    quiet = frames[order[:n_quiet]]
    loud = frames[order[n // 2:]]
    win = np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(quiet * win, axis=1)) ** 2
    mean_spec = np.mean(spec, axis=0)                  # (FRAME//2+1,)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
    # 100 Hz start: FRAME=400 gives 40 Hz bins, so every geomspace band from
    # 100 Hz holds >=1 bin; the nearest-bin fallback is belt-and-braces.
    edges = np.geomspace(100.0, 8000.0, N_BANDS + 1)
    bands = np.empty(N_BANDS)
    for b in range(N_BANDS):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if not np.any(m):
            m = np.abs(freqs - edges[b]) < 1e-9 + np.min(np.abs(freqs - edges[b]))
        bands[b] = np.log10(float(np.mean(mean_spec[m])) + 1e-12)
    floor_db = 10.0 * np.log10(float(np.mean(energy[order[:n_quiet]])) + 1e-12)
    speech_db = 10.0 * np.log10(float(np.mean(np.mean(loud * loud, axis=1))) + 1e-12)
    peak = float(np.max(np.abs(x)))
    clip_frac = (float(np.mean(np.abs(x) >= 0.999 * peak))
                 if peak > 1e-9 else 0.0)
    feats = np.concatenate([bands,
                            [floor_db, speech_db - floor_db,
                             float(np.mean(x)), float(np.sqrt(np.mean(x * x))),
                             clip_frac]])
    return feats if np.all(np.isfinite(feats)) else None


def _clean_features(path: str):
    if path not in _FEAT_CACHE:
        _FEAT_CACHE[path] = _features(load_clip(path))
    return _FEAT_CACHE[path]


# ── the noise/reverb stratum ───────────────────────────────────────────────

def _make_rir(rng):
    import numpy as np
    n = int(RT60_S * SR)
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-6.9077553 * t / RT60_S)
    return h / np.sqrt(np.sum(h * h))


def _pink(rng, n):
    import numpy as np
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / (np.sqrt(np.mean(x * x)) + 1e-12)


def _noisy_features(path, rir, rng):
    """Reverb (one shared room) + pink noise at exactly NOISE_SNR_DB, then
    redelivered through the contract. Returns (features, transformed_enough)."""
    import numpy as np
    from scipy.signal import fftconvolve
    x = load_clip(path)
    y = fftconvolve(x, rir)[:len(x)]
    noise = _pink(rng, len(y))
    p_sig = float(np.mean(y * y))
    noise = noise * np.sqrt(p_sig / (10.0 ** (NOISE_SNR_DB / 10.0)))
    y = deliver(y + noise)
    rel = float(np.sqrt(np.mean((y - x) ** 2)) / (np.sqrt(np.mean(x * x)) + 1e-12))
    return _features(y), rel > 0.01


# ── corpus indexing and the per-seed split ─────────────────────────────────

def _index_corpus():
    """speaker -> chapter -> [paths], duration-filtered; plus a census."""
    import soundfile as sf
    idx, n_files, n_bytes = {}, 0, 0
    for spk in sorted(os.listdir(CORPUS_ROOT)):
        sdir = CORPUS_ROOT / spk
        if not sdir.is_dir():
            continue
        for chap in sorted(os.listdir(sdir)):
            cdir = sdir / chap
            if not cdir.is_dir():
                continue
            for f in sorted(os.listdir(cdir)):
                if not f.endswith(".flac"):
                    continue
                p = cdir / f
                n_files += 1
                n_bytes += p.stat().st_size
                if sf.info(str(p)).duration >= MIN_CLIP_S:
                    idx.setdefault(spk, {}).setdefault(chap, []).append(str(p))
    return idx, n_files, n_bytes


def _split(idx, seed):
    """Deterministic per-seed split. Returns None if construction fails."""
    import numpy as np
    rng = np.random.default_rng(seed)
    eligible = sorted(s for s, ch in idx.items() if len(ch) >= 2)
    others = sorted(s for s in idx if s not in eligible)
    eligible = list(np.array(eligible)[rng.permutation(len(eligible))])

    enrolled, per_spk = [], {}
    for spk in eligible:
        if len(enrolled) == N_ENROLLED_TARGET:
            break
        chaps = list(np.array(sorted(idx[spk]))[rng.permutation(len(idx[spk]))])
        enrol_ch, pool = [], []
        for c in chaps[:-1]:                       # >=1 chapter stays for test
            enrol_ch.append(c)
            pool = sum((idx[spk][c] for c in enrol_ch), [])
            if len(pool) >= N_ENROL_CLIPS + CTRL_SPK_MIN:
                break
        test_ch = [c for c in chaps if c not in enrol_ch]
        test_pool = sum((idx[spk][c] for c in test_ch), [])
        if len(pool) < PER_SPK_MIN or len(test_pool) < PER_SPK_MIN:
            continue
        pool = list(np.array(pool)[rng.permutation(len(pool))])
        enrol = pool[:N_ENROL_CLIPS]
        ctrl = pool[N_ENROL_CLIPS:N_ENROL_CLIPS + N_CTRL_CLIPS]
        test = list(np.array(test_pool)[rng.permutation(len(test_pool))])[:N_TEST_CLIPS]
        per_spk[spk] = {"enrol": enrol, "ctrl": ctrl, "test": test,
                        "enrol_chapters": enrol_ch, "test_chapters": test_ch}
        enrolled.append(spk)

    unknown = [s for s in idx if s not in enrolled][:2 * N_ENROLLED_TARGET]
    if len(enrolled) < N_ENROLLED_MIN or len(unknown) < N_UNKNOWN_MIN:
        return None

    # Disjointness + cross-chapter, verified rather than assumed.
    for spk, d in per_spk.items():
        if set(d["enrol"]) & set(d["test"]) or set(d["enrol"]) & set(d["ctrl"]):
            return None
        for p in d["test"]:
            if Path(p).parent.name in d["enrol_chapters"]:
                return None
    return {"enrolled": enrolled, "unknown": unknown, "per_spk": per_spk}


# ── the probe ──────────────────────────────────────────────────────────────

def _probe(train_X, train_y, tests, n_classes, seed):
    """Multinomial logistic regression, deterministic. tests: {name: (X, y)}.
    Returns {name: accuracy}."""
    import numpy as np
    import torch
    torch.manual_seed(seed)
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xt = torch.tensor((train_X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(train_y, dtype=torch.long)
    lin = torch.nn.Linear(Xt.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    for _ in range(PROBE_STEPS):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xt), yt)
        loss.backward()
        opt.step()
    out = {}
    with torch.no_grad():
        for name, (X, y) in tests.items():
            Xv = torch.tensor((X - mu) / sd, dtype=torch.float32)
            pred = lin(Xv).argmax(dim=1).numpy()
            out[name] = float(np.mean(pred == np.asarray(y)))
    return out


# ── the bundle: everything, once, for all seeds ────────────────────────────

def _bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    import numpy as np

    if not CORPUS_ROOT.is_dir():
        _BUNDLE = {s: {"corpus_present": 0.0} for s in SEEDS}
        return _BUNDLE

    idx, n_files, n_bytes = _index_corpus()
    manifest = {"contract": {"sr": SR, "clip_s": CLIP_S, "target_rms": TARGET_RMS,
                             "dc_removed": True, "loader": "hr_1_voice_corpus_honest.load_clip",
                             "whiten": {"reg": WHITEN_REG,
                                        "smooth_bins": WHITEN_SMOOTH_BINS,
                                        "quiet_frac": QUIET_FRAC,
                                        "frame": FRAME, "hop": HOP}},
                "census": {"n_files": n_files, "n_bytes": n_bytes},
                "seeds": {}}
    out = {}
    for seed in SEEDS:
        r = {"corpus_present": 1.0}
        split = _split(idx, seed)
        if split is None:
            r.update(construction_ok=0.0)
            out[seed] = r
            continue
        enrolled, per_spk = split["enrolled"], split["per_spk"]
        n_cls = len(enrolled)
        lab = {s: i for i, s in enumerate(enrolled)}
        rng = np.random.default_rng(10_000 + seed)
        rir = _make_rir(rng)

        def feats_of(paths, spk, noisy=False):
            X, y = [], []
            for p in paths:
                f = (_noisy_features(p, rir, rng) if noisy
                     else (_clean_features(p), True))
                if f[0] is None or not f[1]:
                    return None, None
                X.append(f[0])
                y.append(lab[spk])
            return X, y

        TX, Ty, CX, Cy, NX, Ny, LX, Ly = [], [], [], [], [], [], [], []
        ok = True
        n_ctrl_spk = 0
        for spk in enrolled:
            d = per_spk[spk]
            a, b = feats_of(d["enrol"], spk)
            c, e = feats_of(d["test"], spk)
            f, g = feats_of(d["test"], spk, noisy=True)
            if a is None or c is None or f is None:
                ok = False
                break
            TX += a; Ty += b; CX += c; Cy += e; NX += f; Ny += g
            if len(d["ctrl"]) >= CTRL_SPK_MIN:
                h, i = feats_of(d["ctrl"], spk)
                if h is None:
                    ok = False
                    break
                LX += h; Ly += i
                n_ctrl_spk += 1
        if not ok or n_ctrl_spk < N_ENROLLED_MIN:
            r.update(construction_ok=0.0)
            out[seed] = r
            continue

        accs = _probe(np.array(TX), np.array(Ty),
                      {"clean": (np.array(CX), Cy),
                       "noise": (np.array(NX), Ny),
                       "leaky": (np.array(LX), Ly)}, n_cls, seed)
        chance = 1.0 / n_cls
        bar = chance + LEAK_EXCESS
        r.update(construction_ok=1.0,
                 n_enrolled=float(n_cls),
                 n_unknown=float(len(split["unknown"])),
                 n_ctrl_speakers=float(n_ctrl_spk),
                 chance=chance, leak_bar=bar,
                 control_floor=chance + CONTROL_FLOOR_EXCESS,
                 clean_acc=accs["clean"], noise_acc=accs["noise"],
                 control_acc=accs["leaky"],
                 margin=min(bar - accs["clean"], bar - accs["noise"]))
        manifest["seeds"][str(seed)] = {
            "enrolled": enrolled, "unknown": split["unknown"],
            "per_speaker": per_spk, "accs": accs}
        out[seed] = r

    try:
        MANIFEST.write_text(json.dumps(manifest, indent=1))
    except OSError:
        pass
    _BUNDLE = out
    return out


# ── the runner contract ────────────────────────────────────────────────────

def _experiment(seed: int) -> dict:
    b = _bundle()
    r = b[seed]
    out = {"min_channel_leak_margin": r.get("margin", float("nan")),
           "clean_probe_acc": r.get("clean_acc", float("nan")),
           "noise_probe_acc": r.get("noise_acc", float("nan"))}
    for s in SEEDS:
        for k in ("corpus_present", "construction_ok", "n_enrolled",
                  "n_unknown", "n_ctrl_speakers", "chance", "leak_bar",
                  "clean_acc", "noise_acc", "margin"):
            out[f"{k}_s{s}"] = float(b[s].get(k, 0.0))
    return out


def _control(seed: int) -> dict:
    b = _bundle()
    out = {"leaky_probe_acc": b[seed].get("control_acc", float("nan"))}
    for s in SEEDS:
        out[f"control_acc_s{s}"] = float(b[s].get("control_acc", 0.0))
        out[f"control_floor_s{s}"] = float(b[s].get("control_floor", 0.0))
    return out


def _check(m: dict, c: dict):
    try:
        present = [m[f"corpus_present_s{s}"] for s in SEEDS]
        constructed = [m[f"construction_ok_s{s}"] for s in SEEDS]
        n_enr = [m[f"n_enrolled_s{s}"] for s in SEEDS]
        n_unk = [m[f"n_unknown_s{s}"] for s in SEEDS]
        margins = [m[f"margin_s{s}"] for s in SEEDS]
        ctrl = [c[f"control_acc_s{s}"] for s in SEEDS]
        floor = [c[f"control_floor_s{s}"] for s in SEEDS]
    except KeyError:
        return Status.VOID
    # Rig-aliveness: a corpus that is absent, a split that failed construction,
    # or a probe that cannot see the PLANTED leak buys no verdict either way.
    if (min(present) < 1.0 or min(constructed) < 1.0
            or min(n_enr) < N_ENROLLED_MIN or min(n_unk) < N_UNKNOWN_MIN):
        return Status.VOID
    if any(a < f for a, f in zip(ctrl, floor)):
        return Status.VOID          # instrument dead: planted leak unseen
    return min(margins) >= 0.0


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["HR.1"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
