"""PG.7 — the heard-not-seen fixture must leak nothing but the intended bit.

UB.9 is the smallest experiment that could establish "his senses work in
unison" (GOAL.md). It works only if the scene is an honest XOR: audio says
WHICH SIZE fell, a pre-event frame says WHICH SIZE IS WHERE, and only the two
together say WHICH SLOT fell. If audio alone can name the slot, UB.9 measures a
leak and calls it fusion. This spec is the certification, on the PG.5
precedent — the fixture is tested before anything trains on it.

WHAT IS PROBED, AND WHY THAT IS THE RIGHT TARGET.  The label of UB.9 is the
SLOT (which of the two standing candidates was released), not the size. Audio
is supposed to be at chance on it — not approximately, but by construction,
because ``hns_scene.draw_quad`` redraws the size->slot assignment every episode
and shares every nuisance parameter between the slots. Two probes, both on the
rendered stereo buffer alone and never on the synth's event labels:

  P1  all audio features -> SLOT     must stay at chance   (protects UB.9's null (i))
  P2  level and pan only -> SIZE     must stay at chance   (protects UB.9's
                                       spectrum-flattened control: if identity
                                       also rides on loudness, flattening the
                                       spectrum would not remove it)

and one probe that must SUCCEED, because a fixture that leaks nothing and also
carries nothing is not a fixture:

  P3  level-normalised band energies -> SIZE   must be near-perfect
                                       (the intended bit is really present)

THE POSITIVE CONTROLS.  A leak detector that has never caught a leak is not a
detector, it is a null result. Both deliberately-broken variants live in
``hns_scene.Leak`` and the SAME probes must catch them:

  Leak.GEOMETRY  slot 1 un-mirrored (azimuth -theta) and 0.6 m further out, so
                 pan and 1/distance track the slot -> P1 must exceed 0.70
  Leak.MASS      mass by volume, so impact force and amplitude track size
                 -> P2 must exceed 0.70   (this is the control named in the
                 spec; the geometry arm is added because a mass leak alone
                 cannot move P1, and P1 is the probe UB.9's null depends on)

FIXTURE ASSERTIONS, paired within a quad so the two episodes compared differ
only in the label (the leak table, UNIFIED_BRAIN_BAKEOFF.md 3.2):

    max_pan_gap        < 1e-6      pan identical between slots
    max_distance_gap   < 1e-3 m    equal listener range
    max_amp_rel_gap    <= 0.05     equal impact amplitude between sizes
    voiced_correct     == 1.0      the candidate rings, never floor or decoy
    single_event       == 1.0      exactly one onset in the decode window
    both_spheres       == 1.0      geom type 2, so char_size is derived alike

PRE-REGISTERED GATES (all must hold, and each on EVERY seed — the per-seed
indicators below are averaged by run_spec, so a mean of 1.0 means no seed
failed):

    slot_leak_acc            <= 0.53   (chance 0.5 + 3%)   <- the falsifier
    nonspectral_identity_acc <= 0.53
    spectral_identity_acc    >= 0.95
    all six fixture assertions above
    control GEOMETRY: slot_leak_acc            >= 0.70
    control MASS:     nonspectral_identity_acc >= 0.70

Probes are ridge logistic regressions (IRLS) under 4-fold cross-validation with
folds assigned BY QUAD, so the four episodes sharing a nuisance draw never
straddle a fold boundary. 800 quads = 3200 balanced episodes per seed puts the
binomial standard error at 0.88%, so the 0.53 gate sits 3.4 sigma above chance
and a clean fixture is not failed by luck.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# The implementation under test. Undeclared until 2026-09-06 (78th audit
# finding 1.1; grandfather set shrunk here).
IMPL_DEPS = ['ContactAudio.py']

REPO = Path(__file__).resolve().parents[2]

N_QUADS = 800            # x4 episodes; the balanced arm
N_QUADS_CTRL = 200       # the leak arms only need to show a LARGE effect
EPISODE_S = 0.90         # longest fall (0.60 m) lands at ~0.35 s
WIN_S = 0.10             # decode window from the audio-detected onset
N_BANDS = 12
BAND_LO, BAND_HI = 100.0, 7600.0
CHANCE = 0.5
LEAK_GATE = 0.53         # chance + 3%
DETECT_GATE = 0.70       # the positive controls must clear this
EPS = 1e-12


# ── the probe ────────────────────────────────────────────────────────────

def _logreg_cv(X, y, groups, folds: int = 4, l2: float = 1.0, iters: int = 25):
    """Ridge logistic regression, IRLS, grouped k-fold. Returns accuracy on the
    held-out folds pooled — every episode is predicted exactly once."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    correct = 0
    for f in range(folds):
        te = (groups % folds) == f
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        A = np.hstack([(X[tr] - mu) / sd, np.ones((tr.sum(), 1))])
        B = np.hstack([(X[te] - mu) / sd, np.ones((te.sum(), 1))])
        w = np.zeros(A.shape[1])
        reg = l2 * np.eye(A.shape[1])
        reg[-1, -1] = 0.0                      # never penalise the intercept
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
            s = np.clip(p * (1 - p), 1e-6, None)
            H = A.T @ (A * s[:, None]) + reg
            g = A.T @ (y[tr] - p) - reg @ w
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            w += step
            if np.max(np.abs(step)) < 1e-8:
                break
        pred = (B @ w) > 0.0
        correct += int((pred == (y[te] > 0.5)).sum())
    return correct / len(y)


def _features(stereo, sr):
    """Audio-only features. Uses NOTHING but the rendered buffer — the onset is
    found from the waveform, not from the synth's event time, so no label can
    reach the probe by the back door (PG.5's circularity guard, applied here to
    the time axis as well as to the truth)."""
    import numpy as np
    mid = stereo[0] + stereo[1]
    env = np.abs(mid)
    peak = float(env.max())
    if peak <= 0.0:
        return None
    i0 = int(np.argmax(env > 0.05 * peak))
    n = int(WIN_S * sr)
    if i0 + n > stereo.shape[1]:
        return None
    L = stereo[0, i0:i0 + n]
    R = stereo[1, i0:i0 + n]
    m = L + R
    spec = np.abs(np.fft.rfft(m * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges = np.geomspace(BAND_LO, BAND_HI, N_BANDS + 1)
    bands = np.array([
        math.log10(float(spec[(freqs >= edges[b]) & (freqs < edges[b + 1])].sum()) + EPS)
        for b in range(N_BANDS)])
    el, er = float((L ** 2).sum()), float((R ** 2).sum())
    pan = (er - el) / (el + er + EPS)
    level = math.log10(math.sqrt(float((m ** 2).mean())) + EPS)
    return bands, level, pan


# ── one arm ──────────────────────────────────────────────────────────────

def _arm(seed: int, leak, n_quads: int) -> dict:
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    from ContactAudio import ContactAudioSynth
    from experiments.hns_scene import build, draw_quad, listener_pose

    rng = np.random.RandomState(seed * 9173 + 41)
    feats, y_slot, y_large, quad_id = [], [], [], []
    voiced_ok = single_ok = spheres_ok = 0
    n_ep = 0
    pan_gap = dist_gap = amp_gap = 0.0
    for q in range(n_quads):
        quad = draw_quad(rng, leak)
        # (pan, distance) keyed by (faller_size_is_large, slot); (amp) by
        # (slot, size) — the paired fixture comparisons.
        obs = {}
        for ep in quad:
            model, data = build(ep)
            lpos, lyaw = listener_pose(ep)
            synth = ContactAudioSynth(model)
            synth.set_listener(lpos, lyaw)
            for _ in range(int(EPISODE_S / model.opt.timestep)):
                mujoco.mj_step(model, data)
                synth.step(data)
            n_ep += 1
            cand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     f"cand{ep.faller_slot}")
            other = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                      f"cand{1 - ep.faller_slot}")
            spheres_ok += int(model.geom_type[cand] == 2 and model.geom_type[other] == 2)
            if not synth.events:
                continue
            e0 = synth.events[0]
            voiced_ok += int(e0.voiced_geom == cand)
            in_win = [e for e in synth.events if e.t < e0.t + WIN_S]
            single_ok += int(len(in_win) == 1)

            audio = synth.render(duration=e0.t + WIN_S + 0.05)
            f = _features(audio, synth.sr)
            if f is None:
                continue
            bands, level, pan = f
            feats.append(np.concatenate([bands, [level, pan]]))
            y_slot.append(float(ep.faller_slot))
            y_large.append(float(ep.faller_radius > 0.18))
            quad_id.append(q)

            is_large = ep.faller_radius > 0.18
            obs[(is_large, ep.faller_slot)] = (-math.sin(e0.azimuth), e0.distance, e0.amp)

        for is_large in (False, True):                   # slot pairing
            a, b = obs.get((is_large, 0)), obs.get((is_large, 1))
            if a and b:
                pan_gap = max(pan_gap, abs(a[0] - b[0]))
                dist_gap = max(dist_gap, abs(a[1] - b[1]))
        for slot in (0, 1):                              # size pairing
            a, b = obs.get((False, slot)), obs.get((True, slot))
            if a and b:
                amp_gap = max(amp_gap, 2 * abs(a[2] - b[2]) / (a[2] + b[2] + EPS))

    X = np.asarray(feats)
    g = np.asarray(quad_id)
    slot_acc = _logreg_cv(X, y_slot, g)                            # P1
    nonspec_acc = _logreg_cv(X[:, -2:], y_large, g)                # P2
    spec_norm = X[:, :N_BANDS] - X[:, [N_BANDS]]                   # level-normalised
    spectral_acc = _logreg_cv(spec_norm, y_large, g)               # P3
    return {
        "episodes": n_ep, "usable": len(feats),
        "slot_leak_acc": round(slot_acc, 4),
        "nonspectral_identity_acc": round(nonspec_acc, 4),
        "spectral_identity_acc": round(spectral_acc, 4),
        "max_pan_gap": pan_gap, "max_distance_gap": dist_gap,
        "max_amp_rel_gap": round(amp_gap, 6),
        "voiced_correct": round(voiced_ok / max(n_ep, 1), 4),
        "single_event": round(single_ok / max(n_ep, 1), 4),
        "both_spheres": round(spheres_ok / max(n_ep, 1), 4),
    }


def _experiment(seed: int) -> dict:
    from experiments.hns_scene import Leak
    m = _arm(seed, Leak.NONE, N_QUADS)
    # Per-seed indicators: run_spec averages across seeds, so a mean of 1.0 is
    # the statement "no seed failed", which a mean of the raw numbers is not.
    m["seed_gates_ok"] = float(
        m["slot_leak_acc"] <= LEAK_GATE
        and m["nonspectral_identity_acc"] <= LEAK_GATE
        and m["spectral_identity_acc"] >= 0.95
        and m["max_pan_gap"] < 1e-6
        and m["max_distance_gap"] < 1e-3
        and m["max_amp_rel_gap"] <= 0.05
        and m["voiced_correct"] == 1.0
        and m["single_event"] == 1.0
        and m["both_spheres"] == 1.0)
    m["audio_only_leak_margin"] = round(m["slot_leak_acc"] - CHANCE, 4)
    return m


def _control(seed: int) -> dict:
    from experiments.hns_scene import Leak
    geo = _arm(seed, Leak.GEOMETRY, N_QUADS_CTRL)
    mass = _arm(seed, Leak.MASS, N_QUADS_CTRL)
    return {
        "geometry_leak_slot_acc": geo["slot_leak_acc"],
        "geometry_leak_pan_gap": geo["max_pan_gap"],
        "mass_leak_nonspectral_acc": mass["nonspectral_identity_acc"],
        "mass_leak_amp_rel_gap": mass["max_amp_rel_gap"],
        "control_seed_gates_ok": float(
            geo["slot_leak_acc"] >= DETECT_GATE
            and mass["nonspectral_identity_acc"] >= DETECT_GATE),
    }


def _check(m: dict, c: dict) -> bool:
    return (m["seed_gates_ok"] == 1.0
            and m["slot_leak_acc"] <= LEAK_GATE
            and m["nonspectral_identity_acc"] <= LEAK_GATE
            and m["spectral_identity_acc"] >= 0.95
            and c["control_seed_gates_ok"] == 1.0
            and c["geometry_leak_slot_acc"] >= DETECT_GATE
            and c["mass_leak_nonspectral_acc"] >= DETECT_GATE)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.7"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
