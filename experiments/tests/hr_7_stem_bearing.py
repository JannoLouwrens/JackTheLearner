"""HR.7 — the audio stem does not deafen him to direction.

Directional hearing is the ONLY thing PG.5 certifies about this world's audio,
and it is what makes hearing useful for ACTION (turn toward the sound). A stem
whose wiring discards bearing reduces audio to an event detector — and the
canonical way that happens is one line: a first op that averages the CHANNEL
dimension IS the mono control, silently. This spec is the guard that fires
BEFORE a stem is ever trained (HR.6 depends on it).

WHAT IS PROBED: the A2 incumbent from `experiments/hearing.py` — 2-channel
log-mel (64 bins, 25/10 ms) -> Conv2d stack -> 4 tokens — at deterministic
RANDOM INIT. The guard is architectural: if bearing cannot be read back out
of the untrained stem's tokens, no amount of training on a task loss is
entitled to assume it survived. HR.6's other arms owe this same fixture
before entering the bakeoff; the discrete-token arm (A4) is pre-registered in
the registry to FAIL it, but A4 needs a codec download and the speech-arm
disk escalation (D19, DECISIONS_NEEDED.md) is NO-FETCH by default — so A4's
HR.7 run waits on D19 and is NOT silently waved through by this PASS.

PRE-STATED EXPECTED VERDICT (the T2.05 discipline): PASS. HEARING_BAKEOFF.md
§1.4.3 measured that 2-channel log-mel preserves bearing to the full PG.5
gate (energy-pooled analytic decode 0.99-1.00, mono 0.10), and the stem's
random conv mixes but does not destroy the two input planes. If this FAILs,
the finding is real: the A2 wiring itself eats bearing, and HR.6 must not
train it.

THE TWO MEASURED PROBE TRAPS, designed around per the registry hypothesis:
the log-domain ILD is exactly atanh(pan), so (1) the probe predicts the
ATANH-DOMAIN target and inverts through tanh — a linear-in-angle readout
scored 0.40 on the correct representation and would kill the winning arm;
(2) instrument aliveness uses ENERGY-domain pooling (`hearing.analytic_
lateral`), never a mean over per-bin log ILD (0.69 vs 1.00, the log-floor
drag).

PROTOCOL per seed: N_DROPS PG.5-style drop episodes (free bodies cycled
round-robin, sampled bearing/radius around a listener with random yaw, same
clear-spot rules as PG.5). Per episode the first target event's 80 ms window
is rendered STEREO and MONO from the same physics; ground truth is this
file's own trig from the sampled drop point (PG.5's circularity guard). The
stem maps each window's log-mel to 4x64 tokens; an RBF kernel-ridge probe on
the flat tokens predicts atanh(pan), trained with the pan law's mirror
symmetry ((tokens, +y) AND (swapped-tokens, -y) — see `_probe_crossfit` for
why, with the measured OOD offset that forced it), and is scored 8-fold
cross-fit, so every event's prediction is out-of-fold. An undetected drop
counts as a MISS (PG.5's own accounting).

N_DROPS = 384 AND THE KERNEL PROBE ARE SIZED, NOT ARBITRARY (pilot on this
box, 2026-09-03, gates untouched throughout): the probe — not the stem — was
the bottleneck at small n and with a linear readout, which is precisely the
false-negative mode the registry hypothesis warns kills winning arms.
Seed-0 stem accuracy vs episode count, linear ridge: 48 -> 0.77, 96 -> 0.81,
192 -> 0.93 — while the ENERGY-POOLED ANALYTIC decode read 1.00 on the
identical windows at every n (the information was always fully present; only
the estimator starved). With mirrored training the linear readout still
plateaued at 0.87-0.93 across seeds at n=288; the RBF kernel probe (the
non-linear probe the hypothesis mandates) read 0.91/0.93 there and 0.956 on
the worst seed at n=384. Undersizing this fixture and calling the result
FAIL would blame the stem for the probe's variance. Gates, all pre-registered
in the registry:

  stem_bearing_probe_accuracy >= 0.9   (|decoded - truth| <= 10 deg, worst seed)
  mono_probe_accuracy         <= 0.30  (null: same stem+probe on PG.5's own
                                        mono render — bearing undecodable)
  swap_sign_inversion_frac    >= 0.9   (control: L/R-swapped input INVERTS the
                                        out-of-fold prediction's sign on
                                        clearly-lateral events (|truth| >=
                                        10 deg); degradation is what a broken
                                        probe produces, only inversion shows
                                        the stem read the interaural
                                        difference)

INSTRUMENT-ALIVENESS (VOID, never FAIL, when the rig cannot testify — the
at-chance-control lesson, 24th audit): audio finite; >= 0.9 of drops produce
a detected event; PG.5's certified raw decode (`decode_lateral`) >= 0.9 on
detected windows; the energy-pooled mel analytic decode >= 0.9 (if raw
passes and mel fails, MY mel is broken, not the stem); >= 10 clearly-lateral
events per seed for the swap count to mean anything. A dead microphone must
not buy a verdict in either direction.

Gate purity (the LG.02 standard): every gated metric is recorded per seed as
an explicit `<key>_s<seed>` key — each run returns the full per-seed set,
identical across calls, so the aggregation carries the values verbatim into
the row — and `_check` is a pure function of the row: 25 statically-named
keys read up front, a missing key -> VOID.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# The stem contract lives in experiments/hearing.py: an edit to the stem or
# the mel front end must go loudly stale on this certificate (PG.5 precedent —
# ContactAudio.py's absence from an IMPL_DEPS list once left a certificate
# green over code it had never run against; docs/LESSONS.md).
IMPL_DEPS = ["ContactAudio.py", "playground.py", "experiments/hearing.py"]

REPO = Path(__file__).resolve().parents[2]

SEEDS = (0, 1, 2)            # registry: seeds=3
N_DROPS = 384                # episodes per seed; sized by pilot, see docstring
N_FOLDS = 8
DROP_Z = 1.2
EPISODE_S = 1.2
LISTENER = (0.0, 0.0, 1.4)
WIN_S = 0.08                 # PG.5's decode window
TOL = math.radians(10.0)     # the PG.5 gate this spec inherits
ACC_GATE = 0.90
MONO_GATE = 0.30
SWAP_GATE = 0.90
SWAP_MIN_LAT = math.radians(10.0)   # only clearly-lateral events count for
                                    # sign inversion; at truth ~0 the sign is
                                    # noise, not evidence
SWAP_MIN_N = 10
DETECT_FLOOR = 0.90
ALIVE_GATE = 0.90
KRR_LAMBDA = 1e-2            # kernel ridge regularizer
KRR_GAMMA_SCALE = 1.0        # RBF gamma = this / median pairwise sq-distance
ATANH_CLIP = 0.999


def _free_bodies(model):
    import mujoco
    out = []
    for bid in range(model.nbody):
        jadr = model.body_jntadr[bid]
        if jadr >= 0 and model.jnt_type[jadr] == 0:   # mjJNT_FREE
            out.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid), bid))
    return out


def _place(model, data, bid, pos):
    jadr = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jadr]
    vadr = model.jnt_dofadr[jadr]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[vadr:vadr + 6] = 0.0


def _sample_drop(rng, yaw, pool_half):
    """PG.5's rules: avoid the pool basin and the ladder/platform column."""
    for _ in range(200):
        az = float(rng.uniform(-math.pi, math.pi))
        r = float(rng.uniform(1.5, 3.2))
        wx = LISTENER[0] + r * math.cos(yaw + az)
        wy = LISTENER[1] + r * math.sin(yaw + az)
        in_pool = abs(wx - 2.6) < pool_half + 0.5 and abs(wy + 2.4) < pool_half + 0.5
        in_ladder = -0.9 < wx < 0.9 and -3.3 < wy < -1.6
        if not in_pool and not in_ladder:
            return az, r, wx, wy
    raise RuntimeError("drop sampling failed to find a clear spot")


def _probe_crossfit(feats, lat_true, detected, rng_seed):
    """8-fold out-of-fold RBF kernel ridge predicting atanh(pan) from flat
    tokens — the NON-LINEAR probe the registry hypothesis mandates, with the
    analytic link's target.

    feats: list of (2, F) arrays or None per episode — variant 0 is the
    as-rendered window's tokens, variant 1 the CHANNEL-SWAPPED window's.
    Training uses BOTH with mirrored targets: (v0, +y) and (v1, -y). That is
    the pan law's exact mirror symmetry, handed to the probe the same way the
    analytic tanh link gets it — not a hint about any particular event. It is
    also what makes the swap control a sharp instrument instead of an
    extrapolation test: measured on seed 1 before this existed, a probe
    trained on unswapped tokens only answered swapped inputs with
    y_swap ~ 0.86*(-y_norm) - 1.07 (corr 0.58) — differentially inverted but
    with a constant out-of-distribution offset that broke the sign on a
    quarter of events. With mirrored training the OOD region does not exist.
    And the failure modes the control hunts are UNTOUCHED: a channel-AVERAGING
    stem has swap-invariant tokens, so (f, +y) and (f, -y) at the SAME point
    force its predictions to ~0 — inversion impossible, accuracy at the ~11%
    base rate; a stem whose swap response is incoherent cannot satisfy both
    halves and scatters. Only a stem that encodes the interaural difference
    coherently can invert.

    Every variant gets an out-of-fold prediction from its fold's model.
    Returns per-episode predictions, shape (N, 2), NaN where undetected."""
    import numpy as np

    n = len(lat_true)
    preds = np.full((n, 2), np.nan)
    y = np.array([math.atanh(max(-ATANH_CLIP, min(ATANH_CLIP,
                                                  -math.sin(l))))
                  for l in lat_true])
    order = np.random.RandomState(rng_seed).permutation(n)
    folds = [order[k::N_FOLDS] for k in range(N_FOLDS)]
    for fold in folds:
        te = set(int(i) for i in fold)
        tr = [i for i in range(n) if i not in te and detected[i]]
        if not tr:
            continue
        Xtr = np.array([feats[i][0] for i in tr]
                       + [feats[i][1] for i in tr])
        ytr = np.concatenate([y[tr], -y[tr]])
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
        Xs = (Xtr - mu) / sd
        d2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
        pos = d2[d2 > 0]
        gamma = KRR_GAMMA_SCALE / (float(np.median(pos)) if pos.size else 1.0)
        K = np.exp(-gamma * d2)
        alpha = np.linalg.solve(K + KRR_LAMBDA * np.eye(len(K)), ytr)
        for i in te:
            if detected[i]:
                Xq = (feats[i] - mu) / sd
                dq = ((Xq[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
                preds[i] = np.exp(-gamma * dq) @ alpha
    return preds


def _hit_frac(preds_col, lat_true, n_total):
    """|inverted prediction - truth| <= TOL; misses (NaN) count against."""
    import numpy as np

    hits = 0
    for i, l in enumerate(lat_true):
        p = preds_col[i]
        if not np.isfinite(p):
            continue
        lat_hat = -math.asin(max(-1.0, min(1.0, math.tanh(p))))
        if abs(lat_hat - l) <= TOL:
            hits += 1
    return hits / n_total


def _seed_run(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    import torch
    from playground import PlaygroundParams, make_playground
    from ContactAudio import ContactAudioSynth, decode_lateral
    from ..hearing import (SR, analytic_lateral, make_stem, stereo_logmel,
                           stereo_melpow)

    p = PlaygroundParams(seed=seed, n_objects=8)
    model, data, water = make_playground(p)
    rng = np.random.RandomState(seed * 313 + 17)
    yaw = float(rng.uniform(-math.pi, math.pi))
    bodies = _free_bodies(model)
    steps = int(EPISODE_S / model.opt.timestep)
    synth0 = ContactAudioSynth(model)
    assert synth0.sr == SR, "hearing.SR out of step with ContactAudio"

    lat_true, wins_st, wins_mo = [], [], []
    detected = []
    finite = True
    raw_hits = mel_hits = 0
    for k in range(N_DROPS):
        tname, tbid = bodies[k % len(bodies)]
        mujoco.mj_resetData(model, data)
        for j, (_, bid) in enumerate(bodies):
            _place(model, data, bid, (-4.5 + 0.5 * j, -5.2, 0.25))
        az_true, r_true, wx, wy = _sample_drop(rng, yaw, p.pool_size)
        _place(model, data, tbid, (wx, wy, DROP_Z))
        mujoco.mj_forward(model, data)

        synth = ContactAudioSynth(model)
        synth.set_listener(LISTENER, yaw)
        for _ in range(steps):
            data.xfrc_applied[:] = 0
            if water:
                water.apply(model, data)
            mujoco.mj_step(model, data)
            synth.step(data)

        lt = math.asin(math.sin(az_true))     # independent truth, own trig
        lat_true.append(lt)
        tgeoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == tbid}
        evs = [e for e in synth.events
               if (e.geom1 in tgeoms or e.geom2 in tgeoms) and e.t > 0.05]
        if not evs:
            detected.append(False)
            wins_st.append(None)
            wins_mo.append(None)
            continue
        e0 = evs[0]
        detected.append(True)
        w0 = int(e0.t * SR)
        wlen = int(WIN_S * SR)
        st = synth.render(duration=e0.t + WIN_S + 0.02)[:, w0:w0 + wlen]
        mo = synth.render(duration=e0.t + WIN_S + 0.02,
                          mode="mono")[:, w0:w0 + wlen]
        finite = finite and bool(np.all(np.isfinite(st))
                                 and np.all(np.isfinite(mo)))
        wins_st.append(st)
        wins_mo.append(mo)
        # Instrument aliveness on THIS window: the PG.5-certified raw decode,
        # then the energy-pooled mel analytic decode (traps doc, §1.4.3).
        if abs(decode_lateral(st) - lt) <= TOL:
            raw_hits += 1
        if abs(analytic_lateral(stereo_melpow(st)) - lt) <= TOL:
            mel_hits += 1

    n_det = sum(detected)
    stem = make_stem(seed)

    def tokens_of(win_variants):
        x = np.stack([stereo_logmel(w) for w in win_variants])
        with torch.no_grad():
            t = stem(torch.from_numpy(x).float())
        return t.reshape(t.shape[0], -1).numpy()

    # Variants per episode: 0 = stereo (trained on), 1 = channel-swapped.
    feats_st = [tokens_of([w, w[::-1].copy()]) if w is not None else None
                for w in wins_st]
    feats_mo = [tokens_of([w, w[::-1].copy()]) if w is not None else None
                for w in wins_mo]   # mono's swap IS mono: channels identical

    preds = _probe_crossfit(feats_st, lat_true, detected, seed * 41 + 5)
    preds_mo = _probe_crossfit(feats_mo, lat_true, detected, seed * 41 + 5)

    acc = _hit_frac(preds[:, 0], lat_true, N_DROPS)
    mono_acc = _hit_frac(preds_mo[:, 0], lat_true, N_DROPS)

    # Swap control: on clearly-lateral detected events, the out-of-fold
    # prediction's SIGN must invert under L/R exchange.
    inv = tot = 0
    for i, l in enumerate(lat_true):
        if not detected[i] or abs(l) < SWAP_MIN_LAT:
            continue
        if not (np.isfinite(preds[i, 0]) and np.isfinite(preds[i, 1])):
            continue
        tot += 1
        if preds[i, 0] * preds[i, 1] < 0:
            inv += 1
    swap_frac = (inv / tot) if tot else 0.0

    stem_params = sum(int(q.numel()) for q in stem.parameters())
    return {
        "stem_acc": round(acc, 4),
        "mono_acc": round(mono_acc, 4),
        "swap_inv": round(swap_frac, 4),
        "swap_n": float(tot),
        "events_frac": round(n_det / N_DROPS, 4),
        "raw_alive": round(raw_hits / max(n_det, 1), 4),
        "mel_alive": round(mel_hits / max(n_det, 1), 4),
        "finite": float(finite),
        "stem_params": float(stem_params),
    }


_ALL: dict[int, dict] = {}


def _bundle() -> dict[int, dict]:
    for s in SEEDS:
        if s not in _ALL:
            _ALL[s] = _seed_run(s)
    return _ALL


def _experiment(seed: int) -> dict:
    b = _bundle()
    out = {
        # This seed's values, unsuffixed: the aggregation reports mean+std.
        "stem_bearing_probe_accuracy": b[seed]["stem_acc"],
        "events_decoded_frac": b[seed]["events_frac"],
        "stem_params": b[seed]["stem_params"],
    }
    # The full per-seed set, identical across calls, so the row carries every
    # gated value verbatim (the LG.02 purity standard).
    for s in SEEDS:
        out[f"stem_acc_s{s}"] = b[s]["stem_acc"]
        out[f"events_frac_s{s}"] = b[s]["events_frac"]
        out[f"raw_alive_s{s}"] = b[s]["raw_alive"]
        out[f"mel_alive_s{s}"] = b[s]["mel_alive"]
        out[f"finite_s{s}"] = b[s]["finite"]
    return out


def _control(seed: int) -> dict:
    b = _bundle()
    out = {
        "mono_probe_accuracy": b[seed]["mono_acc"],
        "swap_sign_inversion_frac": b[seed]["swap_inv"],
    }
    for s in SEEDS:
        out[f"mono_acc_s{s}"] = b[s]["mono_acc"]
        out[f"swap_inv_s{s}"] = b[s]["swap_inv"]
        out[f"swap_n_s{s}"] = b[s]["swap_n"]
    return out


def _check(m: dict, c: dict):
    try:
        stem_acc = [m[f"stem_acc_s{s}"] for s in SEEDS]
        events = [m[f"events_frac_s{s}"] for s in SEEDS]
        raw_alive = [m[f"raw_alive_s{s}"] for s in SEEDS]
        mel_alive = [m[f"mel_alive_s{s}"] for s in SEEDS]
        finite = [m[f"finite_s{s}"] for s in SEEDS]
        mono = [c[f"mono_acc_s{s}"] for s in SEEDS]
        swap = [c[f"swap_inv_s{s}"] for s in SEEDS]
        swap_n = [c[f"swap_n_s{s}"] for s in SEEDS]
    except KeyError:
        return Status.VOID
    # Instrument-dead => VOID, never FAIL: a dead microphone, a broken mel,
    # or too few lateral events must not buy a verdict either way.
    if (min(finite) < 1.0 or min(events) < DETECT_FLOOR
            or min(raw_alive) < ALIVE_GATE or min(mel_alive) < ALIVE_GATE
            or min(swap_n) < SWAP_MIN_N):
        return Status.VOID
    return (min(stem_acc) >= ACC_GATE
            and max(mono) <= MONO_GATE
            and min(swap) >= SWAP_GATE)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["HR.7"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
