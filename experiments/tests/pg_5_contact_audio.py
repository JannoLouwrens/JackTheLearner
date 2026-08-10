"""PG.5 — contact audio must carry honest localization before hearing trains on it.

UB.4 will claim "hearing is load-bearing" by muting audio and watching a task
degrade. That claim is empty if the audio channel never encoded the world to
begin with. This spec certifies the fixture: modal-resonator synthesis on
MuJoCo contact events yields stereo whose PANNING matches the source bearing.

Circularity guard — the trap this test is designed around: the synth computes
an azimuth label and derives the pan from it, so comparing decoded bearing to
the synth's OWN label would pass even with the listener-frame math completely
wrong (label and pan share the bug). Ground truth here is therefore computed
INDEPENDENTLY from the sampled drop point and listener pose, in this file's own
trig. The synth's label is then also checked against that same truth, which is
what certifies the labels UB.4 will train on.

Protocol per seed: 9 free objects (apple + obj0..7), each dropped in its own
episode from 1.2 m at a sampled bearing/radius around a listener with random
yaw; the other 8 are parked by the far wall. Decode the first-impact window's
lateral angle from L/R channel energy. Gates (pre-registered):
  bearing_decode_accuracy >= 0.9   (|decoded - truth| <= 10 deg; a drop whose
                                    impact is never detected counts as a MISS)
  label_accuracy          >= 0.9   (synth azimuth label vs independent truth, 5 deg)
  spectral_match          >= 0.8   (window's dominant frequency == voiced geom's
                                    fundamental within 12% — the ring is modal,
                                    per-geom, not generic noise)
  audio finite, peak in [0.02, 1.0]
Controls that MUST fail: mono render and shuffled-pan render of the same
events decode at <= 0.30. Front-back is out of scope: pan encodes laterality
only, so truth and decode both live in [-90, 90] deg (folded azimuth).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID

# This spec certifies a property of the WORLD, so the world hashes into
# impl_sha. Change playground.py and this certificate goes stale loudly
# instead of standing over a world it no longer describes.
IMPL_DEPS = ["playground.py"]

REPO = Path(__file__).resolve().parents[2]

N_OBJECTS = 8                # + apple = 9 drops per seed
DROP_Z = 1.2
EPISODE_S = 1.2
LISTENER = (0.0, 0.0, 1.4)
TOL_DECODE = math.radians(10.0)
TOL_LABEL = math.radians(5.0)
DECODE_WIN_S = 0.08


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
    """Bearing/radius around the listener, avoiding the pool basin (soft
    underwater contact) and the ladder/platform column (overhead geometry).
    Landing ON the ramp/stairs/seesaw is fine — a contact is a contact."""
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


def _run_arm(seed: int, arm: str) -> dict:
    """arm: 'stereo' (experiment) | 'mono' | 'shuffle' (controls).
    Same seed => identical physics/events across arms; only the render differs."""
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    from playground import PlaygroundParams, make_playground
    from ContactAudio import ContactAudioSynth, decode_lateral

    p = PlaygroundParams(seed=seed, n_objects=N_OBJECTS)
    model, data, water = make_playground(p)
    rng = np.random.RandomState(seed * 101 + 3)
    yaw = float(rng.uniform(-math.pi, math.pi))
    bodies = _free_bodies(model)
    steps = int(EPISODE_S / model.opt.timestep)
    sr = ContactAudioSynth(model).sr

    hits = label_hits = spec_hits = decoded = 0
    peak = 0.0
    finite = True
    rms_by_dist = []
    for k, (tname, tbid) in enumerate(bodies):
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

        tgeoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == tbid}
        evs = [e for e in synth.events
               if (e.geom1 in tgeoms or e.geom2 in tgeoms) and e.t > 0.05]
        if not evs:
            continue                      # undetected impact = a miss on all gates
        e0 = evs[0]
        decoded += 1

        # Independent truth: from the SAMPLED drop point, this file's own trig.
        lat_true = math.asin(math.sin(az_true))

        # The shuffle arm decodes 3 independent random pans and scores the hit
        # fraction: one draw per episode leaves the null's pass/fail to luck
        # (chance ~13% at 10-degree tolerance, so 9 draws have a real shot at
        # crossing 0.30 by coincidence; 27 do not).
        n_draws = 3 if arm == "shuffle" else 1
        ep_hits = 0.0
        win = None
        for _ in range(n_draws):
            pan_override = None
            if arm == "shuffle":
                idx = synth.events.index(e0)
                pan_override = {idx: float(rng.uniform(-1.0, 1.0))}
            audio = synth.render(duration=e0.t + DECODE_WIN_S + 0.02,
                                 mode="mono" if arm == "mono" else "stereo",
                                 pan_override=pan_override)
            finite = finite and bool(np.all(np.isfinite(audio)))
            peak = max(peak, float(np.abs(audio).max()))
            w0 = int(e0.t * sr)
            win = audio[:, w0:w0 + int(DECODE_WIN_S * sr)]
            lat_hat = decode_lateral(win)
            if abs(lat_hat - lat_true) <= TOL_DECODE:
                ep_hits += 1.0 / n_draws
        hits += ep_hits
        if abs(e0.lateral - lat_true) <= TOL_LABEL:
            label_hits += 1

        mono = win.sum(axis=0)
        mags = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
        mags[freqs < 60.0] = 0.0
        f_peak = float(freqs[int(np.argmax(mags))])
        f0 = synth.fundamental(e0.voiced_geom)
        if abs(f_peak - f0) / f0 <= 0.12:
            spec_hits += 1
        rms_by_dist.append((e0.distance, float(np.sqrt(np.mean(win ** 2)))))

    n = len(bodies)
    corr = 0.0
    if len(rms_by_dist) >= 3:
        import numpy as np
        d = np.array([x[0] for x in rms_by_dist])
        v = np.array([x[1] for x in rms_by_dist])
        corr = float(np.corrcoef(1.0 / d, v)[0, 1])
    return {
        "drops": n, "events_decoded": decoded,
        "bearing_decode_accuracy": round(hits / n, 4),
        "label_accuracy": round(label_hits / n, 4),
        "spectral_match": round(spec_hits / n, 4),
        "audio_finite": finite, "peak_amp": round(peak, 4),
        "rms_vs_inv_distance_corr": round(corr, 3),   # info-only: impulse varies
    }


def _experiment(seed: int) -> dict:
    return _run_arm(seed, "stereo")


def _control(seed: int) -> dict:
    mono = _run_arm(seed, "mono")
    shuf = _run_arm(seed, "shuffle")
    return {"mono_accuracy": mono["bearing_decode_accuracy"],
            "shuffled_accuracy": shuf["bearing_decode_accuracy"],
            "control_events_decoded": mono["events_decoded"]}


def _check(m: dict, c: dict) -> bool:
    return (m["bearing_decode_accuracy"] >= 0.9
            and m["label_accuracy"] >= 0.9
            and m["spectral_match"] >= 0.8
            and m["audio_finite"] >= 1.0
            and 0.02 <= m["peak_amp"] <= 1.0
            and c["mono_accuracy"] <= 0.30
            and c["shuffled_accuracy"] <= 0.30)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.5"], _experiment, _check, control_fn=_control, ledger=ledger)
