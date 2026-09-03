"""HR.5 — the playground makes the sounds GOAL.md names. (Expected: FAIL.)

GOAL.md's sentence is a specification: "he must hear the ladder creak, the
splash, the thud of his own fall." This spec measures whether the world can
make those sounds AT ALL — before any encoder bakeoff (HR.6) or content claim
(HR.8) is allowed to mean anything. With only impact onsets, Jack's entire
auditory world is (onset, f0, level, pan): four numbers. A representation
bakeoff run on that measures how well each encoder recovers four numbers.

PRE-STATED EXPECTED VERDICT (the T2.05 discipline — written before the first
run): FAIL. HEARING_BAKEOFF.md measured on 2026-08-09, and ContactAudio.py as
of this registration confirms by reading: the synth fires on contact-pair
ONSETS only (step(), _prev_pairs), so a persisting loaded contact (creak) and
a sliding contact (rolling) are one ring each at first touch; Water is a
FORCE FIELD (playground.Water.apply writes xfrc_applied, generates no MuJoCo
contact), so water entry is silent — a floating object makes NO sound
entering the pool; AudioEvent carries no event-kind label and no self flag;
and the humanoid is not in the scene, so "the thud of his own fall" cannot
occur. The value of recording that as a ledger FAIL: the hearing commitment's
world half moves from unmeasured to measured, and the row names the exact
machinery a repair must build (see WHAT A PASS REQUIRES below).

WHAT IS RUN (per seed):
  Four scenario classes, N_EP episodes each, all using the SAME free body
  (the apple) so class separability cannot ride on object identity:
    impact  drop from 1.2 m onto a sampled clear spot of dry floor
    water   drop from 1.2 m into the pool (centre 2.6,-2.4 + jitter); the
            apple's density ratio ~0.17 means it FLOATS — entry then rest,
            no basin contact, which is precisely the silent case
    creak   the apple laid on the free end of the seesaw plank: a persisting
            contact under load while the plank rotates — the canonical
            creak-under-load situation this world can already stage
    roll    the apple placed on the floor with 2.5 m/s of horizontal
            velocity: a persisting sliding/rolling contact
  Per episode the FIRST target-involving event (t > 0.05 s) is windowed
  (0.10 s) out of the rendered stereo, reduced to 16 log-spaced band energies
  normalised to unit sum (shape, not level — the PG.7 normaliser lesson: a
  level cue here would be distance, i.e. geography). A linear one-vs-rest
  probe, fit on even episodes and scored on odd ones, gives
  four_class_audio_separability; an episode with no event scores as a miss
  (silence is not a classifiable sound).

INSTRUMENT-ALIVENESS (the at-chance-control lesson, 24th audit: an at-chance
reading proves nothing unless the instrument is proven alive in the same
run): the same feature + probe pipeline must separate two objects with
fundamentals >= 25% apart, dropped at the SAME spot, at >= 0.9 (2-class,
chance 0.5). If it cannot — or no impact event fires at all, or audio goes
non-finite — the run returns Status.VOID, never FAIL: a dead microphone must
not buy a red verdict.

THE REGISTRY CONTROL: position-only probe (spawn x,y -> class, same split)
must be AT CHANCE for a PASS (<= 0.45 = chance + 0.20). Today it succeeds —
the pool and the seesaw are at fixed world positions, so class IS geography —
and that success is itself part of the finding: until the world decouples
place from sound-kind, any audio-classification number is a map reading.

WHAT A PASS REQUIRES (forward contract, so the repair and the test agree on
names BEFORE the repair is written):
  - AudioEvent grows a `kind` attribute: "impact" | water-entry events in
    {"water", "splash", "water_entry"} | "creak" | rolling/sliding in
    {"roll", "scrape", "slide", "rolling", "sliding"}; kind_label_acc >= 0.9
    on episode-first events.
  - AudioEvent grows an `is_self` flag (geom_bodyid in Jack's body set).
  - Sustained voices: creak and roll episodes must emit events AFTER the
    placement onset (t > first + 0.25 s), driven by the persisting contact.
  - A surface-crossing detector (Water.apply knows the surface and every
    body's position/velocity — the labels are free and exact) emits a water
    event on entry.
  - The four classes separate on band energies at >= 0.45 while the
    position-only probe stays at chance (which additionally requires the
    world to stop making sound-kind a pure function of place).

VERDICT ANNOTATION (65th audit B4, 2026-09-03 — attempt 1, FAIL, ran 05:25):
the registered metric four_class_audio_separability read 0.583 and that
number is NOT INTERPRETABLE — do not quote it. The position-only control
read 0.708 in the same run: the control OUTSCORED the instrument (the T2.11
control-outscored rule), so the separability figure is a map reading, not an
audio measurement. The FAIL is carried entirely by the structural conjuncts:
classes_present 1.0 of 4, has_kind_label False, has_self_flag False — three
of the four sounds GOAL.md names do not exist in the fixture. The repair is
routed as `hr5-fixture-refuted` in docs/REVIEW_QUEUE.md (same W1 fork as
w0-too-shallow); do NOT re-run this spec to get a cleaner number.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

# World-property spec: the world hashes into impl_sha (the PG.5 precedent —
# ContactAudio.py's absence from PG.5's list once left a certificate green
# over code it had never run against; docs/LESSONS.md).
IMPL_DEPS = ["playground.py", "ContactAudio.py"]

REPO = Path(__file__).resolve().parents[2]

N_EP = 8                    # episodes per class
N_ALIVE = 6                 # drops per object in the aliveness pair
EPISODE_S = 1.6
LISTENER = (0.0, 0.0, 1.4)
WIN_S = 0.10
N_BANDS = 16
BAND_LO, BAND_HI = 60.0, 7200.0
POOL_X, POOL_Y = 2.6, -2.4          # playground.make_playground's Water pos
SEESAW_X, SEESAW_Y = -2.5, -0.5     # plank body position in build_mjcf
CHANCE4 = 0.25
SEP_GATE = CHANCE4 + 0.20           # registry: chance + 20%
POS_GATE = CHANCE4 + 0.20           # control "at chance": may not exceed this
ALIVE_GATE = 0.90
SUSTAIN_AFTER_S = 0.25              # an event this long after the onset is
                                    # sustained-voice evidence, not placement
WATER_KINDS = {"water", "splash", "water_entry"}
ROLL_KINDS = {"roll", "scrape", "slide", "rolling", "sliding"}
CLASSES = ("impact", "water", "creak", "roll")


def _free_bodies(model):
    import mujoco
    out = []
    for bid in range(model.nbody):
        jadr = model.body_jntadr[bid]
        if jadr >= 0 and model.jnt_type[jadr] == 0:   # mjJNT_FREE
            out.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid), bid))
    return out


def _place(model, data, bid, pos, vel=(0.0, 0.0, 0.0)):
    jadr = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jadr]
    vadr = model.jnt_dofadr[jadr]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[vadr:vadr + 6] = 0.0
    data.qvel[vadr:vadr + 3] = vel


def _clear_spot(rng, pool_half):
    """A dry, unobstructed floor point: outside the pool basin, the
    ladder/platform column, the seesaw's sweep and the welded block."""
    for _ in range(300):
        az = float(rng.uniform(-math.pi, math.pi))
        r = float(rng.uniform(1.2, 3.0))
        wx = LISTENER[0] + r * math.cos(az)
        wy = LISTENER[1] + r * math.sin(az)
        if abs(wx - POOL_X) < pool_half + 0.6 and abs(wy - POOL_Y) < pool_half + 0.6:
            continue
        if -0.9 < wx < 0.9 and -3.3 < wy < -1.6:      # ladder/platform
            continue
        if abs(wx - SEESAW_X) < 1.4 and abs(wy - SEESAW_Y) < 0.7:
            continue
        if abs(wx + 1.5) < 0.5 and abs(wy + 1.5) < 0.5:  # welded block
            continue
        return wx, wy
    raise RuntimeError("clear-spot sampling failed")


def _band_features(win, sr):
    """16 log-spaced band energies of the mono sum, normalised to unit total.
    Shape only: level would carry distance, and distance is geography."""
    import numpy as np
    mono = win.sum(axis=0)
    mags = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    edges = np.geomspace(BAND_LO, BAND_HI, N_BANDS + 1)
    feats = np.array([float(mags[(freqs >= lo) & (freqs < hi)].sum())
                      for lo, hi in zip(edges[:-1], edges[1:])])
    total = float(feats.sum())
    return feats / total if total > 0 else feats


def _probe_acc(X, y, n_classes):
    """One-vs-rest least-squares linear probe: fit on even sample indices,
    score on odd. X rows may be None (episode with no event) — such rows are
    misses on eval and are skipped for fitting. Returns eval accuracy over
    ALL eval rows, misses counted wrong."""
    import numpy as np
    idx = np.arange(len(y))
    tr = [i for i in idx if i % 2 == 0 and X[i] is not None]
    ev = [i for i in idx if i % 2 == 1]
    if not tr or not ev:
        return 0.0
    Xtr = np.array([X[i] for i in tr])
    Xtr = np.hstack([Xtr, np.ones((len(tr), 1))])
    Ytr = np.zeros((len(tr), n_classes))
    for row, i in enumerate(tr):
        Ytr[row, y[i]] = 1.0
    W, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
    hits = 0
    for i in ev:
        if X[i] is None:
            continue                        # silence: unclassifiable, a miss
        pred = int(np.argmax(np.append(X[i], 1.0) @ W))
        if pred == y[i]:
            hits += 1
    return hits / len(ev)


def _experiment(seed: int) -> dict:
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    from playground import PlaygroundParams, make_playground
    from ContactAudio import AudioEvent, ContactAudioSynth

    rng = np.random.RandomState(seed * 211 + 7)
    p = PlaygroundParams(seed=seed, n_objects=3)
    model, data, water = make_playground(p)
    steps = int(EPISODE_S / model.opt.timestep)
    bodies = _free_bodies(model)
    by_name = dict(bodies)
    target = by_name["apple"]
    apple_r = 0.06                          # build_mjcf: sphere size 0.06

    probe_synth = ContactAudioSynth(model)  # fundamentals + sr, no stepping
    sr = probe_synth.sr

    def run_episode(place_fn, tbid):
        """Reset, park everything, place the target, step, return (synth,
        first-event, sustained-count, all target events)."""
        mujoco.mj_resetData(model, data)
        for j, (_, bid) in enumerate(bodies):
            _place(model, data, bid, (-4.5 + 0.5 * j, -5.2, 0.25))
        place_fn()
        mujoco.mj_forward(model, data)
        synth = ContactAudioSynth(model)
        synth.set_listener(LISTENER, 0.0)
        for _ in range(steps):
            data.xfrc_applied[:] = 0
            if water:
                water.apply(model, data)
            mujoco.mj_step(model, data)
            synth.step(data)
        tgeoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == tbid}
        evs = [e for e in synth.events
               if (e.geom1 in tgeoms or e.geom2 in tgeoms) and e.t > 0.05]
        sustained = 0
        if evs:
            sustained = sum(1 for e in evs if e.t > evs[0].t + SUSTAIN_AFTER_S)
        return synth, (evs[0] if evs else None), sustained, evs

    def feature_of(synth, e):
        audio = synth.render(duration=e.t + WIN_S + 0.02)
        w0 = int(e.t * sr)
        win = audio[:, w0:w0 + int(WIN_S * sr)]
        finite = bool(np.all(np.isfinite(audio)))
        peak = float(np.abs(audio).max())
        return _band_features(win, sr), finite, peak

    X, y, spawns = [], [], []
    finite = True
    peak = 0.0
    n_events = {c: 0 for c in CLASSES}
    kinds_seen = []
    kind_correct = 0
    kind_total = 0
    sustained_creak = sustained_roll = 0
    water_kind_events = creak_kind_events = roll_kind_events = 0

    for ci, cls in enumerate(CLASSES):
        for _ in range(N_EP):
            if cls == "impact":
                wx, wy = _clear_spot(rng, p.pool_size)
                sx, sy, place = wx, wy, lambda wx=wx, wy=wy: _place(
                    model, data, target, (wx, wy, 1.2))
            elif cls == "water":
                wx = POOL_X + float(rng.uniform(-0.4, 0.4)) * p.pool_size
                wy = POOL_Y + float(rng.uniform(-0.4, 0.4)) * p.pool_size
                sx, sy, place = wx, wy, lambda wx=wx, wy=wy: _place(
                    model, data, target, (wx, wy, 1.2))
            elif cls == "creak":
                dx = float(rng.uniform(0.45, 0.85))
                wx, wy = SEESAW_X + dx, SEESAW_Y + float(rng.uniform(-0.15, 0.15))
                sx, sy, place = wx, wy, lambda wx=wx, wy=wy: _place(
                    model, data, target, (wx, wy, 0.25 + apple_r + 0.02))
            else:                            # roll
                wx, wy = _clear_spot(rng, p.pool_size)
                ang = float(rng.uniform(-math.pi, math.pi))
                vel = (2.5 * math.cos(ang), 2.5 * math.sin(ang), 0.0)
                sx, sy, place = wx, wy, lambda wx=wx, wy=wy, vel=vel: _place(
                    model, data, target, (wx, wy, apple_r + 0.01), vel)

            synth, e0, sustained, evs = run_episode(place, target)
            spawns.append((sx, sy))
            y.append(ci)
            if e0 is None:
                X.append(None)
            else:
                feats, fin, pk = feature_of(synth, e0)
                X.append(feats)
                finite = finite and fin
                peak = max(peak, pk)
            n_events[cls] += len(evs)
            if cls == "creak":
                sustained_creak += sustained
            if cls == "roll":
                sustained_roll += sustained
            for e in evs:
                kind = getattr(e, "kind", None)
                kinds_seen.append(kind)
                kind_total += 1
                expected = ({"impact"} if cls == "impact" else
                            WATER_KINDS if cls == "water" else
                            {"creak"} if cls == "creak" else ROLL_KINDS)
                if kind in expected:
                    kind_correct += 1
                if cls == "water" and kind in WATER_KINDS:
                    water_kind_events += 1
                if cls == "creak" and kind == "creak":
                    creak_kind_events += 1
                if cls == "roll" and kind in ROLL_KINDS:
                    roll_kind_events += 1

    sep = _probe_acc(X, y, len(CLASSES))

    # Instrument aliveness: same features + probe must separate two bodies
    # whose fundamentals differ >= 25%, dropped at the SAME spot.
    f_apple = probe_synth.fundamental(
        next(g for g in range(model.ngeom) if model.geom_bodyid[g] == target))
    partner, ratio = None, 1.0
    for name, bid in bodies:
        if bid == target:
            continue
        gid = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid)
        f = probe_synth.fundamental(gid)
        r = max(f, f_apple) / max(min(f, f_apple), 1e-9)
        if r > ratio:
            partner, ratio = bid, r
    alive_acc = 0.0
    if partner is not None and ratio >= 1.25:
        ax, ay = _clear_spot(rng, p.pool_size)
        Xa, ya = [], []
        for k in range(2 * N_ALIVE):
            # Label sequence 0,1,1,0 repeating: index parity is UNCORRELATED
            # with the label, so the even/odd split trains and evaluates on
            # both classes. (First draft used k % 2, which put every apple in
            # train and every partner in eval — the probe scored 0.0 on a
            # separation the features carry at ratio 2.55. Caught by its own
            # smoke run; kept here as the reason the sequence looks odd.)
            bid = target if (k % 4) in (0, 3) else partner
            synth, e0, _, _ = run_episode(
                lambda bid=bid: _place(model, data, bid, (ax, ay, 1.2)), bid)
            ya.append(0 if (k % 4) in (0, 3) else 1)
            if e0 is None:
                Xa.append(None)
            else:
                feats, fin, pk = feature_of(synth, e0)
                Xa.append(feats)
                finite = finite and fin
                peak = max(peak, pk)
        alive_acc = _probe_acc(Xa, ya, 2)

    # Presence, per the hypothesis's own semantics: a placement onset is not
    # a creak; an underwater thud is not a water entry.
    impact_present = float(n_events["impact"] > 0)
    water_present = float(water_kind_events > 0)
    # Creak/roll presence requires BOTH a sustained event (the persisting
    # contact voiced beyond its placement onset — an onset ring at first
    # touch is not a creak, and neither is the apple rolling off the plank
    # and thudding on the floor) AND a correctly-KINDED event in that class's
    # own episodes.
    creak_present = float(sustained_creak > 0 and creak_kind_events > 0)
    roll_present = float(sustained_roll > 0 and roll_kind_events > 0)
    classes_present = impact_present + water_present + creak_present + roll_present

    has_kind_label = float(kind_total > 0 and all(k is not None for k in kinds_seen))
    kind_label_acc = (kind_correct / kind_total) if kind_total else 0.0
    has_self_flag = float("is_self" in getattr(AudioEvent, "__dataclass_fields__", {}))

    return {
        "four_class_audio_separability": round(sep, 4),
        "classes_present": classes_present,
        "impact_present": impact_present,
        "water_present": water_present,
        "creak_present": creak_present,
        "roll_present": roll_present,
        "impact_events": float(n_events["impact"]),
        "water_events_any": float(n_events["water"]),
        "water_kind_events": float(water_kind_events),
        "creak_events_any": float(n_events["creak"]),
        "roll_events_any": float(n_events["roll"]),
        "sustained_creak_events": float(sustained_creak),
        "sustained_roll_events": float(sustained_roll),
        "creak_kind_events": float(creak_kind_events),
        "roll_kind_events": float(roll_kind_events),
        "has_kind_label": has_kind_label,
        "kind_label_acc": round(kind_label_acc, 4),
        "has_self_flag": has_self_flag,
        "alive_two_pitch_acc": round(alive_acc, 4),
        "alive_f0_ratio": round(ratio, 3),
        "audio_finite": float(finite),
        "peak_amp": round(peak, 4),
    }


def _control(seed: int) -> dict:
    """POSITION-ONLY probe (registry control): predict the class from the
    spawn point alone, same split discipline. It must be at chance for a
    PASS. It re-runs the scenario placements WITHOUT any audio — geography
    in, geography out."""
    sys.path.insert(0, str(REPO))
    import numpy as np
    from playground import PlaygroundParams

    rng = np.random.RandomState(seed * 211 + 7)   # same stream => same spots
    p = PlaygroundParams(seed=seed, n_objects=3)
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(N_EP):
            if cls == "impact":
                wx, wy = _clear_spot(rng, p.pool_size)
            elif cls == "water":
                wx = POOL_X + float(rng.uniform(-0.4, 0.4)) * p.pool_size
                wy = POOL_Y + float(rng.uniform(-0.4, 0.4)) * p.pool_size
            elif cls == "creak":
                dx = float(rng.uniform(0.45, 0.85))
                wx, wy = SEESAW_X + dx, SEESAW_Y + float(rng.uniform(-0.15, 0.15))
            else:
                wx, wy = _clear_spot(rng, p.pool_size)
                rng.uniform(-math.pi, math.pi)      # keep streams aligned
            X.append(np.array([wx, wy]))
            y.append(ci)
    return {"position_only_acc": round(_probe_acc(X, y, len(CLASSES)), 4)}


def _check(m: dict, c: dict):
    # Instrument-dead => VOID, never FAIL (the at-chance-control lesson).
    if not (m.get("audio_finite", 0.0) >= 1.0
            and m.get("impact_present", 0.0) >= 1.0
            and m.get("alive_two_pitch_acc", 0.0) >= ALIVE_GATE):
        return Status.VOID
    return (m["classes_present"] >= 4.0
            and m["has_kind_label"] >= 1.0
            and m["kind_label_acc"] >= 0.9
            and m["has_self_flag"] >= 1.0
            and m["four_class_audio_separability"] >= SEP_GATE
            and c["position_only_acc"] <= POS_GATE)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["HR.5"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
