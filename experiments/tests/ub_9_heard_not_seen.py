"""UB.9 — Heard, not seen: the task that is impossible without fusion.

THE CLAIM. On the certified HNS scene (PG.7), audio carries WHICH SIZE fell
(modal fundamental) and a pre-event frame carries WHICH SIZE IS WHERE — and
only the two together carry WHICH SLOT fell. I(audio;Y) = I(vision;Y) = 0,
I(audio,vision;Y) = 1 bit: a physical XOR, one bit of pure synergy. A fused
model that answers the slot above chance has therefore BOUND a heard identity
to a seen position. This is the smallest experiment that could establish "his
senses work in unison" (GOAL.md), and if it fails no larger experiment rescues
the claim.

WHAT IS REUSED, AND WHY REUSE IS LOAD-BEARING HERE. The audio path is
byte-for-byte PG.7's: `hns_scene.build` -> `ContactAudioSynth` -> PG.7's own
`_features` (12 log band energies, level, pan — the onset found from the
waveform, never from the synth's event labels). PG.7 certified that exactly
this feature vector cannot name the slot (0.503 vs a 0.53 gate) and cannot
name the size from level+pan alone. Re-implementing the features would step
outside that certificate; importing them keeps UB.9 inside it, and IMPL_DEPS
hashes both files so a change to either goes stale loudly.

THE EYE. One render model, compiled once per process and held alive with its
Renderer for the process lifetime (PG.6's `_Eye` lesson: a GC'd Renderer
poisons the shared X display and the NEXT renderer returns plausible corrupted
frames). The scene is rendered in the LISTENER frame — slots at azimuths
+theta and pi-theta, both in the +y half-plane — from a camera at
(0, -CAM_BACK, CAM_H) looking along +y, so both slots are always in view
(worst-case slot bearing from the camera: atan(2.6*cos30 / (1.3+3.5)) = 25.1
deg + 2.6 deg of sphere limb, inside the 30 deg half-FOV of fovy=60 on a
square frame). Rendering egocentrically in the listener frame makes the image
yaw-invariant, exactly as the pan is: yaw is common-mode nuisance in both
modalities. A canary frame is rendered at construction and re-checked after
every arm; if it moves, the run is VOID, not FAIL — a degraded sensor is an
invalid run, not evidence against binding.

WHY VISION IS AT CHANCE BY CONSTRUCTION. The frame is pre-event: both
candidates stand at z = r + fall_h and the freejoint is invisible. Within a
quad, the two episodes sharing a `large_slot` differ ONLY in `faller_slot`, so
their frames are the same frame — this file renders one frame per
(quad, large_slot) and both episodes read it, which makes "vision cannot know
the label" a property of the data structure rather than a hope about the
renderer. Audio is at chance on the slot by PG.7's certificate. All three
nulls are measured anyway, as the registry demands.

THE MODELS. The task is an XOR, so the fused arm is the smallest model that
can represent one: an MLP with one hidden layer stack over the concatenated
(standardised) 48x48 grey frame and the 14 audio features. The unimodal nulls
are the same architecture minus the other modality, and the LATE ENSEMBLE
averages the two unimodal arms' softmax outputs — the arm that is structurally
incapable of synergy. Beating the best single modality is not synergy; beating
the ensemble is the registered metric (`hns_accuracy_over_ensemble`).

PRE-REGISTERED GATES (registry: fused >= 0.75 mean over 3 seeds, lower
bootstrap CI > 0.5; controls: swap-flip >= 0.80, spectrum-flattened audio to
chance, pan-shuffle changes nothing). Implementation constants, chosen before
the first recorded run:

    fused_acc            >= 0.75   (registry, on the 3-seed mean)
    boot_lo              >  0.5    per seed — 2.5th pct of a BY-QUAD cluster
                                   bootstrap (episodes within a quad share
                                   their nuisance draw; resampling episodes
                                   would understate the CI)
    audio_only_acc       <= NULL_GATE   per seed
    vision_only_acc      <= NULL_GATE   per seed
    ensemble_acc         <= NULL_GATE   per seed
    swap_flip_rate       >= 0.80   per seed (on previously-correct test
                                   episodes, the radii-exchanged frame — which
                                   is the quad sibling's frame, i.e. the exact
                                   re-render the spec asks for — must flip the
                                   prediction)
    flat_audio_acc       <= NULL_GATE   per seed (bands set to their own mean,
                                   level and pan kept: identity is only in the
                                   spectrum, PG.7 P2/P3)
    pan_shuffle_delta    <= 0.05   per seed (pan is uninformative here;
                                   sensitivity to it means a leak)
    vision_carries_bit   >= 0.90   per seed — a LINEAR probe on the raw frame
                                   must read large_slot (the PG.8 lesson: a
                                   fixture that carries nothing passes every
                                   leak test ever written)
    dropped_quads_frac   <= 0.02   per seed (quads are dropped WHOLE when any
                                   episode yields no usable audio, so class
                                   balance is exact by construction)

    control (must fail): the same fused architecture trained on CROSS-EPISODE
    SWAPPED pairs — each audio matched to a frame from a different quad, label
    kept with the audio. Correspondence destroyed, marginals preserved (the
    registry's stated ablation primitive). Gate: <= NULL_GATE. If the control
    learns, something unimodal carries the label and PG.7's certificate has
    been violated — investigate, do not celebrate.

NULL_GATE is 0.60: chance + ~3.5 sigma at n_test = 320 episodes in 80 quad
clusters (binomial SE 2.8%, and the cluster structure cannot push the null's
variance past the unclustered bound because labels are exactly balanced within
every quad).

B3 (24th audit) — what distinguishes a NEVER-TRAINED unimodal arm from a
converged-at-chance null: (i) the same training code must drive the fused
arm to >= FUSED_GATE in the same seed, so a globally dead trainer cannot
produce a PASS; (ii) vision_carries_bit / audio_carries_bit >= 0.90 prove
each arm's input features are decodable where signal exists. RECORDED GAP,
not a silently added gate: the unimodal arms have no must-learn target of
their own and their loss descent is not recorded, so a PER-ARM recipe
pathology (UB.10's measured disease — one uniform recipe leaving one
matched-param arm dead) is not fully excluded by this design.

Sizing: 400 quads/seed = 1600 episodes, 320 quads train / 80 test, split by
quad so no nuisance draw straddles the boundary. At 0.90 true accuracy the
binomial SE on 320 test episodes is 1.7%, so the 0.75 gate sits ~9 sigma below
a working fused arm and ~9 above a broken one — the result is not a coin flip.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ensure_gl() must precede the mujoco import (experiments/render.py: GLX under
# Xvfb; there is no libEGL and no libOSMesa on this box).
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402
from .pg_7_hns_leakage import WIN_S, _features, _logreg_cv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# This spec's claim rests on the certified fixture and the certified feature
# extractor; all of it hashes into impl_sha so drift goes stale loudly.
IMPL_DEPS = ["experiments/hns_scene.py",
             "experiments/tests/pg_7_hns_leakage.py",
             "ContactAudio.py"]

N_QUADS = 400            # per seed; first 320 train, last 80 test (by quad)
N_TEST_QUADS = 80
EPISODE_S = 0.90         # PG.7's window: the longest fall lands at ~0.35 s
RES = 96                 # rendered px (square); downsampled 2x2 -> 48 for MLPs
CAM_BACK = 3.5           # m behind the listener, on the -y axis
CAM_H = 1.2              # m
FUSED_GATE = 0.75        # registry
NULL_GATE = 0.60         # chance + ~3.5 sigma at n_test=320; see docstring
FLIP_GATE = 0.80         # registry control
PAN_DELTA_MAX = 0.05
VISION_BIT_GATE = 0.90
DROP_MAX = 0.02
N_BOOT = 1000

EPOCHS = 250
HIDDEN = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 128


# ── the eye: one compile, one GL context, a canary ───────────────────────

def _render_mjcf() -> str:
    """The vision channel's world: floor, two size-editable spheres, a light,
    and the camera. Listener frame — the camera never moves."""
    bodies = []
    for slot, rgba in ((0, "0.85 0.35 0.25 1"), (1, "0.25 0.45 0.85 1")):
        bodies.append(
            f'    <body name="cand{slot}" pos="0 2 1">\n'
            f'      <freejoint/>\n'
            f'      <geom name="cand{slot}" type="sphere" size="0.15" '
            f'mass="0.5" rgba="{rgba}"/>\n'
            f'    </body>')
    return (
        '<mujoco model="hns-eye">\n'
        '  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
        f'  <visual><global offwidth="{RES}" offheight="{RES}"/></visual>\n'
        '  <worldbody>\n'
        '    <light pos="0 0 6" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>\n'
        '    <geom name="floor" type="plane" size="8 8 0.1" '
        'rgba="0.55 0.55 0.58 1"/>\n'
        f'    <camera name="eye" pos="0 {-CAM_BACK} {CAM_H}" '
        'xyaxes="1 0 0 0 0 1" fovy="60"/>\n'
        + '\n'.join(bodies) + '\n'
        '  </worldbody>\n'
        '</mujoco>\n')


class _Eye:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_render_mjcf())
        self.data = mujoco.MjData(self.model)
        self.gid = [self.model.geom(f"cand{s}").id for s in (0, 1)]
        bid = [self.model.body(f"cand{s}").id for s in (0, 1)]
        self.qadr = [self.model.jnt_qposadr[self.model.body_jntadr[b]]
                     for b in bid]
        self.r = mujoco.Renderer(self.model, height=RES, width=RES)
        self._canary_ref = None
        self._canary_ref = self.canary()

    def canary(self) -> float:
        """Fixed reference frame reduced to one number; drift means the GL
        context degraded and the RUN is invalid (VOID), per PG.6."""
        f = self.frame(math.radians(40.0), 2.0, 0.5, large_slot=0)
        return float(np.round(f.sum(), 3))

    def frame(self, theta: float, rng_range: float, fall_h: float,
              large_slot: int) -> np.ndarray:
        """The pre-event frame: both candidates standing at z = r + fall_h.
        Listener frame, so yaw never appears."""
        from ..hns_scene import R_LARGE, R_SMALL
        radii = [R_SMALL, R_SMALL]
        radii[large_slot] = R_LARGE
        az = (theta, math.pi - theta)
        for s in (0, 1):
            self.model.geom_size[self.gid[s], 0] = radii[s]
            q = self.qadr[s]
            self.data.qpos[q:q + 3] = (rng_range * math.cos(az[s]),
                                       rng_range * math.sin(az[s]),
                                       radii[s] + fall_h)
            self.data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.r.update_scene(self.data, camera="eye")
        g = self.r.render().astype(np.float32).mean(axis=2) / 255.0
        return g


_EYE: list = []


def get_eye() -> _Eye:
    if not _EYE:
        _EYE.append(_Eye())
    return _EYE[0]


def _pool2(g: np.ndarray) -> np.ndarray:
    h = g.shape[0] // 2
    return g.reshape(h, 2, h, 2).mean(axis=(1, 3))


# ── data: one quad = one nuisance draw, four episodes, two frames ────────

def _generate(seed: int):
    """Returns per-episode arrays plus the per-(quad, large_slot) frame table.

    Frames are stored ONCE per (quad, large_slot): the two episodes that share
    one are the two faller labels, so the impossibility of reading the label
    from the frame is structural, not photometric. The swap-flip control reads
    the sibling entry — the exact radii-exchanged re-render the spec names.
    """
    sys.path.insert(0, str(REPO))
    from ContactAudio import ContactAudioSynth

    from ..hns_scene import build, draw_quad, listener_pose

    eye = get_eye()
    rng = np.random.RandomState(seed * 7919 + 13)
    frames = {}                       # (quad, large_slot) -> pooled grey frame
    raw_frames = {}                   # (quad, large_slot) -> full-res frame
    audio_f, y_slot, y_large_fell, large_slot_of, quad_of = [], [], [], [], []
    n_quads_dropped = 0
    q = 0
    while q < N_QUADS:
        quad = draw_quad(rng)
        rows = []
        ok = True
        for ep in quad:
            model, data = build(ep)
            lpos, lyaw = listener_pose(ep)
            synth = ContactAudioSynth(model)
            synth.set_listener(lpos, lyaw)
            for _ in range(int(EPISODE_S / model.opt.timestep)):
                mujoco.mj_step(model, data)
                synth.step(data)
            if not synth.events:
                ok = False
                break
            e0 = synth.events[0]
            audio = synth.render(duration=e0.t + WIN_S + 0.05)
            f = _features(audio, synth.sr)
            if f is None:
                ok = False
                break
            bands, level, pan = f
            rows.append((np.concatenate([bands, [level, pan]]),
                         ep.faller_slot, float(ep.faller_radius > 0.18),
                         ep.large_slot))
        if not ok:
            # Drop the quad WHOLE: partial quads would unbalance the labels
            # and hand every probe a class prior.
            n_quads_dropped += 1
            continue
        for ls in (0, 1):
            g = eye.frame(quad[0].theta, quad[0].rng_range, quad[0].fall_h, ls)
            raw_frames[(q, ls)] = g
            frames[(q, ls)] = _pool2(g)
        for af, fs, lf, ls in rows:
            audio_f.append(af)
            y_slot.append(fs)
            y_large_fell.append(lf)
            large_slot_of.append(ls)
            quad_of.append(q)
        q += 1

    return {
        "audio": np.asarray(audio_f, dtype=np.float32),
        "y": np.asarray(y_slot, dtype=np.int64),
        "y_large_fell": np.asarray(y_large_fell, dtype=np.float32),
        "large_slot": np.asarray(large_slot_of, dtype=np.int64),
        "quad": np.asarray(quad_of, dtype=np.int64),
        "frames": frames,
        "raw_frames": raw_frames,
        "dropped_frac": n_quads_dropped / (N_QUADS + n_quads_dropped),
    }


_DATA: dict = {}


def _data_for(seed: int):
    """run_spec calls _experiment and _control separately per seed; the
    generated world is identical for both, so it is cached (the module-cache
    pattern SYSTEM.md mandates for costly work inside seed loops)."""
    if seed not in _DATA:
        _DATA[seed] = _generate(seed)
    return _DATA[seed]


def _vision_matrix(d, idx) -> np.ndarray:
    return np.stack([d["frames"][(int(d["quad"][i]), int(d["large_slot"][i]))].ravel()
                     for i in idx])


# ── the models ───────────────────────────────────────────────────────────

def _train_mlp(Xtr, ytr, seed: int):
    """One hidden stack over standardised features; returns a predict_proba.
    Standardisation params come from TRAIN only."""
    import torch
    torch.manual_seed(seed * 104729 + 7)
    torch.set_num_threads(2)
    mu = Xtr.mean(0)
    sd = np.where(Xtr.std(0) < 1e-9, 1.0, Xtr.std(0))
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    t = torch.tensor(ytr, dtype=torch.long)
    net = torch.nn.Sequential(
        torch.nn.Linear(A.shape[1], HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed * 15485863 + 3)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(A), generator=g)
        for i in range(0, len(A), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(net(A[b]), t[b]).backward()
            opt.step()
    net.eval()

    def proba(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.tensor((X - mu) / sd, dtype=torch.float32)
            return torch.softmax(net(z), dim=1).numpy()
    return proba


def _acc(proba, X, y) -> float:
    return float((proba(X).argmax(1) == y).mean())


def _ridge_cls_cv(X, y, groups, folds: int = 4, l2: float = 1.0) -> float:
    """Linear probe for the WIDE fixture check (d ~ 9216): ridge regression on
    +-1 labels in the DUAL (n x n Gram, PG.6's trick), thresholded at 0,
    grouped k-fold by quad. `_logreg_cv`'s IRLS is primal and would solve a
    9217^2 system 25 times per fold; this is one n_tr^3 solve per fold."""
    X = np.asarray(X, dtype=np.float64)
    t = np.where(np.asarray(y, dtype=float) > 0.5, 1.0, -1.0)
    groups = np.asarray(groups)
    correct = 0
    for f in range(folds):
        te = (groups % folds) == f
        tr = ~te
        A = X[tr] - X[tr].mean(0)
        B = X[te] - X[tr].mean(0)
        alpha = np.linalg.solve(A @ A.T + l2 * np.eye(int(tr.sum())),
                                t[tr] - t[tr].mean())
        pred = B @ (A.T @ alpha) + t[tr].mean()
        correct += int(((pred > 0) == (t[te] > 0)).sum())
    return correct / len(t)


def _cluster_boot_lo(correct: np.ndarray, quads: np.ndarray, seed: int) -> float:
    """2.5th percentile of accuracy under a by-quad cluster bootstrap."""
    rng = np.random.RandomState(seed * 22271 + 5)
    uq = np.unique(quads)
    by_q = {q: correct[quads == q] for q in uq}
    accs = []
    for _ in range(N_BOOT):
        pick = rng.choice(uq, size=len(uq), replace=True)
        accs.append(float(np.concatenate([by_q[q] for q in pick]).mean()))
    return float(np.percentile(accs, 2.5))


# ── the experiment ───────────────────────────────────────────────────────

def _experiment(seed: int) -> dict:
    d = _data_for(seed)
    eye = get_eye()
    quads = d["quad"]
    test_q = np.unique(quads)[-N_TEST_QUADS:]
    te = np.isin(quads, test_q)
    tr = ~te

    Xa = d["audio"]
    Xv_tr = _vision_matrix(d, np.where(tr)[0])
    Xv_te = _vision_matrix(d, np.where(te)[0])
    y = d["y"]

    fused = _train_mlp(np.hstack([Xv_tr, Xa[tr]]), y[tr], seed)
    audio_only = _train_mlp(Xa[tr], y[tr], seed)
    vision_only = _train_mlp(Xv_tr, y[tr], seed)

    Xf_te = np.hstack([Xv_te, Xa[te]])
    fused_p = fused(Xf_te)
    fused_pred = fused_p.argmax(1)
    fused_acc = float((fused_pred == y[te]).mean())
    audio_acc = _acc(audio_only, Xa[te], y[te])
    vision_acc = _acc(vision_only, Xv_te, y[te])
    ens_p = (audio_only(Xa[te]) + vision_only(Xv_te)) / 2.0
    ens_acc = float((ens_p.argmax(1) == y[te]).mean())

    correct = (fused_pred == y[te]).astype(float)
    boot_lo = _cluster_boot_lo(correct, quads[te], seed)

    # SWAP-FLIP: radii exchanged between positions = the sibling frame.
    te_idx = np.where(te)[0]
    swapped = np.stack([
        d["frames"][(int(quads[i]), 1 - int(d["large_slot"][i]))].ravel()
        for i in te_idx])
    swap_pred = fused(np.hstack([swapped, Xa[te]])).argmax(1)
    was_right = fused_pred == y[te]
    flip_rate = float((swap_pred[was_right] != fused_pred[was_right]).mean()) \
        if was_right.any() else 0.0

    # Spectrum-flatten: bands -> their own mean; level and pan kept.
    Xa_flat = Xa[te].copy()
    Xa_flat[:, :12] = Xa_flat[:, :12].mean(axis=1, keepdims=True)
    flat_acc = float((fused(np.hstack([Xv_te, Xa_flat])).argmax(1) == y[te]).mean())

    # Pan-shuffle: permute the pan column across test episodes.
    rng = np.random.RandomState(seed * 31337 + 1)
    Xa_pan = Xa[te].copy()
    Xa_pan[:, 13] = Xa_pan[rng.permutation(len(Xa_pan)), 13]
    pan_acc = float((fused(np.hstack([Xv_te, Xa_pan])).argmax(1) == y[te]).mean())
    pan_delta = abs(pan_acc - fused_acc)

    # Fixture must-succeed probes (linear, grouped-CV over the whole set):
    # the frame must carry WHICH SIZE IS WHERE; the bands must carry WHICH
    # SIZE FELL. A scene carrying neither passes every leak test.
    all_idx = np.arange(len(y))
    Xv_all_raw = np.stack([
        d["raw_frames"][(int(quads[i]), int(d["large_slot"][i]))].ravel()
        for i in all_idx])
    vision_bit = _ridge_cls_cv(Xv_all_raw, d["large_slot"].astype(float), quads)
    bands = Xa[:, :12] - Xa[:, [12]]          # level-normalised, PG.7 P3
    audio_bit = _logreg_cv(bands, d["y_large_fell"], quads)

    canary_ok = float(eye.canary() == eye._canary_ref)

    m = {
        "fused_acc": round(fused_acc, 4),
        "audio_only_acc": round(audio_acc, 4),
        "vision_only_acc": round(vision_acc, 4),
        "ensemble_acc": round(ens_acc, 4),
        "hns_accuracy_over_ensemble": round(fused_acc - ens_acc, 4),
        "boot_lo": round(boot_lo, 4),
        "swap_flip_rate": round(flip_rate, 4),
        "flat_audio_acc": round(flat_acc, 4),
        "pan_shuffle_delta": round(pan_delta, 4),
        "vision_carries_bit": round(vision_bit, 4),
        "audio_carries_bit": round(audio_bit, 4),
        "dropped_quads_frac": round(d["dropped_frac"], 4),
        "n_test": int(te.sum()),
        "canary_ok": canary_ok,
    }
    # Per-seed indicators: run_spec records the seed MEAN, so 1.0 here is the
    # statement "no seed failed" — a mean of raw numbers is not.
    m["seed_gates_ok"] = float(
        m["boot_lo"] > 0.5
        and m["audio_only_acc"] <= NULL_GATE
        and m["vision_only_acc"] <= NULL_GATE
        and m["ensemble_acc"] <= NULL_GATE
        and m["swap_flip_rate"] >= FLIP_GATE
        and m["flat_audio_acc"] <= NULL_GATE
        and m["pan_shuffle_delta"] <= PAN_DELTA_MAX
        and m["vision_carries_bit"] >= VISION_BIT_GATE
        and m["audio_carries_bit"] >= VISION_BIT_GATE
        and m["dropped_quads_frac"] <= DROP_MAX)
    return m


def _control(seed: int) -> dict:
    """Cross-episode SWAP: every audio keeps its label but is paired with a
    frame from a DIFFERENT quad (derangement over quads). Correspondence
    destroyed, marginals preserved. The fused architecture MUST fall to
    chance; if it learns, a single modality carries the label and PG.7's
    certificate is violated somewhere."""
    d = _data_for(seed)
    quads = d["quad"]
    test_q = np.unique(quads)[-N_TEST_QUADS:]
    te = np.isin(quads, test_q)
    tr = ~te
    y = d["y"]
    Xa = d["audio"]

    rng = np.random.RandomState(seed * 48271 + 9)
    pool_tr, pool_te = np.unique(quads[tr]), test_q

    def mismatched(idx):
        # A fixed nonzero rotation within each split pairs every audio with a
        # frame from a DIFFERENT quad, and the frame's large_slot is redrawn —
        # keeping the episode's own large_slot would carry the very bit the
        # control exists to destroy (the frame would still show the large
        # sphere on the true side, and fused would still solve the XOR).
        shift = int(rng.randint(1, N_TEST_QUADS - 1))
        rows = []
        for i in idx:
            pool = pool_te if te[i] else pool_tr
            q2 = int(pool[(np.searchsorted(pool, quads[i]) + shift) % len(pool)])
            rows.append(d["frames"][(q2, int(rng.randint(0, 2)))].ravel())
        return np.stack(rows)

    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    fused = _train_mlp(np.hstack([mismatched(tr_idx), Xa[tr]]), y[tr], seed)
    acc = float((fused(np.hstack([mismatched(te_idx), Xa[te]])).argmax(1)
                 == y[te]).mean())
    return {
        "control_swap_pairing_acc": round(acc, 4),
        "control_seed_gates_ok": float(acc <= NULL_GATE),
    }


def _check(m: dict, c: dict):
    if m.get("canary_ok", 0.0) != 1.0:
        return Status.VOID          # the eye degraded mid-run; invalid, not false
    return (m["seed_gates_ok"] == 1.0
            and m["fused_acc"] >= FUSED_GATE
            and m["hns_accuracy_over_ensemble"] >= 0.15
            and c["control_seed_gates_ok"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["UB.9"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
