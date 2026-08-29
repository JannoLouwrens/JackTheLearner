"""SM.03 — The nose reports what the eye cannot: occluded-source localisation.

SUCCESSOR TO SM.02 (parked 2026-08-20), designed around its measured failure
mode: the park was a LEARNABILITY bottleneck in the RL rig — three
mechanism-level repairs (Euclid shaping, geodesic phi, undiscounted F) each
fixed a real fault and none moved the nosmell ratios — so this claim removes
policy learning entirely. A small SUPERVISED readout (the T3.01/UB.9 pattern
the certified stack demonstrably supports) must recover the DIRECTION of an
occluded odour source from the certified bilateral channel, on held-out
source layouts, while an identically-budgeted readout on vision alone stays
at chance — and proves itself alive by rising well above chance the moment
the occluder is removed.

THE TASK, exactly. A stationary head at the centre of a purpose-built world
performs one pre-declared 360° sniff-scan (8 s, one full revolution at the
5 Hz sniff rate = 40 bilateral samples of the certified `OdourSensor`). From
that window alone, classify the source's bearing RELATIVE TO THE INITIAL
HEADING into N_BINS = 8 bins. Train on N_TRAIN_L source layouts, test on
N_TEST_L layouts whose source positions are ≥ MIN_SEP_M from every training
position (zero overlap, enforced at generation and recorded) — the
memorisation route T2.15 measured on language is closed structurally, and a
raw-bytes hash gate over the discriminative inputs (odour windows, open-world
frames) must read zero train/test collisions (T3.01 v3's gate).

THE WORLD, and why it is purpose-built rather than the playground. A ring of
N_PANELS = 8 identical opaque panels (radius RING_R, height PANEL_H) encloses
the head; sources sit OUTSIDE the ring (radius in SRC_R_RANGE), so every
retained layout has its source occluded — asserted per layout with FIVE rays
(centre, ±lateral, ±vertical offsets on the source ball, `bodyexclude` = the
ball itself) that must ALL be blocked; layouts failing the assert are redrawn
and the rejection rate recorded. The ring is rotationally symmetric and
identical in every layout, so the occluded scene carries no information about
the source bearing BY CONSTRUCTION — that is what an occluder is — and the
burden of proving the comparison is real is carried by the two legs the spec
pre-registers: the five-ray occlusion assert, and the ALIVE-PROOF (the same
vision readout, same budget, on the SAME source positions with the panels
removed, must land well above chance). A vision leg at chance because its
instrument is dead is VOID, not FAIL (T3.01/24th-audit rule, designed in).

WHY THE WIND POINTS AT HIM, declared rather than apologised for. Per layout
the wind is WIND_SPEED along source→centre (crosswind jitter and the
certified meander stay at their defaults). A smell test in which the smell
never reaches the nose measures the wind lottery, not the sense — SM.02's
park is one long receipt for letting the rig's difficulty stand in for the
sense's. What keeps this honest is that NO arm observes the wind: the only
route from wind to label runs through the odour values themselves, and the
placebo (matched-statistics: the same certified field sampled along the same
scan schedule from an iid random pose and initial heading — zero mutual
information with the label) plus the shuffled-field control (each layout's
window swapped for a DIFFERENT layout's, labels kept) must both sit at
chance or the run is VOID. A WHIFF-COVERAGE gate (fraction of layouts whose
window contains ≥1 whiff, SM.01's 10×NOISE_SIGMA line) below its floor is
VOID — the field never delivered, the claim was not tested.

VISION'S INPUT is a 4-frame 90°-fovy panorama (full 360° coverage — the
matched counterpart of the nose's full-revolution scan) rendered at 64×64
from the head's pose via the process-lifetime renderers this box's two GL
traps require (a GC'd renderer poisons the shared display; a uniform frame
looks exactly like a blind sensor). A canary scene is rendered at start and
end of dataset generation: colour count below CANARY_COLORS_MIN or a
start/end byte mismatch is VOID. The hash gate deliberately EXCLUDES the
occluded-world frames: the symmetric ring makes byte-identical occluded
frames across layouts EXPECTED (they carry the "no information" the claim
needs), so a collision there is not a leak.

"IDENTICAL READOUT" means identical PROTOCOL, said plainly: input encoders
necessarily differ per modality (an MLP on the 480-float odour window, a
small CNN on the 12×64×64 panorama; both param counts recorded), but every
arm gets the same pre-registered LR grid, the same epochs/batch/weight-decay,
the same train-internal 1-in-5 validation split for LR selection, and a full
retrain on all rows at the chosen LR (T3.01's protocol verbatim; PROGRESS
08-25 FOR THE BUILDER 2's matched-tuning-budget rule applied at birth). The
shuffled control trains at the odour arm's chosen LR (T3.01's convention)
and its TRAIN fit is recorded — not gated (the 24th-audit demotion) — so an
at-chance test reading can never again be claimed by an arm that never
trained.

VERDICT TREE, pre-registered:
  VOID  — canary fails, alive-proof below VIS_OPEN_MIN, whiff coverage below
          WHIFF_LAYOUT_MIN, any hash collision, or placebo/shuffled above
          CTRL_CEIL (a control exploiting a leak indicts the rig, not the
          claim).
  FAIL  — odour_occ below ODOUR_OCC_MIN on any seed (the certified field
          carries no usable direction information at the sniff rate and
          receiver geometry Jack actually has — the registry's falsifier,
          and it kills the "smell works when sight fails" GOAL.md claim at
          the readout level), OR vis_occ above VIS_OCC_CEIL (the occlusion
          is decorative).
  PASS  — odour_occ ≥ ODOUR_OCC_MIN on every seed AND vis_occ ≤ VIS_OCC_CEIL
          on every seed, with every VOID gate green.

GATE PROVENANCE (all frozen before the registered dispatch; the pilot may
not move them afterward). Chance is 1/8 = 0.125. With N_TEST_L = 240 the
binomial sd at chance is 0.021, so the at-chance ceilings (VIS_OCC_CEIL,
CTRL_CEIL = 0.22) sit ~4.5σ above chance — noise cannot trip them and a real
leak still can. ODOUR_OCC_MIN = 0.25 is 2× chance, ~6σ — "well above chance"
with no reading of the pilot's exact value into the bar. VIS_OPEN_MIN = 0.60
is an instrument-liveness floor (pilot reads far higher; the floor is where
"alive" stops being arguable), WHIFF_LAYOUT_MIN = 0.80 likewise.

PILOT: not yet run. The seed-90 pilot (disjoint from the registered 0/1/2 —
PG.6's precedent) must run before _GATES_FROZEN flips True; its numbers are
recorded HERE when it lands, and the bars above do not move on its account
(the pilot exists to catch rig faults before the registered spend — SM.02's
lesson).

COVERS: smell (claim).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the certified field read by the certified sensor; both
# hash into this certificate (SM.01's convention). render.py carries the GL
# traps the vision leg depends on.
IMPL_DEPS = ["experiments/odour.py", "experiments/render.py"]

SEEDS = [0, 1, 2]
PILOT_SEED = 90

# ── the world, pre-registered ────────────────────────────────────────────
N_BINS = 8
N_PANELS = 8
RING_R = 1.2                   # m, panel ring radius
PANEL_HALF_W = 0.42            # panel half-width → 0.84 m wide, ~0.10 m gaps
PANEL_H = 0.70                 # top of panel; ball top is 0.24 → fully hidden
SRC_R_RANGE = (1.8, 2.6)       # source radius band, outside the ring
SRC_BALL_R = 0.12
Z_SRC = 0.12                   # resting on the ground plane
Z_NOSE = 0.45
FOVY_DEG = 90.0                # 4 frames x 90° = the full panorama
IMG = 64

# ── the field protocol, pre-registered ───────────────────────────────────
WIND_SPEED = 0.5               # m/s, source → centre; NO arm observes it
PREROLL_S = 30.0               # steady state: 500 puffs / 20 Hz = 25 s
T_WIN_S = 8.0                  # one full revolution
SNIFF_DT = 0.2                 # 1 / odour.SNIFF_HZ
N_SNIFF = int(round(T_WIN_S / SNIFF_DT))          # 40
ROT_RATE = 2.0 * math.pi / T_WIN_S                # rad/s, the sniff-scan
ODOUR_GAIN = 3.0               # tanh gains, SM.02's probed values
DERIV_GAIN = 0.5
WHIFF_THR_MULT = 10.0          # a whiff is > this * NOISE_SIGMA (SM.01's line)

# ── dataset, pre-registered ──────────────────────────────────────────────
N_TRAIN_L = 480                # layouts per seed
N_TEST_L = 240
MIN_SEP_M = 0.25               # test source ≥ this from EVERY train source

# ── training budget, identical for every arm (T3.01's protocol) ──────────
LR_GRID = (1e-4, 3e-4, 1e-3)
EPOCHS = 40
BATCH = 64
WEIGHT_DECAY = 1e-4

# ── gates (PROVISIONAL until the seed-90 pilot lands; provenance above) ──
_GATES_FROZEN = False

CHANCE = 1.0 / N_BINS
ODOUR_OCC_MIN = 0.25           # per seed — the claim
VIS_OCC_CEIL = 0.22            # per seed — occlusion must not be decorative
CTRL_CEIL = 0.22               # placebo AND shuffled, per seed — else VOID
VIS_OPEN_MIN = 0.60            # per seed — instrument-liveness, VOID below
WHIFF_LAYOUT_MIN = 0.80        # per seed — the field must have delivered
CANARY_COLORS_MIN = 50
SHUF_FIT_FLOOR = 0.35          # RECORDED diagnostic, not gated (24th audit)

ODOUR_DIM = 12 * N_SNIFF       # 480: [L4, R4, d/dt 4] x 40 sniffs


# ── the world as MJCF ────────────────────────────────────────────────────
def _mjcf(panels: bool) -> str:
    parts = [
        '<mujoco model="sm03">',
        f'<visual><global fovy="{FOVY_DEG}" offwidth="{IMG}" '
        f'offheight="{IMG}"/></visual>',
        '<worldbody>',
        '<light directional="true" pos="0 0 3" dir="0 0 -1" '
        'diffuse="0.9 0.9 0.9" specular="0 0 0"/>',
        '<geom name="floor" type="plane" size="6 6 0.1" '
        'rgba="0.35 0.42 0.35 1"/>',
    ]
    if panels:
        for k in range(N_PANELS):
            a = 2.0 * math.pi * k / N_PANELS
            x, y = RING_R * math.cos(a), RING_R * math.sin(a)
            deg = math.degrees(a)
            parts.append(
                f'<geom name="panel{k}" type="box" '
                f'size="0.05 {PANEL_HALF_W} {PANEL_H / 2}" '
                f'pos="{x:.4f} {y:.4f} {PANEL_H / 2}" '
                f'euler="0 0 {deg:.2f}" rgba="0.45 0.40 0.35 1"/>')
    parts += [
        '<body name="source" pos="2.0 0 0.12">'
        f'<geom name="source_ball" type="sphere" size="{SRC_BALL_R}" '
        'rgba="0.95 0.25 0.10 1"/></body>',
        '</worldbody>', '</mujoco>',
    ]
    return "\n".join(parts)


# Process-lifetime handles (render.py's GC trap: a collected Renderer poisons
# the shared display for every renderer after it).
_WORLDS: dict = {}


def _world(panels: bool):
    import mujoco
    key = bool(panels)
    if key not in _WORLDS:
        model = mujoco.MjModel.from_xml_string(_mjcf(panels))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        _WORLDS[key] = (model, data, mujoco.Renderer(model, IMG, IMG))
    return _WORLDS[key]


def _move_source(model, data, pos) -> None:
    import mujoco
    model.body("source").pos[:] = pos
    mujoco.mj_forward(model, data)


def _ray_clear(model, data, a, b, exclude_body: int) -> bool:
    """One ray a→b; True when nothing solid sits strictly between."""
    import mujoco
    a = np.asarray(a, dtype=float)
    d = np.asarray(b, dtype=float) - a
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return True
    gid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, a, d / dist, None, 1, exclude_body, gid)
    return not (0.0 <= hit < dist - 1e-6)


def _occluded(model, data, src_pos) -> bool:
    """Five rays from the nose to the ball (centre, ±lateral, ±vertical),
    ball excluded from the cast; ALL must be blocked."""
    eye = np.array([0.0, 0.0, Z_NOSE])
    c = np.asarray(src_pos, dtype=float)
    lat = np.array([-c[1], c[0], 0.0])
    lat = lat / (np.linalg.norm(lat) + 1e-12) * SRC_BALL_R
    up = np.array([0.0, 0.0, SRC_BALL_R])
    body = model.body("source").id
    targets = (c, c + lat, c - lat, c + up, c - up)
    return all(not _ray_clear(model, data, eye, t, body) for t in targets)


def _panorama(renderer, data, h0: float) -> np.ndarray:
    """4 frames at h0 + k*90°, (12, IMG, IMG) float32 in [0,1]."""
    import mujoco
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, Z_NOSE)
    cam.distance = 0.01
    cam.elevation = 0.0
    frames = []
    for k in range(4):
        cam.azimuth = math.degrees(h0) + 90.0 * k
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    stack = np.concatenate(frames, axis=2)          # (H, W, 12)
    return (stack.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _canary(renderer, model, data) -> tuple:
    """Fixed scene, fixed camera; (n_colors, sha256). Uniform frame == blind
    sensor; a byte drift between start and end == degraded GL context."""
    _move_source(model, data, (2.0, 0.0, Z_SRC))
    frame = _panorama(renderer, data, 0.0)
    by = np.ascontiguousarray(frame).tobytes()
    q = (frame[:3] * 255).astype(np.uint8).reshape(3, -1)
    colors = len(set(map(tuple, q.T)))
    return colors, hashlib.sha256(by).hexdigest()


# ── odour windows ────────────────────────────────────────────────────────
def _odour_window(model, data, src_pos, h0: float, layout_seed: int,
                  placebo_rng=None) -> tuple:
    """One sniff-scan. Returns (window[480], whiff_any).

    The field is the CERTIFIED PuffField through its public sample() only;
    wind blows source→centre (docstring). With placebo_rng the window is the
    matched-statistics placebo: the SAME field read along the SAME scan
    schedule from an iid random pose and initial heading.
    """
    from .. import odour
    sx, sy = float(src_pos[0]), float(src_pos[1])
    w = -np.array([sx, sy, 0.0])
    w = w / (np.linalg.norm(w) + 1e-12) * WIND_SPEED
    field = odour.PuffField(
        [odour.Source("food0", "food", (sx, sy, Z_SRC))],
        wind=tuple(w), seed=layout_seed, los=True)
    n_pre = int(round(PREROLL_S / SNIFF_DT))
    for _ in range(n_pre):
        field.step(SNIFF_DT)
    sensor = odour.OdourSensor(field)
    rng = np.random.RandomState(layout_seed + 17)

    if placebo_rng is not None:
        pos = np.array([placebo_rng.uniform(-2.6, 2.6),
                        placebo_rng.uniform(-2.6, 2.6), Z_NOSE])
        head0 = float(placebo_rng.uniform(-math.pi, math.pi))
    else:
        pos = np.array([0.0, 0.0, Z_NOSE])
        head0 = h0

    fc = odour.CHANNEL_INDEX["food"]
    out = np.zeros((N_SNIFF, 12), dtype=np.float32)
    whiff = False
    t = PREROLL_S
    for i in range(N_SNIFF):
        h = head0 + ROT_RATE * (t - PREROLL_S)
        raw = sensor.obs(pos, h, t, model=model, data=data, rng=rng)
        out[i, :8] = np.tanh(ODOUR_GAIN * raw[:8])
        out[i, 8:] = np.tanh(DERIV_GAIN * raw[8:])
        if max(raw[fc], raw[4 + fc]) > WHIFF_THR_MULT * odour.NOISE_SIGMA:
            whiff = True
        field.step(SNIFF_DT)
        t += SNIFF_DT
    return out.reshape(-1), whiff


# ── dataset generation ───────────────────────────────────────────────────
def _draw_layout(rng, model, data, avoid: list) -> tuple:
    """(src_pos, h0, label, n_rejected). Rejection: not occluded, or within
    MIN_SEP_M of any position in `avoid`."""
    rej = 0
    for _ in range(4000):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        r = rng.uniform(*SRC_R_RANGE)
        pos = (r * math.cos(theta), r * math.sin(theta), Z_SRC)
        if avoid and min(math.hypot(pos[0] - ax, pos[1] - ay)
                         for ax, ay in avoid) < MIN_SEP_M:
            rej += 1
            continue
        _move_source(model, data, pos)
        if not _occluded(model, data, pos):
            rej += 1
            continue
        h0 = float(rng.uniform(-math.pi, math.pi))
        rel = (theta - h0) % (2.0 * math.pi)
        label = int(rel / (2.0 * math.pi / N_BINS)) % N_BINS
        return pos, h0, label, rej
    raise RuntimeError("no occluded layout in 4000 draws")


def _build_split(seed: int, n: int, base_offset: int, avoid: list) -> dict:
    """All inputs for one split. Renders occluded AND open panoramas."""
    m_occ, d_occ, r_occ = _world(True)
    m_open, d_open, r_open = _world(False)
    rng = np.random.RandomState(seed * 1_000_003 + base_offset)
    placebo_rng = np.random.RandomState(seed * 1_000_003 + base_offset + 7)
    X_od, X_pl, X_vo, X_vn, y = [], [], [], [], []
    pos_list, rejected, whiffs = [], 0, 0
    for i in range(n):
        pos, h0, label, rej = _draw_layout(rng, m_occ, d_occ, avoid)
        rejected += rej
        lseed = seed * 2_000_003 + base_offset + i
        w, whiff_any = _odour_window(m_occ, d_occ, pos, h0, lseed)
        p, _ = _odour_window(m_occ, d_occ, pos, h0, lseed + 500_009,
                             placebo_rng=placebo_rng)
        whiffs += int(whiff_any)
        _move_source(m_occ, d_occ, pos)
        X_vo.append(_panorama(r_occ, d_occ, h0))
        _move_source(m_open, d_open, pos)
        X_vn.append(_panorama(r_open, d_open, h0))
        X_od.append(w)
        X_pl.append(p)
        y.append(label)
        pos_list.append((pos[0], pos[1]))
    return {
        "odour": np.stack(X_od), "placebo": np.stack(X_pl),
        "vis_occ": np.stack(X_vo), "vis_open": np.stack(X_vn),
        "y": np.asarray(y, dtype=np.int64), "positions": pos_list,
        "rejected": rejected, "whiff_frac": whiffs / n,
    }


# ── the readouts (identical protocol; encoders per modality) ─────────────
def _make_mlp(torch, nn, seed: int, dev):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(ODOUR_DIM, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(),
                         nn.Linear(64, N_BINS)).to(dev)


def _make_cnn(torch, nn, seed: int, dev):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(12, 16, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(64, N_BINS)).to(dev)


def _train_one(torch, make, seed: int, lr: float, X, y, dev):
    net = make(torch, torch.nn, seed, dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr,
                           weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed + 31)
    n = X.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for j in range(0, n, BATCH):
            idx = perm[j:j + BATCH]
            opt.zero_grad()
            loss = lossf(net(X[idx].to(dev)), y[idx].to(dev))
            loss.backward()
            opt.step()
    return net


def _acc(torch, net, X, y, dev) -> float:
    with torch.no_grad():
        pred = []
        for j in range(0, X.shape[0], 256):
            pred.append(net(X[j:j + 256].to(dev)).argmax(1).cpu())
        return float((torch.cat(pred) == y).float().mean())


def _fit_arm(torch, make, seed: int, Xtr, ytr, Xte, yte, dev,
             fixed_lr: float | None = None):
    """T3.01's protocol: LR grid on a 1-in-5 val split, retrain on all rows.
    fixed_lr short-circuits selection (the shuffled control trains at the
    real arm's chosen LR)."""
    if fixed_lr is None:
        val = np.arange(Xtr.shape[0]) % 5 == 0
        fit = ~val
        best_lr, best = None, -1.0
        for lr in LR_GRID:
            net = _train_one(torch, make, seed, lr, Xtr[fit], ytr[fit], dev)
            va = _acc(torch, net, Xtr[val], ytr[val], dev)
            if va > best:
                best, best_lr = va, lr
    else:
        best_lr = fixed_lr
    net = _train_one(torch, make, seed, best_lr, Xtr, ytr, dev)
    return _acc(torch, net, Xte, yte, dev), best_lr, net


def _hashes(arr: np.ndarray) -> set:
    return {hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            for a in arr}


# ── everything for the given seeds (runs on the GPU VM or locally) ───────
def remote_run(seeds: list, n_train: int | None = None,
               n_test: int | None = None, epochs: int | None = None,
               lr_grid: tuple | None = None) -> dict:
    """Reduced-size arguments exist ONLY for the smoke; the registered run
    uses the defaults."""
    global EPOCHS, LR_GRID
    if epochs is not None:
        EPOCHS = epochs
    if lr_grid is not None:
        LR_GRID = lr_grid
    n_train = n_train or N_TRAIN_L
    n_test = n_test or N_TEST_L

    from ..render import ensure_gl
    ensure_gl()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_occ, d_occ, r_occ = _world(True)
    canary_colors, canary0 = _canary(r_occ, m_occ, d_occ)

    out = {"gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
           "canary_colors": canary_colors, "seeds": []}
    for seed in seeds:
        tr = _build_split(seed, n_train, 0, avoid=[])
        te = _build_split(seed, n_test, 500_000, avoid=tr["positions"])
        min_sep = min(
            math.hypot(tx - ax, ty - ay)
            for tx, ty in te["positions"] for ax, ay in tr["positions"])

        # Structural leak gate over the DISCRIMINATIVE inputs (docstring:
        # occluded frames are expected to collide across layouts).
        ov_od = len(_hashes(tr["odour"]) & _hashes(te["odour"]))
        ov_vn = len(_hashes(tr["vis_open"]) & _hashes(te["vis_open"]))

        T = torch.from_numpy
        ytr, yte = T(tr["y"]), T(te["y"])
        acc_od, lr_od, _ = _fit_arm(torch, _make_mlp, seed,
                                    T(tr["odour"]), ytr,
                                    T(te["odour"]), yte, dev)
        acc_pl, _, _ = _fit_arm(torch, _make_mlp, seed + 1,
                                T(tr["placebo"]), ytr,
                                T(te["placebo"]), yte, dev)
        acc_vo, _, _ = _fit_arm(torch, _make_cnn, seed + 2,
                                T(tr["vis_occ"]), ytr,
                                T(te["vis_occ"]), yte, dev)
        acc_vn, _, _ = _fit_arm(torch, _make_cnn, seed + 3,
                                T(tr["vis_open"]), ytr,
                                T(te["vis_open"]), yte, dev)

        # Shuffled-field control: DIFFERENT layout's window, labels kept —
        # a roll by one within each split; trained at the odour arm's LR.
        Xsh_tr = T(np.roll(tr["odour"], 1, axis=0))
        Xsh_te = T(np.roll(te["odour"], 1, axis=0))
        acc_sh, _, net_sh = _fit_arm(torch, _make_mlp, seed + 4,
                                     Xsh_tr, ytr, Xsh_te, yte, dev,
                                     fixed_lr=lr_od)
        fit_sh = _acc(torch, net_sh, Xsh_tr, ytr, dev)

        n_mlp = sum(p.numel() for p in
                    _make_mlp(torch, torch.nn, 0, "cpu").parameters())
        n_cnn = sum(p.numel() for p in
                    _make_cnn(torch, torch.nn, 0, "cpu").parameters())
        out["seeds"].append({
            "seed": seed,
            "acc_odour_occ": round(acc_od, 4), "lr_odour": lr_od,
            "acc_vis_occ": round(acc_vo, 4),
            "acc_vis_open": round(acc_vn, 4),
            "acc_placebo": round(acc_pl, 4),
            "acc_shuffled": round(acc_sh, 4),
            "shuffled_train_fit": round(fit_sh, 4),
            "whiff_frac_min": round(min(tr["whiff_frac"],
                                        te["whiff_frac"]), 4),
            "reject_rate": round(
                (tr["rejected"] + te["rejected"])
                / (tr["rejected"] + te["rejected"] + n_train + n_test), 4),
            "min_sep": round(min_sep, 4),
            "hash_overlap_odour": ov_od, "hash_overlap_vis_open": ov_vn,
            "n_params_mlp": n_mlp, "n_params_cnn": n_cnn,
        })
    colors_end, canary1 = _canary(r_occ, m_occ, d_occ)
    out["canary_ok"] = bool(canary0 == canary1)
    out["canary_colors_end"] = colors_end
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
# Pinned + loud on purpose: 2026-08-20's sdist-before-wheels lesson.
_sp.run([_sys.executable, "-m", "pip", "install", "mujoco==3.11.0"],
        check=True)
import json
from experiments.tests.sm_03_nose_reports_occluded import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "sm03.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    # Kaggle first: a kernel computes server-side whether or not anyone
    # watches, and W34's expiring hours are the budget this spec spends.
    res = submit(job, prefer="kaggle",
                 est_hours=round(0.15 + 0.20 * len(seeds), 2),
                 timeout_s=3600 + 1500 * len(seeds),
                 fetch=["sm03.json"])
    if not res.ok:
        raise RuntimeError(f"SM.03 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["sm03.json"]).read_text())
    out["backend"] = res.backend
    return out


def _summarise(rows: list, top: dict) -> dict:
    return {
        "acc_odour_occ": [r["acc_odour_occ"] for r in rows],
        "acc_vis_occ": [r["acc_vis_occ"] for r in rows],
        "acc_vis_open": [r["acc_vis_open"] for r in rows],
        "odour_occ_min": min(r["acc_odour_occ"] for r in rows),
        "vis_occ_max": max(r["acc_vis_occ"] for r in rows),
        "vis_open_min": min(r["acc_vis_open"] for r in rows),
        "whiff_frac_min": min(r["whiff_frac_min"] for r in rows),
        "hash_overlap_max": max(max(r["hash_overlap_odour"],
                                    r["hash_overlap_vis_open"])
                                for r in rows),
        "min_sep": min(r["min_sep"] for r in rows),
        "reject_rate_max": max(r["reject_rate"] for r in rows),
        "canary_ok": bool(top.get("canary_ok", False)),
        "canary_colors": min(top.get("canary_colors", 0),
                             top.get("canary_colors_end", 0)),
        "n_params_mlp": rows[0]["n_params_mlp"],
        "n_params_cnn": rows[0]["n_params_cnn"],
        "lrs_odour": [r["lr_odour"] for r in rows],
    }


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    m = _summarise(_CACHE["seeds"], _CACHE)
    m["gpu"] = _CACHE["gpu"]
    m["backend"] = _CACHE["backend"]
    return m


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    return {
        "acc_placebo": [r["acc_placebo"] for r in rows],
        "acc_shuffled": [r["acc_shuffled"] for r in rows],
        "placebo_max": max(r["acc_placebo"] for r in rows),
        "shuffled_max": max(r["acc_shuffled"] for r in rows),
        "shuffled_fit_min": min(r["shuffled_train_fit"] for r in rows),
    }


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["canary_ok"]:
        return Status.VOID          # GL context degraded mid-run
    if m["canary_colors"] < CANARY_COLORS_MIN:
        return Status.VOID          # uniform frame == blind sensor
    if m["vis_open_min"] < VIS_OPEN_MIN:
        return Status.VOID          # the eye's instrument is dead (registry:
                                    # "VOID, not FAIL")
    if m["whiff_frac_min"] < WHIFF_LAYOUT_MIN:
        return Status.VOID          # the field never delivered — untested
    if m["hash_overlap_max"] != 0:
        return Status.VOID          # train/test share an input byte-for-byte
    if c["placebo_max"] > CTRL_CEIL:
        return Status.VOID          # noise carried the task — a leak
    if c["shuffled_max"] > CTRL_CEIL:
        return Status.VOID          # wrong layout's field carried it — a leak
    # shuffled_fit_min is RECORDED, not gated (SHUF_FIT_FLOOR; 24th-audit
    # demotion): below the floor the control's at-chance test reading proves
    # nothing; the hash gate above carries identity.
    # The claim, per seed on both legs:
    return (m["odour_occ_min"] >= ODOUR_OCC_MIN
            and m["vis_occ_max"] <= VIS_OCC_CEIL)


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "SM.03 gates are provisional — pilot first, freeze the bars in "
            "this file, then run (SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["SM.03"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


# ── dry check: every verdict path against fabricated rows ────────────────
def _dry():
    base = dict(canary_ok=True, canary_colors=800, vis_open_min=0.95,
                whiff_frac_min=0.97, hash_overlap_max=0,
                odour_occ_min=0.34, vis_occ_max=0.15)
    ctrl = {"placebo_max": 0.15, "shuffled_max": 0.14,
            "shuffled_fit_min": 0.9}
    cases = [
        ("planted pass", base, ctrl, Status.PASS),
        ("odour at chance -> FAIL", {**base, "odour_occ_min": 0.14}, ctrl,
         Status.FAIL),
        ("occlusion decorative -> FAIL", {**base, "vis_occ_max": 0.40}, ctrl,
         Status.FAIL),
        ("eye dead -> VOID", {**base, "vis_open_min": 0.20}, ctrl,
         Status.VOID),
        ("no whiffs -> VOID", {**base, "whiff_frac_min": 0.30}, ctrl,
         Status.VOID),
        ("hash overlap -> VOID", {**base, "hash_overlap_max": 2}, ctrl,
         Status.VOID),
        ("placebo leak -> VOID", base, {**ctrl, "placebo_max": 0.35},
         Status.VOID),
        ("shuffled leak -> VOID", base, {**ctrl, "shuffled_max": 0.35},
         Status.VOID),
        ("canary drift -> VOID", {**base, "canary_ok": False}, ctrl,
         Status.VOID),
        ("blind canary -> VOID", {**base, "canary_colors": 3}, ctrl,
         Status.VOID),
        ("dead shuffled arm -> recorded, not gated", base,
         {**ctrl, "shuffled_fit_min": 0.12}, Status.PASS),
    ]
    for name, m, cc, want in cases:
        got = _check(dict(m), dict(cc))
        got = {True: Status.PASS, False: Status.FAIL}.get(got, got)
        assert got == want, f"{name}: wanted {want}, got {got}"
        print(f"  ok: {name}")
    print("DRY OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dry":
        _dry()
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local, tiny, CPU: plumbing, not learnability.
        out = remote_run([PILOT_SEED], n_train=24, n_test=16, epochs=2,
                         lr_grid=(3e-4,))
        print(json.dumps(out, indent=1))
    elif len(sys.argv) > 1 and sys.argv[1] == "pilot":
        # Full-size single off-run seed, LOCALLY on CPU (this spec's dataset
        # generation is CPU-bound; the pilot's job is rig faults, not speed).
        out = remote_run([PILOT_SEED])
        print(json.dumps(out, indent=1))
    else:
        print("usage: sm_03_nose_reports_occluded.py {dry|smoke|pilot}")
