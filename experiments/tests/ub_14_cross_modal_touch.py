"""UB.14 — Cross-modal prediction, against the null that usually wins.

THE CLAIM. In the playground, with Jack's body tumbling under a random policy,
masked TOUCH (per-body contact-force magnitudes from `cfrc_ext`) is predicted
from VISION + PROPRIOCEPTION better than from proprioception ALONE at matched
capacity — and both learned arms sit above the unconditional-mean floor. The
registry names the honest and likely outcome in its own falsified_by: foot
contact is largely inferable from joint state, so vision may add nothing here.
That outcome is REPORTED, never retried.

WHY THIS VENUE. The playground is the world every visual certificate already
runs through (PG.6's eye is part of the world contract) and the humanoid is the
body PG.8 certified against Humanoid-v5. Touch is read from `cfrc_ext` through
`playground.step` — the one stepping kernel, which exists precisely because
bare `mj_step` leaves all 78 contact columns silently zero (PG.8/PG.2 lesson,
recorded in `playground.py:step`). No synthetic modality generator is involved:
this is the first UB spec whose three modalities are all read off the real body
in the real world.

THE PROPRIO DEFINITION, pre-registered because it decides the question. If
"proprioception" includes the root's WORLD position, floor contact is a
near-function of pose + height and vision has nothing left to earn — the test
would be decided by a leak, not a measurement. A body's internal senses do not
include a world position readout (Humanoid-v5 packs root z into its obs for
convenience, not biology). So:

    proprio (44 dims) = 17 joint angles (qpos[7:24])
                      + 17 joint velocities (qvel[6:23])
                      + root orientation quaternion (4)      [vestibular: gravity]
                      + root angular velocity (3)            [vestibular: canals]
                      + root linear velocity in BODY frame (3) [otolith-like]

    excluded: root world x, y, z and anything downstream of cfrc/cinert.

Vision is the fixed playground eye (EYE_POS world contract): 96x96 grey,
2x2-pooled to 48x48 = 2304 features. It sees WHERE the body is relative to the
floor and the props — exactly the information proprio was denied. A recorded,
NON-GATING diagnostic arm (`proprioz_r2`) adds root z back so the Review can
see how much of any vision gain is height alone.

THE SPAWN IS MOVED, AND WHY THAT IS LEGITIMATE. The humanoid's default spawn
(0, -1.6) is MEASURED to be outside the eye's view: 66 deg off the optical
axis against a 30 deg half-FOV, rendered frame diff vs an empty scene exactly
0.0 at every default-region position (grid scan, 2026-09-01). A blind venue
cannot test this claim and cannot honestly fail it. The eye may not move — it
is world contract and every visual certificate hangs off it — but the spawn is
a per-experiment `PlaygroundParams.humanoid_spawn` parameter (PG.8 itself
overrides it). SPAWN_XY = (-2.25, -2.25) is the measured optimum: over the
whole +-0.4 m jitter box the standing-body frame diff is 113-160 and the
LYING-body diff 62-108 (the body stays visible after it falls, which is where
most touch happens), with zero spawn contacts anywhere in the box. The
`vision_sees_body` fixture gate re-verifies visibility on every recorded run.

MATCHED CAPACITY, exactly. All learned arms share ONE architecture over the
full 2348-dim (vision ++ proprio) input: masked arms receive constant zeros in
the channels they are denied (constant-zero inputs get zero gradient; the
standardiser floors sd at 1e-2 so near-constant train pixels can neither blow
up on test nor resurrect a masked block). Parameter counts are therefore
IDENTICAL by construction, not approximately matched by hidden-width algebra.
Every arm is measured at its best epoch on a shared by-episode validation
split carved from TRAIN (VAL_EP episodes) — the shakedown measured the
matched net memorising train and paying on test, which would drown exactly
the small real effects the Kepler note predicts; epoch selection is
symmetric, so no arm gains an optimisation advantage.

    fused        : [vision | proprio] -> 128 -> 64 -> 13
    proprio-only : [zeros  | proprio] -> same net shape, fresh init
    vision-only  : [vision | zeros ]  -> same (recorded diagnostic, non-gating)
    mean null    : train-mean touch vector (the floor, parameter-free)

TOUCH TARGET. 13 channels: log1p of the contact-force magnitude
(||cfrc_ext[b, 3:6]||) for each humanoid body, read after `playground.step`
(which calls `mj_rnePostConstraint`; reading before it is the PG.2 scar).
Channels are standardised by TRAIN stats; a channel is ALIVE for scoring when
its train contact rate is in [0.02, 0.98] and its train std > 1e-4 — a channel
that never fires or never releases carries no question. At least
ALIVE_MIN channels must be alive or the rig is VOID (a world without varied
contact cannot measure touch prediction).

DATA. N_EPISODES short lives per seed. Each starts from the PG.8 reset plus a
deliberately violent draw — root lifted U(0, 0.5) m, tilted, joints scattered
U(-0.4, 0.4), spawn jittered ±0.4 m — so the body falls, tumbles, drags and
settles differently each episode; a uniform-random policy in the actuators'
ctrlrange keeps it twitching. EP_DECISIONS decisions of frame_skip 5; one
sample (frame + proprio + touch) every SAMPLE_EVERY decisions. Episodes with
any non-finite qpos/qvel/touch are dropped WHOLE (dropped_frac <= DROP_MAX or
VOID). Split BY EPISODE: last N_TEST_EP episodes are test — no life straddles
the boundary, and all inference clusters by episode.

PRE-REGISTERED GATES — implementation constants chosen before the first
recorded run. Effect sizes calibrated by the registry's own note
(Kepler-Encoder, arXiv:2607.13522: real fused-vs-single R^2 gains of
0.049/-0.001/0.187 — real, clean, small):

    CLAIM (all must hold):
      r2_gain = fused_r2 - proprio_r2 >= 0.02   on the 3-seed mean (GAIN_MIN;
                                                 under the smallest real
                                                 Kepler effect, above noise)
      boot_lo > 0 per seed — 2.5th pct of the mean per-sample error improvement
                (standardised squared error summed over alive channels,
                 proprio-arm minus fused-arm), by-episode cluster bootstrap
    FLOOR / LEARNING GATE (VOID if unmet — data-starved, not refuted):
      fused_r2 >= 0.05        (nothing predicts touch -> no venue for the claim)
      loss_fell: every learned arm's final train loss <= 0.8x its first-epoch
                 loss (UB.10's measured disease: one matched arm silently dead)
    FIXTURE ALIVENESS (VOID if unmet — instrument dead, not evidence):
      canary_ok               (PG.6: a degraded GL context is an invalid run)
      vision_sees_body >= 0.5 (grouped-CV ridge R^2, pooled frame -> root xy;
                               an eye that cannot locate the body cannot help
                               and cannot honestly lose)
      alive_channels >= 4     of 13
      dropped_frac <= 0.10
    CONTROL (must fail): the fused architecture retrained with vision frames
      DERANGED ACROSS EPISODES (same within-episode index, wrong episode —
      marginals preserved, correspondence destroyed). It must NOT satisfy the
      claim pair (its boot_lo > 0 AND gain >= GAIN_MIN would mean the "vision
      gain" never needed correspondence: a leak, not fusion). Its own aliveness
      is asserted — control_r2 >= proprio_r2 - 0.10 and its loss fell — because
      an at-chance control must carry proof its instrument was alive
      (LESSONS.md, 24th-audit rule).

SIZING. 48 episodes x ~66 samples = ~3168 samples/seed, 792 test in 12 episode
clusters. The by-episode bootstrap prices the heavy within-life autocorrelation
honestly; with 12 clusters the 2.5th percentile is a coarse but conservative
bound, and the per-seed direction gate x3 seeds plus the mean-effect gate is
the claim, not any single interval. Wall cost ~10-15 min/seed on this box
(render ~40 s, sim ~15 s, 4 trained arms) — inside cpu<2h with margin.

WHAT A FAIL MEANS (and it is the registry's own predicted outcome): proprio
matches fused — vision adds nothing to touch prediction in this venue at this
capacity. That deletes the justification for UB.10's vision->touch masked
objective arm (the spec's kills clause) and is DESIGN INPUT for the fusion
bakeoff, exactly as T4.02's FAIL was. Report it; do not re-roll it.

PROBE RECORD (2026-09-01, seed 90, pre-first-recorded-run window; gates and
code untouched by any of it). The full-envelope shakedown read CHECK -> VOID:
vision_sees_body 0.3738 < 0.5, fused_r2 0.0053 < 0.05 (proprio-only 0.1164),
control_alive_ok 0 (control_r2 -0.0239 < proprio - 0.10). Three repair probes
followed, and EVERY candidate lever was measured spent:

  1. RESOLUTION (probe 1, 16 eps): raw-pixel ridge -> root xy reads 0.205 at
     96x96 full, 0.260 at 48x48, 0.264 at 24x24 — flat in resolution. Episode
     scaling 16 -> 48 eps moves 0.26 -> 0.37, rising but saturating.
  2. INSTRUMENT CLASS (probe 2, 48 eps): body-blob centroid features
     (|frame - median-frame| centroid + mass + second moments, the nonlinear
     extraction ridge cannot express) read 0.2747-0.2950 — WORSE than raw
     pixels. l2 in {1e-4..1} flat.
  3. NONLINEAR READOUT (probe 3, 48 eps): the rig's OWN MLP trainer,
     frame -> root xy, best-val epoch, held-out episodes: R^2 0.159. The
     information itself is insufficient — no readout class closes a 0.16-0.37
     reading to 0.5.
  4. FUSED-ARM DROWNING (probes 2-3): pool4 input (576 dims) moves fused_r2
     0.0053 -> 0.0153; weight decay 1e-3 -> 0.0177, 1e-2 -> 0.0391 — still
     under the 0.05 floor, and the WD lever is CAPPED by the loss_fell
     conjunct: at wd 1e-2 the proprio arm's fell-ratio is 0.786 vs the 0.8
     gate, so one more notch trades the floor VOID for the loss_fell VOID.
     pool2 at wd 1e-2: 0.0148. Control aliveness fails identically (pool4:
     -0.0161 vs needed >= 0.0037). vision_only_r2 0.0093 everywhere: vision
     carries ~no touch-relevant signal under this policy in this venue.

DIAGNOSIS, and it is a VENUE fault, not an instrument fault: the eye is world
contract (EYE_POS may not move; 30 deg half-FOV), the spawn is at its measured
in-view optimum, and the in-view region is only the +-0.4 m jitter box — so
var(root xy) is bounded at ~0.05 m^2/axis while the tumbling body's
blob-centroid-vs-root offset varies by a comparable amount. Explainable
variance caps near the measured 0.37 for ANY decoder; the 0.5 aliveness gate
is unreachable in the only region the world contract allows the body to be
seen in. The fused/control failures follow: 576-2304 dims carrying ~zero
touch signal drown a 44-dim proprio fit at matched capacity, and no symmetric
regularisation clears the floor before tripping loss_fell. Probes preserved
at /data/ub14_probes/ub14_probe{,2,3}.py (numbers recorded here).

THE SETTLEMENT (this slot): the recorded 3-seed run is launched to land the
honest VOID on the ledger — the aliveness gates firing IS the rig doing its
job; the row makes it scoreboard-visible. On harvest, the VOID-FORECLOSED
declaration (FORECLOSURE ARITHMETIC + BLAST RADIUS: none — nothing
depends_on UB.14) is owed via the doc-only amend lane, quoting the row's
fired conjuncts. The repair is a venue/world redesign (an eye that can track
the body, or a body that stays in view — the same fork as D9/W0.BAL and
LT.01's C2), routed to the Review on the `w0-too-shallow` row. Do NOT re-run
UB.14 unchanged and do NOT move VISION_BODY_GATE — a fixture gate lowered to
what a blind venue can pass certifies blindness as sight.
"""
from __future__ import annotations

import atexit
import sys
from pathlib import Path

import numpy as np

# ensure_gl() must precede the mujoco import (experiments/render.py: GLX under
# Xvfb; no libEGL, no libOSMesa on this box).
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# The eye camera, the humanoid and the stepping kernel are all world contract.
IMPL_DEPS = ["playground.py"]

# ── data envelope ─────────────────────────────────────────────────────────
N_EPISODES = 48          # per seed; last N_TEST_EP are test, split by episode
N_TEST_EP = 12
EP_DECISIONS = 200       # frame_skip 5 -> ~2 s of sim per life
SAMPLE_EVERY = 3         # one (frame, proprio, touch) sample per 3 decisions
RES = 96                 # rendered px; 2x2-pooled to 48x48 for the models
RESET_NOISE = 0.01       # PG.8's Humanoid-v5 reset noise (base layer)
LIFT_MAX = 0.5           # extra root lift, m
TILT_MAX = 0.3           # quaternion x/y component scatter
JOINT_SCATTER = 0.4      # rad, initial joint angles
SPAWN_JITTER = 0.4       # m, spawn xy
SPAWN_XY = (-2.25, -2.25)  # measured in-view optimum; docstring scan record

# ── pre-registered gates (docstring rationale) ────────────────────────────
GAIN_MIN = 0.02          # 3-seed mean fused_r2 - proprio_r2
FLOOR_R2 = 0.05          # fused below this -> VOID (data-starved)
LOSS_FELL = 0.8          # final train loss / first-epoch loss, every arm
VISION_BODY_GATE = 0.5   # ridge R^2, pooled frame -> root xy
ALIVE_MIN = 4            # of 13 touch channels
DROP_MAX = 0.10
CONTROL_SLACK = 0.10     # control_r2 >= proprio_r2 - this (aliveness)
N_BOOT = 1000

# ── training recipe (UB.9's, plus early stopping) ─────────────────────────
# The shakedown measured the matched net memorising its train set (train loss
# 0.05x first epoch) and paying for it on test — an overfit that would drown
# exactly the small real effects the Kepler note predicts. Every learned arm
# is therefore measured at its best-generalising epoch, selected on a
# by-episode validation split carved from TRAIN (never test), symmetrically:
# same recipe, same split, every arm.
EPOCHS = 200
HIDDEN = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 128
VAL_EP = 8               # train episodes held out for epoch selection

N_BODY = 13              # humanoid bodies; touch channel count


# ── the world: one model, one renderer, held for the process lifetime ────

class _World:
    """One canonical playground (params seed 0 — the certified, unmutated
    world) with Jack, plus the eye. The Renderer is held alive for the process
    lifetime (PG.6: a GC'd Renderer poisons the shared X display and the NEXT
    renderer returns plausible corrupted frames)."""

    def __init__(self):
        sys.path.insert(0, str(REPO))
        from playground import (PlaygroundParams, humanoid_body_ids,
                                humanoid_index, make_playground)

        self.p = PlaygroundParams(seed=0)
        self.model, self.data, _ = make_playground(self.p, with_humanoid=True)
        ix = humanoid_index(self.model)
        self.q = ix["qposadr"]
        self.d = ix["dofadr"]
        self.bodies = list(humanoid_body_ids(self.model))
        lo = self.model.actuator_ctrlrange[:, 0].copy()
        hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_lo, self.ctrl_hi = lo, hi
        self.renderer = mujoco.Renderer(self.model, height=RES, width=RES)
        # Explicit close at EXIT only (render.py's sanctioned pattern): an
        # implicitly GC'd renderer poisons the live display (PG.6), and a
        # renderer still open when atexit kills the Xvfb turns a recorded PASS
        # into rc=1 via an XIO abort. LIFO atexit ordering runs this before
        # ensure_gl's _stop_xvfb.
        atexit.register(self.renderer.close)
        self._canary_ref = None
        self._canary_ref = self.canary()

    def _reset_canonical(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def canary(self) -> float:
        """Fixed reference frame reduced to one number; drift means the GL
        context degraded and the RUN is invalid (VOID), per PG.6."""
        self._reset_canonical()
        f = self.frame()
        return float(np.round(f.sum(), 3))

    def frame(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="eye")
        return self.renderer.render().astype(np.float32).mean(axis=2) / 255.0

    def reset_episode(self, rng: np.random.RandomState):
        """PG.8's reset plus the violent draw the docstring pre-registers."""
        m, d = self.model, self.data
        q, dof = self.q, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[q:q + 24] += rng.uniform(-RESET_NOISE, RESET_NOISE, 24)
        d.qvel[dof:dof + 23] += rng.uniform(-RESET_NOISE, RESET_NOISE, 23)
        # Absolute, not relative: the default spawn is measured BLIND to the
        # eye (docstring). SPAWN_XY is the measured in-view optimum.
        d.qpos[q + 0] = SPAWN_XY[0] + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER)
        d.qpos[q + 1] = SPAWN_XY[1] + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER)
        d.qpos[q + 2] += rng.uniform(0.0, LIFT_MAX)
        quat = np.array([1.0,
                         rng.uniform(-TILT_MAX, TILT_MAX),
                         rng.uniform(-TILT_MAX, TILT_MAX),
                         rng.uniform(-TILT_MAX, TILT_MAX)])
        d.qpos[q + 3:q + 7] = quat / np.linalg.norm(quat)
        d.qpos[q + 7:q + 24] += rng.uniform(-JOINT_SCATTER, JOINT_SCATTER, 17)
        mujoco.mj_forward(m, d)

    def proprio(self) -> np.ndarray:
        """The 44-dim body-internal vector the docstring pre-registers.
        Root linear velocity is rotated into the body frame so no world-frame
        direction leaks in through it."""
        d, q, dof = self.data, self.q, self.d
        quat = d.qpos[q + 3:q + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        v_world = d.qvel[dof:dof + 3]
        v_body = R.T @ v_world
        return np.concatenate([
            d.qpos[q + 7:q + 24],          # 17 joint angles
            d.qvel[dof + 6:dof + 23],      # 17 joint velocities
            quat,                          # 4  orientation
            d.qvel[dof + 3:dof + 6],       # 3  angular velocity
            v_body,                        # 3  linear velocity, body frame
        ]).astype(np.float32)

    def touch(self) -> np.ndarray:
        """log1p contact-force magnitude per humanoid body. Valid ONLY after
        `playground.step`, which calls mj_rnePostConstraint (the PG.2 scar)."""
        f = self.data.cfrc_ext[self.bodies, 3:6]
        return np.log1p(np.linalg.norm(f, axis=1)).astype(np.float32)

    def root_xyz(self) -> np.ndarray:
        return self.data.qpos[self.q:self.q + 3].astype(np.float32).copy()


_WORLD: list = []


def get_world() -> _World:
    if not _WORLD:
        _WORLD.append(_World())
    return _WORLD[0]


def _pool2(g: np.ndarray) -> np.ndarray:
    h = g.shape[0] // 2
    return g.reshape(h, 2, h, 2).mean(axis=(1, 3))


# ── data generation ───────────────────────────────────────────────────────

def _generate(seed: int) -> dict:
    from playground import step as pg_step

    w = get_world()
    rng = np.random.RandomState(seed * 7919 + 41)
    frames, proprios, touches, roots, episode_of = [], [], [], [], []
    n_dropped = 0
    ep = 0
    while ep < N_EPISODES:
        w.reset_episode(rng)
        rows = []
        ok = True
        for t in range(EP_DECISIONS):
            ctrl = rng.uniform(w.ctrl_lo, w.ctrl_hi)
            pg_step(w.model, w.data, ctrl)
            if t % SAMPLE_EVERY:
                continue
            pr = w.proprio()
            tc = w.touch()
            if not (np.isfinite(w.data.qpos).all()
                    and np.isfinite(pr).all() and np.isfinite(tc).all()):
                ok = False
                break
            rows.append((_pool2(w.frame()).ravel(), pr, tc, w.root_xyz()))
        if not ok:
            # Drop the episode WHOLE: a life that went non-finite has no
            # trustworthy samples on either side of the blow-up.
            n_dropped += 1
            if n_dropped > 3 * N_EPISODES:
                raise RuntimeError("physics non-finite on most episodes")
            continue
        for fr, pr, tc, rt in rows:
            frames.append(fr)
            proprios.append(pr)
            touches.append(tc)
            roots.append(rt)
            episode_of.append(ep)
        ep += 1

    return {
        "vision": np.asarray(frames, dtype=np.float32),
        "proprio": np.asarray(proprios, dtype=np.float32),
        "touch": np.asarray(touches, dtype=np.float32),
        "root": np.asarray(roots, dtype=np.float32),
        "episode": np.asarray(episode_of, dtype=np.int64),
        "dropped_frac": n_dropped / (N_EPISODES + n_dropped),
    }


_DATA: dict = {}


def _data_for(seed: int) -> dict:
    """run_spec calls _experiment and _control separately per seed; the world
    rollout is identical for both, so it is cached (the module-cache pattern
    SYSTEM.md mandates)."""
    if seed not in _DATA:
        _DATA[seed] = _generate(seed)
    return _DATA[seed]


# ── models ────────────────────────────────────────────────────────────────

def _train_mlp(Xtr, Ytr, seed: int, tag: int, groups=None):
    """UB.9's recipe on a regression head, measured at the best-validation
    epoch (docstring: symmetric early stopping). Returns
    (predict, loss_fell_ratio). Standardisation params come from FIT only."""
    import torch
    torch.manual_seed(seed * 104729 + tag * 131 + 7)
    torch.set_num_threads(2)

    # By-episode validation split from TRAIN; clamped so a tiny smoke
    # envelope still trains on something.
    if groups is not None:
        ue = np.unique(np.asarray(groups))
        n_val = min(VAL_EP, max(1, len(ue) // 4))
        val = np.isin(groups, ue[-n_val:])
    else:
        val = np.zeros(len(Xtr), dtype=bool)
    fit = ~val

    mu = Xtr[fit].mean(0)
    # Floor, not epsilon-guard: a pixel that is near-constant in FIT but not
    # elsewhere would otherwise be divided by a microscopic sd and reach the
    # net at ~1e7 (measured: fused test R^2 -3.4e7 in the shakedown). All
    # inputs are O(1) (pixels in [0,1], body-frame kinematics), so 1e-2 is
    # inert for live features and defuses dead ones.
    sd = np.maximum(Xtr[fit].std(0), 1e-2)
    A = torch.tensor((Xtr[fit] - mu) / sd, dtype=torch.float32)
    T = torch.tensor(Ytr[fit], dtype=torch.float32)
    Av = torch.tensor((Xtr[val] - mu) / sd, dtype=torch.float32)
    Tv = torch.tensor(Ytr[val], dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(A.shape[1], HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(seed * 15485863 + tag * 17 + 3)
    first = last = None
    best_val = None
    best_state = None
    for e in range(EPOCHS):
        perm = torch.randperm(len(A), generator=g)
        tot = n = 0.0
        for i in range(0, len(A), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(A[b]), T[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        if e == 0:
            first = tot / n
        last = tot / n
        if len(Av):
            with torch.no_grad():
                v = float(loss_fn(net(Av), Tv))
            if best_val is None or v < best_val:
                best_val = v
                best_state = {k: t.detach().clone()
                              for k, t in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.tensor((X - mu) / sd, dtype=torch.float32)
            return net(z).numpy()
    return predict, float(last / max(first, 1e-12))


def _ridge_reg_cv(X, y, groups, folds: int = 4, l2: float = 10.0) -> float:
    """Grouped-CV ridge R^2 in the dual (UB.9's trick), for the
    vision-sees-body fixture probe. y may be multi-column; returns pooled R^2."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    groups = np.asarray(groups)
    sse = sst = 0.0
    for f in range(folds):
        te = (groups % folds) == f
        tr = ~te
        A = X[tr] - X[tr].mean(0)
        B = X[te] - X[tr].mean(0)
        K = A @ A.T + l2 * np.eye(int(tr.sum()))
        Yc = Y[tr] - Y[tr].mean(0)
        alpha = np.linalg.solve(K, Yc)
        pred = B @ (A.T @ alpha) + Y[tr].mean(0)
        sse += float(((Y[te] - pred) ** 2).sum())
        sst += float(((Y[te] - Y[tr].mean(0)) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def _boot_lo(delta: np.ndarray, episodes: np.ndarray, seed: int) -> float:
    """2.5th percentile of the mean paired improvement under a by-episode
    cluster bootstrap."""
    rng = np.random.RandomState(seed * 22271 + 11)
    ue = np.unique(episodes)
    by_e = {e: delta[episodes == e] for e in ue}
    means = []
    for _ in range(N_BOOT):
        pick = rng.choice(ue, size=len(ue), replace=True)
        means.append(float(np.concatenate([by_e[e] for e in pick]).mean()))
    return float(np.percentile(means, 2.5))


# ── shared per-seed evaluation machinery ──────────────────────────────────

def _split(d: dict):
    eps = d["episode"]
    test_e = np.unique(eps)[-N_TEST_EP:]
    te = np.isin(eps, test_e)
    return ~te, te


def _alive_channels(Ttr: np.ndarray):
    rate = (Ttr > 1e-6).mean(0)
    std = Ttr.std(0)
    return np.where((rate >= 0.02) & (rate <= 0.98) & (std > 1e-4))[0]


def _prep(d: dict):
    """Standardised (by TRAIN) touch target on alive channels, plus the input
    blocks. Returns everything both _experiment and _control need."""
    tr, te = _split(d)
    alive = _alive_channels(d["touch"][tr])
    mu = d["touch"][tr][:, alive].mean(0)
    sd = np.where(d["touch"][tr][:, alive].std(0) < 1e-9, 1.0,
                  d["touch"][tr][:, alive].std(0))
    Y = ((d["touch"][:, alive] - mu) / sd).astype(np.float32)
    V, P = d["vision"], d["proprio"]
    zV = np.zeros_like(V)
    zP = np.zeros_like(P)
    return tr, te, alive, Y, V, P, zV, zP


def _r2_and_err(pred: np.ndarray, Ytest: np.ndarray):
    """Pooled R^2 vs the (standardised) train-mean = zero predictor, and the
    per-sample summed squared error for pairing."""
    err = ((pred - Ytest) ** 2).sum(1)
    sse = float(err.sum())
    sst = float((Ytest ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12), err


# ── the experiment ────────────────────────────────────────────────────────

def _experiment(seed: int) -> dict:
    d = _data_for(seed)
    w = get_world()
    tr, te, alive, Y, V, P, zV, zP = _prep(d)
    eps = d["episode"]

    fused, fell_f = _train_mlp(np.hstack([V[tr], P[tr]]), Y[tr], seed,
                               tag=1, groups=eps[tr])
    prop, fell_p = _train_mlp(np.hstack([zV[tr], P[tr]]), Y[tr], seed,
                              tag=2, groups=eps[tr])
    vis, fell_v = _train_mlp(np.hstack([V[tr], zP[tr]]), Y[tr], seed,
                             tag=3, groups=eps[tr])

    fused_r2, err_f = _r2_and_err(fused(np.hstack([V[te], P[te]])), Y[te])
    prop_r2, err_p = _r2_and_err(prop(np.hstack([zV[te], P[te]])), Y[te])
    vis_r2, _ = _r2_and_err(vis(np.hstack([V[te], zP[te]])), Y[te])

    boot = _boot_lo(err_p - err_f, eps[te], seed)

    # Non-gating diagnostic: proprio + root z (how much of the gain is height).
    Pz_tr = np.hstack([zV[tr], P[tr], d["root"][tr][:, 2:3]])
    Pz_te = np.hstack([zV[te], P[te], d["root"][te][:, 2:3]])
    propz, _ = _train_mlp(Pz_tr, Y[tr], seed, tag=4, groups=eps[tr])
    propz_r2, _ = _r2_and_err(propz(Pz_te), Y[te])

    # Fixture probe: the eye must locate the body (pooled frame -> root xy).
    vision_sees = _ridge_reg_cv(V, d["root"][:, :2], eps)

    canary_ok = float(w.canary() == w._canary_ref)

    m = {
        "fused_r2": round(fused_r2, 4),
        "proprio_r2": round(prop_r2, 4),
        "vision_only_r2": round(vis_r2, 4),
        "proprioz_r2": round(propz_r2, 4),
        "r2_gain": round(fused_r2 - prop_r2, 4),
        "boot_lo": round(boot, 5),
        "vision_sees_body": round(vision_sees, 4),
        "alive_channels": float(len(alive)),
        "dropped_frac": round(d["dropped_frac"], 4),
        "loss_fell_fused": round(fell_f, 4),
        "loss_fell_proprio": round(fell_p, 4),
        "n_test": int(te.sum()),
        "canary_ok": canary_ok,
    }
    # Per-seed indicators: run_spec records the seed MEAN, so 1.0 here is the
    # statement "no seed failed" — a mean of raw numbers is not.
    m["seed_claim_ok"] = float(m["boot_lo"] > 0.0)
    m["seed_alive_ok"] = float(
        m["vision_sees_body"] >= VISION_BODY_GATE
        and m["alive_channels"] >= ALIVE_MIN
        and m["dropped_frac"] <= DROP_MAX
        and m["fused_r2"] >= FLOOR_R2
        and m["loss_fell_fused"] <= LOSS_FELL
        and m["loss_fell_proprio"] <= LOSS_FELL
        and fell_v <= LOSS_FELL)
    return m


def _control(seed: int) -> dict:
    """Vision deranged across episodes: every sample keeps its own proprio and
    touch but reads the frame of a DIFFERENT episode at the same within-episode
    index (fixed nonzero episode rotation within each split; ragged tails wrap
    within the donor episode). Marginals preserved, correspondence destroyed.
    If this arm still buys the claim-sized gain, the gain never needed the
    frames to be THIS body's view — a leak, not cross-modal prediction."""
    d = _data_for(seed)
    tr, te, alive, Y, V, P, zV, zP = _prep(d)
    eps = d["episode"]

    def deranged(mask) -> np.ndarray:
        sub_eps = np.unique(eps[mask])
        shift = 1 + (seed % (len(sub_eps) - 1))
        donor = {e: sub_eps[(i + shift) % len(sub_eps)]
                 for i, e in enumerate(sub_eps)}
        rows = np.empty((int(mask.sum()), V.shape[1]), dtype=np.float32)
        r = 0
        for e in sub_eps:
            idx = np.where(eps == e)[0]
            didx = np.where(eps == donor[e])[0]
            take = didx[np.arange(len(idx)) % len(didx)]
            rows[r:r + len(idx)] = V[take]
            r += len(idx)
        return rows

    ctl, fell_c = _train_mlp(np.hstack([deranged(tr), P[tr]]), Y[tr],
                             seed, tag=5, groups=eps[tr])
    ctl_r2, err_c = _r2_and_err(ctl(np.hstack([deranged(te), P[te]])), Y[te])

    prop, _ = _train_mlp(np.hstack([zV[tr], P[tr]]), Y[tr], seed,
                        tag=2, groups=eps[tr])
    prop_r2, err_p = _r2_and_err(prop(np.hstack([zV[te], P[te]])), Y[te])

    boot = _boot_lo(err_p - err_c, eps[te], seed)
    gain = ctl_r2 - prop_r2
    return {
        "control_r2": round(ctl_r2, 4),
        "control_gain": round(gain, 4),
        "control_boot_lo": round(boot, 5),
        # must-fail: the deranged arm may NOT satisfy the claim pair.
        "control_fails_ok": float(not (boot > 0.0 and gain >= GAIN_MIN)),
        # aliveness: it still holds proprio, so it must not be a dead arm.
        "control_alive_ok": float(ctl_r2 >= prop_r2 - CONTROL_SLACK
                                  and fell_c <= LOSS_FELL),
    }


def _check(m: dict, c: dict):
    if m.get("canary_ok", 0.0) != 1.0:
        return Status.VOID      # the eye degraded mid-run: invalid, not false
    if m["seed_alive_ok"] != 1.0 or c["control_alive_ok"] != 1.0:
        return Status.VOID      # fixture/trainer dead or data-starved:
        # the run did not test the claim (VOID), it did not refute it
    return (m["seed_claim_ok"] == 1.0
            and m["r2_gain"] >= GAIN_MIN
            and c["control_fails_ok"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["UB.14"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
