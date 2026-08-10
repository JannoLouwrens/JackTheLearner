"""TA.01 — The poison fixture: sub-lethal first dose, visually identical twin.

HYPOTHESIS (registry, unchanged). Two plant types are IDENTICAL to a visual
probe and DISTINCT to the taste vector; the toxic one produces a delayed,
SURVIVABLE interoceptive insult on a first small dose, following a declared
dose-response curve.

WHY IT MATTERS. `TA.02` is the one-trial conditioned-taste-aversion claim, and
`FROZEN_VS_PLASTIC.md` §8.4 believes it would be the first of its kind. Every
way it can be fake runs through this fixture:

  * if the plants differ visually, TA.02 is a colour-discrimination task with a
    slow reward — solvable by vision plus ordinary RL, and silent about taste;
  * if the first dose is lethal, there is no trial two to be averse in;
  * if the toxin cannot kill at any dose, "sub-lethal first dose" is a fact
    about the toxin rather than about the dose, and the neophobia story is
    decoration;
  * if the illness arrives instantly, the hours-long eligibility window that
    makes CTA a *privileged* channel is not being tested at all.

So TA.01 kills TA.02 (registry `kills`), and it is worth more care than a
fixture normally gets.

WHAT MAKES A NULL RESULT BELIEVABLE HERE — the hard part of this spec. The
headline number is a probe scoring CHANCE, and chance is what a broken probe
also scores. Three separate things defend it, and all three are gated:

  1. **The probe demonstrably has eyes.** The colour-coded control (the registry's
     declared control) recolours the toxic berries and nothing else — same
     seed, same draws, same geometry — and both probes must then classify well
     above chance. This is `PG.7`'s precedent applied exactly.
  2. **The probe demonstrably reads THESE plants, not the room.** Berry radius
     must be recoverable from the same frames by the same ridge (R^2 gate). A
     probe that could not read a plant's size has no standing to report that it
     cannot read a plant's type.
  3. **A second probe, aimed where a leak would actually be.** A linear read-out
     on 27,648 raw pixels is a weak reader, and a weak reader's null is weak
     evidence. So the plant is segmented against a fixed background frame and a
     k-NN classifier runs on the nine summary features a leak would live in —
     pixel count, mean RGB, bounding-box height and width, centroid, and colour
     spread. If type ever touches size, shade or placement, this catches it
     nonlinearly and cheaply. Both probes must sit inside the chance band.

The chance band is TWO-SIDED (|acc - 0.5| <= 3 sigma of a binomial at n_test).
A probe scoring 0.02 has found the type just as surely as one scoring 0.98; a
one-sided ceiling would have called perfect anti-correlation a pass.

PILOTED, LIKE PG.6, ON A SEED DISJOINT FROM THE REGISTERED ONES. Seed 90,
n_train = n_test = 400, 96 px (`PG.6`'s certified operating point — the eye is
certified there, and a lower resolution would make "cannot tell them apart"
easier to achieve for the wrong reason):

    real fixture   linear 0.470   kNN 0.512   kNN-shuffled 0.480   radius R^2 0.685
    colour-coded   linear 0.990   kNN 1.000                        radius R^2 0.652

The gates were then set with margin rather than at the pilot values: chance
band +-0.075 (3 sigma at n=400), control floor 0.90, radius R^2 floor 0.40.

THE RENDERER TRAP, MET IN THE PILOT AND WORTH RECORDING. Piloting the two arms
as `sc = Scene(90)` twice — the second binding freeing the first — produced a
control arm that rendered 800 frames in **one second** and reported linear
accuracy 1.000 with radius R^2 **-0.008**: a "control passes" that would have
been read as the probe having eyes, on frames that were garbage. That is the
failure `PG.6:get_eye` documents (a freed `mujoco.Renderer` poisons the shared
X display), reproduced here from scratch. Scenes are cached for the process
lifetime and every arm carries a canary frame; a canary that moves returns
`Status.VOID`, because that is an invalid run and not a refutation.

THE ILLNESS HALF IS ARITHMETIC, AND ITS SCOPE IS NARROW ON PURPOSE. It
integrates the declared curve against the world's OWN integrity dynamics —
`drives.RHO_HEAL` heals, `drives` clips to [0, 1], `w0.DEATH_FLOOR` is where a
life ends, all imported live rather than copied (T0.14/T0.22) — and asks
whether the fixture's numbers satisfy the properties the spec named. It does
NOT claim that Jack survives a poisoning while doing something else; that needs
a policy in a live world and belongs to TA.02. What it does claim is that the
world he will be tested in has a first dose he can live through, a full dose he
cannot, and a delay that is not a reward in disguise.

"FELT" IS MEASURED AGAINST THE WORLD, NOT AGAINST A GUESS. The registry says
the insult must be interoceptive; a gate of "integrity drops by at least X" is
a number nobody can defend. Instead the illness must move `drives.drive(h)` —
the same scalar the needs channel carries — by at least 3x what the ORDINARY
PASSAGE OF THE SAME TIME costs through basal energy drain. If sickness is not
several times louder than the clock, it is not an event.
"""

from __future__ import annotations

import numpy as np

# ensure_gl() must precede the mujoco import — see experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import playground as pg  # noqa: E402

from .. import drives, plants  # noqa: E402
from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402
from ..w0 import DEATH_FLOOR  # noqa: E402

# The claim is about the WORLD (the eye, the plants, the drive constants), not
# only about this file. PG.6 supplies the camera geometry and the ridge, so a
# change to either must mark this certificate stale rather than silently move
# its numbers.
IMPL_DEPS = ["playground.py", "experiments/plants.py", "experiments/drives.py",
             "experiments/tests/pg_6_playground_eyes.py"]

from .pg_6_playground_eyes import _Ridge, _eye_frame, _place, _r2  # noqa: E402

RES = 96                       # px — PG.6's certified operating point
N_TRAIN, N_TEST = 400, 400     # balanced: alternating types, 200 each
BEARING_BAND = (0.0, 22.0)     # deg — PG.6's in-FOV band
DIST_BAND = (2.0, 3.2)         # m from the eye
MASK_THRESH = 0.05             # per-pixel |frame - background| sum over channels
KNN_K = 5

# 3 sigma of a balanced binomial at n = N_TEST, two-sided.
CHANCE_BAND = 3.0 * float(np.sqrt(0.25 / N_TEST))     # 0.075
CODED_ACC_GATE = 0.90          # the control must be caught this decisively
RADIUS_R2_GATE = 0.40          # the probe must read plant geometry (pilot 0.685)
MIN_PLANT_PX = 10              # every plant must be visible in its own frame
TASTE_ACC_GATE = 0.95          # distinct to taste

# ── the illness gates ───────────────────────────────────────────────────
SURVIVAL_MARGIN = 0.5          # integrity floor after the first dose, over DEATH_FLOOR
FELT_RATIO_GATE = 3.0          # illness must move d(h) >= 3x the same time's basal drift
RECOVERY_GATE = 0.99           # integrity restored by rest, within the bout horizon
BOUT_HORIZON_S = 400.0
DECISION_S = 0.2               # W0's decision length; the resolution he lives at
CURVE_TOL = 1e-9               # the integrator must deliver the declared curve
DOSE_GRID = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00)


# ── the scene ───────────────────────────────────────────────────────────
_SCENES: dict = {}


def get_scene(seed: int, res: int = RES) -> "_Scene":
    """Cached for the process lifetime, and NOT for speed.

    See the module docstring: letting a `_Scene` be collected frees its
    `mujoco.Renderer`, whose teardown poisons the shared X display, and the next
    renderer returns plausible garbage with no error raised. This spec's pilot
    reproduced that and would have recorded a control that "passed" on frames
    with no plant in them.
    """
    key = (seed, res)
    if key not in _SCENES:
        _SCENES[key] = _Scene(seed, res)
    return _SCENES[key]


class _Scene:
    """One compiled playground with one plant in it, plus its renderer.

    The plant's stem height, berry radius, berry offset, colour and world pose
    are edited on the compiled model, so N plants cost one MJCF compile. The
    background frame — the same world with the plant dropped below the floor —
    is rendered once and reused, which is what makes the per-sample
    segmentation free.
    """

    def __init__(self, seed: int, res: int = RES):
        params = pg.PlaygroundParams(seed=seed, n_objects=0)
        xml = plants.with_plant(pg.build_mjcf(params))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.bid = self.model.body(plants.PLANT_BODY).id
        self.gs = self.model.geom(plants.PLANT_STEM).id
        self.gb = self.model.geom(plants.PLANT_BERRY).id
        self.r = mujoco.Renderer(self.model, height=res, width=res)
        self.bg = self.frame(_canary_draw(), hide=True)
        self._canary = None
        self._canary = self.canary()

    def canary(self) -> float:
        """A fixed frame reduced to one number, read before and after each arm.

        A GL context that degrades mid-run keeps returning plausible images. If
        this moves, the RUN was invalid — Status.VOID, not FAIL.
        """
        return float(np.round(self.frame(_canary_draw()).sum(), 3))

    def place(self, d: plants.PlantDraw, coded: bool = False,
              hide: bool = False):
        x, y, _z = _place(d.bearing_deg, d.dist_m, 0.0)
        self.model.body_pos[self.bid] = (x, y, -50.0 if hide else 0.0)
        self.model.geom_size[self.gs, 1] = d.stem_h / 2.0
        self.model.geom_pos[self.gs] = (0.0, 0.0, d.stem_h / 2.0)
        self.model.geom_size[self.gb, 0] = d.berry_r
        self.model.geom_pos[self.gb] = (d.off_x, d.off_y, d.stem_h)
        self.model.geom_rgba[self.gb] = plants.berry_rgba(d, coded)
        mujoco.mj_forward(self.model, self.data)
        return x, y

    def frame(self, d: plants.PlantDraw, coded: bool = False,
              hide: bool = False) -> np.ndarray:
        self.place(d, coded, hide)
        self.r.update_scene(self.data, camera="eye")
        return self.r.render().astype(np.float32) / 255.0

    def unoccluded(self, d: plants.PlantDraw) -> bool:
        """Does a ray from the eye reach this plant's berry cluster?

        Geometric, not photometric, for PG.6's reason: rejecting on the same
        pixels the fixture then asserts would make visibility true by
        construction. The ladder rails hide two vertical stripes of this band
        (PG.6 measured +-17.4 deg), so without this the probe would be handed
        frames with no plant in them.
        """
        x, y = self.place(d)
        origin = np.asarray(_eye_frame()[0], dtype=np.float64)
        target = np.array([x + d.off_x, y + d.off_y, d.stem_h], dtype=np.float64)
        vec = target - origin
        vec /= np.linalg.norm(vec)
        gid = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(self.model, self.data, origin, vec, None, 1, -1, gid)
        return int(gid[0]) == self.gb


def _canary_draw() -> plants.PlantDraw:
    """One fixed plant, independent of any run's rng. Deliberately a literal:
    the canary must not move when a sampler changes."""
    return plants.PlantDraw(type_index=1, stem_h=0.5, berry_r=0.15,
                            off_x=0.0, off_y=0.0, shade=0.0,
                            bearing_deg=5.0, dist_m=2.5)


# ── probes ──────────────────────────────────────────────────────────────
def _segment(frame: np.ndarray, bg: np.ndarray):
    """(n_pixels, feature vector) for the plant against the fixed background."""
    diff = np.abs(frame - bg).sum(axis=2)
    mask = diff > MASK_THRESH
    n = int(mask.sum())
    if n == 0:
        return 0, np.zeros(9, dtype=float)
    rows, cols = np.nonzero(mask)
    px = frame[mask]
    return n, np.array([
        float(n),
        float(px[:, 0].mean()), float(px[:, 1].mean()), float(px[:, 2].mean()),
        float(rows.max() - rows.min() + 1), float(cols.max() - cols.min() + 1),
        float(rows.mean()), float(cols.mean()), float(px.std()),
    ], dtype=float)


def _knn(Ftr: np.ndarray, ytr: np.ndarray, Fte: np.ndarray,
         k: int = KNN_K) -> np.ndarray:
    """Standardised-feature k-NN. Nonlinear, and aimed exactly where a type
    leak would live: size, colour, placement."""
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    A, B = (Ftr - mu) / sd, (Fte - mu) / sd
    dist = ((B[:, None, :] - A[None, :, :]) ** 2).sum(2)
    idx = np.argsort(dist, axis=1)[:, :k]
    return (ytr[idx].mean(1) > 0.5).astype(float)


def _acc(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean())


def _batch(scene: _Scene, rng, n: int, coded: bool):
    """`n` plants, alternating type so the classes are exactly balanced.

    Balance is not cosmetic: an unbalanced test set moves the chance rate away
    from 0.5, and every gate in this file is written against 0.5.
    """
    X, y, rad, F, npx = [], [], [], [], []
    for k in range(n):
        for attempt in range(200):
            d = plants.draw_plant(rng, k % 2, BEARING_BAND, DIST_BAND)
            if scene.unoccluded(d):
                break
        else:
            raise RuntimeError("no unoccluded plant found in 200 draws — the "
                               "eye's view of this band is blocked")
        f = scene.frame(d, coded)
        cnt, feats = _segment(f, scene.bg)
        X.append(f.ravel())
        y.append(float(d.type_index))
        rad.append(d.berry_r)
        F.append(feats)
        npx.append(cnt)
    return (np.asarray(X), np.asarray(y), np.asarray(rad),
            np.asarray(F), np.asarray(npx))


def _vision_arm(seed: int, coded: bool) -> dict:
    """Both probes, plus the two fixture checks, on one arm."""
    scene = get_scene(seed)
    canary_in = scene._canary
    # THE SAME SEED FOR BOTH ARMS, deliberately. The control must differ from
    # the fixture in exactly one thing — berry hue — so the same rng stream
    # draws the same stem heights, radii, offsets, shades, bearings and
    # distances in both. A control with its own draws would leave "geometry" as
    # a second explanation for its accuracy.
    rng = np.random.RandomState(seed)
    Xtr, ytr, rtr, Ftr, ntr = _batch(scene, rng, N_TRAIN, coded)
    Xte, yte, rte, Fte, nte = _batch(scene, rng, N_TEST, coded)

    fit = _Ridge(Xtr)
    lin = _acc((fit.predict(ytr * 2.0 - 1.0, Xte) > 0).astype(float), yte)
    perm = rng.permutation(N_TRAIN)
    lin_shuf = _acc((fit.predict(ytr[perm] * 2.0 - 1.0, Xte) > 0).astype(float), yte)
    knn = _acc(_knn(Ftr, ytr, Fte), yte)
    knn_shuf = _acc(_knn(Ftr, ytr[perm], Fte), yte)
    radius_r2 = _r2(rte, fit.predict(rtr, Xte))
    return {
        "linear_acc": round(lin, 4),
        "linear_acc_shuffled": round(lin_shuf, 4),
        "knn_acc": round(knn, 4),
        "knn_acc_shuffled": round(knn_shuf, 4),
        "radius_r2": round(radius_r2, 4),
        "min_plant_px": float(min(int(ntr.min()), int(nte.min()))),
        "median_plant_px": float(np.median(np.concatenate([ntr, nte]))),
        "canary_stable": float(abs(scene.canary() - canary_in) < 1e-6),
    }


def _taste_arm(seed: int) -> dict:
    """Distinct to taste — a linear probe on the 5-vector, same estimator.

    Not a lookup: each plant's taste is its type mean plus `TASTE_SIGMA` noise,
    so this is a real (if easy) discrimination.

    THE NULL IS A PLACEBO CHANNEL, NOT SHUFFLED LABELS, and the reason is a
    measurement. Shuffling the labels was tried first, by symmetry with the
    pixel probes above, and it is NOT calibrated here: over seeds 0-5 it read
    0.2725, 0.3775, 0.725, 0.5875, 0.285, 0.6725 — decisive, with a sign that
    flips per seed. Five dimensions with one strongly separating axis leave the
    permuted fit with a small residual weight along that axis; applied to a
    test set that is bimodal along it, a tiny weight classifies everything, and
    which way is a coin flip. (The pixel probes do not have this problem:
    27,648 dimensions against 400 samples regularise to the mean, and there is
    no type-aligned direction for a residual to land on. The seed mean over
    0/1/2 was 0.458 — inside the band, which is how a broken null hides.)

    The placebo is `FROZEN_VS_PLASTIC.md` §8.4's own third control — a
    matched-dimension, matched-statistics channel with zero information — and
    it measures 0.47-0.525 over the same six seeds. It is a null a wrong
    evaluator would fail, which is the whole job.
    """
    rng = np.random.RandomState(seed + 104729)

    def draw(n):
        y = np.arange(n) % 2
        T = np.stack([plants.TYPES[int(t)].taste(rng) for t in y])
        return T, y.astype(float)

    def placebo(n):
        mu = np.mean([t.taste_mu for t in plants.TYPES], axis=0)
        y = np.arange(n) % 2
        T = np.clip(mu + rng.normal(0.0, plants.TASTE_SIGMA,
                                    (n, plants.TASTE_DIM)), 0.0, 1.0)
        return T, y.astype(float)

    Ttr, ytr = draw(N_TRAIN)
    Tte, yte = draw(N_TEST)
    fit = _Ridge(Ttr, l2=1e-3)
    acc = _acc((fit.predict(ytr * 2.0 - 1.0, Tte) > 0).astype(float), yte)

    Ptr, pytr = placebo(N_TRAIN)
    Pte, pyte = placebo(N_TEST)
    pfit = _Ridge(Ptr, l2=1e-3)
    pacc = _acc((pfit.predict(pytr * 2.0 - 1.0, Pte) > 0).astype(float), pyte)
    return {"taste_acc": round(acc, 4), "taste_acc_placebo": round(pacc, 4)}


# ── the illness ─────────────────────────────────────────────────────────
def _felt_ratio(delta_i: float, elapsed_s: float) -> float:
    """How much louder the illness is than the clock, in d(h).

    Numerator: the move in `drives.drive` caused by the integrity loss.
    Denominator: the move caused by basal energy drain over the SAME elapsed
    time — the cheapest thing that happens to an interoceptive state when
    nothing happens at all.
    """
    ill = drives.drive(1.0, 1.0 - delta_i, 0.0) - drives.drive(1.0, 1.0, 0.0)
    de = min(1.0, drives.BASAL_B * elapsed_s)
    clock = drives.drive(1.0 - de, 1.0, 0.0) - drives.drive(1.0, 1.0, 0.0)
    return float(ill / clock) if clock > 0 else float("inf")


def _illness(seed: int) -> dict:
    """The dose-response half. Deterministic — reported per seed unchanged, so
    a seed-dependent number here would itself be a defect."""
    first = plants.ingest_bout(plants.TOXIC, plants.Q_FIRST, dt=DECISION_S,
                               horizon_s=BOUT_HORIZON_S, death_floor=DEATH_FLOOR)
    full = plants.ingest_bout(plants.TOXIC, 1.0, dt=DECISION_S,
                              horizon_s=BOUT_HORIZON_S, death_floor=DEATH_FLOOR)
    safe = plants.ingest_bout(plants.SAFE, 1.0, dt=DECISION_S,
                              horizon_s=BOUT_HORIZON_S, death_floor=DEATH_FLOOR)

    # Nothing may happen before the delay elapses: integrity may only heal (and
    # it starts clipped at 1.0, so it may not move at all).
    pre = first.i[first.t < plants.DELAY_S]
    quiet_before_onset = float(pre.min() >= 1.0 - 1e-12)

    # The curve, as DELIVERED by the scheduler. Measured on the toxin's own
    # rate rather than on integrity, because integrity is clipped to [0, 1]:
    # at q = 1 the declared total exceeds the whole integrity range (that is
    # what makes a full plant lethal), so reading the curve off the body would
    # report a ceiling and call it a deviation. This is the check that catches
    # a ramp that clips, a dt that steps over the window, or an event that
    # fires twice.
    dev, losses = 0.0, []
    for q in DOSE_GRID:
        tox = plants.Toxin()
        declared = tox.ingest(0.0, plants.TOXIC, q)
        grid = np.arange(0.0, plants.DELAY_S + plants.ILL_WINDOW_S + 10.0,
                         DECISION_S)
        delivered = float(sum(tox.rate(t) for t in grid) * DECISION_S)
        losses.append(delivered)
        dev = max(dev, abs(delivered - declared),
                  abs(declared - plants.illness_total(q, plants.TOXIC.potency)))
    monotone = float(all(b > a + 1e-9 for a, b in zip(losses, losses[1:])))

    delta_first = 1.0 - first.i_min
    m = {
        "first_dose": plants.Q_FIRST,
        "first_dose_delta_i": round(delta_first, 5),
        "first_dose_i_min": round(first.i_min, 5),
        "first_dose_survived": float(not first.died),
        "first_dose_recovered_i": round(first.i_end, 5),
        "onset_delay_s": round(first.onset_t, 3),
        "quiet_before_onset": quiet_before_onset,
        "felt_vs_clock_ratio": round(
            _felt_ratio(delta_first, plants.DELAY_S + plants.ILL_WINDOW_S), 3),
        "full_dose_lethal": float(full.died),
        "safe_twin_delta_i": round(1.0 - safe.i_min, 6),
        "safe_twin_survived": float(not safe.died),
        "curve_max_deviation": float(dev),
        "dose_response_monotone": monotone,
        "delay_s": plants.DELAY_S,
        "delay_frac_of_starvation": round(plants.delay_fraction(), 5),
    }
    m["illness_gates_ok"] = float(
        m["first_dose_survived"] == 1.0
        and m["first_dose_i_min"] >= SURVIVAL_MARGIN
        and m["first_dose_i_min"] > DEATH_FLOOR
        and m["felt_vs_clock_ratio"] >= FELT_RATIO_GATE
        and m["quiet_before_onset"] == 1.0
        and abs(m["onset_delay_s"] - plants.DELAY_S) <= DECISION_S
        and m["first_dose_recovered_i"] >= RECOVERY_GATE
        and m["full_dose_lethal"] == 1.0
        and m["safe_twin_delta_i"] == 0.0
        and m["safe_twin_survived"] == 1.0
        and m["curve_max_deviation"] <= CURVE_TOL
        and m["dose_response_monotone"] == 1.0
        and plants.DELAY_FRAC_BAND[0] <= m["delay_frac_of_starvation"]
        <= plants.DELAY_FRAC_BAND[1])
    return m


def _at_chance(acc: float) -> bool:
    """Two-sided: an anti-correlated probe has found the type just as surely."""
    return abs(acc - 0.5) <= CHANCE_BAND


def _experiment(seed: int) -> dict:
    m = _vision_arm(seed, coded=False)
    m.update(_taste_arm(seed))
    m.update(_illness(seed))
    m["seed_gates_ok"] = float(
        # invisible to both probes
        _at_chance(m["linear_acc"])
        and _at_chance(m["knn_acc"])
        # the nulls calibrate the band: if these leave it, the evaluator is wrong
        and _at_chance(m["linear_acc_shuffled"])
        and _at_chance(m["knn_acc_shuffled"])
        # ...but the probe DID read the plant
        and m["radius_r2"] >= RADIUS_R2_GATE
        # ...and every plant was in its own frame
        and m["min_plant_px"] >= MIN_PLANT_PX
        # distinct to taste
        and m["taste_acc"] >= TASTE_ACC_GATE
        and _at_chance(m["taste_acc_placebo"])
        # and the toxin behaves as declared
        and m["illness_gates_ok"] == 1.0)
    return m


def _control(seed: int) -> dict:
    """The colour-coded variant: same seed, same draws, same geometry — the
    toxic berries are red. Both probes must catch it, or their null on the real
    fixture is worthless (the registry's declared control; PG.7's precedent)."""
    c = _vision_arm(seed, coded=True)
    return {
        "coded_linear_acc": c["linear_acc"],
        "coded_knn_acc": c["knn_acc"],
        "coded_radius_r2": c["radius_r2"],
        "coded_min_plant_px": c["min_plant_px"],
        "control_canary_stable": c["canary_stable"],
        "control_seed_gates_ok": float(
            c["linear_acc"] >= CODED_ACC_GATE
            and c["knn_acc"] >= CODED_ACC_GATE
            and c["radius_r2"] >= RADIUS_R2_GATE),
    }


def _check(m: dict, c: dict):
    if m.get("canary_stable", 0.0) != 1.0 or c.get("control_canary_stable", 0.0) != 1.0:
        # The renderer changed under us: an invalid run, not a refutation.
        return Status.VOID
    return (m["seed_gates_ok"] == 1.0
            and _at_chance(m["linear_acc"])
            and _at_chance(m["knn_acc"])
            and m["radius_r2"] >= RADIUS_R2_GATE
            and m["taste_acc"] >= TASTE_ACC_GATE
            and m["illness_gates_ok"] == 1.0
            and c["control_seed_gates_ok"] == 1.0
            and c["coded_linear_acc"] >= CODED_ACC_GATE
            and c["coded_knn_acc"] >= CODED_ACC_GATE)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["TA.01"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
