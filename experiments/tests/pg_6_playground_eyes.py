"""PG.6 — The playground has eyes, and they resolve what the test needs.

HYPOTHESIS. An egocentric camera in the playground MJCF renders frames from
which a LINEAR probe recovers object RADIUS (R^2 >= 0.80) and BEARING (median
error <= 5 deg) for objects in FOV.

WHY IT MATTERS. Every visual claim downstream reads through this camera. UB.9
asks whether Jack binds a heard identity to a seen position; if the eye cannot
resolve position in the first place, UB.9 measures the eye, not the binding, and
returns a confident null about a capability it never tested. PG.6 is the
certificate that makes those specs interpretable. `kills`: any visual claim in
UB.9/UB.10 at this resolution.

THE PROBE IS DELIBERATELY LINEAR. A conv net could recover bearing from a
blurred smear; that would certify the net, not the sensor. A linear read-out on
raw pixels is the weakest reader that could work, so passing it is a statement
about the IMAGE. This is also why the probe never sees which pixels matter: no
crop, no attention, no object mask — the full frame, flattened.

RESOLUTION WAS CHOSEN BY PILOT, THE GATE WAS NOT. The spec's own text says "at
the chosen resolution", so choosing one is part of implementing it. The pilot ran
on seed 90 — disjoint from the registered seeds — and its numbers are recorded
here so the choice can be audited rather than trusted (radius R^2 at the best l2,
median bearing error):

    n_train=350    32px 0.624/6.40deg  48px 0.693/5.81  64px 0.722/5.03  96px 0.734/3.98
    n_train=700    96px 0.769/3.35     128px 0.775/3.27
    n_train=1200   96px 0.806/3.38     128px 0.808/3.17
    n_train=1800   96px 0.836/2.85   <- CHOSEN (l2=1.0)

The 1200-sample points clear R^2 >= 0.80 by 0.006. Registering there would have
been choosing an operating point that passes on the pilot seed and coin-flips on
the rest — tuning to the threshold, which is the same sin as moving it. 1800
samples buys 0.036 of real margin for about 25 seconds of extra rendering, and
128 px was rejected because it costs 78% more pixels for +0.002.

The thresholds R^2 >= 0.80 and 5 deg come from the registry and were not touched.
If no resolution had cleared them, the honest output is a FAIL plus the
escalation the registry already names — raise resolution, or move vision to a
frozen tower with cached embeddings — not a softer number.

WHAT THE CONTROL HAD TO CATCH. Objects OUTSIDE the FOV must be unrecoverable.
The naive way to write this control is to put the object behind the camera,
which makes every frame byte-identical and the control passes by construction —
it would certify a probe that reads nothing. Instead the out-of-FOV objects sit
at bearings 40-75 deg: off-frame, but still lit, still on the floor, and still
casting shadow into the arena. That is a control that can actually fail, and it
is the one that would catch the real leak (an object localised by its shadow
rather than by itself).

FIXTURE MUST-SUCCEED PROBE. PG.8's lesson — seven honest fixtures certifying an
empty room — applies directly: a vision test whose scene is blank would show
exactly the numbers a blind sensor shows. So the fixture asserts the object is
VISIBLE: for every in-FOV episode the frame must differ from the same scene with
the object removed, and the brightest-difference column must track the true
bearing. A fixture that cannot see its own object fails before the probe runs.
"""

from __future__ import annotations

import math

import numpy as np

# ensure_gl() must precede the mujoco import: mujoco binds its GL backend when
# first imported. This box has no libEGL and no libOSMesa, and rendering was
# escalated as impossible on 2026-08-09 for exactly that reason — GLX under
# Xvfb was never tried. See experiments/render.py.
from ..render import ensure_gl

ensure_gl()

import mujoco  # noqa: E402  (must follow ensure_gl)

import playground as pg  # noqa: E402

from ..protocol import Ledger, Status, run_spec  # noqa: E402
from ..registry import BY_ID  # noqa: E402

RES = 96                    # px, square. Chosen by pilot; see module docstring.
N_TRAIN, N_TEST = 1800, 300
L2 = 1.0                    # ridge strength, chosen on the pilot
DIST_RANGE = (2.2, 3.6)     # m from the eye — varied, so apparent size alone
                            # does not determine radius; the probe must use the
                            # ground-plane contact cue as well.
RADIUS_RANGE = (0.06, 0.18)  # matches PlaygroundParams.object_size_range
FOV_HALF = 30.0             # deg, from EYE_FOVY=60 on a square frame
IN_FOV_MAX = 22.0           # deg, margin so an object is wholly inside
OOF_RANGE = (40.0, 75.0)    # deg, the control band: off-frame but still lit

R2_GATE = 0.80              # registry
BEARING_GATE_DEG = 5.0      # registry
NULL_R2_CEIL = 0.20         # a null or control that beats this is a leak
NULL_BEARING_FLOOR = 20.0   # deg


# ── camera geometry ──────────────────────────────────────────────────────
def _eye_frame():
    """Return (origin, right, up, forward) of the playground eye, in world."""
    c = np.array(pg.EYE_POS, dtype=float)
    ax = np.array(pg.EYE_XYAXES[:3], dtype=float)
    up = np.array(pg.EYE_XYAXES[3:], dtype=float)
    ax /= np.linalg.norm(ax)
    up /= np.linalg.norm(up)
    back = np.cross(ax, up)          # camera +z points BACKWARD in MuJoCo
    fwd = -back / np.linalg.norm(back)
    return c, ax, up, fwd


def _place(bearing_deg: float, dist: float, radius: float):
    """World (x, y, z) for an object at `bearing` and `dist` from the eye.

    Bearing is measured in the world-horizontal plane about the eye, positive
    toward the camera's right, so it is the quantity the probe must recover.
    """
    c, right, _up, fwd = _eye_frame()
    f = np.array([fwd[0], fwd[1], 0.0])
    f /= np.linalg.norm(f)
    r = np.array([right[0], right[1], 0.0])
    r /= np.linalg.norm(r)
    th = math.radians(bearing_deg)
    p = c + (math.cos(th) * f + math.sin(th) * r) * dist
    return float(p[0]), float(p[1]), float(radius + 0.02)


def _true_bearing(pos) -> float:
    c, right, _up, fwd = _eye_frame()
    v = np.array(pos, dtype=float) - c
    v[2] = 0.0
    f = np.array([fwd[0], fwd[1], 0.0]); f /= np.linalg.norm(f)
    r = np.array([right[0], right[1], 0.0]); r /= np.linalg.norm(r)
    return math.degrees(math.atan2(float(v @ r), float(v @ f)))


# ── ridge, in numpy (no sklearn on this box) ─────────────────────────────
class _Ridge:
    """Ridge with the expensive part computed once.

    Each arm regresses six targets off the SAME frames (radius, sin/cos of
    bearing, and the shuffled-label null for each). The Gram matrix does not
    depend on the target, and at n=1800 with 27,648 pixel features it is ~9e10
    flops — six times more than the solve it feeds. Building it once per feature
    matrix instead of once per target is the difference between a twenty-minute
    spec and an hour-long one on four shared cores, which matters here because
    the box has paying tenants on it.
    """

    def __init__(self, Xtr, l2=L2):
        self.mu = Xtr.mean(0)
        self.A = Xtr - self.mu
        n, d = self.A.shape
        self.dual = d > n                   # dual form: cheaper, identical result
        if self.dual:
            self.M = self.A @ self.A.T + l2 * np.eye(n)
        else:
            self.M = self.A.T @ self.A + l2 * np.eye(d)

    def predict(self, ytr, Xte):
        ym = ytr.mean()
        yc = ytr - ym
        B = Xte - self.mu
        if self.dual:
            alpha = np.linalg.solve(self.M, yc)
            return B @ (self.A.T @ alpha) + ym
        w = np.linalg.solve(self.M, self.A.T @ yc)
        return B @ w + ym


def _ridge_predict(Xtr, ytr, Xte, l2=L2):
    return _Ridge(Xtr, l2).predict(ytr, Xte)


def _r2(y, pred) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _bearing_median_err(btr, bte, fit: "_Ridge", Xte) -> float:
    """Regress sin and cos separately, recombine with atan2.

    Regressing the angle directly would punish the probe for the wrap at +-180
    deg, which is an artifact of the coordinate and not of the image.
    """
    s = fit.predict(np.sin(np.radians(btr)), Xte)
    c = fit.predict(np.cos(np.radians(btr)), Xte)
    pred = np.degrees(np.arctan2(s, c))
    err = np.abs((pred - bte + 180.0) % 360.0 - 180.0)
    return float(np.median(err))


# ── the world, rendered ──────────────────────────────────────────────────
_EYES: dict = {}


def get_eye(seed: int, res: int = RES) -> "_Eye":
    """Cached, and cached for a reason that is not speed.

    Letting an `_Eye` fall out of scope frees its `mujoco.Renderer`, whose
    GLContext teardown poisons the shared X display: the NEXT renderer on the
    same display returns images that are corrupted but entirely plausible.
    Measured 2026-08-09 — a reference frame that renders 1334 distinct colours
    dropped to 200, then 719, then 736 across successive eyes, with no error
    raised and no exception. A spec reading those frames would have reported a
    real-looking, wrong number for the sharpness of Jack's vision.

    So eyes are created once per (seed, res) and held for the life of the
    process. Three renderers at 64 px is a few MB; a silently degraded sensor
    is a false certificate every downstream visual spec would inherit.
    """
    key = (seed, res)
    if key not in _EYES:
        _EYES[key] = _Eye(seed, res)
    return _EYES[key]


class _Eye:
    """One compiled playground plus its renderer.

    The object's radius is edited in `model.geom_size` and its pose in `qpos`,
    so a thousand episodes cost one MJCF compile and one GL context instead of
    a thousand of each. Everything else in the arena is held fixed: any frame
    difference is the probe object.
    """

    def __init__(self, seed: int, res: int = RES):
        params = pg.PlaygroundParams(seed=seed, n_objects=0)
        self.model, self.data, _ = pg.make_playground(
            params, with_water=False, probe_objects=(("probe0", 0.0, 0.0, 0.10),))
        self.gid = self.model.geom("probe0").id
        self.bid = self.model.body("probe0").id
        self.qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.bid]]
        self.r = mujoco.Renderer(self.model, height=res, width=res)
        self._canary = None
        self._canary = self.canary()

    def canary(self) -> float:
        """A fixed reference frame, reduced to one number.

        Rendered at construction and again after the arm finishes. If the GL
        context degrades mid-run the frames stay plausible but change, and this
        is the only thing that notices. A mismatch means the RUN was invalid,
        not that the hypothesis was refuted — so it returns VOID, per the
        Status.VOID lesson (an invalid run is not evidence against a claim).
        """
        f = self.frame(7.0, 3.0, 0.13)
        return float(np.round(f.sum(), 3))

    def frame(self, bearing_deg: float, dist: float, radius: float,
              hide: bool = False) -> np.ndarray:
        x, y, z = _place(bearing_deg, dist, radius)
        if hide:
            z = -50.0                    # far below the floor: same scene, no object
        self.model.geom_size[self.gid, 0] = radius
        q = self.qadr
        self.data.qpos[q:q + 3] = (x, y, z)
        self.data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.r.update_scene(self.data, camera="eye")
        return self.r.render().astype(np.float32) / 255.0

    def truth(self, bearing_deg: float, dist: float, radius: float):
        x, y, z = _place(bearing_deg, dist, radius)
        return _true_bearing((x, y, z))


def _episodes(eye: _Eye, rng, n: int, bearing_lo: float, bearing_hi: float,
              signed: bool = True):
    X, rad, bear = [], [], []
    for _ in range(n):
        b = rng.uniform(bearing_lo, bearing_hi)
        if signed and rng.rand() < 0.5:
            b = -b
        dist = rng.uniform(*DIST_RANGE)
        r = rng.uniform(*RADIUS_RANGE)
        X.append(eye.frame(b, dist, r).ravel())
        rad.append(r)
        bear.append(eye.truth(b, dist, r))
    return np.asarray(X), np.asarray(rad), np.asarray(bear)


def _visibility_check(eye: _Eye, rng, n: int = 40) -> dict:
    """The fixture must SEE its object. Fails before any probe is trained."""
    seen, col_err = 0, []
    for _ in range(n):
        b = rng.uniform(-IN_FOV_MAX, IN_FOV_MAX)
        dist = rng.uniform(*DIST_RANGE)
        r = rng.uniform(*RADIUS_RANGE)
        with_obj = eye.frame(b, dist, r)
        without = eye.frame(b, dist, r, hide=True)
        diff = np.abs(with_obj - without).sum(axis=2)
        if diff.max() > 0.05:
            seen += 1
            col = int(diff.sum(axis=0).argmax())
            # column -> bearing under a pinhole with half-angle FOV_HALF
            u = (col + 0.5) / diff.shape[1] * 2.0 - 1.0
            pred = math.degrees(math.atan(u * math.tan(math.radians(FOV_HALF))))
            col_err.append(abs(pred - b))
    return {"visible_frac": seen / n,
            "geometric_col_err_deg": float(np.median(col_err)) if col_err else 99.0}


def _arm(seed: int, bearing_lo: float, bearing_hi: float) -> dict:
    rng = np.random.RandomState(seed)
    eye = get_eye(seed)
    canary_in = eye._canary
    vis = _visibility_check(eye, rng)
    Xtr, rtr, btr = _episodes(eye, rng, N_TRAIN, bearing_lo, bearing_hi)
    Xte, rte, bte = _episodes(eye, rng, N_TEST, bearing_lo, bearing_hi)

    fit = _Ridge(Xtr)                      # the Gram matrix, built once
    rad_pred = fit.predict(rtr, Xte)
    out = {
        "radius_r2": round(_r2(rte, rad_pred), 4),
        "bearing_med_deg": round(_bearing_median_err(btr, bte, fit, Xte), 3),
        "visible_frac": vis["visible_frac"],
        "geometric_col_err_deg": round(vis["geometric_col_err_deg"], 3),
    }

    # NULL 1 — shuffled pairing. Same probe, same frames, labels permuted.
    perm = rng.permutation(len(rtr))
    out["radius_r2_shuffled"] = round(_r2(rte, fit.predict(rtr[perm], Xte)), 4)
    out["bearing_med_shuffled"] = round(
        _bearing_median_err(btr[perm], bte, fit, Xte), 3)

    # NULL 2 — constant grey frame. Any skill left here is label statistics.
    Gtr = np.full_like(Xtr, 0.5)
    Gte = np.full_like(Xte, 0.5)
    gfit = _Ridge(Gtr)
    out["radius_r2_grey"] = round(_r2(rte, gfit.predict(rtr, Gte)), 4)
    out["bearing_med_grey"] = round(_bearing_median_err(btr, bte, gfit, Gte), 3)
    out["canary_stable"] = float(abs(eye.canary() - canary_in) < 1e-6)
    return out


def _fixed_distance_diagnostic(eye: _Eye, rng) -> float:
    """Radius R^2 with distance HELD CONSTANT. Diagnostic, never gated.

    The registered arm varies distance over 2.2-3.6 m, so apparent size alone
    does not determine radius — the probe must also read the ground-plane
    contact cue, and a linear read-out combining two cues multiplicatively is
    doing something it structurally cannot do well. This number separates the
    two explanations for a marginal result, which are the two branches of the
    spec's own escalation:

      fixed >> varied  -> the SENSOR is fine; the linear probe cannot fuse size
                          with distance. Raising resolution will not help much;
                          the honest move is the frozen-tower/learned-encoder
                          branch the registry already names.
      fixed ~= varied  -> resolution-limited. Raise it.

    Reported either way, so a PASS also records how much of its margin came from
    the cue-fusion problem rather than from acuity.
    """
    n_tr, n_te, d0 = 350, 120, 2.9
    def batch(n):
        X, r = [], []
        for _ in range(n):
            b = rng.uniform(-IN_FOV_MAX, IN_FOV_MAX)
            rad = rng.uniform(*RADIUS_RANGE)
            X.append(eye.frame(b, d0, rad).ravel())
            r.append(rad)
        return np.asarray(X), np.asarray(r)
    Xtr, rtr = batch(n_tr)
    Xte, rte = batch(n_te)
    return round(_r2(rte, _ridge_predict(Xtr, rtr, Xte)), 4)


def _experiment(seed: int) -> dict:
    m = _arm(seed, 0.0, IN_FOV_MAX)
    m["radius_r2_fixed_dist"] = _fixed_distance_diagnostic(
        get_eye(seed), np.random.RandomState(seed + 7919))
    m["seed_gates_ok"] = float(
        m["radius_r2"] >= R2_GATE
        and m["bearing_med_deg"] <= BEARING_GATE_DEG
        and m["visible_frac"] == 1.0
        and m["geometric_col_err_deg"] <= 5.0
        and m["radius_r2_shuffled"] <= NULL_R2_CEIL
        and m["bearing_med_shuffled"] >= NULL_BEARING_FLOOR
        and m["radius_r2_grey"] <= NULL_R2_CEIL
        and m["bearing_med_grey"] >= NULL_BEARING_FLOOR)
    return m


def _control(seed: int) -> dict:
    """Out-of-FOV objects must be unrecoverable — the probe must not read
    episode identity, nor a shadow cast into frame by an object off-screen."""
    c = _arm(seed, *OOF_RANGE)
    return {
        "control_canary_stable": c["canary_stable"],
        "oof_radius_r2": c["radius_r2"],
        "oof_bearing_med_deg": c["bearing_med_deg"],
        "control_seed_gates_ok": float(
            c["radius_r2"] <= NULL_R2_CEIL
            and c["bearing_med_deg"] >= NULL_BEARING_FLOOR),
    }


def _check(m: dict, c: dict):
    if m.get("canary_stable", 0.0) != 1.0 or c.get("control_canary_stable", 0.0) != 1.0:
        # The renderer changed under us. Not a refutation — an invalid run.
        return Status.VOID
    return (m["seed_gates_ok"] == 1.0
            and m["radius_r2"] >= R2_GATE
            and m["bearing_med_deg"] <= BEARING_GATE_DEG
            and c["control_seed_gates_ok"] == 1.0
            and c["oof_radius_r2"] <= NULL_R2_CEIL
            and c["oof_bearing_med_deg"] >= NULL_BEARING_FLOOR)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["PG.6"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
