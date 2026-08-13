"""T2.03 — Pretrained vision features beat random features.

HYPOTHESIS (registry). A linear probe on frozen DINOv2/SigLIP features beats
the same probe on the current 0.24M from-scratch encoder. Falsified by:
from-scratch matches pretrained. Null: random-projection features of equal
dimension. Metric: probe_accuracy.

WHAT THIS MEASURES, SAID PLAINLY. The vision seat is held by a 0.24M conv
encoder that has NEVER received a gradient (CHAMPIONS.md: "DEFAULT, never
defended") — so "the current from-scratch encoder" is, honestly, a structured
random projection. This spec measures the gap between that seat-holder and a
frozen pretrained yardstick on a recognition task in Jack's OWN world, through
Jack's OWN certified eye (PG.6 camera contract: EYE_POS/EYE_XYAXES/EYE_FOVY).

PLASTIC-ONLY CAUTION (Review 2026-08-11, decree 2026-08-09). A pretrained WIN
cannot seat frozen DINOv2/SigLIP inside Jack — frozen components inside him
are constitutionally barred. A PASS here is a MEASUREMENT: it quantifies the
representation gap the plastic path must close (PL.02's question), and it
kills the complacent reading that the never-trained encoder is already fine.
The registry `kills` field ("use_pretrained_vision=False") is therefore an
escalation to the owner/champions ledger, not a config flip this spec performs.

THE TASK. Four-way shape classification (sphere / box / cylinder / capsule)
from single eye frames. One probe body in an otherwise-fixed empty playground
(n_objects=0, no water); per episode its geom TYPE, size, orientation, bearing
(±22°, PG.6's certified band), and distance (2.2–3.6 m) are drawn, occlusion
is rejected by the same centre-ray rule PG.6 registered, and the frame is
rendered at 224 px (DINOv2's native input). All four classes share ONE geom and
ONE rgba, so colour cannot carry identity; scale is drawn iid per episode, so
apparent size is a (weak, honest) shape cue only through its interaction with
outline. Chance = 1/4.

THE ARMS — every one probed by the SAME procedure (z-score on train stats,
one-hot ridge, per-arm l2 chosen on a deterministic train-internal split,
argmax; the probe never sees test labels):

  pretrained   frozen DINOv2-large CLS ++ CLIP-ViT-L vision pooler (2048-d),
               loaded by the SHIPPED `PrismaticVisionEncoder` with
               use_pretrained_vision=True — the class whose pretrained path
               once crashed on its first forward and had therefore never run
               (UnifiedBrain.py's own comment). Exercising the shipped loader
               is part of the point (LESSONS: instantiating a module is not
               exercising it). Inputs get each trunk's published normalisation;
               the shipped forward's raw /255 is measured separately below.
  scratch      the seat holder exactly as it exists: `PrismaticVisionEncoder`
               with default config (CNN + projector, ~245K params, random init
               seeded per spec seed, .eval()). 1024-d output.
  rp2048/rp1024  the registry null: seeded Gaussian random projection of raw
               pixels to each arm's dimension.
  pixels       raw-pixel ridge — the reference arm simple enough that its
               failure indicts the task (T1.02 lesson), and the measure of how
               much of the task is linearly solvable with no features at all.
  pre_shipped  DIAGNOSTIC, not gated: the pretrained trunks fed exactly what
               the shipped forward feeds (raw /255, no normalisation). The gap
               pretrained − pre_shipped prices a wrapper defect, not the
               feature families the hypothesis compares.

CONTROL (registry, must fail): shuffled (frame, label) pairing on the BEST
family must collapse to chance. Anchored to the exogenous 0.25, not to any
pilot number (T2.08 lesson).

RIG GATES → VOID, not FAIL (an invalid run is not evidence): renderer canary
byte-stable across the run and the canary frame non-uniform (>=100 distinct
colours — a GL context can come up rendering a uniform frame and look exactly
like a blind sensor); scratch param count in [220K, 270K] (the spec names
0.24M; a different number means the seat holder changed under the test);
per-seed class balance exact by construction, asserted anyway.

PRE-REGISTERED GATES (pilot seed 90, disjoint from registered seeds 0/1/2;
pilot numbers recorded in LOOP_JOURNAL.md before the registered run; anchors
relative or exogenous, never pilot-bulk absolutes — T2.08 lesson):
  CLAIM   margin_min  = min over seeds of (acc_pretrained − acc_scratch)
                        >= MARGIN_FLOOR, and margin > 0 on EVERY seed
          mean (acc_pretrained − acc_rp2048) >= NULL_MARGIN — beating a
                        dimension-matched projection of the same pixels is
                        what "features" means
  CONTROL |acc_shuffled − 0.25| <= SHUFFLE_BAND
PILOT PENDING — the MARGIN_FLOOR / NULL_MARGIN constants below are
PROVISIONAL until the seed-90 pilot has actually run; its numbers will be
recorded here and in LOOP_JOURNAL.md, and the constants finalised in a
commit BEFORE the registered run. SHUFFLE_BAND is exogenous (chance ± ~4σ of
binomial noise at n_test=300) and is final now.

GPU. One submission for the whole spec (module cache — the T2.01 pattern;
run_spec calls _experiment once per seed). The job clones the pushed repo,
sets MUJOCO_GL=egl (the preamble's "disabled" is for training jobs that never
render), pip-installs mujoco, and imports THIS module — science code lives
here where guards can see it, never in the JOB string (T0.16 lesson).

COVERS: sight (claim).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..gpu import build_job, submit

# The claim is about the world's camera, the shipped encoder, and the PG.6
# geometry this file reuses — all three hash into the certificate.
IMPL_DEPS = ["playground.py", "UnifiedBrain.py",
             "experiments/tests/pg_6_playground_eyes.py"]

SEEDS = [0, 1, 2]
PILOT_SEED = 90

RES = 224                      # DINOv2's native input; rendered directly
N_TRAIN, N_TEST = 1200, 300    # balanced: n/4 per class, exact
CLASSES = ("sphere", "box", "cylinder", "capsule")
CHANCE = 1.0 / len(CLASSES)
DIST_RANGE = (2.2, 3.6)        # PG.6's certified distances
SIZE_RANGE = (0.06, 0.18)      # PG.6's radius range, reused as the scale draw
IN_FOV_MAX = 22.0              # deg, PG.6's certified bearing band
L2_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
VAL_EVERY = 5                  # every 5th train row is the l2-selection split

# Rig gates (VOID):
MIN_CANARY_COLORS = 100
PARAMS_RANGE = (220_000, 270_000)

# Pre-registered claim gates — set from the pilot per the docstring, committed
# before the registered run, then never moved.
MARGIN_FLOOR = 0.10            # min over seeds of (pretrained − scratch)
NULL_MARGIN = 0.15             # mean (pretrained − rp2048)
SHUFFLE_BAND = 0.10            # |shuffled − 0.25|


# ── the world, rendered (runs wherever the job runs) ─────────────────────
_EYES: dict = {}


def _get_eye(seed: int):
    """One compiled empty playground + renderer per seed, held for the process
    lifetime (a garbage-collected mujoco.Renderer poisons the shared GL
    context — PG.6's measured lesson)."""
    if seed not in _EYES:
        _EYES[seed] = _ShapeEye(seed)
    return _EYES[seed]


class _ShapeEye:
    """PG.6's certified eye, pointed at a shape-shifting probe body.

    The probe's geom TYPE is edited in the compiled model per episode —
    `model.geom_type` is writable, and editing beats rebuilding: one MJCF
    compile and one GL context for all 1500 episodes, and zero changes to
    playground.py (so no PG certificate goes stale under this spec).
    `geom_rbound` is kept consistent by hand because mj_forward does not
    recompute it, and mj_ray prunes on it.
    """

    def __init__(self, seed: int):
        # pg_6 FIRST: its module-level ensure_gl() must precede any mujoco
        # import in this process (it raises otherwise; on a GPU VM with
        # MUJOCO_GL=egl it is a no-op and order stops mattering).
        from . import pg_6_playground_eyes as P6   # geometry: reference, don't copy
        import mujoco
        import playground as pg
        self._mujoco, self._pg, self._P6 = mujoco, pg, P6
        params = pg.PlaygroundParams(seed=seed, n_objects=0)
        self.model, self.data, _ = pg.make_playground(
            params, with_water=False, probe_objects=(("probe0", 0.0, 0.0, 0.10),))
        self.gid = self.model.geom("probe0").id
        self.bid = self.model.body("probe0").id
        self.qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.bid]]
        self.r = mujoco.Renderer(self.model, height=RES, width=RES)
        self._canary0 = None
        self._canary0 = self.canary()

    # -- canary ----------------------------------------------------------
    def canary(self) -> float:
        f = self._frame_raw(7.0, 3.0, "sphere", (0.13, 0.0, 0.0),
                            (1.0, 0.0, 0.0, 0.0), 0.13)
        return float(np.round(f.astype(np.float64).sum(), 3))

    def canary_colors(self) -> int:
        f = self._frame_raw(7.0, 3.0, "sphere", (0.13, 0.0, 0.0),
                            (1.0, 0.0, 0.0, 0.0), 0.13)
        return len(np.unique(f.reshape(-1, 3), axis=0))

    # -- shape editing ---------------------------------------------------
    def _set_shape(self, cls: str, size3, rbound: float):
        m, mj = self.model, self._mujoco
        m.geom_type[self.gid] = {
            "sphere": mj.mjtGeom.mjGEOM_SPHERE, "box": mj.mjtGeom.mjGEOM_BOX,
            "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
            "capsule": mj.mjtGeom.mjGEOM_CAPSULE}[cls]
        m.geom_size[self.gid] = size3
        m.geom_rbound[self.gid] = rbound

    def _place(self, bearing_deg: float, dist: float, rbound: float, quat):
        # PG.6's _place computes world xy from (bearing, dist) and uses its
        # third argument only for z = arg + 0.02; passing rbound keeps every
        # orientation of every shape clear of the floor.
        x, y, z = self._P6._place(bearing_deg, dist, rbound)
        q = self.qadr
        self.data.qpos[q:q + 3] = (x, y, z)
        self.data.qpos[q + 3:q + 7] = quat
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        return x, y, z

    def _frame_raw(self, bearing_deg, dist, cls, size3, quat, rbound):
        self._set_shape(cls, size3, rbound)
        self._place(bearing_deg, dist, rbound, quat)
        self.r.update_scene(self.data, camera="eye")
        return self.r.render()                      # uint8 (RES, RES, 3)

    def unoccluded(self, bearing_deg, dist, cls, size3, quat, rbound) -> bool:
        """PG.6's registered visibility rule: ONE centre ray, so partially
        occluded episodes stay in (they are the hard ones)."""
        self._set_shape(cls, size3, rbound)
        x, y, z = self._place(bearing_deg, dist, rbound, quat)
        origin = np.asarray(self._P6._eye_frame()[0], dtype=np.float64)
        vec = np.array([x, y, z], dtype=np.float64) - origin
        vec /= np.linalg.norm(vec)
        gid = np.zeros(1, dtype=np.int32)
        self._mujoco.mj_ray(self.model, self.data, origin, vec, None, 1, -1, gid)
        return int(gid[0]) == self.gid


def _draw_shape(cls: str, rng):
    """(size3, rbound) for one episode. Scale s is the same draw for every
    class; aspect draws give box/cylinder/capsule real outline variety."""
    s = rng.uniform(*SIZE_RANGE)
    if cls == "sphere":
        return (s, 0.0, 0.0), s
    if cls == "box":
        hx, hy, hz = s, s * rng.uniform(0.6, 1.4), s * rng.uniform(0.6, 1.4)
        return (hx, hy, hz), math.sqrt(hx * hx + hy * hy + hz * hz)
    if cls == "cylinder":
        h = s * rng.uniform(0.8, 2.0)
        return (s, h, 0.0), math.sqrt(s * s + h * h)
    h = s * rng.uniform(0.8, 2.0)                   # capsule
    r = s * rng.uniform(0.6, 1.0)
    return (r, h, 0.0), r + h


def _rand_quat(rng):
    q = rng.randn(4)
    return tuple(q / np.linalg.norm(q))


def _build_dataset(seed: int, n: int):
    """n frames, exactly n/len(CLASSES) per class, interleaved. Returns
    (X uint8 [n,RES,RES,3], y int [n], mean rejection tries)."""
    assert n % len(CLASSES) == 0
    eye = _get_eye(seed)
    rng = np.random.RandomState(seed * 100_003 + n)
    X = np.empty((n, RES, RES, 3), dtype=np.uint8)
    y = np.empty(n, dtype=np.int64)
    tries_total = 0
    for i in range(n):
        cls_i = i % len(CLASSES)
        cls = CLASSES[cls_i]
        for k in range(200):
            b = rng.uniform(0.0, IN_FOV_MAX) * (1 if rng.rand() < 0.5 else -1)
            d = rng.uniform(*DIST_RANGE)
            size3, rbound = _draw_shape(cls, rng)
            quat = _rand_quat(rng)
            if eye.unoccluded(b, d, cls, size3, quat, rbound):
                break
        else:
            raise RuntimeError(f"no unoccluded {cls} in 200 draws (seed {seed})")
        tries_total += k + 1
        X[i] = eye._frame_raw(b, d, cls, size3, quat, rbound)
        y[i] = cls_i
    return X, y, tries_total / n


# ── features ─────────────────────────────────────────────────────────────
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _feature_arms(seed: int, X: np.ndarray, device: str) -> dict:
    """Every arm's features off the SAME frames. Returns name -> float32 [n,d],
    plus 'n_params_scratch'."""
    import torch
    from UnifiedBrain import UnifiedBrainConfig, PrismaticVisionEncoder

    n = len(X)
    imgs = torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)

    def batched(fn, tensor, bs=16):
        outs = []
        with torch.no_grad():
            for i in range(0, n, bs):
                outs.append(fn(tensor[i:i + bs].to(device)).float().cpu())
        return torch.cat(outs).numpy().astype(np.float32)

    out = {}

    # scratch: the seat holder exactly as shipped, seeded per spec seed.
    torch.manual_seed(seed)
    enc = PrismaticVisionEncoder(UnifiedBrainConfig()).to(device).eval()
    out["n_params_scratch"] = int(sum(p.numel() for p in enc.parameters()))
    out["scratch"] = batched(lambda t: enc(t), imgs)

    # pretrained trunks, via the shipped loader.
    pre = PrismaticVisionEncoder(
        UnifiedBrainConfig(use_pretrained_vision=True)).to(device).eval()

    def norm(t, mean, std):
        m = torch.tensor(mean, device=t.device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=t.device).view(1, 3, 1, 1)
        return (t - m) / s

    def trunks(t, normalise):
        td = norm(t, _DINO_MEAN, _DINO_STD) if normalise else t
        tc = norm(t, _CLIP_MEAN, _CLIP_STD) if normalise else t
        d = pre.dinov2(pixel_values=td).last_hidden_state[:, 0]
        c = pre.siglip.vision_model(pixel_values=tc).pooler_output
        return torch.cat([d, c], dim=-1)

    out["pretrained"] = batched(lambda t: trunks(t, True), imgs)
    out["pre_shipped"] = batched(lambda t: trunks(t, False), imgs)

    # nulls: seeded Gaussian projection of raw pixels to each arm's dimension.
    flat_dim = RES * RES * 3
    for name, dim in (("rp2048", 2048), ("rp1024", 1024)):
        g = torch.Generator().manual_seed(seed * 7 + dim)
        G = (torch.randn(flat_dim, dim, generator=g) / math.sqrt(flat_dim)).to(device)
        out[name] = batched(lambda t, G=G: t.reshape(len(t), -1) @ G, imgs)

    out["pixels"] = imgs.reshape(n, -1).numpy().astype(np.float32)
    return out


# ── the probe: one-hot ridge, Gram cached across the l2 grid ─────────────
class _Probe:
    """pg_6._Ridge's dual/primal ridge, extended two ways this task needs:
    multi-target one-hot with argmax, and ONE Gram shared across the l2 grid
    (the pixel family's Gram is 2e11 flops; rebuilding it per l2 would turn a
    minutes-long job into an hour)."""

    def __init__(self, Xtr: np.ndarray):
        self.mu = Xtr.mean(0)
        sd = Xtr.std(0)
        sd[sd < 1e-6] = 1e-6
        self.sd = sd
        self.A = ((Xtr - self.mu) / self.sd).astype(np.float32)
        n, d = self.A.shape
        self.dual = d > n
        self.G = (self.A @ self.A.T if self.dual else self.A.T @ self.A)

    def predict_classes(self, ytr: np.ndarray, Xte: np.ndarray, l2: float,
                        n_classes: int) -> np.ndarray:
        Y = np.eye(n_classes, dtype=np.float64)[ytr]
        ym = Y.mean(0)
        Yc = Y - ym
        B = ((Xte - self.mu) / self.sd).astype(np.float32)
        M = self.G.astype(np.float64) + l2 * np.eye(len(self.G))
        if self.dual:
            alpha = np.linalg.solve(M, Yc)
            scores = B @ (self.A.T @ alpha) + ym
        else:
            W = np.linalg.solve(M, self.A.T @ Yc)
            scores = B @ W + ym
        return scores.argmax(1)


def _probe_acc(Xtr, ytr, Xte, yte) -> tuple:
    """(test accuracy, chosen l2). l2 picked on a deterministic every-5th-row
    split of TRAIN; test labels are never consulted."""
    val = np.arange(len(Xtr)) % VAL_EVERY == 0
    fit = ~val
    inner = _Probe(Xtr[fit])
    best_l2, best_acc = None, -1.0
    for l2 in L2_GRID:
        pred = inner.predict_classes(ytr[fit], Xtr[val], l2, len(CLASSES))
        acc = float((pred == ytr[val]).mean())
        if acc > best_acc:
            best_acc, best_l2 = acc, l2
    full = _Probe(Xtr)
    pred = full.predict_classes(ytr, Xte, best_l2, len(CLASSES))
    return float((pred == yte).mean()), best_l2


# ── remote entry point ───────────────────────────────────────────────────
def remote_run(seeds: list) -> dict:
    """Everything for the given seeds; runs on the GPU VM (or locally for a
    smoke test). Returns the JSON-able result dict."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "seeds": []}
    for seed in seeds:
        Xtr, ytr, tries_tr = _build_dataset(seed, N_TRAIN)
        Xte, yte, _ = _build_dataset(seed + 500_009, N_TEST)  # disjoint episode rng
        eye = _get_eye(seed)
        canary_ok = eye.canary() == eye._canary0
        feats_tr = _feature_arms(seed, Xtr, device)
        feats_te = _feature_arms(seed, Xte, device)
        row = {"seed": seed, "canary_ok": bool(canary_ok),
               "canary_colors": eye.canary_colors(),
               "mean_tries": round(tries_tr, 2),
               "n_params_scratch": feats_tr["n_params_scratch"]}
        for arm in ("pretrained", "pre_shipped", "scratch", "rp2048",
                    "rp1024", "pixels"):
            acc, l2 = _probe_acc(feats_tr[arm], ytr, feats_te[arm], yte)
            row[f"acc_{arm}"] = round(acc, 4)
            row[f"l2_{arm}"] = l2
        # control: shuffled train labels on the best (pretrained) family.
        rng = np.random.RandomState(seed + 41)
        ysh = ytr.copy()
        rng.shuffle(ysh)
        acc_sh, _ = _probe_acc(feats_tr["pretrained"], ysh,
                               feats_te["pretrained"], yte)
        row["acc_shuffled"] = round(acc_sh, 4)
        # per-class accuracy of the claim arm (report the minimum — the
        # LESSONS rule about partitions).
        full = _Probe(feats_tr["pretrained"])
        pred = full.predict_classes(ytr, feats_te["pretrained"],
                                    row["l2_pretrained"], len(CLASSES))
        row["per_class_min"] = round(min(
            float((pred[yte == k] == k).mean()) for k in range(len(CLASSES))), 4)
        out["seeds"].append(row)
    return out


# ── GPU submission (one per spec — module cache, T2.01 pattern) ──────────
JOB = r'''
import os as _o
_o.environ["MUJOCO_GL"] = "egl"   # preamble sets "disabled"; this job renders
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
try:
    import transformers  # noqa: F401  both backends ship it; install if not
except ImportError:
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", "transformers"],
            check=True)
import json
from experiments.tests.t2_03_pretrained_vision import remote_run
out = remote_run(__SEEDS__)
json.dump(out, open(_o.path.join(_o.environ["JACK_OUT"], "t203.json"), "w"),
          indent=1)
print("DONE", json.dumps(out["seeds"][0]), flush=True)
'''

_CACHE: dict = {}


def _submit(seeds: list) -> dict:
    body = JOB.replace("__SEEDS__", repr(list(seeds)))
    job = build_job(body)
    res = submit(job, prefer="colab", est_hours=0.4, timeout_s=2940,
                 fetch=["t203.json"])
    if not res.ok:
        raise RuntimeError(f"T2.03 job failed on {res.backend}: {res.message}")
    out = json.loads(Path(res.artifacts["t203.json"]).read_text())
    out["backend"] = res.backend
    return out


def pilot():
    """Seed-90 pilot, disjoint from the registered seeds. Prints, records
    nothing; its numbers go into the docstring and LOOP_JOURNAL by hand."""
    out = _submit([PILOT_SEED])
    print(json.dumps(out, indent=1))
    return out


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit(SEEDS))
    rows = _CACHE["seeds"]
    margins = [r["acc_pretrained"] - r["acc_scratch"] for r in rows]
    null_gaps = [r["acc_pretrained"] - r["acc_rp2048"] for r in rows]
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"],
        "acc_pretrained": [r["acc_pretrained"] for r in rows],
        "acc_scratch": [r["acc_scratch"] for r in rows],
        "acc_rp2048": [r["acc_rp2048"] for r in rows],
        "acc_rp1024": [r["acc_rp1024"] for r in rows],
        "acc_pixels": [r["acc_pixels"] for r in rows],
        "acc_pre_shipped": [r["acc_pre_shipped"] for r in rows],
        "margin_min": round(min(margins), 4),
        "margin_mean": round(sum(margins) / len(margins), 4),
        "null_gap_mean": round(sum(null_gaps) / len(null_gaps), 4),
        "all_seeds_positive": all(m > 0 for m in margins),
        "per_class_min": min(r["per_class_min"] for r in rows),
        "canary_ok_all": all(r["canary_ok"] for r in rows),
        "canary_colors_min": min(r["canary_colors"] for r in rows),
        "n_params_scratch": rows[0]["n_params_scratch"],
        "mean_tries": max(r["mean_tries"] for r in rows),
    }


def _control(seed: int) -> dict:
    rows = _CACHE["seeds"]
    dev = [abs(r["acc_shuffled"] - CHANCE) for r in rows]
    return {"shuffled_dev_max": round(max(dev), 4),
            "acc_shuffled": [r["acc_shuffled"] for r in rows]}


def _check(m: dict, c: dict):
    # Rig first: an invalid run is VOID, not evidence about the hypothesis.
    if not m["canary_ok_all"]:
        return Status.VOID          # GL context degraded mid-run
    if m["canary_colors_min"] < MIN_CANARY_COLORS:
        return Status.VOID          # uniform frame == blind sensor
    if not (PARAMS_RANGE[0] <= m["n_params_scratch"] <= PARAMS_RANGE[1]):
        return Status.VOID          # the seat holder changed under the test
    # Control: shuffled labels must sit at chance or the probe leaks.
    if c["shuffled_dev_max"] > SHUFFLE_BAND:
        return Status.VOID
    # The claim.
    return (m["all_seeds_positive"]
            and m["margin_min"] >= MARGIN_FLOOR
            and m["null_gap_mean"] >= NULL_MARGIN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T2.03"], _experiment, _check, control_fn=_control,
                    ledger=ledger)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        pilot()
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # Local: render a tiny dataset and prove the rig, no GPU, no torch.
        X, y, tries = _build_dataset(PILOT_SEED, 8)
        eye = _get_eye(PILOT_SEED)
        print(json.dumps({
            "frames": list(X.shape), "labels": y.tolist(),
            "mean_tries": tries,
            "canary_ok": eye.canary() == eye._canary0,
            "canary_colors": eye.canary_colors(),
            "distinct_frame_sums": len({int(f.sum()) for f in X}),
        }))
    else:
        run()
