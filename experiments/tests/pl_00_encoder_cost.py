"""PL.00 — what each perception encoder costs on THIS box, and whether the one
architecture the PLASTIC-ONLY decree admits can afford to have eyes.

THE ARENA THAT WAS NAMED AND NEVER BUILT. `GOAL.md:76` decrees PLASTIC ONLY:
nothing inside Jack is frozen, encoders included. `docs/CHAMPIONS.md` records
the decree's pre-registered RE-OPEN TRIGGER verbatim —

    "if a from-scratch encoder cannot hit the PL.00 throughput floor on this
     hardware ... the decision returns to the owner with that number attached"

— and for twenty-one days `PL.00` did not exist. Seven consecutive audits asked
for it; `champions.py` counted the decree among 7/7 UNFALSIFIABLE seats;
`run coverage` read `plasticity  2 specs  0 pass  0 now`, both of its claim
specs blocked behind `T2.01`. This file is the trigger's finger.

WHAT IS MEASURED, SAID PLAINLY, AND WHAT IS NOT.

  * Leg A — **ms/frame per encoder**, batch 1, `torch.set_num_threads(1)`, at
    each encoder's native resolution, per-trial warmup.
  * Leg B — **loop throughput** in simulated seconds of Jack's life per real
    second, physics + a live rendered eye + the encoder, against the **5.0**
    floor. The floor is not this file's: it is `LC.02`'s and
    `LEARNING_CORE.md` §5.0b's, inherited unchanged.
  * The null is the **measured render cost**, measured in this process. An
    encoder cheaper than its own render is free; one costing 10x the render is
    the dominant cost of having eyes. `DIRECTION_AUDIT.md`'s 68 ms at 128x128
    is a figure to re-measure, never to quote.

WHAT THIS SPEC DOES NOT CLAIM. Cost is a function of ARCHITECTURE and input
size, not of the values in the tensors: a randomly-initialised ViT-S/14 at 224
costs exactly what a pretrained DINOv2 ViT-S/14 costs. So the frozen-tower
reference below is locally constructed and needs no download — and **no
accuracy claim is made about it here.** That is `T2.03`'s job and is already
on the ledger. Under `SYSTEM.md`'s 2026-08-24 amendment the frozen reference is
SCORED-AND-INELIGIBLE: its number goes in the record, it cannot take a seat.

THE ACCOUNTING UNIT is `w0.py`'s and `LC.02`'s, so the floor means the same
thing here as where it was set: one decision is 40 substeps of 0.005 s = 0.2
simulated seconds. "Vision live at 5 Hz" is therefore **exactly one rendered
frame per decision** — the two constants coincide, which is why 5 Hz was the
number written into the spec.

TWO CONTROLS, and the second is the one that could have killed the leg.

  1. **IDENTITY** (declared in the registry). A no-op encoder that reads every
     pixel and does nothing else. It must sit at ~0 ms/frame and must NOT move
     loop throughput away from the render-only loop. If swapping the encoder
     for nothing at all changes the throughput, the harness is timing something
     other than the encoder and every cell is uninterpretable.
  2. **HEAVY REFERENCE — the discrimination check.** A 21.6M-parameter ViT-S/14
     at 224 must FAIL the 5.0 floor on this box. A floor that a frozen ViT
     clears is a floor that cannot exclude anything, and "the pure encoder
     cleared it" would then be a sentence about the bar rather than about the
     encoder. This gate returns **VOID, not FAIL** — a floor that does not
     discriminate is an invalid instrument, not a refuted hypothesis. It is the
     third member of the family `LESSONS.md` already carries (an at-chance
     control must prove it could pass; a trained null must prove it reached its
     ceiling; an intervention must prove it landed) — here, **a threshold must
     prove it can reject.**

THE GL TRAP, inherited rather than rediscovered. A `mujoco.Renderer` that is
garbage-collected poisons the shared X display, and the NEXT renderer returns
frames that are corrupted but entirely plausible (`pg_6.get_eye`'s docstring,
measured 2026-08-09). So renderers here are created once per resolution and
held for the process lifetime, and each carries a CANARY frame re-rendered from
a reset state at the end of every seed. A canary that moves returns VOID.
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID

REPO = Path(__file__).resolve().parents[2]
SPEC_ID = "PL.00"

# ── PRE-REGISTERED CONSTANTS. Fixed before the run; see the docstring for the
# provenance of each. Nothing here was fitted to an observation. ────────────
FLOOR = 5.0                  # sim-s per real-s. LC.02 / LEARNING_CORE §5.0b.
SUBSTEPS = 40                # w0.py's decision, unchanged
DT_NOMINAL = 0.005           # s per substep; asserted against the model below
SIM_S_PER_DECISION = SUBSTEPS * DT_NOMINAL          # 0.2

RES_SMALL = 64               # the seat holder's native input
RES_VIT = 224                # ViT-S/14's native input
N_DEC = 60                   # loop decisions per repeat, cheap arms
N_DEC_HEAVY = 15             # loop decisions per repeat, the ViT reference
LOOP_REPEATS = 3
FRAME_TRIALS = 3
FRAME_REPS = 5               # forwards per trial (per-trial warmup, T0.07)

IDENTITY_MS_MAX = 0.20       # a no-op that reads 12,288 floats and nothing else
IDENTITY_T_TOL = 0.10        # identity vs render-only loop throughput
MAX_REL_SPREAD = 0.25        # T0.07's repeat-stability bar, reused unchanged
MIN_PHYSICS_TRAVEL = 1e-6    # qpos must actually move, or nothing was simulated
CANARY_TOL = 1e-6            # relative; the GL context must not degrade

# Held for the process lifetime. See the GL trap in the docstring.
_RIGS: dict = {}


# --------------------------------------------------------------------------
# The encoders. Every one is constructed locally — no download, no network,
# and therefore no silent substitution of a smaller model for a bigger one
# (the fabricated-mocap disease in vision form, UnifiedBrain.py:670).
# --------------------------------------------------------------------------

def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def _build_identity():
    torch = _torch()
    import torch.nn as nn

    class Identity(nn.Module):
        """Reads every pixel, does nothing else. The honest zero."""
        def forward(self, x):
            return x.mean(dim=(2, 3))

    return Identity().eval()


def _build_scratch():
    """The seat holder exactly as shipped: `PrismaticVisionEncoder`'s CNN
    fallback, 0.245M params — the same object `T2.03`, `T3.01` and
    `T3.10` measure. Not a re-implementation."""
    sys.path.insert(0, str(REPO))
    from UnifiedBrain import UnifiedBrainConfig, PrismaticVisionEncoder
    return PrismaticVisionEncoder(UnifiedBrainConfig(), image_size=RES_SMALL).eval()


def _build_dreamer():
    """A dreamer-xs-class conv stack at 64 — the standard cheap world-model
    encoder, present as a candidate because the seat holder being affordable
    and the seat holder being the ONLY affordable thing are different facts."""
    torch = _torch()
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
        nn.Conv2d(128, 256, 4, 2), nn.ReLU(),
        nn.Flatten(), nn.LazyLinear(256),
    ).eval()


def _build_vit():
    """DINOv2 ViT-S/14's ARCHITECTURE at 224: patch 14, dim 384, depth 12,
    6 heads, ~21.6M params. Randomly initialised on purpose — see the docstring:
    this leg measures cost, and cost does not depend on the weight values."""
    torch = _torch()
    import torch.nn as nn

    class ViTS14(nn.Module):
        def __init__(self, img=RES_VIT, p=14, d=384, depth=12, heads=6):
            super().__init__()
            self.patch = nn.Conv2d(3, d, p, p)
            n = (img // p) ** 2
            self.pos = nn.Parameter(torch.zeros(1, n + 1, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(
                d, heads, d * 4, batch_first=True, activation="gelu",
                norm_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            t = self.patch(x).flatten(2).transpose(1, 2)
            t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
            return self.norm(self.enc(t))[:, 0]

    return ViTS14().eval()


def _mel_matrix(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank, written out rather than imported, so this
    spec has no torchaudio dependency and its cost is fully attributable."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(20.0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, min(mid, fb.shape[1])):
            fb[i, j] = (j - lo) / max(mid - lo, 1)
        for j in range(mid, min(hi, fb.shape[1])):
            fb[i, j] = (hi - j) / max(hi - mid, 1)
    return fb


def _build_mel():
    """The audio side of "what does a sense cost". 0.2 s of 16 kHz mono — one
    decision's worth, the same accounting unit as the eye."""
    torch = _torch()
    import torch.nn as nn

    class Mel(nn.Module):
        def __init__(self, sr=16000, n_fft=512, hop=160, n_mels=64):
            super().__init__()
            self.n_fft, self.hop = n_fft, hop
            self.register_buffer("fb", torch.from_numpy(_mel_matrix(n_mels, n_fft, sr)))
            self.register_buffer("win", torch.hann_window(n_fft))

        def forward(self, x):
            s = torch.stft(x, self.n_fft, self.hop, window=self.win,
                           return_complex=True).abs()
            return torch.log1p(self.fb @ s.squeeze(0)).unsqueeze(0)

    return Mel().eval()


# name -> (builder, input kind, resolution, role)
#   candidate  — an architecture that may take the perception seat
#   control    — the declared no-op (must not move throughput)
#   reference  — SCORED-AND-INELIGIBLE under the PLASTIC-ONLY decree
ENCODERS = [
    ("identity",    _build_identity, "image", RES_SMALL, "control"),
    ("scratch-cnn", _build_scratch,  "image", RES_SMALL, "candidate"),
    ("dreamer-cnn", _build_dreamer,  "image", RES_SMALL, "candidate"),
    ("vit-s14",     _build_vit,      "image", RES_VIT,   "reference"),
    ("mel-fbank",   _build_mel,      "audio", 0,         "candidate"),
]
# The one arm the decree admits for the eye. Named here, once, so the claim
# branch cannot quietly widen to "some candidate cleared it".
PURE_ARM = "scratch-cnn"
HEAVY_ARM = "vit-s14"


# --------------------------------------------------------------------------
# The rig: one playground, one renderer per resolution, held forever.
# --------------------------------------------------------------------------

class _Rig:
    def __init__(self, seed: int, res: int):
        from experiments.render import ensure_gl
        ensure_gl()
        import mujoco
        import playground as pg
        self.mj = mujoco
        params = pg.PlaygroundParams(seed=seed)
        self.model, self.data, _ = pg.make_playground(params, with_water=False)
        self.r = mujoco.Renderer(self.model, height=res, width=res)
        self._canary = None
        self._canary = self.canary()

    def canary(self) -> float:
        """A fixed reference frame from a reset state, reduced to one number.
        If the GL context degrades mid-run the frames stay plausible but change,
        and this is the only thing that notices."""
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        self.r.update_scene(self.data, camera="eye")
        return float(self.r.render().astype(np.float64).sum())

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)

    def frame(self) -> np.ndarray:
        self.r.update_scene(self.data, camera="eye")
        return self.r.render()


def _rig(seed: int, res: int) -> _Rig:
    key = (seed, res)
    if key not in _RIGS:
        _RIGS[key] = _Rig(seed, res)
    return _RIGS[key]


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------------
# Leg A — ms/frame.
# --------------------------------------------------------------------------

def _time_forward(model, x, reps=FRAME_REPS, trials=FRAME_TRIALS) -> tuple:
    """Per-trial warmup, not warm-once. T0.07 measured a heavier computation
    running FASTER than a lighter one because the first warmup was still paging
    the model in; the spread fell from 28.9% to 6.4% when warmup moved inside
    the trial loop. The threshold was not what changed."""
    torch = _torch()
    out = []
    with torch.no_grad():
        for _ in range(trials):
            model(x)
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            out.append((time.perf_counter() - t0) / reps * 1000.0)
    med = float(np.median(out))
    spread = float(np.std(out) / max(med, 1e-9))
    return med, spread


def _leg_a(seed: int) -> dict:
    torch = _torch()
    rows = {}
    for name, build, kind, res, role in ENCODERS:
        rss0 = _rss_mb()
        m = build()
        if kind == "image":
            x = torch.zeros(1, 3, res, res)
        else:
            x = torch.zeros(1, 3200)          # 0.2 s @ 16 kHz, one decision
        with torch.no_grad():                  # LazyLinear needs one pass
            m(x)
        ms, spread = _time_forward(m, x)
        rows[name] = {
            "ms_per_frame": round(ms, 4),
            "rel_spread": round(spread, 4),
            "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 4),
            "rss_delta_mb": round(max(_rss_mb() - rss0, 0.0), 1),
            "res": res, "kind": kind, "role": role,
        }
        del m
    return rows


# --------------------------------------------------------------------------
# Leg B — loop throughput, sim-s per real-s, vision live at 5 Hz.
# --------------------------------------------------------------------------

def _loop(rig: _Rig, encoder, n_dec: int, render: bool) -> float:
    torch = _torch()
    mj = rig.mj
    rig.reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_dec):
            for _ in range(SUBSTEPS):
                mj.mj_step(rig.model, rig.data)
            if render:
                f = rig.frame()
                if encoder is not None:
                    x = torch.from_numpy(
                        np.ascontiguousarray(f.transpose(2, 0, 1))
                    ).float().div_(255.0).unsqueeze(0)
                    encoder(x)
    dt = time.perf_counter() - t0
    return n_dec * SIM_S_PER_DECISION / max(dt, 1e-9)


def _loop_median(rig, encoder, n_dec, render) -> tuple:
    ts = [_loop(rig, encoder, n_dec, render) for _ in range(LOOP_REPEATS)]
    med = float(np.median(ts))
    return round(med, 3), round(float(np.std(ts)) / max(med, 1e-9), 4)


def _leg_b(seed: int) -> dict:
    rig64 = _rig(seed, RES_SMALL)
    rig224 = _rig(seed, RES_VIT)
    out: dict = {}

    # Ceilings, in order of what they remove. Each is a denominator for the
    # next, so "the encoder is expensive" can never be confused with "the
    # renderer is expensive" or "the physics is expensive".
    out["physics_only"], out["physics_only_spread"] = _loop_median(rig64, None, N_DEC, False)
    out["render_only_64"], out["render_only_64_spread"] = _loop_median(rig64, None, N_DEC, True)
    out["render_only_224"], _ = _loop_median(rig224, None, N_DEC_HEAVY, True)

    for name, build, kind, res, role in ENCODERS:
        if kind != "image":
            continue
        rig = rig224 if res == RES_VIT else rig64
        n = N_DEC_HEAVY if res == RES_VIT else N_DEC
        enc = build()
        import torch
        with torch.no_grad():
            enc(torch.zeros(1, 3, res, res))
        t, spread = _loop_median(rig, enc, n, True)
        out[f"T_{name}"] = t
        out[f"T_{name}_spread"] = spread
        out[f"clears_floor_{name}"] = int(t >= FLOOR)
        del enc

    # Raw render cost — the null baseline, measured, not quoted.
    for rig, res in ((rig64, RES_SMALL), (rig224, RES_VIT)):
        rig.reset()
        rig.frame()
        t0 = time.perf_counter()
        for _ in range(20):
            rig.frame()
        out[f"render_ms_{res}"] = round((time.perf_counter() - t0) / 20 * 1000, 3)

    out["canary_drift_64"] = round(
        abs(rig64.canary() - rig64._canary) / max(abs(rig64._canary), 1.0), 9)
    out["canary_drift_224"] = round(
        abs(rig224.canary() - rig224._canary) / max(abs(rig224._canary), 1.0), 9)
    return out


# --------------------------------------------------------------------------

def _experiment(seed: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    torch = _torch()

    rig = _rig(seed, RES_SMALL)
    # The physics contract the floor's denominator rests on. Asserted, not
    # assumed: if the timestep is not 0.005 s, "0.2 simulated seconds per
    # decision" is false and every throughput figure below is mislabelled.
    dt_model = float(rig.model.opt.timestep)

    rig.reset()
    q0 = np.array(rig.data.qpos, copy=True)
    for _ in range(SUBSTEPS):
        rig.mj.mj_step(rig.model, rig.data)
    travel = float(np.abs(np.array(rig.data.qpos) - q0).max())

    a = _leg_a(seed)
    b = _leg_b(seed)

    m = {
        "seed": seed,
        "torch_threads": int(torch.get_num_threads()),
        "nice": float(os.nice(0)),
        "model_timestep": dt_model,
        "timestep_ok": int(abs(dt_model - DT_NOMINAL) < 1e-12),
        "physics_travel": round(travel, 9),
        "peak_rss_mb": round(_rss_mb(), 1),
        "floor": FLOOR,
    }
    for name, row in a.items():
        for k, v in row.items():
            if k in ("kind", "role"):
                continue
            m[f"A_{name}_{k}"] = v
    m.update(b)

    # The headline, spelled out so a ledger reader does not have to re-derive it.
    m["pure_arm"] = PURE_ARM
    m["pure_T"] = b[f"T_{PURE_ARM}"]
    m["pure_clears_floor"] = b[f"clears_floor_{PURE_ARM}"]
    m["heavy_T"] = b[f"T_{HEAVY_ARM}"]
    m["heavy_clears_floor"] = b[f"clears_floor_{HEAVY_ARM}"]
    m["pure_ms_vs_render_ratio"] = round(
        a[PURE_ARM]["ms_per_frame"] / max(b[f"render_ms_{RES_SMALL}"], 1e-9), 4)
    m["heavy_ms_vs_render_ratio"] = round(
        a[HEAVY_ARM]["ms_per_frame"] / max(b[f"render_ms_{RES_VIT}"], 1e-9), 4)
    return m


def _control(seed: int) -> dict:
    """The declared control: the identity encoder must not move the loop.

    Its leg-A and leg-B numbers are produced by `_experiment` on the same rig
    in the same process — re-timing them here would compare two different
    thermal and cache states and call the difference an effect. What this
    function does is state the comparison as its own artifact so the control's
    verdict is a recorded number rather than an inference in `_check`."""
    rig = _rig(seed, RES_SMALL)
    enc = _build_identity()
    import torch
    ms, _ = _time_forward(enc, torch.zeros(1, 3, RES_SMALL, RES_SMALL))
    t_id, _ = _loop_median(rig, enc, N_DEC, True)
    t_render, _ = _loop_median(rig, None, N_DEC, True)
    return {
        "identity_ms_per_frame": round(ms, 4),
        "identity_T": round(t_id, 3),
        "render_only_T": round(t_render, 3),
        "identity_rel_shift": round(abs(t_id - t_render) / max(t_render, 1e-9), 4),
    }


def _check(m: dict, c: dict):
    # ── RIG GATES. Each returns VOID: an invalid run is not evidence against
    # a hypothesis (the Status.VOID lesson), and a floor that cannot reject is
    # an invalid instrument, not a refuted claim.
    if not m["timestep_ok"]:
        return Status.VOID
    if m["physics_travel"] < MIN_PHYSICS_TRAVEL:
        return Status.VOID
    if m["torch_threads"] != 1:
        return Status.VOID
    if max(m["canary_drift_64"], m["canary_drift_224"]) > CANARY_TOL:
        return Status.VOID
    for name, _b, kind, _r, _role in ENCODERS:
        if m[f"A_{name}_rel_spread"] > MAX_REL_SPREAD:
            return Status.VOID
        if kind == "image" and m[f"T_{name}_spread"] > MAX_REL_SPREAD:
            return Status.VOID
    # THE DISCRIMINATION CHECK: the floor must be able to reject something.
    if m["heavy_clears_floor"]:
        return Status.VOID

    # ── THE DECLARED CONTROL. A no-op must cost nothing and change nothing.
    control_ok = (c["identity_ms_per_frame"] <= IDENTITY_MS_MAX
                  and c["identity_rel_shift"] <= IDENTITY_T_TOL)

    # ── THE CLAIM. Every encoder measured, and the arm the decree admits
    # clears the floor with vision live at 5 Hz.
    measured = all(m[f"A_{n}_ms_per_frame"] > 0.0 for n, *_ in ENCODERS)
    return bool(measured and control_ok and m["pure_clears_floor"])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    print(run())
