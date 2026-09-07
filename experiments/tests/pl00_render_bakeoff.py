"""PL.00 renderer bakeoff — arm (iii) of `pl02-dependency-on-pl00-verdict-vs-table`
(Review disposition, 2026-09-07): arms measured, not argued (law 3).

WHY THIS EXISTS. PL.00 FAILed (attempt 1, 2026-08-30) and its own decomposition
says the encoder is not why: render-only throughput 4.231 sim-s/real-s sits
under the 5.0 floor with NO encoder in the loop, and the eye's price is
resolution-independent (39.17 ms @224 vs 40.04 ms @64 — 12.25x the pixels for
the same money). The Review ordered the renderer bakeoff instead of ruling on
the PL.02 -> PL.00 edge by exegesis: if an arm clears 5.0 with the eye live,
PL.00 re-runs and the edge dissolves BY BEING SATISFIED — never by being
edited. The 5.0 floor is LC.02's and does not move here or anywhere.

THE DECOMPOSITION THAT SHAPES THE ARMS (measured 2026-09-07, this box, seed 0,
931-colour post-forward scene — the same trap PL.00's near-miss documents):

    full path        39.9 ms      update_scene   0.01 ms     render()  39.4 ms
    shadow flag off  17.1 ms      (the 4096^2 shadow-map pass: ~22.8 ms)
    reflection off   32.3 ms      (the planar reflection pass: ~7.6 ms)
    offsamples=0     27.2 ms      (4x MSAA under llvmpipe: ~12.7 ms)
    offsamples=0 + shadowsize=512          14.1 ms
    offsamples=0 + shadow flag off          8.3 ms

The 40 ms was never rasterisation of Jack's 64x64 frame. It is two fixed
full-scene passes — a 16.7M-pixel shadow map and 4x multisampling — rendered
in software for an eye that reads 4,096 pixels. Nobody chose those numbers;
they are MuJoCo's defaults, inherited silently into the world contract.

THE ARMS, exactly the four the Review named, and what the decomposition
already says about two of them:

  ctx-reuse        reuse the built scene across renders. Its ENTIRE budget is
                   the update_scene component, measured at 0.01 ms of 39.9 —
                   0.03%. FORECLOSED BY MEASURED ARITHMETIC (the ME.11 E/F
                   precedent): the probe re-measures the bound per seed and
                   records it; a placebo implementation would be theatre.
  batched-update   one update_scene serving k renders. Same bound, same
                   foreclosure, same per-seed measurement.
  frame-skip       render every 2nd decision (eye at 2.5 Hz). SCORED AND
                   INELIGIBLE, pre-stated: PL.00's accounting unit is "one
                   rendered frame per decision" (its docstring derives 5 Hz
                   from 40 substeps x 0.005 s), so an arm that renders fewer
                   frames changes the claim rather than satisfying it. Its
                   number goes in the record; it cannot dissolve the edge.
  coarse-shadow512 offsamples=0, shadowsize 4096->512. Shadows KEPT, coarser
                   map, MSAA dropped. ELIGIBLE.
  coarse-flat      offsamples=0, shadow pass OFF (mjRND_SHADOW=0),
                   reflections kept. ELIGIBLE.

NULL: the shipped path — default quality, one update_scene + render per
decision — which measured 4.145 sim-s/real-s with the seat-holder encoder
live. If the null clears the floor on this run the question is moot and the
probe is VOID.

DECISION RULE, pre-stated before any arm ran:

  * Rig gates per seed, VOID on failure: timestep 0.005 exactly, physics
    travel > 1e-6, torch threads == 1, canary drift <= 1e-6 per rig, every
    repeat spread <= 0.25 (PL.00's own bars, unchanged).
  * An ELIGIBLE arm renders exactly one fresh frame per decision.
  * An eligible arm CLEARS iff its WORST-SEED loop throughput, scratch-cnn
    encoder live (PL.00's PURE_ARM, the same object), is >= 5.0.
  * Among clearing arms the winner is the one that discards the LEAST visual
    information, by a ranking declared here before the run:
    coarse-shadow512 > coarse-flat. (Shadows are load-bearing information —
    depth cues under occlusion — and a coarser shadow map degrades them
    where dropping the pass deletes them. If only coarse-flat clears, it
    wins; if neither clears, that is a substrate finding and goes back to
    the queue row with the edge intact.)
  * DISCRIMINATION, carried from PL.00: under the winner's quality the heavy
    reference (ViT-S/14 @224, 219 ms/frame) must still FAIL the floor. If it
    clears, the floor cannot reject anything and no adoption happens.
    Recorded, not assumed — and note what adoption buys here: render-only
    will now clear the floor, so for the first time the floor's rejection of
    the ViT is a sentence about the ENCODER, not about any live eye.

ADOPTION, pre-stated: the winner lands in a NEW module
(`experiments/eye_quality.py`) applied by PL.00's rig, and PL.00 re-runs
through the runner — attempt 2, committed as the runner writes it. It does
NOT land in `playground.py`: 54 test modules declare `playground.py` in
IMPL_DEPS and a quality knob is a property of the RENDERER, not of what is
in the world. Whether existing visual certificates should migrate to the
cheap eye is a separate, routed question — nothing here re-buys or stales
them, and this line says so on purpose.

RESULT (3 seeds, 2026-09-07, artifact /data/pl00_render_bakeoff.json —
appended after the run; nothing above this line was written after an arm ran):

    null (shipped)    worst-seed 4.079  (floor 5.0) — still red, as attempt 1
    frame-skip-2      worst-seed 7.034  ineligible, recorded
    coarse-shadow512  worst-seed 8.594  CLEARS all seeds
    coarse-flat       worst-seed 11.483 CLEARS all seeds
    ctx-reuse bound   0.008 ms of 39.836 full — foreclosed, re-measured/seed
    components/seed   shadow pass 22.55-23.62 ms, MSAA implied ~12.7 ms,
                      reflection 7.11-7.86 ms, update_scene 0.008 ms
    under the winner: heavy ViT @224 worst-seed 0.836 — still FAILS, the
                      discrimination gate holds; render-only worst-seed 8.949
                      — the floor now rejects ENCODERS, not any live eye,
                      which attempt 1 recorded it could not distinguish
    every canary 0.0, every spread <= 0.0175 vs the 0.25 bar

    WINNER: coarse-shadow512 (shadows kept at 512^2, MSAA off), per the
    pre-declared ranking — both eligible arms cleared, the one keeping more
    visual information wins. Adopted in experiments/eye_quality.py; PL.00
    re-runs through the runner as attempt 2.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = Path("/data/pl00_render_bakeoff.json")

# ── PL.00's constants, inherited unchanged. ────────────────────────────────
FLOOR = 5.0
SUBSTEPS = 40
DT_NOMINAL = 0.005
SIM_S_PER_DECISION = SUBSTEPS * DT_NOMINAL
RES = 64
RES_VIT = 224
N_DEC = 60
N_DEC_HEAVY = 15
LOOP_REPEATS = 3
MAX_REL_SPREAD = 0.25
MIN_PHYSICS_TRAVEL = 1e-6
CANARY_TOL = 1e-6
SEEDS = [0, 1, 2]
SKIP_K = 2                      # frame-skip renders every SKIP_K-th decision

# Pre-declared information ranking among eligible arms (most kept first).
ELIGIBLE_RANKING = ["coarse-shadow512", "coarse-flat"]

_RIGS: dict = {}


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


class _Rig:
    """PL.00's rig, with the quality knobs as explicit constructor arguments.
    Renderers are created once and held for the process lifetime (the GL trap,
    pg_6.get_eye). offsamples/shadowsize must be set on the model BEFORE the
    Renderer exists — the framebuffers are allocated at construction."""

    def __init__(self, seed: int, res: int, offsamples: int | None,
                 shadowsize: int | None, shadow_flag: bool):
        from experiments.render import ensure_gl
        ensure_gl()
        import mujoco
        import playground as pg
        self.mj = mujoco
        params = pg.PlaygroundParams(seed=seed)
        self.model, self.data, _ = pg.make_playground(params, with_water=False)
        if offsamples is not None:
            self.model.vis.quality.offsamples = offsamples
        if shadowsize is not None:
            self.model.vis.quality.shadowsize = shadowsize
        self.r = mujoco.Renderer(self.model, height=res, width=res)
        if not shadow_flag:
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._canary = None
        self._canary = self.canary()
        if not shadow_flag:  # canary() leaves flags alone; assert it held
            assert self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] == 0

    def canary(self) -> float:
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


def _rig(seed: int, name: str, **kw) -> _Rig:
    key = (seed, name)
    if key not in _RIGS:
        _RIGS[key] = _Rig(seed, **kw)
    return _RIGS[key]


def _build_scratch(res: int):
    """The seat holder exactly as shipped — the same object PL.00 times."""
    import sys
    sys.path.insert(0, str(REPO))
    from UnifiedBrain import UnifiedBrainConfig, PrismaticVisionEncoder
    torch = _torch()
    enc = PrismaticVisionEncoder(UnifiedBrainConfig(), image_size=res).eval()
    with torch.no_grad():
        enc(torch.zeros(1, 3, res, res))
    return enc


def _build_vit():
    """ViT-S/14 architecture @224, randomly initialised — cost only, PL.00's
    own reasoning: cost does not depend on the weight values."""
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
            self.enc = nn.TransformerEncoder(layer, depth,
                                             enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            t = self.patch(x).flatten(2).transpose(1, 2)
            t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
            return self.norm(self.enc(t))[:, 0]

    enc = ViTS14().eval()
    with torch.no_grad():
        import torch
        enc(torch.zeros(1, 3, RES_VIT, RES_VIT))
    return enc


def _loop(rig: _Rig, encoder, n_dec: int, render_every: int, res: int) -> float:
    """PL.00's loop verbatim, plus render_every for the frame-skip arm
    (render_every=1 is one fresh frame per decision — the accounting unit)."""
    torch = _torch()
    mj = rig.mj
    rig.reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_dec):
            for _ in range(SUBSTEPS):
                mj.mj_step(rig.model, rig.data)
            if render_every and i % render_every == 0:
                f = rig.frame()
                if encoder is not None:
                    x = torch.from_numpy(
                        np.ascontiguousarray(f.transpose(2, 0, 1))
                    ).float().div_(255.0).unsqueeze(0)
                    encoder(x)
    dt = time.perf_counter() - t0
    return n_dec * SIM_S_PER_DECISION / max(dt, 1e-9)


def _loop_median(rig, encoder, n_dec, render_every, res) -> tuple:
    ts = [_loop(rig, encoder, n_dec, render_every, res)
          for _ in range(LOOP_REPEATS)]
    med = float(np.median(ts))
    return round(med, 3), round(float(np.std(ts)) / max(med, 1e-9), 4)


def _components(rig: _Rig, reps: int = 30) -> dict:
    """The per-pass decomposition, measured on this rig in this process.
    `update_scene_ms` is the entire budget of ctx-reuse and batched-update."""
    import mujoco
    rig.reset()
    for _ in range(200):
        rig.mj.mj_step(rig.model, rig.data)   # the 931-colour trap: warm scene

    def t(f):
        f()
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        return round((time.perf_counter() - t0) / reps * 1000, 3)

    full = t(lambda: (rig.r.update_scene(rig.data, camera="eye"),
                      rig.r.render()))
    us = t(lambda: rig.r.update_scene(rig.data, camera="eye"))
    rig.r.update_scene(rig.data, camera="eye")
    rend = t(lambda: rig.r.render())
    rig.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    no_shadow = t(lambda: (rig.r.update_scene(rig.data, camera="eye"),
                           rig.r.render()))
    rig.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    rig.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    no_refl = t(lambda: (rig.r.update_scene(rig.data, camera="eye"),
                         rig.r.render()))
    rig.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    return {"full_ms": full, "update_scene_ms": us, "render_ms": rend,
            "shadow_pass_ms": round(full - no_shadow, 3),
            "reflection_pass_ms": round(full - no_refl, 3)}


def _seed_pass(seed: int) -> dict:
    torch = _torch()
    out: dict = {"seed": seed, "torch_threads": int(torch.get_num_threads())}

    null_rig = _rig(seed, "null", res=RES, offsamples=None,
                    shadowsize=None, shadow_flag=True)
    c512_rig = _rig(seed, "c512", res=RES, offsamples=0,
                    shadowsize=512, shadow_flag=True)
    flat_rig = _rig(seed, "flat", res=RES, offsamples=0,
                    shadowsize=None, shadow_flag=False)

    # Rig gates: physics contract on the null rig (one model, shared physics).
    out["model_timestep"] = float(null_rig.model.opt.timestep)
    out["timestep_ok"] = int(abs(out["model_timestep"] - DT_NOMINAL) < 1e-12)
    null_rig.reset()
    q0 = np.array(null_rig.data.qpos, copy=True)
    for _ in range(SUBSTEPS):
        null_rig.mj.mj_step(null_rig.model, null_rig.data)
    out["physics_travel"] = round(
        float(np.abs(np.array(null_rig.data.qpos) - q0).max()), 9)

    out["components_null"] = _components(null_rig)

    enc = _build_scratch(RES)
    out["physics_only"], out["physics_only_spread"] = _loop_median(
        null_rig, None, N_DEC, 0, RES)
    out["T_null"], out["T_null_spread"] = _loop_median(
        null_rig, enc, N_DEC, 1, RES)
    out["T_frame-skip-2"], out["T_frame-skip-2_spread"] = _loop_median(
        null_rig, enc, N_DEC, SKIP_K, RES)
    out["T_coarse-shadow512"], out["T_coarse-shadow512_spread"] = _loop_median(
        c512_rig, enc, N_DEC, 1, RES)
    out["T_coarse-flat"], out["T_coarse-flat_spread"] = _loop_median(
        flat_rig, enc, N_DEC, 1, RES)
    del enc

    for name, rig in (("null", null_rig), ("c512", c512_rig),
                      ("flat", flat_rig)):
        out[f"canary_drift_{name}"] = round(
            abs(rig.canary() - rig._canary) / max(abs(rig._canary), 1.0), 9)
    return out


def _winner_pass(seed: int, winner: str) -> dict:
    """Under the winner's quality: render-only, and the heavy reference."""
    kw = dict(offsamples=0, shadowsize=512, shadow_flag=True) \
        if winner == "coarse-shadow512" else \
        dict(offsamples=0, shadowsize=None, shadow_flag=False)
    rig64 = _rig(seed, f"win64", res=RES, **kw)
    rig224 = _rig(seed, f"win224", res=RES_VIT, **kw)
    out: dict = {}
    out["render_only_winner"], out["render_only_winner_spread"] = _loop_median(
        rig64, None, N_DEC, 1, RES)
    vit = _build_vit()
    out["T_heavy_winner"], out["T_heavy_winner_spread"] = _loop_median(
        rig224, vit, N_DEC_HEAVY, 1, RES_VIT)
    del vit
    for name, rig in (("win64", rig64), ("win224", rig224)):
        out[f"canary_drift_{name}"] = round(
            abs(rig.canary() - rig._canary) / max(abs(rig._canary), 1.0), 9)
    return out


def main() -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass

    seeds = [_seed_pass(s) for s in SEEDS]
    result: dict = {"floor": FLOOR, "seeds": seeds,
                    "eligible_ranking": ELIGIBLE_RANKING, "skip_k": SKIP_K}

    def worst(key):
        return min(s[key] for s in seeds)

    # ── Rig gates (VOID: an invalid run is not evidence). ──────────────────
    void = []
    for s in seeds:
        if not s["timestep_ok"]:
            void.append(f"seed {s['seed']}: timestep")
        if s["physics_travel"] < MIN_PHYSICS_TRAVEL:
            void.append(f"seed {s['seed']}: no physics travel")
        if s["torch_threads"] != 1:
            void.append(f"seed {s['seed']}: torch threads")
        for k, v in s.items():
            if k.startswith("canary_drift_") and v > CANARY_TOL:
                void.append(f"seed {s['seed']}: {k}={v}")
            if k.endswith("_spread") and v > MAX_REL_SPREAD:
                void.append(f"seed {s['seed']}: {k}={v}")
    # The null must still fail the floor, or the question is moot.
    if worst("T_null") >= FLOOR:
        void.append(f"null clears the floor (worst {worst('T_null')}); moot")

    if void:
        result["verdict"] = "VOID"
        result["void_reasons"] = void
    else:
        # Foreclosure bound for ctx-reuse and batched-update, per seed.
        bound = max(s["components_null"]["update_scene_ms"] for s in seeds)
        full = min(s["components_null"]["full_ms"] for s in seeds)
        result["ctx_reuse_bound_ms"] = bound
        result["foreclosed"] = {
            "ctx-reuse": f"entire budget is update_scene, measured "
                         f"{bound} ms of {full} ms — cannot move the verdict",
            "batched-update": "same bound, same measurement",
        }
        clears = {a: worst(f"T_{a}") >= FLOOR for a in ELIGIBLE_RANKING}
        result["worst_seed"] = {a: worst(f"T_{a}") for a in
                                ["null", "frame-skip-2"] + ELIGIBLE_RANKING}
        result["clears"] = clears
        result["frame_skip_note"] = ("scored and INELIGIBLE, pre-stated: "
                                     "renders 1 frame per 2 decisions, which "
                                     "changes PL.00's accounting unit")
        winner = next((a for a in ELIGIBLE_RANKING if clears[a]), None)
        if winner is None:
            result["verdict"] = "NO-ARM-CLEARS"
        else:
            wp = [_winner_pass(s, winner) for s in SEEDS]
            result["winner_pass"] = wp
            heavy_worst = max(w["T_heavy_winner"] for w in wp)
            result["heavy_still_fails"] = heavy_worst < FLOOR
            drift_bad = any(v > CANARY_TOL for w in wp for k, v in w.items()
                            if k.startswith("canary_drift_"))
            if drift_bad:
                result["verdict"] = "VOID"
                result["void_reasons"] = ["canary drift in winner pass"]
            elif not result["heavy_still_fails"]:
                result["verdict"] = "VOID"
                result["void_reasons"] = [
                    f"heavy reference clears the floor under {winner} "
                    f"({heavy_worst}); the floor cannot reject — no adoption"]
            else:
                result["verdict"] = "WINNER"
                result["winner"] = winner

    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "seeds"},
                     indent=2))
    return result


if __name__ == "__main__":
    main()
