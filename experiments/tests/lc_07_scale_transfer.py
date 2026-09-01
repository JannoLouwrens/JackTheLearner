"""LC.07 — The wm-latent verdict survives ~10x scale (the owner's
scale-transfer guard).

REGISTERED 2026-09-01 by D10's firing commit (54th-audit B1): the commit that
seated wm-latent BY VERDICT must not leave the learning-core seat with a dead
arena. This file is that arena. It re-buys the ONLY clean 3-sigma result the
LC.03 v2 screen produced — wm-latent, t_null 4.65 / t_twin 4.00 / needs_rise
+0.022 / clt +92.2 at 400,000 decisions per arm-seed — at the scale the
owner's adoption guard names (D10/D12: "re-test at ~10x on Kaggle, which is
free"). If either sigma gate misses at 10x, the 1x verdict was a
small-envelope artifact and the seat reverts to contested-VACANT.

SINGLE-ARM ON PURPOSE. This is not a re-run of the five-arm screen (LC.03 is
VOID-FORECLOSED: no v3, no envelope growth, no re-roll) and it does not route
through LC.03 in depends_on. It races nobody; it asks whether the one seated
core's own gates survive deployment scale. Racing new arms is LC.04's job if
the seat's premise is ever repaired.

THE MACHINERY IS INHERITED, NOT REWRITTEN. Every mechanism is imported from
`lc_03_survival_screening` / `experiments.survival` so the definitions CANNOT
drift: `run_survival` computes life_gain (final-third minus first-third mean
survival) — a moved definition is a moved threshold, so this file never
reimplements it; `_randrew_factory`, `_panel_dwell`, `_chaos_detect`,
`_tstat`, `_void` and the gate constants (SIGMA_GATE 3.0, N_LIVES_MIN 12,
NOISE_FLOOR_S 5.0, PANEL_DWELL_MAX 0.15, CHAOS_* §2.10) are the 1x objects,
byte-identical via import. IMPL_DEPS pins all of it.

THE ENVELOPE (the 10x the registry names — these numbers may NOT move):

    N_STEPS_10X   4,000,000 decisions  wm-latent, its wiped twin, and the
                                       paired random null (LC.03 x10)
    HALF_10X      2,000,000 decisions  frozen twin + the three controls
                                       (LC.03's own halving rule, x10)
    E0            1.0                  LC.03's regime, unchanged
    explore_std   (0.3, 0.3)           EXPLORE_STD_LC03, unchanged
    train_ratio   LC.02's committed wm-latent ratio, borrowed (same borrow
                  machinery; an uncalibrated borrow VOIDs, never refutes)
    seeds         0, 1, 2              the registered seeds

TWO RECORDED DEVIATIONS FROM THE 1x RUN SHAPE, neither a gate move:
  - min_core_s (LC.03's W_CLOCK, "whichever is LATER") is DROPPED. It was the
    five-arm matched-COMPUTE fairness floor; a single-arm re-test at matched
    envelope races nobody, and at 1x the two quantities coincided anyway
    (400k decisions cost ~17,280 core-s naturally, which is why W_CLOCK was
    set to exactly that). The registered quantity here is decisions.
  - record_xy is dropped on the null run (at 1x it was recorded and consumed
    by nothing: panel_dwell is computed for arms only). Dropping an unused
    recording is not a control change. The arm's xy IS recorded and its
    panel_dwell gate is inherited verbatim.

THE CHAOS DETECTOR RUNS AT THE SAME EVIDENCE MASS. TRACE_EVERY_10X = 80, so
4,000,000 / 80 = 50,000 transition rows per run — exactly the 400,000 / 8 the
1x detector was validated on. Same rows, same constants, same 5-fold
instrument; only the subsampling stride scales with the envelope. The pool is
{wm-latent, null_random} per seed and it is computed IN THE KERNEL (the raw
rows never cross the wire), which constrains the kernel split: a seed's arm
and null runs must land in the same kernel or that seed's chaos gate cannot
be computed — the freeze step must respect this, and remote_run records
`chaos_missing` loudly if it was violated rather than quietly passing.

THE GATES — the exact conjunct set wm-latent cleared at 1x, restricted to one
arm. Nothing added, nothing dropped, no constant moved:

    CLAIM (all six, aggregated over 3 seeds):
      t(lg_margin_null)  >= 3     life_gain beats the paired random null
      t(lg_margin_twin)  >= 3     ... and its own untrained twin
      lives_ok           == 1     n_lives >= 12 on every seed (arm runs)
      needs_rise         >  0     needs-satisfaction rises within lives
      clt                >  0     cross-life transfer: trained minus wiped
      dwell_ok           == 1     panel_dwell <= 0.15 (PG.4's anti-gaming)

    CONTROLS, each on its pre-registered side, run in the same kernels:
      (a) statue rides the basal ceiling: |mean_life - E0/BASAL_B| <= 10%
          per seed (phantom-servo scar, PS.03) — off-ceiling => VOID
      (b) the wiped-store twin must not trip: |t(wiped_life_gain)| < 3 or
          |value| <= NOISE_FLOOR_S (zero cross-life carryover when memory is
          wiped) — a hot wiped twin => VOID
      (c) randrew (ppo-needs core on a fixed random stationary reward — the
          1x instrument, inherited) must MISS the null gate:
          t(randrew_margin) < 3 — cleared => VOID
      frozen twin hot ( |t(twin_life_gain)| >= 3 AND |value| > 5 s ) => VOID
      chaos conjunction (occupancy >= 3.0 AND ratio >= 2.0) on the arm => VOID
    Every VOID names its firing branch in `void_reason` (the v2 lesson: a
    generic VOID admits every narrative).

    FAIL is a refutation, not an accident: if the instrument branches are all
    clean and a claim conjunct misses, the registry's falsified_by fires —
    the seat reverts to contested-VACANT in CHAMPIONS.md. `data_starved`
    (positive final-half life-span slope) is RECORDED beside a FAIL as
    context, but there is no bigger registered envelope to defer to: this IS
    the 10x re-screen, and the verdict stands.

THE 1x REFERENCE, quoted here so no run re-derives it (ledger row LC.03,
2026-08-23 21:11, VOID "fewer than two learners (1 cleared)" — wm-latent was
the one): t_null 4.65, t_twin 4.00, needs_rise +0.022, clt +92.2 s, statue
599.92 s on the 600 s ceiling, randrew t 0.21. Curves for all arms live in
experiments/artifacts/lc03_curves_seed{0,1,2}.json on the box (gitignored).

================= PROVISIONAL — WHAT THE PILOT MUST FREEZE =================

_GATES_FROZEN is False and run() REFUSES until the Kaggle throughput pilot
(seed 90, disjoint from the registered seeds; records NOTHING in the ledger)
has measured, PER CONDITION — all seven run classes appear in the pilot
record (W0.DIAG's per-condition rule; a projection from one class applied to
another is how a 9h kernel dies at hour 8):

    arm (train) / null_random / twin (frozen) / wiped / ctl_null / statue /
    randrew — decisions per second (process and wall), optimiser wiring
    (arm > 0, twin == 0), physics_finite, peak RSS, and projected hours per
    full-scale run.

FREEZE STEP (writes ONLY _KERNEL_SPLIT, _KERNEL_EST_HOURS, _GATES_FROZEN and
the PILOT RECORD into this docstring — no gate, no envelope number, no
constant may move in the same commit):

  A. If every single full-scale run projects <= 8.5 h wall inside its kernel
     (Kaggle child timeout margin, D1.0's arithmetic) AND the whole plan fits
     the free week: freeze the split — kernels of <= 4 parallel
     single-threaded workers (the Kaggle CPU allocation), each seed's arm +
     null co-located for the chaos pool — and dispatch via
     scripts/dispatch.sh LC.07.
  B. If ANY single run projects > 8.5 h wall: the CHECKPOINT BRANCH fires.
     run_survival has no mid-run checkpoint today; building one is surgery on
     experiments/survival.py (IMPL_DEPS of every LC/XL certificate) and is
     its own reviewed unit, NOT a freeze. run() keeps refusing, the branch is
     recorded in the journal and REVIEW_QUEUE, and the envelope does NOT
     shrink to fit — GPU_LONG's own requirement says checkpoint, not trim.
  C. If the total projected charge exceeds the free week's remaining hours:
     SPLIT ACROSS WEEKS (the owner's T5.01 precedent: never cut seeds to fit
     a week — split). Kernels that fit this week dispatch now; the remainder
     dispatches after the Sunday reset, each from the same frozen commit.

Kaggle P100 kernels are the backend (gpu.py submits GPU kernels only; a
CPU-only-kernel lane would need gpu.py surgery and is a decision for the
branch that needs it, not this file). One dispatch per kernel, module-cached
so run_spec's per-seed calls cannot pay twice (the 5.5-GPU-hour scar).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..drives import BASAL_B
from ..gpu import build_job, submit
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..survival import run_survival
from ..w0 import W0
from .lc_03_survival_screening import (
    E0,
    EXPLORE_STD_LC03,
    N_LIVES_MIN,
    NOISE_FLOOR_S,
    PANEL_DWELL_MAX,
    SIGMA_GATE,
    CHAOS_OCC_VOID,
    CHAOS_RATIO_VOID,
    _borrow,
    _chaos_detect,
    _final_slope,
    _panel_dwell,
    _randrew_factory,
    _tstat,
    _void,
)

IMPL_DEPS = ["experiments/tests/lc_03_survival_screening.py",
             "experiments/survival.py", "experiments/cores.py",
             "experiments/w0.py", "experiments/drives.py", "playground.py"]

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "experiments" / "artifacts"

ARM = "wm-latent"               # the seated core; the only arm on purpose
SEEDS = [0, 1, 2]

# ── THE ENVELOPE (registered; may not move) ────────────────────────────────
N_STEPS_10X = 4_000_000         # arm, wiped, null — 10x LC.03 v2's 400k
HALF_10X = N_STEPS_10X // 2     # frozen twin + controls (LC.03's halving rule)
TRACE_EVERY_10X = 80            # 4M/80 = 50k rows = the 1x evidence mass

# The seven run classes. Every one appears in the pilot record (W0.DIAG's
# per-condition rule) and every full-scale kernel job names one of them.
RUN_KEYS = ("arm", "null", "twin", "wiped", "ctl_null", "statue", "randrew")

# ── PROVISIONAL — frozen by the pilot, and ONLY these ─────────────────────
_GATES_FROZEN = False           # run() refuses until the pilot freezes below
_KERNEL_SPLIT: Optional[tuple] = None   # tuple of kernels; each kernel is a
                                        # tuple of (run_key, seed) pairs; each
                                        # seed's arm+null must share a kernel
_KERNEL_EST_HOURS: Optional[tuple] = None  # per kernel, from measured dec/s
_PILOT_ARTIFACT = "/data/lc07_pilot.json"
_PILOT_OWED = (
    "per-condition decisions/s (process+wall) for all 7 run classes; "
    "wiring: arm/wiped/randrew optimiser_steps > 0, twin == 0, statue == 0; "
    "physics_finite == 1 on every condition; "
    "peak RSS per condition and projected full-scale RSS; "
    "projected hours per full-scale run and the A/B/C branch decision"
)

PILOT_SEED = 90                 # disjoint from registered seeds
PILOT_DECISIONS = 6_000         # per condition, E0=1.0 (the real regime)
PILOT_WIPE_DECISIONS = 2_500    # wiped wiring segment at e0=0.3: force deaths


# ============================================================================
# KERNEL SIDE — runs on Kaggle. Also runs locally for smoke.
# ============================================================================

def _run_kwargs(key: str, seed: int, cal: dict,
                steps_full: int, steps_half: int, trace_every: int,
                e0: float = E0) -> dict:
    """The exact run_survival call for one run class. One place, so the pilot,
    the smoke and the full run cannot quietly diverge (per-condition rule)."""
    j0, alpha, ratios = cal["j0"], cal["alpha"], cal["ratios"]
    base = dict(j0=j0, alpha=alpha, e0=e0)
    if key == "arm":
        return dict(base, n_decisions=steps_full, policy="core", arm=ARM,
                    train=True, train_ratio=ratios[ARM], record_xy=True,
                    record_transitions=trace_every,
                    explore_std=EXPLORE_STD_LC03)
    if key == "null":
        return dict(base, n_decisions=steps_full, policy="random",
                    record_transitions=trace_every)
    if key == "twin":
        return dict(base, n_decisions=steps_half, policy="core", arm=ARM,
                    train=False, train_ratio=ratios[ARM],
                    explore_std=EXPLORE_STD_LC03)
    if key == "wiped":
        return dict(base, n_decisions=steps_full, policy="core", arm=ARM,
                    train=True, train_ratio=ratios[ARM], wipe_at_death=True,
                    explore_std=EXPLORE_STD_LC03)
    if key == "ctl_null":
        return dict(base, n_decisions=steps_half, policy="random")
    if key == "statue":
        return dict(base, n_decisions=steps_half, policy="statue")
    if key == "randrew":
        return dict(base, n_decisions=steps_half, policy="core",
                    arm="ppo-needs", train=True,
                    train_ratio=ratios["ppo-needs"],
                    explore_std=EXPLORE_STD_LC03,
                    reward_fn=_randrew_factory(seed))
    raise ValueError(f"unknown run key {key!r}")


_KEEP = ("life_gain", "mean_life_s", "n_lives", "life_spans",
         "optimiser_steps", "decisions", "sim_seconds", "process_time_s",
         "reward_sum", "thrash_l1", "needs_ok_final_third",
         "needs_ok_first_third", "physics_finite")


def _reduced(r: dict) -> dict:
    """Scalars + life_spans only. xy and transitions stay in the kernel."""
    out = {k: r[k] for k in _KEEP if k in r}
    out["params"] = r.get("params", 0.0)
    return out


def _one_full_run(args) -> tuple:
    """(key, seed, cal, steps_full, steps_half, trace) -> (key, seed, record,
    heavy) where heavy holds the in-kernel-only arrays for chaos/dwell."""
    key, seed, cal, sf, sh, tr = args
    import torch
    torch.set_num_threads(1)
    r = run_survival(seed, **_run_kwargs(key, seed, cal, sf, sh, tr))
    heavy = {}
    if key == "arm":
        heavy["xy"] = r.get("xy")
        heavy["transitions"] = r.get("transitions")
    if key == "null":
        heavy["transitions"] = r.get("transitions")
    return key, seed, _reduced(r), heavy


def _probe_world(seed: int, cal: dict):
    """obs_dim and panel_xy, exactly as LC.03 derives them."""
    w = W0(seed=seed, j0=cal["j0"], alpha=cal["alpha"])
    obs_dim = int(sum(v.shape[0] for v in w.observe().values()))
    panel_xy = (np.array(w.model.geom_pos[w.panel_gid][:2])
                if w.panel_gid >= 0 else None)
    del w
    return obs_dim, panel_xy


def _maxrss_mb() -> float:
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # linux reports KB
    return round(ru / 1024.0, 1)


def remote_run(mode: str, jobs: Optional[List] = None,
               pilot_decisions: int = PILOT_DECISIONS,
               wipe_decisions: int = PILOT_WIPE_DECISIONS) -> dict:
    """Kernel entry point. mode='pilot' (seed 90, every condition timed +
    wired) or mode='full' (jobs = [[run_key, seed], ...] at the registered
    envelope, <=4 parallel workers, chaos/dwell computed here)."""
    t0 = time.time()
    cal, prov = _borrow()
    out = {"mode": mode, "borrowed_ok": float(cal is not None),
           "provenance": {k: v for k, v in prov.items()
                          if isinstance(v, (str, int, float))}}

    def _dump():
        out["wall_minutes"] = round((time.time() - t0) / 60, 1)
        p = os.environ.get("JACK_OUT")
        if p:
            fn = "lc07_pilot.json" if mode == "pilot" else "lc07_full.json"
            json.dump(out, open(os.path.join(p, fn), "w"), indent=1)

    if cal is None:                      # refuses, never refutes — recorded
        _dump()
        return out

    if mode == "pilot":
        import torch
        torch.set_num_threads(1)
        out["conditions"] = []
        for key in RUN_KEYS:
            kw = _run_kwargs(key, PILOT_SEED, cal, pilot_decisions,
                             pilot_decisions, TRACE_EVERY_10X)
            w0_ = time.time()
            p0 = time.process_time()
            r = run_survival(PILOT_SEED, **kw)
            wall = time.time() - w0_
            proc = time.process_time() - p0
            full_target = (N_STEPS_10X if key in ("arm", "null", "wiped")
                           else HALF_10X)
            rec = {"key": key,
                   "decisions": r["decisions"],
                   "dec_per_s_wall": round(r["decisions"] / max(wall, 1e-9), 2),
                   "dec_per_s_proc": round(r["decisions"] / max(proc, 1e-9), 2),
                   "optimiser_steps": r["optimiser_steps"],
                   "physics_finite": r["physics_finite"],
                   "n_lives": r["n_lives"],
                   "maxrss_mb": _maxrss_mb(),
                   "full_scale_decisions": full_target,
                   "projected_hours_wall": round(
                       full_target / max(r["decisions"] / max(wall, 1e-9),
                                         1e-9) / 3600.0, 2)}
            out["conditions"].append(rec)
            _dump()                      # a timeout costs the last condition
            print("PILOT", key, rec["dec_per_s_wall"], "dec/s wall, proj",
                  rec["projected_hours_wall"], "h", flush=True)
        # wiped wiring: e0=0.3 forces deaths so wipe-at-death actually fires
        kwv = _run_kwargs("wiped", PILOT_SEED, cal, wipe_decisions,
                          wipe_decisions, 0, e0=0.3)
        rv = run_survival(PILOT_SEED, **kwv)
        out["wiped_wiring"] = {"n_lives": rv["n_lives"],
                               "optimiser_steps": rv["optimiser_steps"],
                               "physics_finite": rv["physics_finite"]}
        byk = {c["key"]: c for c in out["conditions"]}
        out["wiring_ok"] = float(
            byk["arm"]["optimiser_steps"] > 0
            and byk["wiped"]["optimiser_steps"] > 0
            and byk["randrew"]["optimiser_steps"] > 0
            and byk["twin"]["optimiser_steps"] == 0
            and byk["statue"]["optimiser_steps"] == 0
            and out["wiped_wiring"]["n_lives"] >= 2
            and min(c["physics_finite"] for c in out["conditions"]) == 1.0)
        _dump()
        return out

    # ── mode == "full" ─────────────────────────────────────────────────
    import multiprocessing as mp
    assert jobs, "full mode needs jobs=[[run_key, seed], ...]"
    tasks = [(k, s, cal, N_STEPS_10X, HALF_10X, TRACE_EVERY_10X)
             for k, s in jobs]
    out["runs"] = {}
    heavy_store: Dict[tuple, dict] = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(4, len(tasks)), initializer=_pool_init) as pool:
        for key, seed, rec, heavy in pool.imap_unordered(_one_full_run, tasks):
            out["runs"][f"{key}/{seed}"] = rec
            if heavy:
                heavy_store[(key, seed)] = heavy
            _dump()                      # partial dump per finished run
            print("RUN", key, "seed", seed, "life_gain",
                  round(rec["life_gain"], 2), flush=True)

    # chaos + dwell, per seed, only where this kernel holds both pools
    out["per_seed"] = {}
    for seed in sorted({s for _, s in jobs}):
        entry: dict = {}
        arm_h = heavy_store.get(("arm", seed))
        null_h = heavy_store.get(("null", seed))
        if arm_h is not None:
            obs_dim, panel_xy = _probe_world(seed, cal)
            entry["panel_dwell"] = _panel_dwell(arm_h["xy"], panel_xy)
            if null_h is not None:
                pool_ = {ARM: arm_h["transitions"],
                         "null_random": null_h["transitions"]}
                chaos = _chaos_detect(pool_, "null_random", seed, obs_dim)
                entry["chaos_occupancy"] = chaos[ARM]["occupancy"]
                entry["chaos_ratio"] = chaos[ARM]["ratio"]
            else:
                entry["chaos_missing"] = 1.0   # the split violated the
                # co-location constraint — loud, and _check will VOID on it
        if entry:
            out["per_seed"][str(seed)] = entry
        _dump()
    return out


def _pool_init():
    import torch
    torch.set_num_threads(1)


# ============================================================================
# HOST SIDE — submission, experiment, check.
# ============================================================================

_PILOT_JOB = r'''
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
from experiments.tests.lc_07_scale_transfer import remote_run
remote_run("pilot")
'''

_FULL_JOB = r'''
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "mujoco"], check=True)
from experiments.tests.lc_07_scale_transfer import remote_run
remote_run("full", jobs=__JOBS__)
'''


def pilot(out_path: str = _PILOT_ARTIFACT):
    """Dispatch the throughput/wiring pilot to Kaggle; write its artifact.
    Every one of the 7 run classes is timed (per-condition rule). DO NOT
    dispatch while another spec's watcher holds the GPU lock."""
    job = build_job(_PILOT_JOB)
    res = submit(job, prefer="kaggle", est_hours=1.0, timeout_s=5400,
                 fetch=["lc07_pilot.json"])
    if not res.ok:
        raise RuntimeError(f"pilot failed on {res.backend}: {res.message}")
    path = res.artifacts.get("lc07_pilot.json")
    if not path:
        raise RuntimeError(f"no pilot artifact. message={res.message!r} "
                           f"stdout_tail={res.stdout[-400:]!r}")
    d = json.loads(Path(path).read_text())
    d["backend"] = res.backend
    Path(out_path).write_text(json.dumps(d, indent=1))
    print("PILOT ARTIFACT", out_path)
    for c in d.get("conditions", []):
        print(f"  {c['key']:9s} {c['dec_per_s_wall']:9.1f} dec/s  "
              f"proj {c['projected_hours_wall']:6.2f} h  "
              f"rss {c['maxrss_mb']} MB")
    print("  wiring_ok", d.get("wiring_ok"))
    return d


_CACHE: dict = {}


def _submit_full() -> dict:
    if _KERNEL_SPLIT is None or _KERNEL_EST_HOURS is None:
        raise RuntimeError(
            "LC.07 _KERNEL_SPLIT is not frozen — run the pilot, freeze the "
            "split from its measured per-condition dec/s (branch A of the "
            "docstring's decision tree), then submit. Submitting on an "
            "estimate is how a 9h kernel dies at hour 8.")
    merged = {"runs": {}, "per_seed": {}, "kernels": []}
    for kern, est in zip(_KERNEL_SPLIT, _KERNEL_EST_HOURS):
        body = _FULL_JOB.replace("__JOBS__",
                                 repr([[k, s] for k, s in kern]))
        job = build_job(body)
        res = submit(job, prefer="kaggle", est_hours=min(est, 8.8),
                     timeout_s=32000, fetch=["lc07_full.json"])
        if not res.ok:
            raise RuntimeError(f"kernel {kern} failed on {res.backend}: "
                               f"{res.message}")
        path = res.artifacts.get("lc07_full.json")
        if not path:
            raise RuntimeError(f"no artifact from kernel {kern}: "
                               f"{res.message!r}")
        d = json.loads(Path(path).read_text())
        if d.get("borrowed_ok") != 1.0:
            raise RuntimeError(f"kernel {kern} could not calibrate: "
                               f"{d.get('provenance')}")
        merged["runs"].update(d["runs"])
        for s, e in d.get("per_seed", {}).items():
            merged["per_seed"].setdefault(s, {}).update(e)
        merged["kernels"].append({"jobs": [[k, s] for k, s in kern],
                                  "wall_minutes": d["wall_minutes"],
                                  "backend": res.backend})
    return merged


def _experiment(seed: int) -> dict:
    if not _CACHE:
        _CACHE.update(_submit_full())
    runs, per_seed = _CACHE["runs"], _CACHE["per_seed"].get(str(seed), {})

    def r(key: str) -> dict:
        return runs[f"{key}/{seed}"]

    arm, null, twin, wiped = r("arm"), r("null"), r("twin"), r("wiped")
    m = {
        "borrowed_ok": 1.0,              # _submit_full raised otherwise
        "null_life_gain": null["life_gain"],
        "null_mean_life_s": null["mean_life_s"],
        "null_lives_ok": float(null["n_lives"] >= N_LIVES_MIN),
        f"{ARM}/life_gain": arm["life_gain"],
        f"{ARM}/mean_life_s": arm["mean_life_s"],
        f"{ARM}/n_lives": arm["n_lives"],
        f"{ARM}/lives_ok": float(arm["n_lives"] >= N_LIVES_MIN),
        f"{ARM}/lg_margin_null": arm["life_gain"] - null["life_gain"],
        f"{ARM}/lg_margin_twin": arm["life_gain"] - twin["life_gain"],
        f"{ARM}/twin_life_gain": twin["life_gain"],
        f"{ARM}/wiped_life_gain": wiped["life_gain"],
        f"{ARM}/clt": arm["life_gain"] - wiped["life_gain"],
        f"{ARM}/needs_rise": (arm["needs_ok_final_third"]
                              - arm["needs_ok_first_third"]),
        f"{ARM}/final_slope": _final_slope(arm["life_spans"]),
        f"{ARM}/optimiser_steps": arm["optimiser_steps"],
        f"{ARM}/decisions": arm["decisions"],
        f"{ARM}/core_s": arm["process_time_s"],
        "physics_finite_min": float(min(
            runs[f"{k}/{seed}"]["physics_finite"] for k in RUN_KEYS)),
    }
    m[f"{ARM}/panel_dwell"] = per_seed.get("panel_dwell", -1.0)
    m[f"{ARM}/dwell_ok"] = float(
        0.0 <= m[f"{ARM}/panel_dwell"] <= PANEL_DWELL_MAX)
    m["chaos_missing"] = per_seed.get("chaos_missing", 0.0)
    occ = per_seed.get("chaos_occupancy", 0.0)
    rat = per_seed.get("chaos_ratio", 0.0)
    m[f"{ARM}/chaos_occupancy"] = occ
    m[f"{ARM}/chaos_ratio"] = rat
    m[f"{ARM}/chaos_ok"] = float(not (occ >= CHAOS_OCC_VOID
                                      and rat >= CHAOS_RATIO_VOID))

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"lc07_curves_seed{seed}.json").write_text(json.dumps(
        {"seed": seed, "e0": E0, "n_steps": N_STEPS_10X,
         "runs": {k: runs[f"{k}/{seed}"] for k in RUN_KEYS}}, indent=1))
    return m


def _control(seed: int) -> dict:
    runs = _CACHE["runs"]

    def r(key: str) -> dict:
        return runs[f"{key}/{seed}"]

    ceiling = E0 / BASAL_B
    return {
        "borrowed_ok": 1.0,
        "ctrl_null_life_gain": r("ctl_null")["life_gain"],
        "statue_mean_life_s": r("statue")["mean_life_s"],
        "statue_ceiling_ok": float(
            abs(r("statue")["mean_life_s"] - ceiling) <= 0.10 * ceiling),
        "randrew_life_gain": r("randrew")["life_gain"],
        "randrew_margin": (r("randrew")["life_gain"]
                           - r("ctl_null")["life_gain"]),
        "randrew_opt_steps": r("randrew")["optimiser_steps"],
    }


def _check(m: dict, c: dict):
    # ── instrument validity, every branch named ────────────────────────
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")
    if m.get("physics_finite_min", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    if m.get("null_lives_ok", 0.0) != 1.0:
        return _void(m, "null under 12 lives — the world cannot produce "
                        "N_LIVES_MIN at this envelope; a world problem")
    if m.get("chaos_missing", 0.0) != 0.0:
        return _void(m, "chaos pool split across kernels — the freeze "
                        "violated the arm+null co-location constraint")
    if m.get(f"{ARM}/panel_dwell", -1.0) < 0.0:
        return _void(m, "panel_dwell missing from the kernel record")

    # ── controls, each on its pre-registered side ──────────────────────
    if c.get("statue_ceiling_ok", 0.0) != 1.0:
        return _void(m, f"control (a): statue off the basal ceiling "
                        f"(mean_life {c.get('statue_mean_life_s', 0.0):.2f} s "
                        f"vs ceiling {E0 / BASAL_B:.1f} s +-10%)")
    if _tstat(c, "randrew_margin") >= SIGMA_GATE:
        return _void(m, f"control (c): randrew cleared the null gate "
                        f"(t {_tstat(c, 'randrew_margin'):.2f})")
    for kind in ("twin_life_gain", "wiped_life_gain"):
        k = f"{ARM}/{kind}"
        if (abs(_tstat(m, k)) >= SIGMA_GATE
                and abs(m.get(k, 0.0)) > NOISE_FLOOR_S):
            return _void(m, f"control: {k} = {m.get(k, 0.0):.2f} s, "
                            f"|t| = {abs(_tstat(m, k)):.2f} — lives moved "
                            f"without a persistent learner")
    if m.get(f"{ARM}/chaos_ok", 0.0) != 1.0:
        return _void(m, f"chaos conjunction fired on {ARM}: occupancy "
                        f"{m.get(f'{ARM}/chaos_occupancy', 0.0):.2f}, ratio "
                        f"{m.get(f'{ARM}/chaos_ratio', 0.0):.2f}")

    # ── the claim: the six conjuncts wm-latent cleared at 1x ───────────
    conj = {
        "t_null>=3": _tstat(m, f"{ARM}/lg_margin_null") >= SIGMA_GATE,
        "t_twin>=3": _tstat(m, f"{ARM}/lg_margin_twin") >= SIGMA_GATE,
        "lives_ok": m.get(f"{ARM}/lives_ok", 0.0) == 1.0,
        "needs_rise>0": m.get(f"{ARM}/needs_rise", -1.0) > 0.0,
        "clt>0": m.get(f"{ARM}/clt", -1.0) > 0.0,
        "dwell_ok": m.get(f"{ARM}/dwell_ok", 0.0) == 1.0,
    }
    m[f"{ARM}/data_starved"] = float(
        not all(conj.values()) and m.get(f"{ARM}/final_slope", 0.0) > 0.0)
    if all(conj.values()):
        m["verdict"] = (
            f"PASS — wm-latent re-cleared its 1x gates at 10x scale: "
            f"t_null {_tstat(m, f'{ARM}/lg_margin_null'):.2f}, "
            f"t_twin {_tstat(m, f'{ARM}/lg_margin_twin'):.2f} (bar "
            f"{SIGMA_GATE}), needs_rise {m.get(f'{ARM}/needs_rise', 0.0):+.4f}, "
            f"clt {m.get(f'{ARM}/clt', 0.0):+.1f} s. The BY VERDICT seating "
            f"survives the owner's scale-transfer guard (D10/D12).")
        return True
    failed = [k for k, v in conj.items() if not v]
    m["verdict"] = (
        f"FAIL — gate(s) missed at 10x: {', '.join(failed)} "
        f"(t_null {_tstat(m, f'{ARM}/lg_margin_null'):.2f}, "
        f"t_twin {_tstat(m, f'{ARM}/lg_margin_twin'):.2f}, "
        f"needs_rise {m.get(f'{ARM}/needs_rise', 0.0):+.4f}, "
        f"clt {m.get(f'{ARM}/clt', 0.0):+.1f}, "
        f"dwell {m.get(f'{ARM}/panel_dwell', -1.0):.3f}). "
        f"data_starved={int(m[f'{ARM}/data_starved'])} (recorded context; "
        f"this IS the 10x re-screen — no bigger registered envelope exists). "
        f"Per the registry: the 1x verdict was a small-envelope artifact and "
        f"the learning-core seat reverts to contested-VACANT in CHAMPIONS.md.")
    return False


def run(ledger: Ledger | None = None):
    if not _GATES_FROZEN:
        raise RuntimeError(
            "LC.07 gates are provisional — the Kaggle pilot has not frozen "
            "_KERNEL_SPLIT / _KERNEL_EST_HOURS from measured per-condition "
            "throughput. Run the pilot (python -m experiments.tests."
            "lc_07_scale_transfer pilot — ONLY when no other spec's watcher "
            "holds the GPU lock), take the docstring's A/B/C branch, freeze, "
            "then run (SM.02's _GATES_FROZEN idiom).")
    return run_spec(BY_ID["LC.07"], _experiment, _check, control_fn=_control,
                    ledger=ledger or Ledger())


# ============================================================================
# LOCAL — smoke.
# ============================================================================

def _smoke():
    """Minutes-long local mechanics check at a tiny envelope. Records
    nothing; asserts on the product: all 7 run classes construct and run,
    wiring invariants hold, chaos/dwell compute on real (tiny) pools, and
    _check's aggregation wiring executes end to end."""
    import torch
    torch.set_num_threads(2)
    cal, prov = _borrow()
    assert cal is not None, f"borrow refused: {prov}"
    # e0=0.15 => basal life ~= 0.15/BASAL_B sim-s ~= 450 decisions, so 1200
    # decisions guarantees the wiped run crosses >= 2 deaths (the wiring the
    # smoke exists to assert). e0=0.3's ~900-decision lives would not.
    steps, half, trace = 1200, 1200, 4
    seed = PILOT_SEED
    runs: Dict[str, dict] = {}
    heavy: Dict[str, dict] = {}
    for key in RUN_KEYS:
        kw = _run_kwargs(key, seed, cal, steps, half, trace, e0=0.15)
        r = run_survival(seed, **kw)
        runs[key] = _reduced(r)
        if key in ("arm", "null"):
            heavy[key] = {"xy": r.get("xy"),
                          "transitions": r.get("transitions")}
        assert runs[key]["physics_finite"] == 1.0, f"{key}: physics blew up"
    assert runs["arm"]["optimiser_steps"] > 0, "arm never trained"
    assert runs["wiped"]["optimiser_steps"] > 0, "wiped never trained"
    assert runs["randrew"]["optimiser_steps"] > 0, "randrew never trained"
    assert runs["twin"]["optimiser_steps"] == 0, "frozen twin trained"
    assert runs["statue"]["optimiser_steps"] == 0, "statue trained?!"
    assert runs["wiped"]["n_lives"] >= 2, "wiped wiring never crossed a death"

    obs_dim, panel_xy = _probe_world(seed, cal)
    dwell = _panel_dwell(heavy["arm"]["xy"], panel_xy)
    assert 0.0 <= dwell <= 1.0, f"dwell out of range: {dwell}"
    pool_ = {ARM: heavy["arm"]["transitions"],
             "null_random": heavy["null"]["transitions"]}
    chaos = _chaos_detect(pool_, "null_random", seed, obs_dim)
    assert ARM in chaos and np.isfinite(chaos[ARM]["occupancy"])

    # end-to-end host wiring on the smoke product (no submission, no ledger)
    _CACHE.clear()
    _CACHE["runs"] = {f"{k}/{s}": dict(runs[k]) for k in RUN_KEYS
                      for s in SEEDS}
    _CACHE["per_seed"] = {str(s): {"panel_dwell": dwell,
                                   "chaos_occupancy": chaos[ARM]["occupancy"],
                                   "chaos_ratio": chaos[ARM]["ratio"]}
                          for s in SEEDS}
    _CACHE["kernels"] = [{"jobs": "smoke", "wall_minutes": 0.0}]
    ms = [_experiment(s) for s in SEEDS]
    cs = [_control(s) for s in SEEDS]
    from ..protocol import _aggregate
    verdict = _check(_aggregate(ms), _aggregate(cs))
    # identical seeds => zero spread => t-stats explode or collapse; the smoke
    # asserts the PLUMBING ran and named a branch, not any particular verdict.
    _CACHE.clear()
    for f in (ARTIFACTS / "lc07_curves_seed0.json",
              ARTIFACTS / "lc07_curves_seed1.json",
              ARTIFACTS / "lc07_curves_seed2.json"):
        f.unlink(missing_ok=True)        # smoke must not leave fake curves
    print(json.dumps({"dwell": round(dwell, 4),
                      "chaos_occ": round(chaos[ARM]["occupancy"], 3),
                      "arm_opt_steps": runs["arm"]["optimiser_steps"],
                      "twin_opt_steps": runs["twin"]["optimiser_steps"],
                      "wiped_lives": runs["wiped"]["n_lives"],
                      "check_ran": str(verdict)}, indent=1))
    print("SMOKE OK")


if __name__ == "__main__":
    # `pilot` — dispatch the Kaggle throughput pilot (push first; NEVER while
    #           another spec's watcher holds the GPU lock).
    # `smoke` — local CPU mechanics check, minutes. Run before paying quota.
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "pilot":
        pilot()
    elif cmd == "smoke":
        _smoke()
    else:
        print(f"unknown command {cmd!r}: use pilot|smoke")
        sys.exit(2)
