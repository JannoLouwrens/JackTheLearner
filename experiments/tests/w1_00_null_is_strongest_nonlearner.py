"""W1.00 — The null is the strongest process that has not learned.

THE QUESTION. W0.DIAG measured the mechanism behind `w0-too-shallow`'s Pile A:
same-marginal temporally-correlated action noise buys life in W0 (gain_up
12.12 +- 1.20 where the stationary white null reads 0.0095 +- 0.39, eats 1.0
vs 0.33), so every recorded reading of the form "the null does as well as the
learner" was taken against a null strictly weaker than an unlearned process
can be. This spec asks whether that under-nulling is MATERIAL: select the
honest null — the best-scoring member of {stationary white, correlated
colored at the W0.DIAG schedule, repeat-action} ON THE NULL'S OWN OUTCOME,
never on a claim arm's — and re-score the four recorded Pile A findings
against it. PASS iff at least one recorded margin moves by more than its own
std. Registered 2026-09-06 from the Review FULL's published W1 design
(`w0-too-shallow`, REVIEW_QUEUE.md); implemented the same day.

THE VENUE'S OWN OUTCOME, declared: mean completed-life span (`mean_life_s`)
at the W0.DIAG envelope. W0 is a survival world and survival is what it
scores; `life_gain` is a within-run TREND channel — a property of learners,
which a stationary null holds at zero by V5's own certification — so
selecting on it would select on the claim's channel, the exact practice the
control below exists to expose. W0.DIAG's recorded numbers predict the
selection is a real contest (white 41.23 s, colored-up 52.53 s, repeat ~56 s
at the v1 sizing), and this spec does not assume its winner.

THE FRESH RUNS — only at the W0.DIAG envelope (registered constraint 2; LC.03
is read from its RECORDED row, zero lives at its envelope). Four processes
per seed, all policies random, nothing learns, same world per seed (same
seed => same food layout; deltas are PAIRED per seed):

  white    per-decision uniform actions, rng stream seed*6553+11.
  white2   the same law on stream seed*6553+13 — the INSTRUMENT FLOOR: the
           registered null_baseline says the metric's floor is the seed
           spread of fresh null runs, and white2-vs-white measures it in
           every channel with the same arithmetic as the claim.
  repeat   survival.py's random-repeat law verbatim (hold each action
           HOLD_K=5 decisions).
  colored  W0.DIAG's up-schedule AR(1)-latent-through-erf construction
           verbatim: tau ramped geometrically TAU[0]->TAU[1] across the run,
           exactly Uniform(-1,1) marginals at every decision (the T3.06
           matched-magnitude lesson). Its life_gain is the schedule's trend —
           which is the point: a process that has NOT learned produces the
           trend the life_gain channel scores learners on.

  The loop mirrors w0_diag._scheduled_run decision for decision (that loop is
  the accepted mirror of run_survival's null branch); white here is exactly
  its `ka` branch with constant respawn energy, repeat adds survival.py's
  hold, colored is its `up` branch. Envelope constants (E0, N_DECISIONS=4800,
  TAU, floors) are IMPORTED from w0_diag, not copied.

CHANNELS measured per run, and the venue analogue each finding is re-scored
in (T3.06's grid arithmetic copied verbatim — CELL_M 0.5, 22x22 over
[-5.5, 5.5]^2, 484 cells; W0's arena is 6.0 m so edge positions clip to edge
cells, stated plainly):

  lg    life_gain (LC.03's certified channel, w0_diag arithmetic verbatim).
  cov   mean over completed lives of (cells visited / 484) — T3.06's
        per-life coverage, taken on W0's rover instead of T2.08's.
  dw    max over completed lives of the fraction of decisions spent in a
        food-bearing cell — the venue analogue of T3.06's goal-cell dwell
        (W0 has no goal cell; food is its attractor). Post-death respawn
        positions belong to the next life.

THE EIGHT RECORDED PILE A MARGINS, read at run time from ledger.json with
their rows PINNED by ran_at (a moved row VOIDs, it does not silently
re-anchor). Expected values quoted here for the reader; the run re-reads
them:

  LC.03 attempt 3 (VOID, ran_at 2026-08-23T21:11:17), channel lg:
    darkroom_margin            -53.46 +- 85.58   (control_metrics)
    wm-latent/lg_margin_null  +100.10 +- 37.33
    wm-efe/lg_margin_null      +98.19 +- 82.81
    ppo-needs/lg_margin_null   +67.42 +- 110.10
    ppo-lp/lg_margin_null      +60.17 +- 86.82
    dreamer-xs/lg_margin_null  -45.27 +- 83.60
  T3.06 attempt 1 (VOID, ran_at 2026-08-30T01:06:21):
    wk5 coverage margin (channel cov): coverage_curious - coverage_random
      = +0.0124, std by quadrature 0.0550 (the field-watch wk5 arithmetic,
      re-derived from the committed row, se 0.0317 * sqrt(3) agrees)
    dwell margin (channel dw): task_dwell_worst_life -
      random_dwell_worst_life = +0.0949, std by quadrature 0.0062

TRANSPORT, stated plainly rather than hidden: the fresh deltas are measured
at the W0.DIAG envelope (~41-56 s lives) and the recorded margins were
measured at their own envelopes (LC.03: 600 s ceiling; T3.06: 4000-decision
lives on the static-panel rig). The re-score therefore asks "does the
measured null-upgrade delta, in the finding's own channel, exceed the
finding's recorded seed std" — it does not claim the delta transports
unchanged, and the registered spec's own design (recorded rows + fresh runs
at one envelope only) admits no stronger arithmetic. A PASS orders the
stronger null for FUTURE W-venue registrations (the kills clause); it
re-opens nothing.

PRE-REGISTERED GATES — every bar a t-statistic over seeds (house _tstat,
mean*sqrt(3)/std) or a shift-over-recorded-std; SIGMA_GATE = 3.0 imported.
VOID lanes in test order (an instrument fault is not a world reading, T0.22):

  V1 borrow of PS.01's (j0_ms, alpha) unavailable    "uncalibrated borrow"
  V2 a pinned recorded row is missing, moved (ran_at mismatch) or lacks a
     required field                                  "recorded inputs moved"
  V3 non-finite physics in any fresh run             "non-finite physics"
  V4 completed lives under floor: white/white2/repeat >= MIN_LIVES_NULL
     (12), colored >= MIN_LIVES_SCHED (8) — W0.DIAG's floors unchanged
                                                     "under minimum lives"
  V5 a stationary process trends: |t(gain_white)|, |t(gain_repeat)| or
     |t(gain_white2)| >= 3 — the world drifts within runs and no re-score
     attributes                                      "stationary null trends"
  V6 the self-rescore is nonzero: white re-scored against itself must give
     exactly 0.0 in every channel (arithmetic identity; a nonzero here is a
     code fault, not a measurement)                  "self-rescore nonzero"

SELECTION AND ITS CONTROL (two-sided, both printed on the row):
  by-outcome: strongest = argmax over {white, colored, repeat} of mean
    mean_life_s; sel_real requires t(paired d_ml of the winner vs white)
    >= 3 — an argmax by noise is not a selection.
  by-claim (the OLD practice, run to expose it): per channel, the null a
    claim-fitter would pick — argmax of the null's own CHANNEL score (that
    is the choice that minimises a claim margin). `selection_divergence` =
    1.0 iff any channel's by-claim pick differs from the by-outcome pick.
    Recorded, never smoothed over.

THE CLAIM (branch named in `claim_branch` either way, BA.03's lesson):
  FAIL (a) "white is already the strongest null": argmax is white, or the
    winner is not 3-sigma real vs white. W0.DIAG's result does not
    generalise past food-seeking; Pile A dissolves.
  Else, for each finding j with recorded (margin_j, std_j) and channel c:
    shift_j    = |mean paired (selected - white) delta in c|
    real_j     = |t(paired delta in c)| >= 3        (the shift is not noise)
    floor_ok_j = |mean (white2 - white) delta in c| <= std_j
                                     (the shift exceeds the instrument floor)
    fired_j    = shift_j > std_j AND real_j AND floor_ok_j
  PASS  any fired_j — the under-nulling is material; every W-venue spec
        registered after this row states its null as the selected process
        (kills clause; nothing that already PASSed re-opens).
  FAIL (b) "immaterial": a stronger null exists but no recorded margin moves
        by more than its own std under the three conjuncts.
  metric: max_pile_a_margin_shift_over_own_std = max_j shift_j/std_j,
  printed with every per-finding ratio.

COST: 12 runs x ~144 s = ~29 core-min; ~10 min wall on the 3-worker pool
(w0_diag's pattern). cpu<10min by the coverage class that named this spec
fillable today.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..survival import HOLD_K
from ..w0 import W0, random_action
from .w0_diag_exploration_reaches_food import (
    E0, MIN_LIVES_NULL, MIN_LIVES_SCHED, N_ACT, N_DECISIONS, SIGMA_GATE, TAU,
    _life_gain)

IMPL_DEPS = ["experiments/w0.py", "experiments/drives.py",
             "experiments/survival.py", "playground.py",
             "experiments/tests/w0_diag_exploration_reaches_food.py"]

# T3.06's grid, verbatim (its rig constants are T2.08's; copying rather than
# importing keeps this module free of that module's import-time rig).
CELL_M = 0.5
GRID_LO, GRID_HI = -5.5, 5.5
GRID_N = int(round((GRID_HI - GRID_LO) / CELL_M))     # 22
N_CELLS = GRID_N * GRID_N                              # 484

_SQRT2 = math.sqrt(2.0)
PROCS = ("white", "white2", "repeat", "colored")
PILOT_SEED = 90

# pinned recorded rows — a moved row VOIDs (V2), never silently re-anchors
LC03_RAN_AT = "2026-08-23T21:11:17"
T306_RAN_AT = "2026-08-30T01:06:21"
LC03_ARMS = ("wm-latent", "wm-efe", "ppo-needs", "ppo-lp", "dreamer-xs")


def _cell(x: float, y: float) -> int:
    cx = min(GRID_N - 1, max(0, int((x - GRID_LO) / CELL_M)))
    cy = min(GRID_N - 1, max(0, int((y - GRID_LO) / CELL_M)))
    return cy * GRID_N + cx


def _borrow():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if b.values is None:
        return None, None, 0.0
    return b.values["j0_ms"], b.values["alpha"], 1.0


def _recorded() -> Optional[dict]:
    """The eight Pile A margins, read from the ledger with rows pinned by
    ran_at. Returns None when anything is missing or moved (V2)."""
    path = Path(__file__).resolve().parents[1] / "ledger.json"
    try:
        res = json.loads(path.read_text())["results"]
        lc, t3 = res["LC.03"], res["T3.06"]
        if lc["ran_at"] != LC03_RAN_AT or t3["ran_at"] != T306_RAN_AT:
            return None
        lcm, lcc, t3m = lc["metrics"], lc["control_metrics"], t3["metrics"]
        out = {"darkroom": (lcc["darkroom_margin"], lcc["darkroom_margin_std"],
                            "lg")}
        for arm in LC03_ARMS:
            out[arm.replace("-", "_")] = (lcm[f"{arm}/lg_margin_null"],
                                          lcm[f"{arm}/lg_margin_null_std"],
                                          "lg")
        out["wk5_coverage"] = (
            t3m["coverage_curious"] - t3m["coverage_random"],
            math.hypot(t3m["coverage_curious_std"], t3m["coverage_random_std"]),
            "cov")
        out["dwell"] = (
            t3m["task_dwell_worst_life"] - t3m["random_dwell_worst_life"],
            math.hypot(t3m["task_dwell_worst_life_std"],
                       t3m["random_dwell_worst_life_std"]),
            "dw")
        return out
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _rollout(seed: int, j0: float, alpha: float, proc: str,
             n_decisions: int = N_DECISIONS) -> dict:
    """One process at the W0.DIAG envelope. Mirrors w0_diag._scheduled_run
    decision for decision; only the action law differs by `proc`."""
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True)
    w.drives.state = drives.DriveState(e=E0)
    rng = np.random.RandomState(seed * 6553 + (13 if proc == "white2" else 11))
    z = rng.randn(N_ACT)
    held_a, held_left = None, 0
    food_cells: Optional[set] = None   # read after physics is live (1st step)
    lives_cov: list = []
    lives_dw: list = []
    visited: set = set()
    in_food = 0
    n_dec = 0
    t0 = time.perf_counter()

    for k in range(n_decisions):
        frac = k / max(1, n_decisions - 1)
        if proc in ("white", "white2"):
            a = random_action(rng)
        elif proc == "repeat":
            if held_left == 0:
                held_a, held_left = random_action(rng), HOLD_K
            a, held_left = held_a, held_left - 1
        else:  # colored — w0_diag's `up` branch, verbatim
            tau = TAU[0] * (TAU[1] / TAU[0]) ** frac
            rho = math.exp(-1.0 / tau)
            z = rho * z + math.sqrt(1.0 - rho * rho) * rng.randn(N_ACT)
            a = np.array([math.erf(v / _SQRT2) for v in z])

        w.decide(a)
        if food_cells is None:
            food_cells = {_cell(float(w.data.geom_xpos[gid][0]),
                                float(w.data.geom_xpos[gid][1]))
                          for gid, _ in w.drives._food.values()}
        if w.died_this_decision:
            # the post-decide position is the respawn pose — next life's
            if n_dec:
                lives_cov.append(len(visited) / N_CELLS)
                lives_dw.append(in_food / n_dec)
            visited, in_food, n_dec = set(), 0, 0
            w.drives.state = drives.DriveState(e=E0)
        else:
            xy = w.data.xpos[w.rover_bid][:2]
            c = _cell(float(xy[0]), float(xy[1]))
            visited.add(c)
            n_dec += 1
            if c in food_cells:
                in_food += 1

    spans = list(w.life_lengths)
    return {
        "lg": _life_gain(spans),
        "ml": float(np.mean(spans)) if spans else 0.0,
        "n_lives": float(len(spans)),
        "cov": float(np.mean(lives_cov)) if lives_cov else 0.0,
        "dw": float(np.max(lives_dw)) if lives_dw else 0.0,
        "ate": float(sum(w.drives.ate_total.values())),
        "physics_finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                     and np.all(np.isfinite(w.data.qvel)))),
        "wall_s": float(time.perf_counter() - t0),
    }


def _run_task(task) -> tuple:
    seed, proc = task
    j0, alpha, ok = _borrow()
    if not ok:
        return seed, proc, {"borrowed_ok": 0.0}
    out = _rollout(seed, j0, alpha, proc)
    out["borrowed_ok"] = 1.0
    return seed, proc, out


_CACHE: Dict[int, Dict[str, dict]] = {}


def _bundle(seed: int) -> Dict[str, dict]:
    if seed not in _CACHE:
        _CACHE[seed] = {p: _run_task((seed, p))[2] for p in PROCS}
    return _CACHE[seed]


def _claim_metrics(b: Dict[str, dict]) -> dict:
    if any(r.get("borrowed_ok", 0.0) != 1.0 for r in b.values()):
        return {"borrowed_ok": 0.0}
    rec = _recorded()
    if rec is None:
        return {"borrowed_ok": 1.0, "recorded_ok": 0.0}
    lives_ok = float(
        all(b[p]["n_lives"] >= MIN_LIVES_NULL
            for p in ("white", "white2", "repeat"))
        and b["colored"]["n_lives"] >= MIN_LIVES_SCHED)
    m = {
        "borrowed_ok": 1.0,
        "recorded_ok": 1.0,
        "physics_finite_min": float(min(r["physics_finite"]
                                        for r in b.values())),
        "lives_ok": lives_ok,
        # venue outcome, per process, and paired deltas vs white
        "ml_white": b["white"]["ml"],
        "ml_colored": b["colored"]["ml"],
        "ml_repeat": b["repeat"]["ml"],
        "d_ml_colored": b["colored"]["ml"] - b["white"]["ml"],
        "d_ml_repeat": b["repeat"]["ml"] - b["white"]["ml"],
        # V5 flatness reads
        "gain_white": b["white"]["lg"],
        "gain_repeat": b["repeat"]["lg"],
        # channel deltas vs white, paired per seed
        "d_lg_colored": b["colored"]["lg"] - b["white"]["lg"],
        "d_lg_repeat": b["repeat"]["lg"] - b["white"]["lg"],
        "d_cov_colored": b["colored"]["cov"] - b["white"]["cov"],
        "d_cov_repeat": b["repeat"]["cov"] - b["white"]["cov"],
        "d_dw_colored": b["colored"]["dw"] - b["white"]["dw"],
        "d_dw_repeat": b["repeat"]["dw"] - b["white"]["dw"],
        "cov_white": b["white"]["cov"],
        "dw_white": b["white"]["dw"],
        "eats_white": b["white"]["ate"],
        "eats_colored": b["colored"]["ate"],
        "lives_white": b["white"]["n_lives"],
        "lives_colored": b["colored"]["n_lives"],
        "lives_repeat": b["repeat"]["n_lives"],
    }
    # the recorded margins ride on the row (seed-constant), so the re-score's
    # inputs are auditable from the ledger entry itself
    for name, (margin, std, chan) in rec.items():
        m[f"rec_margin_{name}"] = float(margin)
        m[f"rec_std_{name}"] = float(std)
    return m


def _control_metrics(b: Dict[str, dict]) -> dict:
    if any(r.get("borrowed_ok", 0.0) != 1.0 for r in b.values()):
        return {"borrowed_ok": 0.0}
    return {
        "borrowed_ok": 1.0,
        # the instrument floor: an independent draw of the same null law
        "gain_white2": b["white2"]["lg"],
        "f_lg": b["white2"]["lg"] - b["white"]["lg"],
        "f_cov": b["white2"]["cov"] - b["white"]["cov"],
        "f_dw": b["white2"]["dw"] - b["white"]["dw"],
        "ml_white2": b["white2"]["ml"],
        "lives_white2": b["white2"]["n_lives"],
        # the self-rescore identity (V6): white against itself, every channel
        "self_lg": b["white"]["lg"] - b["white"]["lg"],
        "self_cov": b["white"]["cov"] - b["white"]["cov"],
        "self_dw": b["white"]["dw"] - b["white"]["dw"],
    }


def _experiment(seed: int) -> dict:
    return _claim_metrics(_bundle(seed))


def _control(seed: int) -> dict:
    return _control_metrics(_bundle(seed))


def _tstat(m: dict, key: str) -> float:
    return m.get(key, 0.0) * math.sqrt(3) / max(m.get(f"{key}_std", 0.0), 1e-9)


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # V1-V4: the rig
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")
    if m.get("recorded_ok", 0.0) != 1.0:
        return _void(m, "recorded inputs moved")
    if m.get("physics_finite_min", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    if m.get("lives_ok", 0.0) != 1.0:
        return _void(m, "under minimum lives")
    # V5: stationary processes must be flat in the trend channel
    if (abs(_tstat(m, "gain_white")) >= SIGMA_GATE
            or abs(_tstat(m, "gain_repeat")) >= SIGMA_GATE
            or abs(_tstat(c, "gain_white2")) >= SIGMA_GATE):
        return _void(m, "stationary null trends within runs")
    # V6: the self-rescore identity
    if any(c.get(k, 0.0) != 0.0 for k in ("self_lg", "self_cov", "self_dw")):
        return _void(m, "self-rescore nonzero: arithmetic fault")

    # SELECTION — by the null's own venue outcome
    by_outcome = max(("white", "colored", "repeat"),
                     key=lambda p: m.get(f"ml_{p}", 0.0))
    m["selected_null"] = by_outcome
    sel_real = (by_outcome != "white"
                and _tstat(m, f"d_ml_{by_outcome}") >= SIGMA_GATE)
    # the two-sided control: the pick a claim-fitter would make, per channel
    divergence = 0.0
    for chan in ("lg", "cov", "dw"):
        by_claim = max(
            ("white", "colored", "repeat"),
            key=lambda p, _c=chan: (0.0 if p == "white"
                                    else m.get(f"d_{_c}_{p}", 0.0)))
        m[f"by_claim_pick_{chan}"] = by_claim
        if by_claim != by_outcome:
            divergence = 1.0
    m["selection_divergence"] = divergence

    if not sel_real:
        m["max_pile_a_margin_shift_over_own_std"] = 0.0
        m["claim_branch"] = ("white is already the strongest null: no "
                             "process outscores it at 3 sigma on the venue "
                             "outcome")
        return False

    # THE RE-SCORE — eight findings, three conjuncts each
    findings = [k[len("rec_margin_"):] for k in m if k.startswith("rec_margin_")
                and not k.endswith("_std")]
    chan_of = {"darkroom": "lg", "wm_latent": "lg", "wm_efe": "lg",
               "ppo_needs": "lg", "ppo_lp": "lg", "dreamer_xs": "lg",
               "wk5_coverage": "cov", "dwell": "dw"}
    floor_of = {"lg": "f_lg", "cov": "f_cov", "dw": "f_dw"}
    max_ratio, fired_any, fired_names = 0.0, False, []
    for name in sorted(findings):
        chan = chan_of[name]
        dkey = f"d_{chan}_{by_outcome}"
        shift = abs(m.get(dkey, 0.0))
        std = max(m.get(f"rec_std_{name}", 0.0), 1e-12)
        ratio = shift / std
        real = abs(_tstat(m, dkey)) >= SIGMA_GATE
        floor_ok = abs(c.get(floor_of[chan], 0.0)) <= std
        m[f"shift_ratio_{name}"] = ratio
        m[f"fired_{name}"] = float(ratio > 1.0 and real and floor_ok)
        max_ratio = max(max_ratio, ratio)
        if ratio > 1.0 and real and floor_ok:
            fired_any = True
            fired_names.append(name)
    m["max_pile_a_margin_shift_over_own_std"] = max_ratio

    if fired_any:
        m["claim_branch"] = (f"under-nulled and material: {by_outcome} null "
                             f"moves {', '.join(fired_names)} beyond its own "
                             "std")
        return True
    m["claim_branch"] = (f"immaterial: {by_outcome} outscores white but no "
                         "recorded margin moves by more than its own std")
    return False


def run(ledger: Ledger | None = None):
    """12 (seed, process) tasks over 3 single-threaded niced workers
    (w0_diag's pool pattern), memoised into run_spec."""
    import multiprocessing as mp

    spec = BY_ID["W1.00"]
    seeds = list(range(spec.seeds))
    tasks = [(s, p) for s in seeds for p in PROCS]
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        for seed, proc, out in pool.map(_run_task, tasks):
            _CACHE.setdefault(seed, {})[proc] = out
    return run_spec(spec, _experiment, _check, control_fn=_control,
                    ledger=ledger or Ledger())


def _worker_init():
    import os
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))


def _pilot():
    """Plumbing smoke at seed 90 (disjoint from recorded seeds 0-2), short
    envelope — prints JSON, records NOTHING, reads no gate."""
    j0, alpha, ok = _borrow()
    if not ok:
        raise SystemExit("PS.01 borrow unavailable")
    rec = _recorded()
    print("recorded_ok:", rec is not None)
    if rec is not None:
        for k, v in rec.items():
            print(f"  {k}: margin {v[0]:+.4f} std {v[1]:.4f} chan {v[2]}")
    out = {}
    for proc in PROCS:
        r = _rollout(PILOT_SEED, j0, alpha, proc, n_decisions=600)
        out[proc] = {k: round(v, 4) for k, v in r.items()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _pilot()
