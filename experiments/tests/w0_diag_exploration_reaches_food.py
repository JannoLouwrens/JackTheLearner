"""W0.DIAG — the exploration process, not the world: does correlated random
action buy life in W0?

THE QUESTION. Nine independent instruments read W0 as too shallow to grade
capability (docs/REVIEW_QUEUE.md, `w0-too-shallow`). Nine agreeing instruments
is precisely the condition under which a shared confound is invisible, and the
cheapest candidate confound is this: every one of them scored policies whose
exploration is per-decision white noise (or a learner bootstrapping from it),
and a white-noise action process barely travels — it dithers. If W0's food is
reachable only by SUSTAINED motion, then "the world does not reward
capability" and "no tested policy's exploration process ever reaches what the
world rewards" produce identical readings on every one of those instruments.
This spec is the cheap disagreeing instrument the Review ordered on 2026-08-25
(accepted field-watch wk4-N3; queue row written 2026-08-31) to attack that
confound BEFORE a world redesign is spent on it.

THE DESIGN. Five conditions per seed, all policies random (nothing learns):

  random   LC.03's stationary white null, via `run_survival` unchanged.
  repeat   LC.03's stationary hold-5 null, via `run_survival` unchanged.
  up       colored noise, correlation time tau ramped geometrically
           TAU[0] -> TAU[1] decisions across the run (near-white -> red).
  down     the mirror image: tau ramped TAU[1] -> TAU[0].  CONTROL.
  ka       white actions, respawn energy ramped E0_KA[0] -> E0_KA[1] across
           the run.  KNOWN-ANSWER CONTROL (binding, field-watch wk5-N3).

The readout for every condition is `life_gain` — mean life span of the final
third of completed lives minus the first third, LC.03's certified channel,
arithmetic identical to `survival.run_survival`. A stationary policy has no
within-run trend, so the nulls read ~0; the scheduled conditions turn their
schedule into a within-run trend that life_gain can carry.

MARGINAL MATCHING, exact by construction (the T3.06 matched-magnitude lesson,
designed in rather than retrofitted). The colored policy is AR(1) in a latent
Gaussian: z_k = rho_k z_{k-1} + sqrt(1-rho_k^2) eps, z_0 ~ N(0,1), so z_k is
exactly N(0,1) at every k for ANY rho schedule; the action is
a = erf(z/sqrt(2)) = 2*Phi(z)-1, exactly Uniform(-1,1) per dimension — the
same marginal law as `w0.random_action` at every decision and every tau. Only
the temporal correlation differs from the null. ("beta-scheduled" from the
nomination is realised as this correlation-time ramp: spectral 1/f^beta
synthesis cannot hold the marginal fixed under clipping, and an unmatched
magnitude would hand the arm a magnitude confound — stated plainly, not
hidden.) rho = exp(-1/tau).

TWO MECHANISMS, ONE ATTRIBUTION GATE — the load-bearing design fact, bought
by the pilot before registration. In this body, correlation LOWERS
per-decision jitter (dithering becomes sustained motion), and lower jitter is
less mechanical work, which is a drift toward statuehood — and W0 measurably
rewards statuehood (LC.03: the statue maximises life; darkroom prospered by
learned passivity). So a positive life_gain under the ramp has TWO candidate
causes: the exploration story (sustained motion reaches food) and the
passivity story (less work, slower drain), and only the first attacks the
shallowness reading. The `ate_total` channel separates them: food reached is
counted directly. The PASS therefore requires the food conjunct, and a margin
that fires without it records FAIL with the branch named.

PRE-REGISTERED GATES — all scale-free t-statistics over seeds (house `_tstat`
idiom, mean*sqrt(3)/seed-std); the pilot fixed ONLY the envelope constants
(N_DECISIONS, E0, TAU, E0_KA, lives floors), so no order-statistic bar is
frozen at a pilot n (the T3.06 extreme-value lesson). SIGMA_GATE = 3.0.

  VOID, in test order — an instrument fault is not a world reading (T0.22):
    V1 borrow of PS.01's (j0_ms, alpha) unavailable       "uncalibrated borrow"
    V2 non-finite physics in any run                      "non-finite physics"
    V3 completed lives under floor in any run             "under minimum lives"
       (nulls >= MIN_LIVES_NULL, LC.03's floor; scheduled >= MIN_LIVES_SCHED —
       thirds over fewer lives are not a trend reading)
    V4 KNOWN-ANSWER failed: ka_margin (gain_ka - gain_random, paired) not
       positive at >= 3 sigma. The answer is certified arithmetic — LC.03
       measured statue life = e0/BASAL_B to 0.02%, so energy ramped up across
       a run MUST surface as positive life_gain. An instrument that cannot
       see a real scheduled survival gradient does not get to report on the
       correlation one. Recorded limit, carried from the Review's acceptance:
       the RWG/PIC/POIC inversion paper (arXiv:2602.18856) motivates this
       check by ANALOGY, not arithmetic — it buys the known-answer check only.
    V5 a stationary null trends: |t(gain_random)| or |t(gain_repeat)| >= 3.
       Then the world moves within runs and no schedule reading attributes.
    V6 reversed schedule gains the same sign: t(margin_down) >= +3. Then the
       channel measures run-time drift, not correlation.
    V7 schedule never expressed in the body: the up-run's per-decision jitter
       must FALL from its first third to its final third at >= 3 sigma
       (jit_delta_up = jit_first - jit_final > 0). The direction is the
       pilot's measured signature (jitter drops ~2x, monotone in tau across
       0 -> 32); deltas spanning a death/respawn teleport are excluded.

  PASS   t(margin_up) >= 3.0 AND mean eats_up > mean eats_random — correlated
         exploration buys life THROUGH FOOD. Part of the shallowness reading
         is the exploration process; lands in D10 fork (b) and the
         `w0-too-shallow` design (due 09-06).
  FAIL   every instrument gate green and either branch, named in
         `claim_branch` (the BA.03 one-bit-verdict lesson):
         (a) "no margin" — t(margin_up) < 3: correlation buys nothing here;
         (b) "passivity, not food" — the margin fires with no eats
         advantage: the gain is the statue-ward work reduction LC.03
         already measured, and exploration still never reaches reward.
         On either branch the shallowness finding survives its cheapest
         attack, which is what the attack was for.

PILOT RECORD (seeds 90/91, disjoint from recorded seeds 0-2, envelope sizing
and mechanism probes only — no gate read against them; 2026-08-31, this
commit).
  Mechanism probe, seed 91, FIXED tau, 1200 decisions, e0=0.10:
    tau:        white    2.0      8.0      32.0
    lives:      5        5        3        0 (zero deaths in 240 sim-s)
    mean_life:  41.4 s   47.9 s   69.1 s   censored
    jitter:     0.066    0.058    0.032    0.015 m/decision
    ate:        0        0        1        4
  So the food route EXISTS (tau=32 eats and stops dying where white starves)
  and the jitter signature is monotone — both built into the gates above.
  Envelope pilot, seed 90, N=3200, E0=0.10, E0_KA=(0.10,0.25):
    random: 15 lives, gain +0.2 s (flat), ate 0
    up:     12 lives, gain +11.3 s, jitter 0.033 -> 0.015, ate 0
    down:   12 lives, gain -10.5 s (mirror sign), jitter 0.012 -> 0.029
    ka:     10 lives, gain +40.1 s (the certified gradient surfaces), ate 0
  Note the seed-90 up-gain arrived with ZERO eats — the passivity channel is
  live, which is exactly why the PASS carries the eats conjunct. Frozen from
  this: N_DECISIONS=3200, E0=0.10, E0_KA=(0.10,0.25), TAU=(0.5,32.0),
  MIN_LIVES_NULL=12 (LC.03's floor unchanged, seed-90 margin 3),
  MIN_LIVES_SCHED=8 (ka's 10 is the tightest, margin 2).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..survival import run_survival
from ..w0 import W0, random_action

IMPL_DEPS = ["experiments/w0.py", "experiments/drives.py",
             "experiments/survival.py", "playground.py"]

# ── frozen envelope (pilot, seeds 90/91 — see PILOT RECORD in the docstring) ─
E0 = 0.10                 # respawn energy, all conditions except ka's ramp
E0_KA = (0.10, 0.25)      # known-answer ramp endpoints
N_DECISIONS = 3200        # ~96 s wall per run at the measured throughput
TAU = (0.5, 32.0)         # correlation-time schedule, decisions, geometric
N_ACT = 8                 # rover actuators; W0.__init__ asserts ROVER_NU
MIN_LIVES_NULL = 12       # LC.03's floor, unchanged
MIN_LIVES_SCHED = 8       # >= 2 completed lives per third for the trend read
SIGMA_GATE = 3.0
PILOT_SEED = 90

MODES = ("random", "repeat", "up", "down", "ka")
_SQRT2 = math.sqrt(2.0)


def _borrow():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if b.values is None:
        return None, None, 0.0
    return b.values["j0_ms"], b.values["alpha"], 1.0


def _life_gain(spans) -> float:
    """survival.run_survival's arithmetic, verbatim: final third of completed
    lives minus first third, 0.0 when there is no full third."""
    third = len(spans) // 3
    return (float(np.mean(spans[-third:]) - np.mean(spans[:third]))
            if third >= 1 else 0.0)


def _scheduled_run(seed: int, j0: float, alpha: float, mode: str,
                   n_decisions: int = N_DECISIONS) -> dict:
    """The up/down/ka loop. Mirrors `run_survival`'s null branch decision for
    decision — same world construction, same rng stream construction, same
    post-death drive reset — except the action process (up/down) or the
    respawn energy (ka) follows the declared schedule."""
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True)
    w.drives.state = drives.DriveState(e=E0 if mode != "ka" else E0_KA[0])
    rng = np.random.RandomState(seed * 6553 + 11)
    z = rng.randn(N_ACT)                       # stationary start: z ~ N(0,1)
    prev_xy: Optional[np.ndarray] = None
    jit: list = []                             # per-decision metres, nan at teleports
    t0 = time.perf_counter()

    for k in range(n_decisions):
        frac = k / max(1, n_decisions - 1)
        if mode == "ka":
            a = random_action(rng)
        else:
            f = frac if mode == "up" else 1.0 - frac
            tau = TAU[0] * (TAU[1] / TAU[0]) ** f
            rho = math.exp(-1.0 / tau)
            z = rho * z + math.sqrt(1.0 - rho * rho) * rng.randn(N_ACT)
            a = np.array([math.erf(v / _SQRT2) for v in z])  # exact U(-1,1)

        w.decide(a)
        xy = np.array(w.data.xpos[w.rover_bid][:2], dtype=float)
        if w.died_this_decision:
            jit.append(np.nan)                 # respawn is a teleport
            prev_xy = None
            e_next = (E0 if mode != "ka"
                      else E0_KA[0] + (E0_KA[1] - E0_KA[0]) * frac)
            w.drives.state = drives.DriveState(e=e_next)
        else:
            jit.append(float(np.linalg.norm(xy - prev_xy))
                       if prev_xy is not None else np.nan)
            prev_xy = xy

    spans = list(w.life_lengths)
    third = len(jit) // 3
    return {
        "life_gain": _life_gain(spans),
        "n_lives": float(len(spans)),
        "mean_life_s": float(np.mean(spans)) if spans else 0.0,
        "jit_first": (float(np.nanmean(jit[:third])) if third else 0.0),
        "jit_final": (float(np.nanmean(jit[-third:])) if third else 0.0),
        "ate": float(sum(w.drives.ate_total.values())),
        # censoring made visible: the incomplete life still open at cutoff.
        # life_gain reads completed lives only (LC.03's arithmetic), so a
        # correlated tail that STOPS dying underreads — the reader must be
        # able to see that from the row (the DP.04 coarseness critique).
        "tail_open_life_s": float(w.report()["life_s"]),
        "physics_finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                     and np.all(np.isfinite(w.data.qvel)))),
        "wall_s": float(time.perf_counter() - t0),
    }


def _null_run(seed: int, j0: float, alpha: float, policy: str,
              n_decisions: int = N_DECISIONS) -> dict:
    r = run_survival(seed, j0=j0, alpha=alpha, n_decisions=n_decisions,
                     policy=policy, e0=E0)
    return {"life_gain": r["life_gain"], "n_lives": r["n_lives"],
            "mean_life_s": r["mean_life_s"],
            "ate": float(sum(r["ate_total"].values())),
            "physics_finite": r["physics_finite"],
            "wall_s": r["wall_s"], "jit_first": 0.0, "jit_final": 0.0,
            "tail_open_life_s": 0.0}


def _run_task(task) -> tuple:
    seed, mode = task
    j0, alpha, ok = _borrow()
    if not ok:
        return seed, mode, {"borrowed_ok": 0.0}
    if mode in ("random", "repeat"):
        out = _null_run(seed, j0, alpha,
                        "random" if mode == "random" else "random-repeat")
    else:
        out = _scheduled_run(seed, j0, alpha, mode)
    out["borrowed_ok"] = 1.0
    return seed, mode, out


_CACHE: Dict[int, Dict[str, dict]] = {}


def _bundle(seed: int) -> Dict[str, dict]:
    """All five conditions for one seed, memoised — run() precomputes via the
    pool; a direct call (pilot, ad-hoc) computes serially."""
    if seed not in _CACHE:
        _CACHE[seed] = {m: _run_task((seed, m))[2] for m in MODES}
    return _CACHE[seed]


def _claim_metrics(b: Dict[str, dict]) -> dict:
    if any(r.get("borrowed_ok", 0.0) != 1.0 for r in b.values()):
        return {"borrowed_ok": 0.0}
    lives_ok = float(
        b["random"]["n_lives"] >= MIN_LIVES_NULL
        and b["repeat"]["n_lives"] >= MIN_LIVES_NULL
        and all(b[m]["n_lives"] >= MIN_LIVES_SCHED for m in ("up", "down", "ka")))
    return {
        "borrowed_ok": 1.0,
        "physics_finite_min": float(min(r["physics_finite"] for r in b.values())),
        "lives_ok": lives_ok,
        "gain_random": b["random"]["life_gain"],
        "gain_repeat": b["repeat"]["life_gain"],
        "gain_up": b["up"]["life_gain"],
        "margin_up": b["up"]["life_gain"] - b["random"]["life_gain"],
        "jit_delta_up": b["up"]["jit_first"] - b["up"]["jit_final"],
        "jit_first_up": b["up"]["jit_first"],
        "jit_final_up": b["up"]["jit_final"],
        "lives_random": b["random"]["n_lives"],
        "lives_up": b["up"]["n_lives"],
        "mean_life_random": b["random"]["mean_life_s"],
        "mean_life_up": b["up"]["mean_life_s"],
        "eats_random": b["random"]["ate"],
        "eats_up": b["up"]["ate"],
        "tail_open_up_s": b["up"]["tail_open_life_s"],
    }


def _control_metrics(b: Dict[str, dict]) -> dict:
    if any(r.get("borrowed_ok", 0.0) != 1.0 for r in b.values()):
        return {"borrowed_ok": 0.0}
    return {
        "borrowed_ok": 1.0,
        "gain_down": b["down"]["life_gain"],
        "margin_down": b["down"]["life_gain"] - b["random"]["life_gain"],
        "gain_ka": b["ka"]["life_gain"],
        "ka_margin": b["ka"]["life_gain"] - b["random"]["life_gain"],
        "lives_down": b["down"]["n_lives"],
        "lives_ka": b["ka"]["n_lives"],
        "mean_life_down": b["down"]["mean_life_s"],
        "mean_life_ka": b["ka"]["mean_life_s"],
        "eats_down": b["down"]["ate"],
        "eats_ka": b["ka"]["ate"],
        "tail_open_down_s": b["down"]["tail_open_life_s"],
    }


def _experiment(seed: int) -> dict:
    return _claim_metrics(_bundle(seed))


def _control(seed: int) -> dict:
    return _control_metrics(_bundle(seed))


def _tstat(m: dict, key: str) -> float:
    """The house paired 3-sigma idiom: mean * sqrt(n_seeds) / seed std."""
    return m.get(key, 0.0) * math.sqrt(3) / max(m.get(f"{key}_std", 0.0), 1e-9)


def _void(m: dict, reason: str):
    """Name the firing branch in the recorded metrics (LC.03 v2's lesson: the
    generic VOID message admits every narrative)."""
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # V1-V3: the rig
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")
    if m.get("physics_finite_min", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    if m.get("lives_ok", 0.0) != 1.0:
        return _void(m, "under minimum lives")
    # V4: the binding known-answer — the instrument must see a certified
    # scheduled gradient before its W0 reading is believed (wk5-N3).
    if c["ka_margin"] <= 0.0 or _tstat(c, "ka_margin") < SIGMA_GATE:
        return _void(m, "known-answer failed: life_gain blind to a certified "
                        "scheduled gradient")
    # V5: stationary nulls must be flat, else the world drifts within runs
    if (abs(_tstat(m, "gain_random")) >= SIGMA_GATE
            or abs(_tstat(m, "gain_repeat")) >= SIGMA_GATE):
        return _void(m, "stationary null trends within runs")
    # V6: the reversed schedule must not gain the same sign
    if _tstat(c, "margin_down") >= SIGMA_GATE:
        return _void(m, "reversed schedule gains same sign: drift, not "
                        "correlation")
    # V7: the schedule must have expressed in the body — jitter falls
    if m["jit_delta_up"] <= 0.0 or _tstat(m, "jit_delta_up") < SIGMA_GATE:
        return _void(m, "schedule never expressed in the body: jitter did "
                        "not fall")
    # THE CLAIM, both conjuncts, branch named either way (BA.03's lesson)
    margin_fires = _tstat(m, "margin_up") >= SIGMA_GATE
    food_advantage = m["eats_up"] > m["eats_random"]
    if margin_fires and food_advantage:
        m["claim_branch"] = "correlation buys life through food"
        return True
    m["claim_branch"] = ("passivity, not food: margin fired with no eats "
                         "advantage" if margin_fires else "no margin")
    return False


def run(ledger: Ledger | None = None):
    """The registered run: 15 (seed, condition) tasks over 3 single-threaded
    niced workers (LC.03's pattern), memoised into run_spec."""
    import multiprocessing as mp

    spec = BY_ID["W0.DIAG"]
    seeds = list(range(spec.seeds))
    tasks = [(s, mode) for s in seeds for mode in MODES]
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        for seed, mode, out in pool.map(_run_task, tasks):
            _CACHE.setdefault(seed, {})[mode] = out
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
    """Seed 90+, envelope sizing only — prints JSON, records NOTHING, reads no
    gate. Disjoint from the recorded seeds 0-2 (SM.02's pilot idiom). The
    committed PILOT RECORD in the docstring was produced by this path plus a
    fixed-tau mechanism probe (seed 91) recorded there verbatim."""
    import json
    j0, alpha, ok = _borrow()
    if not ok:
        raise SystemExit("PS.01 borrow unavailable")
    out = {}
    for mode in ("random", "up", "down", "ka"):
        r = (_null_run(PILOT_SEED, j0, alpha, "random") if mode == "random"
             else _scheduled_run(PILOT_SEED, j0, alpha, mode))
        out[mode] = {k: round(v, 4) for k, v in r.items()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _pilot()
