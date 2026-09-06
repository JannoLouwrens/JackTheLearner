"""W1.02 — Outcomes have resolution.

THE QUESTION. DP.04's sizing run measured the venue's incumbent outcome —
censored lifespan — unable to see the effect its own family's claims want to
make: 21 distinct lifespans over 3072 lives, a statistic quantised at 6.25
steps against a MIN_GAIN of 5.0, and >= 5791 lives/arm/task derived as the
price of resolving it. The Review FULL's published W1 design (REVIEW_QUEUE
`w0-too-shallow`, 2026-09-06) promotes that defect from a pilot's refusal to a
world-fidelity gate: REPLACE THE METRIC, NOT THE BAR, and prove the
replacement resolves. W1.01 and W1.03 quote this spec's measured quantum,
which is why it must exist before either of them can state their bars.

THE REPLACEMENT, chosen and declared (the registry offers two): INTEGRATED
NEED-SATISFACTION. Per life, G = sum over the life's observable decisions of
(1 - min(1, d(h))) * SIM_S_PER_DECISION, where d(h) is the world's OWN drive
distance (`drives.drive`, section 2.5) read from `w.drives.state.d()` after
each decide — no new constants, no new thresholds. Units: satisfied-seconds.
The dying decision is excluded from the integral (post-decide state on a death
decision is the respawned body's, which would pollute the reading — W1.00's
loop skips it for the same reason); the span still includes it via
`life_lengths`, and the mismatch is one decision, constant across lives,
declared here. Per-need time-to-first-failure (the registry's alternative) is
NOT taken: it needs a failure threshold short of death, which would be a new
constant this spec would then be fitting.

THE LIVES. Repeat-action process at the certified W0.DIAG envelope (E0 and
PS.01's borrowed j0/alpha, lethal=True, constants imported not copied) for
N_DEC = 28000 decisions per seed — sized by the seed-90 plumbing pilot
(repeat lives ~52.7 s at 0.2 s/decision = ~263 decisions/life; 28000 gives
~106 lives, 10% over the 96-life floor; the pilot sized the ENVELOPE, wall
and lives — it froze no claim bar, per BA.03's sizing precedent). Repeat — not white — because W1.00 MEASURED
it the strongest non-learner on the venue's own outcome (d_ml_repeat +14.82 s,
t ~11.8; ledger row pinned below), and because white lives at this envelope
are near-deterministic (ml_white_std 0.078 across seeds): on them BOTH metrics
resolve trivially and the null could fail nothing — a VOID foreseeable at
design time is a design fault, not a finding. The process choice is pinned to
W1.00's row by ran_at; a moved row VOIDs (V2), never silently re-anchors.

PRE-REGISTERED BEFORE THE FIRST RUN (K and N per the registry's note; no
order-statistic bar frozen at a pilot n — the T3.06 lesson):

  N_LIVES = 96 completed lives per seed (V4 floor: fewer VOIDs).
  E_ARM   = 48 — DP.04's best-in-grid E, so the head-to-head is like for like.
  ARMS    = parity split of the first 96 completed lives: A = even life
            index, B = odd. Alternating, so within-run drift loads both arms.
  K_DISTINCT = 48 distinct values (rounded to 1e-6) over the 96 lives.
  MIN_GAIN read LIVE from dp_04_slow_path_verbal.py's source text (5.0 at
            registration, unmoved). Read, not imported: that module carries a
            bare-string banner between its docstring and its `from __future__`
            import and is unimportable today (SyntaxError); reading the
            assignment from its file keeps the value pinned to its owner —
            it moves if the owner moves, and a vanished line VOIDs (V1)
            rather than silently defaulting. Unit mapping, declared honestly:
            DP.04's MIN_GAIN is 5.0 steps of ITS rig's lifespan; here the
            outcome unit is the (satisfied-)second, and the registry borrows
            the number as a pure 5.0 in the outcome's own unit ("MIN_GAIN
            imported from DP.04's test, 5.0, unmoved"). At this envelope's
            0.2 s/decision, 5.0 seconds is 25 decision-steps — a COARSER bar
            than 5.0 steps, not a tighter one. That is the registry's own
            phrasing, the mapping is printed on the row
            (`sim_s_per_decision`), and any claim wanting sub-5.0-second
            effects in this venue must size its own bar against the measured
            quantum this spec records.
  CENSOR_CAP = 0.20.
  SIGMA_GATE = 3.0, imported from W0.DIAG.

THE FOUR CONJUNCTS, computed per seed, for BOTH metrics on the SAME lives:

  C1 RESOLVES-IN-VALUES:  distinct count over the 96 lives >= K_DISTINCT.
  C2 QUANTUM:  quantum = GMD / E_ARM <= MIN_GAIN/3, where GMD is the Gini
     mean difference (mean |x_i - x_j| over all pairs of the 96 per-life
     values) — the expected shift of a 48-life arm mean when one life is
     resampled, i.e. the statistic's granularity under single-life
     resampling. DIVERGENCE FROM DP.04'S ARITHMETIC, NAMED: on its two-point
     lifespans (p=0.767, gap 300) DP.04's lattice reading is 300/48 = 6.25
     and GMD/E reads 2*0.767*0.233*300/48 = 2.24 — the same side of the 1.67
     bar, and GMD/E is defined for any distribution, which the lattice
     reading is not. W1.01/W1.03 quote THIS quantum.
  C3 CENSORING:  censored/(completed+censored) <= CENSOR_CAP, and EVERY
     terminal transition cause-tagged (NE.08's conventions, imported: tags
     from `death_cause()`, the trailing horizon fragment tagged censored,
     censored-vs-fatal by tag never by step counter, rate reported, and no
     uncensored-only mean stands alone — `mean_G_all_lives` prints the
     fragment-inclusive mean beside the claim statistic).
  C4 KNOWN-ANSWER (BINDING, wk5-N3; the one that matters): a synthetic arm
     with an injected advantage of EXACTLY MIN_GAIN — a DECLARED
     outcome-level arithmetic transform, +MIN_GAIN added to arm B's measured
     per-life outcomes, never a delta-e write (PURPOSE_AND_SCAFFOLDING
     section 5/G-A would ERROR a physics-level injection) — must be detected:
     t_inj = (mean(B) + MIN_GAIN - mean(A)) / sqrt(var(A)/48 + var(B)/48)
     >= SIGMA_GATE. Second leg: the same detector at ZERO injection must NOT
     fire (|t_zero| < SIGMA_GATE) — a detector that fires on nothing
     measures nothing.

THE CLAIM: the replacement (G) passes C1 AND C2 AND C3 AND C4 on EVERY seed.
FAIL names the failed conjunct(s) in `claim_branch` (BA.03's lesson: the
branch is named either way). Per-conjunct pass fractions ride on the row.

THE NULL (must fail — law 2): the incumbent censored-lifespan metric scored
by the SAME conjuncts on the SAME lives. If on ANY seed the incumbent passes
all four, the gate cannot distinguish the broken metric from its replacement
and the run VOIDs (V7 "null indistinguishable") — that is an instrument
statement, not a world reading. Predicted from W1.00's recorded numbers, not
assumed: repeat lifespans are decision-quantised (~20-25 distinct over 96,
C1) and their spread prices a +5.0 s detection at t ~2.8 (C4) — but the
registered seeds decide, not this paragraph.

VOID LANES, in test order (an instrument fault is not a world reading, T0.22):
  V1 PS.01 borrow unavailable or MIN_GAIN unreadable   "uncalibrated borrow"
  V2 W1.00 row missing, moved, or selected_null is not
     "repeat"                                          "recorded inputs moved"
  V3 non-finite physics in any run                     "non-finite physics"
  V4 completed lives < N_LIVES on any seed             "under minimum lives"
  V5 a completed life with an empty cause tag          "untagged terminal"
  V6 the replacement's zero-injection leg fires on any
     seed                                     "detector fires on zero injection"
  V7 the incumbent passes all four conjuncts on any
     seed                                              "null indistinguishable"

COST, from the measured pilot (900 decisions in 26.7 s, ~0.030 s/decision
niced): 3 seeds x 28000 decisions ~= 830 core-s each, ~14 min wall on the
3-worker pool, ~42 core-min billed. cpu<2h, the class `coverage` names
fillable today.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import drives
from ..protocol import Ledger, Status, borrow_metrics, run_spec
from ..registry import BY_ID
from ..survival import HOLD_K
from ..w0 import SIM_S_PER_DECISION, W0, random_action
from .w0_diag_exploration_reaches_food import E0, N_ACT, SIGMA_GATE

IMPL_DEPS = ["experiments/w0.py", "experiments/drives.py",
             "experiments/survival.py", "playground.py",
             "experiments/tests/w0_diag_exploration_reaches_food.py"]

N_DEC = 28000
N_LIVES = 96
E_ARM = 48
K_DISTINCT = 48
CENSOR_CAP = 0.20
ROUND_Q = 1e-6
PILOT_SEED = 90

# pinned recorded row — a moved row VOIDs (V2), never silently re-anchors
W100_RAN_AT = "2026-09-06T10:30:12"


def _borrow():
    b = borrow_metrics("PS.01", ("j0_ms", "alpha"))
    if b.values is None:
        return None, None, 0.0
    return b.values["j0_ms"], b.values["alpha"], 1.0


def _min_gain() -> Optional[float]:
    """MIN_GAIN read from dp_04's source (see docstring for why not import)."""
    try:
        src = (Path(__file__).parent / "dp_04_slow_path_verbal.py").read_text()
    except OSError:
        return None
    hit = re.search(r"^MIN_GAIN = ([0-9.]+)", src, re.M)
    return float(hit.group(1)) if hit else None


def _recorded_ok() -> bool:
    """W1.00's row: pinned by ran_at, and its measured strongest null must be
    the process this spec runs. Selection was W1.00's job, not this spec's."""
    path = Path(__file__).resolve().parents[1] / "ledger.json"
    try:
        row = json.loads(path.read_text())["results"]["W1.00"]
        return (row["ran_at"] == W100_RAN_AT
                and row["metrics"]["selected_null"] == "repeat")
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _rollout(seed: int, j0: float, alpha: float,
             n_decisions: int = N_DEC) -> dict:
    """Repeat-action lives at the W0.DIAG envelope. Per life: span (s), G
    (satisfied-seconds), cause tag. The trailing fragment is censored."""
    w = W0(seed=seed, j0=j0, alpha=alpha, lethal=True)
    w.drives.state = drives.DriveState(e=E0)
    rng = np.random.RandomState(seed * 6553 + 11)
    held_a, held_left = None, 0
    g_acc = 0.0
    lives: list = []          # (span_s, G, cause) per completed life
    t0 = time.perf_counter()

    for _ in range(n_decisions):
        if held_left == 0:
            held_a, held_left = random_action(rng), HOLD_K
        a, held_left = held_a, held_left - 1
        w.decide(a)
        if w.died_this_decision:
            lives.append((w.life_lengths[-1], g_acc,
                          str(w.last_death_cause)))
            g_acc = 0.0
            w.drives.state = drives.DriveState(e=E0)
        else:
            s = 1.0 - min(1.0, w.drives.state.d())
            g_acc += s * SIM_S_PER_DECISION

    frag_span = float(w.sim_seconds - w._life_started_at)
    return {
        "lives": lives,
        "frag": (frag_span, g_acc, "censored:horizon"),
        "physics_finite": float(bool(np.all(np.isfinite(w.data.qpos))
                                     and np.all(np.isfinite(w.data.qvel)))),
        "wall_s": float(time.perf_counter() - t0),
    }


def _gmd(x: np.ndarray) -> float:
    """Gini mean difference: mean |x_i - x_j| over all ordered pairs, i != j."""
    n = len(x)
    xs = np.sort(x)
    # sum over pairs via the order-statistic identity, O(n log n)
    coef = 2.0 * np.arange(1, n + 1) - n - 1
    return float(np.dot(coef, xs) * 2.0 / (n * (n - 1)))


def _conjuncts(vals: np.ndarray, min_gain: float,
               censor_rate: float, causes_ok: bool) -> dict:
    a, b = vals[0::2], vals[1::2]
    se = math.sqrt(np.var(a, ddof=1) / E_ARM + np.var(b, ddof=1) / E_ARM)
    se = max(se, 1e-12)
    t_inj = (float(np.mean(b)) + min_gain - float(np.mean(a))) / se
    t_zero = (float(np.mean(b)) - float(np.mean(a))) / se
    distinct = len(np.unique(np.round(vals / ROUND_Q) * ROUND_Q))
    quantum = _gmd(vals) / E_ARM
    return {
        "distinct": float(distinct),
        "quantum": quantum,
        "t_inj": t_inj,
        "t_zero": t_zero,
        "c1_ok": float(distinct >= K_DISTINCT),
        "c2_ok": float(quantum <= min_gain / 3.0),
        "c3_ok": float(censor_rate <= CENSOR_CAP and causes_ok),
        "c4_ok": float(t_inj >= SIGMA_GATE),
        "zero_quiet": float(abs(t_zero) < SIGMA_GATE),
    }


_CACHE: Dict[int, dict] = {}


def _bundle(seed: int) -> dict:
    if seed not in _CACHE:
        j0, alpha, ok = _borrow()
        mg = _min_gain()
        if not ok or mg is None:
            _CACHE[seed] = {"borrowed_ok": 0.0}
            return _CACHE[seed]
        out = _rollout(seed, j0, alpha)
        out.update(borrowed_ok=1.0, min_gain=mg)
        _CACHE[seed] = out
    return _CACHE[seed]


def _seed_metrics(seed: int, incumbent: bool) -> dict:
    b = _bundle(seed)
    if b.get("borrowed_ok", 0.0) != 1.0:
        return {"borrowed_ok": 0.0}
    m: dict = {"borrowed_ok": 1.0,
               "recorded_ok": float(_recorded_ok()),
               "physics_finite": b["physics_finite"],
               "min_gain": b["min_gain"],
               "sim_s_per_decision": float(SIM_S_PER_DECISION)}
    lives = b["lives"]
    m["n_lives_completed"] = float(len(lives))
    m["lives_ok"] = float(len(lives) >= N_LIVES)
    if len(lives) < N_LIVES:
        return m
    used = lives[:N_LIVES]
    causes_ok = all(cause for _, _, cause in used)
    m["causes_ok"] = float(causes_ok)
    n_censored = 1 if b["frag"][0] > 0.0 else 0
    censor_rate = n_censored / (len(lives) + n_censored)
    m["censor_rate"] = float(censor_rate)
    vals = np.array([(span if incumbent else g) for span, g, _ in used])
    frag_val = b["frag"][0] if incumbent else b["frag"][1]
    m["mean_all_lives"] = float(np.mean(
        [v for v in vals] + ([frag_val] if n_censored else [])))
    cj = _conjuncts(vals, b["min_gain"], censor_rate, causes_ok)
    m.update(cj)
    m["mean_outcome"] = float(np.mean(vals))
    m["sd_outcome"] = float(np.std(vals, ddof=1))
    m["all_ok"] = float(cj["c1_ok"] and cj["c2_ok"]
                        and cj["c3_ok"] and cj["c4_ok"])
    m["fails_some"] = float(not m["all_ok"])
    m["quantum_ratio"] = cj["quantum"] / (b["min_gain"] / 3.0)
    return m


def _experiment(seed: int) -> dict:
    return _seed_metrics(seed, incumbent=False)


def _control(seed: int) -> dict:
    return _seed_metrics(seed, incumbent=True)


def _void(m: dict, reason: str):
    m["void_reason"] = reason
    return Status.VOID


def _check(m: dict, c: dict):
    # V1-V2: the borrow and the pinned row
    if m.get("borrowed_ok", 0.0) != 1.0 or c.get("borrowed_ok", 0.0) != 1.0:
        return _void(m, "uncalibrated borrow")
    if m.get("recorded_ok", 0.0) != 1.0:
        return _void(m, "recorded inputs moved")
    # V3-V5: the rig (means across seeds; 1.0 means every seed)
    if m.get("physics_finite", 0.0) != 1.0:
        return _void(m, "non-finite physics")
    if m.get("lives_ok", 0.0) != 1.0:
        return _void(m, "under minimum lives")
    if m.get("causes_ok", 0.0) != 1.0:
        return _void(m, "untagged terminal transition")
    # V6: the detector must be quiet on zero injection, every seed
    if m.get("zero_quiet", 0.0) != 1.0:
        return _void(m, "detector fires on zero injection")
    # V7: law 2 — the incumbent must fail somewhere on every seed
    if c.get("fails_some", 0.0) != 1.0:
        return _void(m, "null indistinguishable: the incumbent passes all "
                        "four conjuncts, the gate measures nothing")

    if m.get("all_ok", 0.0) == 1.0:
        m["claim_branch"] = ("resolves: the graded outcome passes all four "
                            "conjuncts on every seed while the incumbent "
                            "fails")
        return True
    failed = [name for name, key in (("C1-distinct", "c1_ok"),
                                     ("C2-quantum", "c2_ok"),
                                     ("C3-censoring", "c3_ok"),
                                     ("C4-detection", "c4_ok"))
              if m.get(key, 0.0) != 1.0]
    m["claim_branch"] = ("does not resolve: replacement failed "
                         + ", ".join(failed)
                         + " on at least one seed; the venue still cannot "
                           "host a DP-family claim")
    return False


def _run_task(seed: int) -> tuple:
    return seed, _bundle(seed)


def run(ledger: Ledger | None = None):
    """3 seed tasks over 3 single-threaded niced workers (w0_diag's pool
    pattern), memoised into run_spec."""
    import multiprocessing as mp

    spec = BY_ID["W1.02"]
    seeds = list(range(spec.seeds))
    ctx = mp.get_context("spawn")
    with ctx.Pool(3, initializer=_worker_init) as pool:
        for seed, out in pool.map(_run_task, seeds):
            _CACHE[seed] = out
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
    envelope — prints JSON, records NOTHING, reads no gate, freezes no bar."""
    j0, alpha, ok = _borrow()
    mg = _min_gain()
    if not ok or mg is None:
        raise SystemExit("borrow unavailable (PS.01 or MIN_GAIN)")
    print("recorded_ok:", _recorded_ok(), " min_gain:", mg,
          " sim_s_per_decision:", SIM_S_PER_DECISION)
    out = _rollout(PILOT_SEED, j0, alpha, n_decisions=900)
    lives = out["lives"]
    spans = np.array([s for s, _, _ in lives])
    gs = np.array([g for _, g, _ in lives])
    causes = sorted({cause for _, _, cause in lives})
    print(json.dumps({
        "n_lives": len(lives),
        "physics_finite": out["physics_finite"],
        "wall_s": round(out["wall_s"], 1),
        "span_mean_sd": [round(float(np.mean(spans)), 3),
                         round(float(np.std(spans, ddof=1)), 3)],
        "g_mean_sd": [round(float(np.mean(gs)), 3),
                      round(float(np.std(gs, ddof=1)), 3)],
        "span_distinct": len(np.unique(np.round(spans / ROUND_Q) * ROUND_Q)),
        "g_distinct": len(np.unique(np.round(gs / ROUND_Q) * ROUND_Q)),
        "causes": causes,
        "frag": [round(out["frag"][0], 3), round(out["frag"][1], 3),
                 out["frag"][2]],
    }, indent=2))


if __name__ == "__main__":
    _pilot()
