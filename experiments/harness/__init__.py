"""The comparison harness — the instrument that will choose his brain (order J4).

Source of every rule here: ruling 19 (`docs/DECISIONS_RESOLVED.md`, "Stage 2's
rule, written now from §8") and `docs/research/PUBLISHED_PRIORS.md` §8
("The staged design", "Reproduce first"). SciPy only. AdaStop is NOT used
here; if it ever is, only from the commit §8 pins (f272065a...).

Ours beside theirs (§8 "Take as-is"):
  outcome     decisions until W consecutive lives average >= theta meals,
              censored at T                       | same (theta = 5)
  test        permutation log-rank, looks at 10 and 20 runs/arm, 0.025 each
                                                  | same
  statistic   scipy.stats.logrank's statistic, labels permuted
                                                  | §8 names scipy logrank +
                                                    permutation_test
  sensitivity scipy.stats.boschloo_exact on crossed-by-T
                                                  | same
  mark 2      ruling 22 (11-16): PRIMARY = decisions until W consecutive
              lives are each SUSTAINED (capped AND intake >= basal drain),
              `outcome_sustained`; the theta = 5 mark stays as a reported
              secondary                          | ours (the record has no
                                                   survival-world mark)
  crash/NaN   "is an outcome" (ruling 19): a run whose log ends in a crash
              or NaN never crossed and is censored at T — it stays in the
              risk set to the end, i.e. it counts against its arm. This is
              our reading of the ruling; the record gives no formula.

The thirteen self-checks (§8 "Reproduce first") and who pays each:
  PAID BY TESTS HERE (tests/test_harness.py):
    #10 outcome code right on synthetic logs, edge cases
    #13 runs analysed = runs launched (`check_runs_accounted`)
    #12 tools on a known case: 4/15 vs 0/15 -> Fisher 0.100, Boschloo 0.043
  OWED IN J4 (no learner needed):
    #1  same-against-same, >= 1,000 repetitions through `compare`
        (experiments/harness/same_vs_same.py, rule in its docstring)
    #3  floor and ceiling calibration arms
    #11 analysis script frozen and hashed (`analysis_hash`)
  OWED BY ORDERS THAT BUILD A LEARNER (need Brain B from J3):
    #2  handicapped copy (lr 0) detected at planned power   -> J3 then J4
    #4  same seed twice -> identical logs                    -> J3
    #5  run k sees the same world in every arm               -> J2/J3 wiring
    #6  resume across a kernel boundary = uninterrupted run  -> J3/J5
    #7  every decision counted incl. warm-up and prefill     -> J3/J5
    #8  reward-forever world 1/(1-gamma); death no bootstrap -> J3
    #9  probes on/off leave training logs identical          -> J3
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

THETA = 5.0          # meals per life (ruling 19)
LOOK_ALPHA = 0.025   # each of the two looks (§8)
LOOKS = (10, 20)     # runs per arm at each look


@dataclass(frozen=True)
class Outcome:
    time: float      # decisions at crossing, or T if censored
    event: bool      # True = crossed by T


def outcome(lives, W: int, T: float, theta: float = THETA) -> Outcome:
    """One run's primary outcome.

    `lives`: list of (decisions_in_life, meals_in_life) in order. A NaN in
    either field, or the string "crash" as an entry, ends the run (ruling
    19: an outcome, censored at T). The crossing time is the cumulative
    decision count at the END of the W-th life of the first window of W
    consecutive lives whose mean meals >= theta; it counts only if <= T.
    Fewer than W lives -> censored at T.
    """
    if W < 1 or not T > 0:
        raise ValueError("W >= 1 and T > 0 required")
    cum, meals = 0.0, []
    for life in lives:
        if isinstance(life, str):
            break  # crash: run ends, never crossed
        d, m = float(life[0]), float(life[1])
        if math.isnan(d) or math.isnan(m) or d < 0 or m < 0:
            break
        cum += d
        if cum > T:
            break
        meals.append(m)
        if len(meals) >= W and sum(meals[-W:]) / W >= theta:
            return Outcome(cum, True)
    return Outcome(float(T), False)


def sustained_life(capped, intake: float, drain: float) -> bool:
    """Ruling 22's mark on one life: it reached the time cap (`capped` True)
    AND its food intake over the life >= its basal drain (both in energy
    bars, `drain` = drives.BASAL_B x seconds lived at the rung being run).
    A NaN in either quantity is not a sustained life."""
    intake, drain = float(intake), float(drain)
    if math.isnan(intake) or math.isnan(drain):
        return False
    return bool(capped is True and intake >= drain)


def outcome_sustained(lives, W: int, T: float) -> Outcome:
    """Ruling 22's PRIMARY outcome: decisions until W CONSECUTIVE lives are
    each sustained (`sustained_life`), censored at T.

    `lives`: list of (decisions_in_life, capped, intake, drain) in order. The
    crossing time is the cumulative decision count at the END of the W-th
    sustained life in a row; it counts only if <= T. Crash/NaN rules are
    `outcome`'s (ruling 19): the string "crash" or a NaN/negative decision
    count ends the run, never crossed. Unlike the theta mark (a window MEAN),
    every life of the window must be sustained: one unsustained life resets
    the run of consecutive lives to zero. The theta mark (`outcome`, theta =
    5) stays as the reported secondary, "first signs of eating".
    """
    if W < 1 or not T > 0:
        raise ValueError("W >= 1 and T > 0 required")
    cum, streak = 0.0, 0
    for life in lives:
        if isinstance(life, str):
            break
        d = float(life[0])
        if math.isnan(d) or d < 0:
            break
        cum += d
        if cum > T:
            break
        streak = streak + 1 if sustained_life(life[1], life[2], life[3]) else 0
        if streak >= W:
            return Outcome(cum, True)
    return Outcome(float(T), False)


def _censored(outs):
    t = np.array([o.time for o in outs], float)
    e = np.array([o.event for o in outs], bool)
    return stats.CensoredData(uncensored=t[e], right=t[~e])


def logrank_stat(a, b) -> float:
    """|log-rank Z| between two lists of Outcome; 0 if no events at all.

    Also 0 when the log-rank variance is zero (SciPy returns NaN): e.g. every
    run in both arms crosses at the same decision count. Zero variance means
    the pooled data cannot tell the arms apart. Found 11-16 by check #11's
    identical-arms test: before this, NaN made every permutation a non-hit,
    p = 1/(n_perm+1), and two identical arms read A_DIFFERS.
    """
    if not any(o.event for o in a) and not any(o.event for o in b):
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        z = float(stats.logrank(_censored(a), _censored(b)).statistic)
    return abs(z) if math.isfinite(z) else 0.0


def perm_logrank(a, b, n_perm: int = 2000, rng=None) -> float:
    """Two-sided permutation p of the log-rank statistic (labels shuffled).

    p = (1 + #{perm >= observed}) / (1 + n_perm): never 0, exact in level.
    """
    rng = np.random.default_rng(rng)
    obs = logrank_stat(a, b)
    pool = list(a) + list(b)
    na = len(a)
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pool))
        s = logrank_stat([pool[i] for i in idx[:na]], [pool[i] for i in idx[na:]])
        hits += s >= obs - 1e-12
    return (1 + hits) / (1 + n_perm)


def fisher_p(sa: int, na: int, sb: int, nb: int) -> float:
    return float(stats.fisher_exact([[sa, na - sa], [sb, nb - sb]]).pvalue)


def boschloo_p(sa: int, na: int, sb: int, nb: int) -> float:
    # scipy's table layout: columns are the two arms
    return float(stats.boschloo_exact([[sa, sb], [na - sa, nb - sb]]).pvalue)


def analysis_hash() -> str:
    """sha256 of this file — check #11 freezes it in the pre-registration."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Verdict:
    decision: str    # "A_DIFFERS" | "NOT_DISTINGUISHED" | "FUTILE"
    look: int        # 1 or 2: the look at which it stopped
    p1: float        # look-1 permutation p
    p2: float        # look-2 permutation p (nan if stopped at look 1)


def compare(a, b, n_perm: int = 2000, rng=None,
            alpha: float = LOOK_ALPHA, looks=LOOKS) -> Verdict:
    """Ruling 19's primary test, two looks: A against B on lists of Outcome.

    Look 1 on the first looks[0] runs of each arm: p <= alpha -> differs;
    futility (ruling 19): <= 1 run past the mark in BOTH arms -> FUTILE.
    Else look 2 on the first looks[1] runs: p <= alpha -> differs, else
    NOT_DISTINGUISHED. `alpha` is a parameter only so that check #1's
    must-fail control can run an uncorrected (0.05 per look) version
    through the same code.
    """
    rng = np.random.default_rng(rng)
    n1, n2 = looks
    if len(a) < n2 or len(b) < n2:
        raise ValueError(f"need {n2} runs per arm")
    a1, b1 = a[:n1], b[:n1]
    p1 = perm_logrank(a1, b1, n_perm, rng)
    if p1 <= alpha:
        return Verdict("A_DIFFERS", 1, p1, float("nan"))
    if sum(o.event for o in a1) <= 1 and sum(o.event for o in b1) <= 1:
        return Verdict("FUTILE", 1, p1, float("nan"))
    p2 = perm_logrank(a[:n2], b[:n2], n_perm, rng)
    return Verdict("A_DIFFERS" if p2 <= alpha else "NOT_DISTINGUISHED", 2, p1, p2)


def check_runs_accounted(launched, analysed) -> None:
    """Check #13: runs analysed = runs launched, per arm, by run id.

    `launched`: {arm: [run_id, ...]} as written at launch time (the
    manifest, never reconstructed from what came back). `analysed`:
    {arm: {run_id: Outcome}}. A launched run with no log must be entered as
    a crash (censored at T) by the caller, never dropped (ruling 19: no seed
    replaced, a crash is an outcome). Raises ValueError naming every
    missing, extra or duplicated run; returns None when the books balance.
    """
    errs = []
    for arm in sorted(set(launched) | set(analysed)):
        ids = list(launched.get(arm, []))
        got = analysed.get(arm, {})
        if len(set(ids)) != len(ids):
            errs.append(f"{arm}: duplicate run ids in the launch manifest")
        miss = sorted(set(ids) - set(got))
        extra = sorted(set(got) - set(ids))
        if miss:
            errs.append(f"{arm}: launched but not analysed {miss}")
        if extra:
            errs.append(f"{arm}: analysed but never launched {extra}")
    if errs:
        raise ValueError("; ".join(errs))
