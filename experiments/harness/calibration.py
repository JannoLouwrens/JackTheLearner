"""J4 check #3 — calibration arms at floor and ceiling (PUBLISHED_PRIORS §8).

Source, §8 "Gates": "floor arms <= 1 success in total; scripted forager must
pass; else void. No seed replaced; NaN and crashes are outcomes." §8 stage 2
names the arms: floor = random, statue, untrained twin (and shuffled-input,
our reading: it is a must-fail arm); ceiling = the scripted forager
(record: 24-27 meals per life against theta = 5; random 0-0.2).

Our readings, where the record does not spell them out:
  * "success" = the run's primary outcome crossed by T (ruling 19), i.e.
    `Outcome.event` — the same quantity the A-vs-B test reads, so a gate
    and the test cannot disagree about what a success is.
  * "scripted forager must pass" = EVERY ceiling run crosses by T. No seed
    is replaced (§8), so a single ceiling miss voids the comparison: either
    the world, the outcome code or T is too hard for anything to show.
  * Unknown is not zero (LESSONS_CORE 11): no floor arm, no ceiling arm, or
    an empty arm -> VOID, never a silent pass.

Kept in its own file so `experiments/harness/__init__.py` — whose sha256 the
running check #1 recorded (`analysis_hash`) — is not touched.
"""
from __future__ import annotations

from dataclasses import dataclass

from experiments.harness import Outcome

FLOOR_MAX_SUCCESSES = 1   # §8: in total, across all floor arms


@dataclass(frozen=True)
class GateResult:
    valid: bool
    reasons: tuple       # empty when valid
    floor_successes: int
    ceiling_misses: int


def calibration_gate(floor: dict, ceiling: dict) -> GateResult:
    """Check #3: {arm: [Outcome, ...]} for floor and for ceiling arms.

    VALID iff at least one non-empty arm on each side, floor successes in
    total <= FLOOR_MAX_SUCCESSES, and every ceiling run crossed by T.
    """
    reasons = []
    for side, arms in (("floor", floor), ("ceiling", ceiling)):
        if not arms:
            reasons.append(f"no {side} arm")
        for arm, outs in arms.items():
            if not outs:
                reasons.append(f"{side} arm {arm!r} has no runs")
            for o in outs:
                if not isinstance(o, Outcome):
                    raise TypeError(f"{side} arm {arm!r}: expected Outcome, got {type(o).__name__}")
    fs = sum(o.event for outs in floor.values() for o in outs)
    cm = sum(not o.event for outs in ceiling.values() for o in outs)
    if fs > FLOOR_MAX_SUCCESSES:
        reasons.append(f"floor arms crossed {fs} times (> {FLOOR_MAX_SUCCESSES} in total)")
    if cm:
        reasons.append(f"ceiling arms missed {cm} run(s): the scripted forager must pass")
    return GateResult(not reasons, tuple(reasons), fs, cm)
