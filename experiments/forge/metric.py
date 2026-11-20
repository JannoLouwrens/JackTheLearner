"""The forge's per-run score for rungs S and M — a metric that REQUIRES EATING.

`sustained_rate`: the share of a run's LATE lives that were each sustained,
by the harness's own mark (`experiments.harness.sustained_life`, ruling 22:
the life reached its time cap AND food intake >= basal drain). A life that is
merely long scores nothing. Higher is better; 0 until a learner eats.

  late lives   lives that END after `late_frac` of the run's decisions
               (default the last half), so early luck is not counted.
  crash / NaN  a run whose lives end in "crash" or hit a NaN/negative
               decision count scores 0.0 (ruling 19: it counts against its arm).
  no late life 0.0.

`metric_hash()` covers this file AND the harness file that defines the mark;
a round freezes it in its rule, and a result scored by other code is refused.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import experiments.harness as harness

_FILES = (Path(harness.__file__).resolve(), Path(__file__).resolve())


def metric_hash(files=_FILES) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(hashlib.sha256(Path(f).read_bytes()).digest())
    return h.hexdigest()


def sustained_rate(lives, total_decisions: float, late_frac: float = 0.5) -> float:
    """`lives`: [(decisions_in_life, capped, intake, drain), ...] in order, as
    `harness.outcome_sustained` reads them; `total_decisions`: the rung's size."""
    if not total_decisions > 0 or not 0.0 <= late_frac < 1.0:
        raise ValueError("total_decisions > 0 and 0 <= late_frac < 1 required")
    start, cum, late, good = late_frac * total_decisions, 0.0, 0, 0
    for life in lives:
        if isinstance(life, str):
            return 0.0
        d = float(life[0])
        if math.isnan(d) or d < 0:
            return 0.0
        cum += d
        if cum > total_decisions:
            break
        if cum > start:
            late += 1
            good += harness.sustained_life(life[1], life[2], life[3])
    return good / late if late else 0.0
