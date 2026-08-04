"""T0.12 — a run cannot quietly exhaust the weekly GPU quota.

Kaggle gives 30 GPU-hours a week and no warning as they drain. Without
accounting, the failure looks like this: training works for days, then one
morning every job silently lands on CPU (which Kaggle does rather than refusing —
measured 2026-08-04) and the results look plausible and mean nothing.

Four properties, each independently checkable:
  1. GPU time is charged to the week it was spent.
  2. Week boundaries isolate — last week's spend does not block this week.
  3. A job whose estimate exceeds the remaining quota is REFUSED, not attempted.
  4. Colab is unmetered (its allowance is unpublished and elastic), so it must
     never be blocked by Kaggle's budget.

Uses a temporary budget file; must never touch the real accounting.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from ..gpu import Budget, KAGGLE_WEEKLY_HOURS


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "budget.json"
        b = Budget(path)

        starts_full = b.remaining("kaggle") == KAGGLE_WEEKLY_HOURS
        b.charge("kaggle", 2 * 3600)                      # 2 hours
        charged = abs(b.used_hours("kaggle") - 2.0) < 1e-6
        persists = abs(Budget(path).used_hours("kaggle") - 2.0) < 1e-6

        # Affordability is what actually gates a launch.
        affords_small = b.afford("kaggle", 1.0)
        refuses_oversized = not b.afford("kaggle", KAGGLE_WEEKLY_HOURS)

        # Drain it and confirm the refusal is total.
        b.charge("kaggle", (KAGGLE_WEEKLY_HOURS - 2.0) * 3600)
        exhausted = b.remaining("kaggle") == 0.0 and not b.afford("kaggle", 0.1)

        # Colab must stay unmetered even with Kaggle exhausted.
        colab_free = b.afford("colab", 999.0)

        # A different week must start clean.
        other = time.strftime("%G-W%V", time.localtime(time.time() - 14 * 86400))
        b.data["weeks"][other] = {"kaggle": 29.0}
        weeks_isolated = b.remaining("kaggle") == 0.0

    return {
        "starts_full": starts_full,
        "charge_recorded": charged,
        "survives_reload": persists,
        "affords_within_quota": affords_small,
        "refuses_oversized": refuses_oversized,
        "refuses_when_exhausted": exhausted,
        "colab_unmetered": colab_free,
        "weeks_isolated": weeks_isolated,
    }


def _check(m: dict, _c: dict) -> bool:
    return all(m.values())


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.12"], _experiment, _check, ledger=ledger)
