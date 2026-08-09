"""T0.12 — a run cannot quietly exhaust the weekly GPU quota.

Kaggle gives 30 GPU-hours a week and no warning as they drain. Without
accounting, the failure looks like this: training works for days, then one
morning every job silently lands on CPU (which Kaggle does rather than refusing —
measured 2026-08-04) and the results look plausible and mean nothing.

Properties, each independently checkable:
  1. GPU time is charged to the week it was spent.
  2. Week boundaries isolate — last week's spend does not block this week,
     including for keys left behind in the RETIRED ISO format.
  3. A job whose estimate exceeds the remaining quota is REFUSED, not attempted.
  4. Colab is unmetered (its allowance is unpublished and elastic), so it must
     never be blocked by Kaggle's budget.
  5. This test builds week keys the way the live `Budget._week()` does.

Uses a temporary budget file; must never touch the real accounting.

REWRITTEN 2026-08-09 after the overseer audit. The isolation property was
asserted AFTER draining the quota to its ceiling, where `remaining()` is
`max(0.0, 30.0 - 30.0)` and therefore 0.0 under every possible implementation
of week isolation — including total failure. The property was true by
construction, and the collision bug it exists to catch happened on 2026-08-08
with this spec green throughout. It is now asserted at 28.0 of 30 h, a
mid-range value that moves, and carries a `_LeakyBudget` control that must fail
on isolation specifically.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from ..gpu import Budget, KAGGLE_WEEKLY_HOURS

LIVE_WEEK_FMT = "%Y-W%U"       # what Budget._week() produces (Sunday-start)
RETIRED_WEEK_FMT = "%G-W%V"    # the ISO key format retired on 2026-08-08


def _foreign_week(days_ago: int, fmt: str) -> str:
    return time.strftime(fmt, time.localtime(time.time() - days_ago * 86400))


def _probe(b: Budget, path: Path) -> dict:
    """Run the property battery against a Budget-like object.

    Ordering is load-bearing. Isolation is asserted while the quota is at a
    MID-RANGE value, never after draining it: `remaining()` is
    `max(0.0, 30.0 - used)`, so once `used` is 30.0 the assertion
    `remaining() == 0.0` holds under every possible implementation — including
    one that sums every week in the file. That is how the original version of
    this test stayed green through the 2026-08-08 week-key collision, which a
    human found by reading the budget file by hand.
    """
    starts_full = b.remaining("kaggle") == KAGGLE_WEEKLY_HOURS
    b.charge("kaggle", 2 * 3600)                      # 2 hours
    charged = abs(b.used_hours("kaggle") - 2.0) < 1e-6
    persists = abs(Budget(path).used_hours("kaggle") - 2.0) < 1e-6

    # Affordability is what actually gates a launch.
    affords_small = b.afford("kaggle", 1.0)
    refuses_oversized = not b.afford("kaggle", KAGGLE_WEEKLY_HOURS)

    # ISOLATION, measured where the answer can still move. 2h spent, so a
    # correct implementation reads exactly 28.0. A leaking one that summed the
    # injected foreign week would read max(0, 30 - 31) = 0.0.
    # The test must build keys the way the LIVE code does. The previous version
    # used the retired ISO format, so it wrote into a key space Budget._week()
    # no longer produces — the assertion could not have touched the mechanism
    # even if it had been discriminating.
    live_key = _foreign_week(0, LIVE_WEEK_FMT)
    fmt_matches_live_code = live_key == Budget._week()

    foreign = _foreign_week(14, LIVE_WEEK_FMT)
    assert foreign != live_key, "foreign week key collided with the live one"
    b.data["weeks"][foreign] = {"kaggle": 29.0}
    weeks_isolated = abs(b.remaining("kaggle") - 28.0) < 1e-6

    # The bug that actually happened: hours filed under the RETIRED ISO key
    # format (%G-W%V, Monday-start) left behind by the 2026-08-08 migration.
    # A foreign-week key in the retired format must be just as inert.
    stale = _foreign_week(21, RETIRED_WEEK_FMT)
    stale_collides = stale == live_key
    b.data["weeks"][stale] = {"kaggle": 29.0}
    stale_format_isolated = abs(b.remaining("kaggle") - 28.0) < 1e-6

    # Drain it and confirm the refusal is total. Assert on `used_hours`, which
    # keeps counting, as well as on the clamped `remaining`.
    b.charge("kaggle", (KAGGLE_WEEKLY_HOURS - 2.0) * 3600)
    drained_exactly = abs(b.used_hours("kaggle") - KAGGLE_WEEKLY_HOURS) < 1e-6
    exhausted = b.remaining("kaggle") == 0.0 and not b.afford("kaggle", 0.1)

    # Colab must stay unmetered even with Kaggle exhausted.
    colab_free = b.afford("colab", 999.0)

    return {
        "starts_full": starts_full,
        "charge_recorded": charged,
        "survives_reload": persists,
        "affords_within_quota": affords_small,
        "refuses_oversized": refuses_oversized,
        "drained_to_exact_ceiling": drained_exactly,
        "refuses_when_exhausted": exhausted,
        "colab_unmetered": colab_free,
        "weeks_isolated": weeks_isolated,
        "stale_format_key_isolated": stale_format_isolated,
        "stale_key_collided_with_live": stale_collides,
        "test_key_format_matches_live_code": fmt_matches_live_code,
    }


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "budget.json"
        return _probe(Budget(path), path)


class _LeakyBudget(Budget):
    """Week isolation deliberately broken: every week's hours are summed.

    This is the failure the 08-08 collision produced in practice — a foreign
    key's hours counted against the current quota, so a fresh 30 h week read as
    37.5 h used and refused the job the whole plan depended on. The control
    exists to prove the measurement can SEE that, rather than only that we
    observed the good thing.
    """

    def used_hours(self, backend: str) -> float:
        return float(sum(w.get(backend, 0.0) for w in self.data["weeks"].values()))


def _control(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "budget.json"
        return _probe(_LeakyBudget(path), path)


def _check(m: dict, c: dict) -> bool:
    # Explicit conjunction of named booleans, not `all(m.values())` — the
    # latter passes on ANY truthy value, so a metric that started returning a
    # non-empty string would silently satisfy the gate.
    experiment_ok = (
        m["starts_full"]
        and m["charge_recorded"]
        and m["survives_reload"]
        and m["affords_within_quota"]
        and m["refuses_oversized"]
        and m["drained_to_exact_ceiling"]
        and m["refuses_when_exhausted"]
        and m["colab_unmetered"]
        and m["weeks_isolated"]
        and m["stale_format_key_isolated"]
        and not m["stale_key_collided_with_live"]
        and m["test_key_format_matches_live_code"]
    )
    # The control must fail, and must fail on ISOLATION specifically — if it
    # tripped on some unrelated property the isolation assertion would still be
    # untested.
    control_detected = (not c["weeks_isolated"]) and (not c["stale_format_key_isolated"])
    return experiment_ok and control_detected


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.12"], _experiment, _check, control_fn=_control, ledger=ledger)
