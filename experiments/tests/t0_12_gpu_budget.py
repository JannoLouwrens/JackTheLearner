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
  6. The meter charges what was SPENT: the provider's metered window, not this
     box's wall clock; waste labelled as waste; each remote job billed once.
  7. Crossing the ceiling leaves a mark.

Uses a temporary budget file; must never touch the real accounting.

REWRITTEN 2026-08-09 after the overseer audit. The isolation property was
asserted AFTER draining the quota to its ceiling, where `remaining()` is
`max(0.0, 30.0 - 30.0)` and therefore 0.0 under every possible implementation
of week isolation — including total failure. The property was true by
construction, and the collision bug it exists to catch happened on 2026-08-08
with this spec green throughout. It is now asserted at 28.0 of 30 h, a
mid-range value that moves, and carries a `_LeakyBudget` control that must fail
on isolation specifically.

EXTENDED 2026-08-09 after the second overseer audit, which found that everything
this spec asserted was checked against synthetic `charge()` calls the test made
itself: it verified the accountant's arithmetic and had never looked at the
account. Week 31 closed at **37.4554 of a 30.0 h ceiling** — 27.73 h of it
billed inside at most 12.75 h of wall clock — and T1.02 was refused 0.7 h of GPU
by that number, with this spec green throughout. Properties 6 and 7 are the
three defects that produced it, and their control is the pre-fix `submit()` loop
reproduced verbatim.

STILL OPEN, deliberately not claimed here: nothing reconciles the meter against
Kaggle's OWN reported runtime for a kernel. That needs a live kernel and
network, which a CPU_FAST spec must not spend. What is asserted below is the
half that is decidable offline — that the charged quantity is the metered window
and not the wall clock wrapped around it.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from .. import gpu
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from ..gpu import Budget, JobResult, KAGGLE_WEEKLY_HOURS

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


def _probe_metering(make, td: Path) -> dict:
    """What gets charged, to which bucket, and how many times.

    Every property here is a defect the overseer found in the LIVE meter on
    2026-08-09, not a hypothetical. Asserted on a fresh budget so the ordering
    of the core probe above is untouched.
    """
    path = td / "metering.json"
    b = make(path)

    # (a) A crashed kernel still occupied a GPU, so the hours are real — but
    # they bought nothing, and `charge()` used to sit above `if res.ok`, which
    # made waste indistinguishable from work in the record.
    b.charge("kaggle", 3600.0, ok=False, job_id="kaggle/u/crashed")
    failed_hours_visible = abs(b.failed_hours("kaggle") - 1.0) < 1e-6
    failed_counts_against_quota = abs(b.remaining("kaggle") - (KAGGLE_WEEKLY_HOURS - 1.0)) < 1e-6
    waste_not_counted_as_work = abs(b.productive_hours("kaggle") - 0.0) < 1e-6

    # (b) JACK_REUSE_KERNEL reattaches to a kernel that is ALREADY RUNNING and
    # skips afford() because reattaching costs nothing. It then billed the whole
    # kernel a second time.
    first = b.charge("kaggle", 7200.0, ok=True, job_id="kaggle/u/job-A")
    again = b.charge("kaggle", 7200.0, ok=True, job_id="kaggle/u/job-A")
    charged_once = bool(first) and not again and abs(b.productive_hours("kaggle") - 2.0) < 1e-6

    # ...and it must be idempotent per JOB, not a blanket refusal to bill twice,
    # which would pass (b) while making the meter read zero forever.
    b.charge("kaggle", 3600.0, ok=True, job_id="kaggle/u/job-B")
    distinct_jobs_both_charged = abs(b.productive_hours("kaggle") - 3.0) < 1e-6

    # The reattach normally happens in a LATER process — a session restart is
    # what motivated the feature — so the guard has to survive a reload.
    reloaded = make(path)
    reloaded.charge("kaggle", 7200.0, ok=True, job_id="kaggle/u/job-A")
    idempotent_across_processes = abs(reloaded.productive_hours("kaggle") - 3.0) < 1e-6

    # (c) afford() gates on the DECLARED estimate, charge() bills the ACTUAL
    # elapsed time, so nothing caps an overrun. Week 31 closed 24.9% over the
    # ceiling and no artefact anywhere recorded that it had happened.
    before = len(reloaded.overruns())
    reloaded.charge("kaggle", KAGGLE_WEEKLY_HOURS * 3600.0, ok=True, job_id="kaggle/u/overrun")
    over = reloaded.overruns()
    overrun_recorded = len(over) == before + 1
    overrun_names_the_job = bool(over) and over[-1].get("job_id") == "kaggle/u/overrun"
    # An overrun must not be recorded when the ceiling was respected — otherwise
    # the marker means nothing.
    quiet = make(td / "quiet.json")
    quiet.charge("kaggle", 3600.0, ok=True, job_id="kaggle/u/small")
    no_false_overrun = len(quiet.overruns()) == 0

    return {
        "failed_hours_visible": failed_hours_visible,
        "failed_counts_against_quota": failed_counts_against_quota,
        "waste_not_counted_as_work": waste_not_counted_as_work,
        "charged_once": charged_once,
        "distinct_jobs_both_charged": distinct_jobs_both_charged,
        "idempotent_across_processes": idempotent_across_processes,
        "overrun_recorded": overrun_recorded,
        "overrun_names_the_job": overrun_names_the_job,
        "no_false_overrun": no_false_overrun,
    }


# Stub backends. The point of the exercise is the ROUTING and BILLING in
# submit(), which no live job can test cheaply: a real Kaggle run costs the very
# quota under audit. The kaggle stub reports a wall clock (3600 s) that is twice
# its metered window (1800 s) — that gap is exactly what the old meter billed.
_STUB_COLAB = dict(duration_s=120.0, job_id="colab/ladder-stub")
_STUB_KAGGLE = dict(duration_s=3600.0, billable_s=1800.0, job_id="kaggle/u/stub")


def _prefix_submit(script, prefer="colab", est_hours=0.1, gpu_name="T4",
                   timeout_s=900, fetch=None, budget=None):
    """`gpu.submit`'s billing loop EXACTLY as it stood before 2026-08-09.

    Reproduced rather than referenced because the fixed version is now the only
    one in the tree. This is the control: it must fail the billing properties.
    """
    order = ["colab", "kaggle"] if prefer == "colab" else ["kaggle", "colab"]
    for backend in order:
        if backend == "kaggle" and not budget.afford("kaggle", est_hours):
            continue
        res = (gpu.run_on_colab(script, gpu_name, timeout_s, fetch) if backend == "colab"
               else gpu.run_on_kaggle(script, timeout_s, fetch))
        budget.charge(backend, res.duration_s)      # unconditional, unlabelled, unguarded
        if res.ok:
            return res
    return JobResult(order[-1], False, message="all backends failed")


def _probe_submit(submit_fn, budget: Budget) -> dict:
    """The wiring: does submit() hand the meter the right number, once?"""
    real_colab, real_kaggle = gpu.run_on_colab, gpu.run_on_kaggle
    gpu.run_on_colab = lambda *a, **k: JobResult("colab", False, message="stub: no VM",
                                                 **_STUB_COLAB)
    gpu.run_on_kaggle = lambda *a, **k: JobResult("kaggle", True, **_STUB_KAGGLE)
    try:
        with tempfile.TemporaryDirectory() as sd:
            script = Path(sd) / "job.py"
            script.write_text("print('stub')\n")
            submit_fn(script, prefer="colab", est_hours=0.1, budget=budget)
            kaggle_after_one = budget.productive_hours("kaggle")
            colab_waste = budget.failed_hours("colab")
            colab_work = budget.productive_hours("colab")
            # Same two job ids: a re-run of the identical remote work.
            submit_fn(script, prefer="colab", est_hours=0.1, budget=budget)
            kaggle_after_two = budget.productive_hours("kaggle")
    finally:
        gpu.run_on_colab, gpu.run_on_kaggle = real_colab, real_kaggle

    return {
        # 1800 s metered, not the 3600 s of wall clock around it.
        "submit_charges_metered_window": abs(kaggle_after_one - 0.5) < 1e-4,
        # The colab attempt failed; its 120 s are waste, and must read as waste.
        "submit_buckets_failure_as_waste": (abs(colab_waste - 120.0 / 3600.0) < 1e-4
                                            and colab_work == 0.0),
        "submit_reattach_is_free": abs(kaggle_after_two - kaggle_after_one) < 1e-9,
    }


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "budget.json"
        out = _probe(Budget(path), path)
        out.update(_probe_metering(Budget, Path(td)))
        out.update(_probe_submit(gpu.submit, Budget(Path(td) / "submit.json")))
        return out


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


class _PreFixBudget(Budget):
    """`Budget.charge` EXACTLY as it stood before 2026-08-09.

    Unconditional (waste billed as work), unlabelled (no failure bucket) and
    unguarded (no per-job idempotency, no overrun mark). This is not a
    hypothetical broken meter — it is the one that closed week 31 at 37.4554 of
    30.0 h and refused T1.02 its 0.7 h. It must fail the billing properties.
    """

    def charge(self, backend: str, seconds: float, *,
               ok: bool = True, job_id: str = "") -> bool:
        wk = self.data["weeks"].setdefault(self._week(), {})
        wk[backend] = round(wk.get(backend, 0.0) + seconds / 3600.0, 4)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        return True


def _control(seed: int) -> dict:
    """Two named broken meters, each answering for the properties it breaks.

    `_LeakyBudget` is the 2026-08-08 week-key collision; `_PreFixBudget` is the
    2026-08-09 billing defect. They are kept separate rather than merged into
    one omni-broken fixture so that `_check` can name which failure proves which
    assertion is discriminating.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "budget.json"
        out = _probe(_LeakyBudget(path), path)
        out.update(_probe_metering(_PreFixBudget, Path(td)))
        out.update(_probe_submit(_prefix_submit, _PreFixBudget(Path(td) / "submit.json")))
        return out


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
    # What the meter charges, to which bucket, how often, and whether crossing
    # the ceiling leaves a mark.
    billing_ok = (
        m["failed_hours_visible"]
        and m["failed_counts_against_quota"]
        and m["waste_not_counted_as_work"]
        and m["charged_once"]
        and m["distinct_jobs_both_charged"]
        and m["idempotent_across_processes"]
        and m["overrun_recorded"]
        and m["overrun_names_the_job"]
        and m["no_false_overrun"]
        and m["submit_charges_metered_window"]
        and m["submit_buckets_failure_as_waste"]
        and m["submit_reattach_is_free"]
    )
    # Each control must fail on the property it exists to break — a control that
    # tripped on something unrelated would leave the assertion untested.
    # `_LeakyBudget` answers for isolation:
    isolation_detected = (not c["weeks_isolated"]) and (not c["stale_format_key_isolated"])
    # `_PreFixBudget` and the pre-fix submit() loop answer for billing:
    billing_detected = (
        (not c["failed_hours_visible"])
        and (not c["waste_not_counted_as_work"])
        and (not c["charged_once"])
        and (not c["idempotent_across_processes"])
        and (not c["overrun_recorded"])
        and (not c["submit_charges_metered_window"])
        and (not c["submit_buckets_failure_as_waste"])
        and (not c["submit_reattach_is_free"])
    )
    return experiment_ok and billing_ok and isolation_detected and billing_detected


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.12"], _experiment, _check, control_fn=_control, ledger=ledger)
