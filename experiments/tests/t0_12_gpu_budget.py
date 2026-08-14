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
  8. A dispatch leaves a receipt written BEFORE the remote call, and a backend
     the budget skipped leaves none — so absence means not-dispatched.
  9. A reattached kernel is billed the window KAGGLE reports it ran, not the
     idle hours until the local process came back to look.
 10. A stale writer cannot erase another process's charges, and idempotency is
     judged against the file, not against a copy loaded hours ago.
 11. A dispatch is attributable: the receipt names the SPEC whose runs bought
     the hours, and the job ids are recoverable by the recorder
     (`gpu.drain_job_ids()`), so the ledger and the receipt log share a join
     key instead of being reconciled by timestamp arithmetic.

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

EXTENDED 2026-08-11 after the seventh overseer audit, with property 8. Every
property above audits the METER; none audited whether a dispatch happened at
all. Commit `6b001e7` handed off a claim that a T1.02 GPU poll was in flight
when nothing had been submitted, and that claim was contradicted by nothing a
gate reads: the budget file was unchanged (= "nothing spent"), the ledger was
unchanged (= "not run"), and the prose was the only witness. `submit()` now
leaves a receipt, and the pre-2026-08-11 dispatch loop is the control.

EXTENDED 2026-08-12 after the tenth overseer audit, with properties 9 and 10.
Property 8's `submit_reattach_is_free` calls `charge()` twice WITH AN AMOUNT THE
TEST SUPPLIES, so it gates the ledger of charges and could never see a meter
that derives the wrong amount: `run_on_kaggle`'s reuse path opened the meter at
the slug's submission epoch and closed it at `time.time()` — the moment the
local process noticed — billing 35 330 s for a kernel whose own metered window
was 2 361.88 s (14.96x). The idempotency key never covered it: it fires only
when the original poll already charged, and JACK_REUSE_KERNEL exists for the
poll that died before charging. Property 9 therefore drives the SHIPPED
`run_on_kaggle` (CLI stubbed at `gpu._run`) with a kernel that finished long
before the reattach, and the pre-fix meter-closing is the control that fails
it. Property 10 is the same audit's second find: `Budget.charge()` wrote the
whole file from state loaded at construction, so the T2.01 poll's 12:59 write
erased a colab charge made at 08:17 (0.5498 h, repaired by hand in dd7186b) —
the stale-writer clobber `Ledger.record` fixed on 2026-08-10 and the meter
never learned.

EXTENDED 2026-08-14 (overseer B3, carried three audits), with property 11. The
receipt log and the ledger shared no field, so "which hours bought which
result" was answerable only by timestamp arithmetic — the 15th audit did the
reconciliation by hand and called the fact that it worked "a coincidence of
durations, not an audit trail". `submit()` now writes the spec id (from
JACK_SPEC_ID, set by `run_spec`) into every receipt, and records job ids for
`run_spec` to drain into `Result.gpu_job_id`. The pre-2026-08-11 dispatch loop
is again the control: it writes nothing and bypasses `submit()`, so both
halves of attribution must read as failures under it.

STILL OPEN, deliberately not claimed here: nothing reconciles a FRESH
submission's poll-window charge against Kaggle's own report — that needs a live
kernel and network, which a CPU_FAST spec must not spend. Property 9 closes the
reattach half offline, because there the provider's report (the console log's
record stamps) is already on disk. What is asserted below is the half that is
decidable offline.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .. import gpu
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from ..gpu import Budget, JobResult, KAGGLE_WEEKLY_HOURS

# Every property here is a property of `experiments/gpu.py`, so a change to that
# file must retire this certificate. Without this line the meter could be
# rewritten and T0.12 would go on reading PASS against code it never saw — the
# same class of blindness `impl_sha` exists to close, one level out. T0.12 is
# CPU_FAST, so re-earning it costs seconds; the other GPU specs (T0.09/T0.10/
# T0.11) have the same gap and clearing it costs real GPU quota, so it is left
# for an iteration that can spend it. Named here so it is not lost.
IMPL_DEPS = ["experiments/gpu.py"]

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
                   timeout_s=900, fetch=None, budget=None, journal=None):
    """`gpu.submit`'s billing loop EXACTLY as it stood before 2026-08-09.

    Reproduced rather than referenced because the fixed version is now the only
    one in the tree. This is the control: it must fail the billing properties.

    It accepts `journal` and ignores it. The pre-fix loop had no such parameter
    at all; the kwarg exists only so the control can be called through the same
    signature as the fixed function. What is being controlled for is that this
    version WRITES NOTHING — a dispatch it makes is indistinguishable from a
    dispatch that never happened, which is the 2026-08-11 defect (property 8).
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


def _probe_submit(submit_fn, budget: Budget, journal: Path) -> dict:
    """The wiring: does submit() hand the meter the right number, once?

    `journal` is not optional and there is a scar behind that. When the receipt
    log landed (2026-08-11) this probe still called `submit()` with the default
    journal, so running T0.12 appended STUB receipts — `kaggle/u/stub`, a job
    that never existed — to the real `experiments/gpu_submissions.jsonl`. An
    evidence file a test can write fiction into is not evidence. Same rule as
    `budget`: a function that hard-codes the path to the record it mutates
    cannot be tested except by corrupting it, and that rule now covers the
    receipt log as well as the meter.
    """
    real_colab, real_kaggle = gpu.run_on_colab, gpu.run_on_kaggle
    gpu.run_on_colab = lambda *a, **k: JobResult("colab", False, message="stub: no VM",
                                                 **_STUB_COLAB)
    gpu.run_on_kaggle = lambda *a, **k: JobResult("kaggle", True, **_STUB_KAGGLE)
    try:
        with tempfile.TemporaryDirectory() as sd:
            script = Path(sd) / "job.py"
            script.write_text("print('stub')\n")
            submit_fn(script, prefer="colab", est_hours=0.1, budget=budget,
                      journal=journal)
            kaggle_after_one = budget.productive_hours("kaggle")
            colab_waste = budget.failed_hours("colab")
            colab_work = budget.productive_hours("colab")
            # Same two job ids: a re-run of the identical remote work.
            submit_fn(script, prefer="colab", est_hours=0.1, budget=budget,
                      journal=journal)
            kaggle_after_two = budget.productive_hours("kaggle")
    finally:
        gpu.run_on_colab, gpu.run_on_kaggle = real_colab, real_kaggle
        # Stub dispatches must not leave their job ids for the recorder to
        # fold into THIS spec's ledger record as if they were real jobs.
        gpu.drain_job_ids()

    return {
        # 1800 s metered, not the 3600 s of wall clock around it.
        "submit_charges_metered_window": abs(kaggle_after_one - 0.5) < 1e-4,
        # The colab attempt failed; its 120 s are waste, and must read as waste.
        "submit_buckets_failure_as_waste": (abs(colab_waste - 120.0 / 3600.0) < 1e-4
                                            and colab_work == 0.0),
        "submit_reattach_is_free": abs(kaggle_after_two - kaggle_after_one) < 1e-9,
    }


def _probe_receipt(submit_fn, make, td: Path) -> dict:
    """Does a dispatch leave evidence that it happened?

    THE SCAR: commit `6b001e7` (2026-08-11) handed off a claim that a `T1.02`
    GPU poll was in flight when nothing had been submitted. The claim survived
    every gate the project owns — an unchanged `gpu_budget.json` reads as
    "nothing spent", an unchanged ledger reads as "not run", and the only
    contradiction was prose no gate reads. An iteration that never called
    `submit()` and one whose submission died mid-flight left byte-identical
    evidence, so the absence of a receipt could not be interpreted.

    Asserted here in both directions, because only the pair is useful: a
    dispatch must leave a receipt BEFORE the remote call (so absence means "not
    dispatched" rather than "died early"), and a backend the budget SKIPPED must
    leave none (so presence means "actually dispatched" rather than "intended
    to"). One without the other is a log, not evidence.
    """
    td.mkdir(parents=True, exist_ok=True)
    log = td / "submissions.jsonl"
    seen: dict[str, int] = {}
    real_colab, real_kaggle = gpu.run_on_colab, gpu.run_on_kaggle

    def _stub(backend, ok, extra):
        def _fn(*a, **k):
            # How much was already on DISK at the instant the remote call was
            # made. Read through the public reader, not the writer's own state.
            seen[backend] = len(gpu.submissions(log))
            return JobResult(backend, ok, message=f"stub: {backend}", **extra)
        return _fn

    with tempfile.TemporaryDirectory() as sd:
        script = Path(sd) / "job.py"
        script.write_text("print('stub')\n")

        # (a) colab fails, kaggle succeeds: the ordinary failover. Run under a
        # fixture spec id, the way run_spec sets one around a real spec's
        # seed loop, so attribution (property 11) is measured on this case.
        gpu.run_on_colab = _stub("colab", False, _STUB_COLAB)
        gpu.run_on_kaggle = _stub("kaggle", True, _STUB_KAGGLE)
        prev_spec = os.environ.get("JACK_SPEC_ID")
        os.environ["JACK_SPEC_ID"] = "T9.99-fixture"
        gpu.drain_job_ids()      # start clean: earlier probes leave stub ids
        try:
            submit_fn(script, prefer="colab", est_hours=0.1,
                      budget=make(td / "receipt_a.json"), journal=log)
        finally:
            gpu.run_on_colab, gpu.run_on_kaggle = real_colab, real_kaggle
            if prev_spec is None:
                os.environ.pop("JACK_SPEC_ID", None)
            else:
                os.environ["JACK_SPEC_ID"] = prev_spec
        drained = gpu.drain_job_ids()
        recs = gpu.submissions(log)
        attempts = [r for r in recs if r.get("phase") == "attempt"]
        results = [r for r in recs if r.get("phase") == "result"]

        before_dispatch = seen.get("colab", 0) >= 1 and seen.get("kaggle", 0) >= 3
        paired = (len(attempts) == 2 and len(results) == 2
                  and {a.get("attempt_id") for a in attempts}
                  == {r.get("attempt_id") for r in results})
        names_job = any(r.get("job_id") == "kaggle/u/stub" and r.get("ok") is True
                        for r in results)
        # Property 11, both halves. The failed colab attempt must be recovered
        # too — it spent (stub) hours, and a record naming only the job that
        # succeeded cannot answer "which hours bought this result".
        names_spec = (len(attempts) == 2
                      and all(a.get("spec") == "T9.99-fixture" for a in attempts))
        recorder_recovers = ("kaggle/u/stub" in drained
                             and "colab/ladder-stub" in drained)

        # (b) every backend fails: the receipt must survive the fact that
        # nothing came back. This is the case the false handoff resembled.
        log_b = td / "submissions_b.jsonl"
        gpu.run_on_colab = _stub("colab", False, _STUB_COLAB)
        gpu.run_on_kaggle = _stub("kaggle", False, _STUB_KAGGLE)
        try:
            submit_fn(script, prefer="colab", est_hours=0.1,
                      budget=make(td / "receipt_b.json"), journal=log_b)
        finally:
            gpu.run_on_colab, gpu.run_on_kaggle = real_colab, real_kaggle
        recs_b = gpu.submissions(log_b)
        survives_failure = (len([r for r in recs_b if r.get("phase") == "attempt"]) == 2
                            and gpu.last_submission(log_b) is not None)

        # (c) THE FALSE-POSITIVE HALF. Drain kaggle so `afford()` skips it, then
        # prefer kaggle. Only colab is really dispatched to, so only colab may
        # appear. An implementation that logged its INTENDED order up front
        # would pass (a) and (b) and fail here — and would have re-created the
        # exact defect, a record claiming a submission that never happened.
        drained = make(td / "receipt_c.json")
        drained.charge("kaggle", KAGGLE_WEEKLY_HOURS * 3600.0, ok=True,
                       job_id="kaggle/u/drain")
        log_c = td / "submissions_c.jsonl"
        gpu.run_on_colab = _stub("colab", False, _STUB_COLAB)
        gpu.run_on_kaggle = _stub("kaggle", True, _STUB_KAGGLE)
        try:
            submit_fn(script, prefer="kaggle", est_hours=1.0,
                      budget=drained, journal=log_c)
        finally:
            gpu.run_on_colab, gpu.run_on_kaggle = real_colab, real_kaggle
        backends_c = {r.get("backend") for r in gpu.submissions(log_c)}
        skipped_unlogged = backends_c == {"colab"}

    # Cases (b) and (c) dispatched through stubs too; their ids must not reach
    # the ledger through the recorder's post-run drain.
    gpu.drain_job_ids()

    return {
        "receipt_written_before_dispatch": before_dispatch,
        "receipt_pairs_attempt_with_result": paired,
        "receipt_names_the_job": names_job,
        "receipt_names_the_spec": names_spec,
        "recorder_recovers_job_ids": recorder_recovers,
        "receipt_survives_all_backends_failing": survives_failure,
        "no_receipt_for_skipped_backend": skipped_unlogged,
    }


_REUSE_WINDOW_S = 2361.88     # what the fixture kernel's log says it ran — the
                              # real number from jack-ladder-1786482462
_REUSE_IDLE_S = 10 * 3600.0   # how long before the reattach it was submitted


class _NoSleepTime:
    """`time` facade for gpu.py: real clock, free sleep.

    `run_on_kaggle` polls on a 20 s cadence and the fixture kernel is terminal
    on the first check. The property under test is about the METER, which must
    see the REAL clock — a faked clock would be feeding in the very quantity
    whose derivation is under audit.
    """

    def __getattr__(self, name):
        return getattr(time, name)

    @staticmethod
    def sleep(_s):
        return None


class _FakeSubprocess:
    """Answers `kaggle config view` so the probe never needs the live CLI."""

    @staticmethod
    def run(cmd, **kw):
        class _R:
            returncode = 0
            stdout = "Configuration values:\n  - username: stubuser\n"
            stderr = ""
        return _R()


def _probe_reattach() -> dict:
    """Drive the SHIPPED `run_on_kaggle` through a reattach to a long-dead kernel.

    THE SCAR (10th overseer audit, 2026-08-12): on reuse the meter opened at the
    slug's submission epoch and closed at `time.time()` — the moment the local
    process noticed — so recovering `jack-ladder-1786482462` computed a charge
    of 35 330 s for a kernel whose own metered window was 2 361.88 s (14.96x).
    It never reached the budget file only because the idempotency check fired
    first, and that check only fires when the original poll ALREADY charged —
    the one case JACK_REUSE_KERNEL does not exist for.

    Property 8's `submit_reattach_is_free` feeds `charge()` an amount the test
    supplies, so it can gate the ledger of charges and never how the amount was
    derived. This probe drives the deriving path itself: only the Kaggle CLI
    (`gpu._run`), the `config view` subprocess and the poll cadence are stubbed;
    slug parsing, the meter, and log collection are the shipped code.
    """
    epoch = int(time.time() - _REUSE_IDLE_S)
    slug = f"jack-ladder-{epoch}"
    records = [
        {"stream_name": "stdout", "time": 0.9, "data": "boot\n"},
        {"stream_name": "stdout", "time": _REUSE_WINDOW_S, "data": "RESULT {}\n"},
    ]

    def _fake_cli(write_log: bool):
        def _fn(cmd, timeout):
            if "status" in cmd:
                return 0, "complete", ""
            if "output" in cmd:
                if write_log:
                    kernel_slug = cmd[cmd.index("output") + 1].split("/")[-1]
                    outdir = Path(cmd[cmd.index("-p") + 1])
                    (outdir / f"{kernel_slug}.log").write_text(json.dumps(records))
                return 0, "", ""
            return 0, "pushed", ""   # `kernels push`, fresh-submission case
        return _fn

    real = (gpu._run, gpu.subprocess, gpu.time)
    try:
        gpu.subprocess = _FakeSubprocess()
        gpu.time = _NoSleepTime()
        with tempfile.TemporaryDirectory() as sd:
            script = Path(sd) / "job.py"
            script.write_text("print('stub')\n")

            gpu._run = _fake_cli(write_log=True)
            os.environ["JACK_REUSE_KERNEL"] = slug
            try:
                reattach = gpu.run_on_kaggle(script, timeout_s=120)
            finally:
                del os.environ["JACK_REUSE_KERNEL"]

            gpu._run = _fake_cli(write_log=False)
            os.environ["JACK_REUSE_KERNEL"] = slug
            try:
                no_log = gpu.run_on_kaggle(script, timeout_s=120)
            finally:
                del os.environ["JACK_REUSE_KERNEL"]

            gpu._run = _fake_cli(write_log=True)
            fresh = gpu.run_on_kaggle(script, timeout_s=120)
    finally:
        gpu._run, gpu.subprocess, gpu.time = real

    return {
        # The kernel's own report closes the window: 2 361.88 s, not ~36 000 s.
        "reattach_bills_kernel_window":
            abs(reattach.billable_s - _REUSE_WINDOW_S) < 1.0,
        # With no readable log the fallback is the local window — an over-count,
        # kept deliberately as an UPPER BOUND (under-billing is how afford()
        # approves a job Kaggle then refuses) and announced on stderr.
        "reattach_missing_log_falls_back_upper_bound":
            no_log.billable_s >= _REUSE_IDLE_S - 60.0,
        # A fresh submission still bills its poll window, not the log's stamps
        # and not anything anchored to the slug epoch.
        "fresh_bills_poll_window":
            0.0 <= fresh.billable_s < 60.0,
    }


def _prefix_reattach_meter(slug: str, reuse: bool) -> float:
    """`run_on_kaggle`'s meter EXACTLY as it stood before 2026-08-12.

    The meter opens at the slug's submission epoch on reuse and closes at
    `time.time()`. The poll loop is elided because the fixture kernel is
    terminal on the first status check and nothing in the loop touches the
    meter. This is the control: fed a kernel that finished long ago, it must
    bill the idle hours and fail the window property.
    """
    t0 = time.time()
    if reuse:
        try:
            t0 = float(slug.rsplit("-", 1)[-1])
        except ValueError:
            pass
    t_meter_open = t0 if reuse else time.time()
    return time.time() - t_meter_open


def _probe_reattach_prefix() -> dict:
    """The pre-fix meter run on the same fixture scenario as `_probe_reattach`."""
    epoch = int(time.time() - _REUSE_IDLE_S)
    slug = f"jack-ladder-{epoch}"
    reused = _prefix_reattach_meter(slug, reuse=True)
    fresh = _prefix_reattach_meter(slug, reuse=False)
    return {
        "reattach_bills_kernel_window": abs(reused - _REUSE_WINDOW_S) < 1.0,
        "reattach_missing_log_falls_back_upper_bound": reused >= _REUSE_IDLE_S - 60.0,
        "fresh_bills_poll_window": 0.0 <= fresh < 60.0,
    }


def _probe_concurrent(make, td: Path) -> dict:
    """Two writers on one meter; the stale one must not erase the fresh one.

    THE SCAR (2026-08-12): `submit()` builds its Budget at dispatch and charges
    when the job returns, so the T2.01 poll held a 07:24 view of the file for
    5.6 h and its 12:59 write erased the colab charge another iteration had
    recorded at 08:17 — 0.5498 h and its charged_jobs entry gone, repaired by
    hand in dd7186b. This is the same stale-writer clobber `Ledger.record`
    fixed on 2026-08-10; its docstring names the disease, and the meter never
    learned it. The amounts below are the real incident's.
    """
    path = td / "concurrent.json"
    poll = make(path)                       # dispatch time: loads the empty file
    other = make(path)
    other.charge("colab", 1979.0, ok=True, job_id="colab/ladder-mid")     # 08:17
    poll.charge("kaggle", 20087.0, ok=True, job_id="kaggle/u/long-poll")  # 12:59
    fresh = make(path)
    survives = (abs(fresh.productive_hours("colab") - 1979.0 / 3600.0) < 1e-3
                and abs(fresh.productive_hours("kaggle") - 20087.0 / 3600.0) < 1e-3
                and "colab/ladder-mid" in fresh.data["charged_jobs"])

    # Idempotency consulted the same stale copy: a job another process already
    # billed must not bill again here, and only the FILE can know about it.
    early = make(path)                      # loads before the duplicate exists
    late = make(path)
    late.charge("kaggle", 3600.0, ok=True, job_id="kaggle/u/dup")
    billed_again = early.charge("kaggle", 3600.0, ok=True, job_id="kaggle/u/dup")
    total = make(path).productive_hours("kaggle")
    idempotent = (not billed_again
                  and abs(total - (20087.0 / 3600.0 + 1.0)) < 1e-3)

    return {
        "concurrent_charge_survives": survives,
        "concurrent_idempotency_reads_disk": idempotent,
    }


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "budget.json"
        out = _probe(Budget(path), path)
        out.update(_probe_metering(Budget, Path(td)))
        out.update(_probe_submit(gpu.submit, Budget(Path(td) / "submit.json"),
                                 Path(td) / "submit_receipts.jsonl"))
        out.update(_probe_receipt(gpu.submit, Budget, Path(td) / "receipt"))
        out.update(_probe_reattach())
        out.update(_probe_concurrent(Budget, Path(td)))
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


class _ClobberBudget(Budget):
    """`Budget.charge` EXACTLY as it stood before 2026-08-12.

    No re-read: the whole file is written from `self.data`, loaded at
    construction, and idempotency is judged against that same stale copy. This
    is the meter whose 12:59 write erased the 08:17 colab charge. The overrun
    block is elided — nothing in the concurrency properties reaches it. It must
    fail both concurrency properties.
    """

    def charge(self, backend: str, seconds: float, *,
               ok: bool = True, job_id: str = "") -> bool:
        if job_id and job_id in self.data["charged_jobs"]:
            return False
        key = backend if ok else backend + FAILED_SUFFIX
        wk = self.data["weeks"].setdefault(self._week(), {})
        wk[key] = round(wk.get(key, 0.0) + seconds / 3600.0, 4)
        if job_id:
            self.data["charged_jobs"][job_id] = {
                "week": self._week(), "backend": backend,
                "hours": round(seconds / 3600.0, 4), "ok": ok,
            }
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        return True


def _control(seed: int) -> dict:
    """Two named broken meters, each answering for the properties it breaks.

    `_LeakyBudget` is the 2026-08-08 week-key collision; `_PreFixBudget` is the
    2026-08-09 billing defect; `_prefix_submit` run against a HEALTHY `Budget`
    is the 2026-08-11 missing-receipt defect — the meter is deliberately the
    good one there, so the only variable is the dispatch loop's silence. They
    are kept separate rather than merged into one omni-broken fixture so that
    `_check` can name which failure proves which assertion is discriminating.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "budget.json"
        out = _probe(_LeakyBudget(path), path)
        out.update(_probe_metering(_PreFixBudget, Path(td)))
        out.update(_probe_submit(_prefix_submit, _PreFixBudget(Path(td) / "submit.json"),
                                 Path(td) / "submit_receipts.jsonl"))
        out.update(_probe_receipt(_prefix_submit, Budget, Path(td) / "receipt"))
        out.update(_probe_reattach_prefix())
        out.update(_probe_concurrent(_ClobberBudget, Path(td)))
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
    # Property 9: the reattach meter bills the window Kaggle reports, keeps the
    # local window only as an announced upper bound, and leaves the fresh path's
    # poll-window billing untouched.
    reattach_ok = (
        m["reattach_bills_kernel_window"]
        and m["reattach_missing_log_falls_back_upper_bound"]
        and m["fresh_bills_poll_window"]
    )
    # Property 10: a stale writer cannot erase another process's charges, and
    # idempotency is judged against the file.
    concurrent_ok = (
        m["concurrent_charge_survives"]
        and m["concurrent_idempotency_reads_disk"]
    )
    # Property 8: a dispatch leaves a receipt, and only a real dispatch does.
    # Property 11: the receipt names its spec, and the recorder can recover
    # the job ids to fold into the ledger record.
    receipt_ok = (
        m["receipt_written_before_dispatch"]
        and m["receipt_pairs_attempt_with_result"]
        and m["receipt_names_the_job"]
        and m["receipt_names_the_spec"]
        and m["recorder_recovers_job_ids"]
        and m["receipt_survives_all_backends_failing"]
        and m["no_receipt_for_skipped_backend"]
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
    # The pre-fix dispatch loop answers for the receipt. `no_receipt_for_skipped
    # _backend` is deliberately NOT required to fail: the control writes nothing
    # at all, so it satisfies that property vacuously. Requiring it would mean
    # demanding a control fail a property it cannot express, which is how a
    # conjunction ends up asserting the control is broken rather than that the
    # measurement is discriminating.
    receipt_detected = (
        (not c["receipt_written_before_dispatch"])
        and (not c["receipt_pairs_attempt_with_result"])
        and (not c["receipt_names_the_job"])
        # Property 11 on the pre-fix loop: it writes no receipt (so nothing
        # names the spec) and calls the backends directly rather than through
        # `submit()` (so no job id is ever recorded for the drain). Both
        # halves must read as failures, or the assertion is not measuring
        # the mechanism.
        and (not c["receipt_names_the_spec"])
        and (not c["recorder_recovers_job_ids"])
        and (not c["receipt_survives_all_backends_failing"])
    )
    # The pre-fix reattach meter answers for property 9. Only the window
    # property is required to fail: the pre-fix local window IS an upper bound
    # and the pre-fix fresh path was correct, so demanding those fail would be
    # demanding the control break properties it never broke.
    reattach_detected = not c["reattach_bills_kernel_window"]
    # The pre-2026-08-12 `charge()` answers for property 10, on both halves.
    concurrent_detected = (
        (not c["concurrent_charge_survives"])
        and (not c["concurrent_idempotency_reads_disk"])
    )
    return (experiment_ok and billing_ok and receipt_ok and reattach_ok
            and concurrent_ok
            and isolation_detected and billing_detected and receipt_detected
            and reattach_detected and concurrent_detected)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.12"], _experiment, _check, control_fn=_control, ledger=ledger)
