"""T0.08 — the ledger must be a durable, honest record.

This is the file every capability claim in the project rests on, so it gets
tested like one. Properties, each with a way to fail:

  1. A recorded result survives a write and a fresh read.
  2. NOT_RUN is the default. A spec never touched must never read as passing —
     that is the exact failure this whole package exists to prevent.
  3. A spec whose dependency is failing resolves to BLOCKED, not to a number.
     A result computed on a broken foundation is worse than no result because it
     looks like evidence.
  4. Concurrent writers do not corrupt it. The hourly loop and a manual session
     can overlap, and a truncated ledger would silently erase the record.
  5. NEW 2026-08-10 — A STALE WRITER CANNOT REVERT WORK IT NEVER SAW. A Ledger
     constructed hours ago and written to now must change exactly the one entry
     it is recording: every other entry keeps its newest value, its attempt
     count and its amendments.

Property 5 exists because properties 1–4 all PASSED while the bug was live.
Property 4 asserted `len(final.results) >= 15` — a COUNT. Nothing was ever lost
by count; entries were REVERTED in place, which a count cannot see. That is the
"measure the quantity you are claiming, not a proxy" lesson landing on the
ledger's own spec. What actually happened: a `run T2.01` GPU poll built its
Ledger at 19:42 on 2026-08-09, waited 5.6 h for a Kaggle P100, and recorded at
01:17 on 2026-08-10. Its one write reverted LC.01, PG.3, PG.8, T0.08, T0.13 and
T0.15 to their 19:42 values and erased five `amended` records — and disguised
the revert as history, because the fresh on-disk verdict was pushed down into
`history` with `attempt` incremented, so it read as an honest re-run.

THE CONTROL is the pre-fix merge replayed verbatim (`_prefix_merge_record`) on
the same battery. It MUST fail property 5. Without it this spec would pass on a
ledger where the reversion is simply not exercised — and, per the overseer's
§1.2, T0.08 was one of five PASSes with no control at all.

Uses a temporary ledger throughout — this test must never touch the real one.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict

from ..protocol import Ledger, Result, Status, run_spec
from ..registry import BY_ID, Spec, Budget


def _safe_record(led: Ledger, result: Result) -> None:
    """The shipped write path."""
    led.record(result)


def _prefix_merge_record(led: Ledger, result: Result) -> None:
    """The merge as it stood BEFORE 2026-08-10, replayed verbatim.

    Kept as executable code rather than prose because a tidied restatement would
    pass while the shipped bug stayed live — the same reason T0.16 parses the
    real `JOB` string instead of a summary of it. The only edit is that it calls
    the Ledger's own lock/write helpers instead of re-implementing them; the
    defect being reproduced is the MERGE, not the I/O.
    """
    import fcntl

    led.results[result.spec_id] = result
    lock_path = led.path.with_suffix(led.path.suffix + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            on_disk: Dict[str, Any] = {}
            if led.path.exists():
                on_disk = json.loads(led.path.read_text()).get("results", {})

            merged = dict(on_disk)
            for rid, r in led.results.items():          # <-- THE BUG: all of them
                prev = on_disk.get(rid)
                hist = list(prev.get("history", [])) if prev else []
                if prev and prev.get("ran_at") != r.ran_at:
                    row_h = {k: prev.get(k) for k in
                             ("status", "ran_at", "commit", "message")}
                    if prev.get("amended"):
                        row_h["amended"] = prev["amended"]
                    hist.append(row_h)
                row = {**asdict(r), "status": r.status.value}
                row["history"] = hist[-20:]
                unknown = r.attempt is None or (prev is not None
                                                and prev.get("attempt", 1) is None)
                row["attempt"] = None if unknown else len(hist) + 1
                merged[rid] = row

            led._write_atomic(merged)
            for rid, raw in merged.items():
                if rid not in led.results:              # <-- and stays stale
                    d = dict(raw)
                    d["status"] = Status(d["status"])
                    led.results[rid] = Result(**d)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _stale_writer_battery(record_fn: Callable[[Ledger, Result], None]) -> dict:
    """Reproduce 2026-08-10 01:17 in miniature, with `record_fn` as the writer.

    The long-running job takes its snapshot, the world moves on in four
    independent ways, then the job records its own unrelated result. Nothing it
    did not record may have changed.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ledger.json"

        # --- the world as the long job saw it at 19:42 -------------------
        seed_led = Ledger(path)
        seed_led.record(Result(spec_id="S.01", status=Status.PASS,
                               metrics={"v": 1}, ran_at="2026-08-09T19:00:00"))
        seed_led.record(Result(spec_id="S.02", status=Status.PASS,
                               metrics={"v": 1}, ran_at="2026-08-09T19:01:00"))

        stale = Ledger(path)          # the GPU poll's snapshot. It waits 5.6 h.

        # --- what six intervening iterations did -------------------------
        Ledger(path).record(Result(spec_id="S.01", status=Status.PASS,
                                   metrics={"v": 2}, ran_at="2026-08-09T22:33:00"))
        Ledger(path).amend("S.02", by="T0.08 fixture",
                           reason="an amendment written after the snapshot",
                           status=Status.VOID)
        Ledger(path).record(Result(spec_id="S.03", status=Status.PASS,
                                   metrics={"v": 1}, ran_at="2026-08-10T00:12:00"))

        # --- the long job finishes and records ITS OWN result ------------
        record_fn(stale, Result(spec_id="S.99", status=Status.FAIL,
                                metrics={"sigma": 1.19},
                                ran_at="2026-08-10T01:17:15"))

        rows = json.loads(path.read_text())["results"]

    s01, s02 = rows.get("S.01", {}), rows.get("S.02", {})
    return {
        # the four things that were actually destroyed, one metric each
        "fresh_metric_survived": s01.get("metrics", {}).get("v") == 2,
        "attempt_not_inflated": s01.get("attempt") == 2,
        "amendment_survived": bool(s02.get("amended")) and s02.get("status") == "VOID",
        "unseen_entry_survived": rows.get("S.03", {}).get("status") == "PASS",
        # and the write the job was entitled to make must still land
        "own_result_recorded": rows.get("S.99", {}).get("status") == "FAIL",
        "entries_after_stale_write": len(rows),
    }


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ledger.json"

        # 1. round-trip
        led = Ledger(path)
        led.record(Result(spec_id="X.01", status=Status.PASS,
                          metrics={"value": 42, "ratio": 3.5}, seeds=[0, 1, 2]))
        reread = Ledger(path)
        r = reread.results.get("X.01")
        roundtrip = (r is not None and r.status is Status.PASS
                     and r.metrics.get("value") == 42 and r.seeds == [0, 1, 2])

        # 2. default is NOT_RUN — an untouched spec must never look passing
        default_not_run = reread.status("NEVER.TOUCHED") is Status.NOT_RUN

        # 3. a failing dependency blocks rather than producing a number
        reread.record(Result(spec_id="X.02", status=Status.FAIL))
        dependent = Spec("X.03", 0, "dependent", hypothesis="h", falsified_by="f",
                         null_baseline="n", metric="m", budget=Budget.CPU_FAST,
                         depends_on=["X.02"])
        blocked = reread.blocked_by(dependent) == ["X.02"]

        ran = run_spec(dependent, lambda s: {"m": 1},
                       lambda m, c: True, ledger=reread)
        blocked_not_run = ran.status is Status.BLOCKED and ran.metrics == {}

        # 4. survives interleaved writers
        a, b = Ledger(path), Ledger(path)
        for i in range(15):
            (a if i % 2 == 0 else b).record(
                Result(spec_id=f"C.{i:02d}", status=Status.PASS, metrics={"i": i}))
        final = Ledger(path)
        parsed = json.loads(path.read_text())
        concurrent_ok = (len(final.results) >= 15 and "results" in parsed)

    # 5. a stale writer changes exactly one key
    out = {
        "roundtrip_ok": roundtrip,
        "default_is_not_run": default_not_run,
        "dependency_blocks": blocked,
        "blocked_records_no_metrics": blocked_not_run,
        "survives_concurrent_writers": concurrent_ok,
        "entries_after_concurrency": len(final.results),
    }
    out.update(_stale_writer_battery(_safe_record))
    return out


def _control(seed: int) -> dict:
    """The pre-2026-08-10 merge on the same battery. It MUST revert."""
    return _stale_writer_battery(_prefix_merge_record)


#: The four stale-writer properties, named once so the gate and the control
#: read the same list and cannot drift apart.
_STALE_PROPS = ("fresh_metric_survived", "attempt_not_inflated",
                "amendment_survived", "unseen_entry_survived")


def _check(m: dict, c: dict) -> bool:
    shipped = all([m["roundtrip_ok"], m["default_is_not_run"],
                   m["dependency_blocks"], m["blocked_records_no_metrics"],
                   m["survives_concurrent_writers"],
                   m["own_result_recorded"],
                   all(m[k] for k in _STALE_PROPS)])
    # The control must be BLIND to the reversion. `.get(k, True)` means an empty
    # control_metrics reads as "the pre-fix code preserved everything", which is
    # false, so the gate fails — a control this gate does not read is a control
    # that is not there (T0.18 probe B).
    control_reverts = not all(c.get(k, True) for k in _STALE_PROPS)
    # ...but it must still have written its own result, or it failed for some
    # unrelated reason and localises nothing.
    control_ran = c.get("own_result_recorded", False) is True
    return shipped and control_reverts and control_ran


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.08"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
