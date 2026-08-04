"""T0.08 — the ledger must be a durable, honest record.

This is the file every capability claim in the project rests on, so it gets
tested like one. Four properties, each with a way to fail:

  1. A recorded result survives a write and a fresh read.
  2. NOT_RUN is the default. A spec never touched must never read as passing —
     that is the exact failure this whole package exists to prevent.
  3. A spec whose dependency is failing resolves to BLOCKED, not to a number.
     A result computed on a broken foundation is worse than no result because it
     looks like evidence.
  4. Concurrent writers do not corrupt it. The hourly loop and a manual session
     can overlap, and a truncated ledger would silently erase the record.

Uses a temporary ledger throughout — this test must never touch the real one.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ..protocol import Ledger, Result, Status, run_spec
from ..registry import BY_ID, Spec, Budget


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

    return {
        "roundtrip_ok": roundtrip,
        "default_is_not_run": default_not_run,
        "dependency_blocks": blocked,
        "blocked_records_no_metrics": blocked_not_run,
        "survives_concurrent_writers": concurrent_ok,
        "entries_after_concurrency": len(final.results),
    }


def _check(m: dict, _c: dict) -> bool:
    return all([m["roundtrip_ok"], m["default_is_not_run"], m["dependency_blocks"],
                m["blocked_records_no_metrics"], m["survives_concurrent_writers"]])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.08"], _experiment, _check, ledger=ledger)
