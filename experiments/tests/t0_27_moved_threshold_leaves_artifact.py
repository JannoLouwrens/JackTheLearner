"""T0.27 — a threshold moved after a FAIL leaves an artifact, not a paragraph.

The overseer's 13th-audit RANK 1, 2026-08-13: T2.08's auxiliary floor moved
0.70 -> 0.50 after the run it failed. The move was honest and disclosed in
four places — and every one of those places was written by the agent that
moved the floor. The FAIL was stamped `75a1938+dirty`, so v1's code exists in
no commit; the failing 0.6975 survived only in prose. The repo's answer to
"was the move honest?" was "read the commit message of the commit that made
the move". Disclosure is a property of the agent. This spec makes it a
property of the ledger.

B1 (T0.17 property 7) gave history the evidence slots. This is B2, in two
mechanisms, both in `experiments/protocol.py`:

  * `Ledger.record` pairs every verdict that supersedes a FAIL with the
    failing evidence, IN the record: `supersedes_fail` = {commit, dirty,
    impl_sha, impl_changed, metrics, ran_at}. The moved constants themselves
    are recovered by `git diff <fail commit> <pass commit> -- <test file>` —
    which is exactly why the fail commit must be real and clean.
  * `audit_supersedes_fail` makes "commit the failing implementation before
    re-running" executable: in any record whose CURRENT status is PASS, a
    FAIL whose impl_sha differs from the run that amended it must be stamped
    at a clean commit that exists in this repo and must carry its metrics.

Nine properties, each with a way to fail:

  1. A PASS superseding a FAIL carries `supersedes_fail`, with the failing
     commit, measurement and `impl_changed: True` when the code moved.
  2. Same-code re-run after FAIL: artifact present, `impl_changed: False` —
     a flake re-run is visible but is not an amendment, and the auditor
     ignores it.
  3. The artifact survives into history when the PASS is itself superseded
     (same rule as `amended`: the pairing is part of what that verdict was).
  4. Either side missing impl_sha => `impl_changed: None` — unknowable never
     reads as false — and the auditor counts the pair unauditable, zero
     violations (B1's no-back-fill rule, executably).
  5. The auditor flags a FAIL commit that does not exist in this repository.
  6. The auditor admits the clean shape: FAIL at a real commit, carrying
     metrics, then an amending PASS — zero violations.
  7. The auditor flags a committed FAIL whose history entry lost its metrics.
  8. Recording a FAIL from a modified tree warns loudly at record time (the
     cheapest possible intervention point — before the unauditable amend can
     exist); a clean-tree FAIL records silently.
  9. THE LIVE LEDGER passes its own audit, zero violations. This property
     reads `experiments/ledger.json`, not a fixture (B3's lesson: a guard
     that only ever sees fixtures guards nothing). The next amend-after-FAIL
     done from an uncommitted tree fails the gate re-run HERE. That is the
     spec working, not flaking.

CONTROL — the T2.08 shape replayed verbatim on a fixture: FAIL stamped
`75a1938+dirty` carrying its 0.6975, impl changed, PASS recorded on top. The
auditor MUST flag it. An auditor that certifies the very case that motivated
it measures nothing.

Uses temporary ledgers throughout except property 9, which reads (never
writes) the real one.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

from ..protocol import (LEDGER_PATH, Ledger, Result, Status,
                        audit_supersedes_fail, run_spec)
from ..registry import BY_ID

IMPL_DEPS = ["experiments/protocol.py"]

_ROOT = Path(__file__).parent.parent.parent


def _fresh(td: str) -> tuple[Path, Ledger]:
    path = Path(td) / "ledger.json"
    return path, Ledger(path)


def _row(path: Path, sid: str) -> dict:
    return json.loads(path.read_text())["results"][sid]


def _real_commit() -> str:
    """A short sha that provably exists in this repository."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=_ROOT,
                          timeout=10).stdout.strip()


def _experiment(seed: int) -> dict:
    real = _real_commit()
    with tempfile.TemporaryDirectory() as td:
        path, led = _fresh(td)

        # 1. FAIL (clean commit, metrics, sha A) -> PASS (sha B): artifact.
        led.record(Result(spec_id="X.1", status=Status.FAIL, commit=real,
                          metrics={"state_coverage": 0.6975},
                          impl_sha="a" * 16, ran_at="2026-01-01T00:00:00"))
        Ledger(path).record(Result(spec_id="X.1", status=Status.PASS,
                                   commit=real, metrics={"state_coverage": 0.71},
                                   impl_sha="b" * 16,
                                   ran_at="2026-01-02T00:00:00"))
        art = _row(path, "X.1").get("supersedes_fail") or {}
        artifact_written = (art.get("commit") == real
                           and art.get("dirty") is False
                           and art.get("impl_changed") is True
                           and art.get("impl_sha") == "a" * 16
                           and art.get("metrics") == {"state_coverage": 0.6975}
                           and art.get("ran_at") == "2026-01-01T00:00:00")

        # 2. Same code re-run after FAIL: visible, not an amendment.
        led2 = Ledger(path)
        led2.record(Result(spec_id="X.2", status=Status.FAIL, commit=real,
                           metrics={"m": 1.0}, impl_sha="c" * 16,
                           ran_at="2026-01-01T00:00:00"))
        Ledger(path).record(Result(spec_id="X.2", status=Status.PASS,
                                   commit=real, metrics={"m": 2.0},
                                   impl_sha="c" * 16,
                                   ran_at="2026-01-02T00:00:00"))
        art2 = _row(path, "X.2").get("supersedes_fail") or {}
        rerun_visible_not_amendment = (art2.get("impl_changed") is False)

        # 3. The pairing survives when the PASS is itself superseded.
        Ledger(path).record(Result(spec_id="X.1", status=Status.PASS,
                                   commit=real, metrics={"state_coverage": 0.72},
                                   impl_sha="b" * 16,
                                   ran_at="2026-01-03T00:00:00"))
        h = _row(path, "X.1")["history"]
        survives_history = (len(h) == 2
                            and (h[1].get("supersedes_fail") or {})
                                .get("metrics") == {"state_coverage": 0.6975})

        # 4. Missing impl_sha on the FAIL: None, unauditable, no violation.
        led3 = Ledger(path)
        led3.record(Result(spec_id="X.3", status=Status.FAIL, commit=real,
                           metrics={"m": 1.0}, impl_sha=None,
                           ran_at="2026-01-01T00:00:00"))
        Ledger(path).record(Result(spec_id="X.3", status=Status.PASS,
                                   commit=real, metrics={"m": 2.0},
                                   impl_sha="d" * 16,
                                   ran_at="2026-01-02T00:00:00"))
        art3 = _row(path, "X.3").get("supersedes_fail") or {}
        a3 = audit_supersedes_fail({"X.3": _row(path, "X.3")})
        unknowable_is_none = (art3.get("impl_changed") is None
                              and a3["unauditable_pairs"] == 1
                              and not a3["violations"])

        # 5. A FAIL commit that exists nowhere is flagged (git check live).
        led4 = Ledger(path)
        led4.record(Result(spec_id="X.4", status=Status.FAIL, commit="0000000",
                           metrics={"m": 1.0}, impl_sha="e" * 16,
                           ran_at="2026-01-01T00:00:00"))
        Ledger(path).record(Result(spec_id="X.4", status=Status.PASS,
                                   commit=real, metrics={"m": 2.0},
                                   impl_sha="f" * 16,
                                   ran_at="2026-01-02T00:00:00"))
        a4 = audit_supersedes_fail({"X.4": _row(path, "X.4")}, repo_root=_ROOT)
        flags_unreachable = (len(a4["violations"]) == 1 and any(
            "does not exist" in r for r in a4["violations"][0]["reasons"]))

        # 6. The clean shape is admitted, with the same git check live.
        a1 = audit_supersedes_fail({"X.1": _row(path, "X.1")}, repo_root=_ROOT)
        admits_clean = (not a1["violations"] and a1["checked_pairs"] == 1)

        # 7. A committed FAIL with no metrics is flagged.
        led5 = Ledger(path)
        led5.record(Result(spec_id="X.5", status=Status.FAIL, commit=real,
                           impl_sha="1" * 16, ran_at="2026-01-01T00:00:00"))
        Ledger(path).record(Result(spec_id="X.5", status=Status.PASS,
                                   commit=real, metrics={"m": 2.0},
                                   impl_sha="2" * 16,
                                   ran_at="2026-01-02T00:00:00"))
        a5 = audit_supersedes_fail({"X.5": _row(path, "X.5")}, repo_root=_ROOT)
        flags_missing_metrics = (len(a5["violations"]) == 1 and any(
            "no metrics" in r for r in a5["violations"][0]["reasons"]))

        # 8. A dirty-tree FAIL warns at record time; a clean one is silent.
        buf_dirty, buf_clean = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_dirty):
            Ledger(path).record(Result(spec_id="X.6", status=Status.FAIL,
                                       commit="abc1234+dirty",
                                       metrics={"m": 1.0}, impl_sha="3" * 16,
                                       ran_at="2026-01-01T00:00:00"))
        with contextlib.redirect_stdout(buf_clean):
            Ledger(path).record(Result(spec_id="X.7", status=Status.FAIL,
                                       commit=real, metrics={"m": 1.0},
                                       impl_sha="4" * 16,
                                       ran_at="2026-01-01T00:00:00"))
        warns_on_dirty_fail = ("unauditable" in buf_dirty.getvalue()
                               and "unauditable" not in buf_clean.getvalue())

    # 9. The live ledger passes its own audit. Read-only, real file.
    live = audit_supersedes_fail(
        json.loads(LEDGER_PATH.read_text())["results"], repo_root=_ROOT)

    props = {
        "artifact_written": artifact_written,
        "rerun_visible_not_amendment": rerun_visible_not_amendment,
        "artifact_survives_history": survives_history,
        "unknowable_is_none_and_unauditable": unknowable_is_none,
        "audit_flags_unreachable_commit": flags_unreachable,
        "audit_admits_clean_shape": admits_clean,
        "audit_flags_missing_metrics": flags_missing_metrics,
        "warns_on_dirty_fail": warns_on_dirty_fail,
        "live_ledger_clean": not live["violations"],
    }
    return {**{k: bool(v) for k, v in props.items()},
            "properties_failed": sum(1 for v in props.values() if not v),
            "live_checked_pairs": live["checked_pairs"],
            "live_unauditable_pairs": live["unauditable_pairs"],
            "live_violations": len(live["violations"])}


def _control(seed: int) -> dict:
    """T2.08's shape, replayed verbatim. The auditor MUST flag it."""
    with tempfile.TemporaryDirectory() as td:
        path, led = _fresh(td)
        led.record(Result(spec_id="T2.08x", status=Status.FAIL,
                          commit="75a1938+dirty",
                          metrics={"state_coverage": 0.6975},
                          impl_sha="5" * 16, ran_at="2026-08-13T02:34:06"))
        Ledger(path).record(Result(spec_id="T2.08x", status=Status.PASS,
                                   commit="1454525",
                                   metrics={"state_coverage": 0.6975},
                                   impl_sha="6" * 16,
                                   ran_at="2026-08-13T02:40:42"))
        audit = audit_supersedes_fail({"T2.08x": _row(path, "T2.08x")})
        return {"violations": len(audit["violations"]),
                "dirty_named": float(any(
                    "never committed" in r
                    for v in audit["violations"] for r in v["reasons"]))}


def _check(m: dict, c: dict) -> bool:
    return (m["properties_failed"] == 0
            # the control must fail: the T2.08 shape is flagged, by name
            and c["violations"] >= 1 and c["dirty_named"] == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.27"], _experiment, _check,
                    control_fn=_control, ledger=ledger)


if __name__ == "__main__":
    r = run()
    print(r.status.value, json.dumps(r.metrics, indent=2, sort_keys=True))
