"""T0.18 — the record can be re-judged, and every control is actually read.

Every integrity check in this repo runs FORWARD from the code: `run stale`
compares a test file's hash to the entry it produced, `impl_sha` pins the code
a claim was made about, T0.13 perturbs a gate and asks whether it still bites.
All of them answer *"did the code change?"*. None re-derives the VERDICT.

It costs nothing to close, because both halves are already on disk: the ledger
stores each entry's `metrics` and `control_metrics`, the repo stores each
spec's `_check`. `experiments/verify.py` feeds one back through the other.

The probe that matters most is the second one. A spec can declare a control,
run it, record its numbers — and never read them in the gate. Law 2 ("a control
that also passes means the test measures nothing") is then unenforceable for
that spec, and NOTHING in the machine could see it: grepping for `_control`
finds the function, `control_metrics` is non-empty, and T0.13 perturbs only the
keys a gate REFERENCES, so a gate referencing no control key has zero inert
keys and reads perfectly clean. Deleting the control and demanding the verdict
move is the only probe that separates "the control is read" from "the control
was merely run".

Control: a planted five-entry record, scanned by the SAME `verify.scan` the
real ledger goes through — one healthy gate that must NOT be flagged, one whose
recorded metrics no longer clear it, one that ignores its control, one that
declares a control it never ran, one that records a control its spec does not
declare. The healthy entry is the load-bearing half: a detector that answers
"defect" to everything has measured nothing, exactly like a detector that
answers "clean" to everything. T0.13's first attempt came back clean on a
known-bad gate because its source extraction silently read nothing, so a
fixture that exercises a *tidied restatement* of the scan would prove nothing
here either.

This spec excludes its own ledger entry and says so in `self_excluded_entries`:
that entry is written AFTER the scan, so it always reflects the previous
version of this file. Its own gate is exercised by the control instead.

Probe C's debt was paid off on 2026-08-10 (19 undeclared -> 0), so this spec
also carries the guard that keeps it paid: `run_spec` now REFUSES a spec that
supplies a `control_fn` while declaring `Spec.control = None`. That guard is
itself a claim, so it is tested here in both directions on throwaway specs and a
throwaway ledger — a guard that refuses everything and a guard that refuses
nothing produce the same clean-looking log, which is the reaper lesson.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..protocol import (Budget, Ledger, Spec, Status, UndeclaredControl,
                        run_spec)
from ..registry import BY_ID
from ..verify import UNDECLARED_CONTROL_BUDGET, collect, fixture, scan

SPEC_ID = "T0.18"

#: A scan of almost nothing is not a clean scan. Both floors are well under
#: today's numbers (55 judged, 50 controls probed) and exist so that a future
#: ledger-loading or import regression that silently empties the population
#: lands as a red entry rather than as a green one.
MIN_VERDICTS = 30
MIN_CONTROLS = 20


def _guard_probe() -> dict:
    """Does `run_spec` refuse an UNDECLARED control, and only that?

    Two throwaway specs, identical but for the `control` declaration, both run
    against a throwaway ledger — `Ledger(path=...)` and the injection rule that
    made it possible (LESSONS.md: a function that hard-codes the path to the
    record it mutates cannot be tested without corrupting it).

    Both directions are asserted because they fail identically in a log: a
    guard that refuses every spec and a guard that refuses none both leave "no
    undeclared controls" behind them. The permissive arm must reach a real
    verdict, not merely not-raise, so a future guard that swallows the run
    instead of refusing it is also caught.
    """
    def _fn(seed: int) -> dict:
        return {"x": 1.0}

    def _ctl(seed: int) -> dict:
        return {"x": 0.0}

    def _chk(m: dict, c: dict) -> bool:
        return m["x"] > c["x"]

    def _spec(control):
        return Spec("FIX.guard", 0, "throwaway",
                    hypothesis="x beats the control's x",
                    falsified_by="it does not",
                    null_baseline="the control", metric="x",
                    budget=Budget.CPU_FAST, control=control)

    out = {"refused_undeclared": 0.0, "ran_declared": 0.0}
    with tempfile.TemporaryDirectory(prefix="t018_guard_") as d:
        led = Ledger(path=Path(d) / "ledger.json")
        try:
            run_spec(_spec(None), _fn, _chk, control_fn=_ctl, ledger=led)
        except UndeclaredControl:
            out["refused_undeclared"] = 1.0
        # …and the same spec WITH a declaration must run through to a verdict.
        res = run_spec(_spec("the control's x must be lower"), _fn, _chk,
                       control_fn=_ctl, ledger=led)
        out["ran_declared"] = 1.0 if res.status is Status.PASS else 0.0
        # Nothing may have been written to the real ledger by either arm.
        out["guard_ledger_entries"] = float(len(Ledger(
            path=Path(d) / "ledger.json").results))
    return out


def _experiment(seed: int) -> dict:
    return {**scan(collect(Ledger(), exclude=(SPEC_ID,))), **_guard_probe()}


def _control(seed: int) -> dict:
    return scan(fixture())


def _check(m: dict, c: dict) -> bool:
    # ── the real record must be clean on all three probes ──────────────────
    record_clean = (
        m["verdict_disagreements"] == 0          # A: every PASS re-derives
        and m["control_blind_specs"] == 0        # B: every control is read
        and m["declared_control_never_ran"] == 0  # C: no promised-but-unrun control
        # C, the debt half. A ratchet, not zero — see verify.py. May be
        # lowered as declarations are backfilled; raising it would convert the
        # guard into a rubber stamp for the rot it was written to stop.
        and m["undeclared_control_ran"] <= UNDECLARED_CONTROL_BUDGET
        # Nothing skipped silently: an unaudited entry that leaves the
        # numerator alone is how a clean scan and a scan that never ran become
        # the same number.
        and m["unevaluable_gates"] == 0
        and m["unavailable_entries"] == 0
        # The one legitimate hole is this spec's own entry, and it is at most
        # one. Zero on the first run (no entry yet), one on every re-run.
        and m["self_excluded_entries"] <= 1
    )
    scanned_enough = (m["verdicts_rejudged"] >= MIN_VERDICTS
                      and m["controls_probed"] >= MIN_CONTROLS)

    # ── and the guard that keeps probe C at zero must bite, and only there ──
    guard_works = (m["refused_undeclared"] == 1.0
                   and m["ran_declared"] == 1.0
                   and m["guard_ledger_entries"] == 1.0)

    # ── and the scan must find exactly the planted defects, and no others ──
    control_caught = (
        c["entries_seen"] == 5
        and c["verdict_disagreements"] == 1
        and c["control_blind_specs"] == 1
        and c["declared_control_never_ran"] == 1
        and c["undeclared_control_ran"] == 1
        # Each flag must land on exactly the planted entry and nowhere else —
        # in particular never on FIX.healthy, because a detector that flags
        # everything is as useless as one that flags nothing.
        #
        # Written as equality rather than `"FIX.healthy" not in ...` on
        # purpose. The negative form is true under every perturbation T0.13
        # can apply to a string (it substitutes "" and a sentinel, neither of
        # which contains "FIX.healthy"), so it would have been four assertions
        # that could not fire — the exact defect T0.13 exists to catch, in the
        # spec written to strengthen the audit. Equality pins WHICH entry was
        # flagged and moves the verdict when the value moves.
        and c["control_blind_detail"] == "FIX.blind"
        and c["disagreement_detail"] == "FIX.disagree(BOOL:False)"
        and c["declared_never_ran_detail"] == "FIX.promised"
        and c["undeclared_ran_detail"] == "FIX.undeclared"
        and c["unevaluable_gates"] == 0
        and c["unavailable_entries"] == 0
    )
    return record_clean and scanned_enough and guard_works and control_caught


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
