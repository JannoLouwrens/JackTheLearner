"""Branch battery for `run_bakeoff`'s admission screen (property 5).

Replays the SO.10 failure shape through the REAL `run_bakeoff` to prove the
`admissible` predicate does what the 2026-09-29 paired ruling ordered — and
that WITHOUT it the old behaviour (an ineligible cheap arm winning the cost
tie-break) reproduces, which is the defect the seam removes, demonstrated
rather than asserted.

Writes nothing real: every case passes `decisions_path` to a temp file, per
the 2026-08-09 fixture scar recorded in `_append_decision`'s docstring.
Costs milliseconds. Run:

    /data/venvs/jackthelearner/bin/python -m experiments.tests.bakeoff_admission_battery

WHY A FILE AND NOT A PARAGRAPH: `t211_mi_battery.py` precedent — a seam
nobody has seen refuse is a seam on the author's word.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from ..bakeoff import Arm, run_bakeoff

# Deterministic per-seed scores: two eligible arms inside the 1.5-sigma
# margin of each other (a genuine tie), one INELIGIBLE arm that scores
# competitively and declares the lowest cost — SO.10's exact shape.
SCORES = {
    "eligible-a":   {0: 1.00, 1: 1.05, 2: 0.95},
    "eligible-b":   {0: 0.98, 1: 1.02, 2: 0.96},
    "ineligible-c": {0: 1.01, 1: 1.04, 2: 0.97},
}
NULL = {0: 0.00, 1: 0.02, 2: -0.02}
ELIGIBLE = ("eligible-a", "eligible-b")


def _spec() -> SimpleNamespace:
    return SimpleNamespace(id="ADM-BATTERY", seeds=3, metric="fixture",
                           gate_mode="validity", screen_rationale="")


def _arms(costs, scores) -> list:
    return [Arm(n, (lambda nm: lambda s: scores[nm][s])(n), cost=costs[n])
            for n in scores]


def _run(costs, predicate, scores=SCORES):
    with tempfile.TemporaryDirectory() as td:
        return run_bakeoff(_spec(), _arms(costs, scores), lambda s: NULL[s],
                           seeds=[0, 1, 2], admissible=predicate,
                           decisions_path=Path(td) / "d.md")


def show(name: str, got, expect) -> None:
    ok = got == expect
    print(f"{'ok  ' if ok else '!! MISFIRE'}  {name}: {got} (expected {expect})")
    assert ok, name


def main() -> None:
    costs = {"eligible-a": 1.0, "eligible-b": 2.0, "ineligible-c": 0.0}

    # 1. THE DEFECT, reproduced first: no predicate, and the ineligible arm
    #    takes the tie on cost — the exact mechanism that handed laplace-full
    #    the Person-model seat.
    r = _run(costs, None)
    show("no predicate: ineligible-c wins the cost tie-break",
         (r.verdict, r.winner), ("TIE", "ineligible-c"))

    # 2. THE REPAIR: same field, same costs, predicate supplied. The
    #    ineligible arm is scored and NOT RANKED; the tie resolves among
    #    eligible arms only, to the cheaper one.
    r = _run(costs, lambda a: a.name in ELIGIBLE)
    show("predicate: tie resolves among eligible arms only",
         (r.verdict, r.winner), ("TIE", "eligible-a"))
    byname = {a.name: a for a in r.arms}
    show("ineligible-c is still SCORED (mean recorded)",
         round(byname["ineligible-c"].mean, 4), round(sum(
             SCORES["ineligible-c"].values()) / 3, 4))
    show("ineligible-c is marked inadmissible",
         byname["ineligible-c"].admissible, False)

    # 3. NOT RANKED LAST: a field screened below two admissible arms VOIDs —
    #    it may not crown its least-inadmissible member.
    r = _run(costs, lambda a: a.name == "eligible-a")
    show("one admissible arm -> VOID, not a winner", r.verdict, "VOID")
    r = _run(costs, lambda a: False)
    show("zero admissible arms -> VOID", r.verdict, "VOID")

    # 4. MONOTONE / NO-OP: a predicate admitting everything changes nothing
    #    against the no-predicate baseline.
    show("admit-all predicate == no predicate",
         (_run(costs, lambda a: True).verdict,
          _run(costs, lambda a: True).winner),
         (_run(costs, None).verdict, _run(costs, None).winner))

    # 5. AN INADMISSIBLE ARM CANNOT VOID THE RUN THROUGH THE LEARNING GATE:
    #    outside the candidate set it arbitrates nothing. eligible-a beats
    #    eligible-b by a clear margin here, so the verdict is a WINNER even
    #    though the (inadmissible) gate-failing arm would have VOIDed the
    #    old validity path.
    spread = {"eligible-a": {0: 2.00, 1: 2.05, 2: 1.95},
              "eligible-b": {0: 1.00, 1: 1.02, 2: 0.98},
              "ineligible-c": {0: 0.01, 1: 0.00, 2: -0.01}}  # below the gate
    r = _run(costs, lambda a: a.name in ELIGIBLE, scores=spread)
    show("inadmissible gate-failer cannot VOID the run",
         (r.verdict, r.winner), ("WINNER", "eligible-a"))
    r = _run(costs, None, scores=spread)
    show("...but WITH no predicate it still VOIDs (gate unchanged)",
         r.verdict, "VOID")

    print("battery green: 9/9")


if __name__ == "__main__":
    main()
