"""T0.19 — `screen` eliminates arms; it does not lower the bar.

`Spec.gate_mode="screen"` was added on 2026-08-10 because the PS.01 impact
bakeoff was VOID by construction: its arms are OBSERVABLES, three of four could
not separate a fall from a collapse, that IS the finding, and the T2.02 validity
gate — written for LEARNERS, where a missed gate cannot be told from a broken
run — refused to crown anything, forever, since the sanctioned repair (add arms,
remove none) can only add more failures.

A new mode on the module that makes every decision in this project is exactly
the kind of thing that becomes a loophole: "my arm failed, so I'll screen it."
So it arrives with the property battery it must survive, and with the pre-mode
machinery kept as EXECUTABLE code to serve as the control (the T0.08 property-5
pattern — a tidied restatement of the bug would pass while the shipped path
stayed broken).

THE LOAD-BEARING PROPERTY is P2: `screen` does NOT change the verdict of the run
that motivated it. Round 1 of PS.01/J had exactly one finisher, and one finisher
is a race with one runner, which the module refuses at the door in BOTH modes.
If the mode had been reverse-engineered to rescue that run, P2 would fail — and
under the control, where MIN_FINISHERS is the pre-guard 1, it does fail.

Every arm here is a constant function of its seed. There is no physics and no
training: the properties are about the ARBITRATION, and a synthetic arm set is
the only way to hold the numbers fixed while the rule varies.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .. import bakeoff as bk
from ..bakeoff import Arm, run_bakeoff
from ..protocol import Budget, Ledger, Spec, Status, run_spec
from ..registry import BY_ID

SPEC_ID = "T0.19"

# Round 1 of PS.01/J, verbatim from docs/DECISIONS_RESOLVED.md (2026-08-10).
# Per-seed AUCs; exactly one arm clears 3 sigma over the null.
ROUND1 = {
    "integral6":  [0.480, 0.500, 0.580],
    "peak6":      [0.340, 0.420, 0.260],
    "peak_force": [0.350, 0.390, 0.270],
    "peak_dvel":  [0.830, 0.770, 0.880],
}
ROUND1_NULL = [0.4966, 0.4966, 0.4966]

# Round 2's two finishers plus one eliminated arm: a legitimate screen.
ROUND2_TWO = {
    "impact_speed": [1.000, 1.000, 0.920],
    "peak_dvel":    [0.830, 0.770, 0.880],
    "peak_force":   [0.350, 0.390, 0.270],
}


def _arms(table: dict, cost: float = 1.0):
    def mk(scores):
        return lambda s: scores[s]
    return [Arm(n, mk(v), cost=cost, description=n) for n, v in table.items()]


def _null(scores):
    return lambda s: scores[s]


def _spec(mode: str, rationale: str | None = "arms are observables") -> Spec:
    return Spec("BAKEOFF/FIXTURE", 0, "fixture", hypothesis="-", falsified_by="-",
                null_baseline="-", metric="auc", budget=Budget.CPU_FAST, seeds=3,
                gate_mode=mode, screen_rationale=rationale)


def _bake(table, mode, null=None, controls=None, rationale="arms are observables"):
    """Run the SHIPPED run_bakeoff into a throwaway decisions file.

    `decisions_path` exists precisely so a test can reach the code without
    reaching the record — on 2026-08-09 the whole of DECISIONS_RESOLVED.md was
    nine fixtures from a self-test."""
    with tempfile.TemporaryDirectory() as td:
        return run_bakeoff(_spec(mode, rationale), _arms(table),
                           _null(null or ROUND1_NULL), seeds=[0, 1, 2],
                           learning_gate_sigma=3.0, margin_sigma=1.5,
                           controls=controls,
                           decisions_path=Path(td) / "decisions.md")


def _probe() -> dict:
    """Seven properties of the arbitration, each with a known answer.

    A property that RAISES counts as failed, not as an error. The control
    genuinely explodes — `MIN_FINISHERS = 1` lets a one-arm field reach
    `ranked[1]` — and an exception escaping here would abort the battery and
    record ERROR, which says nothing about whether the guard works. "The
    property did not hold" is the right reading of a crash."""
    out: dict = {}
    fails: list[str] = []
    raised: list[str] = []

    def prop(name: str, fn):
        try:
            ok = bool(fn())
        except Exception as exc:                       # noqa: BLE001
            ok = False
            raised.append(f"{name}:{type(exc).__name__}")
        out[name] = float(ok)
        if not ok:
            fails.append(name)

    # P1 — a field screened down to ONE survivor is a race with one runner.
    def p1():
        r = _bake(ROUND1, "screen")
        return r.verdict == "VOID" and r.winner is None
    prop("p1_one_finisher_voids", p1)

    # P2 — THE LOAD-BEARING ONE. The mode does not rescue the run that
    # motivated it: round 1 is VOID under both readings, for different reasons.
    def p2():
        return (_bake(ROUND1, "screen").verdict == "VOID"
                and _bake(ROUND1, "validity").verdict == "VOID")
    prop("p2_round1_void_both_modes", p2)

    # P3 — the default reading is untouched: under `validity` a single missed
    # gate still VOIDs even when two arms clear it.
    prop("p3_validity_unchanged",
         lambda: _bake(ROUND2_TWO, "validity").verdict == "VOID")

    # P4 — with two finishers, screen arbitrates, and the winner is one of them.
    def p4():
        r = _bake(ROUND2_TWO, "screen")
        return r.verdict == "WINNER" and r.winner == "impact_speed"
    prop("p4_two_finishers_decide", p4)

    # P5 — the gate is not lowered: whatever screen crowns cleared 3 sigma, and
    # every eliminated arm is still REPORTED (losers stay in the record).
    def p5():
        r = _bake(ROUND2_TWO, "screen")
        won = [a for a in r.arms if a.name == r.winner]
        return (bool(won) and won[0].sigma_over_null >= 3.0
                and {a.name for a in r.arms} >= set(ROUND2_TWO))
    prop("p5_winner_cleared_gate", p5)

    # P6 — a control that CLEARS the gate still inverts the verdict to VOID.
    # Screen mode must not become a way to ignore an escaped control.
    prop("p6_escaped_control_still_voids",
         lambda: _bake(ROUND2_TWO, "screen",
                       controls=[Arm("escapee", _null([0.99, 0.99, 0.99]),
                                     cost=0.0)]).verdict == "VOID")

    # P7 — screen without a written rationale is refused, and an unknown mode
    # is refused. The mode is a pre-registration or it is nothing.
    def p7():
        refused = 0
        for mode, rat in (("screen", None), ("screen", "  "), ("wishful", "x")):
            try:
                _bake(ROUND2_TWO, mode, rationale=rat)
            except ValueError:
                refused += 1
        return refused == 3
    prop("p7_undeclared_mode_refused", p7)

    out["properties_checked"] = float(sum(1 for k in out if k.startswith("p")))
    out["properties_failed"] = float(len(fails))
    out["failed_names"] = ",".join(fails) if fails else "none"
    out["raised"] = ",".join(raised) if raised else "none"
    return out


def _experiment(seed: int) -> dict:
    return _probe()


def _control(seed: int) -> dict:
    """The pre-guard machinery, kept executable: MIN_FINISHERS = 1.

    That is the version of `screen` a hurried author would have written — crown
    the best survivor however few survived — and it is the one that WOULD have
    rescued round 1. Running the same battery against it must break P1 and P2.
    Anything less and a clean battery and a battery that never ran are the same
    output (T0.13 shipped that bug on its own first attempt)."""
    original = bk.MIN_FINISHERS
    try:
        bk.MIN_FINISHERS = 1
        return _probe()
    finally:
        bk.MIN_FINISHERS = original


N_PROPERTIES = 7


def _check(m: dict, c: dict) -> Status | bool:
    # All seven ran AND all seven held. Gating only on `properties_failed == 0`
    # would let a battery that silently stopped after two properties read as
    # clean — a skipped item that leaves the numerator alone is how "clean" and
    # "never ran" become the same number (T0.13's own first bug).
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail on the two properties that define the guard, and it
    # must fail on THOSE — a control that fails for some unrelated reason has
    # not shown the battery can see this bug.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = (c["properties_failed"] > 0.0
                      and {"p1_one_finisher_voids",
                           "p2_round1_void_both_modes"} <= control_names)
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
