"""T0.22 — a constant borrowed from another spec's entry is current, or refused.

T0.14's scar was a measured constant PASTED into a second file, where it drifts
away from the measurement that produced it. XL.00 obeyed that lesson exactly and
read PS.01's `j0`/`alpha` live from the ledger at run time — gating on
`status == PASS` and nothing else.

**Live is not current.** PS.01 measures a WORLD (`playground.py`, `w0.py`,
`drives.py`). Change the world and PS.01's entry becomes a measurement of a
world that no longer exists, while everything scored in that world — XL.00, and
LC.03/LC.04's `life_gain` — keeps computing on its numbers and cannot tell.
Found by the overseer 2026-08-10 at RANK 2; the instance was benign and the
guard was missing.

The repair is `protocol.borrow_metrics`, which refuses on every reason
`staleness_of` gives rather than on a status, and returns the source's
`impl_sha` so the borrower's own record names the version it computed on. This
battery is what makes that falsifiable.

The control is the rule that failed, kept executable: `status == PASS` and
nothing else. It must hand over the numbers for all three stale fixtures while
the guard refuses them — if it also refuses them, the two rules are the same
rule and this test measures nothing (T0.08 property 5, T0.19, T0.20, T0.21).

Fixtures are in-memory `Result` rows planted in a Ledger pointed at a path that
does not exist. Nothing is written, no world is simulated, and the SOURCE is the
real PS.01 so the honest case hashes a file that actually exists — a fixture
whose "current" hash is invented could not tell a working comparison from a
broken one.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from ..protocol import (Ledger, Result, Status, borrow_metrics, impl_sha_of,
                        module_path_for, run_spec)
from ..registry import BY_ID

SPEC_ID = "T0.22"

# The borrow this spec was written about. Real spec, real file, real metric
# names — see the module docstring on why the source is not synthetic.
SOURCE = "PS.01"
KEYS = ("j0_ms", "alpha")
VALUES = {"j0_ms": 2.405, "alpha": 0.0293}      # PS.01's measured constants

# Where the ladder's tests live, for the class check in P9.
TESTS_DIR = Path(__file__).resolve().parent


def _entry(**over) -> Result:
    """A healthy PS.01 PASS row, before whatever the fixture breaks about it."""
    base = dict(spec_id=SOURCE, status=Status.PASS, metrics=dict(VALUES),
                seeds=[0], commit="1234567", ran_at="2026-08-10T05:29:00",
                impl_sha=impl_sha_of(module_path_for(SOURCE)))
    base.update(over)
    return Result(**base)


def _ledger_with(entry) -> Ledger:
    """A Ledger holding exactly this row and touching no file on disk."""
    led = Ledger(path=Path("/nonexistent/t0_22_never_written.json"))
    if entry is not None:
        led.results[SOURCE] = entry
    return led


def _legacy_borrow(led: Ledger, keys) -> tuple:
    """THE RULE THAT FAILED, verbatim: PASS, and nothing else.

    `xl_00_death_and_respawn._calibration` as it stood until 2026-08-10. Kept as
    executable code rather than described, so the difference between the two
    rules is measured on the same fixtures instead of asserted in prose.
    """
    entry = led.results.get(SOURCE)
    if entry is None or entry.status != Status.PASS:
        return False, {}
    out = {}
    for k in keys:
        v = entry.metrics.get(k)
        if v is None:
            return False, {}
        out[k] = float(v)
    return True, out


def _guard_borrow(led: Ledger, keys) -> tuple:
    """The rule under test."""
    b = borrow_metrics(SOURCE, keys, ledger=led)
    return b.ok, b.values


# ---------------------------------------------------------------- the DEPENDENCY
# path, added 2026-08-11. Borrowing a NUMBER from a stale row and RUNNING ON a
# stale row are the same question — "does this entry still describe the code that
# exists?" — asked by two different organs, and only the borrow path was guarded.
# `run._terminal_blockers` restated it as `status is Status.PASS`, so `LC.03` read
# as runnable off a `PS.01` entry that `borrow_metrics` would have VOIDed the
# moment it ran. Overseer 2026-08-10 RANK 2; LESSONS' *"retiring a rule is a
# two-sided job"* is the generalisation, and P12 is that lesson made executable.
DEP_SPEC_ID = "Z.99"          # not a real spec: this is a graph fixture, not a claim


def _dep_spec():
    """A stub spec that depends on the real `SOURCE`, so `module_path_for`
    resolves an implementation file that actually exists."""
    from ..protocol import Budget, Spec
    return Spec(DEP_SPEC_ID, 0, "dependency fixture", "h", "f", "n", "m",
                Budget.CPU_FAST, depends_on=[SOURCE])


def _legacy_unsatisfied(led: Ledger, spec) -> list:
    """`Ledger.blocked_by` as it stood until 2026-08-11: PASS, and nothing else."""
    return [d for d in spec.depends_on if led.status(d) is not Status.PASS]


def _dep_blocked(led: Ledger, rule_is_legacy: bool) -> bool:
    spec = _dep_spec()
    return bool(_legacy_unsatisfied(led, spec) if rule_is_legacy
                else led.unsatisfied(spec))


N_PROPERTIES = 12


def _probe(rule_is_legacy: bool) -> dict:
    failed: list[str] = []
    borrow = _legacy_borrow if rule_is_legacy else _guard_borrow

    # P1 — THE NULL. An empty ledger must yield nothing. A borrower that
    # produces constants for a spec that never ran is reading its own defaults.
    if borrow(_ledger_with(None), KEYS)[0]:
        failed.append("p1_empty_ledger_yields_nothing")

    # P2 — the honest case still works. A rule that refuses everything gets
    # P3-P7 right for free and is useless; this is the property that costs it.
    ok, vals = borrow(_ledger_with(_entry()), KEYS)
    if not ok or vals != VALUES:
        failed.append("p2_current_entry_is_borrowable")

    # P3 — KNOWN ANSWER, the case this spec exists for. PASS, but the
    # implementation hash has moved: the entry describes code that is gone.
    if borrow(_ledger_with(_entry(impl_sha="0" * 16)), KEYS)[0]:
        failed.append("p3_changed_source_is_refused")

    # P4 — DIRTY. The run executed HEAD plus uncommitted edits, so the code
    # behind these numbers exists in no commit and never will.
    if borrow(_ledger_with(_entry(commit="1234567+dirty")), KEYS)[0]:
        failed.append("p4_dirty_source_is_refused")

    # P5 — UNVERIFIABLE. 44 entries predate `impl_sha`; a number from one of
    # them cannot be SHOWN to describe today's code, which is the entire claim
    # a borrow needs. Refusing costs a re-run of the source and buys the claim.
    if borrow(_ledger_with(_entry(impl_sha=None)), KEYS)[0]:
        failed.append("p5_unverifiable_source_is_refused")

    # P6 — not PASS. The one thing the old rule did check, kept so a regression
    # in the new rule cannot hide behind the properties the old one failed.
    if borrow(_ledger_with(_entry(status=Status.FAIL)), KEYS)[0]:
        failed.append("p6_non_pass_source_is_refused")

    # P7 — a missing metric refuses rather than defaulting. The `Arm.cost`
    # lesson: a value that cannot represent "absent" will silently claim one.
    if borrow(_ledger_with(_entry(metrics={"j0_ms": 2.405})), KEYS)[0]:
        failed.append("p7_missing_metric_is_refused")

    # P8 — PROVENANCE, on BOTH paths. The source's impl_sha reaches the
    # borrower on success (so the record names the version it computed on) AND
    # on refusal with a reason (a provenance that only appears when the answer
    # is good cannot explain a VOID). Only the guard returns provenance at all;
    # the legacy rule scores this as a failure, which is honest — an organ that
    # cannot report the fact does not get credit for not misreporting it.
    if rule_is_legacy:
        failed.append("p8_provenance_travels_with_the_number")
    else:
        good = borrow_metrics(SOURCE, KEYS, ledger=_ledger_with(_entry()))
        bad = borrow_metrics(SOURCE, KEYS,
                             ledger=_ledger_with(_entry(impl_sha="0" * 16)))
        cur = impl_sha_of(module_path_for(SOURCE))
        if (good.provenance.get("borrowed_impl_sha") != cur
                or good.provenance.get("borrowed_from") != SOURCE
                or bad.provenance.get("borrowed_impl_sha") != "0" * 16
                or not bad.refusal):
            failed.append("p8_provenance_travels_with_the_number")

    # P9 — THE CLASS, not the instance. No test in the ladder may read another
    # spec's metrics off the ledger directly; that is what the guard is for, and
    # a guard nothing is required to use is a guard one edit away from being
    # bypassed. Matches a literal ledger lookup by a REAL spec id — T0.08's own
    # synthetic rows use "X.01", which is deliberately not a spec, so the check
    # separates borrowing from a test managing its own fixtures.
    ids = "|".join(re.escape(i) for i in BY_ID)
    direct = re.compile(r"results(?:\.get\(|\[)\s*[\"'](" + ids + r")[\"']")
    offenders = []
    for path in sorted(TESTS_DIR.glob("*.py")):
        for hit in direct.finditer(path.read_text()):
            offenders.append(f"{path.name}:{hit.group(1)}")
    if rule_is_legacy or offenders:
        failed.append("p9_no_test_reads_another_specs_metrics_directly")

    # P10 — KNOWN ANSWER on the DEPENDENCY path. A dependency that PASSED but
    # whose implementation has since moved (CHANGED) or was never committed
    # (DIRTY) does not satisfy anything, and a current one still does. This is
    # P2+P3+P4 asked of the other organ; the legacy rule accepts all three.
    if (not _dep_blocked(_ledger_with(_entry(impl_sha="0" * 16)), rule_is_legacy)
            or not _dep_blocked(_ledger_with(_entry(commit="1234567+dirty")),
                                rule_is_legacy)
            or _dep_blocked(_ledger_with(_entry()), rule_is_legacy)
            or not _dep_blocked(_ledger_with(_entry(status=Status.FAIL)),
                                rule_is_legacy)
            or not _dep_blocked(_ledger_with(None), rule_is_legacy)):
        failed.append("p10_stale_dependency_does_not_satisfy")

    # P11 — the DELIBERATE divergence from P5, pinned so it cannot drift in
    # either direction unnoticed. UNVERIFIABLE (predates `impl_sha`) REFUSES a
    # borrow and PERMITS a dependency, because the two organs need different
    # claims: a borrowed number must describe today's code, while a dependency
    # only has to have been demonstrated. The evidence is absent, not contrary.
    # It was measured, not argued — refusing it on the dependency path takes the
    # ladder from 29 runnable specs to 7 on the strength of 40 silent rows.
    unver = _ledger_with(_entry(impl_sha=None))
    if borrow(unver, KEYS)[0] or _dep_blocked(unver, rule_is_legacy):
        failed.append("p11_unverifiable_refuses_a_borrow_but_permits_a_dependency")

    # P12 — THE CLASS. The blocker graph and the dependency rule must give the
    # same answer about the same row. This is the defect itself: two organs,
    # each internally consistent, disagreeing about whether one entry is usable
    # — visible only from outside, which is why it survived the unification that
    # was supposed to close it. A future organ that re-derives the rule instead
    # of calling it fails here rather than in an audit.
    from ..run import _terminal_blockers
    led = _ledger_with(_entry(impl_sha="0" * 16))
    dep, src = _dep_spec(), BY_ID[SOURCE]
    graph = _terminal_blockers(led, ladder=[src, dep],
                               by_id={SOURCE: src, DEP_SPEC_ID: dep})
    if bool(graph.get(DEP_SPEC_ID)) != _dep_blocked(led, rule_is_legacy):
        failed.append("p12_graph_and_rule_agree_on_the_same_row")

    return {
        "properties_checked": float(N_PROPERTIES),
        "properties_failed": float(len(failed)),
        "failed_names": ",".join(failed),
        "direct_ledger_reads": float(len(offenders)),
        "offenders": ",".join(offenders),
        "source_impl_sha": str(impl_sha_of(module_path_for(SOURCE))),
    }


def _experiment(seed: int) -> dict:
    return _probe(rule_is_legacy=False)


def _control(seed: int) -> dict:
    """`status == PASS` and nothing else — the rule XL.00 carried until today.

    It must break P3, P4 and P5: it hands over the numbers of a source whose
    code has changed, whose run was never committed, and whose provenance
    cannot be checked at all. Those three are the whole difference between "the
    source succeeded" and "the source still describes this world".

    It must ALSO break P10 and P12, which is the point of adding them: the same
    rule was still live on the dependency path a day after it was retired on the
    borrow path, and P12 fails for the legacy rule precisely because the graph
    had already been fixed while the rule had not — a disagreement, not a
    symmetric error.
    """
    return _probe(rule_is_legacy=True)


def _check(m: dict, c: dict) -> Status | bool:
    # All nine ran AND all nine held. Gating on `properties_failed == 0` alone
    # lets a battery that stopped early read as clean (T0.13's first bug; T0.19,
    # T0.20 and T0.21 carry the same guard).
    experiment_clean = (m["properties_failed"] == 0.0
                        and m["properties_checked"] == N_PROPERTIES
                        and c["properties_checked"] == N_PROPERTIES)
    # The control must fail, and fail on THE properties that define the guard —
    # not on P8/P9, which it cannot see and is not credited for.
    control_names = set(str(c.get("failed_names", "")).split(","))
    control_broken = {"p3_changed_source_is_refused",
                      "p4_dirty_source_is_refused",
                      "p5_unverifiable_source_is_refused",
                      # the same failure on the other organ, and the
                      # disagreement between them that hid it
                      "p10_stale_dependency_does_not_satisfy",
                      "p12_graph_and_rule_agree_on_the_same_row"} <= control_names
    return bool(experiment_clean and control_broken)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID[SPEC_ID], _experiment, _check, control_fn=_control,
                    ledger=ledger)
