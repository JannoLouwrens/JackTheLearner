"""Re-judge the record without re-running anything.

Every integrity check this project had runs FORWARD from the code: T0.13
perturbs a gate and asks whether it still bites, `run stale` asks whether the
test file still hashes to what produced its entry, `impl_sha` pins the code a
claim was made about. All of them ask *"did the code change?"*.

None of them re-derives the VERDICT. And a verdict is cheap to re-derive,
because the two halves are both already on disk: the ledger stores each entry's
`metrics` and `control_metrics`, and the repo stores each spec's `_check`. Feed
one back through the other and the decision is re-taken for free — no
experiment, no GPU, seconds on CPU.

Three probes, and they fail differently.

  A. RE-VERDICT. For every PASS, evaluate the *committed* `_check` against the
     *recorded* metrics. Disagreement means the entry and the code no longer
     say the same thing. This overlaps T0.13's `stale_gates` deliberately and
     is the WEAKER half — and weaker in a specific direction worth stating,
     because the audit that asked for it got this backwards. Re-verdicting does
     NOT catch a gate that was LOOSENED after the run it certified: loosen a
     threshold and the recorded numbers clear it more easily, the replay
     returns True, and the entry reads clean. It catches the opposite drift —
     a gate tightened, or an entry that stopped satisfying its own code.
     Loosening is caught by `impl_sha` (`run stale`), which sees the file
     change. The two are complements; neither is a substitute, and 48 of 58
     entries predate `impl_sha`, which is why the free backward check earns its
     place anyway.

  B. CONTROL BLINDNESS — the one nothing else could see. Re-evaluate each
     PASSing gate with `control_metrics = {}` and demand the answer changes. A
     spec can declare a control, run it, record its numbers, and still never
     read them in the gate: law 2 ("a control that also passes means the test
     measures nothing") is unenforceable if the control's result never reaches
     the threshold at all. Grepping for `_control` cannot see this.
     `control_metrics` being non-empty cannot see it. T0.13 cannot see it
     either — it perturbs the keys a check REFERENCES, so a gate that
     references no control key has zero inert keys and reads perfectly clean.
     Deleting the input and demanding the verdict move is the only probe that
     distinguishes "the control is read" from "the control was merely run".

  C. DECLARATION COHERENCE. `Spec.control` is the field an auditor greps, and
     it is currently unreliable in one direction: 20 entries record
     `control_metrics` while their spec declares `control=None`. That is a
     false NEGATIVE — the science is fine (probe B proves each of those gates
     reads its control) and the audit surface is not. The opposite direction is
     the dangerous one: a spec that DECLARES a control and has no
     `control_metrics` has claimed a safeguard it never ran, and that is gated
     at zero.

     The undeclared count is gated as a RATCHET rather than at zero, because
     it is a real 20-entry debt and pretending otherwise means either a
     permanently red ladder or a number nobody watches. The overseer recorded
     it at 19, then at 20 — it grows. A ratchet is the honest form: the debt
     stays visible, and it may only ever be paid down.

Everything that could not be inspected is COUNTED, never skipped. A gate whose
module will not import, that exposes no `_check`, or that raises when replayed
is a gate this scan did not audit, and an unaudited item that leaves the
numerator alone is how a clean scan and a scan that never ran become the same
number (T0.13 shipped that bug once already).
"""
from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .protocol import Ledger, Status

#: Marker for the one entry a scan may legitimately not judge: its own.
SELF_EXCLUDED = "self-excluded"

#: The number of PASS entries that record `control_metrics` while their spec
#: declares `control=None`. A DEBT, measured 2026-08-10, and a ratchet: this
#: constant may be LOWERED as declarations are backfilled and must never be
#: raised. Raising it would convert a guard against the audit surface rotting
#: further into a rubber stamp — the overseer watched this number go 19 -> 20
#: across two audits with nothing to stop it.
UNDECLARED_CONTROL_BUDGET = 19


@dataclass
class Entry:
    """One judged claim: the gate, what it was fed, and what its spec promised.

    Passed in rather than looked up, so the known-answer fixture exercises the
    same `scan` the real ledger does. A detector with a tidied re-statement for
    a test is a detector whose test proves nothing about the shipped path.
    """
    spec_id: str
    check: Optional[Callable]
    metrics: Dict[str, Any]
    control_metrics: Dict[str, Any]
    declared_control: Optional[str]
    unavailable: str = ""      # non-empty = this entry could not be audited


def _verdict(fn: Callable, m: dict, c: dict):
    """Deep-copied every call: some checks WRITE to their metrics (T2.02 sets
    m["verdict"]), and a probe that let that leak would score the second
    evaluation against a mutated first."""
    try:
        out = fn(copy.deepcopy(m), copy.deepcopy(c))
    except Exception as e:
        return ("RAISED", type(e).__name__)
    if isinstance(out, Status):
        return ("STATUS", out.value)
    return ("BOOL", bool(out))


def _is_pass(v) -> bool:
    return v in (("BOOL", True), ("STATUS", Status.PASS.value))


def scan(entries: List[Entry]) -> dict:
    """Run all three probes over a list of entries. Pure; no I/O."""
    disagree: List[str] = []
    unevaluable: List[str] = []
    unavailable: List[str] = []
    self_excluded: List[str] = []
    blind: List[str] = []
    control_read_false: List[str] = []
    control_read_raise: List[str] = []
    declared_never_ran: List[str] = []
    undeclared_ran: List[str] = []
    no_control: List[str] = []
    judged = 0

    for e in entries:
        # Self-exclusion is a KNOWN, one-entry hole and is counted apart from
        # the failure bucket. Folding it into `unavailable` would make the gate
        # fail on its own second run — the first run has no entry to exclude,
        # every run after it does. Reported so the hole stays visible.
        if e.unavailable == SELF_EXCLUDED:
            self_excluded.append(e.spec_id)
            continue
        if e.unavailable or e.check is None:
            unavailable.append(f"{e.spec_id}({e.unavailable or 'no _check'})")
            continue

        # ── C. declaration coherence (independent of whether the gate runs) ──
        has_control_metrics = bool(e.control_metrics)
        if e.declared_control and not has_control_metrics:
            declared_never_ran.append(e.spec_id)
        if not e.declared_control and has_control_metrics:
            undeclared_ran.append(e.spec_id)

        # ── A. re-verdict ────────────────────────────────────────────────────
        base = _verdict(e.check, e.metrics, e.control_metrics)
        if base[0] == "RAISED":
            # Counted, not swallowed: a gate that cannot be replayed is a
            # verdict this scan did not re-derive.
            unevaluable.append(f"{e.spec_id}({base[1]})")
            continue
        judged += 1
        if not _is_pass(base):
            disagree.append(f"{e.spec_id}({base[0]}:{base[1]})")

        # ── B. control blindness ─────────────────────────────────────────────
        # Only meaningful where a control actually ran. Where it did, deleting
        # it must change the answer; if the verdict is unmoved, the gate never
        # consulted the control and law 2 is not enforced for that spec.
        if has_control_metrics:
            without = _verdict(e.check, e.metrics, {})
            if without == base and _is_pass(base):
                blind.append(e.spec_id)
            elif without[0] == "RAISED":
                control_read_raise.append(e.spec_id)
            else:
                control_read_false.append(e.spec_id)
        else:
            # No control ran at all, so probe B has nothing to say about this
            # spec. Named rather than absorbed into the denominator: the
            # overseer's §1.2 finding is exactly this set, and a claim resting
            # on a gate that was never shown capable of reporting the bad case
            # should be countable from the record.
            no_control.append(e.spec_id)

    return {
        "entries_seen": len(entries),
        "verdicts_rejudged": judged,
        "verdict_disagreements": len(disagree),
        "unevaluable_gates": len(unevaluable),
        "unavailable_entries": len(unavailable),
        "self_excluded_entries": len(self_excluded),
        "controls_probed": len(blind) + len(control_read_false) + len(control_read_raise),
        "control_blind_specs": len(blind),
        "control_read_by_value": len(control_read_false),
        "control_read_by_key": len(control_read_raise),
        "declared_control_never_ran": len(declared_never_ran),
        "undeclared_control_ran": len(undeclared_ran),
        "no_control_specs": len(no_control),
        "disagreement_detail": ", ".join(sorted(disagree)),
        "unevaluable_detail": ", ".join(sorted(unevaluable)),
        "unavailable_detail": ", ".join(sorted(unavailable)),
        "self_excluded_detail": ", ".join(sorted(self_excluded)),
        "control_blind_detail": ", ".join(sorted(blind)),
        "declared_never_ran_detail": ", ".join(sorted(declared_never_ran)),
        "undeclared_ran_detail": ", ".join(sorted(undeclared_ran)),
        "no_control_detail": ", ".join(sorted(no_control)),
    }


def collect(ledger: Ledger, exclude: tuple = ()) -> List[Entry]:
    """Build the entry list from the live ledger and the live registry.

    `exclude` exists for one reason and it is the same one T0.13 states: a
    spec cannot re-judge its OWN entry, because that entry is written after
    the scan and therefore always reflects the previous version of the file.
    Excluded ids are still returned, marked unavailable, so they are counted
    rather than vanishing from the denominator.
    """
    from .registry import BY_ID
    from .run import _module_for

    out: List[Entry] = []
    for spec_id, r in sorted(ledger.results.items()):
        if r.status is not Status.PASS:
            continue
        spec = BY_ID.get(spec_id)
        declared = spec.control if spec else None
        metrics = dict(r.metrics or {})
        controls = dict(r.control_metrics or {})
        if spec_id in exclude:
            out.append(Entry(spec_id, None, metrics, controls, declared,
                             unavailable=SELF_EXCLUDED))
            continue
        if not metrics and not controls:
            out.append(Entry(spec_id, None, metrics, controls, declared,
                             unavailable="no recorded metrics"))
            continue
        try:
            mod = _module_for(spec_id)
        except Exception as exc:
            out.append(Entry(spec_id, None, metrics, controls, declared,
                             unavailable=f"import {type(exc).__name__}"))
            continue
        fn = getattr(mod, "_check", None) if mod else None
        if fn is None:
            out.append(Entry(spec_id, None, metrics, controls, declared,
                             unavailable="no module" if mod is None else "no _check"))
            continue
        out.append(Entry(spec_id, fn, metrics, controls, declared))
    return out


# ── the known-answer fixture ────────────────────────────────────────────────
# Three planted entries the scan MUST separate. Without this, "nothing is
# wrong" and "the scan did not run" are the same output — the failure this
# repo has already shipped once, in T0.13's first attempt.

def _fixture_healthy(m, c):
    return m["score"] > 0.5 and not c["score"] > 0.5


def _fixture_disagreeing(m, c):
    """A gate that the recorded metrics no longer clear — the stored PASS and
    the committed code have stopped saying the same thing.

    Named for what probe A can actually SEE. The overseer's ask was phrased as
    catching "a check LOOSENED after the run it certified", and re-verdicting
    does not catch that: loosen a gate and the old numbers clear it more
    easily, so the replay returns True and the entry reads clean. Loosening is
    caught by `impl_sha` / `run stale`, which notice the file changed. Probe A
    catches the opposite drift, and the two are complements, not substitutes.
    """
    return m["score"] > 0.99 and not c["score"] > 0.5


def _fixture_control_blind(m, c):
    """Declares and runs a control, records its numbers, and never reads them.
    Structurally indistinguishable from `_fixture_healthy` to every check this
    project had before probe B."""
    return m["score"] > 0.5


def fixture() -> List[Entry]:
    good = {"score": 0.8}
    ctrl = {"score": 0.1}
    return [
        Entry("FIX.healthy", _fixture_healthy, dict(good), dict(ctrl), "a control"),
        Entry("FIX.disagree", _fixture_disagreeing, dict(good), dict(ctrl), "a control"),
        Entry("FIX.blind", _fixture_control_blind, dict(good), dict(ctrl), "a control"),
        # A spec promising a safeguard it never ran.
        Entry("FIX.promised", _fixture_control_blind, dict(good), {}, "a control"),
        # The 19-entry debt shape: control ran, spec declares none.
        Entry("FIX.undeclared", _fixture_healthy, dict(good), dict(ctrl), None),
    ]


def assert_detector_works() -> dict:
    """Run the fixture and refuse to let a clean scan be reported unless the
    known-bad entries were caught and the known-good one was spared.

    Same discipline as `run._check_stale_detector` and `run._check_ranker`, and
    for the same reason: this repo has already shipped an audit tool (T0.13)
    that returned a clean bill of health on a known-bad input because its input
    extraction silently produced nothing. "Nothing is wrong" and "I did not
    look" are the same output unless something makes them differ.
    """
    r = scan(fixture())
    bad = []
    if r["verdict_disagreements"] != 1:
        bad.append(f"re-verdict probe found {r['verdict_disagreements']} of 1")
    if r["control_blind_specs"] != 1:
        bad.append(f"control-blindness probe found {r['control_blind_specs']} of 1")
    if r["declared_control_never_ran"] != 1:
        bad.append(f"declared-never-ran probe found {r['declared_control_never_ran']} of 1")
    if r["undeclared_control_ran"] != 1:
        bad.append(f"undeclared probe found {r['undeclared_control_ran']} of 1")
    for key in ("control_blind_detail", "disagreement_detail",
                "declared_never_ran_detail", "undeclared_ran_detail"):
        if "FIX.healthy" in r[key]:
            bad.append(f"the healthy fixture was flagged in {key}")
    if bad:
        raise RuntimeError(
            "the record verifier failed its own known-answer fixture (" +
            "; ".join(bad) + "); refusing to report a scan it may not have performed")
    return r
