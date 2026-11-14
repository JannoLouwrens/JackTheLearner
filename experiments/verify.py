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

     A refusal (`protocol.NoControlByDecision`, falsy) promises nothing;
     one that ran a control anyway is flagged. History: git, 2026-09-27.

  D. MODULE STATE. A `_check` that reads a list its own module fills while
     running judges from process memory, so the record cannot re-derive it
     (W0.PL, 11-14): such an entry is counted as one that could not be audited.

Everything that could not be inspected is COUNTED, never skipped. A gate whose
module will not import, that exposes no `_check`, or that raises when replayed
is a gate this scan did not audit, and an unaudited item that leaves the
numerator alone is how a clean scan and a scan that never ran become the same
number (T0.13 shipped that bug once already).
"""
from __future__ import annotations

import ast
import copy
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .protocol import (Ledger, NoControlByDecision, Status,
                       coerce_check_return, declares_a_control)

#: Marker for the one entry a scan may legitimately not judge: its own.
SELF_EXCLUDED = "self-excluded"

#: PASS entries recording `control_metrics` while their spec declares none.
#: Paid off 2026-08-10 (19 -> 0); may only be lowered. History: git.
UNDECLARED_CONTROL_BUDGET = 0


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
    issued_at: str = ""        # non-empty = gate loaded from the row's own commit


def _verdict(fn: Callable, m: dict, c: dict, spec_id: str = ""):
    """Deep-copied every call: some checks WRITE to their metrics (T2.02 sets
    m["verdict"]), and a probe that let that leak would score the second
    evaluation against a mutated first.

    THE RETURN TYPE IS VALIDATED THROUGH `protocol.coerce_check_return`, the
    same function the runner uses, and this line is the whole point of that
    function existing (2026-09-26). `bool(out)` used to stand here, and a
    `_check` returning `(Status.VOID, reason)` — `LT.03`, the one
    verdict-inverting row this ledger has held — read `("BOOL", True)`, i.e.
    *"the recorded PASS still re-derives"*. Probe A was silent and probe B was
    worse than silent: with the control deleted the constant-truthy tuple is
    unmoved, so the entry would have been reported as a gate that IGNORES ITS
    CONTROL — a red naming the wrong defect. A refused return lands in
    `unevaluable_gates` instead: this scan did not re-derive that verdict, and
    says so.
    """
    try:
        out = coerce_check_return(fn(copy.deepcopy(m), copy.deepcopy(c)), spec_id)
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
    issued: List[str] = []

    for e in entries:
        if e.issued_at:
            issued.append(f"{e.spec_id}@{e.issued_at}")
        # Self-exclusion: a known one-entry hole, counted apart so it stays visible.
        if e.unavailable == SELF_EXCLUDED:
            self_excluded.append(e.spec_id)
            continue
        if e.unavailable or e.check is None:
            unavailable.append(f"{e.spec_id}({e.unavailable or 'no _check'})")
            continue

        # ── C. declaration coherence (independent of whether the gate runs) ──
        # `declares_a_control`, not `bool(...)`: see REFUSAL in the docstring;
        # shared with the runner's guard so the two cannot drift.
        promises_control = declares_a_control(e.declared_control)
        has_control_metrics = bool(e.control_metrics)
        if promises_control and not has_control_metrics:
            declared_never_ran.append(e.spec_id)
        if not promises_control and has_control_metrics:
            # Both directions: `control=None` debt, and a refusal that ran one.
            undeclared_ran.append(e.spec_id)

        # ── A. re-verdict ────────────────────────────────────────────────────
        base = _verdict(e.check, e.metrics, e.control_metrics, e.spec_id)
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
        "issued_gate_detail": ", ".join(sorted(issued)),
    }


def collect(ledger: Ledger, exclude: tuple = ()) -> List[Entry]:
    """Build the entry list from the live ledger and the live registry.

    `exclude`: a spec cannot re-judge its OWN entry (written after the scan);
    excluded ids are returned marked unavailable, so they are still counted.
    A row whose gate grew a key since is judged by the gate that issued it.
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
        state = gate_reads_module_state(Path(mod.__file__).read_text())
        if state:
            out.append(Entry(spec_id, None, metrics, controls, declared,
                             unavailable="gate reads module state " + "/".join(state)))
            continue
        issued = ""
        if r.commit and _missing_key(fn, metrics, controls):
            fn_at = _check_at(spec_id, r.commit)
            if fn_at is not None:
                fn, issued = fn_at, r.commit
        out.append(Entry(spec_id, fn, metrics, controls, declared,
                         issued_at=issued))
    return out


_MUTATORS = {"append", "extend", "clear", "pop", "update", "insert",
             "setdefault", "popitem", "add", "remove"}


def gate_reads_module_state(src: str) -> list:
    """Module-level containers this module MUTATES that its `_check` reads."""
    t = ast.parse(src)
    box = set()
    for n in t.body:
        v = getattr(n, "value", None)
        tg = n.targets if isinstance(n, ast.Assign) else [getattr(n, "target", None)]
        if isinstance(v, (ast.List, ast.Dict, ast.Set)) or (
                isinstance(v, ast.Call) and getattr(v.func, "id", "") in ("list", "dict", "set")):
            box |= {x.id for x in tg if isinstance(x, ast.Name)}
    hit = {x.func.value.id for x in ast.walk(t) if isinstance(x, ast.Call)
           and isinstance(x.func, ast.Attribute) and x.func.attr in _MUTATORS
           and isinstance(x.func.value, ast.Name)}
    hit |= {x.value.id for x in ast.walk(t) if isinstance(x, ast.Subscript)
            and isinstance(x.ctx, ast.Store) and isinstance(x.value, ast.Name)}
    ch = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_check"]
    return sorted({x.id for c in ch for x in ast.walk(c)
                   if isinstance(x, ast.Name)} & box & hit)


def _missing_key(fn, m, c) -> bool:
    """Today's gate asks for a key the row never recorded (gate newer than the
    row; `run stale` reports that). Any other raise stays unevaluable."""
    try:
        fn(dict(m), dict(c))
    except KeyError as exc:
        return bool(exc.args) and exc.args[0] not in m and exc.args[0] not in c
    except Exception:
        return False
    return False


def _check_at(spec_id: str, commit: str):
    """`_check` from the spec's test file as it stood at `commit`, or None."""
    import subprocess
    import types
    from .run import _REPO, _module_path_for
    path = _module_path_for(spec_id)
    if path is None:
        return None
    rel = Path(path).resolve().relative_to(_REPO)
    try:
        src = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=_REPO,
                             capture_output=True, text=True, timeout=10,
                             check=True).stdout
        mod = types.ModuleType(f"experiments.tests._at_{commit}")
        mod.__package__ = "experiments.tests"
        exec(compile(src, f"{rel}@{commit}", "exec"), mod.__dict__)
    except Exception:
        return None
    return getattr(mod, "_check", None)


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


def _fixture_tuple_return(m, c):
    """The LT.03 shape, planted (2026-09-26). A `_check` that answers with
    `(Status, reason)` instead of a verdict: every branch is truthy, so the
    recorded PASS re-derives no matter what the numbers say, AND the answer
    cannot move when the control is deleted.

    THIS FIXTURE IS THE REASON THE CLASS IS NOT INVISIBLE AGAIN. The scan
    passed its own known-answer battery on the day it certified `LT.03`'s
    false PASS, because no planted entry returned anything but a bool: a
    detector's fixture set is exactly the list of defects it can still see.
    """
    return (Status.VOID, "the panel trap is not live") if m["score"] > 0.5 \
        else (False, "below bar")


#: The live refusal declarations' SHAPE, built through the shipped class rather
#: than imitated — `T0.01`/`T0.10` construct exactly this, so the fixture
#: exercises the real thing and cannot keep passing after somebody changes what
#: a refusal is. Deliberately NOT a bare string starting "NONE, BY DECISION":
#: prose is not what earns the exemption, and a fixture that pretended otherwise
#: would be certifying the repair this file refused (option (ii), see the class).
_REFUSAL = NoControlByDecision(
    "NONE, BY DECISION (fixture): the mechanism has nothing to sabotage, and "
    "the reason is recorded here rather than in a null field.")


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
        # The verdict channel that cannot fail (LT.03, 2026-09-25). Must land
        # in `unevaluable_gates` — never in `judged`, and never in the
        # control-blind bucket, where it would name the wrong defect.
        Entry("FIX.tuple", _fixture_tuple_return, dict(good), dict(ctrl), "a control"),
        # ── the refusal pair, planted 2026-09-27 ────────────────────────────
        # THE FALSE POSITIVE THIS SCAN SHIPPED: a spec that declares, in words,
        # that it has NO control, and ran none. It promised nothing, so it must
        # be flagged NOWHERE — it belongs in `no_control_specs` beside the
        # honest nulls. This is the half that makes the number true.
        #
        # Byte-identical to `FIX.promised` above except for the declaration, on
        # purpose: the predicate under test is the ONLY difference between a
        # promise broken and a refusal kept, so nothing else may vary. (It reads
        # the same control-blind gate for the same mechanical reason that entry
        # does — a gate touching `c["score"]` raises KeyError with no control.)
        Entry("FIX.refused", _fixture_control_blind, dict(good), {}, _REFUSAL),
        # ...AND THE HOLE THAT RECOGNISING THE REFUSAL OPENS, which is why the
        # two are planted together. A declaration asserting "no control here"
        # over recorded control metrics is the audit surface lying in the one
        # direction nothing watched: while probe C read `bool(...)` this entry
        # satisfied NEITHER branch, so it was invisible to both halves at once.
        # It must land in `undeclared_control_ran`.
        Entry("FIX.refused_ran", _fixture_healthy, dict(good), dict(ctrl), _REFUSAL),
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
    if r["undeclared_control_ran"] != 2:
        bad.append(f"undeclared probe found {r['undeclared_control_ran']} of 2")
    # ── the refusal pair, and BOTH directions are load-bearing ──────────────
    # Pinned by NAME, not by count. `declared_control_never_ran == 1` alone is
    # also satisfied by a scan that spares FIX.refused and flags FIX.promised's
    # twin instead, and `undeclared_control_ran == 2` alone is satisfied by a
    # scan that flags the healthy entry. The whole defect this pair exists for
    # was a counter reading a plausible number about the wrong specs.
    if "FIX.refused" in r["declared_never_ran_detail"].split(", "):
        bad.append("the explicit refusal was counted as a control promised and "
                   "never run — the false positive this pair was planted for")
    if "FIX.refused_ran" not in r["undeclared_ran_detail"]:
        bad.append("a declaration refusing a control while control_metrics is "
                   "non-empty went unflagged: "
                   f"undeclared_ran_detail = {r['undeclared_ran_detail']!r}")
    if "FIX.refused" not in r["no_control_detail"].split(", "):
        bad.append("the refusal that ran nothing did not land in no_control, so "
                   "it left the denominator rather than being counted")
    # The planted tuple-returner must be REFUSED, not judged — and it must be
    # refused under its own name. `unevaluable_gates == 1` alone would also be
    # satisfied by a scan that choked on the healthy fixture.
    if r["unevaluable_detail"] != "FIX.tuple(CheckReturnInvalid)":
        bad.append("the tuple-return probe read "
                   f"{r['unevaluable_detail']!r}, not FIX.tuple(CheckReturnInvalid)")
    if "FIX.tuple" in r["control_blind_detail"]:
        bad.append("the tuple-returner was reported as control-blind — the "
                   "wrong defect, which is how LT.03 would have been read")
    for key in ("control_blind_detail", "disagreement_detail",
                "declared_never_ran_detail", "undeclared_ran_detail"):
        if "FIX.healthy" in r[key]:
            bad.append(f"the healthy fixture was flagged in {key}")
    planted = "_S = []\nK = [1]\ndef _e(): _S.append(1)\ndef _check(m, c): return _S and K\n"
    if gate_reads_module_state(planted) != ["_S"]:
        bad.append("probe D missed a planted gate reading module state")
    if bad:
        raise RuntimeError(
            "the record verifier failed its own known-answer fixture (" +
            "; ".join(bad) + "); refusing to report a scan it may not have performed")
    return r
