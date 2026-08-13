"""T0.17 — a verdict that did not come from a run cannot look like one.

The overseer's RANK 1 finding, 2026-08-09: `experiments/ledger.json` carries
the header *"Do not hand-edit — a claim here must come from a test that could
have failed"* and had been hand-edited at least twice (T2.01's status
FAIL->VOID in `9b92d14`, T2.02 restated when `Status.VOID` was introduced).
Both edits were substantively right. The defect is that the file asserted a
distinction it had no field to carry, so a reader could not tell a
runner-recorded verdict from an agent-restated one.

Seven properties, each with a way to fail:

  1. An amendment is attributable: author, reason, prior value, commit, time.
  2. An amendment cannot reach PASS or FAIL — only statuses that assert
     nothing. PASS claims a capability; FAIL fires the spec's `kills`.
  3. An unattributed amendment is refused outright.
  4. `run_spec` never writes `amended`. The field means "not from a run", so a
     run that could set it would destroy the distinction.
  5. `attempt: None` is STICKY — a count that was never kept is not recovered
     by running again, and len(history)+1 would re-assert a number nobody
     measured.
  6. An amended verdict pushed into `history` by a later run keeps its
     amendment. Otherwise a re-run launders a hand-set status into an
     unqualified historical record.
  7. A superseded verdict carries its EVIDENCE into history — `metrics`,
     `control_metrics`, `impl_sha`, `seeds` — not just the verdict line
     (overseer B1, 2026-08-13). Without it, a threshold moved after a FAIL
     cannot be audited against the failing measurement by anyone but its
     author. Entries recorded before the fields existed stay evidence-free:
     absence must be preserved, never back-filled with invented numbers.
  8. A certificate that declares `IMPL_DEPS` while lacking `impl_sha` cannot
     stand over a dependency that has moved (14th audit B1, 2026-08-13).
     `impl_sha`'s staleness alarm was fitted to four world certificates that
     were recorded before the mechanism existed — the alarm was wired and
     structurally could not fire while `playground.py` took +430/-14 lines
     under them. Three sub-properties: `staleness_of` flags a pre-`impl_sha`
     entry whose declared dependency has commits after `ran_at`
     (known-positive), does NOT flag one whose `ran_at` postdates every such
     commit (known-negative), and the REAL ladder holds zero PASS records in
     the flagged state. The third is the class-closer: the four that motivated
     this were re-run on purpose the day it was written, and the next
     unprotected certificate turns this spec red instead of hiding.

CONTROL — the pre-fix path: the literal `9b92d14` edit (read the JSON, set
`status`, write it back) replayed on a temp ledger. Under the same audit it
must be INDISTINGUISHABLE from a recorded verdict. Without it this spec would
pass on a ledger where nothing was ever checkable.

Uses a temporary ledger throughout — this test must never touch the real one.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ..protocol import (Ledger, Result, Status, module_path_for, run_spec,
                        staleness_of)
from ..registry import BY_ID, LADDER, Spec, Budget

# Every property here is a claim about the recorder. Without this, an edit to
# protocol.py leaves T0.17's PASS describing a recorder that no longer exists
# — found 2026-08-13 while building T0.27, the same hole T0.12 had with gpu.py.
IMPL_DEPS = ["experiments/protocol.py"]

_DUMMY = Spec("X.17", 0, "dummy", hypothesis="h", falsified_by="f",
              null_baseline="n", metric="m", budget=Budget.CPU_FAST)


def _audit(path: Path, spec_id: str) -> bool:
    """Can a READER of the file tell this verdict did not come from a run?"""
    row = json.loads(path.read_text())["results"][spec_id]
    for note in row.get("amended") or []:
        if note.get("by") and note.get("reason") and note.get("changes"):
            return True
    return False


def _experiment(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ledger.json"
        led = Ledger(path)
        led.record(Result(spec_id="X.17", status=Status.FAIL,
                          metrics={"reward": 1.0}, ran_at="2026-01-01T00:00:00"))

        # 1. attributable amendment
        led.amend("X.17", by="T0.14", reason="dropout was live at eval",
                  status=Status.VOID)
        row = json.loads(path.read_text())["results"]["X.17"]
        note = row["amended"][-1]
        # Read the detector NOW: property 4 below deliberately re-runs X.17,
        # which supersedes the amendment (property 6 checks it is preserved in
        # history rather than lost).
        detector_sees = _audit(path, "X.17")
        attributable = (detector_sees
                        and row["status"] == "VOID"
                        and note["by"] == "T0.14"
                        and note["changes"][0] == {"field": "status",
                                                   "from": "FAIL", "to": "VOID"}
                        and bool(note.get("at")) and bool(note.get("commit")))

        # 2. an amendment may not assert a capability, nor fire `kills`
        refused_claims = []
        for bad in (Status.PASS, Status.FAIL):
            try:
                led.amend("X.17", by="me", reason="because", status=bad)
                refused_claims.append(False)
            except ValueError:
                refused_claims.append(True)

        # 3. an unattributed edit is refused
        refused_unattributed = []
        for by, why in (("", "reason only"), ("T0.14", "")):
            try:
                led.amend("X.17", by=by, reason=why, status=Status.SKIP)
                refused_unattributed.append(False)
            except ValueError:
                refused_unattributed.append(True)

        # 4. run_spec never writes `amended`
        run_spec(_DUMMY, lambda s: {"m": 1.0}, lambda m, c: True, ledger=Ledger(path))
        fresh = json.loads(path.read_text())["results"]["X.17"]
        run_leaves_clean = not fresh.get("amended")

        # 5. `attempt: None` survives a later run
        led2 = Ledger(path)
        led2.record(Result(spec_id="X.18", status=Status.PASS,
                           ran_at="2026-01-01T00:00:00"))
        led2.amend("X.18", by="overseer", reason="predates the history field",
                   unknown_history=True)
        Ledger(path).record(Result(spec_id="X.18", status=Status.PASS,
                                   ran_at="2026-02-02T00:00:00"))
        after = json.loads(path.read_text())["results"]["X.18"]
        unknown_is_sticky = after["attempt"] is None and len(after["history"]) == 1

        # 6. an amended verdict keeps its amendment when it becomes history
        led3 = Ledger(path)
        led3.record(Result(spec_id="X.19", status=Status.FAIL,
                           ran_at="2026-01-01T00:00:00"))
        led3.amend("X.19", by="T0.14", reason="invalid run", status=Status.VOID)
        Ledger(path).record(Result(spec_id="X.19", status=Status.PASS,
                                   ran_at="2026-03-03T00:00:00"))
        hist = json.loads(path.read_text())["results"]["X.19"]["history"]
        history_keeps_amendment = (len(hist) == 1 and hist[0]["status"] == "VOID"
                                   and bool(hist[0].get("amended")))

        # 7. a superseded verdict carries its evidence into history
        led4 = Ledger(path)
        led4.record(Result(spec_id="X.20", status=Status.FAIL,
                           metrics={"sigma_advantage": 2.67},
                           control_metrics={"untrained_sigma_advantage": 1.26},
                           seeds=[0, 1, 2], impl_sha="deadbeefdeadbeef",
                           ran_at="2026-01-01T00:00:00"))
        Ledger(path).record(Result(spec_id="X.20", status=Status.PASS,
                                   metrics={"sigma_advantage": 6.0},
                                   ran_at="2026-04-04T00:00:00"))
        h20 = json.loads(path.read_text())["results"]["X.20"]["history"]
        history_carries_evidence = (
            len(h20) == 1
            and h20[0]["metrics"] == {"sigma_advantage": 2.67}
            and h20[0]["control_metrics"] == {"untrained_sigma_advantage": 1.26}
            and h20[0]["impl_sha"] == "deadbeefdeadbeef"
            and h20[0]["seeds"] == [0, 1, 2])
        # ... and an entry that never had the fields is NOT back-filled: a
        # pre-B1 history row (verdict-only) superseded again must stay
        # evidence-free rather than acquire invented numbers.
        raw = json.loads(path.read_text())
        raw["results"]["X.21"] = {
            "spec_id": "X.21", "status": "FAIL", "ran_at": "2026-01-01T00:00:00",
            "commit": "old", "message": "", "history": [], "attempt": 1,
        }
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        Ledger(path).record(Result(spec_id="X.21", status=Status.PASS,
                                   ran_at="2026-05-05T00:00:00"))
        h21 = json.loads(path.read_text())["results"]["X.21"]["history"]
        absence_preserved = (len(h21) == 1 and "metrics" not in h21[0]
                             and "impl_sha" not in h21[0])

        # 8. unprotected certificates cannot hide. The wire-through uses THIS
        # file as the probe module: it declares IMPL_DEPS =
        # ["experiments/protocol.py"], and protocol.py has commits after 2020
        # and none after 2999, which gives a known-positive and a
        # known-negative through the real detector against the real git
        # history — no synthetic repo needed.
        me = module_path_for("T0.17")
        kinds_old = {k for k, _ in staleness_of(
            Result(spec_id="X.22", status=Status.PASS, impl_sha=None,
                   ran_at="2020-01-01T00:00:00"), me)}
        kinds_new = {k for k, _ in staleness_of(
            Result(spec_id="X.22", status=Status.PASS, impl_sha=None,
                   ran_at="2999-01-01T00:00:00"), me)}
        detector_flags_moved = ("UNVERIFIABLE_MOVED" in kinds_old
                                and "UNVERIFIABLE" in kinds_old)
        detector_spares_unmoved = ("UNVERIFIABLE_MOVED" not in kinds_new
                                   and "UNVERIFIABLE" in kinds_new)
        # ... and the REAL ladder holds no PASS in the flagged state. Read-only
        # on the real ledger (this test never writes it). The day this was
        # written the four (PG.1, PG.2, PG.4, T2.20) had just been re-run on
        # purpose; a name appearing here again means a new certificate slipped
        # into the unprotected class.
        real = Ledger()
        unprotected = []
        for s in LADDER:
            e = real.results.get(s.id)
            p = module_path_for(s.id)
            if e is None or p is None or real.status(s.id) is not Status.PASS:
                continue
            if any(k == "UNVERIFIABLE_MOVED" for k, _ in staleness_of(e, p)):
                unprotected.append(s.id)
        if unprotected:
            print(f"    T0.17 P8: unprotected PASS certificates: {unprotected}")

        return {
            "detector_flags_moved_dependency": detector_flags_moved,
            "detector_spares_unmoved_dependency": detector_spares_unmoved,
            "unprotected_pass_certificates": len(unprotected),
            "amendment_is_attributable": attributable,
            "refuses_pass_and_fail": all(refused_claims),
            "refuses_unattributed": all(refused_unattributed),
            "run_spec_leaves_amended_empty": run_leaves_clean,
            "unknown_attempt_is_sticky": unknown_is_sticky,
            "history_keeps_amendment": history_keeps_amendment,
            "history_carries_evidence": history_carries_evidence,
            "history_absence_preserved": absence_preserved,
            "detector_sees_amendment": detector_sees,
        }


def _control(seed: int) -> dict:
    """The 9b92d14 hand-edit, replayed. The audit must NOT see it.

    A control that also passed would mean the detector answers "amended" to
    everything, i.e. measures nothing.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ledger.json"
        led = Ledger(path)
        led.record(Result(spec_id="X.17", status=Status.FAIL,
                          metrics={"reward": 1.0}, message="",
                          ran_at="2026-01-01T00:00:00"))

        raw = json.loads(path.read_text())
        raw["results"]["X.17"]["status"] = "VOID"          # verbatim 9b92d14
        raw["results"]["X.17"]["message"] = "invalidated by T0.14"
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")

        hand_set = json.loads(path.read_text())["results"]["X.17"]["status"] == "VOID"
        return {
            "hand_edit_took_effect": hand_set,
            "detector_sees_amendment": _audit(path, "X.17"),
        }


def _check(m: dict, c: dict) -> bool:
    return all([
        m["detector_flags_moved_dependency"],
        m["detector_spares_unmoved_dependency"],
        m["unprotected_pass_certificates"] == 0,
        m["amendment_is_attributable"],
        m["refuses_pass_and_fail"],
        m["refuses_unattributed"],
        m["run_spec_leaves_amended_empty"],
        m["unknown_attempt_is_sticky"],
        m["history_keeps_amendment"],
        m["history_carries_evidence"],
        m["history_absence_preserved"],
        m["detector_sees_amendment"],
        # the control must fail: the hand-edit lands and stays invisible
        c["hand_edit_took_effect"],
        not c["detector_sees_amendment"],
    ])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.17"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
