"""T0.17 — a verdict that did not come from a run cannot look like one.

The overseer's RANK 1 finding, 2026-08-09: `experiments/ledger.json` carries
the header *"Do not hand-edit — a claim here must come from a test that could
have failed"* and had been hand-edited at least twice (T2.01's status
FAIL->VOID in `9b92d14`, T2.02 restated when `Status.VOID` was introduced).
Both edits were substantively right. The defect is that the file asserted a
distinction it had no field to carry, so a reader could not tell a
runner-recorded verdict from an agent-restated one.

Six properties, each with a way to fail:

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

from ..protocol import Ledger, Result, Status, run_spec
from ..registry import BY_ID, Spec, Budget

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

        return {
            "amendment_is_attributable": attributable,
            "refuses_pass_and_fail": all(refused_claims),
            "refuses_unattributed": all(refused_unattributed),
            "run_spec_leaves_amended_empty": run_leaves_clean,
            "unknown_attempt_is_sticky": unknown_is_sticky,
            "history_keeps_amendment": history_keeps_amendment,
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
        m["amendment_is_attributable"],
        m["refuses_pass_and_fail"],
        m["refuses_unattributed"],
        m["run_spec_leaves_amended_empty"],
        m["unknown_attempt_is_sticky"],
        m["history_keeps_amendment"],
        m["detector_sees_amendment"],
        # the control must fail: the hand-edit lands and stays invisible
        c["hand_edit_took_effect"],
        not c["detector_sees_amendment"],
    ])


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.17"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
