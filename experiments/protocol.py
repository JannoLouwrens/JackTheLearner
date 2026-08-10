"""The experiment contract.

Every claim about Jack must be produced by an experiment that COULD HAVE FAILED.
That is the whole point of this package, and it exists because the repo currently
carries 45.5M parameters with no live call site and a README status table reading
"Working" for modules that have never received a gradient.

Three rules are enforced structurally rather than by discipline:

1. A spec must declare `falsified_by` — the observation that would prove the
   hypothesis wrong. A test with no failure mode is not a test.
2. A spec must declare `null_baseline` — what the metric reads when the mechanism
   does nothing. Without it you cannot distinguish "my method works" from
   "anything would have worked", which is the single most common way ML projects
   fool themselves.
3. Status defaults to NOT_RUN and only the runner may change it. Nothing can be
   marked working by editing a document.

The ledger is the source of truth for every capability claim. README status is
rendered FROM it, never written by hand.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LEDGER_PATH = Path(__file__).parent / "ledger.json"


class Status(str, Enum):
    NOT_RUN = "NOT_RUN"      # default; never set by hand
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"      # a dependency failed, so this cannot be trusted
    ERROR = "ERROR"          # the test itself crashed — distinct from FAIL
    SKIP = "SKIP"            # deliberately out of scope, with a reason
    VOID = "VOID"            # the run was INVALID — it did not test the claim
    """VOID is not FAIL, and conflating them corrupts the record.

    FAIL means the hypothesis was tested and lost. VOID means the run could not
    test it at all: an arm that never learned, a fixture that leaked, a
    measurement that turned out to be an artifact. The distinction has teeth
    because specs carry a `kills` field — T2.02 kills "the transformer policy",
    and it was recorded FAIL with the message "pre-registered threshold not
    met" while its own metrics read "VOID — two non-learners cannot arbitrate
    the architecture". Read machine-side, that ledger said the kill criterion
    had fired on a comparison that explicitly refused to arbitrate.

    A VOID spec is not demonstrated and must not be counted as PASS, but it
    also does not trigger `kills` and does not BLOCK its dependents on the
    grounds that the claim was refuted. It means: fix the run and try again.
    """


class Budget(str, Enum):
    """Where a test runs. Ordering matters: the ladder front-loads CPU work so a
    hypothesis dies before it costs GPU quota."""
    CPU_FAST = "cpu<1min"
    CPU = "cpu<10min"
    CPU_LONG = "cpu<2h"
    GPU_SHORT = "gpu<20min"
    GPU = "gpu<2h"
    GPU_LONG = "gpu<8h"      # must checkpoint; Kaggle caps sessions at 12h


@dataclass
class Spec:
    """One falsifiable experiment."""
    id: str
    tier: int
    title: str
    hypothesis: str
    """What we assert. Must be specific enough to be wrong."""
    falsified_by: str
    """The observation that kills it. If you cannot write this, it is not a test."""
    null_baseline: str
    """What the metric reads when the mechanism contributes nothing."""
    metric: str
    budget: Budget
    depends_on: List[str] = field(default_factory=list)
    seeds: int = 1
    """RL effect sizes are routinely smaller than seed noise. Anything claiming a
    performance improvement runs >=3 seeds and reports mean+-std."""
    control: Optional[str] = None
    """A condition that MUST fail. Shuffled labels, frozen weights, injected noise.
    A test whose control also passes is measuring nothing."""
    kills: Optional[str] = None
    """What we delete or abandon if this fails. Forces the cost of failure to be
    decided before the result is known."""
    notes: str = ""
    gate_mode: str = "validity"
    """How `run_bakeoff` reads an arm that misses the learning gate.

    `validity` (default, and the T2.02 rule): a missing arm VOIDs the bakeoff,
    because a LEARNER that did not learn cannot be told apart from a learner
    that is worse, so it cannot arbitrate anything.

    `screen`: a missing arm is ELIMINATED and the survivors still compete. Only
    legitimate when the arms are OBSERVABLES rather than learners — a
    deterministic function of shared, already-collected data, where a low score
    is a property of the arm and not evidence that its run was broken. It is
    NOT a weaker gate: the gate is unchanged, at least `bakeoff.MIN_FINISHERS`
    arms must still clear it, and controls still invert the verdict.

    It lives on the Spec, not on the `run_bakeoff` call, on purpose: the mode
    is a pre-registration, and a caller that could pass it as an argument could
    change it after seeing a VOID. (The LC.01 rule — the artifact names what it
    is, never the auditor.)"""
    screen_rationale: Optional[str] = None
    """REQUIRED when `gate_mode == "screen"`: why these arms are observables and
    not learners. Recorded verbatim in `docs/DECISIONS_RESOLVED.md`, so the
    justification is re-readable next to the verdict it permitted."""


@dataclass
class Result:
    spec_id: str
    status: Status
    metrics: Dict[str, Any] = field(default_factory=dict)
    control_metrics: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=list)
    duration_s: float = 0.0
    commit: str = ""
    hardware: str = ""
    ran_at: str = ""
    message: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    """Previous attempts, trimmed. Written by Ledger.record — see the note there."""
    attempt: Optional[int] = 1
    """Which attempt this is, or None when the count is NOT RECONSTRUCTIBLE.

    Five entries (T2.01, T2.02, T1.02, T0.05, T0.09) read `attempt: 1,
    history: []` because they predate the history mechanism — T2.01 alone has
    four versions in git. A wrong integer is worse than a null: it is the
    `Arm.cost` lesson in a second file, a default that cannot be told from a
    measurement. None is STICKY (see Ledger.record): unknown + one more run is
    still unknown, because no future run recovers a count that was never kept.
    """
    amended: List[Dict[str, Any]] = field(default_factory=list)
    """Changes to this entry that did NOT come from a run.

    `run_spec` never writes here — only `Ledger.amend`, i.e. only
    `python -m experiments.run amend`. It exists because the ledger's own
    header forbids hand-editing and the file had been hand-edited twice
    anyway (T2.01 FAIL->VOID in 9b92d14; T2.02 restated when VOID was
    introduced). Both edits were substantively right and the record could not
    say they were edits. An `amend` may only set a status that ASSERTS
    NOTHING — VOID, SKIP, NOT_RUN. PASS and FAIL still require a run: PASS
    claims a capability, FAIL fires the spec's `kills`.
    """
    impl_sha: Optional[str] = None
    """sha256 of the test file this result was produced by, at run time.

    `commit` alone cannot answer "is this entry still about the code that
    produced it": a test is normally written, RUN, and only then committed, so
    the recorded commit predates the test's own first commit and "any commit
    touching the file since" fires on every honest entry. The file's content
    hash answers it exactly. `None` means the entry predates this field and is
    UNVERIFIABLE — deliberately not `""`, because a sentinel that is also a
    valid value cannot be detected (the `Arm.cost` lesson).
    """

    @staticmethod
    def env_stamp() -> Dict[str, str]:
        root = Path(__file__).parent.parent
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=root, timeout=10,
            ).stdout.strip()
        except Exception:
            commit = "unknown"
        # A COMMIT STAMP ASSUMES A CLEAN TREE, AND SAYS SO WHEN IT IS NOT.
        # `ccd0e84` hoisted this stamp above the seed loop so a GPU entry names
        # the code that RAN rather than whatever HEAD drifted to. That closed
        # the drift; it did not close the other way the same sentence can be
        # false. A spec run from a modified working tree executes HEAD *plus*
        # uncommitted edits, and `rev-parse` cannot see the difference — which
        # is the ordinary case for this loop, because a builder edits a test and
        # runs it before committing. It happened on 2026-08-10: XL.00 attempt 3
        # is stamped `1480126` and ran `1480126` plus a rewritten control gate.
        # `impl_sha` catches it AFTERWARDS, once the file is committed and the
        # hash no longer matches, but only for the one file it hashes and only
        # in hindsight. Naming it at run time costs one subprocess.
        # `ledger.json` is excluded on purpose: it is the runner's own output,
        # not code, and it is legitimately dirty whenever a previous result is
        # waiting to be committed.
        try:
            porcelain = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=root, timeout=10,
            ).stdout.splitlines()
            dirty = [ln for ln in porcelain
                     if ln[3:].strip() and not ln[3:].strip().endswith("ledger.json")]
            if dirty and commit not in ("", "unknown"):
                commit += "+dirty"
        except Exception:
            pass
        hw = f"{platform.machine()}/{platform.system()}"
        try:
            import torch
            hw += f"/torch{torch.__version__}"
            if torch.cuda.is_available():
                hw += f"/{torch.cuda.get_device_name(0)}"
            else:
                hw += "/cpu"
        except Exception:
            pass
        return {"commit": commit, "hardware": hw}


class Ledger:
    """Append-only record of what has actually been demonstrated.

    Deliberately dumb — a JSON file in git. The value is not the format, it is
    that the file is the ONLY thing permitted to assert a capability.
    """

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path
        self.results: Dict[str, Result] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for rid, r in raw.get("results", {}).items():
            r["status"] = Status(r["status"])
            self.results[rid] = Result(**r)

    def record(self, result: Result) -> None:
        """Merge ONE result into the ledger under an exclusive lock.

        Naive save() lost data: each Ledger held an in-memory copy and wrote the
        whole file, so a writer with a stale view silently erased results it had
        never seen. Measured, two interleaved writers kept 11 of 15 records
        (ladder spec T0.08). That is not hypothetical — the hourly loop and a
        manual session overlap by design.

        So: lock, RE-READ from disk, merge, write atomically. The tmp+os.replace
        is the same pattern T0.05 established, because a ledger truncated by a
        SIGKILL would erase the evidence for every capability claimed so far.

        THE WORD `ONE` IS LOAD-BEARING, 2026-08-10. The re-read above was only
        half a fix: having read the fresh file into `merged`, this method then
        looped over **all** of `self.results` and wrote every entry it happened
        to be holding back over it. So a long-lived Ledger did not merely fail to
        see newer work — it actively reverted it, and the revert was disguised as
        legitimate history, because the `ran_at` mismatch below pushed the FRESH
        on-disk verdict down into `history` and re-asserted the stale one as
        current with `attempt` incremented. A reader saw "re-run, attempt 4", not
        "six hours of work deleted".

        It happened. A `run T2.01` GPU poll constructed its Ledger at 19:42 on
        2026-08-09, waited 5.6 h for a Kaggle P100, and recorded at 01:17 on
        2026-08-10. That single write reverted six intervening entries (LC.01,
        PG.3, PG.8, T0.08, T0.13, T0.15) to their 19:42 values and erased the
        five `amended` records the overseer had written at 00:12. Only an
        uncommitted working tree saved it.

        So the merge is now strictly single-key: everything else on disk is
        copied through untouched, whatever this instance believes about it. The
        instance then adopts the merged file wholesale, so it cannot stay stale
        after a write either.
        """
        self.results[result.spec_id] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")

        with open(lock_path, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                on_disk: Dict[str, Any] = {}
                if self.path.exists():
                    try:
                        on_disk = json.loads(self.path.read_text()).get("results", {})
                    except json.JSONDecodeError:
                        on_disk = {}          # corrupt: rebuild from memory rather than lose everything

                merged = dict(on_disk)
                # EXACTLY ONE key may change per call — the one being recorded.
                # See the docstring: iterating self.results here reverted six
                # entries and erased five amendments on 2026-08-10.
                for rid, r in ((result.spec_id, result),):
                    # KEEP THE PREVIOUS ATTEMPT. Overwriting by spec_id made
                    # SYSTEM.md's "the failing version stays in the ledger's
                    # history" unenforceable: a spec redesigned three times
                    # (T1.02) or fixed after a real bug (T2.01) showed only its
                    # final green tick, so the system could not measure its own
                    # first-attempt pass rate — the one number that says whether
                    # the specs are honestly risky or written to pass. History
                    # is a trimmed record: status, when, commit and message, not
                    # full metrics, so the file stays readable.
                    prev = on_disk.get(rid)
                    hist = list(prev.get("history", [])) if prev else []
                    if prev and prev.get("ran_at") != r.ran_at:
                        row_h = {k: prev.get(k) for k in
                                 ("status", "ran_at", "commit", "message")}
                        if prev.get("amended"):
                            # An amendment is part of what that verdict WAS.
                            # Dropping it here would let a re-run launder a
                            # hand-set status back into an unqualified record.
                            row_h["amended"] = prev["amended"]
                        hist.append(row_h)
                    row = {**asdict(r), "status": r.status.value}
                    row["history"] = hist[-20:]
                    # None is sticky: a count that was never kept is not
                    # recovered by running again, and len(hist)+1 would quietly
                    # re-assert a number nobody measured.
                    unknown = r.attempt is None or (prev is not None
                                                    and prev.get("attempt", 1) is None)
                    row["attempt"] = None if unknown else len(hist) + 1
                    merged[rid] = row

                self._write_atomic(merged)

                # Adopt the merged file WHOLESALE, not just the keys we lack.
                # `if rid not in self.results` left every already-known entry at
                # its load-time value, which is precisely how this instance's
                # view went stale in the first place. After a write, memory ==
                # file, so the next record() starts from the truth.
                fresh: Dict[str, Result] = {}
                for rid, raw in merged.items():
                    d = dict(raw)
                    d["status"] = Status(d["status"])
                    fresh[rid] = Result(**d)
                self.results = fresh
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _write_atomic(self, merged: Dict[str, Any]) -> None:
        """tmp + fsync + os.replace. Callers hold the lock."""
        payload = {
            "_comment": "Written by experiments/run.py under an exclusive lock. "
                        "Do not hand-edit — a claim here must come from a test "
                        "that could have failed. A change that did not come "
                        "from a run goes through `run amend` and is recorded "
                        "in that entry's `amended` list.",
            "results": merged,
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    #: Statuses an amendment may set. All three ASSERT NOTHING. PASS is a
    #: capability claim and FAIL fires `kills`; both must come from a run.
    AMENDABLE = (Status.VOID, Status.SKIP, Status.NOT_RUN)

    def amend(self, spec_id: str, by: str, reason: str,
              status: Optional[Status] = None,
              unknown_history: bool = False) -> Dict[str, Any]:
        """Change an entry WITHOUT a run, and make the entry say so.

        The ledger's header forbids hand-editing, and the file was hand-edited
        twice anyway (T2.01's status FAIL->VOID in `9b92d14`, T2.02 restated
        when VOID was introduced). Both edits were right — T0.14 genuinely
        invalidated those runs and leaving FAIL in place would have fired the
        `kills` field off a run that never arbitrated. The defect was that a
        reader could not tell a runner-recorded verdict from an agent-restated
        one, in a file asserting no such distinction exists.

        So: the runner stays the only writer, and every non-run change lands in
        `amended` with its author, reason, prior value, commit and time. The
        guard that makes this safe rather than a licence is `AMENDABLE` — an
        amendment can only ever move an entry to a status that claims nothing.
        """
        if not by or not reason:
            raise ValueError("amend requires both --by (the spec or finding that "
                             "invalidates this) and --reason; an unattributed edit "
                             "is the thing this mechanism exists to prevent")
        if status is not None and status not in self.AMENDABLE:
            raise ValueError(
                f"amend may not set {status.value}: only "
                f"{', '.join(s.value for s in self.AMENDABLE)} assert nothing. "
                "PASS claims a capability and FAIL fires the spec's `kills` — "
                "both require a run that could have failed.")
        if status is None and not unknown_history:
            raise ValueError("amend with nothing to change")

        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                on_disk = json.loads(self.path.read_text()).get("results", {})
                row = on_disk.get(spec_id)
                if row is None:
                    raise KeyError(f"{spec_id} has no ledger entry to amend")

                changes: List[Dict[str, Any]] = []
                if status is not None:
                    changes.append({"field": "status", "from": row.get("status"),
                                    "to": status.value})
                    row["status"] = status.value
                if unknown_history:
                    changes.append({"field": "attempt", "from": row.get("attempt"),
                                    "to": None})
                    row["attempt"] = None

                note = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "by": by, "reason": reason,
                        "changes": changes, **Result.env_stamp()}
                note.pop("hardware", None)      # an edit has no hardware
                row["amended"] = list(row.get("amended") or []) + [note]
                on_disk[spec_id] = row
                self._write_atomic(on_disk)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        self.load()
        return row

    def save(self) -> None:
        """Removed 2026-08-10. `record(result)` is the only write path.

        This wrote "whatever this object happens to hold", which is the shape of
        the bug documented on `record`: an instance's in-memory view is a
        snapshot that goes stale the moment another writer runs, so flushing it
        is never safe. It had no callers; it raises rather than being deleted so
        that a caller added later fails loudly instead of quietly reverting the
        file.
        """
        raise NotImplementedError(
            "Ledger.save() is gone: it flushed a whole stale snapshot. "
            "Call record(result) once per result — it merges exactly one key.")

    def status(self, spec_id: str) -> Status:
        r = self.results.get(spec_id)
        return r.status if r else Status.NOT_RUN

    def blocked_by(self, spec: Spec) -> List[str]:
        """Dependencies that are not passing. A result computed on a broken
        foundation is worse than no result, because it looks like evidence."""
        return [d for d in spec.depends_on if self.status(d) is not Status.PASS]

    def summary(self) -> Dict[str, int]:
        out = {s.value: 0 for s in Status}
        for r in self.results.values():
            out[r.status.value] += 1
        return out


class VoidStatusMismatch(RuntimeError):
    """A `_check` said VOID in its metrics and returned FAIL in its status.

    The two cannot both be recorded, and the FAIL is the dangerous one: it
    fires the spec's `kills` field. Raised so the run lands as ERROR — visibly
    unfinished — rather than as a confident wrong verdict.
    """


class UndeclaredControl(RuntimeError):
    """A spec runs a control it never declared, so `Spec.control` reads None.

    `Spec.control` is the field an auditor greps to ask "which claims here are
    sabotage-tested?". On 2026-08-10 twenty specs ran a `control_fn` while
    declaring nothing — 19 of 53 recorded entries, so the grep answered "no
    control" for more than a third of the ladder's sabotage-tested claims and
    was therefore useless in BOTH directions: it could not be trusted when it
    said yes either. The overseer carried it for four audits.

    The prose declaration is not decoration. It is the only place that says
    which WAY the control must fail; `control_fn` returns numbers, and a
    reader cannot tell a control that must collapse from one that must diverge
    without being told. So the declaration is now a precondition of running,
    checked BEFORE any compute is spent.
    """


def _declares_void(metrics: Dict[str, Any]) -> Optional[str]:
    """Return the metric value that declares VOID, if any."""
    for v in metrics.values():
        if isinstance(v, str) and "VOID" in v.upper():
            return v[:120]
    return None


def impl_deps_of(path) -> tuple:
    """A test module's `IMPL_DEPS`, read STATICALLY from its source.

    Static because the reader of a sha must not import the module to check it:
    `run.stale_claims` scans the whole ladder and importing every test would
    pull in mujoco, GL contexts and torch for a question about bytes on disk.

    Returns `(deps, problem)`. `problem` is non-empty when the declaration
    exists but cannot be read as a literal list of strings — reported rather
    than swallowed, because falling back to "no deps" is exactly the silent
    narrowing that produced the bug this function was extracted to fix.
    """
    import ast
    try:
        tree = ast.parse(Path(path).read_bytes())
    except (OSError, SyntaxError) as e:
        return (), f"unreadable:{type(e).__name__}"
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "IMPL_DEPS"
                   for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return (), "IMPL_DEPS is not a literal"
        if not (isinstance(value, (list, tuple))
                and all(isinstance(x, str) for x in value)):
            return (), "IMPL_DEPS is not a list of paths"
        return tuple(value), ""
    return (), ""


def impl_sha_of(path) -> Optional[str]:
    """sha256 of a test file — the test as it was when it ran.

    PLUS any files the test module declares in `IMPL_DEPS`, because a test file
    is not the whole of what a test measures. PG.6 certifies what Jack's eye can
    resolve; that claim is about `playground.py`'s camera pose, and on
    2026-08-10 moving the eye would have left PG.6's PASS standing and the
    staleness checker silent — a certificate about a world invalidated by a
    change to the world, undetectable by design.

    Modules that declare nothing hash EXACTLY as before, byte for byte. That is
    deliberate: making this unconditional would change every recorded sha at
    once and flag the whole ladder stale, which is a mass false alarm and would
    teach the loop to ignore staleness warnings. Declaring a dependency is
    opt-in and costs one line:

        IMPL_DEPS = ["playground.py"]     # paths relative to the repo root

    Paths that do not resolve are recorded as the literal string `missing:` plus
    the path rather than skipped, so a typo shows up as a permanent mismatch
    instead of silently reverting to test-file-only hashing.

    THIS FUNCTION IS THE WHOLE DEFINITION OF `impl_sha`, and it takes a PATH so
    that the writer and the reader are one code path. They were two, and they
    disagreed: `_impl_sha` hashed file + `IMPL_DEPS` while `run.stale_claims`
    hashed the file alone, so all twelve specs declaring `IMPL_DEPS` were
    reported stale FOREVER and no re-run could clear the flag. XL.00 re-ran
    clean on 2026-08-10 and was still listed. A checker whose false positives
    survive the only action it recommends is worse than absent — it bills real
    iterations for nothing (the previous hand-off had already queued a needless
    LC.02 re-run on its say-so). Two functions computing "the same" hash is a
    thing to remove, not to keep in sync.
    """
    import hashlib
    try:
        h = hashlib.sha256(Path(path).read_bytes())
        deps, problem = impl_deps_of(path)
        if problem:
            h.update(f"undeclarable:{problem}".encode())
        for rel in deps:
            dep = Path(__file__).resolve().parent.parent / rel
            h.update(dep.read_bytes() if dep.is_file()
                     else f"missing:{rel}".encode())
        return h.hexdigest()[:16]
    except (OSError, TypeError):
        return None


def module_path_for(spec_id: str, strict: bool = False):
    """The implementation FILE for a spec, without importing it.

    Lives here rather than in `run.py` because `borrow_metrics` needs it and
    `run.py` imports this module, not the other way round. `run._module_for`
    and `run._module_path_for` call it, so there is one rule for "which file
    implements this spec" — the duplicate-implementation trap it guards is
    described at `run._module_for`, and `strict=True` keeps that raise.
    """
    from .registry import LADDER
    tests = Path(__file__).resolve().parent / "tests"
    prefix = spec_id.lower().replace(".", "_")
    matches = sorted(tests.glob(f"{prefix}_*.py"))
    longer = [s.id.lower().replace(".", "_") for s in LADDER
              if s.id != spec_id and s.id.lower().replace(".", "_").startswith(prefix + "_")]
    if longer:
        matches = [m for m in matches
                   if not any(m.stem.startswith(p + "_") for p in longer)]
    if strict and len(matches) > 1:
        raise RuntimeError(
            f"{spec_id} has {len(matches)} implementations: "
            f"{', '.join(m.name for m in matches)}. Delete or merge — the runner "
            "will not choose between them.")
    return matches[0] if len(matches) == 1 else None


def staleness_of(entry: "Result", path) -> List[tuple]:
    """Every reason this entry is not a claim about the code that exists now.

    Returns a list of `(kind, detail)` — empty means the entry still describes
    the current implementation. Kinds:

      DIRTY        the run's commit stamp ends in `+dirty`, so the code that
                   produced it exists in no commit and cannot be recovered.
      UNVERIFIABLE the entry predates `impl_sha`; nothing can be compared.
      CHANGED      the implementation hash moved since the run.

    An entry can be DIRTY *and* CHANGED — they are different facts about it,
    so this returns a list rather than one verdict.

    THIS IS THE DEFINITION, and it has exactly one home on purpose. It was
    inlined in `run.stale_claims`, which is a REPORT; the moment a second
    consumer appeared (`borrow_metrics`, so a test cannot compute on a stale
    number) the choice was to copy the rule or to call it. The last time this
    repo kept two implementations of "the same" hash they diverged silently and
    every `IMPL_DEPS` spec was flagged stale forever — see `impl_sha_of`.
    """
    out: List[tuple] = []
    stamp = str(getattr(entry, "commit", "") or "")
    if stamp.endswith("+dirty"):
        out.append(("DIRTY", f"ran from a modified tree at {stamp.split('+')[0]}; "
                             f"the code that ran was never committed"))
    recorded = getattr(entry, "impl_sha", None)
    if not recorded:
        out.append(("UNVERIFIABLE",
                    f"recorded at {(entry.commit or '?')[:8]} before impl_sha existed"))
        return out
    cur = impl_sha_of(path)
    if cur != recorded:
        out.append(("CHANGED",
                    f"{Path(path).name}: ran on {recorded}, now {cur}"))
    return out


@dataclass
class Borrowed:
    """The outcome of one spec reading numbers out of another spec's entry."""
    ok: bool
    refusal: str
    values: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def borrow_metrics(source_id: str, keys, ledger: Optional[Ledger] = None) -> Borrowed:
    """Read another spec's measured constants — or refuse, with a reason.

    Reading a calibration live from the ledger is the RIGHT instinct: T0.14's
    scar is a constant pasted into a second file and drifting from the
    measurement that produced it. But live is not the same as current. XL.00
    did this and gated on `status == PASS` and nothing else, so PS.01's entry
    would have kept supplying `j0`/`alpha` after `playground.py` changed the
    world those numbers describe — a certificate about a world that no longer
    exists, feeding a test that cannot tell. (Found by the overseer, 2026-08-10,
    RANK 2; the instance was benign and the guard was absent.)

    So this refuses on ANY reason `staleness_of` gives, not just on a status,
    and it returns the source's `impl_sha` so the borrower can record WHICH
    version of the source it computed on. A provenance that lives in a docstring
    is not provenance.

    UNVERIFIABLE refuses too. 44 entries predate `impl_sha` and a number
    borrowed from one of them cannot be shown to describe today's code — that
    is precisely the claim borrowing needs. Refusing is a VOID for the borrower
    (an uncalibrated test refutes nothing), which is cheap and honest; a re-run
    of the source clears it.

    Callers get `Borrowed.ok == False` and a human-readable `refusal`; the
    convention is to record `provenance` in the metrics either way and return
    `Status.VOID`, never FAIL.
    """
    led = ledger or Ledger()
    entry = led.results.get(source_id)
    prov: Dict[str, Any] = {"borrowed_from": source_id}
    if entry is None:
        return Borrowed(False, f"{source_id} has no ledger entry", {}, prov)
    prov["borrowed_status"] = entry.status.value if hasattr(entry.status, "value") else str(entry.status)
    prov["borrowed_impl_sha"] = str(getattr(entry, "impl_sha", None))
    prov["borrowed_commit"] = str(getattr(entry, "commit", "") or "")
    prov["borrowed_ran_at"] = str(getattr(entry, "ran_at", "") or "")
    if entry.status != Status.PASS:
        return Borrowed(False, f"{source_id} is {prov['borrowed_status']}, not PASS", {}, prov)
    path = module_path_for(source_id)
    if path is None:
        return Borrowed(False, f"{source_id} has no single implementation file to hash",
                        {}, prov)
    stale = staleness_of(entry, path)
    if stale:
        why = "; ".join(f"{k}: {d}" for k, d in stale)
        return Borrowed(False, f"{source_id} is stale — {why}", {}, prov)
    values: Dict[str, float] = {}
    for k in keys:
        v = entry.metrics.get(k)
        if v is None:
            return Borrowed(False, f"{source_id} recorded no metric {k!r}", {}, prov)
        try:
            values[k] = float(v)
        except (TypeError, ValueError):
            return Borrowed(False, f"{source_id}.{k} is not a number: {v!r}", {}, prov)
    return Borrowed(True, "", values, prov)


def _impl_sha(fn: Callable) -> Optional[str]:
    """`impl_sha_of` for the file a function is defined in.

    Also the one moment when both the static and the runtime view of
    `IMPL_DEPS` are available, so it checks that they agree. If the AST reader
    ever stops seeing a declaration the running module has, the divergence
    raises HERE — at write time, on the run that would have recorded the wrong
    sha — instead of being discovered later as an unclearable stale flag.
    """
    import inspect
    import sys
    try:
        src = inspect.getsourcefile(fn)
        if not src:
            return None
        static, problem = impl_deps_of(src)
        mod = sys.modules.get(getattr(fn, "__module__", ""), None)
        runtime = tuple(getattr(mod, "IMPL_DEPS", ()) or ())
        if not problem and static != runtime:
            raise RuntimeError(
                f"IMPL_DEPS disagree for {Path(src).name}: source says "
                f"{list(static)}, the imported module says {list(runtime)}. "
                "The recorded sha and the staleness checker would diverge.")
        return impl_sha_of(src)
    except (OSError, TypeError):
        return None


def run_spec(spec: Spec, fn: Callable[[int], Dict[str, Any]],
             check: Callable[[Dict[str, Any], Dict[str, Any]], bool],
             control_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
             ledger: Optional[Ledger] = None) -> Result:
    """Execute one experiment and record it.

    `fn(seed) -> metrics` is the experiment. `control_fn(seed) -> metrics` is the
    condition that must NOT pass. `check(metrics, control_metrics) -> bool` is the
    pre-registered threshold — it is supplied by the spec author before the run,
    never adjusted afterwards to make a result look better.
    """
    # Before anything is spent: a control that runs must also be declared.
    # Raised rather than warned, and raised BEFORE the experiment, because a
    # warning at the end of a 20,000-second run is a warning nobody reads.
    if control_fn is not None and not spec.control:
        raise UndeclaredControl(
            f"{spec.id} passes control_fn={control_fn.__name__} to run_spec but "
            f"Spec.control is None. Declare in the registry WHAT the control is "
            f"and WHICH WAY it must fail; that field is the audit surface.")

    ledger = ledger or Ledger()
    impl_sha = _impl_sha(fn)
    blocked = ledger.blocked_by(spec)
    if blocked:
        res = Result(spec_id=spec.id, status=Status.BLOCKED,
                     message=f"dependencies not passing: {', '.join(blocked)}",
                     **Result.env_stamp(), ran_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        ledger.record(res)
        return res

    t0 = time.time()
    # STAMPED BEFORE THE RUN, NOT AFTER IT. `env_stamp()` reads `HEAD`, and
    # HEAD moves: this used to be read at Result construction, i.e. whenever
    # the run happened to finish, so every entry named the commit that was
    # checked out at the END. `gpu.py:assert_ref_is_current` refuses to build a
    # job from a HEAD GitHub does not have, on the stated principle that a
    # result is only attributable to a commit if the commit is what ran — and
    # the record then discarded it. OVERSIGHT.md 1.2 ranked this #1 against
    # T2.01's 5.58 GPU-hours, and it is NOT a GPU-only defect: on 2026-08-10 a
    # 14-minute CPU run of PS.01 was stamped `248b160`, three commits after the
    # `ad55a31` that ran it, because a CONCURRENT builder iteration committed
    # while it was mid-flight. Any run longer than the interval between commits
    # is exposed, the error is silent and plausible, and it grows with
    # duration. One line moved fixes the whole class.
    stamp = Result.env_stamp()
    seeds = list(range(spec.seeds))
    try:
        runs = [fn(s) for s in seeds]
        metrics = _aggregate(runs)
        control_metrics = _aggregate([control_fn(s) for s in seeds]) if control_fn else {}
        ok = check(metrics, control_metrics)
        # `check` may return a Status directly to signal VOID — a run that
        # could not test the claim at all (an arm that never learned, a leaky
        # fixture). Bare bools keep their old meaning, so no existing test
        # changes behaviour.
        if isinstance(ok, Status):
            status = ok
            message = {Status.PASS: "",
                       Status.FAIL: "pre-registered threshold not met",
                       Status.VOID: "run did not test the claim; not a refutation"
                       }.get(status, "")
        else:
            # A `_check` that writes "VOID" into its own metrics and then
            # returns a bare False is a metrics/status disagreement: run_spec
            # would record FAIL "pre-registered threshold not met", firing the
            # spec's `kills` field off a run that refused to arbitrate. T2.02
            # did exactly this and the ledger needed hand-repair. Refuse to
            # record it — an ERROR is loud; a wrong FAIL is not.
            if not ok:
                _voided = _declares_void(metrics) or _declares_void(control_metrics)
                if _voided:
                    raise VoidStatusMismatch(
                        f"{spec.id}: _check returned a bare False but its metrics "
                        f"declare VOID ({_voided!r}). Return Status.VOID instead — "
                        "FAIL means the hypothesis was tested and lost.")
            status = Status.PASS if ok else Status.FAIL
            message = "" if ok else "pre-registered threshold not met"
    except Exception as e:
        metrics, control_metrics = {}, {}
        status = Status.ERROR
        message = f"{type(e).__name__}: {e}"[:400]

    res = Result(spec_id=spec.id, status=status, metrics=metrics,
                 control_metrics=control_metrics, seeds=seeds,
                 duration_s=round(time.time() - t0, 2), message=message,
                 impl_sha=impl_sha,
                 ran_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **stamp)
    ledger.record(res)
    return res


def _round6(x: float) -> float:
    """Six decimals for values of ordinary size, six SIGNIFICANT figures below
    1.0. A nonzero metric may never be recorded as exactly 0.0.

    `round(x, 6)` did the rounding here until 2026-08-09, and `check()` reads
    the AGGREGATED metrics — so every pre-registered threshold below ~5e-7 was
    unenforceable by construction: a genuine drift of 3e-7 recorded as 0.0 and
    satisfied `drift <= 0.0`. That is T0.14's bit-identity gate, the one that
    closed the most expensive bug in the project, plus T0.02, T1.10, T1.11 and
    both checkpoint round-trips. They survived only because `_aggregate`
    short-circuits at a single run and all of them are seeds=1 — the moment any
    is re-verified at 3 seeds, as GOAL.md asks for, its tightest check goes
    quietly dead. PG.8 is where it first showed: two 1e-9 gates recorded 0.0.

    LESSONS.md: an assertion made against a saturated quantity cannot fail.
    Here the saturation was manufactured by the recorder, downstream of every
    test, which is why no spec could see it — including T0.13, which perturbs
    the recorded value and finds a live gate either way.
    """
    if x == 0.0 or not math.isfinite(x):
        return float(x)
    return round(x, 6) if abs(x) >= 1.0 else float(f"{x:.6g}")


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean and std across seeds. Std is reported always, not only when it is
    flattering — an effect smaller than its seed noise is not an effect."""
    if not runs:
        return {}
    if len(runs) == 1:
        return dict(runs[0])
    out: Dict[str, Any] = {}
    for k in runs[0]:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        if len(vals) == len(runs):
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            out[k] = _round6(mean)
            out[f"{k}_std"] = _round6(var ** 0.5)
        else:
            out[k] = runs[0][k]
    return out
