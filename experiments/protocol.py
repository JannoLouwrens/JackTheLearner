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
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LEDGER_PATH = Path(__file__).parent / "ledger.json"

#: Files the RUNNER writes as it runs. A dirty working tree normally means "the
#: code that ran is in no commit", but these are outputs, not code, so their
#: being dirty says nothing about the code — see the `+dirty` stamp below.
#:
#: `gpu_budget.json` joined on 2026-08-12, and its absence was a live defect
#: rather than an omission: `Budget.charge()` writes it at the end of EVERY GPU
#: job, so from the moment a job charged until the next commit, every CPU spec
#: run in that window stamped `+dirty` and BLOCKED its dependents — the same
#: "evidence log that invalidates the evidence" failure `gpu_submissions.jsonl`
#: was added to close the day before, in the sibling file, missed. `gpu.py`
#: already knew (it had excluded the file since the guard deadlocked against
#: itself); this list did not, which is why they are now ONE list.
RUNNER_OUTPUTS = ("experiments/ledger.json",
                  "experiments/gpu_submissions.jsonl",
                  "experiments/gpu_budget.json",
                  # Budget.charge's atomic-write staging file (2026-08-12). It
                  # exists only between write_text and os.replace, but a writer
                  # SIGKILLed in that window orphans it, and an orphan that is
                  # not in this set reads as uncommitted code — the exact
                  # T2.00 `+dirty` failure, from a file the meter itself wrote.
                  "experiments/gpu_budget.json.tmp")

#: Files the LOOP writes around a run — rendered status and the journal. Also
#: not code, and the evidence is a stamp this project already paid for: T2.00's
#: `08444b2+dirty` was caused by `docs/LOOP_JOURNAL.md`, nothing else. Commit
#: `ae9693f`, which cleaned that tree, touches THREE files and not one of them
#: is code. The stamp said "the code that ran is in no commit" about a run whose
#: code was fully committed, `blocked_by` propagated it to 47 specs, and it cost
#: a 998-second re-run to clear.
#:
#: This is not incidental — it is guaranteed by the loop's own instructions.
#: Every iteration is told to finish by appending to `LOOP_JOURNAL.md` and
#: re-rendering `CHECKLIST.md`, and the hourly builder overlaps runs that last
#: hours. So a doc edit is uncommitted at the moment some OTHER spec records,
#: routinely. The next occurrence was T2.01, 6.5 Kaggle-hours, recording while
#: this was written.
#:
#: `CHECKLIST.md` is generated FROM the ledger (`run.render`) and
#: `LOOP_JOURNAL.md` is read by no code at all: both are write-only outputs, so
#: excluding them cannot mask a real change to anything executable. The genuine
#: positive is preserved and was checked, not assumed: T0.25's `1ddcd27+dirty`
#: came from an uncommitted `TrainingPipeline.py`, and still stamps.
DOC_OUTPUTS = ("CHECKLIST.md", "docs/LOOP_JOURNAL.md")

#: The whole answer to "does this uncommitted file mean CODE moved". One list,
#: because the two organs that ask it — the `+dirty` stamp and the GPU push
#: guard — were two lists, and they diverged by exactly one entry.
NOT_CODE = RUNNER_OUTPUTS + DOC_OUTPUTS


def porcelain_path(porcelain_line: str) -> str:
    """The path out of a `git status --porcelain` line, stripped or not.

    Split, never a column slice. `subprocess.run(...).stdout.strip()` eats the
    leading space of the FIRST line only (`' M path'` -> `'M path'`), so a
    `line[3:]` slice silently yields `'periments/gpu_budget.json'` for whichever
    file git happened to list first — which is exactly how `gpu.py`'s exclusion
    kept missing its own budget file after being "fixed" twice. A parser that is
    right for one caller's whitespace and wrong for another's is not a parser.
    """
    parts = porcelain_line.strip().split(None, 1)
    return parts[1].strip() if len(parts) == 2 else ""


def is_code_dirt(porcelain_line: str) -> bool:
    """Does this `git status --porcelain` line mean CODE is uncommitted?

    Pulled out of the `+dirty` stamp so the question can be asked of a fixture
    string instead of the real working tree — a predicate that can only be
    exercised by dirtying the repo it audits is a predicate nothing will ever
    test. T0.22 P13 is the test; the pre-2026-08-11 version (`ledger.json`
    alone) is its control.
    """
    path = porcelain_path(porcelain_line)
    if not path:
        return False
    # Exact repo-relative match, never `endswith`: a suffix match would grant
    # the exclusion to any `*ledger.json` anywhere in the tree (overseer B4,
    # 10th/11th audits). Porcelain paths are repo-relative, so the entries are.
    return path not in NOT_CODE


def code_dirt(porcelain_lines) -> list:
    """Every uncommitted CODE path in a `git status --porcelain` listing."""
    return [porcelain_path(ln) for ln in porcelain_lines if is_code_dirt(ln)]


def working_tree_porcelain(root=None) -> list:
    """`git status --porcelain` for the repo, as a list of lines.

    A named function rather than an inline `subprocess.run` so the question
    "what does the tree look like right now" can be answered by a stub in a
    test. The `+dirty` stamp in `run_spec` asks it inline and is therefore
    only testable through `is_code_dirt`; `gate_precondition` below is the
    first caller that needed the whole answer.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(["git", "status", "--porcelain"],
                              capture_output=True, text=True,
                              cwd=root, timeout=10).stdout.splitlines()
    except Exception:
        return []


#: The flag that lets a builder gate a work-in-progress tree on purpose.
GATE_DIRTY_FLAG = "--dirty-ok"


def gate_precondition(porcelain_lines, at_risk: int = 0,
                      dirty_ok: bool = False) -> str:
    """Why a REGRESSION GATE must not start right now — or "" if it may.

    THE SCAR, 2026-08-30 09:19 UTC, commit `7966524`. The builder changed
    `experiments/champions.py`, then ran `run --gate` to check for regressions,
    then committed. Correct instinct, fatal order: every one of the ten specs
    the gate re-ran recorded `e9bd4a0+dirty`, and for `T0.08` and `T0.09` the
    recorded `impl_sha` reconstructed from no committed blob, so both PASSes
    became DIRTY STAMPS. `T0.09` is a dependency of 36 specs. `run blocked`
    spent the next three hours reporting a phantom — *"T0.09 = PASS but STALE,
    frees 36"* — at the top of the project, above the real blocker (`T2.01`,
    frees 35), and clearing it cost two re-runs and a second Colab T4
    round-trip. The gate had been GREEN. Ten tests passed and the ladder got
    worse.

    That is the shape worth naming: a single-spec run from a dirty tree only
    fails to certify, which is honest and normal — `t0_23`'s own fixture note
    says a dirty tree is the ordinary state of the iteration that runs it. A
    GATE run from a dirty tree does something else. It re-runs specs that
    already hold CLEAN stamps and overwrites them with dirty ones, and
    `blocked_by` then propagates the demotion to every dependent. Its expected
    information gain is negative by construction, and it is the one command in
    this runner that can only lose certificates. The same event is already on
    record twice under other names (`T2.00`'s `08444b2+dirty`, which cost a
    998-second re-run and blocked 47 specs, and `T0.25`'s `1ddcd27+dirty`);
    both were repaired as incidents, neither as a class.

    So the gate refuses a code-dirty tree and says which paths and how many
    certificates were at risk. `--dirty-ok` runs it anyway, because "does my
    uncommitted change break anything" is a real question — it is an explicit
    opt-in that changes nothing about what gets recorded. NOTHING here weakens
    a stamp: `+dirty` still fires exactly as before, on exactly the same
    condition. This guard only refuses to VOLUNTEER for it.

    `at_risk` is the number of clean certificates the gate is about to re-run;
    it is reported, never gated on, so the refusal is informative rather than
    merely obstructive.
    """
    dirty = code_dirt(porcelain_lines)
    if not dirty or dirty_ok:
        return ""
    shown = ", ".join(dirty[:6]) + (" …" if len(dirty) > 6 else "")
    return (f"Refusing to gate: {len(dirty)} uncommitted code file(s) — {shown}. "
            f"A gate run from a modified tree stamps every spec it re-runs "
            f"`+dirty`, demoting clean certificates and blocking their "
            f"dependents ({at_risk} PASS row(s) at risk; this cost T0.09 and "
            f"36 dependents on 2026-08-30). Commit first, then gate — or pass "
            f"{GATE_DIRTY_FLAG} to gate the working tree on purpose.")


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

    A VOID spec is not demonstrated and must not be counted as PASS, and it
    does not trigger `kills`. It DOES block its dependents — exactly as
    NOT_RUN does, because the two are the same epistemic state: no verdict on
    the hypothesis. It means: fix the run and try again. (D2, resolved
    2026-08-13 by replaying the ledger's own history: at 2026-08-10T01:00,
    "VOID does not block" would have admitted 11 specs, 9 of them resting on
    T2.01's VOID — and T2.01's very next measurement, 17 minutes later, was
    FAIL. Every one of those results would have rested on a refuted
    foundation. Blocking cost nothing that a re-run does not recover; the
    full working is in docs/DECISIONS_RESOLVED.md. An earlier version of this
    docstring said VOID "does not BLOCK its dependents" while `unsatisfied`
    blocked on it — the contradiction sat shipped for four days as open
    decision D2.)
    """


class Budget(str, Enum):
    """Where a test runs. Ordering matters: the ladder front-loads CPU work so a
    hypothesis dies before it costs GPU quota."""
    CPU_FAST = "cpu<1min"
    CPU = "cpu<10min"
    CPU_LONG = "cpu<2h"
    CPU_DAYS = "cpu<48h"     # detached multi-iteration runs (LC.03's envelope
    # measured ~90 core-h against CPU_LONG's label; a declaration machinery
    # reads is code, and run.py kills a child at the declared budget's timeout
    # — the T2.08 routing lesson and the T2.01 timeout scar, one tier up)
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
    """Wall clock of the RECORDING CALL, not the cost of the work: a harvested
    or reattached run records in seconds what a kernel computed for hours
    (LC.03: 0.02 s for ~45 GPU-hours). For remote work the cost is
    `compute_s`. (Overseer 17th-audit B3.)"""
    compute_s: Optional[float] = None
    """Provider-metered seconds behind this record, summed over every remote
    job the runs paid for (failed attempts included — they spent quota too).
    None, never 0.0, when nothing remote ran: a CPU-only run has no remote
    cost rather than a zero one (the `Arm.cost` lesson)."""
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
    gpu_job_id: Optional[str] = None
    """Comma-joined remote job ids this run dispatched (overseer B3).

    Folded in by `run_spec` from `gpu.drain_job_ids()` — no spec has to
    remember to record it (only T1.02 ever did, by hand, into its metrics).
    `None` means no remote dispatch happened during the run — deliberately not
    `""`, because a sentinel that is also a valid value cannot be detected
    (the `Arm.cost` lesson). This is the one field the ledger shares with
    `gpu_submissions.jsonl`, so "which hours bought which result" is a join,
    not timestamp arithmetic.
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
    spec_sha: Optional[str] = None
    """sha256 of the CLAIM this result was recorded against, at run time.

    `impl_sha`'s sibling and the answer to the other half of "is this row still
    true": the code can hold still while the WORDS move. `LC.01`'s
    `falsified_by` was amended by owner ruling on 2026-08-24 with the note that
    a re-run was owed; the row kept reading PASS at `ran_at 2026-08-09` and no
    instrument could see it, because the ledger hashed the test file and never
    the spec. See `spec_sha_of` for what is covered and `spec_drift` for how a
    row is read.

    `None` means the entry predates this field: UNKNOWN, NOT CLEAN. Deliberately
    not `""` and deliberately not back-filled — the sha of today's words proves
    nothing about a run from before them (the `Arm.cost` lesson again).
    """

    preserved_impl: Optional[str] = None
    """Git ref holding the exact bytes an adverse verdict ran, when the tree
    was dirty. Written by `run_spec` via `preserve_impl_bytes` — see there for
    what the ref proves and for the 2026-08-29 loss that forced it.

    `None` on every clean row and on every PASS, where the commit already names
    the code. On a `+dirty` FAIL/VOID, `None` means preservation was ATTEMPTED
    AND FAILED and the reason is in `message`; it never means "not tried",
    because the row is only reachable through the attempt (the `Arm.cost`
    lesson: a sentinel that is also a valid value cannot be detected).
    """

    unknown_keys = ()
    """Row keys this version's dataclass does not define, set by `from_row`.

    Deliberately NOT a dataclass field: it describes how a row was READ, not
    what a run measured, and `asdict` must never write it back into the ledger.
    """

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Result":
        """Build a Result from a ledger row, IGNORING keys this version lacks.

        THE LEDGER ROW IS A CROSS-PROCESS CONTRACT, and `Result(**row)` made it a
        contract that cannot be changed while anything is running. Every recorder
        re-reads the whole file under the lock and rebuilds EVERY row in memory
        (`Ledger.record`, `Ledger.load`), so one process writing a field a second
        process's dataclass does not define raises `TypeError: unexpected keyword
        argument` inside the second process's `record()` — after its run, before
        its result reaches disk.

        Caught 2026-08-12 as a near-miss, one edit before it fired: the next unit
        of work adds a `deps_sha` field, and at that moment `T2.01` (a 6.5-hour
        Kaggle job, PID 2160973, started 07:24) and `T1.08` were both mid-poll
        holding the PREVIOUS class. Adding the field would have cost both runs at
        the instant they tried to record — a schema change destroying finished
        science it never touched, in a file whose whole purpose is to be the only
        durable record. `run_isolated` would have reported "child recorded
        nothing", so the loss would have read as a crashed test.

        A field addition is therefore only safe if OLD readers tolerate it, which
        must be true BEFORE the field exists — a rolling migration, not a flag
        day. Unknown keys are kept on disk regardless (`record` merges the file
        and only ever replaces the one row it is writing), so tolerating them
        here loses nothing: a future version that defines the field reads its
        value back.

        Dropped keys are NOT silent — they are recorded on the instance as
        `unknown_keys` so a hand-edited typo (`impl_shaa`) is discoverable rather
        than swallowed. They are not dataclass fields, so `asdict` cannot write
        them back out.
        """
        names = {f.name for f in fields(cls)}
        known = {k: v for k, v in row.items() if k in names}
        unknown = tuple(sorted(k for k in row if k not in names))
        obj = cls(**known)
        obj.unknown_keys = unknown
        return obj

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
        # `RUNNER_OUTPUTS` are excluded on purpose: they are the runner's own
        # output, not code, and they are legitimately dirty whenever a previous
        # result is waiting to be committed. `gpu_submissions.jsonl` joined the
        # set on 2026-08-11, the day it was first committed: it is APPEND-ONLY
        # and written by `gpu.submit()` itself, so without this every spec run
        # after any GPU dispatch would stamp `+dirty`, read as DIRTY in
        # `run stale`, and BLOCK its dependents — an evidence log that
        # invalidates the evidence. The rule the set encodes: a file the runner
        # writes cannot be a file the runner audits itself against.
        try:
            porcelain = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=root, timeout=10,
            ).stdout.splitlines()
            dirty = [ln for ln in porcelain if is_code_dirt(ln)]
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
            # from_row, not Result(**r): a row written by a NEWER version must
            # still load here. See Result.from_row — the ledger row is a
            # cross-process contract.
            self.results[rid] = Result.from_row(r)

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
                    # the specs are honestly risky or written to pass.
                    # History carries the EVIDENCE, not just the verdict
                    # (overseer B1, 2026-08-13): metrics, control_metrics,
                    # impl_sha and seeds ride along, because a threshold moved
                    # after a FAIL can only be audited against the failing
                    # measurement, and a verdict-only history entry made every
                    # amend-after-FAIL auditable by nobody but its author. The
                    # 163 entries written before this date stay evidence-free;
                    # back-filling them would invent numbers nobody recorded.
                    prev = on_disk.get(rid)
                    hist = list(prev.get("history", [])) if prev else []
                    if prev and prev.get("ran_at") != r.ran_at:
                        row_h = {k: prev.get(k) for k in
                                 ("status", "ran_at", "commit", "message",
                                  "metrics", "control_metrics", "impl_sha",
                                  # `spec_sha` rides along for the same reason
                                  # `impl_sha` does: auditing a superseded
                                  # verdict needs the WORDS it was bought
                                  # under as much as the code (46th audit B1).
                                  "spec_sha",
                                  "seeds", "gpu_job_id")
                                 if k in prev}
                        if prev.get("amended"):
                            # An amendment is part of what that verdict WAS.
                            # Dropping it here would let a re-run launder a
                            # hand-set status back into an unqualified record.
                            row_h["amended"] = prev["amended"]
                        for sk in ("supersedes_fail", "supersedes_void"):
                            if prev.get(sk):
                                # Same reason as `amended`: the pairing with
                                # the verdict it amended is part of what that
                                # verdict WAS.
                                row_h[sk] = prev[sk]
                        hist.append(row_h)
                    row = {**asdict(r), "status": r.status.value}
                    row["history"] = hist[-20:]
                    # A verdict that supersedes a FAIL carries the failing
                    # evidence IN the record, not only in history (overseer
                    # B2, 2026-08-13): the failing commit, whether that commit
                    # was dirty (a `+dirty` FAIL later amended is unauditable
                    # by construction — the failing code exists nowhere), the
                    # failing impl_sha and measurement. `impl_changed` is the
                    # machine-readable "the code moved between the FAIL and
                    # this verdict"; None when either side predates impl_sha,
                    # because unknowable must never read as false (Arm.cost).
                    # The old and new thresholds themselves are recovered by
                    # `git diff <fail commit> <this commit> -- <test file>`,
                    # which is exactly why the fail commit must be real and
                    # clean — `audit_supersedes_fail` enforces that.
                    # Deliberately NOT a Result field: old readers tolerate
                    # unknown row keys (see Result.from_row), so this is the
                    # rolling-migration-safe shape.
                    # Pair against the previous row CARRYING A REAL VERDICT,
                    # walking back through ERROR rows: an ERROR is an
                    # infrastructure event, not a verdict on the hypothesis,
                    # and a dead kernel between a FAIL and its amended re-run
                    # severed this pairing — worse, the dead kernel records
                    # the UNCHANGED impl_sha, so the pair read as "same code
                    # re-run" (overseer 22nd audit B2; T2.05's live chain
                    # VOID -> ERROR -> ERROR -> FAIL). VOID joins FAIL as a
                    # paired source (B1, same audit): SYSTEM.md's "fix the
                    # arm, do not decide" makes VOID the verdict that
                    # doctrinally precedes a redesign, and it was the one
                    # lane with no artifact — three honest VOID->verdict
                    # transitions were recoverable only from commit messages.
                    # COVERAGE (LESSONS rule — say it): FAIL pairs as
                    # `supersedes_fail`, VOID as `supersedes_void`, each with
                    # a `status` key; ERROR rows are skipped in the walk;
                    # the walk STOPS at PASS/SKIP/BLOCKED/NOT_RUN (a re-run
                    # on top of those amends no adverse verdict).
                    if prev and prev.get("ran_at") != r.ran_at:
                        chain = [prev] + list(reversed(
                            prev.get("history") or []))
                        src = next((e for e in chain
                                    if e.get("status") != Status.ERROR.value),
                                   None)
                        if src and src.get("status") in (Status.FAIL.value,
                                                         Status.VOID.value):
                            both = bool(src.get("impl_sha")) and bool(r.impl_sha)
                            key = ("supersedes_fail"
                                   if src["status"] == Status.FAIL.value
                                   else "supersedes_void")
                            row[key] = {
                                "status": src["status"],
                                "commit": src.get("commit"),
                                "dirty": str(src.get("commit") or ""
                                             ).endswith("+dirty"),
                                "impl_sha": src.get("impl_sha"),
                                "impl_changed": (src.get("impl_sha") != r.impl_sha
                                                 if both else None),
                                # The other way a verdict can be superseded
                                # without the code moving: the CLAIM moved.
                                # Same tri-state discipline as `impl_changed`
                                # — None when either side predates `spec_sha`,
                                # because unknowable must never read as false.
                                "spec_sha": src.get("spec_sha"),
                                "spec_changed": (
                                    src.get("spec_sha") != r.spec_sha
                                    if (src.get("spec_sha") and r.spec_sha)
                                    else None),
                                "metrics": src.get("metrics"),
                                "ran_at": src.get("ran_at"),
                            }
                    if r.status is Status.FAIL and \
                            str(row.get("commit") or "").endswith("+dirty"):
                        # The second sentence changes with `preserved_impl`
                        # and the first does not: a preserved ref keeps the
                        # BYTES, it does not make the row auditable — no
                        # pre-registered gate reads the ref, by decision
                        # (see `preserve_impl_bytes`). Saying otherwise here
                        # would be this warning quietly clearing itself.
                        print("  ! FAIL recorded from a MODIFIED tree. If this "
                              "FAIL is later amended (threshold moved, code "
                              "changed), the failing implementation exists "
                              "nowhere and the amendment is unauditable by "
                              "construction (overseer B2, T2.08's 75a1938+dirty"
                              "). Commit the failing implementation before "
                              "re-running.")
                        if row.get("preserved_impl"):
                            print(f"    the bytes themselves are kept at "
                                  f"{row['preserved_impl']} (git cat-file -p) "
                                  "— an artifact, not an audit lane; the "
                                  "commit is still owed.")
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
                    # THIS is the line that would have killed a 6.5-hour GPU run
                    # the first time a concurrent process wrote a field this
                    # version does not define. See Result.from_row.
                    fresh[rid] = Result.from_row(d)
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
              unknown_history: bool = False,
              fix_hardware: bool = False,
              doc_only: bool = False) -> Dict[str, Any]:
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

        `fix_hardware` (17th-audit B2): nine pre-fix GPU records stamped the
        DISPATCHER (`aarch64/.../cpu`) as `hardware` while the machine that ran
        the work sat in `metrics["gpu"]`. The correction is derivable from data
        already in the row, so unlike `impl_sha` it may be back-filled — but
        only DERIVED, never supplied: this path reconciles `hardware` with the
        row's own `metrics["gpu"]` in the same format `run_spec` now stamps,
        and refuses when there is no gpu recorded or the stamp is already
        remote. Status, metrics and seeds are untouched — a provenance
        amendment, not a re-verdict.

        `doc_only` (25th-audit B3): re-stamps `impl_sha` when — and only when
        — the drift from the recorded sha is PROVABLY prose. The claim a
        ledger row makes is about executable behaviour; a docstring or
        comment edit changes the file's bytes without changing what the run
        measured, and before this lane existed such an edit turned a PASS
        amber forever unless re-bought with a full re-run — so documentation
        was being routed AWAY from the files it describes (UB.9 and T2.06
        both went STALE from writing an audit answer where it belongs, and
        seven more answers were diverted to LESSONS.md to dodge the alarm).
        The proof obligation is total, never argued: the exact code the entry
        certifies is recovered from git by `blob_reconstructing_sha`
        (historical file, current dependency bytes), falling back to
        `tree_reconstructing_sha` (file AND deps at one committed tree
        state) when a declared dependency has itself drifted — in which case
        `prose_only_delta` must clear the file's delta AND every drifted
        dependency's delta. Anything that moves ANY of those ASTs —
        a threshold, a gate, an `IMPL_DEPS` line, a dependency's constant —
        is refused loudly and stays stale. Status, metrics and seeds are
        untouched; the old sha, the new sha, the reconstruction commit and
        the proven-drifted dep list land in `amended`.
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
        if status is None and not unknown_history and not fix_hardware \
                and not doc_only:
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
                if fix_hardware:
                    gpu = (row.get("metrics") or {}).get("gpu")
                    if not (isinstance(gpu, str) and gpu.strip()):
                        raise ValueError(
                            f"{spec_id}: fix_hardware needs metrics['gpu'] in the "
                            "row itself — the correction must be derived, never "
                            "supplied")
                    old_hw = row.get("hardware", "")
                    if old_hw.startswith("remote/"):
                        raise ValueError(
                            f"{spec_id}: hardware already names the remote machine")
                    new_hw = f"remote/{gpu} (dispatched from {old_hw})"
                    changes.append({"field": "hardware", "from": old_hw,
                                    "to": new_hw})
                    row["hardware"] = new_hw
                if doc_only:
                    recorded = row.get("impl_sha")
                    if not recorded:
                        raise ValueError(
                            f"{spec_id}: no impl_sha recorded — the doc-only "
                            "lane re-stamps a hash and this entry never "
                            "carried one; a re-run is the only upgrade")
                    path = module_path_for(spec_id)
                    if path is None:
                        raise ValueError(
                            f"{spec_id}: no implementation file to compare")
                    cur = impl_sha_of(path)
                    if cur == recorded:
                        raise ValueError(
                            f"{spec_id}: impl_sha is current ({cur}); "
                            "nothing is stale, nothing to amend")
                    old_bytes, old_commit, problem = \
                        blob_reconstructing_sha(path, recorded)
                    drifted: Dict[str, Any] = {}
                    if problem:
                        # The dep lane (2026-08-24): a prose-only edit to a
                        # file OTHER specs declare in IMPL_DEPS strands THEIR
                        # certificates — reconstruct the whole tree state and
                        # owe the same proof for every drifted dependency.
                        old_bytes, old_commit, drifted, tree_problem = \
                            tree_reconstructing_sha(path, recorded)
                        if tree_problem:
                            raise ValueError(
                                f"{spec_id}: cannot recover the code this "
                                f"entry certifies — {problem}; tree search: "
                                f"{tree_problem}")
                    delta = prose_only_delta(old_bytes, Path(path).read_bytes())
                    if delta:
                        raise ValueError(
                            f"{spec_id}: the edit since {recorded} is not "
                            f"prose-only ({delta}); a moved AST stays stale — "
                            "re-run the spec instead")
                    root = Path(__file__).resolve().parent.parent
                    for d, db in sorted(drifted.items()):
                        if db is None:
                            raise ValueError(
                                f"{spec_id}: dependency {d} appeared or "
                                f"vanished since {recorded}; existence is "
                                "never prose — re-run the spec instead")
                        ddelta = prose_only_delta(db, (root / d).read_bytes())
                        if ddelta:
                            raise ValueError(
                                f"{spec_id}: the drift in dependency {d} "
                                f"since {recorded} is not prose-only "
                                f"({ddelta}); a moved dependency AST stays "
                                "stale — re-run the spec instead")
                    proof = (f"reconstructed {recorded} from git "
                             f"{old_commit[:12]}; docstring-stripped ASTs "
                             "identical (prose_only_delta)")
                    if drifted:
                        proof += ("; drifted IMPL_DEPS proven prose-only: "
                                  + ", ".join(sorted(drifted)))
                    changes.append({
                        "field": "impl_sha", "from": recorded, "to": cur,
                        "proof": proof})
                    row["impl_sha"] = cur

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

    def unsatisfied(self, spec: Spec) -> List[tuple]:
        """`(dep_id, why)` for every dependency that cannot support a run yet.

        THE ONE DEFINITION of "is this dependency satisfied?". `blocked_by`,
        `registry.ready` (so `run next`), `run_spec`'s refusal and
        `run._terminal_blockers` all reach the question through here; the walk
        in `_terminal_blockers` used to restate it as `status is Status.PASS`
        and disagreed with `borrow_metrics` about whether one row was usable
        (overseer 2026-08-10 RANK 2, and LESSONS' *"retiring a rule is a
        two-sided job"*). Two organs, each internally consistent, is exactly
        the shape that hides.

        A dependency must be a PASS **that still describes the code that
        exists now**, because `depends_on` is an edge between specs while
        staleness is a fact about a ledger ROW: `LC.03` reading "runnable" off
        a `PS.01` entry measured against a world that has since changed is a
        run on a foundation nobody checked.

        WHICH staleness blocks, and why it is not all of it. `staleness_of`
        reports kinds that are not the same evidence:

          DIRTY / CHANGED / UNSTAMPED_CHANGED
                            POSITIVE evidence the implementation moved after
                            the run. The dependency is blocked on a re-run.
                            (UNSTAMPED_CHANGED joined 2026-08-14, 15th audit
                            B1: an entry with no `impl_sha` whose file git
                            nonetheless shows changed since the run is the
                            same evidence as CHANGED, minus the stamp.)
          UNVERIFIABLE / UNSTAMPED_INTACT
                            ABSENT or benign evidence — the entry predates
                            `impl_sha` and either nothing can be compared or
                            git shows the file unchanged.

        `borrow_metrics` refuses on UNVERIFIABLE too, and is right to: it needs
        the number to describe today's code, and that is precisely the claim it
        cannot get. Dependency satisfaction asks something weaker — was this
        capability demonstrated — so the same answer is not automatic, and it
        was measured rather than argued: refusing DIRTY/CHANGED costs the two
        specs that are genuinely resting on stale rows, while also refusing
        UNVERIFIABLE takes the ladder from 29 runnable specs to 7 on the
        strength of 40 rows that are silent, not contradicted. Blocking the
        whole ladder on a backlog nobody is disputing would make `run next`
        useless and get the rule turned off, so UNVERIFIABLE passes here and
        is REPORTED (`run stale` lists all 40; a re-run clears each). If that
        backfill ever lands, tighten this to match `borrow_metrics` exactly.
        """
        out: List[tuple] = []
        for d in spec.depends_on:
            st = self.status(d)
            if st is not Status.PASS:
                # VOID blocks exactly like NOT_RUN (D2, resolved 2026-08-13),
                # but the reader must not mistake it for a refutation: the
                # asymmetry that matters is `kills`, which VOID suppresses.
                why = ("VOID — not demonstrated (the run could not test its "
                       "claim; a fixed re-run clears this, it is not a "
                       "refutation)") if st is Status.VOID else st.value
                out.append((d, why))
                continue
            path = module_path_for(d)
            if path is None:                 # no single impl file: nothing to hash
                continue
            blocking = [(k, det) for k, det in staleness_of(self.results[d], path)
                        if k in ("DIRTY", "CHANGED", "UNSTAMPED_CHANGED")]
            if blocking:
                out.append((d, "PASS but stale — " +
                            "; ".join(f"{k}: {det}" for k, det in blocking)))
        return out

    def blocked_by(self, spec: Spec) -> List[str]:
        """Dependencies that cannot support a run. A result computed on a broken
        foundation is worse than no result, because it looks like evidence."""
        return [d for d, _ in self.unsatisfied(spec)]

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


def impl_deps_of(path, source: Optional[bytes] = None) -> tuple:
    """A test module's `IMPL_DEPS`, read STATICALLY from its source.

    Static because the reader of a sha must not import the module to check it:
    `run.stale_claims` scans the whole ladder and importing every test would
    pull in mujoco, GL contexts and torch for a question about bytes on disk.

    `source`, when given, is read instead of the file at `path` — the doc-only
    amendment lane reconstructs shas for HISTORICAL versions of a test, and the
    declaration must come from the same bytes being hashed or the writer and
    the reader split into two code paths again (see `impl_sha_of`).

    Returns `(deps, problem)`. `problem` is non-empty when the declaration
    exists but cannot be read as a literal list of strings — reported rather
    than swallowed, because falling back to "no deps" is exactly the silent
    narrowing that produced the bug this function was extracted to fix.
    """
    import ast
    try:
        tree = ast.parse(source if source is not None
                         else Path(path).read_bytes())
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


def impl_sha_of(path, file_bytes: Optional[bytes] = None,
                dep_bytes: Optional[dict] = None) -> Optional[str]:
    """sha256 of a test file — the test as it was when it ran.

    `file_bytes`, when given, is hashed in place of the file at `path` (deps
    are still read from disk, i.e. as they stand NOW). This is the doc-only
    amendment lane asking "which historical version of this test does the
    recorded sha certify?" — and it deliberately reuses THIS function, because
    the last time two implementations computed "the same" hash they diverged
    silently (the story two paragraphs down).

    `dep_bytes`, when given, overrides the disk read for the dep paths it
    names (a value of None means "missing at that tree state", hashed with
    the same `missing:` rule as a missing file on disk). This is
    `tree_reconstructing_sha` asking the same question about a whole tree
    state, through the same one true code path.

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
        data = Path(path).read_bytes() if file_bytes is None else file_bytes
        h = hashlib.sha256(data)
        deps, problem = impl_deps_of(path, source=data)
        if problem:
            h.update(f"undeclarable:{problem}".encode())
        for rel in deps:
            if dep_bytes is not None and rel in dep_bytes:
                b = dep_bytes[rel]
                h.update(b if b is not None else f"missing:{rel}".encode())
                continue
            dep = Path(__file__).resolve().parent.parent / rel
            h.update(dep.read_bytes() if dep.is_file()
                     else f"missing:{rel}".encode())
        return h.hexdigest()[:16]
    except (OSError, TypeError):
        return None


#: The Spec fields a PASS is a claim ABOUT. Hashed into `spec_sha`; everything
#: not listed here may be edited without re-buying a certificate.
#:
#: The line is drawn at "would a reader of the row have understood the verdict
#: differently". `hypothesis`/`falsified_by`/`null_baseline`/`metric` ARE the
#: claim. `control` is the sabotage that must fail, `budget`/`seeds` are the
#: envelope the claim was bought at, `depends_on` is what it rests on.
#:
#: THREE ADDITIONS BEYOND THE 46th AUDIT'S B1 LIST, each because this file
#: already calls the field a pre-registration somewhere else:
#:   `gate_mode`         — Spec's own docstring: "the mode is a pre-registration,
#:                         and a caller that could pass it as an argument could
#:                         change it after seeing a VOID". A pre-registration
#:                         that can be edited after the verdict, invisibly, is
#:                         the exact defect this field exists to close.
#:   `screen_rationale`  — required when `gate_mode == "screen"` and recorded
#:                         verbatim beside the verdict it permitted.
#:   `kills`             — "forces the cost of failure to be decided before the
#:                         result is known". Narrowing it after a FAIL retires
#:                         nothing while reading as though it had.
#: Widening the list is STRENGTHENING and is allowed (the T1.02 precedent), but
#: it invalidates every stamp bought under the old list at once — a mass re-buy.
#: Pay that deliberately, in its own commit, and say so.
#:
#: NOT hashed, deliberately: `id`/`tier` are identity and placement, `title` is
#: a label, and `notes` carries the `COVERS:`/`PARKED:` markers that move for
#: honest bookkeeping reasons several times a week. Hashing `notes` would make
#: this alarm fire so often that the loop would learn to ignore it — the
#: `impl_sha_of` "mass false alarm" lesson, one field over.
SPEC_CLAIM_FIELDS = ("hypothesis", "falsified_by", "null_baseline", "metric",
                     "control", "kills", "budget", "seeds", "depends_on",
                     "gate_mode", "screen_rationale")


def spec_sha_of(spec: "Spec") -> Optional[str]:
    """sha256 of the CLAIM a spec makes — the words the verdict was bought under.

    `impl_sha` answers "is this row about the code that produced it". Nothing
    answered "is this row about the CLAIM it was recorded against", and on
    2026-08-24 an owner ruling amended `LC.01`'s `falsified_by`; the amendment
    was exemplary — annotated inline, strengthen-only, and it ended *"Requires
    a re-run to re-buy the certificate under the amended text."* Five days
    later `LC.01` still read PASS at `ran_at 2026-08-09` and that sentence was
    the only record of the debt: `re-buy the certificate` occurred exactly once
    in the whole repository, inside the spec string it described. A ledger row
    carried `impl_sha` and no field for this, so the question had nowhere to be
    answered from and no instrument could ask it. (46th audit, B1.)

    Hashes `SPEC_CLAIM_FIELDS` as an ordered list of `(name, value)` pairs, so
    the payload NAMES the fields it covers: a future reader can tell a hash
    that covered eight fields from one that covered eleven, instead of
    inferring it from a mismatch. `depends_on` is sorted because its order
    carries no meaning and a reorder is not a claim change; every other value
    goes in as written. `Budget` is a `str` Enum, so it serialises as its own
    label (`cpu<10min`) rather than a repr that could drift with the class.

    Truncated to 16 hex like `impl_sha`, for the same reason: it is an
    equality check in a ledger a human reads, not a security boundary.

    Returns None only if the spec cannot be serialised at all — treated by
    `spec_drift` as unknown, never as clean.
    """
    import hashlib
    try:
        payload = []
        for name in SPEC_CLAIM_FIELDS:
            v = getattr(spec, name, None)
            if isinstance(v, Budget):
                v = v.value
            elif isinstance(v, (list, tuple)):
                v = sorted(str(x) for x in v) if name == "depends_on" \
                    else [str(x) for x in v]
            payload.append([name, v])
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    except (TypeError, ValueError):
        return None


def spec_drift(entry: "Result", spec: "Spec") -> List[tuple]:
    """Every reason this entry is not a verdict about the claim as it reads now.

    `(kind, detail)` pairs, empty meaning the row was bought under today's
    words. The shape and the honesty rule are `staleness_of`'s, deliberately:

      SPEC_CHANGED     the row carries a `spec_sha` and it does not match the
                       live spec. POSITIVE evidence that the claim moved after
                       the verdict — the row must be re-bought.
      SPEC_UNSTAMPED   the row predates the field. UNKNOWN, NOT CLEAN — the
                       same rule `impl_changed` already uses for `None`, and
                       the reason back-fill is refused rather than faked: no
                       later run recovers what the words were at `ran_at`.
                       (The registry is in git and a content check like
                       `blob_sha_at_run`'s is possible in principle; it is not
                       done here because a Spec's text is assembled by Python
                       across two registry modules, so reconstructing the
                       claim at an old tree state means executing that tree,
                       not reading a blob. Stated as a known gap with the
                       detector that carries it — the count is reported, so
                       the population shrinks visibly as rows are re-run.)

    Only PASS is worth alarming on and this function does not know the status;
    the caller filters. A FAIL under superseded words is a refutation of a
    claim nobody makes any more, which is untidy, not dangerous.
    """
    out: List[tuple] = []
    recorded = getattr(entry, "spec_sha", None)
    if not recorded:
        out.append(("SPEC_UNSTAMPED",
                    f"recorded at {(getattr(entry, 'ran_at', '') or '?')} "
                    f"before spec_sha existed; whether the claim text has "
                    f"moved since cannot be answered from the record"))
        return out
    cur = spec_sha_of(spec)
    if cur != recorded:
        out.append(("SPEC_CHANGED",
                    f"bought under claim text {recorded}, the registry now "
                    f"reads {cur}; the certificate must be re-bought under "
                    f"the amended words"))
    return out


def prose_only_delta(old_bytes: bytes, new_bytes: bytes) -> str:
    """'' when the edit between two versions is PROVABLY prose-only, else why not.

    Prose-only means comments, whitespace, or docstrings — nothing that
    executes. Proved, never argued: both versions are parsed, every docstring
    (module, class, function) is blanked, and the remaining trees must be
    `ast.dump`-identical. Comments and formatting never reach the AST, so a
    single comparison covers both licensed cases; ANY change that survives it
    — a constant, a threshold, an `IMPL_DEPS` entry, a reordered statement —
    is code, and the caller must refuse.

    Written for the 25th audit's B3 (2026-08-21). `impl_sha` hashes whole
    files because the pre-registered gates LIVE in docstrings, so a
    documentation edit and a threshold edit are indistinguishable to it — and
    the loop had started routing audit answers AWAY from the files they
    describe to avoid staling their claims (LESSONS: "A whole-file integrity
    hash taxes documentation — pay it with an amendment lane, never by
    writing less"). This function is the lane's proof obligation. It must
    stay CONSERVATIVE: a false '' here would let a pre-registration edit
    masquerade as prose, which is the exact T0.27 hazard the whole-file hash
    exists to catch. When in doubt, return a reason.
    """
    import ast

    def _stripped_dump(src: bytes) -> str:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body[0].value.value = ""
        return ast.dump(tree)

    try:
        old_d = _stripped_dump(old_bytes)
        new_d = _stripped_dump(new_bytes)
    except (SyntaxError, ValueError) as e:
        return f"unparseable ({type(e).__name__}: {e})"
    if old_d != new_d:
        return "the docstring-stripped ASTs differ — code moved, not prose"
    return ""


def blob_reconstructing_sha(path, want_sha: str, repo_root=None) -> tuple:
    """The committed version of `path` whose `impl_sha` is `want_sha`.

    Walks the file's git history newest-first and re-derives each version's
    sha THROUGH `impl_sha_of` (historical file bytes + dependency files as
    they stand now). Returns `(blob_bytes, commit, problem)` — `blob_bytes`
    is None whenever `problem` is non-empty.

    Failure to find a match is itself informative, so the reasons are spelled
    out: either the recorded sha folded in `IMPL_DEPS` content that has ITSELF
    moved since the run (real staleness — a dependency change can never be
    prose), or the run executed a tree that was never committed (the DIRTY
    shape). Both must stay stale; the doc-only lane refuses.
    """
    import subprocess
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        rel = str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return None, None, f"{path} is not under {root}"
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%H", "--", rel],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, None, f"git:{type(e).__name__}"
    if r.returncode != 0:
        return None, None, f"git:rc{r.returncode}"
    for commit in r.stdout.split():
        try:
            s = subprocess.run(
                ["git", "-C", str(root), "show", f"{commit}:{rel}"],
                capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            return None, None, f"git:{type(e).__name__} at {commit[:8]}"
        if s.returncode != 0:
            continue
        if impl_sha_of(path, file_bytes=s.stdout) == want_sha:
            return s.stdout, commit, ""
    return None, None, (
        f"no committed version of {rel} reconstructs {want_sha} — either a "
        "declared IMPL_DEPS file has itself changed since the run (real "
        "staleness: a dependency edit is never prose) or the run executed "
        "uncommitted code (the DIRTY shape); both stay stale")


#: Where `preserve_impl_bytes` anchors its manifests. A ref, not a loose
#: object: `git gc` prunes unreachable blobs after two weeks, and an artifact
#: that expires is not an artifact.
FAILIMPL_REF_PREFIX = "refs/jack/failimpl"


def preserve_impl_bytes(path, want_sha: str, spec_id: str, ran_at: str,
                        repo_root=None) -> tuple:
    """Archive the exact bytes an adverse verdict RAN, into git's object
    database, when the tree was too dirty for a commit to name them.

    THE HOLE THIS FILLS, and it is a measured one. `audit_supersedes_fail`
    requires the FAIL under an amendment to be recoverable, so a reader can
    run `git diff <fail> <pass>` and see what moved. A FAIL recorded from a
    dirty tree satisfies nothing: the failing code exists only in the working
    tree, and the very next edit destroys it. That is not hypothetical — on
    2026-08-29 13:14 `T0.17` recorded FAIL at `d84101e+dirty` (impl_sha
    `072ea7a4d72997cc`), was fixed in place, and PASSed at 13:15. Checked with
    `tree_reconstructing_sha` on 2026-08-30: **no committed tree state
    reconstructs that sha** — the failing implementation is gone for good, and
    `T0.27`'s live-ledger property has been red ever since, permanently,
    because nothing can recover bytes that were never written down.

    The runner already WARNED about the dirty stamp (`warns_on_dirty_fail`),
    and the warning did not survive contact with a builder mid-fix. So this
    removes the discipline instead of restating it: `run_spec` calls this on
    every FAIL/VOID stamped `+dirty`, and the bytes are kept whether or not
    anyone remembers to commit them.

    HOW, and why a manifest rather than a tree. Each file (the test plus every
    path in its `IMPL_DEPS`) goes in with `git hash-object -w`; a JSON manifest
    naming path -> blob sha goes in as one more blob; a ref under
    `FAILIMPL_REF_PREFIX` points at the manifest so `git gc` cannot reap any of
    it. `git mktree` would need the paths flattened (it takes no slashes) and
    a flattened path is a path you cannot check out. Recovery is two commands:

        git cat-file -p <ref>                  # the manifest
        git cat-file -p <blob> > /tmp/failing.py

    WHAT IT PROMISES, exactly. Before writing the ref this re-derives
    `impl_sha_of` from the bytes it just stored and refuses unless it equals
    `want_sha` — the same one true code path `blob_reconstructing_sha` and
    `tree_reconstructing_sha` use. So a returned ref means "these bytes ARE the
    implementation the row names", proven, not asserted. On any failure it
    returns `(None, problem)` and the caller records the problem: a
    preservation that quietly did nothing would be worse than none, because the
    row would carry a promise the object store cannot keep.

    WHAT IT DOES NOT DO. It does not clear, weaken or widen any pre-registered
    gate. `audit_supersedes_fail` still demands a COMMITTED tree state and
    still flags the `T0.17` pair; whether a preserved manifest should count as
    an equal artifact is a spec question, routed to the Review (journal
    2026-08-30) rather than settled here by the author of the mechanism.
    Nothing in this function is reachable from a PASS.
    """
    import subprocess
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        rel = str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return None, f"{path} is not under {root}"

    def _git(*args, data=None):
        return subprocess.run(["git", "-C", str(root), *args], input=data,
                              capture_output=True, timeout=30)

    deps, problem = impl_deps_of(path)
    if problem:
        return None, f"IMPL_DEPS unreadable: {problem}"
    manifest, blobs = {}, {}
    for p in (rel, *deps):
        f = root / p
        if not f.exists():
            # Recorded, not skipped: `impl_sha_of` hashes a missing dep as the
            # literal `missing:<path>`, so the manifest has to say the same
            # thing or the verification below would be checking a different
            # tree state than the row names.
            manifest[p] = None
            continue
        raw = f.read_bytes()
        h = _git("hash-object", "-w", "--stdin", data=raw)
        if h.returncode != 0:
            return None, f"git hash-object failed for {p}: {h.stderr[:120]!r}"
        manifest[p] = h.stdout.decode().strip()
        blobs[p] = raw

    got = impl_sha_of(path, file_bytes=blobs.get(rel),
                      dep_bytes={p: blobs.get(p) for p in deps})
    if got != want_sha:
        return None, (f"stored bytes hash {got}, row names {want_sha} — the "
                      "tree moved under the run; nothing preserved")

    body = json.dumps({"spec": spec_id, "ran_at": ran_at, "impl_sha": want_sha,
                       "files": manifest}, indent=1, sort_keys=True).encode()
    mh = _git("hash-object", "-w", "--stdin", data=body)
    if mh.returncode != 0:
        return None, f"git hash-object failed for the manifest: {mh.stderr[:120]!r}"
    manifest_sha = mh.stdout.decode().strip()
    # Ref names forbid ':' — the ISO timestamp has two of them.
    ref = f"{FAILIMPL_REF_PREFIX}/{spec_id}/{ran_at.replace(':', '-')}"
    u = _git("update-ref", ref, manifest_sha)
    if u.returncode != 0:
        return None, f"git update-ref {ref} failed: {u.stderr[:120]!r}"
    return ref, ""


def tree_reconstructing_sha(path, want_sha: str, repo_root=None) -> tuple:
    """The committed TREE STATE (file + declared deps, all at one commit)
    whose `impl_sha` is `want_sha`.

    `blob_reconstructing_sha` reads deps as they stand NOW, so a spec whose
    recorded sha folded in a dependency that later moved cannot be
    reconstructed by it even when the dependency's own edit was provably
    prose. The instance that forced this (2026-08-24, straight after the
    26th audit's B2): the audit-ordered docstring fix to
    `t2_03_pretrained_vision.py` — itself cleared through the doc-only lane —
    stranded T3.01's certificate, because T3.01 declares that file in
    IMPL_DEPS and the lane had no way to prove the dependency's drift safe.
    Per LESSONS ("add a lane, do not lower the bar"), this widens the PROOF,
    not the permission.

    Walks the union history of the file and its declared deps newest-first
    and re-derives each commit's sha THROUGH `impl_sha_of` with every byte
    taken from that commit (the historical source names its own dep list).
    Returns `(file_bytes, commit, drifted, problem)`: `drifted` maps each
    dep whose bytes at the matching commit differ from its bytes on disk now
    to those historical bytes (None when it existed on only one side). A
    match proves NOTHING by itself — it only names the exact bytes the entry
    certifies; the caller owes a prose proof for the file AND every entry in
    `drifted`, or must refuse.
    """
    import subprocess
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        rel = str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return None, None, None, f"{path} is not under {root}"

    def _show(commit, r):
        s = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{r}"],
                           capture_output=True, timeout=30)
        return s.stdout if s.returncode == 0 else None

    deps_now, _ = impl_deps_of(path)
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%H", "--",
             rel, *(deps_now or [])],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, None, None, f"git:{type(e).__name__}"
    if r.returncode != 0:
        return None, None, None, f"git:rc{r.returncode}"
    for commit in r.stdout.split():
        fb = _show(commit, rel)
        if fb is None:
            continue
        hist_deps, _ = impl_deps_of(path, source=fb)
        dep_bytes = {d: _show(commit, d) for d in (hist_deps or [])}
        if impl_sha_of(path, file_bytes=fb, dep_bytes=dep_bytes) != want_sha:
            continue
        drifted = {}
        for d, db in dep_bytes.items():
            cur = root / d
            curb = cur.read_bytes() if cur.is_file() else None
            if db != curb:
                drifted[d] = db if (db is not None and curb is not None) \
                    else None
        return fb, commit, drifted, ""
    return None, None, None, (
        f"no committed tree state reconstructs {want_sha} for {rel} — the "
        "run executed uncommitted code (the DIRTY shape) or predates this "
        "repository's history; both stay stale")


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


#: The name of the module-level flag by which a spec declares that its
#: pre-registered bars are FROZEN. `SM.02` invented the idiom (a pilot on
#: disjoint seeds fixes the numbers, then the flag flips and `run()` stops
#: refusing) and `SM.03` copied it; `gates_frozen` reads it statically so the
#: idiom becomes a fact instruments can join against instead of a convention.
GATES_FROZEN_FLAG = "_GATES_FROZEN"


def gates_frozen(spec_id: str, path=None):
    """Has this spec frozen its pre-registered gates? `True` / `False` / `None`.

    `None` means THE SPEC DOES NOT DECLARE — which is the overwhelming majority
    of them and is NOT an accusation. Only a spec that has chosen the
    provisional-gates idiom can answer this question at all, so `None` reads
    "not applicable", and a caller must test `is False`, never falsiness.

    Read by AST rather than by import, for the same reason `module_path_for`
    answers without importing: importing `sm_03_nose_reports_occluded` costs a
    MuJoCo model build, and an instrument that has to run the thing it is
    auditing is an instrument nobody will run. Only module-level assignments
    count, and the LAST one wins — `SM.02` assigns the flag once at line 288
    and reads it at 804, and a future file that re-assigns it after a pilot
    block must be read the way Python would read it.

    THE LOUD DIRECTION, deliberately: a declaration whose value is not the
    literal `True` — `False`, an env lookup, an expression, a name — returns
    `False`, and a file that cannot be read or parsed returns `False` too. A
    spec whose freeze cannot be established by reading the source is a spec
    whose freeze has not been established; and a file that does not parse
    cannot be dispatched whatever its flag says. The failure mode this guards
    is the one direction that matters: `queue_depth` counting an undispatchable
    spec as shelf stock, which is exactly how `gpu<20min` read 1 while the
    honest answer was 0 (46th audit, RANK 2).

    WHAT IT DOES NOT COVER, stated because the deleted `KNOWN OVER-COUNT`
    paragraph in `coverage.queue_depth` overclaimed in the other direction:
    this detects a DECLARED refusal. A `run()` that refuses for some other
    reason — a missing artefact, an unmet precondition, a raise — still counts
    toward queue depth, because nothing in the repo makes that declarable yet.
    The number is still an upper bound; it is now a tighter one.
    """
    import ast
    if path is None:
        path = module_path_for(spec_id)
    if not path:
        return None
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError, ValueError):
        return False
    found = None
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == GATES_FROZEN_FLAG:
                value = getattr(node, "value", None)
                found = (isinstance(value, ast.Constant)
                         and value.value is True)
    return found


def deps_moved_since(path, ran_at, repo_root=None) -> tuple:
    """Declared `IMPL_DEPS` with commits after `ran_at` — the one staleness
    question still answerable for an entry recorded before `impl_sha` existed.

    The 14th overseer audit (2026-08-13): four world certificates carried
    `IMPL_DEPS = ["playground.py"]` and none of its protection, because they
    were recorded before `impl_sha` was born — `playground.py` took +430/-14
    lines and `run stale` read clean for all four, forever. "Cannot be checked"
    and "cannot be checked AND its declared dependency has demonstrably moved"
    are different facts; only the second can bite, and it was invisible.

    Compares against `ran_at`, NOT against the recorded commit — deliberately.
    `stale_claims` documents why commit-ancestry lies here: code is edited,
    RUN, and only then committed, so the commit that lands minutes after a run
    often contains the very bytes that ran. A commit date after the wall-clock
    moment of the run is the claim actually being made.

    Returns `(moved, problem)` like `impl_deps_of`: `problem` non-empty when
    git could not answer, reported rather than swallowed — an unanswerable
    check that returns "nothing moved" is a clean scan nobody performed.

    The date comparison happens HERE, not in git. The first draft passed
    `--since={ran_at}` and its known-negative probe (ran_at=2999) still
    returned a commit: approxidate quietly reinterprets a date it finds
    implausible instead of erroring — a silent fallback inside the very guard
    against silent staleness. So git is asked only for the dependency's last
    commit date (`%cI`, which carries its offset) and Python does the
    comparison, converting to the box's local clock because `ran_at` is
    recorded from it.
    """
    import subprocess
    from datetime import datetime
    deps, problem = impl_deps_of(path)
    if problem:
        return (), problem
    if not deps or not ran_at:
        return (), ""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        ran = datetime.fromisoformat(str(ran_at))
    except ValueError:
        return (), f"ran_at unparseable: {ran_at!r}"
    if ran.tzinfo is not None:
        ran = ran.astimezone().replace(tzinfo=None)
    moved = []
    for rel in deps:
        try:
            r = subprocess.run(
                ["git", "-C", str(root), "log", "-1", "--format=%cI", "--", rel],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            return (), f"git:{type(e).__name__}"
        if r.returncode != 0:
            return (), f"git:rc{r.returncode}"
        stamp = r.stdout.strip()
        if not stamp:
            continue  # never committed: impl_sha_of already reports `missing:`
        try:
            # %cI can end in 'Z', which fromisoformat rejects before 3.11.
            last = (datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    .astimezone().replace(tzinfo=None))
        except ValueError:
            return (), f"commit date unparseable: {stamp!r}"
        if last > ran:
            moved.append(rel)
    return tuple(moved), ""


# Memo for `blob_sha_at_run`: a report scan asks the same (path, ran_at)
# question several times per invocation (`_check_stale_detector` replays the
# real scan on probe ledgers), and the answer is a function of git history,
# which does not change within a process.
_BLOB_SHA_MEMO: Dict[tuple, tuple] = {}


def blob_sha_at_run(path, ran_at, repo_root=None, grace_min=30) -> tuple:
    """sha256 of a file's content as git last recorded it at run time.

    The declaration-free half of provenance (15th overseer audit, B1).
    `impl_sha` and `IMPL_DEPS` are opt-in: every record old enough to lack the
    stamp is old enough to lack the declaration, so the `UNVERIFIABLE_MOVED`
    detector's domain and the at-risk population are disjoint BY CONSTRUCTION
    — the planted positive lived in the domain and no real record did. This
    check needs nobody's opt-in: its input is a property the artifact cannot
    help having (the file's committed content and the entry's `ran_at`).

    Baseline = the newest commit touching the file whose committer date is at
    most `ran_at + grace_min` minutes. The grace window exists because code is
    edited, RUN, and only then committed — the recording commit routinely lands
    seconds after its own run and contains the very bytes that ran. Comparing
    against the recorded commit alone reports 8 stale where the truth is 3
    (measured, 15th audit); the window fixes exactly that. An over-reporting
    auditor is a defect too.

    Date comparison happens in Python on `%cI`, never via `--since`/`--until`:
    approxidate silently reinterprets implausible dates instead of erroring —
    a silent fallback inside the very guard against silent staleness (the
    `deps_moved_since` scar, kept).

    Returns `(sha, problem)`: `sha` is None whenever `problem` is non-empty,
    and a check that cannot run says so rather than reading clean.
    """
    import hashlib
    import subprocess
    from datetime import datetime, timedelta
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    key = (str(root), str(Path(path)), str(ran_at), grace_min)
    if key in _BLOB_SHA_MEMO:
        return _BLOB_SHA_MEMO[key]

    def _done(sha, problem):
        _BLOB_SHA_MEMO[key] = (sha, problem)
        return (sha, problem)

    try:
        rel = str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return _done(None, f"{path} is not under {root}")
    try:
        ran = datetime.fromisoformat(str(ran_at))
    except (TypeError, ValueError):
        return _done(None, f"ran_at unparseable: {ran_at!r}")
    if ran.tzinfo is not None:
        ran = ran.astimezone().replace(tzinfo=None)
    cutoff = ran + timedelta(minutes=grace_min)
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%H %cI", "--", rel],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return _done(None, f"git:{type(e).__name__}")
    if r.returncode != 0:
        return _done(None, f"git:rc{r.returncode}")
    best = best_date = None
    for line in r.stdout.splitlines():
        try:
            commit, stamp = line.split()
            when = (datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    .astimezone().replace(tzinfo=None))
        except ValueError:
            return _done(None, f"commit line unparseable: {line!r}")
        if when <= cutoff and (best_date is None or when > best_date):
            best, best_date = commit, when
    if best is None:
        return _done(None, f"no commit touches {rel} at or before "
                           f"ran_at+{grace_min}min")
    try:
        s = subprocess.run(["git", "-C", str(root), "show", f"{best}:{rel}"],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return _done(None, f"git:{type(e).__name__}")
    if s.returncode != 0:
        return _done(None, f"git show rc{s.returncode} at {best[:8]}")
    return _done(hashlib.sha256(s.stdout).hexdigest(), "")


def staleness_of(entry: "Result", path) -> List[tuple]:
    """Every reason this entry is not a claim about the code that exists now.

    Returns a list of `(kind, detail)` — empty means the entry still describes
    the current implementation. Kinds:

      DIRTY        the run's commit stamp ends in `+dirty`, so the code that
                   produced it exists in no commit and cannot be recovered.
      UNSTAMPED_CHANGED  the entry predates `impl_sha`, but git can answer
                   anyway: the file's content at HEAD differs from the blob
                   that stood at run time (`blob_sha_at_run`). This is STALE
                   with positive evidence, not "cannot be checked" — the 15th
                   audit's B1: three genuinely stale records (T0.09, T1.07,
                   T2.02) sat inside a bucket the report called unchecked,
                   because the opt-in detector's domain and the at-risk
                   population were disjoint by construction.
      UNSTAMPED_INTACT  the entry predates `impl_sha` and git shows the file
                   byte-identical since the run. Bookkeeping, not a hazard;
                   a re-run still upgrades it to a real stamp (and is still
                   required before `borrow_metrics` will read it — content
                   identity says nothing about undeclared dependencies).
      UNVERIFIABLE the entry predates `impl_sha` AND the declaration-free
                   check could not run (reason in the detail); nothing can be
                   compared. The honest remainder, kept separate on purpose:
                   a clean scan and a scan that never ran must not share a
                   bucket.
      UNVERIFIABLE_MOVED  alongside any of the three above when the module
                   declares `IMPL_DEPS` and a declared dependency has commits
                   after `ran_at` — the alarm is fitted and structurally
                   cannot fire, over a world that has demonstrably moved
                   (14th audit: PG.1/PG.2/PG.4/T2.20).
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
        import hashlib
        ran_at = getattr(entry, "ran_at", None)
        base, cc_problem = blob_sha_at_run(path, ran_at)
        if cc_problem:
            out.append(("UNVERIFIABLE",
                        f"recorded at {(entry.commit or '?')[:8]} before "
                        f"impl_sha existed; content check could not run "
                        f"({cc_problem})"))
        else:
            try:
                cur = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError as e:
                cur = None
                out.append(("UNVERIFIABLE",
                            f"recorded before impl_sha existed; "
                            f"{Path(path).name} unreadable ({type(e).__name__})"))
            if cur is not None and cur != base:
                out.append(("UNSTAMPED_CHANGED",
                            f"{Path(path).name}: content at HEAD differs from "
                            f"the blob that stood at ran_at "
                            f"{ran_at or '?'} ({base[:12]}); the entry is "
                            f"about older code"))
            elif cur is not None:
                out.append(("UNSTAMPED_INTACT",
                            f"recorded at {(entry.commit or '?')[:8]} before "
                            f"impl_sha existed; git shows {Path(path).name} "
                            f"byte-identical since the run"))
        moved, problem = deps_moved_since(path, getattr(entry, "ran_at", None))
        if problem:
            # Conservative direction: a check that cannot run is reported as
            # fired-with-reason, never as clean (the T0.13 shape).
            out.append(("UNVERIFIABLE_MOVED",
                        f"declares IMPL_DEPS but the moved-check could not run "
                        f"({problem})"))
        elif moved:
            out.append(("UNVERIFIABLE_MOVED",
                        f"declares {list(moved)} which took commits after "
                        f"ran_at {getattr(entry, 'ran_at', '?')}; the "
                        f"staleness alarm it declares cannot fire"))
        return out
    cur = impl_sha_of(path)
    if cur != recorded:
        out.append(("CHANGED",
                    f"{Path(path).name}: ran on {recorded}, now {cur}"))
    return out


def audit_supersedes_fail(results: Dict[str, Any],
                          repo_root=None) -> Dict[str, Any]:
    """Every amend-after-adverse-verdict must be auditable by someone who is
    not its author.

    The executable form of overseer B2 (2026-08-13). T2.08's floor moved
    0.70 -> 0.50 between a FAIL and a PASS; the move was disclosed loudly and
    honestly, and the repo could not have caught a dishonest one: the FAIL was
    stamped `75a1938+dirty` (the failing code exists in no commit) and its
    measurement survived only in prose written by the party that moved the
    threshold. Disclosure is a property of the agent; this makes it a property
    of the ledger.

    COVERAGE (22nd audit B1/B2; the LESSONS rule is to say it): the audited
    sources are FAIL **and VOID** — SYSTEM.md's "fix the arm, do not decide"
    makes VOID the verdict that doctrinally precedes a redesign, and it was
    the one lane with no guard. ERROR rows are dropped before pairing (an
    ERROR is infrastructure, not a verdict, and a dead kernel carries the
    UNCHANGED impl_sha — adjacency pairing let the "same code" shortcut skip
    the real amendment; T2.05's chain VOID -> ERROR -> ERROR -> FAIL is the
    fixture). NOT covered: PASS/SKIP/BLOCKED as sources — a re-run on top of
    those amends no adverse verdict.

    THE RULE: in any record whose CURRENT status is PASS, a FAIL (or VOID)
    whose implementation differs from the run that superseded it (the
    threshold or the code moved in between — impl_sha cannot tell those
    apart, and the conservative superset is the point) must be

      * stamped at a CLEAN commit (no `+dirty` — else the failing code is
        unrecoverable by construction),
      * a commit that EXISTS in this repository (else `git diff` between the
        failing and passing code — the artifact that shows exactly which
        constants moved — is impossible), and
      * carrying its METRICS (else the failing measurement exists only in
        prose).

    Pairs where either side predates `impl_sha` are counted `unauditable`, not
    violated: absence is a historical gap, never evidence of dishonesty, and
    back-filling judgement onto it would be inventing verdicts (B1's rule).
    Records whose CURRENT status is not PASS are ignored entirely — no
    capability is asserted until a PASS stands on top, and mid-loop iteration
    on a still-failing spec is the loop working. Once a PASS stands, EVERY
    amended FAIL under it is checked (including FAIL -> FAIL links), because
    the constants' path from first FAIL to the standing PASS is what an
    auditor has to reconstruct.

    `results` is the raw ledger dict (`json.loads(...)["results"]`).
    `repo_root=None` skips the git-existence check (fixture ledgers carry
    synthetic commits); pass the repo root to audit the real ledger.
    """
    violations: List[Dict[str, Any]] = []
    checked = unauditable = 0

    def _commit_exists(sha: str) -> bool:
        try:
            p = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", sha + "^{commit}"],
                capture_output=True, text=True, cwd=repo_root, timeout=10)
            return p.returncode == 0
        except Exception:
            return False

    for sid, row in results.items():
        if row.get("status") != Status.PASS.value:
            continue
        # ERROR rows are dropped BEFORE pairing: an ERROR is an infrastructure
        # event, not a verdict, and a dead kernel records the UNCHANGED
        # impl_sha — so pairing on adjacency let the "same code re-run"
        # shortcut skip the real amendment whenever a kernel died between the
        # adverse verdict and its re-run (overseer 22nd audit B2; T2.05
        # produced three ERROR rows in one day). COVERAGE: FAIL and VOID are
        # audited sources; PASS/SKIP/BLOCKED rows pass through as pair
        # boundaries but are never themselves flagged.
        seq = [e for e in list(row.get("history") or []) + [row]
               if e.get("status") != Status.ERROR.value]
        for e, nxt in zip(seq, seq[1:]):
            if e.get("status") not in (Status.FAIL.value, Status.VOID.value):
                continue
            lbl = e["status"]
            if not e.get("impl_sha") or not nxt.get("impl_sha"):
                unauditable += 1
                continue
            if e["impl_sha"] == nxt["impl_sha"]:
                continue          # same code re-run; nothing was amended
            checked += 1
            reasons = []
            stamp = str(e.get("commit") or "")
            if not stamp or stamp == "unknown":
                reasons.append(f"{lbl} carries no commit stamp")
            elif stamp.endswith("+dirty"):
                reasons.append(f"{lbl} stamped {stamp}: that implementation "
                               "was never committed")
            elif repo_root is not None and not _commit_exists(stamp):
                reasons.append(f"{lbl} commit {stamp} does not exist in this "
                               "repository")
            if not e.get("metrics"):
                reasons.append(f"{lbl} history entry carries no metrics — "
                               "the measurement exists only in prose")
            if reasons:
                violations.append({"spec_id": sid, "status": lbl,
                                   "fail_commit": stamp,
                                   "fail_ran_at": e.get("ran_at"),
                                   "reasons": reasons})
    return {"violations": violations, "checked_pairs": checked,
            "unauditable_pairs": unauditable}


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


def _drain_gpu_job_ids() -> List[str]:
    """Job ids `gpu.submit()` recorded in this process, cleared on read.

    Read through `sys.modules` rather than an import: a spec that never
    imported the gpu module cannot have dispatched anything, and the recorder
    must not be the thing that first loads it. `gpu` imports from this module,
    so an eager import here would also be a cycle.
    """
    mod = sys.modules.get("experiments.gpu")
    if mod is None:
        return []
    try:
        return list(mod.drain_job_ids())
    except Exception:
        return []


def _drain_gpu_reattach_mismatches() -> List[Dict[str, Any]]:
    """Tolerated reattach code divergences, cleared on read (overseer 20th B1).

    Same sys.modules discipline and the same paired call sites as the other
    two drains. When a reattach went ahead under JACK_REATTACH_ACCEPT_MISMATCH,
    the kernel's code differs from the tree `impl_sha` will name — the row's
    `message` must state that fact, or the certificate silently claims code
    that did not run.
    """
    mod = sys.modules.get("experiments.gpu")
    if mod is None:
        return []
    try:
        return list(mod.drain_reattach_mismatches())
    except Exception:
        return []


def _drain_gpu_charge_s() -> Optional[float]:
    """Metered seconds `gpu.submit()` accumulated in this process, cleared on
    read. Same sys.modules discipline as `_drain_gpu_job_ids`, same call
    sites — the two drains must stay paired or one spec's charges could be
    attributed to the next."""
    mod = sys.modules.get("experiments.gpu")
    if mod is None:
        return None
    try:
        return mod.drain_charge_seconds()
    except Exception:
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
    blocked = ledger.unsatisfied(spec)
    if blocked:
        # Carry the REASON, not just the id. "dependencies not passing: PS.01"
        # is unactionable when PS.01 is a PASS — the reader needs to be told it
        # is a PASS about code that has since changed, and that a re-run clears
        # it. A refusal that cannot be acted on gets worked around.
        res = Result(spec_id=spec.id, status=Status.BLOCKED,
                     message="dependencies not satisfied: " + "; ".join(
                         f"{d} ({why})" for d, why in blocked),
                     # Stamped on the BLOCKED path too, so "the runner always
                     # records the words it was handed" holds with no
                     # exceptions. A BLOCKED row asserts nothing about the
                     # claim, but it does say WHEN the refusal happened, and a
                     # field with one silent exemption is a field an auditor
                     # has to special-case — which is how exemptions become
                     # holes. Found by T0.17 P10's round trip, which recorded
                     # BLOCKED and read back `spec_sha: null`.
                     spec_sha=spec_sha_of(spec),
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
    # GPU dispatch provenance (overseer B3). Drained BEFORE the runs so another
    # spec's leftover job ids in this process cannot be attributed to this one;
    # the env var lets `gpu.submit()` write the spec id into its receipt log
    # without every spec having to thread it through.
    _drain_gpu_job_ids()
    _drain_gpu_charge_s()
    _drain_gpu_reattach_mismatches()
    _prev_spec_env = os.environ.get("JACK_SPEC_ID")
    os.environ["JACK_SPEC_ID"] = spec.id
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
    finally:
        if _prev_spec_env is None:
            os.environ.pop("JACK_SPEC_ID", None)
        else:
            os.environ["JACK_SPEC_ID"] = _prev_spec_env

    # Every remote job the runs paid for, folded in here so the ledger and
    # `gpu_submissions.jsonl` share a join key (overseer B3). Includes failed
    # attempts — those spent quota too, and a record that names only the job
    # that succeeded cannot answer "which hours bought this".
    job_ids = _drain_gpu_job_ids()
    compute_s = _drain_gpu_charge_s()
    # A reattach that proceeded despite a code divergence must surface in the
    # row itself: `impl_sha` names the local tree, but the kernel's numbers
    # came from the sha the receipt recorded at push time (overseer 20th B1 —
    # "the sha is the sha of what executed"). Message, not a silent field.
    for _mm in _drain_gpu_reattach_mismatches():
        _note = (f"REATTACH CODE MISMATCH tolerated: kernel {_mm.get('job_id')} "
                 f"ran code sha {str(_mm.get('recorded_sha'))[:16]} (head "
                 f"{_mm.get('submitted_head')}), local script hashed "
                 f"{str(_mm.get('local_sha'))[:16]} — impl_sha names code that "
                 f"did not run remotely; see gpu_submissions.jsonl")
        message = f"{message} | {_note}" if message else _note
    # The stamp names the machine that RAN the work, not the dispatcher: nine
    # GPU records read aarch64/…/cpu while the truth sat in metrics["gpu"]
    # (overseer B3). The dispatcher stays visible because it is also true.
    _gpu_name = metrics.get("gpu")
    if isinstance(_gpu_name, str) and _gpu_name.strip():
        stamp = {**stamp, "hardware":
                 f"remote/{_gpu_name} (dispatched from {stamp['hardware']})"}

    # An adverse verdict from a dirty tree is the one row whose implementation
    # no commit can name, and the next edit destroys it (T0.17, 2026-08-29).
    # Preserve the bytes here, in the object database, before that can happen.
    # Deliberately NOT gated on the spec having a later amendment: the loss
    # happens at the FAIL, and by the time an amendment exists it is too late.
    ran_at_s = time.strftime("%Y-%m-%dT%H:%M:%S")
    preserved = None
    if status in (Status.FAIL, Status.VOID) and str(
            stamp.get("commit", "")).endswith("+dirty"):
        try:
            import inspect
            _src = inspect.getsourcefile(fn)
            preserved, _why = preserve_impl_bytes(
                _src, impl_sha, spec.id, ran_at_s) if _src else (
                None, "no source file for the experiment function")
        except Exception as e:            # never turn a FAIL into an ERROR
            preserved, _why = None, f"{type(e).__name__}: {e}"[:120]
        _note = (f"failing implementation preserved at {preserved}"
                 if preserved else
                 f"FAILING IMPLEMENTATION NOT PRESERVED: {_why} — commit this "
                 "tree before you edit it or the FAIL becomes unauditable")
        message = f"{message} | {_note}" if message else _note

    res = Result(spec_id=spec.id, status=status, metrics=metrics,
                 control_metrics=control_metrics, seeds=seeds,
                 preserved_impl=preserved,
                 duration_s=round(time.time() - t0, 2), message=message,
                 compute_s=(round(compute_s, 2) if compute_s is not None else None),
                 impl_sha=impl_sha,
                 # Stamped from the Spec object THIS run was handed, not from a
                 # registry lookup by id: the two can differ (a bakeoff builds
                 # arms with `replace`), and the row must name the words the
                 # runner actually enforced.
                 spec_sha=spec_sha_of(spec),
                 gpu_job_id=",".join(job_ids) if job_ids else None,
                 ran_at=ran_at_s, **stamp)
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
