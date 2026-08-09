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
    attempt: int = 1

    @staticmethod
    def env_stamp() -> Dict[str, str]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent, timeout=10,
            ).stdout.strip()
        except Exception:
            commit = "unknown"
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
        """Merge one result into the ledger under an exclusive lock.

        Naive save() lost data: each Ledger held an in-memory copy and wrote the
        whole file, so a writer with a stale view silently erased results it had
        never seen. Measured, two interleaved writers kept 11 of 15 records
        (ladder spec T0.08). That is not hypothetical — the hourly loop and a
        manual session overlap by design.

        So: lock, RE-READ from disk, merge, write atomically. The tmp+os.replace
        is the same pattern T0.05 established, because a ledger truncated by a
        SIGKILL would erase the evidence for every capability claimed so far.
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
                for rid, r in self.results.items():
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
                        hist.append({k: prev.get(k) for k in
                                     ("status", "ran_at", "commit", "message")})
                    row = {**asdict(r), "status": r.status.value}
                    row["history"] = hist[-20:]
                    row["attempt"] = len(hist) + 1
                    merged[rid] = row

                payload = {
                    "_comment": "Written by experiments/run.py under an exclusive lock. "
                                "Do not hand-edit — a claim here must come from a test "
                                "that could have failed.",
                    "results": merged,
                }
                fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)

                # Adopt the merged view so this instance stops being stale.
                for rid, raw in merged.items():
                    if rid not in self.results:
                        d = dict(raw)
                        d["status"] = Status(d["status"])
                        self.results[rid] = Result(**d)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def save(self) -> None:
        """Retained for compatibility; record() is the safe path."""
        if self.results:
            self.record(next(iter(self.results.values())))

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
    ledger = ledger or Ledger()
    blocked = ledger.blocked_by(spec)
    if blocked:
        res = Result(spec_id=spec.id, status=Status.BLOCKED,
                     message=f"dependencies not passing: {', '.join(blocked)}",
                     **Result.env_stamp(), ran_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        ledger.record(res)
        return res

    t0 = time.time()
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
            status = Status.PASS if ok else Status.FAIL
            message = "" if ok else "pre-registered threshold not met"
    except Exception as e:
        metrics, control_metrics = {}, {}
        status = Status.ERROR
        message = f"{type(e).__name__}: {e}"[:400]

    res = Result(spec_id=spec.id, status=status, metrics=metrics,
                 control_metrics=control_metrics, seeds=seeds,
                 duration_s=round(time.time() - t0, 2), message=message,
                 ran_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **Result.env_stamp())
    ledger.record(res)
    return res


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
            out[k] = round(mean, 6)
            out[f"{k}_std"] = round(var ** 0.5, 6)
        else:
            out[k] = runs[0][k]
    return out
