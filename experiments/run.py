#!/usr/bin/env python
"""Ladder runner.

    python -m experiments.run status          # the checklist, current state
    python -m experiments.run next            # what is legitimately runnable now
    python -m experiments.run T0.02           # run one experiment
    python -m experiments.run --tier 0        # run a whole tier, in order
    python -m experiments.run --gate          # re-run every PASSing test (regression)

Dependencies are enforced: a spec whose prerequisites are not PASSing is recorded
BLOCKED rather than run, because a number computed on a broken foundation is worse
than no number — it looks like evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .protocol import Ledger, Status
from .registry import BY_ID, LADDER, ready, tier

TESTS_DIR = Path(__file__).parent / "tests"
RUN_LOCK = "/tmp/jack-ladder.lock"          # shared with scripts/ladder_loop.sh


def _lock_for(spec_ids) -> str:
    """CPU work and remote-GPU work do not contend, so they must not share a lock.

    Discovered 2026-08-09 with the box at 4% CPU: a T2.01 run holding the
    single ladder lock for ~6 hours while merely POLLING Kaggle made the
    hourly builder skip every iteration. Six hours of idle cores with ~100
    CPU-core-hours of designed science queued, because a job waiting on a
    REMOTE GPU held the LOCAL CPU-work lock.

    The lock exists to stop two torch processes thrashing 4 shared cores —
    a concern that simply does not apply to a process blocked on a network
    poll. Ledger integrity is NOT this lock's job: Ledger.record already
    takes its own fcntl lock and re-reads-merges-writes atomically (the T0.08
    lesson), so concurrent CPU and GPU specs cannot lose each other's results.
    """
    from .registry import BY_ID as _B
    budgets = {(_B[i].budget.value if i in _B else "") for i in spec_ids}
    if budgets and all(b.startswith("gpu") for b in budgets):
        return "/tmp/jack-ladder-gpu.lock"
    return RUN_LOCK


@contextmanager
def _exclusive(spec_ids=()):
    """Serialise ALL ladder work, manual or looped.

    The hourly loop and a manual session raced and each wrote a different T0.07;
    one silently shadowed the other. The loop script already took this lock, but
    a human at a terminal did not, so the guard only protected one side. Holding
    it here means whoever starts second waits or skips, regardless of who they are.
    """
    lock_path = _lock_for(spec_ids)
    with open(lock_path, "w") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another run holds {lock_path} (probably the hourly loop). "
                  "Wait for it, or `touch .loop-paused` to stop the loop.")
            raise SystemExit(0)
        fh.write(f"{os.getpid()}\n"); fh.flush()
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

MARK = {
    Status.PASS: "PASS   ",
    Status.FAIL: "FAIL   ",
    Status.VOID: "VOID   ",   # invalid run, NOT a refutation — see protocol.Status
    Status.BLOCKED: "blocked",
    Status.ERROR: "ERROR  ",
    Status.SKIP: "skip   ",
    Status.NOT_RUN: "-      ",
}


def _module_for(spec_id: str):
    """tests/t0_02_*.py implements T0.02. Missing module = not yet written.

    Duplicates RAISE. Taking the first alphabetical match silently shadowed a
    second implementation: the hourly loop and a manual session each wrote a
    T0.07, `t0_07_cpu_throughput.py` sorted first, and the other was never run
    again — while the ledger reported a PASS that belonged to whichever file won
    the sort. Two implementations of one spec is an unresolved disagreement about
    what the spec means; it must be settled by a person, not by alphabetical order.
    """
    prefix = spec_id.lower().replace(".", "_")
    # The underscore before the slug is load-bearing: "me_1*" would also match
    # me_10_*, so ME.1 and ME.10 would each see two implementations and raise.
    matches = sorted(TESTS_DIR.glob(f"{prefix}_*.py"))
    # Hierarchical ids (ME.11 and its bakeoff arms ME.11.0/ME.11.A) defeat that
    # underscore: `me_11_*` matches `me_11_0_eval_set_honest.py` too, so ME.11
    # would report a duplicate implementation and refuse to run — a naming
    # choice silently disabling a spec. A longer spec id owns its own files.
    longer = [s.id.lower().replace(".", "_") for s in LADDER
              if s.id != spec_id and s.id.lower().replace(".", "_").startswith(prefix + "_")]
    if longer:
        matches = [m for m in matches
                   if not any(m.stem.startswith(p + "_") for p in longer)]
    if len(matches) > 1:
        raise RuntimeError(
            f"{spec_id} has {len(matches)} implementations: "
            f"{', '.join(m.name for m in matches)}. Delete or merge — the runner "
            "will not choose between them."
        )
    if not matches:
        return None
    return importlib.import_module(f"experiments.tests.{matches[0].stem}")


def cmd_status(ledger: Ledger) -> int:
    counts = ledger.summary()
    total = len(LADDER)
    done = counts[Status.PASS.value]
    print(f"\nJack validation ladder — {done}/{total} demonstrated\n")
    current = None
    for s in LADDER:
        if s.tier != current:
            current = s.tier
            names = {0: "HARNESS", 1: "LEARNING PRIMITIVES", 2: "COMPONENT vs NULL",
                     3: "ABLATION — does it earn its parameters?", 4: "COMPOSITION",
                     5: "THE CLAIMS", 6: "INTEGRATION"}
            print(f"\n  TIER {current} — {names.get(current, '')}")
        st = ledger.status(s.id)
        impl = "" if _module_for(s.id) else "  (not implemented)"
        print(f"    [{MARK[st]}] {s.id}  {s.title}{impl}")
    print(f"\n  {counts}\n")
    print("  A capability is claimed ONLY by a PASS here. Nothing else counts.\n")
    return 0


def cmd_next(ledger: Ledger) -> int:
    avail = ready(ledger)
    if not avail:
        print("Nothing runnable — every unblocked spec already passes.")
        return 0
    # Say what is being hidden. `avail[:12]` silently dropped the rest, and the
    # cheapest unblocked work sorts LAST (ME.11.A sat behind twelve GPU specs),
    # so the one command an iteration runs to choose its work was quietly
    # answering a different question than the one it appears to answer.
    shown = min(12, len(avail))
    more = f" — showing {shown} of {len(avail)}" if len(avail) > shown else ""
    print(f"\nRunnable now (dependencies satisfied){more}:\n")
    for s in avail[:12]:
        impl = "" if _module_for(s.id) else "  [needs implementing]"
        print(f"  {s.id}  {s.title}  ({s.budget.value}){impl}")
        print(f"        hypothesis:  {s.hypothesis}")
        print(f"        falsified by: {s.falsified_by}")
        if s.kills:
            print(f"        kills:       {s.kills}")
        print()
    return 0


def _terminal_blockers(ledger: Ledger) -> dict:
    """For every spec, the ROOTS its unreachability actually rests on.

    A spec's immediate parent is almost never the answer. UB.1 reads as blocked
    by T4.01, which is blocked by T3.02, which is blocked by T2.01 = VOID — and
    only T2.01 can be acted on. Walking to the terminal blocker is what turns a
    list of 40 stuck specs into two things to fix.
    """
    terminal: dict = {}

    def walk(sid: str, seen: frozenset) -> set:
        if sid in terminal:
            return terminal[sid]
        spec = BY_ID.get(sid)
        if spec is None or sid in seen:      # unknown dep, or a dependency cycle
            return {sid}
        roots: set = set()
        for d in spec.depends_on:
            if ledger.status(d) is Status.PASS:
                continue
            upstream = walk(d, seen | {sid})
            # A dependency that is itself stuck resolves to ITS roots; one that
            # is merely not-yet-run is a root of its own.
            roots |= upstream if upstream else {d}
        terminal[sid] = roots
        return roots

    for s in LADDER:
        walk(s.id, frozenset())
    return terminal


def cmd_blocked(ledger: Ledger) -> int:
    """What can this ladder NEVER do, and why — the converse of `next`.

    Written 2026-08-09 because the overseer had to walk the dependency graph by
    hand to discover that 29% of the ladder was dead behind two VOIDs, and that
    the dead set was precisely GOAL.md's headline: all 7 curiosity specs, all 16
    unison specs, all of Tiers 3, 4 and 5. `next` answers "what can I do"; until
    now nothing answered "what is unreachable, and what one fix would free it".
    LESSONS.md carried that as advice to humans. This makes it a command.
    """
    terminal = _terminal_blockers(ledger)
    by_root: dict = {}
    for s in LADDER:
        if ledger.status(s.id) is Status.PASS:
            continue
        for root in terminal.get(s.id, set()):
            if root == s.id:                       # runnable now, not blocked
                continue
            by_root.setdefault(root, []).append(s.id)

    if not by_root:
        print("Nothing is blocked — every unrun spec has its dependencies passing.")
        return 0

    ranked = sorted(by_root.items(), key=lambda kv: -len(kv[1]))
    total = len({sid for ids in by_root.values() for sid in ids})
    print(f"\n{total} of {len(LADDER)} specs are unreachable. Terminal blockers, "
          f"worst first:\n")
    for root, ids in ranked:
        st = ledger.status(root).value if root in BY_ID else "UNKNOWN-SPEC"
        title = BY_ID[root].title if root in BY_ID else "(not in the registry)"
        print(f"  {root} = {st}  blocks {len(ids)}  — {title}")
        print(f"        {', '.join(sorted(ids))}\n")
    summary = "; ".join(
        f"{root}={ledger.status(root).value if root in BY_ID else 'UNKNOWN'} "
        f"blocks {len(ids)}" for root, ids in ranked)
    print(f"  SUMMARY: {summary}\n")
    return 0


def _run_isolated(spec_id: str, ledger: Ledger):
    """Execute one spec in a child process so its memory is reclaimed on exit."""
    import subprocess as sp
    from .protocol import Result, Status

    code = (
        "import sys; sys.path.insert(0, %r);"
        "from experiments.run import _module_for;"
        "from experiments.protocol import Ledger;"
        "m = _module_for(%r);"
        "m.run(Ledger())" % (str(Path(__file__).parent.parent), spec_id)
    )
    # Timeout derived from the spec's declared budget. A flat 3600s cap silently
    # killed T2.01 (budget gpu<2h) at 60 minutes while its Kaggle kernel ran to
    # COMPLETION at 66.7 — the runner recorded an ERROR for a job that had
    # produced a real result, and the artifact was only recovered by hand. A
    # harness that discards finished science is worse than a slow one.
    from .registry import BY_ID as _BY_ID
    _budget_seconds = {
        "cpu<1min": 300, "cpu<10min": 1800, "cpu<2h": 9000,
        "gpu<20min": 3600, "gpu<2h": 10800, "gpu<8h": 36000,
    }
    _spec = _BY_ID.get(spec_id)
    _timeout = _budget_seconds.get(_spec.budget.value if _spec else "", 3600)
    # The budget names one EXPERIMENT; a spec runs seeds x (experiment +
    # control). The 3-seed re-verification killed T1.01/02/06 mid-science at
    # the single-seed timeout — six times the work, one budget of time.
    _timeout *= max(1, getattr(_spec, "seeds", 1)) * 2
    # The ran_at of any PRE-EXISTING entry, so a crashed child cannot pass the
    # old result off as its own. T2.01 v3's child died (SIGPIPE from a killed
    # session pipe) after v2 had recorded a FAIL: the old check — "is there an
    # entry at all?" — found v2's entry and reported it as the rerun's outcome.
    # A rerun that changes nothing must be an ERROR, not an echo.
    _prev = ledger.results.get(spec_id)
    _prev_ran_at = getattr(_prev, "ran_at", None)
    try:
        proc = sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                      cwd=str(Path(__file__).parent.parent), timeout=_timeout)
    except sp.TimeoutExpired:
        # An uncaught timeout used to crash the whole runner invocation and
        # leave the spec's STALE entry standing. A timeout is a result.
        res = Result(spec_id=spec_id, status=Status.ERROR,
                     message=f"timed out after {_timeout}s "
                             f"(budget {_spec.budget.value if _spec else '?'} "
                             f"x {getattr(_spec, 'seeds', 1)} seeds x2)")
        ledger.record(res)
        return res
    # The child wrote the ledger itself; re-read to see what it recorded.
    fresh = Ledger()
    ledger.results.update(fresh.results)
    res = fresh.results.get(spec_id)
    if res is None or getattr(res, "ran_at", None) == _prev_ran_at:
        tail = (proc.stderr or proc.stdout or "")[-300:].strip()
        res = Result(spec_id=spec_id, status=Status.ERROR,
                     message=f"child recorded nothing (rc={proc.returncode}): {tail}")
        ledger.record(res)
    return res


def cmd_run(ledger: Ledger, spec_ids: list[str]) -> int:
    failures = 0
    for sid in spec_ids:
        spec = BY_ID.get(sid)
        if not spec:
            print(f"unknown spec {sid}")
            failures += 1
            continue
        mod = _module_for(sid)
        if not mod:
            print(f"[{sid}] no implementation in experiments/tests/ — skipping")
            continue
        blocked = ledger.blocked_by(spec)
        if blocked:
            print(f"[{sid}] BLOCKED by {', '.join(blocked)}")
            failures += 1
            continue
        print(f"[{sid}] {spec.title} ... ", end="", flush=True)
        # Each spec runs in its OWN process. In-process the regression gate was
        # OOM-killed (exit 137): fifteen tests each constructing a model, with
        # Python holding every allocation until the run ended. On a box shared
        # with paying tenants that is not an inconvenience, it is a hazard.
        # A subprocess also isolates a crashing test from the ledger.
        res = _run_isolated(sid, ledger)
        print(f"{res.status.value} ({res.duration_s}s) {res.message}")
        if res.metrics:
            for k, v in list(res.metrics.items())[:6]:
                print(f"        {k} = {v}")
        if res.status is not Status.PASS:
            failures += 1
    return 1 if failures else 0


def cmd_render(ledger: Ledger) -> int:
    """Regenerate CHECKLIST.md FROM the ledger.

    The checklist is never hand-written. A capability appears as done only when
    a test that could have failed did not. This is the structural cure for a
    README that claimed eleven components were "Working" while none had ever
    received a gradient.
    """
    counts = ledger.summary()
    done, total = counts[Status.PASS.value], len(LADDER)
    names = {0: "HARNESS — can we measure anything?",
             1: "LEARNING PRIMITIVES — can each piece learn at all?",
             2: "COMPONENT vs NULL — does it beat the baseline?",
             3: "ABLATION — does it earn its parameters?",
             4: "COMPOSITION — does adding B break A?",
             5: "THE CLAIMS — the thesis stands or falls",
             6: "INTEGRATION"}
    box = {Status.PASS: "[x]", Status.FAIL: "[!]", Status.VOID: "[~]", Status.ERROR: "[!]",
           Status.BLOCKED: "[-]", Status.SKIP: "[~]", Status.NOT_RUN: "[ ]"}
    out = [
        "# Jack — the checklist",
        "",
        "**Generated by `python -m experiments.run render`. Do not edit by hand.**",
        "Every line here is backed by an experiment that could have failed;",
        "`experiments/ledger.json` holds the evidence.",
        "",
        f"## {done} / {total} demonstrated",
        "",
        "`[x]` proved · `[!]` failed, needs a fix · `[-]` blocked by a dependency · `[ ]` not run",
        "",
    ]
    cur = None
    for s_ in LADDER:
        if s_.tier != cur:
            cur = s_.tier
            out += ["", f"### Tier {cur} — {names.get(cur,'')}", ""]
        st = ledger.status(s_.id)
        r = ledger.results.get(s_.id)
        note = ""
        if st in (Status.FAIL, Status.VOID) and r and r.metrics:
            k = next(iter(r.metrics)), 
            note = "  — " + "; ".join(f"{k}={v}" for k, v in list(r.metrics.items())[:2])
        elif st is Status.BLOCKED and r:
            note = f"  — {r.message}"
        out.append(f"- {box[st]} **{s_.id}** {s_.title}{note}")
        out.append(f"      - _asserts:_ {s_.hypothesis}")
        out.append(f"      - _dies if:_ {s_.falsified_by}")
        if s_.kills:
            out.append(f"      - _then delete:_ {s_.kills}")
    Path("CHECKLIST.md").write_text("\n".join(out) + "\n")
    print(f"wrote CHECKLIST.md — {done}/{total} demonstrated")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="*", help="spec ids, or 'status' / 'next'")
    ap.add_argument("--tier", type=int)
    ap.add_argument("--gate", action="store_true", help="re-run all passing tests")
    args = ap.parse_args()
    ledger = Ledger()

    # status/next/render are read-only and must not block on a running experiment.
    if args.spec and args.spec[0] in ("status", "next", "blocked", "render"):
        return {"status": cmd_status, "next": cmd_next, "blocked": cmd_blocked,
                "render": cmd_render}[args.spec[0]](ledger)
    if not args.spec and not args.gate and args.tier is None:
        return cmd_status(ledger)

    # Fail fast on stale code. The guard used to live inside build_job, i.e.
    # AFTER the runner lock and any setup: T2.01 spent 70 minutes queued before
    # discovering that an unrelated edit (playground.py) had dirtied the tree.
    # A precondition that can be checked in milliseconds must not be checked
    # after an hour.
    if args.spec or args.tier is not None or args.gate:
        needs_gpu = any((BY_ID.get(x) or BY_ID.get("T0.01")).budget.value.startswith("gpu")
                        for x in (args.spec or []) if x in BY_ID)
        if needs_gpu:
            from .gpu import assert_ref_is_current
            try:
                assert_ref_is_current("main")
            except RuntimeError as e:
                print(f"Refusing to start: {e}")
                return 1

    if args.gate:
        ids = [s.id for s in LADDER if ledger.status(s.id) is Status.PASS]
        print(f"Regression gate: {len(ids)} previously-passing tests\n")
        with _exclusive(ids):
            return cmd_run(ledger, ids)
    if args.tier is not None:
        _tier_ids = [s.id for s in tier(args.tier)]
        with _exclusive(_tier_ids):
            return cmd_run(ledger, _tier_ids)
    if not args.spec or args.spec[0] == "status":
        return cmd_status(ledger)
    if args.spec[0] == "render":
        return cmd_render(ledger)
    with _exclusive(args.spec):
        return cmd_run(ledger, args.spec)


if __name__ == "__main__":
    sys.exit(main())
