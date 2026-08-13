#!/usr/bin/env python
"""Ladder runner.

    python -m experiments.run status          # the checklist, current state
    python -m experiments.run next            # what is legitimately runnable now
    python -m experiments.run blocked         # what is unreachable, and what frees it
    python -m experiments.run stale           # claims whose test changed since the run
    python -m experiments.run amend T2.01 --by T0.14 --reason "..." --status VOID
                                              # a change that did NOT come from a run,
                                              # recorded as one. Cannot write PASS/FAIL.
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
import hashlib
import importlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .protocol import (Ledger, Status, impl_deps_of, impl_sha_of,
                       module_path_for, staleness_of)
from .registry import BY_ID, LADDER, ready, tier

TESTS_DIR = Path(__file__).parent / "tests"
_REPO = Path(__file__).resolve().parent.parent
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


CPU_LOCK_B = "/tmp/jack-ladder-cpu-b.lock"   # the one overflow slot; see _exclusive


def _cpu_fraction(pid: int, window_s: float = 1.0):
    """Cores consumed by `pid` over `window_s` seconds, or None if unreadable.

    NOT `ps -o pcpu`, which is CPU averaged over the process's whole LIFETIME.
    That average is exactly wrong for the case this file cares about: a job
    that trained locally for an hour and then blocked on a remote poll still
    reads busy, and — the dangerous direction — a job that polled for three
    hours and has just begun local work still reads idle. Only a differenced
    sample says what a process is doing NOW.
    """
    hz = os.sysconf("SC_CLK_TCK")

    def _ticks():
        # /proc/pid/stat field 2 (comm) may contain spaces and parentheses, so
        # split after the last ')' — utime/stime are fields 14/15 overall.
        raw = open(f"/proc/{pid}/stat").read()
        fields = raw[raw.rindex(")") + 2:].split()
        return int(fields[11]) + int(fields[12])

    try:
        a = _ticks()
        time.sleep(window_s)
        b = _ticks()
    except (OSError, ValueError, IndexError):
        return None
    return (b - a) / hz / window_s


def _holders(lock_path: str):
    """Every live process holding `lock_path` open, with what it is doing.

    The lockfile's own PID line is unreliable (a pre-fix holder wrote nothing —
    the file was 0 bytes), so holders are found by scanning /proc for the open
    descriptor, which cannot go stale.
    """
    import glob
    found = []
    try:
        target = os.path.realpath(lock_path)
    except OSError:
        return found
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.path.realpath(fd) != target:
                continue
            pid = int(fd.split("/")[2])
            if pid == os.getpid():
                continue              # we hold the fd too; flock is what we lost
            argv = open(f"/proc/{pid}/cmdline", "rb").read().decode(
                "utf-8", "replace").split("\0")
            argv = [a for a in argv if a]
            age = os.popen(f"ps -o etime= -p {pid} 2>/dev/null").read().strip() or "?"
            specs = [a for a in argv if a in BY_ID]
            found.append({
                "pid": pid,
                "cmd": " ".join(argv),
                "age": age,
                "specs": specs,
                # A holder is "remote-only" when every ladder spec it names has
                # a gpu budget AND it names at least one. Anything else — a
                # --gate, a tier sweep, an unrecognised argv — is treated as
                # local CPU work, which is the safe direction.
                "remote_only": bool(specs) and all(
                    BY_ID[s].budget.value.startswith("gpu") for s in specs),
                "cores": _cpu_fraction(pid),
            })
        except (OSError, ValueError, IndexError):
            continue
    return found


def _lock_holder(lock_path: str):
    """Human-readable lines for `_holders` — say WHO holds the lock, what they
    are running, and whether they are actually using the CPU it protects.

    "Another run holds the lock (probably the hourly loop)" is a guess dressed
    as a diagnosis, and twice now it has been wrong in the same way. On
    2026-08-09 PG.8's strengthened check could not be re-recorded because a
    T2.01 run held this lock; hours later PG.7 hit the identical wall, and the
    holder turned out to be a T2.01 process started 26 minutes BEFORE the
    lock-split commit (8970638) that would have sent it to the GPU lock — so it
    sat at **0.0% CPU polling a remote GPU** while holding the LOCAL CPU-work
    lock. `_lock_for` cannot fix that case: a process that is already running
    cannot be re-routed, and every fix to it leaves a window of pre-fix
    processes behind. `_exclusive` acts on this; here it is only described.
    """
    out = []
    for h in _holders(lock_path):
        cores = "?" if h["cores"] is None else f"{h['cores']:.2f}"
        out.append(f"holder pid {h['pid']}  {cores} cores now  up {h['age']}  "
                   f"{h['cmd'][:90]}")
    return out or ["holder could not be identified from /proc."]


IDLE_CORES = 0.05      # a holder below this is not using the CPU this lock protects


@contextmanager
def _exclusive(spec_ids=()):
    """Serialise ALL ladder work, manual or looped.

    The hourly loop and a manual session raced and each wrote a different T0.07;
    one silently shadowed the other. The loop script already took this lock, but
    a human at a terminal did not, so the guard only protected one side. Holding
    it here means whoever starts second waits or skips, regardless of who they are.

    THE OVERFLOW SLOT. What this lock actually protects is 4 shared ARM cores
    from two torch processes; it was never about ledger integrity (`Ledger.record`
    takes its own fcntl lock and re-read-merge-writes — the T0.08 lesson). So a
    holder that is provably consuming no local CPU is not protecting anything,
    and blocking behind it costs real science: on 2026-08-09 an idle T2.01
    remote poll blocked PG.8, then PG.7, then PG.6, each time while the box sat
    at 4% CPU. `_lock_for` routes NEW gpu-only runs to a separate lock, but a
    process already running cannot be re-routed, so that fix always leaves a
    window of pre-fix holders behind — this is the part that closes the window.

    When every holder is (a) measured at <IDLE_CORES cores over a live sample,
    not a lifetime average, and (b) running only gpu-budget specs, we take ONE
    overflow slot instead of giving up. Two conditions, not one, and both are
    conservative: an unreadable /proc, an unrecognised argv, or any local work
    blocks exactly as before, and the overflow slot itself is exclusive, so the
    number of processes actually competing for the cores never exceeds one.
    """
    lock_path = _lock_for(spec_ids)
    for path, overflow in ((lock_path, False), (CPU_LOCK_B, True)):
        fh = open(path, "w")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            if overflow:
                print(f"  overflow slot {CPU_LOCK_B} is also held — waiting is correct.")
                break
            print(f"Another run holds {path}.")
            holders = _holders(path)
            for line in _lock_holder(path):
                print(f"  {line}")
            idle_remote = bool(holders) and all(
                h["remote_only"] and h["cores"] is not None and h["cores"] < IDLE_CORES
                for h in holders)
            if not idle_remote or path != RUN_LOCK:
                print("  Wait for it, or `touch .loop-paused` to stop the loop.")
                raise SystemExit(0)
            print("  ^ every holder is a remote-GPU poll using no local CPU. "
                  f"Proceeding on the overflow slot {CPU_LOCK_B}.")
            continue
        fh.write(f"{os.getpid()}\n"); fh.flush()
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        return
    raise SystemExit(0)

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

    The underscore before the slug is load-bearing: "me_1*" would also match
    me_10_*, so ME.1 and ME.10 would each see two implementations and raise.
    Hierarchical ids (ME.11 and its bakeoff arms ME.11.0/ME.11.A) defeat that
    underscore, so a longer spec id owns its own files. Both rules live in
    `protocol.module_path_for` — this is one call, not a second copy of them.
    """
    path = module_path_for(spec_id, strict=True)
    if path is None:
        return None
    return importlib.import_module(f"experiments.tests.{path.stem}")


def _module_path_for(spec_id: str):
    """The implementation FILE for a spec, without importing it."""
    return module_path_for(spec_id)


def stale_claims(ledger: Ledger) -> list:
    """Specs whose test FILE differs from the one that produced their entry.

    A ledger entry is a claim about a specific piece of code. Edit the test
    afterwards and the entry keeps asserting the old result under the new test's
    name — `LESSONS.md`'s "generated artifacts go stale silently", except the
    stale artifact is the scoreboard itself.

    Written 2026-08-09 the moment it bit. PG.8's observation check was
    strengthened (it had been comparing 78 identically-zero columns against 78
    identically-zero columns), verified at 3 seeds, and could NOT be re-recorded:
    a concurrent iteration held the runner lock on a long GPU job. For as long as
    that lock is held the ledger says PG.8 PASS about a file that no longer
    exists in that form, and nothing says so.

    Compares `Result.impl_sha`, not commits. The first attempt used "any commit
    touching the test since the recorded commit" and reported 15 of 54 entries
    stale — because a test is written, RUN, and only then committed, so the
    recorded commit predates the test's own first commit and every honest entry
    fires. A diagnostic with a 100% false-positive rate on healthy entries is
    worse than none: it trains its reader to ignore it.

    Returns (spec_id, status, kind, detail) where kind is "CHANGED" (the file
    hash moved), "UNVERIFIABLE" (the entry predates `impl_sha`),
    "UNVERIFIABLE_MOVED" (predates `impl_sha` AND a declared IMPL_DEPS
    dependency has commits after `ran_at` — the subset that bites) or "DIRTY"
    (the run's commit stamp ends in `+dirty`).

    DIRTY is the strictly worse cousin of CHANGED and was added 2026-08-10, one
    iteration after `env_stamp()` learned to write the flag. The flag alone was
    a fact nothing consumed — LESSONS.md's "a lesson that prescribes a guard is
    not a guard", in its second form: a SIGNAL that no organ reads is not a
    guard either. CHANGED says the file moved after the run, so the code that
    ran is still recoverable from the recorded commit. DIRTY says the run
    executed HEAD *plus* uncommitted edits, so the code that produced the entry
    exists in no commit at all and cannot be recovered by anyone, ever. It is
    reported ALONGSIDE the impl_sha verdict rather than instead of it: an entry
    can be both, and they are different facts about it.
    """
    out = []
    for s in LADDER:
        st = ledger.status(s.id)
        if st is Status.NOT_RUN:
            continue
        entry = ledger.results.get(s.id)
        path = _module_path_for(s.id)
        if entry is None or path is None:
            continue
        # THE RULE IS CALLED, NOT RESTATED. `borrow_metrics` refuses a stale
        # source through the same function, so the number a test may compute on
        # and the number this report calls current cannot drift apart — which
        # is exactly how the two impl_sha implementations diverged.
        for kind, detail in staleness_of(entry, path):
            out.append((s.id, st.value, kind, detail))
    return out


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
    _check_stale_detector(ledger)
    rows = stale_claims(ledger)
    changed = [x for x in rows if x[2] == "CHANGED"]
    moved = [x for x in rows if x[2] == "UNVERIFIABLE_MOVED"]
    moved_ids = {x[0] for x in moved}
    unknown = [x for x in rows if x[2] == "UNVERIFIABLE" and x[0] not in moved_ids]
    dirty = [x for x in rows if x[2] == "DIRTY"]
    if dirty:
        # Above the CHANGED block deliberately: this is the more serious of the
        # two and the scoreboard's top lines are what an iteration actually reads.
        print("  ! DIRTY STAMPS — the run's code exists in no commit:")
        for sid, st, _, detail in dirty:
            print(f"      {sid}  recorded {st}; {detail}. Re-run it from a "
                  f"clean tree.")
        print()
    if changed:
        print("  ! STALE CLAIMS — the test changed after the run that recorded it:")
        for sid, st, _, detail in changed:
            print(f"      {sid}  recorded {st}; {detail}. Re-run it — the entry "
                  f"is about older code.")
        print()
    if moved:
        # The subset of the pre-impl_sha entries that actually bites: the
        # certificate names a dependency, the dependency has moved, and the
        # alarm it declares structurally cannot fire (14th audit, B1).
        print("  ! UNPROTECTED CERTIFICATES — recorded before `impl_sha`, and a "
              "declared dependency\n    has since moved; `run stale` would read "
              "clean forever. Re-run these ON PURPOSE:")
        for sid, st, _, detail in moved:
            print(f"      {sid}  recorded {st}; {detail}.")
        print()
    if unknown:
        # Printed, not filed under "clean". The entry that MOTIVATED this guard
        # (PG.8, strengthened but un-re-runnable behind a held lock) is itself
        # in this bucket, because it was recorded before `impl_sha` existed. A
        # guard whose own motivating case reads green is the "guard built by
        # fixing one file leaves the file that motivated it unfixed" lesson
        # repeating; saying the number out loud is the cheapest way not to.
        print(f"  ? {len(unknown)} entr(y/ies) predate `impl_sha` and CANNOT be "
              f"checked for staleness — `run stale` lists them; a re-run fixes "
              f"each one.\n")
    print("  A capability is claimed ONLY by a PASS here. Nothing else counts.\n")
    return 0


def _check_stale_detector(ledger: Ledger) -> None:
    """Plant a known-stale entry and require the detector to find it.

    `stale_claims` returning "nothing is stale" and `stale_claims` never having
    looked are the same output, and this repo has already shipped an audit tool
    that came back clean on a known-bad input because its source extraction
    silently returned an empty set (T0.13). So the real function is run, on a
    real spec, against a real file, with one `impl_sha` deliberately wrong — and
    a detector that cannot see that refuses to report at all.

    TWO plants since 2026-08-10, one per bucket. The DIRTY bucket reads zero on
    today's ledger and will read zero for as long as every run starts from a
    clean tree — which is indistinguishable from a detector that cannot fire.
    That is the same shape as the CHANGED plant above and it gets the same
    treatment: a bucket whose known-positive has never been seen is not
    evidence of anything.
    """
    import copy

    victim = next((s.id for s in LADDER
                   if ledger.results.get(s.id) is not None
                   and _module_path_for(s.id) is not None), None)
    if victim is None:
        return
    # KNOWN-POSITIVE FOR THE DEPENDENCY HALF OF THE HASH, and the one check
    # that would have caught the writer/reader split outright: the reader must
    # actually SEE an `IMPL_DEPS` declaration somewhere in the ladder. The old
    # reader hashed test files alone, saw zero dependencies, and reported the
    # twelve specs that declare them stale in perpetuity — while every planted
    # probe below passed, because a test-file-only hash detects a test-file-only
    # edit perfectly well. A detector can be right about its own fixture and
    # blind to the scope it claims to cover.
    declared = [(sid, impl_deps_of(_module_path_for(sid)))
                for sid in (s.id for s in LADDER)
                if _module_path_for(sid) is not None]
    problems = [f"{sid}({p})" for sid, (_, p) in declared if p]
    if problems:
        raise RuntimeError(
            "IMPL_DEPS could not be read as a literal list in: "
            + ", ".join(problems) + " — the recorded sha covers files this "
            "scan cannot identify")
    if not any(deps for _, (deps, _) in declared):
        raise RuntimeError(
            "no spec in the ladder declares IMPL_DEPS as far as this scan can "
            "see; the dependency half of every impl_sha is invisible to it")

    for field, value, kind in (("impl_sha", "0" * 16,          "CHANGED"),
                               ("commit",   "0000000+dirty",   "DIRTY")):
        probe = copy.copy(ledger)
        probe.results = dict(ledger.results)
        planted = copy.copy(probe.results[victim])
        setattr(planted, field, value)   # a hash/stamp this entry cannot have
        probe.results[victim] = planted
        hit = [r for r in stale_claims(probe) if r[0] == victim and r[2] == kind]
        if not hit:
            raise RuntimeError(
                f"the stale detector did not flag a planted {kind} on {victim}; "
                "refusing to report a clean scan it may not have performed")


def cmd_stale(ledger: Ledger) -> int:
    _check_stale_detector(ledger)
    rows = stale_claims(ledger)
    changed = [r for r in rows if r[2] == "CHANGED"]
    moved = [r for r in rows if r[2] == "UNVERIFIABLE_MOVED"]
    moved_ids = {r[0] for r in moved}
    unknown = [r for r in rows if r[2] == "UNVERIFIABLE" and r[0] not in moved_ids]
    dirty = [r for r in rows if r[2] == "DIRTY"]
    if dirty:
        print(f"\n{len(dirty)} claim(s) recorded from a MODIFIED tree — the code "
              f"that ran is in no commit:\n")
        for sid, st, _, detail in dirty:
            print(f"  {sid:8} {st:7} {detail}")
        print("\nRe-run each from a clean tree. This is worse than CHANGED: "
              "there is no commit\nto go back to.")
    if not changed:
        print("\nNo stale claims — every verifiable entry names the test as it "
              "stands today.")
    else:
        print(f"\n{len(changed)} claim(s) recorded against code that has since "
              f"changed:\n")
        for sid, st, _, detail in changed:
            print(f"  {sid:8} {st:7} {detail}")
        print("\nRe-run these (or `--gate`). A ledger entry is a claim about a "
              "specific piece of code.")
    if moved:
        print(f"\n{len(moved)} UNPROTECTED certificate(s) — recorded before "
              f"`impl_sha`, and a declared\ndependency has since moved. The "
              f"alarm they declare cannot fire; re-run ON PURPOSE:\n")
        for sid, st, _, detail in moved:
            print(f"  {sid:8} {st:7} {detail}")
    # Reported, never hidden: a skipped item that leaves the numerator alone is
    # how a clean scan and a scan that never ran become the same number.
    print(f"\n{len(unknown)} further entr(y/ies) predate `impl_sha` and cannot "
          f"be checked at all (no declared\ndependency has moved); they become "
          f"verifiable on their next run.\n")
    return 0


def cmd_senses(ledger: Ledger) -> int:
    """Coverage of the HUMAN sensory inventory — the only report in this system
    whose standard comes from outside the repository.

    Every other organ measures us against what we wrote down, so a sense we
    never wrote down is invisible to all of them: on 2026-08-10 five of
    GOAL.md's constitutional senses had zero specs among 137 and no command
    could say so. See `experiments/senses.py`; gated as T0.20.

    Exit code is 0 even when senses are ABSENT. This reports a gap in ambition,
    not a broken build, and turning it into a red exit would tempt someone to
    shrink the inventory to make it green — which is precisely the failure it
    exists to catch.
    """
    from .senses import audit, render
    print(render(audit(ledger=ledger)))
    return 0


def cmd_coverage(ledger: Ledger) -> int:
    """Coverage of GOAL.md's constitutional commitments — is this the RIGHT ladder?

    `run status` says how much of the ladder is demonstrated; a commitment with
    no spec is invisible to it, to `run blocked`, and to every gate. See
    `experiments/coverage.py`; gated as T0.21.

    Coverage is DECLARED (`COVERS:` in a spec's notes), never inferred: the
    regex half of this file once credited the owner's "he builds a shelter" to
    a spec titled "The paraphrase eval set is HONEST...". Regex hits are
    printed as NOMINATIONS — work to do, never coverage.

    Nonzero exit means a commitment has NO declared spec, or a declaration
    names a commitment that does not exist. Both are cheap to fix and expensive
    to leave; "covered but not passing" is normal and exits 0.
    """
    from .coverage import check
    return check()


def cmd_verify(ledger: Ledger) -> int:
    """Re-judge every PASS from the record alone, and probe whether its gate
    actually reads its control. See `experiments/verify.py`; gated as T0.18.

    Costs no experiment: the ledger already stores the numbers and the repo
    already stores the thresholds, so the decision can simply be re-taken.
    """
    from .verify import UNDECLARED_CONTROL_BUDGET, assert_detector_works, collect, scan

    assert_detector_works()      # a scan that cannot see a planted defect is not reported
    r = scan(collect(ledger, exclude=("T0.18",)))

    print(f"\nRe-judged {r['verdicts_rejudged']} PASS entries from the record "
          f"alone; probed {r['controls_probed']} controls.\n")
    rows = [
        ("verdicts that no longer re-derive", r["verdict_disagreements"],
         r["disagreement_detail"]),
        ("gates that IGNORE their control", r["control_blind_specs"],
         r["control_blind_detail"]),
        ("controls declared but never run", r["declared_control_never_ran"],
         r["declared_never_ran_detail"]),
        ("gates that could not be replayed", r["unevaluable_gates"],
         r["unevaluable_detail"]),
        ("entries that could not be audited", r["unavailable_entries"],
         r["unavailable_detail"]),
    ]
    for label, n, detail in rows:
        mark = "  " if n == 0 else "! "
        print(f"  {mark}{label:38} {n}" + (f"   {detail}" if detail else ""))
    print(f"\n  ? controls run but NOT declared in the spec      "
          f"{r['undeclared_control_ran']} / {UNDECLARED_CONTROL_BUDGET} budget")
    if r["undeclared_ran_detail"]:
        print(f"      {r['undeclared_ran_detail']}")
    print("      The science is fine — each of these gates is measured above to "
          "read its\n      control. The DECLARATION is what rots: `Spec.control` "
          "is the field an\n      auditor greps, and a false negative there makes "
          "the grep useless.")
    if r["no_control_specs"]:
        print(f"\n  ? PASSes with NO control at all                 "
              f"{r['no_control_specs']}\n      {r['no_control_detail']}")
        print("      Probe B has nothing to say about these: there is no control "
              "to delete.\n      An existence claim whose gate was never shown "
              "capable of reporting the\n      bad case (OVERSIGHT §1.2).")
    if r["self_excluded_entries"]:
        print(f"\n  ? {r['self_excluded_entries']} entry self-excluded "
              f"({r['self_excluded_detail']}) — a spec cannot re-judge its own "
              f"entry;\n      that entry is written after the scan. Its gate is "
              f"exercised by T0.18's control.")
    print()
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


def _terminal_blockers(ledger: Ledger, ladder=None, by_id=None) -> dict:
    """For every spec, the ROOTS its unreachability actually rests on.

    A spec's immediate parent is almost never the answer. UB.1 reads as blocked
    by T4.01, which is blocked by T3.02, which is blocked by T2.01 = VOID — and
    only T2.01 can be acted on. Walking to the terminal blocker is what turns a
    list of 40 stuck specs into two things to fix.

    `ladder`/`by_id` are injectable so the ranking below can be checked against a
    graph whose answer is known — see `_RANKER_FIXTURE`.
    """
    ladder = LADDER if ladder is None else ladder
    by_id = BY_ID if by_id is None else by_id
    terminal: dict = {}

    def walk(sid: str, seen: frozenset) -> set:
        if sid in terminal:
            return terminal[sid]
        spec = by_id.get(sid)
        if spec is None or sid in seen:      # unknown dep, or a dependency cycle
            return {sid}
        roots: set = set()
        # ONE rule, asked through `Ledger.unsatisfied` — this loop used to
        # restate it as `status is Status.PASS`, which is the test `T0.22`
        # retired, so a spec resting on a STALE pass read as runnable here
        # while `borrow_metrics` VOIDed it the moment it ran.
        for d, _why in ledger.unsatisfied(spec):
            upstream = walk(d, seen | {sid})
            # A dependency that is itself stuck resolves to ITS roots; one that
            # is merely not-yet-run is a root of its own.
            roots |= upstream if upstream else {d}
        terminal[sid] = roots
        return roots

    for s in ladder:
        walk(s.id, frozenset())
    return terminal


def _rank_blockers(terminal: dict, ledger: Ledger, ladder=None) -> tuple:
    """Split "mentions this root" from "fixing this root alone frees it".

    The first version of this command ranked by MENTIONS, and mentions
    double-count: a spec resting on two roots is counted under both. It reported
    `T2.03 blocks 11` and the next iteration's hand-off line duly named T2.03
    "the largest unblocking available without a GPU". Nine of those eleven are
    UB.1-8 + T4.01, which also rest on `T2.01 = VOID` — so fixing T2.03 frees
    **two** specs, not eleven. The ranking sent the loop at the wrong unit.

    The converse error is in the same number: PG.6, PG.7, LC.02 and PS.01 read
    "blocks 7, 7, 4, 4" and each frees **nothing** alone, because their
    dependents need a co-requisite root fixed too. A marginal value of zero was
    being presented as the third-best move on the board.

    So: `frees` (the marginal value of this fix alone) is the ranking key,
    `blocks` is still reported because a root that blocks many and frees none is
    exactly the signal that a PAIR is needed, and `groups` names those pairs.
    """
    ladder = LADDER if ladder is None else ladder
    mentions: dict = {}
    frees: dict = {}
    groups: dict = {}
    for s in ladder:
        if ledger.status(s.id) is Status.PASS:
            continue
        roots = {r for r in terminal.get(s.id, set()) if r != s.id}
        if not roots:                              # runnable now, not blocked
            continue
        for root in roots:
            mentions.setdefault(root, []).append(s.id)
        if len(roots) == 1:
            frees.setdefault(next(iter(roots)), []).append(s.id)
        else:
            groups.setdefault(frozenset(roots), []).append(s.id)
    return mentions, frees, groups


# A graph whose answer is known, checked on every `blocked` run. X is a root that
# blocks two specs and frees one; W blocks one and frees NOTHING alone, because
# Z needs both. That second case is the defect this fixture exists to catch, and
# a ranker with the pre-fix "rank by mentions" logic puts W above nothing at all
# while claiming it is worth one spec. (LESSONS.md: every audit tool needs a
# known-positive fixture it must flag, exercising the same code path as the real
# scan — so this runs `_terminal_blockers`/`_rank_blockers` themselves, not a
# tidied restatement.)
# `S` is deliberately a REAL spec id: the staleness half of the rule resolves an
# implementation FILE through `module_path_for`, so a synthetic id would skip the
# very branch this fixture is here to check. Its planted `impl_sha` is all zeroes,
# which cannot be any file's hash, so the known answer holds whatever PG.1's real
# entry or source happens to be today.
_STALE_ID = "PG.1"


def _ranker_fixture() -> tuple:
    from .protocol import Spec, Budget

    def stub(sid, deps):
        return Spec(sid, 0, sid, "h", "f", "n", "m", Budget.CPU_FAST, depends_on=deps)

    ladder = [stub("X", []), stub("W", []), stub("Y", ["X"]), stub("Z", ["X", "W"]),
              stub(_STALE_ID, []), stub("V", [_STALE_ID])]
    return ladder, {s.id: s for s in ladder}


def _fixture_ledger() -> Ledger:
    """The fixture's ledger is a REAL `Ledger`, pointed at a path that does not
    exist and never written. It used to be a duck-typed stub exposing `status`
    alone, which meant the fixture could not see the freshness half of the
    dependency rule at all — the stub would have kept passing while the rule it
    is guarding changed underneath it. (T0.22's `_ledger_with` pattern.)"""
    from .protocol import Result
    led = Ledger(path=Path("/nonexistent/ranker_fixture_never_written.json"))
    led.results = {_STALE_ID: Result(
        spec_id=_STALE_ID, status=Status.PASS, metrics={}, seeds=[0],
        commit="1234567", ran_at="2026-08-11T00:00:00", impl_sha="0" * 16)}
    return led


def _check_ranker(ledger: Ledger) -> None:
    """Refuse to print a ranking the ranker cannot get right on a known graph."""
    ladder, by_id = _ranker_fixture()
    fixt = _fixture_ledger()
    terminal = _terminal_blockers(fixt, ladder=ladder, by_id=by_id)
    mentions, frees, groups = _rank_blockers(terminal, fixt, ladder=ladder)
    expect = (
        sorted(mentions.get("X", [])) == ["Y", "Z"],
        sorted(mentions.get("W", [])) == ["Z"],
        sorted(frees.get("X", [])) == ["Y"],
        frees.get("W", []) == [],
        groups.get(frozenset({"X", "W"})) == ["Z"],
        # KNOWN ANSWER for the freshness half: a PASS whose implementation hash
        # has moved does NOT satisfy the specs that depend on it.
        sorted(mentions.get(_STALE_ID, [])) == ["V"],
        sorted(frees.get(_STALE_ID, [])) == ["V"],
    )
    if not all(expect):
        raise RuntimeError(
            "the blocked-ranker failed its own fixture "
            f"(mentions={mentions}, frees={frees}, groups={ {set(k): v for k, v in groups.items()} }); "
            "refusing to print a ranking that cannot be trusted")


def cmd_amend(ledger: Ledger, args) -> int:
    """Record a change to a ledger entry that did NOT come from a run.

    Written 2026-08-10 for the overseer's RANK 1 finding: the ledger had been
    hand-edited twice (T2.01 FAIL->VOID, T2.02 restated) while its own header
    said hand-editing was forbidden, so nothing in the file distinguished a
    runner-recorded verdict from an agent-restated one. This keeps the runner
    the only writer and makes the edit part of the record instead of invisible
    in it. `Ledger.AMENDABLE` is the teeth: an amendment can only reach a
    status that asserts nothing.
    """
    if len(args.spec) != 2:
        print("usage: run amend <SPEC> --by <SPEC-or-finding> --reason '...' "
              "[--status VOID|SKIP|NOT_RUN] [--unknown-history]")
        return 2
    spec_id = args.spec[1]
    try:
        status = Status(args.status) if args.status else None
        row = ledger.amend(spec_id, by=args.by or "", reason=args.reason or "",
                           status=status, unknown_history=args.unknown_history)
    except (ValueError, KeyError) as e:
        print(f"Refusing to amend {spec_id}: {e}")
        return 1
    note = row["amended"][-1]
    print(f"{spec_id}: amended by {note['by']} at {note['at']} ({note['commit']})")
    for c in note["changes"]:
        print(f"    {c['field']}: {c['from']!r} -> {c['to']!r}")
    print(f"    reason: {note['reason']}")
    return 0


def cmd_blocked(ledger: Ledger) -> int:
    """What can this ladder NEVER do, and why — the converse of `next`.

    Written 2026-08-09 because the overseer had to walk the dependency graph by
    hand to discover that 29% of the ladder was dead behind two VOIDs, and that
    the dead set was precisely GOAL.md's headline: all 7 curiosity specs, all 16
    unison specs, all of Tiers 3, 4 and 5. `next` answers "what can I do"; until
    now nothing answered "what is unreachable, and what one fix would free it".
    LESSONS.md carried that as advice to humans. This makes it a command.
    """
    _check_ranker(ledger)
    terminal = _terminal_blockers(ledger)
    mentions, frees, groups = _rank_blockers(terminal, ledger)

    if not mentions:
        print("Nothing is blocked — every unrun spec has its dependencies passing.")
        return 0

    def _st(root):
        """The status, and — if it is a PASS that no longer describes the code —
        SAY SO. A root printed bare as `PASS` is unreadable: the reader's next
        question is why a passing spec is blocking anything."""
        if root not in BY_ID:
            return "UNKNOWN-SPEC"
        st = ledger.status(root)
        if st is Status.PASS:
            path = module_path_for(root)
            entry = ledger.results.get(root)
            if path and entry and any(k in ("DIRTY", "CHANGED")
                                      for k, _ in staleness_of(entry, path)):
                return "PASS but STALE — re-run it"
        return st.value

    ranked = sorted(mentions.items(),
                    key=lambda kv: (-len(frees.get(kv[0], [])), -len(kv[1])))
    total = len({sid for ids in mentions.values() for sid in ids})
    print(f"\n{total} of {len(LADDER)} specs are unreachable. Terminal blockers, "
          f"ranked by what fixing ONE of them alone would free:\n")
    for root, ids in ranked:
        title = BY_ID[root].title if root in BY_ID else "(not in the registry)"
        f = sorted(frees.get(root, []))
        print(f"  {root} = {_st(root)}  frees {len(f)}  (blocks {len(ids)})  — {title}")
        print(f"        frees:  {', '.join(f) if f else 'NOTHING on its own'}")
        rest = sorted(set(ids) - set(f))
        if rest:
            print(f"        also blocks (needs a co-requisite too): {', '.join(rest)}")
        print()

    if groups:
        print("  CO-REQUISITE SETS — no single fix frees these; the whole set must go:\n")
        for roots, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            names = " + ".join(f"{r}={_st(r)}" for r in sorted(roots))
            print(f"    {names}  frees {len(ids)}: {', '.join(sorted(ids))}")
        print()

    summary = "; ".join(
        f"{root}={_st(root)} frees {len(frees.get(root, []))}/blocks {len(ids)}"
        for root, ids in ranked)
    print(f"  SUMMARY: {summary}\n")
    return 0


def _dependency_order(ids) -> list:
    """Sort a batch so no spec runs before a dependency that is also in it.

    Written 2026-08-11, the same hour dependency satisfaction started asking the
    freshness question — because that change made batch ORDER able to destroy
    evidence. `--gate` re-runs every PASS in `LADDER` order, and a run whose
    dependency is unsatisfied records BLOCKED. Today `PS.01` is a stale PASS and
    `XL.00` depends on it: reaching `XL.00` first writes BLOCKED over a PASS
    that was legitimately earned, and the re-run of `PS.01` that would have
    cleared it happens five specs later. A certificate deleted by an ordering
    artifact is the thing law 4 exists to prevent. Under the old rule every
    dependency of a PASS was itself a PASS, so the gate never needed to care.

    Stable: ties keep the caller's order, so the tier/LADDER sequence survives
    wherever dependencies do not constrain it. Cycles and out-of-batch
    dependencies are left where they are rather than raising — this is a
    convenience ordering, not a validator, and `run blocked` is where a cycle
    is supposed to surface.
    """
    want = list(dict.fromkeys(ids))
    inside = set(want)
    out, placed = [], set()

    def emit(sid, seen):
        if sid in placed or sid in seen:
            return
        spec = BY_ID.get(sid)
        for d in (spec.depends_on if spec else []):
            if d in inside:
                emit(d, seen | {sid})
        if sid not in placed:
            placed.add(sid)
            out.append(sid)

    for sid in want:
        emit(sid, frozenset())
    return out


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
        "cpu<1min": 300, "cpu<10min": 1800, "cpu<2h": 9000, "cpu<48h": 172800,
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
        blocked = ledger.unsatisfied(spec)
        if blocked:
            print(f"[{sid}] BLOCKED by "
                  + ", ".join(f"{d} ({why})" for d, why in blocked))
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


#: The read-only sub-commands, named ONCE. They used to be a tuple in the
#: dispatch test and a dict in the dispatch itself; a word present in one and
#: absent from the other is how a command silently becomes "not a command".
READ_ONLY_COMMANDS = {"status": cmd_status, "next": cmd_next,
                      "blocked": cmd_blocked, "render": cmd_render,
                      "stale": cmd_stale, "verify": cmd_verify,
                      "senses": cmd_senses, "coverage": cmd_coverage}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="*",
                    help="spec ids, or status / next / blocked / stale / verify / render")
    ap.add_argument("--tier", type=int)
    ap.add_argument("--gate", action="store_true", help="re-run all passing tests")
    ap.add_argument("--by", help="amend: the spec or finding that motivates the change")
    ap.add_argument("--reason", help="amend: why, in a sentence")
    ap.add_argument("--status", help="amend: new status (VOID, SKIP or NOT_RUN only)")
    ap.add_argument("--unknown-history", action="store_true",
                    help="amend: this entry's attempt count is not reconstructible")
    args = ap.parse_args()
    ledger = Ledger()

    if args.spec and args.spec[0] == "amend":
        return cmd_amend(ledger, args)

    # status/next/render are read-only and must not block on a running experiment.
    if args.spec and args.spec[0] in READ_ONLY_COMMANDS:
        return READ_ONLY_COMMANDS[args.spec[0]](ledger)
    if not args.spec and not args.gate and args.tier is None:
        return cmd_status(ledger)

    # ARGV IS A SPEND. Everything below this line can start an experiment, and
    # for a `gpu<*>` spec that means charging the weekly quota, so an argv this
    # runner does not fully understand must stop here rather than run the part
    # it recognised. See `t0_23_argv_is_not_a_spend.py` for the scar.
    unknown = [x for x in (args.spec or []) if x not in BY_ID]
    if unknown:
        print("Refusing to run: unrecognised argument(s): " + ", ".join(unknown))
        print("Commands: " + ", ".join(sorted(READ_ONLY_COMMANDS)) + ", amend.")
        print("Everything else must be a spec id. Nothing was run.")
        return 2

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
        ids = _dependency_order([s.id for s in LADDER
                                 if ledger.status(s.id) is Status.PASS])
        print(f"Regression gate: {len(ids)} previously-passing tests\n")
        with _exclusive(ids):
            return cmd_run(ledger, ids)
    if args.tier is not None:
        _tier_ids = _dependency_order([s.id for s in tier(args.tier)])
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
