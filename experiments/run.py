#!/usr/bin/env python
"""Ladder runner.

    python -m experiments.run status          # the checklist, current state
    python -m experiments.run next            # what is legitimately runnable now
    python -m experiments.run blocked         # what is unreachable, and what frees it
    python -m experiments.run stale           # claims whose test changed since the run
    python -m experiments.run ratchets        # standing-red ratchet counters vs their
                                              # committed readings; `ratchets record`
                                              # refreshes experiments/ratchet_readings.json
    python -m experiments.run amend T2.01 --by T0.14 --reason "..." --status VOID
                                              # a change that did NOT come from a run,
                                              # recorded as one. Cannot write PASS/FAIL.
    python -m experiments.run T0.02           # run one experiment
    python -m experiments.run --tier 0        # run a whole tier, in order
    python -m experiments.run --gate          # re-run every PASSing test (regression)
    python -m experiments.run --gate --max-budget cpu<10min
                                              # bounded regression sweep: only PASSes at
                                              # or below the cost class; the excluded
                                              # stamps are printed, never silently skipped

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
import stat as stat_mod
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .protocol import (GATE_DIRTY_FLAG, Budget, Ledger, Result, Status,
                       dirty_recoverability, gate_precondition,
                       impl_deps_of, impl_sha_of, is_code_dirt,
                       module_path_for, porcelain_path, spec_drift,
                       staleness_of, tree_reconstructing_sha,
                       working_tree_porcelain)
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


def _proc_tree(pid: int):
    """`pid` and every live descendant, from one /proc scan. Raises on trouble.

    Built for `_cpu_fraction`, which measured the wrong process for as long as
    it has existed — see there.
    """
    kids: dict = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            raw = open(f"/proc/{entry}/stat").read()
            # fields after comm: state(0) ppid(1) ... utime(11) stime(12)
            f = raw[raw.rindex(")") + 2:].split()
            kids.setdefault(int(f[1]), []).append(
                (int(entry), int(f[11]) + int(f[12])))
        except (OSError, ValueError, IndexError):
            continue          # a process that exited mid-scan is not a failure
    try:
        root_raw = open(f"/proc/{pid}/stat").read()
        rf = root_raw[root_raw.rindex(")") + 2:].split()
    except (OSError, ValueError, IndexError):
        raise OSError(f"pid {pid} unreadable")
    out = {pid: int(rf[11]) + int(rf[12])}
    frontier = [pid]
    while frontier:
        for child, ticks in kids.get(frontier.pop(), []):
            if child not in out:
                out[child] = ticks
                frontier.append(child)
    return out


def _cpu_fraction(pid: int, window_s: float = 1.0):
    """Cores consumed by `pid` AND ITS DESCENDANTS over `window_s`, or None.

    NOT `ps -o pcpu`, which is CPU averaged over the process's whole LIFETIME.
    That average is exactly wrong for the case this file cares about: a job
    that trained locally for an hour and then blocked on a remote poll still
    reads busy, and — the dangerous direction — a job that polled for three
    hours and has just begun local work still reads idle. Only a differenced
    sample says what a process is doing NOW.

    AND IT MEASURED THE WRONG PROCESS (builder, 2026-08-30, found by being
    misled by it). `run.py` does not do its work in the process that holds the
    lock: `_module_for(...).run(...)` executes in a CHILD, so the holder's own
    utime+stime stays near zero for the entire run. A `BA.03` registered run
    with its worker at a full core for six hours printed `0.00 cores now` in
    the lock message, and the reader's first conclusion was that it had hung.

    That is a misleading display; the load-bearing half is worse. `_exclusive`
    steals the overflow slot when every holder is `remote_only` AND under
    `IDLE_CORES`, and its docstring says *"two conditions, not one, and both are
    conservative"*. The second condition could never fail for a `run.py` holder,
    because it read a supervisor that never computes — so a GPU-labelled run
    genuinely burning local cores (preprocessing before submit, a CPU fallback
    path) could have a second torch process started beside it on four shared
    ARM cores. A decorative gate is T0.13's whole subject, and this one was
    decorative in the permissive direction.

    A DESCENDANT THAT VANISHES MID-WINDOW RETURNS None, not a smaller number.
    Its ticks are lost, so the honest answer is "unreadable" — and None is the
    conservative reading everywhere this is used: the display prints `?` and
    `_exclusive`'s steal requires `cores is not None`, so it blocks. Reporting
    the survivors' sum would under-report exactly when the tree is churning.
    """
    hz = os.sysconf("SC_CLK_TCK")
    try:
        a = _proc_tree(pid)
        time.sleep(window_s)
        b = _proc_tree(pid)
    except (OSError, ValueError, IndexError):
        return None
    if set(a) - set(b):                 # a descendant exited: ticks unaccounted
        return None
    # Pids new in `b` started inside the window, so all their ticks are ours.
    delta = sum(t - a.get(p, 0) for p, t in b.items())
    return max(delta, 0) / hz / window_s


def _cpu_fraction_fixture(hz: int | None = None) -> list:
    """Known-answer battery for `_cpu_fraction`'s arithmetic, stubbing the
    /proc scan so it is deterministic and costs no processes.

    Written with the fix, per LESSONS' *"a test of the detector is not a test of
    the alarm"*: this function is not the alarm, so `main` calls it and prints
    its complaints, which is the only reason it can go red where anyone sees it.

    Case 1 is the whole bug — an idle root with a busy descendant. Against the
    single-pid version it reads 0.00, which is what `_exclusive` treats as
    "not using the CPU this lock protects".
    """
    global _proc_tree
    real, fails = _proc_tree, []
    hz = hz or os.sysconf("SC_CLK_TCK")
    # One core for `W` seconds is `N` ticks. `N` is chosen first and the window
    # derived from it, so the arithmetic is exact at any SC_CLK_TCK — sizing the
    # window first gave `int(hz * W) == 0` and a battery that read 0.0 for
    # everything, which is the value it exists to catch.
    N, W = 2, 2.0 / hz
    cases = [
        # label, sample A, sample B, expected cores (None = unreadable)
        ("idle root, busy child — the defect",
         {1: 10, 2: 500}, {1: 10, 2: 500 + N}, 1.0),
        ("genuinely idle tree", {1: 10, 2: 500}, {1: 10, 2: 500}, 0.0),
        ("busy root, no children", {1: 10}, {1: 10 + N}, 1.0),
        ("a child born inside the window counts all its ticks",
         {1: 10}, {1: 10, 2: N}, 1.0),
        # A descendant that exits takes its ticks with it. Reporting the
        # survivors would UNDER-report, and under-reporting is what lets the
        # overflow slot be stolen from a busy tree.
        ("a vanished descendant is unreadable, not idle",
         {1: 10, 2: 500}, {1: 10}, None),
        ("two busy children sum", {1: 0, 2: 0, 3: 0},
         {1: 0, 2: N, 3: N}, 2.0),
    ]
    try:
        for label, a, b, want in cases:
            seq = iter((a, b))
            _proc_tree = lambda _pid, _s=seq: next(_s)
            got = _cpu_fraction(1, window_s=W)
            ok = (got is None) if want is None else (
                got is not None and abs(got - want) < 0.02)
            if not ok:
                fails.append(f"_cpu_fraction: {label} -> want {want}, "
                             f"got {got}")
        # The root being unreadable must stay None — the pre-existing contract.
        def _raise(_pid):
            raise OSError("gone")
        _proc_tree = _raise
        if _cpu_fraction(1, window_s=W) is not None:
            fails.append("_cpu_fraction: an unreadable root is None, so the "
                         "overflow steal blocks and the display prints ?")
    finally:
        _proc_tree = real
    return fails


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
        # Acquire, with a ghost guard (59th audit B6): the holder now
        # unlinks its lock file on clean exit — so a stale pid can no
        # longer be quoted as a receipt, and a lock file that EXISTS while
        # nothing flocks it now means an unclean exit worth reading. The
        # race the unlink opens: a contender that opened the OLD inode just
        # before the unlink can acquire a flock nobody else contends while
        # a fresh process creates and locks the new inode — two holders,
        # which on 4 shared cores is the exact scar this lock exists for.
        # So after acquiring, verify the path still names our inode; if
        # not, reopen and try again. `contended = None` means we hold it.
        contended = None
        for _ghost_try in range(3):
            contended = None
            fh = open(path, "w")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                contended = e
                break
            try:
                if os.fstat(fh.fileno()).st_ino == os.stat(path).st_ino:
                    break
            except OSError:
                pass
            fh.close()
            contended = BlockingIOError("lock file kept moving underfoot")
        if contended is not None:
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
            # Unlink BEFORE unlocking, while nobody else can hold it, and
            # only if the path still names our inode (59th audit B6): a
            # clean exit leaves no file, so the pid inside a file that
            # exists belongs to a holder that crashed — a receipt, finally.
            try:
                if os.stat(path).st_ino == os.fstat(fh.fileno()).st_ino:
                    os.unlink(path)
            except OSError:
                pass
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

    IMPORT ONLY TO RUN, never to list. `status`/`next` ask this question for
    many specs in one process, and test modules are not import-compatible in
    bulk: a module that calls `ensure_gl()` at import (sh_01) raises if any
    earlier-listed module already imported mujoco bare, so which subset a
    listing walks decides whether the listing survives. Existence questions go
    through `module_path_for(strict=True)` — same duplicate check, no import.

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
    executed HEAD *plus* uncommitted edits, so the recorded commit does not
    name what ran. It is reported ALONGSIDE the impl_sha verdict rather than
    instead of it: an entry can be both, and they are different facts about it.

    THIS PARAGRAPH USED TO END "exists in no commit at all and cannot be
    recovered by anyone, ever", and that was an inference from the stamp, not
    a measurement of the row (corrected 2026-09-13). The stamp is tree-wide
    while `impl_sha` is per-spec, so a run whose own test file and IMPL_DEPS
    were committed still stamps `+dirty` when anything else in the tree is
    modified — which is the ordinary case for this loop. `PL.02` was exactly
    that, and `preserve_impl_bytes` (2026-08-30) had already falsified the
    "ever" for a second class of rows. `dirty_recoverability` decides which of
    the four states a dirty row is in; the DIRTY kind still fires in all of
    them, and every consumer still refuses on it.
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


def drifted_claims(ledger: Ledger) -> list:
    """PASS rows whose SPEC TEXT moved after the run that recorded them.

    `stale_claims`' sibling, and the half nobody had. A ledger entry is a claim
    about a specific piece of code AND about a specific set of words; edit the
    words afterwards and the entry keeps asserting the old verdict under the
    new claim's name. That happened: an owner ruling amended `LC.01`'s
    `falsified_by` on 2026-08-24, the amendment itself said *"Requires a re-run
    to re-buy the certificate under the amended text"*, and five days later the
    row still read PASS at `ran_at 2026-08-09` with no instrument able to ask
    (46th audit, B1). The sentence naming the debt was the only record of it.

    PASS only, and it is the whole point rather than a convenience: a PASS
    under superseded words CLAIMS A CAPABILITY the project never bought. A FAIL
    or VOID under superseded words refutes a claim nobody makes any more, which
    is untidy and asserts nothing.

    Returns `(spec_id, status, kind, detail)` — the same shape `stale_claims`
    returns, so the two blocks in `cmd_status` read the same way. Kinds are
    `spec_drift`'s: SPEC_CHANGED (positive evidence, re-run it) and
    SPEC_UNSTAMPED (the row predates the field — unknown, never counted clean).
    """
    out = []
    for s in LADDER:
        if ledger.status(s.id) is not Status.PASS:
            continue
        entry = ledger.results.get(s.id)
        if entry is None:
            continue
        # Called, never restated — the `staleness_of` rule, for the same
        # reason: two implementations of "the same" hash diverged once here
        # already and every IMPL_DEPS spec read stale forever.
        for kind, detail in spec_drift(entry, s):
            out.append((s.id, Status.PASS.value, kind, detail))
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
        impl = "" if module_path_for(s.id, strict=True) else "  (not implemented)"
        print(f"    [{MARK[st]}] {s.id}  {s.title}{impl}")
    print(f"\n  {counts}\n")
    # CPU day-meter visibility (68th audit B4): a budget refusal returns
    # UNRECORDED by design (tenant protection is not a measurement of the
    # spec), so a foreclosed day produces no FAIL, no VOID and no number —
    # unless it is derived here. Live arithmetic, no persistence: the same
    # numbers gate_cpu_child computes at the moment it refuses.
    from .cpu_budget import (CPU_DAY_CEILING_S, CpuBudget, class_slack,
                             foreclosed_now)
    _cpu_rem = CpuBudget().remaining_s()
    _cpu_fore = foreclosed_now()
    if _cpu_fore:
        _ids = " ".join(_cpu_fore[:8]) + (" …" if len(_cpu_fore) > 8 else "")
        print(f"  ! CPU DAY BUDGET: {CPU_DAY_CEILING_S - _cpu_rem:.0f}s of "
              f"{CPU_DAY_CEILING_S:.0f}s used today; {len(_cpu_fore)} cpu "
              f"spec(s) currently unaffordable until midnight:\n"
              f"      {_ids}")
        # 70th audit B4: the count above GROWS with the registry and has no
        # floor and no denominator, so it cannot say whether a class is one
        # spec short or shut. Slack is what makes it actionable — the day's
        # spend at which a class starts losing members. Instrumentation only:
        # CPU_DAY_CEILING_S is D20's and the owner's, and nothing here moves it.
        print("    SLACK PER CLASS — slack = ceiling − the class's largest "
              "live child estimate,\n    i.e. the spend at which the class "
              "starts foreclosing (D20 owns the ceiling):")
        for _r in class_slack():
            _state = ("CLOSING" if _r["used_s"] > _r["full_slack_s"]
                      else "over" if _r["used_s"] > _r["slack_s"] else "ok")
            print(f"      {_r['budget']:<10s} slack {_r['slack_s']:8.0f}s "
                  f"({_r['slack_s'] / 3600.0:5.2f}h)  spent "
                  f"{_r['used_s']:8.0f}s ({_r['used_s'] / 3600.0:5.2f}h)  "
                  f"{_state:<7s} {_r['n_foreclosed']}/{_r['n']} unaffordable"
                  + (f", {_r['n_unmeasured']} never run"
                     if _r["n_foreclosed"] else ""))
        print()
    _check_orphan_detector()
    orphans = gpu_orphans()
    if orphans:
        # Above the staleness blocks deliberately: this is paid-for compute
        # sitting unharvested RIGHT NOW, and the recovery lane decays (Kaggle
        # kernels expire, colab has no reattach at all).
        print("  ! ORPHANED DISPATCHES — an attempt row with no result row and "
              "a dead watcher pid.\n    The kernel may have completed; the "
              "record did not. Recover, do not re-dispatch:")
        for o in orphans:
            phase = f" ({o['spec_phase']})" if o.get("spec_phase") else ""
            cmd = (f"JACK_REUSE_KERNEL={o['slug']} scripts/dispatch.sh "
                   f"{o['spec']}"
                   if o.get("backend") == "kaggle" and o.get("slug")
                   and not o.get("spec_phase")
                   else "no reattach lane — harvest post-hoc (see CLAUDE.md)")
            print(f"      {o['spec']}{phase}  {o['backend']} attempt "
                  f"{o['iso']}, watcher pid {o['pid']} dead.  {cmd}")
        print()
    _check_stale_detector(ledger)
    rows = stale_claims(ledger)
    changed = [x for x in rows if x[2] == "CHANGED"]
    unstamped_changed = [x for x in rows if x[2] == "UNSTAMPED_CHANGED"]
    intact = [x for x in rows if x[2] == "UNSTAMPED_INTACT"]
    moved = [x for x in rows if x[2] == "UNVERIFIABLE_MOVED"]
    unknown = [x for x in rows if x[2] == "UNVERIFIABLE"]
    dirty = [x for x in rows if x[2] == "DIRTY"]
    if dirty:
        # Above the CHANGED block deliberately: this is the more serious of the
        # two and the scoreboard's top lines are what an iteration actually reads.
        # The header and the prescription both used to be unconditional — "the
        # run's code exists in no commit", then "Re-run it from a clean tree"
        # — and on 2026-09-13 that was false for the only row in the block
        # (PL.02's implementation reconstructs at the very commit the row is
        # stamped at). `dirty_recoverability` now writes the detail AND the
        # owed action, per row, so this prints neither on its own authority.
        print("  ! DIRTY STAMPS — the runner could not name what it executed:")
        for sid, st, _, detail in dirty:
            print(f"      {sid}  recorded {st}; {detail}.")
        print()
    if changed:
        print("  ! STALE CLAIMS — the test changed after the run that recorded it:")
        for sid, st, _, detail in changed:
            print(f"      {sid}  recorded {st}; {detail}. Re-run it — the entry "
                  f"is about older code.")
        print()
    if unstamped_changed:
        # Declaration-free staleness (15th audit, B1): no impl_sha to compare,
        # but git can answer anyway, and the answer is "the file moved". These
        # sat inside "cannot be checked" while being the only unstamped entries
        # that actually bite.
        print("  ! STALE PRE-impl_sha CLAIMS — git shows the test file changed "
              "since the run\n    (declaration-free content check). Re-run "
              "these ON PURPOSE:")
        for sid, st, _, detail in unstamped_changed:
            print(f"      {sid}  recorded {st}; {detail}.")
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
    n_unstamped = len(unstamped_changed) + len(intact) + len(unknown)
    if n_unstamped:
        # Denominator alongside every count (15th audit, B1). The intact set is
        # printed, not filed under "clean": a re-run still upgrades each to a
        # real stamp, and content identity says nothing about undeclared
        # dependencies. The entry that MOTIVATED the original guard (PG.8,
        # strengthened but un-re-runnable behind a held lock) lived in this
        # population.
        declare = _unstamped_deps_denominator(unstamped_changed, intact, unknown)
        print(f"  ? {n_unstamped} entr(y/ies) predate `impl_sha`: "
              f"{len(unstamped_changed)} stale by content (above), "
              f"{len(intact)} verified byte-identical by git, {len(unknown)} "
              f"unanswerable;\n    {declare} of {n_unstamped} declare "
              f"IMPL_DEPS. A re-run upgrades each to a real stamp.\n")
    if unknown:
        print(f"  ? {len(unknown)} of those could not be checked even by "
              f"content — `run stale` prints why.\n")
    # The claim-text half of the same question (46th audit B1). Printed next to
    # the code half on purpose: an iteration reads this block and nothing else,
    # and a defect reported somewhere the loop does not look is a defect that
    # was written down rather than detected (LESSONS.md, 2026-08-29).
    drift = drifted_claims(ledger)
    spec_changed = [x for x in drift if x[2] == "SPEC_CHANGED"]
    spec_unstamped = [x for x in drift if x[2] == "SPEC_UNSTAMPED"]
    if spec_changed:
        print("  ! DRIFTED CLAIMS — the SPEC TEXT changed after the PASS that "
              "bought it:")
        for sid, st, _, detail in spec_changed:
            print(f"      {sid}  recorded {st}; {detail}. Re-run it — the "
                  f"certificate is against words that no longer exist.")
        print()
    if spec_unstamped:
        print(f"  ? {len(spec_unstamped)} PASS row(s) predate `spec_sha`: "
              f"whether the claim text moved since\n    cannot be answered "
              f"from the record. Not back-filled — the sha of today's words "
              f"proves\n    nothing about a run from before them. A re-run "
              f"upgrades each to a real stamp.\n")
    # The BACKING half of certificate integrity, printed beside the two
    # staleness halves for the reason they are printed beside each other: an
    # iteration reads this block and nothing else. The other two ask whether
    # the certificate still describes its own code and its own words; this one
    # asks whether the ground it stands on is still there (94th audit B1).
    _check_unbacked_detector()
    unbacked = unbacked_certificates(ledger)
    if unbacked:
        n_pass = counts[Status.PASS.value]
        print(f"  ? UNBACKED CERTIFICATES — {len(unbacked)} of {n_pass} "
              f"standing PASS row(s) rest on a\n    dependency that is not "
              f"satisfied today. Legal and REPORTING-ONLY: the run happened "
              f"and\n    nothing about it is invalidated — but `run_spec` "
              f"would refuse to re-derive it, so the\n    certificate cannot "
              f"be re-bought until its dependency is. The cost class is the "
              f"bill, and\n    where the chain runs deeper than one hop the "
              f"ROOT is what has to happen first:")
        for sid, deps, cost, roots in unbacked:
            line = f"      {sid:9s} ({cost})  needs {', '.join(deps)}"
            if [r[0] for r in roots] != deps:
                line += ("  ->  root " + ", ".join(
                    f"{r} [{st}] ({c})" for r, st, c in roots))
            print(line)
        print()
    _check_red_delta_detector()
    red = deliberate_red_deltas(ledger)
    if red:
        print("  ! DELIBERATELY-RED GATES — held FAIL by a pending decision. "
              "The number inside\n    still moves; read it as a measurement, "
              "not a token:")
        for sid, metric, cur, prev, prev_at in red:
            if cur is None:
                print(f"      {sid}  {metric} MISSING from the latest row "
                      f"(was {prev} at {prev_at}) — the\n      instrument "
                      f"lost its number; that is a fault, not a quiet day.")
            elif prev is None:
                print(f"      {sid}  {metric} = {cur}  (no prior observation "
                      f"in history)")
            elif cur != prev:
                delta = (f"{cur - prev:+}"
                         if isinstance(cur, (int, float))
                         and isinstance(prev, (int, float)) else f"was {prev}")
                print(f"      {sid}  {metric} = {cur}  (MOVED {delta} since "
                      f"{prev_at}; was {prev}). Say so in\n      your report — "
                      f"a re-run does not remove what moved it.")
            else:
                print(f"      {sid}  {metric} = {cur}  (unchanged since "
                      f"{prev_at})")
        print()
    print_settle_block(ledger)
    breaches = print_ratchet_block(ledger)
    print_steering_block()
    print_fieldwatch_block()
    print_unread_metrics_block()
    print_resolution_block(ledger)
    print("  A capability is claimed ONLY by a PASS here. Nothing else counts.\n")
    # 121st audit FTB 3. `cmd_status` used to `return 0` here unconditionally
    # while the block above printed `!! ABOVE its declared floor`, and every
    # slot summary paired "status rc=0" with that broken ratchet in the same
    # paragraph. Floor state now reaches the exit code; nothing else about
    # this command's output does, and that limit is deliberate — see
    # `ratchet_exit_code`.
    return ratchet_exit_code(**breaches)


def print_resolution_block(ledger: Ledger) -> None:
    """Anchor-decided conjuncts: margin vs the anchor's own seed spread.

    The 109th audit's margin-vs-spread check (`T4.06` certified an arm at 6.9%
    of its anchor's seed noise) got its arithmetic in
    `experiments/resolution.py` and, per the 110th audit's FTB 2, gets its
    reader here — the one block every iteration actually reads. REPORTING-ONLY
    and UNFLOORED: no cutoff, no verdict, nothing reddens; whether a margin
    decides stays with each spec's pre-registration (`SO.10` vacancy
    precedent), and a rule armed on this output is a conjunct owing
    `run blast-radius`. Inventory derivation lives in
    `resolution.anchor_rows`, not here.
    """
    from .resolution import status_lines
    lines = status_lines(ledger.results)
    if lines:
        print("\n".join(lines))
        print()


def print_unread_metrics_block() -> None:
    """Which recorded numbers did the gate never look at? (`D27`, fired
    2026-09-21 by the overseer's 107th audit, default (i).)

    REPORTING-ONLY and UNFLOORED by that default's own text, and it prints its
    own MEASURED FALSE-POSITIVE RATE in the same sentence as its count — 19 of
    20 hand-adjudicated on the day it shipped. An instrument nobody can read
    without also reading how often it is wrong cannot be quoted as if it were
    clean, and at 19/20 this one may not be acted on spec-by-spec at all.

    See `experiments/unread_metrics.py` for the four exclusions, which of them
    is arithmetic rather than taste, and the half of the screen (`bar pairing`)
    that is deliberately NOT built.
    """
    from . import unread_metrics
    unread_metrics._check()
    print(unread_metrics.render())


def cmd_unread(ledger: Ledger) -> int:
    """`run unread` — the same block on its own, with the adjudication draw."""
    from . import unread_metrics
    unread_metrics.measure()
    return 0


def print_steering_block() -> None:
    """Can today's steering-page orders be executed? (95th audit B2.)

    `docs/PROGRESS.md`'s `FOR THE BUILDER` list shipped on 2026-09-13 with two
    dead items — one naming a spec whose measured ceiling forecloses it, one
    (`D1.0`) illegal since 10:05 that morning because `T1.08` went PASS ->
    FAIL beneath it — and three consecutive iterations discovered that by
    hand. It prints HERE, in the command the orientation makes every iteration
    run, because that is where the discovery was being made by hand.

    Reporting-only and unfloored, by the auditor's explicit instruction: an
    order can be legitimately aspirational and a gate here would forbid a
    legal move. See `experiments/steering.py` for what it deliberately does
    NOT attempt (discharge, intent).
    """
    from . import steering
    steering._check()
    print(steering.render())
    # 100th audit B1: the same parse pointed at DECISION deadlines. The
    # register was repaired at ~13:0x on 2026-09-18 and the page published
    # from it at 18:22 still carried the broken date — a written lesson did
    # not survive five hours and one organ boundary, so the check lives here,
    # where a sitting cannot finish without reading it.
    print(steering.render_dates())
    # Builder, 2026-09-23: the same parse pointed at LEDGER METRICS. On
    # 09-22 a stop-rule was armed on both steering pages quoting a reading
    # sixteen days dead (`distractor_abstention` 0.0000; the certificate said
    # 1.0 since 09-06) and ordered the falsified number routed to the owner.
    # Dates had a reader; numbers did not. Same contract: the ledger is the
    # authority, the block is reporting-only, intent stays a human's.
    print(steering.render_metrics())
    # PROGRESS FOR THE BUILDER item 4, 2026-09-21. The 125000-byte ceiling
    # shipped as a SENTENCE in `ladder_prompt.md` — which is the failure this
    # project keeps re-learning, since the page that carries the rule is the
    # page the rule is about and nothing could read it. The outage it guards
    # against cost nineteen slots and twenty-three hours. Reporting-only and
    # unfloored on the Review's explicit instruction: the judgement stays a
    # human's, the visibility is an instrument's.
    steering._check_size()
    print(steering.render_size())


def cmd_steering(ledger: Ledger) -> int:
    """`run steering` — the same block on its own, for a page-edit loop."""
    from . import steering
    steering._check()
    print(steering.render(), end="")
    return 0


def print_fieldwatch_block() -> None:
    """Does every field-watch finding have an owner and a clock? (96th audit
    FTB 1.)

    Week 7's §6 — the Learning-core seat's declared silent-failure guard is
    computed nowhere — sat on `docs/FIELD_WATCH.md` for six hours with no
    queue row and no decision entry while its smaller sibling §6b was
    repaired in twenty-one minutes, because `grep -rn FIELD_WATCH
    --include=*.py` returned zero hits: the page had no reader, so findings
    were consumed in order of mechanisability, not importance. Third
    instance of the class (PROGRESS.md owner-asks, REVIEW_QUEUE.md before
    its reader). Reporting-only and unfloored by the auditor's explicit
    instruction — see `experiments/fieldwatch.py` for what it deliberately
    does NOT attempt (judging content, seeing discharge-in-code).
    """
    from . import fieldwatch
    fieldwatch._check()
    print(fieldwatch.render())


def cmd_fieldwatch(ledger: Ledger) -> int:
    """`run fieldwatch` — the same block on its own."""
    from . import fieldwatch
    fieldwatch._check()
    print(fieldwatch.render(), end="")
    return 0


# Gates deliberately held RED by a pending decision, and the measurement each
# still carries: spec id -> the metric that moves. A RED held by a decision
# stops being read as a number and starts being read as a known token —
# T0.27's `live_violations` moved 1 -> 2 -> 3 over five days while two reports
# called the row "unchanged" and a third called the incident "clear", three
# days before D16 fires on the stale count (62nd audit, FINDING 1). Add an
# entry whenever a gate is parked RED with a moving metric inside; remove it
# when its decision lands.
DELIBERATE_RED_METRICS = {"T0.27": "live_violations"}


def deliberate_red_deltas(ledger: Ledger, watch=None) -> list:
    """(spec_id, metric, current, previous, prev_ran_at) for each watched
    deliberately-RED gate with a ledger entry. `previous` comes from the
    newest history row carrying the metric (None when never observed before);
    `current` is None when the latest row LOST the metric — reported, not
    skipped, because an instrument going quiet is the failure mode this
    reader exists for."""
    watch = DELIBERATE_RED_METRICS if watch is None else watch
    out = []
    for sid, metric in sorted(watch.items()):
        r = ledger.results.get(sid)
        if r is None:
            continue
        cur = (r.metrics or {}).get(metric)
        prev = prev_at = None
        for row in reversed(r.history or []):
            m = row.get("metrics") or {}
            if metric in m:
                prev, prev_at = m[metric], row.get("ran_at", "?")
                break
        out.append((sid, metric, cur, prev, prev_at))
    return out


def _check_red_delta_detector() -> None:
    """Plant a moved metric inside a RED entry and require the reader to
    report the move — plus the two shapes that must stay quiet or degrade
    loudly (no entry at all; entry that lost the metric). The scar: a metric
    inside a permanently-RED gate is unwatched by construction, so the delta
    must come from the reader, not from a reporter remembering to look
    (62nd audit B2)."""
    from types import SimpleNamespace as NS
    planted = NS(results={
        "ZZ.RED": NS(metrics={"violations": 3},
                     history=[{"metrics": {"violations": 1}, "ran_at": "t0"},
                              {"metrics": {"violations": 2}, "ran_at": "t1"}]),
        "ZZ.LOST": NS(metrics={}, history=[{"metrics": {"violations": 2},
                                            "ran_at": "t2"}])})
    got = deliberate_red_deltas(planted, watch={"ZZ.RED": "violations",
                                                "ZZ.LOST": "violations",
                                                "ZZ.ABSENT": "violations"})
    want = [("ZZ.LOST", "violations", None, 2, "t2"),
            ("ZZ.RED", "violations", 3, 2, "t1")]
    if got != want:
        raise RuntimeError(
            f"the red-delta reader returned {got}, expected {want} — "
            "refusing to report a scan it may not have performed")


# Files that are MEASUREMENT MACHINERY: code that decides, records, schedules
# or accounts for a verdict. Deliberately NOT "everything under experiments/" —
# `playground.py`, `w0.py`, `drives.py`, `hearing.py`, `odour.py`, `thermal.py`,
# `render.py` are Jack, his body and his world, and an edit to one of those is
# work ON THE CREATURE even when it costs a certificate. The line is drawn by
# hand, in one place, so a reader can dispute it: everything below sits between
# a run and the ledger row it writes, and none of it is ever a sense, a body
# part or a world rule.
#
# The 92nd audit (2026-09-13, B2) resolved seven of these against the live
# registry: protocol, cpu_budget, coverage, champions, review_queue, gpu, and
# `run.py` (declared by nothing, which is why B1's repair cost no staleness).
# Three more are added here on the same principle and flagged as additions so
# the auditor's 44% figure stays reconstructible: `decisions.py` (the armed-
# default ledger), and the two loop scripts, which schedule runs and are
# declared by `T0.33`.
INSTRUMENT_FILES = (
    "experiments/protocol.py",
    "experiments/run.py",
    "experiments/coverage.py",
    "experiments/champions.py",
    "experiments/review_queue.py",
    "experiments/cpu_budget.py",
    "experiments/gpu.py",
    "experiments/decisions.py",          # added beyond the 92nd audit's map
    "scripts/ladder_loop.sh",            # added beyond the 92nd audit's map
    "scripts/launch_detached.sh",        # added beyond the 92nd audit's map
)


def settle_events(ledger, days: int = 7, now=None, instrument_files=None,
                  deps_of=None) -> dict:
    """Trailing-window ledger settle events, split into three kinds.

    The 92nd audit's B2 (2026-09-13), and the question it exists to make
    answerable: `run status` prints *"107 demonstrated"* and a reader counting
    activity sees twelve PASS events in six days. Every one of those twelve was
    a Tier-0 harness certificate being RE-BOUGHT because an instrument edit
    this project ordered had invalidated it — not one was about Jack. Both
    facts are true, both are already in the record, and nothing prints the
    difference. This does.

    Three kinds, and the split is computed, not judged:

    - ``first``  — the spec's FIRST-EVER verdict. The only kind that can move
      `demonstrated`, and the only kind that is news about the creature.
    - ``rebuy``  — same spec, same resulting status as its previous run. Real
      work and usually owed work (a stamp is a promise about code that moved),
      but it buys back a claim that already stood.
    - ``change`` — the status differs from the previous run: a PASS falling to
      FAIL, a FAIL becoming a PASS, a VOID resolving.

    ``instrument_coupled`` marks a spec that declares a measurement-machinery
    file in `IMPL_DEPS` (`INSTRUMENT_FILES`). That is exactly the population
    whose certificates go stale when this project edits its own tools, so it
    is the mechanism behind a week of re-buys — computable, not a judgement
    call about motive.

    MEASURE AND REPORT. This gates nothing, ratchets nothing and refuses
    nothing, and the audit that ordered it said why: *a cap on re-buys would
    be a cap on honesty*. A re-stamp after an `IMPL_DEPS` edit is the system
    keeping its word. The defect is that the number is invisible, not that it
    is large.

    Two limits, stated rather than discovered later. `Result.history` is
    trimmed to the last 20 rows, so a spec running more than 20 times inside
    the window loses its oldest events here (no spec has come close). And
    AMENDMENTS ARE NOT COUNTED: an amend is not a run, it writes `amended`
    rather than a verdict, and folding it in would let paperwork read as
    measurement — the precise confusion this counter exists to end.
    """
    from datetime import datetime, timedelta
    instrument_files = tuple(INSTRUMENT_FILES if instrument_files is None
                             else instrument_files)
    now = datetime.now() if now is None else now
    cutoff = now - timedelta(days=days)

    def _instrumented(spec_id: str) -> bool:
        if deps_of is not None:
            deps = deps_of(spec_id) or ()
        else:
            from .protocol import impl_deps_of, module_path_for
            path = module_path_for(spec_id)
            if path is None:
                return False
            deps, _problem = impl_deps_of(path)
        return any(str(d).replace("\\", "/") in instrument_files for d in deps)

    events, unparsed = [], 0
    for spec_id, row in sorted(getattr(ledger, "results", {}).items()):
        seq = [{"ran_at": r.get("ran_at"),
                "status": getattr(r.get("status"), "value", r.get("status"))}
               for r in (getattr(row, "history", None) or [])]
        seq.append({"ran_at": getattr(row, "ran_at", None),
                    "status": getattr(getattr(row, "status", None), "value",
                                      getattr(row, "status", None))})
        coupled = None
        for i, ev in enumerate(seq):
            if not ev["ran_at"]:
                continue
            try:
                when = datetime.fromisoformat(str(ev["ran_at"]))
            except ValueError:
                unparsed += 1
                continue
            if when.tzinfo is not None:
                when = when.astimezone().replace(tzinfo=None)
            if when < cutoff or when > now:
                continue
            kind = ("first" if i == 0
                    else "rebuy" if ev["status"] == seq[i - 1]["status"]
                    else "change")
            if coupled is None:
                coupled = _instrumented(spec_id)
            events.append({"spec_id": spec_id, "ran_at": ev["ran_at"],
                           "status": ev["status"], "kind": kind,
                           "instrument_coupled": coupled})
    events.sort(key=lambda e: (e["ran_at"], e["spec_id"]))
    kinds = {k: [e for e in events if e["kind"] == k]
             for k in ("first", "rebuy", "change")}
    passes = [e for e in events if e["status"] == Status.PASS.value]
    return {"days": days, "events": events, "kinds": kinds,
            "passes": passes,
            "instrument_coupled": [e for e in events if e["instrument_coupled"]],
            "pass_instrument_coupled": [e for e in passes
                                        if e["instrument_coupled"]],
            "unparsed_ran_at": unparsed}


def _check_settle_event_reader() -> None:
    """Known-positive plant for `settle_events` — one row per kind, plus the
    two ways the split can silently collapse.

    The scar this copies: `_check_stale_detector` refuses to report a clean
    scan it may not have performed, and the 09-13 DIRTY lesson adds that a
    bucket's plant does not cover the sub-states inside it. Here the whole
    value of the counter IS the sub-states — a reader who is told `12 PASS`
    and shown no split learns nothing new — so every kind gets a planted row
    that must come back labelled, including the two directions that would
    flatter the record: a re-buy read as a first-ever verdict (activity read
    as news), and an out-of-window event counted (an old week's work read as
    this one's).
    """
    from datetime import datetime, timedelta
    from types import SimpleNamespace as NS
    now = datetime(2026, 1, 20, 12, 0, 0)

    def at(d):
        return (now - timedelta(days=d)).isoformat()

    planted = NS(results={
        # first-ever verdict, in window, instrument-coupled
        "ZZ.FIRST": NS(ran_at=at(1), status="PASS", history=[]),
        # same status as the previous run: a re-buy, not news
        "ZZ.REBUY": NS(ran_at=at(2), status="PASS",
                       history=[{"ran_at": at(30), "status": "PASS"}]),
        # status moved
        "ZZ.CHANGE": NS(ran_at=at(3), status="FAIL",
                        history=[{"ran_at": at(31), "status": "PASS"}]),
        # every event older than the window: must contribute nothing
        "ZZ.OLD": NS(ran_at=at(40), status="PASS",
                     history=[{"ran_at": at(90), "status": "FAIL"}]),
        # two runs inside the window: the first is a change, the second a re-buy
        "ZZ.TWICE": NS(ran_at=at(1), status="VOID",
                       history=[{"ran_at": at(60), "status": "PASS"},
                                {"ran_at": at(4), "status": "VOID"}]),
    })
    got = settle_events(planted, days=7, now=now,
                        deps_of=lambda sid: (["experiments/protocol.py"]
                                             if sid == "ZZ.FIRST" else
                                             ["playground.py"]))
    want = [("ZZ.TWICE", "change"), ("ZZ.CHANGE", "change"),
            ("ZZ.REBUY", "rebuy"), ("ZZ.FIRST", "first"),
            ("ZZ.TWICE", "rebuy")]
    saw = [(e["spec_id"], e["kind"]) for e in got["events"]]
    coupled = [e["spec_id"] for e in got["instrument_coupled"]]
    if saw != want or coupled != ["ZZ.FIRST"] or len(got["passes"]) != 2:
        raise RuntimeError(
            f"the settle-event reader returned {saw} / coupled={coupled} / "
            f"{len(got['passes'])} PASS, expected {want} / ['ZZ.FIRST'] / 2 — "
            "refusing to report a split it may not have computed")


def print_settle_block(ledger, days: int = 7) -> None:
    """The B2 counter as `run status` prints it. Reporting only."""
    _check_settle_event_reader()
    s = settle_events(ledger, days=days)
    n = len(s["events"])
    k = s["kinds"]
    print(f"  SETTLE EVENTS — last {days} days: {n} run(s) recorded = "
          f"{len(k['first'])} first-ever verdict, {len(k['rebuy'])} re-buy "
          f"(same\n    status as the run before it), {len(k['change'])} status "
          f"change. Amendments are not runs and\n    are not counted. "
          f"Measure-and-report: nothing here gates, ratchets or refuses.")
    if n:
        ic, pc = s["instrument_coupled"], s["pass_instrument_coupled"]
        print(f"      instrument-coupled  {len(ic)} of {n} "
              f"({100.0 * len(ic) / n:.0f}%) — the spec declares a measurement"
              f"-machinery\n      file in IMPL_DEPS, so this project's own tool "
              f"edits are what staled its certificate.")
        if s["passes"]:
            print(f"      PASS events         {len(s['passes'])}, of which "
                  f"{len(pc)} instrument-coupled and "
                  f"{len([e for e in s['passes'] if e['kind'] == 'first'])} "
                  f"first-ever.")
        # The two kinds that can move what this ladder claims are listed in
        # full; re-buys are ROLLED UP, never dropped, and the roll-up says so
        # (no silent caps — a truncation that reads as coverage is the defect
        # this counter exists to end).
        for e in k["first"] + k["change"]:
            tag = "FIRST-EVER" if e["kind"] == "first" else "CHANGED"
            mark = "  [instrument]" if e["instrument_coupled"] else ""
            print(f"      {e['ran_at'][:16]}  {e['spec_id']:<8} "
                  f"{e['status']:<7} {tag}{mark}")
        if k["rebuy"]:
            tally = {}
            for e in k["rebuy"]:
                tally[e["spec_id"]] = tally.get(e["spec_id"], 0) + 1
            top = sorted(tally.items(), key=lambda x: (-x[1], x[0]))
            shown = ", ".join(f"{sid}x{c}" if c > 1 else sid
                              for sid, c in top[:12])
            more = f", +{len(top) - 12} more spec(s)" if len(top) > 12 else ""
            print(f"      {len(k['rebuy'])} re-buy(s) across {len(tally)} "
                  f"spec(s), rolled up rather than listed:\n"
                  f"        {shown}{more}")
        print("    A re-buy is honest, usually owed work — a stamp is a promise "
              "about code that moved.\n    The point is that a reader counting "
              "PASS events cannot otherwise see how many of them\n    were "
              "about Jack. Only `first` and `change` can move what this ladder "
              "claims.")
    if s["unparsed_ran_at"]:
        print(f"    ! {s['unparsed_ran_at']} row(s) carry an unparseable "
              f"`ran_at` and are excluded — reported, not swallowed.")
    print()


def gpu_attribution(lines) -> tuple:
    """(attributed, named) out of `gpu_submissions.jsonl` records (83rd audit,
    `85d435b`, B1).

    `attributed`: job id -> {"spec", "ran_at"} from `phase: "attribution"`
    lines only — the backfill lane for charged jobs whose ledger rows predate
    the `gpu_job_id` field. A line reaches its job ids through its own
    `job_id` field (a synthesised attempt has no result line to join through)
    or through every result line sharing its `attempt_id`.

    `named`: job id -> True for every charged-joinable record that names ANY
    spec string at all — attribution lines plus the `spec` field the runner
    writes on attempt/result lines, probe/pilot labels included. This is the
    domain of "attributable to no spec by any record this project keeps",
    which is deliberately wider than `attributed`: a probe label is not a
    ledger row, but the money has a name."""
    att_to_jobs = {}
    for l in lines:
        if l.get("job_id") and l.get("attempt_id"):
            att_to_jobs.setdefault(l["attempt_id"], set()).add(l["job_id"])
    attributed, named = {}, {}
    for l in lines:
        spec = (l.get("spec") or "").strip()
        if not spec:
            continue
        jobs = set()
        if l.get("job_id"):
            jobs.add(l["job_id"])
        jobs |= att_to_jobs.get(l.get("attempt_id"), set())
        for j in jobs:
            named[j] = True
            if l.get("phase") == "attribution":
                attributed[j] = {"spec": spec, "ran_at": l.get("ran_at")}
    return attributed, named


def gpu_probe_hours(lines, charged: dict) -> dict:
    """phase label -> {"hours", "jobs"} for charged jobs whose submission
    records carry a non-empty `spec_phase` (99th audit B5).

    A probe buys no ledger row by definition, so `gpu_hours_verdictless` —
    which joins through `ledger.results` — reads probe spend as zero waste:
    2.11 colab hours that retrieved nothing were invisible on the page that
    exists to show hours against verdicts. This reads the OTHER receipt the
    dispatcher already writes. Job ids join through the record's own
    `job_id` or through result lines sharing its `attempt_id`, the same two
    paths `gpu_attribution` walks. REPORTING-ONLY AND UNFLOORED, per D27's
    reasoning: probe spend is legitimate and gating it would punish the
    honest thing."""
    att_to_jobs = {}
    for l in lines:
        if l.get("job_id") and l.get("attempt_id"):
            att_to_jobs.setdefault(l["attempt_id"], set()).add(l["job_id"])
    seen = {}   # job id -> phase (first non-empty label wins)
    for l in lines:
        phase = (l.get("spec_phase") or "").strip()
        if not phase:
            continue
        jobs = set()
        if l.get("job_id"):
            jobs.add(l["job_id"])
        jobs |= att_to_jobs.get(l.get("attempt_id"), set())
        for j in jobs:
            if j in charged:
                seen.setdefault(j, phase)
    out = {}
    for j, phase in seen.items():
        b = out.setdefault(phase, {"hours": 0.0, "jobs": 0})
        b["hours"] = round(b["hours"] + charged[j], 4)
        b["jobs"] += 1
    return out


def gpu_hours_verdictless(results, charged: dict, attributed=None) -> dict:
    """Compute bought against verdicts returned (82nd audit `2b3e8a6` B2;
    attribution path 83rd audit `85d435b` B1).

    For every spec whose MOST RECENT outcome is VOID or FAIL, join each of its
    ledger rows — the current one AND every `history` row — against
    `charged` (job id -> hours from `gpu_budget.json`'s `charged_jobs`).
    `gpu_job_id` is comma-joined for multi-kernel dispatches (`D1.0` carries
    four ids in one field), so split before looking up. A row counts as an
    ATTEMPT only if it names at least one charged job (this reading is about
    charged compute, not about runs in general); it counts as a VERDICT if
    that row's status is PASS or FAIL — a FAIL is an honest measurement, a
    VOID is not, and the verdict count is what separates money that bought an
    answer from money that bought none.

    The ledger field joins FIRST; `attributed` (job id -> {"spec", "ran_at"}
    from `gpu_attribution`) may only ADD charged jobs the ledger does not
    already name — 5 of 21 remote rows predate the field, so the two most
    expensive non-PASS GPU rows in the ladder (T2.01's pair of 5.58 h
    kernels) read as zero here until the backfill. An attributed job whose
    `ran_at` matches one of the spec's rows counts against THAT row, so its
    PASS/FAIL is honestly a verdict; one matching no row counts as an attempt
    that bought none. Returns {"TOTAL": "H h",
    spec_id: "H h / N attempt(s) / V verdict(s)", ...}; specs with no charged
    GPU rows are absent. MEASURE AND REPORT, GATE NOTHING — monotone by
    construction (charged hours are never un-charged), no dispatch refused,
    no threshold moved."""
    def fields(row):
        if isinstance(row, dict):
            s, jid, ra = (row.get("status"), row.get("gpu_job_id"),
                          row.get("ran_at"))
        else:
            s, jid, ra = (getattr(row, "status", None),
                          getattr(row, "gpu_job_id", None),
                          getattr(row, "ran_at", None))
        return getattr(s, "value", s), jid, ra

    att_by_spec = {}
    for jid, a in (attributed or {}).items():
        if jid in charged:
            att_by_spec.setdefault(a["spec"], []).append((jid, a.get("ran_at")))

    out, total = {}, 0.0
    for sid in sorted(set(results) | set(att_by_spec)):
        r = results.get(sid)
        if r is None:
            continue   # attributed to a spec with no ledger row at all
        latest, _, _ = fields(r)
        if latest not in ("VOID", "FAIL"):
            continue
        rows = [r] + list(getattr(r, "history", None) or [])
        hours, attempts, verdicts, used = 0.0, 0, 0, set()
        for row in rows:
            status, jid, _ = fields(row)
            ids = [j.strip() for j in str(jid or "").split(",") if j.strip()]
            if not any(j in charged for j in ids):
                continue
            attempts += 1
            used.update(j for j in ids if j in charged)
            hours += sum(charged[j] for j in ids if j in charged)
            if status in ("PASS", "FAIL"):
                verdicts += 1
        for jid, ran_at in sorted(att_by_spec.get(sid, [])):
            if jid in used:
                continue   # the ledger already names it; attribution adds only
            attempts += 1
            hours += charged[jid]
            row_status = next((fields(row)[0] for row in rows
                               if ran_at and fields(row)[2] == ran_at), None)
            if row_status in ("PASS", "FAIL"):
                verdicts += 1
        if attempts:
            total += hours
            out[sid] = (f"{hours:.2f} h / {attempts} attempt(s) / "
                        f"{verdicts} verdict(s)")
    return {"TOTAL": f"{total:.2f} h", **out}


# Shrink-only floor on charged jobs that join to NO spec by either path —
# ledger `gpu_job_id` or a `gpu_submissions.jsonl` record (83rd audit
# `85d435b` B1, same idiom as coverage's FAIL_UNOWNED_BASELINE: the constant
# moves only in the commit that moves the number, with the reason here).
# Growth log:
#   2026-09-07  declared at 21 jobs / 6.32 h, immediately after the T2.01
#               backfill (was 23 / 17.48 before it). The 21 are probe/error
#               kernels and pre-spec-field dispatches nobody has evidence to
#               name; shrinking is welcome, growing means a dispatch lane
#               stopped writing its receipt.
GPU_UNATTRIBUTED_FLOOR = 21


def gpu_unattributed(charged: dict, ledger_jobs, named) -> dict:
    """Charged jobs joining to no spec by EITHER path. Returns
    {job id: hours}; the counter reports len() and the reading prints the
    hour sum beside it. Pure, so the fixture can pin both known-positives."""
    return {j: h for j, h in charged.items()
            if j not in ledger_jobs and j not in named}


def _ledger_job_ids(results) -> set:
    """Every job id any ledger row (current or history) names, all statuses —
    a PASS's money bought an answer, so its jobs are attributed."""
    ids = set()
    for r in results.values():
        for row in [r] + list(getattr(r, "history", None) or []):
            jid = (row.get("gpu_job_id") if isinstance(row, dict)
                   else getattr(row, "gpu_job_id", None))
            ids.update(j.strip() for j in str(jid or "").split(",")
                       if j.strip())
    return ids


def _load_gpu_attribution() -> tuple:
    import json as _json
    path = _REPO / "experiments" / "gpu_submissions.jsonl"
    if not path.exists():
        return {}, {}
    lines = [_json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return gpu_attribution(lines)


def _check_gpu_hours_reader() -> None:
    """Plant one spec per shape and require the join to read each: a VOID
    spec with a comma-joined field AND a charged history row AND an
    uncharged-only row (split, summed, attempt-counted, non-attempt); a FAIL
    spec (in scope, its row IS a verdict); a PASS spec (excluded — its money
    bought an answer); a CPU-only FAIL (absent, no charged row). The scar:
    D1.0's 33.78 h / 2 attempts / 0 verdicts existed only as a hand join in
    the 82nd audit — a quantity nobody prints is a quantity nobody defends.

    Attribution shapes (83rd audit `85d435b` B1), both known-positives
    ordered by the audit: a charged job reachable ONLY through an
    attribution line MUST be counted (ZZ.ATT — the T2.01 shape: FAIL row,
    `gpu_job_id` None, `ran_at`-matched so its verdict counts; k/j7 on the
    same spec matches no row, an attempt that bought none); one reachable
    through NEITHER path must appear in the unattributed figure and not
    silently vanish. Precedence pinned too: an attribution naming a job the
    ledger already names adds nothing (k/j1)."""
    from types import SimpleNamespace as NS
    charged = {"k/j1": 1.5, "k/j2": 2.25, "k/j3": 4.0, "k/j5": 0.5,
               "k/j6": 3.0, "k/j7": 0.25, "k/orphan": 0.75}
    planted = {
        "ZZ.VOID": NS(status="VOID", gpu_job_id="k/j1,k/j2",
                      history=[{"status": "VOID", "gpu_job_id": "k/j3"},
                               {"status": "ERROR", "gpu_job_id": "k/unc"}]),
        "ZZ.FAIL": NS(status="FAIL", gpu_job_id="k/j5", history=[]),
        "ZZ.PASS": NS(status="PASS", gpu_job_id="k/j5", history=[]),
        "ZZ.CPU": NS(status="FAIL", gpu_job_id=None, history=[]),
        "ZZ.ATT": NS(status="FAIL", gpu_job_id=None, ran_at="t-head",
                     history=[])}
    attributed = {"k/j6": {"spec": "ZZ.ATT", "ran_at": "t-head"},
                  "k/j7": {"spec": "ZZ.ATT", "ran_at": "t-nowhere"},
                  "k/j1": {"spec": "ZZ.VOID", "ran_at": None}}
    got = gpu_hours_verdictless(planted, charged, attributed)
    want = {"TOTAL": "11.50 h",
            "ZZ.ATT": "3.25 h / 2 attempt(s) / 1 verdict(s)",
            "ZZ.VOID": "7.75 h / 2 attempt(s) / 0 verdict(s)",
            "ZZ.FAIL": "0.50 h / 1 attempt(s) / 1 verdict(s)"}
    if got != want:
        raise RuntimeError(
            f"the gpu-hours join returned {got}, expected {want} — "
            "refusing to report a join it may not have performed")
    orphan = gpu_unattributed(
        charged, ledger_jobs={"k/j1", "k/j2", "k/j3", "k/j5"},
        named={"k/j6": True, "k/j7": True})
    if orphan != {"k/orphan": 0.75}:
        raise RuntimeError(
            f"the unattributed reader returned {orphan}, expected the orphan "
            "job alone — refusing to report a residue it may not have read")
    # Probe-bucket shapes (99th audit B5): a result line naming its phase and
    # job_id (counted); an attempt line whose phase reaches the job only
    # through a shared attempt_id (counted); an uncharged probe job (absent);
    # a phase-less registered run (absent).
    probe_lines = [
        {"phase": "result", "attempt_id": "a1", "job_id": "k/j2",
         "spec_phase": "probe"},
        {"phase": "attempt", "attempt_id": "a2", "spec_phase": "pilot"},
        {"phase": "result", "attempt_id": "a2", "job_id": "k/j3"},
        {"phase": "result", "attempt_id": "a3", "job_id": "k/uncharged",
         "spec_phase": "probe"},
        {"phase": "result", "attempt_id": "a4", "job_id": "k/j1",
         "spec_phase": ""}]
    got_probe = gpu_probe_hours(probe_lines, charged)
    want_probe = {"probe": {"hours": 2.25, "jobs": 1},
                  "pilot": {"hours": 4.0, "jobs": 1}}
    if got_probe != want_probe:
        raise RuntimeError(
            f"the probe-hours reader returned {got_probe}, expected "
            f"{want_probe} — refusing to report a bucket it may not have read")


# ── Ratchet counters, read independently of every verdict (64th audit B2) ──
#
# The scar: on 2026-09-02 the unreachable ratchet printed `GREW: 89 vs 85`
# inside `coverage`, whose exit code had been red since 09-01 for a blessed,
# Review-owned reason — and five consecutive iterations read rc=2, recalled
# the blessed red, and skipped the body. The number that moved was read by
# nobody for 5½ hours. A shared exit code is a disjunction: the moment one
# clause is blessed and standing, every other clause it ORs over goes silent.
# So every ratchet counter a standing-red tool carries is ALSO printed here,
# in `run status`, with its delta since the last COMMITTED reading — a
# channel no tool's verdict can silence. Same idiom as
# DELIBERATE_RED_METRICS one block up, generalised from one metric in one
# red gate to the class.
#
# `experiments/ratchet_readings.json` holds the committed readings. The delta
# is computed against the HEAD revision of that file, so an uncommitted
# rewrite cannot quiet the block. `run ratchets record` refreshes it — run it
# in the SAME commit as the change that legitimately moved a counter.

RATCHET_READINGS = _REPO / "experiments" / "ratchet_readings.json"


def ratchet_live(ledger: Ledger) -> dict:
    """name -> (value | None, note). A counter whose computation raises
    reports (None, why) rather than vanishing — LOST is a finding, not a
    skip, because an instrument going quiet is the failure mode this reader
    exists for. Every import is local so `status` pays only when printing."""
    out = {}

    def take(name, fn):
        try:
            out[name] = (fn(), "")
        except Exception as exc:
            out[name] = (None, f"{type(exc).__name__}: {exc}")

    def _unreachable():
        from .coverage import unreachable_ratchet
        u = unreachable_ratchet(ledger=ledger)
        if u["count"] is None:
            raise RuntimeError("; ".join(u["refused"]))
        return u["count"]

    def _claim_dead_count():
        from .coverage import _claim_dead, report
        return sum(1 for r in report() if _claim_dead(r))

    def _commitments_uncovered():
        # 101st audit RANK 2: this class went 0 -> 4 on 2026-09-18 and no
        # machine-readable signal could say so — coverage's exit code was
        # already red on claim_dead, and the number was recorded only as an
        # unread metric on T0.21's ledger row. Same predicate as `check()`'s
        # printer, factored so the two readers cannot drift.
        from .coverage import uncovered_commitments
        return len(uncovered_commitments())

    def _park_release_pairs():
        from .coverage import park_release
        return len(park_release()["violations"])

    def _champions_trigger_debt():
        from . import champions
        _v, seats = champions.audit(champions.DOC.read_text(), BY_ID,
                                    lambda sid: ledger.status(sid).value)
        return len(champions.unreachable_triggers(seats))

    def _champions_unwinnable():
        # 91st audit RANK 2: this class grew from 3 to 4 on 09-12 when LG.03
        # was foreclosed and NOT ONE COUNTER MOVED, in either tool. It now has
        # a baseline in `champions.py`; surfacing it HERE is the other half —
        # the builder reads `run status` every slot and does not run
        # `champions --check` every slot.
        from . import champions
        _v, seats = champions.audit(champions.DOC.read_text(), BY_ID,
                                    lambda sid: ledger.status(sid).value)
        return len(champions.unwinnable_seats(seats))

    def _review_queue_total():
        from . import review_queue as rq
        return rq.live_audit()["total"]

    def _review_queue_net_arrivals():
        # The 69th audit's finding, as an integer that cannot drift quietly:
        # rows ARRIVED minus rows DISPOSED over the trailing window. Positive
        # means the desk fell that far behind. It is deliberately joined here
        # rather than gated anywhere: a slow week is legal (the metric
        # discipline `piled_on` was given), but a queue tripling while every
        # ratchet sat at its floor is what happened on 2026-09-04.
        from . import review_queue as rq
        t = rq.live_audit()["throughput"]
        if t is None:
            raise RuntimeError(
                "no git baseline for the trailing "
                f"{rq.THROUGHPUT_WINDOW_DAYS} days — the disposal rate is "
                "UNMEASURED, which is a fault and not a clean week")
        return t["net_arrivals"]

    def _cpu_foreclosed_now():
        # 69th audit B4, second half. A CPU refusal returns UNRECORDED by
        # design (tenant protection is not a measurement of the spec), so
        # until now a day that closed 35% of the CPU lane existed only as a
        # line in a transient print — B3 of the 68th audit made it visible;
        # this makes it REMEMBERED. A METRIC with no floor: a legitimately
        # spent day SHOULD refuse things, and gating it at zero would forbid
        # the protection working.
        from .cpu_budget import foreclosed_now
        return len(foreclosed_now())

    def _fail_unowned():
        # 72nd audit B1: a settled FAIL with no repaired_by, no REVIEW_QUEUE
        # mention and no FAIL-DISPOSED disposition — the state every
        # dispatch-keyed reader treats as out of scope (XL.01, 17 silent days).
        from .coverage import fail_unowned_ratchet
        f = fail_unowned_ratchet()
        if f["count"] is None:
            raise RuntimeError("; ".join(f["refused"]))
        return f["count"]

    def _pass_on_dead_dependency():
        # Review FULL 2026-09-13, built 2026-09-23 by the Review executing its
        # own overdue row. T2.10 fell to FAIL on 08-31; T6.03 declares it in
        # depends_on and went on rendering [PASS] here for THIRTEEN DAYS,
        # because the board reports a STORED status and nothing re-evaluates a
        # standing PASS when a spec beneath it dies. Counted in PAIRS so one
        # certificate on two dead feet is two repairs, not one. Floored
        # shrink-only at its measured value: growth means another certificate
        # outlived its foundation, which is the entire event.
        from .coverage import pass_on_dead_dependency_ratchet
        f = pass_on_dead_dependency_ratchet()
        if f["count"] is None:
            raise RuntimeError("; ".join(f["refused"]))
        return f["count"]

    def _review_queue_piled_on():
        # 73rd audit B1: the batch blindness. `piled_on` ordered rows by a
        # day-granularity `routed` date, so N rows routed in one commit onto
        # one date all read `prior = 0` — reported 17 where the truth was 22,
        # and six live due-dates were fed by same-day batches the number
        # could not see. Counted here so the corrected number is COMMITTED:
        # a mover must `run ratchets record` in the commit that moves it.
        # 22 is the honest baseline; 17 was never a real reading. A METRIC
        # with no floor — piling can be legal — same discipline as
        # cpu_foreclosed_now.
        from . import review_queue as rq
        return len(rq.live_audit()["piled_on"])

    def _review_queue_violation_forms():
        # 103rd audit item 3: `review_queue_violations` stored only its TOTAL,
        # so the 09-19 excursion 12 -> 13 -> 12 (a MALFORMED row written and
        # repaired by the builder's own commits, three hours apart) left no
        # trace in `ratchet_readings.json`, and a composition change that
        # cancels in the total reads UNCHANGED. The components are the store:
        # counts by violation class, nonzero only (`fail_unowned_owned_forms`'
        # idiom — the count ratchet above is the number, this is the map).
        # Each class carries a declared cause bucket in
        # VIOLATION_CAUSE_BUCKETS; the printer attributes every component
        # delta there. Reporting-only: no floor, gates nothing.
        from . import review_queue as rq
        counts = rq.live_audit()["counts"]
        return {k: v for k, v in sorted(counts.items()) if v}

    def _goal_unrunnable():
        # 73rd audit B3: `new_unrunnable_citation` is RED in coverage's exit
        # code, but the class grew 3 -> 7 under an exit code ALREADY held red
        # by claim_dead — a blessed red silenced the number. This block was
        # built so that cannot happen; the class had no line in it.
        from .coverage import goal_citations
        return len(goal_citations()["unrunnable"])

    def _fail_unowned_owned_forms():
        # 74th audit B2: the ownership map misread 3 of its 5 mention-only
        # rows (a flush-left evidence body ended the block before the id was
        # seen; a BLOCKED-BY: clock — legal payment — was rejected). The
        # corrected breakdown is committed here so a form drifting is a
        # MOVED line, not an inference across two tools. Counts by form,
        # sorted; the count ratchet above is the floor, this is the map.
        from .coverage import fail_unowned
        owned = fail_unowned()["owned"]
        forms = {}
        for v in owned.values():
            forms[v] = forms.get(v, 0) + 1
        return dict(sorted(forms.items()))

    def _gpu_hours_no_verdict():
        # 82nd audit (2b3e8a6) B2: D1.0 bought 33.78 GPU-hours across two
        # attempts for zero verdicts — 113% of a weekly free allocation on
        # one spec — and no instrument could print that sentence; the join of
        # gpu_budget.json against ledger gpu_job_id fields existed only as a
        # hand computation inside the audit. Printed here so the third
        # attempt is authorised by somebody who can see the running total on
        # the same page as the verdict, instead of by a hand-written
        # prohibition in a priority block. A metric with no floor: honest
        # VOIDs and FAILs legitimately cost hours (the gate firing IS the
        # gate working); gating this would punish honesty.
        # 83rd audit (85d435b) B1: the attribution path joins second, and
        # the residue neither path reaches is printed beside the total.
        import json as _json
        _check_gpu_hours_reader()
        from .gpu import BUDGET_FILE
        charged = {jid: rec.get("hours", 0.0)
                   for jid, rec in _json.loads(BUDGET_FILE.read_text())
                   .get("charged_jobs", {}).items()}
        attributed, named = _load_gpu_attribution()
        out = gpu_hours_verdictless(ledger.results, charged, attributed)
        orphans = gpu_unattributed(charged, _ledger_job_ids(ledger.results),
                                   named)
        out["UNATTRIBUTED"] = (f"{sum(orphans.values()):.2f} h / "
                               f"{len(orphans)} job(s)")
        # 99th audit B5: probe/pilot spend buys no ledger row by definition,
        # so it must be named here or it reads as zero waste forever.
        # Reporting-only and unfloored (D27's reasoning).
        path = _REPO / "experiments" / "gpu_submissions.jsonl"
        if path.exists():
            lines = [_json.loads(l) for l in path.read_text().splitlines()
                     if l.strip()]
            for phase, b in sorted(gpu_probe_hours(lines, charged).items()):
                out[phase.upper()] = f"{b['hours']:.2f} h / {b['jobs']} job(s)"
        return out

    def _gpu_unattributed_jobs():
        # 83rd audit (85d435b) B1: 23 charged jobs — 17.48 h, 27.7% of every
        # per-job record — joined to no spec by any record this project
        # keeps. Counted as its own ratchet with a declared floor
        # (GPU_UNATTRIBUTED_FLOOR) so the residue is shrink-only: growth
        # means a dispatch lane stopped writing its receipt.
        import json as _json
        from .gpu import BUDGET_FILE
        charged = {jid: rec.get("hours", 0.0)
                   for jid, rec in _json.loads(BUDGET_FILE.read_text())
                   .get("charged_jobs", {}).items()}
        _, named = _load_gpu_attribution()
        return len(gpu_unattributed(charged,
                                    _ledger_job_ids(ledger.results), named))

    _decisions_debt = {}

    def _decisions_class(kind):
        # 120th audit FINDING 2: `decisions.py` ratchets five classes
        # shrink-only (`RATCHETED`) and none of them was joined here —
        # DEFAULT-ACTION-EXPIRED went 0 -> 1 on 2026-09-23 (D33) and sat red
        # for three days with no committed reading, no `at` date and no
        # `!! MOVED` banner, because `ratchet_readings.json` had no key to
        # hold one. Same stage `champions.py` was at before the 91st audit's
        # repair, one comment up. Counted from `decisions.check_violations()`
        # — the SAME population `--check` exits on, factored there so the two
        # readers cannot drift. Computed once per scan (the firing audit
        # costs ~3 s of `git show`); a failure is cached and re-raised so
        # all five report LOST rather than one line paying five retries.
        if not _decisions_debt:
            from . import decisions as dec
            try:
                _decisions_debt["debt"] = dec.ratchet_debt(
                    dec.check_violations()[0])
            except Exception as exc:
                _decisions_debt["err"] = exc
        if "err" in _decisions_debt:
            raise _decisions_debt["err"]
        n, _base = _decisions_debt["debt"][kind]
        return n

    take("unreachable", _unreachable)
    take("fail_unowned", _fail_unowned)
    take("fail_unowned_owned_forms", _fail_unowned_owned_forms)
    take("pass_on_dead_dependency", _pass_on_dead_dependency)
    take("goal_unrunnable", _goal_unrunnable)
    take("cpu_foreclosed_now", _cpu_foreclosed_now)
    take("gpu_hours_no_verdict", _gpu_hours_no_verdict)
    take("gpu_unattributed_jobs", _gpu_unattributed_jobs)
    take("claim_dead", _claim_dead_count)
    take("commitments_uncovered", _commitments_uncovered)
    take("park_release_pairs", _park_release_pairs)
    take("champions_trigger_debt", _champions_trigger_debt)
    take("champions_unwinnable", _champions_unwinnable)
    take("review_queue_violations", _review_queue_total)
    take("review_queue_violation_forms", _review_queue_violation_forms)
    take("review_queue_net_arrivals", _review_queue_net_arrivals)
    take("review_queue_piled_on", _review_queue_piled_on)
    take("decisions_undeclared",
         lambda: _decisions_class("UNDECLARED"))
    take("decisions_unrouted_owner_ask",
         lambda: _decisions_class("UNROUTED-OWNER-ASK"))
    take("decisions_vanished_owner_ask",
         lambda: _decisions_class("VANISHED-OWNER-ASK"))
    take("decisions_default_action_expired",
         lambda: _decisions_class("DEFAULT-ACTION-EXPIRED"))
    take("decisions_firing_diff",
         lambda: _decisions_class("FIRING-DIFF"))
    return out


def ratchet_floors() -> dict:
    """name -> the declared shrink-only floor, for counters that carry one.

    A floor lives as a CONSTANT in the tool's own module and moves only in a
    commit that edits it against its growth log. That is a different channel
    from `ratchet_readings.json`, which `run ratchets record` refreshes: on
    2026-09-03 (65th audit B3) `record` + commit turned the unreachable
    counter's MOVED line into UNCHANGED, so a reader that compares only
    against the recording can be made quiet by writing a file. The floor
    cannot — so it is compared here too, in the channel no verdict silences.
    """
    from .champions import BASELINE_UNWINNABLE
    from .coverage import (COMMITMENTS_UNCOVERED_BASELINE,
                           FAIL_UNOWNED_BASELINE,
                           PASS_ON_DEAD_DEPENDENCY_BASELINE,
                           UNREACHABLE_BASELINE)
    from .decisions import (BASELINE_ACTION_EXPIRED, BASELINE_FIRING_HAZARDS,
                            BASELINE_UNDECLARED, BASELINE_UNROUTED_ASKS,
                            BASELINE_VANISHED_ASKS)
    return {"unreachable": UNREACHABLE_BASELINE,
            "fail_unowned": FAIL_UNOWNED_BASELINE,
            # Added 2026-09-23 (Review DAILY, executing `pass-certificates-
            # are-not-re-evaluated-when-a-dependency-falls`, routed 09-13).
            # The floor is the channel a `ratchets record` cannot quiet, which
            # is the whole point for a class whose failure mode is a number
            # that looks unchanged because nobody computes it.
            "pass_on_dead_dependency": PASS_ON_DEAD_DEPENDENCY_BASELINE,
            "gpu_unattributed_jobs": GPU_UNATTRIBUTED_FLOOR,
            # Added 2026-09-13 (91st audit B2). The floor is the channel a
            # `ratchets record` cannot quiet, which matters most for a class
            # that spent eleven days with no floor at all.
            "champions_unwinnable": BASELINE_UNWINNABLE,
            # Added 2026-09-19 (101st audit RANK 2 / FTB 1) — the class the
            # coverage charter ranks above every other finding grew 0 -> 4
            # with no counter anywhere to move.
            "commitments_uncovered": COMMITMENTS_UNCOVERED_BASELINE,
            # Added 2026-09-26 (120th audit FINDING 2): decisions.py's five
            # shrink-only classes, joined the day DEFAULT-ACTION-EXPIRED had
            # sat red for three days with no committed reading to date it.
            # The baselines live in decisions.py beside their growth logs
            # (`RATCHETED` is the same five constants keyed by class name).
            "decisions_undeclared": BASELINE_UNDECLARED,
            "decisions_unrouted_owner_ask": BASELINE_UNROUTED_ASKS,
            "decisions_vanished_owner_ask": BASELINE_VANISHED_ASKS,
            "decisions_default_action_expired": BASELINE_ACTION_EXPIRED,
            "decisions_firing_diff": BASELINE_FIRING_HAZARDS}


# 120th audit item 3 — the DURABLE half of FINDING 2, whose instance was the
# five `decisions.py` joins one commit earlier (6a15ad4). The finding was not
# "five take() lines were missing"; it was that NOTHING asserts `ratchet_live`
# covers the tools' own floored-class declarations, so any instrument can ship
# a shrink-only class that no committed reading ever dates — exactly how
# DEFAULT-ACTION-EXPIRED sat red for three days with no `at` and no `!! MOVED`,
# and exactly the stage `champions.py` was at before the 91st audit's repair.
#
# The population is enumerated from the tools' OWN source by the idiom all
# four declare in prose ("Ratchets ... in the BASELINE_UNDECLARED idiom"):
# module-level `BASELINE_*` / `*_BASELINE` constants. A hand-kept list here
# would itself be the drift channel this exists to close. RESIDUAL, named
# rather than called closed (the SYSTEM.md discipline): a tool that floors a
# class under a constant matching NEITHER name form is invisible to this scan
# — `run.py`'s own GPU_UNATTRIBUTED_FLOOR shows the form exists. The four
# audited tools use the idiom without exception today.
FLOORED_CLASS_TOOLS = ("coverage", "champions", "review_queue", "decisions")

# Every declared class maps to the `ratchet_live` counter that carries it.
# Names here are asserted against the LIVE scan in `print_ratchet_block`, so
# an entry pointing at a counter nobody computes refuses by name.
FLOORED_CLASS_JOIN = {
    "coverage.UNREACHABLE_BASELINE": "unreachable",
    "coverage.FAIL_UNOWNED_BASELINE": "fail_unowned",
    "coverage.PASS_ON_DEAD_DEPENDENCY_BASELINE": "pass_on_dead_dependency",
    "coverage.COMMITMENTS_UNCOVERED_BASELINE": "commitments_uncovered",
    "coverage.GOAL_UNRUNNABLE_BASELINE": "goal_unrunnable",
    "coverage.PARK_RELEASE_BASELINE": "park_release_pairs",
    "champions.BASELINE_UNWINNABLE": "champions_unwinnable",
    "champions.BASELINE_TRIGGER_UNREACHABLE": "champions_trigger_debt",
    "decisions.BASELINE_UNDECLARED": "decisions_undeclared",
    "decisions.BASELINE_UNROUTED_ASKS": "decisions_unrouted_owner_ask",
    "decisions.BASELINE_VANISHED_ASKS": "decisions_vanished_owner_ask",
    "decisions.BASELINE_ACTION_EXPIRED": "decisions_default_action_expired",
    "decisions.BASELINE_FIRING_HAZARDS": "decisions_firing_diff",
}

# Pinned UNJOINED, each with its reason — measured at pin time, 2026-09-26.
# Every entry is declared and gated TOOL-SIDE (`--check` exits non-zero on
# growth) but carries NO committed reading in `run status`: no `at` date, no
# `!! MOVED` banner, no attributable cause when it moves. That is the exact
# deficiency FINDING 2 measured, and this pin does not call it good — it makes
# staying in it a VISIBLE DECISION instead of an accident. Joining any entry
# is one `take()` line in `ratchet_live` plus deleting its pin line in the
# same commit. A NEW floored class may not land here as a reflex: join it, or
# write here why not — either way the author of the constant pays the cost in
# this file, which is the FLOORED pin's own contract one block down.
FLOORED_CLASS_UNJOINED = {
    "coverage.GOAL_DANGLING_BASELINE":
        "empty frozenset since 09-01; a NEW dangler is a red exit in "
        "coverage.check, never a re-seed",
    "coverage.QUEUE_EMPTY_BASELINE":
        "empty frozenset since 09-04; an emptied cost class reads amber/red "
        "in coverage.check on its own",
    "champions.BASELINE_ARENA_MISSING":
        "at 0 since 09-25 (phantom arenas closed by REGISTRATION, the only "
        "non-laundering repair); growth is rc=1 tool-side",
    "champions.BASELINE_UNFALSIFIABLE":
        "at 3; sibling of the two joined champions classes — the 91st-audit "
        "repair joined the two the audits had watched move unseen",
    "champions.BASELINE_UNCONTESTABLE":
        "at 4; the SUM class (unfalsifiable + arena-unreachable), asserted "
        "tool-side so conversions between flavours stay neutral",
    "champions.BASELINE_UNDECLARED":
        "at 0; every seat declares SEAT:/HELD:/ARENA: today",
    "champions.BASELINE_VERDICT_UNVERIFIED":
        "at 2; growth is rc=1 tool-side",
    "champions.BASELINE_KINDLESS_DISCHARGES":
        "at 1; growth is rc=1 tool-side",
}


def declared_floored_classes() -> dict:
    """`{"tool.CONSTANT": value}` for every shrink-only class the four audit
    tools declare in their own source, enumerated by the declared idiom
    (module-level `BASELINE_*` / `*_BASELINE` names) rather than by a
    hand-kept list. `review_queue` contributes zero today — scanned anyway,
    so its first floored class is caught the day it ships."""
    import importlib
    out = {}
    for tool in FLOORED_CLASS_TOOLS:
        mod = importlib.import_module(f".{tool}", __package__)
        for name, val in vars(mod).items():
            if name.startswith("BASELINE_") or name.endswith("_BASELINE"):
                out[f"{tool}.{name}"] = val
    return out


def floored_class_gaps(declared, join, unjoined) -> list:
    """Every way the join map and the tools' declarations can disagree, named.
    Pure, so the self-check can plant known shapes (T0.31's assert-on-the-
    TOTAL discipline: the population is the tools' whole declaration set,
    never a filtered view of it)."""
    gaps = []
    dec = set(declared)
    for k in sorted(dec - set(join) - set(unjoined)):
        gaps.append(f"UNJOINED+UNPINNED: {k}")
    for k in sorted((set(join) | set(unjoined)) - dec):
        gaps.append(f"STALE MAPPING: {k}")
    for k in sorted(set(join) & set(unjoined)):
        gaps.append(f"JOINED AND PINNED AT ONCE: {k}")
    return gaps


def floor_status(cur, floor):
    """ABOVE / BELOW / AT, or None when there is no live value — pure, so
    the self-check can pin all four shapes."""
    if cur is None:
        return None
    if cur > floor:
        return "ABOVE"
    if cur < floor:
        return "BELOW"
    return "AT"


# 121st audit FINDING 2 / FTB 3. Floor state reached NO exit code in this
# tool: `print_ratchet_block` was pure output, `cmd_ratchets` returned 0
# unconditionally, and `cmd_status` had no ratchet branch at all — so a slot
# that printed `!! ABOVE its declared floor 0` and reported "status rc=0" in
# the same paragraph was telling the truth about the exit code and a falsehood
# about the machine. The split is `coverage.exit_code`'s, which is already this
# repo's convention and is quoted rather than re-invented:
#   RED (2)   — a shrink-only counter GREW with nobody raising the constant,
#               or a FLOORED counter whose live value refused to compute. The
#               second is red for the reason the block already says out loud
#               about LOST: a floored ratchet nobody can verify is the
#               instrument going quiet, not a quiet day.
#   AMBER (1) — the number FELL and the floor did not follow. A real defect
#               (the ratchet will accept a silent regression back up as clean)
#               but a bookkeeping one: it misleads nobody about a capability.
# Deliberately NOT wired here: MOVED / VANISHED / LOST on unfloored counters.
# The audit asked for floor state and floor state only, and a MOVED is legal
# in the very commit that records it — reddening it would teach the next
# iteration to ignore this exit code, which is the disease, not the cure.
def floor_report(name, cur, fl):
    """`(breach_bucket_or_None, printed_line)` for ONE floored counter.

    The printed banner and the exit-code bucket derive from a SINGLE
    `floor_status` call here, rather than being an `if/elif` chain that
    prints in one branch and appends in another. The parallel version is how
    a breach gets printed and not counted, which is the whole of the 121st
    audit's FINDING 2 — and the four shapes are pinned by known answer in
    `_check_ratchet_exit_wiring`, so a bucket cannot be dropped silently.
    """
    fs = floor_status(cur, fl)
    if fs == "AT":
        return None, f"        vs declared floor {fl}: AT floor — ok"
    if fs == "ABOVE":
        return "above", (
            f"        !! ABOVE its declared floor {fl} — growth nobody "
            f"raised the constant for. `ratchets record` cannot\n        "
            f"bless this; the floor moves only in the commit that grew the "
            f"number, with\n        the reason in its growth log.")
    if fs == "BELOW":
        return "below", (
            f"        !! BELOW its declared floor {fl} — the number fell "
            f"and the floor did not follow.\n        Lower the constant in "
            f"the same commit, or the ratchet will accept a silent\n        "
            f"regression back up as clean.")
    return "unverified", (                # no live value — cannot be verified
        f"        declared floor {fl}: live value unavailable, floor "
        f"UNVERIFIED this scan.")


def ratchet_exit_code(above, below, unverified) -> int:
    """`2` if any red condition is non-empty, else `1` for amber, else `0`.

    Pure and dict/list-shaped for the same reason `coverage.exit_code` is:
    so the battery can name the condition it is exercising, and so a reader
    can enumerate what turns this block red without reading the printer.
    """
    if above or unverified:
        return 2
    return 1 if below else 0


def committed_ratchet_readings() -> tuple:
    """`(readings, provenance)` as of HEAD. Read from git deliberately: an
    uncommitted rewrite of the readings file must not quiet the delta. The
    working-tree file answers only when git cannot, and the provenance
    string names which one did."""
    import json
    rel = RATCHET_READINGS.relative_to(_REPO).as_posix()
    try:
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=_REPO,
                              capture_output=True, text=True, timeout=10)
        if blob.returncode == 0:
            return json.loads(blob.stdout), "HEAD"
    except Exception:
        pass
    if RATCHET_READINGS.exists():
        return (json.loads(RATCHET_READINGS.read_text()),
                "WORKING TREE — the readings file is not in HEAD; commit it")
    return {}, "MISSING"


#: EVERY counter whose value can move with NO COMMIT — because the wall clock
#: is one of its inputs — and what such a movement MEANS. Named as a class
#: rather than discovered one member at a time: `DAY_SCOPED_COUNTERS` below was
#: a ONE-ELEMENT TUPLE for a class that had at least three members, and the
#: 95th audit's RANK 1 found the second one two hours after the builder wrote
#: the lesson *"when you sweep a class the READER transfers and the REPAIR does
#: not"* (`caa4257`).
#:
#: The distinction the members carry is NOT "clock-driven" — all of them are —
#: but **is the moving thing an EVENT or a WINDOW?**
#:
#:   event   the clock reaching a known instant IS the thing being reported.
#:           A promise breaking at midnight is a real event with a real owner;
#:           it must keep bannering and must never be suppressed.
#:   window  the clock moved the MEASURING APPARATUS, not the subject. The
#:           reading changed; nothing happened. This is the class that trains
#:           readers to skim.
#:
#: Membership is MEASURED, not asserted: with `docs/REVIEW_QUEUE.md` held at
#: frozen bytes and only `today` varied across 2026-09-11..18,
#: `review_queue_violations` moved 0/0/0/13/19/25/31/37 (OVERDUE arriving) and
#: `review_queue_net_arrivals` moved 17/15/11/8/7/8/8/11 (the baseline revision
#: sliding), while `review_queue_piled_on` held at 7 on all eight days.
#: `piled_on` is therefore NOT a member — the 95th audit named it as one on
#: inspection, and it reads no clock at all (`audit()` computes it from row
#: order and declared dates alone). `coverage.py` and `champions.py` read no
#: clock, so nothing they feed is in this class.
CLOCK_SENSITIVE_COUNTERS = {
    "cpu_foreclosed_now": "window",
    "review_queue_net_arrivals": "window",
    "review_queue_violations": "event",
}

#: Counters whose value is a point-in-time reading of a meter that RESETS at
#: 00:00 UTC. A delta across that boundary is the clock, not a change: the
#: `!! MOVED` banner fired four times for `cpu_foreclosed_now`'s nightly
#: 0 <-> ~40 swing (84th audit, LOOP_JOURNAL 11342/12205/12370 + the audit
#: itself), and a metric that is right every night and alarming every night
#: trains its readers to skim — the exact failure the 64th audit's block was
#: built to prevent, arriving from the opposite direction. Same-day movement
#: still banners; only the cross-day comparison is suppressed, out loud.
#:
#: THIS IS A SUBSET OF THE `window` CLASS ABOVE AND NOT THE WHOLE OF IT, which
#: is the 95th audit's point: suppression is only the right repair when the
#: counter is read MANY times a day, so that suppressing the cross-day
#: comparison still leaves the counter legible. `review_queue_net_arrivals` is
#: read ONCE a day, so suppressing its cross-day comparison would silence its
#: real movement too — it gets DECOMPOSED instead, never suppressed.
DAY_SCOPED_COUNTERS = ("cpu_foreclosed_now",)


def ratchet_deltas(live: dict, recorded: dict, today: str = "",
                   day_scoped: tuple = DAY_SCOPED_COUNTERS) -> list:
    """(name, kind, cur, prev, prev_at, note) for every counter either side
    knows. Kinds: MOVED / UNCHANGED / LOST (the live computation refused) /
    UNRECORDED (a live counter with no committed reading) / VANISHED (a
    committed reading whose counter is no longer computed at all) /
    DAY-ROLLED (a day-scoped counter whose committed reading is from a
    different UTC day — the comparison crosses the metric's own reset, so a
    delta is the clock and not a change).

    KNOWN BLIND SPOT, declared rather than papered over (85th audit §4): the
    day-roll branch fires on any `cur != prev` across the boundary, so a
    day-scoped meter that FAILED to reset is invisible either way — 39 -> 39
    reads UNCHANGED, 39 -> 20 reads DAY-ROLLED. The stronger form would
    assert the new value is consistent with a reset having happened; worth
    one line if a failed reset is ever observed, not a rewrite before."""
    out = []
    for name in sorted(set(live) | set(recorded)):
        rec = recorded.get(name)
        prev = rec.get("value") if isinstance(rec, dict) else None
        prev_at = rec.get("at", "?") if isinstance(rec, dict) else None
        if name not in live:
            out.append((name, "VANISHED", None, prev, prev_at, ""))
            continue
        cur, note = live[name]
        if cur is None:
            out.append((name, "LOST", None, prev, prev_at, note))
        elif rec is None:
            out.append((name, "UNRECORDED", cur, None, None, note))
        elif (name in day_scoped and today and prev_at != today
              and cur != prev):
            out.append((name, "DAY-ROLLED", cur, prev, prev_at, note))
        elif cur != prev:
            out.append((name, "MOVED", cur, prev, prev_at, note))
        else:
            out.append((name, "UNCHANGED", cur, prev, prev_at, note))
    return out


def ratchet_splits(rows: list) -> dict:
    """`name -> clock/act decomposition` for the counters that have one.

    Only `review_queue_net_arrivals` does today, and deliberately so: a
    decomposition is only meaningful for a `window` member of
    `CLOCK_SENSITIVE_COUNTERS`, and the other window member
    (`cpu_foreclosed_now`) is already suppressed because it is read many times
    a day. A refusal is RECORDED as a refusal rather than swallowed — an
    instrument that goes quiet is a fault, not a clean reading (the rule
    `review_queue.py` already applies to its own absent baseline).
    """
    import datetime as _dt
    out: dict = {}
    for name, _kind, _cur, prev, prev_at, _note in rows:
        if name != "review_queue_net_arrivals" or not isinstance(prev, int):
            continue
        try:
            from . import review_queue as rq
            out[name] = rq.live_net_arrivals_split(
                prev, _dt.date.fromisoformat(str(prev_at)))
        except Exception as exc:
            out[name] = {"refused": f"{type(exc).__name__}: {exc}"}
    return out


#: 103rd audit item 3 — the cause bucket each violation class's ARRIVAL
#: belongs to, so a `review_queue_violation_forms` delta names WHY it moved
#: instead of leaving the next reader to guess (the 09-19 guess was "a
#: midnight CLOCK movement" and the truth was the builder's own MALFORMED
#: row). Three buckets, per the audit, because the 102nd's clock/act pair
#: could not hold that case:
#:
#:   clock          the calendar reached a date or an age. OVERDUE and STALE
#:                  arrive this way and no commit is to blame.
#:   act            an edit ELSEWHERE moved the row's ground: the one member
#:                  is HOLD-ON-A-RESOLVED-BLOCKER, where someone resolved the
#:                  blocker and the row did not follow.
#:   self-inflicted the writing commit itself created the violation — every
#:                  grammar class. Its author is in `git log` on the queue
#:                  file, which is where the 09-19 correction had to be dug
#:                  from after the wrong cause was journalled.
#:
#: SHRINK in any class is always an ACT: no violation unbreaks at midnight —
#: OVERDUE clears by a re-date or disposal, grammar clears by a repair, all
#: of them commits. Direction is handled in `violation_form_lines`, not here.
#: Completeness is pinned in `_check_ratchet_reader` against the queue's own
#: VIOLATIONS tuple, so a new class without a bucket refuses by name.
VIOLATION_CAUSE_BUCKETS = {
    "OVERDUE": "clock",
    "STALE": "clock",
    "HOLD-ON-A-RESOLVED-BLOCKER": "act",
    "MALFORMED": "self-inflicted",
    "HOLD-WITHOUT-A-CLOCK": "self-inflicted",
    "VANISHED": "self-inflicted",
    "CLOCK-REMOVED": "self-inflicted",
    "ACTED-WITHOUT-A-COMMIT": "self-inflicted",
    "UNDECLARED-ROW": "self-inflicted",
}


def violation_form_lines(cur, prev) -> list:
    """Attribution lines for a `review_queue_violation_forms` movement —
    one per component delta, each naming its cause bucket. Pure, so the
    self-check can pin the shapes. Empty when either side is not a dict
    (an UNRECORDED first reading has no delta to attribute) or nothing
    moved. Growth takes the class's declared arrival bucket; shrink is
    always an act (see VIOLATION_CAUSE_BUCKETS — nothing unbreaks at
    midnight). An undeclared class is reported, never guessed."""
    if not isinstance(cur, dict) or not isinstance(prev, dict):
        return []
    out = []
    for cls in sorted(set(cur) | set(prev)):
        d = cur.get(cls, 0) - prev.get(cls, 0)
        if d == 0:
            continue
        if d < 0:
            out.append(f"        {cls} {d:+d} — ACT: a violation only "
                       f"clears by a commit (re-date, disposal or repair).")
            continue
        bucket = VIOLATION_CAUSE_BUCKETS.get(cls)
        if bucket == "clock":
            out.append(f"        {cls} {d:+d} — CLOCK: the calendar reached "
                       f"a date; a real event with a real owner,\n        "
                       f"and no commit is to blame for the rise.")
        elif bucket == "act":
            out.append(f"        {cls} {d:+d} — ACT: an edit elsewhere "
                       f"moved this row's ground.")
        elif bucket == "self-inflicted":
            out.append(f"        {cls} {d:+d} — SELF-INFLICTED: this class "
                       f"is only ever created by the writing\n        "
                       f"commit; its author is in `git log` on the queue "
                       f"file, not in the calendar.")
        else:
            out.append(f"        {cls} {d:+d} — UNDECLARED CLASS: no cause "
                       f"bucket in VIOLATION_CAUSE_BUCKETS.\n        "
                       f"Declare one; a guessed cause is the 09-19 error "
                       f"again.")
    return out


def clock_act_lines(kind: str, split: dict | None) -> tuple:
    """`(headline suffix, extra lines)` for a counter whose movement has been
    decomposed into a CLOCK and an ACT component. Pure, so the self-check can
    pin every shape.

    Two shapes carry the whole repair and they point in opposite directions:

      MOVED with `act 0`   — the banner is the calendar. One clause to report
                             it, instead of an iteration spent looking for the
                             commit that did it (three have been).
      UNCHANGED with an
      `act` that is not 0  — **the dangerous one.** A real routing cancelled by
                             an equal-and-opposite clock drift. Before this, it
                             printed the same `(unchanged since …)` as a quiet
                             day, which is the 2026-09-04 blindness the counter
                             was built to end, reproduced by the counter.
    """
    if split is None:
        return "", []
    if "refused" in split:
        return "", [f"        (clock/act split refused: {split['refused']} — "
                    f"no split is evidence; the delta above is undecomposed)"]
    clock, act = split["clock"], split["act"]
    if kind == "MOVED":
        suffix = f" (clock {clock:+d}, act {act:+d})"
        if act == 0:
            return suffix, [
                "        the whole movement is the sliding trailing window — "
                "no act, nothing to\n        investigate, and no commit can "
                "justify recording it."]
        return suffix, []
    if kind == "UNCHANGED" and act != 0:
        return "", [
            f"        !! UNCHANGED IS A CANCELLATION: act {act:+d} against "
            f"clock {clock:+d}. The desk moved\n        and the calendar "
            f"moved it back. This line is the counter going quiet on exactly "
            f"the\n        event it exists to report — read the act, not the "
            f"total."]
    return "", []


def _check_ratchet_reader() -> None:
    """Plant one counter per class and require the classifier to name each.
    The moved number is the scar (89-vs-85, read by nobody across five
    iterations — 64th audit); the quiet shapes must classify correctly too,
    or the block teaches iterations to ignore it."""
    live = {"a_moved": (89, ""), "b_same": (3, ""),
            "c_lost": (None, "boom"), "d_new": (2, ""),
            "f_dayroll": (0, ""), "g_daysame": (7, "")}
    rec = {"a_moved": {"value": 85, "at": "t0"},
           "b_same": {"value": 3, "at": "t1"},
           "c_lost": {"value": 1, "at": "t2"},
           "e_gone": {"value": 9, "at": "t3"},
           # the 84th-audit shapes: a day-scoped counter compared across the
           # metric's own midnight must suppress the banner; the same counter
           # moving WITHIN its day must still fire it.
           "f_dayroll": {"value": 39, "at": "yesterday"},
           "g_daysame": {"value": 4, "at": "today"}}
    got = ratchet_deltas(live, rec, today="today",
                         day_scoped=("f_dayroll", "g_daysame"))
    want = [("a_moved", "MOVED", 89, 85, "t0", ""),
            ("b_same", "UNCHANGED", 3, 3, "t1", ""),
            ("c_lost", "LOST", None, 1, "t2", "boom"),
            ("d_new", "UNRECORDED", 2, None, None, ""),
            ("e_gone", "VANISHED", None, 9, "t3", ""),
            ("f_dayroll", "DAY-ROLLED", 0, 39, "yesterday", ""),
            ("g_daysame", "MOVED", 7, 4, "today", "")]
    if got != want:
        raise RuntimeError(
            f"the ratchet reader returned {got}, expected {want} — "
            "refusing to report a scan it may not have performed")
    fgot = [floor_status(91, 90), floor_status(89, 90),
            floor_status(90, 90), floor_status(None, 90)]
    if fgot != ["ABOVE", "BELOW", "AT", None]:
        raise RuntimeError(
            f"the floor classifier returned {fgot} — refusing to report a "
            "floor comparison it may not have performed")
    # 95th audit B1. The two shapes that decide whether this repair works at
    # all: a MOVED that is pure calendar must SAY it is pure calendar, and an
    # UNCHANGED hiding a real act must stop reading like a quiet day.
    pure_clock = clock_act_lines("MOVED", {"clock": -3, "act": 0})
    cancelled = clock_act_lines("UNCHANGED", {"clock": -3, "act": 3})
    quiet = clock_act_lines("UNCHANGED", {"clock": 0, "act": 0})
    if (pure_clock[0] != " (clock -3, act +0)" or not pure_clock[1]
            or cancelled[0] != "" or "CANCELLATION" not in cancelled[1][0]
            or quiet != ("", [])
            or clock_act_lines("MOVED", None) != ("", [])
            or not clock_act_lines("MOVED", {"refused": "boom"})[1]):
        raise RuntimeError(
            "the clock/act reader mis-classified one of its four pinned "
            "shapes — refusing to report a decomposition it may not have "
            "performed")
    # 103rd audit item 3: the cause-bucket map must cover the queue's own
    # VIOLATIONS tuple exactly — a class added there without a bucket here
    # would be attributed by guesswork, which is the error the store exists
    # to end. Refuses by name, the FLOORED pin's idiom.
    from .review_queue import VIOLATIONS as _RQ_VIOLATIONS
    if set(VIOLATION_CAUSE_BUCKETS) != set(_RQ_VIOLATIONS):
        raise RuntimeError(
            f"VIOLATION_CAUSE_BUCKETS covers {sorted(VIOLATION_CAUSE_BUCKETS)} "
            f"but review_queue.VIOLATIONS is {sorted(_RQ_VIOLATIONS)} — a "
            "violation class without a declared cause bucket gets its cause "
            "guessed, which is the 09-19 error this store exists to end")
    # And the attribution shapes, pinned: the 09-19 excursion's growth leg
    # (MALFORMED +1 must read SELF-INFLICTED, not clock), a clock arrival,
    # a shrink (always an act), an undeclared class (reported, not guessed),
    # and the non-dict / no-delta quiet shapes.
    vf = violation_form_lines({"OVERDUE": 13, "MALFORMED": 1},
                              {"OVERDUE": 12, "STALE": 1})
    if (len(vf) != 3
            or "SELF-INFLICTED" not in vf[0] or "MALFORMED +1" not in vf[0]
            or "CLOCK" not in vf[1] or "OVERDUE +1" not in vf[1]
            or "ACT" not in vf[2] or "STALE -1" not in vf[2]
            or "UNDECLARED CLASS" not in violation_form_lines(
                {"NEW-CLASS": 1}, {})[0]
            or violation_form_lines(None, {}) != []
            or violation_form_lines({"OVERDUE": 12}, {"OVERDUE": 12}) != []):
        raise RuntimeError(
            "the violation-forms attributor mis-classified one of its pinned "
            "shapes — refusing to report a cause it may not have derived")
    # The class map is the thing that was a 1-tuple. Keep it honest about
    # itself: every suppressed counter must be a declared `window`, and no
    # member may carry a kind this file does not know how to act on.
    unknown = {n: k for n, k in CLOCK_SENSITIVE_COUNTERS.items()
               if k not in ("event", "window")}
    undeclared = [n for n in DAY_SCOPED_COUNTERS
                  if CLOCK_SENSITIVE_COUNTERS.get(n) != "window"]
    if unknown or undeclared:
        raise RuntimeError(
            f"clock-sensitive counters {unknown or ''}{undeclared or ''} are "
            "undeclared or wrongly classed — a suppression whose class is not "
            "declared is the 1-tuple again")
    # 101st audit FTB 1, the `_exit_code_fixture` idiom one level up: pin the
    # floored-counter set BY NAME, so deleting any single entry from
    # `ratchet_floors()` fails here by name rather than silently dropping the
    # floor line from the print. (Deleting the counter's `take()` in
    # `ratchet_live` is the other disconnection channel; `print_ratchet_block`
    # asserts floors ⊆ scanned rows against the LIVE scan, and a recorded
    # counter that stops being computed banners VANISHED.) A new floor is
    # added HERE in the same commit that declares its constant — that cost is
    # the point.
    FLOORED = {"unreachable", "fail_unowned", "pass_on_dead_dependency",
               "gpu_unattributed_jobs",
               "champions_unwinnable", "commitments_uncovered",
               # 120th audit FINDING 2: decisions.py's five ratcheted classes.
               "decisions_undeclared", "decisions_unrouted_owner_ask",
               "decisions_vanished_owner_ask",
               "decisions_default_action_expired", "decisions_firing_diff"}
    got_floors = set(ratchet_floors())
    if got_floors != FLOORED:
        raise RuntimeError(
            f"ratchet_floors() returned {sorted(got_floors)}, expected "
            f"{sorted(FLOORED)} — a floor that vanishes from the map is a "
            "disconnected ratchet, and one added without pinning here is the "
            "next one")
    # 120th audit item 3: the join must cover the TOOLS' OWN declarations,
    # not just its own map — a floored class shipped in any of the four audit
    # tools that neither joins `ratchet_live` nor pins its reason above
    # refuses HERE, by name, on every status print, instead of sitting red
    # for days with no committed reading (DEFAULT-ACTION-EXPIRED, 09-23 ->
    # 09-26). Planted control first, in this guard's own idiom: the checker
    # must name a class neither map covers, accept a covered population, and
    # name a mapping whose constant is gone — or nothing below is a check.
    _fx = {"toolx.BASELINE_NEW": 0, "toolx.OLD_BASELINE": frozenset()}
    if (floored_class_gaps(_fx, {"toolx.BASELINE_NEW": "n"}, {})
            != ["UNJOINED+UNPINNED: toolx.OLD_BASELINE"]
            or floored_class_gaps(_fx, {"toolx.BASELINE_NEW": "n"},
                                  {"toolx.OLD_BASELINE": "why"}) != []
            or floored_class_gaps({}, {"gone.X_BASELINE": "n"}, {})
            != ["STALE MAPPING: gone.X_BASELINE"]
            or floored_class_gaps({"t.B_BASELINE": 0},
                                  {"t.B_BASELINE": "n"},
                                  {"t.B_BASELINE": "why"})
            != ["JOINED AND PINNED AT ONCE: t.B_BASELINE"]):
        raise RuntimeError(
            "the floored-class join checker mis-classified one of its four "
            "planted shapes — refusing to report a coverage it may not have "
            "derived")
    gaps = floored_class_gaps(declared_floored_classes(),
                              FLOORED_CLASS_JOIN, FLOORED_CLASS_UNJOINED)
    if gaps:
        raise RuntimeError(
            "floored-class join broken: " + "; ".join(gaps) + " — a "
            "shrink-only class must be joined into ratchet_live (one take() "
            "line) or pinned in FLOORED_CLASS_UNJOINED with its reason, in "
            "the same commit that declares its constant")


def _check_ratchet_exit_wiring() -> None:
    """The durable half of the 121st audit's FTB 3, and it is the WIRING that
    matters, not the three-line function.

    `print_ratchet_block` printed `!! ABOVE its declared floor` for three days
    while `cmd_status` returned 0, and nothing in this repo could say so —
    because no fixture asserted that a floor state reaches an exit code. The
    defect was not a wrong branch; it was a MISSING EDGE between a printer and
    its caller, and a missing edge is invisible to any test of either end.

    So this guard does two things, in `coverage._exit_code_fixture`'s idiom:
    a per-condition battery through the REAL `ratchet_exit_code` (each term
    individually load-bearing, red dominating amber), and a STATIC read of
    this file asserting that both callers still take the block's RETURN and
    pass it on. Static because `print_ratchet_block` calls this guard, so a
    dynamic check would recurse into itself — the same reason
    `_exit_code_fixture` reads its own source.

    Pinned mutations, each of which left every other fixture in this repo
    green when tried before shipping:
      1. `cmd_status` ends `return 0`                 -> caught (no call)
      2. `ratchet_exit_code(above=[], ...)` literals  -> caught (CONSTANT)
      3. `breaches = {...}` built locally in `cmd_status` rather than taken
         from the block                               -> caught (not from
                                                         print_ratchet_block)
      4. `print_ratchet_block` stops returning        -> caught (no Return)
      5. any of the three terms dropped from the split -> caught by the battery
      6. the block's return taken and then REBOUND to a literal, or emptied
         in place (`breaches["above"] = []`)          -> caught (a name
                                                         qualifies only if
                                                         EVERY binding of it
                                                         is from the block)
      7. the collector deleted from the print loop, or narrowed to one
         bucket                                       -> caught by the
                                                         banner-vs-breach
                                                         reconciliation in
                                                         the block itself
      8. `floor_report` stops returning a bucket      -> caught (known answer)
      9. that reconciliation neutered (`for ... in []`) -> caught statically
                                                           here
    Mutations 6 and 7 both escaped earlier versions of this guard and are why
    the binding test is total rather than existential and why the
    reconciliation reads the EMITTED TEXT: in each case the escape hatch was
    the defect wearing the wiring's signature, the same shape LESSONS.md
    records for `HISTORY_EXEMPT_FIELDS` earlier the same day.

    THE RESIDUAL, named rather than papered over: this function can be
    deleted, or its call at the top of `print_ratchet_block` removed, and
    nothing here fires. A guard cannot guard its own deletion; that edge is
    `T0.13`'s (no gate in the ladder is decorative) and the git history's.
    """
    fails = []
    RED = ["above", "unverified"]
    AMBER = ["below"]
    clean = {k: [] for k in RED + AMBER}
    if ratchet_exit_code(**clean) != 0:
        fails.append("ratchet_exit_code: no breach must be 0")
    for k in RED:
        if ratchet_exit_code(**dict(clean, **{k: ["x"]})) != 2:
            fails.append(f"ratchet_exit_code: red `{k}` alone must exit 2 — "
                         f"a floor state that does not reach the exit code "
                         f"is the 121st audit's FINDING 2 again")
        if ratchet_exit_code(**dict(clean, **{k: ["x"]},
                                    **{a: ["y"] for a in AMBER})) != 2:
            fails.append(f"ratchet_exit_code: red `{k}` must dominate amber")
    for k in AMBER:
        if ratchet_exit_code(**dict(clean, **{k: ["x"]})) != 1:
            fails.append(f"ratchet_exit_code: amber `{k}` alone must exit 1")
    if set(RED) & set(AMBER):
        fails.append("ratchet_exit_code: a term cannot be red and amber")

    # The classifier that feeds it, all four shapes by known answer: the
    # bucket AND the banner, because the exit code is reconciled against the
    # banner and a bucket that stops being returned would otherwise only
    # show up as a quieter exit code.
    for _cur, _want_bucket, _want_mark in ((91, "above", "!! ABOVE"),
                                           (89, "below", "!! BELOW"),
                                           (90, None, "AT floor"),
                                           (None, "unverified",
                                            "UNVERIFIED")):
        _b, _line = floor_report("zz", _cur, 90)
        if _b != _want_bucket or _want_mark not in _line:
            fails.append(
                f"floor_report({_cur}, 90) returned ({_b!r}, ...) expecting "
                f"{_want_bucket!r} with {_want_mark!r} in the line — the "
                f"print and the exit code no longer share one classification")

    import ast as _ast
    try:
        _tree = _ast.parse(Path(__file__).read_text())
    except (OSError, SyntaxError, ValueError) as exc:      # pragma: no cover
        fails.append(f"ratchet_exit_code wiring: could not read own "
                     f"source: {exc}")
        _tree = None
    _fns = {}
    for _n in _ast.walk(_tree) if _tree is not None else []:
        if isinstance(_n, _ast.FunctionDef):
            _fns[_n.name] = _n
    # The printer must still HAVE a return value to wire.
    _blk = _fns.get("print_ratchet_block")
    if _blk is None:
        fails.append("ratchet_exit_code wiring: print_ratchet_block is gone")
    elif not any(isinstance(_r, _ast.Return)
                 and _r.value is not None
                 and not isinstance(_r.value, _ast.Constant)
                 for _r in _ast.walk(_blk)):
        fails.append("ratchet_exit_code wiring: print_ratchet_block no longer "
                     "returns its breaches — the callers have nothing to wire")
    if _blk is not None:
        # ...and the banner-vs-exit-code reconciliation must still be there.
        # It is what catches a collector deleted from the print loop, and a
        # guard is only load-bearing while it is executed: neutering its
        # iterable (`for name, line in []`) leaves every other check green.
        _recon = [_f for _f in _ast.walk(_blk)
                  if isinstance(_f, _ast.For)
                  and getattr(_f.iter, "id", None) == "emitted"
                  and any(isinstance(_r, _ast.Raise) for _r in _ast.walk(_f))]
        if not _recon:
            fails.append(
                "ratchet_exit_code wiring: print_ratchet_block no longer "
                "reconciles its printed banners against the breach set over "
                "`emitted` — without it a collector can be deleted and the "
                "banner keeps printing while the exit code goes quiet")
    for _name in ("cmd_status", "cmd_ratchets"):
        _fn = _fns.get(_name)
        if _fn is None:
            fails.append(f"ratchet_exit_code wiring: `{_name}` is gone")
            continue
        # Which locals are bound from `print_ratchet_block(...)` AND from
        # nothing else. "Bound from the block at least once" is not enough
        # and the first version of this guard made exactly that mistake:
        # `breaches = print_ratchet_block(ledger)` followed by
        # `breaches = {'above': [], ...}` left it green, which is the defect
        # wearing the wiring's signature (LESSONS.md 2026-09-26, one file
        # over). A name qualifies only if EVERY binding of it in this
        # function comes from the block, and it is never mutated in place.
        _bindings = {}
        _tainted = set()
        for _a in _ast.walk(_fn):
            if isinstance(_a, _ast.Assign):
                _ok = (isinstance(_a.value, _ast.Call)
                       and getattr(_a.value.func, "id", None)
                       == "print_ratchet_block")
                for _t in _a.targets:
                    if isinstance(_t, _ast.Name):
                        _bindings.setdefault(_t.id, []).append(_ok)
                    elif (isinstance(_t, _ast.Subscript)
                          and isinstance(_t.value, _ast.Name)):
                        _tainted.add(_t.value.id)   # breaches["above"] = []
            elif isinstance(_a, (_ast.AugAssign, _ast.AnnAssign,
                                 _ast.NamedExpr)):
                _t = getattr(_a, "target", None)
                if isinstance(_t, _ast.Name):
                    _bindings.setdefault(_t.id, []).append(False)
                elif (isinstance(_t, _ast.Subscript)
                      and isinstance(_t.value, _ast.Name)):
                    _tainted.add(_t.value.id)
        _from_block = {n for n, oks in _bindings.items()
                       if all(oks) and n not in _tainted}
        _calls = [_c for _c in _ast.walk(_fn) if isinstance(_c, _ast.Call)
                  and getattr(_c.func, "id", None) == "ratchet_exit_code"]
        if not _calls:
            fails.append(f"ratchet_exit_code wiring: `{_name}` no longer "
                         f"calls ratchet_exit_code — floor state reaches no "
                         f"exit code from it, which is exactly the defect")
            continue
        for _c in _calls:
            _args = [_kw.value for _kw in _c.keywords] + list(_c.args)
            if any(isinstance(_v, _ast.Constant)
                   or (isinstance(_v, (_ast.List, _ast.Tuple, _ast.Dict))
                       and not (getattr(_v, "elts", None)
                                or getattr(_v, "keys", None)))
                   for _v in _args):
                fails.append(f"ratchet_exit_code wiring: `{_name}` passes a "
                             f"CONSTANT — wired in appearance only")
            _names = {_v.id for _v in _args if isinstance(_v, _ast.Name)}
            if _names and not (_names & _from_block):
                fails.append(
                    f"ratchet_exit_code wiring: `{_name}` passes "
                    f"{sorted(_names)}, none of which is bound SOLELY from "
                    f"`print_ratchet_block(...)` — an exit code derived from "
                    f"something other than the printed block is a second "
                    f"opinion, not a receipt")
    if fails:
        raise RuntimeError(
            "the ratchet exit-code wiring failed its own battery: "
            + "; ".join(fails))


def print_ratchet_block(ledger: Ledger) -> dict:
    """Prints the block and RETURNS its floor breaches, so the caller can put
    them in an exit code (121st audit FTB 3). The return is a dict of lists
    keyed `above` / `below` / `unverified`, feeding `ratchet_exit_code`."""
    _check_ratchet_reader()
    _check_ratchet_exit_wiring()
    recorded, prov = committed_ratchet_readings()
    # gmtime, not localtime: the DAY-ROLLED print asserts "resets at 00:00
    # UTC", and a local-time `today` makes that a lie on any box whose TZ
    # drifts from GMT (85th audit §4 — latent here because TZ=GMT, fixed
    # before it is live anywhere else).
    rows = ratchet_deltas(ratchet_live(ledger), recorded,
                          today=time.strftime("%Y-%m-%d", time.gmtime()))
    floors = ratchet_floors()
    # 101st audit FTB 1: a floored counter that vanishes from the scan gets no
    # floor line and no banner — `if name in floors` below simply never fires.
    # Assert the join on the LIVE scan, so deleting a `take()` line in
    # `ratchet_live` after its reading is scrubbed refuses by name instead of
    # printing a block that quietly lost a ratchet. (While a reading is still
    # committed the same deletion banners VANISHED; this covers the remainder.)
    unscanned = sorted(set(floors) - {r[0] for r in rows})
    if unscanned:
        raise RuntimeError(
            f"floored counter(s) {unscanned} missing from the ratchet scan "
            "entirely — refusing to print a block that lost a ratchet")
    # 120th audit item 3, the live-scan half: a FLOORED_CLASS_JOIN entry that
    # names a counter nobody computes is a join in name only — the class
    # would read "covered" in the guard while no reading ever lands. Same
    # channel as the floors ⊆ rows assertion above, same reason.
    ghost_joins = sorted(set(FLOORED_CLASS_JOIN.values())
                         - {r[0] for r in rows})
    if ghost_joins:
        raise RuntimeError(
            f"FLOORED_CLASS_JOIN maps tool classes onto counter(s) "
            f"{ghost_joins} that the live ratchet scan does not compute — "
            "a paper join is the unjoined class wearing a name")
    splits = ratchet_splits(rows)
    breaches = {"above": [], "below": [], "unverified": []}
    emitted = []
    print("  RATCHET COUNTERS — standing-red tools' numbers, printed here so "
          "a blessed red\n    can never silence them (64th audit B2). "
          f"Committed readings from {prov}:")
    for name, kind, cur, prev, prev_at, note in rows:
        suffix, extra = clock_act_lines(kind, splits.get(name))
        if kind == "MOVED":
            d = (f"{cur - prev:+d}" if isinstance(cur, int)
                 and isinstance(prev, int) else f"was {prev}")
            print(f"      {name} = {cur}  !! MOVED {d}{suffix} since "
                  f"{prev_at} (was {prev}). Say so in your\n      report; if a "
                  f"committed change justifies it, `run ratchets record` in "
                  f"that commit.")
        elif kind == "DAY-ROLLED":
            print(f"      {name} = {cur}  (day-scoped: resets at 00:00 UTC; "
                  f"the committed reading {prev}\n      is from {prev_at}, a "
                  f"different UTC day, so the delta is the clock, not a "
                  f"change.\n      Movement within today would still banner.)")
        elif kind == "UNCHANGED":
            print(f"      {name} = {cur}  (unchanged since {prev_at})")
            _n = recorded.get(name)
            # `at` dates the KEY, never the break: a class can read ≥1 for
            # days before anyone creates its reading, and then this line
            # dates an old break to the day it was first recorded (121st
            # audit FINDING 2, second half). A `note` in the readings file
            # is where that history lives, and it is printed HERE because a
            # correction the block does not print needs an archaeologist.
            if isinstance(_n, dict) and _n.get("note"):
                print(f"        note: {_n['note']}")
        elif kind == "LOST":
            print(f"      {name}  !! computation refused ({note}); last "
                  f"committed reading {prev} at {prev_at}.\n      An "
                  f"instrument going quiet is a fault, not a quiet day.")
        elif kind == "UNRECORDED":
            print(f"      {name} = {cur}  (no committed reading — "
                  f"`run ratchets record`, then commit it)")
        else:  # VANISHED
            print(f"      {name}  !! recorded {prev} at {prev_at} and no "
                  f"longer computed at all — a counter\n      does not "
                  f"retire by disappearing.")
        for line in extra:
            print(line)
        if name == "review_queue_violation_forms" and kind == "MOVED":
            # 103rd audit item 3: the delta names its own cause bucket, so
            # the next reader is not left to guess it (and journal the guess).
            for line in violation_form_lines(cur, prev):
                print(line)
        if name == "fail_unowned" and cur is not None:
            # 73rd audit B2: the count went 4 -> 0 in three minutes by
            # routing into a queue whose own drain reads UNBOUNDED, and
            # `AT floor — ok` was the only thing this block printed. The
            # number stays; the map stops implying repair.
            try:
                from .coverage import fail_unowned as _fu
                owned = _fu()["owned"]
                forms = ["repaired_by", "disposed", "queue-row",
                         "held-on-blocker", "mention-only"]
                c = {k: sum(1 for v in owned.values() if v == k)
                     for k in forms}
                print("        owned: " + ", ".join(f"{c[k]} {k}"
                                                    for k in forms)
                      + " — a queue-row owner is a dated promise,\n"
                        "        not a repair; read the queue's own drain "
                        "before calling it handled (D23).")
            except Exception as exc:
                print(f"        (ownership breakdown refused: "
                      f"{type(exc).__name__}: {exc} — no map is evidence)")
        # The floor is the comparison a recording cannot quiet (65th audit
        # B3): `ratchets record` refreshes the readings file, but the floor
        # is a constant that only moves against its own growth log.
        if name in floors:
            bucket, line = floor_report(name, cur, floors[name])
            if bucket is not None:
                breaches[bucket].append(name)
            print(line)
            emitted.append((name, line))
    # The print and the exit code must agree, checked on the ACTUAL EMITTED
    # TEXT rather than on intent. Deleting the collection above leaves the
    # banner printing and the exit code quiet — which IS the 121st audit's
    # FINDING 2, reconstructible one line lower than where it was fixed. The
    # banner is the evidence a reader acts on, so it is the thing the exit
    # code is reconciled against.
    for name, line in emitted:
        for mark, bucket in (("!! ABOVE", "above"), ("!! BELOW", "below"),
                             ("UNVERIFIED", "unverified")):
            if mark in line and name not in breaches[bucket]:
                raise RuntimeError(
                    f"{name} printed `{mark}` and is not in the "
                    f"{bucket!r} breach set — a banner the exit code does "
                    f"not know about is the defect this return value exists "
                    f"to end")
    if breaches["above"] or breaches["below"] or breaches["unverified"]:
        print(f"    FLOOR STATE REACHES THE EXIT CODE (121st audit FTB 3): "
              f"{len(breaches['above'])} ABOVE, "
              f"{len(breaches['below'])} BELOW, "
              f"{len(breaches['unverified'])} UNVERIFIED — this tool exits "
              f"{ratchet_exit_code(**breaches)}.")
    print()
    return breaches


def cmd_ratchets(ledger: Ledger, record: bool = False) -> int:
    print()
    breaches = print_ratchet_block(ledger)
    if record:
        import datetime
        import json
        live = ratchet_live(ledger)
        lost = sorted(n for n, (v, _n) in live.items() if v is None)
        if lost:
            print(f"  refusing to record: {', '.join(lost)} refused to "
                  f"compute — a recording that\n  papers over a lost "
                  f"instrument is the silence this file exists to prevent.")
            return 2
        recorded, _prov = committed_ratchet_readings()
        today = datetime.date.today().isoformat()
        payload = {}
        for name, (val, _n) in sorted(live.items()):
            old = recorded.get(name)
            # Carry the entry FORWARD rather than rebuilding it from two
            # known keys. `value`/`at` are the only fields this tool computes,
            # but they are not the only fields an entry may legitimately hold
            # — `note` says WHEN the class broke, which `at` (the key's
            # creation date) structurally cannot. Rebuilding erased it on the
            # next record, which is the 121st audit's FINDING 1 exactly one
            # file over: a hand-written field list silently drops every field
            # added after it was written.
            entry = dict(old) if isinstance(old, dict) else {}
            at = (entry.get("at", today) if entry.get("value") == val
                  else today)
            entry.update(value=val, at=at)
            payload[name] = entry
        RATCHET_READINGS.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"  recorded {len(payload)} reading(s) to "
              f"{RATCHET_READINGS.relative_to(_REPO)} — commit it in the "
              f"same motion as the\n  change that moved the counter.")
    # A recording does NOT bless a floor breach — the block says so in prose
    # and now says it in the exit code too.
    return ratchet_exit_code(**breaches)


def gpu_orphans() -> list:
    """`gpu.orphaned_dispatches()` behind a local import, so `status` does not
    pay gpu.py's import unless it is about to print the block anyway."""
    from .gpu import orphaned_dispatches
    return orphaned_dispatches()


def _check_orphan_detector() -> None:
    """Plant a known orphan and require the detector to find it — and plant the
    three shapes that must NOT fire (live watcher, matched result, recovered
    orphan), because a detector that alarms on a healthy mid-run watcher would
    teach iterations to ignore the block (`_check_stale_detector`'s rule, one
    receipt log over)."""
    from .gpu import orphaned_dispatches
    rows = [
        # recovered orphan: dead attempt, then a later reattach pair — quiet
        {"phase": "attempt", "spec": "ZZ.RECOVERED", "pid": 1, "ts": 1.0,
         "backend": "kaggle", "iso": "t0"},
        {"phase": "attempt", "spec": "ZZ.RECOVERED", "pid": 2, "ts": 2.0,
         "backend": "kaggle", "iso": "t1", "attempt_id": "a2"},
        {"phase": "result", "spec": "ZZ.RECOVERED", "attempt_id": "a2"},
        # live watcher mid-run — quiet
        {"phase": "attempt", "spec": "ZZ.LIVE", "pid": 7, "ts": 3.0,
         "backend": "kaggle", "iso": "t2"},
        # the scar: dead watcher, no result, last word for its spec — fires
        {"phase": "attempt", "spec": "ZZ.ORPHAN", "pid": 8, "ts": 1788304286.8,
         "backend": "kaggle", "iso": "t3"},
    ]
    hits = orphaned_dispatches(rows, pid_alive=lambda p: p == 7)
    got = [h["spec"] for h in hits]
    if got != ["ZZ.ORPHAN"]:
        raise RuntimeError(
            f"the orphan detector returned {got}, expected ['ZZ.ORPHAN'] — "
            "refusing to report a scan it may not have performed")
    if hits[0]["slug"] != "jack-ladder-1788304286":
        raise RuntimeError(
            "the orphan detector's reattach slug does not reconstruct from "
            "the attempt ts; the printed recovery command would be wrong")


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

    # THE DIRTY BUCKET HAS FOUR SUB-STATES AND THREE OF THEM READ ZERO TODAY
    # (2026-09-13: one live dirty row, COMMITTED). That is the same shape this
    # function's docstring already argues about the bucket as a whole — a
    # sub-state whose known-positive has never been seen is not evidence of
    # anything, and the sub-state is what decides whether a re-run is OWED.
    # Each plant is a constructed case, and each is RED-FIRST in the only
    # sense available to a classifier: it asserts the state the row cannot be
    # in unless the branch that names it actually ran.
    victim_path = _module_path_for(victim)
    committed_sha, where = None, None
    for sid in (victim, *(s.id for s in LADDER)):
        e, p = ledger.results.get(sid), _module_path_for(sid)
        rec = getattr(e, "impl_sha", None) if e is not None else None
        if not rec or p is None:
            continue
        if tree_reconstructing_sha(p, rec)[1]:
            committed_sha, where = rec, p
            break
    if committed_sha is None:
        raise RuntimeError(
            "no ledger row's impl_sha reconstructs from any commit, so the "
            "COMMITTED branch of dirty_recoverability cannot be exercised — "
            "refusing to report a dirty scan whose commonest verdict is untested")
    for want, entry, p in (
            ("COMMITTED", _DirtyProbe("abc1234+dirty", committed_sha), where),
            ("PRESERVED", _DirtyProbe("abc1234+dirty", "f" * 16,
                                      preserved_impl="refs/jack/failimpl/X"),
             victim_path),
            ("LOST",      _DirtyProbe("abc1234+dirty", "f" * 16), victim_path),
            ("UNSTAMPED", _DirtyProbe("abc1234+dirty", None), victim_path)):
        got = dirty_recoverability(entry, p)[0]
        if got != want:
            raise RuntimeError(
                f"dirty_recoverability returned {got} for a constructed "
                f"{want} row; the dirty report's sub-states are not the states "
                "it names, and a re-run prescription derived from them is noise")


class _DirtyProbe:
    """A constructed dirty ledger row, for the known-positive plants above.

    Deliberately not a `Result`: the plants must be readable as "these four
    fields are all the classifier may look at", and a real Result would let a
    future edit quietly start reading a fifth.
    """

    def __init__(self, commit, impl_sha, preserved_impl=None, dirty_files=None):
        self.commit = commit
        self.impl_sha = impl_sha
        self.preserved_impl = preserved_impl
        self.dirty_files = dirty_files


def cmd_stale(ledger: Ledger) -> int:
    _check_stale_detector(ledger)
    rows = stale_claims(ledger)
    changed = [r for r in rows if r[2] == "CHANGED"]
    unstamped_changed = [r for r in rows if r[2] == "UNSTAMPED_CHANGED"]
    intact = [r for r in rows if r[2] == "UNSTAMPED_INTACT"]
    moved = [r for r in rows if r[2] == "UNVERIFIABLE_MOVED"]
    unknown = [r for r in rows if r[2] == "UNVERIFIABLE"]
    dirty = [r for r in rows if r[2] == "DIRTY"]
    declare = _unstamped_deps_denominator(unstamped_changed, intact, unknown)
    if dirty:
        print(f"\n{len(dirty)} claim(s) recorded from a MODIFIED tree — the "
              f"runner could not name\nwhat it executed:\n")
        for sid, st, _, detail in dirty:
            print(f"  {sid:8} {st:7} {detail}")
        print("\nStill worse than CHANGED: CHANGED names a commit by "
              "construction, and a dirty\nstamp only sometimes does. Each row "
              "above says which case it is and what it owes.")
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
    if unstamped_changed:
        print(f"\n{len(unstamped_changed)} pre-`impl_sha` claim(s) whose file "
              f"git shows CHANGED since the run\n(declaration-free content "
              f"check, 15th audit B1) — stale, re-run ON PURPOSE:\n")
        for sid, st, _, detail in unstamped_changed:
            print(f"  {sid:8} {st:7} {detail}")
    if moved:
        print(f"\n{len(moved)} UNPROTECTED certificate(s) — recorded before "
              f"`impl_sha`, and a declared\ndependency has since moved. The "
              f"alarm they declare cannot fire; re-run ON PURPOSE:\n")
        for sid, st, _, detail in moved:
            print(f"  {sid:8} {st:7} {detail}")
    # Denominators, not just counts (15th audit B1): a detector that reports a
    # count must report the size of the population it examined, and how much of
    # that population its opt-in sibling could ever have seen.
    n_unstamped = len(unstamped_changed) + len(intact) + len(unknown)
    if n_unstamped:
        print(f"\nOf {n_unstamped} entr(y/ies) predating `impl_sha`: "
              f"{len(unstamped_changed)} stale by content, {len(intact)} "
              f"verified byte-identical by git,\n{len(unknown)} unanswerable; "
              f"{declare} of {n_unstamped} declare IMPL_DEPS (the opt-in "
              f"detector's whole domain).")
    if unknown:
        # Reported, never hidden: a skipped item that leaves the numerator
        # alone is how a clean scan and a scan that never ran become the same
        # number.
        print(f"\n{len(unknown)} entr(y/ies) could not be checked even by "
              f"content:\n")
        for sid, st, _, detail in unknown:
            print(f"  {sid:8} {st:7} {detail}")
    print()
    return 0


def _unstamped_deps_denominator(*row_groups) -> int:
    """How many of the pre-`impl_sha` entries declare IMPL_DEPS at all.

    The 15th audit's tell: `UNVERIFIABLE_MOVED`'s domain is declarations, and
    0 of 30 unstamped records carried one — the detector's domain and the
    at-risk population were disjoint by construction, and nothing printed the
    fraction. This is that fraction's numerator.
    """
    from .protocol import impl_deps_of
    n = 0
    for rows in row_groups:
        for sid, *_ in rows:
            path = _module_path_for(sid)
            if path is None:
                continue
            deps, _problem = impl_deps_of(path)
            if deps:
                n += 1
    return n


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

    Nonzero exit means a commitment has NO declared spec, OR every claim-kind
    spec it ever had is PARKED (a retirement is not coverage — 28th audit), or
    a declaration/PARKED marker is malformed. All are cheap to fix and
    expensive to leave; "covered but not passing" is normal and exits 0. The
    repair for a claim-dead commitment is a SUCCESSOR SPEC, never unparking.
    """
    from .coverage import check
    return check()


def cmd_review_queue(ledger: Ledger) -> int:
    """The backlog routed to the weekly Review — how many, how old, how late.

    `docs/REVIEW_QUEUE.md` held the rows from 2026-08-24 and had no reader, so
    on 2026-08-30 the Review's FULL run died after 11 minutes owing
    `w0-too-shallow`'s design, that row's own dated promise passed, and no
    number anywhere went red. See `experiments/review_queue.py`; gated as T0.31.

    Nonzero exit means a row is malformed, past a DUE: it declared, older than
    the consumer's whole schedule, HELD without a clock, HELD behind a blocker
    that has resolved, deleted since the last commit, or stripped of its DUE:
    while still live. All are repairable by an honest edit; the escape hatch is
    RE-ARMING with a new DUE: and a reason, never going quiet.
    """
    from .review_queue import check
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


# ── AWAITING: results owed by a detached launch (67th audit B2) ─────────────
# `scripts/launch_detached.sh` (under JACK_AWAITING_SPEC=<id>) writes one row
# per registered detached launch: `<spec>\t<since>\t<pid:starttime>\t<label>`.
# The scar: an iteration launched LF.01 detached, promised in PROSE that "the
# waiter will wake me", and exited; the row landed 8 minutes later with
# nothing scheduled to read it, and every instrument stayed green because no
# organ watches for the ABSENCE of a harvest. This is the reader that prose
# never was. Three states per row:
#   RESOLVED — the ledger has an entry for the spec with ran_at >= since:
#              the result landed; the row is pruned here.
#   PENDING  — the pid (verified by starttime, so a recycled pid cannot
#              satisfy it) is still alive: informational, never blocking.
#   UNRESOLVED — no ledger entry, no live pid. The run died or vanished with
#              its result unread; `next` REFUSES to select new work until
#              someone reads the log and harvests or records the loss.
AWAITING_PATH = os.environ.get("JACK_AWAITING", "/data/jack-logs/awaiting")


def _awaiting_key_alive(key: str) -> bool:
    pid, _, start = key.partition(":")
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().rsplit(") ", 1)[-1].split()
    except OSError:
        return False
    return len(rest) >= 20 and rest[19] == start


def _awaiting_check(ledger: Ledger):
    """(unresolved, pending) AWAITING rows; prunes resolved ones from disk."""
    try:
        with open(AWAITING_PATH) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    except OSError:
        return [], []
    keep, unresolved, pending = [], [], []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 3:              # unparseable row: surface, never drop
            unresolved.append((ln, "", "", "unparseable AWAITING row"))
            keep.append(ln)
            continue
        spec, since, key = parts[0], parts[1], parts[2]
        row = ledger.results.get(spec)
        if row is not None and (row.ran_at or "") >= since:
            continue                    # resolved: the result landed — prune
        keep.append(ln)
        if _awaiting_key_alive(key):
            pending.append((spec, since, key))
        elif spec not in BY_ID:
            # NOT A REGISTERED SPEC — so the RESOLVED branch above can never
            # fire for it, because `ledger.results` is keyed by spec id and a
            # probe by construction writes no ledger row. The refusal is still
            # correct (somebody owes a decision), but calling this "no ledger
            # row since launch" describes it as a dropped result when it is a
            # row that was never resolvable. Say which. (builder 2026-09-12,
            # after `probe:d10_twin_spread` blocked `next` for an iteration
            # whose result had already been harvested and committed twice.)
            unresolved.append((spec, since, key,
                               "NOT A REGISTERED SPEC — no ledger row can "
                               "ever resolve this; hand-clear only"))
        else:
            unresolved.append((spec, since, key,
                               "no ledger row since launch, pid gone"))
    if len(keep) != len(lines):
        tmp = AWAITING_PATH + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join(k + "\n" for k in keep))
        os.replace(tmp, AWAITING_PATH)
    return unresolved, pending


def cmd_next(ledger: Ledger) -> int:
    unresolved, pending = _awaiting_check(ledger)
    for spec, since, key in pending:
        print(f"AWAITING {spec} since {since} — pid {key} still running; "
              f"not blocking, but do not relaunch it.")
    if unresolved:
        print("\nREFUSED: a detached registered run owes a result nobody has "
              "read (67th audit B2 — the organ that would notice a dropped "
              "result is the one that exited):\n")
        for spec, since, key, why in unresolved:
            print(f"  AWAITING {spec} since {since} ({why})")
        print("\nHarvest it before selecting new work: read its launch log, "
              "commit the ledger row if one exists on disk, or record the "
              "loss (amend / lost_iterations.log) — then delete the row from "
              f"{AWAITING_PATH} in the same breath, and say so in the "
              "journal. This check refuses; it never harvests for you.")
        if any(spec not in BY_ID for spec, _s, _k, _w in unresolved):
            print("\nNOTE on the rows marked NOT A REGISTERED SPEC: those were "
                  "armed with a JACK_AWAITING_SPEC that is not a spec id — a "
                  "probe, a sweep, a label. `proc_await` accepts any string "
                  "and the RESOLVED branch here keys on `ledger.results`, so "
                  "such a row is unresolvable BY CONSTRUCTION and will block "
                  "`next` until hand-cleared however completely the work was "
                  "harvested. That is not a fault in the run and it is not a "
                  "lost result — verify the harvest landed, then delete the "
                  "line. Arming this row for a probe buys a hand-clear, not a "
                  "receipt; prefer leaving JACK_AWAITING_SPEC unset for work "
                  "that writes no ledger row.")
        return 3
    avail = ready(ledger)
    if not avail:
        print("Nothing runnable — every unblocked spec already passes.")
        return 0
    _check_next_triage()
    rows = _next_triage(avail, ledger)
    fresh = [r for r in rows if r["lane"] == "FRESH"]
    settled = [r for r in rows if r["lane"] == "SETTLED"]
    held = [r for r in rows if r["lane"] == "HELD"]
    # Say what is being hidden. `avail[:12]` silently dropped the rest, and the
    # cheapest unblocked work sorts LAST (ME.11.A sat behind twelve GPU specs),
    # so the one command an iteration runs to choose its work was quietly
    # answering a different question than the one it appears to answer.
    shown = min(12, len(rows))
    more = f" — showing {shown} of {len(rows)}" if len(rows) > shown else ""
    print(f"\nRunnable now (dependencies satisfied){more}:")
    print(f"  TRIAGE (95th audit, builder): {len(fresh)} fresh · "
          f"{len(settled)} carrying a settled verdict · {len(held)} held "
          f"(parked / foreclosed / decision-held). FRESH sorts first.")
    if not fresh:
        print("  NONE of these is fresh work. Every spec below has already "
              "returned a verdict or is held by a marker its own docstring, "
              "a pilot or an open decision wrote. `ready()` means "
              "'dependencies pass', never 'this is a legitimate dispatch' — "
              "read the hold, and do not re-run a settled row to fill a slot.")
    print()
    for r in rows[:12]:
        s = r["spec"]
        impl = "" if module_path_for(s.id, strict=True) else "  [needs implementing]"
        print(f"  {s.id}  {s.title}  ({s.budget.value}){impl}")
        print(f"        state:       {r['state']}")
        print(f"        hypothesis:  {s.hypothesis}")
        print(f"        falsified by: {s.falsified_by}")
        if s.kills:
            print(f"        kills:       {s.kills}")
        print()
    return 0


def _next_triage(avail, ledger, state_of=None, held_map=None) -> list:
    """Split `ready()`'s output into FRESH / SETTLED / HELD, and say which.

    THE SCAR IS THIS MORNING'S PRIORITY PAGE (builder, 2026-09-13 21:xx).
    `run next` is the second command the orientation tells every iteration to
    run, and the sentence beside it reads *"Take the FIRST one in priority
    order and finish it."* Measured against the live ledger tonight, it listed
    **44 specs and not one of them was a legitimate next move**: 24 FAIL,
    11 VOID, and all 9 remaining `NOT_RUN` held — `T2.11`/`T3.10`/`SM.02`/
    `SH.01` PARKED, `LC.07`/`SM.03`/`DP.04`/`SH.02` PILOT-BLOCKED, `HR.1`
    decision-HELD behind `D19`. Of the twelve it actually printed, eleven were
    settled and the twelfth (`T2.11`) was parked by its own pre-registered
    both-fail branch. Every one rendered identically to a never-run spec.

    WHAT IT COST, and it is not hypothetical. The Sunday FULL page ranked
    `T2.10` first this morning as *"CPU, ten minutes"*; `T2.10` is a settled
    FAIL whose own docstring — written at 20:15 the same day — records that
    every scorer this project has ever measured tops out at 0.0667 against its
    unmoving 0.10 bar, so *"RE-RUNNING THIS SPEC UNCHANGED RETURNS FAIL"* and
    the repair is a 15-certificate retrieval redesign. Three iterations
    derived "the board is empty" by hand from `coverage`, `blocked` and a
    docstring, because the tool that advertises work could not say it.

    THE READERS ALL EXISTED AND NOBODY ASKED THEM. `coverage._liveness_state`
    returns PARKED / VOID-FORECLOSED / PILOT-BLOCKED / welded, factored into
    one place by the 59th audit precisely so instruments cannot drift; and
    `decisions.holds()` opens its own docstring with *"so an instrument can
    refuse to advertise them as work."* `cmd_next` — the one command whose
    entire job is advertising work — asked neither. This is the 65th audit's
    lesson one file over: a blocker written as a sentence is invisible to
    every ranker until it becomes an edge. Here the edge existed; the ranker
    was the one organ not wired to it.

    REPORTING-ONLY AND UNFLOORED. Nothing here refuses a run, moves a
    threshold, marks a spec, or edits the ledger: `ready()` is unchanged and
    every spec it returns is still listed. The lanes only reorder and label.
    A settled row is still a legal thing to take — `T3.09`'s attempt 3 and
    `T1.08`'s re-run were both correct — and the lane says what it is, never
    that it is forbidden.

    `state_of`/`held_map` are injectable so the triage can be checked against
    a known answer without markers on disk — the seam `_terminal_blockers`
    named and never got (`_RANKER_FIXTURE`, T0.36's founding scar), built in
    on the first day this time rather than promised.
    """
    if state_of is None:
        from .coverage import _liveness_state
        state_of = _liveness_state([s.id for s in avail], BY_ID)
    if held_map is None:
        from .decisions import holds
        held_map = holds()
    lanes = {"FRESH": 0, "SETTLED": 1, "HELD": 2}
    rows = []
    for i, s in enumerate(avail):
        st = ledger.status(s.id)
        hold = state_of.get(s.id) or (
            f"decision-HELD {held_map[s.id]}" if s.id in held_map else None)
        if hold:
            lane, state = "HELD", f"HELD — {hold}  (ledger: {st.name})"
        elif st is Status.NOT_RUN:
            lane, state = "FRESH", "NOT_RUN — no verdict, no hold"
        else:
            lane, state = "SETTLED", (
                f"{st.name} — a verdict is already recorded; re-running it "
                f"unchanged buys the same row")
        rows.append({"spec": s, "lane": lane, "state": state,
                     "hold": hold, "status": st.name})
    # Stable: within a lane the incoming `ready()` order is preserved, so this
    # reorders nothing that anyone reasoned about — it only lifts the lane.
    rows.sort(key=lambda r: lanes[r["lane"]])
    return rows


def _check_next_triage() -> None:
    """Refuse to advertise work from a triage that flunks a known graph.

    Same rule as `_check_ranker` and `_check_blast_radius`: a lane count from
    an instrument that cannot get a fixture right is not evidence. Four
    known answers, one per branch, and the third is the one that matters —
    a spec can be NOT_RUN *and* held, and the hold must win, because that is
    exactly the shape of all nine of tonight's `NOT_RUN` rows.

      N  NOT_RUN, no hold          -> FRESH, and sorts FIRST despite being
                                      declared last in the fixture ladder.
      X  FAIL, no hold             -> SETTLED.
      P  NOT_RUN, PARKED           -> HELD, not FRESH. A triage that read the
                                      ledger alone would call this a fresh
                                      dispatch, which is the `T2.11` case.
      H  NOT_RUN, decision-held    -> HELD via `holds()`, the reader whose own
                                      docstring exists for this and which
                                      `cmd_next` never asked. `D19`/`HR.1`.
    """
    from .protocol import Spec

    def stub(sid):
        return Spec(sid, 0, sid, "h", "f", "n", "m", Budget.CPU_FAST)

    specs = [stub(s) for s in ("X", "P", "H", "N")]
    fixt = _fixture_ledger()
    fixt.results["X"] = Result(spec_id="X", status=Status.FAIL, metrics={},
                               seeds=[0], commit="1234567",
                               ran_at="2026-09-13T00:00:00")
    rows = _next_triage(specs, fixt,
                        state_of={"P": "PARKED"},
                        held_map={"H": "D19 (decide_by 2026-09-14)"})
    got = [(r["spec"].id, r["lane"]) for r in rows]
    want = [("N", "FRESH"), ("X", "SETTLED"), ("P", "HELD"), ("H", "HELD")]
    if got != want:
        raise AssertionError(
            f"_next_triage flunked its own fixture: {got} != {want}")


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
            # is merely not-yet-run is a root of its own — AND a dependency can
            # be BOTH, which is the case this line missed for a day and T0.36
            # now pins. `T1.08` fell on 2026-09-13 and `T2.01` — a settled FAIL
            # blocking 35 specs on its own account — was substituted away the
            # moment it acquired an unsatisfied dependency, crediting its whole
            # mass to the spec underneath it: `frees 41` where repairing
            # `T1.08` alone buys 3. A dependency carrying its own settled
            # verdict (FAIL/VOID/BLOCKED/ERROR, or a PASS gone stale) must
            # itself be repaired, so it is a root TOO, not instead.
            own = set() if ledger.status(d) is Status.NOT_RUN else {d}
            roots |= (upstream | own) or {d}
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


def _check_repair_edges(by_id=None) -> dict:
    """Validate every `repaired_by` declaration, and refuse the malformed ones.

    `{spec_id: [repair ids]}` for the declarations that survive, having dropped
    (loudly, via the returned refusals in `_repair_carry`) the two shapes that
    would make the reporting layer lie:

    - **an id that resolves to nothing.** The 59th audit's rule, in its own
      words: an id that resolves to a corpse is a WORSE dangling reference than
      one that resolves to nothing, because a reader stops checking. A repair
      edge is a pointer at work somebody is meant to go do; a pointer at a
      spec that does not exist sends them nowhere with confidence.
    - **a self-loop.** `X.repaired_by = ["X"]` substitutes a root for itself
      and would print "X carries its own mass", which is the ranker's founding
      double-count wearing a new hat.

    A repair id that is ALSO a `depends_on` of the same spec is legal and is
    NOT dropped: it is redundant rather than wrong (the ranker already sees
    that edge, so substitution changes nothing), and refusing a legal
    declaration is the failure mode `piled_on` was deliberately built to
    avoid — this is a metric layer, not a gate.
    """
    by_id = BY_ID if by_id is None else by_id
    good, bad = {}, {}
    for sid, spec in by_id.items():
        declared = list(getattr(spec, "repaired_by", []) or [])
        if not declared:
            continue
        keep = []
        for rid in declared:
            if rid == sid:
                bad.setdefault(sid, []).append(f"{rid} (self-loop)")
            elif rid not in by_id:
                bad.setdefault(sid, []).append(f"{rid} (not in the registry)")
            else:
                keep.append(rid)
        if keep:
            good[sid] = keep
    return {"edges": good, "refused": bad}


def _repair_carry(terminal: dict, ledger: Ledger, ladder=None, by_id=None,
                  edges=None) -> tuple:
    """`(mentions, frees, groups, refused)` for the ranking with every declared
    root SUBSTITUTED by its repair path — the 69th audit's B3.

    The arithmetic is `_rank_blockers`, unchanged and re-used rather than
    restated: substitute each terminal root `R` by `R.repaired_by` where one is
    declared, then ask the same two questions of the same code. That matters
    more than it looks. The ranker's own history is a double-count bug (`ranked
    by MENTIONS, and mentions double-count`) fixed by separating marginal
    `frees` from total `blocks`; a repair layer that computed its own carried
    mass would have to re-learn that lesson, and would drift the first time the
    ranker was touched. Two readers of one quantity share code or they drift.

    What comes back is REPORTING ONLY. `terminal` is copied before
    substitution, `_terminal_blockers` never sees a repair edge, and the
    `unreachable` count `coverage` ratchets against is computed from the
    unsubstituted walk exactly as before.
    """
    ladder = LADDER if ladder is None else ladder
    checked = _check_repair_edges(by_id)
    edges = checked["edges"] if edges is None else edges
    if not edges:
        return {}, {}, {}, checked["refused"]
    substituted = {}
    for sid, roots in terminal.items():
        out = set()
        for r in roots:
            # A self-root (dependency cycle) is dropped BEFORE substitution,
            # not after: `_rank_blockers` filters `r != s.id`, and a
            # substitution would smuggle a cycle past that filter as a repair.
            if r != sid:
                out |= set(edges.get(r, [r]))
        substituted[sid] = out
    mentions, frees, groups = _rank_blockers(substituted, ledger, ladder=ladder)
    # Only the repair specs themselves are this layer's business. Every other
    # root is already ranked, correctly, one section up — reprinting it here
    # under a heading about repair paths would double the board.
    repair_ids = {rid for ids in edges.values() for rid in ids}
    return ({k: v for k, v in mentions.items() if k in repair_ids},
            {k: v for k, v in frees.items() if k in repair_ids},
            {k: v for k, v in groups.items() if set(k) & repair_ids},
            checked["refused"])


def unreachable_count(ledger: Ledger, ladder=None, by_id=None,
                      mentions=None) -> tuple:
    """`(unreachable specs, ladder size)` — the number `blocked` prints.

    Factored out for the 58th audit's B3 so `coverage.unreachable_ratchet`
    and `cmd_blocked` read the SAME union of the SAME walk — the
    `_split_foreclosed` rule: two readers of one quantity share code or they
    drift. `mentions` is injectable for a caller that has already run the
    walk (`cmd_blocked`); everyone else gets a fresh one from the same two
    functions.
    """
    ladder = LADDER if ladder is None else ladder
    if mentions is None:
        terminal = _terminal_blockers(ledger, ladder=ladder, by_id=by_id)
        mentions, _, _ = _rank_blockers(terminal, ledger, ladder=ladder)
    return len({sid for ids in mentions.values() for sid in ids}), len(ladder)


class _AssumeStatus:
    """A read-only Ledger view with ONE spec's status overridden.

    Exists so `blast_radius` can ask the dependency walk a counterfactual
    without mutating the real ledger — only the runner writes that file, and a
    tool that answers "what if X failed" by briefly making X fail is one
    exception away from leaving it that way.

    `unsatisfied`/`blocked_by` are BORROWED from `Ledger`, not restated. That
    is the `_split_foreclosed` rule applied one level down: the freshness half
    of the dependency rule (a PASS whose `impl_sha` moved does not satisfy)
    lives inside `Ledger.unsatisfied`, and a re-implementation here would drift
    from it exactly the way `_terminal_blockers`' own hand-rolled
    `status is Status.PASS` did before `T0.22` retired it.
    """

    unsatisfied = Ledger.unsatisfied
    blocked_by = Ledger.blocked_by

    def __init__(self, ledger: Ledger, spec_id: str, status: Status):
        self._ledger, self._sid, self._status = ledger, spec_id, status
        self.results = ledger.results
        # ASSUMING **PASS** COULD NOT BE ANSWERED HONESTLY, AND FAILED TWO
        # DIFFERENT WAYS. `unsatisfied` is borrowed, and its PASS branch reads
        # `self.results[d]` to ask the freshness question. Forcing the STATUS
        # alone therefore left the real row underneath it, so a never-run root
        # raised `KeyError` (`HR.1`, `T2.11`) and a root whose row was stale
        # came back "PASS but stale" — still unsatisfied, so the counterfactual
        # answered "repairing it buys nothing" for `UB.10`, `T3.06` and `LF.01`,
        # all three CHANGED. `blast_radius` never saw either, because it only
        # ever asked "what if X FAILS"; the other direction went unexercised
        # until T0.36 asked what repairing X alone buys.
        #
        # Repaired-AND-RE-RUN is what this counterfactual MEANS, so PASS gets a
        # stand-in row stamped with the implementation that exists now and no
        # staleness kind fires. Still read-only: a private copy, and only the
        # runner writes the ledger.
        if status is Status.PASS:
            path = module_path_for(spec_id)
            self.results = dict(ledger.results)
            self.results[spec_id] = Result(
                spec_id=spec_id, status=Status.PASS, metrics={},
                control_metrics={}, seeds=[], history=[], amended=[],
                commit="assumed", impl_sha=impl_sha_of(path) if path else None)

    def status(self, spec_id: str) -> Status:
        return self._status if spec_id == self._sid else self._ledger.status(spec_id)


def blast_radius(spec_id: str, ledger: Ledger, assume: Status = None,
                 ladder=None, by_id=None) -> dict:
    """What LEAVES the reachable set if `spec_id` settles non-PASS.

    `protocol.BLAST_RADIUS_DECL` has required this quantity since the 54th
    audit (2026-08-31): a `VOID-FORECLOSED:` declaration is refused unless the
    docstring also carries *"the transitive set of specs the declaration
    renders unreachable, by id and title"*. That contract says the set is
    **"derivable from `depends_on`"** and then validates PRESENCE, NOT TRUTH —
    because for thirteen days nothing derived it. This function derives it.

    IT IS ALSO NEEDED WHERE NOTHING ASKS FOR IT, WHICH IS WHY IT IS HERE AND
    NOT IN `coverage.py`. A foreclosure declaration is not the only graph edit
    that strands downstream specs; **arming a new conjunct on a PASSing spec is
    the same edit with no paperwork at all**, and it fired twice in
    twenty-four hours:

      2026-09-13 06:44  `T6.03` re-run under a strengthened gate -> BLOCKED.
                        Radius `{LF.02}`. `UNREACHABLE_BASELINE` 93 -> 94.
      2026-09-13 10:05  `T1.08` re-run under a conjunct armed four hours
                        earlier -> FAIL. Radius **`{D1.0, T2.01, T2.02}`**,
                        `UNREACHABLE_BASELINE` 94 -> 97 — and it made `T1.08`
                        the project's LARGEST terminal blocker (frees 41 /
                        blocks 45, displacing `T2.01`) while foreclosing the
                        `D1.0` W37 dispatch that the same morning's priority
                        block had ordered for that day.

    Both radii were discovered AFTER the run, by hand, from a ratchet that had
    already gone red. Both were computable before it, with zero seeds, from
    the registry and the ledger alone — which is the 92nd audit's RANK 3
    reachability rule pointed the other way. Reachability asks *"can this bar
    clear?"*; this asks *"what falls if it does not?"* They are different
    questions and only the first had a tool.

    Returns `{"spec", "status", "assumed", "radius", "runnable_lost",
    "unbacked", "frees", "blocks", "before", "after", "ladder"}`. `radius` is
    the set of specs that are reachable today and are not under the
    counterfactual; `runnable_lost` is its sharpest subset — specs whose
    dependencies are satisfied RIGHT NOW, i.e. the dispatches that become
    illegal the moment the run lands. `frees`/`blocks` is the terminal-blocker
    rank the spec would take in `run blocked`.

    `unbacked` IS THE OTHER HALF OF THE PRICE, and until the 94th audit this
    function did not have it (RANK 1, 2026-09-13). Every set above is about
    specs that have NOT run. A graph edit also strands specs that HAVE —
    standing PASS certificates whose dependency stops being satisfied — and
    `unreachable` structurally cannot count them, because they are not
    unreachable. See `unbacked_certificates`; the entries are
    `(spec_id, [dep ids], cost class)` on the NON-PASS side of the
    counterfactual, so the same key answers "what would fall" for a green spec
    and "what is already down" for a red one.

    `assume` defaults to the informative counterfactual: `FAIL` for a spec
    that currently passes, `PASS` for one that does not — so the same command
    prices a strengthening before it is armed and a repair before it is
    bought.
    """
    ladder = LADDER if ladder is None else ladder
    by_id = BY_ID if by_id is None else by_id
    live_status = ledger.status(spec_id)
    if assume is None:
        assume = Status.FAIL if live_status is Status.PASS else Status.PASS

    def _reach(led) -> tuple:
        terminal = _terminal_blockers(led, ladder=ladder, by_id=by_id)
        mentions, frees, _ = _rank_blockers(terminal, led, ladder=ladder)
        stuck = {sid for ids in mentions.values() for sid in ids}
        return stuck, mentions, frees

    def _runnable(led) -> set:
        return {s.id for s in ladder
                if led.status(s.id) is not Status.PASS and not led.unsatisfied(s)}

    before, _, _ = _reach(ledger)
    alt = _AssumeStatus(ledger, spec_id, assume)
    after, mentions, frees = _reach(alt)
    lost_runnable = _runnable(ledger) - _runnable(alt)

    # The backing half. Taken as a DELTA between the two worlds rather than as
    # "certificates that name `spec_id`", so it picks up a certificate whose
    # dependency is satisfied in one world and not the other for any reason the
    # ONE definition recognises — and so it cannot report a certificate that
    # was already unbacked before this edit as a cost OF the edit.
    def _unbacked(led) -> dict:
        return {sid: (deps, cost, roots)
                for sid, deps, cost, roots in unbacked_certificates(
                    led, ladder=ladder, by_id=by_id)}

    ub_pass, ub_fail = ((_unbacked(alt), _unbacked(ledger)) if assume is
                        Status.PASS else (_unbacked(ledger), _unbacked(alt)))
    unbacked = sorted((sid,) + ub_fail[sid]
                      for sid in set(ub_fail) - set(ub_pass)
                      if sid != spec_id)

    # THE SUBJECT IS NEVER ITS OWN RADIUS — but it IS its own count. Both sets
    # move trivially for `spec_id` itself (a spec assumed PASS leaves the stuck
    # set by definition), and printing that read as a finding: the first run of
    # this command announced "REGAINED: T6.03" under the heading "T6.03", and
    # "T1.08 is a dispatch that becomes illegal" under "T1.08". Neither is
    # false; both are the question restated as its own answer — the
    # double-count `_rank_blockers` exists to kill, wearing the
    # counterfactual's clothes.
    #
    # The exclusion is applied to the NAMED sets ONLY. `before`/`after` are the
    # number `run blocked` and `coverage.unreachable_ratchet` print, and a
    # count that quietly drops its subject would disagree with both — the
    # `_split_foreclosed` drift, re-introduced by a cosmetic fix. T6.03 is
    # itself unreachable today, so this distinction is load-bearing, not
    # hypothetical.
    named_before, named_after = before - {spec_id}, after - {spec_id}

    return {
        "spec": spec_id,
        "status": live_status,
        "assumed": assume,
        # Named for the direction of HARM regardless of which way the
        # counterfactual runs: `radius` is always "OTHER specs stranded by the
        # assumed status, relative to today".
        "radius": sorted(named_after - named_before),
        "regained": sorted(named_before - named_after),
        "runnable_lost": sorted(lost_runnable - {spec_id}),
        "unbacked": unbacked,
        "frees": sorted(frees.get(spec_id, [])),
        "blocks": sorted(mentions.get(spec_id, [])),
        "before": len(before),
        "after": len(after),
        "ladder": len(ladder),
    }


def unbacked_certificates(ledger: Ledger, ladder=None, by_id=None) -> list:
    """`(spec_id, [dep ids], cost class, [(root, status, cost)])` for every
    standing PASS certificate that cannot be re-derived today, sorted by id.

    THE HALF `blast_radius` WAS SILENT ABOUT, and the 94th audit's RANK 1
    (2026-09-13). A gate edit is a graph edit, and it strands specs in two
    different ways:

      REACHABILITY   a spec that could have been run can no longer be run.
                     `blast_radius` prices this, and `unreachable` ratchets it.
      BACKING        a spec that has ALREADY run and holds a PASS now rests on
                     a dependency that no longer passes. Nothing prices this,
                     and **no ratchet can**: `unreachable` counts specs that
                     cannot be run, and these have already run, so they are not
                     unreachable — they are UNBACKED. The one number that moved
                     when `T1.08` fell gave false comfort that the cost had
                     been counted.

    It cost three certificates to find. On 2026-09-13 `demonstrated` read 108
    and `T2.03` (gpu<20min), `T2.14` (gpu<2h) and `LF.02` (cpu<10min) could not
    be re-derived — two of them created four hours earlier by a strengthening
    that priced its reachability half correctly and could not see this one.
    The cost class is reported for exactly that reason: a re-buy is a GPU
    dispatch two times in three, and "three certificates" and "three
    certificates, two of them GPU" are different bills.

    REPORTING-ONLY AND UNFLOORED, deliberately. A certificate standing on a
    fallen dependency is a LEGAL state — the run happened, the row is honest,
    and nothing about the past is invalidated by the present. What is true is
    that `run_spec` would refuse to re-derive it today. That is a reading
    somebody must be able to see, not a violation, and a ratchet on it would
    make every honest strengthening look like damage.

    THE RULE IS `Ledger.unsatisfied`, NOT `status is not PASS`. The audit
    derived the class by hand the second way and both readings return the same
    three specs today — but they are not the same rule, and the difference
    binds in one direction only: a dependency that is PASS with a MOVED
    `impl_sha` satisfies the naive test and fails this one, and a certificate
    resting on it is equally un-re-derivable. Sharing `unsatisfied` is also the
    `_split_foreclosed` rule (two readers of one quantity share code or they
    drift) — this is the same question `run_spec` refuses on, so it is asked
    through the same function.

    THE ROOT IS REPORTED BESIDE THE HOP, and the first live reading is why
    (builder, 2026-09-13 ~20:xx, two hours after this function shipped). The
    version that shipped at 18:4x named the FIRST unsatisfied dependency and
    stopped, so its three rows read `LF.02 (cpu<10min) needs T6.03`. But
    `T6.03` is itself BLOCKED behind `T2.10`, whose own docstring carries a
    pre-registered reachability table proving no scorer this project has ever
    measured clears its bar — the repair is an `ME.11`-class retrieval
    redesign plus a 15-certificate re-buy. The one-hop reading priced that
    certificate at one `cpu<10min` re-run.

    That is the day's own headline lesson reproduced INSIDE the instrument
    built to prevent it: *a cost class is a statement about the RUN, a
    priority is a statement about the REPAIR.* The first unsatisfied
    dependency is the one you can SEE; the root of the chain is the one you
    have to PAY. Both are printed, because the hop is what `run_spec` names
    when it refuses and the root is what has to happen first — and a reader
    must be able to tell when they are the same spec. Today two of the three
    rows bottom out in one hop and exactly one does not, which is also why a
    root-only reading would have been the wrong repair: it would have lost the
    fact that `T2.03` and `T2.14` really are one dispatch away.

    A root is a node on the chain with no unsatisfied dependency of its own;
    the walk stops there and at any id outside `by_id`, and `seen` bounds a
    registry that ever admits a cycle. The root's OWN status and cost class
    travel with it for the reason the certificate's did: `T2.10 [FAIL]
    (cpu<10min)` and `T1.08 [FAIL] (gpu<2h)` are different bills.
    """
    ladder = LADDER if ladder is None else ladder
    by_id = BY_ID if by_id is None else by_id

    def _roots(sid: str, seen: set) -> set:
        spec = by_id.get(sid)
        if spec is None or sid in seen:
            return set()
        seen.add(sid)
        deps = sorted(d for d, _why in ledger.unsatisfied(spec))
        if not deps:
            return {(sid, ledger.status(sid).value, spec.budget.value)}
        out = set()
        for d in deps:
            out |= _roots(d, seen)
        return out

    out = []
    for s in ladder:
        if ledger.status(s.id) is not Status.PASS:
            continue
        deps = sorted(d for d, _why in ledger.unsatisfied(s))
        if deps:
            roots = set()
            for d in deps:
                roots |= _roots(d, {s.id})
            out.append((s.id, deps, s.budget.value, sorted(roots)))
    return sorted(out)


def _check_unbacked_detector() -> None:
    """Red-first: plant a certificate on a fallen dependency and require the
    reader to name it, WITH its cost class — and plant the three shapes it must
    stay quiet about, because a reader that flags every PASS is the screen
    `D27` already measured crying wolf at 104 of 107.

    KNOWN ANSWERS, on a graph built for this and nothing else:

      UB.CERT   PASS, depends on UB.RED (FAIL)        -> REPORTED, gpu<20min
                                                         hop == root == UB.RED
      UB.DEEP   PASS, depends on UB.MID (FAIL), which
                depends on UB.RED (FAIL)              -> REPORTED,
                                                         hop UB.MID, ROOT UB.RED
      UB.STALE  PASS, depends on a PASS whose file
                has moved since the run               -> REPORTED
      UB.OK     PASS, depends on UB.GREEN (clean PASS)-> SILENT
      UB.NOPASS FAIL, depends on UB.RED (FAIL)        -> SILENT: this class is
                about STANDING CERTIFICATES. A red spec on a red dependency is
                already counted by `unreachable`, and printing it here would
                double-count the one number this reading exists to complement.
      UB.ROOT   PASS, no dependencies                 -> SILENT

    `UB.STALE` is the conjunct that separates this rule from the hand-derived
    one it replaces. Its dependency IS in PASS; it is the freshness half of
    `unsatisfied` that refuses it, and a deriver written as
    `status is not Status.PASS` passes every other conjunct here and fails this
    one. It is planted for that reason and no other.

    `UB.DEEP` is the conjunct that separates the root reading from the one-hop
    reading that shipped this afternoon, and it is planted for that reason and
    no other. It is the fixture shape of today's `LF.02 -> T6.03 -> T2.10`:
    a certificate whose visible dependency is NOT what has to be repaired
    first. Both directions are gated — `UB.DEEP` must name `UB.RED` as its
    root while `UB.CERT`, one hop deep, must name `UB.RED` as BOTH hop and
    root rather than inventing a deeper one. Measured before shipping: the
    one-hop deriver fails the `UB.DEEP` root conjunct only, and a deriver that
    reports roots INSTEAD of hops fails the `UB.DEEP` hop conjunct only, so
    neither half of the pair is carried by the other.
    """
    from .protocol import Result, Spec, Budget

    def stub(sid, deps, budget=Budget.CPU_FAST):
        return Spec(sid, 0, sid, "h", "f", "n", "m", budget, depends_on=deps)

    ladder = [stub("UB.RED", []), stub("UB.GREEN", []),
              stub("UB.CERT", ["UB.RED"], Budget.GPU_SHORT),
              stub("UB.OK", ["UB.GREEN"]),
              stub("UB.NOPASS", ["UB.RED"]),
              stub("UB.ROOT", []),
              stub("UB.MID", ["UB.RED"]),
              stub("UB.DEEP", ["UB.MID"]),
              stub("UB.STALE", [_STALE_ID]), stub(_STALE_ID, [])]
    by_id = {s.id: s for s in ladder}
    # `_STALE_ID` is a REAL spec with a real file, stamped here with a hash it
    # cannot have — that is how `_check_ranker` exercises the freshness half,
    # and it is borrowed rather than re-invented for the same reason this
    # module shares `unsatisfied`.
    fixt = _fixture_ledger()
    for sid, st in (("UB.RED", Status.FAIL), ("UB.GREEN", Status.PASS),
                    ("UB.CERT", Status.PASS), ("UB.OK", Status.PASS),
                    ("UB.NOPASS", Status.FAIL), ("UB.ROOT", Status.PASS),
                    ("UB.MID", Status.FAIL), ("UB.DEEP", Status.PASS),
                    ("UB.STALE", Status.PASS)):
        fixt.results[sid] = Result(
            spec_id=sid, status=st, metrics={}, seeds=[0], commit="1234567",
            ran_at="2026-08-11T00:00:00", impl_sha="0" * 16)

    got = unbacked_certificates(fixt, ladder=ladder, by_id=by_id)
    named = {sid: (deps, cost, roots) for sid, deps, cost, roots in got}
    expect = (
        # POSITIVE — the fallen dependency is named, and so is the bill.
        named.get("UB.CERT")[:2] == (["UB.RED"], "gpu<20min"),
        # POSITIVE — the freshness half. A dependency in PASS is not enough.
        "UB.STALE" in named and named["UB.STALE"][0] == [_STALE_ID],
        # POSITIVE — the chain bottoms out one hop BELOW what is visible, and
        # the hop is still reported. Fails a one-hop deriver on the root and a
        # root-only deriver on the hop; neither conjunct carries the other.
        named.get("UB.DEEP", (None,))[0] == ["UB.MID"],
        named.get("UB.DEEP", (None, None, None))[2]
        == [("UB.RED", "FAIL", "cpu<1min")],
        # POSITIVE — and a one-hop chain does NOT acquire a deeper root: hop
        # and root are the same spec, said so rather than left to be inferred.
        named.get("UB.CERT", (None, None, None))[2]
        == [("UB.RED", "FAIL", "cpu<1min")],
        # NEGATIVE — without these the reader could pass by flagging everything.
        "UB.OK" not in named,
        "UB.NOPASS" not in named,
        "UB.ROOT" not in named,
        # NEGATIVE — `UB.MID` is RED, so it is `unreachable`'s to count, not
        # this reading's, even though it sits on the same fallen dependency.
        "UB.MID" not in named,
        len(named) == 3,
    )
    if not all(expect):
        raise RuntimeError(
            "unbacked-certificate fixture FAILED — the deriver got a known "
            f"graph wrong, so no count it prints is evidence. "
            f"conjuncts={expect} got={got}")


def _check_blast_radius() -> None:
    """Refuse to print a radius the deriver cannot get right on a known graph.

    Same rule as `_check_ranker` and for the same reason (LESSONS.md, "an
    at-chance control must carry proof its instrument was alive"): a count
    from an instrument that flunks its own fixture is not evidence. Runs the
    REAL `blast_radius` over the REAL ranker fixture, not a tidied restatement.

    KNOWN ANSWERS on `_ranker_fixture`'s graph, where `K` is a clean PASS with
    two dependents and `L` depends on `K` alone while `Z2` needs `K` AND the
    already-stuck `X`:

      K -> FAIL   strands L (reachable today, not under the assumption) and
                  must NOT claim Z2, which is stuck behind X either way — the
                  ranker's founding double-count, in the counterfactual.
      K -> FAIL   `runnable_lost == [L]`: L's deps are satisfied today, so it
                  is a dispatch that becomes illegal. Z2's are not.
      X -> PASS   the reverse direction: Y rejoins the reachable set, so
                  `regained` is non-empty and `radius` is empty.
      Y -> PASS   the subject names itself in NEITHER set while the
                  `before`/`after` COUNTS still include it. Y is unreachable
                  today (it is stuck behind X) and nothing depends on Y, so
                  `regained` is EMPTY and the count still falls by one. A
                  deriver that "fixed" the self-naming by dropping the subject
                  from the count would print `before == after` here — a number
                  `run blocked` and `coverage.unreachable_ratchet` do not
                  print. This conjunct was written believing `X` was the case
                  that exercised it; X is a ROOT with no dependencies, so it
                  is never in the stuck set at all, and the fixture rejected
                  the first version of this check. Recorded because the
                  fixture catching its own author is the only evidence that it
                  is doing anything.

    AND THE BACKING HALF (94th audit B1), which is a different question on the
    same graph. `KC` is a standing GPU certificate on `K`; `XC` is one on the
    already-red `X`:

      K -> FAIL   `unbacked == [("KC", ["K"], "gpu<2h", [K FAIL cpu<1min])]` —
                  hop and root are the same spec here, said rather than
                  inferred, and the root's cost class is `K`'s own, not the
                  certificate's; `_check_unbacked_detector` owns the deep
                  case. A certificate that
                  is not in `radius`, not in `runnable_lost` and not in any
                  count this function printed before today, because it has
                  already run. `L` must NOT appear: it is red, so it is priced
                  by `radius`, and counting it twice is the ranker's founding
                  double-count wearing the other half's clothes.
      X -> PASS   `unbacked == [("XC", ...)]` — the SAME key read the other
                  way. On the non-PASS side of a green counterfactual it is not
                  a forecast at all; it is what is already down, and what
                  repairing `X` would re-buy. That is the direction today's
                  real reading takes (`T1.08` is FAIL), so a fixture that
                  exercised only the red direction would leave the live case
                  untested.
    """
    from .protocol import Result, Spec, Budget

    def stub(sid, deps, budget=Budget.CPU_FAST):
        return Spec(sid, 0, sid, "h", "f", "n", "m", budget, depends_on=deps)

    base, base_by = _ranker_fixture()
    extra = [stub("K", []), stub("L", ["K"]), stub("Z2", ["K", "X"]),
             stub("KC", ["K"], Budget.GPU), stub("XC", ["X"], Budget.GPU)]
    ladder = list(base) + extra
    by_id = dict(base_by, **{s.id: s for s in extra})
    fixt = _fixture_ledger()
    # K is a clean PASS: no registry entry, so `module_path_for` finds no
    # implementation file and the freshness half is skipped rather than
    # accidentally exercised. The staleness branch already has its own known
    # answer in `_check_ranker` via `_STALE_ID`.
    for _sid in ("K", "KC", "XC"):
        fixt.results[_sid] = Result(
            spec_id=_sid, status=Status.PASS, metrics={}, seeds=[0],
            commit="1234567", ran_at="2026-08-11T00:00:00", impl_sha="0" * 16)

    fail_k = blast_radius("K", fixt, assume=Status.FAIL,
                          ladder=ladder, by_id=by_id)
    pass_x = blast_radius("X", fixt, assume=Status.PASS,
                          ladder=ladder, by_id=by_id)
    pass_y = blast_radius("Y", fixt, assume=Status.PASS,
                          ladder=ladder, by_id=by_id)
    expect = (
        fail_k["radius"] == ["L"],
        "Z2" not in fail_k["radius"],
        fail_k["runnable_lost"] == ["L"],
        sorted(fail_k["frees"]) == ["L"],
        fail_k["after"] > fail_k["before"],
        fail_k["regained"] == [],
        "Y" in pass_x["regained"],
        pass_x["radius"] == [],
        # The subject names itself in neither set ...
        "X" not in pass_x["regained"] and "X" not in pass_x["radius"],
        "K" not in fail_k["radius"] and "K" not in fail_k["runnable_lost"],
        # ... and is still inside the COUNT it is honestly part of. Y is stuck
        # behind X and nothing depends on Y, so `regained` is empty while the
        # unreachable count still falls by exactly one — Y itself.
        "Y" not in pass_y["regained"] and "Y" not in pass_y["radius"],
        pass_y["regained"] == [] and pass_y["radius"] == [],
        pass_y["before"] - pass_y["after"] == 1,
        # THE BACKING HALF. A standing certificate falls with its dependency,
        # in the direction of harm, with its cost class attached ...
        fail_k["unbacked"] == [("KC", ["K"], "gpu<2h",
                                [("K", "FAIL", "cpu<1min")])],
        # ... and a red dependent is NOT counted here as well as in `radius`.
        "L" not in [u[0] for u in fail_k["unbacked"]],
        # The green direction reads the same key as what is already down, and
        # the root carries the LIVE status of the spec that is actually down
        # — `NOT_RUN` here, not the counterfactual's `PASS`.
        pass_x["unbacked"] == [("XC", ["X"], "gpu<2h",
                                [("X", "NOT_RUN", "cpu<1min")])],
        # And a counterfactual that moves no certificate says so with an empty
        # set rather than by inheriting the previous subject's.
        pass_y["unbacked"] == [],
    )
    if not all(expect):
        raise RuntimeError(
            "blast-radius fixture FAILED — the deriver got a known graph "
            f"wrong, so no radius it prints is evidence. conjuncts={expect} "
            f"fail_k={fail_k} pass_x={pass_x} pass_y={pass_y}")


def cmd_blast_radius(ledger: Ledger, ids=()) -> int:
    """`run blast-radius <SPEC> ...` — price a graph edit BEFORE it lands.

    Prints the block `protocol.BLAST_RADIUS_DECL` requires, in the form it
    requires it (id and title, "none" said rather than implied), so a
    `VOID-FORECLOSED:` declaration can be priced by derivation instead of by
    assertion — and so a strengthening, which no contract prices at all, can
    be priced by the same command.

    Read-only and spends nothing: no seeds, no GPU, no ledger write.
    """
    ids = [i for i in ids]
    if not ids:
        print("Usage: run blast-radius <SPEC> [<SPEC> ...]")
        print("  What leaves the reachable set if that spec settles non-PASS")
        print("  (or rejoins it, if the spec is already red). Derives the")
        print("  `BLAST RADIUS:` block protocol.py requires by hand.")
        return 2
    unknown = [i for i in ids if i not in BY_ID]
    if unknown:
        print("Refusing: unrecognised spec id(s): " + ", ".join(unknown))
        return 2

    _check_ranker(ledger)
    _check_unbacked_detector()
    _check_blast_radius()

    from .coverage import UNREACHABLE_BASELINE

    rc = 0
    for sid in ids:
        r = blast_radius(sid, ledger)
        arrow = f"{r['status'].value} -> {r['assumed'].value}"
        print(f"\n{sid}  {BY_ID[sid].title}")
        print(f"  counterfactual: {arrow}")
        print(f"  unreachable:    {r['before']} -> {r['after']} of "
              f"{r['ladder']}  (baseline {UNREACHABLE_BASELINE})")

        # Label by the DIRECTION of the counterfactual, never by which set
        # happens to be non-empty. A green-direction question whose answer is
        # "nothing else moves" printed `BLAST RADIUS: none` — the right word
        # for the wrong question, which is how a reader concludes a repair is
        # worthless when what it actually frees is the subject itself.
        going_green = r["assumed"] is Status.PASS
        moved = r["regained"] if going_green else r["radius"]
        label = "REGAINED" if going_green else "BLAST RADIUS"
        if moved:
            print(f"  {label}: {len(moved)} spec(s) —")
            for m in moved:
                print(f"      {m:9s} {BY_ID[m].title}")
        else:
            # "none" must be SAID, not implied — protocol.BLAST_RADIUS_DECL.
            print(f"  {label}: none")

        if r["runnable_lost"]:
            print(f"  RUNNABLE TODAY AND NOT AFTERWARDS: "
                  f"{', '.join(r['runnable_lost'])}")
            print("      — these are dispatches that become illegal the "
                  "moment the run lands;")
            print("        `run_spec` refuses an unsatisfied dependency "
                  "(92nd audit B1).")
        # THE OTHER HALF OF THE PRICE. Everything above is about specs that
        # have not run; this is about certificates that have. Printed for both
        # directions from one key, because "would fall" and "is already down"
        # are the same sentence read from opposite sides of the counterfactual
        # — and the red spec's side is the one a repair is priced from.
        if r["unbacked"]:
            verb = ("cannot be re-derived while it is non-PASS"
                    if going_green else "would no longer be re-derivable")
            print(f"  UNBACKED: {len(r['unbacked'])} standing PASS "
                  f"certificate(s) declare depends_on this spec\n"
                  f"      and {verb} —")
            for sid, deps, cost, roots in r["unbacked"]:
                print(f"      {sid:9s} ({cost})  {BY_ID[sid].title}")
            print("      — a legal state, not a violation: the run happened "
                  "and the row is honest.\n        What is true is that "
                  "`run_spec` would refuse to re-buy them, and the cost\n"
                  "        class is printed because a re-buy is not free.")
        else:
            # "none" must be SAID here too, for the same reason it is said
            # above: this half was invisible, and an absent line reads as an
            # absent question rather than as a zero.
            print("  UNBACKED: none")

        if r["frees"]:
            print(f"  would rank in `run blocked`: frees {len(r['frees'])} / "
                  f"blocks {len(r['blocks'])}")
        if r["after"] > UNREACHABLE_BASELINE:
            print(f"  !! this edit would put `unreachable` ABOVE its floor "
                  f"({r['after']} > {UNREACHABLE_BASELINE}). The floor moves "
                  "only in the commit that grows it, with the reason in its "
                  "growth log, signed by whoever committed it.")
    print()
    # DELIBERATELY 0 EVEN WHEN THE WARNING ABOVE FIRES. Every number here is a
    # counterfactual: nothing is wrong with the tree, the ledger or the
    # ratchet at the moment this is asked. An advisory that exits non-zero on
    # a hypothesis is a false alarm, and a tool that cries wolf gets ignored
    # inside a week — the overseer's own objection to the certificate screen
    # (`D27`, 2026-09-13). The loud line is the product; the exit code says
    # only whether the QUESTION was well-formed (2 for usage / unknown id).
    return rc


def _split_foreclosed(ranked, ledger: Ledger, vf=None, vfr=None) -> tuple:
    """Partition ranked roots into (live, closed, refused): the repairable
    list, the `VOID-FORECLOSED` doors `{root: declared reason}`, and the
    MALFORMED declarations `{root: refusal message}` (54th audit B3): a
    foreclosure that does not price its `FORECLOSURE ARITHMETIC:` and
    `BLAST RADIUS:` is refused, and the refused root stays in the repairable
    ranking — an unpriced weld does not close a door — but must be printed
    WITH its refusal, or the fallback silently re-opens the B2 misroute.

    Written for the 54th audit's B2 (2026-08-31): this module had ZERO
    references to `protocol.void_foreclosed` while `coverage.py` read it, so
    the two instruments disagreed about the same three specs — and this is the
    one the builder consults to pick high-leverage work. On the day of the
    audit it ranked `LC.03 = VOID  frees 8` SECOND on the what-one-fix list,
    a door the project had declared un-re-runnable a week earlier. A closed
    door presented as a repair target sends an iteration at spent evidence.

    The gate is `status is VOID and declares` — the same conjunction as
    `coverage.queue_depth`, deliberately, so the two readers cannot drift: a
    declaration on a non-VOID spec is a mention, not a foreclosure, and a VOID
    without a declaration (T2.02's shape) stays in the repairable ranking.
    """
    from .protocol import void_foreclosed, void_foreclosed_refusal
    vf = void_foreclosed if vf is None else vf
    vfr = void_foreclosed_refusal if vfr is None else vfr
    live, closed, refused = [], {}, {}
    for root, ids in ranked:
        is_void = ledger.status(root) is Status.VOID
        why = vf(root) if is_void else None
        if why:
            closed[root] = why
        else:
            live.append((root, ids))
            if is_void:
                refusal = vfr(root)
                if refusal:
                    refused[root] = refusal
    return live, closed, refused


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

    def stub(sid, deps, repaired_by=()):
        return Spec(sid, 0, sid, "h", "f", "n", "m", Budget.CPU_FAST,
                    depends_on=deps, repaired_by=list(repaired_by))

    # F, R and M are VOID roots for the foreclosure split: F declares
    # VOID-FORECLOSED (via the stubbed reader), R is a plain repairable VOID,
    # M declares but its declaration is REFUSED (unpriced — 54th audit B3).
    # X carries the REPAIR EDGE (69th audit B3) — `X.repaired_by = [P]`, where
    # P is a spec nothing depends on, so the unsubstituted ranker scores it at
    # exactly zero. That is T2.01 -> D1.0 in miniature, and it is the whole
    # defect: a POSITIVE control (P must carry X's mass after substitution)
    # paired with a NEGATIVE one (W declares nothing and must gain nothing, so
    # the layer cannot be passing by crediting every root).
    # B and C are the two malformed shapes `_check_repair_edges` must refuse:
    # a self-loop and an id that is not in the registry.
    ladder = [stub("X", [], repaired_by=["P"]), stub("W", []),
              stub("Y", ["X"]), stub("Z", ["X", "W"]),
              stub(_STALE_ID, []), stub("V", [_STALE_ID]),
              stub("F", []), stub("G", ["F"]), stub("R", []), stub("Q", ["R"]),
              stub("M", []), stub("N", ["M"]), stub("P", []),
              stub("B", [], repaired_by=["B"]),
              stub("C", [], repaired_by=["NOT.A.SPEC"])]
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
    for vid in ("F", "R", "M"):
        led.results[vid] = Result(
            spec_id=vid, status=Status.VOID, metrics={}, seeds=[0],
            commit="1234567", ran_at="2026-08-11T00:00:00", impl_sha="0" * 16)
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
    # KNOWN ANSWER for the foreclosure split (54th audit B2, refusals B3). The
    # stubbed reader declares for F (VOID → closed) and for X (declared but
    # NOT VOID → stays live); R is VOID with no declaration (T2.02's shape →
    # stays live); M is VOID with a REFUSED declaration (unpriced) → stays
    # live AND carries its refusal, and non-VOID X must NOT be asked for one.
    # A split that fails any of these would hide a repairable VOID, keep
    # ranking a closed door, or let an unpriced weld go quiet — the three
    # defects this exists to catch.
    ranked = sorted(mentions.items(),
                    key=lambda kv: (-len(frees.get(kv[0], [])), -len(kv[1])))
    live, closed, refused = _split_foreclosed(
        ranked, fixt,
        vf=lambda sid, path=None: {"F": "declared closed",
                                   "X": "declared but not VOID"}.get(sid),
        vfr=lambda sid, path=None: {"M": "missing `BLAST RADIUS:`",
                                    "X": "must never be read"}.get(sid))
    expect += (
        closed == {"F": "declared closed"},
        "F" not in dict(live),
        "X" in dict(live),
        "R" in dict(live),
        "M" in dict(live),
        refused == {"M": "missing `BLAST RADIUS:`"},
    )
    # KNOWN ANSWER for the repair layer (69th audit B3). `X.repaired_by = [P]`,
    # and P is a root nothing depends on — the unsubstituted ranker scores it
    # zero, which IS the defect. Four conjuncts, and the last two are the ones
    # that keep the layer honest rather than merely present.
    r_ment, r_frees, r_groups, r_bad = _repair_carry(
        terminal, fixt, ladder=ladder, by_id=by_id)
    expect += (
        # POSITIVE: the mass moves. P now carries what X carried.
        sorted(r_ment.get("P", [])) == ["Y", "Z"],
        sorted(r_frees.get("P", [])) == ["Y"],
        r_groups.get(frozenset({"P", "W"})) == ["Z"],
        # NEGATIVE: a root with no declaration gains NOTHING. Without this the
        # layer would pass by crediting every root it saw, which is the
        # "measuring nothing" failure one section up, in a new section.
        "W" not in r_ment and "X" not in r_ment,
        # REPORTING-ONLY: the base ranking is untouched by the declaration —
        # X is still the terminal blocker, P is still absent from it. A repair
        # edge that reached `_terminal_blockers` would be `depends_on` under
        # another name, and would drift X's certificate.
        "P" not in mentions and sorted(mentions.get("X", [])) == ["Y", "Z"],
        # MALFORMED declarations are refused, by shape and by name.
        sorted(r_bad) == ["B", "C"],
        r_bad.get("B") == ["B (self-loop)"],
        r_bad.get("C") == ["NOT.A.SPEC (not in the registry)"],
    )
    if not all(expect):
        # `tuple(sorted(k))`, not `set(k)`: a set is unhashable as a dict key,
        # so the previous rendering raised TypeError INSIDE the refusal — the
        # ranking was still refused, but the diagnostic never printed. Found
        # 2026-08-31 by the foreclosure-split teeth check, latent since birth.
        groups_repr = {tuple(sorted(k)): v for k, v in groups.items()}
        raise RuntimeError(
            "the blocked-ranker failed its own fixture "
            f"(mentions={mentions}, frees={frees}, groups={groups_repr}, "
            f"live={sorted(dict(live))}, closed={closed}, refused={refused}, "
            f"repair_mentions={r_ment}, repair_frees={r_frees}, "
            f"repair_groups={ {tuple(sorted(k)): v for k, v in r_groups.items()} }, "
            f"repair_refused={r_bad}); "
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
              "[--status VOID|SKIP|NOT_RUN] [--unknown-history] [--fix-hardware] "
              "[--fix-heads] [--doc-only]")
        return 2
    spec_id = args.spec[1]
    try:
        status = Status(args.status) if args.status else None
        row = ledger.amend(spec_id, by=args.by or "", reason=args.reason or "",
                           status=status, unknown_history=args.unknown_history,
                           fix_hardware=args.fix_hardware,
                           fix_heads=args.fix_heads,
                           doc_only=args.doc_only)
    except (ValueError, KeyError) as e:
        print(f"Refusing to amend {spec_id}: {e}")
        return 1
    note = row["amended"][-1]
    print(f"{spec_id}: amended by {note['by']} at {note['at']} ({note['commit']})")
    for c in note["changes"]:
        print(f"    {c['field']}: {c['from']!r} -> {c['to']!r}")
    print(f"    reason: {note['reason']}")
    _warn_impl_deps_dependents(ledger, spec_id, note)
    return 0


def _warn_impl_deps_dependents(ledger: Ledger, spec_id: str, note: dict) -> None:
    """Name the OTHER certificates this amendment just stranded.

    The gap this closes, paid for once (builder, 2026-09-12, 90th audit B3).
    A prose-only docstring edit to `pg_4_noisy_tv.py` — ordered as the
    "cheapest honest version" of a visibility repair, and correctly re-stamped
    here by `--doc-only` — silently staled `T2.08` and `T2.09`, which both
    declare that file in `IMPL_DEPS`. `T3.06` then fell behind the stale
    `T2.08` and the shrink-only `unreachable` ratchet went 93 -> 94, i.e. a
    floor was breached by an edit priced as free. The bill was worse than it
    looks: `T2.09` is a GPU certificate (3316 s recorded) and `T3.06` is
    VOID-FORECLOSED and may not be re-run at all, so "just re-run them" was
    not available.

    `Ledger.amend`'s dep lane ALREADY handles this correctly — each dependent
    owes its own `--doc-only` amend and gets the same `prose_only_delta`
    proof. Nothing was broken. What was missing is that **nothing told you the
    debt existed**: the amend that creates it prints a clean EXIT 0, and the
    consequence surfaces later as a ratchet number on a page nobody reads
    beside the edit. This is the `aggregate-hides-worst-seed` shape one
    surface over — a correct instrument whose output does not reach the person
    who can act on it.

    Advisory only: it prints, it never refuses, and it lives in the CLI layer
    because `run.py` is in no spec's `IMPL_DEPS` — so the repair for a
    staleness trap does not itself stale four certificates. (`protocol.py`,
    the other candidate home, is declared by `T0.17`, `T0.27`, `T0.33` and
    `T0.35`, and a code edit there is not prose-only, so it would have cost
    four real re-runs to install a warning about incurring re-runs.)
    """
    if not any(c.get("field") == "impl_sha" for c in note.get("changes", ())):
        return                      # nothing re-stamped: nothing stranded
    try:
        from .protocol import impl_deps_of, module_path_for
        path = module_path_for(spec_id)
        if path is None:
            return
        root = Path(__file__).resolve().parent.parent
        try:
            rel = str(Path(path).resolve().relative_to(root))
        except ValueError:
            return
        owed = []
        for sid, _status, kind, _detail in stale_claims(ledger):
            if sid == spec_id or kind != "CHANGED":
                continue
            dep_path = module_path_for(sid)
            if dep_path is None:
                continue
            deps, _problem = impl_deps_of(dep_path)
            if any(str(d).replace("\\", "/") == rel for d in deps):
                owed.append(sid)
        if not owed:
            return
        print(f"\n    ! {len(owed)} OTHER certificate(s) declare {rel} in "
              f"IMPL_DEPS and are stale:")
        print(f"        {', '.join(sorted(owed))}")
        print("      CANDIDATES, not proven consequences — this cannot tell "
              "a spec staled BY this\n      edit from one already stale on "
              "its own code. Try the same prose-only re-stamp;\n      the "
              "lane REFUSES loudly if that spec's own AST moved, which is "
              "the safe outcome:")
        for sid in sorted(owed):
            print(f"        run amend {sid} --doc-only --by ... --reason ...")
        print("      Leave a real one unpaid and it is a silent certificate "
              "decay that `run status`\n      reports only later, and only "
              "to whoever next reads the stale block.")
    except Exception:
        return                      # an advisory that breaks an amend is worse
                                    # than one that stays quiet (T0.12's rule)


def _impl_age_line(root: str) -> str:
    """`unchanged for N days`, from the last commit touching the impl file.

    62nd audit B5 (2026-09-02): `blocked` ranks by `frees N`, and nothing
    ranked by how long a blocker had been the top blocker — T2.01 sat rank 1
    for 24 days with its implementation untouched since 2026-08-09, and the
    ranking looked equally fresh every day. Age printed beside the rank makes
    a stall a number instead of an archaeology job. Empty string when git or
    the path cannot answer: a missing age must never break the ranking that
    carries it.
    """
    path = module_path_for(root)
    if not path:
        return ""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", str(path)],
            capture_output=True, text=True, timeout=10, cwd=str(_REPO))
        ts = int(out.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return ""
    days = max(0, int((time.time() - ts) // 86400))
    return f"  [impl unchanged {days} d]"


def _blocked_rows(ranked, dead_flavour, closed, held_map=None) -> list:
    """Section each ranked terminal blocker, and carry its decision-hold.

    THE SCAR IS FOUR HOURS OLD AND IT IS MINE (builder, 2026-09-13 22:xx).
    At 21:xx this desk taught `cmd_next` to read BOTH liveness readers, wrote
    the lesson *"the command that ADVERTISES work must read the same holds as
    the command that AUDITS it"*, and pinned the third conjunct of
    `_check_next_triage` on exactly this shape: *"a version that reads
    `_liveness_state` but not `holds()` fails on exactly the `HR.1`/`D19`
    row."* `cmd_blocked` **is** that version, one command over. It has read
    `coverage.root_dead` since the 59th audit — PARKED, PILOT-BLOCKED,
    VOID-FORECLOSED — and has never asked `decisions.holds()`.

    Measured on the live ledger at 22:1x, before this function existed:

        HR.1 = NOT_RUN  frees 3  (blocks 3)  — The voice corpus is honest ...

    ranked fifth in the project, in the live repairable section, with nothing
    beside it — while `coverage` printed, of the same spec, *"cpu<10min
    HR.1 <- D19 (decide_by 2026-09-14) … the run IS the fetch the default
    forbids."* Two instruments, one spec, opposite answers, and the one that
    advertises is the one that is wrong. It is also the most attractive row on
    the board by construction: `NOT_RUN` + `frees 3` is what a fresh unblock
    looks like. The orientation sentence beside this command reads *"run
    `run blocked` for a genuine candidate."*

    A HOLD ANNOTATES; IT NEVER DEMOTES — and that is the difference from
    `_next_triage`, stated because the two repairs deliberately diverge:

      * `next` answers *what should I do now*, and it TRUNCATES at 12, so
        annotating without re-laning left the corpses above the cut. There the
        reorder was the repair.
      * `blocked` answers *what one fix would free the most* — a property of
        the graph, which a dated decision does not change. It prints every
        root, so nothing is hidden by order. Re-ranking here would corrupt the
        quantity the command exists to report, and demoting a held root into
        the PARKED / VOID-FORECLOSED section would say *"redesign would
        recover"* about a door that opens on its own date. `HR.1`'s opens
        tomorrow. So the lane stays LIVE and the hold rides beside it.

    `held_map` is injectable for the same reason `_next_triage`'s is: the
    known answer must not need markers on disk (`_RANKER_FIXTURE`, T0.36).
    """
    if held_map is None:
        from .decisions import holds
        held_map = holds()
    rows = []
    for root, ids in ranked:
        dead = root in closed or root in dead_flavour
        rows.append({"root": root, "ids": ids,
                     "lane": "DEAD" if dead else "LIVE",
                     "hold": held_map.get(root)})
    return rows


def _check_blocked_holds() -> None:
    """Refuse to advertise an unblock list that flunks a known hold graph.

    Same rule as `_check_next_triage`, and the two wrong versions it must
    reject are measured, not asserted — each fails exactly one conjunct, so
    neither conjunct is the other wearing extra words:

      A  FAIL, no hold          -> LIVE, hold None.
      P  PARKED                 -> DEAD.
      H  NOT_RUN, decision-held -> LIVE **and** carrying its hold. This one
                                   row kills both wrong versions and needs
                                   both conjuncts to do it:
                                     - the liveness-only reader (this file
                                       until tonight) returns lane LIVE and
                                       hold None -> fails the hold conjunct,
                                       passes the lane one;
                                     - a reader that treats a decision-hold as
                                       a closed door returns DEAD -> fails the
                                       lane conjunct, passes the hold one.
    """
    ranked = [("A", ["a1"]), ("P", ["p1"]), ("H", ["h1"])]
    rows = {r["root"]: r for r in _blocked_rows(
        ranked, dead_flavour={"P": "PARKED"}, closed={},
        held_map={"H": "D19 (decide_by 2026-09-14)"})}
    got = (rows["A"]["lane"], rows["A"]["hold"],
           rows["P"]["lane"],
           rows["H"]["lane"], rows["H"]["hold"])
    want = ("LIVE", None, "DEAD", "LIVE", "D19 (decide_by 2026-09-14)")
    if got != want:
        raise AssertionError(
            f"_blocked_rows flunked its own fixture: {got} != {want}")


def cmd_blocked(ledger: Ledger) -> int:
    """What can this ladder NEVER do, and why — the converse of `next`.

    Written 2026-08-09 because the overseer had to walk the dependency graph by
    hand to discover that 29% of the ladder was dead behind two VOIDs, and that
    the dead set was precisely GOAL.md's headline: all 7 curiosity specs, all 16
    unison specs, all of Tiers 3, 4 and 5. `next` answers "what can I do"; until
    now nothing answered "what is unreachable, and what one fix would free it".
    LESSONS.md carried that as advice to humans. This makes it a command.

    Foreclosed roots rank in their own section (54th audit B2, 2026-08-31):
    this command printed `LC.03 = VOID  frees 8` second on the what-one-fix
    list for a week after the project declared it un-re-runnable, because only
    `coverage.py` read `protocol.void_foreclosed`. The two readers now share
    the same gate via `_split_foreclosed`.

    Decision-HELD roots are annotated in place and keep their rank — see
    `_blocked_rows` for why that differs from `_next_triage`'s re-laning.
    """
    _check_ranker(ledger)
    _check_blocked_holds()
    terminal = _terminal_blockers(ledger)
    mentions, frees, groups = _rank_blockers(terminal, ledger)

    if not mentions:
        print("Nothing is blocked — every unrun spec has its dependencies passing.")
        return 0

    ranked = sorted(mentions.items(),
                    key=lambda kv: (-len(frees.get(kv[0], [])), -len(kv[1])))
    live, closed, refused = _split_foreclosed(ranked, ledger)

    # 59th audit B1 mirror: `_split_foreclosed` pulls VOID-FORECLOSED roots
    # out of the repairable ranking, but a PARKED or PILOT-BLOCKED root is
    # the same closed door in a different flavour, and T2.11 ranked live
    # ("frees 1: ME.6") for days after its park. One shared predicate
    # (`coverage.root_dead`) for both readers, so they cannot drift.
    from .coverage import parked as _parked_fn
    from .coverage import root_dead
    _parked_map, _ = _parked_fn()
    _dead_flavour: dict = {}
    _still_live = []
    for root, ids in live:
        why = root_dead(root, status=getattr(ledger.status(root), "name",
                                             None), parked_map=_parked_map)
        if why:
            _dead_flavour[root] = why
        else:
            _still_live.append((root, ids))
    live = _still_live
    dead_roots = set(closed) | set(_dead_flavour)

    # The OTHER liveness reader, and the one this command never asked until
    # tonight: an OPEN decision that declares a root in `blocks:`. It rides
    # beside the root and moves nothing — `_blocked_rows` carries the reason.
    _rows = _blocked_rows(ranked, _dead_flavour, closed)
    live = [(r["root"], r["ids"]) for r in _rows if r["lane"] == "LIVE"]
    _hold_of = {r["root"]: r["hold"] for r in _rows if r["hold"]}

    # A REPAIR path is asked the same question, and it has to be asked
    # separately: a repair spec is not a terminal blocker, so it never enters
    # `live` and the loop above cannot reach it. A `repaired_by` pointing at a
    # PARKED or VOID-FORECLOSED spec is the 59th audit's corpse citation with
    # the arrow reversed — the instrument would print "carries 35" beside a
    # door welded shut, which is worse than printing nothing.
    #
    # Its own dict, deliberately, and NOT folded into `dead_roots`: `welded` is
    # computed from the unsubstituted walk, and a repair edge must not be able
    # to move a spec into or out of the welded set. Reporting-only means
    # reporting-only in both directions.
    r_ment, r_frees, r_groups, r_bad = _repair_carry(terminal, ledger)
    _repair_flavour: dict = {}
    for root in r_ment:
        why = root_dead(root, status=getattr(ledger.status(root), "name", None),
                        parked_map=_parked_map)
        if why:
            _repair_flavour[root] = why
    welded = sorted({s for ids in mentions.values() for s in ids
                     if (terminal.get(s, set()) - {s})
                     and (terminal.get(s, set()) - {s}) <= dead_roots})

    def _st(root):
        """The status, and — if it is a PASS that no longer describes the code,
        or a VOID whose declaration says re-running is foreclosed — SAY SO. A
        root printed bare as `PASS` is unreadable, and a foreclosed root
        printed bare as `VOID` reads as a repair target (54th audit B2)."""
        if root in closed:
            return "VOID-FORECLOSED"
        if root in _dead_flavour:
            return _dead_flavour[root]
        if root in _repair_flavour:
            return _repair_flavour[root]
        if root in _hold_of:
            # Not a flavour of dead — a dated hold, printed with the status it
            # actually has so the reader sees both facts at once.
            base = (ledger.status(root).value if root in BY_ID
                    else "UNKNOWN-SPEC")
            return f"{base}, HELD by {_hold_of[root]}"
        if root not in BY_ID:
            return "UNKNOWN-SPEC"
        st = ledger.status(root)
        if st is Status.PASS:
            path = module_path_for(root)
            entry = ledger.results.get(root)
            if path and entry and any(k in ("DIRTY", "CHANGED",
                                            "UNSTAMPED_CHANGED")
                                      for k, _ in staleness_of(entry, path)):
                return "PASS but STALE — re-run it"
        return st.value

    total, _ = unreachable_count(ledger, mentions=mentions)
    print(f"\n{total} of {len(LADDER)} specs are unreachable. Terminal blockers, "
          f"ranked by what fixing ONE of them alone would free:\n")
    for root, ids in live:
        title = BY_ID[root].title if root in BY_ID else "(not in the registry)"
        f = sorted(frees.get(root, []))
        print(f"  {root} = {_st(root)}  frees {len(f)}  (blocks {len(ids)})"
              f"{_impl_age_line(root)}  — {title}")
        if root in _hold_of:
            print(f"        !! HELD by {_hold_of[root]} — an OPEN decision "
                  f"declares this spec in `blocks:`")
            print(f"        !! its frees-count is real and its rank is honest;"
                  f" DISPATCHING IT IS NOT. The hold")
            print(f"        !! lifts when the owner rules or the armed default "
                  f"fires on its own date.")
        if root in refused:
            # An unpriced foreclosure ranks as repairable, but silently ranking
            # it re-opens the B2 misroute in the other direction: somebody
            # tried to weld this door and the weld was refused. Say so.
            print(f"        !! VOID-FORECLOSED {refused[root]}")
            print(f"        !! repair the DECLARATION before dispatching a re-run")
        print(f"        frees:  {', '.join(f) if f else 'NOTHING on its own'}")
        rest = sorted(set(ids) - set(f))
        if rest:
            print(f"        also blocks (needs a co-requisite too): {', '.join(rest)}")
        print()

    if _hold_of:
        print(f"  DECISION-HELD — {len(_hold_of)} of the {len(live)} live "
              f"ranked blocker(s) above sit behind an OPEN\n  decision. They keep "
              f"their rank because the mass they block is real, and they are "
              f"NOT\n  a closed door: no redesign is owed and the hold lifts "
              f"on its own date. Until then a\n  dispatch here walks around "
              f"the decision:\n")
        for root, why in sorted(_hold_of.items()):
            print(f"    {root} <- {why}")
        print()

    if closed:
        print("  VOID-FORECLOSED — these do not free anything by being re-run; "
              "the declaration says PASS\n  is unreachable at any envelope. The "
              "repair is a re-parenting or a redesign, routed\n  through the "
              "Review — not a dispatch:\n")
        for root, ids in ranked:
            if root not in closed:
                continue
            title = BY_ID[root].title if root in BY_ID else "(not in the registry)"
            f = sorted(frees.get(root, []))
            print(f"    {root} = VOID-FORECLOSED  re-parenting would recover "
                  f"{len(f)}  (blocks {len(ids)}){_impl_age_line(root)}  — {title}")
            print(f"        declared: {closed[root]}")
            print(f"        unreachable until re-parented: "
                  f"{', '.join(f) if f else '(co-requisites only)'}")
            print()

    if _dead_flavour:
        print("  PARKED / PILOT-BLOCKED roots — the same closed door in a "
              "different flavour\n  (59th audit B1): re-running or waiting "
              "frees NOTHING; the repair is the redesign\n  each one's own "
              "record routes to the Review:\n")
        for root, ids in ranked:
            if root not in _dead_flavour:
                continue
            title = BY_ID[root].title if root in BY_ID else "(not in the registry)"
            f = sorted(frees.get(root, []))
            print(f"    {root} = {_dead_flavour[root]}  redesign would recover "
                  f"{len(f)}  (blocks {len(ids)}){_impl_age_line(root)}  — {title}")
            print(f"        unreachable until redesigned: "
                  f"{', '.join(f) if f else '(co-requisites only)'}")
            print()

    if welded:
        print(f"  WELDED — every terminal blocker is a closed door "
              f"(VOID-FORECLOSED, PARKED or\n  PILOT-BLOCKED). No dispatch "
              f"anywhere can free these {len(welded)}; nothing on this\n"
              f"  board unblocks them and no ranking above should be read as "
              f"saying otherwise:\n")
        print(f"    {', '.join(welded)}\n")

    if r_ment or r_bad:
        print("  REPAIR PATHS — REPORTING ONLY, not dependencies (69th audit "
              "B3). A blocker that is\n  a settled FAIL is not repaired by "
              "re-running it; `repaired_by` names the spec whose\n  RESULT "
              "would say how. The ranking above scores these at zero because "
              "nothing\n  declares `depends_on` on them, and that is why the "
              "60th audit had to route T2.01's\n  repair by hand. Read the "
              "carried mass as leverage, and the STATUS beside it as\n"
              "  whether the door is even open:\n")
        for root, ids in sorted(r_ment.items(),
                                key=lambda kv: (-len(r_frees.get(kv[0], [])),
                                                -len(kv[1]))):
            title = BY_ID[root].title if root in BY_ID else "(not in the registry)"
            f = sorted(r_frees.get(root, []))
            via = sorted(s for s in BY_ID
                         if root in (getattr(BY_ID[s], "repaired_by", []) or []))
            print(f"    {root} = {_st(root)}  carries frees {len(f)}  "
                  f"(blocks {len(ids)}){_impl_age_line(root)}  — {title}")
            print(f"        declared the repair path for: {', '.join(via)}")
            print(f"        would carry: "
                  f"{', '.join(f) if f else 'NOTHING on its own'}")
        for sid, why in sorted(r_bad.items()):
            print(f"    !! {sid}.repaired_by REFUSED: {', '.join(why)}")
            print(f"    !! an id that resolves to nothing is a pointer at work "
                  f"nobody can go do")
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
    # The table and the seeds-x2 arithmetic live in rtf.py so that the process
    # that KILLS an overlong run (here) and the gate that REFUSES one before it
    # starts (rtf.gate_long_run, T0.32) compute from one source and cannot
    # drift apart. The seeds multiplier: the budget names one EXPERIMENT; a
    # spec runs seeds x (experiment + control) — the 3-seed re-verification
    # killed T1.01/02/06 mid-science at the single-seed timeout.
    from .rtf import spec_child_timeout_seconds
    _spec = _BY_ID.get(spec_id)
    _timeout = spec_child_timeout_seconds(_spec)
    # T0.33: a CPU child is gated by the day's tenant budget BEFORE it spawns
    # and debits its measured wall clock after. A refusal is tenant
    # protection, not a measurement of the spec, so it returns UNRECORDED —
    # an ERROR row here would supersede a real result with scheduling noise.
    from .cpu_budget import charge_cpu_child, gate_cpu_child
    _is_cpu = _spec is not None and _spec.budget.value.startswith("cpu")
    if _is_cpu:
        _cpu_gate = gate_cpu_child(_spec)
        if not _cpu_gate.admitted:
            # 68th audit B4: the refusal is UNRECORDED by design, so this
            # line is its only trace — it lands in ladder.log via the loop's
            # capture. A day that refuses 53 specs must say so somewhere.
            print(f"cpu-refused {spec_id} est={_cpu_gate.est_s:.0f} "
                  f"remaining={_cpu_gate.remaining_s:.0f} "
                  f"load={_cpu_gate.load:.2f}", file=sys.stderr, flush=True)
            return Result(spec_id=spec_id, status=Status.ERROR,
                          message=f"REFUSED before start: {_cpu_gate.reason}")
    # 92nd audit B1: the SAME treatment for the expensive resource, at the SAME
    # site. `dispatch_guard` was built on 09-13 and wired only into
    # `scripts/dispatch.sh`, so `$PY -m experiments.run <GPU-SPEC>` — the
    # command that actually spends the weekly quota, and the one `D1.0`'s pilot
    # went out through on 09-01 — still had no budget, authorisation or
    # projection check. "The cheap resource is guarded by a branch; the
    # irreversible one by a convention" (OVERSIGHT, 92nd audit). This is the
    # branch. Same UNRECORDED-refusal idiom as the CPU gate above and for the
    # same reason: a scheduling refusal must never supersede a real verdict.
    # The projection travels in `JACK_PROJECTED_HOURS`; a reattach is exempt.
    # See `dispatch_guard.runner_preflight` for the three rulings.
    _is_gpu = _spec is not None and _spec.budget.value.startswith("gpu")
    if _is_gpu:
        try:
            from .dispatch_guard import runner_preflight
            _ok, _why, _lines = runner_preflight(spec_id)
        except Exception as e:  # pragma: no cover - defensive
            # A guard that crashes must refuse, not wave the spend through:
            # the failure mode it exists to prevent is irreversible and the
            # failure mode of refusing wrongly is one lost slot.
            _ok, _why, _lines = False, f"pre-flight raised {e!r}", [
                f"REFUSING: {spec_id} — the GPU pre-flight itself failed "
                f"({e!r}). A guard that cannot run does not clear a dispatch."]
        for _line in _lines:
            print(_line, file=sys.stderr, flush=True)
        if not _ok:
            print(f"gpu-refused {spec_id} {_why}", file=sys.stderr, flush=True)
            return Result(spec_id=spec_id, status=Status.ERROR,
                          message=f"REFUSED before start: {_why}")

    def _bill_cpu(t_start: float) -> None:
        # A charge failure must not destroy the result the child already
        # recorded, but it may not pass silently either.
        try:
            charge_cpu_child(spec_id, time.monotonic() - t_start)
        except Exception as e:  # pragma: no cover - defensive
            print(f"!! CPU BUDGET CHARGE FAILED for {spec_id}: {e} — wall "
                  f"clock spent but unbilled", file=sys.stderr, flush=True)
    # The ran_at of any PRE-EXISTING entry, so a crashed child cannot pass the
    # old result off as its own. T2.01 v3's child died (SIGPIPE from a killed
    # session pipe) after v2 had recorded a FAIL: the old check — "is there an
    # entry at all?" — found v2's entry and reported it as the rerun's outcome.
    # A rerun that changes nothing must be an ERROR, not an echo.
    _prev = ledger.results.get(spec_id)
    _prev_ran_at = getattr(_prev, "ran_at", None)
    _t0 = time.monotonic()
    try:
        proc = sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                      cwd=str(Path(__file__).parent.parent), timeout=_timeout)
    except sp.TimeoutExpired:
        # An uncaught timeout used to crash the whole runner invocation and
        # leave the spec's STALE entry standing. A timeout is a result.
        # The killed child still occupied the box for the full window; a
        # timeout that goes unbilled would make waste invisible (T0.12's
        # failed-hours rule, one resource over).
        if _is_cpu:
            _bill_cpu(_t0)
        res = Result(spec_id=spec_id, status=Status.ERROR,
                     message=f"timed out after {_timeout}s "
                             f"(budget {_spec.budget.value if _spec else '?'} "
                             f"x {getattr(_spec, 'seeds', 1)} seeds x2)")
        ledger.record(res)
        return res
    if _is_cpu:
        _bill_cpu(_t0)
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


def _warn_if_dirty_before_running(spec_ids: list[str]) -> bool:
    """Say — BEFORE the run — that a FAIL from this tree can never be audited.

    `env_stamp()` already writes `+dirty`, `staleness_of` already reports it,
    and `audit_supersedes_fail` (T0.27) already refuses a PASS that supersedes
    a `+dirty` FAIL, because the failing code exists in no commit and the
    `git diff` that shows what moved is impossible. Three organs knew. All
    three speak AFTERWARDS, and by then the row is permanent: history keeps
    the pair, no re-run removes it, and the only honest remedies are a red
    ladder or an owner ruling.

    Cost of learning that: 2026-08-29, this function's own commit. The builder
    edited `protocol.py`, ran `T0.17` to see whether the new property held,
    got a genuine FAIL from an uncommitted tree, fixed the CODE (no threshold
    moved), committed, re-ran to PASS — and left `T0.27` permanently red on a
    pair that is unauditable by construction. The documented loop ("Run it.
    Read the output. FAIL -> fix the CODE, re-run") produces exactly this
    shape, so the warning belongs where the loop is, not in a lesson file.

    A WARNING AND NOT A REFUSAL, deliberately. Running a test you have just
    edited is how the loop works and blocking it would push the builder to
    commit code it has never executed — a worse failure with no instrument at
    all. What the loop owes is knowing the price before it pays: commit first,
    and a FAIL becomes an artifact instead of an anecdote.
    """
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=_REPO, timeout=10).stdout.splitlines()
    except Exception:
        return False
    dirty = [porcelain_path(ln) for ln in porcelain if is_code_dirt(ln)]
    if not dirty:
        return False
    shown = ", ".join(sorted(dirty)[:4]) + ("  …" if len(dirty) > 4 else "")
    print(f"  ! DIRTY TREE — {len(dirty)} uncommitted code file(s): {shown}")
    print(f"    {', '.join(spec_ids)} will stamp `+dirty`: the code that runs "
          f"exists in no commit.\n    If this FAILs and a later run PASSes, "
          f"T0.27 flags that pair FOREVER — history keeps it\n    and no re-run "
          f"clears it. Commit first if you can.\n")
    return True


def cmd_run(ledger: Ledger, spec_ids: list[str]) -> int:
    failures = 0
    _warn_if_dirty_before_running(spec_ids)
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
        # duration_s is the recording call, not the work: name the metered
        # remote cost when there is one (17th-audit B3 — LC.03 read 0.02 s
        # for ~45 GPU-hours).
        cost = (f"{res.duration_s}s"
                if getattr(res, "compute_s", None) is None
                else f"gpu {res.compute_s}s metered, recorded in {res.duration_s}s")
        print(f"{res.status.value} ({cost}) {res.message}")
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
    # THE PRE-COMMIT BILL (2026-09-22). `render` is the last command the
    # loop runs before `git add`/`git commit`, so it is the one moment at
    # which knowing what this edit stales can still change what happens.
    # The scar: `eb38ae4` staled `T0.36`'s standing PASS by adding a
    # reading to this very file, and the same slot's journal truthfully
    # reported that no PASS had staled — the author's belief about the
    # author's own blast radius was the only instrument in the loop.
    # Reporting-only and unfloored: editing an instrument is legitimate
    # work here and a gate would refuse the Review's own act.
    from . import stale_cost
    stale_cost._check()
    print(stale_cost.render(), end="")
    return 0


#: The env var that waives the lane guard, LOUDLY. The only sanctioned setter
#: is `scripts/dispatch.sh` (its setsid watcher's whole job is surviving the
#: session; the GPU kernel computes remotely either way). Setting it by hand to
#: background a LOCAL registered run is the exact move the guard exists to
#: refuse — the waiver prints a banner precisely so that doing so leaves a
#: mark the next reader cannot miss.
LANE_WAIVER_ENV = "JACK_LANE_WAIVER"


#: The env var by which a detached launcher DECLARES its lane; the only
#: sanctioned setter is `scripts/launch_detached.sh`. It exists because the
#: guard's topology tests are one interposed process away from blind: the
#: launcher setsids `cpu_budget wrap`, which Popen()s the spend, so the
#: spending process is never the session leader and never orphaned — measured
#: live (105th audit RANK 1) when LT.02's registered row was bought through
#: that lane 6 h 46 m after the guard shipped, and the guard said
#: `launchable`. The launcher knows what it is; the guard must not have to
#: infer it from a topology it can be one Popen away from losing. This marker
#: is a LOUD MARK, never a refusal: whether D20's closure covers the wrapped
#: lane is D32's question (the owner's), and until it rules the refuse/permit
#: line stays exactly where the audit found it.
DETACHED_LANE_ENV = "JACK_DETACHED_LANE"


#: The one soft signal's exact words, named once — the guard prints it, the
#: fixture asserts it, and cmd_lane reuses it. See _lane_verdict for why it
#: is a WARNING and not a refusal.
LANE_SOFT_WARNING = (
    "LANE WARNING: stdin is /dev/null. On this harness that is EITHER a "
    "backgrounded\nlaunch (run_in_background, `&`, nohup, cron) — which DIES "
    "with its session —\nOR an ordinary sandboxed foreground call; the two "
    "are indistinguishable from\nhere (measured 2026-09-19, both lanes "
    "byte-identical in tree/sid/fds/env).\nIf you backgrounded this on a "
    "wake-up promise, kill it and run it in the\nforeground of a session "
    "that stays open until the row is on the ledger.")


def _lane_verdict(settle: bool = True) -> tuple:
    """(refusals, warnings) for this launch; ([], []) = clean foreground.

    The dies-with-parent class, seven occurrences (LESSONS.md): a registered
    run launched as a session-child background task dies the second its
    parent returns — occurrences 4-7 all on 2026-09-19, three of them AFTER
    the lesson naming the lane was written. A lesson is a memory, not a
    control; this function is the control (103rd audit, item 2).

    What is MEASURED on this harness (2026-09-19, both lanes probed twice,
    then re-probed after the first version of this guard false-positived on
    its own re-buy):

    - REFUSABLE, stable: `( cmd & )` orphans to ppid=1 within ms of the
      wrapping shell exiting (a settle recheck below closes the fork race);
      `setsid` makes the child its own session leader — the lane D20 closed
      for registered runs. Both are abandonment BY CONSTRUCTION.
    - NOT refusable, and the first version of this guard got it wrong:
      stdin=/dev/null. A foreground Bash call sometimes holds stdin on a
      live socket and sometimes on /dev/null (sandboxed calls), while a
      `run_in_background` task is byte-identical to the sandboxed foreground
      in everything observable at launch: same tree shape, sid = own shell,
      stdout a harness tasks-file in BOTH lanes, identical env. A refusal
      here blocks legitimate re-buys (it blocked this guard's own T0.36
      re-buy, same day), so the signal is a LOUD MARK instead — and the
      residue (a run_in_background launch the runner cannot see) stays
      covered by notice_exited_dispatches on the live path and by the
      foreground conduct rule.

    An interactive terminal (stdin=/dev/pts/N) and a pipe pass clean — the
    guard refuses ABANDONMENT, not any particular launcher.

    - VISIBLE BY DECLARATION, not refusable here (105th audit item 1): the
      wrapped detached lane. `launch_detached.sh` setsids `cpu_budget wrap`,
      which Popen()s the spend, so NEITHER topology refusal can fire on the
      spending process — occurrences 8 and 9 (2026-09-19 19:11 and 20:13)
      happened while the fixture read 17 green cases, and LT.02's row was
      bought through this lane at 21:11 with the guard saying `launchable`.
      The launcher now declares itself via DETACHED_LANE_ENV and the verdict
      carries a LOUD notice naming it. Refuse/permit is deliberately
      unchanged: that line is D32's (the owner's).
    """
    refusals, warnings = [], []
    if os.getppid() == 1:
        refusals.append("orphaned at launch (ppid=1) — the launching session "
                        "is already gone")
    elif settle:
        # `cmd &`: the launching shell may still be mid-exit at our first
        # read. One short settle turns "usually caught" into "caught".
        time.sleep(0.5)
        if os.getppid() == 1:
            refusals.append("orphaned within 0.5s of launch — the launcher "
                            "exited without waiting (`&`-style background)")
    try:
        if os.getsid(0) == os.getpid():
            refusals.append("session leader (setsid) — detached at birth; "
                            "D20 closed this lane for registered runs")
    except OSError:
        pass
    decl = os.environ.get(DETACHED_LANE_ENV, "").strip()
    if decl:
        warnings.append(
            f"LANE NOTICE: DETACHED LANE, DECLARED — {decl}.\n"
            "setsid sits one wrapper (cpu_budget wrap) above this process, "
            "where the\nsession-leader test cannot see it, so this launch "
            "survives its slot BY\nCONSTRUCTION. Whether D20's closure covers "
            "this lane is D32's question (the\nowner's); until it rules the "
            "lane stays PERMITTED, and this notice is the\nrecord — in the "
            "launch log and on the spend path — that it was used.")
    try:
        st0 = os.stat(0)
        if (stat_mod.S_ISCHR(st0.st_mode)
                and st0.st_rdev == os.stat(os.devnull).st_rdev):
            warnings.append(LANE_SOFT_WARNING)
    except OSError:
        warnings.append("LANE WARNING: stdin is CLOSED — if no live caller "
                        "holds this run, it dies with its session.")
    return refusals, warnings


def cmd_lane(ledger) -> int:
    """Read-only diagnosis of the launch lane; rc 0 = the spend path would
    allow this launch (possibly with a loud mark), 3 = it would refuse.
    Exists so the guard's verdict can be exercised by a fixture (and by a
    curious human) without touching a spec."""
    try:
        stdin_desc = os.readlink("/proc/self/fd/0")
    except OSError:
        stdin_desc = "<unreadable>"
    print(f"stdin={stdin_desc} ppid={os.getppid()} "
          f"sid={os.getsid(0)} pid={os.getpid()}")
    refusals, warnings = _lane_verdict()
    for w in warnings:
        print(w)
    if not refusals:
        print("lane: launchable — the spend path would not refuse this "
              "launch." + (" (with the warning above)" if warnings else ""))
        return 0
    print("lane: ABANDONED launch — the spend path refuses this lane:")
    for r in refusals:
        print(f"    {r}")
    return 3


def cmd_stale_cost(ledger: Ledger, paths=None) -> int:
    """`run stale-cost [<path>...]` — what would THIS edit cost the scoreboard?

    `run stale` asks which certificates are stale NOW; this asks the same
    question one commit EARLIER, which is the only moment at which the answer
    can change what you do. With no arguments it prices the working tree per
    git. See `experiments/stale_cost.py` for the 2026-09-22 scar it was built
    from and for what it deliberately does not attempt.

    Read-only in the strong sense `blast-radius` is: no ledger write, no seeds,
    no GPU — it parses declarations and hashes nothing.
    """
    from . import stale_cost
    stale_cost._check()
    paths = list(paths or []) or stale_cost.changed_paths()
    print(stale_cost.render(stale_cost.price(paths, ledger)), end="")
    return 0


#: The read-only sub-commands, named ONCE. They used to be a tuple in the
#: dispatch test and a dict in the dispatch itself; a word present in one and
#: absent from the other is how a command silently becomes "not a command".
READ_ONLY_COMMANDS = {"status": cmd_status, "next": cmd_next,
                      "blocked": cmd_blocked, "render": cmd_render,
                      "stale": cmd_stale, "verify": cmd_verify,
                      "senses": cmd_senses, "coverage": cmd_coverage,
                      "review-queue": cmd_review_queue,
                      "steering": cmd_steering,
                      "fieldwatch": cmd_fieldwatch,
                      "unread": cmd_unread,
                      "ratchets": cmd_ratchets,
                      "lane": cmd_lane,
                      # Takes spec ids, so `main` routes it one branch earlier;
                      # it is registered HERE anyway because this dict is the
                      # single place a command's name exists (see above) and
                      # it is what prints the `Commands:` line on a typo. Called
                      # with no ids it prints usage and returns 2 — never a
                      # silent zero.
                      "blast-radius": cmd_blast_radius,
                      # Same shape as `blast-radius`: it takes PATHS, so
                      # `main` routes it one branch earlier too. Listed
                      # here because this dict is where a command's name
                      # exists and what prints on a typo.
                      "stale-cost": cmd_stale_cost}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="*",
                    help="spec ids, or status / next / blocked / stale / verify / render")
    ap.add_argument("--tier", type=int)
    ap.add_argument("--gate", action="store_true", help="re-run all passing tests")
    ap.add_argument("--max-budget", choices=[b.value for b in Budget],
                    help="gate only: re-run only PASSes at or below this cost "
                         "class (Budget declaration order). The sweep PRINTS "
                         "every spec it excludes — a bounded gate that names "
                         "its residue, so GPU quota is never spent on a stamp "
                         "refresh and 'gate green' cannot be read as a full "
                         "sweep. Scar: the full gate covers 16 GPU-cost "
                         "PASSes, so it was priced out of ever running "
                         "(46th audit, Finding 2).")
    ap.add_argument("--dirty-ok", action="store_true",
                    help="gate a MODIFIED working tree on purpose — every "
                         "spec re-run stamps `+dirty` and its clean stamp is "
                         "lost (protocol.gate_precondition, 2026-08-30)")
    ap.add_argument("--by", help="amend: the spec or finding that motivates the change")
    ap.add_argument("--reason", help="amend: why, in a sentence")
    ap.add_argument("--status", help="amend: new status (VOID, SKIP or NOT_RUN only)")
    ap.add_argument("--unknown-history", action="store_true",
                    help="amend: this entry's attempt count is not reconstructible")
    ap.add_argument("--fix-hardware", action="store_true",
                    help="amend: reconcile `hardware` with the row's own "
                         "metrics['gpu'] (17th-audit B2 provenance amendment)")
    ap.add_argument("--fix-heads", action="store_true",
                    help="amend: reconcile multi-kernel metrics[...].head "
                         "stamps with the dispatch-time heads in the attempt "
                         "receipts (80th-audit B1 provenance amendment — "
                         "derived from gpu_submissions.jsonl, never supplied)")
    ap.add_argument("--doc-only", action="store_true",
                    help="amend: re-stamp impl_sha after a PROVABLY prose-only "
                         "edit — refuses unless the recorded sha reconstructs "
                         "from git and the docstring-stripped ASTs are "
                         "identical (25th-audit B3)")
    ap.add_argument("--check", action="store_true",
                    help="decisions/champions only: run the owning tool's "
                         "ratchet check (exit code is the tool's). Refused "
                         "beside anything else — a flag the dispatch would "
                         "silently drop is the argv-is-a-spend trap in "
                         "miniature (T0.23 P6/P7)")
    args = ap.parse_args()
    ledger = Ledger()

    # THE ALARM, not just the detector (LESSONS, 2026-08-30). `_cpu_fraction`
    # decides whether a lock holder is idle enough to have its overflow slot
    # taken, and it silently measured the wrong process for the whole life of
    # the function. Its battery is wired HERE — the one path every invocation
    # takes — because a fixture nobody calls is the same thing as no fixture.
    for _f in _cpu_fraction_fixture():
        print(f"  ! {_f}", file=sys.stderr)

    if args.spec and args.spec[0] == "amend":
        return cmd_amend(ledger, args)

    # `ratchets record` carries an argument the READ_ONLY dispatch below would
    # silently drop — and a dropped argument is the argv-is-a-spend trap in
    # miniature (the caller believes a recording happened). Handle the argued
    # form here; the bare `ratchets` flows through READ_ONLY_COMMANDS.
    if args.spec and args.spec[0] == "ratchets" and args.spec[1:]:
        if args.spec[1:] != ["record"]:
            print("ratchets: the only argument is `record`. Nothing was run.")
            return 2
        return cmd_ratchets(ledger, record=True)

    # `run decisions [--check]` / `run champions [--check]` — the forms the
    # governing pages actually document (`ladder_prompt.md:814`,
    # `DECISIONS_NEEDED.md`, `REVIEW_QUEUE.md` all name `run decisions`; none
    # names the module path). Until 2026-09-05 the word never reached this
    # dispatch: argparse rejected `--check` first, and the bare usage error
    # (rc=2) was read by two builder slots as a checker going red. Forward to
    # the tool that owns the flag; extra tokens refuse WHOLE per T0.23's rule.
    if args.spec and args.spec[0] in ("decisions", "champions"):
        if args.spec[1:]:
            print(f"Refusing to run: `{args.spec[0]}` takes no further "
                  "arguments (only --check). Nothing was run.")
            return 2
        if args.spec[0] == "decisions":
            from . import decisions as _tool
        else:
            from . import champions as _tool
        return _tool.main(["--check"] if args.check else [])
    if args.check:
        print("Refusing to run: --check belongs to `decisions` or "
              "`champions` alone. Nothing was run.")
        return 2

    # `blast-radius` is the one read-only command that takes spec ids, so it
    # routes before the no-argument dispatch below. It is read-only in the
    # strong sense the line under it means: no ledger write, no seeds, no GPU
    # — it answers a counterfactual over a view (`_AssumeStatus`), so the
    # argv-is-a-spend rule below is satisfied by construction rather than by
    # care.
    if args.spec and args.spec[0] == "blast-radius":
        return cmd_blast_radius(ledger, args.spec[1:])
    # `stale-cost` takes PATHS for the same reason, and is read-only in
    # the same strong sense: it parses `IMPL_DEPS` declarations and
    # writes nothing.
    if args.spec and args.spec[0] == "stale-cost":
        return cmd_stale_cost(ledger, args.spec[1:])
    # status/next/render are read-only and must not block on a running experiment.
    if args.spec and args.spec[0] in READ_ONLY_COMMANDS:
        return READ_ONLY_COMMANDS[args.spec[0]](ledger)
    if not args.spec and not args.gate and args.tier is None:
        return cmd_status(ledger)

    # THE LANE GUARD (103rd audit item 2). Everything below this line can
    # start an experiment, and an experiment launched into an abandoned lane
    # dies with its parent — seven occurrences, four of them (2026-09-19,
    # 04:1x/06:09/09:5x/10:1x) as session-child background launches ended on
    # a wake-up promise. Every prior remedy told someone AFTERWARDS; this one
    # refuses AT LAUNCH what it can prove abandoned (orphan, setsid) and
    # loudly marks what it cannot distinguish (see _lane_verdict for the
    # measurement). It sits BEFORE argv validation deliberately, so
    # `test_lane_guard.sh` can hit this exact call site with an argv that
    # cannot spend (the sixth/seventh occurrence proved a fixture that tests
    # the function but not the call ordering verifies nothing). The waiver is
    # for `dispatch.sh`'s setsid GPU watcher only, and it is a LOUD MARK, not
    # a silence.
    _refusals, _warnings = _lane_verdict()
    for _w in _warnings:
        print(_w)
    if _refusals:
        _waiver = os.environ.get(LANE_WAIVER_ENV, "").strip()
        if _waiver:
            print(f"LANE WAIVER ({LANE_WAIVER_ENV}): {_waiver}")
            for _r in _refusals:
                print(f"    waived: {_r}")
        else:
            print("Refusing to run: this launch is ABANDONED — no session "
                  "will outlive it:")
            for _r in _refusals:
                print(f"    {_r}")
            print("A registered run's only local lane is the FOREGROUND of a "
                  "session that\nstays open until the row is on the ledger. "
                  "If it cannot fit the slot,\nhand it forward as a unit — do "
                  "not start it and lose it. Multi-hour GPU\ndispatches go "
                  "through scripts/dispatch.sh (which sets the waiver).\n"
                  "Nothing was run.")
            return 3

    # ARGV IS A SPEND. Everything below this line can start an experiment, and
    # for a `gpu<*>` spec that means charging the weekly quota, so an argv this
    # runner does not fully understand must stop here rather than run the part
    # it recognised. See `t0_23_argv_is_not_a_spend.py` for the scar.
    unknown = [x for x in (args.spec or []) if x not in BY_ID]
    if unknown:
        print("Refusing to run: unrecognised argument(s): " + ", ".join(unknown))
        print("Commands: " + ", ".join(sorted(READ_ONLY_COMMANDS))
              + ", amend, decisions [--check], champions [--check].")
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
        passing = [s for s in LADDER if ledger.status(s.id) is Status.PASS]
        excluded = []
        if args.max_budget:
            cost_order = [b.value for b in Budget]
            ceiling = cost_order.index(args.max_budget)
            excluded = [s for s in passing
                        if cost_order.index(s.budget.value) > ceiling]
            passing = [s for s in passing
                       if cost_order.index(s.budget.value) <= ceiling]
        ids = _dependency_order([s.id for s in passing])
        if excluded:
            # A filter converts "not swept" into silence unless the residue is
            # counted (60th-audit lesson). Name every stamp this sweep does NOT
            # re-verify, so a bounded green is never read as a full one.
            print(f"BOUNDED GATE (--max-budget {args.max_budget}): "
                  f"{len(excluded)} PASS stamp(s) above the ceiling are NOT "
                  "re-verified by this sweep:")
            # 63rd audit B3: Budget prices WALL-CLOCK only. On this box the
            # binding dimension is RAM (SYSTEM.md's ~1.5 GB tenant ceiling),
            # and no dimension of this bound measures it — a spec under the
            # time ceiling can still exceed the memory constraint (T2.00
            # peaked at 7.57 GB inside cpu<10min, measured 2026-09-02).
            print("    NOTE: the ceiling bounds TIME only, not memory — a "
                  "spec inside this sweep may still exceed the box's RAM "
                  "constraint.")
            for s in excluded:
                print(f"    {s.id:8s} {s.budget.value}")
        # The gate is the ONE command here that can only lose certificates: it
        # re-runs rows that already hold clean stamps, so a dirty tree turns a
        # green sweep into a demotion. See `protocol.gate_precondition` for the
        # 2026-08-30 event that cost T0.09 and its 36 dependents.
        refusal = gate_precondition(working_tree_porcelain(),
                                    at_risk=len(ids), dirty_ok=args.dirty_ok)
        if refusal:
            print(refusal)
            print("Nothing was run.")
            return 1
        if args.dirty_ok:
            print(f"{GATE_DIRTY_FLAG}: gating a modified tree on purpose — "
                  "every spec re-run below will stamp `+dirty` and its clean "
                  "predecessor is lost. Read the results, do not commit them "
                  "as a certificate.")
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


def _exit_with_receipt(rc) -> None:
    """The exit code is ALSO the last line of stdout, so a pipe cannot erase
    it: `$?` after `tool | tail` reports the pipe's health, not the tool's,
    and that shape produced six false receipts in seven days across two
    organs (77th audit B1, gated by T0.23 P8). Argparse usage errors arrive
    as SystemExit and get the same receipt."""
    rc = 0 if rc is None else rc if isinstance(rc, int) else 1
    print(f"EXIT {rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    try:
        _rc = main()
    except SystemExit as _e:
        _rc = _e.code
    _exit_with_receipt(_rc)
