"""Re-buy stale CHEAP certificates mechanically, spending no Claude budget at all.

WHY THIS EXISTS, with the numbers that justify it (the BUILDER's own
measurement, 2026-09-28 — no owner act is cited or claimed; the 128th audit
found the original "(owner review)" phrasing here had no referent anywhere in
the repo and ordered it struck or substantiated, and struck is the honest one:
the only owner ruling this project has ever recorded is D19, 2026-09-17).

A ledger PASS is a claim about a specific piece of code. Edit that code and the
claim keeps asserting the old result under the new code's name, so `impl_sha`
stales it and it must be run again before it counts. That rule is load-bearing
and is not in question here.

What IS in question is who pays for it. On 2026-09-28 twenty-four certificates
stood stale, and eight Tier-0 harness specs had been re-bought TWENTY-TWO TIMES
each, while `T2.01` — "he can move" — had gone 47 days without an attempt and
`T6.01` — one life, start to finish — had never run at all.

The seductive diagnosis was "the re-buy bill is eating the project". It is not,
and the measurement says so bluntly: **those eight specs cost 80 SECONDS for a
full round.** All 22 rounds together are about 29 minutes of CPU across seven
weeks. Nothing. The expensive rows are the real science (`D1.0` 17.6 h, `T2.02`
6.3 h, `T2.01` 5.6 h).

So the cost was never compute. It was ATTENTION. Each of those 80-second runs
was noticed, chosen, executed, verified, committed and journalled by an hourly
Claude iteration — and that iteration is paid out of a weekly pool of which this
project measured its own share at NINE PERCENT (`usage_attribution.py`, 2026-09:
builder 9%, desks 4%, not-this-project 86%). A 40-second job was consuming a
scarce hour of judgment, and displacing the three specs that decide whether Jack
is a creature.

THE FIX IS NOT FINER DEPENDENCY TRACKING. That was the author's first proposal
and it was wrong on the arithmetic: scoping `IMPL_DEPS` per-symbol would have
saved 29 minutes of CPU and zero attention. The fix is to take the whole class
off the judgment lane, because a re-buy needs no judgment — the spec, the gate
and the thresholds are all already written. Re-running a test whose verdict is
computed by code nobody is changing is CLERICAL WORK, and clerical work should
not cost a model call.

WHAT THIS DOES AND REFUSES TO DO.

  * CHEAP ONLY. `cpu<1min` and `cpu<10min`. Anything longer, and anything that
    touches a GPU, stays with the loop — a 2-hour CPU job or a Kaggle dispatch
    involves budget and scheduling judgment that belongs to an iteration.
  * IT SPENDS NO CLAUDE BUDGET, and therefore is deliberately NOT behind
    `usage_gate` or `pace_gate`. Those two exist to ration the weekly model
    pool; this process never calls a model, so gating it on the model meter
    would starve the cheap lane to protect a resource it does not consume. It
    DOES honour `.paused`, because that switch means "stop everything" and must
    keep meaning that.
  * IT NEVER JUDGES A VERDICT. It shells out to the same runner the loop uses,
    so the runner's own locking, receipts and `impl_sha` stamping apply
    unchanged, and `experiments/ledger.json` is still written only by the
    runner. A re-run that turns a PASS into a FAIL is recorded as a FAIL. That
    is the point: this lane exists to keep the scoreboard TRUE, not green.
  * IT YIELDS TO THE LOOP. If the ladder lock is held, it exits. `run.py` takes
    that lock non-blocking and skips-on-held, so racing it would produce a
    silent no-op that looked like success — the exact failure shape this
    project has recorded five times (a control reading a proxy for the thing it
    governs).

    python scripts/regate.py --list     # both lanes (stale + rotation), no runs
    python scripts/regate.py            # re-buy: stale bills first, then the
                                        # standing-PASS rotation, oldest first
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/data/venvs/jackthelearner/bin/python"
RUN_LOCK = "/tmp/jack-ladder.lock"
CHEAP = ("cpu<1min", "cpu<10min")
# The label's NOMINAL promise, not rtf.BUDGET_SECONDS' padded child allowance.
# "Cheap" was a declaration until the 128th audit measured it: LT.02 declares
# cpu<10min and took 838 s, PS.08 declares cpu<10min and took ~1,048 s — both
# inside the padded kill-timeout, both 40-75% over the label they wear. A spec
# whose RECORDED runtime exceeds its label's nominal seconds is not cheap,
# whatever it declares, until someone re-labels it.
CHEAP_NOMINAL_S = {"cpu<1min": 60.0, "cpu<10min": 600.0}
# A cap, because an unbounded clerical lane on a tenant box is a new background
# service and this project is forbidden from adding one. Whatever is left is
# still stale next tick; nothing is lost by stopping early.
MAX_PER_RUN = int(os.environ.get("JACK_REGATE_MAX", "8"))

sys.path.insert(0, REPO)


def lock_is_held() -> bool:
    """True if the runner's lock is taken. Cheap, and never blocks on it."""
    try:
        import fcntl
        with open(RUN_LOCK, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
                return False
            except OSError:
                return True
    except Exception:
        # Unknown is not free: if the lock cannot be read, assume the loop owns
        # it. Yielding costs one tick; racing costs a silent no-op that reads
        # as a clean sweep.
        return True


def cheap_stale() -> list[tuple]:
    from experiments.cpu_budget import child_estimate_s, measured_child_seconds
    from experiments.protocol import Ledger
    from experiments.registry import BY_ID
    from experiments.run import stale_claims
    ledger = Ledger()
    out, seen, blocked, overran = [], set(), [], []
    for row in stale_claims(ledger):
        sid, status = row[0], row[1]
        # A spec can be reported by more than one staleness path (T6.03 is both
        # dirty-tree and changed-code). Re-running it twice in one sweep would
        # bill the second run against a certificate the first already bought.
        if sid in seen:
            continue
        seen.add(sid)
        spec = BY_ID.get(sid)
        if spec is None:
            continue
        if getattr(spec.budget, "value", "") not in CHEAP:
            continue
        # SKIP WHAT CANNOT PASS. A spec whose dependency is red comes back
        # BLOCKED however many times it is run, and re-running it every two
        # hours forever is the clerical lane inventing its own busywork — the
        # exact disease it was built to remove. Observed on the first two
        # sweeps: T6.03 -> BLOCKED by T2.10 (FAIL), T0.18 -> BLOCKED by T0.13
        # (FAIL), both re-run twice for nothing.
        #
        # The staleness is real and still owed; it is just not payable yet, and
        # it becomes payable the moment the blocker clears — at which point this
        # lane picks it up with no further instruction. Reported, not silently
        # dropped, because a queue that quietly shrinks is one nobody audits.
        # CHEAP BY MEASUREMENT, not by declaration (128th audit, RANK 2b).
        # The label filter above admitted LT.02 (838 s) and PS.08 (~1,048 s)
        # as "cpu<10min"; the recorded duration is the truth about what this
        # lane would actually spend. Reported, never silently dropped.
        measured = measured_child_seconds(sid)
        nominal = CHEAP_NOMINAL_S[spec.budget.value]
        if measured is not None and measured > nominal:
            overran.append((sid, measured, nominal, spec.budget.value))
            continue
        unsat = ledger.unsatisfied(spec)
        if unsat:
            blocked.append((sid, ", ".join(f"{d} ({w})" for d, w in unsat)))
            continue
        out.append((sid, status, spec.budget.value, *child_estimate_s(spec)))
    if blocked:
        for sid, why in blocked:
            print(f"  skip {sid:9s} stale but unpayable — blocked by {why}")
    for sid, measured, nominal, label in overran:
        print(f"  skip {sid:9s} declares {label} but measured {measured:.0f}s "
              f"against the label's {nominal:.0f}s — not cheap until re-labelled; "
              f"stays with the loop")
    return out


def cheap_pass_rotation(exclude: set) -> list[tuple]:
    """THE ADDED SELECTION PATH — ordered by the Review's 2026-10-06 CADENCE
    RULING on `standing-pass-certificates-are-falsifiable-only-by-running-them`
    and deliberately NOT routed through `stale_claims`.

    Why it exists, in the ruling's own arithmetic: `stale_claims` keys on a
    spec's own test-file `impl_sha` plus its DECLARED `IMPL_DEPS`, so a
    certificate that went latently red through an UNDECLARED edge is never
    selected by the lane above — the project re-ran certificates twelve times
    a day and still could not falsify the three it had already found red
    (`T0.13`, `T0.18`, `T0.22`); none of them moved its own `impl_sha`. The
    only thing that falsifies a standing PASS is RUNNING it, so this path runs
    the cheap standing-PASS population unconditionally, OLDEST `ran_at` first,
    behind the staleness lane and under the same slack guard and `MAX_PER_RUN`
    cap. At twelve ticks a day the full cheap population re-derives in a few
    days even when the staleness lane is busy.

    What it inherits from the lane above, deliberately: the cheap-by-
    MEASUREMENT rule (a spec whose recorded runtime exceeds its label's
    nominal seconds stays with the loop), and the skip-what-cannot-pass rule —
    a PASS whose dependency is currently red would come back BLOCKED, and
    demoting certificates in the `pass_on_dead_dependency` class is owned by
    the rows that explain them, not by a clerical cron. Both skips print.

    What it must never grow into: a judgment lane. It selects by age, runs
    through the same runner, and records whatever comes back. A re-run that
    turns a PASS into a FAIL is the point, not an incident.
    """
    from experiments.cpu_budget import child_estimate_s, measured_child_seconds
    from experiments.protocol import Ledger, Status
    from experiments.registry import BY_ID
    ledger = Ledger()
    out, blocked, overran = [], [], []
    for sid, spec in BY_ID.items():
        if sid in exclude:
            continue
        if getattr(spec.budget, "value", "") not in CHEAP:
            continue
        row = ledger.results.get(sid)
        if row is None or row.status is not Status.PASS:
            continue
        measured = measured_child_seconds(sid)
        nominal = CHEAP_NOMINAL_S[spec.budget.value]
        if measured is not None and measured > nominal:
            overran.append((sid, measured, nominal, spec.budget.value))
            continue
        unsat = ledger.unsatisfied(spec)
        if unsat:
            blocked.append((sid, ", ".join(f"{d} ({w})" for d, w in unsat)))
            continue
        out.append((row.ran_at or "", sid, "PASS", spec.budget.value,
                    *child_estimate_s(spec)))
    for sid, why in blocked:
        print(f"  skip {sid:9s} rotation would record BLOCKED, not a verdict — "
              f"dependency red: {why}; owned by its pass_on_dead_dependency row")
    for sid, measured, nominal, label in overran:
        print(f"  skip {sid:9s} declares {label} but measured {measured:.0f}s "
              f"against the label's {nominal:.0f}s — not cheap until re-labelled; "
              f"stays with the loop")
    out.sort()  # oldest ran_at first — the certificate longest unfalsified
    return [row[1:] for row in out]


def crossing_a_class_slack(est_s: float):
    """The 128th audit's RANK 2a repair: on its first day this lane spent
    4,218.6 s of the day's 4,450.2 s and foreclosed `cpu<2h` (37 specs, 36
    never run) — a class whose slack is 13x smaller than this lane's own —
    because it consulted no meter. So the guard is not this lane's own class
    budget: it is the TIGHTEST class's slack. `used_s > slack_s` is exactly
    "that class has >= 1 foreclosed member" (`class_slack`'s own documented
    equivalence), so refusing to be the spender that crosses it means the
    clerical lane can never again be the reason a science class closed.
    Returns the class row it would close (plus the live spend), or None.
    """
    from experiments.cpu_budget import CpuBudget, class_slack
    rows = class_slack()          # sorted tightest-slack first
    if not rows:
        return None
    used = CpuBudget().used_s()   # fresh read — the runner charges between runs
    tightest = rows[0]
    if used + est_s > tightest["slack_s"]:
        return {"used_s": used, **tightest}
    return None


def main(argv: list[str]) -> int:
    stale_rows = cheap_stale()
    rotation = cheap_pass_rotation(exclude={r[0] for r in stale_rows})
    # Staleness is an OWED bill and goes first; the rotation fills whatever is
    # left of the same cap. On a quiet tick (stale empty, the common case at
    # twelve ticks a day) the rotation gets the whole cap.
    queue = ([("stale", *r) for r in stale_rows]
             + [("rotate", *r) for r in rotation])
    if "--list" in argv:
        for lane, sid, status, cost, est_s, prov in queue:
            print(f"  {lane:6s} {sid:9s} {status:8s} {cost}  est {est_s:.0f}s [{prov}]")
        print(f"  {len(stale_rows)} cheap stale certificate(s); "
              f"{len(rotation)} in the standing-PASS rotation (oldest first)")
        return 0

    if not queue:
        print("nothing cheap is stale and the rotation is empty")
        return 0
    if lock_is_held():
        print("ladder lock held by the loop — yielding, these are still stale next tick")
        return 0

    ran, changed = 0, []
    for lane, sid, status, _cost, est_s, prov in queue[:MAX_PER_RUN]:
        hit = crossing_a_class_slack(est_s)
        if hit is not None:
            # Print every spec declined, and why — a lane that quietly stops
            # is one nobody audits (cce6cef's own rule).
            left = [r[1] for r in queue[:MAX_PER_RUN]][ran:]
            print(f"  stop before {sid}: projected {est_s:.0f}s [{prov}] on "
                  f"top of {hit['used_s']:.0f}s already spent today would cross "
                  f"{hit['budget']}'s slack {hit['slack_s']:.0f}s and close that "
                  f"class — declining {', '.join(left)}; still stale next tick")
            break
        p = subprocess.run([PY, "-m", "experiments.run", sid],
                           cwd=REPO, capture_output=True, text=True, timeout=1800)
        ran += 1
        tail = (p.stdout or "").strip().splitlines()
        verdict = next((l for l in reversed(tail)
                        if any(w in l for w in ("PASS", "FAIL", "VOID"))), "(no verdict line)")
        print(f"  {sid:9s} [{lane}] was {status:8s} -> {verdict.strip()[:96]}")
        if status not in verdict:
            changed.append(sid)
    print(f"re-bought {ran} of {len(queue)} cheap certificate(s) "
          f"({len(stale_rows)} stale + {len(rotation)} rotation)"
          + (f"; STATUS MOVED on {', '.join(changed)}" if changed else "; no status moved"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
