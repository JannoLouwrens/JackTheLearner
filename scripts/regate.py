"""Re-buy stale CHEAP certificates mechanically, spending no Claude budget at all.

WHY THIS EXISTS, with the numbers that justify it (owner review, 2026-09-28).

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

    python scripts/regate.py --list     # what is cheap and stale, no runs
    python scripts/regate.py            # re-buy them, oldest bill first
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/data/venvs/jackthelearner/bin/python"
RUN_LOCK = "/tmp/jack-ladder.lock"
CHEAP = ("cpu<1min", "cpu<10min")
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


def cheap_stale() -> list[tuple[str, str, str]]:
    from experiments.protocol import Ledger
    from experiments.registry import BY_ID
    from experiments.run import stale_claims
    out, seen = [], set()
    for row in stale_claims(Ledger()):
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
        if getattr(spec.budget, "value", "") in CHEAP:
            out.append((sid, status, spec.budget.value))
    return out


def main(argv: list[str]) -> int:
    rows = cheap_stale()
    if "--list" in argv:
        for sid, status, cost in rows:
            print(f"  {sid:9s} {status:8s} {cost}")
        print(f"  {len(rows)} cheap stale certificate(s)")
        return 0

    if not rows:
        print("nothing cheap is stale")
        return 0
    if lock_is_held():
        print("ladder lock held by the loop — yielding, these are still stale next tick")
        return 0

    ran, changed = 0, []
    for sid, status, _cost in rows[:MAX_PER_RUN]:
        p = subprocess.run([PY, "-m", "experiments.run", sid],
                           cwd=REPO, capture_output=True, text=True, timeout=1800)
        ran += 1
        tail = (p.stdout or "").strip().splitlines()
        verdict = next((l for l in reversed(tail)
                        if any(w in l for w in ("PASS", "FAIL", "VOID"))), "(no verdict line)")
        print(f"  {sid:9s} was {status:8s} -> {verdict.strip()[:96]}")
        if status not in verdict:
            changed.append(sid)
    print(f"re-bought {ran} of {len(rows)} cheap stale certificate(s)"
          + (f"; STATUS MOVED on {', '.join(changed)}" if changed else "; no status moved"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
