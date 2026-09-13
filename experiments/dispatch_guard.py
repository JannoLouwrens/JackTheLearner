"""Pre-flight refusals for a GPU dispatch — the executable form of rules that
until now existed only as prose.

WHY THIS FILE EXISTS (overseer, 91st audit, 2026-09-13, RANK 1 / B1).
`scripts/dispatch.sh` was 61 lines and refused exactly two things: an unpushed
HEAD and a held GPU lock. Meanwhile three separate documents said `D1.0`'s
attempt 3 may not be dispatched unchanged, `D1.0` had already consumed
**33.78 GPU-hours across two attempts for one ledger row, and that row is
VOID**, and attempt 3 measures ~17.6 h — 59% of a fresh weekly allocation. The
audit asked the question that turns a compliment into a finding: the builder
refused twice in eight hours, voluntarily — *what would have happened if it had
not?* Nothing. It would have dispatched, charged the week, and been discovered
afterwards.

The lesson it earned is in `docs/LESSONS.md`: *a prohibition is a sentence in a
document; a refusal is a branch that returns non-zero.* This module is the
branch. It adds NO new policy — every refusal below encodes a rule that was
already binding in prose, and each one names where the prose lives.

    R1  BUDGET        the projection does not fit the week's remaining hours
    R2  AUTHORISATION an UNCHANGED re-dispatch of a spec that already has a
                      settled ledger row
    R3  PROJECTION    no `--projected-hours` was stated, so neither R1 nor the
                      later projected-vs-actual audit can be computed

R3 is enforced by the CLI (the argument is required); R1 and R2 are functions
here so they can be exercised against constructed cases without dispatching
anything. Red-first: each refusal was shown FIRING before it was shown passing
— see `docs/LOOP_JOURNAL.md`, 2026-09-13.

---------------------------------------------------------------------------
R1 — BUDGET. Two judgements are baked in and both are stated rather than
implied.

*Which number.* `Budget.remaining_range(backend)[0]` — the FLOOR, which counts
unattributable hours as spent. `remaining_range`'s own docstring says the floor
is the safe direction and that "anyone RATIONING against the number should see
the range"; a guard is the definition of rationing, so it takes the floor and
prints the ceiling beside it.

*Which backend, and why this does NOT refuse everything.* Colab is unmetered by
construction (`Budget.remaining` returns `inf` for it), so the only pot that can
run out is Kaggle's 30 h. A job small enough for Colab therefore has a route
even when Kaggle is dry, and refusing it would be a self-inflicted foreclosure —
the exact disease the CPU accountant was repaired for twice in September, and
the thing `SYSTEM.md`'s standing prohibition on "building more meter" is aimed
at. So the rule is: **refuse only when Kaggle is the ONLY backend that can hold
the job.** The boundary is the registry's own — `Budget.GPU` is labelled
`gpu<2h` and `submit()`'s docstring says "short jobs belong on Colab and Kaggle
is spent on work that needs the session length" — so a projection above
`COLAB_MAX_HOURS` is Kaggle-only and is rationed; anything at or below it is
reported and cleared.

Consequence, checked rather than asserted: `D1.0` attempt 3 at 17.6 h against
`W37`'s fresh 30.0 h does NOT trip R1, exactly as B1 predicted it should not.
It trips at a projection above the floor, and the arithmetic is printed either
way.

---------------------------------------------------------------------------
R2 — AUTHORISATION. The comparator is `impl_sha`, and the choice is the whole
substance of the guard, so B1 asked for it to be decided in writing.

*What `impl_sha` is:* `protocol.impl_sha_of` — sha256 of the test module's own
bytes PLUS every file it declares in `IMPL_DEPS`. It is the stamp the ledger row
already carries, meaning "this is the code this verdict is about".

*Why not "has any commit touched `IMPL_DEPS` since the row's commit?"*, which
is the shape B1's wording suggests. `run.stale_claims` tried exactly that
comparator first and recorded the result in its docstring: a test is written,
RUN, and only THEN committed, so the recorded commit predates the test's own
first commit and **every honest entry fires** — 15 of 54 at the time, a 100%
false-positive rate on healthy rows. A refusal with that property would be
switched off within a day. `impl_sha` answers the same question without a clock.

*The ruling B1 asked for, on `D1.0` specifically:* commit `7cb00ea` committed
the successor learning gate, and that gate lives INSIDE
`d1_0_control_path_bakeoff.py`. So the module's bytes moved, so `impl_sha`
moved — recorded `4db15d96b0312e50`, current `08621094c15c473a`. **`7cb00ea`
counts as a change, and R2 does not trip for `D1.0` attempt 3.** Read that
precisely: R2 has nothing to say about attempt 3; it does not authorise it. The
row `d10-successor-rerun-under-adopted-gate` (DUE 09-14) is what authorises a
dispatch, and this module cannot and does not speak for it.

*The hole, named rather than hidden.* `impl_sha` only covers DECLARED deps. A
spec that imports a repo-root module without declaring it can have that module
rewritten under it and the stamp will not move — `D1.0` imports
`TrainingPipeline` and `UnifiedBrain` and declares neither. That is not fixable
here (declaring them re-stamps and stales the row), so R2 PRINTS the undeclared
imports on every verdict, clear or refused, as a standing caveat on its own
authority. A guard that quietly overstates its coverage is the failure this
whole file is a response to.

*Which statuses count as settled.* `PASS`, `FAIL`, `VOID`. A `VOID` counts
deliberately: an unchanged re-run of a VOID is precisely the seed-lottery redraw
that `T2.01` and `D1.0` are both on record refusing. A row with no `impl_sha`
at all cannot be proven changed, so it REFUSES rather than clearing — the safe
direction, and it costs almost nothing: of the 25 GPU-cost specs carrying a
ledger row today, exactly one (`T2.02`, already flagged as predating `impl_sha`)
is in that state.

---------------------------------------------------------------------------
R3 — PROJECTION, and what gets written. `--projected-hours` is required, and
the value is appended to `gpu_budget.json` under `projections` with the spec,
the week key, the head sha and the wall clock. Until now `D1.0`'s 17.61 h
estimate existed only in prose; with a receipt at dispatch time and
`charged_jobs` at settle time, projected-vs-actual becomes a join instead of an
archaeology exercise. The write takes the same `fcntl.flock` on the same lock
path `Budget.charge` uses, and `Budget._load()` preserves unknown top-level
keys, so a concurrent charge cannot erase a projection and vice versa.

This module deliberately does NOT edit `experiments/gpu.py`: `T0.12` declares
that file in its `IMPL_DEPS`, and staling a PASS certificate to add a guard
elsewhere would be a cost with no measurement behind it.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.gpu import BUDGET_FILE, KAGGLE_WEEKLY_HOURS, Budget
from experiments.protocol import (Ledger, impl_sha_of, module_path_for,
                                  undeclared_impl_imports)

REPO = Path(__file__).resolve().parent.parent

#: Above this, Kaggle's session length is the only thing that can hold the job,
#: so the weekly pot becomes a hard constraint. Taken from the registry's own
#: `Budget.GPU = "gpu<2h"` boundary, not invented here.
COLAB_MAX_HOURS = 2.0

#: A row in one of these states is a verdict about a specific `impl_sha`.
#: `ERROR` is absent on purpose: a harness crash measured nothing, and re-running
#: unchanged code after one is the correct act, not a redraw.
SETTLED = ("PASS", "FAIL", "VOID")


def _head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def budget_refusal(projected_hours: float,
                   budget: Budget | None = None) -> tuple:
    """R1. Returns `(refused, lines)`; `lines` is printed either way.

    `budget` is injectable so a constructed case can be run against a temporary
    file — the same reason `submit()` takes one. A guard that can only be
    exercised by spending the real quota is not a guard that gets exercised.
    """
    budget = budget or Budget()
    lo, hi = budget.remaining_range("kaggle")
    wk = budget._week()
    used = budget.used_hours("kaggle")
    lines = [f"BUDGET  week {wk}: kaggle used {used:.2f}h, remaining "
             f"{lo:.2f}h (floor; up to {hi:.2f}h) — projection "
             f"{projected_hours:.2f}h"]
    if projected_hours <= COLAB_MAX_HOURS:
        lines.append(f"        clear: {projected_hours:.2f}h <= "
                     f"COLAB_MAX_HOURS {COLAB_MAX_HOURS:.1f}h, so colab is a "
                     f"legal route and colab is unmetered — not rationed.")
        return False, lines
    if projected_hours > lo:
        lines.append(f"REFUSING: projection {projected_hours:.2f}h exceeds "
                     f"{wk}'s remaining kaggle floor {lo:.2f}h, and "
                     f"{projected_hours:.2f}h > COLAB_MAX_HOURS "
                     f"{COLAB_MAX_HOURS:.1f}h so kaggle is the only backend "
                     f"that can hold it.")
        lines.append(f"          {used:.2f}h already charged this week of "
                     f"{KAGGLE_WEEKLY_HOURS:.1f}h. Wait for the reset, or "
                     f"dispatch something that fits, or state a smaller "
                     f"projection AND make it true.")
        return True, lines
    lines.append(f"        clear: {projected_hours:.2f}h fits the {lo:.2f}h "
                 f"floor.")
    return False, lines


def redispatch_refusal(spec_id: str, ledger: Ledger | None = None) -> tuple:
    """R2. Returns `(refused, lines)`. See the module docstring for the ruling
    on which comparator this uses and why."""
    ledger = ledger if ledger is not None else Ledger()
    row = ledger.results.get(spec_id)
    if row is None:
        return False, [f"REDISPATCH  no ledger row for {spec_id} — first "
                       f"dispatch, nothing to compare."]
    # `status` is a `Status` enum on a `Result`; compare on its value so the
    # SETTLED tuple above stays readable as the three words it means.
    status = getattr(row.status, "value", row.status)
    if status not in SETTLED:
        return False, [f"REDISPATCH  {spec_id} row is {status!r}, not a "
                       f"settled verdict ({'/'.join(SETTLED)}) — not a "
                       f"re-dispatch."]
    path = module_path_for(spec_id)
    if path is None:
        return False, [f"REDISPATCH  {spec_id} has a {status} row but no "
                       f"implementation file — cannot compare; nothing to "
                       f"dispatch either."]
    recorded = row.impl_sha
    current = impl_sha_of(path)
    head = (f"{spec_id} attempt {row.attempt} {status} at {row.ran_at} "
            f"(commit {row.commit})")
    undeclared, problem = undeclared_impl_imports(path)
    caveat = []
    if undeclared:
        caveat = [f"            CAVEAT: impl_sha does not cover "
                  f"{', '.join(undeclared)} — imported by {path.name}, not "
                  f"declared in IMPL_DEPS. A change there is invisible to this "
                  f"refusal."]
    if problem:
        caveat.append(f"            CAVEAT: IMPL_DEPS unreadable ({problem}).")
    if not recorded:
        return True, ([f"REFUSING: unchanged re-dispatch — {head}",
                       f"          that row carries NO impl_sha, so this "
                       f"dispatch cannot be PROVEN to run different code. "
                       f"Re-stamp the row (`run amend`) or re-run on CPU "
                       f"first; a guard may not clear what it cannot check."]
                      + caveat)
    if recorded == current:
        return True, ([f"REFUSING: unchanged re-dispatch — {head}",
                       f"          impl_sha {recorded} is UNCHANGED. Re-running "
                       f"identical code against a settled verdict is a "
                       f"seed-lottery redraw (T2.01 2026-08-13; D1.0 "
                       f"2026-09-04). Change the arm or the gate, and commit "
                       f"it, before spending the quota."]
                      + caveat)
    return False, ([f"REDISPATCH  {head}",
                    f"            impl_sha MOVED {recorded} -> {current}: this "
                    f"is a changed re-dispatch, so R2 has nothing to say about "
                    f"it. NOTE: that is not authorisation — whatever row or "
                    f"ruling governs this spec still does."]
                   + caveat)


def record_projection(spec_id: str, projected_hours: float,
                      path: Path = BUDGET_FILE) -> dict:
    """R3's receipt. Append-only, under `Budget.charge`'s own lock."""
    entry = {"spec": spec_id, "hours": round(float(projected_hours), 4),
             "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "week": Budget._week(), "head": _head()}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        data.setdefault("projections", []).append(entry)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
    return entry


def preflight(spec_id: str, projected_hours: float, *, record: bool = False,
              budget: Budget | None = None,
              ledger: Ledger | None = None) -> tuple:
    """Run every refusal. Returns `(ok, lines)`. Nothing is written unless
    `record` and every refusal cleared — a refused dispatch leaves no
    projection, so the receipt log means "was allowed to go", not "was
    considered"."""
    lines = [f"dispatch pre-flight: {spec_id}, projected "
             f"{projected_hours:.2f} GPU-h"]
    r2, l2 = redispatch_refusal(spec_id, ledger=ledger)
    lines += l2
    r1, l1 = budget_refusal(projected_hours, budget=budget)
    lines += l1
    if r1 or r2:
        return False, lines
    if record:
        e = record_projection(spec_id, projected_hours)
        lines.append(f"PROJECTION recorded in {BUDGET_FILE.name}: "
                     f"{e['spec']} {e['hours']}h week {e['week']} head "
                     f"{e['head']}")
    lines.append("pre-flight CLEAR — this says no rule REFUSED the dispatch. "
                 "It is not an argument that the dispatch is worth making.")
    return True, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.dispatch_guard",
        description="Pre-flight refusals for a GPU dispatch (91st audit B1).")
    ap.add_argument("spec_id")
    ap.add_argument("--projected-hours", type=float, required=True,
                    help="GPU-hours this dispatch is expected to cost. "
                         "REQUIRED: without it neither the budget refusal nor "
                         "the projected-vs-actual audit can be computed.")
    ap.add_argument("--record", action="store_true",
                    help="on a clear verdict, append the projection to "
                         "gpu_budget.json")
    args = ap.parse_args(argv)
    if args.projected_hours <= 0:
        print("REFUSING: --projected-hours must be positive.", file=sys.stderr)
        return 2
    ok, lines = preflight(args.spec_id, args.projected_hours,
                          record=args.record)
    for line in lines:
        print(line, file=sys.stderr if not ok else sys.stdout, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
