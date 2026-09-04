"""CPU-hour accounting for the runner's children on a shared box (T0.33).

This box hosts paying tenants; SYSTEM.md ranks their safety above the ladder.
Until T0.33 the only protections were the loop's single load reading at
iteration start (`ladder_loop.sh MAX_LOAD`) and the per-child kill timeout —
nothing accumulated, so a day of back-to-back cpu<2h children was invisible to
every instrument while it happened. This module is the GPU budget's CPU
sibling: a plain JSON file a person can read, a debit for every runner child,
and a refusal BEFORE a child spawns.

Scope, stated honestly:
  - The metered unit is `run.py:_run_isolated`'s child — the only lane
    `cmd_run` (and therefore `--gate`) uses. The gate still refuses `cpu<48h`
    with a ROUTING reason: that class belongs to the detached lane, which
    since T0.34 keeps its own accounts (`admit_detached` + the heartbeat
    wrapper below, called by `scripts/launch_detached.sh`). A module invoked
    BY HAND remains unmetered — a human at a shell is the owner's lane.
  - GPU-budget children are not charged: their wall clock is mostly waiting on
    a remote kernel, and billing waiting as box CPU would make the meter read
    harm where there is none. Their remote cost is T0.12's.
  - `afford()` gates on `child_estimate_s` — the spec's own MEASURED last
    child duration where the ledger has one, the enum worst case where it does
    not — and `charge()` bills the wall clock actually spent; like the GPU
    meter, an overrun past the ceiling is OBSERVED with a mark, while the
    refusal stops new work from starting.

Estimating on the enum instead of the measurement (69th audit B4): until
2026-09-04 this gate refused on `spec_child_timeout_seconds` alone — the
child-KILL allowance, which is deliberately the worst case a budget class may
legally reach. Across the 108 runner-lane cpu specs that carry a recorded
duration the median ratio of that allowance to the spec's actual cost is
**257x** (`W0.DIAG` 0.02 s against 10800 s; `LG.02` 1.9 s against 54000 s),
and the consequence was measured live: **3600 s of routine housekeeping —
6.25% of the ceiling — foreclosed 53 of 152 CPU specs**, because any `cpu<2h`
spec estimates at 54000 s and the day only ever holds 57600 s.

The projection is bounded on the side that matters: `child_estimate_s` returns
`min(enum, SAFETY x measured + overhead)`, so it can only ever LOWER an
estimate. Admission is therefore never tighter than it was before this change,
and the hard ceiling is untouched — `run.py` still kills the child at
`spec_child_timeout_seconds`, so an admitted child's true worst case is exactly
what it always was, and a projection that undershoots costs a MARKED overrun
(the posture `admit_detached` already takes) rather than an unbounded run.
`SAFETY` and the overhead are engineering choices, not measurements, and are
declared as such at their definitions.

Accounting ownership (68th audit B1): when two billers meter one resource,
their charges must be DISJOINT, and the owner here is the OUTERMOST wrapper.
`_wrap` bills the whole tree's wall clock under its label and exports
JACK_CPU_WRAPPED into the payload's environment; `charge_cpu_child` — the
runner lane's only debit path — skips under that marker, because every second
a wrapped descendant spends is already inside the interval the wrapper's
heartbeat bills. A nested `wrap` likewise defers to the outer one. Before
this, a detached sweep was billed once by the wrapper and again by each
`run_spec` grandchild: 1.7x overcharge, measured live on 2026-09-04, ~35
minutes from foreclosing 53 of 152 CPU specs on 2.4% of the ceiling genuinely
spent. The wrapper owns the charge because it is the only process whose wall
clock covers the tree end to end, including the seams no child bills.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .rtf import spec_child_timeout_seconds

BUDGET_FILE = Path(__file__).parent / "cpu_budget.json"
ENV_OVERRIDE = "JACK_CPU_BUDGET"   # tests point this at a temp file
ENV_WRAPPED = "JACK_CPU_WRAPPED"   # set by _wrap: the tree's charge has an owner

# The ladder's children may occupy at most this much wall clock per calendar
# day. 16 h: (a) it ADMITS the largest legal child from a fresh day — cpu<2h
# x 3 seeds x 2 = 54000 s; a tighter ceiling would foreclose that class by
# arithmetic, the ME.11.E disease, and T0.33 gates `cpu_foreclosed == []` so
# the property is checked, not assumed; (b) children are single-process
# Python, so 16 h is <= 1/6 of this 4-core box's core-day, and the
# instantaneous side stays with the load gate below and the mem watchdog.
CPU_DAY_CEILING_S = 57600.0

# Must equal `ladder_loop.sh`'s MAX_LOAD — one threshold, two languages;
# T0.33 parses the shell file and fails if they drift apart.
LOAD_CEILING = 6.0

# ── The admission estimate (69th audit B4) ──────────────────────────────────
# The ledger is the only per-spec record of what a child ACTUALLY cost:
# `Result.duration_s`, the wall clock of `run_spec` inside the child.
LEDGER_FILE = Path(__file__).parent / "ledger.json"

# Two engineering constants, and they are NOT measurements — say so plainly,
# because this file's own first law is that a number is claimed only by
# something that could have failed:
#   - CHILD_OVERHEAD_S is the one piece with data behind it. The gate meters
#     the CHILD's wall (interpreter start + imports + run_spec) while the
#     ledger records only run_spec's interior; the difference measured across
#     the eight runner children billed on 2026-09-03/04 was 0.55 s (T0.34) to
#     5.38 s (T0.33). 10 s is ~2x the largest observed.
#   - PROJECTION_SAFETY covers run-to-run variance and implementation drift
#     since the recorded run, and there is NO data for it here: the ledger
#     keeps one duration per spec and its `history[]` entries carry none, so
#     per-spec variance is unmeasured. 4x is a choice whose cost of being
#     wrong is bounded by the enum clamp below and by the child-kill timeout,
#     which this projection does not touch. If `history[]` ever carries
#     durations, derive it instead of declaring it.
PROJECTION_SAFETY = 4.0
CHILD_OVERHEAD_S = 10.0


def _budget_path() -> Path:
    override = os.environ.get(ENV_OVERRIDE)
    return Path(override) if override else BUDGET_FILE


def _day() -> str:
    return time.strftime("%Y-%m-%d")


def _loadavg() -> float:
    """1-minute load. Unreadable is +inf, not 0 — a meter that fails open is
    not a limit (T0.12's rule, one resource over)."""
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return float("inf")


@dataclass(frozen=True)
class CpuDecision:
    admitted: bool
    est_s: float
    remaining_s: float
    load: float
    reason: str


class CpuBudget:
    """Daily wall-clock accounting for runner children.

    The same discipline as `gpu.Budget`: lock, RE-READ from disk, mutate,
    write atomically — a writer that writes from state loaded at construction
    erases every charge made in between (the 2026-08-12 stale-writer clobber,
    measured at 0.5498 h).
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else _budget_path()
        self.data = self._load()

    def _load(self) -> dict:
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        data.setdefault("days", {})
        data.setdefault("overruns", [])
        return data

    def _key(self, day: str | None = None) -> str:
        """The bucket a reading or charge lands in. Overridable so T0.33's
        leaky control can collapse days the way the retired ISO week format
        collapsed GPU weeks."""
        return day or _day()

    def used_s(self, day: str | None = None) -> float:
        return float(self.data["days"].get(self._key(day), {}).get("used_s", 0.0))

    def remaining_s(self, day: str | None = None) -> float:
        return max(0.0, CPU_DAY_CEILING_S - self.used_s(day))

    def afford(self, est_s: float, day: str | None = None) -> bool:
        return est_s <= self.remaining_s(day)

    def charge(self, spec_id: str, seconds: float,
               day: str | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            self.data = self._load()
            key = self._key(day)
            bucket = self.data["days"].setdefault(key, {"used_s": 0.0,
                                                        "by_spec": {}})
            bucket["used_s"] = round(bucket["used_s"] + seconds, 2)
            by = bucket["by_spec"]
            by[spec_id] = round(by.get(spec_id, 0.0) + seconds, 2)
            if bucket["used_s"] > CPU_DAY_CEILING_S:
                self.data["overruns"].append({
                    "day": key, "used_s": bucket["used_s"],
                    "ceiling_s": CPU_DAY_CEILING_S, "spec_id": spec_id,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, self.path)


_DURATION_CACHE: dict = {}    # (path, mtime, size) -> {spec_id: duration_s}


def _durations(ledger_path: Path | None = None) -> dict:
    """`{spec_id: duration_s}` from a ledger, cached on (path, mtime, size).

    `foreclosed_now` estimates every registered cpu spec, so an uncached read
    parses the 1 MB ledger 152 times per call — 1.5 s inside `run status`, and
    quadratic the first time T0.33 asked for the foreclosed SET per spec.
    Keyed on mtime+size rather than path alone so a ledger written mid-process
    (the runner does exactly that) is re-read.
    """
    path = Path(ledger_path) if ledger_path else LEDGER_FILE
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    hit = _DURATION_CACHE.get(key)
    if hit is None:
        try:
            rows = json.loads(path.read_text())["results"]
            hit = {sid: float(r["duration_s"]) for sid, r in rows.items()
                   if isinstance(r, dict) and r.get("duration_s")}
        except Exception:
            hit = {}
        if len(_DURATION_CACHE) > 8:   # bounded; superseded revisions age out
            _DURATION_CACHE.clear()
        _DURATION_CACHE[key] = hit
    return hit


def measured_child_seconds(spec_id: str,
                           ledger_path: Path | None = None) -> float | None:
    """The spec's last recorded child duration, or None if there is no
    measurement to project from.

    Fails to None on ANY read problem — a missing, truncated or unparseable
    ledger falls back to the enum, which is the RESTRICTIVE side. (T0.12's
    "a meter that fails open is not a limit" points the other way for a
    reading; here the projection's failure mode is to keep today's behaviour,
    so failing closed is the same instinct pointed correctly.)
    """
    d = _durations(ledger_path).get(spec_id)
    return d if d and d > 0.0 else None


def child_estimate_s(spec, ledger_path: Path | None = None) -> tuple:
    """`(seconds, provenance)` — the admission estimate for one runner child.

    MEASURED when the ledger carries a duration for this spec; ENUM otherwise.
    Clamped at the enum by construction, so the projection may only TIGHTEN an
    estimate and this gate can never refuse something it would have admitted
    before B4 (T0.33 property 12 asserts that over the whole registry).
    """
    enum_s = float(spec_child_timeout_seconds(spec))
    measured = measured_child_seconds(spec.id, ledger_path)
    if measured is None:
        return enum_s, "ENUM (no recorded duration to project from)"
    proj = PROJECTION_SAFETY * measured + CHILD_OVERHEAD_S
    if proj >= enum_s:
        return enum_s, (f"ENUM (projection from {measured:.2f}s reaches the "
                        f"worst case)")
    return round(proj, 2), (f"MEASURED {measured:.2f}s x{PROJECTION_SAFETY:g} "
                            f"+ {CHILD_OVERHEAD_S:g}s")


def gate_cpu_child(spec, *, path: Path | None = None,
                   loadavg: float | None = None,
                   ledger_path: Path | None = None) -> CpuDecision:
    """Admit or refuse one runner child BEFORE it spawns.

    Refusal writes no ledger row — tenant protection is not a measurement of
    the spec — so `_run_isolated` returns the refusal unrecorded.
    """
    budget = spec.budget.value
    load = _loadavg() if loadavg is None else loadavg
    if not budget.startswith("cpu"):
        return CpuDecision(True, 0.0, float("inf"), load,
                           f"{budget} is not a CPU child; its remote cost is "
                           f"metered by T0.12")
    if budget == "cpu<48h":
        return CpuDecision(False, 0.0, 0.0, load,
                           "cpu<48h is the detached lane "
                           "(scripts/launch_detached.sh), not a runner child")
    est, prov = child_estimate_s(spec, ledger_path)
    if load > LOAD_CEILING:
        return CpuDecision(False, est, CpuBudget(path).remaining_s(), load,
                           f"load {load:.2f} above {LOAD_CEILING} — leaving "
                           f"the box to the tenants")
    remaining = CpuBudget(path).remaining_s()
    if est > remaining:
        return CpuDecision(False, est, remaining, load,
                           f"day budget: projected child {est:.0f}s "
                           f"[{prov}] exceeds remaining {remaining:.0f}s of "
                           f"{CPU_DAY_CEILING_S:.0f}s")
    return CpuDecision(True, est, remaining, load,
                       f"within the day budget [{prov}]")


def runner_cpu_specs() -> list:
    """The registered specs this meter gates: cpu, minus the detached lane."""
    from .registry import BY_ID
    return [s for s in BY_ID.values()
            if s.budget.value.startswith("cpu") and s.budget.value != "cpu<48h"]


def foreclosed_now(path: Path | None = None,
                   ledger_path: Path | None = None) -> list:
    """The registered runner-lane cpu specs the LIVE day would refuse right
    now, sorted. One source for three readers — `run status`'s visibility
    block, `run status`'s ratchet counter, and T0.33's `n_foreclosed_now`
    metric — because a refusal returns UNRECORDED by design and three
    independent copies of this arithmetic is the T0.14 pasted-constant scar.

    Monotone within a calendar day: `used_s` only grows and `child_estimate_s`
    does not depend on the clock, so the live reading IS the day's peak so
    far. It falls to 0 at midnight, and that MOVED line is the day rolling
    over, not an instrument fault.
    """
    remaining = CpuBudget(path).remaining_s()
    return sorted(s.id for s in runner_cpu_specs()
                  if child_estimate_s(s, ledger_path)[0] > remaining)


def class_slack(path: Path | None = None,
                ledger_path: Path | None = None) -> list:
    """Per cost class, the arithmetic that turns `foreclosed_now`'s bare count
    into something an iteration can act on (70th audit B4).

    `slack_s = CPU_DAY_CEILING_S - max(child_estimate_s over the class's LIVE
    population)` — the day's spend at which the class starts losing members.
    Compare it against `used_s` and the count stops being a mystery: on
    2026-09-04 `cpu<2h` had 3600 s of slack against 6280 s spent, so all 39 of
    its foreclosed specs were foreclosed by ~01:30 and nothing that happened
    afterwards could have changed that.

    Why max and not min: the class begins foreclosing at its LARGEST member and
    finishes at its smallest, so `slack_s` is the first threshold crossed and
    `full_slack_s` the last. The equivalence `used_s > slack_s` <=>
    `n_foreclosed >= 1` is exact (T0.33 property 16 asserts it at a mid-range
    value), because both sides are the same comparison rearranged — which is
    the point of deriving it here, next to the gate, rather than in the printer.

    Reads the estimate through `child_estimate_s`, so a class row can never
    disagree with the refusal it predicts. Sorted tightest-slack first: the
    class nearest to closing is the one worth reading.
    """
    remaining = CpuBudget(path).remaining_s()
    used = CpuBudget(path).used_s()
    rows: dict = {}
    for s in runner_cpu_specs():
        est = child_estimate_s(s, ledger_path)[0]
        r = rows.setdefault(s.budget.value, {"budget": s.budget.value, "n": 0,
                                             "max_est_s": 0.0,
                                             "min_est_s": float("inf"),
                                             "n_foreclosed": 0,
                                             "n_unmeasured": 0})
        r["n"] += 1
        r["max_est_s"] = max(r["max_est_s"], est)
        r["min_est_s"] = min(r["min_est_s"], est)
        if est > remaining:                       # the gate's own comparison
            r["n_foreclosed"] += 1
            if measured_child_seconds(s.id, ledger_path) is None:
                r["n_unmeasured"] += 1
    for r in rows.values():
        r["used_s"] = round(used, 2)
        r["slack_s"] = round(CPU_DAY_CEILING_S - r["max_est_s"], 2)
        r["full_slack_s"] = round(CPU_DAY_CEILING_S - r["min_est_s"], 2)
    return sorted(rows.values(), key=lambda r: (r["slack_s"], r["budget"]))


def charge_cpu_child(spec_id: str, seconds: float,
                     path: Path | None = None) -> None:
    """The runner lane's debit. Under a detached wrapper this is a NO-OP:
    the wrapper's heartbeat already bills the wall interval this child ran
    inside, and a second debit here is the 1.7x double-charge the 68th audit
    caught foreclosing the day (charges must be disjoint; the outermost
    wrapper owns the tree's charge)."""
    if os.environ.get(ENV_WRAPPED):
        return
    CpuBudget(path).charge(spec_id, seconds)


# ── The detached lane (T0.34) ────────────────────────────────────────────────
# `scripts/launch_detached.sh` calls `admit` before setsid and runs the
# payload under `wrap`, which bills measured wall clock every heartbeat.
# Billing incrementally (not lump-sum at exit) is load-bearing twice: a
# multi-day child is charged to every day it occupied instead of dumping
# 57 h into one 16 h bucket, and a SIGKILL of the process group loses at
# most one heartbeat of charge instead of the whole life.

HEARTBEAT_S = 600.0
ENV_HEARTBEAT = "JACK_CPU_HEARTBEAT_S"   # tests shorten it; default stands


def _heartbeat_s() -> float:
    try:
        return float(os.environ.get(ENV_HEARTBEAT, HEARTBEAT_S))
    except ValueError:
        return HEARTBEAT_S


def bill_interval(label: str, t0: float, t1: float,
                  path: Path | None = None) -> None:
    """Charge the wall interval [t0, t1], split across the calendar days it
    spans — each day is billed exactly the seconds the interval spent inside
    it, so the day ledger stays meaningful for children that outlive the day
    that admitted them."""
    b = CpuBudget(path)
    while t0 < t1:
        lt = time.localtime(t0)
        day = time.strftime("%Y-%m-%d", lt)
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                                0, 0, 0, 0, 0, -1))
        day_end = midnight + 86400.0
        seg = min(t1, day_end)
        if seg > t0:
            b.charge(label, seg - t0, day=day)
        t0 = seg


def admit_detached(label: str, *, path: Path | None = None,
                   loadavg: float | None = None) -> CpuDecision:
    """Admit or refuse a detached launch BEFORE setsid. Est-free by design —
    a cpu<48h child cannot pre-fit a 16 h day, so the gate asks only whether
    TODAY has headroom and the box is calm; the ceiling binds the running
    child through `charge`'s overrun marks, never through a kill."""
    load = _loadavg() if loadavg is None else loadavg
    if load > LOAD_CEILING:
        return CpuDecision(False, 0.0, CpuBudget(path).remaining_s(), load,
                           f"load {load:.2f} above {LOAD_CEILING} — leaving "
                           f"the box to the tenants")
    remaining = CpuBudget(path).remaining_s()
    if remaining <= 0.0:
        return CpuDecision(False, 0.0, 0.0, load,
                           f"day budget: {CPU_DAY_CEILING_S:.0f}s already "
                           f"used — no new detached launch today "
                           f"(label {label})")
    return CpuDecision(True, 0.0, remaining, load,
                       f"admitted (label {label}); wall billed per "
                       f"{_heartbeat_s():.0f}s heartbeat")


def _wrap(label: str, argv: list) -> int:
    """Run argv as a child, bill its wall clock every heartbeat, propagate
    its exit code. SIGTERM is forwarded so a polite kill of the wrapper
    takes the payload with it; a SIGKILL must target the process group.

    The OUTERMOST wrapper owns the whole tree's charge: it exports
    ENV_WRAPPED so every `charge_cpu_child` beneath it skips (their wall
    clock is inside this heartbeat's interval), and a nested wrap finds the
    marker already set and bills nothing itself."""
    import signal
    import subprocess
    import sys
    owner = not os.environ.get(ENV_WRAPPED)
    env = dict(os.environ)
    env.setdefault(ENV_WRAPPED, label)
    try:
        proc = subprocess.Popen(argv, env=env)
    except OSError as e:
        print(f"cpu_budget wrap: cannot spawn {argv[0]!r}: {e}",
              file=sys.stderr)
        return 127
    signal.signal(signal.SIGTERM, lambda s, f: proc.terminate())
    last = time.time()
    while True:
        try:
            proc.wait(timeout=_heartbeat_s())
            done = True
        except subprocess.TimeoutExpired:
            done = False
        now = time.time()
        if owner and now > last:
            bill_interval(label, last, now)
            last = now
        if done:
            return int(proc.returncode)


def _main(argv: list) -> int:
    import sys
    if len(argv) >= 2 and argv[0] == "admit":
        d = admit_detached(argv[1])
        print(("ADMITTED: " if d.admitted else "REFUSED: ") + d.reason)
        return 0 if d.admitted else 1
    if len(argv) >= 3 and argv[0] == "wrap":
        return _wrap(argv[1], argv[2:])
    print("usage: python -m experiments.cpu_budget "
          "{admit LABEL | wrap LABEL CMD [ARG...]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
