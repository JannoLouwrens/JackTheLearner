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
  - `afford()` gates on the canonical worst case (`spec_child_timeout_seconds`
    — the same arithmetic that kills the child), and `charge()` bills the wall
    clock actually spent; like the GPU meter, an overrun past the ceiling is
    OBSERVED with a mark, while the refusal stops new work from starting.
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


def gate_cpu_child(spec, *, path: Path | None = None,
                   loadavg: float | None = None) -> CpuDecision:
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
    est = float(spec_child_timeout_seconds(spec))
    if load > LOAD_CEILING:
        return CpuDecision(False, est, CpuBudget(path).remaining_s(), load,
                           f"load {load:.2f} above {LOAD_CEILING} — leaving "
                           f"the box to the tenants")
    remaining = CpuBudget(path).remaining_s()
    if est > remaining:
        return CpuDecision(False, est, remaining, load,
                           f"day budget: worst-case child {est:.0f}s exceeds "
                           f"remaining {remaining:.0f}s of "
                           f"{CPU_DAY_CEILING_S:.0f}s")
    return CpuDecision(True, est, remaining, load, "within the day budget")


def charge_cpu_child(spec_id: str, seconds: float,
                     path: Path | None = None) -> None:
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
    takes the payload with it; a SIGKILL must target the process group."""
    import signal
    import subprocess
    import sys
    try:
        proc = subprocess.Popen(argv)
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
        if now > last:
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
