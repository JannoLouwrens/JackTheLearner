"""CPU-hour accounting for the runner's children on a shared box (T0.33).

This box hosts paying tenants; SYSTEM.md ranks their safety above the ladder.
Until T0.33 the only protections were the loop's single load reading at
iteration start (`ladder_loop.sh MAX_LOAD`) and the per-child kill timeout —
nothing accumulated, so a day of back-to-back cpu<2h children was invisible to
every instrument while it happened. This module is the GPU budget's CPU
sibling: a plain JSON file a person can read, a debit for every runner child,
and a refusal BEFORE a child spawns.

Scope, stated honestly:
  - The metered unit is WALL SECONDS of the child, not core-seconds: a child
    permitted 2 threads by `ladder_loop.sh:253` can consume up to 2
    core-seconds per billed second — measured 1.68x/1.72x on two live probes
    on 2026-09-07, and 1.241x on an LC.03-class arm-seed the same day. A
    planner writing core-hours must divide by the workload's thread width
    before comparing against `CPU_DAY_CEILING_S`.
  - The metered child is `run.py:_run_isolated`'s — the only lane
    `cmd_run` (and therefore `--gate`) uses. The gate refuses `cpu<48h`
    outright: D20/D39 CLOSED that class to registered spec work in BOTH
    lanes. The detached lane (`admit_detached` + the heartbeat wrapper below,
    `scripts/launch_detached.sh`) meters lab and waiter jobs, not specs. A
    module invoked BY HAND is unmetered — a human at a shell is the owner's.
  - GPU-budget children are not charged: their wall clock is mostly waiting on
    a remote kernel, and billing waiting as box CPU would make the meter read
    harm where there is none. Their remote cost is T0.12's.
  - `afford()` gates on `child_estimate_s` — the spec's own MEASURED last
    child duration where the ledger has one, the enum worst case where it does
    not — and `charge()` bills the wall clock actually spent; like the GPU
    meter, an overrun past the ceiling is OBSERVED with a mark, while the
    refusal stops new work from starting.

Estimates come from the spec's own measured duration (`child_estimate_s`
returns `min(enum, SAFETY x measured + overhead)`, so it can only LOWER an
estimate); the enum worst case foreclosed 53 of 152 specs (69th audit B4).

Accounting ownership (68th audit B1): charges must be DISJOINT, and the
OUTERMOST `_wrap` owns the tree's: it exports JACK_CPU_WRAPPED, under which
`charge_cpu_child` and nested wraps bill nothing (before: 1.7x overcharge,
2026-09-04). It bills wall per heartbeat, or CPU when the tree idles.
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
# day. 16 h ADMITS the largest legal child from a fresh day (cpu<2h x 3
# seeds x 2 = 54000 s; T0.33 gates `cpu_foreclosed == []`). It counts WALL,
# so a multi-worker job costs (workers x billed) core-seconds — LC.08 ran
# 51,841 core-s against 16,912 billed (161st audit, 2026-10-25); the load
# gate and mem watchdog are the instantaneous guard. Core-seconds is D39,
# the owner's, and this comment moves no constant.
CPU_DAY_CEILING_S = 57600.0

# Must equal `ladder_loop.sh`'s MAX_LOAD — one threshold, two languages;
# T0.33 parses the shell file and fails if they drift apart.
LOAD_CEILING = 6.0

# ── The admission estimate (69th audit B4) ──────────────────────────────────
# Two per-spec records of what a child cost: the ledger's `duration_s`
# (run_spec's interior only) and `last_child_s`, the wall the runner last
# BILLED for it (`charge_cpu_child`). The gate predicts what it bills, so the
# live estimate reads the larger. The ledger alone undercounted by 3,500x on
# 2026-11-14: W1.02 recorded 0.22 s and billed 762 s, W0.DIAG 0.03 s vs 1,609 s
# (pools built in run() before run_spec's timer, and the hash-salt child).
LEDGER_FILE = Path(__file__).parent / "ledger.json"

# Engineering constants, not measurements: CHILD_OVERHEAD_S is interpreter
# start + imports (0.55-5.38 s observed 09-03/04); PROJECTION_SAFETY covers
# run-to-run variance, which is unmeasured. Both are bounded by the enum clamp.
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
               day: str | None = None, child: bool = False) -> None:
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
            if child:
                self.data.setdefault("last_child_s", {})[spec_id] = round(seconds, 2)
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
    """The spec's last measured cost, or None (-> the enum, the RESTRICTIVE
    side, on any read problem). The live reading (no `ledger_path`) is the
    larger of the ledger's duration and the wall last billed for the child;
    an explicit ledger path is read alone, so T0.33's fixtures stay sealed."""
    d = _durations(ledger_path).get(spec_id) or 0.0
    if ledger_path is None:
        try:
            p = _budget_path()
            key = ("billed", str(p), p.stat().st_mtime_ns)
            if key not in _DURATION_CACHE:
                _DURATION_CACHE[key] = CpuBudget(p).data.get("last_child_s", {})
            d = max(d, float(_DURATION_CACHE[key].get(spec_id, 0.0)))
        except Exception:
            pass
    return d if d > 0.0 else None


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
                           "cpu<48h is CLOSED to registered specs in the "
                           "runner AND the detached lane (D20, D39); "
                           "re-register in cpu<2h or run off-box (S1)")
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
    CpuBudget(path).charge(spec_id, seconds, child=True)


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


# Ruling 3 (2026-11-14): detached jobs share at most this much of the day,
# and at most this many run at once. Before it, detached jobs overran the
# whole day twice (59,599 s on 11-13) and starved registered runs.
DETACHED_DAY_CAP_S = 34560.0
DETACHED_MAX_ALIVE = 2


def detached_used_s(path: Path | None = None) -> float:
    by = CpuBudget(path).data["days"].get(_day(), {}).get("by_spec", {})
    return sum(v for k, v in by.items() if k.startswith("detached:"))


# Ruling 5 (2026-11-15): a job whose session has averaged under
# IDLE_CORES_BILL cores since launch is a POLLER (a Kaggle watcher) — it does
# not hold one of the DETACHED_MAX_ALIVE seats, but at most this many may
# wait at once. Younger than POLLER_MIN_AGE_S it counts busy (safe side).
DETACHED_MAX_POLLERS = 6
POLLER_MIN_AGE_S = 60.0


def is_poller(cpu_s: float, age_s: float) -> bool:
    return age_s >= POLLER_MIN_AGE_S and cpu_s < IDLE_CORES_BILL * age_s


def detached_jobs() -> tuple:
    """(busy, pollers) over live `cpu_budget wrap` processes, one per
    detached job; each is classed by its setsid session's lifetime CPU."""
    tick = os.sysconf("SC_CLK_TCK")
    uptime = float(open("/proc/uptime").read().split()[0])
    wraps, sess = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            raw = open(f"/proc/{d}/stat").read()
            f = raw[raw.rindex(")") + 2:].split()
            sid = int(f[3])
            sess[sid] = sess.get(sid, 0) + sum(int(x) for x in f[11:15])
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
            if b"experiments.cpu_budget" in argv and b"wrap" in argv:
                wraps[int(d)] = (sid, uptime - int(f[19]) / tick)
        except (OSError, ValueError, IndexError):
            continue
    pollers = sum(is_poller(sess.get(sid, 0) / tick, age)
                  for sid, age in wraps.values())
    return len(wraps) - pollers, pollers


def admit_detached(label: str, *, path: Path | None = None,
                   loadavg: float | None = None, est_s: float = 0.0,
                   alive: int | None = None,
                   pollers: int | None = None) -> CpuDecision:
    """Admit or refuse a detached launch BEFORE setsid. The gate asks whether
    TODAY has headroom, the detached share (`DETACHED_DAY_CAP_S`, plus the
    caller's optional estimate) and the concurrency cap have room, and the
    box is calm; the ceiling binds a running child through `charge`'s
    overrun marks, never through a kill."""
    load = _loadavg() if loadavg is None else loadavg
    if alive is None:
        alive, live_pollers = detached_jobs()
        pollers = live_pollers if pollers is None else pollers
    pollers = pollers or 0
    if alive >= DETACHED_MAX_ALIVE or pollers >= DETACHED_MAX_POLLERS:
        return CpuDecision(False, est_s, 0.0, load,
                           f"detached cap: {alive} busy detached jobs alive "
                           f"(max {DETACHED_MAX_ALIVE}, ruling 3), {pollers} "
                           f"pollers (max {DETACHED_MAX_POLLERS}, ruling 5)")
    det = detached_used_s(path)
    if label.startswith("detached:") and det + est_s > DETACHED_DAY_CAP_S:
        return CpuDecision(False, est_s, 0.0, load,
                           f"detached cap: {det:.0f}s + est {est_s:.0f}s > "
                           f"{DETACHED_DAY_CAP_S:.0f}s today (ruling 3)")
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


# Under this many cores over a heartbeat the tree is WAITING (a Kaggle-CPU
# poller: 5 s CPU billed 6,600 s wall, day foreclosed by 15:00 on 11-14):
# bill its CPU. Busier intervals bill wall as before (D39 untouched).
IDLE_CORES_BILL = 0.10


def _session_ticks() -> dict:
    """{pid: utime+stime+cutime+cstime} for every process in this session.
    `launch_detached.sh` setsids the wrapper, so the session is the job; a
    process that exits is reaped into its parent's c-times, still counted."""
    sid, out = os.getsid(0), {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            raw = open(f"/proc/{d}/stat").read()
            f = raw[raw.rindex(")") + 2:].split()
            if int(f[3]) == sid:
                out[int(d)] = sum(int(x) for x in f[11:15])
        except (OSError, ValueError, IndexError):
            continue
    return out


def _cpu_delta_s(prev: dict, cur: dict) -> float:
    """CPU seconds between two samples; an unseen pid counts whole, a vanished
    one's tail is lost but its reaper's c-times jump (the safe direction)."""
    tick = os.sysconf("SC_CLK_TCK")
    return sum(max(0, v - prev.get(p, 0)) for p, v in cur.items()) / tick


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
    last, ticks = time.time(), _session_ticks()
    while True:
        try:
            proc.wait(timeout=_heartbeat_s())
            done = True
        except subprocess.TimeoutExpired:
            done = False
        now = time.time()
        if owner and now > last:
            cur = _session_ticks()
            cpu = _cpu_delta_s(ticks, cur)
            if cpu < IDLE_CORES_BILL * (now - last):
                bill_interval(label, now - cpu, now)
            else:
                bill_interval(label, last, now)
            last, ticks = now, cur
        if done:
            return int(proc.returncode)


def _main(argv: list) -> int:
    import sys
    if len(argv) >= 2 and argv[0] == "admit":
        d = admit_detached(argv[1], est_s=float(argv[2]) if len(argv) > 2
                           else 0.0)
        print(("ADMITTED: " if d.admitted else "REFUSED: ") + d.reason)
        return 0 if d.admitted else 1
    if len(argv) >= 3 and argv[0] == "wrap":
        return _wrap(argv[1], argv[2:])
    print("usage: python -m experiments.cpu_budget "
          "{admit LABEL [EST_S] | wrap LABEL CMD [ARG...]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
