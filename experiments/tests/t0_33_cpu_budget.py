"""T0.33 — CPU-hours on a shared box are accounted like GPU-hours.

This box hosts paying tenants. The GPU quota has had an accountant since
T0.12; the box's own CPU had none — the null baseline is exact: only GPU
hours are tracked, and the loop checks instantaneous load ONCE, at iteration
start (`ladder_loop.sh MAX_LOAD`), so a day of back-to-back cpu<2h children
was invisible to every instrument while it happened. The claim: every CPU
child the runner spawns debits a daily wall-clock budget
(`experiments/cpu_budget.json`), and the runner REFUSES to start a child when
the day's accumulated share or the box's current load would harm the tenants.

Properties, each independently checkable (T0.12's rewritten form is the
template — every accounting assertion is made at a MID-RANGE value that
moves, never at an exhausted one, where the assertion holds under every
implementation including a broken one):

  1. A charge MOVES the reading and PERSISTS across instances — the
     falsified_by's second clause ("a budget that reads the same whether or
     not runs happened") verbatim.
  2. Day boundaries isolate, asserted at mid-range: yesterday's charge does
     not move today's remaining.
  3. Affordability: a run whose worst-case child exceeds the remaining day is
     REFUSED, not attempted; a small one is admitted.
  4. The gate binds the right lane: cpu children gated and charged; GPU
     children unbound (their wall is remote waiting, metered by T0.12);
     cpu<48h refused with a ROUTING reason (the detached lane, which this
     meter honestly does not see).
  5. NO FORECLOSURE: the largest legal child (cpu<2h x 3 seeds x 2 = 54000 s)
     is admitted from a fresh day, and no registered cpu spec's canonical
     estimate exceeds the full ceiling — a tenant gate must not silently
     foreclose a cost class by arithmetic (the ME.11.E disease).
  6. Load refuses independently of the budget, and the threshold equals
     `ladder_loop.sh`'s MAX_LOAD — one number, two languages, parsed not
     assumed.
  7. An overrun past the ceiling leaves a mark (the meter observes what the
     refusal could not prevent — T0.12 property 7).
  8. A stale writer cannot erase another process's charges: two instances
     loaded from the same file both charge; the file shows both (the
     2026-08-12 GPU clobber, one resource over).
  9. WIRING, scanned live on `run._run_isolated`'s source: the gate is called
     BEFORE the child spawns, the debit after (on the success AND the timeout
     path — a killed child occupied the box for its full window), and a
     refusal returns UNRECORDED (an ERROR row would supersede a real result
     with scheduling noise).
 10. SHIPPED-PATH refusal, end to end: `_run_isolated` itself, pointed at an
     exhausted temp budget via JACK_CPU_BUDGET, refuses a real cheap spec
     before spawning — no child, no ledger byte moves. (If the refusal is
     broken this deliberately runs T0.01 for real, a few seconds and an
     honest re-run recorded by the runner — disclosed, not hidden.)

Control (registry: "A leaky accountant must FAIL isolation, and the assertion
must be made at a MID-RANGE value"): `_LeakyBudget` collapses every day into
one bucket — the same disease as the retired ISO GPU week format, which
charged Sunday's runs to the exhausted week. Its isolation property must FAIL
(yesterday's charge visibly moves today's reading) while its arithmetic stays
alive (the charge itself lands), so the control cannot pass vacuously by
being broken everywhere.

VOID lane: a live 1-minute load above LOAD_CEILING is a co-tenant condition —
the shipped-path refusal would fire for load rather than budget and the
admission properties cannot be honestly asserted. Instrument fault, not
refutation.

The STILL OPEN paragraph that stood here is CLOSED by T0.34 (2026-09-04):
the detached lane (`launch_detached.sh`, cpu<48h) now admits against the
same day ledger and bills per heartbeat — this spec's cpu<48h property
still checks only the ROUTING (the runner must not pretend to meter a lane
it never sees); the routed-to lane's receipts are T0.34's claim. Modules
invoked BY HAND remain unmetered: a human at a shell is the owner's lane.

Uses temp budget files; must never touch the real accounting.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
import time
from pathlib import Path

from ..cpu_budget import (CPU_DAY_CEILING_S, ENV_OVERRIDE, LOAD_CEILING,
                          CpuBudget, _loadavg, gate_cpu_child)
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from ..rtf import spec_child_timeout_seconds

REPO = Path(__file__).resolve().parents[2]

# Every property here is a property of experiments/cpu_budget.py; run.py's
# wiring is scanned LIVE at run time (property 9), so a drift there is caught
# by the next run rather than by a stale flag.
IMPL_DEPS = ["experiments/cpu_budget.py"]

MID_CHARGE_S = 600.0          # the mid-range value the isolation assert uses
WORST_LEGAL_CHILD_S = 54000.0  # cpu<2h x 3 seeds x 2 — must fit a fresh day
GPU_WITNESS = "T2.01"          # gpu<8h — must be unbound by this meter
DETACHED_WITNESS = "PS.04"     # cpu<48h — must be refused with a routing reason
SHIPPED_WITNESS = "T0.01"      # cheap real spec offered to the shipped path
WORST_WITNESS = "XL.01"        # cpu<2h, 3 seeds: the largest legal child


def _yesterday() -> str:
    return time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))


class _LeakyBudget(CpuBudget):
    """The control: every day is one bucket, so isolation MUST fail."""

    def _key(self, day: str | None = None) -> str:
        return "ALL-DAYS"


def _experiment(seed: int) -> dict:
    live_load = _loadavg()
    day0 = time.strftime("%Y-%m-%d")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cpu_budget.json"
        b = CpuBudget(path)

        starts_full = b.remaining_s(day0) == CPU_DAY_CEILING_S
        b.charge("T0.33-probe", MID_CHARGE_S, day=day0)
        charged_moves = b.used_s(day0) == MID_CHARGE_S
        persists = CpuBudget(path).used_s(day0) == MID_CHARGE_S

        # Isolation at MID-RANGE: yesterday's charge, today's exact reading.
        b.charge("T0.33-probe", 7200.0, day=_yesterday())
        remaining_mid = CpuBudget(path).remaining_s(day0)
        isolation_ok = remaining_mid == CPU_DAY_CEILING_S - MID_CHARGE_S

        affords_small = b.afford(MID_CHARGE_S, day=day0)
        refuses_oversized = not b.afford(remaining_mid + 1.0, day=day0)

        # Stale writer: both instances loaded now; both charges must survive.
        b1, b2 = CpuBudget(path), CpuBudget(path)
        b1.charge("T0.33-w1", 100.0, day=day0)
        b2.charge("T0.33-w2", 200.0, day=day0)
        stale_writer_safe = (CpuBudget(path).used_s(day0)
                             == MID_CHARGE_S + 300.0)

        # Gates, load injected so a busy co-tenant cannot decide them.
        worst = BY_ID[WORST_WITNESS]
        fresh_path = Path(td) / "fresh.json"
        d = gate_cpu_child(worst, path=fresh_path, loadavg=0.0)
        fresh_admits_worst = d.admitted and d.est_s == WORST_LEGAL_CHILD_S

        CpuBudget(fresh_path).charge("T0.33-drain", CPU_DAY_CEILING_S)
        d = gate_cpu_child(worst, path=fresh_path, loadavg=0.0)
        exhausted_refused = (not d.admitted) and "day budget" in d.reason

        d = gate_cpu_child(worst, path=Path(td) / "fresh2.json",
                           loadavg=LOAD_CEILING + 1.0)
        load_refused = (not d.admitted) and "load" in d.reason

        d = gate_cpu_child(BY_ID[GPU_WITNESS], path=fresh_path, loadavg=0.0)
        gpu_unbound = d.admitted and "T0.12" in d.reason

        d = gate_cpu_child(BY_ID[DETACHED_WITNESS], path=fresh_path,
                           loadavg=0.0)
        detached_routed = (not d.admitted) and "detached" in d.reason

        # Overrun leaves a mark.
        over_path = Path(td) / "over.json"
        ob = CpuBudget(over_path)
        ob.charge("T0.33-over", CPU_DAY_CEILING_S + 1.0)
        marks = CpuBudget(over_path).data["overruns"]
        overrun_marked = len(marks) == 1 and marks[0]["spec_id"] == "T0.33-over"

        # No registered cpu spec is foreclosed by the ceiling's arithmetic.
        foreclosed = sorted(
            s.id for s in BY_ID.values()
            if s.budget.value.startswith("cpu") and s.budget.value != "cpu<48h"
            and spec_child_timeout_seconds(s) > CPU_DAY_CEILING_S)

        # One threshold, two languages.
        loop_src = (REPO / "scripts" / "ladder_loop.sh").read_text()
        m = re.search(r"^MAX_LOAD=([0-9.]+)", loop_src, re.M)
        loop_load_agrees = m is not None and float(m.group(1)) == LOAD_CEILING

        # Wiring, scanned on the live function.
        from .. import run as run_mod
        src = inspect.getsource(run_mod._run_isolated)
        spawn_at = src.find("sp.run([sys.executable")
        gate_at = src.find("gate_cpu_child(_spec)")
        bills = [i for i in range(len(src)) if src.startswith("_bill_cpu(_t0)", i)]
        wiring_ok = (0 <= gate_at < spawn_at
                     and "charge_cpu_child(spec_id" in src
                     and len(bills) == 2          # timeout path AND success path
                     and all(i > spawn_at for i in bills)
                     and "REFUSED before start" in src
                     and "ledger.record" not in src[gate_at:spawn_at])

        # Shipped path: the real runner refuses a real spec on an exhausted
        # day, before spawning, without a ledger byte moving.
        exhausted = Path(td) / "exhausted.json"
        CpuBudget(exhausted).charge("T0.33-drain", CPU_DAY_CEILING_S)
        ledger_path = REPO / "experiments" / "ledger.json"
        before = ledger_path.read_bytes()
        old_env = os.environ.get(ENV_OVERRIDE)
        os.environ[ENV_OVERRIDE] = str(exhausted)
        try:
            t0 = time.perf_counter()
            res = run_mod._run_isolated(SHIPPED_WITNESS, Ledger())
            shipped_s = time.perf_counter() - t0
        finally:
            if old_env is None:
                os.environ.pop(ENV_OVERRIDE, None)
            else:
                os.environ[ENV_OVERRIDE] = old_env
        shipped_refusal = (res.status is Status.ERROR
                           and "REFUSED before start" in res.message
                           and "day budget" in res.message
                           and ledger_path.read_bytes() == before
                           and shipped_s < 2.0)   # refused, not run-and-failed

    return {
        "cpu_quota_enforced": float(shipped_refusal and exhausted_refused),
        "live_load": round(live_load, 2),
        "starts_full": starts_full,
        "charged_moves": charged_moves,
        "persists": persists,
        "remaining_mid_s": remaining_mid,
        "isolation_ok": isolation_ok,
        "affords_small": affords_small,
        "refuses_oversized": refuses_oversized,
        "stale_writer_safe": stale_writer_safe,
        "fresh_admits_worst": fresh_admits_worst,
        "exhausted_refused": exhausted_refused,
        "load_refused": load_refused,
        "gpu_unbound": gpu_unbound,
        "detached_routed": detached_routed,
        "overrun_marked": overrun_marked,
        "cpu_foreclosed": foreclosed,
        "loop_load_agrees": loop_load_agrees,
        "wiring_ok": wiring_ok,
        "shipped_refusal": shipped_refusal,
        "shipped_refusal_s": round(shipped_s, 3),
    }


def _control(seed: int) -> dict:
    """The leaky accountant. It must FAIL isolation specifically — and be
    demonstrably alive while failing, so a broken-everywhere control cannot
    pass by never charging at all."""
    day0 = time.strftime("%Y-%m-%d")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "leaky.json"
        lb = _LeakyBudget(path)
        lb.charge("T0.33-leak", MID_CHARGE_S, day=_yesterday())
        today_reading = _LeakyBudget(path).used_s(day0)
    return {
        "alive": today_reading > 0.0,                 # the charge landed
        "leak_visible": today_reading == MID_CHARGE_S,  # ...in TODAY's bucket
    }


def _check(m: dict, c: dict):
    # Co-tenant load spike: admissions cannot be asserted and the shipped
    # refusal cannot be attributed to the budget. Instrument fault.
    if m["live_load"] > LOAD_CEILING:
        return Status.VOID
    return (m["starts_full"] is True
            and m["charged_moves"] is True
            and m["persists"] is True
            and m["isolation_ok"] is True
            and m["remaining_mid_s"] == CPU_DAY_CEILING_S - MID_CHARGE_S
            and m["affords_small"] is True
            and m["refuses_oversized"] is True
            and m["stale_writer_safe"] is True
            and m["fresh_admits_worst"] is True
            and m["exhausted_refused"] is True
            and m["load_refused"] is True
            and m["gpu_unbound"] is True
            and m["detached_routed"] is True
            and m["overrun_marked"] is True
            and m["cpu_foreclosed"] == []
            and m["loop_load_agrees"] is True
            and m["wiring_ok"] is True
            and m["shipped_refusal"] is True
            and c["alive"] is True
            and c["leak_visible"] is True)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.33"], _experiment, _check, ledger=ledger,
                    control_fn=_control)
