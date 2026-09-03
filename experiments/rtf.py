"""Real-time-factor gate — measure sim-seconds per real second, project a run's
wall-clock duration BEFORE it starts, and refuse what cannot finish (T0.32).

The scar (registry T0.32 notes): LC.07's pilot did this arithmetic BY HAND on
2026-09-01 — its cheapest class projected 14.49 h against an 8.5 h kernel rule
and branch B fired — and before that, the numbers measured 2026-08-09 (57M
trunk 0.17 sim-s/real-s, 160K MLP 23.0) mean a 3-seed 1-sim-hour spec costs
52 h with the former against run.py's 15 h cpu<2h ceiling. Every long run to
date either did this projection in its author's head or discovered the answer
by being killed at the timeout. This module makes it a standing refusal.

Scope, stated honestly:
  - `BUDGET_SECONDS` / `spec_child_timeout_seconds` are THE canonical timeout
    arithmetic. `run.py:_run_isolated` imports them (it used to carry a private
    copy), so the process that KILLS an overlong run and the gate that REFUSES
    one compute from one table and cannot drift apart.
  - The gate binds a caller that declares its control path: it cannot project a
    run whose sim-cost nobody states. Long-run specs (LF.01 is the registered
    one; its notes already call the real-time factor "a GATE, not a note")
    call `require_feasible` before stepping. This module cannot prove future
    callers call it — each long-run spec's own certificate carries that.
  - The shared-box CPU-hour QUOTA (how much of a day the ladder may take from
    the tenants) is T0.33's, not this gate's. `TENANT_WALL_CEILING_S` here is
    only the per-run absurdity bound: no single declared run may project past
    the largest budget class the harness itself will tolerate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

# Canonical per-experiment wall allowances by budget class. run.py imports this
# (do not re-copy it anywhere — T0.14's pasted-constant scar).
BUDGET_SECONDS = {
    "cpu<1min": 300, "cpu<10min": 1800, "cpu<2h": 9000, "cpu<48h": 172800,
    "gpu<20min": 3600, "gpu<2h": 10800, "gpu<8h": 36000,
}

# No single declared run may project past the largest class's child allowance.
TENANT_WALL_CEILING_S = BUDGET_SECONDS["cpu<48h"] * 2


def spec_child_timeout_seconds(spec) -> int:
    """run.py's child-kill arithmetic, single-sourced.

    The budget names one EXPERIMENT; a child runs seeds x (experiment +
    control), hence the seeds multiplier and the x2 slack (the T1.01/02/06
    mid-science kill, and T2.01's 66.7-min kernel killed at a flat 60).
    """
    base = BUDGET_SECONDS.get(spec.budget.value if spec else "", 3600)
    return base * max(1, getattr(spec, "seeds", 1)) * 2


@dataclass(frozen=True)
class RTFReading:
    """A measured real-time factor for one declared control path."""
    rtf: float             # sim-seconds advanced per real second
    rel_spread: float      # trial-to-trial relative spread of the rtf
    probe_sim_s: float     # sim-seconds actually advanced by one probe trial
    probe_real_s: float    # mean real seconds one probe trial took


@dataclass(frozen=True)
class Decision:
    admitted: bool
    projected_s: float     # projected wall seconds for the declared run
    limit_s: float         # the binding allowance it was compared against
    reason: str


class RTFRefusal(RuntimeError):
    """Raised by require_feasible; carries the Decision that refused."""

    def __init__(self, decision: Decision):
        super().__init__(decision.reason)
        self.decision = decision


def measure_rtf(step_fn, sim_dt_per_step: float, n_steps: int = 100,
                warmup: int = 20, trials: int = 3) -> RTFReading:
    """Measure sim-s/real-s for one step function that advances sim_dt per call.

    Warmup runs before EVERY trial, not just the first — T0.07 measured a
    heavier computation as *faster* than a lighter one until per-trial warmup
    stopped the timer catching page-in instead of the work.
    """
    if sim_dt_per_step <= 0:
        raise ValueError("sim_dt_per_step must be positive")
    reals = []
    for _ in range(trials):
        for _ in range(warmup):
            step_fn()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            step_fn()
        reals.append(time.perf_counter() - t0)
    mean_real = sum(reals) / len(reals)
    var = sum((r - mean_real) ** 2 for r in reals) / len(reals)
    spread = (var ** 0.5) / mean_real if mean_real > 0 else float("inf")
    sim_s = n_steps * sim_dt_per_step
    return RTFReading(rtf=sim_s / mean_real if mean_real > 0 else float("inf"),
                      rel_spread=spread, probe_sim_s=sim_s, probe_real_s=mean_real)


def project_real_seconds(reading: RTFReading, sim_seconds: float) -> float:
    """Projected wall seconds for a run that must advance `sim_seconds`."""
    if reading.rtf <= 0:
        return float("inf")
    return sim_seconds / reading.rtf


def gate_long_run(reading: RTFReading, sim_seconds: float, spec=None,
                  limit_s: float | None = None,
                  ceiling_s: float = TENANT_WALL_CEILING_S) -> Decision:
    """Admit or refuse a declared run BEFORE it starts.

    `sim_seconds` is the TOTAL sim time the run will execute (all seeds,
    experiment + control — everything inside the child the limit applies to).
    The binding limit is the tightest of: the spec's own child timeout (if a
    spec is given), an explicit `limit_s`, and the per-run ceiling.
    """
    limits = [ceiling_s]
    if spec is not None:
        limits.append(float(spec_child_timeout_seconds(spec)))
    if limit_s is not None:
        limits.append(float(limit_s))
    limit = min(limits)
    projected = project_real_seconds(reading, sim_seconds)
    if projected > limit:
        return Decision(False, projected, limit,
                        f"REFUSED: {sim_seconds:.0f} sim-s at rtf "
                        f"{reading.rtf:.3f} projects {projected:.0f}s wall "
                        f"> limit {limit:.0f}s")
    return Decision(True, projected, limit,
                    f"admitted: projects {projected:.0f}s wall "
                    f"<= limit {limit:.0f}s")


def require_feasible(reading: RTFReading, sim_seconds: float, spec=None,
                     limit_s: float | None = None,
                     ceiling_s: float = TENANT_WALL_CEILING_S) -> Decision:
    """gate_long_run, but a refusal RAISES — the form long-run callers use."""
    d = gate_long_run(reading, sim_seconds, spec=spec, limit_s=limit_s,
                      ceiling_s=ceiling_s)
    if not d.admitted:
        raise RTFRefusal(d)
    return d
