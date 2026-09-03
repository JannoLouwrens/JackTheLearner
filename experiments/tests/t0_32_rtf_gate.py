"""T0.32 — the real-time factor is measured, recorded, and gates long runs.

The claim: for a declared control path, the harness measures sim-seconds per
real second BEFORE a long run starts, projects the run's wall-clock duration,
and REFUSES a run whose projection exceeds its spec's timeout or the per-run
ceiling — and the projection is honest, agreeing with the achieved duration to
within 25% (the registry's own bar; a projection looser than that would have
admitted LC.07's 14.49 h run against its 8.5 h rule anyway).

The scar this generalises: LC.07's pilot (2026-09-01) did this arithmetic by
hand and fired its checkpoint branch; the 2026-08-09 measurements (57M trunk
0.17 sim-s/real-s vs 160K MLP 23.0) mean a 3-seed 1-sim-hour life costs 52 h
with the former against run.py's 15 h cpu<2h child kill. Until this spec, the
only enforcement was being killed AT the timeout — science discarded after it
was paid for, the exact waste the projection exists to prevent.

What is exercised, on the real declared path (the playground world stepping
through `playground.step`, the only kernel lives use):

  1. MEASURE on a settled world (the probe runs after the humanoid has
     collapsed to rest, because that is the regime a long run spends its hours
     in), with per-trial warmup and a spread gate — T0.07's timing discipline.
  2. PROJECT a verification run several probe-windows long, RUN it, and
     compare: |projected - achieved| / achieved <= 0.25.
  3. GATE: at the measured rtf, a declared run that fits its allowance is
     ADMITTED; the same allowance with 2x the sim-work is REFUSED; and a run
     projected past `TENANT_WALL_CEILING_S` is REFUSED with no spec at all.
  4. SINGLE SOURCE: run.py's child-kill timeout now imports
     `rtf.spec_child_timeout_seconds` and carries no private table, so the
     killer and the gate cannot drift apart; the arithmetic is verified on
     LF.01 (cpu<2h, 3 seeds -> 54000 s). The negative half no longer greps
     only for the literal `_budget_seconds` (66th audit B2: a private table
     under any other name passed it) — a table needs QUOTED budget-class
     keys, so run.py must contain no quoted `cpu<`/`gpu<` literal at all;
     prose mentions in docstrings and comments are unquoted and still pass.
  5. BINDING (66th audit B2): every IMPLEMENTED spec at a long-run budget —
     wall allowance >= cpu<2h's, by ALLOWANCE rather than enum order, which
     would drag gpu<20min's 3600 s in — must name `rtf.require_feasible` in
     its impl source, unless grandfathered in RTF_GRANDFATHERED. The audit
     called this conjunct "vacuously true today — LF.01 is the only such
     spec"; that premise was WRONG by 34 files (34 long-budget specs were
     implemented before this gate existed, none calling it), and firing on
     history the gate never bound would buy a red ordering 34 mechanical
     edits and certificate re-buys overnight. So the exemption set freezes
     exactly those 34 ids (2026-09-03) and is SHRINK-ONLY, self-policing in
     both directions: an id NOT in the set whose long-run impl lacks the
     call fails the check, and an id IN the set whose impl now calls the
     gate — or no longer exists — is a STALE EXEMPTION and also fails, so
     every adoption must prune the set (and re-buy this certificate), and
     nothing may ever be added. A source scan proves the call EXISTS, not
     that it is reached with honest arguments — that half stays with each
     long-run spec's own certificate (LF.01's registry notes already bind
     it: "the real-time factor is therefore a GATE, not a note").

Control (registry: "a deliberately slow policy must be REFUSED; a gate that
never refuses is decorative — T0.13's own rule"): the same world stepped with a
sleep injected per decision, sized from the honest per-decision cost so it is
slow by construction, offered to the gate with the SAME declared run the honest
path would have admitted. The gate must refuse it. Control aliveness is gated
too: if the "slow" policy is not measurably slower than the honest one, the
control exercised nothing and the test measures nothing (the at-chance-control
lesson).

VOID lane: a probe whose trial-to-trial spread exceeds MAX_PROBE_SPREAD cannot
support a 25% projection claim in either direction — that is an instrument
fault (a co-tenant load spike), not a refutation.

Recorded but NOT gated (a bar on box speed would measure the box, T0.07's
rule): what LF.01's full child (3 seeds x (life + control) x 1 sim-hour)
projects to at the measured rtf, and whether the gate would admit it today.

Scope, honestly: this proves the gate, the shared arithmetic, and (point 5)
that no post-gate long-run implementation ships without at least naming the
gate. What it still cannot prove is that the named call runs before the
stepping starts, or with the run's true sim-cost — each long-run spec's own
certificate carries that.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..protocol import Ledger, Status, module_path_for, run_spec
from ..registry import BY_ID
from ..rtf import (BUDGET_SECONDS, TENANT_WALL_CEILING_S, gate_long_run,
                   measure_rtf, spec_child_timeout_seconds)

REPO = Path(__file__).resolve().parents[2]

IMPL_DEPS = ["experiments/rtf.py", "playground.py"]

# --- pre-registered constants -------------------------------------------------
PROJ_TOL = 0.25            # the registry's falsified_by bar, verbatim
MAX_PROBE_SPREAD = 0.25    # T0.07's repeat-spread discipline; noisier is VOID
SETTLE_DECISIONS = 400     # ~6 sim-s: past the zero-ctrl collapse transient
PROBE_STEPS = 120
PROBE_WARMUP = 30
PROBE_TRIALS = 3
VERIFY_MULT = 6            # verification run is 6 probe-windows long
LIMIT_S = 60.0             # declared allowance for the gate exercise
FIT_FRAC = 0.5             # fitting run projects to half the allowance
OVER_FRAC = 2.0            # overflowing run projects to double the allowance
SLOW_SLEEP_MULT = 10.0     # control sleeps 10x the honest per-decision cost
MIN_SLOWDOWN = 5.0         # control aliveness: must actually be slow
FRAME_SKIP = 5
LF01_SIM_S = 3600.0 * 3 * 2   # LF.01's child: 3 seeds x (life + control) x 1 sim-h

# --- long-run binding (66th audit B2) -----------------------------------------
# A budget binds when its wall allowance reaches cpu<2h's. Allowance, not enum
# order: the enum fronts CPU work for scheduling, and by ITS order gpu<20min
# (3600 s) would sit past cpu<2h (9000 s).
LONGRUN_FLOOR_S = BUDGET_SECONDS["cpu<2h"]

# The 34 long-budget specs implemented BEFORE this conjunct existed, frozen
# 2026-09-03. SHRINK-ONLY: remove an id when its impl adopts require_feasible
# or is deleted (a stale exemption FAILS the check until pruned); never add
# one — a new long-run impl that cannot afford the call is exactly what the
# conjunct refuses. Docstring point 5 records why history is exempt at all.
RTF_GRANDFATHERED = frozenset({
    "BA.02", "BA.03", "D1.0", "DP.05", "LC.03", "LC.07", "LG.02", "LT.01",
    "ME.5", "ME.11.D", "ME.11.F", "PG.4", "PG.6", "SH.01", "SH.02", "SM.02",
    "T1.07", "T1.08", "T2.01", "T2.02", "T2.05", "T2.09", "T2.11", "T2.14",
    "T2.20", "T3.01", "T3.06", "T3.09", "TA.02", "UB.9", "UB.10", "UB.14",
    "VO.02", "XL.01",
})


def _longrun_binding() -> dict:
    """Source-scan every implemented long-run spec for the gate call.

    strict=True so a duplicated implementation raises (ERROR) rather than
    silently reading as unimplemented and slipping the scan.
    """
    unbound, stale = [], []
    for spec in BY_ID.values():
        if BUDGET_SECONDS.get(spec.budget.value, 0) < LONGRUN_FLOOR_S:
            continue
        path = module_path_for(spec.id, strict=True)
        calls = path is not None and "require_feasible" in path.read_text()
        if path is None or calls:
            if spec.id in RTF_GRANDFATHERED:
                stale.append(spec.id)
            continue
        if spec.id not in RTF_GRANDFATHERED:
            unbound.append(spec.id)
    return {"longrun_unbound": sorted(unbound),
            "longrun_stale_exemptions": sorted(stale),
            "longrun_grandfathered_n": len(RTF_GRANDFATHERED)}


def _make_stepper(sleep_per_decision: float = 0.0):
    """The declared control path: the playground's only stepping kernel."""
    os.environ.setdefault("MUJOCO_GL", "disabled")
    sys.path.insert(0, str(REPO))
    from playground import make_playground, step
    model, data, water = make_playground(with_humanoid=True)
    sim_dt = model.opt.timestep * FRAME_SKIP

    def one_decision():
        step(model, data, ctrl=None, frame_skip=FRAME_SKIP, water=water)
        if sleep_per_decision:
            time.sleep(sleep_per_decision)

    for _ in range(SETTLE_DECISIONS):
        step(model, data, ctrl=None, frame_skip=FRAME_SKIP, water=water)
    return one_decision, sim_dt


def _experiment(seed: int) -> dict:
    one_decision, sim_dt = _make_stepper()

    reading = measure_rtf(one_decision, sim_dt, n_steps=PROBE_STEPS,
                          warmup=PROBE_WARMUP, trials=PROBE_TRIALS)

    # PROJECT, then RUN, then compare — the projection must survive contact
    # with the duration it predicted.
    verify_steps = PROBE_STEPS * VERIFY_MULT
    projected = (verify_steps * sim_dt) / reading.rtf
    t0 = time.perf_counter()
    for _ in range(verify_steps):
        one_decision()
    achieved = time.perf_counter() - t0
    proj_err = abs(projected - achieved) / achieved if achieved > 0 else float("inf")

    # GATE at the measured rtf: fit admitted, overflow refused, ceiling binds.
    sim_fit = FIT_FRAC * LIMIT_S * reading.rtf
    sim_over = OVER_FRAC * LIMIT_S * reading.rtf
    fit = gate_long_run(reading, sim_fit, limit_s=LIMIT_S)
    over = gate_long_run(reading, sim_over, limit_s=LIMIT_S)
    ceiling = gate_long_run(reading, 1.5 * TENANT_WALL_CEILING_S * reading.rtf)

    # SINGLE SOURCE: the killer imports the gate's arithmetic; no private copy.
    # A private table under ANY name needs quoted budget-class keys, so the
    # negative half bans the quoted literals, not one table's name (B2).
    run_src = (REPO / "experiments" / "run.py").read_text()
    single_source_ok = ("from .rtf import spec_child_timeout_seconds" in run_src
                        and "_budget_seconds" not in run_src
                        and '"cpu<' not in run_src and "'cpu<" not in run_src
                        and '"gpu<' not in run_src and "'gpu<" not in run_src)
    lf01 = BY_ID["LF.01"]
    timeout_arithmetic_ok = (spec_child_timeout_seconds(lf01)
                             == BUDGET_SECONDS["cpu<2h"] * 3 * 2 == 54000)

    # Recorded, not gated: what the one registered life spec costs here today.
    lf01_gate = gate_long_run(reading, LF01_SIM_S, spec=lf01)

    binding = _longrun_binding()

    return {
        **binding,
        "rtf": round(reading.rtf, 3),
        "probe_rel_spread": round(reading.rel_spread, 4),
        "sim_dt_per_decision": round(sim_dt, 5),
        "projected_s": round(projected, 3),
        "achieved_s": round(achieved, 3),
        "rtf_projection_error": round(proj_err, 4),
        "fit_admitted": fit.admitted,
        "over_admitted": over.admitted,
        "ceiling_admitted": ceiling.admitted,
        "single_source_ok": single_source_ok,
        "timeout_arithmetic_ok": timeout_arithmetic_ok,
        "lf01_projected_h": round(lf01_gate.projected_s / 3600.0, 2),
        "lf01_limit_h": round(lf01_gate.limit_s / 3600.0, 2),
        "lf01_admitted_today": lf01_gate.admitted,
    }


def _control(seed: int) -> dict:
    """The deliberately slow policy, offered the run the honest path fit into.

    Self-contained: it measures its own honest baseline (short probe), sizes
    the sleep from the measured per-decision cost, and declares the SAME
    sim-work that projects to half the allowance at the HONEST rtf. If the
    gate admits this, it is decorative and the test measures nothing.
    """
    honest_step, sim_dt = _make_stepper()
    honest = measure_rtf(honest_step, sim_dt, n_steps=60, warmup=15, trials=2)

    per_decision_real = honest.probe_real_s / 60.0
    slow_step, _ = _make_stepper(
        sleep_per_decision=SLOW_SLEEP_MULT * per_decision_real)
    slow = measure_rtf(slow_step, sim_dt, n_steps=30, warmup=5, trials=2)

    sim_fit = FIT_FRAC * LIMIT_S * honest.rtf     # admitted for the honest path
    decision = gate_long_run(slow, sim_fit, limit_s=LIMIT_S)
    return {
        "honest_rtf": round(honest.rtf, 3),
        "slow_rtf": round(slow.rtf, 3),
        "slowdown": round(honest.rtf / slow.rtf if slow.rtf > 0 else float("inf"), 2),
        "slow_admitted": decision.admitted,
        "slow_projected_s": round(decision.projected_s, 1),
    }


def _check(m: dict, c: dict):
    # Instrument fault, not refutation: a probe too noisy to support a 25%
    # projection claim in either direction (co-tenant load spike).
    if m["probe_rel_spread"] >= MAX_PROBE_SPREAD:
        return Status.VOID
    return (m["rtf"] > 0
            and m["rtf_projection_error"] <= PROJ_TOL
            and m["fit_admitted"] is True
            and m["over_admitted"] is False
            and m["ceiling_admitted"] is False
            and m["single_source_ok"] is True
            and m["timeout_arithmetic_ok"] is True
            and m["longrun_unbound"] == []
            and m["longrun_stale_exemptions"] == []
            and c["slow_admitted"] is False
            and c["slowdown"] >= MIN_SLOWDOWN)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.32"], _experiment, _check, ledger=ledger,
                    control_fn=_control)
