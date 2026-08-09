"""The bakeoff: how this project decides things.

The owner's requirement, 2026-08-09: "I want a system where different stuff are
tested and best ones chosen." This module is that, as infrastructure rather
than as a habit.

A bakeoff runs N candidate implementations (ARMS) head-to-head on one
pre-registered metric, multi-seed, against a shared null. The winner is chosen
by arithmetic, not by argument, and the rule is fixed before any number exists.

THREE PROPERTIES THAT MAKE IT HONEST, each learned the hard way here:

  1. THE LEARNING GATE (invented by spec T2.02, generalised here).
     T2.02 compared a 57M transformer against a 125K MLP on locomotion. The
     MLP cleared random by 7.11 sigma; the transformer managed 2.46. The spec
     could have declared the MLP the winner. It declared itself VOID instead,
     with the verdict "two non-learners cannot arbitrate the architecture" —
     because an arm that has not demonstrably learned tells you nothing about
     its architecture, only about that run. A bakeoff where any arm fails the
     gate returns VOID and blocks the decision until the arms actually work.
     Without this, a bakeoff is a machine for converting broken runs into
     confident architectural conclusions.

  2. A MARGIN, NOT A MAXIMUM. Picking argmax over noisy seeds picks noise.
     The winner must beat the runner-up by `margin_sigma` of the pooled seed
     spread, or the result is a TIE — which is real information: it means the
     choice does not matter yet and the cheaper arm should be preferred.

  3. THE DECISION IS WRITTEN DOWN, INCLUDING THE LOSERS. A bakeoff that
     records only its winner cannot be re-opened when evidence changes, and
     the deleted alternatives get silently reinvented six weeks later.

WHAT A BAKEOFF MAY NOT DO: it may not choose its own metric after seeing the
data, drop an arm that embarrasses it, or re-run until an arm wins. The metric,
the arms, the gate and the margin live in the Spec, which is committed before
the run.
"""
from __future__ import annotations

import statistics as st
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .protocol import Ledger, Result, Spec, Status

DECISIONS = Path(__file__).parent.parent / "docs" / "DECISIONS_RESOLVED.md"


@dataclass
class Arm:
    """One candidate implementation.

    `run(seed) -> float` returns the metric for that seed. Higher is better;
    pass `higher_is_better=False` on the bakeoff to flip it. `cost` is a
    tie-breaker (params, latency, GPU-hours — whatever the spec declared), so
    a TIE resolves toward the cheaper option rather than the flashier one.
    """
    name: str
    run: Callable[[int], float]
    description: str = ""
    cost: Optional[float] = None
    """UNDECLARED by default, not zero. A TIE is resolved by cost, so a default
    of 0.0 let an arm that never declared one win by appearing free — an
    arbitrary pick reported as a measurement. None forces the spec to say what
    the arms cost, in the units it named."""


@dataclass
class ArmResult:
    name: str
    scores: List[float]
    mean: float
    std: float
    sigma_over_null: float
    passed_gate: bool
    cost: Optional[float] = None
    description: str = ""


@dataclass
class BakeoffResult:
    spec_id: str
    verdict: str                      # WINNER | TIE | VOID
    winner: Optional[str]
    arms: List[ArmResult]
    null_mean: float
    null_std: float
    reason: str
    metric: str = ""

    def to_metrics(self) -> Dict[str, float | str]:
        m: Dict[str, float | str] = {
            "verdict": self.verdict,
            "winner": self.winner or "none",
            "reason": self.reason,
            "null_mean": round(self.null_mean, 4),
            "null_std": round(self.null_std, 4),
        }
        for a in self.arms:
            m[f"{a.name}_mean"] = round(a.mean, 4)
            m[f"{a.name}_sigma"] = round(a.sigma_over_null, 3)
            m[f"{a.name}_gate"] = float(a.passed_gate)
            if a.cost is not None:
                m[f"{a.name}_cost"] = a.cost
        return m


def run_bakeoff(spec: Spec,
                arms: List[Arm],
                null_run: Callable[[int], float],
                seeds: Optional[List[int]] = None,
                learning_gate_sigma: float = 3.0,
                margin_sigma: float = 1.5,
                higher_is_better: bool = True,
                controls: Optional[List[Arm]] = None,
                ledger: Optional[Ledger] = None,
                decisions_path: Optional[Path] = None) -> BakeoffResult:
    """Run every arm on every seed, gate them, and pick a winner or refuse to.

    `arms` COMPETE and must clear the learning gate. `controls` are expected to
    FAIL it, and are scored without being allowed to VOID the run.

    The distinction was forced by the curiosity bakeoff, and it is not a
    convenience. That design needs ICM and RND present as arms that MUST be
    beaten — noisy-TV fixation is the whole point of including them. But every
    arm must clear the gate or the run is VOID, so entering a designed-to-fail
    control as an arm would VOID that bakeoff permanently, by construction. The
    generalisation: any bakeoff with a control has this problem, and the fix is
    that a control is a different KIND of thing, not a weak arm.

    A control that CLEARS the gate inverts the verdict to VOID — same logic as
    run_spec's controls. If the thing that was supposed to fail succeeds, the
    metric is not measuring what the spec claims, and no comparison built on it
    can be trusted.

    Records to the ledger as PASS only when a decision was actually reached:
    a VOID bakeoff is not a passing spec, because the question is still open.
    """
    seeds = seeds or list(range(max(spec.seeds, 3)))
    controls = controls or []
    if len(arms) < 2:
        raise ValueError("a bakeoff needs at least two arms; one arm is just a test")
    clash = {a.name for a in arms} & {c.name for c in controls}
    if clash:
        raise ValueError(f"{clash} declared as both arm and control")

    null_scores = [float(null_run(s)) for s in seeds]
    null_mean = st.mean(null_scores)
    null_std = st.stdev(null_scores) if len(null_scores) > 1 else 0.0

    results: List[ArmResult] = []
    for arm in arms:
        scores = [float(arm.run(s)) for s in seeds]
        mean = st.mean(scores)
        std = st.stdev(scores) if len(scores) > 1 else 0.0
        # Sigma against the LARGER of the two noise sources. Using only the
        # null's spread flatters an arm whose own seeds disagree wildly.
        sigma_unit = max(std, null_std, 1e-9)
        delta = (mean - null_mean) if higher_is_better else (null_mean - mean)
        sigma = delta / sigma_unit
        results.append(ArmResult(arm.name, scores, mean, std, sigma,
                                 sigma >= learning_gate_sigma, arm.cost,
                                 arm.description))

    # Controls are scored on the same ruler but never compete. One that
    # CLEARS the gate inverts the verdict: the metric is not measuring what
    # the spec claims, so nothing built on it can be trusted.
    control_results: List[ArmResult] = []
    for c in controls:
        scores = [float(c.run(s)) for s in seeds]
        mean = st.mean(scores)
        std = st.stdev(scores) if len(scores) > 1 else 0.0
        sigma = ((mean - null_mean) if higher_is_better else (null_mean - mean)) \
            / max(std, null_std, 1e-9)
        control_results.append(ArmResult(f"control:{c.name}", scores, mean, std,
                                         sigma, sigma >= learning_gate_sigma,
                                         c.cost, c.description))
    escaped = [c.name for c in control_results if c.passed_gate]
    if escaped:
        return _finish(spec, BakeoffResult(
            spec.id, "VOID", None, results + control_results, null_mean, null_std,
            f"control(s) {', '.join(escaped)} CLEARED the {learning_gate_sigma}-"
            f"sigma gate. A control that succeeds means the metric does not "
            f"measure what the spec claims; no comparison on it is valid.",
            spec.metric), ledger, decisions_path)

    failed = [a.name for a in results if not a.passed_gate]
    if failed:
        # See property 1 above. This is the whole point of the module.
        return _finish(spec, BakeoffResult(
            spec.id, "VOID", None, results + control_results, null_mean, null_std,
            f"arms below the {learning_gate_sigma}-sigma learning gate: "
            f"{', '.join(failed)}. An arm that has not demonstrably learned "
            f"cannot arbitrate the decision.", spec.metric), ledger, decisions_path)

    ranked = sorted(results, key=lambda a: a.mean, reverse=higher_is_better)
    best, second = ranked[0], ranked[1]
    unit = max(best.std, second.std, null_std, 1e-9)
    gap = abs(best.mean - second.mean) / unit

    if gap < margin_sigma:
        tied = [a for a in ranked if abs(a.mean - best.mean) / unit < margin_sigma]
        # A TIE is resolved by COST, so cost must be real. It defaults to 0.0,
        # and with every arm at 0.0 `min` returns whichever happened to sort
        # first — an arbitrary pick, reported as "the cheapest". Refuse rather
        # than dress up a coin flip as a measurement.
        if any(a.cost is None for a in tied):
            return _finish(spec, BakeoffResult(
                spec.id, "VOID", None, results + control_results, null_mean, null_std,
                f"{best.name} and {second.name} are within {gap:.2f} sigma so "
                f"the decision falls to cost, but "
                f"{', '.join(a.name for a in tied if a.cost is None)} declared "
                f"none. Declare "
                f"Arm(cost=...) in the units the spec named (params, latency, "
                f"GPU-hours) and re-run.", spec.metric), ledger, decisions_path)
        cheapest = min(tied, key=lambda a: a.cost)
        return _finish(spec, BakeoffResult(
            spec.id, "TIE", cheapest.name, results + control_results, null_mean, null_std,
            f"{best.name} leads {second.name} by only {gap:.2f} sigma "
            f"(margin {margin_sigma}). The choice does not matter yet; "
            f"taking the cheapest tied arm ({cheapest.name}, cost "
            f"{cheapest.cost:g}).", spec.metric), ledger, decisions_path)

    return _finish(spec, BakeoffResult(
        spec.id, "WINNER", best.name, results + control_results, null_mean, null_std,
        f"{best.name} beats {second.name} by {gap:.2f} sigma and clears the "
        f"null by {best.sigma_over_null:.2f} sigma.", spec.metric), ledger, decisions_path)


def _finish(spec: Spec, res: BakeoffResult, ledger: Optional[Ledger],
            decisions_path: Optional[Path] = None) -> BakeoffResult:
    if ledger is not None:
        # VOID maps to Status.VOID, never FAIL. A bakeoff that could not
        # arbitrate has NOT refuted anything, and specs carry a `kills` field:
        # recording VOID as FAIL reads machine-side as the kill criterion
        # firing. That exact corruption is live in T2.02's entry today.
        status = {"WINNER": Status.PASS, "TIE": Status.PASS,
                  "VOID": Status.VOID}[res.verdict]
        ledger.record(Result(
            spec_id=spec.id, status=status,
            metrics=res.to_metrics(),
            message=f"{res.verdict}: {res.reason}",
            ran_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            **Result.env_stamp(),   # commit + hardware: an unattributable
        ))                          # result defeats the overseer's own audit
    _append_decision(res, decisions_path)
    return res


def _append_decision(res: BakeoffResult, path: Optional[Path] = None) -> None:
    """Write the decision — losers included — so it can be re-opened later.

    `path` exists because this function used to hard-code the real record, so
    `bakeoff.py`'s own self-tests appended to it: on 2026-08-09 the entirety of
    `docs/DECISIONS_RESOLVED.md` was nine fixtures on a spec called `TEST`,
    which made the file useless as evidence that any decision had been made.
    Same shape as `submit()` hard-coding `gpu_budget.json`. A test must be able
    to reach the code without reaching the record.
    """
    DECISIONS_FILE = path or DECISIONS
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DECISIONS_FILE.exists():
        DECISIONS_FILE.write_text(
            "# Decisions resolved by bakeoff\n\n"
            "Written by experiments/bakeoff.py. Losing arms are recorded on "
            "purpose: a decision whose alternatives were discarded cannot be "
            "re-opened when the evidence changes, and the alternatives get "
            "silently reinvented later.\n")
    lines = [f"\n## {res.spec_id} — {res.verdict}"
             + (f" — {res.winner}" if res.winner else ""),
             f"\n{res.reason}\n",
             f"\nmetric: `{res.metric}`  ·  null {res.null_mean:.3f} "
             f"± {res.null_std:.3f}\n",
             "\n| arm | mean | sigma over null | gate | cost |",
             "\n|---|---|---|---|---|"]
    for a in sorted(res.arms, key=lambda x: x.mean, reverse=True):
        lines.append(f"\n| {a.name} | {a.mean:.3f} | {a.sigma_over_null:.2f} | "
                     f"{'pass' if a.passed_gate else 'FAIL'} | "
                     f"{a.cost if a.cost is not None else '—'} |")
    lines.append("\n")
    with open(DECISIONS_FILE, "a", encoding="utf-8") as fh:
        fh.write("".join(lines))
