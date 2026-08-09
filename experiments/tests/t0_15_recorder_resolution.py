"""T0.15 — the recorder must not be able to disarm a pre-registered threshold.

`run_spec` does not call `check()` on what a spec measured. It calls it on
`_aggregate(runs)` — the mean across seeds, rounded. So the recorder sits
downstream of every gate in the ladder, and its resolution is an upper bound on
how tight any threshold in this project can be.

Until 2026-08-09 that rounding was `round(mean, 6)`. A genuine drift of 3e-7 was
therefore recorded as `0.0` and satisfied `drift <= 0.0`. The affected gates are
the strictest ones in the repo, which is not a coincidence — bit-identity and
parity claims are exactly the ones that live below 1e-6:

    T0.14  MAX_EVAL_DRIFT = 0.0   the gate that closed the dropout bug
    T0.02  determinism            T1.10  cpu/gpu agreement
    T1.11  path parity            T0.03/T0.04  checkpoint round-trips

No PASS was ever falsely green: `_aggregate` returns the single run untouched
when `len(runs) == 1`, and all six are `seeds=1`. The defect was latent and
aimed precisely where this project keeps going — GOAL.md asks for >= 3 seeds and
the overseer has asked for more 3-seed re-verification. The first spec to run
three seeds with a sub-microscale gate was PG.8, and both of its 1e-9 deviation
checks recorded `0.0`.

It is invisible to T0.13, the spec built to find dead gates: T0.13 perturbs a
key of the RECORDED metrics and asks whether the verdict moves. Perturb a
recorded 0.0 and it moves. The saturation is manufactured after the measurement
and before the check, in the one place no `_check` can look at.

Three pre-registered checks against `protocol._aggregate` as run_spec uses it:

  resolvable    across magnitudes 1e-1 down to 1e-18, a set of identical
                nonzero runs never aggregates to exactly 0.0.
  gate_fires    the motivating case end to end: three seeds whose true drift is
                3e-7, checked against `drift <= 0.0`, must record FAIL.
  faithful      for ordinary-size metrics the recorded value stays within 1e-5
                relative of the true mean — the fix must not have bought
                resolution by mangling the numbers anyone reads.

CONTROL: the pre-fix aggregator, `round(x, 6)` applied by the identical
harness, must FAIL `resolvable` and `gate_fires`. This is the load-bearing part.
Without it the spec passes on every implementation including the broken one, and
would be one more decorative Tier-0 gate of exactly the kind T0.09 and T0.12
turned out to be. It is a control, not a weak arm: it is supposed to fail.
"""
from __future__ import annotations

from ..protocol import Budget, Ledger, Spec, run_spec
from ..registry import BY_ID

MAGNITUDES = [10.0 ** -e for e in range(1, 19)]
TRUE_DRIFT = 3e-7           # below the pre-fix 5e-7 rounding floor
FAITHFUL_REL_TOL = 1e-5
ORDINARY = [1.118133, 0.019357, 42.0, 0.5, 1234.56789, 0.123456789]


def _aggregate_with(round_fn):
    """`protocol._aggregate` with its rounding swapped out.

    Re-implemented rather than monkey-patched so the control exercises the same
    code path the experiment does. A control built out of a tidier restatement
    of the real thing is how T0.13's first scan came back clean on a
    known-broken gate.
    """
    def agg(runs):
        if len(runs) == 1:
            return dict(runs[0])
        out = {}
        for k in runs[0]:
            vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
            if len(vals) == len(runs):
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                out[k] = round_fn(mean)
                out[f"{k}_std"] = round_fn(var ** 0.5)
            else:
                out[k] = runs[0][k]
        return out
    return agg


def _measure(round_fn) -> dict:
    """Score one rounding rule on the three checks."""
    agg = _aggregate_with(round_fn)

    # resolvable — a nonzero must never be recorded as zero
    zeroed = []
    for mag in MAGNITUDES:
        rec = agg([{"x": mag}, {"x": mag}, {"x": mag}])["x"]
        if rec == 0.0:
            zeroed.append(mag)
    min_resolvable = min((m for m in MAGNITUDES if m not in zeroed),
                         default=0.0)

    # gate_fires — the motivating case, through run_spec's own machinery
    runs = [{"drift": TRUE_DRIFT * f} for f in (1.0, 1.0, 1.0)]
    recorded_drift = agg(runs)["drift"]
    gate_verdict = "PASS" if recorded_drift <= 0.0 else "FAIL"

    # faithful — resolution must not cost accuracy on readable numbers
    rel = 0.0
    for v in ORDINARY:
        rec = agg([{"x": v}, {"x": v}, {"x": v}])["x"]
        rel = max(rel, abs(rec - v) / abs(v))

    return {
        "n_magnitudes_zeroed": len(zeroed),
        "min_resolvable_magnitude": min_resolvable,
        "recorded_drift": recorded_drift,
        "true_drift": TRUE_DRIFT,
        "gate_verdict_on_3e7_drift": gate_verdict,
        "max_relative_error": rel,
    }


def _live_spec_check(seed: int) -> dict:
    """End-to-end: a throwaway spec whose _check is `drift <= 0.0`, driven by
    the REAL run_spec against a ledger that is never written to disk.

    The unit test above measures `_aggregate`; this measures the thing that
    actually decides a ladder entry. They can disagree — run_spec could round
    again, or check per-seed — and if they ever do, that is the finding.
    """
    spec = Spec("T0.15.probe", 0, "in-memory probe, never recorded",
                hypothesis="a 3e-7 drift must not satisfy `drift <= 0.0`",
                falsified_by="it does",
                null_baseline="n/a", metric="drift", budget=Budget.CPU, seeds=3)
    res = run_spec(spec, lambda s: {"drift": TRUE_DRIFT},
                   lambda m, c: m["drift"] <= 0.0,
                   ledger=_MemoryLedger())
    return {"end_to_end_status": str(getattr(res.status, "value", res.status)),
            "end_to_end_drift": res.metrics.get("drift", -1.0)}


class _MemoryLedger:
    """A Ledger that answers `blocked_by` and swallows `record`.

    T0.15's probe spec must never touch experiments/ledger.json — the file's
    own header forbids hand-written entries, and a probe id appearing there is
    indistinguishable from a claim. This is the same failure `bakeoff.py` has
    with DECISIONS_RESOLVED.md (overseer item 4), avoided rather than repeated.
    """

    def __init__(self):
        self.recorded = []

    def blocked_by(self, spec):
        return []

    def record(self, res):
        self.recorded.append(res)


def _experiment(seed: int) -> dict:
    from ..protocol import _round6

    out = _measure(_round6)
    out.update(_live_spec_check(seed))
    return out


def _control(seed: int) -> dict:
    """The aggregator as it was until 2026-08-09. It must fail."""
    return _measure(lambda x: round(x, 6))


def _check(m: dict, c: dict) -> bool:
    return (
        # the fix works
        m["n_magnitudes_zeroed"] == 0
        and m["min_resolvable_magnitude"] <= 1e-18
        and m["recorded_drift"] > 0.0
        and m["gate_verdict_on_3e7_drift"] == "FAIL"
        and m["end_to_end_status"] == "FAIL"
        and m["end_to_end_drift"] > 0.0
        and m["max_relative_error"] <= FAITHFUL_REL_TOL
        # and the measurement can see the bug it was written for
        and c["n_magnitudes_zeroed"] > 0
        and c["recorded_drift"] == 0.0
        and c["gate_verdict_on_3e7_drift"] == "PASS"
    )


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T0.15"], _experiment, _check, control_fn=_control,
                    ledger=ledger)
