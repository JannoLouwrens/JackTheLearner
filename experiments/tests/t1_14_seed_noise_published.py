"""T1.14 — seed noise is measured and published.

REGISTERED 2026-11-11 (ORDERS.md J1, ruling recorded in
`docs/DECISIONS_RESOLVED.md` under that date). This is `T1.08`'s ORIGINAL claim,
the one it passed twice in August, split out as its own spec:

  * seeds change the outcome — three arms that differ only by seed give three
    distinct held-out results (control: spread > 0; an arm that ignored its
    seed would read spread == 0 and VOID this, not pass it);
  * the effect is large next to seed noise — snr = effect / seed_noise >= 3;
  * the noise floor is PUBLISHED — `min_detectable_effect` (2 x seed noise) is
    recorded, finite and > 0, so a downstream multi-seed spec has a number to
    compare its own effect against.

WHAT THIS DOES NOT CLAIM. `T1.08`'s second conjunct — the held-out metric's
own seed CV <= 7% — is NOT part of this spec and is NOT weakened by it. It stays
on `T1.08`, byte-unmoved, and `T1.08` stays FAIL (17.3% at attempt 5). That
conjunct protects exactly one thing: quoting a SINGLE-seed number downstream.
The dependents moved to this spec (`T2.01`, `T2.02`, `T2.03`, `T2.14`, `D1.0`)
each run 3 seeds and gate on their own sigma; `T2.13` (seeds = 1) keeps `T1.08`.
`heldout_cv_pct` is still recorded here so the red fact travels with the green
one — reported, not gated.

SAME JOB. The GPU job, the recipe (STEP 2b mechanisms (i)+(ii)), the task and
the seeds are imported from `t1_08_seed_variance` unchanged, so both specs
measure the same pipeline; that file is in IMPL_DEPS and an edit to it stales
this certificate. `snr` here is computed by the same `_stats` from the same
`improvement` field `T1.08` reads.
"""
from __future__ import annotations

import math

from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from . import t1_08_seed_variance as t108

IMPL_DEPS = ["UnifiedBrain.py", "experiments/protocol.py",
             "experiments/tests/t1_08_seed_variance.py"]

SEEDS = t108.SEEDS
MIN_SNR = 3.0          # T1.08's August bar, verbatim
MIN_SPREAD = 1e-6      # T1.08's control, verbatim

_CACHE: dict = {}


def _experiment(seed: int) -> dict:
    _CACHE.update(t108._submit())
    arms = _CACHE["arms"]
    imps = [a["improvement"] for a in arms]
    held = [a["heldout"] for a in arms]
    eff, noise = t108._stats(imps)
    h_mean, h_std = t108._stats(held)
    return {
        "gpu": _CACHE["gpu"], "backend": _CACHE["backend"], "seeds": len(arms),
        "effect": round(eff, 5),
        "seed_noise": round(noise, 6),
        "snr": round(eff / max(noise, 1e-9), 2),
        "min_detectable_effect": round(2 * noise, 6),
        "heldout_mean": round(h_mean, 5),
        "heldout_std": round(h_std, 6),
        # REPORTED, NOT GATED: T1.08's conjunct, carried so the red fact is
        # visible beside this verdict. Gating it here would just be T1.08.
        "heldout_cv_pct": round(100 * h_std / max(abs(h_mean), 1e-9), 3),
        "lr_final_max": max(a.get("lr_final", -1.0) for a in arms),
    }


def _control(seed: int) -> dict:
    """Seeds must actually change the outcome, or the noise floor is fictional."""
    held = [a["heldout"] for a in _CACHE["arms"]]
    return {"distinct_results": len(set(round(h, 9) for h in held)),
            "spread": round(max(held) - min(held), 8)}


def _check(m: dict, c: dict) -> bool:
    mde = m.get("min_detectable_effect")
    return (m["snr"] >= MIN_SNR
            and mde is not None and math.isfinite(mde) and mde > 0
            and c["spread"] > MIN_SPREAD
            and c["distinct_results"] == len(SEEDS))


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.14"], _experiment, _check, control_fn=_control, ledger=ledger)
