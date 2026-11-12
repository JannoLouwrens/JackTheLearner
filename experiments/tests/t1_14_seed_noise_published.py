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

SEED-IGNORING ARM — RUN, NOT DESCRIBED (strengthening, 2026-11-12, ORDERS J1
step 2). Attempt 1's control ("an arm that ignored its seed would read zero
spread") was a property of the three real arms only; nothing ever showed the
detector fires on a seed-ignoring triple. The job now trains FIVE arms in one
session: seeds 0, 1, 2 (the real triple, exactly as before) and seed 0 twice
more. The seed-ignoring triple is (arm 0, arm 0', arm 0'') — the same seed fed
to all three runs. The spec's OWN spread/distinct clause (`_seed_clause`, the
same code `_check` applies to the real triple, bars unmoved) must REJECT it;
if it does not — e.g. GPU non-determinism makes same-seed runs differ by more
than 1e-6, or into three distinct results — then "seed noise" is not
separable from run-to-run noise on this pipeline and the spec FAILS. Reported
beside it: the ignored triple's spread and its ratio to the real spread.
The SNR bar (3.0) and MIN_SPREAD (1e-6) are byte-unmoved. Same `t108.JOB`
text — only the substituted seed list grows — so the pipeline is unchanged.

SAME JOB. The GPU job, the recipe (STEP 2b mechanisms (i)+(ii)), the task and
the seeds are imported from `t1_08_seed_variance` unchanged, so both specs
measure the same pipeline; that file is in IMPL_DEPS and an edit to it stales
this certificate. `snr` here is computed by the same `_stats` from the same
`improvement` field `T1.08` reads.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..gpu import build_job, submit
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from . import t1_08_seed_variance as t108

IMPL_DEPS = ["UnifiedBrain.py", "experiments/protocol.py",
             "experiments/tests/t1_08_seed_variance.py"]

SEEDS = t108.SEEDS
IGNORED_EXTRA = [SEEDS[0], SEEDS[0]]   # two more runs at seed 0: the
                                      # seed-ignoring triple is arm0, these two
MIN_SNR = 3.0          # T1.08's August bar, verbatim
MIN_SPREAD = 1e-6      # T1.08's control, verbatim

_CACHE: dict = {}


def _submit() -> dict:
    """`t108._submit`'s job, with the seed list extended by IGNORED_EXTRA.

    Same JOB text, same build_job/submit/fetch path; the only difference is
    the substituted `__SEEDS__`. Kept local so `t1_08_seed_variance.py` (and
    T1.08's standing FAIL certificate) is byte-unmoved.
    """
    job = build_job(t108.JOB.replace("__SEEDS__", repr(SEEDS + IGNORED_EXTRA))
                             .replace("__TAIL_FRAC__", repr(t108.TAIL_FRAC)))
    # est_hours 0.5 -> 0.9: five arms where attempt 1 trained three
    # (attempt 1 on a Colab T4: 0.65 GPU-h for three arms incl. setup).
    res = submit(job, prefer="kaggle", est_hours=0.9, timeout_s=6600,
                 fetch=["t108.json"])
    _keep_stdout(res)
    if not res.ok:
        raise RuntimeError(f"GPU job failed on {res.backend}: {res.message}")
    path = res.artifacts.get("t108.json")
    if path:
        d = json.loads(Path(path).read_text())
    else:
        # The artifact download lost the run VM (2026-11-12, attempt 2: colab
        # exited 0 after 66 min, `fetch failed: /content/t108.json exists=False`,
        # 1.10 GPU-h of five finished arms thrown away). The JOB prints the
        # whole payload on one `JACKRESULT {...}` line (t108 ruling §9d) and the
        # run session's stdout is the channel that survives; read it there.
        d = _payload_from_stdout(res.stdout)
        if d is None:
            raise RuntimeError(
                f"no artifact and no JACKRESULT line from {res.backend}. "
                f"message={res.message!r} stdout_tail={res.stdout[-400:]!r} "
                f"stderr_tail={res.stderr[-400:]!r}")
        d["recovered_from"] = "stdout JACKRESULT line (t108 ruling 9d)"
    d["backend"] = res.backend
    all_arms = d["arms"]
    n = len(SEEDS)
    if len(all_arms) != n + len(IGNORED_EXTRA):
        raise RuntimeError(f"expected {n + len(IGNORED_EXTRA)} arms, got {len(all_arms)}")
    d["arms"] = all_arms[:n]                       # the real triple, as before
    d["ignored_arms"] = [all_arms[0]] + all_arms[n:]  # same seed, three runs
    return d


STDOUT_DIR = Path("/data/jack-logs/gpu_stdout")


def _keep_stdout(res) -> None:
    """Write the job's WHOLE stdout/stderr to disk before anything can fail.
    Attempt 2's payload existed only in that stdout, and the ledger kept a
    400-char tail of it. Best-effort: a failed write never fails the run."""
    try:
        STDOUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"T1.14-{res.job_id or 'nojob'}".replace("/", "_")
        (STDOUT_DIR / f"{stem}.stdout").write_text(res.stdout or "")
        (STDOUT_DIR / f"{stem}.stderr").write_text(res.stderr or "")
    except Exception:
        pass


def _payload_from_stdout(stdout: str):
    """The last parseable `JACKRESULT {...}` line, or None."""
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("JACKRESULT "):
            try:
                return json.loads(line[len("JACKRESULT "):])
            except json.JSONDecodeError:
                return None
    return None


def _seed_clause(held: list) -> tuple:
    """The spread/distinct clause of `_check`, factored so the SAME code judges
    the real triple and the seed-ignoring triple. Returns (passes, spread,
    distinct)."""
    distinct = len(set(round(h, 9) for h in held))
    spread = round(max(held) - min(held), 8)
    return (spread > MIN_SPREAD and distinct == len(SEEDS)), spread, distinct


def _experiment(seed: int) -> dict:
    _CACHE.update(_submit())
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
        "artifact_channel": _CACHE.get("recovered_from", "t108.json download"),
    }


def _control(seed: int) -> dict:
    """Seeds must actually change the outcome, or the noise floor is fictional."""
    ok, spread, distinct = _seed_clause([a["heldout"] for a in _CACHE["arms"]])
    ig = _CACHE.get("ignored_arms") or []
    out = {"distinct_results": distinct, "spread": spread,
           "ignored_seeds": json.dumps([a.get("seed") for a in ig])}
    if len(ig) == len(SEEDS):
        ig_ok, ig_spread, ig_distinct = _seed_clause([a["heldout"] for a in ig])
        out.update({
            "ignored_spread": ig_spread,
            "ignored_distinct_results": ig_distinct,
            # 1.0 iff the clause REJECTS the seed-ignoring triple — the
            # control must fire for this spec to pass.
            "ignored_flagged": float(not ig_ok),
            "ignored_to_real_spread": round(ig_spread / max(spread, 1e-12), 6),
        })
    return out


def _check(m: dict, c: dict) -> bool:
    mde = m.get("min_detectable_effect")
    return (m["snr"] >= MIN_SNR
            and mde is not None and math.isfinite(mde) and mde > 0
            and c["spread"] > MIN_SPREAD
            and c["distinct_results"] == len(SEEDS)
            # the seed-ignoring arm was RUN and the same clause rejected it
            and c.get("ignored_flagged") == 1.0)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["T1.14"], _experiment, _check, control_fn=_control, ledger=ledger)
