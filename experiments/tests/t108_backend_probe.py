"""T1.08 backend-confound probe — same kernel, same seed list, both venues.

NOT A SPEC. It writes no ledger row (never calls `run_spec`), freezes no gate,
moves no constant, and `T1.08` STAYS FAIL ON EVERY BRANCH of its read. Same
idiom as `d10_twin_spread_probe.py` and `sm03_vis_open_probe.py`: a recorded
diagnostic that a dated Review row asked for by name, committed BEFORE its
number exists.

AUTHORISATION. The RULING of 2026-09-14 (Review DAILY) on
`t108-bar-set-from-n1-is-now-the-projects-largest-blocker`
(docs/REVIEW_QUEUE.md), §2: *"AUTHORISED: the backend-confound arm pair, as a
PROBE, at n=5 per backend"* — 1.20 GPU-h of W37's 29.18 (expires Sat
2026-09-19), *"same kernel, same seed list, same commit, both backends"*, with
`SEEDS` inside the JOB and never the registry's `seeds` field. §8 orders (1) a
pre-registration commit with NO dispatch in it — this file is that commit —
then (2) the dispatch.

WHY IT EXISTS, in two sentences. `T1.08` attempt 2 (Colab T4, 2026-08-12) read
`heldout_cv_pct` 5.717 and attempt 3 (Kaggle P100, 2026-09-13) read 40.006
against the pre-registered `MAX_HELDOUT_CV_PCT` 7.0 — `heldout_std` moved
8.2x, backend confounded with the jump, and the row's annotation (a)
eliminated code drift (identical JOB text, task tensors, initial weights and
batch order across devices), so what differs across venues is CUDA-RNG draws
and kernel nondeterminism. The ruling's §1: 40.006 is NOT explained by the n=3
estimator (95% interval at true cv 5.717 is [0.92%, 11.00%]); something real
moved, and the two candidates — venue and pipeline — cannot be separated by
the readings on the ledger, which is exactly what this probe separates.

--------------------------------------------------------------------------
THE SEED LIST, FIXED IN THIS PRE-REGISTRATION COMMIT
--------------------------------------------------------------------------
`PROBE_SEEDS = [0, 1, 2, 3, 4]` — n=5 per backend, per the ruling (the row's
own (d) priced the upgrade from n=3: alpha 4.0% -> 0.2%, power 92.2% -> 98.2%,
for 0.48 h). The list extends the spec's registered `SEEDS = [0, 1, 2]` in the
only order that cannot be a choice, and deliberately CONTAINS the three seeds
that produced 40.006 on the P100 and 5.717 on the T4, so each venue's n=5
reading is directly interpretable against the recorded n=3 one.

--------------------------------------------------------------------------
THE READ, PRE-REGISTERED — ruling §4, quoted verbatim
--------------------------------------------------------------------------
    "Let cv_T4 and cv_P100 be the n=5 readings.

    - (i) BOTH > 7.0. The noise is the repo's. The FAIL is a fact about the
      pipeline, and the repair is the pipeline — never the bar, and never the
      venue. T1.08 stays FAIL and the ladder's honest statement becomes 'every
      Tier-2+ claim on this pipeline is made against a held-out seed spread
      this large', which is item 6 below.
    - (ii) BOTH < 7.0. Attempt 3's 40.006 was neither venue nor pipeline but a
      tail draw at n=3 — a reading the row's own (b) says is very unlikely,
      which is exactly why this branch must be pre-registered rather than
      reached for. The repair is then the estimator, not the bar: T1.08's
      registered seed count rises and the spec is re-run at the higher n. That
      is a different run and its dispatch is legal. Raising n is NOT a
      weakening and the desk states why: more seeds move the sample CV toward
      the truth in both directions — if the true cv is above 7.0 they make
      the FAIL more certain, not less. The number of seeds is fixed in branch
      (ii)'s pre-registration commit, before the probe's numbers are read, so
      it cannot be chosen to buy a pass.
    - (iii) THEY SPLIT. The venue term is real. Then T1.08 cannot certify a
      venue-invariant noise floor from one venue at all, and its claim is
      venue-scoped and must say so in the spec text. A venue-scoped noise
      floor is a smaller claim than the one the docstring makes today, and
      shrinking a claim to match what was measured is the one direction this
      desk is always allowed to go."

The boundary is read off the conjunct itself: `heldout_cv_pct <= 7.0` CLEARS,
so "above" means `cv > MAX_HELDOUT_CV_PCT` (imported from the spec, not
restated). cv_T4 is the colab arm's reading, cv_P100 the kaggle arm's.

--------------------------------------------------------------------------
BRANCH (ii)'s SEED COUNT — FIXED HERE, BEFORE ANY PROBE NUMBER EXISTS
--------------------------------------------------------------------------
`BRANCH_II_SEEDS = 20`. If (and only if) branch (ii) fires, `T1.08`'s
registered seed count rises to 20 and the spec re-runs at that n. The number
is chosen from arithmetic that predates this probe — the row's (c) table:
false-fail at the bar's own source value (true cv 5.717) is 22.6% at n=3,
7.5% at n=20 (2.4 GPU-h at the measured 0.12 GPU-h/seed), 4.2% at n=30
(3.6 GPU-h). n=20 buys the steep part of that curve at 8% of a weekly pot;
n=30's extra 1.2 h buys 3.3 points. It may not be revisited after the probe's
numbers are read. The re-run's VENUE ROUTING is also fixed now, before any
number: it stays the spec's existing `prefer="colab"` with Kaggle fallback,
untouched — so the venue cannot be selected after the probe reports which one
is kind.

--------------------------------------------------------------------------
WHAT THIS PROBE MAY NOT DO — ruling §3, the load-bearing half
--------------------------------------------------------------------------
It may not buy `T1.08` a verdict, on any branch. Specifically forbidden:
*"dispatching T1.08 to a T4 because the probe reported that the T4 reads
lower. That is run-until-pass wearing a hardware argument ... The venue a
certificate is bought on may never be selected after seeing which venue is
kind."* It also moves no bar: `MAX_HELDOUT_CV_PCT` 7.0 is pre-registered and
law 3 is unconditional.

--------------------------------------------------------------------------
MECHANICS, and one honesty note about venue pinning
--------------------------------------------------------------------------
The kernel is `t1_08_seed_variance.JOB` — IMPORTED, not copied, so the probe
cannot drift from what it diagnoses — with `__SEEDS__` replaced by
`PROBE_SEEDS` exactly as the spec's own `_submit` does. Both submissions go
out from one pushed HEAD (`build_job` pins the ref), so "same commit, both
backends" is enforced by the same guard every dispatch obeys.

`gpu.submit()` cannot be pinned to a venue: it tries `prefer` and falls back.
So each arm is stored under the venue that ACTUALLY ran (`res.backend`), never
the intended one, and the probe is idempotent state on disk: colab is
attempted first (a Colab 503 falls back to Kaggle, and that result still
fills the kaggle arm — it is the byte-identical kernel — with zero wasted
quota), an arm already filled is never re-bought, a same-venue duplicate goes
to `extras` and the probe reports INCOMPLETE rather than fabricating a
separation. The pre-registered read is only taken when both venues hold an
n=5 reading.

Spend is attributed (`JACK_SPEC_ID=T1.08`, `JACK_SPEC_PHASE=probe`) so
`gpu_hours_no_verdict` counts it against T1.08's question rather than growing
`gpu_unattributed_jobs` — the counter WILL move and the harvest commit should
say so. The probe takes the GPU serialisation lock for its lifetime, like any
other GPU run.

    setsid nohup /data/venvs/jackthelearner/bin/python -m \
        experiments.tests.t108_backend_probe > /data/tmp/t108_backend_probe.log 2>&1 &

Launched in the dispatch.sh idiom (setsid + proc_declare + 15 s liveness and
artifact check), NOT via launch_detached.sh: its cpu_budget wrapper bills wall
clock, and this process's wall is remote-kernel waiting — the exact case
cpu_budget.py's own header exempts ("billing waiting as box CPU would make
the meter read harm where there is none").
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

from ..gpu import Budget, build_job, submit
from . import t1_08_seed_variance as T

#: n=5 per backend, fixed in the pre-registration commit. See the docstring.
PROBE_SEEDS = [0, 1, 2, 3, 4]

#: branch (ii)'s registered-seed-count repair, fixed BEFORE any probe number
#: exists. May not be revisited after the numbers are read.
BRANCH_II_SEEDS = 20

#: the bar, imported from the spec rather than restated — 7.0, and it does
#: not move on any branch.
BAR = T.MAX_HELDOUT_CV_PCT

OUT = Path("/data/t108_backend_probe.json")
GPU_LOCK = "/tmp/jack-ladder-gpu.lock"

#: measured basis: attempt 3 charged 0.36 h for 3 seeds -> 0.12 h/seed -> 0.60 h
#: for 5. 0.70 leaves margin for clone overhead; affordability only.
EST_HOURS = 0.70
TIMEOUT_S = 5400


def _fresh_state() -> dict:
    return {
        "probe": "t108_backend_confound",
        "authorised_by": ("t108-bar-set-from-n1-is-now-the-projects-largest-"
                          "blocker RULING 2026-09-14 §2/§8"),
        "probe_seeds": PROBE_SEEDS,
        "branch_ii_seeds": BRANCH_II_SEEDS,
        "bar": BAR,
        "arms": {"kaggle": None, "colab": None},
        "extras": [],
        "failures": [],
        "branch": None,
    }


def _load_state() -> dict:
    if OUT.exists():
        state = json.loads(OUT.read_text())
        # The pre-registered constants are pinned: a state file carrying other
        # values is from a different probe and must not be silently merged.
        assert state["probe_seeds"] == PROBE_SEEDS and state["bar"] == BAR, (
            f"state file {OUT} disagrees with the pre-registered constants")
        return state
    return _fresh_state()


def _save_state(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=1))


class _KaggleAlreadyRead(Budget):
    """Refuses kaggle so `submit`'s fallback cannot buy a DUPLICATE reading.

    Used only when the kaggle arm is already filled and the colab arm is being
    attempted: colab's last two real attempts (2026-09-07) failed in ~5 s, and
    a fast colab failure would otherwise fall back to kaggle and spend ~0.6 h
    on a venue whose n=5 reading already exists. This is the documented use of
    `submit(budget=...)` — routing control without touching the accounting
    file — and it never blocks the FIRST kaggle purchase.
    """

    def afford(self, backend: str, est_hours: float) -> bool:  # noqa: D102
        if backend == "kaggle":
            return False
        return super().afford(backend, est_hours)


def _arm_record(intended: str, *, block_kaggle: bool = False) -> dict:
    """One submission of the spec's own kernel. Returns the record keyed by
    the venue that ACTUALLY ran, which may differ from `intended`."""
    job = build_job(T.JOB.replace("__SEEDS__", repr(PROBE_SEEDS)))
    res = submit(job, prefer=intended, est_hours=EST_HOURS,
                 timeout_s=TIMEOUT_S, fetch=["t108.json"],
                 budget=_KaggleAlreadyRead() if block_kaggle else None)
    if not res.ok:
        return {"ok": False, "intended": intended,
                "message": res.message, "backend": res.backend}
    path = res.artifacts.get("t108.json")
    if not path:
        return {"ok": False, "intended": intended, "backend": res.backend,
                "message": f"no artifact; stdout_tail={res.stdout[-300:]!r} "
                           f"stderr_tail={res.stderr[-300:]!r}"}
    d = json.loads(Path(path).read_text())
    held = [a["heldout"] for a in d["arms"]]
    imps = [a["improvement"] for a in d["arms"]]
    h_mean, h_std = T._stats(held)
    eff, noise = T._stats(imps)
    return {
        "ok": True, "intended": intended, "backend": res.backend,
        "gpu": d["gpu"], "job_id": res.job_id, "head": res.head,
        "duration_s": round(res.duration_s or 0.0, 1),
        "seeds": [a["seed"] for a in d["arms"]],
        "heldout": [round(h, 6) for h in held],
        "improvement": [round(i, 6) for i in imps],
        "heldout_mean": round(h_mean, 5),
        "heldout_std": round(h_std, 6),
        # identical arithmetic to the spec's _experiment, via the same _stats
        "heldout_cv_pct": round(100 * h_std / max(abs(h_mean), 1e-9), 3),
        "effect": round(eff, 5),
        "seed_noise": round(noise, 6),
        "snr": round(eff / max(noise, 1e-9), 2),
    }


def _finalise(state: dict) -> None:
    """Both venues hold an n=5 reading: take the pre-registered read."""
    cv_p100 = state["arms"]["kaggle"]["heldout_cv_pct"]
    cv_t4 = state["arms"]["colab"]["heldout_cv_pct"]
    above_p100, above_t4 = cv_p100 > BAR, cv_t4 > BAR
    if above_p100 and above_t4:
        state["branch"] = "(i) BOTH_ABOVE — the noise is the repo's; the repair is the pipeline, never the bar, never the venue"
    elif not above_p100 and not above_t4:
        state["branch"] = (f"(ii) BOTH_BELOW — attempt 3 was a tail draw at n=3; the repair is the "
                           f"ESTIMATOR: T1.08's registered seed count rises to {BRANCH_II_SEEDS} "
                           f"(fixed pre-probe) and the spec re-runs; venue routing unchanged")
    else:
        state["branch"] = "(iii) SPLIT — the venue term is real; T1.08's claim is venue-scoped and the spec text must say so"
    lo, hi = sorted([cv_p100, cv_t4])
    state["cv_T4"] = cv_t4
    state["cv_P100"] = cv_p100
    state["discordance_ratio"] = round(hi / max(lo, 1e-9), 3)
    print("\n" + "=" * 70)
    print(f"cv_T4 (colab) = {cv_t4}   cv_P100 (kaggle) = {cv_p100}   bar = {BAR}")
    print(f"BRANCH (pre-registered): {state['branch']}")
    print(f"T1.08 STAYS FAIL — this probe buys no verdict (ruling §3).")
    print("=" * 70)


def main() -> int:
    # Attribute the spend to the question it serves; "probe" mirrors the
    # recorded "pilot" phase so it is summable separately from registered runs.
    os.environ["JACK_SPEC_ID"] = "T1.08"
    os.environ["JACK_SPEC_PHASE"] = "probe"
    # The deadline guard protects watchers that are CHILDREN of a builder
    # session; this process is detached by its launch contract and outlives
    # any slot. Launched from a slot, the env leaks in and silently reroutes
    # the colab arm to kaggle — measured on this probe's first invocation
    # (2026-09-14 07:15: intended=colab, attempt row backend=kaggle).
    os.environ.pop("JACK_ITER_DEADLINE", None)

    lock_fh = open(GPU_LOCK, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"REFUSED: {GPU_LOCK} is held — GPU runs are serialised. "
              f"Retry when the holder finishes.")
        return 1

    state = _load_state()
    t0 = time.time()
    # Colab first — thrift: a Colab failure falls back to Kaggle inside
    # submit(), and that result still fills the kaggle arm (byte-identical
    # kernel), so nothing is wasted. See the docstring.
    for intended in ("colab", "kaggle"):
        if all(state["arms"].values()):
            break
        if state["arms"].get(intended):
            continue
        print(f"--- submitting n={len(PROBE_SEEDS)} arm, intended={intended}", flush=True)
        rec = _arm_record(intended,
                          block_kaggle=(intended == "colab"
                                        and state["arms"]["kaggle"] is not None))
        if not rec.get("ok"):
            state["failures"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      **rec})
            _save_state(state)
            print(f"arm FAILED ({intended}): {rec.get('message')}", flush=True)
            continue
        actual = rec["backend"]
        if state["arms"].get(actual) is None:
            state["arms"][actual] = rec
            print(f"arm LANDED on {actual} ({rec['gpu']}): "
                  f"heldout_cv_pct={rec['heldout_cv_pct']}", flush=True)
        else:
            state["extras"].append(rec)
            print(f"DUPLICATE venue {actual} (intended {intended}) — stored "
                  f"under extras; the pair is still unseparated", flush=True)
        _save_state(state)

    state["elapsed_s"] = round(time.time() - t0, 1)
    if all(state["arms"].values()):
        _finalise(state)
        _save_state(state)
        print(f"wrote {OUT}")
        return 0
    _save_state(state)
    print(f"INCOMPLETE — arms present: "
          f"{[k for k, v in state['arms'].items() if v]}; failures: "
          f"{len(state['failures'])}. Idempotent: re-run to fill the missing "
          f"arm. wrote {OUT}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
