"""D1.0 twin-spread probe — the missing premise of the ADOPTED successor gate.

NOT A SPEC. It writes nothing to the ledger, freezes no gate, moves no
constant, and nothing depends on it. Same idiom as
`lg03_blind_twin_probe.py` and the SM.03 `vis_open` probe: a recorded
diagnostic that a dated Review row asked for by name.

WHY IT EXISTS. `docs/REVIEW_QUEUE.md`,
`d10-successor-rerun-under-adopted-gate` (DISPOSITIONED 2026-09-08, DUE
2026-09-14) adopted a TWIN DENOMINATOR for `D1.0`'s learning gate: each arm is
scored against its OWN untrained twin, never against the random policy. The
disposition's binding condition 3 says the design is undefined until somebody
measures whether a twin HAS a spread:

    "Twin eval is deterministic *at a fixed init seed*, so a twin has no
     spread to divide by and the design is undefined until somebody measures
     whether it has one ACROSS init seeds. ... Run K >= 16 untrained twins per
     architecture at distinct init seeds and report the mean and std of each
     architecture's prior."

THE ECONOMY OF THIS DESIGN, which is the reason it is a probe and not a
dispatch: **nothing is trained.** `_train_arm` calls `_eval_policy` on the
freshly-built arm BEFORE its first optimiser step, so an untrained twin costs
exactly `EVAL_EPISODES` forward-pass rollouts. K twins therefore cost K forward
passes and zero gradient steps, and the expensive term does not scale with the
denominator's seed count at all. Measured on this box: ~69 s for all four arms
at one init seed, so K=32 is ~37 CPU-minutes against attempt 2's 17.61
GPU-hours.

WHY K=32 AND NOT THE DECLARED MINIMUM OF 16. Chosen BEFORE any measurement, for
a reason that is about the branch test rather than about the answer: the
quantity the branch turns on is a STANDARD DEVIATION, and the standard error of
a std estimate is ~std/sqrt(2(K-1)) — 18.3% of the std at K=16, 12.7% at K=32.
The defect this whole row exists to repair is a gate whose verdict moved
because a denominator's std moved 27% between two draws. Estimating the
replacement denominator to +-18% would re-buy a weaker version of the same
disease. K=32 is the cheap half of that fix and it costs 18 extra CPU-minutes.

--------------------------------------------------------------------------
PRE-REGISTERED BRANCH CRITERION — WRITTEN AND COMMITTED BEFORE THE RUN
--------------------------------------------------------------------------
The disposition declares both branches but does NOT quantify "materially
non-zero", and choosing the branch after seeing the number is the move it
explicitly forbids. So the criterion is fixed here, in the commit that
precedes the run:

    SPREAD IS REAL   iff  twin_std > 0 AND twin_cv >= CV_MIN  for EVERY
                          one of the four architectures,
                          where twin_cv = twin_std / |twin_mean|.
    SPREAD IS ~ZERO  otherwise.

`CV_MIN = 0.05`. The rationale, stated so it can be argued with rather than
merely obeyed: the successor gate divides by `twin_std`, so the question is
whether that divisor is a real quantity or a rounding artifact. At CV = 0.05 a
3-sigma band is 15% of the architectural prior — coarser than the 5-episode
deterministic eval can resolve spuriously, and fine enough that a real learning
signal (attempt 2's trained arms banked 350-506 against twin means near 198)
clears it by orders of magnitude. The bar is deliberately LOW because its job
is to detect a DEGENERATE denominator, not to be a second learning gate.

**UNANIMITY IS REQUIRED, not an average.** The gate is applied per-arm, so a
single architecture with a degenerate prior would make the gate undefined for
that arm while the other three looked fine. `min(cv)` is therefore the test
statistic and it is reported as such.

DISCLOSURE, because this criterion must not be fitted to data I had already
seen. Before writing it I ran a TIMING smoke — one init seed (1234), all four
arms, `EVAL_EPISODES` each — which incidentally printed four single-seed twin
returns: aprime 206.8, b_split 322.1, c_e2e 139.7, d_mlp 196.1. Those are ONE
draw per arm and contain no std, so they cannot have tuned `CV_MIN`; but they
do already show the twin mean moving off the recorded attempt-2 values (aprime
198.4, d_mlp 197.6) at a different init seed, which is weak prior evidence that
the spread is real. It is recorded rather than omitted, because a criterion
written by someone who had seen data is a different object from one written
blind, and the reader should get to know which this is.

WHAT THIS PROBE MAY NOT DO. It may not choose the branch by any rule other than
the one above; it may not borrow a spread from another distribution if the
answer is ~zero (the disposition forbids that in terms); it may not move
`MIN_LEARN_SIGMA`, which stays 3.0; and it does not authorise attempt 3 —
that is authorised by this result landing on the row AND the successor gate
being committed in a non-dispatch commit, and it goes to W37.

    /data/venvs/jackthelearner/bin/python -m experiments.tests.d10_twin_spread_probe
"""
from __future__ import annotations

import json
import time

from . import d1_0_control_path_bakeoff as D

#: distinct init seeds. K=32 — see the docstring for why not the declared 16.
K = 32
INIT_SEEDS = list(range(4000, 4000 + K))

#: the pre-registered branch criterion. NOT a ledger gate; this probe records
#: no row. Written before the run; see the docstring.
CV_MIN = 0.05

OUT = "/data/d10_twin_spread.json"


def _twin_return(arm: str, init_seed: int) -> float:
    """One untrained twin's deterministic eval mean. Forward passes only.

    Mirrors `_train_arm`'s own first three lines exactly — seed torch and
    numpy, build the arm, evaluate — and then STOPS, before the optimiser
    exists. Using the spec's own `_make_arm`/`_eval_policy` rather than a
    re-implementation is the point: a probe that built its own twin would be
    measuring a different object from the one the gate divides by.
    """
    np, torch, _ = D._lazy_torch()
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    tp, _aux, _meta = D._make_arm(arm)
    returns = D._eval_policy(tp, D.EVAL_EPISODES)
    return float(np.mean(returns))


def main() -> dict:
    np, _torch, _nn = D._lazy_torch()
    t0 = time.time()
    out: dict = {"K": K, "init_seeds": INIT_SEEDS, "cv_min": CV_MIN,
                 "eval_episodes": D.EVAL_EPISODES,
                 "eval_seed_base": D.EVAL_SEED_BASE, "arms": {}}

    for arm in D.ARMS:
        vals = []
        for s in INIT_SEEDS:
            vals.append(_twin_return(arm, s))
            print(f"  {arm:8s} seed {s} -> {vals[-1]:8.2f}", flush=True)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))      # sample std: these are K draws
        out["arms"][arm] = {
            "returns": [round(v, 4) for v in vals],
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv": round(std / abs(mean), 6) if mean else None,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            # standard error of the std estimate, the quantity K was chosen for
            "std_se_frac": round((2 * (K - 1)) ** -0.5, 4),
        }
        print(f"{arm:8s} mean {mean:8.2f}  std {std:7.2f}  "
              f"cv {std / abs(mean):.4f}", flush=True)

    cvs = {a: out["arms"][a]["cv"] for a in D.ARMS}
    min_cv = min(cvs.values())
    out["min_cv"] = round(min_cv, 6)
    out["branch"] = "SPREAD_IS_REAL" if min_cv >= CV_MIN else "SPREAD_IS_ZERO"
    out["elapsed_s"] = round(time.time() - t0, 1)

    print("\n" + "=" * 66)
    print(f"min cv = {min_cv:.4f} over {cvs}")
    print(f"BRANCH (pre-registered, CV_MIN={CV_MIN}): {out['branch']}")
    print("=" * 66)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}  ({out['elapsed_s']} s)")
    return out


if __name__ == "__main__":
    main()
