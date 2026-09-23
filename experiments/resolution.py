"""Anchor-margin resolution report — did a conjunct DECIDE, or merely land?

THE SCAR (109th audit, 2026-09-23; LESSONS.md "AN ANTI-SATURATION RULE ARMED
ONLY AGAINST THE CEILING LEAVES THE FLOOR OPEN"). `T4.06`'s winner rule
certified `loss_reweight` on `min_modality_latent_r2 > bar` by **+0.0187**
against an anchor whose own seed-to-seed spread of the same statistic was
**0.2699** — 6.9%, with one of three paired seeds regressing — while the same
bare `>` refuted `grad_norm` at **−0.1529**, 8.2× that margin, all three seeds
agreeing. The guard worked as a refuter and was read as a certifier. The
audit's transferable check, verbatim: *"put the winning margin next to the
within-arm seed spread of the same statistic. If the margin is a small
fraction of the spread, the conjunct did not decide"* — and its closing
diagnosis: *"nothing computes the ratio and prints it."*

WHAT THIS MODULE IS: that check, computed and printed. Nothing more.

WHAT IT IS NOT — deliberately. It declares NO cutoff and returns NO verdict.
Whether a given margin-over-spread ratio "decides" is a design decision that
belongs to each spec's pre-registration (and, for seats, to the `SO.10`
precedent: a result inside the noise margin leaves the seat VACANT rather
than handing it to the arm that happened to be ahead). A cutoff chosen here
would be a threshold picked by argument, once, for every future spec — the
move SYSTEM.md law 3 forbids. Callers embed the returned numbers in their row
and pre-register their own rule over them; a rule armed on this output is a
conjunct like any other and owes the `run blast-radius` line in its commit.

WHY THIS IS NOT IN `bakeoff.py`, where it conceptually belongs. `run_bakeoff`
already carries the margin discipline for winner-vs-runner-up (property 2,
`margin_sigma`). The uncovered class is the hand-rolled conjunct decided
against an IN-RUN ANCHOR — `T4.06`'s shape — which never passes through
`run_bakeoff`. And `experiments/bakeoff.py` is declared in the `IMPL_DEPS` of
a standing PASS certificate (`LG.13`), so adding even a pure reporter there
would stale a certificate to ship a printout. A new module moves no sha that
any certificate hashes.

USAGE, at the point where an anchor conjunct is about to be recorded:

    from ..resolution import anchor_margin
    rep = anchor_margin(challenger_r2_per_seed, incumbent_r2_per_seed,
                        higher_is_better=True, decided_at="min",
                        label="min_modality_latent_r2")
    metrics.update({f"resolution_{k}": v for k, v in rep.items()
                    if k != "line"})           # numbers into the row
    # rep["line"] has already been printed; the caller's PRE-REGISTERED rule,
    # not this module, then decides what the ratio means.

The fixture in `_selftest()` replays the committed `T4.06` row (attempt 1,
`159e165`) byte-for-byte and asserts both calibration points: the certified
winner at 6.9% of anchor spread with a regressing seed, and the refuted arm
at 8.2× that margin with every seed agreeing.

    /data/venvs/jackthelearner/bin/python -m experiments.resolution
"""
from __future__ import annotations

import statistics as st
import sys
from typing import Dict, List, Optional, Sequence

_AGG = {"min": min, "max": max, "mean": st.mean}


def anchor_margin(challenger_scores: Sequence[float],
                  anchor_scores: Sequence[float],
                  *,
                  higher_is_better: bool = True,
                  decided_at: str = "min",
                  label: str = "",
                  quiet: bool = False) -> Dict[str, object]:
    """Report how a challenger-vs-anchor conjunct resolved, per seed.

    `challenger_scores` / `anchor_scores` are the PER-SEED values of the one
    statistic the conjunct is decided on. `decided_at` names the aggregation
    the caller's rule actually uses ("min" for this project's gate-the-minimum
    discipline, "max", or "mean"). All returned margins are signed in the
    IMPROVING direction: positive means the challenger is better, whatever
    `higher_is_better` says.

    Returns a dict of plain numbers plus a formatted `line` (printed unless
    `quiet`):

      margin              improving-signed gap at the deciding aggregation
      anchor_spread       max − min of the anchor's own per-seed values — the
                          unit the 109th audit used for the 6.9% reading
      anchor_std          sample std of the same (0.0 when n < 2)
      margin_over_spread  margin / max(anchor_spread, 1e-9) — THE ratio
      n_seeds_paired      seeds compared pairwise, or None when the lists'
                          lengths differ (then the three paired fields are None)
      paired_diffs        improving-signed per-seed differences
      n_improving / n_regressing   strictly-positive / strictly-negative counts

    No verdict. See the module docstring for why.
    """
    if decided_at not in _AGG:
        raise ValueError(f"decided_at must be one of {sorted(_AGG)}, "
                         f"not {decided_at!r}")
    chal = [float(x) for x in challenger_scores]
    anch = [float(x) for x in anchor_scores]
    if not chal or not anch:
        raise ValueError("anchor_margin needs at least one score on each side")
    sign = 1.0 if higher_is_better else -1.0
    agg = _AGG[decided_at]
    margin = sign * (agg(chal) - agg(anch))
    spread = max(anch) - min(anch)
    std = st.stdev(anch) if len(anch) > 1 else 0.0
    ratio = margin / max(spread, 1e-9)

    paired: Optional[List[float]] = None
    n_up: Optional[int] = None
    n_down: Optional[int] = None
    if len(chal) == len(anch):
        paired = [round(sign * (c - a), 6) for c, a in zip(chal, anch)]
        n_up = sum(1 for d in paired if d > 0)
        n_down = sum(1 for d in paired if d < 0)

    head = f"{label}: " if label else ""
    if paired is None:
        seed_part = (f"UNPAIRED ({len(chal)} vs {len(anch)} seeds — "
                     f"per-seed agreement unreadable)")
    else:
        seed_part = (f"paired seeds: {n_up} improving, {n_down} "
                     + ("REGRESSING" if n_down else "regressing"))
    line = (f"RESOLUTION {head}margin {margin:+.4f} at {decided_at} over "
            f"anchor = {100.0 * ratio:+.1f}% of the anchor's own seed spread "
            f"{spread:.4f} (std {std:.4f}); {seed_part}")
    if not quiet:
        print(line)
    return {
        "margin": round(margin, 6),
        "anchor_spread": round(spread, 6),
        "anchor_std": round(std, 6),
        "margin_over_spread": round(ratio, 6),
        "n_seeds_paired": len(chal) if paired is not None else None,
        "paired_diffs": paired,
        "n_improving": n_up,
        "n_regressing": n_down,
        "line": line,
    }


def _selftest() -> int:
    """Replay the committed T4.06 row (attempt 1, `159e165`) as the fixture.

    Every literal below is copied from `ledger.json`'s T4.06 entry
    (`metrics.arms.<arm>.r2_per_seed`). The two assertions that matter are the
    109th audit's own calibration points, re-derived mechanically: the guard
    that read +6.9%/one-seed-regressing on the arm it certified read
    −57%/all-seeds-agreeing (8.2× the margin) on the arm it refuted.
    """
    incumbent = [-2.3939, -2.1417, -2.1240]
    loss_reweight = [-2.3752, -2.1311, -2.1252]
    grad_norm = [-2.5468, -2.3257, -2.2305]
    modality_dropout = [-2.4162, -2.6808, -2.5041]

    failures: List[str] = []

    def check(name: str, ok: bool, got: object) -> None:
        if not ok:
            failures.append(f"{name}: got {got!r}")

    win = anchor_margin(loss_reweight, incumbent,
                        label="T4.06 loss_reweight (certified)")
    check("winner margin", abs(win["margin"] - 0.0187) < 1e-9, win["margin"])
    check("anchor spread", abs(win["anchor_spread"] - 0.2699) < 1e-9,
          win["anchor_spread"])
    check("winner ratio 6.9%", abs(win["margin_over_spread"] - 0.0693) < 5e-4,
          win["margin_over_spread"])
    check("winner one seed regressing", win["n_regressing"] == 1
          and win["n_improving"] == 2, (win["n_improving"], win["n_regressing"]))

    ref = anchor_margin(grad_norm, incumbent,
                        label="T4.06 grad_norm (refuted)")
    check("refuter margin", abs(ref["margin"] - (-0.1529)) < 1e-9, ref["margin"])
    check("refuter all seeds regress", ref["n_regressing"] == 3, ref["n_regressing"])
    ratio_8x = abs(ref["margin"] / win["margin"])
    check("refuter decided at ~8.2x the winner's margin",
          8.1 < ratio_8x < 8.3, ratio_8x)

    drop = anchor_margin(modality_dropout, incumbent,
                         label="T4.06 modality_dropout (refuted)")
    check("dropout margin", abs(drop["margin"] - (-0.2869)) < 1e-9, drop["margin"])

    flipped = anchor_margin(loss_reweight, incumbent, higher_is_better=False,
                            quiet=True)
    check("lower-is-better flips the sign", abs(flipped["margin"] - (-0.0187)) < 1e-9,
          flipped["margin"])
    unpaired = anchor_margin(loss_reweight, incumbent[:2], quiet=True)
    check("unequal lengths -> paired fields None",
          unpaired["paired_diffs"] is None and unpaired["n_regressing"] is None,
          unpaired["paired_diffs"])
    try:
        anchor_margin(loss_reweight, incumbent, decided_at="median")
        check("bad decided_at raises", False, "no exception")
    except ValueError:
        pass

    if failures:
        print(f"SELFTEST FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("selftest: 8 checks, 0 failures — fixture is the committed "
          "T4.06 row (159e165)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
