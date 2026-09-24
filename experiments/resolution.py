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

AND ITS READER (110th audit FTB 2, same day): the module shipped with zero
callers, so `run status` now prints `status_lines()` — the same numbers for
every RECORDED row in the in-run-anchor inventory, derived from the ledger
(see `anchor_rows` for the one-line derivation) in the `UNBACKED
CERTIFICATES` idiom: unfloored, reporting-only, reddening nothing. A row the
inventory finds but `ANCHOR_CONJUNCTS` does not describe prints as
UNDESCRIBED rather than with a guessed direction.

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


# ---------------------------------------------------------------------------
# The reader (110th audit, FTB 2). `anchor_margin` shipped with zero callers —
# "the next bakeoff decided against an in-run anchor will be decided exactly
# as T4.06 was, unless a human remembers to invoke a module by hand." This is
# the caller: `run status` prints the block below for every RECORDED row the
# inventory derives. REPORTING-ONLY in the `UNBACKED CERTIFICATES` idiom —
# unfloored, no cutoff, no verdict, reddening nothing, refusing nothing.
# A rule armed on this output is a conjunct and owes `run blast-radius`.
# ---------------------------------------------------------------------------

ANCHOR_ARM_NAMES = ("incumbent", "anchor")

# THE DESCRIPTOR TABLE, and why it exists rather than auto-derivation. The
# INVENTORY is cheaply derivable from the record (a row whose `metrics.arms`
# dict carries an arm named in ANCHOR_ARM_NAMES — checked against the live
# ledger 2026-09-23: selects T4.06, excludes D1.0, whose arms have no anchor).
# The CONJUNCT is not: which per-seed statistic the `_check` compared, at
# which aggregation, and in which direction live in the spec's pre-registered
# rule, not in the row. Guessing the direction would print wrong-signed
# margins, so an undescribed anchor-relative conjunct is reported LOUDLY as
# UNDESCRIBED instead of being described wrongly. One LIST per spec, one
# descriptor per anchor-relative conjunct, copied from the spec's own
# pre-registration at the time its row first lands. This used to read "One
# entry per spec", and that sentence was the defect the 112th audit named:
# T4.06 pre-registered TWO anchor-relative conjuncts, one was described, and
# the row printed as fully described while conjunct (3) was printed by
# nothing. A descriptor may carry `strict: False` for a conjunct decided at
# `<=` with zero required margin — marked in the printed line because
# strictness is the one part of the rule a reader cannot infer from the row.
# A descriptor may instead carry `not_a_conjunct: <reason>` for a per-seed
# list the anchor arm records that is NOT decided against the anchor (an
# exogenous gate, a diagnostic) — described-as-not-a-conjunct is the honest
# third state between rendered and UNDESCRIBED, and the reason is copied
# from the spec's pre-registration like everything else here.
ANCHOR_CONJUNCTS: Dict[str, List[Dict[str, object]]] = {
    # t4_06_fusion_balancing_bakeoff: `_winner_lanes` is the pre-registered
    # rule. Its two anchor-relative conjuncts:
    # (2) s["min_r2"] > bar_r2 where bar_r2 = min over seeds of the
    #     incumbent's min-modality latent R^2 — strict.
    # (3) s["eval_loss_mean"] <= the incumbent's mean eval loss — NON-STRICT
    #     (`<=`, zero required margin; docstring line 94).
    # Conjunct (1), ratio_ok, is decided against the EXOGENOUS RATIO_MAX
    # (the 10x gate), not the anchor; the other two per-seed lists are
    # diagnostics that decide nothing.
    "T4.06": [
        {"stat": "r2_per_seed", "decided_at": "min",
         "higher_is_better": True, "label": "min_modality_latent_r2"},
        {"stat": "eval_loss_per_seed", "decided_at": "mean",
         "higher_is_better": False, "label": "eval_loss_mean",
         "strict": False},
        {"stat": "ratio_per_seed",
         "not_a_conjunct": "decided against the exogenous 10x gate "
                           "RATIO_MAX, not the anchor"},
        {"stat": "latent_r2_per_seed",
         "not_a_conjunct": "per-modality diagnostic behind min_r2; "
                           "decides nothing"},
        {"stat": "norms_per_seed",
         "not_a_conjunct": "raw per-modality grad norms; diagnostic only"},
    ],
}


def _field(row: object, name: str) -> object:
    """One row shape for dicts (raw ledger JSON, selftest fixtures) and
    `protocol.Result` objects (what `run status` actually holds)."""
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def anchor_rows(results) -> List[tuple]:
    """The in-run-anchor inventory: (spec_id, row, anchor_arm_name, arms).

    Derivation, named because FTB 2 asked for it: a recorded row belongs iff
    its `metrics.arms` is a dict containing an arm named in ANCHOR_ARM_NAMES.
    That is the whole rule — no spec list is consulted, so a future bakeoff
    that records an anchor arm enters this inventory the day its row lands.
    """
    out = []
    for sid in sorted(results):
        metrics = _field(results[sid], "metrics")
        arms = metrics.get("arms") if isinstance(metrics, dict) else None
        if not isinstance(arms, dict):
            continue
        anchor = next((a for a in ANCHOR_ARM_NAMES if a in arms), None)
        if anchor is None:
            continue
        out.append((sid, results[sid], anchor, arms))
    return out


def status_lines(results) -> List[str]:
    """The `run status` block. Empty list when the inventory is empty."""
    rows = anchor_rows(results)
    if not rows:
        return []
    lines = [
        f"  ? ANCHOR-DECIDED CONJUNCTS — {len(rows)} recorded row(s) decide a "
        "conjunct against an IN-RUN\n    anchor arm (derivation: metrics.arms "
        "carries an arm named 'incumbent'/'anchor').\n    Legal and "
        "REPORTING-ONLY: no cutoff, no verdict, nothing reddens — whether a "
        "margin\n    this size DECIDES stays with each spec's pre-registration "
        "(SO.10 vacancy precedent):"]
    for sid, row, anchor_name, arms in rows:
        status = _field(row, "status")
        status = getattr(status, "value", status)
        head = f"      {sid} ({status}, attempt {_field(row, 'attempt')})"
        descs = ANCHOR_CONJUNCTS.get(sid) or []
        conjuncts = [d for d in descs if "not_a_conjunct" not in d]
        non_conjuncts = [d for d in descs if "not_a_conjunct" in d]
        described = {str(d["stat"]) for d in descs}
        # The candidate anchor-relative conjuncts are derived from the ROW:
        # every per-seed list the anchor arm itself carries is a statistic
        # a challenger can be decided against. UNDESCRIBED fires per STAT,
        # not per row (112th audit item 4): a row with one described and one
        # undescribed conjunct used to print as fully described.
        anchor_stats = {k for k, v in arms.get(anchor_name, {}).items()
                        if isinstance(v, list)}
        for stat in sorted(anchor_stats - described):
            lines.append(
                f"{head}  anchor arm '{anchor_name}' records '{stat}' but "
                f"that conjunct is UNDESCRIBED —\n        its direction and "
                f"aggregation live in the spec's pre-registration, not the "
                f"row;\n        add the ANCHOR_CONJUNCTS descriptor rather "
                f"than letting this reader guess a sign.")
        noted = [d for d in non_conjuncts if str(d["stat"]) in anchor_stats]
        if noted:
            lines.append(
                f"{head}  recorded but NOT anchor-relative: "
                + "; ".join(f"{d['stat']} ({d['not_a_conjunct']})"
                            for d in noted))
        metrics = _field(row, "metrics") or {}
        winning = set(metrics.get("winning_arms") or [])
        refuted = set(metrics.get("refuted_arms") or [])
        for desc in conjuncts:
            stat = str(desc["stat"])
            anch_scores = arms.get(anchor_name, {}).get(stat)
            strictness = ("" if desc.get("strict", True)
                          else "  [non-strict <=, zero required margin]")
            lines.append(f"{head}  {desc['label']} vs '{anchor_name}' at "
                         f"{desc['decided_at']}:{strictness}")
            for arm_name in sorted(arms):
                if arm_name == anchor_name:
                    continue
                chal = arms[arm_name].get(stat)
                if not (isinstance(chal, list)
                        and isinstance(anch_scores, list)):
                    lines.append(f"        {arm_name:<16s} '{stat}' missing "
                                 f"on one side — unreadable from the row")
                    continue
                rep = anchor_margin(
                    chal, anch_scores, quiet=True,
                    higher_is_better=bool(desc["higher_is_better"]),
                    decided_at=str(desc["decided_at"]))
                tag = ("CERTIFIED" if arm_name in winning
                       else "refuted" if arm_name in refuted else "recorded")
                if rep["paired_diffs"] is None:
                    seeds = "seeds unpaired"
                else:
                    seeds = (f"seeds {rep['n_improving']} improving / "
                             f"{rep['n_regressing']} "
                             + ("REGRESSING" if rep["n_regressing"]
                                else "regressing"))
                lines.append(
                    f"        {arm_name:<16s} {tag:<9s} margin "
                    f"{rep['margin']:+.4f} = "
                    f"{100.0 * rep['margin_over_spread']:+.1f}% "
                    f"of anchor seed spread {rep['anchor_spread']:.4f}; "
                    f"{seeds}")
    return lines


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
    n_checks = 0

    def check(name: str, ok: bool, got: object) -> None:
        nonlocal n_checks
        n_checks += 1
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

    # The reader (110th audit FTB 2), on a fixture shaped exactly like the
    # committed rows: T4.06's arms (same literals as above) must enter the
    # inventory and re-derive both calibration points in the printed block;
    # D1.0's arms dict carries NO anchor-named arm and must be excluded; an
    # anchor row with no ANCHOR_CONJUNCTS entry must be reported UNDESCRIBED
    # rather than described with a guessed sign.
    def _arms(scores):
        return {n: {"r2_per_seed": s} for n, s in scores.items()}
    # The eval-loss literals are also copied from the committed T4.06 row —
    # conjunct (3), the one the row's first descriptor table never printed.
    eval_loss = {"incumbent": [0.4842, 0.4943, 0.49],
                 "loss_reweight": [0.4831, 0.4936, 0.4891],
                 "grad_norm": [0.4973, 0.4938, 0.4989],
                 "modality_dropout": [0.5047, 0.5027, 0.5159]}
    t406_arms = _arms({"incumbent": incumbent,
                       "loss_reweight": loss_reweight,
                       "grad_norm": grad_norm,
                       "modality_dropout": modality_dropout})
    for n, s in eval_loss.items():
        t406_arms[n]["eval_loss_per_seed"] = s
    # Committed literals again: the anchor also records the exogenous-gated
    # ratio, which must render as described-not-anchor-relative, not as
    # UNDESCRIBED and not with a guessed margin.
    t406_arms["incumbent"]["ratio_per_seed"] = [29.8302, 13.6677, 27.7323]
    fixture = {
        "T4.06": {"status": "PASS", "attempt": 1, "metrics": {
            "arms": t406_arms,
            "winning_arms": ["loss_reweight"],
            "refuted_arms": ["grad_norm"]}},
        "D1.0": {"status": "VOID", "attempt": 2, "metrics": {
            "arms": _arms({"aprime": [0.1], "c_e2e": [0.2]})}},
        "X.99": {"status": "FAIL", "attempt": 1, "metrics": {
            "arms": _arms({"anchor": [0.0], "arm_a": [0.1]})}},
    }
    inv = [sid for sid, *_ in anchor_rows(fixture)]
    check("inventory selects anchor rows, excludes D1.0",
          inv == ["T4.06", "X.99"], inv)
    block = "\n".join(status_lines(fixture))
    check("block re-derives the certified point (+6.9%, 1 REGRESSING)",
          "+6.9% of anchor seed spread 0.2699" in block
          and "1 REGRESSING" in block, block)
    check("block re-derives the refuted point (-56.7%)",
          "-56.7% of anchor seed spread 0.2699" in block, block)
    def _line_with(text: str, word: str) -> str:
        return next((l for l in text.splitlines() if word in l), "")
    check("winner/refuter tagged from the row's own verdict fields",
          "CERTIFIED" in _line_with(block, "loss_reweight")
          and "refuted" in _line_with(block, "grad_norm")
          and "recorded" in _line_with(block, "modality_dropout"), block)
    check("undescribed anchor row reported, not guessed",
          "UNDESCRIBED" in block and "X.99" in block, block)
    # 112th audit item 4: BOTH T4.06 conjuncts must render. Conjunct (3) is
    # decided at the mean, lower-is-better, and NON-STRICT — the marker and
    # the +8.9%-of-spread / 3-improving reading are the parts the 112th
    # audit found printed by nothing.
    t406_lines = [l for l in block.splitlines() if "T4.06" in l]
    check("both T4.06 conjunct headers render",
          any("min_modality_latent_r2" in l for l in t406_lines)
          and any("eval_loss_mean" in l for l in t406_lines), t406_lines)
    eloss_head = _line_with(block, "eval_loss_mean")
    check("conjunct (3) marked non-strict in the printed line",
          "[non-strict <=, zero required margin]" in eloss_head, eloss_head)
    eloss_idx = block.splitlines().index(eloss_head)
    eloss_block = "\n".join(block.splitlines()[eloss_idx:eloss_idx + 4])
    check("conjunct (3) re-derives +8.9% of anchor spread 0.0101, 3 improving",
          "+8.9% of anchor seed spread 0.0101" in eloss_block
          and "3 improving / 0 regressing" in eloss_block, eloss_block)
    # T4.06 is now FULLY described, so it must fire no UNDESCRIBED line —
    # only X.99's genuinely undescribed conjunct does.
    check("described row fires no UNDESCRIBED",
          not any("UNDESCRIBED" in l for l in t406_lines), t406_lines)
    # The third state: a stat the anchor records but the pre-registration
    # decides elsewhere (the exogenous 10x gate) renders as described-not-
    # anchor-relative, with no margin computed against the anchor for it.
    ratio_line = _line_with(block, "ratio_per_seed")
    check("exogenous-gated stat described, not UNDESCRIBED, no margin",
          "NOT anchor-relative" in ratio_line and "RATIO_MAX" in ratio_line
          and "margin" not in ratio_line, ratio_line)
    check("empty inventory prints nothing",
          status_lines({"T0.01": {"status": "PASS", "metrics": {}}}) == [], "")

    if failures:
        print(f"SELFTEST FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"selftest: {n_checks} checks, 0 failures — fixture is the "
          "committed T4.06 row (159e165)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
