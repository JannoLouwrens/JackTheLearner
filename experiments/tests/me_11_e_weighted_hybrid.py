"""ME.11.E — Arm E, the weighted hybrid, settled WITHOUT instantiating it.

VOID-FORECLOSED: the beat-both-parents gate is arithmetically unreachable.
    The lexical parent (Arm B, ledger row 2026-08-31) measured paraphrase
    recall@1 0.0000 on 160 cues x 3 seeds, and this run re-measures the
    stronger fact that forecloses fusion itself: on the certified
    stem-disjoint fixture the BM25 score of every cue against its own gold
    events is exactly 0.0 (probe 2026-08-31, seeds 0/1/2), so the lexical
    channel gives gold ZERO and non-gold >= 0 at every convex weight — fused
    recall <= the dense parent's recall at every w < 1, and at w = 1 the
    hybrid IS the dense parent. "Beats both parents" therefore requires
    beating the dense parent using a signal that can only demote its gold.
    The abstention half is equally closed: every dense scorer the family
    measured (C, D, and three variants) is INFEASIBLE (tau_fpr > tau_cov on
    3/3 seeds), a property of the score distributions being fused, not of
    the fusion weight, and the best unthresholded dense ceiling is 0.250
    against the family's 0.80 bar.

FORECLOSURE ARITHMETIC: no multiplier converges. The fused score at weight w
    is w*dense + (1-w)*lex (TMM-normalised, per the spec's own notes).
    lex(cue, gold) = 0.0 measured on every seed — certified stem-disjointness
    leaves the stemmed, stopword-stripped query no term in common with its
    target — while lex(cue, non-gold) >= 0, so max over w of fused recall@1
    equals the dense parent's own 0.0667. More events or seeds tighten the
    conformal quantiles toward the same population tau_fpr > tau_cov gap;
    no N moves lex(cue, gold) off zero, because the zero is the fixture's
    certified construction (ME.11.0: stem_leak_cues 0/160), not noise.

BLAST RADIUS: none — no registered spec declares depends_on ME.11.E or
    ME.11.F (verified by grep over registry.py and registry_expansion.py,
    2026-08-31); the parent ME.11 depends on ME.1 and ME.11.0 only and
    remains runnable.

WHAT THIS RUN IS. It does NOT build the hybrid. It buys the family's ledger
row for Arm E by verifying, live and per seed, every conjunct of the
arithmetic above, and records VOID — the pre-registered "run did not test
the claim" status — because no configuration of the arm can produce a
different family verdict. `_check` has exactly two outcomes:

  Status.VOID  — every foreclosing conjunct held: the lexical channel scored
                 its gold at 0.0 on every seed, the recorded parent rows say
                 what the declaration above says they say, and the leaky-cue
                 aliveness control proves the instrument that read those
                 zeros was alive.
  raise        — any conjunct failed. The runner records a loud ERROR, never
                 a quiet VOID: a refuted foreclosure means the declaration
                 above must be DELETED and the arm implemented for real, per
                 the registry.

WHY THERE IS NO `_control`. The registry's control (evaluate at w=0 and w=1;
a fitted w within noise of an endpoint is one parent in a costume) sabotages
a mechanism this run deliberately never instantiates; running it would
require building the arm the foreclosure exists to spare. The obligation the
control exists to discharge — proof the instrument behind a 0.0000 reading
was alive (LESSONS: "an at-chance control must carry proof its instrument
was alive") — rides inside `_experiment` instead: the same index that scores
the real cues at zero must score the deliberately-leaky cues >= 0.80, every
seed, or `_check` refuses to certify the foreclosure.

GATES, frozen before the run (probe 2026-08-31: lex_recall@1 0.0/0.0/0.0,
lex_gold_score_max 0.0/0.0/0.0, leaky 1.0/1.0/1.0 on seeds 0/1/2):

  1. `instrument_alive` == 1.0 — leaky lexical recall >= 0.80 every seed.
  2. `lex_zero` == 1.0 — recall@1 == 0.0 AND gold-score-max == 0.0, every
     seed, measured live by Arm B's own committed index code.
  3. `parents_replayed_ok` == 1.0 — the ledger rows this settlement stands
     on still say what it cites: ME.11.B recall 0.0000; ME.11.C and ME.11.D
     both feasible_ok 0.0 with tau_fpr > tau_cov; best dense unthresholded
     ceiling < the 0.80 family bar. If any parent is ever re-run to a
     different answer, this run ERRORS rather than re-certifying a stale
     foreclosure.
"""
from __future__ import annotations

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .me_11_b_bm25s_stemming import _Bm25Index, _bm25_recall, _compat

IMPL_DEPS = ["experiments/fixtures/paraphrase_eval.py",
             "experiments/tests/me_11_b_bm25s_stemming.py"]

FAMILY_BAR = 0.80              # ME.11's own hypothesis: recall >= 80%
MIN_LEAKY_RECALL = 0.80        # aliveness floor, same bar ME.11.0 used
DENSE_PARENT_IDS = ("ME.11.C", "ME.11.D")
LEX_PARENT_ID = "ME.11.B"


def _replayed() -> dict:
    """The recorded parent rows the foreclosure stands on, re-read from the
    ledger at run time — never hard-coded, so a re-run parent that changed
    its answer breaks this settlement loudly instead of being ignored."""
    rows = Ledger().results
    out = {}
    b = rows.get(LEX_PARENT_ID)
    out["b_recall_row"] = (b.metrics.get("paraphrase_recall_at_1", -1.0)
                           if b else -1.0)
    best_recall, best_ceiling, feasible_any, tau_gap_min = -1.0, -1.0, 0.0, None
    for sid in DENSE_PARENT_IDS:
        r = rows.get(sid)
        if r is None:
            return {**out, "dense_rows_present": 0.0}
        m = r.metrics
        best_recall = max(best_recall, m.get("paraphrase_recall_at_1", -1.0))
        best_ceiling = max(best_ceiling, m.get("recall_unthresholded", -1.0))
        feasible_any = max(feasible_any, m.get("feasible_ok", 1.0))
        gap = m.get("tau_fpr", 0.0) - m.get("tau_cov", 1.0)
        tau_gap_min = gap if tau_gap_min is None else min(tau_gap_min, gap)
    out.update({"dense_rows_present": 1.0,
                "dense_best_recall_row": round(best_recall, 4),
                "dense_best_ceiling_row": round(best_ceiling, 4),
                "dense_feasible_any_row": feasible_any,
                "dense_tau_gap_min_row": round(tau_gap_min, 4)})
    return out


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    idx = _Bm25Index(texts)
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    # The foreclosing facts, measured live by Arm B's own committed code:
    # top-1 recall, and the maximum BM25 score any gold event achieves for
    # its own cue at full retrieval depth. Zero at full depth means the
    # lexical channel contributes nothing to gold at ANY fusion weight.
    lex_recall = _bm25_recall(idx, events, headline)["recall"]
    gold_max = 0.0
    for c in headline:
        gold = set(c["gold"])
        for d, s in idx.query(c["text"], k=len(texts)):
            if d in gold:
                gold_max = max(gold_max, s)

    # Aliveness: the index that read those zeros must ace the leaky twin.
    hits = 0
    for c in fx["leaky_cues"]:
        for d, _s in idx.query(c["text"]):
            if _compat(events[d], c.get("speaker"), c.get("channel")):
                hits += d in c["gold"]
                break
    leaky = hits / max(1, len(fx["leaky_cues"]))

    rep = _replayed()
    parents_ok = 1.0 if (
        rep.get("dense_rows_present", 0.0) >= 1.0
        and rep["b_recall_row"] == 0.0
        and rep["dense_feasible_any_row"] == 0.0
        and rep["dense_tau_gap_min_row"] > 0.0
        and 0.0 <= rep["dense_best_ceiling_row"] < FAMILY_BAR) else 0.0

    return {
        "lex_recall_at_1": round(lex_recall, 4),
        "lex_gold_score_max": round(gold_max, 6),
        "lex_zero": 1.0 if (lex_recall == 0.0 and gold_max == 0.0) else 0.0,
        "leaky_recall": round(leaky, 4),
        "instrument_alive": 1.0 if leaky >= MIN_LEAKY_RECALL else 0.0,
        "parents_replayed_ok": parents_ok,
        **rep,
        "family_bar": FAMILY_BAR,
        "headline_cues": float(len(headline)),
        "fixture_hash_seed_only": fx["hash"],
    }


def _check(m: dict, c: dict):
    from ..protocol import Status
    if m["instrument_alive"] < 1.0:
        raise RuntimeError(
            "ME.11.E foreclosure NOT certified: the lexical index failed its "
            f"leaky-cue aliveness floor (leaky_recall {m['leaky_recall']}) — "
            "a dead instrument's zeros prove nothing. Fix the rig; do not "
            "record this settlement.")
    if m["lex_zero"] < 1.0:
        raise RuntimeError(
            "ME.11.E foreclosure REFUTED: the lexical channel scored gold "
            f"(recall@1 {m['lex_recall_at_1']}, gold_score_max "
            f"{m['lex_gold_score_max']}). Delete the VOID-FORECLOSED "
            "declaration in this file and implement the arm per the registry.")
    if m["parents_replayed_ok"] < 1.0:
        raise RuntimeError(
            "ME.11.E foreclosure REFUTED: a parent ledger row no longer says "
            "what the declaration cites (B recall "
            f"{m.get('b_recall_row')}, dense feasible_any "
            f"{m.get('dense_feasible_any_row')}, dense ceiling "
            f"{m.get('dense_best_ceiling_row')}). Re-derive the arithmetic "
            "before trusting this settlement.")
    return Status.VOID    # foreclosed: no run of the arm can test the claim


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11.E"], _experiment, _check, ledger=ledger)
