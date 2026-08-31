"""ME.11.F — Arm F, the cascade, settled WITHOUT instantiating the reranker.

VOID-FORECLOSED: the cascade's first stage caps it below the family bar by
    arithmetic. The premise in the registry — "Arm C retrieves top-50 (pilot
    recall@10 was 1.000, so the answer is present)" — is falsified on the
    certified stem-disjoint fixture: Arm C's unthresholded recall@50,
    measured 2026-08-31 with Arm C's own committed index code and re-measured
    live by this run, is 0.475/0.381/0.463 on seeds 0/1/2 (mean 0.44). A
    reranker permutes the candidate list; it never adds to it, so a PERFECT
    reranker's recall@1 is bounded by recall@50 — 0.44 against the ME.11
    family bar of 0.80. And the arm's own pre-registered control pins the
    abstention decision byte-identically to Arm C's first-stage conformal
    threshold, which the recorded Arm C row measured INFEASIBLE (tau_fpr
    0.365 > tau_cov 0.184, 3/3 seeds), so the registered metric — recall at
    certified abstention — is capped by a threshold arithmetic no rerank
    quality can touch.

FORECLOSURE ARITHMETIC: no multiplier converges. Raising k raises the
    unthresholded candidate ceiling toward Arm C's recall@all, but the
    registered metric is paraphrase recall AT certified abstention >= 0.95,
    and by the arm's own control that decision belongs to Arm C's first
    stage: tau_fpr and tau_cov are order statistics of the first-stage score
    distributions on 300+300 negatives and do not move with k. The best
    seed's candidate ceiling (0.475) is 1.7x short of the 0.80 bar even
    before the infeasible threshold applies; larger corpora tighten the
    conformal quantiles toward the same population gap.

BLAST RADIUS: none — no registered spec declares depends_on ME.11.F or
    ME.11.E (verified by grep over registry.py and registry_expansion.py,
    2026-08-31); the parent ME.11 depends on ME.1 and ME.11.0 only and
    remains runnable.

WHAT THIS RUN IS. It does NOT build the cross-encoder cascade. It buys the
family's ledger row for Arm F by verifying the arithmetic above live, per
seed, and records VOID — the pre-registered "run did not test the claim"
status — because no rerank quality can produce a different family verdict.
`_check` has exactly two outcomes:

  Status.VOID  — every foreclosing conjunct held: the first stage's
                 recall@50 sits below the family bar on every seed, the
                 recorded Arm C row still says its threshold is INFEASIBLE,
                 and the leaky-cue aliveness control proves the index that
                 read those numbers was alive.
  raise        — any conjunct failed. The runner records a loud ERROR, never
                 a quiet VOID: a refuted foreclosure means the declaration
                 above must be DELETED and the arm implemented for real, per
                 the registry.

WHY THERE IS NO `_control`. The registry's control (the reranker must not
change the abstention decision) sabotages a reranker this run deliberately
never instantiates. The obligation it discharges — proof the instrument
behind a low reading was alive (LESSONS: "an at-chance control must carry
proof its instrument was alive") — rides inside `_experiment`: the same
dense index that misses gold on 56% of real cues must place the leaky
twin's gold at rank 1 on >= 0.80 of leaky cues, every seed, or `_check`
refuses to certify the foreclosure.

GATES, frozen before the run (probe scripts/probe_me11c_recall_at_k.py,
2026-08-31: recall@50 0.475/0.381/0.463, recall@10 0.294/0.238/0.306):

  1. `instrument_alive` == 1.0 — leaky top-1-in-gold recall >= 0.80 every
     seed, unthresholded, same index that scores the real cues.
  2. `cap_below_bar` == 1.0 — recall@50 < 0.80 on every seed, measured live
     with Arm C's committed model/index code (no private code path).
  3. `c_row_replayed_ok` == 1.0 — the recorded ME.11.C row still measures
     feasible_ok 0.0 with tau_fpr > tau_cov. If Arm C is ever re-run to a
     feasible threshold, this run ERRORS rather than re-certifying a stale
     foreclosure.
"""
from __future__ import annotations

import numpy as np

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .me_11_c_static_embeddings import MODEL, _DenseIndex, _load_model, _Prov

IMPL_DEPS = ["experiments/fixtures/paraphrase_eval.py",
             "experiments/tests/me_11_c_static_embeddings.py"]

FAMILY_BAR = 0.80              # ME.11's own hypothesis: recall >= 80%
MIN_LEAKY_RECALL = 0.80        # aliveness floor, same bar ME.11.0 used
TOP_K = 50                     # the cascade's own first-stage depth
FIRST_STAGE_ID = "ME.11.C"

_MODEL_CACHE: dict = {}


def _model():
    if "m" not in _MODEL_CACHE:
        _MODEL_CACHE["m"] = _load_model(MODEL)
    return _MODEL_CACHE["m"]


def _replayed() -> dict:
    """The recorded first-stage row the foreclosure stands on, re-read from
    the ledger at run time — never hard-coded."""
    r = Ledger().results.get(FIRST_STAGE_ID)
    if r is None:
        return {"c_row_present": 0.0}
    m = r.metrics
    gap = m.get("tau_fpr", 0.0) - m.get("tau_cov", 1.0)
    return {"c_row_present": 1.0,
            "c_recall_row": m.get("paraphrase_recall_at_1", -1.0),
            "c_feasible_ok_row": m.get("feasible_ok", 1.0),
            "c_tau_gap_row": round(gap, 4)}


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    prov = _Prov(events)
    idx = _DenseIndex(_model(), texts)
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    # The foreclosing fact, live: how often the answer is IN the first
    # stage's top-50 at all. Recall@k here is unthresholded and provenance-
    # filtered, exactly the candidate set a reranker would receive.
    hits = {1: 0, 10: 0, TOP_K: 0}
    for c in headline:
        mask = prov.mask(c.get("speaker"), c.get("channel"))
        q = idx.embed_query(c["text"])
        sims = idx.mat @ q
        sims[~mask] = -np.inf
        order = np.argsort(-sims)
        gold = set(c["gold"])
        for k in hits:
            if gold & set(int(i) for i in order[:k]):
                hits[k] += 1
    n = max(1, len(headline))
    r50 = hits[TOP_K] / n

    # Aliveness: the same index must put the leaky twin's gold at rank 1.
    alive = 0
    for c in fx["leaky_cues"]:
        d, _s = idx.top1(c["text"], prov.mask(c.get("speaker"),
                                              c.get("channel")))
        alive += d in c["gold"]
    leaky = alive / max(1, len(fx["leaky_cues"]))

    rep = _replayed()
    c_ok = 1.0 if (rep.get("c_row_present", 0.0) >= 1.0
                   and rep["c_feasible_ok_row"] == 0.0
                   and rep["c_tau_gap_row"] > 0.0) else 0.0

    return {
        "recall_at_50": round(r50, 4),
        "recall_at_10": round(hits[10] / n, 4),
        "recall_at_1": round(hits[1] / n, 4),
        "cap_below_bar": 1.0 if r50 < FAMILY_BAR else 0.0,
        "leaky_recall": round(leaky, 4),
        "instrument_alive": 1.0 if leaky >= MIN_LEAKY_RECALL else 0.0,
        "c_row_replayed_ok": c_ok,
        **rep,
        "family_bar": FAMILY_BAR,
        "headline_cues": float(n),
        "fixture_hash_seed_only": fx["hash"],
    }


def _check(m: dict, c: dict):
    from ..protocol import Status
    if m["instrument_alive"] < 1.0:
        raise RuntimeError(
            "ME.11.F foreclosure NOT certified: the dense index failed its "
            f"leaky-cue aliveness floor (leaky_recall {m['leaky_recall']}) — "
            "a dead instrument's misses prove nothing. Fix the rig; do not "
            "record this settlement.")
    if m["cap_below_bar"] < 1.0:
        raise RuntimeError(
            "ME.11.F foreclosure REFUTED: the first stage's recall@50 "
            f"({m['recall_at_50']}) reaches the 0.80 family bar on some seed "
            "— a perfect reranker is no longer capped below the gate. Delete "
            "the VOID-FORECLOSED declaration in this file and implement the "
            "arm per the registry.")
    if m["c_row_replayed_ok"] < 1.0:
        raise RuntimeError(
            "ME.11.F foreclosure REFUTED: the recorded ME.11.C row no longer "
            "measures an infeasible threshold (feasible_ok "
            f"{m.get('c_feasible_ok_row')}, tau gap {m.get('c_tau_gap_row')})."
            " Re-derive the arithmetic before trusting this settlement.")
    return Status.VOID    # foreclosed: no run of the arm can test the claim


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11.F"], _experiment, _check, ledger=ledger)
