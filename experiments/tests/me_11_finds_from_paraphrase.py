"""ME.11 — the FAMILY VERDICT: finds the memory from a paraphrase, still
never invents one. Measured, and the answer is NO — recorded as an honest
FAIL, not narrated around.

WHAT THIS RUN IS. ME.11's claim was measured by its own bakeoff family on
the certified stem-disjoint fixture (ME.11.0, oracle ceiling 1.0, lexical
null 0.0 by construction): six arms, every one settled on the ledger.

    A  lexical incumbent   PASS (honest-and-useless): recall 0.0000,
                           abstention 1.0 — quantified, exactly as specced
    B  BM25S + stemming    FAIL: recall 0.0000 (stemming buys nothing on
                           certified stem-disjoint cues)
    C  static embeddings   FAIL: recall 0.0437, tau INFEASIBLE
                           (tau_fpr 0.365 > tau_cov 0.184, 3/3 seeds)
    D  MiniLM-L6 (ONNX)    FAIL: recall 0.0667, ceiling 0.250 unthresholded,
                           tau INFEASIBLE (0.388 > 0.227, 3/3 seeds)
    E  weighted hybrid     VOID-FORECLOSED: lexical channel scores gold 0.0,
                           fusion cannot exceed its dense parent at any w
    F  cascade + rerank    VOID-FORECLOSED: first-stage recall@50 0.44 caps
                           a perfect reranker below the 0.80 bar

The family's best measured paraphrase recall at the registry's certified
abstention is 0.0667, its best unthresholded ceiling 0.250, against the
hypothesis's 0.80 — and every dense scorer measured bought what recall it
has with credulity (certified abstention infeasible at alpha=0.05), which
the falsified_by clause names explicitly. The bars are the registry's and
do not move (Review 2026-09-02, item 2: implement the family verdict
against the rows already on the ledger; the family's redesign disposition
belongs to the 09-06 Review, not to this run).

WHY FAIL AND NOT VOID. VOID means "the run did not test the claim". The
family DID test it: rig alive on every row (leaky cues ace, oracle ceiling
1.0, controls collapsed where they must), verdicts settled, foreclosures
certified live by E and F. A claim tested and lost is a FAIL. The point
this row buys: the GOAL.md memory commitment behind ME.11 moves from
*unmeasured* to *measured*, which no registration can do.

WHAT IS RUN LIVE (so the verdict is a measurement, not a summary):
  1. The family's BEST configuration — Arm D's committed encoder (MiniLM
     fp32, mean pooling) through the family's shared `_score_config`
     pipeline, per seed on the same fixture. This re-buys the deciding row
     with today's code; if it no longer reproduces, the run raises rather
     than citing a stale number.
  2. The registry's DISTRACTOR control (see `_control` below).
  3. Aliveness, both channels: the lexical index must ace the leaky twin
     (Arm B's committed code) AND the dense index must ace it
     unthresholded — a dead instrument's near-zeros prove nothing, so a
     dead rig returns VOID, never FAIL.
  4. VERBATIM: every answered cue's returned text must be byte-identical
     to the stored record. Structural for a quoting retriever — asserted
     so a future pipeline that paraphrases its output cannot inherit this
     row.
  5. Row replay: the six family rows are re-read from the ledger at run
     time and PINNED. If any parent is ever re-run to a different answer,
     this run ERRORS loudly — the verdict must be re-derived, never
     re-certified stale (ME.11.E's precedent).

THE CONTROL (registry): a DISTRACTOR store — each cue's true target is
masked out of the candidate set while the other 4,999 events of the same
life (topically similar by construction) remain, tau calibrated exactly as
the experiment's. The retriever must abstain rather than return the
nearest neighbour: semantic matching makes confabulation EASIER, so the
abstention floor is the thing under test, not the recall. Its answer rate
in the distractor store is the "invents one" rate, and the claim branch
requires distractor abstention >= 0.95.

GATES, frozen before the run (constants below; family numbers quoted from
the ledger rows of 2026-08-09..08-31):
  VOID   — instrument_alive < 1.0 on either channel (dead rig).
  raise  — rows_replayed_ok < 1.0 (a parent changed its answer), or the
           live Arm D re-measure diverges from its row mean by more than
           ROW_REPRO_TOL (the deciding row no longer reproduces).
  else   — PASS iff the family recorded a clearing arm AND the live
           re-measure clears the bar at certified, feasible abstention AND
           verbatim holds AND the distractor store forces abstention.
           On the rows as they stand this conjunction is False: FAIL.

BLAST RADIUS: no registered spec declares depends_on ME.11 (grep over
registry.py + registry_expansion.py, 2026-09-02). T2.10's hypothesis cites
the ME.11 fixture and the family's measured weakness by name; it depends on
this fixture's honesty (ME.11.0, PASS), not on this verdict.
"""
from __future__ import annotations

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, Status, run_spec
from ..registry import BY_ID
from .me_11_b_bm25s_stemming import _Bm25Index, _bm25_recall, _compat
from .me_11_c_static_embeddings import (ALPHA, _conformal_tau, _DenseIndex,
                                        _Prov, _score_config)
from .me_11_d_minilm_onnx import _OnnxEncoder, _session, _tokenizer
from .me_11_d_minilm_onnx import ONNX_FP32, REPO

IMPL_DEPS = ["experiments/fixtures/paraphrase_eval.py",
             "experiments/tests/me_11_b_bm25s_stemming.py",
             "experiments/tests/me_11_c_static_embeddings.py",
             "experiments/tests/me_11_d_minilm_onnx.py"]

FAMILY_BAR = 0.80          # ME.11's own hypothesis: paraphrase recall >= 80%
MIN_ABSTENTION = 0.95      # the hypothesis's own abstention floor
MIN_LEAKY_RECALL = 0.80    # aliveness floor, same bar ME.11.0 used
ROW_REPRO_TOL = 0.05       # live Arm D mean vs its row mean, else raise
ARM_IDS = ("ME.11.A", "ME.11.B", "ME.11.C", "ME.11.D")
FORECLOSED_IDS = ("ME.11.E", "ME.11.F")

_IDX: dict = {}            # per-seed index cache: experiment and control read
                           # the SAME instrument, which is the point


def _dense_index(seed: int, texts: list[str]) -> _DenseIndex:
    if seed not in _IDX:
        enc = _OnnxEncoder(_session(REPO, ONNX_FP32), _tokenizer(REPO),
                           pooling="mean")
        _IDX[seed] = _DenseIndex(enc, texts)
    return _IDX[seed]


def _replayed() -> dict:
    """The recorded family rows the verdict stands on, re-read from the
    ledger at run time — never hard-coded, so a re-run arm that changed its
    answer breaks this verdict loudly instead of being ignored."""
    rows = Ledger().results
    got = {sid: rows.get(sid) for sid in ARM_IDS + FORECLOSED_IDS}
    if any(r is None for r in got.values()):
        return {"rows_present": 0.0, "rows_replayed_ok": 0.0}
    a, b, c, d = (got[s] for s in ARM_IDS)
    e, f = (got[s] for s in FORECLOSED_IDS)

    recalls = [r.metrics.get("paraphrase_recall_at_1", -1.0)
               for r in (a, b, c, d)]
    ceiling = max(c.metrics.get("recall_unthresholded", -1.0),
                  d.metrics.get("recall_unthresholded", -1.0))
    cleared = any(
        r.metrics.get("paraphrase_recall_at_1", -1.0) >= FAMILY_BAR
        and r.metrics.get("abstention_certify", 0.0) >= MIN_ABSTENTION
        and r.metrics.get("feasible_ok", 1.0) >= 1.0
        for r in (a, b, c, d))

    pinned = (a.status == Status.PASS
              and a.metrics.get("paraphrase_recall_at_1") == 0.0
              and b.status == Status.FAIL
              and b.metrics.get("paraphrase_recall_at_1") == 0.0
              and c.status == Status.FAIL
              and c.metrics.get("feasible_ok", 1.0) == 0.0
              and d.status == Status.FAIL
              and d.metrics.get("feasible_ok", 1.0) == 0.0
              and e.status == Status.VOID
              and f.status == Status.VOID
              and max(recalls) < FAMILY_BAR
              and ceiling < FAMILY_BAR
              and not cleared)

    return {"rows_present": 1.0,
            "rows_replayed_ok": 1.0 if pinned else 0.0,
            "family_best_recall_row": round(max(recalls), 4),
            "family_best_ceiling_row": round(ceiling, 4),
            "family_cleared_row": 1.0 if cleared else 0.0,
            "d_recall_row": round(
                d.metrics.get("paraphrase_recall_at_1", -1.0), 4)}


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    prov = _Prov(events)
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    # Lexical channel, live: the null the family started from, and the
    # aliveness twin — measured by Arm B's own committed code.
    bidx = _Bm25Index(texts)
    lex_recall = _bm25_recall(bidx, events, headline)["recall"]
    hits = 0
    for c in fx["leaky_cues"]:
        for d, _s in bidx.query(c["text"]):
            if _compat(events[d], c.get("speaker"), c.get("channel")):
                hits += d in c["gold"]
                break
    leaky = hits / max(1, len(fx["leaky_cues"]))

    # The family's best arm, re-measured live through the family's shared
    # pipeline — the same code path that produced the deciding row.
    idx = _dense_index(seed, texts)
    r = _score_config(idx, prov, fx)

    # VERBATIM: every answered cue must return a byte-identical stored
    # record. Quoting retrievers satisfy this by construction; asserted so
    # a generating successor cannot inherit the row.
    answered, verbatim = 0, 0
    for c in headline:
        d, s = idx.top1(c["text"], prov.mask(c.get("speaker"),
                                             c.get("channel")))
        if d >= 0 and s > r["tau_fpr"]:
            answered += 1
            verbatim += texts[d] == events[d]["text"] and texts[d] in set(texts)

    return {
        "paraphrase_recall_at_1": r["recall"],
        "recall_unthresholded": r["recall_unthresholded"],
        "abstention_certify": r["abstention"],
        "tau_fpr": r["tau_fpr"],
        "tau_cov": r["tau_cov"],
        "feasible_ok": r["feasible"],
        "lex_null_recall": round(lex_recall, 4),
        "leaky_recall": round(leaky, 4),
        "instrument_alive": 1.0 if leaky >= MIN_LEAKY_RECALL else 0.0,
        "answered_cues": float(answered),
        "verbatim_ok": 1.0 if answered == verbatim else 0.0,
        **_replayed(),
        "family_bar": FAMILY_BAR,
        "headline_cues": float(len(headline)),
        "fixture_hash_seed_only": fx["hash"],
    }


def _control(seed: int) -> dict:
    """The registry's DISTRACTOR store: each cue's true target is masked
    out of the candidate set; the topically-similar rest of the life
    remains; tau calibrated exactly as the experiment's. Answering here IS
    inventing — the rate is the confabulation the claim forbids. Also
    carries the dense aliveness twin: the same index must ace the leaky
    cues unthresholded, or the near-zeros upstairs prove nothing."""
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    prov = _Prov(events)
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    idx = _dense_index(seed, texts)
    tune_scores = [
        idx.top1(neg["text"], prov.mask(neg.get("speaker"),
                                        neg.get("channel"),
                                        neg.get("exclude_eids", ())))[1]
        for neg in fx["negatives"]["tune"]]
    tau = _conformal_tau(tune_scores, ALPHA, upper=True)

    answered = 0
    for c in headline:
        d, s = idx.top1(c["text"], prov.mask(c.get("speaker"),
                                             c.get("channel"),
                                             exclude=tuple(c["gold"])))
        answered += int(d >= 0 and s > tau)
    rate = answered / max(1, len(headline))

    hits = 0
    for c in fx["leaky_cues"]:
        d, _s = idx.top1(c["text"], prov.mask(c.get("speaker"),
                                              c.get("channel")))
        hits += d in set(c["gold"])
    leaky_dense = hits / max(1, len(fx["leaky_cues"]))

    return {"distractor_answer_rate": round(rate, 4),
            "distractor_abstention": round(1.0 - rate, 4),
            "distractor_tau": round(tau, 4),
            "leaky_dense_recall": round(leaky_dense, 4),
            "instrument_alive_dense":
                1.0 if leaky_dense >= MIN_LEAKY_RECALL else 0.0}


def _check(m: dict, c: dict):
    # A dead rig refutes nothing: VOID, never a manufactured FAIL.
    if m["instrument_alive"] < 1.0 or c["instrument_alive_dense"] < 1.0:
        return Status.VOID
    if m.get("rows_replayed_ok", 0.0) < 1.0:
        raise RuntimeError(
            "ME.11 verdict NOT recorded: a family ledger row no longer says "
            "what this verdict cites (best recall row "
            f"{m.get('family_best_recall_row')}, ceiling row "
            f"{m.get('family_best_ceiling_row')}, cleared "
            f"{m.get('family_cleared_row')}). Re-derive the family verdict "
            "before buying this row.")
    if abs(m["paraphrase_recall_at_1"] - m["d_recall_row"]) > ROW_REPRO_TOL:
        raise RuntimeError(
            "ME.11 verdict NOT recorded: the family's deciding row does not "
            f"reproduce live (live {m['paraphrase_recall_at_1']} vs row "
            f"{m['d_recall_row']}, tol {ROW_REPRO_TOL}). A verdict may not "
            "stand on a number today's code cannot re-buy.")
    # The claim, exactly as registered. On the rows as they stand this is
    # False — the honest FAIL the family already paid for.
    return (m["family_cleared_row"] >= 1.0
            and m["paraphrase_recall_at_1"] >= FAMILY_BAR
            and m["abstention_certify"] >= MIN_ABSTENTION
            and m["feasible_ok"] >= 1.0
            and m["verbatim_ok"] >= 1.0
            and c["distractor_abstention"] >= MIN_ABSTENTION)


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
