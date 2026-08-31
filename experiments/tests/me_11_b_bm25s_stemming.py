"""ME.11.B — Arm B: BM25S with stemming, real lexical SOTA.

The question this arm answers: is the incumbent's 0.000 on paraphrases a
LEXICAL-METHOD defect or a lexical-CEILING fact? BM25 with Snowball stemming,
stopword removal and k1=1.2 b=0.75 is what a competent lexical retriever
actually looks like — stemming buys morphological variants ("tasting" hits
"tasted") which containment cannot see. If proper lexical SOTA still scores
near zero, the weakness is semantic and the dense arms (C/D) are justified; if
it recovers real recall, ME.11's premise ("the index did not help") needed
this measurement. The registry pilot read 0.125 vs 0.000: a real but tiny
gain, which is exactly the boundary this spec exists to pin down.

Scored on the SAME frozen fixture ME.11.0 certified honest (hash reported so
the cross-arm comparison is checkable), with the same advantages Arm A got:
provenance filters when the cue is attributed, top-1 counted correct against
the whole gold set. Arm A is the registered null and is re-measured
IN-PROCESS on the identical fixture build — imported from
`me_11_a_lexical_incumbent`, not re-transcribed (LESSONS.md: when you can
reference, reference).

FREE ABSTENTION, the property the hypothesis says must survive: BM25 scores a
query whose terms appear nowhere at exactly 0.0 on every document (verified in
the smoke test: OOV and empty queries return all-zero score vectors), so
"return nothing at score <= 0" is abstention with no tuned threshold. The
risk family is N1 (held-out-target): its cue terms DO appear in
topically-similar surviving events, so stemmed BM25 can buy recall with
credulity there — which is why pooled certify abstention >= 0.95 is a gate,
not a report.

LATENCY: the hypothesis claims <= 2 ms/query at 100k events. Measured on a
100,000-document index (the fixture's events tiled — term distribution
preserved, index size honest), end-to-end per query in a Python loop:
tokenize + retrieve + score filter. No batching, because "2 ms/query" that
only holds amortised over a batch is a different claim.

TWO CONTROLS, one per failure direction, because attempt 1 measured 0.0000
and a zero needs both:

- SHUFFLE (must collapse where the experiment scores): the term-document row
  mapping is read through a seeded permutation before scoring. A BM25 whose
  recall survives that is being scored by something other than the content of
  what it returns. Bar: recall <= 0.02, generous against a true chance floor
  of ~|G|/N ~ 0.0006.
- ALIVENESS (must score where the experiment collapses): the fixture's LEAKY
  cues — word subsets of their target sentence — must reach recall >= 0.80
  through this exact query path. Added after attempt 1, per LESSONS.md ("an
  at-chance control must carry proof its instrument was alive"): the shuffle
  control cannot discriminate a dead rig from a true ceiling when the
  experiment itself reads 0.0, and attempt 1's shuffled 0.0 sat beside an
  experiment 0.0 saying nothing. Probed before re-running: leaky recall 1.0
  at seed 0, and the mechanism of the zero is measured, not guessed —
  0 of 160 headline cues share even a Snowball STEM with their target (the
  synonym vocabulary is disjoint at the stem level, e.g. "the freshly coated
  block tackle" vs "the pulley was repainted"), so stemming has nothing to
  buy on this fixture BY MEASURE. `stem_leak_cues` reports that count
  per-seed on the claim row.

Worst-seed gating (aggregate-hides-worst-seed, REVIEW_QUEUE 2026-08-30):
every gated conjunct is returned per-seed as a 0/1 indicator, so the
aggregate mean equals 1.0 only when EVERY seed cleared it. A mean margin can
be one lucky seed; "gained on all three" cannot.
"""
from __future__ import annotations

import time

import bm25s
import Stemmer

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .me_11_a_lexical_incumbent import FAMILIES, REGISTERS, _recall_at_1

IMPL_DEPS = ["experiments/fixtures/paraphrase_eval.py",
             "experiments/tests/me_11_a_lexical_incumbent.py"]

MIN_ABSTENTION = 0.95          # same pooled certify bar Arm A was held to
MIN_LEAKY_RECALL = 0.80        # aliveness floor, same bar ME.11.0 used
MAX_MS_PER_QUERY = 2.0         # the hypothesis's own latency claim, at 100k
MAX_SHUFFLED_RECALL = 0.02     # ~30x the true chance floor of |G|/N
LATENCY_N_DOCS = 100_000
TOP_K = 50                     # >> any gold or exclude set (K_AMB=3, N1<=~12)
K1, B = 1.2, 0.75              # pre-registered in the spec's hypothesis


class _Bm25Index:
    """The arm: bm25s over stemmed, stopword-stripped event text."""

    def __init__(self, texts: list[str]):
        self.stemmer = Stemmer.Stemmer("english")
        self.n = len(texts)
        tok = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer,
                             show_progress=False)
        self.bm25 = bm25s.BM25(k1=K1, b=B)
        self.bm25.index(tok, show_progress=False)

    def query(self, text: str, k: int = TOP_K) -> list[tuple[int, float]]:
        """(doc_id, score) pairs with score > 0 — the free-abstention filter.
        An all-OOV or empty query scores 0.0 everywhere and returns []."""
        q = bm25s.tokenize([text], stopwords="en", stemmer=self.stemmer,
                           show_progress=False)
        docs, scores = self.bm25.retrieve(q, k=min(k, self.n),
                                          show_progress=False)
        return [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]


def _compat(ev: dict, speaker, channel) -> bool:
    """Identical provenance semantics to `mem.recall(channel=, speaker=)`."""
    return ((not channel or ev["channel"] == channel)
            and (not speaker or ev["speaker"] == speaker))


def _bm25_recall(idx: _Bm25Index, events: list[dict], cues,
                 remap=None) -> dict:
    """Top-1 after provenance filter, correct against the whole gold set.
    `remap` is the control's row permutation: retrieval is untouched, the
    returned row is read through it before any scoring."""
    hits = {r: 0 for r in REGISTERS}
    n = {r: 0 for r in REGISTERS}
    for c in cues:
        n[c["register"]] += 1
        for d, _s in idx.query(c["text"]):
            eid = remap[d] if remap is not None else d
            if _compat(events[eid], c.get("speaker"), c.get("channel")):
                hits[c["register"]] += eid in c["gold"]
                break
    total = sum(n.values())
    return {
        "recall": round(sum(hits.values()) / max(1, total), 4),
        **{f"recall_{r}": round(hits[r] / max(1, n[r]), 4) for r in REGISTERS},
    }


def _abstains(idx: _Bm25Index, events: list[dict], neg: dict) -> bool:
    """Abstain iff nothing provenance-compatible and non-excluded scores > 0.
    Same exclusion semantics as Arm A: hits inside `exclude_eids` are the
    deleted target itself and do not count as answering."""
    excl = set(neg.get("exclude_eids", ()))
    return not any(
        d not in excl and _compat(events[d], neg.get("speaker"),
                                  neg.get("channel"))
        for d, _s in idx.query(neg["text"]))


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    events = fx["events"]
    now = fx["now"]
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    # The registered null, re-measured in-process on the identical fixture.
    mem = F.load_into_memory(fx)
    arm_a = _recall_at_1(mem, headline, now)["paraphrase_recall_at_1"]

    idx = _Bm25Index([e["text"] for e in events])
    m = _bm25_recall(idx, events, headline)
    recall_b = m.pop("recall")

    certify = fx["negatives"]["certify"]
    fam_ok = {f: 0 for f in FAMILIES}
    fam_n = {f: 0 for f in FAMILIES}
    for neg in certify:
        fam_n[neg["family"]] += 1
        fam_ok[neg["family"]] += _abstains(idx, events, neg)
    abstention = sum(fam_ok.values()) / max(1, len(certify))

    # Latency at 100k documents, end-to-end, one query at a time.
    tiled = [events[i % len(events)]["text"] for i in range(LATENCY_N_DOCS)]
    big = _Bm25Index(tiled)
    t0 = time.perf_counter()
    for c in headline:
        big.query(c["text"])
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(headline))

    # How much the fixture even OFFERS a stemmer: headline cues sharing >= 1
    # Snowball stem with any of their gold events. Reported, not gated — it
    # is the measured ceiling on what any stemming method could recover.
    stem = idx.stemmer
    def _stems(t: str) -> set:
        return set(stem.stemWords(list(F._tokens(t))))
    stem_leaks = sum(
        1 for c in headline
        if any(_stems(c["text"]) & _stems(events[g]["text"]) for g in c["gold"]))

    return {
        "paraphrase_recall_at_1": recall_b,
        "stem_leak_cues": float(stem_leaks),
        "arm_a_recall": round(arm_a, 4),
        "margin_over_arm_a": round(recall_b - arm_a, 4),
        "gained": 1.0 if recall_b > arm_a else 0.0,
        **m,                                        # per-register recall
        "abstention_certify": round(abstention, 4),
        **{f"abstain_{f}": round(fam_ok[f] / max(1, fam_n[f]), 4)
           for f in FAMILIES},
        "abstention_family_min": round(
            min(fam_ok[f] / max(1, fam_n[f]) for f in FAMILIES), 4),
        "abstain_ok": 1.0 if abstention >= MIN_ABSTENTION else 0.0,
        "ms_per_query_100k": round(ms, 3),
        "latency_ok": 1.0 if ms <= MAX_MS_PER_QUERY else 0.0,
        "headline_cues": len(headline),
        "n_certify": len(certify),
        "fixture_hash_seed_only": fx["hash"],   # _aggregate keeps run[0]
    }


def _control(seed: int) -> dict:
    """Shuffled mapping must collapse to ~chance; leaky cues must score.
    Together they bracket a 0.0 experiment: dead rig fails the second,
    content-blind rig fails the first."""
    import random
    fx = F.build(seed)
    events = fx["events"]
    headline = [c for c in fx["cues"] if not c["ambiguous"]]
    idx = _Bm25Index([e["text"] for e in events])
    perm = list(range(len(events)))
    random.Random(9000 + seed).shuffle(perm)
    m = _bm25_recall(idx, events, headline, remap=perm)

    hits = 0
    for c in fx["leaky_cues"]:
        for d, _s in idx.query(c["text"]):
            if _compat(events[d], c.get("speaker"), c.get("channel")):
                hits += d in c["gold"]
                break
    leaky = hits / max(1, len(fx["leaky_cues"]))

    return {"shuffled_recall": m["recall"],
            "shuffle_collapsed": 1.0 if m["recall"] <= MAX_SHUFFLED_RECALL
            else 0.0,
            "leaky_recall": round(leaky, 4),
            "instrument_alive": 1.0 if leaky >= MIN_LEAKY_RECALL else 0.0}


def _check(m: dict, c: dict):
    from ..protocol import Status
    if c["instrument_alive"] < 1.0:
        return Status.VOID    # a dead rig refutes nothing — not a measurement
    return (m["gained"] >= 1.0                 # every seed beat Arm A
            and m["abstain_ok"] >= 1.0         # every seed held the floor
            and m["latency_ok"] >= 1.0         # every seed under 2 ms/query
            and c["shuffle_collapsed"] >= 1.0) # every seed's control collapsed


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11.B"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
