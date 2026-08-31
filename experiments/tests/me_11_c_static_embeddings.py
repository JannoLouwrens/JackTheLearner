"""ME.11.C — Arm C: static embeddings (potion-base-8M), near-free semantics.

The question this arm answers: does MEANING — a distilled static embedding
table, no attention, 30 MB — buy the recall that lexical methods provably
cannot? ME.11.B settled the premise by measurement: real lexical SOTA (bm25s +
Snowball) scores exactly 0.0000 on the certified fixture because the cue
vocabulary is stem-disjoint from the stored vocabulary (stem_leak_cues 0/160).
The incumbent's weakness is SEMANTIC. This arm is the cheapest thing that
could possibly fix it, and the spec prices the claim: beat Arm B by >= 0.30
absolute paraphrase recall@1 WHILE holding certified abstention >= 0.95, at
<= 20 ms/query at 100k events. Semantics that buys recall with credulity —
answering adversarial negatives it should refuse — is explicitly forbidden by
ME.11 and is a FAIL here, not a trade-off.

Scored on the SAME frozen fixture ME.11.0 certified honest (hash on the claim
row), with the same advantages every arm gets: provenance filter when the cue
is attributed, top-1 counted correct against the whole gold set. Arm B is the
registered null and is re-measured IN-PROCESS on the identical fixture build —
imported from `me_11_b_bm25s_stemming`, not re-transcribed (LESSONS.md: when
you can reference, reference).

THE PIPELINE, pre-registered: encode every event with potion-base-8M (256d),
subtract the CORPUS mean (mean-centering removes the shared "hub" direction
that static embeddings are known to carry), L2-normalise, and score queries by
cosine in that centered space. Top-1 = argmax over provenance-compatible,
non-excluded events — full argmax, no top-k truncation, so the abstention
semantics have no cutoff artefact.

ABSTENTION IS NOT FREE HERE — that is Arm B's advantage and this arm's risk.
Cosine scores every pair, so refusal needs a THRESHOLD, and a tuned threshold
is exactly the kind of dial law 4 forbids twiddling. So it is split-conformal,
set by order statistic on the TUNE negatives and certified on the held-out
CERTIFY negatives, never fit to the certify split:

  tau_fpr = k-th smallest of the 300 tune-negative top-1 scores,
            k = ceil((n_tune + 1) * 0.95) = 286.
  Answer iff top-1 score > tau_fpr, else abstain.

By exchangeability a fresh negative exceeds tau_fpr with probability
<= 1 - k/(n+1) ~ 0.0498, so certified abstention >= 0.95 is the EXPECTED
outcome if tune and certify are exchangeable — the certify measurement is the
check, not the fit. The mirror threshold prices feasibility:

  tau_cov = j-th smallest of the 160 headline-cue top-1 scores,
            j = floor((n_cues + 1) * 0.05) = 8.

Feasible iff tau_fpr <= tau_cov (>= ~95% of cues still answer at the operative
threshold). tau_fpr > tau_cov is the spec's own INFEASIBLE branch — the
negative-rejection bar sits above where the positives live, semantics bought
recall with credulity — and it FAILS the claim regardless of raw recall.

TWO CONTROLS, one per failure direction (the ME.11.B lesson, kept even though
this experiment is not expected to read zero):

- RANDOM-PROJECTION (must collapse where the experiment scores): the learned
  embedding table is replaced by a seeded Gaussian matrix of IDENTICAL shape,
  same tokenizer, then the whole pipeline re-runs — re-center, re-calibrate
  tau on the tune negatives, re-score. If recall survives that, the arm is
  measuring sentence length or token count, not meaning. Bar: <= 0.02,
  generous against a true chance floor of ~|G|/N ~ 0.0006.
- ALIVENESS (must score where the experiment might collapse): the fixture's
  LEAKY cues — word subsets of their target's own text — must reach recall
  >= 0.80 through this exact query path, threshold included. A bag-of-words
  static encoder maps a word-subset cue almost onto its source vector; a rig
  that cannot retrieve THOSE is dead, and a dead rig's low recall refutes
  nothing (Status.VOID, per "an at-chance control must carry proof its
  instrument was alive").

WITHIN-ARM VARIANTS, reported not gated (the spec's null_baseline names them:
the arm is "static embeddings", not one checkpoint): potion-base-2M (64d) and
static-retrieval-mrl-en-v1 truncated to its first 256 Matryoshka dims, each
through the identical pipeline with its own conformal threshold. Their numbers
land on the claim row so the Review can see whether the 8M checkpoint is
load-bearing or interchangeable — no gate reads them.

LATENCY: the hypothesis claims <= 20 ms/query at 100k events. Measured on a
100,000-document index (the fixture's events tiled — distribution preserved,
index size honest), end-to-end per query in a Python loop: encode + center +
normalise + full cosine scan + argmax. No batching, same rule as Arm B:
amortised latency is a different claim. Bench on this box: the 100k x 256
matvec+argmax alone is ~4 ms.

Worst-seed gating (aggregate-hides-worst-seed, REVIEW_QUEUE 2026-08-30):
every gated conjunct returns per-seed as a 0/1 indicator, so the aggregate
mean equals 1.0 only when EVERY seed cleared it.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .me_11_a_lexical_incumbent import FAMILIES, REGISTERS
from .me_11_b_bm25s_stemming import _Bm25Index, _bm25_recall

IMPL_DEPS = ["experiments/fixtures/paraphrase_eval.py",
             "experiments/tests/me_11_a_lexical_incumbent.py",
             "experiments/tests/me_11_b_bm25s_stemming.py"]

MODEL = "minishlab/potion-base-8M"
VARIANT_2M = "minishlab/potion-base-2M"
VARIANT_MRL = "sentence-transformers/static-retrieval-mrl-en-v1"
MRL_DIMS = 256                 # Matryoshka truncation, per the spec's text

MIN_GAIN_OVER_B = 0.30         # the hypothesis's own margin, absolute
MIN_ABSTENTION = 0.95          # pooled certify bar, same as Arms A/B
MIN_LEAKY_RECALL = 0.80        # aliveness floor, same bar ME.11.0 used
MAX_MS_PER_QUERY = 20.0        # the hypothesis's own latency claim, at 100k
MAX_RAND_RECALL = 0.02         # ~30x the true chance floor of |G|/N
LATENCY_N_DOCS = 100_000
ALPHA = 0.05                   # conformal level for both thresholds


def _load_model(name: str):
    """potion checkpoints load natively; the mrl variant ships in
    sentence-transformers layout (0_StaticEmbedding/) and is assembled by
    hand, truncated to its first MRL_DIMS Matryoshka dimensions."""
    from model2vec import StaticModel
    if name != VARIANT_MRL:
        return StaticModel.from_pretrained(name)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from tokenizers import Tokenizer
    w = hf_hub_download(name, "0_StaticEmbedding/model.safetensors")
    t = hf_hub_download(name, "0_StaticEmbedding/tokenizer.json")
    with safe_open(w, framework="numpy") as f:
        emb = f.get_tensor("embedding.weight")
    return StaticModel(vectors=emb[:, :MRL_DIMS].copy(),
                       tokenizer=Tokenizer.from_file(t), normalize=False)


class _DenseIndex:
    """The arm: mean-centered, L2-normalised static embeddings, full-scan
    cosine. Center comes from the CORPUS (never the queries); zero-norm
    encodings (all-OOV text) score -1 everywhere, which can only abstain."""

    def __init__(self, model, texts: list[str]):
        self.model = model
        raw = np.asarray(model.encode(texts), dtype=np.float32)
        self.mu = raw.mean(axis=0)
        self.mat = self._norm(raw - self.mu)

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return np.where(n > 1e-12, x / np.maximum(n, 1e-12), 0.0)

    def embed_query(self, text: str) -> np.ndarray:
        q = np.asarray(self.model.encode([text]), dtype=np.float32)[0]
        return self._norm(q - self.mu)

    def top1(self, text: str, mask: np.ndarray) -> tuple[int, float]:
        """(eid, cosine) of the best provenance-compatible, non-excluded
        event; (-1, -1.0) when the mask is empty or the query is all-OOV."""
        if not mask.any():
            return -1, -1.0
        q = self.embed_query(text)
        if not q.any():
            return -1, -1.0
        sims = self.mat @ q
        sims[~mask] = -np.inf
        d = int(np.argmax(sims))
        return d, float(sims[d])


class _Prov:
    """Vectorised twin of me_11_b._compat, verified by the same semantics:
    empty/None speaker or channel means 'no filter'."""

    def __init__(self, events: list[dict]):
        self.channel = np.array([e["channel"] for e in events])
        self.speaker = np.array([e["speaker"] for e in events])
        self.n = len(events)

    def mask(self, speaker, channel, exclude=()) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if channel:
            m &= self.channel == channel
        if speaker:
            m &= self.speaker == speaker
        for e in exclude:
            m[e] = False
        return m


def _conformal_tau(scores: list[float], alpha: float, upper: bool) -> float:
    """Order-statistic threshold. upper=True -> the ceil((n+1)(1-alpha))-th
    smallest (rejection bar from negatives); upper=False -> the
    floor((n+1)alpha)-th smallest (coverage bar from positives)."""
    s = sorted(scores)
    n = len(s)
    if upper:
        k = math.ceil((n + 1) * (1 - alpha))
        if k > n:
            raise ValueError(f"conformal level infeasible at n={n}")
        return s[k - 1]
    j = max(1, math.floor((n + 1) * alpha))
    return s[j - 1]


def _score_config(idx: _DenseIndex, prov: _Prov, fx: dict) -> dict:
    """One full pipeline pass: calibrate tau on TUNE, score cues and CERTIFY.
    Shared verbatim by the experiment, the variants and the random control so
    no configuration can be scored by a private code path."""
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    tune_scores = [
        idx.top1(neg["text"], prov.mask(neg.get("speaker"),
                                        neg.get("channel"),
                                        neg.get("exclude_eids", ())))[1]
        for neg in fx["negatives"]["tune"]]
    tau_fpr = _conformal_tau(tune_scores, ALPHA, upper=True)

    hits = {r: 0 for r in REGISTERS}
    n = {r: 0 for r in REGISTERS}
    cue_scores = []
    raw_hits = 0                # top-1 correct BEFORE the threshold — reported
    for c in headline:          # so a FAIL names its mechanism: retrieval vs
        n[c["register"]] += 1   # separation (threshold eats the recall)
        d, s = idx.top1(c["text"], prov.mask(c.get("speaker"),
                                             c.get("channel")))
        cue_scores.append(s)
        raw_hits += d in c["gold"]
        if s > tau_fpr and d in c["gold"]:
            hits[c["register"]] += 1
    tau_cov = _conformal_tau(cue_scores, ALPHA, upper=False)

    fam_ok = {f: 0 for f in FAMILIES}
    fam_n = {f: 0 for f in FAMILIES}
    for neg in fx["negatives"]["certify"]:
        fam_n[neg["family"]] += 1
        _d, s = idx.top1(neg["text"], prov.mask(neg.get("speaker"),
                                                neg.get("channel"),
                                                neg.get("exclude_eids", ())))
        fam_ok[neg["family"]] += s <= tau_fpr
    total_neg = sum(fam_n.values())

    return {
        "recall": round(sum(hits.values()) / max(1, sum(n.values())), 4),
        "recall_unthresholded": round(raw_hits / max(1, sum(n.values())), 4),
        "per_register": {r: round(hits[r] / max(1, n[r]), 4)
                         for r in REGISTERS},
        "abstention": round(sum(fam_ok.values()) / max(1, total_neg), 4),
        "per_family": {f: round(fam_ok[f] / max(1, fam_n[f]), 4)
                       for f in FAMILIES},
        "tau_fpr": round(tau_fpr, 4),
        "tau_cov": round(tau_cov, 4),
        "feasible": 1.0 if tau_fpr <= tau_cov else 0.0,
    }


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    prov = _Prov(events)
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    # The registered null, re-measured in-process on the identical fixture.
    arm_b = _bm25_recall(_Bm25Index(texts), events, headline)["recall"]

    model = _load_model(MODEL)
    idx = _DenseIndex(model, texts)
    r = _score_config(idx, prov, fx)

    # Latency at 100k documents, end-to-end, one query at a time.
    tiled = [texts[i % len(texts)] for i in range(LATENCY_N_DOCS)]
    big = _DenseIndex(model, tiled)
    all_mask = np.ones(LATENCY_N_DOCS, dtype=bool)
    t0 = time.perf_counter()
    for c in headline:
        big.top1(c["text"], all_mask)
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(headline))

    # Within-arm variants: same pipeline, own calibration, reported not gated.
    variants = {}
    for key, name in (("2m", VARIANT_2M), ("mrl256", VARIANT_MRL)):
        v = _score_config(_DenseIndex(_load_model(name), texts), prov, fx)
        variants[f"variant_{key}_recall"] = v["recall"]
        variants[f"variant_{key}_abstention"] = v["abstention"]
        variants[f"variant_{key}_feasible"] = v["feasible"]

    return {
        "paraphrase_recall_at_1": r["recall"],
        "recall_unthresholded": r["recall_unthresholded"],
        "arm_b_recall": round(arm_b, 4),
        "margin_over_arm_b": round(r["recall"] - arm_b, 4),
        "gained_030": 1.0 if r["recall"] - arm_b >= MIN_GAIN_OVER_B else 0.0,
        **{f"recall_{k}": v for k, v in r["per_register"].items()},
        "abstention_certify": r["abstention"],
        **{f"abstain_{k}": v for k, v in r["per_family"].items()},
        "abstention_family_min": round(min(r["per_family"].values()), 4),
        "abstain_ok": 1.0 if r["abstention"] >= MIN_ABSTENTION else 0.0,
        "tau_fpr": r["tau_fpr"],
        "tau_cov": r["tau_cov"],
        "feasible_ok": r["feasible"],
        "ms_per_query_100k": round(ms, 3),
        "latency_ok": 1.0 if ms <= MAX_MS_PER_QUERY else 0.0,
        **variants,
        "headline_cues": len(headline),
        "n_tune": len(fx["negatives"]["tune"]),
        "n_certify": len(fx["negatives"]["certify"]),
        "fixture_hash_seed_only": fx["hash"],   # _aggregate keeps run[0]
    }


def _control(seed: int) -> dict:
    """Random Gaussian table of identical shape must collapse; leaky cues
    through the REAL pipeline must score. Together they bracket the reading:
    content-blind rig fails the first, dead rig fails the second."""
    from model2vec import StaticModel
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    prov = _Prov(events)

    real = _load_model(MODEL)
    rng = np.random.default_rng(9100 + seed)
    rand = StaticModel(
        vectors=rng.standard_normal(real.embedding.shape).astype(np.float32),
        tokenizer=real.tokenizer, normalize=False)
    rr = _score_config(_DenseIndex(rand, texts), prov, fx)

    idx = _DenseIndex(real, texts)
    tune_scores = [
        idx.top1(neg["text"], prov.mask(neg.get("speaker"),
                                        neg.get("channel"),
                                        neg.get("exclude_eids", ())))[1]
        for neg in fx["negatives"]["tune"]]
    tau = _conformal_tau(tune_scores, ALPHA, upper=True)
    hits = 0
    for c in fx["leaky_cues"]:
        d, s = idx.top1(c["text"], prov.mask(c.get("speaker"),
                                             c.get("channel")))
        hits += s > tau and d in c["gold"]
    leaky = hits / max(1, len(fx["leaky_cues"]))

    return {"rand_recall": rr["recall"],
            "rand_collapsed": 1.0 if rr["recall"] <= MAX_RAND_RECALL else 0.0,
            "leaky_recall": round(leaky, 4),
            "instrument_alive": 1.0 if leaky >= MIN_LEAKY_RECALL else 0.0}


def _check(m: dict, c: dict):
    from ..protocol import Status
    if c["instrument_alive"] < 1.0:
        return Status.VOID    # a dead rig refutes nothing — not a measurement
    return (m["gained_030"] >= 1.0             # every seed: +0.30 over Arm B
            and m["abstain_ok"] >= 1.0         # every seed held the 0.95 floor
            and m["feasible_ok"] >= 1.0        # every seed: tau_fpr <= tau_cov
            and m["latency_ok"] >= 1.0         # every seed under 20 ms/query
            and c["rand_collapsed"] >= 1.0)    # every seed's control collapsed


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11.C"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
