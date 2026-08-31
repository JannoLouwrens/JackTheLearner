"""ME.11.D — Arm D: a real sentence encoder (all-MiniLM-L6-v2, ONNX).

The question this arm answers: does CONTEXT — a 6-layer transformer bi-encoder,
22.7M params, attention over the whole sentence — buy what static semantics
measurably cannot? ME.11.C settled the premise: static embeddings produced the
first non-zero on the certified fixture (0.0437) but the ceiling even at zero
credulity was recall_unthresholded 0.123, and the conformal arithmetic was
INFEASIBLE on every seed (tau_fpr 0.365 > tau_cov 0.184 — the rejection bar
sits above where the true-positive cosines live). So the live question is
FEASIBILITY: does a contextual encoder SEPARATE the paraphrase-cosine
distribution from the adversarial-negative distribution, or does it merely
shift both? The smoke probe says separation exists at n=3 (cos(paraphrase)
0.714 vs cos(unrelated) -0.02); this test prices it at n=160 x 3 seeds under
the family's certified-abstention rules.

THE NULL IS ARM C, not Arm B — the spec's own words: "the question is not
whether MiniLM beats lexical, it is whether it beats FREE semantics." Arm C
(potion-base-8M through the identical pipeline) is re-measured IN-PROCESS on
the identical fixture build, imported from `me_11_c_static_embeddings`, not
re-transcribed. The falsified_by is a TIE: "recall within one seed-std of Arm
C — in which case the static table wins on cost and the transformer is
deleted." Operationalised, pre-registered: FAIL unless every seed's Arm-D
recall strictly exceeds that seed's Arm-C recall AND the aggregate margin
exceeds one seed-std of Arm C's recall (`arm_c_recall_std` from the runner's
own aggregation — no private statistics).

THE PIPELINE is ME.11.C's, imported verbatim (`_DenseIndex`, `_Prov`,
`_conformal_tau`, `_score_config`) so no configuration can be scored by a
private code path: encode every event (mean pooling over the attention mask,
onnxruntime CPUExecutionProvider, 4 intra-op threads), subtract the CORPUS
mean, L2-normalise, cosine, full argmax over provenance-compatible events,
split-conformal tau_fpr from the 300 TUNE negatives (order statistic 286/301),
certified on the held-out CERTIFY negatives. Same 0.95 abstention floor as
Arms A/B/C — the metric is recall AT fixed abstention, so a seed that cannot
hold the floor has not met the metric, whatever its raw recall. tau_cov and
the feasible indicator are REPORTED for the Review (they were C's verdict
mechanism) but not gated here: D's falsified_by names the tie, and with a
valid tau_fpr the credulity bound is carried by the abstention gate itself —
infeasibility can only eat coverage, which the recall gate already prices.

TWO SPEC-NAMED CONTROLS, one per failure direction, plus the family's
aliveness gate:

- RANDOM-PROJECTION (must collapse): the transformer's learned token-embedding
  table (`embeddings.word_embeddings.weight`, the graph initializer) is
  replaced with a seeded Gaussian matrix of IDENTICAL shape; positions,
  LayerNorms and attention stay trained; the whole pipeline re-runs —
  re-center, re-calibrate, re-score. If recall survives, the arm is measuring
  sentence length or token count, not meaning. Bar <= 0.02, same as Arm C.
- SHUFFLED-TOKEN (must degrade): every EVENT is re-encoded with its word order
  randomised (seeded), queries untouched, own calibration. "If recall survives
  shuffling, the encoder is a bag of words with extra steps and Arm C
  dominates it by construction" — the spec's words. Operationalised with the
  same tie logic the spec applies to Arm C, pre-registered: the control FIRES
  (FAIL) unless every seed's real recall strictly exceeds its shuffled twin
  AND the aggregate drop exceeds one seed-std of the shuffled recall. It lives
  in `_experiment`, not `_control`, so the real/shuffled comparison is paired
  per seed on the identical fixture build.
- ALIVENESS (must score): the fixture's LEAKY cues — word subsets of their
  target's own text — must reach recall >= 0.80 through the real pipeline,
  threshold included. A rig that cannot retrieve those is dead, and a dead
  rig's low recall refutes nothing (Status.VOID).

WITHIN-ARM VARIANTS, reported not gated (the spec's notes name them):
- bge-small-en-v1.5 (33M, ONNX from BAAI), with its OWN conventions — CLS
  pooling and its documented query prefix on queries only — because running a
  checkpoint against its published usage would measure our misuse, not the
  checkpoint. Its compressed cosine band (spec notes: real 0.617 vs fabricated
  0.595) predicts the worst abstention of any arm; the claim row shows whether
  that held.
- int8 (model_qint8_arm64): LATENCY ONLY. The spec's notes measured int8
  SLOWER than fp32 on this Neoverse-N1 (no i8mm) — "a disk win, not a speed
  win. Report both." Both query-encode latencies land on the claim row; the
  scored arm is fp32.

LATENCY, reported not gated (the falsified_by sets no bar; the hypothesis
prices ~13 ms encode and an 18-minute cold reindex as the cost side of
"worth"): ms/query end-to-end at 100k docs — encode + center + normalise +
full cosine scan + argmax, one query at a time, no batching. The 100k index is
built by TILING the corpus's raw embedding rows to 100k then centering, which
is mathematically identical to encoding the tiled texts (the tiled texts ARE
repeats; the encoder is deterministic per text) and does not spend 18 minutes
re-encoding duplicates — the honest reindex number is reported separately as
`cold_reindex_s_100k` projected from the measured corpus encode rate.

Worst-seed gating throughout: every gated conjunct returns per-seed as a 0/1
indicator, so the aggregate mean equals 1.0 only when EVERY seed cleared it.
"""
from __future__ import annotations

import time

import numpy as np

from ..fixtures import paraphrase_eval as F
from ..protocol import Ledger, run_spec
from ..registry import BY_ID
from .me_11_c_static_embeddings import (ALPHA, MAX_RAND_RECALL,
                                        MIN_ABSTENTION, MIN_LEAKY_RECALL,
                                        _conformal_tau, _DenseIndex, _Prov,
                                        _score_config)
from .me_11_c_static_embeddings import MODEL as ARM_C_MODEL
from .me_11_c_static_embeddings import _load_model as _load_arm_c

IMPL_DEPS = ["experiments/fixtures/paraphrase_eval.py",
             "experiments/tests/me_11_a_lexical_incumbent.py",
             "experiments/tests/me_11_b_bm25s_stemming.py",
             "experiments/tests/me_11_c_static_embeddings.py"]

REPO = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FP32 = "onnx/model.onnx"
ONNX_INT8 = "onnx/model_qint8_arm64.onnx"
BGE_REPO = "BAAI/bge-small-en-v1.5"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_SEQ = 256
BATCH = 64
INTRA_OP_THREADS = 4           # tenant-serving box; do not take all 4 cores' worth of spin
LATENCY_N_DOCS = 100_000

_SESSIONS: dict = {}           # module cache: sessions are seed-independent


def _session(repo: str, filename: str):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    key = (repo, filename)
    if key not in _SESSIONS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = INTRA_OP_THREADS
        _SESSIONS[key] = ort.InferenceSession(
            hf_hub_download(repo, filename), so,
            providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def _tokenizer(repo: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    key = ("tok", repo)
    if key not in _SESSIONS:
        t = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        t.enable_truncation(MAX_SEQ)
        _SESSIONS[key] = t
    return _SESSIONS[key]


class _OnnxEncoder:
    """The `.encode(list[str]) -> np.ndarray` face `_DenseIndex` expects.

    pooling="mean": masked mean over last_hidden_state (MiniLM convention).
    pooling="cls":  first token (bge convention).
    `prefix` is prepended to every text at encode time; the bge index is built
    with prefix="" and the attribute is set to the query prefix AFTER
    construction, so the corpus is stored bare and queries arrive prefixed —
    the checkpoint's published asymmetric-retrieval usage."""

    def __init__(self, sess, tok, pooling: str = "mean", prefix: str = ""):
        self.sess, self.tok, self.pooling, self.prefix = sess, tok, pooling, prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH):
            chunk = [self.prefix + t for t in texts[i:i + BATCH]]
            self.tok.enable_padding()
            enc = self.tok.encode_batch(chunk)
            self.tok.no_padding()
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            am = np.array([e.attention_mask for e in enc], dtype=np.int64)
            h = self.sess.run(None, {"input_ids": ids,
                                     "attention_mask": am,
                                     "token_type_ids": np.zeros_like(ids)})[0]
            if self.pooling == "cls":
                out.append(h[:, 0, :])
            else:
                m = am[..., None].astype(np.float32)
                out.append((h * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9))
        return np.concatenate(out, axis=0)


def _rand_projection_session(seed: int):
    """The spec's random-projection control, transformer edition: the learned
    token-embedding INITIALIZER is swapped for a seeded Gaussian of identical
    shape; everything else in the graph (positions, LayerNorm, attention)
    stays trained. Rebuilt per call — never cached, never reused across
    seeds."""
    import onnx
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from onnx import numpy_helper
    m = onnx.load(hf_hub_download(REPO, ONNX_FP32))
    rng = np.random.default_rng(9110 + seed)
    for i, init in enumerate(m.graph.initializer):
        if init.name == "embeddings.word_embeddings.weight":
            w = numpy_helper.to_array(init)
            r = rng.standard_normal(w.shape).astype(w.dtype) * w.std()
            m.graph.initializer[i].CopyFrom(numpy_helper.from_array(r, init.name))
            break
    else:
        raise RuntimeError("word-embedding initializer not found in graph")
    so = ort.SessionOptions()
    so.intra_op_num_threads = INTRA_OP_THREADS
    return ort.InferenceSession(m.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])


def _shuffle_words(text: str, rng) -> str:
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def _experiment(seed: int) -> dict:
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    prov = _Prov(events)
    headline = [c for c in fx["cues"] if not c["ambiguous"]]

    # The registered null: Arm C, identical pipeline, identical fixture build.
    arm_c = _score_config(
        _DenseIndex(_load_arm_c(ARM_C_MODEL), texts), prov, fx)

    enc = _OnnxEncoder(_session(REPO, ONNX_FP32), _tokenizer(REPO))
    t0 = time.perf_counter()
    idx = _DenseIndex(enc, texts)
    encode_s = time.perf_counter() - t0
    r = _score_config(idx, prov, fx)

    # Shuffled-token sabotage twin: same encoder, events with word order
    # randomised, own calibration — paired on this seed's fixture build.
    rng = np.random.default_rng(9120 + seed)
    shuf = _score_config(
        _DenseIndex(enc, [_shuffle_words(t, rng) for t in texts]), prov, fx)

    # Latency at 100k docs, end-to-end, one query at a time. Index = corpus
    # rows tiled to 100k then centered (identical math to encoding the tiled
    # texts; see docstring).
    raw = np.asarray(enc.encode(texts), dtype=np.float32)
    reps = -(-LATENCY_N_DOCS // len(raw))
    tiled = np.tile(raw, (reps, 1))[:LATENCY_N_DOCS]
    big = _DenseIndex.__new__(_DenseIndex)
    big.model = enc
    big.mu = tiled.mean(axis=0)
    big.mat = _DenseIndex._norm(tiled - big.mu)
    all_mask = np.ones(LATENCY_N_DOCS, dtype=bool)
    t0 = time.perf_counter()
    for c in headline:
        big.top1(c["text"], all_mask)
    ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(headline))
    del big, tiled

    # int8 variant: query-encode latency only (the notes' "report both").
    enc8 = _OnnxEncoder(_session(REPO, ONNX_INT8), _tokenizer(REPO))
    t0 = time.perf_counter()
    for c in headline[:40]:
        enc8.encode([c["text"]])
    int8_ms = (time.perf_counter() - t0) * 1000.0 / 40

    # bge-small variant: CLS pooling, query prefix on queries only.
    bge = _OnnxEncoder(_session(BGE_REPO, ONNX_FP32), _tokenizer(BGE_REPO),
                       pooling="cls")
    bge_idx = _DenseIndex(bge, texts)
    bge.prefix = BGE_PREFIX
    v = _score_config(bge_idx, prov, fx)
    bge.prefix = ""

    return {
        "paraphrase_recall_at_1": r["recall"],
        "recall_unthresholded": r["recall_unthresholded"],
        "arm_c_recall": arm_c["recall"],
        "arm_c_unthresholded": arm_c["recall_unthresholded"],
        "margin_over_arm_c": round(r["recall"] - arm_c["recall"], 4),
        "beat_c": 1.0 if r["recall"] > arm_c["recall"] else 0.0,
        **{f"recall_{k}": val for k, val in r["per_register"].items()},
        "abstention_certify": r["abstention"],
        **{f"abstain_{k}": val for k, val in r["per_family"].items()},
        "abstention_family_min": round(min(r["per_family"].values()), 4),
        "abstain_ok": 1.0 if r["abstention"] >= MIN_ABSTENTION else 0.0,
        "tau_fpr": r["tau_fpr"],
        "tau_cov": r["tau_cov"],
        "feasible_ok": r["feasible"],
        "shuffled_recall": shuf["recall"],
        "shuffled_unthresholded": shuf["recall_unthresholded"],
        "shuffle_dropped": 1.0 if r["recall"] > shuf["recall"] else 0.0,
        "variant_bge_recall": v["recall"],
        "variant_bge_unthresholded": v["recall_unthresholded"],
        "variant_bge_abstention": v["abstention"],
        "variant_bge_feasible": v["feasible"],
        "ms_per_query_100k": round(ms, 3),
        "int8_encode_ms": round(int8_ms, 3),
        "encode_docs_per_s": round(len(texts) / encode_s, 1),
        "cold_reindex_s_100k": round(LATENCY_N_DOCS / (len(texts) / encode_s), 1),
        "headline_cues": len(headline),
        "n_tune": len(fx["negatives"]["tune"]),
        "n_certify": len(fx["negatives"]["certify"]),
        "fixture_hash_seed_only": fx["hash"],   # _aggregate keeps run[0]
    }


def _control(seed: int) -> dict:
    """Random-embedding transformer must collapse; leaky cues through the REAL
    pipeline must score. Same bracket as Arm C: content-blind rig fails the
    first, dead rig fails the second."""
    fx = F.build(seed)
    events = fx["events"]
    texts = [e["text"] for e in events]
    prov = _Prov(events)

    rand = _OnnxEncoder(_rand_projection_session(seed), _tokenizer(REPO))
    rr = _score_config(_DenseIndex(rand, texts), prov, fx)

    enc = _OnnxEncoder(_session(REPO, ONNX_FP32), _tokenizer(REPO))
    idx = _DenseIndex(enc, texts)
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
    return (m["beat_c"] >= 1.0                  # every seed strictly beat Arm C
            # the falsified_by's tie bar: aggregate margin > one seed-std of C
            and m["margin_over_arm_c"] > m.get("arm_c_recall_std", 0.0)
            and m["abstain_ok"] >= 1.0          # every seed held the 0.95 floor
            and m["shuffle_dropped"] >= 1.0     # every seed: real > shuffled
            # shuffle must cost more than the shuffled twin's own seed noise
            and (m["paraphrase_recall_at_1"] - m["shuffled_recall"]
                 > m.get("shuffled_recall_std", 0.0))
            and c["rand_collapsed"] >= 1.0)     # every seed's control collapsed


def run(ledger: Ledger | None = None):
    return run_spec(BY_ID["ME.11.D"], _experiment, _check,
                    control_fn=_control, ledger=ledger)
