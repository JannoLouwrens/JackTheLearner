# The Memory-Retrieval Bakeoff — finding the memory from a paraphrase, never inventing one

> Researched and measured 2026-08-09 on this box (`aarch64`, **Neoverse-N1**,
> **4 cores**, 22 GB RAM, no GPU). Serves GOAL.md *"Memory makes it him … He
> remembers YOU — what he HEARD, what he SAID, what he DID, attributed per
> person"* and the owner's principle of the same date:
>
> > **Memory must be EXTRACTIVE, NEVER GENERATIVE.** What Jack reports about his
> > past must be a literal stored record or nothing at all. A language model may
> > INDEX the log (embeddings are a distance function) but must NEVER author the
> > answer, because a generator cannot abstain honestly — fluency is not evidence.
>
> Companion to `MEMORY.md` (mechanism catalogue) and the ME.1–ME.11 specs in
> `experiments/registry_expansion.py`. This document makes **ME.11 decidable by
> bakeoff**: it surveys what could win, designs an evaluation set that cannot be
> gamed by its author, specifies six precisely implementable arms, and fixes the
> two-dimensional decision rule **before** any arm is run.

---

## 0. The three findings that reframe the problem

Everything below is subordinate to these. Each was measured on this box today;
raw numbers and method are in §1.9 and §2.5.

**Finding 1 — the current retriever is not "80% good", it is 0% good at the
job.** ME.1 records `cued_recall 0.8667` and ME.9 records `1.0` on all three
channels. Both are true and both are about *templated cues that are word
subsets of their target*, which is precisely the case lexical containment aces.
Against eight paraphrase cues written to share no content words with their
target, the same retriever scored **0/8**, and only 1 of 8 cues even cleared its
own `abstain_below=0.34` similarity floor. BM25S — real lexical SOTA, stemmed
and stopworded — scored **1/8**. The passing ledger entries are not wrong; they
are *narrow*, and ME.11 exists because nobody had asked the wider question.

**Finding 2 — the abstention floor is already unsound, before any embedding is
added.** On the same 30-event diary, the worst real cue scored `0.000`
containment and the *best fabricated* cue scored `0.333` — one basis point under
the 0.34 floor. The current threshold survives ME.1 only because ME.1's
fabricated vocabulary is constructed disjoint from the event vocabulary. Real
fabricated questions share function-shaped words with real events and land right
on the floor. Adding semantics to this is adding a louder voice to a system that
cannot tell truth from noise; the threshold, not the encoder, is the hard part.

**Finding 3 — the calibration advice in the literature is backwards for a
diary, and we measured the inversion.** The 2024–2026 consensus for
out-of-corpus detection is *per-query normalization*: standardize the top score
against the tail of the candidate list, or use the top1−top2 margin, because raw
cosine is not comparable across queries (Ethayarajh 2019; Bruch et al.
arXiv:2210.11934; Intrator et al. arXiv:2410.02914; Rossi et al. arXiv:2408.04887).
On a 2,030-event diary, ranked by how well each statistic separates real cues
from fabricated ones (AUC over 8 real × 10 fabricated):

| abstention statistic | potion-8M | static-mrl | MiniLM-L6 |
|---|---|---|---|
| **raw top-1 cosine** | **0.975** | **0.988** | **1.000** |
| tail-standardized z₁ = (s₁−μ_tail)/σ_tail | 0.688 | 0.775 | 0.637 |
| absolute margin s₁−s₂ | 0.725 | 0.800 | 0.662 |
| normalized margin (s₁−s₂)/σ_tail | 0.625 | 0.750 | 0.675 |
| relative margin (s₁−s₂)/s₁ | 0.537 | 0.625 | 0.600 |

Raw similarity wins outright and every normalization *hurts*. The mechanism is
specific to episodic memory and worth naming, because it will recur: **a diary
is internally redundant and externally sparse.** A real cue ("what did ada say
was broken about the steps") has dozens of near-duplicate competitors in a life
log — Jack mentions ladders constantly — so its top hit is *not* an outlier
against the tail, and its margin is small. A fabricated cue ("who mentioned the
wedding cake") is far from everything, so the tail collapses, σ_tail shrinks,
and the nearest neighbour becomes a *huge* z-score. Normalization measures
"how unusual is the best hit relative to this query's neighbourhood", and for a
diary that quantity is **anti-correlated** with truth.

Consequence for the bakeoff: the abstention statistic is not a settled
implementation detail to be inherited from the RAG literature. It is a
**measured, per-arm choice**, and the spec below requires every arm to report
the separability of all five statistics so the choice is evidence, not fashion.

---

## 1. Survey, with honest CPU cost on 4 ARM cores

### 1.0 What "CPU-only" means on *this* box, exactly

This matters more than any benchmark table, because two of the standard
accelerations are unavailable here.

```
$ lscpu | grep -E 'Model name|CPU\(s\)'      Neoverse-N1, 4 cores  (Ampere Altra class)
$ grep Features /proc/cpuinfo
fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid
asimdrdm lrcpc dcpop asimddp
```

- `asimddp` **present** → SDOT/UDOT int8 dot-product is available.
- `i8mm` (SMMLA int8 matmul) **absent**.
- `bf16` **absent**.

AWS's Graviton3 ONNX Runtime results — +65 % QPS from bf16 fastmath SGEMM and
+30 % from int8 MMLA ([AWS ML blog, 15 May 2024](https://aws.amazon.com/blogs/machine-learning/accelerate-nlp-inference-with-onnx-runtime-on-aws-graviton-processors/))
— **do not transfer to this box**: both headline paths need instructions we do
not have. Any CPU cost quoted from a Graviton3 or x86 source must be treated as
an upper bound on our speed, and we measured our own numbers rather than
inherit theirs.

Two more hard constraints:

- **Disk.** `/data` is **100 % full (929 MB free)**; `/` has ~11 GB. The project
  venv lives at `/data/venvs/jackthelearner` (1.1 GB). Model weights must go to
  `/` (e.g. `HF_HOME=/home/opc/.cache/huggingface`) or the box falls over. This
  eliminates any arm needing >500 MB of weights.
- **Tenancy.** This box serves paying customers (`/home/opc/CLAUDE.md`). The
  ladder loop already runs `nice 19` with `OMP_NUM_THREADS=2`. Every measurement
  below was taken at `nice 19`; load average during the runs was **3.1–9.3**, so
  the medians are *pessimistic* and the best-of-N figures are the better
  estimate of a quiet box. Both are reported.

**Verified available for Python 3.9 / aarch64:** `onnxruntime==1.19.2`,
`model2vec==0.7.0`, `bm25s==0.3.10`, `tokenizers==0.22.2`, `scipy==1.13.1` —
all installed cleanly from PyPI wheels in this session (total 343 MB). The
project venv already has `torch 2.8.0+cpu` and `transformers 4.57.6`. Note that
`onnxruntime` 1.28 no longer ships a cp39 aarch64 wheel; **1.19.2 does**, and
that is what any ONNX arm must pin.

### 1.1 Lexical: BM25 and its modern implementations

BM25 remains the baseline that refuses to die out-of-domain. Thakur et al.,
**BEIR** ([arXiv:2104.08663](https://arxiv.org/abs/2104.08663), NeurIPS 2021
D&B) found BM25 outperforms most dense models zero-shot without domain
adaptation. Sciavolino et al., **Simple Entity-Centric Questions Challenge
Dense Retrievers** ([arXiv:2109.08535](https://arxiv.org/abs/2109.08535), EMNLP
2021) is the sharper result for us: on entity-centric questions dense scored
49.7 % vs BM25's 72.0 %, a gap reaching 60 points on some patterns — and a
diary is *nothing but* entity-centric questions ("what did **ada** say about the
**ladder**").

**BM25S** (Xing Han Lù, [arXiv:2407.03618](https://arxiv.org/abs/2407.03618),
2024) precomputes all term-document impacts into a sparse CSC matrix at index
time, so query scoring is a slice-and-sum with no per-query BM25 arithmetic;
reported up to 500× faster than `rank_bm25` on a single thread, in pure Python
+ scipy. **Measured here** on 100k short events:

| | index build | query (k=10) |
|---|---|---|
| BM25S, 10k events | 0.33 s | **0.348 ms** |
| BM25S, 100k events | 4.24 s | **0.876 ms** |

For comparison, the **current** `EpisodicMemory` linear scan is 35.4 ms/query at
100k events (ledger, ME.5 `lat_ms_100000`) — BM25S is 40× faster *and* a better
ranker. Any arm may use it essentially for free.

Cheaper still and worth naming since it needs no dependency at all: **SQLite
FTS5** with the `bm25()` ranking function, already available in stdlib
`sqlite3`. It gives on-disk, incrementally-updatable BM25 with zero new packages
— relevant given the disk situation.

### 1.2 Static / distilled embeddings — the "near-free semantics" tier

**Model2Vec** (Tulkens & van Dongen, MinishLab, 2024; MIT; no arXiv paper —
cite the Zenodo record `10.5281/zenodo.17270888` and the launch post
<https://huggingface.co/blog/Pringled/model2vec>) distils any sentence
transformer into a *static* token-embedding table: pass every vocabulary token
through the teacher individually, keep the **output** embedding, apply PCA, then
Zipf-weight by token rank. Inference is a mean of rows — no attention, no
sequence-length limit, no matmul. Their own ablations
(<https://github.com/MinishLab/model2vec/blob/main/results/README.md>) give
PCA ≈ +2.8, Zipf ≈ +3.1, output-over-input ≈ +6.1 points Avg-All.

**POTION** adds Tokenlearn (train the static matrix against ~1M teacher C4
embeddings, then re-regularize with PCA + SIF weighting):
<https://minishlab.github.io/tokenlearn_blogpost/>.

**static-retrieval-mrl-en-v1** (Tom Aarsen, HF, 15 Jan 2025, Apache-2.0,
<https://huggingface.co/blog/static-embeddings>) trains a `StaticEmbedding`
(bert-base-uncased tokenizer, 30 522 × 1024 `EmbeddingBag`) *from scratch* with
`MatryoshkaLoss(MultipleNegativesRankingLoss)` at batch 2048 over 13 retrieval
corpora — 17.8 h on one RTX 3090. Matryoshka dims [1024…32] mean it can be
truncated at will: NanoBEIR NDCG@10 1024d → 0.5031, 512d → 0.4957, 256d →
0.4819, 128d → 0.4622, 64d → 0.4176.

Retrieval quality, MTEB Retrieval (MinishLab's own comparison table, one
consistent column — **do not mix columns across sources**, see §1.10):

| model | MTEB Retrieval | params | dim | disk |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | **42.92** | 22.7 M | 384 | 90 MB fp32 |
| potion-retrieval-32M | 35.06 | 32.3 M | 512 | 129 MB |
| static-retrieval-mrl-en-v1 | 34.95 | ~31.3 M | 1024 | 125 MB |
| potion-base-32M | 32.67 | 32.3 M | 512 | 129 MB |
| potion-base-8M | 31.11 | 7.56 M | 256 | **30 MB** |
| potion-base-2M | 22.99 | 1.89 M | 64 | 7.6 MB |
| GloVe 300d | 21.80 | — | 300 | — |

So the best static retriever gives up ~8 MTEB-Retrieval points to MiniLM and
~16 to bge-small. **The question this bakeoff answers is whether those points
are the points that matter for a 100k-event personal diary** — a domain where
BM25's own NanoBEIR score (0.4518) beats every potion model, and where the
sentences are 10 words long.

### 1.3 Small transformer sentence encoders

Verified specs; BEIR-15 NDCG@10 as published by Snowflake
([arXiv:2405.05374](https://arxiv.org/abs/2405.05374)) where available.

| model | params | dim | layers | BEIR-15 | licence |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22.71 M | 384 | 6 | 41.95 | Apache-2.0 |
| snowflake-arctic-embed-xs | 22.57 M | 384 | 6 | 50.15 | Apache-2.0 (card prose) |
| bge-small-en-v1.5 | 33.36 M | 384 | 12 | 51.68 | MIT |
| gte-small | 33.36 M | 384 | 12 | 49.46 | MIT |
| e5-small-v2 | 33.36 M | 384 | 12 | 49.04 | MIT |
| snowflake-arctic-embed-s | 33.21 M | 384 | 12 | 51.98 | Apache-2.0 |
| granite-embedding-30m-english | 30.30 M | 384 | 6 | 49.1 (IBM col.) | Apache-2.0 |
| granite-embedding-small-english-r2 | 47 M | 384 | — | 50.9 | Apache-2.0 |
| EmbeddingGemma-300M | 302.9 M | 768 (MRL) | — | n/p | **Gemma Terms**, not Apache |
| Qwen3-Embedding-0.6B | 595.8 M | 1024 (MRL) | 28 | Ret. 61.83 (MTEB-en-v2) | Apache-2.0 |

The last two are excluded on cost grounds before measurement: 300 M–600 M
params at ~13 ms *per 22 M params* on this box (§1.9) puts a single query
encode in the 200 ms–1 s range, and EmbeddingGemma's licence is the Gemma Terms
of Use, not Apache-2.0 (a widely repeated error).

### 1.4 Learned sparse (SPLADE) — attractive in theory, one variant survives

SPLADE++ / SPLADE-v3 (Naver, [arXiv:2403.06789](https://arxiv.org/abs/2403.06789))
expand documents into weighted term distributions over the BERT vocabulary,
giving dense-like recall on an inverted index. The obstacle for us is the
**query encoder**: standard SPLADE runs a transformer at query time, i.e. it
costs exactly what MiniLM costs and inherits an uninterpretable score scale.

The variant that *does* fit the extractive constraint is the **document-side-only**
family — SPLADE-v3-Doc and OpenSearch's
`opensearch-neural-sparse-encoding-doc-v2-distill` — where expansion happens at
**index** time and the query is tokenised lexically with no model at all. That
is architecturally ideal for Jack: a model touches the log once, offline, and
query time is a pure inverted-index lookup that cannot hallucinate. It is
**not** in the bakeoff arms below only because it needs a torch forward pass per
event at write time (~13 ms/event, acceptable) *plus* an inverted index that
supports weighted terms, which BM25S does not provide out of the box. It is
recorded here as the **strongest deferred candidate** and should be revisited if
no arm below clears the ME.11 gate (§4.5).

### 1.5 Hybrid fusion — and a measured warning

**Reciprocal Rank Fusion** (Cormack, Clarke, Buettcher, SIGIR 2009):
`RRF(d) = Σ_i 1/(k + rank_i(d))`, conventionally `k = 60`. It needs no score
calibration, which is exactly why it is popular and exactly why it is dangerous
here: **it discards the magnitude information that §0/Finding 3 shows is our
only working abstention signal.** Bruch, Gai & Ingber, *An Analysis of Fusion
Functions for Hybrid Retrieval*
([arXiv:2210.11934](https://arxiv.org/abs/2210.11934), ACM TOIS) formalise
min-max, theoretical-min-max (TMM) and z-score convex fusion and show that
lexical and semantic score distributions are query-dependent, so normalization
is *necessary*; but they also warn that min-max forces `max = 1` for every
query, destroying the abstention signal. **TMM is therefore the only convex
normalization admissible for us**, because its lower anchor is the scoring
function's theoretical minimum, not the empirical minimum of whatever happened
to be retrieved.

**Measured here, and it is a negative result:** RRF(k=60) over BM25S + a dense
arm scored **p@1 = 0.375** on the eight paraphrase cues, against 0.625 for
potion-8M alone and 0.750 for MiniLM alone. Fusing a 0.125-accuracy lexical arm
with a 0.625-accuracy dense arm at equal weight *destroyed* a third of the
recall. RRF is not free; it is an average, and averaging with a near-random
ranker is a loss. Any hybrid arm must therefore be **weighted** and its weight
must be fit on a calibration split, not assumed.

### 1.6 Rerankers and late interaction — measured, and mostly too slow

Cross-encoders read query and document jointly, so they cost
`n_candidates × forward_pass`. **Measured on this box, reranking 20 candidates
for one query:**

| reranker | disk | rerank-20 latency | cascade p@1 |
|---|---|---|---|
| ms-marco-MiniLM-L-6-v2, ONNX fp32 | 91.0 MB | **516 ms** | 0.875 |
| ms-marco-MiniLM-L-6-v2, ONNX int8 | 23.1 MB | **329 ms** | 0.875 |
| mxbai-rerank-xsmall-v1, ONNX fp32 | 284.2 MB | **1899 ms** | 0.875 |

The quality is real — a potion-8M → cross-encoder cascade hit **0.875**, the
best paraphrase recall of anything measured, and potion-8M's recall@10 was
**1.000**, so the candidate list genuinely contains the answer. The cost is also
real: 329 ms is ~9× the entire current retrieval budget and ~375× a
static-embedding query. Worse, the cross-encoder scores **did not separate**
real from fabricated cues (`ms-marco` real-min 9.06 *below* fabricated-max 7.81
in logit terms, i.e. a negative gap), so a cascade would still need the
first-stage similarity to carry the abstention decision. This is a strong
argument for the cascade arm being *conditional*: use it only when the
first-stage decision is already "answer", never to decide *whether* to answer.

**ColBERT-style late interaction** (ColBERTv2; `answerai-colbert-small-v1`,
33 M) is excluded on index-size grounds: it stores one vector *per token*, so a
100k-event log at ~12 tokens/event becomes 1.2 M vectors — 460 MB at 96d fp16,
on a box with 929 MB free on `/data`.

### 1.7 Query expansion — and why HyDE is banned outright

Doc2query/docTTTTTquery and HyDE (generate a hypothetical answer document, then
retrieve with its embedding) both raise recall. **HyDE is disqualified by the
owner's principle**: it puts a generator in the query path. Even though it never
emits the final answer, it *authors the thing the search is conditioned on*, so
what Jack retrieves becomes a function of what a language model imagined. The
extractive constraint is about the causal chain, not just the returned bytes.

Non-generative expansion is admissible: RM3 / Rocchio pseudo-relevance feedback
(reweight the query using terms from the top-k retrieved *stored* events) uses
only the log's own vocabulary. It is cheap and is left as a within-arm option
for the lexical arms, disabled by default.

### 1.8 Calibrated abstention — the actual hard part

**Split conformal prediction** gives the clean half of the guarantee. With a
calibration set of `n` (query, true-event) pairs and nonconformity score
`A_i = −s(q_i, d_i*)`, the threshold is

```
τ_cov = s_(ℓ),   ℓ = ⌊(n+1)·α⌋
```

— the ℓ-th **smallest** true-item similarity in calibration. Then
`P(true event ∈ {d : s(q,d) ≥ τ}) ≥ 1 − α`, distribution-free
(Angelopoulos & Bates, [arXiv:2107.07511](https://arxiv.org/abs/2107.07511);
Lei et al., [arXiv:1604.04173](https://arxiv.org/abs/1604.04173); survey for
NLP: Campos et al., [arXiv:2405.01976](https://arxiv.org/abs/2405.01976), TACL
2024). Feasibility needs `n ≥ ⌈1/α⌉ − 1`, i.e. **n ≥ 19** at α = 0.05.
Retrieval-specific instantiations: CONFLARE
([arXiv:2404.04287](https://arxiv.org/abs/2404.04287)), and Intrator et al.,
*Streamlining Conformal Information Retrieval via Score Refinement*
([arXiv:2410.02914](https://arxiv.org/abs/2410.02914)), whose monotone
rank-discounted transform shrinks conformal set sizes by 77–96 % without
touching the guarantee.

**Conformal prediction says nothing about fabricated queries.** It is calibrated
on positives only; it bounds the *miss* rate, not the *false-answer* rate. The
ME.11 control — abstain on events that never happened — is the other half and
must be certified separately, on labelled negatives, with a binomial argument:

```
Clopper–Pearson one-sided lower bound on abstention rate:
  a_L = BetaInv(γ; k, m − k + 1)        (0 if k = 0)
Perfect run (k = m):  a_L = γ^(1/m) ≥ 0.95  ⟹  m ≥ ln γ / ln 0.95
  γ=0.05 → m ≥ 59      γ=0.01 → m ≥ 90
True rate 0.97, want a_L ≥ 0.95 at 95 % confidence  ⟹  m ≈ 300
True rate 0.96                                       ⟹  m ≈ 1040
```

**ME.1 and ME.5 currently use m = 60 and m = 30 fabricated probes.** At m = 30 a
perfect run certifies only `0.05^(1/30) = 0.905` — the ledger's
`abstain_100000 = 1.0` is compatible with a true abstention rate of 90 %. This
is not a criticism of those specs (they predate the question) but it fixes the
bakeoff's budget: **≥ 300 tuning negatives and ≥ 300 held-out certification
negatives per arm per seed.**

Selecting τ on negatives and then certifying on the same negatives is selection
bias. The valid options are (a) a fresh held-out negative split, (b) SGR-style
Bonferroni over the `⌈log₂ m⌉` binary-search steps (Geifman & El-Yaniv,
[arXiv:1705.08500](https://arxiv.org/abs/1705.08500)), or (c) **Learn-then-Test**
(Angelopoulos et al., [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)) with
**fixed-sequence testing** — because false-answer rate is monotone decreasing in
τ, walk τ upward from most permissive and stop at the first rejection, which
controls family-wise error at γ with *no* Bonferroni penalty. The bakeoff uses
(a) for reporting and (c) for selection.

The feasibility check is the part that must be reported honestly:

```
τ_cov  = largest τ preserving 1−α coverage on positives     (upper limit)
τ_fpr  = smallest τ certified to hold false-answer ≤ ε      (lower limit)
if τ_fpr > τ_cov:  INFEASIBLE — no threshold satisfies both.
```

An infeasible arm is a **result**, not a bug, and the correct response is a
better score function, never a split-the-difference threshold.

Finally, the whole-curve metric. A single operating point can be gamed;
**E-AURC** (Geifman, Uziel & El-Yaniv, [arXiv:1805.08206](https://arxiv.org/abs/1805.08206))
— area under the risk–coverage curve minus that of the optimal-ordering oracle
at the same accuracy — is comparable across arms of different base accuracy in
a way raw AURC is not. Every arm reports it.

Context on why this is under-served literature: essentially all 2024–2026
abstention work is generator-side (Self-RAG
[arXiv:2310.11511](https://arxiv.org/abs/2310.11511); *Sufficient Context*
[arXiv:2411.06037](https://arxiv.org/abs/2411.06037), ICLR 2025;
AbstentionBench [arXiv:2506.09038](https://arxiv.org/abs/2506.09038), NeurIPS
2025). Retrieval-side calibrated abstention with a certified false-answer rate
is rare — Walmart's Cosine Adapter
([arXiv:2408.04887](https://arxiv.org/abs/2408.04887), CIKM 2024) and C3R
([arXiv:2607.14157](https://arxiv.org/abs/2607.14157)) are the closest. The
best off-the-shelf negative set is **NoMIRACL**
([arXiv:2312.11361](https://arxiv.org/abs/2312.11361), EMNLP 2024 Findings),
whose human-verified non-relevant subset exposed hallucination rates above 88 %
in LLaMA-2/Orca-2. We build our own (§2) because Jack's corpus is his own life.

### 1.9 Measured CPU cost on this box — the table that decides feasibility

Method: `nice -n 19`, `OMP_NUM_THREADS=4`, `onnxruntime` CPUExecutionProvider,
100k synthetic events averaging ~10 words. Query-encode is batch-1, mean of 30
after 3 warm-ups. Doc throughput is batch-32. Load average 3.1–9.3 during the
runs (other ladder work was resident), so these are **pessimistic**; where
contention was visibly dominating, best-of-25 is reported alongside the median.
Script: `bench.py` / `search2.py`, reproduced in §6.3.

**Encoders**

| arm candidate | dim | params | query encode | docs/s (b=32) | index 100k | weights on disk | 100k index RAM |
|---|---|---|---|---|---|---|---|
| lexical containment (current) | — | 0 | **0.35 ms** ¹ | — | 0 s | 0 | ~40 MB (token sets) |
| BM25S | — | 0 | **0.88 ms** | — | **4.2 s** | 0 | ~60 MB |
| potion-base-2M | 64 | 1.89 M | **0.121 ms** | 10 651 | 9.4 s | 7.6 MB | 25.6 MB |
| **potion-base-8M** | 256 | 7.56 M | **0.123 ms** | 15 258 | **6.6 s** | 30 MB | 102 MB |
| potion-retrieval-32M | 512 | 32.3 M | 0.170 ms | 14 303 | 7.0 s | 129 MB | 205 MB |
| static-retrieval-mrl-en-v1 | 1024 | 31.3 M | **0.067 ms** | 13 971 | 7.2 s | 125 MB | 410 MB |
| static-retrieval-mrl @256d (MRL trunc.) | 256 | 31.3 M | 0.072 ms | 10 343 | 9.7 s | 125 MB | 102 MB |
| all-MiniLM-L6-v2 ONNX fp32 | 384 | 22.7 M | **13.4 ms** | 93.2 | **1073 s** | 90 MB | 154 MB |
| all-MiniLM-L6-v2 ONNX int8-arm64 | 384 | 22.7 M | 17.8 ms | 96.4 | 1038 s | **23 MB** | 154 MB |
| bge-small-en-v1.5 ONNX fp32 | 384 | 33.4 M | **46.9–58.7 ms** | 19.9–33.1 | 3020–5035 s | 133 MB | 154 MB |

¹ derived: ledger ME.5 `lat_ms_100000 = 35.38` is a full linear scan including
scoring; the tokenisation share is ~0.35 ms.

**The int8 result is worth stating plainly: dynamic int8 quantization made
MiniLM *slower* on this box (17.8 ms vs 13.4 ms).** That is consistent with
§1.0 — without `i8mm`, ONNX Runtime's int8 QGEMM has no fast path on
Neoverse-N1, and the dequantize/requantize overhead dominates. int8 is a
**disk-size** win here (23 MB vs 90 MB), not a speed win. Do not assume the
published "3.08× on CPU" figure (sbert efficiency docs, i7-13700K) applies.

**Search (brute-force cosine, numpy, top-10)** — best / median ms per query:

| N × d | fp32 | int8 (via int16 matmul) |
|---|---|---|
| 10 000 × 256 | 9.7 / 20.9 | 3.2 / 3.2 |
| 100 000 × 64 | 6.0 / 21.3 | 8.6 / 8.7 |
| 100 000 × 256 | **7.7 / 15.9** | 50.9 / 55.3 |
| 100 000 × 512 | 12.3 / 26.3 | 95.8 / 192.6 |
| 100 000 × 1024 | 25.4 / 62.0 | 199.3 / 201.0 |

numpy has no integer BLAS, so int8 scoring falls off a cliff above 10k rows.
**Conclusion: at 100k events, quantize for RAM if you must, but score in fp32.**

**End-to-end query cost at 100k events** (encode + brute-force scan, fp32,
median column):

| arm | ms/query @100k |
|---|---|
| lexical containment (current, from ledger) | 35.4 |
| BM25S | **0.9** |
| potion-base-8M (256d) | **16.0** |
| static-retrieval-mrl @256d | 16.0 |
| static-retrieval-mrl @1024d | 62.1 |
| all-MiniLM-L6-v2 (384d, fp32) | ~37 |
| + cross-encoder rerank of top-20 (int8) | **~350** |

**No ANN index is needed at 100k.** A flat numpy scan at 256d costs 8–16 ms,
comfortably inside ME.5's 1000 ms gate, which means the bakeoff can avoid
faiss/hnswlib entirely — a real saving on a box with 929 MB of free `/data` and
a tenancy rule against new background services. At 1 M events a flat 256d scan
would be ~160 ms, still passable; ANN can be deferred until then.

### 1.10 Provenance filtering: BEFORE semantic ranking, with per-stratum thresholds

ME.9 asks `what_did_they_tell_me("ada", …)`, i.e. retrieval restricted to
`channel="heard", speaker="ada"`. There are two places to apply that restriction
and only one is correct.

**Post-filter (retrieve top-k, then drop wrong-provenance hits) is wrong**, for
a reason specific to ME.11: it manufactures **false abstentions**. If the right
`heard/ada` event sits at rank 12 behind eleven `said/jack` events on the same
topic, a top-10-then-filter pipeline returns nothing and the ledger records an
abstention that looks honest and is in fact a miss. Since abstention is the
quantity under test, a pipeline that can silently convert misses into
abstentions is unmeasurable.

**Pre-filter is correct and also cheaper.** Keep `channel` and `speaker` as
int8 arrays parallel to the embedding matrix; build a boolean mask and score
only masked rows. On the ME.9 distribution (4 channels × ~4 speakers) this cuts
the scanned rows by ~10×, so a provenance-restricted query at 100k costs ~2 ms
rather than 16 ms.

**But pre-filtering breaks a single global threshold**, and this is the subtle
part. Restricting the candidate set changes the score distribution: the
best-of-4000 similarity is systematically lower than the best-of-100 000. A τ
calibrated on unrestricted queries will over-abstain on restricted ones.
The fix is **Mondrian / group-conditional conformal prediction** (Vovk,
[arXiv:1209.2673](https://arxiv.org/abs/1209.2673)): one τ_g per provenance
stratum g. Budget consequence: each stratum needs `n_g ≥ 19` calibration
positives at α = 0.05, so strata are `channel` (4) with speakers pooled, not
`channel × speaker` (16), unless the calibration set is large enough. The
bakeoff specifies `channel`-level strata plus one unrestricted stratum: **5 τ
values per arm**.

### 1.11 Scaling to 100k, and what ME.5 will need re-run

ME.5 currently passes at `u_p1 = 1.0` to 100k with lexical matching, because its
cues are unique 4-tuples drawn from disjoint pools — by construction exactly one
event matches. A semantic retriever will *not* trivially reproduce that: it will
find near-neighbours in embedding space that the lexical matcher rejected
outright. **Whichever arm wins ME.11 must re-run ME.5 before it can be adopted**,
and the expected failure mode is `u_p1` (unique cues) dropping below 0.95 while
`a_match` (ambiguous cues) holds. That is why ME.11's winner is adopted as an
*additional* index, not a replacement (§4.4): the hybrid keeps lexical's exact-match
precision and adds semantic recall, rather than trading one for the other.

Growth costs for the winner, at one event per minute (Jack's ME.5 assumption):

- **incremental write:** potion-8M 0.07 ms/event; MiniLM 11 ms/event. Both
  negligible against a 60 s inter-event interval.
- **cold rebuild of a 100k index:** potion-8M **6.6 s**; MiniLM **18 minutes**.
  This is the number that should decide, because the index will be rebuilt every
  time the encoder changes, and an 18-minute rebuild on a tenant-serving box is
  a real operational cost.
- **RAM:** 102 MB at 256d fp32; the aggregator's 3 GiB limit and the tenancy
  rule make anything above ~500 MB unacceptable.

---

## 2. The paraphrase evaluation set — design, and how it resists its own author

This is where a bakeoff is usually lost. The failure is not writing a bad test;
it is writing a test the author unconsciously made solvable, then reading the
resulting number as evidence. Six explicit mechanisms.

### 2.1 The generative grammar, and the lexical-disjointness invariant

Events are generated from a **paired vocabulary** in which every content concept
has a *stored form* and a disjoint set of *cue forms*:

```python
CONCEPTS = {
  "ladder":  {"stored": "ladder",  "cues": ["steps", "climbing frame", "rungs",
                                            "the thing you climb"]},
  "cracked": {"stored": "cracked", "cues": ["broken", "split", "fractured",
                                            "damaged"]},
  "pond":    {"stored": "pond",    "cues": ["water", "pool"]},
  "repaired":{"stored": "repaired","cues": ["fixed", "mended", "sorted out"]},
  # ... one entry per concept the generator may place in an event
}
```

The **invariant, asserted by the test itself and not by the author's belief**:

```python
assert _tokens(cue) & _tokens(event.text) <= ALLOWED_OVERLAP
```

where `ALLOWED_OVERLAP` is the speaker name **and nothing else** for attributed
questions, and the empty set for unattributed ones. A cue that violates this is
a generation bug and aborts the run. This is the single most important line in
the harness: it makes "the cues share no content words" a *machine-checked
property* rather than a claim in a docstring. Its corollary is the null
baseline's expected score: **lexical containment must score ≈ 0 by
construction**, and if it does not, the eval set is broken.

### 2.2 Four cue registers, reported separately

Averaging over cue types hides which capability an arm actually has. Four
registers, equal counts, each scored and reported independently:

| register | example (target: `"ada mentioned the ladder was cracked on the third rung"`) |
|---|---|
| **R1 synonym substitution** | `"the fractured climbing frame"` |
| **R2 indirect question** | `"what did ada say was broken about the steps"` |
| **R3 circumlocution** (no noun at all) | `"the thing you climb, she said it had a split"` |
| **R4 superordinate / vague** | `"did anyone mention something damaged"` |

R4 is deliberately the hardest and is expected to be *partly unanswerable*: with
several damaged things in the log, "something damaged" has no unique referent.
R4 therefore doubles as an **ambiguity probe** — see §2.4.

### 2.3 Distractor events: the abstention trap, three grades

For each target event the generator emits distractors that are topically
adjacent and factually wrong, in three grades of nastiness:

- **D1 same object, different predicate** — `"bruno mentioned the ladder had
  been freshly painted last spring"`. Tests that the retriever reads the
  predicate, not just the entity.
- **D2 same predicate, different object** — `"chika mentioned the drum skin was
  torn at the edge"`. Tests the converse.
- **D3 right content, wrong provenance** — the same sentence recorded on the
  wrong `channel`/`speaker`. This is ME.9's failure mode meeting ME.11's, and it
  is the one that a semantic retriever is most likely to fail, because the
  embedding of "ada mentioned X" and "jack replied X" is nearly identical.

Grade counts are fixed per target (2 / 2 / 1) so an arm cannot be advantaged by
a lucky distractor draw.

### 2.4 Gold labels: adjudicated, multi-label, and with an explicit "ambiguous" class

The most likely way this test lies is a single-gold label on a cue that has two
correct answers. In today's pilot, `"who talked about the thing you climb"` was
scored a MISS against a ladder-damage event because the retriever returned
`"jack climbed the ladder to reach the apple on the platform"` — which is a
perfectly good answer to the question as written. That is a **test bug scored as
a model failure**, and at scale it silently depresses every arm.

Therefore:

1. Gold is a **set** `G(cue) ⊆ eids`, not a single id. Correct = top-1 ∈ G.
2. `G` is computed **mechanically from the generator's own concept bindings**
   (every event whose concept-tuple satisfies the cue's concept constraints),
   never hand-written after seeing an arm's output.
3. Cues where `|G| > k_amb` (default 3) are moved to a separate
   **AMBIGUOUS** partition, reported but excluded from the headline recall.
4. **Frozen before any arm runs.** The cue set, gold sets and negative probes
   are generated from `seed`, hashed, and the hash is written into the ledger
   entry. An arm that runs against a different hash is not comparable.
5. **Gate on the minimum register count, never the total.** Added after
   implementation, and this is the one thing the design as first written got
   wrong. The first build produced 113 headline cues — a healthy-looking total —
   which decomposed as R1 26, R3 26, **R4 1**. Rule 3 was working *correctly*:
   in a corpus where one target's distractors legitimately answer another
   target's vaguer question, the superordinate register is almost entirely
   ambiguous and had been deleted by the labelling logic. A four-register
   headline recall would silently have been a three-register average, and **no
   arm would ever have been scored on the hardest cue type** — the exact
   register the incumbent fails worst. `ME.11.0` now asserts
   `min_register_cues >= 30` and reports the count per register, so this is a
   red ledger entry rather than a silent narrowing. Realised after the fix:
   **160 headline cues, 40 per register, 12 ambiguous** (LESSONS, *"An aggregate
   count hides a stratum the labelling logic has deleted"*).

### 2.5 Pilot evidence that the design discriminates

An 8-cue / 10-fabricated pilot over a 30-event hand-written diary (and again
over the same diary plus 2 000 synthetic filler events) already separates the
arms — which is the property a bakeoff needs. Numbers are small-n and are
reported as *signal that the design works*, not as results:

| arm | p@1, 30 events | p@1, 2 030 events | recall@10 |
|---|---|---|---|
| lexical containment (current) | **0.000** | — | — |
| BM25S | 0.125 | — | — |
| potion-base-8M | 0.625 | 0.625 | **1.000** |
| potion-retrieval-32M | 0.375 | — | — |
| static-retrieval-mrl-en-v1 | 0.500 | 0.500 | **1.000** |
| all-MiniLM-L6-v2 | **0.750** | 0.625 | **1.000** |
| bge-small-en-v1.5 (no query prefix) | 0.500 | — | — |
| bge-small-en-v1.5 (+ query prefix) | 0.625 | — | — |
| RRF(BM25S + potion-8M), k=60 | 0.375 | — | — |
| RRF(BM25S + MiniLM), k=60 | 0.375 | — | — |
| **cascade potion-8M → ms-marco-MiniLM-L-6 CE** | **0.875** | — | — |

Three things this pilot already establishes. (i) The spread is wide — 0.000 to
0.875 — so the eval discriminates. (ii) **recall@10 = 1.000 for every dense
arm**: the answer is always in the candidate list, so the entire remaining
problem is *ranking and thresholding*, not retrieval capacity. (iii) The
ME.11 gate of ≥ 0.80 is genuinely at risk — only the cascade cleared it, and it
costs 350 ms/query. **This spec can fail, which is the point.**

The dense arms' score separation on the same pilot (real cues vs fabricated,
30 events):

| arm | worst real top-1 | best fabricated top-1 | separable? |
|---|---|---|---|
| lexical containment | 0.000 | 0.333 | **no** (floor is 0.34) |
| BM25S | 0.000 | 0.308 | **no** |
| potion-base-8M | 0.301 | 0.300 | yes, by 0.001 |
| static-retrieval-mrl | 0.164 | 0.153 | yes, by 0.011 |
| all-MiniLM-L6-v2 | 0.425 | 0.348 | yes, by 0.077 |
| bge-small (+ prefix) | 0.562 | 0.573 | **no** |

bge-small's compressed band (everything in 0.55–0.82) is textbook anisotropy
(Ethayarajh 2019) and is why §3 requires mean-centering for every dense arm.

### 2.6 Negative probes: adversarial, not random, and sized for the claim

SQuAD 2.0's design lesson ([arXiv:1806.03822](https://arxiv.org/abs/1806.03822))
is that unanswerable questions must *look* answerable. Random off-topic queries
("what did the astronaut say about the submarine") are trivially rejected and
will inflate measured abstention by tens of points. Four negative families,
equal counts, 150 each → **600 per seed**, split 300 tune / 300 certify (the
sample size §1.8 shows is required to certify ≥ 0.95 at 95 % confidence):

- **N1 held-out-target** — take a real cue and *delete its target event from the
  index*, leaving all its distractors. This is ME.11's stated control and the
  hardest possible negative: the answer nearly exists.
- **N2 entity substitution** — a real cue with its object swapped for a
  same-type object absent from the log (`ladder` → `stepladder`… no: an object
  never recorded, e.g. `trampoline`). This is exactly the Sciavolino et al.
  ([arXiv:2109.08535](https://arxiv.org/abs/2109.08535)) failure mode, aimed at
  the retriever's own inductive bias.
- **N3 provenance-impossible** — ask what Ada said about a topic only Bruno ever
  raised. Content matches, provenance does not.
- **N4 out-of-world** — genuinely off-domain (the pilot's "wedding cake" class).
  Kept as an easy floor so a total collapse is visible.

Reporting N1–N4 separately is mandatory: an arm that abstains at 0.99 overall
but 0.60 on N1 has not solved anything, and a pooled number would hide it.

---

## 3. The bakeoff spec — seven specs, in registry format

> **STATUS, 2026-08-09.** All seven `Spec(...)` blocks below are **LIVE** in
> `experiments/registry_expansion.py` (commit `0c1ff06`, ids `ME.11.0` and
> `ME.11.A`–`ME.11.F`), and **`ME.11.0` has since PASSED** (commit `ea5b236`).
> They were adopted with light editorial trimming and one substantive change:
> `ME.11.0`'s budget is `Budget.CPU`, not `Budget.CPU_FAST` as first drafted
> here — building three 5 000-event lives plus a leak-control fixture does not
> fit in a minute. **The registry is authoritative**; this document is the
> design record and the place the reasoning lives. The blocks below have been
> re-verified to parse against the real `experiments/protocol.py`, to contain no
> duplicate ids, and to have every `depends_on` resolve in `registry.BY_ID`.
>
> What `ME.11.0` actually produced, per seed: **5 000 events, 160 headline cues
> (40 per register, minimum), 12 ambiguous cues held out, 300 tuning + 300
> certifying negatives, 52 positives in the smallest provenance stratum,
> `overlap_violations = 0`, `oracle_ceiling = 1.000`, and
> `lexical_null_recall = 0.000`.** The incumbent scores **zero** on an eval set
> whose oracle scores **one**. Finding 1 of §0 is now a ledger entry, not a
> pilot.
>
> **One flag for whoever next touches the registry:** the implementation of
> `ME.11.0` asserts `min_register_cues >= 30` (and the ledger records 40), but
> the `Spec.notes` in `registry_expansion.py` do not mention it. The doc block
> below has been corrected; the registry's prose should be brought into line, or
> the strongest guard in that fixture is undocumented where the specs live.

Drop-in for `experiments/registry_expansion.py` (same `Spec(...)` dataclass:
`id, tier, title, hypothesis, falsified_by, null_baseline, metric, budget,
depends_on, seeds, control, kills, notes`). ME.11 as already written becomes the
**adoption** spec; these make it decidable.

`ME.11.0` must PASS before any arm runs — it is the "is the test honest" gate,
and every arm `depends_on` it, so `protocol.blocked_by()` structurally prevents
running an arm against an unvalidated eval set.

```python
    # ── ME.11 BAKEOFF: the arms that make ME.11 decidable ────────────────
    # One shared fixture (experiments/fixtures/paraphrase_eval.py) generates,
    # for each seed: a 5,000-event life, paraphrase cues in 4 registers with
    # MECHANICALLY-derived gold SETS, 600 adversarial negatives in 4 families,
    # and a 100k-event scale life for latency only. The fixture hash is written
    # into every arm's metrics so two arms cannot silently be scored on
    # different data. Realised at seed 0: 160 headline cues, 40 per register,
    # 12 ambiguous held out, min stratum 52 positives, hash 9c915329f4755c3e.

    Spec("ME.11.0", 2, "The paraphrase eval set is honest before anyone is scored",
         hypothesis="Every cue shares NO content word with its target beyond an "
                    "explicitly allowed speaker name; the lexical-containment "
                    "null therefore scores <=0.10 on the cue set; gold sets are "
                    "derived from the generator's concept bindings, not hand "
                    "labels; and the ORACLE ceiling (score events by their "
                    "concept-tuple overlap with the cue's concept constraints) "
                    "is >=0.95, proving the questions are answerable at all.",
         falsified_by="Any cue-target content-word intersection outside the "
                      "allowed set, OR lexical null >0.10 (the cues leaked "
                      "surface form), OR oracle ceiling <0.95 (the cues are "
                      "not answerable and every arm's score is a floor effect), "
                      "OR the fixture hash differing across two builds at the "
                      "same seed (the eval set is not frozen).",
         null_baseline="Lexical containment on the cue set — must be ~0 BY "
                       "CONSTRUCTION. This spec exists to verify the "
                       "construction, so its null is its own primary assertion.",
         metric="eval_set_validity", budget=Budget.CPU, depends_on=["ME.1"],
         seeds=3,
         control="A DELIBERATELY LEAKY cue set (cues built by deleting words "
                 "from the target rather than by synonym substitution) must "
                 "make the lexical null score >=0.80. If the leak detector "
                 "cannot detect a planted leak it is not a detector.",
         kills="The entire bakeoff. An arm scored against an unvalidated eval "
               "set produces a number nobody may cite.",
         notes="Also asserts >=19 calibration positives per provenance stratum "
               "(the Mondrian conformal minimum at alpha=0.05) and >=300 "
               "tune + >=300 certify negatives per family-balanced split (the "
               "Clopper-Pearson minimum to certify abstention >=0.95 at 95% "
               "confidence). AND min_register_cues >= 30: gate on the THINNEST "
               "register, never the total — the first build had 113 headline "
               "cues of which register R4 held ONE, so the hardest cue type had "
               "been silently deleted by correct ambiguity labelling. "
               "Freezes cue set, gold sets and negatives by hash."),

    Spec("ME.11.A", 2, "Arm A — lexical containment, the incumbent, as the null",
         hypothesis="The shipped EpisodicMemory retriever (content-word "
                    "containment x recency x importance, abstain_below=0.34) "
                    "scores <=0.10 paraphrase recall@1 while abstaining >=0.95 "
                    "on adversarial negatives: honest and useless, quantified.",
         falsified_by="Paraphrase recall@1 >0.30 — in which case the premise of "
                      "ME.11 is wrong, lexical matching does generalise, and no "
                      "encoder is needed. This arm is written to be beatable; if "
                      "it is not beaten the bakeoff is cancelled and the money "
                      "is saved.",
         null_baseline="Recency-only retrieval (ME.1's null), carried forward "
                       "unchanged so all three specs share one floor.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="On the ME.1-style TEMPLATED cue set this same code must still "
                 "score >=0.80. An arm that fails its own home benchmark is "
                 "mis-wired, and its 0.10 on paraphrases would mean nothing.",
         notes="Measured pilot: 0/8 paraphrase cues, and only 1 of 8 cleared "
               "the 0.34 floor. Worst real cue 0.000 vs best fabricated 0.333 — "
               "the incumbent's abstention margin is ONE BASIS POINT and exists "
               "only because ME.1's fabricated vocabulary is disjoint by "
               "construction. Report N1 (held-out-target) abstention separately; "
               "that is where the floor is expected to fail."),

    Spec("ME.11.B", 2, "Arm B — BM25S with stemming, real lexical SOTA",
         hypothesis="A properly implemented BM25 (bm25s, Snowball stemming, "
                    "stopwords, k1=1.2 b=0.75) beats Arm A on paraphrase "
                    "recall@1 while keeping lexical retrieval's free abstention "
                    "(a query whose terms appear nowhere returns an EMPTY list, "
                    "no threshold needed), at <=2 ms/query at 100k events.",
         falsified_by="No gain over Arm A — i.e. the incumbent's weakness is "
                      "semantic, not an implementation defect, and stemming "
                      "buys nothing. (Pilot says 0.125 vs 0.000: a real but "
                      "tiny gain.)",
         null_baseline="Arm A.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="Shuffle the term-document matrix rows: recall must collapse "
                 "to ~1/N. A BM25 that scores the same on a shuffled index is "
                 "reading document length, not content.",
         notes="Also the cheap half of every hybrid arm, so its cost is "
               "measured once here: build 100k = 4.24 s, query = 0.876 ms — 40x "
               "FASTER than the incumbent's 35.4 ms linear scan. Whatever wins, "
               "this replaces the incumbent's scan on efficiency grounds alone. "
               "BM25S: Lu, arXiv:2407.03618."),

    Spec("ME.11.C", 2, "Arm C — static embeddings (potion-base-8M), near-free semantics",
         hypothesis="A distilled STATIC embedding table (model2vec potion-base-8M, "
                    "256d, 7.56M params, 30 MB, no attention) with corpus "
                    "mean-centering and a split-conformal threshold beats Arm B "
                    "on paraphrase recall@1 by >=0.30 absolute while holding "
                    "certified abstention >=0.95, at <=20 ms/query at 100k events.",
         falsified_by="Recall gain over Arm B <0.30, OR certified abstention "
                      "<0.95 at the conformal threshold, OR the "
                      "coverage/false-answer thresholds proving INFEASIBLE "
                      "(tau_fpr > tau_cov) — semantics bought recall with "
                      "credulity, which ME.11 explicitly forbids.",
         null_baseline="Arm B (BM25S). Also reported: potion-base-2M (64d) and "
                       "static-retrieval-mrl-en-v1 truncated to 256d, as "
                       "within-arm variants — the arm is 'static embeddings', "
                       "not one checkpoint.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="RANDOM-PROJECTION control: replace the learned embedding "
                 "table with a random Gaussian matrix of identical shape, "
                 "re-center, re-calibrate. Recall must collapse to ~chance. If "
                 "a random table scores anywhere near the learned one, the arm "
                 "is measuring sentence length or token count, not meaning.",
         notes="Measured on this box: 0.123 ms/query encode, 15,258 docs/s, "
               "100k index built in 6.6 s and held in 102 MB, weights 30 MB. "
               "Pilot p@1 0.625, recall@10 1.000. This is the cheapest arm that "
               "could plausibly win, and its 6.6 s reindex (vs MiniLM's 18 min) "
               "is an operational argument in its favour on a tenant-serving "
               "box. Model2Vec: Zenodo 10.5281/zenodo.17270888; POTION: "
               "minishlab.github.io/tokenlearn_blogpost."),

    Spec("ME.11.D", 2, "Arm D — a real sentence encoder (all-MiniLM-L6-v2, ONNX)",
         hypothesis="A 6-layer transformer bi-encoder (22.7M params, ONNX "
                    "CPUExecutionProvider, mean pooling, corpus mean-centering, "
                    "split-conformal threshold) beats Arm C on paraphrase "
                    "recall@1, and the recall it buys is worth its ~13 ms query "
                    "encode and 18-minute cold reindex at 100k.",
         falsified_by="Recall within one seed-std of Arm C — in which case the "
                      "static table wins on cost and the transformer is deleted. "
                      "This is the genuine falsification risk of the whole "
                      "bakeoff and the pilot says it is close (0.625 vs 0.625 "
                      "at 2,030 events).",
         null_baseline="Arm C (static embeddings) — the question is not whether "
                       "MiniLM beats lexical, it is whether it beats FREE "
                       "semantics.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU_LONG,
         depends_on=["ME.11.0"], seeds=3,
         control="Same random-projection control as Arm C, plus a "
                 "SHUFFLED-TOKEN control: encode each event with its word order "
                 "randomised. If recall survives shuffling, the encoder is a "
                 "bag of words with extra steps and Arm C dominates it by "
                 "construction.",
         kills="If Arm D ties Arm C, every transformer encoder is removed from "
               "the memory path and the 90 MB of weights, the onnxruntime "
               "dependency and the 18-minute reindex go with it.",
         notes="Measured: 13.4 ms/query (fp32), 93 docs/s, 1073 s to index 100k. "
               "int8-arm64 dynamic quantization made it SLOWER (17.8 ms) because "
               "this Neoverse-N1 has asimddp but NOT i8mm; int8 is a disk win "
               "(23 MB vs 90 MB), not a speed win. Report both. Also report "
               "bge-small-en-v1.5 as a within-arm variant WITH its query "
               "instruction prefix, but note its 47-59 ms/query and its "
               "compressed cosine band (real 0.617 vs fabricated 0.595 in the "
               "pilot) which makes it the worst arm for abstention despite the "
               "best BEIR score."),

    Spec("ME.11.E", 2, "Arm E — weighted hybrid, calibrated not assumed",
         hypothesis="Fusing Arm B's lexical scores with the best dense arm's, "
                    "using theoretical-min-max normalisation and a convex "
                    "weight w fit on the CALIBRATION split, beats both parents "
                    "on paraphrase recall@1 AND improves certified abstention, "
                    "because lexical overlap is most informative exactly where "
                    "the dense score is least trustworthy.",
         falsified_by="No gain over the better parent, OR — the specific risk — "
                      "fusion DEGRADING recall, which unweighted RRF already "
                      "did in the pilot (0.375 vs 0.625/0.750).",
         null_baseline="Unweighted RRF at k=60, the default everyone ships. It "
                       "is the null precisely because it is the popular choice "
                       "and it LOST here; beating it is the arm's minimum duty.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["ME.11.0"], seeds=3,
         control="Fit w on the calibration split, then evaluate with w=0 and "
                 "w=1 (each parent alone). If the fitted w lands within noise of "
                 "0 or 1, the hybrid is one parent wearing a costume and must be "
                 "reported as such rather than as a third method.",
         notes="Min-max normalisation is FORBIDDEN in this arm: it forces "
               "max=1 for every query, which destroys the absolute-similarity "
               "magnitude that section 0/Finding 3 shows is our only working "
               "abstention signal. Use TMM (Bruch et al., arXiv:2210.11934). "
               "The abstention decision is taken on the DENSE score, not the "
               "fused score, unless the fused score measurably separates better."),

    Spec("ME.11.F", 2, "Arm F — cascade: cheap recall, cross-encoder rerank, cheap abstention",
         hypothesis="Arm C retrieves top-50 (pilot recall@10 was 1.000, so the "
                    "answer is present), a 22.7M cross-encoder (ms-marco-"
                    "MiniLM-L-6-v2, ONNX int8) reranks them, and the ABSTENTION "
                    "decision stays with Arm C's calibrated first-stage score. "
                    "This yields the highest paraphrase recall of any arm at a "
                    "latency the live agent can still pay.",
         falsified_by="Recall gain over Arm C <0.10, OR mean latency at 100k "
                      "events >250 ms, OR the reranker changing the abstention "
                      "decision at all (it must not — see control).",
         null_baseline="Arm C alone (the cascade's own first stage). The "
                       "reranker must earn its 330 ms.",
         metric="paraphrase_recall_at_fixed_abstention", budget=Budget.CPU_LONG,
         depends_on=["ME.11.0"], seeds=3,
         control="ABSTENTION MUST BE UNCHANGED by reranking. Measured pilot: "
                 "the cross-encoder's own scores do NOT separate real from "
                 "fabricated cues (real-min -9.06 BELOW fabricated-max -7.78), "
                 "so any pipeline that lets the reranker decide whether to "
                 "answer is buying recall with confabulation. The test asserts "
                 "the abstention decision is byte-identical to Arm C's on every "
                 "query, and FAILS the arm if it is not.",
         kills="If Arm F wins on recall but breaks the 250 ms budget, it is "
               "recorded as the OFFLINE-only retriever (reflection generation, "
               "ME.3) and Arm C or E ships in the live loop. Two answers is an "
               "acceptable outcome; a slow live loop is not.",
         notes="Measured rerank of 20 candidates on this box: 516 ms fp32, "
               "329 ms int8, 1899 ms for mxbai-rerank-xsmall. At top-50 expect "
               "~800 ms int8, so the arm as specified will likely BREACH its "
               "own 250 ms gate and must be run at top-10 (~165 ms) as well. "
               "Report the recall/latency curve over k in {10,20,50}, not one "
               "point. Pilot cascade p@1 was 0.875 — the only configuration "
               "that cleared ME.11's 0.80 hypothesis."),
```

**Deferred, deliberately, and why** — recorded so the omission is a decision and
not an oversight: SPLADE document-side-only expansion (§1.4) is the strongest
candidate not run, because it is the only method that puts the model *entirely*
at index time; it is held back only because it needs a weighted inverted index
BM25S does not provide. ColBERT late interaction is excluded on index size
(460 MB at 100k events, §1.6). HyDE and any generative expansion are excluded on
principle (§1.7). A Jack-specific distilled static model is a stretch goal
(§6.4).

### 3.1 Wiring the arms into `experiments/bakeoff.py`

`experiments/bakeoff.py` is now the decision primitive (SYSTEM.md law 3), and it
imposes two things this design must satisfy exactly. Both are satisfied below;
neither is optional, and getting either wrong returns `VOID` rather than a
wrong answer — which is the module working, not failing.

**Which spec is the null, and which specs are arms.** `run_bakeoff(spec, arms,
null_run, …)` takes *one* null and *≥2* arms, and it gates **every arm** against
that null. Mapping the six arm specs onto that shape:

| spec | role in `run_bakeoff` | why |
|---|---|---|
| `ME.11.A` lexical containment | **`null_run`** | It *is* ME.11's declared null. An incumbent scoring 0.000 (ME.11.0, measured) is the floor every arm must clear, not a competitor. |
| `ME.11.B` BM25S | **not a bakeoff arm** — plain `run_spec` | Run first, standalone. Its job is to decide whether the null gets *upgraded*: if BM25S beats containment, `null_run` becomes BM25S for the bakeoff proper. Its pilot score (0.125) is far below the learning gate, so entering it as an arm would `VOID` the whole bakeoff for a method nobody expected to win. |
| `ME.11.C` static embeddings | **arm** | |
| `ME.11.D` MiniLM ONNX | **arm** | |
| `ME.11.E` weighted hybrid | **arm** | |
| `ME.11.F` cascade + reranker | **arm** | |

Four arms, one null, `seeds = [0, 1, 2]`. This is the ordering the ladder loop
should run: `ME.11.0` → `ME.11.A` → `ME.11.B` → *then* the bakeoff over C/D/E/F.

**The metric each `Arm.run(seed)` returns.** A single float, higher-is-better:

```
R_matched(seed) = paraphrase recall@1 on the held-out cue split, measured at
                  the tau for which THAT ARM's abstention on the held-out
                  negatives equals exactly 0.95 (linear interpolation between
                  adjacent grid points).
```

This is the whole two-dimensional rule of §4.3 collapsed into one scalar, which
is what `run_bakeoff` needs. **The collapse is the point:** every arm is scored
at the *same* abstention, so an arm cannot buy recall by abstaining less — the
thing the coordinator's brief demands, enforced by the metric's definition
rather than by a reviewer noticing. The remaining requirements (verbatim
identity, latency, RSS, threshold feasibility) are **admissibility predicates**
evaluated in each arm's own spec *before* the bakeoff, not extra metrics — see
"withdrawal" below.

**(a) COST — declared, with its unit.** `Arm.cost` defaults to `None` and a TIE
with any undeclared cost now returns `VOID` (`bakeoff.py` L161-173; LESSONS,
*"A default of zero is not 'unknown'"*). The unit this spec declares is:

> **`cost` = mean end-to-end query latency in MILLISECONDS at 100 000 events,
> measured on this box (4 × Neoverse-N1, `nice 19`, `OMP_NUM_THREADS=4`), seed 0
> of the scale leg, encode + score + threshold, excluding index build.**

Latency, not resident MB, because latency is what the live agent pays on every
single recall and it spans 20× across these arms, while RSS spans 4× and is
already bounded by an admissibility gate. Resident MB and cold-reindex seconds
are **reported for every arm** and are the second and third tie-breaks in §4.3's
prose rule, but `Arm.cost` carries exactly one number and this is it.

From §1.9, the costs to declare (to be **re-measured** at run time and written
into the ledger, never copied from here):

```python
Arm("C_static",  run=..., cost=16.0,  description="potion-base-8M 256d, flat fp32 scan")
Arm("D_minilm",  run=..., cost=37.0,  description="all-MiniLM-L6-v2 ONNX fp32 384d")
Arm("E_hybrid",  run=..., cost=17.0,  description="BM25S + potion-8M, TMM convex fusion")
Arm("F_cascade", run=..., cost=181.0, description="potion-8M top-10 -> ms-marco-MiniLM-L-6 int8 rerank")
```

`F_cascade` is declared at **k = 10**, not k = 20 or 50, because k = 10 (~181 ms)
is the only configuration inside the 250 ms admissibility gate of §4.3. The
k ∈ {20, 50} points are reported in `ME.11.F`'s own metrics as a
recall/latency curve; they are not eligible to enter the bakeoff.

**(b) THE LEARNING GATE — stated explicitly.** `run_bakeoff` is called with the
defaults, and this spec adopts them as pre-registered:

```
learning_gate_sigma = 3.0        margin_sigma = 1.5        higher_is_better = True
```

Every arm must clear the null by **≥ 3.0 σ** on `R_matched`, where σ is
`max(arm_std, null_std)` across the three seeds. **If any single arm fails, the
entire bakeoff returns `VOID`** — no winner is chosen, `Status.VOID` is recorded,
and the correct response is to fix or withdraw that arm and re-run, never to
decide among the survivors. This is T2.02's lesson (*"two non-learners cannot
arbitrate an architecture"*), and it applies with full force here: an arm that
cannot beat lexical containment on paraphrase cues has not demonstrated
paraphrase retrieval, so its opinion about static-vs-transformer is worthless.

Two consequences worth stating before any number exists, so that neither can be
rationalised afterwards:

- **The gate is easy to clear here, and that is not a loophole.** The null
  measured `0.000 ± 0.000` on this eval set. With `null_std = 0`, σ collapses to
  the arm's own seed spread, so an arm scoring 0.55 ± 0.05 clears at 11 σ.
  The gate is doing real work only for an arm that is both weak *and* unstable —
  which is exactly the arm that should not be allowed to arbitrate.
- **`margin_sigma = 1.5` will probably produce a TIE, and a TIE is a result.**
  The pilot separation between Arm C and Arm D was 0.000–0.125 with seed noise
  unmeasured. If C and D land within 1.5 σ, `run_bakeoff` returns
  `TIE → cheapest`, which selects **C at 16 ms over D at 37 ms** and deletes the
  transformer, the 90 MB of weights, the `onnxruntime` dependency and the
  18-minute reindex with it. That outcome is *stated in advance* as acceptable
  and is `ME.11.D`'s declared `kills`.

**Withdrawal, not silent exclusion.** An arm that fails admissibility (verbatim
< 1.000, latency > 250 ms, RSS > 500 MB, or thresholds infeasible) is
**withdrawn before `run_bakeoff` is called**, and its own spec records `FAIL`
with the failing predicate named in its metrics. It does not enter the bakeoff
with a zeroed score, because a zero would trip the learning gate and `VOID` the
decision for everyone else — punishing three working arms for a fourth's
latency. If fewer than two arms remain admissible, the bakeoff is not run and
`ME.11` records `Status.VOID` with the reason (§4.6).

---

## 4. The decision rule — two-dimensional, fixed before the run

### 4.1 Why one number cannot decide this

Recall and abstention trade against each other through a single knob. An arm can
buy any recall it likes by lowering τ, and any abstention it likes by raising τ.
**A comparison of arms at their own favourite operating points is not a
comparison.** The rule below fixes the operating point *by certificate* first,
then compares recall, then breaks ties by cost.

### 4.2 Per-arm operating-point selection (no test data touched)

For each arm, per seed, using only the calibration and tuning splits:

1. **Centre.** Subtract the corpus mean from all event and cue embeddings
   (§1.8; the cheap 80 % of whitening, one vector subtraction). Re-measure
   recall@10; keep centering only if it does not hurt.
2. **Choose the abstention statistic empirically.** Compute all five statistics
   of §0/Finding 3 on the calibration split; select the one with the best
   AUC(real vs tuning-negatives). **Record which one won.** Do not inherit raw
   cosine from this document's pilot — verify it.
3. **Coverage limit.** `τ_cov = s_(ℓ)`, `ℓ = ⌊(n+1)·α⌋`, α = 0.10, per
   provenance stratum (§1.10). This is the *highest* τ that still finds the
   memory when the memory exists.
4. **False-answer limit.** Fixed-sequence Learn-then-Test over an ascending τ
   grid on the 300 tuning negatives, ε = 0.05, γ = 0.05: `τ_fpr` = the smallest
   τ whose exact binomial p-value `P(Bin(m,ε) ≤ F(τ)) ≤ γ` rejects. This is the
   *lowest* τ that is certified not to invent memories.
5. **Feasibility.** If `τ_fpr > τ_cov`, the arm is **INFEASIBLE** and is
   recorded as such with both thresholds. It does not get a recall number,
   because there is no honest operating point at which to measure one.
6. Otherwise `τ* = τ_fpr` (maximum recall subject to the certificate).

### 4.3 The joint criterion, stated exactly

On the **held-out** cue split and the **held-out** 300 negatives:

```
ADMISSIBLE(arm) ⟺
    a_L        ≥ 0.95      # Clopper–Pearson lower bound on abstention,
                           #   γ = 0.05, computed per negative family N1..N4
                           #   AND on the pool. Every family must clear it;
                           #   a pooled pass with N1 at 0.60 is a FAIL.
  ∧ verbatim   = 1.000     # §5, no tolerance, no rounding
  ∧ lat_mean_100k ≤ 250 ms # live-loop budget (ME.5 allows 1000 ms; we keep 4x
                           #   headroom because the agent also has to think)
  ∧ rss_index_100k ≤ 500 MB
  ∧ feasible               # step 5 above

WINNER = argmax over ADMISSIBLE arms of  R_matched
  where R_matched = paraphrase recall@1 measured at the τ for which that arm's
  abstention on the held-out negatives equals exactly 0.95 (linear interpolation
  between adjacent grid points).

TIES are decided by run_bakeoff(margin_sigma=1.5) -> cheapest declared cost,
  where cost = mean query latency (ms) at 100k events   [§3.1(a)].
  Reported but NOT in Arm.cost, and used in prose only if latency also ties:
     2. lower cold-reindex time at 100k events   (6.6 s vs 18 min is a real cost)
     3. lower index RSS
     4. fewer new dependencies
```

**`R_matched`, not `R@τ*`, is the headline number, and it is the whole point of
this section.** Reporting recall at each arm's own certified τ* would still
reward an arm whose certificate happens to sit at a permissive threshold.
Matching every arm at exactly 0.95 measured abstention removes the last degree
of freedom: **an arm that wins recall by abstaining less cannot win, because it
is re-measured at the same abstention as everyone else.** It is also the reason
the rule survives being handed to `run_bakeoff`, which accepts one scalar per
arm per seed: the two dimensions are already fused inside `R_matched`, so
nothing about the abstention constraint depends on a human remembering it.

The admissibility predicate is evaluated **per arm, in that arm's own spec,
before the bakeoff runs** (§3.1, "withdrawal"). `run_bakeoff` never sees an
inadmissible arm and never sees a zero standing in for one.

Also reported for every arm, and required for the ledger entry, but not part of
the argmax: recall at 0.99 abstention, **E-AURC** over the full risk–coverage
curve, per-register recall (R1–R4, each with its cue count — see §2.4), per-family
abstention (N1–N4), the selected abstention statistic from step 2, and both
threshold limits `τ_fpr` and `τ_cov` even when feasible.

### 4.4 What the winner triggers

ME.11's own gate (recall ≥ 0.80, abstention ≥ 0.95, verbatim = 1.0) is unchanged
and is applied to the winner. Three outcomes:

- **Winner clears 0.80.** Adopt as an *additional* index inside
  `EpisodicMemory`, not a replacement (§1.11). Then **re-run ME.1, ME.5 and
  ME.9** before the ledger records ME.11 — a semantic index can plausibly break
  ME.5's `u_p1 ≥ 0.95` unique-cue precision, and adopting a retriever that
  quietly regresses three passing specs would be exactly the disease this repo
  was built to cure.
- **Winner is admissible but below 0.80.** ME.11 **FAILS** and the ladder
  records the best achievable paraphrase recall at certified abstention as a
  measured ceiling. This is a good failure: it tells us the next move is a
  Jack-specific encoder (§6.4) or document-side SPLADE (§1.4), not more
  threshold tuning.
- **No arm is admissible.** ME.11 records **`Status.VOID`**, not FAIL. Report the
  infeasibility interval `[τ_fpr, τ_cov]` per arm. This says the score functions
  available to us cannot separate a paraphrased memory from a plausible
  fabrication at the required rates — a genuine negative result about CPU-only
  episodic memory, far more useful than a tuned number, and **not** a refutation
  of ME.11's hypothesis (which was never tested). See §4.6 for why the
  distinction has to be in code.

### 4.5 What we refuse to do

- No threshold retuned after seeing test results.
- No arm added after the first arm is scored.
- No cue removed for being "unfair" after an arm misses it — ambiguity is
  handled *ex ante* by §2.4's mechanical gold sets, or not at all.
- No pooled abstention number used to hide a failing negative family.
- No arm re-entered into the bakeoff after being withdrawn for inadmissibility,
  unless the change that fixed it is committed first and all three seeds re-run.
- The winner does not enter `EpisodicMemory` until ME.1/ME.5/ME.9 re-pass.

### 4.6 The three outcomes, encoded — `_check` returns a `Status`, never a bool

`protocol.run_spec` now accepts a `Status` from `check` (`protocol.py` L295-303),
and it *raises* `VoidStatusMismatch` if metrics say VOID while `check` returns a
bare `False` — because T2.02 did exactly that, and a VOID recorded as FAIL reads
machine-side as the spec's `kills` field firing on a run that refused to
arbitrate (LESSONS, *"VOID is not FAIL, and the difference is load-bearing"*).
ME.11's `kills` is real, so this is not a stylistic point: a fallthrough here
would tell the ladder that semantic episodic retrieval had been *refuted*.

The check is therefore written with **no implicit fallthrough** — every return
path names its outcome and its reason:

```python
def _check(m: dict, c: dict) -> Status:
    # (0) the eval set must have been the honest one, this seed, this hash
    if m.get("fixture_hash") != m.get("fixture_hash_expected"):
        m["verdict"] = ("VOID: fixture hash mismatch — arms were not scored "
                        "on the eval set ME.11.0 certified")
        return Status.VOID

    # (1) fewer than two admissible arms: nothing was compared
    if m["n_admissible"] < 2:
        m["verdict"] = (f"VOID: {m['n_admissible']} admissible arm(s) "
                        f"({m['withdrawn']}); a bakeoff needs two. "
                        f"Infeasible thresholds: {m['infeasible_intervals']}")
        return Status.VOID

    # (2) the bakeoff itself refused to arbitrate (learning gate / undeclared
    #     cost on a tie). run_bakeoff already returned VOID; do not re-judge it.
    if m["bakeoff_verdict"] == "VOID":
        m["verdict"] = "VOID: " + m["bakeoff_reason"]
        return Status.VOID

    # (3) a decision exists. NOW the hypothesis may be tested, and may lose.
    if m["verbatim_ok"] < 1.0:
        m["verdict"] = ("FAIL: winner returned a string not byte-identical to "
                        "a stored record — extractive constraint violated")
        return Status.FAIL                       # this one SHOULD fire `kills`
    if m["winner_recall_matched"] < 0.80:
        m["verdict"] = (f"FAIL: best certified paraphrase recall "
                        f"{m['winner_recall_matched']:.3f} < 0.80 at 0.95 "
                        f"abstention (ceiling measured, not a broken run)")
        return Status.FAIL
    if m["winner_abstention_lower_bound"] < 0.95:
        m["verdict"] = "FAIL: abstention certificate not met by the winner"
        return Status.FAIL
    if min(m[f"abstain_N{i}"] for i in (1, 2, 3, 4)) < 0.95:
        m["verdict"] = ("FAIL: a negative family fell below 0.95 "
                        "(a pooled pass would have hidden it)")
        return Status.FAIL

    m["verdict"] = f"PASS: {m['winner']} at {m['winner_recall_matched']:.3f}"
    return Status.PASS
```

Three properties of this that are deliberate. **`Status.VOID` is checked before
any FAIL branch**, so an undecidable run can never be recorded as a refutation.
**Every VOID branch writes its reason into `m["verdict"]`**, so the ledger's
metrics and its status agree — the exact disagreement that needed hand-repair on
T2.02. And **`kills` fires only on a real FAIL**: the only outcome that deletes
work is "we compared the arms and semantic retrieval lost", never "we could not
compare the arms".

---

## 5. The verbatim-answer check — enforcing "extractive, never generative"

The owner's principle is an architectural constraint, so the test enforces it
architecturally, not by inspecting output strings and hoping.

### 5.1 Make generation structurally impossible (the primary mechanism)

Change the retrieval contract so that the retriever **cannot return text at
all**:

```python
# retrieval returns provenance-stamped POINTERS, never prose
@dataclass(frozen=True)
class Citation:
    eid: int          # position in the append-only log
    score: float
    stratum: str      # the channel/speaker filter that produced it

def recall(query: str, top_k: int = 3, channel=None, speaker=None,
           now=None) -> "list[Citation]":
    ...

# the ONLY function that produces a string, and it is pure I/O
def quote(eid: int) -> str:
    """Re-read line `eid` from the JSONL on disk and return its `text` field
    verbatim. No model, no cache, no formatting."""
```

With this contract the set of strings Jack can ever utter about his past is,
**by construction**, a subset of the lines in the log. An embedding model can
reorder pointers; it has no channel through which to author bytes. Everything in
§5.2 is then a *verification* that the contract holds, not the contract itself.

### 5.2 The four assertions the test makes, per returned answer

For every one of the 172 cues × 3 seeds that produces an answer:

1. **Byte identity.** `quote(eid).encode("utf-8") == stored_bytes[eid]`, where
   `stored_bytes` is parsed **in a fresh process** from the JSONL by an
   independent reader that shares no code with the retriever. Comparison is on
   bytes, not on `str` — NFKD normalisation, smart quotes and stripped
   whitespace are all silent corruptions that `==` on `str` can mask after a
   Unicode round-trip.
2. **Substring of the raw file.** The returned bytes appear verbatim in
   `Path(log).read_bytes()`. This catches an answer assembled from two real
   events, which passes assertion 1 for neither and would otherwise slip through
   a per-field check.
3. **Provenance integrity.** The `channel` and `speaker` of the returned eid
   equal those requested. A right sentence with the wrong attribution is a false
   memory (ME.9's whole thesis) and must count as an error here too.
4. **No generative component in the query path.** Assert that the retriever
   module's loaded objects expose no text-producing entry point — concretely, a
   whitelist check that the only model artefacts loaded are embedding/scoring
   sessions whose outputs are float arrays, and an assertion that the ONNX
   session's output rank/dtype is a 2-D float tensor, never token ids. A model
   that can emit tokens is a model that can author an answer.

### 5.3 The controls, without which the check checks nothing

Three planted violations, each of which the checker **must** catch. If any
control passes, the verbatim check is a decoration and the arm's result is void:

- **Paraphrase control.** An adversary arm returns the correct event's text with
  one word replaced by a synonym (`"cracked"` → `"broken"`). Must fail
  assertion 1 *and* 2.
- **Splice control.** An adversary arm returns the first half of the correct
  event concatenated with the second half of a distractor. Must fail assertion 2
  (this is the one a naive `text in file` check would miss if the check were
  done per-field rather than on the whole string).
- **Whitespace/normalisation control.** Return the correct text with a trailing
  space stripped and a curly apostrophe substituted. Must fail assertion 1 —
  this is the control that proves the comparison is on bytes.

### 5.4 The abstention side of extractiveness

Abstention must return an **empty list**, never a hedge, never a nearest
neighbour with a low-confidence caveat. The test asserts `recall(...) == []`
and, separately, that the surrounding agent code has no fallback path that turns
an empty citation list into prose. Concretely: a `NoMemory` sentinel is
forbidden; the empty list is the answer, and any caller that renders it must
render silence.

---

## 6. Cost — free compute only

### 6.1 Licences and weights (all open, all CPU-runnable)

| artefact | licence | disk |
|---|---|---|
| `minishlab/potion-base-8M` | MIT | 30 MB |
| `minishlab/potion-base-2M` | MIT | 7.6 MB |
| `sentence-transformers/static-retrieval-mrl-en-v1` | Apache-2.0 | 125 MB |
| `sentence-transformers/all-MiniLM-L6-v2` (ONNX fp32 / int8) | Apache-2.0 | 90 / 23 MB |
| `BAAI/bge-small-en-v1.5` (ONNX) | MIT | 133 MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (ONNX int8) | Apache-2.0 | 23 MB |
| **total, all arms** | | **~430 MB** |

**Install to `/`, never `/data`** (929 MB free there, and it is the volume that
already caused a 45 GB-WAL incident on this box). Set
`HF_HOME=/home/opc/.cache/huggingface`.

New Python dependencies for the project venv, all confirmed as PyPI aarch64
wheels for Python 3.9 in this session: `model2vec==0.7.0` (pulls `tokenizers`,
`safetensors` — `numpy` and `huggingface_hub` already present),
`bm25s==0.3.10` (pulls nothing new; `scipy` already present),
`onnxruntime==1.19.2` (**pin this** — 1.28 dropped cp39 aarch64 wheels).
Installed size measured: **343 MB** for the full set. `torch` and `transformers`
are already in `/data/venvs/jackthelearner` and are **not** required by any arm —
all encoders run through `numpy` (static) or `onnxruntime` (transformer), which
keeps the arms runnable even if the torch venv is ever rebuilt.

### 6.2 Wall-clock budget for the whole bakeoff

Structure the run as a **quality leg** (5 000 events, 3 seeds, all arms) plus a
**scale leg** (100 000 events, seed 0 only, latency and RAM only). This is the
choice that keeps the bakeoff inside `Budget.CPU_LONG`; running all arms at
100k × 3 seeds would put Arm D alone at 54 minutes of pure indexing.

Per seed, quality leg — realised fixture size from `ME.11.0`: 5 000 events,
160 headline + 12 ambiguous cues, 600 negatives = **772 queries**:

| arm | index | 772 queries | per seed |
|---|---|---|---|
| A lexical (the null) | 0 s | 772 × ~2 ms | 2 s |
| B BM25S (standalone spec) | 0.2 s | 772 × 0.4 ms | 1 s |
| C potion-8M | 0.4 s | 772 × ~1 ms | 1 s |
| D MiniLM | 54 s | 772 × 14 ms | 65 s |
| E hybrid | (reuses B + C) | 772 × ~2 ms | 2 s |
| F cascade k=10 | (reuses C) | 772 × 181 ms | 140 s |

Quality leg total ≈ **(2+1+1+65+2+140) × 3 seeds ≈ 11 minutes**. F is quoted at
k = 10 because that is the only configuration admissible under §4.3's 250 ms
gate and therefore the only one that enters the bakeoff; the k ∈ {20, 50} curve
points add ~14 minutes and are run once, at seed 0, inside `ME.11.F` alone.

Scale leg, seed 0 only: index 100k for B (4 s), C (7 s), D (1073 s) = **~18
minutes**, plus 200 timing queries per arm (≈ 2 minutes with F at k=10).

**Whole bakeoff ≈ 45 minutes of wall clock at `nice 19`** (11 quality + 18 scale
+ 14 for `ME.11.F`'s k-curve + 2 timing), comfortably inside `Budget.CPU_LONG`
(2 h) and splittable across two hourly ladder-loop slots: `ME.11.A`/`ME.11.B`
(the null and its possible upgrade) in one, then the C/D/E/F bakeoff in the
next. **Zero GPU hours.** No new background services, no daemon restarts,
nothing that touches the tenant containers.

### 6.3 Reproducing the measurements in this document

The three scripts used are in the session scratchpad and should be committed
alongside the fixture when the bakeoff is implemented:
`bench.py` (encoder load/latency/throughput/RAM), `search2.py` (brute-force
scan, fp32 vs int8, best-of-25 and median), `probe.py`/`probe2.py`/`probe3.py`/
`probe4.py` (the 8-cue pilot, abstention-statistic AUCs, cascade). All run
under a standalone venv with the four packages of §6.1 and total ~10 minutes.

### 6.4 The stretch goal, and what would justify it

If no arm clears ME.11's 0.80 gate (§4.4), the next move is **not** a bigger
encoder — it is a **Jack-specific static embedding**. The static-embeddings
recipe (`MatryoshkaLoss(MultipleNegativesRankingLoss)`, batch 2048, 1 epoch)
took 17.8 h on one RTX 3090; the Kaggle free tier gives 30 GPU-hours/week on a
P100/T4, which fits with room to spare, and Model2Vec distillation of an
existing teacher is ~30 s on CPU. Training pairs come free from Jack's own
generator: (paraphrase cue, stored event) pairs are exactly the
`MultipleNegativesRankingLoss` format, with D1/D2/D3 distractors as hard
negatives. **Gate it:** this is only justified if (a) some arm proves the
candidate list contains the answer (pilot recall@10 = 1.000 says it does) and
(b) the failure is ranking, not retrieval. It buys a 30 MB, 0.12 ms/query index
tuned to the one domain Jack lives in, which is the correct end state for a
memory that must run forever on four shared ARM cores.

---

## 7. What this document does not claim

- **`ME.11.0` has PASSED; no arm has been run.** The only thing now established
  on the ledger is that the *evaluation set is honest* — 0 overlap violations,
  oracle ceiling 1.000, incumbent 0.000, 40 cues in the thinnest register, hash
  stable across builds. Every arm number in this document is still a pilot.
- The pilot numbers in §2.5 and §0 are **n = 8 cues and n = 10 negatives**. They
  establish that the arms are separable and that the eval design discriminates.
  They are **not** results, they do not decide anything, and no ledger entry may
  cite them. Section 3 exists precisely because they are too small to trust.
- The abstention-statistic inversion (§0/Finding 3) is measured on one 2 030-event
  diary. It is a strong enough signal to make "which statistic" a *measured*
  choice in every arm (§4.2 step 2) rather than an inherited assumption — that
  is all it is asked to support.
- No claim is made that any arm will pass. Arm A is written to be beaten and
  Arm D is written so that a tie kills the transformer. §4.4's third outcome —
  nothing is admissible — is a live possibility, is encoded as `Status.VOID` in
  §4.6's `_check` rather than left as a fallthrough, and is a reportable result
  rather than a reason to loosen a threshold.
- The `Arm.cost` figures in §3.1 are **this document's measurements, and are not
  the ones that will be recorded**. Each must be re-measured on the scale leg at
  run time and written into the ledger. Copying them forward would be a
  generated artifact going stale silently, which is its own lesson in
  `docs/LESSONS.md`.
- Nothing here has been demonstrated about **ME.5 under a semantic index**.
  §1.11 predicts `u_p1` is the metric at risk, but that is a prediction. The
  winner is not adopted until ME.1, ME.5 and ME.9 re-pass (§4.4), and if they do
  not, the winner is discarded regardless of how well it did on ME.11.
