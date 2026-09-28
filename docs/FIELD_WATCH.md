> **INCOMPLETE RUN — THIS IS A DRAFT, NOT A FINDING.**
> The field-watch run that wrote this file exited rc=124 and did not
> complete its own checklist (2026-09-28T06:07:11+00:00). Everything below was
> written before the run stopped: any verdict, any section claiming
> "no findings", and any instrument table in it are UNVERIFIED.
> Sealed automatically by scripts/lib_seal.sh; the exit code is in
> the log, and this banner is what joins the two.

# FIELD_WATCH.md — the scout's current-state report

> **Rewritten weekly. This is a state, not a log.** Every entry here is either
> live (nominated, awaiting the builder/owner) or on the watchlist. Superseded
> entries are deleted, not archived — the one-line history lives in
> `docs/FIELD_WATCH_LOG.md`.
>
> **What this file may and may not do.** It NOMINATES arms. It adopts nothing,
> changes no spec, no threshold, no decision. Every nomination below is a
> candidate for a bakeoff that the builder and the owner decide to run.
> `SYSTEM.md` law 3: decisions are made by bakeoff, never by argument — and a
> field watch that argued would be making decisions.

**Sweep date:** 2026-09-28 · **Window:** ~2026-04 → 2026-09 (6 months)
**Scout:** field watch, week 9. **Seven days since week 8** (2026-09-21), the
embargo (*"not before ~2026-09-28"*) spent to the day. **Fifth consecutive
sweep on the intended cadence.** **Front 3 (MEMORY) RETURNS** on its endorsed
two-week cadence, and it returns with a nomination for the first time in six
sweeps of looking.

**Confidence markers:** **[V]** fetched and read · **[c]** claimed by the authors,
not checked against their table · **[s]** asserted by a search engine *about* a
paper I have not opened · **[C]** computed by me, arithmetic shown · **[M]**
measured here, on this box, command shown.

---

## 0. WHAT MOVED IN THE INTERVAL — the week two of this desk's nominations became part of a spec that ran

**326 commits** landed in seven days. Five facts re-point this sweep, and the
first one is the best news this page has carried.

**(1) `t402` IS ACTED. THE BALANCING BAKEOFF THIS DESK REFUSED TO NOMINATE A
METHOD INTO FOR FOUR SWEEPS WAS DESIGNED, RAN, AND PASSED — AND IT CARRIES THE
REFERENCE ARM WEEK 8 NOMINATED.** `T4.06` *Fusion balancing bakeoff: three arms
vs the shipped brain* — **PASS, attempt 1**, `ran_at 2026-09-23T10:46:27`,
1588.66 s on a **kaggle Tesla T4**, `dirty_files None`, commit `aa7d49c` [M,
from the ledger row]. Its `null_baseline` reads, verbatim: *"The incumbent:
`T4.02`'s shipped rig re-run unchanged as arm zero, first in the same
submission; its worst-modality latent R2 IS the bar, pre-registered as a rule
before any number exists."*

That is week 8's N2 — *"`t402` must carry the UNBALANCED INCUMBENT as a scored
reference arm, and a balancing arm must beat THAT on the capability metric, not
merely move 30.12 → 1.0"* — and the spec was designed **2026-09-23, two days
after** that nomination was published. **I do not claim causation and I did not
find a citation of this page in the row.** What I can say is that the control
is in the spec, the spec ran, and the reference arm did exactly the work the
nominated paper predicted it would: the family's own numbers are now on our
ledger instead of in someone else's classification benchmark.

And what it measured is the paper's finding reproduced here. `refuted_arms
['grad_norm']`, `winning_arms ['loss_reweight']`, `n_winning_arms 1.0` [M].
`grad_norm` drove the ratio to a **definitional** 1.0 — its five per-modality
norms are byte-identical per seed (0.000651 / 0.000586 / 0.000904) because the
arm equalises them by construction — and **failed** the harder conjunct. The
one arm that won, won by **+0.0187** of latent R². **§6 is about what that
number is measured against.**

**(2) THE `a4` FORK IS RULED — QUEUED #2 CLOSES POSITIVE.**
`a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere` (week 7 §6)
is **DISPOSITIONED 2026-09-27**: the three-way fork is **RULED (i)+(iii)**, and
the (iii) half — amending `LEARNING_CORE.md` §5.4 to record that the guard was
never built and the seat was awarded without it — **was executed in that
sitting's commit**, not merely designed. That is the disposition this desk's
recorded leaning asked for, and it arrived.

The (i) BUILD half did **not** simply land: the sitting found a collision —
`D29` had resolved the same question on 09-23 as **(iii) alone** — so building
the readout is **CONTINGENT and routed to the owner as `D37`**, with the row
re-dated **2026-10-04** to check `D37` and, if it has ruled or defaulted, hand
the `Δ_k` readout to the builder then. **The row names week 8's N1 explicitly**
as one of three questions converging on the same 14.40 core-h, and says the
point of ruling this one first is *"what stops the project paying up to three
times for one run."* That sequencing is right and I am not contesting it.

**(3) MY OWN §6b WAS EXECUTED, AND IT CAUGHT A BUG THE ROW DID NOT ASK FOR.**
`fieldwatch-quotation-channel-is-0-for-5` is **ACTED 2026-09-24**, executing
builder commit `d901cb4` (09-23): `ROUTED:`-header stripping plus
**`MIN_QUOTE_OVERLAP = 6`** in `experiments/fieldwatch.py` **only** —
`decisions._shingles` left untouched, which was the row's own load-bearing
caution. Post-fix on the live corpus: **0 spurious of 18 sub-threshold pairs**,
both true quotations retained. The threshold was picked *after* enumerating 20
live overlaps (spurious cap 4 post-strip; true quotes 12, 18, 66), not guessed.

**The measurement bought something nobody asked for**, and week 8's page had
said in terms that it made *no* claim about the sibling reader: **`decisions.owner_asks`
has parsed 0 items since 2026-09-09** — a `**N.` format against a `^digit`
pattern — now routed separately as `owner-ask-reader-blind-since-0909` (OPEN,
DUE 10-01). A reader built to catch un-owned findings was itself blind for
fourteen days, and the thing that found it was measuring the false-positive
rate of its neighbour.

**(4) QUEUED #4 CLOSES POSITIVE — AND WEEK 8's FALSE `ROUTED` IS GONE.** The
reader now reports, on week 8's page [M, `run status`]:

```
FIELD-WATCH FINDINGS — week 8: 2 finding section(s); 2 cited, 0 quoted,
  0 UNROUTED-FIELD-FINDING.
  §6    ROUTED — cited by queue-row `lc03-five-controls-never-switch-off-the-term-a4-is-named-for`
  §6b   ROUTED — cited by queue-row `fieldwatch-quotation-channel-is-0-for-5`
```

Both citations are **real**. Week 8 predicted exactly this — that if a row
opened for either finding it would route by **citation**, the channel that
works — and recorded that it was leaving two *false* `quoted` readings standing
rather than editing prose until a counter looked right. **The false positives
were removed by fixing the reader, which is the correct repair, and both
findings reached a desk.** Two sweeps ago this page had no instrument at all.

**(5) THE ONE THING THAT DID NOT MOVE, AND THE QUEUE WENT BACKWARDS AGAIN.**
Week 7's N1 (the free-embedding critical-`d` probe) was **ACCEPTED and ORDERED
FIRST on 09-14** *"because it can FORECLOSE a family"*. At HEAD, **fourteen
days on** [M]: `grep -rln "free.embedding\|critical_d\|critical-d"` over the
repo returns **four files, all of them documents** —
`docs/INTEGRATION_QUEUE.md`, `docs/PROGRESS_LOG.md`, and this page and its log.
**No `.py` hit, no spec beyond the pre-existing `ME.1`…`ME.11.F`. STILL NOT
RUN**, and it remains the cheapest live item on the desk. Queue instrument
[M, `run review-queue`]: **64 OPEN, 88 live rows**, arrived **45** against
disposed **8** over the trailing 7 days, **drain UNBOUNDED**, oldest live 35 d.
§2's N2 lands beside that unrun probe, and I price it knowing this.

---

## 1. Coverage — what was actually searched, so the gaps are visible

The arXiv API answered on the first attempt. Week 8's operational note holds
and is worth one line: `https://` or `-L` returns HTTP 200; bare `http://`
returns a **301 with an empty body**, which is a silent failure and worse than
a 429 [M, re-verified this sweep].

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **3 · MEMORY** *(returns on its two-week cadence — the primary front)* | episodic / agent memory; **retrieval-side abstention and calibrated refusal**; extractive retrieval, sparse distributed memory, consolidation | **3 enumerations (40 + 25 + 30)**; **2 full HTML** (2609.22056, 2609.19942 abs); ledger re-read |
| **1 · LEARNING CORES** | world models / MBRL × latent, JEPA, proprioceptive, state-based; action-conditioning sufficiency | **40-entry enumeration**; 1 abstract (2609.31161) |
| **2 · MULTIMODAL FUSION** | probing / linear-probe / decodability × multimodal, fusion, representation; probe control tasks, selectivity, shuffled labels | **2 enumerations (30 + 25)**; **1 abstract** (2609.30210) |
| **4 · CURIOSITY & OPEN-ENDEDNESS** | intrinsic motivation / intrinsic reward / autotelic / open-ended learning | **40-entry enumeration**; **no fetch** |
| **5 · WORLDS & EMBODIMENT** | survival / homeostatic / foraging × cs.RO, cs.LG, cs.AI, cs.NE; **plus one WebSearch on the MuJoCo ecosystem** (mandate item) | **40-entry enumeration**; 1 WebSearch; **no fetch** |
| Biology-as-oracle | **`q-bio.NC` standing search** (wk7 §7) × RL, world model, embodied, curiosity, exploration | **40-entry enumeration**; **2 abstracts + 1 full HTML** (2605.27929, 2607.29476) |
| Small-model end | tiny / compact / parameter-efficient / lightweight × RL, control policy, world model | **30-entry enumeration**; no fetch |
| Queued closures | 3M-Progress full HTML (3rd pass), ARC-Bench code, 2608.29434 | **3 fetches [V]** |
| Our own artifacts | `T4.06` ledger row (29 metrics) + its implementation + `resolution.py` + `LESSONS.md` §16550; `ME.11` family rows; `fieldwatch.py`; `cores.py` dims; `run status`, `run review-queue`; **a known-answer reconstruction of `T4.06`'s ridge probe** | **[M]/[C] — §6, commands and script shown** |

**Known gaps, stated so nobody assumes coverage:**

- **Fronts 4 and 5 were enumerated but not fetched.** Nothing survived the
  abstract line. §5 says so rather than manufacturing a fetch to look thorough.
- **The probe-control-task literature returned nothing in window.** I searched
  for it specifically (control tasks, probe selectivity, shuffled-label
  baselines) because §6 needed it, and the in-window returns were unrelated.
  **§6's floor is therefore my own arithmetic, not an import**, and it is
  marked [M]/[C] throughout.
- **No conference main-track enumeration** (dropped permanently, week 4; still
  an acknowledged permanent gap). **No non-English sources.**
- **N2's code is not released and its hardware is unstated**; N3 is
  **abstract-level only**; N1 reports **no p-values or CIs for its geometric
  metrics and no compute statement at all**. Each is said again in place.
- **I did not measure the real fused representation's effective rank.** That
  would need the brain instantiated and trained, which is a run and not a
  scout's job. §6's whole nomination is that this quantity be recorded; its
  absence is the finding, not a gap I could have closed by reading harder.

---

## 2. NOMINATIONS

Three. **One is a set of cheap latent-geometry readouts from a mouse study that
can see the failure effective rank cannot; one is front 3's first
constitutionally admissible arm in six sweeps of refusals; and one is a control
for the readout metric §6 measures a defect in.** Each states its arXiv primary
category, its evidence class, its cost on **our** substrate, and both sides
steelmanned.

---

### N1 — three transition-geometry readouts for the `A4` diagnostic venue, because effective rank cannot see the failure four groups now name

**Source:** *Exploratory Experience Shapes the Geometry of Predictive
Representations* — [arXiv:2605.27929](https://arxiv.org/abs/2605.27929),
primary category **q-bio.NC**, 2026-05-27, Shilova, Sharafeldin, Balakrishnan &
Choi. **Abstract and full HTML read [V].**

**Why it is nominated where it is.** `LEARNING_CORE.md` §5.4 declares `A4`'s
mandatory diagnostic to be **effective rank and per-dimension latent
variance**, and week 7 §6 established that neither is computed anywhere in the
repository — now RULED (i)+(iii), with the BUILD half contingent on `D37`
(§0(2)). So there is a live, owner-bound question about *what readout to build*,
and this is the first paper this desk has found that proposes cheap alternatives
measured on a latent of our size.

**The verified content [V].** Three quantities, all computed on the prior latent
space (not on a UMAP projection):

| readout | definition, verbatim | what it asks |
|---|---|---|
| **spatial alignment** | `ρ_depth = \|Spearman(PC₁(z₁:L), d₁:L)\|` | does the leading latent direction track a task-relevant scalar |
| **transition consistency** | `C_i = (1/k)∑_{j∈N_k(i)} 𝕀{dir(j)=dir(i)}` | do latent **neighbours** share transition **direction** |
| **tortuosity** | `τ = ∑‖z_{ℓ+1}−z_ℓ‖ / ‖z_L−z_1‖` | is a monotone trajectory a path or a scribble in latent space |

Rig: online predictive-coding agent in a **binary-tree maze, N = 127 nodes**,
**64-dimensional recurrent state, 16-dimensional latent transition state**,
**30 seeds** — the best seed discipline in this sweep. Then the same predictive
model is trained on **natural trajectories of water-deprived mice** in the same
maze, and *"more exploratory mice show representational geometries that closely
match those of exploratory agents, whereas mice with more restricted visitation
patterns resemble reward-driven, exploitative agents."* Mouse behaviour is
quantified by normalised visitation entropy `E_mouse = −∑pᵢ log pᵢ / log N`.

**THE FALSIFIABLE REASON IT MIGHT WIN, and it is a gap in the incumbent
readout, not a preference.** Week 6's Context-Collapse nomination established
from three independent groups — and week 8's ARC-Bench made a fourth at
p = 1.5e-9 — that a latent world model can maintain healthy prediction
similarity while producing *action-insensitive* futures. **Context Collapse has
a HEALTHY effective rank.** Rank is a statistic of the *marginal* distribution
of latent states; it cannot tell an ordered trajectory from its own shuffle. `C_i`
and `τ` are statistics of *transitions*, so they can: a latent whose
neighbourhoods mix inward and outward transitions reads `C_i ≈ 0.5` at full
rank. So the two readouts are not competitors — **rank catches collapse, `C_i`
catches scrambling, and the declared diagnostic only has the first.** That
makes this a candidate for `D37`'s BUILD half rather than a fifth challenger to
a seat.

Second reason, and it is GOAL.md-shaped: the paper's result is that **latent
geometry is a function of the behavioural regime that generated the
experience** — exploratory policies organise it, exploitative ones do not. That
is a directly falsifiable prediction about `W0` (a curious arm and a task arm
should differ measurably in `C_i`), and it is the biology oracle in the form
GOAL.md asks for: the agent claim is checked against real animals rather than
asserted.

**Cost on our substrate.** The readouts themselves are **pure NumPy post-hoc on
recorded latents** — a Spearman correlation, a k-NN majority vote, and a path-length
ratio, at `LATENT = 64` [M, `cores.py:86`], which is the same order as the
paper's 64-dim recurrent state. **Zero new parameters, zero new network,
single-digit CPU-seconds** given latents on disk. But **no weights exist on
disk** (week 8 re-verified; I did not re-verify it this sweep and do not claim
to), so like `Δ_k` it is free only if bolted to whatever run `D37` authorises,
and standalone it inherits the same **14.40 core-h / 3 seeds** [M, week 8] —
which is precisely why the `a4` row's sequencing argument is right.

**WHY IT MIGHT LOSE — steelmanned.**

1. **There are no p-values, no confidence intervals and no statistical tests
   for any of the three geometric metrics** [V, checked explicitly]. Thirty
   seeds were run; what they produced is shown in figures. This is the same
   evidence class this desk has objected to for eight weeks and the objection
   does not weaken because I like the mechanism.
2. **A binary-tree maze of 127 nodes is not a body.** `ρ_depth` needs a scalar
   like "maze depth" to correlate `PC₁` against; `W0` has no such canonical
   scalar, and choosing one (drive level? time alive?) would be a free
   parameter of *ours*, which is how a readout becomes a story. `C_i` and `τ`
   are the two that transfer without inventing anything, and only `C_i` needs
   a discrete "direction" label — in `W0` that would also have to be defined.
   **I am nominating the readouts; the direction label is an open design
   question and it is mine, not the paper's.**
3. **Predictive coding is not `A4`'s objective**, and no compute, hardware or
   wall-clock is stated anywhere in the paper.
4. `A4`'s own untrained twin is the must-fail side, and it is already in the
   rig — but a twin scores near-zero on *everything*, so it cannot distinguish
   a healthy `C_i` from a lucky one. **A shuffled-transition control would be
   the real must-fail side, and the paper does not carry one.**

---

### N2 — RegimeAbstain's retrieval-structural score features: the "better score function" `MEMORY_RETRIEVAL_BAKEOFF` §1.8 pre-committed to, and the first admissible front-3 candidate in six sweeps

**Source:** *Predictable Failure in Multi-Hop Retrieval: Score-Distributional
Confidence Scoring and Abstention* —
[arXiv:2609.22056](https://arxiv.org/abs/2609.22056), primary category
**cs.IR**, 2026-09-18, Andre Bacellar (**single author**). **Abstract and full
HTML read [V].**

**Why this clears the constitutional bar that killed five sweeps of
candidates.** Every front-3 rejection since week 1 has been the same: the
method is generative, or it moves generation one step upstream into the write
path. This one has **no language model in the scoring path at all**, and the
paper says so in terms [V]: *"all features are computed from ANN similarity
scores available after retrieval — no additional model inference is
required"*, and *"RCS is computed in <1ms from ANN scores available at
retrieval time — no LLM call is needed for the abstention decision."* Retrieval
stays extractive; the abstention decision is a logistic function over numbers
the retriever already produced.

**Where it lands, and this is a pre-committed venue rather than my suggestion.**
`MEMORY_RETRIEVAL_BAKEOFF.md` §1.8, verbatim: *"An infeasible arm is a
**result**, not a bug, and the correct response is a **better score function**,
never a split-the-difference threshold."* Our own readings [M, re-derived from
the ledger this sweep, not inherited]:

| row | `feasible_ok` | `tau_cov` | `tau_fpr` | gap to close |
|---|---|---|---|---|
| `ME.11` a1 | **0.0** | 0.2272 | 0.3882 | **0.1610** |
| `ME.11.C` | **0.0** | 0.1840 | 0.3649 | 0.1809 |
| `ME.11.D` | **0.0** | 0.2272 | 0.3882 | 0.1610 |
| `ME.11.E` / `.F` | VOID | — | — | 0.1609 / 0.1809 |

Every arm and **every variant** (`2m`, `mrl256`, `bge`) reads
`feasible_ok 0.0`; `answered_cues` is **27.33 ± 4.78** of **160** headline cues.
The row `me11-every-arm-hits-the-same-infeasible-branch` is ACTED, but what it
executed was the **arithmetic verification** (E and F recorded VOID-FORECLOSED);
it explicitly says *"the semantic-retrieval redesign need is carried by the
`T2.10` paraphrase-venue conjunct… not a new row."* **The redesign is still
owed, and this is a candidate for it.**

**The verified numbers [V].** Nine features, all from the score distribution
rather than the top-1 score: `hop1-max`, `hop1-margin` (s₁−s₂), `hop1-top3`
mean, `hop1-H` (normalised Shannon entropy), `hop1-lift` (s₁ / mean of top-50),
three hop-2 analogues, and query length (d = 6 for single-hop pipelines).

| dataset / retriever | base CWAR | RCS AUC-AC | operating point |
|---|---|---|---|
| MuSiQue PropH | 39.5 % | **0.790** | 50 % coverage → CWAR **20.6 %** (−47.8 % rel.), acc 79.4 % |
| 2Wiki PropH | 14.5 % | **0.947** | cross-dataset transfer 0.942 (**−0.5 pp**) |
| HoVer Dense | 31.7 % | **0.873** | — |
| MuSiQue Dense | 62.1 % | **0.556** | — |
| 2Wiki Dense | 58.3 % | **0.649** | — |

Calibration **ECE 0.035**, Brier **0.183 vs 0.239** baseline, **95 % bootstrap
CIs (2,000 resamples)** in the tables. CWAR is defined
`|{q ∈ 𝒞ₜ : y(q)=0}| / |𝒞ₜ|` with `y(q)=1` iff **all** gold passages land in
top-k — which is our shape: `ME.11.0` records `max_gold_size 2.0` [M].

**THE FALSIFIABLE REASON IT MIGHT WIN.** Our thresholds are set on a **single
similarity score**, and `τ_fpr > τ_cov` says no single-score cut separates
must-answer from must-abstain on our fixture. The entropy, margin and lift
features carry information a scalar top-1 score cannot: *the shape of the
score distribution distinguishes "one clear hit" from "fifty near-ties at the
same height."* **And it is orthogonal to week 7's N1 in a way that matters
operationally** — wk7-N1 asks whether *any* embedding geometry can work
(a foreclosure probe on the representation); RCS changes the *decision rule*
over the scores the existing embedding already produces. **If wk7-N1 comes back
saying the geometry is hopeless, N2 can still win, because it never needed a
better embedding.** Those two are a genuine pair, and both are cheap.

**Cost on our substrate.** Fitting a 9-parameter logistic model: **milliseconds,
4 ARM cores, no GPU, no new encoder, no disk.** Inference **<1 ms/query** [V]
against `MEMORY_RETRIEVAL_BAKEOFF` §1.9's CPU table — the first front-3
candidate in nine sweeps that reports a latency at all. **And the supervision is
free here in a way it is not in the paper:** their `y(q)` came from dataset
annotations, whereas our fixture *generates* its gold sets (`oracle_ceiling
1.0`, `fixture_hash_seed_only 9c915329f4755c3e` [M]), so the label for all 160
cues exists at zero cost.

**WHY IT MIGHT LOSE — steelmanned.**

1. **It is SUPERVISED, and our labelling budget is 3× smaller than the paper's
   smallest.** Their tune sets are **486 / 509 / 1,007** examples; we have
   **160** headline cues [M]. At 10 parameters that is 16 cues/parameter, which
   is *not* the 1.5 rows/parameter disease of §6 and I will not pretend it is —
   but 160 must serve tune **and** certify, and §1.8 already demands ≥300
   negatives for certification. **The fixture may have to grow before the arm
   can be fairly read**, and that is a real cost, not a footnote.
2. **Its own weakest regime is near chance.** MuSiQue Dense AUC-AC **0.556**
   and 2Wiki Dense **0.649** — the method is not uniformly good and the paper
   is honest about it. `ME.11`'s dense arms are the ones that failed; the
   regime where RCS is weakest is the regime we are in.
3. **Single author, no code released, no hardware, and no statistical tests** —
   CI overlap is used for co-best determination, which is not a test. Weakest
   provenance nominated this week.
4. **Multi-hop public QA is not a diary.** Their features include two-hop
   structure that our single-hop retrieval does not have, so we would run the
   d = 6 variant, which is *not* the configuration carrying their best numbers.
5. Their CWAR ground truth is retrieval-side, but their headline MuSiQue
   reduction is quoted *"with LLM-judge"* — the judge is in the **evaluation**,
   not the scorer. I checked this because it is exactly the distinction five
   refusals turned on, and it holds — but it means their eval pipeline is not
   reproducible here without substitution.

---

### N3 — NOT AN ARM: the corruption-intervention control for fusion readout metrics, because §6 measures the defect and this is the discipline that catches it

**Source:** *The Alignment Illusion in Multimodal Large Language Models* —
[arXiv:2609.30210](https://arxiv.org/abs/2609.30210), primary category
**cs.CV**, 2026-09-24, Wang, Wang & Ding. **Accepted to NeurIPS 2026.**
**Abstract level only [V]** — stated here, not in a footnote.

**The claim.** Standard layer-wise alignment metrics — **CKA, SVCCA, MIR,
principal-angle cosine** — *"fail to distinguish corrupted visual tokens from
original ones despite sharp accuracy drops."* The diagnosis is structural:
*"anisotropic MLP down-projections pull visual and text tokens toward common
output directions, producing weight-induced alignment."* Their conclusion,
quoted because it is the transferable part: internal visual-text alignment
*"is therefore best read as a geometric diagnostic of the visual stream inside
the language model rather than a direct proxy for content-level cross-modal
interaction."* 13 MLLMs, five families, 0.5 B–72 B.

**The nomination is one line, and it is a method, not a metric:** the way they
established the illusion is by **controlled corruption** — replace the visual
tokens with Gaussian noise, observe that task accuracy collapses while the
similarity metric does not move, and conclude the metric is not measuring what
its name says. **Any readout used to decide a fusion bakeoff should carry that
control: corrupt one modality's input, and the readout for that modality must
move.** `T4.06`'s `min_modality_latent_r2` has no such lane. It has a ceiling
saturation lane (`R2_SATURATION = 0.99`) and a grad-scale dominance control that
guards the *ratio*, but nothing establishes that the R² readout responds to the
presence or absence of the sense it is named for.

This is the literature's version of §6's finding and I am presenting them as
**one problem stated twice, not two problems.** §6 measures that the readout's
floor is unknown and rank-dependent; N3 supplies the standard discipline for
showing a readout measures anything at all. It also completes a control set
this desk has been assembling for four sweeps: **wk6-N3** (pathway decoding
before *and* after fusion) catches a winner degrading the pathway it balanced,
**wk8-N2** (the do-nothing reference arm, now in the spec) catches the family
being worthless, and **N3** catches the readout being blind. Cost: **zero new
parameters**; one corrupted-input evaluation pass per modality on a model that
is being evaluated anyway.

**WHY IT MIGHT LOSE — steelmanned.**

1. **Their metrics are not our metric.** CKA/SVCCA/MIR are representational
   *similarity* measures between two streams; ours is a *supervised ridge
   readout* of a known target. A ridge probe is in principle harder to fool than
   CKA, and §6's measurement in fact shows **our probe is not capacity-limited**
   — so the analogy transfers the *discipline* and not the *defect*, and week 3's
   rule says to say so.
2. **Abstract level only**: no seeds, no hardware, no parameter counts, no code
   statement extracted. I did not open the full text; the mechanism paragraph
   above is the authors' abstract, marked [V] at that level and no further.
3. **An MLLM at 0.5–72 B with a language backbone is not a five-sense RSSM at
   861,545 params** [M, week 7], and their dominant stream is vision-in-text
   where ours is touch. This is the fourth consecutive sweep in which the
   fusion literature's regime is nothing like ours.
4. A corruption control can pass and still leave the readout uninformative:
   showing R² *moves* when a sense is destroyed does not show the absolute
   value means anything. **It is necessary, not sufficient** — §6's floor is the
   other half.

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **2607.29476** — *Resource depletion accelerates rate learning but not composition learning in patch foraging* ([abs](https://arxiv.org/abs/2607.29476), 2026-07-31, Kilpatrick & El Hady) | **q-bio.NC** | **Front 5's first on-axis item in seven sweeps, and it came from the biology category rather than `cs.*`** — week 7's lesson recurring. A normative Bayesian forager learning a patchy world while exploiting it, with a **dissociation**: depletion *accelerates* rate learning within a patch (*"successive encounters occur at falling rates whose spacing pins down the initial rate"*) but composition learning across patches is *"slow, set by the number of patches sampled rather than the time spent in each, and unaffected by depletion."* Reward-maximising and information-seeking strategies **diverge**; with replenishment the reward-maximising policy *"collapses onto a stable orbit over the high-yield patches."* | **Nothing promotes it to an arm — it is a WORLD-DESIGN reference, like ForageWorld.** It is **theory with no experiments, no subjects, no p-values, no agent** [V], so it cannot be a bakeoff arm and I am not proposing it as one. Its value is a pre-registerable constraint on `W1`: **if food depletes, the number of distinct patches — not time alive — bounds what Jack can learn about the world's composition**, and a world with few patches cannot teach it at all. `w1-world-edit-window` is OPEN and +1 OVERDUE with `W1.01`/`W1.03`/`W1.04` still unregistered; this is a design input for that window, offered to it and nothing more. |
| **2609.31161** — *I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms?* ([abs](https://arxiv.org/abs/2609.31161), 2026-09-25, Liu, Huang & Shi) | cs.LG | **A FIFTH independent group on the action-conditioning theme**, and the first to ask for the *condition* rather than propose another fix. Information-theoretic identifiability: representations recover causal states *"up to component-wise invertible transformations and permutation"* when there is **"sufficient action-induced variation in the transition mechanisms"**; instantiated as A-JEPA. | **A computable post-hoc diagnostic.** The condition as stated is not measurable on a trained model — *"the provided content does not specify a computable post-hoc diagnostic metric"* [V, abstract level] — so it cannot be an instrument, which is the only thing this desk needs on that seat. **No numbers, no seeds, no hardware, no params, no code.** Promote if a later version gives an estimator for action-induced variation; otherwise it is a theory that agrees with four measurements we already have. Week 5's rule applies: a theory nomination must carry its assumption list, and this one's assumptions are about the data-generating process, not about a trained `A4`. |
| **2609.19942** — *Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives* ([abs](https://arxiv.org/abs/2609.19942), 2026-09-17, Lee et al.) | cs.CL | **Recorded for its NEGATIVE result, in our exact task shape.** Extractive QA where questions were generated from the passages containing their answers, so *"retrieval recovers 92–99.8 % of what any mode combination could reach"* — and in that regime *"confidence-driven mechanisms have little to gain."* Two **pre-specified negatives** (a distillation trigger and a routing-and-abstention policy) both failed. AUROC 0.65–0.81; adaptation moved closed-book F1 *"by at most +0.03"*. **Code and data released** (two Zenodo DOIs). | **Nothing — it is a caution, not a candidate.** Its confidence signal is **generator-side sequence likelihood**, inadmissible here, so the method cannot enter. Its *warning* is worth carrying next to N2: **where retrieval already saturates, an abstention mechanism has no headroom to demonstrate anything.** Our fixture has `oracle_ceiling 1.0` but `answered_cues` **27.33 of 160** [M], so we are nowhere near their saturated regime — which is the check that keeps this from undercutting N2, and I ran it rather than assuming it. |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **2609.21787** — *Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models* | cs.RO | **Re-examined and its relevance went UP while its evidence stayed at zero.** Its subject — whether a compact latent geometry measured at one state transfers to another, *"a moving, state-dependent local geometry"* — is now the live question under both N1 and §6, since a `C_i` measured on one trajectory segment inherits exactly that worry. **Still no metrics, no seeds, no hardware, no params, no code**, and its result is stated across *"two of three checkpoints"*. **Promote on any number. Second sweep carried; per this desk's deferral rule it is dropped with cause next sweep if still numberless.** |
| **3M-Progress** ([arXiv:2506.00138](https://arxiv.org/abs/2506.00138)) | q-bio.NC | **DEMOTED FROM §2 TO THE WATCHLIST — queued #5 closed, third and last pass, exactly as week 8 committed.** Full HTML read: the paper reports **no agent exploration or survival metric of any kind** — no coverage, no states visited, no reward, no time alive [V, checked explicitly]. What it reports is alignment: *"3M-Progress agents captured nearly all of the explainable variance in neural and astrocytic activity, markedly outperforming existing intrinsic motivation algorithms"*, over **250,000 cells (~125 K neurons + 125 K astrocytes)**, **11 subjects**, behavioural state transitions discovered *"by 10 million environment steps"*; inter-animal alignment *"nearly 100 %"*; Figures 3/4A carry the comparisons against ICM, RND, Disagreement, γ-Progress, homeostatic, max-entropy and random. **The demotion is on GOAL.md's own terms, not on a technicality:** biology enters as a bakeoff arm and must win on *our* ruler, and a neural-fit win is not a survival win — *"planes do not flap."* The `CURIOSITY_BAKEOFF` hold the Review placed on it was right and made this costless. **Promote on: any exploration or survival number, from any source.** |
| **2608.29434** — *Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations* | cs.LG | **DROPPED WITH CAUSE — queued #8 executed.** Now identified in full (Oberweger & Schwingshackl, 2026-08-29, v1 only). Carried three sweeps at abstract level and there is still **no benchmark table, no seeds, no hardware, no parameter count and no code statement** [V]; the only numbers on the page (*"0.3–15 % of scene points moving"*) describe the dataset, not a result. Point clouds are not our regime either. Week 3's deferral rule applied without further argument. |
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | Unchanged, sixth sweep. Rollout-horizon deterioration in the fully observable state space — our regime. Still no compute cost, no hardware, no environment size, no code. **Promote on: any statement that a domain runs on CPU.** |
| **ForageWorld** ([arXiv:2506.06981](https://arxiv.org/abs/2506.06981)) | cs.AI | **Kept, and it now has a companion** — 2607.29476 above supplies the *theory* of depleting patches where ForageWorld supplies an *implementation*. Still Craftax/GPU and a gridworld where `W0` is a body. **Design reference only; it is not an arm and never was.** |
| **MINERVA** ([arXiv:2609.03715](https://arxiv.org/abs/2609.03715)) | cs.RO | Unchanged. 0.54 M params, 95.1 % on LIBERO, laptop CPU at 5–9 ms — still the only paper whose *inference* runs on our substrate class with a latency number, still **imitation learning from demonstrations**, the ground week 3 rejected DOOM-1.3M on. **Promote on: an RL result, or nothing.** |
| **PRIME** ([arXiv:2607.16858](https://arxiv.org/abs/2607.16858)) | cs.LG | Unchanged. Pseudocount epistemic term, free on a 10×10 grid, undefined on a continuous body state. |
| **IIBalance** ([arXiv:2603.17347](https://arxiv.org/abs/2603.17347)) | cs.MM | Unchanged, and now largely spent: `T4.06` has measured the balancing question on our own rig, which is better evidence than any classification paper. |
| **Eywa** ([2605.30771](https://arxiv.org/abs/2605.30771)), **ScrubJay-MEM** ([2608.04746](https://arxiv.org/abs/2608.04746)), **RARE/RedQA** ([2604.19047](https://arxiv.org/abs/2604.19047)), **CoDeR** ([2606.13204](https://arxiv.org/abs/2606.13204)), **Argus Eyes** ([2602.09616](https://arxiv.org/abs/2602.09616)) | cs.CL / cs.IR | **Front 3 was swept and none of these was re-opened.** Their dispositions stand from weeks 4–7 (generation in the write path for the first three; wrong failure axis for CoDeR). **They are superseded in role by N2**, which is the first of this family to keep generation out of the read path *and* report a latency. Not re-checked this sweep and not claimed to be. |
| **POBAX** ([arXiv:2508.00046](https://arxiv.org/abs/2508.00046)) | cs.LG | Unchanged. `W1.01` still unregistered, which is why it is still listed. |
| **Curiosity-Critic** ([arXiv:2604.18701](https://arxiv.org/abs/2604.18701)) | cs.LG | Unchanged. Weaker-evidenced sibling of LPM, already cited by `CURIOSITY_BAKEOFF.md`. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS

| nomination | entered | status now |
|---|---|---|
| **wk8-N2 · do-nothing reference arm (2609.11247)** | `t402`'s ordered bakeoff | **THE CONTROL IS IN THE SPEC AND THE SPEC RAN.** `T4.06`'s `null_baseline` is the incumbent re-run unchanged as arm zero, and its spec-level control is *"the incumbent evaluated under the winner rule MUST NOT win"* — which held (`ctrl_incumbent_wins 0.0`, `ctrl_incumbent_still_red 1.0` [M]). §0(1). **No citation of this page appears in the row and I claim no causation.** The nominated paper's prediction — that the balancing family barely beats doing nothing — is now measured here: **+0.0187**. |
| **wk6-N3 · IAF pathway-decoding control** | `t402` row | **PARTIALLY IN, AND THE MISSING HALF IS THE HALF THAT MATTERED.** `T4.06` measures per-modality latent R² *after* fusion (`latent_r2_per_seed`, all five senses, every arm [M]) — that is IAF's "after". It measures **no unimodal "before"**, so it cannot say whether fusion *degraded* a pathway or never carried it. The nomination survives, narrowed to exactly that: **a unimodal ceiling for each sense.** §6 explains why that is now the cheapest thing on this front. |
| **wk7 · §6 (A4's mandatory diagnostic computed nowhere)** | `REVIEW_QUEUE.md` row | **RULED (i)+(iii); (iii) EXECUTED 09-27; (i) CONTINGENT on the owner via `D37`**, row re-dated 10-04. §0(2). The desk's recorded leaning was (i)+(iii) and that is what it ruled. **§2's N1 is a candidate for the (i) half's readout choice.** |
| **wk8 · §6 (`LC.03`'s five controls never switch off `l_bind`)** | `lc03-five-controls-never-switch-off-the-term-a4-is-named-for` | **ROUTED BY CITATION, confirmed by the reader [M].** A row exists and carries it. Not re-argued here. |
| **wk8 · §6b (fieldwatch quotation channel 0-for-5)** | `fieldwatch-quotation-channel-is-0-for-5` | **ACTED 2026-09-24, and it bought more than it asked for** — `MIN_QUOTE_OVERLAP = 6`, header stripping, 0 spurious of 18, `decisions._shingles` untouched, **plus the discovery that `owner_asks` had parsed 0 items for 14 days.** §0(3)–(4). **Closed.** |
| **wk8-N1 · replanning-frequency ablation (2609.05461)** | the `a4` row | **LIVE, and named in the row as one of three items wanting the same 14.40 core-h.** Unrun, correctly sequenced behind `D37`. **Queued #7 CLOSED NEGATIVE: ARC-Bench's code is still not released** — the page is v1 (12 Aug 2026), with no repository, no release statement, and still no hardware or wall-clock anywhere [V]. The reimplementation risk this desk flagged as N1's largest unpriced part is **unchanged**. |
| **wk8-N3 · temporally-centered SIGReg (2607.26924)** | `A4c` amendment | **Unchanged. `A4b`/`A4c` are LIVE AND UNRUN for a NINTH sweep.** §5 declines a sixth route again. |
| **wk7-N1 · free-embedding critical-`d` probe (2508.21038)** | `ME.11` successor pre-gate | **ACCEPTED, ORDERED FIRST 09-14, STILL NOT RUN AT 14 DAYS** [M, §0(5)]. Not withdrawn and not re-argued: it is still the cheapest live item on the desk. **§2's N2 is its complement, not its replacement** — wk7-N1 asks whether the embedding geometry *can* work; N2 changes the decision rule over the scores that geometry already produces, so N2 survives a negative answer from wk7-N1. |
| **wk7-N2 · 3M-Progress (2506.00138)** | `CURIOSITY_BAKEOFF` arm | **WITHDRAWN FROM §2 TO THE WATCHLIST**, on this desk's own deferral rule, after the promised third pass found no agent performance number of any kind. §3. The Review's HELD status made the withdrawal costless. |
| **wk7-N3 · ActSWM `Δ_k` (2607.26712)** | merged into the `a4` row | **Unchanged, and now explicitly what the 10-04 sitting will hand to the builder if `D37` permits.** |
| **wk6-N2 · MULTIBENCH++ redundancy pre-gate** | `ub10` | **Unchanged.** Stands as a pre-registration on the hardened battery's next reading. `ub10` is ACTED but `T4.03` (*Fusion actually fuses*) is still unimplemented. |
| **wk5-N1 · SIGReg as variational free energy (2607.13612)** | `LEARNING_CORE` §5.4 | **Unchanged, in its wk8-amended temporally-centered form.** Its re-pricing from zero stands (the curves it was to be read on do not exist). |
| **wk5-N2 · prosociality by coupling (2604.10760)** | `NE.07` | **Unchanged.** Arm DEFERRED, shuffled-partner CONTROL accepted. `NE.07`/`NE.02` still never run. |
| **wk4-N1 · spectral-radius constraint (2607.19719)** | `A4` variant | **ACCEPTED, narrowed. Unrun.** |
| **wk4-N2 · PSG-JEPA (2608.06799)** | `A4` ×2 | **ACCEPTED as two arms. Unrun**, sequenced behind `D9`'s PARK. |
| **wk4-N3 · infant motor noise** | `W0.DIAG` | **RUN AND PASSED.** Still the only field-watch *arm-side* nomination in nine weeks to become a number — though §0(1) and §0(3) are now two *control-side* ones that did. |
| wk1 · anti-collapse regularisers → `A4b`/`A4c` | `A4` variants | **LIVE, UNRUN — ninth sweep.** |
| wk1 · certificate-gated identifiability → `UB.11` pre-gate | `UB.11` | **LIVE, unrun.** Still the only route to `UB.11`'s certificate. |
| wk1 · interoceptive precision (2608.04232) | `NE` §2.4b | **LIVE, unrun.** Still the cheapest item on the desk with released code, nine weeks running. |
| wk1 · entity-collision protocol (2605.29630) | `MR` §2 | **LIVE, unrun.** |
| wk2 · the whiff clock → `SM.02` | `SM.02` | **HELD, correctly — `SM.02` is PARKED.** |
| wk2 · RPE-prioritised replay → `NE.05` | `NE.05` | **LIVE, unrun.** |
| wk3 · CIG (2605.20878) → `A3` | `A3` | **Remains DEMOTED** (`wm-efe` t = 2.05). |
| wk3 · Optimistic World Models (2602.10044) → `A2` | `A2` | **Remains DEMOTED, hard** (`dreamer-xs` t = −0.94). |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

**FRONT 4 · CURIOSITY & OPEN-ENDEDNESS — NOTHING, SEVENTH CONSECUTIVE TIME, AND
THE CRITERION HAS NOT MOVED.** A 40-entry enumeration on intrinsic motivation /
intrinsic reward / autotelic / open-ended learning returned an agenda paper
(`2609.17325`), developmental-framework position pieces (`2609.11660`), LLM and
code-reasoning bonuses (`2608.07531`, `2606.20881`, `2606.19476`), federated RL
(`2608.10499`), multi-agent influence terms, and social-robot HCI. **Not one has
a body under homeostatic drive; not one evaluates an intrinsic reward against a
random or noise baseline.** Two are recorded without being fetched or
nominated: `2606.11417` (*Signed Compression Progress on a Sealed Audit is
Goodhart-Resistant*, cs.LG) is the only in-window item whose framing is about
an intrinsic signal resisting Goodharting, which is this project's standing
worry on this front; and `2605.22814` (*Remember to be Curious*, cs.LG) pairs
episodic context with persistent worlds, which is the cross-life shape `GOAL.md`
describes. **Neither was opened and no claim of theirs is relied on.** Front 4's
only live arm is now withdrawn to the watchlist (§3), so **this front has zero
live nominations for the first time since week 7** — and the honest reason is
that `CU.1`–`CU.7` remain seven specs with none implemented, which ages an arm
faster than any literature can supply one.

**FRONT 5 · WORLDS & EMBODIMENT — NO ARM, AND THE WORD "SURVIVAL" COST ME A
THIRD SWEEP.** The 40-entry enumeration on survival / homeostatic / foraging
returned quantisation papers, malware detection, tokamak disruption alarms,
crystal-structure prediction, and **two medical survival-prediction papers**
(`2609.25088` GBM survival, `2609.21811` MIST) — week 2's lesson recurring for
the third time, now as a search discipline that *still* does not fully filter.
The only on-axis items were locomotion controllers (`2609.27001` fly-inspired
recurrent, `2609.25687` severity-gated CPGs), and neither has needs, death or a
fidelity claim. **The MuJoCo ecosystem check (a standing mandate item) returned
the same answer as week 3:** the live developments are **MuJoCo Warp** and
**MuJoCo Playground**, both **GPU-accelerated parallel-environment** plays — the
axis `SURVIVAL_WORLD` §2.2 ruled out, and we run one life serially on four
shared ARM cores. **The genuinely useful front-5 item this sweep came from
`q-bio.NC`, not `cs.*`** (§3, 2607.29476), which is the second consecutive
sweep where the biology category outperformed the computer-science ones on a
front that is not nominally about biology. `W1.01`/`W1.03`/`W1.04` **remain NOT
REGISTERED** and `w1-world-edit-window` is OPEN and +1 OVERDUE at 22 days.
**These instruments are still ours to build.**

**FRONT 1 · LEARNING CORES — A NOMINATION FROM BIOLOGY, AND THE SAME REFUSAL AS
LAST WEEK ON THE SAME GROUND.** The 40-entry enumeration returned a genuinely
crowded field: `2609.24749` (D-JEPA, decision-aligned latent world model),
`2609.30264` (AD-WM, action-discriminative), `2609.23252` (*Robot World Models
Are Not Invariant to How the Actions Are Written*), `2609.29171`, `2609.22816`
(FIRM-WM), `2609.15770` (JEPLO, LiDAR legged locomotion), `2609.25541` (a JEPA
recipe for tabular models). **I nominate none of them as arms**, for the reason
week 8 gave and which has only got stronger: this desk has promoted **five**
anti-collapse / latent-structuring routes for `A4` and **run zero**, now across
nine sweeps. `2609.31161` goes to the watchlist as a *fifth* independent group on
the action-conditioning theme — note that the theme itself is now
well-corroborated and **corroboration is not news, seventh time of saying it**.
**N1 enters this front from `q-bio.NC` instead, and it enters as a READOUT for
an owner-bound decision rather than as a sixth challenger to a seat** — which is
the distinction that makes it not padding.

**FRONT 2 · FUSION — NO METHOD, FIFTH CONSECUTIVE SWEEP, AND THIS TIME THE
REFUSAL IS BACKED BY OUR OWN LEDGER INSTEAD OF SOMEBODY ELSE'S.** The probing
and imbalance enumerations returned classification, recommendation, medical EHR
and autonomous driving, as in every prior sweep. What changed is that **the
question no longer needs the literature**: `T4.06` measured the balancing
family on our own rig, and the answer was one arm winning by +0.0187 with the
`grad_norm` arm's ratio equalised by construction. **The Goodhart objection this
desk raised four times was correct in its prediction and is now spent as an
argument** — the gate moved (29.83 → 2.45) and §6 is about whether the capability
did. N3 is a control, not a method. **The probe-control-task literature I went
looking for specifically does not exist in window**, which is why §6 had to
compute its own floor.

**SMALL-MODEL END — NOTHING IN WINDOW, and MINERVA remains the only item in
nine sweeps whose inference runs on our substrate class.** The 30-entry
enumeration returned VLA residual-RL, driving world models, speculative
decoding and grippers; nothing under 1 M parameters with an RL result. Our own
`ppo-needs` at **135,961** params and `wm-latent` at **861,545** [M, week 7]
remain smaller than anything this literature is proud of.

**BIOLOGY-AS-ORACLE — the standing `q-bio.NC` search produced BOTH of this
sweep's non-`cs.LG` items and is now the highest-yield slot on the desk.** N1
and the front-5 watchlist entry both came from it. Also returned and **not**
fetched: `2606.00667` (*Cortex and subcortex play distinct roles over learning
when cortical memory is limited* — the hippocampus/cortex split `GOAL.md` names
as the diary-vs-weights oracle), `2609.16217` (a neural-**astrocyte**
architecture implementing a hybrid automaton, converging with 3M-Progress's
astrocytic story from a different direction), `2606.26733` (*Surviving by
Serving*, bearing on wk5-N2's prosociality route), `2609.02243` (*Mus
siliconus*, carried), and `2607.13560` (*Grounded world models in biological
organisms and future embodied AI*). **None is nominated and none was opened.**
The slot keeps its place and has now earned it twice.

---

## 6. A FINDING IN OUR OWN ARTIFACTS — `T4.06`'s deciding statistic was read against a floor that nobody computed, the floor is `−r/(n_fit − r)`, and the *other* half of the spec's own open question has an answer

Week 3's rule: a scout reading our own ledger has no abstract to doubt, so the
finding must carry the arithmetic that survives rather than the story that
motivated it. Everything below is reproducible with NumPy and no GPU.

**WHAT IS ALREADY OWNED, STATED FIRST AND IN FULL, BECAUSE MOST OF THIS IS.**
The 109th audit (2026-09-23) caught the margin problem the day the row landed,
`LESSONS.md` §16550 records it as a two-part general rule, `experiments/resolution.py`
computes and prints margin-over-spread for exactly this class of conjunct, and
`run status` has printed it since the 110th audit. **The spec's own docstring
already says the thing a reader would expect me to announce** [V, lines 45–60]:

> *"the anchor ARRIVED at `min_modality_latent_r2 = −2.3939` … the
> latent-recovery conjunct certified the winner INSIDE the anchor's noise and
> must not be quoted as demonstrated. … In every arm including the winner, four
> of five senses read latent R2 in [−2.38, −0.17] from the fused
> representation, and only proprioception is positive — **whether that is the
> brain or a ridge probe fitting 513 params to 768 rows is not answerable from
> this run and is now askable.**"*

`LESSONS.md` goes further and asserts the mechanism: *"at p/n = 0.67 with
regularisation that small, a held-out R² near −2 is what variance alone
produces."* **I am not reporting any of that as new.** What I am reporting is
that the question the docstring declares unanswerable **is answerable, at zero
compute, and the two halves of it come apart.**

**THE RECONSTRUCTION.** I replicated the probe's arithmetic exactly — same
solver (`t4_06…py:408–420`), `RIDGE_LAMBDA = 1e-3`, `PROBE_N = 1152`,
`n_fit = (2·1152)//3 = 768`, test 384, design width `d_model = 512` → **513
columns with the bias**, target `k = 8`, R² averaged per dimension — and fed it
designs whose answer is known by construction, 5 seeds each [M]:

| condition | measured R² |
|---|---|
| **NULL**: X iid Gaussian, z independent | **−1.99** [−2.05, −1.83] |
| **NULL**: X LayerNorm-shaped, z independent | **−1.97 … −2.13** |
| POSITIVE: `X = z @ R`, LayerNorm-shaped | **+0.94** |
| POSITIVE: `X = 0.5·(z @ R) + noise` | **+0.80** |
| POSITIVE: `X = 0.25·(z @ R) + noise` | **+0.37** |
| POSITIVE: `X = 0.1·(z @ R) + noise` | −0.88 |

**ANSWER TO HALF THE SPEC'S QUESTION: IT IS NOT A CAPACITY ARTEFACT.** At the
*identical* 513 params / 768 rows / λ = 1e-3, a linearly present signal is
recovered at **+0.94**, and stays positive down to a quarter-amplitude signal
buried in full-rank noise. **The probe is a working instrument**, so "the
magnitudes are probe artefacts" is too strong a reading of the row: the
*floor* is an artefact of the regime, the *instrument* is not broken. That
distinction matters because it decides whether the fix is a better probe or a
declared floor.

**AND THE FLOOR IS NOT A CONSTANT — IT IS SET BY THE EFFECTIVE RANK OF THE
FUSED REPRESENTATION, IN CLOSED FORM.** I stress-tested my own lead objection
(that the floor was measured on synthetic X). Correlation, tails and scale do
**not** move it; **rank is the only structural knob that does** [M, 5 seeds
each]:

| structure of X (z independent throughout) | measured floor |
|---|---|
| all 512 dims correlated ρ = 0.5 / 0.9 / **0.99** | **−2.06** (unchanged across ρ) |
| heavy-tailed (Student-t, df = 3) | −2.03 |
| unnormalised, scale ×10 | −2.15 |
| z correlated across its 8 dims | −2.05 |
| **effective rank 256 of 512** | **−0.53** |
| **effective rank 64** | **−0.09** |
| **effective rank 16** | **−0.02** |

And the whole table is one formula. With `n_fit = 768` and effective rank `r`,
the no-information floor is **≈ −r/(n_fit − r)** [C, and [M] against it]:

| effective rank `r` | measured floor | `−r/(768 − r)` |
|---|---|---|
| 512 | −2.011 | **−2.000** |
| 384 | −1.044 | −1.000 |
| 256 | −0.526 | −0.500 |
| 128 | −0.213 | −0.200 |
| 64 | −0.090 | −0.091 |
| 32 | −0.047 | −0.043 |
| 16 | −0.026 | −0.021 |
| 8 | −0.012 | −0.011 |

**THREE CONSEQUENCES, offered as observations and not as decisions.**

**(1) The bar itself sits at the full-rank no-information floor.** The
incumbent's `min_modality_latent_r2` bar is **−2.3939** (vision) [M]. The
formula's full-rank value is **−2.00**, and my ten full-rank draws spanned
−1.83 to −2.25. So the quantity `T4.06`'s winner rule compared two arms on is,
for the worst modality, **at or past the value a probe returns when the
representation carries nothing at all**. `loss_reweight` beat it by **+0.0187**.
This does not contradict the 109th audit — it supplies the reason the margin
*had* to be small against the spread: **a statistic pinned at its own
no-information floor has only noise left to vary.**

**(2) Which senses are readable depends on a rank the run does not record, and
both answers are consequential.** `latent_r2` for the incumbent [M]: proprio
**+0.7172**, touch **−0.2154**, audio **−2.1045**, language **−2.2408**, vision
**−2.3939**.

- *If the fused CLS vector is near full rank*, the floor is ≈ −2.00, so
  **vision, language and audio are indistinguishable from carrying no linear
  information**, while touch and proprio require some information to explain.
- *If it is low rank* — which is what representation collapse **is**, and what
  `A4`'s declared-but-unbuilt diagnostic exists to detect — the floor rises
  toward −0.02, and then **touch falls below its floor too, leaving
  proprioception as the only sense above it.**

**The robust statement, true either way: at most two of five senses are
linearly recoverable from the fused representation, and which two is not
determinable from the committed row.** That is `GOAL.md` stage 4 territory
(*"senses fused; each proven load-bearing; no modality collapse"*) and `T4.03`
(*Fusion actually fuses*) is still unimplemented.

**(3) A TYPED floor constant would be the wrong repair, and that sharpens the
lesson already on the books.** `LESSONS.md` §16550's rule 1 says an in-run bar
needs an unsaturated-null lane at **both** ends; `T4.06` armed only the ceiling
(`R2_SATURATION = 0.99`). My measurement says the low end **cannot** be a typed
number, because it moves by two orders of magnitude with a property of the
representation that can differ *between arms of the same bakeoff*. **The floor
has to be measured in-run**, and the cheapest form is the one this project
already uses everywhere else — `T1.02` is literally *Shuffled-target control*:
**permute `Z`'s rows against the same `X` and re-fit.** That is one line, it
needs no new run, it costs no parameters, and it yields the floor for the actual
fused vector rather than for my synthetic stand-in. `T4.06` has **no shuffle
and no permutation anywhere** [M, grep returns nothing].

**THE CONVERGENCE, and it is why this is in the same report as N1.** The
quantity that sets this floor — **the effective rank of the fused
representation** — is the *same* quantity `LEARNING_CORE.md` §5.4 declares as
`A4`'s mandatory diagnostic and which week 7 §6 found computed nowhere. **Two
different seats, one uncomputed number**: on the `A4` seat its absence left a
declared VOID condition unarmed; on the fusion seat its absence leaves a PASSED
spec's deciding statistic without a floor to be read against. I did not go
looking for that link; it fell out of testing my own objection.

**LEAD OBJECTION, AGAINST MYSELF.** **My floors are measured on synthetic
designs, not on the real fused CLS vectors**, whose rank I did not measure and
could not have without training the brain. So I cannot say *which* row of my
table applies, and every consequence above is therefore conditional in exactly
the way it is written. **That is not a hedge, it is the nomination**: the repair
is not to compare `T4.06`'s numbers against my −2.00, it is to make the run
report its own shuffled-label floor and the rank that predicts it. A second,
smaller caveat: the fit/score split is **sequential** (first 2/3, last 1/3), not
random, so any non-stationarity in the fixture draws would push the real floor
*below* my estimate; the fixture draws iid per batch, so I expect this to be
small, but I did not measure it and do not claim it is zero.

**WHAT I AM NOT CLAIMING.** Not that `T4.06` should not have passed — its
`n_winning_arms ≥ 1` claim and the **ratio** result (29.83 → 2.45 against the
exogenous 10× gate, on three seeds, with the incumbent's control correctly
refusing to win) are demonstrated and untouched by any of this. Not that the
winner is wrong. Not that anything should be re-run. **The single sentence of
this section is: the spec asked whether its negative R²s were the brain or the
probe, said the question was not answerable from that run, and it is answerable
for free — the probe works, and the floor is `−r/(n_fit − r)` for a rank nobody
recorded.**

---

## 7. What this report does NOT claim

- **No arm here has been run.** Every number in §2 and §3 is someone else's
  measurement on someone else's hardware. The §6 probe reconstruction, the
  `ME.11` threshold table, the `T4.06` row reads, the queue and reader
  readings, and the `LATENT`/params figures are **ours**, marked **[C]/[M]**,
  with the arithmetic and commands shown.
- **§0(1) claims a coincidence, not a cause.** `T4.06` was designed two days
  after week 8 nominated the reference arm it carries, and **no citation of this
  page appears in that row.** I state the sequence and decline the credit.
- **§6 is mostly NOT new and says so first.** The margin defect, the mechanism
  sentence about p/n, the reporter, and the observation that four of five senses
  read negative were all already owned by the 109th audit, `LESSONS.md`,
  `resolution.py` and the spec's own docstring. **What is new is the measured
  floor, the closed form, the rank dependence, and the answer to the capacity
  half of the spec's own question.**
- **§6's floors are synthetic.** I did not measure the real fused
  representation's rank or its shuffled-label floor, so every consequence in §6
  is conditional on which rank regime holds, and both branches are given.
- **N1's paper reports no p-values, no CIs and no compute**; its `ρ_depth`
  needs a task scalar `W0` does not have, and the transition-direction label
  `C_i` needs would be **ours to define** — I say so in N1 rather than in a
  footnote, because a readout whose labels the analyst chooses is how a
  diagnostic becomes a story.
- **N2 is supervised and our labelling budget is 160 cues against the paper's
  smallest 486.** I also state plainly that 16 cues/parameter is **not** the
  same disease as §6's 1.5 rows/parameter, because applying my own finding
  unfairly to a nomination I like would be as bad as the reverse.
- **N3 is abstract-level only**, and its defect (similarity metrics fooled by
  corruption) is **not** shown to be our defect — §6 measures that our probe is
  *not* capacity-limited. Only the **discipline** transfers, and week 3's rule
  says an analogy is not arithmetic.
- **3M-Progress was withdrawn by me, on my own deferral rule**, after three
  passes. That is a nomination of mine failing its own test, and it is recorded
  in §4 as a withdrawal rather than quietly dropped from §2.
- **Front 4 was enumerated but not fetched; front 5 was enumerated and searched
  but not fetched.** Two front-4 items (`2606.11417`, `2605.22814`) and five
  `q-bio.NC` items are named without being opened, and **no claim of theirs is
  relied on anywhere in this report.**
- **The `T4.06`, `ME.11`, `a4`, `t402` and queue states are the builder's and
  the Review's records, not mine.** I read them out of the ledger, the queue and
  `run status`; I did not run them.

---

## 8. Queued for next sweep (**not before ~2026-10-05**)

1. **Did `D37` rule, and did the `a4` row's 10-04 sitting hand anything to the
   builder?** That date falls the day before this embargo expires, so it is the
   first thing to read. **If `D37` permitted the BUILD half, N1 and wk7-N3's
   `Δ_k` are both live readout choices for it and §2's N1 should be re-costed
   against whatever run was authorised.** If it did not, the seat's guard is
   unarmed for a sixth week and that is the number to print.
2. **Does `T4.06`'s successor — or any row — carry a shuffled-label floor?**
   §6 is this sweep's finding and the reader will report whether it reached a
   desk. The specific check is cheap: `grep -n "shuffle\|permut"` in the fusion
   spec, and whether any row records the fused representation's **effective
   rank**. **If the rank is ever recorded, §6's conditional collapses to one
   branch and I should say which.**
3. **FRONT 3 next returns 2026-10-12** on its two-week cadence, and it has a
   question rather than a survey: **did wk7-N1's critical-`d` probe run, and did
   N2's feature class enter anywhere?** Two nominations now sit on the same
   redesign, both cheap, one of them 14 days unrun. **If both are still unrun on
   10-12 that is the finding, not the literature.**
4. **`w1-world-edit-window` (OPEN, +1 d, 22 days old) and `W1.01`/`W1.03`/
   `W1.04`.** The front-5 watchlist entry (2607.29476) is a design input for
   that window and will be stale if the window opens without it. **Also check
   whether `w0-too-shallow`'s 10-01 date held**, since it carries the
   registration of those three specs.
5. **`owner-ask-reader-blind-since-0909` (DUE 10-01)** — the sibling bug my own
   §6b measurement turned up. Worth one line next sweep: did the second blind
   reader get fixed as fast as the first one did?
6. **`q-bio.NC` standing search — upheld and now load-bearing.** It produced
   both of this sweep's non-`cs.LG` items and the only on-axis front-5 entry.
   **Plus one addition of the same kind:** `2606.00667` (cortex/subcortex under
   limited cortical memory) is the closest thing yet to a biological oracle for
   the diary-vs-weights split, and it is queued for a fetch rather than left as
   a name in §5.
7. **2609.21787 — promote on numbers or DROP WITH CAUSE.** Second sweep carried,
   relevance rising, evidence still zero. Week 3's deferral rule applies.
8. **NOT queued, deliberately:** conference proceedings (dropped wk4); 2607.22430,
   Simulus, UED-as-a-cheap-score (closed wk6); LIMIT's *conclusion* (closed
   wk7, instrument retained); 2608.29434 (**dropped with cause this sweep**);
   3M-Progress's numbers (**closed this sweep after three passes**);
   ARC-Bench's code (**checked and unreleased; will not be re-checked weekly —
   it is now a promote-if-it-appears, not a queue item**); **and the
   anti-collapse family — four more routes appeared last week, seven more this
   week, and §5 declines all of them until one of the five already accepted has
   been run.**
