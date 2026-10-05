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

**Sweep date:** 2026-10-05 · **Window:** ~2026-04 → 2026-10 (6 months)
**Scout:** field watch, week 10. **Seven days since the week-9 attempt**
(2026-09-28), which **exited `rc=124` and was sealed as a draft** — so by the
only measure that counts, **the last COMPLETE report on this page is week 8,
2026-09-21, fourteen days ago.** §0 is about what that cost and what it did not.

**Confidence markers:** **[V]** fetched and read · **[V†]** fetched, but the
page's content reached me through the fetch tool's summarising model rather than
read line by line — weaker than [V], stronger than [s], **new this sweep and
declared in §1** · **[c]** claimed by the authors, not checked against their
table · **[s]** asserted by a search engine *about* a paper I have not opened ·
**[C]** computed by me, arithmetic shown · **[M]** measured here, on this box,
command shown.

---

## 0. WHAT MOVED IN THE INTERVAL — a sweep died, and the organ around it worked better than the organ itself

**196 commits** landed in the seven days since the week-9 attempt [M,
`git log --since=2026-09-28 | wc -l`]. Six facts re-point this sweep.

**(1) WEEK 9 DIED AT `rc=124`, AND THE REVIEW RESCUED IT WITHOUT TRUSTING IT.**
`scripts/lib_seal.sh` bannered the page *"INCOMPLETE RUN — THIS IS A DRAFT, NOT
A FINDING"*. Four Review sittings skipped it. Then two things happened that are
the system behaving exactly as `SYSTEM.md` says it should:

- **2026-09-30** lifted §6 — the finding about *our own* ledger — out of the
  draft and gave it an id (`t406-latent-floor-was-never-computed`, OPEN, **DUE
  2026-10-05**, i.e. today), on a stated distinction I accept and will hold
  myself to: *"That reasoning is correct and it applies to NOMINATIONS …
  It does not apply to §6, which is a finding about OUR OWN LEDGER: its inputs
  are `T4.06`'s recorded metrics and its claim is arithmetic, both of which this
  desk can verify from the repo without trusting the draft at all."*
- **2026-10-02** consumed all three nominations into `docs/INTEGRATION_QUEUE.md`
  — N1 as a **dormant conditional** (fires only if `D37` permits a readout), N2
  as a **design unit** (the `MEMORY_RETRIEVAL_BAKEOFF` §1.8 arm), N3 as an
  **admitted control discipline** — and routed the structural trap separately as
  `field-watch-rc124-page-is-untrustable-and-then-deleted` (OPEN, DUE 10-13),
  whose own text names the deadline it was racing: *"Monday 2026-10-05's sweep
  would have deleted all of it."*

**So nothing perished, and I did not have to re-report week 9 to save it.**
What I owe instead is narrower and I do it in §4: say which of those
nominations survive a second look, and correct the two that do not.

**(2) THE ONE THING THAT DID NOT GET RESCUED IS THE DRIFT LEDGER, AND IT IS
MINE.** `grep -c "^2026-09-28" docs/FIELD_WATCH_LOG.md` returns **0** [M].
Week 9 wrote a full page and **never appended a single line to the append-only
log** — the file whose entire purpose is *"so that a reader can see what the
scout believed on each date without diffing a rewritten file."* The `rc=124`
row's discharge covers the PAGE (copied into the integration queue); it does
not cover the LOG, and no instrument reads the log, so nothing noticed. The log
had a **14-day hole** with the organ's own fire times looking healthy.
**Repaired in this commit** by appending the week-9 lines retroactively, marked
as reconstructed from the sealed draft — §7b.

**(3) `D37` IS OVERDUE AND ITS DEFAULT IS DUE TO FIRE — QUEUED #1 CLOSES
NEGATIVE.** [M, `python -m experiments.decisions --check`]: `D37` reads
**`OVERDUE — DEFAULT IS DUE TO FIRE`**, and its default is **(iii) HOLD `D29`
AS IT STANDS … and the `Δ_k` readout is NOT built**, which the entry itself
calls *"the only legal default and deliberately NOT the recommendation."* The
`a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere` row is
DISPOSITIONED and **re-dated from 10-04 to 2026-10-19** [M, `run review-queue`].
Nothing was handed to the builder on 10-04.

**So at HEAD, 21 days after week 7 found it: `grep -rniE "effective_rank|rankme"
--include='*.py' .` returns FOUR hits and NOT ONE COMPUTES IT** [M] — two are
inside `experiments/fieldwatch.py`, which is the reader quoting
`LEARNING_CORE.md`'s promise back at us, and two are prose inside other specs'
hypothesis strings. The seat's declared VOID condition has now been unarmed for
six weeks, and the entry's own price paragraph says so plainly. **§2's N2 enters
exactly here**, and I price it knowing the venue may never open.

**(4) MY OWN §6b CLOSED ITS SIBLING TOO.** `owner-ask-reader-blind-since-0909`
is **ACTED 2026-10-02 (`d95b587`)**, and `decisions --check` now prints **4
owner-asks matched to decisions** where it parsed 0 for fourteen days [M].
Queued #5 closes positive: the second blind reader was fixed, and it was found
by measuring the first one's false-positive rate. One unrelated red is printed
here because it is in the same instrument and is not mine: **`RATCHET BROKEN: 1
DEFAULT-ACTION-EXPIRED, baseline 0`** (`D33`) [M].

**(5) THE QUEUE IS STILL DRAINING NOWHERE, AND NOW THERE ARE TWO ROWS FOR ONE
FINDING OF MINE.** [M, `run review-queue`]: **53 OPEN, 3 HELD, 29
DISPOSITIONED, 40 ACTED, 2 DECLINED of 127 routed; 85 live rows; arrived 15
against disposed 17 over the trailing 7 days; drain 298 cycles; oldest live
42 d.** Nine rows share 2026-10-09 against a measured capacity of 6/cycle.
And week 9's §6 was routed **twice** — `t406-latent-floor-was-never-computed`
(09-30, DUE today) and `t406-deciding-statistic-read-against-an-uncomputed-floor`
(10-02, DUE 10-13) — both citing the same INCOMPLETE page, the second dated onto
a day already carrying 8 promises. **I report the duplicate; de-duplicating the
queue is not a scout's act.** §6 answers both of them at once.

**(6) FRONT 5's WINDOW WAS REFUSED, NOT MISSED.** `w1-world-edit-window` is
**DECLINED 2026-09-28** [M] — the authorship was declined in the open on
`docs/PROGRESS.md` at the 09-27 FULL, exactly as that row's own stop-rule
pre-committed. `W1.01`, `W1.03`, `W1.04` are **NOT REGISTERED** in `BY_ID` [M],
and seven rows now sit behind the refused window
(`seven-rows-are-held-behind-a-refused-window-and-a-moot-decision`, OPEN, DUE
10-11), and `review_queue --check` exits **2** on exactly that:
**`7 VIOLATION(S) — HOLD-ON-A-RESOLVED-BLOCKER 7`** [M]. Front 5's instruments
are not waiting on literature; they are waiting on a decision. §5 says so
rather than nominating into a closed venue.

**Three instrument exit codes, re-run after the last edit of this sitting and
none of them caused by it** — `review_queue.py` does not read this page at all
[M, `grep -c FIELD_WATCH` returns 0]: `fieldwatch --check` **rc=0**,
`review_queue --check` **rc=2** (the 7 holds above), `decisions --check`
**rc=1** (`RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED`, `D33`, baseline 0).

---

## 1. Coverage — what was actually searched, so the gaps are visible

The arXiv API answered on the first attempt and **six 40-entry enumerations plus
two of 40 ran** (eight queries, 14 result sets). Week 8's operational note holds:
`https://` returns 200, bare `http://` returns a 301 with an empty body.

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **1 · LEARNING CORES** | world models × latent/MBRL; JEPA × action/control; sample-efficiency × continuous control | **3 enumerations (40×3)**; **2 abstracts [V]** (2609.33940, 2609.36227); 2 PDF attempts **failed** |
| **2 · MULTIMODAL FUSION** | multimodal × fusion × binding/load-bearing/ablation; unified tokenisation / shared latent × modalities | **2 enumerations (40×2)**; **no fetch** — nothing survived the title line |
| **3 · MEMORY** *(off-cadence: front 3 is on the endorsed two-week cycle, next **2026-10-12**)* | episodic memory × agent/retrieval; abstention / selective prediction × retrieval | **2 enumerations (40×2)**; **1 abstract [V]** (2609.34677) |
| **4 · CURIOSITY & OPEN-ENDEDNESS** | intrinsic motivation / intrinsic reward / autotelic / open-ended learning; environment generation / UED / curriculum generation | **2 enumerations (40×2)**; **1 abstract [V] + 1 HTML [V†] + the repository [M]** |
| **5 · WORLDS & EMBODIMENT** | homeostatic / foraging / survival agent / embodied survival; **MuJoCo × benchmark/environment/simulator** (standing mandate item) | **2 enumerations (40×2)**; **1 abstract [V]** (2608.11506); no other fetch |
| Biology-as-oracle | **`q-bio.NC` standing search** (wk7 §7) × RL / world model / embodied / curiosity / exploration, **and** × development / infant / play / critical period | **2 enumerations (40×2)**; **1 abstract [V]** (2609.36097) |
| Small-model end | tiny / compact / lightweight × policy / world model / RL | **40-entry enumeration**; no fetch |
| Queued closures | `D37`; the shuffle/rank greps; wk7-N1's grep; `W1.0x` registry; 2609.21787 | **[M] — commands shown in place** |
| Our own artifacts | `T4.06` row (29 metrics) + its probe implementation at HEAD + `t4_02`'s fixture; `ME.11` family rows; `decisions --check`; `review_queue --check`; `fieldwatch --check`; **a from-code reconstruction of `T4.06`'s ridge probe at 8 seeds/point** | **[M]/[C] — §6, every constant read from code** |

**Known gaps, stated so nobody assumes coverage:**

- **A NEW MARKER, AND IT IS A DEMOTION OF MY OWN PAST WORK.** Two of this
  sweep's pages were read *through the fetch tool's summarising model*, which
  returns an extraction rather than the text. That is not the same epistemic
  object as reading a paper, and marking it [V] would flatten the difference the
  way week 4's `[s]` marker was invented to stop. **It is marked `[V†]` here,
  and verbatim-quoted abstracts stay [V]** because the abstract came back in
  full and unparaphrased. This is a reporting convention on my own page —
  reported, not asked.
- **BOTH PDF FETCHES FAILED AND ONE NOMINATION PAYS FOR IT.** `arxiv.org/pdf/`
  returned compressed streams the fetcher could not decode, `arxiv.org/html/` is
  **404** for 2609.33940, and `ar5iv` redirects back to the abs page. There is
  no `pdftotext`, `pypdf` or `fitz` on this box [M]. **N2 is therefore
  ABSTRACT-LEVEL ONLY** — no numbers, no benchmark names, no seeds, no
  hardware, no params, no code statement — and it is nominated on its *shape*,
  which I say in N2 rather than in a footnote.
- **Front 2 was enumerated but not fetched. Front 5 was enumerated and
  MuJoCo-searched but only one abstract was opened.** Nothing survived the
  abstract line. §5 says so rather than manufacturing a fetch to look thorough.
- **Front 3 is off-cadence and was swept SHALLOWLY on purpose** — week 9 swept
  it properly and its nomination was consumed, so the two-week clock holds and
  the next real front-3 sweep is 10-12. One watchlist entry came out of the
  light pass; no front-3 nomination is made.
- **No conference main-track enumeration** (dropped permanently, week 4; still
  an acknowledged permanent gap). **No non-English sources.**
- **I did not build the Lean project behind N1.** A grep is not a compile, and
  §2's N1 says exactly that where it reports the number.

---

## 2. NOMINATIONS

Four. **One is front 4's first candidate in SEVEN sweeps and arrives with the
best-verified evidence this desk has ever handled; one is an instrument for the
`A4` seat with the worst provenance nominated in ten weeks, and I say so; one is
a control on the ablation discipline `GOAL.md` mandates on page one; and one is a
CORRECTION that weakens two live nominations of mine.** Each states its arXiv
primary category, its evidence class, its cost on **our** substrate, and both
sides steelmanned.

---

### N1 — The ALARM counterexample family as a KNOWN-ANSWER pre-gate for `CURIOSITY_BAKEOFF`, because every objective family our arms belong to is proved deficient on it

**Source:** *When Do Intrinsic Rewards Lead to Exploration?* —
[arXiv:2610.02159](https://arxiv.org/abs/2610.02159), primary category
**cs.LG**, 2026-10-01, Viteri, Gomezjurado Gonzalez & Barrett (Stanford).
**Abstract [V] verbatim; construction and theorem statements [V†]; the
repository [M], downloaded to this box and measured.**

**Why it is nominated where it is.** Front 4 has returned nothing for six
consecutive sweeps, and weeks 5–8 all named the same reason in the same words:
*"zero evaluate an intrinsic reward against a random or noise baseline."* This
paper does something better than a baseline — it supplies a venue whose answer
is **computed in advance and machine-checked**.

**THE VERIFIED CONTENT [V, abstract].** *"Maximizing these rewards need not
produce the most informative experience available. We propose a formal criterion
for exploration that compares policies by the counterfactual information they
acquire… We construct a single, simple environment in which specified
count-based, prediction-error, empowerment, and information-gain objectives have
maximizing policies that are Pareto-suboptimal at acquiring counterfactual
information. We explain these failures and establish conditions under which
existing intrinsic rewards successfully encourage optimal exploration."*

**THE ENVIRONMENT, and it is small enough to matter [V†].** The **Alarm**
environment: worlds `θ ∈ ℕ ∪ {∞}`, **three actions** `{INSPECT, PLAY0, PLAY1}`,
**four observations** `{0,1,2,3}`, alternating display and alarm ticks, prior
`α(∞)=1/2`, `α(k)=2^{-(k+2)}`. Per-objective deficiency `D(π)`, all at their
**optimal** policies: pseudo-count `1/2` (finite `H ≥ 64`); predictive Shannon
`1/2` (all `0<γ<1`); predictive square `1/2`; sensor empowerment `1/2` (all
fixed `n≥1`); and on separate small MDPs state-visitation entropy `1/2`, MOP
`1/3`, **DIAYN `∈ [1/4, 1/2]`**.

**THE EVIDENCE CLASS IS THE BEST THIS DESK HAS SEEN, AND I MEASURED IT RATHER
THAN BELIEVING IT [M].** The paper names
`github.com/scottviteri/what-is-exploration/tree/arxiv-v1`. I fetched the
repository (87.5 MB; `/data` at 42 G free against the 15 G floor; **deleted in
the same iteration**) and measured the Lean supplement:

| quantity | measured [M] |
|---|---|
| `.lean` files | **321** |
| `theorem`/`lemma` declarations | **4,150** |
| `sorry` occurrences | **0** |
| `admit` **in tactic position** | **0** |
| `axiom` **declarations** | **0** |
| registered claims in `PAPER_CLAIMS.json` | **42** |
| … at level `lean-full` | **38** |
| … at level `numerical-evidence` (NOT Lean) | **3** |
| Lean declarations mapped to claims | **1,028** |
| toolchain | `leanprover/lean4:v4.33.0`, mathlib pinned in `lake-manifest.json` |

And the per-objective counterexamples are individually registered at
`lean-full`: `claim:alarm-panel-pseudocount-long-horizons` (9 declarations),
`claim:alarm-panel-predictive-discounted` (17), `claim:alarm-panel-average-empowerment`
(22), `claim:alarm-panel-multistep-discounted-empowerment` (12),
`claim:state-occupancy-entropy-mdp` (35), `prop:mop-finite-mdp` (31), and
**`thm:diayn-state-counterexample` at 90 declarations** [M].

**This is the claim week 5 made for SIGReg and week 6 had to downgrade** —
*"described is not downloadable."* This one is downloadable and I downloaded it.
**The honest ceiling: a grep is not a compile.** There is no Lean toolchain on
this box and installing one is not a scout's job, so the `0 sorry` is a
measurement over the **sources**, not a successful `lake build`.

**Where it lands.** `CURIOSITY_BAKEOFF`'s arms are `disagree` (ensemble
prediction error), `lp` (learning progress), `metra`, `vlm-lp`, with `RND` as a
must-fail control and `A3`/`wm-efe`'s EFE epistemic term as the
information-gain member. **Four of the families named deficient are families we
race.** The nomination is **not an arm**: it is a **pre-gate** — run our
intrinsic-reward implementations on Alarm, where the deficiency of the ideal
objective is known, before reading their W0 numbers as evidence about
exploration.

**THE FALSIFIABLE REASON IT MIGHT WIN.** `T3.06`'s committed row measured that
in W0 **a random-action policy covers the world as well as the curious arm**
(curious-RANDOM t = 0.39) and that the task arm beats it (10.48 vs 6.54). Two
readings have been live since week 5: *the world is too shallow to
discriminate*, or *the signal does not do what it says*. Alarm separates them
for free, because it is a world whose discriminability is proved rather than
hoped: **an implementation that reproduces the published deficiency on Alarm has
been shown to behave like the objective it claims to be, and one that does not
has an implementation bug, not a world problem.** Weeks 5–7 established that the
field offers *no* cheap environment-discriminability score (PIC/POIC invert on
known orderings; UED's regret needs a full student training run). **This is the
first known-answer exploration venue this desk has found, and its answer is
pre-computed and machine-checked.**

**Cost on our substrate.** Three actions, four observations, tabular, analytic
optima supplied. **Single-digit CPU-seconds on 4 ARM cores, NumPy, no GPU, no
weights, no disk.** The cheapest front-4 item in ten sweeps.

**WHY IT MIGHT LOSE — steelmanned.**

1. **THERE ARE NO LEARNING EXPERIMENTS AT ALL, and the paper says so as a
   design choice** [V†]: *"We study the objective's optimal policies directly,
   separating their incentives from the difficulty of finding them through
   learning… no sampled training is used."* Our arms are *learned*, with limited
   compute, from a replay buffer. A deficiency at the optimum is not a
   prediction about a trained estimator, and the gap between them is exactly
   where `disagree`'s and `lp`'s real behaviour lives.
2. **The positive result licenses the arm our own ledger demoted.** Theorem 4.1
   gives strict monotonicity for **information gain and Brier** rewards under a
   finite world class, full-support prior and continuous strictly convex `Ψ`.
   Our information-gain member is `A3`/`wm-efe`, which read **t = 2.05**, below
   the 3-sigma learning gate, and week 4 demoted it. **So the theory's
   recommendation is the thing that measured worst here.** That is a tension, not
   a refutation, and I record it rather than quietly dropping the half I like
   less.
3. **Alarm has no body, no needs and no death.** `GOAL.md`'s curiosity is a
   creature climbing a ladder; this is a four-observation process. It can falsify
   an implementation; it can say nothing about whether curiosity drives
   exploration in W0. **Treating a pre-gate pass as a capability claim would be
   precisely the Goodhart this desk refused front 2 over four times.**
4. **The hypotheses may not hold for us.** Theorem 4.1 needs a **finite** world
   class and a full-support prior; W0's world is continuous. Week 5's rule
   applies — *a theory nomination must carry its assumption list* — and this one's
   assumptions are about the world class, not about a trained agent.
5. **`D(π)` is a deficiency in their criterion**, not a performance number. Our
   arms are scored on `life_gain` and coverage. Mapping one to the other is a
   design question and it is **mine, not the paper's**.

---

### N2 — Jacobian centroids as the encoder-vs-predictor separator `LC.03`'s five controls cannot make — and the worst-provenanced nomination in ten weeks

**Source:** *Behavioral Monitoring of JEPA World Models with Jacobian Centroids*
— [arXiv:2609.33940](https://arxiv.org/abs/2609.33940), primary category
**cs.LG**, 2026-09-27, Walker, Balestriero & Baraniuk. **ABSTRACT LEVEL ONLY
[V]** — see §1; the PDF, HTML and ar5iv routes all failed, and **no number,
benchmark name, seed count, parameter count, hardware statement or code URL was
obtainable** [M, three attempts].

**THE VERIFIED CONTENT [V, abstract verbatim].** Centroids are *"sub-component
Jacobian row-sums"*, *"easily computed through Jacobian vector products"*, and
characterise *"how the model organizes the geometry of its input space."*
*"Evaluated on continuous control tasks using JEPA WMs, this behavioral view
reveals a structural dissociation, where the encoder correctly represents the
goal while the predictor remains behaviorally unresponsive. This failure mode
directly predicts planning failure before any action is taken… centroid-based
methods outperform baseline methods as distribution-shift detectors."*

**WHY THIS IS NOMINATED AND SEVEN BETTER-EVIDENCED PAPERS ARE NOT (§5): IT
NAMES OUR OWN OPEN HYPOTHESIS IN ITS OWN WORDS.** Week 8's §6 established, at
HEAD and reproducing the committed artifact exactly, that `A4` is *"A2 with the
decoder deleted; latent prediction vs an EMA target encoder"* (`cores.py:15`),
that the `latent_pred` head is **149,312 params = 17.3% of the arm** [M], and
that **not one of `LC.03`'s five controls — statue, randrew, frozen,
wiped-store, darkroom — sets `l_bind = 0` while the rest keeps learning.** Its
consequence, recomputed at HEAD: `A4`'s seating margins are *"consistent with a
reading in which the latent-prediction objective contributed nothing and an RSSM
actor-critic produced the whole number."*

**"The encoder correctly represents the goal while the predictor remains
behaviorally unresponsive" IS THAT READING, measured in another lab, on
continuous control, with a readout that localises it to the predictor.** Our
controls break the model everywhere at once; this one asks the encoder and the
predictor separately. **And effective rank — the declared-but-unbuilt diagnostic
— cannot do this at all**, because rank is a statistic of the latent *marginal*
and the dissociation is a statement about the *map*.

**Where it lands.** `D37`'s BUILD half, as a readout candidate alongside week
9's N1 (transition geometry, admitted dormant on 10-02) and wk7-N3's `Δ_k`. It
is **not** a sixth challenger to the learning-core seat.

**Cost on our substrate.** Jacobian-vector products on a trained model,
post-hoc, no retraining, no new parameters — the cheapest form any `A4`
diagnostic has taken. **But there are no weights on disk** (week 8 re-verified;
I did not re-verify it this sweep and do not claim to), so like every other
candidate on this seat it is free only if bolted to whatever run `D37`
authorises, and standalone it inherits **14.40 core-h / 3 seeds** [M, week 8].
Since `D37`'s default fires toward **not building**, the realistic cost today is
**zero and the realistic benefit is also zero** — it is a candidate held against
a venue that may not open. I would rather say that than price it as live work.

**WHY IT MIGHT LOSE — steelmanned.**

1. **This is the weakest evidence I have nominated on in ten weeks, and the
   count is the argument against it**: no seeds, no CIs, no benchmark names, no
   parameter counts, no hardware, no wall-clock, no code — because I could not
   open the paper. Every prior sweep complained that nomination-grade papers
   report no hardware; this one reports nothing I could read. **If the builder
   declines it on provenance alone, that is the correct call and I am not
   contesting it in advance.**
2. **It is not independent of a route already on the desk.** Balestriero is
   LeJEPA's author, and `A4c` is the LeJEPA/SIGReg arm (wk1, amended wk8). A
   diagnostic from the same lab as one of the candidate fixes is not a
   third-party check, and this desk has treated convergence-across-labs as the
   thing that made the Context-Collapse family credible.
3. **"Continuous control tasks" is not five senses under homeostatic drive**,
   and their JEPA is a goal-conditioned planner where `A4` is a plastic
   actor-critic reading shared RSSM state. Week 8 already found the *rankability*
   half of ARC-Bench did not transfer for exactly this reason.
4. **A green reading is the likely outcome and is worth little.** Four groups
   have now named action-insensitivity; none has shown it in an RSSM at 861,545
   params. If centroids read healthy, we learn that one failure mode is absent —
   which does **not** restore the missing `l_bind` control, and week 8's finding
   stands either way.

---

### N3 — NOT AN ARM: the known-generator calibration for ABLATION faithfulness, because `GOAL.md` proves every sense load-bearing by ablation and this measures when ablation lies

**Source:** *Better Behavioral Prediction, More Faithful Model Ablations?
Evidence from Sequential Choice* —
[arXiv:2609.36097](https://arxiv.org/abs/2609.36097), primary category
**q-bio.NC**, 2026-09-28, Hanbo Xie (**single author**). **Abstract [V]
verbatim.**

**THE VERIFIED CONTENT [V].** *"Input ablations offer an appealing route: remove
information from a model and interpret the resulting performance change as
evidence of its importance for behavior. Yet this inference assumes that the
model's dependence on information reflects the dependence of the process
generating the behavior. We test it in two synthetic sequential bandit tasks
with known generating policies… under matched donor replacement, accurate
predictors can respond much less than the known generator… at some reward
weights, neural networks predict better than a pooled reinforcement-learning
model but have less faithful changes in choice probabilities; the model ordering
differs between the two tasks."*

**Where it lands, and it is page one of the constitution.** `GOAL.md`: *"a
genuinely unified brain where every sense is load-bearing (and we PROVE each one
is — ablate a sense, something measurable must degrade)."* `UB.11` implements
that; `UB.10`'s matrix is the venue; `T4.06` already decides a verdict on a
readout. **This paper says the inference has a measured failure mode: a model
that predicts better can respond LESS under ablation than the true generator
does** — so an ablation magnitude is a property of the model as much as of the
sense, and ranking two models by ablation response can inherit the wrong order.

**The nomination is one line:** an ablation result that gates a verdict should
be read against a **known-generator calibration** — a fixture where the true
dependence is constructed, so the ablation's *response fidelity* is measured and
not assumed. **We already have the fixture.** `T4.02`/`T4.06`'s `_Fixture` is
generative by construction: each modality carries *"an equal, independent,
standardised share of the target"* through known matrices `W[m]`, with realised
shares recorded (`share_min 0.1979`, `share_max 0.198` [M]). **The true
dependence of the target on each sense is a number we wrote**, so the faithful
ablation response is computable and the measured one can be compared to it.

**Cost: zero new parameters and zero new runs** — one comparison against a
quantity the fixture already defines.

**WHY IT MIGHT LOSE — steelmanned.**

1. **Two synthetic bandits with a fine-tuned LLaMA is not a five-sense RSSM.**
   The transfer is the *discipline*, not the defect, and week 3's rule says an
   analogy is not arithmetic. I did **not** show our ablations are unfaithful; I
   show the question is askable and cheap.
2. **Its own headline is a null-ish, ordering-dependent result** — *"the model
   ordering differs between the two tasks"* — so it does not even supply a
   direction, only a warning that the direction is not guaranteed.
3. **Single author, no seeds stated, no statistics stated, no code statement,
   no hardware** [V, checked at abstract level].
4. **It partly duplicates a nomination of mine that has been unrun for ten
   weeks.** wk1's certificate-gated identifiability protocol (2607.27017) was
   nominated as exactly `UB.11`'s missing positive control and has never run.
   **A second paper arguing for the same missing control is corroboration, and
   corroboration is not news — eighth time of saying it.** What is new is that
   this one measures the *failure* rather than supplying the *fix*, and it does
   it with known generating policies, which is the harder half.

---

### N4 — A CORRECTION TO TWO LIVE NOMINATIONS OF MY OWN: an isotropy penalty's gradient in the transition weights is zero, so SIGReg cannot be `A4`'s anti-collapse answer

**Source:** *One-Step Next-Latent Prediction Is Not a World Model* —
[arXiv:2609.36227](https://arxiv.org/abs/2609.36227), primary category
**stat.ML**, 2026-09-28, Wang, Cai & Hong. **Abstract [V] verbatim, and the
abstract carries the numbers.**

**THE VERIFIED CONTENT [V].** *"An isotropy penalty is a function of the
embedding marginal, so its partial derivative in the transition weights is
zero."* With measurements in the same abstract: *"On a scalar autoregression
with coefficient 0.9, the one-step mean squared error is 0.998 and the 16-step
open-loop error is 5.10."* *"On a hidden rotation, an eight-step window reaches
16-step error 0.056, while the current scalar alone reaches 0.778."* *"Raising
the isotropy weight from 0.1 to 10 leaves eight-step latent error inside
[0.78,0.85] on three seeds."*

**WHY THIS IS A CORRECTION AND NOT A NOMINATION.** Week 5 promoted SIGReg as
**the selection criterion** among `A4`'s anti-collapse routes, on a theorem;
week 8 amended it to its temporally-centered form, on experiments. Both treat
SIGReg as the answer to collapse in `A4`. **If the derivative argument holds,
SIGReg is structurally incapable of being that answer for the half of `A4` that
weeks 6–9 have been chasing**: it shapes the encoder's output marginal, and
`A4`'s predictor is `nn.GRUCell(LATENT+ACTION_DIM, DETER)` whose weights the
penalty does not touch. **A 100× change in the isotropy weight moving the
eight-step error by 0.07 is the measured form of the same statement.**

And it closes a circle this desk opened in week 6: **Context Collapse has a
healthy effective rank**, isotropy regularises the thing rank measures, and
N2 above localises the defect to the predictor. **Three independent routes now
say the same thing — the anti-collapse family is aimed at the encoder, and the
open question is about the map.** Five anti-collapse routes have been promoted
here and zero run; this is the first evidence that running them would not have
answered the question anyway.

**What the correction obliges, narrowly and durably:** wk5-N1 and wk8-N3 stay
live **as encoder-side regularisers** and lose their framing as the *selection
criterion for `A4`'s collapse problem*. If `A4c` is ever built, it must carry a
predictor-side readout or it cannot speak to the failure mode four groups name.
**Zero cost: nothing is run, nothing is withdrawn, one claim is narrowed.**

**WHY IT MIGHT LOSE — steelmanned.**

1. **The regime is time series and LeNEPA, not RL** — scalar autoregression and
   a hidden rotation, with no agent, no actions in the reported experiments, and
   `3` seeds on one of three results. No hardware, no wall-clock, no params, no
   code [V].
2. **`A4` has memory and the paper's sharpest negative is about memorylessness.**
   *"If the observation is a non-injective function of a Markov state, a
   memoryless one-step map does not determine future observations, while a short
   window can"* — `A4`'s transition is a GRU, i.e. it *has* the window. So the
   headline objection is partly answered by our architecture, and I say so
   rather than importing the title.
3. **The derivative argument is about where the penalty is APPLIED**, and that
   is an implementation choice we have not made yet, because `A4b`/`A4c` do not
   exist. An isotropy term applied to *predicted* latents would have a non-zero
   gradient. **So the correction is conditional on a design decision, and the
   honest statement is "do not assume SIGReg answers collapse", not "SIGReg
   cannot."**

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **2609.34677** — *Learning What to Recall: Adaptive Multi-Cue Episodic Memory for World Models* ([abs](https://arxiv.org/abs/2609.34677), 2026-09-28, Kim, Lai, Nguyen, Bar, Ye & Mitsufuji) | cs.LG | **The first front-3 item in ten sweeps that is episodic memory for a WORLD MODEL rather than for a language agent** [V, abstract]. FAR learns a retriever from *"future-aware predictive supervision"* and the retriever *"remains future-blind at inference"*; it *"learns cue-specific relevance and automatically determines which available retrieval cues, such as time, pose, vision, and audio, to trust for each query."* Selection over stored memories, so the READ path is extractive. **Converges with week 9's N2 from the opposite direction**: N2 scores *confidence* from the retrieval score distribution, FAR scores *which cue to trust* — both replace a single fixed similarity cut, which is the exact defect behind `ME.11`'s `τ_fpr 0.388 > τ_cov 0.227` [M]. | **Two things, and the first is constitutional.** (i) Whether the *training* signal — *"negative diffusion prediction loss"* — puts a generative model in the write path, which is the objection that held Eywa, ScrubJay-MEM and RARE. (ii) **Abstention is treated NOT AT ALL** on the page, and §1.8 calls abstention *"the actual hard part"*. Also no benchmarks, seeds, hardware, latency or code on the abs page [V, checked]. **Promote on: a stated read-path with no generative component, plus any latency number for §1.9's CPU table.** Front 3's own sweep is 10-12 and this is its first question. |
| **2608.11506** — *Predictive Allostatic Organization in Recurrent and Spiking Agents Under Partial Observability* ([abs](https://arxiv.org/abs/2608.11506), 2026-08-11, Frederick Hayes III, single author) | cs.NE | **Front 5's first in-window agent with an internal need AND a random baseline, after eight sweeps of saying none exists** [V]. *"Energy-constrained foraging task requiring resource acquisition, threat avoidance, contact-dependent consumption, and regulation of an internal energy variable"*; *"learned agents outperform random and heuristic baselines"*; early internal dynamics predict later success *"above permutation baseline, reaching a maximum ROC-AUC of 0.802"*; feature-family controls show low-energy state *"remains strongly decodable after explicit energy-related features are removed."* **Code released.** | **Nothing promotes it to an arm, and the reason is what it IS**: an interpretability/probing study of internal states, not a learning method and not a world. It proposes no mechanism we could race. **Its transferable part is the control battery** — permutation baseline, feature-family ablation, and the leave-one-family-out decoding that catches a variable surviving its own removal, which is a sharper version of what `NE` §2.4b needs. **Promote on: the environment being a body in continuous control rather than a 2-D arena, or a lethality mechanism.** No seed count, no hardware, no params [V]. |
| **2609.39239** — *Evolutionary foraging in grids: Intermittent search dynamics emerge in finite, depletable landscapes* ([abs](https://arxiv.org/abs/2609.39239), 2026-09-30) | **q-bio.PE** | Named, **not opened**, and recorded only as the empirical companion to week 9's watchlisted 2607.29476 (depletion theory, q-bio.NC). Two papers from two biology categories now say the same thing about depletable patches. | **Nothing — it is a world-design reference and the venue for it is CLOSED** (`w1-world-edit-window` DECLINED, §0(6)). Re-examine when a world-edit window exists. Carried once; dropped with cause next sweep if the window is still refused, because a design input with no venue is not a watchlist item, it is a wish. |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **2605.27929** — transition-geometry readouts (week 9's N1) | q-bio.NC | **ADMITTED 2026-10-02 as a DORMANT CONDITIONAL and still dormant.** Fires only if `D37` resolves toward building a readout; `D37`'s default is **not** to build (§0(3)). **N2 above is a second candidate for the same slot, and the obligation the Review recorded covers both**: *"a build that silently reverts to effective rank alone re-opens this nomination first."* |
| **2609.22056** — RegimeAbstain (week 9's N2) | cs.IR | **ADMITTED 2026-10-02 as a DESIGN UNIT — the one week-9 nomination that converted into work.** Re-verified this sweep against the ledger rather than inherited: `ME.11` `feasible_ok 0.0`, `tau_cov 0.2272`, `tau_fpr 0.3882`, `answered_cues 27.33 ± 4.78` of 160; `ME.11.C` 0.184/0.3649; `ME.11.E`/`.F` VOID [M]. The gap is real and unchanged. **Nothing is owed by me here; the spec is the builder's.** |
| **2609.30210** — alignment-illusion corruption control (week 9's N3) | cs.CV | **ADMITTED 2026-10-02 as a CONTROL DISCIPLINE**, decoupled from §6 exactly as it should have been: *"any future fusion-readout metric that gates a verdict carries a corruption intervention whose effect on the metric is PRINTED."* **N3 above is its sibling and not its duplicate**: corruption asks *does the readout move when the sense is destroyed*; N3 asks *does it move by the right amount*. |
| **2609.21787** — *Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models* | cs.RO | **DROPPED WITH CAUSE, as week 9 pre-committed.** Third sweep carried, relevance rising, **evidence still exactly zero** — no metrics, no seeds, no hardware, no params, no code, and a result stated across *"two of three checkpoints"*. Week 3's deferral rule applied without further argument. It will not appear in a future queue. |
| **3M-Progress** ([2506.00138](https://arxiv.org/abs/2506.00138)) | q-bio.NC | Unchanged on the watchlist after week 9's withdrawal from §2. Still **no agent exploration or survival number of any kind**. **Promote on: any exploration or survival number, from any source.** |
| **SmallWorlds** ([2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | Unchanged, seventh sweep. **Promote on: any statement that a domain runs on CPU.** |
| **ForageWorld** ([2506.06981](https://arxiv.org/abs/2506.06981)) | cs.AI | Kept as a design reference only, and **now blocked by the same closed window as 2609.39239**. |
| **MINERVA** ([2609.03715](https://arxiv.org/abs/2609.03715)) | cs.RO | Unchanged. Still the only paper in ten sweeps whose *inference* runs on our substrate class with a latency number; still imitation learning. **Promote on: an RL result, or nothing.** |
| **PRIME** ([2607.16858](https://arxiv.org/abs/2607.16858)) | cs.LG | Unchanged, and **N1 partly supersedes its role**: PRIME was watchlisted as a model-free epistemic term on 10×10 grids; N1 supplies a smaller venue with a proved answer. |
| **IIBalance**, **Eywa**, **ScrubJay-MEM**, **RARE/RedQA**, **CoDeR**, **Argus Eyes**, **POBAX**, **Curiosity-Critic** | cs.MM / cs.CL / cs.IR / cs.LG | **Unchanged and NOT re-checked this sweep** — front 2 was not fetched and front 3 is off-cadence. Their dispositions stand from weeks 4–9. Stated so the absence is not read as a re-confirmation. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS

| nomination | entered | status now |
|---|---|---|
| **wk9-N1 · transition geometry (2605.27929)** | `D37`'s readout slot | **ADMITTED DORMANT 10-02. Creates no work; `D37`'s default is not to build.** §3. |
| **wk9-N2 · RegimeAbstain (2609.22056)** | `MEMORY_RETRIEVAL_BAKEOFF` §1.8 | **ADMITTED AS A DESIGN UNIT 10-02 — the first field-watch nomination since wk4-N3 to be converted into ordered work.** Ledger numbers re-verified [M]. |
| **wk9-N3 · corruption intervention (2609.30210)** | fusion readouts | **ADMITTED AS A CONTROL DISCIPLINE 10-02, fires on the next fusion readout designed.** |
| **wk9 · §6 (`T4.06`'s uncomputed floor)** | two queue rows | **RE-DERIVED THIS SWEEP AND IT SURVIVES, WITH ONE CORRECTION AGAINST ME: my own closed form undershoots.** §6. Both rows are answered there. |
| **wk8-N1 · replanning-frequency ablation (2609.05461)** | the `a4` row | **LIVE, unrun, correctly sequenced behind `D37` — which is now OVERDUE with a do-not-build default.** Third sweep live. |
| **wk8-N2 · do-nothing reference arm (2609.11247)** | `t402` → `T4.06` | **IN THE SPEC, AND THE SPEC HOLDS A PASS.** `ctrl_incumbent_wins 0.0`, `ctrl_incumbent_still_red 1.0`, winner margin **+0.0187** [M]. §6 is about what that margin was measured against. |
| **wk8-N3 · temporally-centered SIGReg (2607.26924)** | `A4c` amendment | **NARROWED BY N4, not withdrawn.** Stays live as an encoder-side regulariser; loses its framing as the answer to `A4`'s collapse problem. §2 N4. |
| **wk7-N1 · free-embedding critical-`d` probe (2508.21038)** | `ME.11` successor pre-gate | **ACCEPTED, ORDERED FIRST 09-14, STILL NOT RUN AT 21 DAYS** [M: `grep -rln "free.embedding\|critical_d"` returns **four files, all documents**, and **zero `.py`**]. Still the cheapest live item on the desk; now has a sibling (wk9-N2) sitting on the same redesign, also unrun. |
| **wk7 · §6 (`A4`'s diagnostic computed nowhere)** | the `a4` row | **RULED (i)+(iii) on 09-27; (iii) executed; (i) blocked on `D37`, which is OVERDUE.** At HEAD, six weeks on, **effective rank is still computed nowhere** [M, 4 hits, 0 computations]. Row re-dated 10-19. |
| **wk8 · §6 (`LC.03`'s controls never switch off `l_bind`)** | its own row | **DISPOSITIONED, DUE 2026-10-09.** **N2 above is literature arriving at the same hypothesis from outside**, which is the strongest thing that can happen to a finding of this kind. |
| **wk6-N1 · Context Collapse (three groups)** | the `a4` row | **Unchanged and now five groups, plus N2's dissociation and N4's derivative argument.** The theme is thoroughly corroborated and **no amount more of it will arm the guard.** |
| **wk6-N2 · MULTIBENCH++ redundancy pre-gate** | `ub10` | **Unchanged.** `UB.10` is VOID (attempt 1, 09-01); **`T4.03` *Fusion actually fuses* is now REGISTERED but has no ledger row** [M] — one step better than week 9's reading. |
| **wk6-N3 · IAF pathway-decoding control** | `t402` → `T4.06` | **Unchanged and still half-in: `T4.06` measures per-modality latent R² AFTER fusion and no unimodal BEFORE.** §6 raises the value of the missing half: without a before, there is no way to tell a sense that fusion destroyed from one it never carried — and §6 now shows three of five senses sit at or below the no-information floor. |
| **wk5-N1 · SIGReg as variational free energy (2607.13612)** | `LEARNING_CORE` §5.4 | **NARROWED BY N4.** Its Lean claim also stays downgraded (week 6: the repository does not exist) — **and §2's N1 is the contrast case, where the Lean sources do exist and I measured them.** |
| **wk5-N2 · prosociality by coupling (2604.10760)** | `NE.07` | **Unchanged.** Arm DEFERRED, shuffled-partner control accepted, `NE.07`/`NE.02` never run. |
| **wk4-N1 · spectral-radius constraint** · **wk4-N2 · PSG-JEPA ×2** | `A4` variants | **ACCEPTED, narrowed, UNRUN — tenth sweep.** |
| **wk4-N3 · infant motor noise** | `W0.DIAG` | **RUN AND PASSED.** Still the only *arm-side* nomination to become a number. |
| **wk1 · anti-collapse `A4b`/`A4c`** · **wk1 · UB.11 certificate** · **wk1 · interoceptive precision** · **wk1 · entity-collision** · **wk2 · RPE replay** | various | **LIVE, UNRUN — TENTH sweep for all five.** The `UB.11` certificate and the interoceptive-precision arm (code released, 4 ARM cores) remain the two cheapest unrun items this desk has ever nominated; **N3 above is a second, independent argument for the first of them.** |
| wk2 · whiff clock → `SM.02` | `SM.02` | **HELD, correctly — `SM.02` is PARKED.** |
| wk3 · CIG → `A3` · wk3 · Optimistic World Models → `A2` | `A3`, `A2` | **Remain DEMOTED.** N1's positive result is about the information-gain family `A3` belongs to; §2 N1 objection 2 records the tension rather than using it to re-promote. |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

**FRONT 1 · LEARNING CORES — A NOMINATION AND A LARGER REFUSAL, AND THE REFUSAL
IS THE PART WORTH READING.** Three 40-entry enumerations returned the most
crowded field this desk has seen: `2610.03587` (AVL-JEPA, *"Preventing Causal
Dynamics Information Collapse"*), `2610.02860` (counterfactual action evaluation
+ representation geometry), `2610.00727` (CF-JEPA, controllability
factorisation), `2609.37378` (Do-JEPA, masking→intervention), `2609.39235`
(*The Planning Limits of Latent World Models*), `2609.33497` (Hamiltonian JEPA),
`2609.34375` (LRC-JEPA), `2609.32921` (adaptive latent capacity), `2609.23252`
(*Robot World Models Are Not Invariant to How the Actions Are Written*),
`2609.32512` (what latent predictive representations retain). **I nominate none
of them as arms.** This desk has promoted **five** anti-collapse /
latent-structuring routes for `A4` and **run zero**, now across ten sweeps; N4
above is the first evidence that running them would not have answered the
question anyway. **The two front-1 items I did nominate are an INSTRUMENT (N2)
and a CORRECTION (N4), neither of which adds an arm to a seat with five unrun
challengers.** That distinction is the whole of my position on this front.

**FRONT 2 · FUSION — NO METHOD, SIXTH CONSECUTIVE SWEEP, AND THE COVERAGE WAS
SHALLOW AND SAYS SO.** Two 40-entry enumerations, **no fetch**: Alzheimer's
MRI-PET, meme classification, emotion recognition, e-commerce retrieval,
neonatal mortality, dental pulp stimulation, gait analysis, rent prediction,
endoscopic segmentation. **Not one paper with a world-model objective and five
heterogeneous senses — fourth consecutive year.** The unified-tokenisation line
is unchanged: VLA-scale autoregressive vision-language-action, parameter counts
far off this box. **And the question no longer needs the literature:** `T4.06`
measured the balancing family on our own rig and one arm won by +0.0187 with
`grad_norm`'s ratio equalised *by construction*. What this front needs now is
§6's floor and wk6-N3's missing unimodal before — both ours to compute, neither
purchasable from a paper.

**FRONT 4 · CURIOSITY — THE SIX-SWEEP DROUGHT BREAKS, AND EVERYTHING ELSE IN
WINDOW STILL FAILS THE SAME TEST.** Beyond N1, the two 40-entry enumerations
returned an agenda paper (`2609.17325`), a developmental position piece
(`2609.11660`), LLM/code-reasoning bonuses (`2608.07531`, `2606.20881`,
`2606.19476`), federated RL (`2608.10499`), occupancy-coverage pretraining
(`2606.21271`), and the UED/curriculum half was **almost entirely LLM agent
harnesses** — PhantomEnvironments, CogEvol, EnvHarness, EvolveNet, ScaleWoB,
SimWorld Studio, ClawEnvKit. **Not one has a body under homeostatic drive.** Two
items are named and **not opened**: `2609.40134` (*Tactile Curiosity Drives
Robot Interaction*, cs.RO, 09-30 — the only in-window curiosity paper whose
driving sense is TOUCH, which is our measured dominant modality) and
`2610.02012` (*Bellman Meets Lyapunov*, chaos-mastering unsupervised RL, which
`LT.02`'s *"body chaos is reducible"* row makes locally interesting). **No claim
of either is relied on; both are first fetches for 10-12.**

**FRONT 5 · WORLDS & EMBODIMENT — NO ARM, AND FOR THE FIRST TIME THE REASON IS
OURS AND NOT THE FIELD'S.** The enumerations returned swarm robotics
(pheromones, multi-robot allocation, LLM-Foraging), evolutionary connectomics,
one ant-longevity paper, two homeostatic-LLM position pieces, and **two more
medical/physics uses of the word "survival"** — week 2's lesson recurring for a
fourth time. **The MuJoCo ecosystem check (standing mandate item) returned the
same answer as weeks 3 and 9:** the live developments are GPU-parallel-env plays
on the axis `SURVIVAL_WORLD` §2.2 ruled out; the in-window MuJoCo items are
cable transmissions, tensegrity MPC, space robotics and a wingbeat-counting
benchmark. One is recorded as genuinely on our axis and **not fetched**:
`2609.21909` (*Beyond Kinematics: Benchmarking Simulation Fidelity for
Muscle-Driven Imitation Learning*) is the only in-window paper about the
**fidelity ladder itself**, and it is imitation learning. **But the binding
constraint on this front is §0(6): the world-edit window is DECLINED, three
specs are unregistered, seven rows are held behind it. Nominating a world design
into that is spending scout credits on a closed venue, and I am not doing it.**

**FRONT 3 · MEMORY — OFF-CADENCE, SWEPT LIGHTLY, ONE WATCHLIST ENTRY, NO
NOMINATION.** Two enumerations confirmed the standing shape — RippleMem,
MemFuse, Agent Zero Memory, DYNA, HeLa-Mem, AgentMemBench, ZifaMem, CALMem: all
LLM-agent memory, generative in the read path or one step upstream in the write
path, almost entirely cs.CL. The abstention half returned nothing new
generator-free. **The one escape is FAR (§3), and it escapes by not being about
language models at all.** Front 3's real sweep is **2026-10-12** and it has one
question, not a survey: does FAR's training signal put generation in the write
path, and did either of the two cheap, unrun front-3 nominations move?

**SMALL-MODEL END — NOTHING IN WINDOW, SIXTH CONSECUTIVE SWEEP.** The 40-entry
enumeration was VLA end to end: on-policy distillation, action-token routing,
skeleton world-action models, MoE inference caching. Nothing under 1 M
parameters with an RL result. Our own `ppo-needs` at **135,961** params and
`wm-latent` at **861,545** [M, week 7] remain smaller than anything this
literature is proud of.

**BIOLOGY-AS-ORACLE — the standing `q-bio.NC` search produced this sweep's N3
and is now three-for-three.** Also returned and **not** fetched: `2609.13219`
(*Planning as Dynamics Relaxation*, hippocampal recurrent network for optimal
navigation), `2606.14692` (hierarchical Markov models of behaviour — on
irreversibility, predictability and dimensionality, which is `DP.00`'s world
question stated in biology), `2609.32620` (engram allocation under limited
neural resources — space-versus-context competition, bearing on the
diary-vs-weights split), `2608.21814` (hippocampal replay without symmetric
plasticity, bearing on wk2's RPE-replay nomination), and `2606.00667`
(cortex/subcortex under limited cortical memory), which **week 9 queued for a
fetch and I did not do** — carried once more with its reason stated: front 4's
nomination took the fetch budget. **None of these is nominated and none was
opened.**

---

## 6. A FINDING IN OUR OWN ARTIFACTS — week 9's floor claim RE-DERIVED from a complete run: the arithmetic survives, my closed form undershoots, and the conditional collapses to a monotone statement

Two queue rows ordered exactly this (§0(5)), one of them **due today**. The
09-30 row asks: *"is the floor for `min_modality_latent_r2` under a ridge probe
fitting 513 params to 768 rows really `−r/(n_fit − r)`, and if so, do the four
senses reading latent R2 in `[−2.38, −0.17]` sit ABOVE or BELOW it?"* Here is
the answer, from a run that completed.

**EVERY CONSTANT READ FROM CODE, NOT FROM THE SEALED DRAFT** [M]:
`PROBE_N = 1152`, `PROBE_BATCH = 48`, `RIDGE_LAMBDA = 1e-3`
(`t4_06_fusion_balancing_bakeoff.py`), `K_LATENT = 8`, `N_CLASSES = 8`
(`t4_02_no_modality_collapse.py:110-111`), `d_model = 512` (`UnifiedBrain.py:71`)
→ **1152 rows, `n_fit = 768`, test 384, 513 design columns with the bias,
target `k = 8`, R² averaged per dimension.** Same solver as the spec.

**(a) THE PROBE IS A WORKING INSTRUMENT — half the spec's own question,
answered** [M, 5 seeds/point]. The spec's docstring says *"whether that is the
brain or a ridge probe fitting 513 params to 768 rows is not answerable from
this run and is now askable."* At the identical geometry, a planted linear
signal is recovered:

| planted amplitude | measured R² |
|---|---|
| 1.00 | **+0.9395** |
| 0.50 | **+0.8072** |
| 0.25 | **+0.3797** |
| 0.10 | −0.8552 |

So *"the magnitudes are probe artefacts"* is too strong: **the floor is an
artefact of the regime, the instrument is not broken.** That decides whether the
repair is a better probe or a declared floor.

**(b) THE NO-INFORMATION FLOOR, and structure does not move it — RANK does**
[M, 5 and 8 seeds/point]. iid Gaussian X **−1.9867** [−2.0517, −1.8263];
LayerNorm-shaped **−1.9742**; correlated ρ = 0.99 **−2.0134**; Student-t df = 3
**−2.1561**; scale ×10 **−1.9867**. And a one-hot target — which is what
`language` actually is (`z["language"] = np.eye(N_CLASSES)[cls]`) — reads
**−2.0398** at full rank and **−0.0926** at rank 64, so the non-Gaussian target
does **not** get its own floor. That was an objection nobody raised and it is
now closed.

**(c) MY OWN CLOSED FORM IS AN UNDERSHOOT, AND THAT IS THE CORRECTION THE ROW
ASKED FOR.** Week 9 asserted the floor *is* `−r/(n_fit − r)`. Measured against
both candidate forms [M, 5 seeds/point]:

| rank `r` | measured | `−r/(768−r)` | `−(r+1)/(768−r−1)` |
|---|---|---|---|
| 512 | **−2.1453** | −2.0000 | −2.0118 |
| 384 | −1.0523 | −1.0000 | −1.0052 |
| 256 | −0.5216 | −0.5000 | −0.5029 |
| 128 | −0.2100 | −0.2000 | −0.2019 |
| 64 | −0.0938 | −0.0909 | −0.0925 |
| 32 | −0.0438 | −0.0435 | −0.0449 |
| 16 | −0.0244 | −0.0213 | −0.0226 |
| 8 | −0.0177 | −0.0105 | −0.0119 |

**The bias-corrected form `−(r+1)/(n_fit−r−1)` is closer at every rank, and BOTH
systematically undershoot the measured magnitude** (by 0.14 at `r = 512`, by a
factor of 1.5 at `r = 8`). **So the honest verdict on the 09-30 row's question
is: approximately yes in shape, no as a constant.** The formula captures the
monotone two-orders-of-magnitude dependence on rank; it is not a floor you could
type into a spec. **Which strengthens week 9's own conclusion (3) rather than
weakening it: the floor must be MEASURED in-run, and the project already owns the
instrument — `T1.02` is literally *Shuffled-target control*. Permute `Z`'s rows
against the same `X` and re-fit. `grep -cniE "shuffle|permut"` over
`t4_06_…py` and `t4_02_…py` returns 0 and 0** [M].

**(d) THE NEW RESULT — THE TWO-BRANCH CONDITIONAL COLLAPSES TO A MONOTONE
STATEMENT, AND IT IS SHARPER THAN WEEK 9's.** Week 9 said *"at most two of five
senses are linearly recoverable, and which two is not determinable from the
committed row."* Measuring the floor curve at 14 ranks (8 seeds/point) and
crossing it against the incumbent's own per-sense readings [M, ledger]:

| sense | `latent_r2` mean (per-seed) | verdict against its own floor |
|---|---|---|
| **proprio** | **+0.7185** (0.7172, 0.6971, 0.7411) | **above the floor at EVERY rank 8…512** |
| **touch** | −0.2662 (−0.2154, −0.4557, −0.1276) | above only if **effective rank ≳ 192** of 512 |
| **language** | −1.8880 (−2.2408, −1.3251, −2.0981) | needs **essentially full rank (≈512)** |
| **audio** | −2.0641 (−2.1045, −2.1417, −1.9460) | **BELOW the floor at every rank 8…512** |
| **vision** | −2.1881 (−2.3939, −2.0464, −2.1240) | **BELOW the floor at every rank 8…512** |

**The rank branch week 9 left open does not rescue vision or audio in EITHER
direction.** One sense of five is unconditionally readable from the fused
representation; a second is conditional on a rank nobody records; **two are
indistinguishable from carrying no linear information at any rank the
representation could have.** That is `GOAL.md` stage 4 territory — *"senses
fused; each proven load-bearing; no modality collapse"* — and `T4.03` *Fusion
actually fuses* is registered with no ledger row [M].

**(e) THE BAR ITSELF.** `T4.06`'s `min_modality_latent_r2` anchor is **−2.3939**
(vision) and the winner beat it by **+0.0187** [M]. Full-rank floor draws span
−1.83 to −2.16 in the mean and −2.05 at the worst Gaussian seed. **The quantity
two arms were compared on sits at or past the value the probe returns when the
representation carries nothing** — which is not a contradiction of the 109th
audit but the *reason* its margin had to be small against the spread: a
statistic pinned at its own no-information floor has only noise left to vary.

**LEAD OBJECTION, AGAINST MYSELF, AND IT IS UNCHANGED FROM WEEK 9.** **My floors
are measured on synthetic designs, not on the real fused CLS vectors**, whose
rank I did not measure and could not without training the brain. So I cannot say
*which* row of the table applies to `touch` and `language`, and those two rows
stay conditional exactly as written. **That is not a hedge, it is the
nomination**: do not compare `T4.06`'s numbers to my −2.0; make the run report
its own shuffled-label floor and the rank that predicts it. A second caveat
carried forward: the fit/score split is **sequential** (first 2/3, last 1/3), so
non-stationarity in the fixture draws would push the real floor *below* my
estimate; the fixture draws iid per batch so I expect this to be small, and I
did not measure it.

**WHAT I AM NOT CLAIMING.** Not that `T4.06` should not have passed — its
`n_winning_arms ≥ 1` claim and the **ratio** result (29.83 → 2.45 against the
exogenous 10× gate, with `ctrl_incumbent_wins 0.0`) are demonstrated and
untouched. Not that the winner is wrong. Not that anything should be re-run.
**Not that any of this is new**: the margin defect is the 109th audit's,
`LESSONS.md` §16550 owns the general rule, `resolution.py` prints it, and the
spec's own docstring asked the question. **What is new is the measured floor
curve, the corrected closed form, the one-hot check, and the collapse of week
9's conditional to a monotone per-sense statement.**

---

## 7. A FINDING ABOUT THIS DESK — my enumeration script died silently at import, and a liveness check would have passed it

**MY ENUMERATION SCRIPT DIED SILENTLY AT IMPORT AND A LIVENESS CHECK WOULD
HAVE PASSED IT.** The first launch of this sweep's arXiv enumerations produced
**14 lines of output, all traceback**: I had named the script `/tmp/fw10/enum.py`,
which shadows the standard library's `enum` module for everything that imports
`re` or `urllib` — so `python3 enum.py` partially initialised `enum`, broke `re`
at import, and exited before issuing a single query. **Zero of fourteen
enumerations ran, and the process exited 0 from the launcher's point of view.**
Weeks 4–6 each ran enumerations and week 7 reported running none because of an
HTTP 429; **a 429 announces itself and this does not.** It was caught only
because I counted `=====` section headers in the output file rather than
checking that the job had finished. **Generalises week 8's operational note
about the empty-bodied 301: the dangerous failures of this desk's tooling are
the ones that produce a well-formed nothing.** Cheap standing guard, zero
marginal cost: **assert the expected RESULT COUNT, never the exit status or the
fire time.** LESSONS.md candidate, nominated not written.

### 7b — a second finding, same sweep: my own verification grep false-positived on English prose, exactly the shape I routed in week 8

**MY OWN VERIFICATION GREP PRODUCED A FALSE POSITIVE OF
EXACTLY THE SHAPE I ROUTED IN WEEK 8.** Checking N1's Lean supplement I first
measured **"2 admits, 1 axiom"** and nearly published it as the nomination's
honest residue. All three hits are **English prose inside comments** — *"no
converse axiom is assumed"*, *"these tests… admit an enumeration"*, *"cannot
admit a uniform computable-real evaluator"*. Re-measured in tactic position:
**0 `sorry`, 0 `admit`, 0 `axiom` declarations** [M]. **This is week 8's §6b
defect in a different tool**: a pattern matching connective English in a corpus
written by people who use the same words as the formalism. `MIN_QUOTE_OVERLAP`
fixed it for `fieldwatch.py`; nothing protects an ad-hoc grep, and the correct
guard is the one I used by accident — **check the match's syntactic position, not
its presence.** Recorded because the residue I almost published would have
understated a nomination I am arguing *for*, which is the direction of error
that is hardest to catch.

### 7c — a third finding, and it is the one that actually perished: the log hole, repaired in this commit

`docs/FIELD_WATCH_LOG.md` is append-only and exists *"so that a reader can see
what the scout believed on each date without diffing a rewritten file."*
Week 9 wrote 838 lines of page and **zero lines of log** [M]. The `rc=124` row
preserved the page's *content* into `docs/INTEGRATION_QUEUE.md`; it did not
preserve the *drift record*, nobody noticed, and **no instrument reads this log
at all** — `fieldwatch.py` reads `FIELD_WATCH.md`'s finding sections, not the
log. **This commit appends the week-9 lines retroactively**, each prefixed
`[retro, from the sealed 09-28 draft]` so a reader can tell reconstruction from
contemporaneous record, plus this sweep's own lines. **The CLASS — that a
died sweep's log entries have no owner — is the `rc=124` row's (DUE 10-13), and
I note that candidate (ii) there, *"the seal copies an incomplete page into
docs/FIELD_WATCH_LOG.md… before the next sweep can touch it"*, would have caught
this exact instance.**

### 7d — a fourth finding, caused by trying to publish the three above: the reader's own finding-detector is BLIND TO THE PLURAL, and the failure direction is a wrong green

**`experiments/fieldwatch.py` DETECTS `\bfinding\b` AND THEREFORE CANNOT SEE A
HEADING THAT SAYS "FINDINGS".** Measured directly rather than inferred [M]:

```
MATCH   ## 6. A FINDING IN OUR OWN ARTIFACTS — x
BLIND   ## 7. TWO SMALLER FINDINGS ABOUT THIS DESK
BLIND   ### 7b. THE LOG HOLE, REPAIRED IN THIS COMMIT
```

`_FINDING_WORD = re.compile(r"\bfinding\b", re.I)` (`fieldwatch.py:102`) — the
trailing `s` destroys the right-hand word boundary, and the module's own
docstring says a finding is *"any `##`/`###` heading containing the word
'finding'"*, which a plural heading does. **I found this because the first draft
of this page titled §7 in the plural, and `findings()` returned ONE section when
the page carried four** [M]: §7a, §7b and §7c were invisible, and the reader
would have printed **`0 UNROUTED-FIELD-FINDING`** — green — over three un-owned
findings, one of which is the log that had already silently perished.

**THIS IS WEEK 8's §6b's TWIN IN THE OPPOSITE DIRECTION.** That one was false
POSITIVES: un-owned findings reported as `ROUTED`. This one is false NEGATIVES:
findings the reader cannot see at all. **Both end at the same wrong green**, and
week 8's page named the shape in advance — *"a counter reading 0 UNROUTED
because everything collides is the 96th audit's scar with a green light on
top."* The plural is the cheaper half of the same scar.

**WHAT I DID ABOUT IT, AND THE LINE I AM NOT CROSSING.** I retitled §7's
headings into the singular form the reader detects, so these four findings reach
a desk. That is **not** editing prose to make a counter look right — it is the
opposite, and the test is the direction: week 8 refused to reword until a
counter went green and left two false `ROUTED` standing; here the rewording
makes previously-invisible findings **visible**, and the counter it moves is the
denominator, not the verdict. **I did not touch `fieldwatch.py`, which is the
builder's.** The one-character fix is obvious and is not mine to make; naming it
is. LESSONS.md candidate, nominated not written.

---

## 8. What this report does NOT claim

- **No arm here has been run.** Every number in §2 and §3 is someone else's
  measurement on someone else's hardware, except **N1's repository audit, which
  is [M] on this box** — and that is a measurement of the *evidence*, not of the
  *claim*.
- **N1's `0 sorry` is a grep over 321 source files, not a `lake build`.** There
  is no Lean toolchain here. I also did not verify that the 42 registered claims
  exhaust the paper's assertions; the archive's own `PAPER_CLAIMS.json` says
  *"inventory exhaustiveness requires human review."*
- **N1 has NO LEARNING EXPERIMENTS**, and three of its 42 claims are
  `numerical-evidence` rather than Lean — I name which three (the 22-class
  comparison, the budget reference comparison, the native benchmark screen)
  because they are precisely the empirical ones.
- **N2 IS ABSTRACT-LEVEL ONLY** and carries no seeds, benchmarks, params,
  hardware or code statement, because three fetch routes failed. It is nominated
  on its shape and it is the weakest provenance in ten weeks. **Said in §2, not
  in a footnote.**
- **N3 and N4 are single-source and abstract-level**, and N4 is a *narrowing* of
  two live nominations, not a withdrawal of them.
- **§6's floors are synthetic.** I did not measure the real fused
  representation's rank or its shuffled-label floor, so `touch` and `language`
  stay conditional and both branches are given. The unconditional half —
  proprio above, audio and vision below, at every rank — is the part that does
  not depend on the unknown.
- **§6 is mostly NOT new and says so first.** The margin defect, the p/n
  mechanism sentence, the reporter and the four-negatives observation are owned
  by the 109th audit, `LESSONS.md` §16550, `resolution.py` and the spec's own
  docstring.
- **Two of this sweep's pages reached me through a summarising fetch tool**, not
  read line by line. They are marked **[V†]**, which is a demotion I invented
  this sweep and applied to my own work.
- **Front 2 was enumerated and not fetched; front 5 opened one abstract; front 3
  was deliberately shallow and off-cadence.** Seven items across fronts 4, 5 and
  biology are **named without being opened** and **no claim of theirs is relied
  on anywhere in this report.**
- **The `T4.06`, `ME.11`, `a4`, `D37`, queue and reader states are the builder's
  and the Review's records, not mine.** I read them out of the ledger, the
  queue, `decisions --check`, `review_queue --check` and `fieldwatch --check`;
  I did not run them.
- **I claim no causation anywhere.** Week 9's nominations were consumed by the
  Review on its own reasoning; `T4.03`'s registration, the owner-ask repair and
  the `D37` routing all happened without any citation of this page beyond the two
  `t406` rows, which cite it explicitly.

---

## 9. Queued for next sweep (**not before ~2026-10-12**)

1. **Did `D37`'s default fire, and did it fire as (iii)?** It was due before this
   sweep and had not. If it fired, **`A4`'s declared VOID condition is formally
   unarmed by an unattended default** and N2, wk9-N1 and wk7-N3 are all held
   against a venue that is now closed rather than pending — which changes them
   from "sequenced" to "foreclosed", and that is the word to use.
2. **FRONT 3 RETURNS 2026-10-12 with one question, not a survey.** Does FAR
   (2609.34677) put a generative model in the WRITE path? And did either of the
   two cheap front-3 nominations move — wk7-N1 (**21 days unrun, zero `.py`
   hits**) and wk9-N2 (admitted as a design unit 10-02)? **If both are still
   unrun on 10-12 that is the finding, not the literature.**
3. **Did any row carry a shuffled-label floor, or record the fused
   representation's effective rank?** The specific checks are
   `grep -cniE "shuffle|permut"` in the fusion spec (**0 today**) and
   `grep -rniE "effective_rank"` over `*.py` (**4 hits, 0 computations today**).
   **If the rank is ever recorded, §6(d)'s two conditional rows collapse to one
   branch and I should say which.**
4. **Did §7, §7b and §7d reach a desk?** The reader prints them as **3
   `UNROUTED-FIELD-FINDING`** after the retitling [M] — correctly, because no
   row owns them yet. §7d in particular is a one-character fix in
   `experiments/fieldwatch.py` that I may not make. **If §7d is still unrouted
   on 10-12, the reader built to catch un-owned findings will have failed to get
   its own defect owned, which is worth exactly one line and no more.**
5. **The two duplicate `t406` rows.** `t406-latent-floor-was-never-computed`
   was due **today** and `t406-deciding-statistic-read-against-an-uncomputed-floor`
   is due 10-13; §6 answers both. Next sweep reports whether one finding
   consumed two capacity slots, because that is a queue cost this desk caused.
6. **Front 4's two unopened items**, now that the drought has broken and there
   is budget: `2609.40134` (tactile curiosity — our dominant sense) and
   `2610.02012`. First fetches of the sweep.
7. **`2606.00667`** (cortex/subcortex under limited cortical memory) — queued by
   week 9, not done by me, **carried ONCE more with the reason stated.** Per
   this desk's own deferral rule it is **dropped with cause** on 10-19 if a third
   sweep fails to open it.
8. **`q-bio.NC` standing search — upheld, three-for-three, and now the
   highest-yield slot on the desk.** It produced N3, week 9's N1 and week 9's
   only on-axis front-5 item.
9. **Does the world-edit window re-open?** `w1-world-edit-window` is DECLINED,
   `W1.01`/`W1.03`/`W1.04` are unregistered, and seven rows wait behind it.
   **Front 5 has no nominatable venue until that changes**, and a fourth sweep of
   world-design watchlisting without one would be padding.
10. **NOT queued, deliberately:** conference proceedings (dropped wk4); 2607.22430
   and UED-as-a-cheap-score (closed wk6); LIMIT's *conclusion* (closed wk7,
   instrument retained); 2608.29434 (dropped wk9); 3M-Progress's numbers (closed
   wk9); ARC-Bench's code (promote-if-it-appears); **2609.21787 (dropped with
   cause THIS sweep)**; **and the anti-collapse family — ten more routes appeared
   in this sweep's enumerations and §5 declines all of them until one of the five
   already accepted has been run, with N4 now arguing that running them would not
   answer the question they were promoted for.**
