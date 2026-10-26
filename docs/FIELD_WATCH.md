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

**Sweep date:** 2026-10-26 · **Window:** ~2026-04 → 2026-10-26
**Scout:** field watch, week 11 — **and that is the eleventh REPORT in the
thirteenth Monday slot.** The 10-12 and 10-19 sweeps were **refused by the
usage gate, not missed**: `/data/jack-logs/field_watch.log` reads
`2026-10-12T05:37:04 STOPPED at 98% weekly usage` and
`2026-10-19T05:37:03 STOPPED at 92% weekly usage` [M]. **Twenty-one days since
the last complete report** (week 10, 2026-10-05), and §7 is about the fact that
no committed file in this repository records either refusal.

**Confidence markers:** **[V]** fetched and read · **[V†]** fetched, but the
page reached me through a summarising model rather than read line by line ·
**[c]** claimed by the authors, not checked against their table · **[s]**
asserted by a search engine *about* a paper I have not opened · **[C]** computed
by me, arithmetic shown · **[M]** measured here, on this box, command shown.

**Operational note, and it upgrades this sweep's evidence class over week
10's.** Week 10 could not open a single PDF and had to invent `[V†]` for pages
read through the fetch tool's summariser. This sweep took a different route:
the arXiv **API** returns `<summary>` verbatim and unparaphrased, so a 20-id
`id_list` query yields twenty full abstracts as text. **`FETCHED=20/20` [M].**
Every nomination below is therefore **[V] at abstract level** — and `[V†]` is
used nowhere on this page. The ceiling is unchanged and stated in §8: an
abstract is not a table, and there is still no `pdftotext`, `pypdf` or `fitz`
on this box [M].

---

## 0. WHAT MOVED IN THE INTERVAL — a finding of mine became a spec, ran, and came back AGAINST me

**459 commits** landed in the twenty-one days since week 10 [M,
`git log --since=2026-10-05 --oneline | wc -l`]. Seven facts re-point this
sweep, and the first is the most important thing this desk has ever had to
report.

**(1) `LC.08` PASSED YESTERDAY AND IT REFUTES WEEK 8's §6 READING — MINE.**
Week 8's §6 found that `LC.03`'s five controls never switch off the term `A4`
is named for, and drew the consequence that `A4`'s seating margins were
*"consistent with a reading in which the latent-prediction objective
contributed nothing and an RSSM actor-critic produced the whole number."* That
finding was routed, ruled 2026-09-29, and **built as `LC.08` — the binding
objective, paired on/off.** Ledger row **PASS, attempt 1,
2026-10-25T18:00:57** [M]:

| statistic | seat (term ON, replayed) | comparator (term OFF) | deficit | seat seed σ |
|---|---|---|---|---|
| `life_gain` | **96.83 ± 39.21** | **−19.31 ± 153.18** | — | — |
| `lg_margin_null` | **100.10 ± 37.33** | **−16.04 ± 155.08** | **116.14** | 37.33 |
| `lg_margin_twin` | **97.66 ± 42.25** | **−18.48 ± 148.25** | **116.14** | 42.25 |

`deadweight_reading = 0.0`; `params 861,545` with `params_match 1.0`;
3 seeds; 431,852 decisions and 17,280 core-s per seed; implementation borrowed
hash-pinned from `LC.02` (`47e4db0`). My own arithmetic on the row [C]:
`100.101537 − (−16.04235) = 116.143887`, exactly the recorded `deficit_null`,
and it clears the seat's own seed noise by **3.11×** and **2.75×** against a
pre-registered bar of 1.0×.

**The spec's own pre-registered FORECAST was "lean FAIL with the dead-weight
reading."** It passed instead. **So the binding objective is load-bearing: it
carries essentially the entire seating margin, and 17.3% of the arm is not dead
weight.** Three consequences, and I take them in the order that costs me most:

- **§2's withdrawal is forced.** `wk10-N2` (Jacobian centroids) was nominated
  because *"the encoder correctly represents the goal while the predictor
  remains behaviorally unresponsive"* **is** week 8's reading, measured in
  another lab. That reading is now measured FALSE in our arm. The nomination
  loses its premise and is **WITHDRAWN** in §2, not quietly re-framed.
- **It makes §2's N1 the strongest nomination on the desk.** A term the ledger
  has just proved load-bearing makes the *capacity of the machinery computing
  it* a live design choice rather than an argument about a vestige.
- **The system worked and I did not.** A field-watch finding reached a desk, was
  ruled, was built as a falsifiable spec with a pre-registered dead-weight
  branch, and the measurement went against the scout. That is `SYSTEM.md` law 3
  doing exactly what it is for, and it is the first time this desk's own
  hypothesis has been killed by a run rather than by a better paper.

**(2) `D37` FIRED (iii) ON 2026-10-05 AND THE WORD IS NOW "FORECLOSED", AS WEEK
10 PRE-COMMITTED.** `docs/DECISIONS_RESOLVED.md:2137` [M]: *"RESOLVED BY ARMED
DEFAULT (fired 2026-10-05 ~06:5x UTC by the OVERSEER, 140th audit, on the first
legal day) … (iii) HOLD `D29` AS IT STANDS … the `Δ_k` collapse readout is NOT
built."* Week 10's queued #1 asked exactly this and said what to call it:
**`A4`'s declared VOID condition is formally unarmed by an unattended default,
and the readout slot is closed rather than pending.** `wk9-N1` (transition
geometry), `wk7-N3` (`Δ_k`) and the now-withdrawn `wk10-N2` were all held
against that slot. At HEAD, `grep -rniE "effective_rank|rankme" --include='*.py'`
returns **ZERO** hits where week 10 measured four [M] — the prose hits are gone
too — while `LEARNING_CORE.md` §5.4 still declares the diagnostic. **The gap
between the governing document and the implementation is now total, and settled
by hold.** `LC.08`'s own docstring names the consequence: the comparator is
*"THE ONLY LIVE MEASUREMENT the seat will ever get."*

**(3) FRONT 5's VENUE OPENED, AND WEEK 10 WAS WRONG TO SAY IT COULD NOT.**
Week 10 declined to nominate on front 5 on the ground that *"Front 5 has no
nominatable venue"* because `w1-world-edit-window` is DECLINED and
`W1.01`/`W1.03`/`W1.04` were unregistered. Measured at HEAD: **`W1.01` and
`W1.04` are REGISTERED** (`W1.03` is not), and three rows exist — **`W1.01`
PASS** (attempt 1, 2026-10-06), **`W1.02` PASS** (attempt 6, 2026-10-25),
**`W1.04` VOID** (attempt 1, 2026-10-06) [M]. The window stayed DECLINED and the
specs ran anyway, under the `w0-too-shallow` row's 10-02 split — *"this spec
RUNS on W0 as built; the world edit is needed to PASS, not to RUN."* **My
inference from a DECLINED row to a closed venue was wrong, and the correction is
load-bearing: §2's N3 is a front-5/needs nomination that week 10's reasoning
would have suppressed.**

**(4) AND `W1.04`'s VOID REASON IS THIS SWEEP'S BEST CONVERGENCE.**
`void_reason = "control: survival gate cannot tell a fatal world from a
survivable one"`, `p5_ok 0.0`, with 5th-percentile life spans
**`p5_w101 54.40` against `p5_w102 51.88`** across a declared `horizon_s 5600`
[M]. A survival statistic that cannot separate two worlds is a statistic with
no resolution. **N3's source paper closes on the sentence *"Survival alone
therefore gives an incomplete account of how a model prioritises needs and
learns about its body"* [V]** — the same claim, arrived at from the mechanism
side in another lab.

**(5) MY OWN §7d IS FIXED AND ALL FIVE WEEK-10 FINDINGS REACHED A DESK.**
`experiments/fieldwatch.py:102` now reads
`_FINDING_WORD = re.compile(r"\bfindings?\b", re.I)` [M] — the one-character
fix I named and did not make. `fieldwatch --check` **rc=0**: *week 10, 5 finding
sections; 0 cited, 5 quoted, **0 UNROUTED-FIELD-FINDING***, with §7/§7b/§7d
owned by `seven-instrument-readers-are-gated-by-no-spec` (OPEN, DUE 10-29), §7c
by `field-watch-rc124-page-is-untrustable-and-then-deleted` (OPEN, DUE 11-04),
and §6 by `t406-latent-floor-was-never-computed`. **Queued #4 closes
positive.**

**(6) THE QUEUE MOVED A LOT AND ITS DRAIN GOT WORSE — BOTH HALVES, NOT THE
SCARY ONE ALONE.** [M, `run review-queue`]: **63 OPEN, 0 HELD, 18
DISPOSITIONED, 76 ACTED, 15 DECLINED of 172 routed; 81 live rows; oldest live
57 d; arrived 29 (4.14/cycle) against disposed 20 (2.86/cycle); drain
UNBOUNDED.** Against week 10: ACTED **40 → 76**, DECLINED **2 → 15**, HELD
**3 → 0**, live **85 → 81**. **The desk disposed 49 rows in 21 days** — the
largest clearance in this file's history — and the drain still reads UNBOUNDED
because arrivals now outrun disposals. `review_queue --check` exits **2** on
**`10 VIOLATION(S) — OVERDUE 10`**, all ten promised 2026-10-25. Week 10's
`7 HOLD-ON-A-RESOLVED-BLOCKER` violations are gone.

**(7) `decisions --check` IS GREEN FOR THE FIRST TIME ON THIS PAGE.** rc=0,
**`ratchet ok (0/0 undeclared, 0/0 unrouted-owner-ask, 0/0 vanished-owner-ask,
0/0 default-action-expired, 0/0 firing-diff)`**, and `35 of 35` firings
transcribed [M]. Week 10 printed `RATCHET BROKEN: 1 DEFAULT-ACTION-EXPIRED`
(`D33`); it is repaired. Printed because it is in the same instrument and is not
mine: four CONDUCT-DESK entries are stale — `D39` by 11 days, `D40` by 16,
`D46` by 9, `D47` by 2.

**Three instrument exit codes, re-run after the last edit of this sitting**
(`review_queue.py` does not read this page — `grep -c FIELD_WATCH` is 0 [M]):
`fieldwatch --check` **rc=0**, `decisions --check` **rc=0**,
`review_queue --check` **rc=2** (the 10 OVERDUE above). The reader's own reading
of THIS page is **week 11: 4 finding sections; 0 cited, 3 quoted, 1
UNROUTED-FIELD-FINDING** — and **§7b is the finding that three of those four
routings are not to be trusted, because the channel that produced them is
date-blind.** Read §7b before reading the counter.

---

## 1. Coverage — what was actually searched, so the gaps are visible

The arXiv API answered on the first attempt. **16 queries × 40 entries =
`QUERIES_WITH_RESULTS=16/16`, 640 result rows** [M] — and the count was
asserted, not the exit status, which is week 10's §7 lesson applied to its own
author. Then **20 ids fetched for verbatim abstracts, `FETCHED=20/20`** [M].

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **1 · LEARNING CORES** | world model × latent/model-based; JEPA / joint-embedding-predictive; sample-efficiency × continuous control; active inference × agent/scaling | **4 enumerations (40×4)**; **6 abstracts [V]** (2610.32562, 2610.22655, 2610.05550, 2610.19375, 2610.27387, 2610.14302) |
| **2 · MULTIMODAL FUSION** | multimodal × fusion × binding/ablation/load-bearing; unified tokenisation / shared latent / modality collapse | **2 enumerations (40×2)**; **no dedicated fetch** — nothing survived the title line. **N1 and N2 are the fusion-relevant fetches and they came in through front 1** |
| **3 · MEMORY** *(TWO CYCLES OVERDUE — the endorsed two-week clock due 10-12 and 10-19 was refused twice)* | episodic memory × agent/retrieval; abstention / selective prediction × retrieval; consolidation × replay × memory | **3 enumerations (40×3)**; **5 abstracts [V]** (2609.34677, 2610.31446, 2610.30996, 2610.24030, 2610.10265) |
| **4 · CURIOSITY & OPEN-ENDEDNESS** | intrinsic motivation / intrinsic reward / autotelic; open-ended × curriculum / environment generation / UED | **2 enumerations (40×2)**; **3 abstracts [V]** (2609.40134v3, 2610.02012, 2609.39342) |
| **5 · WORLDS & EMBODIMENT** | homeostatic / foraging / survival agent / embodied survival; **MuJoCo × benchmark/environment/simulator** (standing mandate item) | **2 enumerations (40×2)**; **4 abstracts [V]** (2610.18617, 2610.26395, 2609.13771, 2610.05997) |
| Biology-as-oracle | **`q-bio.NC` standing search** (wk7 §7) × RL / world model / embodied / curiosity / exploration, **and** × development / infant / play / critical period | **2 enumerations (40×2)**; **2 abstracts [V]** (2610.25377, **2606.00667 — queued #7, opened on its last legal sweep**) |
| Small-model end | tiny / compact / lightweight × policy / world model / RL | **40-entry enumeration**; no fetch |
| Queued closures | `D37`'s firing; the `effective_rank` and shuffle/permut greps; wk7-N1's grep; `W1.0x` registry; FAR's write path; the gate refusals | **[M] — commands shown in place** |
| Our own artifacts | **`LC.08`'s row (46 metrics) and its full docstring**; `T4.06`'s four arms × 3 seeds (29 metrics); `W1.01`/`W1.02`/`W1.04` rows; `ME.11`/`ME.11.C` rows; **a from-code parameter measurement of `A4` and of the widened predictor**; `cores.py:376-414`; the three instruments | **[M]/[C] — §0, §2, §6** |

**Known gaps, stated so nobody assumes coverage:**

- **FRONT 3 IS TWO CYCLES OVERDUE, NOT OFF-CADENCE BY CHOICE.** Its endorsed
  two-week clock was due 10-12 and 10-19 and both sweeps were refused by the
  gate. It was swept properly today (3 enumerations, 5 abstracts) and still
  **produces no nomination** — §5 and §6 say why, and §6 says the binding
  constraint is in our own ledger rather than in the literature.
- **Front 2 was enumerated and not fetched, SEVENTH consecutive sweep.** Nothing
  survived the abstract line. §5 says so rather than manufacturing a fetch.
- **No PDF or table was read anywhere.** Every paper number below is from a
  verbatim abstract. Week 10's objection to itself applies in full: *"this is
  not the same epistemic object as reading a paper."* Where an abstract carries
  no seeds, no CIs, no hardware or no params, **I say which of those is missing
  per nomination**, in §2, not in a footnote.
- **No conference main-track enumeration** (dropped permanently, week 4; still an
  acknowledged permanent gap). **No non-English sources.**
- **I did not run the released simulator behind N1, nor the released code behind
  N3 or N4.** Downloading a 321-file Lean tree (week 10's N1) was within a
  scout's remit; training an encoder is not. Each nomination says what was and
  was not executed.
- **Nine items across fronts 1, 3, 4, 5 and biology are NAMED AND NOT OPENED**
  in §5, and **no claim of theirs is relied on anywhere in this report.**

---

## 2. NOMINATIONS

**Five, plus one withdrawal of my own.** One is an ARM in our exact parameter
regime whose intervention is a single integer — the best-priced front-1
nomination this desk has made; one is a CONTROL aimed at the only POSITIVE
reading in my own week-10 finding, because that is the reading nobody has
attacked; one is an ARM PAIR plus a required secondary on `NE`, arriving the
week `W1.04` VOIDed on the same claim; one is an ARM on `CURIOSITY_BAKEOFF`
whose mechanism is a dict our code already computes every gradient step and
throws away; one is a CONTROL DISCIPLINE that attaches to every auxiliary-
objective arm this desk has ever accepted. Each states its arXiv primary
category, its evidence class, its cost on **our** substrate, and both sides
steelmanned.

---

### N1 — PREDICTOR WIDTH as an `A4` arm: the half-width proportion is measured to suppress factor readability at 0.7M params, and ours is exactly half-width — a one-integer change that `LC.08` has just made worth making

**Source:** *Shared Is Not Enough: Predictor Width and Factor Readability Under
Masked Latent Prediction* —
[arXiv:2610.32562](https://arxiv.org/abs/2610.32562), primary category
**cs.LG**, 2026-10-23, **Nilotpal Kakati (single author)**. **Abstract [V]
verbatim.**

**THE VERIFIED CONTENT [V].** *"We built a simulator in which every factor
behind an image is known, so a probe can measure how readable each factor
remains after training. We trained small encoders (0.7M parameters) with
I-JEPA, SimCLR and BYOL on the same pairs of views and probed a factor that
both views share. I-JEPA had a predictor with half the encoder's width and
depth, the proportions it uses for ViT-B. Its median score was about 40% of
SimCLR's. Widening the predictor to the encoder's width reversed this ordering
at every training seed, against both other methods. We release the simulator
and the trained models."*

**WHY THIS ONE AND NOT THE TWENTY OTHER FRONT-1 PAPERS (§5).** Three properties
no other front-1 item in eleven sweeps has had together. **(i) It is in OUR
PARAMETER REGIME.** 0.7M-parameter encoders, against `A4`'s **861,545** [M].
Every prior transfer failure on this page came from hardware or scale unlike
ours; `LESSONS.md` is explicit that published numbers have failed to transfer
to this box three separate times. This is the first front-1 result whose
subject is smaller than our arm. **(ii) The intervention is an INTEGER, not an
objective.** This desk has promoted **five** anti-collapse / latent-structuring
routes for `A4` and run **zero**, and week 10's N4 argued that running them
would not have answered the question anyway. A width is not a sixth loss term.
**(iii) `LC.08` just made it matter.** A width change to machinery proved dead
weight would be worthless; §0(1) measured that machinery carrying **116.14** of
margin against **37.33** of seed noise.

**OUR GEOMETRY, MEASURED FROM CODE AT HEAD [M]** (`experiments/cores.py`,
`build_arm("wm-latent")`):

| quantity | measured |
|---|---|
| `A4` total params | **861,545** |
| `latent_pred` | **149,312 (17.3%)** — `_mlp(state_dim=512, 256, LATENT=64)` |
| `post` | 149,120 — `_mlp(320, **256**, 256)` |
| `prior` | 132,608 — `_mlp(256, **256**, 256)` |
| `state_dim` = `DETER` + `STOCH_N·STOCH_C` | 256 + 256 = **512** |
| **predictor hidden ÷ state width** | **256 / 512 = exactly 0.5** |

**So every head on `A4`'s prediction path is half as wide as the state it reads,
which is the paper's named proportion, to the digit.** The widening it measures
costs, on our arm [M]:

| variant | `latent_pred` | arm total | vs `A2` `dreamer-xs` (1,671,065) |
|---|---|---|---|
| as seated | 149,312 | **861,545** | 52% of A2 |
| predictor widened 256→512 | **297,536** (+148,224) | **1,009,769 (+17.2%)** | **60% of A2** |
| `post`/`prior` widened too | — | **1,289,833 (+49.7%)** | **77% of A2** |

**THE FALSIFIABLE REASON IT MIGHT WIN, and it is OUR number, not theirs.**
`T4.06`'s committed row carries per-sense latent R² for **four arms × 3 seeds =
12 runs** [M], and in **all twelve**, exactly one sense is positive (`proprio`
0.5316–0.7411) while `vision` (−1.94 to −2.63), `audio` (−1.95 to −2.68) and
`language` (−0.93 to −2.65) sit at or below the no-information floor week 10
measured at ≈ −2.0. **That is "shared is not enough" as a committed
measurement on our own rig, replicated twelve times.** This paper supplies the
only mechanism anyone has offered for it that is (a) in our size class, (b) a
single hyperparameter, and (c) **predictor-side**, which is where week 10's N4
argued the defect must live because an isotropy penalty's gradient in the
transition weights is zero. And the win condition is pre-stateable: *widening
the predictor should move the per-sense readability of the senses that are
currently at the floor, at fixed objective and fixed data.*

**Cost on our substrate.** The code change is `_mlp(self.state_dim, 256,
LATENT)` → `_mlp(self.state_dim, self.state_dim, LATENT)`. **Zero new
hyperparameters, no new module, no new objective.** The run is the expensive
half and it is knowable: `LC.08` paid **17,280 core-s × 3 seeds = 14.4 core-h**
[M] for one paired comparator in the same venue, and a widened arm is the same
shape of purchase. **+17.2% params against an arm that is still 40% smaller
than the arm it already beats** is the cheapest honest framing of the risk.

**WHY IT MIGHT LOSE — steelmanned.**

1. **THE MAPPING IS AMBIGUOUS AND THE TWO READINGS POINT OPPOSITE WAYS. THIS IS
   THE LEAD OBJECTION AND IT IS MINE, NOT THE PAPER'S.** In a ViT the encoder's
   width *is* the representation width the predictor reads, so our 256-vs-512 is
   the paper's configuration. But our encoder's *output* is `LATENT = 64`, and
   against that the predictor's 256 hidden is **4× over-wide**, not half-width.
   **Under the first mapping the nomination is live; under the second it is
   already satisfied and there is nothing to buy.** I nominate on the first and
   say plainly that I cannot settle it from an abstract — the paper's own
   simulator and released models could settle it, and that is a pre-gate, not an
   arm.
2. **Depth is half of the paper's axis and ours is already matched.** It says
   *"half the encoder's width AND depth"*; `_mlp` is two linear layers
   everywhere in `cores.py`, so only the width half can transfer. A result
   driven by depth would not reach us.
3. **Single author, and the abstract carries no CIs, no seeds count beyond
   "every training seed", no hardware, no wall-clock, and one effect size
   stated as "about 40%".** By week 5's rule this is weak provenance for a claim
   this load-bearing — though it is the ONLY front-1 paper this sweep that
   states a parameter count at all.
4. **Vision pairs are not five heterogeneous senses, and "a factor both views
   share" is not a modality.** Their probe reads a known generative factor off
   image views; ours reads a sense off a fused latent under homeostatic drive.
   Week 8's ARC-Bench transfer failed on exactly this gap.
5. **It measures READABILITY, not control.** A wider predictor could raise
   per-sense R² and leave `life_gain` flat — and `life_gain` is what seats the
   arm. `GOAL.md` wants both (*"every sense load-bearing"* and a creature that
   lives), so a readability-only win is a real result and not a seat.

---

### N2 — NOT AN ARM: three asymmetric probe diagnostics as the control on the ONE POSITIVE reading week 10 left unexamined, because a high probe score is the thing nobody here has doubted

**Source:** *Is a Linear Probe Evidence of a Linear Representation?* —
[arXiv:2610.22655](https://arxiv.org/abs/2610.22655), primary category
**stat.ML**, 2026-10-15, Kutluozen, Internò & Klindt. **Accepted at NeurIPS
2026** (author comment [V]). **Abstract [V] verbatim.**

**THE VERIFIED CONTENT [V].** *"The linear representation hypothesis holds that
pretrained vision models encode generative factors as linearly decodable
directions in feature space, and comes with a standard methodology: train a
linear probe, declare a factor represented if accuracy is high. We show this
test is too permissive. Asymptotically, any sufficiently rich feature map,
including a random one, admits a linear readout, so high probe accuracy is
consistent with two very different geometries: a true linear world model …
or a kernel machine … We propose three asymmetric diagnostics that distinguish
the two cases: reverse predictivity, spectral concentration, and
out-of-distribution generalization. We validate them in a fully observed
simulation where ground truth is known … On dSprites, 3D Shapes and MPI3D,
forward decodability and latent-aligned variance dissociate. Thus, forward
decoding alone does not show that the probed factors align with the dominant
geometry of the representation."*

**WHY IT LANDS WHERE IT DOES, AND IT IS AIMED AT ME.** Week 10's §6 spent a
whole section on the NEGATIVE tail — the no-information floor that `vision` and
`audio` sit below. **It never once doubted the positive.** `proprio` reads
**+0.7185** mean, positive in all 12 `T4.06` runs [M], and every reading of
that number on this page and in the spec treats it as *the one sense that is
demonstrably in the latent.* **This paper says a high forward-decoding score is
exactly the quantity that does not license that conclusion**, and gives three
diagnostics that do — one of which, **spectral concentration**, is a
rank-of-the-aligned-subspace statistic, i.e. the family of the diagnostic
`LEARNING_CORE.md` §5.4 declares and `D37` has now settled by hold.

**And we already own the venue the paper needs.** Its validation is *"a fully
observed simulation where ground truth is known"*; `T4.02`/`T4.06`'s `_Fixture`
is generative by construction, each modality carrying *"an equal, independent,
standardised share of the target"* through known matrices `W[m]`, with realised
shares recorded (`share_min 0.1979`, `share_max 0.1980` [M]). **Reverse
predictivity — predict the feature from the latent instead of the latent from
the feature — is computable on that fixture with no model and no run.**

**Where it lands.** A control on any verdict that reads a per-modality probe:
`T4.06`'s `min_modality_latent_r2` (which decided a bakeoff by **+0.0187**
[M]), `T4.03` *Fusion actually fuses* (REGISTERED, still **no ledger row** [M]),
and `UB.11`'s load-bearing proof. **It is the sibling of `wk9-N3`'s corruption
intervention and of `wk10-N3`'s known-generator calibration, and it is not their
duplicate:** corruption asks *does the readout move when the sense is
destroyed*; wk10-N3 asks *does it move by the right amount*; **this one asks
whether a readout that moves correctly is measuring the representation's
dominant geometry at all.**

**Cost on our substrate.** Reverse predictivity and spectral concentration on
the existing fixture: **zero new parameters, zero GPU, single-digit CPU-minutes
on 4 ARM cores.** The OOD third needs a shifted fixture draw, which the fixture
can generate. **No run is re-done and no verdict is moved by computing it.**

**WHY IT MIGHT LOSE — steelmanned.**

1. **ITS DIRECTION IS THE OPPOSITE OF OUR PROBLEM, AND I SAY SO FIRST.** The
   paper attacks false POSITIVES — permissive high scores. Four of our five
   senses read at or below a no-information floor, which is a false-negative
   regime its diagnostics were not built for. **It can only attack `proprio`.**
   The honest claim is therefore narrow: it audits one sense of five, and that
   sense is the only good news `T4.06` carries.
2. **Pretrained vision transformers are not an 861,545-param RSSM**, and its
   benchmarks (SVG-World, CelebA, dSprites, 3D Shapes, MPI3D) are image-factor
   datasets with no body, no actions and no needs.
3. **No seed count, no hardware, no wall-clock, and no code statement in the
   abstract** [V, checked]. The NeurIPS acceptance raises the prior on the
   claims and lowers nothing about what I could read.
4. **It could tell us nothing actionable.** If `proprio` passes all three
   diagnostics, we learn the one good reading is sound — which changes no
   verdict, buys no arm, and leaves §6(d)'s four negatives exactly where they
   are. That is a real possibility and it is why this is a control and not an
   arm.

---

### N3 — TWO ROUTES TO NEED-PRIORITISATION as an `NE` §2.4b arm PAIR plus a required secondary readout, arriving the same month `W1.04` VOIDed on this paper's own conclusion

**Source:** *Two Routes to Interoceptive Prioritisation: Dissociating Likelihood
Precision and Preference Reweighting in Active Inference* —
[arXiv:2610.18617](https://arxiv.org/abs/2610.18617), primary category
**cs.AI**, 2026-10-13, Grimbly, Kuske, Boonstra, van Hoof, Hodson, Rosman,
Smith & Solms (9 authors). **Accepted at IWAI 2026, Springer proceedings;
CODE RELEASED** (`sgrimbly/interoceptive-prioritisation-iwai2026-snapshot`,
author comment [V]). **Abstract [V] verbatim.**

**THE VERIFIED CONTENT [V].** *"An embodied agent must prioritise among
competing bodily needs. … it can respond to an urgent need by increasing the
precision of information about that need or by increasing its preference for
outcomes that relieve it. We compare these two interventions, separately and
together, in a foraging gridworld with three lethal needs … Precision sharpens
interoceptive posteriors and improves survival. Preference reweighting keeps
bodily states further from their lethal boundaries and changes the states from
which the agent learns, so its learned model covers fewer need levels, with
lower quality across all levels but slightly higher quality on the levels it
visits. An experiment on 72 further layouts does not establish the predicted
preference benefit on difficult layouts, but both interventions improve average
survival there and their interaction is positive. Survival alone therefore
gives an incomplete account of how a model prioritises needs and learns about
its body."*

**Where it lands, and it is `GOAL.md`'s own sentence.** *"Jack has the needs of
a human … the needs ARE the curriculum."* `NEEDS_AND_DEATH` §2.4b holds
**`wk1`'s interoceptive-precision arm** (2608.04232, code released, 4 ARM
cores) — **tenth sweep live, unrun.** That arm is route ONE of the two this
paper dissociates. **Route TWO — preference reweighting — is not in our design
at all**, and the paper's measurement is that the two do different things to the
same survival number.

**THE PART THAT IS NEW AND IS NOT ANOTHER SURVIVAL ARM.** *"its learned model
covers fewer need levels, with lower quality across all levels but slightly
higher quality on the levels it visits."* **A mechanism that improves survival
by DEGRADING the body model is a measured trade this project has no instrument
for.** `GOAL.md`'s capability target marks suspect any choice that *"would make
some class of learning permanently impossible"* — and an agent that survives by
never visiting its own low-need states has foreclosed learning about them. So
the nomination is **two arms and one required secondary**: race precision
against preference-reweighting against both, and **report need-level coverage of
the learned body model alongside survival**, because the paper measures that the
two can move in opposite directions.

**AND IT CONVERGES WITH OUR OWN VOID.** §0(4): `W1.04` is VOID on *"the survival
gate cannot tell a fatal world from a survivable one"*, `p5_w101 54.40` against
`p5_w102 51.88` [M]. **Two independent routes now say the same thing — a
survival statistic alone is an incomplete account — one from a paper's
mechanism study and one from our own world-fidelity control.**

**Cost on our substrate.** A foraging gridworld with three lethal needs, active
inference with discrete beliefs, code released. **CPU-minutes to CPU-hours on 4
ARM cores, no GPU, no weights.** Porting the *preference-reweighting* route into
`NE`'s own rig is the real cost and it is a design unit, not a run.

**WHY IT MIGHT LOSE — steelmanned.**

1. **ITS OWN HEADLINE PREDICTION FAILED, AND THAT IS THE ARM'S WIN CONDITION.**
   *"An experiment on 72 further layouts does not establish the predicted
   preference benefit on difficult layouts"* [V]. The route we do not have is
   the route whose benefit its authors could not confirm. **I nominate it for
   the dissociation and the secondary readout, not for the effect** — and a
   builder declining it on the ground that the effect is unestablished is making
   the correct call.
2. **A gridworld is not a body.** Jack's needs are a 6-dim continuous drive
   vector inside an RSSM; theirs are discrete belief states in a tabular
   gridworld. Week 3's rule: an analogy is not arithmetic.
3. **It is active-inference-native, and our EFE arm is demoted.** `A3`/`wm-efe`
   read **t = 2.05**, below the 3-sigma learning gate. A mechanism expressed in
   precision and prior preferences lands most naturally on the arm our ledger
   already demoted — the same tension week 10's N1 recorded, and I record it
   again rather than using it to re-promote `A3`.
4. **No seed count, no hardware, no wall-clock and no params in the abstract**
   [V, checked]. Nine authors and a workshop acceptance are provenance about
   care, not about transfer.
5. **It partly corroborates an unrun nomination of mine — the tenth time this
   has happened.** `wk1`'s precision arm has been live and unrun since
   2026-08-10. **A second paper arguing about route one is not news; the second
   route and the coverage readout are.**

---

### N4 — PER-MODALITY DECOMPOSITION OF THE CURIOSITY SIGNAL as a `CURIOSITY_BAKEOFF` arm, because `cores.py` already computes the dict every gradient step and throws it away

**Source:** *Tactile Curiosity Drives Robot Interaction* (TacEx) —
[arXiv:2609.40134v3](https://arxiv.org/abs/2609.40134), primary category
**cs.RO**, v1 2026-09-30 / **v3 2026-10-13**, Iten, Proshkin, Sukhija, Coros,
Krause, Abbeel & Sferrazza. **Accepted at CoRL 2026**; project site in the
author comment [V]. **Abstract [V] verbatim.** Week 10's queued #6 — **opened,
as promised.**

**THE VERIFIED CONTENT [V].** *"The most common RL algorithms rely on random
action sampling … resulting in agents that allocate most of their training
budget to motions in free space, away from the contacts from which manipulation
skills emerge. Existing intrinsic motivation methods based on model disagreement
or epistemic uncertainty improve on isotropic noise, but they can also reward
uncertainty in functionally irrelevant transitions, such as erratic motions in
free space. … TacEx … incorporates touch into epistemic
uncertainty-driven exploration by decomposing model uncertainty across sensory
modalities and directing curiosity toward the tactile channel."*

**WHY THIS IS FRONT 4's SECOND NOMINATION IN ELEVEN SWEEPS AND WHY IT IS
DIFFERENT FROM N1-OF-WEEK-10.** Week 10's N1 (ALARM) is a **known-answer
pre-gate**: it can prove an implementation is faithful to its objective.
**This is a candidate for the objective itself**, and it names our measured
defect in its own words. `T3.06`'s committed row measured that **a
random-action policy covers W0 as well as the curious arm** (curious−RANDOM
t = 0.39) while the task arm beats it (10.48 vs 6.54). *"They can also reward
uncertainty in functionally irrelevant transitions, such as erratic motions in
free space"* **is that result's mechanism, stated by another lab.** And touch
is **our measured dominant modality**: `T4.06`'s incumbent gradient fingerprint
reads `touch 2.359e-3` against `audio 7.9e-5` and `vision 6.75e-4` [M].

**THE COST, MEASURED FROM CODE, AND IT IS THE CHEAPEST THING ON THE DESK.**
`experiments/cores.py:382-398` [M] — `A4`'s `_per_key` already computes, for
**each of six senses** (`vision` 32, `audio` 16, `touch` 8, `proprio` 12,
`needs` 6, `language` 32), a leave-one-out latent-prediction error: drop sense
`k`, re-encode, predict the latent, MSE against the full-input target. Six RSSM
forwards per step, **already paid every gradient step** because `binding_loss`
calls it. `binding_loss` then `.sum()`s the dict; and `uncertainty()`
(`cores.py:406-414`) **ignores it entirely**, returning
`preds.var(0).mean()` — the ensemble's variance over the **shared 64-dim
latent**, a single scalar from which no modality can be recovered.

**So the per-modality decomposition this paper is built on already exists in our
code, is computed on every optimiser step, and is discarded by one `.sum()` and
one `.mean()`.** Zero new parameters; zero new forward passes for the
attribution form. The faithful *epistemic* form — ensemble variance evaluated on
the six leave-one-out states — is also **zero new parameters** and costs
5 extra forwards of a ~76k-param MLP per step, which at `LC.08`'s measured
53,974 optimiser steps is **≈1 TFLOP of added arithmetic over a 14.4 core-h run
[C], i.e. noise.**

**WHY IT MIGHT LOSE — steelmanned.**

1. **THE CHEAP FORM IS NOT THE PAPER'S QUANTITY, AND THE SUBSTITUTION IS MINE.**
   `_per_key` is a leave-one-out **attribution** of one predictor's error;
   TacEx decomposes **epistemic uncertainty** across modalities. *A sense whose
   removal hurts prediction is not a sense whose future is unpredictable.*
   Using the first as the second would be the kind of quiet proxy substitution
   this desk has refused four times on front 2. The faithful form is priced
   above and is the one I nominate.
2. **IT MAY MAKE `T4.02` WORSE, AND THAT IS A CONSTITUTIONAL RISK NOT A
   PERFORMANCE ONE.** `T4.02` FAILed twice on modality imbalance (30.12 against
   a 10× gate) with **touch as the measured dominator**. An arm that
   deliberately steers curiosity INTO the already-dominant channel could deepen
   exactly the imbalance `GOAL.md` forbids (*"no modality collapse"*). **Any
   `UB.10`/`T4.02` reading must be a reported primary for this arm, not a
   secondary.**
3. **It re-introduces designer knowledge, which this sweep's own front-4
   literature attacks.** `2610.02012` [V] argues intrinsic objectives
   *"involve the selection of information variables, which re-introduces domain
   expertise the field has sought to eliminate."* **Choosing touch is choosing
   an information variable.** The defence is that our choice would be
   *measured*, not assumed — the fingerprint says touch dominates — but that
   defence is circular if the fingerprint is itself a consequence of the loss.
4. **Robot manipulation, GPU, no params / hardware / seeds / wall-clock in the
   abstract** [V, checked]; pick-and-place and VLA post-training are not a
   creature with needs climbing a ladder.
5. **A4 is the arm where this is most interesting and `A0`/`A1` are where it is
   free.** `PPOCore.uncertainty` (`cores.py:249-254`) collapses its own per-key
   dict the same way, and `CURIOSITY_BAKEOFF`'s model-free `LT.04` is where a
   PPO arm would live. The two halves of the nomination sit on different seats
   and that is a sequencing problem for the builder, not a tidy single arm.

---

### N5 — NOT AN ARM: the MATCHED CONTINUED-TRAINING control, which this paper measured absorbing an auxiliary objective's ENTIRE reported gain

**Source:** *One-Step Prediction Fidelity Is a Poor Proxy for Planning
Performance in JEPA World Models* —
[arXiv:2610.19375](https://arxiv.org/abs/2610.19375), primary category
**cs.LG**, 2026-10-14, Cheung & He. **Accepted at the NeurIPS 2026 Workshop on
World Models in Physical AI** (author comment [V]). **Abstract [V] verbatim,
and the abstract carries the numbers and the seeds.**

**THE VERIFIED CONTENT [V].** *"Latent world models are routinely selected,
checkpointed and ablated on one-step prediction loss … three continuous-control
environments, with three training seeds and paired planning seeds throughout.
… the first epoch improves fidelity by 32% and planning by 4.4 pp, but across
the fifteen checkpoints that follow — a further 15% fidelity improvement,
monotone in every training seed — the correlation between one-step error and
planning success is r = −0.010. … a 93% fidelity reduction on TwoRoom yields
−0.7 pp of planning, while a 40% reduction on PushT yields +4.9 pp [+1.1,
+8.9]. All results replicate under a decoder-free measure, latent drift. **We
also show that a matched continued-training control absorbs the entire fidelity
gain we had attributed to an auxiliary objective.**"*

**WHY IT IS A DISCIPLINE AND NOT AN ARM, AND WHY IT IS THE CHEAPEST THING ON
THIS PAGE.** The last sentence is a **must-fail control result**, and
`SYSTEM.md` and `GOAL.md` both make must-fail controls constitutional
(*"with controls that must fail"*). This desk has **accepted and not run**:
`A4b`/`A4c` (wk1), the spectral-radius constraint (wk4), PSG-JEPA's two terms
(wk4), SIGReg in two forms (wk5, wk8, narrowed by wk10-N4) — **every one of them
an auxiliary objective added to a core.** This paper measured that in at least
one such case the entire reported gain was *more training*, not the objective.

**The nomination is one line, and it costs nothing to adopt and everything to
skip:** any arm that adds an auxiliary objective carries a **matched
continued-training control** — the incumbent, trained for the same additional
compute without the new term — and that control's effect is **PRINTED on the
row.** `LC.08` is the proof this project can do it and already has: it matched
*"identical seeds, identical step budget (400,000 decisions AND 17,280 core-s,
whichever is later), identical venue"* and multiplied the term by zero. **The
discipline is to require that shape BEFORE an auxiliary-objective arm is
scored, not after it wins.**

**It also narrows a live reading of mine in the useful direction.** Week 10's N4
said *"do not assume SIGReg answers collapse."* This says *"do not assume an
auxiliary objective's gain is the objective's."* Together they bracket all five
unrun anti-collapse routes with two cheap, pre-registerable obligations and **no
new compute.**

**Cost: zero new parameters and zero new runs TODAY.** It is a pre-registration
obligation that doubles the compute of any auxiliary-objective arm at the moment
one is bought — which is the honest price and it is not small. Said here rather
than hidden: **N5 makes N1 more expensive**, because a widened predictor is an
architecture change, not an objective, and the matched control for it is the
seated arm itself at matched steps — which `LC.08` has, as of yesterday,
already run.

**WHY IT MIGHT LOSE — steelmanned.**

1. **Goal-conditioned planning in three continuous-control environments is not
   survival in W0**, and our deciding statistic is `life_gain`, not planning
   success. The fidelity-vs-planning decoupling may not even be the axis we
   score on.
2. **The headline claim is about CHECKPOINT SELECTION**, which this project does
   not do by one-step loss — our seats are bought on `lg_margin_null` and
   `lg_margin_twin`. So most of the paper does not reach us, and I nominate one
   sentence of it.
3. **The absorbed-gain result is reported in one clause with no number, no
   objective named and no table reference** [V, checked at abstract level]. It
   is the single most consequential sentence in this sweep's reading and it is
   also the least substantiated one I could reach.
4. **Two authors, no hardware, no params, no wall-clock, no code statement.**
5. **It is partly already ours.** `LC.08` did this unprompted and `SYSTEM.md`
   arguably already requires it. **Then the nomination is corroboration, and
   corroboration is not news — the ninth time this desk has had to say so.**
   What is new is a *measured instance* of the control killing a published gain,
   which is the half that makes it an obligation rather than good taste.

---

### WITHDRAWN — `wk10-N2` (Jacobian centroids, arXiv:2609.33940): its premise was a reading of OUR ledger, and `LC.08` measured that reading false

Week 10 nominated Jacobian centroids as *"the encoder-vs-predictor separator
`LC.03`'s five controls cannot make"*, on the strength of the paper's
*"structural dissociation, where the encoder correctly represents the goal while
the predictor remains behaviorally unresponsive."* That nomination's whole
argument was that this **is** week 8's §6 reading, found in another lab.

**§0(1): `LC.08` switched `A4`'s binding objective off and the arm collapsed —
`life_gain` 96.83 → −19.31, both seat statistics down by 116.14 against seed
noise of 37.33 and 42.25, `deadweight_reading 0.0` [M].** A predictor whose
removal costs three sigma of the seat's own margin is not behaviorally
unresponsive in this arm. **The premise is gone.** The venue is gone too —
`D37` fired (iii), §0(2) — but I am withdrawing on the premise and not on the
venue, because a closed venue is a reason to defer and a false premise is a
reason to stop. Week 10 also called it *"the worst-provenanced nomination in ten
weeks"* and pre-committed that *"if the builder declines it on provenance alone,
that is the correct call"*; it is withdrawn before anyone had to.

**What survives, narrowly:** nothing about the paper is impeached — I never
opened it — and `A4c`'s LeJEPA arm is untouched. **What does not survive is my
use of it.**

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **2610.05550** — *When Low Prediction Error Misleads Planning: Diagnosing Representation, Dynamics, and Decision Failures in Latent World Models* ([abs](https://arxiv.org/abs/2610.05550), 2026-10-04, Min, Li, Xu & Du) | cs.LG | **A three-way localisation of latent-world-model failure — representation vs dynamics vs decision — which is the separator `wk10-N2` was withdrawn from doing** [V]. Measured across *"four model families and four tasks"*, with *"a confirmation pool of 256 new starts per task and 300 shared candidates per start"*, *"center error dominates MSE in 14/16 model-task cells"*, yet in six cells correcting only the action-relative responses beats correcting the center. `Spearman 0.464 → 0.975` on one Reacher setting [V]. | **Nothing, while `D37` stands.** It is a readout, the readout slot closed by armed default on 10-05, and promoting a replacement into a foreclosed venue is the error week 10 named. **Promote on: `D37`'s option (i) being ruled by the owner** (the resolution says it *"remains the owner's to rule at any time"*) — at which point this is a better-evidenced candidate than either of the two it would replace, because its localisation is three-way and measured on 16 cells. |
| **2610.31446** — *When Is a Memory Safe Enough to Act On? Action-Conditioned Verification for Long-Lived Agents* ([abs](https://arxiv.org/abs/2610.31446), 2026-10-22, Wang et al.) | cs.MA | **The first front-3 paper in eleven sweeps whose abstention decision is conditioned on the ACT rather than on the retrieval score** [V]: *"whether an entry must be re-checked before acting is not a property of the entry"*; a sequential stopping problem where *"a verification is worth its cost only if it can change what the agent does next"*; `−33.9%` verification cost at `100%` safe-proceed recall and `0%` gate-harmful, coverage `38.83% → 44.17%` on ToolSandbox. **This is `MEMORY_RETRIEVAL_BAKEOFF` §1.8's "actual hard part" re-posed with the consequence of the action in it — and `GOAL.md`'s world has irreversible acts (death) where a stale memory costs a life.** | **Two things.** (i) Whether the belief calibration and premise extraction can run without an LLM in the loop — "extracts the premises a candidate action rests on" reads like generation, and the constitution forbids it in the read path. (ii) Any CPU latency figure; ours is now **6.397 ms/query at 100k** (`ME.11.C`, [M]) so §1.9's table finally has a local anchor to compare against. **Promote on: a generator-free premise extractor, or an explicit statement that the VOI core is tabular.** |
| **2610.10265** — *Stale, Misattributed, or Late: Where Personal Memory Fails Before Generation* ([abs](https://arxiv.org/abs/2610.10265), 2026-10-07, Deng & Sinchaisri) | cs.AI | **Measures the three failure modes of a PERSONAL memory before the answer is generated, and the read path is extractive** [V]: *"participant-aware BM25 is equivalent to the reference ranker within a prespecified 0.02 margin"*; without update resolution *"70.3% of prompts expose a superseded value"*; *"open-domain merge recall on LongMemEval never exceeds 0.062"*; and **misattribution survives validity filtering — an entity posterior *"cannot distinguish identically named speakers"***. `ME.9` is attributed recall per person; `GOAL.md` says his diary records *whose* advice proved true. **STALENESS is the part `ME.9` does not test at all** — a diary that serves "the berry bush at X has fruit" after the bush is stripped is a survival failure, not a retrieval metric. | **It is close to nominatable and is held back by one thing: it is a benchmark for language agents, and the identity-collision half duplicates `wk1`'s entity-collision protocol (2605.29630), live and unrun for ELEVEN sweeps.** A third paper asking for a control we already ordered is not news. **Promote on: the TEMPORAL-VALIDITY half being separable from the LLM harness** — i.e. a superseded-value eval design that does not need a generator. Then it is an `ME.9` eval-set unit, not an arm. |
| **2610.25377** — *Simulating Infant Senses with Age-Tuned Networks* ([abs](https://arxiv.org/abs/2610.25377), 2026-10-18, Somesh Mohapatra, single author, **code + site released**) | cs.LG | **Biology-as-oracle on `GOAL.md`'s own unmined shelf — "critical periods"** [V]. Per-sense age-indexed input filters with a small prediction network learning behind them; *"a critical period and perceptual narrowing emerge without being built in"*; and the measured hook: at the newborn stage *"a toy's position cannot be read out, R² = 0.00, against 0.57 for a network trained directly on the same input."* **A per-sense input filter plus an R² readout is structurally the same instrument as `T4.06`'s per-modality probe**, which is why it is legible to us at all. | **Not an arm: there is no agent, no action and no need in it** — the networks predict, they do not live. **Its transferable part is the DEVELOPMENTAL SCHEDULE as a curriculum that is not a curriculum** — `GOAL.md` forbids a written curriculum and this is an *input* schedule, which is a different object and may be admissible. **Promote on: any result where the age-filtered schedule beats full-fidelity input on a DOWNSTREAM control or survival task** — i.e. evidence that starting degraded is better, not merely different. Without that it is a measurement tool for a phenomenon we have not claimed. |
| **2606.00667** — *Cortex and subcortex play distinct roles over learning when cortical memory is limited* ([abs](https://arxiv.org/abs/2606.00667), 2026-05-30, Farrell & Toyoizumi) | q-bio.NC | **QUEUED #7 SINCE WEEK 9, OPENED TODAY ON ITS LAST LEGAL SWEEP** [V]. A model-based module with **explicitly constrained memory** learning in tandem with a model-free one; *"when the rewarded states change often, it can be advantageous for the model-based module to focus its memory resources not on exploiting the current reward, but on capturing general structure of the environment."* **This is `DP.00`'s world question and the diary-vs-weights split in one theory**: a prediction that under non-stationarity the model-based path should specialise AWAY from reward and toward structure. | **Promote on: an implementable statistic.** It is theory in *"a simple decision-making setting"* with no agent, no body and no code statement, and `DP.00` has not yet established that our world rewards looking ahead at all. **Its real value is a pre-registerable DIRECTION for `DP.02`'s lesion** — if the shared trunk is lesioned, the structure-learning and reward-learning halves should degrade differently under non-stationarity. **Carried as a design input with its deferral rule spent: it has now been opened, so the week-9 deadline is DISCHARGED rather than dropped.** |
| **2610.26395** — *ForagerSim* ([abs](https://arxiv.org/abs/2610.26395), 2026-10-19, Musil et al., **code + data released**) | cs.RO | A modular embodied platform with *independent* difficulty axes — perception (abstract 2D → RGB/LiDAR), scale (room → multi-km), topology, dynamics (smoke, debris, **sensorimotor failures**), tasks (coverage → foraging well-alternation) — benchmarked against RL baselines, an engineered stack and **a human player** on 8 worlds [V]. The measured finding: with 2D sensing learned policies do well in simple topologies but *"fall short of human performance in larger or more complex-topology environments."* | **Promote on a statement that a configuration runs single-threaded on CPU.** This is the SAME gate that resolved-and-deleted MuJoCoUni (wk3) and still blocks SmallWorlds (eighth sweep): we run ONE life serially on 4 shared ARM cores. **Its genuinely interesting half needs no port: the independent-difficulty-axis design is what `SURVIVAL_WORLD`'s fidelity ladder is, and a world-design reference now has a venue again (§0(3)) where week 10 said it had none.** |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **2609.34677** — FAR (week 10's front-3 watchlist) | cs.LG | **DROPPED AS AN ARM — the promote question is ASKED AND ANSWERED *NO*, which is queued #2 closing negative.** The abstract [V] names the training signal exactly: predictive utility is *"the conditional log-likelihood of the realized future given recalled context, **approximated by negative diffusion prediction loss**"*. **A diffusion world model scores which memories are worth keeping.** The READ path is clean — the retriever *"remains future-blind at inference"* and selects over stored memories, so recall stays extractive and constitutionally admissible — but the WRITE/training path requires training a diffusion model, which is the objection that held Eywa, ScrubJay-MEM and RARE, and it is independently impossible here on cost. **Its one durable contribution is recorded and the paper is not carried: learned per-cue trust (time, pose, vision, audio) is a second independent argument against a single fixed similarity cut, the defect behind `ME.11`'s `τ_fpr 0.388 > τ_cov 0.227`.** |
| **2609.22056** — RegimeAbstain (`wk9-N2`) | cs.IR | **ADMITTED AS A DESIGN UNIT 10-02 and the gap it addresses is now WORSE-EVIDENCED, not better.** `ME.11` re-ran to attempt 2 (FAIL, 2026-10-20) with the headline numbers **unchanged to four decimals** — `tau_cov 0.227233`, `tau_fpr 0.388167`, `feasible_ok 0.0`, `answered_cues 27.33 ± 4.78` of 160 [M]. **§6 is about what else that row now says.** |
| **2609.30210** — alignment-illusion corruption control (`wk9-N3`) | cs.CV | **ADMITTED AS A CONTROL DISCIPLINE 10-02; fires on the next fusion readout designed. N2 above is its third sibling and not its duplicate** — the distinction is drawn in N2. |
| **2609.36097** — known-generator ablation calibration (`wk10-N3`) | q-bio.NC | **LIVE, one sweep old, unrouted.** `UB.11`'s positive control and `UB.10`'s matrix are still the venue; `UB.10` is **VOID** (attempt 1, 09-01) and **`T4.03` is REGISTERED with no ledger row** [M] — unchanged from week 10. |
| **2609.36227** — one-step latent prediction is not a world model (`wk10-N4`) | stat.ML | **UNCHANGED AND NOW CORROBORATED FROM A SECOND LAB.** `2610.19375` (N5) is an independent experimental paper reaching the compatible conclusion that one-step latent fidelity does not buy planning (`r = −0.010` across 15 checkpoints, 3 seeds). The narrowing of `wk5-N1` and `wk8-N3` stands. |
| **2610.02159** — ALARM counterexample pre-gate (`wk10-N1`) | cs.LG | **LIVE, UNRUN, and still the cheapest front-4 item this desk has ever nominated** (three actions, four observations, tabular, single-digit CPU-seconds). Nothing in the repository references it [M]. **One new piece of evidence FOR it arrived this sweep: `2609.39342`** [V] extends the **Maximum Occupancy Principle** — one of the objective families ALARM's Lean supplement proves deficient at `D(π) = 1/3` — to partial observability, and finds MOP agents *"switch between goal-directed (food seeking) behavior and exploration … depending on their energy available and their belief state."* **A family being actively developed is a family worth pre-gating.** |
| **2605.27929** — transition geometry (`wk9-N1`) · **`Δ_k`** (`wk7-N3`) | q-bio.NC | **FORECLOSED, not dormant.** `D37` fired (iii) on 10-05 (§0(2)); the Review's admission condition was *"fires only if `D37` resolves toward building a readout"* and it resolved the other way. **The word week 10 pre-committed is the word that now applies.** |
| **2508.21038** — free-embedding critical-`d` probe (`wk7-N1`) | cs.IR | **ACCEPTED, ORDERED FIRST 09-14, STILL NOT RUN AT 42 DAYS** [M: `grep -rln "free.embedding\|critical_d"` returns **four files, all documents, zero `.py`**]. **§6 is the argument that this is now the single most mis-prioritised item on the desk**, because `ME.11`'s own numbers say the retriever — not the threshold — is the binding constraint. |
| **2608.04232** — interoceptive precision (`wk1`) | cs.AI | **LIVE, UNRUN, ELEVENTH SWEEP** — code released, 4 ARM cores, 2.08× survival over the uniform-precision baseline that IS our current design. **N3 above is route two of the dissociation this arm is route one of.** |
| **2605.29630** — entity-collision protocol (`wk1`) · **2607.27017** — `UB.11` certificate (`wk1`) | cs.IR / cs.LG | **LIVE, UNRUN, ELEVENTH SWEEP.** Both now have a second independent paper arguing for them (2610.10265 and `wk10-N3` respectively). **Corroboration is not news; the eleventh time of saying it.** |
| **SmallWorlds** ([2511.23465](https://arxiv.org/abs/2511.23465)) · **ForageWorld** ([2506.06981](https://arxiv.org/abs/2506.06981)) · **MINERVA** ([2609.03715](https://arxiv.org/abs/2609.03715)) · **3M-Progress** ([2506.00138](https://arxiv.org/abs/2506.00138)) · **PRIME** ([2607.16858](https://arxiv.org/abs/2607.16858)) · **2609.39239** (depletable foraging) | cs.LG / cs.AI / cs.RO / q-bio.NC / q-bio.PE | **Unchanged, and ForageWorld and 2609.39239 are UNBLOCKED by §0(3) rather than promoted** — week 10 held both behind a venue it believed closed, and the venue was open. Promote conditions are unchanged: SmallWorlds on *any* CPU statement; MINERVA on an RL result; 3M-Progress on *any* exploration or survival number; PRIME partly superseded by `wk10-N1`. |
| **IIBalance**, **Eywa**, **ScrubJay-MEM**, **RARE/RedQA**, **CoDeR**, **Argus Eyes**, **POBAX**, **Curiosity-Critic**, **2608.11506** (allostatic probing) | cs.MM / cs.CL / cs.IR / cs.LG / cs.NE | **Unchanged and NOT re-checked this sweep.** Dispositions stand from weeks 4–10. Stated so the absence is not read as a re-confirmation. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS

| nomination | entered | status now |
|---|---|---|
| **wk8 · §6 (`LC.03`'s controls never switch off `l_bind`)** | its own row | **BUILT AS `LC.08`, RAN, AND REFUTED THE SCOUT'S READING. PASS attempt 1, 2026-10-25** [M]. **The first finding on this page to become a passing ledger spec, and it closed against me.** §0(1). |
| **wk10-N2 · Jacobian centroids (2609.33940)** | `D37`'s BUILD half | **WITHDRAWN BY ME THIS SWEEP** — premise refuted by `LC.08`, venue separately foreclosed by `D37`. §2. |
| **wk10-N1 · ALARM pre-gate (2610.02159)** | `CURIOSITY_BAKEOFF` pre-gate | **LIVE, UNRUN, one sweep old.** Strengthened by `2609.39342`'s active development of a family ALARM proves deficient. §3. |
| **wk10-N3 · known-generator calibration (2609.36097)** | `UB.11`/`UB.10`/`T4.06` | **LIVE, UNROUTED, one sweep old.** `T4.03` still REGISTERED with no row [M]. |
| **wk10-N4 · the derivative argument (2609.36227)** | narrowed wk5-N1 + wk8-N3 | **STANDS, and is independently corroborated by N5's source.** |
| **wk9-N1 · transition geometry** · **wk7-N3 · `Δ_k`** | `D37`'s readout slot | **FORECLOSED 2026-10-05 by armed default.** Not dormant. §0(2). |
| **wk9-N2 · RegimeAbstain** | `MEMORY_RETRIEVAL_BAKEOFF` §1.8 | **ADMITTED AS A DESIGN UNIT 10-02; `ME.11` re-ran to attempt 2 FAIL with identical τ values** [M]. **§6 argues the unit is correctly specified and incorrectly prioritised.** |
| **wk9-N3 · corruption intervention** | fusion readouts | **ADMITTED AS A CONTROL DISCIPLINE 10-02.** No fusion readout has been designed since. |
| **wk8-N1 · replanning-frequency ablation (2609.05461)** | the `a4` row | **FORECLOSED with the rest of the readout slot.** Fourth sweep live, never run, and now sequenced behind a decision that resolved against building. |
| **wk8-N2 · do-nothing reference arm (2609.11247)** | `t402` → `T4.06` | **IN THE SPEC AND THE SPEC HOLDS A PASS.** `ctrl_incumbent_wins 0.0`, `refuted_arms ['grad_norm']`, `n_winning_arms 1.0` [M]. §6 is about what the winning margin was measured against. |
| **wk8-N3 · temporally-centered SIGReg** | `A4c` amendment | **NARROWED BY wk10-N4 AND NOW DOUBLY OBLIGED BY N5** — if it is ever built it carries a predictor-side readout *and* a matched continued-training control. |
| **wk7-N1 · free-embedding critical-`d` probe** | `ME.11` successor pre-gate | **ACCEPTED, ORDERED FIRST, UNRUN AT 42 DAYS, ZERO `.py` HITS** [M]. **§6 re-prices it from "cheapest live item" to "the item the ledger's own numbers say should be first."** |
| **wk7 · §6 (`A4`'s diagnostic computed nowhere)** | the `a4` row | **SETTLED BY HOLD, AND THE GREP IS NOW ZERO.** `effective_rank|rankme` over `*.py` returns **0** where week 10 read 4 [M]; `LEARNING_CORE.md` §5.4 still declares it. **Six weeks of "declared and uncomputed" became "declared, uncomputed, and formally held."** |
| **wk6-N1 · Context Collapse** | the `a4` row | **CORROBORATED TO EXHAUSTION AND NOW PARTLY CONTRADICTED IN OUR ARM.** `LC.08` says our latent-prediction term is load-bearing, which is not what a collapsed predictor looks like. **No amount more of this literature will arm the guard, and the guard is now held shut.** |
| **wk6-N2 · MULTIBENCH++ redundancy pre-gate** | `ub10` | **Unchanged.** `UB.10` VOID; `T4.03` registered, no row [M]. |
| **wk6-N3 · IAF pathway-decoding control** | `t402` → `T4.06` | **Unchanged and still half-in.** `T4.06` measures per-modality latent R² AFTER fusion and no unimodal BEFORE — and §6 raises the price of the missing half again now that the 12-run replication is in hand. |
| **wk5-N1 · SIGReg as variational free energy** | `LEARNING_CORE` §5.4 | **NARROWED (wk10-N4), and N5 adds the control it must carry.** |
| **wk5-N2 · prosociality by coupling** | `NE.07` | **Unchanged. `NE.07` and `NE.02` have no ledger rows** [M]. |
| **wk4-N1 · spectral radius** · **wk4-N2 · PSG-JEPA ×2** | `A4` variants | **ACCEPTED, narrowed, UNRUN — eleventh sweep.** |
| **wk4-N3 · infant motor noise** | `W0.DIAG` | **RUN AND PASSED. Still the only arm-side nomination to become a number.** |
| **wk1 · anti-collapse `A4b`/`A4c`** · **wk1 · UB.11 certificate** · **wk1 · interoceptive precision** · **wk1 · entity collision** · **wk2 · RPE replay** | various | **LIVE, UNRUN — ELEVENTH sweep for all five.** |
| wk2 · whiff clock → `SM.02` | `SM.02` | **HELD, correctly — `SM.02` is PARKED.** |
| wk3 · CIG → `A3` · wk3 · Optimistic World Models → `A2` | `A3`, `A2` | **Remain DEMOTED.** N3's objection 3 records the same `A3` tension week 10's N1 did, rather than using it to re-promote. |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

**FRONT 1 · LEARNING CORES — ONE ARM, ONE CONTROL, ONE DISCIPLINE, AND A
REFUSAL OF TWENTY PAPERS.** Four enumerations returned the most crowded field
this desk has ever enumerated. Named and **not nominated**: `2610.32657`
(DT-JEPA), `2610.31194` (Inside-JEPA, relational grounding), `2610.27364`
(predictive closure), `2610.28458` (LocaLeJEPA, maximum-entropy under geometric
priors), `2610.15429` (spectral alignment), `2610.15267` (LeHami, symplectic
latent dynamics), `2610.09457` (DSReg, *"provably recovering individual world
latents without reconstruction"*), `2610.12016` (CausalDreamer), `2610.25013`
(planning along the latent manifold), `2610.24534` (structured latent
dynamics), `2610.28074` (Slot-JEPA), `2610.27921` (Quantum-JEPA), `2610.06805`
(H-JEPA), `2610.25296` (world-model interpretability), `2610.19683` (associative
memory in latent WMs), `2610.28519` (**Touch-JEPA** — tactile representation
learning, our dominant sense, and still a vision-video pipeline),
`2610.06582` (action-semantic mismatch), `2610.32414` (dynamics randomisation
as a context-inference test), `2610.07028` (identifiable WMs from diffusion
representations), `2610.16517` (trust-calibrated VLA). **I nominate none of
them as arms.** Twelve sweeps of this desk have promoted **five** anti-collapse
/ latent-structuring routes and run **zero**. **The three front-1 items I did
nominate are an ARM THAT IS AN INTEGER (N1), a CONTROL (N2) and a DISCIPLINE
(N5), and not one of them is a sixth loss term on a seat with five unrun
challengers.** That distinction is the whole of my position on this front and
it is now four sweeps old.

**FRONT 2 · FUSION — NO METHOD, SEVENTH CONSECUTIVE SWEEP, AND THE COVERAGE WAS
SHALLOW AND SAYS SO.** Two 40-entry enumerations, **no dedicated fetch**:
Alzheimer's MRI-PET, chest X-ray, ICU delirium, dental pulp stimulation,
pedestrian intention, meme classification, e-commerce retrieval, sinkhole
susceptibility, food nutrition estimation, ECG integrity screening. **Not one
paper with a world-model objective and five heterogeneous senses — fourth
consecutive year.** The unified-tokenisation line is unchanged: any-to-any
generation and VLA-scale autoregression, parameter counts far off this box.
**But the front's real news came in sideways and is in §2: N1's title sentence
is literally this front's question** (*"shared is not enough"*), measured at
0.7M params, and N2 is the control on how we read our own fusion probe.
**Front 2's instruments no longer need the literature; they need §6's floor,
wk6-N3's missing unimodal before, and a `T4.03` row.**

**FRONT 3 · MEMORY — SWEPT PROPERLY AFTER TWO REFUSED CYCLES, THREE WATCHLIST
ENTRIES, AND NO NOMINATION FOR THE SIXTH CONSECUTIVE TIME.** Three enumerations
confirmed the standing shape — MEMORA, RL-guided temporal localisation,
multimodal filesystem memory, Gamma, MemCo, RippleMem, MemFuse, Agent Zero
Memory, CapMem, MemTransfer, Hippo (*"training-free agent self-evolution via
hippocampal replay"*), GROVE, Consolidator: **LLM-agent memory, generative in
the read path or one step upstream in the write path, almost entirely cs.CL.**
The abstention half returned a genuinely better crop than any prior sweep —
`2610.24030` (label-free boundary-cut abstention, `AUC 0.736 → 0.785`,
95% CI `[+0.008, +0.092]`, **a zero-cut subset of 71/400 queries at 100%
observed precision** [V]), `2609.38222` (conformal factuality control),
`2609.19942` (intrinsic sequence-likelihood confidence with two pre-specified
negatives) — and **all three are scored on public multi-hop QA corpora where
Jack's corpus is his own life, and all three are threshold-side.** **§6 is why
no threshold-side paper can be nominated onto `ME.11` this sweep, and it is a
finding about our ledger rather than about the field.** `2609.34677` (FAR) is
dropped with its question answered (§3).

**FRONT 4 · CURIOSITY — ONE ARM, AND EVERYTHING ELSE IN WINDOW STILL FAILS THE
SAME TEST.** Beyond N4, the two enumerations returned: `2610.03563` (a unified
framework for empowerment and predictive control), `2610.25957` (productive
state-space exploration for multi-turn LLM RL), `2608.27507` (marginal coverage
credit in parallel state-entropy optimisation), `2608.17373` (novelty +
surprise for experience prioritisation), `2607.18433` (*Intelligence from
Learnable Novelty*), `2606.19924` (*The Tao of Agency*, autotelic position
piece), `2609.17325` (research agenda), `2609.11660` (developmental position
piece), and a UED/curriculum half that was **again almost entirely LLM
harnesses** — EvoRubrics, COvolve, GASP, Tool-R0, SearchGym, *Dreaming in Code*.
**Not one has a body under homeostatic drive.** `2610.14302` (DySCOVER,
predictive curiosity for robot vision) was **opened and not nominated** [V]: it
measures that predictive curiosity beats reconstruction curiosity for *visual
representation learning* on robot arms, with **no survival, no needs, and vision
as the only sense** — the arm it would add is `disagree` against `disagree`,
which we already race. `2610.02012` (F-CIP) was **opened and not nominated**
[V]: *"defined by the system's dynamics alone and requires no such selection"*
is the most `GOAL.md`-shaped claim in the enumeration, and it produces
*"balancing and maintaining controllability"* plus gaits **paired with a
forward-velocity reward** — a task reward, in a world with no needs, with no
params, hardware or seeds in the abstract. **Recorded as the standing
counter-argument to N4's designer-chosen channel rather than as an arm.**

**FRONT 5 · WORLDS & EMBODIMENT — A NOMINATION AT LAST, AND THE REASON IS THAT
WEEK 10's REFUSAL WAS WRONG.** The enumerations returned collective-foraging
physics (percolating recruitment cascades, emergent aggregation, intermittent
random walks), camelid territoriality, three-trophic foraging optima,
evolutionary connectomics, multi-robot allocation, and **two more
medical/physics uses of the word "survival"** — week 2's lesson recurring for a
fifth time. **The MuJoCo ecosystem check (standing mandate item) returned the
same answer as weeks 3, 9 and 10:** the live developments are GPU-parallel and
manipulation-side (CusADi2 batch kinematics on GPU, DexForge, Code2Worlds-2,
I2CD collision geometry, a tendon-driven soft-arm quadrotor rig). **Two items
are on our axis and both are watchlisted rather than nominated** —
`2610.26395` (ForagerSim, §3) and `2610.05997` (**the MIMo caregiver extension**
[V]: an articulated caregiver in MuJoCo generating *"infant-perspective touch
and vision"* with **origin-aware contact logging that disambiguates self-,
caregiver- and environment-generated contact**). That last one is
`GOAL.md`'s *"his people are part of his world"* with a touch channel attached,
and the only reason it is not a nomination is that it is a **dataset generator
with no agent and no learning** — it replays recorded interaction rather than
letting a creature act. **Promote on: a closed-loop infant policy, or an
admission that replay suffices for a `UB`-family input condition.**
`2609.13771` (*Homeostatic Continual Learning*) was **opened and declined** [V]:
single author, *"I discuss the works needed to render the method practical"* —
a proposal paper with no experiment, no baseline and no number, which week 3's
deferral rule disposes of without further argument.

**SMALL-MODEL END — NOTHING IN WINDOW, SEVENTH CONSECUTIVE SWEEP, AND ONE
NEAR-MISS WORTH A LINE.** The enumeration was VLA and agent-harness end to end:
backtracking recovery for frozen VLAs, skeleton-grounded latent action models,
METTA, DIAL, RIPPLE. Nothing under 1M parameters with an RL result. **The
near-miss is N1 itself — 0.7M-parameter encoders — and it arrived through
front 1, not here**, which says the search terms on this front are looking for
the word "tiny" in a literature that says "small" and reports a number instead.
Our own `ppo-needs` at **135,961** and `wm-latent` at **861,545** [M, re-measured
at HEAD today] remain smaller than almost anything this literature is proud of.

**BIOLOGY-AS-ORACLE — the standing `q-bio.NC` search is four-for-four and
produced a watchlist entry plus the discharge of a two-sweep debt.** It returned
`2610.25377` (infant senses, §3) and **closed queued #7 by opening
`2606.00667`** (§3). Also returned and **not** fetched: `2610.26424` (online
preplay and offline replay in a closed-loop navigation model — the live form of
`wk2`'s RPE-replay nomination), `2610.04697` (efficiency and robustness
partition the solution space for memory in recurrent networks), `2610.18024`
(structural plasticity across scales), `2610.21486` (a systems account of
sleep as cyclic calibration), and `2608.21814` (hippocampal replay without
symmetric plasticity — carried unopened for a third sweep and **dropped with
cause next sweep** if a fourth fails to open it). **None of these is nominated
and none was opened.**

---

## 6. A FINDING IN OUR OWN ARTIFACTS — `ME.11`'s own row says the binding constraint is the RETRIEVER, not the threshold, and front 3's doctrine and its one ordered nomination are pointed at the other half

Front 3 has returned no arm for six sweeps and the stated reason has always been
constitutional. **This sweep the reason is different and it is ours.**
`MEMORY_RETRIEVAL_BAKEOFF` §1.8 calls abstention *"the actual hard part"*, every
front-3 nomination for three sweeps has been an abstention mechanism
(`wk9-N2` admitted, `2610.24030`/`2609.38222`/`2609.19942` declined above), and
**`ME.11`'s own committed row says abstention is not where the failure is.**

**EVERY NUMBER READ FROM THE LEDGER AT HEAD** [M, `ME.11` attempt 2, FAIL,
2026-10-20T12:47:49, 3 seeds]:

| quantity | reading | what it says |
|---|---|---|
| `recall_unthresholded` | **0.2500 ± 0.0102** | **with NO threshold at all, the right memory is in the result 25% of the time** |
| `paraphrase_recall_at_1` | **0.0667 ± 0.0147** | **at rank 1, 6.7%** |
| `answered_cues` / `headline_cues` | **27.33 ± 4.78 of 160** | 17% of cues answered |
| `tau_cov` / `tau_fpr` | 0.2272 / 0.3882 | the thresholds are **inverted** — the coverage cut sits below the false-positive cut |
| `feasible_ok` | **0.0** | no feasible operating point exists |
| `abstention_certify` | **0.9433 ± 0.0125** | **the abstention machinery itself certifies at 94%** |
| `lex_null_recall` / `leaky_recall` | 0.0 / 1.0 | the lexical null is dead and the leak control fires — the instrument is honest |
| `family_best_recall_row` vs `family_bar` | **0.0667 vs 0.80** | the family bar is twelve times the best recall any member reaches |

**THE ARITHMETIC THAT MAKES THIS A FINDING AND NOT A RESTATEMENT [C].** A
threshold is a map from a score distribution to an answer/abstain decision. **Its
ceiling is `recall_unthresholded`.** At 0.25, *a perfect abstention rule —
one that answers exactly when the retrieval is correct and abstains otherwise —
would answer at most 25% of cues, against a family bar of 0.80.* So **no
threshold mechanism of any kind can clear `ME.11`'s own bar on this retriever**,
and `feasible_ok 0.0` is not a tuning failure but an arithmetic consequence
reported three attempts running. `ME.11.C` makes it harder to blame the
threshold: it tried **two embedding variants** and both moved recall the wrong
way (`variant_mrl256_recall 0.0146`, `variant_2m_recall 0.0312` against its own
`recall_unthresholded 0.1229`), while `margin_over_arm_b 0.0437` and
`arm_b_recall 0.0` [M].

**THE INDEPENDENT ARGUMENT, FROM A FRONT NOBODY WOULD HAVE SEARCHED.**
`2610.30996` (eess.SY, power-grid transient stability, [V]) is the same claim
with the same structure: *"Abstention turns estimator uncertainty into a demand
on the archive"*; coverage measured at **99.9996%** on a dense synthetic archive
against **73.47%** operational, with only **59.37%** of query states having a
compatible candidate; conclusion, *"archive availability constrains deployment
more than conditional accuracy."* And the operationally useful half: *"Compatible
coverage and chronological availability are **auditable before an estimator is
fitted**."* **Our analogue is auditable before a threshold is tuned, and it is
already measured: 0.25.**

**WHAT THIS RE-PRICES, AND IT IS A PRIORITY CLAIM, NOT A WITHDRAWAL.**
`wk9-N2`'s design unit is correctly specified — a score-distributional abstention
rule is strictly better than a fixed cut — and it is **second in line behind the
thing nobody ran.** `wk7-N1`'s free-embedding critical-`d` probe
(2508.21038, ACCEPTED, **ORDERED FIRST on 09-14, zero `.py` hits at 42 days**
[M]) asks precisely whether the embedding dimension can represent the required
top-`k` combinatorics at all — i.e. whether `recall_unthresholded = 0.25` is a
property of the corpus or a capacity ceiling. **That is the question the row's
own numbers make first, and the ONE nomination aimed at it is the one that has
been ordered and not run for six weeks.**

**LEAD OBJECTION, AGAINST MYSELF.** **`recall_unthresholded` may not be the
ceiling I claim it is**, and I did not read `me_11`'s implementation to confirm
that it is measured over the same cue set and the same `k` as the thresholded
statistic. If it is measured at a larger `k`, or over a different cue
population, a threshold rule could in principle answer cues that the
unthresholded number does not cover, and my arithmetic would be an upper bound
on the wrong quantity. **The claim that does not depend on that:
`feasible_ok 0.0` with `abstention_certify 0.9433` is a row saying its
abstention machinery works and its operating point does not exist** — and two
independent embedding variants failed to move recall. A second caveat: this is
`ME.11`'s FIXTURE, whose cues are synthetic paraphrases, and a 6.7% rank-1
recall on a synthetic paraphrase set may say more about the fixture than about
any retriever. **I did not examine the fixture, and `fixture_hash_seed_only
9c915329f4755c3e` is identical across `ME.11` and `ME.11.C` [M], so both
readings share whatever that fixture is.**

**WHAT I AM NOT CLAIMING.** Not that `ME.11` should pass. Not that `wk9-N2`
should be dropped — it should be sequenced. Not that the constitutional
objection to generative recall is weakened; it is untouched, and FAR's write
path (§3) is this sweep's demonstration that it still bites. **Not that any of
this is new to the project**: the row is FAIL and says so, `feasible_ok 0.0` has
been printed three attempts running, and §1.8 already owns the thresholds.
**What is new is the ceiling arithmetic, the two-variant evidence that the
retriever and not the cut is the binding half, the independent structural
argument from outside the field, and the consequence for the ORDER of two live
nominations of mine.**

---

## 7. A FINDING ABOUT THIS DESK — two sweeps were REFUSED and no committed file in this repository records either refusal

`docs/FIELD_WATCH_LOG.md` is append-only and exists *"so that a reader can see
what the scout believed on each date without diffing a rewritten file."*
`grep -c "^2026-10-12\|^2026-10-19"` on it returns **0** [M]. The sweeps did not
die and did not run: `/data/jack-logs/field_watch.log` lines 212–213 read
`STOPPED at 98% weekly usage` and `STOPPED at 92% weekly usage` [M]. **Both
refusals are recorded, correctly, by the shell script — in an untracked log file
under `/data`, which no committed document references and no instrument reads.**

**THIS IS WEEK 10's §7c CLASS RECURRING THROUGH A DIFFERENT MECHANISM, AND THE
DIFFERENCE MATTERS.** That one was *a sweep that DIES has no owner for its log
line*; its row (`field-watch-rc124-page-is-untrustable-and-then-deleted`) is
**still OPEN, now 24 days old, DUE 2026-11-04** [M], and the candidate repair
its text carries — *"the seal copies an incomplete page into
`docs/FIELD_WATCH_LOG.md`"* — **cannot catch this instance, because there is no
page to copy.** A refusal produces no page, no draft, no `rc`, and no commit.
**So the organ's own refusals are structurally invisible to every reader of the
repository, and the failure direction is a wrong green: `fieldwatch --check`
read rc=0 throughout, because a page from 10-05 with all findings routed looks
exactly like a healthy organ.**

**Generalises week 10's §7 one level up.** That finding was *assert the expected
RESULT COUNT, never the exit status or the fire time.* This one is the same
sentence applied to the organ instead of to its tooling: **a reader that
verifies the content of the last page cannot distinguish "the page is current"
from "the page is three weeks old because the agent was refused twice."** The
cheap standing guard, zero marginal cost: **the log line is owed by the GATE,
not by the sweep** — whichever branch exits should append its own line before
exiting, including the refusal branches. **I have appended the two missing lines
retroactively in this commit**, each prefixed `[retro, from
/data/jack-logs/field_watch.log]` so a reader can tell reconstruction from
contemporaneous record. **I did not touch `scripts/field_watch.sh`, which is the
builder's.** The one-line fix is in `usage_gate`'s caller and naming it is my
part. LESSONS.md candidate, nominated not written.

### 7b — a second finding, and I caused it by publishing the first: the reader's QUOTATION channel is date-blind, so a six-week-old row was credited with owning a finding written today

**THE READER ANCHORS ITS CITATION CHANNEL TO THE SWEEP'S WEEK AND LEAVES ITS
QUOTATION CHANNEL UNANCHORED — AND THE QUOTATION CHANNEL IS THE ONLY ONE THAT
CAN FIRE ON A FRESH PAGE.** `experiments/fieldwatch.py:179-191` is explicit
about why the first guard exists: *"Anchored to THIS sweep's week so a stale
citation cannot route a fresh finding, and `§6` must not be claimed by `§6b` or
vice versa."* The fallback at `:215-219` is a bare 6-shingle overlap
(`MIN_QUOTE_OVERLAP = 6`) against every desk chunk, **with no week anchor, no
date comparison, and no check that the row was written after the finding.**

**The mechanism, and it is structural rather than unlucky.** `resolve`'s rule is
*"Citation beats quotation"* — so quotation is the fallback that fires whenever
no week-anchored citation exists. **On the day a sweep is published no desk has
yet written "wk11 §6", so the citation channel CANNOT fire for any fresh
finding, and every one of them is resolved by the unanchored channel.** The
reader's strongest guard is off at exactly the moment it is needed.

**MEASURED, on this page, before I touched a heading** [M, `fieldwatch --check`]:

```
§0   ROUTED — quoted by queue-row `a4-mandatory-collapse-diagnostic-is-declared-and-computed-nowhere`
```

That row reads **`ROUTED: … | 2026-09-14 | 9075d58 (field watch wk7 §6…)`** [M,
`docs/REVIEW_QUEUE.md:11986`]. **A row routed six weeks ago, against a different
sweep's finding, is credited with owning a section written on 2026-10-26** — and
it matched for an honest reason: my §0 and that row quote the same source,
`LEARNING_CORE.md` §5.4's promise and the `effective_rank` grep. **The quote is
real; the ownership is fiction.**

**AND THEN THE DEFECT DEMONSTRATED ITSELF ON THE FINDING THAT REPORTS IT.**
Re-running the reader after this section was written [M]:

```
§7b  ROUTED — quoted by queue-row `lc03-five-controls-never-switch-off-the-term-a4-is-named-for`
```

That row is **`ROUTED: … | 2026-09-21 | 785f921 (field watch week 8, §6) |
ACTED 2026-10-22`** [M, `docs/REVIEW_QUEUE.md:13314`] — routed five weeks ago,
about week 8's binding-term finding, **and already ACTED four days ago.** So a
**closed** row about an unrelated finding is now credited with owning the
section that reports this very bug, because §7b quotes `LC.08`'s subject matter
while explaining the mechanism. **A finding about the routing channel was
mis-routed by the channel it describes, into a row that can never discharge it
because it is already closed.** I did not construct this; I measured it on the
second run.

**THIS IS THE THIRD INSTANCE OF ONE CLASS AND THE FAILURE DIRECTION IS THE
DANGEROUS ONE.** Week 8's §6b was false POSITIVES from prose overlap; week 10's
§7d was false NEGATIVES from a lost word boundary; **this one is false positives
from missing TIME.** All three end at the same wrong green, and this one is the
most specific: *the channel that routes fresh findings is the one with the
fewest guards.* I cannot tell from the reader's output whether §7's own routing
to the `rc124` row — which I believe IS legitimate, since §7 is in that row's
class and quotes it deliberately — differs in kind from §0's, **and being unable
to distinguish them is the defect, not a complaint about one match.**

**WHAT I DID AND THE LINE I AM NOT CROSSING.** I changed **one nomination
heading** — N2's, which contained the word "finding" and was therefore counted
as a finding section when it is a nomination. That is a miscount repair in the
safe direction: a nomination is not a finding, N2 stays fully reported in §2,
and nothing became less visible. **I did NOT reword §0 to make its false
`ROUTED` go away**, because that would delete the evidence for this very
finding — week 8 refused to reword until a counter went green, and that
precedent binds me here in the opposite direction. **I did not touch
`fieldwatch.py`, which is the builder's.** The repair is a date on the desk
chunk compared against the sweep date; naming it is my part. LESSONS.md
candidate, nominated not written.

---

## 8. What this report does NOT claim

- **No arm here has been run.** Every paper number in §2 and §3 is someone
  else's measurement on someone else's hardware. The only [M] numbers are from
  this repository: the ledger rows, the three instruments, the greps, and the
  parameter counts I measured by building the arms.
- **Every paper is ABSTRACT-LEVEL ONLY.** No PDF, HTML or table was read
  anywhere this sweep. `FETCHED=20/20` means twenty verbatim abstracts, not
  twenty papers. **Per nomination I name what is missing**: N1 has no CIs, no
  hardware, no wall-clock and a single author; N2 has no seeds, no hardware and
  no code statement; N3 has no seeds, no hardware and no params; N4 has no
  params, no hardware and no seeds; N5's single load-bearing clause carries no
  number at all.
- **N1's central mapping is unresolved and I say so in the nomination, not in a
  footnote.** Whether our predictor is half-width or 4× over-wide depends on
  which width the paper's "encoder width" maps to, and the two readings point
  opposite ways. **I nominate on one of them and name the other as the lead
  objection.**
- **I did not run the released simulator behind N1, the released code behind N3,
  or anything behind N4.** Nor did I verify N1's 0.7M figure against a
  configuration file, because I did not open the paper.
- **§6's ceiling argument depends on an assumption I did not verify** — that
  `recall_unthresholded` is measured over the same cue set and `k` as the
  thresholded statistics. The part that does not depend on it is named in §6's
  lead objection.
- **§6 says nothing about the fixture's validity.** A 6.7% rank-1 recall on
  synthetic paraphrases may be a fact about the fixture. I did not read it.
- **`LC.08`'s result is the builder's, not mine.** I read it from the ledger and
  the spec's docstring; I did not run it, and I claim no credit for a spec that
  refuted me. **The arithmetic check (`116.143887`, 3.11× and 2.75×) is [C],
  mine, and shown.**
- **The `W1.01`/`W1.02`/`W1.04`, `T4.06`, `ME.11`, `D37`, queue and reader
  states are the builder's and the Review's records.** I read them out of the
  ledger, `decisions --check`, `review_queue --check` and `fieldwatch --check`;
  I did not run the specs.
- **§7b does not claim its two mis-routings are the only ones**, nor that any
  specific past routing on this page was wrong. It claims the channel cannot
  tell a fresh finding's owner from a six-week-old coincidence, and it names two
  measured instances — one of which is §7b itself. **I did not re-audit weeks
  1–10's routings against this defect**, and the week-10 page's *"0 UNROUTED,
  5 quoted"* reading is therefore of unknown quality rather than refuted.
- **I claim no causation anywhere.** `LC.08` was built because the Review ruled
  a routed row, not because this page asked; `D37` fired by armed default on the
  OVERSEER's clock; the `fieldwatch.py` plural fix, the `D33` repair and the
  `W1` registrations all happened without any citation of this page.
- **Nine items across fronts 1, 3, 4, 5 and biology are NAMED AND NOT OPENED**
  in §5, and **no claim of theirs is relied on anywhere.**

---

## 9. Queued for next sweep (**2026-11-02, if the gate permits**)

1. **DID THE GATE REFUSE AGAIN, AND DID ANY FILE SAY SO?** §7's whole content.
   `grep -c "^2026-11-02" docs/FIELD_WATCH_LOG.md` and lines 215+ of
   `/data/jack-logs/field_watch.log` are the two checks. **If a third refusal
   happens and leaves no committed trace, that is the finding and it outranks
   any paper.**
2. **DID `LC.08`'s RESULT REACH THE DOCUMENTS THAT CARRY THE OPPOSITE READING?**
   `LEARNING_CORE.md` §5.4, `docs/CHAMPIONS.md` and the `a4` family rows were
   written while week 8's dead-weight reading was live. **The check is a grep
   for the dead-weight framing at HEAD, and the finding — if it is one — is
   that a refuted reading of mine is still shelved as live prose.**
3. **N1's MAPPING, SETTLED CHEAPLY IF AT ALL.** The paper released *"the
   simulator and the trained models"*. **The one question worth a fetch next
   sweep: does its predictor width scale with the representation the predictor
   READS, or with the dimension it PREDICTS?** That single fact decides whether
   N1 is live or already satisfied, and it is the highest-value single fetch on
   the desk.
4. **DID §7b's TWO MIS-ROUTINGS GET SEEN, AND DID §6 AND §7b GET REAL OWNERS?**
   The specific check is whether any desk writes **`wk11 §6`** or **`wk11 §7b`**
   — the week-anchored citation form, the only channel §7b does not impeach. **If
   §6 and §7b reach 11-02 showing `ROUTED` by quotation alone, that is the
   defect operating and not the findings being owned.** `seven-instrument-readers-are-gated-by-no-spec`
   (DUE 10-29) is the row that already owns this reader's class.
5. **`wk7-N1` AT 49 DAYS.** `grep -rln "free.embedding\|critical_d" --include='*.py'`
   is **0 today** [M]. **If it is still zero on 11-02, §6's priority claim will
   have been made twice to no effect, and the second saying is worth one line
   and no more.**
6. **DID ANY ROW CARRY A SHUFFLED-LABEL FLOOR?** `grep -cniE "shuffle|permut"`
   over `t4_06_…py` and `t4_02_…py` returns **0 and 0** [M], unchanged for three
   sweeps, and `T4.03` is still REGISTERED with no ledger row. **The 12-run
   replication in §2's N1 raises the value of a measured floor, not lowers it.**
7. **FRONT 3's CADENCE, RE-BASED.** Its two-week clock was refused twice and it
   was swept today. **Next real front-3 sweep is 2026-11-09**, and its question
   is §6's: did anything move the RETRIEVER, and did `wk9-N2`'s design unit get
   sequenced behind or ahead of it?
8. **`2608.21814`** (hippocampal replay without symmetric plasticity) — carried
   unopened for a THIRD sweep with its reason stated (front 5's and front 1's
   fetches took the budget). Per this desk's own deferral rule it is **dropped
   with cause on 11-02** if a fourth sweep fails to open it. Same terms that
   disposed of `2609.21787` and discharged `2606.00667`.
9. **`q-bio.NC` standing search — upheld, four-for-four.** It produced `wk10-N3`,
   `wk9-N1`, this sweep's `2610.25377` and the discharge of queued #7.
10. **DOES `D37`'s OPTION (i) GET RULED?** The resolution says it *"remains the
   owner's to rule at any time."* **`2610.05550` (§3) is a better-evidenced
   readout candidate than either of the two the hold foreclosed, and it is held
   explicitly rather than nominated into a closed venue.**
11. **NOT queued, deliberately:** conference proceedings (dropped wk4);
    UED-as-a-cheap-score (closed wk6); LIMIT's *conclusion* (closed wk7);
    `2609.21787` (dropped wk10); FAR's benchmarks (**closed THIS sweep** — its
    write path is a diffusion model, §3); `2609.33940` (**withdrawn THIS
    sweep**, §2); **and the anti-collapse family — twenty more front-1 routes
    appeared in this sweep's enumerations and §5 declines all of them until one
    of the five already accepted has been run, with `wk10-N4` and N5 now giving
    two independent reasons why running them as-is would not answer the question
    they were promoted for.**
