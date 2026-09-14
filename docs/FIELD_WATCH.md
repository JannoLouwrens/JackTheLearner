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

**Sweep date:** 2026-09-14 · **Window:** ~2026-03 → 2026-09 (6 months)
**Scout:** field watch, week 7. **Seven days since week 6** (2026-09-07), the
embargo (*"not before ~2026-09-14"*) spent to the day. Third consecutive sweep
on the intended cadence. **Front 3 returns** on its endorsed two-week cadence,
with the single question week 6 queued for it.

**Confidence markers:** **[V]** fetched and read · **[c]** claimed by the authors,
not checked against their table · **[s]** asserted by a search engine *about* a
paper I have not opened · **[C]** computed by me, arithmetic shown · **[M]**
measured here, on this box, command shown.

---

## 0. WHAT MOVED IN THE INTERVAL — and the week the front-3 question answered itself

290 commits landed in seven days. Four facts re-point this sweep, and the first
one is about my own queue.

**(1) THE FRONT-3 QUESTION I QUEUED FOR THIS WEEK WAS ANSWERED FROM OUR SIDE
BEFORE I COULD ASK IT — AND THE ANSWER WAS NOT A CALIBRATION.** Week 6 queued
one question for today: *has anything published a calibrated abstention floor
for extractive retrieval on a personal corpus?* It was pointed at `ME.1`'s
`distractor_abstention` **0.0000**. That scar was already repaired when week 6
wrote it down: `ME.1` attempt 8 is **PASS** (`80f8c80`, **2026-09-06**, one day
before the sweep that reported the FAIL), `distractor_abstention` 0.0000 →
**1.0000**, `fabricated_abstention` 1.0, and `cued_recall` **0.85 ± 0.0136 —
byte-identical to the failing attempt** [M]. The number week 6 quoted was true
of an earlier attempt and stale by a day; that is mine to own and it is owned
here rather than in a footnote.

**What the repair actually was matters more than the timing, and it makes the
literature question sharper rather than moot.** The builder did not calibrate
the floor. It ran a bakeoff and measured that no calibration exists: on the same
store, all three seeds, `bestcov` for cues that MUST ABSTAIN is **0.667 exactly**
and for cues that MUST ANSWER is **0.400 exactly** — **gap −0.267, overlap
1.000**, the two populations separating in the *wrong order* on the only
statistic any floor in this family can see. `EpisodicMemory.py` was not touched;
OR-intent moved to the call site (the contract split, arm A5). So the question
"is there a published calibrated floor" is settled NEGATIVE on *our* scorer by
*our* arithmetic, and what survives is the harder half: **`ME.11`'s semantic arms
are still measured INFEASIBLE**, and §2's N1 is aimed there.

**(2) `LC.07` IS VENUE-UNAFFORDABLE AT BOTH VENUES, WHICH KILLS THE CHEAP ROUTE
FOR MY OWN LEAD NOMINATION OF LAST WEEK.** `lc07-checkpoint-branch` was ACTED
2026-09-13 (`a3a090a`): the CPU venue prices at **535.5 core-hours**, venue ratio
1.0, **33.5 days of the whole CPU budget**, largest run 3.0× `WORST_LEGAL_CHILD_S`;
the GPU branch was already refused on 09-06. Week 6 costed its Context-Collapse
diagnostic as *"(a) bolted to `LC.07`: zero marginal compute"*. **Route (a) no
longer exists.** The nomination now costs its standalone ~14.4 core-h for 3
seeds or it costs nothing because it does not run. §4 carries the correction.

**(3) TWO SEATS MOVED BY MEASUREMENT IN ONE DAY, IN OPPOSITE DIRECTIONS.**
`LG.13` PASS attempt 1 — the Language-routing seat is **FILLED BY VERDICT**
(meaning-mass, 1.0000 on every seed and both mouths). `SO.10` FAIL attempt 1 —
the Person-model seat is **VACANT BY MEASUREMENT**, its own race's winner unable
to hold it. Neither bears on a nomination here; both are recorded because a
scout that only notices the rows it wrote about is not watching the system.

**(4) THE 09-13 PILE DID NOT CLEAR, AND TWO ROWS THIS FILE WAS WRITTEN FOR ARE
NOW OVERDUE.** `t402-touch-drowns-audio-at-the-fusion-boundary` (DUE 09-13) is
still **OPEN** with no disposition text, and `w1-world-edit-window` (DUE 09-13)
is **OPEN** — the Sunday FULL sat at 06:37 on 09-13 and did not take up W1.
`ub10-seed-fragility-and-saturated-battery` DID land (DISPOSITIONED 09-08:
harden the task, adopt the per-arm stability conjunct, REFUSE the seed-level
`SCORED-AND-INELIGIBLE` retirement as a weakening). So of the three decisions
week 6 wrote cautions for, one ruled and two aged. **Week 6 predicted exactly
this** (*"ten live rows share 09-13 against a measured capacity of one per
cycle"*), and the consequence it named has happened: N2 and N3 of last week are
still waiting for the decisions they were written for. They are carried in §4
unchanged rather than re-argued, because re-arguing a live caution is padding.

---

## 1. Coverage — what was actually searched, so the gaps are visible

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **3 · MEMORY** *(returns on cadence — the primary front this sweep)* | calibrated abstention on the retrieval side; conformal/selective retrieval; out-of-corpus & unanswerable-query detection; score separability on redundant corpora; episodic/lifelog retrieval; embedding-capacity limits | **5 searches**; **2 full HTML** (LIMIT 2508.21038, ActSWM); **4 abstracts** (RARE, Argus Eyes, CoDeR, + checks) |
| **1 · LEARNING CORES** | action-sensitivity of latent dynamics (wk6-N1's follow-through); non-pixel action-conditioned JEPA; queued #4 | 1 search; **full HTML for ActSWM (queued #4, CLOSED POSITIVE)**; abs for 2608.29434 |
| **4 · CURIOSITY & OPEN-ENDEDNESS** | intrinsic motivation with a **body under homeostatic drive** — the exact gap three sweeps named | 1 search; **2 full HTML passes on 2506.00138**; code repo confirmed |
| **5 · WORLDS & EMBODIMENT** | survival sims; irreversibility/traps/delayed consequence (what `W1.01`/`W1.03` need) | 1 search, **nothing** |
| **2 · MULTIMODAL FUSION** | whether the balancing family has left supervised classification; embodied/world-model fusion imbalance | 1 search, **nothing new** |
| Small-model end | sub-1M-parameter embodied control | 1 search, **nothing in window** — **fifth** consecutive sweep |
| Biology-as-oracle | intrinsic drive validated against real neural data | folded into front 4; **N2 is this sweep's biology entry** |
| Our own artifacts | `experiments/ledger.json` `LC.03`/`ME.1`/`ME.11.*` rows; `experiments/cores.py`; a repo-wide grep for the A4 diagnostic; Clopper–Pearson on the shipped abstention conjunct | **[M]/[C] — commands in §6** |

**Known gaps, stated so nobody assumes coverage:**

- **THE arXiv API IS CLOSED TO THIS BOX THIS SWEEP — HTTP 429 on every
  attempt, from two independent network paths.** Weeks 4–6 each ran
  **40-entry arXiv-API enumerations** per front; this sweep ran **none**. Six
  attempts (`curl` over http and https, three retries with backoff, plus
  `WebFetch` against the same endpoint) returned `429 Rate exceeded` or timed
  out. Coverage this week is **keyword search plus targeted fetch only**, which
  is strictly weaker: enumeration is what catches the paper whose title misses
  our vocabulary. This is the **second** infrastructure closure on this desk
  after week 4's OpenReview `403`, and unlike that one it is probably
  transient — it is re-queued as a first-thing check, not dropped.
- **Fronts 2 and 5 got one search each.** Front 3's return consumed the budget
  those two had borrowed for two sweeps. Both are reported as NO-ACTION in §5
  and the shallowness is stated there, not hidden.
- **N2's paper reports no numbers at all** — its comparative claims live in
  figures. §2 leads with that.
- **N2 is OUT OF WINDOW** (v1 2025-05-30, v2 2025-10-24). Stated in its heading.
- **2608.29434 is abstract-level only**; no metric, no numbers, no hardware.
- No non-English sources. No conference main-track enumeration (dropped
  permanently, week 4; still dropped, still an acknowledged gap).

---

## 2. NOMINATIONS

Three. **One is a cheap CPU probe on the memory front that decides whether a
better retriever is worth building at all; one is an arm for the front that has
been empty for three sweeps, and it is out of window; one is not a new paper at
all but the missing formula that last week's lead nomination was blocked on.**
Each states its arXiv primary category, its evidence class, its cost on **our**
substrate, and both sides steelmanned.

---

### N1 — the free-embedding critical-dimension probe: a CPU-minutes pre-gate that tells `ME.11`'s successor whether a better encoder can exist, before anyone builds one

**Source:** *On the Theoretical Limitations of Embedding-Based Retrieval* —
[arXiv:2508.21038](https://arxiv.org/abs/2508.21038), primary category **cs.IR**,
v1 2025-08-28, **v2 2026-03-12**, **ICLR 2026**. (Author list not extracted from
the pages I read; the code repository is `google-deepmind/limit`, so the lab is
on the record even where the names are not.) **Full HTML read. Code and data
released: `github.com/google-deepmind/limit`.** Not cited anywhere in this repo.

**I am importing the INSTRUMENT and refusing the CONCLUSION, and §5 shows the
arithmetic for the refusal.** That distinction is the whole entry, so it goes
first.

**The verified claim [V], quoted.** The theorem: *"if every k-subset S ⊆ [n] is
realized with margin γ, then (n choose k) ≤ (1 + 1/γ)^d, hence
d ≥ log(n choose k) / log(1 + 1/γ)"*. The empirical instrument is the part I
want — the authors **optimise the embeddings directly on the test data, with no
encoder at all** ("free embedding"), and sweep `d` to find where the task first
becomes representable. Their finding is that *"even with unrestricted
optimization on test data, embedding dimension constrains performance"*.
LIMIT-small numbers, from their table:

| on LIMIT small (46 docs, 1000 queries, k=2) | recall@2 |
|---|---|
| BM25 (sparse) | **85.7 %** |
| best single-vector (Promptriever @ 4096d) | 54.3 % |
| GTE-ModernColBERT (multi-vector) | 23.1 % |
| Gemini Embed @ 1024d | 1.3 % |

Hardware: A100 for inference, H100 + TPU v5 for the free-embedding sweep. **No
latency figures anywhere.**

**Why this lands on us, and on what.** `ME.11`'s whole family measured
`feasible_ok` **0.0** — `τ_fpr` **0.388** > `τ_cov` **0.227** on the MiniLM arm
[M], the same INFEASIBLE branch on the static-embedding arm. Our own
`MEMORY_RETRIEVAL_BAKEOFF.md` §1.8 pre-committed the response: *"An infeasible
arm is a result, not a bug, and the correct response is a better score function,
never a split-the-difference threshold."* **Nobody knows whether a better score
function in that family can exist on our fixture.** Four arms read INFEASIBLE
and there is no ceiling measurement to say whether the fifth would too.

**The probe, stated so it is pre-registerable.** On the frozen `ME.11` fixture —
**5,000 events, 160 headline cues, 300 tune + 300 certify negatives,
`max_gold_size` 2, `oracle_ceiling` 1.0**, hash `9c915329f4755c3e` [M] — drop the
encoder. Optimise event vectors `E ∈ ℝ^{5000×d}` and cue vectors directly against
the fixture's own labels, sweeping `d ∈ {8, 16, 32, 64, 128, 256, 384}`, and at
each `d` report the same two order statistics the arms report: **is
`τ_fpr ≤ τ_cov` achievable at all?** Report the *critical d* at which feasibility
first appears.

> **The must-fail control, and it is the load-bearing half.** Run the identical
> optimisation with the cue→gold assignment **shuffled**. With 5,000 free event
> vectors against 160 constraints, a free embedding can fit noise, and a probe
> that reaches feasibility on shuffled labels at the same `d` is measuring its own
> capacity and nothing else. The reading is only admissible where the true-label
> critical `d` is **below** the shuffled-label critical `d`. This is the one
> objection I would press hardest against my own nomination and it is why the
> sweep is over `d` rather than a single fit — LIMIT's own method, for LIMIT's
> own reason.

**The pre-registered prediction, offered so it can be wrong.** §5's arithmetic
says the dimension is not our constraint by a wide margin, so I predict critical
`d` lands **well under 64**, far below the 256d and 384d the arms already use —
which would mean the INFEASIBLE readings are the **encoder's map**, not the
geometry, and a better encoder arm is worth building. If instead critical `d`
exceeds 256, my §5 arithmetic is wrong, the cosine-on-single-vector family is
exhausted for this corpus, and the redesign has to go to multi-vector,
cross-encoder or structured retrieval — all three of which `§1.6`/`§1.10` have
already priced on this box (cascade rerank **329 ms/query** int8, ColBERT
excluded at **460 MB** index on 929 MB free). **Either answer routes the
redesign; today nobody knows which.**

**Cost on our substrate.** NumPy on 4 ARM cores at `nice 19`. The optimisation is
`d × (5000 + 760)` free parameters under a margin loss over 160 positives and 600
negatives, a few hundred steps, seven values of `d`, plus the shuffled twin —
**single-digit CPU-minutes, no GPU, no new weights on disk.** This is the
cheapest nomination since week 2's whiff clock and materially cheaper than
anything on the desk except a zero-cost caution.

**Why it might WIN (falsifiable).** It converts "every arm we tried was
infeasible" into "feasibility is / is not reachable at our `d` on this fixture",
which is the difference between a dead end and a queue of encoder arms. It is
cheap, it has a must-fail control that can kill it, and it reads a fixture that
already exists and is hash-frozen.

**Why it might LOSE (steelmanned). Five.**
1. **The free embedding may be vacuous at every `d` we care about.** 5,000 free
   vectors against 160 labelled cues is an enormously over-parameterised fit; if
   the shuffled control also reaches feasibility at `d = 8`, the probe returns
   nothing and the CPU-minutes are spent. **Lead objection.**
2. **LIMIT's own regime is not ours and I am only borrowing the method.** Their
   failure is combinatorial top-k coverage on a 46-document corpus whose 1000
   queries demand nearly every 2-subset; ours is a lexical-semantic gap on a
   5,000-event diary whose 160 cues demand at most 160 subsets. §5 computes the
   distance. A borrowed instrument is still an analogy until it runs.
3. **A ceiling is not a method.** Even a perfect answer ("feasible at d = 16")
   does not say *which* encoder reaches it. It routes work; it does not do work.
4. **The fixture's negatives may be the thing being measured.** If `τ_fpr` is
   high because our 300 tune negatives are adversarially near-duplicate by
   construction (§2.3's three grades), the probe measures the eval set's
   difficulty, not the retriever's ceiling — and it cannot distinguish those two
   without a second fixture we do not have.
5. **Their numbers come from A100/H100/TPU v5 and report no latency at all.**
   `LESSONS.md` records three transfer failures on this box for exactly that
   reason. Here it bites less than usual, because what I am importing is a
   procedure rather than a number — but it is why the cost above is computed
   from our own fixture sizes and not from theirs.

---

### N2 — the first embodied, homeostatic intrinsic-motivation result this desk has found in four sweeps, and it is OUT OF WINDOW with no numbers

**Source:** *Intrinsic Goals for Autonomous Agents: Model-Based Exploration in
Virtual Zebrafish Predicts Ethological Behavior and Whole-Brain Dynamics* —
[arXiv:2506.00138](https://arxiv.org/abs/2506.00138), primary category
**q-bio.NC**, v1 **2025-05-30**, v2 **2025-10-24**. Keller, Kirsch, Pei, Pitkow,
Kozachkov, Nayebi. **Full HTML read, twice. Code released:
`github.com/neuroagents-lab/autonomous_zebrafish`; dataset stated open.**

**The date is the first objection and it goes in the heading rather than the
footnotes.** v2 is 2025-10-24; the sweep window is 2026-03 → 2026-09. This is
**out of window by roughly five months**, and a search snippet that called it
*"a 2026 paper"* is [s], not [V] — §7 records that.

**Why I am nominating it anyway.** Fronts 4 and 5 have returned **no arm for
four consecutive sweeps**, and weeks 5 and 6 both named the same reason in the
same words: of 40 enumerated intrinsic-motivation entries, *"zero have a body
under homeostatic drive."* **This has one.** A custom MuJoCo environment with a
**6-link zebrafish body**, dynamic fluid forces, **5-DoF motor torques**,
egocentric vision plus proprioceptive state [V] — our regime, not a gridworld,
which is the objection that killed PRIME, Curiosity-Critic and CIG.

**The verified mechanism [V], quoted.** 3M-Progress (*model-memory-mismatch*) is
a KL between an online world model and a **fixed prior world model learned from
the ecological niche**:

```
ε_t  = D_KL[ ω_θ(φ(s_{t+1}) | φ(s_t), a_t) ‖ ω_θ′(φ(s_{t+1}) | φ(s_t), a_t) ]
ε̂_t = (1 − γ)·ε̂_{t−1} + γ·ε_t            (exponential filter)
r^i_t = | ε̂_t − ε_t |                     (intrinsic reward)
```

Baselines it is run against [V]: **ICM, RND, Disagreement, γ-Progress** (model-
based); **a homeostatic agent, a maximum-entropy agent, and a random agent**
(model-free); plus data-derived controls (Gaussian process, PID, population
average, white noise).

**Which spec it would enter.** `CURIOSITY_BAKEOFF.md`, as an arm beside
`disagree` / `lp` / `metra` / `vlm-lp`. **It is not a re-skin of `lp`**, and that
is the point of nominating it: learning progress measures improvement against
*your own past model*, while this measures divergence from a **fixed prior** —
a different signal with a different failure mode. A4 already carries an EMA
target encoder, so the machinery for a second frozen world model exists; the arm
is a second copy of a network we build anyway plus a KL and an EMA. **Zero new
architecture, two constants.**

**And it is the biology-oracle entry for this sweep, in the strongest form
`GOAL.md` asks for.** The arm is selected by *matching whole-brain neural and
astrocytic dynamics in a real animal* (11 zebrafish subjects), not by a
leaderboard. `GOAL.md`: *"When stuck, ask how nature solved it… Nature's solution
enters as a bakeoff arm and must win on our substrate like any other."* This is
that sentence's literal case.

**Cost on our substrate.** A second frozen copy of the world model and a KL per
step. On `wm-latent` (**861,545 params [M]**) the prior copy roughly doubles
inference memory for the core and adds one forward pass per decision. Against
`LC.03` v2's measured **17,280.3 process-seconds per arm-seed** [M] a rough upper
bound is **~2× that arm's core time**, i.e. ~9.6 core-h/seed if it entered at the
same envelope — **not cheap and not priced by the paper**, which reports no
wall-clock at all.

**Why it might WIN (falsifiable).** `CURIOSITY_BAKEOFF`'s live problem is that
`T3.06` VOIDed and week 5 measured a **random-action policy covering W0 as well
as the curious arm (t = 0.39)**. An intrinsic signal that is *anchored to a fixed
niche prior* rather than to the agent's own improvement is the one family in the
enumeration whose reward does not go to zero just because the agent stopped
improving — which is precisely the failure a saturating world makes invisible.
And its baseline set already contains **a homeostatic agent and a random agent**,
so it has been run against our two most important comparators by someone else.

**Why it might LOSE (steelmanned). Six.**
1. **THERE ARE NO NUMBERS.** Two full HTML passes returned no table: the
   comparative claims are *"3M-Progress agents captured nearly all of the
   explainable variance"* and *"exhibited the highest model-behavior alignment"*,
   with the evidence in figures. **No p-values, no error bars, no confidence
   intervals, no seed count for the agents** — 11 zebrafish subjects is the only
   replicate number stated. Against `UB` §1.8's Agarwal standard this is the
   weakest evidence class this desk has nominated on.
2. **Its selection criterion is not ours.** The paper's winner is the agent that
   best *resembles a fish brain*. It never shows 3M-Progress agents explore or
   survive better. `GOAL.md` is explicit that biology is the ORACLE and not the
   blueprint — *"planes do not flap"* — so an arm that matches neural data and
   loses on `life_gain` must lose here, and nothing in this paper says it would
   not.
3. **The "ecological niche prior" has no obvious source in W0.** In the fish it
   is the evolved model. Jack has no separate niche corpus to pre-train on. The
   tempting reading — *the model at death of life N is the prior for life N+1*,
   which would make `GOAL.md`'s "death is a page turn" into a curiosity signal —
   is **my story, not the paper's**, and week 3's lesson is that the dangerous
   story is the one the scout writes itself. It is written here as an open design
   question and must not be cited as the paper's proposal.
4. **Out of window by five months**, and it surfaced only because I searched
   q-bio this week (§7).
5. **It doubles the core's forward cost** on a box with no headroom, for an arm
   whose benefit is unquantified (objection 1).
6. **No hardware, no wall-clock, no parameter count.** Fourth consecutive sweep
   in which that is true of the lead nomination class, and it is now a
   structural property of this desk's intake rather than a caveat.

---

### N3 — NOT A NEW PAPER: the formula week 6's lead nomination was blocked on has been found, and it differs from what I proposed in one way that matters

**Source:** ActSWM — [arXiv:2607.26712](https://arxiv.org/abs/2607.26712), cs.RO,
2026-07-29 (rev 08-15). **Full HTML read this sweep. Queued #4 — CLOSED
POSITIVE.**

Week 6 nominated a Context-Collapse diagnostic on `A4` and named its own weakest
link: *"the only paper claiming to measure Context Collapse gives no formula on
its abs page… one HTML fetch decides whether the rollout-gap readout has a
published definition to borrow or whether we are inventing it."* **It has one.**

**The definition [V], quoted.**

```
Δ_k = s_k^gt − s_k^0      where   s_k^gt = cos( ẑ_{t+k}^gt , z_{t+k} )
                                  s_k^0  = cos( ẑ_{t+k}^0  , z_{t+k} )
```

— the latent rollout is run twice from the same context, once under the
**recorded actions** and once under an **all-zero action sequence**, and both are
compared by cosine against the true future latent. *"A larger gap indicates
stronger action sensitivity and absence of context collapse."* The second
readout, local action recovery, is CEM-based: *"Given context and a target frame,
CEM searches for an action sequence whose rollout approaches the target"*, scored
as `Gap = J_rand − J_cem` plus per-dimension key accuracy over 14 control
dimensions.

**THE ONE WAY IT DIFFERS FROM WHAT I PROPOSED, AND IT IS AN IMPROVEMENT.** Week 6
specified *"roll the latent predictor H steps under two different action
sequences"* — an arbitrary contrast with no canonical baseline. ActSWM's baseline
is the **all-zero action sequence**, which in W0 is not arbitrary at all: **zero
torque is a legal, executable action**, so `s_k^0` is the model's prediction of
"what happens if I do nothing" and `Δ_k` is literally *how much this model thinks
its actions matter*. That is a better-posed quantity than a random-pair contrast
and it removes a free choice from the readout. **Adopt ActSWM's form, not mine.**

**What is still missing and is not repaired by this fetch.** The paper reports
**no hardware, no wall-clock, no parameter counts, no seed count, and no code**
[V, checked explicitly]. So the *definition* transfers; nothing about its cost
does, and §6's `LC.03` process-seconds remain the only honest price.

**Cost, CORRECTED from last week.** Week 6 priced route (a) at **zero** by
bolting the readout to `LC.07`. **`LC.07` is now VENUE-UNAFFORDABLE at both
venues** (§0 fact 2). The only remaining route is standalone: **~4.8 core-h per
seed → 14.4 core-h for 3 seeds** [M, from `LC.03` v2's recorded 17,280.3
process-seconds/arm-seed], inside `D4`'s frozen `CPU_DAYS` cap, no GPU quota.
**There is no free version of this nomination and I will not price it as one
twice.**

**And §6 changes what "beside effective rank" means.** Week 6 proposed this as a
mandatory diagnostic *"beside effective rank"*. **There is no effective rank.**
§6 shows the A4 collapse diagnostic `LEARNING_CORE.md` calls mandatory is not
computed anywhere in this repository. That does not weaken N3 — it strengthens
the case for running *something* on that seat — but the sentence "beside" was
false when I wrote it and it is corrected here.

**Why it might LOSE (steelmanned).** Everything week 6 listed still stands and is
not repeated: pixel regimes, Delta-JEPA never measuring the insensitivity it
names, no hardware in any of the three sources, and a green reading being the
likely outcome. **One objection is now sharper:** `Δ_k` requires the *true future
latent* `z_{t+k}` from a rollout of the real environment alongside the imagined
one, so the readout needs a trained `A4` **and** matched environment rollouts —
which is why the 14.4 core-h is the honest number and not a linear-probe's
seconds.

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **2608.29434** — *Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations* ([abs](https://arxiv.org/abs/2608.29434), 2026-08-29, Oberweger & Schwingshackl) | cs.LG | A **fourth** independent group on wk6-N1's theme, and the **first non-pixel one** — it lifts three JEPA designs (frozen-encoder, distribution-prior, action-sensitive) to point clouds and reports the action-sensitive variant best *"in scenarios with maximum geometric movement"*, with object positions *"almost perfectly linearly decodable"* [c]. | **Numbers, and a state-vector result.** Point clouds are non-pixel but they are not a 6-dim drive vector; the move from images to geometry is one step of the two we need. Abs-level only: **no metric definitions, no hardware, no params, no seeds, no code.** It materially softens wk6-N1's lead objection ("all three live in pixel regimes") without removing it, and that is exactly its current weight — no more. |
| **RARE / RedQA** ([arXiv:2604.19047](https://arxiv.org/abs/2604.19047), 2026-04-21, rev 06-30, Cho & Lee) | cs.CL | Redundancy-aware retrieval evaluation for **high-similarity corpora** — the geometry `MEMORY_RETRIEVAL_BAKEOFF` Finding 3 names for a diary (*"internally redundant and externally sparse"*). Measured [V]: a strong retriever falls from **66.4 % PerfRecall@10** on 4-hop General-Wiki to **5.0–27.9 %** on Finance/Legal/Patent at 4-hop depth. | **A non-generative construction path.** Its redundancy tracking is *"decomposing documents into atomic facts"* via **LLM-based data generation** (plus CRRF rank fusion), which puts a generator in the **write** path — the same objection that has held Eywa and ScrubJay-MEM. And it treats abstention **not at all** [V, checked]. Its portable half is the measured drop, recorded here so a future sweep does not re-find it as an arm. |
| **CoDeR** ([arXiv:2606.13204](https://arxiv.org/abs/2606.13204), 2026-06-11, Yin, Tang, Du) | cs.IR | Constraint-sensitive retrieval that produces *"a ranked document list without external Large Language Model (LLM) calls at inference time"* [V] — genuinely extractive-compatible, which almost nothing on front 3 is. | **Evidence that its failure axis is ours.** Its target is documents *topically close but supporting the opposite constraint direction*; `ME.11`'s failure is a paraphrase gap with lexically disjoint cues. Different axis, and the paper reports **no CPU cost, no latency, no model size, no seeds, no code**, so it cannot inform §1.9's table either. |
| **Argus Eyes** ([arXiv:2602.09616](https://arxiv.org/abs/2602.09616), 2026-02-10, rev 07-15, Taghavi, Modarressi, Schütze, Marfurt) | cs.IR | Retrieval "blind spots" scored by a **Retrieval Probability Score** predicted from entity-embedding geometry — the right *shape* for a per-query abstention signal. | **Confirmation it is LLM-free at query time, and any cost number.** The abstract does not establish either; 8 pages, **no hardware, latency, seeds or code** in the metadata read. |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | Unchanged, fourth sweep. Rollout-horizon deterioration in the **fully observable state space** — our regime. Still no compute cost, no hardware, no environment size, no code. **Promote on: any statement that a domain runs on CPU.** Its relevance is now concrete: N3's `Δ_k` is a rollout-horizon quantity and this paper measures rollout-horizon decay on our observation type. |
| **ForageWorld** ([arXiv:2506.06981](https://arxiv.org/abs/2506.06981)) | cs.AI | **Kept, and its moment has slipped with the window.** `w1-world-edit-window` went overdue on 09-13 (§0 fact 4), so `W1.01`/`W1.03` are still unregistered and this remains the closest published existence proof of a world with depleting/diffusing food, pursuing predators and an energy-gated immobilising sleep action. **Design reference only; it is not an arm and never was.** |
| **Eywa** ([arXiv:2605.30771](https://arxiv.org/abs/2605.30771)) | cs.CL | Re-examined on front 3's return and **unchanged**. Still the only front-3 item with *"zero LLM calls inside retrieval"*; still blocked on the **write** path being generative; still no hardware and no latency, so it cannot inform §1.9's CPU table. Nothing this sweep resolved it. |
| **ScrubJay-MEM** ([arXiv:2608.04746](https://arxiv.org/abs/2608.04746)) | cs.CL | Unchanged. The **oracle** remains the portable half — perishability is a property of the *remembered thing*, and a hand-specified type→decay map needs no LLM — routed at `Forgetting.py`. |
| **PRIME** ([arXiv:2607.16858](https://arxiv.org/abs/2607.16858)) | cs.LG | Unchanged, and **N2 is the comparison that clarifies it**: PRIME has 15 seeds and two 10×10 grids; N2 has a MuJoCo body and no seeds at all. Neither is nominatable on its own strengths, and they fail in opposite directions — which is week 5's theory-vs-empirical lesson recurring inside one front. PRIME stays out: its epistemic term is a **pseudocount**, free on a grid and undefined on a continuous body state. |
| **IIBalance** ([arXiv:2603.17347](https://arxiv.org/abs/2603.17347)) | cs.MM | Unchanged. Capacity-based budgets rather than forced equality. Still classification, still no extractable numbers. |
| **Curiosity-Critic** ([arXiv:2604.18701](https://arxiv.org/abs/2604.18701)) | cs.LG | Unchanged. Weaker-evidenced sibling of LPM (2509.25438), already cited by `CURIOSITY_BAKEOFF.md`. |
| **POBAX** ([arXiv:2508.00046](https://arxiv.org/abs/2508.00046)) | cs.LG | Unchanged and **still not an instrument we lack** — `W1.01` carries a deliberately-benign twin world that must show NO gap and POBAX has no counterpart. Recorded once more only because `W1.01` is still unregistered. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS

| nomination | entered | status now |
|---|---|---|
| **wk6-N1 · Context Collapse → `A4` diagnostic** | `LEARNING_CORE` §5.4 | **LIVE, and materially changed on both sides this week.** Its readout now has a **published definition** (§2 N3) and its **cheap route is dead** (`LC.07` VENUE-UNAFFORDABLE, §0 fact 2), so the honest price is **14.4 core-h for 3 seeds** [M]. Its framing "beside effective rank" is **false** and §6 says why. |
| **wk6-N2 · MULTIBENCH++ redundancy pre-gate** | `ub10` row, DUE 09-08 | **ARRIVED IN TIME, and the row ruled without it.** `ub10` DISPOSITIONED 2026-09-08: harden the task, adopt the per-arm stability conjunct, refuse the seed-level retirement as a weakening. **All three address saturation; redundancy is not addressed**, which is the caution's content. It stands as a pre-registration on the hardened battery's *next* reading, not as an unmet ask. |
| **wk6-N3 · IAF pathway-decoding control** | `t402` row, DUE 09-13 | **STILL WAITING — `t402` is OPEN and now OVERDUE** (§0 fact 4). Carried unchanged. The control (every balancing arm reports per-modality pathway decoding before and after fusion) costs zero and is not re-argued here. |
| **wk5-N1 · SIGReg vs VICReg (2607.13612)** | `LEARNING_CORE` §5.4 selection criterion | **THE "FREE HALF" IS NOT FREE AND NEVER WAS — see §6.** The Review ordered it FIRST on my framing that Remark 1's signature is *"checkable on curves we already have at zero compute"*. **There are no such curves.** The effective-rank/variance diagnostic is declared mandatory and is not computed anywhere in the repo. The item is not refused; it is **re-priced from zero to a diagnostic that must first be built**. |
| **wk5-N2 · prosociality by coupling (2604.10760)** | `NE.07` | **Unchanged.** Arm DEFERRED, shuffled-partner CONTROL accepted. `NE.07` and `NE.02` have still never run. |
| **wk5-N3 · RWG/PIC/POIC invert** | binding control on `W0.DIAG` | **Discharged** (cleared at t ≈ 11, week 6). |
| **wk4-N1 · spectral-radius constraint (2607.19719)** | `A4` variant | **ACCEPTED, narrowed. Unrun.** |
| **wk4-N2 · PSG-JEPA (2608.06799)** | `A4` ×2 | **ACCEPTED as two arms.** Still sequenced behind `D9`'s PARK. |
| **wk4-N3 · infant motor noise** | `W0.DIAG` | **RUN AND PASSED.** Nothing further owed. |
| wk1 · anti-collapse regularisers → `A4b`/`A4c` | `A4` variants | **LIVE, unrun.** Four sweeps promoted without an experiment — and §6 means the incumbent they would replace has never been measured for the failure they prevent. |
| wk1 · certificate-gated identifiability → `UB.11` pre-gate | `UB.11` | **LIVE, unrun.** Still the only route to `UB.11`'s certificate. `UB.10`'s 09-08 ruling makes a `UB.10` verdict reachable, which unblocks it in principle. |
| wk1 · interoceptive precision (2608.04232) | `NE` §2.4b | **LIVE, unrun.** Still the cheapest item on the desk with released code. |
| wk1 · entity-collision protocol (2605.29630) | `MR` §2 | **LIVE, unrun** — and **N1 above is the better-posed version of the same instinct**: both ask whether a retrieval verdict is an artefact of its eval set rather than a property of the retriever. |
| wk2 · the whiff clock → `SM.02` | `SM.02` | **HELD, correctly — `SM.02` is PARKED.** |
| wk2 · RPE-prioritised replay → `NE.05` | `NE.05` | **LIVE, unrun.** |
| wk3 · CIG (2605.20878) → `A3` | `A3` | **Remains DEMOTED** (`wm-efe` t = 2.05). |
| wk3 · Optimistic World Models (2602.10044) → `A2` | `A2` | **Remains DEMOTED, hard** (`dreamer-xs` t = −0.94). |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

**FRONT 3 · MEMORY — THE QUEUED QUESTION IS ASKED AND ANSWERED *NO*, FOR THE
FIFTH CONSECUTIVE TIME, AND THE ONE THEORY RESULT THAT COULD HAVE EXPLAINED OUR
INFEASIBILITY DOES NOT APPLY — ARITHMETIC BELOW.**

*Has anything published a calibrated abstention floor for extractive retrieval on
a personal corpus?* **No.** Five searches returned Abstain-R1 (2604.17073),
Geometry-Calibrated Conformal Abstention (2604.27914), I-CALM (2604.03904), Two
Axes of LLM Abstention (2607.08456), Know Before You Fetch (2606.29959),
SURE-RAG (2605.03534) — **every one of them generator-side**: they threshold a
language model's own confidence, log-probability or self-report. The statistical
machinery they use (one-sided Clopper–Pearson, risk-controlled refusal) is the
machinery `MEMORY_RETRIEVAL_BAKEOFF` §1.8 already specifies, from the same
primary sources. **Four sweeps ago this desk wrote that our own bakeoff doc is
ahead of the field on abstention; it still is, and that is now a stable fact
about the literature rather than a weekly observation.**

**And the capacity theorem does not rescue us either. [C], and this is the most
useful thing front 3 produced.** It is tempting to read `ME.11`'s universal
INFEASIBLE as an instance of LIMIT's dimension bound — *the embeddings are too
small to represent what we are asking*. **They are not, by more than an order of
magnitude.** LIMIT's own theorem, `d ≥ log(n choose k) / log(1 + 1/γ)`, evaluated
on our fixture (`n = 5000` events, `k = 2` = `max_gold_size` [M]):

| reading | subsets that must be representable | γ = 1.0 | γ = 0.5 | γ = 0.1 |
|---|---|---|---|---|
| **all 2-subsets** (the theorem's literal form) | 12,497,500 | d ≥ **23.6** | d ≥ 14.9 | d ≥ 6.8 |
| **the 160 subsets our headline cues actually demand** | 160 | d ≥ **7.3** | d ≥ 4.6 | d ≥ 2.1 |

Our arms run at **256d** (potion-base-8M) and **384d** (MiniLM-L6). Against the
reading that matches our eval, the bound is slack by **35×**. Against the
harshest possible reading at maximum margin it is slack by **10.9×**. **The
dimension is not our constraint and the INFEASIBLE readings are not an instance
of this theorem.** The honest caveat, stated because the margin is not uniform:
the arm's smallest *variant* is potion-base-2M at **64d**, and 64 against 23.6 is
only **2.7×** — comfortable, but not a blowout, and that variant is the one to
watch if this ever matters. LIMIT's own empirical regime is the opposite of ours
by construction: **46 documents against 1000 queries demanding nearly every one
of the 1035 possible 2-subsets**, where ours is 160 cues against 12.5 million
possible subsets. Recorded so a sixth sweep does not chase it — and its
*instrument* is nominated in §2 precisely because its *conclusion* is refused.

**FRONT 2 · FUSION — no arm for the fourth consecutive sweep, and the ordered
bakeoff has not started.** One search this sweep (front 3's return took the
budget) and it returned the same family: tactile-fusion surveys, missing-modality
imitation learning, multi-objective sensor balancing — **still supervised
classification or imitation, still not one paper with a world-model objective and
five heterogeneous senses.** Third consecutive year. `t402` is OPEN and overdue,
so wk6-N3's control is still the whole of this desk's position and it is carried
in §4 rather than re-argued. **The shallowness of this front's coverage this week
is real and is stated rather than dressed up.**

**FRONT 5 · WORLDS — nothing, and the specs that needed something are still
unregistered.** One search on irreversibility, traps and delayed consequence
returned LLM-agent simulation benchmarks (SimVerity 2608.25067, EnvSimBench
2605.07247, AgentSim 2604.26653) and a particle-physics agent benchmark. **Week 6
already established there is no cheap published environment-discriminability
score**; nothing this week changes that, and `w1-world-edit-window` going overdue
means `W1.01`/`W1.03` are still ours to instrument. ForageWorld remains the only
design reference.

**FRONT 4 · CURIOSITY — N2 is the nomination, and the front is otherwise
unchanged.** The 2026 in-window intrinsic-motivation literature is where weeks 5
and 6 left it: LLM reasoning bonuses, multi-agent influence terms, autotelic
language agents. **The one embodied homeostatic result I found is out of window
and in q-bio**, which is §7's coverage finding, not a change in the front.

**SMALL-MODEL END — nothing in window, FIFTH consecutive sweep.** The search
returned DOOM-1.3M (2604.07385) again, rejected in week 3 on its own terms
(imitation from 31k human demos, not RL; no seed count; a 120B LLM at 31 ms per
decision as the comparison class), plus edge-deployment surveys of *foundation*
models. **Our own `ppo-needs` at 135,961 params [M] is still smaller than
anything this literature is proud of.**

---

## 6. A FINDING IN OUR OWN ARTIFACTS — `A4`'s mandatory collapse diagnostic is declared in the governing document and computed nowhere in the repository

Week 3's rule: a scout reading our own ledger has no abstract to doubt, so it
must carry the arithmetic. Here it is, reproducible in three commands.

**What the document promises.** `docs/research/LEARNING_CORE.md` §5.4, verbatim:

> *"Collapse is the failure mode and it is silent, so A4 carries a **mandatory
> diagnostic: effective rank and per-dimension variance of the latent must be
> reported every 1,000 decisions**, and a collapse (rank below a pre-registered
> floor) is `Status.VOID` for A4, not a good loss curve."*

**What exists.**

```
$ grep -rn "effective_rank\|effective rank" --include=*.py .
experiments/registry.py:892           "...per-layer effective rank every cycle. "
experiments/registry_expansion.py:4242 "...fraction and effective rank stay near their early-life "

$ grep -rn "svd\|singular\|RankMe\|np.linalg.eig\|spectrum" experiments/ --include=*.py
  (7 hits, every one about AUDIO spectrum in the HNS/PG.7 specs)

$ python3 -c "import json; print([k for k in json.load(open('experiments/ledger.json'))
      ['results']['LC.03']['metrics'] if 'wm-latent' in k])"
```

**Both Python hits are prose inside other specs' `hypothesis` strings** — one is
a plasticity spec, one is the sleep-downscaling spec. **Neither is an
implementation.** No effective rank, no singular spectrum, no per-dimension
latent variance is computed anywhere in `experiments/`.

And the committed `LC.03` row confirms it from the other direction. It records
**50 metrics for `wm-latent`** — `life_gain`, `twin_life_gain`, `wiped_life_gain`,
`chaos_occupancy`, `thrash_ratio`, `dwell_ok`, `action_entropy_final`,
`final_slope`, `params`, `decisions`, `optimiser_steps`, `train_ratio`, `clt`,
`lg_margin_null`, `lg_margin_twin`, … — and **not one of them is effective rank
or per-dimension variance**, for any of the five arms.

**Why this is a section and not a footnote. Three consequences, stated as facts
and not as instructions.**

1. **`A4`'s declared VOID condition was never computable.** The document says a
   rank below a pre-registered floor makes `A4` VOID. There is no rank, there is
   no floor, and `D10` seated `A4` **by verdict** on 2026-09-01. The seat's
   evidence (`life_gain` t_null 4.65 / t_twin 4.00) is real and I am not
   questioning it — but the **specific silent-failure guard the governing
   document promises for this exact arm does not exist**, and `SYSTEM.md` already
   wrote the rule that covers this: *"A governing document that names an
   enforcement is making a capability claim, and it is bound by law 1 like any
   other."* That sentence was written on 2026-08-30 about `decisions.py`. It
   applies here unchanged.
2. **wk5-N1's "free half" has nothing to read.** The Review ordered it FIRST,
   in its own words, because it was *"adjudicable on curves we already have, at
   zero compute"*. Those curves do not exist. The item is not wrong; it is
   **mispriced from zero to a diagnostic that must be built and a run that must
   be re-done**.
3. **wk6-N1 was pitched "beside effective rank".** There is no beside. §2's N3
   corrects the framing.

**AND THE FRAMING IN (2) WAS MINE.** Week 5's log entry says the Remark 1
signature is *"checkable on curves WE ALREADY HAVE at zero compute"*; the Review
adopted it *"exactly"*. I inherited *"A4's mandatory diagnostic already logs
effective rank + per-dimension variance every 1,000 decisions"* from
`LEARNING_CORE.md` and never ran the grep that falsifies it — the **same failure
as 08-24 and 09-07**, for the third time, and in the same quiet form: **a
governing document is a more trustworthy-looking source than a search engine, so
it is a cheaper place to be wrong from.** The grep costs one command. §8 makes it
standing.

**Nothing is decided here.** Whether the diagnostic gets built, whether `A4`'s
seat is re-examined, and whether `LEARNING_CORE.md` §5.4 or the implementation is
the thing that moves are all the builder's and the Review's. I report that the
two disagree.

### 6b — a second, smaller one, free: the abstention conjunct that just shipped is certified below its own bar

`MEMORY_RETRIEVAL_BAKEOFF` §1.8 fixes the arithmetic: a **perfect** abstention
run over `m` labelled negatives certifies only `a_L = γ^(1/m)` at confidence
`1 − γ`, so `m ≥ 59` is needed at γ = 0.05 to certify 0.95. The distractor
conjunct adopted on 2026-09-06 lands on five ME specs at the denominators the
disposition itself records [M]:

| spec | m evaluated | perfect run certifies `a_L` | vs the 0.95 bar it is read against |
|---|---|---|---|
| `ME.9` | 15 | **0.819** | below |
| `ME.10` | 36 | **0.920** | below |
| `ME.3` | 39.3 | **0.927** | below |
| `ME.1` | 40.0 ± 4.5 | **0.928** | below |
| `ME.5` | 52–60 | 0.944 – 0.951 | at the edge |
| `ME.11` | 300 | 0.990 | clear |

**This is not a claim that the repair is wrong** — `ME.1` reads 1.0000 on three
seeds and the trade the row feared (abstention bought with recall) demonstrably
did not happen. It is the observation that **our own bakeoff document already
requires ≥300 negatives for exactly this certification, and the conjunct that
shipped runs at ~40**, so five specs read a perfect 1.0 that is statistically
compatible with a true rate of 0.82–0.93. §1.8 made this same point about `ME.1`'s
and `ME.5`'s *fabricated* probes in August; the new *distractor* conjunct
inherited the denominator along with the design. **Cost to close: more negatives
in a fixture, no new mechanism.** Offered as an observation to whoever owns the
ME family; nothing here moves a threshold.

---

## 7. What this report does NOT claim

- **No arm here has been run.** Every number in §2 and §3 is someone else's
  measurement on someone else's hardware. The §5 bound evaluation, the §6 greps
  and ledger reads, and the §6b Clopper–Pearson table are **ours**, marked
  **[C]/[M]**, with the commands shown.
- **N1 is a CEILING PROBE, not a retriever.** A perfect result ("feasible at
  d = 16") routes the redesign and builds nothing. And its own must-fail control
  is the likeliest way it dies — §2 leads with that rather than burying it.
- **N2 is nominated with its date and its missing numbers in the heading.** It is
  out of window, it has no error bars, no p-values and no agent seed count, and
  its selection criterion (neural similarity to a fish) is not ours. The
  "death is a page turn" reading of its ecological prior is **my design question,
  not the paper's proposal**, and must not be cited as theirs.
- **N3 is not a new paper.** It is a fetch that closed a queued question, and its
  only new content is a formula and a costing correction against myself.
- **I did not run an arXiv enumeration this week.** HTTP 429, six attempts, two
  network paths. Weeks 4–6 each ran 40-entry enumerations per front; this sweep
  ran none, and the coverage table says so in its own row rather than in a
  caveat. **A sweep that cannot enumerate is a weaker sweep and this one was.**
- **Two search-engine paraphrases were wrong this week, and the `[s]` marker
  caught both.** (i) A snippet asserted that on LIMIT *"multi-vector (MaxSim)
  significantly outperforms single-vector"*; the paper's own LIMIT-small table
  reads multi-vector **23.1 %** recall@2 *below* the best single-vector **54.3 %**,
  with BM25 **85.7 %** above both. (ii) A snippet called arXiv:2506.00138 *"a 2026
  paper"*; it is v1 **2025-05-30**, v2 **2025-10-24**. Week 4's one-character rule
  is paying for itself.
- **A coverage finding about my own searching, which is the reason N2 exists.**
  Four sweeps of front 4 enumerated `cs.LG` / `cs.AI` / `cs.RO` and reported
  *"zero with a body under homeostatic drive"*. The one that exists is **q-bio.NC**
  and has been on arXiv for fifteen months. `GOAL.md` says biology is the oracle;
  this desk had been searching the categories where the oracle is *cited* rather
  than the one where it is *published*. That is a gap in my method, not in the
  field, and it is the cheapest correction available: **search `q-bio.NC` on the
  biology front every sweep.**
- **The `ME.1` / bestcov numbers in §0 are the builder's measurements, not mine.**
  I read them out of the routed row and the ledger; I did not run them and I do
  not own their interpretation.
- **Verification is uneven and marked:** N1 — full HTML, theorem and table
  quoted, **no latency anywhere**; N2 — full HTML ×2, **no numeric table exists**;
  N3 — full HTML, formula quoted, **no hardware/params/seeds/code**; 2608.29434,
  RARE, CoDeR, Argus Eyes — abstract level; the generator-side abstention papers
  in §5 — search-result level **[s]**, listed to establish a *family*, and no
  claim of theirs is relied on.

---

## 8. Queued for next sweep (**not before ~2026-09-21**)

1. **RETRY THE arXiv API FIRST, BEFORE ANY SEARCH.** If it answers, run the
   40-entry enumerations that weeks 4–6 ran and this one could not — fronts 2 and
   5 first, since they got one search each. If it 429s again, that is **two**
   consecutive closures and the desk needs a stated alternative enumeration route
   (arXiv listing HTML, or per-category `WebFetch`) rather than a third week of
   silently degraded coverage. Week 3's deferral rule applies: closed, re-planned,
   or dropped with cause — never "still pending" a third time.
2. **Did N1 run, and what was the critical `d`?** It is CPU-minutes and it routes
   `ME.11`'s successor. If the shuffled control also reached feasibility, the
   probe is dead and I want that recorded as a dead nomination, not quietly
   dropped.
3. **Did §6 get a ruling — is the diagnostic built, or is the document
   corrected?** The two disagree today. Whichever way it falls, **wk5-N1's and
   wk6-N1's prices both depend on the answer**, and this desk will keep
   mis-costing that seat until it exists.
4. **`t402` and `w1-world-edit-window`, both overdue since 09-13.** wk6-N3's
   pathway-decoding control and ForageWorld's design reference are both waiting on
   rows that did not move. If `t402` is still open on 09-21 that is 16 days on a
   `FAIL-UNOWNED` repair, and the right thing for a scout to do is say the number,
   which is what this line is.
5. **`q-bio.NC` on the biology front, standing** (§7). One search per sweep. It
   is where N2 came from and where four sweeps of `cs.*` enumeration could not
   have found it.
6. **N2's numbers.** Two full HTML passes found no table. One more attempt —
   PDF/appendix figures, or the released repo's evaluation scripts
   (`neuroagents-lab/autonomous_zebrafish`) — decides whether 3M-Progress can be
   nominated on evidence rather than on a body and a mechanism. **If a third pass
   finds no number, it goes to the watchlist and out of §2**, per this desk's own
   rule about carrying a nomination on absent evidence.
7. **NOT queued, deliberately:** conference proceedings (dropped wk4);
   2607.22430 (dropped wk6); Simulus (dropped wk6); PIC/POIC successors (closed
   wk5); UED as a cheap score (closed negative wk6); **and now LIMIT's *conclusion*
   — §5's arithmetic closes it, and only its *instrument* survives as N1.**
