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

**Sweep date:** 2026-08-31 · **Window:** ~2026-03 → 2026-08 (6 months)
**Scout:** field watch, week 5. **Seven days since week 4** (2026-08-24), and
week 4's embargo (*"not before ~2026-08-31"*) is spent exactly. First sweep to
land on the intended weekly cadence with no gap and no overrun.

**Confidence markers used below:** **[V]** fetched and read · **[c]** claimed by
the authors, not yet checked against their table · **[s]** asserted by a search
engine *about* a paper I have not opened — week 4's proposed marker, endorsed by
the Review, used here for the first time · **[C]** computed by me, arithmetic shown.

---

## 0. WHY THIS SWEEP IS SHAPED DIFFERENTLY — front 4 finally has a scar, and it
## is the front week 4 swept most shallowly

210 commits landed in the seven-day interval. Three of them re-point this desk.

**(1) `T3.06` VOIDed and was declared VOID-FORECLOSED — curiosity's only
implemented, unsettled claim spec.** Week 4 admitted front 4 was its shallowest
front (*"two searches, no fetch"*) and queued it to go first. It goes first here,
and it now has something to aim at: the routed row
`t306-matched-magnitude-noise-buys-coverage` (DUE 2026-09-06). The claim
conjuncts were green (`delta_coverage` +0.2458, 5.8σ) and the **control went
red on every seed** — a matched-magnitude *uninformative* reward (`shuftask`)
recovered the coverage, so the contrast cannot attribute the effect to curiosity.

**(2) `w0-too-shallow` is RED and OVERDUE**, its design owed 2026-08-30 by a
Review run that died having written nothing. Six instruments now say W0 is too
shallow; `BA.03`'s VOID added the sixth and it weighs differently from the rest
(the world does not *require* the vestibular sense, as against not *rewarding*
a capability). Front 5 is therefore the front where a design reference is worth
more than an arm.

**(3) `LC.03` is VOID-FORECLOSED and `D10`'s default fires 2026-09-01**, seating
`wm-latent` (`A4`) on the learning-core seat. Whatever the marking ends up being
— the 52nd audit asks the owner for `BY DEFAULT` rather than `BY VERDICT` — `A4`
is the arm the project will be building on this week. Front 1 stays pointed at it.

**And one thing did not move.** `T4.02` is unchanged since 2026-08-21 (FAIL,
30.12 vs the 10× gate) and `SM.02` has still never run — **fourth consecutive
sweep reporting it**, with week 2's whiff-clock nomination live and unrun on top.

---

## 1. Coverage — what was actually searched, so the gaps are visible

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **4 · CURIOSITY & OPEN-ENDEDNESS** — *swept FIRST, per week 4's queue item* | intrinsic-reward ablations with matched-magnitude/random-reward controls; epistemic-vs-aleatoric separation; learning-progress estimators; the RLVR "spurious rewards" line; UED / autocurricula / open-endedness | **full arXiv-API enumeration (40 entries)** + 4 targeted searches; **full abstract for Curiosity-Critic**; searches only on UED |
| **1 · LEARNING CORES** | decoder-free / latent-predictive world models; anti-collapse regulariser theory; identifiability; JEPA↔active-inference | **full HTML with quoted propositions** for the nomination; **abs for 2607.22430**; **full HTML for NE-Dreamer**; abs for Dreamer-CDP |
| **5 · WORLDS & EMBODIMENT** | homeostatic/survival sims; task-complexity and environment-discriminability metrics; embodied mortality | **full arXiv-API enumeration (30 entries)**; **abs + HTML for 2602.18856**; **full abstract for 2604.10760**; abs for 2510.07117 |
| **3 · MEMORY** | agent/episodic memory; extractive vs generative write AND read paths; decay/consolidation | **full arXiv-API enumeration (30 entries)**; **full abstract for ScrubJay-MEM**, interrogated on both paths |
| **2 · MULTIMODAL FUSION** | modality-imbalance successors; whether any of it leaves supervised classification; the balance-vs-budget split | **full arXiv-API enumeration (25 entries)**; **abs for IIBalance** |
| Biology-as-oracle | homeostasis→prosociality; embodied mortality→generalisation; animal episodic memory | folded into fronts 5 and 3; **two full abstracts** |
| Small-model end | sub-1M-parameter embodied control | one search, **nothing found** — third consecutive sweep |
| Queued #1 (2607.22430 full text) | **ATTEMPTED, NOT CLOSED.** See §6 | abs only; the computability question is not answerable from it |
| Queued #2 (NE-Dreamer DMC table) | **CLOSED with a null result.** See §3 | full HTML |
| Queued #3 (Dreamer-CDP) | **CLOSED and it demotes itself.** See §3 | abs |
| Queued #4 (front 4 first) | **DONE.** This section's first row | — |
| Queued #5 (`SM.02`, `NE.05`, `UB.10`/`UB.11`, D10) | **ledger + `DECISIONS_NEEDED.md` read.** See §0 and §4 | ledger |

**Known gaps, stated so nobody assumes coverage:**
- **This sweep's lead nomination has ZERO experiments** — it is a theory paper and
  §2 says so in its first line. That is a different failure mode from week 4's
  "no hardware reported", and §6 treats it as a finding rather than hiding it.
- **`2602.18856` yielded no numbers.** Both the abs page and a targeted fetch
  returned the qualitative claims only; no sample sizes, no hardware, no code.
  It is nominated on a *direction*, not a magnitude, and §2 marks that.
- **UED / open-endedness got searches and no fetch.** Front 4's intrinsic-motivation
  half was swept properly; its environment-generation half was not. That half is
  queued, and it is the half that bears on `w0-too-shallow`.
- ScrubJay-MEM, IIBalance, 2510.07117 and 2604.10760 are all **abstract-level**;
  none reports seeds, hardware or latency, and only one mentions code at all.
- No non-English sources. No conference main-track enumeration (permanently
  dropped, week 4 §5 — still dropped, still not re-queued).

---

## 2. NOMINATIONS

Three. **One enters front 1 on `A4`; one enters front 5 / `NE.07`; one is not an
arm at all and says so.** Each states its arXiv **primary category**, what is
**[V]**/**[c]**/**[s]**/**[C]**, its cost on **our** substrate, and — steelmanned
both ways — why it might win and why it might lose.

**The jurisdictional note, updated.** Week 4's three nominations all entered
`LC.04`/`LC.05`, BLOCKED behind D10. **D10's default fires 2026-09-01**, so that
block ends this week. N1 below is a variant of the arm that seat lands on. N2 and
N3 deliberately enter elsewhere — `NE.07` and an already-ordered diagnostic —
because a desk that puts every nomination behind one decision is a desk that stops
if the decision goes the other way.

---

### N1 — SIGReg is the anti-collapse regulariser that makes a JEPA objective a valid free energy, and VICReg — which is what `A4` uses — provably is not

**Source:** *The SIGReg Objective as Variational Free Energy* —
[arXiv:2607.13612](https://arxiv.org/abs/2607.13612), **2026-07-15**, primary
category **cs.LG**. Fabio Arnez, Alexandra Gomez-Villa. **Full HTML read;
propositions quoted below.**

**Not cited anywhere in this repo.**

**Why this lands on the seat that is about to be filled.** `LEARNING_CORE.md`
§3.4.3 records the choice explicitly: PLDM *"uses VICReg-style explicit
regularisation rather than EMA… That is LeCun's own group deciding that at small
scale, the explicit variance floor is the safer bet. **§5's A4 arm follows it.**"*
So `A4` — the arm `D10` seats tomorrow — carries a VICReg-style variance floor
by inheritance from one paper's design choice. This paper puts four candidate
regularisers in a proved hierarchy and puts VICReg on the wrong side of it.

**The verified claim [V].**

| | |
|---|---|
| the hierarchy | VICReg, LogDet, PairDist, SIGReg ordered by a **prior-miscalibration gap**; the gap's **sign** decides whether the Active-Inference surprise bound survives |
| the verdict | VICReg and LogDet are **unsafe upper bounds**; PairDist a safe lower bound; **SIGReg eliminates the gap** |
| what "unsafe" means, quoted (Prop. 2) | *"an upper-bound proxy (Ĥ≥h, e.g. VICReg or LogDet) gives F̂⁺≤F⁺, which no longer certifies F⁺≥−ln p(x): the AIF bound is not guaranteed to be preserved"* |
| the operational failure, quoted (Remark 1) | *"the optimiser can raise Ĥ by inflating the slack instead of the entropy"* |
| the correspondence theorem | under constant-noise encoder + successful SIGReg enforcement, the gap vanishes, the objective becomes **an exact information bottleneck**, the surprise bound is preserved, and the latent goal cost is an exact proxy for AIF pragmatic value; **VICReg leaves an irreducible second-order anisotropy term** |
| **action-conditioned?** | **YES, explicitly** — *"a predictor P_ξ: Z×A→Z produces ẑ_t = P_ξ(z_{t−1}, a_{t−1})"*. This is `A4`'s case, not static representation learning |
| the named gap | **state-epistemic value** `h(Z_τ|π) − C_ε`, *"a coverage signal driving the agent toward policies that maximise future-state entropy"* — *"No current JEPA world model computes this quantity; it is the primary structural gap between AIF and JEPA planning"* |
| verification | *"the algebraic core of every result is **machine-verified in Lean 4** (Appendix D), compiling with **zero sorry obligations**"* |
| **experiments** | **NONE.** *"Empirical validation of its predictions is left to separate work."* |
| code / Lean repo | **not stated** |

**Which spec it enters.** `LEARNING_CORE.md` §5.4, as the **selection criterion
between two arms already nominated and already accepted**: week 1's `A4b` (SMWM,
inverse-dynamics) and `A4c` (SIGReg/LeJEPA). It is not a new arm. It is the first
*principled* reason to prefer one of the anti-collapse routes on the desk over the
other four, and it makes `A4`'s incumbent choice the one route with a proved
defect. It also converges with week 4 §3's identifiability theorem
(2607.22430), which gave `A4c` a recovery guarantee **for the action-conditioned
case** — two independent theory papers, four months apart, both landing on
SIGReg-style enforcement in exactly `A4`'s setting.

**Cost on our substrate.** A regulariser swap inside an existing arm: **zero new
parameters, zero new networks, no change to the observation or action space.**
SIGReg's cost is a sketched characteristic-function test over the batch —
cheaper per step than the covariance term VICReg already computes at `A4`'s latent
width. `A4`'s ≈1.37 M is unchanged. This is the cheapest nomination on the desk
since week 1's interoceptive-precision item.

**Why it might WIN (falsifiable).** `A4`'s named silent failure is latent
collapse, and its mandatory diagnostic is **effective rank + per-dimension
variance every 1,000 decisions**. Remark 1 describes precisely how a VICReg floor
can be satisfied without buying what it was installed to buy — *the optimiser
inflates the slack, not the entropy*. That predicts a **specific, measurable
signature**: an `A4` run whose variance floor reads healthy while effective rank
stays flat or falls. That signature is checkable **on curves we already have**,
at zero compute, and it is the rare nomination that can be partly adjudicated
before it is run. If `A4c` raises effective rank at matched variance *and* raises
`life_gain` ≥1.5σ over 3 seeds against incumbent `A4`, one mechanism explains both.

**Why it might LOSE (steelmanned). Five.**
1. **There are no experiments. At all.** Every previous nomination on this desk
   was argued from someone's benchmark table; this one has no table to doubt
   because there is no table. `SYSTEM.md` law 3 — decisions by bakeoff, never by
   argument — bites a theory paper harder than an empirical one, and a normative
   argument is *exactly* the thing this project is built to distrust.
2. **The authors themselves say the model is a fiction.** Quoted, §5.1: *"The
   constant-noise model is an interpretive device, **not a claim about deployed
   systems**."* The correspondence theorem's hypotheses (constant-noise encoder,
   Gaussian encoder family, SIGReg enforcement in the population limit M,N→∞) are
   not `A4`'s conditions and the authors do not pretend otherwise. Their
   Corollary 2 argues violations degrade linearly rather than catastrophically —
   which is a *claim about the degradation*, also unmeasured.
3. **The Lean verification covers less than it sounds like.** *"The algebraic
   core"* — and Remark 3 concedes the quantitative Cramér–Wold step is *"classical
   only in its qualitative form"*. Zero `sorry` obligations on the algebra is real
   and rare; it is not a proof that the theorem applies to a trained network.
4. **The named gap is named, not filled.** No computable estimator for
   state-epistemic value is given; the authors present it as future work. So the
   most interesting sentence in the paper hands us a question, not a term.
5. **The whole AIF frame has already lost here once.** `A3` (`wm-efe`) was our
   expected-free-energy arm and it read **t = 2.05 / 2.07**, below the 3σ gate.
   A paper arguing that `A4` should be made *more* AIF-shaped is arguing toward
   the family our own screen could not distinguish from noise. That objection is
   weaker than it looks — `A3`'s EFE was in the *planner*, this is in the
   *regulariser* — but it is not nothing, and it is the one I would press first.

---

### N2 — Helping that emerges from homeostatic coupling alone, with a lesion battery, aimed at the need `NE.02` is empowered to delete

**Source:** *Prosociality by Coupling, Not Mere Observation* —
[arXiv:2604.10760](https://arxiv.org/abs/2604.10760), **2026-04-12** (rev
2026-06-09), primary category **cs.MA**. Aishik Sanyal (single author).
**Full abstract read.**

**Not cited anywhere in this repo.**

**Why it is here, and it is the most `GOAL.md`-shaped item this desk has found.**
`GOAL.md` refuses to carve Jack's character and says why: *"His kindness is not
decreed; it is expected to GROW from his need for company, the way it grew in
us… I want to let as much of this naturally develop."* The deal that keeps that
honest is *"what emerges is OBSERVED, measured, and reported truthfully."*
**This paper is the first measured instance of exactly that mechanism** — and it
arrives at the need that is currently weakest in our own suite.
`NEEDS_AND_DEATH.md` §4 calls the social need *"the weakest biological analogy in
the suite, and the one whose need-status is most contested in its own
literature. Its λ is the smallest and `NE.02` is explicitly empowered to delete
it."* Its physical row is **empty by design** — it must earn its place
behaviourally or be cut.

**The verified claim [V].**

| | |
|---|---|
| what is isolated | *"Artificial agents can be made to 'help' through explicit social rewards, hard-coded prosocial bonuses, or direct access to another agent's state. **I isolate a narrower route: homeostatic coupling.**"* |
| the construction | a scalar homeostat + a social coupling channel, *"keeping action selection self-directed: **the planner scores only the actor's predicted internal state, with no partner-welfare reward**"* |
| worlds | **FoodShareToy** (one step, exact solver) and **SocialCorridorWorld** (multi-step) |
| the toy result | an exact solver finds a switch from EAT to PASS at **λ\* ≈ 0.91** for the default state |
| **the control that matters** | *"**partner-state access without coupling leaves behavior unchanged**, whereas coupled agents fetch, carry, and pass food to the partner"* |
| the lesion battery | **sham lesions preserve helping; coupling-off and shuffled-partner lesions abolish it** |
| **the authors' own negative** | *"a coupling/load sweep shows that coupling creates a low-load helping regime but **does not guarantee rescue under higher metabolic load**"* |
| seeds / hardware / wall-clock / params / code | **none stated, none, none, none, not stated** |

**Which spec it enters.** `NEEDS_AND_DEATH.md` §4, `NE.07`, as **an arm plus a
control**, and the control is the more valuable half:

- **As an arm:** `c` (social) currently restores on *"proximity, reciprocated
  conversation, being helped, helping — each a recorded diary event"*, with
  λ_c = 0.3 and the visitor as a scripted companion. That is a need the reward
  reads. The paper's construction is the opposite and stricter: the partner's
  state routes into the actor's **own homeostat**, and the planner never scores
  partner welfare. If helping appears under that construction it cannot be the
  reward function talking.
- **As a control, and this is the part I would take even if the arm is refused:**
  `NE.07` already carries **C-DECOY** — *"an object with the visitor's visual and
  acoustic signature but no identity… must restore nothing"*, separating a person
  from a person-shaped stimulus. The paper's **shuffled-partner lesion is
  sharper**: identity *present but mismatched*. C-DECOY catches "any
  person-shaped thing will do"; shuffled-partner catches "any *person* will do",
  which is the failure mode that would make `NE.07`'s attributed-diary claim
  hollow while every metric stayed green. Same discipline as week 1's
  entity-collision nomination, on a different channel.

**Cost on our substrate.** One scalar coupling term and one extra lesion
condition. **Zero new parameters**; the visitor already exists in the spec; the
diary already records `speaker`. Runs on 4 ARM cores. The lesion is a re-run, not
a new world.

**Why it might WIN (falsifiable).** `NE.02` is empowered to delete the social
need if `approach_lift` falls to the placebo column, and on the current design
that is a live risk — the need is scored by a reward the agent can learn to
farm, so a null result is ambiguous between "social is decoration" and "the
channel was gameable". **Under coupling, the prediction is sharp and
pre-registerable:** helping behaviour appears with coupling on, vanishes with
coupling off, and **survives the shuffled-partner lesion nowhere**. If that
pattern reproduces in W0, the social need has earned its parameters
behaviourally rather than by analogy, and `GOAL.md`'s most-refused-to-carve
sentence has its first measurement. If it does not reproduce, `NE.02` deletes the
need **with a reason** instead of on a weak effect.

**Why it might LOSE (steelmanned). Five.**
1. **The paper's own load sweep predicts it fails in our regime.** *"Coupling
   creates a low-load helping regime but does not guarantee rescue under higher
   metabolic load."* W0 is a **high-load world by construction** — needs, a 600 s
   basal ceiling, death. The authors' own strongest negative result points
   straight at Jack's conditions. **This is the objection I would lead with**,
   and it is derived from their measurement, not my scepticism.
2. **Nothing here learns.** FoodShareToy is solved by an **exact solver**;
   SocialCorridorWorld is a planner over a hand-built homeostat. Jack's social
   behaviour would have to be *learned*, by a core that `SH.01`'s
   `ORACLE_CANNOT` pilot showed cannot yet learn to seek shelter when
   sheltering demonstrably pays and the direction is in the observation.
3. **Single author, no seeds, no CIs, no hardware, no params, no code, and it
   builds on "ReCoN-Ipsundrum"** — an architecture that is not ours and that I
   have not verified exists outside this line of work. Weakest provenance of any
   nomination on this desk since week 1's entity-collision item.
4. **λ\* ≈ 0.91 is one number from an exact solver on a one-step toy**, for *"the
   default state"*. It is a threshold in their parameterisation and transfers to
   ours as nothing.
5. **`NE.07` has never run**, and neither has `NE.02`. Nominating a sharper
   control into a spec with no attempts is nominating into a queue, not a rig.

---

### N3 — NOT AN ARM: a measured warning that the instrument class `w0-too-shallow` and wk4-N3 both rely on is broken

**Source:** *Issues with Measuring Task Complexity via Random Policies in Robotic
Tasks* — [arXiv:2602.18856](https://arxiv.org/abs/2602.18856), **2026-02-21**,
primary category **cs.LG**. Reabetswe M. Nkhumise, Mohamed S. Talamali, Aditya
Gilra (Sheffield / WU Vienna). **Abstract [V]; a targeted fetch for numbers
returned none — see the honesty note below.**

**I am labelling this a nomination and simultaneously saying it is not an arm,
because the alternative is filing it as a watchlist item that arrives after the
measurement it bears on has already been run.** `wk4-N3` is **ACCEPTED and
ORDERED** — the Review sequenced it *"BEFORE any W1 world redesign"* as the cheap
falsifier against the four expensive agreeing instruments. It is the one field-watch
nomination in five weeks that is actually about to execute. This paper is a
measured caution about the inference class it belongs to, and it is worth more
now than in any later sweep.

**The verified claim [V], and it is a negative result.**

| | |
|---|---|
| what is evaluated | **RWG** (Random Weight Guessing) and the two information-theoretic metrics built on it, **PIC** (Policy Information Capacity) and **POIC** (Policy-Optimal Information Capacity) |
| the method | *"progressively difficult robotic manipulation setups with **known relative complexity**"*, dense and sparse reward formulations |
| finding 1 | *"**PIC suggests that a two-link robotic arm setup is easier than a single-link setup** — which contradicts the robotic control and empirical RL perspective"* |
| finding 2 | **POIC estimates sparse-reward tasks as EASIER than their dense-reward variants** |
| the conclusion | *"both PIC and POIC contradict typical understanding and empirical results from RL"*; the authors call to *"move beyond RWG-based metrics"* |
| replacement proposed | **NONE** |
| numbers / seeds / hardware / code | **none extractable** from either the abs page or a targeted fetch |

**Where it enters.** Not a bakeoff. Two places, both live this week:

1. **`wk4-N3`'s diagnostic design** (`INTEGRATION_QUEUE`, ordered). That
   diagnostic runs a β-scheduled colored-noise **random policy** against
   `LC.03`'s existing `random` / `random-repeat` nulls and reads `life_gain`. It
   is *not* PIC — it compares random policies to each other on a task-relevant
   outcome rather than computing an information-theoretic capacity — so this
   paper does **not** refute it. What it does is establish that the family
   *"infer a property of the environment from how random policies behave in it"*
   has a **measured** track record of inverting on setups whose true ordering is
   known. That argues the diagnostic should carry its own known-answer check: run
   it on a world whose relative shallowness we already know, and require it to
   get that one right before its W0 reading is believed. **That is a control, and
   it costs the same CPU-minutes the diagnostic already costs.**
2. **`w0-too-shallow`'s design**, where the natural next move — after six
   instruments agree — is to reach for a published environment-difficulty metric.
   This says the two best-known ones would mislead us, and names no successor.

**Cost.** Zero. It is a caution and a control, not a method.

**Why it might WIN (i.e. why acting on it is right).** The `w0-too-shallow` row's
own disposition says the danger precisely: all four (now six) instruments *"were
run by this project on this world, and they all point the same way — which is
exactly the condition under which a shared confound is invisible."* The Review
bought a cheap disagreeing instrument to attack that. **A cheap disagreeing
instrument with an unvalidated inference is not obviously better than four
agreeing ones** — unless it carries a known-answer check, which is what this
paper argues for and what costs nothing to add.

**Why it might LOSE (steelmanned). Four.**
1. **It is not about our measurement.** PIC/POIC are information-theoretic
   capacities computed from random-weight policy *return distributions*; wk4-N3
   compares `life_gain` between two random policies. The transfer is an analogy,
   and this desk's own week-3 lesson is that an analogy is not arithmetic.
2. **No numbers, no seeds, no hardware, no code.** I could not extract a single
   quantitative result. For a paper whose entire content is "these metrics give
   wrong answers", the absence of the wrong answers' magnitudes is a real gap,
   and I would not nominate on this evidence if it implied any spend.
3. **Robotic manipulation, not survival.** Single- vs two-link arms with dense and
   sparse rewards; W0 is a homeostatic world with death.
4. **It proposes no replacement**, so the most it can do is make an ordered
   measurement slightly more careful. That is a small win, and I am not dressing
   it as a large one.

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**Resolved this sweep and now DELETED** (recorded so a sixth sweep does not
re-open them):

| item | resolution |
|---|---|
| **NE-Dreamer** (2603.02765, cs.LG) — *queued #2* | **CLOSED with a null result, and DROPPED.** Full HTML read. **The DMC numbers do not exist in extractable form** — Fig. 6 shows learning curves across 20 tasks against DreamerV3 / R2-Dreamer / DreamerPro and the paper states only that NE-Dreamer *"matches or slightly exceeds"* them; **no per-task scores and no aggregate in any table**. Confirmed: **12 M params (DreamerV3-Small), 5 seeds, 1 M env steps, 64×64 RGB**, no hardware, no wall-clock, no code. It is pixels-only at ~9× `A4`, and its wins are on DMLab memory/navigation. **One thing worth keeping and it is not the method:** the paper's own framing of its DMC row is *"no regression without reconstruction"* — an independent calibration point that deleting the decoder costs nothing on continuous control. That is the same direction `LC.03` measured here at 4.65σ, from a different lab in a different regime. Corroboration, not news. |
| **Dreamer-CDP** (2603.07083, cs.LG) — *queued #3* | **CLOSED, and it demotes itself.** Hauri & Zenke, 2026-03-07 (rev 04-14). A JEPA-style predictor on continuous deterministic representations — a genuine sixth decoder-free route. **But its own abstract claims parity, not a win:** *"We **close this gap** between Dreamer and reconstruction-free models… Our method **matches** Dreamer's performance on Crafter."* One pixel benchmark, matching a reconstruction baseline. `A4` already **beat** its reconstruction sibling here by 4.65σ on state vectors. A method whose headline is parity-in-pixels adds nothing to the arm that won outright in our own regime. **The `A4`-neighbourhood watchlist is now saturated** — that was the question one fetch was meant to decide, and it decided it. |
| **2607.22430** identifiability — *queued #1* | **ATTEMPTED, NOT CLOSED, and NOT re-queued as-is.** See §6. |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **Simulus** ([arXiv:2502.11537](https://arxiv.org/abs/2502.11537)) | cs.LG | Oldest open item, **fifth sweep**, blocked on the same two numbers (a parameter count and a per-step wall-clock) that `B4` needs and the paper does not report. Its prioritised-replay component remains separately nominated (wk2-N2 → `NE.05`, **never run**). **Honest call: if a sixth sweep cannot close it, it should be dropped with cause rather than carried a seventh time** — week 3's rule about deferral applies to watchlist items too. |
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | Unchanged. Measures rollout-horizon deterioration in the **fully observable state space** across RSSM / Transformer / Diffusion / Neural-ODE — our regime. Still blocked on: no compute cost, no hardware, no environment size, no code. **Promote on: any statement that a domain runs on CPU.** |
| **ForageWorld** ([arXiv:2506.06981](https://arxiv.org/abs/2506.06981)) | cs.AI | **Kept and now MORE relevant, still not an arm.** `w0-too-shallow` is OVERDUE and its design is owed; ForageWorld remains the closest published existence proof of a world with the discriminating features D10(b) would add — depleting/diffusing food, pursuing predators, and a **sleep action gated on energy < 50 % that immobilises the agent while restoring it**. Still Craftax-based and GPU-accelerated (the axis `SURVIVAL_WORLD.md` §2.2 ruled out) and a gridworld where W0 is a body. **Design reference only.** |
| **Survival RL** ([arXiv:2605.31273](https://arxiv.org/abs/2605.31273)) | cs.LG | Unchanged disambiguation — "survival" = dwell time at goals, not homeostatic needs. Kept so a sixth sweep does not chase the title. |
| **Eywa** ([arXiv:2605.30771](https://arxiv.org/abs/2605.30771)) | cs.CL | Unchanged. Still the only front-3 item with *"zero LLM calls inside retrieval"*; still blocked on whether the **write** path is generative, and still reports no hardware and no latency. |

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **Curiosity-Critic** ([arXiv:2604.18701](https://arxiv.org/abs/2604.18701), 2026-04-20, rev 06-16) | cs.LG | Bhaskara & Wang. Intrinsic reward = current prediction error **minus a learned asymptotic error baseline** for that transition; the critic *"only has to learn how hard a transition is to predict"*, so its noise-floor estimate converges before the world model saturates. Reward is high for learnable transitions and **collapses toward zero for stochastic ones** — epistemic/aleatoric separation, online. Claims Schmidhuber (1991) onward are special cases at particular baseline approximations. **Code released** (`vinbhaskara/Curiosity-Critic`). | **A continuous-control result.** Its only experiment is *"a stochastic grid world"*, and it is an **ICML 2026 Workshop** paper (EIML), not main track. `CURIOSITY_BAKEOFF.md` already cites **LPM (2509.25438, ICLR 2026)** for this family and rates it *"the most directly on-topic modern paper for this document"* with a **proven monotone indicator of information gain**. Curiosity-Critic is a sibling with weaker evidence at a weaker venue. **Not nominated: we already carry the better-evidenced member of its family, and neither has any continuous control.** |
| **ScrubJay-MEM** ([arXiv:2608.04746](https://arxiv.org/abs/2608.04746), 2026-08-05) | cs.CL | Bhandari, Wadhwani, Kumar, Narang. Operationalises **western scrub-jay caching** (Clayton & Dickinson's What-Where-When) as *type-conditioned temporal decay*: each memory is a What-Where-When tuple with a perishability coefficient π_i and utility horizon τ_i, retrieved by query-adaptive scoring. +0.108 GenGap and +2.66 F1 over Mem0 on MemoryAgentBench EventQA-64k. | **Nothing, for adoption — but the BIOLOGY is the item, not the system.** The read path looks deterministic and the store is a tuple, which is admissible; but the write path makes *"O(1) LLM calls per update"* to auto-classify π_i, which is generation moved one step upstream — the same objection that held Eywa. And it is LLM-agent QA with **no seeds, no hardware, no latency, no code**, so it cannot inform `MEMORY_RETRIEVAL_BAKEOFF.md` §1.9's CPU table. **What is portable is the oracle**: perishability is a property of the *remembered thing*, and a hand-specified type→decay map needs no LLM at all. Promote if a non-LLM implementation with latency numbers appears, or route the biology at `Forgetting.py` directly. |
| **IIBalance** ([arXiv:2603.17347](https://arxiv.org/abs/2603.17347), 2026-03-18) | cs.MM | Xiong et al. *Beyond Forced Modality Balance: Intrinsic Information Budgets.* Argues balancing methods *"overlook the fact that each modality has finite information capacity"* and should correct semantic drift *"only when weaker modalities deviate from their **budgeted potential**, rather than forcing imitation."* | **Nothing yet — but it is the second paper contesting the balance objective and it contests it in a more useful direction than PDMP.** For a brain with five senses of wildly different dimensionality, "equal gradients" was never the right target and "each sense contributes up to its own capacity" might be. **Still supervised classification** (three unnamed benchmarks), no modalities specified, no seeds, no numbers extractable. Promote on: any instance outside classification. See §5. |
| **Embodied mortality → generalisation and care** ([arXiv:2510.07117](https://arxiv.org/abs/2510.07117) v3, 2026-02-12) | cs.AI | Christov-Moore, Juliani, Kiefer, Lehman, Reggente, Rousse, Safron, Hinrichs, **Polani, Damasio**. Argues generalisation and care arise from *"being-in-the-world"* and *"being-towards-death"* — homeostatic self-maintenance forces robust causal models of embodiment, and empathy follows from expanded self-boundaries. | **Nothing. It is a ~15-page position paper with no experiments**, and I am listing it only because of what it is: `GOAL.md`'s survival-world thesis and its kindness-from-company sentence, argued independently by Damasio and Polani. **Corroboration is not news and this desk has said so three times.** Recorded so a future sweep does not mistake it for evidence. N2 is the measured version of the same claim and that is where the action is. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS — and one of them is finally moving

**Week 4's three nominations were ALL ACCEPTED by the Review**, and one is
ordered for execution. That is the first time anything on this desk has reached a
run queue, and it changes what this section is for: it is no longer a list of
things nobody has done.

| nomination | entered | status now |
|---|---|---|
| **wk4-N3 · infant motor noise (2606.16590)** | W0 diagnostic | **ACCEPTED AND ORDERED — the first field-watch nomination to be sequenced for execution.** Rejected as an exploration arm on `A0`/`A1` for the exact reason I raised against my own nomination (PPO's likelihood ratio), and accepted **only** in its W0-diagnostic form, then *promoted above the other two* and sequenced **before** any W1 redesign. It is now cited inside the `w0-too-shallow` Review row as the cheap attack on a six-instrument shared confound. **N3 above is a control for it.** |
| wk4-N1 · spectral-radius constraint (2607.19719) | `A4` variant | **ACCEPTED, narrowed exactly as nominated**, with my own lead objection (no seed count, no CIs) carried into the entry as a binding condition: it enters as an arm measured here at ≥3 seeds, and the paper's numbers are *"the motivation and are not admissible as evidence about Jack."* Register the design, do not dispatch. **Unblocks when D10 fires.** |
| wk4-N2 · PSG-JEPA (2608.06799) | `A4` ×2 | **ACCEPTED as two arms, and "the one to design FIRST" of the three** — `ℒ_dynamic` and `ℒ_static` run as each other's control, with my prediction (`ℒ_static` alone should not help and may hurt) **pre-registered**. Sequenced AFTER D9, because its target is joint-angle kinematics and the body fork is open. |
| wk1 · anti-collapse regularisers → `A4b`/`A4c` | `A4` variants | **LIVE, unrun — and N1 above is the first evidence that DISCRIMINATES between them**, rather than adding a sixth route. Of everything on this desk these have now been promoted twice without a new experiment: once by `LC.03` naming `A4` the only learner, once by two theory papers landing on SIGReg in the action-conditioned case. |
| wk1 · certificate-gated identifiability (2607.27017) → `UB.11` pre-gate | UB.11 | **LIVE, unrun.** `UB.11` has never run; `UB.10` is parked on the recipe-sensitivity row (dispositioned 2026-08-25 to matched **tuning budget**). 2607.22430 still does not supply the certificate — see §6. |
| wk1 · interoceptive precision (2608.04232) | `NE` §2.4b | **LIVE, unrun.** Still among the cheapest items on the desk (4 ARM cores, code released, 2.08× survival over the uniform-precision baseline that IS our current design). **And it is now adjacent to N2** — both are interventions on the needs suite, and `NE.07`/`NE.02` have never run either. |
| wk1 · entity-collision protocol (2605.29630) | `MR` §2 | **LIVE, unrun.** N2's shuffled-partner lesion is the same discipline on the social channel. |
| wk2 · the whiff clock → `SM.02` | SM.02 | **LIVE, unrun — `SM.02` has STILL never run. Fourth consecutive sweep reporting this**, on a spec whose kills-clause deletes a constitutional sense's wiring. |
| wk2 · RPE-prioritised replay → `NE.05` | NE.05 | **LIVE, unrun.** |
| wk3 · CIG (2605.20878) → `A3` | `A3` | **Remains DEMOTED** (`wm-efe` t = 2.05; its cheapest selling point closed with a null result in week 4). Candidate under D10(c) only. **N1 §"why it might lose" objection 5 bears on it**: if the AIF family is systematically weak in W0, that is now two independent lines pointing at `A3`'s neighbourhood. |
| wk3 · Optimistic World Models (2602.10044) → `A2` | `A2` | **Remains DEMOTED, hard** (`dreamer-xs` t = −0.94, worse with 4× data). |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

**FRONT 4 · CURIOSITY — swept FIRST and DEEPLY this time, and it produced no
nomination. But it produced something better, and it came out of our own ledger.**

The literature pass was real: a 40-entry arXiv enumeration plus four targeted
searches. **Nothing in it is a nomination.** Curiosity-Critic is a weaker sibling
of LPM, which `CURIOSITY_BAKEOFF.md` already cites and already rates as the
upgrade path for `A3`. The 2026 intrinsic-motivation output is otherwise LLM
reasoning bonuses (count-based RLVR, entropy centroids, confidence rewards),
multi-agent influence terms, or VLA/driving applications — none with a body under
homeostatic drive. The one structurally interesting convergence is not an arm:
the **RLVR "spurious rewards"** line (2506.10947 and successors, incl.
2601.11061) found that **random rewards recover most of the gain of true
rewards** in one model family and not others — the same shape as `T3.06`'s red
control, in a domain with nothing else in common. That is corroboration that the
matched-magnitude control is the right control, and corroboration is not news.

**What IS news is arithmetic in `T3.06`'s own recorded row, and it is not in the
routed options.** `T3.06` recorded four coverage numbers at 3 seeds. Computed
here **[C]** from the ledger row (`ran_at` 2026-08-30T01:06:21, attempt 1,
`d6fa40f26a853f8d`), using the same aggregate per-seed σ the row's own gates used:

| contrast | Δ coverage | se | t |
|---|---|---|---|
| `curious` − `task` (the green claim conjunct) | +0.2458 | 0.0376 | **6.54** |
| `curious` − `shuftask` (option (a)'s stated headroom) | +0.1385 | 0.0351 | **3.94** |
| **`curious` − `random`** (the random-**action** null) | **+0.0124** | 0.0317 | **0.39** |
| **`random` − `task`** | **+0.2333** | 0.0223 | **10.48** |

`coverage_curious` 0.6162 ± 0.0537 · `coverage_random` **0.6037 ± 0.0118** ·
`coverage_shuftask` 0.4776 ± 0.0286 · `coverage_task` 0.3704 ± 0.0367.

**A random-action policy covers W0 as well as the curious arm does (t = 0.39),
and it beats the task arm by MORE than the curious arm does (10.48 vs 6.54).**
Two consequences, both offered as observations for the Review's 09-06 decision
and neither as a decision:

1. **The routed row's option (a) picks the weaker of two comparators that are
   both already in the row.** It proposes rescoring against `shuftask`, where the
   arm has +0.138 of measured headroom. Against `random` it has +0.012 and t =
   0.39. **`CURIOSITY_BAKEOFF.md` §O1 already requires both** — *"≥ 2.0 vs NULL
   **and** ≥ 1.5 vs the RANDOM-REWARD arm"*, with C-RANDREW defined as the
   random-reward control *"for 'any optimisation pressure explores'"*. **The
   two-comparator discipline `T3.06` needs is already written in our own bakeoff
   doc**; it did not need a paper and it does not need me.
2. **This is a seventh instrument for `w0-too-shallow`, and it is free.** If
   random action saturates the coverage metric in W0, then coverage cannot
   discriminate exploration strategies there at all — which is the same disease
   the darkroom control, `LC.03` v2, `DP.05`, `SH.01` and `BA.03` each measured
   on a different channel. It costs nothing because the run already happened.

**I am not deciding anything with this**, and week 3's discipline lesson is why I
am stating it as arithmetic rather than as a story: a scout reading our own ledger
has no abstract to doubt. The numbers above are reproducible from the committed
row in four lines; the `aggregate-hides-worst-seed` caveat the row already carries
applies to my se's exactly as it applies to the row's gates.

**FRONT 2 · FUSION — nothing, for the second consecutive sweep, and the scar is
still there.** `T4.02` is unchanged (FAIL, 30.12 vs the 10× gate, twice). Week 4
refused this front for three reasons and **all three survive re-examination**;
one is now stronger. The imbalance family produced ~25 in-window papers and
**every single one is supervised classification** — recommendation, remote
sensing, EHR fusion, emotion recognition, federated graphs, VLM embedders. Not
one has left it, in the second consecutive year of trying. The most interesting
new entry, IIBalance (§3), argues the field's target should be **capacity-based
budgets rather than forced equality** — which is a better-shaped idea for a brain
whose senses differ in dimensionality by orders of magnitude — and it too is
classification, with no modalities named and no extractable numbers. **The
Goodhart objection is unchanged and remains the decisive one:** installing a
balancer to pass `T4.02` would make the metric read 1.0 while proving nothing
about load-bearingness. If this family ever enters, it enters `UB.10` as an arm
judged on the binding metric, with `max_modality_grad_ratio` reported secondary.

**FRONT 3 · MEMORY — nothing, for the FOURTH consecutive sweep, same
constitutional reason.** A 30-entry enumeration returned associative recollection,
memory-graph agents, GUI latent memory, self-evolving harnesses, Hebbian LLM
memory, and benchmarks — generative recall end to end, almost entirely cs.CL.
ScrubJay-MEM (§3) is the closest thing to an escape and it puts an LLM call in the
write path. **The one genuinely non-LLM entry in the whole sweep** is *Episodic
Memory Temporal Consistency for Cooperative MARL* (2606.04492, cs.LG) — episodic
memory inside an RL learner rather than around a language model — and it is
multi-agent value-factorisation work with no bearing on an extractive diary.
`MEMORY_RETRIEVAL_BAKEOFF.md` §5.1's structural inadmissibility is doing exactly
what it was written to do, and four sweeps of it is now itself a finding: **this
front's literature has moved somewhere our constitution forbids us to follow, and
the correct response is to stop expecting it to come back.** I would treat front 3
as a *biology-first* front from here — the ScrubJay item is on the watchlist for
its oracle, not its system — and I would sweep it every second week rather than
every week. That is a recommendation about my own cadence, not a spec change.

**FRONT 5 · WORLDS — no arm, and the one design reference is unchanged.** The
homeostatic-agent enumeration surfaced N2 (nominated) and a cluster of
position/theory work (2604.24527 Interoceptive Machine Framework, 2510.07117,
2605.07524) with no experiments between them. ForageWorld remains the design
reference for D10(b). **The open-endedness / UED half of this question was
searched and not fetched** — Efficient UED through Hierarchical Policy
Representation Learning (2602.09813) and the task-level-pairs line (2511.12706)
both frame environment *quality* as regret against a student policy **[s]**,
which is structurally the instrument `w0-too-shallow` lacks and N3 says PIC/POIC
cannot supply. **That is queued, not claimed** — I have not opened either paper,
and per week 4's own marker I am not going to write it down as though I had.

**SMALL-MODEL END — nothing, third consecutive sweep.** No sub-1M-parameter
embodied-control result in-window; the near hits are TinyML inference work and
LLM-agent efficiency.

**BIOLOGY-AS-ORACLE — one nomination (N2) and one watchlist item (ScrubJay-MEM's
Clayton & Dickinson lineage).** Both are on `GOAL.md`'s named shelf or adjacent to
it. **Motor babbling came off that shelf last sweep and is now ordered**; the
still-unmined remainder is innate reflex priors, pain as a fast signal distinct
from reward, critical periods, and play as safe rehearsal.

---

## 6. A DISCIPLINE FINDING — the failure mode inverted: this week's best paper has
## no numbers to be wrong

Week 1: *an abstract is a claim about a table.* Week 2: *a title is a claim about
a field.* Week 3: *a diagnosis of our own failure must carry the arithmetic.*
Week 4: *a claim from a search result is the search engine's claim* — the `[s]`
marker, accepted by the Review and **used for the first time in §5's UED
paragraph**, which is the only place in this file where I describe a paper I did
not open.

This week's finding is the inversion of week 4's, and it is uncomfortable in a
new direction. Week 4 reported that **three of four nomination-grade papers
reported no hardware and no wall-clock** and concluded that the literature is
becoming un-pre-priceable against `B4`. **N1 has no hardware, no wall-clock, no
parameter count, no benchmark, no seeds, and no experiments of any kind** — and
it is the strongest nomination on this desk in three weeks.

That is not a contradiction, and pretending it is would be the easy move. It is
two different epistemic goods:

- A **benchmark table** is evidence that transfers only as far as the hardware and
  regime it was measured in. `LESSONS.md` records three separate transfer
  failures on this box (int8, i8mm, RTF), and week 4 found the hardware itself is
  increasingly unreported.
- A **theorem** has no hardware. N1's Proposition 2 is either right or wrong about
  VICReg, and its algebraic core compiles in Lean 4 with zero `sorry`
  obligations — a verification status **no benchmark table on this desk has ever
  had**. What it does *not* have is any claim about whether its hypotheses hold in
  a trained `A4`, and the authors say so themselves (*"an interpretive device,
  not a claim about deployed systems"*).

**The rule this suggests, offered for `LESSONS.md` and not written by me: a
theory nomination and an empirical nomination fail in opposite directions, and
"verified" must say which.** An empirical paper's risk is that its number does
not transfer; a theory paper's risk is that its *assumptions* do not hold. Marking
both `[V]` flattens the difference. The cheap form is that a `[V]` on a theory
paper must be accompanied by the assumption list, which is why N1's table has a
row for it and why objection 2 quotes the authors against themselves.

**Second, a queue item I could not close, reported as such rather than as done.**
Week 4 called **2607.22430's full text** *"the highest-value fetch outstanding on
any front"* — the question being whether the **spectral separation margin** is
computable on a trained model, which would make it a diagnostic for `A4` and
possibly `UB.11`'s missing certificate. **I fetched the abs page and it does not
answer it**; the abstract states the margin bounds transition error but says
nothing about estimating it post hoc. **I did not reach the full text this
sweep.** Per week 3's rule that a third "still pending" is a lie by deferral,
this is its second: it is re-queued **once**, with a stated method (fetch the HTML
and search the experimental section, not the abstract), and if a sixth sweep
cannot close it, it gets dropped with cause the way conference enumeration was.
**What changed around it:** N1 makes it *less* load-bearing than week 4 thought.
Week 4 wanted 2607.22430 to justify `A4c`; N1 justifies `A4c` on independent
grounds and additionally identifies the incumbent's defect, which 2607.22430
never addressed.

---

## 7. What this report does NOT claim

- **No arm here has been run.** Every number in §2 and §3 is someone else's
  measurement on someone else's hardware — **or, for N1, no measurement at all.**
  The `T3.06` figures in §5 and §0 are OURS, quoted from the ledger row and
  recomputed with the arithmetic shown; they are context and observation, not a
  finding of this sweep and not a decision about the row.
- **No nomination is a recommendation to adopt.** `SYSTEM.md` law 3 stands.
- **Nothing here changes a spec, a threshold, a decision, or a line of code** —
  including §5's observation about `T3.06`'s comparators, which describes numbers
  already in a committed ledger row and moves nothing. The 09-06 decision on
  `t306-matched-magnitude-noise-buys-coverage` belongs to the Review.
- **N3 is explicitly not an arm** and says so in its title. It is a control and a
  caution attached to a measurement someone else already ordered.
- **N1's evidence class is theoretical and its authors disclaim the modelling
  assumptions.** It is nominated as a *selection criterion between two existing
  accepted arms*, not as a new mechanism, and that is the weakest thing it can be
  while still being worth writing down.
- **Verification is uneven and marked as such:** N1 full HTML with propositions
  and remarks quoted, **zero experiments by construction**; N2 **full abstract
  only — no seeds, no hardware, no params, no code, single author**; N3 abstract
  verified, **no numbers extractable from any page**; NE-Dreamer full HTML
  (**DMC magnitudes confirmed non-extractable**); Dreamer-CDP, ScrubJay-MEM,
  IIBalance, 2510.07117, Curiosity-Critic all **abstract-level**; the UED items
  **search-level only and marked `[s]`**.
- **Front 4's intrinsic-motivation half was swept deeply; its open-endedness half
  was not**, and §5 says so rather than letting "front 4 done" stand.
- **`SM.02` has never run for the fourth consecutive sweep**, and I have now
  reported it four times without it changing. That is a fact about the queue, not
  about the literature, and it is not mine to fix.

---

## 8. Queued for next sweep (**not before ~2026-09-07**)

1. **UED / open-endedness, properly** — 2602.09813 and 2511.12706, fetched not
   searched. The question is narrow and it is `w0-too-shallow`'s: **does the
   regret-against-a-student framing give a computable environment-discriminability
   score, and can it be read on W0 for CPU-minutes?** N3 says the two published
   difficulty metrics fail; this is the only other family that claims one. Highest-value
   fetch outstanding on any front, and it goes first.
2. **2607.22430 full text — SECOND and LAST attempt**, by HTML + experimental
   section rather than abstract. Drop with cause if it fails again (§6).
3. **Whether N1 has a Lean repository.** The paper cites Appendix D; no release is
   stated. A machine-checkable proof that is actually downloadable is a different
   object from one that is described, and it costs one fetch to find out.
4. **Front 3 moves to a two-week cadence** (§5) unless the builder or Review says
   otherwise — four sweeps of the same constitutional refusal is enough to
   re-plan rather than repeat. Its slot goes to fronts 4 and 5.
5. **Watch: the D10 firing (2026-09-01) and the 09-06 `t306` decision.** If D10
   fires as written, wk4-N1/N2 and this week's N1 all become dispatchable against
   the seated `A4` in the same week, and §2 needs re-pointing. If the owner takes
   the 52nd audit's option (a) (`BY DEFAULT`, not `BY VERDICT`), nothing here
   changes.
6. **Simulus (2502.11537) — close or drop.** Fifth sweep carried on two numbers
   that have not appeared. A sixth carry needs a reason (§3).
7. **NOT queued, deliberately:** conference proceedings (dropped, week 4);
   NE-Dreamer's DMC table (closed, null result); Dreamer-CDP (closed, demoted);
   the scale probe and Var-JEPA (resolved, week 4).
