# Language Grounding — how a word comes to mean something, and how we prove it did

> Researched 2026-08-09. Serves GOAL.md and the owner's stated end goal:
>
> *"we want this to one day listen — if you ask him to do something he does it,
> you can talk to it. This is the goal, for us to have Jack there, be able to
> communicate with him and ask him to do stuff... the end goal must still be a
> general brain who understands everything."*
>
> Companion to `UNIFIED_BRAIN_BAKEOFF.md` (whose trunk→readout→percept contract
> this document obeys without amending), `CURIOSITY_BAKEOFF.md` (which supplies
> the childhood this document attaches words to), `MEMORY_RETRIEVAL_BAKEOFF.md`
> (whose extractive-never-generative rule binds §6), and `CAPABILITIES.md`
> §C (L1–L6), which this document turns from a taxonomy into runnable specs.

**Citation hygiene, same convention as `UNIFIED_BRAIN_BAKEOFF.md`:** IDs marked
**[V]** were fetched from arxiv.org during this research pass and the
title/venue/date confirmed. IDs marked **[c]** are carried from
`CAPABILITIES.md` or another in-repo document and were **not** re-verified
here — treat their numbers as second-hand. Nothing is cited for a number
nobody saw.

---

## 0. Three findings that reframe the question

**Finding 1 — the field's own mechanistic evidence says most
language-conditioned policies are not listening, and the failure is invisible
to every metric currently in this project's ladder.**

`UNIFIED_BRAIN_BAKEOFF.md` §1.1 already logged the decisive study, but logged
it as a *fusion* result. Read again as a *language* result it is much worse
news. **"Not All Features Are Created Equal: A Mechanistic Study of
Vision-Language-Action Models" (2603.19233 [V])**, six models 80 M–7 B,
394,000+ rollout episodes: *"The visual pathway dominates action generation
across all architectures: injecting baseline activations into null-prompt
episodes recovers near-identical behavior."* And the sentence this whole
document is organised around:

> X-VLA on `libero_goal` drops 94 % → 10 % under a wrong prompt, while
> `libero_object` scores **60–100 % regardless of the prompt**.

Two policies, same benchmark family, same architecture. One is listening; one
is a very good visual prior with a text input wired to nothing. **Success rate
cannot tell them apart.** Neither can a loss curve, a retrieval accuracy, a
paraphrase-routing score, or T2.06/T2.07 as currently specified — every one of
those numbers is high for both.

The corollary that dictates §5's design: **language sensitivity is a property
of the TASK, not of the model.** A benchmark cell where only one action is
plausible cannot measure listening at any model scale. So the eval set has to
be *certified language-necessary before any arm is scored on it* — the direct
analogue of `PG.7`'s leak probe, and the reason LG.00 exists and is the first
thing that runs.

**Finding 2 — the project's cheapest grounding signal is already on disk, and
it is Jack's own diary.**

T1.13 PASSES on 2,747 real CMU/KIT clips with text (label-signal advantage
7.956, 5,590 samples, 18 distinct labels). That is the *imitation* half. The
*hindsight* half is `EpisodicMemory`: every `did` event is, by construction, a
(trajectory, description) pair produced by living. Hindsight instruction
relabelling — relabel what he DID with a description of what he did — needs no
annotator, no reward model, and no curated demo set, and it is the one
grounding mechanism whose training data grows every second Jack is alive.
§2.2 gives the literature; §6 gives the wiring; arm `A5` in §7 makes it
compete rather than be assumed.

**Finding 3 — the developmental ordering the plan assumes (childhood →
grounding → direction) is the single least-tested load-bearing assumption in
the project, and the literature does not settle it.**

The claim "you cannot ground a word in an experience the agent never had" is
*true as a definition* and *unproven as an engineering strategy*. The field's
most capable instruction-followers are built the other way round: a
language-first pretrained trunk, then control, with skills and words trained
jointly. Whether skills-first *wins on Jack's budget* is an empirical question
with a cheap answer, and §8 answers it as a bakeoff rather than an assertion.

---

## 1. What "the word means something" is going to be taken to mean

Before any survey, the operational definition, because the rest of the
document is only as good as this paragraph.

A word **means something to Jack** iff all five hold:

1. **It changes behaviour, in the direction it names.** Not "changes
   behaviour" — an agent that does something different for every string is
   sensitive to text and understands nothing.
2. **The change survives combinations never trained.** Held-out verb × object
   cells, above the language-blind baseline, per-cell, not on average.
3. **The change is destroyed by destroying the word.** Scramble the order,
   swap in a different instruction, delete it — behaviour must move to the
   pre-registered place in each case, and the three places are different.
4. **It is anchored in something he can do.** There is a `did` record in his
   log whose canonical act string contains the word, or the word decomposes
   into ones that do.
5. **The anchoring is causal, not correlational.** Established by
   intervention, never by a probe. `UNIFIED_BRAIN_BAKEOFF.md` §1.1's lesson
   verbatim: *encoded is not used, and architecture does not predict use. A
   test that measures encoding has not measured binding. Only an intervention
   has.*

Conditions 1–3 are §5, the Understanding Test. Condition 4 is §6. Condition 5
is why every metric in this document is a *difference between two
interventions on the same initial state*, never a success rate.

**And the thing that must NOT be conceded:** the counter-position — that a
sufficiently large table of (phrase → behaviour) pairs *is* meaning — is not
refuted by any experiment here and is not the project's business to refute.
What LG.05 measures is narrower and sufficient: whether Jack's behaviour on
cells nobody put in the table is better than chance, and whether the machinery
he uses to get there is the language pathway. A lookup table scores zero on
the first by construction. That is the entire content of the claim.

---

## 2. Survey

*(filled in below — demonstrated vs argued separated in each subsection)*

### 2.1 The three architectures for "ask him to do something"

They are not variants of one idea. They fail differently, cost differently,
and only one of them is a *general brain*.

| | (a) **Language → primitive** | (b) **Language-conditioned policy** | (c) **LLM planner → learned skills** |
|---|---|---|---|
| Mechanism | classify utterance into one of N library entries | instruction tokens condition continuous control end-to-end | frozen LLM emits a call sequence; a learned value/affordance gates each call |
| Composes? | only over the library | in principle over verb × object | over *sequences*, not over new atoms |
| Fails by | out-of-library requests; new verb impossible | ignoring the instruction (Finding 1) | planner hallucinating skills that do not exist |
| Free-tier | trivial | plausible at Jack's scale | plausible **iff** a success estimator exists (T2.17) |
| In this repo | `SemanticActionAnchors`, T2.06/T2.07/T2.15 | does not exist | does not exist |

The reason to be precise about this: **T2.06, T2.07 and T2.15 as written test
(a) and only (a)**, and (a) cannot deliver the owner's end goal. "Free-form
language routes to the right task" is a *routing* claim. A router that scores
1.0 on every paraphrase in the language is still a device that maps strings to
a fixed menu; ask it for something not on the menu and it returns the nearest
menu item with high confidence. Routing is worth having — it is the cheapest
arm and it may win the atomic-instruction bakeoff — but it must be *scored on
held-out cells*, where it is expected to fail, or it will look like
understanding forever. §7 therefore enters the router twice: as a candidate
arm (`A1`, frozen-LLM embeddings) and as a **control that must fail**
(`C-router-tfidf`, bag-of-words), because the gap between those two is exactly
the value the LLM adds, which is what T3.08 was written to decide.

**CORRECTION to the sentence above, 2026-09-04 (the §2.2–§11 pass, found by
cross-checking this document against `SYSTEM.md` rather than against the
literature).** `A1` may not be entered as a *candidate* arm. Its text tower is a
**frozen** encoder sitting inside Jack, and the owner's PLASTIC-ONLY decree of
2026-08-09 forecloses exactly that — the decree's own justification is
arithmetic ("a frozen tower's reshaping gain is identically zero"), so no score
`A1` posts can seat it. But `SYSTEM.md`'s hard constraint is equally explicit
that this does **not** mean exclusion: the owner's 2026-08-24 ruling replaced
*excluded* with **SCORED-AND-INELIGIBLE**, because *"an assumption that cannot
lose is not a finding"*. So `A1` runs, is measured on the same ruler as every
other arm, and its number goes to the ledger and to `CHAMPIONS.md` as a standing
challenger — it simply cannot take the seat while it is frozen. §7 enters it
that way. If it wins, that is a finding the owner is owed loudly, not a result
the harness was built to be unable to see.

### 2.2 Grounding: how words attach to sensorimotor experience

Three mechanisms exist in the literature. They differ in **what they need a
human to do**, which on this project is the only axis that decides anything:
there is no annotation budget, ever, and there never will be.

**(i) Paired supervision — someone describes the demonstrations.** The
reference points are **"Language Conditioned Imitation Learning over
Unstructured Data" (2005.07648 [V]**, Lynch & Sermanet, 2020**)**, which trains
perception, language and control end-to-end and reports that language
annotation is *"less than 1% of total data"*; **CLIPort (2109.12098 [V]**,
Shridhar, Manuelli & Fox, 2021**)**, a two-stream *semantic* and *spatial*
architecture whose single multi-task policy matched or beat single-task
policies across 10 simulated and 9 real tasks; and **CALVIN (2112.03227 [V]**,
Mees, Hermann, Rosete-Beas & Burgard, 2021**)**, the benchmark that made
*"zero-shot to novel language instructions and to novel environments and
objects"* the standard split. The 1% figure is the encouraging one and it is
still 1% of somebody's afternoon. **Jack has no annotator. This family is
admissible here only if the pairing is generated by living.**

**(ii) Hindsight relabelling — the trajectory writes its own instruction.**
**HIGhER (1910.09451 [V]**, Cideron, Seurin, Strub & Pietquin, 2019**)** extends
Hindsight Experience Replay to language: when an episode fails its instruction,
*generate the instruction it did satisfy* and learn from that, which
*"eliminat[es] the need for external expert intervention"*. This is the family
that fits, and the reason is Finding 2: **`EpisodicMemory`'s `did` channel is
already a stream of (trajectory, description) pairs, produced by living, at zero
marginal cost, growing every second Jack is alive.** §6 wires it; arm `A4` in §7
makes it compete rather than be assumed.

**(iii) Cross-situational statistics — no single scene is disambiguated, the
set is.** **Smith & Yu (2008 [V]**, *Cognition* 106(3):1558–1568**)** showed 12-
and 14-month-olds learning word→referent mappings from individually **ambiguous**
word-scene pairings, by accumulating co-occurrence across trials. This is the
cheapest mechanism in the document and the one closest to Jack's actual
situation — a parent talking while things happen, no pointing, no labels. It is
not a competing architecture so much as a *supervision schedule*, and it is what
makes the LLM-as-parent decree trainable rather than decorative.

**The trap in (ii), stated before it is built, because it is the same trap as
Finding 1 in another dress.** Relabelling with what he *did* teaches "this word
describes what happened". That is grounding only if the description is not a
function of the observation alone — otherwise the model learns the observation→
description mapping and the word is redundant at training time exactly as it is
redundant at test time. **So §5's certification is a precondition for §6's
supervision, not only for the eval set.** A hindsight channel built on
uncertified cells manufactures its own posterior collapse and calls it data.

### 2.3 Compositional generalisation — the actual test

**SCAN (1711.00350 [V]**, Lake & Baroni, 2017; ICML 2018**)** is the origin
point: seq2seq networks *"can make successful zero-shot generalizations when the
differences between training and test commands are small"* but *"when
generalization requires systematic compositional skills… RNNs fail
spectacularly."* **gSCAN (2003.05161 [V]**, Ruis, Andreas, Baroni, Bouchacourt &
Lake, NeurIPS 2020**)** moved it into a grounded grid world — adjectives
interpreted relative to world state, adverbs composed with novel verbs — and
found that a strong multimodal baseline and state-of-the-art compositional
methods *"fail dramatically."*

Two design consequences, both of which change what §5 may report:

1. **Per-cell, never on average.** A held-out set that mixes cell families is
   carried by its easiest family; the number then describes the split, not the
   model. `LG.05` reports the **per-cell table** and gates on *how many cells*
   clear the language-blind baseline, not on the mean advantage.
2. **The split must be by CELL, not by PHRASING — and this project has already
   measured the easier one and failed it.** `T2.07` (held-out grounding) is
   **FAIL** on the ledger: held-out *phrasings* scored [2,2,2] against a
   pre-registered 4/5 bar on all three seeds, while a naive-Bayes lexical
   reference scored 5/5 — the split was resolvable by token overlap and the
   model still did not transfer. The verb×object cell split is **strictly
   harder**. Therefore `LG.05` may not be dispatched until an arm exists that
   clears the phrasing split; §7 encodes that as `LG.05.depends_on = [LG.04]`
   rather than as advice.

### 2.4 The skeptical literature

**The mechanistic result already logged as Finding 1** — *"Not All Features Are
Created Equal"* (**2603.19233 [c]**, carried from `UNIFIED_BRAIN_BAKEOFF.md`
§1.1, **not** re-verified in this pass) — says the failure exists. The pass
below found the paper that says **why**, and it is the more useful half.

**CAST (2508.13446 [V]**, Glossop, Chen, Bhorkar, Shah & Levine, 2025**)** names
the mechanism, verbatim from its introduction:

> *"the future action distribution typically collapses given any single
> observation (e.g., given an observation of a chest of drawers, the only
> probable task for a robot is 'open the drawer'). Thus, even powerful models
> have little incentive to pay attention to the language command, suffering from
> posterior collapse"*

**Three things follow, and they are the spine of §5 and §7.**

- **This is a property of the DATASET, not of the model.** It restates Finding
  1's corollary with a mechanism attached: where the observation determines the
  act, attending to language earns nothing, so no architecture and no scale will
  make a policy listen.
- **The repair in the literature is on the data side.** CAST augments existing
  data with counterfactual language for the same observations and reports
  *"a 53% success rate on challenging, diverse language prompts, outperforming a
  VLA trained without CAST by 27%"* — i.e. the fix for "policies ignore
  language" was to make the corpus language-necessary. That is `LG.03`'s job,
  moved from training corpus to eval set and pre-registered.
- **The certification is therefore cheap and comes FIRST.** The condition to
  certify is not "is this task hard" but **"is more than one act plausible from
  this observation"** — which is checkable with a privileged planner and no
  learning at all. `LG.03` is minutes of CPU and it can foreclose the entire
  family before a single arm is trained.

**And one citation this document deliberately refuses to lean on.**
**Bender & Koller, *"Climbing towards NLU: On Meaning, Form, and Understanding
in the Age of Data"* (ACL 2020 [V])** argues that *"a system trained only on
form has a priori no way to learn meaning."* It is the intellectual ancestor of
`LG.00` and it agrees with this project's structure — which is exactly why it
may not be cited as evidence for it. It is an argument, and law 3 says arguments
do not decide things here. `LG.00`'s measurement decides it; this paragraph
exists so that a future reader does not mistake agreement for support.

### 2.5 Memory × language

Three facts from this repo's own ledger, which together price every arm in §7.

- **`ME.9` PASSES**: heard / said / did are recorded and **attributed**. That is
  the substrate hindsight relabelling reads, and it is the only reason (ii) is
  free here.
- **The extractive-never-generative rule binds** (`MEMORY_RETRIEVAL_BAKEOFF.md`,
  and `ME.11`'s own `kills` clause: *"any retriever that generates its answer
  instead of quoting one, however good its numbers"*). Language may **report**
  the record; it may never **compose** it.
- **`ME.11` is FAIL, and this is the constraint a naive LG design would walk
  straight into.** Cued recall by *paraphrase* — cues sharing no content word
  with the stored event — has a measured best dense ceiling of **0.250 against
  the registry's 0.80 bar**, with arm `A` at **0.0000** paraphrase recall.
  **So any grounding arm that needs to retrieve the lived episode by a
  paraphrased word inherits a component that is red on the ledger today.** §7's
  arms are therefore specified to train on the diary **offline and in bulk**
  (every `did` record, no query) rather than to *retrieve* from it at decision
  time. An arm that needs paraphrase retrieval is not forbidden — it is required
  to declare `ME.11` in `depends_on`, which makes it structurally unrunnable
  until that number moves, and that is the honest place for it to wait.

### 2.6 The ordering question

Finding 3 said the literature does not settle this. Having read it, that is
still the answer — but the pass found something sharper than "unsettled", and it
is a **control**, not an opinion.

**The strongest counter-evidence to skills-first is
`Pre-Trained Language Models for Interactive Decision-Making` (2202.01771 [V]**,
Li, Puig, Paxton, Du, Wang, Fan, Chen, Huang, Akyürek, Anandkumar, Andreas,
Mordatch, Torralba & Zhu, 2022**)**: initialising policies with a language model
and fine-tuning by behaviour cloning improved task completion by **43.6%** in
VirtualHome. Language-first wins, and it wins on a household-task benchmark much
closer to Jack than SCAN is.

**Except read the paper's other finding, which is the one that matters here:**
*"the format of the policy inputs encoding (e.g. as a natural language string vs
an arbitrary sequential encoding) has little influence."* **If an arbitrary
sequential encoding does as well as English, then what transferred was a
sequential-structure prior, not word meaning.** That is a completely different
claim from "language-first grounding wins", and every argument this project
might build on the 43.6% would be building on the wrong half.

**So `LG.06` carries a SCRAMBLED-VOCABULARY arm** — the same pretrained model,
same parameter count, same optimiser steps, vocabulary permuted by a fixed
seed — and the ordering question is only answered by the **gap between the
language arm and the scrambled arm**. If they tie, the win is structure and the
project has learned that its "childhood → grounding" ordering was never the
thing being tested. This control is taken directly from the cited paper's own
result, which is the cheapest kind of control to trust.

**The developmental evidence on the other side, with its replication record
attached, because omitting that would be the disease this file exists to
prevent.** **Smith & Gasser, *"The Development of Embodied Cognition: Six
Lessons from Babies"* (*Artificial Life* 11(1–2):13–30, 2005 [V])** is the
canonical statement that intelligence emerges from sensorimotor interaction and
that starting *"as a baby grounded in a physical, social, and linguistic
world"* is what buys flexible intelligence. The sharpest single experiment is
**Needham, Barrett & Peterman, "A pick-me-up for infants' exploratory skills"
(*Infant Behavior & Development* 25:279–295, 2002 [V])** — the "sticky mittens"
study, in which 10–14 ten-minute sessions of *simulated reaching* changed
pre-reaching infants' subsequent object exploration, i.e. motor experience
preceded and enabled a perceptual competence. **It has a mixed replication
record**: a pre-registered Swedish replication reports that active training did
**not** increase reaching and grasping, and there is a published critical
appraisal of the paradigm. It is cited here as *motivating* and explicitly not
as settled; anyone who wants to lean on it should read the appraisal first.

**Net.** One verified result says language-first wins by a lot and may be
winning for a reason that has nothing to do with language; one developmental
tradition says skills-first, on evidence that is partly contested. That is
precisely the shape law 3 was written for. §8 runs it.

---

## 3. Recommended architecture

**Route (b) — a language-conditioned policy trained on Jack's own lived data via
hindsight relabelling — with route (a) kept as the incumbent arm and its
bag-of-words twin kept as a control that must fail.**

The reasoning is budget and constitution, not preference:

- **(c) LLM-planner-over-skills is out**, and not on principle: it needs a
  success estimator per skill (`T2.17`) that does not exist, and its failure
  mode — a planner emitting calls to skills Jack does not have — is unmeasurable
  until the skill library is real. It re-enters when `T2.17` lands.
- **(a) routing cannot deliver the owner's end goal** (§2.1) but is the cheapest
  thing on the board and *may still win the atomic-instruction cell*. It stays,
  scored on held-out cells where it is expected to fail.
- **(b) is the only family that composes over verb × object**, and it is the
  only one whose training data Jack generates by living.

**What sits inside him and what does not.** The parent LLM speaks *into the
world*; its tokens arrive through his ears. Nothing frozen conditions his policy
— the text tower inside the policy is plastic, as `T2.06`'s shipped path already
is. The frozen-encoder arm exists only as the scored-and-ineligible challenger
of §2.1's correction.

## 4. Why the LLM stays frozen, and what "swappable" costs

The decree is the owner's (2026-08-09) and is not re-litigated here. What this
pass owes it is the **price tag**, per `SYSTEM.md`'s rule that an owner
directive is recorded with its strongest counterargument and cost:

- **The cost is (2202.01771 [V])'s 43.6%.** A frozen out-of-process parent
  cannot contribute its representations to the policy, so whatever that
  initialisation buys, Jack forgoes by construction. `LG.06`'s scrambled arm is
  what tells us whether the forgone quantity is *meaning* or *sequence
  structure* — and if it is the latter, the decree costs much less than it
  looks, because a sequence prior is cheap to obtain elsewhere.
- **The benefit is measurable and already specced.** `LG.10` makes swappability
  falsifiable: swap the frozen model, meaning must survive and style must change.
  A frozen model that cannot be swapped without changing what he *means* was
  never outside him.

## 5. THE UNDERSTANDING TEST

Conditions 1–3 of §1, operationalised. The whole design turns on one idea:
**every number is a difference between two interventions on the same initial
state, and the three destruction interventions must land in three DIFFERENT
pre-registered places.**

| intervention | where behaviour must go | what it rules out if it does not |
|---|---|---|
| **scramble word order** | to the language-blind baseline | it is a bag of words |
| **swap in a different valid instruction** | to *that instruction's* behaviour | it is sensitive to text, not to meaning |
| **delete the instruction** | to the language-blind baseline | the instruction channel is decorative |

Two of the three share a destination on purpose: the informative comparison is
**swap vs scramble**. A model that treats a scrambled instruction like a
*different* instruction is doing lookup on a token bag; a model that treats a
swapped instruction like a *deleted* one has an instruction channel wired to
nothing. Success rate cannot separate any of these; three destinations can.

**Scoring.** Per held-out (verb, object) cell, the advantage over the
language-blind baseline on that cell. Gate: **≥ K of N cells** clear their own
per-cell band, at ≥ 3 seeds. Never the mean (§2.3).

**Ordering.** `LG.03` certifies the cells → `LG.04` finds an arm that can clear
the *phrasing* split → `LG.05` asks the *composition* question. Skipping to
`LG.05` is dispatching a spec whose predecessor is FAIL on the ledger (`T2.07`).

## 6. Instructions and outcomes in the episodic record

The wiring, in one paragraph, because the mechanism is small and the trap is not.

Every `did` record already carries a canonical act string and the trajectory
that produced it (`ME.9`). Hindsight relabelling reads that stream **offline and
in bulk** — never as a query, per §2.5's `ME.11` constraint — and emits
(instruction, trajectory) pairs whose instruction is *what he actually did*
rather than *what he was told*. Parent utterances heard during the episode enter
as a second, weaker channel under the cross-situational schedule of §2.2(iii):
the referent of a word is whatever is common across the episodes it was heard
in, and no single episode is required to disambiguate anything.

**The trap, restated as a build rule.** A hindsight pair drawn from a cell where
the observation determines the act is a pair in which the instruction is
redundant — training on it manufactures posterior collapse. So the relabelling
channel is **restricted to `LG.03`-certified cells**, and the fraction of the
diary that survives that restriction is itself a reported number: if it is near
zero, Jack's life does not contain language-necessary situations, which is a
finding about the *world* and belongs on `w0-too-shallow`, not a bug in the
channel.

## 7. The grounding bakeoff — registry entries

> **STATUS, 2026-09-04. These four blocks are DRAFTS. They are NOT in the
> registry.** They were AST-parsed against the live `experiments/protocol.py`,
> checked for id collisions against `registry.BY_ID` (`LG.03`–`LG.06` are free;
> `LG.00`/`LG.01`/`LG.02`/`LG.10`/`LG.11` are taken) and every `depends_on`
> resolves. Registration is a separate unit under `INTEGRATION_QUEUE.md`'s
> 5-step protocol — step 1's cross-check belongs to *that* iteration, and this
> document does not get to skip it by having done a partial one.
>
> **What the PARTIAL cross-check already found, recorded so the registering
> iteration starts from it rather than repeating it — and both findings changed
> a draft above, which is the point of doing it before registration rather than
> after.** No refutation was found; two conflicts were.
>
> - **`CAPABILITIES.md` §C **L2** is this family's ancestor and carries a
>   control these drafts had missed:** *"held-out verb×object cells above
>   chance — GRID MUST BE DESIGNED INTO TRAINING DATA BEFORE TRAINING. Control:
>   untrained verb must fail."* The grid clause is already honoured (`LG.03`
>   certifies before any arm trains); the **untrained-verb control was absent
>   from `LG.05` and has been added**. Without it, held-out cells can be solved
>   by the object alone and nothing compositional is measured. L2 also cites
>   **CALVIN (2112.03227)**, independently re-verified **[V]** in §2.2 of this
>   pass — the two documents agree on the benchmark, which is reassuring rather
>   than load-bearing.
> - **`CURIOSITY_BAKEOFF.md`'s F2 branch names `A4`'s confound before `A4`
>   existed:** *"the fix is hindsight relabeling density, not curiosity."*
>   Relabelling multiplies training pairs, so an `A4`-over-`A3` gap is
>   otherwise indistinguishable from more data. **`C-DENSITY` has been added to
>   `LG.04`** — `A3` at `A4`'s pair count, resampled rather than relabelled.
> - **`T2.16` ("Hindsight goal-reaching — the flow-matching weld") already owns
>   this mechanism in another family** and has **no ledger row** (never run).
>   Not a conflict and not a blocker, but `A4` and `T2.16` should share an
>   implementation rather than grow two; whichever runs first should be written
>   to be imported by the other.
> - **Also mapped, for whoever registers:** `L1` (atomic instructions) is what
>   `T2.06`/`T2.15` already test; `L2` is `LG.05`; `L4` is `D1`/`LG.06`'s
>   territory; `L3`, `L5` and `L6` are **unclaimed by any spec in this
>   document** and remain open inventory.

`LG.03` must PASS before any arm runs, and every arm declares it in
`depends_on`, so `protocol.blocked_by()` structurally prevents scoring an arm
against an uncertified cell set. That is `ME.11.0`'s pattern, and §11 explains
why it is load-bearing rather than tidy.

```python
    # ── THE LG GROUNDING BAKEOFF: cells, arms, claim, ordering ───────────
    # One shared fixture generates, per seed: the (verb x object) cell grid,
    # a language-blind twin trained on identical observations, and the
    # privileged-planner reachability certificate. The fixture hash is written
    # into every arm's metrics so two arms cannot be scored on different cells.

    Spec("LG.03", 2, "The command cells are language-necessary — certified "
                     "before any arm is scored",
         hypothesis="Every (verb, object) cell RETAINED for LG.04/LG.05 is "
                    "certified on two legs, PER CELL, never on average: "
                    "(1) NECESSITY — a language-blind policy trained on the "
                    "identical observation stream, instruction channel zeroed, "
                    "sits inside its pre-registered chance band on that cell; "
                    "and (2) PLURALITY — from the cell's initial observation a "
                    "privileged planner reaches at least two DISTINCT cell "
                    "targets, so more than one act is achievable from what he "
                    "can see. >= 12 cells spanning >= 4 verbs and >= 4 objects "
                    "must survive both legs, with every verb and every object "
                    "represented at least twice.",
         falsified_by="Fewer than 12 cells surviving, or any verb/object "
                      "falling below two retained cells — this world does not "
                      "admit language-necessary commands at this horizon. The "
                      "LG bakeoff is then VENUE-BLOCKED, not model-blocked, and "
                      "the reading is routed to w0-too-shallow as an "
                      "instrument. A cell that fails PLURALITY is excluded and "
                      "the exclusion logged with the planner's two targets.",
         null_baseline="The language-blind policy, identical observations, "
                       "instruction channel zeroed. Its per-cell accuracy "
                       "DEFINES the exclusion; it is the leg that makes "
                       "retention falsifiable rather than curated.",
         metric="retained_cells", budget=Budget.CPU,
         depends_on=["ME.9"], seeds=3,
         control="THE PLANNER, STRIPPED: re-run the plurality leg with the "
                 "target identity withheld from the planner. It must FAIL to "
                 "reach both targets. If it still reaches them, 'achievable' "
                 "was being read out of the world state rather than chosen, "
                 "and the plurality certificate is void.",
         kills="Any LG arm scored on cells where the observation alone "
               "determines the act. CAST (2508.13446): 'the future action "
               "distribution typically collapses given any single "
               "observation... even powerful models have little incentive to "
               "pay attention to the language command'. Pre-registered out, "
               "before an arm is trained, at minutes of CPU.",
         notes="LG.01's shape moved from Q&A to BEHAVIOUR; PG.7's leak probe "
               "one layer further out. The plurality leg is the half that is "
               "new: necessity alone certifies that the blind twin fails, "
               "which a merely IMPOSSIBLE cell also satisfies. "
               "  COVERS: language (parent) (fixture)"),

    Spec("LG.04", 3, "The grounding bakeoff: five arms, one certified cell set",
         hypothesis="Among arms that map an utterance to behaviour on the "
                    "LG.03-certified cells at matched optimiser steps AND "
                    "matched environment steps, at least one beats the "
                    "language-blind null by >= 3 sigma on held-out PHRASINGS, "
                    "and the winner beats the runner-up by the pre-registered "
                    "margin. Arms: A0 language-blind (the null); A1 frozen-LLM "
                    "embedding router (SCORED BUT INELIGIBLE — a frozen tower "
                    "inside Jack, foreclosed by the PLASTIC-ONLY decree, run "
                    "and recorded because an assumption that cannot lose is "
                    "not a finding); A2 the shipped plastic text tower router "
                    "(T2.06's incumbent); A3 a language-conditioned policy "
                    "trained end-to-end; A4 = A3 plus hindsight relabelling "
                    "from the diary's did channel (HIGhER, 1910.09451).",
         falsified_by="No arm clears the 3-sigma learning gate: VOID, not a "
                      "verdict — two non-learners cannot arbitrate an "
                      "architecture (T2.02's law). The seat stays UNDECIDED "
                      "and the repair is the arm, not the ranking.",
         null_baseline="A0, the language-blind policy: identical observations "
                       "and identical budget, instruction channel zeroed.",
         metric="best_arm_advantage_sigma", budget=Budget.CPU_LONG,
         depends_on=["LG.03", "ME.9"], seeds=3,
         control="TWO, and both must fail. C-TFIDF, a bag-of-words "
                 "nearest-name router with no training at all, MUST FAIL on "
                 "the held-out phrasings — if it passes, the cells are "
                 "resolvable by token overlap and every arm's score is lexical "
                 "(T2.07's naive-Bayes reference scored 5/5 where the model "
                 "scored 2/5; this control is that finding made mandatory). "
                 "C-DENSITY, A3 trained on the SAME NUMBER of (instruction, "
                 "trajectory) pairs A4 receives — resampled, not relabelled — "
                 "must fail to reach A4. Hindsight relabelling MULTIPLIES "
                 "training pairs, so an A4-over-A3 gap is otherwise "
                 "indistinguishable from more data (CURIOSITY_BAKEOFF.md's F2 "
                 "branch names this exact confound: 'the fix is hindsight "
                 "relabeling density, not curiosity').",
         kills="The router family as an ANSWER to the owner's end goal, if A3 "
               "or A4 wins: a device that maps strings to a fixed menu cannot "
               "compose over verb x object, whatever it scores. And if A2 "
               "wins, it kills the assumption that end-to-end conditioning is "
               "worth its compute at Jack's scale.",
         notes="This is the ring CHAMPIONS.md's 'Language grounding (word -> "
               "lived skill)' seat has never had (ARENA: NONE, one of three "
               "UNFALSIFIABLE seats). A1's ineligibility is recorded per "
               "SYSTEM.md's scored-and-ineligible rule, the way LC.04 already "
               "treats sb3-ppo. Arms train on the diary OFFLINE AND IN BULK, "
               "never by query: ME.11 is FAIL at 0.250 paraphrase recall "
               "against a 0.80 bar, so an arm that RETRIEVES by paraphrase "
               "must declare ME.11 in depends_on instead. "
               "  COVERS: language (parent) (claim)"),

    Spec("LG.05", 4, "The Understanding Test: three destructions, three "
                     "different destinations",
         hypothesis="On HELD-OUT (verb, object) cells never trained together, "
                    "the LG.04 winner beats the language-blind baseline PER "
                    "CELL on >= 8 of 12 cells at >= 3 seeds; AND the three "
                    "destruction interventions land in three pre-registered "
                    "and DIFFERENT places: scrambling word order moves "
                    "behaviour to the language-blind baseline, deleting the "
                    "instruction moves it to the same place, and SWAPPING in a "
                    "different valid instruction moves it to THAT "
                    "instruction's behaviour.",
         falsified_by="Fewer than 8 of 12 held-out cells clearing their own "
                      "band — composition is not there, whatever the average "
                      "says (gSCAN 2003.05161: strong baselines 'fail "
                      "dramatically' exactly here). OR swap and scramble "
                      "landing in the SAME place: a model that treats a "
                      "scrambled instruction as a different one is doing "
                      "lookup on a token bag, and a model that treats a "
                      "swapped one as deleted has a channel wired to nothing.",
         null_baseline="The language-blind baseline, per cell. The claim is a "
                       "per-cell difference between two interventions on the "
                       "same initial state, never a success rate.",
         metric="cells_clearing_band", budget=Budget.CPU_LONG,
         depends_on=["LG.03", "LG.04"], seeds=3,
         control="TWO. (1) THE LOOKUP TABLE: an arm given the full training "
                 "cell set as an explicit (phrase -> behaviour) table must "
                 "score ZERO on held-out cells by construction — if the real "
                 "arm scores like the table, nothing was learned that a table "
                 "does not already contain. (2) THE UNTRAINED VERB "
                 "(CAPABILITIES.md section C, L2, verbatim: 'Control: "
                 "untrained verb must fail'): a verb held out of training "
                 "ENTIRELY, not merely out of a cell, must fail — otherwise "
                 "the held-out cells are being solved by the object alone and "
                 "no composition was measured.",
         kills="The claim that Jack understands an instruction rather than "
               "recognising one. Conditions 1-3 of this document's section 1 "
               "stand or fall here.",
         notes="Deliberately downstream of LG.04: T2.07 already FAILED the "
               "easier held-out-PHRASING split ([2,2,2] vs a 4/5 bar) while a "
               "naive-Bayes lexical reference scored 5/5, so dispatching the "
               "harder CELL split against the same tower would be paying for a "
               "predictable red. "
               "  COVERS: language (parent) (claim)"),

    Spec("LG.06", 3, "The ordering experiment: does skills-first buy anything, "
                     "and is it the WORDS that transfer?",
         hypothesis="At matched optimiser steps and matched environment steps, "
                    "three orderings are raced on the LG.03-certified cells: "
                    "O1 skills-first (childhood, then language), O2 "
                    "language-first (a pretrained-sequence-initialised policy, "
                    "then control), O3 joint. The winner is separated by the "
                    "pre-registered margin — AND O2 is accompanied by O2s, an "
                    "identical arm whose vocabulary is permuted by a fixed "
                    "seed. The ORDERING question is answered by O2 - O1; the "
                    "MEANING question is answered by O2 - O2s.",
         falsified_by="O2 and O2s tie. Then whatever language-first transfers "
                      "is a sequential-structure prior and not word meaning — "
                      "2202.01771's own finding that 'the format of the policy "
                      "inputs encoding (natural language string vs an "
                      "arbitrary sequential encoding) has little influence' — "
                      "and this project's childhood-then-grounding ordering "
                      "was never the thing being tested.",
         null_baseline="The language-blind policy at the same budget, carried "
                       "from LG.03, so all three orderings are scored against "
                       "one floor.",
         metric="ordering_gap_sigma", budget=Budget.CPU_LONG,
         depends_on=["LG.03"], seeds=3,
         control="O2s, the SCRAMBLED-VOCABULARY twin. It is a control in the "
                 "strict sense: if it matches O2, the language arm's advantage "
                 "was not about language, and any conclusion drawn from O2 - "
                 "O1 alone would have been drawn from the wrong half of the "
                 "effect.",
         kills="The developmental ordering this project assumes without "
               "testing (Finding 3). A tie between O1 and O2 says the ordering "
               "does not matter at Jack's budget and the cheaper one wins by "
               "law 3's TIE rule; an O2 win with O2s tied says the whole "
               "framing was wrong.",
         notes="Matched optimiser steps AND env steps, both reported "
               "(D1_CONTROL_ARCHITECTURE's lesson). O2 does NOT put a frozen "
               "model inside Jack: it initialises a PLASTIC policy from "
               "pretrained weights and continues to train them, which the "
               "PLASTIC-ONLY decree permits. An arm that froze them would be "
               "A1's situation and would be scored-but-ineligible. "
               "  COVERS: language (parent) (claim)"),
```

## 8. THE ORDERING EXPERIMENT

`LG.06` above is the experiment; this section records the two things about it
that are easy to get wrong.

**First, what "language-first" is allowed to mean here.** It may **not** mean a
frozen tower — that is `A1`'s situation and the decree forecloses it. It means
initialising a **plastic** policy from pretrained weights and continuing to
train them. This is not a loophole: the decree's stated arithmetic is about
*reshaping gain being identically zero*, which is a statement about frozen
weights, not about where weights came from.

**Second, the result this experiment is most likely to produce, named in
advance so nobody is surprised into re-interpreting it.** Given (2202.01771
[V])'s own encoding-format finding, **O2 ≈ O2s is a live and arguably likely
outcome.** That is not a null result — it is the finding that the field's
headline evidence for language-first is evidence for something else, measured on
our substrate. It should be reported as loudly as a win.

## 9. Cost, against free compute only

Everything in §7 is **CPU**. There is no GPU line in this family, deliberately:
the certification is arithmetic and the arms are small, and the project's
standing lesson is that a dying free quota does not make a dispatch worth
making.

| spec | budget | class | what dominates the cost |
|---|---|---|---|
| `LG.03` | `Budget.CPU` | `cpu<10min` | one blind-twin fit per cell + planner rollouts; no learning in the plurality leg |
| `LG.04` | `Budget.CPU_LONG` | `cpu<2h` | five arms × 3 seeds; A3/A4 dominate |
| `LG.05` | `Budget.CPU_LONG` | `cpu<2h` | the winner re-run under 3 interventions × 12 cells × 3 seeds |
| `LG.06` | `Budget.CPU_LONG` | `cpu<2h` | four orderings × 3 seeds at matched steps |

**The wall-clock numbers are NOT stated here, and that is a decision rather than
an omission.** This box has no measured rate for any of these arms, `history[]`
carries no per-spec durations, and the repo's own CPU accountant records a
median allowance/actual ratio of **257×** precisely because somebody estimated
instead of measuring. `LG.03` is the cheapest spec in the family and it is also
the one that makes the others measurable: **implement and run it first, and the
other three get costed off a real duration instead of a guess.**

**One live scheduling fact the implementer will hit.** `cpu<2h` first-run specs
are currently refused past ~6.25% of a CPU day by the day meter
(`cpu48h-class-self-forecloses-the-day-meter`, DUE 09-08). `LG.03` at
`cpu<10min` is unaffected. `LG.04`–`LG.06` should be expected to need either
that row's resolution or an early-in-day slot.

## 10. What this document does not settle

1. **Whether W0 contains any language-necessary cell at all.** `LG.03` is
   designed to answer it cheaply and to answer it as a *venue* verdict, but the
   honest expectation, given that ten instruments already say W0 is too shallow
   and five specs are pilot-blocked against it, is that **`LG.03` may fail**.
   That is why it is 10 minutes and not 2 hours.
2. **Whether the seat should be filled at all before W1.** If `LG.03` fails,
   this family becomes W1 content and joins the queue behind the world design.
3. **The counter-position from §1** — that a sufficiently large (phrase →
   behaviour) table *is* meaning — remains unrefuted and out of scope. `LG.05`'s
   lookup-table control measures the narrower thing and nothing more.
4. **Whether the parent's utterances should be scripted or generated.** Every
   spec above is agnostic; `LG.02` already runs advisors through the world
   channel and could supply the pattern.
5. **The CHAMPIONS tie-break itself.** This document is an *input* to
   `champions-language-grounding-arena` (DUE 09-07) and deliberately does not
   pre-empt it — see §11.

## 11. What this changed about the machine

**(a) It converts an inventory debt into a written ring — which is the input the
`champions-language-grounding-arena` row was waiting for.** That row's tie-break
is between naming `LG.00` as the seat's arena (the 51st audit's reading) and
keeping `ARENA: NONE` with *"an unwritten grounding bakeoff"* as inventory debt
(this file's reading). **The bakeoff is no longer unwritten.** `LG.04` is a
drafted, cost-classed, control-carrying arena that decides *which grounding
approach holds the seat* — which is what the seat's own challenger cell asks
for, and which `LG.00` (a puppet test, not an approach race) cannot do. The
Review still owns the tie-break; what has changed is that the second option is
now a draft with a cost rather than a promise.

**(b) A generalisable lesson about pilots, and it is a 1-vs-5 comparison already
on this disk.** Five specs are PILOT-BLOCKED today (`SH.02`, `SM.03`, `DP.04`,
`LC.07`, `T2.11`). In every one, the venue fault was real, was found by a pilot,
and the pilot's finding lives in a **docstring** — gate-provisional furniture
whose `run()` refuses, with `_GATES_FROZEN` still `False` and no ledger row. The
one venue certification in this repo that was written as a **registered fixture
spec** — `ME.11.0` — PASSED, produced a ledger row, and its numbers
(`lexical_null_recall = 0.000`, `oracle_ceiling = 1.000`) are quotable by every
downstream arm because they are *on the scoreboard*. **`LG.03` is deliberately
the second of those and not the sixth of the first kind:** if this world cannot
support a language-necessary command, that is a *measurement about the world*
and it belongs in the ledger where `w0-too-shallow` can cite it, not in a
docstring that only the next reader of that file will ever see.

**(c) It priced three arms against a red component before anyone built them.**
`ME.11`'s 0.250 paraphrase-recall ceiling would have been discovered *inside*
`LG.04` as a mysteriously weak arm. §2.5 turns it into a `depends_on` rule.
