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

### 2.2 Grounding: how words attach to sensorimotor experience

### 2.3 Compositional generalisation — the actual test

### 2.4 The skeptical literature

### 2.5 Memory × language

### 2.6 The ordering question

---

## 3. Recommended architecture

## 4. Why the LLM stays frozen, and what "swappable" costs

## 5. THE UNDERSTANDING TEST

## 6. Instructions and outcomes in the episodic record

## 7. The grounding bakeoff — registry entries

## 8. THE ORDERING EXPERIMENT

## 9. Cost, against free compute only

## 10. What this document does not settle

## 11. What this changed about the machine
