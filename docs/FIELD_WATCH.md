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

**Sweep date:** 2026-08-11 · **Window:** ~2026-02 → 2026-08 (6 months)
**Scout:** field watch, week 2.

**This is a GAP-CLOSING sweep, and that choice needs stating up front.** The
first sweep was **2026-08-10 — one day ago**, and its own §6 queued the next one
for 2026-08-17. A one-day delta against a six-month window is not a new sweep:
re-running fronts 1–5 broadly would have re-read the same papers and produced
the *appearance* of coverage. So this sweep executed the previous one's queue
instead — the three constitutional senses it never searched, the four abstracts
it listed without opening, and the biology citation whose fetch failed.

That turned out to be the right call for a reason that was not visible when the
queue was written: **in the 24 hours between the two sweeps, the builder shipped
and PASSed `SM.01` (smell) and `TA.01` (taste)** — the exact senses the last
sweep declared it had not searched. The field watch was one day from being
overtaken on the front it had already flagged as its largest hole.

---

## 0. Coverage — what was actually searched, so the gaps are visible

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **SMELL** (queued #1) | olfactory navigation/plume-tracking RL, turbulent search, intermittency, released environments | **full HTML + quantitative results** for the nomination; abstracts for 2 siblings |
| **TASTE** (queued #1) | computational gustation, CTA models, chemosensory embodied agents; **plus primary-source verification of two `TA.02` citations carried as [k]** | abstracts + multiple independent secondary sources |
| **VOICE** (queued #1) | emergent communication, positive listening, causal-influence metrics, 2026 work | abstract fetched; compared against `VO.02`'s existing controls |
| Biology-as-oracle (queued #3) | the replay paper last week could not fetch | **full text via PMC — resolved and promoted** |
| Watchlist backlog (queued #2) | the four unopened abstracts | **all four fetched; two removed as off-target** |
| Conference proceedings (queued #4) | ICML/NeurIPS/ICLR 2026 accepted lists | **attempted, low yield — see §4** |
| Fronts 1, 2, 3 (cores, fusion, memory) | **NOT re-swept — deliberate, see above** | carried forward unchanged from 2026-08-10 |

**Known gaps, stated so nobody assumes coverage:**
- Fronts 1/2/3 carry last week's state verbatim. If a core/fusion/memory result
  landed in the last seven days, this sweep would not have seen it.
- The olfaction nomination's code is **announced but not released** (§1, N1).
- Garcia & Koelling 1966 is a two-page 1966 *Psychonomic Science* article that is
  not freely available; it is verified to **secondary-source** level only (§2).
- No non-English sources.

---

## 1. NOMINATIONS

Two. Both are **CPU-only and cheap** — neither needs the GPU quota. Each states
the source, what is **[V]**erified (fetched and read) versus **[c]**laimed,
which spec it enters, its cost on **our** substrate, and — steelmanned both ways
— why it might win and why it might lose.

---

### N1 — The whiff clock: `SM.02` may be about to measure its own observation vector

**Sources — three independent 2026 groups, converging on one variable:**
- *Clock-state olfactory search in turbulent flows using Q-learning: the geometry
  of plume recovery* — [arXiv:2605.15938](https://arxiv.org/abs/2605.15938),
  2026-05-15 (Rando, Heinonen, Qi, Seminara). **Full HTML read.**
- *Smart strategies to navigate turbulent odor plumes reorienting to local wind*
  — [arXiv:2605.21329](https://arxiv.org/abs/2605.21329), 2026-05.
- *Emergence of a Flow-Assisted Casting Strategy for Olfactory Navigation via
  Memory-Augmented RL* — [arXiv:2605.18881](https://arxiv.org/abs/2605.18881),
  2026-05-16 (Zhao, Zhao, Bian, Li).

**None is cited anywhere in this repo.** `odour.py` cites FlyGym, Celani/
Villermaux/Vergassola, Farrell and GADEN — good, primary, and all about **the
field**. This is the adjacent literature about **the agent that has to use it**,
and it is the half the module's own docstring does not cover.

**The verified claim [V].** Rando et al. train a **tabular Q-learner** whose
entire memory is *a running clock since the last whiff* — incremented each
timestep odour is below threshold, reset to 0 on detection. Measured on DNS
turbulent-channel-flow data (2598 timesteps, 123×27 grid):

| | value |
|---|---|
| success rate | **≥ 90 %** across sparsity settings, → ~100 % at low sparsity |
| vs. `cast-and-surge` biomimetic baseline | **significantly outperforms** |
| vs. Bayesian optimal-policy reference | **near-matches**, with far simpler memory |
| state space | **500 states** (single-Q, horizon H = 500); 999 (two-Q) |
| training | 3 × 10⁶ episodes, ε-greedy 1.0 → 10⁻⁴ |

The learned strategy is interpretable and reproduces the insect repertoire —
**surging, casting, downwind return**. The two other groups reach the same place
independently: 2605.21329 uses "elapsed time since the last odor detection" as
its *single* internal variable and finds performance peaks at an *intermediate*
wind-memory integration time; 2605.18881 finds a **non-monotonic** dependence of
navigation speed on memory length. Three groups, three methods, one state
variable.

**Why this bears on us, specifically.** `experiments/odour.py` defines the nose
Jack actually carries:

```python
OBS_DIM = 2 * C + C          # 12 floats at C = 4
# [left C, right C, d(mean)/dt C]
```

Bilateral concentrations plus a **one-step temporal derivative**. There is no
blank-duration state anywhere in the vector, and `OdourSensor` is documented as
owning "no clock". Meanwhile the module's own docstring reports this world's
measured intermittency at **roughly 40–55 % blanks** — so on nearly half of all
samples the derivative is computed across a blank, where it is zero and carries
no information about *how long* the blank has lasted. That duration is precisely
the quantity all three papers identify as sufficient.

**Which spec it enters.** `SM.02` ("smell finds what vision cannot see"), in two
parts — and the second is the load-bearing one:

1. **As an observation arm.** A `+C` whiff-clock (per channel, time since that
   channel last exceeded threshold) against the current 12-float vector. One
   changed term, `odour.py`'s existing isolation discipline. A third arm —
   **wind-relative action framing** with an exponential wind-memory kernel
   (2605.21329) — is available but Jack has no wind sense in the odour path, so
   it implies a spec rather than fitting one.

2. **As a cheap CPU pre-gate on the GPU run.** `SM.02` is `Budget.GPU`, and its
   `kills` field is unambiguous: *"The odour modality. A sense whose ablation
   column is placebo-indistinguishable loses its parameters... this document
   carves no exception for a constitutional sense."* A tabular Q-learner on the
   clock state, run against **our** `PuffField`, is a reference arm simple enough
   that **its failure indicts the task** — `LESSONS.md`, "when the simplest
   possible learner also fails, the TASK is broken". If a 500-state table cannot
   find the source in our field, `SM.02` cannot distinguish "smell is decorative"
   from "this field is not navigable by anything" — and the default action on
   that ambiguity is deletion of a constitutional sense's wiring.

This is the same shape as last week's N1 certificate, one front over: a positive
control that makes a null result attributable to the mechanism rather than to
the fixture.

**Cost on our substrate.** The cheapest nomination in either sweep. A 500-state
Q-table over 3 × 10⁶ episodes is a NumPy loop — **minutes on 4 ARM cores at
`nice 19`**, no GPU, no new dependency. The observation arm is `+4 floats` and a
per-channel counter. The `PuffField` and `mj_ray` occlusion already exist and are
certified by `SM.01` (PASS, 3.9 s).

**Why it might WIN (falsifiable).** Run the tabular reference arm on our `O2`
field before `SM.02` burns GPU. Two discoverable outcomes, both valuable: if the
table succeeds with the clock and fails without it, then the whiff clock is
load-bearing in *this* world and `SM.02`'s current 12-float vector is
under-specified — caught before a GPU run whose kill criterion deletes an
encoder. If the table fails *with* the clock, our field is not navigable and
`SM.02` was never going to measure smell at all.

**Why it might LOSE (steelmanned).** Four real ways:
1. **Our field is far less intermittent than theirs, and that may be decisive.**
   Farrell's field data is 83–90 % blank; `odour.py` measures this world at
   **40–55 %**. At 40 % blanks a reactive bilateral gradient may simply suffice,
   and the clock buys nothing — which is exactly the hypothesis `SM.02` already
   tests with `O1` (static field) as its control. If `O1` is as good as `O2`,
   this whole literature is inapplicable to us and `SM.01`'s notes say so
   already.
2. **A hand-designed feature is in tension with `GOAL.md`.** "PLASTIC ONLY —
   nothing inside him is frozen" and the capability target both lean against
   engineering a sufficient statistic the network could derive. Jack's policy is
   not a 500-state table: it has 348-dim proprioception and, if recurrent, can
   in principle construct its own blank-duration state. Adding the clock by hand
   may be solving a problem the architecture should solve, and would then be a
   permanent hand-hold in a constitutional sense.
3. **Substrate gap, per `LESSONS.md`.** Every number above is a 2-D point agent
   on DNS turbulence data with one source and no body, no other senses, no
   contact physics. Jack is a humanoid. "What transfers is the protocol, not the
   ceiling" applies here at least as hard as it did to last week's N1.
4. **`SYSTEM.md`: no new organ without a scar — and `SM.02` has not run.**
   Nothing has failed. This is a pre-emptive nomination, and that is the
   strongest argument against it. The counter, stated so the builder can weigh
   it rather than take my word: the pre-gate is CPU-minutes and the run it
   guards is GPU-hours with an irreversible `kills` clause, so the asymmetry is
   unusually lopsided. But it is still a nomination against an unscarred spec,
   and that is a real cost under this project's own rules.

**Code status [c]:** Rando et al. state *"the dataset and code will be shared
shortly."* It is **not released**. Nothing above depends on their code — the
method is a Q-table and four lines of state update — but any claim that we could
reuse their environment is unsupported today.

---

### N2 — Replay is prioritised by prediction ERROR, and reward magnitude is a published control that FAILS

**Source:** *Post-learning replay of hippocampal-striatal activity is biased by
reward-prediction signals* — Roscow, Howe, Lepora & Jones (University of
Bristol), **Nature Communications**, `s41467-025-65354-2`, 2025-11-24. Open
preprint: [bioRxiv 10.1101/716290](https://www.biorxiv.org/content/10.1101/716290v3).

**This resolves last week's watchlist entry**, which carried the flag *"UNVERIFIED
— the fetch failed. Do not cite until read."* Full text now read via PMC
(`PMC12644820`). The finding is **sharper than last week's guess**: last week's
line said "biased by reward-prediction signals"; the measured result is
specifically **reward-prediction ERROR**, and **reward magnitude specifically
fails**.

**The verified claim [V].**

| | |
|---|---|
| species / n | adult male Lister hooded rats; **6** behavioural (22 sessions), **3** ephys (17–20 sessions) |
| units | **617** CA1, **1406** striatal |
| design | probabilistic arms (75/50/25 %, then revaluation to 87.5/12.5 %) so that reward receipt yields low / medium / high RPE **per arm** — this is what dissociates RPE from reward |
| **RPE-prioritised replay** | Q-learning error score **significantly improved, p < 0.05** |
| **reward-magnitude-prioritised replay** | **no improvement over baseline, p > 0.05** |
| controls | shuffled-data control eliminates the effect; holds in **all 6** animals and all state–action pairs |
| effect size | "modest but consistent"; significant at **1 replay event per session**, persisting over 22 sessions |
| timescale | offline rest/sleep **between daily sessions** |

**Why this one is worth the builder's attention.** The design contains a
**control that must fail and does** — reward-biased replay, run alongside
RPE-biased replay in the same animals on the same data, showing no improvement.
That is the Garcia & Koelling shape `TA.02` already prizes: *a
control-that-must-fail, published in advance by someone else.* It is also the
cheapest kind of evidence to import, because the control comes with it.

**Which spec it enters.** `NEEDS_AND_DEATH.md` §3.4, stage **S1** — the SIESTA
latent-rehearsal stage — whose sampling rule is currently declared as:

> *"Sample transitions from the compressed lifetime buffer: a mixture of this
> day's experience and a reservoir sample from **earlier lives**."*

That is uniform-plus-reservoir. Nothing prioritises. The nomination is an S1
sampling arm — **|RPE|-prioritised** — carrying **reward-magnitude
prioritisation as a control that must fail**, scored on the same ruler and never
competing (the `controls=` pattern `run_bakeoff` already supports, per
`LESSONS.md` "a designed-to-fail control is not a weak arm").

It also converges with the ML side from an independent direction: last week's
watchlist flagged **Simulus**'s prioritised world-model replay. Biology and the
sample-efficiency literature arriving at the same mechanism from opposite ends
is the kind of agreement `GOAL.md`'s biology-as-oracle principle says to take
seriously — and to then make win a bakeoff like anything else.

**Cost.** Near zero. A sampling-distribution change over an existing buffer; no
new parameters, no new forward passes. `NEEDS_AND_DEATH` §2903 budgets `NE.05`
at **4.0 GPU-hours** for 5 arms + 1 control × 3 seeds; one added control is
roughly **+0.7 GPU-h** on that line, and the arm itself is free.

**Why it might WIN (falsifiable).** If |RPE|-prioritised S1 beats uniform S1 at
≥3σ over 3 seeds **while reward-magnitude prioritisation does not**, then the
consolidation stage has a mechanism with a biological rationale, a measured
effect, and a control that separates it from the obvious confound. If both win
equally, the prioritisation is just "replay the salient stuff" and the biology
bought nothing — which is also a finding, and one nothing currently measures.

**Why it might LOSE (steelmanned).** Four, and the first is the strongest:
1. **The ML version of this is eleven years old.** Prioritised Experience Replay
   (Schaul et al., 2015) prioritises by **TD error**, which *is* the RPE. So the
   mechanism is not new and arguably not nominatable — a bakeoff arm that
   re-derives a 2015 standard is not news. What is genuinely new here is the
   **control**, not the arm: PER's literature does not routinely run
   reward-magnitude prioritisation as a must-fail condition. If the builder takes
   only one thing from N2, it should be the control, not the mechanism.
2. **Small n, modest effect.** 6 rats, 3 with ephys, an effect its own authors
   call "modest". This is a real animal result, not a press release — but it is
   not a large one.
3. **The regime does not match S1.** The rats consolidate across **days within
   one life**. S1's distinguishing feature is a reservoir sample from **earlier
   lives** — Lamarckian inheritance across death, which `GOAL.md` explicitly
   names as the place Jack *surpasses* biology. Nothing in this paper speaks to
   cross-life prioritisation, which is the part we care most about.
4. **RPE may not be available where S1 needs it.** S1 rehearses in latent space
   to update the trunk; computing |RPE| per stored transition at sleep time
   needs a value function and per-transition bookkeeping the compressed lifetime
   buffer may not carry. The mechanism could be sound and simply not have a
   cheap port — which must be settled before it is an arm.

---

## 2. CORROBORATION — two `TA.02` citations moved off [k]

Not nominations. `TA.02`'s own notes instruct: *"VERIFY Garcia & Koelling 1966
and the CTA delay tolerance against the primary sources before running; both are
currently carried as [k]."* That is queued work the field watch is the right
organ to do, so this sweep did it.

**Garcia & Koelling 1966 — citation and design CONFIRMED [c+, secondary sources].**
*Relation of Cue to Consequence in Avoidance Learning*, **Psychonomic Science 4:
123–124, 1966**. The design is as `TA.02`'s control (a) describes it: a taste CS
(saccharin water) and an audiovisual CS ("bright-noisy water") each paired with
either nausea (X-irradiation / toxin) or shock. Rats made ill avoided **the
taste**; rats shocked avoided **the audiovisual cue**. A double dissociation, and
`TA.02`'s control (a) is a faithful reading of it.

**Honest verification level:** confirmed consistently across several independent
secondary sources (a published review, the journal's own reference record, a
history-of-science treatment). The original two-page 1966 article is **not
freely available and I did not read it.** This is stronger than [k] and weaker
than [V]; it should be recorded as such rather than promoted to verified.

**CTA delay tolerance — CONSISTENT with `plants.py`'s declared band [V].**
`plants.py` derives `DELAY_S` from *"Rat CTA tolerates 1–6 h reliably (Riley,
Hempel & Clasen, Psychon. Bull. Rev. 25:429–441, 2018)"* against a ~3-day
starvation horizon, giving `DELAY_FRAC_BAND = (0.014, 0.083)` and `DELAY_S = 30 s`
in this world. Independently found: typical CTA protocols use **15 min – 1 h**;
Garcia et al. 1966 demonstrated learning at delays up to **3 h**; Smith & Roll
1967 extended it to **12 h**; and the contrast that matters is that other
classical conditioning tolerates **milliseconds to seconds**.

So the band is defensible and, if anything, **conservative at both edges** — it
excludes the 15-minute typical case at the bottom and the 12-hour reported case
at the top. One observation for the builder, offered as a number rather than a
change: the lower edge (1.4 % ≈ 1 h in rat terms) sits at the *upper* end of the
"typical protocol" range, so this world's delay is on the demanding side of the
biology rather than the easy side. That makes `TA.02` harder, not easier, which
is the right direction for a claim — but it is worth knowing deliberately.

The 3-orders-of-magnitude gap between CTA delays and all other conditioning is
the quantitative backing for `TA.02`'s assertion that *"standard RL cannot do
this task"*. That assertion is now sourced.

---

## 3. WATCHLIST

**Carried forward, unchanged** (fronts 1–3 were not re-swept, so these are
last week's state, not this week's judgement):

| item | status |
|---|---|
| **Simulus** ([arXiv:2502.11537v4](https://arxiv.org/abs/2502.11537)) | Unchanged. Still blocked on **a parameter count and a per-step wall-clock**, which `B4`'s 5.0 sim-s/real-s floor needs and the paper does not report. Its **prioritised replay** component now has independent biological support (§1, N2) and is the cheapest piece to test in isolation. |
| **Survival RL** ([arXiv:2605.31273](https://arxiv.org/abs/2605.31273)) | Unchanged disambiguation — "survival" = dwell time at goals, not homeostatic needs. |
| Last week's N1–N4 | Live nominations, not re-litigated here. See `FIELD_WATCH_LOG.md` 2026-08-10. |

**New this sweep:**

| item | what it is | what would PROMOTE it |
|---|---|---|
| **Optimistic World Models** ([arXiv:2602.10044](https://arxiv.org/abs/2602.10044), 2026-02) | Brings reward-biased maximum-likelihood estimation (RBMLE) from adaptive control into deep RL as an **optimistic dynamics loss** biasing imagined transitions toward higher-reward outcomes. Fully gradient-based — **no uncertainty estimates, no constrained optimisation**, which is what makes it cheap in principle. Instantiated as Optimistic DreamerV3 and Optimistic STORM. Would be an exploration arm bearing on `CURIOSITY_BAKEOFF` (`disagree`/`lp`/`metra`) and on `LEARNING_CORE`. | **Any quantitative result at all.** The abstract claims "significant improvements" and the landing page reports **no benchmarks, no numbers, no parameter counts, no hardware, no code**. Unnominatable until someone opens the full text. Flagged as the highest-value single fetch for next sweep. |
| **Var-JEPA** ([arXiv:2603.20111](https://arxiv.org/abs/2603.20111), 2026-03-20, Gögl & Yau) | **Fetched — a third anti-collapse approach**, bearing directly on last week's N2 (`A4b`/`A4c`). Reframes JEPA as variational inference over coupled latent-variable models, optimising a single ELBO, which it claims **"eliminates the need for ad-hoc anti-collapse regularizers"** and adds latent uncertainty quantification — i.e. it would delete `A4`'s EMA target encoder by a third, independent route. | **A dynamics or control result.** It is instantiated only as **Var-T-JEPA on tabular data**, with no world model, no RL, no sequential prediction, and no reported hardware or parameter counts. `LEARNING_CORE` §5.4's regime objection applies with full force. Promote if anyone runs it on sequences. |
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465), 2025-11) | A benchmark for assessing world-model dynamics understanding in **isolated, minimal environments** — potentially a cheap fidelity-ladder instrument. **Outside the 6-month window; abstract not fetched.** | Read it, and check whether "minimal" means CPU-runnable on 4 ARM cores. Listed so it is not re-discovered as new. |

**REMOVED from the watchlist — resolved as off-target:**

- **Equilibrium World Models** ([arXiv:2606.23463](https://arxiv.org/abs/2606.23463))
  — **not machine learning.** Primary category **`econ.GN` (General Economics)**;
  Scheidegger & Schaab, a deep-learning method for globally solving *dynamic
  stochastic economic models* with rare disasters and binding constraints. "World
  model" here is the economics term. It has nothing to do with Jack. **Deleted.**
- **Multimodal Latent Reasoning via Predictive Embeddings**
  ([arXiv:2604.08065](https://arxiv.org/abs/2604.08065)) — fetched. "Pearl", a
  JEPA-inspired framework that learns **VLM tool-use trajectories** (cropping,
  depth estimation) in latent space, from expert demonstrations. It is a
  vision-language tool-use method, not a multimodal-binding objective, and it
  learns from **expert trajectories** rather than from living. Not a unified-brain
  arm. **Deleted.**

---

## 4. NO-ACTION — fronts where nothing cleared the bar

Stated plainly. An empty week honestly reported beats a padded one.

**TASTE — no arm exists to nominate, and the reason is constitutional.** The
2026 computational-gustation literature runs in two directions, and both are
inadmissible here. One is **receptor biophysics**: multiscale models with
modality-specific receptor dynamics (T1R/T2R for sweet and bitter, ENaC for
salty, H⁺ for sour) and Goldman–Hodgkin–Katz ion-current calculations, aimed at
"ionic realism" from transduction to neural coding. That is **chemistry**, and
`GOAL.md`'s caveman standard rules it out by construction — *"we don't actually
need to understand chemistry for this — just like cavemen didn't"* — as does
`plants.py`'s own declaration that its taste vector is "never chemistry". The
other is **gustatory VR hardware** (e-Taste and the bioelectronic-tongue line):
real electrochemical sensors and edible-chemical actuators for remote tasting.
Excellent work; there is no virtual creature in it. **Nothing to nominate.** The
useful taste work this sweep was verification, not discovery — §2.

**VOICE — nothing clears `VO.02`'s existing bar, which is higher.** The one
in-window hit is *Do Latent Channels Actually Communicate? A Causal Audit of
Latent Multi-Agent LLM Communication*
([arXiv:2607.26773](https://arxiv.org/abs/2607.26773), 2026-07-29, Zhang & Emu).
Its method — controlled message replacement at the boundary where the
sender-produced representation enters the receiver — is **a subset of what
`VO.02` already specifies**: `VO.02` carries scrambled messages, untrained
communication parameters, a muted pair, *and* the positive-listening/causal-
influence requirement, all from Lowe et al. (arXiv:1903.05168) with the exact
numbers recorded. The 2026 paper does not cite Lowe, is measured on **Qwen3-4B
and Qwen3-8B latent relay** rather than any embodied or RL setting, reports
results that **reverse sign between the 4B and 8B scales**, and releases no code.
Nothing here strengthens `VO.02`. Recording it mainly so a future sweep does not
mistake it for a methodological advance over the controls we already have.

**CONFERENCE PROCEEDINGS (queued #4) — attempted, low yield, reported as
partial.** ICML/NeurIPS/ICLR 2026 accepted lists are not enumerable through
search in the form this sweep needed; what surfaced was workshop pages (a NeurIPS
2026 test-time-continual-learning-agents workshop, an ICLR 2026 recursive-self-
improvement workshop) rather than main-track paper lists. This queue item is
**not complete** and should not be marked so. It needs OpenReview enumeration,
which is a different tool than web search.

**FRONTS 1, 2, 3 (learning cores, multimodal fusion, memory) — not re-swept, by
design.** One day is not a research window. Their state is last week's, and last
week's four nominations stand as written. This is a deliberate non-action, not a
finding of emptiness.

---

## 5. A DISCIPLINE FINDING — a title is a claim about a field, and nothing was checking the field

**Three of the five items last week's watchlist carried on title-and-abstract
alone turned out to be off-target when opened, and one of them was not machine
learning at all.**

- *Equilibrium World Models* — primary category **`econ.GN`**. A method for
  solving dynamic stochastic **economic** models. It sat on our watchlist for a
  week as a world-model paper.
- *Multimodal Latent Reasoning via Predictive Embeddings* — a **VLM tool-use**
  method learning from expert trajectories, not a multimodal binding objective.
- *Survival RL* — already caught last week by the previous scout, who recorded
  the disambiguation explicitly so it would not be chased again. That entry is
  the reason this is a pattern and not an incident.

All three were surfaced by title-similarity search against our own vocabulary —
"world model", "multimodal", "survival" — which is exactly the failure mode that
vocabulary creates: **our search terms are the terms other fields also use.**

**The cheap guard that catches all three costs nothing:** every watchlist entry
records the **arXiv primary category** alongside its identifier. `econ.GN` would
have been visible at zero marginal cost at the moment the entry was written, and
would have prevented a week of a dead item occupying a watchlist slot and a
future scout's fetch budget.

This generalises last week's discipline finding one level up. That one was *an
abstract is a claim about a table, and nothing was checking the table agreed*.
This one is **a title is a claim about a field, and the primary category is the
field's own statement of itself.** Same shape, cheaper check, and it applies to
the watchlist rather than to nominations — which matters because watchlist
entries are exactly the ones that get carried forward unverified by design.

Recorded for the builder as a candidate `LESSONS.md` entry. **I am not writing
it; `LESSONS.md` is not mine to edit.** It is nominated like everything else
here. If it is taken, the cheapest form is a convention on this file, not a new
organ — `SYSTEM.md`'s "no new organ without a scar" applies to the field watch's
own machinery too, and a one-column change to a markdown table is not an organ.

---

## 6. What this report does NOT claim

- **No arm here has been run.** Every number is someone else's measurement on
  someone else's hardware. Nothing in this file is evidence about Jack.
- **No nomination is a recommendation to adopt.** `SYSTEM.md` law 3 stands.
- **Nothing here changes a spec, a threshold, a decision, or a line of code.**
  N1 in particular *describes* `odour.py`'s observation vector in order to
  nominate an arm against it; it does not touch it.
- **This sweep did not cover fronts 1, 2 and 3 at all**, and says so in §0 and
  §4 rather than restating last week's findings as if they were re-checked.
- **Verification is uneven and marked as such**: N1 full HTML with numbers for
  the primary source, abstracts for its two siblings, **code not released**; N2
  full text via PMC; Garcia & Koelling **secondary sources only [c+]**; the CTA
  delay band verified against multiple independent sources [V]; Optimistic World
  Models **abstract only, with no numbers in it**.
- **N1 is a pre-emptive nomination against a spec that has not run or failed**,
  which `SYSTEM.md`'s "no new organ without a scar" counts against it. That is
  stated in the nomination itself, not buried here.

---

## 7. Queued for next sweep (2026-08-17)

1. **Fronts 1, 2, 3 — a genuine six-month sweep.** They will then be two weeks
   stale, which is the right cadence for a research window.
2. **Optimistic World Models full text** ([arXiv:2602.10044](https://arxiv.org/abs/2602.10044))
   — the single highest-value fetch outstanding; an exploration method with no
   published numbers is either a nomination or a deletion, and one read decides.
3. **Conference proceedings, properly** — via OpenReview enumeration, not web
   search. Queued #4 last week and **still not done**; recorded as incomplete
   rather than quietly dropped.
4. **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) — is
   "minimal environments" cheap enough for 4 ARM cores?
5. **Watch `SM.02` and `TA.02`.** Both are now the next specs in their families,
   both are `Budget.GPU`, and both carry `kills` clauses aimed at constitutional
   senses. If either runs before the next sweep, this file's N1 either mattered
   or is moot — and which one it is should be recorded here honestly.
6. **Track whether the primary-category convention (§5) was adopted**, and if so
   apply it retroactively to every carried watchlist entry.
