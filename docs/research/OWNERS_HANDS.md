# OWNERS_HANDS.md — provisioning, provenance, and the limit that makes "never puppeteering" a number

> **Status: RESEARCH PASS COMPLETE, SPECS DRAFTED, NOT REGISTERED.**
> Written 2026-09-04 (builder) under `docs/INTEGRATION_QUEUE.md`'s empty-queue
> rule, against its row *"OWNER DECISION: the owner's hands (no doc yet —
> empty-queue rule applies)"*, PENDING since 2026-08-09. That row also binds the
> id range: `SO.03` and `SO.05` are occupied by held `DIRECTION_AUDIT.md` stubs,
> so **this document extends from `SO.06` up**.
>
> Ten citations were fetched and verified in this pass and are marked `[V]`.
> One is carried from another document in this repo without re-verification and
> is marked `[c]`, which is the convention `LANGUAGE_GROUNDING.md` already uses.

---

## 0. Three findings that reframe the question

**Finding 1 — "Never puppeteering" is currently a promise, and it is the only
sentence in `GOAL.md`'s social paragraph with no number attached.** The
neighbouring commitments all became falsifiable: *"trust in a person can be
earned and checked"* is `LG.02`, which PASSED on 2026-09-02; *"he remembers what
he heard, said and did — attributed"* is `ME.9`. The hands sentence — *"what is
left must still be found, learned, and chosen by him"* — has no spec, no metric
and no control, and so it is the one clause in that paragraph that cannot fail.
The whole design below exists to fix that, and §5.2 states the exact arithmetic
that would convict us of puppeteering.

**Finding 2 — W0 already has an undeclared provisioning channel, and it cost a
probe.** Food in W0 is three geoms (`drives.FOOD_GEOMS`: `apple` at ν=0.50,
`obj0`/`obj1` at ν=0.08 — one apple is worth ≈ 300 s of basal life, one floor
food ≈ 48 s) and they are free bodies that **the world never re-places**;
`DriveLayer.new_body` resets the regrowth timers' owner state and never the
positions, deliberately, so that death is not a free refill (`w0.py:_place`,
which pointedly does not call `mj_resetData`). That means food *drifts under
traffic*, and `experiments/lc03_food_probe.py` was written precisely to ask
whether the residual it saw was a **food ratchet** — *"the apple leaves its
platform once and is ground-reachable forever after"* — or eating shot noise.
An object-displacement channel that nobody declared is very hard to rule out
after the fact. A declared one is a log line. **The hands do not add a new class
of risk to this world; they make an existing one legible.**

**Finding 3 — the hands are the cheapest possible person-in-the-world channel,
because they need no ears.** Every other social spec in the ladder needs
something W0 does not have: `SO.03` (company is a need) needs an avatar,
`SO.05` (interruption) needs the voice path, and `LG.*` needs the language
modality — which `w0.py` reports as **ABSENT as an input condition**, not zeros
(*"W0 has no talker in it yet"*). Provisioning needs exactly one primitive that
is already built and already certified: writing `model.body_pos` to move a
static body, which is `plants.py`'s idiom (*"No joint: the body is static and
moved through `model.body_pos`. A freejoint would add dofs to a world whose
observation width other specs depend on."*) and is what `TA.01` certified. So
the first person to reach into Jack's world can be a hand, and it can happen
before he can hear.

---

## 1. What was actually asked for, and what would count as failure

The owner, in `GOAL.md`'s *His people are part of his world*:

> *"their hands may leave things in his world for him to find — food where he
> might look, a tool he has not made yet. **Never puppeteering: what is left
> must still be found, learned, and chosen by him.** His diary records who left
> it — so gratitude, like trust, has somewhere real to grow."*

Three separable claims, and this document keeps them apart because they have
different failure modes and different costs:

| | claim | fails when |
|---|---|---|
| **C1 provisioning** | leaving things changes what he learns | nothing changes (a dead channel), or everything changes and nothing survives withdrawal (a crutch) |
| **C2 provenance** | the diary records *who*, and the record is used | the donor field is written and never read — decoration |
| **C3 the limit** | help stays help | the hands solve the problem; the "learning" is the hand's policy read back |

**C3 is the one that cannot be tested by testing C1.** A provisioned Jack who
scores well is exactly what a puppeteered Jack looks like from outside, which is
the whole content of `de Haan, Jayaraman & Levine 2019` [V] (§2.5): with a
helper in the observation, *access to more information can yield worse
performance*, because the learner takes the helper as the cause. So C3 needs its
own instrument, and §6's `SO.09` is that instrument — an accountant, not a
claim.

---

## 2. Survey

### 2.1 Biology solved this exact problem, and it is called provisioning

`GOAL.md` says biology is the oracle. For "an adult puts an object where a
juvenile will find it", biology has not a metaphor but a measured instance.

**Caro, T. M. & Hauser, M. D. (1992), "Is there teaching in nonhuman animals?",
*Quarterly Review of Biology* 67(2), DOI 10.1086/417553** [V] give the
operational definition that the field still uses, and it is three conjuncts:
a teacher **modifies its behaviour in the presence of a naive observer**, at
**a cost or at least no immediate benefit to itself**, and **the pupil learns
something it would otherwise have learned more slowly or not at all**. This is a
falsification schema, not a definition of a mental state — which is exactly the
shape this repo needs, and it maps one-to-one onto §5's arms (the hands' policy
is contingent on Jack; the hands cost the experiment something declared; the
provisioned learner beats the unprovisioned one).

**Thornton, A. & McAuliffe, K. (2006), "Teaching in wild meerkats", *Science*
313(5784):227–229, DOI 10.1126/science.1128727** [V] is the empirical instance
and it is startlingly close to the owner's sentence. Meerkat helpers provision
pups with prey, and **they grade the item by the pup's age**: young pups get
dead or disabled scorpions, older pups get live ones, and the switch is driven
by the pups' begging calls rather than by anything cognitively expensive. The
pups end up handling live prey faster than they otherwise would. Three things
transfer directly:

1. **The help is an object, not an instruction.** Nothing about the handling
   technique is demonstrated. The adult changes what is *available*, and the pup
   still has to do the whole task on the item it is given.
2. **The grading is contingent on the learner, not on a schedule.** This is the
   difference between a curriculum and a hand, and §5.1 makes it an arm.
3. **It is withdrawn.** Provisioning ends and the adult is not required for the
   competence to persist. That is `SO.07`'s retention measurement, and it is the
   biological reason to expect it to survive.

### 2.2 The taxonomy that separates help from instruction

**Heyes, C. M. (1994), "Social learning in animals: categories and mechanisms",
*Biological Reviews* 69(2):207–231, DOI 10.1111/j.1469-185X.1994.tb01506.x** [V]
is the standard partition, and the relevant row is the *weakest* one.
**Stimulus/local enhancement** is socially mediated exposure to a stimulus or a
place: the observer's subsequent behaviour changes because its *attention* was
moved, and the learning that follows is entirely its own individual learning.
It sits at the opposite end of the same axis from **imitation** (copying the
form of an act) with **emulation** (copying the outcome) between them.

**This is the taxonomy the owner's sentence lives in, and it names the design
target precisely: the hands must be stimulus enhancement and must be shown not
to be emulation or imitation.** A dropped apple moves where Jack looks. It does
not tell him how to climb. The distinction is not rhetorical — §5.2's `C-GIVE`
control is what emulation-by-the-back-door would look like as a number, and
`SO.09` refuses any run that crossed the line.

### 2.3 The failure mode has a name, and children have it badly

Two results say that a helpful adult is not a neutral input to a learner.

**Lyons, D. E., Young, A. G. & Keil, F. C. (2007), "The hidden structure of
overimitation", *PNAS*, DOI 10.1073/pnas.0704452104** [V]: children shown an
adult retrieving an object with a sequence containing **causally irrelevant**
actions reproduce the irrelevant action on their first attempt **86%** of the
time (baseline **16%**), and **81%** of responses across all trials contain it
(baseline **9%**). The children are not confused about physics; they encode the
adult's actions as causally meaningful *because an adult did them*.

**Csibra, G. & Gergely, G. (2009), "Natural pedagogy", *Trends in Cognitive
Sciences* 13(4):148–153, DOI 10.1016/j.tics.2009.01.005** [V] gives the
mechanism: ostensive signals — being addressed — flip the learner into treating
what follows as **kind-relevant and generalisable** rather than as one episode.
Teaching is efficient exactly because the learner over-generalises from it.

**The consequence for this design is a constraint, and it is the reason `SO.06`
is a fixture rather than a paragraph: the provisioning event must carry NO
ostensive cue.** No signal that says *this was for you*. The object appears; it
is indistinguishable in Jack's senses from an object that was always there. The
donor is recorded in the **diary**, which is a record he can consult, not a
salience marker on the object itself. If the hands ever acquire a cue Jack can
perceive at drop time, this literature predicts he will over-generalise from it,
and every downstream claim becomes a claim about the cue.

### 2.4 The ML mirror: environment design, and the designer who solves the task

**Dennis, M. et al. (2020), "Emergent Complexity and Zero-shot Transfer via
Unsupervised Environment Design", arXiv:2012.02096, NeurIPS 2020** [V] is the
formal version of "a person shapes the world instead of the reward", and its
framing of the failure modes is directly usable. Domain randomisation *"cannot
generate structure or adapt difficulty to the agent's learning progress"* — that
is a hand that drops at random, which is `C-MISPLACE` below. Minimax adversarial
design produces *"worst-case environments that are often unsolvable"* — the
opposite hand. PAIRED's answer is to drive the designer by **regret**, the gap
between a protagonist and an antagonist, which is a real candidate for a hand
policy and is deliberately **not** what §5 proposes first (see §3.2: a
regret-driven hand is a second arm, after a fixed contingent hand has shown the
channel exists at all).

**Co-Reyes, J. et al. (2020), "Ecological Reinforcement Learning",
arXiv:2006.12478** [c] — carried from `PURPOSE_AND_SCAFFOLDING.md` §1.6, not
re-verified in this pass — is the closest published statement of the thesis:
modifying the *world* can substitute for shaping the *reward* in non-episodic
settings. `PURPOSE_AND_SCAFFOLDING.md` already cites it for "nutritious, not
rewarded"; it applies verbatim to the hands, which are world modification with
a name attached.

**Ng, A. Y., Harada, D. & Russell, S. (1999), "Policy invariance under reward
transformations", ICML** [V] gives the only clean formal guarantee in this
neighbourhood — potential-based shaping `F(s,a,s') = γΦ(s') − Φ(s)` is
*necessary and sufficient* for preserving the optimal policy — and this repo has
already worked out why it does not rescue us. `PURPOSE_AND_SCAFFOLDING.md` §2.6
derives, for W0's identically-zero environment reward, that
`V^π_shaped(s) = −Φ(s)` **for every π**: *"Policy invariance is not a safety
property here; it is vacuity."* **So there is no theorem available that makes
the hands safe. The safety has to be measured, which is §5.2 and `SO.09`.**
That derivation is the single most useful thing already in this repo for this
document, and it is why the anti-puppeteering guard is empirical.

### 2.5 Why more help can measure worse

**de Haan, P., Jayaraman, D. & Levine, S. (2019), "Causal Confusion in Imitation
Learning", arXiv:1905.11979, NeurIPS 2019** [V]: a discriminative policy trained
on observations that contain a *correlate* of the right action will latch onto
the correlate, and *"access to more information can yield worse performance"* —
the paper's own phrasing of causal misidentification. Their fix requires
**targeted intervention**: you must be able to vary the suspect variable
independently and see what happens.

This is the argument for `SO.09` being an accountant with a **positive control**
rather than a policy restriction. A hand in the world is a variable correlated
with success by construction. The only way to know it was not the cause is to
withdraw it and to have logged enough to say what it did — which is why every
provisioning event is logged with `(t, agent, object, position, need-state)` and
why `SO.07`'s scoring phase happens with the hands **off**.

### 2.6 Scaffolding, and the part everyone forgets

**Wood, D., Bruner, J. S. & Ross, G. (1976), "The role of tutoring in problem
solving", *Journal of Child Psychology and Psychiatry* 17:89–100** [V] coined
scaffolding as *"a process that enables a child or novice to solve a task or
achieve a goal that would be beyond his unassisted efforts"* and listed six
tutor functions. Two matter here and one is a trap:

- **"Reduction in degrees of freedom"** is exactly what a dropped object does:
  it does not do the task, it shrinks the search. This is the honest
  description of the hands.
- **"Demonstration"** is the sixth function and it is the one the hands must
  **not** have, per §2.2. A hand that demonstrates is an imitation channel and
  belongs to a different spec with different controls.
- The literature's own four-decade retrospectives note that **removal** —
  Bruner's fading — is the least-studied half. This repo already knows that:
  `PURPOSE_AND_SCAFFOLDING.md` §1.6 searched for it and reports *"on scaffold
  removal the record is thin and mostly definitional… no systematic study of
  competence retention after removal of a drive-like auxiliary reward was
  found."*

**And this repo has already built the measurement the literature lacks, for a
different scaffold.** `PURPOSE_AND_SCAFFOLDING.md` §3.6 defines

```
retention_ratio      R = C_off / C_on     "does the competence survive removal?"
scaffolding_benefit  B = C_off / C₀       "was the scaffold worth having?"
```

with a full pre-decided outcome table over `(R, B)` and a `C-BEELINE` control
whose job is to *calibrate what a failed retention looks like*. **`SO.07` below
reuses `R` and `B` verbatim and cites them; it does not invent a parallel
metric.** The difference between the two documents is the *kind* of scaffold —
PS's is an internal reward channel wired into the return, the hands are an
external, intermittent, attributed modification of the world — and that
difference is why they are two specs and not one. It is also why the outcome
table transfers but the *interpretation* of `R < 0.5` does not: a drive that
turns out to be load-bearing is a component we keep; **a hand that turns out to
be load-bearing is a puppeteer we must stop.**

### 2.7 Provenance is a separate faculty from memory

**Johnson, M. K., Hashtroudi, S. & Lindsay, D. S. (1993), "Source monitoring",
*Psychological Bulletin* 114(1):3–28, DOI 10.1037/0033-2909.114.1.3** [V] is the
canonical treatment, and its load-bearing claim for us is structural: knowing
*that* something is the case and knowing *where it came from* are supported by
different processes and **dissociate** — misattributed familiarity, cryptomnesia
and confabulation are all cases where the content survives and the source does
not. Source attributions are reconstructive judgements made from qualities of
the record, not tags read off it.

Two consequences:

1. **A donor field that is written and never read is not provenance.** It has to
   change a decision, or the system has the content without the source — the
   exact dissociation this literature describes. That is why `SO.08`'s claim is
   a **behavioural divergence by donor**, not a recall score.
2. **The control writes itself.** Shuffle the donor field across otherwise
   identical records: the content is intact and only the source is wrong. If
   behaviour is unchanged, the source was never read. This is the same move
   `LG.02` made for advice and `FROZEN_VS_PLASTIC.md` §10.2 mandates for the
   parent's words; see §3.3 for why `SO.08` is not a duplicate of `LG.02`.

---

## 3. Cross-check (`INTEGRATION_QUEUE.md` step 1), run before drafting

Terms grepped across every other `docs/research/*.md` and `docs/LESSONS.md`:
*puppet, provision, drop-in, scaffold, teaching, demonstration, imitation,
social learning, provenance, attribution, "who left", donor, gift, shaping*.
**No refutation was found. Four constraints were, and all four changed a draft
below.**

### 3.1 `PURPOSE_AND_SCAFFOLDING.md` §2.6/§3.6 owns the removal metric — REUSE, do not re-invent
Already discharged in §2.6 above: `SO.07` uses `R` and `B` as defined there, and
the `C-STATUE` and `C-SHUFFLE` control shapes come with them. **Without this
check `SO.07` would have shipped a third parallel definition of "did the help
help", which is how two specs end up disagreeing about the same world.**

### 3.2 §2.6(ii) forecloses the theoretical safety argument
`PURPOSE_AND_SCAFFOLDING.md` proves policy invariance is vacuous on W0's
zero-reward world. **This deletes an arm this document would otherwise have
proposed** — "make the hands potential-based and prove they cannot change the
optimum" — and forces the anti-puppeteering guard to be an empirical accountant
(`SO.09`). It also demotes a regret-driven PAIRED-style hand (§2.4) to a *later*
arm: an adaptive designer is much harder to bound than a fixed contingent one,
and the channel has not yet been shown to exist.

### 3.3 `FROZEN_VS_PLASTIC.md` §10.2 mandates two controls for anything acting as a parent — and the hands are a parent
§10.2 requires, for *"every spec that runs with a parent"*, a **MUTE-PARENT**
twin and a **SHUFFLED-PARENT** twin (*"same words, same rate, wrong events… if
Jack's grounding survives shuffling, he learned the words' distributional
statistics and not their referents"*), and says the second **MUST fail**. The
hands are a parent that acts through objects instead of words, so the analogues
are mandatory and **both were added to the drafts by this check**:

- `A0` / **ABSENT-HANDS** — the mute twin, already present as the null.
- **`C-MISPLACE`** — *the shuffled twin, which this document did not have before
  the cross-check.* Same drop rate, same objects, same object mix, same
  optimiser steps; positions and times drawn independently of where Jack is and
  what his needs are. It **must fail**. It is not redundant with `A0`: `A0` has
  no channel at all, `C-MISPLACE` has a live channel carrying no information, so
  an arm whose whole advantage is *the mere presence of more objects in the
  world* beats `A0` and ties `C-MISPLACE`. This is the same hole the
  `LANGUAGE_GROUNDING.md` registration caught one iteration late; here it is
  caught before the draft.

### 3.4 `LG.02` owns attributed trust — `SO.08` must share its implementation, not re-derive it
`LG.02` PASSED 2026-09-02: two advisors at 0.9/0.1 accuracy, trust as a Laplace
posterior over verified claims, joined to Jack's search only through the
attributed diary, worst-seed divergence 0.689 ± 0.103 against a 0.40 gate,
stripped-attribution null 0.028. `SO.08` is the same posterior over a different
channel, and `GOAL.md` deliberately writes the two as two sentences: *"his diary
records whose advice proved true"* (a **claim**, verifiable when uttered) and
*"his diary records who left it"* (an **object**, whose value is only discovered
by using it — a delayed, self-administered outcome). The distinction is real and
worth a spec; the *mechanism* is one implementation imported by both, exactly as
the `LANGUAGE_GROUNDING.md` row required of `T2.16` and `LG.04`'s `A4`.
**If the registering iteration judges the gap too thin, the correct outcome is
to REFUSE `SO.08` and widen `LG.02`'s successor — not to register a second
posterior.** Recorded here so that refusal is available rather than awkward.

### 3.5 `SO.04` ("being watched does not change him") sets the discipline for `SO.06`
`SO.04` PASSED with the rule that an observer path must not consume RNG, and its
control deliberately draws one RNG value so the detector can see its own
positive control. **The hands are an intervention path with the same hazard in a
stronger form** — they are *supposed* to change the world, which makes it much
easier for them to also change the body, the timing or the stream by accident.
`SO.06` therefore carries `SO.04`'s discipline verbatim: with the hands present
but dropping nothing, the trajectory must be **bit-identical** to no-hands, and
a deliberately-perturbing hand must be **caught**.

### 3.6 Nothing here is W1-held
`WP.01`–`WP.04`, `PS.07`, `LF.04`, `SO.03` and `SO.05` are held behind the
Review's 2026-09-06 W1/`w0-too-shallow` design. **The specs below are not**:
they need no new world content, no avatar, no voice and no language channel —
only `model.body_pos`, the drive layer and the diary, all of which exist and are
certified today (§4). This document deliberately designs nothing that requires
the held material, and says so here so the registering iteration does not have
to re-derive it.

---

## 4. The venue, measured rather than assumed

The `LG.03` lesson from 2026-09-04 is one day old and it is the reason this
section exists: *a spec whose venue is not certified first will spend its run
discovering the venue.* What W0 supplies today, read out of the code:

| requirement | supplied by | status |
|---|---|---|
| place an object at runtime | `model.body_pos` write; `plants.py:plant_mjcf` docstring — *"the body is static and moved through `model.body_pos`"* | **exists, certified by `TA.01`** |
| something worth provisioning | `drives.FOOD_GEOMS` — `apple` ν=0.50 (≈ +300 s basal), `obj0`/`obj1` ν=0.08 (≈ +48 s each) | **exists** |
| something worth provisioning that is NOT food | `obj2`–`obj4`, ladder, platform, ramp, stairs (39 geoms measured seed 0) | **exists** |
| an unreliable gift | `plants.py` — two visually identical types, one toxic, declared dose-response and delay | **exists, certified by `TA.01`** |
| objects persist across death | `w0.py:_place` does not call `mj_resetData`, deliberately | **exists** |
| a diary that crosses death with attribution | `EpisodicMemory`, `meta["life"]`, `ME.9`'s attributed channels; certified crossing death by `XL.00` | **exists** |
| a need the help can address | `drives.DriveLayer` `e`/`i`/`w`, `d(h)`, death at `e→0` | **exists** |
| an avatar / a voice / a language input | — | **ABSENT; not required by anything below** |

Two venue hazards, both already paid for by somebody else:

- **Adding dofs is forbidden.** `plants.py` says why: *"A freejoint would add
  dofs to a world whose observation width other specs depend on."* A provisioned
  object must be a static body moved by `body_pos`, or nine specs go stale.
- **Food geoms already drift** (§0, Finding 2). Any hands experiment must log
  positions at every drop *and* at the end of each life, or it cannot separate
  its own effect from the drift `lc03_food_probe.py` was written to chase.

---

## 5. The design

### 5.1 The hand policies (arms)

| arm | the hand's policy | role |
|---|---|---|
| **A0 — ABSENT** | never drops | the null; the mute-parent twin (§3.3) |
| **A1 — CONTINGENT** | drops one food item at a legal, unoccupied, **currently unseen** position within a declared radius of Jack when `e` falls below a declared floor | the claim arm; Caro & Hauser's "modifies behaviour in the presence of the learner" |
| **A2 — GRADED** | as A1, plus the item's *value* is graded by lives survived (floor food early, apple never; apple only after life *k*) | the meerkat arm (§2.1); an arm, not an assumption |
| **C-MISPLACE** | same rate, same item mix, positions and times independent of Jack and of `e` | **MUST fail** (§3.3) |
| **C-GIVE** | places the item **in contact** with Jack's body, or restores `e` directly | the puppeteer; **must show high on-hands competence and near-zero retention** — it calibrates what a failure of `R` looks like, the `C-BEELINE` role |

`A2` is listed because the literature's most-replicated finding is that grading
is what makes provisioning teaching. It is an arm and not a design decision.

### 5.2 The arithmetic that would convict us of puppeteering

Scored with the hands **withdrawn**, reusing `PURPOSE_AND_SCAFFOLDING.md` §3.6:

```
C_on   = competence measured with the hands live
C_off  = competence measured after withdrawal          ← the number that matters
C₀     = competence of A0, matched compute

R = C_off / C_on    B = C_off / C₀
```

- **`B ≈ 1`** → the hands bought nothing. Honest red; the channel is dead.
- **`R` high, `B > 1`** → *"found, learned, and chosen by him"* is measured: the
  help helped and the competence is his, because it outlived the helper.
- **`R` low, `B > 1`** → **this is the puppeteering result**, and it must be
  reported as one. He is only competent while a hand is feeding him. Under
  `PURPOSE_AND_SCAFFOLDING.md`'s table the analogous cell reads *"drives
  LOAD-BEARING"* and is a design finding; **here the same numbers are a
  violation of a `GOAL.md` ENDS-class commitment**, and the response is to stop
  provisioning, not to keep it. Written down now, before any run, because after
  a run it would be tempting to read it the other way.
- **`C-GIVE` must land in that same low-`R` cell.** If it does not, the
  instrument cannot see puppeteering and nothing else on this page means
  anything.

### 5.3 What is deliberately not proposed
- **No ostensive cue** on the drop (§2.3). Ever, in any arm.
- **No demonstration arm.** That is imitation, a different family, different
  controls (§2.6).
- **No regret-driven adaptive hand** until the fixed contingent hand has shown
  the channel exists (§3.2).
- **No new world content.** Everything above runs in W0 as it stands (§4).

---

## 6. Registry entries — DRAFTS, not registered

House format, `experiments/registry_expansion.py`'s `Spec(...)`. Ids start at
`SO.06` per `INTEGRATION_QUEUE.md`. **These have had the step-1 cross-check of
§3 but not the full registering pass; the `LANGUAGE_GROUNDING.md` precedent is
that the registering iteration re-runs step 1 in full and finds more.**

```python
Spec("SO.06", 2, "A hand can reach into a running life, and it reaches ONLY through the world",
     hypothesis="A declared external agent can place an object into a live W0 "
                "at a legal, unoccupied, currently-unseen position mid-life; "
                "the placement is visible in Jack's own senses within a "
                "declared time once he looks, is logged with (t, agent, "
                "object, position, need-state), and changes NOTHING else — "
                "with the hand present but dropping nothing, the trajectory is "
                "bit-identical to no-hand at the same seed.",
     falsified_by="No placement exists that is simultaneously legal, "
                  "reachable and initially unseen (the venue cannot host "
                  "provisioning), OR the hand path perturbing the RNG stream, "
                  "the body state, the need variables or the timing, OR a "
                  "placement that never becomes perceptible.",
     null_baseline="No hand — the current state, in which no spec has ever "
                   "placed an object into a running life.",
     metric="provision_channel_ok", budget=Budget.CPU, seeds=3,
     depends_on=["XL.00", "TA.01"],
     control="TWO, both of which must fire. (1) A hand that deliberately draws "
             "one RNG value and nudges the body: the invariance detector MUST "
             "catch it — SO.04's rule, a detector that cannot see its own "
             "positive control has measured nothing. (2) Place the object "
             "OUTSIDE every ray's reach and behind occlusion: the observation "
             "must NOT change. An observation that moves for an object he "
             "cannot see is a side-channel, not a sense.",
     kills="'Their hands may leave things in his world' as a sentence rather "
           "than a channel.",
     notes="THE FIXTURE. Every SO.07-SO.09 number is scored against this, so "
           "it is certified first and separately (TA.01/plants.py precedent). "
           "VENUE, measured 2026-09-04: the object is a STATIC body moved by "
           "`model.body_pos` — plants.py's idiom, and a freejoint is forbidden "
           "because it would add dofs to a world whose observation width nine "
           "specs depend on. Food geoms already drift under traffic and the "
           "world never re-places them (w0.py:_place omits mj_resetData "
           "deliberately), so positions MUST be logged at every drop and at "
           "each life's end or this spec cannot separate the hand from the "
           "drift lc03_food_probe.py was written to chase. NO OSTENSIVE CUE, "
           "in this or any successor: Csibra & Gergely 2009 predicts a "
           "perceptible 'this is for you' marker would make every downstream "
           "claim a claim about the marker. "
           "  COVERS: social/other agents (fixture)"),

Spec("SO.07", 3, "What the hands leave is FOUND, and what he learns outlives them",
     hypothesis="A learner provisioned by a need-contingent hand reaches a "
                "higher competence than an unprovisioned twin at matched "
                "compute, AND retains it when the hand is withdrawn: "
                "retention_ratio R = C_off/C_on above its floor with "
                "scaffolding_benefit B = C_off/C0 above 1 "
                "(PURPOSE_AND_SCAFFOLDING.md §3.6's metrics, reused verbatim).",
     falsified_by="B ~ 1 (the hand bought nothing), OR R below its floor while "
                  "B > 1 — competence that exists only while a hand is "
                  "feeding him. THE SECOND IS THE PUPPETEERING RESULT and is "
                  "reported as a violation of a GOAL.md commitment, not as a "
                  "design finding; the response is to stop provisioning.",
     null_baseline="A0, no hand, matched architecture, matched optimiser "
                   "steps, matched wall-clock.",
     metric="hand_retention_ratio", budget=Budget.CPU_LONG, seeds=3,
     depends_on=["SO.06", "PS.01"],
     control="C-MISPLACE (MANDATORY, FROZEN_VS_PLASTIC.md §10.2's shuffled-"
             "parent analogue): identical drop rate, item mix and optimiser "
             "steps, positions and times drawn independently of Jack's "
             "position and needs. MUST FAIL. It is not redundant with A0 — A0 "
             "has no channel, C-MISPLACE has a channel carrying noise, so an "
             "arm whose advantage is merely that the world contains more "
             "objects beats A0 and ties C-MISPLACE. "
             "C-GIVE: the hand places the item IN CONTACT or restores energy "
             "directly. MUST show high C_on and near-zero R — it calibrates "
             "what a failed retention looks like (the C-BEELINE role). If "
             "C-GIVE does not land low, the instrument cannot see "
             "puppeteering and no other number here means anything.",
     kills="'Never puppeteering: what is left must still be found, learned, "
           "and chosen by him' as an unfalsifiable promise.",
     notes="ARMS: A1 contingent (drop when e < floor, within radius, unseen), "
           "A2 GRADED by lives survived — the meerkat arm (Thornton & "
           "McAuliffe 2006: helpers grade prey by pup age and pups learn "
           "faster), listed as an ARM because grading is the field's "
           "most-replicated finding and this repo decides by bakeoff. "
           "R and B are PURPOSE_AND_SCAFFOLDING.md §3.6's, deliberately not "
           "re-derived; its (R,B) outcome table transfers but its "
           "INTERPRETATION does not — a load-bearing drive is a component we "
           "keep, a load-bearing hand is a puppeteer we must stop. "
           "  COVERS: social/other agents (claim)"),

Spec("SO.08", 3, "The diary records WHOSE hands, and he acts on it",
     hypothesis="With two donors of different reliability leaving visually "
                "indistinguishable gifts, Jack's approach rate to a newly "
                "dropped object diverges by donor above a base-rate null, and "
                "the divergence runs THROUGH the attributed diary: strip or "
                "shuffle the donor field and it collapses.",
     falsified_by="No divergence (the donor field is written and never read — "
                   "decoration), OR divergence surviving donor-shuffling (it "
                   "was carried by something other than the attribution).",
     null_baseline="Donor-stripped diary, same events, same counts — LG.02's "
                   "null, reused.",
     metric="donor_trust_divergence", budget=Budget.CPU_LONG, seeds=3,
     depends_on=["SO.06", "ME.9", "LG.02"],
     control="Donor-shuffled diary (content intact, source permuted): MUST "
             "collapse — Johnson, Hashtroudi & Lindsay 1993's dissociation "
             "made a control. Plus an EQUAL-DONORS run: two donors of "
             "identical reliability must produce NO divergence. A detector "
             "that reports a difference where none exists is broken, and "
             "without this leg a divergence is unattributable.",
     kills="'His diary records who left it — so gratitude, like trust, has "
           "somewhere real to grow' as decoration.",
     notes="THE UNRELIABLE GIFT IS ALREADY BUILT: plants.py's two visually "
           "identical types, one toxic, with a declared dose-response and "
           "delay, certified by TA.01. SHARES LG.02's MECHANISM AND MUST "
           "SHARE ITS IMPLEMENTATION — one Laplace posterior over verified "
           "outcomes, imported by both, never re-derived (the T2.16/LG.04-A4 "
           "rule). The channels differ as GOAL.md's two sentences differ: "
           "LG.02's evidence is an ADVICE CLAIM checkable when uttered, "
           "SO.08's is an OBJECT whose value is only discovered by using it — "
           "a delayed, self-administered outcome. IF THE REGISTERING "
           "ITERATION JUDGES THAT GAP TOO THIN, REFUSE THIS SPEC and widen "
           "LG.02's successor instead; do not register a second posterior. "
           "  COVERS: social/other agents (claim)"),

Spec("SO.09", 0, "A life the hands bought is not evidence, and the harness says so",
     hypothesis="Every provisioning event is logged, and any run that claims "
                "learning reports hand_share (the fraction of need-restoration "
                "events causally downstream of a hand inside a declared "
                "window) and hand_contact_frac (the fraction of placements "
                "made within body-contact distance) against a ceiling the "
                "spec declared BEFORE the run; a run over its ceiling is "
                "REFUSED by the runner, not reported with a caveat.",
     falsified_by="The accountant passing a deliberately puppeteered run "
                  "(C-GIVE), OR refusing a clean one, OR a provisioning event "
                  "reaching the world without a log line.",
     null_baseline="Today: nothing measures this, and 'never puppeteering' is "
                   "enforced by the intention of whoever wrote the hand.",
     metric="hand_share_audited", budget=Budget.CPU_FAST, seeds=1,
     depends_on=["SO.06"],
     control="SO.07's C-GIVE arm, replayed: the accountant MUST refuse it. A "
             "guard that has never rejected anything has not been shown to "
             "work (LESSONS: a check whose failure mode is going quiet needs "
             "an alarm on the silence).",
     kills="The practice of reading a provisioned run as a learning result "
           "because the hand 'only helped a little'.",
     notes="THE GUARD, and the reason it is empirical rather than "
           "theoretical: PURPOSE_AND_SCAFFOLDING.md §2.6(ii) proves "
           "potential-based shaping is VACUOUS on W0's zero-reward world "
           "(V^pi_shaped = -Phi for every pi), so Ng, Harada & Russell 1999's "
           "policy-invariance guarantee is unavailable here and safety must "
           "be measured. de Haan et al. 2019 is the reason the ceiling is on "
           "the HAND'S CONTRIBUTION rather than on the outcome: a helper in "
           "the observation is correlated with success by construction, and "
           "only withdrawal plus a log can separate them. Ceilings are per-"
           "spec and declared in the spec; this one owns the accounting, "
           "never the number. "
           "  COVERS: social/other agents (rule)"),
```

**Cost classes and what registration would actually do — stated precisely,
because "fills an empty class" is the kind of claim that flatters itself.**
All five distinct dependencies are **PASS today** (`XL.00`, `TA.01`, `PS.01`,
`ME.9`, `LG.02` — read from the ledger 2026-09-04), so:

- **`SO.06` (`cpu<10min`) is runnable the moment it is registered** — the only
  FRESH dispatch on the board, which today has none.
- **`SO.09` (`cpu<1min`), `SO.07` and `SO.08` (`cpu<2h`) are NOT** — each
  depends on `SO.06`, so they are one PASS away. `cpu<1min` therefore stays
  *no-path-in* by `coverage`'s definition until `SO.06` lands. Registering does
  not clear it and this document does not claim it does.
- `cpu<10min` currently reads *HELD by an open decision (implement NOTHING
  here): `HR.1` ← `D19`*. That hold is on **`HR.1`**, the one unimplemented
  runnable spec in the class; it is not a hold on the class. The registering
  iteration should confirm that reading rather than take it from this line.
- **`SO.07` is the one to worry about**: `cpu<2h` has ~1 h of daily slack
  against a 54,000 s admission cost per never-run spec, so registering it may
  foreclose the class for a day. See §7.

---

## 7. Cost, against free compute only

Nothing here needs a GPU. `SO.06` is a fixture over a handful of lives.
`SO.09` is an accountant over a recorded log. `SO.07` is the expensive one — it
is `PURPOSE_AND_SCAFFOLDING.md`'s two-phase shape (train with the scaffold,
score without) times five hand policies times three seeds — and it must be sized
by a pilot before registration commits a class, because `cpu<2h` currently has
**one hour of daily slack** against a 54,000 s admission cost per never-run
spec (`run status`, 70th audit B4). **`SO.07` is the spec most likely to be
foreclosed by our own CPU bookkeeping, and that is a scheduling fact to hand to
the Review, not a reason to shrink the design.**

---

## 8. What this document does not settle

1. **Whether W0 is deep enough for the hands to matter.** If food is easy, a
   dropped apple buys nothing measurable and `B ≈ 1` for reasons about the
   world, not about provisioning. This is `w0-too-shallow`'s question and
   `SO.06`/`SO.07` would become the tenth and eleventh instruments pointed at
   it. Not re-litigated here.
2. **Who the donors are.** Two labelled hands are a fixture, not people. The
   step from "donor A" to a named human with a persistent identity across lives
   is `SO.03`/`SO.05` territory and is W1-held.
3. **Whether gratitude is measurable at all.** `GOAL.md` says gratitude should
   have *"somewhere real to grow"*. `SO.08` measures donor-differentiated
   approach, which is trust's behavioural shadow, not gratitude. This document
   deliberately does not claim otherwise, and `GOAL.md`'s *"who he becomes is
   not written here"* says the emergent version is to be observed, not specced.
4. **The grading schedule in `A2`.** Meerkats grade on begging calls; W0 has no
   begging. Lives-survived is a stand-in chosen because it is observable without
   a voice, and it may be the wrong variable.

---

## 9. What this changed about the machine

- **A `GOAL.md` clause that could not fail now has a proposed number.** *"Never
  puppeteering"* becomes `R` measured with the hands withdrawn, plus a `C-GIVE`
  arm that must land in the failing cell, plus a runner-level refusal.
- **The interpretation asymmetry is recorded before any run** (§5.2): identical
  `(R, B)` numbers mean *keep it* for a drive and *stop it* for a hand, because
  one is an ARCHITECTURE-class question and the other is an ENDS-class
  commitment. Writing that down after a run would be indistinguishable from
  motivated reading.
- **The mandatory shuffled-parent control was applied prospectively.** The
  `LANGUAGE_GROUNDING.md` registration found `FROZEN_VS_PLASTIC.md` §10.2's
  second control one iteration late, and its lesson was that a self-run
  cross-check searches the neighbourhood it already knows. `C-MISPLACE` is in
  these drafts because that lesson was applied on purpose, and §3.3 records the
  provenance so the next reader can tell the difference between a control that
  was designed in and one that was caught.
- **A refusal is pre-authorised.** §3.4 says plainly that if `SO.08`'s gap from
  `LG.02` is too thin, the registering iteration should REFUSE it. A research
  document that leaves its author no way to be turned down is an advocacy
  document.
