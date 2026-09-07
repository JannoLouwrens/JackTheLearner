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

**Sweep date:** 2026-09-07 · **Window:** ~2026-03 → 2026-09 (6 months)
**Scout:** field watch, week 6. **Seven days since week 5** (2026-08-31), the
embargo (*"not before ~2026-09-07"*) spent to the day. Second consecutive sweep
on the intended cadence.

**Confidence markers:** **[V]** fetched and read · **[c]** claimed by the authors,
not checked against their table · **[s]** asserted by a search engine *about* a
paper I have not opened · **[C]** computed by me, arithmetic shown · **[M]**
measured here, on this box, command shown.

---

## 0. WHAT MOVED IN THE INTERVAL — and the first thing this desk ever measured

418 commits landed in seven days. Five facts re-point this sweep.

**(1) THE DESK'S FIRST MEASUREMENT LANDED, AND ITS BINDING CONTROL CLEARED.**
`W0.DIAG` — wk4-N3's diagnostic, carrying wk5-N3 as a **binding** known-answer
control — is **PASS** (attempt 3, `dad5f27`, 3 seeds). The correlated null buys
`gain_up` **12.12 ± 1.20** against the stationary null's **0.0095 ± 0.39**,
`eats_up` 1.0 vs `eats_random` 0.33 — and the wk5-N3 control **cleared at t≈11
before the W0 reading was believed**, exactly as ordered. That is five weeks of
nominations reaching a number for the first time. It is also the strongest
available evidence that this desk's cheap-control nominations are worth more
than its arm nominations, which is a fact about what a scout is good for.

**(2) `D10` FIRED 2026-09-01. `wm-latent` (`A4`) HOLDS THE LEARNING-CORE SEAT
BY VERDICT (single-arm), AND ITS REGISTERED CHALLENGER IS `LC.07`** — the
owner's ~10× scale-transfer guard, `depends_on` all-PASS, unrun. Front 1 is
therefore pointed at one arm that everything now rests on, and §2's lead
nomination is a check ON that arm rather than a competitor to it.

**(3) THE W1 SPEC FAMILY IS PUBLISHED (Review FULL, 09-06): `W1.00`–`W1.04`.**
`W1.00` **FAIL** attempt 1 (informative branch) and `W1.02` **PASS** attempt 1,
both on W0 as built. `W1.01`/`W1.03`/`W1.04` — passivity dies, traps/delays/
irreversibility, horizon ≥ 3× time-to-consequence — are **unregistered behind
`w1-world-edit-window` (DUE 2026-09-13)**. Front 5's whole question this week
is whether the literature can supply anything those three specs need. It
cannot, and §5 says so with the arithmetic that closed the queue item.

**(4) MY OWN wk5 FINDING WAS ROUTED INTO PILE A AND HAS NOW BEEN TESTED.**
The W1 design filed wk5's *"a random policy covers W0 as well as the curious
arm"* as **Pile A — under-nulled**, i.e. possibly an artifact of a too-weak
null. `W1.00` measured that directly: re-scoring Pile A against the stronger
correlated null moves recorded margins by **0.022–0.084 of their own std**, and
the one larger shift (dwell, `shift_ratio` 16.23) fails its own reality gate at
t≈1.98. **The under-nulling worry is measured immaterial; the wk5 arithmetic
stands and the world question tips toward Pile B.** I record this because a
scout whose finding survives its own falsifier should say so in the same voice
it would have used to say the opposite.

**(5) TWO FUSION ROWS ARE NOW DATED, AND ONE OF THEM ORDERS THE BAKEOFF I
REFUSED TO NOMINATE INTO FOR TWO SWEEPS.** `ub10-seed-fragility-and-saturated-
battery` is DUE **2026-09-08**; `t402-touch-drowns-audio-at-the-fusion-boundary`
is DUE **2026-09-13** and says in terms: *"the candidate repairs are runnable
arms — per-modality gradient normalisation, loss reweighting, modality dropout
schedules — so this is a bakeoff to design."* My Goodhart refusal cannot stand
as a refusal any more; it has to become a **design condition on a bakeoff
somebody else has ordered**, and §2's N3 is that, with a measured mechanism
behind it instead of an argument.

**And one correction I owe.** Week 5 reported `SM.02` unrun for the fourth
consecutive sweep. The Review's answer is correct and taken: **`SM.02` is
PARKED**, a parked spec is not a pending one, and the fourth report should not
have been written. I read `run status` this sweep (`[-] SM.02`, not
implemented, parked by the builder's both-fail branch of 2026-08-20). It is not
reported as a gap here and will not be again.

---

## 1. Coverage — what was actually searched, so the gaps are visible

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **1 · LEARNING CORES** | decoder-free / latent-predictive world models; **action-sensitivity of latent dynamics**; scale-transfer of small-model verdicts | **full arXiv-API enumeration (40 entries)**; **3 abstracts + 1 full HTML** for the nomination cluster |
| **5 · WORLDS & EMBODIMENT** *(front 3's freed slot, per wk5)* | survival/homeostatic sims; environment-difficulty and discriminability metrics; UED/open-endedness **fetched, not searched**; irreversibility; headroom-certified benchmark design | **full arXiv-API enumeration (40 entries)**; **2 full HTML** (2602.09813, 2508.00046); abs for ATLAS |
| **4 · CURIOSITY & OPEN-ENDEDNESS** *(front 3's freed slot)* | intrinsic motivation; epistemic/aleatoric separation; autotelic agents; environment generation | **full arXiv-API enumeration (40 entries)** + 2 searches; **full HTML for PRIME** |
| **2 · MULTIMODAL FUSION** | balancing successors; whether any of it left classification; fusion-vs-concat discriminability; dominant-modality effects | 2 searches; **full HTML for MULTIBENCH++**; abs for IAF |
| **3 · MEMORY** | **NOT SWEPT — two-week cadence, endorsed by the Review 2026-08-31. Next due 2026-09-14.** See §5 | — |
| Biology-as-oracle | the unmined shelf: play as safe rehearsal, innate reflex priors, pain as a fast signal, critical periods | 2 searches; 1 abs |
| Small-model end | sub-1M-parameter embodied control | 1 search, **nothing in window** — fourth consecutive sweep |
| Queued #1 (UED, properly) | **CLOSED with a NEGATIVE answer.** See §5 | full HTML |
| Queued #2 (2607.22430, last attempt) | **CLOSED NEGATIVE and DROPPED with cause.** See §3 | full HTML |
| Queued #3 (wk5-N1's Lean repo) | **CLOSED: no repository is stated.** See §3 | abs + search |
| Queued #6 (Simulus, close or drop) | **DROPPED with cause.** See §3 | — |
| Our own artifacts | `lc03_curves_seed{0,1,2}.json`, arm construction at HEAD | **[M] — commands in §6** |

**Known gaps, stated so nobody assumes coverage:**
- **Front 3 was deliberately not swept.** That is the endorsed cadence, not an
  omission — but a new scar landed on it during the interval (`ME.1` FAIL,
  `distractor_abstention` **0.0000** on three seeds) and §5 says what that
  means for 09-14.
- **N1's three sources are all abstract-level except one.** Only Delta-JEPA was
  read in full HTML; ActSWM and Dueling World Models are abstracts, and neither
  abs page yields a metric definition, seeds, hardware or code.
- **The UED closure rests on one paper's full text plus one abstract.** ATLAS
  (2511.12706) was fetched at abs level only and its difficulty criterion is not
  extractable from it; the negative conclusion is carried by 2602.09813's
  experimental section, which is explicit.
- **N2's decisive quantity — cross-modal redundancy — is not operationalised by
  its own source.** §2 leads with that rather than burying it.
- No non-English sources. No conference main-track enumeration (dropped
  permanently, week 4; still dropped, still an acknowledged gap).

---

## 2. NOMINATIONS

Three. **One is a diagnostic on the seated learning core and is the only one
that implies any spend; two are cautions aimed at fusion rows that are decided
this week and next.** Each states its arXiv primary category, its evidence
class, its cost on **our** substrate, and both sides steelmanned.

---

### N1 — Three independent 2026 groups name the same silent failure of latent world models, and it is the one failure `A4`'s mandatory diagnostic does not look for

**Sources — a convergence, not a paper:**

| | | |
|---|---|---|
| **ActSWM** — [arXiv:2607.26712](https://arxiv.org/abs/2607.26712), cs.RO, 2026-07-29 (rev 08-15), Gan, Zeng, Cheng, Song, Tang, Wang | **abs [V]** | names the failure mode |
| **Delta-JEPA** — [arXiv:2606.31232](https://arxiv.org/abs/2606.31232), cs.AI, 2026-06-30, Zhang et al. | **abs + full HTML [V]** | supplies the probe |
| **Dueling World Models** — [arXiv:2608.06706](https://arxiv.org/abs/2608.06706), cs.LG, 2026-08-07, Li, Fei, Zhou, Hayashi | **abs [V]** | supplies the readout-only fix |

None is cited anywhere in this repo.

**The verified claim [V].** ActSWM states the failure in one sentence, quoted:
*"We identify **Context Collapse**, a failure mode in which autoregressive
latent predictors maintain high similarity to future states while producing
**nearly indistinguishable futures under different action sequences**."* Its
principle names two measurable quantities: *"a planning-useful latent dynamics
model should keep **alternative-action futures distinguishable** and make the
**action associated with each local transition recoverable**."* Delta-JEPA
arrives independently — *"reconstruction-free joint-embedding objectives can
collapse to **action-insensitive representations**"* — and supplies the second
quantity as a module (LDAD, which *"reconstructs the executed action from the
latent displacement between consecutive observations"*). Dueling World Models
arrives from a third direction (distractor rejection) with a fix that is
*"only a subtraction at readout time"* and *"applies unchanged to any
action-conditioned world model"*.

**Why this lands on `A4` specifically, and why nothing here looks for it.**
`A4` is `WorldModelCore(decoder=False)`: actions enter through
`nn.GRUCell(LATENT + ACTION_DIM, DETER)` and the latent predictor maps the model
state to the next latent against an EMA target. Its **named** silent failure is
representation collapse, and its mandatory diagnostic is effective rank +
per-dimension variance. **Context Collapse is a different failure with a healthy
effective rank**: the representation can be full-rank and the *futures* still
action-independent. Nothing in `LC.03`, in `cores.py`, or in `LC.07`'s
registration measures whether the predictor's output depends on the action.

**And the seat's own evidence does not certify it.** `A4` was seated on
`life_gain` (t_null **4.65**, t_twin **4.00**). That number is produced by an
actor–critic actor reading the model state; **nothing in it requires the latent
predictor to condition on the action at all.** The representation could be doing
all of the work while the world-model half is inert. The seat is named for a
world model; the measurement certifies a representation.

**Which spec it enters.** `LEARNING_CORE.md` §5.4, as a **mandatory diagnostic
on `A4` beside effective rank** — and the natural host is **`LC.07`**, the
seat's registered challenger, so the scale-transfer run also certifies action
sensitivity at zero marginal compute. Two readouts on a trained model, no new
training, both with a must-fail side that already exists in the rig:

1. **Alternative-action rollout gap.** From a state, roll the latent predictor
   H steps under two different action sequences; report the latent distance,
   normalised by the distance between rollouts of the *same* action sequence
   from different states. Context Collapse is that ratio → 0.
2. **Action recoverability.** Fit a small probe from the latent displacement
   `ẑ_{t+1} − z_t` to `a_t`; report R² per action dimension. This is
   Delta-JEPA's LDAD **used as a probe, not as a loss.**
   **The must-fail control is free:** `A4`'s own untrained twin already runs in
   the rig and must score near zero on both. A diagnostic that cannot fail on an
   untrained network is measuring nothing — this desk's standing rule, and here
   the negative control is already built.

**Cost on our substrate.** `A4` = **861,545 params [M]** (§6 — not the 1,370,000
this desk quoted last week). The readouts themselves are a rollout sweep plus a
linear fit: seconds. What they need is a *trained* `A4`, and **no weights exist
on disk** — `lc03_curves_seed*.json` holds `life_spans`, `params`,
`optimiser_steps`, `process_time_s` and no tensors [M]. So, honestly:

- **(a) bolted to `LC.07`: zero marginal compute** — but `LC.07`'s venue is
  itself an open decision (`D24`, ~618 core-h on CPU, the checkpoint branch
  refused 09-06), so this route is hostage to a decision that is not the
  builder's to hurry.
- **(b) standalone: ~4.8 core-h per seed** — `wm-latent` recorded
  **17,280.3 process-seconds** per arm-seed at LC.03 v2's 1× envelope [M] →
  **14.4 core-h for 3 seeds**, inside `D4`'s frozen `CPU_DAYS` cap, no GPU
  quota. This is not a free nomination and I am not going to price it as one.

**Why it might WIN (falsifiable).** If the rollout gap is ~0 and action
recoverability is at twin level, then three things are hollow simultaneously and
for one reason: any planning arm (the DP family), `disagree`-style curiosity
built on forward-model ensembles, and `LC.07`'s claim that the *world model*
survives scale. That is a single cheap readout deciding the standing of a seat
and two downstream families — and it is pre-registerable with a must-fail
control that already exists. If instead `A4` is action-sensitive, the seat gets
a certificate it currently lacks.

**Why it might LOSE (steelmanned). Five.**
1. **All three papers live in pixel regimes with long horizons** — Minecraft
   closed-loop planning, ViT-Tiny visual control, Atari distractors. `A4` sees a
   6-dim drive vector and rays at 861,545 params, and in a body with strong
   immediate action effects Context Collapse may simply not arise.
   `LESSONS.md` records three transfer failures on this box and all three are
   *"the published regime was not ours"*.
2. **Delta-JEPA never measures the insensitivity it names.** Table 2 compares
   two decoders by **downstream planning success**; I could find **no
   action-decoding accuracy anywhere in the paper**. So the fixes are evidenced
   and the *failure* is asserted — and only ActSWM claims to measure it directly
   (*"preserves larger action-dependent rollout gaps"*), from an abs page with
   no metric definition and no numbers. **This is the objection I would press
   first**, and it is the mirror of week 1's lesson: an abstract is a claim
   about a table, and here one of the tables is missing.
3. **None of the three reports hardware, wall-clock, parameter counts or code.**
   Third consecutive sweep in which that is true of the front-1 nomination class.
   Delta-JEPA: 4 pixel tasks, **3 seeds**, λ=10.0, ViT-Tiny + 6-layer causal
   transformer, and nothing else.
4. **It costs 14.4 core-h unless `LC.07` runs**, and `LC.07`'s venue is open.
5. **The likely outcome is a green light**, which is worth having on a seat
   everything now depends on but is not a discovery. A diagnostic that mostly
   passes is still the right spend here; it is not a headline and I will not
   dress it as one.

---

### N2 — NOT AN ARM: a 37-dataset measurement that a fusion bakeoff on a saturated battery tests nothing — and names a *second* condition the redesign has not considered

**Source:** *MULTIBENCH++: A Unified and Comprehensive Multimodal Fusion
Benchmarking Across Specialized Domains* —
[arXiv:2511.06452](https://arxiv.org/abs/2511.06452) **v3, 2026-05-06**,
primary category **cs.LG**, AAAI 2026. Xue, Zhang, Xue, Liu, Wang, Han.
**Full HTML read.**

**The verified claim [V], quoted.**

| | |
|---|---|
| the conditional | *"the marginal utility of advanced fusion is strictly positive **when and only when cross-modal redundancy is low**; otherwise, naive concatenation attains near-optimal performance"* |
| the saturation half | *"Early fusion like Concat and TF collapses on weakly-aligned modalities, yet **shows no gain on saturated tasks** (SIIM-ISIC, eICU)"* — methods cluster near **97.8 %** on SIIM-ISIC, ~**90 %** on eICU |
| the positive half | *"Our algorithms … yield the highest accuracy on **25 of 37 datasets**, routinely beating plain concatenation"* |
| seeds / hardware | **3 seeds**; *"same hardware configuration"* — unspecified |
| tasks | 37 datasets, mixed classification / MSE regression / ranking. **Nothing embodied, no world-model objective** |
| **redundancy as a metric** | **NOT operationalised. No formula, no computed values.** |

**Where it enters.** `ub10-seed-fragility-and-saturated-battery` (ROUTED
2026-09-01, **DUE 2026-09-08**). Not an arm — a pre-registration on a decision
that happens tomorrow. `UB.10` attempt 1 measured exactly this condition: **A0
reads slot 1.0 on all three seeds, the winner ties it** (paired_boot_lo
−0.0104, ranking gap 0.0), and the row's own words are *"at this budget the
fused battery discriminates nothing among healthy arms."* The row's option list
is (i) harden the battery, (ii) a per-arm stability conjunct, (iii)
SCORED-AND-INELIGIBLE. **All three address saturation and none addresses
redundancy** — and this paper's measurement is that de-saturating is
*necessary but not sufficient*: off the ceiling, if the modalities carry the
same information, concatenation still ties every fusion arm.

**The prediction, offered so it can be wrong.** W0's senses plausibly *are*
mutually redundant — rays, drives and touch reporting the same physical events
in the same body. If so, hardening the battery alone reproduces the tie. That
is checkable at the cost of the redesign already planned, and stating it before
the redesign is the only time it is worth anything.

**Cost.** Zero. It is a caution and a pre-registration, not a method.

**Why it might LOSE (steelmanned). Five.**
1. **Its decisive quantity is not computable from the paper.** Redundancy is
   named, never defined or measured. A pre-gate on a quantity we would have to
   invent ourselves, applied to our own battery, is *precisely* the Goodhart
   risk this desk refused this front over twice. **Lead objection, and it is
   serious.**
2. **37 supervised datasets, none embodied**, none a world-model objective.
   Third consecutive sweep in which this family has not left classification.
3. **"When and only when" is the authors' summary of their own leaderboard**,
   not a measured redundancy sweep. It is an interpretation with 25-of-37 behind
   it, and interpretations do not transfer the way tables do.
4. **`UB.10`'s row reached the saturation half unaided.** The marginal content
   here is one axis, not a finding, and the row is already smarter about its own
   rig than this paper is about ours.
5. **3 seeds, hardware unspecified**, against `UB` §1.8's Agarwal standard.

---

### N3 — NOT AN ARM: the newest balancing paper measures that symmetric fusion *damages* the dominant pathway — which turns this desk's Goodhart objection into a required control on the bakeoff the Review has now ordered

**Source:** *Mitigating Strong-Modality Collapse in Multimodal Learning via
Inverted Asymmetric Fusion* —
[arXiv:2608.26879](https://arxiv.org/abs/2608.26879), **2026-08-27**, primary
category **cs.LG**. Kenneth, Khosmood, Edalat. **Full abstract read.**

**The verified claim [V], quoted.** *"IAF **preserves the dominant modality's
internal accuracy at its unimodal ceiling** across all tested configurations,
whereas **symmetric fusion degrades it by up to 18.5 %** on MultiHuSE"*; the
mechanism is measured at pathway level — *"the text-pathway accuracy drops from
**74.9 % to 56.4 %** after fusion in one such setting"*. IAF *"improves over the
strongest unimodal baseline by up to 8.25 %."* Three datasets (MultiHuSE,
UR-FUNNY, MUStARD), **classification**, **no seeds, no hardware, no parameter
counts, no code stated**.

**Where it enters, and why the framing changed this week.**
`t402-touch-drowns-audio-at-the-fusion-boundary` (ROUTED 2026-09-05, **DUE
2026-09-13**) orders the repair as a bakeoff and names the candidate arms:
*"per-modality gradient normalisation, loss reweighting, modality dropout
schedules."* **Every one of those is a symmetric-balancing move**, and our
measured fingerprint is touch (~2.9e-3) drowning audio (~1e-4) at ratio
**30.12** against a 10× gate.

For two sweeps I refused to nominate this family because installing a balancer
to pass `T4.02` would Goodhart the gate. That refusal was an *argument*, and
`SYSTEM.md` law 3 says arguments do not decide. **This paper converts it into a
measured mechanism**: symmetric fusion is observed to buy balance by degrading
the dominant pathway, 74.9 → 56.4 at pathway level. So the nomination is not the
method — IAF's own remedy protects the dominant modality, which is the *opposite*
of what `T4.02` wants and is not an arm we would want. **The nomination is one
control on the ordered bakeoff:**

> every balancing arm must report per-modality pathway decoding **before and
> after fusion**, and an arm that improves `max_modality_grad_ratio` while
> degrading touch's own pathway has bought the gate, not the capability.

**Cost.** Zero new parameters; the pathway readout is a probe on a model the
bakeoff trains anyway. Same discipline as wk1's entity-collision protocol and
wk5-N3: a control attached to a measurement somebody else already ordered.

**Why it might LOSE (steelmanned). Four.**
1. **Three text/audio-visual classification datasets**, dominance always in
   text or in a two-stream setting. Our dominator is **touch**, in a
   five-sense world-model objective. The transfer is by analogy, and week 3's
   lesson is that an analogy is not arithmetic.
2. **No seeds, no hardware, no params, no code** — the weakest provenance of
   anything I am nominating this week, and I would not nominate it if it implied
   any spend.
3. **A pathway-decoding control could itself be gamed** by an arm that keeps
   touch's pathway intact and starves audio further; the control catches one
   failure direction, not both. It should be stated as necessary, not
   sufficient.
4. **`T4.02`'s gate is constitutional here**, not performance-derived — GOAL.md
   stage 4, *"no modality collapse"*. A paper arguing that protecting the
   dominant modality is good is arguing against our constitution's direction,
   and importing its frame wholesale would be exactly the authority-import wk4
   refused. I am importing its **measurement**, not its recommendation.

---

## 3. WATCHLIST

Every entry records its arXiv **primary category**.

**Resolved this sweep and now DELETED** (recorded so a seventh sweep does not
re-open them):

| item | resolution |
|---|---|
| **2607.22430** identifiability — *queued #2, second and last attempt* | **CLOSED NEGATIVE and DROPPED WITH CAUSE**, by the stated method (HTML + experimental section). **The spectral separation margin is NOT estimable post hoc.** `γ_rep(π) := λ_min(R_π) − λ_max(R_π)²` is computed from the *true* system matrices, known by design (*"spec(AA^⊤ + BB^⊤) = {0.6, q}, γ_rep(q) = q − 0.36"*); **no procedure for estimating it from a trained encoder or predictor is given.** The rig is two-dimensional synthetic systems, 8-layer SiLU MLPs at width 512, 100,000 one-step transitions, five runs per observation map, **no hardware, no code**. So it cannot be an `A4` diagnostic and it does not supply `UB.11`'s missing certificate. Week 4 called this *"the highest-value fetch outstanding on any front"*; two sweeps later the honest answer is that it was worth one fetch, not three, and it is dropped the way conference enumeration was. |
| **Simulus** (2502.11537, cs.LG) — *sixth sweep* | **DROPPED WITH CAUSE**, per this desk's own rule and the Review's endorsement. Five sweeps carried on two numbers — a parameter count and a per-step wall-clock — that have not appeared and are not going to. Its prioritised-replay component survives separately as wk2-N2 → `NE.05` (live, unrun) and does not depend on this entry. |
| **wk5-N1's Lean repository** — *queued #3* | **CLOSED: there is none stated.** The abs page, the comments field and a targeted search return no repository URL; the comments read verbatim *"Theoretical paper; empirical validation of the stated predictions is left to separate work. 28 pages, 4 figures, 4 tables."* The Lean 4 verification is **described in an appendix, not downloadable** — which is a weaker object than week 5 implied when it called that verification status *"a status no benchmark table on this desk has ever had"*. It still has no `sorry` obligations by the authors' account; we simply cannot run it. Nothing else about wk5-N1 changes. |

**Carried and re-examined:**

| item | cat | status |
|---|---|---|
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | Unchanged, third sweep. Rollout-horizon deterioration in the **fully observable state space** across RSSM / Transformer / Diffusion / Neural-ODE — our regime. Still no compute cost, no hardware, no environment size, no code. **Promote on: any statement that a domain runs on CPU.** Now *more* relevant, because N1's question (does the latent predictor stay useful over a rollout) is the question this paper measures on our observation type. |
| **ForageWorld** ([arXiv:2506.06981](https://arxiv.org/abs/2506.06981)) | cs.AI | **Kept, and its moment is now.** `W1.03` (traps, delays, irreversibility, with a features-removed twin) and `W1.01` (passivity dies) are the specs waiting on `w1-world-edit-window` (DUE 09-13), and ForageWorld remains the closest published existence proof of a world with those features — depleting/diffusing food, pursuing predators, a **sleep action gated on energy < 50 % that immobilises while restoring**. Still Craftax/GPU (the axis `SURVIVAL_WORLD.md` §2.2 ruled out) and a gridworld where W0 is a body. **Design reference only; it is not an arm and never was.** |
| **Eywa** ([arXiv:2605.30771](https://arxiv.org/abs/2605.30771)) | cs.CL | Unchanged and **not re-examined this sweep** — front 3 is on a two-week cadence. Still the only front-3 item with *"zero LLM calls inside retrieval"*; still blocked on whether the **write** path is generative. |
| **ScrubJay-MEM** ([arXiv:2608.04746](https://arxiv.org/abs/2608.04746)) | cs.CL | Unchanged, **not re-examined** (front 3 cadence). The oracle — perishability is a property of the *remembered thing*, and a hand-specified type→decay map needs no LLM — remains the portable half, routed at `Forgetting.py`. |
| **IIBalance** ([arXiv:2603.17347](https://arxiv.org/abs/2603.17347)) | cs.MM | Unchanged. Capacity-based budgets rather than forced equality. Still classification, still no modalities named, still no extractable numbers. **Now adjacent to N2/N3**: three papers in two sweeps contesting the balance objective from three directions, none of them outside classification. |
| **Curiosity-Critic** ([arXiv:2604.18701](https://arxiv.org/abs/2604.18701)) | cs.LG | Unchanged. Weaker-evidenced sibling of LPM (2509.25438), which `CURIOSITY_BAKEOFF.md` already cites. Still one stochastic gridworld, still no continuous control. |

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **PRIME** ([arXiv:2607.16858](https://arxiv.org/abs/2607.16858), 2026-07-18) | cs.LG | Furutanpey & Dustdar. *Principled Direction-Free Intrinsic Motivation through Model-Free Epistemic Free-Energy Estimators.* **Explicitly model-free** — *"without fitting an explicit next-state predictor"* — which is the misfit that kept wk3's CIG out of the model-free slot. Epistemic value from a pseudocount, aleatoric variance from a probe-based penalty, a window-freeze giving a stationary Bellman operator. **15 seeds** — the best seed discipline in this sweep. | **A continuous-state result.** Both environments are **10×10 grids** (Butterflies 250k steps, Maze 200k). Butterflies: PRIME **3.47 ± 0.10** vs RND 3.26 ± 0.14 vs random walker 2.29 ± 1.21 — a real but small win. **Maze: RND reads 1.00 ± 0.00 and beats PRIME's 0.79 ± 0.11** — their own second environment is saturated, and RND is a **must-fail control** in `CURIOSITY_BAKEOFF.md`. Decisive objection for us: **the epistemic term is a pseudocount**, which is free on a 10×10 grid and undefined on a continuous body state without a hashing or density scheme the paper never has to build. No hardware, no wall-clock, no params, no code. **Not nominated.** |
| **POBAX** ([arXiv:2508.00046](https://arxiv.org/abs/2508.00046), 2025-07-31) | cs.LG | Tao, Guo, Allen, Konidaris. A benchmark suite built on a **certified-headroom** discipline: *"An environment is memory-improvable if there exists a gap between the performance of agents with less or more state information."* Floor / ceiling / test-agent triple; **30 seeds**; **code released** (`taodav/pobax`); and they **excluded a domain that failed the check** (DMLab maze_id = 03). | **Nothing, and I want to say why plainly.** This is the published version of `W1.01`'s discipline, arrived at independently for a different property — and **`W1.01` is already the stricter instrument**: it carries a *deliberately-benign twin world that must show NO gap*, which POBAX has no counterpart for. Corroboration is not news, this desk has said so four times, and the honest report is that our own design is ahead of the nearest published one. Also **13 months old** (outside the window) and JAX/GPU. Recorded so a future sweep does not mistake it for an instrument we lack. |

---

## 4. DISPOSITION OF PRIOR NOMINATIONS

| nomination | entered | status now |
|---|---|---|
| **wk4-N3 · infant motor noise → `W0.DIAG`** | W0 diagnostic | **RUN AND PASSED (2026-08-31, attempt 3, 3 seeds).** The first field-watch nomination to become a number. Claim branch *"correlation buys life through food"*; `gain_up` 12.12 ± 1.20 vs `gain_random` 0.0095 ± 0.39. Its finding then **re-pointed the entire W0/W1 design** (§0). Nothing further owed. |
| **wk5-N3 · RWG/PIC/POIC are broken** | binding control on `W0.DIAG` | **CARRIED AND CLEARED (t ≈ 11)** before the W0 reading was believed, exactly as ordered. Discharged. |
| **wk5-N1 · SIGReg vs VICReg (2607.13612)** | `LEARNING_CORE` §5.4 selection criterion | **ACCEPTED, split, and the free half is ORDERED FIRST**: read `A4`'s existing effective-rank/variance curves for Remark 1's signature (variance floor healthy, effective rank flat) before any arm is built. **Not yet executed.** The Lean repository question is closed negative (§3). **N1 above is the second diagnostic on the same seat and the two should be read together** — one asks whether the latent collapsed, the other whether the *futures* did. |
| **wk5-N2 · prosociality by coupling (2604.10760)** | `NE.07` | **ARM DEFERRED** (the paper's own load sweep predicts it fails in W0's high-load regime); **CONTROL ACCEPTED** — the shuffled-partner lesion registers on `NE.07` as a strengthening. `NE.07` and `NE.02` still have never run, and that caveat stands as the Review wrote it. |
| **wk4-N1 · spectral-radius constraint (2607.19719)** | `A4` variant | **ACCEPTED, narrowed, unblocked by D10's firing.** Register the design, do not dispatch. Unrun. |
| **wk4-N2 · PSG-JEPA (2608.06799)** | `A4` ×2 | **ACCEPTED as two arms**, `ℒ_dynamic` and `ℒ_static` as each other's control, my prediction pre-registered. Sequenced after D9 — which **fired 2026-09-01 as a PARK**, so the body fork is parked rather than resolved and this stays sequenced behind a park. |
| wk1 · anti-collapse regularisers → `A4b`/`A4c` | `A4` variants | **LIVE, unrun.** Now three sweeps promoted without a new experiment. |
| wk1 · certificate-gated identifiability → `UB.11` pre-gate | UB.11 | **LIVE, unrun.** `UB.11` blocked behind a `UB.10` verdict the 09-08 redesign must first make reachable. **2607.22430 will never supply the certificate** (§3), so wk1's protocol is the only route on the desk. |
| wk1 · interoceptive precision (2608.04232) | `NE` §2.4b | **LIVE, unrun.** Still the cheapest item on the desk with released code. |
| wk1 · entity-collision protocol (2605.29630) | `MR` §2 | **LIVE, unrun.** |
| wk2 · the whiff clock → `SM.02` | SM.02 | **HELD, correctly — `SM.02` is PARKED** (§0). Not a queue defect and not reported as one. |
| wk2 · RPE-prioritised replay → `NE.05` | NE.05 | **LIVE, unrun.** |
| wk3 · CIG (2605.20878) → `A3` | `A3` | **Remains DEMOTED** (`wm-efe` t = 2.05). |
| wk3 · Optimistic World Models (2602.10044) → `A2` | `A2` | **Remains DEMOTED, hard** (`dreamer-xs` t = −0.94). |

---

## 5. NO-ACTION — fronts where nothing cleared the bar

**FRONT 5 · WORLDS — no arm, and the queued question is CLOSED WITH A NEGATIVE
ANSWER that `W1.01`/`W1.03` needed before the world-edit window.** Week 5's
highest-value outstanding fetch asked: *does the regret-against-a-student
framing give a computable environment-discriminability score, readable on W0 for
CPU-minutes?* Fetched, full HTML. **No, twice over.** (i) 2602.09813's teacher
score is **not regret** — it is learning progress plus a fairness term,
`R = Σᵢ(pᵢ′ − pᵢ) − η·cv(·)`. (ii) Computing it is **the opposite of cheap**:
*"each transition requires the student to complete a full training horizon of C
timesteps in the generated environment"*, with ~2,500 environments to train the
teacher, on *"a single NVIDIA GeForce RTX 3090 GPU and 16 CPUs"* (code released).
And the regret line's own literature states the structural reason [s]: regret
*"relies on the optimal policy, which is generally not available"*. ATLAS
(2511.12706, cs.LG, Nov 2025) is abs-level only and yields no extractable
criterion. **So: wk5-N3 said the two published difficulty metrics invert; this
sweep says the only other family that claims one prices it at a student training
run per level. There is no cheap published environment-discriminability score,
and `W1.01`/`W1.03` will have to be our own instruments.** That is a real answer
to a real question and it is the most useful thing front 5 produced. The
40-entry survival enumeration otherwise returned LLM agent societies, narrative
benchmarks and game environments; nothing with a body under homeostatic drive.

**FRONT 4 · CURIOSITY — nothing, and the enumeration is now itself the
finding.** 40 entries on intrinsic motivation / open-endedness / autotelic
agents: **28 of 40 are pure LLM or text-reasoning work**, and **zero** have a
body under homeostatic drive; **zero** evaluate an intrinsic reward against a
random or noise baseline. PRIME is watchlisted with a decisive objection
(pseudocounts are free on a 10×10 grid and undefined on a body). The open-
endedness half is closed above. Two sweeps of deep coverage on this front have
produced no arm, and the reason is not that we searched badly: **the 2026
intrinsic-motivation literature has moved into language agents, and the part
that has not is a gridworld.** `T3.06`'s venue repair-arm pick (DUE 09-09) will
not get help from this literature, and should be decided on `CURIOSITY_BAKEOFF`
§O1's own dual-comparator rule, which wk5 showed was already written.

**FRONT 2 · FUSION — no arm for the third consecutive sweep, but the posture
changed and §2 says how.** N2 and N3 are cautions, not arms, and I want the
distinction on the record: the Review has now **ordered** a balancing bakeoff
(`t402`, DUE 09-13), so refusing to nominate is no longer a position I can hold
— what I can do is attach the control that keeps the ordered bakeoff honest.
The family itself is unchanged: MULTIBENCH++ (37 datasets), IAF (3 datasets),
IIBalance, PDMP — **not one has left supervised classification**, in the third
consecutive year of trying.

**FRONT 3 · MEMORY — NOT SWEPT, by the endorsed two-week cadence. Next due
2026-09-14.** This is the plan working, not a gap, and I am naming it rather than
letting silence stand. **But the interval put a new scar on the front and it
should point the 09-14 sweep:** `ME.1` FAILed on 09-06 with
`distractor_abstention` **0.0000 ± 0.0** on three seeds — the store confabulates
the nearest neighbour on *every* absent-target cue, and had read 1.0000 for 29
days against a control whose cues a keyword filter passes. `ME.11` A–D all FAIL,
E/F VOID. That is an **abstention** failure, and abstention is the one place
where wk3 recorded that our own bakeoff doc is *ahead* of the field (conformal
coverage, Clopper–Pearson, Learn-then-Test, E-AURC, against a generator-side
literature). So 09-14's front-3 sweep has one question, not a survey: **has
anything published a calibrated abstention floor for extractive retrieval on a
personal corpus?** If the builder wants that earlier than 09-14 because
`me1-similarity-floor-never-abstains` is DUE 09-13, that is their call to make
and I would not argue with it.

**SMALL-MODEL END — nothing in window, fourth consecutive sweep.** The nearest
hit, Tiny Recursive Control (2512.16824), is out of window, ~1.5 M parameters,
GPU-millisecond framing, and its comparison class is language models. Our own
`ppo-needs` at **135,961 params [M]** is smaller than anything the small-model
literature is currently proud of, which is the 54K-beat-57M lesson still holding.

**BIOLOGY-AS-ORACLE — one shelf item has a 2026 paper and it is
constitutionally inadmissible, which is worth recording once so it is not
re-found.** `GOAL.md`'s shelf names *play as safe rehearsal*; **Playful Agentic
Robot Learning** ([arXiv:2606.19419](https://arxiv.org/abs/2606.19419), cs.RO,
2026-06-23, **code released**) is exactly that title with real numbers (+20.6
and +17.0 percentage points over CaP-Agent0 on LIBERO-PRO and MolmoSpaces,
against no-play and random-play baselines) [c]. **It is not admissible here and
the reason is structural, not a preference:** the play is performed by an LLM
coding agent that proposes its own tasks, writes robot-code policies, diagnoses
its failures and distils them into a persistent code skill library. That is the
borrowed model *inside* the learning loop — the puppet risk `LG.00` exists to
catch, and the opposite of `GOAL.md`'s decision that the LLM is a **parent in
his world**, not a component in him. Jack's play must be learned by the thing
that lives. The rest of the shelf — innate reflex priors, pain as a fast signal
distinct from reward, critical periods — returned nothing measured in window.

---

## 6. A FINDING IN OUR OWN ARTIFACTS — the learning-core parameter table is stale on all five arms, including two rows marked MEASURED

Week 3's rule: a scout reading our own ledger has no abstract to doubt, so it
must carry the arithmetic. Here it is, reproducible in two lines.

    /data/venvs/jackthelearner/bin/python -c "
    from experiments.cores import build_arm, n_params, CANDIDATE_ARMS
    for a in CANDIDATE_ARMS: print(a, n_params(build_arm(a)))"

| arm | `LEARNING_CORE.md` §5 table | recorded by LC.03 v2, all 3 seeds | rebuilt at HEAD | delta |
|---|---|---|---|---|
| `ppo-needs` | 120,841 **[M]** | 135,961 | **135,961** | **+12.5 %** |
| `ppo-lp` | ≈ 211,000 [C] | 144,794 | **144,794** | −31.4 % |
| `dreamer-xs` | 1,896,047 **[M]** | 1,671,065 | **1,671,065** | **−11.9 %** |
| `wm-efe` | ≈ 1,900,000 [C] | 2,052,265 | **2,052,265** | +8.0 % |
| **`wm-latent` (the seated arm)** | ≈ 1,370,000 [C] | 861,545 | **861,545** | **−37.1 %** |

HEAD's constructor reproduces the recorded artifact **exactly, on every arm**,
so the artifact is right and the document is stale. Two rows are marked **[M]**,
i.e. measured — and a figure marked measured that disagrees with what the
registered run recorded is a different kind of defect from an estimate that
drifted. The likely cause is benign and is the builder's to confirm, not mine to
assert: `cores.py` records that `LC.02` was the first spec to push a gradient
through these arms and repaired a critic that could not accept its own shared
state, which would change parameter counts after the table was written.

**Why it is worth a section rather than a footnote.** `LEARNING_CORE.md` §5
states its own consequence: *"only the two PPO arms are inside its W0 soft
target (750K) — … if a world-model arm wins, the soft target moves and the
reason is on the record."* A world-model arm **did** win and **is** seated. On
the document's number the concession is 1,370,000 / 750,000 = **83 % over
target**; on the measured number it is 861,545 / 750,000 = **14.9 % over** [C].
Same decision, very different size of concession — and `LC.06` (*"the simplicity
budget is enforced, not promised"*) is the spec that would read exactly this
number.

**And one of the stale figures is mine.** Week 5 costed its lead nomination with
*"`A4`'s ≈1.37 M is unchanged"*. The arm is 861,545. The number was inherited
from the document in good faith and never checked against the run that seated
it, which is the same failure this desk logged on 08-24 in a harsher form — the
difference being that a document is a more trustworthy-looking source than a
search engine, and therefore a quieter one to be wrong from. **Nothing is
decided here**: the numbers above are a measurement, the reconciliation is the
builder's, and no threshold moves on my account.

---

## 7. What this report does NOT claim

- **No arm here has been run.** Every number in §2 and §3 is someone else's
  measurement on someone else's hardware. The §6 numbers and the `LC.03` cost
  figures in N1 are **ours**, marked **[M]**, with the commands shown.
- **N1 is a DIAGNOSTIC, not a replacement core.** It does not propose to unseat
  `A4`; it proposes to certify a property of `A4` that its seating measurement
  does not cover. If the reading is green, the seat is stronger and that is a
  fine outcome.
- **N2 and N3 are explicitly not arms**, and both say so in their titles. N2's
  decisive quantity is not computable from its own source, and §2 leads with
  that rather than hiding it. N3 imports a measurement and explicitly refuses
  the recommendation attached to it.
- **The action-insensitivity story does NOT explain `DP.05`.** The obvious and
  attractive move was to claim that *"deeper lookahead buys less"* is the
  signature of an action-insensitive learned model. I checked before writing:
  `DP.05`'s planner uses **the simulator itself** via full state
  snapshot/restore — *"there is no copy at all"* — so there is no learned model
  in that result to be insensitive. The connection is refuted and it is recorded
  here rather than deleted, because week 3's lesson is that the dangerous story
  is the one the scout writes itself.
- **The `W1.00` reading in §0 is the builder's measurement, not mine.** I quote
  it because it tests a claim I made last week; I did not run it and I do not own
  its interpretation.
- **Front 3 was not swept** and §5 says so in its own heading rather than in a
  caveat.
- **Verification is uneven and marked:** N1 — one full HTML (Delta-JEPA, and its
  key number is *absent*), two abstracts, **no hardware/params/code in any of
  the three**; N2 — full HTML, **redundancy not operationalised**; N3 —
  **abstract only, no seeds, no hardware, no code**; PRIME full HTML; POBAX full
  HTML; 2607.22430 and 2602.09813 full HTML (both closing negative); ATLAS,
  IIBalance, ForageWorld, Eywa, ScrubJay-MEM abstract-level; the regret-needs-
  an-optimal-policy statement in §5 marked **[s]**.

---

## 8. Queued for next sweep (**not before ~2026-09-14**)

1. **FRONT 3 RETURNS on its two-week cadence, with one question rather than a
   survey** (§5): has anything published a **calibrated abstention floor for
   extractive retrieval** on a personal corpus? `ME.1`'s
   `distractor_abstention` 0.0000 is the scar it must speak to.
2. **Did wk5-N1's free half get read?** The effective-rank-vs-variance-floor
   signature on `A4`'s existing curves was ordered FIRST and costs nothing. If
   it has been read, N1's rollout-gap diagnostic should be designed beside it as
   one pass over the same seat, not two.
3. **`W1.01` and `W1.03` after the world-edit window (09-13).** The literature
   supplies no cheap environment-discriminability score (§5), so the question
   for next week is narrower and it is about *design references only*: does
   anything published measure a passive-vs-oracle headroom gap **in a body**
   rather than a gridworld? ForageWorld and POBAX are the current answers and
   both are GPU gridworlds.
4. **ActSWM's metric definition.** N1's weakest link is that the only paper
   claiming to *measure* Context Collapse gives no formula on its abs page. One
   HTML fetch decides whether the rollout-gap readout has a published definition
   to borrow or whether we are inventing it. **Highest-value single fetch
   outstanding.**
5. **Watch the three decisions that consume this file:** `ub10` (09-08),
   `t306`/`w0-kills-a-forager` (09-09), and the 09-13 pile —
   `w1-world-edit-window`, `t402`, `me1-similarity-floor`. Ten live rows share
   09-13 against a measured capacity of one per cycle; if that pile slips, N2
   and N3 arrive before the decisions they were written for, which is the one
   thing this desk can do about timing.
6. **NOT queued, deliberately:** conference proceedings (dropped wk4);
   2607.22430 (dropped with cause, §3); Simulus (dropped with cause, §3);
   PIC/POIC successors (wk5-N3 closed the family); UED as an instrument
   (closed negative this sweep — it may return as a *design* reference, never
   again as a cheap score).
