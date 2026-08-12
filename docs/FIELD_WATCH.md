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

**Sweep date:** 2026-08-12 · **Window:** ~2026-02 → 2026-08 (6 months)
**Scout:** field watch, week 3.

**This sweep executed the previous one's queue item #1: a genuine six-month
sweep of fronts 1, 2 and 3** (learning cores, multimodal fusion, memory), which
week 2 deliberately did not touch and carried forward verbatim from week 1. It
also closed queued #2 (Optimistic World Models, which is now nominated), #4
(SmallWorlds), and #3 (conference proceedings — **now blocked for a reason, see
§4**).

**The cadence needs stating, again and honestly.** Week 1 was 2026-08-10, week 2
was 2026-08-11, this is 2026-08-12. Three sweeps in three days against a
six-month window. The justification for running today rather than on 08-17 is
specific and expires immediately: fronts 1–3 had been swept exactly **once**,
eight weeks of new arXiv content had never been read against them, and the queue
put that first. That mandate is now spent. **Fronts 1–3 have been swept properly
and should not be re-swept before ~2026-08-19**; a fourth consecutive daily sweep
would be theatre.

One event in the interval also bears on the report: **`VO.01` FAILed at attempt 2
on 2026-08-11 (23.84 s)** — after last week's sweep declared NO-ACTION on voice.
§2 reports what the acoustics literature says about it, and what it does *not*
say.

---

## 0. Coverage — what was actually searched, so the gaps are visible

| Front | Searched this sweep | Depth reached |
|---|---|---|
| **1 · LEARNING CORES** | world-model sample efficiency; Dreamer successors incl. Dreamer 4; SSM/RSSM alternatives; JEPA-family control results; optimistic exploration | **full HTML + tables** for the nomination (2602.10044); abstracts for 5 siblings |
| **2 · MULTIMODAL FUSION** | binding objectives; modality collapse; visuo-tactile/proprioceptive cross-modal prediction; unified tokenisation for VLA-world models | abstracts + search-level; **cross-checked against `UNIFIED_BRAIN_BAKEOFF.md` §1.2–1.5's existing citations** |
| **3 · MEMORY** | episodic/agent memory; consolidation; retrieval benchmarks; abstention & calibration | abstracts; **`HAKARI-Bench` (2606.22778) abstract fetched** |
| **4 · CURIOSITY & OPEN-ENDEDNESS** | intrinsic reward vs disagreement/RND; autotelic & lifelong curricula; unsupervised environment design | **full HTML + formulas + result tables** for the nomination (2605.20878) |
| **5 · WORLDS & EMBODIMENT** | survival/homeostatic sims; MuJoCo ecosystem; cheap embodied benchmarks; UED | **full HTML + throughput numbers** for MuJoCoUni (2605.24922), resolved as inapplicable |
| Biology-as-oracle | infant sleep consolidation / category abstraction; acoustic transmission ecology | secondary sources; **the VO.01-relevant acoustics checked against two meta-analyses** |
| Small-model end | tiny-model control results | abstract fetched (2604.07385) |
| Queued #2 (OWM full text) | **DONE** — numbers extracted, now a nomination | full HTML |
| Queued #3 (proceedings) | **attempted via the OpenReview API — now hard-blocked, see §4** | HTTP 403 |
| Queued #4 (SmallWorlds) | **DONE** — abstract fetched, stays on watchlist | abstract |
| Queued #5 (SM.02 / TA.02) | **checked the ledger: neither has run.** N1 of week 2 is still live and still un-moot | ledger read |

**Known gaps, stated so nobody assumes coverage:**
- Front 2 got **one search pass and a cross-check against our own citations**,
  not the full-text treatment fronts 1 and 4 got. If a fusion result landed that
  our existing vocabulary does not name, this sweep would not have seen it.
- **CIG's hardware and wall-clock (§1, N1) are in its appendix A.6 and I did not
  extract them.** That is the single most decision-relevant missing number in
  this report, because B4 is a throughput gate.
- Neither nomination's **code release is confirmed**. Both are marked [c].
- No non-English sources. No OpenReview main-track enumeration (§4).
- Week 2's nominations (whiff clock → `SM.02`; RPE-prioritised replay → `NE.05`
  S1) are **live and not re-litigated here**; neither spec has run.

---

## 1. NOMINATIONS

Two. Both are **loss-term / reward-term changes to arms this project has already
specified** — no new module, no new network, no new dependency. Each states the
source and its **arXiv primary category** (week 2's discipline finding, applied),
what is **[V]**erified (fetched and read), **[c]**laimed, or **[C]**omputed here,
which spec it enters, its cost on **our** substrate, and — steelmanned both ways
— why it might win and why it might lose.

---

### N1 — Ensemble disagreement is measured against the wrong context, and the fix reuses the ensemble we were already going to build

**Source:** *CIG: Exploration via Conditional Information Gain* —
[arXiv:2605.20878](https://arxiv.org/abs/2605.20878) **v1, 2026-05-20**,
primary category **cs.LG**. Joseph, Fechner, Stegmaier, Daaboul, Zöllner
(FZI Karlsruhe / KIT). **Full HTML read: formulas, algorithm, result figures.**

**Not cited anywhere in this repo.**

**The problem it names, in our own words.** `CURIOSITY_BAKEOFF.md` §1.1 and
`LEARNING_CORE.md` §3.3.3 both settle on the same estimator from opposite
directions: our `disagree` candidate is *"variance across a 5-member one-step
forward-model ensemble"*, and `LEARNING_CORE.md` §3.3.3 records that active
inference's epistemic term **is** expected information gain and **is** estimated
in the literature with exactly Plan2Explore's ensemble — which is why arm `A3`
(`wm-efe`) is specified as *"disagreement across a K=5 ensemble of latent
dynamics heads"*. One estimator, load-bearing in two separate bakeoffs.

CIG's claim is that this estimator conditions on the **replay buffer only**. It
scores each transition against everything the agent has ever seen and is blind to
what the *current rollout* has already probed — so a rollout that revisits its own
novel region collects the bonus repeatedly. Episodic methods (E3B) have the
opposite blindness. Its reward keeps both:

```
r_t = log( K_tt  +  σ²d  −  k_<t ᵀ K̃_<t⁻¹ k_<t )
         ^^^^      ^^^^     ^^^^^^^^^^^^^^^^^^^
      lifelong   aleatoric   prefix redundancy
      (= P2E)     ridge      (Cholesky of the disagreement kernel)
```

`K_tt` is **exactly Plan2Explore's per-step disagreement** — i.e. exactly what
`disagree` and `A3` already compute. The other two terms are arithmetic on top of
it.

**The verified claim [V].**

| | |
|---|---|
| backbone | **DreamerV2, unchanged**; *"identical training budget across CIG and P2E; any performance difference is attributable to the reward computation alone"* |
| ensemble | **M = 5** one-step latent MLPs — *"same ensemble as Plan2Explore; no bootstrap masks or diversity penalties"* |
| new hyperparameters | **none.** The aleatoric scale σ̂² is estimated post-hoc from ensemble-mean residuals (their Algorithm 2), not tuned |
| tasks | **12**: MiniGrid (MultiRoom N7S8, KeyCorridor S4R3, ObstructedMaze 2Dlh) and **OGBench continuous control** (AntMaze, Cube-Triple, Puzzle 3×3, Scene-Explore), each with noisy-TV variants |
| baselines | **6**: P2E, RND, ICM, APT, E3B, **E3B × P2E** (the obvious hybrid, run explicitly) |
| statistics | **5 seeds**, IQM with 95 % stratified bootstrap CIs. *"CIG's IQM confidence interval does not overlap with any baseline"*; P(CIG better on a random task) **≥ 0.79 against every baseline** |
| aggregate | CIG ≈ **0.80** normalised vs P2E ≈ **0.65** — **read off Figure 5, not a table**; treat as weaker than the per-task statements |
| **noisy-TV** | RND, ICM, APT, E3B and E3B×P2E **collapse to ≈ 0 successes**; CIG and P2E retain near-full coverage, and **CIG beats P2E on every noisy-TV task** |
| cost | kernel O(T²Md), Cholesky O(T³); at imagination horizon **T ≈ 15** both are *"negligible relative to the O(TMd) ensemble forward passes"* |

**Which spec it enters — and this is the load-bearing part, because it is not
where a reader would first put it.**

1. **`LEARNING_CORE.md` §5.4, arm `A3` (`wm-efe`) — the natural home.** `A3` is
   defined as *"A2's world model, byte-identical, same training loss, same
   train_ratio. The only change is the actor"*, scoring imagined rollouts by
   expected free energy whose epistemic term is the K=5 ensemble disagreement.
   CIG is a **drop-in replacement for that epistemic term over the same imagined
   rollouts at the same horizon** — the regime its O(T³) cost was measured in.
   It enters as `A3b`, or as a variant of `A3`, with `A3` itself as the control
   that isolates the prefix term.
2. **`CURIOSITY_BAKEOFF.md` §3.1 — where it does NOT cleanly fit, stated so the
   builder does not discover it late.** `disagree` runs on a ~150 K model-free
   policy with **no world model and no imagined rollout**. The only "prefix"
   available there is the real episode, whose length is thousands of steps, and
   O(T³) at T = 2,000 is ~8 × 10⁹ operations per bonus — **not** negligible. A
   `disagree`+CIG arm in LT.04 would need a windowed prefix (a design decision
   with its own hyperparameter, which forfeits the "no new hyperparameters"
   property that is half of why this nomination is cheap). **`LEARNING_CORE`
   `A3` is the honest home; `CURIOSITY_BAKEOFF` is a stretch.**

**Cost on our substrate.** Near zero *in the world-model setting*. No new
parameters. No new network. No new hyperparameter. The ensemble is one we have
already specified and costed — `LEARNING_CORE.md` §5.5 puts `A3` at ≈1.9 M
parameters plus 4 extra ensemble heads, and that cost is unchanged. The added
work per imagined rollout is a 15×15 kernel and its Cholesky, which is a NumPy
call. **This is the rare nomination that costs a few dozen lines and no compute
budget.** Against that: **CIG's own wall-clock is in an appendix I did not
open**, and B4's 5.0 sim-s/real-s floor is the gate that has already killed one
design, so *"negligible"* is their word on an A100-class machine and not a
measurement on 4 ARM cores.

**Why it might WIN (falsifiable).** `CURIOSITY_BAKEOFF.md` §3.1 already records
the exact failure mode CIG's prefix term is built to remove, as `disagree`'s
declared risk: *"the ensemble sharing so much data that disagreement collapses
everywhere, not only on noise."* If `A3b` beats `A3` at ≥1.5σ over 3 seeds while
both clear the learning gate, then the epistemic term this project inherited from
Plan2Explore was under-specified, and the fix cost nothing. If they tie, the
prefix redundancy does not exist in a survival world where the agent moves
continuously through a non-repeating state space — which is also a finding, and a
cheap one, and it would let `A3` keep the simpler estimator with a reason rather
than by default.

**Why it might LOSE (steelmanned). Five, and the third is the strongest.**
1. **Its own evidence says the ensemble was already enough on noise.** In the
   noisy-TV tasks *both* CIG and P2E survive — the paper's own explanation is
   that *"ensemble disagreement vanishes for irreducible stochastic transitions,
   so CIG and P2E are immune to the distractor by construction."* CIG's margin
   there is an improvement in an already-solved column. Our `LT.02`/`PG.4` trap is
   precisely a noise trap. **On the axis this project cares most about, CIG's
   contribution is second-order.**
2. **Twelve tasks, none of them a body.** MiniGrid is a gridworld; OGBench's
   AntMaze/Cube/Puzzle are manipulation and quadruped navigation.
   `CURIOSITY_BAKEOFF.md` §1.1's "Humanoid?" column is **No** for every method in
   the family and stays No for this one. `LESSONS.md`: what transfers is the
   protocol, not the ceiling.
3. **`SYSTEM.md`: no new organ without a scar — and neither `A3` nor `disagree`
   has ever run.** `LC.02` PASSed 2026-08-10; `LC.03` has not run; no `LT` spec
   has run. This is a pre-emptive nomination against an unscarred spec, exactly
   like week 2's N1, and the same objection applies with the same force. The
   counter, offered rather than asserted: the change is a few dozen lines inside
   an arm that has to be written anyway, so the marginal cost of carrying it as
   `A3b` from the start is far below the cost of retro-fitting it after `A3`
   loses ambiguously.
4. **The aggregate number I quoted is a figure, not a table.** ≈0.80 vs ≈0.65 is
   read off Figure 5. The per-task and CI statements are stronger evidence than
   that ratio and should be the ones weighed.
5. **Code release unconfirmed [c].** The paper is CC BY 4.0 with full pseudocode
   (their Algorithms 1 and 2), so the method is reimplementable from the text —
   but I searched and could not find a released repository, and nothing above
   should be read as "we can clone it".

---

### N2 — Optimistic World Models: last week's numberless abstract now has numbers, and they argue both ways

**Source:** *Optimistic World Models* —
[arXiv:2602.10044](https://arxiv.org/abs/2602.10044), **2026-02-10**, primary
category **cs.LG**. Mete, Sheikh, Lin, Kalathil, Kumar. **Full HTML read.**

**This resolves last week's highest-value outstanding fetch**, which was carried
as *"the abstract claims 'significant improvements' and the landing page reports
no benchmarks, no numbers, no parameter counts, no hardware, no code.
Unnominatable until someone opens the full text."* Opened.

**The method.** Classical reward-biased maximum-likelihood estimation (RBMLE)
brought into deep RL as an extra term on the **dynamics** loss that tilts the
learned model toward higher-reward transitions:

```
L_opt = −α(t)·Σ A_ℓ log p_φ(s_{ℓ+1}|s_ℓ,a_ℓ)  −  η·Σ H(p_φ(s_{ℓ+1}|s_ℓ,a_ℓ))
```

Fully gradient-based: *"requires neither uncertainty estimates nor constrained
optimization"* — no ensemble, no extra forward passes, **no new parameters**.

**The verified claim [V].**

| | value |
|---|---|
| Atari100K, mean HNS | **O-DreamerV3 152.68 %** vs DreamerV3 **97.45 %** |
| Atari100K, mean HNS | **O-STORM 80.68 %** vs STORM **75.90 %** |
| seeds | **10** for O-DreamerV3 on Atari100K and DMC; **5** for everything else |
| DMC **Proprio**, Acrobot Swingup Sparse | **8.4 → 34.6** |
| DMC Proprio, Cartpole Swingup Sparse | **664.2 → 747.1** |
| DMC Proprio, Cartpole Balance Sparse | **964.0 → 1000.0** |
| training wall-clock | O-DreamerV3 **138 min** vs DreamerV3 115 min (**+20 %**); O-STORM 178 vs STORM 170 (+4.7 %) — measured on an **RTX 4090**; experiments run on **one A100** |
| hyperparameters | **two new**: α (optimism), η (entropy). Used **α = 1×10⁻⁴, η = 3×10⁻⁶** |
| sensitivity | *"high values such as α = 0.1 or η = 0.03 can degrade performance drastically, while smaller values are beneficial"* |
| parameter counts | **not stated** |
| code | modifications of the official DreamerV3 / STORM repos; **no release of their own stated** [c] |

**Which spec it enters.** `LEARNING_CORE.md` §5.4 as a variant of **`A2`
(`dreamer-xs`)** — `A2` byte-identical, plus the optimistic dynamics term — with
`A2` itself as the control that isolates it. It is a **loss-term arm**, the same
shape as week 1's `A4b`/`A4c` anti-collapse nominations, and it bears on the
exploration question `CURIOSITY_BAKEOFF` owns without needing a curiosity module:
optimism *is* an exploration mechanism, priced at zero extra parameters.

**The relevant number for us is not the Atari one.** The DMC **Proprio** rows are
the only ones measured in a regime that resembles W0 — a low-dimensional **state
vector**, no pixels. `LEARNING_CORE.md` §5.4 puts `A4`'s prior as *"unknown at 2M
parameters on a ray retina; reconstruction is a much stronger learning signal
when the observation is 96-dimensional"* — the same regime argument applies here,
and here it points the *other* way: proprio-state DMC is close to us, and the
sparse-task gains there (Acrobot 8.4 → 34.6) are the ones worth weighing.

**Cost on our substrate.** No new parameters. One added loss term. The honest
cost is the **+20 % training wall-clock**, and that is not free here: `LC.02`
measured the arms at `train_ratio` **0.25**, sixteen times below what
`LEARNING_CORE.md` §5.1's derivation had assumed, and B4's 5.0 sim-s/real-s floor
is what killed the 36.7 M trunk. A 20 % tax on the learner, measured on a 4090
against a full-size DreamerV3, is a number that must be re-measured on `A2` at
our scale before it is believed — `LESSONS.md`, *"a budget is a claim about a
composition and must be measured as one."*

**Why it might WIN (falsifiable).** If `A2`+optimism beats `A2` at ≥1.5σ over 3
seeds on `life_gain` **without** violating B4, then W0's exploration problem has a
solution that costs zero parameters and no separate curiosity module — which
would be evidence bearing directly on `PURPOSE_AND_SCAFFOLDING.md` §2.1's open
question of whether needs and curiosity are substitutes. And unlike most
exploration results, this one comes with 10 seeds.

**Why it might LOSE (steelmanned). Five, and the first two are close to
disqualifying.**
1. **The optimistic term deliberately makes the world model WRONG, and in this
   project the world model IS the unified brain.** `LEARNING_CORE.md` §5.4 says
   of `A2`: *"the brief's candidate 2, and simultaneously
   `UNIFIED_BRAIN_BAKEOFF.md`'s binding objective: a model that predicts all
   senses jointly is the unified brain."* `SYSTEM.md`'s unison constraint says an
   adoption is VOID until the UB gates pass under it. Biasing the dynamics toward
   high-reward futures corrupts the very representation the UB ablation matrix
   measures — and nothing in this paper tests representation quality, only
   return. **This is the objection that could make it inadmissible rather than
   merely losing.**
2. **In a survival world, an optimistic model is a dead agent.** W0's reward is
   the drive channel `r_h`. A dynamics model tilted toward higher-reward
   transitions is one that imagines food where there is none, warmth where there
   is none, and water on the far side of a river. Atari and DMC have no death.
   `NEEDS_AND_DEATH.md`'s whole premise is that they should. This is a concrete,
   falsifiable prediction: **if the optimistic arm dies younger while its
   imagined returns rise, that is the mechanism showing itself**, and it is worth
   measuring even if the arm loses.
3. **α = 1 × 10⁻⁴, and 0.1 "degrades performance drastically".** A three-order-of-
   magnitude window between "beneficial" and "drastic degradation", tuned on
   Atari and DMC, on a coefficient whose correct value at our scale is unknown.
   `LEARNING_CORE.md` §6's simplicity budget counts hyperparameters, and this arm
   adds **two**. Compare N1, which adds none.
4. **Mean HNS on Atari100K is an outlier statistic, and the paper's own per-game
   table says so.** The largest gains are Private Eye (893.9 → 1,676.4) and Up N
   Down (24,954 → 91,717) — the two games most notorious for dominating Atari100K
   means. **I could not cleanly extract a median HNS and am not asserting one.**
   The headline 152.68 % vs 97.45 % should be read as "driven by a few games
   until someone reads the median", not as a broad 57-point improvement.
5. **Nothing here is multimodal, and nothing is a body.** Atari is pixels; DMC
   Proprio is a handful of joint angles. No touch, no audio, no smell, no needs.

---

## 2. CORROBORATION AND ONE RULED-OUT HYPOTHESIS — the voice front, after `VO.01` FAILed

Not a nomination. `VO.01` FAILed on 2026-08-11 at attempt 2, one day after this
file declared NO-ACTION on voice. A scout that reports "nothing new" on a front
that then fails owes it a look.

**The acoustics `VO.01` relies on: the physics half holds, the famous half does
not.** `VO.01`'s docstring justifies its muffling gate with the mass law —
*"`OCC_TRANSMISSION` is frequency-dependent (the mass law: walls pass bass), so
the received spectral centroid must FALL behind the block"* — and carries **no
citation**. Checked:

- **The physics is uncontested [V, secondary].** Frequency-dependent attenuation
  is real and steep: absorption, scattering and reverberation are all greater for
  high frequencies, and the slope steepens in dense media. `VO.01`'s measured
  `ref_clear_centroid` **485.7 Hz → `ref_occ_centroid` 251.4 Hz** and
  `occ_centroid_drop` **0.482** are the expected sign and a plausible magnitude.
- **The *biological* corollary — the Acoustic Adaptation Hypothesis, that animals
  in dense habitat evolve lower-frequency calls — is REFUTED by the current
  literature.** Freitas et al., *Biological Reviews* 2025
  ([10.1111/brv.13163](https://onlinelibrary.wiley.com/doi/10.1111/brv.13163),
  PMID 39530314): a meta-analysis across terrestrial vertebrates finding **no
  support for an effect of vegetation structure on acoustic signalling**. Mikula
  et al., *Ecology Letters* 2021
  ([10.1111/ele.13662](https://onlinelibrary.wiley.com/doi/10.1111/ele.13662)):
  a global passerine analysis finding **no support for AAH**, and pointing at
  sexual selection instead.

**Why this matters to us specifically.** `GOAL.md` names biology as the oracle,
and the oracle's *acoustics* is sound while the oracle's *adaptationism* is not.
`VO.01` happens to claim only the physics half, so it is clean. But it is exactly
the sort of place where a future spec might reach for "animals in the jungle
evolved low calls, so Jack's emitter should" — and that inference has now been
tested twice at scale and failed. **Recorded so it is not made.**

**And one hypothesis about the FAIL, computed and RULED OUT — so the builder does
not spend a run on it.** `VO.01` fails its clear-line gate: `recov_r2_amp`
**0.432** and `recov_r2_bright` **0.332** against `R2_MIN_PER_DIM = 0.50`, mean
**0.584** against `R2_MIN_MEAN = 0.60`. The obvious explanation is a
self-inflicted confound of exactly the kind `LESSONS.md` warns about — the
docstring randomises range *"so loudness alone can never identify the call"*
(`RANGE_M = (1.0, 4.5)`), and the probe receives only a log-band spectrogram with
**no range channel**, so it must invert `ear ∝ amp / r` while blind to `r`. That
is textbook "the step that removes a confound is itself a confound until
measured."

It is measurable in five lines, so I measured it rather than writing it up as an
insight. `amp` is log-spaced over `VOICE_AMP = (0.05, 1.0)` (26.0 dB) and range
spans 4.5× (13.1 dB of geometric spread):

```
Var(ln amp) = 0.748    Var(ln r) = 0.169
R² ceiling on the emitted amp from received LEVEL alone, range-blind = 0.816   [C]
```

**0.816 is well above the 0.50 gate, so range-blindness does not explain the
failure.** It caps amplitude recovery at ~0.82, and the run measured 0.432. The
confound is real and quantified, and it is **not** the cause. Whatever is
costing amplitude and brightness their gates is something else — and brightness,
at 0.347 on attempt 1 and **0.332** on attempt 2, did not move when the emitter's
loudness/timbre entanglement was fixed, which is itself information.

**This is a measurement, not a recommendation.** I am not proposing a threshold
change, a gate change, or a fixture change; `SYSTEM.md` law 4 and my own brief
both forbid it, and diagnosis is the builder's step of the loop, not mine. The
one thing the scout owed here was to stop a plausible wrong answer from being
chased, and the arithmetic above is that.

---

## 3. WATCHLIST

Every entry now records its **arXiv primary category** — week 2's discipline
finding, adopted here as a convention on this file.

**Carried forward from week 2, genuinely re-examined this sweep:**

| item | cat | status |
|---|---|---|
| **Simulus** ([arXiv:2502.11537](https://arxiv.org/abs/2502.11537)) | cs.LG | Unchanged and still the oldest open item. Still blocked on **a parameter count and a per-step wall-clock**, which B4's 5.0 sim-s/real-s floor needs and the paper does not report. Its prioritised-replay component is now nominated separately (week 2 N2) and is the cheapest piece to test in isolation. |
| **Var-JEPA** ([arXiv:2603.20111](https://arxiv.org/abs/2603.20111)) | cs.LG | **Partially superseded by TD-JEPA below**, which supplies the dynamics/control result Var-JEPA was blocked on — from a different paper and a different mechanism. Var-JEPA's own single-ELBO anti-collapse route remains tabular-only. |
| **Survival RL** ([arXiv:2605.31273](https://arxiv.org/abs/2605.31273)) | cs.LG | Unchanged disambiguation — "survival" = dwell time at goals, not homeostatic needs. Kept so a fourth sweep does not chase the title. |
| **SmallWorlds** ([arXiv:2511.23465](https://arxiv.org/abs/2511.23465)) | cs.LG | **Queued #4 — fetched.** *SmallWorld Benchmark*: world-model evaluation *"under isolated and precisely controlled dynamics without relying on handcrafted reward signals"*, **in the fully observable state space**, over six domains, comparing RSSM / Transformer / Diffusion / Neural-ODE and measuring **how predictions deteriorate over extended rollouts**. Fully-observable state-space is *our* regime, and rollout-horizon decay is what a world-model arm lives or dies by. **Still blocked on the same question as before: no compute cost, no hardware, no environment size, no code statement anywhere on the landing page.** Promote on: any statement that a domain runs on CPU. Note it is dated **2025-11-28 — outside the six-month window** and stays here on relevance, not recency. |

**New this sweep:**

| item | cat | what it is | what would PROMOTE it |
|---|---|---|---|
| **TD-JEPA** ([arXiv:2607.25337](https://arxiv.org/abs/2607.25337), 2026-07) | cs.LG | *Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control.* A **fourth** independent route into `LEARNING_CORE` §5.4's `A4` (`wm-latent`) neighbourhood, and the first with a control result: **15 M parameters, end-to-end from raw pixels on a single GPU in hours**, OGB-Cube **+14.2 points** over LeWM, Two-Room **100.0 % vs 97.4 %**, plans **48× faster** than world models on frozen foundation encoders, and *"detecting physically implausible rollouts more reliably than reconstruction-based baselines"* [c — search-level, abstract not fetched]. | **A state-vector or proprioceptive result, or a parameter-count ablation below ~2 M.** All of it is pixels at 15 M params, against `A4`'s ≈1.37 M on a 96-d ray retina — the same regime objection `LEARNING_CORE` §5.4 already records against the whole JEPA argument. One full-text fetch decides whether this is an `A4` arm or another pixels-only datapoint. |
| **Probing the Impact of Scale on Data-Efficient Generalist Transformer World Models for Atari** ([arXiv:2605.08578](https://arxiv.org/abs/2605.08578), 2026-05-09) | cs.LG | Single-author. Finds *"environments fundamentally fall into distinct scaling regimes"* at identical data budgets, and that **joint training across 26 environments stabilises scaling dynamics, ensuring monotonic gains**, where individual tasks show **degradation in larger models**. Median expert-random-normalised **0.770** for policies acting entirely inside the learned model. Bears on the 54K-beat-57M lesson and on `LEARNING_CORE` §6's simplicity budget: it is a direct claim that *bigger is sometimes worse, and multi-task training is what fixes it*. | **The parameter axis.** The landing page states **no parameter counts, no hardware, no seed count, no code** — which is the whole content of a scaling paper. Highest-value single fetch outstanding for front 1. |
| **On the Identifiability of Controlled World Models** ([arXiv:2607.22430](https://arxiv.org/abs/2607.22430), 2026-07) | cs.LG | Adjacent to week 1's N1 (certificate-gated identifiability, arXiv:2607.27017 → `UB.11` pre-gate) and possibly the theory under it. Not read. | Read it, and check whether it strengthens or undercuts week 1's N1 — which is a **live nomination**, so this is the higher-priority of the two identifiability items. |

---

## 4. NO-ACTION — fronts where nothing cleared the bar

Stated plainly. An empty front honestly reported beats a padded one.

**FRONT 3 · MEMORY — nothing, for the second consecutive sweep, and the reason is
the same constitutional one.** The 2026 agent-memory literature is, without
exception in what surfaced, **generative recall**: MemRL ([arXiv:2601.03192],
runtime RL over episodic memory), RecMem ([arXiv:2605.16045], LLM-extracted
episodic/semantic memory on recurrence), E-mem, RaMem, AgeMem, SSGM. Every one
routes retrieval through a language model that *writes* the memory. `GOAL.md`
forbids it and `MEMORY_RETRIEVAL_BAKEOFF.md` §5.1 makes it structurally
impossible on purpose (*"retrieval returns provenance-stamped POINTERS, never
prose"*).

The one non-generative candidate was **HAKARI-Bench**
([arXiv:2606.22778](https://arxiv.org/abs/2606.22778), **cs.IR**, 2026-06-22,
MIT-licensed, code + data + leaderboard released) — "Nano-sets" over 35
benchmarks and 551 tasks in 43 languages, comparing **five retrieval families**
(BM25, dense, sparse, late-interaction, rerankers) with **Spearman > 0.97**
against full MTEB/MMTEB/BEIR rankings across 55 models. Attractive as an
instrument. **Rejected on three counts, all checkable:** it reports **no hardware
and no latency**, so it cannot inform §1.9's CPU-cost table, which is the table
that decides feasibility here; it **does not measure abstention at all**, which
§1.8 identifies as *"the actual hard part"*; and its corpora are public IR
datasets while **Jack's corpus is his own life**, which §2 built a generative
grammar for precisely because no public set has the property we need.

Worth recording plainly: **on abstention, our own bakeoff document is ahead of
what the field is publishing.** §1.8 already carries split conformal coverage,
Clopper–Pearson certification of the false-answer rate with the sample sizes
derived, Learn-then-Test with fixed-sequence testing to avoid selection bias, and
E-AURC as the whole-curve metric. The 2024–2026 abstention literature is
generator-side almost in its entirety. There is nothing to import.

**FRONT 2 · MULTIMODAL FUSION — nothing that adds an arm.** Everything found sits
inside families `UNIFIED_BRAIN_BAKEOFF.md` §1.2–1.5 already cites with better
sources:
- *ViTacFormer* ([arXiv:2506.15953]) and *MSDP* / masked visual-tactile
  pre-training are the **same cross-modal-masked-prediction family** as M3L
  (2311.00924 [V]) and Fang et al. (2410.16424 [V]), which §1.2 already reads for
  the sharper finding — that **cross-modal reconstruction beats joint masking and
  beats contrastive**. ViTacFormer's ablation ("removing the tactile prediction
  module significantly degrades performance") restates week 1's N1 result
  (stiffness enters the latent only when touch is a prediction TARGET) with less
  rigour and no certificate.
- *Cross-Modal Visuo-Tactile Object Perception* ([arXiv:2604.02108]) — a Bayesian
  latent filter for object properties; perception, no policy, no unified agent.
- The **unified-tokenisation** line (WorldBagel 2607.03461, LatentUM 2604.02097,
  UniAR 2606.18249) is **VLA-scale autoregressive multimodal modelling** —
  vision, language, action, video. No touch, no proprioception-as-dominant-
  modality, no audio, and parameter counts far outside anything this box runs.
  Noted for ideas, inadmissible as arms.
- *A Closer Look at Multimodal Representation Collapse* (2505.22483) remains the
  best mechanism paper and is **already cited** in §1.5. Corroboration is not
  news.

**FRONT 5 · WORLDS & EMBODIMENT — nothing, and one candidate resolved with
numbers so it is not chased again.** **MuJoCoUni**
([arXiv:2605.24922](https://arxiv.org/abs/2605.24922), **cs.RO**, 2026-05-24,
`pip install mujoco-uni`, open source) looked like the front's best hope, because
throughput is this project's actual binding constraint — `LC.02` measured
`train_ratio` at 0.25, sixteen times below the derived budget. Full text read.
**It does not apply to us, and the numbers say why:**

| its claim | our situation |
|---|---|
| ~290 k steps/s (CMU Humanoid), ~1.8 M (Allegro hand) | measured on an **Intel i9-14900HX with 16 simulation threads** |
| throughput *"saturates around 256–512 environments"* | we run **one** environment — a life, serially |
| 15× reset, 22× Jacobian, 555× height-field speedups | all **batched** primitives; the paper states **no special optimisation for single-threaded scenarios** |

`SURVIVAL_WORLD.md` §2.2 already ruled out the GPU-parallel-env axis. This is the
same axis in C++ on a 16-thread x86 laptop, and **`LESSONS.md`'s "flag any
nomination whose numbers come from hardware unlike ours" disposes of it at a
glance.** Recorded with its numbers so a fourth sweep does not re-discover it.

The UED line (*Efficient UED through Hierarchical Policy Representation
Learning*, [arXiv:2602.09813]) is a **teacher generating environment parameters
for a student**, which is the opposite of `GOAL.md`'s position that the
environment plus intrinsic motivation IS the curriculum. Not an arm; possibly a
future contrast.

**BIOLOGY-AS-ORACLE — nothing new in-window.** The infant sleep-consolidation
line that surfaced traces back to Friedrich et al., *Nature Communications* 2015
(*Generalization of word meanings during infant sleep*) and Seehagen's 2020–2022
work — real, relevant to `NEEDS_AND_DEATH.md` §3.4's S1 stage, and **outside the
six-month window**. The one in-window item found was a 2026 review, not a result.
The useful biology this sweep produced was **negative** and is in §2: a famous
adaptationist hypothesis that two meta-analyses have now failed to support.

**SMALL-MODEL END — one candidate, rejected.** *Playing DOOM with 1.3M
Parameters* ([arXiv:2604.07385](https://arxiv.org/abs/2604.07385), cs.LG,
2026-04-08): a 1.3 M-parameter ModernBERT-based model at **31 ms/decision**
scoring **178 frags across 10 episodes** against **13 total** for Nemotron-120B,
Qwen3.5-27B and GPT-4o-mini combined; code and weights released. Rejected as a
nomination on three counts: it is **imitation learning from 31,000 human
demonstrations**, not RL and not learning-by-living; **no seed count is stated**;
and asking a 120 B LLM to act at 31 ms is a baseline that cannot win, so the
comparison measures latency, not architecture. It corroborates the 54K-beat-57M
lesson, and corroboration is not news.

**CONFERENCE PROCEEDINGS (queued #4 in week 1, #3 in week 2) — now BLOCKED, with
a cause.** Week 2 recorded this as *"not complete... it needs OpenReview
enumeration, which is a different tool than web search."* I tried that tool. The
OpenReview API now returns:

```
HTTP 403  {"name":"ChallengeRequiredError",
           "message":"Challenge verification required (2026-08-12-3326298)"}
```

— an interactive bot challenge on `api2.openreview.net/notes`, and
`/venues?id=venues` returns an empty list. **Programmatic main-track enumeration
is closed to an automated scout.** This is no longer "not done"; it is "the route
is shut", and the item should either be re-planned (per-paper search by author or
title, which does not enumerate) or dropped with that reason recorded. Marking it
as pending for a third week would be pretending.

---

## 5. A DISCIPLINE FINDING — this week the unverified claim was mine

Week 1's finding was *an abstract is a claim about a table, and nothing was
checking the table agreed*. Week 2's was *a title is a claim about a field, and
the primary category is the field's own statement of itself*. Both are about
other people's papers. This week's is not.

**I formed a confident, mechanically specific, literature-shaped diagnosis of
`VO.01`'s failure — and five lines of arithmetic refuted it.** The story was
good: the fixture randomises range to kill a loudness shortcut, the probe gets no
range channel, so amplitude recovery is underdetermined by construction; it is
`LESSONS.md`'s own *"the step that removes a confound is itself a confound until
measured"*, it names the right lesson, it cites the right physics, and it would
have read as insight. The ceiling it actually implies is **R² = 0.816** against a
**0.50** gate. The mechanism is real and it is not the cause.

Had I written it as a nomination, the builder would have had a plausible
paper-shaped explanation pointing at the fixture, and the real defect — whatever
is costing brightness its gate, unmoved across two attempts — would have been
one step further away.

The rule this project already has covers it exactly, twice over: *"a claim about
how a mechanism behaves is a two-line experiment. Run it"*, and *"this applies
hardest to reports from sources that have just been right about several harder
things."* **The field-watch-specific form is the one worth adding: a nomination
derived from a local FAIL is not a literature claim, it is a mechanism claim
about our own code, and it must carry the arithmetic that survives — not the
literature that motivated it.** A scout reading papers is protected by the
verify-before-nominating rule. A scout reading our own ledger has no such
protection, because there is no abstract to be sceptical of; the story is one it
wrote itself.

Recorded for the builder as a candidate `LESSONS.md` entry. **I am not writing
it; `LESSONS.md` is not mine to edit.** Note also that week 2's proposed
convention — record the arXiv primary category on every watchlist entry — **is
adopted in §3 of this file**, which is the cheapest possible form and needs no
organ.

---

## 6. What this report does NOT claim

- **No arm here has been run.** Every number except the one marked **[C]** in §2
  is someone else's measurement on someone else's hardware. Nothing in this file
  is evidence about Jack.
- **No nomination is a recommendation to adopt.** `SYSTEM.md` law 3 stands.
- **Nothing here changes a spec, a threshold, a decision, or a line of code** —
  including §2, which measures a quantity about `VO.01` in order to *withdraw* a
  hypothesis, not to move a gate.
- **Both nominations are pre-emptive, against specs that have never run.**
  `LC.03` has not run; no `LT` spec has run; `NE.05` has not run. `SYSTEM.md`'s
  "no new organ without a scar" counts against both, and it is stated in each
  nomination rather than buried here.
- **Verification is uneven and marked as such:** N1 full HTML with formulas,
  algorithms and result figures, **hardware/wall-clock not extracted**, **code
  release unconfirmed**; N2 full HTML with tables, **median HNS not extracted**,
  **no parameter counts published**, code release unconfirmed; TD-JEPA and the
  scale-probing paper are **search-level and abstract-level respectively**; the
  AAH meta-analyses are **secondary-source**; MuJoCoUni full HTML.
- **Front 2 was swept less deeply than fronts 1, 4 and 5** and §0 says so.
- **Week 2's two nominations are untouched and unresolved.** `SM.02` and `TA.02`
  have not run — the ledger was read to confirm it.

---

## 7. Queued for next sweep (**not before ~2026-08-19** — see the header)

1. **TD-JEPA full text** ([arXiv:2607.25337](https://arxiv.org/abs/2607.25337))
   — the highest-value fetch outstanding. A parameter-count ablation or a
   state-vector result promotes it to an `A4` arm; anything else and the JEPA
   family stays a pixels-only argument for a fourth consecutive week.
2. **CIG appendix A.6** — the hardware and wall-clock behind N1's *"negligible"*.
   B4 is a throughput gate and this project has been burned three times by
   published speedups that did not transfer.
3. **The scale-probing paper's parameter axis**
   ([arXiv:2605.08578](https://arxiv.org/abs/2605.08578)) — a scaling paper whose
   landing page states no parameter counts is either a real result about the
   small end or nothing, and one read decides.
4. **On the Identifiability of Controlled World Models**
   ([arXiv:2607.22430](https://arxiv.org/abs/2607.22430)) — it bears on week 1's
   N1, which is a **live** nomination; that ranks it above the other watchlist
   reads.
5. **Conference proceedings — decide, do not re-queue.** The OpenReview route is
   hard-blocked (§4). Either re-plan it as targeted per-venue searches, or drop
   the item with the reason recorded. A third "still pending" would be a lie by
   deferral.
6. **Watch `SM.02`, `TA.02` and `VO.01` attempt 3.** All three bear on live
   entries in this file. If `VO.01` passes on a fix unrelated to §2's ruled-out
   confound, say so here plainly — a scout's withdrawn hypothesis is worth as
   much on the record as a confirmed one.
