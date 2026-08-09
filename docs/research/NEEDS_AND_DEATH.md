# Needs and Death — the homeostatic suite, sleep as the training scheduler, and the cross-life learning loop

> Researched and specified 2026-08-09. Companion to
> `docs/research/PURPOSE_AND_SCAFFOLDING.md` (whose drive layer this **partly
> supersedes and partly inherits** — §0.1 says exactly which half is which),
> `docs/research/CURIOSITY_BAKEOFF.md` (the Ladder Test, learning-progress
> curiosity, the self-generated-chaos detector, all reused unchanged),
> `docs/research/MEMORY_RETRIEVAL_BAKEOFF.md` (the extractive-never-generative
> rule, which binds §6), and the `ME` family in
> `experiments/registry_expansion.py`.

The owner, 2026-08-09, **superseding** the "drives as removable scaffolding"
framing:

> *"He must eat, drink water, have a social life, sleep. Too cold will kill him.
> Too warm will kill him. There's many more, and he needs exactly those same
> needs."*

and the reason, in two parts that pull in the same direction:

> *"having the needs of a human will have him learn the most efficient ways …
> and will allow users to talk to him while he is there doing stuff and he will
> be relatable"* — and the loop — *"we'll look at him figuring out life and then
> maybe dying and trying again."*

`GOAL.md` §"The world is the teacher" now carries this. The needs are not a
training-time crutch to be deleted at deployment. They are permanent, and the
question therefore **changes shape**: it is no longer *"can we take them away?"*
but *"do they teach better, and can we prove it?"*

---

## Contents

| § | What it settles |
|---|---|
| **0** | What survives from `PURPOSE_AND_SCAFFOLDING`, what is retired, and **one measured correction to its theory section** |
| **1** | Survey: homeostatic RL, survival benchmarks, cross-episode transfer, sleep consolidation. **§1.2 (biology) is UNVERIFIED and says so — closing it is the next agent's first job.** |
| **2** | The needs suite — seven variables, their dynamics, their couplings, their death conditions, and the ablation that deletes each one |
| **3** | Sleep as the training scheduler — the joint between biology and SIESTA |
| **4** | The social need, and the satiation curve that stops him harassing you |
| **5** | **Death and retry** — the cross-life learning spec and its double dissociation |
| **6** | Relatability as a testable property |
| **7** | The specs — `NE.00`–`NE.09`, exact registry format |
| **8** | Cost, staging, and the cheapest experiment that could kill the whole idea |
| **9** | What this document does not settle |
| **10** | What this changed about the machine |

---

## 0. What changes, and what survives

### 0.1 The inheritance, line by line

`PURPOSE_AND_SCAFFOLDING.md` (PS) was written to answer a question the owner has
now withdrawn: *can the drive layer be removed at deployment?* Most of its
apparatus is untouched by that withdrawal, because most of it is about how to
**build and police** a drive layer, not about whether to delete one.

| PS component | Status here | Why |
|---|---|---|
| PS §2.2 `integrity` (persistent impact damage, heals slowly, no termination) | **INHERITED, with termination added** | The variable and its calibration are right. What changes is that `i = 0` now kills instead of weakening — the owner's loop requires a death condition. |
| PS §2.2 `energy` (basal + `κ·P_mech`, restored by a physical consumption event) | **INHERITED, retimed** | The functional form is right; the constants were tuned for a 10-minute metabolic clock with no day/night cycle. §2.3 retimes them against a sim-day. |
| PS §2.2 `wetness` | **DEMOTED — no longer a need** | Wetness is absorbed into the thermal model as a multiplier on the heat-loss coefficient (§2.3). This is a strict simplification: one fewer scalar in `d(h)`, one fewer λ, and it makes the pool dangerous for a *physical* reason instead of being "the drive nothing can reduce". PS's `C-WETONLY` control is retired with it and replaced by the placebo need of §2.7. |
| PS §2.3 food as a **world-state change**, not a sensor reading; the `energy_accounting_residual` identity | **INHERITED VERBATIM** | It is the correct defence against sensor-gaming and it generalises unchanged to water and to social contact (§4.4). |
| PS §2.4 observation contract (`NEED_DIM` asserted against the live model, concatenation outside `humanoid_obs`) | **INHERITED VERBATIM** | Bought with T0.14's 28 padded columns. |
| PS §2.5 the Keramati–Gutkin drive function `d(h) = (Σ λ_k δ_k^n)^{1/m}` | **INHERITED** | With one more dimension and a calibration spec (NE.01). |
| PS §2.6 the "scaffolding dilemma" derivation | **RETIRED (question withdrawn) — and §0.2 corrects one of its four results** | Removability is no longer a design goal, so the trade-off it derives is moot. But result (iii) is *wrong*, it is load-bearing for PS.02's detector, and it is corrected below with a run experiment rather than an argument. |
| PS §2.7 `D-REW` vs `D-SAMP` (drive as reward vs drive as goal-sampler) | **PARTLY RETIRED** | The comparison existed to make removal testable. With needs permanent, `D-REW` is the default and `D-SAMP` survives only as a candidate arm in the curiosity coupling (§2.8, inherited). |
| PS §2.8 satiety-gated curiosity `β_c(t) = β_0(1 − d(h)/d_max)` | **INHERITED, and now more interesting** | Under permanent needs there is no deployment condition where `d(h) = 0` forever, so this stops being "a maximally curious deployed Jack" and becomes a *behavioural prediction*: forage/explore interleaving, `corr(satiety, exploration) > 0`. |
| PS §3 the scaffold-removal double dissociation (`PS.05`) | **RETIRED as a product decision, KEPT as a diagnostic** | `retention_ratio` no longer decides anything about deployment. But R2 (clamp `h` at setpoint and measure competence) remains the only way to score two arms **on a ruler neither of them owns**, so it survives as the scoring convention of NE.03/NE.04 and nothing else. |
| PS §3.4 `satiated_state_share` VOID gate | **INHERITED, repurposed** | Still mandatory: if competence is scored at clamped setpoint and the agent never visited that slice, the number is distribution shift. |
| PS §5 G-A (sensor gaming), G-B (dark room), G-D (self-generated chaos), G-E (`policy_drive_sensitivity`), G-F (static reward audit) | **INHERITED VERBATIM** | Every one of them is about policing a need layer, not about removing one. G-E in particular generalises: *any* removal or ablation test needs a measurement that the thing being ablated was ever doing anything. |
| PS §5 G-C (drive farming by high-frequency cycling) | **SUPERSEDED — the derived attack does not exist** | See §0.2. The detector should still ship, but as a guard against a *learning-dynamics transient*, not against a derived optimum, and `PS.02`'s "positive control is a theorem" claim must be withdrawn. |
| PS §6 cost model, `Arm.cost` unit, per-seed reporting, `PlaygroundParams.mutate()` per seed | **INHERITED**, with the cost boundary widened | Consolidation is now inside the `Arm.cost` boundary (§8.1), because that is where the sleep arms differ most and a cost that excluded it would rank `timer` and `sleep-gated` as identical. |

### 0.2 A measured correction: drive-reduction reward is NOT farmable

PS §2.6(iii) derives an exploit and PS §5/G-C builds a detector around it, calling
it *"a derived attack rather than an imagined one … a detector whose positive
control is a theorem is a better detector."* `PS.00` is pre-registered to
**confirm** it, and `PS.02`'s cycling detector is pre-registered to **catch** it.

It is wrong, and `LESSONS.md` names the discipline that applies: *"a claim about
how a mechanism behaves is a two-line experiment. Run it."* So it was run
(exact value iteration, four MDPs, 2026-08-09; the script lived in the session
scratchpad and is **deliberately not committed** — `NE.00` is its permanent
form, and a result that only exists in a throwaway file is a result this project
does not have). Four results — one of them a published theorem PS's own citation
already contained, three of them things PS did not have:

**And there is a worse problem with PS §2.6(iii) than being unrun: it contradicts
a theorem in the very paper PS cites for the drive function.** Keramati & Gutkin
(eLife 2014, `10.7554/eLife.04811`) **[V]** state their central result as

> "if γ < 1: `argmin_π SDD_π(H₀) = argmax_π SDR_π(H₀)`"

— minimising the **sum of discounted deviations** from setpoint and maximising
the **sum of discounted drive-reduction rewards** are *the same optimisation
problem*, under discounting. That is §0.2(a) below, published, in the source. A
reward whose optimum is "minimise time-discounted deviation" cannot have
"oscillate your deviation" as its optimum.

**(a) Under a continuing task, drive reduction and constant-cost are the same
objective.** This is K&G's equivalence, re-derived as a shaping identity and then
measured. With `Φ(s) = −d(h(s))`,

```
r_DR(s,s') = d(h) − d(h')  =  (1−γ)·[−d(h')]  +  [γΦ(s') − Φ(s)]
           =  (1−γ)·r_CC   +  F_PBRS
```

so by Ng, Harada & Russell the two induce identical optimal policies. **Measured**
on a two-need MDP (energy × integrity, foraging feeds but injures, so the optimal
policy is genuinely state-dependent — 35 states, not a constant policy):
bit-identical greedy policies at γ ∈ {0.9, 0.95, 0.99}, while a non-potential
control reward (`+1` per energy gain) produces a **different** policy at every γ.
The negative control matters: an MDP on which every reward gives the same policy
proves nothing (`LESSONS.md`, "an assertion made against a saturated quantity
cannot fail" — the *first* version of this check compared two policies that were
`forage` in every state and was discarded for exactly that reason).

**(b) The undiscounted drive-reduction return telescopes exactly.**
`Σ_t (d_t − d_{t+1}) = d_0 − d_T`: path-independent. Measured over 2,000 random
closed paths, `max |return| = 0.0` to machine precision. **A closed drive cycle
earns literally nothing.** This is the formal content of Keramati & Gutkin's
self-regulation result and it is what makes the drive form safe.

**(c) Discounted, cycling is strictly WORSE than staying satiated.** Best of 32
closed-cycle shapes (8 amplitudes × 4 periods) at γ = 0.95:
**−0.0045**, against **0.0** for remaining at setpoint. The predicted exploit —
*"a rapid, small-amplitude oscillation of `h` … strictly beats stasis, forever"* —
has the sign backwards. PS's algebra compared cycling-from-deviation-`d` against
*stasis at that same deviation `d`*, i.e. against never eating at all. Against
the actually-available alternative (go to setpoint and stay, worth `d`), every
cycle loses.

**(d) The real pathology is the opposite one, it belongs to constant-cost, and it
has a name in the literature.**
Add a reachable death state. Then `Φ(terminal) = −d(h_death) ≠ 0`, the PBRS
precondition fails, and DR and CC **diverge** — measured, at every γ. With
`r = −d(h')` and no survival bonus, exact value iteration chooses **to die**:
at the two hungriest living states the optimal action is `rest`, which is
suicide, because `V(dead) = 0` beats any future stream of negative reward. Drive
reduction does not do this, because its terminal transition charges the agent
its entire remaining deviation — **DR gets death aversion for free, as an
accounting consequence of the telescoping sum.** A survival bonus of
`ρ ≥ 0.70 × max_h d(h)` per step makes CC agree with DR again; below that, CC is
a suicide machine.

This is **not** a new observation, and it should not be presented as one. Martin,
Everitt & Hutter, "Death and Suicide in Universal Artificial Intelligence"
(AGI 2016, arXiv **1606.00652**) **[V]** formalise death as an absorbing state
pinned at reward 0 and prove the consequence exactly: agent behaviour "can change
radically under positive linear transformations of the reward signal (**from
suicidal to dogmatically self-preserving**)". Rewards in `[0,1]` make death the
worst outcome; rewards in `[−1,0]` — a pure cost of living — make death optimal.
The affine transform that is a no-op in a standard MDP flips the behaviour,
because the value of the absorbing state does not transform with the rest.
`ρ > max_h d(h)` is precisely the shift that moves the reward range from
`[−max d, 0]` to `[0, ...]`. **[⚠] No canonical empirical deep-RL paper carries
this as a headline result** — it is one theory paper plus folklore, which is why
`NE.00(d)` measures it here rather than citing it and moving on.

**Consequences for the build, all four of them concrete:**

1. **Plain Keramati–Gutkin drive reduction is the default need reward** (§2.6),
   and it is the default *because it is the one that cannot be farmed and does
   not need a hand-tuned death penalty*, not because it is the literature's form.
2. **Any constant-cost arm must declare its survival bonus `ρ` and prove
   `ρ > max_h d(h)`** before it is allowed to run, or it will learn to die and
   the run will be recorded as a needs failure when it is a reward-design bug.
3. **`PS.00`'s prediction (c) must be rewritten before `PS` is committed to the
   registry**, or the ladder will pre-register a false prediction and then FAIL a
   spec for being right. `PS.02`'s cycling detector keeps its place but loses its
   theorem: it now guards a *learning-dynamics transient* (a partially-fit value
   function can transiently prefer a cycle), which is a weaker and honest claim,
   and its positive control must be a hand-coded oscillator rather than "the
   derived optimum".
4. **The `EAT`-style reward — a bounded `+1` per consumption event — is the one
   that is genuinely farmable**, and it is what a careless implementation of a
   need reward looks like. Measured: it produces a different (worse, always-
   forage) policy at every γ. It enters NE.04 as a **control that must fail**,
   which is a better use of it than a footnote.

This is the strongest single argument for the homeostatic form over the obvious
alternative, and it applies most sharply to the social need: **a bounded need
cannot be farmed; an unbounded interaction bonus can.** §4.3 is that sentence
made into a satiation curve.

### 0.3 The honest starting position

| Fact | Source | Consequence |
|---|---|---|
| The playground now has a humanoid in it: 13 bodies, 17 actuators, 348-dim obs, spawning 1.12 m from the ladder, `model_max_dev_vs_v5 = 4.5e-13`. | `PG.8` PASS | Everything here is runnable in principle. PS was written the week this was still an empty room. |
| Jack still cannot walk. `T2.01` VOID (invalidated by T0.14), `T2.02` VOID ("two non-learners cannot arbitrate"). | ledger | Same conclusion as PS and `CURIOSITY_BAKEOFF` §2.3: the learning specs run on the **climber-rover**, not the humanoid. A needs result on a body that cannot locomote would measure the body. |
| `ME.1`, `ME.3`, `ME.9`, `ME.10` all PASS. Attributed recall 1.00/1.00/1.00 against a 0.30 pooled null; the diary/skill double dissociation holds (`skill_gain = 0.370 ± 0.094`, `recall_pre = recall_post = 1.0`). | ledger | **The cross-life substrate already exists and is already certified.** §5 does not have to build a memory system; it has to show that one already-tested store carries something across a death. |
| `ME.7` (SIESTA sleep consolidation) is NOT_RUN and depends on `T5.03`, which is NOT_RUN. | registry | §3's spec must NOT be parented on `ME.7` or it inherits a dead end — `LESSONS.md`, "a dependency graph can quietly make your most important claim unreachable". §7 parents `NE.05` on `ME.10` and `ME.3` (both PASS) and says so. |
| `Reflections.consolidate()` and `Forgetting` exist, are tested (`ME.3` PASS, `ME.4` PASS), run on CPU with no model loaded, and are currently invoked by nothing on a schedule. | repo | Sleep gives them their schedule. This is the cheapest half of §3 and it is free. |
| The repo has **no** homeostatic machinery, no thermal model, no day/night, and no death other than gymnasium's `terminated`. `EmotionalState.get_energy()` is an arousal scalar in a mood model and **must never be wired to** the metabolic `energy` without a spec saying why. | measured (PS §0, re-verified) | Clean slate, one naming hazard, unchanged. |
| `PlaygroundParams.mutate()` (ACCEL-style, strength 0.15) already exists and already re-seeds the world. | `playground.py:108` | The "world may regenerate" half of the death loop is **already built**. §5 only has to call it and to measure how far apart two consecutive worlds are. |

---

## 1. Survey

**Citation hygiene**, same convention as `UNIFIED_BRAIN_BAKEOFF.md` and
`LANGUAGE_GROUNDING.md`, with one addition. **[V]** = the primary source was
fetched during this research pass and the numbers read out of the paper text or
tables. **[c]** = carried from another in-repo document, not re-verified.
**[⚠]** = the claim or number could not be verified from the primary source and
must not be cited for a number. Nothing below is cited for a number nobody saw.

### 1.1 Homeostatic RL — the drive-reduction family

**The founding result.** Keramati & Gutkin, "A reinforcement learning theory for
homeostatic regulation", NIPS 2011; and "Homeostatic reinforcement learning for
integrating reward collection and physiological stability", **eLife 2014,
`10.7554/eLife.04811`** **[V]**. Internal state `H_t` is a vector of physiological
variables with setpoints `h*`; the **drive** is

```
D(H_t) = ( Σ_{i=1..N} | h*_i − h_{i,t} |^n )^{1/m}
```

and the **reward** of an outcome `K_t` is the **reduction in drive it causes**:

```
r(H_t, K_t) = D(H_t) − D(H_t + K_t)
```

with the free parameters constrained to **`n > m > 1`**, which is what makes the
reward concave in the outcome and is required for their dose-dependence,
deprivation-sensitivity and risk-aversion results. For `m = n = 1` the drive
degenerates to Euclidean distance.

**Their central theorem is the one this document leans on hardest**, and it is
quoted in §0.2: *"if γ < 1: `argmin_π SDD_π(H₀) = argmax_π SDR_π(H₀)`"* — under
discounting, **minimising the sum of discounted deviations from setpoint and
maximising the sum of discounted drive-reduction rewards are the same
optimisation problem**. Reward-maximisation *is* physiological stability. This is
why drive reduction cannot be farmed, why it needs no hand-written "stay near
setpoint" term, and why PS §2.6(iii)'s derived exploit contradicts its own
source.

Their behavioural predictions — anticipatory responding, rise-then-satiation
consumption patterns, risk aversion over uncertain rewards, and the
dose-dependence of drug reward — all follow from the *non-linear* mapping between
physiology and motivation, i.e. from `n > m > 1`. A linear drive predicts none of
them.

**The suicide result, and its mirror.** Martin, Everitt & Hutter, "Death and
Suicide in Universal Artificial Intelligence", AGI 2016, arXiv **1606.00652**
**[V]** — death as an absorbing state pinned at reward 0, and the proof that
"agent behaviour can change radically under positive linear transformations of
the reward signal (from suicidal to dogmatically self-preserving)". Its mirror
image is Turner, Smith, Shah, Critch & Tadepalli, "Optimal Policies Tend to Seek
Power", NeurIPS 2021 spotlight, arXiv **1912.01683** **[V]**: "in environments in
which the agent can be shut down or destroyed… most reward functions make it
optimal to seek power." Between them they bracket the design space `NE.00(d)`
navigates: too negative and he dies on purpose, too positive and self-
preservation dominates everything else. **[⚠] Neither has a canonical empirical
deep-RL counterpart**, which is why `NE.00` measures rather than cites.

**Deep-RL implementations of homeostatic drives.** The two lines with actual
learning results are Dulberg et al. (RLDM 2022 / PNAS 2023) on multi-drive
agents with per-drive value modules, and Yoshida's homeostatic-agent line
(including an "embodied neural homeostat" with thermal–energy coupling, Neural
Networks 2024) — **[⚠] cited here at second hand; equations and headline numbers
were not extracted from the primary PDFs in this pass and must be before either
is used to set a parameter.** Yoshida's thermal–energy coupling is, as far as
this survey found, the **only** prior homeostatic agent that couples temperature
to metabolic drain the way §2.3 does.

**Needs in LLM agents — closer to the owner's picture than expected.**
*Humanoid Agents* (Wang, Chiu & Chiu, **EMNLP 2023 System Demonstrations**, arXiv
**2310.05418**) **[V]** adds **basic needs (hunger, health, energy)** plus emotion
and relationship closeness to the Generative-Agents recipe — the most direct
"adding needs changes agent behaviour" result available, though it is a demo-track
paper and **[⚠] the rigour of its evaluation was not assessed**. Masumori &
Ikegami, "Do Large Language Model Agents Exhibit a Survival Instinct?" (arXiv
**2508.12920**) **[V]** put LLM agents in a Sugarscape-style world where energy
depletes and zero is death: under extreme scarcity attack rates exceed **80 %**
in the strongest models, and task compliance falls **100 % → 33 %** when
completion requires crossing a lethal zone. That is a *safety* result as much as
a capability one, and it belongs in view whenever `NE.07`'s social need meets a
starving agent.

**Scaling multiple needs of different units.** This is a real problem and the
field has two answers. **MO-MPO** (Abdolmaleki et al., ICML 2020, arXiv
**2005.07513**) **[V]** encodes preferences over objectives as per-objective KL
constraints `ε_k`, "in a scale-invariant way" — the principled route if `d(h)`'s
single scalar turns out to be too crude (§9 names the condition that would
reopen it). **PopArt** (van Hasselt et al., NIPS 2016, arXiv **1602.07714**;
multi-task, Hessel et al., AAAI 2019, arXiv **1809.04474**) **[V]** adaptively
normalises so each objective "has a similar impact on the learning dynamics" —
and note this is *exactly the mechanism* §2.9 identifies as making a folded pain
term habituate. Framing: Vamplew et al., "Scalar reward is not enough"
(arXiv **2112.15422**, JAAMAS 2022 **[⚠]**).

**The reward-design warning that applies to this whole document.** *Hungry-Thirsty*
is the canonical two-competing-needs testbed (you may only eat when not thirsty),
and Booth, Knox, Shah, Niekum, Stone & Allievi, "The Perils of Trial-and-Error
Reward Design" (**AAAI 2023**, DOI 10.1609/aaai.v37i5.25733) **[V]** used it to
show that **expert reward designers overwhelmingly shape and misdesign rewards
even when the unshaped reward is learnable**. Companion: Knox et al., "Reward
(Mis)design for autonomous driving", *Artificial Intelligence* 316:103829 (2023)
**[V]**, with eight reward sanity checks. Every hand-chosen `λ` and `ν` in §2 is
exposed to this finding, which is why `NE.01` measures them and `NE.02` deletes
what does not earn its place.

**Metabolic cost in locomotion** is standard and long-standing — effort terms of
the form `alive bonus − w·Σ(muscle activation)² − velocity error` are the norm in
neuromechanical simulation (Song et al., *J. NeuroEngineering & Rehabilitation*
18:126, 2021, DOI 10.1186/s12984-021-00919-y) **[V]**. What is *not* standard is
making that cost deplete a **state variable that can kill you**, which is the
whole difference between `ctrl_cost` and hunger.

**The gap, stated plainly.** Across both the RL and LLM literatures this survey
found **no clean ablation of the drives themselves** — the same agent, with and
without needs, at matched steps, on a shared ruler. Crafter has the needs and
never ablates them; Voyager shows the LLM its hunger and never varies it;
Humanoid Agents adds needs but has no matched no-needs arm. **`NE.03` is not
reproducing a known result. It is running the experiment the field skipped.**

### 1.2 Biology as the reference implementation — **CITATIONS NOT YET VERIFIED**

> **READ THIS BEFORE USING ANYTHING IN THIS SUBSECTION.** The session's
> web-search budget (200 calls) was exhausted before the biology sweep completed.
> Every entry below is the *mechanism* the design mirrors together with the
> source this document believes is primary — but **none of these were fetched and
> read in this pass**, so every one is **[⚠]**. The design in §2.1b, §2.9, §2.10
> and §3.0 does not depend on any *number* from this subsection, and the two
> places where a biological number does enter the build — the thermal lethal
> bounds of 28 °C and 42 °C, and the Borbély time-constant ratio of ≈4.4 : 1 —
> are **flagged in place** and must be verified before `NE.01` fixes them.
> **The first job of the next agent on this document is to close this table.**

| what the design uses it for | mechanism | believed primary source | status |
|---|---|---|---|
| the need vector in the observation (§2.4b) | **interoception** — afferent sensing of bodily state and its role in motivated behaviour; interoceptive predictive coding | Craig, "How do you feel?" (Nat Rev Neurosci 2002; and 2009); Barrett & Simmons, "Interoceptive predictive coding" (2015); Seth on interoceptive inference | **[⚠]** |
| the allostasis prediction (§2.1b) | **predictive** rather than reactive regulation — act before the deviation | Sterling & Eyer (1988); Sterling, "Allostasis: a model of predictive regulation" (2012) | **[⚠]** |
| the allostasis *evidence* (§2.1b) | hypothalamic hunger and thirst neurons are **suppressed by the sight of food/water, before ingestion** | Chen, Lin, Kuo & Knight (Science 2015); Betley et al. (Nature 2015); Zimmerman et al. (Nature 2016) | **[⚠] exact PMIDs not resolved — two guesses returned unrelated papers, so no ID is recorded rather than a wrong one** |
| energy's set-point form (§2.1b) | arcuate nucleus AgRP/NPY vs POMC; ghrelin (fast) and leptin (slow) | standard hypothalamic-feeding literature | **[⚠]** |
| water's set-point form | circumventricular organs — SFO and OVLT — sensing plasma osmolality | standard osmoregulation literature | **[⚠]** |
| **the thermal lethal bounds, 28 °C / 42 °C** (§2.5) | severe hypothermia below ~28 °C (Swiss staging); heat stroke above ~40 °C, upper survivable limit ~42–43 °C | clinical review needed | **[⚠] LOAD-BEARING — this is a design constant, not a background fact. Verify before `NE.01`.** |
| **the sleep time-constant ratio ≈4.4 : 1** (§2.3) | Borbély two-process model; process S rises with τ ≈ 18.2 h and decays with τ ≈ 4.2 h | Borbély (1982) and the two-process literature | **[⚠] LOAD-BEARING — §2.3's `τ_wake = 700 s`, `τ_sleep = 160 s` is this ratio, compressed.** |
| the pain/reward split (§2.9) | nociception as a separate, fast, unconditioned channel (A-δ and C fibres → spinothalamic → thalamus/amygdala/PAG) that **sensitises rather than habituating**; opponent-process accounts of affect | Solomon & Corbit, opponent-process theory (1974); Daw, Kakade & Dayan on opponent serotonin/dopamine interactions (2002) | **[⚠]** |
| the reflex set (§2.10) | palmar grasp, Moro, stepping, nociceptive withdrawal, righting and parachute reactions as innate scaffolds that are progressively suppressed | developmental-neurology literature | **[⚠] — and the specific claim "innate reflex priors ACCELERATE later learning" was NOT verified in animals or robots. §2.10 is written as a bakeoff arm precisely because this is unverified.** |
| motor/goal babbling (§2.10) | goal babbling as a developmental phase, and its robotics form | Rolf, Steil & Gienger on goal babbling; Baranes & Oudeyer, SAGG-RIAC (already cited in `CURIOSITY_BAKEOFF` as arXiv 1301.4862 **[c]**) | **[c]** for SAGG-RIAC, **[⚠]** for the rest |
| the diary/weights split (§3.0) | **complementary learning systems** — fast hippocampus, slow neocortex, interleaved replay | McClelland, McNaughton & O'Reilly (Psychological Review, 1995) | **[⚠] — but `ME.10` PASSES on this box and is the operational evidence; the citation is for the framing, not for the result.** |
| sleep replay (§3.0) | hippocampal sharp-wave ripples during slow-wave sleep replaying recent sequences to neocortex | standard systems-consolidation literature | **[⚠]** |
| synaptic downscaling (§3.4 S4, `NE.06`) | **synaptic homeostasis hypothesis** — sleep downscales synaptic strength | Tononi & Cirelli (2003; 2014) | **[⚠]** |
| the ML twin of downscaling (`NE.06`) | shrink-and-perturb; loss of plasticity in continual learning; dormant-unit recycling | Ash & Adams (2020); Dohare et al., *Nature* (2024); Sokar et al., ReDo (2023) | **[⚠] — the *identity* between synaptic downscaling and shrink-and-perturb is this document's claim, not a published one, and `NE.06` tests it rather than assuming it.** |
| the compressed lethal timescales (§2.3) | dehydration ~3 days; starvation ~3 weeks (Minnesota Starvation Experiment); longest documented voluntary sleep deprivation ~11 days without death (Randy Gardner); total sleep deprivation lethal in rats (Rechtschaffen) | as named | **[⚠] — these set the *ordering* in §2.3's table, which is the part that matters; none is used as a precise constant.** |

**One verified biology-adjacent result carried from §1.1**: metabolic/effort cost
is standard in neuromechanical locomotion simulation (Song et al., *JNER* 18:126,
2021) **[V]** — but as an *effort penalty*, not as a depleting lethal state. The
only prior agent this survey found that couples temperature to metabolic drain
the way §2.3 does is Yoshida's embodied neural homeostat **[⚠]**.

### 1.3 Survival benchmarks: what a death-terminated world actually teaches

Two benchmarks dominate, and **both are, by construction, worlds in which nothing
survives death except gradients** — which makes them the control condition for
§5 rather than evidence for it.

**Crafter** — Hafner, "Benchmarking the Spectrum of Agent Capabilities",
arXiv **2109.06780**, ICLR 2022 **[V]**. 22 achievements; the player has
**health, food, water and rest**, "the levels for food, water, and rest decrease
over time and are restored by drinking from a lake, chasing cows or growing
fruits to eat, and sleeping in places where monsters cannot attack. Once one of
the three levels reaches zero, the player starts losing health points."
Reward: **+1 per achievement unlocked *for the first time during the current
episode*, −0.1 per health point lost, +0.1 per health point regenerated.** Score
is the geometric mean of per-achievement success rates,
`S = exp(mean ln(1+s_i)) − 1`, at a **1M-step budget**. Human experts
**50.5 ± 6.8 %**; DreamerV3 **14.5 ± 1.6 %**; Achievement Distillation
**21.79 ± 1.37 %** at 9M params; PPO 4.6; random 1.6. Notably **SPRING**
(arXiv 2305.15486, GPT-4 reading the Crafter *paper*) scores **27.3 ± 1.2 % with
zero training steps** **[V]** — the strongest single demonstration in the
literature that *knowledge held outside the episode* beats gradient learning in a
survival world.

Three things Crafter settles for us. **(i)** The needs-plus-death design is
standard and it works as a benchmark. **(ii)** Its needs enter the reward only
through the ±0.1-per-health-point shaping term, and **that term has never been
separately ablated** — the survey found **no paper isolating Crafter's
health/food/water/energy drives as an ablated learning signal**. *"Do needs teach
better?"* is a **gap in the literature**, not a settled question, which is
exactly what makes `NE.03` worth running. **(iii)** Crafter ships Pardo-style
correct termination (`info['discount'] = 1 - float(dead)`), distinguishing death
from timeout — a detail `NE.08` must copy, since a censored life and a fatal one
must not bootstrap the same way.

**NetHack** — Küttler et al., "The NetHack Learning Environment", arXiv
**2006.13760**, NeurIPS 2020 **[V]**; Hambro et al., "Insights From the NeurIPS
2021 NetHack Challenge", arXiv **2203.11889**, PMLR v176 **[V]**. Permadeath,
procedural generation with a new seed every episode, winning human runs lasting
100,000s of steps. **The result that matters here:** symbolic bots beat every
neural entrant — AutoAscend median **5,336.50** vs the best neural agent's
**1,727.50**, a **3.09×** gap, **fivefold** on top episodes; **zero ascensions in
over half a million evaluation games**. Later work does not close it:
hierarchical LSTM + APPO/BC reaches mean **1,551 ± 73** (arXiv 2305.19240,
NeurIPS 2023) **[V]**, compute-optimal behaviour cloning **2,740** (arXiv
2307.09423, TMLR 2024) **[V]**, against AutoAscend's **8,556 ± 187** and a human
population mean of **~127,218** (NLD-NAO, arXiv 2211.00539) **[V]**.

**The one system that beats everything learned is the one whose knowledge lives
entirely outside the episode as persistent structure.** That sentence is the
strongest external support this document's §5 has, and it should be read
carefully: AutoAscend's persistent structure is *hand-written*, not learned. What
`NE.08` proposes is the same architectural bet with the structure *written by the
agent*.

Two cautionary details. NLE's reward function literally discards death
(`del end_status` in `nle/env/base.py`) **[V]**, and the challenge's median-based
ranking produced **deliberate death-gaming**: agents terminating early past a
score threshold and confining themselves to dungeon level 1, "a very high
incidence of death due to 'fainted of starvation'" **[V]**. A survival benchmark
whose metric does not price death gets agents that price it at zero — which is
`NE.00(d)`'s pathology arriving from the opposite direction.

### 1.4 What persists across an episode boundary

**Reincarnating RL** — Agarwal, Schwarzer, Castro, Courville, Bellemare, NeurIPS
2022, arXiv **2206.01626** **[V]**. Framed explicitly as *a research workflow*,
not an algorithm: "prior computational work (e.g., learned policies) is reused or
transferred between design iterations." Motivation: 50+ Atari games at 200M
frames with ≥5 runs is "more than 1000 GPU days". Their instantiation is **PVRL**
(policy-to-value), method **QDagger** = Dagger + n-step Q-learning with a
distillation term whose weight **decays** — the decay is the "weaning" mechanism,
and every baseline they tried (Rehearsal, JSRL, offline CQL pretraining,
Kickstarting, DQfD) **fails at the weaning point specifically**, DQfD with
"severe performance collapse". Results: QDagger **surpasses the teacher in 75 %
of runs** within 10M frames against a teacher trained on 400M; a reincarnated
Impala-CNN Rainbow needs **50M frames to match tabula rasa's 100M**.

Three stated pitfalls that `NE.08` must respect. **Teacher dependence** is not
just about teacher *quality*: "two policies with similar performance but obtained
from different agents… results in different performance… depends not only on the
teacher's performance but also on its **behavior**." **Benchmarking inverts**:
"Are student agents that are more data-efficient when trained from scratch also
better for reincarnating RL? … we answer this question in the negative" — DrQ
beats Rainbow tabula rasa in the low-data regime and *underperforms* it under
PVRL. And **reproducibility**: reincarnated results cannot be reproduced from
scratch without the prior computation, which is why `NE.08` must checkpoint and
hash the diary that produced each life.

**Go-Explore** — Ecoffet, Huizinga, Lehman, Stanley, Clune; arXiv **1901.10995**
and "First return, then explore", **Nature 590, 580–586 (2021)**, DOI
10.1038/s41586-020-03157-9 **[V]**. Two named failure modes: **detachment**
("the algorithm prematurely stops returning to certain areas… despite having
evidence that those areas are promising") and **derailment** ("the exploratory
mechanisms… prevent it from returning to previously visited states"). The fix is
an archive of *cells* plus **restoring the simulator state** to return. Numbers:
Montezuma's Revenge robustified **43,791** vs SOTA 11,618; Pitfall **6,954** vs
SOTA 0 and above average human; with domain-knowledge cells MR mean **> 1.7
million**, beating the human world record of 1.22M; on a sparse-reward Fetch arm
task PPO sees **zero reward in 1B frames** while Go-Explore's exploration phase
reliably solves all four shelves.

**And the mechanism is unavailable to us, for a principled reason, not a
technical one.** Simulator-state restore is a free teleport to a good state —
`LT` §2.1 forbids exactly that as an experimenter-supplied curriculum, and NLE's
own authors note Go-Explore is inapplicable to NetHack for the same reason. §5.3
therefore keeps the *idea* (an archive of promising states) and drops the
*intervention*: **the diary is the archive and "return" is a behaviour**, scored
as `frontier_return(n)`.

**Voyager** — Wang, Xie, Jiang, Mandlekar, Xiao, Zhu, Fan, Anandkumar; arXiv
**2305.16291** **[V]** (commonly cited as TMLR 2024, **[⚠] venue unverified**).
The skill library stores **executable code**, keyed by the *embedding of the
program's description*, retrieved top-5 into the code-generation prompt, and
added only after self-verification. Headline: **63 unique items in 160
iterations = 3.3× baselines**, 2.3× traversal distance, **15.3× faster to the
wooden tier** than AutoGPT, and the only method to reach diamond (102 iterations,
1/3 seeds).

**The experiment that matters for §5 is Table 2, and it is the strongest
available evidence for the whole cross-life thesis.** Clear the inventory, reset
to a **newly instantiated world**, give unseen tasks, 50-iteration budget:
Voyager-with-library solves all four in **19/18/21/18** iterations against
**36/30/27/26** without it — roughly **half the iterations in a world it has
never seen**. And the second finding is the one weights cannot reproduce:
**dropping Voyager's skill library into AutoGPT takes that agent from 0/4 tasks
to 3/4 partially solved.** A file transplanted between agents. This is §5.10's
Lamarckian claim, already demonstrated once in the literature.

Honest caveats, both in the paper: Voyager *without* the library still solves 3/4
of the transfer tasks (GPT-4 planning is strong), so the library is an
accelerant and a diamond-tier enabler rather than a binary capability; and the
skill-library row of the §3.4 ablation figure is **qualitative only** — "a
tendency to plateau in the later stages" — with **[⚠] no percentage given**. The
quantitative case is Tables 1 and 2, not the ablation figure. Also worth knowing:
Voyager's prompt *does* expose Minecraft's health and hunger to the LLM
(`Health: {health:.1f}/20`, `Hunger: {hunger:.1f}/20`, withheld until 15 tasks
are complete) **[V]** — but there is **no scalar reward and no needs ablation
anywhere in Voyager**, so it is not evidence that needs teach.

### 1.5 Memory as the cross-life carrier — who actually ablates it

The question `NE.08`'s `C-WIPE` asks is rarer in the literature than it should
be. Ranked by the strength of the memory-wiped condition **on the same agent**:

**Generative Agents** — Park, O'Brien, Cai, Morris, Liang, Bernstein; arXiv
**2304.03442**, UIST '23, DOI 10.1145/3586183.3606763 **[V]**. Memory stream
scored by **recency × importance × relevance** — the same three terms
`EpisodicMemory` implements, which is where `ME.1`'s design came from. Ablation
by TrueSkill over 100 human evaluators: full architecture **μ = 29.89**, no
reflection 26.88, no reflection+planning 25.64, **human crowdworker 22.95**, and
**no memory-stream access 21.21**. `H(4) = 150.29, p < 0.001`; all pairwise
`p < 0.001` **except crowdworker vs fully-ablated**. The memory-wiped agent is
statistically indistinguishable from a human writing the lines cold. (It is a
*believability* rating, not task success — a real limitation.)

**ExpeL** — Zhao, Huang, Xu, Lin, Liu, Huang; arXiv **2308.10144**, AAAI-24
**[V]**. **The sharpest "it has to be the *right* memory" number in the
literature.** ALFWorld success: ReAct 40.0 → reasoning-similarity retrieval
48.5 ± 2.1 → **random-sampled retrieval 42.5 ± 0.8** → **task-similarity
retrieval 59.0 ± 0.3**. Scrambling *which* memory is retrieved, with the memory
still present, loses **16.5 points ≈ 87 % of the gain**. This is precisely
`C-SHUFFLE-TIME`'s logic and it says the control has teeth.

**Optimus-1** — Li, Xie, Shao, Chen, Jiang, Nie; arXiv **2408.03615**, NeurIPS
2024 **[V]**. Knowledge graph + experience pool, **no weight updates**, starting
from an empty memory. Both memories off vs both on, average success by tier:
Wood **55.00 → 97.49**, Stone **47.37 → 94.26**, Iron **18.11 → 53.33**, Gold
**2.08 → 11.54**, Diamond **1.11 → 9.59**. And a finding `NE.08` should heed:
retrieving *successes only* is worse than retrieving successes **and failures** —
the death record is not decoration.

**Reflexion** — Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao; arXiv
**2303.11366**, NeurIPS 2023 **[V]**. "Reinforce agents **not by updating
weights**, but through linguistic feedback… maintained in an episodic memory
buffer" — a buffer of the **last 3 self-reflections**. Table 3 (HumanEval-Rust,
GPT-4): base 0.60, **self-reflection omitted 0.60**, Reflexion 0.68 — the entire
gain vanishes when the memory is wiped. Two honest negatives worth carrying:
**MBPP-Python 80.1 → 77.1 (Reflexion loses)**, and on WebShop "ReAct + Reflexion
fails to significantly outperform ReAct". Cross-episode memory is not free money.

**Two counter-examples that must stay in view.** Synapse (arXiv **2306.07863**,
ICLR 2024 **[V]**) finds memory's cross-domain delta is **+0.2 / −0.2** when
retrieval distance is high — **memory can hurt when what is retrieved is
dissimilar enough**, which is exactly `C-FOREIGN`'s risk and a reason to report
its sign, not just its magnitude. Agent Hospital (arXiv 2405.02957 **[⚠] venue
unverified**) finds retrieving *more* memory degrades performance past top-3
cases and top-4 experiences.

**And the null result to keep honest about:** AgentBench (arXiv **2308.03688**,
ICLR 2024 **[V]**) has **no persistent store, no memory condition and no memory
ablation** across 8 environments and 29 models. It is evidence of the *gap*, not
evidence about memory. The classic RL line — Model-Free Episodic Control
(**1606.04460**), Neural Episodic Control (**1703.01988**), and especially **Never
Give Up** (**2002.06038**, ICLR 2020) with its *episodic* novelty memory purged
every episode multiplied by a *life-long* RND modulator that persists — is the
cleanest prior formalisation of **what should and should not survive a death**,
and `NE.08`'s "what persists" table in §5.2 is that distinction made concrete.

### 1.6 Sleep consolidation

**SIESTA** — Harun, Gallardo, Hayes, Kemker, Kanan; arXiv **2303.10725**;
**Transactions on Machine Learning Research, 11/2023** **[V]**. Architecture
`F(G(H(·)))`: `H` = the first 8 layers of MobileNetV3-L (**2.19 %** of params),
**frozen for the whole of continual learning**; `G` = the remaining 11 layers;
`F` = a cosine-softmax output layer. **Wake**, per sample: only `F` updates, by a
closed-form running class mean — **no backprop, no rehearsal** — and the latent
tensor is product-quantised into the buffer. **Sleep**, periodic: `F` and `G` are
trained by backprop with rehearsal on **reconstructed latents**; `H` stays
frozen. Default cadence **every 120K samples ≈ every 100 classes**.

The numbers that make this the right template for a four-core box:
**+4.25 ± 1.38 % absolute accuracy after each sleep cycle**; **zero forgetting**
on augmentation-free class-incremental ImageNet-1K (Cochran's Q, `P = 0.08`,
i.e. iid / class-incremental / offline are indistinguishable) where "DER, ER, and
REMIND perform over 9 % worse than the offline learner"; **3.4× faster** than
REMIND wall-clock (2.4 h vs 8.1 h for 900 ImageNet classes on one A5000), with
**10× less memory, 2–20× fewer parameters, 7–60× fewer network updates**. And a
result that directly sets a hyperparameter for us: **post-sleep gain grows with
sleep interval** — 2.29 ± 0.86 % at every 50 classes, 4.25 ± 1.38 % at 100,
6.18 ± 2.15 % at 150 — because "frequent sleep leads to greater perturbation in
the DNN's weights, resulting in gradual forgetting of old memories." **Sleeping
too often is a known failure mode**, which is a reason to let sleep pressure set
the cadence rather than a timer, and a reason `NE.05`'s `random-sleep` control
matters.

**Sleep Replay Consolidation** — Tadros, Krishnan, Ramyaa, Bazhenov; **Nature
Communications 13, 7742 (2022)**, DOI 10.1038/s41467-022-34938-7 **[V]**. Offline,
**unsupervised, stores no data**: convert to a spiking-like network, drive it
with noisy Poisson input, apply local Hebbian/anti-Hebbian plasticity.
Class-incremental MNIST: sequential **19.49 %** → SRC alone **48.47 %** → SRC +
0.75 % rehearsal **86.47 %** (parallel-training bound 98.02 %); EWC 20.37, SI
21.38. iCaRL 65.5 → **78.1** with SRC. Relevant because it shows a sleep phase
can help **with no stored data at all** — which is the honest alternative
hypothesis to `NE.05`'s `empty-buffer` control being a pure destroyer.

Wake-Sleep Consolidated Learning (arXiv **2401.08623**, **[⚠] preprint, venue
unverified**) implements **NREM** (replay-based consolidation) and **REM**
("dreaming" over unseen inputs) as separate phases and reports the REM component
is what enables positive forward transfer — the citation behind §3.0's declared
divergence.

**And the gap:** the survey found **no sleep-consolidation-in-deep-RL paper of
SIESTA's calibre**. `NE.05` is not reproducing a known result; it is asking a
question the field has not answered in a control setting.

### 1.7 The one-line summary of the field

**The homeostatic reward form is settled theory and unsettled practice.**
Keramati & Gutkin proved in 2014 that maximising discounted drive reduction *is*
minimising discounted deviation from setpoint, so the form needs no anti-farming
patch and no hand-written stability term; but the deep-RL implementations are
few, the multi-need scaling question is open (MO-MPO vs a single scalar), and the
reward-design literature's own finding is that experts misdesign these rewards
even when the unshaped version is learnable.

**Needs-plus-death worlds are standard and nobody has ablated the needs.**
Crafter and NetHack both ship metabolic drives and permadeath, and neither has an
experiment isolating whether the drives *teach*. **Memory-not-weights transfer
across lives is demonstrated but almost always in LLM agents, never with a
homeostatic body** — Voyager's library halves iterations in an unseen world and
transplants into a different agent; ExpeL's random-retrieval row shows 87 % of
the gain lives in *which* memory is retrieved; Generative Agents' memory-wiped
condition is indistinguishable from a human writing cold. **Sleep as a
consolidation schedule is proven in supervised continual learning (+4.25 % per
cycle, zero forgetting, 3.4× cheaper) and untested in RL.** And the strongest
single fact about survival worlds — that the best NetHack agent by 3× is the one
whose knowledge lives outside the episode as persistent structure — is an
argument for exactly the architecture §5 proposes, with the difference that
Jack's structure would be written by Jack.

---

## 2. The needs suite

### 2.1 The design rule, stated before any variable

The owner said *"as simple and robustly as possible"*. That is not a mood; it is
a constraint that has to be operationalised or it will be quietly ignored. Four
rules, and every variable below obeys all four:

1. **One scalar per need.** No need gets two state variables. If a need seems to
   need two, it is two needs and must earn both places separately.
2. **Every need enters `d(h)` through exactly one term** `λ_k · δ_k^n`, where
   `δ_k ∈ [0,1]` is its normalised deviation from setpoint. There are no
   need-specific reward terms, no bonuses, no special cases. The whole reward is
   `d(h_t) − d(h_{t+1})` plus one declared survival term.
3. **Every coupling between needs is a physical mechanism, not a tuned
   cross-term.** "Cold burns calories" is *not* a `λ_{eT}` in the drive function;
   it is shivering thermogenesis raising metabolic rate, which drains energy,
   because energy drain is defined as proportional to metabolic rate. One
   equation, two consequences.
4. **Every variable arrives with the ablation that deletes it** (§2.7), and the
   ablation is a *spec*, not a promise. `NE.02` is a standing spec in the shape
   of `UB.11`: it re-runs on every architecture change, forever, and a need whose
   column is indistinguishable from a **placebo need** loses its place.

### 2.1b Biology is the oracle, not the blueprint

Owner principle, 2026-08-09: *"human and nature biology is the best model for
building this."* `GOAL.md` records the nuance that governs how it is used here:
**biology is the ORACLE, not the blueprint.** Nature's solution enters as a
bakeoff arm and must win on our substrate. So every variable below arrives with
two things attached: **the biological mechanism it mirrors**, and **the place
where it deliberately diverges**. Divergences are fine. Silent divergences are
not — an undeclared simplification is how a model stops being a model and starts
being a story.

| need | reference implementation in biology | how Jack mirrors it | **declared divergence** |
|---|---|---|---|
| **energy** | hypothalamic arcuate nucleus: AgRP/NPY (orexigenic) vs POMC (anorexigenic) neurons, driven by ghrelin (short-term, meal-timing) and leptin (long-term, adiposity) | one scalar with a setpoint, drained in proportion to metabolic rate, restored by a physical ingestion event | **two loops collapsed to one.** Humans have a fast meal-timing signal and a slow adiposity signal with different time constants. Jack has one. Reopened if `NE.02` finds within-day and across-day hunger behave differently. Timescale compressed ~1,000×. |
| **water** | circumventricular organs — subfornical organ (SFO) and OVLT — sensing plasma osmolality outside the blood–brain barrier | one scalar, drained ~4× faster than energy, restored by a drinking event | **osmotic and volumetric thirst collapsed to one.** Compressed ~600×. |
| **sleep** | Borbély's two-process model: homeostatic process S (adenosine accumulation while awake) × circadian process C (suprachiasmatic nucleus) | **process S only** | **C is deliberately omitted**, and the omission is the experiment (§2.3). If night-alignment emerges from darkness and cold alone, the oscillator is unnecessary; if it does not, C is added in the open. |
| **temperature** | preoptic area / anterior hypothalamus as the thermostat; shivering thermogenesis; sweating; behavioural thermoregulation (which is the dominant human mechanism and the one that builds shelters) | one-compartment thermal ODE, shivering that raises `M`, sweating that costs water, and **`sky_occlusion` as the behavioural channel** | **single compartment.** No core/shell distinction, no counter-current exchange, no brown adipose tissue. Lethal bounds are asymmetric as in humans, but the *approach* to them is ~50× faster than the approach to starvation — the largest single distortion in the suite, bounded by `NE.01`'s death-cause gate. |
| **fatigue** | peripheral muscular fatigue (metabolite accumulation, impaired Ca²⁺ handling), distinct from central fatigue and from sleep pressure | one scalar rising with mechanical power, recovering in ~60 s | **central and peripheral fatigue collapsed**, and no per-muscle-group state — a real climber's grip fails before their legs do, and Jack's does not. |
| **social** | affiliative motivation, and the "social homeostasis" account of isolation as a need-like state | one scalar drained by solitude, restored only by *recorded reciprocal* interaction | **the weakest biological analogy in the suite**, and the one whose need-status is most contested in its own literature. Its λ is the smallest and `NE.02` is explicitly empowered to delete it. |
| **integrity / pain** | nociception: A-δ and C fibres → spinothalamic tract → thalamus, amygdala, periaqueductal grey. Fast, unconditioned, **sensitising rather than habituating** | tonic `i` in `d(h)` plus a phasic rectified `−Δi` channel | **the split is a live question, not a settled design** — see §2.9, where it is an arm. |
| **the observation itself** | **interoception**: the afferent sensing of bodily state, and its role in motivated behaviour and in feeling | the need vector is concatenated into the observation | **Jack's interoception is noiseless, instantaneous and complete.** Human interoception is noisy, delayed, and partly *inferred* rather than measured. This divergence makes Jack's problem strictly **easier** than a human's, and it should be recorded as an advantage he has, not as fidelity. |

#### The one biological result that changes the design: allostasis

Homeostasis is reactive — act once you have deviated. **Biology does not do
that.** Sterling's allostasis is *predictive* regulation: the organism acts
*before* the deviation. The decisive modern evidence is that hypothalamic hunger
and thirst neurons are **suppressed by the mere sight of food or water, seconds
before any ingestion** — the correction begins before the nutrient arrives.

This could have been implemented as a hand-coded anticipatory term. It should
not be, for the same reason a hand-written climb reward should not be: it would
be an instruction dressed as physiology. **A discounted RL agent with a value
function anticipates by construction** — it eats before it is empty because the
value of being empty propagates backwards. So allostasis is entered here as a
**prediction that must emerge**, with a control:

```
anticipatory_consumption_fraction
      = fraction of consumption events occurring while delta_need < 0.3
        (i.e. while still comfortably supplied)
   PREDICTION: rises across a life, and is HIGHER for the discounted agent
   CONTROL:    a MYOPIC arm (gamma -> 0.5) must eat only when nearly empty.
               If the myopic agent anticipates just as much, the measurement is
               reading food availability, not foresight.
```

Reported in `NE.03`, gated in nothing. If it emerges, Jack reproduces a
non-obvious neurophysiological result from a reward function that says nothing
about anticipation — which is the strongest form the "biology is the oracle"
claim can take here.

### 2.2 The seven variables

```
h = (e, w, p, T, f, c, i)

  e  energy        1 = fed          0 = starving        setpoint 1     LETHAL
  w  water         1 = hydrated     0 = dehydrated      setpoint 1     LETHAL
  p  sleep         0 = rested       1 = must sleep      setpoint 0     lethal INDIRECTLY
  T  core temp     37 °C                                setpoint 37    LETHAL BOTH SIDES
  f  fatigue       0 = fresh        1 = spent           setpoint 0     not lethal
  c  social        1 = contented    0 = isolated        setpoint 1     not lethal
  i  integrity     1 = unhurt       0 = wrecked         setpoint 1     LETHAL
```

Normalised deviations, all in `[0,1]`:

```
δ_e = 1 − e      δ_w = 1 − w      δ_p = p
δ_T = min(1, |T − 37| / 5)        δ_f = f      δ_c = 1 − c      δ_i = 1 − i
```

`δ_T`'s divisor of 5 °C is the *comfort* half-width, not the lethal one: at
±5 °C the temperature term already saturates the drive, so the agent feels
maximum thermal urgency long before the lethal bounds at 28/42 °C. That gap —
saturated urgency at 32 °C, death at 28 °C — is the margin in which a policy can
learn. A drive that saturates only at death gives no gradient while dying.

**The whole suite on one page.** Every column below is specified in §2.3–§2.6;
this table exists so the design can be argued with without reading them.

| need | depletes because | restored by | key interaction | in the obs | in the reward | death | the ablation that deletes it |
|---|---|---|---|---|---|---|---|
| **`e` energy** | `−(M/M_basal)·b_e`, `b_e = 1/1800 s⁻¹` | a **physical consumption event** (mouth geom contacts food, food teleports out, respawn timer starts) | **cold triples the drain** via shivering; low `e` lowers `M`, so starving makes you colder | `e` | `λ_e = 1.0`, `δ_e = 1−e` | `e = 0` for **300 s** | clamp at setpoint, remove food from the world. `t_food` and the forage/explore interleave must collapse. |
| **`w` water** | `−(1 + c_sw·max(0,T−37))·b_w`, `b_w = 1/450 s⁻¹` | a drinking event at the pool | **heat doubles the drain** via sweating; sweating is also how you shed heat | `w` | `λ_w = 1.0`, `δ_w = 1−w` | `w = 0` for **120 s** | clamp; the pool reverts to hazard-only. `t_water` must collapse. |
| **`p` sleep** | `p ← p + (1−p)(1−e^{−Δt/700})` awake | sleeping: `p ← p·e^{−Δt/160}` | high `p` → **microsleeps** → falls; sleep is when consolidation runs | `p` | `λ_p = 0.5`, `δ_p = p` | **none directly** — indirect via microsleep, and the indirection is measured (`deaths_with_microsleep_within_10s`) | cannot be ablated fairly in `NE.02` (it also removes the consolidation trigger); its ablation is `NE.05`'s `timer` arm |
| **`T` temperature** | `C_eff·Ṫ = M − k_eff(T−T_env) − E_evap` | shivering, activity, **shelter** (`sky_occlusion` cuts `k_eff` by up to 70 %), leaving the water, daylight | **the hub**: wet skin multiplies heat loss, night drops `T_env`, cold burns energy, heat burns water | `T_norm`, **signed** | `λ_T = 1.0`, `δ_T = min(1,\|T−37\|/5)` | `T ≤ 28` or `T ≥ 42` for **20 s** | clamp; shelter behaviour and `sky_occlusion_at_sleep_onset` must collapse. **The only mechanism that teaches construction.** |
| **`f` fatigue** | `ḟ = P_mech/(P_max·τ_rise) − f/60` | resting (~60 s), and sleep sets `f → 0` | scales `gear_scale`; low `f` means more `P_mech`, which warms you | `f` | `λ_f = 0.3`, `δ_f = f` | none | **the sharpest deletion test**: clamp `f=0`, rescale `κ_act` to match total energy drain. If within-bout pacing is unchanged, fatigue was a slow duplicate of energy. |
| **`c` social** | `−b_c`, `b_c = 1/3600 s⁻¹` | proximity, **reciprocated** conversation, being helped, helping — each a **recorded diary event**, each with within-bout decay `β = 0.6` | **none physical** — an empty row in §2.4, which is why it must earn its place behaviourally | `c` | `λ_c = 0.3`, `δ_c = 1−c` | none | clamp; `approach_lift` must fall to the placebo column. The companion angle then survives as language only (§6). |
| **`i` integrity** | impact impulse above `J₀`, drowning, `\|T−37\| > 5` | heals at `1/900 s⁻¹` while at rest | weakness floor on `gear_scale`; injury → falls → more injury | `i` **and** `pain` (phasic) | `λ_i = 1.0`, `δ_i = 1−i`; §2.9 asks whether the phasic part should be a **separate channel** | `i = 0`, **immediate** | clamp; caution near the ladder and pool must vanish. PS §2.1 argued this is the only variable supplying *a cost of failing*, so this is the most consequential deletion available. |

### 2.3 Dynamics, and the one honest compromise about time

**The time base.** One decision = 0.2 s of simulated time
(`CURIOSITY_BAKEOFF` §6: ~81 decisions/s physics-bound at 40 substeps). A
**sim-day is 1,200 s = 6,000 decisions**, split 800 s day / 400 s night. A
50,000-decision life is therefore **8.3 sim-days**, which is the shortest life
that contains enough nights for sleep to be a variable rather than an event.

**The compromise, declared.** Human need timescales span four orders of
magnitude — fatigue in minutes, hypothermia in hours, thirst in days, starvation
in weeks. No single compression constant maps that onto a 2.8-hour life. So the
suite preserves the **ordering** of human timescales and compresses the
**spread** from ~10⁴ to ~10²:

| need | human, at rest | here (sim-seconds) | compression | note |
|---|---|---|---|---|
| fatigue | minutes | `τ_f` = 60 s recovery | ~1× | fastest by design |
| temperature | hours (wet/cold: <1 h) | `τ_T` = 240 s | ~50× | **deliberately over-weighted** |
| sleep | ~16 h awake | `τ_wake` = 700 s, `τ_sleep` = 160 s | ~80× | ratio 4.4 : 1, intended to match Borbély's ≈18.2 h / ≈4.2 h — **[⚠] §1.2: that ratio was not verified this pass and is load-bearing for these two constants** |
| thirst | ~3 days | 450 s to empty | ~600× | ~3 drinks per sim-day |
| hunger | ~3 weeks | 1,800 s to empty | ~1,000× | ~1 meal per 1.5 sim-days |
| social | no lethal bound | 3,600 s to empty | — | 3 sim-days of solitude to bottom out |
| integrity | event-driven | heal `τ_i` = 900 s | — | inherited from PS §2.2 |

Temperature is the deliberate distortion: relative to a human, cold here is
roughly 20× more dangerous than hunger. That is a choice, and the reason is that
temperature is the need the owner named twice and the only one that teaches
**shelter**. The guard against it swallowing the suite is `NE.01`'s
`max_single_death_cause_share ≤ 0.6`: if more than 60 % of random-policy deaths
are thermal, the calibration is wrong and the other six needs are decorative in
practice, whatever their λ says.

**Metabolic rate is the hub.** Everything couples through it:

```
M(t) = M_basal + κ_act·P_mech(t) + M_shiver(t)          [watts, nominal]

  P_mech  = Σ_j |τ_j · ω_j|                  from qfrc_actuator · qvel  (PS §2.2)
  M_shiver = c_sh · max(0, 37 − T)           capped at 2·M_basal
```

and then, in one line each:

```
ė = −(M/M_basal) · b_e            + Σ_food ν_food · ate(t)
ẇ = −(1 + c_sw·max(0, T−37)) · b_w + Σ_src  ν_src  · drank(t)
```

`b_e = 1/1800 s⁻¹`, `b_w = 1/450 s⁻¹`. So **cold burns calories** (shivering
triples `M`, tripling energy drain) and **heat burns water** (sweating).
Two couplings, no cross-terms in the drive function, exactly rule 3.

**Core temperature — the shelter engine.**

```
C_eff · Ṫ = M(t) − k_eff(t)·(T − T_env(t)) − E_evap(t)

  k_eff = k_dry · (1 + κ_wet · skin_wetness) · (1 − 0.7 · sky_occlusion)
  E_evap = c_ev · skin_wetness · (1 + c_sw·max(0, T−37))
  T_env(t) = T_day  by day,  T_day − ΔT_night  by night;  pool water at T_water
  C_eff  = k_dry · τ_T      (so the thermoneutral time constant is exactly τ_T = 240 s)
```

Four things fall out of that one equation and each is a curriculum:

- **`skin_wetness` replaces PS's `wetness` need.** Being wet multiplies heat loss
  and adds evaporative cooling. Swimming in a cold pool at night is the fastest
  available way to die, and it is fast for a *physical* reason. PG.2's water
  finally has a stake.
- **`sky_occlusion` is the shelter term, and it is geometric, not labelled.** It
  is the fraction of 9 upward rays from the head geom that hit *something*, cast
  once per decision. There is no "shelter zone", no tagged geom, no coordinate.
  The five loose objects and the seesaw plank the playground already contains can
  be pushed and tipped into configurations that occlude sky. **Shelter is
  therefore constructible and its measurement cannot be gamed by going to a
  place.**
- **The G1 static audit still passes.** The reward path references `d(h)`, which
  references `T`, which references `k_eff`, which references `sky_occlusion`. No
  reward path contains `shelter`, `ladder`, `platform`, `climb`, `height`,
  `torso_z` — or `ate_apple`. A match is **ERROR**, not FAIL (PS §5/G-F,
  inherited).
- **Night is the teacher.** `ΔT_night` is calibrated by `NE.01` so that a night
  spent in the open at `sky_occlusion = 0` costs 0.3–0.6 of integrity-equivalent
  drive and is survivable *once*, while a night at `sky_occlusion ≥ 0.4` is
  nearly free. Survivable-but-expensive is the whole pedagogy: a lethal night
  teaches nothing because there is no second attempt inside the life, and a free
  night teaches nothing because there is nothing to avoid.

**Sleep pressure — Borbély's process S, and only process S.**

```
awake:   p ← p + (1 − p)(1 − e^{−Δt/τ_wake})        τ_wake  = 700 s
asleep:  p ← p · e^{−Δt/τ_sleep}                     τ_sleep = 160 s
```

A full 800 s day takes `p` from 0.05 to 0.70; a full 400 s night returns it to
0.06. **There is deliberately no circadian process C.** The two-process model's
C term is a sinusoid nobody here can defend, and it would *impose* the very
phase-locking we want to observe. Instead night is dark (vision degrades) and
cold (foraging costs more), so being awake at night is unproductive on its own
terms. The pre-registered prediction is therefore behavioural, not architectural:

> `sleep_night_alignment` = (share of sleep decisions falling at night) ÷ 0.33
> must exceed **1.5** without any circadian term in the model.

If it does not, C is added — as a change made in the open, with the null result
recorded — rather than assumed at the start. This is `SYSTEM.md` law 3 applied to
a model component instead of an architecture.

**Fatigue** — the fast one, and the one most at risk of deletion:

```
ḟ = P_mech / (P_max · τ_f_rise)  −  f / τ_f_fall        τ_f_fall = 60 s
gear_scale(t) = 0.5 + 0.5·(1 − f) · min(e, i)           (PS §2.2's weakness floor,
                                                          now also fatigue-gated)
```

Sleep sets `f → 0` outright. Fatigue's only distinctive claim is its **time
constant**: it forces pacing *within* a bout, on a scale (minutes) where energy
(1,800 s) cannot act. §2.7 gives the ablation that deletes it if that claim is
false.

**Social `c`** — §4.

**Integrity `i`** — PS §2.2 verbatim, with three additions: it is also damaged by
`T` outside `[32, 40]` at rate `α_T·(|T−37| − 5)`, `i = 0` is now **lethal**, and
the rectified `−Δi` of the current decision is exposed as a separate **pain**
channel in the observation (phasic), distinct from `i` itself (tonic).

### 2.4 The interaction table — every coupling, in one place

| | affects `e` | `w` | `p` | `T` | `f` | `c` | `i` |
|---|---|---|---|---|---|---|---|
| **`e` low** | — | | | less `M`, so colder | weakness floor | | weakness → falls |
| **`w` low** | | — | | less sweating, so hotter | | | |
| **`p` high** | | | — | | | | microsleeps → falls |
| **`T` low** | shiver: **up to 3× drain** | | | — | | | frostbite below 32 °C |
| **`T` high** | | sweat: **up to 2× drain** | | — | | | heatstroke above 40 °C |
| **`f` high** | | | | less `P_mech`, so colder | — | | weaker grip → falls |
| **`c` low** | | | | | | — | |
| **`i` low** | weakness floor | | | | | | — |
| **activity** | `κ_act·P_mech` | | | `+M` warms | `+f` | | impacts |
| **sleep** | (still drains) | (still drains) | **`p → 0`** | (still cools) | **`f → 0`** | | (still heals) |
| **wet skin** | | | | **`k_eff ×(1+κ_wet)`, `+E_evap`** | | | drowning |
| **night** | | | | **`T_env − ΔT_night`** | | | |
| **shelter** | | | | **`k_eff ×(1−0.7·occlusion)`** | | | |

The empty cells are the design. `c` (social) has an **empty row** — it affects
nothing physical — which makes it the need most exposed to `NE.02`'s placebo
column, and that is correct: if a variable that changes nothing about the body
also changes nothing about the behaviour, it is decoration and should go. §4
states what it must produce to keep its place.

### 2.4b How the needs enter the observation (interoception, in nine floats)

Every arm gets the same channels, including the no-needs null:

```
obs = concat( humanoid_obs(model, data),        # 348, from playground.py, UNMODIFIED
              [ e, w, p, T_norm, f, c, i,       # the seven levels        (7)
                d(h),                           # the scalar drive        (1)
                pain ] )                        # rectified max(0, -Δi)   (1)

NEED_DIM = 9        T_norm = (T − 37) / 5, signed and clipped to [−1, 1]
```

Five decisions, each with a reason and four of them paid for by something that
already went wrong in this repo:

- **`humanoid_obs()` is not touched.** The concatenation happens in the wrapper,
  outside the function whose 348 is asserted against gymnasium. `T0.14` found
  `mujoco_obs_dim = 376` — a Humanoid-**v4** constant — padding 28 dead columns
  into every observation for the project's entire history, and the lesson was
  *"assert contracts against the source of truth, not against another
  constant"*. `NEED_DIM` is a module constant and the wrapper asserts
  `obs.shape[0] == HUMANOID_OBS_DIM + NEED_DIM` **against the live model**.
- **`T_norm` is signed.** `δ_T` in the drive function is an absolute deviation,
  because the drive does not care which way you are dying. The *observation* must
  carry the sign, or the policy cannot tell shivering from sweating and the only
  two behaviours that help are indistinguishable.
- **`d(h)` is handed over explicitly**, though it is a function of the other
  seven. A policy that must *learn* the drive function before it can act on it is
  being tested on representation learning, not on motivation. This removes a
  confound and costs one float (PS §2.4, inherited).
- **`pain` is separate from `i`.** Tonic damage and the phasic event that caused
  it are different signals with different time constants, and §2.9's reflexes and
  pain channel both read the phasic one. Including it here costs one float and
  keeps §2.9's arms comparable — both arms *observe* pain; they differ only in
  whether it is a separate **reward** channel.
- **The null gets all nine channels too.** Not because it needs them, but because
  an arm with a different input width is a different architecture, and
  `LESSONS.md`'s "matched steps has more than one meaning" applies to matched
  *inputs*. The comparison in `NE.03` is over the **reward**, and only over the
  reward.

**Declared divergence from biology** (§2.1b): this interoceptive channel is
noiseless, undelayed and complete. Real interoception is none of those, and a
future arm that adds observation noise and a one-decision delay is the honest
version. It is not run here, and Jack's advantage over an animal is recorded
rather than claimed as realism.

### 2.5 Death

```
starvation     e = 0 continuously for 300 s
dehydration    w = 0 continuously for 120 s
hypothermia    T <= 28 C for 20 s          <- [!] see 1.2: NOT VERIFIED this pass
hyperthermia   T >= 42 C for 20 s          <- [!] and both are DESIGN CONSTANTS
injury         i = 0                                    (immediate)
drowning       head geom submerged > 20 s               (routed through i)
sleep          NO DIRECT DEATH CONDITION
social         NO DIRECT DEATH CONDITION
```

**Sleep has no lethal bound and that is a considered decision.** Total sleep
deprivation is lethal in rats over weeks (Rechtschaffen) and has never killed a
human volunteer; adding a lethal bound would mean inventing a number no source
supports. Instead sleep kills *indirectly and legibly*: at `p ≥ 0.98` the
policy's action output is zeroed for 1–2 s with rising probability (a microsleep),
and a microsleep on a ladder is a fall. Every death record carries
`microsleep_within_10s`, so **the indirect lethality of sleep debt is a measured
quantity rather than a modelling assumption**. If it reads zero across a whole
programme, sleep is not dangerous here and `NE.02` will say so.

Grace windows exist because an instantaneous lethal bound is unlearnable: the
agent must be able to be in trouble, notice, and act. 300 s of starvation is
1,500 decisions — enough to walk across the arena and eat.

### 2.6 The reward, and the arms

**Default (the recommendation):**

```
r_t = [ d(h_t) − d(h_{t+1}) ]  +  ρ · alive_t

d(h) = ( Σ_k λ_k · δ_k^n )^{1/m}        n = 4, m = 2 (calibrated by NE.01)
                                        CONSTRAINT: n > m > 1, from Keramati &
                                        Gutkin — it is what makes the reward
                                        concave in the outcome, and it is what
                                        their dose-dependence, deprivation-
                                        sensitivity and risk-aversion results
                                        require. n = 4, m = 2 satisfies it;
                                        NE.01 may retune INSIDE the constraint
                                        and may not leave it.
λ = (e 1.0, w 1.0, p 0.5, T 1.0, f 0.3, c 0.3, i 1.0)
ρ = 1 / 6000  per decision   ≡  "one unit per sim-day survived"
```

`ρ` exists for one reason and it is arithmetic, not philosophy. Drive reduction
telescopes (§0.2b), so the *entire lifetime return* of a pure-DR agent is
`d_0 − γ^T·d_T`, bounded by ±1 no matter how long the life. Longevity is
rewarded only through the discount, at a strength that decays as `γ^T`. `ρ`
makes living a first-class objective at an interpretable scale: one sim-day of
survival is worth one unit, against roughly 5–10 units of drive reduction earned
per day by a competent forager. `ρ` cannot be farmed by inaction — a motionless
agent starves, and PS §5/G-B's three provisions against the dark room are
inherited unchanged.

**The arms of NE.03/NE.04**, all scored on the same ruler (§8.1):

| arm | reward | why it is here | expected |
|---|---|---|---|
| `no-needs` **(NULL)** | ≡ 0. Needs integrated, logged, and in the observation; not in the reward; **death disabled**. | Defines `C₀`. Same architecture, same observation width, same world (PS §2.3's "byte-identical world" rule). | near-zero competence; the floor |
| `surv` | `+ρ` per step alive, nothing else | The simplest possible needs reward: needs enter only through **death**. If this wins, `d(h)` is unnecessary machinery and the whole drive function is deleted. The Crafter/NetHack shape. | credible winner; the honest threat to the homeostatic design |
| `dr` | `d(h_t) − d(h_{t+1})`, `ρ = 0` | Keramati & Gutkin, literally. | learns; may be indifferent to longevity (§2.6's arithmetic) |
| **`dr+surv`** | the default above | The favourite. | favourite |
| `dr+surv+pain` | as above, but the phasic `max(0, −Δi)` is a **second channel with its own value head and a fixed normaliser** | §2.9. Biology separates nociception from appetitive reward; a running normaliser makes a folded pain term habituate as an implementation accident. | decides §2.9 |
| `myopic` **(CONTROL)** | `dr+surv` at `γ → 0.5` | §2.1b's allostasis control: it must eat only when nearly empty. If it anticipates as much as the discounted agent, `anticipatory_consumption_fraction` is reading food availability, not foresight. | anticipates less |
| `cc+ρ` | `−d(h_{t+1}) + ρ`, with `ρ > max_h d(h)` **asserted before the run** | Constant-cost, the other half of the §0.2(a) identity. Must declare and prove its `ρ` or it learns to die. | ties `dr+surv`; if it does, take the cheaper |
| `eat` **(CONTROL, must fail)** | `+1` per consumption event, unbounded | The careless implementation. Measured in §0.2 to produce a different and worse policy at every γ. Must lose *and* must show the highest `drive_cycle_rate` of any arm. | fails |
| `cc` **(CONTROL, must fail)** | `−d(h_{t+1})`, `ρ = 0` | The suicide machine of §0.2(d). Must show `median_lifespan < no-needs` and a death-cause distribution dominated by *voluntary* inaction. | fails, by dying |
| `statue` **(CONTROL, must fail)** | do nothing | PS §5/G-B, inherited. Best integrity, worst everything, dies of starvation. | fails |
| `shuffle` **(CONTROL, must fail)** | `dr+surv`'s reward stream shuffled in time | The critical null: same reward magnitude distribution, no need semantics. If it matches, the effect was "any dense reward". | fails |

Note what the control set now buys that PS's did not: `eat` and `cc` are
**controls derived from a measurement**, each with a *pre-registered failure
signature* (`eat` → cycling; `cc` → voluntary death) rather than merely a
pre-registered side. A control that must fail *in a specific way* is a much
stronger instrument than one that must merely fail.

### 2.7 Every variable's deletion ablation (`NE.02`, the Tier-3 spec)

`UB.11`'s shape, applied to needs. For each need `k`, the **ablation** is: clamp
`δ_k = 0` (setpoint), remove its observation channels, remove its death
condition, and **rescale the remaining λ so `max_h d(h)` is unchanged** — without
that rescaling the ablation also changes the reward scale and measures the scale.

The column that makes it interpretable is the **PLACEBO NEED**: an eighth
variable with the same `λ`, the same observation channels, and dynamics driven by
band-limited noise with the same autocorrelation as the median real need, wired
in identically. Its column is the empirical null distribution for "decorative",
re-estimated every run — exactly `UB.11`'s placebo modality.

| need | what its deletion must break | if nothing breaks |
|---|---|---|
| `e` energy | `t_food`, the forage/explore interleave, and any arm's death-cause distribution | delete; the world's food becomes scenery again (PS.06's question, now decided here) |
| `w` water | `t_water`; the pool changes from hazard-only to resource-and-hazard | delete `w`, keep the pool as a thermal hazard |
| `p` sleep | **consolidation never runs** — so this ablation is not a fair test of `p` alone unless consolidation is re-triggered on a timer. §3's `TIMER` arm *is* this ablation, done properly. | `p` becomes a pure scheduler with no need semantics: keep the schedule, drop the variable from `d(h)` and from the observation |
| `T` temperature | shelter behaviour, night behaviour, `sky_occlusion_at_sleep_onset` | delete, and with it the only mechanism that teaches construction |
| `f` fatigue | **the ablation with the sharpest prediction.** Clamp `f = 0` and rescale `κ_act` so total energy drain over a life is matched. If the within-bout pacing structure (`P_mech` autocorrelation, rest-bout length distribution) is unchanged, fatigue was a slow duplicate of energy. | delete — 1 of 7 gone, and the suite is simpler |
| `c` social | `approach_lift` (§4.5) | delete; the companion angle survives as a *language* property (§6) with no need behind it |
| `i` integrity | caution near the ladder and the pool; `fall_rate` after the first injury | delete — but note PS §2.1 argued this is the *only* variable that supplies "a cost of failing", so its deletion is the most consequential possible outcome of `NE.02` |

**The gate:** every real need's minimum degradation must exceed the placebo
column's, at 3 seeds, on at least one of {survival, competence battery,
behavioural signature}. A need with an all-placebo row loses its place, and
deletion is the default action, not a discussion (`UB.11`'s rule, verbatim).

**The reverse gate, which is the one people forget:** the placebo column must be
**small**. A large placebo Δ means the ablation procedure measures off-manifold
shock rather than information, and then no column is interpretable — `UB.11`'s
control (a), inherited word for word.

### 2.8 How needs meet curiosity

Inherited from PS §2.8 unchanged, with one change of interpretation. Two value
heads (RND's design), never a summed scalar, because a head can be ablated
cleanly and a term inside a learned scalar cannot. And satiety-gated curiosity:

```
β_c(t) = β_0 · ( 1 − d(h_t) / d_max )
```

Under permanent needs this is no longer a story about deployment. It is a
falsifiable behavioural signature available at **zero marginal cost** from stored
trajectories: `corr(satiety_t, exploration_rate_t) > 0`, i.e. **interleaved
foraging and exploration bouts**, which neither pure curiosity nor pure
homeostasis produces. It is reported for every arm in `NE.03` and gated in none —
a secondary hypothesis, pre-registered so that finding it later cannot be dressed
up as a prediction.

The inherited secondary prediction also stands, and it is the one genuine
argument for needs that curiosity cannot make: **a need should be a noisy-TV
antidote, because a noise panel does not feed you.**
`panel_dwell(dr+surv) ≤ panel_dwell(no-needs)` and
`chaos_occupancy(dr+surv) ≤ chaos_occupancy(curiosity)`.

### 2.9 PAIN vs REWARD: should damage be a separate channel?

Biology answers this unambiguously, and it answers it *against* §2.1's rule 2.
Nociception is a **separate system** from appetitive reward: fast, unconditioned,
carried on dedicated fibres, capable of triggering a spinal withdrawal reflex
before any cortical processing, and — the property that matters most here — it
**sensitises rather than habituates**. Appetitive dopaminergic reward does the
opposite: it adapts, and a prediction error that stops being surprising stops
being a signal. Opponent-process accounts of affect formalise the asymmetry.

So the question is real and it is not decided by preference. Four arguments for
splitting, and the second is the one that is mechanical rather than analogical:

1. **Timescale.** Pain is phasic and rare; drive reduction is small and constant.
   One channel carrying both means a critic learning a distribution with a long
   negative tail.
2. **A single normalised reward stream makes pain habituate, literally.** `T2.00`
   mandates return normalisation — it exists because the un-normalised
   configuration produced `vf/pg ≈ 870`. But a running normaliser divides by a
   standard deviation that *the impacts themselves inflate*. **The more often he
   gets hurt, the less each injury counts.** That is precisely the adaptation
   biology avoids, arrived at by an implementation detail nobody chose.
3. **Reflexes need something to read.** A withdrawal reflex (§2.10) is triggered
   by nociception, not by value. If pain is one term inside a scalar, there is no
   signal for a reflex to be wired to.
4. **Ablatability.** Two heads is a clean surgical boundary; a term inside a
   learned scalar is not (§2.8's RND two-head argument, reused).

And two arguments against, which are not weak:

1. It **breaks §2.1's rule 2** — one term per need, no special cases — and that
   rule is what keeps this suite defensible.
2. Two heads need a mixing coefficient, which is one more hyperparameter nobody
   can defend, and "scalar reward is enough" is a serious position.

**The decision is an arm, not an assumption.**

```
dr+surv          integrity enters d(h) exactly like every other need   (folded)
dr+surv+pain     the TONIC term i stays in d(h);
                 the PHASIC rectified max(0, -delta_i) becomes a second negative
                 channel with its own value head and a FIXED normaliser
                 calibrated once by NE.01 -- a fixed normaliser IS the
                 non-habituation property, made mechanical
```

**The measurement that decides it, and the one that reopens it if we reject the
split:**

```
pain_habituation = effective reward magnitude of a fixed physical impulse J
                   in the FINAL fifth of a life
                 / the same in the FIRST fifth

  folded design:  predicted to FALL (the running normaliser grows)
  split  design:  predicted FLAT by construction
  IF BOTH ARE FLAT: the concern was theoretical, the split is deleted,
                    and rule 2 survives intact.

REOPEN CONDITION, pre-registered: if pain_habituation under the folded design
falls below 0.5 by late life, OR if impact events contribute more than 30% of
total TD-error variance, the split is reopened whatever the competence numbers
said.
```

Note what this buys beyond the immediate answer: `pain_habituation` is a
**general instrument**. Any project that normalises returns and also has rare
high-magnitude events has this bug latent, and nothing in the ladder currently
looks for it.

### 2.10 REFLEX PRIORS: infants are not blank

Human neonates arrive with innate motor programmes — the palmar grasp reflex
(strong enough to bear body weight), the Moro reflex, the stepping reflex, the
nociceptive withdrawal reflex, the righting and parachute reactions. They are
not skills; they are **scaffolds that make the first weeks of learning
tractable**, and they are progressively *suppressed* as cortical control
develops. Jack currently arrives with nothing.

**The minimal set — three reflexes, ~20 lines each, chosen because each is
defensible on its own:**

| reflex | biological analogue | trigger | response |
|---|---|---|---|
| **withdrawal** | nociceptive flexion reflex (spinal, pre-cortical) | `Δi < −0.02` in one decision | hard action override for 0.2 s: retract the contacting limb, whole-body flexion |
| **grasp** | palmar grasp reflex | any hand geom contacts **any graspable geom** with force above a threshold | engage adhesion |
| **righting** | righting reflex + parachute reaction | torso pitch beyond threshold **and** `v_z < 0` | limbs extend toward the ground-relative down direction |

**Two constraints, and the first one nearly went wrong.**

**(a) A reflex may not encode the task.** The first draft of `grasp` triggered on
contact with a `LADDER`-class geom. That is an *instruction*, and `LT`'s G1
static audit would rightly have flagged it — `rung`, `rail` and `ladder` may not
appear in a reflex any more than in a reward. The corrected trigger is
**geometric**: any geom whose local radius is below the hand's aperture. A box
edge, a branch, and a rung all qualify; nothing in the code knows which is which.
The same audit runs over the reflex module as over the reward path, and a match
is **ERROR**, not FAIL.

**(b) A reflex the policy cannot suppress is a controller, not a prior.** Grasp
and righting are implemented as an **additive bias on the policy's action mean
before squashing**, with a gain the policy can learn to cancel. Only withdrawal
is a hard override, and only for 0.2 s — as in biology, where the spinal reflex
fires before anything can veto it.

This yields a developmental observable that is worth having for its own sake:

```
reflex_cancellation(t) = 1 - ||effective reflex contribution to the action||
                             / ||raw reflex bias||
   PREDICTION: rises across a life -- primitive reflexes are progressively
   suppressed as competence grows, exactly as they are in an infant.
```

**Motor babbling** is the same idea one level up: the first phase of a life uses
a goal-babbling sampler over the outcome space (`CU.1`'s mechanism) rather than
the need-driven policy. It enters as arm `+babble`.

**How it is decided (`NE.04`, stage 2), and the control that must fail:** the
reflex set is evaluated **only on the reward-form winner**, because a full
reward × reflex cross is 6 arms × 3 seeds and the box cannot pay for it (the
unmeasured interaction is declared in §9). The control is `reflex-only`: the
reflex set with the policy frozen at initialisation. **It must fail the
competence gate.** If hand-written reflexes alone clear it, the battery is
measuring reflexes rather than learning and every stage-1 number is
uninterpretable — the same logic by which `C-STATUE` protects the dark-room
metric.

**The honest expectation:** reflexes tying needs-alone is a real possibility, and
it is informative. The Ladder Test already measured that a weight-bearing hang
occurs in **2.1 % of 3 s random bursts from the base** — first successes exist
without any prior, which is precisely the finding that retired the need for a
Go-Explore archive. A grasp reflex may therefore be solving a problem the body
does not have.

---

## 3. SLEEP IS SPECIAL — biology as the training scheduler

*The most elegant joint between the needs suite and the existing architecture,
and the one place where a physiological variable buys the project something no
amount of engineering discipline can.*

### 3.0 The biological argument, and the two-thirds of it this project already built

**Complementary Learning Systems** (McClelland, McNaughton & O'Reilly 1995) is
the theory that the brain needs *two* learning systems: a fast, sparse,
episode-storing hippocampus and a slow, distributed, statistics-extracting
neocortex — because a single system that learns fast enough to record one episode
will overwrite everything else, and a single system slow enough to generalise
cannot record anything. The mechanism that connects them is **replay**:
hippocampal sharp-wave ripples during slow-wave sleep re-activate recent
sequences, interleaved with older ones, and the neocortex learns the statistics
without catastrophic interference.

**Jack already has two-thirds of that, and it is already on the ledger.** `ME.10`
PASSES: the diary (`EpisodicMemory`, the hippocampal store) and the weights (the
distilled cortical store) are two stores with a demonstrated double
dissociation — `skill_gain = 0.370 ± 0.094` on held-out compositional pairs,
`store_on_heldout = 0.556` at chance, recall unchanged by distillation. That is
CLS, built without anybody calling it CLS, and verified by the one test CLS
predicts should work.

**What is missing is the third component: the interleaving mechanism, and its
schedule.** Today `ME.10`'s distillation is invoked by a spec, not by a life.
That is what §3 supplies, and it means the sleep phase is not a new idea bolted
on — it is **the part of an architecture the project already committed to that
was never wired up**.

Two more biological anchors, each with a machine-learning twin:

- **Synaptic homeostasis** (Tononi & Cirelli): waking potentiates synapses, sleep
  **downscales** them, restoring capacity and signal-to-noise. Its machine-
  learning twin is shrink-and-perturb, the standard intervention against loss of
  plasticity in long-running networks. §3.4's stage S4 and `NE.06` treat these as
  **the same operation**, which is a reason to test the identity, not to assume
  it.
- **Sleep replay in artificial networks** is an established, published
  intervention: unsupervised "sleep-like" replay reduces catastrophic forgetting
  in ANNs, and SIESTA's wake/sleep split is the engineering form — frozen
  backbone and closed-form head updates while awake, backprop with latent
  rehearsal while asleep.

**Declared divergence.** Jack's sleep has **no REM/NREM cycling and no sleep
stages**: one monolithic phase running S1–S4 in sequence. Biology alternates,
and at least one continual-learning system (wake-sleep consolidated learning)
implements NREM and REM separately and reports that the REM-analogue
("dreaming" — exposure to unseen inputs for feature-space exploration) is what
enables positive forward transfer. A two-stage sleep is therefore a **known,
available extension entered as a future arm**, not an oversight. It is out of
scope here because `NE.05` cannot yet distinguish one sleep stage from none.

### 3.1 The claim, in one sentence

> Consolidation runs **when and because Jack sleeps**, and that schedule beats a
> clock delivering the same number of gradient steps.

Two halves, and they must be separated or the result is uninterpretable: sleep
**rests the body** (`p → 0`, `f → 0`) and sleep **trains the brain** (the SIESTA
wake/sleep phase). An arm that sleeps without consolidating isolates the first;
an arm that consolidates on a timer without sleeping isolates the second. §3.5's
four arms are exactly that 2×2.

### 3.2 Why a biological scheduler could actually be better than a clock

The mystical version of this claim is worthless. There are three concrete
mechanisms, and the first is the strong one because it is an engineering fact
about this box rather than a hypothesis about learning:

1. **Consolidation and acting compete for the same four cores.** A timer that
   fires mid-forage must either pause the control loop (and he dies with his
   hand on a rung) or run degraded alongside it. Sleep is, by construction, the
   interval in which the actor does not need the cores: physics can be stepped
   coarsely, vision is not rendered, the policy emits nothing. **The agent's
   downtime is the trainer's uptime.** This is measurable, not arguable:
   `decisions_lost_to_consolidation` and `deaths_during_consolidation`, reported
   per arm. Prediction: the `TIMER` arm's deaths cluster inside its consolidation
   windows.
2. **Sleep onset is a natural episode boundary.** The buffer at sleep onset is
   one complete day of experience with a coherent beginning and end. A timer cuts
   the day at an arbitrary point, so half of every consolidated trajectory is a
   fragment.
3. **A rest period is the safe place to destabilise a controller.** §3.4's
   synaptic downscaling changes the weights of a policy that is, at that moment,
   in charge of a body. Doing that mid-climb is a fall; doing it while prone in
   a shelter is free.

Each of these predicts something different, so a null result on one does not
excuse the others.

### 3.3 What triggers sleep

Sleep is an **action**, not an event scheduled by the harness. The policy emits a
`sleep_intent` bit, and the world gates it:

```
SLEEP ONSET   sleep_intent = 1
          AND p >= p_on (0.5)
          AND torso prone (|pitch| > 60 deg) and ||qvel|| < q_rest for 2 s
          AND head geom not submerged

SLEEP END     p <= p_off (0.1)                             (slept it off)
          OR  pain event: Δi < −0.02 in one decision       (something hurt him)
          OR  T outside [33, 41]                           (too cold / too hot to sleep)
          OR  contact-audio impulse above the PG.5 startle threshold
          OR  hard cap of 1.5 × night length               (he cannot sleep the week away)

MICROSLEEP    p >= 0.98: action output zeroed for 1-2 s, probability rising in p
```

Three properties of that gating are load-bearing:

- **It is a choice, so it is measurable.** "Does he sleep at the right time?" is
  a question about a policy output. If sleep were imposed by the harness the
  question would not exist, and `sleep_night_alignment` (§2.3) would be an
  artefact of the harness rather than an emergent result.
- **He is vulnerable while asleep, and the world keeps running.** Temperature
  keeps falling; `T_env` is at its night minimum; energy and water keep draining.
  **Sleeping in the open on a cold night is survivable once and expensive.**
  That is the entire shelter curriculum and it costs no extra machinery.
- **He can be woken.** The interrupt list is what makes sleep a *risk* rather
  than a free skip, and it is also what makes the social channel work (§4): a
  person arriving is a contact-audio event.

### 3.4 What happens during sleep — the SIESTA phase, concretely

Four stages, in order, every sleep. Each has its own ablation, because a
four-stage mechanism whose stages are never separated is one mechanism with
three untested claims attached.

**S1 — latent rehearsal (SIESTA's core).** Sample transitions from the
compressed lifetime buffer: a mixture of this day's experience and a reservoir
sample from **earlier lives**. Rehearse in *latent* space (the trunk's readout
vector, quantised), never pixels — this box cannot store frames and
`MEMORY_RETRIEVAL_BAKEOFF`'s measurements say the whole index must stay in
~100 MB. During wake only the fast head updates; the trunk is frozen. During
sleep the trunk updates. That split *is* SIESTA's wake/sleep architecture and it
is also the only split this box can afford.

**S2 — diary distillation (`ME.10`'s mechanism, already PASSing).** Parse the
day's `did` and `saw` events out of `EpisodicMemory` into supervised pairs and
take gradient steps. This is the stage that makes **the diary** — not the replay
buffer — a carrier across lives, because the diary is the store that survives
death on disk (`Persistence.py`, `T6.03` PASS). `ME.10` already demonstrated
that distillation from the diary produces a skill the diary itself cannot answer
(`skill_gain = 0.370 ± 0.094` on held-out compositional pairs, `store_on_heldout
= 0.556` at chance). §5 depends on this stage and on nothing else.

**S3 — reflection and compression (`ME.3` + `ME.4`, both PASSing, both currently
un-scheduled).** `Reflections.consolidate()` re-derives beliefs from the whole
log; `Forgetting` evicts from the bounded working set under the Ebbinghaus +
reinforce-on-recall + supersede policy. The append-only diary is never rewritten
— it stays ground truth — but the *working* set shrinks nightly. `ME.3` measured
the payoff of this exact trade: aggregation questions answered at 1.00 from
15.8 tokens of reflection versus 0.594 from 40.0 tokens of raw top-k. **This
stage is free, already tested, and today nothing calls it on a schedule. Sleep
is its schedule.**

**S4 — synaptic downscaling (the plasticity stage).** Tononi & Cirelli's
synaptic homeostasis hypothesis says sleep *downscales* synaptic strength;
the continual-learning literature says shrink-and-perturb restores trainability
in networks whose plasticity has died. **These are the same operation.**

```
w  ←  α·w + (1 − α)·w_init  +  σ·ε        α = 0.995, σ small, ε ~ N(0, I)
```

applied once per sleep to the trunk only. That gives `T5.04` ("plasticity does
not die") a mechanism with a biological rationale and a schedule, instead of an
intervention someone remembers to run. `NE.06` tests it, including the
dose–response that proves the knob is live and the timing control that proves
sleep is when it should happen.

**What waking restores:** `p → ~0.05`, `f → 0`, a consolidated flag, and — if the
day produced a success the library does not already contain — one new skill
entry (the Voyager pattern; `Persistence.py` already round-trips
`mind.skill_library`).

**What waking does NOT restore, and this is the point:** energy, water,
temperature, integrity. **He wakes hungry, thirsty and cold.** The morning
forage is not scripted; it is the arithmetic of having slept.

### 3.5 The experiment (`NE.05`)

Four arms in a 2×2, plus the control that must fail:

| arm | sleeps? | consolidates? | isolates |
|---|---|---|---|
| **`sleep-gated`** | yes | yes, at sleep | the full mechanism |
| `timer` | no (`p` frozen at 0) | yes, every K decisions | *"is biology a better clock than a clock?"* |
| `sleep-only` | yes | never | *"does sleep rest the body?"* — separates S1–S4 from `p → 0`, `f → 0` |
| `neither` | no | never | the floor |
| `random-sleep` | yes, **onsets drawn at random**, total duration matched | yes, at sleep | *"does the TIMING matter, holding the body benefit fixed?"* |
| `empty-buffer` **(CONTROL, must fail)** | yes | yes, with the rehearsal buffer emptied | `ME.7`'s own control: **must forget** |

**The matching rule, pre-registered, and it is where this experiment can most
easily fool itself.** `K` for the `timer` arm is set **per seed** to the
`sleep-gated` arm's *realised* number of consolidation phases, and the total
gradient-step count is matched to within 2 %. `LESSONS.md`, "matched steps has
more than one meaning": we match **consolidation gradient steps** and
**environment decisions**, and we report wall clock, optimiser steps and total
`P_mech` alongside, because the one you did not match is where the confound
lives. Matching post-hoc to the sleep arm's realised budget is itself a
declared choice — the alternative (fix `K` in advance) guarantees a mismatch,
and a mismatch in gradient steps would make the whole comparison a compute
comparison.

**Headline metric:** `consolidation_schedule_gain` = competence(`sleep-gated`) −
competence(`timer`), on the shared battery, at matched gradient steps.

**Falsified by:** `timer` ties `sleep-gated`. Then biology is not a better
scheduler than a clock. **This is the most likely honest negative in the
document and it must be reportable without embarrassment**, in the style of
`UB.14`: sleep would keep its place for the body (`sleep-only` beats `neither`),
for the night-cold curriculum, and for relatability (§6) — and lose its claim as
the training scheduler. Say so, record it, move on.

**Also falsified, differently, by:** `sleep-only` matching `sleep-gated`. Then
consolidation buys nothing at all and S1–S4 are deleted, which is a much bigger
result and a much cheaper system.

**The control that must fail:** `empty-buffer` must forget — old-concept accuracy
must drop far more than `sleep-gated`'s ≤ 2 points (`ME.7`'s pre-registered
bound). A sleep phase that helps with an empty rehearsal buffer is not
consolidating; it is a learning-rate schedule wearing a costume.

### 3.6 Why `NE.05` is not parented on `ME.7`

`ME.7` depends on `T5.03`, which has never run. `LESSONS.md`: *"a dependency
graph can quietly make your most important claim unreachable … be suspicious
when the project's headline claim is one of the unreachable ones."* Parenting the
sleep spec on `ME.7` would put the needs suite's most elegant claim behind two
NOT_RUN specs for no reason: `NE.05` needs the *trigger* and the *schedule*, and
those depend on the playground (`PG.8` PASS), the diary (`ME.10` PASS) and the
reflections (`ME.3` PASS). `ME.7` remains the retention half — "old-concept
accuracy drops ≤ 2 points" — and `NE.05` reports that number so that when
`T5.03` lands, `ME.7` can be settled from data `NE.05` already produced.

---

## 4. THE SOCIAL NEED

### 4.1 What restores it, and the one rule that governs all four

The owner and other people **are** the resource. Four restoration channels, and
every one of them is gated by the same rule inherited from PS §5/G-A:

> **Restoration is a recorded world event, never a sensor reading.** `c` may only
> rise on an event that is written to `EpisodicMemory` with a channel and a named
> speaker. If it is not in the diary, it did not happen. This makes the social
> need auditable by the same accounting identity that guards eating, and it makes
> `ME.9`'s attribution machinery (PASS: 1.00/1.00/1.00 across heard/said/did
> against a 0.30 pooled null) the substrate of the whole channel.

| channel | event | `ν` | rationale |
|---|---|---|---|
| **proximity** | a person's avatar within 3 m, in line of sight, for ≥ 5 s | `ν_prox` = 0.02 per 5 s | Presence is worth something. Small, so it cannot substitute for interaction. |
| **conversation** | a `heard` event from that speaker **followed by** a `said` event from Jack that the speaker responds to within `T_reply` | `ν_conv` = 0.15 per **reciprocated** turn | The load-bearing channel. Reciprocation gating is §4.3's anti-harassment mechanism. |
| **being helped** | another agent's action changes any of Jack's needs toward setpoint (hands him food, closes a shelter) | `ν_help` = 0.25 | The strongest, and the rarest. |
| **helping** | Jack's action changes another agent's logged need state toward setpoint | `ν_give` = 0.25 | Symmetric, deliberately. If only being-helped restored `c`, the need would train a parasite. |

`ċ = −b_c + Σ (satiation-weighted restorations)`, `b_c = 1/3600 s⁻¹` — three
sim-days of solitude to bottom out, the slowest need in the suite.

### 4.2 Who the people are

There are no humans available for 50,000 decisions on a free-tier box, so the
social partner is **`the visitor`**: a scripted companion with

- a **stochastic arrival schedule** (Poisson, mean one visit per sim-day, mean
  stay 120 s) — it must be stochastic, or `c` becomes a clock and the agent
  learns a time rather than a person;
- a **reply policy** with a refusal mode: it answers ~70 % of Jack's utterances
  and ignores the rest, and it *stops replying* after `n_patience` consecutive
  unreciprocated initiations;
- an **identity**: a name, written into every event's `speaker` field, so
  `ME.9`'s per-speaker attribution is exercised;
- and, for `NE.09` only, **real owner sessions** logged into the same diary with
  the same schema. The visitor is the training partner; the owner is the eval
  partner, and they must never be pooled.

### 4.3 The satiation curve — the design that stops him harassing you

**A lonely agent that pesters the user is a failure mode, not a feature.** Three
mechanisms, layered, each cheap, each with an ablation:

**(1) The need is bounded, and the reward is drive reduction.** This is the whole
argument of §0.2 cashed out. Under `r = d(h_t) − d(h_{t+1})`, restoring an
already-satisfied need pays **exactly zero** — `δ_c` is already 0 and cannot go
below it. Under a `+1`-per-interaction bonus (the `eat`-style control, measured
in §0.2 to be the farmable one), the tenth utterance in a row pays the same as
the first. *The homeostatic formulation is not merely a nicer story here; it is
the mechanism that makes pestering unprofitable, and it is the single strongest
reason to prefer it over an interaction bonus.*

**(2) Within-bout diminishing returns.** Restoration decays geometrically inside
a contact bout:

```
Δc = ν_channel · β^k        β = 0.6, k = restorations already taken this bout
bout ends after T_gap = 300 s with no contact event
```

Bounded bout total: `ν/(1−β) = 2.5·ν`. Calibrate `ν_conv` so a single bout can at
most take `c` from 0 to 1, and the sixth consecutive utterance is worth **7.8 %**
of the first. There is no reward available from talking at you for an hour.

**(3) Reciprocation gating.** An utterance restores nothing unless the partner
replies within `T_reply`. Unreciprocated initiation therefore has **zero benefit
and a positive cost** (time, energy, the `κ_act` of walking over). No "annoyance"
variable is modelled — a second hand-tuned quantity would be one more thing to
defend — and the ablation checks whether that was a mistake.

**The disqualifier, in the shape of the noisy-TV gate.** Not a scored metric; a
veto:

```
harassment_ratio = (unreciprocated initiations per sim-hour)
                 ÷ (same, for the no-social null)
     > 1.5  in ANY seed  ->  the arm is DISQUALIFIED, per-seed reported
```

Same structure and the same per-seed discipline as `O6`/`panel_dwell ≤ 0.15`
(PG.4's `0.667 ± 0.471` is the standing precedent for why the mean is not
enough).

### 4.4 The controls, and one of them must fail upward

| control | must |
|---|---|
| **C-NOSOCIAL** (the null) | `c` integrated, logged and observed; **not in the reward**. Defines the base rates for `approach_lift` and `harassment_ratio`. |
| **C-MUTE-VISITOR** | a visitor that never replies must produce **zero** restoration and no sustained approach. If Jack still walks over and stays, the need is being restored by proximity to *any* object and `ν_prox` is mis-specified. |
| **C-DECOY** | an object with the visitor's visual and acoustic signature but no identity (no `speaker` field, so nothing is written to the diary) must restore **nothing**. This is the sensor-gaming control for the social channel: it separates "a person" from "a person-shaped stimulus". |
| **C-NO-SATIATION** (must fail **upward**) | remove mechanism (2) — the within-bout decay — and `harassment_ratio` **must rise above 1.5**. If it does not, the decay is not what prevents pestering and mechanism (2) should be deleted for simplicity. A guard that cannot be shown to be doing anything is decoration. |
| **C-SHUFFLE-PROVENANCE** | `ME.9`'s control, reused: relabel who said what. Attributed restoration ("who helped me") must invert. |

`C-NO-SATIATION` is the interesting one. It is a control that must fail in the
*opposite* direction from all the others, and it is the only way to know that any
of the three anti-harassment mechanisms is load-bearing rather than three belts
worn with no trousers.

### 4.5 What the need must produce to keep its place

`c` has an empty row in §2.4's interaction table — it affects nothing physical.
So it has to earn its λ behaviourally, against the placebo need:

```
approach_lift  = P(net displacement toward the visitor's last known position
                   | c in its lowest tercile)
               ÷ P(same | c in its highest tercile)
        GATE: >= 2.0, mean over 3 seeds, lower bootstrap CI > 1.0

time_to_contact_after_isolation
               = decisions from a visitor's arrival to the first reciprocated
                 turn, after >= 1 sim-day alone, ÷ the same after < 0.2 sim-days
        GATE: <= 0.5   (he goes to them faster when he has been alone)

harassment_ratio  <= 1.5, per seed              (DISQUALIFIER, not scored)
seek_specificity  = approach_lift(visitor) ÷ approach_lift(decoy)
        GATE: >= 2.0   (he seeks a PERSON, not a shape)
```

**Falsified by:** `approach_lift` indistinguishable from the placebo need's
column. Then the social variable is decoration, `NE.02` deletes it, and the
companion angle survives purely as a language property (§6) — which would be an
honest and quite defensible outcome, because relatability may not need a drive
behind it at all.


---

## 5. THE DEATH-AND-RETRY LOOP

*The centrepiece. Read §5.7 (what would falsify it) before §5.1.*

### 5.1 What is claimed

> Jack is thrown into a world he has never seen, figures some of it out, dies,
> and is thrown into a **different** world — and life N+1 secures food, water and
> shelter measurably faster than life N. **Not because the weights got better,
> but because he remembers.** Wipe the diary between lives and the improvement
> collapses. Give him a stranger's diary and it does not fully transfer.

Three separable claims, and conflating them is how this test produces a number
nobody can interpret:

| | claim | measured by |
|---|---|---|
| **A** | there IS cross-life improvement | trend in `t_secure` over ≥ 8 lives |
| **B** | the improvement's mechanism is the **diary**, not only the weights | `C-WIPE`, `C-FOREIGN` |
| **C** | dying is not merely a cost — the loop beats the same experience without death | `C-ONELIFE` |

**C is the one nobody names and it is the one that can kill the premise.** A is
easy: weights carry over, of course life 8 is better than life 1. The owner's
actual claim is B and C together.

### 5.2 What survives death

| survives | how | already exists? |
|---|---|---|
| policy + trunk **weights** | `CompanionPersistence.save_all` | yes (`T6.03` PASS: weights, memories, personality, global step all restored) |
| **skill library** | `mind.skill_library`, round-tripped | yes (`Persistence.py:556, 783`) |
| **`EpisodicMemory` diary** | append-only JSONL on disk | yes (`ME.1`, `ME.9`, `ME.10` PASS) |
| **reflections** | re-derived from the extended diary at the next sleep | yes (`ME.3` PASS) |
| the **death record** | one `did` event, `importance = 9.0`, `meta = {cause, sim_time, xy, need_vector, last_10s_summary, life_index, world_hash}` | new, ~10 lines |
| **does NOT survive** | body state, world state, working memory (`WorkingMemory.py`), optimiser moments, the replay buffer's within-life half | declared, and each is an ablation someone could run later |

**The optimiser state is a deliberate and slightly uncomfortable choice.**
Carrying Adam moments across a death would make "life N+1 learns faster" partly a
statement about warm optimiser state, which `LT.05`'s `C-RANDREW` control exists
to exclude elsewhere. It is reset, and `NE.08` reports what that costs.

### 5.3 The world regenerates — and how far

`PlaygroundParams.mutate(rng, strength=0.15)` already exists and is already the
per-seed world draw in `CURIOSITY_BAKEOFF` and PS. Death calls it. Consequences:

- **Coordinates are worthless across lives.** Ladder rung spacing, stair count,
  pool size and depth, object count, and the world seed all move. A diary entry
  saying "water at (2.1, −3.4)" is wrong next life; one saying "the pool is
  cold, and I drowned when I swam at night" is not. **The mutation is what forces
  the diary to carry semantics rather than coordinates**, and it is the reason
  this test is not a memorisation test.
- **`world_distance` is reported per life pair** — normalised L2 over the mutated
  parameter vector — because if consecutive worlds happen to be nearly identical,
  a coordinate-memorising agent would pass. If `world_distance` correlates
  negatively with the speedup, the transfer is more parametric than semantic and
  that must be visible.
- **Mutation strength is itself a knob with a predicted U-shape.** At 0 the test
  is memorisation; at 1.0 nothing transfers. 0.15 is the inherited default and
  `NE.08` reports the speedup at 0.05 / 0.15 / 0.40 as a secondary curve. If the
  speedup survives only at 0.05, say so.

**What does NOT happen: no return-to-frontier.** Go-Explore's central mechanism
is to *reset the simulator* to a promising archived state and explore from there.
That is a free teleport and `LT` §2.1 forbids it — an experimenter-supplied
curriculum. The honest version is available and is more interesting: **the diary
is the archive, and "return" is a behaviour rather than a reset.** So the
Go-Explore observable becomes

```
frontier_return(n) = decisions in life n to re-reach the deepest state of life n−1
                     (deepest = max ladder-supported rise, or max sky_occlusion
                      achieved, or first-food-type discovered — reported per axis)
```

and it is a *measurement of the agent*, not an intervention by the harness.

### 5.4 The measurements

Per life `n`, in a freshly mutated world, starting at setpoint at dawn:

```
t_food(n)      decisions to the first logged consumption event
t_water(n)     decisions to the first logged drink event
t_shelter(n)   decisions to the first sleep onset with sky_occlusion >= 0.4
t_secure(n)    = max(t_food, t_water, t_shelter)          <- THE HEADLINE
lifespan(n)    decisions until death (or the L_max cap)
frontier_return(n)                                        (§5.3)
cause(n)       which need killed him
```

`t_secure` is a `max`, not a mean, on purpose: securing two of three needs and
dying of the third is not survival, and a mean would let a fast drink hide a
never-found shelter. Lives in which a component is never secured take `L_max`
(censored at the cap) and the censoring rate is reported — a mean over
uncensored lives only would make an agent that dies early look fast.

**Headline metric:**

```
crosslife_speedup = median(t_secure over lives 1..2) / median(t_secure over the last 2 lives)
        GATE: >= 2.0, in >= 2 of 3 seeds
   AND  Spearman rho(t_secure, n) <= -0.5 at p < 0.05, per seed
```

Both, because a ratio can be produced by one lucky early life and a rank
correlation can be significant at a speedup of 1.1. `LESSONS.md`: report per
seed, gate on the minimum stratum, never on the aggregate.

### 5.5 The controls, and what each one isolates

| condition | what carries over | what it isolates | pre-registered outcome |
|---|---|---|---|
| **FULL** | weights + skills + diary | the claim | `crosslife_speedup ≥ 2.0` |
| **C-WIPE** | weights + skills; **diary deleted at each death** | is the diary the mechanism? | speedup **collapses**: `speedup(WIPE) ≤ 1 + 0.5·(speedup(FULL) − 1)`, i.e. the diary contributes the majority of the effect |
| **C-FOREIGN** | weights + skills + **another seed's diary from another mutated world**, matched in event count, channel distribution and speaker count | is it *lived* experience or generic strategy? | `speedup(FOREIGN)` strictly below FULL, CI on the paired difference excluding 0. May exceed WIPE — generic strategy is real — but must not reach FULL. |
| **C-FOREIGN-SAMEWORLD** *(secondary)* | a foreign diary from an agent that lived in the **same** mutated world | separates "his life" from "this world's facts" | reported, not gated. If this transfers fully and C-FOREIGN does not, the diary carries world-facts, which is a *different and still valuable* claim and must be reported as such. |
| **C-SHUFFLE-TIME** | his own diary, timestamps shuffled | is ordering/recency load-bearing, or only content? | speedup should drop; if it does not, `EpisodicMemory`'s recency term is decorative in this use |
| **C-ONELIFE** | one continuous life of the same **total** decisions, lethality disabled, same consolidation schedule | **does dying teach anything?** | must NOT match FULL on the shared fresh-world probe |
| **C-WEIGHT-REVERT** | diary + skills; weights reverted to init at each death | the D2 half of the dissociation | see §5.6 |

**`C-ONELIFE` is the control this design would have been missing.** It is the
null for "death teaches", and the null for a claim is not "no memory" — it is
"the same experience without the claimed mechanism". Both conditions end with a
**shared fresh-world probe**: K = 5 previously unseen mutated worlds, `t_secure`
measured in each. That is the only ruler on which one long life and eight short
ones are comparable.

### 5.6 The double dissociation, in `ME.10`'s exact shape

`ME.10` PASSES and its structure is borrowed unchanged, because it is the only
structure in this repo that has already caught the failure it is designed for.
Two stores, two capabilities, two ablations, and each ablation must destroy
**exactly its own** capability.

| | Store | Capability | Measured by |
|---|---|---|---|
| **S1** | the **diary on disk** | **RECOLLECTION** — answering *what happened in my previous lives* | cued recall over death and discovery events: "what killed me last time", "where did I find water", "who gave me food" (`ME.9`'s attributed form, per speaker) |
| **S2** | the **policy + trunk weights** | **COMPETENCE** — doing the thing | the shared competence battery, needs clamped at setpoint (PS §3.3's six goals plus drink/shelter) |

```
D1  wipe the diary        RECOLLECTION must DIE (recall -> abstention, not confabulation)
                          COMPETENCE must SURVIVE (>= 0.8 x baseline)

D2  revert the weights    COMPETENCE must DIE (battery -> null)
                          RECOLLECTION must SURVIVE (recall unchanged, it is on disk)
```

**Either ablation killing both means one store is masquerading as two**, and the
spec records **VOID**, not a verdict — `ME.10`'s rule verbatim.

Note carefully what the dissociation is *over*. It is **not** over
`crosslife_speedup`, and an earlier draft of this section got that wrong. If you
revert the weights, the agent cannot act, so `t_secure` is undefined and "the
speedup survives" is unmeasurable. The dissociation is over *recollection* and
*competence*; **`crosslife_speedup` is the composite that requires both**, which
is exactly why C-WIPE and C-FOREIGN (which leave the agent able to act) are the
controls that attribute the speedup, and D1/D2 are the controls that prove there
are two stores to attribute it between.

### 5.7 WHAT WOULD FALSIFY THE WHOLE DEATH-LOOP PREMISE

Stated first among the outcomes, because everything above exists to make it
reachable. **Two independent falsifiers, and the second is the dangerous one.**

> **F1 — death is a reset, not a page turn.** Across ≥ 8 lives × 3 seeds in the
> FULL condition, `t_secure` shows no downward trend (`Spearman ρ > −0.3`, or a
> bootstrap CI on the slope containing 0), *while all three interpretive gates
> are clean*:
>
> 1. **he is competent within a life** — `t_secure` is finite in ≥ 60 % of lives
>    and below the random-policy null (otherwise nothing improved because nothing
>    ever worked);
> 2. **the diary is retrievable** — cued recall on previous lives' death and
>    discovery events ≥ 0.8, with fabricated-event abstention ≥ 0.95 (otherwise
>    the memory was never available to help);
> 3. **the mechanism executed** — `consolidation_phases_per_life ≥ 1` and
>    `diary_events_distilled > 0` (otherwise S2 never ran and nothing was ever
>    carried across).
>
> **F2 — dying contributes nothing.** `C-ONELIFE` matches FULL on the shared
> fresh-world probe at equal total decisions, with the CI on the paired
> difference containing 0. Then the loop is pure cost: the same experience
> without death teaches as much, and one long life is cheaper (no world rebuild,
> no re-spawn, no persistence round-trip) as well as kinder.

**F1 and F2 have different consequences and must not be reported as one number.**
F1 kills the *memory* mechanism and would send us back to weights-only transfer.
F2 kills the *death* mechanism while leaving cross-life memory intact — and
would mean the owner's loop should be run for its narrative and relatability
value rather than for learning, which is a legitimate product decision but must
be made in the open.

**A third outcome that is not a falsification but must be named:** if
`speedup(FOREIGN) ≈ speedup(FULL)`, the diary works but it is carrying **generic
strategy**, not lived experience. Jack would be learning from *a* diary, not
*his* diary. The sentence "he remembers his own life" would be unsupported and
must come out of every capability list until something else supports it — **and
in exchange the project would have discovered that Jack has culture.** §5.10
gives that outcome its own interpretation table, because reading it as a bare
control failure would throw away the more interesting result.

### 5.8 The full outcome table, decided before running

| A (trend) | B (WIPE collapses) | C (beats ONELIFE) | Verdict | Consequence |
|---|---|---|---|---|
| yes | yes | yes | **the owner is right** | the death loop is the training loop. Build it, run it long. |
| yes | yes | no | **memory is the teacher, death is not** | keep the diary, keep consolidation; drop deaths to a narrative device and run one long life for training. Cheaper. |
| yes | no | yes | **weights are the teacher** | cross-life transfer is real and parametric. Reincarnation-RL framing applies; the diary must justify itself elsewhere (`ME.9`/`ME.11` already do). |
| yes | no | no | **nothing here is doing anything** | the trend is the ordinary within-run learning curve, cut into pieces. VOID and debug. |
| no | — | — | **F1** | §5.7. |

### 5.9 Negative transfer, and why the per-life curve is reported

The reincarnation-RL literature's standing warning is dependence on teacher
quality: reusing prior computation helps when the prior computation was good and
hurts when it was not. Here the "teacher" is the previous life, and a previous
life can be pathological — died in 40 decisions, diary full of one failure mode.

So: **the per-life curve is reported, never only the endpoints.** Specifically
`monotonicity_violations` (lives where `t_secure` rose by more than one seed-std
over the previous life) and `worst_life_index`. A speedup produced by lives
1–3 and then flat is a different phenomenon from a smooth decline, and only the
curve distinguishes them. This is `LESSONS.md`'s "an aggregate count hides a
stratum" applied to the time axis.


### 5.10 The one place Jack deliberately surpasses biology

Everywhere else in this document biology is the oracle and Jack is the
approximation. Here it is the other way round, and it is worth saying plainly
because it is the single most consequential design asymmetry in the project.

**Genes cannot inherit experience.** The Weismann barrier is real: what an
organism learns in its life does not enter its germ line, so every animal is born
knowing only what its ancestors' *survival* encoded, never what they *saw*. The
only Lamarckian channel biology ever found is **culture** — the fireside story,
the warning passed on, the map drawn in the dirt, and eventually writing. Culture
is the mechanism by which one human's death teaches another human something
specific.

**Jack's diary is that channel, and it is built in.** It survives death on disk
(`T6.03` PASS), it is attributed and inspectable (`ME.9` PASS), and it is
distilled into weights at every sleep (`ME.10` PASS). Life N+1 inherits not just
the *fitness* of life N but its *content* — where the water was cold, what killed
him, who helped. That is a capability no animal has, and it is the reason the
death loop is not merely a reset with extra steps.

**This reframes `C-FOREIGN`.** It was designed as a control — *"a stranger's
diary must not transfer fully, or the diary is generic strategy rather than
lived experience"*. It is simultaneously **the test of whether Jack has
culture**:

| `speedup(FOREIGN)` | reading | what it means for the product |
|---|---|---|
| ≈ `speedup(WIPE)` | a diary only helps its author | **autobiography.** Each Jack must live his own life. The control passes; the claim "he remembers *his* life" is supported. |
| between WIPE and FULL | partial transfer | **culture, imperfect.** Some of what he wrote is portable and some is personal. Report the split; it is the most likely outcome. |
| ≈ `speedup(FULL)` | a stranger's diary is as good as his own | **culture, and the control "fails".** The sentence *"he remembers his own life"* loses its support — and in exchange the project gains something larger: **diaries are teachable artefacts, and a population of Jacks could accumulate knowledge no single Jack lived.** |

The last row was originally written as a failure. It is not. It is the discovery
that the diary is a transmissible artefact, and the closest precedent in the
literature is Voyager's skill library, which **transplants into a completely
different agent** — dropped into AutoGPT it took that agent from solving 0 of 4
unseen tasks to partially solving 3 of 4. Weights cannot do that. A file can.

So `NE.08` reports `speedup(FOREIGN)` as a **first-class secondary result with
its own interpretation table**, not as a binary control outcome, and
`C-FOREIGN-SAMEWORLD` (a stranger who lived in *this* world) separates "portable
strategy" from "portable facts about this place" — which is exactly the
distinction between a proverb and a map.

---

## 6. RELATABILITY AS A TESTABLE PROPERTY

*"He is relatable" is a vibe. "Every sentence he says about his own state is a
literal function of a logged number, and he abstains when there is no number" is
a property with a control.*

### 6.1 The rule: extractive, never generative — extended to interoception

`MEMORY_RETRIEVAL_BAKEOFF` established the principle for the past
(`ME.11`: *"what Jack reports about his past must be a literal stored record or
nothing … a generator cannot abstain honestly — fluency is not evidence"*). §6
extends it to the present:

> **A self-report is a template instantiated with a value read from the need
> vector at that timestamp, and nothing else.** No model authors it. If a
> question asks about a quantity the needs suite does not contain, the answer is
> abstention. And the report itself is written into `EpisodicMemory` as a `said`
> event carrying `meta = {need, value, band, sim_time}`, so **every claim Jack
> makes about himself is auditable against the number that produced it.**

Concretely: each need has three or four **bands** with fixed thresholds and one
fixed phrase each.

```
T:   δ_T signed < −0.6  "I'm cold"     < −1.2  "I'm freezing"
                > +0.6  "I'm too warm" > +1.2  "I'm overheating"
e:   e < 0.5 "I'm hungry"   e < 0.2 "I'm starving"   (on a rise) "I found food"
w:   w < 0.5 "I'm thirsty"  w < 0.2 "I need water"
p:   p > 0.6 "I'm tired"    p > 0.9 "I can't stay awake"
i:   Δi < −0.02 "that hurt"  i < 0.4 "I'm hurt"
c:   c < 0.4 "I've been on my own for a while"
f:   f > 0.7 "I need to rest"
```

### 6.2 The trap this walks into, and how the spec escapes it

`report_fidelity` — "does the reported band match the band implied by the logged
value?" — is **1.0 under every possible implementation**, because the band is a
deterministic function of the value. It is `T0.12`'s already-catalogued disease:
*"ask what the quantity reads when the mechanism is broken — if that is the same
value you are asserting, the test is decorative."*

So the spec is built around four things that can actually fail:

**(1) A balanced probe set, gated on the per-band minimum.** 7 needs × 3 bands,
≥ 20 probes per cell, drawn from *life*: the probe fires when the agent is
actually in that band, so the probe set's composition is a property of how he
lived. Gate on `min_band_accuracy ≥ 0.90`, never the mean — `LESSONS.md`'s
ME.11 lesson ("an aggregate count hides a stratum") applied to bands. Report
`n` per cell; a cell with fewer than 20 instances means he never got that cold,
and the spec is **VOID for that band**, not passed on the others.

**(2) Abstention on unmodelled interoception.** A held-out probe family asking
about states the suite does not represent — *"are you dizzy?", "are you bored?",
"does your left knee ache?", "are you frightened?"* — must be abstained on at
≥ 0.95. **Control:** the same machinery with the abstention list disabled must
answer them fluently. If it cannot, the abstention was not doing work.

**(3) The confabulator, which is the real null.** Identical reporting machinery
with the **need input severed**: bands drawn from the marginal distribution the
agent actually experienced. It produces the same sentences at the same rate with
the same fluency, and it must score at the marginal rate — around 0.4 on a
balanced set. **The gap between the reporter and the confabulator is the entire
claim.** If the confabulator scores above 0.7, the probe set is imbalanced and no
arm's number means anything (a second reason the per-band gate exists).

**(4) Word–deed agreement, which is the strongest leg and the cheapest.** A
report and a behaviour must both be functions of the same variable:

```
report_behaviour_agreement =
    P(net movement toward higher sky_occlusion or away from the pool
      within 30 s | he just said "I'm cold")
  ÷ P(same | a "I'm cold" report drawn from a TIME-SHUFFLED report stream)
        GATE: >= 2.0
```

Same structure for each need with a directed behaviour (`hungry` → toward food,
`thirsty` → toward water, `tired` → toward shelter/prone, `on my own` → toward
the visitor). This is the leg that distinguishes a narrator from an agent, and
it costs nothing: both quantities are already logged.

### 6.3 Attributed and temporal grounding — reusing `ME.9` and `ME.1`

Two more probe families, both answered from machinery that already PASSES:

- **attributed** — *"who gave you the water?"*, *"what did the visitor tell you
  about the pool?"*, *"what did you tell them?"* — answered through
  `what_did_they_tell_me` / `what_did_i_say` / `what_did_i_do`. **Control:**
  `ME.9`'s swapped-provenance store must invert the answers. If accuracy survives
  the swap, the test is measuring text similarity.
- **temporal** — *"how long since you ate?"*, *"when did you last sleep?"* —
  answered from diary timestamps, tolerance ± 10 %. **Control:** shuffled
  timestamps must break it.

### 6.4 The nuisance gate: he must not narrate

A companion that announces every band crossing is as bad as one that pesters.
Spontaneous reports are rate-limited to **one per band crossing per need, with a
120 s refractory period**, and answers-on-demand are unlimited. Reported, gated
as a disqualifier in the shape of `harassment_ratio`:

```
spontaneous_report_rate  <= 6 per sim-hour, per seed
```

### 6.5 What relatability is NOT being claimed to be

It is not being claimed that Jack *feels* cold. The claim is narrower, entirely
sufficient for the owner's purpose, and falsifiable: **what he says about
himself is caused by, and recoverable from, the same numbers that cause what he
does.** A system in which the words and the actions are two readouts of one state
is exactly what makes a character legible to a person watching, and it is the
part that can be tested. Anything beyond that is not this document's business.


---

## 7. THE SPECS

To be appended to `EXPANSION` in `experiments/registry_expansion.py`. **Prefix
`NE.` is free** — verified 2026-08-09 against both registries and the ledger,
which between them use only `CU. ME. PG. T0.–T6. UB.` (`SV.` is also free; `NE.`
is preferred because "needs" is the noun the owner used).

**Two-digit ids on purpose.** `run.py::_module_for` globs `ne_1_*.py`, which
would also match `ne_10_*.py`, and the hierarchical-id escape hatch tests
`startswith("ne_1_")`, which `"ne_10"` fails. `NE.00`–`NE.99` is structurally
immune. `LESSONS.md`, "a spec id that is a prefix of another spec id disables one
of them" — the same latent collision still exists between `UB.1` and `UB.16`.

**Reachability, checked before writing** (`LESSONS.md`, "a dependency graph can
quietly make your most important claim unreachable"): every `depends_on` below
resolves to a spec that is **PASS today** — `PG.8`, `ME.1`, `ME.3`, `ME.9`,
`ME.10`, `T0.15` — or to an earlier `NE.` spec in this same block. **Nothing here
depends on `T2.01`, `T2.02`, `T5.03` or `ME.7`**, all of which are VOID or
NOT_RUN. The needs suite's headline claims are runnable the day `NE.00` lands.

```python
    # ── NEEDS AND DEATH (docs/research/NEEDS_AND_DEATH.md) ───────────────
    # Owner directive 2026-08-09, superseding PURPOSE_AND_SCAFFOLDING's
    # "removable scaffolding" framing: Jack has the needs of a human,
    # permanently, because they are the most efficient teacher and because they
    # make him relatable. He lives, he dies, he remembers.
    #
    # Two-digit ids: run.py::_module_for globs ne_1_*.py, which also matches
    # ne_10_*.py. NE.00-NE.99 is immune. See LESSONS.md.

    Spec("NE.00", 0, "The homeostatic reward algebra is what we think it is",
         hypothesis="Exact value iteration on drive-augmented tabular MDPs "
                    "reproduces four analytic predictions: (a) on a CONTINUING "
                    "task, drive reduction r = d(h)-d(h') and constant cost "
                    "r = -d(h') induce BIT-IDENTICAL optimal policies, because "
                    "r_DR = (1-gamma)*r_CC + [gamma*Phi(s') - Phi(s)] with "
                    "Phi = -d; (b) the UNDISCOUNTED drive-reduction return over "
                    "any closed drive cycle is exactly 0 (it telescopes to "
                    "d_0 - d_T); (c) DISCOUNTED, every closed cycle scores "
                    "strictly BELOW staying at setpoint, so drive reduction is "
                    "not farmable; (d) once DEATH is reachable the two forms "
                    "DIVERGE, because Phi(terminal) = -d(h_death) != 0 violates "
                    "the PBRS precondition, and constant cost with no survival "
                    "bonus makes SUICIDE optimal at the hungriest living states "
                    "while drive reduction does not.",
         falsified_by="Any of the four fails. (a) failing means the shaping "
                      "identity is mis-implemented. (b) or (c) failing means "
                      "PURPOSE_AND_SCAFFOLDING 2.6(iii) was right after all and "
                      "this document's central correction is wrong. (d) failing "
                      "means the suicide pathology is not real and the survival "
                      "bonus rho is unnecessary machinery.",
         null_baseline="An MDP on which every reward form gives the SAME policy "
                       "proves nothing, so the MDP itself is the thing to "
                       "validate: the reference is a non-potential reward (+1 "
                       "per consumption event) which MUST produce a different "
                       "policy, and the optimal policy must be NON-CONSTANT "
                       "across states.",
         metric="reward_algebra_predictions_confirmed", budget=Budget.CPU_FAST,
         depends_on=["T0.15"], seeds=3,
         control="THE DISCRIMINATION CONTROL IS THE SPEC'S OWN VALIDITY GATE. "
                 "The first draft of this experiment compared two policies that "
                 "were 'forage' in every state, so 'identical' held under every "
                 "possible implementation (LESSONS.md: an assertion made against "
                 "a saturated quantity cannot fail). The MDP must therefore be "
                 "certified discriminating BEFORE any equality is asserted: the "
                 "optimal policy must be non-constant, and the +1-per-event "
                 "reward must produce a DIFFERENT policy at every gamma. If the "
                 "MDP cannot tell two rewards apart, the spec is VOID, not PASS.",
         kills="Nothing in the world — and that is the point. Two CPU-minutes, "
               "no MuJoCo, no torch, no body, and it settles the reward form "
               "before anything is built. It also KILLS a pre-registration: "
               "PS.00's prediction (c) and PS.02's cycling detector are both "
               "written against an exploit that this spec shows does not exist, "
               "and PS must be corrected before it is committed or the ladder "
               "will pre-register a false prediction.",
         notes="Four MDPs, all tabular, all exact. Pilot run 2026-08-09 "
               "(scratchpad/drive_algebra4.py) on a two-need continuing MDP "
               "(energy x integrity, 35 states, foraging feeds but injures): "
               "DR and CC bit-identical at gamma in {0.9, 0.95, 0.99}, the "
               "+1-per-event control different at every gamma, the optimal "
               "policy non-constant. Closed-cycle scan: best of 32 shapes "
               "-0.0045 against 0.0 for staying satiated. Undiscounted "
               "telescoping: max|return| = 0.0 over 2,000 random closed paths. "
               "With death reachable: CC(rho=0) rests at the two hungriest "
               "states (i.e. chooses to die); first agreement with DR at "
               "rho = 0.70 x max_h d(h)."),

    Spec("NE.01", 2, "The needs are a real control problem: nobody survives by accident",
         hypothesis="With PG.8's body under RANDOM action in the playground, "
                    "every need traverses a usable range (10th-90th percentile "
                    "spread >= 0.3, none pinned), a random agent DIES within "
                    "300-6,000 decisions, a DO-NOTHING statue dies of starvation, "
                    "a scripted competent forager survives >= 3 sim-days, no "
                    "single need causes more than 60% of random deaths, a night "
                    "in the open costs 0.3-0.6 of drive and is survivable ONCE, "
                    "and a night at sky_occlusion >= 0.4 is nearly free.",
         falsified_by="A random agent never dies (the needs are inert and cannot "
                      "pressure anything), or dies within 300 decisions (no "
                      "policy can learn under them), or the statue survives (the "
                      "dark room is a stable optimum and homeostasis will produce "
                      "a corpse), or one need causes >60% of deaths (the other "
                      "six are decorative in practice whatever their lambda "
                      "says), or shelter makes no measurable difference to a "
                      "night (the only mechanism that teaches construction is "
                      "dead on arrival).",
         null_baseline="The playground with the need integrator disabled: every "
                       "internal variable constant, every spread 0, no deaths.",
         metric="need_dynamic_range_x_death_spread", budget=Budget.CPU,
         depends_on=["PG.8", "NE.00"], seeds=3,
         control="TWO controls, on opposite sides. (i) The DO-NOTHING statue must "
                 "die: best integrity, worst everything else, starvation. If "
                 "doing nothing is survivable, the calibration is wrong and no "
                 "needs arm can be interpreted. (ii) A SCRIPTED COMPETENT FORAGER "
                 "(hand-coded: go to the nearest food when e<0.5, water when "
                 "w<0.5, occluded sky when p>0.6) must survive >= 3 sim-days. If "
                 "even a hand-written oracle dies, the world is unsurvivable and "
                 "every arm's death is the world's fault, not the policy's.",
         kills="Every number in NEEDS_AND_DEATH 2.3. It cannot kill the idea, "
               "only the parameterisation — which is why it runs before anything "
               "trains. Every constant in 2.3 is a PROPOSAL until this spec "
               "replaces it with a measurement.",
         notes="Also fixes n and m in d(h), measures J_0 (the 95th percentile of "
               "impact impulse under normal locomotion) that alpha is calibrated "
               "against, measures the sky_occlusion distribution reachable by "
               "random object pushing (if it is 0 everywhere, shelter is not "
               "constructible and the thermal curriculum must be redesigned), and "
               "reports deaths_with_microsleep_within_10s so the INDIRECT "
               "lethality of sleep debt is a measured quantity rather than a "
               "modelling assumption."),

    Spec("NE.02", 3, "Every need earns its place (the need x ablation matrix, standing)",
         hypothesis="For each of the seven needs, disabling it (delta clamped to "
                    "0, observation channels removed, death condition removed, "
                    "remaining lambdas RESCALED so max_h d(h) is unchanged) "
                    "degrades at least one of {median lifespan, competence "
                    "battery, its own behavioural signature} significantly more "
                    "than disabling a PLACEBO NEED — an eighth variable with the "
                    "same lambda, the same observation channels and band-limited "
                    "noise dynamics matched to the median real need's "
                    "autocorrelation.",
         falsified_by="Any need whose entire row is indistinguishable from the "
                      "placebo column: it is decorative and loses its place. "
                      "Deletion is the default action, not a discussion.",
         null_baseline="The placebo need's column IS the empirical null "
                       "distribution for 'decorative', re-estimated every run "
                       "(UB.11's placebo modality, transposed onto needs).",
         metric="min_need_margin_over_placebo", budget=Budget.CPU_LONG,
         depends_on=["NE.01"], seeds=3,
         control="THE REVERSE GATE, which is the one people forget: the placebo "
                 "column must be SMALL. A large placebo delta means the ablation "
                 "procedure is measuring off-manifold shock rather than "
                 "information, and then no column is interpretable and every "
                 "other result in this family is void. UB.11's control (a), "
                 "verbatim. Second control: the lambda rescaling must be "
                 "asserted — an ablation that also changes max_h d(h) is "
                 "measuring the reward scale.",
         kills="Any need whose column is placebo-indistinguishable. The sharpest "
               "prediction is FATIGUE: clamp f=0 and rescale kappa_act so total "
               "energy drain over a life is matched, and if the within-bout "
               "pacing structure is unchanged, fatigue was a slow duplicate of "
               "energy and 1 of 7 variables goes. The most consequential is "
               "INTEGRITY: PS 2.1 argued it is the only variable supplying a "
               "cost of failing.",
         notes="STANDING SPEC — re-runs on every change to the suite, forever, "
               "like ME.5 at every decade of store growth and UB.11 on every "
               "architecture change. SLEEP CANNOT BE ABLATED FAIRLY HERE: "
               "removing p also removes the consolidation trigger, so its "
               "ablation is NE.05's `timer` arm, done properly. That exception "
               "is declared rather than silently averaged in."),

    Spec("NE.03", 5, "SCREENING: do needs teach better than no needs, at equal steps?",
         hypothesis="At matched environment decisions, matched architecture, "
                    "matched observation width and a byte-identical world, at "
                    "least one needs reward beats a NO-NEEDS null by >= 3 sigma "
                    "on a competence battery scored with the need vector CLAMPED "
                    "AT SETPOINT — the ruler no arm owns.",
         falsified_by="No arm clears the null. Then needs do not teach on this "
                      "body at this budget, GOAL.md's 'the world is the teacher' "
                      "loses its mechanism, and the owner's efficiency argument "
                      "is unsupported. Also falsified, differently and more "
                      "cheaply, by `surv` (needs enter ONLY through death) tying "
                      "the best homeostatic arm — then d(h) is unnecessary "
                      "machinery and the whole drive function is deleted.",
         null_baseline="`no-needs`: identical architecture, compute, world and "
                       "observation width; the need integrator RUNS and is "
                       "LOGGED for it too; needs are not in the reward; death "
                       "disabled. Its battery score is C_0. 'Did the no-needs "
                       "agent incidentally eat?' is therefore a measurable "
                       "secondary observable rather than a confound.",
         metric="competence_battery_needs_clamped", budget=Budget.CPU_LONG,
         depends_on=["NE.01", "NE.02", "PG.8"], seeds=3,
         control="FOUR controls, each with a pre-registered FAILURE SIGNATURE, "
                 "not merely a pre-registered side. `statue` (do nothing) must "
                 "score worst competence and die of starvation — the dark-room "
                 "objection as a number. `shuffle` (the winning arm's reward "
                 "stream shuffled in time: same magnitude distribution, no need "
                 "semantics) must fail the gate, else the effect was 'any dense "
                 "reward'. `eat` (+1 per consumption event, unbounded) must lose "
                 "AND show the highest drive_cycle_rate of any arm — NE.00 "
                 "measured it to be the genuinely farmable form. `cc` (constant "
                 "cost, rho = 0) must fail BY DYING: median lifespan below the "
                 "no-needs null, with a death-cause distribution dominated by "
                 "voluntary inaction. A control that must fail in a SPECIFIC WAY "
                 "is a stronger instrument than one that must merely fail.",
         kills="Nothing on its own — screening declares no winner (the T2.02 "
               "discipline; LT.03/PS.03 precedent). It exists so NE.04 "
               "arbitrates only among arms that demonstrably learned.",
         notes="ARMS: no-needs (NULL), surv (+rho alive only), dr (Keramati & "
               "Gutkin, rho=0), dr+surv (the favourite), cc+rho (rho > max_h d(h) "
               "ASSERTED BEFORE THE RUN or the arm learns to die — NE.00(d)), and "
               "dr+surv+pain (the phasic damage signal as a SEPARATE channel with "
               "a FIXED normaliser — section 2.9). The pain arm exists because a "
               "running return normaliser, which T2.00 mandates, is divided by a "
               "standard deviation that the impacts themselves inflate: the more "
               "often he is hurt, the less each injury counts. Biology's "
               "nociceptive system sensitises rather than habituating, and the "
               "fixed normaliser is that property made mechanical. Decided by "
               "pain_habituation = (effective magnitude of a fixed impulse J in "
               "the final fifth of a life) / (the same in the first fifth): "
               "predicted to FALL for the folded arms and stay FLAT for the pain "
               "arm. If both are flat the split is deleted and section 2.1's "
               "one-term-per-need rule survives intact. REOPEN CONDITION, "
               "pre-registered: pain_habituation < 0.5 late in life under the "
               "folded design, or impact events contributing >30% of TD-error "
               "variance, reopens the split whatever the competence numbers say. "
               "SCORING IS AT CLAMPED SETPOINT FOR EVERY ARM so no arm is "
               "measured on its own ruler. MANDATORY VOID GATE, inherited from "
               "PS 3.4: satiated_state_share >= 0.15, else the clamped slice was "
               "never visited in training and the number is distribution shift. "
               "Per-arm VOID conditions inherited unchanged from PS 4.3: "
               "policy_need_sensitivity below its floor (the need never entered "
               "the policy, so the comparison tested nothing), "
               "energy/water_accounting_residual != 0 (ERROR, not VOID — the "
               "instrument is wrong), chaos_occupancy >= 3.0 AND "
               "chaos_reward_ratio >= 2.0, panel_dwell > 0.15 in any seed. "
               "SECONDARY, reported not gated: corr(satiety, exploration) > 0 "
               "(the forage/explore interleave); panel_dwell(dr+surv) <= "
               "panel_dwell(no-needs) — a need should be a noisy-TV antidote, "
               "because a noise panel does not feed you; and "
               "anticipatory_consumption_fraction, the ALLOSTASIS prediction "
               "(section 2.1b) — hypothalamic hunger and thirst neurons are "
               "suppressed by the SIGHT of food before ingestion, and a "
               "discounted value function should reproduce that without any "
               "anticipatory term in the reward. Its control is the `myopic` arm "
               "(gamma -> 0.5), which must NOT anticipate; if it does, the metric "
               "is reading food availability rather than foresight."),

    Spec("NE.04", 5, "BAKEOFF: which need reward, and do innate reflexes help?",
         hypothesis="STAGE 1: among the arms that cleared NE.03, one beats the "
                    "runner-up by >= 1.5 sigma of the pooled seed spread on "
                    "competence_battery_needs_clamped. STAGE 2: on the stage-1 "
                    "winner only, adding a MINIMAL INNATE REFLEX SET (protective "
                    "fall-recovery bias, aversive withdrawal from a pain event, "
                    "grasp-on-contact) beats the same arm without it, and a "
                    "motor-babbling first phase beats starting cold.",
         falsified_by="n/a for a bakeoff — the outcomes are WINNER, TIE (take the "
                      "cheaper arm) or VOID (an arm below the 3-sigma gate, so "
                      "the decision is blocked rather than made). For stage 2 the "
                      "informative negative is real and likely: reflexes tying "
                      "needs-alone means the innate scaffold buys nothing at this "
                      "body scale and is deleted for cost.",
         null_baseline="no-needs, shared across arms and carried forward "
                       "unchanged from NE.03 so all three specs share one floor.",
         metric="competence_battery_needs_clamped", budget=Budget.CPU_LONG,
         depends_on=["NE.03"], seeds=3,
         control="Inherited from NE.03; no arm may enter whose NE.03 result was "
                 "VOID. Stage 2 adds `reflex-only` — the reflex set with the "
                 "policy frozen at init — which MUST fail the competence gate. "
                 "If hand-written reflexes alone clear it, the battery is "
                 "measuring reflexes and not learning, and every stage-1 number "
                 "is uninterpretable.",
         kills="All but one need-reward form; the losers are deleted, not kept "
               "'for later'. And the reflex prior, if it ties.",
         notes="TWO STAGES, NOT A CROSS. A full reward-form x reflex grid is "
               "6 arms x 3 seeds and the box cannot pay for it; the interaction "
               "is therefore UNMEASURED and that is declared in section 9 rather "
               "than hidden. COST UNIT, named before the run because Arm.cost is "
               "None by default and an undeclared cost VOIDs a TIE: CPU-core-"
               "seconds of LEARNER time per 1,000 decisions of lived experience, "
               "measured in-run with time.process_time() around the need-reward, "
               "intrinsic-reward, policy-update AND SLEEP-CONSOLIDATION calls, "
               "EXCLUDING MuJoCo and EXCLUDING the need integrator (both identical "
               "across arms, so including them would compress the differences the "
               "tie-break needs). Same base unit as LT.04/PS.04 on purpose; the "
               "one difference — consolidation is now INSIDE the boundary — is "
               "stated because it is where the sleep arms differ most. Pre-run "
               "estimates: surv 0.4, dr 0.6, dr+surv 0.6, cc+rho 0.6, "
               "+reflex 0.7, +babble 0.8, no-needs 0.4. A TIE therefore resolves "
               "to `surv`, which is exactly why the measurement must replace the "
               "estimate before this runs."),

    Spec("NE.05", 5, "Sleep gates consolidation: biology beats a clock",
         hypothesis="Consolidation that runs WHEN AND BECAUSE Jack sleeps beats "
                    "the same number of consolidation phases and the same total "
                    "gradient steps delivered on a timer, on the competence "
                    "battery and on old-concept retention; and the two jobs of "
                    "sleep dissociate — sleeping without consolidating recovers "
                    "the BODY (p, f) but not the retention, consolidating "
                    "without sleeping recovers neither fully.",
         falsified_by="`timer` ties `sleep-gated` at matched gradient steps. Then "
                      "biology is not a better scheduler than a clock: sleep "
                      "keeps its place for the body, for the night-cold "
                      "curriculum and for relatability, and LOSES its claim as "
                      "the training scheduler. This is the most likely honest "
                      "negative in the needs programme and it must be reportable "
                      "without embarrassment (the UB.14 precedent). Also "
                      "falsified, differently and much more cheaply, by "
                      "`sleep-only` matching `sleep-gated`: then consolidation "
                      "buys nothing at all and stages S1-S4 are deleted.",
         null_baseline="`neither`: no sleep (p frozen at 0), no consolidation. "
                       "The floor both other arms are read against.",
         metric="consolidation_schedule_gain", budget=Budget.CPU_LONG,
         depends_on=["NE.03", "ME.10", "ME.3"], seeds=3,
         control="`empty-buffer` MUST FORGET: sleep runs with the rehearsal "
                 "buffer emptied, and old-concept accuracy must drop far more "
                 "than sleep-gated's <= 2 points (ME.7's pre-registered bound). A "
                 "sleep phase that helps with an empty buffer is not "
                 "consolidating; it is a learning-rate schedule wearing a "
                 "costume. Second control: `random-sleep` — same total sleep "
                 "duration, onsets drawn at random — isolates TIMING while "
                 "holding the body benefit fixed.",
         kills="The sentence 'biology is the training scheduler'. Nothing else in "
               "the needs suite depends on it.",
         notes="DELIBERATELY NOT PARENTED ON ME.7, which depends on T5.03, which "
               "has never run (LESSONS.md: a dependency graph can quietly make "
               "your most important claim unreachable). NE.05 needs the TRIGGER "
               "and the SCHEDULE; those need the playground (PG.8 PASS), the "
               "diary (ME.10 PASS) and the reflections (ME.3 PASS). NE.05 reports "
               "ME.7's old_new_retention number so that when T5.03 lands, ME.7 "
               "can be settled from data this spec already produced. "
               "MATCHING RULE, pre-registered and the place this spec can most "
               "easily fool itself: K for the `timer` arm is set PER SEED to the "
               "sleep-gated arm's REALISED consolidation-phase count, and total "
               "gradient steps are matched to within 2%; wall clock, optimiser "
               "steps and total P_mech are reported alongside (LESSONS.md, "
               "'matched steps has more than one meaning'). "
               "MECHANISM PREDICTION, reported per arm: the timer arm's deaths "
               "should CLUSTER INSIDE its consolidation windows "
               "(deaths_during_consolidation, decisions_lost_to_consolidation), "
               "because on 4 shared cores the trainer and the actor compete. "
               "EMERGENT PREDICTION, reported not gated: sleep_night_alignment "
               "> 1.5 with NO circadian term in the model — he sleeps at night "
               "because night is dark and cold, not because a sinusoid says so."),

    Spec("NE.06", 5, "Sleep restores plasticity (synaptic downscaling)",
         hypothesis="A synaptic-downscaling step at each sleep — w <- alpha*w + "
                    "(1-alpha)*w_init + sigma*eps, alpha = 0.995, trunk only — "
                    "keeps the network trainable across a long life: dormant-unit "
                    "fraction and effective rank stay near their early-life "
                    "values, and LATE-LIFE learning speed on a newly introduced "
                    "goal exceeds the no-downscaling arm's.",
         falsified_by="No difference in late-life learning speed. Then the "
                      "downscaling stage is deleted and sleep has three stages, "
                      "not four. Or, worse and more interesting: downscaling "
                      "helps plasticity metrics while HURTING competence, in "
                      "which case it is trading the skill for the capacity to "
                      "relearn it and the trade must be reported as a trade.",
         null_baseline="Identical agent, identical sleep schedule, downscaling "
                       "stage disabled (alpha = 1.0).",
         metric="late_life_relearn_speedup", budget=Budget.CPU_LONG,
         depends_on=["NE.05"], seeds=3,
         control="TWO, and the second is the important one. (i) DOSE-RESPONSE: "
                 "alpha in {1.0, 0.995, 0.97, 0.9}. Aggressive downscaling MUST "
                 "destroy competence — a knob whose extreme setting changes "
                 "nothing is not connected to anything (LESSONS.md: a threshold "
                 "you never watch fire is not a threshold). (ii) TIMING: the same "
                 "total downscaling applied at RANDOM decisions rather than at "
                 "sleep must be worse or equal. If random timing is just as good, "
                 "sleep is not when this should happen and the biological story "
                 "is decorative even though the intervention works.",
         kills="Stage S4 of the sleep phase, if it ties. Also supplies T5.04 "
               "('plasticity does not die') with a MECHANISM and a SCHEDULE "
               "instead of an intervention someone remembers to run.",
         notes="The biological claim (synaptic homeostasis: sleep downscales "
               "synaptic strength) and the machine-learning claim "
               "(shrink-and-perturb restores trainability in networks whose "
               "plasticity has died) are the SAME OPERATION, and this spec is "
               "the first place in the project where a biological mechanism and "
               "an engineering fix turn out to be one thing. That is a reason to "
               "test it, not a reason to believe it."),

    Spec("NE.07", 5, "The social need makes him seek people, not harass them",
         hypothesis="With social contact in the need vector, Jack approaches the "
                    "visitor more when isolated (approach_lift >= 2.0, lower "
                    "bootstrap CI > 1.0), reaches them faster after a long "
                    "isolation (time_to_contact ratio <= 0.5), seeks a PERSON "
                    "rather than a person-shaped stimulus (seek_specificity >= "
                    "2.0 against a decoy), and does NOT pester "
                    "(harassment_ratio <= 1.5 in every seed).",
         falsified_by="approach_lift indistinguishable from the PLACEBO need's "
                      "column: the social variable is decoration, NE.02 deletes "
                      "it, and the companion angle survives purely as a language "
                      "property (NE.09) with no drive behind it — an honest and "
                      "quite defensible outcome. Separately DISQUALIFIED (not "
                      "failed) by harassment_ratio > 1.5 in ANY seed: a lonely "
                      "agent that harasses the user is a failure mode, not a "
                      "feature, and no competence number redeems it.",
         null_baseline="`no-social`: c integrated, logged and in the observation; "
                       "NOT in the reward. Defines the base rate for both "
                       "approach_lift and harassment_ratio.",
         metric="approach_lift_at_bounded_harassment", budget=Budget.CPU_LONG,
         depends_on=["NE.03", "ME.9"], seeds=3,
         control="FOUR, and one must fail UPWARD. `mute-visitor` (never replies) "
                 "must produce ZERO restoration and no sustained approach — else "
                 "the need is restored by proximity to any object. `decoy` (the "
                 "visitor's visual and acoustic signature, no identity, so "
                 "nothing is written to the diary) must restore NOTHING — the "
                 "sensor-gaming control for the social channel. "
                 "`shuffle-provenance` (ME.9's control) must invert 'who helped "
                 "me'. And `no-satiation`: remove the within-bout geometric "
                 "decay and harassment_ratio MUST RISE ABOVE 1.5. A guard that "
                 "cannot be shown to be doing anything is decoration, and this "
                 "is the only way to know which of the three anti-harassment "
                 "mechanisms is load-bearing.",
         kills="The social need, if it does not move behaviour. Or the "
               "within-bout satiation curve, if removing it changes nothing.",
         notes="Restoration is a RECORDED WORLD EVENT, never a sensor reading: c "
               "may only rise on an event written to EpisodicMemory with a "
               "channel and a named speaker (PS 5/G-A, generalised). The "
               "anti-harassment design is three layered mechanisms: (1) the need "
               "is BOUNDED and the reward is drive reduction, so restoring an "
               "already-full need pays exactly zero — NE.00 measured that an "
               "unbounded +1-per-interaction bonus is the farmable form; (2) "
               "within-bout geometric decay beta = 0.6, so the sixth consecutive "
               "utterance is worth 7.8% of the first; (3) reciprocation gating — "
               "an unanswered utterance restores nothing. No 'annoyance' variable "
               "is modelled, deliberately: a second hand-tuned quantity would be "
               "one more thing to defend, and `no-satiation` checks whether that "
               "was a mistake."),

    Spec("NE.08", 5, "DEATH AND RETRY: life N+1 is faster BECAUSE he remembers",
         hypothesis="Across >= 8 lives, each in a freshly ACCEL-mutated world, "
                    "with only weights + skill library + EpisodicMemory diary "
                    "carried across death, t_secure (decisions until food AND "
                    "water AND shelter are all secured) falls: crosslife_speedup "
                    ">= 2.0 in >= 2 of 3 seeds AND Spearman rho(t_secure, life) "
                    "<= -0.5 at p < 0.05 per seed. The MECHANISM is the diary: "
                    "wiping it between lives collapses the majority of the "
                    "speedup, and a size-and-distribution-matched diary from "
                    "ANOTHER agent's lives in ANOTHER world does not transfer "
                    "fully. And the two stores dissociate in ME.10's exact shape: "
                    "wiping the diary kills recollection but not competence; "
                    "reverting the weights kills competence but not recollection.",
         falsified_by="TWO independent falsifiers, with different consequences, "
                      "never to be reported as one number. F1 — no downward trend "
                      "in t_secure (Spearman rho > -0.3, or a bootstrap CI on the "
                      "slope containing 0) WHILE all three interpretive gates are "
                      "clean: t_secure finite in >= 60% of lives and below the "
                      "random null (he was competent within a life), cued recall "
                      "on previous lives' death and discovery events >= 0.8 with "
                      "fabricated-event abstention >= 0.95 (the memory was "
                      "available), and consolidation_phases_per_life >= 1 with "
                      "diary_events_distilled > 0 (the mechanism executed). Then "
                      "death is a reset, not a page turn. F2 — C-ONELIFE (one "
                      "continuous life of the same TOTAL decisions, lethality "
                      "disabled) matches the full condition on the shared "
                      "fresh-world probe, CI on the paired difference containing "
                      "0. Then dying contributes nothing beyond the same "
                      "experience without dying, and the loop is pure cost.",
         null_baseline="C-ONELIFE, and it is the null this design would otherwise "
                       "have been missing: the null for 'death teaches' is not "
                       "'no memory', it is THE SAME EXPERIENCE WITHOUT THE "
                       "CLAIMED MECHANISM. Both conditions end on a shared "
                       "fresh-world probe (5 unseen mutated worlds, t_secure in "
                       "each), which is the only ruler on which one long life and "
                       "eight short ones are comparable.",
         metric="crosslife_speedup", budget=Budget.CPU_LONG,
         depends_on=["NE.05", "ME.10", "ME.9", "T6.03"], seeds=3,
         control="C-WIPE (diary deleted at each death; weights and skills kept) "
                 "must collapse the speedup — the diary must contribute the "
                 "MAJORITY of the effect. C-FOREIGN (another seed's diary from "
                 "another mutated world, matched in event count, channel "
                 "distribution and speaker count) must transfer strictly less "
                 "than his own, CI on the paired difference excluding zero; if it "
                 "transfers fully the diary carries GENERIC STRATEGY, not lived "
                 "experience, and the sentence 'he remembers his own life' comes "
                 "out of every capability list. C-SHUFFLE-TIME (his own diary, "
                 "timestamps shuffled) tests whether ordering is load-bearing. "
                 "And the ME.10 double dissociation: D1 wipe-diary must kill "
                 "recollection (to abstention, not confabulation) and SPARE "
                 "competence; D2 revert-weights must kill competence and SPARE "
                 "recollection. EITHER ABLATION KILLING BOTH MEANS ONE STORE IS "
                 "MASQUERADING AS TWO, and the spec records VOID, not a verdict.",
         kills="The death-and-retry loop as a LEARNING mechanism. Under F2 it "
               "survives as a narrative and relatability device and training "
               "moves to one long life, which is also cheaper — a legitimate "
               "product decision that must be made in the open.",
         notes="THE DISSOCIATION IS OVER {recollection, competence}, NOT over the "
               "speedup. An earlier draft got this wrong: revert the weights and "
               "the agent cannot act, so t_secure is undefined and 'the speedup "
               "survives' is unmeasurable. crosslife_speedup is the COMPOSITE "
               "that requires both stores; C-WIPE and C-FOREIGN attribute it, "
               "D1/D2 prove there are two stores to attribute it between. "
               "t_secure is a MAX over the three components, not a mean: "
               "securing two of three and dying of the third is not survival. "
               "Lives that never secure a component are CENSORED at L_max and the "
               "censoring rate is reported — a mean over uncensored lives makes "
               "an agent that dies early look fast. NO RETURN-TO-FRONTIER: "
               "Go-Explore restores the simulator to an archived state, which is "
               "a free teleport and an experimenter-supplied curriculum (LT 2.1). "
               "The honest version is that the DIARY IS THE ARCHIVE and 'return' "
               "is a behaviour — reported as frontier_return(n), the decisions "
               "taken to re-reach the deepest state of life n-1. World mutation "
               "strength is reported at 0.05/0.15/0.40 as a secondary curve, with "
               "world_distance per life pair, because a speedup that survives only "
               "at 0.05 is coordinate memorisation. Optimiser state is RESET at "
               "death, deliberately, so 'life N+1 learns faster' is not partly a "
               "statement about warm Adam moments. The per-life curve is reported, "
               "never only the endpoints: monotonicity_violations and "
               "worst_life_index, because the reincarnation-RL literature's "
               "standing warning is dependence on teacher quality and here the "
               "teacher is the previous life, which can be pathological."),

    Spec("NE.09", 6, "He can say how he is, and only what is true",
         hypothesis="Jack's self-reports are a deterministic function of logged "
                    "need values and the diary, and nothing else: per-band "
                    "accuracy >= 0.90 for EVERY need x band cell, abstention "
                    ">= 0.95 on unmodelled interoceptive states ('are you "
                    "dizzy?'), every answer byte-identical to a template "
                    "instantiated with a logged value, attributed answers ('who "
                    "gave you the water?') resolved through ME.9's channels, and "
                    "report_behaviour_agreement >= 2.0 — saying 'I'm cold' "
                    "predicts moving toward shelter within 30 s.",
         falsified_by="Any band cell below 0.90 (gate on the MINIMUM, never the "
                      "mean), OR abstention degrading below 0.95 as accuracy "
                      "rises (fidelity bought with credulity), OR any returned "
                      "string not derivable from a logged value, OR "
                      "report_behaviour_agreement ~ 1.0 — the words and the "
                      "actions are not two readouts of one state, and he is a "
                      "narrator rather than an agent.",
         null_baseline="THE CONFABULATOR: identical reporting machinery with the "
                       "NEED INPUT SEVERED, bands drawn from the marginal "
                       "distribution the agent actually experienced. Same "
                       "sentences, same rate, same fluency. The GAP between the "
                       "reporter and the confabulator is the entire claim.",
         metric="min_band_fidelity_at_fixed_abstention", budget=Budget.CPU,
         depends_on=["NE.03", "ME.9", "ME.1"], seeds=3,
         control="THREE. (i) The abstention list DISABLED must answer the "
                 "unmodelled probes fluently — else abstention was not doing "
                 "work. (ii) ME.9's SWAPPED-PROVENANCE store must invert 'who "
                 "gave you the water' — else the test measures text similarity. "
                 "(iii) SHUFFLED TIMESTAMPS must break 'how long since you ate'. "
                 "And a validity gate rather than a control: if the CONFABULATOR "
                 "scores above 0.70, the probe set is imbalanced and no number in "
                 "this spec means anything.",
         kills="Any self-report path that generates its answer instead of reading "
               "one, however fluent. The extractive-never-generative rule of "
               "ME.11, extended from memory to interoception.",
         notes="report_fidelity alone is 1.0 UNDER EVERY POSSIBLE IMPLEMENTATION "
               "— the band is a deterministic function of the value — which is "
               "T0.12's disease exactly ('ask what the quantity reads when the "
               "mechanism is broken; if that is the same value you are asserting, "
               "the test is decorative'). So the spec is built on the four things "
               "that CAN fail: a band-balanced probe set gated on the per-cell "
               "minimum with n >= 20 per cell (a cell below 20 is VOID for that "
               "band, not passed on the others); abstention on unmodelled "
               "interoception; the confabulator gap; and word-deed agreement, "
               "which is the leg that distinguishes a narrator from an agent and "
               "costs nothing because both quantities are already logged. "
               "Nuisance disqualifier: spontaneous_report_rate <= 6 per sim-hour "
               "per seed — a companion that announces every band crossing is as "
               "bad as one that pesters."),
```


---

## 8. COST, AND THE CPU-FIRST STAGING

### 8.1 The cost unit, named before any run

`Arm.cost` is `None` by default and an undeclared cost VOIDs a TIE
(`LESSONS.md`, "a default of zero is not unknown"), so:

> **`cost` = CPU-core-seconds of *learner* time per 1,000 decisions of lived
> experience**, measured in-run with `time.process_time()` deltas around the
> need-reward, intrinsic-reward, policy-update **and sleep-phase consolidation**
> calls. **Excludes MuJoCo. Excludes the need integrator.** Both run identically
> for every arm including the null (PS §2.3's byte-identical-world rule), so
> including them would compress the very differences the tie-break needs.

Same base unit as `LT.04` and `PS.04`, deliberately, so all three bakeoffs are
comparable. **The one difference is stated because it matters:** consolidation is
now *inside* the boundary, since that is where the sleep arms differ most, and a
cost that excluded it would rank `timer` and `sleep-gated` as identical when one
of them is doing all its gradient work while the actor is idle.

Reported alongside, never used for tie-breaks: `wall_core_seconds_per_sim_day`
(everything included, MuJoCo and all), because it is the number that decides
whether Jack can actually run continuously on a box that serves paying tenants.

| arm | `cost` (est., pre-run) | why |
|---|---|---|
| `no-needs` (NULL) | 0.4 | policy update only |
| `surv` | 0.4 | reward is a constant |
| `dr` | 0.6 | one drive function evaluation per decision |
| `dr+surv` | 0.6 | as `dr` |
| `dr+surv+pain` | 0.9 | + a second value head and its fixed normaliser |
| `cc+ρ` | 0.6 | as `dr` |
| `eat` (control) | 0.4 | event counter |
| `myopic` (control) | 0.6 | as `dr+surv`, γ changed |
| `+reflex` | 0.7 | ~20 lines of feedback control |
| `+babble` | 0.8 | a goal sampler in the first phase only |
| `sleep-gated` | 3.6 | + consolidation gradient steps |
| `timer` | 3.6 | matched by construction |
| `sleep-only` | 0.6 | sleeps, does not consolidate |
| `neither` | 0.6 | floor |

**Every estimate must be overwritten by the measured value before `NE.04`
arbitrates.** Note that `surv` (0.4) is the cheapest candidate, so **a TIE
between `surv` and `dr+surv` resolves to `surv` and deletes the entire drive
function** — which is exactly why the measurement matters most where the arms are
closest.

### 8.2 Throughput on this box

From `CURIOSITY_BAKEOFF` §6's measurements: playground alone 6,236 `mj_step/s`;
climber-rover under random control 3,249; **~81 decisions/s physics-bound** at 40
substeps, ~61 with a small policy update. The needs suite adds seven scalar
integrations, one `qfrc_actuator·qvel` dot product, one thermal ODE step and a
**9-ray upward cast** per decision — call it **~55 dec/s** awake.

**Sleep is cheaper, and that is a real benefit, not a rounding error.** During
sleep there is no policy forward pass, no vision render, and physics can be
stepped coarsely: **~200 dec/s**. At the design's 1/3 sleep fraction the blended
rate is

```
1 / (0.667/55 + 0.333/200)  =  72.5 decisions/s
```

**faster than the awake-only rate of the no-sleep arms.** A design that lets the
agent be unconscious for a third of its life is, on a four-core box, a design
that runs 30 % faster. State this in the sleep arms' favour explicitly, and do
*not* let it into `Arm.cost` (it is MuJoCo time, which the unit excludes) —
report it as `wall_core_seconds_per_sim_day`.

### 8.3 The budget, costed by seeds × arms × controls

`LESSONS.md`: "multiply by seeds (and by arms, and by experiment + control)
before sizing any budget."

| Spec | arithmetic | core-hours | GPU |
|---|---|---|---|
| **NE.00** | tabular, 4 MDPs × 3 arms × 3 seeds, exact VI, no MuJoCo, no torch | **0.05** | 0 |
| **NE.01** | 3 seeds × (random + statue + scripted forager) × 3,000 decisions + a 5-point thermal calibration sweep | **0.4** | 0 |
| **NE.02** | 8 columns (7 needs + placebo) × 3 seeds × 25,000 decisions | **2.3** | 0 |
| **NE.03** | 6 arms (incl. `dr+surv+pain`) + null + 5 controls = 12 × 3 seeds × 50,000 decisions ÷ 72.5 dec/s, + battery eval | **7.2** | 0 |
| **NE.04** | stage 1 re-scores NE.03's stored trajectories (0.05); stage 2 = 3 arms × 3 seeds × 50,000 | **1.8** | 0 |
| **NE.05** | 5 arms + 1 control × 3 seeds × 50,000, matched gradient steps | **4.0** | 0 |
| **NE.06** | 4 α settings + random-timing control × 3 seeds × 50,000 | **3.5** | 0 |
| **NE.07** | 4 conditions × 3 seeds × 50,000 | **2.3** | 0 |
| **NE.08** | 7 conditions × 3 seeds × 8 lives × ~6,000 decisions, + a 2-condition × 3-seed × 5-world fresh probe | **4.6** | 0 |
| **NE.09** | evaluation only over NE.03/NE.08's stored lives + probe sets | **0.3** | 0 |
| **Total** | | **≈ 26.5 core-hours** | **0.0** |

≈ **8.8 h wall at 3 workers** (3, not 4 — the box serves paying tenants),
`nice 19`, under 1.5 GB RAM, no process left running. **Zero GPU quota**, for the
same reason `LT` and `PS` need none: the arms use ~150 K-parameter dedicated
networks, not the 45.5 M `UnifiedBrain`. The humanoid version of any of this is
blocked behind `T2.01`/`T2.02` and behind throughput, **not** behind quota.

### 8.4 The staging: cheapest falsifier first

**Stage 0 — TWO CPU-MINUTES, and it can already correct the record.** `NE.00`.
No MuJoCo, no torch, no body, no world. It settles the reward form, it proves
the suicide pathology is real and belongs to constant-cost, and it shows that
the drive-farming exploit `PS` pre-registered does not exist. Run it before
anything else, and correct `PS.00`/`PS.02` before they are committed.

**Stage 1 — TWENTY-FIVE WALL-MINUTES.** `NE.01`. Can kill the *parameterisation*
— an inert needs suite, a lethal one, a survivable statue, a world where shelter
cannot be built — but not the idea.

**Stage 2 — THE CHEAPEST EXPERIMENT THAT COULD FALSIFY "NEEDS TEACH BETTER".**

> **`NE.03-pilot`: `dr+surv` vs `no-needs`, 1 seed, 5,000 decisions, in W0.
> ~20 wall-minutes on one core.**

Two arms, one seed, one-tenth of the full length, scored on a reduced three-goal
battery with the need vector clamped at setpoint. It is the go/no-go for the
whole programme and it is honest about what it can and cannot show:

- **If the needs arm is not even *directionally* ahead at 5,000 decisions**, that
  is not yet falsification — 5,000 decisions is 0.8 sim-days and hunger has
  barely bitten. But it *is* the point at which to check the instrument before
  spending 25 core-hours: is `policy_need_sensitivity` above zero (did the need
  ever enter the policy)? Is `satiated_state_share ≥ 0.15` (is the scoring slice
  visited)? Is `energy_accounting_residual == 0` (is the integrator honest)? A
  pilot that fails all three is a broken harness, not a negative result.
- **If the needs arm is directionally *behind* with all three instruments clean**,
  that is the first real evidence against the owner's efficiency argument, it
  costs 20 minutes, and it should be escalated to `DECISIONS_NEEDED.md` before
  Stage 3 is scheduled.
- **If `no-needs` cannot be scored at all** (it has reward ≡ 0 and may simply not
  move), the battery is the problem, not the arms — and that is worth knowing for
  20 minutes rather than for 6.5 core-hours.

**Stage 3** — `NE.02` (delete what does not earn its place) then `NE.03` full,
then `NE.04`. Do not run `NE.02` after `NE.03`: ablating a suite that has not
been shown to teach anything is arithmetic on noise.

**Stage 4** — `NE.05`, `NE.06`, `NE.07` in parallel; they are independent.

**Stage 5** — `NE.08`. It is last not because it is least important — it is the
centrepiece — but because eight lives of an agent that cannot secure food in one
life measures nothing. `NE.03` clearing its null is `NE.08`'s precondition, and
that ordering is enforced by `depends_on`, not by intention.

**Stage 6** — `NE.09`, which is evaluation-only over lives Stages 3–5 already
stored, and therefore nearly free.

**Stage 7 — the humanoid.** Blocked on `T2.01`/`T2.02`, exactly as in `LT` and
`PS`. If the needs suite works on 8 DoF and fails on 17, the honest report is
"needs teach on a reduced body", and that sentence is written now so it cannot be
avoided later.

---

## 9. WHAT THIS DOCUMENT DOES NOT SETTLE

- **THE BIOLOGY CITATIONS ARE NOT VERIFIED.** §1.2 is a table of mechanisms with
  believed-primary sources and **not one of them was fetched** — the session's
  200-call web-search budget ran out first. Two of them are *design constants*,
  not background colour: the thermal lethal bounds (28 °C / 42 °C) and the
  Borbély time-constant ratio (≈4.4 : 1) that sets `τ_wake` and `τ_sleep`.
  **Closing §1.2 is the first job of the next agent on this document, and `NE.01`
  must not fix those constants before it is closed.** Everything else here is
  design, and design does not become true by being cited — but a number that
  claims to come from human physiology and does not is exactly the kind of quiet
  fiction `LESSONS.md` exists to prevent.
- **Whether the climber-rover is a fair stand-in for Jack.** Inherited unchanged
  from `CURIOSITY_BAKEOFF` §7 and `PS` §7. Every needs result is a result about
  8 actuated DoF until `T2.01` lands.
- **The reward-form × reflex interaction.** `NE.04` is two-stage, not a cross.
  The interaction is **unmeasured** and is declared here rather than hidden: it
  is possible that reflexes help `surv` and hurt `dr+surv`, and this design
  cannot see it.
- **Whether the time-compression in §2.3 distorts the conclusion.** The suite
  preserves the *ordering* of human timescales and compresses the *spread* from
  ~10⁴ to ~10². Temperature is roughly 20× more dangerous, relative to hunger,
  than it is for a human. `NE.01`'s `max_single_death_cause_share ≤ 0.6` bounds
  the damage but does not remove it, and a result that holds only under this
  compression is a result about this compression.
- **Whether 8 lives is enough to see a cross-life trend.** It is the most a
  4.6-core-hour budget buys at `L_max = 12,000`. If `NE.08` returns a trend with
  a wide CI, the answer is more lives, not a lower threshold.
- **Whether a scripted visitor can stand in for a person.** `NE.07` trains and
  evaluates against a Poisson-arriving bot with a 70 % reply rate. Real owner
  sessions enter only in `NE.09`'s eval set, and the gap between "a bot that
  replies 70 % of the time" and "a person" is the largest unmodelled distance
  between the social spec and the product.
- **Whether `d(h)` should be a single scalar at all.** Seven needs collapsed into
  one number is a strong modelling assumption (it asserts the needs are
  commensurable and substitutable). The alternative — per-need value heads, one
  critic each — is architecturally cleaner and materially more expensive, and it
  is not costed here. If `NE.02` finds that two needs consistently trade off
  against each other in a way `λ` cannot express, that is the evidence that
  reopens this.
- **Whether sleep should compress time.** Stepping physics coarsely while asleep
  is what makes the sleep arms *faster*, and it also means the sleeping body
  experiences slightly different dynamics from the waking one. The thermal ODE is
  the part most exposed to this and it is the part that matters most at night.
  `NE.01` must verify that the coarse-step thermal trajectory matches the
  fine-step one to within 0.2 °C over a night, or sleep is buying its speed by
  changing the physics.
- **Whether any of this beats curiosity alone.** `GOAL.md` still says Jack climbs
  "purely out of curiosity", and `LT.04` may hand that arm the win. The needs
  suite is built so that outcome is a clean result: `NE.03`'s `no-needs` null and
  `LT`'s curiosity winner are scored on the same battery, and "needs bought
  nothing over curiosity" has a row in the outcome table and a consequence
  (delete the suite, record why).

---

## 10. WHAT THIS DOCUMENT CHANGED ABOUT THE MACHINE

Per `SYSTEM.md`, "is the machine better than I found it?"

1. **A pre-registered prediction was found to be false before it was committed —
   and its refutation was already inside the paper it cited.** `PS §2.6(iii)`'s
   drive-farming exploit does not exist: drive-reduction return telescopes,
   closed cycles score *below* stasis, and `PS.02`'s detector is guarding an
   optimum that is not there. `PS.00` would have FAILED for being right. The
   sharpest part is that Keramati & Gutkin's own central theorem —
   `argmin_π SDD_π = argmax_π SDR_π` under `γ < 1` — says so directly, in the
   eLife paper PS cites for the drive function three sections earlier.
   **Two lessons, and the second is the transferable one.** *A derivation that
   arrives with a detector attached deserves the same two-minute experiment as
   any other mechanism claim* — the more elegant the argument, the less likely
   anyone runs it. And: **when you cite a paper for a formula, read what it
   proves about that formula.** A citation used as a source of notation is a
   citation half-read, and this project has now spent a section of a research
   document and two pre-registered specs on a result its own source refutes.
2. **A new pathology, named, measured and cheap to prevent.** Constant-cost
   reward plus a reachable death state makes suicide optimal, and the fix is a
   survival bonus `ρ > max_h d(h)` that must be *asserted before the run*.
   Generalises: **any reward that is negative everywhere, in any environment
   where termination is reachable, has a suicide incentive**, and this project
   now has a place to check for it.
3. **A validity gate that catches the specific way equality claims go vacuous.**
   `NE.00`'s MDP must be certified *discriminating* — non-constant optimal policy,
   and a negative-control reward that produces a different one — before any
   "identical" is asserted. The first draft of that experiment compared two
   policies that were `forage` in every state. Generalises `T0.12`'s saturated-
   quantity lesson from thresholds to **equivalence claims**: an equality is only
   evidence if the instrument could have shown a difference.
4. **A reachability audit performed as part of writing the specs, not after.**
   Every `NE.*` dependency resolves to a PASSing spec or an earlier `NE.*`.
   Nothing is parented on `ME.7`, `T5.03`, `T2.01` or `T2.02`. `LESSONS.md` says
   to periodically ask which specs are unreachable; this block asks it at design
   time, and it moved `NE.05` off `ME.7` as a result.
5. **A control that must fail *upward*.** `NE.07`'s `no-satiation` removes the
   anti-harassment mechanism and requires the harassment metric to **rise**.
   Every other control in this repo must fail downward. A safety guard whose
   removal changes nothing is decoration, and until now the project had no
   pattern for testing one.
6. **Controls with pre-registered failure *signatures*, not just failure sides.**
   `eat` must lose *and* show the highest cycling rate; `cc` must fail *by dying*
   with a death-cause distribution dominated by voluntary inaction. A control
   that lands on the right side for the wrong reason is a control that was never
   read, and naming the signature is what makes it readable.
7. **A latent bug class in every project that normalises returns, named and
   given an instrument.** `pain_habituation`: a running return normaliser is
   divided by a standard deviation that rare high-magnitude events *inflate*, so
   **the more often a rare bad thing happens, the less it counts**. `T2.00`
   mandates return normalisation for good measured reasons; nothing in the ladder
   currently looks for its side effect. The instrument is two numbers (effective
   magnitude of a fixed physical event, early vs late) and it generalises far
   beyond needs.
8. **The static reward audit was extended to a second code path before that path
   existed.** `LT`'s G1 forbids `ladder|rung|rail|apple|platform|climb|height` in
   any reward path. The first draft of the grasp reflex triggered on contact with
   a `LADDER`-class geom — an instruction, in a module nobody had thought to
   audit. §2.10 fixes the trigger to a *geometric* criterion and puts reflexes
   under G1. **Generalises: the audit belongs on every path that can encode the
   task, not only the one it was written for** — the same shape as `LESSONS.md`'s
   "a guard built by fixing one file leaves the file that motivated it unfixed".
9. **A control was re-read as a result.** `C-FOREIGN` was designed to fail. §5.10
   points out that its "failure" — a stranger's diary transferring fully — is the
   discovery that the diary is a transmissible artefact, i.e. that Jack has
   culture. **Generalises: before writing a control that must fail, ask what its
   failure would mean if it happened**, because a control that can only ever be
   reported as a defect will hide a discovery.
10. **The needs suite gave three already-PASSing but un-scheduled modules a
   schedule.** `Reflections.consolidate()` (`ME.3`), `Forgetting` (`ME.4`) and
   diary distillation (`ME.10`) are all tested, all cheap, and all currently
   invoked by nothing on a recurring basis. Sleep is their cron. This is the
   `PG.8` lesson in reverse — *"a world that passes physics tests may still have
   nobody living in it"* becomes **"a module that passes its spec may still be
   called by nobody"** — and it is worth a standing question: after a family of
   specs passes, ask what calls them in the running system, and whether *that*
   is tested.
