# DIRECTION_AUDIT.md — the whole ladder re-examined against the survival-world direction

> Written 2026-08-09 in response to the owner's directive recorded in `GOAL.md`
> §"The world is the teacher": *"This must be checked over every single test
> we've done and every single architecture decision we have made... we need to
> build a Jack that can see, that can talk, that can walk, and that can learn,
> before we throw him in a world and see him using all of those things
> together."*
>
> This is an audit, not a design. The survival world itself is being designed in
> `SURVIVAL_WORLD.md`, `NEEDS_AND_DEATH.md` and `LEARNING_CORE.md` (all three
> were stubs when this was written; nothing here depends on them). **Nothing is
> deleted here and no registry file was touched.** Deletion and architecture
> calls are the owner's; this document classifies and justifies.

---

## 0. Method, and what was actually run

`LESSONS.md` — *"verify a mechanism claim before fixing it, even from a careful
source"* — applies to this document. Every mechanical claim below was executed
on this box, not reasoned about. The commands and their outputs:

| check | command | result |
|---|---|---|
| registry loads, spec count | `python -c "from experiments.registry import BY_ID; print(len(BY_ID))"` | **128** |
| ledger state | `Ledger()` over `experiments/ledger.json` | 51 entries: **48 PASS, 2 VOID (T2.01, T2.02), 1 ERROR (T1.02)**; 77 specs never run |
| implementations | files in `experiments/tests/` | **51 of 128** specs have an implementation |
| dangling dependencies | set(all `depends_on`) − set(BY_ID) | **none** |
| transitive blockage | fixed-point over `depends_on` from {T1.02, T2.01, T2.02} | **40 specs** unreachable until a currently-bad spec turns PASS |
| id-prefix hazard | `fnmatch("wp_10_x.py", "wp_1_*.py")` etc. | **False** — proposed ids below are safe, and two-digit ids are used anyway |
| control-step duration | gymnasium `Humanoid-v5`: `timestep 0.003 × frame_skip 5` | **`dt` = 0.015 s; a 1000-step episode is 15 SECONDS of simulated time** |
| throughput, 160K MLP policy | measured now, `nice 19`, single env, 2000 steps | **1531.6 env-steps/s → 23.0 sim-seconds per real second** |
| throughput, 57M trunk | T0.07 ledger, `policy_steps_per_s` | **11.48 env-steps/s → 0.17 sim-seconds per real second** |
| throughput, zero action | measured now | 2105.9 env-steps/s |
| offscreen rendering | `MUJOCO_GL=osmesa`, `MUJOCO_GL=egl` | **both FAIL — no `libOSMesa`, no `libEGL` on this box** |
| offscreen rendering, working path | `xvfb-run -a -s "-screen 0 640x480x24" MUJOCO_GL=glfw` (swrast/llvmpipe) | **works: 14.6 fps @128², 5.4 fps @320²** |
| harness timeouts | `run.py:_budget_seconds × seeds × 2` | `cpu<2h` at 3 seeds = **54,000 s (15 h)** wall cap per spec |
| builder loop cap | `scripts/ladder_loop.sh` | `timeout 50m` per iteration |

Two of these overturn things currently believed in the repo, and both are
recorded as findings in §6.

Docstrings were not trusted over code: verdicts below were assigned from the
`Spec(...)` bodies in `experiments/registry.py` and `experiments/registry_expansion.py`
(read in full), cross-checked against the ledger's recorded metrics, not against
`CHECKLIST.md` (which `LESSONS.md` already records as having gone stale).

---

## 1. The direction, restated as testable deltas

The owner's directive is not a mood; it changes five specific things about what
a test must look like. Everything in §3 and §4 is scored against these.

| # | delta | what it invalidates | what it demands |
|---|---|---|---|
| **D-a** | **Needs are the curriculum.** Hunger, thirst, sleep, temperature (both ends lethal), company. | Any spec whose *null* is "no task, free play" — because free play now only exists when needs are satisfied. | Every intrinsic-motivation spec gains a **needs-only arm** and a **needs-off arm**; without both, curiosity and starvation are confounded. |
| **D-b** | **Death is a page turn, not a reset.** Life N+1 must beat life N *because of* what life N recorded. | The episode as the unit of analysis. `run_spec`'s `seeds` and the ledger's one-result-per-spec both assume a run, not a lifetime. | A **life** unit: cross-life metrics, death-vs-crash disambiguation, and persistence of world+need+diary state across a session boundary. |
| **D-c** | **Lives are hours long.** | "1000-step episodes" — measured above at **15 seconds of simulated time**. An hour of life is 240,000 control steps, 240× a current episode. | A **real-time-factor gate**. This is the delta with teeth: see §4.5. |
| **D-d** | **The owner watches and interacts.** | The assumption that a spec's only consumer is the ledger. | Rendering, a stream, and a proof that being watched does not change behaviour. Nothing in 128 specs touches any of this. |
| **D-e** | **The staging is unchanged: see / talk / walk / learn, THEN the world.** | Nothing — this *protects* the existing ladder. It is the reason §3 is 85 KEEPs and not a rewrite. | An ordering discipline: the world specs must be **gated behind**, not substituted for, the capability specs. §7. |

D-e is worth dwelling on, because it is the finding that most of this audit
turns on: **the survival direction is overwhelmingly additive.** It re-aims the
Tier-5 claims and the curiosity family, it obsoletes one small cluster (which
was already obsolete for unrelated reasons), and it leaves Tiers 0–3 and the
memory family essentially untouched. The ladder was not built in the wrong
direction. It was built in the right direction and stopped short of a world.

---

## 2. Verdict key

- **KEEP** — load-bearing under the survival direction, unchanged.
- **ADAPT** — the claim survives; the venue, null, control or metric must
  change. The required change is named in every case; "run it in the new world"
  alone is not an adaptation.
- **OBSOLETE** — measures something no longer wanted, or measures it worse than
  another spec already in the registry. The replacement is named. **No deletion
  is proposed or performed** — the owner decides.
- **BLOCKED-ON-DIRECTION** — cannot be classified until a named open question
  resolves. The question is named.

---

## 3. Spec-by-spec verdicts (all 128)

### 3.1 T0 — harness (16) · KEEP 14 · ADAPT 2

| id | title | verdict | reason |
|---|---|---|---|
| T0.01 | Repo imports clean | KEEP | Structural; world-agnostic. |
| T0.02 | Deterministic seeding | KEEP | Every A/B in the world rests on it. |
| T0.03 | Checkpoint round-trip fidelity | KEEP | Prerequisite for the life-state round trip (LF.02 gap). |
| T0.04 | Resume continues, does not restart | KEEP | A multi-hour life will span a session boundary; this is the mechanism. |
| T0.05 | Preemption survival | KEEP | Same; strengthened in importance, unchanged in content. |
| T0.06 | Env/policy dimension contract | **ADAPT** | Interoception (temperature, hunger, thirst, energy, sleep-pressure) appends channels to the 348-dim obs. The contract must be asserted against the *world's* declared drive vector, not just `env.observation_space` — otherwise the padding bug it was written to catch recurs on the need channels. |
| T0.07 | CPU throughput baseline | **ADAPT** | It records `steps_per_s` and `hours_for_2M_steps`. The survival world needs **sim-seconds per real second** and **real-hours per sim-hour** — the same measurement in the unit that decides feasibility. Add those two derived figures and the render cost; see §4.5. |
| T0.08 | Metrics land in the ledger | KEEP | |
| T0.09 | Colab T4 round-trip | KEEP | |
| T0.10 | Kaggle round-trip | KEEP | |
| T0.11 | Backend failover | KEEP | |
| T0.12 | GPU-hour accounting | KEEP | Unchanged. The *new* scarce resource is CPU wall-clock on a tenant-shared box, which is a separate spec (T0.18 gap), not a change to this one. |
| T0.13 | No gate in the ladder is decorative | KEEP | Applies to every new world gate on day one. |
| T0.14 | Evaluation deterministic; obs contract | KEEP | |
| T0.15 | The recorder cannot disarm a threshold | KEEP | Becomes *more* load-bearing: long lives are 3-seed by default and `_aggregate` only short-circuits at one run. |
| T0.16 | The SHIPPED eval is deterministic | KEEP | |

### 3.2 T1 — learning primitives (13) · KEEP 12 · ADAPT 1

| id | title | verdict | reason |
|---|---|---|---|
| T1.01 | Overfit a single batch | KEEP | |
| T1.02 | Shuffled-target control (generalisation) | KEEP | Currently ERROR for an infrastructure reason (`kaggle: 0.0h left`), not a scientific one. The claim is unaffected by the direction. |
| T1.03 | Gradient reaches every parameter | KEEP | |
| T1.04 | Weights actually move | KEEP | |
| T1.05 | Frozen stays frozen | KEEP | Strengthened: the frozen-swappable-tower decision (§4.2) is enforced by this spec. |
| T1.06 | Numerical stability | KEEP | Strengthened: 1000 steps is 15 s of life. A multi-hour life is 240× the exposure to a slow divergence. Consider a standing re-run at life length (folded into LF.01). |
| T1.07 | Not knife-edge on learning rate | KEEP | |
| T1.08 | Seed variance measured | KEEP | |
| T1.09 | Fits in T4 memory | KEEP | |
| T1.10 | CPU and GPU agree | KEEP | |
| T1.11 | Train/inference path parity | KEEP | |
| T1.12 | Flow matching actually denoises | KEEP | Contingent on the flow head surviving §4.4, but the spec is a correct test of the mechanism as it stands. |
| T1.13 | The grounding pairs are real | **ADAPT** | Its title claims "the grounding pairs"; its content certifies the **CMU/KIT imitation corpus**. Under the new direction that corpus serves the *walk* stage only — in-world language ("I'm cold") is grounded in Jack's own state, which this dataset cannot supply. Narrow the title and scope to the imitation corpus so it stops implying coverage of grounding in general. |

### 3.3 T2 — component competence (21) · KEEP 10 · ADAPT 11

| id | title | verdict | reason |
|---|---|---|---|
| T2.00 | The RL update is sane | KEEP | Cheap CPU guard; survives whatever the learning-core bakeoff picks, as long as PPO is in the repo. |
| T2.01 | Locomotion beats a random policy | KEEP | This **is** the "walk" gate the owner reaffirmed. Must be re-run post-T0.14/T0.16 regardless. |
| T2.02 | Locomotion beats the honest MLP | **ADAPT** | Add a second pre-registered axis: **sim-seconds per real second**, gated, not merely reported. Under D-c the arms differ by 133× on this axis (§4.5) and the survival world is only buildable on one side of it. An architecture bakeoff that ignores the axis that decides feasibility is measuring the wrong thing. |
| T2.03 | Pretrained vision beats random features | KEEP | The "see" gate. |
| T2.04 | Behaviour cloning on scripted trajectories | KEEP | |
| T2.05 | World model beats constant prediction | **ADAPT** | The state to predict now includes need dynamics (core temperature under wind/water, satiety decay). Persistence is a *much* stronger null for a slow-moving need variable than for a joint angle — say so before running, or the null wins and it will look like a model failure. |
| T2.06 | Language-action alignment beats chance | **ADAPT** | Venue moves off the synthetic `ACTION_CATEGORIES` anchor table onto in-world commands. `LANGUAGE_GROUNDING.md` §0 Finding 1 already indicts the metric: retrieval accuracy is high for a policy that is not listening. The eval must be certified **language-necessary** first (its LG.00). |
| T2.07 | Grounding generalises to held-out phrasings | **ADAPT** | Same as T2.06; held-out cells must exist in an in-world verb×object grid. |
| T2.08 | Curiosity drives coverage | **ADAPT** | The canonical case. Its null is "uniform random actions"; under D-a the honest null is a **needs-only agent**, which covers ground because it is looking for water. Add that arm, and report coverage in *free time* (needs satisfied) separately from coverage under duress. |
| T2.09 | Noisy-TV control | **ADAPT** | A needs-driven agent avoids the panel because it starves, not because its curiosity signal is sound. Immunity must be measured with **needs OFF**, or the fixture certifies nothing about the curiosity signal. This is a genuine weakening the direction introduces and it must be pre-empted. |
| T2.10 | Memory retrieval beats recency | KEEP | PASS; unaffected. |
| T2.11 | Skills are distinguishable | KEEP | |
| T2.12 | Emotion states are distinguishable | **ADAPT** | `EmotionalState.get_energy()` is an *arousal* scalar; the world's `energy` will be metabolic. `PURPOSE_AND_SCAFFOLDING.md` §0 flags the collision. The spec must additionally assert the two are not wired together, or a mood variable will silently become a drive variable and no ablation will be able to tell them apart. |
| T2.13 | Train to convergence, not to a step count | **ADAPT** | "Converged" is not defined for an agent that lives. The increment must become a **cohort of lives**, and the stopping quantity cross-life improvement, not per-increment held-out gain. |
| T2.14 | Imitation from real motion capture | KEEP | Walk stage. |
| T2.15 | Free-form language routes to the right task | **ADAPT** | As T2.06/T2.07. |
| T2.16 | Hindsight goal-reaching (flow weld) | KEEP | This is a candidate arm of the learning-core bakeoff and stands on its own either way. |
| T2.17 | Progress and success estimation | **ADAPT** | The survival world hands out a **free progress label**: need-delta. Add it to the null set — a progress head that cannot beat "hunger went down" has not earned its parameters. Strictly strengthens the spec. |
| T2.18 | Chunking earns its keep under latency | KEEP | Promoted in priority: chunking amortises the policy forward, which is the exact bottleneck D-c exposes. |
| T2.19 | Flow head handles multimodal actions | KEEP | |
| T2.20 | Episodic memory helps the next episode | **ADAPT** | Change the unit from **episode N+1** to **life N+1**, across a death. That single word change turns an already-PASSing spec (search time 0.046× the memoryless null, with two controls at null) into the first direct evidence for D-b. It is the cheapest survival-direction result available and it is already 90% built. |

### 3.4 T3 — earn your parameters (10) · KEEP 9 · ADAPT 1

The direction *strengthens* this whole tier — the owner said the system, not the
model, is the product, and that complexity must earn its place.

| id | title | verdict | reason |
|---|---|---|---|
| T3.01 | Ablate vision | KEEP | |
| T3.02 | Ablate proprioception | KEEP | |
| T3.03 | Ablate the world model | KEEP | |
| T3.04 | Ablate the hierarchical planner (37.17M, zero call sites) | KEEP | |
| T3.05 | Ablate temporal memory (12.64M, never passed `memory=`) | KEEP | |
| T3.06 | Ablate curiosity | KEEP | Its meaning sharpens: with needs present, "removing intrinsic reward" now has a competitor explanation, but the ablation itself is unchanged. |
| T3.07 | Ablate mood conditioning | **ADAPT** | Mood must be need-grounded before the ablation means anything (see T2.12). Ablating a free-running PAD model answers a question nobody is asking. |
| T3.08 | Ablate the LLM | KEEP | Strengthened by D-c: T0.07 measured SmolLM2-1.7B at 6.9 GB resident and **0.0% of rollout time** — it is 96.8% of the process's parameters and never runs in `forward()`. In a multi-hour life that is pure resident cost. |
| T3.09 | The creative loop earns its existence | KEEP | Still "wire it or delete it". The direction supplies no role for `AlphaGeometryLoop.py`, so the likely outcome is delete — but the test is the right one and costs CPU. |
| T3.10 | Trunk knowledge survives action training | KEEP | The cheapest direct D1 evidence; promoted. |

### 3.5 T4 — composition (5) · KEEP 1 · ADAPT 1 · OBSOLETE 3

| id | title | verdict | reason |
|---|---|---|---|
| T4.01 | Modality dropout robustness | **OBSOLETE** | Superseded by **UB.11**, which measures the same matrix with two methodological upgrades T4.01 lacks: a **placebo modality** supplying the empirical null for "decorative", and **cross-episode swap** instead of zeroing (UB.11's own notes: "Ablation uses the learned [MISSING-m] token, never zeros, or the matrix measures brittleness"). Cause is pre-existing supersession, not the new direction. **Consequence:** T4.01 is UB.1's only parent and T3.02's only route in — see §6.1. |
| T4.02 | No modality collapse | **OBSOLETE** | Superseded by UB.11 (the all-null-row assertion) plus UB.12. Per-modality gradient norms remain worth logging, but as an instrument, not a claim. |
| T4.03 | Fusion actually fuses | **OBSOLETE** | Superseded by UB.10's swap control ("every arm must FAIL the cross-episode SWAP ablation on at least one sense") and UB.12's ensemble null. T4.03's single-modality batch shuffle is the weakest of the four perturbations UB.11 enumerates. |
| T4.04 | Task interference | **ADAPT** | There are no discrete tasks in a survival world; there are **competing needs**. Reframe: learning to find water must not destroy shelter-building, measured across a life. The 10%-retention threshold and the "A trained alone" null both carry over intact. |
| T4.05 | Full regression gate | KEEP | Becomes the gate on every world change. |

### 3.6 T5 — the claims (9) · KEEP 2 · ADAPT 5 · BLOCKED-ON-DIRECTION 2

| id | title | verdict | reason |
|---|---|---|---|
| T5.01 | Physics pre-training transfers (THE thesis test) | **BLOCKED-ON-DIRECTION** | Its premise is that a **Phase-0 SymPy symbolic physics pre-training** stage exists and is the project's differentiator. `GOAL.md` now says *the world is the teacher* and the owner says the product is "just a system that can learn". Whether a symbolic pre-training phase exists at all is exactly the **learning-core bakeoff** question (`LEARNING_CORE.md`). Do not start it: at `seeds=5 × GPU_LONG` it is the most expensive unrun spec on the ladder and its premise may not survive the week. |
| T5.02 | Physics violation detection | **BLOCKED-ON-DIRECTION** | Child of T5.01; same question. |
| T5.03 | Continual learning: forgetting measured | **ADAPT** | Central to D-b. The task sequence becomes a **life sequence**; backward transfer is measured on life-1 competences after life-N. |
| T5.04 | Plasticity does not die | **ADAPT** | "N consolidation cycles" becomes "N lives". The most direct test that death-and-retry does not silently ossify him. |
| T5.05 | Sleep consolidation beats online-only | **ADAPT** | Sleep is now a **need with an in-world night**, not an offline GPU pass. The null ("online head, no consolidation") survives; the phase boundary must be the world's, and the spec must state what happens to the body while he sleeps (he is vulnerable — that is the point). |
| T5.06 | Unprompted exploration is real | **ADAPT** | "Left alone" becomes "**when needs are satisfied**". This is precisely the owner's framing and it makes the spec sharper, because free time is now an earned, measurable state rather than an assumed default. |
| T5.07 | Behaviour visibly changes after training | KEEP | **Promoted.** Its own note says it is "the only test whose result the owner can verify with his own eyes", and the owner now says he wants to watch. It is also the cheapest consumer of the rendering work in §5.4. |
| T5.08 | Open-endedness: learning does not saturate | **ADAPT** | Venue becomes **world mutation across lives** rather than scene mutation within a run. `PlaygroundParams.mutate()` and PG.8's `world_mutated` metric already exist; the unit changes. |
| T5.09 | Skills transfer across bodies | KEEP | Unaffected; low priority. |

### 3.7 T6 — the living Jack (5) · KEEP 1 · ADAPT 4

| id | title | verdict | reason |
|---|---|---|---|
| T6.01 | Full episode completes | **ADAPT** | Three changes, all forced: (a) unit becomes a **life**, measured in *simulated* hours not wall minutes; (b) the falsifier "any crash, hang, or non-finite action" must **distinguish death from crash** — under D-b death is the expected outcome and this spec currently reads it as failure; (c) `minutes_survived` becomes a survival-time distribution over lives. |
| T6.02 | Long-run stability | **ADAPT** | Same unit change, plus need-variable drift (a temperature integrator that ratchets) joins action saturation and mood lock in the falsifier. |
| T6.03 | Cross-session persistence | KEEP | PASS. It is the substrate for "what survives death" and needs no change — what needs adding is that the *world and need state* also survive, which is a new spec (LF.02), not a modification of this one. |
| T6.04 | Everything at once, end to end | **ADAPT** | This spec **becomes** the survival-world test. Its structure is already right — the null is each capability's own Tier-2 score measured alone, and integration may not cost more than seed noise. Restate the venue as a life in the world and it needs nothing else. |
| T6.05 | Companion battery | **ADAPT** | Gains the **social need** (company as a drive, not only as a safety constraint) and the requirement that the owner interacts with a *living* Jack mid-life rather than with a session. Its four legs survive verbatim. |

### 3.8 PG — playground fixtures (8) · KEEP 7 · ADAPT 1

| id | title | verdict | reason |
|---|---|---|---|
| PG.1 | Playground generates and is physically sound | KEEP | The fixture pattern the world specs should copy. |
| PG.2 | Water works: buoyancy + drag | KEEP | Unchanged, and it acquires two survival jobs (drinking, hypothermia) that need no change to the buoyancy claim. |
| PG.3 | Ladder is climbable in principle (adhesion hands) | **ADAPT** | Its own docstring calls the climber "a certification jig, not a humanoid". PG.8 has since put a real 17-actuator humanoid in the world. Re-run with Jack, or state the adhesion actuator as a permanent scope exclusion — the survival world will otherwise inherit a grasping model nobody has tested. |
| PG.4 | Noisy-TV panel traps naive curiosity | KEEP | Mandatory reporting fixture; unchanged. (See T2.09 for the needs-off caveat, which is a change to the *claims*, not to the fixture.) |
| PG.5 | Procedural contact audio with localization | KEEP | |
| PG.6 | The playground has eyes | KEEP | **Promoted** — it is the "see" fixture and the prerequisite for the owner watching. But its premise is wrong on this box: see §6.2. |
| PG.7 | The heard-not-seen fixture leaks nothing | KEEP | |
| PG.8 | Jack is IN the playground and can act | KEEP | The current frontier and the correct one. |

### 3.9 UB — unified brain (16) · KEEP 10 · ADAPT 2 · OBSOLETE 4

The four OBSOLETEs are all supersession by the later, better-designed UB.9–UB.16
block written on 2026-08-09 — a pre-existing redundancy the new direction merely
makes expensive to carry. In each case the *architecture* half is subsumed by
UB.10's arm set and the *action-relevance* half by UB.15/UB.16.

| id | title | verdict | reason |
|---|---|---|---|
| UB.1 | No modality collapse (ablation matrix) | **OBSOLETE** | UB.11 is the same claim with a placebo column and the swap primitive, and it is reachable while UB.1 is not (§6.1). |
| UB.2 | The shared trunk beats late fusion | **OBSOLETE** | Exactly UB.10 arm **A0 vs A1**, at matched params, matched tokens, matched steps, paired seeds. UB.10 strictly dominates. |
| UB.3 | Cross-modal masking helps the policy | **OBSOLETE** | Architecture half is UB.10 arm **A3**; the "helps the policy" half is UB.16 + UB.15. Carrying UB.3 separately re-litigates A3 with a weaker design. |
| UB.4 | Hearing is load-bearing | **ADAPT** | Overlaps UB.11's audio row and UB.15. Under the survival direction hearing acquires a *survival* function (water, weather, another creature) which the current framing ("turns toward a falling object") does not cover. Restate as the audio row of the standing matrix plus one survival-audio task, or fold into UB.11/UB.15. |
| UB.5 | Touch is load-bearing (or honestly redundant) | **ADAPT** | Touch becomes the **thermoreceptive** channel — contact with cold ground, wind, water is how a body knows it is cold. The current framing (blind push-recovery, honestly likely redundant with proprioception) tests the one job touch is worst at. Add the thermal channel and the honest-redundancy risk mostly disappears. |
| UB.6 | Contrastive binding: keep only if it moves action | **OBSOLETE** | UB.10 arm **A4** plus UB.13 (the retrieval gate that makes an A4 null result interpretable). |
| UB.7 | UNISON — the headline claim | KEEP | Still the gate on the sentence "his senses work in unison". |
| UB.8 | Flow-head attention ablation | KEEP | Cheap; contingent on the flow head surviving §4.4. |
| UB.9 | Heard, not seen | KEEP | **Promoted to frontier.** CPU-only, no controller, one bit of pure synergy, and — verified in §6.3 — the only branch of the unison claim that is not blocked by locomotion. |
| UB.10 | Fusion bakeoff: six arms | KEEP | |
| UB.11 | The modality ablation matrix (standing) | KEEP | Should absorb **interoception** as a sense with its own row and its own placebo comparison — the drive vector must earn its channels like every other input. |
| UB.12 | Synergy, not redundancy | KEEP | |
| UB.13 | Cross-modal retrieval: the gate, never the claim | KEEP | |
| UB.14 | Cross-modal prediction vs the null that usually wins | KEEP | Runnable today (deps: PG.1 PASS). |
| UB.15 | Heard, not seen — embodied | KEEP | |
| UB.16 | Sensory information reaches the controller (D1-agnostic) | KEEP | **Promoted.** Deliberately written to hold under either D1 outcome; it is the only spec that certifies the trunk→controller channel without pre-judging D1. |

### 3.10 CU — curiosity (7) · KEEP 1 · ADAPT 6

This family is the one the directive most directly re-aims, and the required
change is the same in every case: **curiosity is no longer the only driver, so a
curiosity claim needs a needs-only arm to beat and a needs-off condition to be
measured cleanly in.** `PURPOSE_AND_SCAFFOLDING.md` (specs PS.03/PS.04,
researched, **not registered**) already designs that bakeoff; these specs should
be re-parented onto it rather than re-deriving it.

| id | title | verdict | reason |
|---|---|---|---|
| CU.1 | Goal babbling beats action babbling | **ADAPT** | The outcome space must include need-relevant outcomes (core temp, satiety) or the coverage metric will score him highly for exploring things that cannot keep him alive. |
| CU.2 | LP produces an emergent curriculum | **ADAPT** | The predicted ordering (stand → walk → push → ramp) becomes need-shaped (drink → warm → shelter). The *ordering* claim is what matters and it survives; the named sequence must be pre-registered against the new world before running, not after. |
| CU.3 | Curious without being trapped | **ADAPT** | Must be run **needs-off**, else starvation supplies the immunity and the LP stack gets the credit. Same defect as T2.09 and it is worth fixing once, in the fixture. |
| CU.4 | Unsupervised skills are real and distilled | KEEP | METRA on trunk embeddings is venue-independent. |
| CU.5 | The VLM proposes, LP disposes | **ADAPT** | The proposer must be told, or deliberately not told, about need state; "climb the ladder" and "you are freezing, get inside" are different classes of proposal and the scrambled-caption control does not separate them. |
| CU.6 | Affordances emerge from interaction | **ADAPT** | Extend the held-out affordance set from pushable/liftable to **edible / drinkable / warm / shelter**. Strictly a strengthening — those are the affordances that decide whether he lives. |
| CU.7 | Lessons from failure improve retries | **ADAPT** | "Retry" becomes "**next life**". With T2.20, this is the second existing spec that becomes a cross-life learning test by changing its unit. |

### 3.11 ME — memory (18) · KEEP 17 · ADAPT 1

The memory family is the part of the ladder the new direction validates most
completely: `GOAL.md` says explicitly that "the ME family already proves the
substrate" for what survives death.

| id | title | verdict | reason |
|---|---|---|---|
| ME.1 | Event log: what happened is retrievable | KEEP | PASS. |
| ME.2 | Owner memory lives on disk | KEEP | PASS. |
| ME.3 | Reflections beat raw events | KEEP | PASS. |
| ME.4 | Forgetting keeps what matters | KEEP | PASS. |
| ME.5 | Retrieval survives growth | KEEP | PASS, standing. Binding constraint under D-c: a multi-hour life generates events at a rate no current fixture approaches. Re-run at the decade a real life actually produces. |
| ME.6 | Skill library accelerates composites | KEEP | |
| ME.7 | Sleep consolidation (SIESTA) holds old knowledge | **ADAPT** | Same change as T5.05: the wake/sleep boundary becomes the world's night, and the rehearsal buffer is the life's diary. The emptied-buffer control carries over unchanged. |
| ME.8 | Working memory survives restarts | KEEP | PASS. |
| ME.9 | He remembers what he hears, says, and does | KEEP | PASS. |
| ME.10 | Keeps the memory AND learns the general skill | KEEP | PASS. **Promoted:** the double dissociation *is* the mechanism by which "what survives death is the point" is falsifiable. Nothing new needs inventing for D-b's memory half. |
| ME.11 | Paraphrase recall, never invents one | KEEP | |
| ME.11.0 | The eval set is honest before anyone is scored | KEEP | PASS. |
| ME.11.A–F | The six retrieval arms | KEEP (6) | Direction-neutral, CPU-only, and the largest block of *immediately runnable* work in the registry (§6.4). |

### 3.12 Counts

| verdict | count | share |
|---|---:|---:|
| KEEP | **84** | 66% |
| ADAPT | **35** | 27% |
| OBSOLETE | **7** | 5% |
| BLOCKED-ON-DIRECTION | **2** | 2% |
| **total** | **128** | |

By family:

| family | n | KEEP | ADAPT | OBSOLETE | BLOCKED |
|---|---:|---:|---:|---:|---:|
| T0 harness | 16 | 14 | 2 | – | – |
| T1 primitives | 13 | 12 | 1 | – | – |
| T2 competence | 21 | 10 | 11 | – | – |
| T3 ablation | 10 | 9 | 1 | – | – |
| T4 composition | 5 | 1 | 1 | 3 | – |
| T5 claims | 9 | 2 | 5 | – | 2 |
| T6 living Jack | 5 | 1 | 4 | – | – |
| PG playground | 8 | 7 | 1 | – | – |
| UB unified brain | 16 | 10 | 2 | 4 | – |
| CU curiosity | 7 | 1 | 6 | – | – |
| ME memory | 18 | 17 | 1 | – | – |

**The headline:** 93% of the ladder survives the re-aim (KEEP + ADAPT), and of
the 7 obsoletions, **zero were caused by the survival direction** — all seven are
supersession by better specs already in the registry. The direction costs the
project two specs (T5.01/T5.02, and only pending a bakeoff), not a rebuild.

---

## 4. Architecture decisions

For each: does the survival direction **strengthen**, **weaken**, or **reframe**
it? "Reframe" means the decision still has to be made but the question changed.

### 4.1 D1 — does the 57M trunk stay in the control path? · **REFRAMED, and a new axis arrives**

Status: OPEN, and `DECISIONS_NEEDED.md` correctly says **do not decide on the
current evidence** — the three runs that favoured the MLP were confounded by
live dropout (42% policy-mean drift on an identical state), by 16× unmatched
optimiser steps, by a 28-column zero pad, and by a plateau claim resting on
`curve_seed0[:8]` of 172 iterations.

The survival direction **adds a decision axis that the existing evidence does not
contain and that does not require the re-run to measure**. Measured on this box
today (§0):

| control path | env-steps/s | sim-seconds per real second | real time for a 1-hour life |
|---|---:|---:|---:|
| 57M trunk (T0.07 ledger) | 11.48 | 0.17 | **5.8 hours** |
| ~160K MLP (measured now) | 1531.6 | 22.97 | **2.6 minutes** |
| zero action (physics ceiling) | 2105.9 | 31.6 | 1.9 minutes |

**133×.** Under D-c, a 3-seed × 3-life × 1-sim-hour spec costs 23 minutes with
the small head and **52 hours** with the trunk — the latter exceeding
`run.py`'s own `cpu<2h × 3 seeds × 2` ceiling of 15 hours, on a box that also
serves paying tenants and whose builder loop is capped at 50 minutes per
iteration.

This does not decide D1 and must not be presented as if it did — it is a cost
measurement, not a competence measurement, and option A must still be *earned*
by the re-run/bakeoff exactly as `DECISIONS_NEEDED.md` insists. But it does
change what D1 is a decision *about*. It was "which architecture controls
better". It is now "which architecture controls better **and permits the world
the owner asked for**", and those two questions can have different answers. If
the re-run shows the trunk winning on return, D1 becomes a genuine trade rather
than a formality, and the trade should be stated in sim-hours-per-real-hour.

**Recommendation for the audit trail, not a decision:** the re-run should carry
the real-time factor as a pre-registered, gated metric (the T2.02 ADAPT in
§3.3), and `T2.21`/`D1.0` (designed in `D1_CONTROL_ARCHITECTURE.md`, **not
registered**) should be preferred over a bare T2.01/T2.02 re-run, because they
answer *where the trunk belongs* rather than only *whether it learned*.

### 4.2 The frozen swappable LLM · **STRENGTHENED as a principle, WEAKENED as implemented**

The principle — frozen pretrained towers, a small trained core, swap as better
models ship — is exactly the owner's "the system is the product, not the model",
and the survival direction strengthens it: a system that must run for hours
cannot afford to retrain a tower.

The *implementation* is indicted by the same arithmetic. T0.07 measured SmolLM2-
1.7B at **6.9 GB resident, 1.767B of the process's 1.767B total parameters,
55.9M trainable, and `llm_removal_speedup` = 1.03** — i.e. it consumes almost
all the memory and contributes **0.0%** of rollout time because it never runs in
`forward()`. In a 1000-step 15-second episode that is merely wasteful. In a
multi-hour life on a 4-core shared box it is disqualifying.

Nothing changes about the decision; what changes is its urgency. Out-of-process
dialogue (the "D-dialogue" decision T0.07's docstring already names) moves from
a tidy-up to a prerequisite. T1.05 ("frozen stays frozen") is the spec that
enforces the principle and stays exactly as written.

### 4.3 Extractive-never-generative memory · **STRENGTHENED, unambiguously**

ME.11's rule — what Jack reports about his past must be a literal stored record
or nothing, because a generator cannot abstain honestly — becomes *more*
important under D-b, not less. Cross-life learning is a claim **about the
record**: "life N+1 is better because of what life N recorded" is only
falsifiable if the record cannot be confabulated. A generative memory would make
every cross-life result unfalsifiable in exactly the way the project's original
disease was unfalsifiable.

The one thing to watch: in-world language ("I'm cold") is *generative* output.
The rule must be stated as scoped to **memory reports**, not to speech, or the
social/language gap specs in §5 will appear to violate a project law. The
separating principle is already available: speech about *current state* is
grounded in a live variable and is checkable against it (SO.02 in §5); speech
about the *past* must quote the diary.

### 4.4 The flow-matching action head · **REFRAMED, and now carries a cost argument against it**

The justification is T2.19: on genuinely bimodal tasks (pass left or right) a
flow head succeeds where MSE regression collapses to the mean, and the spec
honestly records that OFT found L1 ties on some benchmarks — "genuine
falsification risk". That test is unchanged by the direction and should be run.

What the direction adds is a **second cost the original decision did not price**:
flow matching samples by integrating 10 Euler steps (T1.12's note), so the
action head costs ~10 forwards per control step. Under D-c that multiplies the
number in §4.1 by up to an order of magnitude on whichever trunk is chosen. This
makes T2.18 (action chunking) not an optimisation but a **precondition** — a
chunk of length *k* amortises the integration over *k* control steps — and it
means T2.19 and T2.18 should be run as a pair, with the flow head's advantage
reported per unit of compute rather than per step.

### 4.5 The 57M UnifiedBrain trunk itself · **WEAKENED by the direction, independently of D1**

D1 asks where the trunk sits. This asks whether 57M is the right size at all,
and it is a separate question that the owner has now answered in principle:
*"at the end of the day it won't be the most complex model that Jack is."*

Three measured facts bear on it, none of which is the confounded locomotion
comparison:

1. **Throughput** (§4.1): 133× against a 160K MLP, in the unit that decides
   whether hours-long lives exist.
2. **Resident cost**: 6.9 GB (T0.07 `policy_peak_rss_mb` = 6933.8) on a box
   whose loop refuses to start below a free-memory floor and which serves paying
   tenants.
3. **Dead weight**: T1.03's own null baseline records **45,538,295 parameters
   (38.6%) receiving no gradient**, and T3.03/T3.04/T3.05 name 2.97M + 37.17M +
   12.64M of components with zero or near-zero call sites.

The honest statement is that the trunk has never been shown to be *needed* for
anything, and the direction raises the price of keeping it while lowering the
tolerance for unearned parameters. The correct response is not to delete it —
that is the owner's call and UB.10/UB.11/UB.16 are precisely the specs that
would earn or condemn it — but to note that **Tier 3 has 0 of 12 PASSes** and
that this is now the tier with the highest expected value per GPU-hour.

### 4.6 EpisodicMemory's design · **STRENGTHENED, with one measured weakness already on the record**

The two-store design (verbatim diary + distilled skill, double-dissociated by
ME.10) is exactly what D-b requires and it already PASSes. `EpisodicMemory`'s
abstention behaviour — 100% rejection of fabricated events — is what makes the
cross-life claim credible.

The known weakness is measured and pre-registered rather than hidden: the
shipped lexical-containment retriever scores **0/8 on paraphrase cues** and
ME.11.0 PASSes with the incumbent at 0.000. Under D-b the diary is queried
across lives by an agent whose vocabulary has drifted, which is the paraphrase
case, so the ME.11 bakeoff moves from "nice retrieval upgrade" to "the mechanism
cross-life learning runs through". It is also entirely CPU and entirely
unblocked (§6.4).

One structural note: `ME.5` is a standing spec at store decades 10²→10⁵. A
multi-hour life at 66.7 control steps per second will cross those decades far
faster than the current fixtures assume; the event *rate*, not just the count,
needs stating.

### 4.7 The bakeoff / VOID / overseer machinery · **STRENGTHENED, and it is the part of the system the owner is actually buying**

The owner's "we build tests, throw him in, get results, build bigger tests,
throw him in again" *is* this machinery. Everything in it survives:

- **The learning gate** (an arm below 3σ cannot arbitrate) becomes more, not
  less, necessary — a survival world produces many ways for an arm to fail to
  learn while still producing motion.
- **`controls=` as a separate parameter** (a designed-to-fail control is not a
  weak arm) is required by every needs spec: the no-needs null must be scored on
  the same ruler without VOIDing the bakeoff.
- **VOID vs FAIL** matters more when runs are long: a 6-hour life that ends
  because the world was misconfigured must not fire a `kills` field.
- **The overseer** is the only independent check on a system whose runs are
  about to get much longer and much harder to eyeball.

Two open items the direction makes more expensive to leave open:

- **D2 (does a VOID dependency block its dependents?)** — currently code blocks
  and the docstring says it does not. With 40 specs transitively blocked behind
  two VOIDs (§6.1), this is no longer a documentation inconsistency; it is the
  single switch that decides how much of the ladder is runnable.
- **`bakeoff.py` writes to the real `DECISIONS_RESOLVED.md` from its own unit
  tests** (six `TEST` entries with invented arms). The first *real* decision —
  which will be a survival-world decision — will land in a file a reader has
  already learned to distrust.

### 4.8 Smaller decisions, briefly

| decision | verdict | note |
|---|---|---|
| Learning progress as the curiosity signal (no raw ICM/RND rewards) | **REFRAMED** | LP competes with needs now. `PURPOSE_AND_SCAFFOLDING.md` §2.8 already designs the interaction; CU.2/CU.3 must be re-parented onto it. |
| VLM proposes, LP disposes | **REFRAMED** | The proposer needs a stated position on need-awareness (CU.5 ADAPT). |
| Hindsight goal-conditioned regression as the learning rule | **STRENGTHENED** | "Every failure is a success at what it did achieve" is the natural learning rule for an agent that dies a lot. Feeds the learning-core bakeoff as a leading arm. |
| All memory as plain files on this box | **STRENGTHENED** | Cross-life persistence is trivially inspectable and trivially wipeable, which is what makes ME.10's double dissociation cheap. |
| Free compute only | **STRENGTHENED into a design constraint** | It is now the binding constraint on world fidelity, not merely on training. §4.1 is a free-compute argument. |
| The playground as procedural MJCF | **KEEP** | PG.1–PG.8 give a working, physically-audited, mutable world with a real humanoid in it. Replacing it with a heavier platform would restart eight passing fixtures for fidelity the box cannot render (§6.2). |
| Modality dropout + cross-modal masking as the binding objective | **REFRAMED** | Now arms A2/A3 of UB.10 rather than standing commitments; interoception joins as a modality. |
| The episode as the harness's unit | **WEAKENED — this is the one piece of machinery the direction breaks** | See §6.5. |
| The ledger as sole scoreboard | **STRENGTHENED** | Unchanged and non-negotiable. |

---

## 5. The gap list — what the direction requires that no spec covers

Draft `Spec(...)` stubs, **id suggestions only, no registry edits**. Two-digit
ids throughout (verified in §0 against the `_module_for` prefix hazard).

Three families are proposed — `WP.*` world physics, `LF.*` life/death, `SO.*`
social and spectating — plus two Tier-0 harness specs. Where an existing but
**unregistered** research spec already covers a gap, it is named and **not
duplicated**: `PS.00–PS.06` (`PURPOSE_AND_SCAFFOLDING.md`) own the
needs-vs-no-needs question and the anti-gaming detectors; `LT.01–LT.09`
(`CURIOSITY_BAKEOFF.md`) own the ladder test; `HR.1–HR.8` (`HEARING_BAKEOFF.md`)
own hearing; `LG.00`/`LG.05` own language-necessity; `D1.0`/`T2.21` own the D1
bakeoff. **That is 28 researched specs sitting outside the registry** — see
§6.6.

### 5.1 World physics with analytic gates (`WP.*`) — tier 2, CPU

The PG family's discipline is the model: assert against a closed-form answer, so
the fixture cannot be graded by the thing it grades. PG.2's Archimedes depth and
PG.1's `tan θ > μ` are the precedent.

```python
Spec("WP.01", 2, "Body temperature obeys a lumped-capacitance model",
     hypothesis="Core temperature tracks the analytic solution of a "
                "lumped-capacitance body exchanging heat with air, ground "
                "contact, wind and water within 5% over a 30-minute sim, and "
                "the two lethal boundaries are reachable from the spawn state "
                "in finite time in both directions.",
     falsified_by="Core temp diverges from the closed-form curve by >5%, OR "
                  "either lethal boundary is unreachable (a need that cannot "
                  "kill is decoration), OR temperature depends on a quantity "
                  "with the wrong units (PG.2's radius-from-inertia bug).",
     null_baseline="A constant-temperature body — must fail every check.",
     metric="thermal_model_error", budget=Budget.CPU, seeds=3,
     control="Remove the water/wind coupling: immersion must then NOT cool "
             "him. If cooling survives the ablation it is coming from the "
             "integrator, not the world.",
     kills="Every cold-night claim. A shelter result in a world with broken "
           "thermodynamics measures the integrator.",
     notes="Time-average, do not sample: LESSONS 'time-average anything that "
           "oscillates'. Derive convection from geometry, never from a "
           "quantity that happens to have the right units."),

Spec("WP.02", 2, "Metabolism: energy, hunger and thirst are conserved quantities",
     hypothesis="Energy expenditure equals mechanical work plus a basal rate "
                "to within 3%; eating and drinking move satiety by the "
                "declared amount; and no action sequence increases stored "
                "energy without consuming a world resource.",
     falsified_by="A closed loop of actions that nets positive energy "
                  "(perpetual motion), OR satiety changing with no eat event.",
     null_baseline="A statue: basal drain only, monotone decline to death.",
     metric="energy_conservation_error", budget=Budget.CPU, seeds=3,
     control="A deliberately leaky variant (work not charged) must be caught "
             "by the same conservation check.",
     kills="Every efficiency claim. 'He learned the efficient way' is "
           "meaningless if the ledger of energy does not balance.",
     notes="Overlaps PS.01's drive-range assertion deliberately: PS.01 asks "
           "whether the drive is a real control problem, this asks whether "
           "the bookkeeping is sound. Run this FIRST; it is cheaper."),

Spec("WP.03", 2, "Day, night and weather are real inputs, not a clock variable",
     hypothesis="The diurnal cycle changes at least three measurable world "
                "quantities (illumination at the camera, ambient temperature, "
                "audio floor), each recoverable by a probe from sensors alone.",
     falsified_by="A probe recovers time-of-day from a channel that should "
                  "not carry it, OR cannot recover it from any channel — "
                  "either the clock leaks or the cycle is cosmetic.",
     null_baseline="Constant-daylight world.",
     metric="diurnal_probe_recovery", budget=Budget.CPU, seeds=3,
     control="Freeze the sun and keep the clock running: probes must fall to "
             "chance. Otherwise the probe reads the clock, not the world.",
     kills="Sleep-need claims and any 'cold nights teach shelter' result."),

Spec("WP.04", 2, "Resources are finite, located, and depletable",
     hypothesis="Food and water sources deplete on use, replenish on a stated "
                "schedule, and their positions are recoverable from vision "
                "but not from proprioception.",
     falsified_by="Infinite resources, OR position recoverable without "
                  "perception (the location leaked into the state vector).",
     null_baseline="Unlimited resources at the spawn point — must make every "
                   "foraging metric trivial.",
     metric="resource_leak_margin", budget=Budget.CPU, seeds=3,
     control="A blind agent must forage at chance. If it does not, the "
             "resource position is in the observation.",
     kills="Every foraging and efficiency claim."),
```

### 5.2 Needs-driven behaviour vs the no-needs null — **already designed, not registered**

`PURPOSE_AND_SCAFFOLDING.md` owns this and should be registered rather than
re-derived: **PS.00** (the scaffolding dilemma in tabular form, 2 CPU-minutes),
**PS.01** (the drive layer is a real control problem, a statue loses), **PS.02**
(the anti-gaming detectors see their own positive controls), **PS.03**
(screening: which purpose signal produces competence at all), **PS.04** (bakeoff:
does a purpose beat curiosity), **PS.05** (competence survives the drive that
produced it), **PS.06** (does he need FOOD, or just a cost of failing?).

That document also contains the single most important theoretical result for
this direction, and it should not be lost in a registry entry: *a homeostatic
drive reward that is made provably removable provably cannot create purpose* —
the useful part of a drive is exactly its non-potential-based component.

**One gap PS.\* does not cover**, because it was written before the world was
declared lethal:

```python
Spec("PS.07", 5, "Needs beat no-needs at LEARNING, not only at surviving",
     hypothesis="An agent with needs acquires transferable competence "
                "(measured on a needs-OFF held-out task battery) faster than "
                "a matched no-needs curiosity-only agent at equal env-steps.",
     falsified_by="The no-needs agent matches it on the needs-off battery — "
                  "then needs bought survival behaviour, not learning, and "
                  "GOAL.md's 'the needs ARE the curriculum' is wrong.",
     null_baseline="Curiosity-only agent, matched steps and params. AND a "
                   "hand-written task-reward agent — the arm nobody likes and "
                   "everybody must beat.",
     metric="needs_off_transfer", budget=Budget.GPU, seeds=3,
     control="Needs whose variables are DECOUPLED from the world (they drift "
             "on a timer, eating does nothing) must give no advantage. If "
             "they do, the benefit is reward shaping, not embodiment.",
     kills="The whole survival-world premise, if the null wins. This is the "
           "spec that could prove the direction wrong and it must exist."),
```

### 5.3 Death, retry, and cross-life learning (`LF.*`)

The two cheapest entries here are **not new specs** — they are T2.20 and CU.7
with their unit changed from episode to life (§3.3, §3.10). What follows is what
those two cannot cover.

```python
Spec("LF.01", 2, "A life runs to its natural end, and the harness survives it",
     hypothesis="A single life of >=1 simulated hour (>=240,000 control "
                "steps) completes at a stated real-time factor with bounded "
                "memory, bounded diary growth, and no non-finite state; and "
                "the run's DEATH is recorded distinctly from a CRASH.",
     falsified_by="Unbounded memory or diary growth, non-finite state, OR a "
                  "death indistinguishable from a crash in the record.",
     null_baseline="The current 1000-step episode = 15 simulated seconds. The "
                   "claim is a 240x extension, and what breaks at 240x is the "
                   "finding.",
     metric="life_completion_and_rtf", budget=Budget.CPU_LONG, seeds=3,
     control="Inject a NaN mid-life: it must be reported as a crash, never as "
             "a death. A harness that cannot tell them apart will score every "
             "bug as mortality.",
     kills="Every hours-long claim, and the survival world's schedule.",
     notes="Measured 2026-08-09: 160K MLP gives 23.0 sim-s per real s (1 "
           "sim-hour in 2.6 min); the 57M trunk gives 0.17 (5.8 h). At 3 "
           "seeds the latter exceeds run.py's own 15-hour ceiling for "
           "cpu<2h. The real-time factor is therefore a GATE, not a note."),

Spec("LF.02", 2, "A life can be saved and resumed — world, needs, diary, working memory",
     hypothesis="Killing a process mid-life and resuming restores the world "
                "state, the need variables, the episodic store and the "
                "recurrent state such that the continuation is "
                "indistinguishable from the uninterrupted run over the next "
                "1000 steps.",
     falsified_by="Any of the four stores lost or silently defaulted; a "
                   "resumed trajectory that diverges beyond float tolerance.",
     null_baseline="Weights-only resume — T0.04's null, one level up.",
     metric="life_resume_fidelity", budget=Budget.CPU, seeds=3,
     control="Corrupt (truncate) each store in turn: load must FAIL LOUDLY, "
             "never silently default. T6.03's byteflip control is the model.",
     kills="Multi-session lives, hence every life longer than one Kaggle "
           "session or one 50-minute loop iteration."),

Spec("LF.03", 5, "Life N+1 is better BECAUSE of what life N recorded",
     hypothesis="Across a sequence of lives, survival time rises, and the "
                "rise disappears when the diary is wiped between lives while "
                "the weights are kept, AND when the weights are reverted "
                "while the diary is kept the rise is only partly removed.",
     falsified_by="Survival time flat across lives, OR the wipe-the-diary arm "
                   "improving just as fast (the learning was never in the "
                   "record).",
     null_baseline="Diary wiped between lives; a fresh agent each life.",
     metric="cross_life_improvement", budget=Budget.GPU, seeds=3,
     control="SHUFFLE the diary between lives (give him another life's "
             "record). Improvement must collapse — otherwise the benefit is "
             "having any text at all, not having HIS text.",
     kills="'Death is a page turn, not a reset.' This is the spec that claim "
           "stands or falls on.",
     notes="The double dissociation is ME.10's, applied across lives instead "
           "of across a distillation step. ME.10 already PASSes, so the "
           "substrate exists; this is the cross-life instance of it."),

Spec("LF.04", 5, "Sleep is gated by the world, and consolidation happens in it",
     hypothesis="Sleep pressure rises with wake time, sleeping restores it, "
                "and the consolidation that happens during in-world sleep "
                "produces better old-concept retention than the same "
                "consolidation run at an arbitrary wall-clock moment.",
     falsified_by="Consolidation timing makes no difference — then sleep is "
                  "an offline GPU pass with a costume on, and ME.7/T5.05 "
                  "should keep their current framing.",
     null_baseline="Wake-only forever; consolidation at random times.",
     metric="sleep_gated_retention", budget=Budget.GPU, seeds=3,
     control="Sleep with the rehearsal buffer emptied must forget (ME.7's "
             "control, carried forward), AND sleeping while unsafe must cost "
             "survival time — a need with no downside is not a need.",
     kills="The claim that sleep is a NEED rather than a scheduler."),

Spec("LF.05", 5, "The world grows with him across lives",
     hypothesis="Mutating the world between lives toward the frontier of what "
                "he can just barely survive produces more distinct mastered "
                "outcomes than a fixed world at equal total steps.",
     falsified_by="Fixed world keeps pace.",
     null_baseline="Fixed world, same total steps (T5.08's null, per life).",
     metric="cross_life_cluster_growth", budget=Budget.GPU_LONG, seeds=3,
     control="Mutation WITHOUT the learnability filter must degenerate into "
             "unsurvivable worlds — else the filter does nothing.",
     notes="PlaygroundParams.mutate() and PG.8's world_mutated metric already "
           "exist. This is T5.08 with the life as the unit; register whichever "
           "of the two the owner prefers, not both."),
```

### 5.4 Social need, in-world language, and the owner as participant (`SO.*`)

**Nothing in 128 specs touches spectating or live interaction.** This is the
largest true blank in the audit.

```python
Spec("SO.01", 2, "Jack can be watched: a third-person stream exists and costs what we say it costs",
     hypothesis="A third-person view of a running life renders at >=5 fps at "
                "320x240 on this box and is deliverable to the owner without "
                "a persistent listening service; the measured render cost is "
                "reported as a fraction of the life's compute budget.",
     falsified_by="Rendering unavailable, OR the render cost pushing the "
                  "real-time factor below 1.0 (watching would then be slower "
                  "than living).",
     null_baseline="No rendering — the current state, in which no spec has "
                   "ever produced a frame.",
     metric="stream_fps_and_cost_share", budget=Budget.CPU, seeds=3,
     control="Render a scene with the humanoid removed: the frame must "
             "measurably change. A renderer that produces identical frames is "
             "producing a background.",
     kills="'I want to watch him figure out the world himself.'",
     notes="MEASURED 2026-08-09 on this box: MUJOCO_GL=osmesa and =egl both "
           "FAIL (no libOSMesa, no libEGL). The working path is `xvfb-run -a "
           "-s '-screen 0 640x480x24' MUJOCO_GL=glfw` on swrast/llvmpipe: "
           "14.6 fps at 128x128 (68 ms/frame), 5.4 fps at 320x320 (185 ms). "
           "One 128x128 frame costs ~104 env-steps of compute. PG.6's note "
           "('render on CPU via MUJOCO_GL=osmesa') is FALSE on this box and "
           "should be corrected. A persistent stream is a background service "
           "on a tenant-serving box and is an OWNER decision, not an "
           "engineering one."),

Spec("SO.02", 2, "'I'm cold' is true when he is cold",
     hypothesis="Jack's utterances about his own internal state are "
                "predictive of that state: a listener with only the "
                "utterances recovers the need variable above a "
                "base-rate-matched null, and utterances do not fire when the "
                "variable is nominal.",
     falsified_by="Utterances uncorrelated with the variable, OR correlated "
                  "but always firing (a thermostat that is always on is not "
                  "communicating).",
     null_baseline="Utterances sampled from the marginal distribution at the "
                   "same rate — the fluency null.",
     metric="utterance_state_mutual_information", budget=Budget.CPU, seeds=3,
     control="Freeze the need variable and let him talk: the recovered signal "
             "must collapse. Also swap the labels between two need channels "
             "('cold' emitted for hunger): a listener must be misled, "
             "proving the channel is read and not merely present.",
     kills="Language grounded in state, as distinct from language that "
           "pattern-matches a situation.",
     notes="This is the ONE place a generative channel is legitimate under "
           "ME.11's extractive-never-generative law, because the claim is "
           "checkable against a live variable. Speech about the PAST must "
           "still quote the diary."),

Spec("SO.03", 2, "Company is a need, and isolation costs something",
     hypothesis="A social variable depletes in isolation, is restored by "
                "interaction, and its depletion measurably changes behaviour "
                "(approach rate to the owner avatar) beyond a no-social-need "
                "twin.",
     falsified_by="Behaviour identical with the variable clamped — the need "
                  "is decorative.",
     null_baseline="No-social-need twin, matched everything else.",
     metric="social_need_behaviour_delta", budget=Budget.CPU_LONG, seeds=3,
     control="A non-responsive avatar (present but inert) must NOT restore "
             "the variable. Otherwise 'company' is proximity.",
     kills="'He needs company' as a claim rather than a sentence."),

Spec("SO.04", 6, "Being watched does not change him",
     hypothesis="Behaviour statistics over a life are indistinguishable "
                "between a rendered/streamed run and an unrendered one at the "
                "same seed, and the rendered run's trajectory matches the "
                "unrendered one bit-for-bit until the first stochastic draw.",
     falsified_by="Any behavioural divergence attributable to the observer "
                  "path — a spectator that perturbs the physics, the RNG "
                  "stream, or the timing.",
     null_baseline="n/a — an invariant of the composition.",
     metric="observer_divergence", budget=Budget.CPU, seeds=3,
     control="Deliberately draw one RNG value in the render path: the "
             "detector MUST catch it. Otherwise it cannot see its own "
             "positive control.",
     kills="Any claim measured while the owner was watching, which under this "
           "direction will eventually be most of them.",
     notes="LESSONS: 'a detector that cannot see its own positive control has "
           "measured nothing.' The render path consuming RNG is the exact "
           "failure mode and it is easy to introduce by accident."),

Spec("SO.05", 6, "He can be interrupted mid-life and go back to what he was doing",
     hypothesis="An owner utterance mid-life is heard, answered, recorded "
                "with attribution (ME.9's channels), and the interrupted "
                "activity resumes; need variables continue to evolve during "
                "the exchange.",
     falsified_by="The world pausing for conversation, OR the activity not "
                  "resuming, OR the exchange missing from the diary.",
     null_baseline="A session-based companion: conversation and living are "
                   "separate modes. That is the current design.",
     metric="interruption_resume_rate", budget=Budget.CPU_LONG, seeds=3,
     control="Deliver the utterance to a Jack with the audio path muted: no "
             "response, and the diary must record nothing. If it records "
             "something, the text reached him out of band.",
     kills="'Users can talk to him while he is there doing stuff.'"),
```

### 5.5 Harness gaps (`T0.*`)

```python
Spec("T0.17", 0, "The real-time factor is measured, recorded, and gates long runs",
     hypothesis="For any declared control path, the harness measures "
                "sim-seconds per real second before a long run starts, and "
                "REFUSES a run whose projected duration exceeds the spec's "
                "timeout or the box's tenant-safety budget.",
     falsified_by="A long run launching with a projected duration past its "
                  "own timeout, OR a projection that differs from the "
                  "achieved duration by >25%.",
     null_baseline="Today: nothing measures this and run.py's timeout is "
                   "budget x seeds x 2 regardless of what the run will cost.",
     metric="rtf_projection_error", budget=Budget.CPU, seeds=1,
     control="A deliberately slow policy must be REFUSED. A gate that never "
             "refuses is decorative (T0.13's own rule).",
     kills="Nothing directly; it prevents burning a Sunday quota on a run "
           "that could never have finished.",
     notes="The numbers this spec would have surfaced, measured 2026-08-09: "
           "57M trunk 0.17 sim-s/real-s; 160K MLP 23.0. A 3-seed 1-sim-hour "
           "spec costs 52 h with the former against a 15 h ceiling."),

Spec("T0.18", 0, "CPU-hours on a shared box are accounted like GPU-hours",
     hypothesis="Every CPU_LONG run debits a wall-clock budget, and the "
                "ladder refuses to start when the box's load or the day's "
                "accumulated share would harm the tenants.",
     falsified_by="A run proceeding past the budget, or a budget that reads "
                  "the same whether or not runs happened.",
     null_baseline="Today: only GPU hours are tracked (T0.12); the loop "
                   "checks instantaneous load once, at start.",
     metric="cpu_quota_enforced", budget=Budget.CPU, seeds=1,
     control="A leaky accountant must FAIL isolation, and the assertion must "
             "be made at a MID-RANGE value, not at an exhausted one "
             "(T0.12's rewritten form is the template).",
     kills="Nothing on the ladder; it protects the tenants, which SYSTEM.md "
           "ranks above the ladder."),
```

### 5.6 Gaps deliberately NOT proposed

- **A new world platform.** PG.1–PG.8 pass, are physically audited, contain a
  real humanoid, and mutate. Replacing MJCF would restart eight fixtures for
  fidelity this box cannot render.
- **A predator / other agents.** Implied by "the jungle" but not by the owner's
  words, and it multiplies compute by the number of bodies. Escalate rather than
  assume.
- **Anything requiring a resident GPU.** Already a written scope exclusion in
  `MASTER_PLAN.md` and unchanged.

---

## 6. Findings the audit turned up (each verified, §0)

### 6.1 The UB.1 lesson has recurred, and the fix that was reported did not land

`LESSONS.md` records that UB.1 — the project's namesake claim — was dead-ended
behind T2.01, that binding was miscategorised as a CONTROL claim, and that
re-parenting made eight specs runnable. `OVERSIGHT.md` §3.3 then certifies the
fix: *"UB.1 was dead-ended behind T2.01's FAIL and is now parented UB.1 → T4.01 →
T3.01 → T2.03, and T2.03 is NOT_RUN-but-runnable rather than FAILed."*

**T4.01 has two parents, not one.** Traced from the live registry:

```
UB.1 [NOT_RUN] -> T4.01 [NOT_RUN] -> T3.01 -> T2.03 -> T1.08 [PASS]   (clear)
                                  -> T3.02 -> T2.01 [VOID]            (blocked)
```

`blocked_by` returns any dependency not PASSing, so UB.1 is **still unreachable
behind locomotion**. The overseer followed one branch of a two-branch parent and
declared the whole path clear. The lesson's own rule — *"periodically ask which
specs are unreachable and why"* — was applied, and the answer was wrong because
the check looked at a path instead of the closure.

**40 of 128 specs** are transitively blocked behind {T1.02 ERROR, T2.01 VOID,
T2.02 VOID}, computed as a fixed point over `depends_on`:

```
CU.1 CU.2 CU.3 CU.4 CU.5 CU.6 CU.7  ME.7
T2.13 T2.16 T2.17 T2.18  T3.02 T3.04 T3.05  T4.01 T4.04 T4.05
T5.01 T5.02 T5.03 T5.04 T5.05 T5.07 T5.08 T5.09
T6.01 T6.02 T6.04 T6.05
UB.1 UB.2 UB.3 UB.4 UB.5 UB.6 UB.7 UB.8 UB.15 UB.16
```

Note what is *not* in that list and is therefore reachable: `UB.9`–`UB.14`,
`T3.01`, `T3.03`, `T3.06`–`T3.10`, `T4.02`, `T4.03`, `T5.06`, `T6.03`, all of
`PG` and all of `ME` except `ME.7`.

The generalisable rule, which belongs in `LESSONS.md` (not added here — this
document does not edit other files): **an unreachability audit must compute the
transitive closure over ALL parents, and an "it is reachable now" claim must name
the full parent set it checked.** A second, cheaper guard: `run status` should
print, for every non-runnable spec, the *deepest* blocking ancestor, so a
two-parent node cannot hide one of them.

### 6.2 This box cannot render the way PG.6 assumes, and can render the way nothing assumes

PG.6's notes say *"Render on CPU via MUJOCO_GL=osmesa"*. Measured: **there is no
`libOSMesa` and no `libEGL` on this box** (`ctypes.util.find_library` returns
`None` for both; `mujoco.Renderer` raises out of PyOpenGL's loader for each).

There **is** a working path, found and measured: `xvfb-run -a -s "-screen 0
640x480x24"` with `MUJOCO_GL=glfw` over `swrast_dri.so` (llvmpipe):

| resolution | fps | ms/frame | cost in env-steps (160K MLP) |
|---|---:|---:|---:|
| 128×128 | 14.6 | 68 | ~104 |
| 320×320 | 5.4 | 185 | ~283 |

Consequences that should be priced before PG.6, UB.9 or any vision spec is
scheduled:

- **Egocentric vision at control rate is impossible.** 66.7 Hz control against
  14.6 fps maximum render.
- **Vision at 5 Hz is affordable**: 5 × 68 ms = 0.34 s of render per simulated
  second, against 0.044 s of physics+policy — real-time factor falls from 23×
  to ~2.6×, so a sim-hour goes from 2.6 minutes to ~23 minutes. Workable.
- **Spectating is itself a cost above 1×**: 320² at 15 fps is 2.8 s of render
  per simulated second, i.e. **slower than real time**. Watching Jack live will
  need either a lower resolution, a lower frame rate, or a decision to record
  and replay rather than stream.
- `xvfb-run` spawns an X server. It cleans up after the command, so it is fine
  inside a spec run; a *persistent* spectator stream would be a standing
  background service on a box that serves paying tenants, which SYSTEM.md and
  `/home/opc/CLAUDE.md` both place outside the loop's authority.

### 6.3 The unison claim has a locomotion-free route, and it is the shortest path to a Tier-4 result

Verified against the live graph:

```
PG.6 [NOT_RUN, dep PG.1 PASS]  ┐
PG.7 [NOT_RUN, dep PG.5 PASS]  ├─> UB.9 -> UB.10 -> UB.11 -> UB.12, UB.13
T1.06 [PASS]                   ┘        (+T2.00 PASS)
```

Every dependency is either PASS or unblocked. **No locomotion, no D1, no GPU for
UB.9** (`CPU_LONG`). UB.9's own note says the design is deliberate: *"I(audio;Y)=0,
I(vision;Y)=0, I(audio,vision;Y)=1 bit — physical XOR... Proprioception, Jack's
dominant modality, is uninformative here by design."* UB.16 then certifies the
trunk→controller channel under either D1 outcome.

This is the one place where "0 of 37 unison specs" can start moving while D1 is
still open — subject to §6.2, since UB.9 needs PG.6's frames.

### 6.4 The largest block of immediately-runnable work is CPU and already designed

Twenty specs are unimplemented **and** unblocked right now: `ME.11.A–F`, `PG.6`,
`PG.7`, `T2.03`, `T2.04`, `T2.05`, `T2.06`, `T2.08`, `T2.11`, `T2.14`, `T2.19`,
`T3.07`, `T3.09`, `T4.02`, `UB.14`. Of these the six ME.11 arms and UB.14 are
pure CPU with pilot measurements already taken, and PG.6/PG.7 unlock §6.3.

`OVERSIGHT.md`'s closing number stands and this audit widens it: **128 specs, 51
results, and the registry is growing faster than the ledger.** This document adds
14 more proposed specs; that is only defensible if the sequencing in §7 is
followed rather than the registry extended again.

### 6.5 What breaks in the harness at life scale — enumerated

| mechanism | current behaviour | at a 1-hour life | verdict |
|---|---|---|---|
| Episode length | `Humanoid-v5` `max_episode_steps=1000` = **15 sim-seconds** | 240 episode-lengths | The `TimeLimit` wrapper must go or be raised; death must come from the world, not the wrapper. **Breaks.** |
| `run.py` timeout | `budget × seeds × 2`; `cpu<2h` at 3 seeds = 15 h | 23 min with a small head; 52 h with the 57M trunk | **Holds iff the controller is small.** This is §4.1 restated as a harness constraint. |
| Builder loop | `timeout 50m` per iteration | cannot supervise any life longer than ~20 sim-minutes at 2.6×/min | **Breaks.** Needs the detached-poller pattern that already exists for GPU jobs, extended to long CPU runs. |
| Kaggle / Colab sessions | 12 h cap, 30 h/week | a GPU-trained multi-life curriculum spans sessions | **Holds** via T0.04/T0.05 for weights — but **not** for world/need/diary state. LF.02. |
| `run_spec` unit | one `fn(seed)` per declared seed, aggregated by mean | a life is not a seed; deaths per life, lives per seed | **Breaks.** A life-cohort needs a nesting the aggregator does not have; and `_aggregate` reports mean±std over a distribution (survival time) that is heavily skewed. |
| `_aggregate` rounding | `_round6` keeps 6 significant figures below 1.0 | fine | **Holds** (T0.15 closed this). |
| Ledger entry | one result per spec, `duration_s` scalar | a life-cohort has a survival distribution | **Strained** — recordable, but the interesting statistic will be hidden inside `metrics`. |
| Diary growth | unbounded `events.db` | 240,000 control steps of events per life | **Unknown, untested.** ME.5 is the standing spec; the event *rate* has never been stated. |
| `_experiment` GPU submission | one submission per seed, module-cached | unchanged | **Holds.** |

### 6.6 Twenty-eight researched specs are outside the registry

`LT.01–LT.09`, `PS.00–PS.06`, `HR.1–HR.8`, `UB.9–UB.16`/`PG.6`/`PG.7` (these
*are* registered), `D1.0`, `T2.21`, `LG.00`, `LG.05`. Excluding the registered
UB block, **28 specs exist as fully-designed `Spec(...)` bodies in
`docs/research/` and are invisible to `run next`, to `blocked_by`, and to every
count in `OVERSIGHT.md`.**

This is the §6.1 failure mode in a different disguise: work that cannot be
listed will not be done. Two of them (`D1.0`, `T2.21`) are the D1 bakeoff the
whole ladder is waiting on. Registering them is a mechanical change and is not
in this document's scope, but it should precede any new registration.

---

## 7. The sequenced path

The owner reaffirmed **see / talk / walk / learn, THEN the world.** The sequence
below is dependency-honest: every item's prerequisites are either PASS today or
listed above it.

### Stage 0 — unblock the machine (days, mostly CPU, no science)

Nothing below Stage 0 is worth starting until these are done, because each one
makes a whole branch visible or runnable.

| # | item | why it is first |
|---|---|---|
| 0.1 | **Register `D1.0`, `T2.21`, `PS.00–PS.06`, `LT.01–LT.09`, `HR.1–HR.8`, `LG.00`** | 28 designed specs are invisible to the runner (§6.6). |
| 0.2 | **Settle D2 (does VOID block?)** | One line from the owner decides whether 40 specs are reachable (§6.1). |
| 0.3 | **Fix the unreachability audit**: closure over all parents, deepest-blocking-ancestor in `run status` | The UB.1 lesson recurred and was certified fixed while still broken (§6.1). |
| 0.4 | **`T0.17` (real-time-factor gate)** | Prevents scheduling a life that cannot finish; two CPU-minutes of measurement (§5.5). |
| 0.5 | **Correct PG.6's rendering premise** and record the xvfb/swrast path | Everything visual, including the owner watching, rests on it (§6.2). |

### Stage 1 — SEE (CPU, unblocked today)

`PG.6` → `T2.03` → `PG.7` → **`UB.9`** → `UB.10` → `UB.11` → `UB.12`, `UB.13`.

This is the §6.3 chain. It requires no locomotion, no D1, and — through UB.9 —
no GPU. It is the only route by which the "0 of 37 unison specs" number moves
this week. `UB.14` runs in parallel (deps: PG.1 PASS) and can delete arm A3's
justification before the bakeoff pays for it.

### Stage 2 — TALK (CPU + short GPU, mostly unblocked)

`LG.00` (certify the eval is language-necessary) → `HR.1` → `HR.2`/`HR.3` →
`HR.4`; then `T2.06`/`T2.07`/`T2.15` **in their ADAPTed form**, on the certified
eval. In parallel and independently: the **`ME.11.A–F` bakeoff** — six CPU arms,
pilots already measured, and the mechanism cross-life memory will run through
(§4.6).

Do not run T2.06/T2.07 before LG.00. `LANGUAGE_GROUNDING.md`'s Finding 1 is that
success rate cannot distinguish a listening policy from a visual prior with a
text input wired to nothing; running them first buys a number nobody may cite.

### Stage 3 — WALK (the GPU frontier; D1 lives here)

`T0.14` ✓, `T0.16` ✓ → **`D1.0`** (CPU: prove the update is no longer measuring
its own dropout) → **`T2.21`** (the D1 bakeoff, ~6.3 GPU-h) → `T2.01` re-run →
`T2.02` re-run **with the real-time-factor gate** (§3.3) → `T3.02`, `T3.10` →
`T2.16`, `T2.17`, `T2.18`.

Prefer `D1.0`+`T2.21` over a bare T2.01/T2.02 re-run: the bare re-run answers
*did the trunk learn*, the bakeoff answers *where the trunk belongs*, and only
the second unblocks the 40 specs.

`T2.18` (chunking) is promoted into this stage rather than left in Tier 2
miscellany, because it is the mechanism that makes a flow head affordable at
life scale (§4.4).

### Stage 4 — LEARN (the learning-core question)

The **learning-core bakeoff** (`LEARNING_CORE.md`, in progress) decides the
mathematics. Its arms already exist as specs: `T2.16` (hindsight flow
regression), `T2.05` (world model), PPO-as-incumbent (`T2.00`/`T2.01`), and
`CU.2`'s LP goal sampling. `T5.01`/`T5.02` are BLOCKED-ON-DIRECTION until it
resolves and **must not be started** — at `seeds=5 × GPU_LONG` they are the
most expensive unrun pair on the ladder.

Also here: `T3.01–T3.10`, the ablation tier, which is at **0 of 12** and which
the owner's "complexity must earn its place" makes the highest-value GPU spend
after D1.

### Stage 5 — W0, the needs playground

**Entry gate — all of these must PASS first:**

- Stage 1 complete through `UB.11` (he sees, and each sense is load-bearing);
- Stage 3 complete through `T2.01` (he walks) and D1 decided;
- `WP.01`–`WP.04` (the world's thermodynamics, metabolism, diurnal cycle and
  resources are analytically correct — §5.1);
- `PS.00`–`PS.02` (the drive layer is a real control problem and the anti-gaming
  detectors see their own positive controls);
- `LF.01` (a 1-sim-hour life completes and the harness survives it) and `LF.02`
  (it can be saved and resumed);
- `T0.17`, `T0.18` (the box will not be harmed by the schedule).

**What W0 must prove before W1 is contemplated:**

1. `PS.03`/`PS.04`/**`PS.07`** — needs beat no-needs at *learning*, not only at
   surviving, on a needs-off held-out battery. If this fails, the direction is
   wrong and the ladder should say so.
2. `LF.03` — life N+1 beats life N *because of the diary*, with the shuffled-diary
   control at null.
3. `LF.04` — sleep is a need, not a scheduler.
4. `T2.20`-as-cross-life and `CU.7`-as-cross-life (the two cheapest, already
   built).
5. `SO.02` — "I'm cold" is true when he is cold.
6. `T2.09`/`CU.3` re-run **needs-off**, so noisy-TV immunity is not bought by
   starvation.
7. `SO.01`/`SO.04` — the owner can watch, and watching does not change him.

### Stage 6 — W1 and beyond

`LF.05`/`T5.08` (the world grows with him), `SO.03` (company), `SO.05`
(interruption), `T6.01`/`T6.02`/`T6.04` in their ADAPTed life-scale form,
`T6.05`. Then the jungle.

### 7.1 Where the current frontier sits

| currently in flight | verdict |
|---|---|
| **T2.01 re-run** (~13 GPU-h queued, blocked on the `git push` question D3) | **Worth finishing, but re-scope first.** Add the real-time-factor gate, and prefer `D1.0`+`T2.21` ahead of it. Running the bare re-run first spends 13 of the ~23.6 remaining Kaggle hours to answer the smaller of the two questions. |
| **PG.8 humanoid-in-playground** | **Done and correct** — PASS at 3 seeds. It is the precondition for everything in §5. |
| **D1** | **Do not decide on the current evidence** (`DECISIONS_NEEDED.md` is right). The new throughput axis (§4.1) should be added to the decision memo *before* the owner is asked again, because it can change the answer. |
| **ME.11 bakeoff arms** | **Worth finishing, unchanged, and promoted** — CPU, unblocked, and §4.6 makes it load-bearing for cross-life memory. |
| **UB.9–UB.16 pre-registrations** | **Worth implementing, in the §6.3 order.** Ten specs were written and zero run. |

### 7.2 Not worth finishing under the new direction

| item | why |
|---|---|
| **`T5.01` / `T5.02`** — SymPy physics pre-training as THE thesis test | Premise superseded by "the world is the teacher" and pending the learning-core bakeoff. 5 seeds × GPU_LONG. **Do not start.** |
| **`T4.01`, `T4.02`, `T4.03`** | Superseded by UB.10/UB.11/UB.12 with strictly better methodology. Implementing them spends CPU/GPU re-litigating a matrix UB.11 measures better — and T4.01 is the node that keeps UB.1 unreachable (§6.1). |
| **`UB.1`, `UB.2`, `UB.3`, `UB.6`** | Same claims as UB.10's arms A0–A4 and UB.11, at weaker design. Implementing them is the expensive way to learn what UB.10 will say. |
| **`T3.09`** — wiring `AlphaGeometryLoop` | Worth *running* (CPU, and it deletes 559 lines), not worth *wiring*. The direction supplies no role for it. |
| **Any further registry expansion** before Stage 0.1 | 128 registered specs, 51 results, 28 designed-and-unregistered specs, and this document proposes 14 more. The gap between design and evidence is now the project's largest number, and `OVERSIGHT.md` said so a day ago. |

---

## 8. What this document does not settle

- **The world's fidelity ladder (W0→W3)** and its cost — `SURVIVAL_WORLD.md`.
- **The learning core** — `LEARNING_CORE.md`. Two verdicts above
  (T5.01, T5.02) are explicitly parked on it.
- **The needs/death mechanics in detail** — `NEEDS_AND_DEATH.md`, which did not
  exist when this was written; §5.3's `LF.*` stubs should be reconciled against
  it and the duplicates dropped.
- **Whether any of the seven OBSOLETE specs is actually deleted.** That is the
  owner's call, as is every architecture decision in §4. Nothing here was
  removed.
- **D1, D2, D3.** All three remain open and all three now cost more than they
  did yesterday: D1 gained an axis, D2 gates 40 specs, D3 gates the Sunday
  quota.

---

*Audit performed 2026-08-09 against HEAD `ab53ef0`, ledger 48 PASS / 2 VOID /
1 ERROR of 128 specs. Every mechanical claim in §0 and §6 was executed on this
box; the commands are in §0 so they can be re-run and disagreed with. No file
outside this one was modified.*
