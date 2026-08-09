# The Ladder Test and the Curiosity Bakeoff

> Researched and specified 2026-08-09. Companion to `docs/research/CURIOSITY.md`,
> which surveys mechanisms. This document does one thing that survey does not:
> it turns the owner's sentence into an experiment that can fail, and specifies
> the bakeoff that decides which mechanism gets Jack up the ladder.

The sentence, verbatim, is the specification:

> *"in his free time he must be trying to figure out his environment and like
> learn random stuff himself but REALLY REALLY LEARN like if theres ladder with
> apple on top he must try to climb that ladder and fall and learn purely out of
> curiosity like and play outside and if theres water he must try to swim and
> struggle and learn himself"*

Everything below exists to make that falsifiable. The playground is built
(`playground.py`), the ladder is certified climbable (PG.3, `ascent_frac 0.97`),
the noisy-TV trap is certified live (PG.4), contact audio works (PG.5). What is
missing is the thing that drives him up it — and a metric that cannot be faked.

---

## 0. The honest starting position

Four facts from this repo's own ledger frame everything:

| Fact | Source | Consequence |
|---|---|---|
| The ladder is physically climbable and falls are clean and resumable | PG.3 PASS, `ascent_frac 0.973 ± 0.020`, `resume_max_dev 0.0` | The fixture is not the blocker. Go-Explore-style *state restore* is certified to bit-exactness — unusual and valuable (§1.4). |
| The noise panel really does hypnotise naive curiosity | PG.4 PASS, `icm_dwell_share 0.667 ± 0.471`, null `0.061 ± 0.027`, static-texture control `0.000` | The trap works — **but read the std**: across 3 seeds the ICM arm scored roughly {1, 1, 0}. Dwell must be reported **per seed**, never as a mean alone. |
| **But the panel only catches half the failure.** | §2.10, new in this document | An agent can farm irreducible surprise from its *own thrashing body* with `panel_dwell = 0.000`. PG.4 is structurally blind to it; RGSD (2510.06203) measured exactly that failure on a humanoid. The self-generated-chaos check (LT.02) is the other half. |
| **Jack cannot walk.** | T2.01 FAIL, T2.02 VOID (22,604 s CPU, "two non-learners cannot arbitrate") | "Curiosity climbs the ladder" on the full humanoid is currently blocked on locomotion. Any plan that starts there is a plan to burn quota. §5 stages around this, and LT.08 is the only spec in this document that depends on it. |

And one fact from the literature (§1): **no published system has produced
ladder-climbing in a humanoid from intrinsic motivation alone.** The only method
that has ever produced humanoid ladder climbing — LadderMan (arXiv:2606.05873,
2026) — does it by tracking a *single human reference motion*, then distilling
experts into a visuomotor policy. That is the opposite of the goal here. This
makes the Ladder Test genuinely novel, and genuinely at risk of failing. It
should be run in the order that discovers failure cheapest.

---

## 1. Survey: what has actually worked, and where

Assessment column is deliberately blunt. "Humanoid?" means: has this method been
shown to produce *structured, coherent* behaviour on a humanoid-scale body
(≥20 DoF, whole-body contact), not merely been run on one.

### 1.1 Prediction-error curiosity (the family PG.4 was built to catch)

| Method | arXiv / venue | Demonstrated on | Humanoid? | Honest assessment |
|---|---|---|---|---|
| ICM | 1705.05363, ICML 2017 | VizDoom, Mario | No | Inverse-dynamics features filter *uncontrollable* noise, not *action-conditioned* noise. Jack's own physics chaos is action-conditioned. |
| Large-Scale Study of Curiosity | 1808.04355, 2018 | 54 Atari + Mario, Unity 9-room maze with a TV | No | The paper that named the noisy-TV failure with a measurement. **Cite it accurately**: the authors report the TV *"drastically slows down learning, but… if you run the experiment for long enough the agents do sometimes converge"* — a slowdown, not a permanent trap. The common "stuck forever" paraphrase is a miscitation. PG.4 measures the fixation directly and does not need the stronger claim. |
| RND | 1810.12894, ICLR 2019 | Montezuma's Revenge | No | The de-facto pseudo-count for continuous state. Its bonus is over *observations*, not over the controllable manifold: every novel flailing pose scores novel. On a re-randomising texture the target-net output is itself never repeated, so RND is trapped too. |
| DRND / RDD | 2401.09750 (ICML 2024), 2505.11044 | Atari, MuJoCo tasks | No | Fixes RND's "bonus inconsistency" by distilling a *distribution* of random nets. Better pseudo-counts; does nothing about irreducible noise. |
| NovelD | NeurIPS 2021 | MiniGrid, NetHack | No | Rewards the *increase* in novelty across a transition — fixes RND's depth-first collapse. Discrete. |
| E3B | 2210.05805, NeurIPS 2022 | MiniHack, VizDoom, Habitat | No | Elliptical episodic bonus over an **inverse-dynamics embedding** — the embedding is the transferable idea (it filters uncontrollable noise). Never evaluated on continuous control. |
| VIME | 1605.09674, NeurIPS 2016 | rllab low-dim state only: CartPole, MountainCar, HalfCheetah, Walker2D, SwimmerGather | No | **Largest action space in the entire paper is 6-D**, and it runs **no stochastic / noisy-TV experiment at all** — its noise robustness is a theoretical property of the KL objective that was never tested. SimHash counting (#Exploration, NeurIPS 2017) matches it on two of its own tasks. |
| BYOL-Explore | 2206.08332, NeurIPS 2022 | DM-HARD-8, Atari | No | **Correction to `CURIOSITY.md`, which lists this among the noisy-TV fixes:** it is not one. BYOL-Hindsight (below) exists *because* BYOL-Explore degrades under sticky-action Atari. It is a member of the vulnerable prediction-error class. |
| Disagreement | 1906.04161, ICML 2019 | noisy-MNIST; Atari incl. sticky-action; **Unity 3D maze containing a TV whose channel the agent can change**; MuJoCo 7-DoF arm; real Sawyer arm with RGBD | No | The TV experiment is the direct evidence. Does **not** report HalfCheetah/Ant/Humanoid — the locomotion attribution often made for it belongs to Plan2Explore. Own stated limit: the differentiable variant works only on short horizons. |
| Plan2Explore | 2005.05960, ICML 2020 | DM Control from pixels (walker, cheetah, hopper, quadruped) | No (quadruped max) | Ensemble disagreement over a latent world model; zero-shot to downstream tasks. Best-evidenced member of this family that scales to **pixel continuous control**. |

**Verdict.** Every member of this family is a *reference* arm, not a candidate
winner. ICM and RND are in the bakeoff because CU.3 requires a trap-victim that
must fail. Disagreement is the only one with a real chance — and its supporting
evidence is a 7-DoF arm and a 3D maze, not a body.

### 1.1b The 2022–2026 noisy-TV literature (which `CURIOSITY.md` predates)

Three papers matter here and none of them has ever been run on continuous control.

| Method | arXiv / venue | Mechanism | Domains | Honest assessment |
|---|---|---|---|---|
| **LPM — Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring** | **2509.25438, ICLR 2026** | Dual network: an *error model* predicts the dynamics model's expected prediction error at the previous iteration; the bonus is the **difference between current and previous model error**. Proven zero-equivariant and a **monotone indicator of information gain**; the error model is proven *necessary* for that monotonicity. | noisy-MNIST, a 160×120 RGB 3D maze, Atari | **The most directly on-topic modern paper for this document, and the strongest theoretical statement that "reward model improvement, not surprise" is the right family.** It is also brand new, unreplicated, and its LP estimate is a small difference between two large noisy quantities. Zero continuous control. Treat as the upgrade path for arm A3, not as a settled result. |
| **BYOL-Hindsight — Curiosity in Hindsight** | 2211.10515, ICML 2023 | Structural-causal framing: learn a *hindsight representation of the future* that captures precisely the unpredictable part of each outcome and feed it to the predictor, so residual error contains only predictable dynamics. | gridworld + sticky-action Atari | The best-argued noisy-TV paper in the literature and **it has never been run on a continuous-control task**. Central fragility: the hindsight encoder can cheat by leaking predictable information into the hindsight variable, collapsing the bonus to zero everywhere; needs an information bottleneck. |
| **Aleatoric Mapping Agents** | 2102.04399, ICML 2022 | Heteroscedastic head predicts mean *and variance* of the next state; down-weight the bonus where aleatoric variance is high. | small custom action-dependent stochastic-trap environments | Only handles noise expressible as per-transition scalar Gaussian variance. A panel emitting structured high-dimensional novel images is not that — so it would probably fail PG.4. Heteroscedastic regression also under-estimates variance early, exactly when it matters. |

Schmidhuber's compression-progress lineage is the ancestor of all three:
arXiv:0812.4360 (*J. SICE* 48(1):21–32, 2009) and *Formal Theory of Creativity,
Fun, and Intrinsic Motivation*, IEEE TAMD 2(3):230–247, 2010.

### 1.2 Learning progress / competence progress

| Method | arXiv / venue | Demonstrated on | Humanoid? | Honest assessment |
|---|---|---|---|---|
| IAC → R-IAC → SAGG-RIAC → IMGEP | Oudeyer 2007; Baranes & Oudeyer 2013; survey 1708.02190, autotelic survey 2012.09830 | Robot arms, playground robots, low-dim outcome spaces | No | The conceptual backbone. Noisy-TV-proof *by construction*: irreducible noise yields zero competence improvement, so its LP decays to zero and allocation collapses. The catch is that LP needs a **competence signal**, i.e. a goal space with a success detector. |
| ALP-GMM | 1910.07224, CoRL 2019 | BipedalWalker stump/hexagon tracks | No (4-DoF 2D walker) | Absolute LP over a **2–3 dimensional environment-parameter space**, fitted with a GMM. Cheap, robust, well-replicated. Its demonstrated task space is tiny — this is a teacher over env parameters, not over goals in a rich scene. |
| TeachMyAgent | 2103.09815, ICML 2021 | Box2D benchmark of ACL teachers, incl. a **climbing morphology** | No | **The most sobering datapoint in this section.** The LP-teacher family reaches roughly **1 % mastery on the 2D climbing morphology.** The one time anyone pointed learning-progress curricula at a climbing task, in two dimensions, it barely moved. |
| Learning-progress curriculum at scale | Kanitscheider et al. (Minecraft, 2021) | Minecraft, symbolic goal list | No | Shows LP scales to a hand-enumerated goal list. Goals were given, not invented. |
| Curious Replay | 2306.15934, ICML 2023 | Crafter, DM Control | No | **Not a noisy-TV solution and possibly the opposite.** Its replay priority includes an adversarial/loss-based (surprise) term — exactly the quantity a noisy TV maximises — so a stochastic distractor would be *preferentially replayed into the world model*. The paper never tests this. Adjacent, not aligned. |
| MAGELLAN | 2502.07709, 2025 | Text-world autotelic agents | No | Learned LP *predictor* so LP generalises to unseen goals in large goal spaces — the fix for LP's biggest scaling problem (you cannot measure LP for a goal you have never sampled). Text domain, and its authors explicitly write *"we do not recommend generalizing our findings to real-world open-ended learning settings."* |
| LPM | 2509.25438, ICLR 2026 | noisy-MNIST, 3D maze, Atari | No | Learning progress at the **model** level rather than the goal level (§1.1b). The only formulation with a proof that its bonus monotonically indicates information gain. Never run on continuous control. |
| CURIOUS | 1810.06284 | Fetch arm, modular goals | No | Modular LP over goal *modules* — the useful idea is per-module LP, and it survives inside any LP implementation. |

**Known failure modes of LP** (all must be designed against, §3.5):
1. **Noisy LP estimates.** LP is a difference of two noisy success rates; with
   few attempts per goal the difference is mostly variance. Mitigation: EMA over
   a window, and a minimum attempt count before a goal's LP is trusted.
2. **Forget–relearn farming.** An agent can harvest LP forever by forgetting a
   mastered goal and relearning it. Mitigation: track LP on a *held-out*
   evaluation of each goal region, and cap cumulative allocation per region.
3. **Window hyperparameter sensitivity.** The LP window length silently sets the
   curriculum's timescale. Mitigation: report results at 3 window lengths;
   any conclusion that flips is not a conclusion.
4. **Non-stationarity of the goal space.** As the outcome space grows, old LP
   estimates become stale. Mitigation: recency-weighted, and re-estimate on
   region split.
5. **LP is zero where competence is zero.** This is the important one for the
   ladder. A goal that has *never once* been achieved has LP = 0 and is
   indistinguishable from an impossible goal. LP cannot bootstrap a skill whose
   first success has probability zero — which is exactly what §4's pilot
   measured for weight-bearing on the ladder. **LP needs a novelty/goal-babbling
   partner to produce the first success; it is a selector, not a discoverer.**

### 1.3 Unsupervised skill discovery

| Method | arXiv / venue | Demonstrated on | Humanoid? | Honest assessment |
|---|---|---|---|---|
| DIAYN | 1802.06070, ICLR 2019 | 2D nav, inverted pendulum, mountain car, Hopper (3), HalfCheetah (6), Ant (8) | **Never evaluated on humanoid** | Only requires skills be *distinguishable* — satisfied by static poses. Zero object interaction anywhere in the paper. Its own admissions: *"most skills move in arcs rather than straight lines"*, and Ant required prior knowledge of which state features to discriminate on. |
| DADS | 1907.01657, ICLR 2020 | HalfCheetah, Ant, Humanoid | **Primed, not discovered** | Usually cited as the first humanoid skill-discovery result. The caveat that citation drops: the humanoid variant restricts the skill-dynamics observation to **COM x–y only** — later papers name this configuration **DADS-XYO, "XY oracle."** "Move in different directions" was *handed to* the algorithm. |
| LSD | 2202.00914, ICLR 2022 | MuJoCo locomotion | Goal-following only | Diagnoses that MI objectives "prefer static skills to dynamic ones"; the Lipschitz fix rewards travelling far — which *biases the whole family toward locomotion*. |
| CSD | 2302.05103, ICML 2023 | 6 manipulation + locomotion envs | No | The honest exception: controllability-awareness does discover **object manipulation** without supervision. Arm-scale and Ant-scale, table-top primitives. |
| **METRA** | **2310.08887, ICLR 2024** | Ant, HalfCheetah; pixel DMC Quadruped/Cheetah/**Humanoid**; Franka Kitchen | **Locomotion only** | The best humanoid result in the field, and it is *diverse gaits and travel directions* over a 2-D latent (16 skills: running, backflipping, crawling). Its only object-interaction result is Kitchen, where 3–4 of 6 tasks are completed **"coincidentally"** as a by-product of state coverage. Nothing vertical. Nothing structured. |
| **RGSD** | **2510.06203, 2025** | **SMPL humanoid: 359-D obs, 69-D action, 23 spherical joints** | **Collapses** | **The most important citation in this document.** An order of magnitude past every other paper here. Ran METRA on it: *"skills fail to yield meaningful behaviors. Joints move randomly, producing highly unstructured motions in which arms, legs, torso, and head move independently and arbitrarily."* Cartesian error 42–52 cm, FID 32.8–140.3. Diagnosis, verbatim: *"As the DoF increases, the exploration space grows exponentially, while the portion of the semantically meaningful manifold remains relatively small."* Fix: contrastive pretraining on **20 ACCAD mocap clips**. Also notes a *structural* conflict — Wasserstein/maximal-difference objectives (METRA, LSD) actively **penalise cyclic, in-place, whole-body motion**, which is most of what is interesting. |
| **Meta Motivo / FB-CPR** | **2504.11054, ICLR 2025** | Full humanoid whole-body control | **Needs a motion prior** | The decisive contrast case. Framed as unsupervised RL (forward–backward, no task reward) — and its central contribution is *"regularizing unsupervised RL towards imitating trajectories from unlabeled behavior datasets"* (observation-only AMASS mocap). **A Meta FAIR team with effectively unlimited compute concluded that unsupervised RL alone does not produce humanlike whole-body behaviour, and bolted a motion prior onto it.** |
| URLB (the field's own benchmark) | NeurIPS 2021 D&B | 12-DoF quadruped maximum; **no humanoid** | — | Verbatim verdict on the skill-discovery family: *"there is no competence-based approach that achieves state-of-the-art mean performance on any of the URLB tasks."* DIAYN, APS and SMM specifically lag; the knowledge-based (ICM/Disagreement/RND) and data-based (APT/ProtoRL) families do better. |
| MOD-Skill | 2602.09767, 2026 | **Unitree A1, 12 DoF, real hardware** | No | **The honest 2026 state of the art for genuinely data-free skill discovery: a 12-DoF quadruped.** No mocap, no task reward, no instruction videos. There is no humanoid equivalent. |
| GISD | 2601.14000, 2026 | locomotion benchmarks | No | Exploits the environment's symmetry group; beats METRA on coverage. Another prior injected, this time a mathematical one. |
| Heess et al., *Emergence of Locomotion Behaviours in Rich Environments* | 1707.02286, 2017 | Humanoid, rich terrain | **Yes — with extrinsic reward** | Not intrinsic motivation, but the standing existence proof that **the body is not the blocker**: a humanoid learns running, jumping, crouching and turning from *"a simple reward function based on forward progress"* plus terrain variety. Structure came from extrinsic reward + environment design. The objective is the blocker, not the DoF count. |
| SLIM | 2402.00823, ICRA 2024 | Table-top arm | No | Names the split cleanly: MI maximisation covers *the agent's own state*; affecting DoF **outside** the agent's own state needs exploration the objective does not reward. |
| Can a MISL Fly? / CSF | 2412.08021, ICLR 2025 (Oral) | URLB-scale | No | Deflationary: METRA's gains are reproducible inside plain MI skill learning with contrastive successor features. The headline advance is a representation detail, not a capability class. |
| SDAX | 2508.08982, CoRL 2025 | **Quadruped parkour** | No (12-DoF) | The closest anything gets to vertical structure from skill discovery: crawling, climbing, leaping, jumping off vertical walls — via bi-level optimisation of how much to explore. Quadruped, task-specific courses. |
| BFM-Zero | 2511.04131, 2025 | Real Unitree G1 | Yes, but | Forward–Backward behavioural foundation model, zero-shot goal reaching on real hardware. "Unsupervised" means *unsupervised w.r.t. reward* — the dataset is mocap-derived. Still locomotion/pose-tracking. |
| DoDont / Divide-Discover-Deploy / URSA | 2406.00324 (NeurIPS 2024), 2508.19953, 2508.19172 (CoRL 2025) | Continuous control, quadrupeds | No | Note the 2024–2026 pattern: **every method that gets coherent behaviour injects a prior** — instruction videos, state factorisation + symmetry, or quality-diversity over hand-chosen descriptors. |

**Verdict.** Skill discovery finds *self-motion*, not *interaction with
structures*. On a 69-DoF humanoid it does not even find that (RGSD). Every
purely intrinsic "humanoid" result in the literature tops out at COM directional
locomotion, and in DADS's case the direction axis was handed to the algorithm.
The field's own benchmark caps at a 12-DoF quadruped and reports that the
competence-based family fails on every task in it. Meta FAIR reached the same
conclusion at scale and added mocap.

METRA belongs in the bakeoff as the strongest reward-free skill prior available,
but the prior probability that it climbs a ladder unaided is low, and the bakeoff
must be designed so that this is discovered in CPU hours, not GPU weeks. **Note
also that Jack's climber-rover has 8 actuated DoF — inside the range where these
methods do work (Ant is 8, the A1 quadruped is 12) and far below the 69 where
they demonstrably collapse.** That is a second, independent reason to run the
Ladder Test on the reduced body first: it is the only regime where the
literature gives the methods a chance at all.

### 1.4 Goal-conditioned self-play, hindsight, and archive exploration

| Method | arXiv / venue | Demonstrated on | Humanoid? | Honest assessment |
|---|---|---|---|---|
| HER | 1707.01495, NeurIPS 2017 | 7-DoF Fetch, sparse push/slide/pick | No | Every failure is a success at what it *did* achieve. Composes perfectly with a flow-matching head (no log-likelihood needed — see CURIOSITY.md §0). Degrades when the goal is a long structured contact sequence. |
| Go-Explore / First return, then explore | 1901.10995; **Nature 590:580 (2021)**, 2004.12919 | Montezuma, Pitfall; high-level pick-and-place | No | **Structurally the best fit for Jack that nobody in this repo has noticed.** Its hard requirement is the ability to *return* to an archived state — and PG.3 already certified MuJoCo snapshot/restore at `resume_max_dev = 0.0`. The weakness is that the *cell representation is the algorithm*, and it is hand-designed. Choosing cells is where the design intelligence hides; choosing them from ladder coordinates would be instruction. |
| Intelligent Go-Explore | 2405.15143, ICLR 2025 | Text/symbolic games | No | Replaces hand-designed cells with an LLM judging archive states. Requires states renderable as text. |
| PLR / ACCEL | 2010.03934 (ICML 2021), 2203.01302 (ICML 2022) | Procgen, MiniGrid; **2D 4-DoF BipedalWalker** | No | Environment curricula by regret/learning potential. The only continuous-control instance in ACCEL is a 2D walker. UED assumes a parameterisable level generator — `PlaygroundParams.mutate()` already is one, which is why CURIOSITY.md §3 keeps it. |
| PEG | 2303.13002, ICLR 2023 (Spotlight) | Ant-maze, cluttered tabletop 3-block stacking | No | **The most humanoid-plausible member of this family.** Plans *goal commands* through a world model to maximise downstream exploration value, then hands off. Directed goal-space exploration in continuous control that actually reaches object interaction. |
| HILP / OGBench / Horizon Reduction | 2402.15567 (ICML 2024), ICLR 2025, NeurIPS 2025 | DMC, kitchen | No | Argues *horizon*, not exploration, is the binding constraint at scale. Relevant caution: a 40-attempt ladder climb is a long-horizon credit-assignment problem dressed as an exploration problem. |
| Qflex | 2601.19707, 2026 | SMPL Humanoid-Jump, Unitree H1, MyoLeg | Yes (task RL) | **Cite for the theory, not the method.** Proves that under isotropic joint-angle perturbation, end-effector position variance scales as **O(1/\|A\|)** — undirected exploration provably vanishes as DoF grows. This is the formal reason §5 stages the Ladder Test on a reduced body first. |
| Empowerment (Karl 1710.05101; Zhao 2007.07356; CAIMAN 2502.00835) | various | Low-dim stabilisation; legged loco-manipulation | **No** | Estimator cost grows badly with action dimension; the field routed around it into MI skill discovery and temporal-distance metrics. CAIMAN's causal-action-influence signal (does my action affect the *object*?) is the one salvageable idea and it maps onto CU.6. |

### 1.5 LLM-proposed goals and automatic curricula

| Method | arXiv / venue | Action space | Honest assessment |
|---|---|---|---|
| Voyager | 2305.16291, TMLR 2024 | Minecraft JS API | Genuinely open-ended curriculum + verified skill library. **Zero motor control** — every "skill" is a program calling scripted primitives. The action space does all the work. |
| OMNI | 2306.01711, 2023 | Crafter, BabyAI, AI2-THOR (13 discrete actions) | The dual filter is the contribution worth stealing: sample tasks that are both **learnable** (progress is happening) and **interesting** (FM-judged), because learnability alone yields infinite trivial variations. No continuous control. |
| OMNI-EPIC | 2405.15568, 2024 | PyBullet, **6 discrete actions** | FM writes environment *and* reward as code. Stated limitations are directly on point: trains a *population of specialists*, and "VLM success detectors are not yet accurate enough". |
| Eureka | 2310.12931, ICLR 2024 | Shadow Hand, 29 envs | **This is reward design, i.e. instruction.** The task is human-named; the LLM optimises the specification. No interestingness model, no novelty term, no archive. Not a curiosity system. Its pen-spinning curriculum was hand-staged. |
| Eurekaverse | 2411.01775, CoRL 2024 | Quadruped parkour | Closest to LLM-curriculum for low-level control: the LLM writes *terrain code* of increasing difficulty and beats human-designed courses. But it is unsupervised **environment** design within one fixed task+reward family. |
| ELLM | 2302.06692, ICML 2023 | Crafter (260 discrete), Housekeep | Its limitations section is the citation: fails "when human common-sense... cannot be expressed in language (e.g. fine-grained manipulation)" and "requires states and transition captions". |
| Motif | 2310.00166, ICLR 2024 | NetHack | Architecturally the most transferable: LLM gives **preferences over pairs of event captions**, distilled into a cheap reward model. Right compute split (FM offline, RM online). Still needs a captioner. |

**Verdict.** LLM-proposed goals have **never** driven low-level humanoid
continuous control. The gap is captioning/grounding, not effort. For Jack the
correct form is therefore: the VLM proposes **goals as predicates in an existing
outcome space**, never rewards and never code that mentions the ladder — and LP
has the final vote (CURIOSITY.md §6). Anything else is Eureka, i.e. instruction,
i.e. disqualified by §3.5/G1.

### 1.6 The one-line summary of the field

> Humanoids climb when you give them the motion (LadderMan, 2606.05873) or the
> reward (Heess 1707.02286; robot/humanoid parkour). Curiosity methods produce
> coverage, locomotion diversity, and — on a 69-DoF body — flailing (RGSD,
> 2510.06203). **The intersection is empty.** That is the gap this project is
> aiming at, and the honest prior is that it is hard, not that everyone missed
> it.

Five specific things the literature says that this design must respect:

1. **Nobody has proposed an intrinsic objective whose maximiser is neither
   dominated by COM translation nor saturable by incoherent joint noise.** That
   is the open problem, stated precisely. Scaling METRA is not it — RGSD and
   GISD already answered that question.
2. **The noise-robust-by-construction family is learning progress in the
   Schmidhuber/LPM sense**, and its most rigorous member (LPM, ICLR 2026) has
   been run on noisy-MNIST, a maze, and Atari. Nothing embodied.
3. **The one time learning-progress curricula were pointed at a climbing task**
   — TeachMyAgent's 2D climbing morphology — the family reached ~1 % mastery.
4. **DoF is the axis that predicts failure.** 8-DoF Ant works, 12-DoF quadruped
   works (MOD-Skill, on real hardware, data-free), 69-DoF humanoid flails. Jack's
   climber-rover sits at 8.
5. **Every 2024–2026 method that produces coherent humanoid behaviour injects a
   prior**: mocap (RGSD, Meta Motivo, BFM-Zero), instruction videos (DoDont),
   LLM-defined subspaces (LGSD 2406.06615), symmetry groups (GISD), or state
   factorisation. If the Ladder Test passes without one, that is the finding.

---

## 2. THE LADDER TEST

*The single most important specification in the project. Read `falsified_by`
before `hypothesis`.*

### 2.1 What is being claimed

> Placed in a world containing a ladder he has never been told about, with no
> external reward of any kind, Jack repeatedly attempts to get his weight onto
> that ladder; he falls; he comes back; the height he reaches grows across
> attempts; eventually he tops out; and the skill survives the removal of the
> intrinsic drive that produced it.

Six clauses, six observables, all measured in one continuous unbroken life.
"Unbroken" matters: an episode boundary that teleports him back to the ladder
base would be an experimenter-supplied curriculum. Episodes reset only on
divergence.

### 2.2 The world

`playground.py` at default `PlaygroundParams`, **per-seed mutated** via
`PlaygroundParams.mutate()` so that no arm ever sees a hand-picked world. All
fixtures present and mandatory:

- **the ladder** at `(0, −2.6)`, 6 rungs, platform at `ladder_height` (1.8 m).
- **the apple** on the platform, carrying **no reward** — it is an object like
  any other (`playground.py:138`). This is load-bearing: a rewarded apple would
  make the whole test a reward-following test.
- **the noise panel** at `(0, +5.9)` — mandatory fixture, §4.
- **the distractors that make height gameable**: stairs (top 0.72 m), ramp
  (top 0.52 m), seesaw, five loose objects. §2.5/G4 depends on these existing.

Ladder and panel sit at opposite ends of a 12 m arena, so dwell zones cannot
overlap.

**The noise must be realised in the observation, not the texture.** MuJoCo's
`noise_mat` is a flat texture; PG.4 re-randomises panel-hitting rays inside the
`_Retina` (`pg_4_noisy_tv.py:160`). Any Ladder-Test rig must do the same, with
the same `R_RESOLVE = 2.5 m` acuity falloff, or the trap is not present and §4
certifies nothing.

### 2.3 The body — and why not the humanoid, yet

The full humanoid is the eventual subject (LT.08). It is **not** the subject of
the first Ladder Test, for three independently sufficient reasons:

1. **T2.01 and T2.02 FAIL.** He cannot walk. A negative Ladder Test on a body
   that cannot locomote tells you nothing about curiosity.
2. **Qflex (2601.19707) proves the exploration variance scales as O(1/|A|).**
   Undirected exploration on a 17-actuator body is provably weaker than on a
   6-actuator one, by a known factor.
3. **RGSD (2510.06203) measured the failure**: skill discovery on a 69-DoF
   humanoid yields joints "moving independently and arbitrarily".

So the Ladder Test is first run on the **climber-rover**: the PG.3 certified
climbing rig plus the ability to get to the ladder.

```
climber-rover  (8 actuated DoF + free root)
  torso        capsule, 30 kg, contype 1     ── masked out of the ladder class
  foot         sphere, 2 kg, contype 1       ── floor contact, gates the drive
  armL/armR    reach(y) + lift(z) slides, damping 40, PG.3 parameters exactly
  handL/handR  adhesion actuators, gain 900 N, contype 5
  drive        bounded horizontal force on the torso, GATED on floor contact
```

Declared rig conveniences, in PG.3's style — each stated so a reader can attack
it:

- **The drive is a cheat, deliberately.** It grants locomotion, which is
  T2.01's problem, not the Ladder Test's. It is force-limited (600 N) and
  **gated on floor or stair contact**, so it cannot fly, cannot climb, and
  cannot contribute a single newton once the feet leave the ground. Every metre
  of ladder-supported height is earned by the arms.
- **Torso and foot are masked out of the ladder contact class** (contype 1 vs
  the ladder's 4), exactly as in PG.3. Hands (contype 5) collide with
  everything. So the body cannot wedge itself on a rung; it must grip.
- **Adhesion stands in for fingers**, as certified by PG.3. Holding adhesion
  permanently on is a *legal* strategy (children grip hard too); report
  `adhesion_duty_cycle`, do not penalise it.

The arm parameters, adhesion gain and contact classes are copied unchanged from
`pg_3_ladder_climbable.py`, so the rover **inherits PG.3's certification by
construction**. LT.01 verifies that inheritance rather than assuming it.

### 2.4 Mechanical definitions (pre-registered; no post-hoc tuning)

Geom sets resolved by *name* at build time, so they survive world mutation:

```
LADDER  = {rung0..rungN, ladder_railL, ladder_railR}
CLIMB   = {handL, handR, climber_foot}
GROUND  = {floor, ramp, stair*, seesaw_plank, poolwall*, pool_floor,
           obj*, welded_block, platform}
```

**LADDER-SUPPORTED RISE `h(t)`** — the anti-gaming core of the whole test. It
took three iterations against measurement to get right, and the two rejected
versions are documented here because they are exactly the attacks a careless
implementation would fall to.

```
h(t) = z(climber_torso) − z_rest        if  ALL THREE of:
                                          (i)   ∃ contact in CLIMB × LADDER at t
                                          (ii)  no contact between the body and GROUND at t
                                          (iii) (i) and (ii) have held continuously
                                                for >= 0.5 s, and the ladder's
                                                vertical contact force on the body
                                                is >= 0.5 x body weight
       0                                  otherwise
```

`z_rest` is the torso height of the body standing at rest, **measured at build
time per world** (pilot: 0.360 m; body weight 322 N). Reporting a *rise* rather
than an absolute height is not cosmetic — see below.

Why each clause exists, with the number that forced it (pilot, 2026-08-09,
800 random 3 s bursts started at the ladder base):

| Definition | P(score ≥ 0.25 | 3 s random burst) | Verdict |
|---|---|---|---|
| absolute torso z, contact ∧ airborne, instantaneous | **0.55** | **Broken.** `z_rest = 0.36 m` already exceeds the 0.25 m bar, so *any* momentary airborne frame while brushing the rail scored. This measured the tumbling rate, not climbing. |
| rise above `z_rest`, instantaneous | 0.063 | Still over-counts 2–4×: a mid-tumble frame with a hand grazing a rail is not a hang. |
| **rise, persistent ≥ 0.5 s** | 0.026 | Nearly right. |
| **rise, persistent ≥ 0.5 s, load-bearing ≥ 0.5 × weight** | **0.021 ± 0.009** | **Adopted.** He is genuinely hanging on the ladder with his own weight on it. |

The conjunction is what makes the metric un-hackable. It requires the **ladder
to be bearing his weight, for long enough to be a hang and not a bounce**. It
credits nothing for jumping (no rung contact), nothing for standing on the
seesaw or a box or the stairs (no rung contact), nothing for leaning on the
ladder with feet on the floor (ground contact), nothing for a tumble that
happens to brush a rail (persistence + force), and nothing for standing on the
platform having arrived some other way (`platform` is in GROUND — arrival is
scored by SUCCESS, not by height).

**ATTEMPT** — maximal interval bracketed by CLIMB×LADDER contact, opened by the
first such contact after ≥ 3.0 s without one, closed by the last such contact
before the next ≥ 3.0 s gap. Peak `H_k = max h(t)` over attempt *k*.

**ENGAGED ATTEMPT** — an attempt with `H_k ≥ 0.25 m` of rise. Brushing the
ladder while walking past is not an attempt. Only engaged attempts enter the
ascent curve.

**FALL** — an engaged attempt ending with torso `vz ≤ −1.5 m/s` at some point in
its final 2 s **and** a GROUND contact within 2 s of close, settling at torso
`z ≤ 0.5 m`. This separates a fall from a controlled climb-down; both are fine
behaviour, only one is the owner's word.

**SUCCESS** — `z(torso) ≥ ladder_height − 0.15` (a rise of ≈ 1.44 m, well beyond
the measured random ceiling of 0.83 m) **and** torso xy inside the platform
footprint **and** ≥ 3 *distinct* rung geoms contacted during the attempt. The
third clause forbids arriving by any route that is not climbing.

**APPLE-TOUCH** — any body geom contacts the `apple` geom while torso
`z ≥ 1.0`. Secondary; the owner's literal image, reported not gated.

### 2.5 The six observables

Every one is a *ratio to a measured baseline* or a *trend*, never a raw level.
Raw levels are what get gamed.

| # | Observable | Definition | Threshold | Why it cannot be faked |
|---|---|---|---|---|
| **O1** | **Visitation lift** | (fraction of decisions with torso within 1.0 m of the ladder base) ÷ same for the NULL arm | ≥ 2.0 vs NULL **and** ≥ 1.5 vs the RANDOM-REWARD arm | The second comparison is the hard one: it controls for "any optimisation pressure makes you wander more". |
| **O2** | **Attempt count** | number of ENGAGED attempts in one life | ≥ 20 | You cannot have a learning curve without attempts. Encodes the owner's "attempt 40". |
| **O3** | **Return lift (distance-matched)** | P(new engaged attempt starts within 60 s \| just fell) ÷ P(new engaged attempt starts in a 60 s window \| torso within 1.5 m of base, not post-fall) | ≥ 2.0 | **The signature of "trying to figure it out".** Distance-matching removes the trivial confound that after a fall he is already next to the ladder. It is a *self-matched* ratio, so it is immune to between-arm differences in overall activity. Null ≈ 1.0 by construction. |
| **O4** | **Ascent gain + trend + ceiling** | mean `H` over the final quintile of engaged attempts − mean over the first quintile; Spearman ρ(`H_k`, k); and the final-quintile mean itself | gain ≥ 0.35 m **and** ρ ≥ 0.35 at p < 0.01 **and** final-quintile mean ≥ 0.85 m | Random flailing has a *distribution*, not a *trend*. The 0.85 m clause is calibrated directly against measurement: the **single best of 800 random 3 s bursts reached 0.83 m of rise**, so requiring the *mean* of the final quintile to exceed the *maximum* a random agent ever produced makes "occasionally ascends" arithmetically impossible. |
| **O5** | **Success** | ≥ 1 SUCCESS event, in ≥ 2 of 3 seeds; report the attempt index of first success | gated | The owner's sentence ends with him making it. |
| **O6** | **Panel dwell** | fraction of decisions within 2.0 m of the noise panel, **per seed** | ≤ 0.15, else **DISQUALIFIED** | PG.4's own control threshold. Makes the ladder score non-purchasable by a surprise-seeker. |

**Headline metric for the ledger:** `unforced_ascent_gain` (O4's first term, in
metres). Single number, but the spec's `_check` is the **conjunction** of all
six — the repo's existing style (PG.3, PG.4).

**Retention is a separate spec (LT.05), and it has a trap in it.** Two measures:

- **R1 — capability retention (gated).** Intrinsic module removed, reward ≡ 0,
  deterministic policy, 10 episodes. Best `H` must be ≥ 0.8 × the best `H`
  reached during training, and ≥ 1 SUCCESS in 10 episodes.
- **R2 — spontaneous frequency (reported, NOT gated).** Attempts per unit time
  with the bonus off. **This is expected to DROP for a learning-progress arm and
  that is correct behaviour**, not failure: once the ladder is mastered its LP
  goes to zero and he gets bored, exactly as a child does. Gating retention on
  frequency would systematically penalise the mechanism most likely to be right.
  A *rise* in R2 for a novelty arm is likewise not a virtue — it means the bonus
  was never what drove it.

### 2.6 The null baseline: what an uncurious agent scores

**Measured, not assumed** — pilot run on this box, 2026-08-09, 3 seeds ×
3,000 decisions (600 s of simulated life each) of uniform random action on the
climber-rover in the real playground:

| Quantity | seed 0 | seed 1 | seed 2 | Null floor |
|---|---|---|---|---|
| P(rung contact per decision) | 0.000 | 0.059 | 0.011 | 0.023 ± 0.025 |
| Decisions within 1 m of the ladder | 0.000 | 0.216 | 0.141 | 0.119 ± 0.090 |
| **Max ladder-supported rise `H`** | **0.00 m** | **0.00 m** | **0.00 m** | **0.00 m** |
| ENGAGED attempts (`H ≥ 0.25 m`) | 0 | 0 | 0 | **0 in 9,000 decisions** |
| Max torso z reached **without** the ladder | 1.007 m | 0.473 m | 0.767 m | **1.007 m** (`z_rest` = 0.360 m) |
| Panel dwell | 0.000 | 0.000 | 0.000 | 0.000 |

Three things fall out of this table, and all three change the design:

1. **The null floor for the thing we care about is exactly zero.** Zero engaged
   attempts in 9,000 random decisions — 30 minutes of simulated free-roaming
   life across three seeds, and not one weight-bearing hang. So O2 ≥ 20 is not
   an arbitrary bar; it is infinitely far outside what chance produces in a
   free-roaming agent.
2. **A naive "max torso z" metric would have been badly gameable.** Random
   action reached **1.007 m** of torso height with no ladder involvement
   whatsoever (stairs, tumbles, the seesaw) — 56 % of the way to the platform,
   and 0.65 m of "rise" above resting height. This is the empirical
   justification for the `h(t)` conjunction in §2.4, and it was the *first*
   thing the pilot measured.
3. **Seed 0 never came within a metre of the ladder in 600 s.** Visitation is
   itself seed-fragile at the null. Hence O1 is a ratio and every number in this
   test is reported per seed — the PG.4 lesson (`0.667 ± 0.471`) applied.

### 2.7 The credit-assignment gap — and the good news in it

A second pilot placed the rover *at the ladder base, hands at rung height*, and
asked how often 3 s of random action produces a genuine hang. 400 bursts × 2
seeds, three definitions side by side (this is the experiment that produced the
table in §2.4):

| | seed 0 | seed 1 | mean |
|---|---|---|---|
| P(hang, instantaneous defn) | 0.068 | 0.058 | 0.063 |
| P(hang, persistent ≥ 0.5 s) | 0.033 | 0.018 | 0.026 |
| **P(hang, persistent + load-bearing)** | **0.030** | **0.013** | **0.021 ± 0.009** |
| Best rise achieved in any burst | 0.674 m | 0.830 m | **0.83 m ceiling** |

**This is the single most consequential measurement in the document, and it is
good news.** Read it against the §2.6 table:

- **From the ladder base, a real weight-bearing hang is ~1-in-50 by pure chance.**
  The first success is therefore *reachable*, which means learning progress has
  something to select over and the §1.2 failure mode 5 ("LP is zero where
  competence is zero") does **not** bite. LP does not need a Go-Explore archive
  to bootstrap. That was the main architectural risk and the measurement retires
  it.
- **Yet a free-roaming random agent achieves zero hangs in 9,000 decisions.** So
  the difficulty is not the hang. It decomposes into two separable problems:
  **approach** (get to the base and stay there — random spends 0–22 % of its
  life within a metre) and **commitment** (spend a contiguous ~3 s trying rather
  than wandering off). An intrinsic signal that solves *those two* gets the hang
  almost for free.
- **The random ceiling is 0.83 m of rise** — about 2.8 rung spacings, over 800
  bursts. Everything above that is not luck. This is where O4's 0.85 m
  final-quintile threshold and the 1.44 m SUCCESS bar come from.

Design consequence, stated plainly: **the bakeoff is really a test of whether an
intrinsic signal produces approach-and-commitment.** Arms should be judged on
O1 (visitation) and O3 (return) as much as on O4, and an arm that scores O1 and
O3 but not O4 has still told us something true.

### 2.8 The controls that must fail

| Control | What it is | Must score |
|---|---|---|
| **C-NULL** | Uniform random action, and random-*repeat* action sequences (flailing covers ground too — CU.1's null). | O2 = 0, O4 ≈ 0, O3 ≈ 1.0 |
| **C-RANDREW** | Identical learner, identical architecture, identical optimiser; reward = a **fixed random stationary projection** of the state vector. Controls for "any optimisation pressure explores". | O1 lift < 1.5× vs the winning arm |
| **C-ICM** | The trap victim. Must **fixate on the panel in this rig**, not merely in PG.4's rover rig. | O6 > 0.4 → confirms the trap is live here |
| **C-SHUFFLED-GOALS** | (goal-based arms only) identical training with goal labels shuffled — CURIOSITY.md's critical null for the hindsight weld. | O4 ≈ 0 |
| **C-NOLADDER** | The identical world with the ladder geoms deleted. Every metric must be undefined/zero, and **coverage must not collapse** — if it does, the arm was only ever doing one thing. | sanity |

If C-ICM does *not* fixate in the Ladder-Test rig, the rig's noise realisation is
broken and **no arm's O6 = 0 result may be reported as immunity**. This is the
same logic PG.4 used on itself.

### 2.9 Anti-gaming provisions

| # | Attack | Provision |
|---|---|---|
| **G1** | *A hand-coded climb reward — instruction dressed as curiosity.* | **(a) Static audit, executed inside the test and recorded as a metric.** The arm's intrinsic-reward module is parsed; a match on any of `ladder, rung, rail, apple, platform, climb, height, up, torso_z, qpos\[2\], xipos\[.\]\[2\]` in the reward path sets `reward_audit_clean = 0` → **ERROR, not FAIL** (the spec is void, not falsified). **(b) Runtime assertion** `env_reward_absmax == 0.0` — the environment returns literally zero to the policy, always. **(c) LT.06**: the same unmodified code in a world with the ladder moved and reshaped. |
| **G2** | *Random flailing that occasionally ascends.* | C-NULL (measured: 0 engaged attempts in 9,000 decisions). Plus O4's Spearman clause — flailing has no trend — and O3's distance-matched return lift, which flailing cannot produce because it has no memory of falling. |
| **G3** | *The noisy-TV trap.* | O6 disqualifier at 0.15, **per seed**; the panel is mandatory in every arm's world; C-ICM must fixate to prove the trap is live in this rig. |
| **G4** | *Reward-hacking the height sensor — jumping, stairs, the seesaw, standing on a box, a tumble that grazes a rail.* | The three-clause `h(t)` conjunction (§2.4), which was **built by attacking it**: absolute-z scored 0.55 under random action, instantaneous-rise 0.063, persistent+load-bearing 0.021. The pilot's non-ladder height ceiling is **1.007 m**, so a raw-height metric would have been over half gameable. LT.01 re-measures both ceilings per world mutation; if any mutation lets a non-ladder route exceed the ladder-supported record, that mutation is rejected. |
| **G5** | *Seed luck.* | 3 seeds; O5 required in ≥ 2 of 3; per-seed reporting mandatory for every observable; mean ± std for continuous ones. PG.4's `0.667 ± 0.471` is the standing cautionary precedent. |
| **G6** | *Threshold fiddling after the fact.* | All thresholds in §2.5 were fixed from the §2.6 pilot **before any arm ran**, and are written here with the pilot numbers alongside them. The `_check` function is written before the run, per `protocol.py`. |
| **G7** | *Experimenter curriculum leakage.* | The world is drawn per seed by `PlaygroundParams.mutate()`. No arm sees a hand-picked world. Episodes never reset him to the ladder base. The apple carries no reward. |
| **G8** | *Selection over arms — running several arms and reporting the winner.* | Screening (LT.03) and arbitration (LT.04) are **separate specs**: LT.03 tests each arm against the null and declares no winner; only arms that clear it enter `run_bakeoff`. All arms' full metric tables are recorded either way. The winner is then **re-run at 3 fresh seeds** (LT.07) at the pre-registered thresholds before any claim is made — 40 CPU-minutes to remove the argument entirely. |
| **G9** | *Self-generated chaos — the agent thrashes its own body to farm irreducible proprioceptive surprise, with panel dwell reading 0.000 the whole time.* | **§2.10, in full.** `chaos_occupancy` (is he living in irreducibly unpredictable states, on a ruler set by the null?) AND `chaos_reward_ratio` (is that what pays him?) → **VOID**, plus a model-free `thrash_ratio` as the independent second signal. Certified by LT.02 against a positive control (`ragdoll-icm`, which PG.4 scores as perfectly clean) and a negative control (the scripted climber, which must NOT trip it). **This is the failure mode PG.4 is structurally blind to**, and without this check every arm's "immunity" claim is worth half of what it appears to be. |

### 2.10 THE SELF-GENERATED CHAOS CHECK — the gap PG.4 cannot see

**The hole.** PG.4 catches curiosity captured by *external* noise: a panel, at a
fixed location, so dwell time within 2 m of it is a valid detector. It is blind
to the other half of the failure. An agent that **thrashes its own body** —
rapid uncorrelated torques, tumbling, stick-slip self-collision — manufactures
irreducible unpredictability in its own proprioception and farms surprise from
it. That failure has **no spatial signature**: panel dwell reads 0.000 and the
arm is certified "immune" while it spends its life convulsing on the floor.

This is not hypothetical on three counts. `CURIOSITY.md:24` already names it
("fails on action-conditioned noise — physics chaos Jack himself causes") and
then never operationalises it. It is ICM's specific documented weakness, as
distinct from the *uncontrollable* noise inverse-dynamics features do filter.
And RGSD (2510.06203) **measured it on a humanoid**: *"joints move randomly …
arms, legs, torso, and head move independently and arbitrarily."* That sentence
is a description of an agent that has found self-generated chaos.

So: a second detector, spatially agnostic, built on the same two-clause logic
PG.4 uses (*it lives there* **and** *its reward is fed by there*).

#### The construction

Let `D_arm` be an arm's **late-half** decisions (the second 25,000 of 50,000),
each a transition `(s_t, a_t, s_{t+1})` with the arm's own intrinsic reward `r_t`.

**Step 1 — a shared irreducibility ruler.** For each seed, pool the late-half
transitions of *every* arm and the null, and fit one fresh forward model
`g: (s,a) → s'` (2×256 MLP, Adam 1e-3) by 5-fold cross-validation. Fit it
**twice**: once on half the pooled training data (`e^half`) and once on all of
it (`e^full`), evaluated out-of-fold both times. Pooling is what makes the
errors comparable across arms; out-of-fold is what makes them honest.

**Step 2 — separate ignorance from noise.** High prediction error alone does
*not* indicate chaos: a genuinely exploratory agent visits novel states, and
novelty raises held-out error too. Penalising that would penalise exactly the
behaviour we want. The discriminator is whether the error **goes away when the
model gets more data**:

```
reducibility_t = (e_t^half − e_t^full) / max(e_t^half, eps)

CHAOTIC(t)  iff   e_t^full >= θ        (as unpredictable as the worst the null produces)
             and  reducibility_t < 0.1  (and doubling the data did not help)
```

`θ` is fixed **once per seed** as the 90th percentile of `e^full` over the
**null's** transitions. So the null scores `chaos_occupancy ≈ 0.10` by
construction and every arm is measured on the same ruler. This "high error *and*
no learning progress" test is LPM's criterion (arXiv:2509.25438, ICLR 2026)
repurposed as a diagnostic rather than as a reward, which is the one use of it
that needs no continuous-control track record.

**Step 3 — the two numbers, mirroring PG.4 exactly.**

| Metric | Definition | PG.4 analogue |
|---|---|---|
| `chaos_occupancy` | (fraction of the arm's late decisions that are CHAOTIC) ÷ 0.10 | `icm_dwell_share` |
| `chaos_reward_ratio` | mean(`r_t` \| CHAOTIC) ÷ mean(`r_t` \| not CHAOTIC), each arm's `r` normalised by its own mean, **reported with both means and clipped at 1e3** | `panel_reward_ratio` |
| `thrash_ratio` | mean ‖`a_t` − `a_{t−1}`‖₁ ÷ action range, ÷ the null's value. Model-free, independent | (none — new) |

The clip is not cosmetic: PG.4's `panel_reward_ratio` came back as
**641,131,327** because its out-of-zone error had collapsed to ~0. An unbounded
ratio is a number nobody can threshold. Report both means; clip the ratio.

`thrash_ratio` is deliberately a *second, independent* signal — no model, no
pooling, no fitting — because `LESSONS.md` ("when one measurement cannot
distinguish two cases, find the second independent signal"). Random-per-decision
action gives ≈ 1.0; a coordinated climber is far below it.

#### The disqualification rule, and why it is VOID and not FAIL

```
chaos_occupancy >= 3.0   AND   chaos_reward_ratio >= 2.0
   →  this arm's LT.03 result is Status.VOID
```

**Conjunction, not disjunction.** Occupancy alone would catch an agent that
tumbles a lot — and *falling off the ladder is the owner's own word*, the
behaviour the whole project wants. The reward clause is what establishes that
the chaos is what is **paying** him. Same two-clause structure PG.4 used
(`icm_dwell_share` AND `panel_reward_ratio`), for the same reason.

**VOID, not FAIL**, per `protocol.py`: the arm's curiosity signal has
*degenerated*, so the run did not test the claim "this mechanism drives
structured exploration" — it demonstrated the mechanism collapsed into
self-noise. FAIL would assert that the hypothesis was tested and lost, which is
false, and a spec's `kills` field reads machine-side off that distinction. A
VOID arm also cannot enter the LT.04 bakeoff, so fewer than two surviving arms
blocks the decision rather than resolving it — the T2.02 discipline.

For the `null` arm there is no intrinsic reward, so `chaos_reward_ratio` is
`n/a` and only occupancy is reported.

#### Certifying the detector (LT.02) — it needs its own positive control

`LESSONS.md`: *"a detector that cannot see its own positive control has measured
nothing"*, and T0.13 returned a clean scan on a known-bad fixture because a
clean scan and a scan that never ran are the same number. So the detector is
certified before any arm's clean score may be reported, exactly as PG.4 certifies
the noise panel before CU.3 may claim immunity.

| Control | What it is | Must read |
|---|---|---|
| **POSITIVE — `ragdoll-icm`** | The ICM agent in a world with the **noise panel deleted** and **adhesion gain 0**. The only irreducible source left in the universe is its own contact chaos. | `chaos_occupancy ≥ 3.0` **and** `chaos_reward_ratio ≥ 2.0` — **and `panel_dwell = 0.000`** |
| **NEGATIVE — `scripted-climber`** | LT.01's deterministic scripted climb: high-motion, coordinated, and it falls off the ladder repeatedly. | `chaos_occupancy ≤ 1.0` |
| **CROSS-CHECK — `icm` with the panel present** | The PG.4 trap victim. | flagged by **both** detectors: `panel_dwell > 0.4` **and** `chaos_occupancy ≥ 3.0` |

The positive control is the spec's whole point, and it fits in one sentence:
**`ragdoll-icm` exhibits a total curiosity failure that PG.4 scores as perfectly
clean (`panel_dwell = 0.000`).** That number pair *is* the gap, demonstrated.

The negative control is the one that could genuinely fail. If the scripted
climber trips the detector, the detector is measuring "the body moves and
sometimes tumbles" rather than "the agent farms noise" — and it would disqualify
the exact behaviour GOAL.md asks for. That is a real falsification risk, which
is what makes LT.02 a test rather than a formality.

#### Which arms it is wired into

**All of them**, per seed, reported alongside panel dwell — it is a post-hoc
analysis of stored trajectories and costs nothing extra per arm. It matters most
for `icm`, `rnd`, `disagree` and `metra`, whose rewards are all functions of
prediction error or state statistics. It is *not* vacuous for `lp`: its outcome
space contains proprioceptive dimensions, so a thrash-reachable region of that
space would be farmable, and this is the only check that would notice.

For `metra` specifically, LT.02 + LT.03 together give a **quantitative
replication of RGSD's finding in Jack's world for about ten CPU-minutes** —
turning "skill discovery flails on high-DoF bodies" from a citation into a
measurement on our own body.

### 2.11 What would falsify the whole approach

Stated plainly, because §5 is built to find these cheaply:

- **F1** — No arm produces a single engaged attempt (`H ≥ 0.25`) in a full life.
  Then intrinsic motivation does not reach the ladder at all, and the answer is
  a structured goal/skill layer (PEG-style goal planning, or a Go-Explore
  archive), not a better bonus.
- **F2** — Arms produce attempts but no trend (O4 fails everywhere). Then the
  bottleneck is credit assignment over a long horizon, not exploration —
  consistent with the Horizon Reduction line (NeurIPS 2025) — and the fix is
  hindsight relabeling density, not curiosity.
- **F3** — The only arm that passes also fails O6. Then curiosity in this world
  is inseparable from the noisy-TV trap, and the project needs a different
  signal class entirely.
- **F4** — Everything passes on the climber-rover and nothing transfers to the
  humanoid at any affordable budget. Then the claim is honestly scoped down to
  "curiosity climbs on a reduced body", and the humanoid claim waits for a
  batched simulator (§6).

---

## 3. THE BAKEOFF

### 3.1 The arms

Six arms. Every arm shares the same body, the same world seeds, the same
observation vector, the same episode budget, the same tiny policy network
(≈150 K params — **not** the 45.5 M `UnifiedBrain`; §6 explains why this is what
makes the bakeoff free), and the same six observables. Only the intrinsic
objective differs.

#### The split that `experiments/bakeoff.py` forces: candidates vs controls

`run_bakeoff` **VOIDs the entire bakeoff if any arm falls below the 3σ learning
gate** — the T2.02 rule, "two non-learners cannot arbitrate an architecture".
That rule and the composition of this bakeoff are in direct conflict, and the
conflict has to be resolved on paper before anything runs:

> `icm` and `rnd` are **expected and required to fail**. They are the trap
> victims CU.3 depends on. Passing them to `run_bakeoff` as `Arm`s would VOID
> every bakeoff this project ever runs on curiosity — permanently, by design.

So they are **not arms. They are controls.** The separation is not bookkeeping;
it is what keeps the learning gate meaningful:

| Role | Members | Primitive | Rule |
|---|---|---|---|
| **CANDIDATES** (`Arm`s) | `disagree`, `lp`, `metra` (+`vlm-lp` conditionally) | `run_bakeoff` in **LT.04** | must clear the 3σ gate; winner by 1.5σ margin; TIE → cheapest |
| **NULL** | `null` (random, random-repeat) | `null_run=` | defines 0 |
| **CONTROLS THAT MUST FAIL** | `icm`, `rnd`, `randrew`, `shuffled-goals`, `ragdoll-icm` | `run_spec(control_fn=)` in **LT.02 / LT.03** | each must land on its pre-registered side |

And screening is a **separate spec from arbitration**: LT.03 tests every
candidate against the null and declares *no winner*; LT.04 arbitrates among only
those that cleared it. If fewer than two clear, LT.04 records VOID with reason
"fewer than two learners" — which is true, expressible, and blocks the decision
instead of manufacturing one. Given RGSD (2510.06203), `metra` failing the gate
is a live possibility, and this structure means that outcome costs us a *known
negative result* rather than an unusable bakeoff.

#### The cost unit

`Arm.cost` is undeclared by default and an undeclared cost VOIDs a TIE, so the
unit must be named here, before the run:

> **`cost` = CPU-core-seconds of *learner* time per 1,000 decisions of lived
> experience**, measured in-run via `time.process_time()` deltas around the
> intrinsic-reward and policy-update calls, **excluding MuJoCo**.

Three reasons for that unit rather than parameter count. MuJoCo is identical
across arms, so including it would compress the very differences the tie-break
needs. Jack is meant to run *continuously* on 4 shared cores, so the binding
resource is core-seconds per unit of life, not memory. And it is **measured, not
declared** — `LESSONS.md`: "a default of zero is not unknown".

| Arm | Role | Objective | Implementation in this repo | Params | `cost` (CPU-core-s / 1k decisions, estimated pre-run) | Prior | Expected outcome |
|---|---|---|---|---|---|---|---|
| `null` | NULL | none | random and random-repeat action | 0 | **0.0** | — | 0 engaged attempts (measured, §2.6) |
| `icm` | **control (must fail)** | forward-model error on inverse-dynamics features | `UnifiedBrain.IntrinsicCuriosityModule.compute_icm_reward`, standalone on the 76-d observation | ~40 K | ~3.5 | 1705.05363 | **trapped**: O6 > 0.4 (panel) — required by CU.3 |
| `rnd` | **control (must fail)** | ‖predictor − frozen random target‖ | `IntrinsicCuriosityModule` RND branch, same module | ~40 K | ~2.5 | 1810.12894 | trapped: the panel's observation never repeats, so the predictor never converges |
| `randrew` | **control (must fail)** | fixed random stationary projection of the state | 20 lines | ~5 K | ~0.5 | — | controls for "any optimisation pressure explores" |
| `ragdoll-icm` | **control (must be FLAGGED)** | `icm`, panel deleted, adhesion gain 0 | as `icm` | ~40 K | ~3.5 | §2.10 | `chaos_occupancy ≥ 3.0` with `panel_dwell = 0.000` |
| `scripted-climber` | **control (must NOT be flagged)** | LT.01's deterministic script | LT.01 | 0 | 0.0 | §2.10 | `chaos_occupancy ≤ 1.0` |
| **`disagree`** | **CANDIDATE** | variance across a 5-member one-step forward-model ensemble | new, ~120 lines; five 2×64 MLPs, different inits, 0.8 bootstrap masks | 5 × 12 K | **~9.0** | 2005.05960, 1906.04161 | the best-evidenced noisy-TV fix; risk is the ensemble sharing so much data that disagreement collapses *everywhere*, not only on noise |
| **`lp`** | **CANDIDATE** | absolute learning progress over an auto-partitioned outcome space, with hindsight relabeling | `UnifiedBrain.AutotelicGoalGenerator` (`strategy="learning_progress"`) + SAGG-RIAC region split; goal space = (torso xyz, `z\|no-ground-contact`, climb-contact count, nearest-object displacement) | ~90 K | **~2.0** | 1910.07224, 1708.02190, 2502.07709 | **the favourite, and the cheapest candidate** — but LP is a *selector*, not a discoverer (§1.2 failure mode 5) |
| **`metra`** | **CANDIDATE** | METRA temporal-distance skill latents + LP selection over skills | new; 2-d latent, 16 skills, small Gaussian policy on the same observation | ~200 K | **~14.0** | 2310.08887 | RGSD (2510.06203) predicts travel directions only. Most at risk of both the learning gate *and* the chaos check |
| `vlm-lp` | CANDIDATE *(conditional, LT.09)* | frozen VLM proposes goals **as predicates in `lp`'s outcome space**; LP has the final vote | new; 4 snapshots per 2,000 decisions → captions → ~10 (goal-text, predicate) pairs → the LP buffer | LLM frozen | **~2.3** | 2306.01711, 2302.06692, 2310.00166 | run only if `lp` wins LT.04. The VLM may never emit a reward or code — that is Eureka, i.e. instruction (§1.5) |

Note that `icm`, `rnd` and `lp` **instantiate modules that already exist in
`UnifiedBrain.py` and have never received a gradient in any test**
(`protocol.py`: "45.5M parameters with no live call site"). This programme is
therefore also the evidence that decides T3.06 — those modules either earn their
parameters here or get deleted.

**The cost column is a pre-run estimate and must be overwritten with the
measured value before LT.04 arbitrates.** If a TIE lands on estimates, the
bakeoff has resolved a coin flip with a guess. Note that `lp` is both the
favourite and 4.5× cheaper than `disagree` and 7× cheaper than `metra`, so a TIE
between `lp` and anything else resolves to `lp` — which makes the measurement
matter most exactly where the arms are closest.

### 3.2 The critical design decision the pilots force

§2.7's measurement determines whether `lp` can work alone. Because LP is zero for
a never-achieved goal, the outcome space **must include a dimension that is
non-zero for partial ladder engagement** — specifically
`z_torso | no ground contact`. With that dimension, hindsight relabeling turns
every failed grab into a success at a small value of it, LP becomes measurable
at *0.05 m of hang*, and the curriculum can climb continuously from brushing the
ladder to hanging to ascending. Without it, the first success has probability
zero and LP has nothing to select over.

§2.7 also says this is *sufficient*: a real hang occurs in 2.1 % of 3 s random
bursts from the base, so first successes exist and LP has something to select
over. That retires the main architectural risk and means no Go-Explore archive
is needed to bootstrap.

This is a genuine design choice with a gaming risk attached, and it must be
declared: the outcome-space dimension `z | no-ground-contact` is *close* to the
metric `h(t)`. The defence is that (a) it does not reference the ladder — it is
"how high am I with nothing under me", which is equally satisfied by jumping off
the stairs, and G1's symbol audit permits it on exactly those grounds; (b) LT.06
moves the ladder; (c) `disagree` and `metra` do not use it at all, so if only
`lp` passes we know the dimension is doing the work and must say so.

**And it is the dimension the chaos check must watch.** `z | no-ground-contact`
is maximised by being airborne, and being airborne is also how a body generates
contact chaos. If `lp` scores well on O4 *and* trips G9, the outcome dimension
was farmable and the arm is VOID, not a winner. This is the one place where the
two provisions of §2.10 and §3.2 have to be read together.

### 3.3 Registry entries — exact `Spec(...)` format

To be appended to `EXPANSION` in `experiments/registry_expansion.py`. Tier
mapping follows the existing convention (fixtures → 2, claims → 5).

```python
    # ── THE LADDER TEST (docs/research/CURIOSITY_BAKEOFF.md) ────────────
    # Two-digit ids on purpose. `run.py::_module_for` globs `lt_1_*.py`, which
    # would also match `lt_10_*.py`, and its hierarchical-id escape hatch does
    # NOT cover that case (it tests startswith("lt_1_"), and "lt_10" fails it).
    # The same latent collision exists today between UB.1 and UB.16. LT.01-LT.99
    # is structurally immune. See LESSONS.md, "A spec id that is a prefix of
    # another spec id disables one of them".

    Spec("LT.01", 2, "The Ladder Test is measurable: null floor and un-gameable rise",
         hypothesis="A free-roaming random climber-rover produces ZERO engaged "
                    "ladder attempts, while reaching >=0.6 m of torso RISE by "
                    "non-ladder routes; and from the ladder base a genuine "
                    "weight-bearing hang occurs in 1-5% of 3 s random bursts — "
                    "so ladder-supported rise (contact AND airborne AND held "
                    ">=0.5 s AND load-bearing) discriminates, raw torso z does "
                    "not, and the first success is reachable by chance.",
         falsified_by="A free-roaming random agent produces engaged attempts "
                      "(the null floor is not zero), or a non-ladder route "
                      "reaches the platform, or P(hang from the base) is 0 in "
                      "800 bursts (no bootstrap exists and no learning-progress "
                      "method can work without an archive).",
         null_baseline="n/a — this spec IS the null floor measurement.",
         metric="null_engaged_attempts", budget=Budget.CPU_LONG,
         depends_on=["PG.1", "PG.3", "PG.4"], seeds=3,
         control="A greedy height-maximising oracle with adhesion DISABLED must "
                 "still be unable to reach the platform — else an alternate "
                 "route exists and SUCCESS is not evidence of climbing.",
         kills="The entire Ladder Test, before a single arm is trained. Costs "
               "20 CPU-minutes; every threshold in the programme is set from it.",
         notes="Pilot 2026-08-09 (aarch64, mujoco 3.2.3). Free-roaming: 0 "
               "engaged attempts in 9,000 random decisions; max NON-ladder "
               "torso z 1.007 m against z_rest 0.360 m. From the base, 800 x "
               "3 s bursts: P(hang) = 0.55 under an ABSOLUTE-z definition "
               "(broken - z_rest already clears the bar), 0.063 instantaneous, "
               "0.026 persistent, 0.021 +- 0.009 persistent AND load-bearing; "
               "random rise ceiling 0.83 m. Those four numbers ARE the "
               "definition of h(t) and every threshold in LT.03."),

    Spec("LT.02", 2, "The self-generated-chaos detector works (PG.4's blind spot)",
         hypothesis="A curiosity agent can farm irreducible surprise from its "
                    "OWN body with zero noise-panel dwell, and the chaos "
                    "detector sees it: ragdoll-ICM (panel deleted, adhesion 0) "
                    "scores chaos_occupancy >= 3.0 and chaos_reward_ratio >= "
                    "2.0 while PG.4's dwell metric reads 0.000, and the "
                    "scripted climber — which moves hard and falls repeatedly — "
                    "scores chaos_occupancy <= 1.0.",
         falsified_by="Ragdoll-ICM is NOT flagged (the detector is blind and no "
                      "arm's immunity may be reported), or the scripted climber "
                      "IS flagged (the detector penalises coordinated motion and "
                      "falling, i.e. the behaviour GOAL.md asks for).",
         null_baseline="The random policy, which DEFINES the ruler: theta is its "
                       "90th-percentile irreducible error, so it reads "
                       "chaos_occupancy = 1.0 by construction.",
         metric="chaos_detector_separation", budget=Budget.CPU_LONG,
         depends_on=["LT.01", "PG.4"], seeds=3,
         control="Cross-check: the ICM agent WITH the panel present must be "
                 "flagged by BOTH detectors (panel_dwell > 0.4 AND "
                 "chaos_occupancy >= 3.0). Two independent detectors must agree "
                 "on a known positive, or one of them is reading noise.",
         kills="Every 'his curiosity is not trapped' claim that rests on panel "
               "dwell alone — which is all of them, including CU.3 as currently "
               "written.",
         notes="separation = chaos_occupancy(ragdoll-icm) - "
               "chaos_occupancy(scripted-climber), with panel_dwell(ragdoll-icm) "
               "asserted == 0.0. That number pair IS the gap: a total curiosity "
               "failure that PG.4 scores as perfectly clean. Detector = "
               "pooled-fit forward model, out-of-fold, high error AND no "
               "reducibility when the training data doubles (LPM criterion, "
               "arXiv:2509.25438, used as a diagnostic not a reward). "
               "thrash_ratio is reported as the model-free second signal."),

    Spec("LT.03", 5, "THE LADDER TEST: curiosity alone climbs the ladder",
         hypothesis="With the environment returning reward identically zero, at "
                    "least one candidate arm produces >=20 engaged ladder "
                    "attempts, a distance-matched post-fall return lift >= 2.0, "
                    "an ascent gain >= 0.35 m with Spearman rho >= 0.35 (p<0.01) "
                    "and a final-quintile mean rise >= 0.85 m (above the "
                    "measured random ceiling of 0.83 m), and at least one "
                    "topping-out, in >=2 of 3 seeds — while dwelling <= 0.15 at "
                    "the noise panel in every seed and never tripping the "
                    "self-generated-chaos check.",
         falsified_by="No arm produces a single engaged attempt (exploration "
                      "never reaches the ladder), or attempts occur with no "
                      "ascent trend (credit assignment, not curiosity, is the "
                      "bottleneck), or every arm that climbs also fixates on the "
                      "panel or farms its own body noise.",
         null_baseline="Random and random-repeat action: measured at 0 engaged "
                       "attempts in 9,000 decisions (LT.01). Plus randrew, a "
                       "random-stationary-reward learner at matched compute, "
                       "which controls for 'any optimisation pressure explores'.",
         metric="unforced_ascent_gain", budget=Budget.CPU_LONG,
         depends_on=["LT.01", "LT.02", "PG.4"], seeds=3,
         control="Three, each must land on its declared side: (1) the ICM "
                 "control MUST fixate on the panel in THIS rig (dwell > 0.4) — "
                 "proving the trap is live here and not only in PG.4's rover; "
                 "(2) randrew must not match the winner's visitation lift; "
                 "(3) a goal-shuffled variant of the winning arm must show no "
                 "ascent trend.",
         kills="The 'intrinsic motivation is enough' thesis for structured "
               "vertical behaviour. If it fails, GOAL.md's ladder image needs a "
               "goal/skill layer (PEG 2303.13002, or a Go-Explore archive over "
               "h(t)-bearing states — PG.3 already certified the state restore "
               "it needs at resume_max_dev 0.0), and that pivot is decided by "
               "this result, not by preference.",
         notes="SCREENING ONLY — each candidate against the null, no winner "
               "declared; arbitration is LT.04, because run_bakeoff VOIDs on a "
               "sub-gate arm and icm/rnd are REQUIRED to fail. An arm whose "
               "chaos_occupancy >= 3.0 AND chaos_reward_ratio >= 2.0 returns "
               "Status.VOID for that arm: its curiosity signal degenerated, so "
               "the run did not test the claim. Every arm's reward code passes "
               "a static audit for ladder-referencing symbols; a match is ERROR. "
               "No published system has done this — LadderMan (2606.05873) "
               "climbs from a human reference motion, METRA on a 69-DoF humanoid "
               "flails (RGSD 2510.06203), and the one time LP curricula were "
               "pointed at a 2D climbing morphology they reached ~1% mastery "
               "(TeachMyAgent 2103.09815)."),

    Spec("LT.04", 5, "Bakeoff: which curiosity mechanism climbs best",
         hypothesis="Among the arms that cleared LT.03, one beats the runner-up "
                    "by >=1.5 sigma of the pooled seed spread on "
                    "unforced_ascent_gain.",
         falsified_by="n/a for a bakeoff — the outcomes are WINNER, TIE (take "
                      "the cheaper arm) or VOID (an arm is below the 3-sigma "
                      "learning gate, so the decision is blocked, not made).",
         null_baseline="Random-repeat action, shared across arms.",
         metric="unforced_ascent_gain", budget=Budget.CPU_LONG,
         depends_on=["LT.03"], seeds=3,
         control="Inherited from LT.03; no arm may enter this bakeoff whose "
                 "LT.03 result was VOID for self-generated chaos.",
         notes="run_bakeoff(arms=[disagree, lp, metra], null_run=random_repeat, "
               "learning_gate_sigma=3.0, margin_sigma=1.5). Arm.cost is declared "
               "in CPU-CORE-SECONDS OF LEARNER TIME PER 1,000 DECISIONS, measured "
               "in-run with time.process_time() around the intrinsic-reward and "
               "policy-update calls and EXCLUDING MuJoCo (identical across arms, "
               "so including it would compress the differences the tie-break "
               "needs). Pre-run estimates: lp 2.0, disagree 9.0, metra 14.0 — a "
               "TIE therefore resolves to lp, which is why the measurement must "
               "replace the estimate before this runs. Fewer than two arms "
               "clearing LT.03 records VOID: 'fewer than two learners'."),

    Spec("LT.05", 5, "The climb survives the curiosity that produced it",
         hypothesis="With the intrinsic module removed and reward identically "
                    "zero, the winning arm's deterministic policy still reaches "
                    ">= 0.8x its best training ladder-supported rise and tops "
                    "out at least once in 10 episodes.",
         falsified_by="Ladder-supported rise collapses without the bonus — then "
                      "the behaviour was bonus-chasing, not a skill.",
         null_baseline="The same policy at initialisation, bonus off.",
         metric="retention_ratio", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="A policy trained with the random-stationary reward (randrew) "
                 "must show no retained climbing — else retention measures "
                 "architecture, not learning.",
         notes="Spontaneous attempt FREQUENCY with the bonus off is reported but "
               "explicitly NOT gated: a learning-progress agent is supposed to "
               "lose interest once the ladder is mastered, exactly as a child "
               "does. Gating on frequency would systematically penalise the "
               "mechanism most likely to be right."),

    Spec("LT.06", 5, "It is the ladder he is curious about, not the coordinates",
         hypothesis="The identical unmodified arm, in a world where the ladder "
                    "is moved, re-yawed and re-spaced, scores >= 0.5x its "
                    "home-world ascent gain.",
         falsified_by="Performance collapses when the ladder moves — the arm "
                      "learned a location, or the reward was hard-coded.",
         null_baseline="Home-world score for the same arm and seed.",
         metric="moved_ladder_ratio", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="A deliberately hard-coded climb reward, written for this "
                 "control only and keyed to the home ladder's xy, MUST fail "
                 "here — that is what makes the spec an instruction detector "
                 "rather than a generalisation test.",
         notes="Together with LT.03's static symbol audit this is the "
               "anti-instruction provision. Eureka-style LLM reward writing "
               "(2310.12931) is caught by exactly this pair."),

    Spec("LT.07", 5, "The winner survives fresh seeds",
         hypothesis="Re-run at 3 seeds never used during screening or "
                     "arbitration, the winning arm clears every one of LT.03's "
                     "six observables at the same pre-registered thresholds.",
         falsified_by="Any observable falls below its LT.03 threshold on fresh "
                      "seeds — the win was selection over arms and seeds.",
         null_baseline="LT.01's null floor, re-measured on the fresh seeds.",
         metric="unforced_ascent_gain", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="The same three controls as LT.03, re-run: the ICM control "
                 "must still fixate on these worlds.",
         kills="Nothing is written to the README before this passes.",
         notes="Seeds 10/11/12. Costs ~40 CPU-minutes and removes the "
               "multiple-comparison argument entirely (G8)."),

    Spec("LT.08", 5, "The humanoid climbs — same test, real body",
         hypothesis="With locomotion in hand, the winning arm reproduces LT.03's "
                    "six observables on the full humanoid in the same playground.",
         falsified_by="Any of LT.03's six clauses fails on the humanoid at the "
                      "budgeted step count with the curve flat.",
         null_baseline="LT.01's nulls, re-measured on the humanoid body.",
         metric="unforced_ascent_gain", budget=Budget.GPU_LONG,
         depends_on=["LT.07", "T2.01", "T2.02"], seeds=3,
         control="Same as LT.03, re-run on this body: the ICM control must "
                 "fixate, and the chaos check matters MORE here — RGSD "
                 "(2510.06203) reports exactly this failure at 69 DoF.",
         kills="Nothing on its own — a FAIL here with LT.03 passing scopes the "
               "claim honestly to the reduced body and points at throughput.",
         notes="BLOCKED until T2.01/T2.02 pass. Also blocked on throughput: at "
               "T2.01's measured ~128 env-steps/s a 20M-step arm-seed costs "
               "43 h, so 3 seeds exceed a whole week of Kaggle quota for ONE "
               "arm. Getting the 45.5M trunk out of the inner loop makes it "
               "MuJoCo-bound at ~2,000 steps/s (~2.8 h/arm-seed). The "
               "prerequisite is a throughput spec, not more quota."),

    Spec("LT.09", 5, "The VLM proposes ladder-shaped goals; learning progress disposes",
         hypothesis="Frozen-VLM-proposed goals, expressed ONLY as predicates in "
                    "the existing outcome space and filtered by LP, reach the "
                    "first engaged ladder attempt in fewer decisions than "
                    "LP-only at matched goal count.",
         falsified_by="No speedup, or VLM goals flood the buffer while their "
                      "achievement stays ~0 (a hallucinated curriculum).",
         null_baseline="LP-only (the lp arm) at matched goal count.",
         metric="time_to_first_engaged_attempt", budget=Budget.CPU_LONG,
         depends_on=["LT.04"], seeds=3,
         control="A scrambled-caption VLM fed a DIFFERENT scene must not beat "
                 "LP-only — else the benefit was 'more goals', not grounded "
                 "interestingness. Additionally the VLM may never emit reward "
                 "code: a proposal that is not a predicate over existing outcome "
                 "dimensions is rejected before it reaches the buffer.",
         notes="Only run if lp wins LT.04. LLM-proposed goals have never driven "
               "low-level continuous control (ELLM 2302.06692 limitations; "
               "OMNI-EPIC 2405.15568 uses 6 discrete actions), so this is the "
               "genuinely unoccupied combination — and the reason it is "
               "unoccupied is a grounding gap, which the predicate restriction "
               "is designed to sidestep."),
```

### 3.4 Per-arm implementation notes

Everything below runs on one ARM core, in float32, with no GPU.

- **Observation (identical for all arms, 76-d):** torso xyz + quaternion (7),
  torso linear/angular velocity (6), arm joint positions and velocities (8),
  adhesion states (2), 8 contact-boolean flags, a 24-ray retina at torso height
  giving `[distance, texture]` per ray — reusing `pg_4_noisy_tv.py::_Retina`
  verbatim, including `R_RESOLVE = 2.5` and the panel re-randomisation. **Do not
  reimplement the retina**; the trap's calibration lives in that class.
- **Decision rate:** 0.2 s (40 substeps at `dt = 0.005`), matching PG.4.
- **Policy:** 2×128 tanh MLP → Gaussian over 6 continuous ctrl + 2 sigmoid
  adhesion. PPO with `T2.00`'s pre-registered sanity guards in force
  (return normalisation, bounded `log_std`, actions clipped to env range) —
  T2.00 exists precisely so a broken update cannot silently invalidate this.
- **Episode:** one continuous life of 50,000 decisions ≈ 2.8 h of simulated
  time. Reset only on numerical divergence. **No reset to the ladder base ever.**
- **`disagree`:** five one-step forward models `f_i(s, a) → s'`, each with its
  own init and a 0.8 bootstrap mask over the replay stream; bonus =
  `var_i f_i(s,a)` averaged over dimensions, normalised by a running std. The
  panel check is the point: disagreement must decay to ≈ 0 there while ICM's
  error stays high.
- **`lp`:** outcome vector `o = (x, y, z, z|no-ground-contact,
  n_climb_contacts, ‖Δp_nearest_object‖)`. Region tree split on outcome variance
  (SAGG-RIAC); per-region ALP = |EMA_recent(success) − EMA_old(success)| over a
  window of 20 attempts, with a minimum of 8 attempts before a region's ALP is
  trusted (failure mode 1). Goal sampling: ε-greedy over ALP with ε = 0.2
  uniform for babbling. Hindsight: every trajectory is relabeled with its own
  achieved `o` and regressed — this is the flow-matching weld (T2.16), and it is
  what makes goal-reaching trainable without RL machinery.
  **Report LP at 3 window lengths (10/20/40); a conclusion that flips is not
  a conclusion** (failure mode 3).
- **`metra`:** 2-d latent, 16 discretised skills, temporal-distance
  constraint per 2310.08887. Ablation of the temporal-distance constraint is
  CU.4's required control and should be run in the same job.
- **`vlm-lp`:** render 4 × 128 px snapshots via osmesa every 2,000 decisions,
  caption with the frozen VLM off-box, parse into predicates over the existing
  outcome dimensions, reject anything else. LLM cost is 4 images per 400 s of
  simulated time — negligible.

### 3.5 What each arm is expected to teach us even if it loses

- `icm`/`rnd` losing to the panel is **the confirmation that PG.4's trap
  generalises to the real body** — required by CU.3, and worth the CPU on its own.
- `metra` producing only travel directions is **a replication of RGSD's finding
  (2510.06203) in Jack's world at 40 CPU-minutes**, which is a publishable-grade
  negative result and settles CU.4's scope.
- `disagree` collapsing everywhere (not only on noise) diagnoses ensemble
  data-sharing, which is a fixable implementation issue, not a refutation.
- `lp` failing *only* for lack of a first success is the strongest possible
  argument for adding a Go-Explore archive (§1.4) — and PG.3 already certified
  the state-restore that Go-Explore needs at `resume_max_dev = 0.0`.

---

## 4. Mandatory controls for every arm

Non-negotiable, applied identically to every arm and control and recorded in the
ledger for each. **Two independent noise detectors, because they fail
differently** — the panel catches noise the world supplies, the chaos check
catches noise the body manufactures, and an arm must be clean on both.

*External noise (PG.4's detector):*

1. **The noise panel is present in every arm's world.** `noise_panel=True` is
   asserted at build time; a run with it disabled is void. The one exception is
   `ragdoll-icm`, whose entire purpose is to have no panel (§2.10).
2. **Panel dwell is reported per seed**, never as a mean alone. PG.4's
   `0.667 ± 0.471` — roughly {1, 1, 0} across three seeds — is why.
3. **Dwell > 0.15 in any seed disqualifies the arm**, regardless of its ladder
   score. A disqualified arm is recorded with its full metrics and the
   disqualification reason — it is data, not a deletion.
4. **The `icm` control must fixate in this rig** (dwell > 0.4). If it does not,
   the noise realisation is broken and no arm's immunity may be claimed.

*Self-generated noise (§2.10's detector — new):*

5. **`chaos_occupancy` and `chaos_reward_ratio` are reported per arm, per seed**,
   next to panel dwell. An arm reporting `panel_dwell = 0.000` and nothing else
   has not demonstrated immunity; it has demonstrated one of the two.
6. **`chaos_occupancy ≥ 3.0` AND `chaos_reward_ratio ≥ 2.0` → that arm is
   `Status.VOID`**, not FAIL. Its curiosity signal degenerated, so the run did
   not test the claim. A VOID arm may not enter LT.04.
7. **`thrash_ratio` is reported always** as the model-free second signal.
8. **LT.02 must PASS before any arm's chaos numbers may be interpreted** — the
   positive control (`ragdoll-icm`) must be flagged and the negative control
   (`scripted-climber`) must not. A detector that has never caught anything
   certifies nothing; this is PG.4's own argument applied to PG.4's blind spot.

*Shared:*

9. **`env_reward_absmax == 0.0`** is asserted every step. The environment never
   returns a reward to any arm.
10. **`reward_audit_clean == 1`** — the static symbol audit of §2.9/G1 passes,
    or the arm's result is ERROR (the spec is void), not FAIL (falsified).
11. **Coverage is reported alongside both detectors** (`visited_cell_frac`,
    PG.4's metric). An arm that avoids the panel by exploring nothing is not
    immune, it is inert — CU.3's `coverage_vs_dwell` exists for exactly this,
    and the same logic now applies to an arm that avoids chaos by never moving.

---

## 5. The staged plan: cheapest falsifier first

The ordering rule is the ladder's own: *a hypothesis must die before it costs
GPU quota* (`protocol.py::Budget`).

### Stage 0 — THE ~20-MINUTE CPU EXPERIMENT THAT COULD KILL THE WHOLE THING

**LT.01 — the null-floor and metric-gameability probe. Already piloted; §2.6 and
§2.7 are its output.** Formalise the pilot into
`experiments/tests/lt_01_null_floor.py` and run it at 3 seeds. Four questions,
answered before a single parameter is trained:

| Question | Answer, measured | What it decided |
|---|---|---|
| Is the null floor actually zero? | **Yes** — 0 engaged attempts in 9,000 random decisions | O2 ≥ 20 is not an arbitrary bar |
| Is raw height gameable? | **Yes** — 1.007 m torso z with no ladder involvement | forced the three-clause `h(t)`, killed two earlier definitions |
| Is the *first success* reachable by chance? | **Yes** — 2.1 % of 3 s random bursts from the base | **retired the main architectural risk**: LP can bootstrap, no Go-Explore archive needed |
| Where is the random ceiling? | **0.83 m of rise** over 800 bursts | set O4's 0.85 m threshold and the 1.44 m SUCCESS bar |

**Cost: ~20 CPU-minutes, zero GPU. Kills or reshapes the entire programme.**
Re-confirmed as the correct first experiment, and it has *already run* — the
four numbers above are measurements from this box on 2026-08-09, not estimates.
Its most valuable output was negative and self-inflicted: the first two versions
of the headline observable were gameable (0.55 and 0.063 false-positive rates
under random action), and twenty minutes of CPU found that before any GPU hour
could have been spent measuring the wrong thing.

Note what Stage 0 did *not* do: it did not tell us curiosity works. It told us
the test is honest and the target is reachable. That is the correct job for the
cheapest experiment.

### Stage 1 — certify the body and the detectors (CPU, ~1.5 core-hours)

**LT.01** (above) → **LT.02** (the self-generated-chaos detector, §2.10). LT.02
is the addition this revision makes to the staging, and it belongs *before* the
bakeoff for the same reason PG.4 came before CU.3: a detector that has never
caught anything certifies nothing, and until `ragdoll-icm` is flagged while
reading `panel_dwell = 0.000`, every "his curiosity is not trapped" claim in the
project rests on half a measurement.

Also in Stage 1: script the approach-and-climb on the climber-rover (folded into
LT.01's control) — if the reduced body cannot be *scripted* up the ladder, no arm
will learn it and the body needs redesign before any learning runs. PG.3's logic
applied one level up. The scripted climber then doubles as LT.02's negative
control, which is why the two specs share a stage.

### Stage 2 — screening, then arbitration (CPU, ~3.5 core-hours; **zero GPU**)

**LT.03 (screening).** 3 candidate arms + `null` + `randrew`, × 3 seeds, 50,000
decisions each. Per `LESSONS.md` ("the budget names one experiment; the spec runs
seeds × arms"): 5 × 3 × 14 min = **3.5 core-hours**, ~1.2 h wall at 3 parallel
workers. `icm` is not re-run here — LT.02 already ran it in this rig with the
panel present, which is exactly what `depends_on=["LT.02"]` buys.

**LT.04 (arbitration).** `run_bakeoff` over only those arms that cleared LT.03,
with measured costs substituted for the §3.1 estimates. Reuses LT.03's
trajectories; near-zero marginal cost.

Decision gate after Stage 2:
- **≥ 2 arms clear the gate** → LT.04 arbitrates → Stage 3.
- **Exactly 1 arm clears** → LT.04 records VOID ("fewer than two learners"),
  which is honest. Proceed to Stage 3 with the single survivor, and say plainly
  in the ledger that no comparison was made.
- **An arm trips the chaos check (G9)** → that arm is VOID, not FAIL, and the
  finding is reported: *this mechanism degenerates into self-noise on this body*.
  For `metra` that is a quantitative replication of RGSD in Jack's world.
- **Attempts happen, no trend (F2)** → the bottleneck is horizon, not curiosity.
  Pivot to hindsight density and PEG-style goal planning; do *not* buy GPU.
- **No engaged attempts anywhere (F1)** → the bottleneck is approach and
  commitment (§2.7), not the hang. Add a Go-Explore archive over `h(t)`-bearing
  states — PG.3 certified the state restore it needs at `resume_max_dev = 0.0` —
  and re-run Stage 2. Still CPU.

### Stage 3 — retain, generalise, confirm (CPU, ~2 core-hours)

**LT.05** (retention), **LT.06** (moved ladder), **LT.07** (winner re-run at 3
fresh seeds, G8). Only after all three does anything go in the README.

### Stage 4 — the VLM arm (CPU + a few API calls)

**LT.09**, conditional on `lp` winning LT.04. Cheap: 4 images per 400 s of
simulated time.

### Stage 5 — the humanoid (GPU; **currently blocked on two things, neither of
them quota**)

**LT.08** is BLOCKED on `T2.01`/`T2.02` and on throughput (§6). Do not schedule it
until both are cleared. Scheduling it earlier is the single most expensive
mistake available in this plan.

### The swimming clause

The owner's sentence has a second half — *"if theres water he must try to swim
and struggle"*. The pool exists and PG.2 certifies buoyancy at `worst_depth_error_frac = 0.0`.
Everything in §2 transposes: `submerged_time` and `net_displacement_while_submerged`
replace `h(t)`, "fall" becomes "sink", and the same six observables apply. **Do
not run it until the Ladder Test resolves** — one falsifiable claim at a time,
and the ladder is the one the owner named first.

---

## 6. Cost, against free compute only

Measured on this box (aarch64, 4 shared cores, mujoco 3.2.3, single-threaded),
2026-08-09:

| Configuration | Throughput |
|---|---|
| playground alone | **6,236 mj_step/s** |
| playground + climber-rover, random control | **3,249 mj_step/s** |
| ...at 40 substeps/decision, with per-substep contact scanning in Python | **~50 decisions/s** (pilot, measured) |
| ...with contact scanning done once per decision (the correct implementation) | **~81 decisions/s** (physics-bound) |
| PG.4's rover + online ICM training, for reference | 242 decisions/s (496 s for 120 K decisions) |

### The programme budget

Costed by **seeds × arms**, not per experiment — `LESSONS.md`, "the budget names
one experiment; the spec runs seeds × arms", the lesson that cost this project
5.5 GPU-hours on one unguarded submission.

| Spec | Arithmetic | Core-hours | Wall (3 workers) |
|---|---|---|---|
| One arm-seed | 50,000 decisions ÷ ~61 dec/s (physics + a small torch update) | 0.23 | — |
| **LT.01** | 3 seeds × 3,000 decisions + 800 hang bursts + oracle audit | **0.35** | ~20 min |
| **LT.02** | (`icm`, `ragdoll-icm`) × 3 seeds × 15,000 dec + scripted + null + pooled model fits | **1.1** | ~25 min |
| **LT.03** | 5 arms (3 candidates + `null` + `randrew`) × 3 seeds × 0.23 h | **3.5** | ~1.2 h |
| **LT.04** | re-scores LT.03's stored trajectories | **~0.05** | minutes |
| **LT.05 + LT.06 + LT.07** | 1 arm × 3 seeds × 3 specs | **2.1** | ~45 min |
| **LT.09** | 1 arm × 3 seeds + ~450 VLM images | **0.7** | ~15 min |
| **Total, Stages 0–4** | | **≈ 7.8 core-hours** | **≈ 3 h** |
| **GPU** | | **0.0** | — |

**The entire Ladder Test programme through Stage 4 costs about eight
CPU-core-hours and no GPU quota at all.** That is the single most important
number in this document. It is affordable because the arms use ~150 K-parameter
dedicated networks rather than the 45.5 M `UnifiedBrain` — which is also
CURIOSITY.md's recommendation ("a small trained core") and the reason T2.02 cost
6.3 CPU-hours and still could not arbitrate.

**Kaggle state, 2026-08-09: ~23 h remaining this week, resets Sunday.** Nothing
in Stages 0–4 needs any of it. The right use of this week's remaining quota is
**not** the Ladder Test — it is whatever unblocks `T2.01`/`T2.02`, because those
are what gate LT.08 and, per `LESSONS.md` ("a dependency graph can quietly make
your most important claim unreachable"), eleven other specs. Spending quota on
LT.08 before Stage 2 has run would be buying the answer to a question we have
not yet earned the right to ask.

Box discipline: 3 workers, not 4; `nice 19`; under ~1.5 GB RAM; leave no process
running. The aggregator's 3 GiB ceiling and the tenant containers are not
negotiable (`SYSTEM.md`, `/home/opc/CLAUDE.md`).

### The humanoid budget — and why quota is not the blocker

| Path | Arithmetic | Verdict |
|---|---|---|
| Kaggle P100, current pipeline | T2.01 measured **~128 env-steps/s** (bottleneck: the PPO update on the big trunk, *not* MuJoCo — 1024 Humanoid steps cost MuJoCo ~0.5 s against ~13 s per iteration). 20 M steps/arm-seed = **43 h**. 3 seeds = 130 h. | **Impossible.** Kaggle gives 30 h/week. One arm would take a month. |
| Kaggle P100, small dedicated nets | Remove the trunk from the inner loop → MuJoCo-bound at ~2,000 steps/s → 20 M steps = **2.8 h/arm-seed** | **Feasible: ~8.3 h for 3 seeds of one arm.** Fits one week's quota for the winning arm only. |
| Batched sim (MJX/Brax) | 10³–10⁴ × throughput | The right answer, but JAX on Kaggle's **P100 (sm_60)** is exactly the compatibility class that already bit this repo (`gpu.py`: preinstalled torch ships sm_70+ kernels only). Needs its own certification spec before it is trusted. |

**Conclusion: LT.08's prerequisite is a throughput spec, not more quota.** The
actionable item is "get the trunk out of the inner loop" — worth ~16× and
already implied by T2.01's own post-mortem comment.

Current quota state (`experiments/gpu_budget.json`): 2026-W31 spent 37.5 Kaggle
hours against a 30 h allowance; W32 spent 6.4. The bakeoff needs none of it.

---

## 7. What this document does not settle

Stated so the next session does not mistake confidence for evidence.

- **Whether the climber-rover is a fair stand-in for the humanoid.** It is a
  declared convenience with a declared purpose (isolate climbing from
  locomotion). If LT.03 passes and LT.08 fails, the honest report is "curiosity
  climbs on a 8-DoF body", not "curiosity climbs".
- **Whether `z | no-ground-contact` in the outcome space is too much of a hint.**
  §3.2 argues it is not (it does not mention the ladder, and jumping satisfies
  it), but it is the weakest joint in the design and LT.06 is what tests it.
- **Whether one life of 50,000 decisions is long enough.** The owner said
  "attempt 40". If arms produce 5 attempts, the answer is more decisions, not a
  lower threshold — and at 14 min per arm-seed, 4× is affordable.
- **Whether disagreement's noisy-TV immunity survives a body that generates its
  own chaos.** ICM's known weakness is *action-conditioned* noise; the rover
  tumbling is action-conditioned noise. A5 and A3 both need the panel check to
  be reported next to a "self-generated chaos" check, and this document does not
  specify the latter. It should.
