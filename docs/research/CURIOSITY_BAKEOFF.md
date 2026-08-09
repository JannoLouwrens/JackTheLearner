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

Three facts from this repo's own ledger frame everything:

| Fact | Source | Consequence |
|---|---|---|
| The ladder is physically climbable and falls are clean and resumable | PG.3 PASS, `ascent_frac 0.973 ± 0.020`, `resume_max_dev 0.0` | The fixture is not the blocker. Go-Explore-style *state restore* is certified to bit-exactness — unusual and valuable (§1.4). |
| The noise panel really does hypnotise naive curiosity | PG.4 PASS, `icm_dwell_share 0.667 ± 0.471`, null `0.061 ± 0.027`, static-texture control `0.000` | The trap works — **but read the std**: across 3 seeds the ICM arm scored roughly {1, 1, 0}. Dwell must be reported **per seed**, never as a mean alone. |
| **Jack cannot walk.** | T2.01 FAIL, T2.02 FAIL (22,604 s CPU) | "Curiosity climbs the ladder" on the full humanoid is currently blocked on locomotion. Any plan that starts there is a plan to burn quota. §5 stages around this. |

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

The full humanoid is the eventual subject (LT.5). It is **not** the subject of
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
construction**. LT.1 verifies that inheritance rather than assuming it.

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

**Retention is a separate spec (LT.3), and it has a trap in it.** Two measures:

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
| **G1** | *A hand-coded climb reward — instruction dressed as curiosity.* | **(a) Static audit, executed inside the test and recorded as a metric.** The arm's intrinsic-reward module is parsed; a match on any of `ladder, rung, rail, apple, platform, climb, height, up, torso_z, qpos\[2\], xipos\[.\]\[2\]` in the reward path sets `reward_audit_clean = 0` → **ERROR, not FAIL** (the spec is void, not falsified). **(b) Runtime assertion** `env_reward_absmax == 0.0` — the environment returns literally zero to the policy, always. **(c) LT.4**: the same unmodified code in a world with the ladder moved and reshaped. |
| **G2** | *Random flailing that occasionally ascends.* | C-NULL (measured: 0 engaged attempts in 9,000 decisions). Plus O4's Spearman clause — flailing has no trend — and O3's distance-matched return lift, which flailing cannot produce because it has no memory of falling. |
| **G3** | *The noisy-TV trap.* | O6 disqualifier at 0.15, **per seed**; the panel is mandatory in every arm's world; C-ICM must fixate to prove the trap is live in this rig. |
| **G4** | *Reward-hacking the height sensor — jumping, stairs, the seesaw, standing on a box, a tumble that grazes a rail.* | The three-clause `h(t)` conjunction (§2.4), which was **built by attacking it**: absolute-z scored 0.55 under random action, instantaneous-rise 0.063, persistent+load-bearing 0.021. The pilot's non-ladder height ceiling is **1.007 m**, so a raw-height metric would have been over half gameable. LT.0 re-measures both ceilings per world mutation; if any mutation lets a non-ladder route exceed the ladder-supported record, that mutation is rejected. |
| **G5** | *Seed luck.* | 3 seeds; O5 required in ≥ 2 of 3; per-seed reporting mandatory for every observable; mean ± std for continuous ones. PG.4's `0.667 ± 0.471` is the standing cautionary precedent. |
| **G6** | *Threshold fiddling after the fact.* | All thresholds in §2.5 were fixed from the §2.6 pilot **before any arm ran**, and are written here with the pilot numbers alongside them. The `_check` function is written before the run, per `protocol.py`. |
| **G7** | *Experimenter curriculum leakage.* | The world is drawn per seed by `PlaygroundParams.mutate()`. No arm sees a hand-picked world. Episodes never reset him to the ladder base. The apple carries no reward. |
| **G8** | *Selection over arms — running six arms and reporting the winner.* | The bakeoff reports **all six arms' full metric tables** in the ledger, pass or fail. A winning arm must then be **re-run at 3 fresh seeds** (LT.2b) at the pre-registered thresholds before any claim is made. Six arms × 3 seeds at p<0.01 with a conjunction of 6 clauses is not a multiple-comparison worry, but the confirmation run costs 40 CPU-minutes and removes the argument. |

### 2.10 What would falsify the whole approach

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

| Arm | Objective | Implementation in this repo | Params | Prior | Expected failure |
|---|---|---|---|---|---|
| **A0 `null`** | none | random and random-repeat action | 0 | — | passes nothing (measured, §2.6) |
| **A1 `icm`** | forward-model error on inverse-dynamics features | `UnifiedBrain.IntrinsicCuriosityModule.compute_icm_reward`, instantiated standalone on a 64-d proprio+ray observation | ~40 K | 1705.05363 | **DISQUALIFIED by O6.** Serves as the mandatory trap victim (C-ICM). |
| **A2 `rnd`** | ‖predictor − frozen random target‖ | `IntrinsicCuriosityModule` RND branch, same module | ~40 K | 1810.12894 | trapped by O6; the panel's observation is never repeated so the predictor never converges |
| **A3 `disagree`** | variance across a 5-member one-step forward-model ensemble | new, ~120 lines; five 2-layer MLPs, different inits and different bootstrap masks | 5 × 12 K | 2005.05960, 1906.04161 | **the principled noisy-TV fix**; risk is that the ensemble shares too much data and disagreement collapses everywhere, not just on noise |
| **A4 `lp`** | absolute learning progress over an auto-partitioned outcome space, with hindsight relabeling | `UnifiedBrain.AutotelicGoalGenerator` (`strategy="learning_progress"`) + ALP-GMM-style region split; goal space = **(torso xyz, torso z with-no-ground-contact, hand-contact count, nearest-object displacement)** | ~90 K | 1910.07224, 1708.02190, 2502.07709 | **the favourite** — but LP is a *selector*, not a discoverer (§1.2 failure mode 5). It cannot bootstrap a goal with zero successes. Must be paired with goal babbling. |
| **A5 `metra`** | METRA temporal-distance skill latents + LP selection over skills | new; 2-d latent, 16 skills, small Gaussian policy on the same observation | ~200 K | 2310.08887 | RGSD (2510.06203) predicts locomotion diversity only. If A5 produces only travel directions and never vertical structure, that is a *result*, and it is the cheapest possible replication of RGSD's finding in Jack's world. |
| **A6 `vlm-lp`** *(conditional, Stage 3)* | frozen VLM proposes goals **as predicates in A4's outcome space**; LP has the final vote | new; renders 4 snapshots every K minutes → captions → ~10 candidate (goal-text, outcome-space predicate) pairs → A4's buffer | LLM frozen | 2306.01711, 2302.06692, 2310.00166 | only run if A4 wins. **The VLM may never propose a reward or code** (that is Eureka = instruction, §1.5). Control: scrambled-caption VLM fed another scene must not beat LP-only. |

Note that A1, A2 and A4 **instantiate modules that already exist in
`UnifiedBrain.py` and have never received a gradient in a test** (`protocol.py`
docstring: "45.5M parameters with no live call site"). The bakeoff is therefore
also the evidence that decides T3.06 — those modules either earn their
parameters here or get deleted.

### 3.2 The critical design decision the pilots force

§2.7's measurement determines whether A4 can work alone. Because LP is zero for
a never-achieved goal, the outcome space **must include a dimension that is
non-zero for partial ladder engagement** — specifically
`z_torso | no ground contact`. With that dimension, hindsight relabeling turns
every failed grab into a success at a small value of it, LP becomes measurable
at *0.05 m of hang*, and the curriculum can climb continuously from brushing the
ladder to hanging to ascending. Without it, the first success has probability
zero and LP has nothing to select over.

This is a genuine design choice with a gaming risk attached, and it must be
declared: the outcome-space dimension `z | no-ground-contact` is *close* to the
metric `h(t)`. The defence is that (a) it does not reference the ladder — it is
"how high am I with nothing under me", which is equally satisfied by jumping off
the stairs, and G1's symbol audit permits it on exactly those grounds; (b) LT.4
moves the ladder; (c) A3 and A5 do not use it at all, so if only A4 passes we
know the dimension is doing the work and must say so.

### 3.3 Registry entries — exact `Spec(...)` format

To be appended to `EXPANSION` in `experiments/registry_expansion.py`. Tier
mapping follows the existing convention (fixtures → 2, claims → 5).

```python
    # ── THE LADDER TEST (docs/research/CURIOSITY_BAKEOFF.md) ────────────
    Spec("LT.0", 2, "The Ladder Test is measurable: null floor and un-gameable rise",
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
         metric="null_engaged_attempts", budget=Budget.CPU, depends_on=["PG.1", "PG.3", "PG.4"],
         seeds=3,
         control="A greedy height-maximising oracle with adhesion DISABLED must "
                 "still be unable to reach the platform — else an alternate "
                 "route exists and SUCCESS is not evidence of climbing.",
         kills="The entire Ladder Test, before a single arm is trained. Costs "
               "20 CPU-minutes; every threshold in the bakeoff is set from it.",
         notes="Pilot 2026-08-09 (aarch64, mujoco 3.2.3). Free-roaming: 0 "
               "engaged attempts in 9,000 random decisions; max NON-ladder "
               "torso z 1.007 m against z_rest 0.360 m. From the base, 800 x "
               "3 s bursts: P(hang) = 0.55 under an ABSOLUTE-z definition "
               "(broken - z_rest already clears the bar), 0.063 instantaneous, "
               "0.026 persistent, 0.021 +- 0.009 persistent AND load-bearing; "
               "random rise ceiling 0.83 m. Those four numbers ARE the "
               "definition of h(t) and every threshold in LT.2."),

    Spec("LT.1", 2, "The climber-rover can approach and climb when scripted",
         hypothesis="The PG.3 arm/adhesion parameters, carried onto a mobile "
                    "body with a floor-gated planar drive, still ascend: a "
                    "scripted sequence walks to the ladder, gains >= 3 rung "
                    "spacings of ladder-supported height, and reaches the "
                    "platform; falls stay finite and resumable.",
         falsified_by="The drive interferes with the hang, or the foot geom "
                      "blocks the arms, or ascent < 3 rung spacings — then the "
                      "reduced body is not a valid stand-in and LT.2 is void.",
         null_baseline="Zero adhesion — must slip (PG.3's null, inherited).",
         metric="scripted_platform_reached", budget=Budget.CPU,
         depends_on=["PG.3", "LT.0"], seeds=3,
         control="With the drive force set to zero the rover must never reach "
                 "the ladder — else the body is starting close enough that "
                 "approach is not part of the task.",
         notes="Certifies by measurement what LT.2 otherwise assumes by "
               "construction. Seeds map to rung spacings 0.26/0.30/0.34 m."),

    Spec("LT.2", 5, "THE LADDER TEST: curiosity alone climbs the ladder",
         hypothesis="At least one intrinsic-motivation arm, with the "
                    "environment returning reward identically zero, produces "
                    ">=20 engaged ladder attempts, a distance-matched "
                    "post-fall return lift >= 2.0, an ascent gain >= 0.35 m "
                    "with Spearman rho >= 0.35 (p<0.01) and a final-quintile "
                    "mean rise >= 0.85 m (above the measured random ceiling of "
                    "0.83 m), and at least one topping-out, in >=2 of 3 seeds — "
                    "while dwelling <= 0.15 at the noise panel in every seed.",
         falsified_by="No arm produces a single engaged attempt (exploration "
                      "never reaches the ladder), or attempts occur with no "
                      "ascent trend (credit assignment, not curiosity, is the "
                      "bottleneck), or the only passing arm fixates on the panel.",
         null_baseline="Random and random-repeat action: measured at 0 engaged "
                       "attempts in 9,000 decisions (LT.0). Plus a "
                       "random-stationary-reward learner at matched compute, "
                       "which controls for 'any optimisation pressure explores'.",
         metric="unforced_ascent_gain", budget=Budget.CPU_LONG,
         depends_on=["LT.0", "LT.1", "PG.4"], seeds=3,
         control="The ICM arm MUST fixate on the noise panel in THIS rig "
                 "(dwell > 0.4) — proving the trap is live here and not only "
                 "in PG.4's rover. And a goal-shuffled variant of the winning "
                 "arm must show no ascent trend.",
         kills="The 'intrinsic motivation is enough' thesis for structured "
               "vertical behaviour. If it fails, GOAL.md's ladder image needs "
               "a goal/skill layer (PEG 2303.13002, Go-Explore archive), and "
               "that pivot is decided by this result, not by preference.",
         notes="Six arms (null/icm/rnd/disagree/lp/metra) x 3 seeds, all "
               "reported pass or fail. Every arm's reward code passes a static "
               "audit for ladder-referencing symbols; a match is ERROR, not "
               "FAIL. No published system has done this — LadderMan "
               "(2606.05873) climbs from a human reference motion, and METRA "
               "on a 69-DoF humanoid flails (RGSD 2510.06203)."),

    Spec("LT.3", 5, "The climb survives the curiosity that produced it",
         hypothesis="With the intrinsic module removed and reward identically "
                    "zero, the winning arm's deterministic policy still reaches "
                    ">= 0.8x its best training ladder-supported height and tops "
                    "out at least once in 10 episodes.",
         falsified_by="Ladder-supported height collapses without the bonus — "
                      "then the behaviour was bonus-chasing, not a skill.",
         null_baseline="The same policy at initialisation, bonus off.",
         metric="retention_ratio", budget=Budget.CPU, depends_on=["LT.2"], seeds=3,
         control="A policy trained with the random-stationary reward must show "
                 "no retained climbing — else retention measures architecture, "
                 "not learning.",
         notes="Spontaneous attempt FREQUENCY with the bonus off is reported "
               "but explicitly NOT gated: a learning-progress agent is supposed "
               "to lose interest once the ladder is mastered. Gating on "
               "frequency would penalise the mechanism most likely to be right."),

    Spec("LT.4", 5, "It is the ladder he is curious about, not the coordinates",
         hypothesis="The identical unmodified arm, in a world where the ladder "
                    "is moved, re-yawed and re-spaced, scores >= 0.5x its "
                    "home-world ascent gain.",
         falsified_by="Performance collapses when the ladder moves — the arm "
                      "learned a location, or the reward was hard-coded.",
         null_baseline="Home-world score for the same arm and seed.",
         metric="moved_ladder_ratio", budget=Budget.CPU_LONG,
         depends_on=["LT.2"], seeds=3,
         control="A deliberately hard-coded climb reward (written for this "
                 "control only, keyed to the home ladder's xy) MUST fail here — "
                 "this is what makes the spec an instruction detector.",
         notes="Together with LT.2's static symbol audit, this is the anti-"
               "instruction provision. Eureka-style LLM reward writing "
               "(2310.12931) would be caught by exactly this pair."),

    Spec("LT.5", 5, "The humanoid climbs — same test, real body",
         hypothesis="With locomotion in hand, the winning arm reproduces LT.2's "
                    "six observables on the full humanoid in the same playground.",
         falsified_by="Any of LT.2's six clauses fails on the humanoid at the "
                      "budgeted step count with the curve flat.",
         null_baseline="LT.2's nulls, re-measured on the humanoid body.",
         metric="unforced_ascent_gain", budget=Budget.GPU_LONG,
         depends_on=["LT.2", "LT.3", "T2.01", "T2.02"], seeds=3,
         control="Same as LT.2, re-run on this body: the ICM arm must fixate.",
         kills="Nothing on its own — a FAIL here with LT.2 passing scopes the "
               "claim honestly to the reduced body and points at throughput.",
         notes="BLOCKED until T2.01/T2.02 pass. Also blocked on throughput: at "
               "T2.01's measured ~128 env-steps/s a 20M-step arm-seed costs "
               "43 h, so 3 seeds exceed a whole week of Kaggle quota for ONE "
               "arm. The prerequisite is a batched simulator, not more quota."),

    Spec("LT.6", 5, "The VLM proposes ladder-shaped goals; learning progress disposes",
         hypothesis="Frozen-VLM-proposed goals, expressed ONLY as predicates in "
                    "the existing outcome space and filtered by LP, reach the "
                    "first engaged ladder attempt in fewer decisions than "
                    "LP-only at matched goal count.",
         falsified_by="No speedup, or VLM goals flood the buffer while their "
                      "achievement stays ~0 (a hallucinated curriculum).",
         null_baseline="LP-only (arm A4) at matched goal count.",
         metric="time_to_first_engaged_attempt", budget=Budget.CPU_LONG,
         depends_on=["LT.2"], seeds=3,
         control="A scrambled-caption VLM fed a DIFFERENT scene must not beat "
                 "LP-only — else the benefit was 'more goals', not grounded "
                 "interestingness. Additionally: the VLM may never emit reward "
                 "code; a proposal that is not a predicate over existing "
                 "outcome dimensions is rejected before it reaches the buffer.",
         notes="Only run if A4 wins LT.2. LLM-proposed goals have never driven "
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
- **A3 `disagree`:** five one-step forward models `f_i(s, a) → s'`, each with its
  own init and a 0.8 bootstrap mask over the replay stream; bonus =
  `var_i f_i(s,a)` averaged over dimensions, normalised by a running std. The
  panel check is the point: disagreement must decay to ≈ 0 there while ICM's
  error stays high.
- **A4 `lp`:** outcome vector `o = (x, y, z, z|no-ground-contact,
  n_climb_contacts, ‖Δp_nearest_object‖)`. Region tree split on outcome variance
  (SAGG-RIAC); per-region ALP = |EMA_recent(success) − EMA_old(success)| over a
  window of 20 attempts, with a minimum of 8 attempts before a region's ALP is
  trusted (failure mode 1). Goal sampling: ε-greedy over ALP with ε = 0.2
  uniform for babbling. Hindsight: every trajectory is relabeled with its own
  achieved `o` and regressed — this is the flow-matching weld (T2.16), and it is
  what makes goal-reaching trainable without RL machinery.
  **Report LP at 3 window lengths (10/20/40); a conclusion that flips is not
  a conclusion** (failure mode 3).
- **A5 `metra`:** 2-d latent, 16 discretised skills, temporal-distance
  constraint per 2310.08887. Ablation of the temporal-distance constraint is
  CU.4's required control and should be run in the same job.
- **A6 `vlm-lp`:** render 4 × 128 px snapshots via osmesa every 2,000 decisions,
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

Non-negotiable, applied identically to all six arms and recorded in the ledger
for each:

1. **The noise panel is present in every arm's world.** `noise_panel=True` is
   asserted at build time; a run with it disabled is void.
2. **Panel dwell is reported per seed**, never as a mean alone. PG.4's
   `0.667 ± 0.471` is why.
3. **Dwell > 0.15 in any seed disqualifies the arm**, regardless of its ladder
   score. A disqualified arm is recorded with its full metrics and the
   disqualification reason — it is data, not a deletion.
4. **The ICM arm must fixate in this rig** (dwell > 0.4). If it does not, the
   noise realisation is broken and no arm's immunity may be claimed.
5. **`env_reward_absmax == 0.0`** is asserted every step. The environment never
   returns a reward to any arm.
6. **`reward_audit_clean == 1`** — the static symbol audit of §2.9/G1 passes, or
   the arm's result is ERROR (void), not FAIL (falsified).
7. **Coverage is reported alongside dwell** (`visited_cell_frac`, PG.4's
   metric). An arm that avoids the panel by exploring nothing is not immune, it
   is inert — CU.3's `coverage_vs_dwell` is the two-number metric for exactly
   this reason.

---

## 5. The staged plan: cheapest falsifier first

The ordering rule is the ladder's own: *a hypothesis must die before it costs
GPU quota* (`protocol.py::Budget`).

### Stage 0 — the 20-minute CPU experiment that could kill the whole thing

**LT.0. Already piloted; §2.6 is its output.** Formalise the pilot into
`experiments/tests/lt_0_null_floor.py` and run it at 3 seeds. It answers, before
anything is trained:

- Is the null floor actually zero? *(Measured: yes — 0 engaged attempts in
  9,000 random decisions.)*
- Is raw height gameable? *(Measured: yes — 1.007 m without the ladder. So the
  `h(t)` conjunction is mandatory, not stylistic.)*
- Can a greedy height-maximising oracle reach the platform without rungs?
  *(The alternate-route audit; if yes, SUCCESS is not evidence of climbing and
  the world needs redesign.)*
- What is P(weight-bearing hang | random action from the ladder base)? *(§2.7 —
  the number that decides whether LP can work alone or needs an archive.)*

**Cost: ~20 CPU-minutes. Kills or reshapes the entire programme.** This is the
experiment to run first, and it has already returned its most important number.

### Stage 1 — certify the body (CPU, ~30 min)

**LT.1.** Script the approach-and-climb on the climber-rover; verify the drive
does not interfere with the hang and the foot does not block the arms. If the
reduced body cannot be *scripted* up the ladder, no arm will learn it, and the
body needs redesign before any learning runs. This is PG.3's logic applied one
level up.

### Stage 2 — the bakeoff (CPU, ~4 core-hours; **zero GPU**)

**LT.2.** Six arms × 3 seeds, 50,000 decisions each. Run 3 workers in parallel
on the box's 4 shared cores (never 4 — the box hosts production tenants; see
`/home/opc/CLAUDE.md`). **~2 h wall clock.**

Decision gate after Stage 2:
- **≥ 1 arm passes all six observables** → Stage 3.
- **Attempts happen, no trend (F2)** → the bottleneck is horizon, not curiosity.
  Pivot to hindsight density and PEG-style goal planning; do *not* buy GPU.
- **No engaged attempts anywhere (F1)** → the bottleneck is discovery. Add a
  Go-Explore archive over `h(t)`-bearing states (state restore already certified
  by PG.3) and re-run Stage 2. Still CPU.

### Stage 3 — confirm, retain, generalise (CPU, ~2 core-hours)

**LT.2b** (winner re-run at 3 fresh seeds, G8), **LT.3** (retention),
**LT.4** (moved ladder). Only after these three does anything go in the README.

### Stage 4 — the VLM arm (CPU + a few API calls)

**LT.6**, conditional on A4 winning. Cheap: 4 images per 400 s of simulated time.

### Stage 5 — the humanoid (GPU; **currently blocked on two things, neither of
them quota**)

**LT.5** is BLOCKED on `T2.01`/`T2.02` and on throughput (§6). Do not schedule it
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

### The bakeoff budget

| Item | Arithmetic | Cost |
|---|---|---|
| One arm-seed | 50,000 decisions ÷ ~61 dec/s (physics + a small torch update) | **~14 min core-time** |
| LT.2 full bakeoff | 6 arms × 3 seeds × 14 min | **4.1 core-hours** |
| LT.2 wall clock | 3 parallel workers on 4 shared cores | **~1.5 h** |
| LT.0 + LT.1 | pilots, measured | **~50 min** |
| LT.2b + LT.3 + LT.4 | 1 arm × 3 seeds × 3 specs | **~2 core-hours** |
| LT.6 | 1 arm × 3 seeds + ~450 VLM images | **~45 min + pennies** |
| **Total for Stages 0–4** | | **≈ 7 core-hours, 0 GPU-hours** |

**The entire Ladder Test programme through Stage 4 costs seven CPU-core-hours
and no GPU quota at all.** That is the single most important number in this
document. It is affordable because the arms use ~150 K-parameter dedicated
networks rather than the 45.5 M `UnifiedBrain` — which is also the CURIOSITY.md
recommendation ("a small trained core") and the reason T2.02 cost 6.3 CPU-hours
and still failed.

Box discipline: 3 workers, not 4; `nice -n 10`; the aggregator's 3 GiB ceiling
and the tenant containers are not negotiable (`/home/opc/CLAUDE.md`).

### The humanoid budget — and why quota is not the blocker

| Path | Arithmetic | Verdict |
|---|---|---|
| Kaggle P100, current pipeline | T2.01 measured **~128 env-steps/s** (bottleneck: the PPO update on the big trunk, *not* MuJoCo — 1024 Humanoid steps cost MuJoCo ~0.5 s against ~13 s per iteration). 20 M steps/arm-seed = **43 h**. 3 seeds = 130 h. | **Impossible.** Kaggle gives 30 h/week. One arm would take a month. |
| Kaggle P100, small dedicated nets | Remove the trunk from the inner loop → MuJoCo-bound at ~2,000 steps/s → 20 M steps = **2.8 h/arm-seed** | **Feasible: ~8.3 h for 3 seeds of one arm.** Fits one week's quota for the winning arm only. |
| Batched sim (MJX/Brax) | 10³–10⁴ × throughput | The right answer, but JAX on Kaggle's **P100 (sm_60)** is exactly the compatibility class that already bit this repo (`gpu.py`: preinstalled torch ships sm_70+ kernels only). Needs its own certification spec before it is trusted. |

**Conclusion: LT.5's prerequisite is a throughput spec, not more quota.** The
actionable item is "get the trunk out of the inner loop" — worth ~16× and
already implied by T2.01's own post-mortem comment.

Current quota state (`experiments/gpu_budget.json`): 2026-W31 spent 37.5 Kaggle
hours against a 30 h allowance; W32 spent 6.4. The bakeoff needs none of it.

---

## 7. What this document does not settle

Stated so the next session does not mistake confidence for evidence.

- **Whether the climber-rover is a fair stand-in for the humanoid.** It is a
  declared convenience with a declared purpose (isolate climbing from
  locomotion). If LT.2 passes and LT.5 fails, the honest report is "curiosity
  climbs on a 8-DoF body", not "curiosity climbs".
- **Whether `z | no-ground-contact` in the outcome space is too much of a hint.**
  §3.2 argues it is not (it does not mention the ladder, and jumping satisfies
  it), but it is the weakest joint in the design and LT.4 is what tests it.
- **Whether one life of 50,000 decisions is long enough.** The owner said
  "attempt 40". If arms produce 5 attempts, the answer is more decisions, not a
  lower threshold — and at 14 min per arm-seed, 4× is affordable.
- **Whether disagreement's noisy-TV immunity survives a body that generates its
  own chaos.** ICM's known weakness is *action-conditioned* noise; the rover
  tumbling is action-conditioned noise. A5 and A3 both need the panel check to
  be reported next to a "self-generated chaos" check, and this document does not
  specify the latter. It should.
