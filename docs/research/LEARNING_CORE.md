# LEARNING_CORE.md — the first principles of how Jack learns

> Researched and specified 2026-08-09. Answers `SYSTEM.md`'s Q2 — *what
> mathematics turns Jack's experience into a better Jack?* Companions:
> `PURPOSE_AND_SCAFFOLDING.md` (what he wants), `CURIOSITY_BAKEOFF.md` (what he
> explores), `UNIFIED_BRAIN_BAKEOFF.md` and `D1_CONTROL_ARCHITECTURE.md` (what
> shape carries the senses), `SURVIVAL_WORLD.md` (the world he lives in).
> **Nothing here has been run.** §8 lists what this document refuses to claim.

The owner, 2026-08-09:

> *"We need to know the first principles. Like, THIS is how it learns. Like,
> this kind of mathematics or this kind of AI model. We need a lot of research
> and to make it as simple as possible... I am sure at the end of the day it
> won't be the most complex model that Jack is. It will be just a system that
> can learn and get input from every single sense."*

This document answers that question the only way `SYSTEM.md` permits: with a
survey that separates what was **demonstrated** from what was **argued**, a
recommendation with its strongest counterargument stated fairly, and a
pre-registered bakeoff that decides it by arithmetic rather than by this
document's opinion.

## Contents

0. [The one-paragraph answer](#0-the-one-paragraph-answer)
1. [Provenance of every number here](#1-provenance-of-every-number-here)
2. [The question, made decidable](#2-the-question-made-decidable)
3. [The survey](#3-the-survey)
4. [The recommendation, and the strongest case against it](#4-the-recommendation)
5. [The first-principles bakeoff (LC.00–LC.06)](#5-the-first-principles-bakeoff)
6. [The simplicity budget](#6-the-simplicity-budget)
7. [What each candidate keeps, adapts, or deletes](#7-what-each-candidate-keeps-adapts-or-deletes)
8. [What this document refuses to claim](#8-what-this-document-refuses-to-claim)
9. [What this document changed about the machine](#9-what-this-document-changed-about-the-machine)

---

## 0. The one-paragraph answer

**Jack learns by predicting.** One small model takes every sense — vision,
audio, touch, proprioception, need-state, language — into one latent state, and
is trained on a single objective: *given what I sense and what I do, predict
what I will sense next, in every modality at once.* His **needs** are a
preference over his own predicted interoceptive future (fed, warm, unhurt, not
alone); his **curiosity** is the places his own model most disagrees with
itself; his **memory** is a separate, literal diary. Acting is choosing, inside
the predicted world, what best serves the first two. Four moving parts — a
predictor, a preference, a disagreement, a diary — and it is the only
formulation surveyed in which *"one brain, all senses in unison"* is the
**learning rule itself** rather than a wiring diagram bolted onto one.

**Three things must be said in the same breath, or this paragraph is
propaganda.** (i) It is a **recommendation, not a decision** — §5's bakeoff
decides, on ~33 CPU-core-hours and **zero GPU quota**. (ii) The argument for it
is **unison, not sample efficiency**: on Crafter — the nearest published
analogue of Jack's world — a properly-tuned **4M-parameter PPO scores 15.60 %
against DreamerV3-201M's 14.77 %** (arXiv:2307.03486), and the widely-quoted
"PPO 4.6 %" is an under-engineered PPO. (iii) The incumbent is **not** simply
PPO: under `SYSTEM.md`'s constitutional unison constraint, an admissible PPO
arm must carry a cross-modal masked-prediction loss — *which is a world-model
objective with the action-conditioning removed*. So the real question is
narrower and much more answerable than "PPO versus Dreamer": **everyone has to
build half a world model; does the other half pay for itself?**

---

## 1. Provenance of every number here

Tags, used inline, following `D1_CONTROL_ARCHITECTURE.md`'s convention:

| tag | meaning |
|---|---|
| **[V]** | verified against the primary source (paper read, number quoted) |
| **[c]** | cited from a secondary/aggregating source; not independently verified |
| **[M]** | **measured on this box today**, command recorded |
| **[C]** | computed by arithmetic from [M]/[V] quantities, derivation shown |
| **[L]** | read from `experiments/ledger.json` |
| **[—]** | asserted with no evidence; flagged as such on purpose |

And the distinction the owner's question actually turns on, applied to every
row of the survey:

> **DEMONSTRATED** = a number on a named benchmark, in a paper, that someone
> could have failed to reproduce.
> **ARGUED** = a mathematical property, an elegance claim, or a position paper.
>
> Both are useful. Confusing them is how a project spends six GPU-weeks on a
> beautiful equation. This repo has already paid for the inverse mistake — 57M
> parameters that were never *argued* for either.

### Measurements taken on this box for this document

```
$ /data/venvs/jackthelearner/bin/python  (torch 2.8.0+cpu, 4 ARM cores)
```

| quantity | value | how |
|---|---|---|
| `UnifiedBrain.WorldModel` parameters | **2,974,977** [M] | AST-extracted the class, instantiated at the shipped config (`d_model=512, latent_dim=256, action_dim=17, obs_dim=256`), summed `p.numel()`. Breakdown: encoder 659,200 · dynamics 800,000 · residual_proj 65,792 · decoder 527,616 · reward_predictor 263,169 · target_encoder 659,200 |
| `WorldModel` live call sites | **zero** [M] | `UnifiedBrainConfig.enable_world_model = False` (`UnifiedBrain.py:231`), and `self.world_model = WorldModel(config) if config.enable_world_model else None` (`:3965`). It is never constructed under the shipped config. Every other reference is inside a branch that dereferences `None`. |
| torch on this box | **CPU-only, 2.8.0** [M] | `/data/venvs/jackthelearner/bin/python`. No GPU here; every GPU figure below is a *projection* from a measured throughput, and says so. |

From the ledger [L]:

| quantity | value | spec |
|---|---|---|
| trunk inference params | 41,525,008 (`layers` 36,710,400 · `action_expert` 4,615,696 · `proprio_encoder` 198,912) | T1.11 |
| flow-matching head works | sampler error 1.065 → **0.00134**; conditioning ratio **1578**; shuffled-conditioning error 1.958 | T1.12 |
| SB3 MLP on Humanoid-v5 | **530.2 ± 59.0** return, 124,707 params, 7.11σ over random | T2.02 |
| 57M trunk, same budget | **317.7 ± 84.2**, 2.46σ — *below its own learning gate*, and **confounded** (dropout live) | T2.02, D1 correction |
| untrained MLP | 175.1, **2.74σ** over random | T2.02 controls |
| random policy | 110.9 ± 23.5 | T2.02 |
| P100 throughput, trunk arm | ~106 env-steps/s incl. update; ~0.8 s/optimiser step at minibatch 512 | T2.02 [L][C] |

---

## 2. The question, made decidable

"How does Jack learn" is three questions wearing one coat, and this project has
already assigned two of them to other documents. Keeping them apart is the
single most useful thing this section does, because a bakeoff that varies more
than one of them at a time decides nothing.

| # | question | owned by | status |
|---|---|---|---|
| **Q1** | **What is the objective?** What quantity does Jack try to make big or small? | `PURPOSE_AND_SCAFFOLDING.md` (drives: `d(h)`, DR vs PBRS) + `CURIOSITY_BAKEOFF.md` (intrinsic signal: LP vs disagreement vs METRA) | specified, not yet run (PS.00–PS.06, LT.01–LT.09) |
| **Q2** | **What is the LEARNING CORE?** Given an objective, what mathematics converts experience into a better policy? | **this document** | open — the incumbent (PPO) has never produced a clean win here |
| **Q3** | **What is the architecture?** What network shape carries the senses? | `UNIFIED_BRAIN_BAKEOFF.md` (trunk, `z`, binding) + `D1_CONTROL_ARCHITECTURE.md` (where the trunk sits in the motor path) | T2.21 pre-registered, ~26 GPU-h, blocked on a push |

Q2 is the owner's "THIS is how it learns". It is *not* "what does he want" (Q1)
and *not* "how many layers" (Q3). The bakeoff in §5 therefore **holds Q1 and Q3
fixed** — same needs, same reward form, same trunk-free small network — and
varies only the learning rule. Any other design would reproduce T2.02's
mistake at a higher level: a comparison in which two things moved.

### The five questions every candidate must answer

Taken verbatim from the brief, because they are the right five:

- **(a) MULTIMODAL NATIVELY.** Does it take all senses into one latent, or does
  it need a bolt-on encoder per sense? `GOAL.md` demands the former.
- **(b) SPARSE, NEEDS-DRIVEN REWARD.** Does it work when the only signal is
  "you got hungrier" and "you died", with no hand-shaping?
- **(c) SAMPLE EFFICIENCY AT OUR COMPUTE.** 30 h/week of P100 and an elastic
  T4. An algorithm that needs 10⁸ env-steps is not available to us at any
  quality.
- **(d) SIMPLICITY.** Parameters, **hyperparameters**, moving parts, and —
  the one this repo has bled for — *what breaks silently*. The dropout bug cost
  ~13 GPU-hours and an owner-facing recommendation, and there was **no line of
  code to read**. Silent defaults are the most expensive bug class we have.
- **(e) SUBSUME OR CONFLICT.** What of T1.11/T1.12's flow-matching action head,
  `EpisodicMemory`, the frozen-LLM decision, and UB.16's trunk contract does it
  keep?

### One structural observation, stated up front because it reframes the survey

`UNIFIED_BRAIN_BAKEOFF.md` §1.2 already concluded that the *binding force* for
"one brain, all senses" is **cross-modal masked prediction** — predict the
masked-out modality from the others. `GOAL.md` demands "what he hears can teach
what he sees".

**That objective is a world model.** A model that predicts all senses jointly,
conditioned on action, *is* the thing the world-model literature calls a world
model; a model that predicts all senses jointly without conditioning on action
is its perception half. So the unified-brain direction this project already
committed to and the world-model direction of candidate 2 are not two choices —
they are the same objective, viewed from the perception side and the control
side. This is the most important fact in the document and §4 turns on it.

---

## 3. The survey

### 3.0 A cost measurement taken for this document, because it changes the answer

Everything below is judged against what this project can actually run. So
before the literature: **how much does a Dreamer-class update cost on our
hardware?** Measured on this box, 2026-08-09, `torch 2.8.0+cpu`,
`torch.set_num_threads(3)` (box discipline: 3 workers, not 4), `nice 19` [M]:

| configuration | cost |
|---|---|
| PPO-scale separate actor-critic, **120,841 params** (π 96→128→128→8, V 96→256→256→1) | update at minibatch 512: **68.4 ms** core-time · single-obs act: **168 µs** |
| **RSSM at a DreamerV3-XS shape** — GRU deter 256, 32×8 categorical stochastic, width-256 MLPs, encoder/decoder/reward(255-bin twohot)/continue heads: **1,432,160 params**; actor+critic **463,887**; **1,896,047 total** | one train step at batch 16 × length 32 (512 transitions) with a 15-step imagination rollout from every posterior state (7,680 imagined states): **3.754 s core-time**, 1.963 s wall |

Converted to the project's declared cost unit (`CURIOSITY_BAKEOFF.md` §3.1,
`PURPOSE_AND_SCAFFOLDING.md` §4.2 — *CPU-core-seconds of learner time per 1,000
decisions*) [C]:

| core | learner cost / 1,000 decisions | + MuJoCo climber-rover physics (~81 dec/s [c], `CURIOSITY_BAKEOFF.md` §6) | total |
|---|---|---|---|
| PPO (n_steps 512, 5 epochs, minibatch 512 ⇒ 5 updates per 512 decisions) | **0.84** core-s | 12.3 core-s | **13.1** core-s |
| Dreamer-XS at **train_ratio 1** | **7.3** core-s | 12.3 core-s | **19.6** core-s |
| Dreamer-XS at **train_ratio 8** | 58.6 core-s | 12.3 core-s | **70.9** core-s |
| Dreamer-XS at **train_ratio 64** | 469 core-s | 12.3 core-s | **481** core-s |
| Dreamer-XS at **train_ratio 512** (DreamerV3's own DMC/Atari-100k regime) | 3,750 core-s | 12.3 core-s | **3,762** core-s ≈ **1.05 core-hours per 1,000 decisions** |

**Read that last column before reading any sample-efficiency claim in the
literature.** On this hardware, a world model is *not* intrinsically expensive —
at train_ratio 1 it costs 1.5× a PPO learner, because **MuJoCo is the
bottleneck, not the network**. What is expensive is the **replay ratio**, and
the replay ratio is exactly where DreamerV3's published sample efficiency comes
from. The number that decides whether candidate 2 is affordable for us is
therefore not "how many parameters" but "how much can we cut train_ratio before
the sample efficiency evaporates" — and that is a question no paper answers for
us, so §5 makes it a **measured, pre-registered variable** rather than an
assumption.

### 3.1 Model-free RL (PPO) — the incumbent

#### The simplicity claim, inverted

PPO is universally described as "the simple one". Measured by the thing that
actually costs us — **the number of decisions a human must get right, each of
which can be silently wrong** — it is the most complex candidate in this
survey.

| source | count |
|---|---|
| Huang et al., *The 37 Implementation Details of PPO*, ICLR Blog Track 2022 [V] | **13 core** + 9 Atari + 9 continuous-action/robotics + 5 LSTM + 1 MultiDiscrete = **37**, plus 4 auxiliary = **41**. Their own framing: *"reproducing PPO's results has been a challenging issue."* |
| Engstrom et al., *Implementation Matters in Deep Policy Gradients*, arXiv:2005.12729, ICLR 2020 [V] | **nine code-level optimisations** are *"substantially responsible for PPO's performance gains over TRPO"* — value clipping, reward scaling, orthogonal init + layer scaling, Adam LR annealing, reward clipping, observation normalisation, observation clipping, tanh, global gradient clipping. Stripped of them, PPO ≈ TRPO. And PPO's clipping **does not actually enforce a trust region**. |
| Andrychowicz et al., *What Matters in On-Policy RL*, arXiv:2006.05990 [V] | **>50 design choices**, **250,000+ agents trained**, 5 MuJoCo envs. Even after all recommendations, **γ, learning rate and the clip threshold still require per-environment tuning**. |
| our `PipelineConfig` | **20** training-rule knobs [M] |

Contrast the competing claim, which is *demonstrated* rather than argued:
**DreamerV3 runs 150+ tasks on a single configuration** (arXiv:2301.04104 [V]),
and **TD-MPC2 runs 104 tasks on a single hyperparameter set** while beating
DreamerV3 (arXiv:2310.16828 [V]). On the specific axis of *how much per-task
tuning is needed*, the world-model methods have the stronger case and it is not
close.

**This is the single most counter-intuitive finding in the survey**, and it
matters here more than anywhere because of `LESSONS.md`: the two most expensive
bugs in this project were a dropout default and an observation-dimension
mismatch, both of them items on Engstrom's and Huang's lists, both invisible,
both costing GPU-hours and a wrong owner-facing recommendation.

Andrychowicz's specific recommendations are already partly absorbed by D1
(separate policy and value nets; narrow policy, wide value; policy last layer
100× small; tanh; initial action std ≈ 0.5; GAE λ = 0.9; always normalise
observations; multiple passes over the data is *"crucial"*). §5's PPO arms use
that configuration, so the incumbent is tested at its best rather than as we
happen to have it.

#### What this repo has actually measured [L]

| | |
|---|---|
| SB3 PPO MLP, Humanoid-v5, 124,707 params | **530.2 ± 59.0**, 7.11σ over random |
| our own PPO harness + 57M trunk, matched env-steps | 317.7 ± 84.2, 2.46σ — **below the learning gate** |
| the same comparison, honestly | **confounded**: dropout live in rollout, update and eval (42 % action drift); 6,240 vs 99,840 optimiser steps at "matched" env-steps |
| an **untrained** MLP | 175.1 — **2.74σ** over random, against a 3.0σ gate |

Three things follow, and only the third is about PPO:

1. **We have never run a clean PPO experiment on the big model.** D1's
   correction is explicit: option C ("keep training end-to-end") is *untested,
   not refuted*. Any claim here that PPO "failed" is not supported.
2. **Our null is weak.** An untrained network clears 2.74σ against random
   actions. Every gate in this document is therefore against the arm's **own
   untrained twin**, not only against random (D1 §6.1's Gate B).
3. **PPO's real cost on this project has been hyperparameters and silent
   defaults, not sample complexity.** 20 training-rule knobs in
   `PipelineConfig` [M], plus a dropout default nobody set, plus an
   observation-normalisation mode, plus a minibatch change that divided
   gradient steps by 8 without anyone noticing. That is the (d) column, and it
   is where PPO scores worst.

### 3.2 World models — DreamerV3, TD-MPC2, STORM

#### 3.2.1 First, a correction that changes the argument

**v1 (arXiv:2301.04104, Jan 2023) and the 2025 Nature version are different
papers with different numbers.** v1 uses the XS–XL / 8M–200M classes and V100s;
the Nature version uses a 12M–400M sweep, A100s, and **drops Crafter as a
headline benchmark**. Most secondary sources mix them. Below is kept separate.

#### 3.2.2 Size classes, and what the small ones actually achieved

v1 Table B.1 [V]:

| | XS | S | M | L | XL |
|---|---|---|---|---|---|
| GRU units | 256 | 512 | 1024 | 2048 | 4096 |
| CNN multiplier | 24 | 32 | 48 | 64 | 96 |
| dense units | 256 | 512 | 640 | 768 | 1024 |
| MLP layers | 1 | 2 | 3 | 4 | 5 |
| **params** | **8M** | **18M** | **37M** | **77M** | **200M** |

**The honest answer to "what does the smallest size class achieve": nobody
published it.** The model-size sweep (v1 Fig. 6b) was run on four tasks at
*large* step budgets (Breakout/MsPacman 0–500M, Crafter 0–60M, DMLab 0–450M).
The claim is qualitative and monotonic. **Any "DreamerV3-XS gets X on
Atari100k" is an extrapolation.** What *is* stated is which size each benchmark
used: v1 uses **S (18M) for both DMC suites and for Atari100k**, XL for
Crafter/BSuite/DMLab/Minecraft; the Nature version uses **12M for both control
suites** and notes *"the same performance using the substantially faster 12M
model, making it more accessible to researchers"* [V].

**Consequence for us, stated plainly:** the benchmarked shape in §3.0 is
**1.9M parameters — below XS.** No published result exists at that size. §4.3's
counterargument (5) stands.

#### 3.2.3 Crafter, and the number that must not be quoted

Crafter score (arXiv:2109.06780 [V]) is the geometric mean over 22 achievements,
`S = exp( (1/N) Σ ln(1 + s_i) ) − 1`, at a standard budget of **1M env steps**.

DreamerV3 v1 Table M.1 [V], all at 1M steps:

| method | score % |
|---|---|
| Human experts | 50.5 ± 6.8 |
| **DreamerV3 (XL, 200M)** | **14.5 ± 1.6** |
| DreamerV2 | 10.0 ± 1.2 |
| **PPO** | **4.6 ± 0.3** |
| Rainbow | 4.3 ± 0.2 |
| Plan2Explore (unsupervised) | 2.1 ± 0.1 |
| RND (unsupervised) | 2.0 ± 0.1 |
| Random | 1.6 ± 0.0 |

> ### **That PPO number is a strawman, and this is the most important finding in the survey.**

Moon et al., *Discovering Hierarchical Achievements in RL via Contrastive
Learning* (arXiv:2307.03486, **NeurIPS 2023**), Table 1 — same 1M steps, 10
seeds [V]:

| method | params | score % |
|---|---|---|
| Human expert | — | 50.5 ± 6.8 |
| **Achievement Distillation (PPO + contrastive auxiliary)** | **9M** | **21.79 ± 1.37** |
| **PPO, modern implementation practices** | **4M** | **15.60 ± 1.66** |
| **DreamerV3** | **201M** | **14.77 ± 1.42** |
| LSTM-SPCNN | 135M | 11.67 ± 0.80 |
| MuZero + SPR (from scratch) | 54M | 4.4 ± 0.4 |

The fixes that take PPO from **8.17 % to 15.60 %** are mundane: widen the IMPALA
ResNet channels from [16,32,32] to [64,128,128] and hidden 256→1024, **add
LayerNorm before every dense and conv layer**, and **normalise value targets by
a running mean/std**.

So on the one survival-flavoured benchmark in the literature, at the standard
budget: **a properly-implemented 4M-parameter PPO matches or beats a
201M-parameter DreamerV3 using ~2 % of the parameters**, and PPO plus one
auxiliary loss beats it by 47 % relative — unlocking all 22 achievements,
collecting iron at 20× DreamerV3's rate, and crafting iron tools DreamerV3
never crafts.

Two things follow, and they are both binding on this document:

1. **The sample-efficiency case for a world model does not survive contact with
   a Crafter-class environment at 1M steps.** §4 is rewritten accordingly: the
   recommendation now rests on unison, not on efficiency.
2. **Our PPO arms must be the tuned kind.** Comparing a world model to an
   under-engineered PPO is how the 4.6 % number happened, and it is the exact
   shape of `LESSONS.md`'s dropout confound — *one arm had a handicap and the
   difference was attributed to architecture*. §5.1 gains invariant **F9**.

#### 3.2.4 Where world models genuinely do buy sample efficiency

Not everywhere, and the pattern is legible:

| domain | evidence | gain |
|---|---|---|
| **Atari100k** (400k frames) | v2 Table 9 [V] | DreamerV3 **125 % mean / 49 % median** vs PPO **11 % / 2 %**. Unambiguous and enormous |
| **DMLab** | v1 [V] | DreamerV3 matches IMPALA's final performance at **50M steps vs IMPALA's 10B** — a stated **>13,000 %** data-efficiency gain (~200×). The paper's own caveat: *"IMPALA was not designed for data-efficiency"* |
| **Minecraft diamonds** | v1 [V] | **17 V100-days, one GPU**, 100M steps, 24/40 seeds got a diamond, first at 29.3M steps. Versus VPT (arXiv:2206.11795): **720 V100s × 9 days ≈ 6,480 GPU-days** *plus* human data |
| **Crafter @1M** | arXiv:2307.03486 [V] | **no advantage** over tuned PPO |
| **Crafter beyond ~3M** | Δ-IRIS, arXiv:2406.19320 [V] | DreamerV3 XL wins at 1M (9.2 return vs 7.7) and **loses beyond ~3M**; Δ-IRIS reaches score **39.67 @5M, 42.47 @10M** vs human 50.5 |

**The synthesis, and it is directly actionable for W0:** world models dominate
at *very small* step budgets and on *long-horizon sparse-reward 3D* tasks. At
1M steps in a 2D-ish survival world, the advantage has already evaporated. And
Δ-IRIS's Crafter curve says something W0 should hear: **Crafter is not an
unsolved-capability problem, the 1M budget is the binding constraint** — *if
your environment is cheap to step, buying 10× more env-steps beats buying a
better algorithm.* Our climber-rover **is** cheap to step (~81 dec/s on 4 ARM
cores, no GPU).

#### 3.2.5 "Fixed hyperparameters" — what the claim covers, precisely

v1 Table W.1 / v2 Table 4 hold **25 hyperparameters** constant across 150+
tasks [V]: replay 5×10⁶, batch 16 × length 64, RMSNorm+SiLU, lr 4e-5, AGC(0.3),
LaProp(ε=1e-20), β_pred 1 / β_dyn 1 / β_rep 0.1, unimix 1 %, free nats 1,
imagination horizon 15, discount horizon 333 (γ=0.997), λ 0.95, β_val 1,
β_repval 0.3, critic EMA reg 1 / decay 0.98, β_pol 1, entropy 3e-4, actor
unimix 1 %, RetNorm Per(R,95)−Per(R,5) / limit 1 / decay 0.99. *"We do not use
any hyperparameter annealing, prioritized replay, weight decay, or dropout."*

> **But two things are explicitly NOT fixed and are chosen per benchmark:
> MODEL SIZE and REPLAY/TRAIN RATIO** — both are columns in v1 Table A.1 and v2
> Table 2. Wall-clock is roughly linear in train ratio and superlinear in size.
> **"Fixed hyperparameters" does not mean "fixed compute", and the two knobs
> the paper varies are exactly the two that decide whether a run fits on free
> compute.** The real config surface is `configs.yaml`, **220 lines, well over
> 100 knobs.**

The five tricks, and what each is for:

| trick | form | what it removes |
|---|---|---|
| **symlog** | `symlog(x) = sign(x)·ln(\|x\|+1)` on vector obs, `symexp` on decoder outputs | per-domain observation normalisation |
| **symexp twohot** | reward head and critic emit logits over **255 exponentially spaced bins**, `B = symexp(linspace(−20,20,255))`; targets twohot; loss cross-entropy | couples gradient magnitude to target magnitude |
| **free bits + KL balancing** | `L_dyn = max(1, KL(sg(q)‖p))`, `L_rep = max(1, KL(q‖sg(p)))`, β_dyn 1 / β_rep 0.1 | per-domain "how strong a regulariser" tuning |
| **percentile return normalisation** | `S = EMA(Per(R,95) − Per(R,5), 0.99)`; divide returns by `max(1, S)` | the `limit=1` floor stops sparse-reward noise being amplified. The paper is explicit that normalising *advantages* (PPO's convention) **fails** here |
| **unimix** | all categoricals = 1 % uniform + 99 % network | determinism collapse and KL spikes |

#### 3.2.6 The silent-default landmines — read this section twice

Taken from the live `configs.yaml` on `main` [c, code read by the sweep]. This
is the (d) column, and every one of these is the `nn.Dropout(p=0.1)` shape: a
value nobody set, with no line of code to read.

1. **The default model size is 200M.** The `defaults:` block sets
   `rssm.deter: 8192, units: 1024, depth: 64` — i.e. `size200m`. **If you do
   not pass a size preset you get the biggest sane model.** The single most
   expensive default in the file.
2. **`train_ratio` defaults to 32 but the presets override hard**: crafter 512,
   dmc_proprio 1024, bsuite 1024, atari100k 256, dmc_vision 256, atari/dmlab
   32, procgen 64. Wall-clock is ~linear in it. **Copying the Crafter preset
   costs 16× the default.**
3. **`enc.simple.symlog: True` squashes vector observations.** For Jack, whose
   needs are already in [0,1], symlog compresses them toward zero — **the
   hunger signal gets roughly half the dynamic range of a raw pixel.**
4. **`loss_scales.rec: 1.0` is shared across all reconstruction keys.** A
   64×64×3 image contributes **12,288** reconstruction terms; a 10-dim needs
   vector contributes **10**. The world model will spend essentially all
   capacity on pixels and **ignore the modality the whole project is about.**
   This is the exact target of HarmonyDream (arXiv:2310.00344 [c, numbers not
   verified]).
5. **`replay.capacity: 5e6`** — at 64×64×3 uint8 that is **~61 GB** of frames.
   On Kaggle/Colab (13–30 GB RAM) this is a hard blocker if missed.
6. Also fixed and easy to miss: `retnorm.limit 1.0`, `free_nats 1.0`, `unimix
   0.01` on **both** RSSM and policy, `horizon 333`, `imag_length 15`,
   `bins 255`, `actent 3e-4`, `opt.warmup 1000`.

**Items 3 and 4 together are a project-specific hazard of the first order.**
They would make a "unified brain" that has quietly deleted the need-state
modality, while every loss curve looks correct. §5.1 gains invariant **F10**
because of them.

#### 3.2.7 Compute — the numbers that decide feasibility

v1 Table A.1, **all on ONE V100** [V]:

| benchmark | steps | envs | train ratio | **GPU-days** | size |
|---|---|---|---|---|---|
| DMC Proprio | 500K | 4 | 512 | **<1** | S |
| DMC Vision | 1M | 4 | 512 | **<1** | S |
| **Crafter** | **1M** | 1 | **512** | **2** | **XL** |
| Atari100k | 400K | 1 | 1024 | <1 | S |
| Minecraft | 100M | 16 | 16 | **17** | XL |

v2, **all on ONE A100** [V]: Minecraft 8.9 · Atari100k **0.1** · Proprio
Control **0.3 (12M)** · Visual Control **0.1 (12M)**.

**The conversion factor is the paper's own** (v1 Table T.1 caption): *"GPU days
are converted to V100 days by assuming **P100 is twice as slow** and **A100 is
twice as fast**."* So **1 A100-day ≈ 2 V100-days ≈ 4 P100-days** [V].

Derived for our hardware [C]:

| run | P100-hours | fits 30 h/week? |
|---|---|---|
| Crafter 1M, **XL/200M**, train_ratio 512 | **~96** (paper) / **~37** (from Δ-IRIS's measured 30 env-FPS for DreamerV3-XL on an A100) | **No** — honest range 37–96 |
| **Visual Control 1M, 12M model, train_ratio 256** | **~9.6** | **Yes, comfortably** |
| Atari100k, one game | ~9.6 (v2) / ~24 (v1) | yes per game; ×26 games no |

> **Bottom line: a 12M-parameter DreamerV3 at train_ratio ~256 on a 1M-step
> lightweight environment is roughly a 10–20 hour P100 job — one Kaggle week.
> The 200M default at train_ratio 512 is not.** Scale linearly in train ratio.

#### 3.2.8 Multimodal dict observations — verified in the code, and it is free

`dreamerv3/rssm.py`, `Encoder.__init__` [c, code read]:

```python
assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
```

Routing is **automatic by observation shape** — the old `mlp_keys`/`cnn_keys`
regexes are gone. Vector keys go through `DictConcat(..., squish=symlog)` then
3 × (Linear + Norm + SiLU); image keys are channel-concatenated, scaled to
[−0.5, 0.5], and run through the CNN; **fusion is late concatenation** of the
two branch outputs, and that concatenation is what the RSSM consumes. The
default `dmc` env config is `{proprio: True, image: True}` and the benchmark
presets *disable* one for baseline comparability — **both-on is the natural
mode.** DayDreamer (arXiv:2206.14176 [c]) is the real-robot image+proprio
precedent.

Caveats: multiple image keys must share resolution; and the late-concatenation
fusion plus landmine 4 means **the multimodality is structurally free but not
automatically balanced.**

#### 3.2.9 Reproduction difficulty — the honest record

- **`danijar/dreamerv3` issue #175, open, no maintainer response** [c]: a
  reporter gets **−21 (the floor) on Pong across seeds on RTX 3090s**, agent
  motionless in sample trajectories; confirms the *2023* code reproduces ~18,
  the current code does not for them; tried train_ratio 128 and 256 and the 12M
  model. Matched published scores on Alien, Amidar, Assault, Boxing; failed on
  Asterix, Battle Zone, Up N Down. **This is a live, unresolved reproduction
  failure in the flagship benchmark.**
- Open issues #43 (determinism), #55 (XLA `RET_CHECK`), #154 (*published
  configs and the repo disagree* on Atari 200M's train ratio).
- TD-MPC2's paper states DreamerV3 *"experiences occasional numerical
  instabilities"* on object manipulation [V].
- A 2025 paper exists specifically on **tuning DreamerV3 hyperparameters** for
  traffic-signal control (arXiv:2503.02279 [c]), which undercuts the no-tuning
  framing.

#### 3.2.10 Implementations and their size

| repo | what | LOC |
|---|---|---|
| official `danijar/dreamerv3` (JAX) | `agent.py` 490 + `rssm.py` 359 + `configs.yaml` 220 — **tiny**, but sits on `embodied`, `elements`, `ninjax`, JAX | ~1,070 core |
| **`NM512/dreamerv3-torch`** (887★) | the de facto PyTorch port: `tools.py` 1000, `networks.py` 810, `models.py` 441, `dreamer.py` 365, `parallel.py` 209, `exploration.py` 135 = **2,960 LOC** + 1,151 LOC env wrappers. **README now warns it "does not reflect" major DreamerV3 updates** | 2,960 |
| **`NM512/r2dreamer`** | official R2-Dreamer (ICLR 2026, arXiv:2603.18202; decoder-free, augmentation-free, redundancy-reduced latents). Ships *"an efficient PyTorch DreamerV3 reproduction that trains **~5× faster** than dreamerv3-torch"*, R2-Dreamer a further ~1.6×. Includes **Crafter**, DMC, Atari100k, Meta-World | — [c, speedups unverified] |
| `jurgisp/pydreamer` | DreamerV2, unmaintained since Apr 2023 | not recommended |
| "tinyworldmodel" | **could not be found; probably does not exist** | — |

Note against B3 (§6): the official core is ~1,070 lines and the PyTorch port is
~3,000. **A from-scratch W0 implementation targeting B3's 1,500-line ceiling is
therefore plausible but not comfortable**, and that is a real cost of this
candidate.

#### 3.2.11 TD-MPC2 — strong, and disqualified for Jack

arXiv:2310.16828 [V]. Decoder-free implicit latent model + **MPPI** planning at
act time. **104 continuous-control tasks** (DMControl 39 incl. Dog ℝ³⁸ and
Humanoid ℝ²¹, Meta-World 50, ManiSkill2 5, MyoSuite 10), beating SAC, DreamerV3
and TD-MPC, with **one hyperparameter set (~40 values)** and only two
task-dependent heuristics.

Scaling on one RTX 3090 (80-task multitask):

| params | GPU-days | score |
|---|---|---|
| 1M | 3.7 | **16.0** |
| 5M | 4.2 | 49.5 |
| 19M | 5.3 | 57.1 |
| 48M | 12 | 68.0 |
| 317M | 33 | 70.6 |

**Three disqualifiers, and they are structural, not fixable by tuning:**

1. **State-based proprioceptive observations only — no vision, no images.**
   Under the constitutional unison constraint (§3.7) that is an **inadmissible
   core**, not a weak one.
2. **Continuous actions only** — no discrete action support.
3. **Planning costs at every decision**: horizon 3, 6 iterations (+2 if
   |A| ≥ 20), population 512, 64 elites. Dreamer acts with one policy forward.
   Against §5.0b's throughput floor that is a heavy tax, every step, forever.

And note the small-scale numbers: **1M params scores 16.0, 5M scores 49.5.**
The useful regime starts around 19M–48M. At the 1–2M scale we can afford,
TD-MPC2's own sweep says it barely works.

**So TD-MPC2 is not a candidate.** Its *decoder-free* idea survives as arm A4
(§5.4), which is the right way to test the idea without importing the
architecture.

#### 3.2.12 STORM, DIAMOND, Δ-IRIS, R2I — the transformer/diffusion wave

Atari100k (100k agent steps = 400k frames), mean human-normalised:

| method | mean HNS | median/IQM | cost |
|---|---|---|---|
| SPR | 62–70 % | 40–41 % | 0.2 V100-d |
| TWM (2303.07109) | 95–96 % | 50–51 % | 0.8 V100-d |
| IRIS (2209.00588) | 104–105 % | 29 % | 7 V100-d |
| **DreamerV3** | **112 % (v1) / 125 % (v2)** | 49 % | **0.5 V100-d** |
| **STORM (2310.09615)** | **126.7 %** | **58.4 %** | **4.3 h on one RTX 3090** |
| **DIAMOND (2405.12399)** | **145.9 %** | IQM 0.641, **11/26 superhuman** | **~2.9 days on one RTX 4090, per game per seed** (~1.03 GPU-**years** for the full benchmark); **12 GB VRAM**, 13M params |
| EfficientZero | 190 % | 109 % | 1.2 V100-d |

**STORM is the cheapest strong result in the table** — 4.3 h on a 3090, roughly
12–20 P100-hours [C, and P100 has no tensor cores so transformer workloads
suffer more than the blanket 2× implies].

**Δ-IRIS (arXiv:2406.19320) is the most relevant of the four for us** because it
benchmarks on **Crafter**: context-aware tokenisation encodes a frame in **4
tokens / 40 bits** vs IRIS's 16/160. Crafter geometric score: **9.30 @1M,
39.67 @5M, 42.47 @10M** (human 50.5) at 25M params and 20 FPS on an A100;
DreamerV3-XL is 30 FPS and wins at 1M (9.2 vs 7.7 return) then loses beyond
~3M. Atari100k in **26 h/game on an A100** — still >30 h on a P100.

**R2I (arXiv:2403.04253, ICLR 2024)** — DreamerV3 with a modified **S4** state-
space model in the world model. **Up to 9× computational speedup over
DreamerV3**, SOTA on BSuite and POPGym, **superhuman on Memory Maze**;
"comparable" on Atari and DMC (i.e. no gain there). Its Memory Maze runs are
2× A100 40GB at ~350 FPS — not free-compute-reachable. **But R2I is the right
reference the day Jack needs long-horizon memory inside the model** — *where
did I leave the water* — and it is the published way to get it.

#### 3.2.13 Director — the best cost/capability point in the entire survey

Hafner et al., *Deep Hierarchical Planning from Pixels*, arXiv:2206.04114 [V].
A manager selects **latent goals** every fixed number of steps (discretised by
a goal autoencoder into 8×8 categoricals); a worker reaches them with primitive
actions; everything inside a learned world model; goals are decodable to images
for inspection.

- Solves **Egocentric Ant Maze** at all four difficulty levels **from an
  egocentric camera plus proprioception**, without the global position or
  top-down view prior work needed — and **none of the baselines solve the
  larger mazes.** Also Visual Pin Pad (long-horizon sparse credit), Atari, DMC,
  DMLab, Crafter.
- **~250 lines on top of DreamerV2. 20 % slower than DreamerV2. Four env
  instances. ONE gradient step per SIXTEEN policy steps** — *train_ratio ≈
  0.06*, versus Crafter's default 512, described in-paper as *"drastically
  reduced wall-clock time and decreased sample-efficiency mildly"*. **Each run
  completed in under 24 hours on a single V100** ≈ 2 P100-days.

Three reasons this matters more than its citation count suggests:

1. **It is the existence proof that the replay ratio is a dial, not a
   property.** §3.0's cost table said the replay ratio decides affordability;
   Director ran at **train_ratio 0.06** and still solved mazes nothing else
   solved. That collapses §4.3's counterargument (3) from "probably fatal" to
   "measure it".
2. **Its observation space is ours.** Egocentric camera + proprioception is the
   closest published precedent to W0's retina + proprio + needs.
3. **250 lines.** Against B3's 1,500-line ceiling, hierarchical goal-directed
   behaviour on top of a world model is a *small* addition — which is the
   opposite of the intuition that hierarchy is complexity.

#### 3.2.14 The gap this project would be filling

The sweep looked for it and did not find it: **no published work applies
DreamerV3 (or any world model) to an explicit homeostatic drive function.**
Crafter is the only survival-flavoured environment DreamerV3 was run on, and
Crafter's agent sees hunger/thirst only as **pixels in a HUD** — there is no
internal-needs vector exposed to the agent. Likewise **no HRRL + world-model
paper exists** (§3.3.4's homeostatic literature is PPO- and tabular-based).

So *"world model + homeostatic drive as the reward source, with the needs
vector as a first-class modality"* is **unclaimed territory**. That is a reason
to be interested and a reason to be careful: unclaimed territory has no
baseline to beat and no prior art to inherit, which is precisely the condition
under which a project convinces itself it has succeeded. Every gate in §5.5
exists because of that sentence.

And one negative result from the same family that must be carried forward:
**Plan2Explore scores 2.1 ± 0.1 % on Crafter** — barely above random's 1.6 %
[V]. Pure novelty-seeking on a world model does **not** produce survival
competence. `GOAL.md` says curiosity is the curriculum; this is the sharpest
published evidence that curiosity *alone* is not, and it is a direct argument
for `PURPOSE_AND_SCAFFOLDING.md`'s needs.

### 3.3 Active inference / the Free Energy Principle

This is the candidate with the strongest *a priori* case for Jack — one
equation that is claimed to produce homeostasis and curiosity together — and it
is the candidate where the demonstrated/argued gap is widest. Both halves of
that sentence are true and the section is organised to keep them apart.

#### 3.3.1 The mathematics, stated plainly (ARGUED — and correct)

**Variational free energy is the negative ELBO.** For a generative model
`p(o,s)` and a recognition density `q(s)`:

```
F[q,o] = E_q[ln q(s) − ln p(o,s)]
       = KL[q(s) ‖ p(s|o)]  −  ln p(o)      ≥  −ln p(o)
         └── ≥ 0 ──────────┘   └ surprisal ┘
```

so minimising `F` tightens a bound on surprisal, i.e. maximises log model
evidence [V, Parr/Pezzulo/Friston 2022, MIT Press, open access,
doi:10.7551/mitpress/12441.001.0001]. Equivalent readings: `complexity −
accuracy`, and `energy − entropy`. **None of this is novel relative to
variational Bayes.** The FEP's distinctive move is that *action* also minimises
`F`, by changing `o`.

**Expected free energy (EFE)** scores a policy `π` over the future:

```
G(π) = −E_Q(o|π)[ln C(o)]  −  E_Q(o|π) KL[ Q(s|o,π) ‖ Q(s|π) ]
       └── PRAGMATIC ─────┘     └── EPISTEMIC = I(s;o|π) ─────┘
```

with `Q(π) = σ(−γ·G(π))`. Set `ln C(o)` to be high for "fed, warm, unhurt" and
the pragmatic term *is* homeostasis; the epistemic term *is* expected
information gain. **That is the unification the brief points at, and it is
real algebra, not hand-waving.** For Jack specifically the mapping is
attractive: prior preferences over *interoceptive* observations are exactly
`PURPOSE_AND_SCAFFOLDING.md`'s setpoint `h* = (1,1,0)`.

Four results that qualify it, all of them from inside the active-inference
literature:

- **EFE is not "the free energy of the future."** Millidge, Tschantz & Buckley,
  *Whence the Expected Free Energy?* (arXiv:2004.08128, Neural Computation
  33(2):447–482, 2021) [V]: the natural temporal extension of VFE yields an
  objective that **actively discourages** exploration. Exploration does not
  fall out of free-energy minimisation for free; it is obtained by *choosing*
  EFE over the natural extension. Their alternative (FEEF) is proposed
  precisely because EFE's derivation is not as clean as its presentation.
- **The four EFE formulations in circulation are not equal.** Champion et al.,
  *Reframing the Expected Free Energy* (arXiv:2402.14460) [V] prove
  `G = C_ROA = C_IGPV ≤ C_RSA = C_3E`. The "risk over states + ambiguity" form
  that many implementations code is an **upper bound**, not the EFE. Papers
  swap between forms as if they were identical, and nothing warns you.
- **On MDPs, active inference is KL control.** Millidge, Tschantz, Seth &
  Buckley, *On the Relationship Between Active Inference and Control as
  Inference* (arXiv:2006.12964) [V]: the only difference from control-as-
  inference (Levine, arXiv:1805.00909 [V]) is *where value enters the
  generative model* — a likelihood `e^r` versus a biased prior over
  observations. They state it outright: *"AIF on MDPs is equivalent to KL
  control."*
- **The preferences `C` are a reward function in log space, and the field says
  so.** Sajid, Ball & Friston (arXiv:1909.10863, Neural Computation 33(3), 2021)
  [V]: adjusting preference values *"corresponds to a type of reward tuning."*
  Torresan, Kanai & Baltieri (arXiv:2512.03293, Dec 2025) [V] run the 2×2 and
  find *"goal shaping enables the best performance overall (promotes
  exploitation) while sacrificing learning about the environment's transition
  dynamics"* — reward shaping's exact trade-off, renamed. And there is an
  inverse-RL paper for learning `C` (arXiv:2101.08937 [c]).

**The consequence for the brief's central attraction.** The claim is that EFE
unifies homeostasis and curiosity *in one objective with no free parameter
balancing them*. That is **false as stated**: the balance is set by the scale
of `ln C` in nats and by the precision `γ`. It is the intrinsic/extrinsic
coefficient, wearing a different notation. The honest version of the claim is:
*active inference gives a principled vocabulary in which homeostasis and
curiosity are the same kind of quantity* — which is genuinely valuable for how
we think and design, and buys us nothing at implementation time.

#### 3.3.2 The empirical record (DEMONSTRATED — and thin)

| paper | task | baseline | result |
|---|---|---|---|
| Ueltzhöffer, arXiv:1709.02341 | **Mountain Car** | none | works; needed **evolution strategies** — the objective is not differentiable through the env |
| Millidge, arXiv:1907.03876 | Gym classic control | RL | "comparable" |
| Tschantz et al., arXiv:1911.10601 / arXiv:2002.12636 | MountainCar, Cup Catch, HalfCheetah, Ant Maze | **SAC** (model-free) | beats SAC at 100 episodes on HalfCheetah; **no benefit** on Cup Catch |
| van der Himst & Lanillos, arXiv:2009.03622 | **CartPole** from pixels | DQN | "comparable or better" |
| Fountas et al., NeurIPS 2020, arXiv:2006.04176 | dSprites, Animal-AI | **none named** | qualitative only; no head-to-head numbers |
| Mazzaglia et al., NeurIPS 2021, arXiv:2110.10083 | image control with distractors | likelihood-AIF; RL with hand-crafted reward | **matches** RL; better under distractors. The contribution is *dropping the reconstruction likelihood* — i.e. it is a JEPA-flavoured result (§3.4) |
| **Champion et al., arXiv:2303.01618** | dSprites | reward-maximiser | **the EFE agent FAILED** — *"always picks the action down"*; the reward-maximiser solved it. The epistemic term caused policy collapse |
| Paul et al., arXiv:2307.00504, *Expert Systems* 2024 | 100/400-state grid worlds | Dyna-Q | "at par" |
| **Nguyen et al., R-AIF, arXiv:2409.14216** | pixel MountainCar, 13 Meta-World, 2 Robosuite; **sparse reward, POMDP** | **DreamerV3** | **the one legitimate Dreamer-class win.** Robosuite Door **100 % (425.8±10.0) vs 0 % for every baseline**; MountainCar −68.3±1.0 @100 % vs DreamerV3 −79.7±12.4 @90 %. On easy dense tasks it **ties** (Button Press −38.2 vs −37.2) |
| Yokozawa et al., arXiv:2510.23258 (Oct 2025) | **real TurtleBot 4** indoor nav, perceptual aliasing | own ablations | 75 % vs 64 % (no RSSM) vs 53 % (extrinsic-only); on the exploration-heavy subset **78 % vs 28 %** — a clean ablation that the epistemic term does real work |
| Heins et al., **AXIOM**, arXiv:2505.24784 (VERSES) | **"Gameworld 10k"** | DreamerV3, BBF | normalised 77 vs 48; 7.6× sample efficiency; 0.95M vs 420M params; ~$0.66 vs $25.54 |

**The question the brief asks, answered directly: has any deep active-inference
agent matched or beaten a Dreamer-class baseline on a *standard* benchmark
(Atari100k, DMC, Crafter, MinAtar)?**

> **No. Not once, in the nine years since Ueltzhöffer.** No AIF agent appears
> in the results table of any standard sample-efficiency benchmark. That
> absence, sustained across a decade and a large literature, is itself
> evidence.

The two apparent counterexamples both need their fine print read:

- **R-AIF** is real and peer-relevant, but its win is in the regime where the
  baseline gets *no signal at all* — DreamerV3 scoring 0 % on Robosuite Door
  means the sparse reward never fired, not that AIF is generally better. Where
  DreamerV3 works, R-AIF ties.
- **AXIOM's benchmark is AXIOM's own.** "Gameworld 10k" is ten games VERSES
  generated with an LLM, with *"deliberately simplified visual elements"*, and
  AXIOM's structural priors (a slot mixture with fixed unlearned projections, a
  switching-linear-dynamics transition mixture, a recurrent mixture conditioned
  on hand-picked features like *"distance to the closest object"*) are matched
  to exactly those dynamics. Its planning costs **252–534 ms per step** for
  64–512 rollouts. The authors do not claim it works at full-Atari scale.
  **The transferable idea in AXIOM is not the EFE framing — it is that it is
  not a neural network** (conjugate mixture models + Bayesian model reduction,
  no backprop), which is why it is cheap.

#### 3.3.3 The epistemic term is expected information gain, and we can have it without the framework

This is the pivotal fact for our decision, and it is algebra, not opinion:

```
E_Q(o|π) KL[Q(s|o,π) ‖ Q(s|π)]  =  I(s;o|π)
```

— expected Bayesian surprise. The "parameter information gain" variant swaps
states for model parameters, `I(θ;o|π)`, which is **exactly what Plan2Explore
(arXiv:2005.05960) and Disagreement (arXiv:1906.04161) maximise**, and the
estimators used are *literally the same code*: Tschantz et al. estimate it with
an ensemble of transition models; R-AIF says maintaining multiple world models
is "impractical" and uses *"a separate lightweight ensemble of MLPs"*. Friston
et al. (2015, doi:10.1080/17588928.2015.1020053) [V] already say epistemic
value is *"formally consistent with the Infomax principle"* and generalises
Bayesian surprise. Mazzaglia et al.'s Entropy 24(2):301 review
(arXiv:2207.06415) [V] tabulates AIF design choices directly onto
Dreamer/Plan2Explore components — **and contains no numerical experiments at
all**; it is a taxonomy, and the taxonomy's content is that AIF and model-based
RL are the same design space.

One genuine theoretical point in AIF's favour, which we should bank: the
epistemic term is **noise-robust in principle** (aleatoric uncertainty cancels
in the mutual information), whereas RND (arXiv:1810.12894) is a novelty proxy
that a noisy TV maximises. This is the same property `CURIOSITY_BAKEOFF.md`
§1.1 attributes to Disagreement and to learning progress — so it is a reason to
prefer *those*, not a reason to adopt EFE.

Two honest counter-data on the epistemic term itself: Yokozawa's real-robot
ablation says it helps a lot under perceptual aliasing (78 % vs 28 %);
Champion's dSprites result says it can **cause** collapse. It is a term with a
regime, not a universal good.

#### 3.3.4 Homeostatic RL — the empirically grounded cousin, and the part we should actually take

`PURPOSE_AND_SCAFFOLDING.md` §2.5 already adopts Keramati & Gutkin's form
(eLife 2014;3:e04811, doi:10.7554/eLife.04811) [V]. Three additions this sweep
turned up that matter to the learning core:

- **The theorem has a γ condition, and it is load-bearing.** For `γ < 1`,
  `argmax_π SDR = argmin_π SDD` — maximising discounted drive-reduction reward
  *is* minimising discounted homeostatic deviation, by telescoping. **At γ = 1
  the sum collapses to `D(H₀) − D(H_∞)` and becomes path-independent: the agent
  becomes indifferent to how badly it starves en route.** `PS.00` should assert
  this; it is a one-character way to silently destroy the objective.
- **The p-norm exponents are where the psychology comes from, for free.**
  `∂r/∂|h*−h| > 0` gives *a hungrier agent values food more*; `∂²r/∂k² < 0`
  (needs `n > 1`) gives risk aversion, i.e. sane multi-need scheduling instead
  of bang-bang; and anticipatory regulation (shivering before the cold) is
  optimal under the same objective — directly relevant to the owner's
  temperature need. This is a real argument that a drive reward is *simpler*
  than a hand-written one: several behaviours we would otherwise have to
  engineer are consequences of the geometry.
- **Modular beats scalarised for multiple needs.** Dulberg, Dubey, Berwian &
  Cohen, arXiv:2204.06608 (CCN 2022) [c]: one Q-learner per physiological
  variable, each with its own reward, beats a monolithic DQN on three axes —
  needs *minimal exogenous exploration*, better sample efficiency, more robust
  out of domain. For Jack's five needs this says **five value heads, not one
  scalarised reward**, and it costs almost nothing.
- **Deep homeostatic RL with multiple needs and real physics already works.**
  Yoshida et al., *PNAS Nexus* (2024), doi:10.1093/pnasnexus/pgae540 [c]:
  **PPO**, 4-layer MLPs, a MuJoCo 8-joint quadruped, two nutrients, weighted
  drive; reproduces all three nutritional-geometry curves from animal data, and
  sweeping the drive-weight ratio 1:1 → 16:1 moves behaviour between them.
  **This is the existence proof for the exact combination Jack needs — and its
  learning core is PPO.** Note the body: 8 joints, the same class as our
  climber-rover.
- Nearest published architecture to the owner's description: Yoshida, Kanazawa
  & Kuniyoshi, IJCNN 2023, doi:10.1109/IJCNN54540.2023.10191925 [c] —
  Interoceptive Mixture of Experts with behaviour switching driven by internal
  body state, continuous motor control.
- One design detail worth stealing, from K&G's "orosensory approximation":
  **reward the *predicted* homeostatic effect at consumption time, not the
  delayed physiological update.** It is why IV nutrition is not rewarding, and
  it removes a credit-assignment delay for free.

#### 3.3.5 Cost and what breaks silently (question (d))

A deep active-inference agent is a **strict superset** of DreamerV3: everything
Dreamer has, plus a preference model `C`, a nested Monte-Carlo EFE estimator, an
ensemble (or MC-dropout) for the information-gain term, a habit/policy prior,
a planner, and the scalars `γ` and the nat-scale of `C`. There is **no minimal
deep implementation with a credibility record**; `pymdp` (arXiv:2201.03904) is
clean, maintained, and **discrete POMDPs only** — no continuous state, no
pixels, so it does not cover our case.

Silent failure modes, each of which would be invisible on a loss curve — this
is the (d) column and it is why the recommendation goes the way it does:

1. **Policy enumeration is exponential.** Classical AIF is `O(|S|·|U|^T)`;
   sophisticated inference is `O((|S|·|U|)^T)`. At `|S|=100, |U|=4, T=30` that
   is ~10⁶⁸ operations [V, arXiv:2307.00504]. You must use their DPEFE
   backward induction (`O(|S|·|U|·T)`) or you never get past `T ≈ 5`.
2. **The EFE expectation is doubly intractable** — over predicted observations
   *and* predicted states. Sampling estimators are high-variance and the bias
   is silent: the agent just becomes quietly less curious. Nothing errors.
3. **Form-swapping introduces a hidden bound** (arXiv:2402.14460): implement
   "risk over states + ambiguity" and you are optimising an upper bound on the
   thing you named. No warning fires.
4. **Epistemic-term collapse** (arXiv:2303.01618): the intrinsic reward stays
   nonzero while coverage collapses to one action. It *looks* like exploration.
5. **The preference/epistemic weighting is an unavoidable hyperparameter in
   nats** — the coefficient the framework promised to abolish.
6. **The objective is not differentiable through the environment.** Every
   implementation either learns a differentiable world model (and thereby
   becomes Dreamer) or plans by sampling (and pays AXIOM's 252–534 ms/step).

Items 2, 3 and 4 are all the same shape as this project's most expensive bug:
*a wrong thing that produces a plausible number and no error*. `LESSONS.md`
("Call .eval()", "an assertion made against a saturated quantity cannot fail")
says that class of defect costs us more than anything else. That is the
strongest practical argument against making EFE the learning core, and it is
independent of the empirical record.

### 3.4 JEPA-family latent-predictive world models

#### 3.4.1 LeCun's position paper (ARGUED — and it is arguing for our design)

*A Path Towards Autonomous Machine Intelligence* (v0.9.2, 2022, OpenReview
`BZ5a1r-kVsf`) [c — the OpenReview PDF would not parse; details via two
secondary sources that quote it directly]. Six modules: configurator,
perception, world model, **cost**, actor, short-term memory. The cost module
splits into:

- **Intrinsic Cost** — explicitly *"hard-wired (immutable, non-trainable)"*, a
  single scalar; LeCun's analogy is the amygdala; proposed contents are **pain,
  pleasure, hunger, curiosity, preference for social interaction, empathy**;
  and he argues it must stay immutable to prevent *"a kind of behavioral
  collapse."*
- **Trainable Critic** — predicts future intrinsic cost from a short-term
  memory of (state, intrinsic-cost) pairs.

That is, almost line for line, `PURPOSE_AND_SCAFFOLDING.md`'s drive layer plus
`GOAL.md`'s needs list, written by someone else three years earlier. It is
worth recording as convergent design, and worth being clear that **it is a
position paper with no experiments**. The named open problem is exactly ours:
the critique of the paper observes that it gives *"only vague descriptions
rather than pseudocode"* for how to write the intrinsic cost, and that encoding
prosociality amounts to *"a few sentence fragments."* H-JEPA hierarchical
planning, the configurator, and the Mode-1/Mode-2 distillation were all
unimplemented.

#### 3.4.2 What JEPA demonstrated, and what it cost

| model | demonstrated | compute |
|---|---|---|
| **I-JEPA** arXiv:2301.08243 (CVPR 2023) [V] | ImageNet-1k linear probe: ViT-B/16 **72.9 %**, ViT-L/16 77.5 %, ViT-H/14 79.3 %, ViT-H/16₄₄₈ **81.1 %**; 1 %-label semi-supervised 77.3 % | *"ViT-Huge/14 on ImageNet using **16 A100 GPUs in under 72 hours**"* ⇒ **≈1,150 A100-hours** |
| **V-JEPA** arXiv:2404.08471 [V] | VideoMix2M (~2M videos), frozen attentive probe ViT-H/16: **K400 81.9 / SSv2 72.2 / IN1k 77.9**; ~**6× cheaper** than equivalent reconstruction video models | 90,000 iterations at batch 3,072 clips |
| **V-JEPA 2** arXiv:2506.09985 (2025) [V] | >1M h video, ViT-g **~1.2B params**, 252K iters. K400 **87.3**, SSv2 **77.3**, Diving-48 90.2. **V-JEPA 2-AC**: a ~300M-param block-causal transformer post-trained on **<62 h of unlabeled Droid video**, deployed **zero-shot on two Franka arms in two different labs** — pick-and-place **80 % (cup) / 65 % (box)**, planning **16 s per action**, 16× faster than Cosmos | full-resolution training would be *"roughly 60 GPU-years"*; progressive resolution gives 8.4× ⇒ the actual run is **≈7 GPU-years ≈ 2,500 GPU-days** |

**Feasibility on our hardware, stated bluntly** [C, from the sweep's
conversion: T4 ≈ 1/7 of an A100 effective, P100 ≈ 1/18 — no tensor cores]:

| run | ≈ P100-hours | at 30 h/week |
|---|---|---|
| I-JEPA ViT-H/14 | ~20,700 | **13 years** |
| **I-JEPA ViT-B/16, the smallest published run** | ~1,900 | **~1.2 years** |
| V-JEPA 2 ViT-g | ~1.1M | **never** |

> **No published JEPA pretraining run is reproducible on free-tier hardware —
> not even the smallest one.** Anyone who claims otherwise is scaling something
> else and calling it JEPA. Candidate 4, *as published*, is not available to
> this project and no amount of cleverness changes that.

#### 3.4.3 But the *idea* is available, and one paper proves it at our scale

**PLDM** — Sobal, Zhang, Cho, Balestriero, Rudner & LeCun, arXiv:2502.14819
(2025), *"Learning from Reward-Free Offline Data: A Case for Planning with
Latent Dynamics Models"* [V]. **This is the single most relevant paper in the
entire survey for a free-compute JEPA-style project**, and it comes from
LeCun's own group:

- 6 methods × **23 datasets** across navigation environments, 6 generalisation
  axes. PLDM wins on all of them — data quality, data efficiency,
  generalisation to new layouts, tasks beyond goal-reaching.
- **Model sizes: encoder 1.43M params (Impala-Small) or 33K (conv); predictor a
  2-layer GRU, 794K params — or 20.4K for PointMaze.**
- Objective: **JEPA latent prediction + VICReg-style collapse prevention +
  inverse dynamics + temporal smoothness**.
- **~80 % success from "a few thousand transitions."** Where goal-conditioned
  baselines (GCIQL, HIQL, CRL, GCBC, HILP) fail outright on short trajectories,
  PLDM holds.
- Cost: **order 1–10 GPU-hours.** Squarely feasible on a P100.

Note the anti-collapse choice: PLDM uses **VICReg-style explicit regularisation
rather than EMA**. That is LeCun's own group deciding that at small scale, the
explicit variance floor is the safer bet. §5's A4 arm follows it.

#### 3.4.4 Collapse is the silent failure, and it has named detectors

This is the (d) column for the whole JEPA family, and it is the reason A4
carries a mandatory diagnostic rather than a loss curve.

**What the JEPA line actually uses:** V-JEPA 2 states it plainly — *"the loss
uses a **stop-gradient** operation and an **exponential moving average** of the
weights of the encoder to prevent representation collapse."* I-JEPA adds
architectural asymmetry and EMA momentum 0.996 → 1.0. **There is no explicit
variance or covariance term anywhere in I-JEPA / V-JEPA / V-JEPA 2.** The loss
is minimised by collapse; only the optimisation dynamics prevent it.

**Why that is exactly our most dangerous bug class.** SimSiam (arXiv:2011.10566
[V]) is explicit: *"collapsing solutions do exist for the loss and structure,
but a stop-gradient operation plays an essential role in preventing
collapsing"*, and *"the collapse can be observed by the minimum possible loss
and the constant outputs."* A collapsed JEPA has a **beautiful, monotonically
falling loss**. `LESSONS.md`: *"A loss curve is not learning."* Here it is
worse than not-learning — it is the signature of the failure.

**Partial (dimensional) collapse** is subtler still: Jing, Vincent, LeCun &
Tian, arXiv:2110.09348 (ICLR 2022) [V] — embeddings *"end up spanning a
lower-dimensional subspace instead of the entire available embedding space"*,
diagnosed by the log-scale eigenvalue spectrum of the embedding covariance.

**Detectors, in the order we should adopt them:**

1. **Per-dimension std of L2-normalised embeddings.** Healthy ≈ `1/√d`;
   collapsed → 0. Costs nothing; log every step.
2. **RankMe**, arXiv:2210.02885 [V]:
   `RankMe(Z) = exp(−Σ_k p_k log p_k)`, `p_k = σ_k(Z)/‖σ(Z)‖₁ + ε` — the
   exponentiated Shannon entropy of the normalised singular spectrum, i.e. a
   smooth effective rank. **Label-free, no hyperparameters, no training**, and
   validated as a hyperparameter-selection criterion with *"nearly no reduction
   in final performance"* versus label-based selection.
3. **LiDAR**, arXiv:2312.04000 [c] — rank of the LDA matrix rather than the raw
   covariance, which closes RankMe's blind spot (high-rank noise).

**And one theory paper that should be read before any latent-only world model
ships here:** Tang et al., *Understanding Self-Predictive Learning for RL*,
arXiv:2212.03319 [V] — *"a faster paced optimization of the predictor and
semi-gradient updates on the representation are crucial to preventing
representation collapse."* The **relative learning rates of predictor versus
encoder are load-bearing, not incidental.** That is a hyperparameter with a
silent failure attached, and B2 counts it.

#### 3.4.5 Latent prediction vs reconstruction — the evidence both ways

**For latent prediction:**
- **TD-MPC2 (arXiv:2310.16828) beats DreamerV3 across all 104 continuous-control
  tasks**, with no decoder at all and one hyperparameter set [V]. This is the
  strongest direct evidence that dropping reconstruction costs nothing on
  control.
- **Denoised MDPs** arXiv:2206.15477 [V] and **TIA** arXiv:2106.15612 [V]:
  reconstruction-based world models cannot distinguish task-relevant features
  from visual distraction; both beat reconstruction on distractor benchmarks.
- **The sharpest single datapoint:** arXiv:2502.11831 [V] — on intuitive-physics
  violation-of-expectation, V-JEPA reaches **98 % on IntPhys** while
  **VideoMAEv2, a pixel-reconstruction video model, is at chance**, and
  Qwen2-VL-7B / Gemini 1.5 Pro are *"only marginally above randomly-initialized"*.
  Same data regime, different objective.

**For reconstruction:**
- **It does not silently collapse.** There is no collapsing solution to "predict
  the observation". Every failure mode above belongs to the latent-only family.
- **The decoder doubles as a diagnostic and as an intrinsic reward.** Curious
  Replay (arXiv:2306.15934, ICML 2023) [V] uses DreamerV3's own reconstruction
  loss as a replay priority and gets **Crafter 19.4 vs 14.5 previous best** —
  a functional benefit latent-only models forfeit.
- **Tang et al. (2212.03319)** again: latent-only needs the predictor optimised
  faster than the encoder and needs semi-gradient updates, or it collapses.
  Reconstruction has no such requirement.

**Honest summary, and it is the sweep's own wording:** latent prediction wins on
compute efficiency, distractor robustness, and head-to-head control benchmarks;
reconstruction wins on *not silently failing*. **No paper was found that shows
reconstruction beating latent prediction at small scale specifically** — that
claim would be an extrapolation from the collapse literature, not a citation.
It is therefore a hypothesis, and §5's A2-vs-A4 contrast is the experiment.

### 3.5 Intrinsic-motivation stacks (learning progress)

`CURIOSITY_BAKEOFF.md` §1 already surveys this family at length and selects
**`lp`** — absolute learning progress over an auto-partitioned outcome space —
as the leading candidate. This sweep adds five things that bear on the
*learning core* rather than on the curiosity signal itself.

**1. The best modern LP result, with its exact estimator.** Kanitscheider et
al., arXiv:2106.14876 (OpenAI, Minecraft) [V]:

```
fast/slow EMAs of success probability, EMA timescale 1,250 optimisation steps
bidirectional LP = | f(p_fast) − f(p_slow) |          <- ABSOLUTE
unidirectional LP =  max(0, f(p_fast) − f(p_slow))
reweighting        f(p) = (1−p_θ)·p / [ p + p_θ·(1−2p) ] ,  p_θ = 0.1
sampling: z-score the LP measures -> sigmoid centred at the 90th percentile
          of the standard normal -> normalise  (≈90 % of samples to the top 20 %)
```

Items discovered above 5 % success, out of 107 obtainable:

| condition | items |
|---|---|
| uniform sampling, no bonus | 17 |
| fixed exploration bonus | 43 |
| dynamic exploration bonus | 70 |
| **unidirectional** LP curriculum | 79 — **with forgetting cycles** |
| **bidirectional (absolute)** LP curriculum | **82** |

That 82-vs-79, with the note that signed LP produces forgetting cycles, is the
cleanest empirical justification anywhere for **absolute** LP, and it matches
CURIOUS's (arXiv:1810.06284) reason for the same choice. **Cost: 21 days on 32
GPUs.** Not reproducible; the estimator is.

**2. The corrective nobody cites.** Taiga et al., arXiv:1908.02388 (ICLR 2020)
[V]: bonus-based exploration methods bolted onto Rainbow *"do not provide
significantly improved performance on Montezuma's Revenge or hard exploration
games"* and *"may negatively impact performance on games in which exploration
is not an issue and may even perform worse than ε-greedy."* **Any claim that an
intrinsic bonus is a general win has to answer this paper.** It is the direct
reason §5's A1 (`ppo-lp`) is a *candidate to be beaten by A0*, not a favourite.

**3. RND's noisy-TV immunity is real and principled, not folklore** — and it is
narrower than usually stated. The paper's own taxonomy of prediction-error
sources is (1) insufficient data (epistemic, desirable), (2) stochasticity,
(3) model misspecification, (4) learning dynamics; RND avoids (2) and (3)
*by construction* because the target is deterministic and inside the
predictor's model class [V, arXiv:1810.12894]. But: RND is a decaying novelty
bonus, so it is **not** immune to **detachment**; it scores **−3 on Pitfall**;
and it needs **observation normalisation** and **non-episodic intrinsic
returns** to work at all — two more silent-default landmines. Scale: **1.97
billion frames on 128 parallel environments.**

**4. Disagreement has the fewest documented failure modes**, on this sweep's
reading [V, arXiv:1906.04161]: immune to aleatoric noise *by construction*
(ensemble members converge to the same conditional mean, so variance → 0 while
error stays high), **empirically verified in the Unity noisy-TV**, and the only
member of the family with a **real-robot** result — a 7-DoF Sawyer from raw
RGBD, **67 % interaction rate with unseen objects vs 17 % random**, in *"less
than 1000 examples"*. This matters for §5 because **the EFE epistemic term's
standard estimator is this** (§3.3.3): choosing the disagreement estimator is
choosing the well-evidenced instance of the active-inference epistemic term.

**5. The gap.** *"There is no Dreamer + learning-progress paper."* The sweep
looked and did not find one. The neighbours are Plan2Explore (ensemble
disagreement in latent space, **not** LP) and Curious Replay (prediction error
as replay priority, **not** its derivative). Kanitscheider's LP is model-free.
So **world model + absolute learning progress over a goal space is unoccupied
territory** — which is either an opportunity or a warning that someone tried it
and it did not work. The literature does not say which, and this document
declines to guess: §5 tests LP on the PPO core, where it has evidence, and does
not add an LP-on-Dreamer arm that nothing supports.

**Compute reality for this whole family:** every headline result here is 10⁸–10¹⁰
frames — RND 1.97B frames on 128 envs, NGU/Agent57 distributed R2D2-scale,
Go-Explore enormous, Kanitscheider 32 GPUs × 21 days. **None is reproducible at
30 h/week.** What transfers is the *estimator*, not the run, and that is what
`CURIOSITY_BAKEOFF.md` §6 already exploits by running the whole Ladder Test
programme on ~8 CPU-core-hours.

### 3.6 The comparison table — the five questions, answered

| | (a) multimodal in one latent | (b) sparse needs-driven reward | (c) sample efficiency at 30 h/week | (d) simplicity / silent failures | (e) subsumes what we built |
|---|---|---|---|---|---|
| **PPO** | **No.** Concatenation only; the "unification" is a `torch.cat`. A sense that is missing at some timesteps has no principled handling | **Weak but proven.** Yoshida 2024 (PNAS Nexus) does exactly this with PPO on an 8-joint MuJoCo body | **Poor per sample, excellent per second.** 0.84 core-s/1k decisions [M] — the cheapest learner here by ~9× | **Worst.** 37–41 implementation details [V]; 20 knobs in our config [M]; two of this project's most expensive bugs are on those lists | Everything. It is what `TrainingPipeline.py` is |
| **DreamerV3-class** | **Yes, natively.** Dict observations into one RSSM latent; predicting all senses jointly **is** the objective `UNIFIED_BRAIN_BAKEOFF.md` §1.2 chose | **Yes** — Crafter and Minecraft-diamond are the demonstrations, both sparse and open-ended | **Best per sample; cost is the replay ratio, and that is the open question** (§3.0) | **Middle, and the claim is demonstrated:** 150+ tasks, one configuration [V]. But symlog / twohot / free-bits / unimix / percentile-return-norm are five silent-default landmines | Subsumes the dead 2.97M `WorldModel` [M], the binding objective, and gives `EpisodicMemory` a latent to index |
| **TD-MPC2 / JEPA-family** | **Yes**, same argument, minus the decoder | **Yes** (TD-MPC2 across 104 tasks) | **PLDM proves 1–10 GPU-h at 1–2M params** [V]; published JEPA pretraining is 1.2–126 P100-**years** [C] | **Worst silent failure in the survey: collapse with a falling loss.** Mandatory RankMe + per-dim-std monitoring, and predictor/encoder LR ratio is load-bearing | Same as Dreamer, and the existing `WorldModel.target_encoder` + `update_target_encoder` is already this shape [M] |
| **Active inference** | Inherits whatever model it wraps — **no independent answer** | **Yes in principle** — this is its best feature; `ln C` over interoceptive observations *is* the setpoint | **Unknown; no standard-benchmark entry in 9 years** (§3.3.2). AXIOM plans at 252–534 ms/step | **Strict superset of Dreamer** plus 6 named silent failures (§3.3.5) | Nothing extra; the epistemic term is Plan2Explore's estimator |
| **Intrinsic motivation (LP)** | N/A — it is a term, not a core | **Complements** needs; LP is a selector that requires first successes, which needs supply | **The estimator is nearly free** (~2.0 core-s/1k [c]); the published runs are not | Good: absolute LP is noise-robust by construction. Failure modes are known and specific (estimator noise, window sensitivity, LP = 0 where competence = 0) | Instantiates `UnifiedBrain.AutotelicGoalGenerator`, which has never received a gradient |

**One row is doing more work than the rest.** Under (a), only the world-model
family answers `GOAL.md`'s demand *structurally* — with a learning rule whose
objective is "predict every sense from every other sense". For PPO, "all senses
in one brain" is a concatenation and a hope. That is not a small difference in
a project whose north star is unison.

### 3.7 The unison question, candidate by candidate — HOW does each sense enter one representation?

The owner, 2026-08-09, now constitutional in `SYSTEM.md` under Hard
constraints: whatever we take from DreamerV3 or anywhere else, *"we will never
forget how we learn GENERALLY and ALL senses combined."* A core that wins the
task but fails binding *"has not won; it has changed the subject."*

So this is not a column in a table. It is an **admission criterion** (§5.0b),
and it requires each candidate to answer a mechanical question: **by what
gradient does information from modality A end up shaping the representation
that modality B is read through?** Vague answers are disqualifying, so each
answer below names the loss term.

Modality set, fixed: **vision · audio · touch · proprioception · need-state ·
language.**

---

**PPO (A0, A1) — HONEST ANSWER: the senses do not meet, except through the
reward.**

The observation is `concat(v, a, t, p, n, l)` into a shared first layer. Two
consequences, and both are mechanical facts, not opinions:

1. The only gradient reaching the vision weights is
   `∂(policy loss + value loss)/∂v`. There is **no term in which audio is a
   target for vision, or vice versa.** "What he hears teaches what he sees" has
   no referent in the update.
2. Worse, the reward is *low-bandwidth*. A scalar per step cannot carry enough
   information to shape a rich joint representation; this is the standard
   argument for auxiliary objectives, and it is the same argument
   `UNIFIED_BRAIN_BAKEOFF.md` §1.1 makes when it concludes that
   architecture is *necessary but not sufficient* and the binding **force** is
   the objective.
3. `π0.5`'s measured pathology is the warning: 99.3 % linear-probe accuracy on
   the language prompt with behaviour **completely invariant** to that prompt
   (`MULTIMODAL_BINDING.md`). A shared trunk can *encode* every sense and use
   none of them. That is exactly what a PPO arm would look like if it failed
   unison, and the task metric would not notice.

**Therefore PPO is admissible only with a declared auxiliary binding
objective**, and the honest thing is to name it now rather than let the arm in
on a promise. The auxiliary is `UNIFIED_BRAIN_BAKEOFF.md` §1.2's, unchanged:

```
L_total = L_PPO  +  λ_bind · L_masked_cross_modal
L_masked_cross_modal:  mask an entire modality's tokens at random (modality
  dropout, §1.4), predict the masked modality's next-step features from the
  surviving modalities through the shared trunk.
```

Note precisely what this costs and what it means:
- **It adds a hyperparameter (`λ_bind`) and a second loss**, so PPO's
  simplicity advantage shrinks by exactly the amount unison costs it. That is
  the point of stating it: PPO's apparent simplicity was partly the cost of
  unison, unpaid.
- **`L_masked_cross_modal` is a world-model objective without the action
  conditioning.** So the honest statement is that *admitting PPO to a
  unison-constrained bakeoff requires giving it half a world model.* This is
  the strongest single argument in §4.2, and it is arrived at from the
  constraint rather than from preference.

---

**World-model arms (A2 `dreamer-xs`, A4 `wm-latent`) — the senses meet in the
objective, by construction.**

The RSSM maintains one latent state `s_t = (h_t, z_t)` and the training loss is

```
L = Σ_m  w_m · L_pred( decode_m(s_t) , o_t^m )     for m ∈ {v,a,t,p,n,l}
    + KL[ posterior(z_t | h_t, enc(o_t)) ‖ prior(z_t | h_t) ]
    + reward head + continue head
```

Three mechanical facts follow, and they are why this is the strongest pro-world-
model argument available:

1. **Every modality is a prediction target for the shared latent.** The
   gradient `∂L_pred(decode_audio(s_t), o^a_t)/∂enc_vision` is *nonzero and
   structural*: the only way the latent can predict audio cheaply is to carry
   the information that vision provides about audio. **Cross-modal prediction
   IS the training objective** — it is not an auxiliary bolted on, it is the
   whole loss.
2. **The posterior/prior KL is the binding pressure, stated exactly.** The prior
   `p(z_t | h_t)` must predict the posterior `q(z_t | h_t, enc(o_t))` *before
   seeing this step's observations*. Whatever one modality lets you anticipate
   about another reduces that KL. A model that keeps six unfused streams pays
   the full KL on all six.
3. **Missing modalities are handled natively.** Drop audio at some timesteps and
   the posterior simply reverts toward the prior — which is the correct
   behaviour and is what `UNIFIED_BRAIN_BAKEOFF.md` §1.4 has to enforce by hand
   as "modality dropout" in an architecture that does not have it.

`A4 (wm-latent)` gets the same argument with one caveat that must be stated:
with the decoder deleted, per-modality reconstruction targets go away and
binding is enforced only through the *latent* prediction and the inverse-
dynamics/VICReg terms. **A latent-only model can bind less than a
reconstructing one, and it can collapse** (§3.4.4). A4 therefore carries both
the collapse diagnostics *and* the unison gates, and if it wins the task while
losing UB.11's ablation matrix, it is not adopted.

---

**Active inference (A3 `wm-efe`) — inherits A2's model, therefore inherits its
unison, and adds nothing of its own.**

This is worth stating plainly because it is easy to credit active inference with
a unison story it does not have. EFE is an objective over *policies*, evaluated
under whatever generative model you supply. It has no opinion about how senses
enter that model. A3 is admissible **only because it uses A2's world model
verbatim** — which is also why A2-vs-A3 is a clean test of the actor objective.

---

**Learning progress (A1's bonus) — a selector over goals, not a representation.**

LP is computed over an outcome space, not over sensory features; it contributes
no binding gradient at all. It is admissible as a *term* on an admissible core,
never as a core.

---

**The summary that the admission criterion turns on:**

| candidate | binds by | admissible as-is? |
|---|---|---|
| PPO | nothing — only the scalar reward | **No.** Admissible only with `L_masked_cross_modal` added, which is a world-model objective minus action-conditioning |
| DreamerV3-class | its own per-modality prediction loss + posterior/prior KL | **Yes, structurally** |
| TD-MPC2 / JEPA-class | latent prediction + inverse dynamics + VICReg; weaker, and collapse-prone | **Yes, conditionally** — with rank diagnostics and the unison gates |
| Active inference | inherits its model's | **Only as an actor on an admissible model** |
| Learning progress | nothing | **Only as a term** |

---

## 4. The recommendation

### 4.1 The first principle, in one paragraph — the answer to the owner's question

> **Jack learns by predicting.** One small model takes every sense into one
> latent state and is trained on a single objective: *given what I am sensing
> and what I do next, predict what I will sense next — in every modality at
> once.* Three things then attach to that one model, and none of them is a
> separate learning system:
>
> - **What he wants** is a *preference over his own interoceptive
>   predictions*: fed, warm, unhurt, not alone. He acts to make the futures he
>   predicts look like that. (Keramati & Gutkin's drive reduction; active
>   inference's `ln C`.)
> - **What he is curious about** is *where his model is still wrong in a way
>   that is fixable*: he seeks the futures his own model most disagrees with
>   itself about. (Plan2Explore's ensemble disagreement; active inference's
>   epistemic term — the same quantity, §3.3.3.)
> - **What he remembers** is *what actually happened*, in a diary he can read
>   back, separate from the model that learned from it. (ME.9/ME.10, already
>   proven.)
>
> Acting is choosing, inside the predicted world, the thing that best serves
> those two: **make it turn out how I need it, and find out what I don't yet
> know.**

That is the whole thing. It is four moving parts — a predictor, a preference, a
disagreement, and a diary — and it is the only formulation in this survey where
"one brain, all senses in unison" is *the learning rule itself* rather than a
wiring diagram bolted onto one.

### 4.2 Why this, laid bare

**Reason 1 — it makes the project smaller, not bigger.** This is the argument
the owner's question is really asking about, and it is easy to get backwards.
PPO is the simpler *algorithm*. It is the more complex *system*, because
everything else has to be bolted on separately:

| job | under PPO | under a predictive world model |
|---|---|---|
| learn a policy | PPO (37–41 implementation details [V]) | actor-critic in imagination |
| fuse the senses | `torch.cat`, plus `UNIFIED_BRAIN_BAKEOFF.md`'s separate masked-cross-modal binding heads (UB.10) with their own loss and their own schedule | **the model's own objective.** Predicting every sense from every other sense *is* cross-modal masked prediction |
| learn a representation | a separate pretraining stage (D1's A4 stage 1, masked-motion on 2,747 mocap clips) | **the same objective again** |
| be curious | a separate module: ICM / RND / LP, each with its own network, its own reward scale, its own failure modes | **the model's own ensemble disagreement**, free |
| plan / imagine | the 2,974,977-parameter `WorldModel` that is currently never constructed [M] | it *is* the core |
| handle a missing sense | undefined — a zero-filled slot | the posterior is simply less certain; this is what a latent-variable model is for |

Five separate mechanisms collapse into one objective. **That is what "just a
system that can learn and get input from every single sense" looks like when
you write it down**, and it is why the recommendation is not "the fancier
option" — it is the option with fewer things in it.

**Reason 2 — it is the only candidate that answers question (a) structurally.**
`GOAL.md` demands *"what he hears can teach what he sees"*. Under PPO that
sentence has no referent: nothing in the update makes audio inform vision.
Under a joint predictive model it is the definition of the loss.

**Reason 3 — the cost objection is weaker than it looks, and both we and the
literature measured it.** §3.0: at train_ratio 1 a Dreamer-shaped core costs
**19.6 core-s per 1,000 decisions against PPO's 13.1** — 1.5×, because on our
hardware **MuJoCo is the bottleneck, not the network** [M]. The replay ratio is
a dial, not a property of the method, and **Director settles that it can be
turned very low: one gradient step per sixteen policy steps — train_ratio ≈
0.06 — under 24 hours on a single V100, solving egocentric-camera mazes no
baseline solved** (arXiv:2206.04114 [V], §3.2.13). At that ratio the world
model is *cheaper than our PPO arm's own update*. `LC.02` measures where the
dial can sit for us.

**Reason 3b — and note what Reason 3 is NOT.** It is not a sample-efficiency
argument. §3.2.3 removed that argument: at Crafter's 1M steps a tuned 4M PPO
scores **15.60 %** against DreamerV3-201M's **14.77 %** (arXiv:2307.03486 [V]).
On a survival-flavoured 2D-ish task at our budget, **the world model buys no
measured sample efficiency at all.** This document's recommendation
deliberately does not rest on the claim the literature does not support.

**Reason 4 — it subsumes work already done and already verified.** The
`WorldModel` class with its EMA target encoder is already in the repo
(2,974,977 params, never constructed [M]). T1.12's flow-matching action head is
a *decoder from a latent to an action* — exactly what an actor in imagination
needs, and it is already measured to work (sampler error 1.065 → 0.00134,
conditioning ratio 1578 [L]). `EpisodicMemory` becomes the store the latent
indexes into. Nothing verified is thrown away.

**Reason 5 — active inference is taken as vocabulary, not as algorithm, and
that is the honest reading of §3.3.** The unification the brief points at is
real *as a way of thinking*: preferences over interoceptive observations are
setpoints, and curiosity is information gain. But (i) the balance between them
is a hyperparameter in nats, not something the equation removes; (ii) no deep
active-inference agent has entered a standard benchmark in nine years; (iii) it
is a strict superset of DreamerV3 with six named silent failure modes. **We
take the mapping and leave the machinery.** §5's A3 arm exists so that this
judgement can be falsified rather than asserted.

### 4.3 The strongest counterargument, stated fairly

It is strong, and a reader who ends up disagreeing with §4.1 should disagree
for these reasons:

**(1) This is exactly the shape of the mistake that produced the 57M trunk.**
An architecture chosen from the literature, on elegance and on what it would
make possible, adopted before a single clean measurement on this project's own
task. The recommended core is **16× the parameters** of the PPO arm and has
never been run here. `LESSONS.md` and `GOAL.md` both record the outcome last
time: *"complexity must earn its place or lose it."* Every measurement this
project has ever produced favours the small simple thing — **530.2 from 124,707
parameters** against **317.7 from 57M** [L].

**(2) The one published existence proof for Jack's exact problem uses PPO.**
Yoshida et al., *PNAS Nexus* 2024: deep homeostatic RL, multiple needs, real
physics, an 8-joint MuJoCo body — the same joint count as our climber-rover —
reproducing three nutritional-geometry curves from animal data. Learning core:
**PPO with 4-layer MLPs** (§3.3.4). There is no comparable Dreamer result on a
homeostatic multi-need body. Recommending the world model means recommending
the option *without* the existence proof.

**(3) On the closest published analogue of W0, the world model has *already
lost* — and to the arm we are calling the incumbent.** This is now the
strongest single objection, and it is a measurement, not a worry.
Crafter at the standard 1M steps, ten seeds (arXiv:2307.03486, NeurIPS 2023
[V]): **tuned 4M-param PPO 15.60 % vs DreamerV3-201M 14.77 %**, and
**PPO + one contrastive auxiliary loss 21.79 %** — 47 % relative *above*
DreamerV3, at 4.5 % of the parameters. Crafter is a survival game with hunger,
thirst, health and a tech tree; it is the nearest thing in the literature to
W0. The world model's whole reputation rests on Atari100k (400k frames),
DMLab and Minecraft — regimes W0 is not in. And the cost objection survives in
weakened form: **at train_ratio 512 the core costs 1.05 core-hours per 1,000
decisions on this box** [C], so the affordable configuration is not the
published one, and five extra silent-default landmines (symlog, twohot, free
bits, unimix, percentile return normalisation) come along regardless.

**(4) The arguments that motivate world models are largely arguments about
pixels.** Distractor robustness, wasted reconstruction capacity, intuitive
physics from video — all of it is high-dimensional visual input. W0's
observation is ~96-dimensional, low-noise, and every dimension matters. The
regime where reconstruction wastes capacity may simply not be our regime, and
§3.4.5 found **no paper showing reconstruction beating latent prediction at
small scale** — but equally none showing the reverse *at 96 dimensions*.

**(5) Our sub-XS size is below anything published.** The benchmarked shape is
**1.9M parameters** against DreamerV3's smallest published class, **XS at 8M**
[V]. The recommended arm is therefore a size class nobody has reported results
for, and "the world model won't fit in our budget" and "the world model doesn't
work" would produce the same observation.

**(6) Nobody has ever done this, and that cuts both ways.** §3.2.14: **no
published work applies a world model to an explicit homeostatic drive
function**, and **no HRRL + world-model paper exists**. Unclaimed territory has
no baseline to beat and no prior art to inherit — which is exactly the
condition under which a project convinces itself it has succeeded. Meanwhile
the one thing the world-model family *has* been pointed at in this direction —
Plan2Explore's pure novelty — scores **2.1 % on Crafter against random's
1.6 %** [V].

**The fair summary of the counterargument:** *the recommendation is a bet that a
structural advantage on unison will outweigh a demonstrated disadvantage on the
nearest published analogue of our own task, at a scale nobody has published, in
a combination nobody has tried.* That is a real bet. It is why §4.4 commits to
building the incumbent first and why §5 is designed so the bet can lose cheaply.

### 4.4 What the recommendation actually commits us to

Not to shipping a world model. To **four things**, in this order:

1. **Adopt the principle as the design frame** (§4.1). It costs nothing and it
   already resolves a tension: `UNIFIED_BRAIN_BAKEOFF.md`'s binding objective
   and this document's learning core are **the same objective**, so UB.10 and
   the learning core stop being two workstreams competing for the same
   CPU-hours.
2. **Build the incumbent first and properly** — and note what the constitutional
   unison constraint has just done to it. §3.7 shows PPO binds nothing except
   through a scalar reward, so an admissible PPO arm is
   `PPO + L_masked_cross_modal`. **That is a world-model objective with the
   action-conditioning removed.** So the honest statement of the incumbent is
   not "PPO"; it is *"PPO plus half a world model"*, and the remaining question
   is whether the other half — action conditioning, imagination, a value
   learned inside the model — pays for itself. **That is a much narrower and
   much more answerable question than "PPO vs Dreamer", and reframing it this
   way is the main thing §4 contributes.**
3. **Make the world model earn the rest of the way** in `LC.04`/`LC.05`, with
   `LC.02` deciding beforehand whether it is affordable. Director's
   train_ratio ≈ 0.06 (§3.2.13) says it very likely is; `LC.02` measures it
   rather than assuming it.
4. **Do not repeat the 4.6 %.** The PPO arms use the tuned configuration
   (§3.2.3 / invariant F9). A world-model win over a handicapped PPO would be
   the dropout confound with a new costume.

**Pre-registered, so this document can be wrong in public:** the recommendation
in §4.1 is **falsified** if `ppo-needs` or `ppo-lp` wins both scorings in
`LC.04`/`LC.05` at ≥ 1.5σ *while passing the unison gates*. In that event the
first principle becomes *"Jack learns by trying things and keeping what reduced
his needs, with a cross-modal prediction loss to hold the senses together"*,
§6's B1 soft target drops to 250K parameters, and the world-model line is
closed until pixels arrive (§6.4's first exception).

**And one outcome would falsify something bigger than this document:** if the
PPO arms win the task *and* fail the unison gates while a world-model arm
passes them, `SYSTEM.md`'s constitutional constraint decides it — the
world-model arm is adopted despite losing on the task metric, and the finding
recorded is *"the task metric and the unison requirement point different ways,
and W0's task is too easy to need unison."* That is a statement about the
**world**, and the response is a harder world (W1), not a weaker constraint.

---

## 5. The first-principles bakeoff

**Namespace `LC.xx`** — new, and checked. `LC.00`–`LC.06` each resolve to
exactly one module under `run.py::_module_for`'s glob, with no cross-collision
(`fnmatch` run on this box [M]; the `ME.11`/`ME.11.0` lesson says to verify the
naming scheme before writing a test, and the `UB.1`/`UB.16` lesson says to run
the check rather than reason about it). Two-digit ids throughout.

### 5.0 What the bakeoff needs from W0, and what it must not have

`SURVIVAL_WORLD.md` is specifying the W0→W3 fidelity ladder in parallel. This
document does not duplicate that; it states the **contract** the learning-core
bakeoff needs from W0, so the two can be reconciled by assertion rather than by
hope (the `T0.14` rule: assert contracts against the source of truth).

W0 must supply, and every arm sees the identical world:

| # | requirement | supplied by | why the bakeoff needs it |
|---|---|---|---|
| **W0-1** | **Needs.** At least two interoceptive scalars that deplete and can be restored by acting on the world, plus one that is a pure nuisance. | `PURPOSE_AND_SCAFFOLDING.md` §2.2: `h = (e, i, w)` — energy, integrity, wetness | (b) sparse needs-driven reward is the question; a world with a hand-shaped task reward cannot ask it |
| **W0-2** | **Death**, and a respawn that is **not a free teleport to a good state**: on death the body reappears at a *uniformly random* legal spawn, not at the ladder base or any previously useful location. | new in W0; W0-2 is the one thing `PURPOSE_AND_SCAFFOLDING.md` deliberately does *not* have (§2.2: "Nothing terminates") | The owner requires it (`GOAL.md`: "He lives, he dies, he remembers"). The random respawn is what keeps `LT` §2.1's objection — an episode boundary is an experimenter-supplied curriculum — from applying. |
| **W0-3** | **Cross-life memory.** The diary (`EpisodicMemory`) and the weights persist across death; a life index is recorded on every row. | `EpisodicMemory.py`, `Persistence.py`; substrate already proven by **ME.10** [L] (wipe the diary → skill survives at 0.944; revert the weights → recall survives at 1.000) | The claim "life N+1 is better than life N *because of* what life N recorded" is not measurable without it |
| **W0-4** | **Multimodal observation.** At minimum proprioception + touch + the drive vector + one exteroceptive channel (the `_Retina` rays PG.4 already uses). | `playground.py`, `pg_4_noisy_tv.py` | question (a). An arm that cannot take a dict observation fails here, and that failure is a *result*, not a bug |
| **W0-5** | **The noise panel**, unchanged, with its `R_RESOLVE = 2.5 m` acuity falloff. | `playground.py`; PG.4 certified the trap fires [L] (`icm_dwell_share` 0.667 vs null 0.061) | Every learning core with an exploration term must report `panel_dwell`; a core that buys its survival score by staring at noise has not learned to survive |
| **W0-6** | **Zero task reward.** `env_reward_absmax == 0.0` asserted every step, as in `LT`. | assertion | otherwise the bakeoff measures reward-following |

**What W0 must NOT have, and each is a named attack this bakeoff would
otherwise be open to:**

- **No 17-DoF humanoid, yet.** T2.01 FAIL / T2.02 VOID [L]: he cannot walk. The
  body is `CURIOSITY_BAKEOFF.md` §2.3's **climber-rover, 8 actuated DoF**, whose
  parameters are inherited from PG.3 by construction. Reason given three times
  independently in that document (Qflex's O(1/|A|) exploration variance; RGSD's
  69-DoF collapse; and the plain fact that a negative result on a body that
  cannot locomote tells you nothing about the *learning rule*). **Testing a
  learning core on a body that cannot act is the T1.02 mistake** — when the
  simplest possible learner also fails, the task is broken.
- **No pixels.** W0's exteroception is the ray retina. Pixels are W2, and they
  would make three of the four arms unaffordable while changing nothing about
  the question. This is declared as a limitation in §8, not hidden.
- **No hand-tuned food placement that makes one arm's job easier.** The world is
  per-seed mutated by `PlaygroundParams.mutate()`, identically across arms.

### 5.0b ADMISSION — two criteria evaluated BEFORE any arm is scored

`SYSTEM.md`, Hard constraints, first entry (owner, 2026-08-09):

> *"No learning core without unison. Any candidate learning core — however it
> scores on task metrics — is INADMISSIBLE unless it accepts every modality
> into one shared representation, and its adoption is VOID until the standing
> unison gates pass under it. A core that wins the task but fails binding has
> not won; it has changed the subject."*

This is **constitutional**, which means it is not a term in the objective and
no margin can trade it away. Mechanically it becomes an **admission stage**
that runs before `run_bakeoff` is called at all, plus an **adoption stage**
that runs after. Both are pre-registered here.

---

#### ADMISSION-1 — UNISON. Can this core take every modality into one latent?

Evaluated by `LC.01`, on CPU, before any learning run. An arm is **excluded
from the bakeoff** — not scored and lost, *excluded* — unless all four hold:

| # | requirement | how it is checked |
|---|---|---|
| **U1** | The core ingests the **full modality token set** — vision, audio, touch, proprioception, need-state, language — into **one** shared representation. Not one encoder per sense feeding separate heads. | shape/route audit: assert every modality key reaches the shared state tensor; assert no modality has a private path to the action |
| **U2** | There is a **named loss term by which modality A's gradient reaches modality B's encoder.** For world models this is the per-modality prediction loss and the posterior/prior KL. For PPO it is `L_masked_cross_modal` and nothing else. "It's a shared trunk" is **not** an answer — `π0.5` encodes its language prompt at 99.3 % linear-probe accuracy while behaving invariantly to it (`MULTIMODAL_BINDING.md`). | the arm declares the term; `LC.01` asserts the gradient is nonzero by finite difference: perturb modality A's input, require a nonzero gradient at modality B's encoder |
| **U3** | **A missing modality is handled by the core, not by a zero-fill convention.** Modality dropout must be a supported input condition. | run the core with each modality dropped; require no shape error and a *changed* internal uncertainty |
| **U4** | **No modality may be silently down-weighted out of existence.** Directly from §3.2.6's landmine 4: with `loss_scales.rec` shared, a 64×64×3 image contributes 12,288 reconstruction terms and a 10-dim needs vector contributes 10 | assert per-modality loss contribution is within a declared band; the need-state modality must carry ≥ 1/|M| of the total reconstruction loss at init |

**U4 is not bureaucracy.** It is the concrete, already-documented mechanism by
which a "unified brain" would quietly delete the modality this whole project is
about, with every loss curve looking correct. It is the `nn.Dropout(p=0.1)`
shape exactly.

**Admission-1 outcomes, decided now:**

- `dreamer-xs`, `wm-latent`, `wm-efe` — **admissible** (§3.7), subject to U4.
- `ppo-needs`, `ppo-lp` — **admissible ONLY with `L_masked_cross_modal`
  attached.** Bare PPO is excluded. This is not a handicap invented to favour
  the world model; it is the constitutional constraint applied evenly, and its
  cost to the world-model arms is zero because they already satisfy it.
- **TD-MPC2 — INADMISSIBLE, and this is why it is not an arm** (§3.2.11): it is
  state-based proprioception only, no vision, by construction.
- A hypothetical "learning progress alone" core — inadmissible; LP is a term,
  not a representation.

---

#### ADMISSION-2 — THROUGHPUT. Can this core live a life at survivable wall-clock?

New, and it comes from a measurement that landed while this document was being
written: `DIRECTION_AUDIT.md` §4.1, measured on this box [c]:

| control path | env-steps/s | **sim-seconds per real second** | real time for a 1-sim-hour life |
|---|---:|---:|---:|
| 57M trunk (T0.07 ledger) | 11.48 | **0.17** | **5.8 hours** |
| ~160K MLP (measured) | 1,531.6 | **22.97** | **2.6 minutes** |
| zero action (physics ceiling) | 2,105.9 | 31.6 | 1.9 minutes |

**133×.** A 3-seed × 3-life × 1-sim-hour spec costs 23 minutes with the small
head and **52 hours** with the trunk — the latter exceeding `run.py`'s own
`cpu<2h × 3 seeds × 2` ceiling of 15 hours, on a box that also serves paying
tenants and whose builder loop is capped at 50 minutes per iteration.

> **A core that cannot run lives at survivable wall-clock is inadmissible for
> the survival world, regardless of its sample efficiency.** Sample efficiency
> is a claim about how many steps you need; throughput decides whether those
> steps can ever happen. `GOAL.md` requires lives, death, and cross-life
> learning; a core at 0.17 sim-s/real-s cannot produce a second life inside a
> builder iteration.

**The floor, declared before the run:**

> **T ≥ 5.0 simulated seconds per real second**, measured on **3 ARM cores of
> this box at `nice 19`**, with the learner in the loop — i.e. rollout *and*
> the amortised update, not the physics alone.

Derivation, so the number is not arbitrary [C]:

- climber-rover physics: ~81 decisions/s [c, `CURIOSITY_BAKEOFF.md` §6] at
  5 decisions per simulated second (`LT`'s pilot: 3,000 decisions = 600 sim-s)
  ⇒ **16.2 sim-s/real-s, physics-only ceiling** for this body;
- `ppo-needs` at 13.1 core-s/1,000 decisions [M] ⇒ 76 dec/s ⇒ **15.3**;
- `dreamer-xs` at train_ratio 1 (19.6 core-s/1,000) ⇒ 51 dec/s ⇒ **10.2**;
- at train_ratio 8 (70.9) ⇒ 14.1 dec/s ⇒ **2.8** — **fails the floor**;
- at train_ratio 64 (481) ⇒ **0.42** — fails by 12×.

So **T ≥ 5.0 admits train_ratio up to ≈ 4 and excludes ≥ 8**, and it admits
Director's 0.06 comfortably.

> **CORRECTED BY MEASUREMENT, 2026-08-09 — LC.02 PASS.** The derivation above is
> wrong by 16x and the error is in its denominator, not its arithmetic. The
> "~81 decisions/s" physics figure was measured with NO SENSES ATTACHED, and the
> core costs were measured with no physics attached, so the composition was
> never measured at all. Built as `experiments/w0.py` and timed on 3 ARM cores
> at nice 19, one decision costs ~20 ms of which `mj_step` is 9.4 ms and the six
> W0-4 senses (16-ray retina, 8-band binaural contact audio, 4-site touch, and
> the drive integrator's per-substep accumulation) are ~11 ms. Measured:
> **null (world + senses, no learner) 10.09 ± 0.96 sim-s/real-s**; every
> admissible arm clears 5.0, and the committed train_ratios are **0.25** for
> `ppo-needs`, `ppo-lp`, `dreamer-xs` and `wm-efe`, and **0.125** for
> `wm-latent` — not 4. The control behaved as pre-registered: the 36.92M
> `UnifiedBrain` trunk on the control path ran at 0.325 sim-s/real-s, 15.4x
> below the floor. Everything downstream that assumed a ratio near 4 —
> including the LC.04 envelope in §5.7 — must be re-costed against 0.25.
> Generalised in `LESSONS.md`, "A budget derived from a component measured
> alone is wrong by everything else". It also means the LC.04 envelope (§5.7) fits
inside `Budget.CPU_LONG`. The floor is set by what the ladder's own budget
ceiling permits, not by taste — and stating it as sim-seconds per real second
rather than as FLOPs is deliberate: it is the unit in which "Jack lives for an
hour" is a sentence with a price.

**`LC.02` measures T for every arm and fixes each arm's train_ratio to the
largest value that clears the floor.** Two anti-gaming provisions, because this
is a selection step and selection steps are where bakeoffs get corrupted:

1. **`LC.02` selects on WALL-CLOCK FIT ONLY. It may not look at `life_gain`.**
   Choosing an arm's hyperparameter by its score is tuning-on-the-metric; the
   spec's `_check` never reads the task metric, and the recorded value is
   committed before `LC.03` runs.
2. The chosen train_ratio is **reported for every arm** in the ledger and in
   `DECISIONS_RESOLVED.md`. If a world-model arm loses at train_ratio 4, the
   honest statement is *"it lost at the ratio our hardware allows"*, and that
   sentence must be visible next to the verdict — not discovered later.

---

#### ADOPTION — the winner is provisional until the unison gates pass under it

Sequencing, fixed now:

```
LC.01 admission (unison)      ─┐
LC.02 admission (throughput)  ─┴─►  LC.03 screening ─► LC.04/LC.05 arbitration
                                                              │
                                                    WINNER (PROVISIONAL)
                                                              │
                    ┌─────────────────────────────────────────┤
                    ▼                                         ▼
        UB.9  binding test ("Heard, Not Seen")     UB.11 ablation matrix
        re-run UNDER THE WINNING CORE              + PLACEBO modality
                    │                                         │
                    └──────────────► both PASS ◄──────────────┘
                                          │
                                    ADOPTION (final)
```

- A `LC.04`/`LC.05` winner is recorded as **PROVISIONAL** in the ledger and in
  `DECISIONS_RESOLVED.md`. It is not implemented as Jack's core, and the losing
  arms are **not deleted**, until UB.9 and UB.11 pass under it.
- **If the provisional winner fails UB.9 or UB.11, its adoption is VOID** (not
  FAIL — the task result stands; the *adoption* did not test binding). The
  runner-up is then promoted to provisional and must clear the same gates.
- **If no admissible arm passes the unison gates, there is no winner.** The
  correct output is a red ledger entry and an escalation, not the least-bad
  core. `SYSTEM.md` Law 4: never weaken a threshold to get an answer.
- The placebo modality is carried unchanged from UB.11 — an extra input channel
  of matched dimension and matched statistics carrying no information. **If
  ablating the placebo degrades performance as much as ablating a real sense,
  the ablation matrix is measuring capacity, not binding**, and both the gate
  and the arm are uninterpretable.

### 5.1 Fixed across all arms — these are repairs and invariants, not variables

Any of these varying between arms means the bakeoff measures it instead of the
learning rule. This table is the direct descendant of `D1_CONTROL_ARCHITECTURE.md`
§3's R1–R6 and of every `LESSONS.md` entry that cost GPU-hours.

| F | invariant | the lesson that forces it |
|---|---|---|
| **F1** | **`.eval()` during every rollout and every evaluation; `.train()` only inside the update.** Asserted by double-forward bit-identity at the *shipped* call site, including inside any GPU `JOB` string. | "Call .eval()" (42 % action drift) and T0.16 (103.6 % drift in the shipped kernel a fix had just missed) |
| **F2** | **Dropout is 0.0 in every learning core.** Not "disabled at eval" — absent. PPO's own sampling is the regularisation. | same; a silent default with no line of code to read is the most expensive bug class here |
| **F3** | **Identical observation vector, identical width, for every arm**, including the null. The drive channels are present for arms that ignore them. | `PURPOSE_AND_SCAFFOLDING.md` §2.4; "matched steps has more than one meaning" applied to matched *inputs* |
| **F4** | **Identical world seeds, identical spawn seeds, identical evaluation lives, paired across arms.** | D1 R6 (arXiv:2108.13264) |
| **F5** | **Every observation dimension is asserted against the live model**, never against a config constant. | T0.14: `mujoco_obs_dim = 376` padded 28 dead columns for the project's entire history |
| **F6** | **Every run records all four budgets** — env-steps (decisions), optimiser steps, wall-clock core-seconds, and gradient-FLOPs estimate — plus the **full decimated curve** (≤ 200 points spanning all lives, never `[:8]`). | "matched steps has more than one meaning" (6,240 vs 99,840 optimiser steps at 'matched' env-steps); T2.01 stored iterations 1–21 of 172 |
| **F7** | **Declared parameter counts are asserted against `sum(p.numel() for p in ... if p.requires_grad)` to ±5 %**, and a mismatch is `Status.VOID`. | "a default of zero is not unknown"; D1 §5's guard |
| **F8** | **The learning core is the ONLY thing that differs.** Same body, same world, same drive integrator, same reward form, same trunk-free network scale, same replay/persistence substrate. | this is Q2 (§2) and nothing else |
| **F9** | **THE PPO ARMS ARE THE TUNED KIND.** LayerNorm before every dense layer; value targets normalised by a running mean/std; separate policy and value networks; policy last layer initialised 100× small; tanh; initial action std ≈ 0.5; GAE λ = 0.9; observation normalisation always on; multiple passes per rollout. Any arm missing one is a spec ERROR, not a weak arm. | §3.2.3: these exact fixes take PPO on Crafter from **8.17 % to 15.60 %**, past DreamerV3's 14.77 % (arXiv:2307.03486 [V]). The published "PPO 4.6 %" is an under-engineered PPO, and comparing a world model to it is `LESSONS.md`'s dropout confound — *one arm had a handicap and the difference was attributed to architecture*. Plus arXiv:2006.05990 [V] |
| **F10** | **PER-MODALITY LOSS BALANCE IS ASSERTED, NOT DEFAULTED.** Every world-model arm reports the per-modality share of its reconstruction/prediction loss at init and at the end, and the **need-state modality must hold ≥ 1/\|M\| of it throughout**. Vector observations in [0,1] are **not** passed through `symlog` without an explicit rescale. | §3.2.6 landmines 3 and 4: with `loss_scales.rec` shared, a 64×64×3 image contributes 12,288 terms against the needs vector's 10, and `enc.simple.symlog` compresses a [0,1] need toward zero. Either would silently delete the modality the project is about, with a perfect-looking loss curve |
| **F11** | **No arm runs at a train_ratio that was not fixed by `LC.02` on wall-clock grounds**, and the value is recorded in the ledger for every arm. | §3.2.5: train ratio and model size are the two things DreamerV3 does **not** hold fixed, and they are exactly the two that decide affordability. Choosing them by score is tuning-on-the-metric |

**F8 has a consequence worth stating loudly.** The reward form is *fixed* here,
which means **this bakeoff does not decide Q1** — it uses whatever
`PURPOSE_AND_SCAFFOLDING.md` PS.04 selects, or, if PS.04 has not run, the
pre-declared default `homeo-dr` (`r = d(h) − d(h′)`) plus death. Every arm gets
the same reward channel. An arm that *reinterprets* that channel — active
inference converting it into a log-preference, learning progress adding an
intrinsic term — is doing so as part of its learning rule, and that is
precisely the variable under test. Where an arm needs a term the others do not
have, it is declared in the arm's row and counted in its hyperparameter budget.

### 5.2 The headline metric

> **`life_gain`** = mean survival time (simulated seconds) over the **final
> third** of lives − mean over the **first third**, at the matched budget,
> reported per seed.

Chosen for four reasons, and each of them is a defence against a specific way
this could be gamed:

1. **It is a trend, not a level.** `CURIOSITY_BAKEOFF.md` §2.5: raw levels are
   what get gamed. An arm that is born lucky scores zero.
2. **It is the owner's sentence, mechanically.** *"Life N+1 must be measurably
   better than life N because of what life N recorded."*
3. **It cannot be bought by dying fast.** Survival time is what is being
   measured, so the dark-room exploit (§5.4, D3) shows up as a *negative*
   `life_gain`, not as a high score.
4. **It does not name any mechanism.** No arm's reward may reference it; the
   `LT` G1 static audit applies unchanged — the reward path may not contain
   `ladder`, `rung`, `platform`, `survive`, `life`, `death`, or `age`.

**And it is a conjunction, not a scalar** — the `_check` follows PG.3/PG.4/`LT`
house style. `life_gain` is the number the ledger carries and the bakeoff ranks
on; the spec additionally requires, and reports per seed:

| # | secondary | requirement |
|---|---|---|
| **S1** | `n_lives` ≥ 12 in every seed | you cannot measure a trend across lives with three lives |
| **S2** | `needs_satisfied_rate` (fraction of decisions with `d(h) < d_max/2`) rises across lives | the trend is in the needs, not only in the clock |
| **S3** | `cross_life_transfer` = `life_gain` with the diary wiped at every death, subtracted from `life_gain` with it intact | **this is the W0-3 claim.** If wiping the diary changes nothing, the "cross-life memory" half of the world is decorative and the spec says so |
| **S4** | `panel_dwell` ≤ 0.15 per seed | PG.4's own threshold; **disqualifies**, does not merely reduce |
| **S5** | `chaos_occupancy` / `chaos_reward_ratio` | `CURIOSITY_BAKEOFF.md` §2.10 detector, reused unchanged; ≥ 3.0 **and** ≥ 2.0 → `Status.VOID` for that arm |

### 5.3 THE DUAL-BUDGET RULE — the design decision that makes the verdict trustworthy

This is the most important paragraph in §5, and it exists because of one
`LESSONS.md` entry: *"'Matched steps' has more than one meaning."* T2.02 matched
env-steps and hid a 16× optimiser-step gap.

The learning cores in this bakeoff differ **in exactly the way that makes that
trap fatal**. A world model's whole claim is that it extracts more from each
env-step; it pays for that with far more compute per env-step. So:

- **Match env-steps** and you have pre-decided in favour of the world model.
- **Match wall-clock** and you have pre-decided in favour of PPO.
- **Match optimiser steps** and you have compared nothing, because an optimiser
  step means something different in each arm.

There is no neutral single budget. So the bakeoff **runs once and scores
twice**:

> **Run to the envelope.** Every arm-seed runs until it has consumed **both**
> `N_STEPS` decisions of lived experience **and** `W_CLOCK` core-seconds of
> wall clock on the same hardware — whichever comes later. The full curve is
> recorded against both axes (F6).
>
> **Score A — matched experience.** `life_gain` read off the curve at exactly
> `N_STEPS` decisions. *"Which core learns most per unit of life?"*
>
> **Score B — matched compute.** `life_gain` read off the curve at exactly
> `W_CLOCK` core-seconds. *"Which core learns most per unit of the free
> compute we actually have?"*

One set of runs, two arbitrations, no arm advantaged by the choice of ruler,
and the cost is `max` over arms rather than `sum` over budgets.

**Pre-registered handling of the four outcomes**, fixed before any number
exists:

| Score A | Score B | verdict | what ships |
|---|---|---|---|
| X wins | X wins | **WINNER: X** | X is the learning core. Unambiguous. |
| X wins | Y wins | **SPLIT** (recorded as `Status.VOID` for the *core decision*, PASS for the *finding*) | Nothing ships yet. The finding — "X is more sample-efficient, Y is more compute-efficient, and at our budget that difference decides it" — is recorded, and `LC.05` re-runs at the **projected deployment budget** (Jack lives continuously on 4 shared cores, so `W_CLOCK` is extended 10× and `N_STEPS` with it) to break it. A SPLIT is real information and must not be resolved by preference. |
| TIE | X wins | **WINNER: X** | the ruler that discriminates decides |
| TIE | TIE | **TIE** | cheapest arm by declared cost, per `bakeoff.py` |

**Why SPLIT is VOID and not a coin-flip to the cheaper arm.** `bakeoff.py`
resolves a TIE by cost precisely because a tie means the choice does not matter
*yet*. A split is the opposite: the choice matters a great deal and the two
rulers disagree about it. Reporting that as a TIE and taking the cheap arm
would dress a real disagreement as an absence of one — the `Arm.cost = 0.0` bug
in a new costume.

### 5.4 The arms

Five candidate cores, one reference, one null, five controls. Shorthand: `o` is
the W0 dict observation (proprio ⊕ touch ⊕ retina rays ⊕ drive vector `[e,i,w,
d(h),ė,i̇]`), `r_h = d(h_t) − γ·... ` is the fixed drive-reward channel of §5.1/F8.

---

**A_ref — `sb3-ppo`. The reference arm whose failure indicts the harness.**

`stable_baselines3.PPO("MultiInputPolicy", net_arch=[128,128])` over the W0
env with the drive reward, `VecNormalize`, defaults otherwise. Not eligible to
be adopted — it exists so that `LESSONS.md`'s rule holds: *every comparison
carries a reference arm simple enough that its failure indicts the task.*
If A_ref cannot clear the null on `life_gain`, W0 is not a learnable survival
problem and **every other number in the bakeoff is uninterpretable — VOID, not
FAIL.**

---

**A0 — `ppo-needs`. The incumbent, repaired.**

Our harness with `SeparateActorCritic`: policy `|o|→128→128→8` (tanh), value
`|o|→256→256→1`, state-independent `log_std` (arXiv:2006.05990's measured
configuration, as D1 adopted). Reward is `r_h` only. `γ < 1`, asserted —
§3.3.4: at `γ = 1` Keramati & Gutkin's equivalence becomes path-independent and
the agent stops caring how badly it starves en route.

- **Hypothesis tested:** that a model-free policy-gradient core, given clean
  needs and a body that can act, learns to survive better across lives.
- **Prior:** *good*. Yoshida et al., *PNAS Nexus* 2024 — deep homeostatic RL
  with two needs on an 8-joint MuJoCo body, **using PPO** — is the closest
  published existence proof to W0 that exists (§3.3.4).
- **Params (cost):** ~**121K** [M-benchmarked shape].

---

**A1 — `ppo-lp`. A0 plus learning progress, two value heads.**

A0, plus `CURIOSITY_BAKEOFF.md`'s `lp` arm unchanged: absolute learning
progress over an auto-partitioned outcome space with hindsight relabeling.
Combination is **two value heads** (RND's design), not a summed scalar —
`PURPOSE_AND_SCAFFOLDING.md` §2.8 option 2, chosen there because removal is
then the deletion of a head rather than a re-normalisation. Satiety gating
`β_c(t) = β_0(1 − d(h_t)/d_max)` is **off** in this bakeoff and is PS.04's
variable, not ours (F8).

- **Hypothesis tested:** whether the leading intrinsic signal adds anything
  *on top of needs*, when needs already supply a reason to act.
  `PURPOSE_AND_SCAFFOLDING.md` §2.1 argues they may be substitutes.
- **Prior:** *uncertain, and interesting either way.* LP is a **selector, not a
  discoverer** (`CURIOSITY_BAKEOFF.md` §1.2 failure mode 5): LP is zero for a
  goal never once achieved. In W0 the needs supply the first successes, which
  is exactly the partner LP has been missing.
- **Params (cost):** ~**211K** [C] (A0 + a ~90K LP module).

---

**A2 — `dreamer-xs`. A DreamerV3-shaped world model at the smallest size that
runs here.**

RSSM (GRU deter 256, 32×8 categorical stochastic, 1 % unimix), encoder/decoder
over the dict observation, reward head and continue head with symlog+twohot
targets, KL balancing with free bits, actor-critic trained purely in
imagination with percentile return normalisation. Reconstruction included (that
is what distinguishes it from A4).

- **Hypothesis tested:** the brief's candidate 2, and simultaneously
  `UNIFIED_BRAIN_BAKEOFF.md`'s binding objective (§2): **a model that predicts
  all senses jointly is the unified brain.** A2 is the only arm in which
  question (a) — multimodal in one latent — is answered by the learning rule
  itself rather than by concatenation.
- **Prior:** *strong on paper, unproven at our replay ratio.* §3.0 measures
  that the model is cheap and the **replay ratio** is what costs; LC.02 fixes
  that number before this arm runs.
- **Params (cost):** **1,896,047** measured at this exact shape [M]
  (RSSM 1,432,160 + actor/critic 463,887).

---

**A3 — `wm-efe`. Active inference, given its fairest possible run: A2's world
model, a different actor objective.**

**A2's world model, byte-identical, same training loss, same train_ratio.** The
only change is the actor: instead of maximising a discounted return of `r_h`,
it minimises expected free energy over imagined rollouts,

```
G(π) = −E[ ln C(o_int) ]  +  −E[ information gain ]
        pragmatic: ln C set to −d(h),        epistemic: disagreement across a
        the interoceptive log-preference     K=5 ensemble of latent dynamics
                                             heads (§3.3.3: this IS the EFE
                                             epistemic term's standard estimator)
```

- **Why this is the honest test of candidate 3, and not a strawman.** §3.3
  establishes three things: EFE's epistemic term *is* expected information
  gain and is estimated in the literature with exactly this ensemble; on MDPs
  AIF *is* KL control; and there is no minimal deep AIF implementation with a
  track record. Testing "active inference" by building a whole separate agent
  would therefore be testing an implementation, not an idea. Holding the world
  model fixed and varying only the actor objective isolates the one thing that
  is genuinely different — **does scoring policies by free energy beat scoring
  them by discounted reward, on the same model?** That is a question with a
  clean answer.
- **Prior:** *low, and the mechanism of failure is named.* Champion et al.
  (arXiv:2303.01618) had an EFE actor collapse to a single action while its
  intrinsic reward stayed nonzero. The `chaos`/`panel_dwell` vetoes will not
  catch a *collapse*, so A3 carries its own pre-registered VOID condition:
  `action_entropy` in the final third below 10 % of A2's ⇒ `Status.VOID` for
  A3 with reason "epistemic-term collapse (arXiv:2303.01618)". A named,
  predicted failure that is detected is a result; an undetected one is a lie.
- **Params (cost):** **≈ 1,900,000** + 4 extra ensemble dynamics heads [C].
- **Two extra hyperparameters, declared:** the nat-scale of `ln C` and the
  precision `γ_EFE`. §3.3.1: these are the intrinsic/extrinsic coefficient
  under another name, and B2 counts them.

---

**A4 — `wm-latent`. The JEPA representative: A2 with the decoder deleted.**
*(CONDITIONAL — see §5.5)*

A2 with the reconstruction decoder removed; the dynamics is trained by latent
prediction against an EMA target encoder (the TD-MPC2 / JEPA family's core
move). Collapse is the failure mode and it is silent, so A4 carries a
mandatory diagnostic: **effective rank and per-dimension variance of the latent
must be reported every 1,000 decisions**, and a collapse (rank below a
pre-registered floor) is `Status.VOID` for A4, not a good loss curve.

- **Hypothesis tested:** does pixel-free latent prediction beat reconstruction
  *at our scale*? The literature's argument for latent prediction is that
  reconstruction wastes capacity on irrelevant detail — an argument made at
  ImageNet/video scale, on a body of evidence that is mostly not ours.
- **Prior:** *unknown at 2M parameters on a ray retina.* Reconstruction is a
  much stronger learning signal when the observation is 96-dimensional and
  every dimension matters, which is W0's regime.
- **Params (cost):** ≈ **1,370,000** [C] (A2 minus the ~528K decoder, plus a
  target encoder).

---

**Null and controls.**

| role | name | definition | must |
|---|---|---|---|
| **NULL** | `random` | uniform random action, and `random-repeat` (hold an action for k steps) | defines 0 for the learning gate |
| control | `statue` | do nothing | **fail**: shortest lives, strictly dominated (PS.01's requirement) |
| control | `randrew` | a fixed random stationary projection of the state as reward | **fail**: controls for "any optimisation pressure looks like learning" |
| control | **`frozen`** | the *winning* core, weights frozen, optimiser never stepped, everything else identical | **fail with `life_gain ≈ 0`.** This is the control the metric most needs: it detects a world-side drift in which lives get longer for reasons that have nothing to do with learning (respawn distribution, food respawn phase, integrity healing). Without it, `life_gain > 0` is not evidence of learning at all |
| control | `shuffled-diary` | diary rows permuted across lives before retrieval | **fail S3**: `cross_life_transfer` must collapse. If shuffling the diary changes nothing, W0-3 is decorative |
| control | **`darkroom`** | an arm rewarded for minimising predicted observation entropy | **must produce strongly negative `life_gain`** — it is the positive control for the dark-room detector D3. A detector that never sees its own positive control has measured nothing (T0.13's lesson) |

The `controls=` parameter of `run_bakeoff` takes all five; a control that
**clears** the learning gate inverts the verdict to VOID, which is the
behaviour we want for every row above.

### 5.5 Costs, gates and the decision rule

#### Cost unit, declared before the run

> **`Arm.cost` = trainable parameters in the learning core**, i.e.
> `sum(p.numel() for p in core.parameters() if p.requires_grad)`, asserted
> against the declared value to ±5 % (F7) with `Status.VOID` on mismatch.

Two reasons for parameters rather than core-seconds, departing from `LT`/`PS`
deliberately and saying so: (i) **compute is already Score B**, so using it
again as the tie-break would count it twice; (ii) parameters are the unit of
the simplicity budget (§6, B1), and a TIE that resolves toward the smaller core
is exactly the owner's principle expressed as arithmetic. Core-seconds per
1,000 decisions is a **mandatory reported secondary** for every arm — it is
Score B's axis, and §3.0 gives the measured baseline.

| arm | `cost` (trainable params) | measured/derived |
|---|---|---|
| `sb3-ppo` (reference, ineligible) | ~121,000 | [C] from the A0 shape |
| `ppo-needs` | **120,841** | [M] |
| `ppo-lp` | ≈ 211,000 | [C] |
| `dreamer-xs` | **1,896,047** | [M] |
| `wm-efe` | ≈ 1,900,000 + 4 ensemble heads | [C] |
| `wm-latent` | ≈ 1,370,000 | [C] |

Every one is inside the simplicity budget's B1 ceiling (5M) and only the two
PPO arms are inside its W0 soft target (750K) — which is itself a finding worth
stating in advance: **if a world-model arm wins, the soft target moves and the
reason is on the record.**

#### The gates, all fixed here

An arm must satisfy **all** of these or the bakeoff is VOID:

1. **Reference gate.** `sb3-ppo` must clear the null on `life_gain` by ≥ 3σ.
   Failing ⇒ **VOID**: W0 is not learnable and nothing else is interpretable.
2. **Gate A** (`bakeoff.py`'s own): every arm ≥ **3.0σ** over the shared null,
   σ = `max(arm seed std, null std)`.
3. **Gate B** (D1 §6.1's, and it is the one with teeth): every arm's trained
   `life_gain` must exceed **its own untrained twin's** by ≥ 3σ. T2.02's
   untrained MLP cleared random by 2.74σ against a 3.0σ gate [L]; a gate that
   an ungraded network nearly passes is not a gate.
4. **Matched-experience gate:** every arm within ±10 % of `N_STEPS` at Score A.
5. **Matched-compute gate:** every arm within ±10 % of `W_CLOCK` at Score B,
   measured as `time.process_time()` on identical hardware, with the MuJoCo
   share reported separately so the learner's cost is legible.
6. **Life-count gate:** `n_lives ≥ 12` per seed (S1).
7. **Vetoes:** `panel_dwell > 0.15` in any seed ⇒ **DISQUALIFIED**;
   `chaos_occupancy ≥ 3.0 AND chaos_reward_ratio ≥ 2.0` ⇒ `Status.VOID` for
   that arm; A3's `action_entropy` collapse ⇒ `Status.VOID` for A3.

#### The decision rule

```python
run_bakeoff(spec=LC.04,                       # Score A: matched experience
            arms=[ppo_needs, ppo_lp, dreamer_xs, wm_efe] (+ wm_latent if promoted),
            null_run=random_policy,
            seeds=[0, 1, 2],
            learning_gate_sigma=3.0,
            margin_sigma=1.5,
            higher_is_better=True,
            controls=[statue, randrew, frozen, shuffled_diary, darkroom],
            ledger=ledger)
# LC.05 re-runs the identical call on the SAME stored curves at W_CLOCK (Score B).
```

Then §5.3's table maps (Score A, Score B) → WINNER / SPLIT / TIE. And the
answer to the owner's question is read off *before the numbers exist*:

| outcome | "THIS is how Jack learns" |
|---|---|
| `ppo-needs` wins both | **Model-free policy gradient on a homeostatic drive reward.** The simplest thing on the table, and the one with the closest published existence proof (PNAS Nexus 2024). Simplicity budget tightens to 250K. |
| `ppo-lp` wins both | **The same, plus learning progress.** Curiosity earns its place *on top of* needs rather than instead of them — which is the reconciliation `GOAL.md` and `PURPOSE_AND_SCAFFOLDING.md` are in tension about. |
| `dreamer-xs` wins both | **Learn a model of the world that predicts every sense, and act inside it.** This is the answer that unifies §2's Q2 and Q3: the learning core and the unified brain become one objective, and `UNIFIED_BRAIN_BAKEOFF.md`'s binding heads become the core's own loss. |
| `wm-efe` wins both | Active inference has produced its first Dreamer-class win on a standard-form task, at 2M parameters. Extraordinary, and it would be re-run at 5 seeds before anything ships. |
| `wm-latent` beats `dreamer-xs` by ≥1.5σ | Reconstruction is wasted at our scale; the JEPA family's argument holds even on a 96-d observation. |
| any world-model arm wins Score A, a PPO arm wins Score B | **SPLIT.** The honest statement is "sample efficiency and compute efficiency point different ways *at 30 GPU-h/week*". LC.05b re-runs at the 10× deployment budget. |
| `frozen` clears the gate | **VOID, and the most important possible result:** lives were getting longer without learning. Every other number is void and W0 needs redesign. |
| A_ref fails | **VOID.** W0 is not a learnable survival problem. Fix the world, not the core. |

**One reading forbidden in advance, whatever the numbers say:** no outcome of
this bakeoff may be used to conclude anything about the **57M trunk** or about
**which senses matter**. W0 has no pixels and an 8-DoF body. `D1`/`T2.21` owns
the trunk's place in the motor path; `UB.9`–`UB.16` own the binding claim. This
bakeoff decides the *learning rule* and nothing else.

### 5.6 The specs

Exact `experiments/registry_expansion.py` format, to be appended to `EXPANSION`.

**Checked on this box, 2026-08-09, rather than asserted** [M]:

- the block **parses** (`ast.parse` against a stub `Spec`/`Budget`);
- **all seven** carry `hypothesis`, `falsified_by`, `null_baseline`, `control`
  and `kills` — verified by walking the AST, not by reading;
- **no id clashes** with the 128 ids already in `registry.py` +
  `registry_expansion.py`;
- **no glob hazards**: for every pair drawn from the LC ids and the 128
  existing ids, `fnmatch` confirms no id's module pattern matches another id's
  module. This is the `ME.11`/`ME.11.0` lesson checked by running it, and the
  `UB.1`/`UB.16` lesson about not reasoning it out instead.

**One honest flag before these are registered.** `LC.03` declares
`depends_on=["...", "PS.01"]`, and `PS.01` is *researched but not registered*
(`PURPOSE_AND_SCAFFOLDING.md` §4.4). `Ledger.blocked_by` returns any dependency
that is not `PASS`, so registering `LC.03` today would make it permanently
`BLOCKED` — which is exactly the failure mode `LESSONS.md` records as *"a
dependency graph can quietly make your most important claim unreachable"*
(eleven specs dead-ended behind one failure, and nobody noticed because
`run next` simply never listed them). **So `PS.01` must be registered in the
same commit, or `LC.03`'s dependency on it dropped with a reason.** Do not
register these in isolation.

```python
    # ── THE LEARNING CORE (docs/research/LEARNING_CORE.md) ──────────────
    # Two-digit ids from the start. run.py::_module_for globs lc_00_*.py etc;
    # verified by fnmatch on 2026-08-09 that no LC id shadows another. Do NOT
    # add an LC.0 or an LC.1 — see LESSONS.md, "A spec id that is a prefix of
    # another spec id disables one of them".

    Spec("LC.00", 0, "The learning-core question is decidable in a gridworld first",
         hypothesis="In a 12x12 survival gridworld with two depleting needs, "
                    "death on depletion, random respawn and a persistent "
                    "cross-life visit table, all four learning cores "
                    "(tabular Q on drive reduction; the same plus absolute "
                    "learning progress; a tabular latent-transition model with "
                    "value iteration in the model; and the same model scored by "
                    "expected free energy) run to completion, and at least two "
                    "produce a life_gain that beats the random null by 3 sigma "
                    "over 3 seeds.",
         falsified_by="Fewer than two cores clear the null. Then the METRIC is "
                      "wrong or the world is unlearnable, and no amount of "
                      "MuJoCo will repair either — LC.03 onward must not run.",
         null_baseline="Uniform random action on the same gridworld, same "
                       "seeds: life_gain by construction ~0 (lives do not "
                       "lengthen without learning).",
         metric="life_gain_cores_clearing_null", budget=Budget.CPU_FAST,
         depends_on=[], seeds=3,
         control="A FROZEN core — the same tabular agent with learning "
                 "disabled — must record life_gain within noise of zero. If a "
                 "frozen agent's lives get longer, the world drifts and "
                 "life_gain measures the world, not the learner. This control "
                 "is the reason the spec exists; it is cheaper to discover "
                 "here than after 25 CPU-hours.",
         kills="The whole LC programme, for two CPU-minutes. It is the "
               "cheapest thing that can falsify the metric, the world contract "
               "and the four-core framing before any body, any physics, any "
               "torch or any GPU is involved. Modelled on PS.00.",
         notes="No MuJoCo, no torch. Tabular over (x, y, need0_bucket, "
               "need1_bucket). Also emits the pre-registered numeric value of "
               "the FROZEN control's life_gain, which LC.03/LC.04 reuse as "
               "their own control threshold rather than inventing a new one."),

    Spec("LC.01", 2, "Every candidate core takes every sense into one latent, or it is not a candidate",
         hypothesis="For each admissible arm: (U1) every modality key reaches "
                    "the shared state tensor and no modality has a private path "
                    "to the action; (U2) perturbing modality A's input produces "
                    "a NONZERO finite-difference gradient at modality B's "
                    "encoder through the arm's declared binding loss; (U3) each "
                    "modality can be dropped without a shape error and the "
                    "core's internal uncertainty CHANGES when it is; (U4) the "
                    "need-state modality holds at least 1/|M| of the total "
                    "prediction loss at init.",
         falsified_by="Any arm failing any of U1-U4. That arm is EXCLUDED from "
                      "LC.03/LC.04 — not scored and beaten, excluded — per "
                      "SYSTEM.md's constitutional constraint. An arm cannot buy "
                      "admission with a task score.",
         null_baseline="A deliberately unbound core: per-modality encoders "
                       "feeding a concatenation with NO cross-modal loss term. "
                       "U2's finite-difference gradient must read exactly 0.0 "
                       "for it. That number is what U2 is measured against.",
         metric="unison_admission_conjunction", budget=Budget.CPU, seeds=3,
         depends_on=["PG.8"],
         control="TWO. (a) The unbound core above must FAIL U2 — if a core "
                 "with no binding term shows a cross-modal gradient, the probe "
                 "is reading autograd plumbing rather than the objective. (b) A "
                 "PLACEBO modality of matched dimension and matched statistics "
                 "carrying no information must NOT acquire a loss share above "
                 "1/|M| — if noise binds as well as a sense, U4 measures "
                 "capacity, not binding.",
         kills="Bare PPO as a candidate learning core. Per docs/research/"
               "LEARNING_CORE.md 3.7, PPO's senses meet only through a scalar "
               "reward, so an admissible PPO arm must carry "
               "L_masked_cross_modal. Also kills TD-MPC2 outright "
               "(arXiv:2310.16828 is state-based proprioception only, no "
               "vision, by construction).",
         notes="Runs BEFORE any learning. The finite-difference probe is the "
               "load-bearing part: MULTIMODAL_BINDING.md records pi-0.5 "
               "encoding its language prompt at 99.3% linear-probe accuracy "
               "while behaving invariantly to it, so 'the trunk sees it' is not "
               "evidence that the trunk USES it. U4 exists because DreamerV3's "
               "shipped loss_scales.rec is shared across keys: a 64x64x3 image "
               "contributes 12,288 reconstruction terms and a 10-dim needs "
               "vector contributes 10."),

    Spec("LC.02", 2, "A core that cannot live a life at survivable wall-clock is not a core",
         hypothesis="Every admissible arm sustains at least 5.0 simulated "
                    "seconds of Jack's life per real second on 3 ARM cores at "
                    "nice 19 with the learner in the loop, at the train_ratio "
                    "this spec selects for it; and the selected train_ratio is "
                    "the largest power-of-two value that clears that floor.",
         falsified_by="An arm below 5.0 sim-s/real-s at every train_ratio down "
                      "to its minimum. That arm is EXCLUDED: GOAL.md requires "
                      "lives, death and cross-life learning, and a core that "
                      "cannot produce a second life inside a builder iteration "
                      "cannot deliver them at any sample efficiency.",
         null_baseline="Physics alone, zero-action, same body and world: the "
                       "throughput ceiling no learner can exceed. Measured for "
                       "the humanoid at 31.6 sim-s/real-s (DIRECTION_AUDIT.md "
                       "4.1); measured here for the climber-rover.",
         metric="sim_seconds_per_real_second", budget=Budget.CPU, seeds=3,
         depends_on=["PG.8", "LC.01"],
         control="The 57M UnifiedBrain trunk in the control path MUST FAIL this "
                 "floor. DIRECTION_AUDIT.md 4.1 measured it at 0.17 sim-s/real-"
                 "s against a 160K MLP's 22.97 — 133x. If the trunk PASSES a "
                 "5.0 floor, the instrument is wrong, not the trunk.",
         kills="Any arm's train_ratio above the largest affordable value, and "
               "any arm that cannot reach the floor at all. NOTE THE "
               "ANTI-GAMING RULE: this spec's _check MAY NOT READ life_gain. "
               "Selecting a hyperparameter by its score is tuning on the "
               "metric; selection here is on wall-clock fit only, and the "
               "chosen value is committed to the ledger before LC.03 runs.",
         notes="train_ratio and model size are the two things DreamerV3 does "
               "NOT hold fixed across its 150+ tasks (arXiv:2301.04104 Table "
               "A.1), and they are exactly the two that decide affordability. "
               "Director (arXiv:2206.04114) ran at one gradient step per "
               "sixteen policy steps — train_ratio ~0.06 — under 24h on one "
               "V100, so a low ratio is not obviously crippling. Measured on "
               "this box 2026-08-09: PPO 13.1 and a 1.9M RSSM at train_ratio 1 "
               "19.6 CPU-core-seconds per 1,000 decisions, physics included."),

    Spec("LC.03", 5, "Screening: which learning cores learn to survive at all",
         hypothesis="At the LC.02-fixed train_ratio, run to the LC.04 envelope, "
                    "each admissible arm's life_gain beats the random null by "
                    ">=3 sigma AND beats its own untrained twin by >=3 sigma, "
                    "over 3 seeds, with n_lives >= 12 per seed.",
         falsified_by="Fewer than two arms clear both gates. Recorded VOID "
                      "'fewer than two learners' — which blocks the decision "
                      "instead of manufacturing one — and LC.04 does not run.",
         null_baseline="Uniform random and random-repeat action, same world "
                       "seeds, same evaluation lives. PLUS, per arm, that arm's "
                       "own UNTRAINED twin: T2.02's untrained MLP already "
                       "cleared random by 2.74 sigma against a 3.00 gate, so a "
                       "gate against random alone is nearly cleared by a "
                       "network that has never received a gradient.",
         metric="life_gain", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["LC.00", "LC.01", "LC.02", "PS.01"],
         control="FIVE, each on its pre-registered side. (a) statue (do "
                 "nothing) must die soonest. (b) randrew (fixed random "
                 "stationary reward projection) must miss the gate — it "
                 "controls for 'any optimisation pressure looks like "
                 "learning'. (c) FROZEN: the best arm with the optimiser never "
                 "stepped must record life_gain within noise of zero; if lives "
                 "lengthen without learning, the metric measures the world and "
                 "everything here is void. (d) shuffled-diary must collapse "
                 "cross_life_transfer. (e) darkroom (rewarded for minimising "
                 "predicted observation entropy) must record strongly NEGATIVE "
                 "life_gain — it is the positive control for the dark-room "
                 "detector, and a detector that never sees its own positive "
                 "control has measured nothing (T0.13).",
         kills="Any arm that cannot survive better than a network which has "
               "never received a gradient. Screening declares NO winner — that "
               "is LC.04's job, and separating them is why LT.03/LT.04 are "
               "separate.",
         notes="Headline life_gain = mean survival time over the final third of "
               "lives minus the mean over the first third, per seed. Reported "
               "alongside and gated as a conjunction: n_lives>=12; "
               "needs_satisfied_rate rising; cross_life_transfer > 0; "
               "panel_dwell <= 0.15 per seed (else DISQUALIFIED, PG.4's own "
               "threshold); chaos_occupancy>=3.0 AND chaos_reward_ratio>=2.0 => "
               "VOID for that arm (CURIOSITY_BAKEOFF.md 2.10). Arm wm-efe "
               "additionally VOIDs if its final-third action_entropy falls "
               "below 10% of dreamer-xs's — the epistemic-term collapse "
               "measured in arXiv:2303.01618, where the intrinsic reward stayed "
               "nonzero while coverage collapsed to one action."),

    Spec("LC.04", 5, "The learning core, arbitrated at matched EXPERIENCE",
         hypothesis="Among the arms that cleared LC.03, one core's life_gain at "
                    "exactly N_STEPS decisions of lived experience beats the "
                    "runner-up by >=1.5 sigma of the pooled seed spread.",
         falsified_by="No arm leads by 1.5 sigma => TIE, resolved to the "
                      "cheapest by trainable parameters. That is a real result: "
                      "the choice of learning core does not matter yet and the "
                      "simplest one ships.",
         null_baseline="The shared random null of LC.03, same seeds, same "
                       "evaluation lives, paired.",
         metric="life_gain_at_matched_experience", budget=Budget.CPU_LONG,
         seeds=3, depends_on=["LC.03"],
         control="Inherits LC.03's five controls, passed to run_bakeoff as "
                 "controls= rather than arms= — a designed-to-fail control "
                 "entered as an Arm would VOID this bakeoff permanently by "
                 "construction (LESSONS.md). A control that CLEARS the learning "
                 "gate inverts the verdict to VOID.",
         kills="Three of four learning cores, and the answer to the owner's "
               "question 'THIS is how it learns'. The winner is PROVISIONAL: "
               "adoption is VOID until UB.9 and UB.11 pass under it "
               "(SYSTEM.md's constitutional unison constraint), and the losers "
               "are NOT deleted until then.",
         notes="ARMS, cost declared in TRAINABLE PARAMETERS of the learning "
               "core, asserted to +-5% against the measured value with VOID on "
               "mismatch: ppo-needs 120841 (measured shape, tuned per "
               "arXiv:2307.03486 — LayerNorm before every dense layer, "
               "normalised value targets — plus L_masked_cross_modal for "
               "admission); ppo-lp ~211000 (+ absolute learning progress, two "
               "value heads); dreamer-xs 1896047 (measured: RSSM 1432160 + "
               "actor/critic 463887, GRU deter 256, 32x8 categoricals, symlog/"
               "twohot/free-bits/unimix/percentile-return-norm); wm-efe "
               "~1900000 + 4 ensemble dynamics heads (dreamer-xs's world model "
               "BYTE-IDENTICAL, only the actor objective differs: expected free "
               "energy with ln C = -d(h) and ensemble information gain). "
               "wm-latent ~1370000 is CONDITIONAL, promoted only if dreamer-xs "
               "clears LC.03. REFERENCE ARM sb3-ppo (~121000) is scored but "
               "INELIGIBLE FOR ADOPTION: if it fails to clear the null the "
               "whole bakeoff is VOID because W0 is not a learnable survival "
               "problem. Cost is parameters and NOT core-seconds on purpose: "
               "compute is already LC.05's axis, and counting it twice would "
               "let the tie-break re-decide the thing LC.05 decides."),

    Spec("LC.05", 5, "The same arms, arbitrated at matched COMPUTE",
         hypothesis="Scored off the SAME stored curves at exactly W_CLOCK "
                    "core-seconds instead of N_STEPS decisions, the LC.04 "
                    "winner still wins by >=1.5 sigma.",
         falsified_by="A different arm wins => SPLIT. Recorded as VOID for the "
                      "core decision and PASS for the finding: sample "
                      "efficiency and compute efficiency point different ways "
                      "at 30 GPU-h/week. Nothing ships; LC.05 re-runs at the "
                      "10x deployment budget to break it.",
         null_baseline="The same random null, scored at the same W_CLOCK.",
         metric="life_gain_at_matched_compute", budget=Budget.CPU_LONG,
         seeds=3, depends_on=["LC.04"],
         control="The two scorings must come from ONE set of runs — each "
                 "arm-seed runs until it has consumed BOTH N_STEPS decisions "
                 "AND W_CLOCK core-seconds, whichever comes later, and both "
                 "axes are recorded. A re-run for the second scoring is an "
                 "ERROR, not a convenience: it would let the arms differ in "
                 "anything other than the ruler.",
         kills="The pretence that there is a neutral single budget. T2.02 "
               "matched env-steps and hid a 16x optimiser-step gap "
               "(LESSONS.md, \"'Matched steps' has more than one meaning\"). "
               "Matching env-steps pre-decides for the world model; matching "
               "wall-clock pre-decides for PPO; so both are pre-registered and "
               "their disagreement is a reportable outcome rather than a "
               "choice made after the numbers exist.",
         notes="Every run records all four budgets — decisions, optimiser "
               "steps, core-seconds (MuJoCo share reported separately) and a "
               "gradient-FLOP estimate — plus a decimated curve of <=200 points "
               "spanning all lives. T2.01 stored curve_seed0[:8], iterations "
               "1-21 of 172, which is why its 'the curve PLATEAUED' claim was "
               "not in the ledger."),

    Spec("LC.06", 3, "The simplicity budget is enforced, not promised",
         hypothesis="The adopted learning core satisfies all four "
                    "pre-registered ceilings: B1 trainable parameters <= "
                    "5,000,000; B2 free hyperparameters <= 25, of which ZERO "
                    "are undocumented in the spec that used them; B3 <= 1,500 "
                    "raw lines in the learning rule and learned model; B4 >= "
                    "5.0 simulated seconds per real second on 3 ARM cores.",
         falsified_by="Any ceiling exceeded. The core is not adopted at that "
                      "size; it is reduced, or the ceiling is raised by the "
                      "procedure in LEARNING_CORE.md 6.4 — a bakeoff in which "
                      "the larger core beats the smaller by >=1.5 sigma at "
                      "matched env-steps AND matched wall-clock — never by "
                      "argument.",
         null_baseline="The shipped codebase as of 2026-08-09, which is what "
                       "the ceilings were written against.",
         metric="simplicity_budget_conjunction", budget=Budget.CPU, seeds=1,
         depends_on=["LC.04"],
         control="THE SHIPPED CODEBASE MUST BREACH ALL FOUR. Measured "
                 "2026-08-09: B1 41,525,008 > 5,000,000 (T1.11); B2 92 "
                 "UnifiedBrainConfig fields + 20 PipelineConfig training knobs "
                 "= 112 > 25; B3 6,114 + 1,220 = 7,334 lines > 1,500; B4 0.17 "
                 "< 5.0 (DIRECTION_AUDIT.md 4.1). A budget checker that cannot "
                 "flag the codebase it was written about is measuring nothing "
                 "(T0.13: a detector that cannot see its own positive control "
                 "has measured nothing). B4 needs this most — nothing in "
                 "experiments/ measures sim-seconds per real second today, so "
                 "the 133x gap went unmeasured until an audit looked.",
         kills="Complexity that has not earned itself. The owner, 2026-08-09: "
               "'it won't be the most complex model that Jack is. It will be "
               "just a system that can learn and get input from every single "
               "sense.' This spec is that sentence with numbers on it, and it "
               "is the guard that makes the 57M-vs-124K lesson unrepeatable "
               "rather than merely remembered.",
         notes="Counting rules, fixed here so they cannot be argued later. A "
               "hyperparameter fixed by a paper STILL COUNTS — DreamerV3's "
               "'one configuration for 150+ tasks' is a claim about tuning "
               "effort, not about count, and count is what determines how many "
               "things can be silently wrong (its configs.yaml is 220 lines "
               "and well over 100 knobs). A default counts twice: the audit "
               "reports both the number of knobs and the number whose value is "
               "never written down in the spec that used them, and the second "
               "number must be ZERO. Frozen perception is excluded from B1 — "
               "it is an input, not a learned parameter — which is what makes "
               "the frozen-swappable-tower principle affordable."),
```

### 5.7 Compute — free compute only, and this programme needs no GPU

Throughput, measured on this box unless marked:

| item | value | source |
|---|---|---|
| climber-rover physics, contact scan done once per decision | **~81 decisions/s** | [c] `CURIOSITY_BAKEOFF.md` §6 |
| decisions per simulated second | **5** (`LT` pilot: 3,000 decisions = 600 sim-s) | [c] |
| `ppo-needs` learner | **0.84** core-s / 1,000 decisions | [M] |
| `dreamer-xs` learner, train_ratio 1 | **7.3** core-s / 1,000 decisions | [M] |
| physics | **12.3** core-s / 1,000 decisions | [C] |

**The envelope**, fixed here: `N_STEPS = 100,000` decisions per arm-seed
(= 20,000 simulated seconds ≈ 5.6 sim-hours of life, ≥ 12 lives at ~25 sim-min
each) and `W_CLOCK = 1.2` core-hours per arm-seed. Each arm-seed runs to
**whichever comes later** (§5.3).

| stage | arithmetic | core-hours |
|---|---|---|
| **LC.00** gridworld, 4 cores × 3 seeds, tabular | minutes | **0.1** |
| **LC.01** unison admission, 5 arms × 3 seeds, no learning | probes only | **0.3** |
| **LC.02** throughput + train_ratio selection, 5 arms × 3–4 ratios | short timed rollouts | **0.6** |
| **LC.03/LC.04/LC.05** — one set of runs, two scorings. 4 arms + 1 reference, 3 seeds, `max(N_STEPS, W_CLOCK)` | ppo-needs 3 × 1.2 = 3.6 · ppo-lp 3 × 1.2 = 3.6 · dreamer-xs at train_ratio 4 ≈ 3 × 1.4 = 4.2 · wm-efe ≈ 3 × 1.6 = 4.8 · sb3-ppo 3 × 1.2 = 3.6 | **19.8** |
| untrained twins (5) + random null, 3 seeds, eval only | short | **0.8** |
| 5 controls × 3 seeds at half budget | | **4.5** |
| **`wm-latent`**, conditional | 3 × 1.3 | **+3.9** |
| **subtotal** | | **26.1** (**30.0** with A4) |
| **+25 % slack** (preemption, one re-run, the box's other tenants) | | **≈ 33 core-hours** (**38** with A4) |
| **GPU** | | **0.0** |

> **The entire learning-core bakeoff costs about 33 CPU-core-hours and zero
> GPU quota.** At 3 workers, `nice 19`, under ~1.5 GB, that is ~11 hours of
> wall clock, spread across builder iterations. This is the same property that
> makes `CURIOSITY_BAKEOFF.md`'s programme cost ~8 core-hours, and it comes
> from the same decision: **the arms are 0.1M–1.9M-parameter dedicated cores,
> not the 57M trunk.** T2.02 spent 6.3 hours on a P100 and still could not
> arbitrate.

**What this means for the Kaggle quota:** nothing here competes for it. The
right use of GPU quota remains what `DECISIONS_NEEDED.md` D3 says — the
T2.01/T2.02 re-run that D1 is waiting on and that 34 specs sit behind. The
learning-core question and the trunk question can therefore be answered **in
parallel, on different hardware**, which is the strongest scheduling argument
for running this programme now.

**Staging, cheapest falsifier first** (each stage can kill everything after it):

1. **LC.00, two CPU-minutes.** If fewer than two tabular cores beat the null,
   or the frozen control's lives lengthen, the metric or the world is wrong and
   nothing else runs.
2. **LC.01, ~20 minutes.** Excludes arms on unison grounds before they cost
   anything. Expected to exclude bare PPO and confirm TD-MPC2's exclusion.
3. **LC.02, ~35 minutes.** Fixes every train_ratio on wall-clock grounds and
   excludes anything below 5.0 sim-s/real-s.
4. **LC.03–LC.05, ~30 core-hours.** The bakeoff.
5. **UB.9 + UB.11 under the winner.** Adoption, not arbitration.

**Cheaper if the box tightens:** drop `wm-latent` (−3.9 core-h) — its
hypothesis is a *subset* of `dreamer-xs`'s (is the decoder worth keeping?), so
it is only interesting once `dreamer-xs` has cleared the gate. Sequencing
`dreamer-xs` first makes A4 conditional, which is how it is written.

---

## 6. The simplicity budget

The owner's principle, verbatim: *"it won't be the most complex model that Jack
is. It will be just a system that can learn and get input from every single
sense."* This section turns that sentence into three numbers that a spec can
enforce, plus the evidence that would justify raising each one.

### 6.0 What we are budgeting, and what we are not

The budget applies to **the learning core**: everything that receives a
gradient during a life, plus the code that computes those gradients. It does
**not** apply to:

- frozen perception (a frozen LLM, a frozen vision encoder) — those are
  *inputs*, not learned parameters, and `MULTIMODAL_BINDING.md`'s commitment
  ("frozen LLM on the side, small learned adapter, gradients never reach it")
  is what makes that separation real;
- the environment (`playground.py`, MuJoCo);
- measurement, logging, ledger, tests.

The reason for the split is the one that makes the whole project coherent:
**flexibility is bought by making the frozen part swappable and the learned
part small.** `GOAL.md` §"Flexible above all" already says this. The simplicity
budget is that sentence with numbers on it.

### 6.1 Where we are today, measured

| quantity | today | source |
|---|---|---|
| `UnifiedBrainConfig` fields | **92** | [M] AST count |
| `PipelineConfig` fields | **27**, of which **20** are training-rule hyperparameters (excluding 5 shape fields and 2 paths) | [M] AST count |
| **total human-settable knobs** | **≈ 112** | [C] |
| trunk parameters in the inference path | **41,525,008** | [L] T1.11 |
| parameters with **no live call site** | ≥ 2,974,977 (`WorldModel` alone, never constructed) | [M] |
| `UnifiedBrain.py` | **6,114 lines** | [M] `wc -l` |
| `TrainingPipeline.py` | **1,220 lines** | [M] `wc -l` |
| best measured locomotion result | **530.2**, from a **124,707**-parameter SB3 MLP | [L] T2.02 |
| **throughput of the current control path** | **0.17 sim-s/real-s** (57M trunk) vs **22.97** (160K MLP) — **133×** | [c] `DIRECTION_AUDIT.md` §4.1 |
| parameters with no live call site | **45.5M** | `experiments/protocol.py` docstring — **not** T1.03; see the correction below |
| params receiving no gradient, as actually recorded | **2,698,619** (worst offenders: `language_encoder` 2,624,000; `action_head` 49,761) | [L] T1.03, verified today |
| frozen LLM's contribution to rollout time | **3 %** (`llm_removal_speedup` = 1.03: 11.48 → 11.81 steps/s) for **6,933.8 MB resident** and 1,767,267,976 of the process's parameters | [L] T0.07, verified today |

> **A correction, made by verification rather than by reading.**
> `DIRECTION_AUDIT.md` §4.5 attributes *"45,538,295 parameters (38.6 %)
> receiving no gradient"* to **"T1.03's own null baseline"**. T1.03's recorded
> metrics say `params_without_grad = 2,698,619` [L]. The 45.5M figure is real
> but comes from `experiments/protocol.py`'s module docstring, where it is
> described as *"45.5M parameters with no live call site"* — **a different
> quantity** (unreachable code, not un-gradiented weights). Both statements are
> true; the attribution is not, and the two numbers are 17× apart.
>
> This is `LESSONS.md`'s *"Verify a mechanism claim before fixing it, even from
> a careful source"*, and it applies hardest exactly where it applied here: the
> audit is right about many harder things, **and the credibility does not
> transfer between claims.** Every ledger figure in this document was read out
> of `experiments/ledger.json` today rather than quoted from a summary.

**The last two rows are the whole argument.** The best number this project has
ever produced on its hardest task came from 124,707 parameters and about a
hundred lines of somebody else's library. The 57M trunk, at 457× the size,
scored 317.7 and missed its own learning gate — and even after D1's correction
strips that of its meaning, the honest reading is not "the trunk is better than
it looked", it is *"we do not have a single measurement in which extra
complexity bought anything"*.

### 6.2 The ceiling, pre-registered

Three limits. They apply to the arm that wins §5 and to whatever ships after
it, until raised by the procedure in §6.4.

| # | budget | ceiling | soft target for W0 | rationale |
|---|---|---|---|---|
| **B1** | **Trainable parameters in the learning core** (`sum(p.numel() for p in core.parameters() if p.requires_grad)`), summed over every module that receives a gradient during a life — policy, value, world model, intrinsic-reward model, adapters | **≤ 5,000,000** | **≤ 750,000** | 5M is ~12 % of today's inference trunk and ~8 % of the 57M figure the project has been carrying. It is above every number that has ever *worked* here (124,707; 54,179) by a factor of 40, so it cannot be accused of prejudging the bakeoff, and it is far below the size at which a P100 stops being enough. |
| **B2** | **Free hyperparameters**: every scalar, boolean or enum a human may set that is not (i) derived from the environment by assertion, (ii) a path, or (iii) a seed | **≤ 25** | **≤ 15** | Today's core has **20** in `PipelineConfig` alone, before the 92 in `UnifiedBrainConfig` [M]. 25 is *more* than PPO currently uses, deliberately — the budget must not be gameable by relabelling. What it forbids is the 112-knob status quo. |
| **B4** | **Throughput** — simulated seconds of Jack's life per real second, on 3 ARM cores of this box at `nice 19`, with the learner in the loop | **≥ 5.0 sim-s/real-s** | ≥ 10.0 | **The hard physical justification, and it is a measurement**: `DIRECTION_AUDIT.md` §4.1 measured the 57M trunk at **0.17** sim-s/real-s against a 160K MLP's **22.97** — **133×**. A 1-sim-hour life costs **5.8 real hours** with the trunk and **2.6 minutes** with the MLP; a 3-seed × 3-life × 1-sim-hour spec costs 52 hours versus 23 minutes, and 52 hours exceeds `run.py`'s own 15-hour ceiling on a box that serves paying tenants. **`GOAL.md` asks for lives, death and cross-life learning. A core below this floor cannot produce a second life inside a builder iteration, so it cannot be Jack's core at any sample efficiency.** Derivation of the 5.0 figure: §5.0b |
| **B3** | **Lines in the learning core** (`.py` implementing the learning rule and the learned model, excluding env, logging, tests, docstrings and comments — counted by AST statement nodes as well as raw lines, and both reported) | **≤ 1,500 raw lines** | **≤ 600** | `UnifiedBrain.py` is 6,114 lines [M] and contains at least one 2.97M-parameter module that is never constructed [M]. A core a person cannot read in one sitting is a core in which the next dropout bug will live. |

**Two counting rules, stated now so they cannot be argued later.**

1. **A hyperparameter that is "fixed by the paper" still counts.** DreamerV3's
   headline claim is that one configuration works everywhere; that is a claim
   about *tuning effort*, not about *count*, and the count is what determines
   how many things can be silently wrong. Every constant in the update rule is
   counted, whether or not anybody intends to change it.
2. **A default counts double in the audit, and here is why.** The most
   expensive bug in this project was `nn.Dropout(p=0.1)` — a value nobody set,
   in a module nobody added on purpose, with no line of code to read
   (`LESSONS.md`, "Call .eval()"). So B2's audit reports two numbers: the count
   of knobs, and the count of knobs whose value is **never written down in the
   spec that used them**. The second number must be **zero**.

### 6.3 The budget is enforced by a spec, not by intent

`LESSONS.md` is unambiguous that a promise in a document is not a guard:
*"Fixing one bug is maintenance. Making that bug unrepeatable is building."*
So `LC.06` (§5.6) measures B1/B2/B3/B4 on the winning core — by AST, by
`requires_grad`, and by a timed rollout — and **fails** if any ceiling is
exceeded. Without it this section is a wish.

It also has to be a spec that could fail, per Law 1 — so `LC.06` carries a
positive control: it is run against `UnifiedBrain.UnifiedBrain` +
`PipelineConfig` as they stand today, and **must** report all four budgets
breached — **41.5M > 5M** [L]; **112 > 25** [M]; **7,334 lines > 1,500** [M];
and **0.17 < 5.0 sim-s/real-s** [c]. A budget checker that cannot flag the
codebase it was written about is measuring nothing — the T0.13 lesson, *"a
detector that cannot see its own positive control has measured nothing"*.

B4 is the one that most needs the positive control, because it is the only
budget whose violation is currently **invisible in the ladder**: nothing in
`experiments/` measures sim-seconds per real second, so the 133× gap sat
unmeasured until an audit went looking for it.

### 6.4 What would justify raising a ceiling

Not an argument. Only this:

> A bakeoff on the **same task and metric**, in which the larger core beats the
> smaller by **≥ 1.5σ at matched env-steps AND at matched wall-clock** (§5.3's
> dual-budget rule), at ≥ 3 seeds, with the smaller core clearing the learning
> gate — i.e. the small core is a *learner* that lost, not a *non-learner*
> (T2.02's rule). The raise is then to the smallest size that achieves the win,
> not to the size that was tested.

Three specific, pre-registered exceptions where a raise is *expected* and its
evidence is named in advance, so that hitting them is not treated as a defeat
for simplicity:

| trigger | expected raise | why it is legitimate |
|---|---|---|
| Pixels enter the observation (W2+ in `SURVIVAL_WORLD.md`'s ladder) | B1 → ~10M, for the visual encoder only | a conv/patch encoder has an irreducible size; the *learning rule* does not grow |
| UB.16's `z` channel is demonstrated load-bearing (a sense is proven necessary by ablation) | B1 → whatever the demonstrated trunk needs, capped by the UB bakeoff's own winning arm | `UNIFIED_BRAIN_BAKEOFF.md` §5: *"If the bakeoff's winning arm is 6M, the trunk is 6M."* Parameters that passed an ablation have earned themselves |
| A multi-life claim (T5.x continual learning) needs a consolidation mechanism | B2 → +4, for the consolidation schedule only | measured need, and the knobs are named in the spec that needs them |

Everything else is a re-run of the 57M mistake with a new justification.

---

## 7. What each candidate keeps, adapts, or deletes

File by file for the big ones. Rows are the *learning-core* consequence only —
nothing here proposes deleting anything; deletion is the owner's call
(`SYSTEM.md`) and Tier 3's job.

**Legend:** KEEP (used unchanged) · ADAPT (used, with a named change) ·
BYPASS (not in the learning path, retained for other claims) · DELETE-CANDIDATE
(no remaining consumer under this candidate; goes to Tier-3 ablation).

### 7.1 The big three, file by file

---

#### `UnifiedBrain.py` — 6,114 lines [M], 41,525,008 params in the inference path [L] T1.11, 2,698,619 receiving no gradient [L] T1.03 (see §6.1's correction)

| candidate | verdict | detail |
|---|---|---|
| **PPO (+ binding loss)** | **ADAPT, heavily** | The trunk is not needed for W0's 8-DoF body — D1's own finding is that flat locomotion is proprioception-sufficient. What IS needed is a home for `L_masked_cross_modal`, and that is `UNIFIED_BRAIN_BAKEOFF.md`'s multi-token-per-modality rework of `:4204-4310` and `:1664-1727`, which is required *regardless* of D1 |
| **DreamerV3-class** | **ADAPT, and it gets smaller** | The RSSM replaces `CrossModalFusion` (`:1563`) + `TemporalMemory` (`:1595`) + the per-sense pooling as the fusion mechanism. Per-modality **stems survive** (`ProprioceptionEncoder :950`, `TouchEncoder :963`, `AudioEncoder :980`, `PrismaticVisionEncoder :627`) — a world model needs encoders too. `UNIFIED_BRAIN_BAKEOFF.md` §5 already licenses this: *"Nothing above requires 57M parameters... If the bakeoff's winning arm is 6M, the trunk is 6M."* |
| **JEPA / latent** | same as Dreamer, **minus the decoder** | and plus the RankMe + per-dim-std diagnostics of §3.4.4 |
| **Active inference** | same as Dreamer, **plus** an ensemble and a preference model | strictly more |

**Specific members, called out because they are the expensive ones:**

| member | line | today | under the recommendation |
|---|---|---|---|
| `WorldModel` | `:1757` | **2,974,977 params [M], never constructed** — `enable_world_model = False` at `:231` [M] | **ADAPT and REVIVE.** It is already a TD-MPC2-shaped latent model with `encode`/`predict_next`/`imagine_trajectory`/`plan_action_mpc`/`update_target_encoder` and an EMA target encoder — i.e. **most of arm A4 already exists in the repo and has never received a gradient**. Its `nn.Sequential` dynamics must become recurrent (a GRU cell) for A2, and `plan_action_mpc`'s uniform random shooting must go — §3.2.11's TD-MPC2 tax, unaffordable under B4 |
| `layers` (the transformer trunk) | — | 36,710,400 params [L] | **DELETE-CANDIDATE for the learning core** under every candidate. Not deleted from the repo — that is the owner's call and UB.9–UB.16 are the specs that earn or condemn it — but no W0 arm instantiates it, and B4 forbids it: 0.17 sim-s/real-s [c] |
| `ActionExpert` (flow matching) | `:2306` | 4,615,696 params [L], receiving zero gradient | see §7.2 |
| `LLMEncoder` | `:1221` | SmolLM2-1.7B, **6,933.8 MB resident**, contributing **3 %** of rollout time (`llm_removal_speedup` 1.03; 11.48 → 11.81 steps/s) [L] T0.07 | **BYPASS under every candidate**, and `DIRECTION_AUDIT.md` §4.2 upgrades out-of-process dialogue from tidy-up to prerequisite: *"In a multi-hour life on a 4-core shared box it is disqualifying."* The frozen-LLM *principle* is untouched and strengthened; the in-process *implementation* is what B4 kills |
| `IntrinsicCuriosityModule` | — | ~40K, never trained | **KEEP as a control** (PG.4's trap victim). Under a world model its ensemble-disagreement role is served by the model's own ensemble, so as a *candidate* it is DELETE-CANDIDATE |
| `AutotelicGoalGenerator` | — | ~90K, never trained | **KEEP** — it is arm A1's LP module, and `CURIOSITY_BAKEOFF.md` §3.1 notes this programme is also the evidence that decides T3.06 |
| `HierarchicalPlanner` | `:2093` | 37.17M, near-zero call sites [c] | **DELETE-CANDIDATE under every candidate.** And note the replacement is already published and cheap: Director is **~250 lines on DreamerV2** (§3.2.13) doing what HAC never did here |
| `AMPDiscriminator`, `NavigationPlanner`, `ObjectDetector`, `PhysicsRuleBank`, `Skill`/`MidLevelController` | various | untrained | **DELETE-CANDIDATE**; unchanged by this document, already Tier-3's business |

---

#### `TrainingPipeline.py` — 1,220 lines [M], 27 config fields of which 20 are training-rule knobs [M]

| candidate | verdict |
|---|---|
| **PPO** | **KEEP the file, ADAPT the config.** `rl_update` (`:495`), `collect_rollout_vec` (`:788`), `make_optimizer` (`:477`) stay. The T0.14/T0.16 repairs (`act_deterministic` at `:329`, mode switching) are **load-bearing under every candidate** and must be inherited, not re-implemented. F9 adds LayerNorm and value-target normalisation |
| **World-model candidates** | **ADAPT: the PPO update is replaced; everything around it survives.** `rl_update` gives way to a `(model loss, imagination actor-critic)` update, but `ReplayBuffer` (`:116`), the checkpoint/resume path (`:410`/`:428`, T0.04/T0.05 PASS), `normalize_obs` (`:366`) and `act_deterministic` are unchanged. **`ReplayBuffer` gets *more* important**, not less — a world model is trained from replay by definition, and §3.2.6's landmine 5 (`replay.capacity: 5e6` ≈ 61 GB) is a warning our own buffer must not reproduce |
| **all** | **DELETE-CANDIDATE: `EWC` (`:168`) and `physics_weight`.** Under a world model, catastrophic forgetting across lives is a claim about the model's replay distribution, not about a Fisher penalty. Not decided here — T5.x owns it — but the learning core stops depending on it |
| **all** | `train_phase0` / `train_phase2` / `train_phase8` (`:893`/`:1029`/`:1099`) are **DELETE-CANDIDATE**: they are the phase structure of a curriculum this direction has replaced with a world |

The 20 training-rule knobs are also **the B2 measurement's baseline**. A
world-model core would replace ~11 of them (clip_range, n_epochs_ppo,
ppo_minibatch, entropy_coef, vf_coef, gae_lambda, normalize_returns,
action_std_init, log_std_min, log_std_max, max_grad_norm) with ~8 of its own
(free_nats, unimix, β_dyn, β_rep, imag_horizon, train_ratio, retnorm limit,
twohot bins) — **so B2 is roughly a wash between candidates, and B2 is not a
reason to pick either.** Saying that plainly matters, because "fewer
hyperparameters" is the argument people expect to decide this and it does not.

---

#### The flow-matching action path — T1.11, T1.12, and a cost argument that is new

**What is verified** [L]: T1.12 PASS — sampler error **1.065 → 0.00134**,
conditioning ratio **1578**, shuffled-conditioning error 1.958, `x1`
parameterisation measured better than velocity (0.266 vs 0.407–0.620). This is
a real, multi-seed, control-bearing result and nothing in this document
weakens it.

| candidate | verdict |
|---|---|
| **PPO** | **BYPASS for W0.** D1's A5 already flags that "PPO through a multi-step flow decode is not standard practice and may simply not train" |
| **DreamerV3-class** | **ADAPT — and this is the best home the flow head has.** A conditional flow-matching decoder from a latent to an action is *exactly* what an actor in imagination is. The actor is trained on imagined rollouts where the "environment" is differentiable, which removes the reason it was awkward under PPO |
| **all** | **A new cost argument, from `DIRECTION_AUDIT.md` §4.4:** flow matching integrates ~4–10 Euler steps per decision, so the action head costs ~4–10 forwards per control step. Against **B4's 5.0 sim-s/real-s floor** that is a multiplier on the number that decides whether lives exist. This makes **action chunking (T2.18) a precondition rather than an optimisation** — a chunk of length *k* amortises the integration over *k* control steps — and T2.19/T2.18 must run as a pair with the flow head's advantage reported **per unit of compute**, not per step |

So the flow head's status is: **kept, re-homed, and now carrying a
pre-registered throughput cost it did not previously have to pay.**

### 7.2 The rest, briefly

| file | verdict under every candidate | why |
|---|---|---|
| `EpisodicMemory.py` (213 lines, no torch) | **KEEP, unchanged** | It is the diary, and ME.9/ME.10 already proved the double dissociation (wipe the diary → skill survives at 0.944; revert the weights → recall survives at 1.000) [L]. It is orthogonal to the learning rule by construction — that is the whole point of it being a separate store. `DIRECTION_AUDIT.md` §4.3: extractive-never-generative memory is **strengthened** by cross-life learning, because *"life N+1 is better because of what life N recorded"* is only falsifiable if the record cannot be confabulated |
| `Persistence.py`, `Forgetting.py`, `WorkingMemory.py`, `OwnerProfile.py`, `Reflections.py` | **KEEP** | substrate, not learning rule |
| `playground.py` (548 lines) | **KEEP, EXTEND to W0** | needs the drive integrator, death + random respawn, and the life index (§5.0's W0-1…W0-6). `SURVIVAL_WORLD.md` owns the extension |
| `experiments/*` | **KEEP** | `bakeoff.py`'s arms/controls split, `Status.VOID`, `_round6`, the gate-sensitivity scanner. `DIRECTION_AUDIT.md` §4.7 is right that this machinery *"is the part of the system the owner is actually buying"* |
| `VirtualWorld.py`, `EmotionalState.py`, `Personality.py`, `InnerMonologue.py`, `MovementMoodCoupling.py`, `TaskManager.py`, `SymbolicCalculator.py`, `AlphaGeometryLoop.py` | **BYPASS** | none is in the learning path under any candidate. **Naming hazard, already flagged by `PURPOSE_AND_SCAFFOLDING.md`:** `EmotionalState.get_energy()` is an *arousal* scalar in a mood model, **not** a metabolic variable, and must never be wired to W0's `e` without a spec saying why |
| `MoCapLoader.py`, `mocap_cmu.py` | **KEEP** | T1.13's 2,747 real CMU + KIT clips are the pretraining corpus for D1's A4, and would be a natural pretraining corpus for a world model's proprioceptive stem |

### 7.3 The one-line summary of §7

> **No candidate deletes anything that has ever passed a test.** What every
> candidate deletes is the same set of things: parameters with no live call
> site (`WorldModel`'s **2,974,977**, never constructed [M];
> `HierarchicalPlanner`'s 37.17M [c]; the 45.5M `protocol.py` was written
> about) and a trunk that costs **133×** the throughput of a network that
> outperformed it ([c] `DIRECTION_AUDIT.md` §4.1; [L] T0.07's 11.48 steps/s;
> [L] T2.02's 530.2 vs 317.7). **The learning-core question and the
> earn-your-parameters question turn out to have the same answer**, which is a
> good sign for both.

---

## 8. What this document refuses to claim

Stated so the next session does not mistake a survey for a result.

1. **That the recommendation is a decision.** Nothing in §4 has been measured on
   this project's own task. `LC.03`–`LC.05` decide; this document only says
   what to build first and what would change its mind. Adopting §4.1 without
   running §5 would be `SYSTEM.md` Law 3 violated in the most expensive
   possible place.
2. **That DreamerV3-at-1.9M works.** It is **below the smallest published size
   class (XS, 8M)** [V], and no per-size result exists at any budget we can
   afford (§3.2.2). "The world model does not work" and "the world model does
   not fit in our budget" would produce the same observation, and `LC.03` is
   not designed to distinguish them. If `dreamer-xs` fails the learning gate,
   the correct conclusion is *"not at this size, on this box"* — not *"world
   models do not work for Jack"*.
3. **That any of the throughput and cost projections are measurements of the
   real thing.** §3.0's RSSM benchmark is a *shaped* module of the right size
   on synthetic tensors, not DreamerV3. The GPU figures in §3.2.7 are the
   paper's own V100/A100-days converted by the paper's own factor
   (P100 = ½ V100) [V] — a conversion that flatters transformer workloads on a
   P100, which has no tensor cores. Treat every P100-hour figure as ±2×.
4. **That W0 is the survival world.** It is an 8-DoF climber-rover with a ray
   retina, no pixels, and no humanoid. It is chosen because T2.01/T2.02 mean
   Jack cannot walk [L] and because a negative result on a body that cannot act
   measures nothing (T1.02's lesson). **Nothing here licenses a conclusion
   about the 17-DoF humanoid, about pixels, or about which senses matter.**
5. **That the unison admission gate is sufficient.** `LC.01` checks that a
   cross-modal gradient exists and is nonzero. It does **not** check that
   binding is *useful* — that is UB.9's job, and it is why adoption is
   sequenced after UB.9/UB.11 rather than folded into the bakeoff. A nonzero
   gradient is a necessary condition wearing the clothes of a sufficient one,
   and the placebo control is the only thing standing between them.
6. **That `life_gain` is the right metric.** It is *a* metric with four named
   defences (§5.2) and one known blind spot: a world whose lives lengthen for
   environmental reasons would make every arm look like a learner. That is
   what the `frozen` control is for, and **if the `frozen` control ever clears
   the gate, everything in §5 is void** — including any result already
   recorded.
7. **That active inference has been given a complete hearing.** §5's A3 tests
   *the EFE actor objective on a shared world model*. It does not test
   sophisticated inference, DPEFE planning, AXIOM's non-neural conjugate
   mixtures, or a learned preference model. A3 losing would falsify "EFE as an
   actor objective beats discounted return on the same model" and nothing
   wider. **AXIOM's actual transferable idea — that its efficiency comes from
   not being a neural network at all** (§3.3.2) — is untested here and is a
   legitimate future arm.
8. **That the Crafter correction transfers to W0 unchanged.** §3.2.3 is about a
   2D discrete-action pixel game at 1M steps. W0 is continuous-action, 8-DoF,
   ray-based, and needs-driven. The correction is strong enough to remove the
   sample-efficiency argument from §4's reasoning — it is not strong enough to
   predict W0's outcome, and §4.3(3) is written as a counterargument, not as a
   result.
9. **That any spec here has run.** `LC.00`–`LC.06` are pre-registered designs in
   a research document. They are not in `experiments/registry_expansion.py`,
   nothing has been implemented, and **the ledger is the only place a
   capability may be asserted.** Per `DIRECTION_AUDIT.md` §6.6, twenty-eight
   researched specs are already outside the registry; these seven would make
   thirty-five, and that gap is itself a finding the next iteration should act
   on rather than widen.

## 9. What this document changed about the machine

Per `SYSTEM.md`: *"is the machine better than I found it?"*

1. **A new admission stage before scoring** (§5.0b). Bakeoffs in this repo have
   had nulls, controls, learning gates and margins. They have not had
   *admission criteria* — properties an arm must have to be allowed to compete
   at all, independent of its score. The constitutional unison constraint
   forced the category into existence, and the throughput floor is a second
   instance of it. **This generalises: any requirement that must not be
   tradeable against the metric belongs in admission, not in the objective.**
   `experiments/bakeoff.py` has no `admit=` parameter today; adding one would
   make this structural rather than documentary.
2. **The dual-budget rule** (§5.3): run once to the envelope of *both* budgets,
   score twice, and pre-register what a disagreement means. This is the direct
   descendant of `LESSONS.md`'s *"'Matched steps' has more than one meaning"*,
   and it is reusable by every future bakeoff whose arms differ in
   compute-per-sample. **SPLIT is a new verdict category** — distinct from TIE,
   because a tie means the choice does not matter and a split means it matters
   a great deal and the rulers disagree.
3. **A pre-registered ceiling on complexity, with a positive control**
   (§6, `LC.06`). The 57M-vs-124K lesson currently lives in prose. `LC.06` makes
   it a gate that must flag the shipped codebase on all four budgets or be
   considered broken.
4. **Three new silent-default landmines are now named and guarded before they
   can bite** (F9, F10, F11): the strawman-PPO comparison, per-modality loss
   imbalance deleting the need-state modality, and train_ratio chosen by score.
   Each is the `nn.Dropout(p=0.1)` shape — a value nobody sets, with no line of
   code to read — and each is now a written invariant with a spec attached.
5. **One cross-document attribution error caught and corrected** (§6.1):
   `DIRECTION_AUDIT.md` §4.5 attributes 45,538,295 no-gradient parameters to
   T1.03, whose ledger entry records **2,698,619** — a 17× gap between two real
   but different quantities. Found only because every ledger figure here was
   re-read from `experiments/ledger.json` rather than quoted from a summary.
   **The generalisable guard:** research documents cite each other freely and
   nothing checks that a `[L]`-tagged number matches the ledger. A trivial
   scanner — extract `spec_id` + number pairs from `docs/research/*.md` and
   diff them against `ledger.json` — would make this class of drift a red
   ledger entry rather than a thing a careful reader happens to notice.
6. **A lesson this document earned and which the next session should append to
   `docs/LESSONS.md`** (not appended here: this task was scoped to one file):

   > **A published baseline can be a strawman, and citing it makes it yours.**
   > DreamerV3's Crafter table reports PPO at 4.6 %. A NeurIPS 2023 paper got
   > **15.60 %** from the same algorithm with LayerNorm, a wider net and
   > normalised value targets — past DreamerV3's own 14.77 % at 2 % of the
   > parameters. Had this project adopted a world model on the strength of that
   > table, it would have repeated its own dropout confound at the level of the
   > literature: *one arm had a handicap and the difference was attributed to
   > architecture.* **Rule:** before citing a baseline's number as evidence,
   > check whether anyone has published a tuned version of that baseline. The
   > incumbent in your bakeoff must be the *best* implementation of the
   > incumbent, not the one that is easiest to cite.
