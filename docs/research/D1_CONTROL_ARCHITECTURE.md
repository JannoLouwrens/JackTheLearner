# D1 — Does the 57M UnifiedBrain trunk stay in the motor-control path?

> Researched 2026-08-09. Answers the open owner-decision **D1** in
> `docs/DECISIONS_NEEDED.md`. Serves `GOAL.md`: *one brain, all senses in
> unison, learning by living.*
>
> Companion to `docs/research/UNIFIED_BRAIN_BAKEOFF.md`. Every arm below
> satisfies that document's **UB.16 contract**:
> `senses → per-modality stems → TRUNK → k readout tokens → z → controller`.
> D1 decides *where the trunk sits relative to `z`*, never whether `z` exists.

## The one-paragraph answer

The evidence that the trunk "cannot learn" is, on inspection of the code that
produced it, mostly evidence that **the trunk arm was never given a fair PPO
update**. Three defects are measured below, two of them present in exactly one
arm of T2.02: dropout is active during rollout, update and evaluation, which
corrupts the PPO importance ratio at the source; the trunk received **16× fewer
optimiser steps** than the MLP it lost to, at "matched" environment steps; and
the observation projection destroys the very joint structure the tokenizer was
built to exploit. None of this vindicates the trunk. It means **T2.02 was VOID
for a second, independent reason nobody has recorded**, and D1 cannot be
answered from the runs we have. The bakeoff in §3–§6 repairs the harness first
(on CPU, for free), then puts six arms — including the frozen-trunk-plus-small-
head arm the literature favours — on one pre-registered metric.

## Contents

1. [Evidence-ranked diagnosis](#1-evidence-ranked-diagnosis-of-the-observed-failure)
2. [Survey: frozen backbones and action experts, 2023–2026](#2-survey-frozen-backbones-and-action-experts-20232026)
3. [The bakeoff: six arms](#3-the-bakeoff-six-arms)
4. [The `Spec(...)`](#4-the-spec)
5. [Declared costs](#5-declared-costs)
6. [The learning gate and the decision rule](#6-the-learning-gate-and-the-decision-rule)
7. [Compute estimate — free only](#7-compute-estimate--free-compute-only)
8. [How the senses reach the controller in every arm](#8-how-the-senses-reach-the-controller-in-every-arm)
9. [What this makes the machine better at](#9-what-this-makes-the-machine-better-at)
10. [What we refuse to claim](#10-what-we-refuse-to-claim)

---

## 0. Provenance of every number in this document

Numbers are tagged:

- **[L]** read from `experiments/ledger.json` (T2.00, T2.01, T2.02).
- **[C]** derived by arithmetic from committed code (`TrainingPipeline.py`,
  `UnifiedBrain.py`, `experiments/tests/t2_0*.py`) — reproducible by reading.
- **[M]** measured on this box, 2026-08-09, by instantiating the real
  `TrainingPipeline(PipelineConfig())` on CPU (aarch64, torch 2.8.0+cpu,
  `torch.manual_seed(0)`), **at initialisation**. Init-time measurements say
  what the optimiser sees on step 1; they do not describe step 6,000. Where
  that matters it is stated.
- **[V]** verified against arxiv.org during this research pass.
- **[c]** carried from another document and *not* re-verified here.

The measurement scripts live in the session scratchpad, not the repo. They
instantiate the shipped classes with the shipped config; they patch nothing.

Ground facts, for the reader who wants them once:

| quantity | value | tag |
|---|---|---|
| `Humanoid-v5` observation / action / action range | 348 / 17 / ±0.4 | [M] |
| `PipelineConfig.mujoco_obs_dim` | **376** (the Humanoid-**v4** value) | [C] |
| `UnifiedBrain` total = trainable parameters | 57,052,136 | [M] |
| …of which receive **no gradient** in the control path | **9,838,430 (17.2%)** | [M] |
| `obs_proj` parameters | 325,888 | [M] |
| trunk token-sequence length in the locomotion path | **37** | [M] |
| `nn.Dropout` modules with p > 0 inside the brain | **36** | [M] |
| T2.02 MLP arm parameters | 124,707 | [L] |

---

## 1. Evidence-ranked diagnosis of the observed failure

The question is why a 57M trunk scores 2.46σ where a 124,707-parameter MLP
scores 7.11σ [L]. Six candidate causes were named in the brief. Ranked by the
strength of the evidence *in this repository*, they come out in an order that
is not the intuitive one — and two of the intuitive front-runners are demoted
by measurement, which is the useful part.

### Rank 1 — Dropout is live during rollout, during the PPO update, and during evaluation. The importance ratio measures dropout, not policy improvement. [M]

`TrainingPipeline.__init__` builds `UnifiedBrain` and **never calls `.train()`
or `.eval()`**. The only `.eval()` in the file is inside
`EWC.compute_fisher` (`TrainingPipeline.py:172`), which the RL path never
calls. PyTorch modules default to training mode, so all **36** dropout modules
(p = 0.1: one on the attention softmax and one on the SwiGLU output in each of
8 `TransformerBlock`s, plus the cross-modal fusion stack) are active
everywhere.

Measured at initialisation, batch 256, two forward passes of the *same* state:

| quantity | train mode (as shipped) | eval mode | tag |
|---|---|---|---|
| mean abs difference in the policy mean between two identical forwards | **0.01370** | **0.0** | [M] |
| mean abs magnitude of the policy mean itself | 0.03638 | — | [M] |
| → jitter as a fraction of the signal | **37.7 %** | 0 % | [M] |
| mean abs difference in the value estimate | **0.03099** | — | [M] |
| mean abs magnitude of the value estimate | 0.04686 | — | [M] |
| → value jitter as a fraction of the signal | **66.1 %** | — | [M] |
| PPO ratio for an **unchanged** policy, 5th / 50th / 95th pct | **0.632 / 0.962 / 1.393** | 1 / 1 / 1 | [M] |
| fraction of samples outside the clip range 1 ± 0.3 | **20.3 %** | **0 %** | [M] |
| std of the log-ratio | 0.236 | 0 | [M] |

Four consequences, in order of how much they cost:

1. **The clipped surrogate fires on noise.** One sample in five lands outside
   `clip_range = 0.3` before the parameters have moved at all. For
   `advantage > 0` and `ratio > 1.3` the `min()` selects the constant branch
   and the gradient is exactly zero; for `ratio < 0.7` a large gradient flows
   that pushes the policy toward reproducing one particular random dropout
   mask. Neither is policy improvement.
2. **GAE is built on noisy values.** `old_values` are collected under dropout
   with 66 % relative jitter [M], so `delta = r + γV(s') − V(s)` inherits it,
   and `returns = advantages + old_values` inherits it twice.
3. **The evaluation is not an evaluation.** `eval_policy` in
   `t2_01_locomotion_vs_random.py:70` and `t2_02_mlp_showdown.py:91` is
   documented as *"Deterministic evaluation: mean action, no exploration
   noise"*. It runs `tp.model(...)` under `no_grad()` — which suppresses the
   graph, not dropout. Eval-mode and train-mode means differ by 0.00982 mean
   abs, **27.0 % of the mean's own magnitude** [M]. **The returns 261.0 [L]
   and 317.7 [L] are not the returns of the policies that were trained.**
4. **It is present in exactly one arm.** SB3's `MlpPolicy` contains no dropout,
   and SB3 calls `policy.set_training_mode(False)` for rollout collection and
   for `predict`. So T2.02's head-to-head compared a policy evaluated with
   injected noise against one evaluated without.

**The ledger already contains this failure's signature, unrecognised.** In
`T2.01.metrics.curve_seed0` [L]:

```
iter   1   pg_loss  0.1234    entropy 3.6618   action_std 0.30014   mean_reward 4.697
iter   2   pg_loss  0.0111    entropy 3.6597   action_std 0.30010   mean_reward 4.682
iter   3   pg_loss  0.0061    entropy 3.6555   action_std 0.29988   mean_reward 4.698
iter   4   pg_loss  0.0042    entropy 3.6423   action_std 0.29972   mean_reward 4.702
iter   6   pg_loss -0.0000    entropy 3.5907   action_std 0.29886   mean_reward 4.743
iter  11   pg_loss -0.0053    entropy 3.6043   action_std 0.29916   mean_reward 4.883
iter  21   pg_loss  0.0373    entropy 3.5071   action_std 0.29732   mean_reward 4.883
```

`action_std` moves from 0.30014 to 0.29732 — **0.94 % over 21 iterations,
840 optimiser steps** [C]. The entropy bonus contributes a *constant*
`−entropy_coef × 1` to every element of `d(loss)/d(log_std)` on every step, and
Adam's step size is ≈ `lr` when the gradient is consistent. At `lr = 3e-4` [M]
a consistent push would move `log_std` by ≈ 0.25 over 840 steps. It moved by
0.0094. **A parameter with a constant, known, non-zero gradient component that
refuses to move is Adam reporting `mean/√variance ≈ 0`** — i.e. a gradient whose
variance swamps its mean. That is precisely what a 20 %-outside-the-clip ratio
produces. T2.00's own control metric agrees from the other side:
`unnormalized_grad_ratio = 126.28` [L].

*Honest caveat:* the jitter fractions are measured at initialisation, where the
pre-tanh action head output is small. As `|raw|` grows the tanh saturates and
the relative jitter may fall. It cannot fall to zero while dropout is on, and
the point at which it matters most — deciding whether learning gets off the
ground at all — is exactly where it was measured.

### Rank 2 — "Matched environment steps" was not matched optimisation. The trunk got 16× fewer gradient steps than the MLP, at 457× the size. [C]

Straight arithmetic from the two arms' committed settings:

| | trunk arm | MLP arm | tag |
|---|---|---|---|
| samples per rollout | 32 envs × 128 = **4,096** | 1 env × 2,048 = **2,048** | [C] |
| minibatch | `ppo_minibatch = 512` | `batch_size = 64` | [C] |
| epochs per rollout | `n_epochs_ppo = 5` | `n_epochs = 10` | [C] |
| optimiser steps per rollout | 8 × 5 = **40** | 32 × 10 = **320** | [C] |
| env-steps completed (T2.02) | 638,976 | 638,976 | [L] |
| rollouts | 156 | 312 | [C] |
| **total optimiser steps / seed** | **6,240** | **99,840** | [C] |
| updates per env-step | 0.0098 | 0.156 | [C] |
| parameters updated | 57,052,136 | 124,707 | [M] / [L] |
| sample reuse | 5× | 10× | [C] |

**16.0× fewer updates, on a model 457× larger.** T2.01 v4 is the same story:
704,512 env-steps ÷ 4,096 × 40 = **6,880** optimiser steps for a 57M
transformer [C]. For calibration, Knowledge Insulation (2505.23705 [V])
reports that π₀ trained by flow matching alone needs **7.5× as many training
steps** to reach the performance of the insulated variant — and that is a
*pretrained* backbone with a well-conditioned objective. Six thousand steps is
not a plateau; on a network of this size it is barely the end of the warm-up.

The `ppo_minibatch: 64 → 512` change is documented in
`TrainingPipeline.py:81-86` as a *throughput* fix ("same total sample-passes
either way; bigger minibatches just fill the GPU"). Sample-passes were indeed
preserved. **Optimiser steps were divided by eight, and nothing in the spec
recorded that number.** It is the same class of error as T2.00's loss-ratio
gate: the quantity that was watched was not the quantity that mattered.

### Rank 3 — The observation projection destroys the tokenizer's only inductive bias, and bottlenecks the observation on the way. [C][M]

`JointTokenizer`'s docstring reads *"Tokenize robot state into joint tokens for
meaningful self-attention"* (`UnifiedBrain.py:1643`). What it is actually fed is
`project_obs(obs)`:

```
obs (348)  →  F.pad(+28 zeros) → 376   [mujoco_obs_dim is the Humanoid-v4 value]
           →  Linear(376,512) → LayerNorm(512) → ReLU
           →  Linear(512,256) → LayerNorm(256)
           →  JointTokenizer slices [0:136] into 17 tokens of 8 "features per joint",
              and [136:256] into one 120-d "body" token
```

Three problems, compounding:

1. **There are no joints in that vector.** The 256-d output is a dense learned
   mixture of all 348 observation dimensions, then LayerNormed across all 256
   channels — so slice *k* is not joint *k*, and the slices are statistically
   coupled by the normalisation. The transformer's per-joint attention is
   attending over arbitrary 8-dimensional fragments of an entangled embedding.
   The single structural prior that could justify a transformer over an MLP for
   proprioception is destroyed one module before the transformer sees anything.
2. **348 → 256 is a lossy bottleneck** trained only by the same corrupted PPO
   gradient (Rank 1). The MLP arm reads all 348 dimensions directly.
3. **28 of the 376 projection inputs are always zero** [C] — a Humanoid-v4
   constant left in a v5 pipeline. 7.4 % of the first layer's input weights are
   dead. Harmless on its own; diagnostic of how little attention this path has
   had.

The trunk sequence is 37 tokens [M]: `[CLS] [proprio] [mood] [joint×17] [body]
[action×16]`. Note `policy_mean` reads `output["actions"][:, 0, :]` — **the
first of 16 action tokens; the other 15 are computed and discarded** [C]. And
the `[mood]` token is `EmotionalState.get_mood_embedding()`, which in this
configuration is batch-independent and constant (nothing in the RL loop updates
`pad_vector`), yet its `mood_encoder` sits in the PPO graph and receives
gradient — so the locomotion objective silently edits Jack's emotional
projection while contributing zero state information to control [C][M].

### Rank 4 — Dead weight and gradient clipping: real, measured, and *smaller* than they look. [M]

- **9,838,430 parameters (17.2 %) receive no gradient** in the control path:
  `action_expert` (4,615,696), `language_encoder` (4,714,496), `language_proj`,
  `physics_head`, `task_completion_head`, `touch_encoder`, `touch_proj`,
  `movement_mood` [M]. They are in the optimizer's parameter list, saved in
  every checkpoint, and counted in every "57M brain" statement. AdamW skips
  parameters whose `.grad is None`, so they are not being decayed — the cost is
  memory and honesty, not corruption. It is the same disease `PIPELINE_REVIEW`
  §9 already recorded at 58.8 %; it is now 17.2 % and still not zero.
- **`max_grad_norm = 2.0` binds on every step.** Measured combined gradient
  norm on a realistic 128-sample minibatch with unit-normal advantages:
  **13.11** against a threshold of 2.0 → a **0.153× rescale** [M]. This *looks*
  fatal and mostly is not: Adam is invariant to a constant rescale of the
  gradient (`m/√v` is unchanged), and a clip that binds on *every* step is
  approximately constant. The residual effect is via `eps = 1e-5`, which at a
  post-clip per-parameter gradient of ≈ 2.9e-4 is ~3 % of the typical
  denominator — a mild damping of the smallest-gradient directions, not a
  cause. **Ranked here on purpose: the plausible-sounding mechanism was checked
  and demoted.**
- **A single learning rate of 3e-4 covers all three parameter groups** — the
  57M model, `obs_proj`, and `log_std` [M]. Note `train_phase2` uses
  `lr = 3.57e-5` (the RL-Zoo3 tuned Humanoid value) but T2.01/T2.02 call
  `make_optimizer(phase=3)`, which falls through to
  `config.learning_rate = 3e-4` [C]. So the trunk trained at **8.4× the
  hyperparameter this repo elsewhere believes is correct**, and at exactly the
  rate SB3 used for a 125K MLP. Differential learning rates get an arm (§3, A3)
  for this reason.

### Rank 5 — Value/policy gradient interference on the shared trunk: the front-runner hypothesis, and the measurement does not support it. [M][L]

This is the mechanism D1 option B is built on, so it deserves a real answer
rather than a plausible one.

| quantity, measured on `model.layers` (the 8 transformer blocks) | value | tag |
|---|---|---|
| ‖∇ policy term‖ | 10.851 | [M] |
| ‖∇ value term‖ (including `vf_coef = 0.43`) | 0.563 | [M] |
| ratio vf / pg | **0.052** | [M] |
| **cosine(∇pg, ∇vf)** on the shared trunk | **0.102** | [M] |
| T2.00, during training: max vf/pg trunk grad ratio | 2.71 | [L] |
| T2.00, at the end of training: vf/pg trunk grad ratio | 0.20 | [L] |
| T2.00 control, *without* return normalisation | 126.28 | [L] |

Two readings, both against the hypothesis. At initialisation the **policy** term
dominates the trunk 19:1, not the value term; and across a whole T2.00 run the
ratio peaked at 2.71 and ended at 0.20 [L] — return normalisation
(`normalize_returns`) already fixed the pathology it was written to fix, taking
126× down to ≤ 2.71×. More decisively, the two gradients are **nearly
orthogonal** (cos = 0.102): they occupy different subspaces of the trunk rather
than fighting over the same one. Interference of the kind PPG describes
(2009.04416 [V]) is *possible* here but is not what the numbers show.

**It remains a confound in the experiment even though it is weak as a
mechanism**, and this is the part T2.02 never declared: SB3's
`net_arch=[128, 128]` builds **fully separate** `policy_net` and `value_net`.
The arithmetic confirms it exactly: policy `348·128+128` + `128·128+128` +
`128·17+17` = 63,377, value `348·128+128` + `128·128+128` + `128+1` = 61,313,
`log_std` 17 — total **124,707**, the ledger's `mlp_params` to the digit
[C][L]. A shared-trunk MLP could not produce that number. So T2.02 compared
*transformer + shared trunk*
against *MLP + separate trunks* and attributed the whole difference to
"transformer vs MLP". A decoupled-critic arm is therefore **deliberately not in
the bakeoff** — the measurement says it would spend 7 GPU-hours on the weakest
hypothesis — but §6 pre-registers the condition under which it becomes one.

### Rank 6 — "Insufficient steps for the parameter count": untestable from the record, and the record is missing. [L]

D1's headline evidence is *"curve PLATEAUED by iteration 11 of 171"*. The
ledger cannot support that claim. `t2_01_locomotion_vs_random.py:202` records
`curve_seed0 = _CACHE["seeds"][0].get("curve", [])[:8]` — **the first eight
sampled points, covering iterations 1 to 21 of 172** [C][L]. Iterations 22–172,
where a plateau would have to be visible, were computed in the kernel and
discarded before the artifact was written. What the surviving eight points show
is `mean_reward` (per-step rollout reward, not episode return) rising 4.697 →
4.883 while `vf_loss` rises 0.19 → 0.36 and `value_mean` rises 0.036 → 3.605
[L] — a value function that has not converged, on a policy that had barely
moved.

So: the plateau may well be real, but **it is not in the record**, and a
decision that a 57M architecture is unsuitable currently rests on a curve that
was truncated by a slice. Combined with Rank 1 and Rank 2, "more steps would
not have helped" is not a statement anyone can make yet. Fixing the record is
in §9.

### Rank 7 — "The trunk is the wrong inductive bias for 348-dim proprioception": plausible, and completely untested. [—]

There is **no measurement in this repository that isolates this hypothesis**,
and it is the only hypothesis that D1 option D acts on. Every observation
attributed to it is currently confounded by Ranks 1–3. It is also the
hypothesis with the strongest prior from the literature (Andrychowicz et al.
2006.05990 [V] recommend two-hidden-layer separate MLPs, narrow policy and
wider value, for exactly these MuJoCo tasks) — which is why the bakeoff keeps
both MLP arms and does not assume the answer.

### One thing that is *not* a differentiator, recorded so nobody re-litigates it

`gamma = 0.95` and `gae_lambda = 0.9` are short for Humanoid (Andrychowicz et
al. 2006.05990 [V] call γ "among the most important hyperparameters" and
recommend 0.99 as the starting point). But **both arms of T2.02 used
γ = 0.95** [C], so it cannot explain the gap. It plausibly explains why *both*
arms (530 and 318) are an order of magnitude below RL-Zoo3's published
2M-step Humanoid numbers (SAC 6232, TD3 5567, TQC 7239, carried in the registry
[c]). It is a separate spec, not part of D1.

### Diagnosis summary

| rank | cause | evidence | arm-specific? | addressed by |
|---|---|---|---|---|
| 1 | dropout live in rollout / update / eval | [M] 20.3 % of ratios outside clip at zero policy change; eval ≠ train by 27 % of signal | **yes** (SB3 has none) | fixed in *all* arms before the bakeoff; guarded by D1.0 |
| 2 | 16× fewer optimiser steps | [C] 6,240 vs 99,840 | **yes** | matched-optimisation gate, §6 |
| 3 | obs projection destroys joint structure; 348→256 bottleneck | [C][M] | **yes** | A2/A3 rebuild the stem |
| 4 | 17.2 % dead params; clip binds at 0.153× | [M] | partly | reported; clip effect demoted |
| 5 | value/policy interference on the shared trunk | [M] cos = 0.102, vf/pg = 0.052 | confound, weak mechanism | *not* an arm; trigger in §6 |
| 6 | insufficient steps for the parameter count | [L] unverifiable — curve truncated to 21/172 iters | — | full curve logged, §9 |
| 7 | wrong inductive bias for proprioception | none | — | **the bakeoff** |

---

## 2. Survey: frozen backbones and action experts, 2023–2026

The recurring finding in the brief — *frozen backbone + small action head beats
end-to-end finetuning* — is real but is **not** the finding the literature
actually supports. The literature supports a narrower and more useful claim:
**insulate the pretrained backbone's gradients, adapt perception, keep the
action head small.** The distinction decides whether any of it transfers to a
trunk that was never pretrained on anything.

### 2.1 The action-expert lineage

**π₀** — *π₀: A Vision-Language-Action Flow Model for General Robot Control*,
Black, Brown, Driess, … Levine (Physical Intelligence), **arXiv:2410.24164**,
Oct 2024 [V]. A **3B-parameter PaliGemma VLM** plus a **~300M action expert**
(≈3.3B total). The expert is not a stacked MLP head: it is a second set of
transformer weights in a two-stream / mixture-of-experts arrangement where *the
weights interact only through the transformer's self-attention layers*. Actions
are **conditional flow matching over action chunks of horizon H = 50**, decoded
in **10 integration steps**, running control at up to **50 Hz**. Trained on
>10,000 hours of robot data.

**π₀.₅** — **arXiv:2504.16054**, Apr 2025 [V]. The change is **data, not
architecture**: co-training on multiple embodiments, high-level semantic
subtask prediction, object detection and web data as interleaved multi-modal
examples. Headline: long-horizon dexterous manipulation in **entirely new
homes**. Relevance to D1: the expert design was already settled; what bought
generalisation was heterogeneous co-training.

**Knowledge Insulation** — *Knowledge Insulating Vision-Language-Action Models:
Train Fast, Run Fast, Generalize Better*, Driess, Springenberg, Ichter, Yu,
Li-Bell, Pertsch, Ren, Walke, Vuong, Shi, Levine, **arXiv:2505.23705**, May 2025
[V]. **This is the paper that answers "why".** The mechanism is a
**stop-gradient inside the cross-attention** from action expert to backbone:
the expert computes `Q_a(X_a)·sg(K_b(X_b))ᵀ` and `P_ab·sg(V_b(X_b))`, so it
**reads** backbone features while **no gradient flows back into them**. This is
*gradient isolation, not weight freezing* — the backbone still trains, but only
from its own objective, which is next-token prediction on **discrete FAST
action tokens** (2501.09747 [V]; DCT + quantisation + BPE), described as a
*"substitute learning signal that is unaffected by the uninitialized weights of
the action expert"*. The stated cause of the damage is that **the action expert
is randomly initialised**, so its early gradients corrupt pretrained semantics:
*"naive training with such a randomly initialized action expert harms the
models' ability to follow language commands (presumably due to gradient
interference)"*. Numbers: π₀ with flow matching alone needs **7.5× as many
training steps**; overhead of insulation ≈ **20 % of training time**; control at
**10 Hz** vs **1.3 Hz** for autoregressive VLAs. LIBERO: 98.0 / 97.8 / 95.6 /
85.8 / 96.0 vs π₀'s 96.8 / 98.8 / 95.8 / 85.2 and π₀-FAST's 96.4 / 96.8 / 88.6
/ 60.2 [V]. (The language-following comparison ≈70 % insulated vs ≈45 % joint
vs ≈30 % π₀ is read off a figure, not tabulated — treat as approximate.)

**Octo** — Octo Model Team: Ghosh, Walke, Pertsch, Black, Mees et al.,
**arXiv:2405.12213**, May 2024 [V]. The cleanest architectural statement of the
UB.16 contract in the literature: **readout tokens** that *"attend to
observation and task tokens before it in the sequence, but are not attended to
by any observation or task token"*. Because they only read, a new input or a
new action head can be attached *"wholly retaining the pretrained weights for
the transformer, only adding new positional embeddings, a new lightweight
encoder, or parameters of the new head."* Action head: a **3-layer MLP, hidden
256**, diffusion with 20 steps. Sizes: Octo-Small **27M**, Octo-Base **93M**.
Head ablation on WidowX: **diffusion 83 % vs MSE 35 % vs discrete 18 %** — the
head's *form* matters more than its size. Also scale-dependent: ViT patch
encoders beat ResNets on large diverse data, but **ResNet wins when training
from scratch on ~100 demos**.

**RT-2** — Brohan et al., **arXiv:2307.15818**, Jul 2023 [V]. Co-fine-tuning vs
robot-only fine-tuning: PaLI-X **5B: 42 % → 44 %**; PaLI-X **55B: 52 % → 63 %**.
*"Co-fine-tuning … results in a better generalization performance than simply
fine-tuning it with robotic data."* **The gap widens with scale** (2 points at
5B, 11 at 55B) — the more pretrained knowledge there is, the more robot-only
training destroys. Training the 5B model **from scratch: 9 %**. Latency
1–3 Hz at 55B.

**Gato** — Reed, Żołna, Parisotto et al., **arXiv:2205.06175**, May 2022 [V].
1.2B parameters, 604 tasks, >450 above 50 % expert score. The size is
explicitly a *latency* choice: *"We focus our training at the operating point of
model scale that allows real-time control of real-world robots, currently
around 1.2B parameters."* The specialist gap is the relevant warning: a **1.18B
Atari-only specialist beat human level on 44 games to Gato's 23**, and a
**79M Meta-World specialist reached 96.6 % across all 50 tasks**. Generalist
trunks lose to small specialists on any single task.

### 2.2 The two counterweights — read these before freezing anything

**OpenVLA** — Kim, Pertsch, Karamcheti et al., **arXiv:2406.09246**, Jun 2024
[V]. 7B, Llama-2 + dual DINOv2/SigLIP encoders, 970k OXE demonstrations; beats
RT-2-X (55B) by **+16.5 points** with 7× fewer parameters. LoRA (rank 32,
97.6M trainable = 1.4 %) reaches **68.2 ± 7.5 %** against full finetuning's
**69.7 ± 7.2 %** — PEFT matches full FT. **But the frozen-encoder ablation goes
the other way: frozen vision encoder 47.0 ± 6.9 % vs 69.7 % finetuned**, a
~23-point *loss*; last-layer-only 30.3 %. OpenVLA concludes finetuning the
vision encoder is *crucial*, explicitly contrary to prior VLM findings.

**VC-1 / CortexBench** — Majumdar, Yadav, Arnaud et al., **arXiv:2303.18240**,
2023 [V]. 17 embodied tasks, >10,000 GPU-hours, frozen pretrained visual
representations. **No universal winner**: R3M best on Adroit/MetaWorld/DMControl,
MVP-L best on TriFinger/ImageNav/Mobile-Pick, CLIP best on ObjectNav; VC-1's
68.7 % mean is best on average and *dominant nowhere*. And **frozen is not
enough**: adaptation adds Adroit 59.3 → 72.0, MetaWorld 88.8 → 96.0, DMControl
66.9 → 80.9, ImageNav 70.3 → 81.6. The single most important number for D1:
**a random frozen ViT-B scores 20.4 % against 47.4 % for training from
scratch.** (I could not verify the sharper "frozen PVRs underperform
from-scratch specifically on proprioception-sufficient tasks" framing —
**UNVERIFIED**; the weaker version above is what the paper supports.)
Supporting: R3M **arXiv:2203.12601** [V] (>20 % over from-scratch across 12
manipulation tasks, frozen); MVP **arXiv:2210.03109** [V] (frozen 307M MAE ViT,
up to 81 % relative over from-scratch).

### 2.3 Shared trunks in actor-critic RL

**Phasic Policy Gradient** — Cobbe, Hilton, Klimov, Schulman,
**arXiv:2009.04416**, ICML 2021 [V]. States the dilemma exactly: with a shared
trunk *"there is a risk that the optimization of one objective will interfere
with the optimization of the other"*, while *"using separate networks avoids
interference between objectives, [and] using a shared network allows useful
features to be shared."* PPG's answer is neither pole: disjoint policy and
value networks **plus** a periodic auxiliary phase that distils value features
into the policy network under KL regularisation. **The counter-finding matters
for D1:** in Appendix B (Fig. 8), fully separate networks *underperform* the
feature-sharing PPO baseline on Procgen, and the authors note sharing *"is often
unnecessary in environments with a lower dimensional input space."* So: (a)
decoupling alone is not the fix — controlled gradient *routing* is, which is
structurally the same lesson as Knowledge Insulation; and (b) Humanoid's 348-d
proprioception is exactly the "lower dimensional input space" where sharing
buys least.

### 2.4 Network size in on-policy RL

**What Matters in On-Policy RL** — Andrychowicz, Raichuk, Stańczyk, Orsini,
Girgin, Marinier, Hussenot, Geist, Pietquin, Michalski, Gelly, Bachem,
**arXiv:2006.05990**, Jun 2020 [V]. >250,000 agents, >50 design choices, five
MuJoCo environments including Humanoid. Directly applicable recommendations:
**separate policy and value networks**; **two hidden layers each**; *"tune the
policy width (it might need to be narrower than the value MLP)"* while for the
value network there is *"no downside in using wider networks"*; **tanh**;
*"initialize the last policy layer with 100× smaller weights"*; **initial action
std 0.5** (best on all but Hopper); *"always use observation normalization"*;
Adam **3e-4** as the default; **γ = 0.99** as the starting point; go over the
experience multiple times and **shuffle individual transitions**. (The specific
"64–256 units" figure is present in their swept configurations but I could not
confirm it as a stated recommendation — **partially UNVERIFIED**. Likewise I
found **no** paper whose headline result is "small networks are enough for
MuJoCo/DMC"; this survey does not cite one.)

**Plasticity loss** — Juliani & Ash, **arXiv:2405.19153** [V]; survey
**arXiv:2411.04832** [V]; Abbas et al., **arXiv:2303.07507** [V]. Mechanisms:
dead units, parameter-norm growth, feature-rank collapse — distinct and not
always co-occurring. Relevant as a *later* concern for A2/A3 if they train for
long; not a candidate explanation for a 6,240-step run.

### 2.5 Synthesis — and the part that does not transfer

Four distinct causes are established, and they should not be blurred:

1. **Random-init gradient shock** (2505.23705 [V]): an untrained continuous
   head emits large early gradients that corrupt pretrained representations.
   The fix is a **stop-gradient**, not freezing.
2. **Objective mismatch** (2505.23705, 2009.04416 [V]): the action objective is
   a poor representation-learning signal for the backbone; give the backbone a
   different, better-conditioned objective.
3. **Catastrophic forgetting of pretraining** (2307.15818 [V]): robot-only
   finetuning costs 11 points at 55B, and the penalty grows with scale.
4. **Latency** (2205.06175, 2307.15818, 2410.24164 [V]): the small head is what
   makes 10–50 Hz control possible.

**Now the honest part. Three of those four presuppose a backbone with
pretrained knowledge worth protecting. Jack's trunk has none.** It is 57M
randomly-initialised parameters. Freezing it does not protect knowledge; it
creates a **random-feature encoder**, and VC-1 measured that case directly:
random frozen ViT-B **20.4 %** vs from-scratch **47.4 %** [V]. The only cause
in the list that transfers unconditionally is **(1) gradient shock — which
runs the other way here**: it is the *trunk* that is random, so the trunk's own
gradients are the noisy ones, and the insulation should protect the *controller*
from the trunk rather than the reverse.

Three consequences, and they shape the arms:

- **"Frozen trunk + small head" must be tested in two versions**: frozen
  **random** (A2's cheap cousin — represented by A1's z-free controller and by
  the contrast against A4) and frozen **pretrained** (A4). Only the second is
  what the literature actually endorses, and it obliges us to *pretrain the
  trunk on something first*. The corpus exists and is already PASSing: T1.13's
  **2,747 real CMU + KIT mocap clips**.
- **The insulation boundary is a stop-gradient, not `requires_grad=False`**
  (2505.23705). A4 and A5 use `.eval()` + stop-grad at the readout tokens, so
  the trunk can later be trained by *its own* objective (UB.10's masked
  cross-modal prediction) without touching the control path — which is
  precisely the UB.16 contract.
- **Do not expect a large effect.** `UNIFIED_BRAIN_BAKEOFF.md` §7 already
  calibrated this from Kepler-Encoder: fused-vs-unimodal R² of 0.049 / −0.001 /
  0.187, one of three negative. And on **flat locomotion specifically**,
  proprioception is *sufficient* — so the honest prior for "z helps walking" is
  **zero**, and UB.16 makes "no degradation on flat walking" a **PASS**
  condition, not a failure.

---

## 3. The bakeoff: six arms

**Spec id `T2.21`** (`T2.03`–`T2.20` are taken). Ordering is CPU-first: two arms
run on 4 ARM cores and can void the whole thing for free before Kaggle is
touched.

### Fixed across all arms (these are repairs, not variables)

Every arm inherits the following, and **no arm varies them** — otherwise the
bakeoff measures the repairs:

| R | repair | why | file |
|---|---|---|---|
| R1 | `model.eval()` during rollout collection and evaluation; `model.train()` only inside the update — **or**, preferred, `config.dropout = 0.0` for the control path, since PPO's own sampling is the regularisation | Rank 1 | `TrainingPipeline.collect_rollout_vec`, `rl_update`, both `eval_policy` |
| R2 | rollout obs statistics **frozen during evaluation** (`self.obs_freeze = True`), matching SB3's `vec.training = False` | Rank 1(4) | `TrainingPipeline.normalize_obs` |
| R3 | `mujoco_obs_dim = 348`; the zero-pad path becomes an assertion failure, not a silent pad | Rank 3 | `PipelineConfig`, `project_obs` |
| R4 | GAE bootstraps `V(s_T)` at the rollout boundary instead of 0 | correctness | `rl_update` |
| R5 | every run records `optimiser_steps`, `updates_per_env_step`, the **full decimated curve** (≤ 200 points spanning all iterations, not `[:8]`), and the measured trainable-parameter count | Rank 2, Rank 6 | test files |
| R6 | paired evaluation: identical seeds, identical eval-episode env seeds, identical evaluation count across all arms (2108.13264 [c]) | statistics | test files |

**R1 through R6 land in a preparatory spec `D1.0` (CPU, minutes) and must PASS
before `T2.21` may run.** `D1.0` is where the repairs become falsifiable rather
than assumed.

### The arms

Shorthand: `z` is the UB.16 percept vector — `k = 8` readout tokens from the
trunk, each reduced by a shared `LayerNorm(512) → Linear(512, 8)`, concatenated
to `z ∈ R^64`. Controller shapes follow Andrychowicz et al. 2006.05990 [V]:
**separate** policy and value MLPs, two hidden layers each, **narrow policy
(128) and wider value (256)**, tanh, last policy layer initialised 100× small.

---

**A0 — `sb3_reference`. The reference arm whose failure indicts the harness.**

Verbatim T2.02 MLP arm: `stable_baselines3.PPO("MlpPolicy", net_arch=[128,128],
n_steps=2048, batch_size=64, n_epochs=10, lr=3e-4, γ=0.95, λ=0.9,
clip=0.3, ent_coef=0.002)` over `VecNormalize(Humanoid-v5)`, CPU.

- **Hypothesis tested:** that our environment, step budget and evaluation
  protocol can reproduce the 530.2 [L] that the whole of D1 is being weighed
  against. This is `LESSONS.md`'s rule — *every comparison carries a reference
  arm simple enough that its failure indicts the task.*
- **Changes:** none to the repo; it is an external reference.
- **Params (cost):** **124,707** [L], reproduced by arithmetic [C].
- **If it fails:** the entire bakeoff is VOID. Nothing else is interpretable.

---

**A1 — `mlp_no_z`. Pure MLP control; the trunk is not in the motor path at
all.** (D1 option A/D at the control interface; brief's candidate 5.)

Our `TrainingPipeline` harness with `self.model` replaced by
`SeparateActorCritic(obs=348)`: policy `348→128→128→17` (tanh), value
`348→256→256→1` (tanh), state-independent `log_std` init `log 0.5`
(2006.05990 [V]). Controller input slot is **`[proprio(348) ⊕ z(64)]` with `z`
hard-zeroed**, so A1 and A4 are the *same controller* differing only in whether
`z` is live.

- **Hypothesis tested:** is the gap in T2.02 the *architecture* or the
  *harness*? A1 ≈ A0 means our PPO is fixed and every trunk arm below is
  interpretable. A1 ≪ A0 means it is not, and the bakeoff is VOID.
- **Changes:** new `experiments/tests/t2_21_control_bakeoff.py` +
  a `SeparateActorCritic` module; `TrainingPipeline` gains a
  `policy_module` injection point. No change to `UnifiedBrain.py`.
- **Params (cost):** **218,787** [C].

---

**A2 — `trunk_e2e_repaired`. The 57M trunk end-to-end through PPO, with Ranks
1–3 fixed.** (D1 option C, given its fairest possible run.)

R1–R6, plus: the trunk's stem is rebuilt so the tokenizer gets what it was
designed for — the **raw 348-d Humanoid-v5 observation sliced by its documented
layout** into 17 per-body groups plus a root group, each embedded by its own
`Linear(d_group, 512)`, replacing `obs_proj` + the `[0:136]/[136:256]` slicing.
`policy_mean` reads action token 0 of `action_chunk_size = 1` (not 16).
Optimisation is brought to the matched floor of §6.

- **Hypothesis tested:** **was it trainability, not architecture?** This is the
  arm that makes T2.01/T2.02 honest. If A2 clears the learning gate and ties
  A1, the whole D1 premise collapses in the trunk's favour.
- **Changes:** `UnifiedBrain.JointTokenizer` gains a `raw_proprio_layout` mode;
  `TrainingPipeline.project_obs` becomes identity in that mode.
- **Params (cost):** **47,327,523** trainable-and-gradient-receiving
  (47,466,019 measured [M], which already includes `obs_proj`, minus `obs_proj`
  325,888, plus a 187,392-parameter per-body stem), ≈56.9M resident. Declared
  with a ±5 % assertion (§5).

---

**A3 — `trunk_e2e_difflr`. A2 with a 100× lower trunk learning rate.**
(Brief's candidate 4; the soft form of insulation.)

Identical to A2 except `make_optimizer` builds four groups:
`layers` + `cross_modal_fusion` at **3e-6**; stem, heads, `log_std` at
**3e-4**; weight decay 0 on all norms and on `log_std`.

- **Hypothesis tested:** that one learning rate applied to a 47M trunk and a
  17-element `log_std` is the failure. The measurement that motivates it: the
  optimizer really does apply `3e-4` to all three groups [M], and the repo's
  own `train_phase2` believes `3.57e-5` is correct [C].
- **Changes:** `make_optimizer` only.
- **Params (cost):** **identical to A2 by construction.** See §6 for how a
  TIE between exactly A2 and A3 is reported (it cannot be broken by cost, and
  pretending otherwise is the `Arm.cost` bug in a new costume).

---

**A4 — `frozen_pretrained_trunk_mlp_head`. The recommended arm.**
(D1 option A; brief's candidates 1 and 3.)

Two stages.
*Stage 1, pretraining:* the trunk is trained on **T1.13's 2,747 real CMU + KIT
mocap clips** with a masked-motion objective — mask 40 % of joint tokens and a
contiguous 0.5 s temporal span, predict the masked proprioceptive state and the
next state — the cheapest available instance of the cross-modal masked
prediction that `UNIFIED_BRAIN_BAKEOFF.md` §1.2 identifies as the binding force
(2311.00924, 2410.16424 [c]). No RL, no policy.
*Stage 2, control:* trunk `.eval()`, **stop-gradient at the readout tokens**
(2505.23705 [V]) rather than only `requires_grad=False`, so the trunk can later
be trained by its own objective without touching control. `k = 8` readout
tokens → `z ∈ R^64`. Controller = A1's `SeparateActorCritic` with `z` live.

- **Hypothesis tested:** the literature's actual claim — a frozen backbone helps
  **when it was pretrained on something**. A1 vs A4 is the clean contrast
  (`z` zeroed vs `z` live, identical controller, identical trainable count
  modulo the readout adapter). Per UB.16, **a TIE on flat locomotion is a PASS,
  not a failure**; a *loss* to A1 is the VC-1 random-frozen-features result
  (20.4 % vs 47.4 % [V]) reproducing on us, and would mean `z` must not enter
  the control path until it is demonstrably informative.
- **Changes:** `UnifiedBrain` gains `readout_tokens` (k learnable query tokens,
  read-only, Octo-style 2405.12213 [V]) and a `percept(z)` accessor;
  new pretraining script under `experiments/tests/`.
- **Params (cost):** **248,491** trainable (57.05M frozen resident) [C].
- **Free efficiency win:** with the trunk frozen, **`z` is computed once during
  rollout and cached in the buffer**, so the PPO update never runs the trunk at
  all — the update is a 248K-parameter MLP step. This is the same property
  `UNIFIED_BRAIN_BAKEOFF.md` §6 relies on, and it is destroyed the moment the
  trunk becomes trainable. It is why A4 costs ~4× less GPU than A2.

---

**A5 — `frozen_trunk_insulated_flow_expert`. π₀'s shape, transplanted.**
(Brief's candidate 2.)

Stage 1 as A4. Stage 2: the **existing, currently dead** `ActionExpert`
(4,615,696 parameters receiving zero gradient today [M]) cross-attends the
trunk's `hidden_states` with **`sg()` on the trunk's K and V** — the literal
mechanism of 2505.23705 [V] — and emits the action by conditional flow matching
with `flow_parameterisation = "x1"` and 4 integration steps. The choice of
`"x1"` is not taste: this repo measured it (T1.12 / `UnifiedBrainConfig`
docstring [C]) at 0.266 held-out error against velocity-parameterisation's
0.407–0.620, with integration steps 5–100 moving the result < 2 %.
PPO trains the expert and a state-independent `log_std`; the expert's
chunk-mean is the policy mean, and the whole 4-step decode is differentiable.

- **Hypothesis tested:** does an insulated flow-matching action expert beat a
  plain MLP head on the *same* frozen features (A4)? And — a Tier-3 question
  answered for free — can the 4.6M `action_expert` **earn its parameters** at
  all, or should it be deleted?
- **Changes:** `ActionExpert.forward` gains the stop-gradient on trunk K/V;
  `policy_mean` gains a flow-decode branch.
- **Params (cost):** **4,620,841** trainable (57.05M frozen resident) [C].
- **Named risk, declared before the run:** PPO through a multi-step flow decode
  is not standard practice and may simply not train. That is a legitimate FAIL
  for this arm and an honest VOID for nothing else — A5 failing does not void
  the bakeoff, because A0's reference gate carries the harness.

### Arms deliberately NOT included, with the reason

| considered | why not |
|---|---|
| **decoupled critic** (trunk carries policy only) — D1 option B | cos(∇pg, ∇vf) = 0.102 and vf/pg = 0.052 on the trunk [M]; T2.00 measured the ratio ≤ 2.71 falling to 0.20 [L]. Spending ~7 GPU-h on the weakest measured mechanism is not defensible. §6 pre-registers the trigger that promotes it to an arm. |
| **more compute end-to-end** — D1 option C | it *is* A2/A3, correctly framed as "more optimiser steps and a repaired update" rather than "more env-steps". Buying env-steps alone was already refuted by Rank 2. |
| **delete the trunk entirely** — D1 option D | out of scope by construction: flat locomotion is proprioception-sufficient, so no locomotion bakeoff can license it. §6 forbids that reading explicitly. |
| **LoRA on the trunk** (OpenVLA 2406.09246 [V]) | LoRA's win is *compute*, not quality (68.2 vs 69.7 [V]), and it presupposes pretrained weights. Revisit after A4 shows the pretrained trunk is worth adapting. |

---

## 4. The `Spec(...)`

Exact `experiments/registry_expansion.py` format. `D1.0` is the prerequisite;
`T2.21` is the bakeoff. Both parse (checked by `ast.parse` against a stub
`Spec`/`Budget`) and both carry all five of `hypothesis`, `falsified_by`,
`null_baseline`, `control`, `kills`.

*Id hygiene, per the `ME.11`/`ME.11.0` lesson:* `T2.03`–`T2.20` are already
taken, so the bakeoff is `T2.21`. The prerequisite is deliberately **not**
`T2.21.0` — that would make one id a prefix of the other and put the module
glob back in the situation `_module_for` had to be patched for. `D1.0` opens a
new namespace (`d1_0_*.py`) which is a prefix of nothing currently in the
registry; check that again before adding any `D1.x`.

```python
    # ── D1: is the trunk a motor controller? ────────────────────────────

    Spec("D1.0", 2, "The PPO update is not measuring its own dropout",
         hypothesis="With the control path in eval mode, two forward passes of "
                    "one batch are bit-identical, the PPO importance ratio at "
                    "an unmoved policy is exactly 1.0 for every sample, and "
                    "the recorded optimiser-step count per seed is within 10% "
                    "of the pre-declared target for every arm.",
         falsified_by="Any nonzero spread in the ratio at an unmoved policy, "
                      "OR any arm whose optimiser_steps differs from its "
                      "declared target by more than 10% — either way T2.21 "
                      "would again be comparing training setups, not "
                      "architectures.",
         null_baseline="The SHIPPED configuration, measured on the same batch: "
                       "2026-08-09 it gave ratio p05/p95 = 0.632/1.393, 20.3% "
                       "of samples outside clip_range=0.3, and eval-mode vs "
                       "train-mode action means differing by 27% of the mean's "
                       "own magnitude — all at zero policy change.",
         metric="ratio_spread_x_step_match", budget=Budget.CPU, seeds=3,
         depends_on=["T2.00"],
         control="The PRE-REPAIR configuration must FAIL this same check. A "
                 "repair whose control also passes was not repairing anything, "
                 "and the shipped numbers above are the pre-registered value "
                 "the control must reproduce.",
         kills="T2.21 before it costs a GPU-hour, and the interpretation of "
               "T2.01 v4 and T2.02 as evidence about ARCHITECTURE. Both were "
               "run with dropout live in rollout, update and evaluation "
               "(TrainingPipeline never calls .eval(); 36 Dropout modules at "
               "p=0.1), which is present in the trunk arm and absent from the "
               "SB3 arm.",
         notes="Also asserts the four cheap repairs R2-R5: obs statistics "
               "frozen at eval; mujoco_obs_dim 348 not 376 (Humanoid-v5 emits "
               "348, the pipeline padded 28 zeros); GAE bootstraps V(s_T) at "
               "the rollout boundary; and the full decimated learning curve is "
               "recorded. That last one is not cosmetic: T2.01 stored "
               "curve_seed0[:8], iterations 1-21 of 172, so the 'the curve "
               "PLATEAUED' claim that D1 rests on is not in the ledger."),

    Spec("T2.21", 2, "Where the trunk belongs in the motor path (D1 bakeoff)",
         hypothesis="At matched environment steps AND matched optimiser steps, "
                    "with the D1.0 repairs in force, at least one arm that "
                    "keeps the 57M trunk in or adjacent to the control path "
                    "(A2, A3, A4, A5) reaches within 1.5 sigma of the best "
                    "trunk-free arm (A0, A1) on Humanoid-v5 return.",
         falsified_by="Every trunk arm loses to the best trunk-free arm by "
                      ">=1.5 sigma with all arms above the learning gate. Then "
                      "the trunk is out of the MOTOR path, and D1 resolves to "
                      "option A with z gated off for flat locomotion. "
                      "Symmetrically, if A2 or A3 WINS, D1's premise was a "
                      "training bug and the trunk stays end-to-end.",
         null_baseline="A random-action policy on the same env and the same "
                       "evaluation episodes (T2.02 measured 110.9 +- 23.5). "
                       "PLUS, per arm, that arm's own UNTRAINED network — the "
                       "bar that matters, because T2.02's untrained MLP "
                       "already scored 2.74 sigma over random, so a 3-sigma "
                       "gate against random alone is nearly cleared by a "
                       "network that has never received a gradient.",
         metric="locomotion_return_at_matched_optimiser_steps",
         budget=Budget.GPU_LONG, seeds=3,
         depends_on=["D1.0", "T2.00", "T1.13"],
         control="THREE, in different directions. (a) A0, the SB3 reference, "
                 "must reach >=450 return; below that the harness or the step "
                 "budget is broken and every other number is uninterpretable "
                 "— VOID, not FAIL. (b) Every arm's UNTRAINED twin must stay "
                 "below the learning gate. (c) A SHUFFLED-z control for A4: z "
                 "drawn from a different episode must not IMPROVE the "
                 "controller. If shuffled z is as good as real z, the "
                 "controller is using z as a constant bias and A4's result is "
                 "about extra parameters, not about perception.",
         kills="Four of six arms, and one branch of D1. The survivor is the "
               "control architecture Jack ships; the losers are recorded in "
               "docs/DECISIONS_RESOLVED.md and deleted, not kept 'for later'. "
               "If A5 loses, the 4.6M ActionExpert has failed its only "
               "audition and goes to Tier-3 deletion.",
         notes="ARMS, cost declared in TRAINABLE PARAMETERS (Arm.cost; a TIE "
               "resolves by cost and bakeoff.py returns VOID if a tied arm "
               "leaves it undeclared): A0 sb3_reference 124707 (reference, "
               "SB3 MlpPolicy [128,128], separate actor/critic — the exact "
               "arm that scored 530.2 in T2.02). A1 mlp_no_z 218787 (our "
               "harness, separate 128/256 actor-critic on raw 348-d proprio, "
               "z slot hard-zeroed). A2 trunk_e2e_repaired 47327523 (57M "
               "trunk end-to-end, D1.0 repairs, tokenizer fed the RAW 348-d "
               "observation by documented joint layout instead of a 256-d "
               "LayerNormed projection). A3 trunk_e2e_difflr 47327523 (A2 "
               "with trunk lr 3e-6 vs head lr 3e-4, 100x). A4 "
               "frozen_pretrained_trunk_mlp_head 248491 (trunk pretrained by "
               "masked-motion prediction on T1.13's 2747 CMU+KIT clips, then "
               "eval() + stop-grad at the readout tokens per arXiv:2505.23705; "
               "k=8 readout tokens -> z in R^64; A1's controller with z live). "
               "A5 frozen_trunk_insulated_flow_expert 4620841 (A4's frozen "
               "trunk + the existing 4.6M ActionExpert cross-attending "
               "hidden_states with stop-grad on trunk K/V, conditional flow "
               "matching, flow_parameterisation='x1' per T1.12's measured "
               "0.266 vs 0.407-0.620 for velocity, 4 integration steps). "
               "A2 and A3 have IDENTICAL parameter cost by construction, so a "
               "TIE between exactly those two is reported as 'the trunk's "
               "trainability does not depend on the LR schedule' and both "
               "count as one option; it is NOT broken by an invented cost. "
               "MATCHED OPTIMISATION GATE: every arm >=40000 optimiser steps "
               "per seed and max/min across arms <=2.5 — T2.02 gave the trunk "
               "6240 and the MLP 99840, a 16x gap at 'matched' env-steps, "
               "which is why that comparison could not have been about "
               "architecture. PAIRED evaluation and IQM per arXiv:2108.13264. "
               "This spec CANNOT license D1 option D: flat locomotion is the "
               "one task where proprioception is SUFFICIENT "
               "(docs/research/UNIFIED_BRAIN_BAKEOFF.md finding 2), so a "
               "trunk-free win says the trunk is not a MOTOR CONTROLLER and "
               "says nothing about perception. UB.16 remains the spec that "
               "certifies the z channel."),
```

---

## 5. Declared costs

`experiments/bakeoff.py` resolves a TIE by `Arm.cost` and returns **VOID** if
any tied arm left it `None` — the `LESSONS.md` rule that a default of zero is
not "unknown". The unit is declared here, before the run.

> **Unit: trainable parameter count** — the parameters the optimiser updates.
> It is the natural unit for D1 (the question is literally "does 57M of trunk
> belong in the control path"), it is exactly reproducible, and it is not
> hardware-dependent the way GPU-seconds are. **Resident** parameters are
> reported alongside but are not the tie-break, because a frozen trunk that is
> shared with perception costs Jack nothing extra at the control interface.

| arm | trainable (**cost**) | resident | derivation |
|---|---|---|---|
| A0 `sb3_reference` | **124,707** | 124,707 | policy `348·128+128` + `128·128+128` + `128·17+17` = 63,377; value `348·128+128` + `128·128+128` + `128+1` = 61,313; `log_std` 17. **Matches `mlp_params` in the ledger to the digit** [L][C] — which is also the proof that SB3's `net_arch=[128,128]` builds *separate* actor and critic trunks (§1, Rank 5) |
| A1 `mlp_no_z` | **218,787** | 218,787 | policy 63,377 + value (`348·256+256` + `256·256+256` + `256+1` = 155,393) + `log_std` 17 [C] |
| A2 `trunk_e2e_repaired` | **47,327,523** | ≈ 56.9M | 47,466,019 measured gradient-receiving [M] (which already includes `obs_proj` and `log_std`) − `obs_proj` 325,888 [M] + per-body stem `348·512 + 18·512` = 187,392 [C]. `action_chunk_size` 16→1 removes a further 7,680; absorbed by the ±5 % tolerance |
| A3 `trunk_e2e_difflr` | **47,327,523** | ≈ 56.9M | identical to A2 by construction — a different LR changes no parameter |
| A4 `frozen_pretrained_..._head` | **248,491** | 57,300,627 | policy 71,569 + value 171,777 + readout adapter 5,128 + 17 [C] |
| A5 `frozen_..._flow_expert` | **4,620,841** | 61,672,960 | `ActionExpert` 4,615,696 [M] + readout adapter 5,128 + 17 [C] |

**Guard, pre-registered:** each arm asserts its *measured*
`sum(p.numel() for p in ... if p.requires_grad)` against the declared cost and
returns `Status.VOID` if it differs by more than **±5 %**. A declared cost that
is not the real cost is the `Arm.cost = 0.0` bug wearing a number.

---

## 6. The learning gate and the decision rule

Both fixed here, before any arm exists.

### 6.1 The learning gate — strengthened, and why it had to be

`run_bakeoff(..., learning_gate_sigma=3.0)` requires every arm to beat the
shared null by 3σ. **That gate is nearly cleared by a network that has never
received a gradient.** T2.02's own control metrics [L]:

```
untrained_mlp_mean 175.1   untrained_mlp_sigma 2.74     <- against a 3.00 gate
untrained_tr_mean  150.3   untrained_tr_sigma  1.19
random_mean        110.9   random_std          23.46
```

An arm scoring 3.1σ trained and 2.74σ untrained would pass a gate it had
learned essentially nothing to clear. This is the `LESSONS.md` family *"an
assertion made against a saturated quantity cannot fail"*, one level up. So
T2.21 pre-registers **both** conditions, and an arm must satisfy both:

1. **Gate A (the bakeoff's own, unchanged):** `sigma_over_null ≥ 3.0`, where
   σ = `max(arm seed std, null std)` — `run_bakeoff`'s existing rule.
2. **Gate B (new, the one that has teeth):**
   `(trained_mean − own_untrained_mean) ≥ 3 × max(arm seed std, untrained
   seed std, null std)`. Each arm is gated against **its own untrained twin**,
   not only against random actions.

Any arm failing either gate ⇒ the bakeoff is **VOID** and D1 stays open. That
is the whole point of the primitive and it is not negotiable to get an answer.

Additionally, before the gates are evaluated at all:

3. **Reference gate:** A0 mean return ≥ **450** (T2.02 measured 530.2 ± 59.0
   [L]; 450 is ≈1.4σ below). Failing it ⇒ **VOID**, harness or budget broken.
4. **Matched-optimisation gate:** every arm ≥ **40,000** optimiser steps/seed,
   and `max/min` across arms ≤ **2.5**. Failing it ⇒ **VOID**. (T2.02's ratio
   was 16.0 [C]; this is the gate whose absence made that run unable to test
   its own question.)
5. **Matched-experience gate:** every arm within ±10 % of **480,000** env-steps
   per seed. Failing it ⇒ **VOID** (T2.02's `step_match_ratio`, generalised to
   six arms).

### 6.2 The decision rule

```
run_bakeoff(spec=T2.21, arms=[A0..A5], null_run=random_policy,
            seeds=[0,1,2], learning_gate_sigma=3.0, margin_sigma=1.5,
            higher_is_better=True, ledger=ledger)
```

- **WINNER** — the best arm leads the runner-up by ≥ 1.5σ of the pooled spread.
  Implement it; archive the losers into `docs/DECISIONS_RESOLVED.md` with their
  numbers; delete the losing code.
- **TIE** — within 1.5σ ⇒ take the **cheapest tied arm by trainable
  parameters**. Two named exceptions, declared now:
  - a tie whose tied set is exactly **{A2, A3}** is reported as *"the trunk's
    trainability does not depend on the LR schedule"*; they have identical
    cost by construction and are treated as one option for the D1 answer.
  - a tie between **A1 and A4** is reported as *"z neither helps nor hurts on
    the proprioception-sufficient task"* — which is UB.16's **PASS** condition,
    not a defeat for the trunk. Cost picks A1 for the flat-locomotion
    controller; the trunk's place is then decided by UB.16, not here.
- **VOID** — any gate above fails. Fix the arm; do not decide. `kills` does not
  fire (`Status.VOID`, `protocol.py`).

### 6.3 The D1 answer, mapped from the verdict *before* the numbers exist

| outcome | D1 answer |
|---|---|
| A2 or A3 WINS, or ties the best trunk-free arm | **Option C/B.** The trunk was untrainable under the old harness, not architecturally wrong. It stays end-to-end. T2.01 re-runs under the repaired pipeline. |
| A4 wins or ties A1, A2/A3 lose by ≥1.5σ | **Option A — the recommendation.** Freeze the trunk, small head does control, `z` is the sensory channel. The trunk keeps perception, memory and language. |
| A5 beats A4 by ≥1.5σ | **Option A, with the π₀ head.** The insulated flow expert ships and the 4.6M `ActionExpert` has earned its parameters. |
| A1 beats A4 by ≥1.5σ | A **randomly-conditioned** `z` is actively harmful — VC-1's random-frozen result (20.4 % vs 47.4 % [V]) reproducing on us. `z` is gated OFF for control until UB.16 shows it carries information the controller needs. Ship A1's controller; the `z` slot stays wired and zeroed. |
| A0 beats A1 by ≥1.5σ | The harness is still worse than standard practice even after D1.0. **VOID for D1**; the next spec is about our PPO, not about the trunk. |
| any gate fails | **VOID.** D1 stays open. |

**One reading is forbidden in advance, whatever the numbers say:** no outcome
of T2.21 may be used to remove the trunk from Jack (D1 option D). Flat
locomotion is the single task where proprioception is *sufficient*
(`UNIFIED_BRAIN_BAKEOFF.md` Finding 2), so this bakeoff can only decide whether
the trunk is a **motor controller**. Whether it is a **binder** is UB.9–UB.16's
question and is measurable with no policy at all.

### 6.4 The pre-registered trigger that would add a seventh arm

The decoupled-critic arm (D1 option B) is excluded on measurement, so the
measurement is written down as a condition. **If D1.0 or T2.21 records, at any
point during a real run, either `cos(∇pg, ∇vf)` on `model.layers` **> 0.3** or
a vf/pg trunk gradient-norm ratio **> 3.0**, a `trunk_policy_only` arm is added
and T2.21 re-runs.** At initialisation those read **0.102** and **0.052** [M],
and T2.00's whole-run maximum was **2.71** [L]. Naming the trigger is the
difference between "we decided not to test it" and "we decided not to test it
*yet*, and here is what would change our mind".

---

## 7. Compute estimate — free compute only

Constraints (`SYSTEM.md`): 4 shared ARM cores here at `nice 19` under ~1.5 GB;
Kaggle 30 h/week P100, ~23 h remaining this week; Colab T4 elastic. **No paid
compute is proposed anywhere in this document.**

Throughput figures are measured, not guessed: T2.02's trunk arm ran 638,976
env-steps in 100 min ⇒ **~106 env-steps/s** on a P100 including its update
[L][C]; decomposing 38 s/iteration into ~6 s rollout + ~32 s for 40 updates
gives **~0.8 s per optimiser step at minibatch 512**, i.e. **~0.1 s at
minibatch 64** [C]. The SB3 MLP probe needed ~27 min/seed for 640K steps on a
slower CPU [C].

### CPU stage — can VOID the whole bakeoff for free

| item | where | estimate |
|---|---|---|
| D1.0 repairs + falsification + its must-fail control | 4 ARM cores | **0.5 CPU-h** |
| A0 `sb3_reference`, 3 seeds × 480K steps | 4 ARM cores | **1.3 CPU-h** |
| A1 `mlp_no_z`, 3 seeds × 480K steps, our harness | 4 ARM cores | **1.8 CPU-h** |
| random null + 6 untrained twins, 3 seeds, paired eval | 4 ARM cores | **0.4 CPU-h** |
| **CPU total** | | **≈ 4 CPU-hours** |

If A0 misses 450, or A1 trails A0 by ≥1.5σ, **the GPU stage never runs.**

### GPU stage

| item | backend | estimate | note |
|---|---|---|---|
| trunk masked-motion pretraining, 3 seeds (shared by A4 and A5) | P100 | **1.7 GPU-h** | ~20K steps @ batch 64, 2,747 clips |
| A2 `trunk_e2e_repaired`, 3 seeds | P100 | **7.1 GPU-h** | 40K updates @ mb 64 ≈ 1.1 h/seed + 1.26 h/seed rollout |
| A3 `trunk_e2e_difflr`, 3 seeds | P100 | **7.1 GPU-h** | identical shape to A2 |
| A4 frozen trunk + MLP head, 3 seeds | T4/P100 | **1.6 GPU-h** | **z cached in the rollout buffer ⇒ the update never runs the trunk** |
| A5 frozen trunk + flow expert, 3 seeds | T4/P100 | **2.5 GPU-h** | + 4-step differentiable decode through 4.6M |
| evaluation, controls, shuffled-z | T4 | **0.5 GPU-h** | eval-only |
| subtotal | | **20.5 GPU-h** | |
| **+25 % slack** (preemption, resume, one re-run) | | **≈ 26 GPU-h** | |

**Scheduling against the real quota.** ~23 h remain this Kaggle week, so the
work is staged and each stage is independently useful:

- **Week 32 (now):** CPU stage (4 CPU-h, no quota) → pretraining 1.7 + A4 1.6 +
  A5 2.5 + eval 0.5 = **6.3 GPU-h**. This alone answers *"does a frozen
  pretrained trunk plus a small head match a plain MLP?"* — the recommended
  option — and leaves ~17 h.
- **Week 32 remainder or Week 33:** A2 + A3 = **14.2 GPU-h**, the
  "was-it-trainability" half.

Everything checkpoints every 15 minutes and resumes (T0.04/T0.05 PASS), which
is what makes a 12 h Kaggle cap and Colab teardown survivable, and **one GPU
submission per spec** guarded by a module-level cache — the 11-GPU-hour scar in
`t2_01_locomotion_vs_random.py:186-191`.

**Cheaper if the budget tightens:** drop A3 (−7.1 GPU-h). Its hypothesis
(differential LR) is a *subset* of A2's — if A2 clears the gate, A3's question
is moot; if A2 fails, A3 is the natural retry. Sequencing A2 before A3 makes
A3 conditional and is the recommended order.

---

## 8. How the senses reach the controller in every arm

`GOAL.md` requires *all senses, one brain, involved together*. An arm that
severs perception from action violates it. Each arm's answer, stated against
the **UB.16 contract** (`trunk → k readout tokens → z → controller`):

**A0 `sb3_reference` — the honest exception.** A0 has no sensory path; it is a
proprioception-only reference whose job is to indict the harness, and it is
**declared ineligible to be adopted as Jack's architecture**. If A0 wins
outright the conclusion is *"our harness is worse than SB3"*, which is a
statement about `TrainingPipeline`, not about Jack's senses. Recording this in
the spec is what stops the reference arm from quietly becoming the design.

**A1 `mlp_no_z` — the same controller as A4 with the channel muted.** A1's
policy and value networks take `[proprio(348) ⊕ z(64)]`; A1 hard-zeroes `z`. So
**adopting A1 is adopting A4 with a runtime gate closed, not a different
architecture** — the wiring, the tensor shapes and the checkpoint layout are
identical, and turning `z` back on for a task where it matters is a boolean, not
a re-architecture. Meanwhile the senses still reach Jack: the trunk continues to
consume vision, audio, touch and language and is measured on that by UB.9–UB.14,
which need no controller at all. What A1 asserts is narrow and testable: *on the
one task where proprioception is provably sufficient, the sensory channel is
switched off*. UB.16 is the spec that decides when it switches on.

**A2 / A3 `trunk_e2e_*` — maximal coupling.** The controller *is* the trunk;
every modality token enters the same 37-token self-attention sequence that emits
the action token. Perception and action share every parameter. This is the
strongest possible reading of "one brain" and the arms exist precisely so that
reading gets a fair test. Risk to name: with the whole trunk in the PPO graph,
the locomotion objective rewrites the perceptual representation — which is
exactly the destruction RT-2 (2307.15818 [V]) measured at 11 points and
Knowledge Insulation (2505.23705 [V]) exists to prevent. If A2/A3 win on
locomotion, UB.11's ablation matrix must be re-run immediately to check that
walking did not eat the senses.

**A4 `frozen_pretrained_trunk_mlp_head` — the contract, literally.** Vision,
audio, touch, language and proprioception enter the trunk's per-modality stems;
the trunk fuses them; **k = 8 read-only readout tokens** (Octo, 2405.12213 [V])
compress the fused state to `z ∈ R^64`; the controller reads
`[proprio ⊕ z]` at control rate. **`z` is the entire sensory channel and the
only one**, which is what makes UB.16's asymmetry test meaningful: zero `z` and
the perception-dependent tasks must degrade while flat walking must not. The
stop-gradient (rather than `requires_grad=False`) is what keeps the trunk
*trainable by its own objective* — the binding losses of UB.10 — while the
controller cannot corrupt it. Senses are not severed; they are **routed and
metered**.

**A5 `frozen_trunk_insulated_flow_expert` — the same channel, richer read.**
Identical routing, except the expert cross-attends the trunk's **full
`hidden_states` sequence**, not just the pooled `z`. So A5 has *more* sensory
bandwidth into the controller than A4: it can attend to individual modality
tokens rather than a 64-d summary. The stop-gradient on trunk K/V is what makes
that safe. If A5 wins, Jack's controller reads the senses at token granularity,
which is the closest arrangement in this document to π₀'s.

**Every arm ships the same slot.** All six controllers take an input of shape
`348 + 64`, and the `z` half is populated by A4/A5, zeroed by A1, absent-by-
construction in A0, and superseded in A2/A3 (where the controller reads the
trunk directly). No arm can win in a way that requires deleting the sensory
interface.

---

## 9. What this makes the machine better at

Per `SYSTEM.md` — *is the machine better than I found it?* Four proposals, each
a guard rather than a fix, and each derived from something that actually
happened:

1. **The untrained-twin learning gate (Gate B, §6.1).** T2.02's own control
   recorded an untrained MLP at **2.74σ** against a 3.0σ gate [L] — the gate
   that VOIDed T2.02 was itself ~91 % clearable by a network with no training.
   Gating each arm against its own untrained twin closes it. This belongs in
   `bakeoff.py` as an optional `untrained_run` callable, not only in T2.21.

2. **A rollout-determinism assertion (D1.0's core).** `assert` that two
   `no_grad` forwards of one batch are bit-identical at rollout time. Three
   lines, and it would have caught Rank 1 before 11 GPU-hours were spent
   measuring the consequences. It generalises: *any* stochastic layer left
   enabled in an inference path is the same bug.

3. **Stop truncating the curve.** `curve_seed0 = ...[:8]` [C] discarded
   iterations 22–172, and the plateau claim that D1 rests on is therefore not
   in the ledger (Rank 6). Record a decimated full curve (≤ 200 points spanning
   the whole run). This is the `LESSONS.md` rule *"report per partition, gate on
   the minimum"* applied to time.

4. **Declared cost must equal measured cost.** `Arm.cost` is now `None` by
   default rather than `0.0`, which fixed the *undeclared* case. It does not fix
   the *wrong* case. T2.21 asserts measured trainable parameters against the
   declared cost at ±5 % and returns VOID otherwise.

And one **correction to the record**, which the owner should be aware of before
answering D1: `docs/DECISIONS_NEEDED.md` presents the three runs as *"EVIDENCE,
three independent runs at matched env-steps"*. They were matched on env-steps
and **not** matched on optimiser steps (16.0×, §Rank 2), and the trunk arm ran
with dropout live in rollout, update and evaluation while the MLP arm did not
(§Rank 1). **T2.02 was VOID for a second reason it never recorded.** The
evidence is not wrong, but it is weaker than the file claims, and D1's option C
("keep training end-to-end … not supported by evidence") is currently
*untested* rather than refuted.

---

## 10. What we refuse to claim

- **That the trunk is a bad motor controller.** Nothing in the ledger tests
  that. The two runs that appear to were confounded by dropout, by a 16×
  optimiser-step gap, and by a shared-vs-separate-critic difference nobody
  declared.
- **That the trunk is a good motor controller.** Equally untested. A2 and A3
  exist because it might be, not because it is.
- **That freezing helps.** The literature's frozen-backbone result presupposes
  pretrained knowledge worth protecting (2505.23705, 2307.15818 [V]), and
  Jack's trunk has none. VC-1 [V] measured random frozen features at 20.4 % vs
  47.4 % from scratch. This is why A4 pretrains before it freezes, and why the
  A1-vs-A4 contrast is the real test rather than a formality.
- **That any locomotion result decides whether the senses are fused.** Flat
  locomotion is proprioception-sufficient. T2.21 cannot see fusion and is
  forbidden from being read as if it could.
- **That the effect will be large.** `UNIFIED_BRAIN_BAKEOFF.md` §7 calibrated
  this: the closest measured analogue produced R² of 0.049 / −0.001 / 0.187
  with one of three negative. Design for a small effect (paired seeds, IQM,
  bootstrap CIs, 2108.13264 [c]) or the experiment cannot see the thing it is
  looking for.
- **Any number tagged UNVERIFIED in §2**, specifically: the "frozen PVRs
  underperform from-scratch on proprioception-sufficient tasks" framing of
  CortexBench; a verbatim "64–256 units, 2 layers" recommendation in
  Andrychowicz et al.; and any standalone "small networks are enough for
  MuJoCo/DMC" result. The Knowledge Insulation language-following percentages
  (~70 / ~45 / ~30) are read off a figure; its 7.5×, its 20 % overhead and its
  LIBERO table are exact.
