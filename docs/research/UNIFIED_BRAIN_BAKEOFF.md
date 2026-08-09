# The Unified-Brain Bakeoff — proving the senses are FUSED, not concatenated

> Researched 2026-08-09. Serves GOAL.md: *"ALL SENSES COMBINED IN UNISON …
> everything must be into one brain which processes … We need it to use these
> senses and learn the world."*
>
> Companion to `UNIFIED_BRAIN.md` (which chose a recipe) and
> `MULTIMODAL_BINDING.md` (which audited the code). This document does the one
> thing neither did: it specifies **how the claim could be proven false**, and
> stages the work so most of it dies on 4 CPU cores before touching a GPU.

---

## 0. The two findings that reframe the problem

**Finding 1 — the unison branch is currently unreachable.** Every UB spec is
transitively blocked by a FAILED locomotion spec:

```
UB.1 → T4.01 → T3.02 → T2.01 (FAIL)     and    T4.01 → T3.01 → T2.03 (NOT_RUN)
UB.2..UB.8 → UB.1
```

`experiments/ledger.json` records T2.01 FAIL and T2.02 FAIL. `protocol.py`
`blocked_by()` will therefore return BLOCKED for UB.1 and, by chaining, for the
entire unison ladder. **The least-tested claim in the project is also the only
claim that cannot currently be tested at all**, and the reason is an accident of
dependency wiring, not a fact about fusion.

This is fixable and the fix is the main structural proposal here: *binding is a
perception claim, not a control claim.* Almost everything worth knowing about
whether Jack's senses are fused can be measured by supervised probes on his
percepts, with **no policy, no RL, and no MuJoCo control loop**. The specs below
(UB.9–UB.16) depend only on `PG.1`, `PG.5`, `T1.03`, `T1.06` and two new
fixtures — all PASS or cheap — and they are deliberately parented away from
locomotion. Only the embodied capstone (UB.15) needs a working controller.

**Finding 2 — the D1 evidence says nothing about binding, and the current
argument misuses it.** D1 observes that a 125K MLP beat the 57M trunk at
Humanoid-v5 locomotion. Flat-ground locomotion is the single task in Jack's
world where **proprioception is sufficient**: the 348-dim proprio vector is a
complete state description for the reward. A task where fusion *cannot* help is
not evidence about fusion. Judging the trunk on it is a category error in both
directions — it neither condemns nor vindicates the trunk as a binder. §5 makes
the architecture D1-agnostic so the outcome of D1 changes the wiring diagram but
not a single experiment in this document.

---

## 1. Survey

Citation hygiene: IDs marked **[V]** were fetched from arxiv.org during this
research pass and the title/authors/date confirmed. IDs marked **[c]** are
carried over from `UNIFIED_BRAIN.md` / `MULTIMODAL_BINDING.md` and were **not**
re-verified here — treat their numbers as second-hand until someone checks them.
Nothing below is cited for a number I did not see.

### 1.1 Shared-trunk token approaches — architecture is necessary, not sufficient

The unified-token-stream pattern (every modality becomes tokens; one
self-attention stack reads all of them) is the field's default: Gato
(2205.06175 [c]), Unified-IO 2 (2312.17172 [c]), Octo's readout tokens
(2405.12213 [c]), HPT's shared trunk over per-embodiment stems (2409.20537 [c]),
π0 / π0.5 (2410.24164, 2504.16054 [c]), SmolVLA (2506.01844 [c]). Its virtue is
that cross-modal interaction is *structurally possible and inspectable* — you
can read the attention map and ask whether any mass crossed a modality boundary.
The Perceiver family (2103.03206, 2107.14795 [c]) is the compute-poor variant:
learned queries cross-attend a large token pool at O(NM), which is the right
shape at Jack's scale. Flamingo-style gated cross-attention (2204.14198 [c]) is
the asymmetric case whose gates can close to zero — "coexistence" with a
pathway, not fusion.

But structural possibility is not use. **"Not All Features Are Created Equal: A
Mechanistic Study of Vision-Language-Action Models" (2603.19233 [V])** is the
decisive negative result, and its numbers are worth quoting exactly. Across six
models, 80M–7B params, 394,000+ rollout episodes: *"The visual pathway dominates
action generation across all architectures: injecting baseline activations into
null-prompt episodes recovers near-identical behavior."* Language sensitivity is
a property of the **task**, not the model: X-VLA on libero_goal drops 94%→10%
under a wrong prompt, while libero_object scores 60–100% *regardless of the
prompt*. In all three multi-pathway architectures (π½, SmolVLA, GR00T) the
expert and VLM pathways occupied **separable activation subspaces** — functional
dissociation, not integration — and causal ablation showed **28–92% zero-effect
rates**, i.e. in some architectures nine features in ten can be ablated with no
behavioural consequence.

The operational lesson: *encoded is not used*, and *architecture does not
predict use*. A test that measures encoding (a probe, a retrieval score) has not
measured binding. Only an intervention has.

**ReViP (2601.16667 [V])**, "Mitigating False Completion in Vision-Language-Action
Models with Vision-Proprioception Rebalance", is the same lesson from the token
side: proprioception out-competes vision for attention mass, and the failure mode
is *false completion* — the policy reports success while visibly failing.
Jack's token budget is the worst case of this: `JointTokenizer.forward()` emits
17 joint tokens + 1 body token against **one** token per non-proprio sense
(`UnifiedBrain.py:4204-4310`, six `.unsqueeze(1)` calls). Vision is 1/41 of the
sequence; proprioception is 18/41. Whatever else the bakeoff varies, **token
count per modality must be equalised across arms or the comparison is a
comparison of token budgets.**

**Multi-Modal Manipulation via Multi-Modal Policy Consensus (2509.23468 [V])**
is the credible *non-trunk* alternative and belongs in the bakeoff for exactly
that reason. It factors the policy "into a set of diffusion models, each
specialized for a single representation (e.g., vision or touch)" plus "a router
network that learns consensus weights to adaptively combine their
contributions", and reports outperforming feature-concatenation baselines on
scenarios requiring multimodal reasoning, plus robustness to sensor corruption.
It also ships a diagnostic we should steal: *perturbation-based importance
analysis* showing reliance shifting between modalities across task phases. If
this arm wins, "one brain" as a single trunk is the wrong shape and we should
say so.

### 1.2 Cross-modal masked prediction — the binding FORCE

MultiMAE (2204.01678 [c]) established multi-modal masked pretraining.
**M3L (2311.00924 [V])** — actual title *"The Power of the Senses:
Generalizable Manipulation from Vision and Touch through Masked Multimodal
Learning"*, Sferrazza, Seo, Liu, Lee, Abbeel — is the direct robotics evidence:
masked-autoencoding over vision+touch, learned jointly with the policy,
"improve[s] sample efficiency, and unlock[s] generalization capabilities beyond
those achievable through each of the senses separately", and notably
*"representations learned in a multimodal setting also benefit vision-only
policies at test time"*. That last clause is the signature of real binding: the
touch channel reshaped the vision representation, and the reshaping survived
touch's removal.

The cleanest ablation of *which* masked objective binds is
**2410.16424 [V]** ("Promoting cross-modal representations to improve
multimodal foundation models for physiological signals", Fang et al., Oct 2024).
Three findings, all directly transferable: (i) **cross-modal reconstruction
objectives are important for successful multimodal training, because they force
the model to integrate across modalities** — joint masking alone is weaker;
(ii) **input-space modality dropout improved downstream performance**;
(iii) **contrastive learning approaches proved less effective**. They verify the
mechanism the way we should: by showing the pretraining produced **more
cross-modal attention** and more distributed modality representations.

**Kepler-Encoder-v0.1 (2607.13522 [V])** is the closest thing to a template for
Jack — a small multimodal robot encoder over vision + proprioception +
force/torque, trained with **learned-query cross-attention and masked cross-modal
prediction**, motivated by the observation that cameras alone predict force
poorly (R² ≤ 0.10). It also calibrates expectations, and this is the most
useful thing in it: `UNIFIED_BRAIN.md` records its fused-latent force-prediction
R² as 0.049 / −0.001 / 0.187 across three robots against a compute-matched
vision-only control at 0.010 / −0.019 / 0.067, paired t-test p ≤ 0.012 [c].
**One of three robots was negative. The effect is real, statistically clean, and
small.** Any bakeoff that expects a large effect from binding has mis-specified
its own success criterion.

MMP (2410.03010 [V]) supplies the missing-modality mechanism: rather than
zeroing an absent modality, *"learn to project available input modalities to
estimate the tokens for the masked modalities"*, so one model handles any
missing-modality combination. This matters for the ablation matrix (§4): a
zeroed input is off-manifold, so "performance dropped when I zeroed vision" can
mean "vision is load-bearing" *or* "you handed the model an input it has never
seen". A learned `[MISSING-m]` token removes the confound.

### 1.3 Contrastive alignment — right for retrieval, conditional for control

ImageBind (2305.05665 [c]) / LanguageBind (2310.01852 [c]) build a shared
retrieval geometry from *paired but independent* samples. Jack's modalities are
not paired-but-independent: they are **synchronous** — the same contact event,
the same millisecond, in every channel. InfoNCE on synchronous streams has a
structural pathology: temporally adjacent windows from the same episode are
labelled negatives while being near-identical, so the loss spends its capacity
separating things that should not be separated. That is the same false-negative
problem TMR (2305.00976 [c]) fixes by filtering negatives above a similarity
threshold, and the same reason SigLIP's pairwise sigmoid loss (2303.15343 [c])
with a bias initialised to the negative-heavy prior is preferable to softmax
InfoNCE at the batch sizes a T4 permits.

Evidence that alignment moves *action*, not just retrieval, is mixed and the
mixture is informative:

- **Against**, at global-embedding granularity: 2410.16424 [V] found contrastive
  objectives "less effective" than cross-modal reconstruction for downstream
  tasks. Global InfoNCE discards exactly the time-local complementary detail a
  controller needs.
- **For**, when the contrastive target is *state-relevant rather than semantic*:
  **RS-CL (2510.01711 [V])**, "Contrastive Representation Regularization for
  Vision-Language-Action Models", aligns representations "more closely with the
  robot's proprioceptive states by using relative distances between the states
  as soft supervision", reaching 69.7% on RoboCasa-Kitchen and lifting a real
  robot from **45.0% → 58.3%**.

The synthesis for Jack: contrastive alignment earns a bakeoff arm, but the
positives/negatives must be defined by **physical state proximity** (same
contact event / nearby body state), not by episode identity. And its keep/kill
gate is a *task-success* delta, with retrieval only as the sanity check that the
loss trained at all (UB.13).

### 1.4 Modality dropout, and why it is not optional

ModDrop (1501.00102 [c]) is the origin. The load-bearing modern result is
audio-visual speech: **Robust Self-Supervised AVSR (2201.01763 [V])**, built on
AV-HuBERT (2201.02184 [V]), reports on LRS3 in babble noise **28.0% → 14.1% WER
vs prior SOTA using <10% of the labels (433 h → 30 h)**, and **reduces an
audio-only model's WER by over 75% (25.8% → 5.8%)** on average. Modality dropout
is the standard ingredient in this line precisely because without it the audio
stream — which is nearly sufficient in clean conditions — starves the video
stream, and the model never learns to lip-read. (Note: the specific dropout
ablation table lives in the papers' bodies, not the abstracts I fetched; treat
"dropout is *why* it uses lip video" as the field's working account rather than
a number I verified.) 2410.16424 [V] independently found **input-space** modality
dropout to help across downstream tasks.

Jack's dominant modality is **proprioception**, not language or vision — the
inverse of the VLM case. 348 clean, noise-free, perfectly-correlated-with-action
dimensions against a 10-scalar touch vector and a 2-channel audio stream. Any
arm without dropout should be expected to collapse onto proprio, and the
bakeoff's job is to measure *how far*, not to assume it.

### 1.5 Modality collapse — the mechanism is now known

**"A Closer Look at Multimodal Representation Collapse" (2505.22483 [V],
ICML 2025 Spotlight)** identifies the mechanism: collapse occurs when *"noisy
features from one modality are entangled, via a shared set of neurons in the
fusion head, with predictive features from another"*, masking the second
modality's contribution. Their fixes are cross-modal knowledge distillation
(which works by *"freeing up rank bottlenecks in the student encoder, denoising
the fusion-head outputs"*) and an explicit basis-reallocation algorithm.

The architectural consequence for Jack is sharp and currently violated: with
**one token per sense sharing a 512-d fusion head**, entanglement is maximal —
there is no room for modality-specific subspaces. Multi-token-per-modality is
not a nicety, it is the collapse mitigation.

**BalanceBenchmark (2502.10816 [V])** formalises evaluation "from three
perspectives: performance, imbalance degree, and complexity" and, importantly,
finds no mitigation method wins uniformly — so a balancing hyperparameter cannot
be chosen a priori, only measured. **Masked Imitation Learning (2209.07682 [V],
Hao, Wang, Cao, Wang, Cui, Sadigh)** attacks the dual failure — *state
over-specification*, where "the state contains modalities that are not only
useless for decision-making but also can change data distribution across
environments" — with a binary modality mask learned by bi-level optimisation.
That learned mask is an excellent *diagnostic* for Jack even if we never adopt
the method: **train the mask, read which senses it keeps.** A sense the mask
zeroes out is decorative, and this is a much cheaper attribution than Shapley.

### 1.6 Audio and touch in robotics — where the gains actually come from

**ManiWAV (2406.19464 [V])**, Liu, Chi, Cousineau, Kuppuswamy, Burchfiel, Song:
*"Audio signals provide rich information for the robot interaction and object
properties through contact… This information can surprisingly ease the learning
of contact-rich robot manipulation skills, **especially when the visual
information alone is ambiguous or incomplete**."* **Audio-VLA (2511.09958 [V])**,
Wei et al., adds contact audio to a DINOv2+SigLIP+AudioCLIP+Llama2 VLA, and —
directly relevant to us — **enhanced its simulators with collision-based sound
generation** and introduced a *Task Completion Rate* metric for perceiving
dynamic processes, reporting improvement over vision-only in LIBERO, RLBench and
real hardware.

The consistent shape of the finding across See-Hear-Feel (2212.03858 [c]),
ManiWAV and Audio-VLA is: **audio pays when vision is occluded or ambiguous, and
approximately nothing otherwise.** A binding test that leaves vision unoccluded
is testing on the regime where the literature predicts no effect. This directly
motivates the design in §3.

For touch, the honest constraint stands: Tactile-VLA (2507.09160 [c]),
Sparsh (2410.24090 [c]) and the rest operate on tactile *images* or 6-axis F/T.
Jack has 10 scalars. Those can carry contact detection, foot-strike timing and
left/right load asymmetry; they cannot carry texture, slip or in-hand pose, and
no test should assume them. Kepler's motivation (2607.13522 [V]) — cameras alone
predict force at R² ≤ 0.10 — is the positive case for keeping the channel.

### 1.7 Tasks that are impossible without fusion — prior art

SoundSpaces (1912.11474 [V], Chen, Jain, Schissler, Amengual Gari, Al-Halah,
Ithapu, Robinson, Grauman) established audio-visual navigation to sounding
objects with policies trained "end-to-end from a stream of egocentric
audio-visual observations". **Semantic Audio-Visual Navigation (2012.11583 [V],
Chen, Al-Halah, Grauman)** is the sharper design: "objects in the environment
make sounds consistent with their semantic meaning (e.g., toilet flushing, door
creaking) and **acoustic events are sporadic or short in duration**" — the agent
must reach the goal *after the sound has stopped*, which forces the audio event
to be bound to a visual/spatial representation rather than tracked reactively.

That is the template. §3 builds a version whose unimodal impossibility is
*analytically guaranteed by the fixture's own pan law* rather than empirically
hoped for.

### 1.8 Statistics — the part that decides whether any of this counts

**Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep RL at the Edge of the
Statistical Precipice" (2108.13264 [V])**: with few runs, point estimates of
mean/median are unreliable; report **interval estimates**, **performance
profiles**, and **interquartile mean** as the robust aggregate, via `rliable`.
At Jack's 3–5 seeds every bakeoff comparison must be a **paired** comparison
(same seed, same data order, same eval episodes across arms) with a stratified
bootstrap CI on the *paired difference*. An unpaired 3-seed comparison of two
architectures on this budget can resolve almost nothing.

---

## 2. What the measurement problem actually is, and the ladder that solves it

The four proposals in the brief are all correct and all individually
insufficient. Ranked by strength, with what each cannot show:

| # | Evidence | Establishes | Cannot show |
|---|---|---|---|
| 1 | **Linear probe / retrieval** on the fused latent | information is **encoded** | that it is used (2603.19233: 28–92% zero-effect rates) |
| 2 | **Ablation Δ** at test time | the model is **sensitive** to the channel | whether Δ is information loss or off-manifold shock |
| 3 | **Counterfactual swap-flip** — intervene on one modality so the *correct answer changes*, and require the output to change **in the predicted direction** | the channel is **causally read** | that the two channels are combined rather than one gating the other |
| 4 | **Synergy gap** — beat the unimodal late ensemble | joint computation over ≥2 channels **exists** | that it exists on the task you care about |
| 5 | **Pure-synergy task** where every unimodal null is at chance **by construction** | fusion, unconditionally | scale, generality, transfer |

Three sharpenings the brief's proposals need:

**(a) Ablation needs a placebo column.** Add a **PLACEBO MODALITY**: a channel of
pure noise, same token count, same encoder capacity, same dropout rate, wired in
exactly like a real sense. Its ablation Δ across seeds *is* the empirical null
distribution for "decorative". A sense whose Δ is not significantly above the
placebo's is decorative — and unlike a hand-set threshold, this null is
re-estimated every time the architecture changes. This is the cheapest
high-value addition in this document. It also catches the opposite error: if the
placebo column is *large*, your ablation procedure is measuring distribution
shock, not information, and every other column is uninterpretable.

**(b) Ablation must be four perturbations, and swap is the primitive.**
*zero* (off-manifold), *matched noise* (on-manifold marginals, no content),
*time-shuffle within episode* (destroys temporal binding, preserves everything
else), *cross-episode swap* (destroys cross-modal correspondence, preserves both
marginals and temporal statistics). **Cross-episode swap is the only one that
isolates correspondence**, which is what "binding" means. Load-bearing requires
all four to hurt; swap-only hurting is still a valid, weaker claim.

**(c) "Beats the best single modality" is not synergy, and the right null is
cheap.** Define, at matched data and eval:
- `U_m` = a model on modality *m* alone,
- `E` = the **unimodal late ensemble** — the `U_m` trained independently, their
  *predictions* averaged (logit mean). `E` is structurally incapable of
  synergy: no parameter ever sees two modalities jointly.
- `F` = the fusion model.

Then `F > max_m U_m` is trivially achievable by redundancy. **`F > E`, paired,
across seeds, is the synergy gap and it is the operational definition of "one
brain".** `E` costs 5 tiny models; it should be computed for every arm, on every
task, forever.

Two free mechanistic diagnostics ride along at zero cost:

- **Cross-modal attention mass**: per trunk layer, the fraction of attention mass
  on keys of a *different* modality than the query. ≈0 means the trunk has
  partitioned into covert late fusion — UB.7's control, obtained by reading a
  tensor instead of retraining. Necessary, not sufficient (attention can be
  nonzero and useless), so it is a red flag, never a claim. Precedent:
  2410.16424 [V] validated its pretraining this way.
- **Learned modality mask** (2209.07682 [V]): fit a binary mask by bi-level
  optimisation and read which senses survive. Cheap attribution, no Shapley.

---

## 3. THE BINDING TEST — "Heard, Not Seen" (HNS)

A task with **exactly one bit of pure synergy and nothing else**. In PID terms
(2302.12247 [V]): `I(audio; Y) = 0`, `I(vision; Y) = 0`, `I(audio, vision; Y) = 1 bit`.
It is physical XOR. Built entirely from `playground.py` + `ContactAudio.py` as
they exist.

### 3.1 The construction

Two candidate objects, **spheres of different radii but identical mass**, placed
at **mirrored azimuths** `θ` and `π − θ` relative to the listener, at **equal
radius** and **equal height**. At `t_event` one of them — chosen uniformly at
random, independent of everything else — is released and strikes the floor. Jack
must say which one fell.

Why each single sense is at chance, **by construction, not by hope**:

- **Audio alone is at chance on location.** `ContactAudio.py:26-28` pre-registers
  `pan p = -sin(azimuth)`. `sin(θ) = sin(π − θ)`, so the two candidates produce
  **bit-identical stereo pan**. The module's own docstring states it: *"Panning
  encodes left/right ONLY: front-back disambiguation needs ITD/spectral cues."*
  The impossibility is a documented property of the fixture, checkable to
  floating-point tolerance rather than argued.
- **Audio alone carries identity but not position.** `fundamental(gid) =
  clip(180 / char_size, 80, 4000)` is a deterministic bijection from radius to
  f0, so the ring says *which size fell* — and nothing about where that size is.
- **Vision alone is at chance.** The scene frame is captured **before** the
  event and is the model's only visual input; it shows both candidates, their
  sizes and their positions, and contains zero information about which was
  released. Post-event visual leakage is excluded by construction because vision
  never updates.
- **Proprioception and touch are at chance.** Jack is not in contact with either
  object. This is deliberate: it removes his dominant modality from the task
  entirely, which is exactly the condition under which collapse cannot hide.
- **The unimodal late ensemble `E` is at chance.** Averaging two chance
  predictors is a chance predictor. There is no redundancy and no uniqueness to
  harvest. Every point above 50% is synergy.

Solving it requires binding **f0 → radius** (audio) to **radius → position**
(vision). Ground truth is free from `mj_data`.

**Verified against the code, 2026-08-09** (not asserted — run before writing
this section, using `ContactAudio`'s own listener-frame trig):

```
theta= 25 deg  pan_A=-0.42261826174069944  pan_B=-0.42261826174069950  |diff|=5.6e-17
theta= 40 deg  pan_A=-0.64278760968653925  pan_B=-0.64278760968653947  |diff|=2.2e-16
theta= 60 deg  pan_A=-0.86602540378443860  pan_B=-0.86602540378443849  |diff|=1.1e-16
                                                    (equal radius 2.000000 m)

sphere r=0.07 -> char_size 0.070 -> f0 2571.4 Hz
sphere r=0.16 -> char_size 0.160 -> f0 1125.0 Hz      separation 2.29x
floor plane   -> char_size 2.154                       -> candidate is the voiced geom
```

The pan difference between the two candidates is **at float round-off**, so the
audio-only null is at chance to machine precision rather than to within
experimental error. Both radii sit inside the playground's existing
`object_size_range = (0.06, 0.18)`, so no fixture geometry change is needed —
only equal mass, equal radius from the listener, and mirrored azimuth.

One further legitimate spectral cue falls out of the synth and is worth
recording so nobody later mistakes it for a leak: `_voice()` drops partials at
or above `0.45 * sr = 7200 Hz`, so the r=0.07 sphere rings with 2 audible modes
and the r=0.16 sphere with 3. That is a *spectral* identity cue, which is the
intended channel. It is **not** an amplitude cue, because `_voice` renormalises
by the total gain of the *included* modes (`sig *= e.amp / total_gain`,
`ContactAudio.py:165-166`). PG.7's amplitude probe is the empirical check.

### 3.2 The confounds, and how each is closed

These are not hypothetical; each is a live leak in the current synth and each
must be certified closed *before* the test is trusted. That certification is its
own spec (PG.7), on the PG.5 precedent.

| Leak | Mechanism in code | Closure | Pre-registered check |
|---|---|---|---|
| **Amplitude → size** | `amp = min(1, AMP_K·sqrt(F))`; impact force scales with mass | both candidates **equal mass**, **equal drop height** | audio-only logistic probe on band energies must be ≤ chance + 3% |
| **Distance → identity** | `g = 1/max(distance, 0.5)` | **equal radius** from listener | assert `|d_A − d_B| < 1e-3 m` |
| **Pan → identity** | pan differs if placement is imperfect | mirrored azimuth, equal radius | assert `|p_A − p_B| < 1e-6` |
| **Shape → spectrum** | `MODE_RATIOS` are global, but geom type changes `char_size` derivation | both candidates **spheres** | assert both `geom_type == 2` |
| **Voicing the wrong body** | `_make_event` voices the **smaller** geom of the pair | strike the **floor** (huge `char_size`), never a small striker | assert `voiced_geom == candidate_geom` on 100% of events |
| **Refractory / double events** | `REFRACTORY_S = 0.10`, bounces | decode window `= DECODE_WIN_S` from first onset only | assert exactly 1 event in window |
| **Vision cannot carry size** | render resolution too low to resolve radius | certify first (PG.6) | probe on the frame must recover radius (R² ≥ 0.8) and bearing (≤ 5°) |

That last row is the one people skip. **If a probe cannot read object size out of
the rendered frame, the visual arm of HNS is empty and the whole test measures
nothing.** It is a fixture certification, not an assumption.

### 3.3 The controls that must fail

1. **Spectrum-flattened audio**: every event rendered at a single fixed f0, same
   amplitude, same pan. The identity channel is gone → **must return to chance**.
   If it does not, something other than the modal spectrum is leaking.
2. **Size-swap counterfactual (the SWAP-FLIP control)**: re-render the scene
   frame with the two candidates' radii exchanged between positions, audio
   untouched. The correct answer *flips*. Require the model's prediction to flip
   on ≥80% of trials it originally got right. This is the strongest single piece
   of evidence in the whole document: not "performance dropped" but "**the output
   moved in the direction the intervention predicted**". It is the causal test
   2603.19233 [V] shows the field needs.
3. **Vision-shuffled across batch**: the frame comes from another episode →
   chance.
4. **Pan-shuffled**: pan is uninformative *by construction* here, so this must
   **not** change anything. A model that degrades under pan shuffling is reading
   a leak; this is a control that must *pass* where the others must fail.

### 3.4 Two forms, deliberately

- **HNS-A (discriminative, CPU, no policy).** Single-step 2AFC from
  `[scene frame, stereo audio window, proprio, touch]`. Chance 50%. This is the
  falsifier that costs nothing and depends on no controller.
- **HNS-B (embodied, GPU).** Same scene; Jack must turn toward and reach the
  object that fell. Chance = 50% by bearing sign. Depends on locomotion; runs
  after D1 resolves. HNS-A is the claim; HNS-B is the demonstration.

---

## 4. The specs

Exact `Spec(...)` format from `experiments/registry_expansion.py`. Two fixtures
(PG.6, PG.7), the bakeoff (UB.10), the binding test (UB.9/UB.15), the standing
matrix (UB.11), and the diagnostics that keep the others honest.

```python
    # ── FIXTURES for the binding test ───────────────────────────────────

    Spec("PG.6", 2, "The playground has eyes, and they resolve what the test needs",
         hypothesis="An egocentric camera in the playground MJCF renders frames "
                    "from which a linear probe recovers object RADIUS (R^2>=0.8) "
                    "and BEARING (median error <=5 deg) for objects in FOV.",
         falsified_by="Radius or bearing unrecoverable at the chosen resolution "
                      "— then vision cannot carry HNS's identity->position "
                      "channel and UB.9 would measure nothing.",
         null_baseline="Probe on a shuffled-frame/label pairing; probe on a "
                       "constant grey frame.",
         metric="radius_r2_x_bearing_error", budget=Budget.CPU_LONG,
         depends_on=["PG.1"], seeds=3,
         control="Objects OUTSIDE the FOV must be unrecoverable — else the probe "
                 "is reading episode identity, not the image.",
         kills="Any visual claim in UB.9/UB.10 at this resolution. Escalate "
               "resolution or move vision to a frozen tower with cached "
               "embeddings before proceeding.",
         notes="playground.py:217-243 emits no <camera>. This spec adds one and "
               "certifies it. Render on CPU via MUJOCO_GL=osmesa; only ~500 "
               "distinct layouts are needed because HNS reuses layouts across "
               "episodes."),

    Spec("PG.7", 2, "The heard-not-seen fixture leaks nothing but the intended bit",
         hypothesis="In the HNS scene the two candidates are acoustically "
                    "indistinguishable except by modal fundamental: identical "
                    "pan (<1e-6), identical listener distance (<1e-3 m), "
                    "matched impact amplitude, and the candidate (not the "
                    "striker or floor) is the voiced geom on 100% of events.",
         falsified_by="Any leak: an audio-only probe over band energies, "
                      "amplitude and pan classifies which object fell above "
                      "chance+3%.",
         null_baseline="Chance (0.5) for the audio-only probe.",
         metric="audio_only_leak_margin", budget=Budget.CPU,
         depends_on=["PG.5"], seeds=3,
         control="A DELIBERATELY UNBALANCED variant (unequal mass, so amplitude "
                 "tracks size) must be classified WELL above chance by the same "
                 "probe — else the leak detector is blind and its null result "
                 "is worthless.",
         kills="UB.9. A binding test built on a leaky fixture measures the leak.",
         notes="Closes, in order, the seven leaks tabulated in "
               "docs/research/UNIFIED_BRAIN_BAKEOFF.md section 3.2. PG.5's "
               "circularity guard is the precedent: ground truth is computed in "
               "this file's own trig, never from the synth's labels."),

    # ── THE BINDING TEST ────────────────────────────────────────────────

    Spec("UB.9", 4, "Heard, not seen: the task that is impossible without fusion",
         hypothesis="On a scene where audio gives object IDENTITY (modal "
                    "fundamental) but not position, and a pre-event frame gives "
                    "position but not which object fell, the fused model "
                    "identifies the fallen object well above chance (>=0.75 "
                    "mean over 3 seeds, lower bootstrap CI > 0.5).",
         falsified_by="Fused accuracy indistinguishable from 0.5, OR "
                      "indistinguishable from the unimodal late ensemble — "
                      "either way nothing was bound.",
         null_baseline="Three nulls, all at chance BY CONSTRUCTION and all "
                       "measured anyway: (i) audio-only (pan is identical for "
                       "mirrored azimuths, ContactAudio.py:26), (ii) "
                       "vision-only (the frame predates the event), (iii) the "
                       "UNIMODAL LATE ENSEMBLE of (i) and (ii) — the arm that "
                       "is structurally incapable of synergy.",
         metric="hns_accuracy_over_ensemble", budget=Budget.CPU_LONG,
         depends_on=["PG.6", "PG.7", "T1.06"], seeds=3,
         control="SWAP-FLIP: re-render the frame with the two candidates' radii "
                 "exchanged between positions, audio untouched. The correct "
                 "answer flips, so the prediction MUST flip on >=80% of "
                 "previously-correct trials. Also: spectrum-flattened audio "
                 "must fall to chance, and PAN-SHUFFLED audio must NOT change "
                 "anything (pan is uninformative here; sensitivity to it means "
                 "a leak).",
         kills="The sentence 'his senses work in unison'. This is the smallest "
               "experiment that could establish it and it costs no GPU; if it "
               "fails, no larger experiment rescues the claim.",
         notes="I(audio;Y)=0, I(vision;Y)=0, I(audio,vision;Y)=1 bit — physical "
               "XOR, one bit of PURE synergy (PID framework, arXiv:2302.12247). "
               "Proprioception, Jack's dominant modality, is uninformative here "
               "by design, which is precisely why collapse cannot hide."),

    Spec("UB.15", 4, "Heard, not seen — embodied",
         hypothesis="Jack turns toward and reaches the object he heard fall but "
                    "did not see, above the 0.5 bearing-sign chance rate.",
         falsified_by="Reach target at chance, or unchanged when audio is muted.",
         null_baseline="Audio-muted policy; vision-frozen policy; the UB.9 "
                       "discriminative ceiling (the gap is the control cost).",
         metric="embodied_hns_success", budget=Budget.GPU, seeds=3,
         depends_on=["UB.9", "T2.02"],
         control="Left/right channel swap must invert the turn direction. A "
                 "500 ms audio lag must degrade timing but not identity — the "
                 "two channels fail differently, which is itself evidence they "
                 "are separately read.",
         notes="Deliberately the ONLY binding spec that depends on locomotion. "
               "Everything else in this block is falsifiable without a "
               "controller, so decision D1 cannot block the unison claim."),

    # ── THE BAKEOFF ─────────────────────────────────────────────────────

    Spec("UB.10", 4, "Fusion bakeoff: six arms, matched params, matched steps",
         hypothesis="At matched trainable parameters (+-5%), matched tokens per "
                    "modality, matched optimisation steps and matched data "
                    "order, at least one shared-computation arm beats the "
                    "late-concat null on the binding battery, and the ranking "
                    "is stable across 3 paired seeds.",
         falsified_by="A0 (late concat) ties the best arm everywhere — then at "
                      "this scale 'one brain' buys nothing over bolt-on "
                      "encoders and GOAL.md's architecture claim must be "
                      "restated. Report it; do not re-run until it looks "
                      "better.",
         null_baseline="A0 = per-modality encoders -> pool to one vector each "
                       "-> concat -> head ('concatenate and pray'). Plus the "
                       "UNIMODAL LATE ENSEMBLE computed for every arm.",
         metric="arm_ranking_x_synergy_gap", budget=Budget.GPU, seeds=3,
         depends_on=["UB.9", "T2.00"],
         control="Every arm must FAIL the cross-episode SWAP ablation on at "
                 "least one sense (i.e. swapping a sense's stream between "
                 "episodes must hurt). An arm that is invariant to swapping "
                 "every sense has learned a marginal, not a correspondence, and "
                 "its score on the battery is uninterpretable.",
         kills="Five of six architectures. The survivor is the trunk Jack "
               "ships; the rest are deleted, not kept 'for later'.",
         notes="ARMS. A0 late-concat null. A1 shared token trunk (multi-token "
               "per modality, modality-ID embeddings, readout tokens; "
               "arXiv:2205.06175, 2405.12213, 2409.20537). A2 = A1 + modality "
               "dropout with learned [MISSING-m] tokens (arXiv:2410.03010, "
               "2201.01763). A3 = A2 + cross-modal masked prediction, "
               "cross-signal not joint (arXiv:2311.00924, 2410.16424, "
               "2607.13522). A4 = A2 + contrastive alignment with "
               "state-proximity positives (arXiv:2510.01711, 2303.15343) - "
               "NOT episode-identity positives, which are false negatives on "
               "synchronous streams. A5 = per-modality experts + learned router "
               "(arXiv:2509.23468), the credible non-trunk alternative; if A5 "
               "wins, 'one brain' is the wrong shape and we say so. "
               "A3 and A4 are parallel, not cumulative, so architecture and "
               "objective are separated. TOKEN BUDGET IS EQUALISED ACROSS ARMS "
               "or this measures token counts (arXiv:2601.16667). "
               "PAIRED bootstrap CIs and IQM per arXiv:2108.13264 - unpaired "
               "3-seed architecture comparisons resolve nothing at this budget."),

    # ── THE STANDING AUDIT ──────────────────────────────────────────────

    Spec("UB.11", 4, "The modality ablation matrix (standing)",
         hypothesis="On the tasks x senses matrix, every sense shows a "
                    "degradation significantly above the PLACEBO column under "
                    "at least the cross-episode SWAP perturbation; no sense has "
                    "an all-null row of cells.",
         falsified_by="Any sense whose four perturbations are all "
                      "indistinguishable from the placebo modality — it is "
                      "decorative and loses its parameters (Tier-3 rule).",
         null_baseline="A PLACEBO MODALITY: pure noise, identical token count, "
                       "encoder capacity and dropout rate, wired in like a real "
                       "sense. Its column IS the empirical null distribution "
                       "for 'decorative', re-estimated every run.",
         metric="min_sense_margin_over_placebo", budget=Budget.GPU, seeds=3,
         depends_on=["UB.10"],
         control="TWO controls in opposite directions. (a) The placebo column "
                 "must be SMALL: a large placebo Delta means the procedure "
                 "measures off-manifold shock, not information, and every other "
                 "column is uninterpretable. (b) With proprioception replaced "
                 "by its [MISSING] token, a dropout-trained model must still "
                 "briefly stand using vision - vestibular substitution.",
         kills="Any encoder whose column is placebo-indistinguishable. Deletion "
               "is the default action, not a discussion.",
         notes="STANDING SPEC - re-runs on every architecture change, forever, "
               "like ME.5 at every decade of store growth. FOUR perturbations "
               "per cell: zero (off-manifold), matched noise (marginals kept), "
               "within-episode time-shuffle (temporal binding destroyed), "
               "CROSS-EPISODE SWAP (correspondence destroyed, everything else "
               "kept). Swap is the primitive: it is the only one that isolates "
               "correspondence, which is what binding means. Ablation uses the "
               "learned [MISSING-m] token, never zeros, or the matrix measures "
               "brittleness (arXiv:2410.03010). Logged alongside: per-layer "
               "cross-modal attention mass (arXiv:2410.16424) and the learned "
               "binary modality mask (arXiv:2209.07682) - both free, both "
               "necessary-not-sufficient, both red flags rather than claims."),

    Spec("UB.12", 4, "Synergy, not redundancy: beating the unimodal ensemble",
         hypothesis="On every task in the battery the fused model beats the "
                    "UNIMODAL LATE ENSEMBLE (independently trained per-sense "
                    "models, predictions averaged), paired across seeds, with "
                    "a bootstrap CI on the paired difference excluding zero.",
         falsified_by="Fusion >= best single modality but <= the ensemble on "
                      "every task: the model is exploiting redundancy and "
                      "uniqueness, and computes nothing jointly. This is the "
                      "most likely honest outcome and it must be reportable.",
         null_baseline="max_m U_m (the trivial bar) AND the ensemble E (the "
                       "real bar). Beating max_m U_m is not evidence of fusion.",
         metric="synergy_gap", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["UB.10"],
         control="On UB.9 (pure synergy, all unimodal channels at chance) the "
                 "ensemble MUST sit at chance. An ensemble above chance there "
                 "proves the fixture leaks and PG.7 passed wrongly.",
         notes="The operational definition of 'one brain': the late ensemble is "
               "structurally incapable of synergy because no parameter ever "
               "sees two modalities jointly, so F > E is joint computation by "
               "construction. Costs 5 tiny models per task; compute it for "
               "every arm, every task, forever. Frame results as PID "
               "redundancy/uniqueness/synergy (arXiv:2302.12247)."),

    Spec("UB.13", 4, "Cross-modal retrieval: the gate, never the claim",
         hypothesis="Given a contact-audio window, the matching visual clip is "
                    "retrieved above chance (R@1 and R@10 vs a candidate set of "
                    "known size), including against HARD negatives: the same "
                    "episode at +-0.5 s, and a different object at the same "
                    "instant.",
         falsified_by="At-chance retrieval against hard negatives while easy "
                      "retrieval succeeds — then the model matched onset "
                      "synchrony, not content.",
         null_baseline="Chance = 1/N for the actual candidate-set size, stated "
                       "before the run; plus a retriever over event ONSET TIMES "
                       "only, which is the synchrony-shortcut baseline.",
         metric="hard_negative_recall_at_1", budget=Budget.GPU_SHORT, seeds=3,
         depends_on=["UB.10"],
         control="Time-offset negatives must be harder than random negatives. "
                 "If they are equally easy, the candidate set is trivial.",
         kills="Nothing on its own. This spec exists so that a NULL result on "
               "the contrastive arm (A4) is interpretable: without it, 'A4 did "
               "not help control' cannot be distinguished from 'A4's loss never "
               "trained'. Retrieval is necessary, never sufficient "
               "(arXiv:2603.19233: encoded is not used)."),

    Spec("UB.14", 4, "Cross-modal prediction, against the null that usually wins",
         hypothesis="Masked touch is predicted from vision+proprioception "
                    "better than from proprioception ALONE, and better than the "
                    "unconditional mean, at matched capacity.",
         falsified_by="Proprio-only matches vision+proprio: foot contact is "
                      "inferable from joint torques, so vision adds nothing "
                      "here. An HONEST and likely outcome that must be "
                      "reported, not retried.",
         null_baseline="Unconditional mean (the floor) AND a proprio-only "
                       "predictor of equal capacity (the real bar).",
         metric="touch_r2_over_proprio_only", budget=Budget.CPU_LONG, seeds=3,
         depends_on=["PG.1"],
         control="Touch-from-SHUFFLED-vision must collapse to the "
                 "unconditional mean — else the head ignores its vision input "
                 "and the conditioning is decorative.",
         kills="The vision->touch masked objective in arm A3, if vision adds "
               "nothing over proprio. Run this BEFORE the bakeoff: it costs CPU "
               "minutes and can delete an arm's justification.",
         notes="Calibrate expectations from Kepler-Encoder (arXiv:2607.13522): "
               "fused-vs-vision-only force R^2 of 0.049/-0.001/0.187 across "
               "three robots, one of them NEGATIVE, p<=0.012. Real, clean, "
               "small. A bakeoff expecting a large effect has mis-specified "
               "its success criterion."),

    Spec("UB.16", 4, "Sensory information reaches the controller (D1-agnostic)",
         hypothesis="Zeroing the trunk's percept vector z degrades tasks that "
                    "require non-proprioceptive information, and does NOT "
                    "degrade flat-ground locomotion.",
         falsified_by="z-ablation changes nothing anywhere (the trunk is "
                      "decorative in the control path) OR it degrades flat "
                      "walking too (z is smuggling proprioception the "
                      "controller already has, so the comparison in D1 was "
                      "never about perception).",
         null_baseline="Controller on raw proprioception alone; controller with "
                       "z replaced by its batch mean.",
         metric="z_channel_asymmetry", budget=Budget.GPU, seeds=3,
         depends_on=["UB.11", "T2.02"],
         control="A SHUFFLED-z controller (z drawn from another episode) must "
                 "match the zeroed-z controller. If shuffled-z is WORSE than "
                 "zeroed-z, the controller is reading correspondence, which is "
                 "a stronger result than the hypothesis claims.",
         notes="The asymmetry IS the test, and it holds under either D1 "
               "outcome. If D1 removes the trunk from the control path, z is "
               "the entire sensory channel and this spec certifies it. If the "
               "trunk stays end-to-end, z is the readout-token bundle and the "
               "same measurement applies. Locomotion is the task where "
               "proprioception is SUFFICIENT, so it is the wrong task to judge "
               "a binder by - which is why 'no degradation on flat walking' is "
               "a PASS condition here, not a failure."),
```

Re-parenting note for whoever edits the registry: **UB.1–UB.8 should be
re-parented off `T4.01`.** UB.1's dependency chain routes through T2.01 (FAIL),
so the entire unison ladder is BLOCKED for reasons unrelated to fusion. UB.1's
natural parents are `UB.10` and `PG.7`; UB.4's are `PG.5` + `UB.9`. Only UB.7
and UB.15 legitimately require a controller.

---

## 5. What the trunk is FOR — designed for both D1 outcomes

The architecture is stated as a **contract**, so that D1 changes the wiring and
nothing else:

```
   senses ──► per-modality stems (multi-token, modality-ID + time embeddings)
                                │
                                ▼
                      TRUNK  (shared computation)
                                │
                      k readout tokens ──► z ∈ R^32..64,  written at 5-10 Hz
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   BINDING HEADS (train-time only)        CONTROLLER (50 Hz)
   masked cross-modal prediction,          input: [proprio ⊕ z]
   contrastive, unimodal aux                (SimBa MLP, or the trunk itself)
```

**If D1 removes the trunk from the control path (options A or D):** the trunk is
a **perception and state-estimation encoder**, not a policy. Its job is to turn
five asynchronous, partially-redundant, partially-missing sensory streams into
one small vector `z` that is *sufficient for everything proprioception is not*:
what is in the room, where the sound came from, what the last utterance asked
for, whether a foot is loaded. Sensory information reaches the controller
through `z` and **only** through `z` — which is what makes UB.16's asymmetry
test meaningful, and what makes the whole binding claim testable without a
controller at all. This is the arrangement `MULTIMODAL_BINDING.md` §8 already
sketched, restated as a measurable interface.

**If D1 keeps the trunk end-to-end (options B or C):** `z` is the readout-token
bundle consumed in-graph. Every spec in §4 is unchanged; only UB.16's ablation
target moves from a slot to a tensor slice.

**In both cases three things hold**, and they are the point:

1. **The binding claim is separable from the control claim.** UB.9–UB.14 are
   supervised probes on `z`. They pass or fail regardless of whether Jack can
   walk. Given that T2.01 and T2.02 currently FAIL, this is the difference
   between a testable unison claim and none.
2. **The trunk earns its parameters on tasks proprioception cannot solve.** Not
   locomotion. HNS, occluded contact timing, out-of-view events, language
   routing — the ambiguous-and-occluded regime where ManiWAV (2406.19464 [V])
   and Audio-VLA (2511.09958 [V]) locate all of audio's value.
3. **The trunk can shrink.** Nothing above requires 57M parameters. If the
   bakeoff's winning arm is 6M, the trunk is 6M. The measurements decide the
   size; the size does not get to decide the measurements.

**One thing to change regardless of D1**: multi-token-per-modality. The current
one-token-per-sense fusion head is the maximal-entanglement configuration that
2505.22483 [V] identifies as the collapse mechanism, and the 18-vs-1 token
imbalance is the state-dominance ReViP (2601.16667 [V]) documents. Both are
`UnifiedBrain.py:4204-4310` and `:1664-1727`.

---

## 6. CPU-first staging — what dies on 4 ARM cores

The ordering principle: **every arm should be given a chance to die before it
costs a GPU-hour.** Concretely, seven things are falsifiable on this box.

| # | Falsifier | Cost on 4 ARM cores | Kills |
|---|---|---|---|
| 1 | **PG.7 leak probe** — audio-only classifier over band energies, amplitude, pan | seconds (numeric asserts) + ~2 min | UB.9 if the fixture leaks |
| 2 | **UB.14 touch redundancy** — predict foot contact from proprio alone on rollout data | ~10 min | A3's vision→touch objective |
| 3 | **PG.6 vision certification** — probe radius/bearing from ~500 rendered layouts | ~15-25 min (osmesa) | the visual arm of HNS at that resolution |
| 4 | **UB.9 at small scale** — d=128, ~1-3M params, 10k episodes, all six arms | ~1.5 h total | any arm that cannot pass a **pure-synergy** task |
| 5 | **Unimodal + ensemble nulls** at chance | included in 4 | the fixture, if a null is above chance |
| 6 | **UB.11 machinery incl. the placebo column** | ~20 min | the ablation procedure itself, if placebo Δ is large |
| 7 | **Retrieval chance calibration + hard-negative construction** (UB.13) | ~10 min | a mis-specified candidate set |

Total: **under 3 hours of CPU**, and it can eliminate arms, delete an objective,
invalidate a fixture, and expose a broken measurement procedure — all before
Kaggle is touched.

The reasoning for (4) deserves stating plainly, because small-scale bakeoffs are
usually a bad idea. Small-scale *ranking* does not reliably predict large-scale
ranking. But HNS-A is not a capacity-limited task: it is **one bit**, learnable
by a 1M-param model from 10k examples if and only if the architecture and
objective can represent a cross-modal conjunction at all. An arm that cannot
extract one bit of pure synergy at d=128 has no *mechanism* that appears at
d=512 — it has a missing pathway, and scale does not grow pathways. So the CPU
stage is a **necessary-condition filter**, not a ranking. Arms that survive get
ranked on GPU; arms that fail are deleted, and the deletion is defensible.

Two enabling tricks make the CPU stage possible at all:

- **Cache frozen-tower embeddings once.** HNS reuses ~500 distinct layouts, so
  vision is 500 forward passes through a frozen SigLIP/DINOv2 on ARM CPU
  (~8-10 min, one time) and every subsequent arm × seed trains on cached
  vectors with no vision tower resident. This is the same property that makes
  the whole plan fit a burst GPU schedule, and it is destroyed the moment the
  tower becomes trainable.
- **Audio is free.** `ContactAudio` synthesises microseconds per contact,
  CPU-side, with exact labels. There is no dataset to download and no annotator.

Not falsifiable on CPU: anything needing a trained controller (UB.15, UB.16,
the embodied half of UB.10's battery), and full-resolution end-to-end vision
training.

**Ordering: PG.7 → UB.14 → PG.6 → UB.9(small) → UB.11(machinery) → [GPU] →
UB.10 → UB.11(full) → UB.12 → UB.13 → UB.16 → UB.15.**

---

## 7. Cost — free compute only

Budget: 4 shared ARM cores (always), Kaggle 30 h/week (P100; T0.10/T0.11 PASS,
so the sm_60 issue recorded in `DECISIONS_NEEDED.md` is resolved in practice),
Colab T4 elastic. Current spend, `experiments/gpu_budget.json`: W31 used
37.5 h Kaggle + 7.7 h Colab; W32 is at 6.4 h Kaggle. There is room.

| Item | Where | Estimate | Notes |
|---|---|---|---|
| HNS episode generation (10k eps × ~300 steps) | CPU ×4 | 7-10 min | ~1800 steps/s measured in T0.07 |
| Scene-frame rendering (~500 layouts, 128²) | CPU | 15-25 min | `MUJOCO_GL=osmesa`; llvmpipe is slow but the count is small |
| Frozen-tower embedding cache | CPU | 8-10 min, once | the trick that makes everything after this cheap |
| **CPU-stage total (items 1-7 of §6)** | **CPU** | **< 3 h** | **kills arms for free** |
| UB.10 bakeoff, 6 arms × 3 seeds, d=384 | T4/P100 | 9-18 GPU-h | 30-60 min per arm-seed; checkpoint every 15 min |
| UB.11 full matrix (4 perturbations × senses × tasks + placebo) | T4 | 2-3 GPU-h | eval-only; re-runs on every arch change |
| UB.12 synergy gap (5 unimodal + ensemble per task) | T4 | 2-4 GPU-h | small models, but ×tasks×seeds |
| UB.13 retrieval | T4 | 1 GPU-h | eval-only on trained arms |
| UB.16 z-channel | T4 | 3-5 GPU-h | needs a controller; gated on D1 |
| UB.15 embodied HNS | T4 | 6-10 GPU-h | gated on T2.02 |
| **GPU total to the headline claim** | | **~17-26 GPU-h** | one Kaggle week |
| **GPU total incl. embodied capstone** | | **~26-41 GPU-h** | two Kaggle weeks |

Everything checkpoints every ~15 min and resumes (T0.04/T0.05 PASS), which is
the only reason a 12 h cap and Colab teardown are survivable.

---

## 8. What we refuse to claim

- **That architecture implies fusion.** 2603.19233 [V] measured 28-92%
  zero-effect ablation rates and separable pathway subspaces in three
  production multi-pathway VLAs. A diagram is not evidence.
- **That a probe or a retrieval score is binding.** Encoded is not used. Both
  are gates (UB.13), never claims.
- **That beating the best single modality is synergy.** The bar is the unimodal
  late ensemble (UB.12), and the honest likely outcome is that Jack's unison is
  mostly *state-estimation redundancy* — which is a real result, worth having,
  and must be reportable without embarrassment.
- **A large effect.** Kepler [V] is the closest measured analogue and its
  binding effect was R² ≈ 0.05-0.19 with one of three robots negative. Design
  the statistics for a small effect (paired, IQM, bootstrap CIs, 2108.13264 [V])
  or the experiment cannot see the thing it is looking for.
- **Anything about a sense whose ablation-matrix column is
  placebo-indistinguishable.** It loses its parameters. That is the Tier-3 rule
  and this document does not carve an exception for senses.
- **"The senses work in unison"**, until UB.9 passes with its SWAP-FLIP control
  and UB.12 shows a positive synergy gap. Those two, together, are the claim.
