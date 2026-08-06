# Open-Ended Curiosity-Driven Learning (researched 2026-08-06)

Serves GOAL.md ("He explores because he wants to"). Per mechanism: HOW, whether
it composes with a flow-matching head + frozen trunks, feasibility, and a test
{H hypothesis / K kill / N null / C control-that-must-fail}.

## THE architectural fact that shapes everything
A flow-matching head has no tractable log-likelihood → classic PPO/SAC
reward-gradient RL does not natively compose with it. Escape hatches:
(a) **data-side mechanisms** — hindsight relabeling + conditional flow-matching
regression (HER 1707.01495, GCSL 1912.06088) compose PERFECTLY: flow matching
is supervised conditional generation over whatever data you feed it;
(b) flow-RL fine-tuners — ReinFlow 2505.22094 (+135% legged locomotion, 83%
less wall time than DPPO 2409.00588), FPO 2507.21053.
**Consequence: prefer mechanisms that select GOALS AND DATA over ones that
shape a reward you must backprop through.** This drives everything below.

## 1. Intrinsic motivation signals
Taxonomy: novelty / surprise / info-gain / learning-progress / empowerment —
NOT interchangeable; surprise-seekers get trapped by irreducible noise.

- **ICM** (1705.05363, study 1808.04355): forward-model error in inverse-model
  features. SUPERSEDED — fails on action-conditioned noise (physics chaos Jack
  himself causes). C: action-triggered random-texture panel MUST trap it.
- **RND** (1810.12894): predictor-vs-frozen-random-net error = pure novelty.
  Usable reward-free as a goal-candidate RANKER (that composes natively). Run
  on frozen-trunk embeddings, never raw pixels. C: never-repeating-pattern
  panel must trap RND-as-reward; RND-as-ranker + LP filter must not.
- **Plan2Explore / ensemble disagreement** (2005.05960, 1906.04161,
  BYOL-Explore 2206.08332): disagreement → 0 on irreducible noise, stays high
  where merely ignorant — the principled noisy-TV fix. Needs a small RSSM
  (~5-15M) over trunk embeddings; fits 30 GPU-h/wk. C: disagreement must → 0
  at the noise panel while raw surprise stays high (else ensemble shares too
  much init/data).
- **LEARNING PROGRESS (LP)** — Oudeyer IAC→R-IAC→IMGEP (1708.02190): sample
  goal regions where |Δcompetence| is largest. The ONLY signal that is
  noisy-TV-proof AND saturation-proof AND purely data-side. Near-free bandit
  bookkeeping on the ARM box. MAGELLAN 2502.07709 = learned LP predictor for
  large goal spaces. TEST: staged curriculum must EMERGE (time-ordered mastery
  stand→walk→push→ramp with distinct onsets); C: forever-unlearnable goals
  ("make the noise panel blue") must decay to ε allocation.
- **Empowerment/SMiRL** (1509.08731, 1912.05510): skip as primary — SMiRL's
  homeostasis opposes "fall off the ladder and learn"; legitimises only a
  small stay-upright shaping term.

## 2. Autotelic goal invention
- **IMGEP backbone** (1708.02190; autotelic survey Colas 2012.09830): agent
  samples own goals from learned outcome space, attempts, archives
  (trajectory→achieved outcome), LP biases future sampling. Jack's backbone.
- **HER/GCSL**: every failure is a success at what it DID achieve — relabel,
  regress the flow head. THE weld between goal-conditioning and flow matching,
  zero RL machinery. N (critical): shuffled-goal-label training; C: goals
  outside achieved support (fly 2m up) must score ~0 or the detector is broken.
- Goal-GAN 1705.06366 / Skew-Fit 1903.03698 / CURIOUS 1810.06284: subsumed —
  keep only CURIOUS's modular-LP idea inside LP.
- **LLM-autotelic line**: LMA3 2305.12487, MAGELLAN, HERAKLES 2508.14751
  (compiles achieved LLM-goals into reusable skill hierarchy).

## 3. Open-endedness — learning that never saturates
Saturation closes one of three loops: environment stops producing challenges /
"interesting" stops discriminating / mastery isn't compiled into skills.
Non-saturation = environment generation + interestingness model + skill
compilation.
- POET/Enhanced 1901.01753/2003.08536: co-evolve terrain+agents (too costly
  as-is; keep minimal-criterion filtering).
- **ACCEL 2203.01302**: mutate existing levels toward the agent's frontier —
  for Jack, MJCF scene-parameter editing with LP as the learnability score. CHEAP.
- **OMNI 2306.01711**: FM as model of INTERESTINGNESS filters learnable-but-
  boring ("stack red on blue, blue on red, …").
- **OMNI-EPIC 2405.15568**: FM writes CODE — env + reward — new tasks forever;
  directly implementable (artifact = MJCF/Python).
- **Voyager 2305.16291**: verified skill library = the compilation leg; Jack's
  analog: ledger-gated named skills.
- Genie 3 / SIMA 2 (2512.04797): environments-as-data preview; not local-feasible.
- AdA 2301.07608: distribution breadth, not task count, drives generalization.
- TEST for the whole loop: distinct mastered outcome clusters (in trunk
  embedding space) grow without plateau over 8 weeks vs fixed-scene null;
  C: mutation WITHOUT learnability filter must degenerate.

## 4. Developmental robotics
- **Goal babbling** (SAGG-RIAC, Baranes & Oudeyer 2013): explore in OUTCOME
  space, not action space — exponentially better in high-DoF; one 48-step
  action chunk ≈ one babble.
- **Affordances through interaction** (Playground Experiment; 2008.11503):
  (object features × action) → outcome distribution, free from the IMGEP
  archive. TEST: predict pushability/liftability of held-out objects above
  chance; C: welded-immovable object must classify un-pushable.
- **Curricula are an OUTPUT** (Oudeyer 2007; Craftax human study 2503.23631):
  staged development is the falsifiable signature, not an input.

## 5. Reward-free skill discovery
DIAYN 1802.06070 (static poses failure) → DADS 1907.01657 → LSD 2202.00914 →
CSD 2302.05103 → **METRA 2310.08887 (the default: latent space where distance ≈
temporal distance; first pixel-humanoid diverse locomotion)** → Periodic Skill
Discovery 2511.03187 (gaits, composable with METRA), RGSD 2510.06203, 2406.10127;
Forward-Backward line (2103.07945, 2209.14935) heavier alternative.
Integration: METRA with a small dedicated Gaussian policy on trunk embeddings
(GPU windows), then DISTILL skills into the z-conditioned flow head by
trajectory regression. N: random-repeated-action sequences (flailing covers
ground too); C: ablate the temporal-distance constraint → must degrade toward
DIAYN's static poses.

## 6. FM-guided curiosity (where "climb the ladder" comes from)
- ELLM 2302.06692: LLM proposes plausibly-useful goals from state captions —
  and never proposes "stare at static".
- Motif 2310.00166: LLM preference distilled into a cheap reward model —
  right compute split (FM offline, RM online).
- Eureka 2310.12931: LLM writes/evolves SUCCESS-CHECK code.
- GLANCE 2605.03782: VLM-expectation-vs-observation discrepancy as signal;
  2602.04837 group self-improvement.
- **Jack's mechanism**: every K minutes the box renders 4 snapshots → VLM
  captions, lists affordances ("a ladder against the platform, a pool"),
  proposes ~10 (goal-text, success-check-code) candidates → IMGEP buffer where
  **LP HAS THE FINAL VOTE** — VLM proposes, learning progress disposes.
  C: scrambled-caption VLM (fed another scene) must NOT beat LP-only.

## 7. The playground (must be BUILT — nobody ships it)
dm_control.mjcf / XML templating; parameter vector = ACCEL mutation space:
{ramp angle, stair pitch, ladder rung spacing, objects (mass/friction/size),
seesaw, platform}.
- **Ladder**: ball hands → MuJoCo ADHESION ACTUATORS (native ≥2.2) on hand
  geoms = controllable grasp scalar; falling is the desired data; steep stairs
  are the LP-findable precursor.
- **Water**: no fluid volume in MuJoCo → pool region + `mjcb_passive` callback:
  buoyancy ρVg on submerged volume + quadratic drag below z_water (~40 lines,
  CPU-negligible). C: with buoyancy disabled the ragdoll must sink and
  swim-speed must → 0, else the metric measures floor contact.
- **Noisy-TV panel (mandatory fixture)**: wall panel re-randomizing texture
  every step + action-triggered variant. EVERY curiosity claim reports
  time-share within 2m of it.
- Rendering: osmesa/EGL offscreen 64-128px; trunk runs sparsely (every 25-50
  steps + snapshot batches), NEVER in the control loop.
- Physics validation spec: ragdoll floats at ρ-ratio depth ±10%; boxes slide
  iff tanθ > μ.

## RECOMMENDED STACK
**Signals:** LP primary (goal-level) + episodic k-NN novelty on trunk
embeddings as candidate-ranker (NGU 2002.06038 style, never a backprop reward)
+ ensemble disagreement in phase 4. NO raw ICM/RND rewards, ever.
**Goal invention:** IMGEP + hindsight flow-matching; goal space = proprio ⊕
object poses ⊕ trunk embedding; VLM proposer + Eureka success-checks on top.
**Skills:** METRA in GPU windows → distilled into flow head; mastered goals
promoted to ledger-verified named skills (Voyager).
**Playground:** one procedural room (ramp, stairs, ladder+adhesion, 5 objects,
seesaw, pool, noise panel), ACCEL-mutated, OMNI-EPIC-lite acceptance filter.

**Build order (each ledger-gated):**
P1 playground+physics specs (CPU, 1-2wk) → P2 hindsight flow goal-reaching →
P3 LP sampling → EMERGENT CURRICULUM (first falsifiable "teaches himself") →
P4 METRA skills + distillation → P5 VLM proposer + success-checks →
P6 scene-mutation loop (8-week non-saturation curve) + optional ReinFlow.

**The metric that means it's working:** monotone growth of distinct mastered
outcome clusters with near-zero noise-panel dwell — curiosity without the trap.
