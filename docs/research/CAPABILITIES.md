# Generalist Embodied Agent — Capability Taxonomy (researched 2026-08-06)

Serves GOAL.md. IDs stable for diffing against the ladder. FT = free-tier
feasible: Y / P (proxy) / N.

## SOTA landscape ("general" in 2026)
RT-2 (2307.15818) VLM→actions-as-text · Octo (2405.12213) 93M generalist,
diffusion head · OpenVLA / OFT (2406.09246 / 2502.19645) parallel decoding +
chunking + L1 beats autoregressive · π0 (2410.24164) VLM trunk + 300M
flow-matching expert, 50Hz chunks · π0.5 (2504.16054) one model does subtask
inference AND control · Knowledge Insulation (2505.23705, FAST 2501.09747)
stop-gradient from action expert, 5-7× faster, keeps VLM knowledge · π*0.6 /
Recap: RL from experience + corrections · π0.7 (2604.15483) video-history +
steerable conditioning, zero-shot cross-embodiment · GR00T N1→N1.7 (2503.14734)
dual-system VLM + diffusion transformer, humanoid-first · Gemini Robotics 1.5 /
ER 1.5 (2503.20020 / 2510.03342) thinking embodied-reasoner orchestrating a VLA
· Small/open: SmolVLA 450M (2506.01844), VLA-0 (2510.13054), MolmoAct2
(2605.02881), RDT-1B (2410.07864) · HumanoidBench (2403.10506), FB-CPR
(2504.11054).

**Converged recipe:** frozen-ish VLM trunk + small continuous action expert +
action chunking + cross-embodiment data + think-then-act + RL post-training.
Jack's 58M-trainable + frozen-trunk design matches the field's small end.

## A. PERCEPTION
- **P1 Open-vocab grounding** (FT:Y): locate never-seen named objects. Kill:
  held-out nouns ≤ name-shuffled prompt. Control: ablated vision → null.
- **P2 Spatial/3D** (FT:Y): pointing/relative position vs MuJoCo ground truth.
  Null: centroid prior. Control: swapped camera extrinsics must collapse it.
- **P3 Proprio-vision fusion** (FT:Y): fused beats each alone under occlusion.
  Control: mismatched proprio must hurt (proves it's used).
- **P4 Multi-view** (FT:Y): 2-cam beats best 1-cam on occluded targets.
  Control: duplicated single view must not match true 2-view.
- **P5 Temporal perception** (FT:Y): history model intercepts moving targets;
  single-frame fails. Control: shuffled frame order → null. (π0.7, V-JEPA 2 2506.09985)
- **P6 Visual OOD robustness** (FT:Y): success drop under texture/light shift <
  pre-registered X%. Null: from-scratch encoder. Control: proprio-only task
  must be unaffected. (SimplerEnv 2405.05941)
- **P7 Contact/force sensing** (FT:Y): contact-classification beats vision-only.
  Control: zeroed contact channel → null.
- **P8 Affordances** (FT:P — no hands): predicted approach points beat centroid
  prior. (MolmoAct visual traces)

## B. CONTROL
- **C1 Whole-body locomotion** (FT:Y): beat published Humanoid-v5 baselines.
  Control: gravity ×1.5 must degrade (dynamics, not replay). (TD-MPC2
  2310.16828, DreamerV3 2301.04104)
- **C2 Action chunking** (FT:Y): some k>1 beats k=1 at matched FLOPs; giant k
  must fail under perturbation (tradeoff exists). (ACT 2304.13705)
- **C3 Latency reactivity** (FT:Y): RTC-style chunk overlap beats naive swap at
  100-300ms injected latency; must be equivalent at zero latency. (RTC 2506.07339)
- **C4 Multimodal action distributions** (FT:Y): flow head solves bimodal
  obstacle task where MSE collapses to invalid mean. RISK: OFT found L1 matches
  on some benches — genuine falsification possible. Control: unimodal task must
  show no gap. (Diffusion Policy 2303.04137)
- **C5 Perturbation recovery** (FT:Y): push-trained survives test pushes above
  nominal-trained. Control: push during an existing fall must be unrecoverable.
- **C6 Smoothness/energy** (FT:Y): lower jerk at equal success. Control: added
  action noise must worsen the metric.
- **C7 Long-horizon sequencing** (FT:P): full-sequence success > 0.5 × product
  of stage successes. Control: untrained stage order must fail.
- **C8 Dexterous manipulation** (FT:N): Humanoid-v5 has no hands — EXPLICIT
  SCOPE EXCLUSION, written down, not silently missing.

## C. LANGUAGE
- **L1 Atomic instructions** (FT:Y): per-instruction success > empty-prompt
  behavior. Control: unparseable-language prompt → null. (LIBERO 2306.03310)
- **L2 Compositional generalization** (FT:Y, HIGH VALUE): held-out verb×object
  cells above chance — GRID MUST BE DESIGNED INTO TRAINING DATA BEFORE
  TRAINING. Control: untrained verb must fail. (CALVIN 2112.03227 ABC→D)
- **L3 Mid-episode corrections** (FT:Y): "stop / no, the other one" diverges
  behavior within k steps vs paired no-correction seed. Control: nonsense
  correction → no systematic divergence. (Hi Robot 2502.19417)
- **L4 Knowledge insulation at Jack's scale** (FT:P): frozen/insulated trunk
  beats fully-finetuned on semantic-novel tasks at equal in-distribution
  success. (2505.23705 — the D1 literature)
- **L5 Grounded scene dialogue** (FT:Y): answers about live sim state beat a
  blind LLM. Control: occluded-object questions need calibrated uncertainty.
- **L6 Negation/constraints** (FT:Y): "don't step on the mat" reduces
  violations when violating is the shortest path. Control: constraint naming a
  nonexistent object must change nothing.

## D. MEMORY (see docs/research/MEMORY.md for depth)
- **M1 In-context history** (FT:Y): history-conditioned beats Markov on POMDP
  tasks (light out, target moved while occluded); gap must VANISH on fully
  observed tasks. (MemoryVLA 2508.19236)
- **M2 Cross-episode episodic memory** (FT:Y, DIFFERENTIATING — near-empty
  niche): find hidden object faster in episode N+1; wiping the store must
  restore null search time. (RoboMemArena 2605.10921)
- **M3 Spatial memory/mapping** (FT:Y): return to out-of-view location beats
  random walk; teleporting the agent must break it.
- **M4 User/preference memory** (FT:Y): session-2 behavior reflects session-1
  preference; later contradiction must override earlier.
- **M5 Skill-library persistence** (FT:P): composite task learned faster with
  library; irrelevant library must give no gain. (Voyager 2305.16291)

## E. PLANNING
- **PL1 Hierarchical decomposition** (FT:Y): subgoal-inferring model beats flat
  policy; deliberately WRONG subgoals must hurt (causal use — cheap, strong).
  (π0.5, SayCan 2204.01691)
- **PL2 Embodied chain-of-thought** (FT:P): reasoning-then-acting beats direct
  at MATCHED compute; garbage traces must not match real ones. (ECoT 2407.08693)
- **PL3 Progress/success estimation** (FT:Y, FOUNDATIONAL — gates LE4/LE5/PL4):
  predicted progress correlates with stage on held-out rollouts incl. failures.
  THE null everyone skips: a linear-in-timestep predictor. Reversed video must
  yield reversed progress.
- **PL4 Failure detection & replanning** (FT:Y): recovery after injected
  failure (object teleported) beats no-detector; false-alarm rate bounded.
- **PL5 Tool use** (FT:P): mock tools in sim; no cargo-cult calling on
  info-complete tasks.
- **PL6 World model** (FT:Y): latent rollouts beat copy-last-frame AND planning
  with it beats model-free at equal steps (both, or it's not useful).

## F. LEARNING
- **LE1 Imitation efficiency** (FT:Y): success-vs-N-demos curve; frozen trunk
  must shift it left vs from-scratch same-size.
- **LE2 Cross-embodiment transfer** (FT:Y): pretrain on morphology variants
  speeds learning on new body; white-noise pretraining must give nothing.
  (Open X 2310.08864, CrossFormer 2408.11812, HPT 2409.20537)
- **LE3 Few-shot adaptation** (FT:Y): LoRA on 10 demos beats zero-shot; demos
  of a DIFFERENT task must not help.
- **LE4 RL beyond demonstrations** (FT:Y): BC+RL beats BC plateau; random
  reward must not improve success. (Recap/π*0.6)
- **LE5 Test-time adaptation** (FT:P): within-session improvement vs frozen
  policy — MANDATORY null: best-of-k retries (most "TTA" wins are just
  retries). (2601.06748)
- **LE6 Learning from passive video** (FT:P): action-free rendered-sim video
  pretraining improves downstream BC; temporally-shuffled video must give less.
- **LE7 Continual learning without forgetting** (FT:Y): task-A drop <10% with
  adapters vs >50% naive sequential. Report gap to joint-training ceiling.
- **LE8 Trunk-knowledge preservation** (FT:Y, CHEAPEST D1 EVIDENCE, EARLY):
  linear probes on trunk features hold through action training AND semantic
  task success tracks probe quality; unfreezing must reproduce drift.
- **LE9 Domain randomization transfer** (FT:Y): randomized-dynamics training
  survives held-out parameter draws; beyond-envelope shifts must still fail
  (quantifies the envelope honestly).

## G. SOCIAL / COMPANION (weakest SOTA coverage — Jack can be NOVEL here)
- **S1 Contingent responsiveness** (FT:Y): response contingent on user-avatar
  events; time-shuffled event stream must show nothing.
- **S2 Nonverbal expressiveness** (FT:Y): human raters classify intended state
  from motion clips above chance (pre-registered small-n).
- **S3 Intent inference** (FT:Y): predict user goal from partial trajectory
  above majority class; goal-agnostic random walks must yield chance + high
  entropy.
- **S4 Safety around humans** (FT:Y): user-zone violations <1/1000 even when
  reward tempts crossing, ACROSS reward scales; ablating the safety channel
  must bring violations back.
- **S5 Persistent identity** (FT:Y): classifier matches session-k behavior to
  same agent vs re-seeded twin; persona reset must drop to chance.
- **S6 Joint action** (FT:Y): carry-with-avatar beats solo attempt; adversarial
  partner must reduce success (coordination is bidirectional).

## H. CROSS-CUTTING
- **X1 Calibrated uncertainty** (FT:Y): ECE <0.15; impossible tasks must get
  low confidence.
- **X2 Latency budget** (FT:Y): p99 inference < chunk duration on T4 — measured.
- **X3 Seed discipline** (FT:Y): claims enter the ledger only when the 95% CI
  excludes the null across seeds. (Already ladder policy.)
- **X4 Adversarial instructions** (FT:Y): harmful/contradictory compliance
  bounded; benign compliance must NOT collapse (refusing everything is failure).

## Ladder-diff notes
1. Already covered: C1, C2(part), C4(part), LE1, LE9, X3.
2. Highest-leverage additions: **LE8, L2 (design the grid BEFORE training),
   PL3, C3, M2**.
3. Write down exclusions: C8 (no hands), sim-to-real (no robot), full PL5.
4. Nulls everyone skips: best-of-k (LE5), time-step regression (PL3).
5. Build order the SOTA implies: LE8 → C2/C4 → L1/L2 → PL1/PL3 → LE4.
