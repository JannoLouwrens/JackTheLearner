# PUBLISHED PRIORS — what other groups' tests already tell us

> `SYSTEM.md` law 3a (owner, 2026-11-15): read before you race. This is the
> standing reference for the PRIOR every race order must carry. One section per
> question; the Researcher (`scripts/research_prompt.md`) adds and refreshes
> them. One section per question, each ≤ 6 KB. Long form of §1 with every source link:
> `docs/research/PUBLISHED_EVIDENCE_2026-11-15.md` (six-part review, 2026-11-15;
> its raw notes are in `/data/jack-research/research_notes/`).
> **(inf)** = arithmetic or reasoning from published numbers, measured by nobody.
> `$N` = `/data/jack-research/research_notes/Published evidence for building Jack`.

## 1. Which whole brain, and how to test brains fairly (re-checked 2026-11-26 against 2024–26 work; written 11-15; serves M3 stage 2, `prereg/stage2_a_vs_b.md`, J14, J18)

Long form: `/data/jack-research/reports/sec1_recheck_which_brain_2026-11-26.md` — it holds the whole 11-15 text (its findings stand unless corrected here) and the three parts' notes.

**Question.** Does the 11-15 advice — world model A the main brain, replay learner B "the cheap control", A 10–200× more sample-efficient — survive what was published since? What does A cost with no Mac?

**Already known**
- **CORRECTED: on vector-input control, modern model-free beats both world models under one setting.** DMC proprio, 28 tasks, 500k agent steps: MR.Q 835, TD-MPC2 783, TD7 566, DreamerV3 530, PPO 254; dog-run DreamerV3 4, MR.Q 569 (MR.Q 2025, proprio table, arxiv.org/abs/2501.16142).
- DMC-Hard (dog, humanoid), 1M, IQM: DreamerV3 0.008, SAC 0.069, TD-MPC2 0.464, BRO 0.772, SimbaV2 0.808. DMC-Easy: DreamerV3 0.813, SimbaV2 0.933 (arxiv.org/abs/2502.15280).
- **Plain SAC is not that learner.** 40 tasks, IQM at 100k / 1M: SAC 0.007 / 0.316, TD-MPC2 0.204 / 0.842, BRO 0.254 / 0.945 (BRO 2024, arxiv.org/abs/2405.16158). The gap is closed by layer norm, residual blocks, a ~5M critic, resets, 2–10 updates a step: no world model.
- Cost, HalfCheetah: MR.Q 4.1M params 49 training steps/s; DreamerV3 9.7M, 18; TD-MPC2 5.5M, 14 (MR.Q).
- Survival at 1M (Craftax-classic, pixels; arxiv.org/abs/2502.01591 Table 1, Fig. 4): PPO 4.0M, no memory, 47.4% reward; a bigger net WITHOUT memory 41.8; bigger + GRU 55.5 in 15 min; DreamerV3 201M 47.2 in 2,100 min; best transformer world model 67.4–69.7 in 759–2,749 min, 8 H100 (human 65.0). That world model imagining from step 0: 33.5; without real data mixed in: 55.0, the model-free score.
- Crafter's own episode files: Eat Cow in 1.5–6.5% of episodes at 0–50k, 15–37% at 1M; Collect Drink 24% → 64% by 100k and 90% at 1M (DreamerV2). Still water is learned ~10× sooner than moving food.
- Where memory IS the task Dreamer's line still leads (R2I beats 13 model-free baselines on POPGym, at 400M steps, arxiv.org/abs/2403.04253).
- One life, no resets: DQN, PPO, SAC all fail at 160M frames (AgarCL 2025, arxiv.org/abs/2505.18347).
- DreamerV3 sizes with a score: 12M (v2: proprio 871 at 500k, 0.3 A100-days) and 18M size S (v1) — not "8M". `size1m` is the official proprio default, at train ratio 1024: still no published score. Dreamer 4 (2025) learns from 2,541 h of video: pretrained, ruled out.
- Unchanged from 11-15: curriculum beats architecture (Avalon 0.716 vs 0.001); correlated noise; drive-difference reward; ~20 seeds per SD; 40k is under every foraging budget.

**Candidates**
- **B+ — a normalised, residual, resettable replay learner (BRO / SimBa / MR.Q class). Now the best-supported learner on vector inputs at ≤ 1M.** Proven on 40–118 tasks, one setting, ~5M params. Weak at: never raced on sparse reward, memory or survival; no memory. Smallest: SimbaV2's 0.3M critic (figure only). Expect: the forge's keep setting is this class already (rung M: resets 84%, cautious critic 82% late-eating lives vs ~6% plain; owner line 11-26 02:25) — that B, not plain SAC, is the fair stage-2 arm.
- **A — DreamerV3.** Now the weakest general agent on proprio control and behind a recurrent PPO on survival at 1M (47.2 vs 55.5) at 140× the minutes; its edge is memory and many or large senses. Nothing published supports A beating a B with memory at 1M.
- **B+ with a small GRU.** +8.1 points on Craftax-classic at no extra time; a bigger net without it LOSES 5.6. Needs state stored in replay (§5).
- **Dyna: B's backbone, plus a world model after 200k real steps.** The only way a world model won at 1M (+11.9 points, 50× the time, 58M params). No small version published.

**Ours beside theirs**

| setting | published | ours | verdict |
|---|---|---|---|
| B's net | LN, residual, ~5M critic, resets | MLP [256, 256] (`brain_b.py:39-40`); SimBa option `pilot_b2.py:492` | plain SAC is the record's floor |
| B's memory | GRU-256 | none | stage 2 would confound memory with the world model |
| A size / ratio | 12M / 512 (paper); 1m / 1024 (code) | size S / 512, compile on, `32-true` (`rows.py:42,354`) | same; official is float16 |
| A's cost | 0.3 A100-days per 500k | 5.0 T4-h per 100k (S); size1m 25.9 steps/s (`rows.py:46`) | 1M decisions ≈ 50 T4-h (S), ≈ 11 (1m) a seed (inf) |
| race | 1M, 10 seeds | 1M, seeds OPEN (`prereg/stage2_a_vs_b.md`) | at size S one seed > a Kaggle week |
| imagination starts | after 200k real steps | at once | unknown at our size |

**Known-answer check.** B+: dog-run or humanoid-walk at 1M (MR.Q 569 / 662, SimbaV2 class; plain SAC near 0) — the row that tells B+ from B; cost unmeasured on a 4-core kernel, timing cell first. A: §2. Mixed precision: sheeprl's maintainers read 114 → 32 ms a gradient step on an A100 (issue 336); fp16 on a T4 is unmeasured — a 20-minute cell.

**What the record does NOT answer**
1. A against B+ on sparse reward, memory or survival: published by nobody. That IS stage 2. Prediction (inf): B+ ≥ A unless the task needs memory.
2. Does A at train ratio 64 match 512 on proprio by 500k? No ablation exists (Crafter, 64 → 16: about −20%, read off a figure). One A2 row at ratio 64, ~1 T4-h a seed (inf): predict in band at 500k, a miss at 100k.
3. Any score for `size1m`: rows A1–A3 are the only ones anywhere.
4. JAX on a Kaggle TPU: no public run found; `jax.devices()` smoke first.
5. A hunger/thirst reward learned from scratch with step counts: still none found.

**Not verified:** MR.Q's hardware; which Craftax timings are one A100 (†) and which 8 H100; SimbaV2's DreamerV3 column is copied from MR.Q (500k agent steps, labelled 1M); values read off figures (SimbaV2 0.3M, DreamerV3 Fig. 6); EVAAA (2025 homeostatic benchmark, DQN/PPO/DreamerV3 baselines): README only; TD7 and CrossQ from rivals' tables; T4-hour sums are inference from our two timings; the 11-15 list (long form).

## 2. Building brain A (a Dreamer-style world model) correctly (2026-11-15; serves order J3, M3; plan part "His brain")

Long form: `$N/dreamer_engineering.md` (108 KB). Nothing below was measured by us.

> **RE-CHECK IN PROGRESS 2026-11-27 16:10 (RESEARCHER, rule 4: oldest section).** Against 2024–26 work and our own A rows (A1/A3 in band, A2 1m MISS at 100K, A2-v1S ABOVE, J28 cell 3 A1-r64 frozen, J14 A2-v1S-enc on W48). Parts: (A) train ratio + model size on proprio; (B) Dreamer-family 2024–26 + ports' published scores + plasticity/death; (C) T4 speed, mixed precision, v1-vs-v2 tables. Notes → `/data/jack-research/scratch_sec2r_{A,B,C}/`. Until replaced, the text below stands.

**Already known**
- The agent is fully specified by the official code, `danijar/dreamerv3` at commit `e01491f` (the 2024–25 "v2" agent of the Nature paper): 8-block GRU, 32 categorical latents; symlog encoder/decoder; two-hot reward and critic heads (255 bins, zero-init); continuation head; losses `rec + rew + con + dyn + 0.1·rep`, 1 free nat; 15-step imagination from all 1,024 replayed states; REINFORCE actor, entropy 3e-4, `max(1, S)` percentile return scaling; LaProp + AGC(0.3) at 4e-5, 1,000-step warm-up; batch 16 × 64; replay 5e6.
- For a 30-number observation: `size1m` ≈ 0.64M parameters, `size12m` ≈ 9.7M (inf).
- Load-bearing parts, by the paper's ablations: KL balance + free bits (no learning without them on Reacher Hard); return scaling with the `max(1, ·)` floor (sparse swing-up 850 → 200 without it); two-hot regression; reconstruction gives most of the learning signal (100 → 27 without it).
- The published small-scale scores are 2023 runs of the v1 agent at size S (train ratio 512, action repeat 2, 4 environments): 18-task proprioceptive DMC mean 442 / 626 / 757 at 100K / 200K / 500K env steps. **No score exists for `size1m`, nor for any port's v2 agent.**
- Fails at: directed exploration (BSuite exploration 0.01, deep_sea 0.00, with the 200M model); long memory (BSuite memory 0.62); shrinking (12M needs 38M Crafter steps for what 400M reaches at 7M).
- Train ratio (gradient work per decision): more is better up to the largest published setting, 1024. Cutting 512 → 32 on Walker Walk cost about two thirds of the score (one user report).
- **Death.** Only the v2 REINFORCE actor learns from termination. The v1 actor for continuous actions (backprop through the dynamics; the default in sheeprl's `dreamer_v3` and in dreamerv3-torch) cannot: InvertedPendulum stayed at return 4. The environment must mark true death and cut-offs differently.
- A needs-based precedent (EVAAA; termination when a need leaves range): Dreamer survived longest, agents without the needs inputs did not learn, and **some agents learned to end the episode when food could not be found** (reward: §13).
- Ports' silent bugs: terminated/truncated confused (broke CartPole), 1.6% of replayed actions corrupted, resume regressions, a README timing from a different agent.
- Speed is published nowhere for a T4, P100, CPU or Apple silicon. Arithmetic, ±2×: T4 `size1m` 20–50 gradient steps/s, T4 `size12m` 2–5, 4-core CPU `size1m` 0.5–2.

**Take as-is** (code and designs, never weights)
- Reference: the official repo at `e01491f` (its `configs.yaml`, `agent.py`, `rssm.py`, `scores/*.json.gz`). Runner: **sheeprl 0.8.5 `dreamer_v3_5`** with `algo=dreamer_v3_5_1M` or `_12M`: the only maintained port with the v2 agent, a CPU default, and resume with replay (`buffer.checkpoint=True`, `checkpoint.resume_from`). Six weeks old, no published scores: trusted only through the checks below.
- For him: every published value unchanged; one environment; action repeat 1 (a decision is 0.2 s); train ratio as high as the hardware sustains (≤ 1024), logged; `terminated` only on true death; a meal reward of order 1; needs, touch and rays as one vector; the REINFORCE actor; a 5,000–10,000-decision prefill; DayDreamer's arrangement (one process acts, one learns, shared replay).
- Later parts existing as code: R2-Dreamer's decoder-free loss (eye), R2I's memory core, Dynalang's one-token-per-step input (words), a Plan2Explore-style bonus or Curious Replay if exploration stalls.

**Reproduce first.** Published protocol: v1 agent, size S, ratio 512, repeat 2, 4 environments, 5 seeds; pass = our mean within ~2 SD at each point (official per-seed score file).

| task (proprioceptive DMC) | env steps | published mean ± SD |
|---|---|---|
| cartpole_balance | 100K / 200K | 794 ± 52 / 966 ± 17 |
| cartpole_swingup | 100K / 200K / 500K | 478 ± 20 / 778 ± 19 / 853 ± 13 |
| ball_in_cup_catch (sparse) | 200K | 949 ± 9; 5 of 5 seeds > 900 by 150K |
| reacher_easy | 200K / 500K | 929 ± 19 / 964 ± 7 |
| walker_walk | 200K / 500K | 799 ± 87 / 925 ± 34 |
| cartpole_swingup_sparse | 500K | 492 ± 368: ~3 of 5 seeds near 800, none above 50 before 100K |

Controls that must come out low: walker_walk at train ratio 32 (200–300 at 500K); BSuite deep_sea (0.00). Step-0 check from the code (v2, continuous actions): reward and value loss 5.541, `dyn` = `rep` = 1.0, actor std 0.893, predicted reward and value exactly 0. **Cost: the first three rows at 200K × 3 seeds were estimated at 25–60 T4-hours (one to two free-GPU weeks); a CPU cannot do it at the published train ratio.**

**The last 10%** (costs and predictions in the long form): (1) gradient steps per second on our hardware; (2) does a ~1M model reproduce the published S-model curves; (3) what train ratio the rover needs, and whether free-running the learner is safe; (4) can entropy-only Dreamer find food 3.8 m away in a 3,000-decision life (predicted: about half the seeds, more with near-first food or a disagreement bonus); (5) REINFORCE vs dynamics-backprop when death ends the life; (6) reward scale under `max(1, S)` (0.01 predicted to stay near random); (7) exact resume across 12-hour sessions; (8) a small Dreamer on BSuite's memory and credit tasks; (9) a life longer than any published run.

**Not verified:** the Nature article (arXiv v2 stands in); every speed figure (arithmetic); whether the paper's v2 tables are the 2023 runs (score files bit-identical, the author says v2 uses REINFORCE: unresolved).

## 3. His world, needs and staging: how worlds that taught agents were built (2026-11-15; serves J5, J7, M2; need clocks, fire, cooking, shelter)

Long form (file-and-line parameters): `$N/world_needs_curriculum.md` (114 KB).

**Already known**
- **The first meal is big and near.** One meal restores 44–100% of the need bar in Crafter, Avalon, Neural MMO and Animal-AI; at Avalon's easiest setting food is 0.5 m in front of the agent. Small meals (10%) work only with dense, fully sensed, instantly re-placed food and long lives (the two-resource ant: 10 items in 144 m², a per-type range sensor longer than the arena, a 6,667-step passive lifetime).
- **Life is long vs passive death:** capped at 30× the passive time to death in Crafter (10,000 vs 338 steps) and 34× in Neural MMO. Resources can be sensed from the start position. Every life is a new layout; no level repeats.
- **OUR WORLD IS THE OUTLIER** (from `experiments/drives.py`, arithmetic): a floor meal is 8% of the bar, the smallest of any world reviewed; two floor foods on 66.9 s timers supply 0.0024/s vs a basal drain of 0.0017/s, enough for about a fifth-time activity level; the ladder apple carries ~62% of the maximum intake. A sustained life therefore needs the HARDEST skill first, the reverse of every published teachable world. (Constants were calibrated on the humanoid: confirm for the rover.)
- **A success-gated difficulty dial has the largest measured effect:** 0.716 vs 0.001 on Avalon's eat task (50M steps); 57% wins vs none in another study. One ceiling per skill, each life drawn uniformly below it, up on success and down on failure in small steps.
- In a no-reset life, helpful starting conditions that wear off beat reward shaping on every task compared, and a world that changes by itself took success from 0% to 70% (Ecological RL). Regret-based level designers did not beat plain randomisation (Craftax, Kinetix).
- **Degenerate optima on record (§13):** stationary for ~1.3 billion steps under a "needs above half" reward (ForageWorld); ending a hard life on purpose (EVAAA; its −100 death penalty = ~44 steps at one bound); pulled off-task by a hand-added sub-reward.
- Expected order of learning (Crafter): drink or the adjacent resource first (5×10^4–10^5 steps), mobile food later (12–26% of episodes at 10^6), delayed farming never. 3-D with a curriculum at 5×10^7 steps (Avalon): eating 0.7 of human, moving to food 0.3–0.4, tools and composed tasks 0.00–0.16.
- From scratch in physics worlds: fort-building ~6×10^9 steps; shelter construction 3–12×10^9 samples; one shaped "put meat on a heat source" step ≤ 2M; tree-chopping 1M. **No from-scratch agent makes fire, cooks for its own hunger, or uses shelter for warmth.**

**Take as-is** (designs with code; parameters in the long form)
- Eat and drink by a distance test (0.5–1.0 m) with immediate re-placement at a new random position. Rays that report WHAT they hit (type plus distance). A smell that grows as 1/distance; a few thermal readings around the body. The hanging food needs a sense that can register it; horizontal rays cannot.
- Crafter's need arithmetic as the template: several bars, each emptied in ~190–280 steps awake, health lost only while a bar is empty and regained while none is, half-speed drains asleep; sleep as a commitment; a smooth day-night cycle. Random starting needs each life.
- Physics devices: grab by weld within 0.25 m; lock by freezing joints; transformations as "near, touching or held, then remove and spawn"; per-object temperature with cooked and burnt thresholds.
- Staging: scarcity is the dial, not the starting point. Meal size, item count, distance and respawn delay are functions of a ceiling d (today's world near d = 1); some lives unassisted from day one; progress judged only on fixed unassisted layouts at d = 1; an adult's help is a world setting tied to d, absent part of every life. Accept a stage only if a random or do-nothing policy neither always fails nor succeeds enough.
- Rungs in the order the record says is learnable: eat at the mouth → drive to visible food → food and water → warmth from an existing fire → night and sleep → one-contact gathering → feed a fire → cook → enter, close, then build a shelter.

**Reproduce first** (in our simulator, before our own findings are trusted)
- The two-resource ant as published (authors' `trp_env` and PPO settings): survival ~49,700 of 60,000 steps after 41.7M steps. Then the rover in the same arena: it should reach the cap sooner; if not, the fault is in his senses, actions or learner.
- Dense ray-sensed food collection: replay learners pass at ~1.4–1.7×10^5 decisions, PPO at ~7.5×10^5.
- The curriculum contrast: uniform sampling near zero, gated ceiling clearly above (published 0.001 vs 0.716).
- Minutes on a CPU: the authors' 8×8 grid where environment shaping beats reward shaping (0% vs 70%).

**The last 10%** (long form): a need-driven two-step chain in physics (free the fruit, then eat); warmth from a fire he must keep fed; making fire (predicted: not from undirected play at 10^7 decisions); cooking for his own hunger with no shaped reward (predicted: only with early provisioning); shelter for warmth and sleep (entering and closing: 10^6–10^7; building from parts: beyond a free budget unless much simplified); one value estimate per need; provisioning that fades in a persistent life.

**Not verified:** EVAAA's true need rates (runtime configs unpublished; its reward: §13); no ablation of meal size or food distance alone exists in any survival world; the entry settings at which a ray-sensing rover's random policy eats must be measured here.

## 4. Body, movement and simulation speed (2026-11-15; serves the paused `D1.0`, `T2.01`, M4, J1's speed table; the control path)

Long form: `$N/body_motor_simulation.md` (106 sources; a few figures measured here under load, marked).

**Already known**
- **Walking without mocap is routine only with thousands of parallel copies and PPO:** 2×10^7–4×10^8 steps; 4–6 minutes on an A100; 13–18 minutes for 3×10^7 steps on a free T4 (one community measurement). Policies are small: 4×32 up to 512-256-128.
- **In ONE world, a first walk takes ~1M steps with TD-MPC2, BRO or SimBa** (DeepMind Control humanoid-walk 628–672), and the score is at the random floor before ~200k. Plain SAC, PPO, A3C and DreamerV3 sit at the floor for 1–4 million steps. **`T2.01` failing since August is what the record expects at our budgets.**
- A CPU can do it: a 29-DoF humanoid walked with no mocap after 10–30M samples, 3×64 networks, 4–15 h on 8 threads, given a mirror-symmetry loss and helping forces that are withdrawn; without the help it only stood or fell (Yu et al. 2018).
- **The control path (D1's question) is largely answered.** In the two direct tests (SimBa, BRO) a bigger actor neither helped nor hurt; a bigger critic helped. Working humanoid controllers are 4×32 to 512-256-128; the largest from-scratch model acting every step on a humanoid is ~5M parameters. `D1.0`'s reading (the 0.5M controller leads the 57M trunk) is what the record predicts. The exact 57M contest is not in print.
- **A 0.2 s brain needs a reflex layer.** Published slow brains decide 6–10 times a second over controllers at 30–200 a second; no humanoid balances on decisions slower than 15–20 a second. He decides 5 times a second.
- Flat learners match hierarchy on shaped walking; hierarchy wins on sparse, long or object tasks, mostly through exploration. Where flat failed and two-level succeeded on humanoids (forage, carry, dribble) the low level came from mocap; Heess 2016 is the no-mocap precedent.
- **Hands on a walking humanoid are unsolved from scratch:** every HumanoidBench manipulation task misses its target; 10 billion samples gave 56% carry success; tool use emerged only with an abstract grab action.
- Mocap buys style and exploration, not return. It can judge from outside (a style reward) or be distilled into a module inside him; which is allowed is the owner's rule.
- Body curricula work when well scheduled (withdrawn forces, strong-to-weak torque limits, a child-then-adult body, hands frozen while learning to walk); a poor schedule can be worse than none.
- **Physics is not our bottleneck; our loop is.** A stock wheeled car steps 85,000 times a second on one loaded core of this server and a stock humanoid 20,000; the rover world manages ~2,800 because it uses the RK4 integrator (3.7× slower, measured) and does per-sub-step work and ray casts in Python. Published factors for fixing that pattern: 23× (logic moved out of Python), 2–3× (batching).

**Take as-is**
- Simulator: `integrator="implicitfast"` (or Euler); all sub-steps of a decision in one native call; `mujoco.batch`/rollout for several lives per process; `mj_multiRay` or rangefinder sensors for rays. T4: MJX with Brax PPO and the Playground environments. Mac: native MuJoCo on all performance cores (650K humanoid steps/s on an M3 Max, per the docs); no GPU physics there.
- Reflex walking controller, no mocap: PD joint targets at 20–50 a second; reward and weights from the Playground G1 joystick config, or Yu et al.'s three terms with the mirror loss; termination on torso contact; random pushes; a 3×64 to 512-256-128 MLP. Aids to copy: withdrawn balance and propulsion forces; an effort cost raised only while the task is met; hands frozen at first.
- One-world learner settings with humanoid evidence: SimBa (actor 1×128 block, critic 2×512 blocks, 2 updates per step, batch 256, weight decay 1e-2) or TD-MPC2 defaults.
- Two-level design: the brain emits a velocity or target command every 0.1–0.2 s to the reflex controller; a rover-trained brain can be kept as the high level over a new body-specific low level.

**Reproduce first** (cheapest first)
- Gymnasium Humanoid-v5 floor: random actions 110 ± 30 in 23.5 steps; zero actions 199 ± 3 in 40 steps (seconds).
- Raw physics speed on each of our machines (minutes). The MJX tutorial humanoid on a T4: its reward curve, wall-clock within 1.5× of 13–18 min.
- SAC on Gymnasium Humanoid: ~5,200–5,500 at 1M steps (one CPU kernel, hours).
- The 2018 CPU recipe: walks unaided, symmetry index ~0.01, and the no-help ablation only stands or falls (one 12 h kernel).
- DeepMind Control humanoid-walk with SimBa or TD-MPC2: 628–668 at 1M, ≤ 2 at 200k (3–30 T4-hours a seed).

**The last 10%** (long form has designs, costs, predictions): one human-like body that both moves and forages, flat vs a frozen reflex controller vs a learning one; how slow the brain may decide; the control-path contest as a clean, checkable test (small actor, large actor, large trunk as critic only, on humanoid-walk, pass mark 600); whether his own energy need yields an economical gait; staged release of the hands; a growing body; the rover brain moved into a walking body; hold, carry and strike with a grab primitive before fingers; throughput on our own machines (predicted: 500–2,000 rover decisions a second per 4-core kernel after the simulator changes, from 28).

**Not verified:** MuJoCo Warp on a T4, Kaggle CPU and M4-class figures (none published); the T4 MJX figure is one unmerged measurement; two source conflicts (SAC on humanoid-walk at 1M: 274 vs 2) are flagged in the long form.

## 5. Building brain B (the replay learner) and the PPO reference correctly (2026-11-15; serves order J2, M3; plan part "His brain")

Long form: `$N/control_arm_engineering.md` (223 KB; timings measured here under load).

**Already known**
- **The RL Zoo benchmark table is ONE run per entry, often the best case.** Open RL Benchmark's public run store holds 10–22 seeds per configuration; use those: SAC Pendulum-v1 (row 1; 13 seeds); SAC on sparse MountainCarContinuous-v0 at 50,000 solved on 9 of 10 seeds; PPO Pendulum-v1 at 102,400, −184.5 ± 80.7; DQN CartPole-v1 392 ± 150 where the table says 500. **§1's "Pendulum ~−157 in 20k" came from that one-run table: use the rows below.**
- Plain SAC (2×256, 2 updates per decision): strong on easy low-dimensional tasks, unreliable early on sparse ones (rows 5–6; swingup_sparse 93.8 ± 166.1 at 50K; 10 seeds, SimBa repository).
- PPO is slow & seed-fragile: default SB3 PPO has median 127 over 28 DMC tasks at 500K decisions vs 613–927 for replay & model-based learners; on sparse swing-up three sources give 7, 76, 391. Published optimum: About four parallel environments.
- **Gradient updates are not free on our CPUs** (upper bounds): one SAC update at batch 256 ≈ 50 ms of an ARM core in PyTorch SB3, ~20 ms in SBX (JAX) at 2×256, ~1.9 ms in SBX at 2×64. Big-critic methods (CrossQ, SimBa, BRO, MR.Q) need a GPU.
- Direction-holding exploration solves the sparse classic task: Zoo recipe = gSDE, FIXED entropy coefficient 0.1, discount 0.9999, no random warm-up. Traps: the pink-noise package also changes the noise inside SAC's updates & consumes its sequence three times per step; SBX's `use_sde` is a held noise draw, not SB3's gSDE.
- Memory in a replay learner has one well-documented design (Ni et al. 2022): separate GRU-128 encoders for actor & critic, 64-step windows (but row 7's cited Pendulum config is LSTM-128 on whole 200-step episodes — see the row 7 correction below); fails with a 5-step window (~−1,380) or a shared encoder (~−422). Costly: ~0.3 s per CPU update. Stacking 4 observations equals a GRU on position-only tasks, fails on recalling an early cue.
- A Q-learner over 9 joint actions is a legitimate alternative for a 2-dimensional drive (DecQN matches D4PG on 1- & 2-dimensional tasks). No maintained library has prioritised replay or a recurrent off-policy learner.
- Silent bugs to test: time limit vs death, next observation at episode ends, tanh correction, action scaling, target updates, normalisation statistics, evaluation mode, recurrent state.

**Take as-is** (code, never weights)
- Fresh venv for the checks (not his world's): Python 3.12; `stable-baselines3`, `sb3-contrib`, `rl-zoo3` 2.9.2; `sbx-rl` 0.28.0; `gymnasium` 1.4.0; `mujoco` 3.17.0 with `dm-control` 1.0.50 (later need MuJoCo 4.0); `shimmy` 2.0.1; `pink-noise-rl` 2.0.1. No `box2d` wheel for Linux aarch64 (LunarLander only on Kaggle).
- **Arm B:** SBX `SAC` (SB3's lacks LayerNorm), `policy_kwargs=dict(layer_norm=True)`, 2×256, learning rate 3e-4, batch 256, replay ≥ whole run, tau 0.005, gamma 0.99 to start, 5,000–10,000 warm-up decisions, 1–2 updates per decision, observations normalised, no resets. Death is `terminated=True`, every administrative stop `truncated=True`.
- Exploration for B: the Zoo's sparse-task recipe (PyTorch SB3 holds its public record) or pink noise with a life-long sequence; never both; either must pass row 3 in the library actually used.
- B with memory: first the last 4 observations stacked; then Ni et al.'s design, published sizes. PPO reference: SB3 defaults, `VecNormalize`, four simulator processes, gSDE sample period 4. A teacher showing food: keep demonstrations in replay for good, draw a fixed share of each batch from them (RLPD, DDPGfD).

**Reproduce first** (references are distributions; "accept" = range for a 5-seed mean)

| # | task | run | steps | target |
|---|---|---|---|---|
| 1 | Pendulum-v1 | SAC, Zoo entry (lr 1e-3): unmodified library, then ours | 20,000 | mean of all 100 training episodes −298.4 ± 10.3; accept −314.7 to −282.1 |
| 2 | Pendulum-v1 | PPO, Zoo entry | 102,400 | last-100-episode training return −187.6 ± 17.0; accept −213.2 to −162.0 |
| 3 | MountainCarContinuous-v0 (sparse) | SAC, Zoo entry with gSDE, PyTorch SB3; any SBX or pink-noise variant too | 50,000 | deterministic evaluation > 90 on ≥ 3 of 5 seeds (public: 9 of 10 in 93.4–95.3) |
| 4 | MountainCarContinuous-v0 | PPO, Zoo entry | 20,000 | training return 87.5 ± 4.2 |
| 5 | DMC cartpole-swingup, action repeat 2 | plain SAC 2×256, 2 updates per decision | 50K decisions | 853.0 ± 9.2 |
| 6 | DMC cartpole-swingup_sparse, reacher-easy, reacher-hard | same | 150K decisions | 776.5 ± 30.2, 968.0 ± 23.2, 955.8 ± 30.5; sparse near zero at 50K on most seeds |
| 7 | velocity-only Pendulum | recurrent SAC, Ni et al.'s configuration | 51,000 | −168.6 ± 3.5; both controls must fail |

**Row 7 CORRECTED (11-18 17:40, JACK Next-up 2, read at source before any rule):** the cited config (`pomdp-baselines@e7c19c32`, `configs/pomdp/pendulum/v/rnn.yml`, cloned to `/data/vendor_src/pomdp-baselines`) is **LSTM**, not GRU: `seq_model: lstm`, `separate: True`, `sampled_seq_len: -1` (the whole 200-step episode, not a 64-step window), batch 32 sequences, embeddings obs 32 / action 8 / reward 8, rnn 128, layers 128-128, lr 3e-4, **gamma 0.9**, tau 0.005, auto-entropy (alpha lr 3e-4), 1 update per env step, 5 random init episodes + 250 × 200 = 51,000 steps, 20 deterministic eval episodes. The paper's own source (arXiv e-print 2110.05038, `paper/appendix.tex` §results_arch, Fig. `separate_vs_shared`) gives Pendulum-V **only as a figure**: `sac-lstm-200-oar-separate` reaches ≈ −170 by ~13k steps and holds to 50k; the shared twin ends ≈ −350…−750 with a wide band. **Neither "−168.6 ± 3.5" nor the 5-step "≈ −1,380" appears in the paper text or this figure**, and the seed count is not stated there — both are unverified: the only other source, the authors' results `data.zip` (Google Drive id 1dfulN8acol-qaNR2h4PDpIaWBg9Ck4pY, linked from `docs/plot_curves.md`), returned HTTP 404 on 11-18. Until traced, row 7's rule must take its band from a figure reading of the separate curve at 50k, say so, and use the shared arm as the reported must-be-worse comparison, not a 5-step control with no published number.

**The last 10%** (5 seeds per arm, fixed budget, one outcome: meals per 1,000 decisions over the last 50,000; costs in the long form). First, ~2 kernel-hours: how often a random & a noise-driven policy reach food in a life; gap between meals for a hand-written ray-follower (sets the discount); whether four simulator processes give 4× the decisions; Kaggle update timings. Then: narrow network at high update ratio vs standard; which exploration finds food first (predicted: SAC's automatic entropy coefficient collapses below 0.01 before the first meal); whether the rover needs memory; continuous actor vs nine discrete actions; gap to PPO; whether one demonstration of eating helps; 3-step returns; discount 0.99 vs 0.995.

**Not verified:** the pink-noise paper's tables & the replay-ratio-barrier paper (blocked); all our timings (one sample, load 5–13). Three note sections condensed from sub-drafts, 12 values spot-checked.

## 6. Memory, sleep and learning for a lifetime (2026-11-15; serves M7; plan part "Memory and sleep")

Long form: `$N/memory_sleep_continual.md` (115 KB). Nothing below was measured by us.

**Already known**
- **Working memory:** the GRU is the best-supported small core (POPGym aggregate 0.349, LSTM 0.255, memoryless 0.067). Linear-recurrent cores (FFM, S5, LRU) match it, training faster. Reward-trained memory falters at lengths around 250 steps, and nothing assigns credit across 250 steps. With food in view, memory gives no benefit.
- **Memory trained from replay must see the whole dependency:** chunks shorter than it give zero success (R2I). Store the recurrent state, add a burn-in (R2D2: 40 of 80 steps), never cross a death, feed previous action and reward.
- **Event store (the diary):** NEC's value table (learned embedding → best return, read by nearest neighbours) gives a large early gain that fades: 16.7% median Atari score at 1M frames vs DQN's −0.7%, overtaken at 40M. Recall of specific past events worked only when what is written is trained by prediction (MERLIN, HCAM), at ~10^9 steps. **Retrieval is always by learned similarity, never by word matching.** That is why our diary scores 0 on reworded questions (inf).
- **Across episodes:** memory emerges when it is not wiped, the objective spans episodes, and worlds differ between groups of episodes (RL²: second-episode paths 25% shorter after 1,000 mazes). A 100-episode scene memory added +2.1%. In one fixed world, weights fed by retained replay carry "where the tree was" (Mirowski).
- **Death:** rare, so it leaves replay and is repeated unless a few death examples are kept forever (Lipton).
- **Sleep:** learning while not acting is the most reliable gain on record (Dyna maze: 25 episodes → 3 with 50 planning steps; MuZero Unplugged 126.6% from 20M frames, 99.5% re-analysed). Concentrated: 75,000 critic updates per 10,000 steps beat the same spread evenly, only with dropout, worse above 75,000 (SPEQ, one preprint). Data-free "sleep replay" 48% vs 80% for keeping 0.75% of data.
- **Not forgetting:** a 50-50 mix of new and reservoir-kept old experience restores 31.4/8.0/18.1 vs 18.0/5.0/10.9 (CLEAR); insensitive from 50-50 to 75-25; works keeping 5M of 900M frames. Keep by reservoir or coverage, never by surprise or reward. Replay is not always enough: SAC replaying everything on Continual World 0.12 vs PackNet 0.80. One preprint: a 17M Dreamer's actor forgets while its model does not.
- **Staying able to learn:** reset the fast network to a slow averaged copy (Hare and Tortoise: 0.826 → 0.891, free). Hard resets make replay ratio 8–16 usable (BBF). Shrink-and-perturb made one learner worse (0.139 vs 0.243).
- **Measuring what was kept:** retention + relearning savings + jumpstart, vs naive and wrong-memory controls. A probe beats latency (watermaze: 36.0% in the right zone vs 12.5% chance; latency saving not significant).

**Take as-is** (code and designs, never weights)
- GRU state carried all life; stored-state replay with burn-in; in A, store latents per chunk and carry state across chunks. Long-memory drop-ins for B: FFM, S5, LRU.
- Diary: keys = the world model's latent state, values = what happened (return, need change); k nearest neighbours, inverse-distance kernel (NEC: ≤5×10^5 rows, 50 neighbours); kept at death.
- Replay: recent FIFO plus lifetime reservoir, sampled half and half; a permanent small store of deaths and meals; if the world changes locally, evict by neighbourhood, not age.
- Sleep: burst training of the world model from replay, re-encoding old sequences; a capped dose of imagination training (SPEQ's 75,000 per 10,000 as the upper mark); slow-copy reset during sleep.

**Reproduce first**

| what it checks | task | published |
|---|---|---|
| offline updates | Dyna maze, γ 0.95 | ~25 episodes to near-optimal at n = 0, ~5 at n = 5, 3 at n = 50 |
| recurrent core | passive T-Maze, DDQN | LSTM solves short lengths, falters by 250 |
| memory module | POPGym tasks | GRU beats MLP wherever memory is needed |
| chunk length | bsuite memory_len | 64-step chunks fail past 64; 256 succeed |
| replay buffer | Split MNIST, 200 samples | 80.43 ± 1.89; no replay 19.60; joint 95.57 |
| state across episodes | RL² bandits, n = 100, k = 5 | 78.7 (Gittins 78.3, random 49.9) |

Rows 1, 5, 6: CPU minutes; rows 2–4: CPU-hours.

**The last 10%** (at 100,000 decisions per kernel-hour)
- How long a memory his foraging needs: zero the state every k steps at evaluation only (k = 1 … never), food in and out of ray range. ~6 kernel-hours. Predicted: flat with food in view, hundreds of steps when search is needed.
- Which store carries a life's learning in a fixed world: fork 10 seeds after 1M decisions into all kept, replay wiped, weights reset with replay kept, all wiped. ~220 kernel-hours. Predicted: wiping replay changes little at first; forgetting of rarely visited places follows.
- Does anything beyond weights and replay help him find the last life's food: food re-drawn every K lives; state carried or reset at death; diary kept, wiped or swapped; probe at the start of each life. ~240 kernel-hours. Predicted: no effect if food never moves; value diary helps earliest if it moves.
- Concentrated vs spread sleep at equal updates; model-only vs model-and-actor sleep. 160 kernel-hours (B), ~18 T4-hours (A). Predicted: little difference for the model; model-only safest.
- Also open: re-localising after rebirth (2 kernel-hours); a small Dreamer's actor forgetting (~24 T4-hours); sleep tied to his sleep NEED; diary keys drifting as the model learns (inf: re-encode during sleep).

**Not verified:** SPEQ and the Dreamer-actor result are single preprints; episodic-LSTM and HCAM across-episode numbers are figure-only; nearly every memory result is from far larger agents.

## 7. Ear, words and advice (2026-11-15; serves M5, M6; plan parts "Ear" and "Words and advice")

Long form: `$N/ear_words_advice.md` (99 KB). Nothing below was measured by us.

**Corrections to the first review (§1):** (1) "No published agent grounds words from raw speech from scratch" is too strong: a ViZDoom agent follows spoken "go to the X" through an FFT ear (Hegde 2021), and robots go to the object named by one of 4 real Speech Commands words (Chang 2019). Unpublished: with a world model, no word-label loss, online in one life, with advice. (2) Dynalang is not ~10M: 281M on HomeGrid, 148M on Messenger; HomeGrid cost 3.75 A100-days. (3) SoundSpaces' 0.601 → 0.709 is SPL (path-weighted success), not success rate.

**Already known**
- **Ear:** every from-scratch hearing agent feeds a spectrum (FFT, log-mel, MFCC), not the raw waveform (~5× slower and erratic), through a small encoder into the same recurrent core. Non-speech sound is learned in 3×10^7–10^8 steps.
- **Few words:** 9k–100k weights reach 94–97% on 12 words for unseen speakers with ~3,000 recordings per word from ~2,600 speakers; 95% at 1,000 per word, 90% at 125, falling fast below ~50. Synthetic speech alone: 56.7% (92 voices) to 92.6% (voice-cloned, recogniser-filtered).
- **Words tied to action:** 4 words, 1,000 recordings each, 1.5–5×10^6 steps, 0.87–0.945 on unheard recordings; **without a loss on the sound channel, 0.135–0.185** (Chang). Reward alone teaches words nothing after millions of episodes (Hermann).
- **Self-supervised ear:** predicting future audio on 100 h of clean speech halves phone errors (ABX 10.95/20.94 → 6.24/8.17); on realistic child-centred audio, lexically at chance after 1,024 h.
- **Narrator:** after-the-fact descriptions turn failure into ~80% success, work at 1% of experience, keep 80% of the benefit at 50% noise; a noisy pretrained labeller (54.7% precision) lifted naming 7% → 64%.
- **Advice (the open question):** (a) **language that merely predicts outcomes is used only when a world model must predict it.** HomeGrid hints raised Dynalang from ≈3.3 to ≈4.8–6.0 at 50M steps (figure reading, 2 seeds). Model-free R2D2 used only reward-correlated language and got worse with more. (b) **An agent raised on good teachers cannot doubt a bad one:** the largest social-learning agent "happily follows an incorrect demonstration". (c) **Learned trust exists only in tabular form, with the reliability given to the agent:** Advise weights each label by a stated accuracy C; an estimate near 0.5 is worth nothing. (d) **No deep-RL study tests hints that are sometimes false.** (e) Pretrained models fit best as narrator, labeller and judge; a language-model reward was exploited ("hallucinogens to dream of the goal state", Motif).
- **Voice:** syllabic babble in ~7,200 one-second trials; recogniser-guided syllables at 55% word accuracy vs 81% for natural speech.

**Take as-is** (never weights inside him)
- Log-mel or MFCC front end (40 bins, 25–30 ms window, 10 ms hop); 2–3 conv or FC-256 layers; concatenated into the one recurrent state.
- A loss on the sound channel: in a world model, predict next-step audio features. Words as one unit per step, predicted like any sense; reward only from needs (Dynalang's "advised, not commanded" design).
- ±100 ms time shift, noise at probability 0.8, many voices; filter synthetic clips through a recogniser.
- Parent as narrator via TTS, asynchronous and cached (ONI); a few percent of events is enough to start. Stages: eye → ear by prediction → frequent isolated words → multi-word → speaking.
- Do not take: T5/BERT/SentenceBERT, wav2vec/HuBERT features, goal-suggestion-with-reward, teacher-policy distillation (command channels).

**Reproduce first**

| stage | task | published |
|---|---|---|
| ear | Speech Commands, 12 classes, speaker-disjoint; DS-CNN 24K / BC-ResNet-1 9.2K | 94.4% / BC-ResNet-1 **v2 96.9 ± 0.30** (v1 96.6 ± 0.21) — corrected 11-18: "96.6–96.9" was v1 then v2, not a band (arXiv:2106.04140 e-print `main.tex` l.52, Table l.170) |
| data budget | 35 words at all / 1,000 / 125 recordings per word | 97.4 / 94.8 / 90.3% |
| words in an agent | Chang's 4 words → 4 objects, 5×10^6 steps | 0.945 with label loss, 0.870 without (arm); 0.920 vs 0.135 (navigation) |
| narrator | ACTRCE hindsight descriptions | ≈80% vs no learning; 1% of frames enough |
| trust | Advise, 5×5 Pac-Man, three teacher settings | at or above baselines; assumed C = 0.55 ≈ no feedback |

Keyword rows: one CPU session each. Chang's rows: 1.5–5×10^6 steps.

**The last 10%** (costs inf)
- Can prediction replace the word label in his ear? Reward only / + label loss / + next-step audio prediction, ≥5 paired seeds, 5×10^6 steps. Predicted: ≈ no-sound / ≥0.85 / between, above 0.6.
- One voice or many (1, 4, 16, ~100 TTS voices). CPU hours. Predicted: ~100 voices × ~100 clips gives 55–60% on real speech.
- Hearings in a stream: no replay / recent only / two-tier. Predicted: no replay near chance; ≥100 hearings per word for ~90%.
- Words as a sense in a 1–10M world model (15 tokens, then spoken). T4 days. No published number.
- True, absent and false hints, hungry and sated. Predicted: help a world model only; true-only training does worse than none under false hints; a sated agent ignores food hints (untested anywhere).
- Whom to trust: two voices (90–100% right vs 50% or adversarial), then swap. The open result.
- A voice that serves his needs and tells the truth. Stage 3 has no precedent.

**Not verified:** HomeGrid hint scores are figure readings from 2 seeds; ages and hours for infant hearing are broad ranges; no live TTS parent heard through a from-scratch ear has been published.

## 8. The correct test method (2026-11-15; serves stage 2 and every step-4 experiment)

Long form: `$N/test_method.md` (163 KB). Power figures are the notes' own exact enumeration or simulation (inf).

**Already known**
- **Few runs give little.** On one task, bootstrap intervals need 20–30 runs for their stated coverage. Welch's t runs at ~2× its level on skewed scores. Permutation tests are exact at any n but cannot reject below 4–5 runs per arm.
- **Power for learn-or-not outcomes** (exact test, two-sided 0.05): 10 runs per arm for 0.8 vs 0.2; 15 for a 50-point gap; 25–30 for 40 points. A 30-point gap has 49% power at 20 per arm. A time-to-threshold (log-rank) test is never worse and far better when both arms usually succeed.
- **Peeking** after every 5 runs without correction gave 12% false positives. Two looks at 0.025–0.0294 hold 5% and save a quarter to a third of the runs when the effect is large.
- **Selection:** the best of 8 identical settings on 3 seeds looked 38 points better than it was; settings picked on tuning seeds did 4–5× worse on fresh ones. Fresh confirmation seeds remove the bias.
- **Pairing** on shared world seeds helps learn-or-not tests only if between-arm correlation exceeds ~0.25.
- **Lesions mislead:** 90% of pixels removed cost 13 points with retraining, 76 without. Retrain without the part.
- **World model vs model-free:** no single fair budget. Fix decisions as the axis, state update ratio, size and repeat, show compute, and add a matched-update control.
- **AdaStop warning.** PyPI 0.1.3 rejected 9.7–11.8% of same-against-same comparisons at a stated 5%. The repository fix of 2026-10-17 holds 4.2–5.3%. No fixed release exists; both builds say 0.1.3. On 0/1 scores the bug hides (2.8–6.0%).

**Take as-is**
- SciPy `boschloo_exact`, `permutation_test`, `logrank`, `ecdf` with censoring, `binomtest(...).proportion_ci("exact")`; statsmodels Holm.
- AdaStop only from `git+https://github.com/TimotheeMathieu/adastop@f272065a88e36ba571ce41161232d2d9ec4be3ac`, α = 0.05 set explicitly (default 0.01), N = 5, K = 4–6, run once for all comparisons.
- rliable for plots only at this n. Patterson's 20-error checklist; Eimer's 17-item tuning checklist.
- Outcomes: time to threshold with censoring; success proportion; expected running time = decisions spent ÷ successes; normalised score (score − random)/(oracle − random).

**The staged design (two brains + PPO, ~100 kernel-sessions)**

| stage | sessions | content |
|---|---|---|
| 0 | ~6 | harness self-checks; each arm reproduces its published number; throughput fixes T |
| 1 | ≤24 | pilot + equal tuning on reserved seeds: default + 3 alternatives × 3 seeds at T/2; freeze configs, threshold, correlation |
| 2 | ≤57 | fresh seeds: A 20, B 20, PPO 10; random, statue, untrained twin, scripted forager (3 each), shuffled-input (5) |
| reserve | ~13 | re-runs; B at A's update ratio (5 runs) if A wins |

- **Primary outcome:** decisions until W consecutive lives average ≥ θ meals (θ = 5; scripted forager 24–27, random 0–0.2); unfinished runs censored at T.
- **Primary test:** A vs B, permutation log-rank, looks at 10 and 20 runs per arm, each at 0.025. Sensitivity: Boschloo on runs that crossed by T. Futility: at look 1, ≤1 run past threshold in both arms → "not learnable at this budget". PPO comparisons with Holm.
- **Gates:** floor arms ≤1 success in total; scripted forager must pass; else void. No seed replaced; NaN and crashes are outcomes.
- **Detects:** 4× faster 0.93–1.00; 3× 0.74–0.94; 2× 0.34–0.63. Write in advance that "not distinguished" means "choose on cost and simplicity". A's cost (T4-bound) may force a lower T for all arms; never fewer runs.

**Reproduce first** (harness self-checks #1–#13): (1) same-against-same through the real pipeline, ≥1,000 repetitions, at ≤ the stated rate; (2) a handicapped copy (learning rate 0) detected with the planned power; (3) calibration arms at floor and ceiling; (4) same seed twice → identical logs, harness writes the seed list; (5) run k sees the same world in every arm; (6) resume across a kernel boundary = uninterrupted run; (7) every decision counted, incl. warm-up and prefill; (8) reward-forever world gives 1/(1 − γ), true death no bootstrap, last observation true; (9) probes on or off leave training logs identical; (10) outcome code right on synthetic logs and edge cases; (11) analysis script frozen and hashed in the pre-registration; (12) tools on a known case: 4/15 vs 0/15 gives Fisher 0.100, Boschloo 0.043; (13) runs analysed = runs launched.

**The last 10%** (conservative choices, no literature answer): one run = one observation, never lives; primary outcome from normal living, frozen probes only as diagnostics; death ends the return, the 3,000-decision cap is a truncation; analyse unpaired unless pilot correlation > ~0.3; retention across rebirth = first window after vs last window before death, descriptive only; word every result "on this world, at this budget, as configured".

**Not verified:** no RL paper uses exactly this censored two-look combination; real learning-time distributions are unknown until the pilot; AdaStop's fixed build had one borderline 0.064 that needs re-checking.

## 9. A battery of standard tests for brains (2026-11-15; serves stage 2)

Long form: `$N/capability_battery.md` (144 KB). Install checks and a few timings were run on this server under load (marked M).

**Already known**
- bsuite and the DeepMind Control proprioceptive tasks are the only places where all three families have numbers on the same task. A = Dreamer, B = DQN (discrete) or SAC (continuous).
- **A's references come from 10M–200M models at GPU budgets** (bsuite: 200M). Run its checks at the shortest published points or read them relative to B.
- bsuite is not "<30 CPU-min each": under 30 min per configuration for model-free learners in the paper, 20–290 min in a later measurement; the 468-configuration sweep is not cheap. Two baseline tables disagree by 10–35 points on some categories.
- Sparse tasks have seed SDs of 300–390 on a 0–1,000 scale: fewer than 5 seeds per arm cannot tell correct from broken. Dense tasks: 2–3 suffice.
- Version traps move scores more than family gaps: frames vs decisions (action repeat 2), v2 vs v4 environments, MuJoCo 3 vs 4, training vs final metrics.

**Take as-is: the battery, cheapest first** (A / B / PPO references)

| # | task (faculty) | package | budget | A | B | PPO |
|---|---|---|---|---|---|---|
| 1 | catch / cartpole (basics, harness) | bsuite 0.3.6 | 10k / 1k episodes | 0.96 / 0.93 | DQN 0.92 / 0.85 | 0.91 / 0.84 |
| 2 | noise / scale variants | same | 5 levels × 4 seeds | 0.62 / 0.82 | 0.51 / 0.73 | 0.54 / 0.60 |
| 3 | deep_sea / mountain_car | same | 21 sizes; 1k ep. | 0.00 / 0.92 | 0.00 / 0.93 | 0.00 / 0.10 |
| 4 | umbrella_length / discounting_chain | same | 23 delays; 20×1k | 0.78 / 0.40 | 0.39 / 0.25 | 0.87 / 0.24 |
| 5 | memory_len / memory_size | same | 23 / 17 levels | 0.65 / 0.59 | 0.04 / 0.00 | 0.17 / 0.47 |
| 6 | umbrella_distract | same | 23 counts | 0.74 | 0.30 | 1.00 |
| 7 | cartpole swingup / walker walk | dm_control 1.0.52 | 100k–500k, repeat 2 | 478 ± 20 at 100K (official file; battery notes say 528 ± 36: recompute) | SAC 835 ± 8 / 708 ± 167 at 100k | 266 ± 27 / 77 ± 33 at 500K |
| 8 | swingup_sparse / cup catch / pendulum | same | 500k, ≥5 seeds | 504 ± 393 / 966 ± 9 / 785 ± 150 | SAC 531 ± 376 / 978 ± 0 / 408 ± 301 | 2 ± 2 / 503 ± 299 / 0 |
| 9 | position-only CartPole, Repeat tasks | popgym 1.0.7 | 15M | curves | curves | GRU 1.000; MLP 0.27–0.72 |
| 10 | passive / active T-Maze | Memory-RL code | 0.2–2M per length | — | LSTM fails at 250 | — |
| 11 | continual CartPole | C-CHAIN protocol | 10 × 0.16M | — | — | kept 61.8; reset 299.9 |
| 12 | Craftax-Classic | craftax 1.7.0, T4 | 1M | 47.2 ± 3.9 | — | PPO-GRU 55.5 ± 1.3 |

Cost: rows 1–6 run on a CPU in minutes to hours per configuration for B and PPO; A in bsuite has no CPU figure. Row 7: simulation 1–3 min per 500K steps per core (M); B ~1 hour per 100k decisions on one core (M); A 0.3 A100-days per 500K frames. Row 12: PPO 15–26 min on 8 H100; DreamerV3 2,100 min; T4 unpublished.

**Reproduce first: cheapest checks that catch a broken build**
1. bsuite catch through the real logging path: PPO 0.91, DQN-style 0.92, Dreamer 0.96; a random agent must give 0.00 (catch), 0.04 (cartpole), 0.10 (mountain_car) (reproduced here, M).
2. PPO CartPole-v1 → 500 within 100k steps (M: 128 s); SAC Pendulum-v1 at 20k vs the seed distribution in §5, not the one-run table.
3. DMC cartpole swingup, repeat 2: B 835 ± 8 at 100k; A at the official 100K/250K points (§2 table).
4. Signatures: memoryless fails memory_len above 1; recurrent passes up to its window and fails beyond (recurrent A2C 0.70 at a 30-step window); deep_sea 0.00 for all; PPO 0.10 on mountain_car while DQN and Dreamer reach 0.92–0.93.

Order: rows 1–6 and 11 (no GPU) catch most B and PPO bugs; row 7 is A's first real check, before any in-world Dreamer run; rows 9, 10, 12 measure capability, only after rows 1–8 pass.

**The last 10%** (no small standard test or no reference number)
- Using a newly added sense: nothing for any family. Design (inf): train to plateau without it; switch on; compare with had-it-from-birth, a same-shape noise channel, and never-had; report the dip and steps to catch up; switch off again.
- Continual learning for A and B at small scale (two Dreamer MiniGrid studies disagree; no SAC/DQN result on continual Gym control).
- A below 10M: `size1m` has no score anywhere.
- Long-delay credit and long memory in continuous control; needs-based survival with continuous actions at small cost; noise and reward-scale robustness for continuous actions.
- In-world battery (inf): one faculty per task, one difficulty number, held-out evaluation seeds, random + own-senses scripted + public-family anchors, scores normalised to [0, 1]; for needs, EVAAA's two probes (choose the resource matching the larger deficit; reverse after an imposed need change).

**Not verified:** no learner timed on a Kaggle x86 core or a T4; bsuite per-level targets are inferred from per-experiment means; the battery and Dreamer notes give different Dreamer cartpole numbers at 100K (528 ± 36 vs 478 ± 20) and attribute the official curves to different model sizes; recompute from the per-seed file before use.

## 10. The Mac mini M6: expected speeds, the software path, and the first day (2026-11-15; serves the decisive test in PLAN, order J5, and the move to the Mac)

Long form: `/data/jack-research/reports/Mac mini M6 speed for Jack.md`. No MuJoCo, SAC or Dreamer timing exists for any M6; all workload numbers are scaled **(inf)**.

**Already known**
- **Measured on the M6:** Geekbench 6 4,698 / 21,302 (A1 core ≈ 1,100; our VM ≈ 3,500; Kaggle 786 / 1,687). Cores: 2 super + 4 performance (4.5–4.8 GHz) + 6 efficiency (2.9 GHz). Wall power 2.1–4 W idle, ~55 W sustained, 70–74 W peak; fan ≤ 42 dB(A). 32 GB max, not upgradable.
- **MuJoCo, scaled:** fast M6 core = 2.5–4× an Ampere core (Geekbench discounted by the one Apple MuJoCo figure: M3 Max 650K humanoid steps/s, all cores). Jack's world per fast core: RK4 7,000–12,500 sub-steps/s, implicitfast 26,000–48,000. Loop today 85–150 decisions/s (RK4), 160–250 (implicitfast); after loop fixes 300–800. Whole chip 4.5–8× our VM, 10–16× a Kaggle session; efficiency cores add +15–35%.
- **SAC update (2×256, batch 256), scaled:** SB3 on CPU 3–15 ms; SBX 2–7 ms; MPS no faster. 1M-decision B run: 1.7–5.6 h; **20 runs take 6–22 h on six fast cores (inf).**
- **Dreamer, scaled, steps/s for 0.64M / 9.7M:** JAX on CPU 2–6 / 0.15–0.4; PyTorch on CPU 3–10 / 0.2–0.6; MPS 3–12 / 0.8–2.5; MLX (unwritten) 6–25 / 1.5–5. T4 arithmetic: 20–50 / 2–5. So the Mac is **0.2–0.6× a T4 on PyTorch and 0.3–1× on MLX**. Basis: T4 beat M3 Pro 1.6–1.8× on ResNet-50; a GRU cell ran 2.1–3.5× slower on MPS than CPU (M3 Max).
- **Decisive test for A at 0.64M, full ratio (inf):** 460–1,860 Mac h (3–11 weeks). T4 alone: 4–9 weeks; **the two together need 1.6–5 weeks.** 9.7M at full ratio: 2,200–7,000 Mac h, unaffordable anywhere.
- **No Dreamer code has been reported running on Apple silicon.** JAX has no Mac GPU route (jax-metal abandoned). MuJoCo on Mac GPU: unusable.

**Take as-is**
- Physics: native MuJoCo on CPU, one process per world, 6 on fast cores (≤ 12 total), `nstep=40` in one call. No GPU physics.
- B: SBX on CPU, `OMP_NUM_THREADS=1`, 4–6 runs side by side; SB3 on CPU as reference. Not MPS.
- A 0.64M: keep the fastest of official JAX (CPU), sheeprl `dreamer_v3_5_1M` (CPU), sheeprl on MPS. A 9.7M: sheeprl on MPS. MLX port only after it reproduces §2's DMC rows.
- **Never `nice 19`, `taskpolicy -b` or a launchd `ProcessType` of Background**: background priority pins work to efficiency cores (2–3× loss, no error). Cap worker counts in code.
- `sudo pmset -a sleep 0 disksleep 0 autorestart 1`; App Nap off; long runs under `caffeinate -dimsu`.
- **The camera needs a logged-in desktop session** (else the CGL renderer hangs). Auto-login a dedicated user (FileVault off, or `fdesetup authrestart`), add an HDMI dummy plug, run GPU & render jobs as that user's LaunchAgents; CPU-only services as LaunchDaemons. A ray-cast eye needs neither.
- Data on external case-sensitive APFS drive; paths from env vars; float32 before MPS; checkpoint & restart every N hours; compare with Linux statistically.

**Reproduce first** (first day: twelve measurements, each replacing an estimate above; commands in long form, "The first day"): core layout; `testspeed humanoid.xml` (50–90K steps/s per fast core); Jack's world under integrators 0/1/3; decision loop on 1–12 processes (750–1,300/s today); SAC updates/s on SB3 / SBX / SB3-MPS (70–300 / 150–500 / 50–200), then 1–6 runs at once; 4096² matmul (2.5–5 TFLOPS); GRU cell on CPU, MPS, MLX (15–30 µs per GPU job); Dreamer steps/s on JAX, sheeprl, sheeprl-MPS at 1M then 12M, same script on T4; 2,000 sheeprl steps CPU vs MPS (any divergence bars MPS, PyTorch issue 204359); 64×64 renders logged in, over SSH, from a LaunchAgent and a LaunchDaemon (300–2,000 fps; which contexts work); 24 h soak, 6 worlds + 1 MPS learner, behind a wall meter (55 W, ~$7/month, no throttling or creep).

**The last 10%** (first anywhere; publish logs): MuJoCo per-core speed on an M4-or-later core and the M6's two tiers; SAC & Dreamer timings on any Apple chip; MPS vs CPU vs MLX on a recurrent world model; Apple vs T4 on the same Dreamer script; offscreen render fps on macOS; whether MPS runs in a no-login LaunchDaemon; whether two Dreamer learners (one MPS, one CPU) beat one; multi-day training soak on a Mac mini.

**Not verified:** prices ($1,499 for 32/512 per Abt and AppleInsider; $1,299 for 32/256 per one blog; not Apple's store); M6 GPU's 4.9 TFLOPS (estimate); T4 datasheet figures (not re-read); performance cores' clock and per-clock speed; MLX's 1.3–2.5× gain over MPS (one M5 repository); launchd key names; multi-day thermals.

## 11. Porting Dreamer to MLX, and renting a GPU for the large runs (2026-11-15; serves order J5, the decisive test in PLAN, and the move to the Mac)

Long form: `/data/jack-research/reports/Porting Dreamer to Apple MLX.md` (62-item contract with line links: `research_notes/Porting Dreamer to Apple MLX/dreamer_port_spec.md` §2).

**Already known**
- **Reference:** `danijar/dreamerv3` at `e01491f`; agent unchanged since Dec 2024; ~1,130 of 10,360 lines are algorithm. Port for 74 inputs, 8 continuous actions: ~1,170 lines translated, 430 fresh plumbing, 700 test (~2,300). Faithful PyTorch ports: 2,300–2,800 lines plus tests.
- **Sizes for Jack:** `size1m` 647,705 trainable parameters, `size12m` 9,765,209 (slow critic 66,111 / 853,503). Rule matches r2dreamer's and TorchRL's counts exactly.
- **Corrections:** latents are 32 variables × **4 classes (1M) / 16 (12M)**, not 32 × 32. 12M recurrent width 2,048, not 1,024. Optimiser constant 0.999 (paper 0.99). 1,000-update warm-up; first update uses learning rate 0. Two-hot computed in raw space on exponential bins (paper says otherwise). Actor samples not squashed. Return scale not debiased.
- **MLX 0.32.4** (14 Oct 2026): has grad, compile, stop_gradient, JAX-style keys, sort, gather/scatter. **Lacks** scan, percentile, LaProp. GRU is the PyTorch cell; norm eps default 1e-5 (Dreamer 1e-4). Compile freezes any state not passed in. Gradient fixes landed in 0.32.0–0.32.1. No MLX Dreamer or RL agent exists.
- **Where it runs:** Linux wheels need glibc ≥ 2.35; on our server only in a Debian 12 container (checked). CUDA needs SM ≥ 7.5: T4 is at the floor.
- **Earlier ports:** ~20 bookkeeping vs ~18 maths bugs; bookkeeping lived longest (~8 months). Fidelity came only from same-weights, same-batch loss & gradient comparison (sheeprl PR #377).
- **Rent, $/h (RunPod / Vast median, 11-12 / 11-15):** 3090 0.22–0.50 / 0.26; A5000 0.16–0.27 / 0.23; 4090 0.34–0.89 / 0.54; 5090 0.69–1.39 / 0.87; A100 1.39–2.19 / 0.80. Per-second, prepaid. **(inf)** job costs: Job 1, reproduction at 9.7M, $5–18; Job 2, decisive test, $4–34 at 0.64M, $72–409 at 9.7M; Job 3, month-long lifetime, $115–266 (3090) to $338–511 (4090).

**Take as-is**
- Reference commit `e01491f`. Pin `mlx==0.32.4`; re-run rungs 1–11 on any upgrade. Port config values, never class defaults. Import official weights by name (MLX kernels are transposed; decoder input is [stoch, deter]); never re-seed.
- **Build order** (each step 1–2 slots; none before the rung below it passes): stage 0 spike (container; Kaggle install; compile time and memory at both sizes; golden v0 reproducible) → config → maths → returns/normaliser → optimiser → layers → names → golden v1 → RSSM step → 64-step pass → heads and world-model loss → imagination and actor-critic → gradients → train step → acting step → compiled re-run → replay → driver → ratio loop → checkpoint → smoke. ~60–110 slots **(inf)**. Policy-only slice for owners' devices: ~15–25 slots.
- **Minimal plumbing:** in-memory replay of 65-step samples with stored latents, write-back, online queue; one-world driver; Ratio loop starting at 1,024 items; full-state checkpoint. Omit images, discrete actions, multi-device, logger, prioritised replay.
- **Staging:** official JAX only on paid cards; the port never on the paid path. Order: free proving → 1–2 h test on a 4090 with ≥ 24 cores (≈ $1–2) → Job 1 → Job 2 (only after reproduction, resume on a rented host, watchdog fired once, run sheet frozen) → Job 3. Self-destruct on exit or after 20–30 min stalled; small prepaid balance; delete, never stop. No spend authorised.

**Reproduce first** (fourteen rungs against G, a golden file from the official code: Kaggle CPU, float32, compiler off, noise from file; per-rung detail in long form, "A fourteen-rung ladder"): names, shapes, counts exact → maths 1e-6 (decode exactly 0, also under `mx.compile`) → λ-returns and normaliser 1e-6 → optimiser over 5 steps 1e-6, warm-up exact → initialisers within 2%, zero heads exact → layers and one RSSM step 1e-5 → 64-step pass 1e-4, no flips → world-model losses 1e-5 → imagination and actor-critic losses 1e-4 → gradients: norm 1e-4 (1e-3 for RSSM leaves), cosine ≥ 0.9999 → five optimiser steps, step 1 changes nothing → bookkeeping exact (action alignment, reset flags, queue, write-back, death vs time-out, ratio count, bit-identical resume) → step-0 answers ln 255 = 5.541263, std 0.8927, scale 1 → curves: ≥ 5 seeds × 2–3 tasks, port vs official, intervals overlap (official runs shared with §2's `size1m` test).

**The last 10%** (first anywhere)
- MLX trace & compile time and peak memory for the full unrolled step (64-step pass + 15-step imagination from 1,024 starts, with gradients) at both sizes. Gate G0: proceed if `size12m` compiles in < 10 min and peaks < 8 GB **(threshold is ours)**.
- MLX speed on CUDA (T4, rented cards) and the M6. Projected 6–25 / 1.5–5 steps/s on the Mac **(inf)**.
- Agreement between chip types (CPU, Metal, CUDA) over a 64-step unroll.
- Dreamer of any kind on Apple silicon.
- Real T4 and 4090 Dreamer steps/s; gain from packing several runs on one card.

**Not verified:** Kaggle's glibc and driver for MLX; optax first-step-LR semantics (rung 3 settles it); dependency versions of the published runs (unrecorded); rental speeds (scaled from one 3080 Ti figure); the 1e-4 unroll tolerance; whether owners' devices other than Macs can use MLX.

## 12. His eye, and a sense added later (2026-11-16; serves PLAN's decisive test, stage 3)

Long form: `/data/jack-research/reports/PRIORS section 12.md`.

**Already known**
- Dreamer's eye is 64x64x3 everywhere (V1-V3; Minecraft, Crafter). Vision runs use the 200M-class size, `size1m` only proprio; size ablation 12M-400M, bigger better. **No 1M-from-pixels, T4 or CPU pixel result exists.**
- DreamerV3, DMC task mean (official score files): proprio 456/675/758 at 100k/250k/500k; pixels 381/522/675/787 at 100k/250k/500k/1M: ~2x the steps; small targets lose most (reacher_hard 339 vs 949 at 500k).
- SAC on raw pixels fails (walker 42 vs 948 from state, 500k, CURL T1). Random shift alone repairs it (DrQ 921 vs 964); no auxiliary loss beats it (Li 2022).
- Cost: DrQ-v2 96 fps, DreamerV2 24 fps on a V100: 3 / 12 GPU-h per 1M steps.
- Dreamer with image gradients cut dies (V2 T2: 0.25 -> 0.01, worse on 51 of 55 games). Replacements that hold: R2-Dreamer (2026; Barlow Twins, no augmentation; Walker 4.4 h vs 7.0 h, curves only), MuDreamer (784.7 vs 739.6 on 20 DMC tasks; video backgrounds 517 vs 84), TD-MPC (Cup Catch 933 at 100k). Contrastive is worse (TPC 222 vs 737).
- Depth beats colour for navigation: Habitat PointNav SPL blind 0.42, RGB 0.46, depth 0.79 at 75M steps; RGB caught up at 2.5B (0.92).
- "What if Eye" (2025, MuJoCo, PPO): navigation evolved ~10 eyes of 4 receptors; detection needed two 15x15 eyes.
- Measured here, one ARM core: `mj_multiRay` 16x16 0.10 ms, 32x32 0.34, 64x64 1.27 (distance, geom id, normal; 5-7x our `mj_ray` loop). GL 64x64: 14-16 ms tuned, ~20 ms floor even depth-only. MuJoCo >= 3.5.0 has a GL-free camera rangefinder (one ray per pixel). GPU batch renderers are CUDA-only: none on Mac, T4 untested.
- **Nobody has compared a sense added at step N with the same sense from birth in RL.** Nearest: a pathway degraded early is never used later (Kleinman 2023, up to 20% lost) unless a cross-sense reconstruction loss exists; adding a modality lowered accuracy (Kinetics 72.6 -> 71.4); OpenAI Five added observations with zero outgoing weights, "without loss", never compared with from-birth; resets restore plasticity (SAC IQM 501 -> 656).

**Candidates**
- **A. Ray retina, 16x16 to 32x32, [depth, material] in one `mj_multiRay` call.** 3,000-10,000 frames/s per core on every machine, no GL, no Mac login session. Support: depth-over-RGB, the 15x15 detection eye. Weak: no colour or shading; not a human eye. Expect: usable by both brains at today's budgets (inf).
- **B. 64x64 RGB camera, random shift (B), decoder or Barlow loss (A).** The published default. 10-20 ms/frame on CPU; ~2x the steps at 200M size, unknown at 1M.
- **C. Both: rays from birth, camera added later** (the decisive test's added sense).

**Ours beside theirs**

| setting | published | ours | cost |
|---|---|---|---|
| eye | 64x64 RGB, or depth | 16 rays, 1 row (`w0.py:96`) | no elevation |
| ray call | one batched call | `mj_ray` loop (`w0.py:260`) | 5-7x slower |
| pixel size | Dreamer >= 12M | `size1m` | below record |
| SAC pixels | shift + CNN | none | must add |
| new input | zero outgoing weights | not built | old skills at risk |

**Known-answer check** (GPU only; nothing pixel-sized fits a CPU kernel): DrQ walker_walk 921 at 500k; TD-MPC Cup Catch 933 +- 24 at 100k; R2-Dreamer at 0.63x Dreamer's wall-clock.

**What the record does NOT answer**
- Added later vs from birth: arms from-birth / zero-init at N / random-init / constant placeholder / reset at N / placebo noise sense; measure use (a task only that sense solves; ablate it), cost to old skills at N, catch-up. ~60 kernel-h (6 arms, 5 seeds, 200k decisions; inf) with the ray retina. Predicted: Dreamer catches up (it reconstructs the sense); SAC under-uses it, more so at late N.
- Does 16x16 depth+material beat 16 rays at telling food from wall? 5 seeds, CPU.
- `size1m` from 64x64 on T4: speed, and whether it learns.

**Not verified:** R2-Dreamer and TD-MPC2 pixel scores (curves); DreamerV3's size table; Mac render speed (unpublished); Achille's loss size (figures); Safety Gym / Animal-AI rays vs pixels (none found).

## 13. The lie-still trap: movement cost, and the reward that got agents eating (2026-11-16; serves J8; answers the INBOX `RESEARCH:` line of 11-16 16:04)

Long form: `/data/jack-research/research_notes/j8_lie_still/` (three parts).

**Already known**
- **No published survival world charges the body for moving.** Drain is equal moving or still in the two-resource ant (0.00015/step, `two_resource_env.py` L386-388), EVAAA, Crafter, ForageWorld, Neural MMO, Animal-AI; Avalon's cost is 1e-8 vs a 1e-4 frame cost, and 0 at evaluation. A motionless body there lives as long as a random one (Crafter, run here, 40 seeds: no-op 172 steps, random 187). **Ours: 600 s still vs 302 s random. No from-scratch learner is on record eating reliably under a 2–3× movement drain.**
- **The trap is well recorded in locomotion with an energy price.** Kwiatkowski 2023 (PPO; basal 2.23, movement 1.26·v²): "the global optimum for a typical discount factor around 0.99 is v=0"; pure energy reward failed even from a goal-seeking start. Hwangbo 2019, Solo12 2023, Yu 2018: standing still is the local optimum under a motion penalty. Hollenstein 2022 (SAC/TD3, MountainCarContinuous, +100 goal, force cost): agent settles on "not performing any action", return 0.
- **Cures with evidence:** cost curriculum (Hwangbo: factor from 0.3, k ← k^0.997; Solo12: 0 → 1; Kwiatkowski: worked in 4 of 5 scenarios); γ = 1 or longer (Kwiatkowski); reward toward the goal (same); noise that holds its direction (Hollenstein: OU σ 0.5 solves, Gaussian σ 0.6 does not); more steps (Fu 2021: stands still to ~5×10^7 steps, then walks; ForageWorld: stationary ~1.3×10^9 steps with NO movement cost).
- **Cures with negative evidence:** alive bonus (Reda 2020: bonus 5 "balances but never steps forward"); white-noise floor alone (Hollenstein).
- **Rewards that worked:** two-resource ant, 100·(D_t − D_{t+1}), nothing on eating, PPO γ 0.99, 150M steps (best of five forms, abstract only); Crafter, health change/10, no hunger charge (eat cow 12–26% of episodes at 1M); Avalon, food − a near-zero frame cost. EVAAA's default: −0.01·Σdev² per step, −100 at death; **its +100-on-eating branch, which `dr100-eat` copies, is OFF by default.** Pure +1-per-food lost Yoshida's five-way and failed on rllab Ant Gather (Duan 2016).

**Our arithmetic (inf; `drives.py:66-68, 117-118, 192`, `jack_env.py:282, 606-613`)**
- Drive is (1 − e)², so a life under `dr100-eat` returns ≈ −100·(1 − d_birth) + ~100–120 per meal however lived (pilot: +16 at 1.15 meals, −78 at 0.20: matches). **Lying still does not shrink the bill, it delays it past the 40 s horizon**: 0.067·(1 − e) per decision at rest, 0.133·(1 − e) random. Death step costs ~0 in all three rewards.
- In ENERGY, random foraging does not pay at d 0.75: a 0.31 meal buys 186 s of movement; random finds one per ~238 s (needs 1.3× random skill; at d 1, 0.08 buys 48 s: 5×). **Stillness is the right answer to the body's own economy until he forages better than random**; only the +100 eat pay makes moving worth it in reward (+0.084 per decision vs 0.067·(1 − e) extra).

**Candidates** (none proven in a survival world with a movement drain)
- **A. Movement-cost dial tied to d** (KAPPA × d; a world change). Only cure proven on the same failure, 3 papers, 10^7–10^8 PPO steps. Weak: a badly scheduled ramp falls back to stillness (1 of 5 in Kwiatkowski).
- **B. `dr100-eat-pos`** (exists): no pay for stillness, death ~0, so intake-only: Crafter's and Avalon's shape, but the form that lost Yoshida's five-way. Expect: stops the slide to stillness; eating then rests on exploration alone.
- **C. Longer horizon**, γ 0.999 (200 s) or 3-step returns: γ = 1 held in Kwiatkowski; no n-step SAC number found.
- Not candidates: alive bonus; plain `homeo-dr` (the ant's reward; with our drain it pays stillness most).

**Ours beside theirs**

| setting | published | ours | cost |
|---|---|---|---|
| moving / resting drain | 1.0 everywhere | 2 random, 3 full (`drives.py:84`) | stillness doubles life |
| pay on eating | none (ant); +1 (Crafter) | +100 (`jack_env.py:613`) | EVAAA's off branch |
| step cost | none, or drive rise | drive rise ×100 | pays stillness |
| horizon | 100–333 steps; γ 1 for energy | 200 (`pilot_b.py:39`) | bill hidden past 40 s |
| steps | 1M–150M | 300k (`pilot_b`) | 3–500× short |

**Known-answer check:** Hollenstein's MountainCarContinuous is §5's row 3, which B passed (5/5 > 90). Closer: two-resource ant with drain × (1 + action cost): no published number.

**What the record does NOT answer** (5 seeds, 300k; 4 runs per Kaggle CPU kernel)
- KAPPA 0 vs KAPPA × d vs today, `dr100-eat`: predicted meals/life late ≥ random's 1.15 only in the first two.
- `dr100-eat-pos` vs `dr100-eat` at today's KAPPA: predicted no slide to stillness (unfed lives stay ~300 s), meals near random.
- γ 0.999 vs 0.995: predicted smaller effect than either above.

**Not verified:** every paper row came through a fetch summariser (quotes second-hand); only `trp_env`, EVAAA, Crafter and Avalon code read directly. Yoshida's five-way numbers (abstract only). `trp_env`'s death charge: one reading found −10, another none. EVAAA's steps and published decay rates. n-step SAC, average-reward learners and pay-on-intake-only never tested against this trap.

## 14. Smell: what learned to follow an odour, with which nose, in how many steps (2026-11-17; serves stage 3 "Other senses", seat `SM.02`; INBOX `RESEARCH:` 11-16 06:44)

Long form: `/data/jack-research/research_notes/smell/`. [V] = read in paper or code.

**Already known**
- **OdorSpaces** (arXiv:2611.05738, CoRL 2026; [V, in full]): PPO, depth eye + nose + pose history into a recurrent core, start distance ramped 1.5 → 9 m over 30M steps, one seed. SPL %: learned 65.3 propane / 24.7 fire; best hand-written searcher 36.1 / 0. **Nose: 5 points on 0.5 m arms × 4 heights, raw samples over 1–2.5 s.** Centre point only: 65 → 12.9. One height: fire 0. No curriculum: 53.6. Reward pays each step for true distance closed; without it 51.6 / 0. Smell alone 49.5; no recurrence 54.0 / 28.7 (fire better). Given GPS + compass; starts inside the plume; fails by pushing into obstacles the odour crosses; **no no-smell control, no curve, no result under 30M, no code.** Nothing pretrained named.
- **Our field form, the closest test** (Alonso & Kirkegaard 2024, arXiv:2310.10531 [V]; C0·exp(−λ|x|), 5 sensors, log(M+1), PPO, "millions of simulations", distance-shaped): a memoryless policy "performs poorly" when sensor baseline is small vs noise; a recurrent one reading change over time works there.
- **Intermittent plumes: every published learner had memory AND a wind reference.** Singh 2023 (arXiv:2109.12434 [V]): 1 odour point + egocentric wind, PPO, RNN-64, 5M steps, "RNNs outperform MLPs for every plume tracking task", top 5 of 14 seeds shown. Rando 2024 (arXiv:2404.17495 [V]): tabular Q, wind-frame actions, 100k episodes: memory of 1 sniff 79% success at 0.14 path efficiency, 20 sniffs 95% at 0.5. Verano 2023: a reactive controller fails "in a large fraction of cases". Piro 2026 (arXiv:2605.21329): wind direction is "key to many proposed algorithms". Windless successes (Loisy 2022–23) carry a full Bayesian map.
- **Learned smooth-gradient taxis on a body:** FlyGym's RL demo [V, code]: SAC [32,32], 500k decisions, 1-number turn action over a ready-made walker, reward = distance closed + heading; odour as 2 square-rooted numbers. Its hand-written taxis needs (L−R)/mean × 500 through tanh: "the asymmetry is subtle".
- **Biology:** two-nostril smell is a near-field cue. Moles: nostril block matters "within 4–5 cm", not farther (Catania 2013). Mice at distance: "naris occlusion did not significantly impact performance"; they compare sniff to sniff (Findley 2021 [V]). Best human concentration resolution ~5% (Cain 1977).
- **No published body learned to climb a static two-point gradient from eat-only reward**, at any budget. Every success: ≥ 500k decisions, pay for distance closed, a 1–4-way action (ours: a whole body).

**Ours (inf, from `odour.py:99-131`):** left–right contrast ≤ 8% (0.16 m / 2 m), ~5% over headings; best-heading signal/noise 7.6 / 4.6 / 2.8 at 4 / 5 / 6 m; cue is 0.004–0.011 between two raw inputs of 0.05–0.14. Raw, memoryless, 13k–52k decisions, eat-only: the weak end of every published axis. The sense suffices: a scripted nose-follower fed .375 = the perfect-map ceiling (`505f7ae8`); `steer` on rising log smell 14/40 vs babble 3/40.

**Candidates** (for `SM.02`; none seats anything)
- **A. Wide multi-point nose, raw history** (OdorSpaces). Proven at 30M, shaped, one seed. Strength: direction in one sniff. Weak: a 1 m nose is not a human head (END-FIT).
- **B. Today's nose re-encoded + memory:** log per nostril, (L−R)/(L+R) with gain, d(log mean)/dt, into a recurrent brain (Alonso; FlyGym; biology). Smallest published: 500k decisions, 1-number action. Expect: nothing at 52k; first sign ≥ 500k.
- **C. Head sweep (casting):** the animal's far-field answer. Evolved, not RL (Izquierdo 2010, 1 sensor); our klinotaxis script fed .21. Needs memory.
- Not candidates: memoryless learner on the raw 12 numbers (three kills here); O2 puffs with no wind sense.

**Ours beside theirs**

| setting | published | ours (file:line) | cost |
|---|---|---|---|
| nose | 5 pts × 4 heights, 1.0 m span; or 4–5 on-body | 2 pts, 0.16 m (`odour.py:99`) | centre-only lost 52 SPL |
| encoding | log, sqrt, normalised contrast × gain | raw (`odour.py:455`) | cue is 5% of the input |
| memory | RNN / 10-sample history | one d/dt (`odour.py:453`); reactive learner | the weak arm in 3 papers |
| start distance | 1.5 m, ramped to 9 | 4–6 m from birth | no ramp: −12 SPL |
| reward | distance closed each step | meals only | stop-only: 0 on fire |
| wind | given (plumes); none (CFD room) | no wind sense; O2 gusts 0.6 m/s (`odour.py:123`) | O2: no precedent |
| steps | 500k – 30M | 13k, 52k (`smell_sense.py:40-51`) | 10–2,300× short |

**Known-answer check:** none fits. Closest: Alonso's exponential-field cell: code and step count not found. OTTO (`C0PEP0D/otto`, 1-D: mean arrival ≈ 7.15, "several hours" on a laptop) checks a pipeline, not a nose. Singh: 16 h/seed. Cheap check, ours: scripted follower on the new encoding must still reach the perfect-map ceiling (minutes, box).

**What the record does NOT answer** (stage 3; a brain with memory, start-distance dial, ≥ 500k decisions, 5 seeds, shuffled-nose control; ~1 Kaggle CPU kernel per arm)
- Encoding B vs raw, same nose: predicted B out-eats shuffled, raw does not.
- Does smell add anything once the `row128` eye sees floor food (HIT .909)? Predicted no in the open, yes for hidden food (`SM.02`'s claim).
- Nose span 0.16 vs 0.5 m: predicted to matter less with memory (Alonso).
- O2 puffs with vs without a wind sense: predicted no learning without.

**Not verified:** OdorSpaces' total steps (30M is the ramp), net sizes, PPO settings, figures. Singh's success counts (plot). Verano's table; Alonso's step count (SI); FlyGym RL's score; EVAAA's smell ablation (secondary: "least useful sense"). Biology except Findley: abstracts. Not found: any RL test of one nose point vs two. Not searched: auxiliary losses for an ignored input.

## 15. The home-made brains on the two textbook rows: what each kind of learner is on record to need (2026-11-17; serves J3 (4), M3)

Long form: `/data/jack-research/research_notes/inhouse_rows/` (three parts, `mc_noise.py`).

**Question.** J3 (4): each in-house arm on Pendulum-v1 (a 100-episode window ≥ −314.7 inside 60,000 steps, 3× SAC's) and sparse MountainCarContinuous-v0 (> 90 inside 150,000). Three failed both; four wait. Is a miss "broken" or "this class is that slow"?

**What ours are (probed 11-17).** ONE policy learner under seven names: advantage-weighted regression (AWR) on GAE targets (`cores.py:540`), 256-row ring, batch 16, 0.25 updates per decision, γ 0.95 (`survival.py:63`), i.i.d. Gaussian noise std 0.5 → 0.1 (`inhouse_rows.py:194`). They differ only in an auxiliary loss; none plans. Measured:
- **`wm-efe` IS `dreamer-xs` on these rows.** Same seed: 115 shared tensors identical, only actor/critic initial values differ; the 5 heads get no gradient and nothing calls `uncertainty` (`cores.py:408`).
- `dreamer-xs`, `wm-efe`, `wm-latent` step the GRU once from zero state and zero action (`cores.py:362`; recurrent and action weights get gradient 0.0, as `ef335cf` found): an autoencoder, no model of what comes next. `wm-latent`'s target is its own detached latent (`cores.py:395`), no EMA copy. Only `wm-dyn` predicts the next step.
- Weights: `ppo-needs` 122K, `wm-latent` 857K, `dreamer-xs` 1.53M: `wm-latent`'s ~3× faster row 1 is confounded with a 7× larger network.

**Already known**
- **Tuned on-policy learners miss row 1's rule too.** RL-Zoo's PPO on Pendulum-v1 (gSDE, γ 0.9, budget 100k): 20k-step training windows −1185, −754, **−321**, −214, −220; first rolling-100 ≥ −314.7 at ~60,700 steps (`wm-latent`'s best windows −321…−397 sit there: PPO-class speed). TRPO: 80,600. A2C (budget 1M): 370,800. SAC/TQC: inside 20k. One seed each, public logs (`DLR-RM/rl-trained-agents`; PPO re-derived here). The Zoo budgets on-policy 5–50× off-policy; ours 3×.
- AWR is published as slow online: "not yet as sample efficient as SAC and TD3" (Peng 2019, arXiv:1910.00177; 50k buffer, batch 256, β 0.05; 5k already unstable; no Pendulum result); "an order of magnitude slower" (AWAC, arXiv:2006.09359). Ours: 256 rows, batch 16, β 1.0.
- **White noise cannot find the flag.** Measured here (200 episodes each): i.i.d. Gaussian std 0.1 / 0.3 / 0.5 → **0 / 0 / 0** reaches; std 1.0 → 5. The same std 0.5 held 8 / 32 steps → 79 / 170; OU σ 0.5 → 176.
- Published, same shape: PPO with white noise scores 0.0 on this task after 2,048,000 steps (20 seeds); colored noise β ≥ 0.5 ~93.9 (Hollenstein 2024, arXiv:2312.11091, Table A14.1). Under DDPG/TD3/SAC Gaussian σ 0.6 fails, OU σ 0.5 solves (arXiv:2206.03787).
- γ 0.95 is a second blocker (inf): the best run takes ~106 steps; 100 × 0.95^100 = 0.59 vs ~2 of action cost, so doing nothing is optimal (36.6 at γ 0.99).
- Disagreement solves sparse Mountain Car in the first episode only when it PLANS the actions (Tschantz 2020, arXiv:2002.12636; MAX, arXiv:1810.12162). No result found where a model or ensemble loss that chooses no actions reaches a sparse reward; dropping an active-inference agent's epistemic term made "little to no difference" on CartPole (Millidge 2020).
- A next-state loss does help model-free learners on state inputs: TD7 +276.7% over TD3 at 300k (MuJoCo, arXiv:2306.02451); decoder vs EMA-latent target differ 2–5%, an un-stopped target loses 23–25%; OFENet lifts SAC (Humanoid 6093 → 9561), barely PPO. All NEXT-step; ours (bar `wm-dyn`) is same-step.
- Memory buys nothing on a fully observed task: DRQN worse on 6 of 9 such Atari games (Hausknecht 2015); a shared recurrent encoder hinders (Ni 2022).

**Candidates** (repairs; rule unchanged, ruling 24)
1. **Held noise at action choice** (hold 32 steps, or OU θ 0.15 σ 0.5), with γ ≥ 0.99. Proven: PPO 0 → ~94 here at 2M steps.
2. **A class-fair reading of row 1.** PPO-class needs ~60–100k steps, A2C-class ≥ 370k. A rising curve that misses at 60k is "on-policy speed", not "broken".
3. **A next-step latent loss, stopped target** (`wm-dyn`): proven on MuJoCo state with a 1M replay, not on 256 rows.

**Ours beside theirs**

| Setting | Published | Ours | Cost |
|---|---|---|---|
| Row 1 budget | PPO 100k, A2C 1M, SAC 20k | 60k for all (`inhouse_rows.py:114`) | tuned PPO misses by 6 points |
| Exploration | gSDE / OU / colored | i.i.d. Gaussian 0.5 → 0.1 | 0 flag reaches, any arm |
| γ on row 3 | 0.9999 (Zoo), 0.99 (PPO, colored) | 0.95 | do-nothing is optimal |
| Model loss | next step, 16 × 64 sequences | same step, single rows | an autoencoder |
| Epistemic term | plans the actions | untrained, uncalled | none: `wm-efe` = `dreamer-xs` |

**Predictions for the four waiting arms.** Row 3: 0 flag reaches, 0/5, every arm. Row 1: `dreamer-xs` and `wm-efe` best windows −320…−420, 0–1 of 5 seeds, indistinguishable; `wm-dyn`, `wm-latent+ring` no better than `wm-latent`.

**Known-answer check.** `mc_noise.py`: an exploration option reproduces white 0/200, held-32 ~170/200 before use. Row 1's PPO-class reference: Zoo PPO, 100k steps, windows 3 and 4 ≈ −321, −214.

**What the record does NOT answer**
- Is `wm-latent`'s 3× its loss or its size? `wm-latent` at loss weight 0 (5 seeds, ~1.7 CPU-h); predicted: most of the gain stays.
- AWR on 256 rows with held noise and γ 0.99 on row 3 (5 seeds, ~2 CPU-h); predicted: flag found at once, row still missed (256 rows forget the reach).
- Next-step vs same-step loss on 256 rows: `wm-dyn` vs `wm-latent` reads it first.

**Not verified.** R2D2's burn-in figures; Eberhard 2023's pink-noise table; VIME's and AWAC's figures; OFENet's step budgets; TD7's per-task digits; the TRPO and A2C steps (one seed, a sub-agent's arithmetic). Not found: AWR or Dreamer on state Pendulum; Dreamer on Mountain Car; a discount or replay-size ablation on Pendulum; an RSSM trained on single rows.

## 16. The walking layer on free CPUs: what the 2018 recipe really showed, and the cheapest walk to reproduce first (2026-11-18; serves order J10, M4; plan part "Body and movement")

Long form: `/data/jack-research/research_notes/walking_layer/` (`part_a_yu2018.md` = paper source and code at `b50478f8`; `part_b_cpu_walks.md`; `part_c_transfer.md`).

**Question.** J10 (2) reproduces Yu, Turk & Liu 2018 ([1801.08093](https://arxiv.org/abs/1801.08093)) over chained kernels, then (3) moves the recipe to our Humanoid-v5 body. What did the paper show, is there a cheaper published walk for ONE free CPU kernel, and what happens on a new body?

**Already known**
- Yu 2018 confirmed verbatim: 29 DoF, 76.6 kg, 1.5 m/s; PPO1, 3×64, mirror weight 4; 20k samples an iteration, 500–1,500 iterations, 8 threads, 4–15 h. Humanoid walk, read off its one curve: helper + mirror ends at ~780 iterations (~15.6M samples, return ~1,250); helper alone ~910; mirror alone and plain PPO plateau at ~500 ("fall slowly or stand still").
- **It gives no seed count, no success rate and no definition of "walks".** One curve per method; Table 2 covers "successfully trained policies" only. "Half the data" is the simple biped.
- **The released curriculum is looser than the paper**: a stage that hits its 5M cap **advances anyway, unflagged** (`pposgd_mirror.py:211`, `training_utils.py:81-85`) — all 22 stages can "complete" having learned nothing. `(2000, 2000)` are spring GAINS on sideways position and forward speed, not newtons. The helper is OFF by default (`assist_timeout` 0.0; the script sets 100). An unpublished rule restores the weights when return drops > 500. Energy is 0.3·Σ|a|; lateral damping 0.002·kp (paper: 0.1·kp).
- **Nobody has published a reproduction, or the recipe on any MuJoCo body**: 7 forks with no commits, 5 issues, all install or crash (one a DART solver crash on the humanoid).
- Mirror loss buys symmetry, not speed: "no consistent and predictable impact" on learning (Abdolhosseini 2019, 4 bodies); "detrimental" on a humanoid without the curriculum; "slower convergence" than mirrored data (Mittal 2024).
- **On Gymnasium Humanoid a return near 5,000 is STANDING**: 5 a step × 1,000 steps (`humanoid_v5.py:196-216`); speed ≈ (R − 5,000 + control cost ≤ 272)/1,250 m/s. ARS says so verbatim and sets its bar at 6,000. §4's "SAC ~5,200–5,500 at 1M" is 0.16–0.62 m/s: it does not prove a walk.
- Humanoid-v4 at 1M, 10 seeds (SimBaV2's table): PPO 477; SAC 4,825 [3,784–5,866]; TD3 5,165 [5,020–5,310]; TQC 6,029; CrossQ 10,480 (~4.4 m/s). At 2M on v3 (SB3 Zoo, ONE model, 150 episodes): SAC 6,232 ± 280, TD3 5,567, TQC 7,239 ± 1,647, A2C 388. Humanoid-v5: no comparable published number.
- Withdrawn aids elsewhere: 200 N pull −20 N per success (HoST 2025, standing up; without it fails); torque limits ×0.95 per stage (Tao 2022; weak from the start "fails"). A strong effort cost from the start "causes a performance collapse" (Schumacher 2023).
- A learning low level under a learning high level: DeepLoco froze it (joint training "impairs" the high level; a swapped low level dropped return to 0.06–0.19); HIRO found a frozen one "detrimental" on hard tasks, and its fix is for replay learners only.

**Candidates** (for J10's known-answer walk)
1. **SAC, the Zoo's Humanoid entry, Humanoid-v4, 2M steps** (256×256, one update a step; SBX SAC is brain B's own library). Strength: runs today in `jack-std`; a 10-seed band at 1M. Weakness: the 2M figure is one model; the gait is unshaped. Expect ≥ 6,000 and ≥ 12 m forward per 1,000 steps on most seeds. **Measured here: 0.046 CPU-s a step, so 2M ≈ 26 CPU-h — one seed per 12 h kernel only if it uses ≥ 2.2 cores.**
2. **Yu 2018 as released** (J10 (2) today): the only no-mocap, CPU-sized, natural-gait recipe. Weakness: unreproduced, 2018 stack, ≥ 3 chained kernels a seed, silent cap. Expect ~780 iterations if it trains.
3. **ARS, linear policy** (Mania 2018): CPU-native, 100-seed timing, bar 6,000; median 12–17 CPU-h, heavy tail; on Humanoid-v1.
- Ruled out on CPU: CrossQ (measured here 0.31 CPU-s a step → 85 CPU-h per 1M), REDQ/DroQ (20 updates a step), PPO at ≤ 1M (floor).

**Ours beside theirs**

| Setting | Published | Ours | Cost |
|---|---|---|---|
| Body | Yu: 29 DoF, 23 actions, 76.6 kg, DART | Humanoid-v5, 17 actions, MuJoCo (`drives.py:170`) | untested transfer; helper gains unscaled |
| Control rate | Yu 33 Hz; Gym 67 Hz | 67 Hz under a 5 Hz brain | §4: needs the layer |
| Reward | Yu: speed 3, alive 6, energy 0.3, posture | Gym default pays 5 a step to stand | must be replaced for (3) |
| Success test | none published | `yu2018_walk.py`: 9/10 full rollouts ≥ 0.75 m/s | ours is stricter; keep |
| "Curriculum done" | cap advances silently | `yu2018_walk.py` verdict DID-NOT-FINISH reads stage reached | **must also log which stages exited by cap** |
| Zoo SAC env | v3/v4 | floor measured on v5 (`humanoid_floor.py`) | run the row on v4, then v5 |

**Known-answer check.** SAC row: Humanoid-v4, Zoo settings, return at 1M inside [3,784, 5,866] and at 2M ≥ 5,950 (6,232 − 280) WITH ≥ 12 m forward; control = learning starts after the last step, must stay < 500. ~26 CPU-h a seed. Yu row: as registered, plus a per-stage log of exit reason.

**What the record does NOT answer**
- Does Yu's recipe train at all outside its authors' hands? (J10 (2); ≥ 3 kernels a seed; prediction: build risk > learning risk.)
- Helper + mirror on the Gym Humanoid body, with Yu's reward: no published run. One kernel a seed with SAC or PPO; prediction: helper needed for a low-energy gait, mirror changes symmetry only.
- Does Humanoid-v5 score like v4 (contact cost restored)? The SAC row on both, 2 kernels.
- A torque-level walker that keeps learning under a learned 5 Hz brain: not published.

**Not verified.** The curve figures (iterations read by eye); the Zoo table (read through a summariser); Heess 2017's and the PPO paper's humanoid step counts; the original SAC paper's Humanoid curves; Kaggle-kernel speed (box figure only, loaded 2-core ARM); ports searched once.

## 17. Copying the hand forager into his network, and a learner handed the direction to food (2026-11-18; serves order J15, rows E2b and E2c; plan part "Learning to eat")

**Question.** J15's `prior:` pointed here; no such text existed. (a) A scripted teacher WITH memory copied by plain regression into a net WITHOUT memory, no corrections: what happens? (b) What does a learner handed bearing and distance to its goal need, and prove? Long form: `/data/jack-research/research_notes/j15_clone/`. Read in LaTeX source unless under Not verified.

**Already known.**
- Regression on teacher-only states drifts (error ~T²) and more data does not cure it: Super Tux Kart supervised "does not improve as more data is collected", corrections reach 0 falls/lap in ~15k points; Mario sticks where the expert never stood ([1011.0686](https://arxiv.org/abs/1011.0686)).
- A teacher that is Markov on the student's own input clones easily: 25 trajectories → 77–95% of expert ([1606.03476](https://arxiv.org/abs/1606.03476) T3). At 1M samples (ours): HalfCheetah-expert 105.2 ± 1.7, Hopper 111.5 ± 1.3, Walker2d 56.0 ± 24.9 ([2106.06860](https://arxiv.org/abs/2106.06860) T1).
- **Imitation gap** ([2007.12173](https://arxiv.org/abs/2007.12173) T1): a student blind to what the teacher knows learns the AVERAGE of its choices. Best of BC/DAgger 0.88 with a matched view; 0.00–0.11 where the teacher remembers a switch.
- Squared error on a teacher that picks at random: MLP+MSE 0.67 on the first reach, **0 on the three later stages** of scripted block-push; 0.98 on unimodal CARLA ([2206.11251](https://arxiv.org/abs/2206.11251)).
- Privileged teacher → sensor student (CARLA): offline 16–20% success; teacher labelling the student's OWN states 64%; plus every-branch labels 100 ([1912.12294](https://arxiv.org/abs/1912.12294)). Walking: teacher 76.2, student with history 73.5, without 52.1 ([2107.04034](https://arxiv.org/abs/2107.04034)).
- History in an OFFLINE clone often hurts (it copies its last action): Ant single frame 1,300, history RNN −311; Humanoid 587 / 367 ([2010.14876](https://arxiv.org/abs/2010.14876)).
- Handed direction, navigation: a blind agent with GPS+compass reaches SPL 0.42 / success 0.62 at 75M steps ([1904.01201](https://arxiv.org/abs/1904.01201)), 0.973 success at 2.5B ([1911.00357](https://arxiv.org/abs/1911.00357)); direction removed, success 0.948 → 0.099 ([2009.03231](https://arxiv.org/abs/2009.03231)). The oracle swap is the field's perception-vs-control test ([2206.00997](https://arxiv.org/abs/2206.00997)).
- Handed goal, small budgets, DENSE reward: state SAC reacher-easy 746 ± 25 ([2004.04136](https://arxiv.org/abs/2004.04136)) or 919 ± 123 ([2004.13649](https://arxiv.org/abs/2004.13649)) at 100k. Repeated reaching: SAC PointGoal1 27.0 ± 1.5 at 1M (OmniSafe table, shaped).
- Gathering WITHOUT a direction: Swimmer-Gather 0.0 for all algorithms at 25M ([1604.06778](https://arxiv.org/abs/1604.06778)); Ant-Gather ~3 apples at 10M ([1805.08296](https://arxiv.org/abs/1805.08296)).

**Ours, measured** (seeds 701–712, 269,524 decisions). A target is held on **0.551** of decisions (per seed 0.445–0.698), not the state line's 0.993 (smoke seed 153). Only actions 6–7 vary. With no target the teacher steers along ONE of ~38 open rays, drawn at random, held ≤ 25 decisions (`eye_forager.py:293-299`): 45% of the lesson is a coin the row cannot see. A 90-s copy of the student on unseen worlds: R² 0.29 / −0.11 with a target, ≤ 0 without; its push 0.38 / 0.24 against the teacher's 0.63.

**Candidates.**
1. *Plain regression, as frozen.* Proven for realizable teachers, ≤ 1,000 steps. At ours (inf): approaches food it sees; with none in view, a weak push toward the mean open direction. Bar of 5 meals: unknown, leaning miss.
2. *The teacher labels the student's own lives* (DAgger). 16 → 64% in CARLA; our teacher labels for free. Cures drift and stuck states, NOT averaging over a random heading. A short history only with this; offline it is the copycat (−311).
3. *A teacher the row determines* — wander along THE most open ray, no draw. Makes the lesson realizable (0.88-class). It is a new ruler: it must re-pass gate 2 first.

**Ours beside theirs.**

| Setting | Published success | Ours (file:line) | Cost |
|---|---|---|---|
| Teacher vs student view | same input, Markov | timers, bans, target, random heading (`eye_forager.py:221-299`) | 0.88 → ≤ 0.11 |
| Loss on random choices | unimodal 0.98 | MSE on a 38-way draw (`forager_clone.py:182`) | averaging; inertia |
| Horizon | ≤ 1,000 | 9,000 (`forager_clone.py:99`) | drift, no recovery data |
| Hold-out | — | random 10% of the same lives; last action predicts next at R² 0.86 | rosier than unseen worlds |
| E2c reward | dense or shaped | +100 on eating, drain (J11) | sparse: Fetch-class DDPG "often not able to learn" ([1802.09464](https://arxiv.org/abs/1802.09464)) |
| E2c bearing | lidar, or cos/sin | radians / π — jumps at ±π (rule E2c) | Safety Gym warns of the wrap-around |
| E2c control | input removed and retrained | random fixed point | a fixed point leaks odometry; add zeroed |

**Known-answer check.** Clone: `train_clone` on D4RL hopper-expert should score 111.5 ± 1.3 (not Walker2d); minutes at our 4.4 CPU-s per 1e5 samples (inf). E2c: B's SAC rows already reproduce; DMC reacher-easy at 100k has no single published number (746 vs 919).

**What the record does NOT answer.** (i) Whether a memoryless clone of THIS teacher clears 5 meals — the frozen kernel, ~9 h; prediction: miss, or a pass carried by the 55% of decisions with a target; read the clone's push on no-target frames first. (ii) Survival by repeated eating with the direction handed, at 300k: nobody published it. Prediction (inf): first reliable reach 50k–200k; a miss at 300k is weak evidence.

**Not verified.** Ross & Bagnell 2010; Kart/Mario and DART baselines (figures); Walsman 2023; Ortega 2021 (abstract only); PointGoal1 curve at 300k; D4RL BC's network and normalisation; whether food textures change per world.

## 18. The other senses, all together: touch, balance, taste, pain, temperature, body-state, and many senses in one small brain (2026-11-19; serves PLAN part "Other senses, all together" (step 0), M5; stage 3)

Long form: `/data/jack-research/research_notes/senses_together/` (parts A, B, C). Nothing pretrained is inside any agent below.

**Already known**
- **Nobody has fused more than 3 distinct senses in one from-scratch RL learner with per-sense ablations.** The only Dreamer result with > 4 keys: Minecraft (image + inventory, health, hunger, breath; 200M weights, 100M steps), no per-key ablation (arXiv 2301.04104).
- Dreamer (official `rssm.py:215-230`, `agent.py:80-83`; sheeprl `loss.py:74`): vector keys concatenated into one MLP, one decoder head per key, each loss at scale 1.0 SUMMED over its numbers: a sense's pull is its width; no per-key knob.
- More senses can lose: Habitat SPL depth 0.79, RGB+depth 0.70 (75M, 1904.01201); Kinetics RGB 72.6, +audio 71.4 (1905.12681); two-sense MNIST single 99.14, joint 45.26, rebalanced 91.01 (2202.05306).
- Cures: per-sense loss and weight (CoRAL 2302.05342, 1M steps; M3L 2311.00924, touch weight 10, 95% masking, ~3M); whole-sense dropout in training (TORCS: loss under noise 51.7% → 9.8%, 1705.10422); product-of-experts (MuMMI 2107.02339: walker-run 466.7 vs concat 71.3).
- Noise inputs: SAC 256-256 cannot learn Humanoid at 90% noise features (clean ~4,500, 1M) (2302.06548).
- **Body-state hidden = no learning**: EVAAA (NeurIPS 2025, Unity, DreamerV3-S; four needs): the needs-hidden and eye-hidden arms fail to learn; heat sense and collision mattered more than smell. Figures only. DreamerV3 > DQN > PPO. Its heat sense: 9 numbers over a heat field.
- Reward: drive difference is best of three (Yoshida 2024, abstract; doi 10.1016/j.neunet.2024.106379). Their ant senses each need as ONE number (set-point 0, death at ±1) and cools by a panting action, not by seeking shade.
- Many needs (Dulberg, arXiv 2204.06608 e-print READ 11-19; RLDM 2022 short form of PNAS 2023): 9×9 walkable grid in a wall ring (11×11, Gaussians at the corners), 3×3 view of 4 resource maps + 4 stats, DQN γ 0.5, ONE 30k-step life. Δ (mean Σ|h*−h| over steps 15k–30k, figure readings, n = 50): monolithic DQN 3.2 [1.9–6.0] with 1 annealing step → 0.95 [0.75–1.35] at 5k → 0.8 at 10k; one DQN per need (GmQ, Q summed) ≈ 0.5–0.6 at every annealing length. No needs-hidden arm. The '4 → 8 needs worsens' claim is NOT in this version (PNAS text not read). Rule: `experiments/needs_grid_row.py`.
- Touch: ShadowHand, DDPG+HER, 5 seeds: 92 touch sensors reach the level 2.6× sooner (Egg 355 → 115 epochs of 190k samples), final level +16%; **Boolean 2.45× ≈ continuous**; touch WITHOUT proprio 0.24 of baseline. Null control (92 zeros / noise) moved one task 0.30 → 0.34: < 15% at 5 seeds is noise (Melnik 2021, doi 10.3389/frobt.2021.538773).
- Balance: quadruped, joint positions + gravity 3-vector + base linear and angular velocity give 94–99% of full-state; the rest alone 6–12% (2306.17101). Biped G1, PPO, pushes: with IMU 96.1 ± 0.5% survive, without 90.4 ± 1.1 (2608.29487).
- MIMo infant (2312.04318; head gyro + accelerometer, full skin; PPO): reach 0.5M, stand 1.5M, catch 4M steps; no sense ablated.
- Pain as an INPUT: one 2×2 exists (arm, CACLA): −31% error, p = 0.152 — not significant (doi 10.3389/fnbot.2017.00010). Predicting own future health is what survives poison: Doom with poison vials, 50M steps: DQN 25.4, A3C 59.3, DFP 84.1 (1611.01779).
- Taste: no agent has a contact chemical sense; poison is told by sight. A 30–60 s gap from cause to payoff defeats LSTM agents at γ 0.998; memory recall solves it (1810.06721). No RL agent shows one-trial aversion over minutes.

**Candidates**
- **A. Concat + per-sense prediction, each sense's loss normalised by its width, plus whole-sense dropout.** The record's default. Proven with 2–3 senses, 1M–100M steps. Weak: wide senses dominate. Expect: works for 5 vector senses; touch and audio add less than seed noise at 300k (inf).
- **B. Masked cross-sense prediction** (ours, `cores.py:216-256`; M3L's form). Proven only with attention at ~3M steps.
- **C. One value head per need** (Dulberg), once there are four needs. Grid only.
- **New senses, per the record:** touch = Boolean per region, ≤ 16 regions; balance = gravity 3-vector + gyro 3 in the head frame; pain and taste = no input shown to help — predict the integrity and energy change instead (DFP); temperature = 1 body number + ≤ 9 ambient numbers.

**Ours beside theirs**

| setting | published | ours | cost |
|---|---|---|---|
| senses fused | 2–3, one an image | 5 vectors, 74 numbers (`jack_env.py:336,799`) | beyond the record |
| brain A keys | one head per sense | one flat key `state` (`brain_a/rows.py:76`) | no per-sense loss, no hide-it |
| loss weight | by width (sum) | `needs` ×2 (`cores.py:82`); row128 eye vs needs ≈ 40:1 in A (inf) | needs under-weighted |
| touch | Boolean, 16–92 | 4 × [force, flag] (`w0.py:286`) | coarse for hands |
| balance | gravity 3 + gyro 3 | upright cosine, height (`w0.py:310-313`) | no tilt direction, no turn rate |
| needs | 1 number each, set-point 0 | levels + d(h) + 2 rates (`drives.py:432`) | no precedent, untested |

**Known-answer check** (GPU, not CPU — measured 11-19: ~55 min per 30k-step DQN life on one ARM core): Dulberg's grid, 4 needs, DQN, 30k steps × 30 agents: separate learners near maximum, single needs thousands of exploration steps (curve); add a needs-hidden arm (EVAAA predicts failure). < 2 h (guess).

**What the record does NOT answer**
- Leave-one-sense-out on the brain that eats: 6 arms × 5 seeds × 300k (~30 kernel-h, inf). Predicted: needs-hidden and eye-hidden fail; touch-, audio-hidden inside noise (< 15%).
- A test-time hide / shuffle / matched-noise test of a sense: no published protocol (all above are training-time); ours is new.
- Needs as one deviation each vs our six numbers: 2 arms × 5 seeds. No prediction.
- Gravity + gyro vs the upright cosine, human-like body, pushes. Predicted: ~6 points of survival.
- A taste label at the mouth; aversion after one delayed sickness; seeking warmth at ≤ 1M: nobody.

**Not verified.** Yoshida 2024 full text; EVAAA's steps and effect sizes (figures); Dulberg, the robot ant (summariser); MIMo, M3L, CoRAL scores (curves); MuMMI's and sensor dropout's step counts; Seymour 2019, Safety Gym, Crafter health: not read.

## 19. A learner that ends below its untrained twin: entropy against reward size, the survival traps, and what published foragers were set to (2026-11-19; serves J11 (6), J15 (4) E2c, M3)

**Question.** In J11's partial (20/40) every learning arm of brain B eats LESS than its learning-rate-0 twin (late meals/life 0.08–0.16 vs 0.82). B is SAC with a FIXED entropy coefficient 0.1 on 8 action dims, rewards un-normalised (`pilot_b.py:294-299`). What causes this on record, and what were published foragers set to? Picked because J11's prior ("arm K eats inside 300k") is contradicted and E2c runs the same setting. Long form: `/data/jack-research/research_notes/sec19/`.

**Already known** (read in the papers' TeX source)
- SAC v1, Haarnoja 2018 ([1801.01290](https://arxiv.org/abs/1801.01290) §5.2, Fig 3b, Ant, 2×256): at small reward size "the policy becomes nearly uniform, and consequently fails to exploit the reward signal". Ant's reward scale 5 = α 0.2 (Table 2).
- SAC v2 ([1812.05905](https://arxiv.org/abs/1812.05905)): the cure is a LEARNED α, entropy target −dim(A).
- Yu, Zhang, Xu 2022 ([2201.12434](https://arxiv.org/abs/2201.12434)), 5-node chain, −0.05 a step, fixed α 0.2 → entropy reward 0.14 a step (2.8× the task): the agent "never wants to reach the terminal state", V keeps rising. **The entropy reward inside the critic's target is a pay-for-being-alive.** Taking entropy out of the target (SACLite) cures it; a smaller start α "will not eradicate" it.
- RL-Zoo `sac.yml` (fetched 11-19): 2 fixed entries — MountainCarContinuous 0.1 (ONE action dim, ≤ 0.069 a step: the row B reproduced), BipedalWalkerHardcore 0.005. All others `auto`.
- Unity ML-Agents ([1809.02627](https://arxiv.org/abs/1809.02627), table): FoodCollector SAC 46.4 ± 7.9, PPO 36.6 ± 8.4 (5 runs) — the ONE SAC forager on record. Config (`release_10`): 49 ray values, discrete moves, ±1 a food, α LEARNED from 0.05 (no fixed mode exists), γ 0.99, 1 update per 10 steps, 128×2, **2,000,000 steps**, several agents. Same table, sparse search: Pyramids SAC −1.0 vs PPO 1.79.
- rllab Gather, Duan 2016 ([1604.06778](https://arxiv.org/abs/1604.06778), Table 1; 25M steps, mean over training): Swimmer-Gather 0.0 for every algorithm; Ant-Gather random −5.8, TRPO −0.4, DDPG −0.3, one column −6.7 (below random). Positive food return needed hierarchy: HIRO 3.02 ± 1.49, VIME 1.42 at 10M ([1805.08296](https://arxiv.org/abs/1805.08296)).
- Crafter 1M ([2109.06780](https://arxiv.org/abs/2109.06780)): every rewarded learner beats random on eating (Eat Cow 0.4% → 12–26%); only reward-free learners fall below it (RND 0.7 vs 2.1): **learning without a usable reward destroys what a random walk gets free.**
- Primacy bias, Nikishin 2022 ([2205.07802](https://arxiv.org/abs/2205.07802)): SAC over-fitted to its first data "is unable to recover" in ~1M steps; resets every 2e5 steps, buffer kept: DMC IQM 501 → 656.

**Computed here (inf).** A tanh-squashed Gaussian at mean 0, std 1 has entropy 0.670 nats a dim (ceiling ln 2 = 0.693); at std 2, 0.00; at std 0.37, 0.30. So (i) the untrained twin ALREADY collects 0.536 a decision; (ii) entropy pressure holds std near 1 and PUNISHES saturation — `PILOT_B2.md` §3b's "std/saturation rising = confirmed" reads the wrong way; (iii) pure entropy dominance predicts learner ≈ twin, not below it; (iv) 0.536/(1−γ) ≈ 107, about one meal, is paid in every living state and lost at death: lives 276 → 500 s ≈ +600 of entropy reward, over 5 meals.

**Candidates** (causes; a trace separates them)
1. *Entropy as a survival bonus* (Yu 2022). Signature: V rises toward ~100 with no meals; lives lengthen, meals and movement fall; std stays 0.4–1. Fits B0/P/G. Cure a: `ent_coef="auto"`, target −8 — B's row 1b reproduced with `auto` (−302.0 vs lib −309.3); fixed 0.1 has a known answer on a 1-dim task only. Cure b: SACLite (unproven here). Cure c: fixed α ~0.005 (inf).
2. *Lost coverage*: a learned mean saturates tanh or shrinks the noise; he sweeps less floor. Signature: |action| > 0.95 rises or std falls, lives unchanged, meals fall where moving is free. Fits arm K.
3. *Primacy bias*: decline from the warm-up's end, never reversing. Test: a reset.
4. *The budget*: the one SAC forager took 2M steps with an easier body; flat learners ≈ 0 at 25M on an 8-dim body. Near-random at 300k is what this record expects.

**Ours beside theirs**

| Setting | Published forager / SAC | Ours | Same? cost |
|---|---|---|---|
| temperature | learned, target −dim(A) (SAC v2, ML-Agents, Zoo) | fixed 0.1 (`pilot_b.py:297`) | DIFFERS; no multi-dim precedent |
| entropy vs task reward a step | fails at 2.8× (Yu) | 0.536 vs −0.036…+0.031: ~15× | DIFFERS |
| reward | ±1 a food, 0 between | +100 a meal, small negative drift, death lump (`jack_env.py:50-53`) | DIFFERS; alive bonus ≈ a meal |
| steps, envs | 2M, several agents | 300k, 1 | 1/7 the data |
| updates a step; actions | 0.1; 3 discrete branches | 1; 8 continuous | 10× the fitting on less data; harder search |

**Known-answer check.** Yu's SimpleChain (5 nodes, −0.05 a step, terminal at the end): fixed α 0.2 must FAIL with V rising; `auto`/SACLite must reach the end. Seconds on a CPU; step count unread (figure). FoodCollector needs the Unity player: not a free-CPU row.

**What the record does NOT answer**
- Learner BELOW its lr-0 twin in a free-movement forager: no published curve. E2c's trace reads it free: std ≈ 1, mean ≈ 0, V rising → cause 1; saturation rising or std falling → cause 2.
- Fixed 0.1 vs `auto` in his world: arm K + one cost arm, 5 seeds, 300k (~2–3 kernels). Predicted: `auto` drops α below 0.01 inside 50k, the lengthening of meal-less lives shrinks; meals ≥ twin NOT predicted.
- When a learner passes a held-noise random walk on foraging: nobody.

**Not verified.** FoodCollector's learning curve, area count, random score; Duan's column names; EVAAA's levels (summariser); Yoshida 2024; pink noise; Safety Gym; auto-α collapse on sparse continuous tasks (discrete only: TES-SAC [2112.02852](https://arxiv.org/abs/2112.02852)); SBX's `auto` start value.


## 20. A living Jack: every need at once, a life of hours with no reset, stopped and resumed (2026-11-20; serves M8)

Long form: `/data/jack-research/research_notes/m8_living/` (parts A–C). One hour of his life = 18,000 decisions.

**Question.** Several needs at once, hours of life with no reset, stopped, moved and resumed: who did it from scratch, and what should the multi-hour test measure?

**Already known**
- **Nobody has shown it:** no paper found has a small network still improving at the end of one multi-hour life. Nearest: Jelly Bean World (PPO-LSTM, DQN; one life, 10M steps, 20 runs) learns for ~1M steps, then every agent collects nothing ([2002.06306](https://arxiv.org/abs/2002.06306) Fig. 4).
- **With several needs, the moving food is the one neglected.** Crafter at 1M: drink 24–80% of episodes, eat cow 12–26% (human 89%), eat plant 0.0–0.1% for every agent; "wake up" 93.6% for the RANDOM agent (no learned sleep) ([2109.06780](https://arxiv.org/abs/2109.06780) App. A–C). Craftax at 1B: eat-plant "ignored", lives ~200 steps ([2402.16801](https://arxiv.org/abs/2402.16801)).
- **Four needs in 3-D:** EVAAA (~100k steps a level): PPO and DQN survive only the open field; DreamerV3-S learns, then drops at harder levels; needs hidden = no learning; agents that find no food learn to END the life by colliding.
- **One critic per need beat one critic for the sum** on an 8×8 grid, 4–8 needs, 30k steps: right value map at ~5k vs ~20k steps; the single critic worsens from 4 to 8 needs, the other does not (Dulberg 2023 PNAS Fig. 2D, 3C).
- **Mixed reward sizes starve the small one:** 57 Atari games, unclipped: 0.3% median → 107% with per-task return normalisation (PopArt [1809.04474](https://arxiv.org/abs/1809.04474) Table 1).
- **Long learning decays by itself:** PPO, 2×256, a STATIONARY Ant rises to ~3M steps then collapses, "failing every episode" by 20M; L2 1e-4 or continual backprop held (Dohare 2024, Nature Fig. 3–4). L2 "interfered" in Rainbow; LayerNorm: no reported downside (Lyle [2402.18762](https://arxiv.org/abs/2402.18762)).
- **No reset, a place to get stuck:** SAC reset-free vs episodic 0.00 vs 1.00 (door, peg), 0.32 vs 0.80 (EARL [2112.09605](https://arxiv.org/abs/2112.09605)).
- **A cap is not a death:** bootstrapping at time-outs let PPO trained on 300-step pieces hop 10^6 steps ([1712.00378](https://arxiv.org/abs/1712.00378) Fig. 6).
- **Resume:** SB3's `save()` omits the replay buffer and the observation normaliser; sheeprl saves the buffer by PATH (`memmap_buffer: reference`, same machine only; installed 0.8.5). The only measured dip is offline→online: ~1/3 of return lost (Lee [2107.00591](https://arxiv.org/abs/2107.00591) Fig. 1b), back within 200k steps with or without the old data (Luo [2303.17396](https://arxiv.org/abs/2303.17396)); 50% old + 50% new per batch with a LayerNorm critic avoids it (RLPD [2302.02948](https://arxiv.org/abs/2302.02948)).
- **Hours of real-time learning, on robots:** Dreamer walks an A1 in 1 h (~72k steps), re-learns after pushes in 10 min, SAC only rolls over; a sparse-reward arm: nothing before 2 h of 8 ([2206.14176](https://arxiv.org/abs/2206.14176)). Nobody measured a learner handed to another machine.

**Candidates**
1. **One summed drive reward + return normalisation** (ours; DreamerV3's percentile scaling, PopArt). Proven at 4 needs (EVAAA, DreamerV3-S); slows as needs grow. Ours: 2 needs plausible, 4+ unknown.
2. **One value head per need, summed to act** (Dulberg). Proven only on that grid (DQN, no death, endless food; an attention mask when the world changes). No continuous-action version found.
3. **Staying able to learn for hours:** LayerNorm critic + old/new replay mix (§6) + bootstrap at caps; L2 or continual backprop proven for PPO only. The 3–20M collapse lies past today's 1M runs, inside M8's.

**Ours beside theirs**

| setting | published | ours | same / differs |
|---|---|---|---|
| reward over needs | summed convex drive, 4 needs | `homeo-dr` over `DRIVE_DIM` 6 (`drives.py:129`, `jack_env.py:874`) | same form; >4 needs only on a grid |
| cap vs death | bootstrap at cap (Pardo) | `truncated` ≠ died (`jack_env.py:863-881`) | same |
| life length | cap 10k; learned lives ~200 steps | starves in 3,000 decisions (`drives.py:66`) | 1 h exceeds any reported learned survival |
| B's checkpoint | buffer saved separately | model + buffer (`humanoid_sac_row.py:187-197`); LayerNorm (`pilot_b.py:301`) | same; not yet in his world |
| A's checkpoint | buffer by path, same node | "no buffer checkpoint" (`brain_a/in_world.py:30`) | differs: moved to the Mac, A resumes with an EMPTY buffer unless `buffer.checkpoint=True`, `memmap_buffer=copy` |
| network | 4M–201M weights | 2×256 | no record at our size |

**Known-answer check**
- Dulberg's grid (4 needs, 30k steps): single critic right at ~20k, per-need at ~5k; CPU minutes (inf); code not located.
- Pardo's Hopper pair: hops on only with bootstrap; ~1 CPU-h a run (inf).

**What the record does NOT answer**
- **Stop and resume, his own critic:** B on row 1b, stopped at half, resumed whole / empty buffer / fresh Adam; 5 seeds, box minutes. Predicted (inf): none / a dip ≤ 1/3, gone by the row's end / none.
- **Does a long life collapse:** only once something eats — continue it to 3M decisions, 5 seeds, ~30 kernel-h a run. Predicted: no collapse by 3M with LayerNorm + replay; unknown beyond.
- **The second need:** summed critic vs per-need heads when water joins food (predicted: per-need faster); all needs vs one need; sum vs "weakest need" reward — no published head-to-head for any.
- **The test, to copy:** geometric mean over needs (Crafter); a mid-life push with a recovery bound (DayDreamer: 10 min); a stop-resume twin with a dip bound; late-life rate ≥ early (Jelly Bean's failure).

**Not verified.** Dohare's Nature numbers (summariser; curves only); Keramati–Gutkin, Yoshida 2024, Dulberg's SI not read; PopArt's frame counts; dip sizes in AWAC and Cal-QL (plots); whether SB3 or sheeprl save the recurrent state or the life in flight; any return change from moving MuJoCo builds.

## 21. Win-then-lose: a SAC learner that loses a skill it had learned, and an eval that changes with the CPU (2026-11-21; serves J10's repair, the forge's eating wall, M3, M4; answers INBOX `RESEARCH:` JACK 01:20)

**Question.** Two learners lose a skill they had. J10 s1 (SBX SAC, Zoo Humanoid-v4 entry, no LayerNorm): R 5,621, 0/10 falls at 1.5M → 581, 10/10 at 2M; α .009 → .024; Q(s0) 514 → 547 while the return fell. E2c (SBX SAC **with** `layer_norm=True`, `pilot_b.py:301`): 10/10 food runs win whole lives, then lose them (last capped life 81–189k of 300k), and the critic overestimates in every run, controls too. What does the record say about late loss, its causes, and a published "keep it" recipe? And is a deterministic 10-episode eval that moves 5,834 → 5,552 on the same weights across CPUs normal?

**Already known** (abstracts read verbatim through the arXiv API; 2403.00514, 2110.11222, 2005.04269 read in their TeX source)
- **No source found reports Zoo SAC on Gym Humanoid collapsing between 1M and 2M.** The 2M figure is ONE model (§16); the 1M band [3,784–5,866] is a 10-seed spread, not a per-seed curve. So "1 of 3 seeds collapses late" is neither confirmed nor ruled out by the record.
- **Plasticity / primacy:** Nikishin 2022 ([2205.07802](https://arxiv.org/abs/2205.07802)): agents "rely on early interactions and ignore useful evidence encountered later"; fix = periodic resets, buffer kept (DMC IQM 501 → 656, §19). Its shape is a plateau, not a fall from a best point (§19 cause 3).
- **The regularised-SAC study** (Nauman 2024, [2403.00514](https://arxiv.org/abs/2403.00514); >60 SAC variants, 14 tasks incl. humanoid-walk/run, dog; replay ratio 2 and 16, DMC): "layer norm … consistently outperforms others in mean return" on DMC; "Periodical network resetting is the most robust intervention … in a high replay ratio regime"; "layer or spectral norm and resets are particularly effective in mitigating overestimation also compared to methods specifically designed for that purpose"; "a simple Soft Actor-Critic agent, appropriately regularized, reliably finds a better-performing policy". Their regime is RR ≥ 2 on DMC; ours is RR 1 on Gym.
- **Value instability:** off-policy + bootstrapping + function approximation can make "value estimates becoming unbounded" (van Hasselt 2018, [1812.02648](https://arxiv.org/abs/1812.02648)). TQC (Kuznetsov 2020, [2005.04269](https://arxiv.org/abs/2005.04269)): truncating the top quantiles "seems to reduce the seed variance on Humanoid"; coarse overestimation control (more critics) did NOT beat published MuJoCo results. Zoo TQC Humanoid 2M: 7,239 ± 1,647, one model (§16).
- **Dead units:** dormant neurons grow over training (Sokar 2023, [2302.12902](https://arxiv.org/abs/2302.12902), ReDo); in Nauman's study ReDo is below full resets. Plasticity loss "often occurs in the absence of saturated units" (Lyle 2023, [2303.01486](https://arxiv.org/abs/2303.01486)).
- **Architecture fixes on humanoids:** SimBa ([2410.09754](https://arxiv.org/abs/2410.09754): running obs norm + residual block + LayerNorm, into SAC) matches SOTA on DMC, MyoSuite, HumanoidBench; BRO ([2405.16158](https://arxiv.org/abs/2405.16158): bigger LN critic + optimism) "first model-free algorithm to achieve near-optimal policies in the notoriously challenging Dog and Humanoid tasks". Means only.
- **Variance from outlier runs** (Bjorck 2022, [2110.11222](https://arxiv.org/abs/2110.11222)): pixels, DDPG-type actor-critic; variance "mostly arises early", from saturating activations; penultimate-feature normalisation fixes it, and plain layer norm "can make performance worse" there. It is about EARLY failure; it does not cover a late fall.

**Reading our two signatures against the record**
1. *Critic runs away* — fits both: Q rises while the earned return falls (s1; E2c's generic overestimate). Record's best-supported cure: LayerNorm critic (+ resets). **But E2c already HAS SBX's `layer_norm=True` and still loses 10/10** → LN alone is not the cure in his world; J10's row has none (Zoo recipe).
2. *Primacy / lost plasticity* — fits E2c weakly (it learned, so it was plastic early), s1 poorly (best at 1.5M, then a fall). Test: resets every 2e5, buffer kept.
3. *α rising as the fall begins* (s1 .009 → .024): SAC's auto α rises when entropy falls under its target — a sharpening policy. No published source found tying a rising α to a coming collapse (not verified).

**Recipes on record to keep a learned skill** (published, not home-made; for the forge pool / J10 repair, each a race arm): (a) LN critic — J10 only, E2c already has it; (b) full resets every 2×10^5 steps, buffer kept (Nikishin; Nauman's most robust at RR 16); (c) TQC instead of SAC (Zoo entry exists, SBX has it — check); (d) SimBa encoder. A kept-best checkpoint is NOT a learning fix and breaks "never stops learning".

**The eval across CPUs.** Not answered by any source read here. MuJoCo bit-reproducibility across CPU builds is not promised (not verified — MuJoCo docs not read). rliable (Agarwal 2021, [2108.13264](https://arxiv.org/abs/2108.13264)) argues for interval estimates over point scores; a 10-episode deterministic eval with 1–3 early falls is a coin-flip-sized sample. Consequence for re-registration: more eval episodes (≥ 50) and the verdict read on ONE CPU type, stated.

**What the record does NOT answer** (cheapest first)
- Is s1's fall the recipe or chaos? Continue s1 from its 1.5M `model.zip` + replay with a different RNG seed, twice, to 2M (2 CPU kernels, ~5 h each). Prediction: if the recipe, ≥ 1 of 2 falls again; if chaos, 0.
- Does Zoo SAC on Humanoid-v4 lose its walk late in other seeds? Needs per-seed curves (SimBaV2's released logs, if any) — a read, not a run.
- Win-then-lose in his world with LN already on: resets (b) vs TQC (c) vs plain, E2c's E-food arm, 5 seeds, 300k — the forge's round, frozen by the Director.

**Not verified.** SBX's `layer_norm` placement (critic only or both) and whether SBX ships TQC; MuJoCo's cross-CPU determinism; whether any Zoo per-seed Humanoid curve dips.

## 22. Five keep-it knobs with no prior: 3-step returns, temperature cap and target, critic weight decay, DroQ / CrossQ on a CPU, a narrow net at a high update ratio (2026-11-21; serves the forge's F2 pool, M3; answers INBOX `RESEARCH:` FORGE 02:45)

**Question.** The forge's pool names five settings of brain B (`pilot_b.py:296-301`: SBX SAC, 2×256, LayerNorm critic, 1 update a decision, γ 0.995, +100 a meal, 300k, CPU) with no prior. What was each shown to do, at what scale, and what should it do to a learner that eats, then loses it? Long form: `/data/jack-research/research_notes/sec22/`.

**Already known**
- **3-step returns help value learners with sparse reward and a big replay; on SAC the record is one negative, no number.** Rainbow ([1710.02298](https://arxiv.org/abs/1710.02298), Atari): n=3 best of 1/3/5. D4PG ([1804.08617](https://arxiv.org/abs/1804.08617), DMC): "N=5 is uniformly better", N=1 unstable on a sparse task. Fedus 2020 ([2007.06700](https://arxiv.org/abs/2007.06700)): "uniquely beneficial" with a larger replay, "may hurt … in regimes of high off-policyness". BRO ([2405.16158](https://arxiv.org/abs/2405.16158)): "does not improve the performance" (dense rewards); SimBaV2 uses 1.
- **Nobody has published a temperature cap.** 16 implementations read (SBX, SB3, rlkit, CleanRL, jaxrl, …): none clips α. The one cap in print: Meta-SAC's α ≤ 1 ([2007.01932](https://arxiv.org/abs/2007.01932), Humanoid 10M), above our 0.13–0.65.
- **Every published entropy target other than −|A| is HIGHER:** −|A|/2 (jaxrl, BRO, SimBa; unablated); REDQ's Ant (8 dims, as ours) −4. No sweep found.
- **Our α climb is described in print:** "τ will grow to infinity if the target entropy is infeasible" ([2406.05953](https://arxiv.org/abs/2406.05953)). E2c's fixed-α policy settles at −10…−20 nats; auto holds −8 by raising α. Auto is not reliably better: Hopper fixed 0.2 3,484 ± 324, auto 2,572 ± 901 ([2506.05615](https://arxiv.org/abs/2506.05615)).
- **Critic weight decay alone is weak.** Nauman 2024 ([2403.00514](https://arxiv.org/abs/2403.00514), 2×256 SAC, DMC, replay ratio 2): "generally low performance when used alone", "high synergy with full-parameter resets"; overestimation ≈ no regulariser's (1e3 vs 2e3; LayerNorm 15, spectral norm 17, by eye). Li 2023 ([2304.10466](https://arxiv.org/abs/2304.10466)): 0.01 good on DMC, "poor" on Gym. SimBa 1e-2; SimBaV2 removed it.
- **DroQ** ([2110.02034](https://arxiv.org/abs/2110.02034)): 2×256, dropout 0.01 + LayerNorm, 20 updates a step, cost ≈ a 20-update SAC.
- **CrossQ** ([1902.05605](https://arxiv.org/abs/1902.05605)): 1 update a step, no target nets, BatchRenorm (warm-up 1e5 steps), critic 2048, lr 1e-3, policy delay 3. Critic 256 "reduces performance and yet still significantly outperforms SAC"; plain BatchNorm is "prone to sudden performance collapses". Follow-up ([2502.07523](https://arxiv.org/abs/2502.07523)): its weight norms grow and units die; target nets came back.
- **Narrow net, many updates: no published 2×64 (or ≤ 2×128) result at ≥ 4 updates a step found.** The nearest points the other way (BRO): "a 26M model with RR=1 achieves significantly better performance than a small model with RR=15".

**Candidates** (predictions; knobs read in the installed SBX 0.28)
1. **`n_steps=3`** — a knob. SBX's target is r₁..₃ + γ³(min Q − α log π): uncorrected, entropy only at the bootstrap state (`sbx/sac/sac.py:295`). Fits us on paper: sparse +100, whole-run replay, overestimating critic (Meng 2020, [2006.12692](https://arxiv.org/abs/2006.12692): multi-step "significantly less affected"). At OUR budget: an earlier first capped life; keeping it is unanswered.
2. **A LOWER entropy target with auto α** (`target_entropy=-16`) — a knob. A target the sharp policy can meet, so α need not climb. Mechanism published; the value is ours — no paper ran a target below −|A|. Expect α under 0.1; the loss may remain (fixed 0.1 lost it 5/5 with no climb).
3. **CrossQ small** (`sbx.CrossQ`, critic [256,256]) — a knob, ≈ 1× our cost (inf). SBX's default [1024,1024] ≈ 10×, the paper's 2048 ≈ 40× (inf). Its 100k BatchRenorm warm-up is a third of our run — the collapse-prone plain-BatchNorm phase.
4. **Weight decay 1e-2 (AdamW) ONLY paired with K1's resets**; alone the record predicts no change. SBX's `optimizer_class` sets actor and critic alike; critic-only needs code.
- **Not supported:** a temperature cap (unpublished: `inventor`); DroQ (12–14× update cost (inf) for dropout 0.01 on top of our LayerNorm); 2×64 at a high update ratio (no result for, one against).

**Ours beside theirs**

| Setting | Published | Ours (`pilot_b.py:296-301`) | Costs |
|---|---|---|---|
| n | 3 (Rainbow, DrQ-v2), 5 (D4PG) | 1 | credit for a meal moves 1 step an update |
| entropy target | −dim(A) or higher (−4 on 8 dims) | fixed α 0.1; auto −8 | skilled policy sits at −10…−20: target infeasible |
| critic | LayerNorm; + resets; AdamW | LayerNorm | the top DMC entry, still lost 10/10 |
| updates a step · reward | 1–20 · dense | 1 · +100 sparse | high-ratio results and BRO's n-step negative do not transfer |

**Known-answer check.** n-step: none exists for SAC (D4PG: curves only). Temperature: SAC + gSDE, auto α, PyBullet 1M: Ant 3,459 ± 52, Hopper 2,646 ± 45 ([2005.05719](https://arxiv.org/abs/2005.05719)). CrossQ / DroQ: curves only. CPU wall-clock: NOT measured for any.

**What the record does NOT answer**
- Seconds per 1,000 updates of CrossQ [256,256] / [1024,1024] on one Kaggle core — minutes; before any CrossQ candidate.
- n=3 on E-food, 5 seeds, 300k. Prediction: first capped life before 18–47k; late lives unchanged.
- Target −16, same arm. Prediction: α stays < 0.1; lost anyway in ≥ 3/5.
- Weight decay + resets vs resets alone — only if K1 survives.

**Not verified.** Nauman's per-intervention values (figures, by eye); Meng 2020 (abstract only); all DroQ / CrossQ scores (figures); SimBa's sign on |A|/2 (code only); openrlbenchmark's SBX report, Ota 2021, D'Oro 2023 on width (not opened).

## 23. One parent's voice: a from-scratch ear trained on one speaker (or TTS only), tested on strangers (2026-11-21; serves M5, PLAN "Ear" right column, the lab's `ear_one_voice.py` rule `8cc61917`; answers INBOX `RESEARCH:` LAB 11:53)

Long form: `/data/jack-research/research_notes/one_voice_part{A,B,C,D}.md` (58 KB).

**Question.** Is "one voice (or TTS only), from scratch, 100+ hearings a word → unseen human speakers" answered anywhere? **No.** No from-scratch study goes down to one training speaker; Speech Commands has no speaker-count ablation. The neighbours agree: one voice does not carry, and more hearings of it do not fix that.

**Already known**
- **One human voice, a lot of it:** one speaker × 2,400 sentences, HMM from scratch → word error on three strangers 8.5 / 62.1 / 55.3% (mean 41.9; the two high rows are the other sex) vs 4.5% speaker-independent; a small normaliser fitted on 40 of the stranger's sentences → 6.8% (Huang 1991, ACL H91-1054 Table 8).
- **Voices vs hearings:** 12 speakers × 600 utterances 7.5–9.0% vs 109 × 40: 6.5–7.1% (ACL H90-1060 Table 1).
- **Synthetic voice count** (PRETRAINED encoder: shape only): 1 voice 62.4 ± 7.6%, 2 → 75.7, 5 → 86.3, 10 → 92.3, 22 → 94.2; real speakers 2 → 92.6, 5 → 96.9, 77 → 99.1 (Lugosch 2019, arXiv:1910.09463 Figs 3–4).
- **TTS only, from scratch, real test:** 92 voices × 1 clip a word 56.7% vs 88.7% for as many real clips (Lin 2020, arXiv:2002.01322 Table 1, 385k weights). A new cloned voice for EVERY clip, ~3k a word: 89.3% (73k weights) vs 98.4% real; 92.1–92.6% after a pretrained-recogniser filter (arXiv:2409.12745 Table 1). **Correction to §7:** Lin's 92.6% is the pretrained-embedding head; the from-scratch 92.57% is this paper.
- **A single voice can hurt:** adding 960 h in ONE high-quality TTS voice to 960 h of real speech moved WER 4.77 → 13.55 (Rosenberg 2019, arXiv:1909.11699).
- **Few voices heard a lot lose to many heard once** (50k weights, from scratch, synthetic only, vs real-only): 8 voices × ~6,250 clips 75.6% worse; 90k voices × 1 clip 28.9% worse (Werchniak 2021, Tables 5–6; different TTS engines).
- **A little real speech goes far:** real positives at 1% sampling weight cut the TTS-only miss rate 18.11 → 6.83%, 5% → 3.51, all-real 1.81 (arXiv:2408.10463 Table 3).
- **Augmentation as stand-in voices:** VTLP on speaker-rich data buys ~0.6 points (TIMIT PER 21.7 → 21.1, Jaitly & Hinton 2013); +0.2–0.4 on Speech Commands (arXiv:2501.03523). Under a real population mismatch (92 adults → children): speed 8% of the gap, directed VTLP 41%, voice conversion 61% (Shahnawazuddin 2020). Pitch and tempo shift: nothing or worse (Lin 2020; arXiv:2503.08954).
- **The oracle:** 7.5-month-olds carry a word across same-sex talkers, not across sex; 10.5-month-olds do (Houston & Jusczyk 2000). One talker with near-identical tokens fails at 14 months; 18 talkers succeed (Rost & McMurray 2009) — and so does ONE talker varied in pitch level, contour and duration (Galle 2015). Mother → father works at 7.5 months after fluent passages (van Heugten & Johnson 2012). Real input: ~66% from the mother, ~5 talkers a word (Bergelson 2019).

**Candidates**
1. **A handful of voices (5–10), heard often.** Every curve is steepest from 1 to 5 voices. Needs both sexes. Smallest published: 5 real speakers 96.9% (pretrained encoder). Expect most of the one-to-many gap gone by 5–10 voices.
2. **One parent whose voice varies** (the oracle's lever: pitch level, contour, rate, loud/soft, fluent context) **+ the published warp.** A TTS parent can do this. Weak alone: random VTLP closes ~14–41% of a population gap; expect a few points.
3. **Learn the stranger on arrival** (END-FIT: still learning on their machines). 40 sentences took a one-voice recogniser from 41.9 to 6.8% error. The cheapest route on record; no small-net keyword version is published.
4. **Many cloned voices from a TTS**, one per clip: 89% from scratch with no real speech.

**Ours beside theirs**

| setting | published | ours | cost |
|---|---|---|---|
| warp axis | linear frequency, via the mel filter centres | finished mel axis (`ear_one_voice.py:130`) | the variant Jaitly & Hinton tried and call ineffective (no number) |
| warp size | N(1, 0.1) clipped to [0.9, 1.1]; top of band pinned | log-uniform 0.8–1.25 (`:84`) = 0.64–1.67× in Hz, uneven (computed here) | far outside any vocal tract |
| test-time | average over 5–21 warps doubles the gain | none | half the gain left |
| hearings a word | 92–3,000 | 5 (`:35`) | no published neighbour below 31 |
| parent's sex vs test | the error is across sex | not split in the reading | hides where the loss is |

**Known-answer check.** None exists for one voice. Nearest: Speech Commands 35 words, from scratch, 125 clips a word → 90.3%, 1,000 → 94.8% (Lin 2020; 385k weights, ours 70k); one CPU session.

**What the record does NOT answer** (worth our compute)
- **Voices × hearings, from scratch.** AudioMNIST (60 speakers × 50 recordings of each digit; from memory, not on disk) or VCTK with word cuts: voices 1, 2, 5, 10, 30 × hearings 5, 15, 50, strangers split by sex. CPU minutes a cell. Predicted: one voice at 50 hearings 40–60% on strangers and far lower across sex; 5 voices ≥ 75% (an unreviewed 5-voice digit run reads 74.1%); voices beat hearings at every equal clip count.
- **The published warp on one voice**, with test-time averaging, beside ours. Predicted: +2 to +5 points, never the 5-voice line.
- **One varied voice vs one flat voice** (TTS parent, pitch/rate/contour spread). Predicted: half the gap to 5 voices. Untested in any machine.
- **Adapting to a stranger from k hearings** (k = 1, 5, 20 a word) without forgetting the parent. Predicted: 20 hearings beat every augmentation.

**Not verified:** Houston & Jusczyk's n (abstract only); Jaitly & Hinton's speaker count; Cui 2015 numbers; AudioMNIST's shape (memory); the FSDD 74.1% is a GitHub README; Lugosch's values come from figure data files and rest on a pretrained encoder; every "predicted" is inference.

## 24. A critic that runs away when one big block of inputs is the eye: input share, critic normalisation, and the fixes on record (2026-11-22; serves J17's cause hunt, M3; plan part "Learning to eat" + "Eye")

**Question.** J17 (`WORLD_CARD.md` § J17 + addenda): brain B (SBX 0.28.0 SAC, 2×256, `layer_norm=True`, VecNormalize clip 10, 1 update a decision) — critic loss climbs 2–13× mid→end on 10/10 sighted runs, flat on 10/10 blind/lr0; sighted Q ends ×1.2–2.2 above what its eating supports. `j17_obs_norm_probe` (s923): no clipped outliers, but the eye is 256 of 298 inputs and **91% of the normalised input energy** (rays16: 41 of 74, 56%; blind 0). Does the record say a dominant high-dimensional input block makes a SAC critic overestimate or diverge, and which published fixes exist?

**Read this slot** (abstracts verbatim from `arxiv.org/abs/<id>` meta tags; the arXiv API and Semantic Scholar both answered 429; no full text read)
- **SAC from pixels fails without help.** DrQ ([2004.13649](https://arxiv.org/abs/2004.13649)): "Existing model-free approaches, such as Soft Actor-Critic (SAC), are not able to train deep networks effectively from image pixels"; the fix is input augmentation "to regularize the value function". SAC-AE ([1910.01741](https://arxiv.org/abs/1910.01741)): adding reconstruction to off-policy RL "often leads to training instability" (cause: the VAE). Both are conv encoders on images — the nearest neighbours, not our 256-number row.
- **Critic feature norms.** Bjorck 2022 ([2110.11222](https://arxiv.org/abs/2110.11222), pixels, actor-critic): instability "leads to saturating nonlinearities"; fix "normalizing penultimate features" (§21). Hussing 2024 ([2403.05996](https://arxiv.org/abs/2403.05996)): "Overinflated Q-values are found not only on out-of-distribution but also in-distribution data and can be linked to overestimation on unseen action prediction propelled by optimizer momentum"; fix "a simple unit-ball normalization" — at high update-to-data ratios (ours is 1).
- **LayerNorm in TD.** Gallici 2024 ([2407.04811](https://arxiv.org/abs/2407.04811)): LayerNorm "can yield provably convergent TD algorithms" (theory; conditions not read). Lyle 2024 ([2402.18762](https://arxiv.org/abs/2402.18762)): no single fix suffices; "layer normalization and weight decay" together hold plasticity. Lyle 2024b ([2407.01800](https://arxiv.org/abs/2407.01800)): under normalisation, parameter-norm growth = a silent decay of the effective learning rate.
- **One modality dominating — supervised only.** Wang 2019 ([1905.12681](https://arxiv.org/abs/1905.12681)): "the best single-modal network always outperforms the multi-modal network"; modalities "overfit and generalize at different rates". Peng 2022 OGM-GE ([2203.15332](https://arxiv.org/abs/2203.15332)): a "dominated modality" leaves the others "under-optimized". Classification, not RL; nothing found for a value function.
- **More input numbers are not harmful by themselves.** OFENet ([2003.01629](https://arxiv.org/abs/2003.01629)): SAC-family agents "learn more efficiently with the high-dimensional representation than with the lower-dimensional state" (learned, informative features). Distractors do hurt: Distracting Control Suite ([2101.02722](https://arxiv.org/abs/2101.02722)) "perform poorly under distractions"; DBC ([2006.10742](https://arxiv.org/abs/2006.10742)) learns to ignore "task-irrelevant details" (pixels).
- **Architectures that control magnitude.** SimBa ([2410.09754](https://arxiv.org/abs/2410.09754)): running obs norm + residual block + "layer normalization to control feature magnitudes", into SAC. Lee 2018 ([1810.10191](https://arxiv.org/abs/1810.10191)): a separate representation per sense (vision, touch), fused into "a compact and multimodal representation" (self-supervised). DreamerV3 ([2301.04104](https://arxiv.org/abs/2301.04104)): "normalization, balancing, and transformations" (per-input encoders: memory, not read here).

**Checked in our installed code** (`/data/venvs/jack-std`, sbx 0.28.0, flax): `ContinuousCritic` = `concat(flatten(obs), action)` → [Dense(256) → LayerNorm → ReLU] ×2 → Dense(1); LN goes to the critic only (the actor is built without it). Flax Dense initialises lecun-normal, so at init each normalised input number adds the same variance to a hidden unit: **≈ 256/298 = 86% of every first-layer unit is eye** (the probe's 91% energy is the measured form). LN comes AFTER that mix — it rescales the joint sum, it does not rebalance the blocks. The hunger/thirst/body numbers enter as ~14% of each unit's input. SBX ships SimBa as one keyword: `SAC("SimbaPolicy", …)` (`SimbaContinuousCritic`: Dense → residual blocks → final LN → Dense).

**Reading J17 against the record.** (1) No source found says a dominant block of per-number-normalised VECTOR inputs makes a SAC critic overestimate — the "eye share" suspect is OUR inference, unbacked. (2) The record's cures for a runaway critic act on the critic's features (penultimate / unit-ball normalisation, LN + weight decay, SimBa), not on the input share; ours already has LN per hidden layer and still runs away — consistent with Lyle (one mechanism is not enough), not a test. (3) Modality dominance is on record only in supervised nets, where the dominant sense starves the others — the matching guess here is that the critic fits eye noise and under-uses the needs, which a lr0/blind arm cannot show.

**Published arms for the hunt** (each a recipe on record, not home-made; ranked by cost): (a) **SimBa critic+actor** — one keyword in SBX; differs from ours by its own running obs norm stacked on VecNormalize (disclose or turn VecNormalize off) and the residual blocks; (b) **LN + critic weight decay** (Lyle) — AdamW is one optimiser kwarg; (c) **penultimate / unit-ball feature normalisation** (Bjorck; Hussing) — needs a ~10-line critic class, not in SBX; (d) **a separate small encoder per sense before fusion** (Lee; DreamerV3) — an architecture change, END-FIT-friendly (it scales to more senses); (e) **eye ÷ √256** — no published source: an inventor's arm, so it runs scored-beside, not as the lead. Each arm carries its own box blind twin and reads critic loss at ≥ 100k (J17 addendum 2, 22:40).

**What the record does NOT answer** (cheapest first)
- Does the eye's input share drive the runaway? Arm (e) or (d) against its blind twin, 1 early seed (923), 100k, box (~80 CPU-min an arm).
- Does a published critic fix stop it? Arm (a) SimBa, same pairing — one keyword, the cheapest published test.
- Is it the eye's CONTENT or its WIDTH? A width-matched noise eye (256 numbers of shuffled real views) beside row128 — J17's blind arm carries width with zeros, not variance.

**Not verified.** Hussing's normalisation placement and DreamerV3's per-key encoders (abstracts only); Gallici's theorem's conditions; the action's dimension in the critic's input (not counted); whether SimBa's obs norm composes with VecNormalize in SBX without double scaling.

## 25. Beyond the jungle: a new world, another Jack, and being told (2026-11-22; serves M9 — `GEN.06`, `GEN.02`, `GEN.03`, `GEN.09`, `LG.11`)

**Question.** Does a life in world A speed him in world B over a fresh twin; can he learn by watching, model what another saw, pass knowledge down generations; USE a told fact? Long form: `/data/jack-research/scratch_sec25{A,B,C}/notes.md`.

**Already known** (read in the paper unless under Not verified)
- Continual World 2021, SAC MLP 4×256 + LN, 1M steps a task, 20 seeds: plain fine-tuning forward transfer **0.20** [0.17–0.23] (0.32 first ten tasks, 0.08 on revisits; reference ceiling 0.46); most continual methods negative (Table 1, [2105.10919](https://arxiv.org/abs/2105.10919)). With behaviour cloning of old states: 0.18 → **0.54**, the critic carrying most ([2209.13900](https://arxiv.org/abs/2209.13900)). Meta-World 4-stage: fine-tuned SAC "virtually indistinguishable" from scratch ([2402.02868](https://arxiv.org/abs/2402.02868)).
- Procgen, PPO, 200M steps: overfits small level sets; generalising starts ~500 levels, "as many as 10,000" to close the gap ([1912.01588](https://arxiv.org/abs/1912.01588)). URLB, 2M reward-free steps then 100k: on vector states "a random initialization is competitive with most baselines" ([2110.15191](https://arxiv.org/abs/2110.15191)).
- Watching: PPO+LSTM beside two experts on a 13×13 grid scores the SAME as solo on every seed; with a predict-the-next-view loss 4/5 seeds beat the experts; with no price on solo error it never appears ([2010.00581](https://arxiv.org/abs/2010.00581)). The 3D version fails without memory or an attention loss, and keeps following a WRONG leader ([2203.00715](https://arxiv.org/abs/2203.00715)).
- Generations: each from random init, predecessor in view with annealed probability, **1e5–3e5 steps a generation**, 10 seeds: TSP-24 route 20.6 → 17.4 → 13.1 (gen 0/4/7), beating one lifetime of equal total steps ([2406.00392](https://arxiv.org/abs/2406.00392)).
- False belief: passed only by a SUPERVISED behaviour predictor, not an acting agent ([1802.07740](https://arxiv.org/abs/1802.07740)).
- Told facts, from scratch: RTFM full task 23% plain, 65 ± 17% after a 4-stage curriculum of 50M frames each ([1910.08210](https://arxiv.org/abs/1910.08210)). Messenger's from-scratch baseline fits train 98% and tests 2.5% — memorises, does not read; the model that reads (85% / 85% / 10%) has **pretrained BERT inside** ([2101.07393](https://arxiv.org/abs/2101.07393)). Dynalang uses hints about dynamics with one-hot tokens at **281M weights, 50M steps**; a model-free learner got worse with more language ([2308.01399](https://arxiv.org/abs/2308.01399)).
- New word used at once: 0.98 with memory + a reconstruction loss, 0.33 (chance) without, at 1e9 steps ([2009.01719](https://arxiv.org/abs/2009.01719)). Unlived synonyms: from-scratch embeddings 0.49 (chance .50), frozen BERT 0.77 ([2005.09382](https://arxiv.org/abs/2005.09382)).

**Candidates**
- **Transfer = fine-tune + retention** (carry actor AND critic, clone old behaviour on old states). Proven: SAC, 1M steps a task, 0.54. Weak: pair-dependent, sometimes 0. Smallest: CW pairs. Ours at 300k–1M: 0.2–0.3 between like worlds (inf).
- **Watching = a brain that predicts what it sees (A), a task with hidden information, a price on solo error, mixed solo/social lives.** Proven on grids only, ~500k episodes. Plain B is the published negative. Ours: marginal at ≤ 1M.
- **Generations = predecessor visible, annealed away.** The only M9 result inside our budget (1e5 steps a generation); tiny tasks; by sight, not diaries.
- **Told facts = tokens the world model must predict (A), tiny vocabulary, roles re-drawn every life so memorising fails.** Smallest: Messenger S1, 1M steps (with pretrained T5). Ours: a prediction-probe effect at best at 1M.

**Ours beside theirs**

| Setting | Published | Ours | Cost |
|---|---|---|---|
| Worlds seen before the move | 500–10,000 levels | a new layout every life (`jack_env.py:403`) | SAME-rule world B is already in-distribution: `GEN.06` as written (`registry_expansion.py:7722`) reads accumulated training, not abstraction |
| Seeds | 20 (CW) | 3 (`GEN.*`, `LG.11`) | 0.20 ± pair noise is unreadable at 3 |
| Shuffled-rules / random-demonstrator control | none published | registered (`:7737`, `:7663`) | no known answer to check against |
| Learner that watches | recurrent + predictive loss | B: SAC MLP 256×2, no memory (`brain_b.py:40`) | B is predicted to FAIL `GEN.02`; only A can carry it |
| Transmission | live behaviour in view | diaries + teaching (`:7757`) | no precedent for a diary read by a from-scratch learner |
| Language model | 148–281M, pretrained text inside most | nothing pretrained inside (law 2) | every "reads" number above BERT/T5 is not ours to expect |

**Known-answer check.** Transfer: Continual World pair, SAC 1M steps a task, fine-tune = 0.20 on CW20 (~1 Kaggle CPU session a task, inf). Generations: Cook's Memory Sequence, PPO + GRU, 1e5 steps a generation, later generations beat an equal-budget single life (curve values not extracted). Told: BabyAI GoToRedBallGrey, 15.9–17.4k episodes to 99% ([1810.08272](https://arxiv.org/abs/1810.08272)).

**What the record does NOT answer**
- Biome / rule abstraction ("shelter" from "lean-to") at ≤ 1M steps on vector senses: nothing. Experiment: A → B with one rule changed vs fresh vs shuffled-rules B, ≥ 10 seeds; prediction 0–0.3, no gain beyond the shuffled control.
- A motor or foraging skill learned faster by watching through an eye, with a random-demonstrator control: nothing. Prediction: B no gain; A gains only if the food is hidden.
- False belief in a reward-only acting agent; a diary as the only inheritance; an anchored-against-unanchored told fact with a stripped control: no test found, pass or fail. `LG.11` would be new; the confound to shut out is the word statistics alone (BERT 0.77).

**Not verified.** XLand / AdA scale; Taylor & Stone's transfer measures; Ndousse's and CTG's step counts; Cook's and Dynalang's curve heights; BIB / AGENT (abstracts).

## 26. A sense built from outcome: "what was at my mouth when I ate" as the learner's input (2026-11-24; serves the lab's hindsight eye, rulings 84 and 98, M3; plan part "Learning to eat" + "Eye"; answers INBOX `RESEARCH:` Director 11-22)

**Question.** What is published on a sense built from OUTCOME (what was in view when reward arrived) and fed to the learner as input; did a small agent learn to reach food that way from scratch? Long form: `/data/jack-research/reports/sec26_outcome_built_sense_2026-11-24.md`.

**Already known.**
- The experiment itself is unpublished: no from-scratch agent on record builds an online table of the stimulus present at reward and reads it as an observation, with a plain-agent control.
- A reward table beats a net early, loses late. MFEC (2016), 3-D foraging from pixels: the table reached in < 3M frames what A3C needed > 40M for. NEC (2017), 57 Atari, median: 16.7% at 1M frames, 54.6% at 10M (DQN −0.7%, 15.7%), overtaken by 40M. arXiv 1606.04460, arXiv 1703.01988
- Closest input form: SAC 2×256 with a retrieved high-return past state joined to actor and critic input, 100k steps, 10 seeds: Hopper 1,677 ± 697 (TD3 374), HalfCheetah 2,788 (TD3 1,597), Walker 761 (loses to EMAC 966). Dense reward; input confounded with a reward bonus. arXiv 2401.10516
- A self-taught object channel, frozen BEFORE the learner starts, works at 100k: Transporter keypoints, Atari, 3 seeds: Pong 10.8 (Rainbow −19.5), Breakout 19.3 (3.3); loses Seaquest. arXiv 1906.11883
- A handed direction is decisive: PointGoal 99.9% with GPS+compass, 0 SPL at 100M without. Replacing it with an estimate costs 4–19 points at scale (ObjectNav 0.544 vs 0.731 with true segmentation, 10M frames) — those estimators are label-trained or PRETRAINED: not for inside him. arXiv 1911.00357, arXiv 2007.00643
- One negative for the input form: VizDoom, a detector's output re-fed as input was no better than baseline; as an auxiliary target it helped. arXiv 1609.05521
- Wrong guidance and replay: SAC "could be problematic" unless the error rate is under 0.2 (guidance as shaping, Point Maze; PPO held to 0.3). Our imprints are 26% wrong. [workshop 2025](https://aair-lab.github.io/genplan25/papers/40.pdf)
- Early useless data scars a replay learner: SAC over-fitted on its first transitions does not recover in 1M steps; resets lift DMC IQM 501 → 656. arXiv 2205.07802
- Outcome as reward, not input, costs more: Synthetic Returns 6e7 steps; RCE 200 success examples.
- Biology: one flavour–nutrient pairing gives 69–71% choice, 86% by the sixth (rats, Ackroff 2009); reward-paired colours pull human attention after 240–1,008 trials, graded by value (+8 / +16 ms), still at 4–21 days (Anderson 2011); extinction suppresses, does not erase (Bouton); giving-up time falls as the capture rate rises (Krebs 1974); pigeons can lock on a cue whose pecking cancels food (in a replication 6 of 8 did not).

**Candidates.**
1. **Table output as input (E2c with the sense's own direction).** Bearing + distance of the peak ray + a "table has an entry" bit, joined to SAC's input. Proven nowhere in this form. Strength: one variable from E2c's 4/5. Weakness: zero until the first babbled meal (median ~950 decisions), 26% wrong, drifts as the table fills; E2c's later loss (10/10) untouched.  Needs: the table kept across deaths; one reset that keeps the replay buffer. Expect: eats on fewer than 4/5 seeds, later than E2c, and loses it as E2c does.
2. **Residual on the reflex.** The reflex acts; SAC adds a correction (last layer zero at start, critic burned in). Proven: TD3 on a real robot in 8,000 samples; DDPG+HER lifts a 0.15–0.55 controller by 1M steps where scratch never touches the block (arXiv 1812.03201, 1812.06298). Strength: starts at the reflex's 70 meals, the guide is never removed. Weakness: a hand-written reflex becomes part of him (a ruler only, ruling 84) — END-FIT must rule. Expect: ≥ reflex from step 0, small gain by 100k.
3. **The table as a teacher of the eye** (the lab's unrun net). VizDoom: auxiliary beats input; UNREAL 54% → 87%, at 10M+ frames. The net form read .434 here. Expect: no gain at 100k.

**Ours beside theirs.**

| Setting | Published / biology | Ours (file:line) | Same or differs |
|---|---|---|---|
| Store | graded strength, error-driven; fast + slow stores | count, top texture wins (`hindsight_eye.py:141`) | differs: no value grading |
| Label | the thing pecked / tasted; delay 10–75 min tolerated | nearest ray at the meal (`hindsight_eye.py:130`) | same |
| First imprint | 1 pairing → 69–71%; chicks ~3 ingestions | 1 meal (`live_eye.py:39`) | same; 74% on food |
| Unlearning | suppression, counted in omitted expected rewards | deletion after 400 unfed decisions (`live_eye.py:109`) | differs: forgets what biology keeps |
| Sense ready first | Transporter: yes, frozen | empty at birth | differs: untested anywhere |
| Innate bias | chicks peck small 3-D things | none | differs: costs the ~950-decision wait |

**Known-answer check.** Candidate 1: E2c's handed arm must still read 4/5 beside it (the ceiling); the reset by SAC ± resets on DMC (IQM 501 vs 656; ~2 h a seed on a CPU kernel, our estimate). Candidate 2: Fetch curves only, no table number — its floor is the reflex's own 70 meals on worlds 1401–1408. The table: none at our size (NEC's 16.7% is at 1M frames).

**What the record does NOT answer.**
- Candidate 1 itself: E2c's cell, three arms — the sense's direction / a yoked table's (must fail) / handed (ceiling), 5 seeds, about E2c's cost. Prediction: 2–3 of 5 eat; yoked 0–1.
- A carried, frozen table (the Transporter form): one more arm. Prediction: 3–4 of 5 — then the loss is the empty start, not the 26%.
- Suppression against deletion, and a give-up that scales with his meal rate (ours is a fixed 50 decisions): box-minutes in `live_eye.cell`, no learner. Prediction: suppression ends the patience-50 failure and keeps the 2-of-8 repair.

**Not verified.** Williams & Williams (via Lesaint 2014), Rescorla-Wagner (memory); Hogan's "3 ingestions", Bouton (snippets); Krebs 1974, Yasuda 2012 (abstracts); UNREAL's and RCE's step counts; the VizDoom negative is figure-only, no numbers.

## 27. A ring of range rays under a replay learner, and a sparse reward whose explorer freezes (2026-11-25; serves J22's closed branch, the next eye row, Row B-6a / J23, the forge pool; answers INBOX `RESEARCH` Director 11-24)

Long form, every link and quote: `/data/jack-research/reports/sec27_lidar_eye_and_frozen_explorer_2026-11-25.md`.

**Question.** (a) How did the record feed a range row to a learner, and is h4's ±6-ray roll a valid DrQ augmentation? (b) What does SAC need when α reaches 0 before the first sparse reward (B-6a s3)?

**Already known**
- **No published goal-reacher reads a raw ray row.** Safety Gym: one 16-bin array PER OBJECT CLASS of closeness `max(0, 3 m − d)/3 m`, 0 = nothing, nearest wins; MLP 2×256, no conv (`safety_gym/envs/engine.py:993–1038`; Ray 2019 §4.1.4). ML-Agents FoodCollector: 7 rays × (one-hot tag + miss + fraction) = 49, MLP 2×128, 2M steps.
- **Replay learners on that encoding work, slowly.** OmniSafe PointGoal1: SAC 27.0 ± 1.5, TD3 27.3 at **3e6** steps (PPO 25.4 at 1e7). TD3 lidar navigator, MLP: 13 ± 7 % at 100k, 58 at 500k, 65 at 2M; 1-D CNN 55 (2210.04839).
- **Replay navigators pool hard, then MLP:** 10 beams (Tai 2017, 0.8M), 21 min-pooled sectors (Cimurs 2021, ≤ 400k), 1080 → 36 min-pool (Zhang 2020, SAC, 500k, ~90 %). Raw 1080 beams "fails completely" off its training map (NavRep).
- **DrQ's own rule:** an augmentation must be "optimality invariant", `Q(s,a) = Q(f(s,ν),a)` (2004.13649 §3.2). Rotation symmetry on record moves the ACTION too: `Q*(s,a) = Q*(gs,ga)` (2203.04439). RAD, walker 500k frames: crop 920, rotate 268, flip 223.
- **Critic-only conv encoder + sparse reward is the documented failure:** the "visual deadly triad" (2207.00986); "DrQ-v2 struggles to consistently solve Cartpole Swingup Sparse".
- **Shuffled never beats ordered on record:** Ant 2700 → 232 (2109.02869).
- **α → 0 on zero reward is SAC's dual, not a bug.** SBX `auto`: α₀ = 1.0, Adam on log α at 3e-4 (`sbx/sac/sac.py` L145–171). At the fastest possible fall α(10k) = e⁻³ = 0.0498; B-6a s3 measured 5.0e-2 — entropy stayed ABOVE target the whole time: he froze random, not still (inf). DrQ-v2 §4.4 dropped auto-α for "premature entropy collapse".
- **A zero seed on swingup-sparse is ordinary.** SB3 SAC, 20 seeds, 1e6 steps: white noise **377**, OU 608, **pink 730**; MountainCarContinuous 0 / 90 / 94 (Eberhard 2023, T.H.1); TD7 573 [333, 806] (MR.Q T.5). RL-Zoo's own sparse row: `ent_coef: 0.1` fixed + `use_sde` — the setting our row 3b passed with.

**Candidates**
- **A. Per-class closeness bins → the MLP he has.** 128 rays min-pooled to 16 bins per material class, `1 − d/8 m`, 0 = nothing; no running mean/std (already 0–1). Proven: Safety Gym, PPO 1e7, SAC 3e6. Weak: every published budget is 5–30× our pilot. Expect: little at 100k (13 % on record); apart from blind by 500k (inf).
- **B. Conv eye, roll removed** (or left–right mirror WITH the turn action mirrored), 3-step returns, τ 0.01. Proven on pixels only (DrQ Cup Catch 913 at 25k decisions). Weak: the triad; navigators found conv ≤ MLP. Expect: no better than A.
- **C. For the frozen explorer: correlated action noise.** Pink noise (drawn as one sequence per episode) or gSDE + fixed α 0.1. Proven: 377 → 730 on B-6a's own task. Needs an SBX port (the package targets SB3). An α floor alone keeps the same white noise (inf).

**Ours beside theirs**

| setting | published | ours | same / cost |
|---|---|---|---|
| ray encoding | closeness per class, 0 = nothing | `[d/8, scalar texture]` interleaved, miss = (1, 0) (`fast_sim.py:165–172`) | differs; no published success on ours |
| inputs | 16–49 per class | 256 flat, or 128×2 conv | 5–16× wider |
| augmentation | pixel shift, camera locked to body | roll ±6 rays = ±16.9° yaw, action unmoved (`conv_eye.py:122`) | NOT optimality-invariant (inf): it teaches that a turn does not matter |
| encoder signal | TD + shift; sparse → add aux loss | TD only | the triad's regime |
| n-step, τ | 3, 0.01 (v2) | 1, 0.005 | v2: 3-step "critical" |
| budget | 5e5–3e6 (DrQ-v2 sparse: 1.5M) | 1e5, one seed | under every published row |
| α | init 0.1 (DrQ); none (v2); fixed 0.1 (Zoo sparse) | auto from 1.0 (`brain_b.py:40`) | fine with reward; freezes without |

**Read from our own files this slot.** (1) h4's α stays 0.012–0.053 in all three cells (`j17_hunt_h4.json`): the forager is NOT a frozen explorer; that fault is B-6a's alone. (2) Whole-run meals/life: sighted 0.37 ± 0.13, blind 0.58 ± 0.14, shuffled 0.88 ± 0.29 (32–35 lives; one shuffled life ate 8): the "shuffled ate most" is inside two standard errors — no arm eats, none is ranked. (3) A rolled shuffled ring is still shuffled: the one arm the roll cannot hurt (inf).

**Known-answer check.** C: SB3 SAC + pink noise, MountainCarContinuous 0 → 94 (box-minutes, as row 3b), then swingup-sparse 730 at 1e6 (20 seeds; ≈ 1.5 core-h per 150k cell). A: SafetyPointGoal1 SAC 27.0 ± 1.5 at 3e6 — about a session a seed (inf).

**What the record does NOT answer**
- Does a ±6 roll flip the hand forager's turn? Box-minutes, no learner: his rule on rolled rows. Prediction: yes whenever food is inside ±17°, i.e. on every approach.
- Binned eye vs raw vs blind under B, same seed: 3 cells × 100k ≈ 1.8 core-h. Prediction: nothing eats at 100k; 500k is the first honest read. A new EYE, not an h5 — needs a ruling (B is closed on the wide eye).
- Pink noise on B-6a, 5 seeds × 150k ≈ 8 core-h. Prediction: zero seeds 1 → 0 of 5, mean in band; silent on win-then-lose (s1).
- Scalar id vs one-hot class, raw vs pooled beams under SAC: no published ablation.

**Not verified.** RAD's per-augmentation scores (heatmap text); steps-to-first-success anywhere (curves); the pink-noise paper's α and action repeat; DrQ-v2's entropy ablation (figure); SPR, DBC unread. `conv_eye.py`'s docstring cites two DrQ settings the config does not have (long form).

## 28. What a ray should say about WHAT it hit: a handed class, a stable look, or a code that moves between worlds (2026-11-25; serves J24's stop at (1), the INBOX lines JACK 07:10 + OUTSIDER 11-25, M3; plan parts "Eye" + "Learning to eat")

Long form, every link, table and the sub-agents' notes: `/data/jack-research/reports/sec28_what_a_ray_says_2026-11-25.md`.

**Question.** His eye codes each thing as a hash of its geom id (`w0.py:225`): three foods = three codes, each 0.01 from a wall's, re-dealt in every world. (a) What did published foragers' sensors return about identity? (b) What does a learner need when look→meaning is re-drawn each world? (c) Can an MLP carve a 0.01 slot from one scalar?

**Already known**
- **No published forager's ray returns a per-object code — 0 of 12 worlds read.** rllab Gather: two arrays, `apple_readings[10]`, `bomb_readings[10]`, `1 − d/range` ([gather_env.py](https://raw.githubusercontent.com/rll/rllab/master/rllab/envs/mujoco/gather/gather_env.py) L288–334). ML-Agents FoodCollector: per ray a one-hot over tags + miss + distance, 7 rays × 7 = 49 (`RayPerceptionSensor.cs` L214–233). Safety Gym: one geom GROUP per class, one cast per class; it never reads `geomid` (`engine.py` L973–991). Animal-AI 3: a tag per ray. No published success or failure exists for our sensor.
- **The same kind looks the same in every episode, everywhere.** Pixels too (ViZDoom, DMLab); Avalon switches food-appearance variation OFF (2210.13417). Integer ids are embedded, never read as a size (NetHack: 5,991 glyphs → 32-d each, 2006.13760 App. F).
- **Class in hand was still not enough for flat learners.** SwimmerGather **0.0 ± 0.0 for every algorithm at 25M steps**, AntGather −5.8…−0.1 of 8 (Duan 2016, T.1); HIRO AntGather 3.02 ± 1.49 at 10M (1805.08296 T.1). FoodCollector (PPO 1×256): 2M for reward 10.
- **Look→reward re-drawn per episode has only ever been solved with in-episode memory.** Harlow task (two random images, one pays): conv + LSTM, ~1e5 episodes × 32 threads, at ceiling in 43% of seeds (26% single LSTM) (Wang 2016, 1611.05763); 14 of 50 nets (Wang 2018). Same paper, goal re-drawn each episode: feed-forward A3C "unable to solve the task". Smallest: recurrent TD3/SAC, 64 units, 0.75–1.5M on toys (Ni 2021, 2110.05038).
- **Big agents fail at it too.** Alchemy: 156.2 against 145.7 for a random heuristic, ideal observer 284.4, > 5e8 steps — "a failure of meta-learning" (2102.02926).
- **A network learns smooth functions first.** Raw coordinate vs Fourier features on a text image (thin strokes): PSNR 18.40 → 30.47 (Tancik 2020, T.1); a value MLP needs 5× the epochs of a Fourier net (Yang 2022, 2206.04672).
- **Biology:** cone signals, no label; a kind keeps its signature from place to place; taste labels the look (bees: 1–3 rewarded visits). When looks are re-dealt: Harlow's monkeys, 344 problems.

**Candidates**
- **A. One stable look per KIND, a few 0/1 channels per ray** (floor food · water · rock · wall · panel), same in every world; nothing says which look feeds him. This IS ML-Agents' input: a one-hot tag is a colour with orthogonal channels (inf). Proven: FoodCollector 2M; pixels 1M–50M. Weak: 128 × (1 + K) inputs; no budget under 2M. Expect: apart from blind by 500k (§27).
- **B. The same looks pooled to 16 closeness bins per kind** (Safety Gym / rllab; J24 as written). Proven: SAC 3e6 (§27), HIRO 10M. Our lab's ruler: 16 bins 65 meals, 128 rays 55. Weak: fixes the pooling for him. Expect: as A, cheaper per update.
- **C. Keep today's code, add memory** (recurrent B, or the lab's live table as a sense). Unreliable with an LSTM at 1e5+ episodes; failed in Alchemy. Expect at 5e5 with memoryless B: an indiscriminate toucher at best (inf). Stage 3's question, not the instrument's.

**Ours beside theirs**

| setting | published | ours | same / cost |
|---|---|---|---|
| identity per ray | class array or one-hot; fixed table | one scalar, `(geom_id·0.37 mod 0.8)+0.1` (`w0.py:225`; `fast_sim.py:165–172`) | differs; no published case at all |
| same kind, same look | always | 3 foods = 3 codes; 23/23 worlds re-deal (`j24_class_probe.py`) | a Harlow problem per world |
| nearest non-food look | another channel | 0.01 = ~1/25 sd after `VecNormalize` (`pilot_b.py:462`) | a high-frequency slot, × 128 unshared rays |
| memory | LSTM/GRU wherever looks are re-drawn | none (MLP 2×256) | published feed-forward: fails |
| budget | 2M–25M | 1e5–5e5 | 4–250× under |

**Our own toy (one seed, supervised, clean labels — a pointer, not a row):** 37 codes from our formula, 4 foods, MLP 2×256: one-hot class 100% recall by step 250; scalar code 25–36% at 20,000 (`/data/jack-research/scratch_sec28C/code/toy.py`).

**Known-answer check.** None published for a scalar-coded ray. Nearest: FoodCollector 10 at 2M (Unity); AntGather 3.02 at 10M. Ours: the toy with 5 seeds, the real food ids of worlds 941–943, 200k steps and a third arm (kind look, K channels) — box-minutes. Pass line stated first: kind-look ≥ 99% recall within 10× one-hot's steps; scalar < 50%.

**What the record does NOT answer**
- One-hot kind vs a 3-number colour vs a scalar hue per kind: no ablation anywhere. The toy's third arm; prediction: both multi-channel forms pass, a scalar hue passes only if kinds are ≥ 0.2 apart.
- Does B's MLP read food's bearing from today's row at all? OUTSIDER E1 (~2 CPU-h). Prediction: < 10% explained in new worlds, > 80% with a kind look.
- A learner on a kind-look eye at our budget: J24 (2)–(6) unchanged. Prediction: sighted above blind; EATS at 500k unlikely (record: 2M+).
- Memory that finds food's look inside one life: unpublished under 0.75M, toys only. Stage 3.

**Not verified.** Tancik / Yang numbers: summarising fetch, not the tables. Menzel 1967, Harlow 1949, Cherkin 1969, Regan 2001: secondary sources. IMPALA's frames on DMLab-30; total steps for Harlow and Alchemy; AAI-3 foraging scores (plot only); PettingZoo Waterworld's sensor (404); any RL agent trained on AAI raycasts. More in the long form.

## 29. A from-scratch walker on a custom human-like body, on ONE free GPU: cost, shaping, failure modes, and the row to reproduce first (2026-11-26; serves ruling 119, stage 3's walking layer, M4; plan part "Body and movement"; answers INBOX `RESEARCH:` Director 11-26 06:40)

Long form (sources, configs, the 15-row reward table): `/data/jack-research/scratch_sec29A/notes.md` (scale, hardware), `scratch_sec29B/notes.md` (reward terms, failure modes), `scratch_sec29C/notes.md` (known-answer rows, software path).

**Question.** What does a no-mocap walker on a custom body cost with thousands of copies on one free T4, what shaping does it need, how does it fail, and which published body + number (≤ 9 T4-h) do we reproduce first?

**Already known**
- **Steps: 1.5–2×10^8, PPO, 8,192 copies, net 512-256-128.** MuJoCo Playground ([2502.08844](https://arxiv.org/abs/2502.08844)): Berkeley Humanoid 150M, G1 (29 DoF) / T1 200M; < 15 and < 30 min on two RTX 4090s; 106k–120k steps/s on one A100. **All use PD joint targets at 50 Hz around a standing pose; none uses torques.**
- **A torque humanoid on a Gym-style reward is faster:** Isaac Gym, 21 DoF, 4,096 copies: reward 5,000 in 4 min, 7,000 in ~17 min on an A100; **256 copies → ~2.7 h**; > 4,096 is slower, with worse gaits ([2108.10470](https://arxiv.org/abs/2108.10470)). MJX tutorial humanoid: 20M steps, 3,072 copies, ~6 min on an A100.
- **The only free-GPU measurement:** that tutorial on a Colab T4, 13–18 min, 5 runs (mujoco PR #1465) → a T4 is ~2.2–3× an A100's time. So 200M steps ≈ 1–1.6 h on a T4 **(inf)**.
- **PPO is not a safe default on every humanoid:** Brax's own paper: PPO did not solve its humanoid in 500M steps, SAC did in 5M ([2106.13281](https://arxiv.org/abs/2106.13281)). Playground's DM Control port (5 seeds, curve-read): HumanoidWalk PPO ~470 with seeds 100–720; SAC ~890 (840–920) at 10M; HumanoidStand PPO ~800 (620–870), flat under 100 until ~55M.
- **A custom body's first failure is memory, not learning:** a user's own humanoid in the tutorial ran out of memory on a T4 (406 possible contacts; fewer copies did not help). Fix: explicit contact pairs, 1 solver iteration (mujoco #2236).
- **Published failure modes, each with its fix:** velocity tracking alone → hopping on both feet (fix: a single-foot-contact reward or a gait clock; [2404.19173](https://arxiv.org/abs/2404.19173)); no air-time term → too-fast steps; no energy term → "high-frequency tapping" ([2111.01674](https://arxiv.org/abs/2111.01674)); no mirroring → one foot forward, one behind, seed-dependent (2610.08737, a 13-DoF custom body on MJX); torque + velocity reward → "entertaining but visually unsatisfactory gaits", seed-sensitive (Heess 2017).
- **A reward copied across bodies inverts:** Humanoid-v5's alive bonus (height ≥ 1.1 m) never fires on a 0.79 m body (2611.07679). Per new robot: retune reward weights, PD gains, randomisation ranges (2409.06366).
- **Off-policy at scale:** FastTD3 beats PPO on G1 flat (~31 vs ~22 at 1 A100-h), but a PPO-tuned reward gave it an unusable gait; FastSAC "unstable" ([2505.22642](https://arxiv.org/abs/2505.22642)).

**Candidates**
1. **PD-target velocity walker, Playground G1 recipe** — PPO; ~15 reward terms (tracking 1.0/0.75, air time 2.0, foot phase 1.0, pose, slip; 10% zero commands; pushes). Proven on five robots, 150–200M. Strength: takes the brain's command as input. Weakness: needs a PD layer, a default pose, foot sensors; scores are version-sensitive (G1 flat ~16.5 in one paper, ~22 in another). Ours: 1–2 T4-h a seed **(inf)**, if contacts are pruned.
2. **Torque walker, Gym/Isaac reward** — what his body is today. Proven to 7,000 in 17 A100-min. Weakness: unshaped gait; ~5,000 is standing (§16). Ours: 20–50 T4-min **(inf)**.
3. **Minimal shaped set on torques** (tracking + alternate-foot term + mirror + energy ramped in): **raced by nobody**.

**Ours beside theirs**

| Setting | Published | Ours | Cost |
|---|---|---|---|
| Actuation | PD targets, 50–100 Hz | 17 torque motors (`playground.py:102`) | no "zero action = stand"; add a fixed PD wrap or accept candidate 2's gaits |
| Integrator / step | Euler or implicitfast, 4–5 ms, CG ≤ 6 iterations | RK4 + PGS, 50 iterations (`assets/humanoid_full.xml:24`; `jack_env.py:353`) | **PGS does not exist in MJX**; RK4 costs several × — the GPU body is a new world (ruling 76 applies) |
| Contacts | 4–8 explicit pairs | every geom pair | out of memory on a T4 as-is (#2236) |
| Free GPU | A100 / 4090; T4 once | **Kaggle hands us a P100** (`gpu.py:192`) | see below |
| Brain rate | 5 Hz over a frozen 50 Hz walker, once (2603.00948) | 5 Hz | no ablation; torques degrade fastest at slow rates (Peng 2017) |
| Feet | contact sensors per foot | none | air-time/phase terms need them |

**Software path — the first risk.** JAX on a P100 is unproven: jax#33224 (open; read directly) shows `jax[cuda12]` 0.8.0 failing a convolution on a Tesla P100; MLP-only work is unreported. Playground's `main` is now Warp + torch; its JAX form is frozen at `playground==0.2.0`. Warp's PyPI wheel needs sm_75 (T4 yes, P100 no).

**Known-answer check.** Row W-1: Playground DM Control `HumanoidStand`, Brax PPO, config defaults (2,048 copies, nominal 60M — log the true count, the figure runs to ~88M), ≥ 3 seeds. Published ~800, seeds 620–870 (figure-read, ± 30 — the freezer re-reads the figure). Band: every seed ≥ 600 AND < 150 at 40M; control = zero-learning-rate twin. ~40–50 T4-min a seed **(inf)**. Fallback W-2: `HumanoidWalk`, Brax SAC, 128 copies, 10M: every seed ≥ 800; ~1.2–1.5 T4-h **(inf)**. **Before either: a 10-minute cell on the GPU Kaggle actually gives — `jax.devices()`, then a jitted, vmapped `mjx.step` of the humanoid at 2,048 copies, printing steps/s and peak memory.**

**What the record does NOT answer**
- Does MJX run on a P100 at all, and how fast? (The 10-minute cell. Prediction: MLP + physics run; 0.5–1× a T4.)
- Candidate 1 vs 2 vs 3 on ONE body: PD wrap vs raw torque, with and without mirror/energy. 9 runs ≈ 9–18 T4-h. Prediction: PD + alternate-foot term walks 3/3; raw torque hops or limps on ≥ 1 seed.
- A walker under a 5 Hz learned commander: no ablation in print. Prediction: holds.

**Not verified.** Every reward in the DM Control and joystick rows is read by eye from a figure; no paper tabulates a humanoid reward with its spread. An outside benchmark (SLM-Lab) gives PPO HumanoidStand 115 at 100M — it **contradicts** the ~800 and must be settled before W-1's band is frozen (**settled 11-27 in §30: read it before W-1**). PR #1465 and #2236's fix list: via a summariser. 2610.08737, 2611.07679, 2603.00948: read by one sub-agent, not cross-checked. Heess 2017's step count: not extracted. No P100 number for anything; no T4 run of a joystick task. Gait quality of the torque humanoid at 1e8+ PPO steps: nobody says.

## 30. W-1's band: does Playground's `HumanoidStand` PPO end near 800 or near 115? (2026-11-27; serves J28 cell 1, row W-1, stage 3's walking layer, M4; answers INBOX `RESEARCH:` Director 11-26 19:14)

Long form: `/data/jack-research/scratch_sec30A/notes.md` (paper, figure vertices, configs by version), `scratch_sec30B/notes.md` (the SLM-Lab row and spec), `scratch_sec30C/notes.md` (third sources; issue plot in `img/`), `scratch_sec30_extra/panels_final.txt` (all 25 panels, rough).

**Question.** §29's row W-1 quotes ~800 (eye-read) and an outside benchmark says 115 at 100M. Which is the published number, on which page, and can a band be frozen?

**Answer.** Both are real and they are different programs. The paper's number is **795.5 at 88.47M steps** (Fig. 9, p. 14). The 115 is another PPO implementation that never learned. **But nobody outside the authors has reproduced the 795, and three have tried** — so W-1 as written is not a safe known-answer row.

**Already known**
- **The paper: mean 795.5 at 88,473,600 steps; seed lines end 625.8 – 851.1** ([2502.08844](https://arxiv.org/abs/2502.08844) v1, Fig. 9 p. 14 — a vector figure, vertices read from the PDF stream, axis fit residual 0; no table of rewards exists anywhere in the paper). Caption says 5 seeds; the panel draws 7 lines.
- **The curve is a late jump.** Mean by eval (M steps → reward): 9.8 → 36, 19.7 → 46, 39.3 → 60, 49.2 → 70, **59.0 → 99** (seeds 71–125), 68.8 → 262, 78.6 → 629, 88.5 → 796. At the nominal 60M it is still under 100.
- **"60M" runs 88.47M** because the config's `num_resets_per_eval=10` makes Brax round up (9 × 10 × 983,040) **(inf)**. That key is absent from v0.0.4/v0.0.5 and from the paper's Table XII (p. 37), present from v0.1.0. Without it the same config stops at 61.9M — on the plateau.
- **The 115** (SLM-Lab `docs/BENCHMARKS.md` l. 802, run 2026-03-14): 114.86 = mean of 4 sessions (95–162) over the last ~10M of 100M frames, training-rollout returns. Flat all life (95 at 5M, 110 at 100M), policy entropy rising. Its own PPO: 1 unroll per update (61k transitions) where Brax uses 16 (983k), clip 0.2 vs 0.3, other nets, value normalisation, grad clip. Same code scores WalkerWalk 945, CartpoleBalance 968, CheetahRun 883; HumanoidWalk 47, HumanoidRun 19.
- **Playground's OWN script does not reproduce it.** Issue [#171](https://github.com/google-deepmind/mujoco_playground/issues/171) (closed, won't fix): `learning/train_jax_ppo.py`, 100M, 5 seeds, H100 → HumanoidStand **~50 at 105M**, smooth, no jump. Maintainer: the paper used `brax/training/learner.py`, which "repros locally" (no number posted); he could not find the difference. Issue #265: HumanoidWalk via a custom `ppo.train` → 20 where ~800 was expected, 10× the steps no help.
- **On-policy on this task has no other record:** original DM Control, A3C humanoid-stand **6.0 ± 0.1** at 1e8 (15 seeds); D4PG 946.5 (Tassa 2018, [1801.00690](https://arxiv.org/abs/1801.00690), table at 10^8).
- **Same paper, SAC:** 882.7 at 5M (seeds 776–928); 5,893 steps/s against PPO's 91,927 (Table IV p. 16).
- **Packaging (checked on PyPI 11-27):** `mujoco_playground` is **not a PyPI name (404)**. The package is `playground`; latest 1.1.0 (11-23) is Warp + torch, Python ≥ 3.12 — no `registry`, no `dm_control_suite_params`. The last JAX/Brax release is **`playground==0.2.0`**, whose humanoid defaults to `impl="warp"` (`humanoid.py:45`); the JAX path needs `config_overrides={"impl": "jax"}`.

**Candidates for the known-answer row**
1. **W-1 as written (HumanoidStand, Brax PPO).** One source, a figure; outcome hangs on an unidentified difference between two scripts and on a jump that starts after 59M. Expect at our budget: **~50 by `ppo.train` + Playground wrapper (#171), not 800.** A miss would not tell us whether our GPU path is wrong.
2. **W-1b: `WalkerWalk`, Brax PPO, same config.** Paper ~968, seeds 966–972 (rough read, `panels_final.txt`); SLM-Lab's different PPO 945 — two independent programs agree. Throughput higher than humanoid. Proposed band: every seed ≥ 900.
3. **W-2: `HumanoidWalk`/`Stand`, Brax SAC** (§29): 883–893 at 5M, tight seeds — but Brax SAC NaNs on DMC humanoids (issues #81, #153; open), ~5 h a seed on a T4 **(inf)**.

**Ours beside theirs**

| Setting | Published | Ours (`experiments/body/mjx_timing.py`) | Cost |
|---|---|---|---|
| Install | `playground==0.2.0` | `pip install … mujoco_playground` (:34) | **404 — the armed cell dies at install** |
| Backend | MJX `impl="jax"` (flag default in `train_jax_ppo.py:77`) | `registry.load(name)` (:42) → default `warp` | Warp needs sm_75: fails on a P100 |
| Trainer | `brax/training/learner.py` | `ppo.train` + `wrap_for_brax_training` (:54–55) | this is #171's path: ~50, not 796 |
| Copies | 2,048 (config) | config's own; J28's text says 8,192 | text only |
| Steps | 88.47M real | 10M (:19) — timing only | reward at 10M ≈ 25–36: says nothing |

**Known-answer check.** W-1b: `playground==0.2.0`, `impl=jax`, WalkerWalk, config defaults, 3 seeds, log true step count; band every seed ≥ 900; control = zero-learning-rate twin. ≈ 30–45 T4-min a seed **(inf: 60–88M at a T4's 1/2.5 of an A100)**. HumanoidStand is then run BESIDE it, unbanded.

**What the record does NOT answer**
- Which script difference makes the humanoid learn (`learner.py` vs `ppo.train`). 2 runs to 88M ≈ 1.5–2 T4-h. Prediction: `learner.py` jumps after ~60M on ≥ 1 seed; `ppo.train` stays < 100.
- Whether the jump survives 8,192 copies or a T4: nothing found.
- Fraction of seeds that never jump: 0 of 7 in one figure is all there is.

**Not verified.** The 88.47M arithmetic and "7 lines, mean of 7" are one reader's; my own rough pass over the same vertices found 5 HumanoidStand seed lines (625.8–841.0), so the seed count is open — the mean and the low seed agree. Other panels' values are that rough pass (axis frames not cleaned). #171's ~50 is read off a plot; the maintainer's "repros" has no number. DreamerV3, TD-MPC2, BRO, SimbaV2, FastSAC per-task tables: not opened. `learner.py` itself: not read.
