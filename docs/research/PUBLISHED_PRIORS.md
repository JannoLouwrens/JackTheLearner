# PUBLISHED PRIORS — what other groups' tests already tell us

> `SYSTEM.md` law 3a (owner, 2026-11-15): read before you race. This is the
> standing reference for the PRIOR every race order must carry. One section per
> question; the Researcher (`scripts/research_prompt.md`) adds and refreshes
> them. One section per question, each ≤ 6 KB. Long form of §1 with every source link:
> `docs/research/PUBLISHED_EVIDENCE_2026-11-15.md` (six-part review, 2026-11-15;
> its raw notes are in `/data/jack-research/research_notes/`).
> **(inf)** = arithmetic or reasoning from published numbers, measured by nobody.
> `$N` = `/data/jack-research/research_notes/Published evidence for building Jack`.

## 1. Which whole brain, and how to test brains fairly (2026-11-15; serves M2, M3, order J1)

**Already known**
- One Dreamer configuration learned 150+ tasks in 8 domains from scratch, incl. Minecraft diamonds from image plus health/hunger scalars (DreamerV3, Nature 2025). DayDreamer learned on real robots in 1–8 h (~57k–72k decisions, inf) where SAC, Rainbow and PPO failed.
- Replay learners collect food on ray/lidar tasks in ~60k–170k decisions (Banana DQN ~138k, community result; lidar Goal SAC/TD3 ~100k). PPO: 0.8–2M, same tasks. Sparse or homeostatic versions: 2–50M+.
- At 25–50k steps Crafter agents eat in 2–9% of episodes; at 40–50k steps Animal-AI agents sit in the random range. **40,000 decisions is below every published foraging budget, for any design.**
- Avalon (3-D survival): with a curriculum 0.716, without 0.001: larger than any architecture gap on record.
- Tiny or no replay: SAC/PPO score 0.04–0.07 of normal over 50 tasks. White vs correlated action noise on the closest sparse action-cost task (MountainCarContinuous, PPO): 0 vs ~94.
- Drive-before-minus-drive-after is the published homeostatic reward, best of five forms in the one comparison; only the discount separates eating from not eating, so a 4 s horizon leaves almost no signal.
- A one-SD difference takes ~20 seeds; two rank nothing.
- No large from-scratch result for active inference, predictive coding or curiosity-only agents. Naively bigger networks get worse (BBF, AdA).

**Candidates**
- **A. Dreamer-style recurrent world model — recommended main brain.** One encoder per sense into one GRU state, trained mostly to predict its next sensations; actor and critic learn in imagined 15-step futures. Strengths: broadest from-scratch record; uses extra senses and words where model-free agents get worse (Dynalang); scales without retuning; replay built in. Weak at: compute per step (~2,100 min vs 15 for tuned PPO at 1M steps on Craftax-classic); long memory; high-joint bodies; small details. Needs: action-conditioned dynamics, imagination, a large replay trained hundreds of times per step, the robustness kit (symlog, two-hot, KL balance, scale-down-only return normalisation). Smallest published: `size1m` preset (no published score), 8M/12M with scores. **Our nets are 10–100× smaller than any published Dreamer score, and nobody has published its speed on a T4, a CPU or Apple silicon.** Expect on the rover: first meals in 10^4–10^5 decisions, regular foraging by 10^5–10^6 (inf).
- **B. Replay-based model-free actor-critic (SAC family, LayerNorm, optional small GRU) — the cheap control.** Most direct evidence on our task. Needs: §5's settings. Weak at: sample efficiency vs world models (10–200×), getting worse with added senses. Expect on the rover: improvement by 10k–50k, reliable by 100k–200k (inf).
- **C. Decoder-free latent world model (TD-MPC2 lineage) — for the human-like body later.** Best published efficiency on Humanoid/Dog; no discrete actions, no memory, needs informative reward; not a whole-creature candidate today.
- Watch list: transformer world models (best Crafter numbers, too costly).

**Ours beside theirs**

| setting | published | ours on 11-15 | verdict |
|---|---|---|---|
| experience per arm | 10^5–10^6+ | 40k (15 lives) | a pilot, not a race |
| seeds | 5–20, sized by a pilot | 2 | cannot rank |
| replay | whole run; batch 256; ~1 update/decision | 256 slots, batch 16, 1 per 4–8 | largest learner-side cost |
| exploration | pink noise per life, or held actions | white noise per 0.2 s | all-or-nothing on nearest task |
| discount | 0.99–0.997 (20–67 s at 0.2 s) | 0.95 (4 s) | meal 10 s away keeps 7.7% |
| reward | ×100 drive difference; pay on the eating event | ×1; 0.0025 at energy 0.95 | too small; nothing when sated |
| advantage scale | whole batch or running, scale-down only | per 64-decision segment | amplifies noise |
| world-model arms | action input + imagination | neither | not world models |
| memory | state carried, stored with replay | GRU zeroed each step | no memory was tested |
| start state | energy randomised | always full | early life has no hunger |
| curriculum | food close first, moved out on success | none | largest published effect class |

**Known-answer checks** (before an arm may race): the five probe environments (seconds each); CartPole-v1 500 in 50–100k steps; §5's Pendulum and MountainCar rows (not the one-run −157); bsuite deep_sea / umbrella_length / memory_len (< 30 CPU-min each). A Dreamer arm has no published small score: its gate is matching a model-free learner on the same tasks and the official code's outputs on one seed.

**Learnable?** Not yet shown. Three checks: a hand-written forager using only his senses; a behaviour clone of it into the learner's own network and inputs (closed loop, DAgger); a "cheat" learner given the vector to food. Nothing in print says if 16 undifferentiated rays can tell food from a wall.

**What the record does NOT answer** (E0–E9; long form): E0 pure-noise exploration probe (CPU minutes); E1 known-answer gate per learner; E2 learnability; E3 Dreamer `size1m` speed on T4 and Mac; E4 the first fair race, B vs A at 10^6 decisions; E5 distance-to-food curriculum; E6 reward and horizon ablation; E7 memory across death; E8 adding a sense; E9 plasticity watch. **Do not spend on:** more 40k races; curiosity bonuses while persistent noise reaches food; an active-inference or predictive-coding stack; bigger networks without stabilisers.

**Not verified** (long form): Yoshida et al. 2024 full text (setup from abstract and code; see §13); R2D2 and D'Oro second-hand; Banana Collector is a community report; Food Collector steps rest on an inferred unit conversion; T4-vs-A100 speed assumed; "2.5–4× more food contacts" for persistent noise is a toy sum. No paper ablates our settings jointly (the ranking is a synthesis); no published system combines all of Jack's requirements.

## 2. Building brain A (a Dreamer-style world model) correctly (2026-11-15; serves order J3, M3; plan part "His brain")

Long form: `$N/dreamer_engineering.md` (108 KB). Nothing below was measured by us.

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
- Many needs: 8×8 grid, DQN, 30k steps: one learner per need is near maximum at once; one learner for all needs 1k–10k exploration steps and worsens from 4 to 8 needs (Dulberg 2023).
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

**Known-answer check** (CPU): Dulberg's 8×8 grid, 4 needs, DQN, 30k steps × 30 agents: separate learners near maximum, single needs thousands of exploration steps (curve); add a needs-hidden arm (EVAAA predicts failure). < 2 h (guess).

**What the record does NOT answer**
- Leave-one-sense-out on the brain that eats: 6 arms × 5 seeds × 300k (~30 kernel-h, inf). Predicted: needs-hidden and eye-hidden fail; touch-, audio-hidden inside noise (< 15%).
- A test-time hide / shuffle / matched-noise test of a sense: no published protocol (all above are training-time); ours is new.
- Needs as one deviation each vs our six numbers: 2 arms × 5 seeds. No prediction.
- Gravity + gyro vs the upright cosine, human-like body, pushes. Predicted: ~6 points of survival.
- A taste label at the mouth; aversion after one delayed sickness; seeking warmth at ≤ 1M: nobody.

**Not verified.** Yoshida 2024 full text; EVAAA's steps and effect sizes (figures); Dulberg, the robot ant (summariser); MIMo, M3L, CoRAL scores (curves); MuMMI's and sensor dropout's step counts; Seymour 2019, Safety Gym, Crafter health: not read.
