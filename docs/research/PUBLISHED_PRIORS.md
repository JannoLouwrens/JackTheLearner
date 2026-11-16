# PUBLISHED PRIORS — what other groups' tests already tell us

> `SYSTEM.md` law 3a (owner, 2026-11-15): read before you race. This is the
> standing reference for the PRIOR every race order must carry. One section per
> question; the Researcher (`scripts/research_prompt.md`) adds and refreshes
> them. One section per question, each ≤ 6 KB. Long form of §1 with every source link:
> `docs/research/PUBLISHED_EVIDENCE_2026-11-15.md` (six-part review, 2026-11-15;
> its raw notes are in `/data/jack-research/research_notes/`).
> **(inf)** = arithmetic or reasoning from published numbers, measured by nobody.

## 1. Which whole brain, and how to test brains fairly (2026-11-15; serves M2, M3, order J1)

**Already known**
- One Dreamer configuration learned 150+ tasks in 8 domains from scratch, incl.
  Minecraft diamonds with a first-person image plus health/hunger scalars
  (DreamerV3, Nature 2025). DayDreamer learned on real robots in 1–8 hours
  (~57k–72k decisions, inf) where SAC, Rainbow and PPO failed.
- Replay learners collect food on ray/lidar tasks in ~60k–170k decisions
  (Banana DQN ~138k, community result; lidar Goal SAC/TD3 ~100k). PPO needs
  0.8–2M on the same tasks. Sparse or homeostatic versions: 2–50M and more.
- At 25–50k steps Crafter agents eat in 2–9% of episodes; at 40–50k steps
  Animal-AI agents sit inside the random range. **40,000 decisions is below
  every published foraging budget, whatever the design.**
- Avalon (3-D survival): with a curriculum 0.716, without 0.001. That gap is
  larger than any gap between architectures in the record.
- Tiny or no replay: SAC/PPO score 0.04–0.07 of normal across 50 tasks.
  White action noise vs correlated noise on the closest sparse task with an
  action cost (MountainCarContinuous, PPO): 0 vs ~94.
- Drive-before-minus-drive-after is the published homeostatic reward and the
  best of five forms in the one comparison; but only the discount separates
  eating from not eating, so a 4 s horizon leaves almost no signal.
- Detecting a one-standard-deviation difference takes ~20 seeds. Two
  seeds rank nothing.
- No large from-scratch result exists for active inference, predictive coding
  or curiosity-only agents. Naively bigger networks get worse (BBF, AdA).

**Candidates**
- **A. Dreamer-style recurrent world model — recommended main brain.** One
  encoder per sense into one GRU state; trained mostly to predict its own next
  sensations; actor and critic learn in imagined 15-step futures. Strengths:
  broadest from-scratch record; uses extra senses and words where model-free
  agents get worse (Dynalang); scales with size without retuning; replay built
  in. Weaknesses: compute per step (~2,100 min vs 15 for tuned PPO at 1M
  steps on Craftax-classic); long memory; high-joint bodies; small details.
  Needs: action-conditioned dynamics, imagination, a large replay trained
  hundreds of times per step, the full robustness kit (symlog, two-hot, KL
  balance, scale-down-only return normalisation). Smallest published: `size1m`
  preset (no published score), 8M/12M with scores. **Our nets are 10–100×
  smaller than any published Dreamer score, and nobody has published its speed
  on a T4, a CPU or Apple silicon.** Expect on the rover: first meals in
  10^4–10^5 decisions, regular foraging by 10^5–10^6 (inf).
- **B. Replay-based model-free actor-critic (SAC family, LayerNorm, optional
  small GRU) — the cheap control.** Most direct evidence on our current task.
  Needs: whole-run replay (never under ~50k), batch 256, about one update per
  decision, stochastic actor, observation normalisation. Weak at: sample
  efficiency vs world models (10–200×), getting worse with added senses.
  Expect on the rover: improvement by 10k–50k, reliable by 100k–200k (inf).
- **C. Decoder-free latent world model (TD-MPC2 lineage) — for the human-like
  body later.** Best published efficiency on Humanoid/Dog; no discrete actions,
  no memory, needs informative reward. Not a whole-creature candidate today.
- Watch list only: transformer world models (best Crafter numbers, too costly).

**Ours beside theirs**

| setting | published | ours on 11-15 | verdict |
|---|---|---|---|
| experience per arm | 10^5–10^6+ | 40k (15 lives) | far below; a pilot, not a race |
| seeds | 5–20, sized by a pilot | 2 | cannot rank |
| replay | whole run, 10^6; batch 256; ~1 update/decision | 256 slots, batch 16, 1 per 4–8 | largest learner-side cost |
| exploration | pink noise per life, or held actions | white noise per 0.2 s | all-or-nothing on nearest task |
| discount | 0.99–0.997 (20–67 s at 0.2 s) | 0.95 (4 s) | meal 10 s away keeps 7.7% |
| reward | ×100 drive difference; pay on the eating event | ×1; 0.0025 at energy 0.95 | too small; nothing when sated |
| advantage scale | whole batch or running, scale-down only | per 64-decision segment | amplifies noise when meals are rare |
| world-model arms | action input + imagination | neither | not world models by definition |
| memory | state carried, stored with replay | GRU zeroed each step | no memory was tested |
| start state | energy randomised | always full | early life has no hunger |
| curriculum | food close first, moved out on success | none | largest published effect class |

**Known-answer checks** (before an arm may race): the five probe environments (seconds each); CartPole-v1 500 in 50–100k steps; Pendulum-v1 ~−157 in 20k (SAC); MountainCarContinuous 88–95 in 20–50k with persistent noise; bsuite deep_sea / umbrella_length / memory_len (< 30 CPU-min each). A Dreamer arm has no published small score: its gate is matching a model-free learner on the same small tasks and the official code's outputs on one seed.

**Learnable?** Not yet shown. Three checks: a hand-written forager using only his senses; a behaviour clone of it into the learner's own network and inputs (closed loop, DAgger); a "cheat" learner given the vector to food. Whether 16 undifferentiated rays can tell food from a wall is answered nowhere in print.

**What the record does NOT answer** (E0–E9, costs and predictions in the long form): E0 pure-noise exploration probe (CPU minutes); E1 known-answer gate per learner; E2 learnability; E3 Dreamer `size1m` speed on the T4 and the Mac; E4 the first fair race, B vs A at 10^6 decisions; E5 distance-to-food curriculum; E6 reward and horizon ablation; E7 memory across death; E8 adding a sense; E9 plasticity watch. **Do not spend on:** more 40k races; curiosity bonuses while persistent noise reaches food; an active-inference or predictive-coding stack; bigger networks without stabilisers.

**Not verified** (long form): Yoshida et al. 2024 full text (setup from abstract and code); R2D2 and D'Oro second-hand; Banana Collector is a community report; Food Collector steps rest on an inferred unit conversion; T4-vs-A100 speed is assumed; "2.5–4× more food contacts" for persistent noise is a toy calculation, not calibrated to W0. No paper ablates our settings jointly (the ranking is a synthesis); no published system combines all of Jack's requirements.

## 2. Building brain A (a Dreamer-style world model) correctly (2026-11-15; serves order J3, M3; plan part "His brain")

Long form, every source and line number: `/data/jack-research/research_notes/Published evidence for building Jack/dreamer_engineering.md` (108 KB). Nothing below was measured by us.

**Already known**
- The agent is fully specified by the official code, `danijar/dreamerv3` at commit `e01491f` (the 2024–25 "v2" agent of the Nature paper): an 8-block GRU state with 32 categorical latents; symlog vector encoder and decoder; two-hot reward and critic heads (255 bins, zero-initialised); a continuation head; losses `1·rec + 1·rew + 1·con + 1·dyn + 0.1·rep` with 1 free nat; 15-step imagination from all 1,024 replayed states; a REINFORCE actor with entropy 3e-4 and `max(1, S)` percentile return scaling; LaProp + AGC(0.3) at 4e-5 with a 1,000-step warm-up; batches of 16 × 64; replay 5e6.
- For a 30-number observation: `size1m` ≈ 0.64M parameters, `size12m` ≈ 9.7M (inf).
- Load-bearing parts, by the paper's ablations: KL balance + free bits (no learning without them on Reacher Hard); return scaling with the `max(1, ·)` floor (sparse swing-up 850 → 200 without it); two-hot regression; reconstruction gives most of the learning signal (100 → 27 without it).
- The published small-scale scores are 2023 runs of the v1 agent at size S (train ratio 512, action repeat 2, 4 environments): 18-task proprioceptive DMC mean 442 / 626 / 757 at 100K / 200K / 500K env steps. **No score exists for `size1m`, nor for any port's v2 agent.**
- Where it fails: directed exploration (BSuite exploration 0.01, deep_sea 0.00, with the 200M model); long memory (BSuite memory 0.62); shrinking (12M needs 38M Crafter steps for what 400M reaches at 7M).
- Train ratio (gradient work per decision): more is better up to the largest published setting, 1024. Cutting 512 → 32 on Walker Walk cost about two thirds of the score (one user report).
- **Death.** Only the v2 REINFORCE actor learns from termination. The v1 actor for continuous actions (backprop through the dynamics; the default in sheeprl's `dreamer_v3` and in dreamerv3-torch) cannot: InvertedPendulum stayed at return 4. The environment must mark true death and cut-offs differently.
- A needs-based precedent (EVAAA: Dreamer, drive-reduction reward, termination when a need leaves range): Dreamer survived longest, agents without the needs inputs did not learn, and **some agents learned to end the episode when food could not be found.** Our reward has the same shape.
- Ports have had silent bugs: terminated/truncated confused (broke CartPole), 1.6% of replayed actions corrupted, resume regressions, a README timing taken from a different agent.
- Speed is measured nowhere on a T4, P100, CPU or Apple silicon. Arithmetic only, ±2×: T4 `size1m` 20–50 gradient steps/s, T4 `size12m` 2–5, 4-core CPU `size1m` 0.5–2.

**Take as-is** (code and designs, never weights)
- Reference: the official repo at `e01491f` (its `configs.yaml`, `agent.py`, `rssm.py`, `scores/*.json.gz`). Runner: **sheeprl 0.8.5 `dreamer_v3_5`** with `algo=dreamer_v3_5_1M` or `_12M`: the only maintained port with the v2 agent, a CPU default, and resume with replay (`buffer.checkpoint=True`, `checkpoint.resume_from`). It is six weeks old and has no published scores, so it earns trust only through the checks below.
- For him: every published value unchanged; one environment; action repeat 1 (a decision is already 0.2 s); train ratio as high as the hardware sustains, up to 1024, and logged; `terminated` only on true death; a meal reward of order 1; needs, touch and rays as one vector; the REINFORCE actor; a 5,000–10,000-decision prefill; DayDreamer's arrangement (one process acts, one learns, shared replay).
- Later parts that exist as code: R2-Dreamer's decoder-free loss (eye), R2I's memory core, Dynalang's one-token-per-step input (words), a Plan2Explore-style bonus or Curious Replay if exploration stalls.

**Reproduce first.** Protocol of the published runs: v1 agent, size S, ratio 512, repeat 2, 4 environments, 5 seeds; pass = our mean within ~2 SD at each point (from the official per-seed score file).

| task (proprioceptive DMC) | env steps | published mean ± SD |
|---|---|---|
| cartpole_balance | 100K / 200K | 794 ± 52 / 966 ± 17 |
| cartpole_swingup | 100K / 200K / 500K | 478 ± 20 / 778 ± 19 / 853 ± 13 |
| ball_in_cup_catch (sparse) | 200K | 949 ± 9; 5 of 5 seeds > 900 by 150K |
| reacher_easy | 200K / 500K | 929 ± 19 / 964 ± 7 |
| walker_walk | 200K / 500K | 799 ± 87 / 925 ± 34 |
| cartpole_swingup_sparse | 500K | 492 ± 368: ~3 of 5 seeds near 800, none above 50 before 100K |

Negative controls that must come out low: walker_walk at train ratio 32 (200–300 at 500K); BSuite deep_sea (0.00). A step-0 check from the code (v2, continuous actions): reward and value loss 5.541, `dyn` = `rep` = 1.0, actor std 0.893, predicted reward and value exactly 0. **Cost: the first three rows at 200K × 3 seeds are an estimated 25–60 T4-hours, so A's check takes one to two weeks of free GPU; a CPU cannot do it at the published train ratio.**

**The last 10%** (each an experiment; costs and predictions in the long form): (1) gradient steps per second on our hardware; (2) does a ~1M model reproduce the published S-model curves; (3) what train ratio the rover needs, and whether free-running the learner is safe; (4) can entropy-only Dreamer find food 3.8 m away in a 3,000-decision life (predicted: about half the seeds, more with near-first food or a disagreement bonus); (5) REINFORCE vs dynamics-backprop when death ends the life; (6) reward scale under `max(1, S)` (0.01 predicted to stay near random); (7) exact resume across 12-hour sessions; (8) a small Dreamer on BSuite's memory and credit tasks; (9) a life longer than any published run.

**Not verified:** the Nature article itself (arXiv v2 stands in); every speed figure (arithmetic); whether the paper's v2 tables are the 2023 runs (the score files are bit-identical, the author says v2 uses REINFORCE: unresolved).

## 3. His world, needs and staging: how worlds that taught agents were built (2026-11-15; serves J5, J7, M2; answers the INBOX `RESEARCH:` lines on need clocks and on fire, cooking and shelter)

Long form with file-and-line parameters: `/data/jack-research/research_notes/Published evidence for building Jack/world_needs_curriculum.md` (114 KB).

**Already known**
- **The first meal is big and near.** One meal restores 44–100% of the need bar in Crafter, Avalon, Neural MMO and Animal-AI; at Avalon's easiest setting food is 0.5 m in front of the agent. Small meals (10%) work only with dense, fully sensed, instantly re-placed food and long lives (the two-resource ant: 10 items in 144 m², a per-type range sensor longer than the arena, a 6,667-step passive lifetime).
- **Life is long vs passive death:** capped at 30× the passive time to death in Crafter (10,000 vs 338 steps) and 34× in Neural MMO. Resources can be sensed from the start position. Every life is a new layout; no level repeats.
- **OUR WORLD IS THE OUTLIER** (from `experiments/drives.py`, arithmetic): a floor meal is 8% of the bar, the smallest of any world reviewed; two floor foods on 66.9 s timers supply 0.0024/s vs a basal drain of 0.0017/s, enough for about a fifth-time activity level; the ladder apple carries ~62% of the maximum intake. A sustained life therefore needs the HARDEST skill first, the reverse of every published teachable world. (Constants were calibrated on the humanoid: confirm for the rover.)
- **A success-gated difficulty dial has the largest measured effect:** 0.716 vs 0.001 on Avalon's eat task (50M steps); 57% wins vs none in another study. One ceiling per skill, each life drawn uniformly below it, up on success and down on failure in small steps.
- In a no-reset life, helpful starting conditions that wear off beat reward shaping on every task compared, and a world that changes by itself took success from 0% to 70% (Ecological RL). Regret-based level designers did not beat plain randomisation (Craftax, Kinetix).
- **Degenerate optima on record:** agents stationary for ~1.3 billion steps under a "needs above half" reward (ForageWorld); agents that end a hard life on purpose (EVAAA; its −100 death penalty equals ~44 steps at one bound); being pulled off-task by a hand-added sub-reward.
- Expected order of learning (Crafter): drink or the adjacent resource first (5×10^4–10^5 steps), mobile food later (12–26% of episodes at 10^6), delayed farming never. 3-D with a curriculum at 5×10^7 steps (Avalon): eating 0.7 of human, moving to food 0.3–0.4, tools and composed tasks 0.00–0.16.
- From scratch in physics worlds: fort-building ~6×10^9 steps; shelter construction 3–12×10^9 samples; one shaped "put meat on a heat source" step ≤ 2M; tree-chopping 1M. **No from-scratch agent makes fire, cooks for its own hunger, or uses shelter for warmth.**

**Take as-is** (designs with code; parameters in the long form)
- Eat and drink by a distance test (0.5–1.0 m) with immediate re-placement at a new random position. Rays that report WHAT they hit (type plus distance). A smell that grows as 1/distance; a few thermal readings around the body. The hanging food needs a sense that can register it; horizontal rays cannot.
- Crafter's need arithmetic as the template: several bars, each emptied in ~190–280 steps awake, health lost only while a bar is empty and regained while none is, half-speed drains asleep; sleep as a commitment; a smooth day-night cycle. Random starting needs each life.
- Physics devices: grab by weld constraint within 0.25 m; lock by freezing joints; transformations as "near, touching or held, then remove and spawn"; per-object temperature with cooked and burnt thresholds.
- Staging: scarcity is the dial, not the starting point. Meal size, item count, distance and respawn delay become functions of a ceiling d, with today's world near d = 1; a share of lives unassisted from day one; progress judged only on fixed unassisted layouts at d = 1; an adult's help is a world setting tied to d, absent part of every life. Accept a stage only if a random or do-nothing policy neither always fails nor succeeds enough.
- Rungs in the order the record says is learnable: eat at the mouth → drive to visible food → food and water → warmth from an existing fire → night and sleep → one-contact gathering → feed a fire → cook → enter, close, then build a shelter.

**Reproduce first** (in our simulator, before our own findings are trusted)
- The two-resource ant as published (authors' `trp_env` and PPO settings): survival ~49,700 of 60,000 steps after 41.7M steps. Then the rover in the same arena: it should reach the cap sooner; if not, the fault is in his senses, actions or learner.
- Dense ray-sensed food collection: replay learners pass at ~1.4–1.7×10^5 decisions, PPO at ~7.5×10^5.
- The curriculum contrast: uniform sampling near zero, gated ceiling clearly above (published 0.001 vs 0.716).
- Minutes on a CPU: the authors' 8×8 grid where environment shaping beats reward shaping (0% vs 70%).

**The last 10%** (costs and predictions in the long form): a need-driven two-step chain in physics (free the fruit, then eat); warmth from a fire he must keep fed; making fire (predicted: not from undirected play at 10^7 decisions); cooking for his own hunger with no shaped reward (predicted: only with early provisioning); shelter for warmth and sleep (entering and closing: 10^6–10^7; building from parts: beyond a free budget unless much simplified); one value estimate per need; provisioning that fades in a persistent life.

**Not verified:** EVAAA's true need rates (its runtime configs are not published) and its reward (stated three ways in paper and code); no ablation of meal size or food distance alone exists in any survival world; the entry settings at which a ray-sensing rover's random policy eats must be measured here.

## 4. Body, movement and simulation speed (2026-11-15; serves the paused `D1.0`, `T2.01`, M4, J1's speed table; answers the INBOX `RESEARCH:` line on the control path)

Long form: `/data/jack-research/research_notes/Published evidence for building Jack/body_motor_simulation.md` (106 sources; a few figures measured on this server under load and marked so).

**Already known**
- **Walking with no motion capture is routine only with thousands of parallel copies and PPO:** 2×10^7–4×10^8 steps; 4–6 minutes on an A100; 13–18 minutes for 3×10^7 steps on a free T4 (one community measurement). Policies are small: 4×32 up to 512-256-128.
- **In ONE world, a first walk takes ~1M steps with TD-MPC2, BRO or SimBa** (DeepMind Control humanoid-walk 628–672), and the score is at the random floor before ~200k. Plain SAC, PPO, A3C and DreamerV3 sit at the floor for 1–4 million steps. **`T2.01` failing since August is the published expectation at our budgets.**
- A CPU can do it: a 29-DoF humanoid walked with no mocap after 10–30M samples, 3×64 networks, 4–15 h on 8 threads, given a mirror-symmetry loss and helping forces that are withdrawn; without the help it only stood or fell (Yu et al. 2018).
- **The control path (D1's question) is largely answered.** In the two direct tests (SimBa, BRO) a bigger actor did not help or hurt, and a bigger critic helped. Working humanoid controllers are 4×32 to 512-256-128; the largest from-scratch model acting every step on a humanoid is ~5M parameters. `D1.0`'s reading (the 0.5M controller leads the 57M trunk) is what the record predicts. The exact 57M contest is not in print.
- **A 0.2 s brain needs a reflex layer.** Published slow brains decide 6–10 times a second over controllers at 30–200 a second; no humanoid balances on decisions slower than 15–20 a second. He decides 5 times a second.
- Flat learners match hierarchy on shaped walking; hierarchy wins on sparse, long or object tasks, mostly through exploration. Where flat failed and two-level succeeded on humanoids (forage, carry, dribble) the low level came from mocap; Heess 2016 is the no-mocap precedent.
- **Hands on a walking humanoid are unsolved from scratch:** every HumanoidBench manipulation task misses its target; 10 billion samples gave 56% carry success; tool use emerged only with an abstract grab action.
- Mocap buys style and exploration, not return. It can stand outside as a judge (a style reward) or be distilled into a module inside him; which is allowed is the owner's rule, not the record's.
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

**The last 10%** (designs, costs and predictions in the long form): one human-like body that both moves and forages, flat vs a frozen reflex controller vs a learning one; how slow the brain may decide; the control-path contest as a clean, checkable test (small actor, large actor, large trunk as critic only, on humanoid-walk, pass mark 600); whether his own energy need yields an economical gait; staged release of the hands; a growing body; the rover brain moved into a walking body; hold, carry and strike with a grab primitive before fingers; throughput on our own machines (predicted: 500–2,000 rover decisions a second per 4-core kernel after the simulator changes, from 28).

**Not verified:** MuJoCo Warp on a T4, Kaggle CPU and M4-class figures (none published); the T4 MJX figure is one unmerged measurement; two source conflicts (SAC on humanoid-walk at 1M: 274 vs 2) are flagged in the long form.

## 5. Building brain B (the replay learner) and the PPO reference correctly (2026-11-15; serves order J2, M3; plan part "His brain")

Long form: `/data/jack-research/research_notes/Published evidence for building Jack/control_arm_engineering.md` (223 KB, 219 links; update timings measured on this server under load and marked so).

**Already known**
- **The RL Zoo benchmark table is ONE run per entry, often the best case.** The public Open RL Benchmark run store (anonymous read access) holds 10–22 seeds of the same configurations, and those are the references: SAC Pendulum-v1 at 20,000 steps, mean of all 100 training episodes −298.4 ± 10.3 (13 seeds); SAC on sparse MountainCarContinuous-v0 at 50,000 solved on 9 of 10 seeds; PPO Pendulum-v1 at 102,400, −184.5 ± 80.7; DQN CartPole-v1 392 ± 150 where the table says 500. **§1's "Pendulum ~−157 in 20k" came from that one-run table: use the rows below instead.**
- Plain SAC (2×256, 2 updates per decision) is strong on easy low-dimensional tasks and unreliable early on sparse ones: DMC cartpole-swingup 853.0 ± 9.2 at 50K decisions; swingup_sparse 93.8 ± 166.1 at 50K and 776.5 ± 30.2 at 150K (10 seeds, per-seed files in the SimBa repository).
- PPO is the slow, seed-fragile reference: default SB3 PPO has a median of 127 over 28 DMC tasks at 500K decisions where replay and model-based learners have 613–927; on sparse swing-up three sources give 7, 76 and 391. About four parallel environments is its published optimum.
- **Gradient updates are not free on our CPUs** (measured here, under load, upper bounds): one SAC update at batch 256 is ~50 ms of an ARM core in PyTorch SB3, ~20 ms in SBX (JAX) with 2×256 networks, ~1.9 ms in SBX with 2×64. Big-critic methods (CrossQ, SimBa, BRO, MR.Q) are GPU methods.
- Exploration that holds a direction is what solves the sparse classic task: the Zoo recipe is gSDE with a FIXED entropy coefficient 0.1, discount 0.9999, no random warm-up. Traps: the pink-noise package also changes the noise inside SAC's updates and consumes its sequence three times per step; SBX implements `use_sde` as a held noise draw, not SB3's gSDE.
- Memory in a replay learner has one well-documented design (Ni et al. 2022): separate GRU-128 encoders for actor and critic, 64-step windows; −168.6 ± 3.5 on velocity-only Pendulum at 51,000 steps; it fails with a 5-step window (~−1,380) or a shared encoder (~−422). Costly: ~0.3 s per update on a CPU. A stack of 4 observations equals a GRU on position-only tasks and fails on recalling an early cue.
- A Q-learner over 9 joint actions is a legitimate alternative for a 2-dimensional drive (DecQN matches D4PG on 1- and 2-dimensional tasks). No maintained library has prioritised replay or a recurrent off-policy learner.
- Silent bugs to test for: time limit vs death, next observation at episode ends, tanh correction, action scaling, target updates, normalisation statistics, evaluation mode, recurrent state.

**Take as-is** (code, never weights)
- A fresh venv for the checks (not the one his world is certified in): Python 3.12; `stable-baselines3==2.9.2`, `sb3-contrib==2.9.2`, `sbx-rl==0.28.0`, `rl-zoo3==2.9.2`, `gymnasium==1.4.0`, `mujoco==3.17.0` with `dm-control==1.0.50` (MuJoCo 4.0 is days old and later `dm-control` requires it), `shimmy==2.0.1`, `pink-noise-rl==2.0.1`. `box2d` has no Linux aarch64 wheel (LunarLander only on Kaggle).
- **Arm B:** SBX `SAC` (SB3's SAC does not expose LayerNorm), `policy_kwargs=dict(layer_norm=True)`, 2×256, learning rate 3e-4, batch 256, replay ≥ the whole run, tau 0.005, gamma 0.99 to start, 5,000–10,000 warm-up decisions, 1–2 updates per decision, observations normalised, no resets. Death is `terminated=True`, every administrative stop `truncated=True`.
- Exploration for B: the Zoo's sparse-task recipe (in PyTorch SB3, where its public record is) or pink noise with the sequence as long as a life; never both; either must pass row 3 below in the library actually used.
- B with memory: first the last 4 observations stacked; then Ni et al.'s design with its published sizes. PPO reference: SB3 defaults with `VecNormalize`, four simulator processes, gSDE sample period 4. A teacher showing food: keep the demonstration transitions in replay permanently and draw a fixed share of each batch from them (RLPD, DDPGfD).

**Reproduce first** (references are distributions; "accept" is the range for a 5-seed mean)

| # | task | run | steps | target |
|---|---|---|---|---|
| 1 | Pendulum-v1 | SAC, Zoo entry (lr 1e-3): the unmodified library first, then ours | 20,000 | mean return of all 100 training episodes −298.4 ± 10.3; accept −314.7 to −282.1 |
| 2 | Pendulum-v1 | PPO, Zoo entry | 102,400 | last-100-episode training return −187.6 ± 17.0; accept −213.2 to −162.0 |
| 3 | MountainCarContinuous-v0 (sparse) | SAC, Zoo entry with gSDE, PyTorch SB3; any SBX or pink-noise variant must pass the same | 50,000 | deterministic evaluation > 90 on ≥ 3 of 5 seeds (public: 9 of 10 in 93.4–95.3) |
| 4 | MountainCarContinuous-v0 | PPO, Zoo entry | 20,000 | training return 87.5 ± 4.2 |
| 5 | DMC cartpole-swingup, action repeat 2 | plain SAC 2×256, 2 updates per decision | 50K decisions | 853.0 ± 9.2 |
| 6 | DMC cartpole-swingup_sparse, reacher-easy, reacher-hard | same | 150K decisions | 776.5 ± 30.2, 968.0 ± 23.2, 955.8 ± 30.5; sparse near zero at 50K on most seeds |
| 7 | velocity-only Pendulum | recurrent SAC, Ni et al.'s configuration | 51,000 | −168.6 ± 3.5, and both controls must fail |

**The last 10%** (5 seeds per arm, fixed budget, one outcome: meals per 1,000 decisions over the last 50,000; costs and predictions in the long form). First, ~2 kernel-hours: how often a random and a noise-driven policy reaches food in a life; the gap between meals for a hand-written ray-follower (it sets the discount); whether four simulator processes give 4× the decisions; update timings on a Kaggle kernel. Then: narrow network at a high update ratio vs standard; which exploration finds food first (predicted: SAC's automatic entropy coefficient collapses below 0.01 before the first meal); whether the rover needs memory; continuous actor vs nine discrete actions; the gap to PPO; whether one demonstration of eating helps; 3-step returns; discount 0.99 vs 0.995.

**Not verified:** the pink-noise paper's tables and the replay-ratio-barrier paper (blocked); all our timings (one sample, load average 5–13). Three of the notes' sections were condensed from sub-drafts, 12 values spot-checked.

## 6. Memory, sleep and learning for a lifetime (2026-11-15; serves M7; plan part "Memory and sleep")

Long form, every source and number: `/data/jack-research/research_notes/Published evidence for building Jack/memory_sleep_continual.md` (115 KB). Nothing below was measured by us.

**Already known**
- **Working memory:** the GRU is the best-supported small core (POPGym aggregate 0.349, LSTM 0.255, memoryless 0.067). Linear-recurrent cores (FFM, S5, LRU) match it and train faster. Reward-trained memory falters at lengths around 250 steps, and nothing assigns credit across 250 steps. With food in view, memory gives no benefit.
- **Memory trained from replay must see the whole dependency:** chunks shorter than it give zero success (R2I). Store the recurrent state, add a burn-in (R2D2: 40 of 80 steps), never cross a death, feed previous action and reward.
- **Event store (the diary):** NEC's value table (learned embedding → best return, read by nearest neighbours) gives a large early gain that fades: 16.7% median Atari score at 1M frames vs DQN's −0.7%, overtaken at 40M. Recall of specific past events worked only when what is written is trained by prediction (MERLIN, HCAM), at ~10^9 steps. **Retrieval is always by learned similarity, never by word matching.** That is why our diary scores 0 on reworded questions (inf).
- **Across episodes:** memory emerges when it is not wiped, the objective spans episodes, and worlds differ between groups of episodes (RL²: second-episode paths 25% shorter after 1,000 mazes). A 100-episode scene memory added +2.1%. In one fixed world, weights fed by retained replay carry "where the tree was" (Mirowski).
- **Death:** rare, so it leaves replay, is forgotten and repeated unless a few death examples are kept forever (Lipton).
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

**The last 10%** (costs at 100,000 decisions per kernel-hour)
- How long a memory his foraging needs: zero the state every k steps at evaluation only (k = 1 … never), food in and out of ray range. ~6 kernel-hours. Predicted: flat with food in view, hundreds of steps when search is needed.
- Which store carries a life's learning in a fixed world: fork 10 seeds after 1M decisions into all kept, replay wiped, weights reset with replay kept, all wiped. ~220 kernel-hours. Predicted: wiping replay changes little at first; forgetting of rarely visited places follows.
- Does anything beyond weights and replay help him find the last life's food: food re-drawn every K lives; state carried or reset at death; diary kept, wiped or swapped; probe at the start of each life. ~240 kernel-hours. Predicted: no effect if food never moves; value diary helps earliest if it moves.
- Concentrated vs spread sleep at equal updates; model-only vs model-and-actor sleep. 160 kernel-hours (B), ~18 T4-hours (A). Predicted: little difference for the model; model-only safest.
- Also open: re-localising after rebirth (2 kernel-hours); a small Dreamer's actor forgetting (~24 T4-hours); sleep tied to his sleep NEED; diary keys drifting as the model learns (inf: re-encode during sleep).

**Not verified:** SPEQ and the Dreamer-actor result are single preprints; episodic-LSTM and HCAM across-episode numbers are figure-only; almost every memory result is from agents far larger than him.

## 7. Ear, words and advice (2026-11-15; serves M5, M6; plan parts "Ear" and "Words and advice"; answers the open question on advice in words)

Long form: `/data/jack-research/research_notes/Published evidence for building Jack/ear_words_advice.md` (99 KB). Nothing below was measured by us.

**Corrections to the first review (§1):** (1) "No published agent grounds words from raw speech from scratch" is too strong: a ViZDoom agent follows spoken "go to the X" through an FFT ear (Hegde 2021), and robots go to the object named by one of 4 real Speech Commands words (Chang 2019). Still unpublished: doing it with a world model, without a word-label loss, online in one life, with advice. (2) Dynalang is not ~10M: 281M on HomeGrid, 148M on Messenger; HomeGrid cost 3.75 A100-days. (3) SoundSpaces' 0.601 → 0.709 is SPL (path-weighted success), not success rate.

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
| ear | Speech Commands, 12 classes, speaker-disjoint; DS-CNN 24K / BC-ResNet-1 9.2K | 94.4% / 96.6–96.9% |
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

Long form: `/data/jack-research/research_notes/Published evidence for building Jack/test_method.md` (163 KB). Power figures are the notes' own exact enumeration or simulation (inf).

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

**Reproduce first** (harness self-checks)
1. Same-against-same through the real pipeline, ≥1,000 repetitions, at ≤ the stated rate.
2. A handicapped copy (learning rate 0) detected with the planned power.
3. Calibration arms at floor and ceiling.
4. Same seed twice → identical logs; harness writes the seed list.
5. Run k sees the same world in every arm.
6. Resume across a kernel boundary = uninterrupted run.
7. Every decision counted, including warm-up and prefill.
8. Reward-forever world gives 1/(1 − γ); true death no bootstrap; last observation true.
9. Probes on or off leave training logs identical.
10. Outcome code right on synthetic logs, including edge cases.
11. Analysis script frozen and hashed in the pre-registration.
12. Tools checked on a known case: 4/15 vs 0/15 gives Fisher 0.100, Boschloo 0.043.
13. Runs analysed = runs launched.

**The last 10%** (conservative choices, no literature answer): one run = one observation, never lives; primary outcome from normal living, frozen probes only as diagnostics; death ends the return, the 3,000-decision cap is a truncation; analyse unpaired unless pilot correlation > ~0.3; retention across rebirth = first window after vs last window before death, descriptive only; word every result "on this world, at this budget, as configured".

**Not verified:** no RL paper uses exactly this censored two-look combination; real learning-time distributions are unknown until the pilot; AdaStop's fixed build had one borderline 0.064 that needs re-checking.

## 9. A battery of standard tests for brains (2026-11-15; serves stage 2)

Long form: `/data/jack-research/research_notes/Published evidence for building Jack/capability_battery.md` (144 KB). Install checks and a few timings were run on this server under load (marked M).

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

Long form: `/data/jack-research/reports/Mac mini M6 speed for Jack.md`. No MuJoCo, SAC or Dreamer timing exists for any M6; every workload number is scaled **(inf)**.

**Already known**
- **Measured on the M6:** Geekbench 6 4,698 / 21,302 (A1 core ≈ 1,100; our VM ≈ 3,500; Kaggle 786 / 1,687). Cores: 2 super + 4 performance (4.5–4.8 GHz) + 6 efficiency (2.9 GHz). Wall power 2.1–4 W idle, ~55 W sustained, 70–74 W peak; fan ≤ 42 dB(A). 32 GB is the ceiling and cannot be added later.
- **MuJoCo, scaled:** a fast M6 core is 2.5–4× an Ampere core. The basis is Geekbench discounted by the one Apple MuJoCo figure (M3 Max 650K humanoid steps/s, all cores). Jack's world per fast core: RK4 7,000–12,500 sub-steps/s, implicitfast 26,000–48,000. Loop today 85–150 decisions/s (RK4) and 160–250 (implicitfast); after the loop fixes 300–800. Whole chip 4.5–8× our VM and 10–16× a Kaggle session; the efficiency cores add +15–35%.
- **SAC update (2×256, batch 256), scaled:** SB3 on the processor 3–15 ms; SBX 2–7 ms; MPS no faster. A 1M-decision B run takes 1.7–5.6 h; **20 runs take 6–22 h on six fast cores (inf).**
- **Dreamer, scaled, steps/s for 0.64M / 9.7M:** JAX on the processor 2–6 / 0.15–0.4; PyTorch on the processor 3–10 / 0.2–0.6; MPS 3–12 / 0.8–2.5; MLX (not yet written) 6–25 / 1.5–5. T4 arithmetic: 20–50 / 2–5. So the Mac is **0.2–0.6× a T4 on PyTorch and 0.3–1× on MLX**. Basis: T4 beat an M3 Pro 1.6–1.8× on ResNet-50; a GRU cell ran 2.1–3.5× slower on MPS than on the processor (M3 Max).
- **Decisive test for A at 0.64M, full ratio (inf):** 460–1,860 Mac h (3–11 weeks). The T4 alone needs 4–9 weeks; **the two together need 1.6–5 weeks.** 9.7M at full ratio: 2,200–7,000 Mac h, not affordable anywhere.
- **No Dreamer code has been reported running on Apple silicon.** JAX has no Mac GPU route (jax-metal abandoned). MuJoCo on the Mac GPU is not usable.

**Take as-is**
- Physics: native MuJoCo on the processor, one process per world, 6 on fast cores (≤ 12 in all), `nstep=40` in one call. No GPU physics.
- B: SBX on the processor, `OMP_NUM_THREADS=1`, 4–6 runs side by side; SB3 on the processor as the reference. Not MPS.
- A 0.64M: keep the fastest of official JAX (processor), sheeprl `dreamer_v3_5_1M` (processor) and sheeprl on MPS. A 9.7M: sheeprl on MPS. An MLX port only after it reproduces §2's DMC rows.
- **Never `nice 19`, `taskpolicy -b` or a launchd `ProcessType` of Background**: background priority pins work to efficiency cores (2–3× loss, no error). Cap worker counts in code instead.
- `sudo pmset -a sleep 0 disksleep 0 autorestart 1`; App Nap off; long runs under `caffeinate -dimsu`.
- **The camera needs a logged-in desktop session** (the CGL renderer hangs otherwise). Auto-login a dedicated user (FileVault off, or `fdesetup authrestart`), add an HDMI dummy plug, and run GPU and render jobs as that user's LaunchAgents; processor-only services as LaunchDaemons. A ray-cast eye needs neither.
- Data on an external case-sensitive APFS drive; paths from env vars. float32 before MPS; checkpoint and restart every N hours; compare with Linux statistically.

**Reproduce first** (first day; each replaces an estimate above)
1. `sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`, `pmset -g`, all versions: the core layout.
2. `testspeed humanoid.xml` at 1, 6 and 12 threads: 50–90K steps/s per fast core.
3. `mj_step(m, d, nstep=40000)` on Jack's world, integrators 0, 1 and 3: the RK4 and implicitfast rows.
4. Decision loop for 60 s, one process, with `powermetrics --samplers cpu_power`: 85–150 / 160–250 decisions/s.
5. The same loop as 1, 2, 4, 6, 8 and 12 processes: 750–1,300 today; +15–35% from efficiency cores.
6. SAC update, 200 warm-up and 2,000 timed: SB3 on the processor (1 and 4 threads), SBX, SB3-MPS; then 1–6 runs at once: 70–300 / 150–500 / 50–200 updates/s.
7. 4096² matmul in fp32 and bf16 on MPS and MLX: 2.5–5 TFLOPS.
8. GRU cell forward and backward on processor, MPS and MLX: 15–30 µs per GPU job.
9. Dreamer steps/s after 500 warm-up steps, dummy 30-number world: JAX on the processor, sheeprl on the processor, sheeprl-MPS (`PYTORCH_ENABLE_MPS_FALLBACK=0`) at 1M, then 12M for the best two. Run the same script on the T4 after 21 Nov: the Dreamer rows and the T4 arithmetic.
10. 2,000 sheeprl steps from a fixed seed on processor and MPS; compare losses and per-module gradient norms. Any divergence bars MPS (PyTorch issue 204359).
11. 2,000 renders at 64×64 with the user logged in, then over SSH, from a LaunchAgent and from a LaunchDaemon after a reboot; `torch.backends.mps.is_available()` in each: 300–2,000 fps, and which contexts work.
12. A 24 h soak (6 worlds plus 1 MPS learner) behind a wall meter, logging steps/s, `vm_stat` and `powermetrics` each minute: 55 W, about $7 a month, no throttling, no creep, and the slowdown from sharing the chip.

**The last 10%** (first anywhere; publish the logs): MuJoCo per-core speed on an M4-or-later core and on the M6's performance and efficiency tiers; SAC and Dreamer timings on any Apple chip; MPS vs processor vs MLX on a recurrent world model; Apple vs T4 on the same Dreamer script; offscreen render fps on macOS; whether MPS runs in a no-login LaunchDaemon; whether two Dreamer learners (one on MPS, one on the processor) beat one; a multi-day training soak on a Mac mini.

**Not verified:** prices ($1,499 for 32/512 from Abt and AppleInsider listings; $1,299 for 32/256 from one blog; not read on Apple's store); the M6 GPU's 4.9 TFLOPS (an estimate); T4 datasheet figures (not re-read); the clock and per-clock speed of the performance cores; MLX's 1.3–2.5× gain over MPS (one M5 repository); launchd key names; multi-day thermals.

## 11. Porting Dreamer to MLX, and renting a GPU for the large runs (2026-11-15; serves order J5, the decisive test in PLAN, and the move to the Mac)

Long form: `/data/jack-research/reports/Porting Dreamer to Apple MLX.md` (62-item contract with line links: `research_notes/Porting Dreamer to Apple MLX/dreamer_port_spec.md` §2).

**Already known**
- **Reference:** `danijar/dreamerv3` at `e01491f`; agent unchanged since Dec 2024; ~1,130 of 10,360 lines are the algorithm. Port for 74 inputs and 8 continuous actions: ~1,170 lines translated, 430 fresh plumbing lines, 700 test lines (~2,300). Faithful PyTorch ports: 2,300–2,800 lines plus tests.
- **Sizes for Jack:** `size1m` 647,705 trainable parameters, `size12m` 9,765,209 (slow critic 66,111 / 853,503). Rule matches r2dreamer's and TorchRL's counts exactly.
- **Corrections:** latents are 32 variables × **4 classes (1M) / 16 (12M)**, not 32 × 32. The 12M recurrent width is 2,048, not 1,024. The optimiser constant is 0.999 (paper 0.99). There is a 1,000-update warm-up, and the first update uses learning rate 0. Two-hot is computed in raw space on exponential bins (the paper says otherwise). Actor samples are not squashed. The return scale is not debiased.
- **MLX 0.32.4** (14 Oct 2026): has grad, compile, stop_gradient, JAX-style keys, sort, gather/scatter. **Lacks** scan, percentile and LaProp. Its GRU is the PyTorch cell; its norm eps default is 1e-5 (Dreamer 1e-4). Compile freezes any state not passed in. Gradient fixes landed in 0.32.0–0.32.1. No MLX Dreamer or RL agent exists.
- **Where it runs:** Linux wheels need glibc ≥ 2.35. On our server it runs only in a Debian 12 container (checked). CUDA needs SM ≥ 7.5, so the T4 is at the floor.
- **Earlier ports:** ~20 bookkeeping bugs vs ~18 maths bugs; bookkeeping bugs lived longest (~8 months). Fidelity came only from same-weights, same-batch loss and gradient comparison (sheeprl PR #377).
- **Rent, $/h (RunPod / Vast median, 11-12 / 11-15):** 3090 0.22–0.50 / 0.26; A5000 0.16–0.27 / 0.23; 4090 0.34–0.89 / 0.54; 5090 0.69–1.39 / 0.87; A100 1.39–2.19 / 0.80. Per-second, prepaid. **(inf)** job costs: Job 1, reproduction at 9.7M, $5–18; Job 2, decisive test, $4–34 at 0.64M and $72–409 at 9.7M; Job 3, a month-long lifetime, $115–266 (3090) to $338–511 (4090).

**Take as-is**
- Reference commit `e01491f`. Pin `mlx==0.32.4` and re-run rungs 1–11 on any upgrade. Port the config values, never class defaults. Import official weights by name (MLX kernels are transposed; the decoder input is [stoch, deter]); never re-seed.
- **Build order** (each step 1–2 slots; no step before the rung below it passes): stage 0 spike (container; Kaggle install; compile time and memory at both sizes; golden v0 reproducible) → config → maths → returns/normaliser → optimiser → layers → names → golden v1 → RSSM step → 64-step pass → heads and world-model loss → imagination and actor-critic → gradients → train step → acting step → compiled re-run → replay → driver → ratio loop → checkpoint → smoke. ~60–110 slots **(inf)**. Policy-only slice for owners' devices: ~15–25 slots.
- **Minimal plumbing:** in-memory replay of 65-step samples with stored latents, write-back and online queue; one-world driver; Ratio loop starting at 1,024 items; full-state checkpoint. Leave out images, discrete actions, multi-device, the logger and prioritised replay.
- **Staging:** official JAX only on paid cards; the port never sits on the paid path. Order: free proving → 1–2 h test on a 4090 with ≥ 24 cores (about $1–2) → Job 1 → Job 2 (only after reproduction, resume on a rented host, watchdog fired once, run sheet frozen) → Job 3. Self-destruct on exit or after 20–30 minutes stalled; small prepaid balance; delete, never stop. No spend authorised yet.

**Reproduce first** (G = golden file from the official code, made once on Kaggle CPU in float32 with the compiler off and noise from file)
0. Names, shapes, counts: exact (G).
1. Maths (symlog, bins, two-hot, unimix, KL with free bits, straight-through, Normal, Bernoulli): 1e-6; uniform and mirrored decode exactly 0, also under `mx.compile` (NumPy reference).
2. λ-returns and percentile normaliser: relative 1e-6 (NumPy).
3. Optimiser over 5 steps: relative 1e-6; warm-up schedule exact (G).
4. Initialiser statistics within 2%; zero heads exact.
5. Layers: 1e-5 (G).
6. One RSSM step: 1e-5 (G).
7. 64-step pass, golden samples forced on near-ties: 1e-4, no flips (G).
8. World-model losses: relative 1e-5 (G).
9. Imagination and actor-critic losses: relative 1e-4 (G).
10. Gradients: relative norm 1e-4 (1e-3 for RSSM leaves), cosine ≥ 0.9999 (G).
11. Five optimiser steps, step 1 changes nothing: 1e-3 on masked updates, 1e-6 on weights (G).
12. Bookkeeping (action alignment, reset flags, queue, write-back, death vs time-out, ratio count, bit-identical resume): exact.
13. Step-0 answers: ln 255 = 5.541263, std 0.8927, scale 1.
14. Curves: ≥ 5 seeds × 2–3 tasks, port vs official; confidence intervals overlap. Official runs are shared with §2's `size1m` test.

**The last 10%** (first anywhere)
- MLX trace and compile time and peak memory for the full unrolled step (64-step pass + 15-step imagination from 1,024 starts, with gradients) at both sizes. This is gate G0: proceed if `size12m` compiles in under 10 min and peaks under 8 GB **(threshold is ours)**.
- MLX speed on CUDA (T4, rented cards) and on the M6. Projected 6–25 / 1.5–5 steps/s on the Mac **(inf)**.
- Agreement between chip types (CPU, Metal, CUDA) over a 64-step unroll.
- Dreamer of any kind on Apple silicon.
- Real T4 and 4090 Dreamer steps/s, and the gain from packing several runs on one card.

**Not verified:** Kaggle's glibc and driver for MLX; the optax first-step-LR semantics (rung 3 settles it); the dependency versions of the published runs (unrecorded); rental speeds (scaled from one 3080 Ti figure); the 1e-4 unroll tolerance; whether owners' devices other than Macs can use MLX.

## 12. His eye: sight from his own camera, and a sense added later (2026-11-16; serves the decisive test in PLAN, M-eye, stage 3; answers the INBOX `RESEARCH:` line on his eye)

**Question.** What does the record say about giving a small from-scratch agent sight: the resolution and frame cost learners need, losses with no decoder, the cost of rendering, richer rays as a cheaper first eye, and how "a sense added later against present from birth" has been tested? *(Skeleton, 04:10 UTC; the findings land in this slot.)*
