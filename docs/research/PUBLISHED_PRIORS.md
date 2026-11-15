# PUBLISHED PRIORS — what other groups' tests already tell us

> `SYSTEM.md` law 3a (owner, 2026-11-15): read before you race. This is the
> standing reference for the PRIOR every race order must carry. One section per
> question; the Researcher (`scripts/research_prompt.md`) adds and refreshes
> them. Keep it ≤ 40 KB. Long form with every source link:
> `docs/research/PUBLISHED_EVIDENCE_2026-11-15.md` (six-part review, 2026-11-15;
> its raw notes are in `/data/jack-research/research_notes/`).
> **(inf)** = arithmetic or reasoning from published numbers, measured by nobody.

## 1. Which whole brain, and how to test brains fairly (2026-11-15; serves M2, M3, order J1)

**Already known**
- One Dreamer configuration learned 150+ tasks in 8 domains from scratch, incl.
  Minecraft diamonds with a first-person image plus health/hunger scalars
  (DreamerV3, Nature 2025). DayDreamer learned on real robots in 1–8 hours
  (about 57k–72k decisions, inf) where SAC, Rainbow and PPO failed.
- Replay learners collect food on ray/lidar tasks in about 60k–170k decisions
  (Banana DQN ~138k, community result; lidar Goal SAC/TD3 ~100k). PPO needs
  0.8–2M on the same tasks. Sparse or homeostatic versions: 2–50M and more.
- At 25–50k steps Crafter agents eat in 2–9% of episodes; at 40–50k steps
  Animal-AI agents sit inside the random range. **40,000 decisions is below
  every published foraging budget, whatever the design.**
- Avalon (3-D survival): with a curriculum 0.716, without 0.001. That gap is
  larger than any gap between architectures in the record.
- Tiny or no replay: SAC/PPO score 0.04–0.07 of normal across 50 tasks.
  White action noise vs correlated noise on the closest sparse task with an
  action cost (MountainCarContinuous, PPO): 0 vs about 94.
- Drive-before-minus-drive-after is the published homeostatic reward and the
  best of five forms in the one comparison; but only the discount separates
  eating from not eating, so a 4 s horizon leaves almost no signal.
- Detecting a one-standard-deviation difference takes about 20 seeds. Two
  seeds rank nothing.
- No large from-scratch result exists for active inference, predictive coding
  or curiosity-only agents. Naively bigger networks get worse (BBF, AdA).

**Candidates**
- **A. Dreamer-style recurrent world model — recommended main brain.** One
  encoder per sense into one GRU state; trained mostly to predict its own next
  sensations; actor and critic learn in imagined 15-step futures. Strengths:
  broadest from-scratch record; uses extra senses and words where model-free
  agents get worse (Dynalang); scales with size without retuning; replay built
  in. Weaknesses: compute per step (about 2,100 min vs 15 for tuned PPO at 1M
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
  efficiency against world models (10–200×), getting worse with added senses.
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

**Known-answer checks** (before an arm may race): the five probe environments
(seconds each); CartPole-v1 500 in 50–100k steps; Pendulum-v1 about −157 in
20k (SAC); MountainCarContinuous 88–95 in 20–50k with persistent noise; bsuite
deep_sea / umbrella_length / memory_len (< 30 CPU-min each). A Dreamer arm has
no published small score: its gate is matching a model-free learner on the
same small tasks and the official code's outputs on one seed.

**Learnable?** Not yet shown. Three checks: a hand-written forager using only
his own senses; a behaviour clone of the scripted forager into the learner's
own network and inputs (closed loop, DAgger); a "cheat" learner given the
vector to food. Whether 16 undifferentiated rays can tell food from a wall is
answered nowhere in print.

**What the record does NOT answer** (the only experiments worth our compute;
E0–E9 with costs and predictions are in the long form): E0 pure-noise
exploration probe (CPU minutes); E1 known-answer gate per learner; E2
learnability; E3 Dreamer `size1m` speed on the T4 and the Mac; E4 the first
fair race, B against A at 10^6 decisions; E5 distance-to-food curriculum; E6
reward and horizon ablation; E7 memory carried across death; E8 adding a
sense; E9 plasticity watch. **Do not spend on:** more 40k races; curiosity
bonuses while food is reachable by persistent noise; an active-inference or
predictive-coding stack; bigger networks without stabilisers.

**Not verified** (details in the long form): Yoshida et al. 2024 full text
(setup taken from abstract and public code); R2D2 and D'Oro read second-hand;
Banana Collector is a community report; Food Collector step counts rest on an
inferred unit conversion; T4 speed relative to an A100 is an assumption; the
"2.5–4× more food contacts" for persistent noise is a toy calculation not
calibrated to W0. No paper ablates our settings jointly, so their ranking is a
synthesis. No published system combines all of Jack's requirements.

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

**Reproduce first.** Protocol of the published runs: v1 agent, size S, ratio 512, repeat 2, 4 environments, 5 seeds; pass = our mean within about 2 SD at each point (from the official per-seed score file).

| task (proprioceptive DMC) | env steps | published mean ± SD |
|---|---|---|
| cartpole_balance | 100K / 200K | 794 ± 52 / 966 ± 17 |
| cartpole_swingup | 100K / 200K / 500K | 478 ± 20 / 778 ± 19 / 853 ± 13 |
| ball_in_cup_catch (sparse) | 200K | 949 ± 9; 5 of 5 seeds > 900 by 150K |
| reacher_easy | 200K / 500K | 929 ± 19 / 964 ± 7 |
| walker_walk | 200K / 500K | 799 ± 87 / 925 ± 34 |
| cartpole_swingup_sparse | 500K | 492 ± 368: about 3 of 5 seeds near 800, none above 50 before 100K |

Negative controls that must come out low: walker_walk at train ratio 32 (200–300 at 500K); BSuite deep_sea (0.00). A step-0 check from the code (v2, continuous actions): reward and value loss 5.541, `dyn` = `rep` = 1.0, actor std 0.893, predicted reward and value exactly 0. **Cost: the first three rows at 200K × 3 seeds are an estimated 25–60 T4-hours, so A's check takes one to two weeks of free GPU; a CPU cannot do it at the published train ratio.**

**The last 10%** (each an experiment; costs and predictions in the long form): (1) gradient steps per second on our hardware; (2) does a ~1M model reproduce the published S-model curves; (3) what train ratio the rover needs, and whether free-running the learner is safe; (4) can entropy-only Dreamer find food 3.8 m away in a 3,000-decision life (predicted: about half the seeds, more with near-first food or a disagreement bonus); (5) REINFORCE against dynamics-backprop when death ends the life; (6) reward scale under `max(1, S)` (0.01 predicted to stay near random); (7) exact resume across 12-hour sessions; (8) a small Dreamer on BSuite's memory and credit tasks; (9) a life longer than any published run.

**Not verified:** the Nature article itself (arXiv v2 stands in); every speed figure (arithmetic); whether the paper's v2 tables are the 2023 runs (the score files are bit-identical, the author says v2 uses REINFORCE: unresolved).

## 3. His world, needs and staging: how worlds that taught agents were built (2026-11-15; serves J5, J7, M2; answers the INBOX `RESEARCH:` lines on need clocks and on fire, cooking and shelter)

Long form with file-and-line parameters: `/data/jack-research/research_notes/Published evidence for building Jack/world_needs_curriculum.md` (114 KB).

**Already known**
- **The first meal is big and near.** One meal restores 44–100% of the need bar in Crafter, Avalon, Neural MMO and Animal-AI; at Avalon's easiest setting food is 0.5 m in front of the agent. Small meals (10%) work only with dense, fully sensed, instantly re-placed food and long lives (the two-resource ant: 10 items in 144 m², a per-type range sensor longer than the arena, a 6,667-step passive lifetime).
- **Life is long against passive death:** capped at 30× the passive time to death in Crafter (10,000 against 338 steps) and 34× in Neural MMO. Resources can be sensed from the start position. Every life is a new layout; no level repeats.
- **OUR WORLD IS THE OUTLIER** (from `experiments/drives.py`, arithmetic): a floor meal is 8% of the bar, the smallest of any world reviewed; two floor foods on 66.9 s timers supply 0.0024/s against a basal drain of 0.0017/s, enough for about a fifth-time activity level; the ladder apple carries about 62% of the maximum intake. A sustained life therefore needs the HARDEST skill first, the reverse of every published teachable world. (Constants were calibrated on the humanoid: confirm for the rover.)
- **A success-gated difficulty dial has the largest measured effect:** 0.716 against 0.001 on Avalon's eat task (50M steps); 57% wins against none in another study. One ceiling per skill, each life drawn uniformly below it, up on success and down on failure in small steps.
- In a no-reset life, helpful starting conditions that wear off beat reward shaping on every task compared, and a world that changes by itself took success from 0% to 70% (Ecological RL). Regret-based level designers did not beat plain randomisation (Craftax, Kinetix).
- **Degenerate optima on record:** agents stationary for about 1.3 billion steps under a "needs above half" reward (ForageWorld); agents that end a hard life on purpose (EVAAA; its −100 death penalty equals about 44 steps at one bound); being pulled off-task by a hand-added sub-reward.
- Expected order of learning (Crafter): drink or the adjacent resource first (5×10^4–10^5 steps), mobile food later (12–26% of episodes at 10^6), delayed farming never. 3-D with a curriculum at 5×10^7 steps (Avalon): eating 0.7 of human, moving to food 0.3–0.4, tools and composed tasks 0.00–0.16.
- From scratch in physics worlds: fort-building about 6×10^9 steps; shelter construction 3–12×10^9 samples; one shaped "put meat on a heat source" step ≤ 2M; tree-chopping 1M. **No from-scratch agent makes fire, cooks for its own hunger, or uses shelter for warmth.**

**Take as-is** (designs with code; parameters in the long form)
- Eat and drink by a distance test (0.5–1.0 m) with immediate re-placement at a new random position. Rays that report WHAT they hit (type plus distance). A smell that grows as 1/distance; a few thermal readings around the body. The hanging food needs a sense that can register it; horizontal rays cannot.
- Crafter's need arithmetic as the template: several bars, each emptied in about 190–280 steps awake, health lost only while a bar is empty and regained while none is, half-speed drains asleep; sleep as a commitment; a smooth day-night cycle. Random starting needs each life.
- Physics devices: grab by weld constraint within 0.25 m; lock by freezing joints; transformations as "near, touching or held, then remove and spawn"; per-object temperature with cooked and burnt thresholds.
- Staging: scarcity is the dial, not the starting point. Meal size, item count, distance and respawn delay become functions of a ceiling d, with today's world near d = 1; a share of lives unassisted from day one; progress judged only on fixed unassisted layouts at d = 1; an adult's help is a world setting tied to d, absent part of every life. Accept a stage only if a random or do-nothing policy neither always fails nor succeeds enough.
- Rungs in the order the record says is learnable: eat at the mouth → drive to visible food → food and water → warmth from an existing fire → night and sleep → one-contact gathering → feed a fire → cook → enter, close, then build a shelter.

**Reproduce first** (in our simulator, before our own findings are trusted)
- The two-resource ant as published (authors' `trp_env` and PPO settings): survival about 49,700 of 60,000 steps after 41.7M steps. Then the rover in the same arena: it should reach the cap sooner; if not, the fault is in his senses, actions or learner.
- Dense ray-sensed food collection: replay learners pass at about 1.4–1.7×10^5 decisions, PPO at about 7.5×10^5.
- The curriculum contrast: uniform sampling near zero, gated ceiling clearly above (published 0.001 against 0.716).
- Minutes on a CPU: the authors' 8×8 grid where environment shaping beats reward shaping (0% against 70%).

**The last 10%** (costs and predictions in the long form): a need-driven two-step chain in physics (free the fruit, then eat); warmth from a fire he must keep fed; making fire (predicted: not from undirected play at 10^7 decisions); cooking for his own hunger with no shaped reward (predicted: only with early provisioning); shelter for warmth and sleep (entering and closing: 10^6–10^7; building from parts: beyond a free budget unless much simplified); one value estimate per need; provisioning that fades in a persistent life.

**Not verified:** EVAAA's true need rates (its runtime configs are not published) and its reward (stated three ways in paper and code); no ablation of meal size or food distance alone exists in any survival world; the entry settings at which a ray-sensing rover's random policy eats must be measured here.

## 4. Body, movement and simulation speed (2026-11-15; serves the paused `D1.0`, `T2.01`, M4, J1's speed table; answers the INBOX `RESEARCH:` line on the control path)

Long form: `/data/jack-research/research_notes/Published evidence for building Jack/body_motor_simulation.md` (106 sources; a few figures measured on this server under load and marked so).

**Already known**
- **Walking with no motion capture is routine only with thousands of parallel copies and PPO:** 2×10^7–4×10^8 steps; 4–6 minutes on an A100; 13–18 minutes for 3×10^7 steps on a free T4 (one community measurement). Policies are small: 4×32 up to 512-256-128.
- **In ONE world, a first walk takes about 1M steps with TD-MPC2, BRO or SimBa** (DeepMind Control humanoid-walk 628–672), and the score is at the random floor before about 200k. Plain SAC, PPO, A3C and DreamerV3 sit at the floor for 1–4 million steps. **`T2.01` failing since August is the published expectation at our budgets.**
- A CPU can do it: a 29-DoF humanoid walked with no mocap after 10–30M samples, 3×64 networks, 4–15 h on 8 threads, given a mirror-symmetry loss and helping forces that are withdrawn; without the help it only stood or fell (Yu et al. 2018).
- **The control path (D1's question) is largely answered.** In the two direct tests (SimBa, BRO) a bigger actor did not help or hurt, and a bigger critic helped. Working humanoid controllers are 4×32 to 512-256-128; the largest from-scratch model acting every step on a humanoid is about 5M parameters. `D1.0`'s reading (the 0.5M controller leads the 57M trunk) is what the record predicts. The exact 57M contest is not in print.
- **A 0.2 s brain needs a reflex layer.** Published slow brains decide 6–10 times a second over controllers at 30–200 a second; no humanoid balances on decisions slower than 15–20 a second. He decides 5 times a second.
- Flat learners match hierarchy on shaped walking; hierarchy wins on sparse, long or object tasks, mostly through exploration. Where flat failed and two-level succeeded on humanoids (forage, carry, dribble) the low level came from mocap; Heess 2016 is the no-mocap precedent.
- **Hands on a walking humanoid are unsolved from scratch:** every HumanoidBench manipulation task misses its target; 10 billion samples gave 56% carry success; tool use emerged only with an abstract grab action.
- Mocap buys style and exploration, not return. It can stand outside as a judge (a style reward) or be distilled into a module inside him; which is allowed is the owner's rule, not the record's.
- Body curricula work when well scheduled (withdrawn forces, strong-to-weak torque limits, a child-then-adult body, hands frozen while learning to walk); a poor schedule can be worse than none.
- **Physics is not our bottleneck; our loop is.** A stock wheeled car steps 85,000 times a second on one loaded core of this server and a stock humanoid 20,000; the rover world manages about 2,800 because it uses the RK4 integrator (3.7× slower, measured) and does per-sub-step work and ray casts in Python. Published factors for fixing that pattern: 23× (logic moved out of Python), 2–3× (batching).

**Take as-is**
- Simulator: `integrator="implicitfast"` (or Euler); all sub-steps of a decision in one native call; `mujoco.batch`/rollout for several lives per process; `mj_multiRay` or rangefinder sensors for rays. T4: MJX with Brax PPO and the Playground environments. Mac: native MuJoCo on all performance cores (650K humanoid steps/s on an M3 Max, per the docs); no GPU physics there.
- Reflex walking controller, no mocap: PD joint targets at 20–50 a second; reward and weights from the Playground G1 joystick config, or Yu et al.'s three terms with the mirror loss; termination on torso contact; random pushes; a 3×64 to 512-256-128 MLP. Aids to copy: withdrawn balance and propulsion forces; an effort cost raised only while the task is met; hands frozen at first.
- One-world learner settings with humanoid evidence: SimBa (actor 1×128 block, critic 2×512 blocks, 2 updates per step, batch 256, weight decay 1e-2) or TD-MPC2 defaults.
- Two-level design: the brain emits a velocity or target command every 0.1–0.2 s to the reflex controller; a rover-trained brain can be kept as the high level over a new body-specific low level.

**Reproduce first** (cheapest first)
- Gymnasium Humanoid-v5 floor: random actions 110 ± 30 in 23.5 steps; zero actions 199 ± 3 in 40 steps (seconds).
- Raw physics speed on each of our machines (minutes). The MJX tutorial humanoid on a T4: its reward curve, wall-clock within 1.5× of 13–18 min.
- SAC on Gymnasium Humanoid: about 5,200–5,500 at 1M steps (one CPU kernel, hours).
- The 2018 CPU recipe: walks unaided, symmetry index about 0.01, and the no-help ablation only stands or falls (one 12 h kernel).
- DeepMind Control humanoid-walk with SimBa or TD-MPC2: 628–668 at 1M, ≤ 2 at 200k (3–30 T4-hours a seed).

**The last 10%** (designs, costs and predictions in the long form): one human-like body that both moves and forages, flat against a frozen reflex controller against a learning one; how slow the brain may decide; the control-path contest as a clean, checkable test (small actor, large actor, large trunk as critic only, on humanoid-walk, pass mark 600); whether his own energy need yields an economical gait; staged release of the hands; a growing body; the rover brain moved into a walking body; hold, carry and strike with a grab primitive before fingers; throughput on our own machines (predicted: 500–2,000 rover decisions a second per 4-core kernel after the simulator changes, from 28).

**Not verified:** MuJoCo Warp on a T4, Kaggle CPU and M4-class figures (none published); the T4 MJX figure is one unmerged measurement; two source conflicts (SAC on humanoid-walk at 1M: 274 against 2) are flagged in the long form.
