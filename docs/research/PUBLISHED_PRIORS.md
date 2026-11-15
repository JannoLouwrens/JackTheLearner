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
