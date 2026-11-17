# KNOWN_ANSWERS — our builds against published numbers (ORDERS J3, PUBLISHED_PRIORS §5)

A build is trusted only after it lands a published number on a small standard
task. Environment: `/data/venvs/jack-std` (§5 pins). Code: `experiments/brain_b.py`.
Rules are in that file's docstring, committed before each run.

| # | task | run | steps | published | ours | seeds | verdict |
|---|---|---|---|---|---|---|---|
| 1 | Pendulum-v1 | SB3 SAC, Zoo entry unmodified (`lib`) | 20,000 | −298.4 ± 10.3; accept −314.7…−282.1 | −319.5 ± 34.8 (−312.5 −281.8 −305.6 −321.8 −376.0) | 0–4 | **MISS** |
| 1 | Pendulum-v1 | Brain B: SBX SAC + LayerNorm, Zoo entry (`ours`) | 20,000 | same band | −285.9 ± 10.0 (in band; does not count — lib missed) | 0–4 | not judged |
| 1 | Pendulum-v1 | control: never trains (`ctrl`), must fail | 20,000 | outside band | Kaggle 11-16: −1227.6 ± 35.5 | 0–4 | holds |
| 1b | Pendulum-v1 | re-registration, 15 fresh seeds, `lib` (rule in `brain_b.py`) | 20,000 | same band, unchanged | −309.3 ± 13.7 (SE 3.5) | 5–19 | **REPRODUCES** |
| 1b | Pendulum-v1 | Brain B `ours` (SBX SAC + LayerNorm) | 20,000 | same band | −302.0 ± 22.2 (SE 5.7) | 5–19 | **REPRODUCES** |
| 1b | Pendulum-v1 | control `ctrl`, must fail | 20,000 | VOID if ≥ −314.7 | −1227.7 ± 58.8 | 5–9 | holds |
| 2 | Pendulum-v1 | SB3 PPO, Zoo `ppo.yml` entry unmodified (`r2lib`; rule `7df7160e`) | 102,400 | −187.6 ± 17.0; accept −213.2…−162.0 (5-seed mean of last-100-episode return) | −221.5 ± 44.6 (−204.6 −178.5 −210.5 −217.2 −297.0; Kaggle jack-cpu-1794807554) | 0–4 | **MISS** |
| 2 | Pendulum-v1 | control `r2ctrl` (learning rate 0), must fail | 102,400 | VOID unless every seed < −213.2 | −1257.7 −1214.1 −1239.3 −1236.2 −1239.4 | 0–4 | holds |
| 2b | Pendulum-v1 | re-registration, 15 fresh seeds, `r2lib` (rule `c442a757`) | 102,400 | same band, unchanged | −199.5 ± 24.1 (SE 6.2; −195.8 −192.7 −195.0 −260.9 −211.6 −228.9 −165.6 −193.2 −173.7 −189.8 −183.3 −177.8 −217.2 −192.7 −214.2; Kaggle jack-cpu-1794808095) | 5–19 | **REPRODUCES** |
| 2b | Pendulum-v1 | control `r2ctrl`, must fail | 102,400 | VOID unless every seed < −213.2 | −1241.5 −1229.3 −1224.5 −1266.3 −1235.5 | 5–9 | holds |
| 3 | MountainCarContinuous-v0 | SB3 SAC gSDE Zoo entry unmodified (`r3lib`; rule `824b0a2a`) | 50,000 | deterministic eval > 90 on ≥ 3 of 5 (public 9/10 in 93.4–95.3) | 94.8 94.5 95.5 95.0 93.5 — 5 of 5 | 0–4 | **REPRODUCES** |
| 3 | MountainCarContinuous-v0 | control `r3ctrl` (no gradient step), must fail | 50,000 | VOID if > 90 on any seed | −0.3 −2.8 −3.5 −0.9 −0.8 — 0 of 5 | 0–4 | holds |
| 3b | MountainCarContinuous-v0 | Brain B `r3ours`: SBX SAC + LayerNorm, Zoo numbers (rule `b09adb6a`); ctrl `r3octrl` | 50,000 | same bar | 96.0 96.0 94.9 95.2 95.5 — 5 of 5 (Kaggle jack-cpu-1794805793) | 0–4 | **REPRODUCES** |
| 3b | MountainCarContinuous-v0 | control `r3octrl` (SBX, no gradient step), must fail | 50,000 | VOID if > 90 on any seed | −0.3 −0.1 −20.6 −0.1 −0.0 — 0 of 5 | 0–4 | holds |
| 4 | MountainCarContinuous-v0 | SB3 PPO, Zoo `ppo.yml` entry unmodified (`r4lib`; rule `5d794fba`), ctrl `r4ctrl` lr 0 | 20,000 | training return 87.5 ± 4.2 (n 10); accept 80.6…94.4 (m ± 3s√(1/5+1/10)) | 78.2 (87.4 92.9 84.0 82.1 44.9; seed 4 solved, then collapsed to ≈0 over its last 12 episodes; Kaggle jack-cpu-1794809416) | 0–4 | **MISSES** |
| 4 | MountainCarContinuous-v0 | control `r4ctrl`, must fail | 20,000 | VOID unless every seed < 80.6 | −1.0 −7.8 −2.2 −6.2 −4.9 | 0–4 | holds |
| 4b | MountainCarContinuous-v0 | re-registration, 15 fresh seeds, `r4lib` (rule `9147eef5`) | 20,000 | STRICTER: formula at k 15 → 82.4…92.6 (k 5 band reported only) | 72.6 ± 25.7 (86.4 18.0 22.3 84.9 72.4 85.1 92.3 79.7 66.5 92.0 82.6 90.1 87.3 92.7 35.9; median 84.9; seeds 6, 7, 19 solved then collapsed; Kaggle jack-cpu-1794810241) | 5–19 | **MISSES** → stack hunt |
| 4b | MountainCarContinuous-v0 | control `r4ctrl`, must fail | 20,000 | VOID unless every seed < 82.4 | 18.7 −1.5 −4.4 −9.5 −6.3 | 5–9 | holds |
| 4c | MountainCarContinuous-v0 | stack hunt: same entry under the PUBLIC store's own stack (py3.9, SB3 1.7.0, gym 0.21.0, torch 1.12.0, numpy 1.23.1; `r4pub`, rule `aa209cd5`, `experiments/row4_stack.py`, box) | 20,000 | reading A ≥ 3/20 collapse (< 60) / B ≤ 1 and mean in 82.6…92.4 / C otherwise | 82.3 ± 11.6, median 84.2 (86.0 82.4 82.7 78.9 83.9 91.4 80.1 88.6 93.5 82.0 88.3 42.1 92.4 68.7 88.9 84.0 85.9 84.4 69.2 92.0); 1/20 < 60 (seed 11 solved then collapsed), 2 more partial; vs ours (rows 4+4b, 20 seeds, 74.0 ± 23.9, 4/20): Mann-Whitney p 0.66, Fisher collapses p 0.34 | 0–19 | **VOID** (control) — data: C |
| 4c | MountainCarContinuous-v0 | control `r4pubctrl` lr 0, must fail | 20,000 | VOID unless every seed < 60 | **98.0** −2.1 −1.6 −8.6 −0.7 — seed 0's frozen initial policy solves every episode (98 each, 191–430 steps) | 0–4 | **BREACHED** |
| 4d | MountainCarContinuous-v0 | same-against-same, both stacks on the box: ours `r4dours` (SB3 2.9.2/gymnasium 1.4) vs published `r4dpub` (SB3 1.7.0/gym 0.21); rule `5d62e91a`, `experiments/row4d.py` | 20,000 | REPRODUCES iff Hodges-Lehmann 90% CI of ours − pub ⊂ ±5 AND collapse Fisher p ≥ .05; DIFFERS iff MW or Fisher p < .05; else extend to 40/arm | n 20: ours 74.2 ± 16.0 (4/20 < 60), pub 81.8 ± 15.8 (1/20 < 60; seed 25 = 20.6); HL −7.0, 90% CI [−15.7, −0.8]; MW p 0.064; Fisher p 0.34 | 20–39 | **INCONCLUSIVE** at 20 → extension (seeds 40–59) harvested below |
| 4d | MountainCarContinuous-v0 | the one pre-registered extension, pooled n 40 per arm (same rule, `experiments/row4d.py`) | 20,000 | same three readings; INCONCLUSIVE at 40 = MISSES | ours 72.6 ± 19.0, median 79.4, 9/40 < 60 (seeds 40–59: 71.0); pub 73.4 ± 28.5, median 83.1, 6/40 < 60 (seeds 40–59: 65.0, three runs end below 0); HL −3.75, 90% CI [−9.1, +0.6]; MW p 0.16; Fisher collapses p 0.57 | 20–59 | **MISSES** (not shown equivalent at 40) |
| 4d | MountainCarContinuous-v0 | control lr 0, 10 per stack | 20,000 | VOID if > 3/20 reach 60 or learners-vs-ctrl Fisher p ≥ .001 | 1/20 ≥ 60; learners vs ctrl Fisher p 2.3e−10 | 20–29 | holds |
| 4d | MountainCarContinuous-v0 | control lr 0, 10 per stack, on its distribution | 20,000 | VOID if > 3/20 reach 60 or learners-vs-ctrl Fisher p ≥ .001 | 1/20 ≥ 60 (pub seed 29 = 86.4; two more at 53.6, 50.4); Fisher p 3.7e−10 | 20–29 | holds |
| J10-1 | Humanoid-v5 (Gymnasium 1.4.0, MuJoCo 3.17.0, defaults) | floor: random actions / zero actions, 100 episodes each, reset seeds 0–99 (rule `c4dfd21c`, `experiments/humanoid_floor.py`, box, 3 CPU-s) | ≤ 1,000 per episode | §4: random 110 ± 30 in 23.5 steps; zero 199 ± 3 in 40 steps — an EARLIER ON-BOX measurement, not a literature figure; accept random mean in [80, 140], zero in [196, 202] | random 109.9 (SD 30.3, SE 3.0) in 23.47 steps (16–58); zero 198.5 (SD 3.2) in 40.32 steps (40–41); zero beats random | 0–99 | **REPRODUCES** (the floor; the reference's ± is the per-episode SD) |
| J3-4 | Pendulum-v1 | in-house `ppo-needs` (the seated tanh-AWR learner) through the one adapter (rule `3054fbe6`, `experiments/inhouse_rows.py`, box) | 60,000 (3× 20,000) | a 100-episode training window ≥ −314.7 on ≥ 3 of 5 | best windows −578.4 −937.8 −1005.4 −1062.0 −1119.8 — 0 of 5 | 0–4 | **FAIL** (set aside, ruling 24) |
| J3-4 | Pendulum-v1 | control: same core, optimiser never stepped | 60,000 | reaches on ≤ 1 of 5 | −1215.5 −1199.1 −1227.4 −1214.6 −1234.7 — 0 of 5 | 0–4 | holds |
| J3-4 | MountainCarContinuous-v0 | in-house `ppo-needs`, same rule | 150,000 (3× 50,000) | deterministic eval > 90 at 50k/100k/150k on ≥ 3 of 5 | best checkpoints −0.016 −0.004 −0.001 −0.016 −0.008 — 0 of 5; **0 goals in 750 training episodes** | 0–4 | **FAIL** (set aside) |
| J3-4 | MountainCarContinuous-v0 | control, optimiser never stepped | 150,000 | reaches on ≤ 1 of 5 | −0.0004 −0.0012 −0.0001 −0.0009 −0.0006 — 0 of 5 | 0–4 | holds |

## Row 1 — the miss, and what it is not (11-16)

**The reference is right.** Pulled 11-16 from the Open RL Benchmark store
(project `openrlbenchmark/sb3`, anonymous GraphQL read; raw at
`/data/jack-logs/brain_b/orb/`): 14 finished SB3 SAC Pendulum-v1 runs with the
Zoo entry, final `rollout/ep_rew_mean` at 20,000 steps (= the mean of all 100
training episodes): −298.3 ± 9.9, range −315.1 … −283.5. Per seed: −315.1
−314.7 −308.1 −304.6 −302.5 −299.6 −297.8 −297.2 −293.8 −292.3 −290.8 −289.2
−286.4 −283.5. Stack in their configs: Python 3.8, SB3 1.x, gym 0.21
(Apr 2022 – Jan 2023).

**Our lib is worse, in both phases.** Two of five seeds fall below the
reference's worst. By 4-episode blocks: seed 4 learns about one block late
(eps 13–16 −757; reference mean −277, worst −609) and collapses once after
learning (eps 21–24 −903). Eps 21–100: lib −170.6 (seeds −173.5 −146.3
−177.6 −158.7 −196.7) vs reference −156.8 (−178.0 … −131.4). SBX `ours`
on the same seeds: −162.7.

**Why row 1b and not a fix yet.** Our lib's seed-level sd (34.8) puts the
5-seed mean's standard error (15.6) at the band's half-width (16.3): five
seeds cannot separate a broken stack from a heavy tail (rank argument: two of
five below the min of 14 has p ≈ 0.06 under one distribution). Row 1b runs
15 fresh seeds against the unchanged band; row 1 stays a miss. If lib misses
1b, the stack is the suspect: SB3 1.x / gym 0.21 / Python 3.8 against ours,
one difference at a time.

## Row 1b — both arms reproduce, with one honest caveat (11-16, kernel jack-cpu-1794795073)

Verdict by the pre-registered rule: **lib REPRODUCES (−309.3), ours
REPRODUCES (−302.0)**, control −1227.7 (run valid). Per seed, lib: −301.8
−315.6 −310.2 −300.1 −314.8 −295.5 −311.0 −311.9 −332.7 −317.9 −273.5 −322.5
−314.3 −316.7 −301.7; ours: −287.6 −303.9 −310.9 −308.3 −294.9 −293.1 −313.7
−316.9 −295.9 −358.1 −251.8 −304.9 −306.2 −300.8 −282.9.

Secondary (registered as reported, not deciding): two-sided Mann-Whitney
against the reference's 14 runs — **lib p = 0.011**, ours p = 0.42. Seeds
below the reference's worst (−315.1): lib 5 of 15, ours 2 of 15. So our
PyTorch `lib` stack sits about 11 points below the 2022–23 reference stack
(Python 3.8 / SB3 1.x / gym 0.21) — inside the band, but distinguishable from
it; lib over all 20 seeds (rows 1 + 1b) is −311.9. Brain B (SBX) is not
distinguishable from the reference. Read: the build is not broken (the rule
the order set is met), and the residual lib-vs-reference shift is a stack
difference to remember when row 3 compares SB3 numbers, not a reason to stop.

## Row 3 — the library reproduces; Brain B's exploration is a different recipe (11-16, kernel jack-cpu-1794803164)

`r3lib` (SB3 SAC, Zoo gSDE entry unmodified) solved all 5 seeds, deterministic
eval 93.5–95.5 — inside the public store's 93.4–95.3; first training episode
above 90 at episode 2–4; 352–459 training episodes in 50k steps. Control
(uniform random, no gradient step): −3.5 … −0.3, 0 of 5. Stack on the kernel:
Python 3.13, SB3 2.9.2, torch 2.11.0+cpu, gymnasium 1.4.0.

Before row 3b ran, the SBX source was read (sbx 0.28.0): `SACPolicy` accepts
`log_std_init` and `use_expln` and **silently ignores both** — the actor's
log-std is a state-dependent head (initial std ≈ 1, not exp(−3.67) = 0.025);
and SBX's `use_sde` holds one noise draw between `reset_noise()` calls (every
32-step rollout here), which is not gSDE's state-dependent noise matrix. So
row 3b tests whether B as built explores well enough for sparse reward under
the Zoo's numbers; a miss sends B to the pink-noise rule (§5), not a re-run.

## Row 3b — Brain B reproduces on sparse reward (11-16 05:19, kernel jack-cpu-1794805793)

`r3ours` (SBX SAC + LayerNorm, Zoo numbers, rule `b09adb6a`) solved all 5
seeds: deterministic eval 94.9–96.0 (worst single eval episode 94.5), against
the bar > 90 on ≥ 3 of 5 and the public store's 93.4–95.3. First training
episode above 90 at episode 1–4; 402–532 training episodes in 50k steps;
~335 s a seed. Control `r3octrl` (SBX, uniform random, no gradient step):
−20.6 … −0.0, 0 of 5 → valid. Stack: Python 3.13.18, sbx-rl 0.28.0, jax/jaxlib
0.11.1, flax 0.12.12, gymnasium 1.4.0.
So the two SBX departures named before the run (std starts ≈ 1; noise held
per 32-step rollout rather than gSDE) did not cost the sparse-reward row:
B may use its own exploration as built; the pink-noise fallback is not
triggered. Rows 1b and 3b are both reproduced by B; owed: rows 2 and 4 (PPO).

## Row 4c — the stack hunt (11-16)

**What was diffed.** The public store's runs (wandb `openrlbenchmark/sb3`, run
`2guwltm4` requirements.txt) used Python 3.9.16, SB3 1.7.0a12, rl-zoo3 1.7.0a12,
gym 0.21.0, torch 1.12.0, numpy 1.23.1. Source diff against ours (SB3 2.9.2,
gymnasium 1.4.0): the MountainCarContinuous step function is identical; rl-zoo3
1.7.0 copies gamma into VecNormalize as 2.9.2 does and its ppo.yml entry is the
same; PPO rollouts, timeout bootstrap and VecNormalize are behaviourally equal.
The one behavioural line, SB3 1.7's per-minibatch `reset_noise` in PPO.train,
only consumes RNG.

**What the machine said.** The published stack itself, run here on 20 seeds,
gives 82.3 ± 11.6 — the same solve-then-collapse failure (seed 11: 42.1) and
two partial ones (68.7, 69.2). It would itself miss row 4b's band
[82.4, 92.6]. Against ours it is indistinguishable at 20 seeds (Mann-Whitney
p 0.66; collapses 4/20 vs 1/20, Fisher p 0.34). The store's 10-seed sd 4.2
understates the recipe's spread: our band, built from it, is narrower than the
published code can meet on this machine.

**Why it is VOID anyway.** The registered control (`lr 0`, every seed < 60)
breached: seed 0's untrained initial policy solves MountainCarContinuous on
every episode (98.0). A frozen random network that pushes with the velocity's
sign solves this task; ours drew 0 of 10 such inits, the public stack 1 of 5.
A per-seed "every control below" clause is brittle on a task a lucky init can
solve; the next rule must state its control on the control's distribution,
written before it runs, not relaxed after it.

## Row 4d — same-against-same, first 20 seeds (11-16 13:05)

Both stacks on the box, fresh seeds 20–39, so host and seeds are not factors.
The control held on its distribution (1 of 20 frozen inits reach 60). At 20
seeds the reading is INCONCLUSIVE by the rule, but not neutral: the 90% CI of
ours − published, [−15.7, −0.8], lies wholly below zero (point −7.0) and the
Mann-Whitney p is 0.064 — ours scores lower, mostly by partial runs (30.4,
54.9, 58.4, 59.6) rather than more total collapses (4 vs 1, p 0.34). Row 4c's
20 seeds pointed the other way (HL −1.3, MW p 0.66). The rule's one
extension (seeds 40–59, 40/arm) decides; no verdict is read from these 20.

## Row 4d — the extension, 40 seeds per arm (11-16 14:15)

**Verdict by the rule: MISSES** — the HL 90% CI of ours − published,
[−9.1, +0.6], does not fit inside ±5, and neither DIFFERS clause fires
(Mann-Whitney p 0.16, collapse Fisher p 0.57). The control held (1/20 frozen
inits ≥ 60; learners vs control p 2e−10).

**What the 40 seeds say beyond the verdict.** The two means are now 0.8 apart
(ours 72.6, published code 73.4). The 20-seed lean against ours (HL −7.0)
did not survive: on seeds 40–59 the published stack scored LOWER (65.0 vs
71.0), with three runs ending below zero (−8.4, −9.7, −4.6). The published
code, on this box, at 40 seeds, gives 73.4 ± 28.5 with 6/40 collapses —
against the store's 87.5 ± 4.2 at n 10, which is the number row 4's band was
built from. The rule's sizing note predicted a CI width of ~6.5 at n 40; the
realised width is 9.7, because the reference itself is that spread out.
Equivalence within ±5 cannot be shown between two learners each with sd
19–28 at 40 seeds unless the point estimate sits near zero (the width shrinks only as 1/√n: ~6.1 at n 100).

**Reading.** Row 4's published number does not reproduce with the published
code on this machine (rows 4c and 4d, 60 seeds of the public stack: 76.4
pooled mean, 7/60 collapses). Our build is not shown different from that
code (two tests, MW p 0.66 and 0.16), and not shown equivalent. No
threshold is moved; rows 4, 4b, 4d stay MISSES. No further extension exists
in the rule. Whether row 4 is closed as "the reference does not reproduce
its own number" or re-registered at a size that can show ±5 is the
Director's (INBOX, 11-16 14:15).

## Row J10-2 — Yu, Turk & Liu 2018 (arXiv:1801.08093): PIN and pre-flight, not yet run (11-16 17:55)

Published code, code only (no weights in it): `github.com/VincentYu68/SymmetryCurriculumLocomotion`
at `b50478f8eca673730e3ce1a5441b1948b31a5187` (2019-01-31, last commit; repo
has no licence file — code read and run, not redistributed). Entry point
`siggraph_script/run_script_humanoid.sh`: `mpirun -np 8 python3
baselines/baselines/siggraph/humanoid_walk.py --seed 0`, env
`DartHumanWalker-v1`, PPO1 + `MlpMirrorPolicy` + staged helping forces
(`train_mirror_sig`).

Stack it needs (from `prereq_install.sh`): DART **v6.3.0 built from source**
with the repo's patched `external/lcp.cpp`; `pydart2` built from source with
the repo's `pydart2_draw.cpp`; the bundled `dart-env` (a forked gym) and
bundled `baselines` (`tensorflow >= 1.0.0`, TF1 graph API → Python ≤ 3.7;
Kaggle's image is 3.11, so a separate 3.7 env inside the kernel); MPI
(`libopenmpi-dev`, `mpi4py`). Tested by the authors on Ubuntu 16.04/18.04.

Settings that already differ before we touch anything:
| setting | paper | released script |
|---|---|---|
| total samples | 10–30 M (500–1,500 iters × 20k) | `num_timesteps=5,000,000` |
| workers | 8 threads on a 16-vCPU EC2 node | `mpirun -np 8` |
| our kernel | — | Kaggle CPU: 4 cores, 12 h cap |

Pre-flight cost (arithmetic, not measured): paper 25–45 s/iteration on 8
threads; on 4 cores ≈ 50–90 s → the released 5 M (250 iters) ≈ 3.5–6.3 h,
the paper's 10–30 M ≈ 7–37 h. Plus the DART + pydart2 compile (unmeasured,
expect tens of minutes). So: the RELEASED script fits one kernel; the
paper's upper budget does not.

The rule (not yet committed — next slot, before any kernel): run the
released script unmodified at 5 M on one Kaggle CPU kernel, `-np 4`
(the one forced change, listed); the never-trains control = the same
policy evaluated at iteration 0 (and the ablation without helping forces
if time allows); accept iff the final policy walks UNAIDED (forces at 0)
for a full rollout and its symmetry index is within the published class
(≈ 0.01; Table 2 walk 0.0082). A build that will not compile on Kaggle is
recorded here as such, not worked around with a different simulator.

**11-16 18:30 — correction and RULE COMMITTED (`experiments/yu2018_walk.py`).**
The pre-flight above misread the budget: `num_timesteps=5,000,000` is passed
as `max_timesteps` to EACH curriculum stage's `learn` call
(`training_utils.py:65-80`), and a stage ends early once rollouts outlast
`assist_schedule[2][0]/dt` at > 0.5 × target speed (`pposgd_mirror.py:337-350`).
Forces start at 2,000 N and shrink ×0.75 per stage to < 5 N, then one unaided
stage: ≈ 22 stages. The released script's TOTAL is therefore what the
curriculum takes — expect the paper's 10–30 M — and "5 M fits one kernel" was
wrong; the full run needs chained kernels resumed from the authors' own
10-iteration saves (their commented hook, `training_utils.py:54-55`).
`-np 4` is dropped from the draft: it would halve the 20k batch. Ours keeps
8 ranks, oversubscribed on 4 cores (the batch kept, wall clock ≈ 2×).
Rule (full text in the module docstring): smoke first — one Kaggle CPU
kernel, build only + a 20-minute slice, recording every build stage and the
measured s/iteration; then kernels = ⌈1,500 × s/it ÷ 11 h⌉ registered in a
follow-up commit. Seeds 0, 1, 2 (one landing = PILOT). Primary accept: the
final policy, forces at 0, deterministic, ≥ 9 of 10 full 500-step rollouts
at ≥ 0.75 m/s mean forward speed (the script's own stage criterion); control
= the same seed's iteration-0 policy must fail it, else VOID. Symmetry index
reported only until its definition is pinned from the paper text. Pins:
DART v6.3.0 = `8090aaec`, pydart2 `113132ac`, TF 1.15.5 on Python 3.7 (uv),
FCL 0.5.0 from source as the one automatic fallback if the image's FCL 0.7
breaks DART 6.3.

## Row J3-4 — the seated in-house learner (`ppo-needs`) misses both rows (11-17 01:30, box)

Rule `3054fbe6` (committed before any cell), run on the box when all four Kaggle
CPU sessions were held (log `/data/jack-logs/j3_inhouse/box_ppo-needs.log`,
cells in `/data/jack-logs/j3_inhouse/cells/`). Both rows FAIL; both controls hold,
so the rows are valid. Predicted in the rule's Prior line.

- **Row 1 (Pendulum): learning, too slowly.** Every seed's three windows improve
  monotonically (e.g. s0 −1254 → −968 → −578; s4 −1229 → −1146 → −1120); the
  frozen twin stays at −1199…−1235. Best seed is 1.8× short of the band edge at
  3× the published budget. Reads as rate: 0.25 updates/decision, batch 16 from a
  256-row ring, γ 0.95 vs SAC's 1 update/step, batch 256, 1e6 replay, γ 0.99.
- **Row 3 (sparse MountainCar): never sampled the reward.** Across 5 seeds × 150
  training episodes the car reached the flag **0 times** (max training return
  −0.98). Returns climb from ≈ −23 to ≈ −1.1: the learner correctly learned to
  stop paying the action cost, the "do nothing" optimum. This is an exploration
  failure before it is a credit-assignment one — i.i.d. Gaussian noise (std
  0.5 → 0.1) does not produce the sustained push that reaches the flag; the
  published recipe uses gSDE (state-dependent, temporally correlated), and
  Brain B passed this row only with it (row 3b). γ 0.95 (~20-step horizon on a
  999-step task) would bite next, but it was never reached.
- **What would repair it (not run; a repaired arm returns under the same rule):**
  temporally correlated exploration (gSDE or pink noise at action choice) for
  row 3; replay/update ratio and γ for row 1. Those are the same settings the
  record says separate it from SAC, i.e. a repaired `ppo-needs` converges
  toward Brain B.

## Row J3-4b — `wm-latent` and `ppo-lp` miss both rows; wm-latent's row 1 is a near-miss (11-17, Kaggle CPU)

Rule `3054fbe6`, unchanged. Kernels jack-cpu-1794892152 (wm-latent) and
jack-cpu-1794896687 (ppo-lp); results `/data/jack-logs/j3_inhouse/{wm-latent,ppo-lp}.json`.
Both controls reach 0/5 on both rows, so all four readings are valid FAILs.

| arm | row 1 best 100-ep window per seed (band edge −314.7) | reached | row 3 flag reaches in training (5 × 150 eps) | reached |
|---|---|---|---|---|
| wm-latent | −396.7, −321.0, −397.2, −349.2, −358.0 | 0/5 | 0 | 0/5 |
| ppo-lp | −880.6, −867.2, −1058.8, −960.3, −823.8 | 0/5 | 0 | 0/5 |
| controls (both) | −1194…−1254 | 0/5 | 0 | 0/5 |

- **wm-latent row 1 learns; the 3× budget is the binding clause.** Its last 10
  training episodes average −231…−335 on all 5 seeds (in or at the band), but no
  aligned 100-episode window reaches −314.7 (best s1 −321.0, 6.3 short). The rule
  names no extension, so this is a miss (ruling 33). Cost: ~1,190 CPU-s a seed.
- **ppo-lp row 1 learns slowly** (first 10 eps ≈ −1,240 → last 10 −366…−958).
- **Row 3, both arms: the reward is never sampled** — 0 flag reaches in 1,500
  training episodes per arm, same as ppo-needs.
- **Correction to Row J3-4's reading.** The rise of row 3 training returns from
  ≈ −22 to ≈ −1.3 is shared by the NEVER-TRAINED controls (ctrl first 10 −22.0,
  last 10 −1.3 on every seed, every arm). It is the exploration-noise schedule
  shrinking the action, not learning. "The learner learned to stop paying the
  action cost" (J3-4, ppo-needs) is therefore not evidenced by these data; the
  exploration failure stands.
- Set aside with numbers (ruling 24). A repaired arm returns under this rule:
  row 3 needs temporally correlated exploration (gSDE / pink); wm-latent row 1
  needs only to be ~2× faster in sample terms.
