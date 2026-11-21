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
| N1-dyna | Dyna maze 6×9 (Sutton & Barto Ex. 8.1 / Fig. 8.2), γ 0.95, α 0.1, ε 0.1 | upstream `ShangtongZhang/reinforcement-learning-an-introduction@96bc2036` `chapter08/maze.py` UNMODIFIED, 30 runs (book; repo uses 10), seeded per run (rule `f6c3f074`, control re-registered `b4aa7f65`, `experiments/dyna_maze.py`, box, 146 CPU-s) | 50 episodes | book: ~25 episodes to ε-optimal at n = 0, ~5 at n = 5, 3 at n = 50; E = first episode the 30-run mean is ≤ 21 steps 5 in a row; accept E0 15–40, E5 3–8, E50 2–5, ordered | E0 = 28, E5 = 6, E50 = 3; episode-2 mean 743 / 150 / 39 steps; last-10 mean 17.7 / 17.1 / 17.0 (optimum 14) | 0–29 | **REPRODUCES** |
| N1-dyna | control: never trains (n 0, α 0), must never settle | 50 episodes | VOID unless E = ∞ and last-10 > 100 | E = ∞, last-10 mean 930 steps | 0–29 | holds |
| N6-rl2-task | Bernoulli bandits k 5, n 100 pulls, p_i ~ U[0,1] (Duan et al. 2016, arXiv:1611.02779 Table 1) | TASK CHECK, learner-free: random / greedy (Beta(1,1) counts) / Thompson, 1,000 instances (rule `74371e09`, `experiments/rl2_bandits.py`, box, 6 CPU-s) | 100 pulls | random 49.9, greedy 74.8, TS 74.7; accept \|ours − pub\| ≤ 3·SE·√2 | random 49.60, greedy 75.59, TS 74.80 (SE 0.43–0.53, tol ±1.8–2.3) | instances 10000–10999 | **REPRODUCES** (the task, not RL²) |
| N6-rl2-task | control: Thompson updated by an unrelated coin, must leave TS's band | 100 pulls | VOID unless outside 74.7 ± 2.20 | 49.43 | same | holds |
| N7-sc-task | Speech Commands v2, 35 words (Warden 2018, arXiv:1804.03209 §6) | TASK CHECK, learner-free: corpus counts + split disjointness (rule `b857ff29`, `experiments/speech_commands_task.py`, box, 0.8 CPU-s) | — | 105,829 utterances, 35 words, 2,618 speakers, exact; splits speaker-disjoint | 105,829 / 35 / 2,618; shared speakers 0/0/0; lists = TF `which_set` SHA-1 rule on 100% of files; split 84,843 / 9,981 / 11,005 | `/data/jack_corpora/speech_commands_v2` | **REPRODUCES** (the data, not an ear) |
| N7-sc-task | control: same files re-split at random (seed 0), same sizes; must share speakers | — | VOID unless shared > 0 | shared 2,115 / 2,255 / 2,307 | same | holds |
| N7-sc-testset | Speech Commands v0.02 official 12-class TEST archive (`speech_commands_test_set_v0.02.tar.gz`, the one `bcresnet@656f947a` main.py downloads) | TASK CHECK, learner-free: folders, the authors' test cap, speaker-disjointness from our train split (rule `ecc7c086`, `experiments/speech_commands_testset.py`, box, < 1 CPU-min; fetched 11-18 03:23, 112,563,277 B, sha256 cc2a00c1…acd51df) | — | 12 folders = authors' `label_dict`; `_silence_` = `_unknown_` = 408 (= `sample_per_cls_v2[2]`); 0 test speakers in train | 12 folders exact; 408 / 408; **4,890 clips**, 239 speakers, shared with train 0, with validation 0. Beside the rule: all 4,074 keyword + 408 unknown clips exist in our corpus and are in `testing_list.txt`; 3,775 + 382 byte-identical, the other 299 + 26 are short clips the archive zero-pads to 16,000 samples (the authors' `Padding` pads anyway) | `/data/jack_corpora/speech_commands_test_set_v0.02` | **REPRODUCES** (the test data, not an ear) |
| N7-sc-testset | control: 4,482 clips drawn at random (seed 0) from our train split through the same speaker check; must share speakers | — | VOID unless shared > 0 | shared 1,511 speakers | same | holds |
| N6-rl2 | RL² itself (TRPO, GRU 256, batch 250,000, ≤ 1,000 iterations; published 78.7) | not run: `experiments/rl2_cost_probe.py` floors one iteration at ≥ 138 CPU-s → ≥ 23 CPU-h a seed at 600 iterations (ruling 32: Kaggle or wait) | — | 78.7 | — | — | **NOT RUN** |
| N7-kws | BC-ResNet-1 keyword spotting, Speech Commands v2 12-class (Kim et al. 2021, arXiv:2106.04140; the paper's own table, read from its e-print 11-18: **v2 96.9 ± 0.30**, v1 96.6 ± 0.21 — PRIORS §7's "96.6–96.9%" was the two datasets, not a band; the rule must use 96.9 ± 0.30) | not run: `experiments/kws_cost_probe.py` (11-18, box, 1 thread) on the authors' model VENDORED VERBATIM (`Qualcomm-AI-research/bcresnet@656f947a` → `experiments/kws_bcresnet/`): 9,232 weights (published 9.2K, exact); 785 ms a training step at batch 100 + 59 ms log-mel → 200 epochs × 849 steps = 39.8 CPU-h train + 1.5 valid = **≈ 41 CPU-h a seed** — not an idle-slot row and not one Kaggle CPU session (PRIORS §7's "one CPU session each" is wrong for the published recipe); a free-GPU row. **CORRECTED 11-18:** the authors' `utils.py` caps `_unknown_`/`_silence_` at `sample_per_cls_v2` = [3077, 371, 408], so train is 36,923 clips (paper e-print: 36,923 / 4,445 / 4,890), not 84,843 → 370 steps/epoch → 200 × 370 × 844 ms = 17.4 CPU-h train + ~0.7 valid = **≈ 18 CPU-h a seed** (checked on our corpus: 30,769 train clips of the 10 words + 2 × 3,077 = 36,923 exactly; arithmetic from the stored step timings; probe constants fixed, not re-run). Still not an idle-slot row; at 4 threads it may fit one 12 h Kaggle CPU session — unmeasured. **THREADS 11-18 06:45** (`experiments/kws_thread_probe.py`, box, nice 19, 20 steps): 1 thread 741 ms wall (786 CPU) a step, 2 threads 453 ms wall (895 CPU) → 1.64× → 200 × 370 steps = 9.3 h wall train at 2 threads, + log-mel (~1.2 h at 1 thread, unthreaded here) + valid ≈ 10.5–11 h: at the edge of a 12 h cut ON THIS ARM BOX. 4 threads not run (customer cores); Kaggle is x86 — only a < 1 h Kaggle probe (queue rank 0) settles the fit. Published band per the paper: v2 96.9 ± 0.30 over 5 seeds (Table 1), 96.9 as a 10-seed mean (Table 2). Also: the official 12-class test set is a separate download — fetched and checked 11-18 03:30, row N7-sc-testset (4,890 clips, speaker-disjoint from train) | 200 epochs | v2 96.9 ± 0.30 | — | — | **NOT RUN** |
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

**Row 4 CLOSED (Director ruling 33, 11-17 06:50).** The PPO reference
does not reproduce its own stored number on this box (published code, 60
seeds over rows 4c+4d: pooled 76.4, 7/60 collapses, against the store's
87.5 ± 4.2 at n 10); ours matches the reference code as far as 40 seeds per
arm can show (72.6 vs 73.4; MW p 0.66 and 0.16; equivalence within ±5 not
shown, CI [−9.1, +0.6]). Recorded as a **miss of the stored number, not a
pass**. Not re-registered at 100 per arm (~3 Kaggle CPU-h for a row no seat
rests on). Nothing more is run on row 4; Brain B's seat rests on rows 1b
and 3b.

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

## Row J3-4c — `wm-dyn` misses both rows, further from the band than `wm-latent` (11-19, Kaggle CPU)

Rule `3054fbe6`, unchanged. Kernel jack-cpu-1795050581 (67 min); result
`/data/jack-logs/j3_inhouse/wm-dyn.json` (torch 2.11.0+cpu, gymnasium 1.1.1).
Controls reach 0/5 on both rows → both readings are valid FAILs.

| arm | row 1 best 100-ep window per seed (band edge −314.7) | reached | row 3 flag reaches (5 × 150 eps) | reached |
|---|---|---|---|---|
| wm-dyn | −674.9, −771.9, −547.0, −710.2, −874.2 | 0/5 | 0 | 0/5 |
| control | −1192.1…−1255.3 | 0/5 | 0 | 0/5 |

- **Row 1 learns, slowly:** first 10 → last 10 training episodes −1307→−1111, −1177→−737,
  −1230→−646, −1342→−928, −1482→−1132 (300 eps, ~640 CPU-s a seed). Best seed −547.0 is
  232 short of the band; wm-latent's worst seed (−397.2) beats wm-dyn's best. Ranking so far on
  row 1: wm-latent > wm-dyn > ppo-lp > ppo-needs; none reaches.
- **Row 3: the reward is never sampled** (0 flag reaches), as for every in-house arm — the shared
  learner's white action noise; the named cure (correlated noise, §15) is J3 (5)'s repaired entry.
- Remaining under (4): `dreamer-xs` (jack-cpu-1795054996, launched 11-19 02:23), then `wm-latent+ring`.

## Row J3-4d — `dreamer-xs` misses both rows; the first in-house seed to reach row 1's band (11-19, Kaggle CPU)

Rule `3054fbe6`, unchanged. Kernel jack-cpu-1795054996 (home 03:45); result
`/data/jack-logs/j3_inhouse/dreamer-xs.json`. Controls reach 0/5 on both rows → valid FAILs
(PASS needs ≥ 3 of 5 arm seeds).

| arm | row 1 best 100-ep window per seed (band edge −314.7) | reached | row 3 flag reaches (5 × 150 eps) | reached |
|---|---|---|---|---|
| dreamer-xs | −392.7, −398.9, −340.0, **−286.2**, −378.5 | 1/5 | 0 | 0/5 |
| control | −1195.3…−1258.5 | 0/5 | 0 | 0/5 |

- **Row 1: seed 3 reaches (−286.2)** — the first in-house seed under this rule to do so; 2 more
  were needed. Last 10 training eps per seed −429.8, −311.5, −283.7, −277.0, −228.0. Cost ~844
  CPU-s a seed (wm-latent 1,188), row 3 ~2,022 (wm-latent 2,783).
- **J3 (6), dreamer-xs beside wm-latent: equal within seed noise.** Mean best window −359.2 vs
  −364.4 (wm-latent −396.7, −321.0, −397.2, −349.2, −358.0); 1/5 vs 0/5 reaches. By (6)'s own
  reading, wm-latent's distance from PPO was not its home-made losses — the shared learner at this
  network size is what both are. (`wm-efe` = `dreamer-xs` on these rows, ruling 44: same row.)
- **Correction to Row J3-4b.** Recomputed from `wm-latent.json`: wm-latent seed 0's last 10 training
  eps average −839.6 (it collapsed late: −2.8 … −1,365 alternating), not "−231…−335 on all 5 seeds";
  seeds 1–4 are −334.9, −231.5, −274.0, −247.1. The near-miss reading for row 1 stands on 4 of 5.
- **Row 3: 0 flag reaches**, as every in-house arm (white action noise; J3 (5) is the cure).
- Remaining under (4): `wm-latent+ring`.

## Row J3-4e — `wm-latent+ring` misses both rows; the ring adds nothing measurable (11-19, Kaggle CPU)

Rule `3054fbe6`, unchanged. Kernel jack-cpu-1795090393 at `71324fae` (home 14:45, 9,115 s; receipt
in `gpu_submissions.jsonl` — the launch log names the killed duplicate `…090752`, whose watcher
never wrote); result `/data/jack-logs/j3_inhouse/wm-latent+ring.json`. Controls reach 0/5 on both
rows → valid FAILs (PASS needs ≥ 3 of 5 arm seeds).

| arm | row 1 best 100-ep window per seed (band edge −314.7) | reached | row 3 flag reaches (5 × 150 eps) | reached |
|---|---|---|---|---|
| wm-latent+ring | −425.7, −453.0, −347.3, **−263.2**, −353.7 | 1/5 | 0 | 0/5 |
| control | −1199.2…−1252.3 | 0/5 | 0 | 0/5 |

- **Row 1: seed 3 reaches (−263.2)**; 2 more were needed. Learns on all 5 (first 10 eps
  −1138…−1479 → last 10 −246.3, −481.7, −354.7, −207.7, −188.5). Mean best window −368.6 vs
  wm-latent −364.4 and dreamer-xs −359.2: **the ring adds nothing measurable** — the third arm on
  the same shared learner landing at the same place. Cost ~1,560 CPU-s a seed (wm-latent 1,188).
- **Row 3: 0 flag reaches** (best training return −0.99), as every in-house arm.
- **J3 (4) is complete:** all plain arms have rows, all FAIL with controls holding. J3 (5)
  `wm-latent@repair` launched 11-19 15:24 (jack-cpu-1795101867).

## Row J3-5 — the repaired entry `wm-latent@repair`: row 3 PASSES (the first in-house row reproduced), row 1 collapses (11-19, Kaggle CPU)

Rule `3054fbe6` + the repair block `94406999` (noise direction held 32 decisions, GAE γ 0.99, fresh
seeds 5–9, same verdict clause), unchanged. Kernel jack-cpu-1795101867 at `66b1920c` (home 17:06,
6,113 s; receipt in `gpu_submissions.jsonl`, `74c883e5`); result
`/data/jack-logs/j3_inhouse/wm-latent@repair.json` (torch 2.11.0+cpu, gymnasium 1.1.1).

| arm | row 1 best 100-ep window per seed (edge −314.7) | reached | row 3 best deterministic eval per seed (> 90) | reached |
|---|---|---|---|---|
| wm-latent@repair | −1279.0, −1290.2, −1187.7, −1288.4, −1216.5 | 0/5 → **FAIL** | **95.8, 95.3, 92.6**, 89.4, **94.4** | 4/5 → **PASS** |
| never-trains control | −1193.4…−1239.1 | 0/5 | −0.005…−0.00004 | 0/5 |

- **Row 3 PASS — the first row any in-house arm has reproduced.** 4 of 5 seeds above 90 at the
  **50k** checkpoint, i.e. inside the published SAC budget (§5: 50k), not only the 3× allowance;
  seed 8 is 88.7 at 50k and 89.4 at 150k (a miss by 0.6, counted as a miss). Every plain arm read 0
  flag reaches. Not monotone: at 100k three seeds dipped (59.3, −56.1, −58.5) and recovered by 150k.
  **Disclosed:** the held noise alone finds the flag — the never-trains control's TRAINING episodes
  reach it (first-10 mean 54–79) — but its deterministic eval (the frozen mean) reads ~0, so the
  PASS is the learned policy, as the rule measures. The repair is the exploration (§15 candidate 1).
- **Row 1 FAIL, and worse than the unrepaired arm:** no learning at all — first-10 → last-10
  episode means −1310 → −1192, −1299 → −1449, −1346 → −1126, −1272 → −1476, −1371 → −1306, inside
  the control's range (best windows −1188…−1290 vs control −1193…−1239). Plain `wm-latent` reached
  a best window of −364 mean on seeds 0–4 and learned −1300 → −650. The repair traded row 1 for row 3.
- **Which change broke row 1 is not measured.** Priced candidate (not a finding): value targets are
  in raw task units (docstring "reward raw"); at γ 0.99 Pendulum's discounted return reaches ~−600
  vs ~−120 at 0.95, so the value-loss term is ~25× larger in the core's shared loss. The other
  candidate: noise held 32 decisions (16% of a 200-step episode) on a task that needs fine torque
  corrections. One kernel with `hold-only` and `gamma-only` cells on row 1 separates them.
- **Addendum 11-20 12:20 — measured: the HELD NOISE broke row 1, not γ.** Diagnostic J3 (5′)
  (`a377e014`, reading fixed in `inhouse_rows.py` before any cell; kernel jack-cpu-1795170409, home
  ~11:58, 5,519 s; result `/data/jack-logs/j3_inhouse/wm-latent@gamma-only__wm-latent@hold-only__wm-latent@plain59.json`;
  `inhouse_rows diag` reads **"HELD NOISE kills row 1"**). Row 1 only, seeds 5–9, learns iff best window ≥ −800:
  `@plain59` (neither change) −520.7, −390.3, −364.1, −354.4, −369.4 → KEEPS 5/5 (seed set is not the cause; not VOID);
  `@gamma-only` (γ 0.99, white noise) −567.8, −283.7, −598.9, −758.9, −736.8 → KEEPS 5/5;
  `@hold-only` (held 32, γ 0.95) −1199.1, −1219.6, −1265.3, −1245.0, −1220.2 → KILLS 0/5 (≈ never-trains control).
  The priced γ/value-scale suspect above is **refuted**: γ 0.99 alone still learns row 1 (mean −549 vs plain −400,
  somewhat worse). Held noise is what found row 3's flag and what erased row 1's learning. Diagnostic, reported-only:
  no verdict row moves; which repair goes to the other arms (e.g. hold on sparse-reward rows only, or a shorter hold)
  is the Director's split ruling, now with its cause measured.

## Row J3-6 — the one-setting entry `wm-latent@hold8`: row 3 PASSES 3/5, row 1 FAILS 0/5 → J3 CLOSES (11-20, Kaggle CPU)

Rule written before any cell `5ec9879e` (ruling 59: ONE setting for both rows — noise held 8 decisions,
GAE γ 0.99, everything else as `@repair`; fresh seeds 10–14; the same verdict clauses; never-trains
control must fail; no extension). Kernel jack-cpu-1795198133 at `22d82a05` (launched 18:08, home 19:58,
6,594 s; receipt in `gpu_submissions.jsonl`); result `/data/jack-logs/j3_inhouse/wm-latent@hold8.json`
(python 3.13.18, torch 2.11.0+cpu, gymnasium 1.1.1).

| arm | row 1 best 100-ep window per seed (edge −314.7) | reached | row 3 best deterministic eval per seed (> 90) | reached |
|---|---|---|---|---|
| wm-latent@hold8 | −1097.5, −992.6, −1000.9, −1063.1, −999.6 | 0/5 → **FAIL** | **95.6**, 88.6, 82.9, **94.4**, **91.5** | 3/5 → **PASS** |
| never-trains control | −1171.0…−1225.1 | 0/5 | −0.031…−0.0001 | 0/5 |

- **Row 1: the shorter hold gives back a little learning, nowhere near the band.** First-10 → last-10
  episode means −1253 → −1030, −1435 → −1345, −1148 → −990, −1316 → −1316, −1300 → −1132 (seeds 10–14).
  Best windows sit ~150–200 above the control and ~200 above `@repair` (the same γ 0.99 with hold 32:
  −1188…−1290), but ~600 below `@gamma-only` (γ 0.99, white noise: −284…−759) and ~700 below the edge.
  Held noise at 8 still kills most of dense row 1's learning; shortening the hold 32 → 8 helps a little
  (different seed sets, 5 each — a direction, not a measured dose curve).
- **Row 3 still PASSES but weaker than hold 32:** 3/5 vs `@repair`'s 4/5; seed 11 peaks 88.6, seed 12
  82.9 at 50k then decays to 67–70; seed 14 reaches only at 100k (−18.0 at 50k).
- **The finding kept (ruling 59):** on this on-policy AWR core, temporally held exploration noise is what
  finds the sparse MountainCar flag and what erases the dense Pendulum learning; neither hold tried (8, 32)
  passes both rows. No setting of this learner reproduces both published rows.
- **J3 CLOSES (ruling 59, no third entry).** Every in-house arm is set aside with its numbers on record
  (J3-4 `ppo-needs`; J3-4b `wm-latent`, `ppo-lp`; J3-4c `wm-dyn`; J3-4d `dreamer-xs`; J3-4e
  `wm-latent+ring`; J3-5 `@repair`; J3-6 `@hold8`). Per PLAN they may return only if repaired — the
  measured repair target is an exploration scheme that does not break dense-reward control (e.g. one
  learned or annealed rather than a fixed hold), entered through `docs/FORGE_POOL.md` with its prior.

## Row J10-1 — SAC, the RL Zoo's Humanoid entry on Humanoid-v4: **FAIL** (decided at 2 of 3 seeds; s2 still running) (11-21 00:12, Kaggle CPU)

Rule frozen in `experiments/humanoid_sac_row.py` (`23e375be` era, before seed 0): PASS = at 2M R ≥ 5,950 AND
M ≥ 12 m/1,000 steps on ≥ 2 of 3 seeds; 1M median in [3,784, 5,866]; late-start control < 500 else VOID.
Segments of 500k chained through Kaggle kernel sources; results `/data/jack-logs/j10/sac_row/s<seed>_seg<n>.json`.

| seed | 1M R | 1M M | 1M fell | 2M R | 2M M | 2M fell | reading |
|---|---|---|---|---|---|---|---|
| 0 | 4,065 | 3.47 | 3/10 | **5,834** | 11.91 | 0/10 | misses by 116 R and 0.09 m |
| 1 | 5,294 | 5.08 | 0/10 | **585** | 3.28 | **10/10** | COLLAPSED (`jack-cpu-1795205296`) |
| 2 | 3,650 | 7.25 | 6/10 | — | — | — | seg2 running `jack-cpu-1795212583` |
| control (late start, 50k) | | | | R 68.0, M 0.03 | | | holds (< 500) → run valid |

- **1M check: IN BAND** — median 4,065 inside [3,784, 5,866]. Our build tracks the published 1M curve.
- **2M: FAIL.** Seeds 0 and 1 both miss, so 2 of 3 can no longer walk whatever seed 2 does. STANDS is also out (it
  needs R ≥ 5,950 on 2 seeds). Seed 2's 2M reading is recorded when it lands, and cannot change this one.
- **What the miss is:** seed 0 is a near-miss of the published 6,232 ± 280 (one model, Humanoid-v3). Seed 1 learned a
  gait and then lost it between 1M and 2M (5,294 → 585, every eval episode falls). There was no eval at 1.5M, so it is
  not known whether the loss came in segment 2 or segment 3. The resume path did not break seed 0, so the resume is not
  shown to be the cause, and it is not ruled out either.
- **Cheapest next check (not run):** evaluate seed 1's 1.5M checkpoint (output of `s1_seg2`'s kernel) with the frozen
  10-episode eval. That shows which segment the collapse happened in. Late-run collapse of SAC on Humanoid is a known
  instability. It was not in the rule's prior (§16 quotes one model at 2M).

**Addendum 11-21 01:15 — the cheapest check, RUN (descriptive; the row's verdict does not move).** `1fdd3d58`
`experiments/j10_ckpt_curve.py`: each segment kernel's `model.zip` (20 MB) read on the box with the rule's own eval.
**Validity:** the pre-set gate (box must reproduce the kernels' own 1M/2M evals within 1% R) **FAILED on 4 of 5**
(1.7, 4.8, 4.9, 13%, both signs). A check added afterwards (post-hoc): 12 of 17 early falls land on the IDENTICAL step,
so the weights load exactly, and the gap is late-episode MuJoCo chaos between x86 and ARM. Read these as box numbers:

| seed | 0.5M R / M / fell | 1M | 1.5M | 2M | α 0.5→2M |
|---|---|---|---|---|---|
| 0 | 5,282 / 4.7 / 0 | 3,995 / 3.2 / 4 | 5,580 / 8.7 / 0 | 5,552 / 11.0 / 1 | .022 .013 .008 .009 |
| 1 | 4,501 / 3.9 / 2 | 5,035 / 4.9 / 1 | **5,621 / 9.2 / 0** | **581 / 3.6 / 10** | .022 .013 .009 **.024** |
| 2 | 4,577 / 5.9 / 5 | 4,136 / 7.5 / 5 | **6,039 / 14.4 / 0** | seg3 `jack-cpu-1795223814` | .025 .017 .011 |

- **Seed 1 collapsed inside the LAST segment (1.5M→2M), from its best point** (0/10 fell at 1.5M). It was a resumed
  segment; 7 other resumes did not break, so the resume is still not shown to be the cause. In that segment α rose 2.6×
  (the auto-entropy target pushed back as the policy narrowed), and the critic's Q at the start state rose 514 → 547,
  while the return actually earned fell to ~291 (flat-rate discount). That is the same overestimate-while-losing shape
  E2c showed. It is a lead, not a cause.
- **The eval's own noise is about the size of seed 0's miss:** the same weights read 5,834 on Kaggle and 5,552 on the box,
  because one episode falls at step 633 on the box only. The 116-R miss sits inside cross-CPU noise of the 10-episode
  eval. The rule is frozen, so this is recorded, not used.
- Seed 2 at 1.5M clears both 2M thresholds (not the rule's mark). Its seg3 is running, to 2M as the order says.

**Addendum 11-21 04:45 — seed 2 at 2M home, and the first chaos-or-recipe fork (record only; the FAIL stands).**
- **Seed 2, 2M** (`jack-cpu-1795223814`, `s2_seg3.json`, Kaggle CPU, the rule's eval): **R 6,490, M 20.07, 0/10 fell**
  (worst episode 6,468). By the rule's own per-seed marks (R ≥ 5,950 AND M ≥ 12) seed 2 passes — so the row is
  **1 of 3** (needs 2): s0 5,834/11.91 near-miss, s1 585 collapsed, s2 6,490/20.07. Verdict unchanged: FAIL.
- **Fork f1** (`experiments/j10_fork.py`, rule in docstring `30447a6f` before either fork; `jack-cpu-1795225164`,
  `fork/f1.json`): seed 1 resumed from its 1.5M model + replay, reseeded (model, JAX key, env), to 2M →
  **R 5,863, M 12.19, 0/10 fell = KEPT** (and above both row thresholds). Seed 1's own continuation lost the walk
  (585, 10/10 fell); one other draw of the same segment keeps it. Fork f2 (`jack-cpu-1795225181`) RUNNING. The
  reading is f1 KEPT + f2: KEPT → CHAOS, otherwise MIXED. RECIPE is now impossible.

**Addendum 11-21 05:15 — fork f2 home: the reading is CHAOS (record only; the FAIL stands).**
- **Fork f2** (`jack-cpu-1795225181`, `fork/f2.json`, reseeded `set_random_seed`, `key`, `env`): seed 1 from 1.5M to
  2M → **R 5,303, M 14.07, 2/10 fell = KEPT** (on the edge: KEPT allows ≤ 2 falls; falls at 259 and 560 steps, the
  other 8 episodes R 5,992–6,071). By the row's per-seed marks f2 would miss (R < 5,950); f1 would miss by 87 R.
- `j10_fork.reading` over f1 + f2: **{1: KEPT, 2: KEPT} → CHAOS**. Seed 1's collapse (585, 10/10 fell) was one
  unlucky continuation, not the recipe driving this policy off its walk. Two forks say nothing about a RATE of
  collapse (the rule's limit). So Row J10-1's 1-of-3 reading is noise-limited: two of three seeds end at or
  near the PASS line and one fell, and three seeds cannot tell this recipe's PASS rate from a coin. The FAIL stands
  (a miss is a miss). Next is for the Director: §21's cures (LN critic, resets, TQC) or more seeds of the frozen recipe.

## Row B-5 — DMC cartpole-swingup, 50k decisions: **lib MISSES** at 5 seeds; ours in band but not counted; the one extension is due (11-21 00:12, Kaggle CPU)

Rule `experiments/brain_b_dmc.py` (committed 11-18 before any seed, `0bf84989`): band 837.9–868.1 (853.0 ± 9.2, k = 5);
`ours` counts only if `lib` reproduces; VOID if any ctrl seed ≥ 837.9. Kernel `jack-cpu-1795209812` (launched 11-20
21:23, home in 8,790 s); result `/data/jack-logs/brain_b/row5_dmc.json`.

| arm | per seed (0–4) | mean | reading |
|---|---|---|---|
| lib (SBX SAC, SimBa recipe) | 808.1, 832.4, 843.3, **643.6**, 853.5 | **796.2** | below the band → **lib MISSES** |
| ours (lib + layer_norm, Brain B) | 863.0, 872.0, 875.0, 856.0, 871.9 | 867.6 | inside the band, but not counted (lib missed) |
| ctrl (never trains, seeds 0–2) | 69.2, 162.4, 73.3 | 101.6 | fails as it must → run valid |

- Without seed 3, lib's mean is 834.3, still just under 837.9. Every lib seed scores at or below ours, and the
  10-episode evals are almost deterministic (spread < 3 within a seed), so these are differences between seeds.
- **The rule's ONE extension applies** ("a task that misses at 5 seeds runs seeds 5..14 and is judged once more at
  k = 15", band 841.7–864.3). Not yet built: `kernel 5` runs seeds 0–4 only. Row 6 waits, because the rule says a
  miss on row 5 is fixed first, not stacked.

**Addendum 11-21 05:15 — the one extension home: lib MISSES at k = 15 (final; no further extension).**
Kernel `jack-cpu-1795219999` (`kernel 5 ext`, Kaggle CPU, 16,541 s), `/data/jack-logs/brain_b/row5_dmc_ext.json`,
judged by `judge_ext` over seeds 0–14, band 841.7–864.3.

| arm | seeds 5–14 | mean 5–14 | mean 0–14 | reading |
|---|---|---|---|---|
| lib (SBX SAC, SimBa recipe) | 857.8, 856.1, 846.5, 872.2, 849.3, 834.2, 849.3, 853.5, 858.7, 849.1 | 852.7 | **833.8** | below 841.7 → **lib MISSES** |
| ours (lib + layer_norm, Brain B) | 857.9, 864.9, 870.7, 864.4, 855.6, 852.1, 877.5, 817.9, 867.5, 872.2 | 860.1 | 862.6 | in band, not counted (lib missed) |

- The fresh 10 lib seeds sit inside the band (852.7 ± 9.8); the k = 15 mean misses because seed 3 (643.6) drags it
  21 points down. Without seed 3: 847.4 (in band). Median 849.3. Stated, NOT used — the rule judges the mean.
- So the miss is one outlier seed out of 15, not a shifted curve; our build of the library is close to the
  published 853.0. That is a description; the row is a named MISS and stays one. Row 6 still waits (the rule fixes
  a miss on row 5 first). What a fix would be (the one outlier: an apparatus question about seed 3's cell) is for the Director.

## Row A1 — brain A (sheeprl `dreamer_v3_5`, size 1m) on DMC cartpole_balance: **PASS** at 3 seeds (11-21 02:30, Colab T4)

Rule `experiments/brain_a/rows.py` (frozen `286d6451` before any seed; control corrected to no gradient step before
any seed ran). Kaggle refused the first push at pip (torch pin, fixed `01561922`); the job fell back to Colab
`ladder-1795219944` (1.90 GPU-h, result recovered from stdout). File `/data/jack-logs/brain_a/rows/rows_A1.json`;
scored by `python -m experiments.brain_a.rows score A1`. Unit check passed on all 4 runs (first episode at
policy_step 2,000 for all 4 envs).

| point (env steps) | band | s0 | s1 | s2 | mean | never-trains control | reading |
|---|---|---|---|---|---|---|---|
| 100K | 690–898 | 826.3 | 843.5 | 677.0 | **782.3** | 312.5 | in band |
| 200K | 932–1000 | 973.8 | 994.1 | 984.3 | **984.1** | 296.9 | in band |

- **Verdict PASS; no extension due.** The control sits far below both bands → the row is valid. This is brain A's
  first reproduced published row (and the first reproduced row of any candidate on a DreamerV3 curve).
- Seed 2 alone is under the 100K band (677 < 690); the rule judges the 3-seed mean, which is inside. Stated, not used.
- Not readable: §2's step-0 check. The first logged scalars are at policy step 5,000 (after training began:
  reward_loss 3.29–3.32, value 2.56–2.85), not step 0. The box smoke (11-17) read 5.541 at step 1,100 — the
  only step-0 reading we hold.
- Settings that differ from the published protocol stay as the rule lists them (v2 port, size 1m not S, compile
  on). Wall per seed on the T4: 1,970–2,665 s for 200K env steps.

## Row A2 — brain A on DMC cartpole_swingup: **MISS at 100K, in band at 200K** → the rule's one extension (seeds 3–4) is due (11-21 02:40, Kaggle T4)

Same rule (`286d6451`). Kaggle `jack-ladder-1795220186` (A2b, after the torch re-pin `01561922`), COMPLETE; the
watcher died AFTER the job finished — `kaggle kernels output` of the whole 1.2 GB output dir hit gpu.py's 300 s
timeout (INBOX) — but `brain_a_rows_A2.json` had landed whole (4 runs, rc 0, none cut, 200 episodes each); copied
to `/data/jack-logs/brain_a/rows/rows_A2.json` and scored by `rows score A2`.

| point (env steps) | band | s0 | s1 | s2 | mean | never-trains control | reading |
|---|---|---|---|---|---|---|---|
| 100K | 438–518 | 415.2 | 466.5 | 328.6 | **403.4** | 51.9 | **MISS** (35 under the edge) |
| 200K | 740–816 | 758.6 | 800.3 | 760.3 | **773.0** | 57.2 | in band |

- Control valid at both points. Verdict MISS at 3 seeds → **seeds 3 and 4 run fresh** (`rows kernel A2 --ext`) and
  the row is judged once at 5 seeds, same bands. Nothing after that.
- Two readings, neither explained away (the rule's own words): a broken build, or size 1m (~0.64M params, no
  published score) being slower early than size S. Seen, not a test: by 200K all three seeds are in band, so the
  build learns the task; the miss is the EARLY point, and seed 2 carries most of it (328.6).

### Row A2 at 5 seeds — **named MISS at 100K, in band at 200K** (11-21 04:30, Kaggle T4 `jack-ladder-1795228778`)

The rule's one extension, judged once (`rows score A2` over `rows_A2.json` + `rows_A2_ext.json`; all runs rc 0, none cut;
control from the base file). Nothing runs after this: a miss at 5 seeds is a named miss.

| point | band | s0 | s1 | s2 | s3 | s4 | mean | control | reading |
|---|---|---|---|---|---|---|---|---|---|
| 100K | 438–518 | 415.2 | 466.5 | 328.6 | 349.0 | 407.2 | **393.3** | 51.9 | **MISS** (45 under the edge) |
| 200K | 740–816 | 758.6 | 800.3 | 760.3 | 822.8 | 799.1 | **788.2** | 57.2 | in band |

- The fresh seeds did NOT pull the mean up (3-seed 403.4 → 5-seed 393.3): seed 2 was not the outlier, the EARLY point
  is. 1 of 5 seeds reaches the band at 100K; 5 of 5 are at or above its lower edge by 200K.
- Both readings the rule named stay open: a broken build, or size `1m` (~0.64M params, no published score) learning
  slower early than the published size S. Seen, not a test: every row that passes (A1, A3) and A2 at 200K say the build
  learns; what is short is speed at the first point. The test that separates them is the same row at size S (published
  recipe) — a fix of A, owed before A's pilot (ruling 62: "If A misses a row: fix A, no pilot").
- **Brain A: 2 of 3 rows PASS (A1, A3); A2 a named miss at 100K.**

## Row A3 — brain A on DMC ball_in_cup_catch: **PASS** at 3 seeds (11-21 02:40, Kaggle T4)

Kaggle `jack-ladder-1795220723`, COMPLETE; result file read from the watcher's download dir (4 runs rc 0, none cut).

| point | band | s0 | s1 | s2 | mean | control | reading |
|---|---|---|---|---|---|---|---|
| 200K | 931–967 | 960.0 | 955.8 | 952.4 | **956.1** | 8.0 | in band |

- **Brain A: 2 of 3 rows PASS (A1, A3), A2 owes its extension.**

## A's smoke in his world (J14 b) — **DONE (1)–(4) all hold** (11-21 03:30, Kaggle T4)

Kaggle `jack-ladder-1795227930`, COMPLETE, watcher exited clean (`/data/jack-logs/brain_a/in_world/inworld.json`; GPU receipt 1,485 s). Rule: `experiments/brain_a/in_world.py` docstring, written 11-18 before any run. Flat keys, body motor, seed 0, Tesla T4, torch re-pinned 2.6.0+cu126 (`A_TORCH_REPIN 0`).

| clause | reading |
|---|---|
| (1) STEPS | rc 0, last logged step 20,000 of 20,000, not cut |
| (2) LEARNS | `Loss/world_model_loss` logged at 19 steps: 26.19 @2k → 5.27 @20k; observation 17.14 → 1.80; reward 5.26 → 0.43 |
| (3) NO NaN | every Loss/* finite; 4 episode returns finite |
| (4) SPEED | **18.7 decisions/s** wall (20k / 1,069 s, incl. compile); sheeprl's own sps_env 186, sps_train 29 at replay_ratio 0.5 |

- Not read (the rule forbids it at 20k): meals, lives, return. The 4 finished lives (−97, −97, −99, +100) are a smoke, not A eating.
- Seen, descriptive: `Loss/state_loss` (KL) rises 1.14 @3k → 3.02 @20k while reconstruction falls — the posterior using more of the senses; no claim.
- Cost for the pilot draft: 300k decisions at 18.7 dec/s ≈ 4.5 T4-h a run (an upper figure: compile time amortises), not ruling 62's ≈ 1.6 h. sheeprl reports sps_train 29 vs sps_env 186: the slow part is training, not his world.

### A's smoke, per-sense keys (Next-up 1, §18 item 2) — **(1)–(3) hold; (4) not comparable** (11-21 05:45)

Colab GPU session `ladder-1795237765` (Kaggle refused: 2-session cap held by ear + needs-hidden), rc 0, receipt 1,433 s; `/data/jack-logs/brain_a/in_world/inworld_per_sense.json`. Same rule as the flat smoke (`in_world.py` docstring), same seed 0, body motor, 20k decisions, compile on; only change: `keys=per_sense` (5 encoder AND decoder mlp keys: vision, audio, touch, proprio, needs). GPU model not recorded by the job.

| clause | flat (Kaggle T4) | per-sense (Colab) |
|---|---|---|
| (1) STEPS | 20,000/20,000, rc 0 | 20,000/20,000, rc 0, not cut |
| (2) LEARNS | wm loss 26.19 → 5.27; obs 17.14 → 1.80; reward 5.26 → 0.43 | wm loss 25.38 → 5.30; obs 16.47 → 1.85; reward 5.26 → 0.36 |
| (3) NO NaN | all Loss/* finite | all 8 Loss/* finite at 19 logged steps; 2 lives finite (−98.8, +2.5) |
| (4) SPEED | 18.7 dec/s; sps_env 186, sps_train 29 | 14.4 dec/s; sps_env 104, sps_train 29 |

- Speed is NOT read as a per-sense cost: different host (Colab vs Kaggle), and the gap is all in env interaction (104 vs 186 sps) while training is identical (29 vs 29) — a host difference, not the decoder heads.
- Descriptive: per-sense wm loss plateaus 5.5–5.8 from 6k to 17k before falling to 5.30; KL (state_loss) 3.38 → 3.06 (flat rose 1.14 → 3.02). No claim at 20k; the frozen A1–A3 rows untouched. Next-up 1 is DONE: the per-sense path runs end to end on a GPU. Whether one key per sense HELPS is a race (flat vs per-sense) for a later rule, not this smoke.

## Row NH-1 — Dulberg's homeostatic grid (needs-hidden row, §18 item 4): **R1 FAIL, R2 FAIL** — learning shown, ~10–20× off the published Δ (11-21 06:14, Kaggle T4 `jack-ladder-1795234980`)

Rule frozen `5464c5b8` (`experiments/needs_grid_row.py` docstring), 6 cells × seeds 0–9, one kernel, ~1.9 T4-h (DQN 86 s/run, GmQ 185–194 s/run). `needs_grid_row score` over `/data/jack-logs/needs_grid_row/needs_grid_frozen5000_dqn5000_gmq5000_hidden5000_dqn1_gmq1.json`, verdict SCORED.

| cell | published median [IQR] (figure reading) | ours median Δ (10 seeds) | per-seed Δ |
|---|---|---|---|
| frozen@5000 (CONTROL, must be > 5.0) | — | **384.4** → row valid | 155–685 |
| dqn@5000 (R1) | 0.95 [0.75, 1.35] → band [0.65, 1.45] | **16.8** — MISS | 3.9 3.9 6.4 9.3 16.4 17.2 18.7 25.0 28.1 99.7 |
| gmq@5000 (R1) | 0.5 [0.42, 0.6] → band [0.32, 0.70] | **8.2** — MISS | 2.0 2.5 2.5 3.1 6.1 10.2 16.8 18.6 42.3 61.8 |
| dqn@1 (R2, must be ≥ 1.8) | 3.2 [1.9, 6.0] | **40.8** — holds | 11–116 |
| gmq@1 (R2, must be ≤ 0.80) | 0.6 [0.5, 0.7] | **5.4** — MISS | 2.9–19.0 |
| hidden@5000 (R3, TEST) | no arm | **112.9**; > dqn@5000 on 9/10 seeds | 93–167 |

- **R1 FAIL, R2 FAIL** (gmq@1 5.4 > 0.80). R3 computes MET (112.9 > 1.45, 9/10 pairs) but per the rule is **reported, not read**: R1 failed. No extension seeds; no band moves.
- What DID reproduce (direction only, not a pass): every learner beats the never-trains control 23–70×; GmQ < DQN at both annealing lengths (8.2 vs 16.8; 5.4 vs 40.8); 1-step annealing hurts DQN (40.8 vs 16.8) and not GmQ (5.4 vs 8.2) — the paper's Fig 3a pattern. The MAGNITUDE is off by 10–20× in every learner cell.
- Why, descriptive (from `h_final`, the only per-run state the job kept): the deviation is mostly STARVATION, not overshoot — dqn@5000 has 21 of 40 final stats below 4 and 12 above 6; several sit at −119.5 = 0.5 − 0.004 × 30,000, i.e. that resource was never eaten in the whole life. Only 1 of 10 dqn@5000 runs ends with Σ|h*−h| < 4 (GmQ@1: 5 of 10). The agents keep one or two corners and abandon the rest.
- Candidate causes in OUR apparatus (untested, each one a setting listed "ours" in the rule's table): (a) the Gaussian is the pdf with Σ = I in CELL units, so the 5×5 centre reads < 1e-5 and γ = 0.5 sees ~2 steps — a corner 6 cells away is invisible to both the view and the value; the paper's grid coordinates may not be cells; (b) unbounded stats (−119…+519) feed raw into the net and into an n=4 drive — huge rewards; the paper may clip; (c) our Δ reading of the figure (per-stat vs summed). Next step: read the PNAS 2023 text / code for (a)–(c) before any rerun; rerun is a re-registered rule, not this one.

### NH-1 addendum — the published resource map, read off `Figures/res_map.png` (e-print 2204.06608, 11-21 06:25, descriptive)

- The text (`rldm.tex`) settles (c): Δ is SUMMED over the 4 stats per step (its eq. `delta`), exactly ours — the miss is not a reading of the score.
- (a) gains evidence: in all four published maps the BRIGHTEST cell (viridis yellow) is the walkable CORNER cell, inside the dark wall ring. Ours puts the Gaussian's peak ON the wall cell (10, 10), so our best walkable cell (9, 9) gets e^−1/(2π) = 0.059 — 37 % of the peak — and its edge neighbour 8 % of the peak.
- Pixel colours along res 1's bottom edge from the corner (cells ≈ 24.5 px; background 0.12 on the viridis scale, wall 0): ≈ 1, 0.72, 0.34, 0.2 of the peak above background. Σ = I in cell units gives 1, 0.61, 0.14, 0.01 — the drawn map falls off more slowly (σ ≈ 1.4 cells fits better). Colour → value is our reading, ± ~0.1.
- Also in all four maps: one dark (= wall-valued) cell at walkable (1, 1) from the top-left — an obstacle or an agent marker; the text says neither. Ours has no such cell.
- Not yet a fix: the re-registered rule must pick μ ON the walkable corner (the figure's reading) and say whether σ stays 1 (the text) or not. Cheapest test first: one dqn@5000 seed with μ moved, as a pilot (stated so), before a 60-run kernel.
