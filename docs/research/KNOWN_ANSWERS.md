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
| 2 | Pendulum-v1 | PPO, Zoo entry | 102,400 | −187.6 ± 17.0; accept −213.2…−162.0 | — | — | — |
| 3 | MountainCarContinuous-v0 | SB3 SAC gSDE Zoo entry unmodified (`r3lib`; rule in `brain_b.py`, `824b0a2a`); SBX/pink variant after | 50,000 | deterministic eval > 90 on ≥ 3 of 5 | running (Kaggle jack-cpu-1794803164) | 0–4 | — |
| 4 | MountainCarContinuous-v0 | PPO, Zoo entry | 20,000 | 87.5 ± 4.2 | — | — | — |

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
