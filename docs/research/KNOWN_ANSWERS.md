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
| 2b | Pendulum-v1 | re-registration, 15 fresh seeds, `r2lib` (rule `c442a757`) | 102,400 | same band, unchanged | RUNNING jack-cpu-1794808095 | 5–19 | — |
| 3 | MountainCarContinuous-v0 | SB3 SAC gSDE Zoo entry unmodified (`r3lib`; rule `824b0a2a`) | 50,000 | deterministic eval > 90 on ≥ 3 of 5 (public 9/10 in 93.4–95.3) | 94.8 94.5 95.5 95.0 93.5 — 5 of 5 | 0–4 | **REPRODUCES** |
| 3 | MountainCarContinuous-v0 | control `r3ctrl` (no gradient step), must fail | 50,000 | VOID if > 90 on any seed | −0.3 −2.8 −3.5 −0.9 −0.8 — 0 of 5 | 0–4 | holds |
| 3b | MountainCarContinuous-v0 | Brain B `r3ours`: SBX SAC + LayerNorm, Zoo numbers (rule `b09adb6a`); ctrl `r3octrl` | 50,000 | same bar | 96.0 96.0 94.9 95.2 95.5 — 5 of 5 (Kaggle jack-cpu-1794805793) | 0–4 | **REPRODUCES** |
| 3b | MountainCarContinuous-v0 | control `r3octrl` (SBX, no gradient step), must fail | 50,000 | VOID if > 90 on any seed | −0.3 −0.1 −20.6 −0.1 −0.0 — 0 of 5 | 0–4 | holds |
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
