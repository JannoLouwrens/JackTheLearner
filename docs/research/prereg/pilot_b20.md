# pilot_b20 — twenty lives-long runs of brain B on his own eye, 1M decisions (J32)

**STATUS: DRAFT 2026-11-28 UTC (JACK, ORDERS J32 (1)) — NOT frozen; nothing run. Every term below is ORDERS J32 / ruling 141's; the worker chose none of them. Freeze under ruling 135 (INBOX line, 3 h) only after the build owed below exists.**

This is a **sizing pilot** (ruling 141). It seats nothing; no arm "wins". It
answers the two things PLAN stage 2 needs before any long race: the spread of
"eats from his eye" over 20 seeds, and whether it still holds at 1M decisions.

## Question
J30 (`eye_bins16_k1.md`, read 11-27 14:30, EATS): brain B with K1's resets on
the kind eye ate 27.41 late meals a life (lives starting ≥ 250k of 500k) against
its blind 3.34 and shuffled 2.36 twins, 3 of 3 seeds, still climbing at 450k.
Three seeds give no spread, and 500k gives no answer about a longer life.
**This pilot asks: over 20 fresh seeds, how many does brain B+K1 still eat from
his eye at 1M decisions — and so how many runs must a stage-2 race hold?**

## Prior — why it could lose either way
- For: J30's three seeds all climbed through every 50k window to 24–39 meals a
  life by 450k (WORLD_CARD § J30); resets keep a replay learner learning
  (Nikishin 2022 2205.07802; Nauman 2024 2403.00514: "periodical network
  resetting is the most robust intervention", PUBLISHED_PRIORS §21);
  foragers in the record eat at 1e6–3e6 steps (§28), so 1M is inside the range
  where eating is published, not past it.
- Against: no small agent is published still improving at the end of a long
  life (§20) — a 1M life has twice the resets and twice the replay age of
  J30's, and J24 showed plain SAC unlearning to ~0 after an early peak; three
  seeds out of three can hide a 20–30 % failure share (§8: few runs give little).
  J30's blind twin also climbed (→ 2.6–8.0 meals a life), so "twice the blind
  mean" is a moving bar, not a fixed one.

## Arm — J30's sighted cell exactly, plus one world option
- Learner: J30's `K1-bins16` (`pilot_b2.J24K1_ARMS`): sbx 0.28.0 SAC, MlpPolicy
  LN 2×256, lr 3e-4, batch 256, tau 0.005, γ 0.995, warm-up 5,000,
  train_freq 32 / gradient_steps 32, gSDE, `ent_coef="auto"`, buffer = T;
  **K1: full network resets every 50,000 decisions, replay and temperature
  kept** — at 1M that is 19 resets (50k…950k), the same interval, not retuned.
- World: `PILOT_RULE` (`food_floor_only=True`, `eye_frame="body"`,
  `motor_frame="body"`, `native_substep=True`), `paint="kind"`, `eye="bins16"`,
  entry rung d0, `lazy_contact_audio=True` as J30 ran — **plus
  `terminal_obs="dying"` (ruling 138)**, the one difference from J30's cell.
- Twin: `K1-bins16-blind` (every bin zero, J17's `BlindEye`), same learner and
  world, on seeds 2001–2005.
- No shuffled twin (the order names only the blind twin).

## Seeds, size, runs
- Sighted: seeds **2001–2020** (20 runs). Blind twin: seeds **2001–2005** (5).
- **1,000,000 decisions** a run. 25 runs in all.
- One run = one observation, never its lives (§8, the last 10 %).
- A run not home whole is re-run ONCE from its last whole carry (J32 branch: a
  kernel dies → relaunch that chain once); a seed still lost after that is
  entered LOST and counts as NOT ending above the bar. No seed is replaced.

## Outputs (all read once, when the 25 runs are home)
1. **Score per run:** late meals a life, lives starting ≥ 500,000 (the late
   half, ruling 65), `pilot_b2.j17_late_meals(rec, 500_000)`.
2. **Window table:** meals a life per 50k window, every run (J24/J30's table).
3. **Blind-twin mean** B̄ = mean of the five blind runs' late meals a life.
4. **Share** p̂ = (sighted seeds with late meals > 2 × B̄) / 20, with its exact
   (Clopper–Pearson) 95 % interval (`binomtest(k, 20).proportion_ci("exact")`).
5. **The n a race needs (§8):** per-arm runs for 80 % power, two-sided exact
   Boschloo at 0.05, for a race whose better arm succeeds at p̂ and whose worse
   arm succeeds at p̂ − 0.3, p̂ − 0.4 and p̂ − 0.5 (each clipped at 0), by
   exact enumeration — a table, deciding nothing here; the race's own
   pre-registration chooses the gap. Beside it, the sighted runs' late-meal
   spread (mean, SD, min, max) for a continuous outcome.
6. **The one falsifiable line:** **"B+K1 still eats at 1M" HOLDS iff ≥ 15 of 20
   sighted seeds end above 2 × B̄.** Fewer is written **NOT HELD AT 1M**, and
   stage 2's brain B is in question (ruling 141).

Beside, deciding nothing: critic rulers (ruling 102), per-reset dips, touch
columns, onset (`j17_onset`), the blind twins' own window table.

## Compute (J32 (2) — J30's cells are the timing cell, same learner shape)
J30 ran 3 cells a kernel in parallel at 500k: kernels home in 21,711 /
22,595 / 24,129 s (6.0–6.7 h). The learner shape is unchanged; 1M is twice the
decisions and the replay holds twice as much, so ≈ 12–13.5 h a run, in **two
segments of ≤ 7.5 h** cut into the carry (J30's chains already carry
bit-equal across segments, `tests/test_j24_chain.py`). Packing: 3 cells a
kernel → 9 kernels (8 × 3 + 1 × 1), ≈ 18 segments, ≈ 110–125 kernel-h.
On the 2 sessions free today (J18 holds 3) ≈ 2.5 days; on 5 after J18's
look 1 (≈ 11-29 08:00) ≈ 1.2 days. No GPU. Kaggle CPU only (ruling 85,
rank 2, behind J18).

## Owed before the freeze (build, not judge)
- ~~Arms~~ BUILT 11-28 ~10:30: `pilot_b2.PILOT_B20_ARMS` (`B20-bins16`,
  `B20-bins16-blind`) = J30's K1 dicts with resets 50k…950k and
  `terminal_obs="dying"`; `make_world(terminal_obs=)` / `pilot_world` hand it
  to `JackEnv` (None = the old call, no key); records carry `terminal_obs`.
  J24K1_ARMS unchanged. `tests/test_pilot_b20_arms.py` 4/4 under jack-std; a
  mutant that drops the pass-through is caught.
- ~~The death-step check~~ RUN 11-28 10:10 on the box (jack-std, seed 2001,
  6,000 decisions, `/data/jack-logs/pilot_b20/deathcheck/`): K1-bins16
  (`reborn`) vs B20-bins16 (`dying`) — one death (energy) at 2,868 in both;
  every life row and the 0–5k action-trace row IDENTICAL; ~1,000 decisions of
  training after warm-up (5k) with the death transition in replay did not
  change the second life. So on this cell `dying` changes nothing past the
  death step (SAC does not bootstrap a terminal step). Short: one death, one
  trace row — a pilot check, not a proof. The arm stays `dying` either way.
  Box speed beside: 67 decisions per CPU-s.
- ~~Launcher~~ BUILT 11-28 11:05 (`d1c326c9`): `j24_launch --variant b20`,
  units 1–9 = the 25 cells in order (sighted 2001–2020, then blind 2001–2005)
  three a kernel. Was: a `j24_launch` variant (own folder `/data/jack-logs/pilot_b20/`,
  own freeze guard on THIS file, seeds 2001–2020, 1M, mixed sighted/blind
  packing) with a planted test that it refuses this file while DRAFT.
- Reader: `pilot_b20_read` computing outputs 1–6 once, refusing a pin where this
  file is not FROZEN or a raw file that does not carry the pin; planted tests
  for HOLDS / NOT HELD / a lost seed.

## Settings that differ from the published foragers
One environment, not many; our body and world; 1M decisions, the low end of the
published 1–3e6; resets every 50k decisions (Nikishin resets every 2e5 steps on
DMC, §19 — ours is the forge's interval, carried unchanged); 20 seeds, which is
the pilot size §8 asks for, not a race.
