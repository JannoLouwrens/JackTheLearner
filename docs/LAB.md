# LAB — the inventor's notebook (≤ 10 KB)

One entry per idea: date · seat · the bet · the number · ALIVE / DEAD / ENTERED.
A dead idea with a number is a result. Do not retry one without a new reason.
Folded 2026-11-12 23:40 (was 20 KB): full text of every entry is in git
(`git log -p docs/LAB.md`, last long form at `0275671`).

## Already known from the ladder (so nobody re-invents them)

- 08-23 · learning core · five cores screened in W0 · only `wm-latent` learned to
  survive (4.65σ vs null); `wm-efe` 2.05σ, `ppo-lp` 1.20σ, `ppo-needs` 1.06σ,
  `dreamer-xs` −0.94σ.
- 09-07 · control path · small plastic head 506, MLP 415, 57M-trunk arms 344–350,
  SB3 518 · VOID on the wrong null; re-run as `D1.1`.
- 08-24 · deliberation · lookahead gains 13 s ± 19 in W0 — too shallow a world.
- 08-13 · mood → movement · 0.23–0.38 vs chance 0.25 · DEAD as built.

## Ideas

- 11-11 · **Working memory — ring (rotating) store** · **SEATED, `WM.02` PASS**
  (`2103829`, seeds 100–104): ring 0.902 ± 0.196 vs GRU 0.579, null 0.347; shuffled
  0.127 and no-rotation −0.047 both failed as required. 1,031 params vs 4,872.
  Dead variants: angle init ×4 (0.106), no write gate (0.119). Open: a
  non-commutative register; ME.8 / LF.02 with the ring in place. Philosophy 3, 4.

- 11-12/13 · **Curriculum — weaning** (`wean_pilot.py`, `wean_smell_race.sh`;
  long form `837b4c6f`) · DEAD twice: blind 0/0/0; with smell on the repaired
  world weaned meals 1/9/1 vs null 2/0/3. Nursing feeds him (18/5/6 vs decoy 0)
  but r never holds: he learns eating-where-born, not walking up a gradient.

- 11-12 · **World/body findings the weaning kill uncovered** (long form `4b4c2e3`):
  rays pass over floor food (0 of 584 poses); *eye height* DEAD (1/2/3 vs
  2/2/3); *smell* DEAD three times (52k: 2/0/3 vs shuffled 20/0/3) — PRIORS §14
  says that setup cannot learn; pole body topples ~19/20; the pool was a DRIVE
  TRAP, fixed (`W0.PL`; oracle 23/40 → 38/40); *GAMMA .99* DEAD (1 of 3 seeds).
  88 of 120 meals sit in 5 "jackpot" lives: a meal-count margin is noise.

- 11-12/13 · **World: shelter law** (J6, `NE.01`) · occlusion-law race NO
  WINNER (shipped 4/5, cone 2/5, sleeper 2/5, open 0/5); parked (ruling 35).

- 11-12 · **Exploration in bouts** (white / OU / pink, `explore_bout.py`) ·
  DEAD: fed births 4 / 3 / 6 of 40. Pink roams 1.7× wider and feeds no more:
  direction, not range, starves him.

- 11-13 · **Meal memory — what he replays** (Consolidation; `meal_memory.py`) ·
  DEAD at the claim gate (s94–98): fed meal .450 vs fifo .277 vs yoked .352;
  margin vs yoked .0985 < .10, the only red clause. Stays dead as declared.

- 11-13/14 · **Predict-then-push, three arms** (Deliberation, VACANT; long form
  `1a18168`) · all DEAD on fed births (babble 5/40): sense lookahead 6/40,
  ego-frame drive .017 vs bar .10, gain-field model 4/40 (want 9, shuffled 5).
  He cannot yet predict what a push does to his nose. Philosophy 3, 4.

- 11-14/15 · **Run and tumble — climb without predicting** (Deliberation;
  `run_tumble.py`, long form `837b4c6f`) · *told* DEAD 8/40 vs babble 6;
  *steer* **ALIVE with the want GIVEN: 14/40 vs babble 3, coin-yoked 2** (first
  arm of nine to beat babble; oracle 38/40); *steerwant* (want learned from 14
  meals) DEAD 7/40 vs shuffled 7. The gap is the WANT, not direction. Unrun: a
  want from the CHANGE in each sense before a meal; the critic's value as the
  want. Not entered (a given want is advice, not his). Philosophy 1, 3, 4.

- 11-15 · **Ear — taught by his own prediction of sound** (ASR, VACANT;
  `ear_predict.py`, long form `837b4c6f`; 70k weights, 5.5 h of unlabelled
  Speech Commands, then 25 hearings per word) · **DEAD:** next-tick MSE +1.8
  points at 200 ms, +2.3 at 40 ms (bar +5.0; shuffled twin −0.5 / +0.3); CPC
  +0.5. He predicts the next tick and it buys his words nothing; published
  gains start at 100 h. Unrun: listening hours as the dial; the narrator's
  label as the loss. Data `/data/jack_corpora/speech_commands_v2`. Philosophy 2, 3, 4.

- 11-16 · **Eye — a streak, not a grid** (Vision encoder; `eye_streak.py`, rule
  first `bddffac4`; long form `78902da4`) · **ALIVE, ENTERED (now order J12):**
  on a taught table's ruler (worlds 400–419) 128 azimuths × 2 rows .900 vs
  16×16 grid .517 vs today's 16 rays .528 (ahead 20/20); blind .19, shuffled
  .21 fail as they must. The 16×16 grid does NOT beat 16 rays (§12 answered).
  One level row: 16/32/64/128/256 = .540/.696/.856/.909/.925 — knee at 128.
  Cannot say whether a BRAIN learns to use it. Not tried: a fovea at 64 rays.
  END-FIT 1, 5. Philosophy 1, 3, 4.

- 11-17 · **Finding himself after rebirth — one look, with and without the
  compass** (Death & persistence, VACANT; `find_himself.py`, rule first
  `fe459803`; long form `78902da4`) · **ALIVE, ENTERED (became ruling 38 /
  J12):** recall of a look from within 1 m (worlds 700–719, chance .021): no
  compass 128 rays .933 vs 16 rays .795 (ahead 20/20); with compass .957 /
  .889; shuffled tags .02. Half the bet was wrong — 16 rays DO find him. The
  compass costs .09 / .02 only if he may turn the look; ray-for-ray 128 falls
  to .54. Coverage limits him: 15/60/360 remembered lives = .31/.65/.93.
  Turn-free key (amplitude spectrum, `2a6ffa69`): hit .845 but **VOID** — the
  scrambled-eye control read .157 > .121; re-register with mean distance and
  material dropped from the control before believing it.
  END-FIT 1, 2, 4. Philosophy 1, 3, 4.

- 11-18 · **A place key he learns from the passing of time** (Death &
  persistence, VACANT; the diary's key; PLAN "Memory and sleep" right column;
  `place_key.py`). Bet: yesterday's body-frame eye re-finds his place only by
  trying all 128 turns of a look against every memory (.933; compared once
  .54). A small ring-shaped network trained from scratch only to give looks a
  moment apart in one life close keys — no place tags, no compass — learns a
  key compared ONCE that loses ≤ .10 to the turn search. Why it might: he
  turns faster than he travels, so time itself teaches "same place, other
  heading" (the slowness account of place cells). Kill: the key fails to find
  him (< .50), or leads its untrained and time-shuffled twins by < .10 / on
  < 16 of 20 worlds, or trails the turn search by > .10. Prior: Franzius &
  Wiskott 2007 (slow features of a simulated rat's view are place cells);
  learned recall keys on record needed 10^6–10^9 frames of pixels (PRIORS
  §6); ours has 2,880 looks of 256 numbers per world — the record is silent
  at that size. Rule frozen in the docstring before worlds 730–749 exist;
  tuned on 690–692 only.
