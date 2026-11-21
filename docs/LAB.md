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

- 11-18/20 · **A place key he learns from the passing of time, carried across
  death** (Death & persistence, VACANT; `place_key.py` rule `de328d98`,
  `place_key_carried.py` rule `f1841fc1`; long form `8cc61917`) · **ALIVE
  twice, ENTERED as an INBOX proposal (ruling 63: stage-3 candidate):** a
  10,000-weight ring trained only to give looks a moment apart close keys,
  compared ONCE: **.934 vs the 128-turn search .933** (worlds 730–749;
  untrained .443, time-shuffled .273, tags .026, chance .021; a plain MLP
  .447 — the ring anatomy is needed). Trained once on worlds 720–724 and
  carried to unseen worlds at 15 / 60 / 360 remembered lives: **.294 / .680 /
  .945** vs turn search .266 / .628 / .933 vs home-grown .202 / .542 / .934.
  What survives death is the KEY (carried) and the LOOKS (per world;
  coverage is the limit). Disclosed: judged worlds not fresh for the second
  bet. Cannot say: inside a brain learning to eat; another kind of world;
  online under a real policy. Not tried: the key as the brain's place input.
  END-FIT 2, 4. Philosophy 1, 2, 3, 4.

- 11-20/21 · **Taste teaches the eye where food is** (Vision encoder, BY
  DEFAULT; PLAN "Learning to eat" right column; `taste_eye.py`, rule
  `a84e08d7`, judge repaired before data `fe77023c`). Bet: E2c showed a
  learner handed the direction to food eats — so grow the direction in him.
  A ring over his 128 rays, trained only on "I eat within 2 s" from random
  pushes, points at food with its peak ray. Kill: on 20 fresh worlds the
  peak hits in-sight food < .50, or < .25 over shuffled / untrained /
  "nearest thing". Prior: weak labels localise at ImageNet scale (Oquab
  2015; Zhou 2016, from memory); nothing at ~200 meals. **READ 11-21 21:47
  (JACK slot, rule unmoved, `/data/jack-logs/taste_eye/read.json`): DEAD,
  not VOID.** Peak-ray hit on in-sight food, 20 judge worlds, 33,402 looks,
  3 inits: taste128 **.080** vs untrained .082, shuffled .024, nearest
  .032, taste16 .018; told (handed the target) .829 ≥ .80 → the net and
  data CAN localise, so the miss is the label. Clauses: only "ahead of
  shuffled" holds (19/20); ahead of untrained 10/20. 215 meals, 2,123
  positive looks of 125,840. Untrained .082 ≫ random .020 is an init bias,
  not sight. `taste_eye reflex` is NOT run (stage 2 needed ALIVE).
  END-FIT 1, 5. Philosophy 1, 2, 3, 4.

- 11-21 · **One parent's voice, heard as bigger and smaller people** (ASR,
  VACANT; PLAN "Ear" right column "one voice that carries over to other
  voices"; `ear_one_voice.py`, rule first `8cc61917`). Bet: at 5 hearings a
  word, an ear raised on ONE speaker that warps each hearing's pitch axis
  (0.8–1.25, a longer or shorter vocal tract) understands strangers better
  than on the plain hearings. Kill: under +5.0 points over 12 parents, or
  ahead on < 10, or matched noise does half as well. Prior: the record's
  ears had ~2,600 speakers; the warp is published only with many speakers
  already present (Jaitly & Hinton 2013, from memory); one voice: silent.
  **DEAD** (12 parents, 4,074 strangers' recordings, chance .10): one voice
  **.257**, warped **.279 (+2.2 points, ahead 8/12)**, matched noise .260
  (+0.3, fails as it must), 50 voices .287. Reported: at 5 hearings one
  voice is only 3.0 points behind fifty — but twice as chancy (spread .055
  vs .030: which parent matters more than how many). The warp is real and
  small; it closes .74 of a gap that is itself small. Cannot say: anything
  at the hundreds of hearings a real parent gives (one Speech Commands
  speaker has 5). Unrun: the same question on VCTK (on disk, 400 sentences
  a speaker) where one voice can be heard 100+ times.
  END-FIT 2, 3. Philosophy 2, 3, 4, 7.
