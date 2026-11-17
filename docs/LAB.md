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

- 11-16 · **Eye — a streak, not a grid** (Vision encoder, BY DEFAULT; PLAN
  "Learning to eat" right column and §12's open question; `eye_streak.py`,
  rule first `bddffac4`, dial rule `e583c41e`). Bet: his food is a 0.2 m thing
  on the floor, so 16 azimuths miss it whatever rows are stacked above; the
  same 256 rays as 128 azimuths × 2 rows find it (ground foragers have a
  horizontal streak). Kill: no lead of +.05 over the 16×16 grid and +.10 over
  today's 16 rays on ≥ 16 of 20 worlds. Prior: Habitat depth SPL .79 vs RGB
  .46; "What if Eye" — detection needs resolution; nothing published on ray
  geometry for a floor object at a 0.12 m eye. Ruler: at the entry rung, a
  per-world table material → food (≤ 81 numbers, adult's label) points along
  the nearest food-coloured ray; HIT = within 20° of a food ≤ 4 m away.
  - **ALIVE (worlds 400–419): streak .900 vs grid16 .517 vs seated16 .528**
    (ahead 20/20, 20/20); grid32 (1,024 rays) .758; blind twin .19, shuffled
    table .21 — both fail as they must. **§12's question answered: the 16×16
    grid does NOT beat today's 16 rays** (−.012, ahead 7/20).
  - The second row buys nothing: one level row of 256 scores .907 (streak
    ahead 6/20). Dial (worlds 420–429): row 16 / 32 / 64 / 128 / 256 = .540 /
    .696 / .856 / .909 / .925 — **knee at 128**. Food 4–8 m away: today's eye
    sees it on 14% of frames, row128 on 74%.
  **Reading:** today he is blind to food beyond ~1 m about half the time, and
  the retina planned in "Next up" 3 would not cure it; azimuth in the floor
  band does, for 256 numbers and ~0.1 ms. What this cannot say: whether a
  BRAIN learns to use it (the table is an adult-taught ceiling). ENTERED as
  an INBOX line: build `row128` as a `jack_env` option, then race it under a
  learner. Not tried: a fovea (dense ahead, sparse behind) at 64 rays.
  END-FIT 1, 5. Philosophy 1, 3, 4.

- 11-17 · **Finding himself after rebirth — one look, with and without the
  compass** (Death & persistence, VACANT; Vision encoder; PLAN "Memory and
  sleep" right column; `find_himself.py`, rule first `fe459803`). Bet: today's
  eye casts rays at WORLD azimuths, a free compass the human-like body will
  not have; without it 16 rays cannot re-find his place and the 128 row can
  (best of all turns of the look against 2,880 remembered looks; no training,
  raw looks as keys). Kill: body-frame 128 recalls a look from within 1 m on
  < 50% of rebirths, or leads 16 rays by < .10 / on < 16 of 20 worlds. Prior:
  nothing published for range-only senses in a from-scratch agent (PRIORS §6,
  L4 "expect poor in symmetric arenas"); successes had vision or GPS+compass.
  - **ALIVE (worlds 700–719, chance .021): no compass, 128 rays .933 vs 16
    rays .795** (ahead 20/20, lead +.138); with the compass .957 / .889.
    Shuffled place tags .02; scrambled eye .06–.12 (16 rays no compass
    .1208 against a .1211 VOID line — a hair). Median miss 0.3 m.
  - **Half the bet was wrong:** 16 rays DO find him without a compass (.795,
    20/20 worlds). PLAN's question "with 16 rays" is answered yes.
  - The compass costs .09 at 16 rays, .02 at 128 — IF he may turn the look.
    Compared ray-for-ray with no turning, 128 rays fall to .54.
  - With the compass he mostly reads the ROOM, not his world: recalling from
    ANOTHER world's memory still hits .79 / .88. Without it a stranger's
    memory gives .40 / .47 — the square room is ambiguous, and his own
    world's furniture is what settles it (the record's L4, measured).
  - It is coverage, not sight, that limits him: 15 / 60 / 360 remembered
    lives = .31 / .65 / .93.
  - Follow-on, rule first `2a6ffa69`, fresh worlds 720–724: a key that needs
    NO turning (the row's amplitude spectrum, 130 numbers) · **VOID as
    declared** — hit .845 (5/5 worlds, .087 behind the turn search .932, 16
    rays .608), but the scrambled-eye control read .157 > .121. Diagnosis
    after the fact, deciding nothing: the mean distance and material (2
    numbers, untouched by scrambling) alone recall .31; with them dropped
    the control falls to .03 and the key still hits .795. Re-register with
    that control before believing it.
  **Reading:** one look is enough to find himself in this 12 m room on either
  eye, and the head-turning eye loses little — provided his memory compares
  looks turn-free. What this cannot say: whether a BRAIN's learned key does
  it; a bigger or emptier world. ENTERED as an INBOX line (a body-frame eye
  option; results earned on the compass eye to be earned again without it).
  END-FIT 1, 2, 4. Philosophy 1, 3, 4.
