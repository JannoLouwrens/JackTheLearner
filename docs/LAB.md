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

- 11-12 · **Curriculum — weaning** (`experiments/wean_pilot.py`: the mother bears
  him within r of ready food, r ×1.5 per fed life, then leaves; control = decoy
  points) · **DEAD at first dose (s90, 13k): weaned-window meals nursed 0, null 0,
  decoy 0** — but killed in a world where he could not sense food and one food
  sat in a drive trap (both below). **RE-ENTERED 11-12 23:30 with a new reason —
  see the last entry.**

- 11-12 · **World/body findings the weaning kill uncovered** (J1 + LAB; modules
  and full numbers in git, long form at `4b4c2e3`), one line each:
  - He could not perceive floor food (rays pass over it; 0 of 584 poses). *Eye
    height* (`food_sense.py`) DEAD: meals split 1/2/3 vs blind 2/2/3.
  - *Smell* (`smell_sense.py`, points to food on 99.5 % of poses) DEAD three
    times: 13k 8 vs shuffled 10; 52k repaired world **2/0/3 vs shuffled 20/0/3**.
    A sense that works goes unused by the reactive learner.
  - *Body* (`body_probe.py`): pole topples ~19/20 births; plinth stands. Not the cap.
  - *The pool was a DRIVE TRAP*; fixed (`W0.PL` PASS `3f15c7f`). Perfect-map
    oracle fed per birth 23/40 → **38/40**. Body and world are not the gap; the
    learner is.
  - *Credit horizon:* GAMMA .99 race DEAD 11-13 (for on 1 of 3 seeds). 88 of 120
    meals sit in 5 "jackpot" lives; approach from afar does not carry to the
    next birth. For `LF.04`: a meal-count margin is heavy-tailed noise.
  Philosophy 3, 5.

- 11-12/13 · **World: shelter law** (J6, `NE.01`) · ALIVE, not shipped; roof hides
  the cold sky: sealed → 0.110 on s0. **Occlusion-law race 11-13
  (`occ_law_race.py`): NO WINNER** — shipped 4/5, graded cone 2/5, damped
  sleeper 2/5 (dies of injury), open control 0/5 (as it must). Band-geometry
  arm is next. Rows /data/tmp/occrace/.

- 11-12 · **Exploration in bouts** (white vs OU vs pink action noise, noise-only,
  `explore_bout.py`, rule first `141e84f`) · **DEAD at stage A:** fed births
  white 4/40, ou 3/40, pink 6/40 (lead 2 < 4). Pink roams 1.7× wider (reach
  6.3 m vs 2.9) and finds food no more often: range is not what starves him,
  direction is. Stage B (bout noise inside the learner) never run. Philosophy 3, 4.

- 11-12/13 · **Curriculum — weaning re-raced WITH smell on the repaired world**
  (`wean_smell_race.sh`, rule first) · **DEAD 11-13 03:35** (null clause 1 of 3):
  weaned meals nursed 1/9/1, decoy 0/0/·, null 2/0/3; s92's 9 are one jackpot life
  born 0.70 m. Nursing feeds him (nurse meals 18/5/6 vs decoy 0) but r never holds:
  up to 0.9–1.69 m, back to 0.5, vs uniform births median 3.1–3.9 m. He learns
  eating-where-born, not walking up a gradient. Staged widening (`CR.00`, J5) is
  the remaining world-side arm. Philosophy 4, 5.

- 11-13 · **Meal memory — what he replays** (Consolidation; `meal_memory.py`,
  rule first `dabd2bf`) · **DEAD at the claim gate (11-15, s94–98, 13k, Kaggle
  CPU).** Pooled fed: meal .450 vs fifo .277 vs yoked .352; seed wins 5/5, 4/5;
  far (born ≥ 0.7 m) .335 vs .230/.332. Margin vs yoked **.0985 < .10**, the only red
  clause. Stays dead as declared. Pilot below. A second ring
  keeps every segment with a meal; batches 8 FIFO + 8 meal. Nursery trials
  s91–93: fed **.484** vs fifo .370 vs yoked control .376; wins 2/3 and 3/3,
  margins .114/.108 vs a .10 bar. Mostly birth luck (meals land in 5–16
  decisions): a lean, not approach from afar. Claim gate in the module
  docstring (s94–98, 13k, born ≥ 0.7 m clause). END-FIT 4, 2. Philosophy 3, 4.

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
