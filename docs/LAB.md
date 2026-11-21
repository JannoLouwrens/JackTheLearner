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
  `ear_predict.py`; 70k weights, 5.5 h unlabelled speech, 25 hearings a word) ·
  **DEAD:** +1.8 / +2.3 points (bar +5.0; shuffled −0.5 / +0.3); CPC +0.5.
  Published gains start at 100 h. Unrun: the narrator's label as the loss.

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
  DEFAULT; `taste_eye.py`, rule `a84e08d7`; long form `3c17e958`) · **DEAD:**
  a ring over 128 rays trained only on "I eat within 2 s" points at in-sight
  food on **.080** of 33,402 judge looks vs untrained .082, shuffled .024,
  nearest .032; handed the food's ray (told) .829 — so the miss is the LABEL.
  Found: food wears one of 4 colour sets by world; only .43 is always food.

- 11-21 · **One parent's voice, heard as bigger and smaller people** (ASR,
  VACANT; `ear_one_voice.py`, rule `8cc61917`; long form `3c17e958`) ·
  **DEAD:** at 5 hearings a word, one voice .257 on strangers, pitch-warped
  **.279 (+2.2 points, bar +5.0, ahead 8/12)**, matched noise .260, 50 voices
  .287. Which parent matters more than how many (spread .055 vs .030).
  PRIORS §23 since: the record says voices, not hearings. Unrun: VCTK.

- 11-21 · **The mouth names it, the eye remembers backwards** (Vision
  encoder, BY DEFAULT; PLAN "Learning to eat" right column; `hindsight_eye.py`,
  rules first `3a8495b6`, `4463800b`). Bet: taste_eye died on its label, so
  make the label at the meal — imprint the texture of the NEAREST ray when he
  eats, walk back 8 s marking that texture. Kill: taste_eye's bars (hit
  < .50, or < .25 over a yoked twin imprinted at random moments / untrained
  / nearest). Prior: hindsight relabelling and track-back labels are
  published at 10^4–10^6 clips (Andrychowicz 2017, Pathak 2017, from
  memory); at ~200 meals: silent. **ALIVE, ENTERED (INBOX proposal).**
  215 meals, 20 judge worlds, 33,402 looks: as a plain MEMORY (a count of
  eaten colours, no net) **.890** — above the simulator-taught net's .829 —
  yoked .000, ahead 20/20; as a ring NET on the same marks **.434, DEAD**
  (yoked .201, untrained .082). The imprint is 74% on food. By colour set
  .97 / .99 / .74 / .61. **Stage 2** (reflex along the peak ray, carried
  table, worlds 1301–1305, 2,400 decisions): **50 meals vs held8 2, yoked
  0 — ALIVE-2**; read it as "reaches a food by ~18 s and camps on it": 10
  meals a world is the 50 s regrow clock, 5/5 worlds. Disclosed: judge worlds
  were read once by taste_eye; told not refitted. Cannot say: a learner uses
  it; the table filled from empty as he lives; a camera eye (W0's colour is
  one clean number per object — there the net form is needed, and it is
  the dead one). Unrun: the net with the table's output as its teacher.
  END-FIT 1, 2, 5. Philosophy 1, 2, 3, 4.
